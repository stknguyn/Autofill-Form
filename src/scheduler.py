"""
src/scheduler.py - Jitter Engine, Pacing, and Background Automation Scheduler.

Milestone M3: Worker Fleet Manager & Background Automation.
Implements:
- JitterDistribution enum & JitterConfig
- JitterEngine: Uniform & Truncated Gaussian with Rejection Sampling,
  Human Fatigue Micro-breaks, Interruptible Sleep with threading.Event
- SchedulerState enum & SchedulerConfig & TimeWindow (with overnight wrap)
- SchedulerSnapshot immutable dataclass for UI telemetry
- BackgroundScheduler with start delay countdown, diurnal windows, batch quota & cooldown,
  and full clock/time injection for deterministic testing.
"""

from __future__ import annotations

import datetime as dt
import enum
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("scheduler")


class JitterDistribution(str, enum.Enum):
    """Supported pacing distribution types."""
    UNIFORM = "uniform"
    TRUNCATED_GAUSSIAN = "gaussian"
    GAUSSIAN = "gaussian"


# Backward-compatible alias
DistributionType = JitterDistribution


class SchedulerState(str, enum.Enum):
    """Operational states of the background scheduler."""
    IDLE = "IDLE"
    COUNTDOWN = "COUNTDOWN"
    ACTIVE = "ACTIVE"
    WINDOW_WAIT = "WINDOW_WAIT"
    COOLDOWN = "COOLDOWN"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class PacingDecision:
    """Immutable outcome of a pacing evaluation."""
    delay: float
    is_micro_break: bool
    duration_break: float = 0.0
    submission_count: int = 0
    submissions_to_next_break: int = 0

    def get_status_label(self) -> str:
        if self.is_micro_break:
            return f"PACING (Micro-break: {self.delay:.1f}s)"
        return f"PACING ({self.delay:.1f}s)"


@dataclass(frozen=True)
class TimeWindow:
    """
    Represents a diurnal active time-of-day window [start_time, end_time).
    Supports standard daytime windows and overnight windows (e.g. 22:00-04:00).
    """
    start_time: dt.time
    end_time: dt.time

    @classmethod
    def from_string(cls, window_str: str) -> TimeWindow:
        """Parse 'HH:MM-HH:MM' or 'HH:MM:SS-HH:MM:SS' string into TimeWindow."""
        pattern = r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*-\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$"
        match = re.match(pattern, window_str.strip())
        if not match:
            raise ValueError(f"Invalid time window format: '{window_str}'. Expected 'HH:MM-HH:MM'.")

        h1, m1, s1, h2, m2, s2 = match.groups()
        h1_i, m1_i, s1_i = int(h1), int(m1), int(s1) if s1 else 0
        h2_i, m2_i, s2_i = int(h2), int(m2), int(s2) if s2 else 0

        if not (0 <= h1_i <= 23 and 0 <= m1_i <= 59 and 0 <= s1_i <= 59):
            raise ValueError(f"Start time out of range in window: '{window_str}'")
        if not (0 <= h2_i <= 23 and 0 <= m2_i <= 59 and 0 <= s2_i <= 59):
            raise ValueError(f"End time out of range in window: '{window_str}'")

        t1 = dt.time(h1_i, m1_i, s1_i)
        t2 = dt.time(h2_i, m2_i, s2_i)
        return cls(start_time=t1, end_time=t2)

    def contains(self, t: dt.time) -> bool:
        """Check if time t falls inside this active window."""
        if self.start_time <= self.end_time:
            # Daytime window: e.g. 08:00 to 12:00
            return self.start_time <= t < self.end_time
        # Overnight window: e.g. 22:00 to 04:00
        return t >= self.start_time or t < self.end_time


@dataclass(frozen=True)
class SchedulerSnapshot:
    """Thread-safe, immutable snapshot of the background scheduler for UI display."""
    state: SchedulerState
    is_active: bool
    start_countdown_remaining: float
    window_wait_remaining: float
    cooldown_remaining: float
    current_batch_count: int
    total_batches_completed: int
    active_windows_str: str
    message: str


@dataclass
class JitterConfig:
    """
    Configuration parameters for the Jitter Engine.
    Flexible initializer supporting aliases (min_delay/t_min, max_delay/t_max, mean_delay/mu, std_dev/sigma).
    Defaults to 30 minutes to 1 hour (1800s to 3600s).
    """
    distribution: JitterDistribution = JitterDistribution.TRUNCATED_GAUSSIAN
    min_delay: float = 1800.0
    max_delay: float = 3600.0
    mean_delay: Optional[float] = None
    std_dev: Optional[float] = None
    micro_break_enabled: bool = True
    micro_break_min_interval: int = 15
    micro_break_max_interval: int = 25
    micro_break_min_duration: float = 10.0
    micro_break_max_duration: float = 30.0

    def __init__(
        self,
        distribution: Union[JitterDistribution, str] = JitterDistribution.TRUNCATED_GAUSSIAN,
        min_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        mean_delay: Optional[float] = None,
        std_dev: Optional[float] = None,
        micro_break_enabled: bool = True,
        micro_break_min_interval: Optional[int] = None,
        micro_break_max_interval: Optional[int] = None,
        micro_break_min_duration: Optional[float] = None,
        micro_break_max_duration: Optional[float] = None,
        # Common aliases
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        mu: Optional[float] = None,
        mean: Optional[float] = None,
        sigma: Optional[float] = None,
        std: Optional[float] = None,
        micro_break_interval: Optional[Tuple[int, int]] = None,
        micro_break_freq_range: Optional[Tuple[int, int]] = None,
        micro_break_duration: Optional[Tuple[float, float]] = None,
        micro_break_duration_range: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
    ):
        # Resolve distribution
        if isinstance(distribution, str):
            dist_lower = distribution.strip().lower()
            if dist_lower in ("uniform",):
                self.distribution = JitterDistribution.UNIFORM
            elif dist_lower in ("gaussian", "truncated_gaussian"):
                self.distribution = JitterDistribution.TRUNCATED_GAUSSIAN
            else:
                raise ValueError(f"Unknown jitter distribution: {distribution}")
        else:
            self.distribution = distribution

        # Resolve min_delay / t_min (Default: 1800.0s = 30 minutes)
        resolved_min = min_delay if min_delay is not None else (t_min if t_min is not None else 1800.0)
        # Resolve max_delay / t_max (Default: 3600.0s = 1 hour)
        resolved_max = max_delay if max_delay is not None else (t_max if t_max is not None else 3600.0)

        if resolved_min < 0.0:
            raise ValueError(f"min_delay ({resolved_min}) cannot be negative.")
        if resolved_min > resolved_max:
            raise ValueError(f"min_delay ({resolved_min}) cannot exceed max_delay ({resolved_max}).")

        self.min_delay = float(resolved_min)
        self.max_delay = float(resolved_max)

        # Resolve mean_delay / mu / mean
        resolved_mean = mean_delay if mean_delay is not None else (mu if mu is not None else mean)
        if resolved_mean is not None:
            if not (self.min_delay <= resolved_mean <= self.max_delay):
                raise ValueError(f"mean ({resolved_mean}) must lie within [{self.min_delay}, {self.max_delay}].")
            self.mean_delay = float(resolved_mean)
        else:
            self.mean_delay = (self.min_delay + self.max_delay) / 2.0

        # Resolve std_dev / sigma / std
        resolved_std = std_dev if std_dev is not None else (sigma if sigma is not None else std)
        if resolved_std is not None:
            if resolved_std <= 0.0:
                raise ValueError(f"std_dev ({resolved_std}) must be strictly positive.")
            self.std_dev = float(resolved_std)
        else:
            self.std_dev = (self.max_delay - self.min_delay) / 6.0 if self.max_delay > self.min_delay else 1.0

        self.micro_break_enabled = micro_break_enabled

        # Resolve micro break intervals
        interval_tuple = micro_break_interval or micro_break_freq_range
        if interval_tuple is not None:
            self.micro_break_min_interval, self.micro_break_max_interval = interval_tuple
        else:
            self.micro_break_min_interval = micro_break_min_interval if micro_break_min_interval is not None else 15
            self.micro_break_max_interval = micro_break_max_interval if micro_break_max_interval is not None else 25

        if self.micro_break_min_interval <= 0 or self.micro_break_min_interval > self.micro_break_max_interval:
            raise ValueError(
                f"Invalid micro-break interval: ({self.micro_break_min_interval}, {self.micro_break_max_interval})"
            )

        # Resolve micro break durations
        duration_tuple = micro_break_duration or micro_break_duration_range
        if duration_tuple is not None:
            self.micro_break_min_duration, self.micro_break_max_duration = duration_tuple
        else:
            self.micro_break_min_duration = (
                micro_break_min_duration if micro_break_min_duration is not None else 10.0
            )
            self.micro_break_max_duration = (
                micro_break_max_duration if micro_break_max_duration is not None else 30.0
            )

        if self.micro_break_min_duration < 0 or self.micro_break_min_duration > self.micro_break_max_duration:
            raise ValueError(
                f"Invalid micro-break duration: ({self.micro_break_min_duration}, {self.micro_break_max_duration})"
            )

    @property
    def t_min(self) -> float:
        return self.min_delay

    @property
    def t_max(self) -> float:
        return self.max_delay

    @property
    def mu(self) -> float:
        return self.mean_delay if self.mean_delay is not None else (self.min_delay + self.max_delay) / 2.0

    @property
    def sigma(self) -> float:
        return self.std_dev if self.std_dev is not None else (self.max_delay - self.min_delay) / 6.0

    @property
    def micro_break_interval(self) -> Tuple[int, int]:
        return (self.micro_break_min_interval, self.micro_break_max_interval)

    @property
    def micro_break_duration(self) -> Tuple[float, float]:
        return (self.micro_break_min_duration, self.micro_break_max_duration)


class JitterEngine:
    """
    Dual-Distribution Jitter & Pacing Engine.
    Generates human-like delays and manages interruptible sleep pacing.
    """

    def __init__(
        self,
        config: Optional[JitterConfig] = None,
        seed: Optional[int] = None,
        time_fn: Optional[Callable[[], float]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
        # Allow passing config params directly
        distribution: Optional[Union[JitterDistribution, str]] = None,
        min_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        mean_delay: Optional[float] = None,
        std_dev: Optional[float] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        micro_break_enabled: Optional[bool] = None,
        micro_break_freq_range: Optional[Tuple[int, int]] = None,
        micro_break_duration_range: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            cfg_kwargs: Dict[str, Any] = {}
            if distribution is not None:
                cfg_kwargs["distribution"] = distribution
            if min_delay is not None:
                cfg_kwargs["min_delay"] = min_delay
            if max_delay is not None:
                cfg_kwargs["max_delay"] = max_delay
            if mean_delay is not None:
                cfg_kwargs["mean_delay"] = mean_delay
            elif mean is not None:
                cfg_kwargs["mean_delay"] = mean
            if std_dev is not None:
                cfg_kwargs["std_dev"] = std_dev
            elif std is not None:
                cfg_kwargs["std_dev"] = std
            if micro_break_enabled is not None:
                cfg_kwargs["micro_break_enabled"] = micro_break_enabled
            if micro_break_freq_range is not None:
                cfg_kwargs["micro_break_freq_range"] = micro_break_freq_range
            if micro_break_duration_range is not None:
                cfg_kwargs["micro_break_duration_range"] = micro_break_duration_range
            cfg_kwargs.update(kwargs)
            self.config = JitterConfig(**cfg_kwargs)

        self._rng = random.Random(seed)
        self._time_fn = time_fn or time.time
        self._sleep_fn = sleep_fn

        self._lock = threading.Lock()
        self._submission_count = 0
        self._next_break_target = self._sample_next_break_interval()
        self._total_micro_breaks = 0

    def _sample_next_break_interval(self) -> int:
        return self._rng.randint(
            self.config.micro_break_min_interval,
            self.config.micro_break_max_interval,
        )

    def calculate_base_jitter(self) -> float:
        """
        Sample a single base delay according to the configured distribution
        without advancing micro-break counters.
        """
        cfg = self.config
        t_min, t_max = cfg.min_delay, cfg.max_delay
        if t_min == t_max:
            return t_min

        if cfg.distribution == JitterDistribution.UNIFORM:
            return self._rng.uniform(t_min, t_max)

        # Truncated Gaussian with Rejection Sampling (strictly respects bounds)
        mu = cfg.mean_delay if cfg.mean_delay is not None else (t_min + t_max) / 2.0
        sigma = cfg.std_dev if cfg.std_dev is not None else (t_max - t_min) / 6.0
        if sigma <= 0.0:
            return mu

        # Rejection sampling up to 20 attempts
        for _ in range(20):
            sample = self._rng.gauss(mu, sigma)
            if t_min <= sample <= t_max:
                return sample

        # Fallback clamp on extreme tail
        return min(max(sample, t_min), t_max)

    def get_delay(self, increment_submission: bool = True) -> float:
        """Returns next delay value (including micro-breaks if due)."""
        decision = self.get_pacing_decision(increment_submission=increment_submission)
        return decision.delay

    def get_pacing_decision(self, increment_submission: bool = True) -> PacingDecision:
        """
        Evaluate and return the next pacing decision, assessing fatigue micro-breaks.
        Thread-safe.
        """
        with self._lock:
            cfg = self.config

            # Check if micro-break is due
            if cfg.micro_break_enabled and self._submission_count >= self._next_break_target:
                d_min, d_max = cfg.micro_break_min_duration, cfg.micro_break_max_duration
                break_delay = self._rng.uniform(d_min, d_max)

                # Reset counter and choose next threshold
                self._submission_count = 0
                self._next_break_target = self._sample_next_break_interval()
                self._total_micro_breaks += 1

                return PacingDecision(
                    delay=break_delay,
                    is_micro_break=True,
                    duration_break=break_delay,
                    submission_count=self._submission_count,
                    submissions_to_next_break=self._next_break_target,
                )

            # Normal pacing delay
            delay = self.calculate_base_jitter()
            if increment_submission:
                self._submission_count += 1

            remaining_to_break = max(0, self._next_break_target - self._submission_count)
            return PacingDecision(
                delay=delay,
                is_micro_break=False,
                duration_break=0.0,
                submission_count=self._submission_count,
                submissions_to_next_break=remaining_to_break,
            )

    def sleep(
        self,
        delay: float,
        stop_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
        step: float = 0.05,
    ) -> bool:
        """
        Interruptible pacing sleep.

        Args:
            delay: Duration in seconds to sleep.
            stop_event: Set when the fleet is stopping; causes immediate False return.
            pause_event: When provided and pause_event is CLEARED (not set), pause countdown.
            step: Polling slice duration.

        Returns:
            True if full delay elapsed successfully.
            False if aborted early due to stop_event.
        """
        if delay <= 0.0:
            return not (stop_event is not None and stop_event.is_set())

        if self._sleep_fn is not None:
            # Fast-path for mock virtual sleep testing
            if stop_event is not None and stop_event.is_set():
                return False
            self._sleep_fn(delay)
            return not (stop_event is not None and stop_event.is_set())

        start_time = self._time_fn()
        elapsed = 0.0

        # For long pacing delays (e.g. 1800s-3600s), adapt polling slice to 0.5s to eliminate
        # excessive busy looping (e.g. 72,000 iterations), while preserving fast stop_event responsiveness (<1ms).
        effective_step = step
        if step == 0.05 and delay >= 5.0:
            effective_step = 0.5

        while elapsed < delay:
            if stop_event is not None and stop_event.is_set():
                return False

            # If pause_event is provided and cleared (meaning paused in WorkerManager):
            if pause_event is not None and not pause_event.is_set():
                pause_t0 = self._time_fn()
                while not pause_event.is_set():
                    if stop_event is not None and stop_event.is_set():
                        return False
                    if stop_event is not None:
                        if stop_event.wait(timeout=0.2):
                            return False
                    else:
                        time.sleep(0.2)
                # Adjust start_time so paused duration does not count toward pacing delay
                start_time += (self._time_fn() - pause_t0)

            remaining = delay - elapsed
            wait_slice = min(remaining, effective_step)

            if stop_event is not None:
                if stop_event.wait(timeout=wait_slice):
                    return False
            else:
                time.sleep(wait_slice)

            elapsed = self._time_fn() - start_time

        return True

    def interruptible_sleep(
        self,
        seconds: float,
        stop_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
        step: float = 0.05,
    ) -> bool:
        """Alias for sleep() compliant with test suite signatures."""
        return self.sleep(delay=seconds, stop_event=stop_event, pause_event=pause_event, step=step)

    def reset(self) -> None:
        """Reset submission counters and micro-break state."""
        with self._lock:
            self._submission_count = 0
            self._next_break_target = self._sample_next_break_interval()
            self._total_micro_breaks = 0


@dataclass
class SchedulerConfig:
    """
    Configuration parameters for the Background Scheduler.
    Supports start delay, absolute start time, diurnal windows, batch limits, and cooldown.
    """
    start_delay_seconds: Optional[float] = None
    start_at: Optional[dt.datetime] = None
    active_windows: Optional[Union[List[str], str]] = None
    batch_limit: Optional[int] = None
    cooldown_seconds: float = 300.0

    def __init__(
        self,
        start_delay_seconds: Optional[float] = None,
        start_at: Optional[dt.datetime] = None,
        active_windows: Optional[Union[List[str], str]] = None,
        batch_limit: Optional[int] = None,
        batch_size: Optional[int] = None,
        cooldown_seconds: float = 300.0,
        **kwargs: Any,
    ):
        if start_delay_seconds is not None:
            if start_delay_seconds < 0.0:
                raise ValueError("start_delay_seconds cannot be negative")
            self.start_delay_seconds = float(start_delay_seconds)
        else:
            self.start_delay_seconds = None

        self.start_at = start_at

        # Parse and validate active windows
        self.active_windows = None
        if active_windows is not None:
            if isinstance(active_windows, str):
                parts = [p.strip() for p in re.split(r"[,;\n]+", active_windows) if p.strip()]
            else:
                parts = [str(p).strip() for p in active_windows if str(p).strip()]

            # Validate each window string
            for p in parts:
                TimeWindow.from_string(p)
            self.active_windows = parts if parts else None

        # Resolve batch limit / batch size
        resolved_batch = batch_limit if batch_limit is not None else batch_size
        if resolved_batch is not None:
            if resolved_batch <= 0:
                raise ValueError("batch_limit must be > 0")
            self.batch_limit = int(resolved_batch)
        else:
            self.batch_limit = None

        if cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds cannot be negative")
        self.cooldown_seconds = float(cooldown_seconds)

    @property
    def batch_size(self) -> Optional[int]:
        return self.batch_limit


class BackgroundScheduler:
    """
    Background Automation Scheduler.
    Orchestrates start delay countdown, diurnal operating windows, and batch quota cooldowns.
    Supports full clock injection (time_fn, datetime_fn, sleep_fn or time_provider).
    """

    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        time_fn: Optional[Callable[[], float]] = None,
        datetime_fn: Optional[Callable[[], dt.datetime]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
        time_provider: Optional[Any] = None,
        # Allow passing config parameters directly
        start_delay_seconds: Optional[float] = None,
        start_at: Optional[dt.datetime] = None,
        active_windows: Optional[Union[List[str], str]] = None,
        batch_limit: Optional[int] = None,
        batch_size: Optional[int] = None,
        cooldown_seconds: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            cfg_kwargs: Dict[str, Any] = {}
            if start_delay_seconds is not None:
                cfg_kwargs["start_delay_seconds"] = start_delay_seconds
            if start_at is not None:
                cfg_kwargs["start_at"] = start_at
            if active_windows is not None:
                cfg_kwargs["active_windows"] = active_windows
            if batch_limit is not None:
                cfg_kwargs["batch_limit"] = batch_limit
            elif batch_size is not None:
                cfg_kwargs["batch_limit"] = batch_size
            if cooldown_seconds is not None:
                cfg_kwargs["cooldown_seconds"] = cooldown_seconds
            cfg_kwargs.update(kwargs)
            self.config = SchedulerConfig(**cfg_kwargs)

        # Handle time provider injection
        if time_provider is not None:
            self._time_fn = getattr(time_provider, "time", time.time)
            if hasattr(time_provider, "now"):
                self._datetime_fn = time_provider.now
            elif hasattr(time_provider, "datetime"):
                self._datetime_fn = time_provider.datetime
            else:
                self._datetime_fn = dt.datetime.now
            self._sleep_fn = getattr(time_provider, "sleep", None)
        else:
            self._time_fn = time_fn or time.time
            self._datetime_fn = datetime_fn or dt.datetime.now
            self._sleep_fn = sleep_fn

        self._lock = threading.Lock()
        self._is_stopped = False
        self._init_timestamp = self._time_fn()
        try:
            self._init_datetime = self._datetime_fn()
        except Exception:
            self._init_datetime = dt.datetime.now()

        # Batch & Cooldown state
        self._current_batch_count = 0
        self._total_batches_completed = 0
        self._cooldown_until: Optional[float] = None

        # Parsed diurnal windows
        self._parsed_windows: List[TimeWindow] = []
        self._refresh_windows()

    def _refresh_windows(self) -> None:
        if self.config.active_windows:
            self._parsed_windows = [
                TimeWindow.from_string(w) for w in self.config.active_windows
            ]
        else:
            self._parsed_windows = []

    def configure(self, config: SchedulerConfig) -> None:
        """Dynamically update scheduler configuration."""
        with self._lock:
            self.config = config
            self._refresh_windows()

    # ------------------------------------------------------------------------
    # Start Delay & Countdown Logic
    # ------------------------------------------------------------------------

    def get_start_countdown(self, now: Optional[dt.datetime] = None) -> float:
        """
        Get remaining seconds until execution is scheduled to begin.
        Returns 0.0 if already past start time.
        """
        cfg = self.config

        # 1. Absolute datetime start
        if cfg.start_at is not None:
            current_dt = now or self._datetime_fn()
            # If cfg.start_at is naive and current_dt is tz-aware (or vice versa), reconcile
            if current_dt.tzinfo is not None and cfg.start_at.tzinfo is None:
                current_dt = current_dt.replace(tzinfo=None)
            elif current_dt.tzinfo is None and cfg.start_at.tzinfo is not None:
                cfg_start_at = cfg.start_at.replace(tzinfo=None)
                remaining = (cfg_start_at - current_dt).total_seconds()
                return max(0.0, remaining)

            remaining = (cfg.start_at - current_dt).total_seconds()
            return max(0.0, remaining)

        # 2. Relative countdown start
        if cfg.start_delay_seconds is not None and cfg.start_delay_seconds > 0.0:
            target_time = self._init_timestamp + cfg.start_delay_seconds
            current_time = self._time_fn()
            remaining = target_time - current_time
            return max(0.0, remaining)

        return 0.0

    def is_start_countdown_elapsed(self, now: Optional[dt.datetime] = None) -> bool:
        """Check if scheduled start delay has elapsed."""
        return self.get_start_countdown(now) <= 0.0

    def is_started(self, now: Optional[dt.datetime] = None) -> bool:
        """Alias for is_start_countdown_elapsed()."""
        return self.is_start_countdown_elapsed(now)

    def remaining_start_delay(self, now: Optional[dt.datetime] = None) -> float:
        """Alias for get_start_countdown()."""
        return self.get_start_countdown(now)

    # ------------------------------------------------------------------------
    # Diurnal Active Windows Logic
    # ------------------------------------------------------------------------

    def is_in_active_window(self, now: Optional[dt.datetime] = None) -> bool:
        """
        Check if the current time falls inside any configured active window.
        If no windows configured, returns True (24/7 active).
        """
        if not self._parsed_windows:
            return True

        current_dt = now or self._datetime_fn()
        t = current_dt.time()
        return any(window.contains(t) for window in self._parsed_windows)

    def is_active_now(self, now: Optional[dt.datetime] = None) -> bool:
        """Alias for is_in_active_window()."""
        return self.is_in_active_window(now)

    def get_seconds_until_next_window(self, now: Optional[dt.datetime] = None) -> float:
        """
        Calculate remaining seconds until the next active window opens.
        Returns 0.0 if currently inside an active window or if no windows configured.
        """
        if not self._parsed_windows or self.is_in_active_window(now):
            return 0.0

        current_dt = now or self._datetime_fn()
        today = current_dt.date()

        # Find all upcoming window openings today
        upcoming_starts: List[dt.datetime] = []
        for w in self._parsed_windows:
            w_start_dt = dt.datetime.combine(today, w.start_time)
            if current_dt.tzinfo is not None:
                w_start_dt = w_start_dt.replace(tzinfo=current_dt.tzinfo)
            if w_start_dt > current_dt:
                upcoming_starts.append(w_start_dt)

        if upcoming_starts:
            next_start = min(upcoming_starts)
            return (next_start - current_dt).total_seconds()

        # If all windows today have passed, pick the earliest window tomorrow
        tomorrow = today + dt.timedelta(days=1)
        tomorrow_starts: List[dt.datetime] = []
        for w in self._parsed_windows:
            t_start = dt.datetime.combine(tomorrow, w.start_time)
            if current_dt.tzinfo is not None:
                t_start = t_start.replace(tzinfo=current_dt.tzinfo)
            tomorrow_starts.append(t_start)

        next_start = min(tomorrow_starts)
        return (next_start - current_dt).total_seconds()

    def seconds_until_next_window(self, now: Optional[dt.datetime] = None) -> float:
        """Alias for get_seconds_until_next_window()."""
        return self.get_seconds_until_next_window(now)

    # ------------------------------------------------------------------------
    # Batch Quota & Cooldown Logic
    # ------------------------------------------------------------------------

    def record_submission(self) -> bool:
        """
        Record a successful submission.
        If batch_limit is reached, triggers cooldown period.

        Returns:
            True if batch quota was reached and cooldown initiated.
            False otherwise.
        """
        with self._lock:
            cfg = self.config
            batch_lim = cfg.batch_limit
            if batch_lim is None or batch_lim <= 0:
                return False

            self._current_batch_count += 1
            if self._current_batch_count >= batch_lim:
                self._current_batch_count = 0
                self._total_batches_completed += 1
                self._cooldown_until = self._time_fn() + cfg.cooldown_seconds
                return True

            return False

    def is_in_cooldown(self, now: Optional[float] = None) -> bool:
        """Check if the fleet is currently in a batch cooldown period."""
        with self._lock:
            if self._cooldown_until is None:
                return False

            curr_time = now if now is not None else self._time_fn()
            if curr_time < self._cooldown_until:
                return True

            # Cooldown expired
            self._cooldown_until = None
            return False

    def is_cooling_down(self, now: Optional[float] = None) -> bool:
        """Alias for is_in_cooldown()."""
        return self.is_in_cooldown(now)

    def get_cooldown_remaining(self, now: Optional[float] = None) -> float:
        """Get remaining seconds in current cooldown period. Returns 0.0 if not in cooldown."""
        with self._lock:
            if self._cooldown_until is None:
                return 0.0

            curr_time = now if now is not None else self._time_fn()
            remaining = self._cooldown_until - curr_time
            if remaining <= 0.0:
                self._cooldown_until = None
                return 0.0
            return remaining

    def remaining_cooldown(self, now: Optional[float] = None) -> float:
        """Alias for get_cooldown_remaining()."""
        return self.get_cooldown_remaining(now)

    # ------------------------------------------------------------------------
    # Operational State & Can Execute
    # ------------------------------------------------------------------------

    def get_state(self) -> SchedulerState:
        """Evaluate and return current operational state in order of precedence."""
        if self._is_stopped:
            return SchedulerState.STOPPED

        if not self.is_start_countdown_elapsed():
            return SchedulerState.COUNTDOWN

        if not self.is_in_active_window():
            return SchedulerState.WINDOW_WAIT

        if self.is_in_cooldown():
            return SchedulerState.COOLDOWN

        return SchedulerState.ACTIVE

    def can_submit(self) -> Tuple[bool, SchedulerState, str]:
        """
        Check if workers are currently permitted to submit forms.

        Returns:
            (can_submit, state, reason_message)
        """
        state = self.get_state()
        if state == SchedulerState.ACTIVE:
            return True, state, "Active and ready for submission"

        if state == SchedulerState.COUNTDOWN:
            remaining = self.get_start_countdown()
            return False, state, f"Waiting for start delay ({remaining:.1f}s remaining)"

        if state == SchedulerState.WINDOW_WAIT:
            remaining = self.get_seconds_until_next_window()
            return False, state, f"Outside diurnal window ({remaining:.1f}s until open)"

        if state == SchedulerState.COOLDOWN:
            remaining = self.get_cooldown_remaining()
            return False, state, f"Batch cooldown ({remaining:.1f}s remaining)"

        if state == SchedulerState.STOPPED:
            return False, state, "Scheduler has been stopped"

        return False, state, f"Scheduler state: {state.value}"

    def can_execute(self) -> Tuple[bool, str, float]:
        """
        DISPATCH contract: returns (can_execute, state/reason, seconds_until_available).
        """
        state = self.get_state()
        if state == SchedulerState.ACTIVE:
            return True, "ACTIVE", 0.0
        elif state == SchedulerState.COUNTDOWN:
            rem = self.get_start_countdown()
            return False, f"COUNTDOWN: {rem:.1f}s remaining", rem
        elif state == SchedulerState.WINDOW_WAIT:
            rem = self.get_seconds_until_next_window()
            return False, f"WINDOW_WAIT: {rem:.1f}s until next window", rem
        elif state == SchedulerState.COOLDOWN:
            rem = self.get_cooldown_remaining()
            return False, f"COOLDOWN: {rem:.1f}s remaining", rem
        elif state == SchedulerState.STOPPED:
            return False, "STOPPED", 0.0
        return False, state.value, 0.0

    def get_snapshot(self) -> SchedulerSnapshot:
        """Generate atomic, thread-safe snapshot for UI dashboard."""
        state = self.get_state()
        windows_str = ", ".join(self.config.active_windows) if self.config.active_windows else "24/7 Unrestricted"
        can_sub, _, msg = self.can_submit()

        return SchedulerSnapshot(
            state=state,
            is_active=can_sub,
            start_countdown_remaining=self.get_start_countdown(),
            window_wait_remaining=self.get_seconds_until_next_window(),
            cooldown_remaining=self.get_cooldown_remaining(),
            current_batch_count=self._current_batch_count,
            total_batches_completed=self._total_batches_completed,
            active_windows_str=windows_str,
            message=msg,
        )

    def wait_for_active(
        self,
        stop_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
        poll_interval: float = 0.5,
    ) -> bool:
        """
        Block execution interruptibly until the scheduler enters ACTIVE state.

        Returns:
            True if scheduler entered ACTIVE state.
            False if aborted early due to stop_event or explicit stop().
        """
        while not self._is_stopped:
            if stop_event is not None and stop_event.is_set():
                return False

            state = self.get_state()
            if state == SchedulerState.ACTIVE:
                return True

            if state == SchedulerState.STOPPED:
                return False

            # Determine wait duration based on condition
            if state == SchedulerState.COUNTDOWN:
                wait_sec = min(self.get_start_countdown(), poll_interval)
            elif state == SchedulerState.WINDOW_WAIT:
                wait_sec = min(self.get_seconds_until_next_window(), poll_interval)
            elif state == SchedulerState.COOLDOWN:
                wait_sec = min(self.get_cooldown_remaining(), poll_interval)
            else:
                wait_sec = poll_interval

            wait_sec = max(0.01, wait_sec)

            if self._sleep_fn is not None:
                self._sleep_fn(wait_sec)
            else:
                if stop_event is not None:
                    if stop_event.wait(timeout=wait_sec):
                        return False
                else:
                    time.sleep(wait_sec)

        return False

    def stop(self) -> None:
        """Halt the scheduler."""
        with self._lock:
            self._is_stopped = True

    def reset(self) -> None:
        """Reset internal timers, batch counters, and stopped status."""
        with self._lock:
            self._is_stopped = False
            self._init_timestamp = self._time_fn()
            try:
                self._init_datetime = self._datetime_fn()
            except Exception:
                self._init_datetime = dt.datetime.now()
            self._current_batch_count = 0
            self._total_batches_completed = 0
            self._cooldown_until = None
