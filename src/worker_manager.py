"""
src/worker_manager.py - Multi-Threaded Fleet Manager & Concurrency Orchestrator.

Milestone M3: Worker Fleet Manager & Background Automation.
Implements:
- WorkerState enum & FleetState enum
- WorkerCardSnapshot & WorkerFleetSnapshot (immutable, deep-copied, frozen)
- FleetConfig configuration dataclass
- WorkerManager:
  - 1 to 20 daemon worker threads
  - Deterministic 6-state FSM (IDLE -> SYNTHESIZING -> PACING -> SUBMITTING -> SUCCESS / ERROR)
  - Pre-submission ticket reservation pattern (_reserve_submission_slot) with rollback (_refund_work)
    under _quota_lock guaranteeing exact target count without overshoot
  - Thread-isolated PersonaEngine seeded per worker
  - Thread-safe shared TallyClient with Dry-Run & Live HTTP resilience
  - External _stats_lock synchronization around StatsAnalyzer
  - Interruptible pacing sleep via JitterEngine
  - Atomic snapshot publication for high-frequency Streamlit UI polling
"""

from __future__ import annotations

import enum
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.persona_engine import PersonaEngine, RespondentProfile
from src.scheduler import (
    BackgroundScheduler,
    JitterConfig,
    JitterDistribution,
    JitterEngine,
    PacingDecision,
    SchedulerConfig,
    SchedulerState,
)
from src.stats_analyzer import StatsAnalyzer, StatsSummary
from src.tally_client import (
    TallyClient,
    TallyClientError,
    TallyConnectionError,
    TallyRateLimitError,
    TallyServerError,
    TallySubmissionError,
    TallyValidationError,
)
from src.tally_schema import TallySchema

logger = logging.getLogger("worker_manager")


class WorkerState(str, enum.Enum):
    """6-State Finite State Machine states for each worker thread."""
    IDLE = "IDLE"
    SYNTHESIZING = "SYNTHESIZING"
    PACING = "PACING"
    SUBMITTING = "SUBMITTING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


class FleetState(str, enum.Enum):
    """Operational lifecycle states of the Worker Fleet."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class WorkerCardSnapshot:
    """
    Immutable representation of an individual worker thread's state and metrics.
    Consumed directly by UI cards, table views, and test suites.
    """
    worker_id: int
    state: WorkerState
    current_task: str
    latency_ms: float = 0.0
    last_error: Optional[str] = None
    submissions_count: int = 0
    success_count: int = 0
    error_count: int = 0
    current_persona: Optional[str] = None
    last_updated: float = 0.0

    @property
    def submission_count(self) -> int:
        return self.submissions_count

    @property
    def last_latency_ms(self) -> float:
        return self.latency_ms


# Alias for backward compatibility
WorkerCardState = WorkerCardSnapshot


@dataclass(frozen=True)
class WorkerFleetSnapshot:
    """
    Immutable, point-in-time snapshot of the entire worker fleet.
    Read atomically by the UI thread without holding worker locks.
    """
    total_target: int
    submitted_count: int
    success_count: int
    error_count: int
    active_workers: int
    total_workers: int
    is_running: bool
    is_paused: bool
    is_completed: bool
    fleet_state: str
    workers: Tuple[WorkerCardSnapshot, ...]
    start_time: Optional[float] = None
    elapsed_seconds: float = 0.0
    submissions_per_minute: float = 0.0
    timestamp: float = 0.0

    @property
    def target_count(self) -> int:
        return self.total_target

    @property
    def reserved_count(self) -> int:
        return self.submitted_count

    @property
    def active_worker_count(self) -> int:
        return self.active_workers

    @property
    def worker_cards(self) -> Tuple[WorkerCardSnapshot, ...]:
        return self.workers


@dataclass
class FleetConfig:
    """Configuration bundle for the Worker Manager."""
    target_count: int = 150
    num_workers: int = 6
    dry_run: bool = True
    delay_range: Tuple[float, float] = (1800.0, 3600.0)
    jitter_config: Optional[Union[JitterConfig, Dict[str, Any]]] = None
    demographic_weights: Optional[Dict[str, Any]] = None
    master_seed: Optional[int] = None
    scheduler_config: Optional[Union[SchedulerConfig, Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        if self.target_count < 1:
            raise ValueError(f"target_count must be >= 1, got {self.target_count}")
        if not (1 <= self.num_workers <= 20):
            raise ValueError(f"num_workers must be between 1 and 20, got {self.num_workers}")


class WorkerManager:
    """
    Production-grade multi-threaded Worker Fleet Manager.
    Coordinates 1 to 20 concurrent worker threads with 6-state FSM,
    exact quota reservation, snapshot isolation, and safe module integration.
    """

    def __init__(
        self,
        config: Optional[FleetConfig] = None,
        tally_client: Optional[Any] = None,
        scheduler: Optional[BackgroundScheduler] = None,
        stats_analyzer: Optional[StatsAnalyzer] = None,
    ) -> None:
        self._config = config or FleetConfig()

        # Concurrency synchronization events
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Initial state: not paused
        self._fleet_completed_event = threading.Event()

        # Mutex locks hierarchy (Level 1 -> Level 2 -> Level 3 -> Level 4)
        self._fleet_lock = threading.RLock()    # Level 1: configure, start, pause, resume, stop
        self._quota_lock = threading.RLock()    # Level 2: atomic ticket reservation
        self._state_lock = threading.RLock()    # Level 2: worker card telemetry
        self._snapshot_lock = threading.Lock()  # Level 3: snapshot publication & copy-on-read
        self._stats_lock = threading.Lock()     # Level 4: StatsAnalyzer sample ingestion
        self._history_lock = threading.Lock()   # Level 4: submission history recording

        # Fleet state variables
        self._is_running = False
        self._is_paused = False
        self._start_time: Optional[float] = None
        self._threads: List[threading.Thread] = []

        # Atomic counters (guarded by _quota_lock)
        self._reserved_count = 0
        self._submitted_count = 0
        self._success_count = 0
        self._error_count = 0

        # Sub-modules
        self._stats_analyzer = stats_analyzer or StatsAnalyzer()
        self._tally_client = tally_client or TallyClient(dry_run=self._config.dry_run)

        # Scheduler
        if scheduler is not None:
            self._scheduler = scheduler
        else:
            sched_cfg = self._resolve_scheduler_config(self._config.scheduler_config)
            self._scheduler = BackgroundScheduler(config=sched_cfg)

        self._history_records: List[Dict[str, Any]] = []
        self._calibrated_profiles: List[RespondentProfile] = []

        # Per-worker mutable tracking (guarded by _state_lock)
        self._worker_records: Dict[int, Dict[str, Any]] = {}
        self._init_worker_records(self._config.num_workers)

        # Publish initial snapshot
        self._current_snapshot = self._build_snapshot()

    # ------------------------------------------------------------------------
    # Helpers & Config Resolution
    # ------------------------------------------------------------------------

    def _resolve_jitter_config(self, jcfg: Optional[Union[JitterConfig, Dict[str, Any]]]) -> JitterConfig:
        if jcfg is None:
            return JitterConfig(
                min_delay=self._config.delay_range[0],
                max_delay=self._config.delay_range[1],
            )
        if isinstance(jcfg, JitterConfig):
            return jcfg
        return JitterConfig(**jcfg)

    def _resolve_scheduler_config(self, scfg: Optional[Union[SchedulerConfig, Dict[str, Any]]]) -> SchedulerConfig:
        if scfg is None:
            return SchedulerConfig()
        if isinstance(scfg, SchedulerConfig):
            return scfg
        return SchedulerConfig(**scfg)

    def _init_worker_records(self, count: int) -> None:
        """Initializes blank tracking records for workers."""
        now = time.time()
        with self._state_lock:
            self._worker_records = {
                i: {
                    "worker_id": i,
                    "state": WorkerState.IDLE,
                    "current_task": "Idle",
                    "latency_ms": 0.0,
                    "last_error": None,
                    "submissions_count": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "current_persona": None,
                    "last_updated": now,
                }
                for i in range(count)
            }

    # ------------------------------------------------------------------------
    # Public Fleet Properties
    # ------------------------------------------------------------------------

    @property
    def fleet_state(self) -> FleetState:
        with self._fleet_lock:
            if not self._is_running:
                if self._success_count >= self._config.target_count and self._config.target_count > 0:
                    return FleetState.COMPLETED
                return FleetState.IDLE
            if self._is_paused:
                return FleetState.PAUSED
            if self._fleet_completed_event.is_set() or self._success_count >= self._config.target_count:
                return FleetState.COMPLETED
            return FleetState.RUNNING

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def active_workers(self) -> int:
        with self._state_lock:
            return sum(1 for r in self._worker_records.values() if r["state"] != WorkerState.IDLE)

    @property
    def stats_analyzer(self) -> StatsAnalyzer:
        return self._stats_analyzer

    @property
    def scheduler(self) -> BackgroundScheduler:
        return self._scheduler

    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------

    def configure(
        self,
        target_count: Optional[int] = None,
        num_workers: Optional[int] = None,
        dry_run: Optional[bool] = None,
        delay_range: Optional[Tuple[float, float]] = None,
        jitter_config: Optional[Union[JitterConfig, Dict[str, Any]]] = None,
        demographic_weights: Optional[Dict[str, Any]] = None,
        master_seed: Optional[int] = None,
        scheduler_config: Optional[Union[SchedulerConfig, Dict[str, Any]]] = None,
    ) -> None:
        """
        Reconfigures the fleet settings. Cannot be called while fleet is running.
        """
        with self._fleet_lock:
            if self._is_running:
                raise RuntimeError("Cannot reconfigure fleet while running. Call stop() first.")

            if target_count is not None:
                if target_count < 1:
                    raise ValueError(f"target_count must be >= 1, got {target_count}")
                self._config.target_count = target_count

            if num_workers is not None:
                if not (1 <= num_workers <= 20):
                    raise ValueError(f"num_workers must be between 1 and 20, got {num_workers}")
                self._config.num_workers = num_workers
                self._init_worker_records(num_workers)

            if dry_run is not None:
                self._config.dry_run = dry_run
                if hasattr(self._tally_client, "dry_run"):
                    self._tally_client.dry_run = dry_run

            if delay_range is not None:
                self._config.delay_range = delay_range

            if jitter_config is not None:
                self._config.jitter_config = jitter_config

            if demographic_weights is not None:
                self._config.demographic_weights = demographic_weights

            if master_seed is not None:
                self._config.master_seed = master_seed

            if scheduler_config is not None:
                self._config.scheduler_config = scheduler_config
                resolved_sc = self._resolve_scheduler_config(scheduler_config)
                self._scheduler.configure(resolved_sc)

            if not self._is_running:
                self._calibrated_profiles = []

            self._publish_snapshot()

    # ------------------------------------------------------------------------
    # Lifecycle Controls: start, pause, resume, stop
    # ------------------------------------------------------------------------

    def start(self) -> None:
        """
        Starts the worker fleet threads.
        """
        with self._fleet_lock:
            if self._is_running:
                if self._is_paused:
                    self.resume()
                return

            self._is_running = True
            self._is_paused = False
            self._start_time = time.time()
            self._stop_event.clear()
            self._pause_event.set()
            self._fleet_completed_event.clear()

            # Align reservation counter
            with self._quota_lock:
                self._reserved_count = self._success_count

            # If NCKH calibrated mode is active, pre-calibrate fleet profiles
            engine_cfg = dict(self._config.demographic_weights) if isinstance(self._config.demographic_weights, dict) else {}
            is_nckh = (
                engine_cfg.get("nckh_mode", False)
                or engine_cfg.get("alpha_ceiling") is not None
                or (engine_cfg.get("alpha_threshold", 0.82) < 0.80)
            )
            if is_nckh and not self._calibrated_profiles:
                m_seed = self._config.master_seed if self._config.master_seed is not None else 42
                calibrator = PersonaEngine(
                    config=engine_cfg,
                    seed=m_seed,
                    sample_size=self._config.target_count,
                    nckh_mode=True,
                )
                self._calibrated_profiles = calibrator.generate_batch(
                    sample_size=self._config.target_count,
                    demographic_weights=engine_cfg,
                )

            # Initialize worker records
            self._init_worker_records(self._config.num_workers)

            self._threads = []
            for i in range(self._config.num_workers):
                t = threading.Thread(
                    target=self._worker_loop,
                    args=(i,),
                    name=f"WorkerThread-{i:02d}",
                    daemon=True,
                )
                self._threads.append(t)
                t.start()

        self._publish_snapshot()
        logger.info(
            f"Fleet started with {self._config.num_workers} workers targeting {self._config.target_count} submissions."
        )

    def pause(self) -> None:
        """
        Pauses worker execution at safe checkpoints.
        """
        with self._fleet_lock:
            if not self._is_running or self._is_paused:
                return
            self._is_paused = True
            self._pause_event.clear()

        self._publish_snapshot()
        logger.info("Fleet paused.")

    def resume(self) -> None:
        """
        Resumes worker execution from pause state.
        """
        with self._fleet_lock:
            if not self._is_running or not self._is_paused:
                return
            self._is_paused = False
            self._pause_event.set()

        self._publish_snapshot()
        logger.info("Fleet resumed.")

    def stop(self, timeout: float = 5.0) -> None:
        """
        Gracefully stops the worker fleet and joins all threads.
        """
        with self._fleet_lock:
            if not self._is_running:
                return
            self._stop_event.set()
            self._pause_event.set()  # Unblock any paused threads
            threads_to_join = list(self._threads)

        logger.info(f"Stopping fleet (joining {len(threads_to_join)} threads, timeout={timeout}s)...")
        for t in threads_to_join:
            t.join(timeout=timeout)

        with self._fleet_lock:
            self._is_running = False
            self._is_paused = False
            self._threads.clear()

            # Mark all workers as IDLE
            with self._state_lock:
                for r in self._worker_records.values():
                    r["state"] = WorkerState.IDLE
                    r["current_task"] = "Stopped"

        self._publish_snapshot()
        logger.info("Fleet successfully stopped.")

    def wait_until_complete(self, timeout: Optional[float] = None) -> bool:
        """
        Blocks until target quota is reached or timeout elapses.
        Returns True if target was reached; False if timed out.
        """
        start_wait = time.time()
        while True:
            with self._quota_lock:
                if self._success_count >= self._config.target_count:
                    return True

            if self._stop_event.is_set():
                return False

            if timeout is not None:
                if (time.time() - start_wait) >= timeout:
                    return self._success_count >= self._config.target_count

            # Wait briefly on completion event or 0.05s
            if self._fleet_completed_event.wait(timeout=0.05):
                return True

    # ------------------------------------------------------------------------
    # Atomic Work Reservation & Quota Cut-off
    # ------------------------------------------------------------------------

    def _reserve_submission_slot(self) -> bool:
        """
        Atomically reserve a submission slot.
        Returns True if reservation succeeded within quota; False if quota is full or stopped.
        Guarantees zero overshooting across up to 20 concurrent threads.
        """
        with self._quota_lock:
            if self._stop_event.is_set():
                return False
            if self._success_count >= self._config.target_count:
                self._fleet_completed_event.set()
                return False
            if self._reserved_count >= self._config.target_count:
                return False

            self._reserved_count += 1
            self._submitted_count += 1
            return True

    def _claim_work(self) -> bool:
        """Alias for _reserve_submission_slot()."""
        return self._reserve_submission_slot()

    def _commit_success(self) -> None:
        """Atomically record confirmed success under _quota_lock."""
        with self._quota_lock:
            self._success_count += 1
            if self._success_count >= self._config.target_count:
                self._fleet_completed_event.set()

    def _refund_work(self, profile: Optional[RespondentProfile] = None) -> None:
        """Atomically releases a reservation on failure to allow another attempt and preserves calibrated profile."""
        with self._quota_lock:
            self._reserved_count = max(self._success_count, self._reserved_count - 1)
            if profile is not None and getattr(self, "_calibrated_profiles", None) is not None:
                self._calibrated_profiles.insert(0, profile)

    # ------------------------------------------------------------------------
    # Worker Loop & FSM Transitions
    # ------------------------------------------------------------------------

    def _update_worker(
        self,
        worker_id: int,
        state: WorkerState,
        task: str,
        latency_ms: Optional[float] = None,
        last_error: Optional[str] = None,
        persona_str: Optional[str] = None,
        inc_success: bool = False,
        inc_error: bool = False,
    ) -> None:
        """Updates per-worker telemetry and publishes snapshot atomically."""
        with self._state_lock:
            if worker_id not in self._worker_records:
                return
            rec = self._worker_records[worker_id]
            rec["state"] = state
            rec["current_task"] = task
            rec["last_updated"] = time.time()
            if latency_ms is not None:
                rec["latency_ms"] = latency_ms
            if last_error is not None:
                rec["last_error"] = last_error
            if persona_str is not None:
                rec["current_persona"] = persona_str
            if inc_success:
                rec["success_count"] += 1
                rec["submissions_count"] += 1
            if inc_error:
                rec["error_count"] += 1
                rec["submissions_count"] += 1
                with self._quota_lock:
                    self._error_count += 1

        self._publish_snapshot()

    def _worker_loop(self, worker_id: int) -> None:
        """
        Main execution loop for worker thread `worker_id`.
        Executes the 6-state FSM with complete thread isolation.
        """
        # Independent PRNG seeding per worker
        w_seed = (
            (self._config.master_seed + worker_id * 10007)
            if self._config.master_seed is not None
            else None
        )
        engine_cfg = dict(self._config.demographic_weights) if isinstance(self._config.demographic_weights, dict) else {}
        is_nckh = (
            engine_cfg.get("nckh_mode", False)
            or engine_cfg.get("alpha_ceiling") is not None
            or (engine_cfg.get("alpha_threshold", 0.82) < 0.80)
        )
        persona_engine = PersonaEngine(
            config=engine_cfg,
            seed=w_seed,
            sample_size=self._config.target_count if is_nckh else None,
            nckh_mode=is_nckh if is_nckh else None,
        )

        # Worker-specific JitterEngine
        jitter_cfg = self._resolve_jitter_config(self._config.jitter_config)
        jitter_seed = (w_seed + 99991) if w_seed is not None else None
        jitter_engine = JitterEngine(config=jitter_cfg, seed=jitter_seed)

        logger.debug(f"[Worker-{worker_id:02d}] Thread started.")

        while not self._stop_event.is_set():
            # 1. Pause checkpoint
            if not self._pause_event.is_set():
                self._update_worker(worker_id, WorkerState.IDLE, "Paused")
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break

            # 2. Scheduler active window & cooldown check
            if not self._scheduler.is_in_active_window() or self._scheduler.is_in_cooldown():
                can_run, state, reason = self._scheduler.can_submit()
                if not can_run:
                    self._update_worker(worker_id, WorkerState.IDLE, reason)
                    if self._stop_event.wait(timeout=0.2):
                        break
                    continue

            # 3. Work slot reservation checkpoint (Atomic quota check)
            if not self._reserve_submission_slot():
                self._update_worker(worker_id, WorkerState.IDLE, "Idle (Target Reached)")
                if self._stop_event.wait(timeout=0.1):
                    break
                with self._quota_lock:
                    if self._success_count >= self._config.target_count:
                        break
                continue

            try:
                # 4. SYNTHESIZING State
                self._update_worker(worker_id, WorkerState.SYNTHESIZING, "Synthesizing Respondent Persona...")
                profile = None
                with self._quota_lock:
                    if self._calibrated_profiles:
                        profile = self._calibrated_profiles.pop(0)
                if profile is None:
                    profile = persona_engine.generate_respondent(
                        demographic_weights=self._config.demographic_weights
                    )
                persona_tag = (
                    f"{'Nữ' if 'Nu' in getattr(profile, 'gender', '') else 'Nam'} | "
                    f"{getattr(profile, 'age', 21)}t | "
                    f"{str(getattr(profile, 'education', ''))[:6]}..."
                )

                # 5. PACING State (Human-like delay with interruptible sleep)
                pacing_decision = jitter_engine.get_pacing_decision(increment_submission=True)
                pacing_delay = pacing_decision.delay
                status_label = pacing_decision.get_status_label()

                self._update_worker(
                    worker_id,
                    WorkerState.PACING,
                    status_label,
                    persona_str=persona_tag,
                )

                # Interruptible pacing sleep: aborts cleanly if stop_event is set or pauses cleanly
                sleep_completed = jitter_engine.sleep(
                    delay=pacing_delay,
                    stop_event=self._stop_event,
                    pause_event=self._pause_event,
                )
                if not sleep_completed or self._stop_event.is_set():
                    self._refund_work(profile)
                    break

                # Re-check pause after pacing
                if not self._pause_event.is_set():
                    self._update_worker(worker_id, WorkerState.IDLE, "Paused")
                    self._pause_event.wait()
                    if self._stop_event.is_set():
                        self._refund_work(profile)
                        break

                # 6. SUBMITTING State
                self._update_worker(worker_id, WorkerState.SUBMITTING, "Submitting to Tally API...")

                # Formulate canonical Tally payload
                payload = TallySchema.get_payload(
                    responses=profile.payload_responses,
                    session_uuid=str(uuid.uuid4()),
                    respondent_uuid=str(uuid.uuid4()),
                    is_completed=True,
                )

                t_start = time.perf_counter()
                result = self._tally_client.submit(payload=payload, dry_run=self._config.dry_run)
                latency_ms = (time.perf_counter() - t_start) * 1000.0

                # 7. SUCCESS State
                sub_id = result.get("data", {}).get("submissionId", str(uuid.uuid4()))

                # Feed Likert scores to StatsAnalyzer under mutex lock (_stats_lock)
                with self._stats_lock:
                    self._stats_analyzer.add_sample(profile.likert_scores)

                # Commit success atomically
                self._commit_success()

                # Record submission in scheduler (triggers cooldown if batch limit reached)
                self._scheduler.record_submission()

                # Record history under mutex lock
                with self._history_lock:
                    self._history_records.append({
                        "submission_id": sub_id,
                        "timestamp": time.time(),
                        "worker_id": worker_id,
                        "age": getattr(profile, "age", None),
                        "gender": getattr(profile, "gender", None),
                        "location": getattr(profile, "location", None),
                        "education": getattr(profile, "education", None),
                        "employment": getattr(profile, "employment", None),
                        "profession": getattr(profile, "profession", None),
                        "frequency": getattr(profile, "frequency", None),
                        "categories": getattr(profile, "categories", None),
                        "pain_points": getattr(profile, "pain_points", None),
                        "awareness": getattr(profile, "awareness", None),
                        "latency_ms": latency_ms,
                        "dry_run": self._config.dry_run,
                        "likert_scores": profile.likert_scores,
                    })

                self._update_worker(
                    worker_id,
                    WorkerState.SUCCESS,
                    f"Success: {sub_id[:16]}",
                    latency_ms=latency_ms,
                    inc_success=True,
                )

                # Brief inter-iteration yield
                if self._stop_event.wait(timeout=0.02):
                    break

            except TallyRateLimitError as exc:
                logger.warning(f"[Worker-{worker_id:02d}] Rate limit encountered: {exc}")
                self._refund_work(profile)
                self._update_worker(
                    worker_id,
                    WorkerState.ERROR,
                    "HTTP 429 Rate Limit",
                    last_error=str(exc),
                    inc_error=True,
                )
                retry_after = getattr(exc, "retry_after", None) or 1.0
                self._stop_event.wait(timeout=min(float(retry_after), 10.0))

            except (TallyServerError, TallyConnectionError, TallySubmissionError, TallyClientError) as exc:
                logger.error(f"[Worker-{worker_id:02d}] Tally submission failure: {exc}")
                self._refund_work(profile)
                self._update_worker(
                    worker_id,
                    WorkerState.ERROR,
                    f"Error: {exc.__class__.__name__}",
                    last_error=str(exc),
                    inc_error=True,
                )
                self._stop_event.wait(timeout=0.5)

            except Exception as exc:
                logger.exception(f"[Worker-{worker_id:02d}] Unhandled exception in worker loop: {exc}")
                self._refund_work(profile)
                self._update_worker(
                    worker_id,
                    WorkerState.ERROR,
                    f"Fatal: {str(exc)[:25]}",
                    last_error=str(exc),
                    inc_error=True,
                )
                self._stop_event.wait(timeout=0.5)

        self._update_worker(worker_id, WorkerState.IDLE, "Idle")
        logger.debug(f"[Worker-{worker_id:02d}] Thread terminated cleanly.")

    # ------------------------------------------------------------------------
    # Snapshot Publication & Queries
    # ------------------------------------------------------------------------

    def _build_snapshot(self) -> WorkerFleetSnapshot:
        """Builds an immutable WorkerFleetSnapshot from current state."""
        now = time.time()
        running = self._is_running
        paused = self._is_paused
        start = self._start_time

        elapsed = (now - start) if (running and start) else 0.0

        with self._quota_lock:
            succ = self._success_count
            resv = self._reserved_count
            errs = self._error_count
            tgt = self._config.target_count

        spm = (succ / (elapsed / 60.0)) if elapsed > 1.0 else 0.0

        with self._state_lock:
            worker_cards = tuple(
                WorkerCardSnapshot(
                    worker_id=r["worker_id"],
                    state=r["state"],
                    current_task=r["current_task"],
                    latency_ms=r["latency_ms"],
                    last_error=r["last_error"],
                    submissions_count=r["submissions_count"],
                    success_count=r["success_count"],
                    error_count=r["error_count"],
                    current_persona=r["current_persona"],
                    last_updated=r["last_updated"],
                )
                for r in sorted(self._worker_records.values(), key=lambda x: x["worker_id"])
            )

        active_count = sum(1 for c in worker_cards if c.state != WorkerState.IDLE)
        is_completed = (succ >= tgt and tgt > 0)

        # Compute fleet state string
        if not running:
            f_state = FleetState.COMPLETED.value if is_completed else FleetState.IDLE.value
        elif paused:
            f_state = FleetState.PAUSED.value
        elif is_completed:
            f_state = FleetState.COMPLETED.value
        else:
            f_state = FleetState.RUNNING.value

        return WorkerFleetSnapshot(
            total_target=tgt,
            submitted_count=resv,
            success_count=succ,
            error_count=errs,
            active_workers=active_count,
            total_workers=self._config.num_workers,
            is_running=running,
            is_paused=paused,
            is_completed=is_completed,
            fleet_state=f_state,
            workers=worker_cards,
            start_time=start,
            elapsed_seconds=elapsed,
            submissions_per_minute=round(spm, 2),
            timestamp=now,
        )

    def _publish_snapshot(self) -> None:
        """Publishes fresh immutable snapshot under snapshot lock."""
        new_snap = self._build_snapshot()
        with self._snapshot_lock:
            self._current_snapshot = new_snap

    def get_snapshot(self) -> WorkerFleetSnapshot:
        """
        Returns atomic immutable snapshot. Non-blocking, thread-safe (< 0.05ms).
        """
        with self._snapshot_lock:
            return self._current_snapshot

    def get_stats_summary(self) -> StatsSummary:
        """Returns thread-safe psychometric stats summary."""
        with self._stats_lock:
            return self._stats_analyzer.get_summary()

    def get_history_df(self) -> pd.DataFrame:
        """Returns in-memory submission history as a pandas DataFrame."""
        with self._history_lock:
            if not self._history_records:
                return pd.DataFrame()
            return pd.DataFrame(self._history_records)

    def reset(self) -> None:
        """Resets all metrics, history, and stats (fleet must be stopped)."""
        with self._fleet_lock:
            if self._is_running:
                raise RuntimeError("Cannot reset while fleet is running. Call stop() first.")

            with self._quota_lock:
                self._reserved_count = 0
                self._submitted_count = 0
                self._success_count = 0
                self._error_count = 0

            self._start_time = None
            self._fleet_completed_event.clear()
            self._init_worker_records(self._config.num_workers)

        with self._stats_lock:
            self._stats_analyzer = StatsAnalyzer()

        with self._history_lock:
            self._history_records.clear()

        self._publish_snapshot()
