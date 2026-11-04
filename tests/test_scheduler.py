"""
tests/test_scheduler.py - Comprehensive Unit & Statistical Test Suite for Scheduler & Jitter Engine.

Covers:
- Dual-Distribution Jitter Engine (Uniform & Truncated Gaussian bounds, moments, KS-test)
- Fatigue Micro-breaks (intervals, durations, reset, disabled mode)
- Interruptible Pacing Sleep (abort on stop event, pause handling)
- Diurnal Active Windows (parsing, multi-range matching, overnight wrapping, wait calculations)
- Start Delay Countdown (relative delay, absolute target datetime)
- Batch Quota & Cooldown (quota triggering, duration enforcement, counter reset)
"""

from __future__ import annotations

import datetime as dt
import threading
import time
import unittest
from typing import List, Optional

import numpy as np
from scipy import stats

from src.scheduler import (
    BackgroundScheduler,
    DistributionType,
    JitterConfig,
    JitterDistribution,
    JitterEngine,
    PacingDecision,
    SchedulerConfig,
    SchedulerSnapshot,
    SchedulerState,
    TimeWindow,
)


class MockTimeProvider:
    """Injectable virtual clock for deterministic instantaneous temporal testing."""

    def __init__(
        self,
        start_timestamp: float = 1798848000.0,
        start_dt: Optional[dt.datetime] = None,
    ) -> None:
        if start_dt is not None:
            self._current_dt = start_dt
            self._current_time = start_dt.timestamp()
        else:
            self._current_time = start_timestamp
            self._current_dt = dt.datetime.fromtimestamp(start_timestamp, tz=dt.timezone.utc)
        self.sleep_calls: List[float] = []

    def time(self) -> float:
        return self._current_time

    def now(self) -> dt.datetime:
        return self._current_dt

    def datetime(self) -> dt.datetime:
        return self._current_dt

    def advance(self, seconds: float) -> None:
        self._current_time += seconds
        self._current_dt += dt.timedelta(seconds=seconds)

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.advance(seconds)


class TestJitterEngineDistributions(unittest.TestCase):
    """Statistical and mathematical tests for JitterEngine distributions."""

    def test_uniform_jitter_strict_bounds(self) -> None:
        """Uniform distribution: 2,000 samples strictly within [t_min, t_max]."""
        engine = JitterEngine(
            distribution=JitterDistribution.UNIFORM,
            min_delay=2.5,
            max_delay=7.5,
            seed=12345,
        )
        samples = [engine.calculate_base_jitter() for _ in range(2000)]
        self.assertEqual(len(samples), 2000)
        self.assertTrue(all(2.5 <= s <= 7.5 for s in samples))
        self.assertAlmostEqual(min(samples), 2.5, delta=0.2)
        self.assertAlmostEqual(max(samples), 7.5, delta=0.2)

    def test_uniform_jitter_empirical_moments(self) -> None:
        """Uniform distribution: empirical mean and variance match theoretical moments."""
        # Theoretical: E[X] = (2.0 + 8.0) / 2 = 5.0, Var(X) = (8 - 2)^2 / 12 = 3.0
        engine = JitterEngine(
            distribution=JitterDistribution.UNIFORM,
            min_delay=2.0,
            max_delay=8.0,
            seed=4242,
        )
        samples = np.array([engine.calculate_base_jitter() for _ in range(10000)])
        sample_mean = float(np.mean(samples))
        sample_var = float(np.var(samples))

        self.assertAlmostEqual(sample_mean, 5.0, delta=0.08)
        self.assertAlmostEqual(sample_var, 3.0, delta=0.15)

    def test_uniform_jitter_kolmogorov_smirnov_test(self) -> None:
        """Uniform distribution: Kolmogorov-Smirnov test verifies goodness of fit."""
        engine = JitterEngine(
            distribution=JitterDistribution.UNIFORM,
            min_delay=3.0,
            max_delay=9.0,
            seed=9999,
        )
        samples = [engine.calculate_base_jitter() for _ in range(2500)]
        ks_res = stats.kstest(samples, "uniform", args=(3.0, 6.0))
        # p-value > 0.01 confirms samples are consistent with uniform distribution
        self.assertGreater(ks_res.pvalue, 0.01)

    def test_gaussian_jitter_strict_bounds(self) -> None:
        """Truncated Gaussian: 3,000 samples strictly respect [t_min, t_max]."""
        engine = JitterEngine(
            distribution=JitterDistribution.TRUNCATED_GAUSSIAN,
            min_delay=1.0,
            max_delay=9.0,
            mean=5.0,
            std=1.5,
            seed=54321,
        )
        samples = [engine.calculate_base_jitter() for _ in range(3000)]
        self.assertTrue(all(1.0 <= s <= 9.0 for s in samples))

    def test_gaussian_jitter_empirical_moments(self) -> None:
        """Truncated Gaussian: empirical mean and std dev match target parameters."""
        engine = JitterEngine(
            distribution=JitterDistribution.TRUNCATED_GAUSSIAN,
            min_delay=1.0,
            max_delay=9.0,
            mean=5.0,
            std=1.0,
            seed=7777,
        )
        samples = np.array([engine.calculate_base_jitter() for _ in range(10000)])
        sample_mean = float(np.mean(samples))
        sample_std = float(np.std(samples))

        self.assertAlmostEqual(sample_mean, 5.0, delta=0.05)
        self.assertAlmostEqual(sample_std, 1.0, delta=0.05)

    def test_gaussian_jitter_truncation_clamping(self) -> None:
        """Truncated Gaussian: narrow bounds with heavy tails produce 100% compliant samples."""
        engine = JitterEngine(
            distribution=JitterDistribution.TRUNCATED_GAUSSIAN,
            min_delay=4.0,
            max_delay=6.0,
            mean=5.0,
            std=3.0,
            seed=10101,
        )
        samples = [engine.calculate_base_jitter() for _ in range(2000)]
        self.assertTrue(all(4.0 <= s <= 6.0 for s in samples))

    def test_default_pacing_delay_bounds_30min_to_1hour(self) -> None:
        """Verify default JitterConfig is 30 minutes to 1 hour (1800s to 3600s)."""
        cfg = JitterConfig()
        self.assertEqual(cfg.min_delay, 1800.0)
        self.assertEqual(cfg.max_delay, 3600.0)
        self.assertEqual(cfg.mean_delay, 2700.0)
        self.assertEqual(cfg.std_dev, 300.0)

    def test_default_gaussian_jitter_intervals_within_1800s_3600s(self) -> None:
        """Default Truncated Gaussian: 2,000 samples strictly respect [1800s, 3600s]."""
        engine = JitterEngine(seed=42)
        samples = np.array([engine.calculate_base_jitter() for _ in range(2000)])
        self.assertTrue(all(1800.0 <= s <= 3600.0 for s in samples))
        self.assertAlmostEqual(float(np.mean(samples)), 2700.0, delta=30.0)
        self.assertAlmostEqual(float(np.std(samples)), 300.0, delta=30.0)

    def test_default_uniform_jitter_intervals_within_1800s_3600s(self) -> None:
        """Default Uniform: 2,000 samples strictly within [1800s, 3600s]."""
        engine = JitterEngine(distribution=JitterDistribution.UNIFORM, seed=99)
        samples = np.array([engine.calculate_base_jitter() for _ in range(2000)])
        self.assertTrue(all(1800.0 <= s <= 3600.0 for s in samples))
        self.assertAlmostEqual(float(np.mean(samples)), 2700.0, delta=45.0)

    def test_invalid_jitter_parameters_raise_error(self) -> None:
        """Invalid bounds or parameters must raise ValueError."""
        with self.assertRaises(ValueError):
            JitterConfig(min_delay=10.0, max_delay=5.0)  # min > max

        with self.assertRaises(ValueError):
            JitterConfig(min_delay=-1.0, max_delay=5.0)  # min < 0

        with self.assertRaises(ValueError):
            JitterConfig(min_delay=2.0, max_delay=8.0, mean=12.0)  # mean outside [2, 8]

        with self.assertRaises(ValueError):
            JitterConfig(min_delay=2.0, max_delay=8.0, std_dev=-0.5)  # std <= 0


class TestMicroBreaks(unittest.TestCase):
    """Test suite for human fatigue micro-breaks stochastic model."""

    def test_micro_break_trigger_cadence_window(self) -> None:
        """Verify micro-breaks trigger strictly within [min_interval, max_interval]."""
        engine = JitterEngine(
            micro_break_enabled=True,
            micro_break_freq_range=(15, 25),
            micro_break_duration_range=(10.0, 20.0),
            seed=42,
        )
        break_indices: List[int] = []
        for i in range(120):
            decision = engine.get_pacing_decision(increment_submission=True)
            if decision.is_micro_break:
                break_indices.append(i)

        self.assertGreaterEqual(len(break_indices), 4)
        # Check step intervals between micro-breaks
        last_idx = -1
        for idx in break_indices:
            interval = idx - last_idx
            self.assertGreaterEqual(interval, 15)
            self.assertLessEqual(interval, 26)
            last_idx = idx

    def test_micro_break_duration_within_bounds(self) -> None:
        """Micro-break durations fall strictly within [min_duration, max_duration]."""
        engine = JitterEngine(
            micro_break_enabled=True,
            micro_break_freq_range=(5, 5),
            micro_break_duration_range=(12.0, 18.0),
            seed=123,
        )
        durations: List[float] = []
        for _ in range(50):
            decision = engine.get_pacing_decision(increment_submission=True)
            if decision.is_micro_break:
                durations.append(decision.delay)

        self.assertGreaterEqual(len(durations), 8)
        self.assertTrue(all(12.0 <= d <= 18.0 for d in durations))

    def test_micro_break_counter_resets_post_break(self) -> None:
        """Internal submission counter resets to 0 immediately post-break."""
        engine = JitterEngine(
            micro_break_enabled=True,
            micro_break_freq_range=(10, 10),
            seed=456,
        )
        found_break = False
        for _ in range(25):
            decision = engine.get_pacing_decision(increment_submission=True)
            if decision.is_micro_break:
                self.assertEqual(decision.submission_count, 0)
                found_break = True
                break
        self.assertTrue(found_break)

    def test_micro_break_disabled_mode(self) -> None:
        """When micro_break_enabled=False, no micro-breaks ever trigger."""
        engine = JitterEngine(
            micro_break_enabled=False,
            micro_break_freq_range=(5, 10),
            seed=789,
        )
        for _ in range(60):
            decision = engine.get_pacing_decision(increment_submission=True)
            self.assertFalse(decision.is_micro_break)


class TestInterruptibleSleep(unittest.TestCase):
    """Test suite for interruptible pacing sleep."""

    def test_sleep_uninterrupted_completes_full_duration(self) -> None:
        """Sleep without stop event completes and returns True."""
        mock_clock = MockTimeProvider()
        engine = JitterEngine(sleep_fn=mock_clock.sleep)
        result = engine.sleep(5.0)
        self.assertTrue(result)
        self.assertEqual(mock_clock.sleep_calls, [5.0])

    def test_sleep_aborts_immediately_on_stop_event(self) -> None:
        """Sleep aborts immediately when stop_event is set."""
        engine = JitterEngine()
        stop_event = threading.Event()

        # Set stop event in secondary thread after 0.05s
        def trigger_stop():
            time.sleep(0.05)
            stop_event.set()

        t = threading.Thread(target=trigger_stop, daemon=True)
        t.start()

        start = time.perf_counter()
        result = engine.sleep(10.0, stop_event=stop_event, step=0.02)
        elapsed = time.perf_counter() - start

        self.assertFalse(result)
        self.assertLess(elapsed, 0.5)

    def test_sleep_already_stopped_returns_false_instantly(self) -> None:
        """If stop_event is already set, returns False in < 1ms."""
        engine = JitterEngine()
        stop_event = threading.Event()
        stop_event.set()

        start = time.perf_counter()
        result = engine.sleep(10.0, stop_event=stop_event)
        elapsed = time.perf_counter() - start

        self.assertFalse(result)
        self.assertLess(elapsed, 0.01)

    def test_sleep_pauses_and_resumes_cleanly(self) -> None:
        """Verify sleep pauses cleanly without busy-waiting and resumes when unpaused."""
        engine = JitterEngine()
        pause_event = threading.Event()
        pause_event.set()  # running
        stop_event = threading.Event()

        # Pause after 0.05s, resume after 0.15s
        def pause_flow():
            time.sleep(0.05)
            pause_event.clear()  # pause
            time.sleep(0.10)
            pause_event.set()    # resume

        t = threading.Thread(target=pause_flow, daemon=True)
        t.start()

        start = time.perf_counter()
        result = engine.sleep(0.12, stop_event=stop_event, pause_event=pause_event, step=0.02)
        elapsed = time.perf_counter() - start

        self.assertTrue(result)
        # Total elapsed should include the 0.10s paused time
        self.assertGreaterEqual(elapsed, 0.20)


class TestDiurnalActiveWindows(unittest.TestCase):
    """Test suite for diurnal operating windows."""

    def test_parse_single_and_multiple_window_ranges(self) -> None:
        """Parse comma-separated diurnal windows."""
        sched = BackgroundScheduler(
            active_windows="08:00-12:00, 13:30-17:30, 19:30-22:30"
        )
        self.assertEqual(len(sched._parsed_windows), 3)
        self.assertEqual(sched._parsed_windows[0].start_time, dt.time(8, 0))
        self.assertEqual(sched._parsed_windows[0].end_time, dt.time(12, 0))

    def test_invalid_window_format_raises_error(self) -> None:
        """Malformed window strings raise ValueError."""
        with self.assertRaises(ValueError):
            TimeWindow.from_string("25:00-12:00")
        with self.assertRaises(ValueError):
            TimeWindow.from_string("invalid-format")

    def test_active_window_membership_in_and_out(self) -> None:
        """Verify active membership across in-window and out-of-window timestamps."""
        clock = MockTimeProvider(start_dt=dt.datetime(2026, 11, 2, 9, 15, 0))
        sched = BackgroundScheduler(
            active_windows="08:00-12:00, 13:30-17:30",
            time_provider=clock,
        )

        # 09:15 -> inside morning window
        self.assertTrue(sched.is_in_active_window())

        # Advance to 12:45 -> lunch gap outside windows
        clock.advance(3.5 * 3600)  # 12:45
        self.assertFalse(sched.is_in_active_window())

        # Advance to 14:00 -> inside afternoon window
        clock.advance(1.25 * 3600)  # 14:00
        self.assertTrue(sched.is_in_active_window())

        # Advance to 20:00 -> evening outside windows
        clock.advance(6.0 * 3600)  # 20:00
        self.assertFalse(sched.is_in_active_window())

    def test_boundary_edge_conditions_start_and_end(self) -> None:
        """Start time is inclusive [start, end), end time is exclusive."""
        clock = MockTimeProvider(start_dt=dt.datetime(2026, 11, 2, 8, 0, 0))
        sched = BackgroundScheduler(active_windows="08:00-12:00", time_provider=clock)

        # Exactly at start: 08:00:00 -> True
        self.assertTrue(sched.is_in_active_window())

        # One second before end: 11:59:59 -> True
        clock.advance(3 * 3600 + 59 * 60 + 59)
        self.assertTrue(sched.is_in_active_window())

        # Exactly at end: 12:00:00 -> False
        clock.advance(1)
        self.assertFalse(sched.is_in_active_window())

    def test_seconds_until_next_window_calculation(self) -> None:
        """Calculate exact remaining seconds until next window opening."""
        # Current time: 12:30 -> next window opens at 13:30 (3,600s)
        clock = MockTimeProvider(start_dt=dt.datetime(2026, 11, 2, 12, 30, 0))
        sched = BackgroundScheduler(
            active_windows="08:00-12:00, 13:30-17:30",
            time_provider=clock,
        )
        self.assertAlmostEqual(sched.get_seconds_until_next_window(), 3600.0, delta=1.0)

        # Current time: 23:00 -> next window opens next morning at 08:00 (9 hours = 32,400s)
        clock.advance(10.5 * 3600)  # 23:00
        self.assertAlmostEqual(sched.get_seconds_until_next_window(), 32400.0, delta=1.0)

    def test_midnight_spanning_overnight_window(self) -> None:
        """Overnight window spanning midnight (e.g. 22:00-04:00)."""
        window = TimeWindow.from_string("22:00-04:00")
        self.assertTrue(window.contains(dt.time(22, 0)))
        self.assertTrue(window.contains(dt.time(23, 30)))
        self.assertTrue(window.contains(dt.time(1, 30)))
        self.assertTrue(window.contains(dt.time(3, 59)))
        self.assertFalse(window.contains(dt.time(4, 0)))
        self.assertFalse(window.contains(dt.time(12, 0)))


class TestStartDelayCountdown(unittest.TestCase):
    """Test suite for start delay and countdown management."""

    def test_start_delay_blocks_activation_before_expiry(self) -> None:
        """Start countdown blocks execution before expiry."""
        clock = MockTimeProvider()
        sched = BackgroundScheduler(start_delay_seconds=60.0, time_provider=clock)

        self.assertFalse(sched.is_start_countdown_elapsed())
        self.assertAlmostEqual(sched.get_start_countdown(), 60.0, delta=0.5)
        self.assertEqual(sched.get_state(), SchedulerState.COUNTDOWN)

        # Advance by 30s
        clock.advance(30.0)
        self.assertFalse(sched.is_start_countdown_elapsed())
        self.assertAlmostEqual(sched.get_start_countdown(), 30.0, delta=0.5)

        # Advance by 31s (total 61s)
        clock.advance(31.0)
        self.assertTrue(sched.is_start_countdown_elapsed())
        self.assertEqual(sched.get_start_countdown(), 0.0)
        self.assertEqual(sched.get_state(), SchedulerState.ACTIVE)

    def test_start_delay_zero_activates_immediately(self) -> None:
        """No start delay activates immediately."""
        clock = MockTimeProvider()
        sched = BackgroundScheduler(time_provider=clock)
        self.assertTrue(sched.is_start_countdown_elapsed())
        self.assertEqual(sched.get_state(), SchedulerState.ACTIVE)


class TestBatchQuotaAndCooldown(unittest.TestCase):
    """Test suite for batch quota and mandatory cooldown rest periods."""

    def test_batch_quota_triggers_mandatory_cooldown(self) -> None:
        """Record submissions up to batch_limit; triggers cooldown."""
        clock = MockTimeProvider()
        sched = BackgroundScheduler(
            batch_limit=5,
            cooldown_seconds=120.0,
            time_provider=clock,
        )

        for _ in range(4):
            triggered = sched.record_submission()
            self.assertFalse(triggered)
            self.assertFalse(sched.is_in_cooldown())
            self.assertEqual(sched.get_state(), SchedulerState.ACTIVE)

        # 5th submission hits batch_limit
        triggered = sched.record_submission()
        self.assertTrue(triggered)
        self.assertTrue(sched.is_in_cooldown())
        self.assertEqual(sched.get_state(), SchedulerState.COOLDOWN)
        self.assertAlmostEqual(sched.get_cooldown_remaining(), 120.0, delta=0.5)

    def test_cooldown_duration_blocks_submissions_and_resets(self) -> None:
        """Cooldown elapses when virtual clock advances past cooldown duration."""
        clock = MockTimeProvider()
        sched = BackgroundScheduler(
            batch_limit=3,
            cooldown_seconds=60.0,
            time_provider=clock,
        )

        for _ in range(3):
            sched.record_submission()

        self.assertTrue(sched.is_in_cooldown())
        can_run, state, _ = sched.can_submit()
        self.assertFalse(can_run)
        self.assertEqual(state, SchedulerState.COOLDOWN)

        # Advance clock by 30s -> still in cooldown
        clock.advance(30.0)
        self.assertTrue(sched.is_in_cooldown())
        self.assertAlmostEqual(sched.get_cooldown_remaining(), 30.0, delta=0.5)

        # Advance clock past 60s -> cooldown cleared
        clock.advance(35.0)
        self.assertFalse(sched.is_in_cooldown())
        self.assertEqual(sched.get_cooldown_remaining(), 0.0)
        self.assertEqual(sched.get_state(), SchedulerState.ACTIVE)


if __name__ == "__main__":
    unittest.main()
