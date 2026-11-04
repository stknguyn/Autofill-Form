"""
tests/test_scheduler_adversarial.py - Milestone M3 Challenger 2 Stress & Rigor Test Suite.

Adversarial Stress Harness:
1. Statistical Goodness-of-Fit:
   - 10,000 Uniform samples: empirical mean within (t_min + t_max)/2 +- 0.05,
     empirical variance within (t_max - t_min)^2/12 +- 0.05.
   - 10,000 Truncated Gaussian samples: 0 samples outside [t_min, t_max],
     no endpoint bunching / point mass spikes at boundaries.
2. Micro-Break Cadence Stress:
   - 500 simulated submissions across multiple seeds: micro-breaks trigger strictly
     every 15–25 submissions and last strictly 10–30s.
3. Diurnal Windows Stress:
   - Multi-window configuration with overnight wrap ('22:00-02:00, 06:00-09:00, 13:00-15:00').
   - High-density sampling (8,600+ points across 24h, exact boundary seconds).
4. Batch Cooldown Race:
   - 10 concurrent worker threads reaching batch quota simultaneously.
   - Verifies exactly 1 cooldown trigger occurs and all workers pause cleanly.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
import unittest
from typing import List, Tuple

import numpy as np

from src.scheduler import (
    BackgroundScheduler,
    JitterConfig,
    JitterDistribution,
    JitterEngine,
    SchedulerConfig,
    SchedulerState,
    TimeWindow,
)
from src.worker_manager import FleetConfig, WorkerManager, WorkerState


class TestAdversarialJitterStatisticalRigor(unittest.TestCase):
    """Adversarial stress testing for JitterEngine distribution statistical rigor."""

    def test_uniform_10000_samples_goodness_of_fit(self) -> None:
        """
        Adversarial Test 1A: 10,000 Uniform samples.
        Empirical mean must be within (t_min + t_max)/2 +- 0.05.
        Empirical variance must be within (t_max - t_min)^2/12 +- 0.05.
        """
        test_cases: List[Tuple[float, float, int]] = [
            (2.0, 8.0, 42),
            (1.0, 5.0, 12345),
            (3.0, 7.0, 99999),
        ]

        for t_min, t_max, seed in test_cases:
            engine = JitterEngine(
                distribution=JitterDistribution.UNIFORM,
                min_delay=t_min,
                max_delay=t_max,
                seed=seed,
            )
            samples = np.array([engine.calculate_base_jitter() for _ in range(10000)])

            theo_mean = (t_min + t_max) / 2.0
            theo_var = ((t_max - t_min) ** 2) / 12.0

            emp_mean = float(np.mean(samples))
            emp_var = float(np.var(samples))

            mean_diff = abs(emp_mean - theo_mean)
            var_diff = abs(emp_var - theo_var)

            self.assertTrue(
                np.all((samples >= t_min) & (samples <= t_max)),
                f"Samples outside [{t_min}, {t_max}] detected!",
            )
            self.assertLessEqual(
                mean_diff,
                0.05,
                f"Uniform mean deviation {mean_diff:.5f} exceeds +-0.05 for range [{t_min}, {t_max}]",
            )
            self.assertLessEqual(
                var_diff,
                0.05,
                f"Uniform variance deviation {var_diff:.5f} exceeds +-0.05 for range [{t_min}, {t_max}]",
            )

    def test_truncated_gaussian_10000_samples_bounds_and_no_bunching(self) -> None:
        """
        Adversarial Test 1B: 10,000 Truncated Gaussian samples.
        - Verify exactly zero samples outside [t_min, t_max].
        - Verify no endpoint bunching spikes (zero discrete point mass at exact boundaries).
        - Verify boundary bins have smooth tail frequencies and no mass accumulation.
        """
        param_sets = [
            {"min_delay": 2.0, "max_delay": 8.0, "mean": 5.0, "std": 1.0, "seed": 42},
            {"min_delay": 1.0, "max_delay": 9.0, "mean": 5.0, "std": 1.5, "seed": 101},
            {"min_delay": 3.0, "max_delay": 7.0, "mean": 5.0, "std": 0.8, "seed": 202},
            {"min_delay": 2.0, "max_delay": 8.0, "mean": 4.0, "std": 1.2, "seed": 303},  # Asymmetric mean
        ]

        for p in param_sets:
            t_min = p["min_delay"]
            t_max = p["max_delay"]
            engine = JitterEngine(
                distribution=JitterDistribution.TRUNCATED_GAUSSIAN,
                min_delay=t_min,
                max_delay=t_max,
                mean=p["mean"],
                std=p["std"],
                seed=p["seed"],
            )
            samples = np.array([engine.calculate_base_jitter() for _ in range(10000)])

            # 1. Zero samples strictly outside bounds
            outside_count = int(np.sum((samples < t_min) | (samples > t_max)))
            self.assertEqual(
                outside_count,
                0,
                f"Detected {outside_count} samples strictly outside [{t_min}, {t_max}]",
            )

            # 2. No endpoint bunching spikes (point masses at exact boundaries)
            exact_min_count = int(np.sum(samples == t_min))
            exact_max_count = int(np.sum(samples == t_max))
            self.assertEqual(
                exact_min_count,
                0,
                f"Detected point mass bunching at exact lower bound {t_min}: {exact_min_count} samples",
            )
            self.assertEqual(
                exact_max_count,
                0,
                f"Detected point mass bunching at exact upper bound {t_max}: {exact_max_count} samples",
            )

            # 3. Check boundary tail bins: boundary bin [t_min, t_min + delta] must NOT exceed interior bins
            # In a Gaussian centered at mu=5.0, boundary bin frequency must be small (smooth tail)
            delta = 0.2
            boundary_low_count = int(np.sum((samples >= t_min) & (samples < t_min + delta)))
            interior_count = int(np.sum((samples >= p["mean"] - delta / 2) & (samples <= p["mean"] + delta / 2)))
            self.assertLess(
                boundary_low_count,
                interior_count,
                f"Boundary low bin count {boundary_low_count} unexpectedly exceeds center {interior_count}",
            )


class TestAdversarialMicroBreakCadence(unittest.TestCase):
    """Adversarial stress testing for micro-break cadence and duration bounds."""

    def test_micro_break_cadence_500_submissions(self) -> None:
        """
        Adversarial Test 2: Run 500 simulated submissions.
        Verify micro-breaks trigger strictly every 15–25 submissions and last strictly 10–30s.
        """
        for seed in [42, 1337, 8888, 54321]:
            engine = JitterEngine(
                micro_break_enabled=True,
                micro_break_min_interval=15,
                micro_break_max_interval=25,
                micro_break_min_duration=10.0,
                micro_break_max_duration=30.0,
                seed=seed,
            )

            subs_since_last_break = 0
            break_intervals: List[int] = []
            break_durations: List[float] = []

            for _ in range(500):
                decision = engine.get_pacing_decision(increment_submission=True)
                if decision.is_micro_break:
                    break_intervals.append(subs_since_last_break)
                    break_durations.append(decision.delay)
                    subs_since_last_break = 0
                else:
                    subs_since_last_break += 1

            self.assertGreaterEqual(
                len(break_intervals),
                18,
                f"Expected at least 18 micro-breaks in 500 iterations, got {len(break_intervals)}",
            )

            # Check every single interval
            for idx, interval in enumerate(break_intervals):
                self.assertGreaterEqual(
                    interval,
                    15,
                    f"Micro-break #{idx} triggered too early after only {interval} submissions (min 15)",
                )
                self.assertLessEqual(
                    interval,
                    25,
                    f"Micro-break #{idx} triggered too late after {interval} submissions (max 25)",
                )

            # Check every single duration
            for idx, duration in enumerate(break_durations):
                self.assertGreaterEqual(
                    duration,
                    10.0,
                    f"Micro-break #{idx} duration {duration:.2f}s is below minimum 10.0s",
                )
                self.assertLessEqual(
                    duration,
                    30.0,
                    f"Micro-break #{idx} duration {duration:.2f}s exceeds maximum 30.0s",
                )


class TestAdversarialDiurnalWindows(unittest.TestCase):
    """Adversarial stress testing for multi-window diurnal configurations and overnight wrap."""

    def test_complex_multi_window_overnight_wrap_dense_timeline(self) -> None:
        """
        Adversarial Test 3: Complex multi-window configuration with overnight wrap:
        '22:00-02:00, 06:00-09:00, 13:00-15:00'.
        Densely tests 8,653 discrete timepoints across 24h, verifying transition boundaries
        and exact sleep duration calculations.
        """
        sched = BackgroundScheduler(active_windows="22:00-02:00, 06:00-09:00, 13:00-15:00")
        base_date = dt.date(2026, 11, 2)

        def expected_state_and_wait(t_sec: int) -> Tuple[bool, float]:
            # Timeline in seconds:
            # 00:00:00 to 01:59:59 (0 to 7199) -> W1 active (overnight)
            # 02:00:00 to 05:59:59 (7200 to 21599) -> inactive, next opens at 06:00:00 (21600)
            # 06:00:00 to 08:59:59 (21600 to 32399) -> W2 active (morning)
            # 09:00:00 to 12:59:59 (32400 to 46799) -> inactive, next opens at 13:00:00 (46800)
            # 13:00:00 to 14:59:59 (46800 to 53999) -> W3 active (afternoon)
            # 15:00:00 to 21:59:59 (54000 to 79199) -> inactive, next opens at 22:00:00 (79200)
            # 22:00:00 to 23:59:59 (79200 to 86399) -> W1 active (overnight)
            if (0 <= t_sec < 7200) or (79200 <= t_sec < 86400):
                return True, 0.0
            if 21600 <= t_sec < 32400:
                return True, 0.0
            if 46800 <= t_sec < 54000:
                return True, 0.0

            if 7200 <= t_sec < 21600:
                return False, float(21600 - t_sec)
            if 32400 <= t_sec < 46800:
                return False, float(46800 - t_sec)
            if 54000 <= t_sec < 79200:
                return False, float(79200 - t_sec)

            raise ValueError(f"Out of bounds second: {t_sec}")

        # Test points: every 10 seconds + 1-second boundary deltas around all transition edges
        transition_edges = [0, 7200, 21600, 32400, 46800, 54000, 79200, 86400]
        boundary_deltas = [-2, -1, 0, 1, 2]
        dense_points = set(range(0, 86400, 10))
        for edge in transition_edges:
            for delta in boundary_deltas:
                val = edge + delta
                if 0 <= val < 86400:
                    dense_points.add(val)

        sorted_points = sorted(dense_points)
        self.assertGreaterEqual(len(sorted_points), 8600)

        for s in sorted_points:
            h = s // 3600
            m = (s % 3600) // 60
            sec = s % 60
            curr_time = dt.time(h, m, sec)
            curr_dt = dt.datetime.combine(base_date, curr_time)

            exp_active, exp_wait = expected_state_and_wait(s)
            act_active = sched.is_in_active_window(now=curr_dt)
            act_wait = sched.get_seconds_until_next_window(now=curr_dt)

            self.assertEqual(
                act_active,
                exp_active,
                f"Active status failed at {curr_time}: expected {exp_active}, got {act_active}",
            )
            self.assertAlmostEqual(
                act_wait,
                exp_wait,
                delta=0.001,
                msg=f"Wait calculation failed at {curr_time}: expected {exp_wait}s, got {act_wait}s",
            )


class TestAdversarialBatchCooldownRace(unittest.TestCase):
    """Adversarial stress testing for multi-worker batch limit race conditions."""

    def test_10_workers_simultaneous_quota_race(self) -> None:
        """
        Adversarial Test 4A: 10 worker threads hit batch_limit simultaneously.
        Verify only ONE cooldown trigger occurs and all 10 pause cleanly.
        """
        sched = BackgroundScheduler(batch_limit=10, cooldown_seconds=2.0)
        barrier1 = threading.Barrier(10)
        barrier2 = threading.Barrier(10)

        triggers: List[bool] = []
        paused_cleanly: List[bool] = [False] * 10
        lock = threading.Lock()

        def worker_task(w_id: int):
            barrier1.wait()
            # Concurrent submission call
            is_triggered = sched.record_submission()
            with lock:
                triggers.append(is_triggered)

            # Wait until all 10 calls finish
            barrier2.wait()

            can_sub, state, msg = sched.can_submit()
            if not can_sub and state == SchedulerState.COOLDOWN:
                paused_cleanly[w_id] = True

        threads = [threading.Thread(target=worker_task, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 1. Exactly one thread must have triggered the cooldown
        trigger_count = sum(1 for t in triggers if t is True)
        self.assertEqual(
            trigger_count,
            1,
            f"Expected exactly 1 cooldown trigger, but got {trigger_count}!",
        )

        # 2. All 10 workers must cleanly see the cooldown state
        self.assertTrue(
            all(paused_cleanly),
            f"Not all workers paused cleanly: {paused_cleanly.count(True)}/10",
        )

        # 3. Scheduler state must be COOLDOWN with total_batches_completed = 1
        self.assertTrue(sched.is_in_cooldown())
        self.assertEqual(sched.get_state(), SchedulerState.COOLDOWN)
        self.assertEqual(sched._total_batches_completed, 1)

    def test_10_real_workers_fleet_batch_cooldown_integration(self) -> None:
        """
        Adversarial Test 4B: Live WorkerManager fleet with 10 real worker threads,
        configured with batch_limit=10, target_count=20, cooldown=0.3s.
        Verifies fleet pauses cleanly during cooldown and completes all 20 submissions cleanly.
        """
        cfg = FleetConfig(
            target_count=20,
            num_workers=10,
            delay_range=(0.01, 0.03),
            scheduler_config=SchedulerConfig(batch_limit=10, cooldown_seconds=0.25),
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        cooldown_seen = False
        t_start = time.time()
        while time.time() - t_start < 5.0:
            time.sleep(0.02)
            sched_snap = wm._scheduler.get_snapshot()
            if sched_snap.state == SchedulerState.COOLDOWN:
                cooldown_seen = True
            snap = wm.get_snapshot()
            if snap.submitted_count >= 20:
                break

        completed = wm.wait_until_complete(timeout=6.0)
        wm.stop()

        self.assertTrue(completed, "Fleet did not complete within timeout")
        self.assertTrue(cooldown_seen, "Fleet never entered COOLDOWN state despite reaching batch limit")

        snap = wm.get_snapshot()
        self.assertEqual(snap.success_count, 20)
        self.assertEqual(wm._scheduler._total_batches_completed, 2)


if __name__ == "__main__":
    unittest.main()
