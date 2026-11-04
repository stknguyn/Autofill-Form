"""
tests/test_worker_fleet_stress_challenge.py - Empirical Adversarial Stress & Concurrency Suite.

Milestone M3 Challenger 1: Worker Fleet Concurrency & Stress.
Adversarial Verification Suite covering 5 mission objectives:
1. Extreme Quota Race Test: 20 workers racing for target=1 (single trial + 25 repeated stress trials + error-injected race).
2. High-Throughput Stress Test: 20 workers for target=50 (and target=100) asserting exactly target completed, 0 lost tickets, all return to IDLE.
3. Rapid Pause/Resume Flapping: 50 sequential and concurrent pause/resume toggles, asserting zero deadlocks and state consistency.
4. Immediate Stop Mid-Pacing: Long pacing delay (20s-50s) with 20 workers in PACING, asserting clean thread termination under 2.0s.
5. High-Frequency Snapshot Polling: 1000Hz single-poller and 4-poller concurrent hammering verifying atomicity, monotonicity, and invariants.
"""

from __future__ import annotations

import dataclasses
import threading
import time
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from src.scheduler import JitterConfig, JitterDistribution
from src.stats_analyzer import StatsAnalyzer
from src.tally_client import (
    TallyClient,
    TallyConnectionError,
    TallyRateLimitError,
    TallyServerError,
)
from src.worker_manager import (
    FleetConfig,
    FleetState,
    WorkerCardSnapshot,
    WorkerFleetSnapshot,
    WorkerManager,
    WorkerState,
)


class TestExtremeQuotaRaceChallenge(unittest.TestCase):
    """
    Task 1: Extreme Quota Race Test.
    Launch 20 workers racing for target=1 in Dry-Run mode.
    Assert strictly 1 success, 0 overshoot, and all threads return to IDLE cleanly.
    """

    def test_extreme_quota_race_target_1_single_trial(self) -> None:
        """Launch 20 workers racing for target=1: strictly 1 success, 0 overshoot."""
        cfg = FleetConfig(
            target_count=1,
            num_workers=20,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        completed = wm.wait_until_complete(timeout=5.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed, "Fleet failed to complete target=1 within 5.0s")
        snap = wm.get_snapshot()

        # Strict quota verification
        self.assertEqual(snap.success_count, 1, f"Expected strictly 1 success, got {snap.success_count}")
        self.assertEqual(snap.submitted_count, 1, f"Expected strictly 1 submission, got {snap.submitted_count}")
        self.assertEqual(snap.error_count, 0, f"Expected 0 errors, got {snap.error_count}")
        self.assertEqual(snap.total_target, 1)

        # Worker card distribution: exactly 1 worker got success=1, other 19 got success=0
        worker_successes = [c.success_count for c in snap.workers]
        self.assertEqual(sum(worker_successes), 1)
        self.assertEqual(worker_successes.count(1), 1)
        self.assertEqual(worker_successes.count(0), 19)

        # All workers must return to IDLE
        self.assertEqual(snap.active_workers, 0)
        self.assertTrue(all(c.state == WorkerState.IDLE for c in snap.workers))

    def test_extreme_quota_race_target_1_repeated_25_trials(self) -> None:
        """
        Adversarial repetition: Run 25 consecutive independent trials of 20 workers
        racing for target=1 to empirically prove zero probability of TOCTOU race conditions.
        """
        for trial in range(1, 26):
            cfg = FleetConfig(
                target_count=1,
                num_workers=20,
                dry_run=True,
                delay_range=(0.0, 0.001),
                jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
                master_seed=trial * 101,
            )
            wm = WorkerManager(config=cfg)
            wm.start()
            completed = wm.wait_until_complete(timeout=5.0)
            wm.stop(timeout=2.0)

            self.assertTrue(completed, f"Trial {trial}: Failed to complete within timeout")
            snap = wm.get_snapshot()

            self.assertEqual(
                snap.success_count,
                1,
                f"Trial {trial} OVERSHOOT DETECTED: success_count={snap.success_count} > 1",
            )
            self.assertEqual(
                snap.submitted_count,
                1,
                f"Trial {trial} OVERSHOOT DETECTED: submitted_count={snap.submitted_count} > 1",
            )
            self.assertEqual(snap.error_count, 0)
            self.assertEqual(sum(c.success_count for c in snap.workers), 1)
            self.assertEqual(snap.active_workers, 0)

    def test_extreme_quota_race_with_flaky_failures_and_refund(self) -> None:
        """
        Adversarial error injection: 20 workers racing for target=3, with 5 injected errors.
        Verifies that when a worker fails, its reserved ticket is refunded cleanly via _refund_work(),
        another worker picks up the ticket, and the fleet terminates with exactly 3 successes.
        """
        mock_client = MagicMock()
        error_count = 0
        lock = threading.Lock()

        def flaky_submit(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            nonlocal error_count
            with lock:
                if error_count < 5:
                    error_count += 1
                    raise TallyServerError("Simulated upstream 500 error during quota race")
            return {"success": True, "data": {"submissionId": f"mock-race-ok-{time.time()}"}}

        mock_client.submit.side_effect = flaky_submit

        cfg = FleetConfig(
            target_count=3,
            num_workers=20,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg, tally_client=mock_client)
        wm.start()
        completed = wm.wait_until_complete(timeout=5.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed)
        snap = wm.get_snapshot()
        self.assertEqual(snap.success_count, 3, "Quota reservation failed to fulfill target=3 after refunds")
        self.assertEqual(snap.error_count, 5, "Expected exactly 5 injected errors")
        self.assertEqual(snap.active_workers, 0)
        self.assertTrue(all(c.state == WorkerState.IDLE for c in snap.workers))


class TestHighThroughputStressChallenge(unittest.TestCase):
    """
    Task 2: High-Throughput Stress Test.
    Launch 20 workers for target=50 (and target=100).
    Assert exactly 50 submissions completed, 0 lost tickets, all workers return to IDLE.
    """

    def test_high_throughput_20_workers_target_50(self) -> None:
        """Launch 20 workers for target=50. Verify exact completion, 0 lost tickets, return to IDLE."""
        cfg = FleetConfig(
            target_count=50,
            num_workers=20,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
            master_seed=12345,
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        completed = wm.wait_until_complete(timeout=10.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed, "Fleet failed to complete target=50 within 10.0s")
        snap = wm.get_snapshot()

        # Strict count assertions
        self.assertEqual(snap.success_count, 50, f"Expected 50 successes, got {snap.success_count}")
        self.assertEqual(snap.submitted_count, 50, f"Expected 50 submissions, got {snap.submitted_count}")
        self.assertEqual(snap.error_count, 0, f"Expected 0 errors, got {snap.error_count}")

        # Zero lost tickets: sum of per-worker successes must exactly equal fleet success count
        worker_sum = sum(c.success_count for c in snap.workers)
        self.assertEqual(worker_sum, 50, f"Lost ticket detected: sum of worker cards is {worker_sum} != 50")

        # Verify all workers participated and finished in IDLE
        self.assertEqual(snap.active_workers, 0)
        for card in snap.workers:
            self.assertEqual(card.state, WorkerState.IDLE)
            self.assertEqual(card.error_count, 0)

        # In-memory history check: exactly 50 entries, all unique IDs
        df = wm.get_history_df()
        self.assertEqual(len(df), 50)
        self.assertEqual(len(set(df["submission_id"])), 50, "Duplicate submissionId found in history!")

        # Psychometric analyzer check: exactly 50 samples ingested
        stats_summary = wm.get_stats_summary()
        self.assertEqual(stats_summary.sample_count, 50)

    def test_high_throughput_20_workers_target_100(self) -> None:
        """Higher throughput: 20 workers for target=100. Zero overshoot, zero lost tickets."""
        cfg = FleetConfig(
            target_count=100,
            num_workers=20,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
            master_seed=999,
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        completed = wm.wait_until_complete(timeout=10.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed)
        snap = wm.get_snapshot()

        self.assertEqual(snap.success_count, 100)
        self.assertEqual(snap.submitted_count, 100)
        self.assertEqual(snap.error_count, 0)
        self.assertEqual(sum(c.success_count for c in snap.workers), 100)
        self.assertEqual(snap.active_workers, 0)


class TestRapidPauseResumeFlappingChallenge(unittest.TestCase):
    """
    Task 3: Rapid Pause/Resume Flapping.
    Toggle pause() and resume() 50 times in rapid succession.
    Assert zero deadlocks, thread state consistency, and full recovery.
    """

    def test_rapid_pause_resume_flapping_50_cycles_synchronous(self) -> None:
        """Toggle pause() and resume() 50 times rapidly while 20 workers are actively submitting."""
        cfg = FleetConfig(
            target_count=60,
            num_workers=20,
            dry_run=True,
            delay_range=(0.001, 0.005),
            jitter_config={"min_delay": 0.001, "max_delay": 0.005, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        # Flap pause() and resume() 50 times in rapid succession
        for _ in range(50):
            wm.pause()
            self.assertTrue(wm.is_paused)
            wm.resume()
            self.assertFalse(wm.is_paused)

        # Allow workers to complete target quota
        completed = wm.wait_until_complete(timeout=10.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed, "Fleet deadlocked or failed to finish after 50 pause/resume flaps")
        snap = wm.get_snapshot()
        self.assertEqual(snap.success_count, 60)
        self.assertEqual(snap.active_workers, 0)
        self.assertTrue(all(c.state == WorkerState.IDLE for c in snap.workers))

    def test_rapid_pause_resume_flapping_concurrent_flapper_thread(self) -> None:
        """
        Adversarial concurrent flapping: An asynchronous thread toggles pause() and resume()
        50 times with microsecond sleeps while 20 worker threads race for target=80.
        """
        cfg = FleetConfig(
            target_count=80,
            num_workers=20,
            dry_run=True,
            delay_range=(0.001, 0.005),
            jitter_config={"min_delay": 0.001, "max_delay": 0.005, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        flapper_errors: List[Exception] = []

        def flapper() -> None:
            try:
                for _ in range(50):
                    wm.pause()
                    time.sleep(0.002)
                    wm.resume()
                    time.sleep(0.002)
            except Exception as e:
                flapper_errors.append(e)

        flapper_thread = threading.Thread(target=flapper, name="FlapperThread", daemon=True)
        flapper_thread.start()

        # Join flapper thread first (should take ~0.2s)
        flapper_thread.join(timeout=5.0)
        self.assertFalse(flapper_thread.is_alive(), "Flapper thread deadlocked!")
        self.assertEqual(len(flapper_errors), 0, f"Flapper encountered errors: {flapper_errors}")

        # Ensure fleet is resumed
        wm.resume()

        # Fleet must complete cleanly without deadlocking
        completed = wm.wait_until_complete(timeout=10.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed, "Fleet failed to complete after concurrent pause/resume flapping")
        snap = wm.get_snapshot()
        self.assertEqual(snap.success_count, 80)
        self.assertEqual(snap.active_workers, 0)

    def test_pause_freezes_submissions_until_resume(self) -> None:
        """Verify pause truly halts new submissions during pause duration."""
        cfg = FleetConfig(
            target_count=50,
            num_workers=10,
            dry_run=True,
            delay_range=(0.01, 0.02),
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        # Let a few submissions occur
        time.sleep(0.05)
        wm.pause()
        self.assertTrue(wm.is_paused)

        # Record count after pause
        time.sleep(0.05)  # Allow in-flight submits to land
        snap_paused_1 = wm.get_snapshot()

        # Wait 0.3s while paused
        time.sleep(0.3)
        snap_paused_2 = wm.get_snapshot()

        # Verify no new submissions were initiated during pause
        self.assertEqual(
            snap_paused_1.submitted_count,
            snap_paused_2.submitted_count,
            "New submissions were initiated while fleet was paused!",
        )

        # Resume and complete
        wm.resume()
        completed = wm.wait_until_complete(timeout=10.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed)
        self.assertEqual(wm.get_snapshot().success_count, 50)


class TestImmediateStopMidPacingChallenge(unittest.TestCase):
    """
    Task 4: Immediate Stop Mid-Pacing.
    Call stop(timeout=2.0) while 20 workers are in long sleep.
    Assert all threads join cleanly within timeout (well under 2.0s).
    """

    def test_immediate_stop_mid_pacing_20_workers(self) -> None:
        """20 workers in long pacing sleep (20s to 50s) must abort and join in < 1.0s when stop() is called."""
        cfg = FleetConfig(
            target_count=100,
            num_workers=20,
            dry_run=True,
            delay_range=(20.0, 50.0),
            jitter_config=JitterConfig(
                distribution=JitterDistribution.UNIFORM,
                min_delay=20.0,
                max_delay=50.0,
                micro_break_enabled=False,
            ),
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        # Wait until workers transition into PACING state
        start_wait = time.time()
        pacing_reached = False
        while time.time() - start_wait < 5.0:
            snap = wm.get_snapshot()
            pacing_workers = sum(1 for c in snap.workers if c.state == WorkerState.PACING)
            if pacing_workers >= 10:  # At least half the fleet is sleeping
                pacing_reached = True
                break
            time.sleep(0.02)

        self.assertTrue(pacing_reached, "Workers failed to transition to PACING state within 5.0s")

        # Now measure stop execution latency
        t0 = time.perf_counter()
        wm.stop(timeout=2.0)
        stop_duration = time.perf_counter() - t0

        # Assert stop joined immediately (well under the 2.0s limit and much less than 20s pacing delay)
        self.assertLess(
            stop_duration,
            1.0,
            f"stop() took {stop_duration:.3f}s, exceeding 1.0s fast-join threshold mid-pacing",
        )

        # Assert all internal threads terminated
        self.assertFalse(wm.is_running)
        self.assertEqual(wm.active_workers, 0)
        self.assertEqual(len(wm._threads), 0)

        # Confirm no WorkerThread-* threads remain alive in process
        alive_fleet_threads = [
            t for t in threading.enumerate()
            if t.name.startswith("WorkerThread-") and t.is_alive()
        ]
        self.assertEqual(
            len(alive_fleet_threads),
            0,
            f"Zombie worker threads detected: {[t.name for t in alive_fleet_threads]}",
        )

        # Assert all worker card snapshots reflect IDLE state
        snap = wm.get_snapshot()
        self.assertTrue(all(c.state == WorkerState.IDLE for c in snap.workers))

    def test_immediate_stop_cleans_uncommitted_reservations(self) -> None:
        """When stop is called mid-pacing, reserved but unsubmitted tickets must be refunded."""
        cfg = FleetConfig(
            target_count=100,
            num_workers=20,
            dry_run=True,
            delay_range=(30.0, 60.0),
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        # Wait for workers to reserve slots and enter PACING
        time.sleep(0.3)
        wm.stop(timeout=2.0)

        snap = wm.get_snapshot()
        # All reserved slots that aborted during pacing should be refunded:
        # _reserved_count should match _success_count
        self.assertEqual(
            snap.reserved_count,
            snap.success_count,
            f"Reservation leak detected: reserved={snap.reserved_count} != success={snap.success_count}",
        )


class TestHighFrequencySnapshotPollingChallenge(unittest.TestCase):
    """
    Task 5: High-Frequency Snapshot Polling.
    Poll get_snapshot() at 1000Hz across 20 active workers and verify snapshot consistency.
    """

    def test_high_frequency_snapshot_polling_1000hz_single_poller(self) -> None:
        """Poll get_snapshot() at 1000Hz (1ms intervals) during heavy 20-worker submission load."""
        cfg = FleetConfig(
            target_count=100,
            num_workers=20,
            dry_run=True,
            delay_range=(0.001, 0.005),
            jitter_config={"min_delay": 0.001, "max_delay": 0.005, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        snapshots: List[WorkerFleetSnapshot] = []
        stop_poller = threading.Event()
        poller_exception: Optional[Exception] = None

        def poller() -> None:
            nonlocal poller_exception
            try:
                while not stop_poller.is_set():
                    snap = wm.get_snapshot()
                    snapshots.append(snap)
                    time.sleep(0.001)  # 1000Hz polling rate
            except Exception as exc:
                poller_exception = exc

        poller_thread = threading.Thread(target=poller, name="SnapshotPoller-1000Hz", daemon=True)
        poller_thread.start()

        # Wait for fleet to finish
        completed = wm.wait_until_complete(timeout=10.0)
        stop_poller.set()
        poller_thread.join(timeout=1.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed)
        self.assertIsNone(poller_exception, f"Poller encountered exception: {poller_exception}")
        self.assertGreaterEqual(
            len(snapshots),
            40,
            f"Expected at least 40 snapshots polled, got {len(snapshots)}",
        )

        # Invariant checks across ALL polled snapshots
        prev_success = 0
        for i, snap in enumerate(snapshots):
            # 1. Immutability check
            self.assertIsInstance(snap, WorkerFleetSnapshot)

            # 2. Quota boundary check: never exceeds total_target
            self.assertLessEqual(
                snap.success_count,
                snap.total_target,
                f"Snapshot {i}: success_count ({snap.success_count}) exceeded target ({snap.total_target})",
            )

            # 3. Submitted count invariant: successes never exceed reservations
            self.assertLessEqual(
                snap.success_count,
                snap.submitted_count,
                f"Snapshot {i}: success_count ({snap.success_count}) > submitted_count ({snap.submitted_count})",
            )

            # 4. Fleet card count invariant: always exactly 20 cards
            self.assertEqual(len(snap.workers), 20, f"Snapshot {i}: expected 20 worker cards, got {len(snap.workers)}")

            # 5. Active worker consistency: active_workers must match count of non-IDLE cards
            expected_active = sum(1 for c in snap.workers if c.state != WorkerState.IDLE)
            self.assertEqual(
                snap.active_workers,
                expected_active,
                f"Snapshot {i}: active_workers mismatch ({snap.active_workers} != {expected_active})",
            )

            # 6. Monotonicity: success count must be monotonically non-decreasing
            self.assertGreaterEqual(
                snap.success_count,
                prev_success,
                f"Snapshot {i}: monotonicity violation ({snap.success_count} < {prev_success})",
            )
            prev_success = snap.success_count

            # 7. Card state validity
            for card in snap.workers:
                self.assertIn(card.state, list(WorkerState))
                self.assertGreaterEqual(card.latency_ms, 0.0)

    def test_high_frequency_snapshot_polling_multi_poller_concurrency(self) -> None:
        """
        Adversarial multi-client polling: 4 concurrent threads hammering get_snapshot()
        simultaneously while 20 workers execute.
        """
        cfg = FleetConfig(
            target_count=50,
            num_workers=20,
            dry_run=True,
            delay_range=(0.0, 0.002),
            jitter_config={"min_delay": 0.0, "max_delay": 0.002, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        poll_counts = [0, 0, 0, 0]
        stop_signal = threading.Event()
        errors: List[Exception] = []

        def aggressive_poller(idx: int) -> None:
            try:
                while not stop_signal.is_set():
                    snap = wm.get_snapshot()
                    # Perform field access assertions
                    self.assertIsInstance(snap, WorkerFleetSnapshot)
                    self.assertLessEqual(snap.success_count, snap.total_target)
                    self.assertEqual(len(snap.workers), 20)
                    poll_counts[idx] += 1
                    time.sleep(0.001)  # 1000Hz per poller
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=aggressive_poller, args=(i,), daemon=True)
            for i in range(4)
        ]
        for t in threads:
            t.start()

        completed = wm.wait_until_complete(timeout=10.0)
        stop_signal.set()
        for t in threads:
            t.join(timeout=1.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed)
        self.assertEqual(len(errors), 0, f"Concurrent pollers encountered errors: {errors}")
        total_polls = sum(poll_counts)
        self.assertGreater(total_polls, 50, f"Multi-pollers completed {total_polls} reads")


class TestAdversarialEdgeCases(unittest.TestCase):
    """
    Extended Adversarial Edge Cases:
    - Stop while paused across 20 workers
    - Rapid start-stop cycling (thread thrashing)
    - Concurrent stats analysis queries under heavy write load
    - Mass HTTP 429 rate limit storm across all 20 workers
    """

    def test_stop_while_paused_20_workers(self) -> None:
        """Call stop() while 20 workers are in PAUSED state. All threads must join cleanly in < 1.0s."""
        cfg = FleetConfig(
            target_count=100,
            num_workers=20,
            dry_run=True,
            delay_range=(5.0, 10.0),
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        # Pause fleet immediately
        wm.pause()
        self.assertTrue(wm.is_paused)

        t0 = time.perf_counter()
        wm.stop(timeout=2.0)
        stop_duration = time.perf_counter() - t0

        self.assertLess(stop_duration, 1.0, f"stop() while paused took {stop_duration:.3f}s")
        self.assertFalse(wm.is_running)
        self.assertEqual(wm.active_workers, 0)
        alive_threads = [t for t in threading.enumerate() if t.name.startswith("WorkerThread-") and t.is_alive()]
        self.assertEqual(len(alive_threads), 0)

    def test_rapid_start_stop_cycling_10_rounds(self) -> None:
        """Rapid start() and immediate stop() cycling 10 consecutive times with 20 workers."""
        for round_idx in range(1, 11):
            cfg = FleetConfig(
                target_count=50,
                num_workers=20,
                dry_run=True,
                delay_range=(0.0, 0.001),
                jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
            )
            wm = WorkerManager(config=cfg)
            wm.start()
            # Immediate stop
            wm.stop(timeout=2.0)
            self.assertFalse(wm.is_running)
            self.assertEqual(wm.active_workers, 0)
            alive = [t for t in threading.enumerate() if t.name.startswith("WorkerThread-") and t.is_alive()]
            self.assertEqual(len(alive), 0, f"Round {round_idx}: leaked alive threads: {alive}")

    def test_concurrent_stats_analyzer_query_under_20_worker_load(self) -> None:
        """External reader queries stats_analyzer concurrently while 20 workers ingest samples."""
        cfg = FleetConfig(
            target_count=60,
            num_workers=20,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
            master_seed=42,
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        stop_reader = threading.Event()
        reader_errors: List[Exception] = []

        def stats_reader() -> None:
            try:
                while not stop_reader.is_set():
                    _ = wm.get_stats_summary()
                    _ = wm.stats_analyzer.get_cronbach_alpha()
                    time.sleep(0.002)
            except Exception as ex:
                reader_errors.append(ex)

        reader_t = threading.Thread(target=stats_reader, daemon=True)
        reader_t.start()

        completed = wm.wait_until_complete(timeout=10.0)
        stop_reader.set()
        reader_t.join(timeout=1.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed)
        self.assertEqual(len(reader_errors), 0, f"Stats reader encountered errors: {reader_errors}")
        summary = wm.get_stats_summary()
        self.assertEqual(summary.sample_count, 60)

    def test_mass_rate_limit_storm_across_20_workers(self) -> None:
        """All 20 workers hit HTTP 429 on their first attempt, back off, retry, and fulfill target=20."""
        mock_client = MagicMock()
        attempts_per_worker: Dict[str, int] = {}
        lock = threading.Lock()

        def rate_limited_submit(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            t_name = threading.current_thread().name
            with lock:
                count = attempts_per_worker.get(t_name, 0)
                attempts_per_worker[t_name] = count + 1
                if count == 0:
                    raise TallyRateLimitError("Rate limit storm", retry_after=0.01)
            return {"success": True, "data": {"submissionId": f"mock-storm-{t_name}"}}

        mock_client.submit.side_effect = rate_limited_submit

        cfg = FleetConfig(
            target_count=20,
            num_workers=20,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg, tally_client=mock_client)
        wm.start()
        completed = wm.wait_until_complete(timeout=10.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed)
        snap = wm.get_snapshot()
        self.assertEqual(snap.success_count, 20)
        self.assertGreaterEqual(snap.error_count, 1)
        self.assertEqual(snap.active_workers, 0)


if __name__ == "__main__":
    unittest.main()
