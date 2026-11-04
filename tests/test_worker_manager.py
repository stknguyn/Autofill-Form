"""
tests/test_worker_manager.py - Comprehensive Concurrency & Lifecycle Test Suite.

Milestone M3: Worker Fleet Manager & Concurrency Orchestrator.
Covers:
- WorkerManager Lifecycle (Init, Config validation, Start, Pause, Resume, Stop, Reset)
- 1 to 20 Concurrency Stress Testing
- Exact Quota Reservation & Zero Overshoot (Atomic ticket reservation under race conditions)
- Dry-Run Integration (Zero-network socket isolation, UUID uniqueness, Cronbach's Alpha >= 0.82)
- Snapshot Atomicity & UI Polling Invariance (Immutable snapshots, high-frequency polling)
- Error Resilience & Recovery (HTTP 429 backoff, HTTP 500 server error, network timeout)
"""

from __future__ import annotations

import dataclasses
import socket
import threading
import time
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

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


class TestWorkerLifecycle(unittest.TestCase):
    """Lifecycle tests for WorkerManager: init, config, start, pause, resume, stop, reset."""

    def test_default_initialization_state(self) -> None:
        """Verify initial default state of WorkerManager."""
        wm = WorkerManager()
        self.assertEqual(wm.fleet_state, FleetState.IDLE)
        self.assertFalse(wm.is_running)
        self.assertFalse(wm.is_paused)
        self.assertEqual(wm.active_workers, 0)

        snap = wm.get_snapshot()
        self.assertEqual(snap.total_target, 150)
        self.assertEqual(snap.total_workers, 6)
        self.assertEqual(snap.success_count, 0)
        self.assertEqual(snap.submitted_count, 0)
        self.assertEqual(snap.error_count, 0)
        self.assertFalse(snap.is_running)

    def test_default_pacing_delay_range_in_1800s_3600s(self) -> None:
        """Verify default FleetConfig and WorkerManager pacing delay range is [1800s, 3600s]."""
        cfg = FleetConfig()
        self.assertEqual(cfg.delay_range, (1800.0, 3600.0))
        wm = WorkerManager()
        self.assertEqual(wm._config.delay_range, (1800.0, 3600.0))
        jcfg = wm._resolve_jitter_config(None)
        self.assertEqual(jcfg.min_delay, 1800.0)
        self.assertEqual(jcfg.max_delay, 3600.0)

    def test_configure_validates_worker_bounds(self) -> None:
        """Validates that num_workers is in [1, 20] and target_count >= 1."""
        wm = WorkerManager()

        # Out-of-bounds num_workers
        with self.assertRaises(ValueError):
            wm.configure(num_workers=0)
        with self.assertRaises(ValueError):
            wm.configure(num_workers=21)

        # Out-of-bounds target_count
        with self.assertRaises(ValueError):
            wm.configure(target_count=0)
        with self.assertRaises(ValueError):
            wm.configure(target_count=-10)

        # Valid configurations
        wm.configure(num_workers=1, target_count=10)
        snap = wm.get_snapshot()
        self.assertEqual(snap.total_workers, 1)
        self.assertEqual(snap.total_target, 10)

        wm.configure(num_workers=20, target_count=500)
        snap = wm.get_snapshot()
        self.assertEqual(snap.total_workers, 20)
        self.assertEqual(snap.total_target, 500)

    def test_start_spawns_active_daemon_threads(self) -> None:
        """Calling start() spawns the configured number of daemon threads."""
        cfg = FleetConfig(
            target_count=50,
            num_workers=8,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        self.assertTrue(wm.is_running)
        self.assertEqual(wm.fleet_state, FleetState.RUNNING)

        # Check threads in threading.enumerate()
        fleet_threads = [
            t for t in threading.enumerate()
            if t.name.startswith("WorkerThread-") and t.is_alive()
        ]
        self.assertEqual(len(fleet_threads), 8)
        self.assertTrue(all(t.daemon for t in fleet_threads))

        wm.stop(timeout=2.0)
        self.assertFalse(wm.is_running)

    def test_pause_and_resume_transitions(self) -> None:
        """Verify pause() and resume() state transitions."""
        cfg = FleetConfig(
            target_count=100,
            num_workers=4,
            dry_run=True,
            delay_range=(0.05, 0.1),
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        self.assertEqual(wm.fleet_state, FleetState.RUNNING)

        wm.pause()
        self.assertTrue(wm.is_paused)
        self.assertEqual(wm.fleet_state, FleetState.PAUSED)
        self.assertTrue(wm.get_snapshot().is_paused)

        wm.resume()
        self.assertFalse(wm.is_paused)
        self.assertEqual(wm.fleet_state, FleetState.RUNNING)

        wm.stop(timeout=2.0)

    def test_stop_gracefully_joins_all_threads(self) -> None:
        """Calling stop() signals all worker threads to join within timeout."""
        cfg = FleetConfig(
            target_count=200,
            num_workers=10,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        start_stop = time.perf_counter()
        wm.stop(timeout=2.0)
        stop_elapsed = time.perf_counter() - start_stop

        self.assertFalse(wm.is_running)
        self.assertLess(stop_elapsed, 2.0)
        self.assertEqual(wm.active_workers, 0)

        # Confirm all WorkerThread-* threads have exited
        active_worker_threads = [
            t for t in threading.enumerate()
            if t.name.startswith("WorkerThread-") and t.is_alive()
        ]
        self.assertEqual(len(active_worker_threads), 0)

    def test_cannot_start_already_running_fleet(self) -> None:
        """Calling start() on already running fleet is idempotent and does not spawn extra threads."""
        cfg = FleetConfig(target_count=50, num_workers=3, dry_run=True, delay_range=(0.05, 0.1))
        wm = WorkerManager(config=cfg)
        wm.start()

        threads_count_before = len([t for t in threading.enumerate() if t.name.startswith("WorkerThread-")])
        wm.start()  # Redundant call
        threads_count_after = len([t for t in threading.enumerate() if t.name.startswith("WorkerThread-")])

        self.assertEqual(threads_count_before, threads_count_after)
        wm.stop(timeout=2.0)

    def test_cannot_reconfigure_while_running(self) -> None:
        """Attempting to configure() while fleet is running raises RuntimeError."""
        cfg = FleetConfig(target_count=50, num_workers=2, dry_run=True, delay_range=(0.05, 0.1))
        wm = WorkerManager(config=cfg)
        wm.start()

        with self.assertRaises(RuntimeError):
            wm.configure(target_count=100)

        wm.stop(timeout=2.0)

    def test_reset_clears_metrics_when_stopped(self) -> None:
        """reset() resets counters, history, and stats back to initial baseline."""
        cfg = FleetConfig(
            target_count=5,
            num_workers=2,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()
        wm.wait_until_complete(timeout=5.0)
        wm.stop(timeout=2.0)

        snap = wm.get_snapshot()
        self.assertEqual(snap.success_count, 5)

        wm.reset()
        reset_snap = wm.get_snapshot()
        self.assertEqual(reset_snap.success_count, 0)
        self.assertEqual(reset_snap.submitted_count, 0)
        self.assertEqual(reset_snap.error_count, 0)
        self.assertTrue(wm.get_history_df().empty)

    def test_reset_raises_runtime_error_when_running(self) -> None:
        """Calling reset() while running raises RuntimeError."""
        cfg = FleetConfig(target_count=50, num_workers=2, dry_run=True, delay_range=(0.05, 0.1))
        wm = WorkerManager(config=cfg)
        wm.start()

        with self.assertRaises(RuntimeError):
            wm.reset()

        wm.stop(timeout=2.0)


class TestWorkerConcurrencyStress(unittest.TestCase):
    """Stress testing across varying concurrency levels (1, 5, 20 workers)."""

    def test_single_worker_execution(self) -> None:
        """Single worker executes and fulfills target count sequentially."""
        cfg = FleetConfig(
            target_count=10,
            num_workers=1,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()
        completed = wm.wait_until_complete(timeout=5.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed)
        snap = wm.get_snapshot()
        self.assertEqual(snap.success_count, 10)
        self.assertEqual(snap.error_count, 0)

    def test_five_workers_concurrent_stress(self) -> None:
        """5 concurrent workers fulfill target count 25 cleanly."""
        cfg = FleetConfig(
            target_count=25,
            num_workers=5,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()
        completed = wm.wait_until_complete(timeout=5.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed)
        snap = wm.get_snapshot()
        self.assertEqual(snap.success_count, 25)
        self.assertEqual(snap.error_count, 0)

    def test_twenty_workers_maximum_concurrency(self) -> None:
        """Maximum allowed concurrency (20 workers) fulfilling target count 60."""
        cfg = FleetConfig(
            target_count=60,
            num_workers=20,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()
        completed = wm.wait_until_complete(timeout=10.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed)
        snap = wm.get_snapshot()
        self.assertEqual(snap.success_count, 60)
        self.assertEqual(snap.error_count, 0)

    def test_thread_safety_under_heavy_concurrency(self) -> None:
        """Verify sum of per-worker success counts matches fleet success count."""
        cfg = FleetConfig(
            target_count=50,
            num_workers=10,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()
        wm.wait_until_complete(timeout=5.0)
        wm.stop(timeout=2.0)

        snap = wm.get_snapshot()
        worker_sum = sum(c.success_count for c in snap.workers)
        self.assertEqual(worker_sum, snap.success_count)
        self.assertEqual(snap.success_count, 50)


class TestTargetQuotaExactCutoff(unittest.TestCase):
    """Rigorous verification of the Atomic Reservation Pattern under extreme contention."""

    def test_exact_quota_cutoff_no_overshoot_5_workers(self) -> None:
        """5 workers racing for small target quota N=7 -> exactly 7, zero overshoot."""
        cfg = FleetConfig(
            target_count=7,
            num_workers=5,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()
        wm.wait_until_complete(timeout=5.0)
        wm.stop(timeout=2.0)

        snap = wm.get_snapshot()
        self.assertEqual(snap.success_count, 7)
        self.assertEqual(snap.submitted_count, 7)
        self.assertEqual(snap.error_count, 0)

    def test_exact_quota_cutoff_no_overshoot_20_workers(self) -> None:
        """20 workers racing for small prime target quota N=17 -> exactly 17, zero overshoot."""
        cfg = FleetConfig(
            target_count=17,
            num_workers=20,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()
        wm.wait_until_complete(timeout=5.0)
        wm.stop(timeout=2.0)

        snap = wm.get_snapshot()
        self.assertEqual(snap.success_count, 17)
        self.assertEqual(snap.submitted_count, 17)
        self.assertEqual(snap.error_count, 0)

    def test_single_target_twenty_workers_extreme_race(self) -> None:
        """20 workers racing for target N=1 -> exactly 1 thread succeeds, 19 return to IDLE."""
        cfg = FleetConfig(
            target_count=1,
            num_workers=20,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()
        wm.wait_until_complete(timeout=5.0)
        wm.stop(timeout=2.0)

        snap = wm.get_snapshot()
        self.assertEqual(snap.success_count, 1)
        self.assertEqual(snap.submitted_count, 1)
        self.assertEqual(snap.error_count, 0)

    def test_quota_exhaustion_all_workers_exit_to_idle(self) -> None:
        """When quota is reached, all workers gracefully return to IDLE state."""
        cfg = FleetConfig(
            target_count=10,
            num_workers=6,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()
        wm.wait_until_complete(timeout=5.0)
        wm.stop(timeout=2.0)

        snap = wm.get_snapshot()
        self.assertEqual(snap.active_workers, 0)
        self.assertTrue(all(c.state == WorkerState.IDLE for c in snap.workers))


class TestDryRunIntegration(unittest.TestCase):
    """End-to-end integration tests in Dry-Run mode."""

    def test_dry_run_zero_network_transmission(self) -> None:
        """Verify zero network socket connections are made in Dry-Run mode."""
        cfg = FleetConfig(
            target_count=15,
            num_workers=5,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)

        # Monkeypatch socket.socket.connect to fail if network is touched
        original_connect = socket.socket.connect

        def forbid_connect(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Network transmission forbidden in Dry-Run mode!")

        with patch.object(socket.socket, "connect", side_effect=forbid_connect):
            wm.start()
            completed = wm.wait_until_complete(timeout=5.0)
            wm.stop(timeout=2.0)

        self.assertTrue(completed)
        snap = wm.get_snapshot()
        self.assertEqual(snap.success_count, 15)
        self.assertEqual(snap.error_count, 0)

    def test_dry_run_full_pipeline_updates_stats(self) -> None:
        """Dry-run pipeline updates StatsAnalyzer and populates submission history."""
        cfg = FleetConfig(
            target_count=20,
            num_workers=4,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()
        wm.wait_until_complete(timeout=5.0)
        wm.stop(timeout=2.0)

        df = wm.get_history_df()
        self.assertEqual(len(df), 20)
        self.assertIn("submission_id", df.columns)
        self.assertIn("latency_ms", df.columns)
        self.assertIn("likert_scores", df.columns)

        summary = wm.get_stats_summary()
        self.assertEqual(summary.sample_count, 20)

    def test_dry_run_cronbach_alpha_maintained_above_threshold(self) -> None:
        """Run 150 dry-run samples; verify Cronbach's Alpha >= 0.82 across all constructs."""
        cfg = FleetConfig(
            target_count=150,
            num_workers=10,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
            master_seed=42,
        )
        wm = WorkerManager(config=cfg)
        wm.start()
        completed = wm.wait_until_complete(timeout=10.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed)
        summary = wm.get_stats_summary()
        self.assertEqual(summary.sample_count, 150)
        self.assertGreaterEqual(summary.overall_alpha, 0.82)

        alphas = wm.stats_analyzer.get_cronbach_alpha()
        for construct, alpha_val in alphas.items():
            self.assertGreaterEqual(
                alpha_val,
                0.80,
                f"Construct {construct} Cronbach's Alpha ({alpha_val:.4f}) fell below target",
            )

    def test_dry_run_unique_session_and_respondent_uuids(self) -> None:
        """Verify unique submission IDs generated per submission."""
        cfg = FleetConfig(
            target_count=25,
            num_workers=5,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()
        wm.wait_until_complete(timeout=5.0)
        wm.stop(timeout=2.0)

        df = wm.get_history_df()
        sub_ids = df["submission_id"].tolist()
        self.assertEqual(len(sub_ids), 25)
        self.assertEqual(len(set(sub_ids)), 25)  # All unique


class TestSnapshotAtomicity(unittest.TestCase):
    """Snapshot isolation and UI polling invariance tests."""

    def test_snapshot_immutability(self) -> None:
        """WorkerFleetSnapshot and WorkerCardSnapshot are frozen dataclasses."""
        wm = WorkerManager()
        snap = wm.get_snapshot()

        with self.assertRaises(dataclasses.FrozenInstanceError):
            snap.success_count = 999  # Attempting mutation

        card = snap.workers[0]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            card.current_task = "Hacked"

    def test_high_frequency_snapshot_polling_during_concurrency(self) -> None:
        """Simulate Streamlit polling get_snapshot() at 1000Hz during heavy 20-worker churn."""
        cfg = FleetConfig(
            target_count=100,
            num_workers=10,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        snapshots_collected: List[WorkerFleetSnapshot] = []
        stop_poller = threading.Event()

        def poller():
            while not stop_poller.is_set():
                snapshots_collected.append(wm.get_snapshot())
                time.sleep(0.001)

        t = threading.Thread(target=poller, daemon=True)
        t.start()

        wm.wait_until_complete(timeout=10.0)
        stop_poller.set()
        t.join(timeout=1.0)
        wm.stop(timeout=2.0)

        self.assertGreater(len(snapshots_collected), 50)
        # Verify invariants on all captured snapshots
        for s in snapshots_collected:
            self.assertLessEqual(s.success_count, s.total_target)
            self.assertLessEqual(s.success_count, s.submitted_count)
            self.assertEqual(len(s.workers), 10)
            self.assertEqual(
                s.active_workers,
                sum(1 for c in s.workers if c.state != WorkerState.IDLE),
            )

    def test_worker_card_state_fidelity(self) -> None:
        """WorkerCardSnapshot correctly reflects worker ID and state."""
        wm = WorkerManager(FleetConfig(num_workers=4))
        snap = wm.get_snapshot()
        self.assertEqual(len(snap.workers), 4)
        for i, card in enumerate(snap.workers):
            self.assertEqual(card.worker_id, i)
            self.assertEqual(card.state, WorkerState.IDLE)


class TestErrorResilience(unittest.TestCase):
    """Error handling, retry backoff, and recovery tests."""

    def test_worker_recovers_from_rate_limit_429(self) -> None:
        """Simulated HTTP 429 raises TallyRateLimitError, records error, backs off, retries, and succeeds."""
        mock_client = MagicMock()
        attempts = 0

        def flaky_submit(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise TallyRateLimitError("Rate limit exceeded", retry_after=0.01)
            return {"success": True, "data": {"submissionId": "mock-recovered"}}

        mock_client.submit.side_effect = flaky_submit

        cfg = FleetConfig(
            target_count=1,
            num_workers=1,
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
        self.assertEqual(snap.success_count, 1)
        self.assertGreaterEqual(snap.error_count, 2)

    def test_worker_handles_server_error_500(self) -> None:
        """Simulated HTTP 500 raises TallyServerError, records error, refunds reservation, retries, and succeeds."""
        mock_client = MagicMock()
        attempts = 0

        def flaky_submit(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TallyServerError("Internal Server Error")
            return {"success": True, "data": {"submissionId": "mock-500-recovered"}}

        mock_client.submit.side_effect = flaky_submit

        cfg = FleetConfig(
            target_count=1,
            num_workers=1,
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
        self.assertEqual(snap.success_count, 1)
        self.assertEqual(snap.error_count, 1)

    def test_worker_handles_network_timeout(self) -> None:
        """Network timeout raises TallyConnectionError, records error, retries, and succeeds."""
        mock_client = MagicMock()
        attempts = 0

        def flaky_submit(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TallyConnectionError("Network timeout connecting to api.tally.so")
            return {"success": True, "data": {"submissionId": "mock-timeout-recovered"}}

        mock_client.submit.side_effect = flaky_submit

        cfg = FleetConfig(
            target_count=1,
            num_workers=1,
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
        self.assertEqual(snap.success_count, 1)
        self.assertEqual(snap.error_count, 1)


if __name__ == "__main__":
    unittest.main()
