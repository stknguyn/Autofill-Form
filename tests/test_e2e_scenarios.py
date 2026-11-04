"""
tests/test_e2e_scenarios.py - Real-World E2E Scenarios (Tier 4) Integration Test Suite.
Milestone M5: Unified E2E Test Suite & Adversarial Hardening.

Exercises complete real-world workloads testing the entire integrated stack:
- Scenario 1: Academic Survey Simulation (300 respondents synthesized by PersonaEngine,
              verified Cronbach's Alpha >= 0.82 across all 11 constructs, submitted via 10 workers,
              exported to SPSS CSV & 3-sheet Excel).
- Scenario 2: High-Concurrency Burst (20 racing worker threads hitting exact target quota with 0 overshoot,
              concurrent snapshot telemetry, and ticket rollback under transient errors).
- Scenario 3: Diurnal Scheduling with Overnight Window ("22:00-04:00") & Fatigue Micro-breaks
              under virtual clock control.
- Scenario 4: Streamlit Dashboard End-to-End User Flow (AppTest simulating parameter changes,
              lifecycle buttons, tab navigation, and dataset export).
"""

from __future__ import annotations

import datetime as dt
import io
import math
import random
import sys
import threading
import time
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

# Ensure repo root is in python path
# Robustly resolve repository root
_curr = Path(__file__).resolve().parent
while _curr != _curr.parent:
    if (_curr / "PROJECT.md").exists():
        break
    _curr = _curr.parent
REPO_ROOT = _curr
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.export_service import ExportService
from src.persona_engine import (
    CrossConsistencyValidator,
    PersonaEngine,
    RespondentProfile,
)
from src.scheduler import (
    BackgroundScheduler,
    JitterConfig,
    JitterDistribution,
    JitterEngine,
    PacingDecision,
    SchedulerConfig,
    SchedulerState,
    TimeWindow,
)
from src.stats_analyzer import ConstructStats, StatsAnalyzer, StatsSummary
from src.tally_client import (
    TallyClient,
    TallyRateLimitError,
    TallySubmissionError,
)
from src.tally_schema import TallySchema
from src.worker_manager import (
    FleetConfig,
    FleetState,
    WorkerCardSnapshot,
    WorkerFleetSnapshot,
    WorkerManager,
    WorkerState,
)


class MockTimeProvider:
    """Injectable virtual clock for deterministic temporal testing across diurnal window boundaries."""

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


class TestTier4Scenario1AcademicSurveySimulation(unittest.TestCase):
    """
    Tier 4 Scenario 1: Academic Survey Simulation Campaign.
    Simulates a full research campaign collecting 300 respondents:
    1. 300 respondents synthesized by PersonaEngine with 0 demographic cross-consistency violations.
    2. 10 concurrent workers submit payloads in Dry-Run mode to TallyClient simulator.
    3. Real-time streaming psychometrics verified: Cronbach's Alpha >= 0.82 across all 11 constructs,
       overall Alpha >= 0.90, and Perceived Costs inverted properly.
    4. Dataset exported to IBM SPSS pure numeric CSV and 3-sheet Excel workbook with Variable Codebook.
    """

    def test_academic_survey_campaign_300_respondents(self) -> None:
        """Executes full 300-respondent academic campaign and verifies data integrity end-to-end."""
        target_count = 300
        num_workers = 10

        # Fast jitter for automated testing (sub-second per iteration)
        fast_jitter = JitterConfig(
            min_delay=0.001,
            max_delay=0.005,
            micro_break_enabled=False,
        )

        cfg = FleetConfig(
            target_count=target_count,
            num_workers=num_workers,
            dry_run=True,
            delay_range=(0.001, 0.005),
            jitter_config=fast_jitter,
            master_seed=101,
        )

        wm = WorkerManager(config=cfg)
        t_start = time.perf_counter()

        try:
            wm.start()
            completed = wm.wait_until_complete(timeout=20.0)
            self.assertTrue(completed, "Worker fleet timed out before completing 300 respondents")

            # Final snapshot verification
            snap = wm.get_snapshot()
            self.assertEqual(snap.success_count, target_count, f"Expected {target_count} successes, got {snap.success_count}")
            self.assertEqual(snap.error_count, 0, f"Expected 0 errors in dry-run mode, got {snap.error_count}")
            self.assertEqual(snap.submitted_count, target_count)
            self.assertEqual(snap.total_target, target_count)
            self.assertTrue(snap.is_completed)
            self.assertEqual(snap.fleet_state, FleetState.COMPLETED.value)

            history = wm.get_history_df()
            self.assertEqual(len(history), target_count, f"Expected 300 records in history, got {len(history)}")

            # -------------------------------------------------------------------
            # Step 1: Verify Persona Engine & Cross-Consistency Rules
            # -------------------------------------------------------------------
            validator = CrossConsistencyValidator()
            for idx, rec in enumerate(wm._history_records):
                demographics = {
                    "age": rec["age"],
                    "gender_uuid": rec["gender"],
                    "location_uuid": rec["location"],
                    "education_uuid": rec["education"],
                    "employment_uuid": rec["employment"],
                    "profession_uuid": rec["profession"],
                    "frequency_uuid": rec["frequency"],
                    "categories_uuids": rec["categories"],
                    "pain_points_uuids": rec["pain_points"],
                    "awareness_uuid": rec["awareness"],
                }
                violations = validator.validate(demographics)
                self.assertEqual(
                    violations,
                    [],
                    f"Respondent #{idx} violated cross-consistency rules: {violations}",
                )
                self.assertTrue(18 <= rec["age"] <= 27, f"Age out of Gen Z range [18, 27]: {rec['age']}")

            # -------------------------------------------------------------------
            # Step 2: Verify Psychometrics & Cronbach's Alpha (>= 0.82)
            # -------------------------------------------------------------------
            stats_summary: StatsSummary = wm.stats_analyzer.get_summary()
            self.assertEqual(stats_summary.sample_count, target_count)
            self.assertTrue(stats_summary.is_reliable)

            # Overall Cronbach's Alpha verification (Standard requirement >= 0.82, target >= 0.90)
            self.assertGreaterEqual(
                stats_summary.overall_alpha,
                0.82,
                f"Overall Cronbach's Alpha {stats_summary.overall_alpha:.4f} failed standard threshold 0.820",
            )
            self.assertGreaterEqual(
                stats_summary.overall_alpha,
                0.90,
                f"Overall Cronbach's Alpha {stats_summary.overall_alpha:.4f} lower than expected research target 0.900",
            )

            # 11-Construct Individual Reliability Verification
            expected_constructs = ["PU", "PEOU", "PES", "PC", "PI", "SE", "LC", "ATT", "SN", "PBC", "ITU"]
            self.assertEqual(set(stats_summary.construct_stats.keys()), set(expected_constructs))

            for c_code in expected_constructs:
                c_stats = stats_summary.construct_stats[c_code]
                self.assertGreaterEqual(
                    c_stats.alpha,
                    0.820,
                    f"Construct {c_code} Alpha {c_stats.alpha:.4f} failed scientific requirement (>= 0.820)",
                )
                # Verify non-trivial variance
                self.assertGreater(c_stats.std, 0.2, f"Construct {c_code} variance is suspiciously low: std={c_stats.std}")

            # Verify Perceived Costs (PC) reverse coding
            pc_stats = stats_summary.construct_stats["PC"]
            self.assertTrue(pc_stats.is_reverse)
            self.assertIsNotNone(pc_stats.mean_raw)
            self.assertIsNotNone(pc_stats.mean_reversed)
            self.assertAlmostEqual(pc_stats.mean_raw + pc_stats.mean_reversed, 7.0, places=2)

            # -------------------------------------------------------------------
            # Step 3: Verify IBM SPSS Numeric CSV Export
            # -------------------------------------------------------------------
            csv_bytes = ExportService.export_spss_csv(history)
            self.assertIsInstance(csv_bytes, bytes)
            self.assertTrue(csv_bytes.startswith(b"\xef\xbb\xbf"), "SPSS CSV missing required UTF-8 BOM header")

            # Parse exported CSV into pandas
            df_csv = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig")
            self.assertEqual(df_csv.shape, (target_count, 73), f"Expected (300, 73) shape in CSV, got {df_csv.shape}")

            # Check demographic integer encodings
            self.assertTrue(df_csv["Gender"].isin([1, 2, 3]).all(), "Invalid Gender encoding in CSV")
            self.assertTrue(df_csv["Location"].isin([1, 2, 3, 4]).all(), "Invalid Location encoding in CSV")
            self.assertTrue(df_csv["Education"].isin([1, 2, 3, 4, 5]).all(), "Invalid Education encoding in CSV")
            self.assertTrue(df_csv["Employment"].isin([1, 2, 3, 4, 5, 6]).all(), "Invalid Employment encoding in CSV")

            # Check binary dummy columns
            dummy_cols = [
                "CAT_Fashion", "CAT_Beauty", "CAT_Tech", "CAT_Books", "CAT_Household", "CAT_Other",
                "PAIN_Building", "PAIN_Privacy", "PAIN_None", "PAIN_Other",
            ]
            for col in dummy_cols:
                self.assertIn(col, df_csv.columns)
                self.assertTrue(df_csv[col].isin([0, 1]).all(), f"Dummy column {col} contains non-binary values")

            # Check 39 Likert items are strictly integers in [1, 6]
            likert_cols = [c for c in df_csv.columns if any(c.startswith(prefix) for prefix in expected_constructs) and not c.endswith("_rev")]
            self.assertEqual(len(likert_cols), 39, f"Expected 39 Likert columns, found {len(likert_cols)}")
            for col in likert_cols:
                self.assertTrue(df_csv[col].between(1, 6).all(), f"Likert column {col} has out-of-bounds values")

            # Check PC reversed columns: PC1_rev = 7 - PC1
            for i in (1, 2, 3):
                raw_c = f"PC{i}"
                rev_c = f"PC{i}_rev"
                self.assertIn(rev_c, df_csv.columns)
                self.assertTrue((df_csv[raw_c] + df_csv[rev_c] == 7).all(), f"{raw_c} and {rev_c} do not sum to 7")
                corr = np.corrcoef(df_csv[raw_c], df_csv[rev_c])[0, 1]
                self.assertAlmostEqual(corr, -1.0, places=4, msg=f"Correlation between {raw_c} and {rev_c} is not -1.0")

            # -------------------------------------------------------------------
            # Step 4: Verify 3-Sheet Excel Export with Variable Codebook
            # -------------------------------------------------------------------
            excel_bytes = ExportService.export_excel(history, stats_summary=stats_summary)
            self.assertIsInstance(excel_bytes, bytes)
            self.assertTrue(excel_bytes.startswith(b"PK\x03\x04"), "Excel file missing Open Packaging Convention ZIP magic number")

            zf = zipfile.ZipFile(io.BytesIO(excel_bytes))
            self.assertIn("xl/workbook.xml", zf.namelist())
            wb_xml = zf.read("xl/workbook.xml")
            root = ET.fromstring(wb_xml)
            sheet_names = [elem.attrib["name"] for elem in root.iter() if elem.tag.endswith("sheet")]
            self.assertEqual(
                sheet_names,
                ["Data", "Variable Codebook", "Psychometric Summary"],
                f"Unexpected Excel sheet names: {sheet_names}",
            )

            # Verify Codebook completeness (73 variables)
            df_codebook = ExportService.get_variable_codebook()
            self.assertEqual(len(df_codebook), 73, f"Expected 73 variable definitions in Codebook, got {len(df_codebook)}")
            self.assertIn("Variable Name", df_codebook.columns)
            self.assertIn("Variable Label", df_codebook.columns)
            self.assertIn("Measurement Level", df_codebook.columns)

            # Verify Psychometric Summary completeness (11 constructs + OVERALL)
            df_psych_summary, _ = ExportService.get_psychometric_summary(stats_summary=stats_summary)
            self.assertGreaterEqual(len(df_psych_summary), 12, "Psychometric Summary missing constructs")
            self.assertIn("Construct Code", df_psych_summary.columns)
            self.assertIn("Cronbach's Alpha", df_psych_summary.columns)

        finally:
            wm.stop(timeout=5.0)

        elapsed = time.perf_counter() - t_start
        print(f"\n[Tier 4 Scenario 1] Finished 300 respondents campaign in {elapsed:.2f}s")


class TestTier4Scenario2HighConcurrencyBurst(unittest.TestCase):
    """
    Tier 4 Scenario 2: High-Concurrency Burst (20 Racing Worker Threads).
    Stresses the concurrency coordination layer under maximum load:
    1. 20 concurrent threads race to complete a quota without exceeding target (0 overshoot).
    2. Concurrent snapshot polling at 50Hz during high-contention ticket reservation.
    3. Fault tolerance and reservation rollback (_refund_work) under injected 500/429 failures.
    """

    def test_20_workers_racing_quota_exact_cutoff_zero_overshoot(self) -> None:
        """20 workers simultaneously racing must stop at exact target quota with 0 overshoot."""
        target_count = 100
        num_workers = 20

        cfg = FleetConfig(
            target_count=target_count,
            num_workers=num_workers,
            dry_run=True,
            delay_range=(0.0, 0.002),
            jitter_config=JitterConfig(min_delay=0.0, max_delay=0.002, micro_break_enabled=False),
            master_seed=202,
        )

        wm = WorkerManager(config=cfg)
        try:
            wm.start()
            completed = wm.wait_until_complete(timeout=10.0)
            self.assertTrue(completed, "20-worker burst timed out")

            snap = wm.get_snapshot()
            self.assertEqual(snap.success_count, target_count, "Success count does not match exact quota")
            self.assertEqual(snap.submitted_count, target_count, "Submitted count overshot target quota")
            self.assertEqual(snap.total_target, target_count)
            self.assertEqual(len(wm.get_history_df()), target_count, "History records overshot target quota")

            # Check that work was distributed across worker threads
            worker_contributions = [w.success_count for w in snap.workers]
            self.assertEqual(len(worker_contributions), 20)
            self.assertTrue(all(c > 0 for c in worker_contributions), f"Some workers starved: {worker_contributions}")
            self.assertEqual(sum(worker_contributions), target_count)

        finally:
            wm.stop(timeout=5.0)

    def test_concurrent_snapshot_reading_under_burst(self) -> None:
        """High-frequency snapshot queries from multiple reader threads during burst must not tear or deadlock."""
        target_count = 80
        num_workers = 20

        cfg = FleetConfig(
            target_count=target_count,
            num_workers=num_workers,
            dry_run=True,
            delay_range=(0.0, 0.003),
            jitter_config=JitterConfig(min_delay=0.0, max_delay=0.003, micro_break_enabled=False),
        )

        wm = WorkerManager(config=cfg)
        snapshots_captured: List[WorkerFleetSnapshot] = []
        stop_poller = threading.Event()
        poller_errors: List[Exception] = []

        def poller_loop():
            while not stop_poller.is_set():
                try:
                    s = wm.get_snapshot()
                    snapshots_captured.append(s)
                    time.sleep(0.005)  # 200Hz polling
                except Exception as exc:
                    poller_errors.append(exc)

        pollers = [threading.Thread(target=poller_loop, daemon=True) for _ in range(3)]

        try:
            for p in pollers:
                p.start()

            wm.start()
            completed = wm.wait_until_complete(timeout=10.0)
            self.assertTrue(completed)

        finally:
            stop_poller.set()
            for p in pollers:
                p.join(timeout=2.0)
            wm.stop(timeout=5.0)

        self.assertEqual(len(poller_errors), 0, f"Snapshot poller encountered errors: {poller_errors}")
        self.assertGreater(len(snapshots_captured), 20, "Expected at least 20 snapshots during burst")

        # Verify invariants across all captured snapshots
        last_succ = 0
        for s in snapshots_captured:
            self.assertIsInstance(s, WorkerFleetSnapshot)
            self.assertGreaterEqual(s.success_count, last_succ, "Snapshot success count went backward (tearing)")
            last_succ = s.success_count
            self.assertLessEqual(s.active_workers, 20)
            for w in s.workers:
                self.assertIsInstance(w.state, WorkerState)

    def test_burst_with_transient_failures_rollback_and_exact_quota(self) -> None:
        """Simulated transient server/rate errors must trigger reservation refunds and still hit exact quota."""
        target_count = 50
        num_workers = 20

        class TransientFaultyTallyClient(TallyClient):
            """Injects a 25% failure rate with submission errors and 429 rate limits."""
            def __init__(self):
                super().__init__(dry_run=True)
                self._rng = random.Random(42)
                self._lock = threading.Lock()

            def submit(self, payload: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
                with self._lock:
                    coin = self._rng.random()
                if coin < 0.15:
                    raise TallySubmissionError("Simulated upstream 503 Service Unavailable")
                elif coin < 0.25:
                    raise TallyRateLimitError("Simulated 429 Rate Limit", retry_after=0.01)
                return super().submit(payload, dry_run=True)

        cfg = FleetConfig(
            target_count=target_count,
            num_workers=num_workers,
            dry_run=True,
            delay_range=(0.0, 0.002),
            jitter_config=JitterConfig(min_delay=0.0, max_delay=0.002, micro_break_enabled=False),
        )

        faulty_client = TransientFaultyTallyClient()
        wm = WorkerManager(config=cfg, tally_client=faulty_client)

        try:
            wm.start()
            completed = wm.wait_until_complete(timeout=15.0)
            self.assertTrue(completed, "Burst with transient faults timed out")

            snap = wm.get_snapshot()
            self.assertGreater(snap.error_count, 0, "Expected transient errors to occur")
            self.assertEqual(snap.success_count, target_count, "Failed to reach exact quota despite refunds")
            self.assertEqual(len(wm.get_history_df()), target_count, "History record count exceeded quota")

        finally:
            wm.stop(timeout=5.0)


class TestTier4Scenario3DiurnalSchedulingAndMicroBreaks(unittest.TestCase):
    """
    Tier 4 Scenario 3: Diurnal Scheduling with Overnight Window & Fatigue Micro-breaks.
    Tests temporal automation across complex diurnal boundaries:
    1. Overnight active window ("22:00-04:00") controlled by an injected MockTimeProvider.
    2. Automatic halting outside window and instant resumption when window opens.
    3. Fatigue micro-breaks triggering at configured intervals with proper status labels.
    """

    def test_overnight_window_boundary_transitions_with_mock_clock(self) -> None:
        """Fleet holds outside window, activates upon boundary crossing, and freezes at closing."""
        # Start at 21:45 (outside 22:00-04:00 window)
        clock = MockTimeProvider(start_dt=dt.datetime(2026, 11, 2, 21, 45, 0))
        sched = BackgroundScheduler(
            active_windows="22:00-04:00",
            time_provider=clock,
        )

        cfg = FleetConfig(
            target_count=30,
            num_workers=4,
            dry_run=True,
            delay_range=(0.005, 0.01),
            jitter_config=JitterConfig(min_delay=0.005, max_delay=0.01, micro_break_enabled=False),
        )

        wm = WorkerManager(config=cfg, scheduler=sched)

        try:
            wm.start()

            # Phase 1: 21:45 -> Outside active window
            time.sleep(0.3)
            snap1 = wm.get_snapshot()
            self.assertEqual(snap1.success_count, 0, "Worker submitted while outside active window")
            self.assertEqual(sched.get_state(), SchedulerState.WINDOW_WAIT)
            self.assertAlmostEqual(sched.get_seconds_until_next_window(), 900.0, delta=2.0)

            # Phase 2: Advance clock by 20 minutes to 22:05 (inside window)
            clock.advance(20 * 60)
            time.sleep(0.4)
            snap2 = wm.get_snapshot()
            self.assertGreater(snap2.success_count, 0, "Workers failed to activate when window opened")
            self.assertEqual(sched.get_state(), SchedulerState.ACTIVE)

            # Phase 3: Let workers complete partial work, then advance clock past 04:00 (e.g. 04:15)
            # Reconfigure target to 100 to ensure quota is not yet finished
            wm._config.target_count = 100
            current_done = wm.get_snapshot().success_count

            # Advance clock across midnight to 04:15 next morning (6 hours 10 mins)
            clock.advance(6 * 3600 + 10 * 60)
            time.sleep(0.3)

            snap3 = wm.get_snapshot()
            frozen_count = snap3.success_count
            self.assertEqual(sched.get_state(), SchedulerState.WINDOW_WAIT)

            # Wait additional time; verify no further submissions leak out
            time.sleep(0.3)
            snap4 = wm.get_snapshot()
            self.assertEqual(snap4.success_count, frozen_count, "Worker submitted after window closed")

        finally:
            wm.stop(timeout=5.0)

    def test_fatigue_micro_breaks_cadence_and_status(self) -> None:
        """JitterEngine periodically inserts micro-breaks within [min_dur, max_dur] and sets status."""
        jitter_cfg = JitterConfig(
            distribution=JitterDistribution.UNIFORM,
            min_delay=0.5,
            max_delay=1.5,
            micro_break_enabled=True,
            micro_break_min_interval=5,
            micro_break_max_interval=8,
            micro_break_min_duration=10.0,
            micro_break_max_duration=20.0,
        )

        engine = JitterEngine(config=jitter_cfg, seed=777)
        decisions: List[PacingDecision] = []

        for _ in range(35):
            d = engine.get_pacing_decision(increment_submission=True)
            decisions.append(d)

        micro_breaks = [d for d in decisions if d.is_micro_break]
        self.assertGreaterEqual(len(micro_breaks), 3, "Expected at least 3 micro-breaks across 35 submissions")

        for b in micro_breaks:
            self.assertTrue(10.0 <= b.delay <= 20.0, f"Micro-break duration {b.delay} out of range [10, 20]")
            self.assertIn("Micro-break", b.get_status_label())

        # Check normal decisions
        normal_decisions = [d for d in decisions if not d.is_micro_break]
        for nd in normal_decisions:
            self.assertTrue(0.5 <= nd.delay <= 1.5, f"Normal delay {nd.delay} out of range [0.5, 1.5]")
            self.assertNotIn("Micro-break", nd.get_status_label())

    def test_batch_quota_and_cooldown_with_fleet(self) -> None:
        """Fleet pauses submissions upon hitting batch limit and resumes after cooldown."""
        clock = MockTimeProvider(start_dt=dt.datetime(2026, 11, 2, 10, 0, 0))
        sched = BackgroundScheduler(
            batch_limit=8,
            cooldown_seconds=60.0,
            time_provider=clock,
        )

        cfg = FleetConfig(
            target_count=16,
            num_workers=2,
            dry_run=True,
            delay_range=(0.005, 0.01),
            jitter_config=JitterConfig(min_delay=0.005, max_delay=0.01, micro_break_enabled=False),
        )

        wm = WorkerManager(config=cfg, scheduler=sched)

        try:
            wm.start()

            # Batch 1: Wait until 8 submissions are reached and cooldown triggers
            time.sleep(0.4)
            self.assertTrue(sched.is_in_cooldown(), "Scheduler should enter cooldown after 8 submissions")
            self.assertEqual(sched.get_state(), SchedulerState.COOLDOWN)
            self.assertEqual(wm.get_snapshot().success_count, 8)

            # Advance clock halfway through cooldown (30s): still in cooldown
            clock.advance(30.0)
            time.sleep(0.2)
            self.assertTrue(sched.is_in_cooldown())
            self.assertEqual(wm.get_snapshot().success_count, 8)

            # Advance clock past cooldown (35s, total 65s): cooldown cleared!
            clock.advance(35.0)
            self.assertFalse(sched.is_in_cooldown())

            # Batch 2: Fleet resumes and finishes the remaining 8 submissions (total 16)
            completed = wm.wait_until_complete(timeout=5.0)
            self.assertTrue(completed)
            self.assertEqual(wm.get_snapshot().success_count, 16)

        finally:
            wm.stop(timeout=5.0)


class TestTier4Scenario4StreamlitDashboardLifecycle(unittest.TestCase):
    """
    Tier 4 Scenario 4: Streamlit Dashboard End-to-End User Flow (AppTest).
    Tests the complete user journey through app.py headless simulation:
    1. Initial app mount and verification of UI elements.
    2. Dynamic parameter adjustments via sidebar widgets (target count, worker threads, dry-run toggle).
    3. Fleet lifecycle interactions: Start -> Pause -> Resume -> Stop -> Reset.
    4. Populated data telemetry rendering: Top 5 KPI cards, Worker Cards Grid, Plotly charts.
    5. Dataset download buttons for SPSS pure numeric CSV and 3-sheet Excel codebook.
    """

    def test_streamlit_dashboard_full_user_journey(self) -> None:
        """Executes full headless user workflow through Streamlit AppTest."""
        at = AppTest.from_file("app.py", default_timeout=25)
        at.run()
        self.assertFalse(at.exception, f"AppTest crashed on initial mount: {at.exception}")

        # -------------------------------------------------------------------
        # Step 1: Verify Initial Layout & Default Widgets
        # -------------------------------------------------------------------
        self.assertGreaterEqual(len(at.sidebar.button), 4, "Sidebar missing master action buttons")
        self.assertGreaterEqual(len(at.sidebar.number_input), 1, "Sidebar missing target count input")
        self.assertGreaterEqual(len(at.sidebar.slider), 1, "Sidebar missing worker threads slider")
        self.assertGreaterEqual(len(at.tabs), 4, "Dashboard missing 4 main navigation tabs")

        wm: WorkerManager = at.session_state["worker_manager"]
        self.assertEqual(wm.fleet_state, FleetState.IDLE)

        # -------------------------------------------------------------------
        # Step 2: Simulate User Parameter Adjustments
        # -------------------------------------------------------------------
        target_input = at.sidebar.number_input[0]
        target_input.set_value(250).run()
        self.assertFalse(at.exception)

        worker_slider = at.sidebar.slider[0]
        worker_slider.set_value(8).run()
        self.assertFalse(at.exception)

        dry_run_toggle = at.sidebar.toggle[0]
        dry_run_toggle.set_value(True).run()
        self.assertFalse(at.exception)

        # -------------------------------------------------------------------
        # Step 3: Lifecycle Button Interactions (Start, Pause, Resume, Stop)
        # -------------------------------------------------------------------
        btn_start = at.sidebar.button[0]
        btn_start.click().run()
        self.assertFalse(at.exception)
        self.assertTrue(wm.is_running, "Worker fleet should be running after Start click")

        # Click Pause
        btn_pause = at.sidebar.button[1]
        btn_pause.click().run()
        self.assertFalse(at.exception)
        self.assertTrue(wm.is_paused, "Worker fleet should be paused")

        # Click Resume
        btn_resume = at.sidebar.button[1]
        btn_resume.click().run()
        self.assertFalse(at.exception)
        self.assertFalse(wm.is_paused, "Worker fleet should be resumed")

        # Click Stop
        btn_stop = at.sidebar.button[2]
        btn_stop.click().run()
        self.assertFalse(at.exception)
        self.assertFalse(wm.is_running, "Worker fleet should be stopped")

        # -------------------------------------------------------------------
        # Step 4: Seed 35 Respondents & Verify Reactive Telemetry & Exports
        # -------------------------------------------------------------------
        engine = PersonaEngine(seed=42)
        for i in range(35):
            p = engine.generate_respondent()
            rec = {
                "submission_id": f"apptest-mock-{i:03d}",
                "timestamp": 1730000000.0 + i,
                "worker_id": i % 8,
                "latency_ms": 110.0 + (i % 20),
                "dry_run": True,
                "age": p.age,
                "gender": p.gender,
                "location": p.location,
                "education": p.education,
                "employment": p.employment,
                "profession": p.profession,
                "frequency": p.frequency,
                "categories": p.categories,
                "pain_points": p.pain_points,
                "awareness": p.awareness,
                "likert_scores": p.likert_scores,
            }
            with wm._history_lock:
                wm._history_records.append(rec)
            wm.stats_analyzer.add_sample(p.likert_scores)

        # Re-run app to re-render with populated data
        at.run()
        self.assertFalse(at.exception, f"AppTest crashed rendering populated data: {at.exception}")

        # Check export buttons
        download_btns = at.get("download_button")
        self.assertEqual(len(download_btns), 2, f"Expected 2 download buttons, found {len(download_btns)}")

        csv_btn = download_btns[0]
        self.assertIn("CSV", csv_btn.proto.label)
        self.assertTrue(csv_btn.proto.url.endswith(".csv"))

        excel_btn = download_btns[1]
        self.assertIn("Excel", excel_btn.proto.label)
        self.assertTrue(excel_btn.proto.url.endswith(".sheet") or excel_btn.proto.url.endswith(".xlsx"))

        # -------------------------------------------------------------------
        # Step 5: Test Reset Button Clears State
        # -------------------------------------------------------------------
        btn_reset = at.sidebar.button[3]
        btn_reset.click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(wm.get_history_df()), 0, "Reset button failed to clear history records")
        self.assertEqual(wm.fleet_state, FleetState.IDLE)


if __name__ == "__main__":
    unittest.main()
