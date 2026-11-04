"""
tests/test_adversarial_m4.py - Empirical Adversarial Stress Test Suite for Milestone M4.
Adversarially challenges:
1. ui_components.py under extreme edge cases (None, N=0, N=1, zero variance, all 1s/6s, NaNs, missing demographics).
2. Plotly figure structural invariants (radar closed loops [1.0, 6.0], 95% CI error bars, 3.5 line, alpha thresholds).
3. app.py headless execution via Streamlit AppTest under varied session states and extreme parameters.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.export_service import ExportService
from src.persona_engine import (
    OPT_AWARE_USED_ONCE,
    OPT_CAT_BEAUTY,
    OPT_CAT_TECH,
    OPT_EDU_DAI_HOC,
    OPT_EMP_PART_TIME,
    OPT_FREQ_OVER_8,
    OPT_GENDER_KHAC,
    OPT_GENDER_NAM,
    OPT_GENDER_NU,
    OPT_LOC_HCM,
    OPT_PAIN_BUILDING,
    OPT_PROF_CNTT,
    PersonaEngine,
)
from src.scheduler import JitterConfig, JitterDistribution, SchedulerConfig, SchedulerState
from src.stats_analyzer import ConstructStats, ItemStats, StatsAnalyzer, StatsSummary
from src.tally_schema import TallySchema
from src.ui_components import (
    CONSTRUCT_METADATA,
    DEFAULT_BENCHMARKS,
    SLATE_THEME,
    create_accessible_construct_bar_chart,
    create_age_distribution_chart,
    create_construct_alpha_bars,
    create_cronbach_alpha_gauge,
    create_demographic_charts,
    create_employment_donut_chart,
    create_empty_state_figure,
    create_frequency_chart,
    create_gender_donut_chart,
    create_psychometric_radar_chart,
)
from src.worker_manager import FleetConfig, FleetState, WorkerCardSnapshot, WorkerFleetSnapshot, WorkerManager, WorkerState


class TestAdversarialFigureEdgeCases(unittest.TestCase):
    """Stress-tests Plotly figure generation under boundary conditions, corrupt data, and mathematical anomalies."""

    def test_01_none_and_empty_stats_summary(self):
        """Figures must survive None or empty StatsSummary gracefully."""
        # 1. None StatsSummary
        fig_radar_none = create_psychometric_radar_chart(None)
        self.assertIsInstance(fig_radar_none, go.Figure)
        self.assertEqual(len(fig_radar_none.data), 1, "None summary should show benchmark trace only")

        fig_bar_none = create_accessible_construct_bar_chart(None)
        self.assertIsInstance(fig_bar_none, go.Figure)
        self.assertEqual(len(fig_bar_none.data), 0, "None summary should yield empty state figure")
        self.assertTrue(len(fig_bar_none.layout.annotations) > 0)

        fig_alpha_bars_none = create_construct_alpha_bars(None)
        self.assertIsInstance(fig_alpha_bars_none, go.Figure)
        self.assertEqual(len(fig_alpha_bars_none.data), 0, "None summary should yield empty state figure")

        # 2. StatsSummary with sample_count = 0
        empty_sum = StatsSummary(
            sample_count=0,
            overall_alpha=0.0,
            status="INSUFFICIENT_DATA",
            is_reliable=False,
            construct_stats={},
        )
        fig_radar_zero = create_psychometric_radar_chart(empty_sum)
        self.assertEqual(len(fig_radar_zero.data), 1)

        fig_bar_zero = create_accessible_construct_bar_chart(empty_sum)
        self.assertEqual(len(fig_bar_zero.data), 0)

        fig_bars_zero = create_construct_alpha_bars(empty_sum)
        self.assertEqual(len(fig_bars_zero.data), 0)

    def test_02_sample_count_one_and_sem_singularity(self):
        """When N=1, SEM is mathematically undefined (division by 0). Error bars must be suppressed."""
        cs = {
            c: ConstructStats(
                code=c,
                name=c,
                item_count=3,
                is_reverse=(c == "PC"),
                mean=4.5,
                std=0.0,
                alpha=0.0,
                mean_raw=3.0 if c == "PC" else None,
                mean_reversed=4.0 if c == "PC" else None,
            )
            for c in DEFAULT_BENCHMARKS.keys()
        }
        sum_n1 = StatsSummary(
            sample_count=1,
            overall_alpha=0.0,
            status="INSUFFICIENT_DATA",
            is_reliable=False,
            construct_stats=cs,
        )

        fig_bar = create_accessible_construct_bar_chart(sum_n1)
        self.assertIsInstance(fig_bar, go.Figure)
        self.assertEqual(len(fig_bar.data), 2)
        bar_trace = fig_bar.data[0]
        self.assertFalse(bar_trace.error_x.visible, "When N=1, error bars must be invisible")

        # Radar chart should render observed polygon
        fig_radar = create_psychometric_radar_chart(sum_n1)
        self.assertEqual(len(fig_radar.data), 2)

    def test_03_zero_variance_extreme_all_ones(self):
        """Adversarial stress test: all respondents answer strictly 1 on all 39 questions (Zero Variance)."""
        analyzer = StatsAnalyzer()
        all_ones = {f"{c}{i}": 1 for c, items in ExportService.CONSTRUCT_INDICATORS.items() for i in range(1, len(items) + 1)}
        for _ in range(25):
            analyzer.add_sample(all_ones)

        sum_zero_var = analyzer.get_summary()
        self.assertEqual(sum_zero_var.sample_count, 25)
        self.assertIsInstance(sum_zero_var.overall_alpha, (int, float))

        # Check Radar Chart: PU..ITU = 1.0, but PC is reversed so PC_rev = 7 - 1 = 6.0
        fig_radar = create_psychometric_radar_chart(sum_zero_var)
        self.assertEqual(len(fig_radar.data), 2)
        sample_trace = fig_radar.data[1]
        self.assertEqual(len(sample_trace.r), 12)
        # Construct order starts with PU (1.0), PC is index 3 (6.0)
        self.assertAlmostEqual(sample_trace.r[0], 1.0, places=2)  # PU
        self.assertAlmostEqual(sample_trace.r[3], 6.0, places=2)  # PC reversed

        # Check Horizontal Bar Chart
        fig_bar = create_accessible_construct_bar_chart(sum_zero_var)
        self.assertEqual(len(fig_bar.data), 2)
        bar_trace = fig_bar.data[0]
        # In zero variance, SEM is 0.0
        for ci in bar_trace.error_x.array:
            self.assertAlmostEqual(ci, 0.0, places=4)

        # Check Gauge & Mini bars
        fig_gauge = create_cronbach_alpha_gauge(sum_zero_var.overall_alpha, sample_count=25)
        self.assertIsInstance(fig_gauge, go.Figure)

        fig_mini = create_construct_alpha_bars(sum_zero_var)
        self.assertIsInstance(fig_mini, go.Figure)

    def test_04_zero_variance_extreme_all_sixes(self):
        """Adversarial stress test: all respondents answer strictly 6 on all 39 questions (Zero Variance)."""
        analyzer = StatsAnalyzer()
        all_sixes = {f"{c}{i}": 6 for c, items in ExportService.CONSTRUCT_INDICATORS.items() for i in range(1, len(items) + 1)}
        for _ in range(25):
            analyzer.add_sample(all_sixes)

        sum_zero_var = analyzer.get_summary()
        # PU..ITU = 6.0, but PC is reversed: PC_rev = 7 - 6 = 1.0
        fig_radar = create_psychometric_radar_chart(sum_zero_var)
        sample_trace = fig_radar.data[1]
        self.assertAlmostEqual(sample_trace.r[0], 6.0, places=2)  # PU
        self.assertAlmostEqual(sample_trace.r[3], 1.0, places=2)  # PC reversed

        fig_bar = create_accessible_construct_bar_chart(sum_zero_var)
        self.assertEqual(len(fig_bar.data), 2)

    def test_05_nan_and_corrupt_alpha_values(self):
        """Alpha gauge and mini bars must not crash on NaN, negative, or infinite alpha values."""
        # Test negative alpha (due to severe negative item correlations)
        fig_neg = create_cronbach_alpha_gauge(-1.5, sample_count=20)
        self.assertIsInstance(fig_neg, go.Figure)
        self.assertEqual(fig_neg.data[0].value, 0.0, "Negative alpha must be clamped to 0.0")

        # Test alpha > 1.0
        fig_high = create_cronbach_alpha_gauge(1.25, sample_count=20)
        self.assertEqual(fig_high.data[0].value, 1.0, "Alpha > 1.0 must be clamped to 1.0")

        # Test NaN alpha in mini bars
        cs_nan = {
            c: ConstructStats(
                code=c,
                name=c,
                item_count=3,
                is_reverse=False,
                mean=4.0,
                std=0.5,
                alpha=float("nan"),
            )
            for c in DEFAULT_BENCHMARKS.keys()
        }
        sum_nan = StatsSummary(
            sample_count=15,
            overall_alpha=float("nan"),
            status="ZERO_VARIANCE",
            is_reliable=False,
            construct_stats=cs_nan,
        )
        fig_nan_bars = create_construct_alpha_bars(sum_nan)
        self.assertIsInstance(fig_nan_bars, go.Figure)
        self.assertEqual(len(fig_nan_bars.data), 1)

    def test_06_corrupt_demographics_fuzzing(self):
        """Demographic charts must withstand missing, null, extreme outliers, and fuzzed types."""
        fuzzed_df = pd.DataFrame([
            {"gender": None, "age": None, "employment": None, "frequency": None},
            {"gender": "UnknownAlien", "age": -99, "employment": "SpaceshipCaptain", "frequency": "millisecond"},
            {"gender": 999, "age": "NaN_string", "employment": 888, "frequency": 777},
            {"gender": "Nam", "age": 200, "employment": "Học sinh / Sinh viên", "frequency": "< 2 đơn/tháng"},
            {"gender": "Nữ", "age": 21.75, "employment": 2, "frequency": 4},
            {"gender": 1, "age": "22", "employment": 1, "frequency": 1},
        ])

        charts = create_demographic_charts(fuzzed_df)
        self.assertEqual(set(charts.keys()), {"gender", "age", "employment", "frequency"})
        for name, fig in charts.items():
            self.assertIsInstance(fig, go.Figure, f"Fuzzed DF crashed chart '{name}'")

        # Gender donut: check total count equals valid records mapped
        g_fig = charts["gender"]
        self.assertEqual(len(g_fig.data), 1)
        # All 5 non-null rows in gender should be mapped (Nam, Nữ, or Khác fallback)
        self.assertEqual(sum(g_fig.data[0].values), 5)

        # Age chart: only 22 (from "22") and 21.75 (in mean) are present
        age_fig = charts["age"]
        self.assertEqual(len(age_fig.data), 1)
        self.assertEqual(len(age_fig.data[0].x), 10)


class TestPlotlyStructuralInvariants(unittest.TestCase):
    """Verifies that generated Plotly figures strictly conform to architectural and UX specifications."""

    @classmethod
    def setUpClass(cls):
        cls.engine = PersonaEngine(seed=123)
        cls.analyzer = StatsAnalyzer()
        cls.records = []
        for _ in range(50):
            p = cls.engine.generate_respondent()
            cls.analyzer.add_sample(p.likert_scores)
            cls.records.append({
                "age": p.age,
                "gender": p.gender,
                "employment": p.employment,
                "frequency": p.frequency,
            })
        cls.summary = cls.analyzer.get_summary()
        cls.history_df = pd.DataFrame(cls.records)

    def test_01_radar_closed_loop_and_radial_bounds(self):
        """Radar chart must form a closed polygon with exactly 12 coordinates and radial range [1.0, 6.0]."""
        fig = create_psychometric_radar_chart(self.summary)

        # Traces: 0=Benchmark, 1=Observed
        self.assertEqual(len(fig.data), 2)
        bench = fig.data[0]
        obs = fig.data[1]

        # 12 coordinates (11 constructs + loopback)
        self.assertEqual(len(bench.theta), 12)
        self.assertEqual(len(bench.r), 12)
        self.assertEqual(bench.theta[0], bench.theta[-1], "Benchmark polygon theta loop not closed")
        self.assertEqual(bench.r[0], bench.r[-1], "Benchmark polygon r loop not closed")

        self.assertEqual(len(obs.theta), 12)
        self.assertEqual(len(obs.r), 12)
        self.assertEqual(obs.theta[0], obs.theta[-1], "Sample polygon theta loop not closed")
        self.assertEqual(obs.r[0], obs.r[-1], "Sample polygon r loop not closed")

        # Radial scale bounds [1.0, 6.0]
        radial = fig.layout.polar.radialaxis
        self.assertEqual(radial.range, (1.0, 6.0), "Radar radial range must be exactly [1.0, 6.0]")
        self.assertEqual(list(radial.tickvals), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # Reverse coding transparency: check customdata in sample trace for PC
        custom_data = obs.customdata
        self.assertIsNotNone(custom_data)
        # Construct index 3 is PC
        pc_hover = custom_data[3]
        self.assertIn("PC_rev", pc_hover)
        self.assertIn("PC_raw", pc_hover)

    def test_02_accessible_bar_chart_error_bars_and_midpoint(self):
        """Horizontal bar chart must feature 95% CI error bars and 3.5 midpoint line."""
        fig = create_accessible_construct_bar_chart(self.summary)
        self.assertEqual(len(fig.data), 2)

        bar_trace = fig.data[0]
        # Must have 11 constructs
        self.assertEqual(len(bar_trace.y), 11)
        self.assertEqual(len(bar_trace.x), 11)

        # 95% CI Error bars
        self.assertIsNotNone(bar_trace.error_x)
        self.assertTrue(bar_trace.error_x.visible)
        self.assertEqual(len(bar_trace.error_x.array), 11)
        for ci in bar_trace.error_x.array:
            self.assertGreater(ci, 0.0, "95% CI margin must be positive for N=50")

        # Midpoint reference line at 3.5
        vlines = [s for s in (fig.layout.shapes or []) if s.x0 == 3.5 and s.x1 == 3.5]
        self.assertEqual(len(vlines), 1, "Missing neutral 3.5 midpoint reference line")
        self.assertEqual(vlines[0].line.dash, "dash")

        # Benchmark diamonds
        diamond_trace = fig.data[1]
        self.assertEqual(len(diamond_trace.x), 11)
        self.assertEqual(diamond_trace.marker.symbol, "diamond")

        # autorange="reversed" to keep PU at top down to ITU
        self.assertEqual(fig.layout.yaxis.autorange, "reversed")

    def test_03_cronbach_alpha_gauges_and_color_thresholds(self):
        """Cronbach alpha gauges and mini bars must enforce 0.820 benchmark line and color thresholds."""
        # 1. Gauge indicator
        fig_gauge = create_cronbach_alpha_gauge(0.855, sample_count=50, target_threshold=0.82)
        ind = fig_gauge.data[0]
        self.assertEqual(ind.delta.reference, 0.82)
        self.assertEqual(ind.gauge.threshold.value, 0.82)

        steps = ind.gauge.steps
        self.assertEqual(len(steps), 3, "Gauge must have 3 qualitative reliability zones")
        self.assertEqual(list(steps[0]["range"]), [0.0, 0.70])   # Red
        self.assertEqual(list(steps[1]["range"]), [0.70, 0.82])  # Amber
        self.assertEqual(list(steps[2]["range"]), [0.82, 1.00])  # Green

        # 2. Status text: N >= 10 and alpha >= 0.82 -> ĐẠT CHUẨN NCKH
        self.assertIn("ĐẠT CHUẨN NCKH", fig_gauge.layout.annotations[0].text)

        # Status text: N < 10 -> ĐANG THU THẬP MẪU
        fig_gauge_small = create_cronbach_alpha_gauge(0.890, sample_count=5)
        self.assertIn("ĐANG THU THẬP MẪU", fig_gauge_small.layout.annotations[0].text)

        # 3. Mini horizontal bars color mapping
        # Create synthetic construct stats with known alpha values
        custom_cs = {
            "PU": ConstructStats("PU", "PU", 3, False, 4.5, 0.5, alpha=0.90),    # Green (>= 0.82)
            "PEOU": ConstructStats("PEOU", "PEOU", 3, False, 4.2, 0.5, alpha=0.75), # Amber (0.70 - 0.82)
            "PES": ConstructStats("PES", "PES", 3, False, 3.8, 0.5, alpha=0.60),  # Red (< 0.70)
        }
        for c in DEFAULT_BENCHMARKS.keys():
            if c not in custom_cs:
                custom_cs[c] = ConstructStats(c, c, 3, False, 4.0, 0.5, alpha=0.85)

        synth_summary = StatsSummary(
            sample_count=50,
            overall_alpha=0.85,
            status="RELIABLE",
            is_reliable=True,
            construct_stats=custom_cs,
        )

        fig_bars = create_construct_alpha_bars(synth_summary, target_threshold=0.82)
        bar_trace = fig_bars.data[0]
        colors = bar_trace.marker.color

        # PU (index 0) -> Green
        self.assertEqual(colors[0], SLATE_THEME["benchmark_green"])
        # PEOU (index 1) -> Amber
        self.assertEqual(colors[1], SLATE_THEME["warning_amber"])
        # PES (index 2) -> Red
        self.assertEqual(colors[2], SLATE_THEME["danger_rose"])

        # Reference line at 0.82
        vlines = [s for s in (fig_bars.layout.shapes or []) if s.x0 == 0.82 and s.x1 == 0.82]
        self.assertEqual(len(vlines), 1, "Missing 0.82 benchmark reference line")


class TestStreamlitAppTestHeadless(unittest.TestCase):
    """Stress-tests the full Streamlit app headless lifecycle via AppTest."""

    def test_01_app_initial_boot_clean(self):
        """App must boot and render with default state without any exception."""
        at = AppTest.from_file("app.py", default_timeout=15)
        at.run()
        self.assertFalse(at.exception, f"AppTest threw unhandled exception on boot: {at.exception}")

    def test_02_app_extreme_sidebar_inputs(self):
        """App must handle boundary parameter inputs (quota 1..2000, workers 1..20)."""
        at = AppTest.from_file("app.py", default_timeout=15)
        at.run()

        # Check sidebar widgets exist
        num_inputs = at.sidebar.number_input
        self.assertGreater(len(num_inputs), 0)

        # Set target count to maximum boundary (2000)
        target_input = num_inputs[0]
        target_input.set_value(2000).run()
        self.assertFalse(at.exception)

        # Set target count to minimum boundary (1)
        target_input.set_value(1).run()
        self.assertFalse(at.exception)

        # Check worker slider (1 to 20)
        sliders = at.sidebar.slider
        self.assertGreater(len(sliders), 0)
        worker_slider = sliders[0]
        worker_slider.set_value(20).run()
        self.assertFalse(at.exception)
        worker_slider.set_value(1).run()
        self.assertFalse(at.exception)

    def test_03_app_populated_history_and_export_buttons(self):
        """When history_df has data in session_state, export buttons and tabs must render cleanly."""
        at = AppTest.from_file("app.py", default_timeout=15)
        at.run()

        # Seed 30 respondents into worker_manager
        wm = at.session_state["worker_manager"]
        engine = PersonaEngine(seed=42)
        for i in range(30):
            p = engine.generate_respondent()
            rec = {
                "submission_id": f"mock-test-{i}",
                "timestamp": 1730000000.0 + i,
                "worker_id": i % 4,
                "latency_ms": 120.5,
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

        # Rerun app with seeded state
        at.run()
        self.assertFalse(at.exception, f"AppTest crashed with populated history: {at.exception}")

        # Verify download buttons exist for CSV and Excel via at.get
        download_btns = at.get("download_button")
        self.assertEqual(len(download_btns), 2, "Expected 2 download buttons (CSV and Excel)")
        labels = [b.label for b in download_btns]
        self.assertTrue(any("CSV" in lbl for lbl in labels))
        self.assertTrue(any("Excel" in lbl for lbl in labels))

    def test_04_app_fuzzed_history_in_session_state(self):
        """App must survive severely corrupted records in session_state without crashing."""
        at = AppTest.from_file("app.py", default_timeout=15)
        at.run()

        wm = at.session_state["worker_manager"]
        # Inject corrupted records (missing keys, non-numeric Likert, NaN demographics)
        corrupt_records = [
            {"submission_id": "bad-1", "timestamp": 1730000000.0, "worker_id": 0, "latency_ms": None, "dry_run": True},
            {"submission_id": "bad-2", "timestamp": 1730000001.0, "worker_id": 1, "latency_ms": 50.0, "dry_run": False, "age": "corrupted", "gender": None},
        ]
        with wm._history_lock:
            wm._history_records.extend(corrupt_records)

        at.run()
        self.assertFalse(at.exception, f"AppTest crashed with corrupted history records: {at.exception}")

    def test_05_app_sidebar_buttons_lifecycle(self):
        """Test Start, Pause, Resume, Stop, and Reset buttons via AppTest."""
        at = AppTest.from_file("app.py", default_timeout=15)
        at.run()

        # In default state, buttons: btn_start is enabled, pause/stop disabled
        buttons = at.sidebar.button
        self.assertGreaterEqual(len(buttons), 4)

        # Click reset button
        btn_reset = buttons[3]
        if not btn_reset.disabled:
            btn_reset.click().run()
            self.assertFalse(at.exception)

        # Click start button (Dry run mode)
        btn_start = at.sidebar.button[0]
        if not btn_start.disabled:
            btn_start.click().run()
            self.assertFalse(at.exception)
            wm = at.session_state["worker_manager"]
            self.assertTrue(wm.is_running)

            # Cleanly stop the fleet
            wm.stop(timeout=5.0)

    def test_06_app_dry_run_toggle(self):
        """Toggling Dry-Run mode should re-render without errors."""
        at = AppTest.from_file("app.py", default_timeout=15)
        at.run()

        # Find dry-run toggle
        toggles = at.sidebar.toggle
        self.assertGreater(len(toggles), 0)
        dry_run_toggle = toggles[0]
        self.assertTrue(dry_run_toggle.value)

        # Toggle to False (Live mode warning)
        dry_run_toggle.set_value(False).run()
        self.assertFalse(at.exception)

        # Toggle back to True
        dry_run_toggle.set_value(True).run()
        self.assertFalse(at.exception)

    def test_07_app_demographic_weight_sliders(self):
        """Extreme demographic sliders (0% and 100%) must not trigger exceptions."""
        at = AppTest.from_file("app.py", default_timeout=15)
        at.run()

        sliders = at.sidebar.slider
        # Sliders: workers (idx 0), delay (idx 1), break freq, break dur, age mean, age std, female pct, hcm pct
        for s in sliders:
            # Check slider min and max boundaries
            min_v = s.min
            max_v = s.max
            s.set_value(min_v).run()
            self.assertFalse(at.exception)
            s.set_value(max_v).run()
            self.assertFalse(at.exception)


class TestAdversarialExportService(unittest.TestCase):
    """Stress-tests ExportService with corrupted, out-of-bound, and missing data."""

    def test_01_likert_boundary_clamping_and_pc_inversion(self):
        """ExportService must strictly clamp all Likert scores to [1, 6] integers."""
        corrupt_rec = {
            "submission_id": "c1",
            "age": 22,
            "gender": "Nam",
            "likert_scores": {
                "PU1": -99,      # Underflow -> clamps to 1
                "PU2": 100,      # Overflow -> clamps to 6
                "PU3": 3.7,      # Float -> rounded to 4
                "PU4": "5",      # String int -> converted to 5
                "PC1": 1,        # Reverse check: 7 - 1 = 6
                "PC2": 6,        # Reverse check: 7 - 6 = 1
                "PC3": 3,        # Reverse check: 7 - 3 = 4
            },
        }
        df = ExportService.to_numeric_matrix([corrupt_rec])
        self.assertEqual(len(df), 1)
        self.assertEqual(df["PU1"].iloc[0], 1)
        self.assertEqual(df["PU2"].iloc[0], 6)
        self.assertEqual(df["PU3"].iloc[0], 4)
        self.assertEqual(df["PU4"].iloc[0], 5)

        # Verify exact PC reversal formula
        self.assertEqual(df["PC1"].iloc[0], 1)
        self.assertEqual(df["PC1_rev"].iloc[0], 6)
        self.assertEqual(df["PC2"].iloc[0], 6)
        self.assertEqual(df["PC2_rev"].iloc[0], 1)
        self.assertEqual(df["PC3"].iloc[0], 3)
        self.assertEqual(df["PC3_rev"].iloc[0], 4)

    def test_02_empty_records_export(self):
        """Empty records input must return clean DataFrame with all 73 columns."""
        df = ExportService.to_numeric_matrix([])
        self.assertEqual(len(df), 0)
        self.assertEqual(len(df.columns), 73)

        csv_bytes = ExportService.export_spss_csv([])
        self.assertTrue(len(csv_bytes) > 0)
        # Check UTF-8 BOM
        self.assertTrue(csv_bytes.startswith(b"\xef\xbb\xbf"))

        excel_bytes = ExportService.export_excel([])
        self.assertTrue(len(excel_bytes) > 0)
        # Check ZIP / XLSX magic number PK\x03\x04
        self.assertTrue(excel_bytes.startswith(b"PK\x03\x04"))


if __name__ == "__main__":
    unittest.main()

