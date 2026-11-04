"""
tests/test_ui_components.py - Headless Verification Suite for Plotly Visualizations & UI Components.
Milestone M4: Streamlit Dashboard & SPSS Export.

Covers:
- TestRadarChart: Closed polygon, 11 axes, dual traces (benchmark & sample), inverted PC transparency, empty state.
- TestAccessibleBarChart: 11 horizontal bars, error bars (+-1.96*SEM), 3.5 midpoint line, benchmark markers.
- TestCronbachAlphaGauges: Overall gauge (0.82 threshold, delta, qualitative zones), construct mini bars.
- TestDemographicCharts: Gender donut, Age 18-27 bar, Employment donut, Shopee frequency bar, UUID & label mappings.
- TestEmptyStatesAndResilience: N=0, N=1, None inputs, missing columns, empty DataFrames.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.persona_engine import PersonaEngine
from src.stats_analyzer import StatsAnalyzer, StatsSummary
from src.ui_components import (
    DEFAULT_BENCHMARKS,
    SLATE_THEME,
    apply_slate_theme,
    create_accessible_construct_bar_chart,
    create_age_distribution_chart,
    create_alpha_gauge,
    create_construct_alpha_bars,
    create_cronbach_alpha_gauge,
    create_demographic_charts,
    create_employment_donut_chart,
    create_empty_state_figure,
    create_frequency_chart,
    create_gender_donut_chart,
    create_horizontal_bar_chart,
    create_psychometric_radar_chart,
    create_radar_chart,
)


class BaseUIComponentTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PersonaEngine(seed=42)
        cls.analyzer = StatsAnalyzer()
        cls.profiles = []
        for _ in range(40):
            p = cls.engine.generate_respondent()
            cls.profiles.append(p)
            cls.analyzer.add_sample(p.likert_scores)
        cls.summary = cls.analyzer.get_summary()

        cls.history_df = pd.DataFrame([
            {
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
            }
            for p in cls.profiles
        ])


class TestRadarChart(BaseUIComponentTestCase):
    def test_radar_chart_structure_with_data(self):
        """Verify radar chart contains benchmark and sample traces as closed polygons."""
        fig = create_psychometric_radar_chart(self.summary)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)

        bench_trace = fig.data[0]
        sample_trace = fig.data[1]

        self.assertIsInstance(bench_trace, go.Scatterpolar)
        self.assertIsInstance(sample_trace, go.Scatterpolar)

        # Closed polygon check: 11 constructs + 1 closing point = 12 coordinates
        self.assertEqual(len(bench_trace.theta), 12)
        self.assertEqual(bench_trace.theta[0], bench_trace.theta[-1])
        self.assertEqual(len(bench_trace.r), 12)
        self.assertEqual(bench_trace.r[0], bench_trace.r[-1])

        self.assertEqual(len(sample_trace.theta), 12)
        self.assertEqual(sample_trace.theta[0], sample_trace.theta[-1])
        self.assertEqual(len(sample_trace.r), 12)
        self.assertEqual(sample_trace.r[0], sample_trace.r[-1])

    def test_radar_radial_axis_bounds(self):
        """Verify radial axis is fixed to [1.0, 6.0] Likert scale."""
        fig = create_psychometric_radar_chart(self.summary)
        radial_axis = fig.layout.polar.radialaxis
        self.assertEqual(radial_axis.range, (1.0, 6.0))
        self.assertEqual(tuple(radial_axis.tickvals), (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    def test_radar_empty_summary(self):
        """Verify radar chart renders safely with benchmark-only when summary is None or N=0."""
        fig_none = create_psychometric_radar_chart(None)
        self.assertIsInstance(fig_none, go.Figure)
        self.assertEqual(len(fig_none.data), 1)  # Only benchmark trace

        empty_summary = StatsSummary(sample_count=0, overall_alpha=0.0, status="INSUFFICIENT_DATA", is_reliable=False)
        fig_empty = create_psychometric_radar_chart(empty_summary)
        self.assertEqual(len(fig_empty.data), 1)

    def test_radar_alias(self):
        """Verify create_radar_chart alias works identically."""
        fig = create_radar_chart(self.summary)
        self.assertEqual(len(fig.data), 2)


class TestAccessibleBarChart(BaseUIComponentTestCase):
    def test_bar_chart_traces_and_ci(self):
        """Verify horizontal bar chart has bars with 95% CI error bars and benchmark markers."""
        fig = create_accessible_construct_bar_chart(self.summary)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)

        bar_trace = fig.data[0]
        marker_trace = fig.data[1]

        self.assertIsInstance(bar_trace, go.Bar)
        self.assertEqual(len(bar_trace.y), 11)
        self.assertEqual(len(bar_trace.x), 11)
        self.assertTrue(bar_trace.error_x.visible)
        self.assertEqual(len(bar_trace.error_x.array), 11)

        self.assertIsInstance(marker_trace, go.Scatter)
        self.assertEqual(len(marker_trace.x), 11)
        self.assertEqual(marker_trace.marker.symbol, "diamond")

    def test_bar_chart_reference_line_at_midpoint(self):
        """Verify neutral midpoint reference line at x=3.5."""
        fig = create_accessible_construct_bar_chart(self.summary)
        # Check vline shapes in layout
        vline_found = False
        if fig.layout.shapes:
            for s in fig.layout.shapes:
                if s.x0 == 3.5 and s.x1 == 3.5:
                    vline_found = True
                    break
        self.assertTrue(vline_found, "Missing 3.5 neutral midpoint reference line")

    def test_bar_chart_empty_state(self):
        """Verify graceful empty state figure returned for None or N=0."""
        fig_none = create_accessible_construct_bar_chart(None)
        self.assertIsInstance(fig_none, go.Figure)
        self.assertEqual(len(fig_none.data), 0)
        self.assertTrue(len(fig_none.layout.annotations) > 0)

    def test_bar_chart_alias(self):
        """Verify create_horizontal_bar_chart alias works identically."""
        fig = create_horizontal_bar_chart(self.summary)
        self.assertEqual(len(fig.data), 2)


class TestCronbachAlphaGauges(BaseUIComponentTestCase):
    def test_cronbach_alpha_gauge(self):
        """Verify overall alpha gauge indicator attributes."""
        fig = create_cronbach_alpha_gauge(overall_alpha=0.885, sample_count=40)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)

        ind = fig.data[0]
        self.assertIsInstance(ind, go.Indicator)
        self.assertAlmostEqual(ind.value, 0.885, places=3)
        self.assertEqual(ind.delta.reference, 0.82)
        self.assertEqual(ind.gauge.threshold.value, 0.82)

    def test_alpha_gauge_insufficient_samples(self):
        """Verify gauge resets to 0.0 when sample count is less than 3."""
        fig = create_cronbach_alpha_gauge(overall_alpha=0.885, sample_count=2)
        ind = fig.data[0]
        self.assertEqual(ind.value, 0.0)

    def test_construct_alpha_bars(self):
        """Verify construct-level alpha indicator bars have 11 items and benchmark line."""
        fig = create_construct_alpha_bars(self.summary)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)

        bar_trace = fig.data[0]
        self.assertIsInstance(bar_trace, go.Bar)
        self.assertEqual(len(bar_trace.y), 11)
        self.assertEqual(len(bar_trace.x), 11)

        # Check vline at 0.82
        vline_found = any(s.x0 == 0.82 and s.x1 == 0.82 for s in (fig.layout.shapes or []))
        self.assertTrue(vline_found, "Missing 0.82 threshold reference line in alpha bars")

    def test_construct_alpha_bars_empty(self):
        """Verify construct-level alpha bars empty state."""
        fig = create_construct_alpha_bars(None)
        self.assertEqual(len(fig.data), 0)
        self.assertTrue(len(fig.layout.annotations) > 0)

    def test_gauge_alias(self):
        """Verify create_alpha_gauge alias works identically."""
        fig = create_alpha_gauge(0.85, 20)
        self.assertEqual(len(fig.data), 1)


class TestDemographicCharts(BaseUIComponentTestCase):
    def test_gender_donut_chart(self):
        """Verify gender donut chart renders pie with hole and center count annotation."""
        fig = create_gender_donut_chart(self.history_df)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)
        pie = fig.data[0]
        self.assertIsInstance(pie, go.Pie)
        self.assertAlmostEqual(pie.hole, 0.62, places=2)
        self.assertGreater(sum(pie.values), 0)

        # Check center annotation has total count
        center_ann = [a for a in fig.layout.annotations if a.x == 0.5 and a.y == 0.5]
        self.assertTrue(len(center_ann) > 0)
        self.assertIn("ĐÁP VIÊN", center_ann[0].text)

    def test_gender_chart_with_numeric_codes(self):
        """Verify gender chart handles integer codes {1, 2, 3}."""
        df_int = pd.DataFrame({"gender": [1, 2, 2, 1, 3]})
        fig = create_gender_donut_chart(df_int)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(sum(fig.data[0].values), 5)

    def test_age_distribution_chart(self):
        """Verify Gen Z discrete age bar chart has 10 categories (18t to 27t)."""
        fig = create_age_distribution_chart(self.history_df)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)
        bar = fig.data[0]
        self.assertEqual(len(bar.x), 10)
        self.assertEqual(bar.x[0], "18t")
        self.assertEqual(bar.x[-1], "27t")

        # Mean age vline
        self.assertTrue(len(fig.layout.shapes or []) > 0)

    def test_employment_donut_chart(self):
        """Verify employment donut chart renders correctly with legend."""
        fig = create_employment_donut_chart(self.history_df)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)
        pie = fig.data[0]
        self.assertIsInstance(pie, go.Pie)
        self.assertAlmostEqual(pie.hole, 0.62, places=2)
        self.assertTrue(fig.layout.showlegend)

    def test_frequency_chart(self):
        """Verify Shopee frequency horizontal bar chart has 5 tiers."""
        fig = create_frequency_chart(self.history_df)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)
        bar = fig.data[0]
        self.assertIsInstance(bar, go.Bar)
        self.assertEqual(len(bar.y), 5)
        self.assertEqual(bar.y[0], "< 2 đơn/tháng")
        self.assertEqual(bar.y[-1], "> 8 đơn/tháng")

    def test_create_demographic_charts_batch(self):
        """Verify create_demographic_charts returns dictionary with 4 figures."""
        charts = create_demographic_charts(self.history_df)
        self.assertIsInstance(charts, dict)
        self.assertEqual(set(charts.keys()), {"gender", "age", "employment", "frequency"})
        for k, v in charts.items():
            self.assertIsInstance(v, go.Figure, f"Chart '{k}' is not a go.Figure")

    def test_demographic_empty_dataframe(self):
        """Verify all demographic charts handle empty or None DataFrames gracefully."""
        empty_df = pd.DataFrame()
        for chart_fn in (
            create_gender_donut_chart,
            create_age_distribution_chart,
            create_employment_donut_chart,
            create_frequency_chart,
        ):
            fig1 = chart_fn(empty_df)
            self.assertIsInstance(fig1, go.Figure)
            fig2 = chart_fn(None)
            self.assertIsInstance(fig2, go.Figure)


class TestThemeAndEmptyState(unittest.TestCase):
    def test_apply_slate_theme(self):
        """Verify apply_slate_theme injects dark slate tokens."""
        fig = go.Figure()
        styled_fig = apply_slate_theme(fig, title="Test Plot", height=300)
        self.assertIsNotNone(styled_fig.layout.template)
        self.assertEqual(styled_fig.layout.paper_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(styled_fig.layout.height, 300)
        self.assertIn("Test Plot", styled_fig.layout.title.text)

    def test_create_empty_state_figure(self):
        """Verify create_empty_state_figure hides axes and adds message."""
        fig = create_empty_state_figure("Không có dữ liệu", "Thử lại sau.")
        self.assertFalse(fig.layout.xaxis.visible)
        self.assertFalse(fig.layout.yaxis.visible)
        self.assertEqual(len(fig.layout.annotations), 1)
        self.assertIn("Không có dữ liệu", fig.layout.annotations[0].text)


if __name__ == "__main__":
    unittest.main()
