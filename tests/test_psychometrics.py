"""
tests/test_psychometrics.py
Comprehensive Psychometric Verification Suite for Milestone M2.
Validates Cronbach's Alpha >= 0.82 across all 11 constructs, PC reverse scoring,
and mathematical parity between streaming Welford and batch covariance.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.persona_engine import PersonaEngine
from src.stats_analyzer import StatsAnalyzer, StreamingCovarianceTracker


class TestPsychometricsReliability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PersonaEngine()
        cls.analyzer = StatsAnalyzer()
        # Generate 150 independent respondent profiles
        cls.num_samples = 150
        cls.profiles = [cls.engine.generate_respondent() for _ in range(cls.num_samples)]

        # Populate analyzer
        for p in cls.profiles:
            cls.analyzer.add_sample(p.likert_scores)

    def test_constructs_alpha_above_threshold(self):
        """Verify EVERY single one of the 11 constructs has Cronbach's Alpha >= 0.82."""
        alphas = self.analyzer.get_cronbach_alpha()
        failed_constructs: Dict[str, float] = {}
        for construct in ["PU", "PEOU", "PES", "PC", "PI", "SE", "LC", "ATT", "SN", "PBC", "ITU"]:
            val = alphas.get(construct, 0.0)
            if val < 0.82:
                failed_constructs[construct] = val

        self.assertEqual(
            len(failed_constructs),
            0,
            f"The following constructs failed alpha >= 0.82 threshold: {failed_constructs}",
        )

    def test_overall_model_alpha(self):
        """Verify full 39-item scale (with PC inverted) has alpha >= 0.90."""
        alphas = self.analyzer.get_cronbach_alpha()
        overall = alphas.get("overall", 0.0)
        self.assertGreaterEqual(
            overall,
            0.90,
            f"Overall model alpha {overall:.4f} is below 0.90 threshold",
        )

    def test_pc_reversal_necessity(self):
        """Demonstrate that NOT reversing PC results in degraded overall scale alpha."""
        raw_matrix = np.zeros((self.num_samples, 39), dtype=np.float64)
        rev_matrix = np.zeros((self.num_samples, 39), dtype=np.float64)

        for i, p in enumerate(self.profiles):
            for code, idx in self.analyzer._code_to_idx.items():
                val = float(p.likert_scores[code])
                raw_matrix[i, idx] = val
                rev_matrix[i, idx] = (7.0 - val) if code.startswith("PC") else val

        def calc_batch_alpha(mat: np.ndarray) -> float:
            cov = np.cov(mat, rowvar=False, ddof=1)
            k = mat.shape[1]
            return float((k / (k - 1)) * (1.0 - np.trace(cov) / np.sum(cov)))

        alpha_unreversed = calc_batch_alpha(raw_matrix)
        alpha_reversed = calc_batch_alpha(rev_matrix)

        self.assertLess(
            alpha_unreversed,
            alpha_reversed,
            f"Expected alpha_unreversed ({alpha_unreversed:.4f}) < alpha_reversed ({alpha_reversed:.4f})",
        )

    def test_pc_standalone_alpha_identity(self):
        """Verify Cronbach's alpha of PC construct alone is identical whether raw or reversed."""
        raw_pc = np.array(
            [[p.likert_scores[f"PC{j}"] for j in (1, 2, 3)] for p in self.profiles],
            dtype=np.float64,
        )
        rev_pc = 7.0 - raw_pc

        def calc_alpha(mat: np.ndarray) -> float:
            cov = np.cov(mat, rowvar=False, ddof=1)
            k = mat.shape[1]
            return float((k / (k - 1)) * (1.0 - np.trace(cov) / np.sum(cov)))

        alpha_raw = calc_alpha(raw_pc)
        alpha_rev = calc_alpha(rev_pc)

        self.assertAlmostEqual(alpha_raw, alpha_rev, places=10)
        self.assertGreaterEqual(alpha_raw, 0.82)

    def test_streaming_vs_batch_parity(self):
        """Verify Welford streaming tracker produces exact same results as batch np.cov."""
        matrix = np.zeros((self.num_samples, 39), dtype=np.float64)
        for i, p in enumerate(self.profiles):
            for code, idx in self.analyzer._code_to_idx.items():
                val = float(p.likert_scores[code])
                matrix[i, idx] = (7.0 - val) if code.startswith("PC") else val

        # Batch covariance and alpha
        batch_cov = np.cov(matrix, rowvar=False, ddof=1)
        k = 39
        batch_alpha = float((k / (k - 1)) * (1.0 - np.trace(batch_cov) / np.sum(batch_cov)))

        # Streaming from analyzer
        streaming_cov = self.analyzer._global_tracker.covariance_matrix()
        streaming_alpha, _ = self.analyzer._global_tracker.compute_alpha()

        max_cov_diff = float(np.abs(batch_cov - streaming_cov).max())
        alpha_diff = abs(batch_alpha - streaming_alpha)

        self.assertLess(max_cov_diff, 1e-12, f"Covariance divergence: {max_cov_diff}")
        self.assertLess(alpha_diff, 1e-12, f"Alpha divergence: {alpha_diff}")

    def test_corrected_item_total_correlation(self):
        """Verify all 39 Likert items have Corrected Item-Total Correlation >= 0.30."""
        summary = self.analyzer.get_summary()
        failing_items: Dict[str, float] = {}
        for c_code, c_stat in summary.construct_stats.items():
            for i_code, i_stat in c_stat.item_stats.items():
                if i_stat.citc < 0.30:
                    failing_items[i_code] = i_stat.citc

        self.assertEqual(
            len(failing_items),
            0,
            f"Items failed CITC >= 0.30 threshold: {failing_items}",
        )

    def test_summary_structure_and_serialization(self):
        """Verify get_summary() returns full StatsSummary with valid dictionary serialization."""
        summary = self.analyzer.get_summary()
        self.assertEqual(summary.sample_count, self.num_samples)
        self.assertTrue(summary.is_reliable)
        self.assertEqual(summary.status, "RELIABLE")
        self.assertEqual(len(summary.construct_stats), 11)

        summary_dict = summary.to_dict()
        self.assertIn("sample_count", summary_dict)
        self.assertIn("overall_alpha", summary_dict)
        self.assertIn("constructs", summary_dict)
        self.assertIn("PC", summary_dict["constructs"])
        pc_dict = summary_dict["constructs"]["PC"]
        self.assertIn("mean_raw", pc_dict)
        self.assertIn("mean_reversed", pc_dict)


class TestPsychometricsEdgeCases(unittest.TestCase):
    def test_small_sample_edge_cases(self):
        """Test behavior on N=0, N=1, N=2."""
        tracker = StreamingCovarianceTracker(4)
        # N = 0
        alpha, status = tracker.compute_alpha()
        self.assertEqual(alpha, 0.0)
        self.assertEqual(status, "INSUFFICIENT_DATA")

        # N = 1
        tracker.update(np.array([4.0, 4.0, 5.0, 5.0]))
        alpha, status = tracker.compute_alpha()
        self.assertEqual(alpha, 0.0)
        self.assertEqual(status, "INSUFFICIENT_DATA")

        # N = 2
        tracker.update(np.array([5.0, 4.0, 4.0, 5.0]))
        alpha, status = tracker.compute_alpha()
        self.assertEqual(alpha, 0.0)
        self.assertEqual(status, "INSUFFICIENT_DATA")

    def test_zero_variance_guard(self):
        """Test behavior when all respondents provide identical responses (Var = 0)."""
        tracker = StreamingCovarianceTracker(3)
        for _ in range(15):
            tracker.update(np.array([5.0, 5.0, 5.0]))

        alpha, status = tracker.compute_alpha()
        self.assertEqual(alpha, 0.0)
        self.assertEqual(status, "ZERO_VARIANCE")


if __name__ == "__main__":
    unittest.main()
