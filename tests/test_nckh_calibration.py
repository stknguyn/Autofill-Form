"""
tests/test_nckh_calibration.py
Authoritative Automated Verification Suite for NCKH Survey Autofill Logic:
- Strictly constrains Cronbach's Alpha within [0.76, 0.79] (never exceeds 0.79, no >= 0.80).
- Dynamically adapts lower and upper bounds of 11 measurement scales based on sample size.
- Strictly bounds generated item scores within calibrated scale bounds.
- Enforces minimum respondent age >= 19.
- Enforces Vietnamese demographic coherence (state official / vien chuc requires age >= 22).
"""

from __future__ import annotations

import unittest
from typing import Dict, List, Tuple

import numpy as np

from src.config_service import ConfigService
from src.persona_engine import (
    OPT_EDU_DAI_HOC,
    OPT_EDU_KHAC,
    OPT_EDU_SAU_DAI_HOC,
    OPT_EDU_TRUNG_CAP,
    OPT_PROF_VIEN_CHUC,
    CrossConsistencyValidator,
    PersonaEngine,
    calibrate_scale_bounds,
)
from src.stats_analyzer import StatsAnalyzer
from src.tally_schema import TallySchema
from src.worker_manager import FleetConfig, WorkerManager


class TestNCKHCronbachAlphaBounds(unittest.TestCase):
    """
    R1 Acceptance Criteria:
    - For simulated or generated survey response batches, the computed Cronbach's alpha
      strictly falls in the range [0.76, 0.79] and never exceeds 0.79.
    - The generation logic no longer allows Cronbach's alpha to default to or exceed >= 0.80.
    """

    @classmethod
    def setUpClass(cls):
        TallySchema.initialize()

    def test_alpha_strictly_within_bounds_across_varying_sample_sizes(self):
        """Verify computed Cronbach's Alpha across all 11 scales falls in [0.76, 0.79] for varying N."""
        sample_sizes = [30, 50, 100, 150, 200, 300, 500]
        test_scales = ["PU", "PEOU", "PES", "PC", "PI", "SE", "LC", "ATT", "SN", "PBC", "ITU"]

        for n in sample_sizes:
            engine = PersonaEngine(seed=1000 + n)
            profiles = engine.generate_batch(sample_size=n)
            self.assertEqual(len(profiles), n, f"Expected {n} profiles, got {len(profiles)}")

            analyzer = StatsAnalyzer()
            for p in profiles:
                analyzer.add_sample(p.likert_scores)

            alphas = analyzer.get_cronbach_alpha()

            # Check all 11 measurement scales
            for scale in test_scales:
                alpha_val = alphas.get(scale, 0.0)
                self.assertGreaterEqual(
                    alpha_val,
                    0.760,
                    f"Sample size N={n}, Scale '{scale}' Alpha={alpha_val:.4f} is below lower bound 0.760",
                )
                self.assertLessEqual(
                    alpha_val,
                    0.790,
                    f"Sample size N={n}, Scale '{scale}' Alpha={alpha_val:.4f} exceeds ceiling 0.790",
                )
                self.assertLess(
                    alpha_val,
                    0.800,
                    f"Sample size N={n}, Scale '{scale}' Alpha={alpha_val:.4f} must NOT be >= 0.80",
                )

            # Check overall scale does not exceed 0.790 ceiling
            overall = alphas.get("overall", 0.0)
            self.assertLessEqual(
                overall,
                0.790,
                f"Sample size N={n}, Overall Alpha={overall:.4f} exceeds ceiling 0.790",
            )
            self.assertLess(
                overall,
                0.800,
                f"Sample size N={n}, Overall Alpha={overall:.4f} must NOT be >= 0.80",
            )

            # Check analyzer compliance helper
            self.assertTrue(
                analyzer.is_nckh_compliant(alpha_min=0.760, alpha_max=0.790),
                f"Sample size N={n} failed is_nckh_compliant check: {alphas}",
            )

    def test_alpha_bounded_in_buffered_engine_generation(self):
        """Verify PersonaEngine initialized with sample_size dispenses calibrated profiles."""
        sample_size = 120
        engine = PersonaEngine(sample_size=sample_size, seed=777)
        profiles = [engine.generate_respondent() for _ in range(sample_size)]

        analyzer = StatsAnalyzer()
        for p in profiles:
            analyzer.add_sample(p.likert_scores)

        alphas = analyzer.get_cronbach_alpha()
        for scale in ["PU", "PEOU", "PES", "PC", "PI", "SE", "LC", "ATT", "SN", "PBC", "ITU"]:
            val = alphas[scale]
            self.assertTrue(
                0.760 <= val <= 0.790,
                f"Construct {scale} alpha {val:.4f} outside [0.76, 0.79]",
            )
        self.assertLessEqual(alphas["overall"], 0.790)

    def test_alpha_strictly_within_bounds_large_sample_sizes(self):
        """Verify computed Cronbach's Alpha across all 11 scales falls in [0.76, 0.79] for large N."""
        test_scales = ["PU", "PEOU", "PES", "PC", "PI", "SE", "LC", "ATT", "SN", "PBC", "ITU"]
        for n in [1000, 2000]:
            engine = PersonaEngine(seed=5000 + n)
            profiles = engine.generate_batch(sample_size=n)
            self.assertEqual(len(profiles), n)

            analyzer = StatsAnalyzer()
            for p in profiles:
                analyzer.add_sample(p.likert_scores)

            alphas = analyzer.get_cronbach_alpha()
            for scale in test_scales:
                val = alphas.get(scale, 0.0)
                self.assertGreaterEqual(
                    val,
                    0.760,
                    f"Sample size N={n}, Scale '{scale}' Alpha={val:.4f} is below 0.760",
                )
                self.assertLessEqual(
                    val,
                    0.790,
                    f"Sample size N={n}, Scale '{scale}' Alpha={val:.4f} exceeds ceiling 0.790",
                )
            self.assertLessEqual(alphas.get("overall", 0.0), 0.790)

    def test_alpha_strictly_within_bounds_small_sample_sizes(self):
        """Verify computed Cronbach's Alpha across all 11 scales falls in [0.76, 0.79] for small N (5, 10, 15, 20)."""
        test_scales = ["PU", "PEOU", "PES", "PC", "PI", "SE", "LC", "ATT", "SN", "PBC", "ITU"]
        for n in [5, 10, 15, 20]:
            engine = PersonaEngine(seed=7000 + n)
            profiles = engine.generate_batch(sample_size=n)
            self.assertEqual(len(profiles), n)

            analyzer = StatsAnalyzer()
            for p in profiles:
                analyzer.add_sample(p.likert_scores)

            alphas = analyzer.get_cronbach_alpha()
            for scale in test_scales:
                val = alphas.get(scale, 0.0)
                self.assertGreaterEqual(
                    val,
                    0.760,
                    f"Sample size N={n}, Scale '{scale}' Alpha={val:.4f} is below 0.760",
                )
                self.assertLessEqual(
                    val,
                    0.790,
                    f"Sample size N={n}, Scale '{scale}' Alpha={val:.4f} exceeds ceiling 0.790",
                )
            self.assertLessEqual(
                alphas.get("overall", 0.0),
                0.790,
                f"Sample size N={n}, Overall Alpha={alphas.get('overall', 0.0):.4f} exceeds ceiling 0.790",
            )

    def test_micro_sample_sizes_alpha_convergence(self):
        """Verify computed Cronbach's Alpha strictly converges to [0.76, 0.79] on micro sample size N=3."""
        test_scales = ["PU", "PEOU", "PES", "PC", "PI", "SE", "LC", "ATT", "SN", "PBC", "ITU"]
        for seed in [28, 31, 69, 92, 96]:
            engine = PersonaEngine(seed=seed)
            profiles = engine.generate_batch(sample_size=3)
            self.assertEqual(len(profiles), 3)

            analyzer = StatsAnalyzer()
            for p in profiles:
                analyzer.add_sample(p.likert_scores)

            alphas = analyzer.get_cronbach_alpha()
            for scale in test_scales:
                val = alphas.get(scale, 0.0)
                self.assertGreaterEqual(
                    val,
                    0.760,
                    f"Micro N=3 seed={seed}, Scale '{scale}' Alpha={val:.4f} is below 0.760",
                )
                self.assertLessEqual(
                    val,
                    0.790,
                    f"Micro N=3 seed={seed}, Scale '{scale}' Alpha={val:.4f} exceeds ceiling 0.790",
                )
            self.assertLessEqual(
                alphas.get("overall", 0.0),
                0.790,
                f"Micro N=3 seed={seed}, Overall Alpha={alphas.get('overall', 0.0):.4f} exceeds ceiling 0.790",
            )


class TestNCKHDynamicScaleCalibration(unittest.TestCase):
    """
    R2 Acceptance Criteria:
    - Lower and upper bounds for each of the 11 measurement scales adjust dynamically
      according to the specified sample size.
    - Generated item scores across all 11 scales remain within the calibrated bounds
      for the given sample count.
    """

    def test_dynamic_bounds_adjustment_by_sample_size(self):
        """Verify calibrated bounds change dynamically with increasing sample size."""
        scales = ["PU", "PEOU", "PES", "PC", "PI", "SE", "LC", "ATT", "SN", "PBC", "ITU"]

        b_small = calibrate_scale_bounds(20)
        b_large = calibrate_scale_bounds(300)

        # All 11 scales must be present
        self.assertEqual(set(b_small.keys()), set(scales))
        self.assertEqual(set(b_large.keys()), set(scales))

        # Dynamic variation between small and large sample sizes across ALL 11 scales
        has_dynamic_difference_all = all(b_small[s] != b_large[s] for s in scales)
        self.assertTrue(
            has_dynamic_difference_all,
            f"Calibrated scale bounds must adapt dynamically for EACH of the 11 scales (small={b_small}, large={b_large})",
        )

        # Verify sanity of all calibrated bounds
        for n in [20, 50, 100, 250, 500, 1000]:
            bounds = calibrate_scale_bounds(n)
            for s, (lb, ub) in bounds.items():
                self.assertGreaterEqual(lb, 1, f"Scale {s} lb={lb} < 1 for N={n}")
                self.assertLessEqual(ub, 6, f"Scale {s} ub={ub} > 6 for N={n}")
                self.assertLess(lb, ub, f"Scale {s} lb={lb} >= ub={ub} for N={n}")

    def test_generated_scores_strictly_within_calibrated_bounds(self):
        """Verify all generated Likert items strictly stay inside calibrated bounds across all 11 scales."""
        for n in [30, 75, 150, 300]:
            engine = PersonaEngine(seed=42 + n)
            bounds = calibrate_scale_bounds(n)
            profiles = engine.generate_batch(sample_size=n)

            for idx, p in enumerate(profiles):
                for q_def in TallySchema.LIKERT_QUESTIONS:
                    score = p.likert_scores[q_def.code]
                    c_code = q_def.construct_code or q_def.code.rstrip("0123456789")
                    lb, ub = bounds[c_code]
                    self.assertGreaterEqual(
                        score,
                        lb,
                        f"Profile {idx} item {q_def.code} score {score} < lower bound {lb} (N={n})",
                    )
                    self.assertLessEqual(
                        score,
                        ub,
                        f"Profile {idx} item {q_def.code} score {score} > upper bound {ub} (N={n})",
                    )


class TestNCKHDemographicsAndValidation(unittest.TestCase):
    """
    R3 Acceptance Criteria:
    - Every generated respondent has an age >= 19.
    - All generated survey responses pass demographic consistency checks:
      no respondent under 22 years of age has an occupation requiring higher education completion
      such as state official / viên chức nhà nước / công chức.
    - Age, occupation, and related demographic selections are mutually consistent
      according to defined Vietnamese demographic rules.
    """

    def test_every_respondent_has_age_ge_19(self):
        """Verify every single generated respondent across 500 profiles has age >= 19."""
        engine = PersonaEngine(seed=2026)
        profiles = [engine.generate_respondent() for _ in range(500)]

        for idx, p in enumerate(profiles):
            self.assertGreaterEqual(
                p.age,
                19,
                f"Respondent {idx} has age {p.age} < 19",
            )
            self.assertLessEqual(
                p.age,
                27,
                f"Respondent {idx} has age {p.age} > 27",
            )

    def test_zero_respondents_under_22_with_state_official(self):
        """Verify no respondent under 22 years old holds state official / vien chuc post."""
        engine = PersonaEngine(seed=3030)
        profiles = [engine.generate_respondent() for _ in range(500)]

        violations = []
        for idx, p in enumerate(profiles):
            if p.age < 22 and p.profession == OPT_PROF_VIEN_CHUC:
                violations.append(f"Respondent {idx}: Age={p.age}, Profession={p.profession}")

        self.assertEqual(
            len(violations),
            0,
            f"Found respondents under 22 holding state official post: {violations}",
        )

    def test_all_generated_profiles_pass_cross_consistency(self):
        """Verify 100% of generated respondents pass CrossConsistencyValidator with 0 violations."""
        engine = PersonaEngine(seed=4040)
        batch = engine.generate_batch(sample_size=200)

        for idx, p in enumerate(batch):
            v = CrossConsistencyValidator.validate(p)
            self.assertEqual(
                len(v),
                0,
                f"Batch profile {idx} failed cross-consistency validation: {v}",
            )

    def test_cross_consistency_validator_detects_age_below_19(self):
        """Verify CrossConsistencyValidator flags Rule 8 when age < 19."""
        underage_profile = {
            "age": 18,
            "education": "Đại học",
            "employment": "Toàn thời gian (Full-time)",
            "profession": "Kinh doanh / Bán lẻ / Thương mại điện tử",
            "frequency": "2 - 4 đơn/tháng",
            "categories": ["Thời trang, quần áo"],
            "pain_points": ["Không gặp bất tiện nào"],
        }
        violations = CrossConsistencyValidator.validate(underage_profile)
        self.assertTrue(
            any("Rule 8" in v or "19" in v for v in violations),
            f"Expected Rule 8 violation for age 18, got: {violations}",
        )

    def test_cross_consistency_validator_detects_under_22_state_official(self):
        """Verify CrossConsistencyValidator flags Rule 7 when age < 22 holds OPT_PROF_VIEN_CHUC."""
        bad_profile = {
            "age": 21,
            "education": "Đại học",
            "employment": "Toàn thời gian (Full-time)",
            "profession": OPT_PROF_VIEN_CHUC,
            "frequency": "2 - 4 đơn/tháng",
            "categories": ["Thời trang, quần áo"],
            "pain_points": ["Không gặp bất tiện nào"],
        }
        violations = CrossConsistencyValidator.validate(bad_profile)
        self.assertTrue(
            any("Rule 7" in v for v in violations),
            f"Expected Rule 7 violation for age 21 with OPT_PROF_VIEN_CHUC, got: {violations}",
        )

    def test_cross_consistency_validator_detects_under_22_state_official_text_label(self):
        """Verify CrossConsistencyValidator flags Rule 7 when age < 22 holds Vietnamese text for state official."""
        bad_profile = {
            "age": "21",
            "education": "Đại học",
            "employment": "Toàn thời gian (Full-time)",
            "profession": "Cán bộ, công chức, viên chức nhà nước",
            "frequency": "2 - 4 đơn/tháng",
            "categories": ["Thời trang, quần áo"],
            "pain_points": ["Không gặp bất tiện nào"],
        }
        violations = CrossConsistencyValidator.validate(bad_profile)
        self.assertTrue(
            any("Rule 7" in v for v in violations),
            f"Expected Rule 7 violation for age 21 with text state official, got: {violations}",
        )

    def test_cross_consistency_validator_detects_state_official_with_thpt_education(self):
        """Verify CrossConsistencyValidator flags Rule 7 when state official holds THPT education."""
        bad_profile = {
            "age": 25,
            "education": "Từ Trung học phổ thông (THPT) trở xuống",
            "employment": "Toàn thời gian (Full-time)",
            "profession": OPT_PROF_VIEN_CHUC,
            "frequency": "2 - 4 đơn/tháng",
            "categories": ["Thời trang, quần áo"],
            "pain_points": ["Không gặp bất tiện nào"],
        }
        violations = CrossConsistencyValidator.validate(bad_profile)
        self.assertTrue(
            any("Rule 7" in v for v in violations),
            f"Expected Rule 7 violation for state official with THPT education, got: {violations}",
        )

    def test_cross_consistency_validator_detects_state_official_with_trung_cap_education(self):
        """Verify CrossConsistencyValidator flags Rule 7 when state official holds Trung Cap / vocational education."""
        bad_profile = {
            "age": 25,
            "education": OPT_EDU_TRUNG_CAP,
            "employment": "Toàn thời gian (Full-time)",
            "profession": OPT_PROF_VIEN_CHUC,
            "frequency": "2 - 4 đơn/tháng",
            "categories": ["Thời trang, quần áo"],
            "pain_points": ["Không gặp bất tiện nào"],
        }
        violations = CrossConsistencyValidator.validate(bad_profile)
        self.assertTrue(
            any("Rule 7" in v for v in violations),
            f"Expected Rule 7 violation for state official with OPT_EDU_TRUNG_CAP, got: {violations}",
        )

    def test_cross_consistency_validator_detects_state_official_with_other_low_education(self):
        """Verify CrossConsistencyValidator flags Rule 7 when state official holds 'Khác' or non-higher education."""
        bad_profile = {
            "age": 25,
            "education": OPT_EDU_KHAC,
            "employment": "Toàn thời gian (Full-time)",
            "profession": "Cán bộ nhân viên nhà nước",
            "frequency": "2 - 4 đơn/tháng",
            "categories": ["Thời trang, quần áo"],
            "pain_points": ["Không gặp bất tiện nào"],
        }
        violations = CrossConsistencyValidator.validate(bad_profile)
        self.assertTrue(
            any("Rule 7" in v for v in violations),
            f"Expected Rule 7 violation for 'nhà nước' with OPT_EDU_KHAC, got: {violations}",
        )

    def test_generated_state_officials_never_have_low_education(self):
        """Verify that 100% of generated state officials across 500 respondents have higher education degree."""
        engine = PersonaEngine(seed=8888)
        profiles = [engine.generate_respondent(_bypass_nckh=True) for _ in range(500)]
        state_officials = [p for p in profiles if p.profession == OPT_PROF_VIEN_CHUC]
        self.assertGreater(len(state_officials), 0, "Expected at least some state officials generated")
        for so in state_officials:
            self.assertIn(
                so.education,
                (OPT_EDU_DAI_HOC, OPT_EDU_SAU_DAI_HOC),
                f"Generated state official has non-higher education: {so.education}",
            )

    def test_persona_engine_respects_nested_demographics_config(self):
        """Verify PersonaEngine respects nested demographics (min_age, age_mean) from config dictionary."""
        cfg = {
            "demographics": {
                "min_age": 20,
                "age_mean": 24.5,
                "age_std": 1.2,
                "gender_weights": [0.05, 0.90, 0.05],
            }
        }
        engine = PersonaEngine(config=cfg, seed=777)
        profiles = [engine.generate_respondent(_bypass_nckh=True) for _ in range(100)]
        for p in profiles:
            self.assertGreaterEqual(p.age, 20)
        mean_age = sum(p.age for p in profiles) / len(profiles)
        self.assertGreater(mean_age, 23.0)

    def test_persona_engine_default_session_config_activates_nckh(self):
        """Verify PersonaEngine initialized with default session config automatically enables NCKH mode."""
        cfg = ConfigService.get_default_config()
        engine = PersonaEngine(config=cfg, seed=1234)
        self.assertTrue(engine.nckh_mode, "PersonaEngine must enable nckh_mode when initialized with default session config")


class TestNCKHDeterminismAndReproducibility(unittest.TestCase):
    """Verify bit-exact reproducibility for identical seeds."""

    def test_deterministic_batch_output(self):
        engine_a = PersonaEngine(seed=999)
        engine_b = PersonaEngine(seed=999)

        batch_a = engine_a.generate_batch(sample_size=30)
        batch_b = engine_b.generate_batch(sample_size=30)

        for i, (pa, pb) in enumerate(zip(batch_a, batch_b)):
            self.assertEqual(pa.age, pb.age, f"Age mismatch at {i}")
            self.assertEqual(pa.profession, pb.profession, f"Profession mismatch at {i}")
            self.assertEqual(pa.likert_scores, pb.likert_scores, f"Likert mismatch at {i}")

    def test_generate_batch_rng_integer_determinism(self):
        """Verify generate_batch(..., rng=seed) produces bit-exact identical demographics and Likert scores."""
        engine = PersonaEngine(seed=12345)
        batch1 = engine.generate_batch(sample_size=20, rng=42)
        batch2 = engine.generate_batch(sample_size=20, rng=42)

        for i, (p1, p2) in enumerate(zip(batch1, batch2)):
            self.assertEqual(p1.gender, p2.gender, f"Gender mismatch at {i}")
            self.assertEqual(p1.age, p2.age, f"Age mismatch at {i}")
            self.assertEqual(p1.education, p2.education, f"Education mismatch at {i}")
            self.assertEqual(p1.profession, p2.profession, f"Profession mismatch at {i}")
            self.assertEqual(p1.likert_scores, p2.likert_scores, f"Likert mismatch at {i}")


class TestNCKHWorkerFleetAutofill(unittest.TestCase):
    """Verify WorkerManager fleet autofill strictly enforces Cronbach's alpha within [0.76, 0.79]."""

    def test_worker_manager_autofill_alpha_strictly_within_bounds(self):
        cfg = FleetConfig(
            target_count=60,
            num_workers=4,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config={"min_delay": 0.0, "max_delay": 0.001, "micro_break_enabled": False},
            master_seed=42,
            demographic_weights={
                "nckh_mode": True,
                "alpha_threshold": 0.78,
                "alpha_ceiling": 0.79,
                "min_age": 19,
            },
        )
        wm = WorkerManager(config=cfg)
        wm.start()
        completed = wm.wait_until_complete(timeout=10.0)
        wm.stop(timeout=2.0)

        self.assertTrue(completed)
        alphas = wm.stats_analyzer.get_cronbach_alpha()
        for scale in ["PU", "PEOU", "PES", "PC", "PI", "SE", "LC", "ATT", "SN", "PBC", "ITU"]:
            val = alphas.get(scale, 0.0)
            self.assertGreaterEqual(val, 0.760, f"Scale {scale} alpha {val:.4f} < 0.760")
            self.assertLessEqual(val, 0.790, f"Scale {scale} alpha {val:.4f} > 0.790")
        self.assertLessEqual(alphas.get("overall", 0.0), 0.790)

    def test_worker_manager_refund_preserves_calibrated_profile(self):
        """Verify WorkerManager._refund_work re-queues calibrated profile on failure."""
        cfg = FleetConfig(
            target_count=10,
            num_workers=1,
            dry_run=True,
            master_seed=42,
            demographic_weights={"nckh_mode": True, "alpha_threshold": 0.78, "alpha_ceiling": 0.79},
        )
        wm = WorkerManager(config=cfg)
        wm.start()
        wm.pause()
        initial_len = len(wm._calibrated_profiles)
        p = wm._calibrated_profiles[0] if wm._calibrated_profiles else None
        wm._refund_work(p)
        self.assertEqual(len(wm._calibrated_profiles), initial_len + (1 if p else 0))
        wm.stop(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
