"""
tests/test_persona_engine.py
Automated Verification Suite for Persona Engine & Demographic Cross-Consistency.
Ensures 100% compliance across 150 generated profiles against all 6 logical rules.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.persona_engine import (
    CATEGORIES_QUESTION_UUID,
    CrossConsistencyValidator,
    EDUCATION_QUESTION_UUID,
    EMPLOYMENT_QUESTION_UUID,
    FREQUENCY_QUESTION_UUID,
    LOCATION_QUESTION_UUID,
    PAIN_POINTS_QUESTION_UUID,
    PROFESSION_QUESTION_UUID,
    OPT_CAT_BEAUTY,
    OPT_CAT_BOOKS,
    OPT_CAT_FASHION,
    OPT_CAT_HOUSEHOLD,
    OPT_CAT_OTHER,
    OPT_CAT_TECH,
    OPT_CONSENT_AGREE,
    OPT_CONSENT_DISAGREE,
    OPT_EDU_DAI_HOC,
    OPT_EDU_SAU_DAI_HOC,
    OPT_EDU_THPT,
    OPT_EDU_TRUNG_CAP,
    OPT_EMP_FREELANCER,
    OPT_EMP_FULL_TIME,
    OPT_EMP_KHONG_LAO_DONG,
    OPT_EMP_PART_TIME,
    OPT_EMP_TIM_VIEC,
    OPT_FREQ_2_TO_4,
    OPT_FREQ_4_TO_6,
    OPT_FREQ_6_TO_8,
    OPT_FREQ_OVER_8,
    OPT_FREQ_UNDER_2,
    OPT_LOC_DANANG,
    OPT_LOC_HANOI,
    OPT_LOC_HCM,
    OPT_LOC_KHAC,
    OPT_PAIN_BUILDING,
    OPT_PAIN_NONE,
    OPT_PAIN_OTHER,
    OPT_PAIN_PRIVACY_LOSS,
    OPT_PROF_CNTT,
    OPT_PROF_FREELANCE_CONTENT,
    OPT_PROF_GIAO_DUC,
    OPT_PROF_KINH_DOANH,
    OPT_PROF_SAN_XUAT,
    OPT_PROF_VIEN_CHUC,
    OPT_PROF_VU_TRANG,
    OPT_USAGE_NEVER,
    OPT_USAGE_TODAY_YESTERDAY,
    PersonaEngine,
    REASON_NEVER_QUESTION_UUID,
    RespondentProfile,
)
from src.stats_analyzer import StatsAnalyzer
from src.tally_schema import (
    AGE_QUESTION_UUID,
    CONSENT_AGREE_UUID,
    CONSENT_QUESTION_UUID,
    TallySchema,
    USAGE_NEVER_UUID,
    USAGE_QUESTION_UUID,
)


class TestPersonaEngineCrossConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TallySchema.initialize()
        cls.engine = PersonaEngine(seed=42)
        # Generate exactly 150 samples as mandated by authoritative request
        cls.profiles: List[RespondentProfile] = [
            cls.engine.generate_respondent() for _ in range(150)
        ]

    def test_sample_count(self):
        """Verify exactly 150 profiles are generated."""
        self.assertEqual(len(self.profiles), 150)

    def test_demographics_age_bounds(self):
        """Verify all respondents are Gen Z strictly within [18, 27]."""
        for idx, p in enumerate(self.profiles):
            age = p.age
            self.assertTrue(18 <= age <= 27, f"Sample {idx} Age {age} out of range [18, 27]")
            self.assertEqual(p.demographics_raw["age"], age)

    def test_rule_1_and_2_age_vs_education(self):
        """Rule 1 & 2: Age < 22 (and Age <= 20) cannot have Postgraduate education."""
        violations = []
        for idx, p in enumerate(self.profiles):
            age = p.age
            edu = p.education
            if age <= 20 and edu == OPT_EDU_SAU_DAI_HOC:
                violations.append(f"Sample {idx}: Age={age} has Postgraduate degree")
            if age < 22 and edu == OPT_EDU_SAU_DAI_HOC:
                violations.append(f"Sample {idx}: Age={age} (<22) has Postgraduate degree")
        self.assertEqual(len(violations), 0, f"Rule 1 & 2 Violations: {violations}")

    def test_rule_2_age_vs_employment_distribution(self):
        """
        Rule 2: Age <= 20 full-time <= 10%, Age >= 24 non-labor <= 5%.
        Applies a 3-sigma binomial confidence tolerance to prevent false test rejections
        caused by finite-sample stochastic variance on small sub-samples (N ~ 25-45).
        """
        import math

        young_fulltime = 0
        young_total = 0
        older_nonlabor = 0
        older_total = 0

        for p in self.profiles:
            age = p.age
            emp = p.employment
            if age <= 20:
                young_total += 1
                if emp == OPT_EMP_FULL_TIME:
                    young_fulltime += 1
            if age >= 24:
                older_total += 1
                if emp == OPT_EMP_KHONG_LAO_DONG:
                    older_nonlabor += 1

        if young_total > 0:
            rate_young = young_fulltime / young_total
            # 3-sigma upper confidence bound for H0: p <= 0.10
            tol_young = 3.0 * math.sqrt(0.10 * 0.90 / young_total)
            max_allowed_young = 0.10 + tol_young
            self.assertLessEqual(
                rate_young, max_allowed_young,
                f"Full-time rate for age <= 20 ({rate_young:.2%}, {young_fulltime}/{young_total}) "
                f"exceeds 3-sigma upper threshold ({max_allowed_young:.2%})"
            )

        if older_total > 0:
            rate_older = older_nonlabor / older_total
            # 3-sigma upper confidence bound for H0: p <= 0.05
            tol_older = 3.0 * math.sqrt(0.05 * 0.95 / older_total)
            max_allowed_older = 0.05 + tol_older
            self.assertLessEqual(
                rate_older, max_allowed_older,
                f"Non-labor rate for age >= 24 ({rate_older:.2%}, {older_nonlabor}/{older_total}) "
                f"exceeds 3-sigma upper threshold ({max_allowed_older:.2%})"
            )

    def test_rule_3_employment_vs_profession(self):
        """Rule 3: State occupations (Civil servant / Armed forces) require Full-time; Freelancer cannot hold state post."""
        state_professions = {OPT_PROF_VIEN_CHUC, OPT_PROF_VU_TRANG}
        violations = []
        for idx, p in enumerate(self.profiles):
            emp = p.employment
            prof = p.profession
            if prof in state_professions and emp != OPT_EMP_FULL_TIME:
                violations.append(f"Sample {idx}: State profession ({prof}) is not Full-time ({emp})")
            if emp == OPT_EMP_FREELANCER and prof in state_professions:
                violations.append(f"Sample {idx}: Freelancer holds state post ({prof})")
        self.assertEqual(len(violations), 0, f"Rule 3 Violations: {violations}")

    def test_rule_4_inactive_employment_vs_profession(self):
        """Rule 4: Inactive employment (Seeking job / Non-labor) cannot be civil servant, armed forces, or production."""
        inactive_employments = {OPT_EMP_TIM_VIEC, OPT_EMP_KHONG_LAO_DONG}
        forbidden_professions = {OPT_PROF_VIEN_CHUC, OPT_PROF_VU_TRANG, OPT_PROF_SAN_XUAT}
        violations = []
        for idx, p in enumerate(self.profiles):
            emp = p.employment
            prof = p.profession
            if emp in inactive_employments and prof in forbidden_professions:
                violations.append(f"Sample {idx}: Inactive ({emp}) holds forbidden profession ({prof})")
        self.assertEqual(len(violations), 0, f"Rule 4 Violations: {violations}")

    def test_rule_5_frequency_and_categories(self):
        """Rule 5: Category counts 1 to 3; Infrequent shopper (<2 orders/mo) max 2 cats; Frequent shopper (>8 orders/mo) min 2 cats."""
        violations = []
        for idx, p in enumerate(self.profiles):
            cats = p.categories
            freq = p.frequency
            if len(cats) == 0:
                violations.append(f"Sample {idx}: Empty categories selection")
            if freq == OPT_FREQ_UNDER_2 and len(cats) > 2:
                violations.append(f"Sample {idx}: Infrequent shopper has {len(cats)} categories (> 2)")
            if freq == OPT_FREQ_OVER_8 and len(cats) < 2:
                violations.append(f"Sample {idx}: Frequent shopper has {len(cats)} categories (< 2)")
        self.assertEqual(len(violations), 0, f"Rule 5 Violations: {violations}")

    def test_rule_6_pain_points_exclusivity(self):
        """Rule 6: 'No pain point' is mutually exclusive with any other pain points."""
        violations = []
        for idx, p in enumerate(self.profiles):
            pains = p.pain_points
            if len(pains) == 0:
                violations.append(f"Sample {idx}: Empty pain points")
            if OPT_PAIN_NONE in pains and len(pains) > 1:
                violations.append(f"Sample {idx}: OPT_PAIN_NONE chosen alongside others: {pains}")
        self.assertEqual(len(violations), 0, f"Rule 6 Violations: {violations}")

    def test_screening_logic_invariance(self):
        """Verify 100% consent, qualified usage (never USAGE_NEVER), and no Q3 reason in payload."""
        for idx, p in enumerate(self.profiles):
            consent = p.payload_responses.get(CONSENT_QUESTION_UUID)
            self.assertEqual(consent, [CONSENT_AGREE_UUID], f"Sample {idx} did not agree")
            usage = p.payload_responses.get(USAGE_QUESTION_UUID)
            self.assertNotEqual(usage, [USAGE_NEVER_UUID], f"Sample {idx} selected Never Shopee")
            self.assertNotIn(REASON_NEVER_QUESTION_UUID, p.payload_responses, f"Sample {idx} triggered Q3")

    def test_payload_validity_via_tally_schema(self):
        """Verify 100% of the 150 generated payloads pass TallySchema.validate_payload()."""
        for idx, p in enumerate(self.profiles):
            payload = TallySchema.get_payload(p.payload_responses)
            is_valid = TallySchema.validate_payload(payload, raise_exception=False)
            if not is_valid:
                errors = TallySchema.get_validation_errors(payload)
                self.fail(f"Sample {idx} failed schema validation:\n - " + "\n - ".join(errors))

    def test_zero_violations_via_validator(self):
        """Verify CrossConsistencyValidator reports 0 violations across all 150 profiles."""
        total_violations = 0
        for idx, p in enumerate(self.profiles):
            violations = CrossConsistencyValidator.validate(p)
            if violations:
                total_violations += len(violations)
        self.assertEqual(total_violations, 0, f"Observed {total_violations} cross-consistency violations!")


class TestCrossConsistencyAdversarialFuzzing(unittest.TestCase):
    """Adversarial testing confirming CrossConsistencyValidator catches intentional violations."""

    def test_catch_rule_1_and_2_postgrad_underage(self):
        bad_profile = {
            "age": 19,
            "education": OPT_EDU_SAU_DAI_HOC,
            "employment": OPT_EMP_PART_TIME,
            "profession": OPT_PROF_KINH_DOANH,
            "frequency": OPT_FREQ_2_TO_4,
            "categories": [OPT_CAT_FASHION],
            "pain_points": [OPT_PAIN_BUILDING],
        }
        violations = CrossConsistencyValidator.validate(bad_profile)
        self.assertTrue(any("Rule 1" in v or "Rule 2" in v for v in violations))

    def test_catch_rule_3_freelancer_military(self):
        bad_profile = {
            "age": 25,
            "education": OPT_EDU_DAI_HOC,
            "employment": OPT_EMP_FREELANCER,
            "profession": OPT_PROF_VU_TRANG,
            "frequency": OPT_FREQ_2_TO_4,
            "categories": [OPT_CAT_FASHION],
            "pain_points": [OPT_PAIN_BUILDING],
        }
        violations = CrossConsistencyValidator.validate(bad_profile)
        self.assertTrue(any("Rule 3" in v for v in violations))

    def test_catch_rule_4_inactive_civil_servant(self):
        bad_profile = {
            "age": 22,
            "education": OPT_EDU_DAI_HOC,
            "employment": OPT_EMP_KHONG_LAO_DONG,
            "profession": OPT_PROF_VIEN_CHUC,
            "frequency": OPT_FREQ_2_TO_4,
            "categories": [OPT_CAT_FASHION],
            "pain_points": [OPT_PAIN_BUILDING],
        }
        violations = CrossConsistencyValidator.validate(bad_profile)
        self.assertTrue(any("Rule 4" in v for v in violations))

    def test_catch_rule_5_infrequent_excessive_categories(self):
        bad_profile = {
            "age": 22,
            "education": OPT_EDU_DAI_HOC,
            "employment": OPT_EMP_FULL_TIME,
            "profession": OPT_PROF_KINH_DOANH,
            "frequency": OPT_FREQ_UNDER_2,
            "categories": [OPT_CAT_FASHION, OPT_CAT_BEAUTY, OPT_CAT_TECH],  # 3 categories > 2
            "pain_points": [OPT_PAIN_BUILDING],
        }
        violations = CrossConsistencyValidator.validate(bad_profile)
        self.assertTrue(any("Rule 5" in v for v in violations))

    def test_catch_rule_6_pain_points_mutual_exclusion(self):
        bad_profile = {
            "age": 22,
            "education": OPT_EDU_DAI_HOC,
            "employment": OPT_EMP_FULL_TIME,
            "profession": OPT_PROF_KINH_DOANH,
            "frequency": OPT_FREQ_2_TO_4,
            "categories": [OPT_CAT_FASHION],
            "pain_points": [OPT_PAIN_NONE, OPT_PAIN_PRIVACY_LOSS],  # Contradictory
        }
        violations = CrossConsistencyValidator.validate(bad_profile)
        self.assertTrue(any("Rule 6" in v for v in violations))

    def test_catch_empty_categories_and_pains(self):
        bad_profile = {
            "age": 22,
            "education": OPT_EDU_DAI_HOC,
            "employment": OPT_EMP_FULL_TIME,
            "profession": OPT_PROF_KINH_DOANH,
            "frequency": OPT_FREQ_2_TO_4,
            "categories": [],
            "pain_points": [],
        }
        violations = CrossConsistencyValidator.validate(bad_profile)
        self.assertTrue(any("Rule 5" in v for v in violations))
        self.assertTrue(any("Rule 6" in v for v in violations))

    def test_catch_empty_demographic_lists_no_index_error(self):
        """
        Adversarial Boundary Test: Assert CrossConsistencyValidator.validate does NOT crash
        with IndexError when empty lists [] are passed for single-choice or multi-choice demographic fields.
        """
        # 1. Individual single-choice demographic fields with empty list []
        single_choice_keys = [
            "Q7_EDUCATION",
            "Q8_EMPLOYMENT",
            "Q9_PROFESSION",
            "Q10_FREQUENCY",
            "education_uuid",
            "employment_uuid",
            "profession_uuid",
            "frequency_uuid",
            EDUCATION_QUESTION_UUID,
            EMPLOYMENT_QUESTION_UUID,
            PROFESSION_QUESTION_UUID,
            FREQUENCY_QUESTION_UUID,
        ]
        for key in single_choice_keys:
            malformed_profile = {
                "age": 22,
                "education": OPT_EDU_DAI_HOC,
                "employment": OPT_EMP_FULL_TIME,
                "profession": OPT_PROF_KINH_DOANH,
                "frequency": OPT_FREQ_2_TO_4,
                "categories": [OPT_CAT_FASHION],
                "pain_points": [OPT_PAIN_BUILDING],
            }
            malformed_profile[key] = []
            try:
                violations = CrossConsistencyValidator.validate(malformed_profile)
                self.assertIsInstance(violations, list)
            except IndexError as e:
                self.fail(f"CrossConsistencyValidator crashed with IndexError on key '{key}': {e}")
            except Exception as e:
                self.fail(f"CrossConsistencyValidator raised unexpected exception on key '{key}': {e}")

        # 2. Pathological all-empty-lists profile
        all_empty_profile = {
            "age": 20,
            "Q7_EDUCATION": [],
            "Q8_EMPLOYMENT": [],
            "Q9_PROFESSION": [],
            "Q10_FREQUENCY": [],
            "Q11_CATEGORIES": [],
            "Q12_PAIN_POINTS": [],
            "education": [],
            "employment": [],
            "profession": [],
            "frequency": [],
            "categories": [],
            "pain_points": [],
            EDUCATION_QUESTION_UUID: [],
            EMPLOYMENT_QUESTION_UUID: [],
            PROFESSION_QUESTION_UUID: [],
            FREQUENCY_QUESTION_UUID: [],
            CATEGORIES_QUESTION_UUID: [],
            PAIN_POINTS_QUESTION_UUID: [],
        }
        try:
            violations = CrossConsistencyValidator.validate(all_empty_profile)
            self.assertIsInstance(violations, list)
            self.assertTrue(any("Rule 5" in v for v in violations), "Expected Rule 5 violation for empty categories")
            self.assertTrue(any("Rule 6" in v for v in violations), "Expected Rule 6 violation for empty pain points")
        except IndexError as e:
            self.fail(f"CrossConsistencyValidator crashed with IndexError on all-empty profile: {e}")


class TestPersonaEnginePRNGDeterminism(unittest.TestCase):
    """Test PRNG reproducibility and instance isolation with seeded PersonaEngine."""

    def test_persona_engine_prng_determinism_with_seed(self):
        """
        PRNG Reproducibility Test: Assert PersonaEngine(seed=...) produces 100% identical
        respondent sequences across separate instances, and distinct sequences with different seeds.
        """
        seed = 123
        engine_a = PersonaEngine(seed=seed)
        engine_b = PersonaEngine(seed=seed)

        # Generate sequence of 10 respondents from both engines
        seq_a = [engine_a.generate_respondent() for _ in range(10)]
        seq_b = [engine_b.generate_respondent() for _ in range(10)]

        for idx, (p_a, p_b) in enumerate(zip(seq_a, seq_b)):
            self.assertEqual(p_a.age, p_b.age, f"Respondent #{idx} age mismatch: {p_a.age} != {p_b.age}")
            self.assertEqual(p_a.gender_uuid, p_b.gender_uuid, f"Respondent #{idx} gender mismatch")
            self.assertEqual(p_a.location_uuid, p_b.location_uuid, f"Respondent #{idx} location mismatch")
            self.assertEqual(p_a.education, p_b.education, f"Respondent #{idx} education mismatch")
            self.assertEqual(p_a.employment, p_b.employment, f"Respondent #{idx} employment mismatch")
            self.assertEqual(p_a.profession, p_b.profession, f"Respondent #{idx} profession mismatch")
            self.assertEqual(p_a.frequency, p_b.frequency, f"Respondent #{idx} frequency mismatch")
            self.assertEqual(p_a.categories, p_b.categories, f"Respondent #{idx} categories mismatch")
            self.assertEqual(p_a.pain_points, p_b.pain_points, f"Respondent #{idx} pain points mismatch")
            self.assertEqual(
                p_a.likert_scores,
                p_b.likert_scores,
                f"Respondent #{idx} Likert scores mismatch between identically seeded instances",
            )

        # Different seed generates divergent respondents
        engine_c = PersonaEngine(seed=999)
        seq_c = [engine_c.generate_respondent() for _ in range(10)]
        diffs = sum(1 for p_a, p_c in zip(seq_a, seq_c) if p_a.likert_scores != p_c.likert_scores or p_a.age != p_c.age)
        self.assertGreater(diffs, 0, "Engines with different seeds produced identical profiles (seed inactive)")


class TestPersonaEngineLocationAgeAndPsychometricsRigor(unittest.TestCase):
    """
    Rigorously tests:
    - 0% selection rate for OPT_LOC_KHAC across >= 200 sampled profiles.
    - All profiles strictly allocate among TP.HCM, Hanoi, and Danang.
    - Every profile's age is strictly a clean integer (type int, no float, no decimal dot or comma).
    - Cronbach's Alpha >= 0.82 across all 11 TAM & TPB constructs.
    """

    def setUp(self):
        TallySchema.initialize()

    def test_zero_percent_opt_loc_khac_across_sampled_profiles(self):
        """
        Sample >= 200 profiles (here 250) across multiple seeds.
        Assert that OPT_LOC_KHAC selection count is strictly 0 (0.0%).
        Assert that all selected locations are within {OPT_LOC_HCM, OPT_LOC_HANOI, OPT_LOC_DANANG}.
        """
        engine = PersonaEngine(seed=42)
        sample_size = 250
        profiles: List[RespondentProfile] = [engine.generate_respondent() for _ in range(sample_size)]

        loc_counts: Dict[str, int] = {}
        for p in profiles:
            loc = p.location_uuid
            loc_counts[loc] = loc_counts.get(loc, 0) + 1

        # Strict 0% check for OPT_LOC_KHAC
        khac_count = loc_counts.get(OPT_LOC_KHAC, 0)
        self.assertEqual(
            khac_count,
            0,
            f"Expected strictly 0 selections of OPT_LOC_KHAC, but found {khac_count}/{sample_size}!",
        )

        # All locations must be in {HCM, Hanoi, Danang}
        allowed_locations = {OPT_LOC_HCM, OPT_LOC_HANOI, OPT_LOC_DANANG}
        for idx, p in enumerate(profiles):
            self.assertIn(
                p.location_uuid,
                allowed_locations,
                f"Profile #{idx} location {p.location_uuid} is not in allowed locations!",
            )
            self.assertIn(
                p.payload_responses[LOCATION_QUESTION_UUID][0],
                allowed_locations,
                f"Profile #{idx} payload response location is not in allowed locations!",
            )

        # Adversarial test: Even if demographic_weights explicitly passes a 4-element weight list with high 'Khác' weight
        adversarial_weights = {"location_weights": [0.25, 0.25, 0.25, 0.25]}
        adv_profiles = [
            engine.generate_respondent(demographic_weights=adversarial_weights) for _ in range(50)
        ]
        for p in adv_profiles:
            self.assertNotEqual(
                p.location_uuid,
                OPT_LOC_KHAC,
                "Adversarial weights with 4th element must still yield 0% OPT_LOC_KHAC!",
            )
            self.assertIn(p.location_uuid, allowed_locations)

    def test_clean_integer_age_across_all_profiles(self):
        """
        Assert every respondent's age attribute in profile, demographics dict, and payload
        is strictly of type `int` (clean integer: 18, 19, 20...), and never a float,
        nor formatted with decimal point (20.0) or decimal comma ("20,0").
        """
        engine = PersonaEngine(seed=101)
        profiles = [engine.generate_respondent() for _ in range(250)]

        for idx, p in enumerate(profiles):
            # 1. RespondentProfile.age
            self.assertIsInstance(p.age, int, f"Profile #{idx} age is not an int: {type(p.age)}")
            self.assertNotIsInstance(p.age, float, f"Profile #{idx} age is a float: {p.age}")
            self.assertIs(type(p.age), int, f"Profile #{idx} age type is not exact int: {type(p.age)}")

            # String representations must have no decimal point or comma
            age_str = str(p.age)
            self.assertNotIn(".", age_str, f"Profile #{idx} age string contains dot: {age_str}")
            self.assertNotIn(",", age_str, f"Profile #{idx} age string contains comma: {age_str}")
            self.assertTrue(age_str.isdigit(), f"Profile #{idx} age string is not digits: {age_str}")

            # 2. payload_responses[AGE_QUESTION_UUID]
            payload_age = p.payload_responses.get(AGE_QUESTION_UUID)
            self.assertIsInstance(
                payload_age, int, f"Profile #{idx} payload age is not int: {type(payload_age)}"
            )
            self.assertNotIsInstance(
                payload_age, float, f"Profile #{idx} payload age is float: {payload_age}"
            )
            self.assertIs(
                type(payload_age), int, f"Profile #{idx} payload age type is not exact int"
            )

            # 3. demographics dict
            demo_age = p.demographics.get("age")
            self.assertIsInstance(demo_age, int, f"Profile #{idx} demo age is not int")
            self.assertNotIsInstance(demo_age, float, f"Profile #{idx} demo age is float")

            raw_demo_age = p.demographics_raw.get("age")
            self.assertIsInstance(raw_demo_age, int, f"Profile #{idx} raw demo age is not int")
            self.assertNotIsInstance(raw_demo_age, float, f"Profile #{idx} raw demo age is float")

    def test_cronbach_alpha_reliability_across_all_11_constructs(self):
        """
        Verify that 200 sampled profiles maintain Cronbach's Alpha >= 0.82 across all 11
        TAM & TPB constructs (PU, PEOU, PES, PC, PI, SE, LC, ATT, SN, PBC, ITU)
        and full model overall alpha >= 0.90.
        """
        engine = PersonaEngine(seed=777)
        analyzer = StatsAnalyzer()

        profiles = [engine.generate_respondent() for _ in range(200)]
        for p in profiles:
            analyzer.add_sample(p.likert_scores)

        alphas = analyzer.get_cronbach_alpha()
        construct_list = ["PU", "PEOU", "PES", "PC", "PI", "SE", "LC", "ATT", "SN", "PBC", "ITU"]

        failing_constructs: Dict[str, float] = {}
        for c in construct_list:
            alpha_val = alphas.get(c, 0.0)
            if alpha_val < 0.82:
                failing_constructs[c] = alpha_val

        self.assertEqual(
            len(failing_constructs),
            0,
            f"Constructs failed Cronbach's Alpha >= 0.82: {failing_constructs}",
        )

        overall_alpha = alphas.get("overall", 0.0)
        self.assertGreaterEqual(
            overall_alpha,
            0.90,
            f"Overall model alpha {overall_alpha:.4f} is below 0.90",
        )


if __name__ == "__main__":
    unittest.main()

