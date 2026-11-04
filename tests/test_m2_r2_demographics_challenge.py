"""
tests/test_m2_r2_demographics_challenge.py
Adversarial Stress Test & Demographics Fuzzing Harness for M2 Iteration 2 Challenger 1.
Empirical verification of:
1. Adversarial empty list fuzzing on CrossConsistencyValidator.validate
2. Stress testing 500 generated profiles across all 6 cross-consistency rules
3. Direct validation of canonical payload_responses from PersonaEngine().generate_respondent()
"""

import math
import random
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.persona_engine import (
    AGE_QUESTION_UUID,
    AWARENESS_QUESTION_UUID,
    CATEGORIES_QUESTION_UUID,
    CONSENT_QUESTION_UUID,
    CrossConsistencyValidator,
    EDUCATION_QUESTION_UUID,
    EMPLOYMENT_QUESTION_UUID,
    FREQUENCY_QUESTION_UUID,
    GENDER_QUESTION_UUID,
    LOCATION_QUESTION_UUID,
    OPT_CAT_BEAUTY,
    OPT_CAT_BOOKS,
    OPT_CAT_FASHION,
    OPT_CAT_HOUSEHOLD,
    OPT_CAT_OTHER,
    OPT_CAT_TECH,
    OPT_CONSENT_AGREE,
    OPT_CONSENT_DISAGREE,
    OPT_EDU_DAI_HOC,
    OPT_EDU_KHAC,
    OPT_EDU_SAU_DAI_HOC,
    OPT_EDU_THPT,
    OPT_EDU_TRUNG_CAP,
    OPT_EMP_FREELANCER,
    OPT_EMP_FULL_TIME,
    OPT_EMP_KHAC,
    OPT_EMP_KHONG_LAO_DONG,
    OPT_EMP_PART_TIME,
    OPT_EMP_TIM_VIEC,
    OPT_FREQ_2_TO_4,
    OPT_FREQ_4_TO_6,
    OPT_FREQ_6_TO_8,
    OPT_FREQ_OVER_8,
    OPT_FREQ_UNDER_2,
    OPT_PAIN_BUILDING,
    OPT_PAIN_NONE,
    OPT_PAIN_OTHER,
    OPT_PAIN_PRIVACY_LOSS,
    OPT_PROF_CNTT,
    OPT_PROF_FREELANCE_CONTENT,
    OPT_PROF_GIAO_DUC,
    OPT_PROF_KHAC,
    OPT_PROF_KINH_DOANH,
    OPT_PROF_SAN_XUAT,
    OPT_PROF_VIEN_CHUC,
    OPT_PROF_VU_TRANG,
    OPT_USAGE_NEVER,
    OPT_USAGE_TODAY_YESTERDAY,
    PAIN_POINTS_QUESTION_UUID,
    PersonaEngine,
    PROFESSION_QUESTION_UUID,
    REASON_NEVER_QUESTION_UUID,
    RespondentProfile,
    USAGE_QUESTION_UUID,
)
from src.tally_schema import TallySchema


class TestAdversarialEmptyListFuzzing(unittest.TestCase):
    """
    Task 1: Adversarial empty list fuzzing: pass malformed dictionaries,
    empty lists `[]` on every demographic key, all-empty payloads, and
    verify that `CrossConsistencyValidator.validate` never raises unhandled exceptions.
    """

    def setUp(self):
        TallySchema.initialize()

    def test_completely_empty_dictionary(self):
        """Validating `{}` must not crash and should return standard Rule 5 and Rule 6 missing data violations."""
        try:
            violations = CrossConsistencyValidator.validate({})
            self.assertIsInstance(violations, list)
            # Empty categories and empty pain points should be reported as violations
            self.assertTrue(any("Rule 5" in v for v in violations))
            self.assertTrue(any("Rule 6" in v for v in violations))
        except Exception as exc:
            self.fail(f"CrossConsistencyValidator.validate crashed on empty dict {{}}: {exc}")

    def test_all_keys_empty_lists(self):
        """Pass empty lists `[]` for every possible demographic key and candidate alias."""
        all_empty = {
            "age": [],
            AGE_QUESTION_UUID: [],
            "education": [],
            "education_uuid": [],
            EDUCATION_QUESTION_UUID: [],
            "Q7_EDUCATION": [],
            "employment": [],
            "employment_uuid": [],
            EMPLOYMENT_QUESTION_UUID: [],
            "Q8_EMPLOYMENT": [],
            "profession": [],
            "profession_uuid": [],
            PROFESSION_QUESTION_UUID: [],
            "Q9_PROFESSION": [],
            "frequency": [],
            "frequency_uuid": [],
            FREQUENCY_QUESTION_UUID: [],
            "Q10_FREQUENCY": [],
            "categories": [],
            "categories_uuids": [],
            CATEGORIES_QUESTION_UUID: [],
            "Q11_CATEGORIES": [],
            "pain_points": [],
            "pain_points_uuids": [],
            PAIN_POINTS_QUESTION_UUID: [],
            "Q12_PAIN_POINTS": [],
            "gender": [],
            "gender_uuid": [],
            GENDER_QUESTION_UUID: [],
            "location": [],
            "location_uuid": [],
            LOCATION_QUESTION_UUID: [],
            "awareness": [],
            "awareness_uuid": [],
            AWARENESS_QUESTION_UUID: [],
        }
        try:
            violations = CrossConsistencyValidator.validate(all_empty)
            self.assertIsInstance(violations, list)
            self.assertTrue(any("Rule 5" in v for v in violations))
            self.assertTrue(any("Rule 6" in v for v in violations))
        except Exception as exc:
            self.fail(f"CrossConsistencyValidator.validate crashed on all-empty-lists profile: {exc}")

    def test_individual_key_empty_lists(self):
        """Test each demographic key set to `[]` in isolation on an otherwise valid profile."""
        base_valid = {
            "age": 22,
            "education": OPT_EDU_DAI_HOC,
            "employment": OPT_EMP_FULL_TIME,
            "profession": OPT_PROF_KINH_DOANH,
            "frequency": OPT_FREQ_2_TO_4,
            "categories": [OPT_CAT_FASHION],
            "pain_points": [OPT_PAIN_BUILDING],
        }

        keys_to_test = [
            "age",
            AGE_QUESTION_UUID,
            "education",
            "education_uuid",
            EDUCATION_QUESTION_UUID,
            "Q7_EDUCATION",
            "employment",
            "employment_uuid",
            EMPLOYMENT_QUESTION_UUID,
            "Q8_EMPLOYMENT",
            "profession",
            "profession_uuid",
            PROFESSION_QUESTION_UUID,
            "Q9_PROFESSION",
            "frequency",
            "frequency_uuid",
            FREQUENCY_QUESTION_UUID,
            "Q10_FREQUENCY",
            "categories",
            "categories_uuids",
            CATEGORIES_QUESTION_UUID,
            "Q11_CATEGORIES",
            "pain_points",
            "pain_points_uuids",
            PAIN_POINTS_QUESTION_UUID,
            "Q12_PAIN_POINTS",
        ]

        for key in keys_to_test:
            with self.subTest(key=key):
                test_profile = dict(base_valid)
                test_profile[key] = []
                try:
                    violations = CrossConsistencyValidator.validate(test_profile)
                    self.assertIsInstance(violations, list)
                except Exception as exc:
                    self.fail(f"CrossConsistencyValidator.validate crashed when '{key}' was []: {exc}")

    def test_pathological_fuzz_values(self):
        """Test pathological values: None, nested lists, numbers as strings, non-string objects."""
        pathological_cases = [
            # Case 1: None for all fields
            {
                "age": None,
                "education": None,
                "employment": None,
                "profession": None,
                "frequency": None,
                "categories": None,
                "pain_points": None,
            },
            # Case 2: [None] or [None, None]
            {
                "age": [None],
                "education": [None],
                "employment": [None],
                "profession": [None],
                "frequency": [None],
                "categories": [None, None],
                "pain_points": [None],
            },
            # Case 3: Empty strings & whitespace
            {
                "age": "",
                "education": "   ",
                "employment": ["", "  "],
                "profession": " ",
                "frequency": ["  "],
                "categories": ["", "   "],
                "pain_points": ["   "],
            },
            # Case 4: Nested empty lists
            {
                "age": [[]],
                "education": [[]],
                "employment": [[]],
                "profession": [[]],
                "frequency": [[]],
                "categories": [[]],
                "pain_points": [[]],
            },
            # Case 5: Heterogeneous non-string objects
            {
                "age": {"invalid": 123},
                "education": 9999,
                "employment": False,
                "profession": object(),
                "frequency": 3.14159,
                "categories": {1, 2, 3},
                "pain_points": [None, False, 123],
            },
        ]

        for idx, case in enumerate(pathological_cases):
            with self.subTest(case_index=idx):
                try:
                    violations = CrossConsistencyValidator.validate(case)
                    self.assertIsInstance(violations, list)
                except Exception as exc:
                    self.fail(f"CrossConsistencyValidator.validate crashed on pathological case {idx}: {exc}")

    def test_random_fuzzing_1000_iterations(self):
        """
        Adversarial Monte Carlo Fuzzing: 1,000 iterations of random malformed dictionaries.
        Asserts 0 unhandled exceptions.
        """
        candidate_values = [
            None,
            [],
            [[]],
            [None],
            [""],
            ["   "],
            [None, ""],
            {},
            {1: 2},
            123,
            -1,
            0,
            "string_garbage",
            OPT_EDU_SAU_DAI_HOC,
            OPT_EMP_FREELANCER,
            OPT_PROF_VU_TRANG,
            OPT_FREQ_UNDER_2,
            OPT_PAIN_NONE,
            [OPT_CAT_FASHION, None, ""],
            [OPT_PAIN_NONE, OPT_PAIN_BUILDING],
        ]

        keys = [
            "age",
            AGE_QUESTION_UUID,
            "education",
            "education_uuid",
            EDUCATION_QUESTION_UUID,
            "Q7_EDUCATION",
            "employment",
            "employment_uuid",
            EMPLOYMENT_QUESTION_UUID,
            "Q8_EMPLOYMENT",
            "profession",
            "profession_uuid",
            PROFESSION_QUESTION_UUID,
            "Q9_PROFESSION",
            "frequency",
            "frequency_uuid",
            FREQUENCY_QUESTION_UUID,
            "Q10_FREQUENCY",
            "categories",
            "categories_uuids",
            CATEGORIES_QUESTION_UUID,
            "Q11_CATEGORIES",
            "pain_points",
            "pain_points_uuids",
            PAIN_POINTS_QUESTION_UUID,
            "Q12_PAIN_POINTS",
        ]

        rng = random.Random(42)
        for i in range(1000):
            # Generate random subset of keys with random values
            num_keys = rng.randint(0, len(keys))
            selected_keys = rng.sample(keys, num_keys)
            fuzz_dict = {k: rng.choice(candidate_values) for k in selected_keys}

            try:
                violations = CrossConsistencyValidator.validate(fuzz_dict)
                self.assertIsInstance(violations, list)
            except Exception as exc:
                self.fail(
                    f"CrossConsistencyValidator.validate raised unhandled exception on fuzz iteration {i}: {exc}\n"
                    f"Fuzz dictionary: {fuzz_dict}"
                )


class TestStressTest500Profiles(unittest.TestCase):
    """
    Task 2: Stress test 500 generated profiles: verify 0 cross-consistency violations across all 6 rules.
    """

    @classmethod
    def setUpClass(cls):
        TallySchema.initialize()
        cls.engine = PersonaEngine(seed=2026)
        cls.num_profiles = 500
        cls.profiles = [cls.engine.generate_respondent() for _ in range(cls.num_profiles)]

    def test_generated_count(self):
        self.assertEqual(len(self.profiles), 500)

    def test_validator_zero_violations_all_500(self):
        """CrossConsistencyValidator must return exactly 0 violations for all 500 profiles."""
        all_violations = []
        for idx, p in enumerate(self.profiles):
            violations = CrossConsistencyValidator.validate(p)
            if violations:
                all_violations.append((idx, violations))
        self.assertEqual(
            len(all_violations),
            0,
            f"Observed {len(all_violations)} profiles with cross-consistency violations: {all_violations[:5]}",
        )

    def test_rule_1_and_2_demographic_age_vs_postgraduate(self):
        """Rule 1 & 2: Age < 22 cannot have Postgraduate degree (OPT_EDU_SAU_DAI_HOC)."""
        violations = []
        for idx, p in enumerate(self.profiles):
            if p.age < 22 and p.education == OPT_EDU_SAU_DAI_HOC:
                violations.append(f"Sample {idx}: Age {p.age} (< 22) has Sau Dai Hoc degree")
            if p.age <= 20 and p.education == OPT_EDU_SAU_DAI_HOC:
                violations.append(f"Sample {idx}: Age {p.age} (<= 20) has Sau Dai Hoc degree")
        self.assertEqual(len(violations), 0, f"Rule 1/2 violations: {violations}")

    def test_rule_2_employment_distribution_proportions(self):
        """Rule 2: Age <= 20 full-time <= 10%, Age >= 24 non-labor <= 5%."""
        young_fulltime = 0
        young_total = 0
        older_nonlabor = 0
        older_total = 0

        for p in self.profiles:
            if p.age <= 20:
                young_total += 1
                if p.employment == OPT_EMP_FULL_TIME:
                    young_fulltime += 1
            if p.age >= 24:
                older_total += 1
                if p.employment == OPT_EMP_KHONG_LAO_DONG:
                    older_nonlabor += 1

        self.assertGreater(young_total, 0, "Expected at least one respondent age <= 20 in 500 samples")
        self.assertGreater(older_total, 0, "Expected at least one respondent age >= 24 in 500 samples")

        rate_young = young_fulltime / young_total
        tol_young = 3.0 * math.sqrt(0.10 * 0.90 / young_total)
        max_allowed_young = 0.10 + tol_young
        self.assertLessEqual(
            rate_young,
            max_allowed_young,
            f"Full-time rate for age <= 20 ({rate_young:.2%}, {young_fulltime}/{young_total}) "
            f"exceeds 3-sigma upper threshold ({max_allowed_young:.2%})",
        )

        rate_older = older_nonlabor / older_total
        tol_older = 3.0 * math.sqrt(0.05 * 0.95 / older_total)
        max_allowed_older = 0.05 + tol_older
        self.assertLessEqual(
            rate_older,
            max_allowed_older,
            f"Non-labor rate for age >= 24 ({rate_older:.2%}, {older_nonlabor}/{older_total}) "
            f"exceeds 3-sigma upper threshold ({max_allowed_older:.2%})",
        )

    def test_rule_3_state_occupations_vs_employment(self):
        """Rule 3: Civil Servant / Armed Forces requires Full-Time; Freelancer cannot hold state post."""
        state_professions = {OPT_PROF_VIEN_CHUC, OPT_PROF_VU_TRANG}
        violations = []
        for idx, p in enumerate(self.profiles):
            if p.profession in state_professions and p.employment != OPT_EMP_FULL_TIME:
                violations.append(f"Sample {idx}: State profession ({p.profession}) not Full-Time ({p.employment})")
            if p.employment == OPT_EMP_FREELANCER and p.profession in state_professions:
                violations.append(f"Sample {idx}: Freelancer holds state post ({p.profession})")
        self.assertEqual(len(violations), 0, f"Rule 3 violations: {violations}")

    def test_rule_4_inactive_employment_vs_profession(self):
        """Rule 4: Inactive employment (Seeking job / Non-labor) cannot be Civil Servant, Armed Forces, or Production."""
        inactive_employments = {OPT_EMP_TIM_VIEC, OPT_EMP_KHONG_LAO_DONG}
        forbidden_professions = {OPT_PROF_VIEN_CHUC, OPT_PROF_VU_TRANG, OPT_PROF_SAN_XUAT}
        violations = []
        for idx, p in enumerate(self.profiles):
            if p.employment in inactive_employments and p.profession in forbidden_professions:
                violations.append(
                    f"Sample {idx}: Inactive ({p.employment}) holds forbidden profession ({p.profession})"
                )
        self.assertEqual(len(violations), 0, f"Rule 4 violations: {violations}")

    def test_rule_5_frequency_and_categories(self):
        """
        Rule 5: Categories 1 to 4;
        Infrequent shopper (< 2 orders/mo) max 2 categories;
        Frequent shopper (> 8 orders/mo) min 2 categories;
        Never empty.
        """
        violations = []
        for idx, p in enumerate(self.profiles):
            cats = p.categories
            freq = p.frequency
            if not cats or len(cats) == 0:
                violations.append(f"Sample {idx}: Empty categories")
            if len(cats) > 4:
                violations.append(f"Sample {idx}: Categories count {len(cats)} > 4")
            if freq == OPT_FREQ_UNDER_2 and len(cats) > 2:
                violations.append(f"Sample {idx}: Infrequent shopper has {len(cats)} categories (> 2)")
            if freq == OPT_FREQ_OVER_8 and len(cats) < 2:
                violations.append(f"Sample {idx}: Frequent shopper has {len(cats)} categories (< 2)")
        self.assertEqual(len(violations), 0, f"Rule 5 violations: {violations}")

    def test_rule_6_pain_points_mutual_exclusion(self):
        """Rule 6: 'No pain point' (OPT_PAIN_NONE) is strictly mutually exclusive with all other options."""
        violations = []
        for idx, p in enumerate(self.profiles):
            pains = p.pain_points
            if not pains or len(pains) == 0:
                violations.append(f"Sample {idx}: Empty pain points")
            if OPT_PAIN_NONE in pains and len(pains) > 1:
                violations.append(f"Sample {idx}: OPT_PAIN_NONE chosen alongside others: {pains}")
        self.assertEqual(len(violations), 0, f"Rule 6 violations: {violations}")

    def test_gen_z_age_range_strictly_18_to_27(self):
        """All 500 respondents must have age in [18, 27]."""
        for idx, p in enumerate(self.profiles):
            self.assertTrue(18 <= p.age <= 27, f"Sample {idx}: Age {p.age} out of range [18, 27]")

    def test_screening_logic_invariance(self):
        """100% consent, qualified usage (never USAGE_NEVER), and no Q3 reason in payload."""
        for idx, p in enumerate(self.profiles):
            consent = p.payload_responses.get(CONSENT_QUESTION_UUID)
            self.assertEqual(consent, [OPT_CONSENT_AGREE], f"Sample {idx} consent mismatch")
            usage = p.payload_responses.get(USAGE_QUESTION_UUID)
            self.assertNotEqual(usage, [OPT_USAGE_NEVER], f"Sample {idx} selected Never Shopee")
            self.assertNotIn(REASON_NEVER_QUESTION_UUID, p.payload_responses, f"Sample {idx} triggered Q3")


class TestValidateCanonicalPayloadResponses(unittest.TestCase):
    """
    Task 3: Validate canonical `payload_responses` directly from `PersonaEngine().generate_respondent()`
    and assert 0 false-positive violations.
    """

    @classmethod
    def setUpClass(cls):
        TallySchema.initialize()
        cls.engine = PersonaEngine(seed=777)
        cls.profiles = [cls.engine.generate_respondent() for _ in range(500)]

    def test_payload_responses_zero_false_positive_violations(self):
        """
        Validate `p.payload_responses` directly via CrossConsistencyValidator.validate(p.payload_responses).
        Must produce 0 false-positive violations across all 500 profiles.
        """
        total_false_positives = 0
        details = []

        for idx, p in enumerate(self.profiles):
            payload_resp = p.payload_responses
            self.assertIsInstance(payload_resp, dict, f"Sample {idx} payload_responses is not a dict")

            violations = CrossConsistencyValidator.validate(payload_resp)
            if violations:
                total_false_positives += len(violations)
                details.append((idx, violations))

        self.assertEqual(
            total_false_positives,
            0,
            f"Observed {total_false_positives} false-positive violations across payload_responses: {details[:5]}",
        )

    def test_all_demographic_representations_produce_zero_violations(self):
        """
        Verify all canonical demographic representation dictionaries produce 0 violations:
        1. p (RespondentProfile instance)
        2. p.payload_responses (canonical dictionary mapped with group UUIDs)
        3. p.demographics (dictionary with _uuid suffixed keys)
        4. p.demographics_raw (dictionary with raw keys: age, education, etc.)
        5. p.demographics_uuids supplemented with age
        """
        for idx, p in enumerate(self.profiles[:100]):  # Test first 100 profiles across representations
            # Form 1: RespondentProfile object
            v1 = CrossConsistencyValidator.validate(p)
            self.assertEqual(v1, [], f"Sample {idx} failed as RespondentProfile: {v1}")

            # Form 2: payload_responses (canonical contract)
            v2 = CrossConsistencyValidator.validate(p.payload_responses)
            self.assertEqual(v2, [], f"Sample {idx} failed as payload_responses: {v2}")

            # Form 3: demographics (canonical contract)
            v3 = CrossConsistencyValidator.validate(p.demographics)
            self.assertEqual(v3, [], f"Sample {idx} failed as demographics: {v3}")

            # Form 4: demographics_raw
            v4 = CrossConsistencyValidator.validate(p.demographics_raw)
            self.assertEqual(v4, [], f"Sample {idx} failed as demographics_raw: {v4}")

            # Form 5: demographics_uuids with age (since Q4 Age is numeric, not option UUID)
            v5 = CrossConsistencyValidator.validate({**p.demographics_uuids, "age": p.age})
            self.assertEqual(v5, [], f"Sample {idx} failed as demographics_uuids with age: {v5}")

    def test_payload_schema_compliance(self):
        """
        Verify that TallySchema.get_payload(p.payload_responses) passes TallySchema.validate_payload()
        for all 500 generated profiles.
        """
        for idx, p in enumerate(self.profiles):
            full_payload = TallySchema.get_payload(p.payload_responses)
            is_valid = TallySchema.validate_payload(full_payload, raise_exception=False)
            if not is_valid:
                errors = TallySchema.get_validation_errors(full_payload)
                self.fail(f"Sample {idx} failed TallySchema validation:\n" + "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
