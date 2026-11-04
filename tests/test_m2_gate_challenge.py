"""
tests/test_m2_gate_challenge.py
Authoritative Empirical Stress-Test & Adversarial Challenge Suite for Milestone M2.
Evaluates Persona Engine (src/persona_engine.py), TallySchema validation,
and CrossConsistencyValidator under heavy load and adversarial fuzzing.

Challenge dimensions:
1. 500 consecutive profile generations with 0 cross-consistency violations across all 6 rules.
2. 100% of 500 generated payloads pass TallySchema.validate_payload().
3. Adversarial fuzzing of CrossConsistencyValidator across edge cases, mutations, and boundary conditions.
4. Statistical distribution audits (Age bounds [18, 27], gender balance, employment rates, screening invariance).
"""

from __future__ import annotations

import itertools
import random
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.persona_engine import (
    CrossConsistencyValidator,
    OPT_AWARE_NEVER,
    OPT_AWARE_OTHER,
    OPT_AWARE_SEEN_UNUSED,
    OPT_AWARE_USED_ONCE,
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
    OPT_GENDER_KHAC,
    OPT_GENDER_NAM,
    OPT_GENDER_NU,
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
    OPT_PROF_KHAC,
    OPT_PROF_KINH_DOANH,
    OPT_PROF_SAN_XUAT,
    OPT_PROF_VIEN_CHUC,
    OPT_PROF_VU_TRANG,
    OPT_USAGE_1M_TO_6M,
    OPT_USAGE_2W_TO_1M,
    OPT_USAGE_NEVER,
    OPT_USAGE_OVER_6M,
    OPT_USAGE_TODAY_YESTERDAY,
    OPT_USAGE_WITHIN_1_WEEK,
    PersonaEngine,
    REASON_NEVER_QUESTION_UUID,
    RespondentProfile,
)
from src.tally_schema import (
    AGE_QUESTION_UUID,
    CONSENT_AGREE_UUID,
    CONSENT_QUESTION_UUID,
    TallySchema,
    USAGE_NEVER_UUID,
    USAGE_QUALIFIED_UUIDS,
    USAGE_QUESTION_UUID,
    UUID_V4_PATTERN,
)


class TestPersonaEngine500Profiles(unittest.TestCase):
    """
    Stress-test Persona Engine with 500 consecutive profile generations.
    Verifies 0 cross-consistency violations across all 6 logical rules,
    demographic parameter compliance, and screening invariants.
    """

    @classmethod
    def setUpClass(cls):
        TallySchema.initialize()
        cls.engine = PersonaEngine(seed=42)
        # Generate exactly 500 consecutive profiles
        cls.profiles: List[RespondentProfile] = [
            cls.engine.generate_respondent() for _ in range(500)
        ]

    def test_01_sample_count_and_types(self):
        """Assert exactly 500 valid RespondentProfile instances are generated."""
        self.assertEqual(len(self.profiles), 500)
        for idx, p in enumerate(self.profiles):
            self.assertIsInstance(p, RespondentProfile, f"Profile {idx} is not RespondentProfile")
            self.assertEqual(len(p.likert_scores), 39, f"Profile {idx} has {len(p.likert_scores)} likert items, expected 39")
            self.assertEqual(len(p.payload_responses), 51, f"Profile {idx} has {len(p.payload_responses)} responses, expected 51")

    def test_02_zero_cross_consistency_violations_all_500(self):
        """Assert strictly 0 cross-consistency violations across all 500 profiles."""
        total_violations: List[str] = []
        for idx, p in enumerate(self.profiles):
            violations = CrossConsistencyValidator.validate(p)
            if violations:
                total_violations.extend([f"Profile #{idx}: {v}" for v in violations])

        self.assertEqual(
            len(total_violations),
            0,
            f"Observed {len(total_violations)} cross-consistency violations:\n" + "\n".join(total_violations[:10]),
        )

    def test_03_demographics_age_bounds_gen_z(self):
        """Assert all 500 respondents have age strictly in [18, 27]."""
        ages = [p.age for p in self.profiles]
        min_age, max_age = min(ages), max(ages)
        mean_age = sum(ages) / len(ages)

        self.assertGreaterEqual(min_age, 18, f"Min age {min_age} < 18")
        self.assertLessEqual(max_age, 27, f"Max age {max_age} > 27")
        self.assertTrue(20.5 <= mean_age <= 22.5, f"Mean age {mean_age:.2f} drifted from expected ~21.5")

    def test_04_rule_1_and_2_age_vs_postgrad_education(self):
        """Rule 1 & 2: Age < 22 cannot hold Postgraduate degree (OPT_EDU_SAU_DAI_HOC)."""
        underage_postgrad = [
            idx for idx, p in enumerate(self.profiles)
            if p.age < 22 and p.education == OPT_EDU_SAU_DAI_HOC
        ]
        self.assertEqual(
            len(underage_postgrad),
            0,
            f"Found {len(underage_postgrad)} respondents with age < 22 holding Sau Dai Hoc: {underage_postgrad}",
        )

    def test_05_rule_2_employment_rates_distribution(self):
        """Rule 2: Age <= 20 full-time <= 10%, Age >= 24 non-labor <= 5% (with 3-sigma binomial confidence bounds)."""
        young_total = [p for p in self.profiles if p.age <= 20]
        young_fulltime = [p for p in young_total if p.employment == OPT_EMP_FULL_TIME]

        older_total = [p for p in self.profiles if p.age >= 24]
        older_nonlabor = [p for p in older_total if p.employment == OPT_EMP_KHONG_LAO_DONG]

        if young_total:
            rate_young = len(young_fulltime) / len(young_total)
            tol_young = 3.0 * (0.10 * 0.90 / len(young_total)) ** 0.5
            self.assertLessEqual(
                rate_young,
                0.10 + tol_young,
                f"Full-time rate for age <= 20 ({rate_young:.2%}, {len(young_fulltime)}/{len(young_total)}) exceeds 3-sigma threshold ({0.10 + tol_young:.2%})",
            )

        if older_total:
            rate_older = len(older_nonlabor) / len(older_total)
            tol_older = 3.0 * (0.05 * 0.95 / len(older_total)) ** 0.5
            self.assertLessEqual(
                rate_older,
                0.05 + tol_older,
                f"Non-labor rate for age >= 24 ({rate_older:.2%}, {len(older_nonlabor)}/{len(older_total)}) exceeds 3-sigma threshold ({0.05 + tol_older:.2%})",
            )

    def test_06_rule_3_state_occupations_require_full_time(self):
        """Rule 3: Civil Servant / Armed Forces requires Full-time; Freelancer cannot hold state post."""
        state_professions = {OPT_PROF_VIEN_CHUC, OPT_PROF_VU_TRANG}
        violations = []
        for idx, p in enumerate(self.profiles):
            if p.profession in state_professions and p.employment != OPT_EMP_FULL_TIME:
                violations.append(f"Profile {idx}: state profession ({p.profession}) with non-fulltime ({p.employment})")
            if p.employment == OPT_EMP_FREELANCER and p.profession in state_professions:
                violations.append(f"Profile {idx}: freelancer with state profession ({p.profession})")

        self.assertEqual(len(violations), 0, f"Rule 3 Violations: {violations}")

    def test_07_rule_4_inactive_employment_restrictions(self):
        """Rule 4: Inactive employment (Seeking Job / Non-labor) cannot be Civil Servant, Armed Forces, or Production."""
        inactive_employments = {OPT_EMP_TIM_VIEC, OPT_EMP_KHONG_LAO_DONG}
        forbidden_professions = {OPT_PROF_VIEN_CHUC, OPT_PROF_VU_TRANG, OPT_PROF_SAN_XUAT}
        violations = []
        for idx, p in enumerate(self.profiles):
            if p.employment in inactive_employments and p.profession in forbidden_professions:
                violations.append(f"Profile {idx}: inactive ({p.employment}) with forbidden profession ({p.profession})")

        self.assertEqual(len(violations), 0, f"Rule 4 Violations: {violations}")

    def test_08_rule_5_frequency_vs_categories(self):
        """Rule 5: Categories 1..3; <2 orders/mo max 2 cats; >8 orders/mo min 2 cats."""
        violations = []
        for idx, p in enumerate(self.profiles):
            cats = p.categories
            freq = p.frequency
            if not cats or len(cats) == 0:
                violations.append(f"Profile {idx}: empty categories")
            if freq == OPT_FREQ_UNDER_2 and len(cats) > 2:
                violations.append(f"Profile {idx}: freq < 2 with {len(cats)} categories (> 2)")
            if freq == OPT_FREQ_OVER_8 and len(cats) < 2:
                violations.append(f"Profile {idx}: freq > 8 with {len(cats)} categories (< 2)")

        self.assertEqual(len(violations), 0, f"Rule 5 Violations: {violations}")

    def test_09_rule_6_pain_points_mutual_exclusion(self):
        """Rule 6: 'Khong gap bat tien nao' is mutually exclusive with any other options."""
        violations = []
        for idx, p in enumerate(self.profiles):
            pains = p.pain_points
            if not pains or len(pains) == 0:
                violations.append(f"Profile {idx}: empty pain points")
            if OPT_PAIN_NONE in pains and len(pains) > 1:
                violations.append(f"Profile {idx}: OPT_PAIN_NONE chosen alongside others: {pains}")

        self.assertEqual(len(violations), 0, f"Rule 6 Violations: {violations}")

    def test_10_screening_logic_invariance(self):
        """Screening logic: 100% consent agreed, qualified usage, no Q3 reason."""
        for idx, p in enumerate(self.profiles):
            consent = p.payload_responses.get(CONSENT_QUESTION_UUID)
            self.assertEqual(consent, [CONSENT_AGREE_UUID], f"Profile {idx} consent disagreement")

            usage = p.payload_responses.get(USAGE_QUESTION_UUID)
            self.assertIsInstance(usage, list)
            self.assertEqual(len(usage), 1)
            self.assertIn(usage[0], USAGE_QUALIFIED_UUIDS, f"Profile {idx} usage not in qualified list: {usage}")
            self.assertNotEqual(usage[0], USAGE_NEVER_UUID, f"Profile {idx} selected Never Shopee")

            self.assertNotIn(
                REASON_NEVER_QUESTION_UUID,
                p.payload_responses,
                f"Profile {idx} triggered disqualifying Q3 reason question",
            )


class TestTallySchemaPayloadValidation500(unittest.TestCase):
    """
    Validates each of the 500 generated payloads against TallySchema.validate_payload().
    Verifies envelope structure, distinct RFC 4122 v4 UUIDs, question/option membership,
    and 39 Likert items.
    """

    @classmethod
    def setUpClass(cls):
        TallySchema.initialize()
        engine = PersonaEngine()
        cls.profiles = [engine.generate_respondent() for _ in range(500)]
        cls.payloads = [TallySchema.get_payload(p.payload_responses) for p in cls.profiles]

    def test_01_all_500_payloads_valid_against_schema(self):
        """Assert 100% of the 500 payloads pass TallySchema.validate_payload()."""
        failures = []
        for idx, payload in enumerate(self.payloads):
            is_valid = TallySchema.validate_payload(payload, raise_exception=False)
            if not is_valid:
                errors = TallySchema.get_validation_errors(payload)
                failures.append(f"Payload #{idx} failed: {errors}")

        self.assertEqual(
            len(failures),
            0,
            f"{len(failures)} / 500 payloads failed TallySchema validation:\n" + "\n".join(failures[:10]),
        )

    def test_02_session_and_respondent_uuid_rfc4122_uniqueness(self):
        """Assert sessionUuid and respondentUuid are distinct, valid v4 UUIDs across 500 profiles."""
        session_uuids: Set[str] = set()
        respondent_uuids: Set[str] = set()

        for idx, payload in enumerate(self.payloads):
            s_uuid = payload.get("sessionUuid")
            r_uuid = payload.get("respondentUuid")

            self.assertIsNotNone(s_uuid)
            self.assertIsNotNone(r_uuid)
            self.assertTrue(UUID_V4_PATTERN.match(s_uuid), f"Payload {idx} sessionUuid {s_uuid} not UUIDv4")
            self.assertTrue(UUID_V4_PATTERN.match(r_uuid), f"Payload {idx} respondentUuid {r_uuid} not UUIDv4")
            self.assertNotEqual(s_uuid.lower(), r_uuid.lower(), f"Payload {idx} sessionUuid == respondentUuid")

            session_uuids.add(s_uuid)
            respondent_uuids.add(r_uuid)

        self.assertEqual(len(session_uuids), 500, "Duplicate sessionUuid detected among 500 payloads")
        self.assertEqual(len(respondent_uuids), 500, "Duplicate respondentUuid detected among 500 payloads")
        self.assertEqual(
            len(session_uuids.intersection(respondent_uuids)),
            0,
            "Collision between sessionUuid and respondentUuid sets",
        )

    def test_03_all_group_and_option_uuids_in_canonical_schema(self):
        """Assert all question group UUIDs and option UUIDs exist in canonical schema."""
        for idx, payload in enumerate(self.payloads):
            responses = payload["responses"]
            for g_uuid, val in responses.items():
                self.assertIn(
                    g_uuid,
                    TallySchema.ALL_GROUP_UUIDS,
                    f"Payload {idx} has unrecognized groupUuid: {g_uuid}",
                )
                if isinstance(val, list):
                    for opt_uuid in val:
                        self.assertIn(
                            opt_uuid,
                            TallySchema.ALL_OPTION_UUIDS,
                            f"Payload {idx} has unrecognized optionUuid: {opt_uuid}",
                        )

    def test_04_likert_39_items_integrity(self):
        """Assert all 39 Likert questions exist in payload and options map to scores 1..6."""
        for idx, payload in enumerate(self.payloads):
            responses = payload["responses"]
            for q_def in TallySchema.LIKERT_QUESTIONS:
                self.assertIn(
                    q_def.group_uuid,
                    responses,
                    f"Payload {idx} missing Likert question {q_def.code}",
                )
                val = responses[q_def.group_uuid]
                self.assertIsInstance(val, list)
                self.assertEqual(len(val), 1)
                opt_uuid = val[0]
                self.assertIn(opt_uuid, q_def.option_uuids)
                score = q_def.uuid_to_score[opt_uuid]
                self.assertTrue(1 <= score <= 6, f"Payload {idx} {q_def.code} score {score} out of [1, 6]")


class TestCrossConsistencyAdversarialFuzzing(unittest.TestCase):
    """
    Adversarial fuzzing battery against CrossConsistencyValidator.
    Injects systematic mutations across each rule to verify 100% detection rate.
    """

    def setUp(self):
        self.base_valid_profile: Dict[str, Any] = {
            "age": 22,
            "education": OPT_EDU_DAI_HOC,
            "employment": OPT_EMP_FULL_TIME,
            "profession": OPT_PROF_KINH_DOANH,
            "frequency": OPT_FREQ_4_TO_6,
            "categories": [OPT_CAT_FASHION, OPT_CAT_BEAUTY],
            "pain_points": [OPT_PAIN_BUILDING],
        }
        # Baseline must have 0 violations
        self.assertEqual(len(CrossConsistencyValidator.validate(self.base_valid_profile)), 0)

    def test_01_fuzz_rule_1_and_2_underage_postgrad(self):
        """Fuzz ages 18..21 with Sau Dai Hoc: must ALWAYS be detected."""
        for bad_age in [18, 19, 20, 21]:
            p = dict(self.base_valid_profile, age=bad_age, education=OPT_EDU_SAU_DAI_HOC)
            violations = CrossConsistencyValidator.validate(p)
            self.assertTrue(
                any("Rule 1" in v or "Rule 2" in v for v in violations),
                f"Failed to detect underage postgrad at age {bad_age}",
            )

        # Non-violations: Age >= 22 with Sau Dai Hoc
        for good_age in [22, 23, 24, 25, 26, 27]:
            p = dict(self.base_valid_profile, age=good_age, education=OPT_EDU_SAU_DAI_HOC)
            violations = CrossConsistencyValidator.validate(p)
            rule_1_2_violations = [v for v in violations if "Rule 1" in v or "Rule 2" in v]
            self.assertEqual(
                len(rule_1_2_violations),
                0,
                f"False positive for age {good_age} with Sau Dai Hoc: {rule_1_2_violations}",
            )

    def test_02_fuzz_rule_3_state_occupations_and_freelancers(self):
        """Fuzz civil servant & armed forces with non-fulltime employments."""
        state_professions = [OPT_PROF_VIEN_CHUC, OPT_PROF_VU_TRANG]
        non_fulltime_employments = [
            OPT_EMP_PART_TIME,
            OPT_EMP_FREELANCER,
            OPT_EMP_TIM_VIEC,
            OPT_EMP_KHONG_LAO_DONG,
            OPT_EMP_KHAC,
        ]

        for prof in state_professions:
            for emp in non_fulltime_employments:
                p = dict(self.base_valid_profile, profession=prof, employment=emp)
                violations = CrossConsistencyValidator.validate(p)
                self.assertTrue(
                    any("Rule 3" in v for v in violations),
                    f"Failed to detect state profession {prof} with non-fulltime {emp}",
                )

        # Full-time with state professions must NOT violate Rule 3
        for prof in state_professions:
            p = dict(self.base_valid_profile, profession=prof, employment=OPT_EMP_FULL_TIME)
            violations = CrossConsistencyValidator.validate(p)
            rule_3_violations = [v for v in violations if "Rule 3" in v]
            self.assertEqual(len(rule_3_violations), 0, f"False positive for Rule 3: {rule_3_violations}")

    def test_03_fuzz_rule_4_inactive_employment_professions(self):
        """Fuzz inactive employments with forbidden active careers."""
        inactive_employments = [OPT_EMP_TIM_VIEC, OPT_EMP_KHONG_LAO_DONG]
        forbidden_professions = [OPT_PROF_VIEN_CHUC, OPT_PROF_VU_TRANG, OPT_PROF_SAN_XUAT]
        allowed_professions = [
            OPT_PROF_KINH_DOANH,
            OPT_PROF_CNTT,
            OPT_PROF_GIAO_DUC,
            OPT_PROF_FREELANCE_CONTENT,
            OPT_PROF_KHAC,
        ]

        for emp in inactive_employments:
            for prof in forbidden_professions:
                p = dict(self.base_valid_profile, employment=emp, profession=prof)
                violations = CrossConsistencyValidator.validate(p)
                self.assertTrue(
                    any("Rule 4" in v for v in violations),
                    f"Failed to detect inactive {emp} with forbidden profession {prof}",
                )

        for emp in inactive_employments:
            for prof in allowed_professions:
                p = dict(self.base_valid_profile, employment=emp, profession=prof)
                violations = CrossConsistencyValidator.validate(p)
                rule_4_violations = [v for v in violations if "Rule 4" in v]
                self.assertEqual(len(rule_4_violations), 0, f"False positive for Rule 4: {rule_4_violations}")

    def test_04_fuzz_rule_5_frequency_vs_categories(self):
        """Fuzz categories and shopping frequency constraints."""
        all_cats = [OPT_CAT_FASHION, OPT_CAT_BEAUTY, OPT_CAT_TECH, OPT_CAT_BOOKS, OPT_CAT_HOUSEHOLD]

        # 1. Empty categories must always fail
        p = dict(self.base_valid_profile, categories=[])
        violations = CrossConsistencyValidator.validate(p)
        self.assertTrue(any("Rule 5" in v for v in violations), "Failed to detect empty categories")

        # 2. Infrequent shopper (<2 orders/mo) with > 2 categories must fail
        for n_cats in [3, 4, 5]:
            p = dict(self.base_valid_profile, frequency=OPT_FREQ_UNDER_2, categories=all_cats[:n_cats])
            violations = CrossConsistencyValidator.validate(p)
            self.assertTrue(
                any("Rule 5" in v for v in violations),
                f"Failed to detect infrequent shopper with {n_cats} categories",
            )

        # 3. Infrequent shopper with 1 or 2 categories must pass
        for n_cats in [1, 2]:
            p = dict(self.base_valid_profile, frequency=OPT_FREQ_UNDER_2, categories=all_cats[:n_cats])
            violations = CrossConsistencyValidator.validate(p)
            rule_5_violations = [v for v in violations if "Rule 5" in v]
            self.assertEqual(len(rule_5_violations), 0, f"False positive for Rule 5 infrequent: {rule_5_violations}")

        # 4. Frequent shopper (>8 orders/mo) with < 2 categories must fail
        p = dict(self.base_valid_profile, frequency=OPT_FREQ_OVER_8, categories=[OPT_CAT_FASHION])
        violations = CrossConsistencyValidator.validate(p)
        self.assertTrue(
            any("Rule 5" in v for v in violations),
            "Failed to detect frequent shopper with 1 category",
        )

        # 5. Frequent shopper with >= 2 categories must pass
        for n_cats in [2, 3, 4]:
            p = dict(self.base_valid_profile, frequency=OPT_FREQ_OVER_8, categories=all_cats[:n_cats])
            violations = CrossConsistencyValidator.validate(p)
            rule_5_violations = [v for v in violations if "Rule 5" in v]
            self.assertEqual(len(rule_5_violations), 0, f"False positive for Rule 5 frequent: {rule_5_violations}")

    def test_05_fuzz_rule_6_pain_points_exclusivity(self):
        """Fuzz delivery pain points: empty selection and mutual exclusion of OPT_PAIN_NONE."""
        # 1. Empty pain points must fail
        p = dict(self.base_valid_profile, pain_points=[])
        violations = CrossConsistencyValidator.validate(p)
        self.assertTrue(any("Rule 6" in v for v in violations), "Failed to detect empty pain points")

        # 2. OPT_PAIN_NONE alone must pass
        p = dict(self.base_valid_profile, pain_points=[OPT_PAIN_NONE])
        violations = CrossConsistencyValidator.validate(p)
        rule_6_violations = [v for v in violations if "Rule 6" in v]
        self.assertEqual(len(rule_6_violations), 0, f"False positive on lone OPT_PAIN_NONE: {rule_6_violations}")

        # 3. OPT_PAIN_NONE combined with any other option must fail
        other_pains = [OPT_PAIN_BUILDING, OPT_PAIN_PRIVACY_LOSS, OPT_PAIN_OTHER]
        for opt in other_pains:
            p = dict(self.base_valid_profile, pain_points=[OPT_PAIN_NONE, opt])
            violations = CrossConsistencyValidator.validate(p)
            self.assertTrue(
                any("Rule 6" in v for v in violations),
                f"Failed to detect contradictory pain point combination: {[OPT_PAIN_NONE, opt]}",
            )

        # 4. Combination of 2 non-NONE pain points must pass
        p = dict(self.base_valid_profile, pain_points=[OPT_PAIN_BUILDING, OPT_PAIN_PRIVACY_LOSS])
        violations = CrossConsistencyValidator.validate(p)
        rule_6_violations = [v for v in violations if "Rule 6" in v]
        self.assertEqual(len(rule_6_violations), 0, f"False positive on multiple non-NONE pains: {rule_6_violations}")

    def test_06_combinatorial_adversarial_mutations(self):
        """
        Generate 200 adversarial profiles with known combinations of injected violations (1 to 5).
        Assert 100% detection rate across all injected violation types.
        """
        rng = random.Random(42)
        total_injected = 0
        total_detected = 0

        for _ in range(200):
            p = dict(self.base_valid_profile)
            injected_rules = set()

            # Randomly decide which rules to inject across orthogonal dimensions:
            # Dim 1: Rule 1 (Age vs Education)
            # Dim 2: Rule 3 OR Rule 4 (Employment vs Profession)
            # Dim 3: Rule 5 (Frequency vs Categories)
            # Dim 4: Rule 6 (Pain points)
            inject_r1 = rng.random() < 0.4
            emp_prof_choice = rng.choice(["none", "rule3", "rule4"])
            inject_r5 = rng.random() < 0.4
            inject_r6 = rng.random() < 0.4

            # Ensure at least one rule is injected
            if not inject_r1 and emp_prof_choice == "none" and not inject_r5 and not inject_r6:
                inject_r1 = True

            if inject_r1:
                p["age"] = rng.choice([18, 19, 20, 21])
                p["education"] = OPT_EDU_SAU_DAI_HOC
                if p["age"] <= 20:
                    injected_rules.add("Rule 1")
                else:
                    injected_rules.add("Rule 2")

            if emp_prof_choice == "rule3":
                p["employment"] = OPT_EMP_FREELANCER
                p["profession"] = rng.choice([OPT_PROF_VIEN_CHUC, OPT_PROF_VU_TRANG])
                injected_rules.add("Rule 3")
            elif emp_prof_choice == "rule4":
                p["employment"] = rng.choice([OPT_EMP_TIM_VIEC, OPT_EMP_KHONG_LAO_DONG])
                p["profession"] = OPT_PROF_SAN_XUAT
                injected_rules.add("Rule 4")

            if inject_r5:
                p["frequency"] = OPT_FREQ_UNDER_2
                p["categories"] = [OPT_CAT_FASHION, OPT_CAT_BEAUTY, OPT_CAT_TECH, OPT_CAT_BOOKS]
                injected_rules.add("Rule 5")

            if inject_r6:
                p["pain_points"] = [OPT_PAIN_NONE, OPT_PAIN_BUILDING]
                injected_rules.add("Rule 6")

            violations = CrossConsistencyValidator.validate(p)

            for rule in injected_rules:
                total_injected += 1
                if any(rule in v for v in violations):
                    total_detected += 1
                else:
                    self.fail(f"Failed to detect {rule} in mutated profile: {p}\nActual violations returned: {violations}")

        detection_rate = total_detected / total_injected
        self.assertEqual(
            detection_rate,
            1.0,
            f"Adversarial detection rate {detection_rate:.2%} did not achieve 100% ({total_detected}/{total_injected})",
        )

    def test_07_input_format_resilience(self):
        """Verify validator handles RespondentProfile dataclass, plain dicts, and _uuid suffix dicts."""
        # 1. Plain dict
        plain_dict = dict(self.base_valid_profile)
        self.assertEqual(len(CrossConsistencyValidator.validate(plain_dict)), 0)

        # 2. _uuid suffix dict
        uuid_dict = {
            "age": 22,
            "education_uuid": OPT_EDU_DAI_HOC,
            "employment_uuid": OPT_EMP_FULL_TIME,
            "profession_uuid": OPT_PROF_KINH_DOANH,
            "frequency_uuid": OPT_FREQ_4_TO_6,
            "categories_uuids": [OPT_CAT_FASHION],
            "pain_points_uuids": [OPT_PAIN_BUILDING],
        }
        self.assertEqual(len(CrossConsistencyValidator.validate(uuid_dict)), 0)

        # 3. None resilience for categories and pain points
        none_dict = {
            "age": 22,
            "education": OPT_EDU_DAI_HOC,
            "employment": OPT_EMP_FULL_TIME,
            "profession": OPT_PROF_KINH_DOANH,
            "frequency": OPT_FREQ_4_TO_6,
            "categories": None,
            "pain_points": None,
        }
        v = CrossConsistencyValidator.validate(none_dict)
        self.assertTrue(any("Rule 5" in s for s in v))
        self.assertTrue(any("Rule 6" in s for s in v))


if __name__ == "__main__":
    unittest.main()
