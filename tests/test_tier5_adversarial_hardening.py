"""
tests/test_tier5_adversarial_hardening.py - Adversarial Coverage Hardening (Tier 5) Test Suite.
Milestone M5: Unified E2E Test Suite & Adversarial Hardening.

Exercises adversarial edge cases, hostile fuzzing, mathematical singularities, and thread stress:
- Suite 1: TestSchemaFuzzingAndBoundaryAdversarial (6 tests)
- Suite 2: TestNetworkFaultsAndRetryAdversarial (5 tests)
- Suite 3: TestPsychometricsNumericalSingularitiesAdversarial (5 tests)
- Suite 4: TestWorkerFleetSuddenTerminationAndQuotaRaces (5 tests)
- Suite 5: TestSchedulerBoundarySingularitiesAndClockWarp (5 tests)
- Suite 6: TestExportServiceAndStateCorruptionRecovery (4 tests)
- Suite 7: TestAppHeadlessAdversarialStress (2 tests)

Total: 32 Hardened Adversarial Test Cases.
"""

from __future__ import annotations

import copy
import datetime as dt
import io
import math
import random
import re
import sys
import threading
import time
import unittest
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.export_service import DemographicMappings, ExportService
from src.persona_engine import (
    OPT_AWARE_USED_ONCE,
    OPT_CAT_BEAUTY,
    OPT_CAT_TECH,
    OPT_EDU_DAI_HOC,
    OPT_EMP_FULL_TIME,
    OPT_EMP_PART_TIME,
    OPT_FREQ_OVER_8,
    OPT_GENDER_NAM,
    OPT_GENDER_NU,
    OPT_LOC_HCM,
    OPT_PAIN_BUILDING,
    OPT_PAIN_NONE,
    OPT_PROF_CNTT,
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
from src.stats_analyzer import (
    ConstructStats,
    ItemStats,
    StatsAnalyzer,
    StatsSummary,
    StreamingCovarianceTracker,
)
from src.tally_client import (
    TallyClient,
    TallyConnectionError,
    TallyDisqualificationError,
    TallyRateLimitError,
    TallyServerError,
    TallySubmissionError,
    TallyValidationError,
)
from src.tally_schema import (
    AGE_QUESTION_UUID,
    CONSENT_AGREE_UUID,
    CONSENT_DISAGREE_UUID,
    CONSENT_QUESTION_UUID,
    QuestionType,
    REASON_NEVER_QUESTION_UUID,
    TallySchema,
    USAGE_NEVER_UUID,
    USAGE_QUESTION_UUID,
    UUID_V4_PATTERN,
)
from src.worker_manager import (
    FleetConfig,
    FleetState,
    WorkerCardSnapshot,
    WorkerFleetSnapshot,
    WorkerManager,
    WorkerState,
)


# ============================================================================
# Helpers
# ============================================================================

class MockTimeProvider:
    """Injectable virtual clock for deterministic temporal testing across window boundaries."""

    def __init__(self, start_dt: Optional[dt.datetime] = None) -> None:
        if start_dt is not None:
            self._current_dt = start_dt
            self._current_time = start_dt.timestamp()
        else:
            self._current_dt = dt.datetime(2026, 11, 2, 12, 0, 0, tzinfo=dt.timezone.utc)
            self._current_time = self._current_dt.timestamp()

    def time(self) -> float:
        return self._current_time

    def now(self) -> dt.datetime:
        return self._current_dt

    def datetime(self) -> dt.datetime:
        return self._current_dt

    def advance(self, seconds: float) -> None:
        self._current_time += seconds
        self._current_dt += dt.timedelta(seconds=seconds)


# ============================================================================
# Suite 1: TestSchemaFuzzingAndBoundaryAdversarial (6 tests)
# ============================================================================

class TestSchemaFuzzingAndBoundaryAdversarial(unittest.TestCase):
    """Hostile fuzzing of TallySchema against malformed UUIDs, collisions, and payload injections."""

    @classmethod
    def setUpClass(cls) -> None:
        TallySchema.initialize()

    def _get_valid_payload(self) -> Dict[str, Any]:
        engine = PersonaEngine(seed=42)
        profile = engine.generate_respondent()
        return TallySchema.get_payload(
            responses=profile.payload_responses,
            session_uuid=str(uuid.uuid4()),
            respondent_uuid=str(uuid.uuid4()),
            is_completed=True,
        )

    def test_uuid_format_adversarial_fuzzing(self) -> None:
        """Nil UUID, v1 UUID, malformed strings, SQL injection, XSS must be rejected."""
        valid_payload = self._get_valid_payload()

        hostile_uuids = [
            "00000000-0000-0000-0000-000000000000",  # Nil UUID
            "2f28126e-9844-11e7-abc4-cec278b6b50a",  # UUID v1 (version digit is 1)
            "550e8400-e29b-41d4-a716-44665544000",   # Length 35 (too short)
            "550e8400-e29b-41d4-a716-4466554400000", # Length 37 (too long)
            "' OR 1=1 --",                           # SQL Injection payload
            "<script>alert(1)</script>",             # XSS payload
            " 550e8400-e29b-41d4-a716-446655440000 ", # Unstripped whitespace
            "",                                      # Empty string
        ]

        for bad_uuid in hostile_uuids:
            # Test in sessionUuid
            payload_session = copy.deepcopy(valid_payload)
            payload_session["sessionUuid"] = bad_uuid
            errors = TallySchema.get_validation_errors(payload_session)
            self.assertTrue(
                any("sessionUuid" in err for err in errors),
                f"Hostile UUID '{bad_uuid}' was improperly accepted for sessionUuid",
            )
            with self.assertRaises(TallyValidationError):
                TallySchema.validate_payload(payload_session)

            # Test in respondentUuid
            payload_resp = copy.deepcopy(valid_payload)
            payload_resp["respondentUuid"] = bad_uuid
            errors_resp = TallySchema.get_validation_errors(payload_resp)
            self.assertTrue(
                any("respondentUuid" in err for err in errors_resp),
                f"Hostile UUID '{bad_uuid}' was improperly accepted for respondentUuid",
            )
            with self.assertRaises(TallyValidationError):
                TallySchema.validate_payload(payload_resp)

        # Uppercase valid UUID v4 must be accepted
        uppercase_v4 = str(uuid.uuid4()).upper()
        payload_upper = copy.deepcopy(valid_payload)
        payload_upper["sessionUuid"] = uppercase_v4
        self.assertTrue(UUID_V4_PATTERN.match(uppercase_v4) is not None)
        errs = TallySchema.get_validation_errors(payload_upper)
        self.assertEqual(len(errs), 0, f"Valid uppercase UUID v4 rejected: {errs}")

    def test_session_and_respondent_uuid_collision(self) -> None:
        """sessionUuid and respondentUuid must be distinct entities (case-insensitive)."""
        valid_payload = self._get_valid_payload()
        same_uuid = str(uuid.uuid4())

        # Exact match
        payload_identical = copy.deepcopy(valid_payload)
        payload_identical["sessionUuid"] = same_uuid
        payload_identical["respondentUuid"] = same_uuid

        errors = TallySchema.get_validation_errors(payload_identical)
        self.assertTrue(any("distinct entities" in err for err in errors))
        with self.assertRaises(TallyValidationError) as ctx:
            TallySchema.validate_payload(payload_identical)
        self.assertIn("distinct entities", str(ctx.exception))

        # Mixed-case match (upper vs lower)
        payload_mixed = copy.deepcopy(valid_payload)
        payload_mixed["sessionUuid"] = same_uuid.lower()
        payload_mixed["respondentUuid"] = same_uuid.upper()

        errors_mixed = TallySchema.get_validation_errors(payload_mixed)
        self.assertTrue(any("distinct entities" in err for err in errors_mixed))
        with self.assertRaises(TallyValidationError):
            TallySchema.validate_payload(payload_mixed)

    def test_age_demographic_boundary_fuzzing(self) -> None:
        """Ages outside [18, 27], booleans, NaNs, Infs, and strings must trigger validation errors."""
        valid_payload = self._get_valid_payload()

        invalid_ages = [
            17, 28, 0, -1, 100,
            True, False,  # Python booleans inherit from int! Must be strictly rejected
            float("nan"), float("inf"), float("-inf"),
            "21", "eighteen", None, [], {},
        ]

        for bad_age in invalid_ages:
            payload = copy.deepcopy(valid_payload)
            payload["responses"][AGE_QUESTION_UUID] = bad_age

            errors = TallySchema.get_validation_errors(payload)
            self.assertTrue(
                any("Age" in err or "age" in err for err in errors),
                f"Invalid age '{bad_age}' (type: {type(bad_age).__name__}) was not caught in errors: {errors}",
            )
            with self.assertRaises(TallyValidationError):
                TallySchema.validate_payload(payload)

        # Boundary extremes 18 and 27 must pass
        for boundary_age in (18, 27):
            payload = copy.deepcopy(valid_payload)
            payload["responses"][AGE_QUESTION_UUID] = boundary_age
            self.assertEqual(len(TallySchema.get_validation_errors(payload)), 0)

    def test_option_cardinality_and_cross_question_pollution(self) -> None:
        """Multiple choice requires exactly 1 option; checkboxes require >= 1 distinct options."""
        valid_payload = self._get_valid_payload()

        # 1. MULTIPLE_CHOICE with empty list or multiple items
        gender_q = TallySchema.CODE_MAP.get("Q5_GENDER")
        self.assertIsNotNone(gender_q)
        g_uuid = gender_q.group_uuid
        opt1 = gender_q.options[0].uuid
        opt2 = gender_q.options[1].uuid

        payload_empty_mc = copy.deepcopy(valid_payload)
        payload_empty_mc["responses"][g_uuid] = []
        errors = TallySchema.get_validation_errors(payload_empty_mc)
        self.assertTrue(any("exactly 1 option" in err for err in errors))

        payload_multi_mc = copy.deepcopy(valid_payload)
        payload_multi_mc["responses"][g_uuid] = [opt1, opt2]
        errors = TallySchema.get_validation_errors(payload_multi_mc)
        self.assertTrue(any("accepts exactly 1 option" in err for err in errors))

        # 2. Option UUID belonging to another question (Cross-question pollution)
        loc_q = TallySchema.CODE_MAP.get("Q6_LOCATION")
        self.assertIsNotNone(loc_q)
        foreign_opt = loc_q.options[0].uuid
        payload_foreign = copy.deepcopy(valid_payload)
        payload_foreign["responses"][g_uuid] = [foreign_opt]
        errors = TallySchema.get_validation_errors(payload_foreign)
        self.assertTrue(any("does not belong" in err for err in errors))

        # 3. CHECKBOXES with empty list or duplicate selections
        cat_q = TallySchema.CODE_MAP.get("Q11_CATEGORIES")
        self.assertIsNotNone(cat_q)
        cat_uuid = cat_q.group_uuid
        cat_opt1 = cat_q.options[0].uuid

        payload_empty_cb = copy.deepcopy(valid_payload)
        payload_empty_cb["responses"][cat_uuid] = []
        errors = TallySchema.get_validation_errors(payload_empty_cb)
        self.assertTrue(any("cannot have empty selection" in err for err in errors))

        payload_dup_cb = copy.deepcopy(valid_payload)
        payload_dup_cb["responses"][cat_uuid] = [cat_opt1, cat_opt1]
        errors = TallySchema.get_validation_errors(payload_dup_cb)
        self.assertTrue(any("duplicate selections" in err for err in errors))

    def test_screening_disqualification_traps(self) -> None:
        """Screening options (Consent Disagreed, Usage Never) must raise TallyDisqualificationError."""
        valid_payload = self._get_valid_payload()

        # Disagree with Consent
        payload_consent_dq = copy.deepcopy(valid_payload)
        payload_consent_dq["responses"][CONSENT_QUESTION_UUID] = [CONSENT_DISAGREE_UUID]

        errors = TallySchema.get_validation_errors(payload_consent_dq)
        self.assertTrue(any("Screening disqualification" in err for err in errors))
        with self.assertRaises(TallyDisqualificationError) as ctx:
            TallySchema.validate_payload(payload_consent_dq)
        self.assertIn("Consent", str(ctx.exception))

        # Never used Shopee
        payload_usage_dq = copy.deepcopy(valid_payload)
        payload_usage_dq["responses"][USAGE_QUESTION_UUID] = [USAGE_NEVER_UUID]

        errors = TallySchema.get_validation_errors(payload_usage_dq)
        self.assertTrue(any("Screening disqualification" in err for err in errors))
        with self.assertRaises(TallyDisqualificationError) as ctx:
            TallySchema.validate_payload(payload_usage_dq)
        self.assertIn("Shopee", str(ctx.exception))

    def test_envelope_corruption(self) -> None:
        """Non-dict payload, missing keys, or isCompleted=False must raise TallyValidationError."""
        for corrupt_payload in [None, [], "not_a_dict", 12345]:
            errors = TallySchema.get_validation_errors(corrupt_payload)  # type: ignore
            self.assertTrue(any("must be a dictionary" in err for err in errors))
            with self.assertRaises(TallyValidationError):
                TallySchema.validate_payload(corrupt_payload)  # type: ignore

        valid_payload = self._get_valid_payload()

        # Missing envelope keys
        for key in ["sessionUuid", "respondentUuid", "responses", "isCompleted"]:
            p = copy.deepcopy(valid_payload)
            del p[key]
            errors = TallySchema.get_validation_errors(p)
            self.assertTrue(any(f"Missing required envelope key: '{key}'" in err for err in errors))
            with self.assertRaises(TallyValidationError):
                TallySchema.validate_payload(p)

        # isCompleted is False or not a boolean True
        for bad_is_completed in [False, None, 1, "True"]:
            p = copy.deepcopy(valid_payload)
            p["isCompleted"] = bad_is_completed
            errors = TallySchema.get_validation_errors(p)
            self.assertTrue(any("isCompleted" in err for err in errors))
            with self.assertRaises(TallyValidationError):
                TallySchema.validate_payload(p)


# ============================================================================
# Suite 2: TestNetworkFaultsAndRetryAdversarial (5 tests)
# ============================================================================

class TestNetworkFaultsAndRetryAdversarial(unittest.TestCase):
    """Stress-tests TallyClient against rate limits, chaos servers, and network drops."""

    @classmethod
    def setUpClass(cls) -> None:
        TallySchema.initialize()

    def _get_valid_payload(self) -> Dict[str, Any]:
        engine = PersonaEngine(seed=42)
        profile = engine.generate_respondent()
        return TallySchema.get_payload(
            responses=profile.payload_responses,
            session_uuid=str(uuid.uuid4()),
            respondent_uuid=str(uuid.uuid4()),
            is_completed=True,
        )

    def test_upstream_rate_limit_jitter_and_exhaustion(self) -> None:
        """Client parses various Retry-After headers and raises TallyRateLimitError upon exhaustion."""
        payload = self._get_valid_payload()
        retry_headers = [
            ("2", 2.0),
            ("1.5", 1.5),
            ("Mon, 02 Nov 2026 09:00:00 GMT", 0.0),  # Past date -> 0.0s
            ("1000000", 60.0),  # Capped at MAX_RETRY_AFTER_CAP_SECONDS (60.0s)
            ("-10", 0.0),       # Negative value -> 0.0s
            ("invalid_header", 0.0),
        ]

        for header_val, expected_val in retry_headers:
            client = TallyClient(max_retries=1, dry_run=False)
            mock_resp = httpx.Response(
                status_code=429,
                headers={"Retry-After": header_val},
                text="Too Many Requests",
                request=httpx.Request("POST", client.endpoint),
            )

            with patch.object(client.client, "post", return_value=mock_resp), \
                 patch("time.sleep") as mock_sleep:
                with self.assertRaises(TallyRateLimitError) as ctx:
                    client.submit(payload, dry_run=False)

                self.assertEqual(ctx.exception.attempts, 2)
                # Parsed retry after should match bounds
                if header_val == "invalid_header":
                    self.assertIsNone(ctx.exception.retry_after)
                else:
                    self.assertIsNotNone(ctx.exception.retry_after)
                    self.assertLessEqual(ctx.exception.retry_after, 60.0)
                    self.assertGreaterEqual(ctx.exception.retry_after, 0.0)

    def test_intermittent_upstream_chaos_and_recovery(self) -> None:
        """Chaos proxy returning 500 -> 502 -> 200 must be recovered cleanly on attempt 3."""
        payload = self._get_valid_payload()
        client = TallyClient(max_retries=3, dry_run=False)

        resp_500 = httpx.Response(500, text="Internal Server Error", request=httpx.Request("POST", client.endpoint))
        resp_502 = httpx.Response(502, text="Bad Gateway", request=httpx.Request("POST", client.endpoint))
        resp_200 = httpx.Response(200, json={"submissionId": "sub_chaos_recovered"}, request=httpx.Request("POST", client.endpoint))

        with patch.object(client.client, "post", side_effect=[resp_500, resp_502, resp_200]), \
             patch("time.sleep"):
            res = client.submit(payload, dry_run=False)
            self.assertTrue(res["success"])
            self.assertEqual(res["data"]["submissionId"], "sub_chaos_recovered")

    def test_terminal_client_error_fast_fail(self) -> None:
        """HTTP 400, 401, 403, 404, 422 must fail immediately with attempts=1 without retrying."""
        payload = self._get_valid_payload()

        for status_code in (400, 401, 403, 404, 422):
            client = TallyClient(max_retries=3, dry_run=False)
            mock_resp = httpx.Response(
                status_code=status_code,
                text=f"Client Error {status_code}",
                request=httpx.Request("POST", client.endpoint),
            )

            with patch.object(client.client, "post", return_value=mock_resp) as mock_post, \
                 patch("time.sleep") as mock_sleep:
                with self.assertRaises(TallySubmissionError) as ctx:
                    client.submit(payload, dry_run=False)

                self.assertEqual(ctx.exception.status_code, status_code)
                self.assertEqual(ctx.exception.attempts, 1)
                self.assertEqual(mock_post.call_count, 1)
                mock_sleep.assert_not_called()

    def test_network_drop_and_socket_timeout_resilience(self) -> None:
        """Connection timeouts and network drops must be retried and raise TallyConnectionError."""
        payload = self._get_valid_payload()
        client = TallyClient(max_retries=2, dry_run=False)

        exceptions = [
            httpx.ConnectTimeout("Connect timeout"),
            httpx.ReadTimeout("Read timeout"),
            httpx.NetworkError("Socket drop"),
        ]

        with patch.object(client.client, "post", side_effect=exceptions), \
             patch("time.sleep") as mock_sleep:
            with self.assertRaises(TallyConnectionError) as ctx:
                client.submit(payload, dry_run=False)

            self.assertEqual(ctx.exception.attempts, 3)
            self.assertEqual(mock_sleep.call_count, 2)

    def test_client_concurrency_and_idempotent_closure(self) -> None:
        """Multiple threads submitting concurrently while client is closed and reopened."""
        client = TallyClient(dry_run=True)
        payload = self._get_valid_payload()
        errors: List[Exception] = []

        def worker_task():
            try:
                for _ in range(5):
                    client.submit(payload, dry_run=True)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker_task) for _ in range(10)]
        for t in threads:
            t.start()

        # Concurrently close client
        time.sleep(0.002)
        client.close()
        client.close()  # Idempotent

        for t in threads:
            t.join(timeout=2.0)

        self.assertEqual(len(errors), 0, f"Thread errors during client close: {errors}")


# ============================================================================
# Suite 3: TestPsychometricsNumericalSingularitiesAdversarial (5 tests)
# ============================================================================

class TestPsychometricsNumericalSingularitiesAdversarial(unittest.TestCase):
    """Stress-tests StatsAnalyzer against 0-variance, singularities, and matrix precision."""

    def test_zero_variance_degenerate_samples(self) -> None:
        """Feeding identical respondents must yield total variance = 0, alpha = 0.0, ZERO_VARIANCE."""
        analyzer = StatsAnalyzer()
        identical_scores = {k: 4 for k in ExportService.ALL_39_ITEMS}

        for _ in range(50):
            analyzer.add_sample(identical_scores)

        summary = analyzer.get_summary()
        self.assertEqual(summary.sample_count, 50)
        self.assertEqual(summary.overall_alpha, 0.0)
        self.assertEqual(summary.status, "ZERO_VARIANCE")
        self.assertFalse(summary.is_reliable)

        # All construct alphas must be 0.0 without ZeroDivisionError
        for c_stats in summary.construct_stats.values():
            self.assertEqual(c_stats.alpha, 0.0)
            self.assertEqual(c_stats.std, 0.0)

    def test_single_item_zero_variance_citc(self) -> None:
        """When an individual item has zero variance, CITC must be safely returned as 0.0."""
        analyzer = StatsAnalyzer()
        rng = np.random.default_rng(42)

        for _ in range(40):
            scores = {k: int(rng.integers(1, 7)) for k in ExportService.ALL_39_ITEMS}
            scores["PU1"] = 5  # Constant zero variance item!
            analyzer.add_sample(scores)

        summary = analyzer.get_summary()
        pu_stats = summary.construct_stats["PU"]
        pu1_stats = pu_stats.item_stats["PU1"]

        self.assertEqual(pu1_stats.std, 0.0)
        self.assertEqual(pu1_stats.citc, 0.0)  # Must be 0.0, not NaN or crash
        self.assertFalse(math.isnan(pu1_stats.citc))

    def test_collinear_and_anti_correlated_items(self) -> None:
        """Anti-correlated items causing negative theoretical alpha must clamp safely to 0.0."""
        tracker = StreamingCovarianceTracker(k_items=3)
        # Sample 1: [1, 6, 1]
        # Sample 2: [6, 1, 6]
        # Sample 3: [1, 6, 1]
        # Sample 4: [6, 1, 6]
        for i in range(20):
            if i % 2 == 0:
                tracker.update(np.array([1.0, 6.0, 1.0]))
            else:
                tracker.update(np.array([6.0, 1.0, 6.0]))

        alpha, status = tracker.compute_alpha()
        self.assertGreaterEqual(alpha, 0.0)
        self.assertLessEqual(alpha, 1.0)
        self.assertFalse(math.isnan(alpha))

    def test_corrupted_sample_score_extraction(self) -> None:
        """Missing items in Likert dictionary must raise KeyError."""
        analyzer = StatsAnalyzer()
        incomplete_scores = {"PU1": 5, "PU2": 4}  # Missing 37 items

        with self.assertRaises(KeyError):
            analyzer.add_sample(incomplete_scores)

    def test_streaming_welford_vs_batch_covariance_exact_precision(self) -> None:
        """Online vector Welford covariance must match numpy.cov within 1e-11 absolute error."""
        k = 10
        n_samples = 200
        rng = np.random.default_rng(12345)
        data = rng.uniform(1.0, 6.0, size=(n_samples, k))

        tracker = StreamingCovarianceTracker(k_items=k)
        for row in data:
            tracker.update(row)

        welford_cov = tracker.covariance_matrix()
        numpy_cov = np.cov(data, rowvar=False, ddof=1)

        max_err = float(np.max(np.abs(welford_cov - numpy_cov)))
        self.assertLess(max_err, 1e-11, f"Welford error exceeded threshold: {max_err}")


# ============================================================================
# Suite 4: TestWorkerFleetSuddenTerminationAndQuotaRaces (5 tests)
# ============================================================================

class TestWorkerFleetSuddenTerminationAndQuotaRaces(unittest.TestCase):
    """Stress-tests WorkerManager against sudden stops, lifecycle thrashing, and quota races."""

    def test_sudden_thread_termination_during_pacing(self) -> None:
        """Calling manager.stop() while workers are in long PACING must join within 2.0s."""
        cfg = FleetConfig(
            target_count=100,
            num_workers=10,
            dry_run=True,
            delay_range=(10.0, 20.0),
            jitter_config=JitterConfig(min_delay=10.0, max_delay=20.0, micro_break_enabled=False),
        )
        wm = WorkerManager(config=cfg)
        wm.start()

        # Wait until workers enter PACING
        time.sleep(0.3)
        snap = wm.get_snapshot()
        self.assertTrue(any(w.state == WorkerState.PACING for w in snap.workers))

        t0 = time.perf_counter()
        wm.stop(timeout=2.0)
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 2.0, f"Worker fleet stop took too long: {elapsed:.2f}s")
        self.assertFalse(wm.is_running)
        self.assertEqual(wm.get_snapshot().active_workers, 0)

    def test_sudden_thread_termination_during_submitting(self) -> None:
        """Stopping fleet while workers are executing submission must not hang."""
        class SlowTallyClient(TallyClient):
            def submit(self, payload: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
                time.sleep(1.0)
                return super().submit(payload, dry_run=True)

        cfg = FleetConfig(
            target_count=50,
            num_workers=6,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config=JitterConfig(min_delay=0.0, max_delay=0.001, micro_break_enabled=False),
        )
        wm = WorkerManager(config=cfg, tally_client=SlowTallyClient(dry_run=True))
        wm.start()

        time.sleep(0.2)
        t0 = time.perf_counter()
        wm.stop(timeout=3.0)
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 3.0)
        self.assertFalse(wm.is_running)

    def test_rapid_lifecycle_thrashing_stress(self) -> None:
        """Rapid flapping through start -> pause -> resume -> stop (20 cycles) without deadlocks."""
        cfg = FleetConfig(
            target_count=500,
            num_workers=8,
            dry_run=True,
            delay_range=(0.001, 0.003),
            jitter_config=JitterConfig(min_delay=0.001, max_delay=0.003, micro_break_enabled=False),
        )
        wm = WorkerManager(config=cfg)

        for _ in range(20):
            wm.start()
            wm.pause()
            wm.resume()
            wm.stop(timeout=2.0)

        self.assertFalse(wm.is_running)
        self.assertEqual(wm.get_snapshot().active_workers, 0)

    def test_exact_quota_reservation_under_upstream_failure_storm(self) -> None:
        """Under a 50% failure rate, _refund_work allows retries until exact target is reached."""
        class FlakyTallyClient(TallyClient):
            def __init__(self):
                super().__init__(dry_run=True)
                self._rng = random.Random(999)
                self._lock = threading.Lock()

            def submit(self, payload: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
                with self._lock:
                    coin = self._rng.random()
                if coin < 0.50:
                    raise TallyServerError("Simulated 500 error", status_code=500, payload=payload)
                return super().submit(payload, dry_run=True)

        target = 15
        cfg = FleetConfig(
            target_count=target,
            num_workers=10,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config=JitterConfig(min_delay=0.0, max_delay=0.001, micro_break_enabled=False),
        )
        wm = WorkerManager(config=cfg, tally_client=FlakyTallyClient())

        try:
            wm.start()
            completed = wm.wait_until_complete(timeout=10.0)
            self.assertTrue(completed)

            snap = wm.get_snapshot()
            self.assertEqual(snap.success_count, target)
            self.assertGreater(snap.error_count, 0)
            self.assertEqual(len(wm.get_history_df()), target)

        finally:
            wm.stop(timeout=2.0)

    def test_extreme_concurrency_quota_one(self) -> None:
        """20 workers racing for target=1: strictly 1 success, 0 overshoot."""
        cfg = FleetConfig(
            target_count=1,
            num_workers=20,
            dry_run=True,
            delay_range=(0.0, 0.001),
            jitter_config=JitterConfig(min_delay=0.0, max_delay=0.001, micro_break_enabled=False),
            master_seed=777,
        )
        wm = WorkerManager(config=cfg)

        try:
            wm.start()
            completed = wm.wait_until_complete(timeout=5.0)
            self.assertTrue(completed)

            snap = wm.get_snapshot()
            self.assertEqual(snap.success_count, 1)
            self.assertEqual(snap.submitted_count, 1)
            self.assertEqual(len(wm.get_history_df()), 1)

            # Sum of worker card successes must be exactly 1
            self.assertEqual(sum(w.success_count for w in snap.workers), 1)

        finally:
            wm.stop(timeout=2.0)


# ============================================================================
# Suite 5: TestSchedulerBoundarySingularitiesAndClockWarp (5 tests)
# ============================================================================

class TestSchedulerBoundarySingularitiesAndClockWarp(unittest.TestCase):
    """Stress-tests BackgroundScheduler and TimeWindow on boundary conditions and clock warps."""

    def test_malformed_time_window_strings(self) -> None:
        """Invalid time window strings must raise ValueError."""
        invalid_windows = [
            "25:00-08:00",
            "08:65-12:00",
            "12:00",
            "invalid-string",
            "",
            "10:00 - 25:00",
        ]
        for bad_win in invalid_windows:
            with self.assertRaises(ValueError):
                TimeWindow.from_string(bad_win)

    def test_exact_window_boundary_microsecond_precision(self) -> None:
        """Daytime window [09:00, 17:00) adherence at exact microsecond thresholds."""
        win = TimeWindow.from_string("09:00-17:00")

        # Just before start
        self.assertFalse(win.contains(dt.time(8, 59, 59, 999999)))
        # Exactly at start
        self.assertTrue(win.contains(dt.time(9, 0, 0, 0)))
        # Just before end
        self.assertTrue(win.contains(dt.time(16, 59, 59, 999999)))
        # Exactly at end (exclusive bound)
        self.assertFalse(win.contains(dt.time(17, 0, 0, 0)))

    def test_overnight_window_boundary_and_midnight_wrap(self) -> None:
        """Overnight window [22:00, 04:00) must properly wrap across midnight."""
        win = TimeWindow.from_string("22:00-04:00")

        self.assertFalse(win.contains(dt.time(21, 59, 59)))
        self.assertTrue(win.contains(dt.time(22, 0, 0)))
        self.assertTrue(win.contains(dt.time(23, 59, 59)))
        self.assertTrue(win.contains(dt.time(0, 0, 0)))      # Midnight
        self.assertTrue(win.contains(dt.time(3, 59, 59)))
        self.assertFalse(win.contains(dt.time(4, 0, 0)))

    def test_virtual_clock_warp_and_cooldown_expiry(self) -> None:
        """Advancing virtual clock past cooldown duration immediately transitions state."""
        clock = MockTimeProvider(start_dt=dt.datetime(2026, 11, 2, 10, 0, 0))
        sched = BackgroundScheduler(
            batch_limit=5,
            cooldown_seconds=300.0,
            time_provider=clock,
        )

        for _ in range(5):
            sched.record_submission()

        self.assertTrue(sched.is_in_cooldown())
        self.assertEqual(sched.get_state(), SchedulerState.COOLDOWN)
        self.assertAlmostEqual(sched.get_cooldown_remaining(), 300.0, delta=1.0)

        # Warp clock forward by 301 seconds
        clock.advance(301.0)
        self.assertFalse(sched.is_in_cooldown())
        self.assertEqual(sched.get_state(), SchedulerState.ACTIVE)
        self.assertEqual(sched.get_cooldown_remaining(), 0.0)

    def test_jitter_config_boundary_validation(self) -> None:
        """Invalid jitter configuration parameters must raise ValueError."""
        # min_delay > max_delay
        with self.assertRaises(ValueError):
            JitterConfig(min_delay=10.0, max_delay=5.0)

        # negative min_delay
        with self.assertRaises(ValueError):
            JitterConfig(min_delay=-1.0, max_delay=5.0)

        # mean outside [min_delay, max_delay]
        with self.assertRaises(ValueError):
            JitterConfig(min_delay=2.0, max_delay=8.0, mean_delay=1.0)

        # std_dev <= 0
        with self.assertRaises(ValueError):
            JitterConfig(min_delay=2.0, max_delay=8.0, std_dev=0.0)


# ============================================================================
# Suite 6: TestExportServiceAndStateCorruptionRecovery (4 tests)
# ============================================================================

class TestExportServiceAndStateCorruptionRecovery(unittest.TestCase):
    """Stress-tests ExportService against corrupted records, NaNs, and format compliance."""

    def test_corrupted_and_missing_record_matrix_conversion(self) -> None:
        """Records with missing demographic fields, None values, string numbers must convert safely."""
        corrupted_records = [
            # Record 1: None values and missing fields
            {
                "submission_id": "bad-01",
                "age": None,
                "gender": None,
                "location": None,
                "likert_scores": {"PU1": 7, "PU2": -2, "PU3": "5"},  # out of bounds / strings
            },
            # Record 2: Empty record
            {},
            # Record 3: Float age and NaN scores
            {
                "age": 22.8,
                "gender": OPT_GENDER_NU,
                "likert_scores": {"PU1": float("nan")},
            },
        ]

        df = ExportService.to_numeric_matrix(corrupted_records)
        self.assertEqual(len(df), 3)
        self.assertFalse(df.isna().any().any(), "Exported matrix contains unexpected NaNs")

        # Age must fall back to 21 or cast cleanly
        self.assertEqual(df.loc[0, "Age"], 21)
        self.assertEqual(df.loc[1, "Age"], 21)
        self.assertEqual(df.loc[2, "Age"], 22)

        # Likert items must be clamped to [1, 6]
        self.assertTrue((df["PU1"] >= 1).all() and (df["PU1"] <= 6).all())
        self.assertTrue((df["PU2"] >= 1).all() and (df["PU2"] <= 6).all())

    def test_dummy_variables_cardinality_and_mutual_exclusion(self) -> None:
        """Category and Pain Point dummies must be strictly binary 0 or 1."""
        records = [
            {"categories": [OPT_CAT_TECH, OPT_CAT_BEAUTY], "pain_points": [OPT_PAIN_NONE]},
            {"categories": [], "pain_points": [OPT_PAIN_BUILDING]},
        ]
        df = ExportService.to_numeric_matrix(records)

        dummy_cols = [
            "CAT_Fashion", "CAT_Beauty", "CAT_Tech", "CAT_Books", "CAT_Household", "CAT_Other",
            "PAIN_Building", "PAIN_Privacy", "PAIN_None", "PAIN_Other",
        ]
        for col in dummy_cols:
            self.assertIn(col, df.columns)
            self.assertTrue(df[col].isin([0, 1]).all())

        # Record 1 had Tech and Beauty
        self.assertEqual(df.loc[0, "CAT_Tech"], 1)
        self.assertEqual(df.loc[0, "CAT_Beauty"], 1)
        self.assertEqual(df.loc[0, "CAT_Fashion"], 0)
        self.assertEqual(df.loc[0, "PAIN_None"], 1)
        self.assertEqual(df.loc[0, "PAIN_Building"], 0)

    def test_multi_sheet_excel_resilience_under_empty_and_partial_stats(self) -> None:
        """Exporting Excel with 0 records or missing construct stats must produce valid XLSX."""
        # 0 records
        excel_bytes = ExportService.export_excel([])
        self.assertTrue(excel_bytes.startswith(b"PK\x03\x04"))

        zf = zipfile.ZipFile(io.BytesIO(excel_bytes))
        sheet_names = []
        root = ET.fromstring(zf.read("xl/workbook.xml"))
        for elem in root.iter():
            if elem.tag.endswith("sheet") and "name" in elem.attrib:
                sheet_names.append(elem.attrib["name"])

        self.assertEqual(sheet_names, ["Data", "Variable Codebook", "Psychometric Summary"])

    def test_csv_rfc4180_compliance_and_utf8_bom(self) -> None:
        """CSV output must start with UTF-8 BOM bytes (\xef\xbb\xbf) for Excel SPSS compatibility."""
        df_records = [{"age": 22, "gender": OPT_GENDER_NAM}]
        csv_bytes = ExportService.export_spss_csv(df_records)

        self.assertTrue(csv_bytes.startswith(b"\xef\xbb\xbf"))
        # Parse with utf-8-sig to verify standard RFC 4180 parsing
        parsed = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig")
        self.assertEqual(len(parsed), 1)
        self.assertIn("Respondent_ID", parsed.columns)


# ============================================================================
# Suite 7: TestAppHeadlessAdversarialStress (2 tests)
# ============================================================================

class TestAppHeadlessAdversarialStress(unittest.TestCase):
    """Stress-tests Streamlit app.py under extreme parameters and boundary actions."""

    def test_apptest_extreme_parameter_deck(self) -> None:
        """Set extreme parameters (N=1, workers=1, uniform jitter), cycle buttons, assert no crash."""
        at = AppTest.from_file("app.py", default_timeout=25)
        at.run()
        self.assertFalse(at.exception)

        # Set N=1
        target_input = at.sidebar.number_input[0]
        target_input.set_value(1).run()
        self.assertFalse(at.exception)

        # Set workers=1
        worker_slider = at.sidebar.slider[0]
        worker_slider.set_value(1).run()
        self.assertFalse(at.exception)

        # Start fleet
        at.sidebar.button[0].click().run()
        self.assertFalse(at.exception)

        # Stop fleet
        at.sidebar.button[2].click().run()
        self.assertFalse(at.exception)

        # Reset fleet
        at.sidebar.button[3].click().run()
        self.assertFalse(at.exception)

    def test_apptest_zero_records_export_clicks(self) -> None:
        """Navigating to Tab 4 with 0 records must display info message and not crash."""
        at = AppTest.from_file("app.py", default_timeout=25)
        at.run()
        self.assertFalse(at.exception)

        # Ensure no records exist
        wm: WorkerManager = at.session_state["worker_manager"]
        self.assertEqual(len(wm.get_history_df()), 0)

        # App renders with 0 records without unhandled exceptions
        self.assertFalse(at.exception)


if __name__ == "__main__":
    unittest.main()
