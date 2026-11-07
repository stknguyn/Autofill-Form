"""
tests/test_tally_client.py
Comprehensive Verification & Test Suite for Milestone M1:
Tally Schema Dictionary & HTTP Client Engine.

Covers:
- 100% question UUID & option UUID bidirectional mapping parity against consolidated_schema.json
- Screening logic invariant (Consent Agree & Active Shopee Purchaser)
- Zero-network Dry-Run mode with socket blocking
- Next.js HTTP headers replication
- Resilient exponential backoff retry for HTTP 429 and 5xx
- Multi-threaded concurrency and negative boundary testing
"""

import asyncio
import json
import re
import socket
import sys
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx

from src.tally_schema import (
    OptionDefinition,
    QuestionDefinition,
    QuestionType,
    TallyDisqualificationError,
    TallyResponsePayload,
    TallySchema,
    TallyValidationError,
)
from src.tally_client import (
    TallyClient,
    TallyClientError,
    TallyConnectionError,
    TallyNetworkError,
    TallyRateLimitError,
    TallyServerError,
    TallySubmissionError,
)

UUID_V4_REGEX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

DRY_RUN_ID_REGEX = re.compile(
    r"^mock-dryrun-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _create_valid_completed_payload(
    responses_override: dict = None,
    session_uuid: str = None,
    respondent_uuid: str = None,
    is_completed: bool = True,
) -> dict:
    """
    Constructs a 100% valid, canonical completed submission payload.
    Includes all required questions:
    - Q1 (Consent): Agree
    - Q2 (Shopee usage): Qualified usage frequency
    - Q4 (Age): 21 (valid Gen Z integer 18-27)
    - Q5-Q10, Q13 (Demographics & Habit): Valid single-choice option UUIDs
    - Q11, Q12 (Categories & Pain Points): Valid checkbox option UUID lists
    - 38 Likert questions: Valid 6-point scale option UUIDs
    Excludes Q3 (REASON_NEVER) which is only for the disqualified pathway.
    """
    TallySchema.initialize()
    responses = {}
    for g_uuid, q in TallySchema.QUESTION_MAP.items():
        if q.type == QuestionType.MULTIPLE_CHOICE:
            if g_uuid == TallySchema.CONSENT_QUESTION_UUID:
                responses[g_uuid] = [TallySchema.CONSENT_AGREE_UUID]
            elif g_uuid == TallySchema.USAGE_QUESTION_UUID:
                responses[g_uuid] = [TallySchema.DEFAULT_QUALIFIED_USAGE_UUID]
            else:
                responses[g_uuid] = [q.options[0].uuid]
        elif q.type == QuestionType.CHECKBOXES:
            if g_uuid != TallySchema.REASON_NEVER_QUESTION_UUID:
                responses[g_uuid] = [q.options[0].uuid]
        elif q.type == QuestionType.INPUT_NUMBER:
            responses[g_uuid] = 21

    if responses_override:
        responses.update(responses_override)

    return TallySchema.get_payload(
        responses=responses,
        session_uuid=session_uuid,
        respondent_uuid=respondent_uuid,
        is_completed=is_completed,
        enforce_screening=False,
    )


# ============================================================================
# Class 1: Schema Parity & Canonical Inventory Tests
# ============================================================================

class TestTallySchemaParity(unittest.TestCase):
    """Pillar 2: 100% UUID mapping verification against consolidated_schema.json."""

    @classmethod
    def setUpClass(cls):
        schema_path = Path(__file__).resolve().parent.parent / "consolidated_schema.json"
        with open(schema_path, "r", encoding="utf-8") as f:
            cls.ground_truth = json.load(f)

    def test_01_total_question_count(self):
        """Verify total question blocks count is exactly 51."""
        self.assertEqual(len(TallySchema.QUESTION_MAP), 51)
        self.assertEqual(len(self.ground_truth), 51)
        self.assertEqual(len(TallySchema.QUESTIONS), 51)

    def test_02_total_option_count(self):
        """Verify total choice options count across all questions is exactly 287."""
        total_schema_options = sum(len(q.options) for q in TallySchema.QUESTION_MAP.values())
        total_gt_options = sum(len(q.get("options", [])) for q in self.ground_truth)
        self.assertEqual(total_schema_options, 287)
        self.assertEqual(total_gt_options, 287)
        self.assertEqual(len(TallySchema.ALL_OPTION_UUIDS), 287)

    def test_03_question_type_distribution(self):
        """Verify distribution of question types: 47 MULTIPLE_CHOICE, 3 CHECKBOXES, 1 INPUT_NUMBER."""
        types = {q.type for q in TallySchema.QUESTION_MAP.values()}
        self.assertEqual(types, {"MULTIPLE_CHOICE", "CHECKBOXES", "INPUT_NUMBER"})

        mc = sum(1 for q in TallySchema.QUESTION_MAP.values() if q.type == "MULTIPLE_CHOICE")
        cb = sum(1 for q in TallySchema.QUESTION_MAP.values() if q.type == "CHECKBOXES")
        num = sum(1 for q in TallySchema.QUESTION_MAP.values() if q.type == "INPUT_NUMBER")

        self.assertEqual(mc, 47)
        self.assertEqual(cb, 3)
        self.assertEqual(num, 1)

        # Check exact group UUIDs for non-MULTIPLE_CHOICE
        expected_cb_uuids = {
            "7dbc0210-2d97-42d5-ab13-e09d16fc7d99",  # Q3
            "cf2f5689-c127-484f-8a39-33e270c19974",  # Q11
            "9b8dcdcd-549d-468e-ad59-a9a1dd443aaf",  # Q12
        }
        actual_cb_uuids = {
            q.group_uuid for q in TallySchema.QUESTION_MAP.values() if q.type == "CHECKBOXES"
        }
        self.assertEqual(actual_cb_uuids, expected_cb_uuids)

        expected_num_uuid = "92570a0c-41e3-4adf-8bf7-93fc01b0a66b"  # Q4
        actual_num_uuids = [
            q.group_uuid for q in TallySchema.QUESTION_MAP.values() if q.type == "INPUT_NUMBER"
        ]
        self.assertEqual(actual_num_uuids, [expected_num_uuid])

    def test_04_likert_scale_construct_distribution(self):
        """Verify 38 Likert items across 11 constructs, each having exactly 6 scale points."""
        self.assertEqual(len(TallySchema.LIKERT_QUESTIONS), 38)

        for q in TallySchema.LIKERT_QUESTIONS:
            self.assertTrue(q.is_likert)
            self.assertEqual(len(q.options), 6)
            for idx, opt in enumerate(q.options, 1):
                self.assertEqual(opt.score, idx)

        expected_construct_counts = {
            "PU": 4,
            "PEOU": 4,
            "PES": 4,
            "PC": 3,
            "PI": 3,
            "SE": 4,
            "LC": 3,
            "ATT": 3,
            "SN": 3,
            "PBC": 4,
            "ITU": 3,
        }

        actual_counts = {c: len(ql) for c, ql in TallySchema.CONSTRUCT_MAP.items()}
        self.assertEqual(actual_counts, expected_construct_counts)

    def test_05_pc_scale_reversal_flags(self):
        """Verify Perceived Costs questions have is_reverse=True and reverse scoring."""
        pc_uuids = [q.group_uuid for q in TallySchema.CONSTRUCT_MAP["PC"]]
        self.assertEqual(len(pc_uuids), 3)
        for pc_uuid in pc_uuids:
            q = TallySchema.QUESTION_MAP.get(pc_uuid)
            self.assertIsNotNone(q)
            self.assertTrue(q.is_reverse)

        # Non-PC likert questions must have is_reverse == False
        for q in TallySchema.LIKERT_QUESTIONS:
            if q.group_uuid not in pc_uuids:
                self.assertFalse(q.is_reverse)

        # Test score reversal formula (7 - score)
        self.assertEqual(TallySchema.reverse_likert_score(1), 6)
        self.assertEqual(TallySchema.reverse_likert_score(6), 1)
        self.assertEqual(TallySchema.reverse_likert_score(3), 4)

    def test_06_anti_drift_draft_uuid_blacklist(self):
        """Verify obsolete draft UUIDs from Antigravity.md are absent and canonical PES UUIDs present."""
        blacklist = [
            "ec1f37e5-1311-4775-8eb6-cf3832d2078a",
            "d2d2a420-56ef-4fa5-baae-8bfa85ef664a",
            "ec1f37e5-df45-420a-8bf8-d63674c93eb8",
        ]
        all_uuids = set(TallySchema.ALL_GROUP_UUIDS).union(TallySchema.ALL_OPTION_UUIDS)

        for bad_uuid in blacklist:
            self.assertNotIn(bad_uuid, all_uuids, f"Draft UUID {bad_uuid} detected in schema!")

        canonical_pes_uuids = [q.group_uuid for q in TallySchema.CONSTRUCT_MAP["PES"]]
        self.assertEqual(len(canonical_pes_uuids), 4)
        for pes_uuid in canonical_pes_uuids:
            self.assertIn(pes_uuid, TallySchema.QUESTION_MAP)

    def test_06b_bidirectional_schema_parity(self):
        """Bidirectional forward & reverse fidelity check against consolidated_schema.json."""
        gt_map = {q["groupUuid"]: q for q in self.ground_truth}

        # Forward check: Ground Truth -> TallySchema
        for g_uuid, gt_q in gt_map.items():
            self.assertIn(g_uuid, TallySchema.QUESTION_MAP)
            sq = TallySchema.QUESTION_MAP[g_uuid]
            self.assertEqual(sq.type, gt_q["type"])
            gt_options = gt_q.get("options", [])
            self.assertEqual(len(sq.options), len(gt_options))

            for gt_opt in gt_options:
                opt_uuid = gt_opt["uuid"]
                self.assertIn(opt_uuid, sq.option_map)
                self.assertEqual(sq.option_map[opt_uuid].text, gt_opt["text"])

        # Reverse check: TallySchema -> Ground Truth
        for g_uuid in TallySchema.QUESTION_MAP:
            self.assertIn(g_uuid, gt_map)


# ============================================================================
# Class 2: Screening Logic & Payload Serialization Tests
# ============================================================================

class TestTallyScreeningAndPayload(unittest.TestCase):
    """Pillar 1: Screening invariants, payload serialization, and UUID isolation."""

    def setUp(self):
        self.valid_payload = _create_valid_completed_payload()
        self.valid_responses = dict(self.valid_payload["responses"])

    def test_07_screening_consent_enforcement(self):
        """Verify consent agreement passes and disagreement raises TallyDisqualificationError."""
        # 1. Agree passes
        agree_payload = TallySchema.get_payload(self.valid_responses)
        self.assertTrue(TallySchema.validate_payload(agree_payload))

        # 2. Disagree raises TallyDisqualificationError
        bad_responses = dict(self.valid_responses)
        bad_responses[TallySchema.CONSENT_QUESTION_UUID] = [TallySchema.CONSENT_DISAGREE_UUID]
        disagree_payload = TallySchema.get_payload(bad_responses)

        with self.assertRaises(TallyDisqualificationError):
            TallySchema.validate_payload(disagree_payload)

        # When raise_exception=False, returns False
        self.assertFalse(TallySchema.validate_payload(disagree_payload, raise_exception=False))

    def test_08_screening_shopee_frequency_enforcement(self):
        """Verify active Shopee buyer choices pass and 'never' option triggers disqualification."""
        # 1. All 5 active buyer options pass
        for option_uuid in TallySchema.USAGE_QUALIFIED_UUIDS:
            test_responses = dict(self.valid_responses)
            test_responses[TallySchema.USAGE_QUESTION_UUID] = [option_uuid]
            payload = TallySchema.get_payload(test_responses)
            self.assertTrue(TallySchema.validate_payload(payload))

        # 2. Never used Shopee raises TallyDisqualificationError
        never_responses = dict(self.valid_responses)
        never_responses[TallySchema.USAGE_QUESTION_UUID] = [TallySchema.USAGE_NEVER_UUID]
        never_payload = TallySchema.get_payload(never_responses)

        with self.assertRaises(TallyDisqualificationError):
            TallySchema.validate_payload(never_payload)

        # Test enforce_screening_logic fixes invalid responses
        fixed_responses = TallySchema.enforce_screening_logic(dict(never_responses))
        self.assertEqual(fixed_responses[TallySchema.CONSENT_QUESTION_UUID], [TallySchema.CONSENT_AGREE_UUID])
        self.assertEqual(fixed_responses[TallySchema.USAGE_QUESTION_UUID], [TallySchema.DEFAULT_QUALIFIED_USAGE_UUID])
        self.assertNotIn(TallySchema.REASON_NEVER_QUESTION_UUID, fixed_responses)

    def test_09_payload_top_level_contract(self):
        """Verify all required top-level envelope fields in generated payload."""
        payload = TallySchema.get_payload(self.valid_responses)

        for key in ["sessionUuid", "respondentUuid", "responses", "captchas", "isCompleted", "password", "locale"]:
            self.assertIn(key, payload)

        self.assertEqual(payload["captchas"], {})
        self.assertTrue(payload["isCompleted"])
        self.assertIsNone(payload["password"])
        self.assertEqual(payload["locale"], "vi")
        self.assertIsInstance(payload["responses"], dict)

    def test_10_uuid_v4_format_and_session_isolation(self):
        """Verify sessionUuid and respondentUuid are distinct RFC 4122 v4 UUIDs per submission."""
        p1 = TallySchema.get_payload(self.valid_responses)
        p2 = TallySchema.get_payload(self.valid_responses)

        self.assertRegex(p1["sessionUuid"], UUID_V4_REGEX)
        self.assertRegex(p1["respondentUuid"], UUID_V4_REGEX)
        self.assertNotEqual(p1["sessionUuid"], p1["respondentUuid"])

        self.assertNotEqual(p1["sessionUuid"], p2["sessionUuid"])
        self.assertNotEqual(p1["respondentUuid"], p2["respondentUuid"])

    def test_11_response_value_formatting_rules(self):
        """Verify formatting rules: MC is 1-item list, Checkbox is >= 1 list, Input is numeric."""
        # Provide responses using friendly construct codes and raw types
        code_responses = {
            "Q1_CONSENT": TallySchema.CONSENT_AGREE_UUID,  # bare string to be wrapped
            "Q2_USAGE": [TallySchema.DEFAULT_QUALIFIED_USAGE_UUID],  # list of 1
            "Q4_AGE": "22",  # numeric string to be parsed into int
            "PU1": 5,  # Likert integer score 1..6 mapped to option UUID
            "Q11_CATEGORIES": [
                "277e9238-66d1-4be5-b9f1-d68a513f56e9",
                "a196160d-45db-4414-b4fa-4b89c7467645",
            ],
        }

        payload = TallySchema.get_payload(code_responses)
        res = payload["responses"]

        # Check Q1_CONSENT group UUID has list of 1
        q1_uuid = TallySchema.CODE_MAP["Q1_CONSENT"].group_uuid
        self.assertIsInstance(res[q1_uuid], list)
        self.assertEqual(len(res[q1_uuid]), 1)
        self.assertEqual(res[q1_uuid][0], TallySchema.CONSENT_AGREE_UUID)

        # Check Q4_AGE is numeric scalar (int), NOT a list
        q4_uuid = TallySchema.CODE_MAP["Q4_AGE"].group_uuid
        self.assertIsInstance(res[q4_uuid], int)
        self.assertEqual(res[q4_uuid], 22)

        # Check PU1 score 5 is mapped to correct option UUID
        pu1_def = TallySchema.CODE_MAP["PU1"]
        expected_pu1_uuid = pu1_def.score_to_uuid[5]
        self.assertEqual(res[pu1_def.group_uuid], [expected_pu1_uuid])

        # Check Q11_CATEGORIES retains multi-item list
        q11_uuid = TallySchema.CODE_MAP["Q11_CATEGORIES"].group_uuid
        self.assertIsInstance(res[q11_uuid], list)
        self.assertEqual(len(res[q11_uuid]), 2)


# ============================================================================
# Class 3: Dry-Run Mode & Zero-Network Tests
# ============================================================================

class TestTallyClientDryRun(unittest.TestCase):
    """Pillar 3: Zero-network socket assertions & mock contract."""

    def setUp(self):
        self.client = TallyClient(dry_run=True)
        self.test_payload = _create_valid_completed_payload()

    def test_12_dry_run_zero_network_sockets(self):
        """Assert that dry_run=True NEVER calls socket.socket.connect or httpx send."""
        def forbidden_socket_connect(*args, **kwargs):
            raise AssertionError("CRITICAL VIOLATION: Network socket opened during Dry-Run mode!")

        with patch.object(socket.socket, "connect", side_effect=forbidden_socket_connect):
            with patch.object(httpx.Client, "send") as mock_send:
                response = self.client.submit_form(self.test_payload)
                self.assertEqual(mock_send.call_count, 0)

        self.assertTrue(response["success"])
        self.assertTrue(response["data"]["dryRun"])
        self.assertRegex(response["data"]["submissionId"], DRY_RUN_ID_REGEX)

    def test_13_dry_run_response_envelope_contract(self):
        """Assert dry-run response contains success, mock-dryrun UUID, and dryRun=True."""
        res = self.client.submit(self.test_payload)
        self.assertIsInstance(res, dict)
        self.assertTrue(res.get("success"))
        self.assertIn("data", res)
        self.assertTrue(res["data"].get("dryRun"))
        sub_id = res["data"].get("submissionId")
        self.assertTrue(sub_id.startswith("mock-dryrun-"))
        raw_uuid = sub_id.replace("mock-dryrun-", "")
        self.assertEqual(uuid.UUID(raw_uuid).version, 4)

    def test_14_dry_run_validates_payload_before_return(self):
        """Assert dry-run mode validates payload and raises TallyValidationError on bad data."""
        bad_payload = dict(self.test_payload)
        bad_payload["sessionUuid"] = "corrupted-uuid-string"

        with self.assertRaises(TallyValidationError):
            self.client.submit_form(bad_payload)

        # Missing required keys
        missing_keys_payload = {"sessionUuid": str(uuid.uuid4())}
        with self.assertRaises(TallyValidationError):
            self.client.submit_form(missing_keys_payload)

    def test_15_dry_run_latency_performance(self):
        """Assert Dry-Run execution with default zero latency completes in < 25ms."""
        start = time.perf_counter()
        res = self.client.submit_form(self.test_payload)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 0.025)
        self.assertTrue(res["success"])

    def test_15b_dry_run_latency_simulation_timing(self):
        """Assert Dry-Run with simulate_latency=True respects configured latency range."""
        latency_client = TallyClient(
            dry_run=True,
            simulate_latency=True,
            dry_run_latency_range=(0.04, 0.07),
        )
        start = time.perf_counter()
        res = latency_client.submit_form(self.test_payload)
        elapsed = time.perf_counter() - start
        self.assertGreaterEqual(elapsed, 0.035)
        self.assertTrue(res["success"])


# ============================================================================
# Class 4: Network Transport, Headers, Retries & Error Boundaries
# ============================================================================

class TestTallyClientNetworkAndErrors(unittest.TestCase):
    """Pillar 4: HTTP 429 backoff, 5xx retries, timeouts, and negative error scenarios."""

    def setUp(self):
        self.valid_payload = _create_valid_completed_payload()

    def test_16_client_nextjs_request_headers(self):
        """Verify client constructs all 13 authentic Next.js request headers."""
        client = TallyClient(form_id="laqVzV")
        headers = client.get_headers()

        expected_headers = {
            "Host": "api.tally.so",
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
            "Origin": "https://tally.so",
            "Referer": "https://tally.so/r/laqVzV",
            "tally-version": "2025-01-15",
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
        }

        for key, val in expected_headers.items():
            self.assertIn(key, headers)
            self.assertEqual(headers[key], val)

        self.assertIn("User-Agent", headers)
        self.assertIn("Sec-Ch-Ua", headers)
        self.assertEqual(headers, client._build_headers())

    def test_17_client_target_endpoint(self):
        """Verify client endpoint and SUBMIT_URL constant are strictly formatted."""
        client = TallyClient(form_id="laqVzV")
        self.assertEqual(client.endpoint, "https://api.tally.so/forms/laqVzV/respond")
        self.assertEqual(TallyClient.SUBMIT_URL, "https://api.tally.so/forms/laqVzV/respond")

    def test_18_http_200_success_response_parsing(self):
        """Verify successful HTTP 200 response parses submissionId."""
        client = TallyClient(dry_run=False)
        resp_200 = httpx.Response(status_code=200, json={"submissionId": "sub_test123"})

        with patch.object(httpx.Client, "post", return_value=resp_200) as mock_post:
            res = client.submit_form(self.valid_payload)
            self.assertTrue(res["success"])
            self.assertEqual(res["data"]["submissionId"], "sub_test123")
            self.assertFalse(res["data"]["dryRun"])
            self.assertEqual(mock_post.call_count, 1)

    def test_19_http_429_rate_limit_exponential_backoff_recovery(self):
        """Verify client recovers from HTTP 429 when retry succeeds."""
        client = TallyClient(dry_run=False, max_retries=2, base_delay=0.01)
        resp_429 = httpx.Response(status_code=429, headers={"Retry-After": "0"})
        resp_200 = httpx.Response(status_code=200, json={"submissionId": "sub_recovered"})

        with patch.object(httpx.Client, "post", side_effect=[resp_429, resp_200]) as mock_post:
            res = client.submit_form(self.valid_payload)
            self.assertTrue(res["success"])
            self.assertEqual(res["data"]["submissionId"], "sub_recovered")
            self.assertEqual(mock_post.call_count, 2)

    def test_20_http_429_retry_after_header_parsing(self):
        """Verify client parses and respects Retry-After header delay."""
        client = TallyClient(dry_run=False, max_retries=2)
        resp_429 = httpx.Response(status_code=429, headers={"Retry-After": "1"})
        resp_200 = httpx.Response(status_code=200, json={"submissionId": "sub_recovered"})

        with patch("time.sleep") as mock_sleep:
            with patch.object(httpx.Client, "post", side_effect=[resp_429, resp_200]):
                res = client.submit_form(self.valid_payload)
                self.assertTrue(res["success"])
                mock_sleep.assert_called_once()
                sleep_sec = mock_sleep.call_args[0][0]
                self.assertGreaterEqual(sleep_sec, 1.0)

    def test_20b_http_429_retry_after_rfc7231_date_parsing(self):
        """Verify client parses and respects RFC 7231 HTTP-date in Retry-After header."""
        import email.utils
        client = TallyClient(dry_run=False, max_retries=2)
        future_time = time.time() + 3.0
        rfc7231_date = email.utils.formatdate(future_time, usegmt=True)

        resp_429 = httpx.Response(status_code=429, headers={"Retry-After": rfc7231_date})
        resp_200 = httpx.Response(status_code=200, json={"submissionId": "sub_rfc7231_recovered"})

        with patch("time.sleep") as mock_sleep:
            with patch.object(httpx.Client, "post", side_effect=[resp_429, resp_200]):
                res = client.submit_form(self.valid_payload)
                self.assertTrue(res["success"])
                self.assertEqual(res["data"]["submissionId"], "sub_rfc7231_recovered")
                mock_sleep.assert_called_once()
                sleep_sec = mock_sleep.call_args[0][0]
                self.assertGreaterEqual(sleep_sec, 1.9)
                self.assertLessEqual(sleep_sec, 4.0)

        # Direct unit verification of _parse_retry_after method
        parsed_delay = client._parse_retry_after(resp_429)
        self.assertIsNotNone(parsed_delay)
        self.assertGreaterEqual(parsed_delay, 2.0)
        self.assertLessEqual(parsed_delay, 3.5)

    def test_21_http_429_max_retries_exhaustion(self):
        """Verify persistent HTTP 429 raises TallyRateLimitError after retries exhausted."""
        client = TallyClient(dry_run=False, max_retries=2, base_delay=0.01)
        resp_429 = httpx.Response(status_code=429, headers={"Retry-After": "0"}, text="Too Many Requests")

        with patch.object(httpx.Client, "post", side_effect=[resp_429, resp_429, resp_429]) as mock_post:
            with self.assertRaises(TallyRateLimitError) as ctx:
                client.submit_form(self.valid_payload)

            self.assertEqual(ctx.exception.status_code, 429)
            self.assertEqual(ctx.exception.attempts, 3)
            self.assertEqual(mock_post.call_count, 3)

    def test_22_http_500_transient_error_retry_and_recovery(self):
        """Verify transient HTTP 500 error triggers retry and recovers on attempt 2."""
        client = TallyClient(dry_run=False, max_retries=2, base_delay=0.01)
        resp_500 = httpx.Response(status_code=500, text="Internal Server Error")
        resp_200 = httpx.Response(status_code=200, json={"submissionId": "sub_recovered_500"})

        with patch.object(httpx.Client, "post", side_effect=[resp_500, resp_200]) as mock_post:
            res = client.submit_form(self.valid_payload)
            self.assertTrue(res["success"])
            self.assertEqual(res["data"]["submissionId"], "sub_recovered_500")
            self.assertEqual(mock_post.call_count, 2)

    def test_23_http_503_max_retries_exhaustion(self):
        """Verify persistent HTTP 503 error raises TallyServerError after retries exhausted."""
        client = TallyClient(dry_run=False, max_retries=2, base_delay=0.01)
        resp_503 = httpx.Response(status_code=503, text="Service Unavailable")

        with patch.object(httpx.Client, "post", side_effect=[resp_503, resp_503, resp_503]) as mock_post:
            with self.assertRaises(TallyServerError) as ctx:
                client.submit_form(self.valid_payload)

            self.assertEqual(ctx.exception.status_code, 503)
            self.assertEqual(ctx.exception.attempts, 3)
            self.assertEqual(mock_post.call_count, 3)

    def test_24_http_400_bad_request_fails_fast_without_retries(self):
        """Verify HTTP 400 Bad Request fails immediately without retrying."""
        client = TallyClient(dry_run=False, max_retries=3)
        resp_400 = httpx.Response(status_code=400, text='{"error": "Invalid form block"}')

        with patch.object(httpx.Client, "post", return_value=resp_400) as mock_post:
            with self.assertRaises(TallySubmissionError) as ctx:
                client.submit_form(self.valid_payload)

            self.assertEqual(ctx.exception.status_code, 400)
            self.assertEqual(ctx.exception.attempts, 1)
            self.assertEqual(mock_post.call_count, 1)

    def test_25_network_timeout_retry_and_failure(self):
        """Verify network timeout triggers retries and raises TallyConnectionError."""
        client = TallyClient(dry_run=False, max_retries=2, base_delay=0.01)
        timeout_exc = httpx.ReadTimeout("Read timed out")

        with patch.object(httpx.Client, "post", side_effect=[timeout_exc, timeout_exc, timeout_exc]) as mock_post:
            with self.assertRaises(TallyConnectionError) as ctx:
                client.submit_form(self.valid_payload)

            self.assertEqual(ctx.exception.attempts, 3)
            self.assertEqual(mock_post.call_count, 3)

    def test_26_thread_safe_concurrency(self):
        """Verify 20 concurrent threads submitting form receive distinct mock IDs without race conditions."""
        client = TallyClient(dry_run=True)
        results = []
        errors = []

        def worker():
            try:
                p = dict(self.valid_payload)
                p["sessionUuid"] = str(uuid.uuid4())
                p["respondentUuid"] = str(uuid.uuid4())
                res = client.submit(p)
                results.append(res["data"]["submissionId"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Encountered thread errors: {errors}")
        self.assertEqual(len(results), 20)
        # All 20 mock submission IDs must be unique
        self.assertEqual(len(set(results)), 20)

    def test_26b_client_property_concurrency_singleton(self):
        """Verify 20 concurrent threads accessing client.client receive identical singleton instance."""
        client = TallyClient(dry_run=False)
        client_ids = []
        errors = []
        barrier = threading.Barrier(20)

        def worker():
            try:
                barrier.wait(timeout=5.0)
                c = client.client
                client_ids.append(id(c))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        try:
            self.assertEqual(len(errors), 0, f"Thread execution encountered errors: {errors}")
            self.assertEqual(len(client_ids), 20)
            self.assertEqual(
                len(set(client_ids)),
                1,
                f"Race condition detected: {len(set(client_ids))} distinct httpx.Client instances created across 20 threads!",
            )
        finally:
            client.close()

    def test_27_async_dry_run_submission(self):
        """Verify submit_async executes successfully under asyncio."""
        client = TallyClient(dry_run=True)

        async def run_async():
            return await client.submit_async(self.valid_payload)

        res = asyncio.run(run_async())
        self.assertTrue(res["success"])
        self.assertTrue(res["data"]["dryRun"])
        self.assertRegex(res["data"]["submissionId"], DRY_RUN_ID_REGEX)



# ============================================================================
# Class 5: Adversarial Boundary & Negative Test Suite (NEG-01 to NEG-14)
# ============================================================================

class TestTallyNegativeBoundaries(unittest.TestCase):
    """Pillar 4 (cont.): Negative boundary conditions and schema fault injection."""

    def setUp(self):
        self.valid_payload = _create_valid_completed_payload()

    def test_neg_01_non_uuid_question_key(self):
        """NEG-01: Non-UUID string as question key raises TallyValidationError."""
        p = dict(self.valid_payload)
        p["responses"] = dict(self.valid_payload["responses"])
        p["responses"]["age_field"] = 21
        with self.assertRaises(TallyValidationError):
            TallySchema.validate_payload(p)

    def test_neg_02_unregistered_uuid_question_key(self):
        """NEG-02: Valid UUID v4 not present in canonical 51 questions raises TallyValidationError."""
        p = dict(self.valid_payload)
        p["responses"] = dict(self.valid_payload["responses"])
        p["responses"]["11111111-2222-4333-8444-555555555555"] = ["any-option"]
        with self.assertRaises(TallyValidationError):
            TallySchema.validate_payload(p)

    def test_neg_03_cross_question_option_hijacking(self):
        """NEG-03: Option UUID belonging to Q5 passed into Q14 raises TallyValidationError."""
        p = dict(self.valid_payload)
        p["responses"] = dict(self.valid_payload["responses"])
        q5_def = TallySchema.CODE_MAP["Q5_GENDER"]
        q5_opt = q5_def.options[0].uuid
        pu1_uuid = TallySchema.LIKERT_QUESTIONS[0].group_uuid
        p["responses"][pu1_uuid] = [q5_opt]
        with self.assertRaises(TallyValidationError):
            TallySchema.validate_payload(p)

    def test_neg_04_missing_session_uuid(self):
        """NEG-04: Missing top-level sessionUuid raises TallyValidationError."""
        p = dict(self.valid_payload)
        del p["sessionUuid"]
        with self.assertRaises(TallyValidationError):
            TallySchema.validate_payload(p)

    def test_neg_05_non_v4_uuid_format(self):
        """NEG-05: Non-v4 UUID string format raises TallyValidationError."""
        p = dict(self.valid_payload)
        p["sessionUuid"] = "12345-not-a-uuid"
        with self.assertRaises(TallyValidationError):
            TallySchema.validate_payload(p)

    def test_neg_05b_case_insensitive_session_uuid_collision(self):
        """NEG-05b: Identical sessionUuid and respondentUuid with mixed/differing case raises TallyValidationError."""
        p = dict(self.valid_payload)
        base_uuid = str(uuid.uuid4())
        p["sessionUuid"] = base_uuid.lower()
        p["respondentUuid"] = base_uuid.upper()

        with self.assertRaises(TallyValidationError) as ctx:
            TallySchema.validate_payload(p)

        self.assertIn("must be distinct entities", str(ctx.exception).lower())

    def test_neg_06_empty_responses_dict(self):
        """NEG-06: Empty responses dictionary raises TallyValidationError."""
        p = dict(self.valid_payload)
        p["responses"] = {}
        with self.assertRaises(TallyValidationError):
            TallySchema.validate_payload(p)

    def test_neg_09_age_under_18(self):
        """NEG-09: Age = 17 (outside Gen Z range) raises TallyValidationError."""
        p = dict(self.valid_payload)
        p["responses"] = dict(self.valid_payload["responses"])
        p["responses"][TallySchema.AGE_QUESTION_UUID] = 17
        with self.assertRaises(TallyValidationError):
            TallySchema.validate_payload(p)

    def test_neg_10_age_over_27(self):
        """NEG-10: Age = 28 (outside Gen Z range) raises TallyValidationError."""
        p = dict(self.valid_payload)
        p["responses"] = dict(self.valid_payload["responses"])
        p["responses"][TallySchema.AGE_QUESTION_UUID] = 28
        with self.assertRaises(TallyValidationError):
            TallySchema.validate_payload(p)

    def test_neg_11_age_non_numeric_string(self):
        """NEG-11: Age passed as string 'twenty-one' raises TallyValidationError."""
        p = dict(self.valid_payload)
        p["responses"] = dict(self.valid_payload["responses"])
        p["responses"][TallySchema.AGE_QUESTION_UUID] = "twenty-one"
        with self.assertRaises(TallyValidationError):
            TallySchema.validate_payload(p)

    def test_neg_12_single_choice_multiple_options(self):
        """NEG-12: Multiple options on single-choice question raises TallyValidationError."""
        p = dict(self.valid_payload)
        p["responses"] = dict(self.valid_payload["responses"])
        q5_def = TallySchema.CODE_MAP["Q5_GENDER"]
        p["responses"][q5_def.group_uuid] = [q5_def.options[0].uuid, q5_def.options[1].uuid]
        with self.assertRaises(TallyValidationError):
            TallySchema.validate_payload(p)

    def test_neg_13_checkbox_bare_string(self):
        """NEG-13: Checkbox value passed as bare string instead of list raises TallyValidationError."""
        p = dict(self.valid_payload)
        p["responses"] = dict(self.valid_payload["responses"])
        q11_uuid = TallySchema.CODE_MAP["Q11_CATEGORIES"].group_uuid
        p["responses"][q11_uuid] = "bare-string-instead-of-list"
        with self.assertRaises(TallyValidationError):
            TallySchema.validate_payload(p)

    def test_neg_14_missing_likert_item_in_completed_survey(self):
        """NEG-14: Missing required Likert question in completed payload raises TallyValidationError."""
        p = dict(self.valid_payload)
        p["responses"] = dict(self.valid_payload["responses"])
        del p["responses"][TallySchema.LIKERT_QUESTIONS[0].group_uuid]  # delete first likert item
        with self.assertRaises(TallyValidationError):
            TallySchema.validate_payload(p)

    def test_neg_15_missing_required_non_likert_questions(self):
        """NEG-15: Missing any required non-Likert question (Q1, Q2, Q5-Q13) raises TallyValidationError."""
        required_non_likert_codes = [
            "Q1_CONSENT",
            "Q2_USAGE",
            "Q5_GENDER",
            "Q6_LOCATION",
            "Q7_EDUCATION",
            "Q8_EMPLOYMENT",
            "Q9_PROFESSION",
            "Q10_FREQUENCY",
            "Q11_CATEGORIES",
            "Q12_PAIN_POINTS",
            "Q13_AWARENESS",
        ]

        for code in required_non_likert_codes:
            q_def = TallySchema.CODE_MAP[code]
            p = dict(self.valid_payload)
            p["responses"] = dict(self.valid_payload["responses"])
            # Remove the target required question
            del p["responses"][q_def.group_uuid]

            with self.subTest(missing_question=code):
                with self.assertRaises(TallyValidationError) as ctx:
                    TallySchema.validate_payload(p)
                self.assertIn("missing required question", str(ctx.exception).lower())

    def test_neg_16_checkbox_duplicate_options_rejected(self):
        """NEG-16: Duplicate option UUIDs in checkbox questions raise TallyValidationError."""
        checkbox_codes = ["Q11_CATEGORIES", "Q12_PAIN_POINTS"]
        for code in checkbox_codes:
            q_def = TallySchema.CODE_MAP[code]
            p = dict(self.valid_payload)
            p["responses"] = dict(self.valid_payload["responses"])
            dup_opt = q_def.options[0].uuid
            p["responses"][q_def.group_uuid] = [dup_opt, dup_opt]

            with self.subTest(checkbox_question=code):
                with self.assertRaises(TallyValidationError) as ctx:
                    TallySchema.validate_payload(p)
                self.assertIn("duplicate selections", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
