"""
tally_schema.py - Authoritative Tally Form J118j4 Schema & API Contract
Encapsulates 52 questions, 293 option UUIDs, screening logic, payload serialization,
and validation rules for api.tally.so/forms/J118j4/respond.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import warnings

warnings.filterwarnings(
    "ignore",
    message=r'Field name "construct" .* shadows an attribute in parent "BaseModel"',
    category=UserWarning,
)

logger = logging.getLogger("TallySchema")

# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------
FORM_ID = "J118j4"
FORM_URL = f"https://tally.so/r/{FORM_ID}"
SUBMIT_URL = f"https://api.tally.so/forms/{FORM_ID}/respond"

# Critical Screening UUIDs
CONSENT_QUESTION_UUID = "3e5a203f-e81c-45f4-88c8-70aa4cc36696"
CONSENT_AGREE_UUID = "1b84910e-a410-455c-8595-f4f73c7ea7ff"
CONSENT_DISAGREE_UUID = "18b5cbd6-040e-4ef8-b213-7fc038f1e6df"

USAGE_QUESTION_UUID = "c2a8c967-6d50-4b90-981e-2b4acbb5af01"
USAGE_NEVER_UUID = "42a1fc17-7d7a-4cc4-9eeb-3df094a64b86"  # DISQUALIFICATION TRAP
USAGE_QUALIFIED_UUIDS = [
    "9232238d-d74e-4520-85c4-f87c8b827c1c",  # Today / yesterday
    "15c0ab5e-e45d-4a8b-8234-3a6eb117898b",  # Within 1 week
    "0ed4d205-154f-4333-94ea-3c1cd02812bf",  # 2 weeks to 1 month
    "cd8db0c7-34c8-4a41-a608-a7206f6ec8a9",  # 1 to 6 months
    "ec6f15d1-7feb-49ef-aa4a-58b6c6a22d56",  # > 6 months
]
DEFAULT_QUALIFIED_USAGE_UUID = "15c0ab5e-e45d-4a8b-8234-3a6eb117898b"

REASON_NEVER_QUESTION_UUID = "e2948fdb-2d9a-413c-8a8b-96ff61314f0c"
AGE_QUESTION_UUID = "d8ec1647-4508-481b-b8ea-302737d9a5e9"

UUID_V4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

DEFAULT_HEADERS = {
    "Host": "api.tally.so",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
    "Origin": "https://tally.so",
    "Referer": FORM_URL,
    "tally-version": "2025-01-15",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
    "Sec-Ch-Ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}


class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "MULTIPLE_CHOICE"
    CHECKBOXES = "CHECKBOXES"
    INPUT_NUMBER = "INPUT_NUMBER"


class TallyValidationError(ValueError):
    """Raised when a Tally response payload fails schema validation."""
    pass


class TallyDisqualificationError(TallyValidationError):
    """Raised when a response triggers early survey termination / disqualification."""
    pass


# ---------------------------------------------------------------------------
# Domain Models (Pydantic v2)
# ---------------------------------------------------------------------------
class OptionDefinition(BaseModel):
    uuid: str
    text: str
    score: Optional[int] = None


class QuestionDefinition(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    group_uuid: str
    index: int
    code: str
    construct: str
    construct_code: Optional[str] = None
    item_index: Optional[int] = None
    title: str
    type: QuestionType
    required: bool = True
    is_likert: bool = False
    is_reverse: bool = False
    options: List[OptionDefinition] = Field(default_factory=list)

    @property
    def option_uuids(self) -> Set[str]:
        return {opt.uuid for opt in self.options}

    @property
    def option_map(self) -> Dict[str, OptionDefinition]:
        return {opt.uuid: opt for opt in self.options}

    @property
    def uuid_to_text(self) -> Dict[str, str]:
        return {opt.uuid: opt.text for opt in self.options}

    @property
    def text_to_uuid(self) -> Dict[str, str]:
        return {opt.text: opt.uuid for opt in self.options}

    @property
    def score_to_uuid(self) -> Dict[int, str]:
        return {opt.score: opt.uuid for opt in self.options if opt.score is not None}

    @property
    def uuid_to_score(self) -> Dict[str, int]:
        return {opt.uuid: opt.score for opt in self.options if opt.score is not None}


class TallyResponsePayload(BaseModel):
    sessionUuid: str
    respondentUuid: str
    responses: Dict[str, Any]
    captchas: Dict[str, Any] = Field(default_factory=dict)
    isCompleted: bool = True
    password: Optional[str] = None
    locale: str = "vi"

    @field_validator("sessionUuid", "respondentUuid")
    @classmethod
    def validate_uuid_format(cls, v: str) -> str:
        s = str(v)
        if not UUID_V4_PATTERN.match(s):
            raise ValueError(f"Invalid UUID v4 string format: {v}")
        return s

    @model_validator(mode="after")
    def validate_distinct_uuids(self) -> TallyResponsePayload:
        if (
            self.sessionUuid
            and self.respondentUuid
            and str(self.sessionUuid).lower() == str(self.respondentUuid).lower()
        ):
            raise ValueError("sessionUuid and respondentUuid must be distinct entities")
        return self


# ---------------------------------------------------------------------------
# Canonical Schema Registry Class
# ---------------------------------------------------------------------------
class TallySchema:
    FORM_ID = FORM_ID
    FORM_URL = FORM_URL
    ENDPOINT = SUBMIT_URL
    SUBMIT_URL = SUBMIT_URL
    HEADERS = DEFAULT_HEADERS

    # Screening constants
    CONSENT_QUESTION_UUID = CONSENT_QUESTION_UUID
    CONSENT_AGREE_UUID = CONSENT_AGREE_UUID
    CONSENT_DISAGREE_UUID = CONSENT_DISAGREE_UUID
    USAGE_QUESTION_UUID = USAGE_QUESTION_UUID
    USAGE_NEVER_UUID = USAGE_NEVER_UUID
    USAGE_QUALIFIED_UUIDS = USAGE_QUALIFIED_UUIDS
    DEFAULT_QUALIFIED_USAGE_UUID = DEFAULT_QUALIFIED_USAGE_UUID
    REASON_NEVER_QUESTION_UUID = REASON_NEVER_QUESTION_UUID
    AGE_QUESTION_UUID = AGE_QUESTION_UUID

    # Registries populated at module load
    QUESTIONS: List[QuestionDefinition] = []
    QUESTION_MAP: Dict[str, QuestionDefinition] = {}
    CODE_MAP: Dict[str, QuestionDefinition] = {}
    LIKERT_QUESTIONS: List[QuestionDefinition] = []
    CONSTRUCT_MAP: Dict[str, List[QuestionDefinition]] = {}
    ALL_GROUP_UUIDS: Set[str] = set()
    ALL_OPTION_UUIDS: Set[str] = set()

    @classmethod
    def initialize(cls, schema_path: Optional[Union[str, Path]] = None) -> None:
        """Loads and binds the canonical 52 questions and 293 options."""
        if cls.QUESTIONS:
            return  # Already initialized

        candidates = []
        if schema_path is not None:
            candidates.append(Path(schema_path))
        candidates.extend([
            Path(__file__).resolve().parent.parent / "consolidated_schema.json",
            Path(__file__).resolve().parent / "consolidated_schema.json",
            Path("consolidated_schema.json"),
        ])

        target_file = None
        for c in candidates:
            if c.exists():
                target_file = c
                break

        if target_file is None:
            raise FileNotFoundError(
                f"Cannot locate consolidated_schema.json. Searched: {[str(p) for p in candidates]}"
            )

        with open(target_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        cls._load_raw_schema(raw_data)

    @classmethod
    def _load_raw_schema(cls, raw_data: List[Dict[str, Any]]) -> None:
        cls.QUESTIONS.clear()
        cls.QUESTION_MAP.clear()
        cls.CODE_MAP.clear()
        cls.LIKERT_QUESTIONS.clear()
        cls.CONSTRUCT_MAP.clear()
        cls.ALL_GROUP_UUIDS.clear()
        cls.ALL_OPTION_UUIDS.clear()

        construct_code_map = {
            "Perceived Usefulness: Nhận thức về sự hữu ích": "PU",
            "Perceived Ease of Use: Nhận thức về tính dễ sử dụng": "PEOU",
            "Perceived Environmental Sustainability: Nhận thức về tính bền vững môi trường": "PES",
            "Perceived Costs: Nhận thức về chi phí": "PC",
            "Peer Influence: Ảnh hưởng từ người xung quanh": "PI",
            "Self-Efficacy: Sự tự tin vào năng lực bản thân": "SE",
            "Location Convenience: Sự thuận tiện về vị trí": "LC",
            "Attitude: Thái độ đối với dịch vụ": "ATT",
            "Subjective Norm: Chuẩn chủ quan": "SN",
            "Perceived Behavioral Control: Nhận thức kiểm soát hành vi": "PBC",
            "Intention to Use: Ý định sử dụng dịch vụ": "ITU",
        }

        construct_item_counts: Dict[str, int] = {}

        for idx, item in enumerate(raw_data):
            q_idx = idx + 1
            group_uuid = item["groupUuid"]
            q_type = QuestionType(item["type"])
            raw_construct = item.get("construct", "General")
            title = item.get("title", "")

            c_code = construct_code_map.get(raw_construct)
            is_likert = c_code is not None
            is_reverse = False
            item_index = None

            if is_likert:
                construct_item_counts[c_code] = construct_item_counts.get(c_code, 0) + 1
                item_index = construct_item_counts[c_code]
                code = f"{c_code}{item_index}"
                if c_code == "PC":
                    is_reverse = True
            else:
                code_gen_map = {
                    1: "Q1_CONSENT",
                    2: "Q2_USAGE",
                    3: "Q3_REASON_NEVER",
                    4: "Q4_AGE",
                    5: "Q5_GENDER",
                    6: "Q6_LOCATION",
                    7: "Q7_EDUCATION",
                    8: "Q8_EMPLOYMENT",
                    9: "Q9_PROFESSION",
                    10: "Q10_FREQUENCY",
                    11: "Q11_CATEGORIES",
                    12: "Q12_PAIN_POINTS",
                    13: "Q13_AWARENESS",
                }
                code = code_gen_map.get(q_idx, f"Q{q_idx}")

            options: List[OptionDefinition] = []
            for opt_idx, opt_raw in enumerate(item.get("options", [])):
                opt_uuid = opt_raw["uuid"]
                opt_text = opt_raw["text"]
                cls.ALL_OPTION_UUIDS.add(opt_uuid)
                score = (opt_idx + 1) if is_likert else None
                options.append(OptionDefinition(uuid=opt_uuid, text=opt_text, score=score))

            q_def = QuestionDefinition(
                group_uuid=group_uuid,
                index=q_idx,
                code=code,
                construct=raw_construct,
                construct_code=c_code,
                item_index=item_index,
                title=title,
                type=q_type,
                required=True,
                is_likert=is_likert,
                is_reverse=is_reverse,
                options=options,
            )

            cls.QUESTIONS.append(q_def)
            cls.QUESTION_MAP[group_uuid] = q_def
            cls.CODE_MAP[code] = q_def
            cls.ALL_GROUP_UUIDS.add(group_uuid)

            if is_likert:
                cls.LIKERT_QUESTIONS.append(q_def)
                cls.CONSTRUCT_MAP.setdefault(c_code, []).append(q_def)

    @classmethod
    def get_payload(
        cls,
        responses: Dict[str, Any],
        session_uuid: Optional[str] = None,
        respondent_uuid: Optional[str] = None,
        is_completed: bool = True,
        enforce_screening: bool = False,
    ) -> Dict[str, Any]:
        """Formats and validates the exact payload expected by Tally respond API."""
        if not cls.QUESTIONS:
            cls.initialize()

        s_uuid = session_uuid or str(uuid.uuid4())
        r_uuid = respondent_uuid or str(uuid.uuid4())

        formatted_responses: Dict[str, Any] = {}

        for key, value in responses.items():
            q_def = cls.QUESTION_MAP.get(key) or cls.CODE_MAP.get(key)
            if not q_def:
                # Retain key as is so validator can detect invalid/unknown keys
                formatted_responses[key] = value
                continue

            g_uuid = q_def.group_uuid

            if q_def.type == QuestionType.INPUT_NUMBER:
                if isinstance(value, bool):
                    formatted_responses[g_uuid] = value
                elif isinstance(value, (int, float)):
                    formatted_responses[g_uuid] = int(value) if isinstance(value, int) or value.is_integer() else value
                elif isinstance(value, str):
                    try:
                        num = float(value)
                        formatted_responses[g_uuid] = int(num) if num.is_integer() else num
                    except ValueError:
                        formatted_responses[g_uuid] = value
                else:
                    formatted_responses[g_uuid] = value
            elif q_def.type == QuestionType.MULTIPLE_CHOICE:
                if isinstance(value, int) and not isinstance(value, bool) and q_def.is_likert:
                    opt_uuid = q_def.score_to_uuid.get(value)
                    formatted_responses[g_uuid] = [opt_uuid] if opt_uuid else [str(value)]
                elif isinstance(value, str):
                    formatted_responses[g_uuid] = [value]
                elif isinstance(value, list):
                    formatted_responses[g_uuid] = value
                else:
                    formatted_responses[g_uuid] = value
            elif q_def.type == QuestionType.CHECKBOXES:
                if isinstance(value, str):
                    formatted_responses[g_uuid] = [value]
                elif isinstance(value, list):
                    formatted_responses[g_uuid] = value
                else:
                    formatted_responses[g_uuid] = value

        if enforce_screening:
            cls.enforce_screening_logic(formatted_responses)

        raw_payload = {
            "sessionUuid": s_uuid,
            "respondentUuid": r_uuid,
            "responses": formatted_responses,
            "captchas": {},
            "isCompleted": is_completed,
            "password": None,
            "locale": "vi",
        }

        return raw_payload

    @classmethod
    def enforce_screening_logic(cls, responses: Dict[str, Any]) -> Dict[str, Any]:
        """Enforces screening conditions so respondent never gets disqualified."""
        # Rule 1: Consent must be AGREE
        responses[CONSENT_QUESTION_UUID] = [CONSENT_AGREE_UUID]

        # Rule 2: Shopee usage must not be 'never'
        current_usage = responses.get(USAGE_QUESTION_UUID)
        if not current_usage or current_usage == [USAGE_NEVER_UUID]:
            responses[USAGE_QUESTION_UUID] = [DEFAULT_QUALIFIED_USAGE_UUID]

        # Rule 3: Never trigger Q3 reason checkboxes
        responses.pop(REASON_NEVER_QUESTION_UUID, None)
        return responses

    @classmethod
    def is_disqualified(cls, responses: Dict[str, Any]) -> Tuple[bool, str]:
        """Checks if a responses dict would trigger early termination on Page 7."""
        consent = responses.get(CONSENT_QUESTION_UUID)
        if consent == [CONSENT_DISAGREE_UUID]:
            return True, "Consent Disagreed (Triggers Logic Block 1 -> Page 7 Disqualification)"

        usage = responses.get(USAGE_QUESTION_UUID)
        if usage == [USAGE_NEVER_UUID]:
            return True, "Never Used Shopee (Triggers Logic Block 2 & 3 -> Page 7 Disqualification)"

        return False, "Qualified"

    @classmethod
    def get_validation_errors(cls, payload: Dict[str, Any]) -> List[str]:
        """Returns exhaustive list of validation violations in payload."""
        if not cls.QUESTIONS:
            cls.initialize()

        errors: List[str] = []
        if not isinstance(payload, dict):
            return ["Payload must be a dictionary."]

        # 1. Top-level envelope checks
        for key in ["sessionUuid", "respondentUuid", "responses", "isCompleted"]:
            if key not in payload:
                errors.append(f"Missing required envelope key: '{key}'")

        session_uuid = payload.get("sessionUuid")
        respondent_uuid = payload.get("respondentUuid")

        if session_uuid is not None:
            if not isinstance(session_uuid, str) or not UUID_V4_PATTERN.match(session_uuid):
                errors.append(f"Invalid UUID v4 for sessionUuid: {session_uuid}")

        if respondent_uuid is not None:
            if not isinstance(respondent_uuid, str) or not UUID_V4_PATTERN.match(respondent_uuid):
                errors.append(f"Invalid UUID v4 for respondentUuid: {respondent_uuid}")

        if (
            session_uuid is not None
            and respondent_uuid is not None
            and str(session_uuid).lower() == str(respondent_uuid).lower()
        ):
            errors.append("sessionUuid and respondentUuid must be distinct entities")

        if payload.get("isCompleted") is not True:
            errors.append("Envelope key 'isCompleted' must be True.")

        responses = payload.get("responses")
        if not isinstance(responses, dict) or len(responses) == 0:
            errors.append("Payload 'responses' must be a non-empty dictionary.")
            return errors

        # 2. Check question group UUIDs
        for g_uuid in responses.keys():
            if not isinstance(g_uuid, str):
                errors.append(f"Question group UUID must be a string: {g_uuid}")
                continue
            try:
                uuid.UUID(g_uuid)
            except (ValueError, TypeError):
                errors.append(f"Invalid UUID format for question groupUuid: '{g_uuid}'")
                continue

            if g_uuid not in cls.ALL_GROUP_UUIDS:
                errors.append(f"Unrecognized question groupUuid in responses: '{g_uuid}'")

        # 3. Screening logic check
        is_dq, dq_reason = cls.is_disqualified(responses)
        if is_dq:
            errors.append(f"Screening disqualification: {dq_reason}")

        # 4. Demographic age check (Q4)
        if AGE_QUESTION_UUID in responses:
            age_val = responses[AGE_QUESTION_UUID]
            if isinstance(age_val, bool) or not isinstance(age_val, (int, float)):
                errors.append(f"Age value must be a number: {age_val}")
            elif not (18 <= age_val <= 27):
                errors.append(f"Invalid Gen Z age: {age_val}. Must be between 18 and 27.")
        elif payload.get("isCompleted"):
            errors.append("Missing required Age question (Q4).")

        # 5. Type and Option cardinality checks
        for g_uuid, val in responses.items():
            q_def = cls.QUESTION_MAP.get(g_uuid)
            if not q_def:
                continue

            if q_def.type == QuestionType.INPUT_NUMBER:
                if isinstance(val, list):
                    errors.append(f"INPUT_NUMBER question {q_def.code} must be numeric scalar, not list.")
            elif q_def.type == QuestionType.MULTIPLE_CHOICE:
                if not isinstance(val, list):
                    errors.append(f"MULTIPLE_CHOICE question {q_def.code} must be a list of 1 option UUID.")
                elif len(val) != 1:
                    errors.append(
                        f"MULTIPLE_CHOICE question {q_def.code} accepts exactly 1 option UUID, got {len(val)}."
                    )
                else:
                    opt_uuid = val[0]
                    if opt_uuid not in q_def.option_uuids:
                        errors.append(
                            f"Option UUID '{opt_uuid}' does not belong to question {q_def.code}."
                        )
            elif q_def.type == QuestionType.CHECKBOXES:
                if not isinstance(val, list):
                    errors.append(f"CHECKBOXES question {q_def.code} must be a list of option UUIDs.")
                elif len(val) == 0:
                    errors.append(f"CHECKBOXES question {q_def.code} cannot have empty selection.")
                else:
                    if len(val) != len(set(x.lower() if isinstance(x, str) else x for x in val)):
                        errors.append(f"CHECKBOXES question {q_def.code} contains duplicate selections.")
                    for opt_uuid in val:
                        if opt_uuid not in q_def.option_uuids:
                            errors.append(
                                f"Option UUID '{opt_uuid}' does not belong to checkbox question {q_def.code}."
                            )

        # 6. Completed payload required question checks
        if payload.get("isCompleted"):
            for q_def in cls.QUESTIONS:
                # Q3 REASON_NEVER is only for disqualified non-Shopee flow
                if q_def.group_uuid == cls.REASON_NEVER_QUESTION_UUID:
                    continue
                # Q4 Age is already validated in step 4
                if q_def.group_uuid == cls.AGE_QUESTION_UUID:
                    continue
                ans = responses.get(q_def.group_uuid)
                if ans is None:
                    if q_def.is_likert:
                        errors.append(f"Missing required Likert item: {q_def.code}")
                    else:
                        errors.append(f"Missing required question: {q_def.code}")

        return errors

    @classmethod
    def validate_payload(cls, payload: Dict[str, Any], raise_exception: bool = True) -> bool:
        """
        Validates payload against canonical schema and screening rules.
        Raises TallyDisqualificationError if disqualified, TallyValidationError if invalid.
        Returns True if payload is completely valid.
        """
        responses = payload.get("responses", {}) if isinstance(payload, dict) else {}
        is_dq, dq_reason = cls.is_disqualified(responses)
        if is_dq:
            if raise_exception:
                raise TallyDisqualificationError(f"Screening failure: {dq_reason}")
            return False

        errors = cls.get_validation_errors(payload)
        if errors:
            if raise_exception:
                raise TallyValidationError("Payload validation failed:\n - " + "\n - ".join(errors))
            return False

        return True

    @classmethod
    def get_likert_option_uuid(cls, question_key: str, score: int) -> str:
        """Helper to get option UUID for Likert score 1..6."""
        if not cls.QUESTIONS:
            cls.initialize()
        q_def = cls.QUESTION_MAP.get(question_key) or cls.CODE_MAP.get(question_key)
        if not q_def or not q_def.is_likert:
            raise KeyError(f"Question '{question_key}' is not a valid Likert question.")
        opt_uuid = q_def.score_to_uuid.get(score)
        if not opt_uuid:
            raise ValueError(f"Score {score} out of range [1, 6] for question {question_key}")
        return opt_uuid

    @classmethod
    def reverse_likert_score(cls, score: int) -> int:
        """Inverts 1..6 Likert score for reverse-coded PC construct (7 - score)."""
        return 7 - score


# Initialize automatically upon module import
TallySchema.initialize()
