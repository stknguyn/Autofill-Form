"""
Autofill-Form Source Package.

Core modules:
- tally_schema: Canonical 52-question and 293-option schema, screening logic, payload serialization.
- tally_client: High-resilience HTTP client and zero-network Dry-Run simulator.
"""

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

__all__ = [
    "TallySchema",
    "QuestionType",
    "QuestionDefinition",
    "OptionDefinition",
    "TallyResponsePayload",
    "TallyValidationError",
    "TallyDisqualificationError",
    "TallyClient",
    "TallyClientError",
    "TallySubmissionError",
    "TallyRateLimitError",
    "TallyServerError",
    "TallyNetworkError",
    "TallyConnectionError",
]
