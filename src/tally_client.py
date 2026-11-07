"""
src/tally_client.py - Production-Grade Tally HTTP Client & Dry-Run Simulator
Implements high-resilience form submission to api.tally.so/forms/laqVzV/respond.

Features:
- Thread-safe httpx connection pooling (40 max, 20 keepalive) and structured timeouts.
- Exact Next.js request headers matching browser submissions.
- Exponential backoff retry with AWS Full Jitter for HTTP 429 and 5xx.
- Zero-network Dry-Run mode with pre-flight payload validation.
- Typed exception hierarchy (TallyRateLimitError, TallyServerError, TallyConnectionError).
- Dual synchronous and asynchronous execution interfaces.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import re
import socket
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple, Union

import httpx

from src.tally_schema import (
    TallyDisqualificationError,
    TallySchema,
    TallyValidationError,
)

logger = logging.getLogger("tally_client")

UUID_V4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


# ============================================================================
# Exception Hierarchy
# ============================================================================

class TallyClientError(Exception):
    """Base exception for all TallyClient errors."""
    pass


class TallySubmissionError(TallyClientError):
    """Raised when form submission fails permanently."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        attempts: Optional[int] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.payload = payload
        self.attempts = attempts


class TallyRateLimitError(TallySubmissionError):
    """Raised when HTTP 429 Rate Limit persists after retry exhaustion."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        response_body: Optional[str] = None,
        attempts: Optional[int] = None,
    ):
        super().__init__(
            message,
            status_code=429,
            response_body=response_body,
            attempts=attempts,
        )
        self.retry_after = retry_after


class TallyServerError(TallySubmissionError):
    """Raised when HTTP 5xx Server Error persists after retries."""
    pass


class TallyNetworkError(TallySubmissionError):
    """Raised when low-level network failure persists after retries."""
    pass


class TallyConnectionError(TallyNetworkError):
    """Raised when network connection or timeout fails after retries."""
    pass


# ============================================================================
# Tally Client Implementation
# ============================================================================

class TallyClient:
    """
    High-performance, resilient client for Tally Form submission.
    Thread-safe across multiple concurrent worker threads.
    """

    FORM_ID = "laqVzV"
    SUBMIT_URL = f"https://api.tally.so/forms/{FORM_ID}/respond"

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )

    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    RETRYABLE_EXCEPTIONS = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.NetworkError,
        httpx.TimeoutException,
    )

    def __init__(
        self,
        form_id: str = FORM_ID,
        dry_run: bool = False,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 16.0,
        timeout: float = 15.0,
        user_agent: Optional[str] = None,
        simulate_latency: bool = False,
        dry_run_latency_range: Tuple[float, float] = (0.0, 0.0),
        pool_limits: Optional[httpx.Limits] = None,
    ):
        self.form_id = form_id
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self.simulate_latency = simulate_latency
        self.dry_run_latency_range = dry_run_latency_range

        self.endpoint = f"https://api.tally.so/forms/{self.form_id}/respond"

        # Connection limits sized for fleet of up to 20 concurrent workers
        self._pool_limits = pool_limits or httpx.Limits(
            max_connections=40,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        )

        # Structured timeouts
        self._timeout = httpx.Timeout(
            timeout=timeout,
            connect=5.0,
            read=timeout,
            write=5.0,
            pool=5.0,
        )

        # Concurrency synchronization lock
        self._lock = threading.Lock()

        # Lazy HTTP client singletons
        self._client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------------
    # Header Construction
    # ------------------------------------------------------------------------

    def get_headers(self) -> Dict[str, str]:
        """Construct exact Next.js headers matching web submission."""
        return {
            "Host": "api.tally.so",
            "User-Agent": self.user_agent,
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
            "Origin": "https://tally.so",
            "Referer": f"https://tally.so/r/{self.form_id}",
            "tally-version": "2025-01-15",
            "Sec-Ch-Ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
        }

    def _build_headers(self) -> Dict[str, str]:
        """Alias for get_headers."""
        return self.get_headers()

    # ------------------------------------------------------------------------
    # Pre-flight Validation
    # ------------------------------------------------------------------------

    def validate_payload(self, payload: Dict[str, Any]) -> None:
        """
        Validate outgoing payload integrity before network transmission.
        Ensures compliance with Tally API contract.
        """
        if not isinstance(payload, dict):
            raise TallyValidationError("Payload must be a dictionary")

        required_keys = {"sessionUuid", "respondentUuid", "responses", "isCompleted"}
        missing = required_keys - set(payload.keys())
        if missing:
            raise TallyValidationError(f"Payload missing required top-level keys: {sorted(missing)}")

        for uuid_key in ("sessionUuid", "respondentUuid"):
            val = payload.get(uuid_key)
            if not isinstance(val, str) or not UUID_V4_PATTERN.match(val):
                raise TallyValidationError(f"Invalid UUID v4 for {uuid_key}: {val}")

        if str(payload["sessionUuid"]).strip().lower() == str(payload["respondentUuid"]).strip().lower():
            raise TallyValidationError("sessionUuid and respondentUuid must be distinct entities")

        responses = payload.get("responses")
        if not isinstance(responses, dict) or len(responses) == 0:
            raise TallyValidationError("Payload 'responses' must be a non-empty dictionary")

        if payload.get("isCompleted") is not True:
            raise TallyValidationError("Payload 'isCompleted' must be True for survey submission")

        # Run full schema validation
        TallySchema.validate_payload(payload, raise_exception=True)

    # ------------------------------------------------------------------------
    # Backoff & Retry Helpers
    # ------------------------------------------------------------------------

    def _parse_retry_after(self, response: httpx.Response) -> Optional[float]:
        """Extract and parse Retry-After header as seconds float."""
        raw = response.headers.get("Retry-After")
        if not raw:
            return None

        # Format 1: Direct integer/float seconds
        try:
            val = float(raw)
            return max(0.0, min(60.0, val))
        except ValueError:
            pass

        # Format 2: RFC 7231 HTTP-date
        try:
            target_time = email.utils.parsedate_to_datetime(raw).timestamp()
            delta = target_time - time.time()
            return max(0.0, min(60.0, delta))
        except Exception:
            return None

    def _calculate_backoff(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """
        Calculate backoff delay using AWS Full Jitter.
        attempt is 1-indexed (1, 2, ...).
        """
        if retry_after is not None:
            if retry_after == 0.0:
                return 0.0
            delay = min(self.max_delay, retry_after)
            jitter = random.uniform(0.05, 0.2)
            return delay + jitter

        cap = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        min_sleep = min(cap, self.base_delay * 0.5)
        sleep_duration = random.uniform(min_sleep, cap)
        return sleep_duration

    # ------------------------------------------------------------------------
    # Synchronous Execution
    # ------------------------------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        """Lazy thread-safe httpx.Client initialization using double-checked locking."""
        client = self._client
        if client is None or client.is_closed:
            with self._lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.Client(
                        limits=self._pool_limits,
                        timeout=self._timeout,
                        headers=self.get_headers(),
                    )
                return self._client
        return client

    def submit(
        self,
        payload: Dict[str, Any],
        dry_run: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Synchronously submit form payload to Tally backend or Dry-Run simulator.
        Thread-safe for worker execution.
        """
        effective_dry_run = self.dry_run if dry_run is None else dry_run

        # 1. Pre-flight check (enforced even in dry-run mode)
        self.validate_payload(payload)

        # 2. Zero-Network Dry-Run Intercept
        if effective_dry_run:
            if self.simulate_latency and self.dry_run_latency_range[1] > 0:
                delay = random.uniform(
                    self.dry_run_latency_range[0],
                    self.dry_run_latency_range[1],
                )
                if delay > 0:
                    time.sleep(delay)

            mock_submission_id = f"mock-dryrun-{uuid.uuid4()}"
            logger.info(f"[Dry-Run] Mock submission generated: {mock_submission_id}")
            return {
                "success": True,
                "data": {
                    "submissionId": mock_submission_id,
                    "dryRun": True,
                },
            }

        # 3. Live HTTP Transmission with Resilient Retries
        last_error: Optional[Exception] = None
        last_status: Optional[int] = None
        last_body: Optional[str] = None
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                response = self.client.post(self.endpoint, json=payload)
                status = response.status_code

                # Success
                if status in (200, 201):
                    try:
                        data = response.json()
                    except Exception:
                        data = {"submissionId": f"sub_{uuid.uuid4().hex[:14]}"}
                    logger.info(f"Form submission succeeded: {data.get('submissionId')}")
                    return {
                        "success": True,
                        "data": {
                            "submissionId": data.get("submissionId"),
                            "dryRun": False,
                        },
                    }

                # Retryable Rate Limit (429)
                if status == 429:
                    last_status = 429
                    last_body = response.text
                    retry_after = self._parse_retry_after(response)

                    if attempt < total_attempts:
                        sleep_sec = self._calculate_backoff(attempt, retry_after)
                        logger.warning(
                            f"HTTP 429 Rate Limit (attempt {attempt}/{total_attempts}). "
                            f"Backing off for {sleep_sec:.2f}s..."
                        )
                        if sleep_sec > 0:
                            time.sleep(sleep_sec)
                        continue
                    else:
                        raise TallyRateLimitError(
                            f"Rate limit exceeded after {attempt} attempts: {last_body}",
                            retry_after=retry_after,
                            response_body=last_body,
                            attempts=attempt,
                        )

                # Retryable Server Errors (5xx)
                if status in self.RETRYABLE_STATUS_CODES:
                    last_status = status
                    last_body = response.text
                    if attempt < total_attempts:
                        sleep_sec = self._calculate_backoff(attempt)
                        logger.warning(
                            f"HTTP {status} Server Error (attempt {attempt}/{total_attempts}). "
                            f"Retrying in {sleep_sec:.2f}s..."
                        )
                        if sleep_sec > 0:
                            time.sleep(sleep_sec)
                        continue
                    else:
                        raise TallyServerError(
                            f"HTTP {status} Server Error exhausted {attempt} attempts: {last_body}",
                            status_code=status,
                            response_body=last_body,
                            payload=payload,
                            attempts=attempt,
                        )

                # Terminal Client Errors (400, 401, 403, 404, 422, etc.)
                raise TallySubmissionError(
                    f"Submission rejected by Tally with HTTP {status}: {response.text}",
                    status_code=status,
                    response_body=response.text,
                    payload=payload,
                    attempts=attempt,
                )

            except self.RETRYABLE_EXCEPTIONS as exc:
                last_error = exc
                if attempt < total_attempts:
                    sleep_sec = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Network exception {exc.__class__.__name__} on attempt {attempt}/{total_attempts}. "
                        f"Retrying in {sleep_sec:.2f}s..."
                    )
                    if sleep_sec > 0:
                        time.sleep(sleep_sec)
                    continue
                else:
                    raise TallyConnectionError(
                        f"Network connection failed after {attempt} attempts: {exc}",
                        status_code=last_status,
                        response_body=last_body,
                        payload=payload,
                        attempts=attempt,
                    ) from exc

        # Fallback
        raise TallySubmissionError(
            f"Unexpected submission loop exit: {last_error or last_status}",
            status_code=last_status,
            response_body=last_body,
            payload=payload,
            attempts=total_attempts,
        )

    def submit_form(
        self,
        payload: Dict[str, Any],
        dry_run: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Alias for submit() matching Scope and Test specs."""
        return self.submit(payload=payload, dry_run=dry_run)

    # ------------------------------------------------------------------------
    # Asynchronous Execution
    # ------------------------------------------------------------------------

    @property
    def async_client(self) -> httpx.AsyncClient:
        """Lazy thread-safe httpx.AsyncClient initialization using double-checked locking."""
        client = self._async_client
        if client is None or client.is_closed:
            with self._lock:
                if self._async_client is None or self._async_client.is_closed:
                    self._async_client = httpx.AsyncClient(
                        limits=self._pool_limits,
                        timeout=self._timeout,
                        headers=self.get_headers(),
                    )
                return self._async_client
        return client

    async def submit_async(
        self,
        payload: Dict[str, Any],
        dry_run: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Asynchronous submission counterpart."""
        effective_dry_run = self.dry_run if dry_run is None else dry_run

        self.validate_payload(payload)

        if effective_dry_run:
            if self.simulate_latency and self.dry_run_latency_range[1] > 0:
                delay = random.uniform(
                    self.dry_run_latency_range[0],
                    self.dry_run_latency_range[1],
                )
                if delay > 0:
                    await asyncio.sleep(delay)

            mock_submission_id = f"mock-dryrun-{uuid.uuid4()}"
            return {
                "success": True,
                "data": {
                    "submissionId": mock_submission_id,
                    "dryRun": True,
                },
            }

        last_error: Optional[Exception] = None
        last_status: Optional[int] = None
        last_body: Optional[str] = None
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                response = await self.async_client.post(self.endpoint, json=payload)
                status = response.status_code

                if status in (200, 201):
                    try:
                        data = response.json()
                    except Exception:
                        data = {"submissionId": f"sub_{uuid.uuid4().hex[:14]}"}
                    return {
                        "success": True,
                        "data": {
                            "submissionId": data.get("submissionId"),
                            "dryRun": False,
                        },
                    }

                if status == 429:
                    last_status = 429
                    last_body = response.text
                    retry_after = self._parse_retry_after(response)

                    if attempt < total_attempts:
                        sleep_sec = self._calculate_backoff(attempt, retry_after)
                        if sleep_sec > 0:
                            await asyncio.sleep(sleep_sec)
                        continue
                    else:
                        raise TallyRateLimitError(
                            f"Rate limit exceeded after {attempt} attempts: {last_body}",
                            retry_after=retry_after,
                            response_body=last_body,
                            attempts=attempt,
                        )

                if status in self.RETRYABLE_STATUS_CODES:
                    last_status = status
                    last_body = response.text
                    if attempt < total_attempts:
                        sleep_sec = self._calculate_backoff(attempt)
                        if sleep_sec > 0:
                            await asyncio.sleep(sleep_sec)
                        continue
                    else:
                        raise TallyServerError(
                            f"HTTP {status} Server Error exhausted {attempt} attempts: {last_body}",
                            status_code=status,
                            response_body=last_body,
                            payload=payload,
                            attempts=attempt,
                        )

                raise TallySubmissionError(
                    f"Submission rejected by Tally with HTTP {status}: {response.text}",
                    status_code=status,
                    response_body=response.text,
                    payload=payload,
                    attempts=attempt,
                )

            except self.RETRYABLE_EXCEPTIONS as exc:
                last_error = exc
                if attempt < total_attempts:
                    sleep_sec = self._calculate_backoff(attempt)
                    if sleep_sec > 0:
                        await asyncio.sleep(sleep_sec)
                    continue
                else:
                    raise TallyConnectionError(
                        f"Network connection failed after {attempt} attempts: {exc}",
                        status_code=last_status,
                        response_body=last_body,
                        payload=payload,
                        attempts=attempt,
                    ) from exc

        raise TallySubmissionError(
            f"Unexpected async submission loop exit: {last_error or last_status}",
            status_code=last_status,
            response_body=last_body,
            payload=payload,
            attempts=total_attempts,
        )

    # ------------------------------------------------------------------------
    # Cleanup & Context Managers
    # ------------------------------------------------------------------------

    def close(self) -> None:
        """Close synchronous client connection pool in a thread-safe, idempotent manner."""
        with self._lock:
            if self._client is not None:
                try:
                    if not self._client.is_closed:
                        self._client.close()
                finally:
                    self._client = None

    async def close_async(self) -> None:
        """Close asynchronous client connection pool in a thread-safe, idempotent manner."""
        client_to_close = None
        with self._lock:
            if self._async_client is not None:
                client_to_close = self._async_client
                self._async_client = None
        if client_to_close is not None and not client_to_close.is_closed:
            await client_to_close.aclose()

    def __enter__(self) -> "TallyClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    async def __aenter__(self) -> "TallyClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close_async()
