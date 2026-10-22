"""HTTP adapters expose only typed Request inputs, never source/label records."""

import json
import math
import os
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from .budget import BudgetExceeded, Ledger
from .config import Config, ModelConfig
from .models import Prediction, Request
from .prompts import classification_messages, classification_schema
from .providers import argo_model
from .storage import canonical, read_json


def usage_fields(body: dict) -> dict:
    """Extract token counts and provider-reported cost from a response body.

    Accepts both OpenAI-style (``prompt_tokens``/``completion_tokens``) and Anthropic-style
    (``input_tokens``/``output_tokens``) field names. Returns only the fields that are
    present and valid; missing or malformed fields are silently omitted.
    """
    usage = body.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    result = {}
    for target, keys in (
        ("input_tokens", ("input_tokens", "prompt_tokens")),
        ("output_tokens", ("output_tokens", "completion_tokens")),
    ):
        val = next((usage[k] for k in keys if k in usage), None)
        if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
            result[target] = val
    cost = usage.get("cost")
    if (
        isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and math.isfinite(cost)
        and cost >= 0
    ):
        result["provider_cost_usd"] = cost
    model = body.get("model")
    if isinstance(model, str) and model:
        result["returned_model"] = model
    return result


def parse_decisions(body: dict) -> Prediction:
    """Parse an OpenRouter Decisions API response into a Prediction.

    Expects a ``triage`` answer of type ``choice`` with a full probability distribution.
    Provider abstention and missing distributions are each surfaced as distinct error codes.
    """
    common = usage_fields(body)
    answers = body.get("answers")
    answer = answers.get("triage") if isinstance(answers, dict) else None
    if not isinstance(answer, dict):
        raise ValueError("Expected a triage answer object")
    if answer.get("abstention") is True:
        return Prediction(status="abstain", error_code="provider_abstention", **common)
    if answer.get("type") != "choice":
        raise ValueError("Expected choice answer")
    if "probabilities" not in answer:
        raise ValueError("Decision endpoint omitted its distribution")
    return Prediction(
        status="ok",
        decision=answer["choice"],
        probabilities=answer["probabilities"],
        probability_origin="native",
        **common,
    )


def parse_chat(body: dict, curator: bool = False) -> Prediction:
    """Parse an OpenAI-compatible chat-completions response into a Prediction.

    Recognizes ``content_filter`` finish reasons and message-level refusal fields
    as ABSTAIN rather than errors, then delegates the JSON payload to ``parse_classification``.
    """
    common = usage_fields(body)
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Expected chat choices")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Expected chat message")
    if choice.get("finish_reason") == "content_filter" or message.get("refusal"):
        return Prediction(status="abstain", error_code="provider_refusal", **common)
    if choice.get("finish_reason") != "stop":
        raise ValueError("Incomplete or unsupported chat finish reason")
    return parse_classification(message["content"], curator, **common)


def parse_classification(content: str, curator: bool = False, **common) -> Prediction:
    value = json.loads(content)
    expected = {"decision", "reason"} if curator else {"decision"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Unexpected structured answer fields")
    return Prediction(status="ok", decision=value["decision"], reason=value.get("reason"), **common)


class HTTPModel:
    """Shared HTTP inference adapter used by the ``decisions``, ``chat``, and ``argo`` adapters.

    Validates credentials and pricing configuration at construction time (before any spend).
    Each ``predict`` call atomically reserves budget, makes at most ``retries + 1`` attempts
    for transient errors, settles the actual charge, and returns a fully-populated Prediction.
    """

    def __init__(
        self,
        config: Config,
        model: ModelConfig,
        ledger: Ledger,
        run_key: str,
        *,
        curator: bool = False,
        client=None,
        sleep=time.sleep,
    ):
        self.config, self.model = config, model
        self.ledger, self.run_key = ledger, run_key
        self.curator, self.sleep = curator, sleep
        identity_env = model.argo_user_env if model.adapter == "argo" else model.api_key_env
        self.key = os.getenv(identity_env, "").strip()
        if not self.key:
            raise ValueError(f"Missing credential environment variable: {identity_env}")
        if model.budget_mode == "usd" and (
            model.input_usd_per_million is None
            or model.output_usd_per_million is None
            or model.reservation_usd is None
            or not model.pricing_checked
        ):
            raise ValueError(
                "Live runs require checked pricing and a per-attempt reservation ceiling"
            )
        self.client = client or httpx.Client(timeout=config.runner.timeout_seconds)
        self.owns_client = client is None

    def close(self):
        if self.owns_client:
            self.client.close()

    def payload(self, request: Request, policy: dict) -> dict:
        """Build the request body for the configured adapter type.

        For ``decisions``, emits an OpenRouter System One payload.
        For ``argo``, strips ``response_format`` (unsupported) and injects the Argo
        wire model name and user field. For ``chat``, emits a standard chat-completions body.
        """
        if self.model.adapter == "decisions":
            if self.curator:
                raise ValueError("Curation needs a chat model capable of enumerated reason codes")
            return {
                "model": self.model.model,
                "state": request.model_dump(),
                "questions": {
                    "triage": {
                        "type": "choice",
                        "instructions": policy["instructions"],
                        "criteria": policy["criteria"],
                    }
                },
            }
        schema = classification_schema(policy, self.curator)
        payload = {
            "model": self.model.model,
            "messages": classification_messages(request, policy, schema),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "triage", "strict": True, "schema": schema},
            },
            "max_tokens": self.model.max_output_tokens,
        }
        if self.model.adapter == "argo":
            payload["model"] = argo_model(self.model.model)
            payload["user"] = self.key
            payload.pop("response_format")
        if self.model.temperature is not None:
            payload["temperature"] = self.model.temperature
        if self.model.seed is not None:
            payload["seed"] = self.model.seed
        return payload

    def _delay(self, response, attempt: int) -> float:
        value = response.headers.get("Retry-After") if response is not None else None
        if value:
            try:
                return max(0, float(value))
            except ValueError:
                try:
                    return max(
                        0,
                        (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds(),
                    )
                except (ValueError, TypeError):
                    pass
        return 2**attempt

    def predict(self, request: Request, policy: dict) -> Prediction:
        payload = self.payload(request, policy)
        size = len(canonical(payload).encode())
        if size > self.model.max_input_bytes:
            return Prediction(status="error", error_code="input_too_large")
        # UTF-8 bytes bound visible tokenizer input; the explicit ceiling additionally reserves
        # hidden/provider overhead. If actual charges exceed it, stop rather than hide the overrun.
        metered = self.model.budget_mode == "usd"
        reserve = 0.0
        if metered:
            visible_bound = (
                size * self.model.input_usd_per_million / 1e6
                + self.model.max_output_tokens * self.model.output_usd_per_million / 1e6
                + self.model.per_request_usd
            )
            reserve = max(visible_bound, self.model.reservation_usd)
        started = time.perf_counter()
        overhead = 0.0
        attempted = 0
        estimate = 0.0
        provider_cost = 0.0
        has_provider_cost = False
        last = Prediction(status="error", error_code="not_attempted")
        for attempt in range(self.config.runner.retries + 1):
            try:
                reservation = self.ledger.reserve(
                    self.run_key, reserve, budget_mode=self.model.budget_mode
                )
            except BudgetExceeded:
                last = Prediction(status="error", error_code="budget_exhausted")
                break
            attempted += 1
            response = None
            attempt_started = time.perf_counter()
            retryable = False
            common = {}
            try:
                response = self.client.post(
                    self.model.endpoint,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.key}"},
                )
                if response.status_code == 429 or response.status_code >= 500:
                    retryable = True
                    last = Prediction(status="error", error_code=f"http_{response.status_code}")
                elif response.status_code >= 300:
                    last = Prediction(status="error", error_code=f"http_{response.status_code}")
                else:
                    try:
                        body = response.json()
                        if not isinstance(body, dict):
                            raise ValueError("Not an object")
                        common = usage_fields(body)
                        last = (
                            parse_decisions(body)
                            if self.model.adapter == "decisions"
                            else parse_chat(body, self.curator)
                        )
                    except (ValueError, TypeError, KeyError, IndexError):
                        last = Prediction(status="error", error_code="schema_error", **common)
            except httpx.TransportError:
                retryable = True
                last = Prediction(status="error", error_code="network_error")
            # Unrecognized charges stay reserved, including unsuccessful responses.
            cost = common.get("provider_cost_usd")
            if cost is not None:
                provider_cost += cost
                has_provider_cost = True
            elif metered and "input_tokens" in common and "output_tokens" in common:
                cost = (
                    common["input_tokens"] * self.model.input_usd_per_million / 1e6
                    + common["output_tokens"] * self.model.output_usd_per_million / 1e6
                    + self.model.per_request_usd
                )
            estimate += cost if cost is not None else reserve
            try:
                self.ledger.settle(reservation, cost, last.status)
            except BudgetExceeded:
                last = Prediction(status="error", error_code="reservation_overrun", **common)
                break
            if not retryable or attempt == self.config.runner.retries:
                break
            delay = self._delay(response, attempt)
            if not math.isfinite(delay) or delay > self.config.runner.max_retry_after_seconds:
                last = Prediction(status="error", error_code="rate_limit_deferred")
                break
            overhead += (time.perf_counter() - attempt_started) * 1000 + delay * 1000
            self.sleep(delay)
        return last.model_copy(
            update={
                "latency_ms": (time.perf_counter() - started) * 1000,
                "retry_overhead_ms": overhead,
                "attempts": attempted,
                "estimated_cost_usd": estimate if metered else None,
                "provider_cost_usd": provider_cost if has_provider_cost else None,
            }
        )


class ReplayModel:
    """Offline adapter that returns pre-recorded predictions from a JSON fixture file.

    Keyed by the SHA-256 of the request's canonical JSON form. Used in CI and the
    offline example config so no API credentials or network access are needed.
    """

    def __init__(self, config: Config, model: ModelConfig):
        self.model = model
        self.records = read_json(config.path(model.replay_file))

    def predict(self, request: Request, policy: dict) -> Prediction:
        from .storage import digest

        row = self.records.get(digest(request.model_dump()))
        if row is None:
            return Prediction(status="error", error_code="missing_replay")
        p = Prediction.model_validate(row)
        return p.model_copy(update={"returned_model": self.model.model})

    def close(self):
        pass


def build_model(
    config,
    model,
    ledger=None,
    run_key="",
    *,
    curator=False,
    client=None,
    sleep=time.sleep,
):
    """Instantiate the correct adapter for a given ModelConfig.

    ``replay`` returns a ReplayModel (no ledger required).
    ``codex`` lazy-loads the optional Codex SDK from ``codex.py``.
    All other adapters return an HTTPModel. Raises ``ValueError`` if a live
    adapter is requested without a ledger.
    """
    if model.adapter == "replay":
        return ReplayModel(config, model)
    if ledger is None:
        raise ValueError("Live models require the shared budget ledger")
    if model.adapter == "codex":
        from .codex import CodexModel

        return CodexModel(config, model, ledger, run_key, curator=curator, sleep=sleep)
    return HTTPModel(config, model, ledger, run_key, curator=curator, client=client, sleep=sleep)
