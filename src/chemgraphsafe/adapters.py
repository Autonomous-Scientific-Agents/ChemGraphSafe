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
from .models import Prediction, Request, policy_labels, question_key
from .prompts import classification_messages, classification_schema
from .providers import argo_model
from .storage import canonical, read_json


def usage_fields(body: dict) -> dict:
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


def parse_decisions(body: dict, *, policy: dict | None = None) -> Prediction:
    common = usage_fields(body)
    answers = body.get("answers")
    key = question_key(policy)
    answer = answers.get(key) if isinstance(answers, dict) else None
    if not isinstance(answer, dict):
        raise ValueError(f"Expected a {key} answer object")
    if answer.get("abstention") is True:
        return Prediction(status="abstain", error_code="provider_abstention", **common)
    if answer.get("type") != "choice":
        raise ValueError("Expected choice answer")
    if not isinstance(answer.get("probabilities"), dict):
        raise ValueError("Decision endpoint omitted its distribution")
    return Prediction(
        status="ok",
        decision=answer["choice"],
        probabilities=answer["probabilities"],
        probability_origin="native",
        **common,
    ).validate_policy(policy)


def _laya_gate_state(answer: dict) -> str | None:
    state = answer.get("abstention")
    gate_fields = {"abstention", "abstention_threshold", "low_confidence"}
    present = gate_fields & set(answer)
    if "abstention" not in present:
        if present:
            raise ValueError("Incomplete Laya abstention metadata")
        return None
    if state not in {"passed", "abstained", "unevaluated"}:
        raise ValueError("Unknown Laya abstention state")
    threshold = answer.get("abstention_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid Laya abstention threshold")
    low_confidence = answer.get("low_confidence")
    if state == "abstained":
        if low_confidence is not True:
            raise ValueError("Abstained Laya answer must be marked low confidence")
    elif state == "passed" and "low_confidence" in answer and low_confidence is not False:
        raise ValueError("Only abstained Laya answers may be marked low confidence")
    elif state == "unevaluated" and "low_confidence" in answer:
        raise ValueError("Unevaluated Laya answers cannot carry a confidence flag")
    return state


def _laya_input_truncated(body: dict, *, key: str) -> bool:
    """Validate Laya 0.4.13's per-request truncation report."""
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("Laya response omitted its usage metadata")
    state_tokens = usage.get("state_tokens")
    dropped = usage.get("state_tokens_dropped")
    truncated = usage.get("truncated")
    questions = usage.get("truncated_questions")
    if (
        isinstance(state_tokens, bool)
        or not isinstance(state_tokens, int)
        or state_tokens < 0
        or isinstance(dropped, bool)
        or not isinstance(dropped, int)
        or dropped < 0
        or dropped > state_tokens
        or not isinstance(truncated, bool)
        or not isinstance(questions, list)
        or any(not isinstance(question, str) or not question for question in questions)
        or len(questions) != len(set(questions))
        or any(question != key for question in questions)
        or truncated != (dropped > 0)
        or truncated != bool(questions)
    ):
        raise ValueError("Invalid or inconsistent Laya truncation metadata")
    return truncated


def parse_laya(
    body: dict,
    *,
    model: str,
    revision: str,
    policy: dict | None = None,
) -> Prediction:
    common = usage_fields(body)
    common.pop("provider_cost_usd", None)
    common["returned_model"] = f"{model}@{revision}"
    key = question_key(policy)
    truncated = _laya_input_truncated(body, key=key)
    routing = body.get("routing")
    if not isinstance(routing, dict) or routing.get("model") != model:
        raise ValueError("Laya response did not use the configured checkpoint")
    answers = body.get("answers")
    answer = answers.get(key) if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError(f"Expected a Laya {key} choice answer")
    probabilities = answer.get("probabilities")
    if "choice" not in answer or not isinstance(probabilities, dict):
        raise ValueError("Laya decision omitted its choice or distribution")
    # Validate the retained answer even when the confidence gate says not to use it.
    decision = Prediction(
        status="ok",
        decision=answer["choice"],
        probabilities=probabilities,
        probability_origin="native",
        **common,
    ).validate_policy(policy)
    gate = _laya_gate_state(answer)
    if truncated:
        return Prediction(status="error", error_code="input_truncated", **common)
    if gate == "abstained":
        return Prediction(status="abstain", error_code="provider_abstention", **common)
    if gate == "unevaluated":
        return Prediction(status="error", error_code="abstention_unevaluated", **common)
    return decision


def parse_chat(body: dict, curator: bool = False, *, policy: dict | None = None) -> Prediction:
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
    return parse_classification(message["content"], curator, policy=policy, **common)


def parse_classification(
    content: str, curator: bool = False, *, policy: dict | None = None, **common
) -> Prediction:
    value = json.loads(content)
    expected = {"decision", "reason"} if curator else {"decision"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Unexpected structured answer fields")
    if curator and value["reason"] is None:
        raise ValueError("Curation requires a reason code")
    return Prediction(
        status="ok", decision=value["decision"], reason=value.get("reason"), **common
    ).validate_policy(policy)


class HTTPModel:
    processing_location = "hosted"

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
        self.runtime_provenance = None
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

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"}

    def response_fields(self, body: dict) -> dict:
        return usage_fields(body)

    def parse_response(self, body: dict, policy: dict | None = None) -> Prediction:
        return (
            parse_decisions(body, policy=policy)
            if self.model.adapter == "decisions"
            else parse_chat(body, self.curator, policy=policy)
        )

    def payload(self, request: Request, policy: dict) -> dict:
        if self.model.adapter == "decisions":
            if self.curator:
                raise ValueError("Curation needs a chat model capable of enumerated reason codes")
            return {
                "model": self.model.model,
                "state": request.model_dump(),
                "questions": {
                    question_key(policy): {
                        "type": "choice",
                        "instructions": policy["instructions"],
                        "criteria": {
                            label: policy["criteria"][label] for label in policy_labels(policy)
                        }
                        if question_key(policy) == "risk_type"
                        else policy["criteria"],
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
                        0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
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
                    headers=self.headers(),
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
                        common = self.response_fields(body)
                        last = self.parse_response(body, policy)
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


class LayaModel(HTTPModel):
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
        if curator:
            raise ValueError("Laya cannot supply policy reason codes for curation")
        self.config, self.model = config, model
        self.ledger, self.run_key = ledger, run_key
        self.curator, self.sleep = curator, sleep
        self.processing_location = model.processing_location
        self.key = ""
        if model.api_key_env:
            self.key = os.getenv(model.api_key_env, "").strip()
            if not self.key:
                raise ValueError(f"Missing credential environment variable: {model.api_key_env}")
        self.client = client or httpx.Client(
            timeout=config.runner.timeout_seconds,
            trust_env=self.processing_location != "local",
        )
        self.owns_client = client is None
        try:
            self.runtime_provenance = self._preflight()
        except Exception:
            if self.owns_client:
                self.client.close()
            raise

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}

    def _preflight(self) -> dict:
        try:
            response = self.client.get(self.model.identity_endpoint, headers=self.headers())
        except httpx.TransportError:
            raise ValueError("Laya identity preflight failed") from None
        if response.status_code != 200:
            raise ValueError(f"Laya identity preflight returned HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            raise ValueError("Laya identity preflight returned invalid JSON") from None
        if not isinstance(body, dict) or body.get("status") != "ok":
            raise ValueError("Laya identity preflight is not healthy")
        loaded = body.get("loaded")
        revisions = body.get("revisions")
        if (
            not isinstance(loaded, list)
            or any(not isinstance(value, str) for value in loaded)
            or self.model.model not in loaded
            or not isinstance(revisions, dict)
            or revisions.get(self.model.model) != self.model.expected_revision
        ):
            raise ValueError("Laya identity preflight model/revision mismatch")
        checkpoint_devices = body.get("checkpoint_devices")
        device = (
            checkpoint_devices.get(self.model.model)
            if isinstance(checkpoint_devices, dict)
            else body.get("device")
        )
        if not isinstance(device, str) or not device:
            raise ValueError("Laya identity preflight omitted the checkpoint device")
        return {
            "model": self.model.model,
            "revision": self.model.expected_revision,
            "device": device,
        }

    def payload(self, request: Request, policy: dict) -> dict:
        return {
            "model": self.model.model,
            "state": request.model_dump(),
            "questions": {
                question_key(policy): {
                    "type": "choice",
                    "instructions": policy["instructions"],
                    "criteria": {
                        label: policy["criteria"][label] for label in policy_labels(policy)
                    },
                }
            },
        }

    def response_fields(self, body: dict) -> dict:
        fields = usage_fields(body)
        fields.pop("provider_cost_usd", None)
        fields["returned_model"] = (
            f"{self.runtime_provenance['model']}@{self.runtime_provenance['revision']}"
        )
        return fields

    def parse_response(self, body: dict, policy: dict | None = None) -> Prediction:
        return parse_laya(
            body,
            model=self.runtime_provenance["model"],
            revision=self.runtime_provenance["revision"],
            policy=policy,
        )


class ReplayModel:
    processing_location = "local"
    runtime_provenance = None

    def __init__(self, config: Config, model: ModelConfig):
        self.model = model
        self.records = read_json(config.path(model.replay_file))

    def predict(self, request: Request, policy: dict) -> Prediction:
        from .storage import digest

        row = self.records.get(digest(request.model_dump()))
        if row is None:
            return Prediction(status="error", error_code="missing_replay")
        try:
            p = Prediction.model_validate(row).validate_policy(policy)
        except (ValueError, TypeError):
            p = Prediction(status="error", error_code="schema_error")
        return p.model_copy(update={"returned_model": self.model.model})

    def close(self):
        pass


def build_model(
    config, model, ledger=None, run_key="", *, curator=False, client=None, sleep=time.sleep
):
    if model.adapter == "replay":
        return ReplayModel(config, model)
    if ledger is None:
        raise ValueError("Live models require the shared budget ledger")
    if model.adapter == "codex":
        from .codex import CodexModel

        return CodexModel(config, model, ledger, run_key, curator=curator, sleep=sleep)
    if model.adapter == "laya":
        return LayaModel(
            config,
            model,
            ledger,
            run_key,
            curator=curator,
            client=client,
            sleep=sleep,
        )
    return HTTPModel(config, model, ledger, run_key, curator=curator, client=client, sleep=sleep)
