"""Pinned loopback Kev decisions using the shared HTTP transport and budget ledger."""

import math

import httpx

from .adapters import HTTPModel, parse_decisions, usage_fields
from .models import Prediction, question_key


class KevUnavailable(ValueError):
    """Metadata could not establish the loaded identity; no inference is allowed."""


class KevIdentityChanged(ValueError):
    """Valid metadata reports a different checkpoint or serving setting."""


class KevModel(HTTPModel):
    credential_required = False
    processing_location = "local"
    retryable_preflight_errors = frozenset({"local_unavailable"})

    def __init__(self, config, model, ledger, run_key, **kwargs):
        if kwargs.get("curator"):
            raise ValueError("Kev cannot provide curation reason codes")
        self.settings = model.local
        super().__init__(config, model, ledger, run_key, **kwargs)
        try:
            self.observed = self.server_metadata()
        except Exception:
            self.close()
            raise
        self.runtime_provenance = {
            "compute": "local",
            "declared_runtime_revision": self.settings.runtime_revision,
            "checkpoint": f"{model.model}@{self.settings.revision}",
            "observed_server": self.observed,
            "output_tokens": "unavailable; upstream counters have different semantics",
        }

    def server_metadata(self):
        endpoint = str(httpx.URL(self.model.endpoint).copy_with(path="/v1/models"))
        try:
            response = self.client.get(endpoint, headers=self.headers(), follow_redirects=False)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise KevUnavailable("Kev server identity preflight unavailable") from exc
        try:
            return self.kev_metadata(body)
        except KevUnavailable:
            raise
        except ValueError as exc:
            raise KevIdentityChanged("Kev server identity preflight failed") from exc

    def kev_metadata(self, body):
        cards = body.get("models") if isinstance(body, dict) else None
        if not isinstance(cards, list) or not cards or not isinstance(cards[0], dict):
            raise KevUnavailable("Kev identity preflight metadata is incomplete")
        card = cards[0]
        expected = {
            "run": f"{self.model.model}@{self.settings.revision}",
            "base": self.settings.base_model,
            "device": self.settings.device,
            "dtype": self.settings.dtype,
            "backend": "torch",
            "truncate_states": False,
        }
        if any(type(card.get(k)) is not type(v) for k, v in expected.items()):
            raise KevUnavailable("Kev identity preflight metadata is malformed")
        temperature = card.get("temperature")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise KevUnavailable("Kev identity preflight temperature is malformed")
        if any(card[k] != v for k, v in expected.items()):
            raise ValueError("Kev checkpoint/device/settings mismatch")
        if not math.isclose(temperature, self.settings.calibration_temperature, rel_tol=1e-9):
            raise ValueError("Kev calibration temperature mismatch")
        return {**expected, "temperature": temperature}

    def preflight_attempt(self):
        try:
            observed = self.server_metadata()
        except KevUnavailable:
            return Prediction(status="error", error_code="local_unavailable")
        except KevIdentityChanged:
            return Prediction(status="error", error_code="local_identity_changed")
        if observed != self.observed:
            return Prediction(status="error", error_code="local_identity_changed")

    def headers(self):
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}

    def response_fields(self, body: dict):
        # Local compute has no provider dollar charge or comparable generated-token count.
        common = usage_fields(body)
        return {
            **({"input_tokens": common["input_tokens"]} if "input_tokens" in common else {}),
            "returned_model": f"{self.model.model}@{self.settings.revision}",
        }

    def transport_failure(self, exc):
        return Prediction(
            status="error",
            error_code="local_timeout"
            if isinstance(exc, httpx.TimeoutException)
            else "network_error",
        )

    def parse_response(self, body: dict, policy: dict | None = None) -> Prediction:
        common = self.response_fields(body)

        def error(code):
            return Prediction(status="error", error_code=code, **common)

        if body.get("truncated", False) is not False:
            return error("local_input_truncated")
        answers = body.get("answers")
        answer = answers.get(question_key(policy)) if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            raise ValueError("Missing task answer")
        if answer.get("type") != "choice":
            raise ValueError("Expected a choice answer")
        abstention = answer.get("abstention")
        if abstention is True:
            return Prediction(status="abstain", error_code="provider_abstention", **common)
        if abstention is not None and abstention is not False:
            raise ValueError("Expected a boolean abstention field")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or any(
            isinstance(value, bool) or not isinstance(value, (float, int))
            for value in probabilities.values()
        ):
            raise ValueError("Expected numeric probabilities")
        parsed = parse_decisions(
            {**body, "usage": {}, "model": common["returned_model"]}, policy=policy
        )
        return parsed.model_copy(update=common)
