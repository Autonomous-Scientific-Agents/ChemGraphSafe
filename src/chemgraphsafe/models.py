"""Public extension contracts. Reference labels never enter Request."""

import math
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

Decision = Literal["ALLOW", "REVIEW", "BLOCK"]
Status = Literal["ok", "abstain", "error"]
ProcessingLocation = Literal["local", "hosted"]
Reason = Literal["legitimate_use", "harm_facilitation", "insufficient_context", "operational_risk"]
LABELS = ("ALLOW", "REVIEW", "BLOCK")
BINARY_LABELS = ("ALLOW", "BLOCK")
PROTOCOL_VERSION = "triage-1"


def policy_labels(policy: dict) -> tuple[str, ...]:
    """Return a supported label set in canonical order, independent of JSON key order."""
    criteria = policy.get("criteria")
    if not isinstance(criteria, dict) or set(criteria) not in (set(LABELS), set(BINARY_LABELS)):
        raise ValueError("Policy must define ALLOW/BLOCK or ALLOW/REVIEW/BLOCK")
    return tuple(label for label in LABELS if label in criteria)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Request(Record):
    text: str = Field(min_length=1)
    context: str = ""


class Example(Record):
    id: str
    request: Request
    source: str
    source_id: str
    source_revision: str
    source_split: str
    source_label: Literal["benign", "harmful"] | None
    source_native_harmful: bool | None = None
    domain: str = "Chemistry"
    category: str = "unspecified"
    family: str | None = None
    group: str = ""
    related_source_ids: list[str] = Field(default_factory=list)
    license: str
    hosted_processing_allowed: bool = False
    rights_evidence: str


class Prediction(Record):
    status: Status
    decision: Decision | None = None
    probabilities: dict[str, float] | None = None
    probability_origin: Literal["native", "mock"] | None = None
    choice_argmax_mismatch: bool = False
    returned_model: str | None = None
    reason: Reason | None = None
    error_code: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    provider_cost_usd: float | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    retry_overhead_ms: float = Field(default=0, ge=0)
    attempts: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def consistent(self):
        if (self.status == "ok") != (self.decision is not None):
            raise ValueError("Only successful decisions may carry a policy label")
        if self.probabilities is not None:
            p = self.probabilities
            if (
                self.status != "ok"
                or set(p) not in (set(LABELS), set(BINARY_LABELS))
                or self.decision not in p
            ):
                raise ValueError("A distribution requires a matching two- or three-class decision")
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()):
                raise ValueError("Invalid probability")
            if abs(sum(p.values()) - 1) > 1e-3:
                raise ValueError("Probability sum must be within 1e-3 of one")
            if self.probability_origin is None:
                raise ValueError("Probability provenance is required")
            self.choice_argmax_mismatch = p[self.decision] < max(p.values())
        return self

    def validate_policy(self, policy: dict | None = None) -> "Prediction":
        """Validate the active experiment's label set; omitted policy retains the legacy default."""
        labels = policy_labels(policy) if policy is not None else LABELS
        if self.status == "ok":
            if self.decision not in labels:
                raise ValueError("Decision is outside the active policy labels")
            if self.probabilities is not None and set(self.probabilities) != set(labels):
                raise ValueError("Probability labels differ from the active policy labels")
            if (
                policy is not None
                and self.reason is not None
                and self.reason not in policy.get("reason_codes", [])
            ):
                raise ValueError("Reason is outside the active policy reason codes")
        return self


class Annotation(Record):
    example_id: str
    policy_hash: str
    config_hash: str
    origin: Literal["llm_provisional", "owned_fixture"]
    requested_model: str
    prediction: Prediction
    source_disagreement: bool = False


class RuleAnnotation(Record):
    """A reproducible policy reference, without an invented model/curator identity."""

    example_id: str
    policy_hash: str
    origin: Literal["rule_derived"] = "rule_derived"
    rule_id: str
    rule_hash: str
    provenance: dict[str, str]
    prediction: Prediction


def parse_annotation(row: dict) -> Annotation | RuleAnnotation:
    cls = RuleAnnotation if row.get("origin") == "rule_derived" else Annotation
    return cls.model_validate(row)


class PredictionRow(Record):
    example_id: str
    model_name: str
    split: Literal["dev", "eval", "smoke"]
    run_key: str
    prediction: Prediction


class DatasetLoader(Protocol):
    def load(self, config) -> list[Example]: ...


class DecisionModel(Protocol):
    processing_location: ProcessingLocation

    def predict(self, request: Request, policy: dict) -> Prediction: ...


class Labeler(Protocol):
    def annotate(self, example: Example, policy: dict) -> Annotation: ...
