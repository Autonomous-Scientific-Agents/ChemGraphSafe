"""Public extension contracts. Reference labels never enter Request."""

import math
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

Decision = Literal["ALLOW", "REVIEW", "BLOCK"]
Status = Literal["ok", "abstain", "error"]
Reason = Literal["legitimate_use", "harm_facilitation", "insufficient_context"]
LABELS = ("ALLOW", "REVIEW", "BLOCK")
PROTOCOL_VERSION = "triage-1"


class Record(BaseModel):
    """Base for all data records. Rejects extra fields and non-finite floats."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Request(Record):
    """The only thing a model ever sees — request text plus optional context string.

    Source IDs, labels, and annotations are deliberately excluded to prevent leakage.
    """

    text: str = Field(min_length=1)
    context: str = ""


class Example(Record):
    """A fully-provenance-tracked dataset entry, produced by a DatasetLoader.

    The ``id`` is a stable, source-qualified hash. ``group`` links near-duplicate
    requests so they land in the same train/eval split. Rights fields gate whether
    the example may be sent to a hosted model.
    """

    id: str
    request: Request
    source: str
    source_id: str
    source_revision: str
    source_split: str
    source_label: Literal["benign", "harmful"]
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
    """Everything a model returns for one request, including cost, timing, and error details.

    If ``probabilities`` is present, all three ALLOW/REVIEW/BLOCK classes must sum to 1 ± 1e-3.
    The provider's explicit ``decision`` is authoritative even when it differs from argmax;
    that inconsistency is recorded in ``choice_argmax_mismatch`` rather than silently fixed.
    """

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
            if self.status != "ok" or set(p) != set(LABELS):
                raise ValueError("A distribution requires a successful three-class decision")
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()):
                raise ValueError("Invalid probability")
            if abs(sum(p.values()) - 1) > 1e-3:
                raise ValueError("Probability sum must be within 1e-3 of one")
            if self.probability_origin is None:
                raise ValueError("Probability provenance is required")
            self.choice_argmax_mismatch = p[self.decision] < max(p.values())
        return self


class Annotation(Record):
    """An LLM-generated provisional policy label for one Example.

    ``policy_hash`` and ``config_hash`` tie the annotation to the exact policy JSON
    and curator configuration used. Annotations are stored separately from examples
    so the dataset can be frozen with or without them.
    """

    example_id: str
    policy_hash: str
    config_hash: str
    origin: Literal["llm_provisional", "owned_fixture"]
    requested_model: str
    prediction: Prediction
    source_disagreement: bool = False


class PredictionRow(Record):
    """One model's prediction for one example, keyed to a specific run for resume support."""

    example_id: str
    model_name: str
    split: Literal["dev", "eval", "smoke"]
    run_key: str
    prediction: Prediction


# ── Extension protocols ────────────────────────────────────────────────────────
# Register implementations in DATASET_LOADERS (data.py) and build_model (adapters.py).


class DatasetLoader(Protocol):
    """Loads and normalizes examples from one source into the shared Example schema."""

    def load(self, config) -> list[Example]: ...


class DecisionModel(Protocol):
    """Classifies a single Request as ALLOW, REVIEW, or BLOCK under the given policy."""

    def predict(self, request: Request, policy: dict) -> Prediction: ...


class Labeler(Protocol):
    """Generates a provisional Annotation for an Example using a configured LLM curator."""

    def annotate(self, example: Example, policy: dict) -> Annotation: ...
