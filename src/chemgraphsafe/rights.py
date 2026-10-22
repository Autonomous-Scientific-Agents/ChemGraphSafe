"""Recorded, destination-scoped authorization for hosted dataset processing."""

from datetime import date
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, StringConstraints

from .models import Example, Record
from .storage import read_json

if TYPE_CHECKING:
    from .config import Config, ModelConfig

Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class HostedDestination(Record):
    adapter: Literal["decisions", "chat", "argo", "codex", "laya"]
    model: Nonempty
    endpoint: str


class HostedApproval(Record):
    source: Literal["SciSafetyBench"]
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    approved_by: Nonempty
    approved_on: date
    evidence: Nonempty
    destinations: list[HostedDestination] = Field(min_length=1)


def check_rights_evidence(rows: list[Example]) -> None:
    """Require a nonempty processing-rights record for every evaluated example."""
    if any(not e.rights_evidence.strip() for e in rows):
        raise ValueError("Rights evidence is required for every example")


def check_hosted_rights(
    rows: list[Example],
    *,
    config: "Config | None" = None,
    models: "list[ModelConfig] | None" = None,
) -> dict | None:
    """Validate once before providers/ledger setup; return an auditable approval snapshot.

    SciSafetyBench's legacy per-example flag is not sufficient authorization. An explicit
    record covers its matching source revision and destinations, without editing a freeze.
    Other sources retain their per-example checks and cannot use this approval as an override.
    """
    challenge = [e for e in rows if e.source == "SciSafetyBench"]
    if any(
        not e.hosted_processing_allowed or not e.rights_evidence.strip()
        for e in rows
        if e.source != "SciSafetyBench"
    ):
        raise ValueError("Hosted processing is not authorized in every example's rights record")
    if not challenge:
        return None
    if config is None or not config.hosted_processing_approval_file or not models:
        raise ValueError("SciSafetyBench hosted processing requires a recorded approval file")
    try:
        approval = HostedApproval.model_validate(
            read_json(config.path(config.hosted_processing_approval_file))
        )
    except (OSError, ValueError):
        # Do not echo untrusted record contents or validation input values.
        raise ValueError("Hosted processing approval file is unreadable or invalid") from None
    if any(e.source_revision != approval.source_revision for e in challenge):
        raise ValueError("Hosted processing approval does not match the source revision")
    destinations = {(d.adapter, d.model, d.endpoint) for d in approval.destinations}
    if any((m.adapter, m.model, m.endpoint) not in destinations for m in models):
        raise ValueError("Hosted processing approval does not cover every model destination")
    return approval.model_dump(mode="json")
