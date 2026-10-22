"""Hosted authorization tests use owned fixtures and intercepted HTTP only."""

import json

import httpx
import pytest

from chemgraphsafe import adapters, data, rights, runner
from chemgraphsafe.config import ModelConfig
from chemgraphsafe.scisafetybench import REVISION
from chemgraphsafe.storage import digest, read_json, read_rows, write_json, write_rows


def source_fixture(config, allowed):
    path = config.path(config.data.fixture)
    rows = read_rows(path)
    for row in rows:
        row.update(
            source="SciSafetyBench", source_revision=REVISION, hosted_processing_allowed=allowed
        )
    write_rows(path, rows)


@pytest.fixture
def hosted_config(config):
    source_fixture(config, True)  # Simulates the old loader, using only owned prompt text.
    config.offline = False
    config.models = [
        ModelConfig(
            name="baseline",
            adapter="chat",
            model="test-baseline",
            endpoint="https://example.test/classify",
            api_key_env="TEST_KEY",
            budget_mode="attempts_only",
        )
    ]
    return config


@pytest.fixture
def mock_hosted(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "test-key")
    original = httpx.Client
    calls = []

    def respond(request):
        wire = json.loads(request.content)
        calls.append(wire)
        content = {"decision": "ALLOW"}
        if request.url.path == "/curate":
            content["reason"] = "legitimate_use"
        return httpx.Response(
            200,
            json={
                "model": wire["model"],
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}],
            },
        )

    monkeypatch.setattr(
        adapters.httpx,
        "Client",
        lambda **kw: original(**kw, transport=httpx.MockTransport(respond)),
    )
    return calls


def approval_record(config, models=None):
    return {
        "source": "SciSafetyBench",
        "source_revision": REVISION,
        "approved_by": "Test-only approver",
        "approved_on": "2026-10-22",
        "evidence": "Owned synthetic test fixtures only; not a real dataset approval",
        "destinations": [
            {"adapter": m.adapter, "model": m.model, "endpoint": m.endpoint}
            for m in (models or config.models)
        ],
    }


def save_approval(config, record=None):
    config.hosted_processing_approval_file = ".local/approvals/test.json"
    path = config.path(config.hosted_processing_approval_file)
    write_json(path, record or approval_record(config))
    return path


def block_side_effects(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Unauthorized processing reached provider or budget setup")

    monkeypatch.setattr(runner, "build_model", unexpected)
    monkeypatch.setattr(runner, "ledger_for", unexpected)


def snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.parametrize("action", ["evaluate", "curate"])
def test_missing_approval_blocks_fresh_and_legacy_data_before_side_effects(
    hosted_config, monkeypatch, allowed, action
):
    config = hosted_config
    source_fixture(config, allowed)
    if action == "curate":
        config.curator = config.models[0].model_copy(update={"model": "test-curator"})
    data.prepare(config)
    if action == "evaluate":
        data.freeze(config)
    before = snapshot(config.dataset)
    block_side_effects(monkeypatch)
    with pytest.raises(ValueError, match="recorded approval"):
        getattr(runner, action)(config)
    assert snapshot(config.dataset) == before
    assert not config.run.exists() and not config.path(config.budget.ledger).exists()


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_file",
        "json",
        "source",
        "revision",
        "approver",
        "date",
        "evidence",
        "destinations",
        "endpoint",
        "model",
        "adapter",
        "missing_endpoint",
        "extra",
    ],
)
def test_invalid_or_mismatched_approval_fails_before_side_effects(
    hosted_config, monkeypatch, invalid
):
    config = hosted_config
    data.prepare(config)
    data.freeze(config)
    record = approval_record(config)
    if invalid == "source":
        record["source"] = "unrelated-source"
    elif invalid == "revision":
        record["source_revision"] = "0" * 40
    elif invalid == "approver":
        record["approved_by"] = "  "
    elif invalid == "date":
        record["approved_on"] = "not-a-date"
    elif invalid == "evidence":
        record["evidence"] = "\t"
    elif invalid == "destinations":
        record["destinations"] = []
    elif invalid in {"endpoint", "model", "adapter"}:
        record["destinations"][0][invalid] = {
            "endpoint": "https://different.test/classify",
            "model": "another-model",
            "adapter": "decisions",
        }[invalid]
    elif invalid == "missing_endpoint":
        record["destinations"][0].pop("endpoint")
    elif invalid == "extra":
        record["unexpected"] = "PRIVATE INVALID APPROVAL INPUT"
    path = save_approval(config, record)
    if invalid == "missing_file":
        config.hosted_processing_approval_file = "missing.json"
    elif invalid == "json":
        path.write_text("PRIVATE INVALID APPROVAL INPUT")
    block_side_effects(monkeypatch)
    with pytest.raises(ValueError, match="approval") as error:
        runner.evaluate(config)
    assert "PRIVATE" not in str(error.value)
    assert not config.run.exists() and not config.path(config.budget.ledger).exists()


@pytest.mark.parametrize("allowed", [False, True])
def test_approval_is_bound_to_run_not_freeze_and_is_read_once(
    hosted_config, mock_hosted, monkeypatch, allowed
):
    config = hosted_config
    source_fixture(config, allowed)
    data.prepare(config)
    frozen = data.freeze(config)
    dataset_before = snapshot(config.dataset)
    semantic = config.semantic_hash()
    path = save_approval(config)
    assert config.semantic_hash() == semantic and data.verify_freeze(config) == frozen
    original_read = rights.read_json
    reads = []

    def record_read(path):
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(rights, "read_json", record_read)
    assert runner.evaluate(config)["complete"]
    assert reads == [path] and len(mock_hosted) == 10
    execution = read_json(config.run / "manifest.json")["execution"]
    assert execution["hosted_processing_approval"] == approval_record(config)
    assert execution["hosted_processing_approval_hash"] == digest(approval_record(config))
    assert "Test-only approver" not in json.dumps(mock_hosted)
    assert "Owned synthetic test fixtures" not in json.dumps(mock_hosted)
    before = snapshot(config.run)
    # Formatting and the file's location are not execution semantics.
    path.write_text(json.dumps(approval_record(config), indent=4))
    moved = config.root / "same-approval.json"
    moved.write_bytes(path.read_bytes())
    config.hosted_processing_approval_file = str(moved)
    assert runner.evaluate(config)["complete"]
    assert len(mock_hosted) == 10 and snapshot(config.run) == before
    record = approval_record(config)
    record["approved_by"] = "Different test-only approver"
    write_json(moved, record)
    with pytest.raises(ValueError, match="new run directory"):
        runner.evaluate(config)
    assert snapshot(config.run) == before and len(mock_hosted) == 10
    config.run_dir = "runs/new-approval"
    assert runner.evaluate(config)["complete"] and len(mock_hosted) == 20
    assert read_json(config.run / "manifest.json")["execution"][
        "hosted_processing_approval_hash"
    ] == digest(record)
    assert snapshot(config.dataset) == dataset_before and data.verify_freeze(config) == frozen
    assert len(reads) == 4


def test_approval_must_cover_every_destination_and_cannot_override_other_sources(
    hosted_config, monkeypatch
):
    config = hosted_config
    save_approval(config)
    config.models.append(config.models[0].model_copy(update={"name": "other", "model": "other"}))
    data.prepare(config)
    data.freeze(config)
    block_side_effects(monkeypatch)
    with pytest.raises(ValueError, match="every model destination"):
        runner.evaluate(config)
    rows = data.examples(config)
    rows[0].source = "SciHazard"
    rows[0].hosted_processing_allowed = False
    with pytest.raises(ValueError, match="every example"):
        rights.check_hosted_rights(rows, config=config, models=config.models[:1])


def test_curation_checks_its_own_destination(hosted_config, mock_hosted):
    config = hosted_config
    config.curator = config.models[0].model_copy(
        update={
            "name": "curator",
            "model": "test-curator",
            "endpoint": "https://example.test/curate",
        }
    )
    data.prepare(config)
    save_approval(config)  # Evaluation approval does not cover the different curator.
    with pytest.raises(ValueError, match="every model destination"):
        runner.curate(config)
    assert not mock_hosted and not config.path(config.budget.ledger).exists()
    save_approval(config, approval_record(config, [config.curator]))
    assert runner.curate(config)["annotations"] == 10 and len(mock_hosted) == 10


def test_owned_smoke_does_not_read_dataset_approval(hosted_config, mock_hosted):
    config = hosted_config
    config.hosted_processing_approval_file = "missing.json"
    assert runner.smoke(config)["complete"] and len(mock_hosted) == 5
    execution = read_json(config.run / "smoke/manifest.json")["execution"]
    assert "hosted_processing_approval" not in execution


def test_offline_processing_and_reports_do_not_read_approval(config):
    from chemgraphsafe.reporting import report

    source_fixture(config, False)
    config.hosted_processing_approval_file = "missing.json"
    assert runner.evaluate(config)["complete"]
    assert report(config.run)["mock"]
    assert not config.path(config.budget.ledger).exists()
