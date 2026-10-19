import hashlib
import json

import httpx
import pytest

from chemgraphsafe import data
from chemgraphsafe.adapters import HTTPModel, parse_chat, parse_decisions
from chemgraphsafe.budget import Ledger
from chemgraphsafe.config import Config, ModelConfig
from chemgraphsafe.models import Prediction, Request
from chemgraphsafe.reporting import metrics, report
from chemgraphsafe.runner import curate, evaluate, smoke
from chemgraphsafe.storage import digest, read_json, read_rows, write_json


def test_source_loader_discards_answers_and_quarantines_native_conflicts(config, monkeypatch):
    raw = config.dataset / "raw"
    raw.mkdir(parents=True)
    files = {}
    for label, native in (("benign", None), ("harmful", False)):
        filename = label + ".jsonl"
        row = {
            "idx": 1,
            "subject": "Chemistry",
            "question": "Define an atom.",
            "is_harmful": native,
            "_response": "PRIVATE RESPONSE",
            "classification_reasoning": "PRIVATE RATIONALE",
            "category": "metadata",
        }
        content = (json.dumps(row) + "\n").encode()
        (raw / filename).write_bytes(content)
        files[label] = (filename, hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(data, "SOURCE_FILES", files)
    rows = data.SciHazardLoader().load(config)
    assert len(rows) == 2
    assert rows[0].request.model_dump() == {"text": "Define an atom.", "context": ""}
    assert "PRIVATE" not in json.dumps([e.model_dump() for e in rows])
    pool, stats = data.deduplicate_and_group(rows, 0.9)
    assert len(pool) == 1 and pool[0].source_label == "benign"
    assert len(stats["native_conflict_ids"]) == 1


@pytest.mark.parametrize("body", [{}, {"answers": None}, {"answers": {"triage": None}}])
def test_decision_missing_or_null_fields_fail_validation(body):
    with pytest.raises(ValueError):
        parse_decisions(body)


@pytest.mark.parametrize(
    "body", [{"choices": None}, {"choices": [None]}, {"choices": [{"message": None}]}]
)
def test_chat_missing_or_null_fields_fail_validation(body):
    with pytest.raises(ValueError):
        parse_chat(body)


def test_curation_output_through_freeze_and_metrics(config):
    data.prepare(config)
    records = {}
    for i, e in enumerate(data.examples(config)):
        label = ("ALLOW", "REVIEW", "BLOCK")[i % 3]
        records[digest(e.request.model_dump())] = {
            "status": "ok",
            "decision": label,
            "reason": "insufficient_context",
        }
    write_json(config.root / "curator.json", records)
    config.curator = ModelConfig(
        name="curator", adapter="replay", model="owned-curator", replay_file="curator.json"
    )
    result = curate(config)
    assert result["annotations"] == 10
    before = (config.dataset / "annotations.jsonl").read_bytes()
    assert curate(config) == result
    assert (config.dataset / "annotations.jsonl").read_bytes() == before
    assert data.freeze(config)["annotation_count"] == 10
    assert evaluate(config)["complete"]
    report(config.run)
    stats = read_json(config.run / "metrics.json")["models"]["replay-demo"]["splits"]["eval"]
    assert stats["annotation_coverage"]["value"] == 1
    assert stats["policy_agreement"] is not None


def test_three_class_confusion_counts_abstention_and_errors_as_misses():
    refs = [
        {
            "id": str(i),
            "source_label": "benign",
            "annotation": {"prediction": {"status": "ok", "decision": label}},
        }
        for i, label in enumerate(("ALLOW", "REVIEW", "BLOCK"))
    ]
    p = {
        "0": Prediction(status="ok", decision="ALLOW"),
        "1": Prediction(status="abstain"),
        "2": Prediction(status="error"),
    }
    score = metrics(refs, p)["policy_agreement"]
    assert score["macro_f1"] == pytest.approx(1 / 3)
    assert score["confusion"]["REVIEW"]["ABSTAIN"] == 1
    assert score["confusion"]["BLOCK"]["ERROR"] == 1


def test_reference_tamper_rejected(config):
    evaluate(config)
    path = config.run / "references.json"
    refs = read_json(path)
    refs[0]["source_label"] = "benign" if refs[0]["source_label"] == "harmful" else "harmful"
    write_json(path, refs)
    with pytest.raises(ValueError, match="Reference contents"):
        report(config.run)


def test_resume_retries_errors_preserves_abstentions_and_smoke_disjoint(config, monkeypatch):
    class Adapter:
        calls = []

        def predict(self, request, policy):
            self.calls.append(request.text)
            return Prediction(status="error" if len(self.calls) == 1 else "abstain")

        def close(self):
            pass

    adapter = Adapter()
    monkeypatch.setattr("chemgraphsafe.runner.build_model", lambda *a, **kw: adapter)
    assert not evaluate(config)["complete"]
    assert len(adapter.calls) == 10
    assert evaluate(config)["complete"]
    assert len(adapter.calls) == 11
    assert smoke(config)["complete"]
    pilot = {r["example_id"] for r in read_rows(config.run / "predictions.jsonl")}
    smoked = {r["example_id"] for r in read_rows(config.run / "smoke/predictions.jsonl")}
    assert len(smoked) == 5 and not pilot & smoked


def test_offline_cannot_enable_live_curator(config):
    values = config.model_dump()
    values["curator"] = {
        "name": "curator",
        "adapter": "chat",
        "model": "curator",
        "endpoint": "https://example.test/chat",
        "api_key_env": "TEST_KEY",
    }
    with pytest.raises(ValueError, match="live curator"):
        Config.model_validate(values)


def test_timeout_retains_reservations_and_retries(config, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "test")
    m = ModelConfig(
        name="jev",
        adapter="decisions",
        model="test",
        endpoint="https://example.test/decisions",
        api_key_env="TEST_KEY",
        input_usd_per_million=1,
        output_usd_per_million=0,
        pricing_checked="2026-10-19",
        reservation_usd=0.01,
    )
    ledger = Ledger(config.root / "ledger.sqlite", 1, 10)

    def timeout(request):
        raise httpx.ReadTimeout("timeout", request=request)

    with httpx.Client(transport=httpx.MockTransport(timeout)) as client:
        adapter = HTTPModel(config, m, ledger, "run", client=client, sleep=lambda _: None)
        result = adapter.predict(Request(text="Define an atom."), config.policy)
    assert result.status == "error" and result.attempts == 3
    assert ledger.summary()["unreconciled_attempts"] == 3
    assert ledger.summary()["accounted_usd"] == pytest.approx(0.03)
