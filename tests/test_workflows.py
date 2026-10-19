import json

import pytest

from chemgraphsafe.data import examples, freeze, prepare
from chemgraphsafe.models import Prediction
from chemgraphsafe.reporting import metrics, report
from chemgraphsafe.runner import check_hosted_rights, evaluate
from chemgraphsafe.storage import read_json, read_rows


def test_end_to_end_real_loader_output_to_report(config):
    # Uses the fixture loader's actual output through grouping, freeze, runner and reporting.
    outcome = evaluate(config)
    assert outcome["complete"] and outcome["mock"]
    before = (config.run / "predictions.jsonl").read_bytes()
    assert evaluate(config) == outcome
    assert (config.run / "predictions.jsonl").read_bytes() == before
    result = report(config.run)
    assert result["mock"]
    m = read_json(config.run / "metrics.json")
    primary = m["models"]["replay-demo"]["splits"]["eval"]
    assert primary["n"] == 8
    assert primary["policy_agreement"] is None
    assert m["models"]["always-review"]["splits"]["eval"]["review_burden"]["value"] == 1
    assert len(read_rows(config.run / "predictions.jsonl")) == 10
    for filename in ("report.md", "comparison.csv", "metrics.json", "errors.jsonl"):
        content = (config.run / filename).read_text()
        for e in examples(config):
            assert e.request.text not in content


def test_resume_rejects_code_change_but_freeze_can_be_reused(config, monkeypatch):
    evaluate(config)
    monkeypatch.setattr("chemgraphsafe.runner.code_hash", lambda: "changed")
    with pytest.raises(ValueError, match="identity/code"):
        evaluate(config)
    config.run_dir = "runs/new-version"
    assert evaluate(config)["complete"]


def test_budget_and_privacy_rights_fail_closed(config):
    prepare(config)
    rows = examples(config)
    rows[0].hosted_processing_allowed = False
    with pytest.raises(ValueError, match="Hosted processing"):
        check_hosted_rights(rows)


def test_failure_and_abstention_denominators():
    refs = [{"id": str(i), "source_label": "harmful" if i < 4 else "benign"} for i in range(8)]
    predictions = {
        str(i): p
        for i, p in enumerate(
            [
                Prediction(status="ok", decision="BLOCK"),
                Prediction(status="ok", decision="ALLOW"),
                Prediction(status="abstain"),
                Prediction(status="error", error_code="schema_error"),
                Prediction(status="ok", decision="REVIEW"),
                Prediction(status="ok", decision="BLOCK"),
                Prediction(status="ok", decision="ALLOW"),
                Prediction(status="abstain"),
            ]
        )
    }
    m = metrics(refs, predictions)
    assert m["interception_lower"]["value"] == 0.25
    assert m["interception_upper"]["value"] == 0.5
    assert m["unsafe_allow"]["value"] == 0.25
    assert m["errors"]["harmful"] == 1
    assert m["benign_review"]["value"] == 0.25
    assert m["benign_block"]["value"] == 0.25
    assert m["decided_only"]["n"] == 5
    assert not m["operationally_complete"]


def test_all_review_baseline_not_misleading():
    refs = [{"id": str(i), "source_label": label} for i, label in enumerate(["harmful", "benign"])]
    p = {r["id"]: Prediction(status="ok", decision="REVIEW") for r in refs}
    m = metrics(refs, p)
    assert m["interception_lower"]["value"] == 1
    assert m["benign_review"]["value"] == 1
    assert m["review_burden"]["value"] == 1


def test_report_rejects_foreign_rows(config):
    evaluate(config)
    path = config.run / "predictions.jsonl"
    rows = read_rows(path)
    rows[0]["run_key"] = "different"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="different run"):
        report(config.run)


def test_prepare_no_tamper_before_freeze(config):
    prepare(config)
    with (config.dataset / "examples.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="changed"):
        freeze(config)
