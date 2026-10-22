import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from chemgraphsafe import data
from chemgraphsafe.adapters import HTTPModel, ReplayModel, parse_decisions
from chemgraphsafe.budget import Ledger
from chemgraphsafe.cli import app
from chemgraphsafe.codex import CodexModel
from chemgraphsafe.config import ModelConfig, load_config
from chemgraphsafe.models import BINARY_LABELS, LABELS, Prediction, Request, policy_labels
from chemgraphsafe.prompts import classification_schema
from chemgraphsafe.reporting import metrics, report
from chemgraphsafe.runner import curate, evaluate, predict_under_policy, smoke
from chemgraphsafe.storage import digest, read_json, read_rows, write_json, write_rows


@pytest.fixture
def binary_config(config):
    config.policy_file = "policies/triage-binary-v0.1.json"
    return config


@pytest.fixture
def binary_replay(binary_config):
    config = binary_config
    data.prepare(config)
    records = {
        digest(e.request.model_dump()): {
            "status": "ok",
            "decision": "ALLOW" if e.source_label == "benign" else "BLOCK",
        }
        for e in data.examples(config)
    }
    config.models[0].replay_file = "binary-replay.json"
    write_json(config.root / "binary-replay.json", records)
    return config


def test_policy_selects_exact_supported_label_set(config, binary_config):
    assert policy_labels(binary_config.policy) == BINARY_LABELS
    config.policy_file = "policies/triage-v0.1.json"
    assert policy_labels(config.policy) == LABELS
    assert classification_schema(config.policy)["properties"]["decision"]["enum"] == list(LABELS)
    for criteria in ({"ALLOW": "yes"}, {"ALLOW": "yes", "APPROVE": "yes"}, None):
        with pytest.raises(ValueError, match="Policy must define"):
            policy_labels({"criteria": criteria})


def test_binary_preset_retains_sampling_and_shared_budget():
    repo = Path(__file__).resolve().parents[1]
    original = load_config(repo / "configs/pilot.toml")
    binary = load_config(repo / "configs/pilot-binary.toml")
    assert policy_labels(binary.policy) == BINARY_LABELS
    assert binary.budget == original.budget and binary.models == original.models
    assert binary.data.model_dump(exclude={"directory"}) == original.data.model_dump(
        exclude={"directory"}
    )
    assert binary.dataset != original.dataset and binary.run != original.run


def decision_reply(choice="ALLOW", probabilities=None):
    return {
        "answers": {
            "triage": {
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities or {"ALLOW": 0.8, "BLOCK": 0.2},
            }
        },
        "usage": {"input_tokens": 12, "output_tokens": 3, "cost": 0.000001},
    }


def test_binary_distribution_requires_matching_policy(binary_config):
    body = decision_reply("BLOCK")
    prediction = parse_decisions(body, policy=binary_config.policy)
    assert prediction.decision == "BLOCK" and prediction.choice_argmax_mismatch
    assert prediction.probabilities == {"ALLOW": 0.8, "BLOCK": 0.2}
    with pytest.raises(ValueError, match="active policy"):
        parse_decisions(body)  # Legacy parser callers still expect three labels.


@pytest.mark.parametrize(
    "choice,probabilities",
    [
        ("REVIEW", {"ALLOW": 0.1, "REVIEW": 0.8, "BLOCK": 0.1}),
        ("ALLOW", {"ALLOW": 0.8, "REVIEW": 0, "BLOCK": 0.2}),
        ("REVIEW", {"ALLOW": 0.8, "BLOCK": 0.2}),
        ("ALLOW", {"ALLOW": 0.8}),
        ("ALLOW", {"ALLOW": 0.2, "BLOCK": 0.2}),
        ("ALLOW", {"ALLOW": float("nan"), "BLOCK": 0.2}),
    ],
)
def test_invalid_binary_distributions_rejected(binary_config, choice, probabilities):
    with pytest.raises(ValueError):
        parse_decisions(decision_reply(choice, probabilities), policy=binary_config.policy)


@pytest.mark.parametrize("probabilities", [None, {}, []])
def test_binary_provider_cannot_omit_distribution(binary_config, probabilities):
    body = decision_reply()
    body["answers"]["triage"]["probabilities"] = probabilities
    with pytest.raises(ValueError):
        parse_decisions(body, policy=binary_config.policy)


@pytest.mark.parametrize(
    "adapter,curator,outcome",
    [
        (adapter, curator, outcome)
        for adapter, curator in [
            ("decisions", False),
            ("chat", False),
            ("argo", False),
            ("chat", True),
            ("argo", True),
        ]
        for outcome in ("ALLOW", "REVIEW", "bad_reason", "abstain")
        if outcome != "bad_reason" or curator
    ],
)
def test_http_binary_contract(binary_config, monkeypatch, adapter, curator, outcome):
    monkeypatch.setenv("TEST_KEY", "test-secret")
    monkeypatch.setenv("ARGO_USER", "test-user")
    model = ModelConfig(
        name="test",
        adapter=adapter,
        model="argo:gpt-4.1-mini" if adapter == "argo" else "test",
        endpoint="https://example.test/decide",
        api_key_env="" if adapter == "argo" else "TEST_KEY",
        budget_mode="attempts_only",
    )
    calls = []

    def respond(request):
        calls.append(request)
        if adapter == "decisions":
            if outcome == "abstain":
                body = {"answers": {"triage": {"abstention": True}}}
            else:
                body = decision_reply(outcome)
        else:
            content = {"decision": "ALLOW" if outcome == "bad_reason" else outcome}
            if curator:
                content["reason"] = (
                    "insufficient_context" if outcome == "bad_reason" else "legitimate_use"
                )
            message = {"content": json.dumps(content)}
            if outcome == "abstain":
                message = {"refusal": "Declined"}
            body = {"choices": [{"finish_reason": "stop", "message": message}]}
        body["usage"] = {"input_tokens": 12, "output_tokens": 3}
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        instance = HTTPModel(
            binary_config,
            model,
            Ledger(binary_config.root / "ledger", 1, 10),
            "test",
            client=client,
            curator=curator,
        )
        result = instance.predict(Request(text="Define an atom."), binary_config.policy)
    assert result.status == (
        "ok" if outcome == "ALLOW" else "abstain" if outcome == "abstain" else "error"
    )
    if result.status == "error":
        assert result.error_code == "schema_error"
    assert result.input_tokens == 12 and result.attempts == 1
    wire = json.loads(calls[0].content)
    if adapter == "decisions":
        assert set(wire["questions"]["triage"]["criteria"]) == set(BINARY_LABELS)
        assert wire["state"] == {"text": "Define an atom.", "context": ""}
    else:
        schema = classification_schema(binary_config.policy, curator)
        if adapter == "chat":
            assert wire["response_format"]["json_schema"]["schema"] == schema
        else:
            assert '"enum":["ALLOW","BLOCK"]' in wire["messages"][0]["content"]
        assert schema["properties"]["decision"]["enum"] == list(BINARY_LABELS)
    for forbidden in ("source_label", "source_id", "source_split", "example_id", "scihazard"):
        assert forbidden not in calls[0].content.decode()


@pytest.mark.parametrize(
    "content,status",
    [
        ('{"decision":"BLOCK"}', "ok"),
        ('{"decision":"REVIEW"}', "error"),
        ('{"decision":"ALLOW","reason":"legitimate_use"}', "ok"),
        ('{"decision":"ALLOW","reason":"insufficient_context"}', "error"),
    ],
)
def test_codex_binary_contract(binary_config, fake_codex, content, status):
    fake_codex.content = content
    curator = "reason" in content
    instance = CodexModel(
        binary_config,
        ModelConfig(
            name="test",
            adapter="codex",
            model="codex:test",
            budget_mode="attempts_only",
            reasoning_effort="medium",
        ),
        Ledger(binary_config.root / "ledger", 1, 10),
        "test",
        curator=curator,
    )
    try:
        result = instance.predict(Request(text="Define an atom."), binary_config.policy)
    finally:
        instance.close()
    assert result.status == status and result.input_tokens == 12
    if status == "error":
        assert result.error_code == "schema_error"
    prompt, options = fake_codex.turns[0]
    assert options["output_schema"]["properties"]["decision"]["enum"] == list(BINARY_LABELS)
    assert json.loads(prompt) == {"untrusted_request": {"text": "Define an atom.", "context": ""}}


def test_replay_and_extension_adapters_reject_review(binary_replay):
    request = data.examples(binary_replay)[0].request
    records = {digest(request.model_dump()): {"status": "ok", "decision": "REVIEW"}}
    write_json(binary_replay.root / "binary-replay.json", records)
    replay = ReplayModel(binary_replay, binary_replay.models[0])
    assert replay.predict(request, binary_replay.policy).error_code == "schema_error"

    class Extension:
        def predict(self, request, policy):
            return Prediction(status="ok", decision="REVIEW", attempts=1, provider_cost_usd=0.01)

    rejected = predict_under_policy(Extension(), request, binary_replay.policy)
    assert rejected.status == "error" and rejected.decision is None
    assert rejected.attempts == 1 and rejected.provider_cost_usd == 0.01


def test_binary_metrics_keep_all_cases_and_annotations(binary_config):
    outcomes = ["BLOCK", "ALLOW", "abstain", "error", "BLOCK", "ALLOW", "abstain", "missing"]
    refs = [{"id": str(i), "source_label": "harmful" if i < 4 else "benign"} for i in range(8)]
    predictions = {
        str(i): Prediction(status="ok", decision=outcome)
        if outcome in BINARY_LABELS
        else Prediction(status=outcome)
        for i, outcome in enumerate(outcomes)
        if outcome != "missing"
    }
    m = metrics(refs, predictions, policy=binary_config.policy)
    assert m["source_confusion"] == {
        label: {"ALLOW": 1, "BLOCK": 1, "ABSTAIN": 1, "ERROR": 1} for label in ("benign", "harmful")
    }
    for name in ("accuracy", "strict_block_recall", "benign_block", "unsafe_allow"):
        assert m[name]["value"] == 0.25
    assert m["accuracy"]["n"] == 8 and m["accuracy"]["count"] == 2
    assert m["decision_coverage"]["value"] == 0.5
    assert m["decided_only"] == {"n": 4, "precision": 0.5, "f1": 0.5, "f2": 0.5}
    assert not m["operationally_complete"] and m["policy_agreement"] is None
    for r in refs:
        r["annotation"] = {
            "prediction": {
                "status": "ok",
                "decision": "BLOCK" if r["source_label"] == "harmful" else "ALLOW",
            }
        }
    agreement = metrics(refs, predictions, policy=binary_config.policy)["policy_agreement"]
    assert set(agreement["confusion"]) == set(BINARY_LABELS)
    assert agreement["macro_f1"] == pytest.approx(1 / 3)
    assert metrics([], {}, policy=binary_config.policy)["accuracy"]["value"] is None
    refs[0]["annotation"]["prediction"]["decision"] = "REVIEW"
    with pytest.raises(ValueError, match="Annotation decision"):
        metrics(refs, predictions, policy=binary_config.policy)


def test_binary_cli_workflow_report_and_resume(binary_replay):
    config = binary_replay
    path = config.root / "binary.toml"
    path.write_text(
        'project_root = "."\noffline = true\n'
        'policy_file = "policies/triage-binary-v0.1.json"\nrun_dir = "runs/offline"\n'
        '[data]\nloader = "fixture"\ncount = 10\nminimum_per_class = 4\n'
        'directory = ".local/data/test"\n'
        '[[models]]\nname = "replay-demo"\nadapter = "replay"\n'
        'model = "mock-fixture-v1"\nreplay_file = "binary-replay.json"\n'
    )
    cli = CliRunner()
    for command in (["data", "prepare"], ["data", "freeze"], ["evaluate"]):
        result = cli.invoke(app, [*command, "--config", str(path)])
        assert result.exit_code == 0, result.output
    before = (config.run / "predictions.jsonl").read_bytes()
    result = cli.invoke(app, ["evaluate", "--config", str(path)])
    assert result.exit_code == 0 and (config.run / "predictions.jsonl").read_bytes() == before
    result = cli.invoke(app, ["report", "--run", str(config.run)])
    assert result.exit_code == 0, result.output
    summary = read_json(config.run / "metrics.json")
    assert summary["decision_labels"] == list(BINARY_LABELS)
    assert "always-block" in summary["models"] and "always-review" not in summary["models"]
    primary = summary["models"]["replay-demo"]["splits"]["eval"]
    assert primary["accuracy"]["value"] == 1 and primary["accuracy"]["n"] == 8
    markdown = (config.run / "report.md").read_text()
    assert "source-label confusion" in markdown and "REVIEW" not in markdown
    assert "review_burden" not in (config.run / "comparison.csv").read_text()
    for filename in ("report.md", "comparison.csv", "metrics.json", "errors.jsonl"):
        content = (config.run / filename).read_text()
        assert all(e.request.text not in content for e in data.examples(config))


def test_binary_smoke_report_with_abstention_and_mixed_identities(binary_replay, monkeypatch):
    class Adapter:
        calls = 0

        def predict(self, request, policy):
            self.calls += 1
            common = {"attempts": 1, "returned_model": f"checkpoint-{self.calls % 2}"}
            if self.calls == 1:
                return Prediction(status="abstain", **common)
            return Prediction(
                status="ok",
                decision="ALLOW",
                probabilities={"ALLOW": 0.8, "BLOCK": 0.2},
                probability_origin="mock",
                **common,
            )

        def close(self):
            pass

    monkeypatch.setattr("chemgraphsafe.runner.build_model", lambda *args, **kwargs: Adapter())
    assert smoke(binary_replay)["complete"]
    run = binary_replay.run / "smoke"
    report(run)
    summary = read_json(run / "metrics.json")["models"]["replay-demo"]
    primary = summary["splits"]["smoke"]
    assert primary["accuracy"]["n"] == 5 and primary["accuracy"]["value"] == 0.8
    assert primary["strict_block_recall"]["value"] is None
    assert summary["mixed_identities"] and len(summary["per_identity"]) == 2
    assert all("accuracy" in value for value in summary["per_identity"].values())


@pytest.mark.parametrize("action", [evaluate, report])
@pytest.mark.parametrize("invalid", ["decision", "distribution"])
def test_binary_stored_predictions_rejected_without_writes(binary_replay, action, invalid):
    config = binary_replay
    evaluate(config)
    path = config.run / "predictions.jsonl"
    rows = read_rows(path)
    if invalid == "decision":
        rows[0]["prediction"]["decision"] = "REVIEW"
    else:
        rows[0]["prediction"].update(
            probabilities={"ALLOW": 0.5, "REVIEW": 0, "BLOCK": 0.5}, probability_origin="mock"
        )
    write_rows(path, rows)
    before = {p.name: p.read_bytes() for p in config.run.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="active policy"):
        action(config if action is evaluate else config.run)
    assert before == {p.name: p.read_bytes() for p in config.run.iterdir() if p.is_file()}


def test_binary_policy_requires_new_freeze_and_can_inherit_cases(config):
    evaluate(config)
    frozen = data.verify_freeze(config)
    original_dataset, original_run = config.dataset, config.run
    original_freeze = (original_dataset / "freeze.json").read_bytes()
    config.policy_file = "policies/triage-binary-v0.1.json"
    with pytest.raises(ValueError, match="Frozen policy/model/protocol changed"):
        evaluate(config)
    config.data.directory = ".local/data/binary"
    config.data.inherited_freeze = str(original_dataset / "freeze.json")
    data.prepare(config)
    binary = data.freeze(config)
    assert binary["split_manifest"] == frozen["split_manifest"]
    assert binary["examples_hash"] == frozen["examples_hash"]
    assert binary["semantic_hash"] != frozen["semantic_hash"]
    before = {p.name: p.read_bytes() for p in original_run.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="Run identity"):
        evaluate(config)
    assert before == {p.name: p.read_bytes() for p in original_run.iterdir() if p.is_file()}
    assert (original_dataset / "freeze.json").read_bytes() == original_freeze


@pytest.mark.parametrize("invalid", [None, "decision", "reason"])
def test_binary_curation_freeze_and_report(binary_replay, invalid):
    config = binary_replay
    records = read_json(config.root / "binary-replay.json")
    for prediction in records.values():
        prediction["reason"] = (
            "legitimate_use" if prediction["decision"] == "ALLOW" else "harm_facilitation"
        )
    write_json(config.root / "curator.json", records)
    config.curator = ModelConfig(
        name="curator", adapter="replay", model="owned-curator", replay_file="curator.json"
    )
    assert curate(config)["annotations"] == 10
    if invalid:
        path = config.dataset / "annotations.jsonl"
        rows = read_rows(path)
        rows[0]["prediction"][invalid] = (
            "REVIEW" if invalid == "decision" else "insufficient_context"
        )
        write_rows(path, rows)
        before = path.read_bytes()
        with pytest.raises(ValueError, match="active policy"):
            data.freeze(config)
        assert not (config.dataset / "freeze.json").exists()
        assert path.read_bytes() == before
        return
    assert data.freeze(config)["annotation_count"] == 10
    assert evaluate(config)["complete"]
    report(config.run)
    primary = read_json(config.run / "metrics.json")["models"]["replay-demo"]["splits"]["eval"]
    assert primary["policy_agreement"]["macro_f1"] == 1
