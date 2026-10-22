"""Native category classification using owned requests and mocked providers only."""

import json
import shutil
from pathlib import Path

import httpx
import pytest
from test_scisafetybench import challenge as challenge
from typer.testing import CliRunner

from chemgraphsafe import data, scisafetybench
from chemgraphsafe.adapters import HTTPModel, ReplayModel, parse_chat, parse_decisions
from chemgraphsafe.budget import Ledger
from chemgraphsafe.cli import app
from chemgraphsafe.codex import CodexModel
from chemgraphsafe.config import Config, ModelConfig, load_config
from chemgraphsafe.models import RISK_TYPES, Prediction, Request, policy_labels, risk_baselines
from chemgraphsafe.prompts import (
    classification_messages,
    classification_schema,
    codex_prompt_contract,
)
from chemgraphsafe.reporting import metrics, report
from chemgraphsafe.runner import curate, evaluate, predict_under_policy, smoke
from chemgraphsafe.storage import digest, file_hash, read_json, read_rows, write_json, write_rows


@pytest.fixture
def risk_config(challenge):
    challenge.annotation_rule = None
    challenge.task = "scisafetybench_risk_type"
    challenge.policy_file = "policies/scisafetybench-risk-type-v1.json"
    return challenge


@pytest.fixture
def risk_replay(risk_config):
    data.prepare(risk_config)
    risk_config.models[0].replay_file = "risk-replay.json"
    write_json(
        risk_config.root / "risk-replay.json",
        {
            digest(e.request.model_dump()): {"status": "ok", "decision": e.category}
            for e in data.examples(risk_config)
        },
    )
    return risk_config


def decision_reply(label, probabilities=None):
    return {
        "answers": {
            "risk_type": {
                "type": "choice",
                "choice": label,
                "probabilities": probabilities
                if probabilities is not None
                else {key: float(key == label) for key in RISK_TYPES},
            }
        },
        "model": "test-checkpoint",
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


def http_model(adapter):
    return ModelConfig(
        name="classifier",
        adapter=adapter,
        model="argo:gpt-4o" if adapter == "argo" else "test",
        endpoint="https://example.test/classify",
        api_key_env="" if adapter == "argo" else "TEST_KEY",
        argo_user_env="TEST_USER",
        budget_mode="attempts_only",
    )


@pytest.mark.parametrize("adapter", ["decisions", "chat", "argo"])
@pytest.mark.parametrize("label", RISK_TYPES)
def test_http_risk_contract_and_reference_independent_input(
    risk_config, monkeypatch, adapter, label
):
    monkeypatch.setenv("TEST_KEY", "test")
    monkeypatch.setenv("TEST_USER", "test")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        body = (
            decision_reply(label)
            if adapter == "decisions"
            else {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps({"decision": label}),
                        },
                    }
                ],
                "model": "test-checkpoint",
            }
        )
        return httpx.Response(200, json=body)

    example = scisafetybench.SciSafetyBenchLoader().load(risk_config)[0]
    changed = example.model_copy(
        update={
            "id": "PRIVATE ID",
            "category": RISK_TYPES[-1],
            "domain": "PRIVATE DOMAIN",
            "source_id": "PRIVATE SOURCE",
            "source_revision": "PRIVATE REVISION",
        }
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = HTTPModel(
            risk_config,
            http_model(adapter),
            Ledger(risk_config.root / "ledger", 1, 10),
            "test",
            client=client,
        )
        for row in (example, changed):
            prediction = model.predict(row.request, risk_config.policy)
            assert prediction.decision == label
            assert prediction.returned_model == "test-checkpoint"
            assert (prediction.probabilities is not None) == (adapter == "decisions")
    assert requests[0] == requests[1]
    wire = requests[0]
    assert "PRIVATE" not in json.dumps(wire)
    if adapter == "decisions":
        assert wire["state"] == example.request.model_dump()
        assert set(wire["questions"]) == {"risk_type"}
        assert tuple(wire["questions"]["risk_type"]["criteria"]) == RISK_TYPES
    else:
        assert wire["messages"] == classification_messages(
            example.request, risk_config.policy, classification_schema(risk_config.policy)
        )


@pytest.mark.parametrize("label", RISK_TYPES)
def test_codex_risk_contract(risk_config, fake_codex, label):
    fake_codex.content = json.dumps({"decision": label})
    model = ModelConfig(
        name="codex",
        adapter="codex",
        model="codex:test",
        reasoning_effort="high",
        budget_mode="attempts_only",
    )
    instance = CodexModel(risk_config, model, Ledger(risk_config.root / "ledger", 1, 10), "run")
    request = Request(text="Explain the classroom vocabulary entry.")
    prediction = instance.predict(request, risk_config.policy)
    assert prediction.decision == label and prediction.probabilities is None
    assert fake_codex.turns[0][1]["output_schema"]["properties"]["decision"]["enum"] == list(
        RISK_TYPES
    )
    assert "risk-type classifier" in fake_codex.threads[0]["baseInstructions"]
    assert "Do not invoke any native tools" in fake_codex.threads[0]["baseInstructions"]
    assert json.loads(fake_codex.turns[0][0]) == {"untrusted_request": request.model_dump()}


@pytest.mark.parametrize(
    "bad",
    [
        {"ALLOW": 1.0},
        {RISK_TYPES[0]: 1.0},
        dict.fromkeys(RISK_TYPES, 0.5),
        {**dict.fromkeys(RISK_TYPES, 0.25), "ALLOW": 0.0},
        {**dict.fromkeys(RISK_TYPES, 0.0), RISK_TYPES[0]: float("nan")},
    ],
)
def test_invalid_distributions_fail(risk_config, bad):
    with pytest.raises(ValueError):
        parse_decisions(decision_reply(RISK_TYPES[0], bad), policy=risk_config.policy)


def test_cross_task_labels_keys_and_abstention(risk_config):
    with pytest.raises(ValueError):
        Prediction(status="ok", decision=RISK_TYPES[0]).validate_policy()
    with pytest.raises(ValueError):
        Prediction(status="ok", decision="ALLOW").validate_policy(risk_config.policy)
    with pytest.raises(ValueError):
        parse_decisions(decision_reply(RISK_TYPES[0]))
    wrong_key = {"answers": {"triage": decision_reply(RISK_TYPES[0])["answers"]["risk_type"]}}
    with pytest.raises(ValueError):
        parse_decisions(wrong_key, policy=risk_config.policy)
    assert (
        parse_decisions(
            {"answers": {"risk_type": {"abstention": True}}}, policy=risk_config.policy
        ).status
        == "abstain"
    )
    assert (
        parse_chat(
            {"choices": [{"finish_reason": "stop", "message": {"refusal": "no"}}]},
            policy=risk_config.policy,
        ).status
        == "abstain"
    )


@pytest.mark.parametrize("adapter", ["decisions", "chat", "argo"])
def test_http_risk_failure_is_accounted(risk_config, monkeypatch, adapter):
    monkeypatch.setenv("TEST_KEY", "test")
    monkeypatch.setenv("TEST_USER", "test")
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))
    ) as client:
        ledger = Ledger(risk_config.root / "ledger", 1, 10)
        model = HTTPModel(risk_config, http_model(adapter), ledger, "test", client=client)
        result = model.predict(Request(text="Define an atom."), risk_config.policy)
    assert result.status == "error" and result.error_code == "schema_error"
    assert result.attempts == 1 and ledger.summary()["attempts"] == 1


def test_replay_and_extension_reject_triage(risk_replay):
    row = data.examples(risk_replay)[0]
    write_json(
        risk_replay.root / "risk-replay.json",
        {digest(row.request.model_dump()): {"status": "ok", "decision": "ALLOW"}},
    )
    assert (
        ReplayModel(risk_replay, risk_replay.models[0])
        .predict(row.request, risk_replay.policy)
        .error_code
        == "schema_error"
    )

    class WrongTask:
        def predict(self, request, policy):
            return Prediction(status="ok", decision="REVIEW", attempts=1, input_tokens=7)

    result = predict_under_policy(WrongTask(), row.request, risk_replay.policy)
    assert result.error_code == "schema_error" and result.input_tokens == 7


def test_80_case_native_workflow_reports_and_resume(risk_replay):
    config = risk_replay
    frozen = data.freeze(config)
    assert frozen["protocol"] == "scisafetybench-risk-type-1"
    assert frozen["annotations_hash"] is None and frozen["annotation_count"] == 0
    assert set(frozen["split_manifest"]["splits"].values()) == {"eval"}
    assert evaluate(config)["complete"]
    before = (config.run / "predictions.jsonl").read_bytes()
    assert evaluate(config)["complete"]
    assert (config.run / "predictions.jsonl").read_bytes() == before
    assert report(config.run)["errors"] == 0
    result = read_json(config.run / "metrics.json")
    main = result["models"][config.models[0].name]
    score = main["splits"]["eval"]
    assert score["accuracy"]["value"] == score["macro_f1"] == 1
    assert score["accuracy"]["n"] == score["reference_coverage"]["count"] == 80
    assert score["reference"] == "source_native"
    assert list(score["reference_counts"].values()) == [19, 18, 20, 23]
    assert set(risk_baselines()) < result["models"].keys()
    assert (
        result["models"]["always-unintentional-consequences"]["splits"]["eval"]["accuracy"]["value"]
        == 23 / 80
    )
    assert all(m["n"] == 40 for m in main["slices"]["domain"].values())
    assert all(m["macro_f1"] is None for m in main["slices"]["category_id"].values())
    for name in ("metrics.json", "report.md", "comparison.csv", "references.json"):
        text = (config.run / name).read_text()
        assert "PRIVATE" not in text and "classroom vocabulary" not in text
        assert "unsafe_allow" not in text and "escalation_rate" not in text
    assert not (config.dataset / "annotations.jsonl").exists()
    assert not config.path(config.budget.ledger).exists()


def test_metrics_keep_errors_abstentions_and_missing_in_denominators(risk_config):
    refs = [
        {"id": str(i), "source_risk_type": label, "reference_origin": "source_native"}
        for i, label in enumerate(RISK_TYPES * 2)
    ]
    predictions = {
        str(i): Prediction(status="ok", decision=label) for i, label in enumerate(RISK_TYPES)
    }
    predictions.update(
        {
            "4": Prediction(status="ok", decision=RISK_TYPES[1]),
            "5": Prediction(status="abstain"),
            "6": Prediction(status="error"),
        }
    )
    score = metrics(refs, predictions, policy=risk_config.policy)
    assert score["accuracy"]["value"] == 4 / 8
    assert score["decision_coverage"]["value"] == 5 / 8
    assert score["macro_f1"] == pytest.approx((2 / 3 + 1 / 2 + 2 / 3 + 2 / 3) / 4)
    assert score["confusion"][RISK_TYPES[1]]["ABSTAIN"] == 1
    assert score["confusion"][RISK_TYPES[2]]["ERROR"] == 1
    assert score["confusion"][RISK_TYPES[3]]["ERROR"] == 1
    assert metrics([], {}, policy=risk_config.policy)["macro_f1"] is None
    all_errors = metrics(refs, {}, policy=risk_config.policy)
    assert all_errors["accuracy"]["value"] == all_errors["macro_f1"] == 0
    assert all_errors["precision"][RISK_TYPES[0]]["value"] is None


def test_smoke_has_no_gold_and_reports_operational_results(risk_replay, monkeypatch):
    class Adapter:
        def predict(self, request, policy):
            return Prediction(status="ok", decision=RISK_TYPES[0])

        def close(self):
            pass

    monkeypatch.setattr("chemgraphsafe.runner.build_model", lambda *args: Adapter())
    assert smoke(risk_replay)["complete"]
    run = risk_replay.run / "smoke"
    report(run)
    result = read_json(run / "metrics.json")["models"][risk_replay.models[0].name]["splits"][
        "smoke"
    ]
    assert result["accuracy"]["value"] is None and result["macro_f1"] is None
    assert result["decision_coverage"]["value"] == 1
    assert "Transport/schema smoke only" in (run / "report.md").read_text()


@pytest.mark.parametrize("field", ["curator", "annotation_rule", "protocol", "loader"])
def test_incompatible_configs_rejected(risk_config, field):
    values = risk_config.model_dump()
    if field == "curator":
        values[field] = ModelConfig(name="curator", adapter="replay", model="curator").model_dump()
    elif field == "annotation_rule":
        values[field] = scisafetybench.RULE_ID
    else:
        values["data"][field] = "balanced_pilot" if field == "protocol" else "scihazard"
    with pytest.raises(ValueError):
        Config.model_validate(values)


def test_task_policy_mismatch_and_annotation_commands_fail_before_inference(
    risk_config, monkeypatch
):
    monkeypatch.setattr("chemgraphsafe.runner.build_model", lambda *a: pytest.fail("inference"))
    for action in (data.annotate, curate):
        with pytest.raises(ValueError, match="source labels"):
            action(risk_config)
    risk_config.policy_file = "policies/triage-v0.1.json"
    with pytest.raises(ValueError, match="task"):
        data.prepare(risk_config)
    assert not (risk_config.dataset / "prepare.json").exists()


def test_risk_freeze_rejects_existing_annotations(risk_replay):
    (risk_replay.dataset / "annotations.jsonl").write_text("")
    with pytest.raises(ValueError, match="annotations file"):
        data.freeze(risk_replay)


@pytest.mark.parametrize("field", ["category", "source_revision", "source_id", "domain"])
def test_freeze_checks_reference_against_pinned_source(risk_replay, field):
    path = risk_replay.dataset / "examples.jsonl"
    rows = read_rows(path)
    rows[0][field] = (
        RISK_TYPES[(RISK_TYPES.index(rows[0][field]) + 1) % 4] if field == "category" else "changed"
    )
    write_rows(path, rows)
    manifest = read_json(risk_replay.dataset / "prepare.json")
    manifest["examples_hash"] = file_hash(path)
    write_json(risk_replay.dataset / "prepare.json", manifest)
    with pytest.raises(ValueError, match="pinned source"):
        data.freeze(risk_replay)


def test_freeze_and_report_tampering_and_changed_task_definition(risk_replay):
    data.freeze(risk_replay)
    evaluate(risk_replay)
    path = risk_replay.run / "references.json"
    refs = read_json(path)
    refs[0]["source_risk_type"] = "changed"
    write_json(path, refs)
    with pytest.raises(ValueError, match="Reference contents"):
        report(risk_replay.run)
    definition = risk_replay.path(risk_replay.policy_file)
    policy = read_json(definition)
    policy["instructions"] += " Changed definition."
    write_json(definition, policy)
    with pytest.raises(ValueError, match="Frozen policy/model/protocol changed"):
        evaluate(risk_replay)


def test_cross_task_run_reuse_rejected(risk_replay):
    data.freeze(risk_replay)
    evaluate(risk_replay)
    original = (risk_replay.run / "predictions.jsonl").read_bytes()
    risk_replay.task = "triage"
    risk_replay.policy_file = "policies/triage-chem-materials-oversight-v0.1.json"
    with pytest.raises(ValueError, match="Frozen policy"):
        evaluate(risk_replay)
    risk_replay.data.directory = ".local/data/new-triage"
    # Use the same owned source cache in a fresh dataset, with a distinct freeze.
    shutil.copytree(risk_replay.root / ".local/data/challenge/raw", risk_replay.dataset / "raw")
    data.prepare(risk_replay)
    data.freeze(risk_replay)
    with pytest.raises(ValueError, match="Run identity"):
        evaluate(risk_replay)
    assert (risk_replay.run / "predictions.jsonl").read_bytes() == original


def test_new_preset_and_cli_workflow(risk_replay, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    new = load_config(repo / "configs/scisafetybench-risk-type.toml")
    old = load_config(repo / "configs/scisafetybench.toml")
    assert new.task == "scisafetybench_risk_type" and old.task == "triage"
    assert new.models == old.models and new.budget == old.budget
    assert new.dataset != old.dataset and new.run != old.run
    assert policy_labels(new.policy) == RISK_TYPES
    monkeypatch.setattr("chemgraphsafe.cli.load_config", lambda _: risk_replay)
    config_path = risk_replay.root / "test.toml"
    config_path.touch()
    cli = CliRunner()
    for args in (["data", "prepare"], ["data", "freeze"], ["evaluate"]):
        result = cli.invoke(app, [*args, "--config", str(config_path)])
        assert result.exit_code == 0, result.output
    assert cli.invoke(app, ["report", "--run", str(risk_replay.run)]).exit_code == 0


# Captured from the unmodified implementation before adding the task option.
LEGACY_HASHES = {
    "argo.toml": "c7591be315be74d3d4522904ab872c4e2228070d8bd1e300faaf612ead8b6056",
    "codex-astra-full.toml": "7638154e6baa7f65b39274249288322a90c80b52b146f153168b0ff1340a9913",
    "codex-online-test.toml": "2fde884730978d98e41998a898a632130bb6a5dae24fffc594d96a3560c00b3f",
    "codex.toml": "b369488ca008a8923219f3d7ecb3b5af67b8cb7be06e2742b35b5a9c7a1f4a72",
    "offline.toml": "b3a539a0951205e836faaa793171662435abd89be50b7017066ec42518cb0959",
    "pilot-binary.toml": "0f55dca0a73f9148b05ebc1d3575406e9c0ef1f10f2ac8c69013ad2cc2079108",
    "pilot.toml": "1fc2c1f0260ee91ef570f7b33556202159011697ae50efc928b9c8e8e7f31e91",
    "scisafetybench.toml": "65e2545b2e378a8c7e1a3bc5dab4300da34365affa01da8e3c6fa3cf8321c877",
}


@pytest.mark.parametrize("filename", LEGACY_HASHES)
def test_legacy_semantics_and_prompt_fingerprints_are_unchanged(filename):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / filename)
    assert config.semantic_hash() == LEGACY_HASHES[filename]
    expected = {
        "pilot-binary.toml": "a28f325959ef955b7e4f1fd07a6c85a0528c7ddd84ba8589f012acbc49355a13",
        "scisafetybench.toml": "296eed4b4df961856bcf9b8096f3f7ed487acc9b86b8c310eaf9385edcf25344",
    }.get(filename, "0a4af161234ddfbe995eb130418cb2716d568159a254e11fae7ea933922bf783")
    assert codex_prompt_contract(config.policy) == expected
    config.task = "triage"
    assert config.semantic_hash() == LEGACY_HASHES[filename]


def test_legacy_artifacts_without_task_still_verify_and_report(config):
    evaluate(config)
    frozen = read_json(config.dataset / "freeze.json")
    assert "task" not in frozen["semantic_config"] and frozen["protocol"] == "triage-1"
    assert "task" not in read_json(config.run / "manifest.json")
    assert data.verify_freeze(config) == frozen
    report(config.run)
    assert "task" not in read_json(config.run / "metrics.json")


def test_owned_offline_fixture_workflow(config):
    rows = read_rows(config.path(config.data.fixture))
    for i, row in enumerate(rows):
        row.update(source_label=None, category=RISK_TYPES[i % 4])
    write_rows(config.path(config.data.fixture), rows)
    config.task = "scisafetybench_risk_type"
    config.policy_file = "policies/scisafetybench-risk-type-v1.json"
    config.data.protocol = "challenge_eval"
    data.prepare(config)
    write_json(
        config.path(config.models[0].replay_file),
        {
            digest(e.request.model_dump()): {"status": "ok", "decision": e.category}
            for e in data.examples(config)
        },
    )
    assert evaluate(config)["complete"]  # The existing automatic offline freeze also works.
    report(config.run)
    score = read_json(config.run / "metrics.json")["models"][config.models[0].name]["splits"][
        "eval"
    ]
    assert score["reference"] == "owned_fixture" and score["accuracy"]["value"] == 1


def test_native_hosted_approval_still_precedes_adapter_and_ledger(risk_config, monkeypatch):
    risk_config.offline = False
    risk_config.models = [http_model("decisions")]
    data.prepare(risk_config)
    data.freeze(risk_config)
    monkeypatch.setattr("chemgraphsafe.runner.build_model", lambda *a: pytest.fail("provider"))
    with pytest.raises(ValueError, match="approval"):
        evaluate(risk_config)
    assert not risk_config.path(risk_config.budget.ledger).exists()


def test_resume_failures_and_mixed_identity_reports(risk_replay, monkeypatch):
    rows = data.examples(risk_replay)
    labels = {e.request.text: e.category for e in rows}

    class Adapter:
        calls = 0

        def predict(self, request, policy):
            self.calls += 1
            common = {
                "returned_model": f"checkpoint-{self.calls % 2}",
                "attempts": 1,
                "input_tokens": 3,
                "latency_ms": 2.0,
            }
            if self.calls == 1:
                return Prediction(status="error", **common)
            if self.calls == 2:
                return Prediction(status="abstain", **common)
            label = labels[request.text]
            if self.calls == 3:
                label = RISK_TYPES[(RISK_TYPES.index(label) + 1) % 4]
            return Prediction(status="ok", decision=label, **common)

        def close(self):
            pass

    adapter = Adapter()
    monkeypatch.setattr("chemgraphsafe.runner.build_model", lambda *a: adapter)
    data.freeze(risk_replay)
    assert not evaluate(risk_replay)["complete"]
    assert evaluate(risk_replay)["complete"] and adapter.calls == 81
    assert report(risk_replay.run)["errors"] == 2
    summary = read_json(risk_replay.run / "metrics.json")["models"][risk_replay.models[0].name]
    assert summary["splits"]["eval"]["accuracy"]["value"] == 78 / 80
    assert summary["mixed_identities"] and len(summary["per_identity"]) == 2
    assert summary["latency_ms"]["n"] == 80 and summary["input_tokens"] == 240
    assert sum(m["n"] for m in summary["per_identity"].values()) == 80
    issues = {row["issue"] for row in read_rows(risk_replay.run / "errors.jsonl")}
    assert issues == {"source_risk_type_disagreement", "native_abstention"}
    assert "MIXED RETURNED MODEL IDENTITIES" in (risk_replay.run / "report.md").read_text()


@pytest.mark.parametrize("task", ["triage", "scisafetybench_risk_type"])
def test_code_change_requires_new_run_but_keeps_freeze(risk_replay, task, monkeypatch):
    if task == "triage":
        risk_replay.task = task
        risk_replay.policy_file = "policies/triage-chem-materials-oversight-v0.1.json"
        records = read_json(risk_replay.path(risk_replay.models[0].replay_file))
        for value in records.values():
            value["decision"] = "REVIEW"
        write_json(risk_replay.path(risk_replay.models[0].replay_file), records)
    frozen = data.freeze(risk_replay)
    evaluate(risk_replay)
    monkeypatch.setattr("chemgraphsafe.runner.code_hash", lambda: "updated-code")
    assert data.verify_freeze(risk_replay) == frozen
    report(risk_replay.run)
    with pytest.raises(ValueError, match="Run identity"):
        evaluate(risk_replay)
    risk_replay.run_dir = "runs/updated-code"
    assert evaluate(risk_replay)["complete"]


def test_malformed_policy_and_task_metadata_fail_closed(risk_config):
    policy = risk_config.policy
    with pytest.raises(ValueError, match="four SciSafetyBench"):
        policy_labels({**policy, "criteria": {**policy["criteria"], "ALLOW": "extra"}})
    with pytest.raises(ValueError, match="Unknown classification task"):
        policy_labels({**policy, "task": "unknown"})
    with pytest.raises(ValueError, match="nonempty definitions"):
        policy_labels({**policy, "criteria": dict.fromkeys(RISK_TYPES, "")})
    with pytest.raises(ValueError, match="instructions"):
        policy_labels({**policy, "instructions": ""})


@pytest.mark.parametrize("status", ["bad_label", "tool"])
def test_codex_risk_rejects_wrong_task_or_tool(risk_config, fake_codex, status):
    fake_codex.content = '{"decision":"ALLOW"}'
    if status == "tool":
        from types import SimpleNamespace

        fake_codex.events = [
            SimpleNamespace(
                method="item/started",
                payload={"turnId": "turn-1", "item": {"type": "commandExecution"}},
            )
        ]
    model = ModelConfig(
        name="codex",
        adapter="codex",
        model="codex:test",
        reasoning_effort="high",
        budget_mode="attempts_only",
    )
    instance = CodexModel(risk_config, model, Ledger(risk_config.root / "ledger", 1, 10), "run")
    result = instance.predict(Request(text="Define an atom."), risk_config.policy)
    assert result.error_code == ("schema_error" if status == "bad_label" else "codex_tool_attempt")


@pytest.mark.parametrize("label", RISK_TYPES)
@pytest.mark.parametrize("truncated", [False, True])
def test_laya_native_risk_contract_after_upstream_integration(risk_config, label, truncated):
    from test_laya_adapter import health, laya_config, laya_reply

    from chemgraphsafe.adapters import LayaModel

    requests = []

    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json=health())
        requests.append(json.loads(request.content))
        body = laya_reply()
        body["answers"] = decision_reply(label)["answers"]
        if truncated:
            body["usage"].update(
                state_tokens_dropped=1, truncated=True, truncated_questions=["risk_type"]
            )
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = LayaModel(
            risk_config,
            laya_config(),
            Ledger(risk_config.root / "ledger", 1, 10),
            "risk",
            client=client,
        )
        request = Request(text="Define an atom for a classroom glossary.")
        result = adapter.predict(request, risk_config.policy)
    assert requests[0]["state"] == request.model_dump()
    assert tuple(requests[0]["questions"]["risk_type"]["criteria"]) == RISK_TYPES
    if truncated:
        assert result.error_code == "input_truncated" and result.decision is None
    else:
        assert result.decision == label and set(result.probabilities) == set(RISK_TYPES)
    assert result.returned_model == f"english@{laya_config().expected_revision}"
    assert result.provider_cost_usd is None and result.attempts == 1
