import json

import httpx
import pytest

from chemgraphsafe import adapters, data, runner
from chemgraphsafe.budget import Ledger
from chemgraphsafe.config import ModelConfig
from chemgraphsafe.reporting import report
from chemgraphsafe.storage import read_json, read_rows


@pytest.fixture
def mock_argo(monkeypatch):
    monkeypatch.setenv("ARGO_USER", "test-user-not-for-artifacts")
    original = httpx.Client
    calls = []

    def response(request):
        calls.append(request)
        wire = json.loads(request.content)
        value = {"decision": "ALLOW"}
        # Curation's schema is in the trusted system prompt.
        if '"required":["decision","reason"]' in wire["messages"][0]["content"]:
            value["reason"] = "legitimate_use"
        return httpx.Response(
            200,
            json={
                "model": wire["model"],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}],
            },
        )

    monkeypatch.setattr(
        adapters.httpx,
        "Client",
        lambda **kw: original(**kw, transport=httpx.MockTransport(response)),
    )
    return calls


@pytest.mark.parametrize("backend", ["argo", "codex"])
def test_new_backends_curate_evaluate_report_and_resume(config, mock_argo, fake_codex, backend):
    config.offline = False
    model_ids = (
        ("argo:gpt-4.1-mini", "argo:gpt-4.1")
        if backend == "argo"
        else ("codex:baseline", "codex:curator")
    )
    config.models = [
        ModelConfig(
            name="baseline", adapter=backend, model=model_ids[0], budget_mode="attempts_only"
        )
    ]
    config.curator = ModelConfig(
        name="curator", adapter=backend, model=model_ids[1], budget_mode="attempts_only"
    )
    data.prepare(config)
    fake_codex.content = '{"decision":"ALLOW","reason":"legitimate_use"}'
    assert runner.curate(config)["annotations"] == 10
    assert data.freeze(config)["annotation_count"] == 10
    fake_codex.content = '{"decision":"ALLOW"}'
    assert runner.evaluate(config)["complete"]
    ledger = runner.ledger_for(config)
    assert ledger.summary()["attempts"] == 20
    assert ledger.summary()["accounted_usd"] == 0
    before = (config.run / "predictions.jsonl").read_bytes()
    assert runner.evaluate(config)["complete"]
    assert (config.run / "predictions.jsonl").read_bytes() == before
    assert ledger.summary()["attempts"] == 20
    report(config.run)
    stats = read_json(config.run / "metrics.json")["models"]["baseline"]
    assert stats["budget_mode"] == "attempts_only"
    assert stats["row_estimated_cost_usd"] is None
    assert stats["input_tokens"] > 0
    assert stats["splits"]["eval"]["annotation_coverage"]["value"] == 1
    text = (config.run / "report.md").read_text()
    assert "Attempts-only usage" in text
    for path in (config.run / "manifest.json", config.run / "predictions.jsonl"):
        assert "test-user-not-for-artifacts" not in path.read_text()
        for example in data.examples(config):
            assert example.request.text not in path.read_text()


def test_all_models_preflight_before_inference(config, mock_argo, fake_codex):
    config.offline = False
    config.models = [
        ModelConfig(
            name="argo", adapter="argo", model="argo:gpt-4.1-mini", budget_mode="attempts_only"
        ),
        ModelConfig(name="codex", adapter="codex", model="codex:test", budget_mode="attempts_only"),
    ]
    fake_codex.account = "apiKey"
    with pytest.raises(ValueError, match="preflight"):
        runner.smoke(config)
    assert not mock_argo and not fake_codex.turns
    assert runner.ledger_for(config).summary()["attempts"] == 0


def test_codex_quota_stops_remaining_model_calls(config, fake_codex):
    config.offline = False
    config.models = [
        ModelConfig(name="codex", adapter="codex", model="codex:test", budget_mode="attempts_only")
    ]
    fake_codex.outcome, fake_codex.failure = "failed", "usageLimitExceeded"
    assert not runner.smoke(config)["complete"]
    rows = read_rows(config.run / "smoke/predictions.jsonl")
    assert rows[0]["prediction"]["error_code"] == "codex_quota_exhausted"
    assert all(
        r["prediction"]["error_code"] == "not_attempted_codex_quota_exhausted" for r in rows[1:]
    )
    assert len(fake_codex.turns) == 1


def test_included_smoke_budget_shared_across_providers(config, mock_argo, fake_codex):
    config.offline = False
    config.budget.max_attempts = 6
    config.models = [
        ModelConfig(
            name="argo", adapter="argo", model="argo:gpt-4.1-mini", budget_mode="attempts_only"
        )
    ]
    assert runner.smoke(config)["complete"]
    config.models = [
        ModelConfig(name="codex", adapter="codex", model="codex:test", budget_mode="attempts_only")
    ]
    config.run_dir = "runs/codex"
    assert not runner.smoke(config)["complete"]
    ledger = Ledger(config.path(config.budget.ledger), 10, 6)
    assert ledger.summary()["attempts_only"] == 6
    assert ledger.summary()["unreconciled_attempts"] == 0
