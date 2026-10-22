import json
import sqlite3

import httpx
import pytest

from chemgraphsafe.adapters import HTTPModel, build_model
from chemgraphsafe.budget import BudgetExceeded, Ledger
from chemgraphsafe.config import Config, ModelConfig, load_config
from chemgraphsafe.models import Request
from chemgraphsafe.providers import ARGO_ENDPOINT, model_identity


def argo_config(**kwargs):
    return ModelConfig(
        **{
            "name": "argo",
            "adapter": "argo",
            "model": "argo:gpt-4.1-mini",
            "budget_mode": "attempts_only",
            **kwargs,
        }
    )


def chat_reply(content='{"decision":"ALLOW"}', **kwargs):
    return {
        "model": "gpt41mini",
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        **kwargs,
    }


@pytest.mark.parametrize("curator", [False, True])
def test_argo_wire_contract(config, monkeypatch, curator):
    monkeypatch.setenv("ARGO_USER", "anl-user")
    monkeypatch.setenv("OPENAI_API_KEY", "never-send-this-secret")
    body = {"decision": "ALLOW"}
    if curator:
        body["reason"] = "legitimate_use"
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=chat_reply(json.dumps(body)))

    ledger = Ledger(config.root / "ledger", 1, 10)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        adapter = build_model(config, argo_config(), ledger, "test", curator=curator, client=client)
        result = adapter.predict(
            Request(text="Explain chemical bonds.", context="Classroom"), config.policy
        )
    wire = json.loads(calls[0].content)
    assert str(calls[0].url) == ARGO_ENDPOINT
    assert calls[0].headers["Authorization"] == "Bearer anl-user"
    assert wire["user"] == "anl-user" and wire["model"] == "gpt41mini"
    assert "response_format" not in wire and "tools" not in wire
    assert wire["max_tokens"] == 256
    assert "additionalProperties" in wire["messages"][0]["content"]
    assert json.loads(wire["messages"][1]["content"]) == {
        "untrusted_request": {"text": "Explain chemical bonds.", "context": "Classroom"}
    }
    assert "never-send-this-secret" not in str(calls[0].headers)
    for field in (
        "source_label",
        "source_id",
        "source_split",
        "example_id",
        "scihazard",
    ):
        assert field not in calls[0].content.decode()
    assert result.status == "ok" and result.decision == "ALLOW"
    assert result.reason == ("legitimate_use" if curator else None)
    assert result.input_tokens == 12 and result.output_tokens == 3
    assert result.returned_model == "gpt41mini" and result.probabilities is None
    assert result.estimated_cost_usd is None and result.provider_cost_usd is None
    assert ledger.summary()["attempts_only"] == 1
    assert ledger.summary()["accounted_usd"] == 0
    assert ledger.summary()["unreconciled_attempts"] == 0


@pytest.mark.parametrize(
    "content",
    [
        "I refuse.",
        "```json\n{}\n```",
        '{"decision":"MAYBE"}',
        '{"decision":"ALLOW","extra":1}',
        "null",
    ],
)
def test_argo_malformed_output_is_error(config, monkeypatch, content):
    monkeypatch.setenv("ARGO_USER", "user")
    ledger = Ledger(config.root / "ledger", 1, 10)
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=chat_reply(content)))
    ) as client:
        result = HTTPModel(config, argo_config(), ledger, "test", client=client).predict(
            Request(text="Atom?"), config.policy
        )
    assert result.error_code == "schema_error"
    assert result.input_tokens == 12 and result.attempts == 1


def test_argo_credentials_preflight_and_included_retries(config, monkeypatch):
    monkeypatch.delenv("ARGO_USER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "not-argo")
    ledger = Ledger(config.root / "ledger", 1, 2)
    with pytest.raises(ValueError, match="ARGO_USER"):
        build_model(config, argo_config(), ledger)
    assert ledger.summary()["attempts"] == 0
    monkeypatch.setenv("ARGO_USER", "user")
    delays = []
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(429, headers={"Retry-After": "1"}))
    ) as client:
        adapter = HTTPModel(
            config, argo_config(), ledger, "test", client=client, sleep=delays.append
        )
        result = adapter.predict(Request(text="Atom?"), config.policy)
    assert result.error_code == "budget_exhausted" and result.attempts == 2
    assert delays == [1, 1]
    assert ledger.summary()["attempts_only"] == 2
    assert ledger.summary()["unreconciled_attempts"] == 0


def test_ledger_migrates_and_preserves_metered_charges(tmp_path):
    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE attempts(id INTEGER PRIMARY KEY, run TEXT, reserved REAL,
                                  charged REAL, status TEXT);
            CREATE TABLE limits(usd REAL, attempts INTEGER);
            INSERT INTO limits VALUES (1,5);
            INSERT INTO attempts VALUES (1,'old',0.6,0.4,'ok');
            INSERT INTO attempts VALUES (2,'unknown',0.2,NULL,'reserved');
        """)
    ledger = Ledger(path, 1, 5)
    interrupted = ledger.reserve("included-interrupted", 0, budget_mode="attempts_only")
    assert interrupted == 3
    settled = ledger.reserve("included", 0, budget_mode="attempts_only")
    ledger.settle(settled, None, "ok")
    reopened = Ledger(path, 1, 5)
    assert reopened.summary()["accounted_usd"] == pytest.approx(0.6)
    assert reopened.summary()["unreconciled_attempts"] == 1
    assert reopened.summary()["attempts_only"] == 2
    last = reopened.reserve("metered", 0.4)
    with pytest.raises(BudgetExceeded):
        reopened.reserve("included-after-cap", 0, budget_mode="attempts_only")
    with pytest.raises(BudgetExceeded):
        reopened.settle(last, 0.5, "ok")
    assert reopened.summary()["accounted_usd"] == pytest.approx(1.1)


def test_dollar_limit_does_not_block_included_usage(tmp_path):
    ledger = Ledger(tmp_path / "ledger", 1, 3)
    paid = ledger.reserve("paid", 1)
    with pytest.raises(BudgetExceeded):
        ledger.settle(paid, 1.2, "ok")
    ledger.reserve("included", 0, budget_mode="attempts_only")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("paid", 0.1)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("implicit-zero", 0)
    with pytest.raises(ValueError):
        Ledger(ledger.path, 2, 3)


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "argo:unknown"},
        {"model": "argo:claude-opus-4.8"},
        {"endpoint": "http://localhost:8000/v1/chat/completions"},
        {"endpoint": "https://user:secret@example.test/v1"},
        {"endpoint": "https://example.test/v1?secret=key"},
        {"api_key_env": "OPENAI_API_KEY"},
        {"input_usd_per_million": 0},
    ],
)
def test_invalid_argo_config(changes):
    with pytest.raises(ValueError):
        argo_config(**changes)


def test_identity_and_offline_validation(config):
    assert model_identity("codex:gpt-4.1-mini") == model_identity("gpt41mini")
    values = config.model_dump()
    values.update(offline=False, models=[argo_config().model_dump()])
    values["curator"] = {
        "name": "curator",
        "adapter": "codex",
        "model": "codex:gpt-4.1-mini",
        "reasoning_effort": "medium",
        "budget_mode": "attempts_only",
    }
    with pytest.raises(ValueError, match="differ"):
        Config.model_validate(values)
    values["curator"] = None
    values["offline"] = True
    with pytest.raises(ValueError, match="Offline"):
        Config.model_validate(values)


def test_legacy_config_semantics_unchanged(config):
    spec = config.models[0].spec()
    assert "budget_mode" not in spec and "argo_user_env" not in spec
    assert "reasoning_effort" not in spec
    assert "codex_prompt_contracts" not in config.semantic_config()
    assert ModelConfig.model_validate(spec).spec() == spec


def test_reasoning_effort_rejected_for_non_codex():
    with pytest.raises(ValueError, match="only by Codex"):
        argo_config(reasoning_effort="medium")


def test_argo_rejects_known_unsupported_temperature():
    with pytest.raises(ValueError, match="temperature"):
        argo_config(model="argo:gpt-5.6-sol", temperature=0)


def test_generic_chat_retains_schema_and_credentials(config, monkeypatch):
    monkeypatch.setenv("CUSTOM_LLM_KEY", "custom-key")
    monkeypatch.setenv("ARGO_USER", "must-not-be-injected")
    model = ModelConfig(
        name="generic",
        adapter="chat",
        model="custom-model",
        endpoint="https://example.test/v1/chat/completions",
        api_key_env="CUSTOM_LLM_KEY",
        budget_mode="attempts_only",
    )
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=chat_reply())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = HTTPModel(
            config, model, Ledger(config.root / "ledger", 1, 2), "test", client=client
        )
        assert adapter.predict(Request(text="Atom?"), config.policy).status == "ok"
    wire = json.loads(calls[0].content)
    assert wire["model"] == "custom-model" and "user" not in wire
    assert wire["response_format"]["json_schema"]["strict"] is True
    assert calls[0].headers["Authorization"] == "Bearer custom-key"


def test_argo_native_refusal_retains_usage(config, monkeypatch):
    monkeypatch.setenv("ARGO_USER", "user")
    body = chat_reply(choices=[{"finish_reason": "content_filter", "message": {"content": None}}])
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    ) as client:
        adapter = HTTPModel(
            config,
            argo_config(),
            Ledger(config.root / "ledger", 1, 2),
            "test",
            client=client,
        )
        result = adapter.predict(Request(text="Atom?"), config.policy)
    assert result.status == "abstain" and result.decision is None and result.input_tokens == 12


def test_examples_load_without_credentials():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("argo", "codex", "pilot", "offline"):
        config = load_config(root / "configs" / f"{name}.toml")
        assert config.models
