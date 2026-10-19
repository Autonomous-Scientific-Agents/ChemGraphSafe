import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chemgraphsafe.adapters import build_model
from chemgraphsafe.budget import Ledger
from chemgraphsafe.codex import CodexModel, consume
from chemgraphsafe.config import ModelConfig
from chemgraphsafe.models import Request


def codex_config(**kwargs):
    return ModelConfig(
        **{
            "name": "codex",
            "adapter": "codex",
            "model": "codex:test-model",
            "budget_mode": "attempts_only",
            **kwargs,
        }
    )


def adapter(config, **kwargs):
    return CodexModel(
        config,
        codex_config(),
        Ledger(config.root / "ledger", 1, 20),
        "test",
        sleep=lambda _: None,
        **kwargs,
    )


@pytest.mark.parametrize("curator", [False, True])
def test_codex_isolated_classification_and_usage(config, fake_codex, curator):
    if curator:
        fake_codex.content = '{"decision":"ALLOW","reason":"legitimate_use"}'
    model = adapter(config, curator=curator)
    assert not fake_codex.turns and model.ledger.summary()["attempts"] == 0
    result = model.predict(Request(text="Explain a chemical bond."), config.policy)
    assert result.status == "ok" and result.decision == "ALLOW"
    assert result.reason == ("legitimate_use" if curator else None)
    assert result.input_tokens == 12 and result.output_tokens == 4
    assert result.attempts == 1 and result.estimated_cost_usd is None
    assert result.provider_cost_usd is None and result.returned_model is None
    assert fake_codex.closed == 2
    thread = fake_codex.threads[0]
    assert thread["ephemeral"] and thread["model"] == "test-model"
    assert thread["model_provider"] == "openai" and thread["sandbox"] == "read-only"
    assert thread["approval_mode"] == "deny"
    assert thread["config"] == {"mcp_servers": {"inherited.server": {"enabled": False}}}
    assert not Path(thread["cwd"]).exists()
    sdk_config = fake_codex.configs[-1]
    assert sdk_config.env == {"OPENAI_API_KEY": "", "CODEX_API_KEY": ""}
    assert "features.shell_tool=false" in sdk_config.config_overrides
    assert 'web_search="disabled"' in sdk_config.config_overrides
    prompt, args = fake_codex.turns[0]
    assert args["output_schema"]["required"] == (
        ["decision", "reason"] if curator else ["decision"]
    )
    assert json.loads(json.loads(prompt)[1]["content"]) == {
        "untrusted_request": {"text": "Explain a chemical bond.", "context": ""}
    }
    for field in ("source_label", "source_id", "example_id", "scihazard"):
        assert field not in prompt


@pytest.mark.parametrize("account", [None, "apiKey", "unknown"])
def test_codex_login_rejected_before_inference(config, fake_codex, account):
    fake_codex.account = account
    with pytest.raises(ValueError, match="preflight"):
        adapter(config)
    assert not fake_codex.turns and fake_codex.closed == 1
    assert Ledger(config.root / "ledger", 1, 20).summary()["attempts"] == 0


def test_codex_missing_sdk_is_actionable(monkeypatch):
    from chemgraphsafe import codex

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(codex.importlib, "import_module", missing)
    with pytest.raises(ValueError, match="uv sync --extra codex"):
        codex.load_sdk()


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "I refuse.",
        '{"decision":"INVALID"}',
        '{"decision":"ALLOW","reason":"legitimate_use"}',
    ],
)
def test_codex_invalid_output_keeps_usage(config, fake_codex, content):
    fake_codex.content = content
    result = adapter(config).predict(Request(text="Atom?"), config.policy)
    assert result.error_code == "schema_error" and result.attempts == 1
    assert result.input_tokens == 12 and result.decision is None


def test_codex_timeout_interrupts_closes_and_charges_retries(config, fake_codex):
    config.runner.timeout_seconds = 0.01
    fake_codex.hang = True
    model = adapter(config)
    result = model.predict(Request(text="Atom?"), config.policy)
    assert result.error_code == "codex_timeout" and result.attempts == 3
    assert result.input_tokens == 36 and result.output_tokens == 12
    assert fake_codex.interrupted == 3 and fake_codex.closed == 4
    assert model.ledger.summary()["attempts_only"] == 3
    assert model.ledger.summary()["unreconciled_attempts"] == 0


def test_codex_startup_timeout_is_bounded(config, fake_codex):
    config.runner.timeout_seconds = 0.01
    fake_codex.startup_hang = True
    with pytest.raises(ValueError, match="codex_timeout"):
        adapter(config)
    assert fake_codex.closed == 1 and not fake_codex.turns


@pytest.mark.parametrize(
    "status,error,expected",
    [
        ("interrupted", None, "codex_interrupted"),
        ("failed", "unauthorized", "codex_authentication"),
        ("failed", "usageLimitExceeded", "codex_quota_exhausted"),
        ("failed", "cyberPolicy", "codex_error"),
    ],
)
def test_codex_terminal_failure_is_not_a_decision(config, fake_codex, status, error, expected):
    fake_codex.outcome, fake_codex.failure = status, error
    result = adapter(config).predict(Request(text="Atom?"), config.policy)
    assert result.error_code == expected and result.attempts == 1 and result.decision is None
    assert result.input_tokens == 12


def test_codex_transient_error_retries(config, fake_codex):
    fake_codex.exception = fake_codex.sdk.TransportClosedError("secret provider detail")
    result = adapter(config).predict(Request(text="Atom?"), config.policy)
    assert result.error_code == "codex_network_error" and result.attempts == 3
    assert "secret" not in result.model_dump_json()


def test_codex_tool_events_are_rejected(config, fake_codex):
    fake_codex.events = [
        SimpleNamespace(
            method="item/started",
            payload={
                "turnId": "turn-1",
                "item": {"type": "commandExecution", "command": "never execute"},
            },
        )
    ]
    result = adapter(config).predict(Request(text="Atom?"), config.policy)
    assert result.error_code == "codex_tool_attempt"
    assert fake_codex.interrupted == 1


def test_codex_input_and_budget_limits_prevent_turns(config, fake_codex):
    model = build_model(
        config, codex_config(max_input_bytes=1), Ledger(config.root / "ledger", 1, 1), "test"
    )
    assert model.predict(Request(text="Atom?"), config.policy).error_code == "input_too_large"
    assert not fake_codex.turns
    model.model.max_input_bytes = 32000
    assert model.predict(Request(text="Atom?"), config.policy).status == "ok"
    assert model.predict(Request(text="Atom?"), config.policy).error_code == "budget_exhausted"
    assert len(fake_codex.turns) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "codex:"},
        {"model": "unprefixed"},
        {"temperature": 0},
        {"seed": 42},
        {"api_key_env": "OPENAI_API_KEY"},
        {"endpoint": "https://example.test"},
        {"budget_mode": "usd"},
    ],
)
def test_codex_invalid_configuration(changes):
    with pytest.raises(ValueError):
        codex_config(**changes)


def test_real_pinned_sdk_notification_contract():
    sdk = pytest.importorskip("openai_codex.generated.v2_all")
    common = {}
    counts = {
        "inputTokens": 20,
        "outputTokens": 5,
        "totalTokens": 25,
        "cachedInputTokens": 2,
        "reasoningOutputTokens": 1,
    }
    events = [
        (
            "thread/tokenUsage/updated",
            sdk.ThreadTokenUsageUpdatedNotification.model_validate(
                {
                    "threadId": "thread",
                    "turnId": "turn",
                    "tokenUsage": {"total": counts, "last": counts},
                }
            ),
        ),
        (
            "item/completed",
            sdk.ItemCompletedNotification.model_validate(
                {
                    "threadId": "thread",
                    "turnId": "turn",
                    "completedAtMs": 1,
                    "item": {
                        "id": "item",
                        "type": "agentMessage",
                        "phase": "final_answer",
                        "text": '{"decision":"ALLOW"}',
                    },
                }
            ),
        ),
        (
            "turn/completed",
            sdk.TurnCompletedNotification.model_validate(
                {
                    "threadId": "thread",
                    "turn": {"id": "turn", "items": [], "status": "completed", "itemsView": "full"},
                }
            ),
        ),
    ]

    async def stream():
        for method, payload in events:
            yield SimpleNamespace(method=method, payload=payload)

    result, error = asyncio.run(consume(SimpleNamespace(id="turn", stream=stream), common))
    assert json.loads(result) == {"decision": "ALLOW"} and error is None
    assert common == {"input_tokens": 20, "output_tokens": 5}


def test_real_sdk_blocking_reader_is_released_before_stream_unregister(config, monkeypatch):
    """Exercise the SDK's actual thread-backed queue without starting a runtime."""
    sdk = pytest.importorskip("openai_codex")
    from openai_codex.async_client import AsyncCodexClient

    from chemgraphsafe import codex

    registered_at_close = []

    class Client:
        def __init__(self, config):
            self._client = AsyncCodexClient(config=config)
            self.waiting_queue = None
            original_next = self._client._sync.next_turn_notification

            def next_notification(turn_id):
                self.waiting_queue = self._client._sync._router._turn_notifications[turn_id]
                return original_next(turn_id)

            self._client._sync.next_turn_notification = next_notification

            async def request(*args, **kwargs):
                return {"config": {}}

            self._client.request = request

        async def _ensure_initialized(self):
            pass

        async def __aenter__(self):
            return self

        async def account(self):
            return {"account": {"type": "chatgpt"}}

        async def thread_start(self, **kwargs):
            async def turn(*args, **kwargs):
                return sdk.AsyncTurnHandle(self, "thread", "turn")

            return SimpleNamespace(turn=turn)

        async def close(self):
            router = self._client._sync._router
            if self.waiting_queue is not None:
                registered_at_close.append("turn" in router._turn_notifications)
                # Release even an orphaned queue, so a regression fails rather than hangs pytest.
                self.waiting_queue.put(sdk.TransportClosedError("closed"))
            router.fail_all(sdk.TransportClosedError("closed"))

    monkeypatch.setattr(sdk, "AsyncCodex", Client)
    monkeypatch.setattr(codex, "load_sdk", lambda: sdk)
    config.runner.timeout_seconds = 0.05
    config.runner.retries = 0
    model = adapter(config)
    result = model.predict(Request(text="Explain a chemical bond."), config.policy)
    assert result.error_code == "codex_timeout"
    assert registered_at_close == [True]
