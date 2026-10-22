import json

import httpx
import pytest

from chemgraphsafe.adapters import HTTPModel, parse_chat, parse_decisions
from chemgraphsafe.budget import BudgetExceeded, Ledger
from chemgraphsafe.config import ModelConfig
from chemgraphsafe.models import Prediction, Request


def model_config():
    return ModelConfig(
        name="jev",
        adapter="decisions",
        model="typesafe/jev-1.14",
        endpoint="https://openrouter.ai/api/alpha/decisions",
        api_key_env="TEST_KEY",
        input_usd_per_million=0.042,
        output_usd_per_million=0,
        reservation_usd=0.01,
        pricing_checked="2026-10-19",
    )


def reply():
    return {
        "model": "typesafe/jev-1.14-20261014",
        "answers": {
            "triage": {
                "type": "choice",
                "choice": "ALLOW",
                "probabilities": {"ALLOW": 0.8, "REVIEW": 0.15, "BLOCK": 0.05},
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 20, "cost": 0.0000042},
    }


def test_openrouter_exact_wire_shape_and_no_reference_leakage(config, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret-test-value")
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=reply())

    ledger = Ledger(config.root / "ledger.sqlite", 1, 10)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        adapter = HTTPModel(config, model_config(), ledger, "test", client=client)
        result = adapter.predict(Request(text="Explain a chemical bond."), config.policy)
    assert result.status == "ok"
    assert result.returned_model == "typesafe/jev-1.14-20261014"
    assert str(requests[0].url) == "https://openrouter.ai/api/alpha/decisions"
    body = json.loads(requests[0].content)
    assert set(body) == {"model", "state", "questions"}
    assert body["state"] == {"text": "Explain a chemical bond.", "context": ""}
    assert set(body["questions"]["triage"]["criteria"]) == {"ALLOW", "REVIEW", "BLOCK"}
    for forbidden in (
        "source_label",
        "source_id",
        "source_split",
        "example_id",
        "scihazard",
        "reasoning",
    ):
        assert forbidden not in requests[0].content.decode()
        assert forbidden not in str(requests[0].headers)
    assert ledger.summary()["accounted_usd"] == pytest.approx(0.0000042)


@pytest.mark.parametrize(
    "finish,message",
    [
        ("content_filter", {"content": None}),
        ("stop", {"refusal": "Declined", "content": None}),
    ],
)
def test_documented_refusal_shapes_are_abstentions(finish, message):
    # Constructed from documented response shapes, not captured from live providers.
    p = parse_chat({"model": "test", "choices": [{"finish_reason": finish, "message": message}]})
    assert p.status == "abstain" and p.decision is None


def test_unstructured_refusal_is_not_a_policy_review():
    with pytest.raises(ValueError):
        parse_chat({"choices": [{"finish_reason": "stop", "message": {"content": "I refuse."}}]})


def test_probability_validation_and_authoritative_choice():
    b = reply()
    b["answers"]["triage"]["probabilities"] = {
        "ALLOW": 0.3333,
        "REVIEW": 0.3333,
        "BLOCK": 0.3333,
    }
    assert parse_decisions(b).status == "ok"
    b["answers"]["triage"]["probabilities"] = {
        "ALLOW": 0.2,
        "REVIEW": 0.7,
        "BLOCK": 0.1,
    }
    p = parse_decisions(b)
    assert p.decision == "ALLOW" and p.choice_argmax_mismatch
    for distribution in (
        {"ALLOW": 1},
        {"ALLOW": float("nan"), "REVIEW": 0, "BLOCK": 0},
        {"ALLOW": 0.1, "REVIEW": 0.1, "BLOCK": 0.1},
    ):
        with pytest.raises(ValueError):
            Prediction(
                status="ok",
                decision="ALLOW",
                probabilities=distribution,
                probability_origin="native",
            )


def test_retries_charge_shared_ledger_and_honor_retry_after(config, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "test")
    calls, delays = [], []

    def handle(request):
        calls.append(request)
        return (
            httpx.Response(429, headers={"Retry-After": "2"})
            if len(calls) == 1
            else httpx.Response(200, json=reply())
        )

    ledger = Ledger(config.root / "budget.sqlite", 1, 3)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        adapter = HTTPModel(
            config, model_config(), ledger, "v1", client=client, sleep=delays.append
        )
        p = adapter.predict(Request(text="Molarity definition?"), config.policy)
    assert p.status == "ok" and p.attempts == 2
    assert delays == [2]
    reopened = Ledger(config.root / "budget.sqlite", 1, 3)
    reopened.reserve("different-run-v2", 0.01)
    with pytest.raises(BudgetExceeded):
        reopened.reserve("different-run-v3", 0.01)
    assert reopened.summary()["unreconciled_attempts"] == 2


def test_long_retry_after_and_budget_exhaustion(config, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "test")
    ledger = Ledger(config.root / "budget.sqlite", 1, 1)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(429, headers={"Retry-After": "3600"})
        )
    ) as client:
        a = HTTPModel(
            config,
            model_config(),
            ledger,
            "run",
            client=client,
            sleep=lambda _: pytest.fail("Must not wait a long Retry-After"),
        )
        p = a.predict(Request(text="Atom definition?"), config.policy)
        assert p.error_code == "rate_limit_deferred"
        assert (
            a.predict(Request(text="Atom definition?"), config.policy).error_code
            == "budget_exhausted"
        )


def test_missing_credentials_and_prices_fail_before_calls(config, monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    ledger = Ledger(config.root / "budget.sqlite", 1, 10)
    with pytest.raises(ValueError, match="credential"):
        HTTPModel(config, model_config(), ledger, "run")
    monkeypatch.setenv("TEST_KEY", "test")
    m = model_config().model_copy(update={"pricing_checked": None})
    with pytest.raises(ValueError, match="pricing"):
        HTTPModel(config, m, ledger, "run")
    assert ledger.summary()["attempts"] == 0
