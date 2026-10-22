import json

import httpx
import pytest

from chemgraphsafe import adapters, data, runner
from chemgraphsafe.adapters import LayaModel, build_model, parse_laya
from chemgraphsafe.budget import Ledger
from chemgraphsafe.config import ModelConfig
from chemgraphsafe.models import Prediction, Request, policy_labels
from chemgraphsafe.rights import HostedDestination
from chemgraphsafe.storage import read_json, read_rows, write_json

REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


def laya_config(**changes):
    return ModelConfig(
        **{
            "name": "laya",
            "adapter": "laya",
            "model": "english",
            "endpoint": "http://127.0.0.1:8000/v1/systemone",
            "identity_endpoint": "http://127.0.0.1:8000/health",
            "expected_revision": REVISION,
            "budget_mode": "attempts_only",
            **changes,
        }
    )


def health(*, model="english", revision=REVISION, device="cpu"):
    return {
        "status": "ok",
        "loaded": [model],
        "revisions": {model: revision},
        "device": device,
        "device_is_preference": False,
        "checkpoint_devices": {model: device},
        "cpu_fallbacks": {model: {"count": 0, "last_reason": None}},
    }


def laya_reply(*, answer=None, routing_model="english"):
    return {
        # Upstream documents this as the decision-head name, not the served checkpoint.
        "model": "laya-rl-agent",
        "answers": {
            "triage": answer
            or {
                "type": "choice",
                "choice": "ALLOW",
                "probabilities": {"ALLOW": 0.8, "REVIEW": 0.15, "BLOCK": 0.05},
                "confidence": 0.62,
                "answer_confidence": 0.8,
            }
        },
        "usage": {
            "input_tokens": 18,
            "output_tokens": 0,
            "state_tokens": 12,
            "state_tokens_dropped": 0,
            "truncated": False,
            "truncated_questions": [],
            # A local server has no provider charge. Ignore this even if a proxy adds it.
            "cost": 99,
        },
        "routing": {
            "model": routing_model,
            "repo": "convaiinnovations/laya",
            "reason": "model argument",
            "detection": None,
            "workflow": None,
        },
    }


def truncated_laya_reply():
    body = laya_reply()
    body["usage"] |= {
        "state_tokens_dropped": 4,
        "truncated": True,
        "truncated_questions": ["triage"],
    }
    return body


def null_distribution_laya_reply():
    body = laya_reply()
    body["answers"]["triage"]["probabilities"] = None
    return body


@pytest.mark.parametrize(
    "changes",
    [
        {
            "endpoint": "http://localhost:8000/v1/systemone",
            "identity_endpoint": "http://localhost:8000/health",
        },
        {
            "endpoint": "http://192.0.2.1:8000/v1/systemone",
            "identity_endpoint": "http://192.0.2.1:8000/health",
        },
        {"endpoint": "http://127.0.0.1:8000/v1/systemone?token=secret"},
        {"endpoint": "http://user:secret@127.0.0.1:8000/v1/systemone"},
        {"identity_endpoint": "http://127.0.0.1:8001/health"},
        {"expected_revision": ""},
        {"expected_revision": f" {REVISION}"},
        {"budget_mode": "usd"},
        {"api_key_env": " LAYA_API_KEY"},
        {"temperature": 0},
        {"seed": 42},
    ],
)
def test_laya_configuration_fails_closed(changes):
    with pytest.raises(ValueError):
        laya_config(**changes)


@pytest.mark.parametrize("field", ["identity_endpoint", "expected_revision"])
@pytest.mark.parametrize(
    "values",
    [
        {
            "name": "decisions",
            "adapter": "decisions",
            "model": "model",
            "endpoint": "https://example.test/decisions",
            "api_key_env": "TEST_KEY",
            "budget_mode": "attempts_only",
        },
        {
            "name": "chat",
            "adapter": "chat",
            "model": "model",
            "endpoint": "https://example.test/chat",
            "api_key_env": "TEST_KEY",
            "budget_mode": "attempts_only",
        },
        {"name": "replay", "adapter": "replay", "model": "fixture"},
        {
            "name": "argo",
            "adapter": "argo",
            "model": "argo:gpt-4.1-mini",
            "budget_mode": "attempts_only",
        },
        {
            "name": "codex",
            "adapter": "codex",
            "model": "codex:test",
            "reasoning_effort": "medium",
            "budget_mode": "attempts_only",
        },
    ],
)
def test_laya_identity_fields_are_rejected_by_every_other_adapter(values, field):
    configured = {
        **values,
        field: "https://example.test/health" if field == "identity_endpoint" else REVISION,
    }
    with pytest.raises(ValueError, match="only by Laya"):
        ModelConfig(**configured)


def test_laya_endpoint_location_classification():
    assert laya_config().processing_location == "local"
    assert (
        laya_config(
            endpoint="http://[::1]:8000/v1/systemone",
            identity_endpoint="http://[::1]:8000/health",
        ).processing_location
        == "local"
    )
    assert (
        laya_config(
            endpoint="https://laya.example.test/v1/systemone",
            identity_endpoint="https://laya.example.test/health",
        ).processing_location
        == "hosted"
    )


def test_hosted_laya_is_a_valid_approval_destination():
    destination = HostedDestination(
        adapter="laya",
        model="english",
        endpoint="https://laya.example.test/v1/systemone",
    )
    assert destination.adapter == "laya"


@pytest.mark.parametrize("authenticated", [False, True])
def test_laya_preflight_wire_shape_auth_identity_and_no_leakage(config, monkeypatch, authenticated):
    requests = []
    changes = {}
    if authenticated:
        monkeypatch.setenv("TEST_LAYA_KEY", "local-secret")
        changes["api_key_env"] = "TEST_LAYA_KEY"

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=health())
        return httpx.Response(200, json=laya_reply())

    ledger = Ledger(config.root / "laya-ledger.sqlite", 1, 10)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = build_model(config, laya_config(**changes), ledger, "laya-run", client=client)
        assert isinstance(adapter, LayaModel)
        assert ledger.summary()["attempts"] == 0
        result = adapter.predict(
            Request(text="Explain a chemical bond.", context="Introductory class"),
            config.policy,
        )

    assert [request.method for request in requests] == ["GET", "POST"]
    assert str(requests[0].url) == "http://127.0.0.1:8000/health"
    assert str(requests[1].url) == "http://127.0.0.1:8000/v1/systemone"
    expected_auth = "Bearer local-secret" if authenticated else None
    assert requests[0].headers.get("Authorization") == expected_auth
    assert requests[1].headers.get("Authorization") == expected_auth
    body = json.loads(requests[1].content)
    assert set(body) == {"model", "state", "questions"}
    assert body["model"] == "english"
    assert body["state"] == {
        "text": "Explain a chemical bond.",
        "context": "Introductory class",
    }
    assert list(body["questions"]["triage"]["criteria"]) == ["ALLOW", "REVIEW", "BLOCK"]
    assert "min_confidence" not in body
    serialized = requests[1].content.decode()
    for forbidden in (
        "source_label",
        "source_id",
        "source_split",
        "example_id",
        "scihazard",
        "reasoning",
        "local-secret",
    ):
        assert forbidden not in serialized

    assert adapter.processing_location == "local"
    assert adapter.runtime_provenance == {
        "model": "english",
        "revision": REVISION,
        "device": "cpu",
    }
    assert result.status == "ok" and result.decision == "ALLOW"
    assert result.returned_model == f"english@{REVISION}"
    assert result.probability_origin == "native"
    assert result.probabilities == {"ALLOW": 0.8, "REVIEW": 0.15, "BLOCK": 0.05}
    assert result.input_tokens == 18 and result.output_tokens == 0
    assert result.provider_cost_usd is None and result.estimated_cost_usd is None
    assert result.attempts == 1
    assert ledger.summary()["attempts_only"] == 1
    assert ledger.summary()["accounted_usd"] == 0


@pytest.mark.parametrize(
    "answer,expected_status,expected_decision",
    [
        (
            {
                "type": "choice",
                "choice": "BLOCK",
                "probabilities": {"ALLOW": 0.2, "BLOCK": 0.8},
            },
            "ok",
            "BLOCK",
        ),
        (
            {
                "type": "choice",
                "choice": "REVIEW",
                "probabilities": {"ALLOW": 0.1, "REVIEW": 0.8, "BLOCK": 0.1},
                "abstention": "abstained",
                "abstention_threshold": 0.7,
                "low_confidence": True,
            },
            "error",
            None,
        ),
    ],
)
def test_laya_binary_policy_contract(config, answer, expected_status, expected_decision):
    config.policy_file = "policies/triage-binary-v0.1.json"
    requests = []

    def respond(request):
        requests.append(request)
        return (
            httpx.Response(200, json=health())
            if request.method == "GET"
            else httpx.Response(200, json=laya_reply(answer=answer))
        )

    ledger = Ledger(config.root / "binary-laya-ledger.sqlite", 1, 10)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = LayaModel(config, laya_config(), ledger, "binary-laya", client=client)
        result = adapter.predict(Request(text="Define an atom."), config.policy)

    assert [request.method for request in requests] == ["GET", "POST"]
    body = json.loads(requests[1].content)
    assert list(body["questions"]["triage"]["criteria"]) == ["ALLOW", "BLOCK"]
    assert result.status == expected_status and result.decision == expected_decision
    if expected_status == "error":
        assert result.error_code == "schema_error"
    assert result.returned_model == f"english@{REVISION}"
    assert result.input_tokens == 18 and result.output_tokens == 0
    assert result.attempts == 1 and ledger.summary()["attempts_only"] == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503),
        httpx.Response(200, text="not-json"),
        httpx.Response(200, json={"status": "starting"}),
        httpx.Response(200, json={"status": "ok"}),
        httpx.Response(200, json=health(model="multilingual")),
        httpx.Response(200, json=health(revision="different")),
        httpx.Response(200, json={**health(), "checkpoint_devices": {}, "device": None}),
    ],
)
def test_laya_identity_preflight_fails_before_inference_or_budget(config, response):
    requests = []

    def respond(request):
        requests.append(request)
        return response

    ledger = Ledger(config.root / "preflight-ledger.sqlite", 1, 10)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ValueError, match="preflight"):
            LayaModel(config, laya_config(), ledger, "run", client=client)
    assert len(requests) == 1 and requests[0].method == "GET"
    assert ledger.summary()["attempts"] == 0


def test_laya_identity_timeout_fails_before_inference_or_budget(config):
    def timeout(request):
        raise httpx.ReadTimeout("timed out", request=request)

    ledger = Ledger(config.root / "timeout-preflight-ledger.sqlite", 1, 10)
    with httpx.Client(transport=httpx.MockTransport(timeout)) as client:
        with pytest.raises(ValueError, match="preflight"):
            LayaModel(config, laya_config(), ledger, "run", client=client)
    assert ledger.summary()["attempts"] == 0


def test_laya_configured_auth_requires_its_named_environment_variable(config, monkeypatch):
    monkeypatch.delenv("TEST_LAYA_KEY", raising=False)
    ledger = Ledger(config.root / "missing-auth-ledger.sqlite", 1, 10)
    with pytest.raises(ValueError, match="TEST_LAYA_KEY"):
        LayaModel(
            config,
            laya_config(api_key_env="TEST_LAYA_KEY"),
            ledger,
            "run",
        )
    assert ledger.summary()["attempts"] == 0


def test_owned_local_laya_client_ignores_environment_proxies(config, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://198.51.100.10:3128")
    monkeypatch.setenv("HTTPS_PROXY", "")
    monkeypatch.setenv("ALL_PROXY", "")
    monkeypatch.setenv("NO_PROXY", "")
    original = httpx.Client
    client_options, requests = [], []

    def respond(request):
        requests.append(request)
        return (
            httpx.Response(200, json=health())
            if request.method == "GET"
            else httpx.Response(200, json=laya_reply())
        )

    def client(**kwargs):
        client_options.append(kwargs)
        return original(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(adapters.httpx, "Client", client)
    ledger = Ledger(config.root / "proxy-ledger.sqlite", 1, 10)
    adapter = LayaModel(config, laya_config(), ledger, "run")
    try:
        result = adapter.predict(Request(text="Define an atom."), config.policy)
    finally:
        adapter.close()

    assert client_options[0]["trust_env"] is False
    assert [request.method for request in requests] == ["GET", "POST"]
    assert result.status == "ok"


@pytest.mark.parametrize(
    "gate,status,error_code",
    [
        ({}, "ok", None),
        (
            {"abstention": "passed", "abstention_threshold": 0.7},
            "ok",
            None,
        ),
        (
            {
                "abstention": "passed",
                "abstention_threshold": 0.7,
                "low_confidence": False,
            },
            "ok",
            None,
        ),
        (
            {
                "abstention": "abstained",
                "abstention_threshold": 0.7,
                "low_confidence": True,
            },
            "abstain",
            "provider_abstention",
        ),
        (
            {"abstention": "unevaluated", "abstention_threshold": 0.7},
            "error",
            "abstention_unevaluated",
        ),
    ],
)
def test_laya_gate_states_are_not_policy_review(gate, status, error_code):
    answer = laya_reply()["answers"]["triage"] | gate
    result = parse_laya(laya_reply(answer=answer), model="english", revision=REVISION)
    assert result.status == status and result.error_code == error_code
    assert (result.decision, result.probabilities) == (
        ("ALLOW", {"ALLOW": 0.8, "REVIEW": 0.15, "BLOCK": 0.05}) if status == "ok" else (None, None)
    )
    assert result.decision != "REVIEW"


@pytest.mark.parametrize(
    "gate",
    [
        {"abstention_threshold": 0.7},
        {"low_confidence": True},
        {"abstention": "unknown", "abstention_threshold": 0.7},
        {"abstention": "passed"},
        {"abstention": "passed", "abstention_threshold": 0.7, "low_confidence": True},
        {"abstention": "abstained", "abstention_threshold": 0.7},
        {
            "abstention": "unevaluated",
            "abstention_threshold": 0.7,
            "low_confidence": True,
        },
        {"abstention": "passed", "abstention_threshold": True},
        {"abstention": "passed", "abstention_threshold": float("nan")},
        {"abstention": "passed", "abstention_threshold": 1.1},
    ],
)
def test_laya_malformed_gate_metadata_is_rejected(gate):
    answer = laya_reply()["answers"]["triage"] | gate
    with pytest.raises(ValueError):
        parse_laya(laya_reply(answer=answer), model="english", revision=REVISION)


def test_laya_reported_truncation_is_not_a_successful_prediction():
    result = parse_laya(truncated_laya_reply(), model="english", revision=REVISION)
    assert result.status == "error" and result.error_code == "input_truncated"
    assert result.decision is None and result.probabilities is None
    assert result.probability_origin is None
    assert result.input_tokens == 18 and result.output_tokens == 0
    assert result.returned_model == f"english@{REVISION}"


def malformed_laya_usage_bodies():
    valid = laya_reply()
    usage = valid["usage"]
    yield {key: value for key, value in valid.items() if key != "usage"}
    yield {**valid, "usage": None}
    yield {**valid, "usage": []}
    for field in ("state_tokens", "state_tokens_dropped", "truncated", "truncated_questions"):
        yield {**valid, "usage": {key: value for key, value in usage.items() if key != field}}
    for state_tokens in (True, -1, 1.5):
        yield {**valid, "usage": {**usage, "state_tokens": state_tokens}}
    for dropped in (True, -1, 1.5, 13):
        yield {**valid, "usage": {**usage, "state_tokens_dropped": dropped}}
    for truncated in (None, 0, 1, "false"):
        yield {**valid, "usage": {**usage, "truncated": truncated}}
    yield {**valid, "usage": {**usage, "state_tokens_dropped": 1, "truncated": False}}
    yield {**valid, "usage": {**usage, "truncated": True}}
    for questions in (None, "triage", [True], [""], ["other"], ["triage", "triage"]):
        yield {**valid, "usage": {**usage, "truncated_questions": questions}}
    yield {**valid, "usage": {**usage, "truncated_questions": ["triage"]}}


@pytest.mark.parametrize("body", list(malformed_laya_usage_bodies()))
def test_laya_malformed_or_inconsistent_truncation_metadata_is_rejected(body):
    with pytest.raises(ValueError, match="usage|truncation"):
        parse_laya(body, model="english", revision=REVISION)


def malformed_laya_bodies():
    valid = laya_reply()
    answer = valid["answers"]["triage"]
    yield {}
    yield {**valid, "routing": None}
    yield {**valid, "routing": {"model": "multilingual"}}
    yield {**valid, "answers": None}
    yield {**valid, "answers": {"triage": None}}
    yield {**valid, "answers": {"triage": {**answer, "type": "score"}}}
    yield {**valid, "answers": {"triage": {k: v for k, v in answer.items() if k != "choice"}}}
    yield {
        **valid,
        "answers": {"triage": {k: v for k, v in answer.items() if k != "probabilities"}},
    }
    for probabilities in (None, [], "not-a-distribution"):
        yield {**valid, "answers": {"triage": {**answer, "probabilities": probabilities}}}
    yield {**valid, "answers": {"triage": {**answer, "choice": "MAYBE"}}}
    yield {
        **valid,
        "answers": {"triage": {**answer, "probabilities": {"ALLOW": 1.0}}},
    }
    yield {
        **valid,
        "answers": {
            "triage": {
                **answer,
                "probabilities": {"ALLOW": 0.1, "REVIEW": 0.1, "BLOCK": 0.1},
            }
        },
    }


@pytest.mark.parametrize("body", list(malformed_laya_bodies()))
def test_laya_malformed_identity_answer_or_distribution_is_rejected(body):
    with pytest.raises(ValueError):
        parse_laya(body, model="english", revision=REVISION)


@pytest.mark.parametrize(
    "body,error_code",
    [
        (truncated_laya_reply(), "input_truncated"),
        (null_distribution_laya_reply(), "schema_error"),
        (
            {
                **laya_reply(),
                "usage": {
                    **laya_reply()["usage"],
                    "state_tokens_dropped": 1,
                    "truncated": False,
                },
            },
            "schema_error",
        ),
    ],
)
def test_laya_incomplete_or_malformed_input_contract_is_accounted(config, body, error_code):
    requests = []

    def respond(request):
        requests.append(request)
        return (
            httpx.Response(200, json=health())
            if request.method == "GET"
            else httpx.Response(200, json=body)
        )

    ledger = Ledger(config.root / "incomplete-input-ledger.sqlite", 1, 10)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = LayaModel(config, laya_config(), ledger, "run", client=client)
        result = adapter.predict(Request(text="Define an atom."), config.policy)

    assert [request.method for request in requests] == ["GET", "POST"]
    assert result.status == "error" and result.error_code == error_code
    assert result.decision is None and result.probabilities is None
    assert result.input_tokens == 18 and result.output_tokens == 0
    assert result.returned_model == f"english@{REVISION}"
    assert result.attempts == 1 and ledger.summary()["attempts_only"] == 1


def test_laya_503_retry_and_attempts_only_accounting(config):
    posts = 0
    delays = []

    def respond(request):
        nonlocal posts
        if request.method == "GET":
            return httpx.Response(200, json=health())
        posts += 1
        if posts == 1:
            return httpx.Response(503, headers={"Retry-After": "1"})
        return httpx.Response(200, json=laya_reply())

    ledger = Ledger(config.root / "retry-ledger.sqlite", 1, 10)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = LayaModel(
            config, laya_config(), ledger, "run", client=client, sleep=delays.append
        )
        result = adapter.predict(Request(text="Define an atom."), config.policy)
    assert result.status == "ok" and result.attempts == 2
    assert posts == 2 and delays == [1]
    assert ledger.summary()["attempts_only"] == 2
    assert ledger.summary()["unreconciled_attempts"] == 0


def test_laya_timeouts_retry_without_hiding_attempts(config):
    posts = 0

    def respond(request):
        nonlocal posts
        if request.method == "GET":
            return httpx.Response(200, json=health())
        posts += 1
        raise httpx.ReadTimeout("timed out", request=request)

    ledger = Ledger(config.root / "inference-timeout-ledger.sqlite", 1, 10)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = LayaModel(
            config, laya_config(), ledger, "run", client=client, sleep=lambda _: None
        )
        result = adapter.predict(Request(text="Define an atom."), config.policy)
    assert result.status == "error" and result.error_code == "network_error"
    assert result.attempts == 3 and posts == 3
    assert ledger.summary()["attempts_only"] == 3
    assert ledger.summary()["unreconciled_attempts"] == 0


def test_laya_input_limit_prevents_budget_reservation(config):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=health())

    ledger = Ledger(config.root / "size-ledger.sqlite", 1, 10)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = LayaModel(
            config,
            laya_config(max_input_bytes=1),
            ledger,
            "run",
            client=client,
        )
        result = adapter.predict(Request(text="Define an atom."), config.policy)
    assert result.error_code == "input_too_large" and result.attempts == 0
    assert [request.method for request in calls] == ["GET"]
    assert ledger.summary()["attempts"] == 0


def test_laya_cannot_be_silently_used_as_policy_curator(config):
    ledger = Ledger(config.root / "curator-ledger.sqlite", 1, 10)
    with pytest.raises(ValueError, match="curation"):
        build_model(config, laya_config(), ledger, curator=True)
    assert ledger.summary()["attempts"] == 0


def install_mock_laya_client(monkeypatch, respond):
    original = httpx.Client
    monkeypatch.setattr(
        adapters.httpx,
        "Client",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(respond)),
    )


def make_one_fixture_local_only(
    config,
    rights_evidence="First-party fixture restricted to local processing for this test",
):
    path = config.root / "fixtures/examples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["hosted_processing_allowed"] = False
    rows[0]["rights_evidence"] = rights_evidence
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_local_laya_can_evaluate_local_only_data_and_resume_without_inference(config, monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return (
            httpx.Response(200, json=health())
            if request.method == "GET"
            else httpx.Response(200, json=laya_reply())
        )

    install_mock_laya_client(monkeypatch, respond)
    make_one_fixture_local_only(config)
    config.offline = False
    config.models = [laya_config()]
    data.prepare(config)
    data.freeze(config)

    first = runner.evaluate(config)
    assert first["complete"] and not first["mock"]
    prediction_path = config.run / "predictions.jsonl"
    before = prediction_path.read_bytes()
    assert [request.method for request in requests].count("POST") == 10
    assert runner.ledger_for(config).summary()["attempts_only"] == 10

    second = runner.evaluate(config)
    assert second == first
    assert prediction_path.read_bytes() == before
    # Resume still verifies the serving identity, but never re-runs terminal predictions.
    assert [request.method for request in requests].count("GET") == 2
    assert [request.method for request in requests].count("POST") == 10
    assert runner.ledger_for(config).summary()["attempts_only"] == 10


def test_reversed_policy_order_resumes_with_one_canonical_wire_order(config, monkeypatch):
    requests = []
    posts = 0

    def respond(request):
        nonlocal posts
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=health())
        posts += 1
        # Leave exactly one retryable row in the partial run without relying on another fix here.
        body = laya_reply(routing_model="wrong-checkpoint") if posts == 4 else laya_reply()
        return httpx.Response(200, json=body)

    install_mock_laya_client(monkeypatch, respond)
    make_one_fixture_local_only(config)
    config.offline = False
    config.models = [laya_config()]
    data.prepare(config)
    data.freeze(config)

    semantic_hash = config.semantic_hash()
    first = runner.evaluate(config)
    first_manifest = read_json(config.run / "manifest.json")
    assert not first["complete"] and first["outcomes"] == {"ok": 9, "error": 1}

    policy = read_json(config.path(config.policy_file))
    policy["criteria"] = dict(reversed(list(policy["criteria"].items())))
    write_json(config.path(config.policy_file), policy)
    assert config.semantic_hash() == semantic_hash

    second = runner.evaluate(config)
    second_manifest = read_json(config.run / "manifest.json")
    assert second["complete"] and second["outcomes"] == {"ok": 10}
    assert posts == 11
    assert first_manifest["run_key"] == second_manifest["run_key"]
    assert first_manifest["semantic_hash"] == second_manifest["semantic_hash"] == semantic_hash
    orders = [
        list(json.loads(request.content)["questions"]["triage"]["criteria"])
        for request in requests
        if request.method == "POST"
    ]
    assert orders == [list(policy_labels(config.policy))] * 11
    assert all(
        row["prediction"]["status"] == "ok" for row in read_rows(config.run / "predictions.jsonl")
    )


def test_local_laya_requires_rights_evidence_before_preflight_or_budget(config, monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=health())

    install_mock_laya_client(monkeypatch, respond)
    make_one_fixture_local_only(config, rights_evidence="   ")
    config.offline = False
    config.models = [laya_config()]
    data.prepare(config)
    data.freeze(config)

    with pytest.raises(ValueError, match="Rights evidence"):
        runner.evaluate(config)
    assert requests == []
    assert not config.path(config.budget.ledger).exists()
    assert not (config.run / "predictions.jsonl").exists()


def test_hosted_laya_requires_hosted_processing_rights_before_preflight_or_budget(
    config, monkeypatch
):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=health())

    install_mock_laya_client(monkeypatch, respond)
    make_one_fixture_local_only(config)
    config.offline = False
    config.models = [
        laya_config(
            endpoint="https://laya.example.test/v1/systemone",
            identity_endpoint="https://laya.example.test/health",
        )
    ]
    data.prepare(config)
    data.freeze(config)

    with pytest.raises(ValueError, match="Hosted processing"):
        runner.evaluate(config)
    assert requests == []
    assert not config.path(config.budget.ledger).exists()
    assert not (config.run / "predictions.jsonl").exists()


def test_mixed_run_submits_only_hosted_destinations_for_approval(config, monkeypatch):
    hosted = ModelConfig(
        name="jev",
        adapter="decisions",
        model="typesafe/jev-1.14",
        endpoint="https://openrouter.example.test/api/alpha/decisions",
        api_key_env="TEST_JEV_KEY",
        budget_mode="attempts_only",
    )
    config.offline = False
    config.models = [laya_config(), hosted]
    captured = []
    approval = {"evidence": "test approval snapshot"}

    def check_hosted(rows, *, config, models):
        captured.extend(models)
        return approval

    class FakeAdapter:
        runtime_provenance = None

        def predict(self, request, policy):
            return Prediction(status="ok", decision="ALLOW")

        def close(self):
            pass

    monkeypatch.setattr(runner, "check_hosted_rights", check_hosted)
    monkeypatch.setattr(runner, "build_model", lambda *args, **kwargs: FakeAdapter())

    outcome = runner.smoke(config)
    execution = read_json(config.run / "smoke/manifest.json")["execution"]
    assert outcome["complete"]
    assert [model.name for model in captured] == ["jev"]
    assert execution["hosted_processing_approval"] == approval


def test_laya_and_jev_are_independent_baselines_without_cross_feeding(config, monkeypatch):
    monkeypatch.setenv("TEST_JEV_KEY", "jev-secret")
    requests = []

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=health())
        if request.url.path == "/v1/systemone":
            return httpx.Response(200, json=laya_reply())
        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-1.14",
                "answers": {
                    "triage": {
                        "type": "choice",
                        "choice": "BLOCK",
                        "probabilities": {"ALLOW": 0.05, "REVIEW": 0.15, "BLOCK": 0.8},
                    }
                },
                "usage": {"input_tokens": 12, "output_tokens": 0},
            },
        )

    install_mock_laya_client(monkeypatch, respond)
    config.offline = False
    config.models = [
        laya_config(),
        ModelConfig(
            name="jev",
            adapter="decisions",
            model="typesafe/jev-1.14",
            endpoint="https://openrouter.example.test/api/alpha/decisions",
            api_key_env="TEST_JEV_KEY",
            budget_mode="attempts_only",
        ),
    ]

    outcome = runner.smoke(config)
    assert outcome["complete"]
    posts = [request for request in requests if request.method == "POST"]
    assert len(posts) == 10
    for request in posts:
        wire = json.loads(request.content)
        assert set(wire["state"]) == {"text", "context"}
        assert list(wire["questions"]["triage"]["criteria"]) == [
            "ALLOW",
            "REVIEW",
            "BLOCK",
        ]
        assert "decision" not in wire["state"] and "probabilities" not in wire["state"]
        assert "returned_model" not in wire["state"]

    rows = read_rows(config.run / "smoke/predictions.jsonl")
    by_model = {
        name: [row["prediction"]["decision"] for row in rows if row["model_name"] == name]
        for name in ("laya", "jev")
    }
    assert by_model == {"laya": ["ALLOW"] * 5, "jev": ["BLOCK"] * 5}
    assert runner.ledger_for(config).summary()["attempts_only"] == 10
