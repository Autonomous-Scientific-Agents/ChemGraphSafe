import json

import httpx
import pytest

from chemgraphsafe import adapters, data, runner
from chemgraphsafe.budget import Ledger
from chemgraphsafe.config import KevServingConfig, ModelConfig
from chemgraphsafe.models import Request
from chemgraphsafe.storage import digest, read_json, read_rows

REVISION = "a" * 40
RUNTIME = "b" * 40


def kev_model(**updates):
    return ModelConfig.model_validate(
        {
            "name": "kev",
            "adapter": "kev",
            "model": "jaredpalmer/kev-0.8b",
            "endpoint": "http://127.0.0.1:8008/v1/systemone",
            "budget_mode": "attempts_only",
            "local": {
                "revision": REVISION,
                "runtime_revision": RUNTIME,
                "base_model": "Qwen/Qwen3.5-0.8B-Base",
                "calibration_temperature": 2.35,
            },
            **updates,
        }
    )


def metadata():
    return {
        "models": [
            {
                "run": f"jaredpalmer/kev-0.8b@{REVISION}",
                "base": "Qwen/Qwen3.5-0.8B-Base",
                "device": "cpu",
                "dtype": "float32",
                "backend": "torch",
                "temperature": 2.35,
                "truncate_states": False,
            }
        ]
    }


def reply():
    return {
        "model": "unverified-echo",
        "answers": {
            "triage": {
                "type": "choice",
                "choice": "ALLOW",
                "probabilities": {"ALLOW": 0.8, "REVIEW": 0.15, "BLOCK": 0.05},
            }
        },
        "usage": {"input_tokens": 50, "output_tokens": 123, "cost": 999},
    }


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:8008/v1/systemone",
        "http://0.0.0.0:8008/v1/systemone",
        "http://127.0.0.1.evil.test/v1/systemone",
        "http://192.168.1.5/v1/systemone",
        "http://user:pass@127.0.0.1/v1/systemone",
        "http://127.0.0.1/v1/systemone?key=secret",
        "http://127.0.0.1/v1/systemone#fragment",
        "http://127.0.0.1/v1/chat/completions",
        "https://example.test/v1/systemone",
    ],
)
def test_local_endpoint_boundary(endpoint):
    with pytest.raises(ValueError, match="loopback"):
        kev_model(endpoint=endpoint)


@pytest.mark.parametrize(
    "updates",
    [
        {"local": None},
        {"budget_mode": "usd"},
        {"temperature": 0},
        {"seed": 42},
        {"input_usd_per_million": 0},
        {"model": "kev@main"},
    ],
)
def test_local_configuration_rejects_unsupported_options(updates):
    with pytest.raises(ValueError):
        kev_model(**updates)


def test_legacy_specs_and_remote_endpoint_rules_remain_unchanged():
    model = ModelConfig(name="fixture", adapter="replay", model="mock")
    assert "local" not in model.spec()
    for engine in ("decisions", "chat"):
        with pytest.raises(ValueError, match="HTTPS"):
            ModelConfig(
                name="remote",
                adapter=engine,
                model="test",
                api_key_env="KEY",
                endpoint="http://127.0.0.1:8008/v1/systemone",
            )
    with pytest.raises(ValueError, match="Local settings"):
        ModelConfig(name="fixture", adapter="replay", model="mock", local=kev_model().local)
    with pytest.raises(ValueError):
        KevServingConfig(revision="main", runtime_revision=RUNTIME)


@pytest.mark.parametrize(
    "field,value",
    [
        ("run", "wrong"),
        ("base", "wrong-base"),
        ("device", "cuda"),
        ("dtype", "bfloat16"),
        ("truncate_states", True),
        ("temperature", 1.0),
        ("backend", "mlx"),
    ],
)
def test_kev_preflight_mismatch_never_infers(config, field, value):
    body = metadata()
    body["models"][0][field] = value
    ledger = Ledger(config.root / "ledger", 1, 5)

    def handle(request):
        assert request.method == "GET"
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError, match="preflight"):
            adapters.build_model(config, kev_model(), ledger, client=client)
    assert ledger.summary()["attempts"] == 0


def test_identity_rechecked_before_retry(config):
    calls = []

    def handle(request):
        calls.append(request.method)
        body = metadata()
        if len(calls) > 3:
            body["models"][0]["run"] = "changed"
        return httpx.Response(200, json=body) if request.method == "GET" else httpx.Response(503)

    ledger = Ledger(config.root / "ledger", 1, 5)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = adapters.build_model(
            config, kev_model(), ledger, client=client, sleep=lambda _: None
        ).predict(Request(text="Explain a bond."), config.policy)
    assert result.error_code == "local_identity_changed" and result.attempts == 1
    assert calls == ["GET", "GET", "POST", "GET"]


def test_local_workflow_resume_and_runtime_identity(config, monkeypatch):
    config.offline = False
    config.models = [kev_model()]
    monkeypatch.setattr(
        runner, "check_hosted_rights", lambda rows: pytest.fail("Local calls are not hosted calls")
    )
    original = httpx.Client
    posts = []

    def handle(request):
        if request.method == "POST":
            posts.append(request)
        return httpx.Response(200, json=metadata() if request.method == "GET" else reply())

    monkeypatch.setattr(
        adapters.httpx, "Client", lambda **kw: original(**kw, transport=httpx.MockTransport(handle))
    )
    data.prepare(config)
    data.freeze(config)
    assert runner.evaluate(config)["complete"]
    manifest = read_json(config.run / "manifest.json")
    assert manifest["execution"]["models"]["kev"]["compute"] == "local"
    assert runner.evaluate(config)["complete"] and len(posts) == 10
    before = (config.run / "predictions.jsonl").read_bytes()
    config.models[0].local.runtime_revision = "d" * 40
    with pytest.raises(ValueError, match="Frozen policy/model/protocol"):
        runner.evaluate(config)
    assert len(posts) == 10 and (config.run / "predictions.jsonl").read_bytes() == before
    assert all(
        (
            row["prediction"]["provider_cost_usd"] is None
            for row in read_rows(config.run / "predictions.jsonl")
        )
    )


def test_local_settings_affect_semantic_identity(config):
    config.offline = False
    config.models = [kev_model()]
    initial = config.semantic_hash()
    for key, value in (
        ("revision", "d" * 40),
        ("runtime_revision", "e" * 40),
        ("calibration_temperature", 1.0),
        ("device", "cuda"),
    ):
        changed = config.models[0].model_copy(deep=True)
        setattr(changed.local, key, value)
        assert digest(changed.spec()) != digest(config.models[0].spec())
    assert config.semantic_hash() == initial


@pytest.mark.parametrize(
    "probabilities",
    [
        {"ALLOW": 1},
        {"ALLOW": 0.1, "REVIEW": 0.1, "BLOCK": 0.1},
        {"ALLOW": True, "REVIEW": 0, "BLOCK": 0},
        {"ALLOW": "0.8", "REVIEW": 0.15, "BLOCK": 0.05},
    ],
)
def test_local_malformed_distributions_are_accounted_errors(config, probabilities):
    body = reply()
    body["answers"]["triage"]["probabilities"] = probabilities
    ledger = Ledger(config.root / "ledger", 1, 5)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=metadata() if r.method == "GET" else body)
        )
    ) as client:
        result = adapters.build_model(config, kev_model(), ledger, client=client).predict(
            Request(text="Explain a bond."), config.policy
        )
    assert result.error_code == "schema_error" and result.attempts == 1
    assert result.provider_cost_usd is None
    assert ledger.summary()["attempts_only"] == 1


def test_local_timeouts_and_attempt_cap(config):
    posts = []

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json=metadata())
        posts.append(request)
        raise httpx.ReadTimeout("owned timeout fixture", request=request)

    ledger = Ledger(config.root / "ledger", 1, 2)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        adapter = adapters.build_model(
            config, kev_model(), ledger, client=client, sleep=lambda _: None
        )
        result = adapter.predict(Request(text="Explain a bond."), config.policy)
    assert len(posts) == 2 and result.attempts == 2 and (result.error_code == "budget_exhausted")
    assert ledger.summary()["attempts_only"] == 2
    config.runner.retries = 0
    ledger = Ledger(config.root / "second-ledger", 1, 2)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = adapters.build_model(config, kev_model(), ledger, client=client).predict(
            Request(text="Explain a bond."), config.policy
        )
    assert result.error_code == "local_timeout" and result.attempts == 1


def test_local_optional_authentication(config, monkeypatch):
    monkeypatch.delenv("LOCAL_TEST_KEY", raising=False)
    model = kev_model(api_key_env="LOCAL_TEST_KEY")
    ledger = Ledger(config.root / "ledger", 1, 2)
    with pytest.raises(ValueError, match="credential"):
        adapters.build_model(config, model, ledger)
    monkeypatch.setenv("LOCAL_TEST_KEY", "local-only-secret")

    def handle(request):
        assert request.headers["authorization"] == "Bearer local-only-secret"
        return httpx.Response(200, json=metadata() if request.method == "GET" else reply())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        adapter = adapters.build_model(config, model, ledger, client=client)
        assert adapter.predict(Request(text="Explain a bond."), config.policy).status == "ok"
        assert "local-only-secret" not in json.dumps(adapter.runtime_provenance)


def test_mixed_local_and_hosted_models_still_require_hosted_rights(config):
    config.offline = False
    config.models = [
        kev_model(),
        ModelConfig(
            name="argo", adapter="argo", model="argo:gpt-4.1-mini", budget_mode="attempts_only"
        ),
    ]
    data.prepare(config)
    rows = data.examples(config)
    rows[0].hosted_processing_allowed = False
    with pytest.raises(ValueError, match="Hosted processing"):
        runner._run_models(config, rows, {}, {}, config.run)


def test_local_clients_disable_environment_proxies(config, monkeypatch):
    original = httpx.Client
    seen = []

    def client(**kwargs):
        seen.append(kwargs)
        return original(
            **kwargs, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=metadata()))
        )

    monkeypatch.setattr(adapters.httpx, "Client", client)
    ledger = Ledger(config.root / "ledger", 1, 1)
    adapter = adapters.build_model(config, kev_model(), ledger)
    adapter.close()
    assert seen[0]["trust_env"] is False


def test_kev_wire_leakage_identity_and_accounting(config):
    calls = []
    ledger = Ledger(config.root / "ledger", 1, 3)

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=metadata() if request.method == "GET" else reply())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        adapter = adapters.build_model(config, kev_model(), ledger, "run", client=client)
        assert ledger.summary()["attempts"] == 0
        result = adapter.predict(
            Request(text="Explain chemical bonds.", context="Classroom"), config.policy
        )
    assert [call.method for call in calls] == ["GET", "GET", "POST"]
    wire = json.loads(calls[-1].content)
    assert set(wire) == {"model", "state", "questions"}
    assert wire["state"] == {"text": "Explain chemical bonds.", "context": "Classroom"}
    assert wire["questions"]["triage"]["criteria"] == config.policy["criteria"]
    assert "authorization" not in calls[-1].headers
    assert result.status == "ok" and result.probability_origin == "native"
    assert result.returned_model == f"jaredpalmer/kev-0.8b@{REVISION}"
    assert result.input_tokens == 50 and result.output_tokens is None
    assert result.provider_cost_usd is None and result.estimated_cost_usd is None
    assert ledger.summary()["attempts_only"] == 1
    assert ledger.summary()["accounted_usd"] == 0
    assert ledger.summary()["unreconciled_attempts"] == 0
    assert adapter.runtime_provenance["compute"] == "local"
    assert adapter.runtime_provenance["declared_runtime_revision"] == RUNTIME


@pytest.mark.parametrize(
    "abstention,status", [(True, "abstain"), (False, "ok"), ("unknown", "error")]
)
def test_kev_abstention_is_not_an_allow_decision(config, abstention, status):
    body = reply()
    body["answers"]["triage"]["abstention"] = abstention
    ledger = Ledger(config.root / "ledger", 1, 3)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=metadata() if r.method == "GET" else body)
        )
    ) as client:
        result = adapters.build_model(config, kev_model(), ledger, client=client).predict(
            Request(text="Explain bonds."), config.policy
        )
    assert result.status == status and result.attempts == 1
    if status != "ok":
        assert result.decision is None and result.probabilities is None


@pytest.mark.parametrize(
    "change,code",
    [
        ({"truncated": True}, "local_input_truncated"),
        ({"answers": None}, "schema_error"),
        ({"answers": {"triage": {"type": "text"}}}, "schema_error"),
    ],
)
def test_kev_rejects_truncated_or_malformed_answers(config, change, code):
    body = {**reply(), **change}
    ledger = Ledger(config.root / "ledger", 1, 3)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=metadata() if r.method == "GET" else body)
        )
    ) as client:
        result = adapters.build_model(config, kev_model(), ledger, client=client).predict(
            Request(text="Explain bonds."), config.policy
        )
    assert result.status == "error" and result.error_code == code
    assert result.decision is None and result.attempts == 1


def test_kev_cannot_curate_and_does_not_infer_on_missing_server(config):
    ledger = Ledger(config.root / "ledger", 1, 3)
    with pytest.raises(ValueError, match="curation"):
        adapters.build_model(config, kev_model(), ledger, curator=True)

    def unavailable(request):
        assert request.method == "GET"
        raise httpx.ConnectError("owned unavailable fixture", request=request)

    with httpx.Client(transport=httpx.MockTransport(unavailable)) as client:
        with pytest.raises(ValueError, match="preflight"):
            adapters.build_model(config, kev_model(), ledger, client=client)
    assert ledger.summary()["attempts"] == 0


def test_kev_example_configuration():
    from pathlib import Path

    from chemgraphsafe.config import load_config

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/kev-local.toml")
    assert config.models[0].adapter == "kev"
    assert config.models[0].local.calibration_temperature == 2.3510958125672174
    assert not config.offline


@pytest.mark.parametrize(
    "binary,probabilities,status",
    [
        (True, {"ALLOW": 0.8, "BLOCK": 0.2}, "ok"),
        (True, {"ALLOW": 0.8, "REVIEW": 0.15, "BLOCK": 0.05}, "error"),
        (False, {"ALLOW": 0.8, "BLOCK": 0.2}, "error"),
    ],
)
def test_kev_distribution_must_match_active_policy(config, binary, probabilities, status):
    if binary:
        config.policy_file = "policies/triage-binary-v0.1.json"
    body = reply()
    body["answers"]["triage"]["probabilities"] = probabilities
    ledger = Ledger(config.root / "ledger", 1, 3)

    def handle(request):
        if request.method == "POST":
            wire = json.loads(request.content)
            assert wire["questions"]["triage"]["criteria"] == config.policy["criteria"]
        return httpx.Response(200, json=metadata() if request.method == "GET" else body)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = adapters.build_model(config, kev_model(), ledger, client=client).predict(
            Request(text="Explain chemical bonds."), config.policy
        )
    assert result.status == status and result.attempts == 1
    if status == "error":
        assert result.decision is None and result.error_code == "schema_error"
