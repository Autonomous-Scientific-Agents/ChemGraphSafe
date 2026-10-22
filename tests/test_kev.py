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
        changed_config = config.model_copy(deep=True)
        changed_config.models = [changed]
        assert changed_config.semantic_hash() != initial


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


@pytest.mark.parametrize("failure", ["timeout", "reset", "http500", "json", "empty", "missing"])
def test_post_timeout_and_transient_preflight_recover(config, failure):
    config.runner.retries = 2
    calls, delays = [], []
    posts = 0

    def handle(request):
        nonlocal posts
        calls.append(request.method)
        if request.method == "POST":
            posts += 1
            if posts == 1:
                raise httpx.ReadTimeout("Owned POST timeout", request=request)
            return httpx.Response(200, json=reply())
        if len(calls) == 4:
            if failure == "timeout":
                raise httpx.ReadTimeout("Owned metadata timeout", request=request)
            if failure == "reset":
                raise httpx.ReadError("Owned connection reset", request=request)
            if failure == "http500":
                return httpx.Response(503)
            if failure == "json":
                return httpx.Response(200, content=b"not JSON")
            if failure == "empty":
                return httpx.Response(200, json={"models": []})
            body = metadata()
            body["models"][0].pop("run")
            return httpx.Response(200, json=body)
        return httpx.Response(200, json=metadata())

    ledger = Ledger(config.root / "ledger", 1, 10)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        instance = adapters.build_model(
            config, kev_model(), ledger, client=client, sleep=delays.append
        )
        result = instance.predict(Request(text="Define an atom."), config.policy)
    assert result.status == "ok" and result.attempts == 2
    assert calls == ["GET", "GET", "POST", "GET", "GET", "POST"]
    assert delays == [1, 2] and ledger.summary()["attempts"] == 2
    assert "local_unavailable" not in runner.FATAL_CODES


def test_unavailable_preflight_retries_are_bounded_and_unmetered(config):
    config.runner.retries = 2
    gets, delays = [], []

    def handle(request):
        assert request.method == "GET"
        gets.append(True)
        if len(gets) == 1:
            return httpx.Response(200, json=metadata())
        raise httpx.ReadTimeout("Owned metadata outage", request=request)

    ledger = Ledger(config.root / "ledger", 1, 10)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        instance = adapters.build_model(
            config, kev_model(), ledger, client=client, sleep=delays.append
        )
        result = instance.predict(Request(text="Define an atom."), config.policy)
    assert len(gets) == 4 and delays == [1, 2]
    assert result.error_code == "local_unavailable" and result.attempts == 0
    assert ledger.summary()["attempts"] == 0


def test_metadata_retry_delay_honors_configured_limit(config):
    config.runner.retries = 2
    config.runner.max_retry_after_seconds = 0
    gets = []

    def handle(request):
        assert request.method == "GET"
        gets.append(True)
        if len(gets) == 1:
            return httpx.Response(200, json=metadata())
        return httpx.Response(503)

    ledger = Ledger(config.root / "ledger", 1, 10)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        instance = adapters.build_model(
            config,
            kev_model(),
            ledger,
            client=client,
            sleep=lambda _: pytest.fail("Delay exceeds configured limit"),
        )
        result = instance.predict(Request(text="Define an atom."), config.policy)
    assert result.error_code == "rate_limit_deferred" and result.attempts == 0
    assert len(gets) == 2 and ledger.summary()["attempts"] == 0


def test_metadata_outage_does_not_abort_remaining_cases(config, monkeypatch):
    config.offline = False
    config.models = [kev_model()]
    config.runner.retries = 0
    data.prepare(config)
    data.freeze(config)
    original = httpx.Client
    gets = []

    def handle(request):
        if request.method == "GET":
            gets.append(True)
            if len(gets) == 2:
                raise httpx.ReadTimeout("Owned first-case metadata outage", request=request)
        return httpx.Response(200, json=metadata() if request.method == "GET" else reply())

    monkeypatch.setattr(
        adapters.httpx, "Client", lambda **kw: original(**kw, transport=httpx.MockTransport(handle))
    )
    assert not runner.evaluate(config)["complete"]
    saved = read_rows(config.run / "predictions.jsonl")
    assert len(saved) == 10
    assert saved[0]["prediction"]["error_code"] == "local_unavailable"
    assert saved[0]["prediction"]["attempts"] == 0
    assert all(row["prediction"]["status"] == "ok" for row in saved[1:])
    assert runner.ledger_for(config).summary()["attempts"] == 9


@pytest.mark.parametrize("port", ["99999", "bad", "-1", "0"])
def test_kev_rejects_invalid_port_before_client_creation(port):
    with pytest.raises(ValueError, match="loopback.*port"):
        kev_model(endpoint=f"http://127.0.0.1:{port}/v1/systemone")


@pytest.mark.parametrize("adapter", ["kev", "decisions"])
@pytest.mark.parametrize(
    "policy_name",
    [
        "triage-v0.1.json",
        "triage-binary-v0.1.json",
        "scisafetybench-risk-type-v1.json",
    ],
)
def test_reversed_criteria_send_identical_wire_bytes(config, monkeypatch, adapter, policy_name):
    from chemgraphsafe.models import policy_labels, question_key

    config.policy_file = "policies/" + policy_name
    if policy_name == "scisafetybench-risk-type-v1.json":
        config.task = "scisafetybench_risk_type"
        config.data.protocol = "challenge_eval"
    policy = config.policy
    reversed_policy = {**policy, "criteria": dict(reversed(list(policy["criteria"].items())))}
    monkeypatch.setenv("OWNED_TEST_KEY", "owned-test-key")
    model = (
        kev_model()
        if adapter == "kev"
        else ModelConfig(
            name="decisions",
            adapter="decisions",
            model="owned-test-model",
            endpoint="https://example.test/decide",
            api_key_env="OWNED_TEST_KEY",
            budget_mode="attempts_only",
        )
    )
    labels = policy_labels(policy)
    key = question_key(policy)
    body = {
        "answers": {
            key: {
                "type": "choice",
                "choice": labels[0],
                "probabilities": {label: 1 / len(labels) for label in labels},
            }
        }
    }
    posts = []

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json=metadata())
        posts.append(request.content)
        assert tuple(json.loads(request.content)["questions"][key]["criteria"]) == labels
        return httpx.Response(200, json=body)

    ledger = Ledger(config.root / "ledger", 1, 10)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        instance = adapters.build_model(config, model, ledger, client=client)
        for selected in (policy, reversed_policy):
            result = instance.predict(Request(text="Define an atom."), selected)
            assert result.status == "ok" and set(result.probabilities) == set(labels)
    assert len(posts) == 2 and posts[0] == posts[1]
    assert digest(policy) == digest(reversed_policy)


def test_interrupted_resume_after_policy_reordering_preserves_inputs(config, monkeypatch):
    config.offline = False
    config.models = [kev_model()]
    config.runner.retries = 0
    config.run_dir = "runs/order-reference"
    data.prepare(config)
    data.freeze(config)
    original = httpx.Client
    posts = []
    interrupt = False

    def handle(request):
        nonlocal interrupt
        if request.method == "GET":
            if interrupt and len(posts) == 1:
                interrupt = False
                raise KeyboardInterrupt("Owned interruption before the second inference")
            return httpx.Response(200, json=metadata())
        posts.append(request.content)
        return httpx.Response(200, json=reply())

    monkeypatch.setattr(
        adapters.httpx, "Client", lambda **kw: original(**kw, transport=httpx.MockTransport(handle))
    )
    assert runner.evaluate(config)["complete"]
    reference = posts.copy()
    posts.clear()
    config.run_dir = "runs/order-interrupted"
    interrupt = True
    with pytest.raises(KeyboardInterrupt):
        runner.evaluate(config)
    saved = read_rows(config.run / "predictions.jsonl")
    assert len(saved) == 1
    run_key = read_json(config.run / "manifest.json")["run_key"]
    initial_hash = config.semantic_hash()
    policy = config.policy
    policy["criteria"] = dict(reversed(list(policy["criteria"].items())))
    config.path(config.policy_file).write_text(json.dumps(policy))
    assert config.semantic_hash() == initial_hash
    assert runner.evaluate(config)["complete"]
    assert posts == reference and len(posts) == 10
    assert read_json(config.run / "manifest.json")["run_key"] == run_key
    assert read_rows(config.run / "predictions.jsonl")[0] == saved[0]
