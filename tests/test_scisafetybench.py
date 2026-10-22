"""Source-shaped owned fixtures only; no benchmark request text or live inference."""

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from chemgraphsafe import data
from chemgraphsafe import scisafetybench as ssb
from chemgraphsafe.adapters import HTTPModel
from chemgraphsafe.budget import Ledger
from chemgraphsafe.cli import app
from chemgraphsafe.config import Config, DataConfig, load_config
from chemgraphsafe.models import Prediction, RuleAnnotation
from chemgraphsafe.reporting import metrics, report
from chemgraphsafe.runner import curate, evaluate
from chemgraphsafe.storage import digest, file_hash, read_json, read_rows, write_json, write_rows


@pytest.fixture
def challenge(config, monkeypatch):
    config.data = DataConfig(
        loader="scisafetybench",
        protocol="challenge_eval",
        revision=ssb.REVISION,
        count=80,
        directory=".local/data/challenge",
    )
    config.policy_file = "policies/triage-chem-materials-oversight-v0.1.json"
    config.annotation_rule = ssb.RULE_ID
    specs = copy.deepcopy(ssb.SOURCE_FILES)
    raw = config.dataset / "raw"
    raw.mkdir(parents=True)
    for filename, spec in specs.items():
        rows = []
        for category, count in spec["risk_counts"].items():
            for _ in range(count):
                index = len(rows)
                rows.append(
                    {
                        "Task": f"PRIVATE TASK {index}",
                        "Task Description": "PRIVATE TASK DESCRIPTION",
                        "Risk Type": category,
                        "Source": "PRIVATE SOURCE",
                        "Prompt": f"  Define the classroom vocabulary entry {filename} {index}.\n",
                    }
                )
        # Near duplicates across domains are grouped but retained in the all-eval protocol.
        rows[0]["Prompt"] = "Explain what an atom is to a student in introductory science. " + (
            "A" if filename == "chem.json" else "B"
        )
        write_json(raw / filename, rows)
        spec["sha256"] = file_hash(raw / filename)
    monkeypatch.setattr(ssb, "SOURCE_FILES", specs)
    monkeypatch.setattr(data, "SCISAFETYBENCH_FILES", specs)
    return config


def modify_source(config, change):
    path = config.dataset / "raw/chem.json"
    rows = read_json(path)
    change(rows)
    write_json(path, rows)
    ssb.SOURCE_FILES["chem.json"]["sha256"] = file_hash(path)


def replay_rules(config):
    records = {
        digest(e.request.model_dump()): {
            "status": "ok",
            "decision": ssb.RISK_MAPPING[e.category],
        }
        for e in data.examples(config)
    }
    config.models[0].replay_file = "challenge-replay.json"
    write_json(config.root / "challenge-replay.json", records)


def test_pinned_loader_schema_ids_domains_and_verbatim_input(challenge):
    rows = ssb.SciSafetyBenchLoader().load(challenge)
    assert len(rows) == len({e.id for e in rows}) == 80
    assert Counter(e.domain for e in rows) == {"Chemistry": 40, "Materials Science": 40}
    assert Counter(e.category for e in rows) == dict(
        zip(ssb.RISK_MAPPING, (19, 18, 20, 23), strict=True)
    )
    assert all(e.source_label is None and e.source_native_harmful is None for e in rows)
    for e in rows:
        filename, index = e.source_id.split(":")
        raw = read_json(challenge.dataset / "raw" / filename)[int(index)]
        assert e.request.model_dump() == {"text": raw["Prompt"], "context": ""}
        assert e.id == "scisafetybench-" + digest([ssb.REVISION, filename, int(index)])[:20]
        assert e.source_revision == ssb.REVISION and ssb.REVISION in e.rights_evidence
        assert e.hosted_processing_allowed and e.license == "MIT"
        assert "PRIVATE" not in e.model_dump_json()
    assert rows == ssb.SciSafetyBenchLoader().load(challenge)


@pytest.mark.parametrize("bad", ["missing", "extra", "type", "empty", "category", "count", "mix"])
def test_loader_fails_closed_without_printing_raw_values(challenge, bad):
    def change(rows):
        if bad == "missing":
            rows[0].pop("Source")
        elif bad == "extra":
            rows[0]["Answer"] = "PRIVATE ANSWER"
        elif bad == "type":
            rows[0]["Task"] = 3
        elif bad == "empty":
            rows[0]["Prompt"] = "  "
        elif bad == "category":
            rows[0]["Risk Type"] = "PRIVATE UNKNOWN CATEGORY"
        elif bad == "count":
            rows.pop()
        else:
            rows[0]["Risk Type"] = "Task-Intrinsic Risk"

    modify_source(challenge, change)
    with pytest.raises(ValueError) as exc:
        ssb.SciSafetyBenchLoader().load(challenge)
    assert "PRIVATE" not in str(exc.value)
    assert not (challenge.dataset / "examples.jsonl").exists()


def test_loader_rejects_revision_subset_and_corrupt_cache(challenge):
    challenge.data.revision = "f" * 40
    with pytest.raises(ValueError, match="revision"):
        ssb.SciSafetyBenchLoader().load(challenge)
    challenge.data.revision = ssb.REVISION
    challenge.data.count = 40
    with pytest.raises(ValueError, match="80"):
        ssb.SciSafetyBenchLoader().load(challenge)
    challenge.data.count = 80
    (challenge.dataset / "raw/chem.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="checksum"):
        ssb.SciSafetyBenchLoader().load(challenge)


@pytest.mark.parametrize("corrupt", [False, True])
def test_download_is_pinned_and_checked_before_caching(challenge, monkeypatch, corrupt):
    content = {}
    for filename in ssb.SOURCE_FILES:
        path = challenge.dataset / "raw" / filename
        content[filename] = path.read_bytes()
        path.unlink()
    urls = []

    def respond(request):
        urls.append(str(request.url))
        assert f"/{ssb.REVISION}/{ssb.DATA_PATH}/" in str(request.url)
        return httpx.Response(
            200, content=b"[]" if corrupt else content[request.url.path.rsplit("/", 1)[1]]
        )

    # The loader uses one context manager per file, so give each call a fresh mock client.
    constructor = httpx.Client
    monkeypatch.setattr(
        ssb.httpx, "Client", lambda **kw: constructor(transport=httpx.MockTransport(respond), **kw)
    )
    if corrupt:
        with pytest.raises(ValueError, match="checksum"):
            ssb.SciSafetyBenchLoader().load(challenge)
        assert not (challenge.dataset / "raw/chem.json").exists()
    else:
        assert len(ssb.SciSafetyBenchLoader().load(challenge)) == 80 and len(urls) == 2
        assert ssb.SciSafetyBenchLoader().load(challenge) and len(urls) == 2


def test_all_eval_grouping_annotations_and_immutable_freeze(challenge):
    prepared = data.prepare(challenge)
    assert prepared["classes"] == {} and prepared["unlabeled_count"] == 80
    assert prepared["selected_count"] == 80 and prepared["grouping"]["cross_domain_groups"] > 0
    assert prepared == data.prepare(challenge)
    with pytest.raises(ValueError, match="complete, exact coverage"):
        data.freeze(challenge)
    assert data.annotate(challenge)["classes"] == {"BLOCK": 37, "REVIEW": 43}
    before = (challenge.dataset / "annotations.jsonl").read_bytes()
    data.annotate(challenge)
    assert before == (challenge.dataset / "annotations.jsonl").read_bytes()
    annotations = data.annotations(challenge)
    assert all(isinstance(a, RuleAnnotation) for a in annotations.values())
    assert all("requested_model" not in a.model_dump() for a in annotations.values())
    assert {a.rule_hash for a in annotations.values()} == {
        digest(ssb.rule_spec(ssb.RULE_ID, challenge.policy))
    }
    frozen = data.freeze(challenge)
    assert frozen == data.verify_freeze(challenge) and frozen["annotation_count"] == 80
    assert set(frozen["split_manifest"]["splits"].values()) == {"eval"}
    assert len(frozen["split_manifest"]["case_order"]) == 80
    for action in (data.prepare, data.annotate, data.freeze):
        with pytest.raises(ValueError, match="frozen"):
            action(challenge)
    # Changing the dataset seed after freezing is not a silently accepted run change.
    challenge.data.seed += 1
    with pytest.raises(ValueError, match="Preparation changed"):
        data.verify_freeze(challenge)


def test_challenge_retains_order_across_versions(challenge):
    data.prepare(challenge)
    data.annotate(challenge)
    first = data.freeze(challenge)
    old = challenge.dataset
    challenge.data.directory = ".local/data/second"
    import shutil

    shutil.copytree(old / "raw", challenge.dataset / "raw")
    data.prepare(challenge)
    data.annotate(challenge)
    second = data.freeze(challenge)
    assert first["split_manifest"] == second["split_manifest"]
    assert first["examples_hash"] == second["examples_hash"]


def test_challenge_rejects_exact_duplicates_even_across_domains(challenge):
    material_prompt = read_json(challenge.dataset / "raw/material.json")[0]["Prompt"]
    modify_source(challenge, lambda rows: rows[0].update(Prompt=material_prompt))
    with pytest.raises(ValueError, match="duplicates"):
        data.prepare(challenge)
    assert not (challenge.dataset / "prepare.json").exists()


@pytest.mark.parametrize("bad", ["missing", "decision", "hash", "id", "provenance", "duplicate"])
def test_rule_annotations_reject_tampering_or_partial_coverage(challenge, bad):
    data.prepare(challenge)
    data.annotate(challenge)
    path = challenge.dataset / "annotations.jsonl"
    rows = read_rows(path)
    if bad == "missing":
        rows.pop()
    elif bad == "decision":
        rows[0]["prediction"]["decision"] = "ALLOW"
    elif bad == "hash":
        rows[0]["rule_hash"] = "changed"
    elif bad == "id":
        rows[0]["example_id"] = "foreign"
    elif bad == "provenance":
        rows[0]["provenance"]["risk_type"] = "changed"
    else:
        rows.append(rows[0])
    write_rows(path, rows)
    for action in (data.annotate, data.freeze):
        with pytest.raises(ValueError):
            action(challenge)
    assert not (challenge.dataset / "freeze.json").exists()


def test_rule_policy_binding_and_no_llm_curation(challenge):
    assert digest(challenge.policy) == ssb.POLICY_HASH
    data.prepare(challenge)
    with pytest.raises(ValueError, match="data annotate"):
        curate(challenge)
    policy = challenge.policy
    policy["criteria"]["REVIEW"] = "changed"
    write_json(challenge.path(challenge.policy_file), policy)
    for action in (data.annotate, data.freeze):
        with pytest.raises(ValueError, match="exact shared oversight policy"):
            action(challenge)
    assert not (challenge.dataset / "annotations.jsonl").exists()


def test_new_options_preserve_legacy_specs_and_balanced_gates(config):
    spec = config.data.model_dump()
    spec.pop("protocol")
    assert config.data.spec() == spec
    assert config.semantic_config() == {
        "models": [m.spec() for m in config.models],
        "curator": None,
        "policy": config.policy,
        "offline": True,
    }
    with pytest.raises(ValueError, match="balanced_pilot"):
        DataConfig(loader="scihazard", protocol="challenge_eval")
    with pytest.raises(ValueError, match="challenge_eval"):
        DataConfig(loader="scisafetybench")
    prepared = data.prepare(config)
    assert prepared["preparation_hash"] == digest(
        {**spec, "fixture_hash": file_hash(config.path(config.data.fixture))}
    )
    rows = data.examples(config)
    rows[0].source_label = None
    with pytest.raises(ValueError, match="binary source labels"):
        data.select_pilot(rows, 10, 4, 42)
    values = config.model_dump()
    values["annotation_rule"] = ssb.RULE_ID
    with pytest.raises(ValueError, match="challenge_eval"):
        Config.model_validate(values)


def test_legacy_freeze_remains_self_contained_without_original_fixture(config):
    data.prepare(config)
    original = data.freeze(config)
    config.path(config.data.fixture).unlink()
    assert data.verify_freeze(config) == original


def test_legacy_reports_without_domain_metadata(config):
    evaluate(config)
    refs = read_json(config.run / "references.json")
    for ref in refs:
        ref.pop("domain")
    manifest = read_json(config.run / "manifest.json")
    manifest["references_hash"] = digest(refs)
    write_json(config.run / "manifest.json", manifest)
    write_json(config.run / "references.json", refs)
    report(config.run)
    result = read_json(config.run / "metrics.json")
    assert "domain" not in result["models"]["replay-demo"]["slices"]


def test_missing_annotations_are_unavailable_not_rule_references(challenge):
    challenge.annotation_rule = None
    data.prepare(challenge)
    replay_rules(challenge)
    data.freeze(challenge)
    evaluate(challenge)
    report(challenge.run)
    result = read_json(challenge.run / "metrics.json")
    assert result["models"]["replay-demo"]["splits"]["eval"]["policy_agreement"] is None
    assert "rule-derived" not in (challenge.run / "report.md").read_text()
    assert "unavailable" in (challenge.run / "comparison.csv").read_text()


def test_metrics_use_all_intended_cases_without_fabricating_binary_labels(challenge):
    data.prepare(challenge)
    data.annotate(challenge)
    refs = [
        {"id": a.example_id, "source_label": None, "annotation": a.model_dump()}
        for a in data.annotations(challenge).values()
    ]
    blocks = [r for r in refs if r["annotation"]["prediction"]["decision"] == "BLOCK"]
    predictions = {r["id"]: Prediction(**r["annotation"]["prediction"]) for r in refs}
    predictions[blocks[0]["id"]] = Prediction(status="abstain")
    predictions[blocks[1]["id"]] = Prediction(status="error")
    del predictions[blocks[2]["id"]]
    m = metrics(refs, predictions, policy=challenge.policy)
    assert m["n"] == 80 and m["source_label_coverage"]["value"] == 0
    assert m["policy_agreement"]["accuracy"]["value"] == 77 / 80
    assert m["policy_agreement"]["recall"]["BLOCK"]["value"] == 34 / 37
    assert m["policy_agreement"]["recall"]["REVIEW"]["value"] == 1
    assert m["policy_agreement"]["reference_counts"] == {"ALLOW": 0, "BLOCK": 37, "REVIEW": 43}
    assert m["policy_agreement"]["macro_f1"] is None
    assert m["policy_agreement"]["confusion"]["BLOCK"]["ABSTAIN"] == 1
    assert m["policy_agreement"]["confusion"]["BLOCK"]["ERROR"] == 2
    assert m["error_rate"]["value"] == 2 / 80 and m["abstention_overall"]["value"] == 1 / 80
    assert m["decided_only"]["n"] == 0 and m["benign_block"]["value"] is None
    binary = {"criteria": {"ALLOW": "", "BLOCK": ""}}
    unannotated = [{**r, "annotation": None} for r in refs]
    assert metrics(unannotated, {}, policy=binary)["accuracy"]["value"] is None


def test_challenge_workflow_reports_baselines_slices_and_id_only_errors(challenge):
    data.prepare(challenge)
    data.annotate(challenge)
    replay_rules(challenge)
    records = read_json(challenge.root / "challenge-replay.json")
    records[next(iter(records))]["decision"] = "ALLOW"
    write_json(challenge.root / "challenge-replay.json", records)
    data.freeze(challenge)
    assert evaluate(challenge)["complete"]
    before = (challenge.run / "predictions.jsonl").read_bytes()
    assert evaluate(challenge)["complete"]
    assert before == (challenge.run / "predictions.jsonl").read_bytes()
    report(challenge.run)
    result = read_json(challenge.run / "metrics.json")
    for name, expected in (
        ("always-allow", 0),
        ("always-block", 37 / 80),
        ("always-review", 43 / 80),
    ):
        score = result["models"][name]["splits"]["eval"]["policy_agreement"]
        assert score["accuracy"]["value"] == expected and score["accuracy"]["n"] == 80
    model = result["models"]["replay-demo"]
    assert model["splits"]["eval"]["policy_agreement"]["accuracy"]["value"] == 79 / 80
    assert set(model["splits"]) == {"eval"}
    assert model["slices"]["domain"]["Chemistry"]["n"] == 40
    assert model["slices"]["domain"]["Materials Science"]["n"] == 40
    assert len(model["slices"]["domain_category"]["Materials Science"]) == 4
    assert read_rows(challenge.run / "errors.jsonl")[0]["issue"] == (
        "policy_reference_disagreement_unverified"
    )
    markdown = (challenge.run / "report.md").read_text()
    assert "rule-derived provisional" in markdown and "not expert validated" in markdown
    assert "| ALLOW | 0 | 0 | 0 | 0 | 0 |" in markdown
    assert "unavailable" in markdown and "43" in markdown
    for filename in (
        "report.md",
        "comparison.csv",
        "metrics.json",
        "errors.jsonl",
        "references.json",
    ):
        content = (challenge.run / filename).read_text()
        assert "PRIVATE" not in content
        assert all(e.request.text.strip() not in content for e in data.examples(challenge))


def test_jev_wire_has_only_prompt_and_shared_policy(challenge, monkeypatch):
    preset = load_config(Path(__file__).parents[1] / "configs/scisafetybench.toml")
    model = preset.models[0]
    monkeypatch.setenv(model.api_key_env, "test-key")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "answers": {
                    "triage": {
                        "type": "choice",
                        "choice": "REVIEW",
                        "probabilities": {"ALLOW": 0.1, "REVIEW": 0.8, "BLOCK": 0.1},
                    }
                }
            },
        )

    rows = ssb.SciSafetyBenchLoader().load(challenge)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = HTTPModel(
            challenge, model, Ledger(challenge.root / "ledger", 10, 500), "test", client=client
        )
        for e in (rows[0], rows[40]):
            assert adapter.predict(e.request, challenge.policy).decision == "REVIEW"
    assert requests[0]["questions"] == requests[1]["questions"]
    for e, wire in zip((rows[0], rows[40]), requests, strict=True):
        assert wire["state"] == {"text": e.request.text, "context": ""}
        body = json.dumps(wire)
        for secret in (e.id, e.category, e.source_id, e.source_revision, "PRIVATE", "rule_hash"):
            assert secret not in body
    assert preset.budget.max_attempts == 500 and preset.budget.max_usd == 10
    assert preset.budget.ledger == ".local/state/pilot-budget.sqlite"


def test_cli_prepare_annotate_freeze_without_inference(challenge, monkeypatch):
    monkeypatch.setattr("chemgraphsafe.cli.load_config", lambda _: challenge)
    monkeypatch.setattr(
        "chemgraphsafe.runner.build_model", lambda *a, **kw: pytest.fail("inference")
    )
    path = challenge.root / "test.toml"
    path.touch()
    cli = CliRunner()
    for command in ("prepare", "annotate", "freeze"):
        result = cli.invoke(app, ["data", command, "--config", str(path)])
        assert result.exit_code == 0, result.output
    assert data.verify_freeze(challenge)["annotation_count"] == 80
    assert not challenge.path(challenge.budget.ledger).exists()


def test_release_hashes_and_policy_are_pinned():
    config = load_config(Path(__file__).parents[1] / "configs/scisafetybench.toml")
    assert config.data.revision == ssb.REVISION
    assert digest(config.policy) == ssb.POLICY_HASH
    assert len(ssb.SOURCE_FILES) == 2 and config.data.count == 80
    assert all(
        len(spec["sha256"]) == hashlib.sha256().digest_size * 2
        for spec in ssb.SOURCE_FILES.values()
    )
