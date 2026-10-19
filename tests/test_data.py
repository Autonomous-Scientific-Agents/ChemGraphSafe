import json
from pathlib import Path

import pytest

from chemgraphsafe.config import ModelConfig, load_config
from chemgraphsafe.data import (
    deduplicate_and_group,
    examples,
    freeze,
    prepare,
    select_pilot,
    verify_freeze,
)
from chemgraphsafe.models import Annotation, Prediction
from chemgraphsafe.storage import digest, write_rows


def test_prepare_freeze_roundtrip_and_no_overwrite(config):
    result = prepare(config)
    assert result["classes"] == {"benign": 5, "harmful": 5}
    assert prepare(config) == result
    frozen = freeze(config)
    assert verify_freeze(config) == frozen
    split = frozen["split_manifest"]["splits"]
    assert list(split.values()).count("eval") == 8
    assert frozen["annotation_count"] == 0
    assert len({e.group for e in examples(config)}) == 10
    for action in (prepare, freeze):
        with pytest.raises(ValueError, match="frozen"):
            action(config)


def test_freeze_detects_data_policy_and_new_annotation_tampering(config):
    prepare(config)
    freeze(config)
    path = config.dataset / "annotations.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="annotations changed"):
        verify_freeze(config)
    path.unlink()
    policy_path = config.path(config.policy_file)
    policy = json.loads(policy_path.read_text())
    policy["instructions"] = "changed policy"
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="policy/model"):
        verify_freeze(config)


def test_code_provenance_does_not_make_data_stale(config, monkeypatch):
    prepare(config)
    original = freeze(config)
    monkeypatch.setattr("chemgraphsafe.data.code_hash", lambda: "new-code")
    assert verify_freeze(config) == original


def test_exact_conflicts_quarantine_and_group_before_selection(config):
    prepare(config)
    rows = examples(config)
    clone = rows[0].model_copy(deep=True)
    clone.id = "duplicate-copy"
    clone.source_label = "harmful" if rows[0].source_label == "benign" else "benign"
    grouped, stats = deduplicate_and_group(rows + [clone], 0.9)
    assert set(stats["quarantined_ids"]) == {rows[0].id, clone.id}
    assert len(grouped) == 9
    with pytest.raises(ValueError, match="Insufficient"):
        select_pilot(grouped, 10, 4, 42)


def test_same_family_cannot_be_sampled_twice(config):
    prepare(config)
    rows = examples(config)
    rows[0].family = rows[1].family = "same-variant-family"
    pool, stats = deduplicate_and_group(rows, 1)
    assert stats["group_count"] == 9
    selected = select_pilot(pool, 8, 3, 42)
    assert len({e.group for e in selected}) == len(selected)


def test_later_annotations_keep_original_split(config):
    prepare(config)
    first = freeze(config)
    source = config.dataset
    updated = config.model_copy(deep=True)
    updated.data.directory = ".local/data/annotated-v2"
    updated.data.inherited_freeze = str(source / "freeze.json")
    updated.curator = ModelConfig(name="curator", adapter="replay", model="owned")
    prepare(updated)
    e = examples(updated)[0]
    annotation = Annotation(
        example_id=e.id,
        policy_hash=digest(updated.policy),
        config_hash=digest(updated.curator.spec()),
        origin="owned_fixture",
        requested_model="owned",
        prediction=Prediction(status="ok", decision="ALLOW", reason="legitimate_use"),
    )
    write_rows(updated.dataset / "annotations.jsonl", [annotation.model_dump()])
    second = freeze(updated)
    assert first["split_hash"] == second["split_hash"]
    assert first["examples_hash"] == second["examples_hash"]
    assert second["annotation_count"] == 1


def test_config_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    file = configs / "test.toml"
    file.write_text(
        'project_root=".."\noffline=true\n[[models]]\nname="demo"\nadapter="replay"\nmodel="mock"\n'
    )
    a = load_config(file)
    monkeypatch.chdir(Path("/private/tmp"))
    b = load_config(file)
    assert a.path(a.budget.ledger) == b.path(b.budget.ledger)
    assert a.path(a.budget.ledger) == tmp_path / ".local/state/pilot-budget.sqlite"
