from copy import deepcopy
from types import SimpleNamespace

import pytest

from chemgraphsafe import provenance, runner
from chemgraphsafe.reporting import report
from chemgraphsafe.storage import read_json, write_json


def snapshot(path):
    return {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}


def test_installed_dependency_graph_excludes_dev_and_inactive_extras(monkeypatch):
    packages = {
        "chemgraphsafe": ("1", ["core", 'sdk; extra == "codex"', 'pytest; extra == "dev"']),
        "core": ("2", ["base", 'uninstalled; python_version < "1"', 'fast; extra == "speed"']),
        "base": ("3", []),
        "sdk": ("4", ["core[speed]", "runtime"]),
        "runtime": ("5", []),
        "fast": ("6", []),
    }

    def distribution(name):
        version, requires = packages[name]
        return SimpleNamespace(version=version, requires=requires)

    monkeypatch.setattr(provenance, "distribution", distribution)
    assert provenance.environment_identity()["packages"] == {
        "chemgraphsafe": "1",
        "core": "2",
        "base": "3",
    }
    assert set(provenance.environment_identity(codex=True)["packages"]) == set(packages)
    packages["base"] = ("next", [])
    assert provenance.environment_identity()["packages"]["base"] == "next"


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeout_seconds", 120),
        ("retries", 0),
        ("max_retry_after_seconds", 1),
    ],
)
def test_runner_change_rejects_resume_without_writes_or_predictions(
    config, monkeypatch, field, value
):
    runner.evaluate(config)
    before = snapshot(config.run)
    setattr(config.runner, field, value)

    def unexpected(*args):
        pytest.fail("An incompatible resume made a prediction")

    monkeypatch.setattr("chemgraphsafe.adapters.ReplayModel.predict", unexpected)
    with pytest.raises(ValueError, match="new run directory"):
        runner.evaluate(config)
    assert snapshot(config.run) == before


@pytest.mark.parametrize("field", ["python", "packages"])
def test_environment_change_rejects_resume_but_allows_same_freeze_in_new_run(
    config, monkeypatch, field
):
    runner.evaluate(config)
    manifest = read_json(config.run / "manifest.json")
    changed = deepcopy(manifest["execution"]["environment"])
    if field == "python":
        changed[field]["version"] = "next"
    else:
        changed[field]["httpx"] = "next"
    monkeypatch.setattr(runner, "environment_identity", lambda **kw: changed)
    before = snapshot(config.run)
    with pytest.raises(ValueError, match="new run directory"):
        runner.evaluate(config)
    assert snapshot(config.run) == before
    config.run_dir = "runs/new-environment"
    assert runner.evaluate(config)["complete"]
    updated = read_json(config.run / "manifest.json")
    assert updated["identity"] == manifest["identity"]
    assert updated["run_key"] != manifest["run_key"]


def test_legacy_report_readable_but_resume_requires_new_directory(config):
    runner.evaluate(config)
    manifest = read_json(config.run / "manifest.json")
    for key in ("manifest_version", "execution", "execution_hash", "lockfile_hash"):
        manifest.pop(key)
    write_json(config.run / "manifest.json", manifest)
    assert report(config.run)["mock"]
    before = snapshot(config.run)
    with pytest.raises(ValueError, match="new run directory"):
        runner.evaluate(config)
    assert snapshot(config.run) == before


def test_lockfile_provenance_preserved_without_treating_dev_lock_changes_as_runtime_changes(config):
    lock = config.root / "uv.lock"
    lock.write_text("first development lock")
    runner.evaluate(config)
    before = snapshot(config.run)
    assert read_json(config.run / "manifest.json")["lockfile_hash"] is not None
    lock.write_text("updated development tooling only")
    assert runner.evaluate(config)["complete"]
    assert snapshot(config.run) == before


def test_reference_validation_precedes_resume_writes(config):
    runner.evaluate(config)
    references = read_json(config.run / "references.json")
    references[0]["source_label"] = "different"
    write_json(config.run / "references.json", references)
    before = snapshot(config.run)
    with pytest.raises(ValueError, match="references changed"):
        runner.evaluate(config)
    assert snapshot(config.run) == before
