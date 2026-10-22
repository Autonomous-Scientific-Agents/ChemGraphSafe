"""Pinned sources, conservative grouping, and immutable data freezes."""

import random
import re
import unicodedata
from collections import Counter, defaultdict

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Config
from .models import PROTOCOL_VERSION, Annotation, Example, Request
from .storage import code_hash, digest, file_hash, read_json, read_rows, write_json, write_rows

SOURCE_REPO = "AI45Lab/DeepSafe-Sci"
SOURCE_LICENSE = "https://github.com/AI45Lab/DeepSafe-Sci/blob/main/LICENSE-DATA"
SOURCE_FILES = {
    "benign": (
        "600safe_final.jsonl",
        "024c8ee1c195b04f764c71f7b01c037711a889f25f39b633a90bb74937c46ffe",
    ),
    "harmful": (
        "2400unsafe_final.jsonl",
        "331532208610a5e99dd6ac7b7204bf14ced7404fcc1038abab46fb1c8d6e3922",
    ),
}


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


class SciHazardLoader:
    def load(self, config: Config) -> list[Example]:
        revision = config.data.revision
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Source revision must be an immutable full Git SHA")
        examples = []
        raw = config.dataset / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        for label, (filename, expected) in SOURCE_FILES.items():
            path = raw / filename
            if not path.exists():
                url = (
                    f"https://raw.githubusercontent.com/{SOURCE_REPO}/{revision}/"
                    f"DeHarmScore-trace/dataset/{filename}"
                )
                # Never execute dataset code or load the response/checklist cache.
                with httpx.Client(timeout=60, follow_redirects=True) as client:
                    response = client.get(url)
                    response.raise_for_status()
                if len(response.content) > 25_000_000:
                    raise ValueError("Source file exceeds the download limit")
                import hashlib

                if hashlib.sha256(response.content).hexdigest() != expected:
                    raise ValueError(f"Source checksum mismatch: {filename}")
                path.write_bytes(response.content)
            if file_hash(path) != expected:
                raise ValueError(f"Cached source checksum mismatch: {filename}")
            for index, row in enumerate(read_rows(path)):
                if row.get("subject") != "Chemistry":
                    continue
                question = row.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f"Missing request at {filename}:{index}")
                source_id = str(row.get("idx", index))
                native = row.get("is_harmful")
                if native is not None and not isinstance(native, bool):
                    raise ValueError("Unexpected native harmful label type")
                examples.append(
                    Example(
                        id="scihazard-" + digest([revision, filename, source_id])[:20],
                        request=Request(text=question),
                        source="SciHazard",
                        source_id=source_id,
                        source_revision=revision,
                        source_split=filename,
                        source_label=label,
                        source_native_harmful=native,
                        category=str(row.get("category", "unspecified")),
                        family=str(row["variant_of"]) if row.get("variant_of") else None,
                        license="CC-BY-4.0",
                        hosted_processing_allowed=True,
                        rights_evidence=SOURCE_LICENSE,
                    )
                )
        return examples


class FixtureLoader:
    def load(self, config: Config) -> list[Example]:
        return [Example.model_validate(r) for r in read_rows(config.path(config.data.fixture))]


DATASET_LOADERS = {"scihazard": SciHazardLoader, "fixture": FixtureLoader}


def deduplicate_and_group(examples: list[Example], threshold: float):
    if len({e.id for e in examples}) != len(examples):
        raise ValueError("Duplicate source IDs")
    buckets = defaultdict(list)
    native_conflicts = []
    for e in examples:
        if e.source_native_harmful is not None and e.source_native_harmful != (
            e.source_label == "harmful"
        ):
            native_conflicts.append(e.id)
            continue
        buckets[(normalized(e.request.text), normalized(e.request.context))].append(e)
    unique, quarantine = [], []
    duplicate_count = 0
    for key in sorted(buckets):
        rows = sorted(buckets[key], key=lambda e: e.id)
        if len({e.source_label for e in rows}) != 1:
            quarantine.extend(e.id for e in rows)
            continue
        representative = rows[0].model_copy(deep=True)
        representative.related_source_ids = [e.id for e in rows]
        unique.append(representative)
        duplicate_count += len(rows) - 1
    parent = list(range(len(unique)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    if unique:
        text = [normalized(e.request.text + "\n" + e.request.context) for e in unique]
        try:
            x = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False).fit_transform(
                text
            )
            scores = cosine_similarity(x)
            for i, j in zip(*np.where(np.triu(scores >= threshold, k=1)), strict=True):
                union(int(i), int(j))
        except ValueError as exc:
            if "empty vocabulary" not in str(exc):
                raise
        families = {}
        for i, e in enumerate(unique):
            if e.family:
                key = (e.source, e.source_revision, e.family)
                if key in families:
                    union(i, families[key])
                families[key] = i
    groups = defaultdict(list)
    for i, e in enumerate(unique):
        groups[find(i)].append(e)
    for rows in groups.values():
        group = digest(sorted(e.id for e in rows))[:20]
        for e in rows:
            e.group = group
    stats = {
        "input_count": len(examples),
        "unique_count": len(unique),
        "exact_duplicates_removed": duplicate_count,
        "quarantined_ids": quarantine,
        "native_conflict_ids": sorted(native_conflicts),
        "group_count": len(groups),
        "component_sizes": dict(Counter(str(len(v)) for v in groups.values())),
        "largest_component": max((len(v) for v in groups.values()), default=0),
        "cross_class_groups": sum(len({e.source_label for e in v}) > 1 for v in groups.values()),
    }
    return unique, stats


def select_pilot(pool: list[Example], count: int, minimum: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for e in pool:
        groups[e.group].append(e)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected, used = [], set()
    for group in keys:
        candidates = sorted(
            (e for e in groups[group] if e.source_label == "benign"), key=lambda e: e.id
        )
        if candidates and len(selected) < count // 2:
            selected.append(rng.choice(candidates))
            used.add(group)
    for group in keys:
        if group in used or len(selected) >= count:
            continue
        candidates = sorted(
            (e for e in groups[group] if e.source_label == "harmful"), key=lambda e: e.id
        )
        if candidates:
            selected.append(rng.choice(candidates))
            used.add(group)
    counts = Counter(e.source_label for e in selected)
    if len(selected) != count or any(counts[k] < minimum for k in ("benign", "harmful")):
        raise ValueError(
            f"Insufficient independent cases: selected={len(selected)}, classes={dict(counts)}"
        )
    rng.shuffle(selected)
    return selected


def ensure_unfrozen(config: Config):
    if (config.dataset / "freeze.json").exists():
        raise ValueError("Dataset is frozen; use an explicit new dataset version")


def prepare(config: Config):
    ensure_unfrozen(config)
    manifest_path = config.dataset / "prepare.json"
    spec = config.data.model_dump()
    if config.data.loader == "fixture":
        spec["fixture_hash"] = file_hash(config.path(config.data.fixture))
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old["preparation_hash"] != digest(spec):
            raise ValueError("Preparation changed; use a new dataset directory")
        if old["examples_hash"] != file_hash(config.dataset / "examples.jsonl"):
            raise ValueError("Prepared data changed on disk")
        return old
    if (config.dataset / "annotations.jsonl").exists():
        raise ValueError("Annotations exist without a preparation manifest")
    pool, stats = deduplicate_and_group(
        DATASET_LOADERS[config.data.loader]().load(config), config.data.similarity_threshold
    )
    write_json(config.dataset / "feasibility.json", stats)
    selected = select_pilot(
        pool, config.data.count, config.data.minimum_per_class, config.data.seed
    )
    write_rows(config.dataset / "examples.jsonl", [e.model_dump() for e in selected])
    result = {
        "preparation_hash": digest(spec),
        "preparation": spec,
        "grouping": stats,
        "examples_hash": file_hash(config.dataset / "examples.jsonl"),
        "selected_count": len(selected),
        "selected_groups": len({e.group for e in selected}),
        "classes": dict(Counter(e.source_label for e in selected)),
        "source_checksums": SOURCE_FILES if config.data.loader == "scihazard" else {},
        "source_revision": config.data.revision,
        "mock": config.offline,
    }
    write_json(manifest_path, result)
    return result


def examples(config: Config) -> list[Example]:
    return [Example.model_validate(r) for r in read_rows(config.dataset / "examples.jsonl")]


def annotations(config: Config) -> dict[str, Annotation]:
    rows = [Annotation.model_validate(r) for r in read_rows(config.dataset / "annotations.jsonl")]
    if len({r.example_id for r in rows}) != len(rows):
        raise ValueError("Duplicate annotation IDs")
    return {r.example_id: r for r in rows}


def freeze(config: Config):
    ensure_unfrozen(config)
    prepared = read_json(config.dataset / "prepare.json")
    if file_hash(config.dataset / "examples.jsonl") != prepared["examples_hash"]:
        raise ValueError("Prepared examples changed")
    rows = examples(config)
    if not rows:
        raise ValueError("No prepared examples")
    ann = annotations(config)
    for annotation in ann.values():
        annotation.prediction.validate_policy(config.policy)
    if set(ann) - {e.id for e in rows}:
        raise ValueError("Annotation IDs outside dataset")
    if any(a.policy_hash != digest(config.policy) for a in ann.values()):
        raise ValueError("Annotations refer to a different policy")
    if ann and (
        config.curator is None
        or any(
            a.config_hash != digest(config.curator.spec())
            or a.requested_model != config.curator.model
            for a in ann.values()
        )
    ):
        raise ValueError("Annotations refer to a different curator configuration")
    rng = random.Random(config.data.seed)
    splits = {}
    for label in ("benign", "harmful"):
        ids = sorted(e.id for e in rows if e.source_label == label)
        rng.shuffle(ids)
        n = max(1, round(len(ids) * 0.2))
        if n >= len(ids):
            raise ValueError("Both classes must occur in dev and evaluation")
        splits.update({id_: "dev" if i < n else "eval" for i, id_ in enumerate(ids)})
    order = sorted(e.id for e in rows)
    rng.shuffle(order)
    split_manifest = {"splits": splits, "case_order": order}
    if config.data.inherited_freeze:
        inherited = read_json(config.path(config.data.inherited_freeze))
        if (
            digest({k: v for k, v in inherited.items() if k != "freeze_hash"})
            != inherited["freeze_hash"]
        ):
            raise ValueError("Inherited freeze manifest changed")
        if inherited["examples_hash"] != prepared["examples_hash"]:
            raise ValueError("Inherited freeze has different examples")
        split_manifest = inherited["split_manifest"]
        if (
            len(split_manifest["case_order"]) != len(order)
            or set(split_manifest["case_order"]) != set(order)
            or set(split_manifest["splits"]) != set(order)
            or set(split_manifest["splits"].values()) != {"dev", "eval"}
        ):
            raise ValueError("Inherited case IDs differ")
    split_groups = defaultdict(set)
    for e in rows:
        split_groups[e.group].add(split_manifest["splits"][e.id])
    if any(len(v) > 1 for v in split_groups.values()):
        raise ValueError("Related cases cross split boundary")
    path = config.dataset / "annotations.jsonl"
    result = {
        "examples_hash": prepared["examples_hash"],
        "annotations_hash": file_hash(path) if path.exists() else None,
        "prepare_hash": file_hash(config.dataset / "prepare.json"),
        "split_manifest": split_manifest,
        "split_hash": digest(split_manifest),
        "semantic_config": config.semantic_config(),
        "semantic_hash": config.semantic_hash(),
        "protocol": PROTOCOL_VERSION,
        "code_provenance": code_hash(),
        "annotation_count": len(ann),
        "mock": config.offline,
        "group_count": len(split_groups),
    }
    result["freeze_hash"] = digest(result)
    write_json(config.dataset / "freeze.json", result)
    return result


def verify_freeze(config: Config):
    result = read_json(config.dataset / "freeze.json")
    body = {k: v for k, v in result.items() if k != "freeze_hash"}
    if digest(body) != result["freeze_hash"]:
        raise ValueError("Freeze manifest changed")
    for filename, key in (("examples.jsonl", "examples_hash"), ("prepare.json", "prepare_hash")):
        if file_hash(config.dataset / filename) != result[key]:
            raise ValueError(f"Frozen artifact changed: {filename}")
    ann = config.dataset / "annotations.jsonl"
    if (file_hash(ann) if ann.exists() else None) != result["annotations_hash"]:
        raise ValueError("Frozen annotations changed (including previously absent annotations)")
    if result["semantic_hash"] != config.semantic_hash() or result["protocol"] != PROTOCOL_VERSION:
        raise ValueError("Frozen policy/model/protocol changed; create a new version")
    return result
