"""Canonical hashes, atomic artifacts, and exclusive local workflow locks."""

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def canonical(value) -> str:
    """Deterministic JSON serialization: sorted keys, no whitespace, no NaN/Inf.

    Used as the pre-image for all content hashes so that dict key order never
    changes a hash. Any non-finite float raises ``ValueError``.
    """
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value) -> str:
    """Return the SHA-256 hex digest of the canonical JSON form of ``value``."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    """Read and parse a JSON file."""
    return json.loads(path.read_text())


def write_json(path: Path, value):
    """Write ``value`` as pretty-printed JSON, atomically."""
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_text(path: Path, body: str):
    """Write ``body`` to ``path`` atomically via a temp file and ``os.replace``.

    Creates parent directories if needed. ``fsync`` before rename ensures the
    content reaches disk even on an abrupt shutdown.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_rows(path: Path) -> list[dict]:
    """Read a newline-delimited JSON file. Returns an empty list if the file does not exist."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path: Path, rows):
    """Write a list of dicts as newline-delimited JSON, atomically.

    Each row is serialized with ``canonical`` so hash-based resume keys are stable
    across Python interpreter restarts.
    """
    atomic_text(path, "".join(canonical(r) + "\n" for r in rows))


@contextmanager
def exclusive(path: Path):
    """Non-blocking POSIX ``flock`` that raises ``ValueError`` if already held.

    Used by the CLI to prevent concurrent prepare/freeze/evaluate invocations
    from racing on the same dataset or run directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another workflow is writing this artifact directory") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def code_hash() -> str:
    """Hash all .py files in the package directory.

    Included in run manifests so that any change to inference code produces a new
    run identity and prevents stale-code results from being silently resumed.
    """
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})
