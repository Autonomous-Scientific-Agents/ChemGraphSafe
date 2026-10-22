"""Non-secret execution provenance, independent of a dataset freeze or run directory."""

import platform
from importlib.metadata import distribution

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

EXECUTION_VERSION = 2


def environment_identity(*, codex: bool = False) -> dict:
    """Resolve installed runtime dependencies, excluding inactive extras and dev tooling."""
    pending = [("chemgraphsafe", frozenset({"codex"}) if codex else frozenset())]
    visited, versions = set(), {}
    while pending:
        name, extras = pending.pop()
        name = canonicalize_name(name)
        if (name, extras) in visited:
            continue
        visited.add((name, extras))
        package = distribution(name)
        versions[name] = package.version
        for raw in package.requires or []:
            requirement = Requirement(raw)
            if requirement.marker is None or any(
                requirement.marker.evaluate({"extra": extra}) for extra in ("", *sorted(extras))
            ):
                pending.append((requirement.name, frozenset(requirement.extras)))
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": dict(sorted(versions.items())),
    }
