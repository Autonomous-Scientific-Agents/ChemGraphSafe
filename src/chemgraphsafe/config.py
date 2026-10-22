"""TOML paths resolve against a declared root, independently of the invoking cwd."""

import tomllib
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from .models import Record
from .prompts import codex_prompt_contract
from .providers import ARGO_ENDPOINT, ARGO_WITHOUT_TEMPERATURE, argo_model, model_identity
from .storage import digest, read_json


class DataConfig(Record):
    """Controls which source to load, how to group near-duplicates, and the pilot sample size.

    ``directory`` is the mutable working directory before freeze; once frozen it becomes
    immutable and a new directory is required for any data/policy change.
    ``inherited_freeze`` lets a new data version reuse the split order from a previous freeze,
    keeping case ordering identical across minor data fixes.
    """

    loader: Literal["scihazard", "fixture"] = "scihazard"
    directory: str = ".local/data/pilot-v1"
    fixture: str = "fixtures/examples.jsonl"
    revision: str = "f8809e5af23495e469dfc53e99973f74031fcb11"
    count: int = Field(default=100, ge=4)
    minimum_per_class: int = Field(default=40, ge=2)
    seed: int = 42
    similarity_threshold: float = Field(default=0.9, gt=0, le=1)
    inherited_freeze: str | None = None


class ModelConfig(Record):
    """Configuration for one model adapter in an evaluation run.

    ``adapter`` selects the implementation: ``decisions`` for Jev via OpenRouter,
    ``chat`` for any OpenAI-compatible endpoint, ``argo`` for hosted Argonne models,
    ``codex`` for ChatGPT-login inference, and ``replay`` for offline fixture playback.

    ``budget_mode="attempts_only"`` skips dollar reservations; use it for subscription
    accounts (Argo, Codex) where per-token pricing is not applicable.
    Dollar pricing fields (``input_usd_per_million``, etc.) and ``reservation_usd``
    are rejected outright when ``budget_mode="attempts_only"``.
    """

    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    adapter: Literal["decisions", "chat", "replay", "argo", "codex"]
    model: str = Field(min_length=1)
    endpoint: str = ""
    api_key_env: str = ""
    argo_user_env: str = "ARGO_USER"
    budget_mode: Literal["usd", "attempts_only"] = "usd"
    replay_file: str = "fixtures/predictions.json"
    input_usd_per_million: float | None = Field(default=None, ge=0)
    output_usd_per_million: float | None = Field(default=None, ge=0)
    per_request_usd: float = Field(default=0, ge=0)
    pricing_checked: str | None = None
    # Conservative ceiling that also covers provider-side hidden overhead beyond token counts.
    reservation_usd: float | None = Field(default=None, gt=0)
    max_input_bytes: int = Field(default=32000, ge=1)
    max_output_tokens: int = Field(default=256, ge=1)
    temperature: float | None = None
    seed: int | None = None
    reasoning_effort: str | None = None

    @model_validator(mode="after")
    def endpoint_safe(self):
        if self.name in {"always-allow", "always-review"}:
            raise ValueError("Model name is reserved for a diagnostic baseline")
        if self.adapter == "codex":
            if not self.reasoning_effort or self.reasoning_effort != self.reasoning_effort.strip():
                raise ValueError("Codex requires an explicit nonempty reasoning_effort")
            if not self.model.startswith("codex:") or not self.model[6:].strip():
                raise ValueError("Codex requires codex:<model-id>")
            if self.model[6:] != self.model[6:].strip():
                raise ValueError("Codex model ID cannot contain surrounding whitespace")
            if self.endpoint or self.api_key_env:
                raise ValueError("Codex uses the stored ChatGPT login, not an endpoint/API key")
            if self.budget_mode != "attempts_only":
                raise ValueError("Codex subscription requires budget_mode='attempts_only'")
            if self.temperature is not None or self.seed is not None:
                raise ValueError("Codex does not support temperature or seed")
        elif self.reasoning_effort is not None:
            raise ValueError("reasoning_effort is supported only by Codex")
        if self.adapter == "argo":
            argo_model(self.model)
            if self.model in ARGO_WITHOUT_TEMPERATURE and self.temperature is not None:
                raise ValueError("This Argo model does not support temperature")
            self.endpoint = self.endpoint or ARGO_ENDPOINT
            if not self.argo_user_env or self.api_key_env:
                raise ValueError("Argo requires argo_user_env, not an API key")
        if self.adapter in {"decisions", "chat", "argo"}:
            u = urlsplit(self.endpoint)
            if (
                u.scheme != "https"
                or not u.hostname
                or u.username
                or u.password
                or u.query
                or u.fragment
            ):
                raise ValueError(
                    "Live endpoints must be HTTPS without credentials/query parameters"
                )
            if self.adapter != "argo" and not self.api_key_env:
                raise ValueError("Live adapters need an API key environment-variable name")
        if self.budget_mode == "attempts_only" and (
            self.input_usd_per_million is not None
            or self.output_usd_per_million is not None
            or self.reservation_usd is not None
            or self.per_request_usd
            or self.pricing_checked
        ):
            raise ValueError("Attempts-only accounting cannot include dollar pricing/reservations")
        return self

    def spec(self) -> dict:
        """Keep legacy semantic hashes stable when new provider options are unused."""
        result = self.model_dump()
        if self.adapter != "codex":
            result.pop("reasoning_effort")
        if self.adapter != "argo":
            result.pop("argo_user_env")
        if self.budget_mode == "usd":
            result.pop("budget_mode")
        return result


class BudgetConfig(Record):
    """Shared spend ceiling applied across all live commands (smoke, curate, evaluate).

    The ledger path resolves relative to ``project_root``, not cwd, so all commands
    that share a project share the same ledger regardless of where they are invoked.
    """

    ledger: str = ".local/state/pilot-budget.sqlite"
    max_usd: float = Field(default=10, gt=0)
    max_attempts: int = Field(default=500, ge=1)


class RunnerConfig(Record):
    """HTTP retry and timeout parameters applied uniformly to every live adapter."""

    timeout_seconds: float = Field(default=60, gt=0)
    retries: int = Field(default=2, ge=0, le=2)
    max_retry_after_seconds: float = Field(default=60, ge=0, le=60)


class Config(Record):
    """Root configuration object, parsed from a TOML file by ``load_config``.

    All relative paths in sub-configs resolve against ``project_root``, which itself
    resolves relative to the TOML file's own directory. This makes configs portable:
    moving the repo does not break path resolution as long as the TOML stays inside it.
    """

    project_root: str = ".."
    policy_file: str = "policies/triage-v0.1.json"
    run_dir: str = "runs/pilot"
    offline: bool = False
    data: DataConfig = Field(default_factory=DataConfig)
    models: list[ModelConfig]
    curator: ModelConfig | None = None
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    root: Path = Field(default=Path("."), exclude=True)

    @model_validator(mode="after")
    def distinct(self):
        names = [m.name for m in self.models]
        if len(names) != len(set(names)) or not names:
            raise ValueError("Provide uniquely named models")
        if self.offline and any(m.adapter != "replay" for m in self.models):
            raise ValueError("Offline config cannot invoke live adapters")
        if self.offline and self.curator and self.curator.adapter != "replay":
            raise ValueError("Offline config cannot invoke a live curator")
        if self.curator and any(
            model_identity(self.curator.model) == model_identity(m.model) for m in self.models
        ):
            raise ValueError("Curator must differ from evaluated model IDs")
        if not self.offline and any(m.adapter == "replay" for m in self.models):
            raise ValueError("Live comparison cannot include replay scores")
        return self

    def path(self, name: str) -> Path:
        """Resolve a config-relative path against project_root, or return it as-is if absolute."""
        p = Path(name)
        return (self.root / p).resolve() if not p.is_absolute() else p.resolve()

    @property
    def dataset(self) -> Path:
        return self.path(self.data.directory)

    @property
    def run(self) -> Path:
        return self.path(self.run_dir)

    @property
    def policy(self) -> dict:
        p = read_json(self.path(self.policy_file))
        if set(p.get("criteria", {})) != {"ALLOW", "REVIEW", "BLOCK"}:
            raise ValueError("Policy must define ALLOW/REVIEW/BLOCK")
        return p

    def semantic_config(self) -> dict:
        result = {
            "models": [m.spec() for m in self.models],
            "curator": self.curator.spec() if self.curator else None,
            "policy": self.policy,
            "offline": self.offline,
        }
        contracts = {}
        if any(m.adapter == "codex" for m in self.models):
            contracts["evaluation"] = codex_prompt_contract(self.policy)
        if self.curator and self.curator.adapter == "codex":
            contracts["curation"] = codex_prompt_contract(self.policy, curator=True)
        if contracts:
            result["codex_prompt_contracts"] = contracts
        return result

    def semantic_hash(self) -> str:
        return digest(self.semantic_config())


def load_config(path: Path) -> Config:
    """Parse a TOML config file and pin ``root`` to the directory containing the file.

    Raises ``ValueError`` if the run and dataset directories overlap — a safety check
    that prevents an evaluate command from accidentally overwriting frozen data.
    """
    path = path.resolve()
    c = Config.model_validate(tomllib.loads(path.read_text()))
    c.root = (path.parent / c.project_root).resolve()
    if c.run == c.dataset or c.run in c.dataset.parents or c.dataset in c.run.parents:
        raise ValueError("Run and dataset directories must be separate")
    return c
