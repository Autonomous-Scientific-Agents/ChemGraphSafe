# ChemGraphSafe

Python 3.12 (uv, hatchling, `src/chemgraphsafe`) CLI for evaluating chemistry request triage (ALLOW/REVIEW/BLOCK). Use `uv`, never pip/conda.

## Commands (CI order)

```sh
uv sync --locked --extra dev
uv run ruff check .
uv run ruff format --check src tests     # format is checked only for src/ tests/
uv run pytest -q                         # single: uv run pytest -q tests/test_workflows.py -k name
uv run chemgraphsafe evaluate --config configs/offline.toml
uv run chemgraphsafe report --run runs/offline
```
- Codex adapter tests need the optional SDK resolved in `uv.lock`: `uv sync --locked --extra dev --extra codex && uv run --extra codex pytest -q tests/test_codex_adapter.py`. Other Codex tests use a fake SDK (`tests/conftest.py`) and run without it. Upgrade the SDK and bundled runtime together and rerun the contract tests; see `docs/llm-adapters.md`.
- Ruff line length is 100; rules E,F,I,B.
- CLI subcommands: `data prepare`, `data freeze`, `curate`, `smoke`, `evaluate`, `report` (`src/chemgraphsafe/cli.py`).

## Layout / extension points

- `models.py` protocols (`DatasetLoader`, `DecisionModel`, `Labeler`) + validated records; `config.py` TOML schema; `runner.py` evaluate/resume; `budget.py` SQLite ledger; `reporting.py` metrics; `storage.py` locks/atomic writes.
- New loader: register in `DATASET_LOADERS` (`data.py`) and extend `DataConfig.loader`. New model adapter: register in `build_model` (`adapters.py`) and extend `ModelConfig.adapter`. Details in `CONTRIBUTING.md`.
- Config paths resolve against `project_root` (relative to the TOML file's directory, default `..`), not cwd.

## Gotchas

- Live commands (`smoke`, live `evaluate`, `curate`) make paid/real inference and share one ledger `.local/state/pilot-budget.sqlite` ($10 / 500 attempts). Never run them without explicit user request; tests and CI use fixtures and mock HTTP only.
- Env vars are NOT loaded from `.env`; export them in the shell (`OPENROUTER_API_KEY`, `ARGO_USER`, `CHEMGRAPHSAFE_LLM_API_KEY`). Codex uses `codex login`.
- Only the offline config auto-prepares/freezes. Live runs require `data prepare` -> (optional `curate`) -> `data freeze` first.
- Freezes and run identities are immutable: never delete/overwrite a freeze or edit artifacts to make a check pass. Changed data/policy/prompts/model config => new data dir + freeze; changed inference code => new run dir (run identity includes code hash). See `docs/evaluation.md`.
- Locking uses POSIX `flock` (Linux/macOS only).
- Never put labels, example IDs, source metadata, or source answers into model inputs (label-leakage is tested). Prediction rows store IDs/decisions, not request text.
- Do not commit `runs/`, `.local/`, `data/raw|derived/`, `.env*`, raw dataset requests, or API responses (all gitignored). Fixtures in `fixtures/` must stay owned, harmless examples.
- Add tests for anything that could invalidate an experiment: provider parsing, leakage, provenance, grouping, denominators, budget, resume identity.
- `PLAN.md` is the original design spec (long); `docs/` holds the per-topic contracts.
