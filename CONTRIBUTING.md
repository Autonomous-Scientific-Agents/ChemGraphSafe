# Contributing

## Before you start

1. Install Python 3.12 and [uv](https://docs.astral.sh/uv/).
2. Run `uv sync --locked --extra dev` and then the offline example in the README to verify your
   environment.
3. Pick a [contributor track](docs/contributor-tracks.md) and claim it with a GitHub issue before
   opening a pull request.

Keep changes scoped to one loader, adapter, metric, or curation improvement per pull request.

---

## Quick reference — where things live

| Topic | File |
|---|---|
| Data records and extension protocols | [`src/chemgraphsafe/models.py`](src/chemgraphsafe/models.py) |
| TOML config schema | [`src/chemgraphsafe/config.py`](src/chemgraphsafe/config.py) |
| Dataset loaders | [`src/chemgraphsafe/data.py`](src/chemgraphsafe/data.py) |
| Model adapters | [`src/chemgraphsafe/adapters.py`](src/chemgraphsafe/adapters.py) |
| Evaluation / smoke / curate workflows | [`src/chemgraphsafe/runner.py`](src/chemgraphsafe/runner.py) |
| Metrics and report generation | [`src/chemgraphsafe/reporting.py`](src/chemgraphsafe/reporting.py) |
| Budget ledger | [`src/chemgraphsafe/budget.py`](src/chemgraphsafe/budget.py) |
| LLM adapter auth and accounting | [`docs/llm-adapters.md`](docs/llm-adapters.md) |
| Dataset provenance | [`docs/dataset-card.md`](docs/dataset-card.md) |
| Freeze / run versioning rules | [`docs/evaluation.md`](docs/evaluation.md) |
| SciSafetyBench loader contract | [`docs/scisafetybench-loader.md`](docs/scisafetybench-loader.md) |
| Contributor tracks | [`docs/contributor-tracks.md`](docs/contributor-tracks.md) |

---

## Extension points

### Adding a new dataset loader
1. Create a class with `load(config: Config) -> list[Example]` in `data.py`.
2. Register it in `DATASET_LOADERS` with a string key.
3. Extend `DataConfig.loader` in `config.py` with the new key.
4. Include provenance fields (`source`, `source_revision`, `license`, `rights_evidence`) and
   ensure `hosted_processing_allowed` is set correctly.
5. See [`docs/scisafetybench-loader.md`](docs/scisafetybench-loader.md) for a concrete contract
   example.

### Adding a new model adapter
1. Create a class with `predict(request: Request, policy: dict) -> Prediction` and `close()` in
   `adapters.py` (or a dedicated module for large adapters like `codex.py`).
2. Register it in `build_model` and extend `ModelConfig.adapter` in `config.py`.
3. Only `request.text` and `request.context` may reach the model — never source IDs, labels,
   or annotations. See [`docs/llm-adapters.md`](docs/llm-adapters.md) for the full contract.
4. Classification prompts live in `prompts.py`; runtime dependency fingerprints live in
   `provenance.py`. Codex adapters expose `runtime_provenance` after a no-inference preflight.
   The runner binds `run_key` to live adapters before prediction; construction must not spend attempts.

### Adding new metrics
- Add them to `reporting.py` with explicit denominators and documented missing-data behavior.
- Use successful decisions, native abstentions, and failures as separate categories.
- See [`docs/verification.md`](docs/verification.md) for the denominator contracts.

---

## Checks for a contribution

All three must pass before opening a pull request:

```sh
uv run ruff check .
uv run ruff format --check src tests
uv run pytest -q
```

For Codex adapter changes, also run:

```sh
uv sync --locked --extra dev --extra codex
uv run --extra codex pytest -q tests/test_codex_adapter.py
```

This exercises the locked SDK's actual notification types without launching its runtime or making
inference requests. The rest of the Codex tests use an async fake SDK and run without the optional
dependency. For SDK updates, follow the
[SDK upgrade workflow](docs/llm-adapters.md#codex-subscription) and retain the tested SDK/runtime
versions in `uv.lock`.

**Write tests for anything that could invalidate an experiment:** provider contract parsing, label
leakage, source provenance, grouping denominators, budget handling, or resume identity.

CI uses harmless fixtures and mock HTTP only. Keep credentials in your shell environment; the
program never loads `.env` files automatically.

---

## Experiment discipline

- Use new data directories for changed dataset/policy/prompt/model configurations; never delete a
  freeze to make a check pass.
- Changed inference code requires a new run directory (the run identity includes a code hash).
- Do not train on the current development or evaluation groups.
- Errors inspected on evaluation cases and their derivatives remain evaluation-only for that version.
- Provisional LLM annotations need an independent audit before they become trusted training labels.

See [`docs/evaluation.md`](docs/evaluation.md) for the complete versioning and immutability rules.

---

## What not to commit

- Raw downloads, API responses, annotations, or run artifacts (`runs/`, `.local/`, `data/raw/`).
- Safety-dataset request text — share aggregate metrics and ID-only errors instead.
- Credentials or `.env` files.

New source integrations must document access, redistribution, and hosted-processing rights in their
loader before they can be enabled. Downloaded data retain their own licenses.

---

Open a contribution issue using the [repository template](https://github.com/Autonomous-Scientific-Agents/ChemGraphSafe/issues/new/choose)
with the intended output and proof command. Coordinate policy changes separately, since they
invalidate comparisons across models.
