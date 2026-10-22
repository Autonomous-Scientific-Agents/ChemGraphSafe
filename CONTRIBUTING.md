# Contributing

Start with Python 3.12 and `uv sync --locked --extra dev`. Run the offline example in
the README before selecting a [contributor track](docs/contributor-tracks.md).
Keep changes scoped to one loader, adapter, metric, or curation improvement.

## Extension points

- `models.py`: `DatasetLoader`, `DecisionModel`, and `Labeler` protocols and validated records.
- `data.py`: implement `load(config) -> list[Example]`, register the loader in
  `DATASET_LOADERS`, and extend `DataConfig.loader`. Include provenance and processing rights.
- `adapters.py`: implement `predict(Request, policy) -> Prediction` and `close()`, register it
  in `build_model`, and extend `ModelConfig.adapter`. Only request text/context reaches a model.
  Argo shares the HTTP transport; `codex.py` lazily loads the optional SDK. See
  [LLM adapter setup](docs/llm-adapters.md) for their authentication and accounting contracts.
- `prompts.py` holds shared classification instructions and Codex's freeze-level prompt
  fingerprint. `provenance.py` collects installed runtime dependencies without dev extras.
  Codex constructors expose `runtime_provenance` after no-inference preflight. The runner binds
  the finalized `run_key` to live adapters before prediction; construction must not spend attempts.
- `reporting.py`: add metrics with explicit denominators, missing-data behavior, and reference
  provenance. Use successful decisions, native abstentions, and failures in regression cases.

These are deliberately small extension points. No plugin service or GPU dependency is required.
See [SciSafetyBench's implemented loader contract](docs/scisafetybench-loader.md) for an example
of a pinned challenge set with nullable native labels and separate rule-derived references.

## Checks for a contribution

```sh
uv run ruff check .
uv run ruff format --check src tests
uv run pytest -q
```

Add tests for behavior that could invalidate an experiment: provider contract parsing, label
leakage, source provenance, grouping, denominators, budget handling, or resume identity.
CI uses harmless fixtures and mock HTTP only. Keep keys in your shell environment; `.env.example`
is documentation, and the program does not automatically load `.env` files.

For Codex changes, also run `uv sync --locked --extra dev --extra codex` followed by
`uv run --extra codex pytest -q tests/test_codex_adapter.py`. This exercises the locked SDK's
actual notification types without launching its runtime or making inference requests. The rest
of the Codex tests use an async fake SDK and also run without the optional dependency installed.
For dependency updates, follow the [SDK upgrade workflow](docs/llm-adapters.md#codex-subscription)
and retain the tested SDK/runtime versions in `uv.lock`.

Do not commit raw downloads, annotations, request text from safety datasets, API responses, or
local run directories. Share aggregate results and ID-only errors after checking source terms.
Downloaded data retain their own licenses. New source integrations must document access,
redistribution, and hosted processing rights before being enabled.

## Experiment discipline

Use new data/run directories for changed experiments; never delete a freeze to make a check pass.
Keep the same budget ledger across live reruns. Do not train on the current development/evaluation
groups. Errors inspected on evaluation cases and their derivatives remain evaluation-only.
Provisional annotations need an independent audit before they become trusted training labels.

Open a contribution issue using the repository template with the intended output and proof
command. Coordinate policy changes separately, since they invalidate comparisons.
