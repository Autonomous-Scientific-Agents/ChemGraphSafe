# ChemGraphSafe

An extensible research starter for evaluating chemistry **request triage**: ALLOW, REVIEW, or BLOCK.
Adapters support **Jev through OpenRouter's Decisions API**, **hosted Argo**, **Codex with a
ChatGPT login**, and configurable OpenAI-compatible chat endpoints. Argo and Codex can classify
requests or generate provisional curation labels. Their contracts are tested offline; live
measurements are a separate step.

**Project website:** <https://autonomous-scientific-agents.github.io/ChemGraphSafe/>

ChemGraphSafe is a project for the [Open Scientific Intelligence (OSI) Hackathon](https://llmhackathon.github.io/),
October 21–22, 2026, formerly the LLM Hackathon for Applications in Materials Science & Chemistry.
The longer-term goal is a safety layer for the [ChemGraph](https://github.com/argonne-lcf/ChemGraph)
agent. This milestone covers request triage only: no response evaluation, agent/tool execution,
training, or production enforcement. See [Join the project](#join-the-project) to contribute.

## Run the offline example

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Tested on macOS/Linux; local workflow
locking currently uses POSIX `flock`.

```sh
uv sync --locked --extra dev
uv run chemgraphsafe evaluate --config configs/offline.toml
uv run chemgraphsafe report --run runs/offline
```

Open `runs/offline/report.md`. The owned, harmless fixtures have **synthetic reference labels and
replayed predictions**. All outputs explicitly say MOCK/OFFLINE; they are not model measurements.
The first evaluation prepares a local fixture freeze. Subsequent identical invocations resume.

## Prepare the real pilot

```sh
uv run chemgraphsafe data prepare --config configs/pilot.toml
uv run chemgraphsafe data freeze --config configs/pilot.toml
```

Preparation downloads the pinned [SciHazard release](https://github.com/AI45Lab/DeepSafe-Sci),
checks both SHA256 values, keeps Chemistry requests, deduplicates and groups similar requests,
then samples 100 independent groups (target 50 benign / 50 harmful, minimum 40 of each).
It writes grouping and feasibility statistics before enforcing the size gate.

The freeze preserves source labels, approximately 20 development / 80 evaluation cases, and a
seeded case order. Three-way policy metrics are unavailable until the separate annotation step.
Raw source answers and reasoning are never sent to classifiers. Source labels are references,
not a claim of independently validated ChemGraphSafe ground truth.

## Run Jev when credentials are available

Set `OPENROUTER_API_KEY` locally, verify the prices in `configs/pilot.toml`, then:

```sh
uv run chemgraphsafe smoke --config configs/pilot.toml
uv run chemgraphsafe evaluate --config configs/pilot.toml
uv run chemgraphsafe report --run runs/pilot
```

Jev uses `https://openrouter.ai/api/alpha/decisions`, not chat completions. The model ID is explicit.
See [OpenRouter's official Jev guide](https://openrouter.ai/blog/insights/what-is-jev/).
No inference is attempted when credentials are missing. The smoke test uses five owned harmless
examples outside the pilot. Its API response contract is tested locally with mocked HTTP transport;
mock tests do not establish live-provider behavior.

All live commands share `.local/state/pilot-budget.sqlite`: **$10 and 500 attempts total**, including
retries, curation, smoke, and versioned reruns. Unknown charges retain their reservation. Budget
ceilings are estimates based on configured prices; an actual charge above its reservation stops
the run and is disclosed. Budgets may prevent a full rerun. Missing rates fail before inference.

## Use Argo or Codex

For hosted Argo, export `ARGO_USER` with your Argonne username and use `configs/argo.toml`.
The endpoint must be reachable from your network. The example selects `argo:gpt-4.1-mini`.

For Codex, install `uv sync --locked --extra dev --extra codex`, establish a ChatGPT login with
`codex login`, and replace the model placeholder in `configs/codex.toml` with an account-supported
model ID. Set `reasoning_effort` explicitly for each Codex model; preflight checks model/effort
support without starting an inference turn. The SDK bundles its runtime; the CLI is needed only
if you use it to establish login. The SDK accepts newer releases within its configured version
range; `uv.lock` records the tested SDK/runtime pair. See [SDK upgrades](docs/llm-adapters.md#codex-subscription)
before updating it. Existing Codex experiments require new data/run directories
for the corrected prompt contract; see [evaluation versioning](docs/evaluation.md).

```sh
uv run chemgraphsafe smoke --config configs/argo.toml
# After selecting the Codex model and signing in:
uv run --extra codex chemgraphsafe smoke --config configs/codex.toml
```

These commands perform live inference. Both examples explicitly use `budget_mode="attempts_only"`:
every attempt shares the 500-attempt cap, but subscription/institutional usage has no dollar
estimate and does not consume the metered $10 allowance. See [LLM setup](docs/llm-adapters.md)
for curation, generic endpoints, accounting, and versioned evaluation instructions.

## Interpret the report

- Primary: source-label interception recall together with benign BLOCK/REVIEW rates and burden.
- Native provider refusals are ABSTAIN, separate from policy REVIEW. Interception lower/upper
  bounds exclude/include abstentions; operational errors count as misses in both bounds.
- Decided-only precision/F1/F2 explicitly report their denominator. All intended cases remain in
  primary rates. Always-ALLOW and always-REVIEW diagnostics prevent misleading recall claims.
- Secondary: agreement with provisional policy labels, when present. No confidence or probability
  is invented for label-only LLMs. Calibration analysis is a contributor task.
- `metrics.json` has Wilson intervals, coverage, failures, latency, cost and source/category slices.
  `comparison.csv` is the compact evaluation comparison. `errors.jsonl` and `errors.csv` contain
  IDs and fixed issue codes; join locally to the ignored dataset to inspect request text.

## Join the project

1. Register for the hackathon at [luma.com/ku88xh92](https://luma.com/ku88xh92) and join the
   hackathon Slack from the [hackathon homepage](https://llmhackathon.github.io/).
2. Introduce yourself in a [GitHub issue](https://github.com/Autonomous-Scientific-Agents/ChemGraphSafe/issues)
   or on the hackathon Slack, and name the track you are interested in.
3. Run the offline example above, then read [CONTRIBUTING.md](CONTRIBUTING.md) and
   [the contributor tracks](docs/contributor-tracks.md).
4. Claim one bounded item with the
   [contribution issue template](https://github.com/Autonomous-Scientific-Agents/ChemGraphSafe/issues/new/choose).
   State the track, the observable deliverable, and the proof command.
5. Open a pull request that passes the checks below.

No GPU is needed except for the fine-tuning track. Do not paste safety-dataset request text into
issues, pull requests, or Slack; share aggregate metrics and ID-only errors.

## Extend the project

Start with [CONTRIBUTING.md](CONTRIBUTING.md) and [the contributor tracks](docs/contributor-tracks.md).
See [dataset provenance](docs/dataset-card.md), [evaluation and versioning](docs/evaluation.md),
and [model configuration](docs/model-card.md). The implementation plan and independent review
record are in `PLAN.md` and `PLAN-REVIEW-LOG.md`.

```sh
uv run ruff check .
uv run pytest -q
```

Code is Apache-2.0. Downloaded datasets retain their source licenses and attribution. Keep raw
datasets, local annotations, credentials, and run artifacts out of Git. This is a preliminary
research tool, not a production safety gate or an evaluation of chemistry knowledge.
