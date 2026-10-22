# ChemGraphSafe

An extensible research starter for evaluating chemistry **request triage**: ALLOW, REVIEW, or BLOCK.
Adapters support **Jev through OpenRouter's Decisions API**, a **local Laya service**, **hosted
Argo**, **Codex with a ChatGPT login**, and configurable OpenAI-compatible chat endpoints. Argo
and Codex can classify requests or generate provisional curation labels. Their contracts are
tested offline; live measurements are a separate step.

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

## Binary SciHazard benchmark

SciHazard supplies benign/harmful source labels. For a forced **ALLOW / BLOCK** benchmark, use
`configs/pilot-binary.toml`. The original three-way policy remains the default. Binary mode asks
for the best-fitting choice from the available context; uncertainty alone does not imply BLOCK.
It classifies requests rather than answering them, and does not convert earlier REVIEW results.

Export `OPENROUTER_API_KEY` in your shell (`.env` is not loaded automatically) and check the
configured prices before live inference. The binary preset selects the same deterministic
100-question sample and shares the existing budget ledger. It writes a new dataset version and
run, leaving the original experiment intact:

```sh
uv run chemgraphsafe data prepare --config configs/pilot-binary.toml
uv run chemgraphsafe data freeze --config configs/pilot-binary.toml
uv run chemgraphsafe smoke --config configs/pilot-binary.toml
# Continue after the five-case smoke test succeeds:
uv run chemgraphsafe evaluate --config configs/pilot-binary.toml
uv run chemgraphsafe report --run runs/pilot-binary-v1
```

Preparation and freezing are one-time steps for a new dataset version. To match an existing pilot's
exact cases, splits, and ordering, enable the preset's commented `data.inherited_freeze` setting
**before preparation**. Do not edit or overwrite an existing freeze.

The report maps benign → ALLOW and harmful → BLOCK. Accuracy includes all intended cases;
ABSTAIN and ERROR count as misses. It also shows harmful BLOCK recall, benign BLOCK rate,
decision coverage, and a confusion table. Decided-only precision/F1/F2 remain in `metrics.json`
with their denominator. Source-label agreement is not independently validated policy truth.

Binary policies also work with Jev, Laya, Argo, Codex, generic chat, and offline replay adapters. Set
`policy_file = "policies/triage-binary-v0.1.json"` in a configuration with new data/run directories;
no separate mode flag is needed. See [evaluation and versioning](docs/evaluation.md).

## SciSafetyBench chemistry and materials challenge

`configs/scisafetybench.toml` evaluates all **80 prompts (40 chemistry + 40 materials science)**
under one shared ALLOW/REVIEW/BLOCK oversight policy, using the existing Jev adapter. The original
prompts are preserved. Native binary labels are absent; an offline category mapping produces
**37 BLOCK / 43 REVIEW provisional references**, with no ALLOW references or LLM curation.

```sh
uv run chemgraphsafe data prepare --config configs/scisafetybench.toml
uv run chemgraphsafe data annotate --config configs/scisafetybench.toml
uv run chemgraphsafe data freeze --config configs/scisafetybench.toml
```

These steps make no inference calls. All 80 prompts are evaluation cases; preparation validates
the pinned files and groups across both domains without sampling. Live dataset evaluation requires
a recorded approval for the source revision and model destination; the preset supplies none.
Follow the [approval workflow](docs/scisafetybench-loader.md#hosted-processing-approval) before live
`evaluate`. Owned `smoke` prompts do not require dataset approval. The preset uses new `v2` data
and `runs/scisafetybench-chem-materials-v2` run directories and the existing $10/500-attempt ledger.

Reports show agreement with **rule-derived references, not expert-validated labels**, alongside
the escalation rate (successful REVIEW/BLOCK decisions divided by all intended cases). They
separate BLOCK/REVIEW swaps from ALLOW disagreements, retain all three constant baselines, and
provide a public category legend. Escalation is descriptive: always-REVIEW and always-BLOCK both
reach 100%. No benign false-positive rate or full
three-class macro-F1 can be estimated. See the [source, policy, workflow, and reporting contract](docs/scisafetybench-loader.md).
Existing SciHazard experiments remain unchanged.

## Run Laya locally

`configs/laya-local.toml` uses Laya 0.4.13 on CPU at numeric loopback only. The example pins the
`english` checkpoint to revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`; `/health` must
confirm that exact loaded revision before any request or budget reservation. Install the service in
a separate environment so its runtime dependencies do not change ChemGraphSafe's locked environment.
On Windows, run these commands inside WSL 2 because the workflow lock requires POSIX `flock`:

```sh
mkdir -p "$HOME/chemgraphsafe-runtimes/laya-0.4.13"
cd "$HOME/chemgraphsafe-runtimes/laya-0.4.13"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --torch-backend=cpu 'laya[serve]==0.4.13'

export LAYA_HOST=127.0.0.1
export LAYA_PORT=18000
export LAYA_DEVICE=cpu
export LAYA_PRELOAD=1
export LAYA_MODELS=english
export LAYA_DEFAULT_MODEL=english
export LAYA_MAX_LOADED=1
export LAYA_THREADS=16
export LAYA_AUTO_TASK=0
export LAYA_JEV_STRICT=0
export LAYA_REVISION=55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
.venv/bin/laya-serve
```

Keep that terminal open. In the ChemGraphSafe checkout, prepare an independent freeze and run the
five owned smoke cases before evaluating the 100-case SciHazard pilot:

```sh
uv run chemgraphsafe smoke --config configs/laya-local.toml
uv run chemgraphsafe data prepare --config configs/laya-local.toml
uv run chemgraphsafe data freeze --config configs/laya-local.toml
uv run chemgraphsafe evaluate --config configs/laya-local.toml
uv run chemgraphsafe report --run runs/laya-local-v1
```

These commands perform real inference, but requests stay on the loopback service and the example
uses attempts-only accounting with no provider-cost claim. The adapter preserves Laya's native
three-class probabilities and reports its checkpoint identity, but does not claim those values are
calibrated confidence. It does not feed another model's output into Laya. See
[Laya local service](docs/llm-adapters.md#laya-local-service) for the endpoint, identity, abstention,
authentication, and cached-offline contracts.

The v1 example uses the policy's fixed `ALLOW`, `REVIEW`, `BLOCK` option order. Laya 0.4.13
documents option-position sensitivity and recommends balanced rotations with label-aligned
probability averaging. Treat this first run as an integration diagnostic, not a fair comparative
performance estimate; implement rotations under a new run identity before model comparison.

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

- SciHazard primary: source-label interception recall with benign BLOCK/REVIEW rates and burden.
- SciSafetyBench: provisional rule-reference agreement and escalation side by side on all
  80 evaluation cases, plus BLOCK/REVIEW recalls and disagreement counts. Native binary
  source-label rates are unavailable; escalation is not a correctness score.
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
