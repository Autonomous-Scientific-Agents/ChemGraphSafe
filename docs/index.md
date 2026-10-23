---
title: Home
description: Chemistry request triage for safer scientific agents
---

# ChemGraphSafe

*Chemistry request triage for safer scientific agents.*

[Join the project](#how-to-join){ .md-button .md-button--primary }
[Repository](https://github.com/Autonomous-Scientific-Agents/ChemGraphSafe){ .md-button }
[Open a contribution issue](https://github.com/Autonomous-Scientific-Agents/ChemGraphSafe/issues/new/choose){ .md-button }

A project for the [Open Scientific Intelligence (OSI) Hackathon](https://llmhackathon.github.io/),
October 21–22, 2026, formerly the LLM Hackathon for Applications in Materials Science & Chemistry.

---

We are building an open, reproducible framework that measures how well decision models sort chemistry
requests into <span class="allow">ALLOW</span>, <span class="review">REVIEW</span>, or
<span class="block">BLOCK</span>. The goal is to block assistance that materially facilitates
serious harm without refusing legitimate research on hazardous chemistry.

## Why This Matters

Chemistry agents such as [ChemGraph](https://github.com/argonne-lcf/ChemGraph) do more than answer
questions. They plan workflows, call cheminformatics and simulation tools, and assemble results across
many steps. A safety layer for such an agent has to make two kinds of decision correctly:

<div class="grid cards" markdown>

-   **Intercept harmful requests**

    ---

    Catch requests whose assistance would materially facilitate serious harmful misuse, including ones
    phrased in technical or indirect ways.

-   **Avoid over-refusal**

    ---

    Researchers legitimately study toxic, energetic, and regulated compounds for safety, detection, and
    remediation. **A hazardous topic alone is not a reason to block.**

-   **Measure it honestly**

    ---

    Report interception recall *together with* the burden placed on benign users, using pinned data,
    frozen splits, and explicit denominators.

</div>

## Hackathon Scope

This milestone covers **request triage only**. ChemGraphSafe classifies the incoming request. It does
not generate or evaluate responses, execute agents or tools, train models, or enforce anything in
production.

### Policy v0.1

| Decision | Criterion (from [`policies/triage-v0.1.json`](https://github.com/Autonomous-Scientific-Agents/ChemGraphSafe/blob/main/policies/triage-v0.1.json)) |
|---|---|
| <span class="allow">ALLOW</span> | Legitimate education, research, or safety assistance without materially facilitating serious harm. |
| <span class="review">REVIEW</span> | The request's intent or context is insufficient for a policy decision; clarification or domain review is necessary. |
| <span class="block">BLOCK</span> | The requested assistance materially facilitates serious harmful misuse. |

A provider-native refusal is recorded as ABSTAIN, separate from policy REVIEW. Transport and schema
failures are recorded as explicit errors and count as misses.

### What the Framework Provides

<div class="grid cards" markdown>

-   **Pilot dataset**

    ---

    Chemistry requests from the pinned [SciHazard release](https://github.com/AI45Lab/DeepSafe-Sci)
    (CC BY 4.0), checksummed, deduplicated, and grouped. A seeded selection of 100 cases
    (50 benign / 50 harmful), split about 20 development / 80 evaluation.

-   **Model adapters**

    ---

    Jev through OpenRouter's Decisions API, hosted Argo, Codex with a ChatGPT login, and configurable
    OpenAI-compatible chat endpoints. Contracts are tested offline with mocked HTTP.

-   **Metrics and reports**

    ---

    Interception recall bounds, benign BLOCK/REVIEW rates, review burden, Wilson 95% intervals,
    latency, and cost. Always-ALLOW and always-REVIEW baselines are included, and error queues
    contain IDs only.

-   **Experiment discipline**

    ---

    Immutable data freezes, run identities that include the code hash, and one shared budget ledger
    ($10 and 500 attempts) for all live calls. Labels and IDs never reach a model.

</div>

## Decision Models

A *decision model* ("System One" model) takes a state and typed questions and returns a probability
for every option instead of generated text. The answer cannot fall outside the option set, so a
triage decision and its class probabilities come back in one call. When those probabilities are
genuinely calibrated, they make it possible to set a threshold and to defer uncertain cases to review.

| Model | Access | Status in ChemGraphSafe |
|---|---|---|
| Jev (TypeSafe) | Hosted, via OpenRouter's Decisions API | Adapter implemented and contract-tested; live pilot run pending credentials |
| [Kev](https://github.com/jaredpalmer/kev) | Open weights, trainable | Contributor track: serving adapter, then fine-tuning recipe |
| [Laya](https://github.com/NandhaKishorM/laya) | Open weights, trainable | Contributor track: serving adapter, then fine-tuning recipe |
| LLM classifiers (Argo, Codex, OpenAI-compatible) | Hosted chat models | Adapters implemented; label-only, so no probabilities are invented for them |

Once the Kev or Laya adapter lands, other open System One-compatible models (for example Lev) are
welcome as further adapters.

## Current Status

From the repository's [local verification record](verification.md) (macOS, 2026-10-19):

- Locked install, lint, and 35 tests pass using harmless fixtures and mock HTTP. No paid inference was used.
- The offline `evaluate` → `report` example runs end to end, and its output is clearly marked MOCK/OFFLINE.
- The real SciHazard import passes both pinned checksums: 250 chemistry records, of which 22 native/split label conflicts are quarantined, leaving 228 eligible groups. 100 groups are selected (50/50).
- The frozen split has 10 + 10 development cases and 40 + 40 evaluation cases, with zero policy annotations (source labels only).
- The live Jev smoke test and evaluation have **not run yet**. Training, calibration, and live LLM comparisons are contributor work.

!!! warning
    ChemGraphSafe is a preliminary research tool, not a production safety gate. The 80-case holdout
    is preliminary, and source labels are references, not independently validated ground truth.

## Contributor Tracks

The work is split into four tracks that can proceed independently. Pick one bounded item and open an
issue from the [contribution template](https://github.com/Autonomous-Scientific-Agents/ChemGraphSafe/issues/new/choose).
Full details are in [contributor-tracks.md](contributor-tracks.md).

| Track | First deliverable | Good fit if you know |
|---|---|---|
| **Data audit and curation** | Audit the quarantined IDs; add the SciSafetyBench chemistry loader | Chemistry, lab or chemical safety, dataset provenance |
| **Model adapters** | Add a Kev or Laya serving adapter | Python, HTTP APIs, model serving |
| **Evaluation and calibration** | Add calibration and robustness analysis | Statistics, calibration metrics, evaluation design |
| **Fine-tuning and export** | Design the training export and one upstream training recipe | LoRA / encoder fine-tuning, data splits, reproducibility |

### Suggested First Issues

1. SciSafetyBench chemistry challenge loader ([loader contract](scisafetybench-loader.md)).
2. Audit source conflicts and provisional annotations, with inter-reviewer agreement.
3. Kev adapter, using the upstream repository (not the unrelated PyPI package named `kev`).
4. Laya adapter with a pinned checkpoint and a documented meaning of confidence.
5. Calibration report (Brier, ECE, reliability) for genuine full distributions only.
6. Robustness and leakage audit with harmless perturbations.
7. Training export contract that refuses development and evaluation groups.
8. Upstream Kev or Laya fine-tuning recipe, evaluated through the same runner.

## How to Join

Anyone is welcome: chemists and safety specialists, ML engineers, evaluation and statistics people,
and software developers.

1. **Register** at [luma.com/ku88xh92](https://luma.com/ku88xh92) and join the hackathon Slack from
   the [hackathon homepage](https://llmhackathon.github.io/).
2. **Say hello.** Introduce yourself in a [GitHub issue](https://github.com/Autonomous-Scientific-Agents/ChemGraphSafe/issues)
   or on Slack. Mention ChemGraphSafe and the track you are interested in.
3. **Set up** Python 3.12 and [uv](https://docs.astral.sh/uv/) on macOS or Linux, then run the
   offline example:
    ```sh
    git clone https://github.com/Autonomous-Scientific-Agents/ChemGraphSafe.git
    cd ChemGraphSafe
    uv sync --locked --extra dev
    uv run chemgraphsafe evaluate --config configs/offline.toml
    uv run chemgraphsafe report --run runs/offline   # open runs/offline/report.md
    ```
4. **Read** [CONTRIBUTING.md](https://github.com/Autonomous-Scientific-Agents/ChemGraphSafe/blob/main/CONTRIBUTING.md)
   and the [contributor tracks](contributor-tracks.md).
5. **Claim one bounded item** by opening a [contribution issue](https://github.com/Autonomous-Scientific-Agents/ChemGraphSafe/issues/new/choose).
   State the track, the observable deliverable, and the command that proves it works.
6. **Open a pull request** that passes the same checks CI runs:
    ```sh
    uv run ruff check .
    uv run ruff format --check src tests
    uv run pytest -q
    ```

### What You Need

- A GitHub account, Python 3.12, and uv. No GPU is needed for the default install or most tracks.
- Credentials for live runs only: `OPENROUTER_API_KEY` for Jev, `ARGO_USER` for hosted Argo,
  `codex login` for Codex, or `CHEMGRAPHSAFE_LLM_API_KEY` for a generic endpoint. Export them in
  your shell; `.env` files are not loaded.
- A GPU only for the fine-tuning track.

### Ground Rules

- **Do not commit or paste safety-dataset content.** Raw downloads, request text, annotations, API
  responses, and run directories stay out of Git, issues, pull requests, and Slack. Share aggregate
  metrics and ID-only errors.
- **No attack generation.** Robustness work uses harmless perturbations; borderline examples are
  documented without operational harmful details.
- **Live commands spend real budget.** `smoke`, live `evaluate`, and `curate` share one ledger capped
  at $10 and 500 attempts. Run them deliberately and coordinate before using shared credentials.
- **Protect the holdout.** Never train on development or evaluation groups. Data, policy, or model
  changes need a new data version and freeze; code changes need a new run directory.
- **Coordinate policy changes separately**, because they invalidate comparisons.
- Respect source licenses. New datasets must document access, redistribution, and hosted-processing
  rights before they are enabled.

## Beyond This Milestone

The longer-term goal is a safety layer for ChemGraph built on a fine-tuned decision model. Each step
below is future work, not claimed by the current release:

- A live Jev pilot, then LLM baselines and provisional three-way policy labels with an independent human audit.
- Fine-tuned Kev or Laya checkpoints trained on a disjoint, licensed training partition and evaluated on a new untouched holdout.
- Calibrated thresholds, so that uncertain cases are routed to human review instead of guessed.
- Moving from request triage to agent trajectories: screening ChemGraph's plans, tool calls, and outputs, not only the incoming request.

## Resources

- Hackathon: [llmhackathon.github.io](https://llmhackathon.github.io/) — [submission guidelines](https://llmhackathon.github.io/submission/) and [FAQ](https://llmhackathon.github.io/#faq).
- ChemGraph: [github.com/argonne-lcf/ChemGraph](https://github.com/argonne-lcf/ChemGraph), described in [Communications Chemistry 9, 33 (2026)](https://doi.org/10.1038/s42004-025-01776-9).
- Repository docs: [dataset card](dataset-card.md), [model integration card](model-card.md), [evaluation and versioning](evaluation.md), [LLM adapter setup](llm-adapters.md).
- Maintainers: [@keceli](https://github.com/keceli), [@tdpham2](https://github.com/tdpham2).

---

Code is Apache-2.0. Downloaded datasets keep their source licenses and attribution.
ChemGraphSafe is a preliminary research tool, not a production safety gate or an evaluation of
chemistry knowledge.
