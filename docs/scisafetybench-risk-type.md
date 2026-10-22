# SciSafetyBench original risk-type classification

`configs/scisafetybench-risk-type.toml` opts into single-choice classification of the source
`Risk Type`. It uses the same pinned **40 chemistry + 40 materials science** prompts as the
[triage challenge](scisafetybench-loader.md), without converting categories to policy actions.
The existing `configs/scisafetybench.toml` and all binary/three-way workflows keep their behavior.

The [upstream taxonomy](https://arxiv.org/html/2505.23559v1#S4.SS1) describes sources of risk.
Predicting these categories is a derived classification task, not a reproduction of the paper's
agent evaluation, rejection rates, or safety scores. Categories may overlap, and concealed intent
may not be identifiable from the prompt. Scores measure agreement with the assigned source label.

| Category | Chemistry | Materials | Total |
|---|---:|---:|---:|
| Direct Malicious User | 10 | 9 | 19 |
| Indirect Malicious User | 10 | 8 | 18 |
| Task-Intrinsic Risk | 10 | 10 | 20 |
| Unintentional Consequences | 10 | 13 | 23 |

## Select the task

```toml
task = "scisafetybench_risk_type"
policy_file = "policies/scisafetybench-risk-type-v1.json"
```

Omitted `task` means `triage`. The new definition identifies its task and supplies fixed category
definitions and single-best-match instructions. Configuration and definition task identifiers
must agree. All model adapters support the task; start from the new preset and replace its
`[[models]]` blocks with the desired [adapter configuration](llm-adapters.md). Retain its
SciSafetyBench data settings and shared ledger. Each run evaluates one task, potentially with
multiple models. Model or definition changes require a fresh dataset freeze and run directory.

Jev and Laya receive `questions.risk_type` with `type="choice"` and the four named criteria; parsing reads
`answers.risk_type`. It records the chosen category and native four-category probabilities.
Chat, Argo, Codex, and replay use the existing `decision` field with these four allowed values.
No probabilities or explanations are invented for label-only adapters. Wrong-task labels,
missing/extra probability classes, and invalid distributions fail validation.

Laya retains its checkpoint preflight, truncation validation, and local/hosted processing rules.
It uses the four categories in the table's fixed order. The existing
[option-order limitation](llm-adapters.md#laya-local-service) also applies to this diagnostic;
balanced rotations remain a separate experiment.

Every model receives only the original request text/context and the shared definition. The
candidate names are visible to all examples; the correct answer and example metadata are not.
The original category stays in `Example.category`; `source_label` remains null rather than
overloading its binary meaning. Native references use `reference_origin="source_native"` and
`source_risk_type`, with source revision and record provenance, inside hashed local run references.

## Prepare, freeze, and run

```sh
uv run chemgraphsafe data prepare --config configs/scisafetybench-risk-type.toml
uv run chemgraphsafe data freeze --config configs/scisafetybench-risk-type.toml
```

These steps do not invoke models. Preparation may download the pinned source files. Freeze
checks each category, request, and provenance record against the checksum-verified local source
cache. Keep that cache through freeze; subsequent freeze verification and reporting use the
immutable prepared artifacts. All 80 examples remain evaluation cases, grouped and ordered using
the existing protocol. The new freeze protocol is `scisafetybench-risk-type-1`.

No annotation step is needed. This task rejects curator/annotation-rule configuration, existing
annotation files, `data annotate`, and `curate`. Offline owned fixtures may exercise the same
task using `challenge_eval`; their references are marked `owned_fixture` rather than source-native.

The preset uses `.local/data/scisafetybench-risk-type-v1` and `runs/scisafetybench-risk-type-v1`.
Before choosing to run live inference, follow the existing
[hosted-processing approval workflow](scisafetybench-loader.md#hosted-processing-approval).
The shared `$10 / 500-attempt` ledger and adapter-specific accounting remain in force.

```sh
# Live inference, only after source/destination approval and explicit authorization:
uv run chemgraphsafe evaluate --config configs/scisafetybench-risk-type.toml
uv run chemgraphsafe report --run runs/scisafetybench-risk-type-v1
```

`smoke` remains an explicitly live transport/schema check for live configurations. Its owned benign
prompts have no four-category gold answer. Reports show coverage, failures and usage, with accuracy
and macro-F1 unavailable. A successful smoke does not demonstrate classification quality.

## Scores and compatibility

- Accuracy is correct classifications divided by all referenced cases. Missing predictions,
  operational errors, and abstentions count as misses and as false negatives for the reference class.
- Confusion matrices have four reference rows and four predicted-category columns plus `ABSTAIN`
  and `ERROR`. Per-class precision, recall and F1 include explicit support counts.
- Macro-F1 averages all four class F1 values. It is null if any reference class is absent in a
  slice; undefined ratios are null. An all-error run with all four classes present has macro-F1 zero.
- Reports include domain/category slices, model identity diagnostics, latency, cost, and four
  constant-category baselines. Always-Unintentional-Consequences has accuracy `23/80 = 28.75%`.
- `metrics.json`, `comparison.csv`, and `report.md` contain classification measures instead of
  triage measures. `errors.jsonl` and `errors.csv` contain IDs, outcomes and fixed issue codes,
  including `source_risk_type_disagreement`; no request text is emitted.

Existing triage configs retain their semantic hashes, prompts, serialized examples, and freeze
protocol. Historical manifests without task metadata are interpreted as triage. Existing freezes
remain verifiable and runs remain reportable. Changed inference code still requires a new run
directory under the existing code-hash rule. Never overwrite an old freeze or mix the two tasks'
predictions. Definition contents and task identity participate in the new experiment's hashes.

Tests use only owned harmless requests, HTTP mocks, the fake Codex SDK, and offline replay:

```sh
uv run pytest -q tests/test_risk_type.py
```
