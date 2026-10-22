# Hackathon contributor tracks

The starter is organized so four teams can work independently. Open an issue from the contribution
template for one bounded item; avoid changing shared policy semantics without coordinating.

| Track | First deliverable | Acceptance evidence |
|---|---|---|
| Data audit and curation | Audit quarantined IDs and SciSafetyBench rule references | Adjudication provenance, agreement analysis, versioned annotations |
| Model adapters | Add Kev or Laya serving adapter | Typed input-only contract, native response tests, identity/latency/cost reporting |
| Evaluation and calibration | Add calibration and robustness analysis | Known-answer metric tests, valid probability provenance, explicit denominators, dev-only fitting |
| Fine-tuning and export | Design training export and one upstream training recipe | Group-disjoint training set, audited labels, reproducible checkpoint and untouched evaluation |

## Suggested issues

1. **SciSafetyBench reference audit.** Audit the provisional category mapping for the implemented
   [80-case chemistry/materials challenge](scisafetybench-loader.md). Preserve original references
   and record adjudication in a new version. This challenge cannot establish benign false-positive rates.
2. **Audit source conflicts and provisional annotations.** Review the 22 native/split conflicts,
   preserve adjudication provenance, and measure inter-reviewer agreement. Never silently overwrite
   an upstream label. Add optional LLM curation only with a configured distinct model.
3. **Kev adapter.** Use the [upstream repository](https://github.com/jaredpalmer/kev), not an unrelated
   PyPI package named `kev`. Map its native typed decisions and real probabilities into `Prediction`.
4. **Laya adapter.** Follow [upstream serving](https://github.com/NandhaKishorM/laya); establish what
   confidence means before exposing probability metrics. Preserve checkpoint identity.
   After a Kev or Laya adapter lands, other open System One-compatible models (for example Lev)
   are welcome as further adapters under the same contract.
5. **Calibration report.** Add Brier/ECE/reliability plots only for genuine full distributions with
   matching reference provenance. Return N/A for label-only models. Fit thresholds on development
   data only and test empty/missing-class cases.
6. **Robustness and leakage audit.** Audit semantic near-duplicates and label leakage with harmless
   perturbations. Preserve group inheritance across derivatives; do not generate operational attacks.
7. **Training export contract.** Add a separate training partition in a later release. Export only
   explicitly training-assigned, licensed examples with label provenance; refuse dev/eval groups
   and their derivatives. The current pilot has no training-eligible records.
8. **Upstream fine-tuning recipe.** After export/audit, implement either Kev's or Laya's documented
   training loop with fixed seeds, data/checkpoint hashes and license records. Evaluate the resulting
   checkpoint through the same runner. Report an untouched holdout and source agreement alongside
   provisional policy agreement; avoid training against the model's own reference labels.

Actual training, calibration, and live LLM runs are intentionally outside this starter milestone.
