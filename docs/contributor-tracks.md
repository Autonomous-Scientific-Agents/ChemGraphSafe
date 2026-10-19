# Hackathon contributor tracks

The starter is organized so four teams can work independently. Open an issue from the contribution
template for one bounded item; avoid changing shared policy semantics without coordinating.

| Track | First deliverable | Acceptance evidence |
|---|---|---|
| Data audit and curation | Audit quarantined IDs; add the SciSafetyBench loader | Provenance/rights record, loader tests, separate challenge-set manifest |
| Models | Add Kev or Laya serving adapter | Typed input-only contract, native response tests, identity/latency/cost reporting |
| Evaluation | Add calibration and robustness analysis | Known-answer metric tests, valid probability provenance, explicit denominators, dev-only fitting |
| Fine-tuning | Design training export and one upstream training recipe | Group-disjoint training set, audited labels, reproducible checkpoint and untouched evaluation |

## Suggested issues

1. **SciSafetyBench chemistry challenge loader.** Implement the [loader contract](scisafetybench-loader.md).
   Treat all-risk coverage separately; it cannot establish benign false-positive rates.
2. **Audit source conflicts and provisional annotations.** Review the 22 native/split conflicts,
   preserve adjudication provenance, and measure inter-reviewer agreement. Never silently overwrite
   an upstream label. Add optional LLM curation only with a configured distinct model.
3. **Kev adapter.** Use the [upstream repository](https://github.com/jaredpalmer/kev), not an unrelated
   PyPI package named `kev`. Map its native typed decisions and real probabilities into `Prediction`.
4. **Laya adapter.** Follow [upstream serving](https://github.com/NandhaKishorM/laya); establish what
   confidence means before exposing probability metrics. Preserve checkpoint identity.
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
