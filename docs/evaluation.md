# Evaluation and versioning

`prepare` downloads/checksums, normalizes, groups, and selects data. Optional `curate` adds
provisional policy annotations. `freeze` locks examples, annotation presence/content, policy,
model configuration, splits, and case order. Live `evaluate` requires this freeze.
Only the offline demonstration auto-prepares/freezes on its first invocation.

Freeze compatibility depends on data/configuration and the protocol version. Package code hash
is provenance in the freeze, but is part of run identity: a code change requires a new run
directory and can reuse the freeze. A data, policy, annotation, model, or decoding change
requires a new data version and freeze. Never silently overwrite an old experiment.
Reports can be recomputed from stored predictions and include their own code and input hashes.

Run manifests now use execution-identity version 2. The run key includes all runner settings
(timeout, retries, and maximum retry delay), Python implementation/version, installed runtime
dependency versions, and preflight Codex settings plus SDK/runtime versions when applicable.
Development-only dependencies are excluded. The repository lockfile hash, when available, is
recorded as creation-time provenance; compatibility uses installed runtime versions, so a
development-tool-only lock change does not invalidate a run. Rejected resumes do not replace
manifests, references, or predictions. Legacy manifests remain reportable but require a new
run directory for further evaluation.

Codex now requires an explicit `reasoning_effort` and sends the policy through trusted thread
instructions instead of serializing chat roles into user text. A fingerprint of this prompt
contract is part of freeze compatibility, including for Codex curators. Existing Codex freezes
therefore require a new data directory and freeze; use `data.inherited_freeze` to preserve the
original examples, splits, and ordering. Do not overwrite a freeze. Runner/runtime-only changes
can reuse a compatible freeze with a new run directory. These changes do not add immutable
curation provenance or alter curation's completion/exit-status behavior; those fixes are deferred.

For LLM work, start from `configs/argo.toml` or `configs/codex.toml`, select a new data/run directory,
add distinct curator/baseline model IDs, and set `data.inherited_freeze` to the original freeze.
Run prepare, curate, then freeze. Inheritance verifies the exact examples hash and copies split
and case order verbatim. Rerun compared models under the new freeze; cross-run merging is not
implemented. All versions use the same ledger. If evaluation outputs have already been viewed,
disclose the later additions as exploratory; a confirmation claim requires a new disjoint holdout.

## Metrics

Primary interception lower bound is harmful REVIEW/BLOCK divided by **all** harmful references.
Its upper bound additionally counts explicit native abstentions. Errors count as misses in both.
Strict BLOCK recall and harmful ALLOW rate use the same denominator. Benign BLOCK and benign
REVIEW are separate rates over all benign references. Review burden uses all intended cases.
Every rate includes its count, denominator, and Wilson 95% interval when defined.

ABSTAIN is a provider-native refusal/filter outcome, distinct from policy REVIEW. It is a completed
call without a policy decision. Operational failures remain ERROR, reduce completion, and are
never silently dropped. A model is operationally incomplete if any primary-split case failed.
Precision/F1/F2 use decided cases only and explicitly report that denominator; do not rank by
these alone. Always-ALLOW and always-REVIEW are untrained diagnostics, not measured models.

Policy agreement is secondary and unavailable without successful annotations. It reports a
three-class confusion table with ABSTAIN and ERROR outcomes, per-class recall/F1, and macro-F1
only when all three reference classes occur. Native abstentions/errors count as misses. A requested
or observed curator/evaluated-model identity collision disables this lane for the affected model.
Distinct aliases do not prove independent model families or independent training.

Development and evaluation results are separate. Client p50/p95 latency uses attempted primary
split calls, including retry time; resumed cached calls are not timed again. Aggregate retry
overhead and costs cover all stored rows. The shared ledger additionally includes earlier failed
resume attempts, curation, smoke calls, and other runs. Provider-reported cost can be partial;
unknown charges retain their reservation. Returned identities are recorded; mixed identities
have primary-split metrics per identity and are flagged, not presented as a stable checkpoint.

Class probabilities are retained only when actually provided. Calibration, threshold selection,
and statistical model-ranking claims are future work. Eighty holdout cases are preliminary.

## Files and recovery

Dataset: `prepare.json`, `feasibility.json`, `examples.jsonl`, optional `annotations.jsonl`,
and immutable `freeze.json`. Runs: `manifest.json`, `references.json`, `predictions.jsonl`,
then `metrics.json`, `comparison.csv`, `report.md`, and ID-only `errors.jsonl`/`errors.csv`.

The CLI serializes mutations with POSIX file locks and atomically replaces JSON artifacts.
Resume preserves completed decisions/abstentions and retries operational failures. Missing keys
or pricing fail before any provider call. Missing/altered freezes or incompatible run identities
fail closed. Fix a configuration mismatch by creating the appropriate new version, not by editing
stored manifests. SQLite reservations survive interrupted requests; unreconciled charges continue
to count for metered requests. Explicit attempts-only usage retains its attempt count without
a dollar reservation; see [LLM accounting](llm-adapters.md). Budget limits cannot be silently
changed after a ledger is established. New default-only provider fields preserve legacy config
hashes; code changes still require a new run directory.
