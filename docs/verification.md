# Local verification

## PR 5 review fixes — 2026-10-22

- Locked development sync, Ruff lint/format checks, and the full offline suite passed:
  **233 passed, 3 skipped**. The focused reporting, SciSafetyBench, and hosted-rights suite
  passed **70 tests**; all provider calls used mock HTTP or replay fixtures.
- The 80-case known-answer check gives **74/80 agreement**, **76/80 escalation**, and one each
  of BLOCK→REVIEW, REVIEW→BLOCK, and ALLOW-vs-escalation-reference disagreements.
- Approval regressions cover missing/invalid records, old frozen permission flags, source and
  destination mismatches, curator destinations, approval-bound resume identity, and unchanged
  freezes. They also check rejection before provider/budget setup and no approval leakage into
  model inputs. No real approval record was created.
- Curated SciHazard queue behavior, readable public category mappings, small-slice counts and
  intervals, curator identity exclusions, and `operational_risk` validation are covered.
- Existing pilot (100 cases), offline (10), codex-astra-full (228), and SciSafetyBench `v1` (80)
  freezes passed read-only verification. Artifact hashes were unchanged before/after checks.
- Offline CLI `evaluate` and `report` passed on 10 owned cases in a fresh temporary directory.
  No existing experiment was evaluated or re-reported. No live inference or real dataset import
  was performed for this follow-up; the new `v2` preset remains unprepared and unapproved.

## Original SciSafetyBench integration — 2026-10-22

- Locked development sync, Ruff lint/format checks, and the full offline suite passed:
  **206 passed, 3 skipped** (optional SDK contracts without the optional SDK installed).
- Real pinned import: both file SHA256 values matched; 40 Chemistry + 40 Materials Science
  prompts, no exact duplicates/conflicts, and 80 singleton groups at cosine >=0.9.
- Offline `data annotate` generated **37 BLOCK / 43 REVIEW** provisional rule references.
  All 80 were frozen into evaluation under the shared oversight policy. No native binary
  labels or ALLOW references were invented.
- Offline CLI `evaluate` and `report` passed in fresh local verification directories:
  10 owned examples, 9 replayed decisions, and 1 abstention. Existing artifacts were preserved.
- Existing offline and SciHazard pilot freezes passed compatibility checks. Regression tests
  also cover reading reports without domain metadata and verifying a freeze after the original
  fixture source is removed.
- No live Jev smoke/evaluation or LLM curation was run. Downloads, annotations, the new freeze,
  and replay artifacts remain ignored. Test sources are owned harmless examples; provider
  requests use mock HTTP.

SciSafetyBench examples SHA256: `2dcf3d76e5d0bda345f44c3c23e81038ae72e7dd4f2e80d52eba6126c1fe09b1`.
Split/order hash: `0672af1b69837de6d6b299458fa5e943fbc4b61674c7ecd86b1e407fd4e14239`.
See [the dataset contract](scisafetybench-loader.md) for source pins, mapping, and commands.

## Original pilot verification — 2026-10-19

Verified on macOS with Python 3.12.11 on 2026-10-19.

- Locked installation: `uv sync --locked --extra dev` passed (32 resolved packages).
- Ruff lint and formatting checks passed.
- Pytest: 35 passed, using harmless fixtures and mock HTTP; no paid inference.
- Offline CLI `evaluate` then `report` passed: 10 cases, 9 decisions and 1 native abstention;
  the report clearly identifies MOCK/OFFLINE results and includes both diagnostic baselines.
- Real SciHazard import passed both pinned file checksums. 250 chemistry source records;
  22 native/split conflicts quarantined; 228 eligible groups; 100 selected groups, 50/50 classes.
- Real data freeze validated: development 10 benign + 10 harmful; evaluation 40 + 40;
  zero policy annotations, as requested for the initial source-only workflow.
- OpenRouter key was absent, so live Jev smoke/evaluation were not run. The wire contract,
  malformed response handling, retries, budgets and native refusals were tested with mock HTTP.
- Live LLM work, training, and calibration remain contributor tasks. CI is configured but has
  not run on a remote host; no commit or remote publication was performed.

Pilot examples SHA256: `016a49afa9f3f9495c240309ba2d799f83f9e43eddfc6c882181173e145afb94`.
Split/order hash: `7fa9c77d7ca942937e31248aa1a387f6150b6b519ee2e9b482d21327c85e3ded`.
The local freeze manifest stores these plus configuration, preparation and code provenance.

The local sandbox could not resolve external hosts and caused uv's macOS system-configuration
lookup to fail. Public source download and locked sync succeeded with approved sandbox escalation.
No raw chemistry requests were printed in verification output or committed to the project.

See `PLAN-REVIEW-LOG.md` for the independent Claude inspection transcript and dispositions.
