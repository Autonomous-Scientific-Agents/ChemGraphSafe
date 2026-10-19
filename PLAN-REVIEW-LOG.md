# Claudex review and implementation log

Host/planner/builder: Codex. Independent plan reviewer and final inspector: Claude Code. No explicit model or effort requested; retain CLI defaults. Three in-memory plan-review rounds completed in Plan Mode (REVISE, REVISE, APPROVED); observed model: claude-opus-5-5. Approved SHA256: 861d5ea10b56f7404d80e82ed6e4b569e0a4da986eadc81b109f5e62a721b8c0. Full original responses are in the conversation's tool transcript; no file-bound approval existed then.

Implementation authorized by the user on 2026-10-19. Preserve existing .agents/, .claude/ and .claudex-loop-install.json. No commit, push, publication or deployment authorized. The repository has no commits; there is no pre-build commit or Git diff baseline. Final inspection will use an explicit inventory and hashes of all newly created project files, rather than claim a commit-based diff.

Plan-review budget: five completed rounds (three used before implementation). Final inspection: fresh Claude session, maximum two rounds; maximum two build-fix rounds. Diagnostics stay outside the checkout. The unchanged approved plan is now persisted for a file-bound review before building.

Accepted review changes: explicit class-shortage rules; grouping before sampling; isolated smoke examples; a freeze command; source-reference metrics primary and LLM labels provisional; smaller milestone; cumulative budgets; separate native abstentions; versioned reruns; distinct curator/model identity checks. Remaining low-priority advice: caps may prevent reruns, and refusal-schema fixtures must disclose documentation-derived rather than live-captured provenance.


## File-bound review, round 4

Result: /private/tmp/chemgraphsafe-claudex/claudex-y7493i5y/result.json

```json
{
  "verdict": "REVISE",
  "summary": "The plan is detailed and mostly coherent. Its policy, label-provenance, abstention and leakage rules hold together well. Three medium gaps could break guarantees the plan itself makes. First, the \"cumulative\" budget ledger can be reset by the versioned reruns and smoke runs the plan prescribes. Second, the freeze and resume keys do not include code or adapter identity, and this repository has no commits to supply one. Third, nothing stops prepare, curate or freeze from overwriting frozen artifacts. The low-severity issues are mostly unclear rules that could lead to inconsistent metrics, or small leakage and reporting risks. All are fixable with small, specific changes to the plan.",
  "findings": [
    {
      "id": "F1-budget-ledger-scope",
      "severity": "medium",
      "path": "PLAN.md",
      "evidence": "The budget section says to use 'one persistent SQLite budget ledger under the pilot run directory across all commands/resumes'. The rerun section requires that 'A post-freeze integration/code/config fix creates a new versioned freeze/run' and that the rerun stays 'within the cumulative budget'. If the ledger lives in runs/pilot and a corrected rerun writes to a new versioned run directory, a fresh ledger starts at $0 / 0 attempts. That lets total spend go past the user's USD 10 / 500-attempt cap. The Day-2 5-case smoke test also has no defined CLI command or config; the required commands are only data prepare/curate/data freeze/evaluate/report. Day 2 runs the smoke test before data prepare, curate and freeze, so it is unclear whether smoke spend reaches the same ledger. Curate also runs before evaluate creates runs/pilot.",
      "fix": "Set one ledger path in config, shared by the pilot and every versioned rerun, separate from any single run directory. For example, a fixed project-level path in a git-ignored local state directory, referenced by pilot.toml, smoke and all rerun configs. Live commands must refuse to start if that ledger path is missing or different. Define the smoke test as a named config (e.g. configs/smoke.toml run through evaluate) or as a named command that writes to the same ledger. Add a test that a versioned rerun's reservations count against the same cumulative totals."
    },
    {
      "id": "F2-code-identity-missing",
      "severity": "medium",
      "path": "PLAN.md",
      "evidence": "The freeze records 'hashes of data, annotations, policy, prompts and model configs'. Resume is keyed by 'dataset/policy/prompt/adapter/model-configuration hash'. The plan also says 'do not mix adapter versions under one resume key' and that report-only fixes keep 'report code/version lineage'. But the review log notes this repository has no commits, and no commit is authorized, so there is no git SHA to identify code. Failure scenario: an implementer changes the response parser after freeze, the config hash does not change, evaluate passes the freeze check, and resume quietly combines rows parsed by two different parser versions under one key.",
      "fix": "Define a code-identity field that does not depend on git: an explicit ADAPTER_VERSION/PARSER_VERSION constant plus a content hash of the chemgraphsafe package source files. Put it in the run manifest and resume key, plus the freeze when the frozen prompts or configs depend on code. Evaluate must refuse to resume when the code identity differs from the existing rows. The report manifest records the report code hash."
    },
    {
      "id": "F3-freeze-overwrite",
      "severity": "medium",
      "path": "PLAN.md",
      "evidence": "The plan says freeze creates 'immutable split/dataset manifests' and 'No automatic re-freeze', and later fixes must 'preserve the original artifacts'. It never says what happens if `data prepare`, `curate` or `data freeze` is re-run with configs/pilot.toml after a freeze exists. A careless re-run of curate repeats paid curator calls and overwrites the annotations the freeze referenced. Evaluate would then reject the mismatch, but the original annotations would already be lost, and the curator was meant to run 'once'.",
      "fix": "Have data prepare, curate and data freeze refuse to overwrite frozen artifacts. Changed data should only go into an explicitly new dataset or freeze version ID, so earlier versions stay preserved. Define curate's resume behaviour: it may retry only examples with transport failures, never re-label examples that already have a terminal result, and never after a freeze. Add contract tests for refusing to overwrite."
    },
    {
      "id": "F4-grouping-spec",
      "severity": "low",
      "path": "PLAN.md",
      "evidence": "The plan unions pairs with 'character n-gram (3\u20135) TF-IDF cosine >=0.9 into connected groups' but does not say which fields are vectorized (request only, or request+context). Connected components chain transitively. If SciHazard prompts share long context or templates, chaining can merge many records into a few large components. That leaves fewer than 40 benign groups and the pilot gate fails, or one cross-class component swallows many harmful records. If someone then changes the threshold after seeing the counts, that is a selection decision made on the data and never recorded.",
      "fix": "Specify the vectorized fields and how the vectorizer is fitted (on the whole filtered pool, NFC-normalized). Report the distribution of component sizes, the largest component, and the number of cross-class components in the prepare manifest. State that the threshold and fields are fixed in config before inspection, and that any change creates a new recorded dataset version."
    },
    {
      "id": "F5-failure-denominators",
      "severity": "low",
      "path": "PLAN.md",
      "evidence": "The interception bounds are defined only for REVIEW/BLOCK (lower) and native abstentions (upper), 'divided by all harmful reference cases'. The plan never says where transport/schema failures on harmful cases go: whether they are excluded from both bounds, or whether the upper bound treats them as possibly intercepted. Per-class recall 'uses all labeled terminal outcomes', which reads as leaving failures out. That conflicts with 'Never silently rank on an easier successful subset'. Two implementations could report different numbers from the same rows.",
      "fix": "State explicitly that operational failures stay in every reference-case denominator and count as not intercepted for both bounds, or define a third 'failure-inclusive' upper bound. Report the failure count next to each rate. Add a known-confusion test fixture that includes failures."
    },
    {
      "id": "F6-probability-validation",
      "severity": "low",
      "path": "PLAN.md",
      "evidence": "The plan rejects 'scores not summing to one' but gives no tolerance, and does not say what happens if the Jev choice disagrees with the argmax of its returned probabilities. Real float outputs (e.g. 0.3333\u00d73) would fail an exact check and become schema failures. That would mark Jev operationally incomplete even though its outputs are valid.",
      "fix": "Set an absolute tolerance (e.g. |sum\u22121| \u2264 1e-6, or whatever the provider documents) and state whether renormalization is forbidden or recorded. Define which field becomes the decision when the choice and argmax disagree, and add tests for both cases."
    },
    {
      "id": "F7-id-order-leakage",
      "severity": "low",
      "path": "PLAN.md",
      "evidence": "Examples carry a 'stable source-qualified ID', and the source files are named benign/harmful. If IDs are derived from file names and sent as provider metadata (OpenAI-compatible `user` field, Jev request IDs or tags), the label leaks into inference, despite 'Do not feed labels ... source names'. 'Use identical inputs and case order', with order sorted by such IDs, would also run cases in blocks by class. Provider alias changes or rate limits during the run would then be confounded with class.",
      "fix": "Require that example IDs and source names never appear in provider request bodies or headers, and extend the leakage test to check full serialized requests. Fix the case order as a seeded shuffle recorded in the freeze manifest, not an order sorted by ID."
    },
    {
      "id": "F8-free-text-in-shareable-outputs",
      "severity": "low",
      "path": "PLAN.md",
      "evidence": "Annotation records hold a 'short reason code', but the curator is asked for a 'short policy reason' (free text). Shareable outputs must hold only 'non-operational error summaries', yet report.md is generated automatically and cannot judge whether model or curator text is operational. Curator reasons or text snippets quoted in report.md/comparison.csv could include harmful request details.",
      "fix": "Restrict the curator output to an enumerated set of reason codes, with any free-text rationale stored only in git-ignored local files. Ban request text, context and model or curator free text from metrics.json, comparison.csv and report.md, and add a test that checks for this."
    },
    {
      "id": "F9-returned-identity-drift",
      "severity": "low",
      "path": "PLAN.md",
      "evidence": "Resume keys use the requested model configuration hash, and the returned identity is only recorded per row. If a mutable alias rolls to a new version between resume sessions, one run quietly mixes model versions. The plan only says aliases are 'disclosed'.",
      "fix": "Have the report detect more than one distinct returned identity within a run, flag it prominently, and break metrics down per identity. Optionally make evaluate stop on resume when the returned identity changes."
    },
    {
      "id": "F10-retry-after-cap",
      "severity": "low",
      "path": "PLAN.md",
      "evidence": "The runner must 'respect Retry-After' but sets no maximum wait. A provider returning a very long Retry-After would leave a run hanging inside a 2\u20133 day schedule.",
      "fix": "Cap the Retry-After wait in config (e.g. 120s). If the cap is exceeded, record an explicit rate-limit failure and stop with an incomplete-run report."
    }
  ],
  "coverage": [
    "Read the full PLAN.md content as given (SHA stated by the requester; not recomputed).",
    "Read PLAN-REVIEW-LOG.md: earlier review rounds, accepted changes, and the fact that the repo has no commits or baseline.",
    "Listed the repository tree: only .agents/, .claude/ skill installs, .claudex-loop-install.json, PLAN.md and PLAN-REVIEW-LOG.md. There is no application code, .gitignore, pyproject or config to trace callers in.",
    "Checked the plan for internal consistency across data selection/grouping, curation, freeze/resume, budget ledger, runner, metrics/denominators, abstention handling, leakage, rights and the delivery sequence."
  ],
  "limitations": [
    "There is no implementation code yet, so there were no real callers or writers of shared state to trace. Findings about shared state (ledger, freeze artifacts, resume keys) come from the plan's own described design.",
    "External research claims were not checked: Kev/Laya/Jev repositories and APIs, SciHazard's 50 benign / 200 harmful chemistry counts, licences, SciSafetyBench's 40 records, and SOSBench/SafeSci gating. I had no network access.",
    "I did not recompute the PLAN.md SHA256 (no shell available). I reviewed the plan text supplied in the request and the file path given.",
    "I did not run any tests or commands, and I make no claim that anything passes."
  ]
}
```

Host accepted F1–F10: shared ledger outside version directories; code hashes without commits; immutable freeze lifecycle; grouping details; explicit failure denominators; numeric tolerance and authoritative choice; no ID leakage and seeded order; enumerated reasons; identity drift diagnostics; retry wait bound. User changed scope: Jev through OpenRouter, live LLM work deferred. Updated the main plan consistently.

## Round 5: review limit reached

Result: /private/tmp/chemgraphsafe-claudex/claudex-wm3709bg/result.json

```json
{
  "verdict": "REVISE",
  "summary": "This revision resolves F1\u2013F10 from round 4. It adds a fixed shared ledger and a named smoke command, a code identity that does not need Git, rules that stop frozen artifacts being overwritten, a defined grouping spec, failure-inclusive denominators, a probability tolerance with the provider's choice taking precedence, no ID leakage plus a seeded case order, enumerated reason codes, identity-drift diagnostics, and a 60-second Retry-After cap. The narrower scope (live Jev through OpenRouter, LLM baseline and curation deferred, freeze with source labels only) is mostly consistent. One medium defect remains, introduced by the F2 fix: the freeze binds a hash of all package source files. Read literally, that clashes with `evaluate` refusing mismatched freezes, with the checked-in offline frozen fixture, and with the planned Day-3 code work. Two low gaps also remain: the deferred curation path versus comparisons on the same frozen IDs, and how the ledger path is resolved and git-ignored.",
  "findings": [
    {
      "id": "F11-freeze-bound-to-whole-package-hash",
      "severity": "medium",
      "path": "PLAN.md",
      "evidence": "The freeze records 'a content hash of all package source files and an adapter/parser protocol version', and 'evaluate refuses missing or mismatched freezes'. Several things then break. (1) The offline proof command `evaluate --config configs/offline.toml` uses 'a checked-in harmless frozen fixture'. Any edit to any chemgraphsafe module changes the package hash, so the fixture goes stale and the offline evaluate fails, or the fixture must be regenerated on every code edit. (2) Day 2 freezes the pilot, then Day 3 changes package code ('finish report and ranked error queue'). If credentials arrive afterwards, `evaluate --config configs/pilot.toml` rejects the Day-2 freeze. Fixing that needs a new dataset directory, which conflicts with 'No automatic re-freeze'. (3) Without credentials, the Jev parser is only checked against mocked transport before freeze. A parser fix after the first live smoke counts as a 'post-freeze integration/code fix'. That would label the first real Jev evaluation as an 'integration-corrected ... rerun on previously seen cases, never a fresh untouched holdout', even though no evaluation inference has happened.",
      "fix": "Keep the code hash in the freeze only as a provenance record, not as a compatibility condition. Evaluate's freeze-compatibility check should cover data, split, case order, annotations, policy, prompts and model config, plus the adapter/parser protocol version. Put the code identity in the run manifest and resume key, ideally a hash of only the modules on the inference path (adapters, parsing, runner, prompts and policy). Code changes then start a new run instead of invalidating the freeze. Base holdout contamination on whether evaluation inference or result inspection has happened, not on whether a freeze was created. Code or parser fixes made before any evaluation inference do not taint the holdout and need no re-freeze. Make sure the offline fixture's freeze does not pin a package hash."
    },
    {
      "id": "F12-deferred-curation-vs-same-frozen-ids",
      "severity": "low",
      "path": "PLAN.md",
      "evidence": "Freezes are immutable 'including its explicit absence of annotations', and the later LLM workflow must 'use a new dataset/run directory, run curate before data freeze, then evaluate'. Comparisons must use 'the same frozen evaluation IDs' and evaluate 'refuses ... mismatched freezes'. So Jev predictions made under the Day-2 source-only freeze cannot be scored against later policy labels or compared with the later LLM baseline. That forces either a paid Jev rerun under the new freeze, using the shared USD 10 / 500-attempt cap, or an undefined join across freezes. Nothing guarantees that the new freeze reproduces the same split and case order. The later curation would also happen after Jev's evaluation errors were inspected, which the plan doesn't disclose.",
      "fix": "Require later freezes to reuse the original split and case-order manifest verbatim, checked by hash. Let reports join predictions from different freezes only when the dataset, split and case-order hashes match. Alternatively, treat annotations as a separate versioned layer that pins the original freeze hash. Require the report to state that provisional policy labels and the LLM baseline were added after Jev's evaluation results were inspected."
    },
    {
      "id": "F13-ledger-path-resolution",
      "severity": "low",
      "path": "PLAN.md",
      "evidence": "The ledger is fixed at '.local/state/pilot-budget.sqlite', but the plan doesn't say what that relative path is relative to. If it resolves against the current working directory, running smoke or evaluate from another directory creates a second, empty ledger. That silently resets the cumulative budget, which is the defect F1 was meant to close. The Day-1 .gitignore list (raw/derived data, runs, credentials, error queues) also leaves out .local/.",
      "fix": "Resolve the ledger path relative to the project root or the config file's directory, and record the absolute path in every run manifest. Add .local/ to .gitignore on Day 1. Add a test that running from a different working directory reaches the same ledger."
    }
  ],
  "coverage": [
    "Read the revised PLAN.md text supplied in the request (stated SHA 3a4ba074\u2026; not recomputed).",
    "Re-read PLAN-REVIEW-LOG.md, including the round-4 findings F1\u2013F10 and the host's acceptance note.",
    "Checked each prior finding against the revision. F1 (shared ledger, named smoke), F2 (code identity without Git), F3 (no overwrite of frozen artifacts, curation resume rules), F4 (grouping fields, threshold and histograms), F5 (failures stay in denominators), F6 (1e-3 tolerance, provider choice takes precedence over argmax), F7 (no ID or metadata leakage, seeded order), F8 (enumerated reason codes, no free text in public reports), F9 (mixed-identity flag) and F10 (60s Retry-After cap) are all resolved in the text.",
    "Checked the scope change for consistency: OpenRouter Jev endpoint and credentials, a source-only freeze with policy metrics marked N/A, deferred live runs shown as pending rather than mocked, proof commands gated on credentials, and the later LLM workflow.",
    "Listed the repository again: still only the .agents/ and .claude/ skills, .claudex-loop-install.json, PLAN.md and PLAN-REVIEW-LOG.md. There is no implementation, configs or .gitignore whose callers I could trace."
  ],
  "limitations": [
    "There is no application code yet, so findings about shared state (freeze, ledger, resume keys) come from the plan's described design, not from real callers.",
    "I could not check external claims because I had no network access. That includes the OpenRouter Decisions endpoint (POST /api/alpha/decisions), the model ID typesafe/jev-1.14, the response and refusal schema and pricing units, the blog source, the SciHazard counts and licence, and the other dataset and repository claims.",
    "I did not recompute the PLAN.md SHA256, and I reviewed the plan text supplied in the request.",
    "I did not run any tests or commands, and I make no claim that anything passes."
  ]
}
```

Host accepted F11–F13: separate data freeze compatibility from code provenance; preserve original split/order for later annotation versions; resolve the shared ledger relative to config project root. These final plan edits have not received another plan review. The prior in-memory approval is not represented as approving this changed plan.

## Authorized build

The user explicitly selected “Proceed with implementation and final code inspection”, then reiterated “Proceed with building”. Proceed on the standalone unreviewed-spec path for the final plan amendments. Codex authors the code; a fresh Claude session independently inspects it. No further plan-review rounds. No pre-build commit exists; preserve the original skill directories and inspect all added project files with a content-hash manifest. No commits or publication.


## Implementation and proof before inspection

Codex implemented the Python package, typed records/protocols, SciHazard preparation/grouping,
immutable freezes, optional curation, Jev/OpenRouter and chat/replay adapters, shared SQLite budget,
resumable runner, metrics/reports, harmless fixtures, contributor documentation, issue template,
CI configuration, dependency lockfile and Apache-2.0 license. No external issues, commits or pushes.

Real import correction: the unsafe split contains 200 chemistry rows, of which 22 have
`is_harmful=false`. Both source fields are preserved; conflicts are quarantined by ID without
relabeling. PLAN.md and the dataset card now reflect this observed correction. The eligible pool
has 228 singleton groups; selection has 100 groups, 50 benign / 50 harmful. Freeze has 20 dev
(10/10) and 80 eval (40/40), zero annotations. This change did not involve tuning on model outputs.

Proof: locked uv sync succeeded; Ruff lint/format passed; 35 pytest tests passed; actual offline
CLI evaluate/report succeeded with nine decisions and one abstention. Actual pinned public data
prepare/freeze and artifact hash checks succeeded. No live Jev or LLM inference was performed;
OPENROUTER_API_KEY was absent, and the user deferred live LLM work. The original mirrored skill
directories compare byte-for-byte equal after undoing an incidental lint import reorder; all
project lint/format commands now exclude them.

Independent inspection uses the installed claudex-loop runner's command construction and result
validation with Claude's restricted Read/Glob/Grep tools in a fresh session. Since there is no
pre-build commit, a wrapper supplies a complete 39-file added-project inventory and verifies
before/after SHA256 content fingerprints. The pre-existing skill directories/installer and this
append-only coordination log are explicitly excluded. This is not represented as a Git diff review.
Round 1 artifact: `/private/tmp/chemgraphsafe-claudex/inspect-nbwhdefv/result.json`.
Requested model/effort: CLI defaults; actual observed model will be recorded with the result.


## Review loop stopped by user

The user explicitly requested: “Stop claude review. I don't need that for now.” No further Claude reviews or inspection fix rounds will run. The first inspection had already completed when its process was checked; it is no longer running. Its verdict was REVISE, not approval. The findings are retained below without claiming they were fixed. No project code changed after the inspected snapshot.

Observed model: claude-opus-5-5. Completed inspection rounds: 1; fix rounds: 0; further inspection opted out by the user.

Result: `/private/tmp/chemgraphsafe-claudex/inspect-nbwhdefv/result.json`

```json
{
  "verdict": "REVISE",
  "summary": "The implementation mostly matches the amended plan. Model input is built only from the typed Request (text plus context), so source labels and IDs never reach a provider. The SciHazard import is pinned to a full commit and file checksums; it drops responses and rationales and quarantines the 22 rows whose native flag conflicts with their split. Grouping runs before selection, which picks one case per group. The freeze is self-hashed, records an explicit annotations_hash of null, supports inheriting an earlier split, and keeps code hash as provenance only. Run and resume keys include the code hash. The SQLite ledger is shared, resolved from the config root, reserves before every attempt and keeps the reservation when a charge is unknown. Metrics keep errors and abstentions in every denominator, report interception lower/upper bounds, show 100% review burden for always-REVIEW, and give Wilson intervals and per-identity diagnostics. The local pilot manifests agree with docs/verification.md: 250\u2192228 eligible singleton groups, 22 quarantined, 50/50 selected, a 10+10 / 40+40 split, 0 annotations. One material defect remains. The freeze and run identity hash the full ModelConfig, including prices, the pricing-check date, the reservation ceiling, the key variable name and byte limits. The README tells users to re-check prices before spending, and the overrun error tells them to correct pricing. Doing either invalidates the existing pilot freeze and forces a new dataset version. The other findings are low severity: no stop condition for persistent non-auth failures, an untested and unverified Decisions abstention shape, and missing decoding defaults and disclosure in the deferred chat adapter.",
  "findings": [
    {
      "id": "F1",
      "severity": "medium",
      "path": "src/chemgraphsafe/config.py",
      "evidence": "Config.semantic_config() (config.py:117-123) hashes [m.model_dump() for m in self.models]. That dump includes input_usd_per_million, output_usd_per_million, per_request_usd, pricing_checked, reservation_usd, api_key_env, max_input_bytes and endpoint. data.freeze stores semantic_hash (data.py:326-327), and verify_freeze raises 'Frozen policy/model/protocol changed; create a new version' on any difference (data.py:350-351). runner._run_models also puts semantic_hash in run_key (runner.py:114-121). The actual frozen pilot (.local/data/pilot-v1/freeze.json:218-225) pins pricing_checked=2026-10-19, input_usd_per_million=0.042 and reservation_usd=0.01. README:41 says 'verify the prices in configs/pilot.toml, then' run smoke/evaluate. configs/pilot.toml:32 says 'Re-check current rates before spending'. Ledger.settle tells users to 'stop and correct pricing' after an overrun (budget.py:64). Following any of these instructions makes `evaluate` refuse the existing freeze. The user then has to re-download and re-prepare into a new dataset directory with inherited_freeze, and a mid-run correction orphans the partial run. Budgeting parameters are operational settings, not policy, prompt or decoding identity, yet they are treated as frozen experiment semantics. docs/evaluation.md does not warn about this.",
      "fix": "Split ModelConfig hashing into two parts. The semantic identity covers adapter, model, endpoint, max_output_tokens, temperature, seed and anything else that changes the request or parser; only this part goes into the freeze semantic_hash and the run_key. The operational part (prices, pricing_checked, reservation_usd, per_request_usd, api_key_env, max_input_bytes) is recorded in the run manifest and checked only against the ledger. Do the same for the curator config_hash. Add a test showing that a price or reservation change still verifies the freeze and resumes the run, while a change to model ID, endpoint or decoding is rejected. Document which fields are identity and which are operational."
    },
    {
      "id": "F2",
      "severity": "low",
      "path": "src/chemgraphsafe/runner.py",
      "evidence": "FATAL_CODES (runner.py:12-18) holds only budget, overrun, 401, 403 and rate_limit_deferred. Persistent account- or contract-level failures such as http_402 (OpenRouter credits), http_400/404/422 (a wrong alpha Decisions contract) or schema_error on every 200 response do not stop the loop. Every case is still attempted. adapters.py:248-250 keeps the full reservation when the charge is unknown, so each failed case can consume up to 3 shared attempts and $0.01 of reserved budget. Resume then retries all of them again. With 100 pilot cases, a contract mismatch found only at evaluate time can use a large share of the 500-attempt shared ledger.",
      "fix": "Add http_402 to FATAL_CODES. Add a small circuit breaker: stop a model after N consecutive identical non-retryable errors (for example 3 \u00d7 schema_error/400/404/422), mark the remaining cases not_attempted_<code>, and document that smoke must pass before evaluate."
    },
    {
      "id": "F3",
      "severity": "low",
      "path": "src/chemgraphsafe/adapters.py",
      "evidence": "parse_decisions treats answers.triage.abstention == True as a native ABSTAIN (adapters.py:50-51). No test covers this branch: test_documented_refusal_shapes_are_abstentions covers the chat shapes only, and fixtures/predictions.json replays a pre-parsed abstain row. The plan requires native refusal fields to be recognized 'using benign recorded-schema fixtures before freeze'. The field name is assumed, not captured, and the docs do not flag this shape as unverified.",
      "fix": "Add a unit test for the Decisions abstention shape, plus a malformed-abstention test (for example abstention plus an invalid choice). Mark the field as unverified in docs/model-card.md until the live smoke captures the real schema with secrets removed, and record any parser change as a new run version."
    },
    {
      "id": "F4",
      "severity": "low",
      "path": "src/chemgraphsafe/config.py",
      "evidence": "ModelConfig.temperature and seed default to None (config.py:41-42). HTTPModel.payload sends them only when set (adapters.py:159-162). The plan says the chat baseline and curator should 'Request temperature 0 and seed 42 where supported, record supported/requested settings, and explicitly disclose unsupported decoding controls'. A future chat config that omits these fields silently uses provider defaults, and nothing in the manifest or report discloses unsupported controls. This is a deferred track, but the adapter is shipped as the implemented interface.",
      "fix": "For the chat adapter, default temperature=0 and seed=42 (or require them explicitly). Add a per-model 'unsupported_decoding' list that is recorded in the run manifest and printed in report limitations."
    }
  ],
  "coverage": [
    "PLAN.md (full), docs/verification.md, PLAN-REVIEW-LOG.md (structure, host dispositions and authorized-build section)",
    "src/chemgraphsafe: __init__.py, models.py, data.py, adapters.py, budget.py, runner.py, config.py, storage.py, cli.py, reporting.py (all read in full; callers traced: CLI \u2192 runner/data \u2192 adapters/ledger \u2192 reporting)",
    "tests: conftest.py, test_adapters.py, test_contracts.py, test_data.py, test_workflows.py (read in full and mapped to plan test requirements)",
    "configs/pilot.toml, configs/offline.toml, policies/triage-v0.1.json, fixtures/examples.jsonl, fixtures/predictions.json, fixtures/README.md",
    "README.md, CONTRIBUTING.md, docs/evaluation.md, docs/model-card.md, docs/dataset-card.md, docs/contributor-tracks.md, docs/scisafetybench-loader.md",
    ".gitignore, .env.example, .python-version, pyproject.toml, uv.lock header, .github/workflows/ci.yml, .github/ISSUE_TEMPLATE/contribution.yml, .github/pull_request_template.md",
    "Local evidence (aggregate only): .local/data/pilot-v1/feasibility.json, selected fields of prepare.json and freeze.json, runs/offline/report.md; raw SciHazard JSONL inspected only for field-name keys and counts (no prompt/answer text read)"
  ],
  "limitations": [
    "No commands were executed; build, lint, test and CLI results rely on the host's proof evidence.",
    "The OpenRouter alpha Decisions request/response contract (state/questions payload, answers.triage.choice/probabilities/abstention, usage.cost) and Jev pricing could not be checked against live docs or a live call; the adapter's wire shape is reviewed only for internal consistency and against the repository's own mocked tests.",
    "No Git baseline exists; review covered the declared added-file inventory, not a diff.",
    "uv 0.8.18 compatibility with lockfile revision 3 in CI was not verified; remote CI has not run.",
    "SQLite transaction behavior under real concurrent processes was assessed by reading the code, not tested."
  ]
}
```
