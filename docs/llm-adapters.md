# LLM evaluation and curation

All adapters implement `predict(Request, policy) -> Prediction`. Only policy and request
text/context enter inference. Evaluation returns `decision`; curation additionally returns an
enumerated `reason`. Responses are validated strictly without JSON repair, free-text explanation
storage, or invented confidence. Configs, authentication, and accounting are preflighted for all
models before the first inference attempt.

## Hosted Argo

Export `ARGO_USER` as your Argonne username; `.env` is not loaded automatically. Start with
`configs/argo.toml`, which uses the hosted `/argoapi/v1/chat/completions` endpoint and
`argo:gpt-4.1-mini`. The adapter follows ChemGraph's Argo configuration: it sends the username
as bearer identity and body `user`, and maps the public model name to `gpt41mini` on the wire.
`OPENAI_API_KEY` is never used by this adapter. `argo_user_env` selects a different identity
environment variable if needed; its value is never stored in run artifacts.

Hosted access depends on your Argonne network/account. The explicit model map comes from
ChemGraph and is not a guarantee of current account availability. Unknown aliases and Claude
models fail configuration validation. Local HTTP shims and Anthropic Messages are not included.
Temperature is rejected for models that ChemGraph marks as not supporting it; otherwise sampling
parameters are omitted unless explicitly configured.

Argo omits `response_format` because ChemGraph marks this route as not supporting structured
output. The prompt contains the required JSON schema. Invalid JSON, extra fields, incomplete
responses, and free-text refusals become ERROR. Native chat `message.refusal` and
`finish_reason=content_filter` become ABSTAIN, distinct from policy REVIEW.

## Codex subscription

Install `uv sync --locked --extra dev --extra codex` and establish a stored ChatGPT login through
Codex CLI (`codex login`) or an existing Codex integration. Use `codex login status` to check it.
The optional SDK requirement is `openai-codex>=0.163.0,<1`. The checked-in `uv.lock` records
the tested SDK and its matching bundled runtime, currently `0.163.0`. The adapter uses that
bundled runtime, so updating a separately installed `codex` CLI does not upgrade this adapter.
See the [official SDK documentation](https://learn.chatgpt.com/docs/codex-sdk).

To upgrade the SDK and runtime deliberately without upgrading unrelated dependencies:

```sh
uv lock --upgrade-package openai-codex --upgrade-package openai-codex-cli-bin
uv sync --locked --extra dev --extra codex
uv run --extra codex pytest -q tests/test_codex_adapter.py
```

Commit the updated lockfile after the contract tests pass. Installs and CI use `--locked`;
the version range does not make existing checkouts silently track new releases. SDK/runtime
versions participate in execution identity, so an upgrade requires a new run directory.
Model availability still depends on the account and runtime catalog; verify the exact model
and effort with a fresh smoke run before a full evaluation. Smoke performs live inference.

Edit `configs/codex.toml` and replace `codex:REPLACE_WITH_ACCOUNT_MODEL_ID` with an explicit
account-supported model. Set `reasoning_effort` explicitly, for example `"medium"`, for both
evaluated models and any Codex curator. No effort or model is silently selected or substituted.
Preflight checks the complete account model catalog, including hidden entries and subsequent
pages, for the requested model and effort. Missing/unsupported selections fail before any turn
or inference reservation. API-key-authenticated
sessions are rejected, and API-key environment overrides are cleared for the SDK process.
Use `uv run --extra codex ...` when invoking the CLI so uv retains the optional dependency.

Each attempt uses a fresh ephemeral thread and temporary working directory. Runtime settings
disable shell, browsing, apps/plugins, delegation and memory features; inherited MCP servers
are explicitly disabled. Read-only sandboxing and denied approvals provide additional limits.
The runtime retains some native utility tools; any native tool event interrupts the turn
and becomes `codex_tool_attempt`, never an accepted classification. No ChemGraph tools are exposed.

The SDK receives the same classification schema as the HTTP adapter. The runner enforces its
configured timeout, interrupts timed-out turns, and closes the runtime. Available cumulative
token counts survive parsing and transport failures and are summed across adapter retries.
`max_output_tokens` applies only to HTTP requests: this SDK has no per-turn generation-token cap.
Temperature and seed are rejected for Codex. Returned model identity stays null when the runtime
does not provide it; the requested ID remains in the manifest. Free-text refusals remain schema
errors because this SDK contract exposes no dedicated classification-refusal field.

The policy and required schema are trusted thread instructions; turn input contains only the
canonical `untrusted_request` object with text/context. Input-byte limits include the trusted
instructions, request, and output schema. The prompt contract is fingerprinted in the freeze.

Preflight initializes the SDK and an ephemeral thread without starting a turn. It retains only
an allowlist of execution metadata: SDK/runtime versions, runtime-selected model/provider,
reasoning effort/summary, verbosity, service tier, and context/compaction settings. Null optional
settings mean the runtime did not expose an explicit value; they are not invented defaults.
No complete user configuration, credential, account information, or temporary path is stored.
Reasoning effort overrides personal defaults. Other recorded settings may inherit user defaults;
changing them requires a new run. Every attempt checks the runtime contract again before its
turn; drift becomes fatal `codex_configuration_changed`. These checks cannot identify an
unexposed checkpoint behind a mutable model alias, and runtime-selected names are not recorded
as provider-returned checkpoint identities. A rejected attempt still counts as an adapter
invocation under the existing attempts-only accounting.

The attempt limit counts adapter invocations/retries, not hidden model requests within one Codex
turn. Cumulative token usage includes those internal requests. Neither the local ledger nor the
timeout controls ChatGPT subscription quotas or billing.

## Generic OpenAI-compatible endpoints

Use `adapter="chat"`, an explicit model, the full HTTPS chat-completions URL in `endpoint`, and
`api_key_env` naming your credential variable. This route retains strict JSON-schema
`response_format`. It does not apply Argo model aliases or inject an Argo username.
Dollar-metered runs require checked input/output token prices, `pricing_checked`, and a positive
`reservation_usd`. Optional temperature and seed are sent only when configured; verify provider
support before selecting them. A compatibility claim requires a live smoke test.

## Accounting and versioning

`budget_mode="usd"` is the default. Metered requests retain the existing conservative reservation,
unknown-charge retention, and overrun handling. `budget_mode="attempts_only"` is an explicit opt-in
for included usage and cannot be combined with dollar prices/reservations. Codex requires it;
the Argo example selects it. Genuine provider-reported costs are retained when present, but are
outside the metered ledger in this mode. Missing costs are null, not a claim of free service.

One SQLite ledger governs smoke, curation, evaluation and versioned reruns. Existing ledger rows
migrate as metered without changing their charges or limits. Interrupted included calls still
count as attempts without becoming unknown dollar charges. Reports distinguish accounting modes
and show included usage separately. Exhausting the attempt cap stops both modes; exhausting only
the dollar cap stops metered calls.

For evaluation, use the new data/run directories in the examples. To retain an existing pilot's
examples, splits and ordering, enable `data.inherited_freeze` pointing to its `freeze.json` before
preparation. Keep the original dataset, runs and shared budget ledger intact.

The execution-identity and trusted-prompt changes require a fresh Codex data/run version if the
example directories were already used. Older reports remain readable. See
[evaluation versioning](evaluation.md) for compatibility and provenance details.

```sh
uv run chemgraphsafe data prepare --config configs/argo.toml
# Optional, after enabling the commented distinct-model [curator] block:
uv run chemgraphsafe curate --config configs/argo.toml
uv run chemgraphsafe data freeze --config configs/argo.toml
uv run chemgraphsafe evaluate --config configs/argo.toml
uv run chemgraphsafe report --run runs/argo-v1
```

Skip `curate` for source-label-only evaluation. For Codex substitute `configs/codex.toml`, use
`uv run --extra codex`, and report `runs/codex-v1`. Every curator model must differ from all
evaluated models. Known Codex/Argo prefixes and wire aliases are normalized when checking
collisions; unobserved checkpoint aliases cannot be proven distinct. Labels remain
`llm_provisional`. Follow [evaluation versioning](evaluation.md) when introducing labels later.

Two explicit Codex presets are also provided:

- `configs/codex-online-test.toml`: five harmless smoke cases using `gpt-5.6-sol`, medium effort,
  and no retries. Run it with `smoke`; its output is under `runs/codex-pr3-online-v1/smoke`.
- `configs/codex-astra-full.toml`: `gpt-6-astra`, high effort, a 180-second timeout, and one retry.
  Its data preparation selects all 228 eligible SciHazard chemistry cases: 50 benign and 178
  harmful. The 22 source-label conflicts remain excluded. The new freeze has 46 development
  and 182 evaluation cases; evaluation processes both splits and reports them separately.
  After a successful smoke run, use `data prepare`, `data freeze`, and `evaluate` with this config,
  then report `runs/codex-astra-full-v1`. This larger pool cannot inherit the 100-case pilot freeze.

Both presets retain the shared ledger and require account access to their explicit model/effort.
Use new run directories after an SDK/runtime or inference-code change; do not overwrite an old
manifest to resume. The full preset creates no curator labels and remains a source-label benchmark.

## Validation status

CI uses harmless fixtures, mocked HTTP, and fake SDK sessions. An optional-SDK test also parses
notifications with the actual locked SDK types. These checks do not establish live endpoint
availability or model quality. Run the five-case smoke command before a live pilot, retaining
the same ledger. This integration does not perform a pilot or alter any existing freeze.
