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
The optional `openai-codex==0.144.4` SDK bundles a matching runtime, following ChemGraph's pin.
See the [official SDK documentation](https://learn.chatgpt.com/docs/codex-sdk).

Edit `configs/codex.toml` and replace `codex:REPLACE_WITH_ACCOUNT_MODEL_ID` with an explicit
account-supported model. There is no automatic default/model substitution. API-key-authenticated
sessions are rejected, and API-key environment overrides are cleared for the SDK process.
Use `uv run --extra codex ...` when invoking the CLI so uv retains the optional dependency.

Each attempt uses a fresh ephemeral thread and temporary working directory. Runtime settings
disable shell, browsing, apps/plugins, delegation and memory features; inherited MCP servers
are explicitly disabled. Read-only sandboxing and denied approvals provide additional limits.
The pinned runtime retains some native utility tools; any native tool event interrupts the turn
and becomes `codex_tool_attempt`, never an accepted classification. No ChemGraph tools are exposed.

The SDK receives the same classification schema as the HTTP adapter. The runner enforces its
configured timeout, interrupts timed-out turns, and closes the runtime. Available cumulative
token counts survive parsing and transport failures and are summed across adapter retries.
`max_output_tokens` applies only to HTTP requests: this SDK has no per-turn generation-token cap.
Temperature and seed are rejected for Codex. Returned model identity stays null when the runtime
does not provide it; the requested ID remains in the manifest. Free-text refusals remain schema
errors because this SDK contract exposes no dedicated classification-refusal field.

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

## Validation status

CI uses harmless fixtures, mocked HTTP, and fake SDK sessions. An optional-SDK test also parses
notifications with the actual pinned SDK types. These checks do not establish live endpoint
availability or model quality. Run the five-case smoke command before a live pilot, retaining
the same ledger. This integration does not perform a pilot or alter any existing freeze.
