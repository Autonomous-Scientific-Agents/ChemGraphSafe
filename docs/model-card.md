# Model integration card

The original pilot config selects Jev through OpenRouter:

- Endpoint: `https://openrouter.ai/api/alpha/decisions`.
- Requested ID: `typesafe/jev-1.14` (an alias; returned identity is recorded).
- Secret: `OPENROUTER_API_KEY` environment variable, never stored in artifacts.
- Task: one typed choice question with ALLOW/REVIEW/BLOCK criteria under policy v0.1.

The contract follows [OpenRouter's Jev introduction](https://openrouter.ai/blog/insights/what-is-jev/).
This endpoint uses `state` and `questions`, not chat messages. The framework validates the
native choice and all three probabilities without renormalizing. Provider choice is authoritative;
choice/argmax disagreement is flagged. Provider-specific confidence is not treated as calibration.
Check endpoint access, current prices, and returned schema with the five-case smoke command
before a paid pilot. Mock HTTP contract tests are not evidence of live API compatibility.

The pilot configuration records the checked date, input/output prices, a reservation ceiling,
timeout, and retry limits. Reservations use the greater of the configured ceiling and a visible
input-byte/output-token estimate. Hidden provider costs can exceed an estimate: an observed
overrun is recorded and stops further calls. No local ledger can guarantee the provider's bill.
Requests with input beyond the configured byte limit become explicit errors rather than truncation.

HTTP 429, server errors, and transport failures allow at most two retries. Retry-After is respected;
delays above the configured cap stop instead of retrying early. Authentication, budget, and
reservation failures stop remaining calls for that model with explicit unattempted outcomes.

## LLM baseline and curator

The `chat` adapter supports configurable HTTPS JSON-schema chat-completion endpoints. It requests
only a decision; a curator additionally returns one of three enumerated reason codes. Native
`message.refusal` and `finish_reason=content_filter` become ABSTAIN. Unstructured refusal text,
incomplete output, or invalid JSON becomes ERROR. Raw responses and free-text explanations are
not persisted. Argo and Codex have separate example configs and the same classification contract;
see [LLM setup](llm-adapters.md). Their live behavior has not been measured by the offline tests.

Configure `[curator]` using the same fields as `[[models]]`, selecting `chat`, `argo`, or `codex`
and an explicit distinct model ID. Generic metered chat needs an HTTPS endpoint, credential
environment variable, checked prices and reservation. Argo uses an Argonne username; Codex uses
a stored ChatGPT login. Both examples explicitly select attempts-only accounting. Create a new
dataset/run version and inherit the original split as documented
in [evaluation.md](evaluation.md). Curator labels remain `llm_provisional`; do not score a model
against its own generated labels as independent evidence.

Kev and Laya are future adapters, not installed integrations. Confirm upstream protocols and
model licenses before implementing their serving/training paths. A generic LoRA pipeline is not
assumed to fit either architecture. The `replay` adapter is restricted to explicit offline runs.
