"""Subscription-backed classification using ChemGraph's pinned Codex SDK contract."""

import asyncio
import importlib
import tempfile
import time
from contextlib import suppress

from .adapters import parse_classification
from .budget import BudgetExceeded
from .models import Prediction
from .prompts import (
    CODEX_BASE_INSTRUCTIONS,
    classification_input,
    classification_instructions,
    classification_schema,
)
from .storage import canonical, digest

# Disable executable/provider tools and inherited integrations. A read-only sandbox alone
# still permits shell reads. The runtime retains utility tools; any tool event is rejected.
DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "shell_zsh_fork",
    "apply_patch_freeform",
    "apps",
    "connectors",
    "plugins",
    "remote_plugin",
    "plugin_hooks",
    "hooks",
    "codex_hooks",
    "multi_agent",
    "multi_agent_v2",
    "collab",
    "enable_fanout",
    "goals",
    "js_repl",
    "code_mode",
    "code_mode_host",
    "code_mode_only",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "in_app_browser",
    "image_generation",
    "imagegenext",
    "memories",
    "memory_tool",
    "tool_suggest",
    "skill_mcp_dependency_install",
    "request_permissions_tool",
    "request_permissions",
    "default_mode_request_user_input",
    "token_budget",
    "current_time_reminder",
    "deferred_executor",
    "search_tool",
    "tool_search",
    "standalone_web_search",
)
RETRYABLE = {"codex_network_error", "codex_overloaded", "codex_timeout"}
# Only scalar inference settings are retained; never serialize the full user configuration.
EFFECTIVE_SETTINGS = (
    "model_reasoning_summary",
    "model_verbosity",
    "model_context_window",
    "model_auto_compact_token_limit",
    "model_auto_compact_token_limit_scope",
)


class AuthenticationError(ValueError):
    """The stored account is not ChatGPT-managed."""


class ConfigurationError(ValueError):
    """Fixed, non-sensitive configuration failure code."""


def load_sdk():
    try:
        return importlib.import_module("openai_codex")
    except ImportError:
        raise ValueError("Codex requires `uv sync --extra codex` (openai-codex==0.144.4)") from None


def plain(value):
    return value.model_dump(mode="json", by_alias=True) if hasattr(value, "model_dump") else value


def require_chatgpt(response):
    value = plain(response)
    account = plain(value.get("account")) if isinstance(value, dict) else None
    if not isinstance(account, dict) or account.get("type") != "chatgpt":
        raise AuthenticationError("Codex requires a stored ChatGPT login; run `codex login` first")


def error_code(error):
    """Map structured SDK codes only; never retain provider exception messages."""
    value = plain(error)
    info = value.get("codexErrorInfo") if isinstance(value, dict) else None
    if info in ("unauthorized",):
        return "codex_authentication"
    if info in ("usageLimitExceeded", "sessionBudgetExceeded"):
        return "codex_quota_exhausted"
    if info in ("serverOverloaded", "internalServerError"):
        return "codex_overloaded"
    if isinstance(info, dict):
        for detail in info.values():
            if isinstance(detail, dict):
                status = detail.get("httpStatusCode")
                if status in (401, 403):
                    return "codex_authentication"
                if status == 429:
                    return "codex_quota_exhausted"
                if isinstance(status, int) and status >= 500:
                    return "codex_overloaded"
    return "codex_error"


def token_fields(usage):
    total = usage.get("total", {}) if isinstance(usage, dict) else {}
    if not isinstance(total, dict):
        return {}
    result = {}
    for target, source in (("input_tokens", "inputTokens"), ("output_tokens", "outputTokens")):
        count = total.get(source)
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            result[target] = count
    return result


async def validate_selection(codex, model, effort):
    """Check the entire account catalog, including hidden entries and paginated results."""
    from openai_codex.generated.v2_all import ModelListResponse

    cursor, seen = None, set()
    while True:
        params = {"includeHidden": True}
        if cursor is not None:
            params["cursor"] = cursor
        page = plain(
            await codex._client.request("model/list", params, response_model=ModelListResponse)
        )
        for entry in page["data"]:
            if model in (entry["id"], entry["model"]):
                supported = {v["reasoningEffort"] for v in entry["supportedReasoningEfforts"]}
                if effort not in supported:
                    raise ConfigurationError("codex_effort_unsupported")
                return
        cursor = page.get("nextCursor")
        if cursor is None:
            raise ConfigurationError("codex_model_unavailable")
        if cursor in seen:
            raise ConfigurationError("codex_catalog_invalid")
        seen.add(cursor)


async def start_thread(sdk, codex, *, model, cwd, instructions, config):
    """Keep the typed start response that the high-level SDK otherwise discards."""
    started = plain(
        await codex._client.thread_start(
            {
                "model": model,
                "modelProvider": "openai",
                "cwd": cwd,
                "ephemeral": True,
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "personality": "none",
                "baseInstructions": CODEX_BASE_INSTRUCTIONS,
                "developerInstructions": instructions,
                "config": config,
            }
        )
    )
    return sdk.AsyncThread(codex, started["thread"]["id"]), started


def runtime_identity(sdk, codex, effective, started, instructions, schema):
    server = plain(codex.metadata)["serverInfo"]
    result = {
        "sdk_version": sdk.__version__,
        "runtime_version": server["version"],
        "runtime_name": server["name"],
        "model": started["model"],
        "model_provider": started["modelProvider"],
        "reasoning_effort": started["reasoningEffort"],
        "service_tier": started.get("serviceTier"),
        **{key: effective.get(key) for key in EFFECTIVE_SETTINGS},
        "instructions_hash": digest(
            {"base": CODEX_BASE_INSTRUCTIONS, "developer": instructions, "schema": schema}
        ),
    }
    for key in ("sdk_version", "runtime_version", "runtime_name", "model", "reasoning_effort"):
        if not isinstance(result[key], str) or not result[key]:
            raise ConfigurationError("codex_configuration_unavailable")
    if any(value is not None and not isinstance(value, (str, int)) for value in result.values()):
        raise ConfigurationError("codex_configuration_unavailable")
    return result


async def consume(turn, common):
    final, unphased, completed = None, None, None
    stream = turn.stream()
    try:
        async for event in stream:
            payload = plain(event.payload)
            if not isinstance(payload, dict):
                continue
            if event.method == "turn/completed" and payload.get("turn", {}).get("id") == turn.id:
                completed = payload["turn"]
            elif payload.get("turnId") == turn.id:
                if event.method == "thread/tokenUsage/updated":
                    common.update(token_fields(payload.get("tokenUsage")))
                elif event.method in {"item/started", "item/completed"}:
                    item = payload.get("item", {})
                    kind = item.get("type")
                    if kind not in {"agentMessage", "reasoning", "userMessage"}:
                        return None, "codex_tool_attempt"
                    if event.method == "item/completed" and kind == "agentMessage":
                        if item.get("phase") == "final_answer":
                            final = item.get("text")
                        elif item.get("phase") is None:
                            unphased = item.get("text")
    finally:
        await stream.aclose()
    if completed is None:
        return None, "codex_incomplete"
    if completed.get("status") != "completed":
        if completed.get("status") == "interrupted":
            return None, "codex_interrupted"
        return None, error_code(completed.get("error"))
    return final if final is not None else unphased, None


class CodexModel:
    def __init__(self, config, model, ledger, run_key, *, curator=False, sleep=time.sleep):
        self.config, self.model = config, model
        self.ledger, self.run_key = ledger, run_key
        self.curator, self.sleep = curator, sleep
        if "REPLACE_WITH" in model.model:
            raise ValueError(
                "Replace the Codex model placeholder with an account-supported model ID"
            )
        self.sdk = load_sdk()
        self.runtime_provenance = None
        result = asyncio.run(self._invoke())
        if result.status == "error":
            raise ValueError(
                f"Codex preflight failed: {result.error_code}; check ChatGPT login, model, "
                "and reasoning_effort"
            )

    async def _invoke(self, request=None, policy=None):
        common, turn, consumer = {}, None, None
        result = Prediction(status="error", error_code="codex_error")
        schema = classification_schema(
            policy if policy is not None else self.config.policy, self.curator
        )
        instructions = classification_instructions(
            policy if policy is not None else self.config.policy, schema
        )
        model_id = self.model.model.removeprefix("codex:")
        with tempfile.TemporaryDirectory(prefix="chemgraphsafe-codex-") as cwd:
            overrides = tuple(f"features.{name}=false" for name in DISABLED_FEATURES) + (
                'web_search="disabled"',
                'history.persistence="none"',
                'model_provider="openai"',
                'forced_login_method="chatgpt"',
                "project_doc_max_bytes=0",
                'developer_instructions=""',
                "include_environment_context=false",
                "include_apps_instructions=false",
                "model_reasoning_effort=" + canonical(self.model.reasoning_effort),
            )
            sdk_config = self.sdk.CodexConfig(
                cwd=cwd,
                env={"OPENAI_API_KEY": "", "CODEX_API_KEY": ""},
                config_overrides=overrides,
                client_name="chemgraphsafe",
                client_title="ChemGraphSafe",
            )
            codex = self.sdk.AsyncCodex(config=sdk_config)
            try:
                async with asyncio.timeout(self.config.runner.timeout_seconds):
                    await codex.__aenter__()
                    require_chatgpt(await codex.account())
                    from openai_codex.generated.v2_all import ConfigReadResponse

                    if request is None:
                        await validate_selection(codex, model_id, self.model.reasoning_effort)
                    effective = plain(
                        await codex._client.request(
                            "config/read",
                            {"includeLayers": False, "cwd": cwd},
                            response_model=ConfigReadResponse,
                        )
                    )["config"]
                    thread_config = {
                        "model_reasoning_effort": self.model.reasoning_effort,
                        "mcp_servers": {
                            name: {"enabled": False} for name in effective.get("mcp_servers", {})
                        },
                    }
                    thread, metadata = await start_thread(
                        self.sdk,
                        codex,
                        model=model_id,
                        cwd=cwd,
                        instructions=instructions,
                        config=thread_config,
                    )
                    observed = runtime_identity(
                        self.sdk, codex, effective, metadata, instructions, schema
                    )
                    if (
                        observed["reasoning_effort"] != self.model.reasoning_effort
                        or observed["model_provider"] != "openai"
                        or (
                            self.runtime_provenance is not None
                            and observed != self.runtime_provenance
                        )
                    ):
                        raise ConfigurationError("codex_configuration_changed")
                    if request is None:
                        self.runtime_provenance = observed
                        result = Prediction(status="abstain", error_code="preflight_only")
                    else:
                        from openai_codex.generated.v2_all import ReasoningEffort

                        turn = await thread.turn(
                            classification_input(request),
                            effort=ReasoningEffort(self.model.reasoning_effort),
                            output_schema=schema,
                            approval_mode=self.sdk.ApprovalMode.deny_all,
                            sandbox=self.sdk.Sandbox.read_only,
                        )
                        # The SDK waits on a thread-backed queue. Cancelling its stream first
                        # unregisters that queue, stranding the worker during asyncio shutdown.
                        # Keep it registered until interruption/runtime closure wakes the reader.
                        consumer = asyncio.create_task(consume(turn, common))
                        content, failure = await asyncio.shield(consumer)
                        if failure:
                            result = Prediction(status="error", error_code=failure, **common)
                        else:
                            try:
                                result = parse_classification(content, self.curator, **common)
                            except (ValueError, TypeError, KeyError):
                                result = Prediction(
                                    status="error", error_code="schema_error", **common
                                )
            except TimeoutError:
                result = Prediction(status="error", error_code="codex_timeout", **common)
            except AuthenticationError:
                result = Prediction(status="error", error_code="codex_authentication", **common)
            except ConfigurationError as exc:
                result = Prediction(status="error", error_code=str(exc), **common)
            except self.sdk.TransportClosedError:
                result = Prediction(status="error", error_code="codex_network_error", **common)
            except self.sdk.ServerBusyError:
                result = Prediction(status="error", error_code="codex_overloaded", **common)
            except self.sdk.CodexError as exc:
                result = Prediction(
                    status="error", error_code=error_code(getattr(exc, "data", None)), **common
                )
            except (OSError, KeyError, TypeError, ValueError, RuntimeError):
                result = Prediction(status="error", error_code="codex_runtime_error", **common)
            finally:
                if turn is not None and result.status == "error":
                    with suppress(Exception):
                        await asyncio.wait_for(turn.interrupt(), timeout=0.5)
                # Closing the SDK runtime also releases outstanding blocking notification reads.
                try:
                    await codex.close()
                except (self.sdk.CodexError, OSError, RuntimeError):
                    result = Prediction(status="error", error_code="codex_runtime_error", **common)
                finally:
                    if consumer is not None:
                        if not consumer.done():
                            consumer.cancel()
                        with suppress(asyncio.CancelledError, Exception):
                            await consumer
        return result.model_copy(update=common)

    def predict(self, request, policy):
        schema = classification_schema(policy, self.curator)
        submitted = {
            "base_instructions": CODEX_BASE_INSTRUCTIONS,
            "developer_instructions": classification_instructions(policy, schema),
            "input": classification_input(request),
            "output_schema": schema,
        }
        if len(canonical(submitted).encode()) > self.model.max_input_bytes:
            return Prediction(status="error", error_code="input_too_large")
        started = time.perf_counter()
        attempts, overhead = 0, 0.0
        totals = {}
        for index in range(self.config.runner.retries + 1):
            try:
                reservation = self.ledger.reserve(self.run_key, 0, budget_mode="attempts_only")
            except BudgetExceeded:
                result = Prediction(status="error", error_code="budget_exhausted")
                break
            attempts += 1
            attempt_started = time.perf_counter()
            result = asyncio.run(self._invoke(request, policy))
            for field in ("input_tokens", "output_tokens"):
                value = getattr(result, field)
                if value is not None:
                    totals[field] = totals.get(field, 0) + value
            self.ledger.settle(reservation, None, result.status)
            if result.error_code not in RETRYABLE or index == self.config.runner.retries:
                break
            delay = 2**index
            if delay > self.config.runner.max_retry_after_seconds:
                result = result.model_copy(update={"error_code": "rate_limit_deferred"})
                break
            overhead += (time.perf_counter() - attempt_started + delay) * 1000
            self.sleep(delay)
        return result.model_copy(
            update={
                **totals,
                "attempts": attempts,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "retry_overhead_ms": overhead,
            }
        )

    def close(self):
        pass  # Every attempt owns and closes its runtime.
