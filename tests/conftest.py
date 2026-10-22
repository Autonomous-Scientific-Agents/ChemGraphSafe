import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from chemgraphsafe.config import Config, DataConfig, ModelConfig


@pytest.fixture
def fake_codex(monkeypatch):
    """Network-free SDK double, including async cancellation and notifications."""
    from chemgraphsafe import codex

    state = SimpleNamespace(
        account="chatgpt",
        configs=[],
        threads=[],
        turns=[],
        closed=0,
        interrupted=0,
        outcome="completed",
        content='{"decision":"ALLOW"}',
        hang=False,
        startup_hang=False,
        exception=None,
        events=None,
        preflights=0,
        catalog=["test-model", "test", "baseline", "curator"],
        efforts=["low", "medium", "high"],
        effective={"model_reasoning_effort": "high", "model_verbosity": "medium"},
        started_overrides={},
        runtime_version="0.144.4",
        catalog_pages=None,
    )

    class CodexError(Exception):
        pass

    class TransportClosedError(CodexError):
        pass

    class ServerBusyError(CodexError):
        pass

    class Turn:
        id = "turn-1"

        async def interrupt(self):
            state.interrupted += 1

        async def stream(self):
            if state.events is not None:
                for event in state.events:
                    yield event
                return
            for count in (3, 12):
                yield SimpleNamespace(
                    method="thread/tokenUsage/updated",
                    payload={
                        "turnId": self.id,
                        "tokenUsage": {
                            "total": {"inputTokens": count, "outputTokens": 4},
                            "last": {"inputTokens": 1, "outputTokens": 1},
                        },
                    },
                )
            if state.hang:
                await asyncio.Event().wait()
            if state.exception:
                raise state.exception
            yield SimpleNamespace(
                method="item/completed",
                payload={
                    "turnId": self.id,
                    "item": {
                        "type": "agentMessage",
                        "phase": "final_answer",
                        "text": state.content,
                    },
                },
            )
            # Commentary after a final answer must never replace it.
            yield SimpleNamespace(
                method="item/completed",
                payload={
                    "turnId": self.id,
                    "item": {
                        "type": "agentMessage",
                        "phase": "commentary",
                        "text": "not JSON",
                    },
                },
            )
            yield SimpleNamespace(
                method="turn/completed",
                payload={
                    "turn": {
                        "id": self.id,
                        "status": state.outcome,
                        "error": {"codexErrorInfo": getattr(state, "failure", "other")},
                    },
                },
            )

    class Thread:
        def __init__(self, codex, id_):
            self.id = id_

        async def turn(self, prompt, **kwargs):
            state.turns.append((prompt, kwargs))
            return Turn()

    class Client:
        def __init__(self, config):
            self.config = config

        async def request(self, method, params, **kwargs):
            if method == "model/list":
                if state.catalog_pages is not None:
                    return state.catalog_pages[params.get("cursor")]
                return {
                    "data": [
                        {
                            "id": name,
                            "model": name,
                            "supportedReasoningEfforts": [
                                {"reasoningEffort": effort} for effort in state.efforts
                            ],
                        }
                        for name in state.catalog
                    ]
                }
            assert method == "config/read"
            import tomllib

            effective = dict(state.effective)
            for override in self.config.config_overrides:
                if override.startswith("model_reasoning_effort="):
                    effective.update(tomllib.loads(override))
            return {"config": {**effective, "mcp_servers": {"inherited.server": {"enabled": True}}}}

        async def thread_start(self, params):
            state.threads.append(params)
            return {
                "thread": {"id": "thread"},
                "model": params["model"],
                "modelProvider": params["modelProvider"],
                "reasoningEffort": params["config"]["model_reasoning_effort"],
                "serviceTier": "default",
                **state.started_overrides,
            }

    class AsyncCodex:
        def __init__(self, config):
            state.configs.append(config)
            self._client = Client(config)

        @property
        def metadata(self):
            return {"serverInfo": {"name": "codex", "version": state.runtime_version}}

        async def __aenter__(self):
            if state.startup_hang:
                await asyncio.Event().wait()
            return self

        async def account(self):
            state.preflights += 1
            return {"account": {"type": state.account} if state.account else None}

        async def close(self):
            state.closed += 1

    sdk = SimpleNamespace(
        __version__="0.144.4",
        AsyncCodex=AsyncCodex,
        AsyncThread=Thread,
        CodexConfig=lambda **kw: SimpleNamespace(**kw),
        ApprovalMode=SimpleNamespace(deny_all="deny"),
        Sandbox=SimpleNamespace(read_only="read-only"),
        CodexError=CodexError,
        TransportClosedError=TransportClosedError,
        ServerBusyError=ServerBusyError,
    )
    # The optional SDK need not be installed for the fake contract tests.
    import sys
    from types import ModuleType

    generated = ModuleType("openai_codex.generated.v2_all")
    generated.ConfigReadResponse = object
    generated.ModelListResponse = object
    generated.ReasoningEffort = str
    monkeypatch.setitem(sys.modules, "openai_codex.generated.v2_all", generated)
    monkeypatch.setattr(codex, "load_sdk", lambda: sdk)
    # Workflow provenance uses the fake SDK's distribution graph even in the no-SDK CI job.
    from chemgraphsafe import provenance

    installed = provenance.distribution

    def distribution(name):
        if name == "openai-codex":
            return SimpleNamespace(
                version=sdk.__version__, requires=["openai-codex-cli-bin", "pydantic"]
            )
        if name == "openai-codex-cli-bin":
            return SimpleNamespace(version=state.runtime_version, requires=[])
        return installed(name)

    monkeypatch.setattr(provenance, "distribution", distribution)
    state.sdk = sdk
    return state


@pytest.fixture
def config(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(repo / "fixtures", tmp_path / "fixtures")
    shutil.copytree(repo / "policies", tmp_path / "policies")
    return Config(
        root=tmp_path,
        offline=True,
        run_dir="runs/offline",
        data=DataConfig(
            loader="fixture", count=10, minimum_per_class=4, directory=".local/data/test"
        ),
        models=[ModelConfig(name="replay-demo", adapter="replay", model="mock-fixture-v1")],
    )
