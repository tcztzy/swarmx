"""Tests for the CLI module."""

import json
import runpy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from swarmx.cli import app, main
from swarmx.server import create_server_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def temp_swarmx_file():
    """Create a temporary SwarmX file for testing."""
    data = {
        "name": "test_agent",
        "instructions": "You are a test agent.",
        "model": "gpt-4o",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()  # Ensure data is written
        yield Path(f.name)
    Path(f.name).unlink()


@pytest.fixture
def temp_output_file():
    """Create a temporary output file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        yield Path(f.name)
    Path(f.name).unlink()


def test_cli_main_execution():
    """Test CLI main execution."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "SwarmX Command Line Interface" in result.stdout


async def test_main_stream_cycle_writes_output(tmp_path: Path, monkeypatch):
    """Ensure main loop handles prompts, streaming chunks, and output writing."""

    def make_chunk(
        content: str | None,
        *,
        reasoning: str | None = None,
        refusal: str | None = None,
        finish_reason: str | None = None,
    ):
        delta = SimpleNamespace(
            content=content,
            reasoning_content=reasoning,
            refusal=refusal,
        )
        choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
        chunk = SimpleNamespace(choices=[choice])
        chunk.model_dump_json = lambda: json.dumps({"content": content})
        return chunk

    config_file = tmp_path / "swarm.json"
    config_file.write_text(json.dumps({"config": True}))

    stream_chunks = [
        make_chunk("hi", reasoning="thinking"),
        make_chunk(None, refusal="nope", finish_reason="stop"),
    ]

    class DummyAgent:
        def __init__(self):
            self.name = "primary"
            self.agents = {"primary": self}
            self.calls: list[dict[str, object]] = []

        @classmethod
        def model_validate(cls, data):
            return cls()

        async def run(self, *, stream: bool, **kwargs):
            self.calls.append({"stream": stream, **kwargs})

            if stream:

                async def generator():
                    for chunk in stream_chunks:
                        yield chunk

                return generator()
            return [{"role": "assistant", "content": "done"}]

    monkeypatch.setattr("swarmx.cli.Agent", DummyAgent)

    prompt_calls = {"count": 0}

    def fake_prompt(*args, **kwargs):
        if prompt_calls["count"] == 0:
            prompt_calls["count"] += 1
            return "hello"
        raise KeyboardInterrupt

    echo_calls: list[tuple[str, dict[str, object]]] = []
    secho_calls: list[tuple[str, dict[str, object]]] = []

    def fake_echo(message: str = "", **kwargs):
        echo_calls.append((message, kwargs))

    def fake_secho(message: str = "", **kwargs):
        secho_calls.append((message, kwargs))

    monkeypatch.setattr(typer, "prompt", fake_prompt)
    monkeypatch.setattr(typer, "echo", fake_echo)
    monkeypatch.setattr(typer, "secho", fake_secho)

    output_file = tmp_path / "conversation.json"
    await main(file=config_file, output=output_file, verbose=True)

    saved = json.loads(output_file.read_text())
    assert saved == [
        {
            "role": "user",
            "content": "hello",
        }
    ]
    assert any(message == "nope" for message, _ in secho_calls)
    assert any(message == "" for message, _ in echo_calls)


async def test_main_appends_refusal_on_error(tmp_path: Path, monkeypatch):
    """Capture exceptions from the agent and persist them to the transcript."""

    class FailingAgent:
        def __init__(self):
            self.name = "primary"
            self.agents = {"primary": self}

        @classmethod
        def model_validate(cls, data):
            return cls()

        async def run(self, *, stream: bool, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr("swarmx.cli.Agent", FailingAgent)

    prompt_calls = {"count": 0}

    def fake_prompt(*args, **kwargs):
        if prompt_calls["count"] == 0:
            prompt_calls["count"] += 1
            return "question"
        raise AssertionError("prompt called again unexpectedly")

    messages: list[str] = []

    def fake_secho(message: str = "", **kwargs):
        messages.append(message)

    monkeypatch.setattr(typer, "prompt", fake_prompt)
    monkeypatch.setattr(typer, "secho", fake_secho)
    monkeypatch.setattr(typer, "echo", lambda *args, **kwargs: None)

    output_file = tmp_path / "error.json"
    await main(output=output_file)

    saved = json.loads(output_file.read_text())
    assert saved[0]["content"] == "question"
    assert saved[1]["refusal"] == "boom"
    assert messages[-1] == "boom"


def test_repl_invokes_main(monkeypatch):
    """The Typer callback should invoke the async main entry point."""
    captured: dict[str, object] = {}

    async def fake_main(*, file=None, output=None, verbose=False):
        captured["kwargs"] = {
            "file": file,
            "output": output,
            "verbose": verbose,
        }

    monkeypatch.setattr("swarmx.cli.main", fake_main)

    runner = CliRunner()
    result = runner.invoke(app, ["--verbose"])
    assert result.exit_code == 0
    assert captured["kwargs"] == {"file": None, "output": None, "verbose": True}


def test_cli_serve_invokes_uvicorn(monkeypatch, temp_swarmx_file):
    """Ensure the serve command builds the app and invokes uvicorn."""

    created: dict[str, object] = {}

    def capture_create_server_app(agent, auto_execute_tools: bool = True):
        created["agent"] = agent
        created["auto_execute_tools"] = auto_execute_tools
        return create_server_app(agent, auto_execute_tools=auto_execute_tools)

    run_args: dict[str, object] = {}

    def fake_uvicorn_run(app_obj, host: str, port: int):
        run_args["app"] = app_obj
        run_args["host"] = host
        run_args["port"] = port

    monkeypatch.setattr("swarmx.cli.create_server_app", capture_create_server_app)
    monkeypatch.setattr("swarmx.cli.uvicorn.run", fake_uvicorn_run)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--no-auto-execute-tools",
            "--file",
            str(temp_swarmx_file),
        ],
    )
    assert result.exit_code == 0
    assert created["auto_execute_tools"] is False
    assert getattr(created["agent"], "name") == "test_agent"
    assert "app" in run_args
    assert run_args["host"] == "0.0.0.0"
    assert run_args["port"] == 9001
    assert getattr(run_args["app"], "title") == "SwarmX API"


def test_cli_serve_defaults_to_empty_config(monkeypatch):
    """Serve command should fall back to an empty configuration when no file is provided."""
    from typing import TypedDict

    from fastapi import FastAPI

    from swarmx import Agent

    class Created(TypedDict, total=False):
        agent: Agent
        auto_execute_tools: bool
        app: FastAPI

    created: Created = {}

    def capture_create_server_app(agent, auto_execute_tools: bool = True):
        created["agent"] = agent
        created["auto_execute_tools"] = auto_execute_tools
        created["app"] = create_server_app(agent, auto_execute_tools=auto_execute_tools)
        return created["app"]

    run_args: dict[str, object] = {}

    def fake_uvicorn_run(app_obj, host: str, port: int):
        run_args["app"] = app_obj
        run_args["host"] = host
        run_args["port"] = port

    monkeypatch.setattr("swarmx.cli.create_server_app", capture_create_server_app)
    monkeypatch.setattr("swarmx.cli.uvicorn.run", fake_uvicorn_run)

    runner = CliRunner()
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0
    agent = created.get("agent")
    assert agent is not None
    assert agent.name == "Agent"
    assert agent.agents.get(agent.name) is agent
    assert created.get("auto_execute_tools")
    assert created.get("app") is run_args["app"]
    assert run_args["host"] == "127.0.0.1"
    assert run_args["port"] == 8000


def test_cli_mcp_command_runs_server(monkeypatch, temp_swarmx_file):
    """Ensure the MCP command constructs and runs the server."""

    class DummyAgent:
        def __init__(self):
            self.name = "primary"
            self.agents = {"primary": self}

        @classmethod
        def model_validate(cls, data):
            instance = cls()
            instance.data = data
            return instance

    ran = {}

    class DummyServer:
        def __init__(self, agent):
            self.agent = agent

        def run(self):
            ran["called"] = True

    def fake_create_mcp_server(agent):
        ran["agent"] = agent
        return DummyServer(agent)

    monkeypatch.setattr("swarmx.cli.Agent", DummyAgent)
    monkeypatch.setattr("swarmx.cli.create_mcp_server", fake_create_mcp_server)

    runner = CliRunner()
    result = runner.invoke(app, ["mcp", "--file", str(temp_swarmx_file)])
    assert result.exit_code == 0
    assert ran["called"] is True
    assert ran["agent"].data["name"] == "test_agent"


def test_cli_mcp_command_uses_default_config(monkeypatch):
    """MCP command should work without an explicit configuration file."""

    class DummyAgent:
        def __init__(self):
            self.name = "primary"
            self.agents = {"primary": self}

        @classmethod
        def model_validate(cls, data):
            instance = cls()
            instance.data = data
            return instance

    ran: dict[str, object] = {}

    class DummyServer:
        def __init__(self, agent):
            ran["agent"] = agent

        def run(self):
            ran["called"] = True

    monkeypatch.setattr("swarmx.cli.Agent", DummyAgent)
    monkeypatch.setattr(
        "swarmx.cli.create_mcp_server", lambda agent: DummyServer(agent)
    )

    runner = CliRunner()
    result = runner.invoke(app, ["mcp"])
    assert result.exit_code == 0
    assert ran["called"] is True
    assert ran["agent"].data == {}


def test_module_entry_point_invokes_app(monkeypatch):
    """Running python -m swarmx should invoke the Typer app."""
    called = {}

    class DummyApp:
        def __call__(self):
            called["invoked"] = True

    monkeypatch.setattr("swarmx.cli.app", DummyApp())
    runpy.run_module("swarmx.__main__", run_name="__main__")
    assert called["invoked"] is True
