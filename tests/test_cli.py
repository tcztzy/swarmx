"""Tests for the CLI module."""

import json
import runpy
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swarmx.cli import app
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


def test_repl_invokes_main(monkeypatch):
    """The Typer callback should invoke the async main entry point."""
    captured: dict[str, object] = {}

    async def fake_main(*, file=None, output=None, verbose=False):
        captured["kwargs"] = {
            "file": file,
            "output": output,
            "verbose": verbose,
        }

    monkeypatch.setattr("swarmx.cli.amain", fake_main)

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
