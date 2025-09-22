"""Tests for the CLI module."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from swarmx.agent import Agent
from swarmx.cli import create_server_app, main
from swarmx.utils import now

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


async def test_main_with_file(temp_swarmx_file, temp_output_file):
    """Test main function with file input."""
    with (
        patch("typer.prompt") as mock_prompt,
        patch("swarmx.agent.Agent.run") as mock_run,
    ):
        mock_prompt.side_effect = ["Hello", KeyboardInterrupt()]

        async def mock_stream():
            chunk = ChatCompletionChunk.model_validate(
                {
                    "id": "test",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Hello response"},
                            "finish_reason": "stop",
                        }
                    ],
                    "created": now(),
                    "model": "gpt-4o",
                    "object": "chat.completion.chunk",
                }
            )
            yield chunk

        mock_run.return_value = mock_stream()

        await main(file=temp_swarmx_file, output=temp_output_file, verbose=False)

        assert temp_output_file.exists()
        output_data = json.loads(temp_output_file.read_text())
        assert len(output_data) == 1
        assert output_data[0]["role"] == "user"
        assert output_data[0]["content"] == "Hello"


async def test_main_without_file():
    """Test main function without file input."""
    with patch("typer.prompt") as mock_prompt:
        mock_prompt.side_effect = KeyboardInterrupt()
        await main(file=None, output=None, verbose=False)


async def test_main_with_exception():
    """Test main function when agent run raises an exception."""
    with (
        patch("typer.prompt") as mock_prompt,
        patch("typer.secho") as mock_secho,
        patch("swarmx.agent.Agent.run") as mock_run,
    ):
        mock_prompt.return_value = "Hello"
        mock_run.side_effect = Exception("Test error")
        await main(file=None, output=None, verbose=False)
        mock_secho.assert_called_with("Test error", err=True, fg="red")


async def test_main_with_verbose_reasoning():
    """Test main function with verbose mode and reasoning content."""
    with (
        patch("typer.prompt") as mock_prompt,
        patch("typer.secho") as mock_secho,
        patch("swarmx.agent.Agent.run") as mock_run,
    ):
        mock_prompt.side_effect = ["Hello", KeyboardInterrupt()]

        async def mock_stream():
            chunk = ChatCompletionChunk.model_validate(
                {
                    "id": "test",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": "Hello response",
                                "reasoning_content": "This is reasoning",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "created": now(),
                    "model": "gpt-4o",
                    "object": "chat.completion.chunk",
                }
            )
            yield chunk

        mock_run.return_value = mock_stream()
        await main(file=None, output=None, verbose=True)
        mock_secho.assert_called_with("This is reasoning", nl=False, fg="green")


async def test_main_with_refusal():
    """Test main function with refusal content."""
    with (
        patch("typer.prompt") as mock_prompt,
        patch("typer.secho") as mock_secho,
        patch("swarmx.agent.Agent.run") as mock_run,
    ):
        mock_prompt.side_effect = ["Hello", KeyboardInterrupt()]

        async def mock_stream():
            chunk = ChatCompletionChunk.model_validate(
                {
                    "id": "test",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"refusal": "I cannot help with that"},
                            "finish_reason": "content_filter",
                        }
                    ],
                    "created": now(),
                    "model": "gpt-4o",
                    "object": "chat.completion.chunk",
                }
            )
            yield chunk

        mock_run.return_value = mock_stream()
        await main(file=None, output=None, verbose=False)
        mock_secho.assert_called_with(
            "I cannot help with that", nl=False, err=True, fg="purple"
        )


def test_repl_command_with_subcommand():
    """Test repl command when a subcommand is invoked."""
    with patch("asyncio.run") as mock_run:
        ctx = MagicMock()
        ctx.invoked_subcommand = "serve"
        from swarmx.cli import repl

        repl(ctx, file=None, output=None, verbose=False)
        mock_run.assert_not_called()


def test_serve_command():
    """Test serve command."""
    with (
        patch("uvicorn.run") as mock_uvicorn,
        patch("swarmx.cli.create_server_app") as mock_create_app,
    ):
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        from swarmx.cli import serve

        serve(host="0.0.0.0", port=9000, file=None)
        mock_uvicorn.assert_called_once_with(mock_app, host="0.0.0.0", port=9000)


def test_serve_command_with_file(temp_swarmx_file):
    """Test serve command with file."""
    with (
        patch("uvicorn.run") as mock_uvicorn,
        patch("swarmx.cli.create_server_app") as mock_create_app,
    ):
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        from swarmx.cli import serve

        serve(host="127.0.0.1", port=8000, file=temp_swarmx_file)
        mock_uvicorn.assert_called_once_with(mock_app, host="127.0.0.1", port=8000)


def test_chat_completions_error_handling():
    """Test error handling in chat completions endpoint (lines 151-165)."""
    test_agent = Agent(name="test", instructions="Test agent")
    app = create_server_app(test_agent)
    client = TestClient(app)
    with patch("swarmx.agent.Agent.run", side_effect=Exception("Test error")):
        request_data = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        response = client.post("/chat/completions", json=request_data)
        assert response.status_code == 200
        content = response.content.decode()
        assert "Test error" in content
        assert "data: [DONE]" in content


def test_cli_main_execution():
    """Test CLI main execution."""
    result = subprocess.run(
        [sys.executable, "-c", "import swarmx.cli; swarmx.cli.app()"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode is not None
