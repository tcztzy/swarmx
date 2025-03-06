import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from swarmx import Agent, Swarm, main

pytestmark = pytest.mark.anyio


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_swarm_file():
    """Create a sample swarm file for testing."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
        config = {
            "agent": {
                "name": "TestAgent",
                "instructions": "You are a test agent.",
                "model": "gpt-4o",
            },
            "context_variables": {"user_id": "test123"},
        }
        json.dump(config, tmp)
        tmp_path = Path(tmp.name)

    yield tmp_path
    # Clean up the temporary file
    tmp_path.unlink(missing_ok=True)


async def test_main_with_file(sample_swarm_file):
    # Mock the Swarm.model_validate and agent.run methods
    with (
        patch("swarmx.Swarm.model_validate") as mock_swarm_validate,
        patch("swarmx.Agent.model_validate") as mock_agent_validate,
        patch("swarmx.typer.prompt") as mock_prompt,
    ):
        # Setup mocks
        mock_swarm = MagicMock(spec=Swarm)
        mock_agent = MagicMock(spec=Agent)

        # Create a mock async iterator that can be used with async for
        class MockAsyncIterator:
            def __init__(self, items):
                self.items = items

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.items:
                    raise StopAsyncIteration
                return self.items.pop(0)

        # The main function uses client.run() with stream=True, so it needs to return an async iterator
        mock_swarm.run.return_value = MockAsyncIterator(
            [{"role": "assistant", "content": "Hello from the agent"}]
        )

        mock_swarm_validate.return_value = mock_swarm
        mock_agent_validate.return_value = mock_agent

        # Mock user input to exit after one interaction
        mock_prompt.side_effect = ["Hello", KeyboardInterrupt()]

        # Run the main function with the sample file
        try:
            await main(
                model="gpt-4o", file=sample_swarm_file, output=None, verbose=False
            )
        except KeyboardInterrupt:
            pass  # Expected to be raised to exit the loop

        # Verify swarm.run was called (not agent.run)
        mock_swarm.run.assert_called_once()

        # Check that the context variables were passed
        call_args = mock_swarm.run.call_args.kwargs
        assert "context_variables" in call_args
        assert call_args["context_variables"]["user_id"] == "test123"


async def test_main_with_output_file():
    # Create a temporary output file
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False
    ) as output_file:
        output_path = Path(output_file.name)

    try:
        # Mock the necessary objects and methods
        with (
            patch("swarmx.Swarm.model_validate") as mock_swarm_validate,
            patch("swarmx.Agent.model_validate") as mock_agent_validate,
            patch("swarmx.typer.prompt") as mock_prompt,
        ):
            # Setup mocks
            mock_swarm = MagicMock(spec=Swarm)
            mock_agent = MagicMock(spec=Agent)

            # Create a mock async iterator that can be used with async for
            class MockAsyncIterator:
                def __init__(self, items):
                    self.items = items

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if not self.items:
                        raise StopAsyncIteration
                    return self.items.pop(0)

            # The main function uses client.run() with stream=True, so it needs to return an async iterator
            mock_swarm.run.return_value = MockAsyncIterator(
                [{"role": "assistant", "content": "Hello from the agent"}]
            )

            mock_swarm_validate.return_value = mock_swarm
            mock_agent_validate.return_value = mock_agent

            # Mock user input to exit after one interaction
            mock_prompt.side_effect = ["Hello", KeyboardInterrupt()]

            # Run the main function with output file
            try:
                await main(model="gpt-4o", file=None, output=output_path, verbose=True)
            except KeyboardInterrupt:
                pass  # Expected to be raised to exit the loop

            # Verify that the output file was written to
            assert output_path.exists()

            # Check the contents of the output file
            with open(output_path, "r") as f:
                saved_data = json.load(f)

            # Verify the saved data has the expected structure
            assert isinstance(saved_data, list)
            assert len(saved_data) == 2  # Should have both user and assistant messages
            assert saved_data[0]["role"] == "user"
            assert saved_data[0]["content"] == "Hello"
            assert saved_data[1]["role"] == "assistant"
            assert saved_data[1]["content"] == "Hello from the agent"

            # Verify swarm.run was called
            mock_swarm.run.assert_called_once()

            # Check that the context variables were passed
            call_args = mock_swarm.run.call_args.kwargs
            assert "context_variables" in call_args
            assert isinstance(call_args["context_variables"], dict)

    finally:
        # Clean up the temporary file
        if output_path.exists():
            output_path.unlink()
