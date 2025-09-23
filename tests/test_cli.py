"""Tests for the CLI module."""

import json
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swarmx.cli import app

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
