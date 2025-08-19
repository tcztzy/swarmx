"""Tests for the __main__.py module."""

import subprocess
import sys
from unittest.mock import patch


def test_main_module_execution():
    """Test that the __main__.py module can be executed."""
    # Test that the module can be imported and executed
    with patch("swarmx.cli.app") as mock_app:
        import swarmx.__main__  # noqa: F401

        mock_app.assert_called_once()


def test_main_module_via_python_m():
    """Test that the module can be executed via python -m."""
    # This tests the actual execution path
    result = subprocess.run(
        [sys.executable, "-m", "swarmx", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "SwarmX Command Line Interface" in result.stdout
