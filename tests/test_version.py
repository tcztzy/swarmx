"""Tests for the version module."""

import importlib.metadata
from unittest.mock import patch

import swarmx.version


def test_version_import():
    """Test that version can be imported."""
    assert hasattr(swarmx.version, "__version__")
    assert isinstance(swarmx.version.__version__, str)


def test_version_package_not_found():
    """Test version fallback when package is not found."""

    with patch("importlib.metadata.version") as mock_version:
        mock_version.side_effect = importlib.metadata.PackageNotFoundError()

        importlib.reload(swarmx.version)

        assert swarmx.version.__version__ == "0.0.0"


def test_version_normal_case():
    """Test version retrieval in normal case."""
    with patch("importlib.metadata.version") as mock_version:
        mock_version.return_value = "1.2.3"

        importlib.reload(swarmx.version)

        assert swarmx.version.__version__ == "1.2.3"
