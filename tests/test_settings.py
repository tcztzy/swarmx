"""Tests for swarmx._settings module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swarmx import _settings as settings_module
from swarmx._settings import ClaudeCodeSettingsSource, Settings, json_merge


def test_json_merge_with_nested_mcp_servers():
    left = {"mcpServers": {"a": {"url": "left"}}, "other": 1}
    right = {"mcpServers": {"b": {"url": "right"}}, "other": 2}
    merged = json_merge(left, right)
    assert merged["mcpServers"]["a"] == {"url": "left"}
    assert merged["mcpServers"]["b"] == {"url": "right"}
    assert merged["other"] == 2


def test_claude_code_settings_source_reads_files(tmp_path: Path, monkeypatch):
    claude = tmp_path / ".claude.json"
    mcp = tmp_path / ".mcp.json"
    claude.write_text(
        json.dumps(
            {"projects": {str(tmp_path): {"mcpServers": {"from": {"url": "claude"}}}}}
        )
    )
    mcp.write_text(json.dumps({"mcpServers": {"project": {"url": "project"}}}))

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    source = ClaudeCodeSettingsSource(Settings)
    assert source.init_kwargs["mcpServers"]["from"]["url"] == "claude"
    assert source.init_kwargs["mcpServers"]["project"]["url"] == "project"


def test_get_agents_md_content_handles_list_and_errors(tmp_path: Path, monkeypatch):
    good = tmp_path / "good.md"
    bad = tmp_path / "bad.md"
    good.write_text("Content")
    bad.write_text("Broken")

    settings = Settings(agents_md=[good, bad])

    original_read_text = Path.read_text

    def maybe_fail(self):  # noqa: ANN001
        if self == bad:
            raise IOError("boom")
        return original_read_text(self)

    monkeypatch.setattr(Path, "read_text", maybe_fail, raising=False)
    content = settings.get_agents_md_content()
    assert "good.md" in content


def test_get_agents_md_content_with_single_path(tmp_path: Path):
    md = tmp_path / "agents.md"
    md.write_text("Example")
    settings = Settings(agents_md=md)
    content = settings.get_agents_md_content()
    assert "agents.md" in content


@pytest.fixture(autouse=True)
def restore_global_settings():
    original = settings_module.settings.agents_md
    yield
    settings_module.settings.agents_md = original


def test_claude_code_settings_source_handles_invalid_json(tmp_path: Path, monkeypatch):
    """Gracefully ignore malformed project configuration files."""
    claude = tmp_path / ".claude.json"
    mcp = tmp_path / ".mcp.json"
    claude.write_text("{invalid")
    mcp.write_text("{invalid")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    source = ClaudeCodeSettingsSource(Settings)
    assert source.init_kwargs == {}
