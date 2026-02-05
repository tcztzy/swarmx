"""Tests for swarmx._settings module."""

from pathlib import Path

import pytest

from swarmx import _settings as settings_module
from swarmx._settings import Settings


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
