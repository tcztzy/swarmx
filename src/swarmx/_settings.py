"""SwarmX."""

import json
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, InitSettingsSource

from .types import MCPServer


def json_merge(left: dict[str, Any], *rights: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two JSON objects. Non-dict values are taken from the right object."""
    result: dict[str, Any] = dict(left)
    for right in rights:
        for key, right_val in right.items():
            left_val = result.get(key)
            # Currently, we only care about mcpServers.
            if (
                isinstance(left_val, dict)
                and isinstance(right_val, dict)
                and key == "mcpServers"
            ):
                result[key] = json_merge(left_val, right_val)
            else:
                result[key] = right_val
    return result


class ClaudeCodeSettingsSource(InitSettingsSource):
    """For misanthropic claude code config inherit."""

    def __init__(self, settings_cls: type[BaseSettings]):  # noqa: D107
        user_config, local_config, project_config = {}, {}, {}
        if (dot_claude_json := Path.home() / ".claude.json").exists():
            try:
                user_config = json.loads(dot_claude_json.read_text())
                local_config = user_config["projects"].get(str(Path.cwd()), {})
            except Exception:
                pass
        if (dot_mcp_json := Path.cwd() / ".mcp.json").exists():
            try:
                project_config = json.loads(dot_mcp_json.read_text())
            except Exception:
                pass
        super().__init__(
            settings_cls, json_merge(user_config, local_config, project_config)
        )


class Settings(BaseSettings, case_sensitive=True, env_file=".env", extra="ignore"):
    """Settings."""

    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_API_KEY: str
    mcp_servers: dict[str, MCPServer] = Field(default_factory=dict, alias="mcpServers")
    """MCP configuration for the agent. Should be compatible with claude code."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Set settings source from Claude Code."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            ClaudeCodeSettingsSource(settings_cls),
        )


settings = Settings()  # type: ignore
