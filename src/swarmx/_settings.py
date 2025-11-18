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
    OPENAI_API_KEY: str | None = None
    mcp_servers: dict[str, MCPServer] = Field(default_factory=dict, alias="mcpServers")
    """MCP configuration for the agent. Should be compatible with claude code."""

    agents_md: Path | list[Path] = Path.cwd() / "AGENTS.md"
    """A simple, open format for guiding agents."""

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

    def get_agents_md_content(self) -> str:
        agents_mds = (
            [self.agents_md] if isinstance(self.agents_md, Path) else self.agents_md
        )
        if isinstance(self.agents_md, Path):
            agents_mds = [self.agents_md]
        else:
            agents_mds = self.agents_md
        contents = []
        for agent_md in agents_mds:
            if agent_md.exists():
                try:
                    contents.append(
                        f'```markdown title="{agent_md}"\n{agent_md.read_text().strip()}\n```'
                    )
                except Exception:
                    # ignore any error
                    pass
        return "\n\n".join(contents)


settings = Settings()  # type: ignore
