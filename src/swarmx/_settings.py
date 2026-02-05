"""SwarmX."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

from .types import MCPServer


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line.removeprefix("export ").strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # Best-effort loading; settings will fall back to defaults.
        return


_load_env_file(Path.cwd() / ".env")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")


class Settings(BaseSettings, case_sensitive=True, env_file=".env", extra="ignore"):
    """Settings."""

    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-oss:20b"
    mcp_servers: dict[str, MCPServer] = Field(default_factory=dict, alias="mcpServers")
    """MCP configuration for the agent. Should be compatible with claude code."""

    agents_md: Path | list[Path] = Path.cwd() / "AGENTS.md"
    """A simple, open format for guiding agents."""

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
