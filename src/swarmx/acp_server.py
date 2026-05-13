"""ACP agent server for SwarmX — connects Rust UI to Python swarm engine."""

import asyncio
import logging
from typing import Any
from uuid import uuid4

from acp import (
    Agent as ACPAgent,
)
from acp import (
    Client,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    update_agent_message,
)
from acp.schema import (
    AgentCapabilities,
    ClientCapabilities,
    Implementation,
    TextContentBlock,
)

from .agent import Agent
from .swarm import Swarm

logger = logging.getLogger(__name__)


class SwarmXAgent(ACPAgent):  # type: ignore[abstract]
    """ACP Agent that runs SwarmX swarm executions.

    Each ACP session maps to one swarm run. The swarm config is passed
    via _meta.swarm_config on the first prompt's text block.
    """

    def __init__(self) -> None:
        """Initialize the ACP agent with empty session store."""
        self._conn: Client | None = None
        self._sessions: dict[str, dict[str, Any]] = {}

    def on_connect(self, conn: Client) -> None:  # type: ignore[override]
        """Store the client connection for sending updates back."""
        self._conn = conn

    async def initialize(  # type: ignore[override]
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        """Handle ACP initialization handshake."""
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(
                load_session=False,
                prompt_capabilities={
                    "image": False,
                    "audio": False,
                    "embedded_context": True,
                },
            ),
            agent_info={
                "name": "swarmx",
                "title": "SwarmX Agent Engine",
                "version": "2.0.0",
            },
            auth_methods=[],
        )

    async def new_session(  # type: ignore[override]
        self,
        cwd: str,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        """Create a new session for swarm execution."""
        session_id = uuid4().hex[:12]
        self._sessions[session_id] = {
            "cwd": cwd,
            "mcp_servers": mcp_servers or [],
            "swarm_config": kwargs.get("swarm_config"),
            "history": [],
        }
        logger.info("Session %s created (cwd=%s)", session_id, cwd)
        return NewSessionResponse(session_id=session_id)

    async def prompt(  # type: ignore[override]
        self,
        prompt: list[TextContentBlock],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        """Handle a user prompt by executing the swarm and streaming results."""
        conn = self._conn
        if conn is None:
            return PromptResponse(stop_reason="cancelled")

        session = self._sessions.get(session_id)
        if session is None:
            return PromptResponse(stop_reason="cancelled")

        # Extract user text and optional swarm config from content blocks
        user_text = ""
        swarm_config = session.get("swarm_config")
        for block in prompt:
            text = block.text if hasattr(block, "text") else ""
            if text:
                user_text += text
            meta = getattr(block, "field_meta", None) or {}
            if meta.get("swarm_config"):
                swarm_config = meta["swarm_config"]

        if not user_text.strip():
            return PromptResponse(stop_reason="end_turn")

        # Build swarm from config or use default single-agent mode
        try:
            if swarm_config:
                swarm = self._build_swarm(swarm_config)
            else:
                swarm = self._default_swarm()

            await self._run_swarm(conn, session_id, swarm, user_text)

        except Exception as exc:
            logger.exception("Swarm execution failed")
            await conn.session_update(
                session_id,
                update_agent_message(
                    TextContentBlock(
                        type="text",
                        text=f"[error] {type(exc).__name__}: {exc}",
                        field_meta={"agent": "system", "status": "error"},
                    )
                ),
            )

        return PromptResponse(stop_reason="end_turn")

    # ── swarm execution ─────────────────────────────────────────────────────

    @staticmethod
    def _default_swarm() -> Swarm:
        """Create a default single-agent swarm."""
        agent = Agent(
            name="agent",
            instructions="You are a helpful AI assistant. Be concise and friendly.",
        )
        return Swarm(
            name="default",
            root="agent",
            nodes={"agent": agent},
            edges=[],
            parameters={},
        )

    @staticmethod
    def _build_swarm(config: dict[str, Any]) -> Swarm:
        """Build a Swarm from a JSON-compatible config dict."""
        from pydantic import TypeAdapter

        return TypeAdapter(Swarm).validate_python(config)

    async def _run_swarm(
        self,
        conn: Client,
        session_id: str,
        swarm: Swarm,
        user_text: str,
    ) -> None:
        """Execute swarm, streaming agent status updates via ACP."""
        args: dict[str, Any] = {"messages": [{"role": "user", "content": user_text}]}

        # Stream: execution started
        await conn.session_update(
            session_id,
            update_agent_message(
                TextContentBlock(
                    type="text",
                    text="",
                    field_meta={
                        "swarm_event": "execution_started",
                        "swarm_name": swarm.name,
                        "root": swarm.root,
                        "agents": list(swarm.nodes.keys()),
                    },
                )
            ),
        )

        try:
            result = await swarm(args, context=None)  # type: ignore[arg-type]
        except Exception as exc:
            await conn.session_update(
                session_id,
                update_agent_message(
                    TextContentBlock(
                        type="text",
                        text=f"Execution error: {exc}",
                        field_meta={
                            "swarm_event": "execution_error",
                            "error": str(exc),
                        },
                    )
                ),
            )
            return

        # Stream per-agent results
        for msg in result:
            role = getattr(msg, "role", "assistant")
            content = getattr(msg, "content", "")
            if not content:
                continue

            await conn.session_update(
                session_id,
                update_agent_message(
                    TextContentBlock(
                        type="text",
                        text=str(content),
                        field_meta={
                            "role": str(role),
                            "swarm_event": "agent_message",
                        },
                    )
                ),
            )

        # Stream: execution complete
        await conn.session_update(
            session_id,
            update_agent_message(
                TextContentBlock(
                    type="text",
                    text="",
                    field_meta={
                        "swarm_event": "execution_complete",
                        "message_count": len(result),
                    },
                )
            ),
        )


def main() -> None:
    """Entry point for `python -m swarmx.acp_server`."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_agent(SwarmXAgent()))  # type: ignore[abstract]


if __name__ == "__main__":
    main()
