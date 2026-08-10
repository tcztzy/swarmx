"""MCP client bridge from the durable Python worker to the managed RSI server."""

from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from importlib.metadata import version
from typing import Any, Callable

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .errors import RsiCancelledError, RsiError

TOOL_NAME = "swarmx_rsi_optimize"
MCP_VERSION = version("mcp")
MAX_CANDIDATE_BYTES = 4 * 1024 * 1024


def run_rsi_optimizer(
    request: dict[str, Any],
    artifacts: dict[str, str],
    capability_client: Any,
    cancel_check: Callable[[], bool],
    progress: Callable[[str, float], None],
) -> tuple[dict[str, Any], str]:
    """Run one optimizer call in an isolated MCP server process."""
    return asyncio.run(
        _run(
            request,
            artifacts,
            capability_client,
            cancel_check,
            progress,
        )
    )


async def _run(
    request: dict[str, Any],
    artifacts: dict[str, str],
    capability_client: Any,
    cancel_check: Callable[[], bool],
    progress: Callable[[str, float], None],
) -> tuple[dict[str, Any], str]:
    if cancel_check():
        raise RsiCancelledError("Skill optimization cancelled before RSI launch.")

    async def sample(
        _context: Any, params: types.CreateMessageRequestParams
    ) -> types.CreateMessageResult:
        messages: list[dict[str, str]] = []
        if params.systemPrompt:
            messages.append({"role": "system", "content": params.systemPrompt})
        for message in params.messages:
            if message.content.type != "text":
                raise RsiError("RSI sampling requested unsupported non-text content.")
            messages.append({"role": message.role, "content": message.content.text})
        outcome = capability_client.call(
            "skill_evolution",
            "model.generate",
            {
                "model": request.get("targetModelFingerprint"),
                "messages": messages,
                "temperature": params.temperature,
                "maxTokens": params.maxTokens,
            },
        )
        if outcome.get("status") != "succeeded":
            raise RsiError("Granted RSI model sampling failed.")
        value = outcome.get("value")
        if not isinstance(value, dict) or not isinstance(value.get("content"), str):
            raise RsiError("Granted RSI model sampling returned no text.")
        usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text=value["content"]),
            model=str(request.get("targetModelFingerprint") or "swarmx-gateway"),
            stopReason="endTurn",
            _meta={"swarmxUsage": usage},
        )

    async def report_progress(
        current: float, total: float | None, message: str | None
    ) -> None:
        fraction = current / total if total and total > 0 else current
        progress(message or "RSI optimization progress.", min(max(fraction, 0.0), 1.0))

    server = StdioServerParameters(
        command=sys.executable,
        args=["-I", "-B", "-u", "-m", "swarmx.rsi.server"],
        env={
            "PATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
        },
    )
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(
            reader,
            writer,
            sampling_callback=sample,
            read_timeout_seconds=timedelta(seconds=180),
        ) as session:
            initialized = await session.initialize()
            if (
                initialized.serverInfo.name != "swarmx-rsi"
                or initialized.serverInfo.version != MCP_VERSION
            ):
                raise RsiError("Unexpected RSI MCP server identity.")
            tools = await session.list_tools()
            if [tool.name for tool in tools.tools] != [TOOL_NAME]:
                raise RsiError("Unexpected RSI MCP tool surface.")
            call = asyncio.create_task(
                session.call_tool(
                    TOOL_NAME,
                    {"request": request, "artifacts": artifacts},
                    progress_callback=report_progress,
                )
            )
            while not call.done():
                if cancel_check():
                    call.cancel()
                    try:
                        await call
                    except asyncio.CancelledError:
                        pass
                    raise RsiCancelledError(
                        "Skill optimization cancelled in RSI MCP server."
                    )
                await asyncio.sleep(0.05)
            result = await call
            if result.isError or not isinstance(result.structuredContent, dict):
                raise RsiError("RSI MCP optimization failed.")
            candidate = result.structuredContent.get("candidateMarkdown")
            report = result.structuredContent.get("optimizerReport")
            if (
                not isinstance(candidate, str)
                or len(candidate.encode("utf-8")) > MAX_CANDIDATE_BYTES
                or not isinstance(report, dict)
            ):
                raise RsiError("RSI MCP returned an invalid optimization result.")
            return report, candidate
