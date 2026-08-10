"""Managed SwarmX RSI server exposed only through MCP over stdio."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import sys
from collections.abc import Mapping
from importlib.metadata import version
from typing import Any

from mcp import types
from mcp.server.fastmcp import Context, FastMCP

from .contract import validate_optimization_request

MODULE_VERSION = version("swarmx")
TOOL_NAME = "swarmx_rsi_optimize"
MAX_ARTIFACT_CHARS = 4 * 1024 * 1024
MAX_TOTAL_ARTIFACT_CHARS = 12 * 1024 * 1024
MAX_CANDIDATE_BYTES = 4 * 1024 * 1024
MAX_RESULT_BYTES = MAX_CANDIDATE_BYTES + 64 * 1024

mcp = FastMCP(
    name="swarmx-rsi",
    instructions="Private managed RSI optimizer module for SwarmX.",
    log_level="ERROR",
)


@mcp.tool(name=TOOL_NAME, structured_output=True)
async def optimize_skill(
    request: dict[str, Any],
    artifacts: dict[str, str],
    ctx: Context,
) -> dict[str, Any]:
    """Propose one immutable Skill candidate from a validated optimization request."""
    try:
        normalized_request = _validate_request(request)
        normalized_artifacts = _validate_artifacts(normalized_request, artifacts)
        optimizer = _mapping(normalized_request, "optimizer")
        optimizer_id = _string(optimizer, "optimizerId")
        if optimizer_id != "dspy.gepa.v1":
            raise ValueError("unsupported optimizer")
        result = await _run_gepa(normalized_request, normalized_artifacts, ctx)
        _validate_result(result)
        return result
    except (TypeError, ValueError, KeyError):
        raise ValueError(
            "RSI optimization request is invalid or unsupported."
        ) from None


def _validate_request(request: object) -> dict[str, Any]:
    if not isinstance(request, dict) or not all(
        isinstance(key, str) for key in request
    ):
        raise ValueError("request must be an object")
    return validate_optimization_request(request)


def _validate_artifacts(
    request: Mapping[str, Any], artifacts: object
) -> dict[str, str]:
    if not isinstance(artifacts, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in artifacts.items()
    ):
        raise ValueError("artifacts must be text by ref")
    expected = {
        _string(request, "baselineContentRef"),
        _string(_mapping(request, "trainDataset"), "contentRef"),
        _string(_mapping(request, "devDataset"), "contentRef"),
    }
    if set(artifacts) != expected:
        raise ValueError("artifact grants do not match request")
    sizes = [len(value) for value in artifacts.values()]
    if (
        any(size > MAX_ARTIFACT_CHARS for size in sizes)
        or sum(sizes) > MAX_TOTAL_ARTIFACT_CHARS
    ):
        raise ValueError("artifact input exceeds limit")
    for ref, content in artifacts.items():
        digest = f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
        if ref != digest:
            raise ValueError("artifact digest does not match content")
    return dict(artifacts)


async def _run_gepa(
    request: dict[str, Any], artifacts: Mapping[str, str], ctx: Context
) -> dict[str, Any]:
    from . import optimize

    optimizer = _mapping(request, "optimizer")
    if _string(optimizer, "configDigest") != optimize.canonical_config_digest(request):
        raise ValueError("optimizer configuration digest mismatch")
    loop = asyncio.get_running_loop()
    capability_client = McpSamplingCapabilityClient(ctx, artifacts, loop)

    def progress(message: str, fraction: float) -> None:
        future = asyncio.run_coroutine_threadsafe(
            ctx.report_progress(progress=fraction, total=1.0, message=message), loop
        )
        future.result(timeout=30)

    def run_optimizer() -> tuple[dict[str, Any], str]:
        previous_logging_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            # The MCP transport captures the original stdout buffer before tool calls.
            # DSPy/tqdm output must be discarded so it cannot corrupt JSON-RPC or leak prompts.
            with (
                contextlib.redirect_stdout(_ProtocolNoiseSink()),
                contextlib.redirect_stderr(_ProtocolNoiseSink()),
            ):
                return optimize.run_gepa(
                    request,
                    capability_client,
                    lambda: False,
                    progress,
                )
        finally:
            logging.disable(previous_logging_disable)

    report, candidate = await asyncio.to_thread(run_optimizer)
    return {"candidateMarkdown": candidate, "optimizerReport": report}


class _ProtocolNoiseSink:
    """Drops third-party console output because MCP owns process stdout."""

    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False


class McpSamplingCapabilityClient:
    """Adapts DSPy's synchronous LM calls to MCP client sampling requests."""

    def __init__(
        self,
        ctx: Context,
        artifacts: Mapping[str, str],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._ctx = ctx
        self._artifacts = artifacts
        self._loop = loop

    def call(
        self,
        namespace: str,
        name: str,
        payload: dict[str, Any],
        timeout_ms: int = 60_000,
    ) -> dict[str, Any]:
        if namespace != "skill_evolution":
            return _failed("unsupported capability namespace")
        if name == "read_artifact":
            ref = payload.get("ref")
            content = self._artifacts.get(ref) if isinstance(ref, str) else None
            return (
                {"status": "succeeded", "value": {"content": content}}
                if content is not None
                else _failed("artifact is not granted")
            )
        if name != "model.generate":
            return _failed("unsupported capability")
        future = asyncio.run_coroutine_threadsafe(self._sample(payload), self._loop)
        try:
            return future.result(timeout=max(1, timeout_ms) / 1_000)
        except TimeoutError:
            future.cancel()
            return _failed("model generation timed out")

    async def _sample(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or len(messages) > 256:
            return _failed("model messages are invalid")
        sampling_messages: list[types.SamplingMessage] = []
        system: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                return _failed("model message is invalid")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(content, str) or len(content) > 256_000:
                return _failed("model message content is invalid")
            if role == "system":
                system.append(content)
            elif role in {"user", "assistant"}:
                sampling_messages.append(
                    types.SamplingMessage(
                        role=role,
                        content=types.TextContent(type="text", text=content),
                    )
                )
            else:
                return _failed("model message role is invalid")
        max_tokens = payload.get("maxTokens")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            max_tokens = 2_048
        max_tokens = min(max(max_tokens, 1), 65_536)
        temperature = payload.get("temperature")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            temperature = None
        result = await self._ctx.session.create_message(  # pyright: ignore[reportDeprecated]
            messages=sampling_messages,
            max_tokens=max_tokens,
            system_prompt="\n\n".join(system) or None,
            temperature=float(temperature) if temperature is not None else None,
        )
        if result.content.type != "text":
            return _failed("model generation returned non-text content")
        usage = {}
        metadata = result.meta or {}
        if isinstance(metadata.get("swarmxUsage"), dict):
            usage = metadata["swarmxUsage"]
        return {
            "status": "succeeded",
            "value": {"content": result.content.text, "usage": usage},
        }


def _validate_result(result: Mapping[str, Any]) -> None:
    if set(result) != {"candidateMarkdown", "optimizerReport"}:
        raise ValueError("optimizer result fields are invalid")
    candidate = result.get("candidateMarkdown")
    report = result.get("optimizerReport")
    if (
        not isinstance(candidate, str)
        or len(candidate.encode("utf-8")) > MAX_CANDIDATE_BYTES
        or not isinstance(report, dict)
        or not isinstance(report.get("optimizerId"), str)
        or not isinstance(report.get("actualOptimizer"), str)
        or len(json.dumps(result, separators=(",", ":")).encode("utf-8"))
        > MAX_RESULT_BYTES
    ):
        raise ValueError("optimizer result is invalid")


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    field = value.get(key)
    if not isinstance(field, dict):
        raise ValueError(f"{key} must be an object")
    return field


def _string(value: Mapping[str, Any], key: str) -> str:
    field = value.get(key)
    if not isinstance(field, str) or not field:
        raise ValueError(f"{key} must be a non-empty string")
    return field


def _failed(message: str) -> dict[str, Any]:
    return {"status": "failed", "error": {"message": message}}


def version_json() -> str:
    return json.dumps(
        {
            "name": "swarmx-rsi",
            "version": MODULE_VERSION,
            "protocol": "mcp",
            "tool": TOOL_NAME,
        },
        separators=(",", ":"),
    )


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments == ["--version-json"]:
        print(version_json())
        return 0
    if arguments:
        raise SystemExit("swarmx-rsi accepts only --version-json or stdio mode")
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
