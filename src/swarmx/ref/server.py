"""Private read-only Reference MCP server from the standard SwarmX package."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from typing import Any

from mcp.server.mcpserver import MCPServer

from .service import (
    LibzimReferenceBackend,
    ReferenceBackend,
    ReferenceService,
    ZoteroReferenceBackend,
)

TOOL_NAME = "swarmx_reference"
MODULE_VERSION = version("swarmx")
MCP_VERSION = version("mcp")

mcp = MCPServer(
    name="swarmx-ref",
    version=MCP_VERSION,
    instructions="Private read-only multi-source reference module for SwarmX.",
    log_level="ERROR",
)
_service: ReferenceService | None = None


@mcp.tool(name=TOOL_NAME, structured_output=True)
def reference(request: dict[str, Any]) -> dict[str, Any]:
    """Inspect, search, or read explicitly configured reference sources."""

    if _service is None:
        raise ValueError("Reference source is unavailable.")
    try:
        return _service.handle(request)
    except (TypeError, ValueError):
        raise ValueError("Reference request is invalid or unavailable.") from None


def version_json() -> str:
    return json.dumps(
        {
            "name": "swarmx-ref",
            "version": MODULE_VERSION,
            "protocol": "mcp",
            "tool": TOOL_NAME,
            "access": "read_only",
        },
        separators=(",", ":"),
    )


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments == ["--version-json"]:
        print(version_json())
        return 0
    parser = argparse.ArgumentParser(prog="swarmx-ref")
    parser.add_argument("--zim")
    parser.add_argument("--zotero", action="store_true")
    parser.add_argument("--stdio", action="store_true", required=True)
    parsed = parser.parse_args(arguments)
    backends: dict[str, ReferenceBackend] = {}
    if parsed.zim:
        backends["zim"] = LibzimReferenceBackend(parsed.zim)
    if parsed.zotero:
        backends["zotero"] = ZoteroReferenceBackend()
    if not backends:
        parser.error("at least one of --zim or --zotero is required")
    global _service
    _service = ReferenceService(backends)
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
