"""Shared node definitions for agents and tools."""

from abc import ABCMeta, abstractmethod
from datetime import timedelta
from typing import Annotated, Any

import jsonschema
from mcp import ClientSession
from mcp import Tool as _Tool
from mcp.shared.session import ProgressFnT
from mcp.types import CallToolResult
from pydantic import Field, PrivateAttr


class Node[I, O](_Tool, metaclass=ABCMeta):
    """Base node element in the execution graph."""

    name: Annotated[str, Field(frozen=True)]

    def __hash__(self) -> int:
        """Hashable."""
        return hash(self.name)

    @abstractmethod
    async def __call__(
        self,
        arguments: I | None = None,
        *args,
        **kwargs,
    ) -> O:
        """Callable."""
        ...


class Tool(Node[dict[str, Any] | None, CallToolResult]):
    """Graph node representing an MCP Tool call."""

    name: Annotated[
        str,
        Field(
            pattern=(
                "^mcp__"
                r"([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*)__"
                r"([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*)"
            ),
            frozen=True,
        ),
    ]

    _session: ClientSession | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize MCP session reference."""
        if self.model_extra is None:
            self._session = None
        else:
            self._session = self.model_extra.pop("session", None)

    async def __call__(
        self,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
        progress_callback: ProgressFnT | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Execute the targeted MCP tool call."""
        if self._session is None:
            raise RuntimeError(
                f"Client session not found, this tool `{self.name}` could not be called."
            )
        jsonschema.validate(arguments, self.inputSchema)
        return await self._session.call_tool(
            self.name.split("__", maxsplit=2)[2],
            arguments,
            read_timeout_seconds,
            progress_callback,
            meta=meta,
        )
