"""Message graph utilities."""

import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import networkx as nx
from mcp.types import CallToolResult
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

MessageId = int | str


@dataclass
class Branch:
    """Branch metadata for the messages graph."""

    name: str
    description: str
    start: MessageId | None
    stop: MessageId | None


class Messages(Iterable[ChatCompletionMessageParam]):
    """Messages wrapper that stores a conversation DAG with branch metadata."""

    def __init__(self, messages: Iterable[ChatCompletionMessageParam]):
        """Initialize the messages graph from a linear history.

        Args:
            messages: Initial message sequence to seed the graph.

        """
        self._graph: nx.DiGraph = nx.DiGraph()
        self._origin: list[MessageId] = []
        self._main: list[MessageId] = []
        self._next_external_id = 0
        self._branches: dict[str, Branch] = {}

        previous_id: MessageId | None = None
        for message in messages:
            node_id = self._next_external_id
            self._next_external_id += 1
            self._graph.add_node(node_id, message=message)
            if previous_id is not None:
                self._graph.add_edge(previous_id, node_id)
            self._origin.append(node_id)
            self._main.append(node_id)
            previous_id = node_id

        self._sync_branches()

    def __iter__(self) -> Iterator[ChatCompletionMessageParam]:
        for node_id in self._main:
            yield self._graph.nodes[node_id]["message"]

    def __len__(self) -> int:
        return len(self._main)

    @property
    def graph(self) -> nx.DiGraph:
        """Return the underlying messages graph."""
        return self._graph

    @property
    def branches(self) -> dict[str, Branch]:
        """Return branch metadata."""
        return dict(self._branches)

    @property
    def origin_ids(self) -> list[MessageId]:
        """Return message IDs for the origin branch."""
        return list(self._origin)

    @property
    def main_ids(self) -> list[MessageId]:
        """Return message IDs for the main branch."""
        return list(self._main)

    def append_external_message(self, message: ChatCompletionMessageParam) -> MessageId:
        """Append an external message to both origin and main branches."""
        node_id = self._next_external_id
        self._next_external_id += 1
        return self._append_node(node_id, message)

    def append_llm_message(
        self,
        message: ChatCompletionMessageParam,
        completion: ChatCompletion,
    ) -> MessageId:
        """Append a model-generated message with its completion payload."""
        request_id = getattr(completion, "_request_id", None)
        base_id: MessageId = request_id or completion.id or str(uuid.uuid4())
        node_id = self._append_node(base_id, message, completion=completion)
        return node_id

    def append_tool_message(
        self,
        tool_call_id: str,
        message: ChatCompletionMessageParam,
        result: CallToolResult,
    ) -> MessageId:
        """Append a tool result message with its raw MCP result."""
        node_id = self._append_node(tool_call_id, message, result=result)
        return node_id

    def _append_node(
        self,
        node_id: MessageId,
        message: ChatCompletionMessageParam,
        *,
        completion: ChatCompletion | None = None,
        result: CallToolResult | None = None,
    ) -> MessageId:
        node_id = self._ensure_unique_id(node_id)
        payload: dict[str, Any] = {"message": message}
        if completion is not None:
            payload["completion"] = completion
        if result is not None:
            payload["result"] = result
        self._graph.add_node(node_id, **payload)
        self._append_to_branch(self._origin, node_id)
        self._append_to_branch(self._main, node_id)
        self._sync_branches()
        return node_id

    def _append_to_branch(self, branch: list[MessageId], node_id: MessageId) -> None:
        if branch:
            self._graph.add_edge(branch[-1], node_id)
        branch.append(node_id)

    def _ensure_unique_id(self, node_id: MessageId) -> MessageId:
        if node_id not in self._graph:
            return node_id
        if isinstance(node_id, int):
            return f"{node_id}:{uuid.uuid4().hex}"
        suffix = 1
        candidate = f"{node_id}:{suffix}"
        while candidate in self._graph:
            suffix += 1
            candidate = f"{node_id}:{suffix}"
        return candidate

    def _sync_branches(self) -> None:
        origin_start = self._origin[0] if self._origin else None
        origin_stop = self._origin[-1] if self._origin else None
        main_start = self._main[0] if self._main else None
        main_stop = self._main[-1] if self._main else None
        self._branches["origin"] = Branch(
            name="origin",
            description="full, uncompressed history",
            start=origin_start,
            stop=origin_stop,
        )
        self._branches["main"] = Branch(
            name="main",
            description="active working branch for LLM calls",
            start=main_start,
            stop=main_stop,
        )
