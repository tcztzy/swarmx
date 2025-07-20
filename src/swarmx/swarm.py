"""SwarmX Swarm module."""

import asyncio
import base64
import copy
import json
import mimetypes
import os
from collections import defaultdict
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Hashable,
    Iterable,
    Literal,
    Required,
    Self,
    TypedDict,
    cast,
    overload,
)

import mcp.types
import networkx as nx
from mcp import StdioServerParameters
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)
from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartParam,
    File,
)
from openai.types.chat.chat_completion_content_part_refusal_param import (
    ChatCompletionContentPartRefusalParam,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
)
from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    ModelWrapValidatorHandler,
    TypeAdapter,
    ValidationError,
    model_serializer,
    model_validator,
)
from pygments.lexers import get_lexer_for_filename, get_lexer_for_mimetype, guess_lexer
from pygments.util import ClassNotFound

from .agent import Agent, Node, Result, condition_parser, merge_chunks
from .mcp_client import TOOL_REGISTRY
from .utils import get_random_string, now

mimetypes.add_type("text/markdown", ".md")


class AgentNodeData(TypedDict, total=False):
    """Data for an agent node in a swarm.

    This is used for serialization and deserialization.

    """

    type: Required[Literal["agent"]]  # type: ignore
    agent: Required["Agent"]  # type: ignore
    executed: bool


class SwarmNodeData(TypedDict, total=False):  # noqa
    type: Required[Literal["swarm"]]  # type: ignore
    swarm: Required["Swarm"]  # type: ignore
    executed: bool


NodeData = Annotated[AgentNodeData | SwarmNodeData, Discriminator("type")]


def _image_content_to_url(
    content: mcp.types.ImageContent,
) -> ChatCompletionContentPartImageParam:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{content.mimeType};base64,{content.data}",
        },
    }


def _resource_to_file(  # type: ignore[return]
    resource: mcp.types.EmbeddedResource,
) -> ChatCompletionContentPartTextParam | File:
    def get_filename(c: mcp.types.ResourceContents) -> str:
        if c.uri.path is not None:
            filename = os.path.basename(c.uri.path)
        elif c.uri.host is not None:
            filename = c.uri.host
        elif c.mimeType is not None:
            ext = mimetypes.guess_extension(c.mimeType)
            if ext is None:
                raise ValueError(
                    f"Cannot determine filename for resource. mimeType={c.mimeType}"
                )
            filename = f"file{ext}"
        else:
            raise ValueError("Cannot determine filename for resource.")
        return filename

    filename = get_filename(resource.resource)
    match resource.resource:
        case mcp.types.TextResourceContents() as c:
            if c.mimeType is None:
                try:
                    lexer = get_lexer_for_filename(filename)
                except ClassNotFound:
                    lexer = None
            else:
                try:
                    lexer = get_lexer_for_mimetype(c.mimeType)
                except ClassNotFound:
                    lexer = None
            lang = lexer.aliases[0] if lexer else "text"
            return {
                "type": "text",
                "text": f'\n```{lang} title="{c.uri}"\n{c.text}\n```\n',
            }
        case mcp.types.BlobResourceContents() as c:
            return {
                "type": "file",
                "file": {
                    "file_data": c.blob,
                    "filename": filename,
                },
            }


def _mcp_call_tool_result_to_content(
    result: mcp.types.CallToolResult,
) -> list[ChatCompletionContentPartParam]:
    content: list[ChatCompletionContentPartParam] = []
    for chunk in result.content:
        match chunk.type:
            case "text":
                content.append(
                    cast(
                        ChatCompletionContentPartTextParam,
                        chunk.model_dump(exclude={"annotations"}, exclude_none=True),
                    )
                )
            case "image":
                content.append(_image_content_to_url(chunk))
            case "resource":
                content.append(_resource_to_file(chunk))
    return content


def content_part_to_str(  # type: ignore[return]
    part: ChatCompletionContentPartParam | ChatCompletionContentPartRefusalParam,
) -> str:
    """Convert a content part to string."""
    match part["type"]:
        case "text":
            return part["text"]
        case "refusal":
            return part["refusal"]
        case "image_url":
            return f"![]({part['image_url']['url']})"
        case "input_audio":
            match part["input_audio"]["format"]:
                case "mp3":
                    mimetype = "audio/mpeg"
                case "wav":
                    mimetype = "audio/wav"
            src = "data:" + mimetype + ";base64," + part["input_audio"]["data"]
            return f'<audio controls><source src="{src}" type="{mimetype}"></audio>'
        case "file":
            file_data = part["file"].get("file_data")
            filename = part["file"].get("filename", "file")
            if file_data is None or filename is None:
                raise ValueError("File content/name is not available")
            file_content = base64.decodebytes(file_data.encode())

            try:
                lexer = get_lexer_for_filename(filename)
            except ClassNotFound:
                try:
                    file_content_decoded = file_content.decode()
                    lexer = guess_lexer(file_content_decoded)
                except (UnicodeDecodeError, ClassNotFound):
                    lexer = None
                    mime, _ = mimetypes.guess_type(filename)
                    if mime is None or not mime.startswith("text/"):
                        return f"[{filename}](data:application/octet-stream;base64,{file_data})"

            lang = lexer.aliases[0] if lexer else "text"
            return f'\n```{lang} title="{filename}"\n{file_content.decode()}\n```\n'


def messages_to_chunks(messages: list[ChatCompletionMessageParam]):
    """Convert a list of messages to a list of chunks.

    This function is useful for streaming the messages to the client.

    """
    id_to_name: dict[str, str] = {}
    for message in messages:
        message_id = message.get("tool_call_id", get_random_string(10))
        model = message.get("name", "swarmx")
        match message["role"]:
            case "function":
                raise NotImplementedError("SwarmX would not support function message.")
            case _ as role:
                # We only want support follow style of assistant message.
                # text* (refusal* | tool_calls)
                content = message.get("content")
                refusal = message.get("refusal")
                tool_calls = message.get("tool_calls")
                if content is None and refusal is None and tool_calls is None:
                    raise ValueError(
                        "Assistant message must have content, refusal, or tool_calls"
                    )
                if (
                    refusal is not None
                    or (
                        not isinstance(content, str)
                        and content is not None
                        and any(part["type"] == "refusal" for part in content)
                    )
                ) and tool_calls is not None:
                    raise ValueError(
                        "Assistant message cannot have both refusal and tool_calls"
                    )
                if content is None:
                    content = cast(Iterable[ChatCompletionContentPartTextParam], [])
                elif isinstance(content, str):
                    content = cast(
                        Iterable[ChatCompletionContentPartTextParam],
                        [{"type": "text", "text": content}],
                    )
                if refusal is None:
                    refusal = cast(Iterable[ChatCompletionContentPartRefusalParam], [])
                else:
                    refusal = cast(
                        Iterable[ChatCompletionContentPartRefusalParam],
                        [{"type": "refusal", "refusal": refusal}],
                    )
                content = [part for part in content if part["type"] == "text"]
                refusal = [
                    part for part in content if part["type"] == "refusal"
                ] + list(refusal)
                tool_calls = list(tool_calls) if tool_calls is not None else []
                for tool_call in tool_calls:
                    id_to_name[tool_call["id"]] = tool_call["function"]["name"]
                num_parts = len(content) + len(refusal) + len(tool_calls)
                for i, part in enumerate(content):
                    yield ChatCompletionChunk(
                        id=message_id,
                        choices=[
                            Choice(
                                index=0,
                                delta=ChoiceDelta(
                                    role=role if i == 0 else None,
                                    content=content_part_to_str(part),
                                ),
                                finish_reason="stop" if i == num_parts - 1 else None,
                            )
                        ],
                        created=now(),
                        model=id_to_name.get(message_id, model),
                        object="chat.completion.chunk",
                    )
                for i, part in enumerate(refusal):
                    yield ChatCompletionChunk(
                        id=message_id,
                        choices=[
                            Choice(
                                index=0,
                                delta=ChoiceDelta(
                                    role="assistant"
                                    if i == 0 and len(content) == 0
                                    else None,
                                    refusal=content_part_to_str(part),
                                ),
                                finish_reason="content_filter"
                                if i == num_parts - 1
                                else None,
                            )
                        ],
                        created=now(),
                        model=model,
                        object="chat.completion.chunk",
                    )
                if len(tool_calls) > 0:
                    yield ChatCompletionChunk(
                        id=message_id,
                        choices=[
                            Choice(
                                index=0,
                                delta=ChoiceDelta(
                                    role="assistant" if len(content) == 0 else None,
                                    tool_calls=[
                                        ChoiceDeltaToolCall(
                                            index=i,
                                            id=tool_call["id"],
                                            function=ChoiceDeltaToolCallFunction(
                                                name=tool_call["function"]["name"],
                                                arguments=tool_call["function"][
                                                    "arguments"
                                                ],
                                            ),
                                            type="function",
                                        )
                                        for i, tool_call in enumerate(tool_calls)
                                    ],
                                ),
                                finish_reason="tool_calls",
                            )
                        ],
                        created=now(),
                        model=model,
                        object="chat.completion.chunk",
                    )


class Swarm(BaseModel, nx.DiGraph):  # noqa # type: ignore
    mcpServers: dict[str, StdioServerParameters | str] | None = None
    graph: dict = Field(default_factory=dict)

    @model_validator(mode="wrap")
    @classmethod
    def node_link_validator(
        cls, data: Any, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        """Validate the swarm data using node-link format.

        See https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_data.html

        """
        if not isinstance(data, dict):
            raise ValidationError("Swarm must be a dictionary")
        swarm = handler(data)
        for node in data.get("nodes", []):
            swarm.add_node(
                node["id"],
                **TypeAdapter(NodeData).validate_python(node),
            )
        for edge in data.get("edges", []):
            edge = copy.deepcopy(edge)
            if not isinstance(edge, dict):
                raise ValidationError("Edge must be a dictionary")
            if (source := edge.pop("source", None)) is None or (
                target := edge.pop("target", None)
            ) is None:
                raise ValidationError("Edge must have source and target")
            # Validate condition if present
            if "condition" in edge:
                if isinstance(edge["condition"], str):
                    edge["condition"] = condition_parser(edge["condition"])
                if not callable(edge["condition"]):
                    raise ValueError("Edge condition must be callable")
            swarm.add_edge(source, target, **edge)
        return swarm

    @model_serializer()
    def serialize_swarm(self):  # noqa
        swarm = {
            "nodes": [
                {"id": node}
                | json.loads(TypeAdapter(NodeData).dump_json(data, exclude_none=True))
                for node, data in self.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u if isinstance(u, int) else str(u),
                    "target": v if isinstance(v, int) else str(v),
                }
                | data
                for u, v, data in self.edges(data=True)
            ],
            "graph": self.graph,
        }
        if self.mcpServers:
            swarm["mcpServers"] = {
                name: server_params.model_dump()
                if isinstance(server_params, StdioServerParameters)
                else server_params
                for name, server_params in self.mcpServers.items()
            }
        return swarm

    @property
    def root(self) -> Any:  # noqa
        roots = [node for node, degree in self.in_degree if degree == 0]
        if len(roots) != 1:
            raise ValueError("Swarm must have exactly one root node")
        return roots[0]

    def model_post_init(self, __context: Any) -> None:  # noqa
        nx.DiGraph.__init__(self)

    def add_edge(self, u_of_edge: Hashable, v_of_edge: Hashable, **attr: Any):
        """Add an edge between nodes u and v with optional condition.

        Args:
            u_of_edge: Source node
            v_of_edge: Target node
            condition: Optional callable that takes context variables and returns bool
            **attr: Additional edge attributes

        """
        if u_of_edge not in self._node:
            raise ValueError(f"Node {u_of_edge} not found")
        if v_of_edge not in self._node:
            raise ValueError(f"Node {v_of_edge} not found")
        condition = attr.get("condition", None)
        if condition is not None and not callable(condition_parser(condition)):
            raise ValueError("Edge condition must be callable")
        super().add_edge(u_of_edge, v_of_edge, **attr)

    @property
    def _next_node(self) -> int:
        return max([i for i in self.nodes if isinstance(i, int)] + [-1]) + 1

    def active_successors(self, node: Hashable) -> list[Hashable]:
        """Get successors of a node that have satisfied conditions.

        Args:
            node: The node to get successors for

        Returns:
            List of successor nodes whose conditions are satisfied

        """
        successors = []
        for succ in self.successors(node):
            edge_data = self.edges[node, succ]
            condition = edge_data.get("condition")
            if condition is None or condition(self.graph):
                successors.append(succ)
        return successors

    def _would_early_stop(
        self,
        agents: dict[Hashable, Node],
        tool_calls: dict[Hashable, list[ChatCompletionMessageToolCallParam]],
        execute_tools: bool,
    ) -> bool:
        return (
            len([s for n in agents.keys() for s in self.successors(n)])
            == 0  # no successors
            and (
                len([c for tc in tool_calls.values() for c in tc]) == 0
                or not execute_tools
            )  # no tool calls or tool execution disabled
        )

    async def _execute_agents(
        self,
        agents: dict[Hashable, Node],
        messages: list[ChatCompletionMessageParam],
    ) -> dict[Hashable, list[ChatCompletionMessageParam]]:
        tasks: dict[Hashable, asyncio.Task] = {}
        async with asyncio.TaskGroup() as group:
            for node, agent in agents.items():
                tasks[node] = group.create_task(
                    agent.run(
                        stream=False,
                        context_variables=self.graph,
                        messages=messages,
                    )
                )
        node_messages: dict[Hashable, list[ChatCompletionMessageParam]] = {}
        for node, task in tasks.items():
            node_messages[node] = task.result()
            self.nodes[node]["executed"] = True
        return node_messages

    async def _execute_agents_stream(
        self,
        agents: dict[Hashable, Node],
        messages: list[ChatCompletionMessageParam],
    ) -> AsyncIterator[
        ChatCompletionChunk | dict[Hashable, list[ChatCompletionMessageParam]]
    ]:
        tasks: dict[Hashable, asyncio.Task] = {}
        async with asyncio.TaskGroup() as group:
            for node, agent in agents.items():
                tasks[node] = group.create_task(
                    agent.run(
                        stream=True,
                        context_variables=self.graph,
                        messages=messages,
                    )
                )
        node_messages: dict[Hashable, list[ChatCompletionMessageParam]] = {}
        for node, task in tasks.items():
            chunks = []
            async for chunk in task.result():
                yield chunk
                chunks.append(chunk)
            node_messages[node] = merge_chunks(chunks)  # type: ignore
            self.nodes[node]["executed"] = True
        yield node_messages

    async def _call_tools(
        self,
        tool_calls: dict[Hashable, list[ChatCompletionMessageToolCallParam]],
        current_node: Hashable,
    ) -> dict[Hashable, list[ChatCompletionMessageParam]]:
        messages: dict[Hashable, list[ChatCompletionMessageParam]] = defaultdict(list)
        tasks: dict[
            Hashable, list[tuple[asyncio.Task, ChatCompletionMessageToolCallParam]]
        ] = defaultdict(list)
        async with asyncio.TaskGroup() as group:
            for node, _tool_calls in tool_calls.items():
                for tool_call in _tool_calls:
                    name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                    tasks[node].append(
                        (
                            group.create_task(
                                TOOL_REGISTRY.call_tool(name, arguments, self.graph)
                            ),
                            tool_call,
                        )
                    )
        agents: dict[Hashable, list[Node]] = defaultdict(list)
        for node, _tasks in tasks.items():
            for task, tool_call in _tasks:
                i = tool_call["id"]
                name = tool_call["function"]["name"]
                match result := task.result():
                    case str():
                        messages[node].append(
                            {"role": "tool", "content": result, "tool_call_id": i}
                        )
                    case dict():
                        self.graph |= result
                    case Agent() | Swarm():
                        agents[node].append(result)
                    case list() if all(isinstance(i, Agent | Swarm) for i in result):
                        agents[node].extend(result)
                    case Result() as response:
                        agents[node].extend(response.agents)
                        self.graph |= response.context_variables
                        new_messages = [m for m in response.messages]
                        for m in new_messages:
                            if m["role"] == "tool":
                                m["tool_call_id"] = i
                            else:
                                m["name"] = f"{name} ({i})"
                        messages[node].extend(new_messages)
                    case mcp.types.CallToolResult():
                        content = _mcp_call_tool_result_to_content(result)
                        messages[node].append(
                            {
                                "role": "user",
                                "content": content,
                                "name": f"{name} ({i})",
                            }
                        )
                    case _:
                        raise ValueError(f"Unknown result type: {type(result)}")
        for node, _successors in agents.items():
            for s in _successors:
                next_node = self._next_node
                self.add_node(
                    next_node,
                    **(
                        {"type": "agent", "agent": s}
                        if isinstance(s, Agent)
                        else {"type": "swarm", "swarm": s}
                    ),
                )
                self.add_edge(node, next_node)
        if len(agents) == 0 and len(list(self.successors(current_node))) == 0:
            next_node = self._next_node
            self.add_node(next_node, **(self.nodes[current_node] | {"executed": False}))
            self.add_edge(current_node, next_node)
        return messages

    async def _call_tools_streaming(
        self,
        tool_calls: dict[Hashable, list[ChatCompletionMessageToolCallParam]],
        current_node: Hashable,
    ) -> AsyncIterator[ChatCompletionChunk | ChatCompletionMessageParam]:
        tasks: dict[
            Hashable, list[tuple[asyncio.Task, ChatCompletionMessageToolCallParam]]
        ] = defaultdict(list)
        async with asyncio.TaskGroup() as group:
            for node, _tool_calls in tool_calls.items():
                for tool_call in _tool_calls:
                    name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                    tasks[node].append(
                        (
                            group.create_task(
                                TOOL_REGISTRY.call_tool(name, arguments, self.graph)
                            ),
                            tool_call,
                        )
                    )
        agents: dict[Hashable, list[Node]] = defaultdict(list)
        for node, _tasks in tasks.items():
            for task, tool_call in _tasks:
                tool_call_id = tool_call["id"]
                name = tool_call["function"]["name"]
                match result := task.result():
                    case dict():
                        self.graph |= result
                    case Agent() | Swarm():
                        agents[node].append(result)
                    case list() if all(isinstance(i, Agent | Swarm) for i in result):
                        agents[node].extend(result)
                    case Result() as response:
                        agents[node].extend(response.agents)
                        self.graph |= response.context_variables
                        for message, chunk in zip(
                            response.messages, messages_to_chunks(response.messages)
                        ):
                            yield chunk
                            yield message
                    case mcp.types.CallToolResult() | str():
                        if isinstance(result, str):
                            content: list[ChatCompletionContentPartParam] = [
                                {"type": "text", "text": result},
                            ]
                        else:
                            content = _mcp_call_tool_result_to_content(result)
                        for i, part in enumerate(content):
                            yield ChatCompletionChunk.model_validate(
                                {
                                    "id": tool_call_id,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "role": "tool" if i == 0 else None,
                                                "content": content_part_to_str(part),
                                            },
                                            "finish_reason": "stop"
                                            if i == len(content) - 1
                                            else None,
                                        }
                                    ],
                                    "created": now(),
                                    "model": name,
                                    "object": "chat.completion.chunk",
                                }
                            )
                        yield {
                            "role": "tool",
                            "content": "".join(
                                content_part_to_str(part) for part in content
                            ),
                            "tool_call_id": tool_call_id,
                        }
                    case _:
                        raise ValueError(f"Unknown result type: {type(result)}")
        for node, _successors in agents.items():
            for s in _successors:
                next_node = self._next_node
                self.add_node(
                    next_node,
                    **(
                        {"type": "agent", "agent": s}
                        if isinstance(s, Agent)
                        else {"type": "swarm", "swarm": s}
                    ),
                )
                self.add_edge(node, next_node)
        if len(agents) == 0 and len(list(self.successors(current_node))) == 0:
            next_node = self._next_node
            self.add_node(next_node, **(self.nodes[current_node] | {"executed": False}))
            self.add_edge(current_node, next_node)

    async def _run_and_stream(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        max_turns: int | None = None,
        execute_tools: bool = True,
    ):
        node_data = self.nodes[self.root]
        agents: dict[Hashable, Node] = {
            self.root: cast(Node, node_data[node_data["type"]])
        }
        init_len = len(messages)
        while max_turns is None or len(messages) - init_len < max_turns:
            _messages: dict[Hashable, list[ChatCompletionMessageParam]] = {}
            async for chunk in self._execute_agents_stream(
                messages=messages,
                agents=agents,
            ):
                if isinstance(chunk, dict):
                    _messages.update(chunk)
                else:
                    yield chunk
            messages = [*messages, *[m for nm in _messages.values() for m in nm]]
            tool_calls = {
                node: [
                    tc
                    for m in nm
                    if m["role"] == "assistant" and m.get("tool_calls")
                    for tc in m["tool_calls"]  # type: ignore
                ]
                for node, nm in _messages.items()
            }
            if self._would_early_stop(agents, tool_calls, execute_tools):
                break
            async for chunk in self._call_tools_streaming(
                tool_calls, next(iter(agents))
            ):
                if isinstance(chunk, dict):
                    messages.append(chunk)
                else:
                    yield chunk
            agents = {
                s: (
                    self.nodes[s]["agent"]
                    if self.nodes[s]["type"] == "agent"
                    else self.nodes[s]["swarm"]
                )
                for n in agents.keys()
                for s in self.active_successors(n)
                if all(self.nodes[p].get("executed") for p in self.predecessors(s))
            }

    @overload
    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: Literal[True],
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    @overload
    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: Literal[False] = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
    ) -> list[ChatCompletionMessageParam]: ...

    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: bool = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
    ) -> list[ChatCompletionMessageParam] | AsyncIterator[ChatCompletionChunk]:
        """Run the swarm.

        Args:
            messages: The messages to start the conversation with
            stream: Whether to stream the response
            context_variables: The context variables to pass to the agents
            max_turns: The maximum number of turns to run the swarm for
            execute_tools: Whether to execute tool calls

        """
        for name, server_params in (self.mcpServers or {}).items():
            await TOOL_REGISTRY.add_mcp_server(name, server_params)
        self.graph |= context_variables or {}
        if stream:
            return self._run_and_stream(
                messages=messages,
                max_turns=max_turns,
            )
        node_data = self.nodes[self.root]
        agents: dict[Hashable, Node] = {
            self.root: cast(Node, node_data[node_data["type"]])
        }
        messages = list(copy.deepcopy(messages))
        init_len = len(messages)

        while max_turns is None or len(messages) - init_len < max_turns:
            # get completion with current history, agent
            _messages = await self._execute_agents(agents, messages=messages)
            # dump response to json avoiding pydantic's ValidatorIterator
            messages = [*messages, *[m for nm in _messages.values() for m in nm]]
            tool_calls = {
                node: [
                    tc
                    for m in nm
                    if m["role"] == "assistant" and m.get("tool_calls")
                    for tc in m["tool_calls"]  # type: ignore
                ]
                for node, nm in _messages.items()
            }
            if self._would_early_stop(agents, tool_calls, execute_tools):
                break
            _messages = await self._call_tools(tool_calls, next(iter(agents)))
            messages = [*messages, *[m for nm in _messages.values() for m in nm]]
            agents = {
                s: (
                    self.nodes[s]["agent"]
                    if self.nodes[s]["type"] == "agent"
                    else self.nodes[s]["swarm"]
                )
                for n in agents.keys()
                for s in self.active_successors(n)
                if all(self.nodes[p].get("executed") for p in self.predecessors(s))
            }

        return messages[init_len:]
