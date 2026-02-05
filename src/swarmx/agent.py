"""SwarmX Agent module."""

import json
import uuid
from typing import Any

from jinja2 import Template
from mcp.shared.session import ProgressFnT
from mcp.types import CallToolResult
from openai import NOT_GIVEN, AsyncOpenAI
from openai.types.chat import ChatCompletion, CompletionCreateParams
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from pydantic import Field, PrivateAttr, TypeAdapter

from . import settings
from .clients import PydanticAsyncOpenAI
from .conversion import completion_to_message, result_to_message, stream_to_completion
from .mcp_manager import MCPManager
from .messages import Messages
from .node import Node
from .types import MCPServer, MessagesState
from .utils import GenerateJsonSchemaNoTitles

PARAMS = TypeAdapter(CompletionCreateParams).json_schema(
    schema_generator=GenerateJsonSchemaNoTitles
)
RETURNS = ChatCompletion.model_json_schema(schema_generator=GenerateJsonSchemaNoTitles)


class OpenAIChatCompletionState(MessagesState):
    """State with chat completion."""

    completion: ChatCompletion


class Agent(Node):
    """Agent in the agent graph.

    Using this when you need to break down complex tasks into specialized sub-tasks and
    create reusable, composable AI components. All existing agents are list in tools
    before this one.

    # Examples:
    - Context: User has requested you to write Python code to solve his/her problem.
        user: I want to create a Python script for very professional and academic task.
        assistant: Currently, there are no Python experts in the swarm graph, so we
            need create a new agent who are good at Python programming.
    - Context: User needs to analyze complex data and generate visualizations
        user: I have a large dataset with sales figures and customer demographics that
            needs analysis and visualization
        assistant: The current swarm lacks data analysis expertise. I'll create a
            DataAnalyst agent specialized in pandas, matplotlib, and statistical
            analysis to handle this task.
    - Context: User requests content creation with specific tone and style
        user: I need marketing copy written for a new SaaS product launch, targeting
            enterprise customers with a professional tone
        assistant: There's no marketing content specialist in the swarm. I'll create a
            ContentWriter agent focused on B2B marketing copy, brand voice consistency,
            and conversion optimization.
    - Context: User needs API integration and web scraping capabilities
        user: I want to build a system that scrapes product data from e-commerce sites
            and integrates with our inventory management API
        assistant: The swarm needs web scraping and API integration expertise. I'll
            create a WebIntegration agent specialized in BeautifulSoup, requests, and
            REST API development.
    """

    parameters: dict[str, Any] = PARAMS
    """Empty inputSchema represent OpenAI chat completions API create parameters."""

    returns: dict[str, Any] | None = RETURNS

    model: str | None = None
    """Default model name for chat completions."""

    instructions: str | None = None
    """Agent's instructions, could be a Jinja2 template."""

    client: PydanticAsyncOpenAI | None = None
    """The client to use for the node"""

    mcpServers: dict[str, MCPServer] = Field(default_factory=dict, alias="mcpServers")
    """MCP server configuration for tool execution."""

    _mcp_manager: MCPManager = PrivateAttr(default_factory=MCPManager)

    @property
    def agents(self) -> dict[str, "Agent"]:
        """Expose a single-agent mapping for compatibility."""
        return {self.name: self}

    async def _ensure_mcp_servers(self) -> None:
        for name, server_params in self.mcpServers.items():
            await self._mcp_manager.add_server(name, server_params)

    def _hook_tool_names(self) -> set[str]:
        hook_names: set[str] = set()
        for hook in self.hooks:
            for field in ("on_start", "on_end", "on_handoff", "on_chunk"):
                name = getattr(hook, field)
                if name:
                    hook_names.add(name)
        return hook_names

    async def _execute_hook(
        self,
        hook_name: str,
        messages: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> None:
        result = await self._mcp_manager.call_tool(
            hook_name,
            {
                "messages": messages,
                "context": context,
                "agent": {"name": self.name},
            },
        )
        if isinstance(result, CallToolResult) and result.structuredContent:
            context.update(result.structuredContent)

    async def __call__(
        self,
        arguments: CompletionCreateParams,
        *,
        context: dict[str, Any] | None = None,
        timeout: float | None = None,
        progress_callable: ProgressFnT | None = None,
    ) -> OpenAIChatCompletionState:
        """Run the agent.

        Args:
            arguments: Dict containing messages and completion settings
            context: The context variables to pass to the agent
            timeout: The OpenAI request timeout.
            progress_callable: Called once per streamed chunk when `stream=True`.

        """
        if context is None:
            context = {}
        messages_input = arguments["messages"]
        messages_graph = (
            messages_input if isinstance(messages_input, Messages) else None
        )
        input_messages = list(messages_input)
        filtered_messages = [
            m
            for m in input_messages
            if m.get("role") != "user" or m.get("name") != "approval"
        ]
        completion_messages = list(filtered_messages)
        if self.instructions is not None:
            completion_messages = [
                {
                    "role": "system",
                    "content": await Template(
                        self.instructions, enable_async=True
                    ).render_async(context or {}),
                },
                *completion_messages,
            ]
        await self._ensure_mcp_servers()
        hook_names = self._hook_tool_names()
        for hook in self.hooks:
            if hook.on_start:
                await self._execute_hook(hook.on_start, filtered_messages, context)

        parameters: CompletionCreateParamsBase = dict(arguments)  # type: ignore[arg-type]
        parameters["messages"] = completion_messages
        if not parameters.get("model"):
            parameters["model"] = self.model or settings.OPENAI_MODEL
        if "tools" not in parameters:
            tools = self._mcp_manager.tools_for_openai(
                prefix="mcp__",
                exclude=hook_names,
            )
            if tools:
                parameters["tools"] = tools

        auto_execute_tools = parameters.pop("auto_execute_tools", True)
        stream = parameters.pop("stream", False)
        request_id = str(uuid.uuid4())
        client = self.client or AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )

        async def on_chunk(
            progress: float,
            total: float | None,
            message: str | None,
        ) -> None:
            if progress_callable is not None:
                await progress_callable(progress, total, message)
            for hook in self.hooks:
                if hook.on_chunk:
                    await self._execute_hook(hook.on_chunk, filtered_messages, context)

        conversation = list(parameters["messages"])
        call_parameters = dict(parameters)
        call_parameters.pop("messages", None)
        responses: list[dict[str, Any]] = []
        completion: ChatCompletion | None = None
        remaining_tool_calls = 8
        use_stream = stream

        while True:
            if use_stream:
                _stream = await client.chat.completions.create(
                    **call_parameters,
                    messages=conversation,
                    stream=True,
                    timeout=timeout or NOT_GIVEN,
                )
                completion = await stream_to_completion(_stream, on_chunk=on_chunk)
            else:
                completion = await client.chat.completions.create(
                    **call_parameters,
                    messages=conversation,
                    timeout=timeout or NOT_GIVEN,
                )
            completion._request_id = request_id

            choice = completion.choices[0]
            tool_calls = choice.message.tool_calls or []
            if tool_calls:
                assistant_message = completion_to_message(completion)
                assistant_message["name"] = f"{self.name} ({completion.id})"
                if messages_graph is not None:
                    messages_graph.append_llm_message(assistant_message, completion)
                if not auto_execute_tools:
                    responses.append(assistant_message)
                    break
                conversation.append(assistant_message)
                for tool_call in tool_calls:
                    tool_args: dict[str, Any] = {}
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        tool_args = {}
                    try:
                        result = await self._mcp_manager.call_tool(
                            tool_call.function.name,
                            tool_args,
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        result = exc
                    tool_message = result_to_message(tool_call.id, result)
                    if messages_graph is not None and isinstance(
                        result, CallToolResult
                    ):
                        messages_graph.append_tool_message(
                            tool_call.id, tool_message, result
                        )
                    conversation.append(tool_message)
                remaining_tool_calls -= 1
                if remaining_tool_calls <= 0:
                    break
                use_stream = False
                continue

            assistant_message = completion_to_message(completion)
            if messages_graph is not None:
                messages_graph.append_llm_message(assistant_message, completion)
            responses.append(assistant_message)
            break

        for hook in self.hooks:
            if hook.on_end:
                await self._execute_hook(hook.on_end, responses, context)

        assert completion is not None
        return {
            "messages": responses,
            "completion": completion,
        }
