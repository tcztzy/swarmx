"""SwarmX Agent module."""

import asyncio
import json
import logging
import re
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Iterable,
    Literal,
    TypeVar,
    cast,
    overload,
)

import yaml
from cel import evaluate
from httpx import Timeout
from jinja2 import Template
from markdown_it import MarkdownIt
from mdformat.renderer import MDRenderer
from mdit_py_plugins.front_matter import front_matter_plugin
from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionMessageToolCallUnionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_stream_options_param import (
    ChatCompletionStreamOptionsParam,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsBase,
    ResponseFormat,
)
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.json_schema import GenerateJsonSchema

from . import settings
from .hook import Hook, HookType
from .mcp_client import CLIENT_REGISTRY, exec_tool_call
from .types import MCPServer
from .utils import join

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.basicConfig(filename=".swarmx.log", level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CLIENT: AsyncOpenAI | None = None
T = TypeVar("T")
Mode = Literal["automatic", "semi", "manual"]


def _parse_front_matter(front_matter: str):
    try:
        return yaml.safe_load(front_matter)
    except yaml.YAMLError:
        pass
    data: dict[str, Any] = {}
    for line in front_matter.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        data[key.strip()] = val.strip()
    return data


def _merge_chunk(
    messages: dict[str, ChatCompletionMessageParam],
    chunk: ChatCompletionChunk,
) -> None:
    message = messages[chunk.id]
    delta = chunk.choices[0].delta
    content = message.get("content")
    if delta.content is not None:
        if isinstance(content, str) or content is None:
            message["content"] = (content or "") + delta.content
        else:
            message["content"] = [*content, {"type": "text", "text": delta.content}]  # type: ignore

    if delta.refusal is not None:
        assert message["role"] == "assistant"
        message["refusal"] = (message.get("refusal") or "") + delta.refusal
    if delta.tool_calls is not None:
        # We use defaultdict as intermediate structure here instead of list
        if message["role"] != "assistant":
            raise ValueError("Tool calls can only be added to assistant messages")
        tool_calls = cast(
            defaultdict[int, ChatCompletionMessageToolCallParam],
            message.get(
                "tool_calls",
                defaultdict(
                    lambda: {
                        "id": "",
                        "type": "function",
                        "function": {"arguments": "", "name": ""},
                    },
                ),
            ),
        )
        for call in delta.tool_calls:
            function = call.function
            tool_call = tool_calls[call.index]
            if call.id:
                tool_call["id"] = call.id
            if function:
                tool_call["function"]["arguments"] += function.arguments or ""
                tool_call["function"]["name"] += function.name or ""
            tool_calls[call.index] = tool_call
        message["tool_calls"] = tool_calls  # type: ignore


def _apply_message_slice(
    messages: list[ChatCompletionMessageParam], message_slice: str
) -> list[ChatCompletionMessageParam]:
    """Apply message filters.

    Filters are applied in order. Filters can be either a slice string or a CEL expression. (CEL do not support slice natively)

    >>> _apply_message_slice(messages, "-100:") # take last 100 messages
    >>> _apply_message_slice(messages, "0:10") # take first 10 messages
    >>> _apply_message_slice(messages, ":0") # take no messages
    >>> _apply_message_slice(messages, ":") # take all messages, equivalent to no filter
    >>> _apply_message_slice(messages, "0:10:-1") # RARELY USED: take first 10 messages, reverse order
    """
    if re.match(r"-?\d*:-?\d*(:-?\d*)?", message_slice):
        return messages[
            slice(*[int(v) if v else None for v in message_slice.split(":")])
        ]
    raise ValueError(f"Invalid message slice: {message_slice}")


class SwarmXGenerateJsonSchema(GenerateJsonSchema):
    """Remove the title field from the JSON schema."""

    def field_title_should_be_set(self, schema) -> bool:
        """No title for all fields."""
        return False


class Edge(BaseModel, frozen=True, use_attribute_docstrings=True):
    """Edge in the agent graph.

    Using this when you need to create an edge for transferring conversation control from source agent to target agents. All existing agents are list in tools before this one.

    Examples:
    - Context: User needs to transfer from a general assistant to a specialized Python programming agent
        user: "I need help writing a complex Python script for data processing"
        assistant: "I'll create an edge from the current assistant to the PythonExpert agent, as this requires specialized programming expertise beyond my general capabilities."
    - Context: Data analysis task requires visualization expertise
        user: "I have analyzed the sales data, now I need to create interactive charts and dashboards"
        assistant: "I'll create an edge from DataAnalyst to VisualizationSpecialist agent to handle the chart creation and dashboard development."
    - Context: Content creation task requires multiple specialized agents
        user: "I need to create a technical blog post, then optimize it for SEO and social media sharing"
        assistant: "I'll create edges from ContentWriter to SEOSpecialist and SocialMediaManager agents to handle the optimization and distribution phases."
    - Context: API integration requires security review
        user: "I've built the API integration, but need security validation before deployment"
        assistant: "I'll create an edge from WebIntegration to SecurityReviewer agent to ensure the implementation follows security best practices."

    """

    source: str | tuple[str, ...]
    """Name of the source node, if is array of string, which means transferring to target need all sources done."""
    target: str
    """Name of the target node, could be agent's name or (tool/common expression language) which returns agent names."""


class Parameters(BaseModel, use_attribute_docstrings=True):
    """Popular parameters supported by most providers."""

    frequency_penalty: float | None = None
    """Number between -2.0 and 2.0.

    Positive values penalize new tokens based on their existing frequency in the
    text so far, decreasing the model's likelihood to repeat the same line verbatim.
    """

    logprobs: bool | None = None
    """Whether to return log probabilities of the output tokens or not.

    If true, returns the log probabilities of each output token returned in the
    `content` of `message`.
    """

    max_tokens: int | None = None
    """
    The maximum number of [tokens](/tokenizer) that can be generated in the chat
    completion. This value can be used to control
    [costs](https://openai.com/api/pricing/) for text generated via API.

    This value is now deprecated in favor of `max_completion_tokens`, and is not
    compatible with
    [o-series models](https://platform.openai.com/docs/guides/reasoning).
    """

    presence_penalty: float | None = None
    """Number between -2.0 and 2.0.

    Positive values penalize new tokens based on whether they appear in the text so
    far, increasing the model's likelihood to talk about new topics.
    """

    response_format: ResponseFormat | None = None
    """An object specifying the format that the model must output.

    Setting to `{ "type": "json_schema", "json_schema": {...} }` enables Structured
    Outputs which ensures the model will match your supplied JSON schema. Learn more
    in the
    [Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs).

    Setting to `{ "type": "json_object" }` enables the older JSON mode, which
    ensures the message the model generates is valid JSON. Using `json_schema` is
    preferred for models that support it.
    """

    seed: int | None = None
    """
    This feature is in Beta. If specified, our system will make a best effort to
    sample deterministically, such that repeated requests with the same `seed` and
    parameters should return the same result. Determinism is not guaranteed, and you
    should refer to the `system_fingerprint` response parameter to monitor changes
    in the backend.
    """

    stop: str | list[str] | None = None
    """Not supported with latest reasoning models `o3` and `o4-mini`.

    Up to 4 sequences where the API will stop generating further tokens. The
    returned text will not contain the stop sequence.
    """

    stream_options: ChatCompletionStreamOptionsParam | None = None
    """Options for streaming response. Only set this when you set `stream: true`."""

    temperature: float | None = None
    """What sampling temperature to use, between 0 and 2.

    Higher values like 0.8 will make the output more random, while lower values like
    0.2 will make it more focused and deterministic. We generally recommend altering
    this or `top_p` but not both.
    """

    tool_choice: ChatCompletionToolChoiceOptionParam | None = None
    """
    Controls which (if any) tool is called by the model. `none` means the model will
    not call any tool and instead generates a message. `auto` means the model can
    pick between generating a message or calling one or more tools. `required` means
    the model must call one or more tools. Specifying a particular tool via
    `{"type": "function", "function": {"name": "my_function"}}` forces the model to
    call that tool.

    `none` is the default when no tools are present. `auto` is the default if tools
    are present.
    """

    top_logprobs: int | None = None
    """
    An integer between 0 and 20 specifying the number of most likely tokens to
    return at each token position, each with an associated log probability.
    `logprobs` must be set to `true` if this parameter is used.
    """

    top_p: float | None = None
    """
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass. So 0.1
    means only the tokens comprising the top 10% probability mass are considered.

    We generally recommend altering this or `temperature` but not both.
    """


class Agent(BaseModel, use_attribute_docstrings=True, serialize_by_alias=True):
    """Agent in the agent graph.

    Using this when you need to break down complex tasks into specialized sub-tasks and create reusable, composable AI components. All existing agents are list in tools before this one.

    # Examples:
    - Context: User has requested you to write Python code to solve his/her problem.
        user: "I want to create a Python script for very professional and academic task."
        assistant: "Currently, there are no Python experts in the agent graph (aka swarm), so we need create a new agent who are good at Python programming."
    - Context: User needs to analyze complex data and generate visualizations
        user: "I have a large dataset with sales figures and customer demographics that needs analysis and visualization"
        assistant: "The current swarm lacks data analysis expertise. I'll create a DataAnalyst agent specialized in pandas, matplotlib, and statistical analysis to handle this task."
    - Context: User requests content creation with specific tone and style
        user: "I need marketing copy written for a new SaaS product launch, targeting enterprise customers with a professional tone"
        assistant: "There's no marketing content specialist in the swarm. I'll create a ContentWriter agent focused on B2B marketing copy, brand voice consistency, and conversion optimization."
    - Context: User needs API integration and web scraping capabilities
        user: "I want to build a system that scrapes product data from e-commerce sites and integrates with our inventory management API"
        assistant: "The swarm needs web scraping and API integration expertise. I'll create a WebIntegration agent specialized in BeautifulSoup, requests, and REST API development."
    """

    name: Annotated[str, Field(strict=True, max_length=256, frozen=True)] = "Agent"
    """User-friendly name for the display.
    
    The name is unique among all sub-agents and their nested sub-sub-agents.
    """

    description: str = "You are a helpful AI assistant"
    """Agent's description for tool generation and documentation.

    Here is a template for description.
    ```
    Using this agent when <condition or necessity for this role>.

    Examples:
    - Context: <introduce the background of the real world case>
        user: "<user's query>"
        assistant: "<assistant's response>"
    - <more examples like the first one>
    ```
    """

    model: str = "deepseek-reasoner"
    """The default model to use for the agent."""

    instructions: str | None = None
    """Agent's instructions, could be a Jinja2 template."""

    mcp_servers: dict[str, MCPServer] = Field(default_factory=dict, alias="mcpServers")
    """MCP configuration for the agent. Should be compatible with claude code."""

    parameters: Parameters = Field(default_factory=Parameters)
    """Additional parameters to pass to the chat completion API."""

    client: AsyncOpenAI | None = None
    """The client to use for the node"""

    nodes: "set[Agent]" = Field(default_factory=set)
    """The nodes in the Agent's graph"""

    edges: set[Edge] = Field(default_factory=set)
    """The edges in the Agent's graph"""

    hooks: list[Hook] = Field(default_factory=list)
    """Hooks to execute at various points in the agent lifecycle"""

    _visited: "dict[str, bool]" = PrivateAttr(
        default_factory=lambda: defaultdict(lambda: False)
    )

    def __hash__(self):
        """Since name is unique, make this as hash key."""
        return hash(self.name)

    @classmethod
    def model_validate_md(
        cls,
        md_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ):
        """Markdown parsing with tolerant front matter handling.

        Supports front matter delimited by any line of three or more dashes,
        even when the opening and closing delimiter counts differ. Parses the
        YAML-like key/value pairs manually (one ``key: value`` per line) to avoid
        issues with special-character escaping, then attaches the remaining markdown
        as ``instructions``.
        """
        if isinstance(md_data, (bytes, bytearray)):
            md_data = md_data.decode("utf-8")
        md = MarkdownIt("commonmark", {"breaks": True, "html": True}).use(
            front_matter_plugin
        )
        tokens = md.parse(md_data)
        if (front_matter_token := tokens[0]).type != "front_matter":
            raise ValueError("Invalid agent markdown")
        front_matter = front_matter_token.content
        body = MDRenderer().render(tokens[1:], {}, {})
        try:
            return cls.model_validate(
                _parse_front_matter(front_matter) | {"instructions": body}
            )
        except Exception as e:
            raise ValueError("Invalid agent markdown") from e

    def as_agent_md(self) -> str:
        """Serialize model to markdown with YAML front matter."""
        data = self.model_dump(include={"name", "description", "model"})
        content = self.instructions or "You are a helpful AI assistant."
        front = yaml.safe_dump(data, sort_keys=False)
        return f"---\n{front}---\n\n{content}"

    def dump_agent_md(self, path: Path):
        """Serialize all existing agents to target path."""
        if not path.is_dir():
            raise TypeError("Can only dump agents to a directory.")
        for name, agent in self.agents.items():
            (path / f"{name}.md").write_text(agent.as_agent_md())

    @property
    def agents(self) -> "dict[str, Agent]":
        """Get all agent names in the hierarchy, including self and nested agents."""
        agents: "dict[str, Agent]" = {}

        def collect_agents(agent: "Agent") -> None:
            """Recursively collect all agent names."""
            if agent.name in agents:
                raise ValueError("Duplicated agent name")
            agents[agent.name] = agent
            for subagent in agent.nodes:
                collect_agents(subagent)

        collect_agents(self)

        return agents

    @model_validator(mode="after")
    def validate_unique_agent_name(self):
        """Validate agent names are unique."""
        self.agents
        return self

    @field_validator("client", mode="plain")
    def validate_client(cls, v: Any) -> AsyncOpenAI | None:
        """Validate the client.

        If it's a dict, we create a new AsyncOpenAI client from it.
        If it's None, we use the global DEFAULT_CLIENT.
        Otherwise, we assume it's already a valid AsyncOpenAI client.

        """
        if v is None:
            return None
        if isinstance(v, AsyncOpenAI):
            return v
        if isinstance(timeout_dict := v.get("timeout"), dict):
            v["timeout"] = Timeout(**timeout_dict)
        return AsyncOpenAI(**v)

    @field_serializer("client", mode="plain")
    def serialize_client(self, v: AsyncOpenAI | None) -> dict[str, Any] | None:
        """Serialize the client.

        We only serialize the non-default parameters. api_key would not be serialized
        you can manually set it when deserializing.

        """
        if v is None:
            return None
        client: dict[str, Any] = {}
        if str(v.base_url) != "https://api.openai.com/v1":
            client["base_url"] = str(v.base_url)
        for key in (
            "organization",
            "project",
            "websocket_base_url",
        ):
            if (attr := getattr(v, key, None)) is not None:
                client[key] = attr
        if isinstance(v.timeout, float | None):
            client["timeout"] = v.timeout
        elif isinstance(v.timeout, Timeout):
            client["timeout"] = v.timeout.as_dict()
        if v.max_retries != DEFAULT_MAX_RETRIES:
            client["max_retries"] = v.max_retries
        if bool(v._custom_headers):
            client["default_headers"] = v._custom_headers
        if bool(v._custom_query):
            client["default_query"] = v._custom_query
        return client

    @field_validator("parameters", mode="before")
    def validate_params(cls, v: Any) -> CompletionCreateParamsBase:
        """Validate completion create parameters, ensuring messages and model are dummy."""
        if isinstance(v, dict):
            return v | {"model": "DUMMY", "messages": []}  # type: ignore
        return v

    @field_serializer("parameters", mode="plain")
    def serialize_params(self, v: CompletionCreateParamsBase) -> dict[str, Any]:
        """Serialize completion create parameters, excluding messages and model."""
        r = dict(v)
        r.pop("messages", None)
        r.pop("model", None)
        return r

    def extra_tools(self, mode: Mode = "automatic") -> list[ChatCompletionToolParam]:
        """Extra tools based on operation mode.

        Args:
            mode: Operation mode that affects tool availability:
                - automatic: agent can create new agents and edges for handoff
                - semiautomatic: agent can create edges but not new agents
                - manual: no extra_tools available

        """
        if mode == "manual":
            return []
        base_tools: list[ChatCompletionToolParam] = [
            *[
                {
                    "type": "function",
                    "function": {
                        "name": agent.name,
                        "description": agent.description,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for agent in cast(list[Agent], [self, *self.nodes])
            ]
        ]

        edge_tool: ChatCompletionToolParam = {
            "type": "function",
            "function": {
                "name": "create_edge",
                "description": Edge.__doc__ or "",
                "parameters": Edge.model_json_schema(),
            },
        }

        if mode == "semi":
            return base_tools + [edge_tool]

        # automatic mode - include both create_agent and create_edge
        return base_tools + [
            {
                "type": "function",
                "function": {
                    "name": "create_agent",
                    "description": self.__class__.__doc__ or "",
                    "parameters": self.model_json_schema(),
                },
            },
            edge_tool,
        ]

    def _get_client(self):
        return (
            self.client
            or DEFAULT_CLIENT
            or AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
            )
        )

    async def _execute_hooks(
        self,
        hook_type: HookType,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any],
        *,
        tool_name: str | None = None,
        available_tools: list[ChatCompletionToolParam] | None = None,
        to_agent: "Agent | None" = None,
        chunk: ChatCompletionChunk | None = None,
        completion: ChatCompletion | None = None,
    ):
        """Execute hooks of a specific type.

        Args:
            hook_type: The type of hook to execute (e.g., 'on_start', 'on_end')
            messages: The current messages to pass to hook tools
            context: The context variables to pass to the hook tools
            tool_name: The name of the tool being called (for on_tool_start and on_tool_end)
            available_tools: The available tools can be called
            to_agent: The agent being handed off to (for on_handoff)
            chunk: The ChatCompletionChunk object (for on_chunk)
            completion: The ChatCompletion object (for on_llm_end)

        """
        for hook in [h for h in self.hooks if hasattr(h, hook_type)]:
            hook_name: str = getattr(hook, hook_type)
            hook_tool = CLIENT_REGISTRY.get_tool(hook_name)
            properties = hook_tool.inputSchema["properties"]
            arguments: dict[str, Any] = {}
            available = {"messages": messages, "context": context}
            if chunk is not None:
                available["chunk"] = chunk
            if completion is not None:
                available["completion"] = completion
            if tool_name is not None:
                available["tool"] = CLIENT_REGISTRY.get_tool(tool_name)
            if to_agent is not None:
                available["from_agent"] = self.model_dump(
                    mode="json", exclude_unset=True
                )
                available["to_agent"] = to_agent.model_dump(
                    mode="json", exclude_unset=True
                )
            else:
                available["agent"] = self.model_dump(mode="json", exclude_unset=True)
            if available_tools is not None:
                available["available_tools"] = available_tools
            if chunk is not None:
                available["chunk"] = chunk
            if completion is not None:
                available["completion"] = completion
            for key, value in available.items():
                if key in properties:
                    arguments |= {key: value}
            try:
                result = await CLIENT_REGISTRY.call_tool(hook_name, arguments)
                if result.structuredContent is None:
                    raise ValueError("Hook tool must return structured content")
                context |= result.structuredContent
            except Exception as e:
                raise e

    @overload
    async def _execute_tool_with_hooks(
        self,
        tool_call: ChatCompletionMessageToolCallParam,
        stream: Literal[False],
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any],
    ) -> ChatCompletion: ...

    @overload
    async def _execute_tool_with_hooks(
        self,
        tool_call: ChatCompletionMessageToolCallParam,
        stream: Literal[True],
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any],
    ) -> ChatCompletionChunk: ...

    async def _execute_tool_with_hooks(
        self,
        tool_call: ChatCompletionMessageToolCallParam,
        stream: bool,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any],
    ) -> ChatCompletion | ChatCompletionChunk:
        """Execute a tool call with on_tool_start and on_tool_end hooks.

        Args:
            tool_call: The tool call to execute
            stream: Whether to stream the response
            messages: The current messages
            context: The context variables

        Returns:
            The result of the tool execution

        """
        tool_name = tool_call["function"]["name"]
        await self._execute_hooks(
            "on_tool_start", messages, context, tool_name=tool_name
        )

        try:
            result = await exec_tool_call(tool_call, stream)  # type: ignore
            await self._execute_hooks(
                "on_tool_end", messages, context, tool_name=tool_name
            )
            return result
        except Exception as e:
            logger.warning(f"Tool execution failed for {tool_name}: {e}")
            await self._execute_hooks(
                "on_tool_end", messages, context, tool_name=tool_name
            )
            raise

    async def _get_system_prompt(
        self,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """Get the system prompt for the agent.

        Args:
            context: The context variables to pass to the agent

        """
        if self.instructions is None:
            return None
        return await Template(self.instructions, enable_async=True).render_async(
            context or {}
        )

    async def _prepare_chat_completion_params(
        self,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any] | None = None,
        mode: Mode = "automatic",
    ) -> CompletionCreateParamsBase:
        """Prepare parameters for chat completion."""
        message_slice: str | None = (context or {}).get("message_slice")
        if message_slice is not None:
            messages = _apply_message_slice(messages, message_slice)
        system_prompt = await self._get_system_prompt(context)
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}, *messages]
        parameters = self.parameters.model_dump(mode="json", exclude_none=True) | {
            "messages": messages,
            "model": self.model,
        }
        if len(tools := (context or {}).get("tools", CLIENT_REGISTRY.tools)) > 0:
            parameters["tools"] = self.extra_tools(mode) + tools
        return parameters  # type: ignore

    @overload
    async def _create_chat_completion(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any] | None = None,
        stream: Literal[True],
        mode: Mode = "automatic",
    ) -> AsyncGenerator[ChatCompletionChunk, None]: ...

    @overload
    async def _create_chat_completion(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any] | None = None,
        stream: Literal[False] = False,
        mode: Mode = "automatic",
    ) -> ChatCompletion: ...

    async def _create_chat_completion(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any] | None = None,
        stream: bool = False,
        mode: Mode = "automatic",
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None]:
        """Get a chat completion for the agent with UUID tracing.

        Args:
            messages: The messages to start the conversation with
            context: The context variables to pass to the agent
            stream: Whether to stream the response
            mode: Operation mode that affects extra_tools availability:
                - automatic: agent can create new agents and edges for handoff
                - semiautomatic: agent can create edges but not new agents
                - manual: no extra_tools available

        """
        # Even OpenAI support x-request-id header, but most providers don't support
        # So we should manually set it for each.
        request_id = str(uuid.uuid4())
        parameters = await self._prepare_chat_completion_params(messages, context, mode)
        logger.info(
            json.dumps(parameters | {"stream": stream, "request_id": request_id})
        )
        client = self._get_client()
        if stream:

            async def traced_stream():
                async for chunk in await client.chat.completions.create(
                    stream=stream, **parameters
                ):
                    chunk._request_id = request_id
                    chunk.model = self.name
                    yield chunk

            return traced_stream()
        else:
            result = await client.chat.completions.create(stream=stream, **parameters)
            result._request_id = request_id
            result.model = self.name
            return result

    async def _execute_tool_calls(
        self,
        tool_calls: Iterable[ChatCompletionMessageToolCallUnionParam],
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any],
        stream: bool,
    ):
        async with asyncio.TaskGroup() as tg:
            tasks = []
            for tool_call in tool_calls:
                if tool_call["type"] == "custom":
                    continue
                match tool_call["function"]["name"]:
                    case "create_agent":
                        agent = Agent.model_validate_json(
                            tool_call["function"]["arguments"]
                        )
                        self.nodes.add(agent)
                    case "create_edge":
                        edge = Edge.model_validate_json(
                            tool_call["function"]["arguments"]
                        )
                        self.edges.add(edge)
                    case name if name in [
                        self.name,
                        *[agent.name for agent in self.nodes],
                    ]:
                        self.edges.add(Edge(source=self.name, target=name))
                    case _:
                        task = tg.create_task(
                            self._execute_tool_with_hooks(  # type: ignore[call-overload]
                                tool_call, stream, messages, context
                            )
                        )
                        tasks.append(task)
            for future in asyncio.as_completed(tasks):
                yield await future

    def _get_agent_by_name(self, name: str) -> "Agent":
        """Get agent by name.

        Only self & level 1 sub agents would be returned, avoid directly handoff to
        sub-sub-agent.
        """
        if name == self.name:
            return self
        for agent in self.nodes:
            if agent.name == name:
                return agent
        raise KeyError(f"Agent {name} not exist in nodes")

    async def _resolve_edge_target(
        self, target: str, context: dict[str, Any] | None = None
    ) -> "set[Agent]":
        """Resolve edge target, which can be a node name, function name, or CEL expression."""
        # First check if target exists as a node
        try:
            return {self._get_agent_by_name(target)}
        except KeyError:
            pass

        # Then check if target is a function in CLIENT_REGISTRY
        try:
            result = await CLIENT_REGISTRY.call_tool(target, context or {})

            if result.structuredContent is not None:
                r = result.structuredContent.get("result")
                if isinstance(r, list) and all(isinstance(item, str) for item in r):
                    return {self._get_agent_by_name(s) for s in r}
                elif isinstance(r, str):
                    return {self._get_agent_by_name(r)}
                else:
                    raise TypeError(
                        "Conditional edge should return string or list of string only"
                    )
            else:
                if len(result.content) != 1 or result.content[0].type != "text":
                    raise ValueError(
                        "Conditional edge should return one text content block only"
                    )
                return {self._get_agent_by_name(result.content[0].text)}
        except KeyError:
            pass

        # Finally try to evaluate as CEL expression
        try:
            result = evaluate(target, context)
            if isinstance(result, str):
                return {self._get_agent_by_name(result)}
            elif isinstance(result, list) and all(
                isinstance(item, str) for item in result
            ):
                return {self._get_agent_by_name(s) for s in result}
            else:
                raise ValueError(
                    f"CEL expression must return str or list[str], got {type(result)}"
                )
        except Exception as e:
            raise ValueError(f"Invalid edge target '{target}': {e}")

    async def _handoff(
        self,
        *,
        agent: "Agent",
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> AsyncGenerator[ChatCompletion | ChatCompletionChunk, None]:
        """Handoff to the agent."""
        if context is None:
            context = {}
        await self._execute_hooks("on_handoff", messages, context, to_agent=agent)

        async for chunk in await agent.run(  # type: ignore
            messages=messages, stream=stream, context=context
        ):
            yield chunk  # type: ignore

        self._visited[agent.name] = True

        async for chunk in join(
            *[
                self._handoff(
                    agent=target, messages=messages, context=context, stream=stream
                )
                for edge in self.edges
                if edge.source == agent.name
                or (
                    isinstance(edge.source, tuple)
                    and agent.name in edge.source
                    and all(self._visited[n] for n in edge.source)
                )
                for target in await self._resolve_edge_target(edge.target, context)
            ]  # type: ignore
        ):
            yield chunk

    async def _run_stream(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any] | None = None,
        mode: Mode = "automatic",
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Run the agent and stream the response.

        Args:
            messages: The messages to start the conversation with
            context: The context variables to pass to the agent
            mode: Operation mode that affects extra_tools availability:
                - automatic: agent can create new agents and edges for handoff
                - semiautomatic: agent can create edges but not new agents
                - manual: no extra_tools available

        """
        if context is None:
            context = {}
        init_len = len(messages)
        _messages: dict[str, ChatCompletionMessageParam] = defaultdict(
            lambda: {"role": "assistant"}
        )
        await self._execute_hooks(
            "on_llm_start",
            messages,
            context,
            available_tools=CLIENT_REGISTRY.tools,
        )
        async for chunk in await self._create_chat_completion(
            messages=messages,
            context=context,
            stream=True,
            mode=mode,
        ):
            logger.info(
                json.dumps(
                    chunk.model_dump(mode="json", exclude_unset=True)
                    | {"request_id": chunk._request_id}
                )
            )
            await self._execute_hooks(
                "on_chunk",
                messages
                + [m for i, m in _messages.items() if i != chunk.id]
                + [_messages[chunk.id]],
                context,
                chunk=chunk,
            )
            _merge_chunk(_messages, chunk)
            yield chunk
            if chunk.choices[0].finish_reason is not None:
                messages.append(_messages[chunk.id])
        else:
            if len(messages) - init_len != len(_messages):
                raise ValueError("Number of messages does not match number of chunks")
        await self._execute_hooks("on_llm_end", messages, context)
        latest_message = messages[-1]
        if (
            latest_message["role"] == "assistant"
            and (tool_calls := latest_message.get("tool_calls")) is not None
        ):
            async for chunk in self._execute_tool_calls(
                tool_calls, messages, context, True
            ):
                yield chunk

        generators: list[AsyncGenerator[ChatCompletionChunk, None]] = []
        for edge in self.edges:
            if edge.source == self.name:
                for target in await self._resolve_edge_target(edge.target):
                    generators.append(
                        self._handoff(
                            agent=target,
                            messages=messages,
                            context=context,
                            stream=True,
                        )  # type: ignore
                    )
        async for chunk in join(*generators):
            yield chunk

    async def _run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any] | None = None,
        mode: Mode = "automatic",
    ) -> AsyncGenerator[ChatCompletion, None]:
        """Run the agent and yield ChatCompletion objects.

        Args:
            messages: The messages to start the conversation with
            context: The context variables to pass to the agent
            mode: Operation mode that affects extra_tools availability:
                - automatic: agent can create new agents and edges for handoff
                - semiautomatic: agent can create edges but not new agents
                - manual: no extra_tools available

        """
        if context is None:
            context = {}
        await self._execute_hooks(
            "on_llm_start",
            messages,
            context,
            available_tools=CLIENT_REGISTRY.tools,
        )
        completion = await self._create_chat_completion(
            messages=messages,
            context=context,
            stream=False,
            mode=mode,
        )
        logger.info(
            json.dumps(
                completion.model_dump(mode="json", exclude_unset=True)
                | {"request_id": completion._request_id}
            )
        )
        await self._execute_hooks(
            "on_llm_end", messages, context, completion=completion
        )
        # Yield the completion object to preserve all metadata
        yield completion

        # Extract message for hook processing and tool calls
        message = cast(
            ChatCompletionAssistantMessageParam,
            completion.choices[0].message.model_dump(
                mode="json", exclude_unset=True, exclude_none=True
            ),
        )
        message["name"] = f"{self.name} ({completion.id})"
        messages.append(message)

        if (tool_calls := message.get("tool_calls")) is not None:
            async for chunk in self._execute_tool_calls(
                tool_calls, messages, context, False
            ):
                yield chunk

        generators: list[AsyncGenerator[ChatCompletion, None]] = []
        for edge in self.edges:
            if edge.source == self.name:
                for target in await self._resolve_edge_target(edge.target):
                    generators.append(
                        self._handoff(
                            agent=target,
                            messages=messages,
                            context=context,
                            stream=False,
                        )  # type: ignore
                    )
        async for completion in join(*generators):
            yield completion

    @overload
    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: Literal[True],
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ChatCompletionChunk, None]: ...

    @overload
    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: Literal[False] = False,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ChatCompletion, None]: ...

    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: bool = False,
        context: dict[str, Any] | None = None,
        mode: Mode = "automatic",
    ) -> (
        AsyncGenerator[ChatCompletion, None] | AsyncGenerator[ChatCompletionChunk, None]
    ):
        """Run the agent.

        Args:
            messages: The messages to start the conversation with
            stream: Whether to stream the response
            context: The context variables to pass to the agent
            mode: Operation mode that affects extra_tools availability:
                - automatic: agent can create new agents and edges for handoff
                - semiautomatic: agent can create edges but not new agents
                - manual: no extra_tools available

        """
        for name, server_params in (
            (settings.mcp_servers if settings is not None else {}) | self.mcp_servers
        ).items():
            await CLIENT_REGISTRY.add_server(name, server_params)
        messages = deepcopy(messages)
        # context is intentionally not deep copied since it's mutable
        if context is None:
            context = {}
        await self._execute_hooks("on_start", messages, context)
        if stream:
            g = self._run_stream(
                messages=messages,
                context=context,
                mode=mode,
            )
        else:
            g = self._run(
                messages=messages,
                context=context,
                mode=mode,
            )
        await self._execute_hooks("on_end", messages, context)
        return g
