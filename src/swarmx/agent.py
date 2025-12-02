"""SwarmX Agent module."""

import json
import logging
import uuid
import warnings
from copy import deepcopy
from typing import Annotated, Any, cast

from jinja2 import Template
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsBase,
)
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    TypeAdapter,
)

from . import settings
from .clients import PydanticAsyncOpenAI
from .hook import Hook, HookType
from .mcp_manager import MCPManager
from .quota import QuotaManager
from .types import CompletionCreateParams, MessagesState
from .utils import completion_to_message

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.basicConfig(filename=".swarmx.log", level=logging.INFO)
logger = logging.getLogger(__name__)


class Agent(BaseModel):
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

    name: Annotated[
        str, Field(pattern=r"([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*)", frozen=True)
    ]
    """User-friendly name for the display.

    The name is unique among all nodes.
    """

    description: str | None = None
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

    parameters: dict[str, Any] = Field(
        default_factory=lambda: TypeAdapter(CompletionCreateParams).json_schema()
    )
    """Empty inputSchema represent OpenAI chat completions API create parameters."""

    model: str
    """The default model to use for the agent."""

    instructions: str | None = None
    """Agent's instructions, could be a Jinja2 template."""

    client: PydanticAsyncOpenAI | None = None
    """The client to use for the node"""

    hooks: list[Hook] = Field(default_factory=list)
    """Hooks to execute at various points in the agent lifecycle"""

    _mcp_manager: MCPManager = PrivateAttr(default_factory=MCPManager)

    def __hash__(self):
        """Agent name is unique among all swarm."""
        return hash(self.name)

    @classmethod
    def as_init_tool(cls) -> ChatCompletionFunctionToolParam:
        """As init tool."""
        return {
            "type": "function",
            "function": {
                "name": cls.__name__,
                "description": cls.__doc__ or "",
                "parameters": cls.model_json_schema(),
            },
        }

    def as_call_tool(self) -> ChatCompletionFunctionToolParam:
        """As call tool."""
        tool: ChatCompletionFunctionToolParam = {
            "type": "function",
            "function": {
                "name": self.name,
                "parameters": self.parameters,
            },
        }
        if self.description is not None:
            tool["function"]["description"] = self.description
        return tool

    async def __call__(
        self,
        arguments: CompletionCreateParams,
        *,
        context: dict[str, Any] | None = None,
        quota_manager: QuotaManager | None = None,
    ) -> MessagesState:
        """Run the agent.

        Args:
            arguments: Dict containing messages and completion settings
            context: The context variables to pass to the agent
            quota_manager: The quota manager for tokens and other resources
            auto_execute_tools: Automatically execute tools or not

        """
        stream = arguments.get("stream", False)
        if stream:
            raise NotImplementedError("Stream mode is not yet supported now.")
        arguments = deepcopy(arguments)
        messages = arguments["messages"]
        init_len = len(messages)
        if context is None:
            context = {}
        if quota_manager is None:
            quota_manager = QuotaManager(arguments.get("max_tokens"))  # type: ignore[arg-type]
        await self._execute_hooks("on_start", messages, context)
        completion = await self._create_chat_completion(
            params=arguments,
            context=context,
            quota_manager=quota_manager,
        )
        total_tokens = completion.usage.total_tokens if completion.usage else 0
        await quota_manager.consume(self.name, total_tokens)
        logger.info(
            json.dumps(
                completion.model_dump(mode="json", exclude_unset=True)
                | {"request_id": completion._request_id}
            )
        )

        # Extract message for hook processing and tool calls
        message = completion_to_message(completion)
        message["name"] = f"{self.name} ({completion.id})"
        messages.append(message)

        return {"messages": messages[init_len:]}

    async def _execute_hooks(
        self,
        hook_type: HookType,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any],
        *,
        available_tools: list[ChatCompletionFunctionToolParam] | None = None,
        to_agent: "Agent | None" = None,
        chunk: ChatCompletionChunk | None = None,
        completion: ChatCompletion | None = None,
    ):
        """Execute hooks of a specific type.

        Args:
            hook_type: The type of hook to execute (e.g., 'on_start', 'on_end')
            messages: The current messages to pass to hook tools
            context: The context variables to pass to the hook tools
            available_tools: The available tools can be called
            to_agent: The agent being handed off to (for on_handoff)
            chunk: The ChatCompletionChunk object (for on_chunk)
            completion: The ChatCompletion object (for on_llm_end)

        """
        for hook in [h for h in self.hooks if getattr(h, hook_type, None) is not None]:
            hook_name: str = getattr(hook, hook_type)
            hook_tool = self._mcp_manager.get_tool(hook_name)
            properties = hook_tool.inputSchema["properties"]
            arguments: dict[str, Any] = {}
            available = {"messages": messages, "context": context}
            if chunk is not None:
                available["chunk"] = chunk
            if completion is not None:
                available["completion"] = completion
            if to_agent is not None:
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
                result = await self._mcp_manager.call_tool(hook_name, arguments)
                if result.structuredContent is None:
                    raise ValueError("Hook tool must return structured content")
                context |= result.structuredContent
            except Exception as e:
                raise e

    async def _get_system_prompt(
        self,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """Get the system prompt for the agent.

        Args:
            context: The context variables to pass to the agent

        """
        parts = []
        if self.instructions is not None:
            parts.append(
                await Template(self.instructions, enable_async=True).render_async(
                    context or {}
                )
            )
        if len(agent_md_content := settings.get_agents_md_content()) > 0:
            parts.append(
                "Following are extra contexts, what were considered as long-term memory.\n"
                + agent_md_content
            )
        if len(parts) > 0:
            return "\n\n".join(parts)
        return None

    async def _prepare_chat_completion_params(
        self,
        parameters: CompletionCreateParams,
        context: dict[str, Any] | None = None,
    ) -> CompletionCreateParamsBase:
        """Prepare parameters for chat completion."""
        messages = [
            cast(
                ChatCompletionMessageParam,
                {
                    k: v
                    for k, v in m.items()
                    if k not in ("parsed", "reasoning_content")
                },
            )
            for m in parameters["messages"]
            if not (m.get("role") == "user" and m.get("name") == "approval")
        ]
        system_prompt = await self._get_system_prompt(context)
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}, *messages]
        tools: list[ChatCompletionFunctionToolParam] = [*parameters.get("tools", [])]
        if tools:
            parameters["tools"] = tools
        else:
            parameters.pop("tools", None)
        return parameters | {
            "messages": messages,
            "model": self.model,
        }  # type: ignore

    async def _create_chat_completion(
        self,
        *,
        params: CompletionCreateParams,
        context: dict[str, Any],
        quota_manager: QuotaManager,
    ) -> ChatCompletion:
        """Get a chat completion for the agent with UUID tracing.

        Args:
            params: Parameters to create a chat completion
            context: The context variables to pass to the agent
            quota_manager: Quota manager for tokens and other resource.

        """
        # Even OpenAI support x-request-id header, but most providers don't support
        # So we should manually set it for each.
        request_id = str(uuid.uuid4())
        parameters = await self._prepare_chat_completion_params(
            params,
            context,
        )
        logger.info(json.dumps(parameters | {"request_id": request_id}))
        client = self.client or AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )
        messages = params["messages"]
        # Execute on_llm_start hook before making the request
        await self._execute_hooks(
            "on_llm_start",
            messages,
            context,
        )

        result = await client.chat.completions.create(**parameters)
        result._request_id = request_id
        total_tokens = result.usage.total_tokens if result.usage else 0
        await quota_manager.consume(self.name, total_tokens)
        # Trigger on_llm_end hook for non‑stream response
        await self._execute_hooks("on_llm_end", messages, context, completion=result)
        return result
