"""MCP result conversion utilities."""

import os
from typing import Any, assert_never

from mcp.shared.session import ProgressFnT
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceContents,
    TextResourceContents,
)
from openai import AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartTextParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
)
from pygments.lexers import get_lexer_for_filename, get_lexer_for_mimetype
from pygments.util import ClassNotFound

from .utils import now


def _image_to_md(
    content: ImageContent,
) -> str:
    alt = (content.meta or {}).get("alt", "")
    return f"![{alt}](data:{content.mimeType};base64,{content.data})"


def _resource_to_md(
    resource: EmbeddedResource,
) -> str:
    def get_filename(c: ResourceContents) -> str:
        if c.uri.path is not None:
            filename = os.path.basename(c.uri.path)
        elif c.uri.host is not None:
            filename = c.uri.host
        else:
            raise ValueError("Cannot determine filename for resource.")
        return filename

    def get_lang(c: ResourceContents) -> str:
        if c.mimeType is None:
            try:
                lexer = get_lexer_for_filename(get_filename(c))
            except ClassNotFound:
                lexer = None
        else:
            try:
                lexer = get_lexer_for_mimetype(c.mimeType)
            except ClassNotFound:
                lexer = None
        return lexer.aliases[0] if lexer else "text"

    match resource.resource:
        case TextResourceContents() as c:
            # For markdown/html/plain, insert directly with HTML comment delimiters
            if c.mimeType in ("text/markdown", "text/plain"):
                return f"\n<!-- begin {c.uri} -->\n{c.text}\n<!-- end {c.uri} -->\n"
            # For other types, wrap in code block
            lang = get_lang(c)
            return f'\n```{lang} title="{c.uri}"\n{c.text}\n```\n'
        case BlobResourceContents() as c:
            return (
                "<embed"
                f' type="{c.mimeType or "application/octet-stream"}"'
                f' src="data:{c.mimeType or ""};base64,{c.blob}"'
                f' title="{c.uri}"'
                " />"
            )
        case _ as unreachable:
            assert_never(unreachable)


def result_to_content(
    result: CallToolResult,
) -> list[ChatCompletionContentPartTextParam]:
    """Convert MCP tool call result to text params."""

    def _2text(block: ContentBlock):
        match block.type:
            case "text":
                text = block.text
            case "image":
                text = _image_to_md(block)
            case "resource":
                text = _resource_to_md(block)
            case "audio":
                text = f'<audio src="data:{block.mimeType};base64,{block.data}" />'
            case "resource_link":
                text = f"[{block.name}]({block.uri})"
            case _ as unreachable:
                assert_never(unreachable)
        return text

    return [{"text": _2text(block), "type": "text"} for block in result.content]


def result_to_message(
    tool_call_id: str,
    result: CallToolResult | BaseException,
) -> ChatCompletionToolMessageParam:
    """Convert MCP tool call result to OpenAI's message parameter."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": "".join(p["text"] for p in result_to_content(result))
        if isinstance(result, CallToolResult)
        else str(result),
    }


async def stream_to_completion(
    stream: AsyncStream[ChatCompletionChunk],
    on_chunk: ProgressFnT | None = None,
) -> ChatCompletion:
    """Accumulate a streamed chat completion into a ChatCompletion."""

    def _merge_content(target: list[str], delta_value: str | None):
        if delta_value:
            target.append(delta_value)

    def _merge_function_call(
        target: dict[str, Any], delta_fc: ChoiceDeltaFunctionCall | None
    ):
        if delta_fc is None:
            return
        if delta_fc.name:
            target["name"] = delta_fc.name
        if delta_fc.arguments:
            target.setdefault("arguments_parts", []).append(delta_fc.arguments)

    def _merge_tool_calls(
        target: dict[int, dict[str, Any]],
        delta_tcs: list[ChoiceDeltaToolCall] | None,
    ):
        if not delta_tcs:
            return
        for delta_tc in delta_tcs:
            state = target.setdefault(
                delta_tc.index,
                {"id": None, "name": None, "arguments_parts": [], "type": "function"},
            )
            if delta_tc.id:
                state["id"] = delta_tc.id
            if delta_tc.function:
                if delta_tc.function.name:
                    state["name"] = delta_tc.function.name
                if delta_tc.function.arguments:
                    state["arguments_parts"].append(delta_tc.function.arguments)

    choices: dict[int, dict[str, Any]] = {}
    created: int | None = None
    model: str | None = None
    system_fingerprint: str | None = None
    completion_id: str | None = None
    usage = None
    chunk_index = 0

    async for chunk in stream:
        if on_chunk is not None:
            await on_chunk(chunk_index, None, chunk.model_dump_json(exclude_unset=True))
            chunk_index += 1
        created = created or chunk.created
        model = model or chunk.model
        system_fingerprint = system_fingerprint or chunk.system_fingerprint
        completion_id = completion_id or chunk.id
        usage = chunk.usage or usage
        for choice in chunk.choices:
            state = choices.setdefault(
                choice.index,
                {
                    "content": [],
                    "refusal": [],
                    "role": None,
                    "finish_reason": None,
                    "logprobs": None,
                    "function_call": {},
                    "tool_calls": {},
                },
            )
            delta: ChoiceDelta = choice.delta
            _merge_content(state["content"], delta.content)
            _merge_content(state["refusal"], getattr(delta, "refusal", None))
            if delta.role is not None:
                state["role"] = delta.role
            _merge_function_call(state["function_call"], delta.function_call)
            _merge_tool_calls(state["tool_calls"], delta.tool_calls)
            if choice.finish_reason is not None:
                state["finish_reason"] = choice.finish_reason
            if choice.logprobs is not None:
                state["logprobs"] = choice.logprobs

    if not choices:
        raise ValueError("Stream produced no choices.")

    assembled_choices = []
    for index, state in sorted(choices.items()):
        content = "".join(state["content"])
        refusal = "".join(state["refusal"])
        function_call_data = state["function_call"]
        function_call = None
        if function_call_data:
            function_call = {
                "name": function_call_data.get("name") or "",
                "arguments": "".join(function_call_data.get("arguments_parts", [])),
            }
        tool_calls = []
        for tc_index, tc_state in sorted(state["tool_calls"].items()):
            tool_calls.append(
                {
                    "id": tc_state.get("id") or f"call_{tc_index}",
                    "type": tc_state.get("type") or "function",
                    "function": {
                        "name": tc_state.get("name") or "",
                        "arguments": "".join(tc_state.get("arguments_parts", [])),
                    },
                }
            )
        message: dict[str, Any] = {
            "role": state["role"] or "assistant",
            "content": content,
        }
        if refusal:
            message["refusal"] = refusal
        if function_call is not None:
            message["function_call"] = function_call
        if tool_calls:
            message["tool_calls"] = tool_calls

        assembled_choices.append(
            {
                "index": index,
                "message": message,
                "finish_reason": state["finish_reason"],
                "logprobs": state["logprobs"],
            }
        )

    return ChatCompletion.model_validate(
        {
            "id": completion_id or "",
            "object": "chat.completion",
            "created": created or now(),
            "model": model or "",
            "choices": assembled_choices,
            "system_fingerprint": system_fingerprint,
            "usage": usage,
        }
    )


def completion_to_message(
    completion: ChatCompletion,
) -> ChatCompletionAssistantMessageParam:
    """Convert a ChatCompletion object to ChatCompletionAssistantMessageParam.

    Extract the first choice's message and return its dict representation.
    """
    if not completion.choices:
        raise ValueError("ChatCompletion has no choices")
    choice = completion.choices[0]
    message = choice.message
    result: ChatCompletionAssistantMessageParam = {"role": "assistant"}
    if message.content:
        result["content"] = choice.message.content
    if message.tool_calls is not None:
        result["tool_calls"] = [  # type: ignore
            tool_call.model_dump(exclude={"index"}) for tool_call in message.tool_calls
        ]
    return result
