"""MCP result conversion utilities."""

import os
from typing import assert_never

from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceContents,
    TextResourceContents,
)
from openai.types.chat import (
    ChatCompletionContentPartTextParam,
    ChatCompletionToolMessageParam,
)
from pygments.lexers import get_lexer_for_filename, get_lexer_for_mimetype
from pygments.util import ClassNotFound


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
