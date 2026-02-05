import pytest
from mcp.types import CallToolResult, EmbeddedResource, ImageContent
from openai.types.chat import ChatCompletionChunk

from swarmx.conversion import (
    _image_to_md,
    _resource_to_md,
    result_to_content,
    stream_to_completion,
)

pytestmark = pytest.mark.anyio


async def test_resource_to_md_filename_from_host():
    """Test _resource_to_md filename determination from host."""
    # Create a resource with host but no path
    resource = EmbeddedResource.model_validate(
        {
            "type": "resource",
            "resource": {
                "uri": "example://example.txt",
                "text": "Test content",
            },
        }
    )

    result = _resource_to_md(resource)
    assert "example.txt" in result
    assert "Test content" in result


async def test_resource_to_md_filename_from_mimetype():
    """Test _resource_to_md with text/plain mimeType uses HTML comments."""
    resource = EmbeddedResource.model_validate(
        {
            "type": "resource",
            "resource": {
                "uri": "data://test.txt",
                "text": "Test content",
                "mimeType": "text/plain",
            },
        }
    )

    result = _resource_to_md(resource)
    assert "<!-- begin data://test.txt -->" in result
    assert "Test content" in result
    assert "<!-- end data://test.txt -->" in result


async def test_resource_to_md_markdown_with_nested_code_blocks():
    """Test _resource_to_md with markdown containing nested code blocks."""
    markdown_content = """# Example

```python
def hello():
    print("world")
```

More text here."""

    resource = EmbeddedResource.model_validate(
        {
            "type": "resource",
            "resource": {
                "uri": "file:///example.md",
                "text": markdown_content,
                "mimeType": "text/markdown",
            },
        }
    )

    result = _resource_to_md(resource)
    assert "<!-- begin file:///example.md -->" in result
    assert markdown_content in result
    assert "<!-- end file:///example.md -->" in result
    # Verify the nested code block is preserved
    assert "```python" in result


async def test_resource_to_md_filename_error():
    """Test _resource_to_md filename determination error."""

    # Create a resource with no path, host, or mimeType
    resource = EmbeddedResource.model_validate(
        {"type": "resource", "resource": {"uri": "data://", "text": "Test content"}}
    )

    with pytest.raises(ValueError, match="Cannot determine filename for resource"):
        _resource_to_md(resource)


async def test_resource_to_md_lexer_not_found():
    """Test _resource_to_md with lexer not found"""
    resource = EmbeddedResource.model_validate(
        {
            "type": "resource",
            "resource": {
                "uri": "file:///test.unknown",
                "text": "Test content",
                "mimeType": "application/unknown",
            },
        }
    )

    result = _resource_to_md(resource)
    # Should default to "text" when lexer is not found
    assert "```text" in result
    assert "Test content" in result


async def test_resource_to_md_blob_content():
    """Test _resource_to_md with blob content."""
    resource = EmbeddedResource.model_validate(
        {
            "type": "resource",
            "resource": {
                "uri": "file:///test.png",
                "blob": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                "mimeType": "image/png",
            },
        }
    )

    result = _resource_to_md(resource)
    assert "<embed" in result
    assert 'type="image/png"' in result
    assert "data:image/png;base64," in result
    assert 'title="file:///test.png"' in result


async def test_result_to_content_audio():
    """Test result_to_content with audio content."""

    result = CallToolResult.model_validate(
        {
            "content": [
                {"type": "audio", "data": "base64audiodata", "mimeType": "audio/wav"}
            ]
        }
    )

    content = result_to_content(result)
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert '<audio src="data:audio/wav;base64,base64audiodata"' in content[0]["text"]


async def test_result_to_content_resource_link():
    """Test result_to_content with resource_link content."""

    result = CallToolResult.model_validate(
        {
            "content": [
                {
                    "type": "resource_link",
                    "name": "Test Resource",
                    "uri": "file:///test.txt",
                }
            ]
        }
    )

    content = result_to_content(result)
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert "[Test Resource](file:///test.txt)" in content[0]["text"]


async def test_image_to_md_and_result_to_content_image():
    image = ImageContent.model_validate(
        {
            "type": "image",
            "data": "base64",
            "mimeType": "image/png",
            "meta": {"alt": "alt"},
        }
    )
    md = _image_to_md(image)
    assert md.startswith("![") and "data:image/png" in md

    result = CallToolResult.model_validate({"content": [image.model_dump()]})
    parts = result_to_content(result)
    assert md in parts[0]["text"]


class DummyStream:
    """Simple async iterable over preset chunks."""

    def __init__(self, chunks):
        self._iter = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def _chunk(data: dict) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(data)


async def test_stream_to_chat_completion_basic_aggregation():
    chunks = [
        _chunk(
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello "},
                        "finish_reason": None,
                    }
                ],
                "created": 1,
                "model": "gpt-test",
                "object": "chat.completion.chunk",
            }
        ),
        _chunk(
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "world"},
                        "finish_reason": "stop",
                    }
                ],
                "created": 1,
                "model": "gpt-test",
                "object": "chat.completion.chunk",
                "usage": {
                    "completion_tokens": 2,
                    "prompt_tokens": 3,
                    "total_tokens": 5,
                },
            }
        ),
    ]
    seen = []

    async def progress(progress: float, total: float | None, message: str | None):
        seen.append(message)

    completion = await stream_to_completion(DummyStream(chunks), progress)  # type: ignore
    assert len(seen) == 2
    assert completion.id == "chatcmpl-1"
    assert completion.choices[0].message.content == "Hello world"
    assert completion.choices[0].finish_reason == "stop"
    assert completion.usage is not None
    assert completion.usage.total_tokens == 5


async def test_stream_to_chat_completion_function_call_and_tool_calls():
    chunks = [
        _chunk(
            {
                "id": "chatcmpl-2",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "function_call": {"name": "foo", "arguments": "{'a'"},
                        },
                        "finish_reason": None,
                    }
                ],
                "created": 2,
                "model": "gpt-test",
                "object": "chat.completion.chunk",
            }
        ),
        _chunk(
            {
                "id": "chatcmpl-2",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "function_call": {"arguments": ":1}"},
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_0",
                                    "function": {
                                        "name": "bar",
                                        "arguments": "{'b':",
                                    },
                                    "type": "function",
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
                "created": 2,
                "model": "gpt-test",
                "object": "chat.completion.chunk",
            }
        ),
        _chunk(
            {
                "id": "chatcmpl-2",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": "2}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "created": 2,
                "model": "gpt-test",
                "object": "chat.completion.chunk",
            }
        ),
    ]
    completion = await stream_to_completion(DummyStream(chunks))  # type: ignore
    choice = completion.choices[0]
    assert choice.message.function_call is not None
    assert choice.message.function_call.name == "foo"
    assert choice.message.function_call.arguments == "{'a':1}"
    tool_call = choice.message.tool_calls[0]  # type: ignore[index]
    assert tool_call.function.name == "bar"  # type: ignore[union-attr]
    assert tool_call.function.arguments == "{'b':2}"  # type: ignore[union-attr]
    assert choice.finish_reason == "tool_calls"


async def test_stream_to_chat_completion_empty_stream():
    with pytest.raises(ValueError, match="Stream produced no choices"):
        await stream_to_completion(DummyStream([]))  # type: ignore
