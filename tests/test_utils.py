from collections import defaultdict
from typing import Annotated, Any

import mcp.types
import pytest
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)

from swarmx import (
    Agent,
    Result,
    _image_content_to_url,
    _mcp_call_tool_result_to_content,
    _resource_to_file,
    check_instructions,
    content_part_to_str,
    function_to_json,
    merge_chunk,
    merge_chunks,
    messages_to_chunks,
    validate_tool,
)

from .functions import print_account_details

pytestmark = pytest.mark.anyio


def sample(a: int, b: str) -> str:
    """Sample function"""
    return f"{a}{b}"


async def async_func(c: float) -> Agent:
    return Agent()


def no_params() -> dict:
    return {}


class TestFunctionToJson:
    def test_function_to_openai_tool(self):
        assert function_to_json(print_account_details) == {
            "function": {
                "description": "Simple function to print account details.",
                "name": "print_account_details",
                "parameters": {"properties": {}, "type": "object"},
            },
            "type": "function",
        }

    def test_basic_function_conversion(self):
        tool = function_to_json(sample)
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "sample"
        assert tool["function"]["description"] == "Sample function"
        assert tool["function"]["parameters"] == {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
            "required": ["a", "b"],
        }

    def test_context_variables_exclusion(self):
        def with_context(a: int, context_variables: dict) -> str:
            return ""

        tool = function_to_json(with_context)
        assert "context_variables" not in tool["function"]["parameters"]["properties"]

    def test_async_function_handling(self):
        tool = function_to_json(async_func)
        assert tool["function"]["name"] == "async_func"
        assert tool["function"]["parameters"] == {
            "type": "object",
            "properties": {"c": {"type": "number"}},
            "required": ["c"],
        }

    def test_function_with_complex_types(self):
        def complex_types(
            d: Annotated[str, "metadata"], e: list[dict[str, int]]
        ) -> Result:
            """Complex types"""
            return Result(content=[])

        tool = function_to_json(complex_types)
        assert tool["function"]["parameters"] == {
            "type": "object",
            "properties": {
                "d": {"type": "string"},
                "e": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    },
                },
            },
            "required": ["d", "e"],
        }

    def test_function_with_no_parameters(self):
        tool = function_to_json(no_params)
        assert tool["function"]["parameters"] == {"type": "object", "properties": {}}


def test_merge_content_string():
    message = {"role": "assistant", "content": "Hello"}
    delta = ChoiceDelta(content=" world")
    merge_chunk(message, delta)
    assert message["content"] == "Hello world"


def test_merge_content_list():
    message = {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]}
    delta = ChoiceDelta(content=" there")
    merge_chunk(message, delta)
    assert message["content"] == [
        {"type": "text", "text": "Hi"},
        {"type": "text", "text": " there"},
    ]


def test_merge_refusal():
    message = {"role": "assistant", "refusal": "No"}
    delta = ChoiceDelta(refusal=" way")
    merge_chunk(message, delta)
    assert message["refusal"] == "No way"


def test_new_tool_call():
    message = {
        "role": "assistant",
        "tool_calls": defaultdict(
            lambda: {
                "id": "",
                "type": "function",
                "function": {"arguments": "", "name": ""},
            }
        ),
    }
    delta = ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=0,
                id="call_1",
                function=ChoiceDeltaToolCallFunction(arguments="arg", name="func"),
            )
        ]
    )
    merge_chunk(message, delta)
    assert message["tool_calls"][0] == {
        "id": "call_1",
        "function": {"arguments": "arg", "name": "func"},
        "type": "function",
    }


def test_update_existing_tool_call():
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {"arguments": "arg1", "name": ""},
                "type": "function",
            }
        ],
    }
    delta = ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=0,
                function=ChoiceDeltaToolCallFunction(arguments="arg2", name="func2"),
            )
        ]
    )
    merge_chunk(message, delta)
    assert message["tool_calls"][0]["function"]["arguments"] == "arg1arg2"
    assert message["tool_calls"][0]["function"]["name"] == "func2"


def test_multiple_tool_calls():
    message = {
        "role": "assistant",
        "tool_calls": defaultdict(
            lambda: {
                "id": "",
                "type": "function",
                "function": {"arguments": "", "name": ""},
            }
        ),
    }
    delta = ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=0,
                id="call_1",
                function=ChoiceDeltaToolCallFunction(arguments="arg1", name="func1"),
            ),
            ChoiceDeltaToolCall(
                index=1,
                id="call_2",
                function=ChoiceDeltaToolCallFunction(arguments="arg2", name="func2"),
            ),
        ]
    )
    merge_chunk(message, delta)
    assert len(message["tool_calls"]) == 2
    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["tool_calls"][1]["id"] == "call_2"


def test_merge_chunk_with_content():
    """Test merging chunk with content into a message."""
    message = {
        "role": "assistant",
        "content": "Hello, ",
        "reasoning_content": "Thinking... ",
    }

    delta = ChoiceDelta(content=" world!", reasoning_content=" Almost done!")

    merge_chunk(message, delta)

    assert message["content"] == "Hello,  world!"
    assert message["reasoning_content"] == "Thinking...  Almost done!"


def test_merge_chunk_with_content_parts():
    """Test merging chunk with content parts into a message that already has content parts."""
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello, "}],
        "reasoning_content": [{"type": "text", "text": "Thinking... "}],
    }

    delta = ChoiceDelta(content=" world!", reasoning_content=" Almost done!")

    merge_chunk(message, delta)

    assert len(message["content"]) == 2
    assert message["content"][0]["text"] == "Hello, "
    assert message["content"][1]["text"] == " world!"

    assert len(message["reasoning_content"]) == 2
    assert message["reasoning_content"][0]["text"] == "Thinking... "
    assert message["reasoning_content"][1]["text"] == " Almost done!"


def test_merge_chunk_with_refusal():
    """Test merging chunk with refusal into a message."""
    message = {"role": "assistant"}

    delta = ChoiceDelta(refusal="I cannot provide information about that topic.")

    merge_chunk(message, delta)

    assert message["refusal"] == "I cannot provide information about that topic."

    # Test appending to existing refusal
    delta2 = ChoiceDelta(refusal=" Please ask something else.")
    merge_chunk(message, delta2)

    assert (
        message["refusal"]
        == "I cannot provide information about that topic. Please ask something else."
    )


def test_merge_chunk_with_tool_calls():
    """Test merging chunk with tool calls into a message."""
    message = {
        "role": "assistant",
        "tool_calls": defaultdict(
            lambda: {
                "id": "",
                "type": "function",
                "function": {"arguments": "", "name": ""},
            }
        ),
    }

    # First tool call chunk
    delta = ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=0,
                id="call_123",
                function=ChoiceDeltaToolCallFunction(
                    name="search",
                    arguments='{"query": "',
                ),
                type="function",
            )
        ]
    )

    merge_chunk(message, delta)

    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["id"] == "call_123"
    assert message["tool_calls"][0]["function"]["name"] == "search"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"query": "'

    # Continuation of the first tool call
    delta2 = ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=0,
                id="call_123",
                function=ChoiceDeltaToolCallFunction(arguments='weather forecast"}'),
                type="function",
            )
        ]
    )

    merge_chunk(message, delta2)

    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["id"] == "call_123"
    assert message["tool_calls"][0]["function"]["name"] == "search"
    assert (
        message["tool_calls"][0]["function"]["arguments"]
        == '{"query": "weather forecast"}'
    )

    # Second tool call
    delta3 = ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=1,
                id="call_456",
                function=ChoiceDeltaToolCallFunction(name="get_time", arguments="{}"),
                type="function",
            )
        ]
    )

    merge_chunk(message, delta3)

    assert len(message["tool_calls"]) == 2
    assert message["tool_calls"][0]["id"] == "call_123"
    assert message["tool_calls"][1]["id"] == "call_456"
    assert message["tool_calls"][1]["function"]["name"] == "get_time"
    assert message["tool_calls"][1]["function"]["arguments"] == "{}"


def test_merge_chunk_multiple_updates():
    """Test merging multiple types of updates into a message."""
    message = {
        "role": "assistant",
        "content": "Initial content. ",
        "reasoning_content": "Initial reasoning. ",
    }

    delta = ChoiceDelta(
        content="Additional content.",
        reasoning_content="Additional reasoning.",
        refusal="This is a refusal message.",
        tool_calls=[
            ChoiceDeltaToolCall(
                index=0,
                id="tool_call_1",
                function=ChoiceDeltaToolCallFunction(
                    name="test_tool", arguments='{"param": "value"}'
                ),
                type="function",
            )
        ],
    )

    merge_chunk(message, delta)

    assert message["content"] == "Initial content. Additional content."
    assert message["reasoning_content"] == "Initial reasoning. Additional reasoning."
    assert message["refusal"] == "This is a refusal message."
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["id"] == "tool_call_1"
    assert message["tool_calls"][0]["function"]["name"] == "test_tool"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"param": "value"}'


def test_merge_chunks():
    """Test merging multiple chunks into a list of messages."""

    chunks = [
        ChatCompletionChunk(
            id="chunk_1",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        content="Hello, ",
                        reasoning_content="Thinking about greeting the user... ",
                    ),
                    finish_reason=None,
                )
            ],
            created=1716000000,
            model="test-model",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chunk_1",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(
                        content="world!",
                        reasoning_content="Decided to use a simple greeting.",
                    ),
                    finish_reason=None,
                )
            ],
            created=1716000001,
            model="test-model",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chunk_1",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_abc",
                                function=ChoiceDeltaToolCallFunction(
                                    name="get_time", arguments="{}"
                                ),
                                type="function",
                            )
                        ]
                    ),
                    finish_reason=None,
                )
            ],
            created=1716000002,
            model="test-model",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chunk_1",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason="stop",
                )
            ],
            created=1716000003,
            model="test-model",
            object="chat.completion.chunk",
        ),
    ]

    # Run the merge_chunks function with our async generator
    messages = merge_chunks(chunks)

    # Verify results
    assert len(messages) == 1
    message = messages[0]
    assert message["role"] == "assistant"
    assert message["content"] == "Hello, world!"
    assert (
        message["reasoning_content"]
        == "Thinking about greeting the user... Decided to use a simple greeting."
    )
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["id"] == "call_abc"
    assert message["tool_calls"][0]["function"]["name"] == "get_time"
    assert message["tool_calls"][0]["function"]["arguments"] == "{}"


def test_merge_chunks_multiple_messages():
    """Test merging multiple chunks with different IDs into multiple messages."""

    chunks = [
        ChatCompletionChunk(
            id="msg_1",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="First message"),
                    finish_reason=None,
                )
            ],
            created=1716000000,
            model="test-model",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="msg_2",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Second message"),
                    finish_reason=None,
                )
            ],
            created=1716000001,
            model="test-model",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="msg_1",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason="stop",
                )
            ],
            created=1716000002,
            model="test-model",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="msg_2",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason="stop",
                )
            ],
            created=1716000003,
            model="test-model",
            object="chat.completion.chunk",
        ),
    ]
    # Run the merge_chunks function
    messages = merge_chunks(chunks)

    # Sort messages by ID to ensure consistent order for testing
    messages.sort(key=lambda x: x.get("id", ""))

    # Verify results
    assert len(messages) == 2
    assert messages[0]["content"] == "First message"
    assert messages[1]["content"] == "Second message"


class TestCheckInstructions:
    def test_valid_string(self):
        instructions = "You are a helpful assistant"
        result = check_instructions(instructions)
        assert result == instructions

    def test_valid_callable(self):
        def valid_func(context_variables: dict) -> str:
            return "Hello"

        result = check_instructions(valid_func)
        assert result == valid_func

    def test_non_callable_non_string(self):
        with pytest.raises(ValueError) as excinfo:
            check_instructions(42)
        assert "a string or a callable" in str(excinfo.value)

    def test_generic_dict_annotation(self):
        def generic_anno(context_variables: dict[str, str]) -> str:
            return ""

        result = check_instructions(generic_anno)
        assert result == generic_anno

    def test_no_annotation_callable(self):
        def no_anno(context_variables):
            return ""

        result = check_instructions(no_anno)
        assert result == no_anno


class TestValidateTool:
    def test_valid_string_return(self):
        def valid_str() -> str:
            return "valid"

        assert validate_tool(valid_str)["function"]["name"] == "valid_str"

    def test_valid_agent_return(self):
        def valid_agent() -> Agent:
            return Agent()

        assert validate_tool(valid_agent)["function"]["name"] == "valid_agent"

    def test_valid_dict_return(self):
        def valid_dict() -> dict[str, Any]:
            return {"key": "value"}

        assert validate_tool(valid_dict)["function"]["name"] == "valid_dict"

    def test_valid_result_return(self):
        def valid_result() -> Result:
            return Result(content=[])

        assert validate_tool(valid_result)["function"]["name"] == "valid_result"

    def test_unannotated_return(self):
        def unannotated():
            return "no annotation"

        with pytest.warns(FutureWarning):
            checked = validate_tool(unannotated)
        assert checked["function"]["name"] == "unannotated"

    def test_invalid_return_type(self):
        def invalid_return() -> int:
            return 42

        with pytest.raises(TypeError) as excinfo:
            validate_tool(invalid_return)
        assert "must be str, Agent, dict[str, Any], or Result" in str(excinfo.value)

    def test_non_callable_input(self):
        with pytest.raises(TypeError) as excinfo:
            validate_tool(None)
        assert "must be str, Agent, dict[str, Any], or Result" in str(excinfo.value)

    def test_none_return_annotation(self):
        def none_return() -> None:
            return None

        with pytest.raises(TypeError) as excinfo:
            validate_tool(none_return)
        assert "must be str, Agent, dict[str, Any], or Result" in str(excinfo.value)


def test_image_content_to_url():
    """Test converting MCP image content to OpenAI image URL content part."""
    # Create a mock ImageContent
    image_content = mcp.types.ImageContent(
        type="image", mimeType="image/png", data="base64encodeddata"
    )

    # Convert to OpenAI format
    result = _image_content_to_url(image_content)

    # Verify result
    assert result["type"] == "image_url"
    assert result["image_url"]["url"] == "data:image/png;base64,base64encodeddata"


def test_resource_to_file_text():
    """Test converting MCP text resource to OpenAI text content part."""
    # Create a mock TextResourceContents
    text_content = mcp.types.TextResourceContents(
        uri="file:///some/path", mimeType="text/plain", text="Sample text"
    )

    # Create a mock EmbeddedResource with TextResourceContents
    resource = mcp.types.EmbeddedResource(type="resource", resource=text_content)

    # Convert to OpenAI format
    result = _resource_to_file(resource)

    # Verify result
    assert result["type"] == "text"
    assert result["text"] == '\n```text title="file:///some/path"\nSample text\n```\n'


def test_resource_to_file_blob():
    """Test converting MCP blob resource to OpenAI file content part."""
    # Create a mock BlobResourceContents
    blob_content = mcp.types.BlobResourceContents(
        uri="file:///some/path", mimeType="application/pdf", blob="base64encodedblob"
    )

    # Create a mock EmbeddedResource with BlobResourceContents
    resource = mcp.types.EmbeddedResource(type="resource", resource=blob_content)

    # Convert to OpenAI format
    result = _resource_to_file(resource)

    # Verify result
    assert result["type"] == "file"
    assert result["file"]["file_data"] == "base64encodedblob"


def test_content_part_to_str():
    """Test content_part_to_str function with different content part types."""
    # Test text part
    text_part = {"type": "text", "text": "hello"}
    assert content_part_to_str(text_part) == "hello"

    # Test refusal part
    refusal_part = {"type": "refusal", "refusal": "no"}
    assert content_part_to_str(refusal_part) == "no"

    # Test image_url part
    image_part = {
        "type": "image_url",
        "image_url": {"url": "http://example.com/image.png"},
    }
    assert content_part_to_str(image_part) == "![](http://example.com/image.png)"

    # Test unknown type
    unknown_part = {
        "type": "file",
        "file": {"file_data": "SGVsbG8sIFdvcmxkIQ==", "filename": "example.txt"},
    }
    assert (
        content_part_to_str(unknown_part)
        == '\n```text title="example.txt"\nHello, World!\n```\n'
    )

    binary_part = {
        "type": "file",
        "file": {
            "file_data": "iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAANCSURBVEiJtZZPbBtFFMZ/M7ubXdtdb1xSFyeilBapySVU8h8OoFaooFSqiihIVIpQBKci6KEg9Q6H9kovIHoCIVQJJCKE1ENFjnAgcaSGC6rEnxBwA04Tx43t2FnvDAfjkNibxgHxnWb2e/u992bee7tCa00YFsffekFY+nUzFtjW0LrvjRXrCDIAaPLlW0nHL0SsZtVoaF98mLrx3pdhOqLtYPHChahZcYYO7KvPFxvRl5XPp1sN3adWiD1ZAqD6XYK1b/dvE5IWryTt2udLFedwc1+9kLp+vbbpoDh+6TklxBeAi9TL0taeWpdmZzQDry0AcO+jQ12RyohqqoYoo8RDwJrU+qXkjWtfi8Xxt58BdQuwQs9qC/afLwCw8tnQbqYAPsgxE1S6F3EAIXux2oQFKm0ihMsOF71dHYx+f3NND68ghCu1YIoePPQN1pGRABkJ6Bus96CutRZMydTl+TvuiRW1m3n0eDl0vRPcEysqdXn+jsQPsrHMquGeXEaY4Yk4wxWcY5V/9scqOMOVUFthatyTy8QyqwZ+kDURKoMWxNKr2EeqVKcTNOajqKoBgOE28U4tdQl5p5bwCw7BWquaZSzAPlwjlithJtp3pTImSqQRrb2Z8PHGigD4RZuNX6JYj6wj7O4TFLbCO/Mn/m8R+h6rYSUb3ekokRY6f/YukArN979jcW+V/S8g0eT/N3VN3kTqWbQ428m9/8k0P/1aIhF36PccEl6EhOcAUCrXKZXXWS3XKd2vc/TRBG9O5ELC17MmWubD2nKhUKZa26Ba2+D3P+4/MNCFwg59oWVeYhkzgN/JDR8deKBoD7Y+ljEjGZ0sosXVTvbc6RHirr2reNy1OXd6pJsQ+gqjk8VWFYmHrwBzW/n+uMPFiRwHB2I7ih8ciHFxIkd/3Omk5tCDV1t+2nNu5sxxpDFNx+huNhVT3/zMDz8usXC3ddaHBj1GHj/As08fwTS7Kt1HBTmyN29vdwAw+/wbwLVOJ3uAD1wi/dUH7Qei66PfyuRj4Ik9is+hglfbkbfR3cnZm7chlUWLdwmprtCohX4HUtlOcQjLYCu+fzGJH2QRKvP3UNz8bWk1qMxjGTOMThZ3kvgLI5AzFfo379UAAAAASUVORK5CYII=",
            "filename": "emoji.png",
        },
    }
    download_link = content_part_to_str(binary_part)
    assert download_link.startswith(
        "[emoji.png](data:application/octet-stream;base64,"
    ) and download_link.endswith(")")


def test_messages_to_chunks():
    """Test messages_to_chunks function with different message types."""
    # Test simple text message
    text_message = {"role": "assistant", "content": "hello"}
    chunks = list(messages_to_chunks([text_message]))
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "hello"
    assert chunks[0].choices[0].finish_reason == "stop"

    # Test message with multiple content parts
    multi_part_message = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": " world"},
        ],
    }
    chunks = list(messages_to_chunks([multi_part_message]))
    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "hello"
    assert chunks[1].choices[0].delta.content == " world"
    assert chunks[1].choices[0].finish_reason == "stop"

    # Test message with refusal
    refusal_message = {"role": "assistant", "refusal": "no"}
    chunks = list(messages_to_chunks([refusal_message]))
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.refusal == "no"
    assert chunks[0].choices[0].finish_reason == "content_filter"

    # Test message with tool calls
    tool_message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "test", "arguments": "{}"},
            }
        ],
    }
    chunks = list(messages_to_chunks([tool_message]))
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.tool_calls[0].function.name == "test"
    assert chunks[0].choices[0].finish_reason == "tool_calls"


def test_mcp_call_tool_result_to_content():
    """Test converting MCP CallToolResult to OpenAI content parts."""
    # Create a mock CallToolResult with various content types
    call_result = mcp.types.CallToolResult(
        content=[
            mcp.types.TextContent(type="text", text="Some text"),
            mcp.types.ImageContent(
                type="image", mimeType="image/jpeg", data="base64imagedata"
            ),
            mcp.types.EmbeddedResource(
                type="resource",
                resource=mcp.types.TextResourceContents(
                    uri="file:///some/path", mimeType="text/plain", text="Resource text"
                ),
            ),
        ]
    )

    result = _mcp_call_tool_result_to_content(call_result)

    # Verify results
    assert len(result) == 3
    assert result[0] == {"type": "text", "text": "Some text"}
    assert result[1]["type"] == "image_url"
    assert result[2]["type"] == "text"
