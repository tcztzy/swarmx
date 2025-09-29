"""Extended coverage tests for swarmx.agent internals."""

from collections import defaultdict

import pytest
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)

from swarmx.agent import (
    QuotaManager,
    _apply_message_slice,
    _merge_chunk,
)
from swarmx.quota import QuotaExceededError
from swarmx.utils import join

pytestmark = pytest.mark.anyio


def _chunk_from_delta(content: str) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chunk-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
            "created": 1,
            "model": "test-model",
            "object": "chat.completion.chunk",
        }
    )


def test_apply_message_slice_various_cases():
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": f"m{i}"} for i in range(6)
    ]
    assert _apply_message_slice(messages, ":") == messages
    assert _apply_message_slice(messages, "-2:") == messages[-2:]
    assert _apply_message_slice(messages, "0:3") == messages[:3]
    assert _apply_message_slice(messages, "::-1") == list(reversed(messages))


def test_merge_chunk_accumulates_content():
    messages: dict[str, ChatCompletionMessageParam] = defaultdict(
        lambda: {"role": "assistant", "content": ""}
    )
    chunk = _chunk_from_delta("Hello")
    _merge_chunk(messages, chunk)
    chunk = _chunk_from_delta(" world")
    _merge_chunk(messages, chunk)
    assert messages["chunk-1"].get("content") == "Hello world"


async def test_quota_manager_integration():
    manager = QuotaManager(max_tokens=5)
    await manager.consume("agent", 3)
    assert manager.used_tokens == 3
    with pytest.raises(QuotaExceededError):
        await manager.consume("agent", 3)


async def test_join_combines_generators():
    async def gen(prefix: str):
        for ch in "ab":
            yield f"{prefix}-{ch}"

    combined = [item async for item in join(gen("x"), gen("y"))]
    assert set(combined) == {"x-a", "x-b", "y-a", "y-b"}


async def test_merge_chunk_handles_refusal_and_tool_calls():
    messages: dict[str, ChatCompletionMessageParam] = {
        "chunk-2": {
            "role": "assistant",
            "content": [{"type": "text", "text": "start"}],
        },
        "chunk-empty": {"role": "assistant"},
    }
    refusal_chunk = ChatCompletionChunk.model_validate(
        {
            "id": "chunk-2",
            "object": "chat.completion.chunk",
            "created": 3,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": " appended",
                        "refusal": " due to policy",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-2",
                                "type": "function",
                                "function": {"name": "tool", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
    )
    _merge_chunk(messages, refusal_chunk)
    assert (message := messages["chunk-2"])["role"] == "assistant"
    assert (message.get("refusal") or "").endswith("policy")
    assert message.get("tool_calls")[0]["id"] == "call-2"  # type: ignore

    empty_chunk = ChatCompletionChunk.model_validate(
        {
            "id": "chunk-empty",
            "object": "chat.completion.chunk",
            "created": 4,
            "model": "test-model",
            "choices": [],
        }
    )
    assert _merge_chunk(messages, empty_chunk) is None
