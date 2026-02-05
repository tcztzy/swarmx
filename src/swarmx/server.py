"""OpenAI-compatible server."""

import asyncio
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from .node import Node
from .swarm import Swarm
from .utils import get_random_string, now
from .version import __version__


class ChatCompletionCreateParams(BaseModel):
    """Request model for chat completions."""

    messages: list[ChatCompletionMessageParam]
    model: str
    stream: bool = False
    max_tokens: int | None = None


def create_server_app(
    node: Node,
    auto_execute_tools: bool = True,
) -> FastAPI:
    """Create FastAPI app with OpenAI-compatible endpoints."""
    app = FastAPI(title="SwarmX API", version=__version__)
    app.state.auto_execute_tools = auto_execute_tools

    def get_models() -> dict[str, Node]:
        models: dict[str, Node] = {node.name: node}
        if isinstance(node, Swarm):
            models |= node.nodes
        return models

    @app.get("/models")
    async def list_models():
        """List available models."""
        models = get_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": now(),
                    "owned_by": "swarmx",
                }
                for name in sorted(models)
            ],
        }

    @app.post("/chat/completions")
    async def create_chat_completions(
        params: ChatCompletionCreateParams,
    ):
        """Handle chat completions with streaming support, routing to the requested agent model."""
        model = get_models().get(params.model)
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{params.model}' not found in swarm agents.",
            )

        if not params.stream:
            # Run the target agent synchronously (non‑stream mode) and build an OpenAI‑compatible response
            state = await model(
                {
                    "messages": params.messages,
                    "stream": False,
                    "auto_execute_tools": app.state.auto_execute_tools,
                    **(
                        {"max_tokens": params.max_tokens}
                        if params.max_tokens is not None
                        else {}
                    ),
                },
            )
            messages = (
                list(state.get("messages", []))
                if isinstance(state, dict)
                else list(state)
            )
            return ChatCompletion.model_validate(
                {
                    "id": f"chatcmpl-{get_random_string(10)}",
                    "object": "chat.completion",
                    "created": now(),
                    "model": params.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(messages),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            )

        async def generate_stream():
            """Generate streaming response."""
            queue: asyncio.Queue[str] = asyncio.Queue()
            task: asyncio.Task | None = None

            async def on_chunk(
                progress: float,  # noqa: ARG001
                total: float | None,  # noqa: ARG001
                message: str | None,
            ) -> None:
                if message is not None:
                    await queue.put(message)

            try:
                task = asyncio.create_task(
                    model(
                        {
                            "messages": params.messages,
                            "stream": True,
                            "auto_execute_tools": app.state.auto_execute_tools,
                            **(
                                {"max_tokens": params.max_tokens}
                                if params.max_tokens is not None
                                else {}
                            ),
                        },
                        progress_callable=on_chunk,
                    )
                )
                while not task.done() or not queue.empty():
                    try:
                        chunk_json = await asyncio.wait_for(queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    yield f"data: {chunk_json}\n\n"

                await task
            except asyncio.CancelledError:
                if task is not None:
                    task.cancel()
                raise
            except Exception as e:
                error_chunk = {
                    "id": f"chatcmpl-{get_random_string(10)}",
                    "object": "chat.completion.chunk",
                    "created": now(),
                    "model": params.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": str(e)},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream())

    return app
