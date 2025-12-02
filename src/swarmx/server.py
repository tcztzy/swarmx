"""OpenAI-compatible server."""

import json
from typing import cast

from fastapi import FastAPI, HTTPException
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

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
    swarm: Swarm,
    *,
    auto_execute_tools: bool = True,
) -> FastAPI:
    """Create FastAPI app with OpenAI-compatible endpoints."""
    app = FastAPI(title="SwarmX API", version=__version__)

    @app.get("/models")
    async def list_models():
        """List available models."""
        # List all agents in the swarm as models
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": now(),
                    "owned_by": "swarmx",
                }
                for name in swarm
            ],
        }

    @app.post("/chat/completions")
    async def create_chat_completions(
        params: ChatCompletionCreateParams,
    ):
        """Handle chat completions with streaming support, routing to the requested agent model."""
        # Resolve the target agent based on the model name
        if params.model == swarm.name:
            target_agent: Swarm | None = swarm
        else:
            node = swarm.nodes.get(params.model)
            target_agent = cast(
                Swarm | None, node[node["type"]] if node is not None else None
            )
        if target_agent is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{params.model}' not found in swarm agents.",
            )

        if not params.stream:
            # Run the target agent synchronously (non‑stream mode) and build an OpenAI‑compatible response
            messages = await target_agent(
                {
                    "messages": params.messages,
                    "stream": False,
                    "max_tokens": params.max_tokens,
                },
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

        # async def generate_stream():
        #     """Generate streaming response."""
        #     try:
        #         async for chunk in await target_agent(
        #             {
        #                 "messages": params.messages,
        #                 "stream": True,
        #                 "max_tokens": params.max_tokens,
        #             },
        #         ):
        #             yield f"data: {chunk.model_dump_json()}\n\n"
        #     except Exception as e:
        #         error_chunk = {
        #             "id": f"chatcmpl-{get_random_string(10)}",
        #             "object": "chat.completion.chunk",
        #             "created": now(),
        #             "model": params.model,
        #             "choices": [
        #                 {
        #                     "index": 0,
        #                     "delta": {"content": str(e)},
        #                     "finish_reason": "stop",
        #                 }
        #             ],
        #         }
        #         yield f"data: {json.dumps(error_chunk)}\n\n"
        #     finally:
        #         yield "data: [DONE]\n\n"

        # return StreamingResponse(generate_stream())

    return app
