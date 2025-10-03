"""OpenAI-compatible server."""

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from .agent import Agent
from .types import GraphMode
from .utils import get_random_string, now
from .version import __version__


class ChatCompletionCreateParams(BaseModel):
    """Request model for chat completions."""

    messages: list[ChatCompletionMessageParam]
    model: str
    stream: bool = False
    max_tokens: int | None = None


def create_server_app(
    swarm: Agent,
    *,
    graph_mode: GraphMode = "locked",
    auto_execute_tools: bool = True,
) -> FastAPI:
    """Create FastAPI app with OpenAI-compatible endpoints."""
    app = FastAPI(title="SwarmX API", version=__version__)

    @app.get("/models")
    async def list_models():
        """List available models."""
        # List all agents in the swarm as models
        agents = swarm.agents
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": now(),
                    "owned_by": "swarmx",
                }
                for name in agents
            ],
        }

    @app.post("/chat/completions")
    async def create_chat_completions(
        params: ChatCompletionCreateParams,
    ):
        """Handle chat completions with streaming support, routing to the requested agent model."""
        # Resolve the target agent based on the model name
        target_agent = (
            swarm if params.model == swarm.name else swarm.agents.get(params.model)
        )
        if target_agent is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{params.model}' not found in swarm agents.",
            )

        if not params.stream:
            # Run the target agent synchronously (non‑stream mode) and build an OpenAI‑compatible response
            messages = await target_agent.run(
                messages=params.messages,
                stream=False,
                max_tokens=params.max_tokens,
                graph_mode=graph_mode,
                auto_execute_tools=auto_execute_tools,
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
            try:
                async for chunk in await target_agent.run(
                    messages=params.messages,
                    stream=True,
                    max_tokens=params.max_tokens,
                    graph_mode=graph_mode,
                    auto_execute_tools=auto_execute_tools,
                ):
                    yield f"data: {chunk.model_dump_json()}\n\n"
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
