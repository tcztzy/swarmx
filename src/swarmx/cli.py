"""Command line interface for SwarmX."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import CompletionCreateParams
from pydantic import BaseModel, RootModel

from .agent import (
    CLIENT_REGISTRY,
    Agent,
    T,
)
from .utils import get_random_string
from .version import __version__


async def main(
    *,
    file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            "-f",
            exists=True,
            help="The path to the swarmx file (networkx node_link_data with additional `mcpServers` key)",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            writable=True,
            help="The path to the output file to save the conversation",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose/--quiet", "-v/-q", help="Print the data sent to the model"
        ),
    ] = False,
):
    """SwarmX Command Line Interface."""
    if file is None:
        data = {}
    else:
        data = json.loads(file.read_text())
    client: Agent = Agent.model_validate(data)
    messages: list[ChatCompletionMessageParam] = []
    while True:
        try:
            user_prompt = typer.prompt(">>>", prompt_suffix=" ")
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt,
                }
            )
            async for chunk in await client.run(
                messages=messages,
                stream=True,
            ):
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    typer.echo(delta.content, nl=False)
                if (
                    isinstance(c := getattr(delta, "reasoning_content", None), str)
                    and verbose
                ):
                    typer.secho(c, nl=False, fg="green")
                if delta.refusal is not None:
                    typer.secho(delta.refusal, nl=False, err=True, fg="purple")
                if chunk.choices[0].finish_reason is not None:
                    typer.echo()
        except KeyboardInterrupt:
            break
        except Exception as e:
            messages.append(
                {
                    "role": "assistant",
                    "refusal": f"{e}",
                }
            )
            typer.secho(f"{e}", err=True, fg="red")
            break
    if output is not None:
        output.write_text(json.dumps(messages, indent=2, ensure_ascii=False))
    await CLIENT_REGISTRY.close()


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions."""

    messages: list[ChatCompletionMessageParam]
    model: str = "gpt-4o"
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


def create_server_app(swarm: Agent[T]) -> FastAPI:
    """Create FastAPI app with OpenAI-compatible endpoints."""
    app = FastAPI(title="SwarmX API", version=__version__)

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        # Get unique models from all agents in the swarm

        return {
            "object": "list",
            "data": [
                {
                    "id": swarm.name,
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "swarmx",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: RootModel[CompletionCreateParams]):
        """Handle chat completions with streaming support."""
        messages = list(request.root["messages"])
        stream = request.root.get("stream", False) or False
        model = request.root["model"]

        if not stream:
            raise NotImplementedError("Non-streaming response is not supported.")

        async def generate_stream():
            """Generate streaming response."""
            try:
                async for chunk in await swarm.run(
                    messages=messages,
                    stream=True,
                ):
                    # Convert SwarmX chunk to OpenAI format
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"
            except Exception as e:
                error_chunk = {
                    "id": f"chatcmpl-{get_random_string(10)}",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"Error: {str(e)}"},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream())

    return app


# Create the main typer app
app = typer.Typer(help="SwarmX Command Line Interface")


@app.callback(invoke_without_command=True)
def repl(
    ctx: typer.Context,
    file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            "-f",
            exists=True,
            help="The path to the swarmx file (networkx node_link_data with additional `mcpServers` key)",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            writable=True,
            help="The path to the output file to save the conversation",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose/--quiet", "-v/-q", help="Print the data sent to the model"
        ),
    ] = False,
):
    """Start SwarmX REPL (default command)."""
    if ctx.invoked_subcommand is not None:
        return
    asyncio.run(main(file=file, output=output, verbose=verbose))


@app.command()
def serve(
    host: Annotated[
        str, typer.Option("--host", help="Host to bind the server to")
    ] = "127.0.0.1",
    port: Annotated[
        int, typer.Option("--port", help="Port to bind the server to")
    ] = 8000,
    file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            "-f",
            exists=True,
            help="The path to the swarmx file (networkx node_link_data with additional `mcpServers` key)",
        ),
    ] = None,
):
    """Start SwarmX as an OpenAI-compatible API server."""
    # Load swarm configuration
    if file is None:
        data = {}
    else:
        data = json.loads(file.read_text())

    # Create FastAPI app
    fastapi_app = create_server_app(Agent.model_validate(data))

    # Start the server
    uvicorn.run(fastapi_app, host=host, port=port)


if __name__ == "__main__":
    app()
