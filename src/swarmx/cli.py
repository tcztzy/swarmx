"""Command line interface for SwarmX."""

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import TypeAdapter, ValidationError

from .agent import Agent
from .mcp_server import create_mcp_server
from .server import create_server_app
from .swarm import Swarm
from .version import __version__


def load_node(data: dict) -> Agent | Swarm:
    """Load a Swarm config when possible, otherwise fall back to a single Agent."""
    if data:
        try:
            return TypeAdapter(Swarm).validate_python(data)
        except ValidationError:
            return Agent.model_validate(data)
    try:
        return Agent.model_validate(data)
    except ValidationError:
        return Agent(name="Agent")


async def amain(
    *,
    file: Path | None = None,
    output: Path | None = None,
    verbose: bool = False,
):
    """SwarmX REPL."""
    if file is None:
        data = {}
    else:
        data = json.loads(file.read_text())
    node = load_node(data)
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
            if verbose:
                typer.secho("Request:", err=True, fg="cyan")
                typer.echo(
                    json.dumps({"messages": messages}, indent=2, ensure_ascii=False),
                    err=True,
                )
            result = await node(
                {
                    "messages": messages,
                }
            )
            response_messages = (
                list(result.get("messages", []))
                if isinstance(result, dict)
                else list(result)
            )
            for assistant_message in response_messages:
                messages.append(assistant_message)
                content = assistant_message.get("content")
                if content:
                    typer.echo(content)
                else:
                    typer.echo(
                        json.dumps(assistant_message, indent=2, ensure_ascii=False)
                    )
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


# Create the main typer app
app = typer.Typer(help="SwarmX Command Line Interface")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            "-f",
            exists=True,
            help="The path to the swarmx file",
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
    version: Annotated[
        bool, typer.Option("--version", is_eager=True, help="Print the app version")
    ] = False,
):
    """SwarmX Command Line Interface."""
    if ctx.invoked_subcommand is not None:
        return
    if version:
        typer.echo(f"SwarmX v{__version__}")
        raise typer.Exit()
    asyncio.run(amain(file=file, output=output, verbose=verbose))


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
    auto_execute_tools: bool = True,
):
    """Start SwarmX as an OpenAI-compatible API server."""
    # Load swarm configuration
    if file is None:
        data = {}
    else:
        data = json.loads(file.read_text())

    # Create FastAPI app
    node = load_node(data)
    fastapi_app = create_server_app(node, auto_execute_tools=auto_execute_tools)

    # Start the server
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command()
def mcp(
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
    """Run the agent as an MCP server over stdio."""
    if file is None:
        data = {}
    else:
        data = json.loads(file.read_text())
    create_mcp_server(load_node(data)).run()


if __name__ == "__main__":
    app()
