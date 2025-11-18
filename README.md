# SwarmX (forked from OpenAI's Swarm)

[![PyPI version](https://img.shields.io/pypi/v/swarmx)](https://pypi.org/project/swarm)
[![Python Version](https://img.shields.io/pypi/pyversions/swarmx)](https://pypi.org/project/swarmx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/psf/black)

An extreme simple framework exploring ergonomic, lightweight multi-agent orchestration.

## Highlights
1. SwarmX is both Agent and Workflow
2. MCP servers support
3. OpenAI-compatible streaming-server
4. Workflow import/export in JSON format

![asciicast](./docs/demo.svg)

## Quick start

SwarmX automatically loads environment variables from a `.env` file if present. You can either:

1. **Use a .env file** (recommended):
   ```shell
   # Create a .env file in your project directory
   echo "OPENAI_API_KEY=your-api-key" > .env
   echo "OPENAI_BASE_URL=http://localhost:11434/v1" >> .env  # optional
   uvx swarmx  # Start interactive REPL
   ```

2. **Set environment variables manually**:
   ```shell
   export OPENAI_API_KEY="your-api-key"
   # export OPENAI_BASE_URL="http://localhost:11434/v1"  # optional
   uvx swarmx  # Start interactive REPL
   ```

### API Server

You can also start SwarmX as an OpenAI-compatible API server:

```shell
uvx swarmx serve --host 0.0.0.0 --port 8000
```

This provides OpenAI-compatible endpoints:

- `POST /chat/completions` - Chat completions with streaming support
- `GET /models` - List available models

Use it with any OpenAI-compatible client:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000",
    api_key="dummy"  # SwarmX doesn't require authentication
)

response = client.chat.completions.create(
    model="agent-created-by-yourself",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Installation

Requires Python 3.12+

```console
$ pip install swarmx # or `uv tool install swarmx`
```

## Usage

```python
import asyncio
from swarmx import Swarm, Agent

client = Swarm()

def transfer_to_agent_b():
    return agent_b


agent_a = Agent(
    name="Agent A",
    instructions="You are a helpful agent.",
    functions=[transfer_to_agent_b],
)

agent_b = Agent(
    name="Agent B",
    model="deepseek-r1:7b",
    instructions="你只能说中文。",  # You can only speak Chinese.
)


async def main():
    response = await client.run(
        agent=agent_a,
        messages=[{"role": "user", "content": "I want to talk to agent B."}],
    )

    print(response.messages[-1]["content"])


asyncio.run(main())
```

### Advanced Usage Examples

**Dynamic Tool Selection:**
```python
# Based on conversation topic, show only relevant tools
if "weather" in user_message:
    context = {"tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]}
elif "search" in user_message:
    context = {"tools": [{"type": "function", "function": {"name": "search_web", "description": "Search web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]}
```

[1]: https://platform.openai.com/docs/api-reference/chat/create
