# SwarmX (forked from OpenAI's Swarm)

[![PyPI version](https://img.shields.io/pypi/v/swarmx)](https://pypi.org/project/swarmx/)
[![Python Version](https://img.shields.io/pypi/pyversions/swarmx)](https://pypi.org/project/swarmx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

An extreme simple framework exploring ergonomic, lightweight multi-agent orchestration.

## Highlights
1. SwarmX is both Agent and Workflow
2. MCP servers support
3. OpenAI-compatible streaming-server
4. Workflow import/export in JSON format

## Core Concepts

### Swarm vs Trace: Two Types of Graphs

SwarmX uses two distinct graph structures to separate workflow design from execution history:

**🔷 Swarm Graph (Your Workflow Blueprint)**
- Think of it as a flowchart that defines all possible paths your agents can take
- Agents can loop back to previous steps when needed (with conditions to prevent infinite loops)
- Smart routing: decisions can be made using simple conditions like "if score > 0.5 go to agent_a, else agent_b"
- One agent can connect to multiple others based on different conditions

**🔸 Trace Graph (What Actually Happened)**
- A record of one specific execution through your workflow
- Like a trail of breadcrumbs showing exactly which path was taken
- Each time an agent runs, it gets a unique ID (even if it's the same agent running multiple times)
- Forms a tree structure - no merging paths, just a clear sequence of what happened when

**Why This Matters:**
- Your workflow (Swarm) stays clean and reusable - define once, run many times
- Each execution (Trace) gives you a complete audit trail for debugging and analysis
- The same workflow can produce different traces based on runtime conditions

![asciicast](./docs/demo.svg)

## Quick start

SwarmX automatically loads environment variables from a `.env` file if present. You can either:

1. **Use a .env file** (recommended):
   ```shell
   # Create a .env file in your project directory
   # Local OpenAI-compatible servers accept any non-empty API key.
   echo "OPENAI_API_KEY=dummy" > .env
   echo "OPENAI_BASE_URL=http://localhost:11434/v1" >> .env  # optional
   uvx swarmx  # Start interactive REPL
   ```

2. **Set environment variables manually**:
   ```shell
   export OPENAI_API_KEY="dummy"  # any non-empty value for local servers
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
from swarmx import Agent, Edge, Swarm

agent_a = Agent(
    name="agent_a",
    instructions="You are a helpful agent.",
)

agent_b = Agent(
    name="agent_b",
    model="deepseek-r1:7b",
    instructions="你只能说中文。",  # You can only speak Chinese.
)

swarm = Swarm(
    name="demo_swarm",
    parameters={},
    nodes={agent_a.name: agent_a, agent_b.name: agent_b},
    edges=[Edge(source=agent_a.name, target=agent_b.name)],
    root=agent_a.name,
)


async def main():
    messages = await swarm(
        {"messages": [{"role": "user", "content": "I want to talk to agent B."}]},
    )

    print(messages[-1]["content"])


asyncio.run(main())
```

### Advanced Usage Examples

**Dynamic Tool Selection:**
```python
# Based on conversation topic, include only relevant tools
arguments = {"messages": messages}
if "weather" in user_message:
    arguments["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
elif "search" in user_message:
    arguments["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
response = await agent(arguments)
```

[1]: https://platform.openai.com/docs/api-reference/chat/create
