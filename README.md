# SwarmX (forked from OpenAI's Swarm)

[![PyPI version](https://img.shields.io/pypi/v/swarmx)](https://pypi.org/project/swarm)
[![Python Version](https://img.shields.io/pypi/pyversions/swarmx)](https://pypi.org/project/swarmx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/swarmx)](https://pepy.tech/project/swarmx)
[![GitHub stars](https://img.shields.io/github/stars/tcztzy/swarmx.svg)](https://github.com/tcztzy/swarmx/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/tcztzy/swarmx.svg)](https://github.com/tcztzy/swarmx/network)
[![GitHub issues](https://img.shields.io/github/issues/tcztzy/swarmx.svg)](https://github.com/tcztzy/swarmx/issues)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/psf/black)

An extreme simple framework exploring ergonomic, lightweight multi-agent orchestration.

## Highlights
1. SwarmX is both Agent and Workflow
2. MCP servers support
3. OpenAI-compatible streaming-server
4. Workflow import/export in JSON format

![asciicast](./docs/demo.svg)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tcztzy/swarmx&type=Date)](https://www.star-history.com/#tcztzy/swarmx&Date)

## Quick start

After setting `OPENAI_API_KEY` environment variable, you can start a simple REPL by running the following command:

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

- `POST /v1/chat/completions` - Chat completions with streaming support
- `GET /v1/models` - List available models

Use it with any OpenAI-compatible client:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # SwarmX doesn't require authentication
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Installation

Requires Python 3.11+

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

## Architecture

```mermaid
graph TD
   classDef QA fill:#ffffff;
   classDef agent fill:#ffd8ac;
   classDef tool fill:#d3ecee;
   classDef result fill:#b4f2be;
   func1("transfer_to_weather_assistant()"):::tool
   Weather["Weather Assistant"]:::agent
   func2("get_weather('New York')"):::tool
   temp(64):::result
   A["It's 64 degrees in New York."]:::QA
   Q["What's the weather in ny?"]:::QA --> 
   Triage["Triage Agent"]:::agent --> Weather --> A
   Triage --> func1 --> Weather
   Weather --> func2 --> temp --> A
```

[1]: https://platform.openai.com/docs/api-reference/chat/create
