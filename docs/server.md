# OpenAI-Compatible Server

SwarmX exposes an optional Node.js HTTP server through `createServer()` from
`@swarmx/core` and the `swarmx serve` CLI command.

The server provides:

- `GET /models` - lists agent nodes in the configured swarm as OpenAI-style models.
- `GET /sessions` - lists local SwarmX sessions.
- `POST /chat/completions` - accepts OpenAI-compatible chat completion requests with optional SSE streaming.

## Boundary Rules

The server is local-first by default:

- It binds to `127.0.0.1` unless a host is provided.
- Non-loopback hosts such as `0.0.0.0` require an API token before the server starts.
- Browser `Origin` headers are rejected unless explicitly allowlisted.
- Wildcard origins are rejected.
- `Origin: null` is rejected unless trusted desktop bridge mode is explicitly enabled.
- When an API token is configured, requests must include `Authorization: Bearer <token>`.

## Start From The CLI

Start a loopback server:

```shell
swarmx serve --port 8000
```

Start a non-loopback server with explicit bearer auth:

```shell
swarmx serve \
  --host 0.0.0.0 \
  --port 8000 \
  --api-token "$SWARMX_API_TOKEN" \
  --allowed-origin "http://localhost:5173"
```

Use `--config <path>` to load a `SwarmConfig` JSON file instead of the default
single-agent swarm.

## Embed In TypeScript

```ts
import { Agent, Swarm, createServer } from "@swarmx/core";

const agent = new Agent({
  name: "agent",
  instructions: "You are a helpful assistant.",
  model: "gpt-4o",
});

const swarm = new Swarm({
  name: "default",
  root: "agent",
  nodes: {
    agent: {
      kind: "agent",
      agent: {
        name: agent.name,
        instructions: agent.instructions,
        model: agent.model,
      },
    },
  },
  edges: [],
});

createServer(swarm, {
  host: "127.0.0.1",
  port: 8000,
});
```

`createServer()` starts listening immediately and returns the underlying
`http.Server` instance so callers can close it or attach lifecycle handling.

## Use With An OpenAI-Compatible Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="agent",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
```

For a bearer-protected server, pass the token as the OpenAI client API key:

```python
client = OpenAI(
    base_url="http://127.0.0.1:8000",
    api_key="your-swarmx-api-token",
)
```

## Streaming Semantics

When `stream: true` is supplied, SwarmX responds with Server-Sent Events:

1. Each request creates an independent stream.
2. Message chunks are yielded in the order produced by the root agent or swarm.
3. The stream ends with `data: [DONE]`.
4. If an error occurs after streaming starts, an error event is emitted before
   `data: [DONE]`.

## Request Shape

The chat endpoint accepts the subset used by SwarmX:

```json
{
  "model": "agent",
  "messages": [
    { "role": "user", "content": "Hello!" }
  ],
  "stream": false
}
```

The non-streaming response joins emitted SwarmX message chunks into one
assistant message. The streaming response emits OpenAI-style
`chat.completion.chunk` events for message chunks.
