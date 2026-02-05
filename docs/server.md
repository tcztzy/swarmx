# OpenAI‑compatible Server

`swarmx.server` provides a FastAPI application exposing OpenAI‑compatible endpoints:

- **GET /models** – lists all agents in the swarm as models.
- **POST /chat/completions** – handles chat requests with optional streaming.

## Streaming semantics

The server forwards chunk payloads emitted by the underlying agent stream. Important points:

1. **Per-request stream** – each request creates its own independent stream.
2. **Chunk ordering** – chunks are yielded to the client in the order they are produced by the agent.
3. **Termination** – the stream always ends with `data: [DONE]`, even if an error occurs (an error chunk is emitted first).

## Usage example

```bash
uvx swarmx serve --host 0.0.0.0 --port 8000
```

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000", api_key="dummy")
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content, end="")
```

The above script will print the assistant’s reply as it arrives, respecting the ordering guarantees described.
