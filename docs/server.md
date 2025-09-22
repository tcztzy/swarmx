# OpenAI‑compatible Server

`swarmx.server` provides a FastAPI application exposing OpenAI‑compatible endpoints:

- **GET /v1/models** – lists all agents in the swarm as models.
- **POST /v1/chat/completions** – handles chat requests with optional streaming.

## Streaming semantics

The server streams responses using the same chunk format as the internal `Agent._run_stream` implementation. Important points:

1. **Multiple concurrent streams** – each request creates its own independent stream. Streams are ordered internally; the client receives chunks in the order they are produced.
2. **Chunk IDs** – every chunk carries an `id` that uniquely identifies the message it belongs to. When merging chunks into a final message, the order of the resulting message follows the order of chunks **with the same `id`** as they appear in the stream.
3. **First chunk of each stream** – the first chunk for a given `id` is always emitted before any subsequent chunks for that `id`, guaranteeing that the message can be reconstructed incrementally.

This behavior mirrors the OpenAI API contract, allowing clients to consume streamed data safely while the server internally aggregates chunks into complete messages.

## Usage example

```bash
uvx swarmx serve --host 0.0.0.0 --port 8000
```

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content, end="")
```

The above script will print the assistant’s reply as it arrives, respecting the ordering guarantees described.
