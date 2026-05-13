# OpenAI-compatible Server

`swarmx-core::server` provides an Axum application exposing OpenAI-compatible endpoints:

- **GET /models** – lists all agents in the swarm as models.
- **POST /chat/completions** – handles chat requests with optional streaming.

## Streaming semantics

The server forwards chunk payloads emitted by the underlying agent stream. Important points:

1. **Per-request stream** – each request creates its own independent stream.
2. **Chunk ordering** – chunks are yielded to the client in the order they are produced by the agent.
3. **Termination** – the stream always ends with `data: [DONE]`, even if an error occurs (an error chunk is emitted first).

## Usage example

Start the server via CLI:

```bash
cargo run -p swarmx-cli -- serve --host 0.0.0.0 --port 8000
```

Or embed the server in your own application:

```rust
use std::sync::Arc;
use swarmx_core::{Swarm, server::create_server_app};

let swarm = Arc::new(Swarm::new("my_swarm", "root"));
let app = create_server_app(swarm, true);

let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await?;
axum::serve(listener, app).await?;
```

Connect with any OpenAI-compatible client:

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

The above script will print the assistant's reply as it arrives, respecting the ordering guarantees described.

## Server State

The server holds an `AppState` containing:
- `swarm: Arc<Swarm>` – the swarm to route requests through
- `auto_execute_tools: bool` – whether to auto-execute tool calls

Each request is routed to the model name specified in the request; if the model matches a node in the swarm, that node handles the completion.
