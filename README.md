# SwarmX (Rust Rewrite)

A lightweight, stateless multi-agent orchestration framework rewritten in Rust.

## Architecture

This workspace contains:

- **`crates/swarmx-core`** — Core library with agents, swarms, MCP tools, messages graph, and OpenAI-compatible server
- **`crates/swarmx-cli`** — Command-line interface (`swarmx` binary)
- **`apps/tauri`** — Desktop application built with Tauri
- **`apps/web`** — Web application built with Leptos

## Quick Start

### Prerequisites

- Rust 1.85+
- (Optional) `cargo-tauri` for desktop app development

### Environment Variables

Create a `.env` file in the project root:

```shell
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1  # optional
OPENAI_MODEL=gpt-4o                        # optional
```

### CLI

```shell
cargo run -p swarmx-cli
```

Start the OpenAI-compatible API server:

```shell
cargo run -p swarmx-cli -- serve --host 0.0.0.0 --port 8000
```

### Desktop App (Tauri)

```shell
cargo tauri dev
```

### Web App (Leptos)

```shell
cargo run -p swarmx-web
```

## Core Concepts

### Swarm vs Trace: Two Types of Graphs

**Swarm Graph (Your Workflow Blueprint)**
- A flowchart defining all possible paths agents can take
- Agents can loop back with conditions to prevent infinite loops
- Conditional routing using CEL expressions

**Trace Graph (What Actually Happened)**
- A record of one specific execution through your workflow
- Each agent run gets a unique ID
- Forms a tree structure for audit trails

### Usage Example

```rust
use swarmx_core::{Agent, Edge, Swarm};

let agent_a = Agent::new("agent_a")
    .with_instructions("You are a helpful agent.");

let agent_b = Agent::new("agent_b")
    .with_model("deepseek-r1:7b")
    .with_instructions("You can only speak Chinese.");

let swarm = Swarm::new("demo_swarm", "agent_a")
    .with_node(SwarmNode::Agent(agent_a))
    .with_node(SwarmNode::Agent(agent_b))
    .with_edge(Edge::new("agent_a", "agent_b"));

let result = swarm.execute(
    serde_json::json!({"messages": [{"role": "user", "content": "I want to talk to agent B."}]}),
    None,
).await?;
```

## Project Structure

```
swarmx/
├── Cargo.toml                  # Workspace definition
├── crates/
│   ├── swarmx-core/            # Core orchestration library
│   │   ├── src/
│   │   │   ├── agent.rs        # Agent node
│   │   │   ├── swarm.rs        # Swarm orchestrator
│   │   │   ├── edge.rs         # Graph edges
│   │   │   ├── node.rs         # Node trait
│   │   │   ├── messages.rs     # Message graph
│   │   │   ├── mcp.rs          # MCP manager
│   │   │   ├── server.rs       # Axum OpenAI-compatible server
│   │   │   └── ...
│   │   └── Cargo.toml
│   └── swarmx-cli/             # CLI binary
│       └── src/main.rs
├── apps/
│   ├── tauri/                  # Desktop app
│   │   └── src-tauri/
│   └── web/                    # Leptos web app
│       └── src/
└── python-legacy/              # Original Python codebase (archived)
```

## Testing

```shell
cargo test -p swarmx-core
```

## License

MIT
