# SwarmX

A lightweight, stateless multi-agent orchestration framework with a TypeScript core and Electron desktop app.

## Architecture

This workspace contains:

- **`packages/core/`** — Core library with agents, swarms, MCP tools, session persistence, and OpenAI-compatible server
- **`packages/cli/`** — Command-line interface (`swarmx` binary)
- **`packages/acp-server/`** — Agent Client Protocol server
- **`packages/desktop/`** — Electron desktop application with React 19 UI

## Quick Start

### Prerequisites

- Node.js >= 20
- pnpm >= 9

### Install & Build

```shell
pnpm install
pnpm build
```

### Configuration

Set your OpenAI-compatible API credentials via environment variables:

```shell
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional
export OPENAI_MODEL="gpt-4o"                         # optional, default: gpt-4o
```

Or configure per agent via the `client` field:

```typescript
new Agent({
  name: "my-agent",
  client: { apiKey: "sk-...", baseUrl: "https://api.openai.com/v1" },
})
```

### CLI

```shell
pnpm --filter @swarmx/cli swarmx
```

Start the OpenAI-compatible API server:

```shell
pnpm --filter @swarmx/cli swarmx serve --host 0.0.0.0 --port 8000
```

### Desktop App

```shell
pnpm --filter @swarmx/desktop dev
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

```typescript
import { Agent, Edge, Swarm, SwarmNode } from "@swarmx/core";

const agentA = new Agent({
  name: "agent_a",
  instructions: "You are a helpful agent.",
  model: "gpt-4o",
});

const agentB = new Agent({
  name: "agent_b",
  model: "deepseek-r1:7b",
  instructions: "You can only speak Chinese.",
});

const swarm = new Swarm({
  name: "demo",
  root: "agent_a",
  nodes: {
    agent_a: { kind: "agent", agent: { name: "agent_a", model: "gpt-4o", instructions: "You are helpful." } },
    agent_b: { kind: "agent", agent: { name: "agent_b", model: "deepseek", instructions: "Speak Chinese." } },
  },
  edges: [{ source: "agent_a", target: "agent_b" }],
});

const result = await swarm.execute({
  messages: [{ role: "user", content: "I want to talk to agent B." }],
});
```

## Project Structure

```
swarmx/
├── package.json                # Workspace root
├── packages/
│   ├── core/                   # Core orchestration library
│   ├── cli/                    # CLI binary (Commander)
│   ├── acp-server/             # ACP server implementation
│   └── desktop/                # Electron desktop app
│       ├── src/main/           # Main process + IPC
│       ├── src/preload/        # Context bridge API
│       └── src/renderer/       # React 19 SPA
└── docs/                       # Documentation
```

## Testing

```shell
pnpm test
```

## License

MIT
