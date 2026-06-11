# Design Notes

Supplemental guidance for coding agents. Use this alongside `AGENTS.md` for deeper architectural context.

## Agent-Specific Notes
- **Design principle:** Build focused, single-purpose agents; avoid overloading a single agent with unrelated concerns.
- **Hooks:** Implement cross-cutting behavior with `on_start`, `on_end`, `on_handoff`, or `on_chunk` hooks rather than embedding ad hoc logic in core flows.
- **Routing:** Use edge-based transfers for routing between agents; keep routing logic explicit via CEL expressions on edges.
- **Tools:** Expose tool capabilities explicitly so routing and orchestration can select them dynamically.
- **MCP integration:** Configure MCP servers via environment variables before use; interact through `packages/core/src/mcp.ts`. Validate tool schemas and authentication flows when adding or modifying MCP integrations.

## Graph Architecture
- **Swarm Graph (workflow definition):**
  - Structure: `petgraph::DiGraph` DAG of agent nodes with conditional transitions.
  - Edge targets: agent names, CEL expressions (e.g., `"score > 0.5 ? 'agent_b' : 'agent_c'"`), or MCP tool calls that compute destinations.
  - Cycle rule: Any cycle must include at least one conditional edge that can break the loop; unconditional cycles are invalid.
  - Validation: Enforce DAG property after accounting for conditional escape edges; reject graphs that would create unavoidable cycles.
- **Messages Graph (conversation DAG):**
  - Structure: `petgraph::DiGraph` DAG of message nodes, wrapping a `Vec<ChatCompletionRequestMessage>` so that agents can treat it like a simple sequence while the framework maintains a richer graph.
  - Default shape: a straight single chain built from the initial vector (numeric IDs `0..N-1` for external messages, connected in order).
  - Context rewriting: some agents may rewrite context (including the user query); downstream sub-agents consume only from the rewritten segment, so the "main" conversation line may not be fully connected.
  - Branches (Git-style):
    - Each branch has: `name` (e.g. `main`, `origin`), `description` (e.g. "main thread of conversation", "isolated conversation history for [specific task]"), and `start` / `stop` message IDs.
    - `origin` stores the full, uncompressed history: `start` is always the first message, `stop` tracks the latest message in the raw log.
    - `main` is the active working branch for LLM calls; its `start` can move forward when history is compressed.
  - Message IDs:
    - External messages entering the system get numeric IDs (`0..N-1`) in insertion order.
    - New LLM messages use a UUID derived from the underlying completion request.
    - New tool messages use the tool call identifier as their node ID.
  - Per-message metadata:
    - Every node stores the `ChatCompletionRequestMessage` as `message`.
    - LLM-generated nodes additionally carry the raw completion response as `completion`.
    - Tool-generated nodes additionally carry the raw `CallToolResult` as `result`.
  - Purpose:
    - Agents remain "atomic": they see `Messages` as just an iterable of messages.
    - Only swarms' queen can see the `Messages` graph and handle with it in **Context Engineering** ways (such as Retrieve/Compress/Isolated/Remember)
    - The framework uses the Messages DAG to track which agent/tool produced which message, on which branch, with which raw completion/result, providing an execution trace over the conversation itself.

## UI Architecture (desktop)

### Technology Stack

The desktop GUI uses Electron with React 19 for the renderer:

| Layer | Technology | Version | Role |
|-------|------------|---------|------|
| Shell | Electron | 33 | Cross-platform desktop shell |
| Renderer | React | 19 | UI framework (functional components + hooks) |
| Build | electron-vite | — | Build toolchain for Electron |
| Validation | zod | 3.24 | Schema validation |
| ACP transport | @agentclientprotocol/sdk | 0.22 | TypeScript SDK for Agent Client Protocol |
| Lint/Format | Biome | 1.9 | Linter + formatter |

### Architecture: Main / Preload / Renderer

Electron security model with `contextIsolation: true`, `nodeIntegration: false`:

```
┌──────────────────────────────────────────────┐
│ Main Process (Node.js)                       │
│ ├── IPC handlers (swarm.execute, sessions)   │
│ └── BrowserWindow management                 │
└──────────────┬───────────────────────────────┘
               │ contextBridge
               ▼
┌──────────────────────────────────────────────┐
│ Preload (contextBridge API)                  │
│ └── window.swarmxAPI { sendMessage, ... }   │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ Renderer (React 19 SPA)                      │
│ ├── App component (state management)         │
│ ├── Sidebar (session list, harness selector) │
│ ├── ChatArea (message bubbles, input)        │
│ └── Settings (providers, instances)          │
└──────────────────────────────────────────────┘
```

### Message Data Model

Five message types for rendering agent responses:

| Kind | Render As | Icon | Initial State |
|------|-----------|------|---------------|
| `Message` — user | Right-aligned tinted bubble | — | — |
| `Message` — assistant | Markdown with copy button | — | — |
| `Thinking` | Collapsible section | Brain | Collapsed |
| `ToolCall` | Card with collapsible | Wrench + badge | Expanded |
| `ToolResult` | Muted container | FileText | — |

### Session Persistence

Sessions stored as individual JSON files: `~/.swarmx/sessions/{id}.json`
Each file is a serialized `Session` with:
- Full message history (`ChatMessage[]`)
- Agent metadata (label, command line, backend index)
- ACP session ID for resume support
- Created/updated timestamps

Legacy `~/.swarmx/sessions.json` (single file) is auto-migrated on startup.

### ACP Integration in Desktop

The desktop app acts as an **ACP client** — it starts agents as subprocesses and communicates via JSON-RPC 2.0 over stdio:

1. App startup: scan for available agents (bun, opencode, hermes, openclaw) via `core/src/harness.ts`
2. Session creation: run agent process, send `initialize` + `session/new`
3. User prompt: IPC handler calls `swarm.execute()`, which spawns ACP subprocess
4. Response handling: parse `session/update` notifications, map `kind` field to message variant
5. Error handling: catch transport errors, surface in UI

### Agent Runtime Detection

`core/src/harness.ts` defines known agent backends:
- **SwarmX** — native TypeScript engine with OpenAI SDK, no subprocess needed
- **Bun** (for Claude Code, Codex ACP adapters) — checked via `bun --version`
- **OpenCode** — checked via `opencode --version`
- **Hermes** — checked via `hermes --version`
- **OpenClaw** — checked via `openclaw --version`

Missing deps show an install banner. After installation, agent registry is rebuilt to include newly available agents.
