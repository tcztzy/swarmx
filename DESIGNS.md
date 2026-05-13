# Design Notes

Supplemental guidance for coding agents. Use this alongside `AGENTS.md` for deeper architectural context.

## Agent-Specific Notes
- **Design principle:** Build focused, single-purpose agents; avoid overloading a single agent with unrelated concerns.
- **Hooks:** Implement cross-cutting behavior with `on_start`, `on_end`, `on_handoff`, or `on_chunk` hooks rather than embedding ad hoc logic in core flows.
- **Routing:** Use edge-based transfers for routing between agents; keep routing logic explicit via CEL expressions on edges.
- **Tools:** Expose tool capabilities explicitly so routing and orchestration can select them dynamically.
- **MCP integration:** Configure MCP servers via environment variables before use; interact through `crates/swarmx-core/src/mcp.rs`. Validate tool schemas and authentication flows when adding or modifying MCP integrations.

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

## UI Architecture (swarmx-ui)

### Technology Stack

The desktop GUI uses a pure Rust stack targeting native rendering:

| Layer | Crate | Version | Role |
|-------|-------|---------|------|
| GUI framework | `iced` | 0.14 | Cross-platform native GUI (wgpu + tiny-skia fallback) |
| Widget library | `iced-shadcn` | 0.5 | shadcn-inspired components (Button, Card, Input, Badge, Collapsible, ScrollArea, Spinner, Tooltip, etc.) |
| Icons | `lucide-icons` | 0.575 | 850+ icons via `Icon` enum + embedded font |
| ACP transport | `agent-client-protocol` | 0.11.1 | Rust SDK for Agent Client Protocol (SACP types + tokio) |
| File dialogs | `rfd` | 0.15 | Native file/folder picker dialogs |

### Architecture Pattern: MVU (Model-View-Update)

Iced enforces strict unidirectional data flow, same pattern as Elm/The Elm Architecture:

```
┌──────────────────────────────────────────────────────┐
│ App (State)                                          │
│ ├── agents, sessions, active_session                 │
│ ├── input, loading, error, theme                     │
│ ├── thinking_expanded, tool_expanded (HashSets)      │
│ └── md_cache (HashMap<session_id, Vec<Content>>)     │
└───────────┬──────────────────────────┬───────────────┘
            │                          │
            ▼                          ▼
    ┌───────────┐              ┌────────────┐
    │ update()  │◄─────────────│  Message   │
    │ returns   │   enum       │  (events)  │
    │ Task<Msg> │              └────────────┘
    └─────┬─────┘                     ▲
          │                           │
          ▼                           │
    ┌───────────┐              ┌────────────┐
    │  view()   │──────────────│  on_press  │
    │ returns   │   user taps  │  callbacks │
    │ Element   │              └────────────┘
    └───────────┘
```

### Component Patterns

**Props builder pattern** — all iced-shadcn components use:
```rust
ComponentProps::new()
    .variant(ComponentVariant::Xxx)
    .size(ComponentSize::Xxx)
    .optional_setting(value)
```

**Theme injection** — every component function takes `&Theme` as the last arg:
```rust
button("Label", Some(msg), ButtonProps::new().variant(v).size(s), &app.theme)
```

**Collapsible state** — tracked in `HashSet<usize>` in App state, toggled via Message enum. This keeps expanded/collapsed state in the model, consistent with MVU.

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
Each file is a serialized `Session` struct with:
- Full message history (`Vec<ChatMessage>`)
- Agent metadata (label, command line, backend index)
- ACP session ID for resume support
- Created/updated timestamps

Legacy `~/.swarmx/sessions.json` (single file) is auto-migrated on startup.

### ACP Integration in GUI

The GUI acts as an **ACP client** — it starts agents as subprocesses and communicates via JSON-RPC 2.0 over stdio:

1. App startup: scan for available agents (bun, opencode, hermes, openclaw) via `environment.rs`
2. Session creation: run agent process, send `initialize` + `session/new`
3. User prompt: send `session/prompt` with full message history
4. Response handling: parse `session/update` notifications, map `kind` field to `ChatMessage` variant
5. Error handling: catch transport errors, display in `t.palette.destructive` color

### Agent Runtime Detection

`environment.rs` scans the system for agent dependencies:
- **Bun** (for Claude Code, Codex ACP adapters) — checked via `bun --version`
- **OpenCode** — checked via `opencode --version`
- **Hermes** — checked via `hermes --version`
- **OpenClaw** — checked via `openclaw --version`
- **Python** (for SwarmX backend) — checked via `python3 --version`

Missing deps show an install banner. After installation, agent registry is rebuilt to include newly available agents.
