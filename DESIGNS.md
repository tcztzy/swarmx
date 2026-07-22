# Design Notes

Supplemental guidance for coding agents. Use this alongside `AGENTS.md` for deeper architectural context.

## Agent-Specific Notes
- **Design principle:** Build focused, single-purpose agents; avoid overloading a single agent with unrelated concerns.
- **Hooks:** Implement cross-cutting behavior with `on_start`, `on_end`, `on_handoff`, or `on_chunk` hooks rather than embedding ad hoc logic in core flows.
- **Routing:** Use edge-based transfers for routing between agents; keep routing logic explicit via CEL expressions on edges.
- **Tools:** Expose tool capabilities explicitly so routing and orchestration can select them dynamically.
- **MCP integration:** Configure MCP servers via environment variables before use; interact through `packages/core/src/mcp.ts`. Validate tool schemas and authentication flows when adding or modifying MCP integrations.

## Model, Supply, and Agent Identity

- **Composition boundary:** an Extension distributes Software, Skills, MCP
  servers, and other components. A Harness is a reproducible recipe composed as
  `Software + Skills + MCP servers + project context + policies + ...`. An Agent
  is one Harness paired with one Model. Extension membership is provenance, not
  Agent identity.
- **Agent-specific Skills:** one logical Skill may own multiple immutable
  variants targeted to Agent/Model/Harness capabilities. Deterministic
  resolution and explicit `off | auto | required` bindings prevent unnecessary
  Skill text from consuming every Model's context.
- **Model is primary:** `Model` owns its stable id, label, runtime model name,
  supported API protocols, and capability references. It is never nested under
  or owned by a Provider.
- **Provider is a supply label:** a Provider profile contains only connection,
  API-label, base-URL, authentication-mode, secret-reference, and
  runtime-readiness metadata. Provider profiles cannot define Model catalogs, a
  default Model, or Harness-specific Model overrides.
- **Many-to-many routing:** `ModelSupply` explicitly links one `modelId` to one
  `providerProfileId`. Multiple Providers may supply one Model and one Provider
  may supply multiple Models. Route aliases and yallm bridge metadata belong on
  this join record.
- **Agent identity:** every resolved Agent id is `harnessId:modelId`. Effort,
  internal ModelSupply/Provider routing, runtime aliases, and bridge routes do
  not create a new Model or Agent identity.
- **Native Agent definitions:** Claude Code Markdown and Codex TOML are import
  and projection formats around the canonical Agent profile. Desktop discovery
  is read-only and limited to user/project native Agent directories. It keeps
  native unknown fields inert, preserves `inherit` as unresolved, namespaces
  profiles by host, and never activates host hooks, MCP servers, or sessions.
- **Compatibility:** Harness compatibility depends only on Harness model-control
  and API metadata plus Model API capabilities. Provider readiness may annotate
  a supply route, but cannot add or remove a Harness x Model pair.
- **Native direct execution:** the built-in SwarmX Harness directly implements
  Anthropic Messages, OpenAI Responses, and OpenAI Chat Completions. The
  resolved API protocol is carried into the request-scoped Agent and retains
  native streaming, reasoning, cancellation, and MCP tool-call continuation.
- **Native-first routing:** among APIs already compatible with the Harness x
  Model pair, route selection prefers the chosen ModelSupply's Provider kind or
  declared native entrypoint. yallm is an explicit or last-resort compatibility
  bridge, never a normalization layer for a route that can execute natively.
- **Dynamic catalog:** desktop list sources are Provider API discovery,
  extension-declared Models, and manually persisted Models. The built-in
  registry only enriches matching ids with verified API/capability metadata; it
  does not display undiscovered Models by itself.
- **Discovery normalization:** Provider responses become independent Model
  records plus internal ModelSupply links. A Provider is the connection that
  observed/supplies a Model, never its owner or identity namespace.
- **Composer surface:** ordinary desktop composition exposes exactly three
  choices: Harness, Model, and Effort. It never asks the user to select a
  Provider or ModelSupply; trusted core code owns those launch details.
- **Provider configuration:** desktop Provider CRUD lives under the lower-left
  account menu's dedicated Settings workspace, never inside Agent composition.
  A user supplies a Base URL and API key or auth token; saving updates
  connection metadata and encrypted auth state together without exposing a
  Supply selector. A normal Custom Provider keeps one credential and treats its
  Base URL as the exact API root; OpenAI-compatible discovery appends `/models`.
  URL or response shape never opts the connection into New API account logic.
- **No ambient discovery:** desktop never scans known environment variables to
  synthesize Provider connections or start discovery. Connections must come
  from explicit encrypted Settings or extension metadata; an extension env
  secret reference may satisfy that declared connection but cannot create one.
- **Integrated usage:** Settings opens one fixed-column Provider matrix.
  Normalized balance and rate-limit meters render beside their connection, and
  the signed-in local Codex account appears as an OpenAI Provider peer instead
  of a separate tool-account group. This presentation identity does not change
  core Model, ModelSupply, or Agent identity.
- **Usage adapter boundary:** only the Electron main process resolves secrets
  and contacts adapter-owned official HTTPS endpoints, or the configured HTTPS
  origin at `/api/usage/token/` after an explicit New API selection. An
  explicitly connected New API account credential may additionally call that
  origin's `/api/status`, `/api/user/self`, and bounded paginated `/api/token/`
  endpoints. Requests
  refuse redirects, bound time and streamed bytes, validate vendor bodies, and
  return only sanitized meters/status through IPC. Only id-bound
  desktop-managed keychain Providers are eligible; ambient env and extension
  metadata cannot trigger usage requests. Codex uses the official local
  app-server protocol rather than auth-file or browser-session parsing.
- **Unsupported usage:** Anthropic/Claude Code, Gemini, and OpenCode Zen are
  displayed honestly when their configured credential has no official quota
  query. SwarmX does not scrape consoles or collect browser cookies/private
  OAuth state to fill those cards.
- **OpenCode Go key pool:** only the exact official `/zen/go` or `/zen/go/v1`
  connection may declare secret-free key slots backed by separately encrypted
  credentials. Main records normalized per-key request/token/cooldown state
  locally and calls no undocumented usage endpoint. A request changes key only
  for explicit quota/balance exhaustion before any output or tool event; generic
  rate limiting and post-output failures are returned without replay. Reset
  metadata wins, otherwise the failed key receives a bounded five-hour cooldown.
- **Account footer:** the persistent anonymous-user popover contains only
  Settings. The account row has no update affordance until npm
  reports a newer stable desktop package; it then shows the Codex-style
  circle-to-`Update` hover control and keeps progress/install/restart feedback in
  that same slot.
- **Desktop update boundary:** read-only npm checks are automatic and silent;
  download is explicitly user-triggered, SHA-512 verified, installed without
  lifecycle scripts into a versioned root, version-checked, and relaunched only
  for an Electron default-app/npm launch. Packaged and embedded hosts retain
  deployment ownership and never expose this control.
- **Secret boundary:** desktop settings contain only a `local_keychain`
  reference. Electron `safeStorage` ciphertext lives in the mode-`0600`
  Provider auth document; only the main process resolves it for discovery and
  request-scoped Harness execution, with no plaintext fallback.
- **Multi-protocol connection:** the canonical DeepSeek origin declares native
  Anthropic Messages and OpenAI Chat Completions entrypoints behind one secret.
  Anthropic is preferred unless the user explicitly selects OpenAI Chat; native
  routing does not require a bridge or duplicate Provider.
- **New API account boundary:** a Provider may hold one separately encrypted,
  explicitly entered account access token plus non-secret user id. The account
  wallet is displayed once; masked API-token limits remain individual and are
  never summed as a Provider total.
- **Runtime control:** direct Harnesses receive the chosen Model through scoped
  API variables; ACP Harnesses apply stable session `configOptions` for Model
  and then effort. A session matrix cell requires either a fixed adapter runtime
  model or a ModelSupply with an explicit adapter id and runtime model; `any`
  API compatibility alone never creates a cell.
- **Fixed pair routes:** built-in Harness x Model overrides are code-owned and
  deterministic. In particular, `claude_code:deepseek-v4-pro` requires
  `DEEPSEEK_API_KEY`, uses `https://api.deepseek.com/anthropic`, maps the main,
  Opus, and Sonnet aliases to `deepseek-v4-pro[1m]`, maps Haiku and sub-Agents
  to `deepseek-v4-flash`, and forwards the selected Effort through
  `CLAUDE_CODE_EFFORT_LEVEL`. Custom Harness identities carry the selected
  Software adapter id separately so they reuse this bootstrap without losing
  their own Agent identity.
- **Protected routing:** container launches allowlist scoped Model/bootstrap
  variables, including `CODEX_CONFIG` but not the adapter-ignored
  `CODEX_ACCESS_TOKEN`. Protected Codex reuses only the private official
  `auth.json` through an exact read-only file volume (or the adapter's supported
  `CODEX_API_KEY`/`OPENAI_API_KEY`). Loopback yallm URLs are translated to the
  container host bridge without changing Model identity or ModelSupply
  membership.
- **Failure policy:** Provider refreshes are independently addressable, bounded,
  and failure-isolated. The last successful row data and catalog cache remain
  usable, and desktop restart or Renderer remount reads the Model catalog cache
  without automatically calling Provider Models APIs. Model discovery runs only
  from explicit refresh; Provider create/update does not query models, and
  manual Models persist in desktop settings. Neither cache nor settings stores
  resolved secret values.
  Cache entries stay partitioned by Provider and retain optional New API
  `owned_by` group metadata. The Model menu presents Provider/group hierarchy
  while carrying the chosen ModelSupply id internally. Codex model discovery
  uses official app-server `model/list`; every route remains available to
  direct SwarmX, while only runtime ids also proven by pinned Codex ACP are
  Codex-Harness-scoped. Discovery never parses or repurposes Codex
  authentication tokens.

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
│ ├── Sidebar (Projects, tasks, harness)        │
│ ├── ChatArea (message bubbles, input)        │
│ └── Settings                                 │
│     ├── Providers                            │
│     ├── Extensions                           │
│     ├── Custom Agents (Harness + Model)      │
│     └── Runtime                              │
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
- Optional local Project identity and canonical working directory (`projectId`, `cwd`)
- ACP session ID for resume support
- Created/updated timestamps

Legacy `~/.swarmx/sessions.json` (single file) is auto-migrated on startup.

Desktop Projects are local folder bookmarks stored in
`~/.swarmx/projects.json`. They group tasks and provide the working directory
for agent execution and workspace reads; they are not a remote collaboration or
authorization domain. Pin and display-name changes are persisted in that
registry. Removing a Project records a local dismissal so startup does not add
the same working directory back automatically; explicitly adding the folder
restores it. Archiving tasks marks matching session records as archived without
deleting their history or the Project directory.

The Projects heading overflow controls only collection presentation: grouped or
flat tasks and priority, last-updated, or stable source ordering. Folder actions
belong to the overflow on each Project row, so commands always target the row
that opened the menu rather than whichever Project happens to be active.

### ACP Integration in Desktop

The desktop app acts as an **ACP client** — it starts agents as subprocesses and communicates via JSON-RPC 2.0 over stdio:

1. App startup: configure GUI-safe PATH entries and detect runtime tools plus the protected container backend via the shared `@swarmx/runtime` host service
2. Session creation: run agent process, send `initialize` + `session/new`
3. User prompt: IPC handler wraps protected built-in ACP harness backends when required, then calls `swarm.execute()`, which spawns the ACP subprocess
4. Response handling: parse `session/update` notifications, map `kind` field to message variant
5. Error handling: catch transport errors, surface in UI

### Agent Runtime Detection

`core/src/harness.ts` defines known agent backends:
- **SwarmX** — native TypeScript engine with OpenAI SDK, no subprocess needed
- **Node.js** — standalone baseline runtime for npm/npx package management and ACP adapters
- **Apple Container** — preferred protected backend on supported macOS, checked via `container --version` and `container system status`
- **Claude Code** — independently checked via `claude --version`
- **Codex** — independently checked via `codex --version`
- **Pi** — native `pi --version`; ACP uses pinned `pi-acp@0.0.31`, which bridges to Pi RPC
- **OpenCode** — native fallback checked via `opencode --version`
- **Hermes** — native fallback checked via `hermes --version`
- **OpenClaw** — native fallback checked via `openclaw --version`

The shared Doctor service performs the host-side check for both CLI and desktop. Inspection is read-only. A repair request first returns a deterministic risk-labelled plan and executes only after explicit confirmation, then redetects service readiness. On macOS, Apple Container is preferred over Docker; a confirmed repair verifies Apple silicon and a supported macOS version, may install the signed `apple/container` package, and may start `container system start`. Hermes detection prefers `~/.hermes/hermes-agent` and never clones, fetches, pulls, or updates it. Core harness-management and dependency primitives remain side-effect free; host adapters alone may execute installers or start services after confirmation.

Protected mode is the default for external built-in ACP harnesses. Claude Code and Codex adapters use pinned `npx` packages on the host and a Node 22 image in Apple Container; Bun is not installed or required. Pi uses pinned `pi-acp@0.0.31` but remains native because that adapter launches the user's Pi RPC process, performs filesystem/terminal work locally, and relies on Pi-owned auth, settings, extensions, and sessions. Pi does not inherit SwarmX direct-tool permissions or client MCP configuration. The desktop main process wraps protected backends as ordinary `AgentBackend` custom commands using `container run --rm -i --init`, a deliberate workspace bind mount, selected environment passthrough keys, and resource limits. Harness Software health appears inline in Settings → Custom Agents, while Runtime presents Node.js, the complete Harness tool inventory, the embedded Doctor, and container dependencies. The conversation list has no permanent Doctor indicator. If protected mode is required but Apple Container is missing, unsupported, or stopped, the send path opens the transient Doctor panel for that harness instead of silently falling back to native execution. `/doctor`, `/doctor --fix`, and `/setup` share the same read-only/confirmed-repair APIs; repair remains explicitly confirmed.
