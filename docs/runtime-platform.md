# Agent platform

## Local development

`pnpm dev` uses `tsx watch` to rebuild
Host and library TypeScript and restart Electron on backend edits. Renderer files are excluded from
that watcher: Vite serves the renderer and HMR on a local development server. Electron loads its
URL only in explicit development mode; packaged builds load the static renderer. Product operations
use the sandboxed preload's Electron IPC bridge in both modes. Backend restarts shut down the Host
and active native work before exit.

Renderer-only HMR preserves state where React Fast Refresh supports it; changing component exports
can require a page reload. Backend restarts recreate the window and end live runs. Persisted sessions,
permission grants and saved evidence survive, but unsent drafts may not.

## Native integrations

| Agent | Interface | Setup |
| --- | --- | --- |
| External ACP | Operator-selected stdio subprocess | Absolute `SWARMX_ACP_AGENT` descriptor and explicit `acp` Host grant; see `acp.md` |
| Codex (default) | Installed Codex App Server | Native login/config; `CODEX_PATH` can select the executable |
| Claude | Packaged `@anthropic-ai/claude-agent-sdk` | Native authentication and Claude settings |
| Hermes | Installed native TUI Gateway over stdio | `SWARMX_HERMES_PYTHON` selects its Python interpreter; otherwise use the installed `hermes` interpreter |
| OpenClaw | Official Gateway Client | `OPENCLAW_GATEWAY_URL` (default `ws://127.0.0.1:18789`), `OPENCLAW_GATEWAY_TOKEN` or `OPENCLAW_GATEWAY_PASSWORD` |
| DSH | Official SDK Client and matching runtime | Native SDK profile/authentication; one independent execution per task |

`--agent` → `SWARMX_AGENT` → configured `SWARMX_ACP_AGENT` (`acp`) → `codex`. The desktop can select any configured Agent. Only a selected
integration loads; errors are reported without fallback. ZCode and Kimi are not registered.
Host startup does not connect to a native CLI. Bootstrap reports session-list failures visibly
while keeping research and settings available. A later explicit native request may reconnect.

Native runtimes are Host-owned: one lazy instance per harness serves tasks in the execution
directory with the Host's permission ceiling and a scoped product-tool credential. The Host
disposes the runtimes at shutdown.

The retired `pi` value remains readable in historical settings and execution records but
cannot launch a builtin runtime. Configure a standalone ACP agent to use Pi or any other
framework. SwarmX does not copy credentials or migrate native session IDs. Validate an
existing native store explicitly with that external agent before resuming its sessions.
The historical adapter and tests remain under `examples/legacy-pi/`; SwarmX production source
and direct dependencies no longer import the Pi SDK. DSH still brings `pi-ai` transitively
through its own upstream backend; see `native-agents.md`.

SwarmX preserves native settings and exposes advertised native mode choices per conversation.
Ordinary tasks retain native tools, delegation, hooks and ambient MCP. The Host injects a scoped
product-tool bridge where supported; OpenClaw uses its Gateway tool configuration. External
ACP currently receives no Host product-tool bridge or automatic Host Memory injection.
Host API grants are enforced independently of native execution modes. SwarmX does not rewrite
global configuration. Native history is read on demand. Empty
Claude sessions have no native transcript. Codex/Claude filter native
history by directory. Hermes/OpenClaw expose global native catalogs, so the Host lists and
accepts only sessions created or previously run in the execution directory's journal. Empty created tasks
have durable ownership records; a memory review record cannot establish task ownership.
Hermes uses ordinary native resume for execution and permits its configured automatic continuation.
History/catalog access uses the native lazy watch interface. OpenClaw does not discover local CLI
credentials: configure its connection explicitly. DSH lists tasks and replays observed output from
the Host journal; SDK sessions are not restored across processes. Create a new DSH task for new work.

## External interfaces

After `pnpm build`, an ACP client launches:

```sh
node apps/desktop/dist/acp-main.js --agent codex
```

Or run `pnpm --silent acp`. ACP uses the official SDK and stdio; stdout contains only protocol
messages. ACP uses the Host's canonical execution directory and prints the A2A URL to stderr.
ACP supports initialize, list/new/load/prompt/cancel and form elicitation. Requests for another
directory and client-injected MCP servers are rejected; configure the native Agent.

A2A discovery: `/a2a/:agent/.well-known/agent-card.json`.
JSON-RPC: `/a2a/:agent`, with `swarm` as the default Agent.
Calls require `A2A-Version: 1.0` and a bearer token. Set `SWARMX_API_TOKEN` before startup for
external clients; otherwise the Host generates a private process token for product carriers.
Only text SendMessage, GetTask and CancelTask are provided. Native approvals/questions require
ACP with form elicitation or the desktop, not this A2A endpoint. Calls use the Host's execution
directory; an explicitly supplied different directory is rejected.

The renderer uses official AG-UI input/events and assistant-ui's adapter over Electron IPC.
Cancellation stops native work. Interaction resume completes the original request without starting
another run. The `models.read` bridge operation reads the selected native model catalog behind
the trusted-window and session-ownership boundary. AG-UI `forwardedProps` accepts only
assistant-ui's `modelName` and `reasoningEffort`; these become per-run model settings through
the recursive Agent interface. See `desktop-ui.md` for Harness-specific capabilities.

The Host execution journal persists observed runs and product-tool operations below all gateways.
Every product MCP call requires an active session/run binding; a process token alone cannot create
an unscoped Agent call with Host authority.
Native product tools connect through the stdio MCP bridge to the Host socket with an execution
credential. The `logs.read` bridge operation provides directory-scoped cursor reads without
loading a native Agent. See `execution-log.md` for record semantics and coverage limits.
The `runs.control` bridge operation accepts a `runId` and `{ action: "steer", text }` or
`{ action: "cancel" }`. It invokes the active run's recorded native Agent and rejects inactive IDs
and pending confirmations. It never starts a new
run or looks up a different run by session ID. Child confirmation replies use AG-UI resume.
MCP text content preserves the product result as JSON; non-object results, including arrays,
use `{ value: result }` in the protocol's object-valued `structuredContent` field.

## Settings and domain boundaries

The Host resolves one canonical `SWARMX_CWD` (or starting directory) for its lifetime. Native
sessions and execution logs retain that directory ownership. Private settings, Memory preferences
and language survive restart. `settings.read/update` validates policy; language changes do not
restart active work. Busy work rejects permission changes.

The Host authorizes Memory, Work submissions, harness/model admission and delegation. Native
modes and approvals remain owned by each harness and do not grant Host authority. Memory reviews
retain their separate restricted native configuration; no cross-harness tool-free or filesystem
isolation guarantee is claimed. See [permissions](permissions.md).

Scientific models, notebook/figure execution and artifact stores belong to domain applications.
There is no Science or environment IPC surface. A trusted embedding Host may supply a
workspace-bound domain reference provider; Work requires exact ID/revision and read grants before
calling it. External ACP launch does not register one automatically. See [domain integration](domain-projects.md).
Legacy scientific settings and reference strings stay readable without activating a runtime.

The `tool` bridge invokes registered generic product tools, and `cancelTool` aborts its matching call.
`logs.read` and `logs.evidence` expose scoped execution evidence. Evaluation exports retain their
payload identities and are private unless the user explicitly shares them.

## Verification

`agents.test.ts` checks native integration selection and session ownership;
`gateways.test.ts` exercises the official ACP/A2A clients, run-bound product tools and AG-UI streaming.
Provider tests cover native events, interactions, cancellation and lifecycle. Codex types are generated
by its official CLI; SDK integrations use their published types. Internal Swarm composition has no ACP connection; the external boundary has its own adapter.

Opt-in real checks:

```sh
SWARMX_REAL_CODEX=1 pnpm exec vitest run apps/desktop/tests/agents-real.test.ts
SWARMX_HERMES_PYTHON=/path/to/hermes/venv/bin/python pnpm exec vitest run apps/desktop/tests/agents-real.test.ts
```

The Codex check reads native history after a no-tool prompt, then checks a product status call
and its execution provenance on the same thread. Its test observer accepts only the exact native
SwarmX `swarm` approval form; the test product boundary rejects every call except `{ action:
"status" }` before execution. Production confirmations remain interactive. The Hermes check exercises
session and model discovery without an LLM call. OpenClaw and DSH have native SDK
fixtures; external ACP has separate contract tests. DVC tests run when its CLI is
available. UI trace timing is observational; it is not a verified provenance claim.
