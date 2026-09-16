# Agent platform

## Local development

`pnpm dev` prepares the native compiler once, then uses `tsx watch` to rebuild
Host and library TypeScript and restart Electron on backend edits. Renderer files are excluded from
that watcher: Vite serves the renderer and HMR on a local development server. Electron loads its
URL only in explicit development mode; packaged builds load the static renderer. Product operations
use the sandboxed preload's Electron IPC bridge in both modes. Backend restarts shut down the Host
and active native work before exit.

Renderer-only HMR preserves state where React Fast Refresh supports it; changing component exports
can require a page reload. Backend restarts recreate the window and end live runs. Persisted sessions,
permission grants and research data survive, but unsent drafts may not. Rust compiler/dependency setup
changes require restarting `pnpm dev`.

## Native integrations

| Agent | Interface | Setup |
| --- | --- | --- |
| Pi (default) | Embedded Pi SDK | Native Pi authentication, models and settings in `~/.pi/agent/`; provider environment variables are supported |
| Codex | Installed Codex App Server | Native login/config; `CODEX_PATH` can select the executable |
| Claude | Packaged `@anthropic-ai/claude-agent-sdk` | Native authentication and Claude settings |
| Hermes | Installed native TUI Gateway over stdio | `SWARMX_HERMES_PYTHON` selects its Python interpreter; otherwise use the installed `hermes` interpreter |
| OpenClaw | Official Gateway Client | `OPENCLAW_GATEWAY_URL` (default `ws://127.0.0.1:18789`), `OPENCLAW_GATEWAY_TOKEN` or `OPENCLAW_GATEWAY_PASSWORD` |
| DSH | Official SDK Client and matching runtime | Native SDK profile/authentication; one independent execution per task |

`--agent` → `SWARMX_AGENT` → `pi`. The desktop can select any configured Agent. Only a selected
integration loads; errors are reported without fallback. ZCode and Kimi are not registered.
Host startup does not connect to a native CLI. Bootstrap reports session-list failures visibly
while keeping research and settings available. A later explicit native request may reconnect.

Native runtimes are Host-owned: one lazy instance per harness serves tasks in the execution
directory with the Host's permission ceiling and a scoped product-tool credential. The Host
disposes the runtimes at shutdown.

Pi uses its own authentication and settings. Run `pnpm --filter @swarmx/desktop exec pi` and
use `/login`, or configure a provider environment variable as described in the
[Pi provider guide](https://pi.dev/docs/latest/providers). Explicit harness
allowlists must include `pi` to admit it; changing the default does not widen saved grants.

SwarmX preserves native settings and exposes advertised native mode choices per conversation.
Ordinary tasks retain native tools, delegation, hooks and ambient MCP. The Host injects a scoped
product-tool bridge where supported; Pi custom tools call the same services directly and
OpenClaw still uses its Gateway tool configuration.
Host API grants are enforced independently of native execution modes. SwarmX does not rewrite
global configuration. Native history is read on demand. Empty
Pi/Claude sessions have no native transcript. Pi/Codex/Claude filter native
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

## Settings and scientific execution

The bundled image derives from the official `quay.io/jupyter/datascience-notebook` stack, pinned
to a dated release and immutable multi-architecture digest. Docker uses its native architecture;
the Host records the resolved image ID, platform and actual Python packages. No custom pip
dependency stack is overlaid. Python runs and metadata probes explicitly override the image's
Jupyter startup entrypoint; a notebook server is not exposed. R and Julia are available in the
base image but are not additional SwarmX execution modes.

The Host resolves `SWARMX_CWD`, or the process's starting directory, to one canonical execution
directory for its lifetime. Native sessions, scientific records and execution logs retain their
directory ownership. Research collections use the Science API's `project` entity type.

The private product home stores execution policy and the resolved environment in `settings.json`,
Memory preferences in `memory.json` and the UI language in `language.json`. Settings are shared
across launches. The `language.write` bridge operation accepts only `zh` or `en`; language changes
are allowed during execution and never restart a run or rewrite scientific content.

Settings shows the execution directory, language, permissions, environment and Memory preferences.
The `settings.read/update` bridge operations read configuration and update the strict policy object.
`environment.read` returns setup state, bounded logs, active process count and the resolved image
manifest; `environment.act` accepts `setup`, `inspect` or `cancel`. IPC validates callers and payloads.
Active work rejects permission changes and environment setup. Configuration and environment
operations enter the execution log.

The Host authorizes its Memory/Science APIs, selected harness/model and Swarm delegation.
Native mode selections retain their advertised IDs/labels and are reapplied for that conversation;
normal native approvals reach the connected user with exact options. Native slash commands,
hooks, internal delegation and title-model calls remain governed by the harness. A Host model
allowlist controls Host launches, not every autonomous native model call. A Host tool grant is not
a machine filesystem boundary. See `permissions.md` for inheritance and examples.

Background memory reviews use an admitted model, receive no Host product MCP credential,
and reject observed tool calls and approval requests. The Host requests native restrictions;
unmodified upstream adapters determine their effect, including title calls and session persistence.
These requests do not establish a cross-harness tool-free or ephemeral execution guarantee.
Old settings that contain `policy.approval` reject with an explicit update instruction. Historical
filesystem grants stay readable but cannot authorize new execution under the Host-only semantics.
User-originated research edits and Host bookkeeping are distinct from Agent API grants.

Desktop notebook execution is stateless Python through Docker. Each cell must declare its inputs
and contain its imports; notebook history records cells but does not preserve a live kernel.
The public Science package retains its separately configured JupyMCP runtime for existing library
consumers. The Desktop Host never selects that path. Scientific Python containers have no network,
a read-only root, no capabilities or new privileges, a PID limit, CPU/memory limits and a wall-clock
timeout. The execution directory and verified artifact input files are mounted; input files are
read-only. Cancellation removes the owned container, including its descendants. Docker daemon
availability is required. The Host and daemon are trusted; containers are not a security
certification or a boundary against a compromised daemon.

The bundled Typst document compiler and explicit Git/DVC operations retain their existing host
execution boundary. Python environment settings do not sandbox those native components.

The `tool` bridge operation invokes registered product tools; `cancelTool` aborts the matching call.
The `science` bridge reads research collections, RO-Crate metadata, artifact previews and immutable
artifact content. Artifact imports are limited to 8 MiB; downloads verify immutable bytes and are
limited to 32 MiB. SVG previews use an image element, not executable inline markup. Exports contain
RO-Crate metadata; payload files download separately. `science.notebookExecutions` reads the latest
100 execution summaries for a research collection directly from the journal; it omits repeated
notebook snapshots and rejects records owned by another execution directory.

## Verification

`agents.test.ts` checks native integration selection and session ownership;
`gateways.test.ts` exercises the official ACP/A2A clients, run-bound product tools and AG-UI streaming.
Provider tests cover native events, interactions, cancellation and lifecycle. Codex types are generated
by its official CLI; SDK integrations use their published types. Internal code has no ACP adapter.

Opt-in real checks:

```sh
SWARMX_REAL_CODEX=1 pnpm exec vitest run apps/desktop/tests/agents-real.test.ts
SWARMX_HERMES_PYTHON=/path/to/hermes/venv/bin/python pnpm exec vitest run apps/desktop/tests/agents-real.test.ts
SWARMX_TEST_DOCKER_IMAGE=swarmx-research:validation pnpm vitest run apps/desktop/tests/research-environment.test.ts
```

The Codex check reads native history after a no-tool prompt, then checks a product status call
and its execution provenance on the same thread. Its test observer accepts only the exact native
SwarmX `swarm` approval form; the test product boundary rejects every call except `{ action:
"status" }` before execution. Production confirmations remain interactive. The Hermes check exercises
session and model discovery without an LLM call. OpenClaw and DSH have native SDK
fixtures; external ACP has separate contract tests. DVC tests run when its CLI is
available. UI trace timing is observational; it is not a verified provenance claim.
