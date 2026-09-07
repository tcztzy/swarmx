# Agent platform

## Local development

`pnpm dev` prepares the native compiler once, then uses `tsx watch` to rebuild
Host/workspace TypeScript and restart Electron on backend edits. Renderer files are excluded from
that watcher: the Host mounts Vite middleware behind the existing cookie, Host and Origin checks,
with HMR on the same loopback server. Development HTML/scripts/styles receive a per-Host CSP nonce;
production keeps its static renderer and existing CSP. Vite is loaded only in explicit development
mode, which the Electron entry point ignores in packaged builds. No alternate API origin or CORS
exception is introduced. Backend restarts shut down the Host and active native work before exit.

Renderer-only HMR preserves state where React Fast Refresh supports it; changing component exports
can require a page reload. Backend restarts recreate the window and end live runs. Persisted sessions,
permission grants and research data survive, but unsent drafts may not. Rust compiler/dependency setup
changes require restarting `pnpm dev`.

## Native integrations

| Agent | Interface | Setup |
| --- | --- | --- |
| Codex (default) | Packaged `@agentclientprotocol/codex-acp` | Native login/config; adapter includes Codex, `CODEX_PATH` can override it |
| Claude | Packaged `@agentclientprotocol/claude-agent-acp` | Native authentication and Claude settings |
| Hermes | Installed `hermes acp` over stdio | Hermes with ACP extras; existing `SWARMX_HERMES_PYTHON` selects `python -m acp_adapter` |
| OpenClaw | Installed `openclaw acp` over stdio | Native Gateway config, optional `OPENCLAW_GATEWAY_URL` and `OPENCLAW_GATEWAY_TOKEN` |

`--agent` → `SWARMX_AGENT` → `codex`. The browser can select any configured Agent. Only a selected
integration loads; errors are reported without fallback. DSH, ZCode and Kimi are not registered.
Host startup does not connect to a native CLI. Bootstrap reports session-list failures visibly
while keeping research and settings available. A later explicit native request may reconnect.

SwarmX preserves native settings and exposes advertised native mode choices per conversation.
Ordinary tasks retain native tools, delegation, hooks and ambient MCP. The Host injects a scoped
product MCP endpoint where supported; OpenClaw still uses its Gateway tool configuration.
Hermes/OpenClaw no longer require a native filesystem-ceiling mapping to launch. Host API grants
are enforced independently. SwarmX does not rewrite global configuration. Native history is read on demand. Empty
Claude sessions have no native transcript until the first prompt. Codex/Claude filter native
history by directory. Hermes/OpenClaw expose global native catalogs, so the Host lists and
accepts only sessions created or previously run in this project's journal. Empty created tasks
have durable ownership records; a memory review record cannot establish task ownership.

## External interfaces

After `pnpm build`, an ACP client launches:

```sh
node apps/desktop/dist/acp-main.js --agent codex
```

Or run `pnpm --silent acp`. ACP uses the official SDK and stdio; stdout contains only protocol
messages. ACP binds to the initially opened project and prints its project-scoped A2A
URL to stderr. ACP supports initialize, list/new/load/prompt/cancel and form elicitation.
Client-injected MCP servers and alternate workspaces are rejected; configure the native Agent.

A2A discovery: `/projects/:id/a2a/swarm/.well-known/agent-card.json`.
JSON-RPC: `/projects/:id/a2a/swarm`.
Calls require `A2A-Version: 1.0` and a bearer token. Set `SWARMX_API_TOKEN` before startup for
external clients; otherwise the Host generates a private process token for product carriers.
Only text SendMessage, GetTask and CancelTask are provided. Native approvals/questions require
ACP with form elicitation or the browser, not this A2A endpoint.

The browser uses official AG-UI input/events and assistant-ui's adapter. Stream disconnect stops
native work. Interaction resume completes the original request without starting another run.
`GET /api/v1/models?agent=…&session=…` reads the selected native model catalog behind the same
browser authentication and session-ownership boundary. AG-UI `forwardedProps` accepts only
assistant-ui's `modelName` and `reasoningEffort`; these become per-run model settings through
the recursive Agent interface. See `desktop-ui.md` for Harness-specific capabilities.

The Host execution journal persists observed runs and product-tool operations below all gateways.
Every product MCP call requires an active session/run identity; bearer authentication alone cannot
create an unscoped Agent call with project-level authority.
Codex product MCP metadata carries native thread/turn IDs; Claude MCP URLs carry session/execution
IDs. The Host checks both against the active run. `GET /api/v1/logs` provides authenticated, workspace-scoped cursor reads without
loading a native Agent. See `execution-log.md` for record semantics and coverage limits.
`POST /api/v1/runs/:runId` accepts `{ action: "steer", text }` or `{ action: "cancel" }`
behind the browser cookie/Host/Origin boundary. It invokes the active run's recorded native
Agent and rejects inactive IDs and pending confirmations with HTTP 409. It never starts a new
run or looks up a different run by session ID. Child confirmation replies use AG-UI resume.
MCP text content preserves the product result as JSON; non-object results, including arrays,
use `{ value: result }` in the protocol's object-valued `structuredContent` field.

## Workspace and scientific execution

The bundled image derives from the official `quay.io/jupyter/datascience-notebook` stack, pinned
to a dated release and immutable multi-architecture digest. Docker uses its native architecture;
the Host records the resolved image ID, platform and actual Python packages. No custom pip
dependency stack is overlaid. Python runs and metadata probes explicitly override the image's
Jupyter startup entrypoint; a notebook server is not exposed. R and Julia are available in the
base image but are not additional SwarmX execution modes.

`PUT /api/v1/language` accepts only `zh` or `en` behind the browser Host/Origin/cookie boundary.
The private product home's `language.json` stores this UI preference independently of workspace
execution policy. Bootstrap restores it even when the Host port or workspace changes. Language
changes are allowed during execution and never restart a run or rewrite scientific content.

The private product home's `projects.json` stores named projects, each with its own canonical
directory, and the last opened project ID. The first launch registers `SWARMX_WORKSPACE` (otherwise
cwd); subsequent launches restore the last opened project. Adding an existing directory resolves
symlinks and deduplicates aliases. Directories are checked again when loaded. Settings remain
atomically stored under `workspaces/<canonical-id>/settings.json`; the ID hashes the real path,
preserving existing settings, memory and research data. Data is never copied between projects.

The sidebar adds and opens projects through `GET/POST /api/v1/projects` and
`POST /api/v1/projects/:id/open`. Each browser page lives at `/projects/:id/`; all its REST,
AG-UI, MCP and A2A requests carry that immutable project prefix. Loaded projects retain separate
service owners until Host shutdown. Selecting a project updates navigation preference without
retargeting other pages or interrupting their work. Leaving a page still disconnects its own
foreground stream under the existing cancellation contract. Research collections inside a
filesystem project retain the Science API's existing `project` entity type.

Global Settings contain language and user preferences. Project Settings contain the read-only
directory, execution policy, environment and project memory; add/open a different project to
work in another directory.

`GET/PUT /api/v1/settings` reads the current configuration or updates the strict policy object.
Legacy unprefixed API routes remain available. `PUT /api/v1/workspace` accepts `{ root }` and
registers/opens a default project only while idle; the UI does not use it.
`GET /api/v1/environment` returns setup state, bounded
logs, active process count and the resolved image manifest; `POST` accepts `setup`, `inspect` or
`cancel`. Mutations share the browser cookie/Host/Origin boundary. Active work rejects permission
changes and environment setup. Configuration and environment operations enter the execution log.

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
timeout. The selected workspace and verified artifact input files are mounted; input files are
read-only. Cancellation removes the owned container, including its descendants. Docker daemon
availability is required. The Host and daemon are trusted; containers are not a security
certification or a boundary against a compromised daemon.

The bundled Typst document compiler and explicit Git/DVC operations retain their existing host
execution boundary. Python environment settings do not sandbox those native components.

`POST /api/v1/tools/:name` accepts only registered Science tools and aborts execution when the
request disconnects. `GET /api/v1/science`, `/research-object?project=…`, `/artifact-preview?id=…`
and `/artifacts/:id/content` read current workspace data. Artifact imports are limited to 8 MiB;
downloads verify immutable bytes and are limited to 32 MiB. SVG previews use an image element,
not executable inline markup. Exports contain RO-Crate metadata; payload files download separately.
`GET /api/v1/notebook-executions?project=…` reads the latest 100 execution summaries directly from
the existing journal; it omits repeated notebook snapshots and rejects foreign project IDs.

## Verification

`agents.test.ts` checks upstream ACP lifecycle, configuration, permission callbacks and cancellation;
`gateways.test.ts` exercises the official ACP/A2A clients, run-bound MCP and browser security/streaming.
Upstream adapters own native types and event translation; SwarmX does not generate vendor protocol types.

Opt-in real checks:

```sh
SWARMX_REAL_CODEX=1 pnpm exec vitest run apps/desktop/tests/agents-real.test.ts
SWARMX_REAL_ACP_HANDSHAKE=1 pnpm exec vitest run apps/desktop/tests/acp-process.test.ts
SWARMX_HERMES_PYTHON=/path/to/hermes/venv/bin/python pnpm exec vitest run apps/desktop/tests/agents-real.test.ts
SWARMX_TEST_DOCKER_IMAGE=swarmx-research:validation pnpm vitest run apps/desktop/tests/research-environment.test.ts
```

The Codex check reads native history after a no-tool prompt, then checks a product status call
and its execution provenance on the same thread. Its test observer accepts only the exact native
SwarmX `swarm` approval form; the test product boundary rejects every call except `{ action:
"status" }` before execution. Production confirmations remain interactive. The Hermes check exercises
session discovery/create/history/interrupt without an LLM call. The handshake check starts both
packaged ACP executables without a model prompt; OpenClaw has simulated ACP tests. DVC tests run when its CLI is
available. UI trace timing is observational; it is not a verified provenance claim.
