# SwarmX Architecture

This document records current architectural boundaries and the reasons behind
them. Product requirements live in `SPEC.md`; exact schemas and behavior live in
source code and tests. Dependency versions belong in package manifests and the
lockfile, not here.

## Packages

| Package | Owns |
| --- | --- |
| `@swarmx/core` | Agent and workflow execution, ACP/MCP clients, Sessions, Projects, schemas, and reusable platform contracts |
| `@swarmx/runtime` | Host runtime detection, Doctor reports, and explicit repair planning |
| `@swarmx/acp-server` | ACP server implementation backed by Core Sessions |
| `@swarmx/cli` | Commander-based terminal interface and HTTP server commands |
| `@swarmx/desktop` | Electron Main, Preload, Renderer, and host integrations |
| `swarmx` | Desktop-first npm launcher and CLI compatibility entry point |

Core may expose Node-specific APIs from its root. Browser consumers must use a
documented browser-safe subpath such as `@swarmx/core/rendering`.

## Identity and composition

SwarmX keeps distribution, execution, and supply identities separate:

- An **Extension** distributes Software, Skills, MCP servers, Agent profiles,
  and other passive metadata.
- A **Harness** is a reproducible runtime recipe: Software, selected Skills and
  MCP servers, Project context, delivery capabilities, and permission policy.
- A **Model** is an independent primary entity with API and capability metadata.
- A **Provider** is an explicit connection and credential source that may supply
  Models.
- A **ModelSupply** links one Model to one Provider route.
- An **Agent** is exactly one Harness paired with one Model and is identified by
  `harnessId:modelId`.

Provider selection, ModelSupply routing, runtime aliases, and reasoning effort
do not create new Agent identities.

The ordinary Desktop composer exposes Harness, Model, and Effort. It does not
ask users to select an internal Provider route. Composition preflight validates
the resulting matrix cell and reports missing runtime, connection, Skill, MCP,
context, or permission requirements before execution.

## Execution paths

### Direct SwarmX

Core's `Agent` executes supported Provider APIs natively. The selected API mode
and request-scoped environment remain explicit. Native execution preserves
streaming, cancellation, tool continuation, and Provider-specific message
shapes instead of normalizing every request through a compatibility bridge.

Desktop may inject host-owned Project tools into a direct SwarmX task. The
selected built-in tool style changes model-facing names and schemas while all
styles dispatch through the same containment, permission, cancellation, and
output boundaries.

### External ACP Harness

`AcpClient` launches an ACP-compatible subprocess, initializes it, creates or
loads a Session, negotiates configuration, sends prompts, and consumes
`session/update` notifications.

External Harnesses own their native tools, authentication, configuration, and
permission behavior. SwarmX does not inject duplicate Project tools.

Desktop can wrap selected external Harnesses in a protected container runtime.
The wrapper receives an explicit workspace mount and allowlisted request
environment. Harnesses that intentionally reuse a user's native runtime remain
native and are described honestly in runtime diagnostics.

### ACP server

`@swarmx/acp-server` presents SwarmX as an ACP agent. A persisted Core Session is
the conversation authority. Advertised cwd, resources, MCP support, history,
and cancellation must match implemented behavior.

## Workflow engine

`SwarmConfigSchema` is the only workflow schema. A workflow contains:

- a named `root`;
- a map of `agent`, `tool`, or nested `swarm` nodes;
- explicit edges with optional CEL conditions;
- optional MCP server and hook metadata.

`Swarm` parses the config, materializes nodes into a `Map`, and stores edges as
`Edge` objects. Construction rejects unconditional cycles and warns about
conditional cycles that require an escape condition.

Execution starts at `root`, evaluates outgoing edges after each node, waits for
declared predecessors, schedules a node at most once, and enforces a step bound.
Execution output is an ordered collection of normalized message chunks. Eval
execution additionally records deterministic step metadata and metrics.

The n8n importer is a boundary adapter into `SwarmConfig`; it is not another
runtime.

## Sessions and Projects

Canonical Sessions are append-only JSONL event logs under
`~/.swarmx/sessions/`. Events create a Session, append or replace messages, and
update metadata. A rebuildable JSONL index supports task lists without loading
message bodies.

Replay accepts one torn, unterminated final record as a recoverable crash tail.
A complete malformed record fails closed. Legacy JSON Sessions remain readable
and can be migrated only after replay equivalence is verified.

Projects are local folder bookmarks stored separately from Sessions. A Project
groups tasks and supplies the canonical working root for direct tools. It is not
a remote workspace, identity boundary, or authorization domain.

Side chats use transient Session forks anchored to the effective parent
history. They remain in memory and read-only until an explicit promotion creates
a normal Session.

## Desktop architecture

Desktop follows Electron's Main/Preload/Renderer security model.

### Main

Main owns all privileged capabilities:

- filesystem and managed media;
- subprocess, PTY, LSP, and container execution;
- Provider credentials and network requests;
- Sessions, Projects, settings, and Extension lifecycle;
- runtime diagnostics, updates, browser hosting, and IPC authorization.

IPC handlers validate inputs before calling domain services. Privileged handlers
accept requests only from the configured main frame.

### Preload

Preload exposes a narrow `contextBridge` API. It transports typed requests,
responses, and request-scoped event subscriptions without exposing Node.js
objects or generic IPC access.

### Renderer

Renderer is a React application. It owns presentation and transient UI state,
but no direct filesystem, subprocess, credential, or arbitrary network
authority.

Renderer-facing data is normalized and sanitized in Main or browser-safe Core
modules before display.

## Project tools and permissions

Direct Project tools share one safety boundary:

- canonical Project containment and symlink checks;
- complete-read digest checks before modifying existing files;
- bounded input, output, file size, and runtime;
- atomic writes;
- request cancellation and process-group termination;
- sanitized child environments;
- platform sandboxing that fails closed when required but unavailable.

Permission policy is independent from operating-system sandboxing. Effective
authority combines managed, Project, personal, Agent, and conversation layers.
Explicit denial and lower-authority ceilings win. A one-call approval never
grants path, network, environment, or sandbox escalation.

Model-facing tool profiles may emulate implemented Claude Code, Codex, or Kimi
Code contracts. A public tool is exposed only when the host can provide its real
schema and behavior.

## Providers and Models

Desktop Provider connections are explicit settings records backed by the
user-editable `~/.swarmx/provider-auth.json` file. Credentials are plaintext at
rest in that file, which is written with restrictive permissions. Ambient
environment variables do not create visible Desktop connections.

Provider discovery produces independent Model records and ModelSupply links.
The built-in Model registry enriches known ids with verified capability
metadata; it does not own the visible catalog.

Only Main resolves Provider secrets, calls Provider endpoints, or constructs a
request environment. Renderer receives readiness, catalog, usage, balance, and
rate-limit summaries without plaintext credentials.

## Extensions and Custom Agents

Extension inventory is passive. Parsing a manifest may discover Software,
Harnesses, Models, Providers, Agents, Skills, MCP servers, commands, LSP
servers, hooks, monitors, assets, policies, connectors, and UI contribution
references, but it does not execute them.

Installation, update, rollback, trust changes, enablement, and repair are
separate validated actions. Executable UI components are host-registered React
components; manifests cannot deliver inline scripts, HTML, or render functions.

Custom Agents store a composition recipe and resolve it through the same
preflight and execution path as Extension-provided profiles. Native Claude Code
and Codex Agent definitions are read-only import/projection formats around the
canonical profile.

## Messages, rendering, and media

Runtime output uses typed message chunks for user/assistant messages, reasoning,
tool calls, tool results, progress, system notices, and attachments.

Normalized render events are derived presentation state. They carry sanitized
summaries, status, provenance, and artifact references but do not replace
canonical Session messages or raw host logs.

Desktop imports attachments into a content-addressed managed store. Sessions
persist bounded metadata and local URIs, never Base64 payloads. Main validates
path, MIME, size, identity, and capability before preview or transport.

Remote Markdown media is blocked by default. Tool payloads remain literal unless
they have passed through the normalized rendering boundary.

## Settings and secrets

Desktop settings use a queued atomic store so narrow section updates do not
overwrite unrelated state. Zod schemas validate persisted documents and IPC
updates.

Settings contain secret references only. The dedicated Provider auth document
may contain plaintext credentials so users can inspect and edit it directly.
Plaintext is resolved only for the current Provider request or child process and
is never returned to Renderer, telemetry, traces, or inventory metadata.

## Architectural rules

- Prefer one canonical model over synchronization between competing models.
- Validate data at process, persistence, protocol, and plugin boundaries.
- Keep generic schema and decision modules side-effect free.
- Put host effects behind explicit adapters and user actions.
- Keep Renderer imports browser-safe.
- Preserve external host semantics instead of advertising approximate parity.
- Add a package or abstraction only when it creates a real ownership boundary.
- Keep volatile dependency versions in manifests and the lockfile.
- Use focused tests as the executable contract for field-level behavior.
