# SwarmX Architecture

This document records current architectural boundaries and the reasons behind
them. Product requirements live in `SPEC.md`; exact schemas and behavior live in
source code and tests. Dependency versions belong in package manifests and the
lockfile, not here.

## Packages

| Package | Owns |
| --- | --- |
| `@swarmx/core` | Agent and workflow execution, the durable task control plane, ACP/MCP clients, Sessions, Projects, schemas, and reusable platform contracts |
| `@swarmx/runtime` | Host runtime detection, Python worker environment inspection, Doctor reports, and explicit setup/repair planning |
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

The Core Harness registry is the single source for built-in software names,
pinned adapter versions, launch commands, and backends. Runtime protection and
Renderer workflow examples consume that registry instead of copying package
versions or commands.

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

## Durable task runtime

Durable tasks have a control plane separate from conversations and workflow
graphs. `@swarmx/core` is the protocol and state authority; a worker is a
replaceable executor that receives one leased operation and reports bounded
events. The first backend is Python, but the kernel does not depend on Python or
on engineering-specific lifecycle enums such as `project_iteration` and
`analysis_execution`.

### Kernel and persistence

The generic kernel models `WorkItem`, `Run`, fenced `Lease`, `Budget`, progress,
execution `Checkpoint`, artifact reference, approval, schedule, Session link,
and external side-effect receipt. Engineering lifecycle and analysis schemas in
`autonomy.ts` remain an upper domain layer rather than constraints on the task
kernel.

Task state is rebuilt from strict versioned events. The Node store appends and
fsyncs JSONL under `~/.swarmx/task-runtime/events.jsonl`, uses a narrow writer
lock, and stores JSON inputs, checkpoint payloads, and results as
secret-scanned content-addressed JSON or binary blobs. Replay ignores an exact
duplicate event, rejects a reused event or idempotency key with different
content, accepts one torn final record for explicit truncation recovery, and
fails closed on a complete corrupt record.

The app-attached control service acquires monotonically fenced leases, records
worker heartbeats and progress, persists cancellation before signaling the
worker, and decides retry only after authoritative failure or expired-lease
recovery. An unleased run proposal does not consume an attempt or make the task
active, and startup reconciliation repairs a crash between a retryable failure
event and its retry-scheduled event. A checkpoint carries its environment digest
and parent link. Resume reparses the content-addressed payload and requires its
checkpoint identity and environment digest to agree with persisted metadata,
the new Run, the protocol `start`, and the verified worker launch. Artifact
receipts must use protocol-safe relative paths; the host resolves them from the
dedicated artifact root, rejects symlink and containment escapes, verifies the
opened file's declared size and SHA-256, and copies it into the runtime's
immutable content-addressed store before it becomes a runtime reference. OS
process sandboxing and stronger defenses against filesystem races remain part
of the production host boundary.

The runtime promises at-least-once handling, not exactly-once effects. An
external capability call carries an idempotency key. Core persists an
`uncertain` receipt before dispatch and upgrades it to `committed` only after a
gateway returns durable evidence. The committed receipt includes a replayable,
validated outcome; reuse of its idempotency key by different work fails closed
without disclosing the first result. A lost response remains unknown; replay
must not invent success or automatically repeat an effect whose outcome is
unclear.

### Session and context relationship

A WorkItem is authoritative independently of any Session. Multiple Sessions may
link to the same WorkItem as creator or observer, and unlinking, switching, or
archiving a Session does not cancel it. Session events remain the conversation
authority only.

`ContextPacket` and `SummaryCheckpoint` select and summarize model context. They
cannot resume worker execution and never substitute for a task-runtime
checkpoint. Conversely, a task checkpoint is opaque executor state and does not
replace Session history or model-context summaries.

### Worker protocol and process boundary

Core owns version 1 of a strict, size-bounded JSONL protocol over stdio. A worker
first sends `hello` with its backend, environment digest, operations, and
features. Core replies with selected capabilities and grants, then `start` with
the WorkItem/run/lease identity, fencing token, operation, budget, and optional
execution checkpoint. Worker messages include heartbeat, progress, checkpoint,
artifact, human-needed, completion, failure, cancellation acknowledgement, and
capability calls. Every run event has a monotonic sequence and must match the
active lease and fencing token.

The process host passes an explicit child environment rather than inheriting
ambient Provider credentials, bounds and sanitizes stderr, checks the initial
handshake and terminal record, enforces heartbeat, wall-time, artifact-size, and
terminal-exit watchdogs, rejects output after a terminal record, and terminates
the process group after a canceled run exceeds its grace period. A persisted
human request may resume with a bounded approval decision; it is not held only
in process memory. Provider or tool access crosses a narrow, grant-checked
capability gateway. A Main-owned gateway may resolve a Provider secret
internally, but plaintext credentials cannot enter worker protocol messages or
worker environment metadata.

The worker protocol is not ACP. ACP Harnesses continue to own their native
tools, authentication, Sessions, and permissions. Python operations are task
executor capabilities rather than Agents, Harnesses, or a new `SwarmConfig`
node kind. Desktop `WorkspaceShell` remains a temporary interactive process
surface and is not used as durable task authority.

### Python environment boundary

`@swarmx/runtime` performs read-only discovery of runtime assets, `uv`, a
compatible uv-managed Python, and a digest-addressed locked environment. Status
checks use offline/no-download modes. A verified launch bypasses `uv` and starts
the environment's interpreter directly with isolated, unbuffered Python flags
and a sanitized environment. Launch verification reruns the health check and
executes an in-memory snapshot whose hash matches the environment digest, so a
mutable worker source path is not reopened by the child. Missing Python or stale
dependencies yield an explicit setup/repair plan; task execution never installs
Python or synchronizes dependencies as a side effect.

The root Python project is named `swarmx`. Its product worker has no required
third-party Python dependencies; Inspect evaluation tooling is isolated in the
opt-in `inspect` dependency group and setup uses no default groups. The
environment digest covers the project metadata, lock, worker source, `uv`,
Python implementation/version, platform, and architecture so incompatible
checkpoints fail closed.

The current service lifecycle is explicitly `app_attached`. Persisted work and
startup lease recovery survive a Desktop restart, but no `swarmxd`, launchd, or
systemd supervisor currently keeps execution alive after Desktop exits. A
future daemon must reuse the same Core schemas, event store, fencing, and worker
protocol behind an authenticated local control boundary; it must not introduce
a second task authority.

## Audit event authority

Audit events answer who requested or decided what boundary action, which
Session/task/request it belonged to, and whether it was attempted, denied,
completed, cancelled, or failed. They do not duplicate Session messages, task
payloads, source, terminal streams, HTTP bodies, headers, credentials, or
environment snapshots.

Core owns one strict event schema and a local append-only JSONL store under
`~/.swarmx/audit/`. Each event has a monotonic sequence, stable correlations,
bounded sanitized metadata, the previous event hash, and its own SHA-256 hash.
A separately fsynced head checkpoint makes ordinary truncation visible. Writers
serialize through a narrow lock, create files with restrictive permissions, and
require explicit recovery for one torn final record, a missing newline, or a
fully verified tail left ahead of its checkpoint; complete corruption fails
closed. This supplies local tamper-evidence, not remote attestation when an
attacker can rewrite both the event file and checkpoint.

Privileged boundaries persist an `attempted` event before authorization or
effect. They then append a terminal outcome. If the attempt cannot be persisted,
the boundary does not expand authority or start the effect. A failure after an
external effect has begun is reported honestly rather than rewriting history.
Read-only query, chain verification, and JSONL export operate on the same strict
replay path.

### Boundary taxonomy

Audit actions identify a semantic effect or a stable transport family, not an
individual UI method. Desktop IPC therefore uses the single transport action
`ipc.request` and identifies the normalized channel as target
`{ kind: "ipc-channel", id }`. Each channel has one explicit emission policy:

| Policy | Use | Events |
| --- | --- | --- |
| `intent_outcome` | Privileged or mutating requests | Durable intent before authorization/effect, then terminal outcome |
| `failure_only` | Benign reads, pure transforms, or transient UI state | No success noise; only denied, cancelled, or failed outcomes |
| `semantic_only` | Calls whose host service records the real effect | No duplicate successful transport event; authorization or dispatch failures remain visible as `ipc.request` |

Terminal IPC uses `semantic_only`. Its semantic action set is exactly
`terminal.create`, `terminal.write`, `terminal.resize`, `terminal.close`, and
`terminal.exit`. A close records one bounded `closeReason`:
`user_kill`, `owner_cleanup`, or `app_dispose`; terminal data, cwd, and
environment remain excluded.

CLI `send`, `eval-run`, and REPL turns share semantic action `agent.run`; the
bounded `surface` metadata (`cli_send`, `eval`, or `repl`) preserves the useful
distinction without multiplying action names.

### Activity is not audit

Activity persistence is disposable profile statistics, not decision evidence.
It stores exactly one `run_summary` for each run, containing status, duration,
token totals, and aggregate tool/Skill counts. It does not store per-tool or
per-Skill timeline events and cannot substitute for the verified audit chain or
canonical Session history.

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

Direct Project tools likewise define each structured input once as a Zod schema.
The same definition generates the model-facing JSON Schema and validates the
arguments before any workspace or process effect, so advertised and enforced
tool contracts cannot drift independently.

Markdown-based agent and Skill definitions share Core's YAML-backed frontmatter
parser; host adapters do not maintain partial, line-oriented YAML parsers.

Renderer styling remains feature-owned semantic CSS. Shared declarations may
group related semantic selectors (for example with `:is()`), but utility-class
rewrites and orphaned styles for removed UI variants are not retained.

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

## Skill self-improvement loop

Skill evolution is a closed loop with strict ownership boundaries: DSPy only
proposes candidates, evaluation only provides evidence, and TypeScript Core
decides promotion. Learning never happens inside an active request and never
mutates a running Session, active Skill files, or the persisted `SwarmConfig`.

### Ownership

- **TypeScript Core** owns every Zod boundary, the candidate/evaluation/
  promotion state machine, the active revision pointer, compare-and-swap
  promotion, idempotency, persistence, audit, and rollback. It reuses the
  durable task runtime (`swarmx.evolve_skill` WorkItems), `TaskRuntimeStore`
  content-addressed blobs, leases, checkpoints, and the capability gateway.
- **Python sidecar** reads only granted baseline/train/dev artifacts through
  the capability gateway, runs the optimizer (deterministic fake for the
  vertical slice, locked DSPy/GEPA in the `evolution` dependency group), and
  reports heartbeat/progress/checkpoint plus an immutable candidate artifact.
  It cannot move the active pointer, decide promotion, read
  `provider-auth.json`, scan Sessions, or write the Skill install directory.
- **Evaluation** runs baseline and candidate through the same real SwarmX
  execution path on a hidden holdout; Inspect produces independent evidence,
  Core computes the gate verdict.

### Trust boundaries

The optimization request names only baseline and train/dev content digests —
a holdout ref is rejected by the strict schema before the WorkItem is created.
The optimizer worker receives exactly three granted artifact refs and a
grant-checked `model.generate` capability whose credentials are resolved inside
a host-owned handler that never crosses the protocol. Candidates are
untrusted data: ingestion verifies the content digest against the artifact
receipt, re-checks lineage against the request, enforces the artifact budget,
and secret-scans content before anything may proceed to evaluation.

### State machine

`proposed -> evaluating -> staged | rejected | quarantined`. Only an eligible
evaluation moves a candidate to `staged`; promotion requires the human gate
(the policy gate is fail-closed until canary and drift monitoring exist) and a
compare-and-swap: the active revision must still equal the candidate's parent
revision at the moment the promotion receipt is appended. Promotion changes
only the ledger pointer used to resolve deliveries for future executions;
already-constructed Swarms and frozen Sessions are untouched. Rollback restores
any retained revision (the optimization baseline or a previously promoted
revision) through the same CAS, audit, and idempotency path.

### Persistence and audit

The append-only evolution ledger under `~/.swarmx/skill-evolution/` stores only
strict, per-kind secret-free records (request, candidate manifest, evaluation
manifest, promotion receipts with CAS expectations); every record payload is
validated by its own strict schema and candidates/evaluations are immutable —
a duplicate candidate id or an unanchored promotion receipt is rejected at
replay. Promotion is verified against the recorded optimization request
(staged candidate, eligible evaluation, request-anchored baseline, promoted
content coordinates equal to the candidate manifest, revision id derived from
the content digest) before the compare-and-swap is applied. The audit chain records `attempted` intent before
promotion/rollback/decide effects and fails closed when the intent cannot be
persisted; a terminal audit failure after the CAS is reported as failed and
left for operator verification — it is never silently claimed completed. A
CAS rejection itself records a failed outcome beside the durable intent. Audit
metadata is limited to ids, digests, metric summaries, budget usage, and
correlation ids — never raw Skill text, prompts, responses, or credentials.

### Skill delivery

`prompt_fragment` deliveries are request-scoped: the caller loads the Skill
Markdown from a trusted content-addressed artifact, verifies its digest, and
hands the verified fragment to the Agent constructor, which appends it to the
model-visible system/developer instructions. Deliveries bind to a named Agent
node (`skillInstructionsByAgent`) so a promoted revision never leaks to every
agent in a swarm, and non-native backends reject deliveries at construction.
The production entry point is `swarmx send --resolve-skill <skillId>:<variantId>`
and `swarmx eval-run --resolve-skill ...`, which read the promoted active
revision for new executions while already-constructed Swarms stay frozen.
Persisted config and original Skill files never change. `host_native_plugin`,
`rules_file`, `unsupported`, and external ACP Harness delivery are rejected
explicitly rather than approximated.

### Evaluation and budgets

Paired evaluation executes baseline and candidate in the seeded-randomized
order it records, so caching, rate limits, and stateful models cannot
systematically favor one side. Evaluation swarms are restricted to direct
native agents without queen agents, hooks, MCP servers, tool nodes, or external
backends; evidence binds the target agent, target model fingerprint, and a
host-verified config digest, and the ledger refuses evaluation records that
are not structurally and arithmetically consistent with a real run. The worker budget is keyed exactly like the capability receipts
(`skill_evolution:read_artifact`, `skill_evolution:model.generate`), a zero
model-call budget is a hard error for GEPA (never silently defaulted), and
tokens are a hard budget: zero denies every call before dispatch, the worker
denies exhausted budgets before the next call, and the host gateway re-checks
remaining tokens from durable receipts before each dispatch. The CLI launch
digest covers the worker source, pyproject, `uv.lock`, evolution sidecar
sources, the resolved Python version, and the interpreter's installed `dspy`
version verified against the pinned `evolution` group; the runtime environment
service can synchronize the opt-in group explicitly. Reflection/model calls
cross the grant-checked capability gateway; the CLI only enables
`proposer: gateway` with an explicit `--model-command` whose environment
carries no credentials.

## Settings and secrets

Desktop settings use a queued atomic store so narrow section updates do not
overwrite unrelated state. Zod schemas validate persisted documents and IPC
updates.

Settings contain secret references only. The dedicated Provider auth document
may contain plaintext credentials so users can inspect and edit it directly.
Plaintext is resolved only for the current Provider request or child process and
is never returned to Renderer, telemetry, traces, or inventory metadata.
Secret-free Core metadata boundaries share one recursive sensitive-field
classifier. Reference fields and redacted placeholders are allowed; plaintext
vault values are accepted only through schema-owned, path-scoped exceptions.

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
