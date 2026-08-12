# SwarmX Architecture

This document records current architectural boundaries and the reasons behind
them. Product requirements live in `SPEC.md`; exact schemas and behavior live in
source code and tests. Dependency versions belong in package manifests and the
lockfile, not here.

## Packages

| Package | Owns |
| --- | --- |
| `@swarmx/core` | Agent and workflow execution, the durable task control plane, ACP/MCP clients, Sessions, Projects, schemas, and reusable platform contracts |
| `@swarmx/runtime` | Host runtime detection, Python worker and managed module environment inspection, Doctor reports, and explicit setup/repair planning |
| `@swarmx/acp-server` | ACP server implementation backed by Core Sessions |
| `@swarmx/cli` | Commander-based terminal interface and HTTP server commands |
| `@swarmx/desktop` | Electron Main, Preload, Renderer, and host integrations |
| `swarmx` | Desktop-first npm launcher and CLI compatibility entry point |

Core may expose Node-specific APIs from its root. Browser consumers must use a
documented browser-safe subpath such as `@swarmx/core/rendering`.

## Renderer styling

The Desktop Renderer has one CSS compilation path: Tailwind CSS runs through
its first-party Vite plugin and emits the stylesheet used by both the Electron
application and the public `@swarmx/desktop/styles.css` export. The public
subpath keeps its stable name and resolves to compiled CSS; package consumers
must not be required to install or compile Tailwind themselves.

Tailwind Preflight is intentionally disabled. SwarmX already owns base element
behavior for an Electron window, Markdown, KaTeX, and xterm, so an additional
opinionated reset would be an uncontrolled compatibility change. The stylesheet
declares explicit `theme`, `base`, `components`, and `utilities` cascade layers:

- `theme` exposes semantic application tokens to Tailwind utilities;
- `base` owns the existing element reset, host-specific behavior, and
  third-party base imports;
- `components` is reserved for future component recipes that do not participate
  in local utility overrides;
- `utilities` contains both generated, statically discoverable classes and the
  bounded relationship, pseudo-element, rich-content, and third-party rules
  that must share their cascade layer. Residual rules load after generated
  utilities so their higher selector specificity continues to override a
  migrated element's base utilities just as it did before the migration.

Reusable components with visual variants use Class Variance Authority as the
single mapping from semantic props to complete Tailwind class strings. Variant
types are derived from that mapping. Runtime data never constructs Tailwind
utility names by interpolation; finite states use CVA, explicit lookup tables,
or `data-*` variants. Plain conditional class joining remains appropriate for
non-variant structural state. A static class string must not contain two
arbitrary utilities for the same variant/property scope with different values;
such an override belongs in a CVA branch or an explicit state selector rather
than depending on Tailwind's generated order. Biome parses Tailwind directives
so this sole CSS entry remains inside the repository's normal lint and
formatting gate.

Relationship utilities that escape underscores in BEM class names use static
`String.raw` templates. This keeps the class candidate seen by Tailwind's source
scanner byte-for-byte identical to the class token rendered by React; ordinary
JavaScript string escaping must not introduce an additional backslash. Responsive
utilities use the ordered `max-1100`, `max-860`, `max-680`, and `max-520` custom
variants declared by the stylesheet entry. Their descending declaration order
preserves the original inclusive `max-width` cascade when multiple narrow-window
rules target the same property.

Host-sensitive backdrop filters remain on the existing prefix-controlled CSS
path. Tailwind arbitrary utilities must not add an unprefixed `backdrop-filter`
path independently: the compiled stylesheet consumed by the supported Electron
runtime is the compatibility surface, and enabling a second path changes both
compositing and visible translucency.

Authored residual feature CSS is a bounded exception rather than the primary
styling surface. Ordinary standalone component rules belong in statically
discoverable Tailwind classes beside the rendered element; reusable visual
variants belong in CVA. The architecture test caps all residual Renderer CSS at
3,000 logical lines and rejects standalone single-class rules outside the owned
base stylesheet. Remaining CSS therefore requires selector relationships,
pseudo-elements, rich-content/third-party markup, keyframes, or host/media
queries that are clearer as cohesive rules.

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

Task suitability is a separate passive Core catalog. `agent-guidance.ts` owns
the browser-safe schemas, source metadata, curated records, and deterministic
queries for Model-, Harness-, and exact Agent-targeted guidance. Capability and
Harness registries remain the hard compatibility authorities; guidance cannot
make an unavailable route executable. Each record cites normalized evidence,
the exact benchmark configuration, a review date, and limitations. A benchmark
of an upstream native Harness is marked as indirect evidence for SwarmX's ACP
adapter and never advertised as runtime parity. Missing records are unrated.
The catalog is product-distributed metadata, not user Memory, Activity, or an
evaluation-results store; user-specific conclusions may be admitted to Memory
separately, while raw local benchmark runs keep their own structured evidence.

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
- optional MCP servers and Agent/Swarm lifecycle hooks.

`Swarm` parses the config, materializes nodes into a `Map`, and stores edges as
`Edge` objects. Construction rejects unconditional cycles and warns about
conditional cycles that require an escape condition.

Execution starts at `root`, evaluates outgoing edges after each node, waits for
declared predecessors, schedules a node at most once, and enforces a step bound.
Execution output is an ordered collection of normalized message chunks. Eval
execution additionally records deterministic step metadata and metrics.

Lifecycle hooks preserve the existing `onStart`, `onChunk`, `onHandoff`, and
`onEnd` configuration shape. A hook target is resolved only by an explicitly
injected host capability executor; Core never treats the target string as a
shell command. Matching targets for one event start concurrently, receive a
structured invocation, and have a bounded timeout. Start and handoff handlers
may stop execution or add bounded model-visible context. Missing executors,
timeouts, malformed results, denials, and failures fail closed. End hooks run
for both success and failure, while chunk hooks observe streamed chunks in
order. This follows the event/handler and structured-I/O model used by Claude
Code and Codex while retaining SwarmX's host-authority boundary.

The n8n importer is a boundary adapter into `SwarmConfig`; it is not another
runtime.

## Managed feature modules over MCP

Language-native feature processes use MCP over private stdio as the common
capability boundary. MCP supplies initialization, server identity, tool
discovery, calls, progress, cancellation, structured results, and—where
explicitly granted—client sampling. It does not supply trust by itself. The
owning SwarmX host still verifies the packaged runtime or locked source digest,
starts it with a sanitized environment, requires one exact allowlisted tool
surface, enforces request/response size and timeout limits, and validates the
TypeScript-facing boundary with zod.

These private servers are implementation modules, not user-configured MCP
servers. Their tools are never registered with the Agent-facing `McpManager`.
The host projects only the product capability it has authorized, preserving
confirmation, audit, credential, persistence, and Renderer boundaries. A new
module may follow this runtime pattern without becoming a `SwarmConfig` node or
receiving filesystem, scheduling, Provider, or durable-state authority.

The first modules are:

- `swarmx-mem`, a Rust MCP server with the single private
  `swarmx_memory` tool. Desktop Main owns its lifecycle and projects it to the
  confirmed `Memory` Agent tool. The current linked Markdown organization is
  an implementation behind that generic product contract.
- `swarmx-rsi`, a locked Python MCP server with the single private
  `swarmx_rsi_optimize` tool. The durable Python worker launches it for
  DSPy/GEPA optimization and implements MCP sampling by forwarding only
  grant-checked `model.generate` calls to the host capability gateway.
- `swarmx-ref`, a locked Python MCP server with the single private
  `swarmx_reference` tool. It exposes bounded `status`, `search`, and `get`
  operations over explicitly configured ZIM and Zotero sources. ZIM uses the
  official `python-libzim` binding.
  Zotero uses only the fixed unauthenticated loopback read API and excludes
  attachments and full text. The module strips active HTML, never mutates a
  source, and has no path-scanning or Memory authority.

Provider-hosted Web Search is composed above that private module. When a direct
Agent has credentials for an exact official DeepSeek, OpenAI API, or Codex
Responses endpoint, the native Responses adapter supplies the server-side
`web_search` tool. Responses continuation preserves the opaque hosted-search
item, and each observed hosted invocation is projected as correlated
`tool_call` and `tool_result` events for the Session, activity, and audit paths.
An Anthropic `pause_turn` response is continued with its complete assistant
content and the same tool definitions, without a synthesized user tool-result
message. DeepSeek's official Anthropic endpoint retains its versioned server
tool for models that do not support Responses. The Provider credential remains
inside the Agent request boundary and is never passed to `swarmx-ref`. Endpoint
and protocol matching are strict so a proxy, bridge, lookalike host, or ordinary
Chat Completions route cannot silently gain hosted search behavior. Official
DeepSeek discovery exposes Responses only for models documented to support that
protocol. SwarmX does not provide a local or generic Web Search fallback.

Python is one standard root `swarmx` distribution with a regular
`src/swarmx/__init__.py` package. The worker and the `rsi` and `ref` private MCP
implementations are ordinary subpackages in the same wheel and locked
environment; there is no namespace-package discovery, module entry-point
registry, uv member workspace, or module-specific dependency group. Hosts
launch the explicitly owned `swarmx.rsi.server` or `swarmx.ref.server` module
over private stdio and still validate its MCP identity and exact tool surface.
Rust separately uses a root Cargo workspace with `crates/*` members, so
Node/pnpm, Python/uv, and Rust/Cargo ownership remain visible from the repository
root.

The durable worker protocol remains a control-plane protocol rather than a
feature-module tool surface. Its lease fencing, monotonic event stream,
heartbeats, checkpoints, human suspension, artifact receipts, and uncertain
external-effect receipts are authoritative WorkItem semantics. An MCP
transport could carry those messages later, but replacing the framing cannot
remove or transfer those semantics. Feature work belongs in a private MCP
module; task authority remains in Core and the supervisor.

## Memory organization and knowledge graph projection

The current Memory organization follows the LLM Wiki pattern of interlinked
Markdown entity pages. SwarmX exposes only the smallest reusable graph
primitive from that pattern: a pure, browser-safe projection from one entity's
Markdown and a caller-supplied entity registry to directed `memory_link` edges.
This knowledge relationship is distinct from the executable transition in
`SwarmConfig` and never enters `Edge` or `Swarm` scheduling.

The linked Markdown projection accepts the forms `[[Target]]`,
`[[Target.md]]`, `[[Target#Heading]]`, `[[Target|Label]]`, and `![[Target]]`.
Titles and caller-declared aliases resolve case-insensitively after Unicode NFC
normalization. The resolver never creates a Memory entity: unknown, ambiguous,
malformed, and self references produce bounded diagnostics. Repeated links
collapse into one source-to-target edge while retaining bounded occurrence
metadata, and double-bracket links inside inline or fenced code are ignored.

The pure projection remains browser-safe and has no filesystem access. The
production persistence host is a SwarmX-owned Rust sidecar built against an
exactly pinned `llm-wiki-engine` crate. It stores one Markdown page per stable
entity id under `~/.swarmx/memory/`, commits every accepted mutation to the same
local Git repository, and treats the engine's Tantivy index plus the host's
linked graph as rebuildable projections. Titles and aliases must resolve uniquely, and
updates/deletes/restores require `expectedRevision` so a stale LLM or client
cannot silently overwrite newer knowledge. There is no JSON fallback, legacy
importer, or second persistence authority.

Create, get, list, BM25 search, update, recoverable delete, history, version
read, diff, and restore all cross zod schemas. A successful mutation means the
Markdown write, validation, Git commit, and index refresh have completed as one
semantic lifecycle. Markdown plus Git remains authoritative if index refresh
fails, and reopening the runtime rebuilds that projection before serving.
`graph()` is derived from indexed current pages and edges are never a second
authority. A delete writes a content-free
tombstone so history remains recoverable; restore creates a new revision from a
selected historical version rather than moving Git HEAD backwards.

SwarmX-owned native Agent execution receives one `Memory` local tool with
strict list/get/search/graph/history/get-version/diff/create/update/delete/
restore operations. Reads are bounded. Every create, update, delete, or restore
request is brokered to the owning Renderer for one-time confirmation, and its
semantic audit event excludes title, aliases, Markdown content, and diffs.
Custom ACP Harnesses keep ownership of their tool surface and never receive
this local tool.

The sidecar deliberately does not call an LLM, ingest source files, admit
claims, inject all pages into model context, provide vector retrieval, or render
Markdown. The Desktop host grants bounded on-demand access and retains
responsibility for approvals.

### Memory runtime boundary

`@swarmx/runtime` owns read-only Memory executable inspection and returns either
a verified launch description or an explicit missing/invalid/repair state. The
description pins the platform target, version, protocol version, executable
path, and SHA-256 digest. Desktop Main recomputes the digest immediately before
launch, passes an explicit credential-free environment, and owns one lazy Memory
process for the app lifecycle. There is no `cargo install` path and an ordinary
Memory operation never downloads or repairs code.

The host talks to `swarmx-mem` through a private MCP-over-stdio connection
and requires its only tool to be `swarmx_memory`. Core zod schemas validate
every request and operation-matched structured response; malformed, oversized,
mismatched, text-only, or contradictory responses fail closed. The existing
public `Memory` local tool is the only Agent surface, so mutation
confirmation and body-free audit remain host-owned. The Memory server is not a
WorkItem executor: it has no lease, checkpoint, artifact, capability-grant, or
detached-supervisor semantics, and it stops when Desktop exits.

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
and external side-effect receipt. Engineering- and analysis-specific lifecycle
schemas are deliberately outside Core; downstream adapters may map domain
states onto the generic kernel without introducing a second lease or replay
authority.

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

The Context Engine compiles a bounded model-visible projection from a fixed
event snapshot. Canonical Session and WorkItem logs remain the truth. Its
standalone EventStore uses SQLite WAL for reproducible replay, while a JSONL
implementation remains available for fixtures and controlled comparisons. Local
content-addressed artifacts may externalize large observations without deleting
their source event, hash, exit status, or bounded capsule. Normalization keeps a
tool call and its result in one indivisible unit. Deterministic masking,
structured task state, lexical evidence, and context manifests are rebuildable
projections and never become execution authority.

The engine uses SQLite WAL, deterministic masking, rule-derived sourced state,
BM25 retrieval, a priority-and-slot assembler, and deterministic evidence
verification. Repository, filesystem, branch, and test freshness must be
supplied as current observations at assembly time rather than inferred from an
old summary. The assembler records the exact snapshot, configuration hash,
model version, included event ids, and token estimates. Mandatory content that
cannot fit produces `ContextOverflow`; Provider adapters must not truncate it.

Named profiles are immutable policy recipes over this shared compiler, not new
history authorities. Open-source harness profiles reproduce publicly visible
selection, prompt, and tail rules where those rules are portable; the Claude
Code profile is explicitly behavior-level because its runtime is closed, and
the Codex profile covers the public local compactor rather than the opaque
hosted `/responses/compact` service. Paper profiles implement Lossless Context
Management, Parallel Context Compaction, and ReSum using the same atomic event
and manifest contracts. An injected `ContextSummaryProvider` has model-call
authority but no persistence or tool authority. Preflight never calls it;
finalization may call it with a bounded, source-identified transcript and a
request cancellation signal. Missing or failed summarization either fails
explicitly or activates the configured deterministic source-linked fallback,
which the manifest records.

Lossless Context Management additionally exposes read-only `context_search`
and `context_read` tools over the immutable compile snapshot. These tools can
recover exact source text but cannot mutate canonical history or expand host
authority. Recursive Language Models remain a separate future runtime because
their programmable REPL and recursive inference calls require a sandbox,
budgets, and authorization that a projection policy does not possess.

Context-policy evaluation is a separate Core layer over these projection
contracts. A versioned suite expands a bounded profile/parameter matrix, clones
one immutable case and in-memory action environment for each arm, randomizes
arm order from a declared repetition seed, and executes arms sequentially.
The continuation Agent receives only the Context Engine, one simulator tool,
and trusted current-state observations. Evaluation rejects MCP servers, hooks,
external Harnesses, hosted Web Search, and real Project tools. A separately
configured summary Agent receives only one bounded summary request and no tool
authority.

Scoring uses simulator final state and content-free action receipts as the
primary oracle. Exact output fragments are limited to identifiers or constraints
whose verbatim retention is the construct. Unsafe actions or protected-state
mutation globally zero a completed run; Provider/infrastructure failures remain
distinct from profile overflow or summary-policy failure. JSONL records contain
hashes, ids, score evidence, manifests, usage, cost, and latency but never raw
history, prompts, responses, tool output, credentials, or state values.
Leaderboards aggregate only interpretable runs while reporting both strategy
and infrastructure failure rates. Optional bounded hill-climb rounds
round-robin neighbors from each Agent's best profile configuration, rerun the
canonical baseline in every search round for same-round deltas, and never
self-promote a production default. The
CLI exclusively reserves any requested 0600 JSONL artifact before model calls;
stdout reports and failures remain content-free.

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

The root Python project is the sole `swarmx` distribution. DSPy, MCP, and libzim
are normal locked product dependencies shared by its worker, RSI, and Reference
subpackages; only Inspect evaluation tooling remains isolated in the opt-in
`inspect` dependency group. The environment digest covers the project metadata,
lock, worker and explicitly monitored module sources, `uv`, Python
implementation/version, platform, and architecture. Changing from grouped to
the unified environment changes the digest schema so incompatible checkpoints
fail closed.

`AppAttachedTaskControlService` remains the in-process control primitive, but
Desktop hosts it in one on-demand detached local supervisor. The supervisor owns
active run controllers, leases, recovery, cancellation, and human decisions;
Electron is only an authenticated client and may exit without ending an active
eligible worker. Requests cross a strict bounded JSONL socket protocol and use a
random token stored with mode `0600`. Renderer receives only list, cancel, and
decision IPC; worker launch specs, the token, event-store authority, and process
creation remain in Main/Core. The supervisor reuses the sole canonical event
store and never creates a second task format. Once it accepts a strictly
validated run request, it retains that launch and grant recipe only in memory
until the WorkItem reaches a terminal or blocked state. A retryable worker
failure or an approved human pause is redispatched automatically with the same
recipe, while the controller rechecks the attempt budget, fencing, checkpoint
identity, and environment digest on every Run. The recipe is not persisted,
returned to Renderer, or reconstructed after the supervisor itself exits.

The local supervisor is not installed at login and does not yet provide remote
execution, production capability gateways, or OS isolation for untrusted
workers. Its detached lifecycle covers active credential-free WorkItems after
Desktop closes; reboot/login activation and sandbox expansion remain separate
explicit work.

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

## Personal Memory

Personal Memory is a single user-edited record in
`~/.swarmx/settings.json`, separate from Activity Profile, Session history,
Project context, Agent profiles, Skills, and WorkItems. Its schema rejects empty
writes, control characters, unknown IPC fields, and content beyond 4,000
characters. Forget is a dedicated confirmed mutation that removes the record;
deleting or archiving a Session never changes Memory.

Main reads one immutable snapshot for each Agent-bearing run. Core serializes
that snapshot into a dedicated read-only instruction block before native
Provider execution, into the explicit Agent-instructions section of ACP prompt
text, and into each Agent node of a `SwarmConfig`, including nested swarms. A
tool-only workflow reports no consumer. The snapshot can be sent to the selected
Provider or Harness as required model input, but is never copied into audit,
Activity, trace, telemetry, hook input, or unrelated tool transport.

Direct Agents receive a host-owned `PersonalMemory` mutation tool. Its strict
input can propose `save` or `forget`; Main always asks the owning Renderer for a
one-call confirmation and writes a secret-free audit intent before applying the
settings mutation. Denial and lost Renderer ownership fail closed. The active
Agent keeps its frozen starting snapshot, so a confirmed edit affects only later
runs. ACP Harnesses keep their native tool surface and do not receive this tool.

Each attempted Agent Composition run publishes and persists a concise Session
message stating `used` or `not used`, the Settings source, and either a bounded
preview plus snapshot size/update time or a reason such as empty Memory or an
execution path. This receipt is deliberately excluded when
Session messages are rebuilt for subsequent model calls, so it remains UI
provenance rather than a second context source. Full Memory remains inspectable
only through the dedicated Settings IPC surface.

## Sessions and Projects

Canonical Sessions follow a Claude Code-style Project layout under
`~/.swarmx/projects/`. Each working directory maps to a stable, collision-safe
child directory containing append-only Session JSONL event logs and its own
rebuildable `sessions-index.json`. Sessions without Project context live in the
reserved `__recents__` child directory. Project directory keys derive from persisted
Session context rather than the mutable Project bookmark registry, so renamed,
removed, or temporarily unavailable Projects do not hide history. Events create
a Session, append or replace messages, and update metadata.

Replay accepts one torn, unterminated final record as a recoverable crash tail.
A complete malformed record fails closed. Session discovery, replay, mutation,
and indexing accept only `.jsonl`; older `.json` files are unsupported. A
pre-Project-layout JSONL file under `~/.swarmx/sessions/` remains readable during
the layout transition and moves atomically to its canonical Project directory
on the next mutation. Read-only discovery never performs that move.

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

Renderer styling follows the single Tailwind/CVA compilation boundary defined
above. Feature-owned CSS is retained for cohesive layout, relationship
selectors, pseudo-elements, rich content, and host integrations; orphaned
selectors and parallel primitive variant paths are removed.

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

Auto mode uses three ordered tiers. Explicit allow, ask, and deny policy is
resolved first. Read-only tools and bounded edits inside the active Project keep
their deterministic fast path. Every remaining one-call request is reviewed by
a separate, tool-free model invocation before the action starts. The classifier
receives bounded user messages and the pending executable tool payload; it does
not receive assistant prose or prior tool results, which prevents either source
from arguing the classifier into approving an action. Model output crosses a
strict JSON boundary and can select only an offered `allow_once` option. Any
other verdict, unavailable reviewer route, timeout, malformed response, or
unsupported option falls back to the ordinary human prompt. The static policy
and platform sandbox remain authoritative regardless of the model verdict.

An automatic decision is persisted and audited as model-made before the tool
effect begins. Receipts contain only bounded provenance such as source, tool,
risk class, and reviewer model; prompts, executable payloads, classifier prose,
and tool results are never audit or settings data.

Auto is the immediate product fallback whenever no layer declares a mode. A
Session persisted as `inherit` therefore resolves through Auto regardless of
when the Session was created; Session creation does not rewrite `inherit` into
an explicit mode. A mode declared by managed, Project, personal, Agent, or
conversation policy remains unchanged, and disabling the Auto profile still
applies the existing safe degradation. Repository-controlled policy cannot opt
itself into more authority. External vendor rollout dates are research context,
not runtime feature gates.

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

The OpenCode Go catalog is discovered from its authenticated Models endpoint,
but that endpoint does not advertise the wire protocol for each Model. Main
therefore projects documented Go model ids onto their native Anthropic
Messages, OpenAI Chat Completions, or OpenAI Responses route, augmented only by
runtime-verified compatibility exceptions such as DeepSeek V4 Flash supporting
both Chat Completions and Messages. An unknown discovered id receives only the
user-selected preferred protocol instead of being advertised across unverified
routes.

An official OpenRouter connection is one Provider with three API entrypoints:
Anthropic uses the `/api` base while OpenAI Chat and Responses use `/api/v1`;
model discovery always uses `/api/v1/models`. Either accepted base form is
normalized into that routing table, so an Anthropic SDK never produces a
duplicated `/v1/v1/messages` path. OpenRouter API keys use Bearer authentication
for all three routes.

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
- **Durable Python worker** owns no durable state. It obtains exactly the three
  granted baseline/train/dev artifacts, records WorkItem progress/checkpoints,
  launches the locked RSI MCP module for DSPy/GEPA, and writes the returned
  immutable candidate into the granted artifact root. The dependency-free
  deterministic fake remains a worker test path.
- **Python RSI MCP server** runs the DSPy/GEPA optimizer from the locked
  `swarmx.rsi` subpackage in the standard `swarmx` distribution behind
  one `swarmx_rsi_optimize` tool. It
  receives only the three verified artifact bodies, and any reflection/model
  request uses MCP sampling. The worker-side MCP client maps sampling to the
  grant-checked capability gateway; Provider credentials never enter either
  Python process. Neither process can move the active pointer, decide
  promotion, read `provider-auth.json`, scan Sessions, or write the Skill
  install directory.
- **Evaluation** runs baseline and candidate through the same real SwarmX
  execution path on a hidden holdout; Inspect produces independent evidence,
  Core computes the gate verdict.

### Trust boundaries

The optimization request names only baseline and train/dev content digests —
a holdout ref is rejected by the strict schema before the WorkItem is created.
The optimizer worker receives exactly three granted artifact refs. It resolves
and digest-checks them before the private RSI MCP call, while MCP sampling is
mapped to a grant-checked `model.generate` capability whose credentials are
resolved inside a host-owned handler and never cross either protocol.
Candidates are
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
digest covers the worker source, the `swarmx.rsi` server/client and optimizer
sources, pyproject, `uv.lock`, the resolved Python version, and the
interpreter's installed `dspy` and `mcp` versions verified against the pinned
root `swarmx` project. Reflection/model calls cross MCP sampling and then the
grant-checked capability gateway; the CLI only enables
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
