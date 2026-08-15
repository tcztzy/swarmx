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

### Deterministic Extension composition

Core projects every selected Extension bundle into one normalized composition
node. Identity/version, source, integrity, and trust come from the bundle and
installed revision state. Capability ownership and requested permissions are
derived from the existing manifest arrays; an optional declarative composition
block adds cross-bundle `requires`, `conflicts`, phase, and `before`/`after`
constraints. It contains no executable callback.

Preflight closes required dependencies, then validates the complete selected
graph before any MCP connection, subprocess launch, installation, trust change,
or permission change. One capability has one owner. Duplicate Provider ids,
duplicate tool names, missing requirements, dependency/order cycles, a
later-phase dependency, unresolved order-sensitive peers, or an attempt to
provide a kernel capability blocks the graph. Independent, order-insensitive
peers are ordered by normalized Extension id, making the result byte-stable
without assigning semantic meaning to discovery order.

The protected kernel set covers Session and task authority, identity,
composition enforcement, approval policy, credential storage, audit policy,
Extension trust, and foreground completion. Ordinary Extensions may consume a
host projection of a capability but cannot claim ownership of those concepts.
External ACP Harnesses remain outside Agent-facing Extension tool injection;
their native permission and tool surface is not converted into a duplicate
SwarmX grant list.

The preflight result is a browser-safe preview: load order and load reason,
provided/required capabilities, requested/granted/missing permission ids,
source/trust/integrity, and stable actionable issue codes. It has no clock,
randomness, filesystem write, network call, process launch, or authority-changing
callback. Execution accepts only a ready result for the same normalized input.

## Execution paths

### Direct SwarmX

Core's `Agent` executes supported Provider APIs natively. The selected API mode
and request-scoped environment remain explicit. Native execution preserves
streaming, cancellation, tool continuation, and Provider-specific message
shapes instead of normalizing every request through a compatibility bridge.

The evaluation-only built-in service registry resolves exactly three seams
before a direct SwarmX Agent can run: Context Engine, Memory projection/tools,
and evolved Skill delivery. Every seam selects one named variant through a strict
`AblationProfile`; the shipped `production` variant preserves the supplied
service and the shipped `baseline` variant removes it. Variant resolution
receives deterministic Swarm/Agent topology rather than consulting ambient
registration order. Startup asserts that every selected `(seam, variant)` is
registered and returns one content-free activation receipt. Missing or duplicate
entries fail before MCP startup or a Provider request. Echo and external ACP
Harnesses do not enter this registry. Evaluation may select an explicit profile
and records its profile, variants, and activated topology alongside ordinary
metrics; the registry never persists a new workflow format or promotes a
variant.

Provider-independent local tool contracts live in a browser-safe leaf module;
MCP and native Provider adapters consume that contract instead of owning it.
Request cancellation is a separate Node execution scope. ACP, MCP, native
Providers, and host tools consume its `AbortSignal`, while process-owning
adapters register bounded cancellation and cleanup participants. The scope does
not depend on any protocol adapter.

Desktop may inject host-owned Project tools into a direct SwarmX task. The
selected built-in tool style changes model-facing names and schemas while all
styles dispatch through the same containment, permission, cancellation, and
output boundaries.

### Project-scoped Extension services

An Extension MCP capability declares an activation of `off`, `auto`, or
`required`. `off` is omitted, `auto` preserves the existing best-effort MCP
behavior, and `required` turns connection failure or the ten-second connection
deadline into a pre-Provider error. Required services are supported only by the
direct SwarmX backend; external Harnesses continue to own their native tools.
For `scope: project`, Main supplies an explicit Project id and normalized root;
Core overrides the stdio server working directory with that root and rejects a
required binding when the Agent working directory names a different root.
Manifest-provided environment values remain explicit and are not augmented with
ambient Project, credential, or quota variables.

One selected required Project MCP may additionally declare a Project bootstrap
contract: expected MCP server name and version, exact remote tool names, and the
single read-only bootstrap tool. Connection validates that surface before any
tool is exposed. The bootstrap tool itself remains host-only and is never placed
in the model-facing MCP tool list. After Context Engine preflight succeeds and
required MCPs are connected, each direct Agent execution attempt calls it
exactly once with the Project id and root. It requires matching JSON text and
structured content, validates the returned Project id, and rejects unknown or
oversized state before the primary Provider request. This ordering preserves
the existing rule that impossible context fails before starting external
services while still failing
closed before scientific work begins.

The versioned bootstrap is an immutable per-execution-attempt projection, not
Project authority. It contains only `projectId`, `registryRevision`, bounded
active Run and open Decision references, optional site-profile version, and
coarse storage and quota states. Core renders it as a bounded instruction block
and recompiles any configured Context Engine for final request budgeting.
Session history stores only a receipt with service identity, revision, digest,
counts, and coarse statuses. Core's shared model-replay projection excludes both
Project-bootstrap and Memory receipts from ordinary history and Context Engine
input while retaining them in canonical history.
Provider credential failover starts a new execution attempt and therefore
reloads the authority; Desktop emits an identical request/Agent/digest receipt
at most once. A new run always reloads the service, so separate Sessions
synchronize through the Project authority rather than through one another's
transcripts.

The service contract is deliberately domain-neutral. A biology Extension owns
Sample, Assay, Reference, Method, Artifact, and scientific validation semantics;
a site Extension owns LSF queues, mounts, quota, and GPU policy. SwarmX owns the
Project binding, bounded snapshot, failure semantics, and receipt. Mutation
tools remain ordinary Agent-facing MCP tools and do not gain host permissions,
audit authority, or cross-Project access merely because the bootstrap service is
required.

Legacy and `auto` Project MCPs remain best-effort: without an explicit Project
binding they retain their manifest launch configuration. A host may explicitly
disable all Agent-facing Extension MCPs; Desktop does so for read-only side
chats, skipping optional services and rejecting required ones before a Provider
request. This prevents an MCP tool from bypassing the side-chat mutation
boundary. Multiple required Project MCPs may be selected, but at most one may
be the bootstrap authority.

Version 1 requires the Agent working directory and Project root to be exactly
the same canonical path. Entering a host-managed Git worktree changes only the
execution directory, not persisted Project authority, so a child Agent using a
required Project MCP fails closed until a future contract can separately attest
an authority root and registered worktree execution root.

### External ACP Harness

`AcpClient` launches an ACP-compatible subprocess, initializes it, creates or
loads a Session, negotiates configuration, sends prompts, and consumes
`session/update` notifications.

External Harnesses own their native tools, authentication, configuration, and
permission behavior. SwarmX does not inject duplicate Project tools.

Desktop can wrap external custom Harnesses in a protected container runtime.
Under `protected_required`, every such boundary needs a host-registered profile;
an absent profile or unavailable runtime blocks execution. Native execution is
available only under an explicit `native_allowed` strategy. A protected wrapper
receives an explicit Project mount and allowlisted request environment.

### Generic OS sandbox policy

Executable host boundaries resolve a host-owned `native_allowed` or
`protected_required` strategy before starting a process. Protected profiles are
registered by the host, never accepted from Extension metadata, and are
immutable after validation: they pin an image digest, command argv,
environment-name allowlist, project/temporary/credential mount permissions,
network-deny mode, and CPU, memory, and temporary-space limits. The Apple
Container adapter maps that profile to a read-only image root, one writable
Project mount, bounded tmpfs, no DNS, and the no-network network selector.
If the runtime or profile is unavailable, the protected path returns a blocking
diagnostic and never uses the native command. Doctor and Runtime Settings
surface both the requested strategy and the observed execution mode.

### ACP server

`@swarmx/acp-server` presents SwarmX as an ACP agent. A persisted Core Session is
the conversation authority. Advertised cwd, resources, MCP support, history,
and cancellation must match implemented behavior.

### Foreground completion barrier

A foreground request is one host-owned turn. On a native Provider path, each
Provider response plus any tool results that it requests is one execution step.
An external ACP Harness may perform its own internal steps, but SwarmX treats the
terminal response to the one ACP prompt as that ownership boundary. These terms
describe runtime ordering only; they add neither a persisted workflow format nor
a second Session authority.

Native tool calls, host permission decisions, child-Agent calls, and lifecycle
hooks are admitted synchronous obligations. Their promises must settle before
the next Provider step or foreground completion. Child Agents return through
the parent tool call and have no independent late-report channel. Tool lifecycle
chunks use `invocationId`; the foreground request uses `requestId` for live IPC,
cancellation, activity/audit correlation, and the final `messages_appended`
Session event. A `Swarm` node becomes an obligation when it enters the scheduler
queue and remains one until that node settles.

Success crosses the completion barrier only when the selected execution adapter
has returned a terminal result, all admitted synchronous obligations have
settled, and the workflow queue is empty. Exhausting a Provider continuation or
workflow step bound with work still owed fails explicitly. Cancellation and
failure may terminalize observed tool presentation state, but do not satisfy the
success barrier.

At the barrier, Main stops accepting foreground chunks and closes the live chunk
publisher. For a Session-backed request, Main first appends the user message
together with a `started` receipt containing a normalized request digest. Only
then may Provider, tool, or ACP execution begin. The terminal message batch and
a matching `settled` receipt (`completed`, `canceled`, or `failed`) are appended
and fsynced before IPC success is returned. Reusing `(sessionId, requestId)` with
the same digest replays the persisted terminal batch; a `started`-only receipt
returns `REQUEST_OUTCOME_UNKNOWN`, a digest mismatch returns
`REQUEST_ID_CONFLICT`, and an active request returns `REQUEST_ALREADY_ACTIVE`.
The Renderer may show the optimistic user message, but does not call ordinary
`saveSession` for it. Session-less requests and transient side chats remain
explicitly non-durable. An ACP `session/update`, local-tool progress callback,
or other event arriving after the adapter's terminal result is dropped; it
cannot mutate the finalized batch, append a follow-up turn, or reopen the
completed request.

Background Session activations, scheduled tasks, and durable WorkItems are
separate executions with explicit ownership and terminal records. They are not
hidden obligations of a foreground request. This design deliberately does not
add a generic Inbox or make every subsystem a plugin: the existing synchronous
boundaries remain the smallest enforceable settlement model, while the durable
task runtime keeps its stronger lease, receipt, replay, and post-terminal rules.

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
Reaching that bound with a scheduled node still queued fails the foreground
completion barrier. Execution output is an ordered collection of normalized
message chunks. Eval execution additionally records deterministic step metadata
and metrics.

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
exactly pinned `llm-wiki-engine` crate. `~/.swarmx/memory/` is both the
persistence root and a directly openable Obsidian vault. Active entity pages
live recursively under `pages/` at portable human-readable paths such as
`pages/Herdr.md`; the stable generated entity id is frontmatter/API identity,
not a filename, and survives a human move or rename. The sidecar commits every
accepted mutation to the same local Git repository and treats the engine's
Tantivy index plus the host's linked graph as rebuildable projections. Titles,
aliases, ids, and active paths must resolve uniquely, and
updates/deletes/restores require `expectedRevision` so a stale LLM or client
cannot silently overwrite newer knowledge. There is no JSON fallback or second
persistence authority.

Before serving an operation, the sidecar reconciles bounded working-tree
changes under `pages/`. A human-authored Markdown file is adopted with its
filename stem as the default title and a generated stable id; body or supported
frontmatter edits advance the revision; moves preserve identity and revision;
and deletions become hidden recoverable tombstones. Unknown frontmatter is
preserved so editor metadata is not destroyed. One reconciliation commit and
index refresh make the same files visible to Agent reads. Legacy
`pages/mem_<id>.md` files migrate once to collision-safe title-derived paths
without changing the stored page, revision, or prior Git versions. SwarmX does
not create or overwrite `.obsidian` settings: any Markdown editor can use the
vault, and Obsidian can open its root without setup.

Create, get, list, BM25 search, update, recoverable delete, history, version
read, diff, and restore all cross zod schemas. A successful mutation means the
Markdown write, validation, Git commit, and index refresh have completed as one
semantic lifecycle. Markdown plus Git remains authoritative if index refresh
fails, and reopening the runtime rebuilds that projection before serving.
`graph()` is derived from indexed current pages and edges are never a second
authority. A delete removes the active human-facing page and writes a hidden,
content-free tombstone so history remains recoverable; restore creates a new
revision from a selected historical version rather than moving Git HEAD
backwards.

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

### Crash-recoverable Memory publication

Memory uses one short-lived write-ahead log under `.runtime/transactions/` and
the Git repository's HEAD as its publication point. A root single-writer lock
is held for the sidecar lifetime. Before touching the working tree, a mutation
constructs its complete in-memory path patch (pages, moves, inbound-link
rewrites, tombstones, globals, and generated views), records before-images and
the intended commit in a durable transaction manifest, then applies each path
with file and directory synchronization. The intended commit is created with
`HEAD` unchanged; a locked compare-and-swap advances the current branch only
from the recorded base commit. The index and search engine are refreshed after
publication, and the WAL is removed only after those projections are durable.

Startup recovery precedes migration, external-edit reconciliation, and search
initialization. If HEAD is still the manifest base, every affected path must
be either its before-image or intended image and recovery restores the before
state. If HEAD is the intended commit, recovery completes the intended Git
tree. Any other HEAD or path contents are an unknown external state: the WAL
is retained and the runtime fails closed without overwriting it. Recovery does
not create another commit, so reopening twice is idempotent. There is no
best-effort rollback fallback outside this transaction model.

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
whose verbatim retention is the construct. A prohibited action that the
simulator blocks is a contained attempt: it lowers action efficiency and is
reported separately, but it does not fabricate a state mutation or globally
zero capability. Delivered forbidden output or actual protected-state mutation
is an uncontained safety violation and globally zeros safety-adjusted quality.
Provider/infrastructure failures remain distinct from profile overflow or
summary-policy failure. JSONL records contain hashes, ids, task-family ids,
score evidence, manifests, usage, cost, and request-to-completion time but never
raw history, prompts, responses, tool output, credentials, or state values.
Leaderboards aggregate only interpretable runs while reporting both strategy
and infrastructure failure rates. Optional bounded hill-climb rounds
round-robin neighbors from each Agent's best profile configuration, rerun the
canonical baseline in every search round for same-round deltas, and never
self-promote a production default. The
CLI exclusively reserves any requested 0600 JSONL artifact before model calls;
stdout reports and failures remain content-free.

Summary-prompt evolution is an evaluation-only overlay, not another named
Context Engine profile. A suite may bind candidate prompt text and optimizer
provenance to exactly one existing model-backed baseline profile. Candidate and
baseline arms share the same immutable profile configuration; only the prompt
material carried by `ContextSummaryRequest` changes. The summary checkpoint and
arm receipt bind the effective prompt digest, while JSONL records omit the raw
prompt. Confirmation suites declare deterministic gates over paired capability,
constraint retention, pass rate, prohibited-attempt regression, uncontained
safety violations, strategy/infrastructure failure, total tokens, and Agent
completion time. Confidence intervals resample declared task families, not
case-seed rows. Completion time starts immediately before the continuation
request and ends only when the complete streamed Agent call resolves; its
primary statistic is the geometric mean of within-pair ratios. Median and p95
ratios remain descriptive because Provider timing is unstable. Gate eligibility
is evidence for human review and cannot mutate a profile or application default.

Context compilation records two configuration identities. `sourceConfigHash`
binds the selected profile and declared experimental parameters before request
budgeting. `configHash` binds the effective configuration after the actual
Provider input window, output reserve, and slot ceilings are applied. Context
evaluation compares an arm with `sourceConfigHash`; comparing with the effective
hash would incorrectly reject every arm whose runtime window differs from the
factory fallback.

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

Extension trust or permission expansion uses a semantic `extension.authority`
decision in addition to the transport boundary. Its intent contains only the
Extension id, change kind, and permission counts/ids that already passed the
bounded identifier schema. The intent must be durable before settings change.
Audit failure therefore leaves persisted trust and grants unchanged. A terminal
audit failure after an effect is reported honestly; it never rewrites the
attempt as success. Permission reduction and trust revocation are explicit,
idempotent reductions and never restore authority as a side effect.

### Boundary taxonomy

Audit actions identify a semantic effect or a stable transport family, not an
individual UI method. Desktop IPC therefore uses the single transport action
`ipc.request` and identifies the normalized channel as target
`{ kind: "ipc-channel", id }`. Each channel has one explicit emission policy:

Renderer transport contracts are browser-safe and composed by feature into one
logical registry. A migrated invoke contract owns its argument tuple, result
schema, and audit policy; Main authorizes and audits first, then parses arguments
before the service effect and parses sync or async results before returning.
Main-to-Renderer events are likewise parsed before publication. During the
incremental migration, unconverted channels remain explicitly legacy and may
not also appear in the contract registry.

Project transport is one complete feature slice rather than eight unrelated
handlers. Its browser-safe contract owns all eight invoke tuples and results;
the Main feature router receives only the audited registrar and a Project
service. The service alone owns folder dialogs, canonical Project registry
operations, reveal, and the running-Session gate for task archival. Project
transport types derive from the browser-safe Core Project contract, never the
Node registry implementation. The synchronous bootstrap projection and the
asynchronous Project list share that service, so both register the default
workspace before returning the canonical list.

Workspace inspection is a separate read-only feature slice covering the
workspace root, Git review, one-level directory listing, and bounded text-file
preview. Its browser-safe contract owns those four invoke boundaries, while the
Main feature router alone resolves the requested working directory and adapts
`WorkspaceTools` host results to the narrower Renderer projection. In
particular, the contract enforces a bounded aggregate Git-patch payload and the
host-only read digest is not exposed through IPC. Native file and folder
selection remains a distinct authority-expanding capability and is not part of
this inspection slice merely because it shares the `workspace:` channel
prefix.

Embedded Browser transport is one feature slice with eight invoke contracts and
one owner-scoped state event. The browser-safe contract owns bounded Renderer
DTOs and audit policy, while the Main feature router adapts the owner-scoped
`BrowserHost` service without owning Electron registration authority. Invoke
results and Host-published state events share one explicit bounded projection;
Preload parses the event again before it reaches Renderer code. The composition
root retains the single Browser/Terminal owner-destruction listener, so neither
feature can bypass the other's cleanup; it attempts both Host cleanups even when
the first one fails, then reports the first failure. URL normalization, sandboxing,
permission denial, navigation policy, persistent partition semantics, and view
lifecycle remain Host responsibilities.

Global Memory transport keeps the established `personalMemory:*` channel and
Preload method names as user-facing compatibility surfaces, but its only runtime
shape is the current strict `USER.md` / `MEMORY.md` state and target-aware
save/forget inputs. Task Runtime transport is a separate browser-safe slice with
exactly list, cancel, and human-decision invokes. Main alone constructs those
Supervisor commands; launch specifications, authentication tokens, sockets, and
create/run authority never enter the shared contract or Renderer bridge. Human
decision responses pass an iterative encoded-size, depth, node-count, and cycle
preflight before the recursive worker-payload schema reaches Main services.

Interactive Terminal transport is one feature slice with four invoke contracts
and two owner-scoped events. Its browser-safe contract owns the Renderer request,
receipt, data, and exit DTOs, while the Main feature router delegates only to the
owner-scoped `TerminalHost`. Main validates Host-published events before sending
them and Preload validates them again before notifying Renderer listeners. The
transport contract preserves Host-owned normalization and semantic rejection:
blank working directories, oversized writes, duplicate identifiers, and
non-finite dimensions reach the Host so its fail-closed semantic audit remains
the single record of the attempted effect.

Terminal creation acquires the PTY and its event subscriptions as one resource
unit. If subscription setup fails after spawn, the Host releases every resource
already acquired before reporting the original create failure; cleanup failures
remain visible through a fixed, secret-free diagnostic and never replace the
setup error. A partially created PTY never enters the live owner registry. Once
a Terminal is live, natural exit and explicit owner/app cleanup remove it from
the registry and attempt every subscription/process release even if semantic
audit or one release step fails. Bulk cleanup continues across every matching
Terminal, then reports the first failure after no further resource can be
released.

Both Renderer terminal surfaces share one lifecycle hook for xterm setup,
subscriptions, resize deduplication, buffered input, restart, and cleanup. The
visual panels retain their own markup and themes. Pending creation is distinct
from a live PTY: unmount or React StrictMode generation changes invalidate the
old continuation, and any PTY that resolves afterward is killed exactly once
instead of becoming an ownerless process. Each controller instance is bound to
one working directory, restart is serialized through its live-process kill, and
a failed buffered write keeps the already-created PTY tracked for explicit
retry or cleanup. Async create/write continuations recheck both generation and
visibility before mutating the current terminal or moving focus.

Each audited IPC dispatch owns an explicit semantic-audit receipt. A feature
router passes that receipt only to the Host operation for the current request,
and the Host marks it only after the current operation's terminal outcome is
persisted successfully. Background terminal
exits, owner cleanup, app disposal, and concurrent semantic requests therefore
cannot suppress another request's transport failure. This replaces
process-global semantic counters and keeps validation/authorization failures
visible without duplicating successful Host actions.

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

## Global Memory

Global Memory is the pair of bounded, versioned `USER.md` and `MEMORY.md` files
in the Git-backed Memory authority. `USER.md` owns durable user preferences and
facts; `MEMORY.md` owns compact cross-Project experience. The old
`settings.json` Personal Memory record is read only as a migration overlay and
is removed after a successful `USER.md` save; it is never a second writable
authority. Settings writes and explicit forgets carry the revision observed by
Renderer, so a concurrent Obsidian or host edit produces a conflict instead of
being overwritten.

Desktop Main keeps this authority in a dedicated Global Memory service that
alone composes the Memory backend with the legacy overlay and reflection
cursors. The old settings-backed Personal Memory module remains only for
compatibility and migration; IPC and new Main callers do not route authoritative
`USER.md` / `MEMORY.md` behavior through it.

Main reads one immutable snapshot for each Agent-bearing run. Core serializes
that snapshot into a dedicated read-only instruction block before native
Provider execution, into the explicit Agent-instructions section of ACP prompt
text, and into each Agent node of a `SwarmConfig`, including nested swarms. A
tool-only workflow reports no consumer. The snapshot can be sent to the selected
Provider or Harness as required model input, but is never copied into audit,
Activity, trace, telemetry, hook input, or unrelated tool transport.

SwarmX-owned direct Agents receive the host-owned `Memory` tool. Reads may inspect
global files and entity pages; every save, forget, create, update, restore, or
delete proposal requires one-call confirmation through the owning Renderer.
Denial and lost Renderer ownership fail closed. The active Agent keeps its
frozen starting snapshot, so a confirmed edit affects only later runs. External
ACP Harnesses keep their native tool surface and do not receive this tool.

Each attempted Agent Composition run publishes and persists a concise Session
message stating whether Global Memory was used, its bounded source/size summary,
or why it was unavailable. This receipt is excluded when Session messages are
rebuilt for subsequent model calls, so it remains UI provenance rather than a
second context source. Full file bodies remain inspectable only through the
dedicated Settings surface and requesting Memory operations.

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

### Causal timeline projection

The Session timeline is rebuilt from the ordered Session JSONL records. A
`messages_appended.requestId` anchors a foreground Turn when present; legacy
logs use a deterministic local Turn id and mark its confidence as inferred.
Background Session observer records carry a durable `activationId`; that id
opens a separate system-origin Turn whose messages and correlated audit
evidence remain isolated from the foreground `currentTurn` and completion
obligations. The projector never binds an untagged record to the most recent
background activation by proximity.
Tool lifecycle messages use their existing `render.invocationId` as the
strongest Step correlation. Exact duplicate lifecycle observations collapse in
the projection, while a result observed after its originating Turn retains that
original Turn and is marked late. A tool call without a terminal result appears
as unsettled diagnostic work; it does not alter the completion barrier.

Correlated audit records may enrich the same projection with permission
decisions or host effects. They are referenced, not copied into Session JSONL,
and keep their own semantic authority. The projector assigns deterministic
event, Turn, Step, correlation, causation, and sequence fields and emits only
fixed summaries such as a tool name and outcome. It never includes message
content, tool arguments/results, source, terminal streams, credentials, or
environment data. Replaying the same ordered inputs yields the same projection;
the result is diagnostics, not a second Session or execution state machine.

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

Runtime discovery, diagnosis, repair planning, and setup effects remain owned
by `@swarmx/runtime` and Main. Renderer uses one Doctor feature controller for
both the slash-command/right-panel entry point and Runtime Settings. The
controller owns only transient report, version, install, and confirmation
state; it invokes the finite Preload methods, keeps scoped Harness reports out
of the global environment cache, and lets the latest request own visible state.
`/doctor` and `/setup` only inspect and open the review surface; `--fix`
requests confirmation and never performs a repair itself. A setup or fix effect
starts only from an explicit UI action, and fix always sends
`confirmed: true`.

Doctor represents each finding with a stable classification plus four human
fields: symptom, cause, impact, and next action. A report distinguishes a clean
baseline, optional warnings, blockers, idempotent host repairs, and decisions
that only the user can make. The repair plan lists the exact bounded setup
request and what it changes; inspection and planning never call setup. Runtime
checks are offline by default. Provider authentication and writable Project
observations are supplied by the owning host without moving credential or
filesystem authority into `@swarmx/runtime`.

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

The direct-tool permission adapter is a separate boundary from filesystem,
shell, LSP, browser, and task implementations. It wraps both structured and
text tools immediately before invocation, defaults unknown tools to execute
access, and never calls the underlying tool until policy or one-call approval
succeeds. Automatic review receives the original pending input, while the human
prompt receives only a bounded allowlisted summary: command and patch bodies are
replaced with fixed descriptions and never cross the Renderer interaction
bridge.

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

`SWARMX_EXTENSION_PATHS` and `SWARMX_EXTENSION_ROOTS` only add discovery roots.
The loader records a host-observed source and content digest beside each
discovered bundle. Manifest-declared `local`, `verified`, or `builtin` trust
cannot replace that observation or raise effective trust. Built-ins are the
only executable bundles trusted by their host identity; every other executable
bundle must match an installed immutable revision, enabled state, persisted
trust, and required permission grants before any process boundary is entered.
Declarative metadata remains displayable when this gate is not satisfied.

Installation, update, rollback, trust changes, enablement, and repair are
separate validated actions. Executable UI components are host-registered React
components; manifests cannot deliver inline scripts, HTML, or render functions.

Installed state records the immutable source revision/digest, effective trust,
requested permission ids, and granted permission ids. Discovery and install
start with no grant. A permission-set change is evaluated as a set difference:
removal is an authority reduction, while any addition requires explicit user
confirmation, trusted source state, and audit intent before persistence. An
update intersects old grants with the new request and may lower effective trust;
candidate metadata can never raise either. Revoking trust disables the
Extension and clears grants. Requests attributed to an Extension cannot change
approval policy, credential storage, audit policy, or trust state.

Untrusted declarative metadata may remain visible in inventory. Untrusted
executable capabilities without a complete installed-state match are blocked;
supported third-party code runs through an existing external process boundary
such as MCP, LSP, or ACP with sanitized input and explicit grants. Executable
authority includes custom Harnesses, stdio MCP, LSP, Hook, Command, Software
command, and connector entrypoints. SwarmX does not load arbitrary Extension
code into Core, Electron Main, Preload, or Renderer.

Desktop LSP completion and Agent-facing LSP operations resolve their owning
bundle and rerun this same preflight, including dependency closure. LSP child
processes receive only a host-maintained environment whitelist; Provider
credentials and arbitrary inherited environment variables are excluded.

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

Main-only JSON stores share one mechanical private-file writer that emits the
same two-space JSON plus trailing newline, creates exclusive temporary files,
fsyncs file and directory metadata, atomically replaces the target, and fixes
file permissions to `0600`. The helper owns no schema, read/modify/write queue,
locking, or conflict policy; those remain explicit in each domain store.

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
