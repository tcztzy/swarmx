# Durable task runtime

SwarmX has a language-independent durable task control plane in TypeScript and
a dependency-minimal Python worker as its first backend. A durable task is a
`WorkItem`, not a Session, Agent, Harness, ACP Session, shell process, or
`SwarmConfig` node.

## Authority and lifecycle

Core owns all authoritative state. A WorkItem points to an executor backend and
operation, while each attempt is a `Run` protected by a monotonically fenced
lease. Workers receive one leased run, execute it, and report events; replacing
or restarting a worker never transfers scheduling or persistence authority to
that process.

The event store defaults to:

```text
~/.swarmx/task-runtime/events.jsonl
~/.swarmx/task-runtime/blobs/<sha256>.blob
```

Events are appended and fsynced, then replayed into current WorkItem, Run,
checkpoint, artifact, approval, link, and receipt state. One unterminated crash
tail can be inspected and explicitly truncated. A complete malformed record,
event-id collision, or idempotency-key collision fails closed.

`AppAttachedTaskControlService` is the runnable controller in this release. A
host calls `recoverOnStartup()` to repair a torn tail and mark expired leases,
then may retry eligible work within its attempt budget. A run does not become
active or consume an attempt until its lease is durably acquired, so a crash
after only `run_created` leaves the WorkItem safely runnable. Startup also
repairs a crash between a retryable failure and its retry-scheduled event. The
service does not silently start all queued work and does not install runtime
dependencies.

Cancellation is also event-first: Core records `cancel_requested` before it
signals the worker. A cooperative worker returns `canceled`; an unresponsive
process is terminated after its grace period. Retry is a control-plane decision,
not a worker decision.

## Sessions and checkpoints

A creator Session and any number of observer Sessions can link to the same
WorkItem. These links are navigation/provenance metadata only. Switching,
unlinking, or archiving a Session does not cancel the WorkItem, and Session
replay is never used to reconstruct task state.

There are two deliberately different checkpoint concepts:

- A task-runtime checkpoint is executor state with a sequence, parent link,
  content reference, checksum, and verified environment digest. It can resume a
  compatible worker run.
- `ContextPacket` and `SummaryCheckpoint` select or summarize model context.
  They cannot resume execution and never replace the task event log.

Core parses the content-addressed resume payload again and verifies its schema,
checkpoint identity, metadata checksum, and environment digest. The checkpoint
metadata, payload, new Run, protocol `start`, and verified launch environment
must agree before execution resumes.

## Worker protocol

Protocol version 1 is newline-delimited JSON over stdio. Core's strict Zod
schemas are authoritative. The handshake and run flow are:

```text
worker: hello
host:   capabilities
host:   start
worker: heartbeat | progress | checkpoint | artifact | capability_call
worker: needs_human | complete | fail | canceled
host:   capability_result | cancel
```

Messages are direction-tagged, versioned, bounded to one MiB, and reject extra
fields, unsafe artifact paths, inline secret-shaped fields, and invalid unions.
Every worker run event repeats the WorkItem, Run, lease, and fencing identity
and has an exact monotonic sequence. Core rejects messages from a stale or
different lease.

The subprocess host starts a worker with an explicit sanitized environment. It
does not copy ambient Provider credentials. Tool or model access must cross a
grant-scoped capability gateway. The Main-owned gateway may resolve a secret for
an authorized request internally, but it returns only protocol-safe results and
receipts to the worker.

External effects use at-least-once delivery. The gateway records an `uncertain`
receipt before dispatch and a `committed` receipt only after durable success
evidence. A committed receipt stores the validated result needed for idempotent
replay; the same key cannot be rebound to another WorkItem or capability and
does not disclose the original result across that boundary. If the process or
connection is lost between those points, the outcome stays unknown. SwarmX does
not claim exactly-once execution.

This protocol is independent of ACP. External ACP Harnesses continue to own
their tools, authentication, Sessions, and permissions.

## Python backend

The reference Python 3.11+ worker currently provides a minimal vertical slice:

| Operation | Behavior |
| --- | --- |
| `swarmx.echo` | Emits heartbeat, progress, an execution checkpoint, and the input result. |
| `swarmx.count` | Emits per-step heartbeat, progress, and resumable checkpoints; supports cooperative cancellation. |
| `swarmx.fail` | Produces a requested retryable or non-retryable failure. |
| `swarmx.needs_human` | Checkpoints, persists the full request, and can resume with a bounded human decision payload. |

The worker is dependency-free and does not currently call the capability
gateway itself. The gateway protocol is present as the boundary for future
model/tool-backed operations.

`@swarmx/runtime` inspects three product assets (`pyproject.toml`, `uv.lock`, and
the worker source), `uv`, a compatible uv-managed Python, and the locked worker
environment. Inspection is read-only and uses offline/no-download checks. The
environment path is derived from a digest covering those assets plus `uv`, the
Python implementation/version, platform, and architecture.

When ready, task execution launches the verified environment interpreter
directly; it does not invoke `uv` on the run path. The launch method reruns the
read-only health check, rejects a changed digest, and passes a hash-verified
snapshot of the worker source to isolated Python rather than reopening a mutable
source path. Missing Python or a stale environment produces an explicit
setup/repair plan for a separately confirmed host action. The root Python
project is named `swarmx`, has no required product dependencies, and keeps
Inspect-only evaluation packages in the opt-in `inspect` dependency group.

## Current boundary

Durable state survives a controller or Desktop restart, and expired leases can
be recovered on the next startup. Execution does not continue after Desktop
closes: there is no independently supervised `swarmxd`, launchd, or systemd
service yet. Desktop IPC/Renderer integration, a background orphan-lease
sweeper, production capability adapters, artifact-backed checkpoint
materialization, asynchronous store isolation from Electron Main, and stronger
OS-level worker sandboxing remain explicit roadmap work. Active app-attached
workers do have heartbeat and wall-time watchdogs; they do not constitute an
independent service. The bundled Python operations do not read Provider auth and
receive no Provider credential variables, but the current same-user process has
not yet been placed in an OS/filesystem sandbox. Strong isolation from the
Main-owned Provider auth file is therefore a required boundary before untrusted
or third-party worker operations can run.

`WorkspaceShell` is intentionally not reused for this purpose; it remains a
temporary interactive execution surface.
