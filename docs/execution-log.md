# Execution journal

The Host owns one private SQLite execution journal at `$SWARMX_HOME/logs/execution.sqlite`
(the default product home is `~/.swarmx`). It records operations observed through SwarmX;
it does not claim to capture hidden model calls, activity outside SwarmX, or earlier native history.

## Record contract

- Records have a storage schema version, monotonically increasing sequence, immutable event ID,
  workspace, session, execution ID, optional causing event ID, attributes, and an AG-UI event.
- Existing AG-UI schemas define run, text, reasoning, tool, and RAW events. SwarmX-specific
  interaction, steering, interruption and membership facts use namespaced CUSTOM events.
  The local record envelope is a storage contract, not a replacement network protocol.
- Delegation uses the causing tool event. AG-UI `parentRunId` is reserved for conversation
  branching and is not used to mean sub-Agent delegation.
- Requested model/effort are snapshotted at dispatch. Reported model, response ID and Harness
  version are recorded only when supplied by a native interface. Attributes use OpenTelemetry
  GenAI names where applicable; Harness identity uses the `swarmx.harness.*` namespace.
  Missing information stays absent or explicitly null. Menu changes without dispatch are not executions.
- RAW records retain the JSON values exposed by native callbacks, including unknown fields.
  The journal does not compact, truncate, sample, or delete them. Transport credentials and
  process environments are not added to records. Native payloads may contain private content.
- SQLite transactions with WAL and FULL synchronization commit before dispatching an operation
  or delivering an observed event/result. Write failures propagate. UPDATE and DELETE triggers
  protect journal rows; this is not tamper-proof storage against the filesystem/database owner.
- A start without a terminal record means incomplete/unknown. Opening the journal never invents
  a successful outcome, retries an operation, or rewrites a crashed execution.
- `swarmx.session.created.value.permissions` records a managed session's creation grant, including
  empty Codex/Claude sessions. `RUN_STARTED.input.forwardedProps.permissions` records every admitted
  turn's effective grant. Their intersection is the persisted conversation ceiling; reads are scoped
  by workspace and native session ID, with no pagination limit. A missing legacy grant is established
  on first managed execution; an invalid recorded grant fails closed. No parallel permission store or
  mutable copy of native conversation history is introduced.

## Coverage and correlation

Native Agent wrapping is below browser AG-UI, ACP, A2A and recursive Swarms. It records user input,
requested configuration, native output, tool events, interaction requests/replies, errors,
steering and interruption requests. Native callbacks are recorded before consumer callbacks.
Control completion/failure records reference the original steering or interruption request.
An interruption request alone does not prove that the Harness stopped; the UI distinguishes
pending requests and runs that ended after such a request from confirmed successful completion.
New RUN_FINISHED records include the original ACP stopReason. Missing native terminal results
are errors, never successful finishes. Legacy interruptionRequested records remain readable.
Native resume and UI history hydration continue to use the Harness's own API.

Product tools and browser DVC mutations record their inputs and returned results in the same journal. Science results retain
their entity IDs, revisions and journal locators; Memory results retain their own concept references.
The domain stores and execution journal are separate transactions. If a domain operation commits
and result logging fails, the caller sees a failure and the started operation remains inspectable;
this is not exactly-once external execution and no automatic retry is performed.

Each upstream ACP process receives a random authenticated product MCP endpoint. The Host binds
that endpoint to the current native session and Host run before prompting, then revokes it when
the run ends. Later turns open a new endpoint and resume the existing native conversation.
Stale endpoints and missing execution identifiers reject before product-tool dispatch. The Host
does not derive authority from native metadata or event timing. Original ACP updates are retained,
and the adapter-reported version is recorded separately from an unavailable native harness version.
Child credentials authorize only their registered endpoint; the general Host bearer is not sent
to child adapters. Changing the URL cannot convert a child's credential into a Host credential.
Hermes/OpenClaw keep their external MCP configuration but cannot enter managed execution without
a permission mapping. Unbound HTTP MCP calls are rejected. A delegated child run links to
the initiating product-tool event. Swarm membership changes are recorded even though live membership
still belongs to the current Host process.

Authenticated `GET /api/v1/logs` reads records from the current workspace, ordered by sequence.
It accepts `after`, `limit`, `session`, and `run`; `descendants=true` with `session` also follows
causing-event links into delegated executions. Filtering by cursor happens after discovering
descendants, so paginated reads retain their ancestry. Pagination uses `nextAfter`. The response's
`activeRunIds` is a live Host snapshot, not a persisted outcome; it is empty after restart. This works after a
restart without launching any Harness. These private records are not automatically included in
public RO-Crate exports. Back up the SQLite database consistently, including committed WAL data.

## Protocol basis

- [AG-UI serialization](https://docs.ag-ui.com/concepts/serialization): events and archival/replay concepts.
- [OpenTelemetry GenAI](https://github.com/open-telemetry/semantic-conventions-genai): requested and
  reported generation attributes; this implementation does not install an OTLP exporter.
- [Codex MCP request metadata](https://github.com/openai/codex/blob/main/codex-rs/core/src/mcp_tool_call.rs):
  native call IDs and `x-codex-turn-metadata` correlate product operations with observed turns.
- [RO-Crate](https://www.researchobject.org/ro-crate/): existing public scientific provenance;
private execution records preserve the links returned by product tools.

Memory uses `swarmx.memory.*` CUSTOM events for frozen session context, review snapshots/outcomes,
proposed changes and user decisions. Pending approvals are derived from those records after restart.
A derived SQLite FTS5 trigram index reconstructs observed user/assistant messages, joining streamed
chunks before indexing. Recall is workspace-scoped and returns the original source event IDs;
short queries use literal substring matching. This index is not another authoritative transcript.
Review snapshots and memory proposals remain private and are not automatically exported to RO-Crate.

Focused acceptance covers persistence/reopen, append-only guards, workspace/cursor filtering,
unmodified RAW values, distinct requested/reported models, failures and cancellation requests,
interaction decisions, delegation causality, and authenticated reads independent of native history.
