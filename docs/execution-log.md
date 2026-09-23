# Execution journal

The Host owns one private SQLite execution journal at `$SWARMX_HOME/logs/execution.sqlite`
(the default product home is `~/.swarmx`). It records operations observed through SwarmX;
it does not claim to capture hidden model calls, activity outside SwarmX, or earlier native history.

## Record contract

- Records have a storage schema version, monotonically increasing sequence, immutable event ID,
  execution-directory identity, session, execution ID, optional causing event ID, attributes, and an AG-UI event.
- Existing AG-UI schemas define run, text, reasoning, tool, and RAW events. SwarmX-specific
  interaction, steering, interruption and membership facts use namespaced CUSTOM events.
  The local record envelope is a storage contract, not a replacement network protocol.
- Delegation uses the causing tool event. AG-UI `parentRunId` is reserved for conversation
  branching and is not used to mean sub-Agent delegation.
- Requested model/effort are snapshotted at dispatch and are the shared configuration identity.
  Explicit native model/effort conflicts fail execution; original native reports remain in RAW
  records without another reported-configuration projection. Response IDs and Harness versions
  are recorded when supplied. Attributes use OpenTelemetry GenAI names where applicable and
  `swarmx.harness.*` for Harness identity. Missing information remains absent or null; omitted
  effort never implies a default level. Menu changes without dispatch are not executions.
- RAW records retain the JSON values exposed by native callbacks, including unknown fields.
  The journal does not compact, truncate, sample, or delete them. Transport credentials and
  process environments are not added to records. Native payloads may contain private content.
- SQLite transactions with WAL and FULL synchronization commit before dispatching an operation
  or delivering an observed event/result. Write failures propagate. UPDATE and DELETE triggers
  protect journal rows; this is not tamper-proof storage against the filesystem/database owner.
- Trusted Host publications can supply a directory-scoped idempotency key. Identical replays
  return the original event; changed payloads or attribution reject. Work feedback uses this
  after committing its domain fact, so restart can finish interrupted publication without
  duplicating the feedback. Its cause, run identity and learning grant come from the original
  execution; runtime completion and task acceptance remain separate facts.
- A start without a terminal record means incomplete/unknown. Opening the journal never invents
  a successful outcome, retries an operation, or rewrites a crashed execution.
- `swarmx.session.created.value.permissions` records a managed session's creation grant, including
  empty Codex/Claude sessions. `RUN_STARTED.input.forwardedProps.permissions` records every admitted
  turn's effective grant. Their intersection is the persisted conversation ceiling; reads are scoped
  by execution directory and native session ID, with no pagination limit. A missing legacy grant is established
  on first managed execution; an invalid recorded grant fails closed. No parallel permission store or
  mutable copy of native conversation history is introduced.

## Coverage and correlation

DSH task listing and output replay use these directory-scoped records, without starting its SDK
or reconstructing native sessions. The composer accepts one execution per DSH task; new work
requires a new task. Logs remain available after Host restart.

Native Agent wrapping is below desktop AG-UI, ACP, A2A and recursive Swarms. It records user input,
requested configuration, native output, tool events, interaction requests/replies, errors,
steering and interruption requests. Native callbacks are recorded before consumer callbacks.
Sensitive interactions (including Codex `isSecret` questions and Hermes sudo/secret input) record
the request and answer status with `redacted: true`, without recording the answer value. The live native request still receives
the actual response.
Control completion/failure records reference the original steering or interruption request.
An interruption request alone does not prove that the Harness stopped; the UI distinguishes
pending requests and runs that ended after such a request from confirmed successful completion.
New RUN_FINISHED records retain the public stopReason values. Missing native terminal results
are errors, never successful finishes. Legacy interruptionRequested records remain readable.
Native resume and UI history hydration continue to use the Harness's own API.

Product tools record their inputs and returned results in the same journal. Science results retain
their entity IDs, revisions and journal locators; Memory results retain their own concept references.
The domain stores and execution journal are separate transactions. If a domain operation commits
and result logging fails, the caller sees a failure and the started operation remains inspectable;
this is not exactly-once external execution and no automatic retry is performed.

Each supported native integration receives a random product-tool credential over the Host
stdio bridge. The Host binds that credential to the native session and Host run before prompting,
then revokes it at the execution boundary. Later turns get a new credential for the existing
native conversation. Revoked or unbound credentials reject before product-tool dispatch. The Host
does not derive authority from native metadata or event timing. Original native events are retained;
an unavailable native version is not inferred from the Host or SDK version.
Child credentials authorize only their own bridge connection; the general Host token is not sent
to child runtimes. Presenting a child token elsewhere cannot convert it into a Host-bound call.
Native tools keep their own configuration and authorization; Host grants are checked independently.
Unbound bridge calls are rejected. A delegated child run links to
the initiating product-tool event. Swarm membership changes are recorded even though live membership
still belongs to the current Host process.

The `logs.read` Electron bridge operation reads records for the execution directory, ordered by sequence.
It accepts `after`, `limit`, `session`, and `run`; `descendants=true` with `session` also follows
causing-event links into delegated executions. Filtering by cursor happens after discovering
descendants, so paginated reads retain their ancestry. Pagination uses `nextAfter`. The response's
`activeRunIds` is a live Host snapshot, not a persisted outcome; it is empty after restart. The journal can also
be read without launching any Harness. These private records are not automatically included in
public RO-Crate exports. Back up the SQLite database consistently, including committed WAL data.

## Evaluation evidence

`logs.evidence {sources}` resolves 1–64 `urn:swarmx:execution:<UUID>` references in the current
execution directory without starting a Harness. Missing or foreign references reject. The
same resolver validates evaluation writes and supplies Memory lint diagnostics. Ordinary
citations return their exact records and summarize the associated run lifecycles. Each summary
retains its lifecycle source IDs. Review-snapshot citations expand only the original records
saved in that snapshot, validating them against the journal; omitted records remain omitted.
Any omitted record makes a snapshot's recursive costs incomplete: known independent charges
remain visible, but total costs are null and cannot become complete configuration-comparison
samples. This is conservative even when only text was omitted. Reading the full original
execution still computes its complete cost when all charges are available.
Parent/tool ancestry provides context without becoming another execution sample. The review
queue fixes the batch scope, including when reopening older snapshots.

Statistics use the `swarmx.execution.v1` recipe, distinct run IDs and their time window. A
normal `end_turn`, error, cancellation, other stop, and absent terminal record are separate
outcomes. None establishes task correctness. Median elapsed time measures Host wall time,
including tools and waiting. Token totals cover only runs with both known input and output
counts; cost coverage is counted separately. Unknown values stay null, while known zero stays
zero. These are citation-selected observations, not an unbiased provider error rate.

Normalized usage comes only from unambiguous per-run native reports. Each summary identifies the
execution purpose, parent run, token coverage and money source. Cache input and reasoning output
are subsets of input/output, not additional token charges. `costUsd` is the run's own native query
cost, including internal subcalls already covered by that SDK. `totalCostUsd` adds separately
recorded Host descendants and independent tool costs through existing causing-event links. A
missing native cost makes the total null and `totalCostComplete=false`. Configuration comparisons
use these recursive totals. Statistics sum independent charges once, never overlapping parent totals.

Tool counts and costs derive from original tool events, deduplicated by run and tool-call ID.
`tools` reports `callCount`, `usd` and `unpricedCalls`. A separate `swarmx.tool.cost_usd` charge is
counted once; it must not repeat a fee already in the SDK total. Unpriced tools initially count as
zero while remaining visible as unpriced, including local bash calls. No human-resource registry
or duplicate aggregate ledger is maintained. Known estimates are not cash invoices.

Pi sums finalized main-loop
assistant messages once per turn, including input cache tokens; native output already includes
reasoning tokens. It excludes compaction, helpers, extensions and unreported provider retries.
Unknown/error placeholder usage keeps that run's aggregate unknown. Pi USD values are SDK
estimates, not invoices, and every aggregate retains its usage basis. Claude differences
`modelUsage` and `total_cost_usd` against the cumulative snapshot at the start of the Host run,
within the actual SDK query process. Repeated results and queued native turns use the latest
cumulative report once. A new query process starts its own accounting, including a resumed
conversation; a reset, missing report, invalid count or zeroed error report marks that run's
usage unknown. Positive reported usage survives a failed or cancelled outcome. Claude's totals
include native Task subagents and compaction but exclude helpers outside its query pipeline;
its USD values are native estimates. Both integrations explicitly report partial coverage.
Other unavailable totals remain unknown. Original native reports remain available for inspection;
an SDK that omits effective-model metadata does not prove the backend model's identity.

Codex uses the installed App Server's `rawResponse/completed` notification, which identifies
one upstream response and its exact usage. Reports are scoped to the observed native turn and
deduplicated by response ID. Session cumulative totals and the most recent response counters
from `thread/tokenUsage/updated` are retained without charging them again; native context
estimates and repeated rate-limit updates cannot become new usage. Cache reads/writes and
reasoning are subsets of input/output. Missing or conflicting response usage makes the aggregate
unknown. A later available report for the same response can fill a previously null report within
the active run. Reported tokens survive cancellation or failure; unreported calls stay outside
the partial coverage, and USD remains unknown. Usage metadata with an unspecified monetary unit
is retained as RAW and is not converted to USD.

These semantics follow the matching upstream [response emission](https://github.com/openai/codex/blob/rust-v0.153.4/codex-rs/core/src/session/mod.rs#L4155),
[App Server projection](https://github.com/openai/codex/blob/rust-v0.153.4/codex-rs/app-server/src/bespoke_event_handling.rs#L1176)
and [provider usage conversion](https://github.com/openai/codex/blob/rust-v0.153.4/codex-rs/codex-api/src/sse/responses.rs#L130).

Review events retain the supplied snapshot, complete prompt and its SHA-256 identity. The
restricted reviewer runs through the same recorded Agent lifecycle, with its full RAW callbacks,
usage, control requests and outcomes. It has `swarmx.execution.purpose=memory-review`, links to
the review-start event and source run IDs, and inherits admitted work accounting attributes.
Review executions have no AgentMemory hook and are ineligible for automatic review, preventing
recursive review jobs. Review tool and permission requests still reject, and cancellation before
native dispatch cannot start a later request after the cancelled preparation finishes. The
original model response is recorded before JSON parsing, including invalid replies; response and
validated plan retain the requested reviewer identity. Original native reports remain in RAW records.
Earlier reviews without a response record remain explicitly incomplete in an evidence export.
Selection concepts and prompt/skill proposals link back to that review and the original cited
events. Evidence availability does not certify the reviewer's qualitative conclusion.

## Exported evidence archives

Publication cases and their independent verifiers belong to the separate `swarmx-paper` project.
Evidence exports retain the original closed SQLite database and all `record_json` values exported
in sequence order as UTF-8 JSONL, without reserialization or payload reduction. An independent
reader can verify byte hashes, SQLite/export agreement, sequence and causal references, then
replay records as inert data without a Harness implementation or model connection.

Raw observations are authoritative evidence; reports and UI projections are derived views.
Archive the prompt, frozen code, input identities and results alongside the log. Keep later
recomputations separate from the captured run. Preserve unknown fields and native identifiers;
do not mistake a notification capture for a complete transport trace, or event replay for
re-execution. These case files are an explicit reviewed export, not automatic publication of
the private journal.

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
chunks before indexing. Recall is directory-scoped and returns the original source event IDs;
short queries use literal substring matching. This index is not another authoritative transcript.
Review snapshots and memory proposals remain private and are not automatically exported to RO-Crate.

Focused acceptance covers persistence/reopen, append-only guards, directory/cursor filtering,
unmodified RAW values, requested configurations and native mismatch failures, cancellation requests,
interaction decisions, delegation causality, and authenticated reads independent of native history.
