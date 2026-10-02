# Native integrations

SwarmX orchestrates external Agents. Codex App Server is the default; an explicit ACP
endpoint may select `acp` after Host policy admits it. Claude Agent SDK, DSH SDK Client,
Hermes TUI Gateway and OpenClaw Gateway Client retain their native integrations. Each runtime
owns its loop, context, configuration, transcripts, tools and skill discovery. Only the
selected integration loads. SwarmX ships no builtin Pi runtime and has no direct Pi SDK
dependency. The existing DSH SDK still transitively depends on `pi-ai`; it owns that backend.
This migration removes SwarmX's own loop and tool bridge, not every upstream use of Pi libraries.

The historical Pi adapter and its tests are preserved under `examples/legacy-pi/` for evidence
and migration review. They are not an executable fallback or a supported optional runtime.
Existing `pi` settings and journal records remain parseable; selecting that retired runtime
fails with guidance to configure an external ACP agent. Native Pi session/auth files remain
untouched. A replacement agent must explicitly use a compatible native store and validate its
session ownership; Host observations are never imported as a new native transcript.

Swarm composition borrows a lead Agent and forwards method calls and Observer callbacks in
process. It has no wire format, connection handshake or provider-specific branches. The Host
intersects Host policy, caller, captured Swarm and persisted conversation grants at every hop.
Configuration validation and cancellation before dispatch remain Host responsibilities.
The shared Agent identity is the requested harness/model/effort/profile. An explicit effective-model
or effort conflict reported by a native runtime fails execution; raw reports remain in the journal.
Claude resolves model aliases with
its official catalog and checks main-thread output; native internal subagents can use their own
models. Codex rejects explicit model rerouting. Native acknowledgement without effective-setting
metadata is not independent proof of the backend configuration.
Native terminal outcomes confirm completion or cancellation; missing terminal results fail.
The existing public `stopReason` values remain stable for stored execution records and callers.

ACP is an optional external stdio gateway into the same protected Agent. The separate `acp`
integration calls an explicitly configured external agent through the official SDK. That agent
owns its runtime, tools, native configuration and session history. SwarmX has no dependency
on its internal Agent framework. History and resume consult the remote agent, never replay Host journal text into
a new session. Capability gaps fail explicitly. Internal recursive composition still calls the
Agent interface directly. See `acp.md` for endpoint admission and child lifecycle.

Codex history uses read-only `thread/read`, including while another client holds a writer.
Paginated histories use `thread/turns/list` and `thread/items/list`. New local threads select
the native legacy history store; an empty thread keeps its runtime until its first prompt because
it has no resumable rollout yet. A first prompt rejected before its rollout is written keeps its
still-live runtime for a corrected retry instead of releasing an unresumable thread. Runtime
failure releases the process and its Host tool credential and removes its empty-thread cache
entry. A later attempt opens a new runtime and resumes native history; without a persisted
rollout, native resume fails and the caller must create a new thread.
Pending stop and steering requests settle when that attempt ends, even if its runtime is retained;
they cannot dispatch against a later retry.
Subsequent turns open independent runtimes and release them
after the native terminal outcome. Catalog reads use separate unbound processes; an empty
thread's configuration is read through its owning runtime without closing it or starting a turn.
Immediately after creation, the session supports configuration queries and a first turn with
explicit model, effort and mode selections, including through the external ACP gateway.
Previously persisted Codex approval mode IDs retain their approval/reviewer policies and select
the corresponding native permissions profile; new native profile IDs pass through unchanged.
Native message phases, turn timing, tools and errors survive projection into the existing UI.
Codex records exact per-response usage from `rawResponse/completed`, grouped by the current
native turn and deduplicated by response ID. Input includes cache reads/writes; output includes
reasoning. Session-level `thread/tokenUsage/updated` totals and last-response snapshots remain
RAW evidence and are not added again. Missing or inconsistent usage remains unknown, including
when a run completes normally. Known observed tokens survive failure and cancellation; no USD
price is inferred. See [execution accounting](execution-log.md) for coverage and provenance.
Claude consumes native output through its idle notification after a result. Ordinary turns keep
the SDK session alive so delayed native title generation can finish. Host-budgeted executions
open a fresh or resumed SDK query with its registered `maxBudgetUsd` and close that query after
completion. Queued steering stays within the same query and budget. The limit uses Claude's
reported USD estimates; the SDK stops after observing the budget exceeded, so this is not an
invoice cap or a guarantee of zero overspend. The SDK reads
session metadata and history without opening a writer. Stop closes only the owned session
process and waits for its exit, discarding any queued steering inputs. Native errors and terminal
stops settle immediately and close the runtime; successful turns require idle. Output EOF without
a terminal outcome fails the run. Neither integration adds title-generation requests.
Each prompted local runtime gets a revocable MCP credential
bound to its Host execution; history/catalog reads never receive a run credential.
Hermes and OpenClaw retain their native tool configuration and directory-scoped session filtering.
Hermes uses its installed Python environment and TUI Gateway. The small Python launcher adds
the Host MCP tools through Hermes' native registration helper without editing user configuration.
A fresh session keeps its process until its first turn; subsequent turns resume the stored ID in
an independently owned process. Catalog/history processes carry no execution credential. The Host
uses the native lazy watch API for history/catalog access and ordinary resume for execution.
Ordinary resume may automatically continue unfinished work according to Hermes configuration;
SwarmX does not patch or disable that behavior. A prior turn that finishes while a new prompt's
submit is in flight cannot satisfy the new run; the Host waits for the submitted turn's own
terminal outcome. A submitted turn may complete or fail before its submit acknowledgement;
its terminal and idle state still settle that run and release its resources. The launcher registers
a private, read-only `swarmx.session.wait` method in the native dispatcher. It joins the native
execution thread and follows any replacement thread before returning the running state. Candidate
idle outcomes wait at this boundary because Hermes can emit idle before dispatching a queued
follow-up. Only one wait is active per execution; events, approvals and Stop remain available.
Queued input waits for its own turn. Steering consumed in the current turn shares that turn's
outcome; late steering requeued by Hermes stays observed through its follow-up, including failure
and cancellation. In-flight steering acknowledgements settle before resource disposal, and new
steering is rejected after a terminal frame until another native turn starts.
Runtime disposal terminates the owned process group, so descendants holding inherited stdout
cannot keep the transport open.
The Host
waits for a terminal message and the following native idle session info after interrupt
acknowledgement, allowing native title and turn cleanup to finish before closing the process.
ACP-specific edit approval modes
have no equivalent TUI method and are rejected; native Hermes approval configuration stays in force.
Hermes clarification preserves single choices, multiple choices, custom answers and batched
questions. Model cost confirmations require an explicit response before the selection is applied.
Secret and sudo input is masked in the desktop and excluded from durable interaction answers;
the actual value is delivered only to the pending native request. Expired requests cannot submit
a late answer.

OpenClaw uses the official Gateway Client for transport and request correlation. Session keys
remain native, history pages are replayed oldest first, and model/thinking choices come from the
Gateway catalog. Each submitted turn owns a separate idempotency/run ID; events for other runs
are ignored. Steering requests own additional native run IDs within the same Host execution;
their output and terminal outcomes remain observed even when native steering becomes a queued
follow-up. The Host execution finishes only after its main request and steering requests settle.
Cancellation names the session and each owned unfinished run and preserves unrelated side runs. A
disconnect, malformed owned event or native error cannot be reported as successful completion.
Native Gateway tool configuration remains authoritative; the Host never writes MCP servers into it.
The bundled gateway plugin (`apps/desktop/resources/openclaw-plugin`) is how a session reaches Host
product tools. While it runs, the Host publishes a bridge descriptor under
`$SWARMX_HOME/openclaw/bridges`; the plugin forwards each call over the Host's private socket
together with the trusted gateway session key, session id and tool call id. The Host authorizes the
call against the execution that leased that session, and the lease exists only for the run that bound
it. Calls without a lease, with another credential, or after the run ends are rejected, and the
plugin rechecks the gateway invocation before contacting the Host. Without the plugin, OpenClaw
sessions keep their native tools and Memory context only.
Listings also expose the previous ACP bridge UUID for matching native bridge sessions so the
Host can find conversations already owned by its journal; both IDs address the same writer.
An approval's native expiry or resolution by another client cancels the pending interaction and
prevents a late answer from being submitted.
Native questions are fetched and answered only for an owned run and matching session. Their
single/multiple choices, custom answers and question IDs survive projection. Secret-store
requests disclose the native storage name, host scope and replacement before masked input;
sensitive answers are not journaled. Dismissal cancels the native question. Expiry, resolution
elsewhere, run completion and stop prevent late answers.
SwarmX persists its Ed25519 device identity privately under its product home. The SDK owns the
challenge proof and pairing handshake; issued device tokens are stored separately for each
Gateway origin, device and role. Shared Gateway token/password environment settings authenticate
connections; a Gateway that does not issue a device token continues to require its shared credential.

DSH uses its published SDK client and the matching runtime. A run ends at native idle after its
input receipt. Stop closes that execution's runtime; closing one execution cannot stop another.
The SDK launches its runtime with the Host executable. SwarmX preserves the inherited environment
and sets `ELECTRON_RUN_AS_NODE=1` for that child so desktop builds run the DSH entry point as Node.
Idle is a lifecycle boundary, not proof of success: native `turn/end` errors, blocked turns and
unfinished turns cannot become successful Host results. Committed assistant messages and tool
events are projected as the SDK delivers them; child-session events remain diagnostic records
and cannot supply the root turn's answer or outcome. This SDK does not expose token deltas.
The SDK currently has no interactive server-to-client approval contract. Unsupported operations
are reported explicitly and never implemented by granting approval or silently switching agents.
Its session handles can continue within a live runtime, but the SDK exposes no stored-session
list, history or resume method. Reusing an existing ID in a new runtime is not a resume operation.
SwarmX therefore offers one independent execution per DSH task, with stop implemented by SDK
close. New work requires a new task. Task listing and past output come from the existing Host
execution journal, including after restart; they do not reconstruct or resume native sessions.
Model discovery and interactive steering are not exposed. Explicit model selection uses
`provider/model`; the provider is the first segment and the remaining path is the native model
ID. Omitted selection preserves the SDK's default route. Reasoning effort passes through to
the SDK, which validates the configured provider/model route and its supported effort.
DSH `profile` accepts `sdk` or `sdk-minimal`; omission preserves the native `sdk` default.
Profiles select native tools/plugins and are separate from permission `mode`, which DSH does
not support. Invalid route syntax, profiles and permission modes fail before runtime creation.
Host tools use the native MCP plugin through a private, temporary
launch configuration, removed when the owned runtime closes.
The SDK profile supplies deterministic titles; SwarmX adds no title-generation request.

## Acceptance

- Codex is the default external lead; explicit supported selections retain their own sessions.
- SwarmX production source and direct dependencies contain no Pi SDK, built-in loop or direct Pi
  product-tool bridge; DSH's upstream transitive backend is separately identified.
- Retired `pi` selections fail clearly without changing saved native data or selecting a fallback.
- External ACP requires explicit admission before spawning and retains remote native history.
- OpenClaw history preserves native tool failures without rerunning the calls.
- Recursive calls preserve observer identity, native failures, terminal outcomes and controls.
- Concurrent callers and nested Swarms cannot share or widen authority; stopped preparation
  cannot later dispatch. Stale execution controls cannot affect a subsequent turn.
- Native integration tests cover history, settings, streaming, tools, interaction, cancellation,
  process failure and shutdown. Claude tests assert no explicit title generation.
- A newly created Codex session exposes its native settings before its first prompt. Catalog
  reads preserve that session's runtime, and explicit first-turn settings pass Host validation.
- External ACP clients retain initialization, history, configuration, permissions and cancellation.
- Production imports contain no upstream ACP harness adapters, and the Swarm package contains
  no protocol dependency. Public package code maps and native-runtime guidance stay synchronized.
- Built native entry modules load directly in Node without a test transformer or starting a runtime.
