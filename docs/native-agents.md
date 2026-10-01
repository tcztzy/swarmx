# Native integrations

The default Swarm lead uses the embedded Pi SDK. The Host also calls Codex App Server,
Claude Agent SDK, DSH SDK Client, Hermes TUI Gateway and
OpenClaw Gateway Client directly. Their native runtimes own configuration, transcripts,
title generation, tools and execution. Only the selected integration loads.

Pi uses its native model/auth configuration, SessionManager, built-in tools, compaction and
DefaultResourceLoader. Skills retain Pi's standard discovery and on-demand full-text loading.
Sessions with the product `swarm` tool resolve the shared `@swarmx/swarm/skills/delegate/SKILL.md`
package resource and add its directory through the SDK's
`additionalSkillPaths`; restricted review sessions do not add it. Pi advertises its metadata
and reads its body on demand. `swarm.prepare` still loads the current skill with relevant
private evaluations before each delegated task; native discovery does not replace that check.
Frozen Memory context enters through the SDK's appended system prompt. Per-turn model/thinking
choices belong to the session and do not rewrite global Pi defaults. Custom tools invoke the
existing ProductServices directly, with the active execution's permissions, cancellation and
causal journal context. No internal protocol connection is needed.
Successful product-tool results project their `details` as the business result in both live
output and native history, so Memory cards and Science navigation receive the same data.
Built-in/extension tool results and failed calls retain their native payloads. RAW events and
Pi transcripts preserve the complete native result, including `content` and `details`.
History and model reads do not send a prompt. Catalog inspection uses Pi's in-memory snapshot
so its disposal cannot release the running session's resources or modify its transcript.
Session listing remains directory-scoped, and
the Host restores empty session reservations until Pi writes its first native assistant message.
Streaming, tool events, steering and abort use the SDK. Stop clears the SDK's pending message
queue before aborting so queued steering cannot launch another turn. The Host also checks cancellation
at the SDK's synchronous preflight acceptance callback before the Agent loop starts. Stop or disposal
during an asynchronous input or startup extension prevents the accepted prompt from reaching the
model once preparation returns; the run ends cancelled and releases its session. A later attempt
uses a fresh cancellation scope. Native preparation failures remain failures.
Native errors and missing terminal outcomes reject. A native resource-cleanup failure releases
the session for the next turn and is
recorded as a raw cleanup event; it only rejects the run that otherwise succeeded.
SDK extensions run in headless mode; terminal-only UI is not exposed.
See the upstream [SDK](https://pi.dev/docs/latest/sdk) and
[skills](https://pi.dev/docs/latest/skills) documentation for native configuration and discovery.

Swarm composition borrows a lead Agent and forwards method calls and Observer callbacks in
process. It has no wire format, connection handshake or provider-specific branches. The Host
intersects Host policy, caller, captured Swarm and persisted conversation grants at every hop.
Configuration validation and cancellation before dispatch remain Host responsibilities.
The shared Agent identity is the requested harness/model/effort/profile. An explicit effective-model
or effort conflict reported by a native runtime fails execution; raw reports remain in the journal.
Pi checks SDK session settings at preflight and during callbacks. A provider `responseModel` that
identifies another model in the same native catalog also fails before its tools execute. Unknown
response names remain raw evidence because Pi supplies no alias-to-canonical mapping; an unfamiliar
name alone cannot prove that a provider changed models. Claude resolves model aliases with
its official catalog and checks main-thread output; native internal subagents can use their own
models. Codex rejects explicit model rerouting. Native acknowledgement without effective-setting
metadata is not independent proof of the backend configuration.
Native terminal outcomes confirm completion or cancellation; missing terminal results fail.
The existing public `stopReason` values remain stable for stored execution records and callers.

ACP is an optional external stdio gateway into the same protected Agent. Its SDK and negotiated
extension belong at that boundary only. Desktop AG-UI, product MCP, A2A, recursive delegation and
background memory reviews do not cross an internal ACP connection.

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

- Pi is the default lead; explicit native selections retain their own sessions and behavior.
- Pi sessions resume from native storage, including Host-owned empty reservations; built-in tools
  and skills come from the SDK. Product tools retain Host permission checks and delegation context.
- Pi model catalogs, streaming, history, steering, cancellation and failures project into the
  existing Agent contract without a second execution loop or transcript store.
- Pi cancellation tests pause real SDK input/startup extensions: after Stop or disposal, releasing
  the extension causes no provider call; a stopped session can accept a later prompt.
- Successful Pi Memory/Science results render and navigate identically in live output and
  restored history; failed calls and non-product tool payloads are not unwrapped.
- Reloaded Pi and OpenClaw histories preserve native tool failure statuses in desktop and
  external ACP projections without rerunning the calls.
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
