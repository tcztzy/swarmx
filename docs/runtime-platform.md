# Conversation runtime boundary

SwarmX has one application UI and plugin host: the published DeepSeek Harness Web profile. DeepSeek Harness (`dsh`) and Codex App Server (`codex`) are peer conversation runtimes below that UI. Selecting Codex does not start another web server, renderer, Session store, or Agent loop.

## Audited DSH Web seam

The Host half is extensible through ordinary Cordis plugins: `ctx.webServer.register` can add bounded same-origin HTTP/SSE routes, and Typert/Remote can add business services. The browser conversation backend is not currently a registry seam. `@deepseek-ai/dsh-client-runtime` constructs one `SessionRuntime` directly over the DSH API connection and Mux frames; `conversationEvents` and `conversationViews` extend presentation of an already-open DSH Session, not list/create/read/start lifecycle.

Therefore Codex must not be implemented as a model provider, an `IApiClient`/Mux compatibility server, or a synthetic DSH `Session`. The minimum product-owned boundary is:

1. a Host Cordis `conversationRuntimes` service that registers peer adapters and owns a narrow browser protocol on the existing DSH `webServer`;
2. a product conversation client that consumes only that protocol and projects native items into the existing SwarmX conversation presentation;
3. the unchanged DSH `SessionRuntime` below the DSH adapter, not above Codex.

Replacing the existing DSH `sessions` service would require emulating unrelated workspace, subagent, projection, queue, command, attachment, and Agent-scope behavior. That is explicitly outside this seam.

## Ownership

Each native runtime owns its Session/Thread transcript, model loop, native tools, approvals, authentication, archive state, and resume format. SwarmX reads native state into disposable UI snapshots. It stores at most a lightweight `(runtimeKind, nativeConversationId)` address and never persists a third transcript or converts a Codex thread into a DSH Session.

DSH Web owns the only browser origin, application shell, Conversation, Science, PKB, Swarm, settings, and plugin inventory surfaces. The runtime protocol is mounted below `/api/swarmx/conversation-runtimes`; it never claims `/`, serves HTML, or starts a second listener. The browser never connects to the experimental app-server WebSocket transport.

Desktop quit owns the startup race as well as teardown. Once platform boot begins, `before-quit`
awaits that exact boot promise before disposal, so a late-created Harness, App Server, bridge, MCP
child, or recovery owner cannot outlive an exit requested during startup. App Server initialization
and root Swarm binding reconciliation share one finite 30-second startup deadline; a silent or
incompatible child is terminated and every pending RPC is rejected before startup fails.
Initial and macOS-reactivation BrowserWindow creation are owned surface transitions: if either
throws after the platform starts, SwarmX disposes the complete platform before reporting the fatal
surface failure.
Disposal attempts every layer and performs final Swarm recovery last; startup and cleanup failures reach the same fail-loud
exit boundary. If the Codex product MCP Host is constructed but its stdio transport cannot connect,
that entrypoint disposes the initialized Server, Swarm journal, Science resources, and Cordis context
before reporting the connection failure; a cleanup failure is reported alongside the original cause.

## Neutral runtime contract

`ConversationRuntime` exposes only operations used by the product:

| Operation | Contract |
| --- | --- |
| `list` | Return newest-first bounded unarchived native conversation summaries; adapters consume native pagination within their declared bound. |
| `create` | Create one native conversation in a Host-authorized workspace. An unsent Codex blank stays process-local until its first turn because App Server has not materialized durable history yet. |
| `read` | Rebuild one authoritative disposable snapshot from native state and reattach adapter-owned pending approvals. |
| `start` | Resume an unloaded stored conversation when its runtime requires it, then start one ordinary native turn exactly once and return its native turn identity. |
| `steer` | Submit input to the addressed active native turn. |
| `interrupt` | Interrupt only the addressed active native turn. |
| `revise` | Replace one terminal user turn: use native same-thread revision only where the adapter guarantees it, otherwise create a branch before that turn and start the replacement there. |
| `fork` | Create a native branch before a terminal native turn. |
| `archive` | Use native archive semantics; never delete domain state. Exact archived history remains readable, while start/steer/interrupt/revise/fork fail closed. Codex archive includes App Server's native spawned-descendant cascade. |
| `subscribe` | Emit ordered runtime-qualified native changes, including exact adapter-owned approval request resolution. |
| `respondToApproval` | Resolve one exactly matching pending native request. |
| `dispose` | Reject pending work and stop owned native resources. |

`ConversationRuntimeRegistry` qualifies every browser operation by runtime kind and rejects an unregistered runtime. It assigns one globally monotonic sequence exactly once before multiplexing adapter events to subscribers while retaining runtime/native identities. Retry and Edit call the selected adapter's `revise` operation; explicit Fork always calls `fork`. Edit keeps the draft in browser state and performs no native mutation until the user submits it.

## Browser protocol

The Host Cordis plugin reuses DSH Web's loopback listener and same origin. Requests use strict bounded JSON schemas, always include `runtimeKind` where an address is ambiguous, derive workspace authority from the Host, and abort work when the browser disconnects. SSE events retain `runtimeKind`, native conversation/turn/item identities, adapter order, and the registry's global sequence. A completed item is authoritative in Host and browser projections; a later or duplicate delta cannot change its text or make it provisional again.

Protocol shutdown stops admission, aborts every ordinary request, and drains routes only to a fixed deadline before adapter teardown continues; a non-cooperative route is reported through the desktop fatal cleanup boundary. SSE consumers that stop accepting frames are disconnected instead of accumulating an unbounded response buffer.

The protocol exposes list/create/read/start/steer/interrupt, Retry/Edit revision, fork, archive, approval, and subscription only. Revision is presented as Retry/Edit rather than as a raw adapter or Codex-specific method. The protocol does not expose DSH Session API, Agent API, Mux frames, Codex JSON-RPC payloads, absolute paths, environment values, or MCP transport.

The product conversation client is installed as an ordinary DSH client plugin. When DSH is the default runtime, the published DSH Conversation occupant remains unchanged. When Codex is selected, one runtime-neutral occupant shadows only the single `conversation` slot at a lower priority and consumes the Host protocol; it does not replace the DSH application frame, sidebar, settings, Science, PKB, Swarm, details, or plugin host. This is the direct UI boundary required by the audited coupling, not a second renderer or a Session compatibility layer.

## Adapter differences

The DSH adapter uses `ctx.sessionQuery` for native reads, `ctx.agents` for live execution, and completed DSH event boundaries for forks. Its `revise` implementation creates that branch and starts the replacement there, so DSH Session history stays immutable. DSH Web already owns the sole DSH user-question and approval providers, so the adapter must not register a second provider or answerer. DSH approvals continue through that native browser channel; `respondToApproval` fails explicitly if called on the adapter. Existing DSH Cordis Tool and Remote carriers remain the integration path for product capabilities.

The Codex adapter owns one `codex app-server` process over JSONL JSON-RPC stdio. It initializes once with the experimental API capability because exact native branching uses verified `thread/fork.beforeTurnId`, while last-turn editing can use experimental paginated thread creation and `thread/revert`; all other ordinary lifecycle, item, and approval traffic stays on stable methods. Standalone App Server builds can publish experimental schemas before implementing their backing store, and there is no side-effect-free general capability handshake. Each allowlisted experimental call therefore fails actionably and never falls back to different semantics.

`thread/read` does not load or subscribe to a stored thread. The adapter tracks threads loaded by create, fork, and resume; before `turn/start` on an unknown stored thread it calls `thread/resume` exactly once. `thread/closed`, `thread/status/changed` to `notLoaded`, and every `thread/archived` notification clear that state. A cascading archive notification also rejects approvals for the exact archived descendant, not only the parent named by the browser.

Codex list consumes unique `nextCursor` pages up to 1,000 summaries and explicitly includes interactive `cli`, `vscode`, and `appServer` sources. A newly created thread has no durable rollout before its first user message, so the adapter merges that process-local empty summary into list results; successful `turn/start` removes it. Closing App Server intentionally discards an unsent blank instead of inventing a transcript or reusable fake thread id.

SwarmX creates legacy threads by default. `SWARMX_CODEX_PAGINATED_HISTORY=1` explicitly opts new threads into paginated history; an incompatible App Server then fails actionably instead of falling back after Edit. User conversation reads remain on bounded stable `thread/read(includeTurns: true)`. The exact pre-first-message unmaterialized response is read once more with `includeTurns: false` and projected as an empty thread; no other read failure is reinterpreted. Persisted legacy items can be lossy and a full-history frame can exceed the adapter bound; SwarmX exposes only native persisted items and fails rather than manufacturing missing tool history. The adapter validates wire messages and terminates the child plus pending calls on disposal. It does not enter a DSH Agent loop and does not expose an app-server socket to the browser.

For an enabled and readable paginated Codex thread, `revise` may target only the latest terminal turn. It loads the thread when necessary, calls `thread/revert` with that turn as the excluded boundary, then starts the replacement text in the same native thread. `thread/revert` changes persisted conversation history only; it does not reverse file edits or commands already performed in the workspace. Older turns and legacy-history threads call `thread/fork.beforeTurnId` directly, including the first turn, so native ancestry and source configuration survive. A rejected fork boundary, paginated history, or `thread/revert` method is an actionable protocol error: the adapter does not silently select another method after failure. If revert succeeds but the replacement turn cannot start, the native thread remains reverted and the UI retains the draft so the user can retry deliberately.

App Server can clear an approval because its turn completed, was interrupted, or otherwise invalidated the request. The adapter maps the server JSON-RPC request id to the exact neutral approval; `serverRequest/resolved` rejects/removes only that pending request and emits `approval_resolved` so the browser removes its prompt. When command approval params include `availableDecisions`, the adapter presents only supported string decisions from that ordered set rather than offering a choice App Server did not allow.

MCP form elicitation is typed end to end. The adapter requires explicit `mode=form`, accepts only
the protocol's supported string/boolean/number/integer/string-array field schemas, rejects unknown
keywords plus inconsistent options/defaults/bounds, retains required/options/bounds, and validates
submitted content against the exact original schema before resolving the native request. The browser
renders and submits typed values rather than coercing every field to text.

Capability differences stay explicit. DSH-only Session projections, slash commands, queued-message editing, subagent catalogs, Agent presets, questions, and approvals stay on the existing DSH client/runtime channel and are not fabricated for Codex. Codex-native reasoning, item status, questions/elicitation, approval choices, fork/archive, turn control, paginated history, and last-turn revert are projected only when App Server supplies them.

For Codex archive state, the bounded native archived-Thread listing is authoritative: a matching
entry overrides an omitted or stale `archived: false` field returned by `thread/read`. Archive
notifications still clear loaded/transient/approval state immediately.

## Product capabilities

Science, PKB, and Swarm policy live in runtime-neutral cores with structural workspace/actor/abort inputs. DSH exposes them through Cordis services, Remote, Tool, and existing client plugins. Codex receives the same owned operations through one configured MCP stdio server and skills. MCP is a tool/domain carrier only; it never lists, creates, resumes, interrupts, forks, archives, or subscribes to Codex conversations.

The MCP carrier calls those operation contexts directly and validates before dispatch. Each
operation owns one strict schema and explicit carrier projections: MCP publishes the complete JSON
Schema, while DSH receives the structurally equivalent subset its Tool-schema dialect accepts.
The DSH projection may omit unsupported bounds, but the same strict operation validator still
enforces them before either carrier invokes the implementation; action/request coupling and
defaults therefore remain carrier-independent. Neither carrier reserializes the other's projection
through a carrier-specific schema library. For authority it first uses Codex's exact client-supplied native
Thread metadata after a bridge read proves that exact Thread belongs to the canonical workspace,
then an exact MCP transport session when available, and otherwise an opaque call-scoped request
identity. Lead Thread and transport-session identities and the
private reserved-member-to-native-child binding are stable across MCP restarts; the latter lets a
child's own MCP call recover its exact member authority without storing a transcript in Swarm. A
server-lifetime nonce prevents session-less reused JSON-RPC request ids from inheriting authority
after restart, and its temporary actor is discarded when that exact call settles. Calls without
Thread or session metadata cannot inherit another call's Swarm authority merely because they share
a transport or workspace, and the same native Thread metadata on a different workspace grants no
authority. The carrier does not construct a fake DSH Agent, Session, Thread, or Tool execution in
order to reuse DSH-only wrappers.

Swarm members use the selected runtime's native child-conversation mechanism. A Codex child remains
a Codex Thread and a DSH child remains a DSH Session; the common coordinator stores orchestration
facts, while the Codex bottom adapter stores only its workspace-scoped native child handle. Those
handles are individual transactional rows: concurrent per-Thread MCP processes use exact
claim/release operations and never replace a whole-workspace snapshot from process-local actors.

The Electron primary instance is the sole `$SWARMX_HOME` owner. `DesktopPlatform` creates a
non-configurable recovery owner before mounting the user-configurable Harness tree or starting App
Server, and closes it only after both are stopped. That boundary alone performs cold or final
crash-closed task/effect/admission recovery; plugin disable/reload cannot start another recovery
epoch. DSH monitoring skips every Team for which its runtime has no live actor, so it cannot mutate a
Codex Team merely because both carriers share the journal. Codex startup then reconciles Codex
bindings once at the root boundary.

Per-Thread MCP processes open only an already initialized journal, without migration, salt writes,
or projection rebuild, and never run workspace recovery during startup, first Tool use, or disposal.
They may re-read a committed exact native-Thread binding and hydrate or synchronize that member (and
a lead may reload its own Team handles), but they do not change a `provisioning` member, archive
sibling Threads, or prune shared bindings. A present but inactive/revoked binding is denied rather
than reinterpreted as lead identity, and cached member actors are revalidated against the current
claim and member phase. App Server can start a child's required MCP process while the parent is still
provisioning it; a binding miss is retryable observation, not evidence of a crashed owner. Native
member creation completes inside the root bridge even if the auxiliary MCP response
disconnects, and the bridge commits the exact still-provisioning member claim before returning the
Thread. The bridge stops admitting work during shutdown, cancels ordinary requests, and gives
in-flight creation/retirement a bounded drain window before App Server teardown. A carrier timeout or
unclassified root error is never treated as proof of no handle. Confirmed rollback uses an independent
cleanup signal: an exact source-tagged zero-turn provisioning Thread is deleted with `thread/delete`,
while a materialized Thread is archived. Cleanup is single-flighted, and the exact binding is released
only after native acknowledgement; a handle conflict proves another owner and forbids both cleanup
and release. Root reconciliation accepts an interrupted claim
only when the native Thread is in the same workspace, is not archived, and contains an observed
initial turn; empty or archived claims remain failed and are conditionally cleaned. Cancellation,
transient transport failure, and an unavailable `thread/read` method retain the claim and fail loud
instead of masquerading as a missing Thread. Live child synchronization applies the same exact
Thread/workspace/unarchived checks and fails the member before it can receive more work. A lead may
reload current child handles for scheduling and lifecycle observation, but only the exact caller
recovers its queued mailbox; queue sequence/idempotency/pending-limit and delivery claims are atomic
across MCP processes. Team archive begins with a replayable journal fence shared by every MCP
process. A native create already in flight continues to its root claim; if the fence won, the bridge
deletes that exact still-empty tagged Thread without fabricating an initial turn, or archives it after
materialization, then atomically records member retirement with exact claim release. Final archive
cannot precede that acknowledgement. A secondary
Electron launch focuses the primary window and exits before starting any platform service.

Codex's provisioning `threadSource` is not durable identity. A zero-turn Thread is absent from both
active and archived lists, and materialization clears its source tag. The owning App Server runtime
may use the tag only to correlate a still-transient exact Thread before first delivery. The root
bridge therefore retains the returned exact id until claim/cleanup settles; after delivery, only the
transactional member binding can identify the Thread. Cold recovery never guesses a handle from a
missing tag or list entry.

Science image results keep carrier parity: the Codex inline-image adapter fully decodes supported
image bytes, verifies their declared MIME family, and publishes actual intrinsic pixel dimensions
and byte count. It rejects malformed image bytes instead of inventing attachment metadata.

## Workspace and product state

`WorkspaceScope` contains a canonical absolute root and an opaque Host token. Only the Host can mint it. Domain services authorize relative paths beneath that root, including symlink resolution; runtime ids and model-supplied paths grant no file authority.

`$SWARMX_HOME` is the sole product-state root. A one-time import may copy only known legacy Science/PKB/Swarm state from `$DSH_HOME`; one shared whole-import entry/byte budget is enforced during traversal before each copy, and every copied tree is verified before the atomic publish. Native Session/Thread transcripts are never imported.
Launcher-enforced Science, PKB, and Swarm roots extend the fully composed profile/home plugin
configuration, so privacy limits, metadata opt-outs, runtime choices, and monitor policy are not
replaced when the product-state root is injected.

## Failure behavior

A missing selected runtime executable, incompatible app-server protocol, malformed native record, route collision, unsupported required method, failed runtime metadata request, or failed product-state migration is actionable. The selected-runtime occupant displays the failure instead of leaving DSH visible as a substitute, and synchronous selection errors reach the same fatal startup boundary as asynchronous boot errors. In particular, a failed Codex revert is never converted into a fork, because that would change the user's chosen history semantics. SwarmX does not silently select another runtime, emulate a missing native capability, or route Codex through the DSH model/provider stack.

`SWARMX_CODEX_FULL_ACCEPTANCE=1` runs the opt-in real App Server matrix, including a real streamed
turn, steer and interrupt, Retry/Edit, first-turn and later Fork, same-thread revert, restart/resume,
approval accept/reject, shared Science/PKB/Swarm MCP operations, archive/disposal, and owned-process
cleanup. The default test run may retain a bounded smoke subset, but it cannot be reported as that
full acceptance matrix.
