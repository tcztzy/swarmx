# Swarm

Swarm supplies lightweight native Agent composition within the
[long-term research system](product-direction.md). The Host owns persistent work, budgets,
acceptance and policy; the composition package retains its direct Agent contract.

## Compatibility contract

The lead is an external Agent reached through a native integration or admitted ACP endpoint.
Its runtime owns sessions, tools and skills; SwarmX owns composition and grants. Product tools
use the existing execution-bound Host bridges where supported. External ACP currently has no
product-tool bridge, and no embedded Pi path remains.

Composition uses the direct `Agent` contract with `AgentCapabilities` and `RunResult` types.
Native integrations preserve the public `stopReason` values, including cancellation and limits.
Errors and unconfirmed native outcomes reject instead of becoming `end_turn`. Cancel requests
do not prove cancellation: the original prompt's terminal response does. Swarm composition
forwards method calls and Observer callbacks recursively, without protocol connections.
See [ACP and the SwarmX extension](acp.md) for the optional external protocol boundary.

Directory-scoped native session IDs remain conversation identities; each invocation has a
separate execution ID. The Host persists empty Claude session reservations and A2A context-to-session
bindings in its journal. Native history takes precedence over an empty reservation. Resuming a
conversation after restart starts a new execution; running executions and unanswered interactions
cannot resume after Host restart. A2A task handles remain process-local and are distinct from
persisted conversation bindings. Concurrent calls for one A2A context cannot allocate two sessions.

The Host owns interaction cancellation for an execution; gateways project requests and deliver
answers. Interrupting or ending an execution settles its unanswered callbacks without approval.
Cancelling an active desktop stream requests cancellation; the intentional AG-UI interaction
handoff leaves the pending request active. Late answers cannot approve an ended execution. A text-only A2A
entry point explicitly rejects interactions. Native raw events remain available for diagnosis.

Permissions are enforced by the Host, not by a skill or prompt. A skill may explain delegation,
but cannot grant authority. Host policy is the ceiling; each named Swarm captures its creator's
effective permissions. Every invocation intersects that grant with the current caller and Host policy.
Explicit requests to widen authority fail before native execution. Permissions are immutable for
an active run and recovered from its authenticated execution context for product MCP calls.

Permissions also belong to the conversation, identified by its directory-scoped native session ID.
Creation records the effective grant even before the first message. Every admitted turn records its
effective permissions before native dispatch; later turns intersect all saved grants with the current
Host policy, caller and Swarm grants. Switching entry points or Swarm aliases, omitting
permission arguments, resuming after restart, or loosening Host policy cannot widen that conversation.
Further restrictions persist even if an admitted run fails or is cancelled. Rejected requests do not
change the grant. Session-specific model catalogs use the same saved model allowlist.

Existing Host grant snapshots are honored without rewriting history. Conversations containing the old
filesystem grant remain readable but cannot execute under changed semantics. Older native sessions
with no recorded grant acquire one on their first admitted SwarmX turn. Independent conversations
have independent grants; create a new conversation to use a broader grant permitted by Host policy.

Host grants contain `tools`, `harnesses` and `delegation`. The tool grants are `memory.read`,
`memory.write`, `science.read` and `science.write`. The harness map associates native IDs with
model-ID arrays, or `null` for all native models; an omitted harness or empty model list denies
its admission. Restricted model lists require an explicit admitted model for Host-dispatched runs.

`swarm.create` and `swarm.send_message` accept optional `permissions`; omitted fields inherit.
`send_message` also accepts `model` and `effort`. `delegation: false` denies creating or starting
more work through the product tool; status and cancellation retain their ownership checks.
A caller cannot acquire a missing Host tool grant through a child, another Swarm, a native alias,
a resumed conversation or a new MCP carrier. Tool authorization is checked before dispatch.
Automatic memory reviews are not scheduled without both `memory.write` and delegation authority.

## Delegation preparation

Before choosing a child, the lead calls `swarm.prepare` with the exact task text and up to four
short search queries for the task, harnesses, models or providers. It receives the admitted harnesses
and model allowlists, the shared `@swarmx/swarm/skills/delegate/SKILL.md` skill with its revision, and fresh
private user notes and matching Memory concepts with their prerequisites and stale flags. The returned
`knowledge.content` is the skill body with relevant private evaluations and Host-computed facts appended;
the existing `memory` field retains the structured sources. Each evaluation preserves its written
assessment, appropriate scenarios, observation/judgment/preference kind, revision and original evidence.
Statistics distinguish harness/model/effort/provider/profile/version combinations, runtime outcomes,
independent acceptance, cost coverage and task wall time. Missing identity or price stays unknown.
The requested configuration is the shared identity. Explicit native model/effort conflicts fail;
an SDK that does not report effective settings provides no independent confirmation. Different
requested effort levels and omitted effort are separate evidence groups. Native effort names
retain their harness/model meaning; identical names across runtimes do not imply equal computation.
For cited executions, preparation also reads current Host feedback and later corrections; a superseded
acceptance marks the old assessment as corrected even when its Memory body has not changed. This is
work-attempt acceptance, not an isolated estimate of each contributing run's quality. Stale or unresolved
concepts remain explicitly unverified. Capability-based scenarios in the shipped skill are integration guidance,
not measured rankings. Private assessments are loaded afresh and never written into shipped resources.
Concepts
tagged `agent-selection` are also searched. Retrieval returns bounded complete concepts and explicitly
lists omitted matches; use `memory.load_memory` for additional evidence. Disabled Memory or a missing
`memory.read` grant returns an explicit unavailable status without reading private knowledge.
Admitted harnesses are candidates, not proof that their native runtime or credentials are available.
`swarm.models` reads one candidate's native catalog lazily under the same permission checks.
Managed work also supplies its goal, criteria, shared remaining budget, Agent choices and local
acceptance/cost evidence before the selected supervisor runs. Its native Agent loop delegates and
uses returned results to decide subsequent instructions. Every child retains the work identity and
shares its root runtime budget; see [work management](work-management.md).

An Agent-originated `send_message` must supply the completed preparation's `preparationId` and a
nonempty `reason`. The Host checks the same parent session, current run and exact delegated task
before creating a child session or dispatching work. Pending or failed preparation, another parent's
preparation, or a previous turn's preparation cannot pass. This ensures that selection context was
returned before dispatch; it cannot prove that a model understood or followed the evidence. The
existing execution journal records the prepared context, its revisions and the lead's reason.
Unscoped external calls retain their existing API behavior. Permissions are always checked again
at dispatch; knowledge and preparation never grant authority.

The lead follows explicit user choices, weighs matching private experience against project defaults,
and considers the full harness/profile/provider/model/effort/task combination. User-specific provider
incidents stay in private Memory; bundled knowledge is maintained with the application and is never
copied over user concepts. Swarm composition does not rank candidates or implement provider routing.
DSH accepts a qualified `provider/model` and optional `profile` (`sdk` or `sdk-minimal`) in
`send_message`. Profile selects the runtime composition; it is separate from native permission mode.

Ordinary tasks retain native modes, tools, hooks, delegation, MCP configuration and approvals.
The Agent catalog exposes native mode choices; the Host does not manufacture a cross-harness ranking.
Selecting Plan or Full access does not alter Host grants. These grants authorize Host APIs;
they do not prevent a native process from accessing the same data directly through its own tools.
The separate background-review path rejects observed tools and approvals and requests native
restrictions; upstream internal execution is not covered by a cross-harness tool-free guarantee.
See [Permissions and native execution](permissions.md) for boundaries, examples and legacy data.

`createSwarm(name, lead)` borrows an Agent and binds its methods directly. Disposing the Swarm
does not dispose its lead. Session operations, model configuration and cancellation travel down;
text, tool activity, approvals and questions travel up through the same Observer.
ProductServices binds caller, Host policy, saved conversation and captured Swarm permissions at each hop.
A parent does not inspect provider identity; there is no configured nesting limit.

The Host's `swarm` MCP tool creates named Swarms and exposes status/new_session/send_message/cancel.
The same ProductServices instance serves MCP and Electron IPC. Membership is in-memory; native Agents
own resume state and context. The Host's append-only execution journal records membership changes,
delegation causes and observed child events even when a delegation caller only consumes final text.
See `execution-log.md` for coverage and native-process correlation.

ACP stdio and A2A are external gateways. A delegated
interaction forwards to the connected parent's interaction callback, identifying the child.
The desktop queues concurrent confirmations through AG-UI interrupt/resume. Interactive delegation
that has no connected user fails explicitly; it never auto-approves tools.

The desktop can inspect causally linked child executions in the persistent journal and steer or
stop a currently active child by execution ID. Stale controls cannot affect a later run of the same
session. Pending confirmations must be answered or cancelled in the parent conversation first.

Persistent work scheduling and independent acceptance belong to the Host above this composition
layer. Native execution outcome, content acceptance and approved knowledge changes retain separate
identities and authority. See [product readiness](product-readiness.md) for the current executable
scope and [ROADMAP](../ROADMAP.md) for the next scheduling and learning milestones.
