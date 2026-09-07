# Swarm

## Compatibility contract

ACP is the communication contract. Composition uses official SDK connections, `AgentCapabilities` and
`PromptResponse` types; adapters preserve ACP `stopReason`, including cancellation and limits.
Errors and unconfirmed native outcomes reject instead of becoming `end_turn`. Cancel requests
do not prove cancellation: the original prompt's terminal response does. Swarm composition
forwards ACP requests, responses, notifications and reverse requests recursively.
See [ACP and the SwarmX extension](acp.md) for negotiation, permissions and delegation metadata.

Workspace-scoped native session IDs remain conversation identities; each invocation has a
separate execution ID. The Host persists empty Claude session reservations and A2A context-to-session
bindings in its journal. Native history takes precedence over an empty reservation. Resuming a
conversation after restart starts a new execution; running executions and unanswered interactions
cannot resume after Host restart. A2A task handles remain process-local and are distinct from
persisted conversation bindings. Concurrent calls for one A2A context cannot allocate two sessions.

The Host owns interaction cancellation for an execution; gateways project requests and deliver
answers. Interrupting or ending an execution settles its unanswered callbacks without approval.
Disconnected active browser streams request cancellation; the intentional AG-UI interaction
handoff is not a disconnect. Late answers cannot approve an ended execution. A text-only A2A
entry point explicitly rejects interactions. Native raw events remain available for diagnosis.

Permissions are enforced by the Host, not by a skill or prompt. A skill may explain delegation,
but cannot grant authority. Project policy is the ceiling; each named Swarm captures its creator's
effective permissions. Every invocation intersects that grant with the current caller and project.
Explicit requests to widen authority fail before native execution. Permissions are immutable for
an active run and recovered from its authenticated execution context for product MCP calls.

Permissions also belong to the conversation, identified by its workspace-scoped native session ID.
Creation records the effective grant even before the first message. Every admitted turn records its
effective permissions before native dispatch; later turns intersect all saved grants with the current
project, caller and Swarm grants. Switching entry points or Swarm aliases, omitting
permission arguments, resuming after restart, or loosening project policy cannot widen that conversation.
Further restrictions persist even if an admitted run fails or is cancelled. Rejected requests do not
change the grant. Session-specific model catalogs use the same saved model allowlist.

Existing Host grant snapshots are honored without rewriting history. Conversations containing the old
filesystem grant remain readable but cannot execute under changed semantics. Older native sessions
with no recorded grant acquire one on their first admitted SwarmX turn. Independent conversations
have independent grants; create a new conversation to use a broader grant permitted by project policy.

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

Ordinary tasks retain native modes, tools, hooks, delegation, MCP configuration and approvals.
ACP advertises native mode choices; the Host does not manufacture a cross-harness ranking.
Selecting Plan or Full access does not alter Host grants. These grants authorize Host APIs;
they do not prevent a native process from accessing the same data directly through its own tools.
Only the separately restricted background-review path forces tool-free execution.
See [Permissions and native execution](permissions.md) for boundaries, examples and legacy data.

`createSwarm(name, connectLead)` returns an official ACP Agent app. Its connector opens a downstream
ACP Client connection to another Swarm or leaf adapter. Initialization, session operations, model
configuration, cancellation and negotiated extensions travel down; updates, approvals and forms
travel up. Host projections serve browser/A2A/tool callers; leaf connections use upstream ACP adapters.
Internal operation-scoped connections bind caller and captured Swarm permissions at each hop.
A parent does not inspect provider identity; there is no configured nesting limit.

The Host's `swarm` MCP tool creates named Swarms and exposes status/new_session/send_message/cancel.
The same ProductServices instance serves MCP and REST. Membership is in-memory; native Agents
own resume state and context. The Host's append-only execution journal records membership changes,
delegation causes and observed child events even when a delegation caller only consumes final text.
See `execution-log.md` for coverage and native-process correlation.

ACP can connect in process or over stdio using the official SDK; A2A is an external gateway. A delegated
interaction forwards to the connected parent's interaction callback, identifying the child.
The browser queues concurrent confirmations through AG-UI interrupt/resume. Interactive delegation
that has no connected user fails explicitly; it never auto-approves tools.

The browser can inspect causally linked child executions in the persistent journal and steer or
stop a currently active child by execution ID. Stale controls cannot affect a later run of the same
session. Pending confirmations must be answered or cancelled in the parent conversation first.

This is recursive composition and explicit delegation, not a durable scheduler, verification DAG,
automatic knowledge admission, or proof that every manuscript claim is implemented.
