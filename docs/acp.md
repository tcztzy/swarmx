# ACP compatibility

## Upstream harness adapters

Harness integration uses maintained ACP servers over stdio, with the official SDK owning
JSON-RPC framing and validation. Third-party servers are not required to implement the
SwarmX permission extension. That extension acknowledges the trusted Host's own API grants,
not the native process's sandbox. Ordinary native modes and approvals retain their upstream
semantics. Native slash commands and automatic title generation stay available; model allowlists
cover Host-dispatched model selection, not every call a native harness can make internally.
Only background reviews disable native tools, delegation, ambient MCP and title-model calls.
Each child gets a separately registered, revocable MCP bearer credential. It never receives the
Host's general bearer token and cannot select another execution by changing URL parameters.

The Host discovers session capabilities and model/config options from ACP. It preserves
upstream update payloads in the execution log, forwards exact permission options and form
elicitations, and uses session/cancel on the same connection as the active prompt. Steering
requires the upstream steering extension. History remains owned by the upstream harness.

SwarmX uses official SDK ACP connections at every Swarm-to-Swarm and Swarm-to-leaf edge.
Each Swarm is an ACP Agent upstream and an ACP Client downstream. The leaf boundary connects to
upstream ACP executables through official SDK stdio connections. Browser/A2A/MCP entry points consume ACP through a Host projection.
MCP remains the harness's tool carrier; delegation dispatched by that tool uses recursive ACP.
Background memory reviews also use the upstream ACP adapters, with tools disabled and approvals rejected.
Internal connections are operation-scoped and created inside the authenticated caller's execution
context. Each nested Swarm binds the intersection of caller, project and captured Swarm grants
before opening its downstream connection. Connection readers inherit that trusted context;
wire metadata never selects a parent execution or grants authority. Concurrent callers do not
share connection policy or model selections. Permission negotiation and acknowledgements are
required between trusted Host boundaries, not from third-party adapters. Closing a connection cancels its active prompts and approvals.
Cancellation during Host configuration prevents dispatch; cancellation after dispatch uses
session/cancel. Run-specific controls carry expectedRunId, checked again at the leaf so delayed
controls cannot target a subsequent execution of the same conversation.
ACP owns initialization, capability negotiation, session/new, list, load, resume, prompt, cancel,
session/update, request_permission and elicitation/create. Text and resource links are supported.
Load replays native history; resume validates the existing session without emitting history.
Neither operation claims to resume a running model call. Session/close and fork are not advertised.
Model and reasoning selection use configOptions and session/set_config_option, validated against
the permission-filtered native catalog. An unresolved native default is represented by the empty
selection; restricted model grants still require selecting an explicit admitted model before prompt.
Connection-local selections use standard session/set_config_option. Native mode selection also
accepts session/set_mode. At the upstream leaf, advertised mode config options take precedence over
legacy session modes. The Host persists dispatched mode choices and reported native changes in the
conversation journal and reapplies them on later turns. No mode is inherited as a Host grant.
Create a separate ACP Agent app for each connection so negotiation cannot bleed between clients.
The negotiated `_swarmx/models {sessionId?}` and `_swarmx/session/permissions {sessionId}`
queries support the Host's pre-session model picker and effective-grant inspection. Model/effort/mode
changes still use standard session/set_config_option. These queries do not mutate authority.

An observed normal native completion maps to end_turn; observed cancellation maps to cancelled.
Native limits/refusals use the corresponding ACP stop reasons when reported. Native errors and
streams without a terminal result reject. Cancel is a notification; its completion is confirmed
by the original prompt result, not by receipt of the notification. Tool permission requests use
request_permission with exact option IDs. Questions use elicitation/create. Cancellation settles
pending interactions without approval; late permission answers cannot authorize ended executions.

## SwarmX extension, version 2

Capabilities are advertised under `agentCapabilities._meta.swarmx`. A client requiring permission
ceilings sends `clientCapabilities._meta.swarmx = {version:2, permissions:true}` in initialize and
must verify the same version and permission support in the response before creating work.
Hosts lacking enforcement reject this negotiation. Permissions describe authorization, not
protocol capabilities; ACP filesystem capabilities alone do not sandbox an agent process.

After negotiation, session/new and session/prompt accept:

```json
{"_meta":{"swarmx":{"version":2,"permissions":{"tools":["memory.read","science.read"],"delegation":false}}}}
```

The permissions schema is the same source used by Host policy and the swarm tool: product-tool grants,
harness/model allowlists and delegation. The Host intersects project, caller, saved conversation,
and captured Swarm grants. Explicit widening rejects before native execution. Each response
acknowledges the full effective grant at `_meta.swarmx.permissions`; callers must validate it
(the `acknowledgedPermissions` helper rejects missing or wider grants). An unnegotiated, malformed
or unsupported extension is rejected; it is never ignored. Other vendors' `_meta` is unaffected.
Omission retains saved restrictions. Load/resume return the grant but reject permission changes;
use prompt for further narrowing. Ordinary ACP clients use the existing Host ceiling without
an extension handshake. Acknowledgement attests Host API authorization. It does not attest filesystem or native-process
isolation. Version 1 is rejected because its filesystem meaning is different; see `permissions.md`
for the explicit handling of older settings and conversation records.

Before dispatch, the Host emits a session_info_update notification whose `_meta.swarmx.execution`
contains runId, parentRunId, causedBy (journal event ID), and effective permissions. These values
come from the Host's execution context. They describe delegation, not conversation branching or
client-supplied authority. Incoming requests cannot assert a parent to gain its permissions.

The advertised `steer` capability enables `_swarmx/session/steer {sessionId,text}`. It requires
negotiation and changes the active turn. An empty response means input was submitted, not that
the model has consumed it. It does not replace ACP prompt/cancel.
Run-bound steering optionally includes expectedRunId; cancellation carries it alongside version
in `_meta.swarmx`. Form elicitation carries `_meta.swarmx.interactionId` to preserve the Host's
pending-interaction identity across gateways. Native diagnostics retain their event attributes.

The extension also reports activeRunResume:false and interactionResume:false. Empty Claude
reservations and A2A context bindings are journal-backed; native transcripts remain native-owned.
Only never-dispatched Claude reservations are reconstructed. Missing native state after a started
execution must not silently create a replacement conversation. A2A task handles and named Swarm
membership remain process-local. Recovery claims are distinct from ACP session resume support.

## Sources

- [ACP initialization](https://agentclientprotocol.com/protocol/v1/initialization)
- [ACP prompt and cancellation lifecycle](https://agentclientprotocol.com/protocol/v1/prompt-turn)
- [ACP extensibility](https://agentclientprotocol.com/protocol/v1/extensibility)
