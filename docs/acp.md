# ACP compatibility

## External boundary

ACP is an external stdio interface into the Host-protected Agent. Its official SDK owns framing,
validation and connection lifecycle. Swarm composition, desktop/A2A/product-tool calls and memory reviews
invoke Agents directly; they do not negotiate internal ACP connections. Native integration
contracts and setup are described in `native-agents.md` and `runtime-platform.md`.

The Host exposes the selected Agent's capabilities, native catalog, history and observed events
through ACP. Native runtimes retain configuration, titles, transcripts, tools and approvals.
The SwarmX extension acknowledges Host API grants, not a native process sandbox. The Host records
native callbacks before projecting them into ACP updates, permission requests and form elicitations.
Message phases, turn timing and tool kinds/statuses are presentation metadata, never grants.

Each external connection has independent negotiation and selections. Closing it cancels its active
prompts and pending interactions. Cancellation during Host configuration prevents dispatch;
after dispatch it invokes the native Agent's interrupt. Run-specific controls carry expectedRunId,
checked at the execution owner so delayed controls cannot target a later turn of the conversation.
ACP owns initialization, capability negotiation, session/new, list, load, resume, prompt, cancel,
session/update, request_permission and elicitation/create. Text and resource links are supported.
Load replays native history when advertised. Resume validates the existing session without
emitting history; an Agent without replay support is validated through its native model/session
lookup rather than a history read.
Neither operation claims to resume a running model call. Session/close and fork are not advertised.
Model and reasoning selection use configOptions and session/set_config_option, validated against
the permission-filtered native catalog. An unresolved native default is represented by the empty
selection; restricted model grants still require selecting an explicit admitted model before prompt.
Connection-local selections use standard session/set_config_option. Native mode selection also
accepts session/set_mode. The Host persists dispatched mode choices and reported native changes in the
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
harness/model allowlists and delegation. The Host intersects its policy, caller, saved conversation,
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


## Outbound external agent

The optional `acp` integration calls one operator-selected external stdio agent per Host.
A nonempty descriptor selects `acp` when `SWARMX_AGENT` is unset; otherwise the default is
Codex. Explicit selection wins. Admission remains separate from selection. The public
`pnpm acp` server entrypoint retains its inbound meaning.
Set `SWARMX_ACP_AGENT` to an absolute JSON descriptor path:

```json
{"id":"external-agent","command":"/absolute/agent-command","args":[],"env":{}}
```

The descriptor is strict configuration, not a shell command. Arguments are passed directly;
the child runs in the Host's canonical execution directory. The environment contains the
declared overrides and inherited process environment, except that Host-only `SWARMX_API_TOKEN`
and `SWARMX_MCP_TOKEN` credentials are excluded case-insensitively. Descriptor overrides of
those keys are rejected before spawning. Other native model credentials remain operator-owned.
SwarmX does not supply its private MCP credentials or client-injected MCP servers. This
first external client cannot expose Host `swarm`, Memory or Science product tools; existing
native integrations retain their execution-bound MCP bridges. External native tools remain
owned by that agent. No automatic Host Memory injection, snapshot or post-turn review is
attached to external ACP sessions; explicit unsupported instructions, profiles and budgets
fail rather than being ignored. Enable `acp` explicitly in Host policy;
it is not admitted by old/default policy. Endpoint IDs namespace remote session references.

The official ACP SDK owns framing and protocol validation. The remote agent owns its tools,
native model configuration and transcript. The integration reports only supported capabilities
and configuration options; unsupported operations fail rather than silently selecting another
runtime. Remote session list/load/resume are authoritative. A missing session cannot be replaced
with a fresh session, replayed Host text or an automatically repeated scientific request.
Native list and resume capabilities must be advertised during initialization; either missing
capability rejects before session creation, listing, resumption, configuration or prompt dispatch,
and the rejected connection's owned process is closed. History reading is optional and is
advertised only after each initialized connection reports `loadSession`. Without it, history
reads fail explicitly before dispatching `session/load`; resume never synthesizes history.

Standard permission requests preserve their option IDs and exact native tool input, and reach
the connected authorized user. A request may identify a tool by ID after `tool_call` and partial
`tool_call_update` notifications supplied its title and `rawInput`. The client retains these
observations only for the active session turn; explicit request fields replace observed values
(including an explicit null input). History replay, earlier turns and other sessions cannot supply
approval arguments. Desktop confirmation shows the resulting input read-only before submitting
the exact selected option ID. Displayed arguments do not widen the execution grant.
Rejection, cancellation, disconnect or a late answer cannot authorize execution. Protocol
capabilities are not execution grants; this client does not advertise client filesystem or
terminal operations and is not a process sandbox. The external agent independently enforces
its native tool boundary. Host completion remains independent of scientific acceptance.

Each owned process has an explicit close lifecycle. Cancellation during asynchronous preparation
prevents later prompt dispatch; after dispatch, Stop sends `session/cancel` and waits for the
original prompt's terminal result. A connection failure is a failure, never successful
completion. Disposal closes the connection and awaits owned child exit; restarting creates a
new process and loads the existing native session, without resubmitting its previous prompt.
Disposal cancels pending permissions immediately. If an active prompt does not return a terminal
result within three seconds, disposal closes its connection and the prompt fails. From connection
closure, an unresponsive owned process group receives SIGTERM after 500 ms and SIGKILL after three
seconds; disposal still waits for actual child exit. These are two successive deadlines, not a
promise of exit within three seconds of starting disposal. Late permission answers cannot approve
ended work, and a disposed integration rejects further operations without starting another child.
This does not promise that arbitrary scientific subprocesses are cancelled with a model turn.

Acceptance uses real SDK stdio child processes with a local scripted provider. Check new/load/
resume identity across process exit, native history ownership, permission allow/reject, Stop
during preparation with zero later dispatch, active-turn cancellation, disconnect cleanup and
retention of completed results. Scripted-provider runs establish engineering behavior, not model
reasoning quality or paid-model cost.

## Sources

- [ACP initialization](https://agentclientprotocol.com/protocol/v1/initialization)
- [ACP prompt and cancellation lifecycle](https://agentclientprotocol.com/protocol/v1/prompt-turn)
- [ACP extensibility](https://agentclientprotocol.com/protocol/v1/extensibility)
