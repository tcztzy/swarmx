# Permissions and native execution

SwarmX has two distinct controls. The Host authorizes operations on its own APIs. Each harness
owns its native execution permissions, sandbox, command approvals and mode names. ACP transports
configuration and approval messages; it does not sandbox a process.

## Host grants

`permissions` contains `tools`, `harnesses` and `delegation`:

```json
{
  "tools": ["memory.read", "science.read"],
  "harnesses": {"codex": null, "claude": ["permitted-model-id"]},
  "delegation": false
}
```

Tool grants are `memory.read`, `memory.write`, `science.read` and `science.write`. Memory reads
include session recall and vault/core-note reads. Science reads use `science_query`; other Science
tools require `science.write`, including scientific execution. The latter still runs under the
separate research environment's filesystem, network and resource limits. A missing tool grant
rejects the operation before dispatch. `tools: []` disables both product-tool families.
Read and write are separate grants; neither implies the other. These grants apply to Agent API
calls. Frozen conversation context, authenticated user edits and Host bookkeeping have their own
contracts; a tool grant is not a confidentiality boundary for the shared workspace.

Omitted request fields inherit. Project defaults allow all four tool grants, configured harnesses
and delegation. An omitted harness is unavailable; a null model list permits its native models,
while an empty list permits none. `delegation: false` blocks creating or starting more work through
the Swarm tool. Status and cancellation remain available subject to session ownership.

Each run uses the intersection of project policy, the authenticated caller's grant, the named
Swarm's captured grant, the saved conversation grant and the requested child grant. Explicit
widening rejects. Permissions belong to the execution context, never to untrusted wire parent IDs.
Concurrent callers do not share grants. Reusing an existing Swarm or conversation does not lend
its broader authority to a restricted caller. Child product MCP credentials are separately scoped,
bound to the active execution and revoked when its connection closes.

## Native modes and approvals

Ordinary tasks expose the upstream adapter's advertised permission selectors or session modes.
Labels, IDs and choices stay native; unsupported selectors are not invented. Choices apply to
the selected native conversation, not vendor-global configuration. Dispatched choices and reported
native mode changes are saved in the journal and reapplied on later turns, including after restart.
Model admission remains a Host
decision for Host-dispatched calls; native internal model calls (titles, native subagents or slash
commands) are governed by the harness. Selecting a native mode cannot grant another harness/model
or additional Host tools.

The Host does not force ordinary tasks into a sandbox, disable native delegation, suppress native
MCP configuration or replace their approval policy. A planning lead can delegate coding when its
Host grant permits delegation. Conversely, a child running in YOLO cannot gain permission to call
a Host API denied by its grant. This is API authorization, not isolation against the native process:
a process with filesystem access could modify the same data directly. Do not describe a Host tool
grant as a read-only machine, or promise that all descendants share a filesystem sandbox.
Host authorization never bypasses native restrictions; the harness can still reject an allowed
operation. In particular, SwarmX does not revoke delegation merely because a mode is called Plan.

Native approval requests are forwarded with their exact options. The connected user chooses;
an Agent cannot answer its own approval. Allow-once does not change Host grants or siblings.
Remembered native approvals have the scope defined by the harness and may outlive a conversation.
Cancellation settles pending approvals, and late responses do not authorize ended executions.

## Restricted background reviews

Memory reviews are a distinct internal operation: an admitted model, no Host product MCP credential,
rejected observed tool calls and approvals, and cancellation. The Host requests additional native
restrictions, but unmodified upstream adapters own title calls, session persistence and internal
execution. No cross-harness tool-free or ephemeral guarantee is claimed. Review configuration does
not affect ordinary task modes.

## Previous permission data

The previous `permissions.filesystem` described a native filesystem ceiling. It must not silently
become a Host-only grant. New ACP permission negotiation uses version 2 and rejects version 1.
Historical journal events remain unchanged and readable; conversations with old filesystem grants
cannot run again under the new semantics. Create a new conversation and select its native mode.

Old project settings containing `policy.approval` must be deliberately updated: remove that field,
set `policy.tools` to the desired Host grants, and configure native modes in the harness/task.
`policy.filesystem` now controls only the isolated research environment. The settings loader rejects
old settings with an actionable path instead of silently changing their native execution policy.

## Examples

| User authorization | Lead's native mode | Child behavior |
| --- | --- | --- |
| Memory reads and delegation | Plan | Can delegate; descendants may call Memory reads, never Memory writes through the Host |
| All Host tools, no delegation | YOLO | Can use granted Host tools; cannot start another task through the Swarm tool |
| Whole task tree must never modify files | Any | Not a cross-harness guarantee provided by SwarmX; use an actually isolated execution environment |

Host authentication, project/session ownership and research container isolation remain independent
of native permission mode selection.
