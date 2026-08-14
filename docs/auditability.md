# Auditability

SwarmX keeps one compact local audit chain for privileged decisions and side
effects. It is meant to reconstruct control flow without creating a second copy
of user content.

## What an event contains

Each event records a sequence and timestamp, category and action, outcome,
optional actor/target, stable request/Session/task correlations, bounded
sanitized metadata, and hashes linking it to the previous event. A privileged
or mutating boundary records intent before authorization or effect and then a
terminal outcome.

The audit payload never includes raw prompts or responses, Session messages,
source/file contents, terminal input or output, HTTP bodies or headers,
credentials, tokens, or environment snapshots. Those values belong only in
their canonical stores, if the product retains them at all.

## Event taxonomy

Actions are deliberately broad. A semantic effect gets a domain action; a
transport does not create one action name per method. Production emitters use
this complete action set:

| Surface | Canonical representation |
| --- | --- |
| Desktop IPC | Action `ipc.request`; target `{ kind: "ipc-channel", id: <normalized-channel> }` |
| Terminal | `terminal.create`, `terminal.write`, `terminal.resize`, `terminal.close`, `terminal.exit` |
| Tool and permission decisions | `tool.invoke`, `tool.decision` |
| CLI agent execution | Action `agent.run`; metadata `surface` is `cli_send`, `eval`, or `repl` |
| Other CLI lifecycles | `cli.doctor`, `cli.serve`, `cli.repl.session` |
| HTTP server | `http.request`; bounded metadata identifies method and allowlisted route |
| ACP server | `acp.initialize`, `acp.authenticate`, `acp.session.new`, `acp.session.load`, `acp.session.list`, `acp.session.resume`, `acp.session.close`, `acp.prompt` |
| Verified export | `audit.export` |

The outcome set is exactly `attempted`, `completed`, `failed`, `denied`,
`cancel_requested`, and `cancelled`. ACP prompt cancellation uses the same
`acp.prompt` action and correlates the cancellation request to the active prompt
instead of creating a second action type. Categories remain the bounded query
facets `session`, `task`, `tool`, `permission`, `provider`, `secret`,
`extension`, `workspace`, `telemetry`, and `system`.

`terminal.close` carries only the structural reason `user_kill`,
`owner_cleanup`, or `app_dispose`. Terminal writes record byte counts, never
input data; terminal lifecycle events never include cwd, environment, or stream
content.

Desktop IPC channels use one explicit policy:

| Policy | Emission rule |
| --- | --- |
| `intent_outcome` | Privileged/mutating call: intent before authority/effect, then terminal outcome |
| `failure_only` | Benign read, pure transform, or transient UI state: emit only an abnormal outcome |
| `semantic_only` | Host semantic events replace successful transport events; authorization and dispatch failures still emit `ipc.request` |

This keeps denied and failed boundaries reconstructable while removing routine
getter noise and duplicate terminal transport records.
Persisting only a semantic intent/attempt never suppresses the transport failure;
the current operation needs a successfully persisted semantic terminal outcome.

## Activity summaries

`~/.swarmx/activity.jsonl` is profile statistics, not part of the audit chain.
It contains exactly one `run_summary` per run: status, duration, token totals,
and aggregate tool/Skill counts. It contains no raw content or per-call
timeline, and must not be used as authority for a security decision.

## Storage and integrity

The default store is `~/.swarmx/audit/events.jsonl`; its fsynced head checkpoint
is `~/.swarmx/audit/events.head.json`. Both files and their directory are
created with restrictive local permissions. Appends use a single-writer lock,
fsync, strict schema replay, and a SHA-256 chain. Verification detects broken
links, modified/reordered/deleted records, a log behind its checkpoint, and an
unadopted valid tail ahead of it. Recovery is explicit and limited to one torn
final line, a missing final newline, or adoption of a fully verified tail after
an interrupted checkpoint update.

This is local tamper-evidence. It is not remote attestation or non-repudiation
against an administrator who can replace both the log and checkpoint.

## Inspect and export

Use the CLI audit command to verify the chain, print bounded filtered events, or
export verified JSONL:

```shell
swarmx-cli audit --verify
swarmx-cli audit --category permission --limit 50 --json
swarmx-cli audit --request-id req_123 --output audit.jsonl
```

Query and export always replay and verify the chain first. Export is itself an
audited operation. Successful list/verify reads do not append to the chain and
therefore do not change the evidence being observed; denied Desktop inspection
attempts do append a content-free `ipc.request`. Treat exported JSONL as
potentially sensitive operational metadata even though secret-bearing values
are rejected or redacted.

## Failure behavior

If an intent event cannot be made durable, a privileged boundary fails closed:
it does not approve the permission, dispatch the HTTP/IPC operation, or start
the process. Once an external effect may have begun, SwarmX reports failure or
unknown state honestly and does not invent success.
