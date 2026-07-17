# Claude Code trained-in tool parity

Baseline: the 42 tools listed by the public
[Claude Code tools reference](https://code.claude.com/docs/en/tools-reference), audited on
2026-07-17. Claude Code varies the available set by platform, settings, account, and
provider. SwarmX uses the same rule: a compatible name is exposed only when the selected
runtime can perform the documented operation.

Status meanings:

- **Implemented**: exposed to direct Claude-family SwarmX models with the documented name,
  principal input schema, real side effect/state, and structured result.
- **Partial**: useful compatible implementation exists, but a documented media, ordering,
  spill-file, or host integration behavior remains missing.
- **Conditional**: must be backed by a configured provider, account, MCP server, scheduler,
  browser, or remote-control service; SwarmX does not install a fake local substitute.
- **TODO**: known parity work has an explicit pending implementation task and the name is not exposed.

ACP-launched Claude Code owns its native tools. This matrix covers only host-injected tools
for direct SwarmX model execution.

| # | Claude Code tool | SwarmX status | Current boundary |
| -: | --- | --- | --- |
| 1 | `Agent` | Partial | Direct desktop Claude requests run a real child instance of the same resolved Agent Composition with independent tools, the current dynamic Project/worktree root, exact completed output fields, and request-scoped `agentId` resume. Background/team agents, specialized definitions, model-route switching, isolation, and cross-request resume are rejected explicitly. |
| 2 | `Artifact` | Conditional | Claude account/cloud artifact service; no local placeholder. |
| 3 | `AskUserQuestion` | Implemented | Desktop bridge pauses the request for 1-4 validated questions with single-select, multi-select, and automatic free-text Other; task cancellation aborts the wait. |
| 4 | `Bash` | Partial | Real sandboxed foreground/background processes, explicit background tasks, and timeout-to-background fallback; full-output spill files and persistent safe `cd` remain. |
| 5 | `CronCreate` | Implemented | Exact strict `cron`/`prompt` plus optional `recurring`/`durable` input and Claude-shaped output. The local-time scheduler validates five fields and a next run within one year, caps the combined Project/session inventory at 50 jobs, serially reinjects due prompts, and emits raw eight-hex-character ids. Session jobs remain in memory and expire after three days; durable jobs use Claude Code's `.claude/scheduled_tasks.json` format, creator identity, seven-day recurring lifetime, and restart recovery. |
| 6 | `CronDelete` | Implemented | Exact strict `{id}` input/output; cancels and removes either a live session timer or a Project-durable task using an atomic file mutation. |
| 7 | `CronList` | Implemented | Exact strict empty input and `{jobs}` output combines current-session jobs with every valid durable Project job, including Claude field names and human schedule text. |
| 8 | `Edit` | Implemented | Exact replacement, read digest, stale-file rejection, and atomic Project write. |
| 9 | `EnterPlanMode` | Implemented | Creates a private request-scoped plan file and blocks Project mutation plus Shell execution while preserving exact plan-file `Read`/`Write`. |
| 10 | `EnterWorktree` | Implemented | Exact optional `name`; creates or safely resumes `.claude/worktrees/<name>` on `worktree-<name>`, then rebinds guarded file, Shell, and conditional LSP operations to the canonical worktree root. |
| 11 | `ExitPlanMode` | Implemented | Reads the real plan file and waits without an idle timeout for explicit desktop approval; rejection keeps plan mode active with feedback. |
| 12 | `ExitWorktree` | Implemented | Exact `action: keep/remove` and optional `discard_changes`; restores the original root, preserves on keep/manager close, and refuses dirty or post-entry commits before removal unless explicitly confirmed. |
| 13 | `Glob` | Implemented | Includes hidden/gitignored files except `.git`, sorts newest first, caps at 100, and reports exact-count state. |
| 14 | `Grep` | Implemented | Ripgrep-backed modes, context/limit flags, Project containment, and structured counts. |
| 15 | `ListMcpResourcesTool` | Conditional | Implemented after at least one configured MCP server connects; paginated static resources only. |
| 16 | `LSP` | Conditional | Implemented for direct desktop Claude requests when a command-backed inventory LSP matches the contained Project file; all nine 2.1.187 read-only operations use one-based input and exact structured result fields. |
| 17 | `Monitor` | Implemented | Exact strict `command`, `description`, optional `timeout_ms`/`persistent` input and `{taskId, timeoutMs, persistent?}` output. It runs in the Project sandbox, retains the session shell, coalesces and bounds stdout-line notifications, rate-limits floods, marks process text untrusted, serially reinjects events between foreground turns, and remains stoppable through `TaskStop`; persistent tasks end on stop, Session deletion, or app shutdown. |
| 18 | `NotebookEdit` | Implemented | Guarded replace/insert/delete by cell id with Agent SDK-compatible output. |
| 19 | `PowerShell` | TODO | Requires a Windows-native sandboxed process host, background lifecycle, and schema/output tests; the current macOS Seatbelt Shell cannot honestly stand in for it. Tracked by `T195`. |
| 20 | `PushNotification` | Conditional | Remote Control/account notification service; no local placeholder. |
| 21 | `Read` | Partial | Bounded UTF-8 text and notebook JSON work; native image/PDF rendering remains. |
| 22 | `ReadMcpResourceTool` | Conditional | Implemented for connected MCP text resources; binary resources require a future authorized file sink. |
| 23 | `RemoteTrigger` | Conditional | Remote Control service and authenticated session required. |
| 24 | `ReportFindings` | Implemented | Up to 32 validated repo-relative findings with exact structured echo. |
| 25 | `ScheduleWakeup` | Conditional | Requires durable desktop/task wakeups; no response-only timer substitute. |
| 26 | `SendMessage` | TODO | Exact recipient/message/summary delivery requires concurrently live parent/teammate agents, identities, mailboxes, and shutdown/plan protocol frames; synchronous child resume is insufficient. Tracked by `T196`. |
| 27 | `SendUserFile` | Conditional | Remote Control file-transfer service and user authorization required. |
| 28 | `ShareOnboardingGuide` | Conditional | Claude account/onboarding surface; not a Project operation. |
| 29 | `Skill` | Conditional | Implemented for composition-selected filesystem Skills with native argument and environment substitutions. |
| 30 | `TaskCreate` | Implemented | Request-scoped task creation using Agent SDK 0.3.211 fields/output. |
| 31 | `TaskGet` | Implemented | Request-scoped task lookup and dependency output. |
| 32 | `TaskList` | Implemented | Request-scoped ordered task summary. |
| 33 | `TaskOutput` | Implemented | Real background process polling/wait; upstream marks this deprecated in favor of reading the output file. |
| 34 | `TaskStop` | Implemented | Stops the real sandboxed process group. |
| 35 | `TaskUpdate` | Implemented | Fields, status/delete, owner, metadata merge/delete, and reciprocal dependency links. |
| 36 | `TodoWrite` | Implemented | Atomic request-scoped list replacement with exact `oldTodos`/`newTodos`. |
| 37 | `ToolSearch` | Conditional | Implemented when MCP servers are configured: exact `select:` and ranked keyword search activate matching deferred `mcp__server__tool` schemas for the next model step and report pending servers when no match exists. |
| 38 | `WaitForMcpServers` | Conditional | Implemented while configured MCP servers are pending: waits at most five seconds, returns all Claude status buckets, and makes successfully connected server tools visible on the next model step. |
| 39 | `WebFetch` | Conditional | Expose only through a provider/browser capability with URL policy and documented extraction behavior. |
| 40 | `WebSearch` | Conditional | Expose only when the selected provider advertises hosted web search. |
| 41 | `Workflow` | TODO | Claude Code runs deterministic JavaScript `agent()`/`parallel()`/`pipeline()` scripts asynchronously with persisted script files and resumable run caches. SwarmX graph workflows are a different contract and are not exposed under this name. Tracked by `T197`. |
| 42 | `Write` | Implemented | Bounded UTF-8 create/replace with read digest, stale-file rejection, and atomic rename. |

The headless/non-interactive direct Claude profile exposes 17 of 42 names. A session-backed desktop direct Claude
request has real interaction, child-composition, and background-activation bridges and exposes 25: 22 implemented and 3 partial. `Skill` appears
only when the composition selects a readable filesystem Skill. With configured MCP servers,
`ToolSearch` appears for deferred schemas, `WaitForMcpServers` appears while any server is pending,
and two resource names appear after one connects. A matching configured language server also adds
`LSP`; a mixed connected/pending MCP configuration with LSP therefore raises the maximum currently
exposed set to 31 of 42. This is not full parity. The three remaining TODO names are
`PowerShell`, `SendMessage`, and `Workflow`; 14 account/provider/service-dependent names remain
conditional. The remaining
inventory stays normative until each tool has a real backend, contract tests, and documented
structured output.

Session tool results visible to the model now use these shapes:

```json
{"taskId":"7","timeoutMs":300000,"persistent":true}
{"id":"12ab34cd","humanSchedule":"Every minute","recurring":true,"durable":true}
{"id":"12ab34cd"}
{"jobs":[{"id":"12ab34cd","cron":"* * * * *","humanSchedule":"Every minute","prompt":"Check the build","recurring":true,"durable":true}]}
```

Durable create persists the input as project state rather than copying the output envelope:

```json
{
  "tasks": [
    {
      "id": "12ab34cd",
      "cron": "* * * * *",
      "prompt": "Check the build",
      "createdAt": 1784271630000,
      "recurring": true,
      "createdBySessionId": "session-123",
      "createdByPid": 12345,
      "createdByProcStart": "Fri Jul 17 10:00:00 2026"
    }
  ]
}
```

The input-only `durable` field is intentionally absent. Recurring execution atomically adds or
updates `lastFiredAt`; one-shot execution atomically removes the task before reinjection. A private
`.claude/scheduled_tasks.lock` uses session id, PID, process start, and acquisition time to prevent
duplicate legacy/orphan execution across live sessions. Stale/PID-reused locks are recovered.
Future one-shots re-arm on startup. A one-shot missed while the app was closed is removed and
reintroduced only as a confirmation-required prompt; SwarmX does not silently execute stale work.
