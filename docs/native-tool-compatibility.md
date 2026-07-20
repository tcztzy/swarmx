# Model-trained tool compatibility

SwarmX exposes Project tools using the public interfaces that Claude Code and
OpenAI Codex models see in their native hosts. The goal is interface
compatibility, not host impersonation: SwarmX keeps its own containment,
stale-file, output, timeout, and sandbox policies.

## Why align the interface

Anthropic explicitly describes its Bash and text-editor tools as
"trained-in": Claude is trained on successful trajectories using those tool
interfaces and should call them more reliably than unfamiliar equivalents.
See [How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works)
and the [tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference).
This supports preserving names, argument spelling, required fields, and input
mode when a model family has a well-known native host.

There are two related Anthropic interfaces:

- The Claude API offers versioned server/client tools such as
  `bash_20250124`, `text_editor_20250728`, `web_search_20250305`, and
  `web_fetch_20250910`.
- Claude Code exposes client-side coding tools. The public Claude Agent SDK
  types use `Bash`, `Read`, `Edit`, `Write`, `Glob`, and `Grep`, with fields
  such as `command`, `file_path`, `old_string`, `new_string`, `pattern`, and
  `output_mode`.

SwarmX's direct Project runtime follows the Claude Code surface because it is
the coding-agent host being matched. The definitions were audited against
Claude Code 2.1.187 and `@anthropic-ai/claude-agent-sdk` 0.3.211.

OpenAI Codex exposes `exec_command` as a JSON-schema function tool and
`apply_patch` as a freeform custom tool constrained by a Lark grammar. The
definitions were audited against Codex CLI 0.144.4 at tag
[`rust-v0.144.4`](https://github.com/openai/codex/tree/rust-v0.144.4), including
[`shell_spec.rs`](https://github.com/openai/codex/blob/rust-v0.144.4/codex-rs/core/src/tools/handlers/shell_spec.rs),
[`apply_patch_spec.rs`](https://github.com/openai/codex/blob/rust-v0.144.4/codex-rs/core/src/tools/handlers/apply_patch_spec.rs),
and [`apply_patch.lark`](https://github.com/openai/codex/blob/rust-v0.144.4/codex-rs/core/src/tools/handlers/apply_patch.lark).

## Runtime profiles

| Model family | Project profile | Exposed tools |
| --- | --- | --- |
| Claude, Sonnet, Opus, Haiku, Fable | `claude_code` | `Bash`, `Read`, `Edit`, `Write`, `Glob`, `Grep`, `NotebookEdit`, `ReportFindings`, `AskUserQuestion`, `EnterPlanMode`, `ExitPlanMode`, `TaskCreate`, `TaskGet`, `TaskList`, `TaskUpdate`, `TodoWrite`, `TaskOutput`, `TaskStop` |
| Other direct models | `codex` | `exec_command`, `write_stdin`, `apply_patch` |

Only direct `swarmx` compositions receive these tools. ACP harnesses such as
Claude Code or Codex already own their host tools, so SwarmX does not inject a
second, conflicting copy.

## Permission model and source audit

Permissions and sandboxing are separate controls. Claude Code evaluates
fine-grained deny, ask, and allow rules with deny taking precedence, then uses
OS-level filesystem/network isolation as a second boundary. It offers default,
accept-edits, plan, auto, pre-approved-only, and bypass modes; managed settings
can prevent local overrides and disable bypass. See the official
[permissions](https://code.claude.com/docs/en/permissions),
[permission modes](https://code.claude.com/docs/en/permission-modes), and
[sandboxing](https://code.claude.com/docs/en/sandboxing) documentation.

Codex likewise separates `sandbox_mode` from `approval_policy`. Its normal local
automation preset uses workspace-scoped writes, no command network access, and
on-request approval for crossing the boundary. Read-only and full-access modes,
protected repository/config paths, command-prefix rules, granular approval
categories, managed requirements, and optional approval review remain distinct
controls. See the official [sandboxing](https://learn.chatgpt.com/docs/sandboxing),
[configuration](https://learn.chatgpt.com/docs/config-file/config-advanced#approval-policies-and-sandbox-modes),
[rules](https://learn.chatgpt.com/docs/agent-configuration/rules), and
[desktop permissions](https://learn.chatgpt.com/docs/permissions) pages.

SwarmX follows the common core without pretending that its modes are identical
to either vendor:

| Harness mode | Direct SwarmX behavior |
| --- | --- |
| `default` | Read-only tools run; remaining Project tool calls require one-call desktop approval. |
| `auto` | Read-only and Project-contained write calls run after deterministic host review; execute/control calls still require one-call approval. This does not claim Codex's separate reviewer-model implementation. |
| `plan` | Hard read-only boundary; allow rules cannot enable mutation or execution. |
| `restricted` | Read-only and explicitly pre-approved tools run; remaining calls fail closed. |
| `trusted` | Tools run without prompts, but only inside unchanged Project path, Seatbelt, network, environment, output, timeout, stale-write, and cancellation boundaries. |

Exact `deniedTools` rules win first. Outside plan mode, exact `allowedTools`
rules pre-approve a tool. Unknown modes fail schema validation. Approval changes
only one call's permission decision; `dangerouslyDisableSandbox` and
`sandbox_permissions=require_escalated` remain rejected.

ACP Harnesses keep their own option semantics. SwarmX Desktop preserves the
offered option ids and kinds, bounds their display names, and returns only the
explicitly selected offered id. Headless Core callers without a permission
handler still return ACP `cancelled`. The desktop prompt carries only bounded
title/kind/summary data and never ACP raw input/output, patch bodies, file
contents, or credentials.

### Layered governance and desktop UX

Direct SwarmX policy resolves four durable sources plus an optional conversation
override in least-authority order:

1. `SWARMX_MANAGED_PERMISSION_POLICY` supplies an optional read-only managed
   JSON layer.
2. `<Project>/.swarmx/permissions.json` supplies a read-only, restriction-only
   Project layer. It may set a mode ceiling or deny exact tools, but it cannot
   pre-approve tools.
3. Desktop Settings supplies editable personal defaults.
4. The selected Custom Agent supplies its Harness policy.

A persisted conversation may additionally select `inherit`, `default`, `auto`,
`plan`, or `trusted`. `inherit` keeps the four durable layers. An explicit
conversation choice replaces the personal and Agent mode defaults for that
conversation, but managed/Project mode ceilings and every explicit deny rule
remain in force.
The Main process reads this value from the authoritative saved session before
creating direct tools and again for background activations. External ACP
Harnesses retain their own permission semantics.

The least-authority declared mode wins in the order `plan < restricted <
default < auto < trusted`. Denials union across all layers and remove any matching
pre-approval. Missing managed/Project sources inherit neutrally; malformed
configured sources are visible and fail closed before direct tools are
created. This prevents repository content from granting authority while still
letting a Project enforce local restrictions.

General Settings follows Codex's three-row information architecture: independent
Default permissions, Auto-review, and Full access switches control whether those
profiles are available to conversations. They default on and persist separately.
If a disabled profile remains in an older conversation or inherited personal or
Agent policy, authoritative Main-process resolution substitutes `plan`; hiding an
option in the Renderer is never the safety boundary. Advanced permissions owns
the inherited fallback (including Plan only and Restricted), effective mode,
exact allow/deny counts, source provenance, editable rules, selected Project
policy path, and newest-first local approval history. Every direct SwarmX
conversation Composer exposes a compact local choice before send and persists it
immediately for an existing session or with creation of a new session.
Custom Agents use structured exact-tool chips with duplicate/conflict checking
instead of newline rule text. Approval dialogs explain that allow-once applies
to one call and does not expand the host sandbox; rejection receives initial
focus.

Approval receipts are bounded to the newest 200 entries and contain only id,
time, direct/ACP source, bounded tool name/kind, decision, offered option kind,
and policy-source ids. They never create an allow rule and never store raw tool
arguments, output, patch/file content, environment values, or credentials.

Remaining staged follow-up:

1. Add tool-argument scopes (path, command prefix, MCP server/tool, and network
   domain) plus protected-path policy; exact tool names are intentionally the
   first narrow contract.
2. Add policy simulation and organization-managed distribution/signature
   metadata for administrators.
3. Add MCP/app side-effect annotations and shared enforcement so direct local,
   MCP, ACP, and connector actions use one auditable decision vocabulary.

The three interactive names require the desktop request bridge. Headless callers
without that bridge receive the 15 non-interactive names and do not receive
placeholder prompts that can never resolve.

After at least one configured MCP server connects, the Claude profile also
adds `ListMcpResourcesTool` and `ReadMcpResourceTool`. Text resources retain
their MCP URI and MIME type. Binary resources fail explicitly until an
authorized Project file sink is available.

When the active composition selects filesystem-backed Skills, the profile adds
`Skill({ skill, args? })`. Its description lists only those selected Skills;
invocation loads the configured `SKILL.md` under a 512 KiB cap and expands
Claude Code argument, Skill-directory, effort, and session substitutions. The
model cannot supply a path or invoke an unselected Skill.

## Interactive questions and plan mode

`AskUserQuestion` sends a request/interaction-scoped event to the owning desktop
window and suspends the tool call. The dialog accepts 1-4 questions, 2-4 choices
per question, single or multiple selection, and an automatic free-text Other
choice. It has no default idle timeout. A different window cannot answer it;
task cancellation or window destruction rejects and cleans the pending call.
The Agent SDK-compatible structured result is:

```json
{
  "questions": [
    {
      "question": "Which runtime?",
      "header": "Runtime",
      "options": [
        { "label": "Node", "description": "Use Node.js" },
        { "label": "Bun", "description": "Use Bun" }
      ],
      "multiSelect": false
    }
  ],
  "answers": { "Which runtime?": "Node" }
}
```

`EnterPlanMode` creates a mode-private 0600 plan file under a private temporary
directory. Its result tells Claude the exact path. In plan mode `Bash`, `Edit`,
`NotebookEdit`, and ordinary `Write` calls fail; only `Read` and `Write` for that
exact plan file bypass Project-root containment. `ExitPlanMode` reads that file
and displays its contents for explicit approval. Rejection returns feedback and
keeps read-only mode active. Approval returns the documented `plan`, `isAgent`,
`filePath`, and `planWasEdited` fields, then restores normal execution. Manager
close deletes the private plan directory.

Compatibility targets names, input modes, principal argument schemas, and the
split between model-facing output and client-facing structured output. A tool
call resolves to:

```ts
{
  content: string;             // sent back to the model
  structuredContent?: unknown; // retained on the SwarmX tool-result event
  isError: boolean;
}
```

The model never receives that outer SwarmX object. For example, Claude `Read`
receives numbered text such as `     1→const ready = true;`, while the client
event carries the Agent SDK-compatible value:

```json
{
  "type": "text",
  "file": {
    "filePath": "/project/src/index.ts",
    "content": "const ready = true;\n",
    "numLines": 1,
    "startLine": 1,
    "totalLines": 1
  }
}
```

Compatible MCP adapters that still return a plain JSON object are normalized at
the same boundary: JSON remains the model-facing text and the original object
becomes `structuredContent`.

`Edit` and `Write` similarly expose `filePath`, `originalFile`, and
`structuredPatch`; `Glob` exposes `durationMs`, `numFiles`, `filenames`, and
truncation metadata; `Grep` exposes mode, file/match counts, content, limit,
and offset metadata. `Bash` exposes `stdout`, `stderr`, `interrupted`, and the
optional Agent SDK fields used for background and timeout state.

Codex terminal calls send its formatted text back to the model:

```text
Chunk ID: 0ebee7c1-...
Wall time: 0.2500 seconds
Process running with session ID 1
Output:
starting
```

The SwarmX event simultaneously carries the Codex output-schema value:

```json
{
  "chunk_id": "0ebee7c1-...",
  "wall_time_seconds": 0.25,
  "session_id": 1,
  "output": "starting\n"
}
```

When the process finishes during a later `write_stdin` call, `session_id` is
omitted and `exit_code` is present. `original_token_count` appears only when
captured output had to be truncated.

## Background command lifecycle

Claude `Bash({ run_in_background: true })` starts a real sandboxed process and
returns a string `backgroundTaskId`. `TaskOutput` can poll or wait up to its
bounded timeout; `TaskStop` terminates the whole process group. Codex
`exec_command` waits for `yield_time_ms`, then returns a numeric `session_id`
when the command is still alive. `write_stdin` writes pipe-backed stdin or,
for `tty: true`, writes bytes to a real pseudoterminal. Empty `chars` polls for
incremental output in either mode.

Background commands use the same canonical Project root, scrubbed environment,
Seatbelt network denial, output cap, and process-group termination as foreground
commands. Default background runtime is one hour; host configuration may lower
it or raise it only to the 24-hour hard cap. Completed sessions remain readable
for five minutes. Request cancellation or tool-manager close terminates every
live session and removes its private temporary directory. Sessions never persist
across tasks.

`exec_command({ tty: true })` allocates an 80x24 `node-pty` terminal around the
same `sandbox-exec` command. The child observes a real terminal on stdin,
stdout, and stderr, with `TERM=xterm-256color`; omitted or false `tty` retains
plain pipes and `TERM=dumb`. PTY output is necessarily one merged terminal
stream, so SwarmX reports it as `stdout`/Codex `output` and leaves structured
`stderr` empty. `write_stdin` supports line input and control bytes such as
Ctrl-C. PTY sessions retain the same workdir validation, scrubbed environment,
Seatbelt policy, runtime/output caps, cancellation, process-group termination,
retention, and cleanup rules. Seatbelt adds `file-ioctl` only for `/dev/tty`
and `/dev/ttys*`, enabling canonical input and `stty` without broadening file
or network access. PDF page reads remain unsupported.

A foreground Claude `Bash` command that outlives its requested timeout is kept
in the same managed session and returned with `backgroundTaskId`, matching
Claude Code's timeout-to-background lifecycle. Commands whose first executable
is `sleep` retain terminal timeout behavior. SwarmX does not yet persist the
uncapped command stream to a session output file, so large-output spill-file
parity remains open.

For OpenAI Responses, `apply_patch` uses a custom/freeform tool and continues
with `custom_tool_call_output`. APIs that only support JSON function tools get
an object fallback containing one `patch` string. Both routes call the same
guarded patch handler.

The schemas intentionally tolerate unknown optional fields. Native hosts can
be permissive when a model emits a newly introduced or unused parameter.
Tolerance never changes authority: missing required input, path traversal,
escaping symlinks, stale files, sandbox escalation, and invalid patch
operations remain errors. Selecting PTY changes terminal semantics only;
it does not grant more filesystem or network authority.

The complete 42-tool baseline and current gaps are tracked in
[`claude-code-tool-parity.md`](./claude-code-tool-parity.md). A tool absent from
that profile is not claimed as supported; provider/account-only operations are
not represented by local success stubs.

## Network boundary

Web search is not represented as a local shell or fake function tool.
Anthropic documents [web search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)
and [web fetch](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool)
as server-executed tools. Codex likewise registers search through its hosted
Responses tool path. SwarmX should expose search only after the selected
provider and protocol advertise that hosted capability; the local Project
shell continues to deny network access.

## Maintenance checklist

When upgrading Claude Code, the Claude Agent SDK, Codex, Anthropic SDK, or
OpenAI SDK:

1. Diff public tool names, schemas, required fields, and freeform grammars.
2. Update profile contract tests before changing adapters.
3. Preserve SwarmX security checks even if the upstream host adds a more
   permissive execution option.
4. Re-test JSON function continuation and Responses custom-tool continuation.
5. Re-test background start, wait, stdin, stop, cancellation, timeout, and close.
6. Update versions, source links, and third-party notices in the same change.
