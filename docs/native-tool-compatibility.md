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
