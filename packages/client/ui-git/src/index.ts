import type { Context } from "@deepseek-ai/cordis";
import { SessionId } from "@deepseek-ai/dsh-session";
import type { SubprocessHandle, SubprocessOutcome } from "@deepseek-ai/dsh-subprocess";
import { TypertRemoteService } from "@deepseek-ai/dsh-typert-protocol";
import s from "@deepseek-ai/schemastery";
import type { GitUiEntry, GitUiRepositorySnapshot, GitUiSnapshot } from "./contracts.js";

const DEFAULT_GRACE_MS = 2_000;
const DEFAULT_MAX_ENTRIES = 500;
const DEFAULT_MAX_OUTPUT_BYTES = 2 * 1024 * 1024;
const FULL_OBJECT_ID = /^(?:[0-9a-f]{40}|[0-9a-f]{64})$/iu;

export interface Config {
  readonly command?: string;
  readonly graceMs?: number;
  readonly maxEntries?: number;
  readonly maxOutputBytes?: number;
}

interface ResolvedConfig {
  readonly command: string;
  readonly graceMs: number;
  readonly maxEntries: number;
  readonly maxOutputBytes: number;
}

interface CommandResult {
  readonly outcome: SubprocessOutcome;
  readonly stderr: string;
  readonly stdout: string;
}

interface ParsedStatus {
  readonly ahead: number | null;
  readonly behind: number | null;
  readonly branch: string | null;
  readonly conflicted: number;
  readonly entries: readonly GitUiEntry[];
  readonly head: string;
  readonly staged: number;
  readonly truncated: boolean;
  readonly unstaged: number;
  readonly untracked: number;
  readonly upstream: string | null;
}

export class GitUiError extends Error {
  readonly code: "SESSION_NOT_FOUND" | "WORKSPACE_UNAVAILABLE";

  constructor(
    message: string,
    code: "SESSION_NOT_FOUND" | "WORKSPACE_UNAVAILABLE",
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "GitUiError";
    this.code = code;
  }
}

declare module "@deepseek-ai/cordis" {
  interface Context {
    gitUi: GitUiService;
  }
}

function successful(result: CommandResult): boolean {
  return result.outcome.exitCode === 0 && result.outcome.signal === null;
}

function line(value: string): string {
  return value.replace(/[\r\n]+$/u, "");
}

function boundedPath(path: string): string {
  if (path.length === 0 || path.length > 4_096 || path.includes("\0")) {
    throw new Error("Git returned an invalid repository-relative path");
  }
  return path;
}

function xyEntry(
  kind: GitUiEntry["kind"],
  xy: string,
  path: string,
  previousPath?: string,
): GitUiEntry {
  const index = xy[0];
  const worktree = xy[1];
  if (index === undefined || worktree === undefined) throw new Error("Git returned invalid status");
  return {
    kind,
    path: boundedPath(path),
    index,
    worktree,
    ...(previousPath === undefined ? {} : { previousPath: boundedPath(previousPath) }),
  };
}

function parseStatus(output: string, maxEntries: number): ParsedStatus {
  let ahead: number | null = null;
  let behind: number | null = null;
  let branch: string | null = null;
  let conflicted = 0;
  let head = "";
  let staged = 0;
  let truncated = false;
  let unstaged = 0;
  let untracked = 0;
  let upstream: string | null = null;
  const entries: GitUiEntry[] = [];
  const records = output.split("\0");
  const publish = (entry: GitUiEntry) => {
    if (entries.length < maxEntries) entries.push(entry);
    else truncated = true;
  };

  for (let index = 0; index < records.length; index += 1) {
    const record = records[index] ?? "";
    if (record === "") continue;
    if (record.startsWith("# ")) {
      if (record.startsWith("# branch.oid ")) head = record.slice("# branch.oid ".length);
      else if (record.startsWith("# branch.head ")) {
        const value = record.slice("# branch.head ".length);
        branch = value === "(detached)" ? null : value;
      } else if (record.startsWith("# branch.upstream ")) {
        upstream = record.slice("# branch.upstream ".length);
      } else if (record.startsWith("# branch.ab ")) {
        const match = /^# branch\.ab \+(\d+) -(\d+)$/u.exec(record);
        if (!match) throw new Error("Git returned invalid divergence counts");
        ahead = Number(match[1]);
        behind = Number(match[2]);
      }
      continue;
    }
    if (record.startsWith("? ")) {
      untracked += 1;
      publish(xyEntry("untracked", "??", record.slice(2)));
      continue;
    }
    if (record.startsWith("! ")) continue;

    const ordinary = /^1 ([^ ]{2}) [^ ]+ [^ ]+ [^ ]+ [^ ]+ [^ ]+ [^ ]+ (.*)$/u.exec(record);
    if (ordinary) {
      const xy = ordinary[1];
      const path = ordinary[2];
      if (xy === undefined || path === undefined) throw new Error("Git returned invalid status");
      if (xy[0] !== ".") staged += 1;
      if (xy[1] !== ".") unstaged += 1;
      publish(xyEntry("ordinary", xy, path));
      continue;
    }

    const renamed = /^2 ([^ ]{2}) [^ ]+ [^ ]+ [^ ]+ [^ ]+ [^ ]+ [^ ]+ [^ ]+ (.*)$/u.exec(record);
    if (renamed) {
      const xy = renamed[1];
      const path = renamed[2];
      const previousPath = records[index + 1];
      if (
        xy === undefined ||
        path === undefined ||
        previousPath === undefined ||
        previousPath === ""
      ) {
        throw new Error("Git returned invalid rename status");
      }
      index += 1;
      if (xy[0] !== ".") staged += 1;
      if (xy[1] !== ".") unstaged += 1;
      publish(xyEntry("renamed", xy, path, previousPath));
      continue;
    }

    const unmerged = /^u ([^ ]{2}) [^ ]+ [^ ]+ [^ ]+ [^ ]+ [^ ]+ [^ ]+ [^ ]+ [^ ]+ (.*)$/u.exec(
      record,
    );
    if (unmerged) {
      const xy = unmerged[1];
      const path = unmerged[2];
      if (xy === undefined || path === undefined)
        throw new Error("Git returned invalid conflict status");
      conflicted += 1;
      publish(xyEntry("unmerged", xy, path));
      continue;
    }
    throw new Error("Git returned an unknown status record");
  }

  if (!FULL_OBJECT_ID.test(head)) throw new Error("Git did not report a committed HEAD");
  return {
    ahead,
    behind,
    branch,
    conflicted,
    entries,
    head: head.toLowerCase(),
    staged,
    truncated,
    unstaged,
    untracked,
    upstream,
  };
}

export class GitUiService extends TypertRemoteService {
  static inject = ["sessions", "subprocess"];
  static Config = s.object({
    command: s.string().default("git"),
    graceMs: s.natural().min(1).max(60_000).default(DEFAULT_GRACE_MS),
    maxEntries: s.natural().min(1).max(5_000).default(DEFAULT_MAX_ENTRIES),
    maxOutputBytes: s
      .natural()
      .min(4_096)
      .max(16 * 1024 * 1024)
      .default(DEFAULT_MAX_OUTPUT_BYTES),
  });

  private readonly active = new Set<SubprocessHandle>();
  private readonly config: ResolvedConfig;
  private open = true;

  constructor(ctx: Context, config: Config) {
    super(ctx, "gitUi");
    this.config = {
      command: config.command ?? "git",
      graceMs: config.graceMs ?? DEFAULT_GRACE_MS,
      maxEntries: config.maxEntries ?? DEFAULT_MAX_ENTRIES,
      maxOutputBytes: config.maxOutputBytes ?? DEFAULT_MAX_OUTPUT_BYTES,
    };
    ctx.effect(() => () => this.close(), "dsh-ui-git: close status processes");
  }

  async snapshot(sessionId: SessionId, signal?: AbortSignal): Promise<GitUiSnapshot> {
    signal?.throwIfAborted();
    const cwd = this.workspace(sessionId);
    let executable: string;
    try {
      executable = await this.ctx.subprocess.resolveExecutable(this.config.command, {}, signal);
    } catch {
      if (signal?.aborted) signal.throwIfAborted();
      return { kind: "unavailable", message: "Git executable is unavailable" };
    }

    try {
      const inside = await this.run(
        executable,
        cwd,
        ["--no-optional-locks", "rev-parse", "--is-inside-work-tree"],
        signal,
      );
      if (!successful(inside) || line(inside.stdout) !== "true") {
        return { kind: "not-repository", message: "Workspace is not a Git repository" };
      }
      const headResult = await this.run(
        executable,
        cwd,
        ["--no-optional-locks", "rev-parse", "--verify", "HEAD^{commit}"],
        signal,
      );
      if (!successful(headResult) || !FULL_OBJECT_ID.test(line(headResult.stdout))) {
        return { kind: "not-repository", message: "Git repository has no committed HEAD" };
      }
      const [versionResult, formatResult, statusResult] = await Promise.all([
        this.run(executable, cwd, ["--version"], signal),
        this.run(
          executable,
          cwd,
          ["--no-optional-locks", "rev-parse", "--show-object-format=storage"],
          signal,
        ),
        this.run(
          executable,
          cwd,
          [
            "--no-optional-locks",
            "status",
            "--porcelain=v2",
            "-z",
            "--branch",
            "--untracked-files=all",
          ],
          signal,
        ),
      ]);
      if (![versionResult, formatResult, statusResult].every(successful)) {
        return { kind: "unavailable", message: "Git status could not be read" };
      }
      const status = parseStatus(statusResult.stdout, this.config.maxEntries);
      const head = line(headResult.stdout).toLowerCase();
      if (status.head !== head) {
        return { kind: "unavailable", message: "Git HEAD changed while status was read" };
      }
      const snapshot: GitUiRepositorySnapshot = {
        kind: "repository",
        version: line(versionResult.stdout).replace(/^git version /u, ""),
        objectFormat: line(formatResult.stdout),
        head,
        branch: status.branch,
        upstream: status.upstream,
        ahead: status.ahead,
        behind: status.behind,
        clean:
          status.staged === 0 &&
          status.unstaged === 0 &&
          status.untracked === 0 &&
          status.conflicted === 0,
        staged: status.staged,
        unstaged: status.unstaged,
        untracked: status.untracked,
        conflicted: status.conflicted,
        truncated: status.truncated,
        entries: [...status.entries],
      };
      return snapshot;
    } catch {
      if (signal?.aborted) signal.throwIfAborted();
      return { kind: "unavailable", message: "Git status could not be read" };
    }
  }

  private workspace(sessionId: SessionId): string {
    const session = this.ctx.sessions.get(SessionId(sessionId));
    if (!session) throw new GitUiError("Live session not found", "SESSION_NOT_FOUND");
    const cwd = session.header.cwd;
    if (!cwd) throw new GitUiError("Session has no workspace directory", "WORKSPACE_UNAVAILABLE");
    return cwd;
  }

  private async run(
    executable: string,
    cwd: string,
    args: readonly string[],
    signal?: AbortSignal,
  ): Promise<CommandResult> {
    if (!this.open) throw new Error("Git UI service is closed");
    signal?.throwIfAborted();
    const handle = this.ctx.subprocess.spawn({
      argv: [executable, ...args],
      cwd,
      stdio: {
        stdin: "ignore",
        stdout: { maxBytes: this.config.maxOutputBytes },
        stderr: { maxBytes: this.config.maxOutputBytes },
      },
      graceMs: this.config.graceMs,
      signal,
      env: { GIT_TERMINAL_PROMPT: "0", LC_ALL: "C" },
    });
    this.active.add(handle);
    try {
      const outcome = await handle.done;
      if (signal?.aborted) signal.throwIfAborted();
      const stdout = handle.collected.stdout?.readFrom(0);
      const stderr = handle.collected.stderr?.readFrom(0);
      if (!stdout || !stderr || stdout.lossy || stderr.lossy) {
        throw new Error("Git output exceeded its bound");
      }
      return { outcome, stdout: stdout.text, stderr: stderr.text };
    } finally {
      this.active.delete(handle);
    }
  }

  private async close(): Promise<void> {
    if (!this.open) return;
    this.open = false;
    const active = [...this.active];
    for (const handle of active) handle.terminate();
    await Promise.all(active.map((handle) => handle.waitForExit()));
  }
}

export * from "./contracts.js";
export { GIT_UI_REMOTE } from "./remote.js";
export default GitUiService;
