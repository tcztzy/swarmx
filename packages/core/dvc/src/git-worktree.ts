import { chmodSync, mkdtempSync, realpathSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import type { ProcessHandle, ProcessOutcome, ProcessRunner } from "./process.js";

const FULL_OBJECT_ID = /^(?:[0-9a-f]{40}|[0-9a-f]{64})$/iu;

export type GitRuntimeErrorCode =
  | "COMMAND_FAILED"
  | "OUTPUT_INVALID"
  | "OUTPUT_TOO_LARGE"
  | "PATH_INVALID"
  | "REPOSITORY_INVALID"
  | "UNAVAILABLE"
  | "WORKTREE_FAILED";

export class GitRuntimeError extends Error {
  readonly code: GitRuntimeErrorCode;

  constructor(message: string, code: GitRuntimeErrorCode, options?: ErrorOptions) {
    super(message, options);
    this.name = "GitRuntimeError";
    this.code = code;
  }
}

export interface GitPublicSnapshot {
  readonly ahead: number | null;
  readonly behind: number | null;
  readonly branch: string | null;
  readonly clean: boolean;
  readonly conflicted: number;
  readonly head: string;
  readonly objectFormat: string;
  readonly staged: number;
  readonly unstaged: number;
  readonly untracked: number;
  readonly upstream: string | null;
  readonly version: string;
}

export interface GitWorkspaceSnapshot {
  readonly root: string;
  readonly public: GitPublicSnapshot;
}

export interface GitWorktreeHandle {
  readonly path: string;
  readonly revision: string;
  dispose(): Promise<void>;
}

export interface GitRuntimeConfig {
  readonly command: string;
  readonly graceMs: number;
  readonly maxOutputBytes: number;
}

interface CommandResult {
  readonly outcome: ProcessOutcome;
  readonly stderr: string;
  readonly stdout: string;
}

interface WorktreeState {
  readonly executable: string;
  readonly owner: string;
  path: string;
  readonly repositoryRoot: string;
  disposed: boolean;
}

interface StatusSummary {
  readonly ahead: number | null;
  readonly behind: number | null;
  readonly branch: string | null;
  readonly conflicted: number;
  readonly head: string;
  readonly staged: number;
  readonly unstaged: number;
  readonly untracked: number;
  readonly upstream: string | null;
}

function successful(result: CommandResult): boolean {
  return result.outcome.exitCode === 0 && result.outcome.signal === null;
}

function line(value: string): string {
  return value.replace(/[\r\n]+$/u, "");
}

function canonicalDirectory(input: string): string {
  if (input.length === 0 || input.includes("\0")) {
    throw new GitRuntimeError("Git workspace path is invalid", "PATH_INVALID");
  }
  try {
    const path = realpathSync(resolve(input));
    if (!statSync(path).isDirectory()) throw new Error("not a directory");
    return path;
  } catch (error) {
    if (error instanceof GitRuntimeError) throw error;
    throw new GitRuntimeError("Git workspace directory is unavailable", "PATH_INVALID", {
      cause: error,
    });
  }
}

function isContained(parent: string, child: string): boolean {
  const path = relative(parent, child);
  return path === "" || (!path.startsWith(`..${sep}`) && path !== ".." && !isAbsolute(path));
}

function parseStatus(output: string): StatusSummary {
  let ahead: number | null = null;
  let behind: number | null = null;
  let branch: string | null = null;
  let conflicted = 0;
  let head = "";
  let staged = 0;
  let unstaged = 0;
  let untracked = 0;
  let upstream: string | null = null;
  const records = output.split("\0");

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
        if (!match) throw new GitRuntimeError("Git status is malformed", "OUTPUT_INVALID");
        ahead = Number(match[1]);
        behind = Number(match[2]);
      }
      continue;
    }
    if (record.startsWith("? ")) {
      untracked += 1;
      continue;
    }
    if (record.startsWith("! ")) continue;
    if (record.startsWith("u ")) {
      conflicted += 1;
      continue;
    }
    const match = /^(1|2) ([^ ]{2}) /u.exec(record);
    const kind = match?.[1];
    const xy = match?.[2];
    if (kind === undefined || xy === undefined) {
      throw new GitRuntimeError("Git status is malformed", "OUTPUT_INVALID");
    }
    if (xy[0] !== ".") staged += 1;
    if (xy[1] !== ".") unstaged += 1;
    if (kind === "2") {
      index += 1;
      if ((records[index] ?? "") === "") {
        throw new GitRuntimeError("Git rename status is malformed", "OUTPUT_INVALID");
      }
    }
  }
  if (!FULL_OBJECT_ID.test(head)) {
    throw new GitRuntimeError("Git status has no committed HEAD", "OUTPUT_INVALID");
  }
  return {
    ahead,
    behind,
    branch,
    conflicted,
    head: head.toLowerCase(),
    staged,
    unstaged,
    untracked,
    upstream,
  };
}

/** Package-private Git identity and disposable-worktree runtime for DVC replay. */
export class GitWorktreeRuntime {
  private readonly active = new Set<ProcessHandle>();
  private open = true;
  private readonly worktrees = new Set<WorktreeState>();

  constructor(
    private readonly subprocess: ProcessRunner,
    private readonly config: GitRuntimeConfig,
  ) {}

  async inspect(cwd: string, signal?: AbortSignal): Promise<GitWorkspaceSnapshot> {
    this.ensureOpen();
    signal?.throwIfAborted();
    const directory = canonicalDirectory(cwd);
    const executable = await this.resolveExecutable(signal);
    const inside = await this.run(
      executable,
      directory,
      ["--no-optional-locks", "rev-parse", "--is-inside-work-tree"],
      signal,
    );
    if (!successful(inside) || line(inside.stdout) !== "true") {
      throw new GitRuntimeError(
        "DVC workspace is not inside a Git working tree",
        "REPOSITORY_INVALID",
      );
    }
    const rootResult = await this.required(
      executable,
      directory,
      ["--no-optional-locks", "rev-parse", "--show-toplevel"],
      signal,
    );
    const root = canonicalDirectory(line(rootResult.stdout));
    if (!isContained(root, directory)) {
      throw new GitRuntimeError("Git reported a root outside the workspace", "PATH_INVALID");
    }
    const headResult = await this.run(
      executable,
      root,
      ["--no-optional-locks", "rev-parse", "--verify", "HEAD^{commit}"],
      signal,
    );
    const head = line(headResult.stdout).toLowerCase();
    if (!successful(headResult) || !FULL_OBJECT_ID.test(head)) {
      throw new GitRuntimeError("Git repository has no committed HEAD", "REPOSITORY_INVALID");
    }
    const [versionResult, formatResult, statusResult] = await Promise.all([
      this.required(executable, root, ["--version"], signal),
      this.required(
        executable,
        root,
        ["--no-optional-locks", "rev-parse", "--show-object-format=storage"],
        signal,
      ),
      this.required(
        executable,
        root,
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
    const status = parseStatus(statusResult.stdout);
    if (status.head !== head) {
      throw new GitRuntimeError("Git HEAD changed during inspection", "OUTPUT_INVALID");
    }
    return {
      root,
      public: {
        ahead: status.ahead,
        behind: status.behind,
        branch: status.branch,
        clean:
          status.staged === 0 &&
          status.unstaged === 0 &&
          status.untracked === 0 &&
          status.conflicted === 0,
        conflicted: status.conflicted,
        head,
        objectFormat: line(formatResult.stdout),
        staged: status.staged,
        unstaged: status.unstaged,
        untracked: status.untracked,
        upstream: status.upstream,
        version: line(versionResult.stdout).replace(/^git version /u, ""),
      },
    };
  }

  async createDetachedWorktree(
    workspace: GitWorkspaceSnapshot,
    signal?: AbortSignal,
  ): Promise<GitWorktreeHandle> {
    this.ensureOpen();
    signal?.throwIfAborted();
    const executable = await this.resolveExecutable(signal);
    const revision = workspace.public.head;
    const verified = await this.run(
      executable,
      workspace.root,
      ["--no-optional-locks", "rev-parse", "--verify", `${revision}^{commit}`],
      signal,
    );
    if (!successful(verified) || line(verified.stdout).toLowerCase() !== revision) {
      throw new GitRuntimeError("Git commit could not be verified", "WORKTREE_FAILED");
    }

    const owner = realpathSync(mkdtempSync(join(tmpdir(), "swarmx-dvc-repro-")));
    chmodSync(owner, 0o700);
    const state: WorktreeState = {
      executable,
      owner,
      path: join(owner, "worktree"),
      repositoryRoot: workspace.root,
      disposed: false,
    };
    try {
      const added = await this.run(
        executable,
        workspace.root,
        ["--no-optional-locks", "worktree", "add", "--detach", state.path, revision],
        signal,
      );
      if (!successful(added)) {
        throw new GitRuntimeError("Git could not create a detached worktree", "WORKTREE_FAILED");
      }
      state.path = realpathSync(state.path);
      if (!isContained(owner, state.path)) {
        throw new GitRuntimeError("Git worktree escaped its private owner", "WORKTREE_FAILED");
      }
      this.worktrees.add(state);
    } catch (error) {
      await this.removeWorktree(state);
      if (signal?.aborted) signal.throwIfAborted();
      if (error instanceof GitRuntimeError) throw error;
      throw new GitRuntimeError("Git could not create a detached worktree", "WORKTREE_FAILED", {
        cause: error,
      });
    }
    return {
      path: state.path,
      revision,
      dispose: () => this.removeWorktree(state),
    };
  }

  async close(): Promise<void> {
    if (!this.open) return;
    const active = [...this.active];
    for (const handle of active) handle.terminate();
    await Promise.all(active.map((handle) => handle.waitForExit()));
    for (const worktree of [...this.worktrees]) await this.removeWorktree(worktree);
    this.open = false;
  }

  private async resolveExecutable(signal?: AbortSignal): Promise<string> {
    try {
      return await this.subprocess.resolveExecutable(this.config.command, {}, signal);
    } catch (error) {
      if (signal?.aborted) signal.throwIfAborted();
      throw new GitRuntimeError("Configured Git executable is unavailable", "UNAVAILABLE", {
        cause: error,
      });
    }
  }

  private async required(
    executable: string,
    cwd: string,
    args: readonly string[],
    signal?: AbortSignal,
  ): Promise<CommandResult> {
    const result = await this.run(executable, cwd, args, signal);
    if (!successful(result)) throw new GitRuntimeError("Git command failed", "COMMAND_FAILED");
    return result;
  }

  private async run(
    executable: string,
    cwd: string,
    args: readonly string[],
    signal?: AbortSignal,
  ): Promise<CommandResult> {
    this.ensureOpen();
    signal?.throwIfAborted();
    let handle: ProcessHandle;
    try {
      handle = this.subprocess.spawn({
        argv: [executable, ...args],
        cwd,
        stdio: {
          stdin: "ignore",
          stdout: { maxBytes: this.config.maxOutputBytes },
          stderr: { maxBytes: this.config.maxOutputBytes },
        },
        graceMs: this.config.graceMs,
        ...(signal === undefined ? {} : { signal }),
        env: { GIT_TERMINAL_PROMPT: "0", LC_ALL: "C" },
      });
    } catch (error) {
      if (signal?.aborted) signal.throwIfAborted();
      throw new GitRuntimeError("Git process could not be started", "UNAVAILABLE", {
        cause: error,
      });
    }
    this.active.add(handle);
    try {
      const outcome = await handle.done;
      if (signal?.aborted) signal.throwIfAborted();
      const stdout = handle.collected.stdout?.readFrom(0);
      const stderr = handle.collected.stderr?.readFrom(0);
      if (!stdout || !stderr) {
        throw new GitRuntimeError("Git output collection is unavailable", "COMMAND_FAILED");
      }
      if (stdout.lossy || stderr.lossy) {
        throw new GitRuntimeError("Git output exceeded its limit", "OUTPUT_TOO_LARGE");
      }
      return { outcome, stdout: stdout.text, stderr: stderr.text };
    } finally {
      this.active.delete(handle);
    }
  }

  private async removeWorktree(state: WorktreeState): Promise<void> {
    if (state.disposed) return;
    state.disposed = true;
    this.worktrees.delete(state);
    try {
      await this.runForCleanup(state);
    } finally {
      rmSync(state.owner, { recursive: true, force: true });
    }
  }

  private async runForCleanup(state: WorktreeState): Promise<void> {
    try {
      const handle = this.subprocess.spawn({
        argv: [
          state.executable,
          "--no-optional-locks",
          "worktree",
          "remove",
          "--force",
          state.path,
        ],
        cwd: state.repositoryRoot,
        stdio: {
          stdin: "ignore",
          stdout: { maxBytes: 4_096 },
          stderr: { maxBytes: 4_096 },
        },
        graceMs: this.config.graceMs,
        env: { GIT_TERMINAL_PROMPT: "0", LC_ALL: "C" },
      });
      await handle.done;
    } catch {
      // Only the exact generated owner is removed if Git cleanup is unavailable.
    }
  }

  private ensureOpen(): void {
    if (!this.open) throw new GitRuntimeError("Git runtime is closed", "COMMAND_FAILED");
  }
}
