import { createHash } from "node:crypto";
import {
  chmodSync,
  copyFileSync,
  existsSync,
  lstatSync,
  readFileSync,
  realpathSync,
  statSync,
} from "node:fs";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import { z } from "zod";
import {
  type GitPublicSnapshot,
  GitRuntimeError,
  type GitWorkspaceSnapshot,
  type GitWorktreeHandle,
  GitWorktreeRuntime,
} from "./git-worktree.js";
import type { ProcessHandle, ProcessOutcome, ProcessRunner } from "./process.js";

export * from "./process.js";

const DEFAULT_GRACE_MS = 2_000;
const DEFAULT_MAX_MANIFEST_BYTES = 4 * 1024 * 1024;
const DEFAULT_MAX_OUTPUT_BYTES = 1024 * 1024;
const MAX_CONFIG_LOCAL_BYTES = 1024 * 1024;
const MAX_TARGETS = 32;
const REMOTE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u;
const WINDOWS_ABSOLUTE = /^(?:[A-Za-z]:[\\/]|\\\\)/u;
const SAFE_CATEGORIES = new Set([
  "added",
  "changed",
  "changed deps",
  "changed outs",
  "committed",
  "deleted",
  "missing",
  "modified",
  "not_in_cache",
  "not_in_remote",
  "not_in_workspace",
  "uncommitted",
  "unchanged",
]);
const ConfigSchema = z.strictObject({
  command: z.string().min(1).optional(),
  gitCommand: z.string().min(1).optional(),
  graceMs: z.number().int().min(1).max(60_000).optional(),
  maxManifestBytes: z
    .number()
    .int()
    .min(1_024)
    .max(16 * 1024 * 1024)
    .optional(),
  maxOutputBytes: z
    .number()
    .int()
    .min(4_096)
    .max(16 * 1024 * 1024)
    .optional(),
});

export type DvcErrorCode =
  | "DVC_CLOSED"
  | "DVC_COMMAND_FAILED"
  | "DVC_GIT_INVALID"
  | "DVC_GIT_UNAVAILABLE"
  | "DVC_MANIFEST_INVALID"
  | "DVC_OUTPUT_TOO_LARGE"
  | "DVC_REQUEST_INVALID"
  | "DVC_STATUS_INVALID"
  | "DVC_UNAVAILABLE"
  | "DVC_WORKSPACE_DIRTY"
  | "DVC_WORKTREE_FAILED"
  | "NOT_A_DVC_REPOSITORY";

export class DvcError extends Error {
  readonly code: DvcErrorCode;

  constructor(message: string, code: DvcErrorCode, options?: ErrorOptions) {
    super(message, options);
    this.name = "DvcError";
    this.code = code;
  }
}

export interface DvcStatusCategory {
  readonly count: number;
  readonly name: string;
}

export interface DvcStatusSummary {
  readonly categories: readonly DvcStatusCategory[];
  readonly digest: string;
  readonly entries: number;
}

export interface DvcInspection {
  readonly data: DvcStatusSummary;
  readonly dvcLockDigest: string | null;
  readonly dvcYamlDigest: string | null;
  readonly git: GitPublicSnapshot;
  /** DVC root relative to the Git root, or `.`. */
  readonly root: string;
  readonly pipeline: DvcStatusSummary;
  readonly version: string;
}

export interface DvcRequest {
  readonly remote?: string;
  readonly targets?: readonly string[];
}

export interface DvcReproduceRequest {
  readonly pull?: boolean;
  readonly remote?: string;
  readonly targets?: readonly string[];
}

export interface DvcCommandOutput {
  readonly text: string;
  readonly truncated: boolean;
}

export interface DvcCommandResult {
  readonly exitCode: number | null;
  readonly signal: NodeJS.Signals | null;
  readonly status: "cancelled" | "failed" | "succeeded";
  readonly stderr: DvcCommandOutput;
  readonly stdout: DvcCommandOutput;
}

export interface DvcReproductionHandle {
  readonly after: DvcInspection | null;
  /** Host-only DVC root inside the disposable worktree. */
  readonly path: string;
  readonly result: DvcCommandResult;
  readonly source: DvcInspection;
  dispose(): Promise<void>;
}

export interface Config {
  readonly command?: string;
  readonly gitCommand?: string;
  readonly graceMs?: number;
  readonly maxManifestBytes?: number;
  readonly maxOutputBytes?: number;
}

interface ResolvedConfig {
  readonly command: string;
  readonly graceMs: number;
  readonly maxManifestBytes: number;
  readonly maxOutputBytes: number;
}

interface CommandCapture {
  readonly outcome: ProcessOutcome;
  readonly stderr: DvcCommandOutput;
  readonly stdout: DvcCommandOutput;
}

interface DvcProject {
  readonly executable: string;
  readonly git: GitWorkspaceSnapshot;
  readonly root: string;
}

interface ReproductionState {
  disposed: boolean;
  readonly worktree: GitWorktreeHandle;
}

function successful(capture: CommandCapture): boolean {
  return capture.outcome.exitCode === 0 && capture.outcome.signal === null;
}

function line(value: string): string {
  return value.replace(/[\r\n]+$/u, "");
}

function isContained(parent: string, child: string): boolean {
  const path = relative(parent, child);
  return path === "" || (!path.startsWith(`..${sep}`) && path !== ".." && !isAbsolute(path));
}

function canonicalDirectory(input: string): string {
  try {
    const path = realpathSync(resolve(input));
    if (!statSync(path).isDirectory()) throw new Error("not a directory");
    return path;
  } catch (error) {
    throw new DvcError("DVC workspace directory is unavailable", "NOT_A_DVC_REPOSITORY", {
      cause: error,
    });
  }
}

function gitRelative(root: string, path: string): string {
  const value = relative(root, path);
  return value === "" ? "." : value.split(sep).join("/");
}

function digest(bytes: string | Buffer): string {
  return `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
}

function canonicalJson(value: z.infer<ReturnType<typeof z.json>>): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.keys(value)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key] as never)}`)
    .join(",")}}`;
}

function leafCount(value: z.infer<ReturnType<typeof z.json>>): number {
  if (value === null || typeof value !== "object") return 1;
  let count = 0;
  for (const item of Array.isArray(value) ? value : Object.values(value)) {
    count += leafCount(item);
  }
  return count;
}

function summarizeStatus(text: string): DvcStatusSummary {
  let value: z.infer<ReturnType<typeof z.json>>;
  try {
    value = z.json().parse(JSON.parse(text));
  } catch (error) {
    throw new DvcError("DVC returned invalid status JSON", "DVC_STATUS_INVALID", {
      cause: error,
    });
  }
  const counts = new Map<string, number>();
  const visit = (current: z.infer<ReturnType<typeof z.json>>): void => {
    if (current === null || typeof current !== "object") return;
    if (Array.isArray(current)) {
      for (const item of current) visit(item);
      return;
    }
    for (const [key, item] of Object.entries(current)) {
      if (SAFE_CATEGORIES.has(key)) counts.set(key, (counts.get(key) ?? 0) + leafCount(item));
      visit(item);
    }
  };
  visit(value);
  const entries = leafCount(value);
  if (counts.size === 0 && entries > 0) counts.set("other", entries);
  const canonical = canonicalJson(value);
  return {
    categories: [...counts]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([name, count]) => ({ name, count })),
    digest: digest(canonical),
    entries,
  };
}

function manifestDigest(
  root: string,
  name: "dvc.lock" | "dvc.yaml",
  maxBytes: number,
): string | null {
  const path = join(root, name);
  if (!existsSync(path)) return null;
  let metadata: ReturnType<typeof lstatSync>;
  try {
    metadata = lstatSync(path);
  } catch (error) {
    throw new DvcError(`DVC ${name} could not be inspected`, "DVC_MANIFEST_INVALID", {
      cause: error,
    });
  }
  if (!metadata.isFile() || metadata.isSymbolicLink() || metadata.size > maxBytes) {
    throw new DvcError(`DVC ${name} must be a bounded regular file`, "DVC_MANIFEST_INVALID");
  }
  return digest(readFileSync(path));
}

function validateTargets(input: readonly string[] | undefined): readonly string[] {
  if (input === undefined) return [];
  if (input.length > MAX_TARGETS) {
    throw new DvcError("DVC request has too many targets", "DVC_REQUEST_INVALID");
  }
  return input.map((target) => {
    if (
      target.length === 0 ||
      target.length > 500 ||
      target.startsWith("-") ||
      [...target].some((character) => {
        const code = character.charCodeAt(0);
        return code <= 0x1f || code === 0x7f;
      }) ||
      isAbsolute(target) ||
      WINDOWS_ABSOLUTE.test(target) ||
      target.split(/[\\/]/u).includes("..")
    ) {
      throw new DvcError("DVC target is invalid", "DVC_REQUEST_INVALID");
    }
    return target;
  });
}

function validateRemote(remote: string | undefined): string | undefined {
  if (remote !== undefined && !REMOTE_NAME.test(remote)) {
    throw new DvcError("DVC remote name is invalid", "DVC_REQUEST_INVALID");
  }
  return remote;
}

function redact(text: string, paths: readonly string[]): string {
  let value = text;
  for (const path of [...paths].sort((left, right) => right.length - left.length)) {
    if (path.length > 0) value = value.replaceAll(path, "<workspace>");
  }
  return value
    .replace(/([a-z][a-z0-9+.-]*:\/\/)[^/@\s]+@/giu, "$1***@")
    .replace(/\b(password|secret|token)=([^\s]+)/giu, "$1=<redacted>");
}

function resultFrom(
  capture: CommandCapture,
  signal: AbortSignal | undefined,
  paths: readonly string[],
): DvcCommandResult {
  return {
    exitCode: capture.outcome.exitCode,
    signal: capture.outcome.signal,
    status: signal?.aborted ? "cancelled" : successful(capture) ? "succeeded" : "failed",
    stderr: { ...capture.stderr, text: redact(capture.stderr.text, paths) },
    stdout: { ...capture.stdout, text: redact(capture.stdout.text, paths) },
  };
}

function mapGitError(error: unknown): DvcError {
  if (!(error instanceof GitRuntimeError)) {
    return new DvcError("Git operation failed", "DVC_GIT_INVALID", { cause: error });
  }
  if (error.code === "UNAVAILABLE") {
    return new DvcError("Configured Git executable is unavailable", "DVC_GIT_UNAVAILABLE", {
      cause: error,
    });
  }
  if (error.code === "WORKTREE_FAILED") {
    return new DvcError("Disposable Git worktree could not be created", "DVC_WORKTREE_FAILED", {
      cause: error,
    });
  }
  return new DvcError("DVC requires a committed non-bare Git workspace", "DVC_GIT_INVALID", {
    cause: error,
  });
}

export class DvcService {
  private readonly active = new Set<ProcessHandle>();
  private readonly config: ResolvedConfig;
  private readonly git: GitWorktreeRuntime;
  private readonly mutations = new Map<string, Promise<void>>();
  private open = true;
  private readonly reproductions = new Set<ReproductionState>();

  constructor(
    private readonly subprocess: ProcessRunner,
    config: Config = {},
  ) {
    const input = ConfigSchema.parse(config);
    this.config = {
      command: input.command ?? "dvc",
      graceMs: input.graceMs ?? DEFAULT_GRACE_MS,
      maxManifestBytes: input.maxManifestBytes ?? DEFAULT_MAX_MANIFEST_BYTES,
      maxOutputBytes: input.maxOutputBytes ?? DEFAULT_MAX_OUTPUT_BYTES,
    };
    this.git = new GitWorktreeRuntime(subprocess, {
      command: input.gitCommand ?? "git",
      graceMs: this.config.graceMs,
      maxOutputBytes: this.config.maxOutputBytes,
    });
  }

  async inspect(cwd: string, signal?: AbortSignal): Promise<DvcInspection> {
    this.ensureOpen();
    signal?.throwIfAborted();
    const project = await this.resolveProject(cwd, signal);
    return this.inspectProject(project, signal);
  }

  async pull(
    cwd: string,
    request: DvcRequest = {},
    signal?: AbortSignal,
  ): Promise<DvcCommandResult> {
    this.ensureOpen();
    signal?.throwIfAborted();
    const targets = validateTargets(request.targets);
    const remote = validateRemote(request.remote);
    const initial = await this.inspectGit(cwd, signal);
    return this.serialize(initial.root, async () => {
      signal?.throwIfAborted();
      const project = await this.resolveProject(cwd, signal);
      const capture = await this.run(
        project.executable,
        project.root,
        ["pull", ...(remote === undefined ? [] : ["-r", remote]), ...targets],
        signal,
      );
      return resultFrom(capture, signal, [project.git.root, project.root]);
    });
  }

  async reproduce(
    cwd: string,
    request: DvcReproduceRequest = {},
    signal?: AbortSignal,
  ): Promise<DvcReproductionHandle> {
    this.ensureOpen();
    signal?.throwIfAborted();
    const targets = validateTargets(request.targets);
    const remote = validateRemote(request.remote);
    const initial = await this.inspectGit(cwd, signal);
    return this.serialize(initial.root, async () => {
      signal?.throwIfAborted();
      const project = await this.resolveProject(cwd, signal);
      if (!project.git.public.clean) {
        throw new DvcError(
          "DVC reproduction requires a clean Git working tree",
          "DVC_WORKSPACE_DIRTY",
        );
      }
      const source = await this.inspectProject(project, signal);
      let worktree: GitWorktreeHandle;
      try {
        worktree = await this.git.createDetachedWorktree(project.git, signal);
      } catch (error) {
        if (signal?.aborted) signal.throwIfAborted();
        throw mapGitError(error);
      }

      try {
        const targetRoot = this.worktreeDvcRoot(worktree.path, source.root);
        await this.preparePrivateDvc(project, targetRoot, signal);
        let capture: CommandCapture;
        if (request.pull === true) {
          capture = await this.run(
            project.executable,
            targetRoot,
            ["pull", ...(remote === undefined ? [] : ["-r", remote]), ...targets],
            signal,
          );
          if (successful(capture) && !signal?.aborted) {
            capture = await this.run(project.executable, targetRoot, ["repro", ...targets], signal);
          }
        } else {
          capture = await this.run(project.executable, targetRoot, ["repro", ...targets], signal);
        }
        const result = resultFrom(capture, signal, [
          project.git.root,
          project.root,
          worktree.path,
          targetRoot,
        ]);
        let after: DvcInspection | null = null;
        if (!signal?.aborted) {
          try {
            after = await this.inspect(targetRoot, signal);
          } catch {
            after = null;
          }
        }
        const state: ReproductionState = { disposed: false, worktree };
        this.reproductions.add(state);
        return {
          after,
          path: targetRoot,
          result,
          source,
          dispose: () => this.disposeReproduction(state),
        };
      } catch (error) {
        await worktree.dispose();
        if (signal?.aborted) signal.throwIfAborted();
        throw error;
      }
    });
  }

  private async inspectGit(cwd: string, signal?: AbortSignal): Promise<GitWorkspaceSnapshot> {
    try {
      return await this.git.inspect(cwd, signal);
    } catch (error) {
      if (signal?.aborted) signal.throwIfAborted();
      throw mapGitError(error);
    }
  }

  private async resolveProject(cwd: string, signal?: AbortSignal): Promise<DvcProject> {
    const git = await this.inspectGit(cwd, signal);
    const executable = await this.resolveExecutable(signal);
    const rootCapture = await this.run(executable, cwd, ["root"], signal);
    if (!successful(rootCapture)) {
      throw new DvcError("Workspace is not an existing DVC project", "NOT_A_DVC_REPOSITORY");
    }
    this.requireComplete(rootCapture, signal);
    const root = canonicalDirectory(resolve(cwd, line(rootCapture.stdout.text)));
    if (!isContained(git.root, root)) {
      throw new DvcError("DVC root is outside the Git repository", "NOT_A_DVC_REPOSITORY");
    }
    return { executable, git, root };
  }

  private async inspectProject(project: DvcProject, signal?: AbortSignal): Promise<DvcInspection> {
    const [version, data, pipeline] = await Promise.all([
      this.required(project.executable, project.root, ["--version"], signal),
      this.required(
        project.executable,
        project.root,
        ["data", "status", "--json", "--no-remote-refresh"],
        signal,
      ),
      this.required(project.executable, project.root, ["status", "--json"], signal),
    ]);
    return {
      data: summarizeStatus(data.stdout.text),
      dvcLockDigest: manifestDigest(project.root, "dvc.lock", this.config.maxManifestBytes),
      dvcYamlDigest: manifestDigest(project.root, "dvc.yaml", this.config.maxManifestBytes),
      git: project.git.public,
      root: gitRelative(project.git.root, project.root),
      pipeline: summarizeStatus(pipeline.stdout.text),
      version: line(version.stdout.text),
    };
  }

  private worktreeDvcRoot(worktreeRoot: string, relativeRoot: string): string {
    const target =
      relativeRoot === "." ? worktreeRoot : join(worktreeRoot, ...relativeRoot.split("/"));
    const canonical = canonicalDirectory(target);
    if (!isContained(worktreeRoot, canonical)) {
      throw new DvcError("Disposable DVC root escaped its worktree", "DVC_WORKTREE_FAILED");
    }
    return canonical;
  }

  private async preparePrivateDvc(
    project: DvcProject,
    targetRoot: string,
    signal?: AbortSignal,
  ): Promise<void> {
    const sourceConfig = join(project.root, ".dvc", "config.local");
    const targetDvc = canonicalDirectory(join(targetRoot, ".dvc"));
    if (!isContained(targetRoot, targetDvc)) {
      throw new DvcError("Disposable DVC metadata escaped its worktree", "DVC_WORKTREE_FAILED");
    }
    const targetConfig = join(targetDvc, "config.local");
    if (existsSync(sourceConfig)) {
      const metadata = lstatSync(sourceConfig);
      if (
        !metadata.isFile() ||
        metadata.isSymbolicLink() ||
        metadata.size > MAX_CONFIG_LOCAL_BYTES ||
        existsSync(targetConfig)
      ) {
        throw new DvcError(
          "DVC local configuration is not a bounded regular file",
          "DVC_WORKTREE_FAILED",
        );
      }
      copyFileSync(sourceConfig, targetConfig);
      chmodSync(targetConfig, 0o600);
    }

    const cache = await this.required(project.executable, project.root, ["cache", "dir"], signal);
    const cacheValue = line(cache.stdout.text);
    if (cacheValue.length === 0 || cacheValue.includes("\0") || /[\r\n]/u.test(cacheValue)) {
      throw new DvcError("DVC cache directory response is invalid", "DVC_WORKTREE_FAILED");
    }
    const cachePath = isAbsolute(cacheValue) ? cacheValue : resolve(project.root, cacheValue);
    await this.required(
      project.executable,
      targetRoot,
      ["cache", "dir", "--local", cachePath],
      signal,
    );
    if (!existsSync(targetConfig)) {
      throw new DvcError("DVC did not create private local configuration", "DVC_WORKTREE_FAILED");
    }
    chmodSync(targetConfig, 0o600);
  }

  private async resolveExecutable(signal?: AbortSignal): Promise<string> {
    try {
      return await this.subprocess.resolveExecutable(this.config.command, {}, signal);
    } catch (error) {
      if (signal?.aborted) signal.throwIfAborted();
      throw new DvcError("Configured DVC executable is unavailable", "DVC_UNAVAILABLE", {
        cause: error,
      });
    }
  }

  private async required(
    executable: string,
    cwd: string,
    args: readonly string[],
    signal?: AbortSignal,
  ): Promise<CommandCapture> {
    const capture = await this.run(executable, cwd, args, signal);
    signal?.throwIfAborted();
    this.requireComplete(capture, signal);
    if (!successful(capture)) throw new DvcError("DVC command failed", "DVC_COMMAND_FAILED");
    return capture;
  }

  private requireComplete(capture: CommandCapture, signal?: AbortSignal): void {
    signal?.throwIfAborted();
    if (capture.stdout.truncated || capture.stderr.truncated) {
      throw new DvcError("DVC command output exceeded its limit", "DVC_OUTPUT_TOO_LARGE");
    }
  }

  private async run(
    executable: string,
    cwd: string,
    args: readonly string[],
    signal?: AbortSignal,
  ): Promise<CommandCapture> {
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
        env: { DVC_NO_ANALYTICS: "1", GIT_TERMINAL_PROMPT: "0", LC_ALL: "C" },
      });
    } catch (error) {
      if (signal?.aborted) signal.throwIfAborted();
      throw new DvcError("DVC process could not be started", "DVC_UNAVAILABLE", {
        cause: error,
      });
    }
    this.active.add(handle);
    try {
      const outcome = await handle.done;
      const stdout = handle.collected.stdout?.readFrom(0);
      const stderr = handle.collected.stderr?.readFrom(0);
      if (!stdout || !stderr) {
        throw new DvcError("DVC output collection is unavailable", "DVC_COMMAND_FAILED");
      }
      return {
        outcome,
        stdout: { text: stdout.text, truncated: stdout.lossy },
        stderr: { text: stderr.text, truncated: stderr.lossy },
      };
    } catch (error) {
      if (signal?.aborted && !(error instanceof DvcError)) signal.throwIfAborted();
      if (error instanceof DvcError) throw error;
      throw new DvcError("DVC process failed to settle", "DVC_COMMAND_FAILED", { cause: error });
    } finally {
      this.active.delete(handle);
    }
  }

  private async serialize<T>(key: string, task: () => Promise<T>): Promise<T> {
    const previous = this.mutations.get(key) ?? Promise.resolve();
    const result = previous.then(task);
    const settled = result.then(
      () => undefined,
      () => undefined,
    );
    this.mutations.set(key, settled);
    try {
      return await result;
    } finally {
      if (this.mutations.get(key) === settled) this.mutations.delete(key);
    }
  }

  private async disposeReproduction(state: ReproductionState): Promise<void> {
    if (state.disposed) return;
    state.disposed = true;
    this.reproductions.delete(state);
    await state.worktree.dispose();
  }

  async close(): Promise<void> {
    if (!this.open) return;
    const active = [...this.active];
    for (const handle of active) handle.terminate();
    await Promise.all(active.map((handle) => handle.waitForExit()));
    for (const reproduction of [...this.reproductions]) {
      await this.disposeReproduction(reproduction);
    }
    await this.git.close();
    this.open = false;
  }

  private ensureOpen(): void {
    if (!this.open) throw new DvcError("DVC service is closed", "DVC_CLOSED");
  }
}

export default DvcService;
