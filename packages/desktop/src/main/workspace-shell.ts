import { type ChildProcess, spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { access, mkdtemp, realpath, rm, stat } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { type LocalMcpTool, currentRequestSignal } from "@swarmx/core";
import type { IDisposable, IPty } from "node-pty";
import * as pty from "node-pty";
import { ensurePtySpawnHelperExecutable } from "./pty-runtime.js";

const MAX_COMMAND_BYTES = 64 * 1024;
const MAX_STDIN_BYTES = 64 * 1024;
const PROCESS_KILL_GRACE_MS = 500;
const DEFAULT_EXEC_YIELD_MS = 10_000;
const DEFAULT_WRITE_YIELD_MS = 250;
const DEFAULT_POLL_YIELD_MS = 5_000;
const DEFAULT_MAX_OUTPUT_TOKENS = 10_000;
const MAX_OUTPUT_TOKENS = 50_000;
const COMPLETED_SESSION_RETENTION_MS = 5 * 60_000;
const PTY_COLUMNS = 80;
const PTY_ROWS = 24;

export const WORKSPACE_SHELL_DEFAULTS = {
  timeoutMs: 120_000,
  maxTimeoutMs: 3_600_000,
  backgroundTimeoutMs: 3_600_000,
  maxOutputBytes: 64 * 1024,
} as const;

const WORKSPACE_SHELL_HARD_LIMITS = {
  timeoutMs: 600_000,
  maxTimeoutMs: 24 * 3_600_000,
  backgroundTimeoutMs: 24 * 3_600_000,
  maxOutputBytes: 1024 * 1024,
} as const;

export interface WorkspaceShellOptions {
  timeoutMs?: number;
  maxTimeoutMs?: number;
  backgroundTimeoutMs?: number;
  maxOutputBytes?: number;
  sandboxExecutable?: string;
  shellExecutable?: string;
  platform?: NodeJS.Platform;
}

export interface WorkspaceShellRunOptions {
  timeoutMs?: number;
  workdir?: string;
  tty?: boolean;
  /** Internal observer used by session-scoped tools such as Claude Monitor. */
  onStdout?: (chunk: string) => void;
  /** Request-scoped observer for ordered terminal presentation updates. */
  onOutput?: (output: WorkspaceShellOutput) => void;
  /** Internal lifecycle observer; callback failures never affect process cleanup. */
  onExit?: (snapshot: WorkspaceShellSessionSnapshot) => void;
  /** Keep the process alive until explicit stop or WorkspaceShell.close(). */
  sessionLifetime?: boolean;
}

export type WorkspaceShellSessionStatus =
  | "running"
  | "completed"
  | "failed"
  | "stopped"
  | "timed_out";

export interface WorkspaceShellResult {
  command: string;
  cwd: string;
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  stdout: string;
  stderr: string;
  durationMs: number;
  timedOut: boolean;
  truncated: boolean;
}

export interface WorkspaceShellSessionSnapshot extends WorkspaceShellResult {
  sessionId: number;
  status: WorkspaceShellSessionStatus;
}

export interface WorkspaceShellExecOptions extends WorkspaceShellRunOptions {
  yieldTimeMs?: number;
  maxOutputTokens?: number;
}

export interface WorkspaceShellInteractionOptions {
  yieldTimeMs?: number;
  maxOutputTokens?: number;
  onOutput?: (output: WorkspaceShellOutput) => void;
}

export interface WorkspaceShellOutput {
  content: string;
  stream: "stdout" | "stderr";
}

export interface WorkspaceShellExecResult {
  chunkId: string;
  wallTimeSeconds: number;
  exitCode?: number;
  sessionId?: number;
  originalTokenCount?: number;
  output: string;
  status: WorkspaceShellSessionStatus;
}

interface ResolvedWorkspaceShellOptions {
  timeoutMs: number;
  maxTimeoutMs: number;
  backgroundTimeoutMs: number;
  maxOutputBytes: number;
  sandboxExecutable: string;
  shellExecutable: string;
  platform: NodeJS.Platform;
}

interface ManagedShellSession {
  id: number;
  command: string;
  cwd: string;
  process: ShellProcess;
  temporaryDirectory: string;
  startedAt: number;
  endedAt?: number;
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  status: WorkspaceShellSessionStatus;
  stdout: BoundedOutput;
  stderr: BoundedOutput;
  combined: RollingOutput;
  combinedCursor: number;
  stopReason?: "cancelled" | "stopped" | "timeout" | "closed";
  cancellationReason?: unknown;
  spawnError?: Error;
  timeout?: ReturnType<typeof setTimeout>;
  forceKillTimer?: ReturnType<typeof setTimeout>;
  retentionTimer?: ReturnType<typeof setTimeout>;
  requestSignal?: AbortSignal;
  onAbort?: () => void;
  onStdout?: (chunk: string) => void;
  onOutput?: (output: WorkspaceShellOutput) => void;
  onExit?: (snapshot: WorkspaceShellSessionSnapshot) => void;
  done: Promise<void>;
  resolveDone: () => void;
  finalized: boolean;
}

interface PipeShellProcess {
  kind: "pipe";
  child: ChildProcess;
}

interface PtyShellProcess {
  kind: "pty";
  pty: IPty;
  dataSubscription?: IDisposable;
  exitSubscription?: IDisposable;
}

type ShellProcess = PipeShellProcess | PtyShellProcess;

/** Executes and manages bounded commands in a fail-closed Project write sandbox. */
export class WorkspaceShell {
  #root: string;
  #additionalWritablePaths: string[] = [];
  readonly #options: ResolvedWorkspaceShellOptions;
  readonly #sessions = new Map<number, ManagedShellSession>();
  #nextSessionId = 1;
  #closed = false;

  constructor(root: string, options: WorkspaceShellOptions = {}) {
    if (typeof root !== "string" || !root.trim() || root.includes("\0")) {
      throw new Error("A valid workspace root is required.");
    }
    this.#root = path.resolve(root);
    this.#options = resolveOptions(options);
  }

  get root(): string {
    return this.#root;
  }

  hasRunningSessions(): boolean {
    return [...this.#sessions.values()].some((session) => session.status === "running");
  }

  rebindRoot(root: string, additionalWritablePaths: readonly string[] = []): void {
    validateWorkspaceRoot(root);
    for (const writablePath of additionalWritablePaths) validateWorkspaceRoot(writablePath);
    this.#root = path.resolve(root);
    this.#additionalWritablePaths = additionalWritablePaths.map((writablePath) =>
      path.resolve(writablePath),
    );
  }

  async run(
    command: string,
    options: WorkspaceShellRunOptions = {},
  ): Promise<WorkspaceShellResult> {
    const session = await this.#start(command, options, false);
    await session.done;
    if (session.cancellationReason !== undefined) {
      throw session.cancellationReason instanceof Error
        ? session.cancellationReason
        : new Error("Request was cancelled.");
    }
    if (session.spawnError) throw session.spawnError;
    return shellResult(session);
  }

  async startBackground(
    command: string,
    options: WorkspaceShellRunOptions = {},
  ): Promise<WorkspaceShellSessionSnapshot> {
    return sessionSnapshot(await this.#start(command, options, true));
  }

  async runWithBackgroundFallback(
    command: string,
    foregroundTimeoutMs: number = WORKSPACE_SHELL_DEFAULTS.timeoutMs,
    options: WorkspaceShellRunOptions = {},
  ): Promise<WorkspaceShellSessionSnapshot> {
    const timeoutMs = boundedInteger(
      foregroundTimeoutMs,
      WORKSPACE_SHELL_DEFAULTS.timeoutMs,
      1,
      WORKSPACE_SHELL_HARD_LIMITS.timeoutMs,
      "timeout",
    );
    const session = await this.#start(command, options, true);
    await waitForSession(session, timeoutMs);
    if (session.cancellationReason !== undefined) {
      throw session.cancellationReason instanceof Error
        ? session.cancellationReason
        : new Error("Request was cancelled.");
    }
    if (session.spawnError) throw session.spawnError;
    return sessionSnapshot(session);
  }

  async exec(
    command: string,
    options: WorkspaceShellExecOptions = {},
  ): Promise<WorkspaceShellExecResult> {
    const startedAt = Date.now();
    const session = await this.#start(command, options, true);
    const yieldTimeMs = boundedInteger(
      options.yieldTimeMs,
      DEFAULT_EXEC_YIELD_MS,
      250,
      30_000,
      "yieldTime_ms",
    );
    await waitForSession(session, yieldTimeMs);
    return this.#execResult(session, startedAt, options.maxOutputTokens);
  }

  async writeStdin(
    sessionId: number,
    chars = "",
    options: WorkspaceShellInteractionOptions = {},
  ): Promise<WorkspaceShellExecResult> {
    const startedAt = Date.now();
    const session = this.#requiredSession(sessionId);
    validateStdin(chars);
    if (chars && session.status !== "running") {
      throw new Error(`Shell session ${sessionId} is not running.`);
    }
    if (chars) await writeSessionInput(session, chars);
    const yieldTimeMs = chars
      ? boundedInteger(options.yieldTimeMs, DEFAULT_WRITE_YIELD_MS, 0, 30_000, "yield_time_ms")
      : boundedInteger(options.yieldTimeMs, DEFAULT_POLL_YIELD_MS, 0, 300_000, "yield_time_ms");
    const previousOutputObserver = session.onOutput;
    if (options.onOutput) session.onOutput = options.onOutput;
    try {
      await waitForSession(session, yieldTimeMs);
      return this.#execResult(session, startedAt, options.maxOutputTokens);
    } finally {
      if (options.onOutput && session.onOutput === options.onOutput) {
        session.onOutput = previousOutputObserver;
      }
    }
  }

  async taskOutput(
    sessionId: number,
    options: { block: boolean; timeoutMs: number },
  ): Promise<WorkspaceShellSessionSnapshot> {
    const session = this.#requiredSession(sessionId);
    if (options.block) {
      const timeoutMs = boundedInteger(options.timeoutMs, 30_000, 1, 600_000, "timeout");
      await waitForSession(session, timeoutMs);
    }
    return sessionSnapshot(session);
  }

  async stop(sessionId: number): Promise<WorkspaceShellSessionSnapshot> {
    const session = this.#requiredSession(sessionId);
    if (session.status === "running") {
      this.#terminate(session, "stopped");
      await session.done;
    }
    return sessionSnapshot(session);
  }

  async stopSessionsWithin(root: string): Promise<number> {
    validateWorkspaceRoot(root);
    const canonicalRoot = await realWorkspaceRoot(root);
    const sessions = [...this.#sessions.values()].filter(
      (session) => session.status === "running" && isContainedPath(canonicalRoot, session.cwd),
    );
    for (const session of sessions) this.#terminate(session, "stopped");
    await Promise.allSettled(sessions.map((session) => session.done));
    return sessions.length;
  }

  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    const sessions = [...this.#sessions.values()];
    for (const session of sessions) {
      if (session.retentionTimer) clearTimeout(session.retentionTimer);
      if (session.status === "running") this.#terminate(session, "closed");
    }
    await Promise.allSettled(sessions.map((session) => session.done));
    await Promise.allSettled(
      sessions.map((session) => rm(session.temporaryDirectory, { recursive: true, force: true })),
    );
    this.#sessions.clear();
  }

  async #start(
    command: string,
    options: WorkspaceShellRunOptions,
    longRunning: boolean,
  ): Promise<ManagedShellSession> {
    if (this.#closed) throw new Error("Workspace shell is closed.");
    validateCommand(command);
    if (this.#options.platform !== "darwin") {
      throw new Error(
        `Sandboxed workspace shell is unavailable on ${this.#options.platform}; refusing unrestricted execution.`,
      );
    }

    const root = await realWorkspaceRoot(this.root);
    const additionalWritablePaths = await Promise.all(
      this.#additionalWritablePaths.map((writablePath) => realWorkspaceRoot(writablePath)),
    );
    const workdir = await resolveWorkspaceWorkdir(root, options.workdir);
    const timeoutMs = requestedTimeout(options.timeoutMs, this.#options, longRunning);
    const requestSignal = currentRequestSignal();
    throwIfAborted(requestSignal);
    await ensureExecutable(this.#options.sandboxExecutable);

    const temporaryRoot = await realpath(os.tmpdir());
    const temporaryDirectory = await mkdtemp(path.join(temporaryRoot, "swarmx-shell-"));
    const profile = seatbeltProfile(
      root,
      temporaryDirectory,
      options.tty === true,
      additionalWritablePaths,
    );
    let resolveDone = (): void => {};
    const done = new Promise<void>((resolve) => {
      resolveDone = resolve;
    });
    let shellProcess: ShellProcess;
    try {
      shellProcess = spawnShellProcess({
        sandboxExecutable: this.#options.sandboxExecutable,
        shellExecutable: this.#options.shellExecutable,
        command,
        profile,
        cwd: workdir,
        env: shellEnvironment(root, temporaryDirectory, options.tty === true),
        tty: options.tty === true,
        platform: this.#options.platform,
      });
    } catch (cause) {
      await rm(temporaryDirectory, { recursive: true, force: true });
      throw cause;
    }
    const session: ManagedShellSession = {
      id: this.#nextSessionId++,
      command,
      cwd: workdir,
      process: shellProcess,
      temporaryDirectory,
      startedAt: Date.now(),
      exitCode: null,
      signal: null,
      status: "running",
      stdout: new BoundedOutput(this.#options.maxOutputBytes),
      stderr: new BoundedOutput(this.#options.maxOutputBytes),
      combined: new RollingOutput(this.#options.maxOutputBytes),
      combinedCursor: 0,
      timeout: undefined,
      requestSignal,
      onStdout: options.onStdout,
      onOutput: options.onOutput,
      onExit: options.onExit,
      done,
      resolveDone,
      finalized: false,
    };
    this.#sessions.set(session.id, session);

    if (shellProcess.kind === "pipe") {
      shellProcess.child.stdout?.on("data", (chunk: Buffer | string) => {
        session.stdout.append(chunk);
        session.combined.append(chunk);
        notifyStdout(session, chunk);
      });
      shellProcess.child.stderr?.on("data", (chunk: Buffer | string) => {
        session.stderr.append(chunk);
        session.combined.append(chunk);
        notifyOutput(session, "stderr", chunk);
      });
      shellProcess.child.once("error", (cause) => {
        session.spawnError = isFileNotFoundError(cause)
          ? new Error(
              "The macOS sandbox executable is unavailable; refusing unrestricted shell execution.",
            )
          : cause;
        void this.#finalize(session, null, null);
      });
      shellProcess.child.once("close", (exitCode, exitSignal) => {
        void this.#finalize(session, exitCode, exitSignal);
      });
    } else {
      shellProcess.dataSubscription = shellProcess.pty.onData((chunk) => {
        session.stdout.append(chunk);
        session.combined.append(chunk);
        notifyStdout(session, chunk);
      });
      shellProcess.exitSubscription = shellProcess.pty.onExit(({ exitCode, signal }) => {
        void this.#finalize(session, exitCode, signalName(signal));
      });
    }

    session.onAbort = () => {
      session.cancellationReason = requestSignal?.reason ?? new Error("Request was cancelled.");
      this.#terminate(session, "cancelled");
    };
    requestSignal?.addEventListener("abort", session.onAbort, { once: true });
    if (requestSignal?.aborted) session.onAbort();

    if (!options.sessionLifetime) {
      session.timeout = setTimeout(() => this.#terminate(session, "timeout"), timeoutMs);
      session.timeout.unref?.();
    }

    return session;
  }

  #requiredSession(sessionId: number): ManagedShellSession {
    if (!Number.isSafeInteger(sessionId) || sessionId <= 0) {
      throw new Error("session_id must be a positive integer.");
    }
    const session = this.#sessions.get(sessionId);
    if (!session) throw new Error(`Shell session ${sessionId} was not found or has expired.`);
    return session;
  }

  #terminate(
    session: ManagedShellSession,
    reason: "cancelled" | "stopped" | "timeout" | "closed",
  ): void {
    if (session.status !== "running") return;
    session.stopReason ??= reason;
    terminateProcessGroup(session.process, "SIGTERM");
    session.forceKillTimer ??= setTimeout(
      () => terminateProcessGroup(session.process, "SIGKILL"),
      PROCESS_KILL_GRACE_MS,
    );
    session.forceKillTimer.unref?.();
  }

  async #finalize(
    session: ManagedShellSession,
    exitCode: number | null,
    exitSignal: NodeJS.Signals | null,
  ): Promise<void> {
    if (session.finalized) return;
    session.finalized = true;
    if (session.timeout) clearTimeout(session.timeout);
    if (session.forceKillTimer) clearTimeout(session.forceKillTimer);
    if (session.onAbort) session.requestSignal?.removeEventListener("abort", session.onAbort);
    session.exitCode = exitCode;
    session.signal = exitSignal;
    session.endedAt = Date.now();
    session.status = session.spawnError
      ? "failed"
      : session.stopReason === "timeout"
        ? "timed_out"
        : session.stopReason
          ? "stopped"
          : exitCode === 0
            ? "completed"
            : "failed";
    closeSessionInput(session);
    if (session.process.kind === "pty") {
      session.process.dataSubscription?.dispose();
      session.process.exitSubscription?.dispose();
    }
    await rm(session.temporaryDirectory, { recursive: true, force: true });
    notifyExit(session);
    session.resolveDone();
    if (this.#closed) {
      this.#sessions.delete(session.id);
      return;
    }
    session.retentionTimer = setTimeout(() => {
      this.#sessions.delete(session.id);
    }, COMPLETED_SESSION_RETENTION_MS);
    session.retentionTimer.unref?.();
  }

  #execResult(
    session: ManagedShellSession,
    callStartedAt: number,
    maxOutputTokens: number | undefined,
  ): WorkspaceShellExecResult {
    const tokenBudget = boundedInteger(
      maxOutputTokens,
      DEFAULT_MAX_OUTPUT_TOKENS,
      1,
      MAX_OUTPUT_TOKENS,
      "max_output_tokens",
    );
    const read = session.combined.readFrom(session.combinedCursor);
    session.combinedCursor = read.endOffset;
    const truncated = truncateForTokenBudget(read.value, tokenBudget);
    const originalBytes = Math.max(0, read.endOffset - read.requestedOffset);
    return {
      chunkId: randomUUID(),
      wallTimeSeconds: Math.max(0, Date.now() - callStartedAt) / 1000,
      ...(session.status === "running" ? { sessionId: session.id } : {}),
      ...(session.status !== "running" && session.exitCode !== null
        ? { exitCode: session.exitCode }
        : {}),
      ...(read.missed || truncated.truncated
        ? { originalTokenCount: approximateTokenCount(originalBytes) }
        : {}),
      output: truncated.value,
      status: session.status,
    };
  }
}

export function workspaceShellAgentTool(shell: WorkspaceShell): LocalMcpTool {
  return {
    name: "workspace_shell",
    description:
      "Run one bounded shell command from the active Project root for builds, tests, search, and other repository work. The macOS sandbox denies network access and writes outside the Project.",
    inputSchema: {
      type: "object",
      properties: {
        command: { type: "string", minLength: 1, description: "Shell command to execute." },
        timeoutMs: {
          type: "integer",
          minimum: 1,
          maximum: WORKSPACE_SHELL_DEFAULTS.maxTimeoutMs,
          description: "Optional execution timeout in milliseconds.",
        },
      },
      required: ["command"],
      additionalProperties: false,
    },
    dispose: () => shell.close(),
    call: async (input) =>
      shell.run(requiredCommand(input.command), {
        ...(input.timeoutMs === undefined
          ? {}
          : { timeoutMs: requiredPositiveInteger(input.timeoutMs, "timeoutMs") }),
      }),
  };
}

class RollingOutput {
  readonly #maximumBytes: number;
  #buffer = Buffer.alloc(0);
  #startOffset = 0;
  #endOffset = 0;
  truncated = false;

  constructor(maximumBytes: number) {
    this.#maximumBytes = maximumBytes;
  }

  append(chunk: Buffer | string): void {
    const buffer = typeof chunk === "string" ? Buffer.from(chunk) : chunk;
    this.#endOffset += buffer.length;
    this.#buffer = Buffer.concat([this.#buffer, buffer]);
    if (this.#buffer.length <= this.#maximumBytes) return;
    const dropped = this.#buffer.length - this.#maximumBytes;
    this.#buffer = this.#buffer.subarray(dropped);
    this.#startOffset += dropped;
    this.truncated = true;
  }

  value(): string {
    return this.#buffer.toString("utf8");
  }

  readFrom(requestedOffset: number): {
    value: string;
    requestedOffset: number;
    endOffset: number;
    missed: boolean;
  } {
    const actualOffset = Math.max(requestedOffset, this.#startOffset);
    return {
      value: this.#buffer.subarray(actualOffset - this.#startOffset).toString("utf8"),
      requestedOffset,
      endOffset: this.#endOffset,
      missed: requestedOffset < this.#startOffset,
    };
  }
}

class BoundedOutput {
  readonly #maximumBytes: number;
  readonly #chunks: Buffer[] = [];
  #bytes = 0;
  truncated = false;

  constructor(maximumBytes: number) {
    this.#maximumBytes = maximumBytes;
  }

  append(chunk: Buffer | string): void {
    const buffer = typeof chunk === "string" ? Buffer.from(chunk) : chunk;
    const remaining = this.#maximumBytes - this.#bytes;
    if (remaining > 0) {
      const captured = buffer.subarray(0, remaining);
      this.#chunks.push(captured);
      this.#bytes += captured.length;
    }
    if (buffer.length > Math.max(0, remaining)) this.truncated = true;
  }

  value(): string {
    return Buffer.concat(this.#chunks).toString("utf8");
  }
}

function shellResult(session: ManagedShellSession): WorkspaceShellResult {
  return {
    command: session.command,
    cwd: session.cwd,
    exitCode: session.exitCode,
    signal: session.signal,
    stdout: session.stdout.value(),
    stderr: session.stderr.value(),
    durationMs: Math.max(0, (session.endedAt ?? Date.now()) - session.startedAt),
    timedOut: session.status === "timed_out",
    truncated: session.stdout.truncated || session.stderr.truncated || session.combined.truncated,
  };
}

function sessionSnapshot(session: ManagedShellSession): WorkspaceShellSessionSnapshot {
  return { sessionId: session.id, status: session.status, ...shellResult(session) };
}

function notifyStdout(session: ManagedShellSession, chunk: Buffer | string): void {
  notifyOutput(session, "stdout", chunk);
  try {
    session.onStdout?.(typeof chunk === "string" ? chunk : chunk.toString("utf8"));
  } catch {
    // Observers must not interfere with process output collection.
  }
}

function notifyOutput(
  session: ManagedShellSession,
  stream: WorkspaceShellOutput["stream"],
  chunk: Buffer | string,
): void {
  try {
    session.onOutput?.({
      content: typeof chunk === "string" ? chunk : chunk.toString("utf8"),
      stream,
    });
  } catch {
    // Presentation observers must not interfere with process output collection.
  }
}

function notifyExit(session: ManagedShellSession): void {
  try {
    session.onExit?.(sessionSnapshot(session));
  } catch {
    // Observers must not interfere with process cleanup.
  }
}

async function waitForSession(session: ManagedShellSession, milliseconds: number): Promise<void> {
  if (session.status !== "running" || milliseconds === 0) return;
  let timer: ReturnType<typeof setTimeout> | undefined;
  await Promise.race([
    session.done,
    new Promise<void>((resolve) => {
      timer = setTimeout(resolve, milliseconds);
      timer.unref?.();
    }),
  ]);
  if (timer) clearTimeout(timer);
}

function resolveOptions(options: WorkspaceShellOptions): ResolvedWorkspaceShellOptions {
  const maxTimeoutMs = configuredLimit(
    options.maxTimeoutMs,
    WORKSPACE_SHELL_DEFAULTS.maxTimeoutMs,
    WORKSPACE_SHELL_HARD_LIMITS.maxTimeoutMs,
    "maxTimeoutMs",
  );
  return {
    timeoutMs: Math.min(
      maxTimeoutMs,
      configuredLimit(
        options.timeoutMs,
        WORKSPACE_SHELL_DEFAULTS.timeoutMs,
        WORKSPACE_SHELL_HARD_LIMITS.timeoutMs,
        "timeoutMs",
      ),
    ),
    maxTimeoutMs,
    backgroundTimeoutMs: Math.min(
      maxTimeoutMs,
      configuredLimit(
        options.backgroundTimeoutMs,
        WORKSPACE_SHELL_DEFAULTS.backgroundTimeoutMs,
        WORKSPACE_SHELL_HARD_LIMITS.backgroundTimeoutMs,
        "backgroundTimeoutMs",
      ),
    ),
    maxOutputBytes: configuredLimit(
      options.maxOutputBytes,
      WORKSPACE_SHELL_DEFAULTS.maxOutputBytes,
      WORKSPACE_SHELL_HARD_LIMITS.maxOutputBytes,
      "maxOutputBytes",
    ),
    sandboxExecutable: options.sandboxExecutable ?? "/usr/bin/sandbox-exec",
    shellExecutable: options.shellExecutable ?? "/bin/zsh",
    platform: options.platform ?? process.platform,
  };
}

function configuredLimit(
  value: number | undefined,
  fallback: number,
  maximum: number,
  name: string,
): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value <= 0 || value > maximum) {
    throw new Error(`${name} must be an integer between 1 and ${maximum}.`);
  }
  return value;
}

function requestedTimeout(
  value: number | undefined,
  options: ResolvedWorkspaceShellOptions,
  longRunning: boolean,
): number {
  if (value === undefined) return longRunning ? options.backgroundTimeoutMs : options.timeoutMs;
  if (!Number.isSafeInteger(value) || value <= 0 || value > options.maxTimeoutMs) {
    throw new Error(`timeoutMs must be an integer between 1 and ${options.maxTimeoutMs}.`);
  }
  return value;
}

function boundedInteger(
  value: number | undefined,
  fallback: number,
  minimum: number,
  maximum: number,
  name: string,
): number {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved < minimum || resolved > maximum) {
    throw new Error(`${name} must be an integer between ${minimum} and ${maximum}.`);
  }
  return resolved;
}

function validateCommand(command: string): void {
  if (typeof command !== "string" || !command.trim() || command.includes("\0")) {
    throw new Error("A non-empty shell command is required.");
  }
  if (Buffer.byteLength(command) > MAX_COMMAND_BYTES) {
    throw new Error(`Shell command exceeds the ${MAX_COMMAND_BYTES}-byte limit.`);
  }
}

function validateWorkspaceRoot(root: string): void {
  if (typeof root !== "string" || !root.trim() || root.includes("\0")) {
    throw new Error("A valid workspace root is required.");
  }
}

function validateStdin(chars: string): void {
  if (typeof chars !== "string" || chars.includes("\0")) {
    throw new Error("chars must be valid text.");
  }
  if (Buffer.byteLength(chars) > MAX_STDIN_BYTES) {
    throw new Error(`chars exceeds the ${MAX_STDIN_BYTES}-byte limit.`);
  }
}

function requiredCommand(value: unknown): string {
  if (typeof value !== "string") throw new Error("A non-empty shell command is required.");
  return value;
}

function requiredPositiveInteger(value: unknown, name: string): number {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
  return value as number;
}

async function realWorkspaceRoot(root: string): Promise<string> {
  const resolved = await realpath(root);
  const rootStat = await stat(resolved);
  if (!rootStat.isDirectory()) throw new Error("Workspace root is not a directory.");
  return resolved;
}

async function resolveWorkspaceWorkdir(root: string, value: string | undefined): Promise<string> {
  if (value === undefined || value === "") return root;
  if (typeof value !== "string" || value.includes("\0")) {
    throw new Error("workdir must be a valid Project directory.");
  }
  const lexical = path.isAbsolute(value) ? path.resolve(value) : path.resolve(root, value);
  const resolved = await realpath(lexical);
  if (!isContainedPath(root, resolved)) throw new Error("workdir escapes the Project root.");
  const directoryStat = await stat(resolved);
  if (!directoryStat.isDirectory()) throw new Error("workdir must be a Project directory.");
  return resolved;
}

function isContainedPath(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return (
    relative === "" ||
    (relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative))
  );
}

async function ensureExecutable(executable: string): Promise<void> {
  try {
    await access(executable, constants.X_OK);
  } catch {
    throw new Error(
      "The macOS sandbox executable is unavailable; refusing unrestricted shell execution.",
    );
  }
}

function seatbeltProfile(
  root: string,
  temporaryDirectory: string,
  tty: boolean,
  additionalWritablePaths: readonly string[],
): string {
  const writablePaths = [root, temporaryDirectory, ...additionalWritablePaths]
    .map((writablePath) => `(subpath ${sandboxString(writablePath)})`)
    .join(" ");
  return [
    "(version 1)",
    "(deny default)",
    "(allow process*)",
    "(allow file-read*)",
    `(allow file-write* ${writablePaths} (literal \"/dev/null\"))`,
    ...(tty ? ['(allow file-ioctl (literal "/dev/tty") (regex #"^/dev/ttys[0-9]+$"))'] : []),
    "(allow sysctl-read)",
    "(allow mach-lookup)",
  ].join(" ");
}

function sandboxString(value: string): string {
  return JSON.stringify(value);
}

function shellEnvironment(
  root: string,
  temporaryDirectory: string,
  tty: boolean,
): NodeJS.ProcessEnv {
  const environment: NodeJS.ProcessEnv = {
    PATH: process.env.PATH ?? "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    HOME: process.env.HOME ?? os.homedir(),
    SHELL: "/bin/zsh",
    PWD: root,
    OLDPWD: root,
    TMPDIR: `${temporaryDirectory}${path.sep}`,
    TMP: temporaryDirectory,
    TEMP: temporaryDirectory,
    TERM: tty ? "xterm-256color" : "dumb",
    ...(tty ? { COLORTERM: "truecolor", TERM_PROGRAM: "SwarmX" } : { NO_COLOR: "1" }),
    PAGER: "cat",
    GIT_PAGER: "cat",
    GIT_TERMINAL_PROMPT: "0",
    LC_ALL: "C.UTF-8",
  };
  for (const name of ["USER", "LOGNAME", "LANG", "LC_CTYPE"]) {
    const value = process.env[name];
    if (value) environment[name] = value;
  }
  return environment;
}

interface SpawnShellProcessOptions {
  sandboxExecutable: string;
  shellExecutable: string;
  command: string;
  profile: string;
  cwd: string;
  env: NodeJS.ProcessEnv;
  tty: boolean;
  platform: NodeJS.Platform;
}

function spawnShellProcess(options: SpawnShellProcessOptions): ShellProcess {
  const args = ["-p", options.profile, options.shellExecutable, "-lc", options.command];
  if (options.tty) {
    ensurePtySpawnHelperExecutable(options.platform);
    return {
      kind: "pty",
      pty: pty.spawn(options.sandboxExecutable, args, {
        name: "xterm-256color",
        cols: PTY_COLUMNS,
        rows: PTY_ROWS,
        cwd: options.cwd,
        env: options.env,
      }),
    };
  }
  return {
    kind: "pipe",
    child: spawn(options.sandboxExecutable, args, {
      cwd: options.cwd,
      env: options.env,
      detached: true,
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    }),
  };
}

async function writeSessionInput(session: ManagedShellSession, chars: string): Promise<void> {
  if (session.process.kind === "pty") {
    session.process.pty.write(chars);
    return;
  }
  const stdin = session.process.child.stdin;
  if (!stdin?.writable) {
    throw new Error(`Shell session ${session.id} does not accept stdin.`);
  }
  await new Promise<void>((resolve, reject) => {
    stdin.write(chars, (error) => (error ? reject(error) : resolve()));
  });
}

function closeSessionInput(session: ManagedShellSession): void {
  if (session.process.kind === "pipe") session.process.child.stdin?.end();
}

function terminateProcessGroup(shellProcess: ShellProcess, signal: NodeJS.Signals): void {
  const pid = shellProcess.kind === "pipe" ? shellProcess.child.pid : shellProcess.pty.pid;
  if (pid) {
    try {
      process.kill(-pid, signal);
      return;
    } catch {
      // Process group may already be gone; try direct child handle.
    }
  }
  try {
    if (shellProcess.kind === "pipe") shellProcess.child.kill(signal);
    else shellProcess.pty.kill(signal);
  } catch {
    // Process already exited.
  }
}

function signalName(signal: number | undefined): NodeJS.Signals | null {
  if (!signal) return null;
  const entry = Object.entries(os.constants.signals).find(([, value]) => value === signal);
  return (entry?.[0] as NodeJS.Signals | undefined) ?? null;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return;
  throw signal.reason instanceof Error ? signal.reason : new Error("Request was cancelled.");
}

function truncateForTokenBudget(
  value: string,
  maximumTokens: number,
): { value: string; truncated: boolean } {
  const maximumBytes = maximumTokens * 4;
  const encoded = Buffer.from(value);
  if (encoded.length <= maximumBytes) return { value, truncated: false };
  return { value: encoded.subarray(0, maximumBytes).toString("utf8"), truncated: true };
}

function approximateTokenCount(bytes: number): number {
  return Math.ceil(bytes / 4);
}

function isFileNotFoundError(cause: unknown): cause is Error {
  return cause instanceof Error && "code" in cause && cause.code === "ENOENT";
}
