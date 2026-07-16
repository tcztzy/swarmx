import { type ChildProcess, spawn } from "node:child_process";
import { constants } from "node:fs";
import { access, mkdtemp, realpath, rm, stat } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { type LocalMcpTool, currentRequestSignal } from "@swarmx/core";

const MAX_COMMAND_BYTES = 64 * 1024;
const PROCESS_KILL_GRACE_MS = 500;

export const WORKSPACE_SHELL_DEFAULTS = {
  timeoutMs: 120_000,
  maxTimeoutMs: 600_000,
  maxOutputBytes: 64 * 1024,
} as const;

const WORKSPACE_SHELL_HARD_LIMITS = {
  timeoutMs: 600_000,
  maxTimeoutMs: 3_600_000,
  maxOutputBytes: 1024 * 1024,
} as const;

export interface WorkspaceShellOptions {
  timeoutMs?: number;
  maxTimeoutMs?: number;
  maxOutputBytes?: number;
  sandboxExecutable?: string;
  shellExecutable?: string;
  platform?: NodeJS.Platform;
}

export interface WorkspaceShellRunOptions {
  timeoutMs?: number;
}

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

interface ResolvedWorkspaceShellOptions {
  timeoutMs: number;
  maxTimeoutMs: number;
  maxOutputBytes: number;
  sandboxExecutable: string;
  shellExecutable: string;
  platform: NodeJS.Platform;
}

/** Executes bounded commands in a fail-closed Project write sandbox. */
export class WorkspaceShell {
  readonly root: string;
  readonly #options: ResolvedWorkspaceShellOptions;

  constructor(root: string, options: WorkspaceShellOptions = {}) {
    if (typeof root !== "string" || !root.trim() || root.includes("\0")) {
      throw new Error("A valid workspace root is required.");
    }
    this.root = path.resolve(root);
    this.#options = resolveOptions(options);
  }

  async run(
    command: string,
    options: WorkspaceShellRunOptions = {},
  ): Promise<WorkspaceShellResult> {
    validateCommand(command);
    if (this.#options.platform !== "darwin") {
      throw new Error(
        `Sandboxed workspace shell is unavailable on ${this.#options.platform}; refusing unrestricted execution.`,
      );
    }

    const root = await realWorkspaceRoot(this.root);
    const timeoutMs = requestedTimeout(options.timeoutMs, this.#options);
    const signal = currentRequestSignal();
    throwIfAborted(signal);
    await ensureExecutable(this.#options.sandboxExecutable);

    const temporaryRoot = await realpath(os.tmpdir());
    const temporaryDirectory = await mkdtemp(path.join(temporaryRoot, "swarmx-shell-"));
    const startedAt = Date.now();
    try {
      const profile = seatbeltProfile(root, temporaryDirectory);
      return await runChildProcess({
        command,
        root,
        timeoutMs,
        maxOutputBytes: this.#options.maxOutputBytes,
        sandboxExecutable: this.#options.sandboxExecutable,
        shellExecutable: this.#options.shellExecutable,
        profile,
        temporaryDirectory,
        requestSignal: signal,
        startedAt,
      });
    } finally {
      await rm(temporaryDirectory, { recursive: true, force: true });
    }
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
    call: async (input) =>
      shell.run(requiredCommand(input.command), {
        ...(input.timeoutMs === undefined
          ? {}
          : { timeoutMs: requiredPositiveInteger(input.timeoutMs, "timeoutMs") }),
      }),
  };
}

interface ChildProcessRequest {
  command: string;
  root: string;
  timeoutMs: number;
  maxOutputBytes: number;
  sandboxExecutable: string;
  shellExecutable: string;
  profile: string;
  temporaryDirectory: string;
  requestSignal?: AbortSignal;
  startedAt: number;
}

function runChildProcess(request: ChildProcessRequest): Promise<WorkspaceShellResult> {
  return new Promise((resolve, reject) => {
    const stdout = new BoundedOutput(request.maxOutputBytes);
    const stderr = new BoundedOutput(request.maxOutputBytes);
    let timedOut = false;
    let cancellationReason: unknown;
    let forceKillTimer: ReturnType<typeof setTimeout> | undefined;
    let settled = false;

    const child = spawn(
      request.sandboxExecutable,
      ["-p", request.profile, request.shellExecutable, "-lc", request.command],
      {
        cwd: request.root,
        env: shellEnvironment(request.root, request.temporaryDirectory),
        detached: true,
        stdio: ["ignore", "pipe", "pipe"],
        windowsHide: true,
      },
    );

    child.stdout?.on("data", (chunk: Buffer | string) => stdout.append(chunk));
    child.stderr?.on("data", (chunk: Buffer | string) => stderr.append(chunk));

    const terminate = (): void => {
      terminateProcessGroup(child, "SIGTERM");
      forceKillTimer ??= setTimeout(
        () => terminateProcessGroup(child, "SIGKILL"),
        PROCESS_KILL_GRACE_MS,
      );
      forceKillTimer.unref?.();
    };
    const onAbort = (): void => {
      cancellationReason = request.requestSignal?.reason ?? new Error("Request was cancelled.");
      terminate();
    };
    request.requestSignal?.addEventListener("abort", onAbort, { once: true });
    if (request.requestSignal?.aborted) onAbort();

    const timeout = setTimeout(() => {
      timedOut = true;
      terminate();
    }, request.timeoutMs);
    timeout.unref?.();

    const cleanup = (): void => {
      clearTimeout(timeout);
      if (forceKillTimer) clearTimeout(forceKillTimer);
      request.requestSignal?.removeEventListener("abort", onAbort);
    };

    child.once("error", (cause) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(
        isFileNotFoundError(cause)
          ? new Error(
              "The macOS sandbox executable is unavailable; refusing unrestricted shell execution.",
            )
          : cause,
      );
    });
    child.once("close", (exitCode, exitSignal) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (cancellationReason !== undefined) {
        reject(
          cancellationReason instanceof Error
            ? cancellationReason
            : new Error("Request was cancelled."),
        );
        return;
      }
      resolve({
        command: request.command,
        cwd: request.root,
        exitCode,
        signal: exitSignal,
        stdout: stdout.value(),
        stderr: stderr.value(),
        durationMs: Math.max(0, Date.now() - request.startedAt),
        timedOut,
        truncated: stdout.truncated || stderr.truncated,
      });
    });
  });
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
): number {
  if (value === undefined) return options.timeoutMs;
  if (!Number.isSafeInteger(value) || value <= 0 || value > options.maxTimeoutMs) {
    throw new Error(`timeoutMs must be an integer between 1 and ${options.maxTimeoutMs}.`);
  }
  return value;
}

function validateCommand(command: string): void {
  if (typeof command !== "string" || !command.trim() || command.includes("\0")) {
    throw new Error("A non-empty shell command is required.");
  }
  if (Buffer.byteLength(command) > MAX_COMMAND_BYTES) {
    throw new Error(`Shell command exceeds the ${MAX_COMMAND_BYTES}-byte limit.`);
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

async function ensureExecutable(executable: string): Promise<void> {
  try {
    await access(executable, constants.X_OK);
  } catch {
    throw new Error(
      "The macOS sandbox executable is unavailable; refusing unrestricted shell execution.",
    );
  }
}

function seatbeltProfile(root: string, temporaryDirectory: string): string {
  return [
    "(version 1)",
    "(deny default)",
    "(allow process*)",
    "(allow file-read*)",
    `(allow file-write* (subpath ${sandboxString(root)}) (subpath ${sandboxString(temporaryDirectory)}) (literal \"/dev/null\"))`,
    "(allow sysctl-read)",
    "(allow mach-lookup)",
  ].join(" ");
}

function sandboxString(value: string): string {
  return JSON.stringify(value);
}

function shellEnvironment(root: string, temporaryDirectory: string): NodeJS.ProcessEnv {
  const environment: NodeJS.ProcessEnv = {
    PATH: process.env.PATH ?? "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    HOME: process.env.HOME ?? os.homedir(),
    SHELL: "/bin/zsh",
    PWD: root,
    OLDPWD: root,
    TMPDIR: `${temporaryDirectory}${path.sep}`,
    TMP: temporaryDirectory,
    TEMP: temporaryDirectory,
    TERM: "dumb",
    NO_COLOR: "1",
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

function terminateProcessGroup(child: ChildProcess, signal: NodeJS.Signals): void {
  if (child.pid) {
    try {
      process.kill(-child.pid, signal);
      return;
    } catch {
      // The group may already be gone; fall back to the direct child handle.
    }
  }
  try {
    child.kill(signal);
  } catch {
    // The process already exited.
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return;
  throw signal.reason instanceof Error ? signal.reason : new Error("Request was cancelled.");
}

function isFileNotFoundError(cause: unknown): boolean {
  return cause instanceof Error && "code" in cause && cause.code === "ENOENT";
}
