import { spawn } from "node:child_process";
import * as fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  ensureTaskSupervisorToken,
  TaskSupervisorClient,
  type TaskSupervisorCommand,
  type TaskSupervisorResponse,
  taskSupervisorPaths,
} from "@swarmx/core";

const STARTUP_TIMEOUT_MS = 5_000;
const STARTUP_POLL_MS = 50;

export type TaskSupervisorSuccessResponse = Exclude<TaskSupervisorResponse, { ok: false }>;

export interface DesktopTaskSupervisorLike {
  request(command: TaskSupervisorCommand): Promise<TaskSupervisorSuccessResponse>;
}

export interface DesktopTaskSupervisorOptions {
  rootDir?: string;
  entryPath?: string;
  executablePath?: string;
}

/** Starts or reconnects to the local supervisor without exposing its token to Renderer. */
export class DesktopTaskSupervisor implements DesktopTaskSupervisorLike {
  private readonly rootDir?: string;
  private readonly entryPath: string;
  private readonly executablePath: string;
  private startup?: Promise<TaskSupervisorClient>;

  constructor(options: DesktopTaskSupervisorOptions = {}) {
    this.rootDir = options.rootDir;
    this.entryPath =
      options.entryPath ??
      path.join(path.dirname(fileURLToPath(import.meta.url)), "task-supervisor-entry.js");
    this.executablePath = options.executablePath ?? process.execPath;
  }

  async request(command: TaskSupervisorCommand): Promise<TaskSupervisorSuccessResponse> {
    const client = await this.client();
    return await client.request(command);
  }

  private async client(): Promise<TaskSupervisorClient> {
    this.startup ??= this.connectOrStart().catch((error) => {
      this.startup = undefined;
      throw error;
    });
    return await this.startup;
  }

  private async connectOrStart(): Promise<TaskSupervisorClient> {
    const paths = taskSupervisorPaths(this.rootDir);
    const token = ensureTaskSupervisorToken(paths.rootDir);
    const client = new TaskSupervisorClient({ socketPath: paths.socketPath, token });
    const staleSocketIdentity = socketIdentity(paths.socketPath);
    try {
      await client.request({ operation: "ping" });
      return client;
    } catch (error) {
      if (!isUnavailableSupervisor(error)) throw error;
      removeMatchingStaleSocket(paths.socketPath, staleSocketIdentity);
    }

    const child = spawn(this.executablePath, [this.entryPath], {
      detached: true,
      stdio: "ignore",
      cwd: paths.rootDir,
      env: supervisorEnvironment(paths.rootDir),
    });
    child.unref();

    const deadline = Date.now() + STARTUP_TIMEOUT_MS;
    let lastError: unknown;
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, STARTUP_POLL_MS));
      try {
        await client.request({ operation: "ping" });
        return client;
      } catch (error) {
        lastError = error;
      }
    }
    throw new Error("The local task supervisor did not start in time.", { cause: lastError });
  }
}

function supervisorEnvironment(rootDir: string): NodeJS.ProcessEnv {
  const allowed = ["PATH", "TMPDIR", "TMP", "TEMP", "SYSTEMROOT", "WINDIR"];
  const env = Object.fromEntries(
    allowed.flatMap((key) => (process.env[key] === undefined ? [] : [[key, process.env[key]]])),
  );
  return {
    ...env,
    ELECTRON_RUN_AS_NODE: "1",
    SWARMX_TASK_RUNTIME_ROOT: rootDir,
  };
}

function socketIdentity(socketPath: string): string | undefined {
  if (process.platform === "win32") return undefined;
  try {
    const status = fs.statSync(socketPath);
    return `${status.dev}:${status.ino}`;
  } catch {
    return undefined;
  }
}

function removeMatchingStaleSocket(socketPath: string, expected: string | undefined): void {
  if (process.platform === "win32" || !expected || socketIdentity(socketPath) !== expected) return;
  try {
    fs.unlinkSync(socketPath);
  } catch (error) {
    if (!isNodeError(error) || error.code !== "ENOENT") throw error;
  }
}

function isUnavailableSupervisor(error: unknown): boolean {
  return (
    isNodeError(error) &&
    ["ENOENT", "ECONNREFUSED", "ECONNRESET", "EPIPE"].includes(error.code ?? "")
  );
}

function isNodeError(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && "code" in error;
}
