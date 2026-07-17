import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { constants, type FSWatcher, watch } from "node:fs";
import { lstat, mkdir, open, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const SCHEDULED_TASKS_FILE = "scheduled_tasks.json";
const SCHEDULED_TASKS_LOCK = "scheduled_tasks.lock";
const WATCH_DEBOUNCE_MS = 300;

export interface ClaudeScheduledTask {
  id: string;
  cron: string;
  prompt: string;
  createdAt: number;
  lastFiredAt?: number;
  recurring?: true;
  permanent?: true;
  createdBySessionId?: string;
  createdByPid?: number;
  createdByProcStart?: string;
}

export interface ClaudeScheduledTaskOwner {
  sessionId: string;
  pid: number;
  procStart: string;
}

interface ClaudeScheduledTaskLock extends ClaudeScheduledTaskOwner {
  acquiredAt: number;
}

export interface ClaudeScheduledTaskStoreOptions {
  validateCron?: (cron: string) => boolean;
  isProcessAlive?: (pid: number) => boolean | Promise<boolean>;
  processStart?: (pid: number) => string | undefined | Promise<string | undefined>;
  now?: () => number;
  watch?: boolean;
  watchDebounceMs?: number;
}

export class ClaudeScheduledTaskStore {
  readonly root: string;
  readonly directoryPath: string;
  readonly filePath: string;
  readonly lockPath: string;
  readonly #validateCron: (cron: string) => boolean;
  readonly #isProcessAlive: (pid: number) => boolean | Promise<boolean>;
  readonly #processStart: (pid: number) => string | undefined | Promise<string | undefined>;
  readonly #now: () => number;
  readonly #watchEnabled: boolean;
  readonly #watchDebounceMs: number;
  readonly #listeners = new Set<() => void>();
  #mutation = Promise.resolve();
  #lockMutation = Promise.resolve();
  #watcher?: FSWatcher;
  #watchTimer?: ReturnType<typeof setTimeout>;
  #startPromise?: Promise<void>;
  #started = false;
  #closed = false;

  constructor(root: string, options: ClaudeScheduledTaskStoreOptions = {}) {
    this.root = path.resolve(root);
    this.directoryPath = path.join(this.root, ".claude");
    this.filePath = path.join(this.directoryPath, SCHEDULED_TASKS_FILE);
    this.lockPath = path.join(this.directoryPath, SCHEDULED_TASKS_LOCK);
    this.#validateCron = options.validateCron ?? (() => true);
    this.#isProcessAlive = options.isProcessAlive ?? defaultIsProcessAlive;
    this.#processStart = options.processStart ?? resolveClaudeProcessStart;
    this.#now = options.now ?? Date.now;
    this.#watchEnabled = options.watch ?? true;
    this.#watchDebounceMs = options.watchDebounceMs ?? WATCH_DEBOUNCE_MS;
  }

  async start(): Promise<void> {
    this.#assertOpen();
    if (this.#started) return;
    this.#startPromise ??= this.#start();
    try {
      await this.#startPromise;
    } catch (error) {
      this.#startPromise = undefined;
      throw error;
    }
  }

  async #start(): Promise<void> {
    await mkdir(this.directoryPath, { recursive: true, mode: 0o700 });
    const directory = await lstat(this.directoryPath);
    if (!directory.isDirectory() || directory.isSymbolicLink()) {
      throw new Error("Claude scheduled-task directory must be a real Project directory.");
    }
    if (this.#watchEnabled) {
      this.#watcher = watch(this.directoryPath, { persistent: false }, (_event, filename) => {
        if (filename !== null && filename.toString() !== SCHEDULED_TASKS_FILE) return;
        if (this.#watchTimer) clearTimeout(this.#watchTimer);
        this.#watchTimer = setTimeout(() => {
          this.#watchTimer = undefined;
          this.#notify();
        }, this.#watchDebounceMs);
      });
    }
    this.#started = true;
  }

  async read(): Promise<ClaudeScheduledTask[]> {
    this.#assertOpen();
    let parsed: unknown;
    try {
      parsed = JSON.parse(await readRegularFile(this.filePath));
    } catch (error) {
      if (isNodeError(error, "ENOENT") || error instanceof SyntaxError) return [];
      throw error;
    }
    if (!isRecord(parsed) || !Array.isArray(parsed.tasks)) return [];
    return parsed.tasks.flatMap((value) => {
      const task = parseScheduledTask(value, this.#validateCron);
      return task ? [task] : [];
    });
  }

  async add(task: ClaudeScheduledTask): Promise<void> {
    if (!/^[0-9a-f]{8}$/i.test(task.id)) {
      throw new Error("Durable Claude cron ids must be eight hexadecimal characters.");
    }
    const normalized = parseScheduledTask(task, this.#validateCron);
    if (!normalized) throw new Error("Cannot persist an invalid Claude scheduled task.");
    await this.#mutate((tasks) => {
      if (tasks.some((candidate) => candidate.id === normalized.id)) {
        throw new Error(`A scheduled task with id '${normalized.id}' already exists.`);
      }
      return { tasks: [...tasks, normalized], result: undefined };
    });
  }

  async remove(id: string): Promise<boolean> {
    return this.#mutate((tasks) => {
      const next = tasks.filter((task) => task.id !== id);
      return { tasks: next, result: next.length !== tasks.length };
    });
  }

  async markFired(id: string, firedAt: number): Promise<boolean> {
    return this.#mutate((tasks) => {
      let found = false;
      const next = tasks.map((task) => {
        if (task.id !== id) return task;
        found = true;
        return { ...task, lastFiredAt: firedAt };
      });
      return { tasks: next, result: found };
    });
  }

  subscribe(listener: () => void): () => void {
    this.#assertOpen();
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  async acquireLock(owner: ClaudeScheduledTaskOwner): Promise<boolean> {
    return this.#withLockMutation(async () => {
      await this.start();
      for (let attempt = 0; attempt < 3; attempt++) {
        const lock: ClaudeScheduledTaskLock = { ...owner, acquiredAt: this.#now() };
        try {
          await writeFile(this.lockPath, `${JSON.stringify(lock)}\n`, {
            encoding: "utf8",
            flag: "wx",
            mode: 0o600,
          });
          return true;
        } catch (error) {
          if (!isNodeError(error, "EEXIST")) throw error;
        }
        const existing = await this.#readLock();
        if (existing && sameOwner(existing, owner)) return true;
        if (existing && (await this.#lockIsLive(existing))) return false;
        await rm(this.lockPath, { force: true });
      }
      return false;
    });
  }

  async releaseLock(owner: ClaudeScheduledTaskOwner): Promise<void> {
    await this.#withLockMutation(async () => {
      const existing = await this.#readLock();
      if (existing && sameOwner(existing, owner)) {
        await rm(this.lockPath, { force: true });
      }
    });
  }

  async creatorIsLive(task: ClaudeScheduledTask): Promise<boolean> {
    if (task.createdByPid === undefined || task.createdByProcStart === undefined) return false;
    if (!(await this.#isProcessAlive(task.createdByPid))) return false;
    if (task.createdByProcStart.startsWith("unknown-process-start-")) return true;
    const actualStart = await this.#processStart(task.createdByPid);
    return actualStart === undefined || actualStart === task.createdByProcStart;
  }

  async close(): Promise<void> {
    if (this.#closed) return;
    if (this.#watchTimer) clearTimeout(this.#watchTimer);
    this.#watchTimer = undefined;
    this.#watcher?.close();
    this.#watcher = undefined;
    this.#listeners.clear();
    await Promise.allSettled([this.#startPromise, this.#mutation, this.#lockMutation]);
    this.#closed = true;
  }

  async #mutate<Result>(
    mutate: (tasks: ClaudeScheduledTask[]) => {
      tasks: ClaudeScheduledTask[];
      result: Result;
    },
  ): Promise<Result> {
    this.#assertOpen();
    await this.start();
    const operation = this.#mutation.then(async () => {
      const current = await this.read();
      const { tasks, result } = mutate(current);
      if (tasks !== current) await this.#write(tasks);
      return result;
    });
    this.#mutation = operation.then(
      () => undefined,
      () => undefined,
    );
    return operation;
  }

  async #write(tasks: ClaudeScheduledTask[]): Promise<void> {
    const temporary = path.join(
      this.directoryPath,
      `.${SCHEDULED_TASKS_FILE}.${process.pid}.${randomUUID()}.tmp`,
    );
    const contents = `${JSON.stringify({ tasks }, null, 2)}\n`;
    try {
      await writeFile(temporary, contents, { encoding: "utf8", flag: "wx", mode: 0o600 });
      await rename(temporary, this.filePath);
    } catch (error) {
      await rm(temporary, { force: true });
      throw error;
    }
    this.#notify();
  }

  async #withLockMutation<Result>(operation: () => Promise<Result>): Promise<Result> {
    this.#assertOpen();
    const result = this.#lockMutation.then(operation);
    this.#lockMutation = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  async #readLock(): Promise<ClaudeScheduledTaskLock | undefined> {
    let value: unknown;
    try {
      value = JSON.parse(await readRegularFile(this.lockPath));
    } catch (error) {
      if (isNodeError(error, "ENOENT") || error instanceof SyntaxError) return undefined;
      throw error;
    }
    if (
      !isRecord(value) ||
      typeof value.sessionId !== "string" ||
      !Number.isInteger(value.pid) ||
      typeof value.procStart !== "string" ||
      typeof value.acquiredAt !== "number" ||
      !Number.isFinite(value.acquiredAt)
    ) {
      return undefined;
    }
    return {
      sessionId: value.sessionId,
      pid: value.pid as number,
      procStart: value.procStart,
      acquiredAt: value.acquiredAt,
    };
  }

  async #lockIsLive(lock: ClaudeScheduledTaskLock): Promise<boolean> {
    if (!(await this.#isProcessAlive(lock.pid))) return false;
    if (lock.procStart.startsWith("unknown-process-start-")) return true;
    const actualStart = await this.#processStart(lock.pid);
    return actualStart === undefined || actualStart === lock.procStart;
  }

  #notify(): void {
    for (const listener of this.#listeners) listener();
  }

  #assertOpen(): void {
    if (this.#closed) throw new Error("Claude scheduled-task store is closed.");
  }
}

export async function resolveClaudeProcessStart(pid: number): Promise<string | undefined> {
  try {
    const { stdout } = await execFileAsync("ps", ["-o", "lstart=", "-p", String(pid)], {
      env: { LC_ALL: "C", PATH: process.env.PATH, TZ: "UTC" },
      timeout: 2_000,
    });
    const value = stdout.trim();
    return value || undefined;
  } catch {
    return undefined;
  }
}

function parseScheduledTask(
  value: unknown,
  validateCron: (cron: string) => boolean,
): ClaudeScheduledTask | undefined {
  if (
    !isRecord(value) ||
    typeof value.id !== "string" ||
    !value.id ||
    typeof value.cron !== "string" ||
    !validateCron(value.cron) ||
    typeof value.prompt !== "string" ||
    typeof value.createdAt !== "number" ||
    !Number.isFinite(value.createdAt)
  ) {
    return undefined;
  }
  return {
    id: value.id,
    cron: value.cron,
    prompt: value.prompt,
    createdAt: value.createdAt,
    ...(isFiniteNumber(value.lastFiredAt) ? { lastFiredAt: value.lastFiredAt } : {}),
    ...(value.recurring === true ? { recurring: true as const } : {}),
    ...(value.permanent === true ? { permanent: true as const } : {}),
    ...(typeof value.createdBySessionId === "string"
      ? { createdBySessionId: value.createdBySessionId }
      : {}),
    ...(Number.isInteger(value.createdByPid) ? { createdByPid: value.createdByPid as number } : {}),
    ...(typeof value.createdByProcStart === "string"
      ? { createdByProcStart: value.createdByProcStart }
      : {}),
  };
}

function defaultIsProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return isNodeError(error, "EPERM");
  }
}

function sameOwner(left: ClaudeScheduledTaskOwner, right: ClaudeScheduledTaskOwner): boolean {
  return (
    left.sessionId === right.sessionId &&
    left.pid === right.pid &&
    left.procStart === right.procStart
  );
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNodeError(error: unknown, code: string): error is NodeJS.ErrnoException {
  return error instanceof Error && "code" in error && error.code === code;
}

async function readRegularFile(filePath: string): Promise<string> {
  const handle = await open(filePath, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const file = await handle.stat();
    if (!file.isFile()) throw new Error(`Expected a regular file at ${filePath}.`);
    return await handle.readFile("utf8");
  } finally {
    await handle.close();
  }
}
