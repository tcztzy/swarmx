import { createHash } from "node:crypto";
import * as fs from "node:fs";
import { homedir } from "node:os";
import * as path from "node:path";
import {
  applyTaskRuntimeEvent,
  replayTaskRuntimeEvents,
  type TaskRuntimeEvent,
  TaskRuntimeEventSchema,
  type TaskRuntimeState,
} from "./task-runtime.js";
import { TaskWorkerPayloadSchema } from "./task-worker-protocol.js";

const DEFAULT_TASK_RUNTIME_ROOT = path.join(homedir(), ".swarmx", "task-runtime");
const EVENT_LOG_FILE = "events.jsonl";
const BLOB_DIRECTORY = "blobs";
const LOCK_TIMEOUT_MS = 5_000;
const LOCK_STALE_MS = 30_000;
const LOCK_POLL_MS = 10;

export interface TaskRuntimeStoreOptions {
  rootDir?: string;
  lockTimeoutMs?: number;
  lockStaleMs?: number;
}

export interface TaskRuntimeStoreInspection {
  state: TaskRuntimeState;
  tornTail: boolean;
  tornTailBytes: number;
}

export interface TaskRuntimeAppendResult {
  state: TaskRuntimeState;
  events: TaskRuntimeEvent[];
  appended: boolean;
}

export interface TaskRuntimeRecoveryResult extends TaskRuntimeStoreInspection {
  recovered: boolean;
  removedBytes: number;
}

export interface TaskRuntimeBlobReference {
  ref: string;
  sha256: string;
  sizeBytes: number;
}

export class TaskRuntimeTornTailError extends Error {
  readonly filePath: string;
  readonly tornTailBytes: number;

  constructor(filePath: string, tornTailBytes: number) {
    super(
      `Task runtime event log has a torn final record; call recoverTornTail() before appending: ${filePath}`,
    );
    this.name = "TaskRuntimeTornTailError";
    this.filePath = filePath;
    this.tornTailBytes = tornTailBytes;
  }
}

/** Append-only authority for WorkItems. Session logs never participate in this replay. */
export class TaskRuntimeStore {
  readonly rootDir: string;
  readonly eventLogPath: string;
  readonly blobDir: string;
  private readonly lockTimeoutMs: number;
  private readonly lockStaleMs: number;

  constructor(options: TaskRuntimeStoreOptions = {}) {
    this.rootDir = path.resolve(options.rootDir ?? DEFAULT_TASK_RUNTIME_ROOT);
    this.eventLogPath = path.join(this.rootDir, EVENT_LOG_FILE);
    this.blobDir = path.join(this.rootDir, BLOB_DIRECTORY);
    this.lockTimeoutMs = options.lockTimeoutMs ?? LOCK_TIMEOUT_MS;
    this.lockStaleMs = options.lockStaleMs ?? LOCK_STALE_MS;
  }

  inspect(): TaskRuntimeStoreInspection {
    const parsed = this.readEvents();
    return {
      state: replayTaskRuntimeEvents(parsed.records),
      tornTail: parsed.tornTail,
      tornTailBytes: parsed.tornTailBytes,
    };
  }

  state(): TaskRuntimeState {
    return this.inspect().state;
  }

  append(input: unknown | readonly unknown[]): TaskRuntimeAppendResult {
    const candidates = (Array.isArray(input) ? input : [input]).map((event) =>
      TaskRuntimeEventSchema.parse(event),
    );
    if (candidates.length === 0) {
      return { state: this.state(), events: [], appended: false };
    }

    this.ensureDirectories();
    return this.withLock(() => {
      const parsed = this.readEvents();
      if (parsed.tornTail) {
        throw new TaskRuntimeTornTailError(this.eventLogPath, parsed.tornTailBytes);
      }
      let state = replayTaskRuntimeEvents(parsed.records);
      const events: TaskRuntimeEvent[] = [];
      for (const event of candidates) {
        const next = applyTaskRuntimeEvent(state, event);
        if (next !== state) events.push(event);
        state = next;
      }
      if (events.length > 0) appendFileDurably(this.eventLogPath, jsonl(events));
      return { state, events, appended: events.length > 0 };
    });
  }

  recoverTornTail(): TaskRuntimeRecoveryResult {
    this.ensureDirectories();
    return this.withLock(() => {
      const parsed = this.readEvents();
      if (!parsed.tornTail) {
        return {
          state: replayTaskRuntimeEvents(parsed.records),
          tornTail: false,
          tornTailBytes: 0,
          recovered: false,
          removedBytes: 0,
        };
      }
      const descriptor = fs.openSync(this.eventLogPath, "r+");
      try {
        fs.ftruncateSync(descriptor, parsed.completeBytes);
        fs.fsyncSync(descriptor);
      } finally {
        fs.closeSync(descriptor);
      }
      fs.chmodSync(this.eventLogPath, 0o600);
      return {
        state: replayTaskRuntimeEvents(parsed.records),
        tornTail: false,
        tornTailBytes: 0,
        recovered: true,
        removedBytes: parsed.tornTailBytes,
      };
    });
  }

  putJson(value: unknown): TaskRuntimeBlobReference {
    const json = TaskWorkerPayloadSchema.parse(value);
    const encoded = Buffer.from(`${JSON.stringify(json)}\n`, "utf8");
    return this.putBytes(encoded);
  }

  putBytes(value: Uint8Array): TaskRuntimeBlobReference {
    const encoded = Buffer.from(value);
    const sha256 = hashBytes(encoded);
    const ref = `sha256:${sha256}`;
    this.ensureDirectories();
    const filePath = this.blobPath(ref);
    if (!fs.existsSync(filePath)) {
      const temporaryPath = `${filePath}.${process.pid}.${Date.now()}.tmp`;
      writeNewFileDurably(temporaryPath, encoded);
      try {
        fs.renameSync(temporaryPath, filePath);
        fsyncDirectory(this.blobDir);
      } catch (error) {
        if (!fs.existsSync(filePath)) throw error;
        try {
          fs.unlinkSync(temporaryPath);
        } catch {
          // Another writer won the content-addressed race.
        }
      }
    }
    return { ref, sha256, sizeBytes: encoded.byteLength };
  }

  readJson(ref: string): unknown {
    const encoded = this.readBytes(ref);
    try {
      return JSON.parse(encoded.toString("utf8"));
    } catch (error) {
      throw new Error(`Task runtime blob is not valid JSON: ${ref}`, { cause: error });
    }
  }

  readBytes(ref: string): Buffer {
    const encoded = fs.readFileSync(this.blobPath(ref));
    if (hashBytes(encoded) !== ref.slice("sha256:".length)) {
      throw new Error(`Task runtime blob checksum mismatch: ${ref}`);
    }
    return encoded;
  }

  pathForBlob(ref: string): string {
    return this.blobPath(ref);
  }

  private blobPath(ref: string): string {
    const match = /^sha256:([a-f0-9]{64})$/u.exec(ref);
    if (!match?.[1]) throw new Error("Task runtime blob refs must use sha256:<64 lowercase hex>.");
    return path.join(this.blobDir, `${match[1]}.blob`);
  }

  private ensureDirectories(): void {
    fs.mkdirSync(this.blobDir, { recursive: true, mode: 0o700 });
    fs.chmodSync(this.rootDir, 0o700);
    fs.chmodSync(this.blobDir, 0o700);
  }

  private readEvents(): ParsedTaskEvents {
    if (!fs.existsSync(this.eventLogPath)) {
      return { records: [], tornTail: false, tornTailBytes: 0, completeBytes: 0 };
    }
    const source = fs.readFileSync(this.eventLogPath);
    const terminated = source.length === 0 || source.at(-1) === 0x0a;
    const records: TaskRuntimeEvent[] = [];
    let offset = 0;
    let completeBytes = 0;

    while (offset < source.length) {
      const newline = source.indexOf(0x0a, offset);
      const isFinal = newline < 0;
      const end = isFinal ? source.length : newline;
      const line = source.subarray(offset, end);
      const lineNumber = records.length + 1;
      if (line.length > 0) {
        try {
          records.push(TaskRuntimeEventSchema.parse(JSON.parse(line.toString("utf8"))));
        } catch (error) {
          if (isFinal && !terminated) {
            return {
              records,
              tornTail: true,
              tornTailBytes: source.length - offset,
              completeBytes,
            };
          }
          const message = error instanceof Error ? error.message : String(error);
          throw new Error(
            `${path.basename(this.eventLogPath)} line ${lineNumber} is corrupt: ${message}`,
          );
        }
      }
      if (isFinal) {
        completeBytes = source.length;
        break;
      }
      offset = newline + 1;
      completeBytes = offset;
    }
    return { records, tornTail: false, tornTailBytes: 0, completeBytes };
  }

  private withLock<T>(action: () => T): T {
    const lockPath = `${this.eventLogPath}.lock`;
    const startedAt = Date.now();
    const token = `${process.pid}:${Date.now()}:${Math.random()}`;
    while (true) {
      try {
        const descriptor = fs.openSync(lockPath, "wx", 0o600);
        try {
          fs.writeFileSync(
            descriptor,
            JSON.stringify({ pid: process.pid, token, createdAt: new Date().toISOString() }),
            "utf8",
          );
          fs.fsyncSync(descriptor);
        } finally {
          fs.closeSync(descriptor);
        }
        break;
      } catch (error) {
        if (!isFileExists(error)) throw error;
        if (removeStaleLock(lockPath, this.lockStaleMs)) continue;
        if (Date.now() - startedAt >= this.lockTimeoutMs) {
          throw new Error(`Timed out waiting for task runtime writer lock: ${lockPath}`);
        }
        sleepSync(LOCK_POLL_MS);
      }
    }

    try {
      return action();
    } finally {
      try {
        const lock = parseRecord(JSON.parse(fs.readFileSync(lockPath, "utf8")), "lock");
        if (lock.token === token) fs.unlinkSync(lockPath);
      } catch {
        // A racing recovery may already have removed a stale lock.
      }
    }
  }
}

interface ParsedTaskEvents {
  records: TaskRuntimeEvent[];
  tornTail: boolean;
  tornTailBytes: number;
  completeBytes: number;
}

function appendFileDurably(filePath: string, content: string): void {
  const existed = fs.existsSync(filePath);
  const prefix =
    existed && fs.statSync(filePath).size > 0 && !fileEndsWithNewline(filePath) ? "\n" : "";
  const descriptor = fs.openSync(filePath, "a", 0o600);
  try {
    fs.writeFileSync(descriptor, `${prefix}${content}`, "utf8");
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
  fs.chmodSync(filePath, 0o600);
  if (!existed) fsyncDirectory(path.dirname(filePath));
}

function writeNewFileDurably(filePath: string, content: Uint8Array): void {
  const descriptor = fs.openSync(filePath, "wx", 0o600);
  try {
    fs.writeFileSync(descriptor, content);
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
}

function fileEndsWithNewline(filePath: string): boolean {
  const descriptor = fs.openSync(filePath, "r");
  try {
    const size = fs.fstatSync(descriptor).size;
    if (size === 0) return true;
    const byte = Buffer.allocUnsafe(1);
    fs.readSync(descriptor, byte, 0, 1, size - 1);
    return byte[0] === 0x0a;
  } finally {
    fs.closeSync(descriptor);
  }
}

function jsonl(records: readonly TaskRuntimeEvent[]): string {
  return `${records.map((record) => JSON.stringify(record)).join("\n")}\n`;
}

function hashBytes(value: Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function fsyncDirectory(directory: string): void {
  const descriptor = fs.openSync(directory, "r");
  try {
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
}

function removeStaleLock(lockPath: string, staleMs: number): boolean {
  try {
    const age = Date.now() - fs.statSync(lockPath).mtimeMs;
    const lock = parseRecord(JSON.parse(fs.readFileSync(lockPath, "utf8")), "lock");
    const pid = typeof lock.pid === "number" ? lock.pid : undefined;
    if (pid && processIsAlive(pid)) return false;
    if (!pid && age < staleMs) return false;
    fs.unlinkSync(lockPath);
    return true;
  } catch (error) {
    if (isNoSuchFile(error)) return true;
    try {
      if (Date.now() - fs.statSync(lockPath).mtimeMs < staleMs) return false;
      fs.unlinkSync(lockPath);
      return true;
    } catch (nested) {
      if (isNoSuchFile(nested)) return true;
      return false;
    }
  }
}

function processIsAlive(pid: number): boolean {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return !isNoSuchProcess(error);
  }
}

function sleepSync(milliseconds: number): void {
  const signal = new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT));
  Atomics.wait(signal, 0, 0, milliseconds);
}

function parseRecord(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function isFileExists(error: unknown): boolean {
  return errorCode(error) === "EEXIST";
}

function isNoSuchFile(error: unknown): boolean {
  return errorCode(error) === "ENOENT";
}

function isNoSuchProcess(error: unknown): boolean {
  return errorCode(error) === "ESRCH";
}

function errorCode(error: unknown): string | undefined {
  return typeof error === "object" && error !== null && "code" in error
    ? String((error as { code?: unknown }).code)
    : undefined;
}
