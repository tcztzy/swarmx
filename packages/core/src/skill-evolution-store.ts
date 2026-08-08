import { createHash } from "node:crypto";
import * as fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  applySkillEvolutionRecord,
  replaySkillEvolutionRecords,
  type SkillEvolutionRecord,
  SkillEvolutionRecordSchema,
  type SkillEvolutionState,
} from "./skill-evolution.js";

const DEFAULT_EVOLUTION_ROOT = path.join(
  process.env.SWARMX_HOME ?? os.homedir(),
  ".swarmx",
  "skill-evolution",
);

const EVENT_LOG_FILE = "evolution.jsonl";
const BLOB_DIRECTORY = "blobs";
const LOCK_FILE = "evolution.jsonl.lock";
const LOCK_TIMEOUT_MS = 15_000;
const LOCK_STALE_MS = 30_000;
const LOCK_POLL_MS = 25;

export interface SkillEvolutionStoreOptions {
  rootDir?: string;
  lockTimeoutMs?: number;
  lockStaleMs?: number;
}

export interface SkillEvolutionStoreInspection {
  state: SkillEvolutionState;
  tornTail: boolean;
  tornTailBytes: number;
}

export interface SkillEvolutionStoreAppendResult {
  state: SkillEvolutionState;
  appended: boolean;
}

export interface SkillEvolutionStoreRecoveryResult extends SkillEvolutionStoreInspection {
  recovered: boolean;
  removedBytes: number;
}

export interface SkillEvolutionBlobReference {
  ref: string;
  sha256: string;
  sizeBytes: number;
}

export class SkillEvolutionTornTailError extends Error {
  constructor(logPath: string, bytes: number) {
    super(`Skill evolution log ${logPath} has a torn tail of ${bytes} bytes.`);
    this.name = "SkillEvolutionTornTailError";
  }
}

/**
 * Append-only skill evolution ledger under `~/.swarmx/skill-evolution/`.
 * Records are strict, secret-free, and idempotent; promotion records carry
 * compare-and-swap expectations that replay enforces in order. Candidate and
 * dataset content lives in content-addressed blobs, never in the ledger.
 */
export class SkillEvolutionStore {
  readonly rootDir: string;
  readonly eventLogPath: string;
  readonly blobDir: string;
  private readonly lockTimeoutMs: number;
  private readonly lockStaleMs: number;

  constructor(options: SkillEvolutionStoreOptions = {}) {
    this.rootDir = path.resolve(options.rootDir ?? DEFAULT_EVOLUTION_ROOT);
    this.eventLogPath = path.join(this.rootDir, EVENT_LOG_FILE);
    this.blobDir = path.join(this.rootDir, BLOB_DIRECTORY);
    this.lockTimeoutMs = options.lockTimeoutMs ?? LOCK_TIMEOUT_MS;
    this.lockStaleMs = options.lockStaleMs ?? LOCK_STALE_MS;
  }

  inspect(): SkillEvolutionStoreInspection {
    const parsed = this.readRecords();
    return {
      state: replaySkillEvolutionRecords(parsed.records),
      tornTail: parsed.tornTail,
      tornTailBytes: parsed.tornTailBytes,
    };
  }

  state(): SkillEvolutionState {
    return this.inspect().state;
  }

  append(input: unknown | readonly unknown[]): SkillEvolutionStoreAppendResult {
    const candidates = (Array.isArray(input) ? input : [input]).map((record) =>
      SkillEvolutionRecordSchema.parse(record),
    );
    if (candidates.length === 0) {
      return { state: this.state(), appended: false };
    }
    this.ensureDirectories();
    return this.withLock(() => {
      const parsed = this.readRecords();
      if (parsed.tornTail) {
        throw new SkillEvolutionTornTailError(this.eventLogPath, parsed.tornTailBytes);
      }
      let state = replaySkillEvolutionRecords(parsed.records);
      let appended = false;
      const appendedRecords: SkillEvolutionRecord[] = [];
      for (const candidate of candidates) {
        const next = applySkillEvolutionRecord(state, candidate);
        if (next !== state) {
          appendedRecords.push(candidate);
          appended = true;
        }
        state = next;
      }
      if (appendedRecords.length > 0) {
        appendDurably(
          this.eventLogPath,
          appendedRecords.map((record) => `${JSON.stringify(record)}\n`).join(""),
        );
      }
      return { state, appended };
    });
  }

  recoverTornTail(): SkillEvolutionStoreRecoveryResult {
    this.ensureDirectories();
    return this.withLock(() => {
      const parsed = this.readRecords();
      if (!parsed.tornTail) {
        return {
          state: replaySkillEvolutionRecords(parsed.records),
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
        state: replaySkillEvolutionRecords(parsed.records),
        tornTail: false,
        tornTailBytes: 0,
        recovered: true,
        removedBytes: parsed.tornTailBytes,
      };
    });
  }

  putJson(value: unknown): SkillEvolutionBlobReference {
    const encoded = Buffer.from(`${JSON.stringify(value)}\n`, "utf8");
    return this.putBytes(encoded);
  }

  putBytes(value: Uint8Array): SkillEvolutionBlobReference {
    const encoded = Buffer.from(value);
    const sha256 = hashBytes(encoded);
    const ref = `sha256:${sha256}`;
    this.ensureDirectories();
    const filePath = this.blobPath(ref);
    if (!fs.existsSync(filePath)) {
      const temporaryPath = `${filePath}.${process.pid}.${Date.now()}.tmp`;
      writeDurably(temporaryPath, encoded);
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
      throw new Error(`Skill evolution blob is not valid JSON: ${ref}`, { cause: error });
    }
  }

  readBytes(ref: string): Buffer {
    const encoded = fs.readFileSync(this.blobPath(ref));
    if (hashBytes(encoded) !== ref.slice("sha256:".length)) {
      throw new Error(`Skill evolution blob checksum mismatch: ${ref}`);
    }
    return encoded;
  }

  pathForBlob(ref: string): string {
    return this.blobPath(ref);
  }

  private blobPath(ref: string): string {
    const match = /^sha256:([a-f0-9]{64})$/u.exec(ref);
    if (!match?.[1]) {
      throw new Error("Skill evolution blob refs must use sha256:<64 lowercase hex>.");
    }
    return path.join(this.blobDir, `${match[1]}.blob`);
  }

  private ensureDirectories(): void {
    fs.mkdirSync(this.blobDir, { recursive: true, mode: 0o700 });
    fs.chmodSync(this.rootDir, 0o700);
    fs.chmodSync(this.blobDir, 0o700);
  }

  private readRecords(): ParsedEvolutionRecords {
    if (!fs.existsSync(this.eventLogPath)) {
      return { records: [], tornTail: false, tornTailBytes: 0, completeBytes: 0 };
    }
    const source = fs.readFileSync(this.eventLogPath);
    const terminated = source.length === 0 || source.at(-1) === 0x0a;
    const records: SkillEvolutionRecord[] = [];
    let offset = 0;
    let completeBytes = 0;

    while (offset < source.length) {
      const newline = source.indexOf(0x0a, offset);
      const isFinal = newline < 0;
      const end = isFinal ? source.length : newline;
      const line = source.subarray(offset, end);
      if (line.length > 0) {
        try {
          records.push(SkillEvolutionRecordSchema.parse(JSON.parse(line.toString("utf8"))));
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
            `${path.basename(this.eventLogPath)} line ${records.length + 1} is corrupt: ${message}`,
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
    const lockPath = path.join(this.rootDir, LOCK_FILE);
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
          throw new Error(`Timed out waiting for skill evolution writer lock: ${lockPath}`);
        }
        sleepSync(LOCK_POLL_MS);
      }
    }

    try {
      return action();
    } finally {
      try {
        const lock = JSON.parse(fs.readFileSync(lockPath, "utf8")) as { token?: string };
        if (lock.token === token) fs.unlinkSync(lockPath);
      } catch {
        // A racing recovery may already have removed a stale lock.
      }
    }
  }
}

interface ParsedEvolutionRecords {
  records: SkillEvolutionRecord[];
  tornTail: boolean;
  tornTailBytes: number;
  completeBytes: number;
}

function hashBytes(value: Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function appendDurably(filePath: string, content: string): void {
  const descriptor = fs.openSync(filePath, "a", 0o600);
  try {
    fs.writeSync(descriptor, content);
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
}

function writeDurably(filePath: string, content: Uint8Array): void {
  const descriptor = fs.openSync(filePath, "w", 0o600);
  try {
    fs.writeSync(descriptor, content);
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
}

function fsyncDirectory(directoryPath: string): void {
  const descriptor = fs.openSync(directoryPath, "r");
  try {
    fs.fsyncSync(descriptor);
  } catch {
    // Some platforms cannot fsync directories; the write itself was durable.
  } finally {
    fs.closeSync(descriptor);
  }
}

function removeStaleLock(lockPath: string, staleMs: number): boolean {
  try {
    const stat = fs.statSync(lockPath);
    if (Date.now() - stat.mtimeMs < staleMs) return false;
    fs.unlinkSync(lockPath);
    return true;
  } catch {
    return false;
  }
}

function isFileExists(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error as NodeJS.ErrnoException).code === "EEXIST"
  );
}

function sleepSync(ms: number): void {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}
