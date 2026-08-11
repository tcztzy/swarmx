import { createHash } from "node:crypto";
import * as fs from "node:fs";
import { homedir } from "node:os";
import * as path from "node:path";
import { DatabaseSync } from "node:sqlite";
import {
  type ContextArtifactReference,
  ContextArtifactReferenceSchema,
  type ContextArtifactStore,
  type ContextEngineEvent,
  type ContextEngineEventInput,
  ContextEngineEventSchema,
  type ContextEventScanSpec,
  type ContextEventStore,
  type ContextHistorySnapshot,
  createContextEvent,
  createContextHistorySnapshot,
} from "./context-engine.js";

const DEFAULT_ROOT = path.join(homedir(), ".swarmx", "context-engine-replay");
const SQLITE_DATABASE_FILE = "events.sqlite";
const EVENT_LOG_FILE = "events.jsonl";
const ARTIFACT_DIRECTORY = "artifacts";
const LOCK_TIMEOUT_MS = 5_000;
const LOCK_STALE_MS = 30_000;
const LOCK_POLL_MS = 10;

export interface JsonlContextEventStoreOptions {
  rootDir?: string;
  lockTimeoutMs?: number;
  lockStaleMs?: number;
}

export interface SqliteContextEventStoreOptions {
  rootDir?: string;
}

/** SQLite WAL replay store with schema-level guards against update and delete. */
export class SqliteContextEventStore implements ContextEventStore {
  readonly rootDir: string;
  readonly databasePath: string;
  readonly journalMode: string;
  private readonly database: DatabaseSync;

  constructor(options: SqliteContextEventStoreOptions = {}) {
    this.rootDir = path.resolve(options.rootDir ?? DEFAULT_ROOT);
    this.databasePath = path.join(this.rootDir, SQLITE_DATABASE_FILE);
    fs.mkdirSync(this.rootDir, { recursive: true, mode: 0o700 });
    fs.chmodSync(this.rootDir, 0o700);
    this.database = new DatabaseSync(this.databasePath, {
      allowExtension: false,
      enableDoubleQuotedStringLiterals: false,
      enableForeignKeyConstraints: true,
    });
    const journal = this.database.prepare("PRAGMA journal_mode = WAL").get() as
      | { journal_mode?: unknown }
      | undefined;
    this.journalMode = String(journal?.journal_mode ?? "").toLocaleLowerCase();
    if (this.journalMode !== "wal") {
      this.database.close();
      throw new Error(`Context SQLite store requires WAL mode; received ${this.journalMode}.`);
    }
    this.database.exec(`
      PRAGMA synchronous = FULL;
      PRAGMA foreign_keys = ON;
      PRAGMA trusted_schema = OFF;
      CREATE TABLE IF NOT EXISTS context_events (
        event_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        task_id TEXT,
        seq INTEGER NOT NULL CHECK (seq >= 0),
        kind TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        event_json TEXT NOT NULL,
        UNIQUE (session_id, seq)
      ) STRICT;
      CREATE INDEX IF NOT EXISTS context_events_session_seq
        ON context_events (session_id, seq);
      CREATE INDEX IF NOT EXISTS context_events_task_seq
        ON context_events (task_id, seq) WHERE task_id IS NOT NULL;
      CREATE TRIGGER IF NOT EXISTS context_events_no_update
        BEFORE UPDATE ON context_events
        BEGIN
          SELECT RAISE(ABORT, 'context_events is append-only');
        END;
      CREATE TRIGGER IF NOT EXISTS context_events_no_delete
        BEFORE DELETE ON context_events
        BEGIN
          SELECT RAISE(ABORT, 'context_events is append-only');
        END;
    `);
    this.secureDatabaseFiles();
  }

  append(input: unknown | readonly unknown[]): void {
    const candidates = (Array.isArray(input) ? input : [input]).map((event) =>
      ContextEngineEventSchema.parse(event),
    );
    if (candidates.length === 0) return;
    this.database.exec("BEGIN IMMEDIATE");
    try {
      for (const event of candidates) this.appendOne(event);
      this.database.exec("COMMIT");
      this.secureDatabaseFiles();
    } catch (error) {
      try {
        this.database.exec("ROLLBACK");
      } catch {
        // The failing statement may already have ended the transaction.
      }
      throw error;
    }
  }

  get(eventId: string): ContextEngineEvent | undefined {
    const row = this.database
      .prepare("SELECT event_json AS eventJson FROM context_events WHERE event_id = ?")
      .get(eventId) as { eventJson?: unknown } | undefined;
    return row ? parseSqliteEventJson(row.eventJson) : undefined;
  }

  scan(spec: ContextEventScanSpec = {}): ContextEngineEvent[] {
    const limit = spec.limit ?? Number.POSITIVE_INFINITY;
    if (!Number.isInteger(limit) && limit !== Number.POSITIVE_INFINITY) {
      throw new Error("Context event scan limit must be an integer.");
    }
    if (limit < 0) throw new Error("Context event scan limit must not be negative.");
    const predicates: string[] = [];
    const parameters: Array<string | number> = [];
    if (spec.sessionId !== undefined) {
      predicates.push("session_id = ?");
      parameters.push(spec.sessionId);
    }
    if (spec.taskId !== undefined) {
      predicates.push("task_id = ?");
      parameters.push(spec.taskId);
    }
    if (spec.afterSeq !== undefined) {
      predicates.push("seq > ?");
      parameters.push(spec.afterSeq);
    }
    if (spec.beforeSeq !== undefined) {
      predicates.push("seq < ?");
      parameters.push(spec.beforeSeq);
    }
    if (spec.kinds && spec.kinds.length > 0) {
      predicates.push(`kind IN (${spec.kinds.map(() => "?").join(", ")})`);
      parameters.push(...spec.kinds);
    }
    const where = predicates.length > 0 ? `WHERE ${predicates.join(" AND ")}` : "";
    const limitSql = limit === Number.POSITIVE_INFINITY ? "" : "LIMIT ?";
    if (limit !== Number.POSITIVE_INFINITY) parameters.push(limit);
    const rows = this.database
      .prepare(
        `SELECT event_json AS eventJson FROM context_events ${where} ORDER BY rowid ${limitSql}`,
      )
      .all(...parameters) as Array<{ eventJson?: unknown }>;
    return rows.map((row) => parseSqliteEventJson(row.eventJson));
  }

  snapshot(scope: { sessionId: string }): ContextHistorySnapshot {
    this.database.exec("BEGIN");
    try {
      const events = this.scan({ sessionId: scope.sessionId });
      const snapshot = createContextHistorySnapshot(events);
      this.database.exec("COMMIT");
      if (events.length > 0) return snapshot;
      return {
        ...snapshot,
        snapshotId: `snapshot_${sha256Hex(scope.sessionId)}`,
        sessionId: scope.sessionId,
      };
    } catch (error) {
      try {
        this.database.exec("ROLLBACK");
      } catch {
        // Keep the original read or validation error.
      }
      throw error;
    }
  }

  close(): void {
    this.database.close();
  }

  private appendOne(event: ContextEngineEvent): void {
    const existing = this.get(event.id);
    if (existing) {
      if (existing.contentHash === event.contentHash) return;
      throw new Error(`Context event id collision: ${event.id}.`);
    }
    const last = this.database
      .prepare("SELECT MAX(seq) AS lastSeq FROM context_events WHERE session_id = ?")
      .get(event.sessionId) as { lastSeq?: unknown } | undefined;
    const lastSeq = typeof last?.lastSeq === "number" ? last.lastSeq : -1;
    if (event.seq <= lastSeq) {
      throw new Error(
        `Context event sequence must increase for ${event.sessionId}: ${event.seq} after ${lastSeq}.`,
      );
    }
    for (const parentId of event.causalParents) {
      if (!this.get(parentId)) {
        throw new Error(`Context event ${event.id} has missing causal parent ${parentId}.`);
      }
    }
    this.database
      .prepare(
        `INSERT INTO context_events
          (event_id, session_id, task_id, seq, kind, content_hash, event_json)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        event.id,
        event.sessionId,
        event.taskId ?? null,
        event.seq,
        event.kind,
        event.contentHash,
        JSON.stringify(event),
      );
  }

  private secureDatabaseFiles(): void {
    for (const filePath of [
      this.databasePath,
      `${this.databasePath}-wal`,
      `${this.databasePath}-shm`,
    ]) {
      if (fs.existsSync(filePath)) fs.chmodSync(filePath, 0o600);
    }
  }
}

/** Standalone replay implementation; canonical Session and WorkItem stores remain authoritative. */
export class JsonlContextEventStore implements ContextEventStore {
  readonly rootDir: string;
  readonly eventLogPath: string;
  private readonly lockTimeoutMs: number;
  private readonly lockStaleMs: number;

  constructor(options: JsonlContextEventStoreOptions = {}) {
    this.rootDir = path.resolve(options.rootDir ?? DEFAULT_ROOT);
    this.eventLogPath = path.join(this.rootDir, EVENT_LOG_FILE);
    this.lockTimeoutMs = options.lockTimeoutMs ?? LOCK_TIMEOUT_MS;
    this.lockStaleMs = options.lockStaleMs ?? LOCK_STALE_MS;
  }

  append(input: unknown | readonly unknown[]): void {
    const candidates = (Array.isArray(input) ? input : [input]).map((event) =>
      ContextEngineEventSchema.parse(event),
    );
    if (candidates.length === 0) return;
    this.ensureRoot();
    this.withLock(() => {
      const existing = this.readEvents();
      const byId = new Map(existing.map((event) => [event.id, event]));
      const lastSeqBySession = new Map<string, number>();
      for (const event of existing) {
        lastSeqBySession.set(
          event.sessionId,
          Math.max(lastSeqBySession.get(event.sessionId) ?? -1, event.seq),
        );
      }
      const accepted: ContextEngineEvent[] = [];
      for (const event of candidates) {
        const collision = byId.get(event.id);
        if (collision) {
          if (collision.contentHash === event.contentHash) continue;
          throw new Error(`Context event id collision: ${event.id}.`);
        }
        const lastSeq = lastSeqBySession.get(event.sessionId) ?? -1;
        if (event.seq <= lastSeq) {
          throw new Error(
            `Context event sequence must increase for ${event.sessionId}: ${event.seq} after ${lastSeq}.`,
          );
        }
        for (const parentId of event.causalParents) {
          if (!byId.has(parentId)) {
            throw new Error(`Context event ${event.id} has missing causal parent ${parentId}.`);
          }
        }
        byId.set(event.id, event);
        lastSeqBySession.set(event.sessionId, event.seq);
        accepted.push(event);
      }
      if (accepted.length > 0) appendFileDurably(this.eventLogPath, jsonl(accepted));
    });
  }

  get(eventId: string): ContextEngineEvent | undefined {
    return this.readEvents().find((event) => event.id === eventId);
  }

  scan(spec: ContextEventScanSpec = {}): ContextEngineEvent[] {
    const limit = spec.limit ?? Number.POSITIVE_INFINITY;
    if (!Number.isInteger(limit) && limit !== Number.POSITIVE_INFINITY) {
      throw new Error("Context event scan limit must be an integer.");
    }
    if (limit < 0) throw new Error("Context event scan limit must not be negative.");
    const kinds = spec.kinds ? new Set(spec.kinds) : undefined;
    return this.readEvents()
      .filter(
        (event) =>
          (spec.sessionId === undefined || event.sessionId === spec.sessionId) &&
          (spec.taskId === undefined || event.taskId === spec.taskId) &&
          (spec.afterSeq === undefined || event.seq > spec.afterSeq) &&
          (spec.beforeSeq === undefined || event.seq < spec.beforeSeq) &&
          (kinds === undefined || kinds.has(event.kind)),
      )
      .slice(0, limit);
  }

  snapshot(scope: { sessionId: string }): ContextHistorySnapshot {
    const events = this.scan({ sessionId: scope.sessionId });
    const snapshot = createContextHistorySnapshot(events);
    if (events.length > 0) return snapshot;
    return {
      ...snapshot,
      snapshotId: `snapshot_${sha256Hex(scope.sessionId)}`,
      sessionId: scope.sessionId,
    };
  }

  private ensureRoot(): void {
    fs.mkdirSync(this.rootDir, { recursive: true, mode: 0o700 });
    fs.chmodSync(this.rootDir, 0o700);
  }

  private readEvents(): ContextEngineEvent[] {
    if (!fs.existsSync(this.eventLogPath)) return [];
    const content = fs.readFileSync(this.eventLogPath, "utf8");
    if (content.length > 0 && !content.endsWith("\n")) {
      throw new Error(`${EVENT_LOG_FILE} has a torn final record.`);
    }
    const events = content.split("\n").flatMap((line, index) => {
      if (!line) return [];
      try {
        return [ContextEngineEventSchema.parse(JSON.parse(line))];
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`${EVENT_LOG_FILE} line ${index + 1} is corrupt: ${message}`);
      }
    });
    validateStoredEvents(events);
    return events;
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
        if (!isErrorCode(error, "EEXIST")) throw error;
        if (removeStaleLock(lockPath, this.lockStaleMs)) continue;
        if (Date.now() - startedAt >= this.lockTimeoutMs) {
          throw new Error(`Timed out waiting for context event writer lock: ${lockPath}`);
        }
        sleepSync(LOCK_POLL_MS);
      }
    }

    try {
      return action();
    } finally {
      try {
        const lock = JSON.parse(fs.readFileSync(lockPath, "utf8")) as { token?: unknown };
        if (lock.token === token) fs.unlinkSync(lockPath);
      } catch {
        // A stale-lock recovery may already have removed it.
      }
    }
  }
}

export interface LocalContextArtifactStoreOptions {
  rootDir?: string;
}

export interface ContextExternalizationPolicy {
  thresholdBytes: number;
  maxSalientLines?: number;
  maxSalientLineChars?: number;
}

const EXTERNALIZABLE_KINDS = new Set(["tool_result", "patch", "test_result"]);

export function externalizeContextEventPayload(
  input: ContextEngineEventInput,
  artifactStore: ContextArtifactStore,
  policy: ContextExternalizationPolicy,
): ContextEngineEvent {
  const event = createContextEvent(input);
  if (!Number.isInteger(policy.thresholdBytes) || policy.thresholdBytes < 0) {
    throw new Error("Context externalization thresholdBytes must be a non-negative integer.");
  }
  if (event.artifactRef || !EXTERNALIZABLE_KINDS.has(event.kind) || event.payload === undefined) {
    return event;
  }
  const textual = typeof event.payload === "string";
  const encoded = Buffer.from(
    typeof event.payload === "string" ? event.payload : stableArtifactJson(event.payload),
    "utf8",
  );
  if (encoded.byteLength <= policy.thresholdBytes) return event;
  const artifactRef = artifactStore.put(encoded, {
    mediaType: textual ? "text/plain" : "application/json",
  });
  const { contentHash: _contentHash, ...content } = event;
  return createContextEvent({
    ...content,
    artifactRef,
    payload: {
      externalized: true,
      originalBytes: encoded.byteLength,
      salient: salientPayloadLines(
        event.payload,
        policy.maxSalientLines ?? 8,
        policy.maxSalientLineChars ?? 240,
      ),
    },
  });
}

export class LocalContextArtifactStore implements ContextArtifactStore {
  readonly rootDir: string;
  readonly artifactDir: string;

  constructor(options: LocalContextArtifactStoreOptions = {}) {
    this.rootDir = path.resolve(options.rootDir ?? DEFAULT_ROOT);
    this.artifactDir = path.join(this.rootDir, ARTIFACT_DIRECTORY);
  }

  put(content: Uint8Array, options: { mediaType?: string } = {}): ContextArtifactReference {
    const encoded = Buffer.from(content);
    const digest = sha256Bytes(encoded);
    const reference = ContextArtifactReferenceSchema.parse({
      uri: `artifact://sha256/${digest}`,
      contentHash: `sha256:${digest}`,
      sizeBytes: encoded.byteLength,
      mediaType: options.mediaType,
    });
    this.ensureDirectories();
    const filePath = this.pathFor(reference);
    if (!fs.existsSync(filePath)) {
      const temporaryPath = `${filePath}.${process.pid}.${Date.now()}.tmp`;
      writeNewFileDurably(temporaryPath, encoded);
      try {
        fs.renameSync(temporaryPath, filePath);
        fsyncDirectory(this.artifactDir);
      } catch (error) {
        if (!fs.existsSync(filePath)) throw error;
        try {
          fs.unlinkSync(temporaryPath);
        } catch {
          // Another writer won the content-addressed race.
        }
      }
    }
    return reference;
  }

  readRange(
    input: ContextArtifactReference,
    range: { startByte: number; endByte?: number },
  ): Buffer {
    const reference = ContextArtifactReferenceSchema.parse(input);
    const encoded = this.readVerified(reference);
    const start = range.startByte;
    const end = range.endByte ?? encoded.byteLength;
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start) {
      throw new Error("Artifact byte range must use non-negative ordered integers.");
    }
    if (end > encoded.byteLength) throw new Error("Artifact byte range exceeds content size.");
    return encoded.subarray(start, end);
  }

  preview(
    input: ContextArtifactReference,
    options: { maxBytes?: number } = {},
  ): { text: string; truncated: boolean } {
    const reference = ContextArtifactReferenceSchema.parse(input);
    const maxBytes = options.maxBytes ?? 4_096;
    if (!Number.isInteger(maxBytes) || maxBytes < 0) {
      throw new Error("Artifact preview maxBytes must be a non-negative integer.");
    }
    const encoded = this.readVerified(reference);
    return {
      text: encoded.subarray(0, maxBytes).toString("utf8"),
      truncated: encoded.byteLength > maxBytes,
    };
  }

  pathFor(input: ContextArtifactReference): string {
    const reference = ContextArtifactReferenceSchema.parse(input);
    const uriDigest = reference.uri.slice("artifact://sha256/".length);
    const hashDigest = reference.contentHash.slice("sha256:".length);
    if (uriDigest !== hashDigest) throw new Error("Artifact URI and content hash disagree.");
    return path.join(this.artifactDir, `${uriDigest}.blob`);
  }

  private readVerified(reference: ContextArtifactReference): Buffer {
    const encoded = fs.readFileSync(this.pathFor(reference));
    const digest = sha256Bytes(encoded);
    if (`sha256:${digest}` !== reference.contentHash) {
      throw new Error(`Context artifact checksum mismatch: ${reference.uri}`);
    }
    if (encoded.byteLength !== reference.sizeBytes) {
      throw new Error(`Context artifact size mismatch: ${reference.uri}`);
    }
    return encoded;
  }

  private ensureDirectories(): void {
    fs.mkdirSync(this.artifactDir, { recursive: true, mode: 0o700 });
    fs.chmodSync(this.rootDir, 0o700);
    fs.chmodSync(this.artifactDir, 0o700);
  }
}

function validateStoredEvents(events: ContextEngineEvent[]): void {
  const byId = new Map<string, ContextEngineEvent>();
  const lastSeqBySession = new Map<string, number>();
  for (const event of events) {
    const collision = byId.get(event.id);
    if (collision) throw new Error(`Context event id collision: ${event.id}.`);
    const lastSeq = lastSeqBySession.get(event.sessionId) ?? -1;
    if (event.seq <= lastSeq) {
      throw new Error(`Context event sequence regressed for ${event.sessionId}.`);
    }
    for (const parentId of event.causalParents) {
      if (!byId.has(parentId)) {
        throw new Error(`Context event ${event.id} has missing causal parent ${parentId}.`);
      }
    }
    byId.set(event.id, event);
    lastSeqBySession.set(event.sessionId, event.seq);
  }
}

function parseSqliteEventJson(value: unknown): ContextEngineEvent {
  if (typeof value !== "string") throw new Error("Context SQLite event row is malformed.");
  try {
    return ContextEngineEventSchema.parse(JSON.parse(value));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Context SQLite event row is corrupt: ${message}`);
  }
}

function stableArtifactJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableArtifactJson).join(",")}]`;
  return `{${Object.entries(value)
    .filter(([, child]) => child !== undefined)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, child]) => `${JSON.stringify(key)}:${stableArtifactJson(child)}`)
    .join(",")}}`;
}

function salientPayloadLines(value: unknown, maxLines: number, maxChars: number): string[] {
  const record =
    typeof value === "object" && value !== null && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : undefined;
  const declared = record?.salient;
  const candidates = Array.isArray(declared)
    ? declared.filter((item): item is string => typeof item === "string")
    : (typeof value === "string" ? value : stableArtifactJson(value))
        .split(/\r?\n/u)
        .filter((line) => line.trim());
  return candidates
    .slice(0, Math.max(0, maxLines))
    .map((line) => line.slice(0, Math.max(0, maxChars)));
}

function appendFileDurably(filePath: string, content: string): void {
  const existed = fs.existsSync(filePath);
  const descriptor = fs.openSync(filePath, "a", 0o600);
  try {
    fs.writeFileSync(descriptor, content, "utf8");
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

function jsonl(events: ContextEngineEvent[]): string {
  return `${events.map((event) => JSON.stringify(event)).join("\n")}\n`;
}

function sha256Hex(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function sha256Bytes(value: Uint8Array): string {
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
    const lock = JSON.parse(fs.readFileSync(lockPath, "utf8")) as { pid?: unknown };
    const pid = typeof lock.pid === "number" ? lock.pid : undefined;
    if (pid !== undefined && processIsAlive(pid)) return false;
    if (pid === undefined && age < staleMs) return false;
    fs.unlinkSync(lockPath);
    return true;
  } catch (error) {
    if (isErrorCode(error, "ENOENT")) return true;
    return false;
  }
}

function processIsAlive(pid: number): boolean {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return !isErrorCode(error, "ESRCH");
  }
}

function sleepSync(milliseconds: number): void {
  const signal = new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT));
  Atomics.wait(signal, 0, 0, milliseconds);
}

function isErrorCode(error: unknown, code: string): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    String((error as { code?: unknown }).code) === code
  );
}
