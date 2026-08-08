import { createHash, randomUUID } from "node:crypto";
import * as fs from "node:fs";
import { homedir } from "node:os";
import * as path from "node:path";
import { z } from "zod";
import { stableJson } from "./canonical-json.js";

const AUDIT_SCHEMA_VERSION = 1;
const GENESIS_HASH = "0".repeat(64);
const DEFAULT_AUDIT_FILE = path.join(homedir(), ".swarmx", "audit", "events.jsonl");
const DEFAULT_LOCK_TIMEOUT_MS = 5_000;
const DEFAULT_LOCK_STALE_MS = 30_000;
const LOCK_POLL_MS = 10;

export const AUDIT_METADATA_MAX_DEPTH = 4;
export const AUDIT_METADATA_MAX_ENTRIES = 64;
export const AUDIT_METADATA_MAX_ARRAY_ITEMS = 16;
export const AUDIT_METADATA_MAX_STRING_LENGTH = 160;
export const AUDIT_METADATA_MAX_KEY_LENGTH = 64;
export const AUDIT_INTEGRITY_LIMITATION =
  "The SHA-256 chain and local head checkpoint are not externally signed; an attacker who can rewrite both files can forge a new history.";

const REDACTED_VALUE = "[redacted]";
const OMITTED_VALUE = "[omitted]";

const SECRET_KEY_PATTERN =
  /(api[_-]?key|api[_-]?token|access[_-]?token|auth(?:orization)?|bearer|cookie|password|passwd|secret|credential|private[_-]?key|client[_-]?secret|session[_-]?token|refresh[_-]?token|smtp[_-]?password)/i;
const RAW_CONTENT_KEY_PATTERN =
  /(^|[_-])(prompt|response|conversation|message|content|body|source|sourcecode|source_code|code|command|terminal|terminaloutput|terminal_output|stdout|stderr|stack|stacktrace|stack_trace|traceback|runlog|run_log|processlog|process_log|workerlog|worker_log|raw|rawpayload|raw_payload|input|output|arguments?|result)([_-]|$)/i;
const SECRET_VALUE_PATTERNS = [
  /-----BEGIN [A-Z ]*PRIVATE KEY-----/i,
  /\bBearer\s+[A-Za-z0-9._~+/=-]+/i,
  /\b(?:sk|rk|pk|ghp|gho|ghu|ghs|github_pat|xoxb|xoxp|xoxa|xoxr)-[A-Za-z0-9_-]{4,}\b/i,
  /\bAIza[A-Za-z0-9_-]{12,}\b/,
  /\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret|credential|authorization)\s*[:=]\s*[^\s,;]+/i,
];
const RAW_VALUE_PATTERN =
  /(?:raw\s+(?:user\s+)?prompt|raw\s+response|terminal\s+output|source\s+code|stack\s+trace|traceback\s+\(most recent call last\))/i;

const AuditTokenSchema = z
  .string()
  .min(1)
  .max(160)
  .regex(/^[A-Za-z0-9][A-Za-z0-9_.:@/-]*$/);
const AuditIdentifierSchema = AuditTokenSchema.refine(
  (value) => !SECRET_VALUE_PATTERNS.some((pattern) => pattern.test(value)),
  { message: "Audit identifiers must not contain secret-bearing values." },
);
const AuditActionSchema = z
  .string()
  .min(1)
  .max(96)
  .regex(/^[a-z][a-z0-9_.-]*$/);
const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);

export const AuditCategorySchema = z.enum([
  "session",
  "task",
  "tool",
  "permission",
  "provider",
  "secret",
  "extension",
  "workspace",
  "telemetry",
  "system",
]);

export const AuditOutcomeSchema = z.enum([
  "attempted",
  "completed",
  "failed",
  "denied",
  "cancel_requested",
  "cancelled",
]);

export const AuditActorSchema = z
  .object({
    kind: z.enum(["user", "agent", "system", "process", "service"]),
    id: AuditIdentifierSchema.optional(),
  })
  .strict();

export const AuditTargetSchema = z
  .object({
    kind: AuditTokenSchema,
    id: AuditIdentifierSchema.optional(),
  })
  .strict();

export type AuditMetadataValue =
  | null
  | boolean
  | number
  | string
  | AuditMetadataValue[]
  | { [key: string]: AuditMetadataValue };

export const AuditMetadataValueSchema: z.ZodType<AuditMetadataValue> = z.lazy(() =>
  z.union([
    z.null(),
    z.boolean(),
    z.number().finite(),
    z.string().max(AUDIT_METADATA_MAX_STRING_LENGTH),
    z.array(AuditMetadataValueSchema).max(AUDIT_METADATA_MAX_ARRAY_ITEMS),
    z.record(z.string().min(1).max(AUDIT_METADATA_MAX_KEY_LENGTH), AuditMetadataValueSchema),
  ]),
);

export const AuditMetadataSchema = z
  .record(
    z
      .string()
      .min(1)
      .max(AUDIT_METADATA_MAX_KEY_LENGTH)
      .regex(/^[A-Za-z][A-Za-z0-9_.-]*$/),
    AuditMetadataValueSchema,
  )
  .superRefine((metadata, ctx) => {
    const sanitized = sanitizeAuditMetadata(metadata);
    if (stableJson(metadata) !== stableJson(sanitized)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Audit metadata must already be sanitized and within all depth/entry limits.",
      });
    }
  });

export const AuditInputSchema = z
  .object({
    category: AuditCategorySchema,
    action: AuditActionSchema,
    outcome: AuditOutcomeSchema.default("completed"),
    actor: AuditActorSchema.optional(),
    target: AuditTargetSchema.optional(),
    sessionId: AuditIdentifierSchema.optional(),
    taskId: AuditIdentifierSchema.optional(),
    requestId: AuditIdentifierSchema.optional(),
    metadata: z.preprocess(sanitizeAuditMetadata, AuditMetadataSchema).default({}),
  })
  .strict();

export const AuditEventSchema = z
  .object({
    schemaVersion: z.literal(AUDIT_SCHEMA_VERSION),
    sequence: z.number().int().positive(),
    eventId: z
      .string()
      .regex(/^aud_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i),
    timestamp: z.string().datetime({ offset: true }),
    category: AuditCategorySchema,
    action: AuditActionSchema,
    outcome: AuditOutcomeSchema,
    actor: AuditActorSchema.optional(),
    target: AuditTargetSchema.optional(),
    sessionId: AuditIdentifierSchema.optional(),
    taskId: AuditIdentifierSchema.optional(),
    requestId: AuditIdentifierSchema.optional(),
    metadata: AuditMetadataSchema,
    previousHash: Sha256Schema,
    eventHash: Sha256Schema,
  })
  .strict();

export const AuditQuerySchema = z
  .object({
    category: AuditCategorySchema.optional(),
    action: AuditActionSchema.optional(),
    outcome: AuditOutcomeSchema.optional(),
    actorId: AuditIdentifierSchema.optional(),
    targetId: AuditIdentifierSchema.optional(),
    sessionId: AuditIdentifierSchema.optional(),
    taskId: AuditIdentifierSchema.optional(),
    requestId: AuditIdentifierSchema.optional(),
    from: z.string().datetime({ offset: true }).optional(),
    to: z.string().datetime({ offset: true }).optional(),
    limit: z.number().int().min(1).max(10_000).optional(),
    reverse: z.boolean().default(false),
  })
  .strict()
  .superRefine((query, ctx) => {
    if (query.from && query.to && query.from > query.to) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["to"],
        message: "Audit query 'to' must not precede 'from'.",
      });
    }
  });

const AuditHeadCheckpointSchema = z
  .object({
    schemaVersion: z.literal(AUDIT_SCHEMA_VERSION),
    sequence: z.number().int().nonnegative(),
    eventHash: Sha256Schema,
    fileBytes: z.number().int().nonnegative(),
    updatedAt: z.string().datetime({ offset: true }),
  })
  .strict();

const AuditLockSchema = z
  .object({
    pid: z.number().int().positive(),
    token: z.string().uuid(),
    createdAt: z.string().datetime({ offset: true }),
  })
  .strict();

type AuditLock = z.infer<typeof AuditLockSchema>;

export type AuditCategory = z.infer<typeof AuditCategorySchema>;
export type AuditOutcome = z.infer<typeof AuditOutcomeSchema>;
export type AuditActor = z.infer<typeof AuditActorSchema>;
export type AuditTarget = z.infer<typeof AuditTargetSchema>;
export type AuditInput = z.input<typeof AuditInputSchema>;
export type ParsedAuditInput = z.output<typeof AuditInputSchema>;
export type AuditEvent = z.infer<typeof AuditEventSchema>;
export type AuditQuery = z.input<typeof AuditQuerySchema>;

export type AuditVerificationIssueCode =
  | "corrupt_record"
  | "torn_tail"
  | "missing_final_newline"
  | "invalid_sequence"
  | "invalid_previous_hash"
  | "invalid_event_hash"
  | "checkpoint_missing"
  | "checkpoint_corrupt"
  | "checkpoint_ahead"
  | "checkpoint_lagging"
  | "checkpoint_hash_mismatch"
  | "checkpoint_size_mismatch";

export interface AuditVerificationIssue {
  code: AuditVerificationIssueCode;
  message: string;
  line?: number;
  sequence?: number;
}

export interface AuditVerification {
  ok: boolean;
  eventCount: number;
  headSequence: number;
  headHash: string;
  checkpointStatus: "matched" | "lagging" | "missing" | "not_applicable" | "mismatch";
  issue?: AuditVerificationIssue;
}

export interface AuditRecoveryResult {
  recovered: boolean;
  discardedBytes: number;
  appendedFinalNewline: boolean;
  verification: AuditVerification;
}

export interface AuditStoreOptions {
  filePath?: string;
  now?: () => Date;
  lockTimeoutMs?: number;
  lockStaleMs?: number;
}

interface AuditHeadCheckpoint {
  schemaVersion: typeof AUDIT_SCHEMA_VERSION;
  sequence: number;
  eventHash: string;
  fileBytes: number;
  updatedAt: string;
}

interface ParsedAuditLog {
  events: AuditEvent[];
  fileBytes: number;
  eventEndBytes: number[];
  tornTail?: {
    validBytes: number;
    discardedBytes: number;
  };
  missingFinalNewline: boolean;
}

interface VerifiedAuditLog extends ParsedAuditLog {
  verification: AuditVerification;
}

export class AuditIntegrityError extends Error {
  readonly verification: AuditVerification;

  constructor(verification: AuditVerification) {
    super(verification.issue?.message ?? "Audit log integrity verification failed.");
    this.name = "AuditIntegrityError";
    this.verification = verification;
  }
}

export function sanitizeAuditMetadata(input: unknown): AuditMetadataValue {
  const state: SanitizeState = { entries: 0, seen: new WeakSet<object>() };
  return AuditMetadataValueSchema.parse(sanitizeMetadataValue(input, "", 0, state));
}

export function verifyAuditEventChain(eventsInput: readonly unknown[]): AuditVerification {
  const events: AuditEvent[] = [];
  for (const [index, input] of eventsInput.entries()) {
    const parsed = AuditEventSchema.safeParse(input);
    if (!parsed.success) {
      return verificationFailure(events, "corrupt_record", `Audit event ${index + 1} is invalid.`, {
        line: index + 1,
      });
    }
    events.push(parsed.data);
  }
  return verifyParsedEventChain(events);
}

export class AuditStore {
  readonly filePath: string;
  readonly checkpointPath: string;
  readonly lockPath: string;
  private readonly now: () => Date;
  private readonly lockTimeoutMs: number;
  private readonly lockStaleMs: number;

  constructor(options: AuditStoreOptions = {}) {
    this.filePath = path.resolve(options.filePath ?? DEFAULT_AUDIT_FILE);
    const extension = path.extname(this.filePath);
    const stem = path.basename(this.filePath, extension);
    this.checkpointPath = path.join(path.dirname(this.filePath), `${stem}.head.json`);
    this.lockPath = `${this.filePath}.lock`;
    this.now = options.now ?? (() => new Date());
    this.lockTimeoutMs = options.lockTimeoutMs ?? DEFAULT_LOCK_TIMEOUT_MS;
    this.lockStaleMs = options.lockStaleMs ?? DEFAULT_LOCK_STALE_MS;
    if (!Number.isFinite(this.lockTimeoutMs) || this.lockTimeoutMs < 0) {
      throw new Error("Audit lock timeout must be a non-negative finite number.");
    }
    if (!Number.isFinite(this.lockStaleMs) || this.lockStaleMs < 0) {
      throw new Error("Audit stale-lock age must be a non-negative finite number.");
    }
  }

  append(input: AuditInput): AuditEvent {
    const parsedInput = AuditInputSchema.parse(input);
    this.ensureStorageDirectory();
    return this.withWriterLock(() => {
      const current = this.readVerifiedLog();
      const previous = current.events.at(-1);
      const withoutHash = {
        schemaVersion: AUDIT_SCHEMA_VERSION,
        sequence: (previous?.sequence ?? 0) + 1,
        eventId: `aud_${randomUUID()}`,
        timestamp: this.now().toISOString(),
        category: parsedInput.category,
        action: parsedInput.action,
        outcome: parsedInput.outcome,
        ...(parsedInput.actor ? { actor: parsedInput.actor } : {}),
        ...(parsedInput.target ? { target: parsedInput.target } : {}),
        ...(parsedInput.sessionId ? { sessionId: parsedInput.sessionId } : {}),
        ...(parsedInput.taskId ? { taskId: parsedInput.taskId } : {}),
        ...(parsedInput.requestId ? { requestId: parsedInput.requestId } : {}),
        metadata: sanitizeAuditMetadata(parsedInput.metadata) as Record<string, AuditMetadataValue>,
        previousHash: previous?.eventHash ?? GENESIS_HASH,
      };
      const event = AuditEventSchema.parse({
        ...withoutHash,
        eventHash: sha256(stableJson(withoutHash)),
      });
      appendFileDurably(this.filePath, `${JSON.stringify(event)}\n`);
      fsyncDirectory(path.dirname(this.filePath));
      const fileBytes = fs.statSync(this.filePath).size;
      this.writeCheckpoint(event.sequence, event.eventHash, fileBytes);
      return event;
    });
  }

  query(queryInput: AuditQuery = {}): AuditEvent[] {
    const query = AuditQuerySchema.parse(queryInput);
    this.ensureStorageDirectory();
    return this.withWriterLock(() => {
      const events = filterAuditEvents(this.readVerifiedLog().events, query);
      return events.slice(0, query.limit ?? 100).map((event) => AuditEventSchema.parse(event));
    });
  }

  exportJsonl(queryInput: AuditQuery = {}): string {
    const query = AuditQuerySchema.parse(queryInput);
    this.ensureStorageDirectory();
    return this.withWriterLock(() => {
      const events = filterAuditEvents(this.readVerifiedLog().events, query);
      const selected = query.limit === undefined ? events : events.slice(0, query.limit);
      return selected.length > 0
        ? `${selected.map((event) => JSON.stringify(event)).join("\n")}\n`
        : "";
    });
  }

  verify(): AuditVerification {
    this.ensureStorageDirectory();
    return this.withWriterLock(() => this.verifyUnlocked());
  }

  recoverTornTail(): AuditRecoveryResult {
    this.ensureStorageDirectory();
    return this.withWriterLock(() => {
      let parsed: ParsedAuditLog;
      try {
        parsed = parseAuditLog(this.filePath);
      } catch (error) {
        const verification = verificationFromReadError(error);
        throw new AuditIntegrityError(verification);
      }
      const chain = verifyParsedEventChain(parsed.events);
      if (!chain.ok) throw new AuditIntegrityError(chain);

      const checkpoint = this.readCheckpoint();
      const checkpointVerification = verifyCheckpoint(parsed, chain, checkpoint);
      const recoverableMissingNewlineCheckpoint =
        parsed.missingFinalNewline &&
        checkpointVerification.issue?.code === "checkpoint_size_mismatch" &&
        checkpoint?.sequence === parsed.events.length &&
        checkpoint.fileBytes === parsed.fileBytes + 1;
      const recoverableLaggingCheckpoint =
        checkpointVerification.issue?.code === "checkpoint_lagging";
      if (
        !checkpointVerification.ok &&
        !recoverableMissingNewlineCheckpoint &&
        !recoverableLaggingCheckpoint
      ) {
        throw new AuditIntegrityError(checkpointVerification);
      }

      let recovered = false;
      let discardedBytes = 0;
      let appendedFinalNewline = false;
      if (parsed.tornTail) {
        fs.truncateSync(this.filePath, parsed.tornTail.validBytes);
        fsyncFile(this.filePath);
        discardedBytes = parsed.tornTail.discardedBytes;
        recovered = true;
      } else if (parsed.missingFinalNewline) {
        appendFileDurably(this.filePath, "\n");
        appendedFinalNewline = true;
        recovered = true;
      } else if (recoverableLaggingCheckpoint) {
        recovered = true;
      } else if (!checkpointVerification.ok) {
        throw new AuditIntegrityError(checkpointVerification);
      }

      if (recovered) {
        const last = parsed.events.at(-1);
        const fileBytes = fs.existsSync(this.filePath) ? fs.statSync(this.filePath).size : 0;
        this.writeCheckpoint(last?.sequence ?? 0, last?.eventHash ?? GENESIS_HASH, fileBytes);
      }
      const verification = this.verifyUnlocked();
      if (!verification.ok) throw new AuditIntegrityError(verification);
      return { recovered, discardedBytes, appendedFinalNewline, verification };
    });
  }

  private verifyUnlocked(): AuditVerification {
    let parsed: ParsedAuditLog;
    try {
      parsed = parseAuditLog(this.filePath);
    } catch (error) {
      return verificationFromReadError(error);
    }
    const chain = verifyParsedEventChain(parsed.events);
    if (!chain.ok) return chain;
    if (parsed.tornTail) {
      return {
        ...chain,
        ok: false,
        checkpointStatus: "mismatch",
        issue: {
          code: "torn_tail",
          message: `Audit log has an incomplete final record (${parsed.tornTail.discardedBytes} bytes).`,
        },
      };
    }
    if (parsed.missingFinalNewline) {
      return {
        ...chain,
        ok: false,
        checkpointStatus: "mismatch",
        issue: {
          code: "missing_final_newline",
          message: "Audit log has a valid final record without its JSONL newline.",
        },
      };
    }
    let checkpoint: AuditHeadCheckpoint | undefined;
    try {
      checkpoint = this.readCheckpoint();
    } catch (error) {
      return {
        ...chain,
        ok: false,
        checkpointStatus: "mismatch",
        issue: {
          code: "checkpoint_corrupt",
          message: errorMessage(error),
        },
      };
    }
    return verifyCheckpoint(parsed, chain, checkpoint);
  }

  private readVerifiedLog(): VerifiedAuditLog {
    const parsed = parseAuditLog(this.filePath);
    const chain = verifyParsedEventChain(parsed.events);
    if (!chain.ok) throw new AuditIntegrityError(chain);
    if (parsed.tornTail) {
      throw new AuditIntegrityError({
        ...chain,
        ok: false,
        checkpointStatus: "mismatch",
        issue: {
          code: "torn_tail",
          message: `Audit log has an incomplete final record; call recoverTornTail() before continuing: ${this.filePath}`,
        },
      });
    }
    if (parsed.missingFinalNewline) {
      throw new AuditIntegrityError({
        ...chain,
        ok: false,
        checkpointStatus: "mismatch",
        issue: {
          code: "missing_final_newline",
          message: `Audit log is missing its final JSONL newline; call recoverTornTail() before continuing: ${this.filePath}`,
        },
      });
    }
    const verification = verifyCheckpoint(parsed, chain, this.readCheckpoint());
    if (!verification.ok) throw new AuditIntegrityError(verification);
    return { ...parsed, verification };
  }

  private ensureStorageDirectory(): void {
    const directory = path.dirname(this.filePath);
    if (!fs.existsSync(directory)) fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
    const stat = fs.lstatSync(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink()) {
      throw new Error(`Audit storage path must be a real directory: ${directory}`);
    }
    fs.chmodSync(directory, 0o700);
    for (const filePath of [this.filePath, this.checkpointPath]) {
      if (!fs.existsSync(filePath)) continue;
      const fileStat = fs.lstatSync(filePath);
      if (!fileStat.isFile() || fileStat.isSymbolicLink()) {
        throw new Error(`Audit storage path must be a regular file: ${filePath}`);
      }
      fs.chmodSync(filePath, 0o600);
    }
  }

  private readCheckpoint(): AuditHeadCheckpoint | undefined {
    if (!fs.existsSync(this.checkpointPath)) return undefined;
    return AuditHeadCheckpointSchema.parse(
      JSON.parse(fs.readFileSync(this.checkpointPath, "utf8")),
    );
  }

  private writeCheckpoint(sequence: number, eventHash: string, fileBytes: number): void {
    const checkpoint = AuditHeadCheckpointSchema.parse({
      schemaVersion: AUDIT_SCHEMA_VERSION,
      sequence,
      eventHash,
      fileBytes,
      updatedAt: this.now().toISOString(),
    });
    writeFileAtomically(this.checkpointPath, `${JSON.stringify(checkpoint)}\n`);
  }

  private withWriterLock<T>(action: () => T): T {
    const token = randomUUID();
    const deadline = Date.now() + this.lockTimeoutMs;
    while (true) {
      try {
        const descriptor = fs.openSync(this.lockPath, "wx", 0o600);
        try {
          const lock = AuditLockSchema.parse({
            pid: process.pid,
            token,
            createdAt: new Date().toISOString(),
          });
          fs.writeFileSync(descriptor, JSON.stringify(lock), "utf8");
          fs.fsyncSync(descriptor);
        } finally {
          fs.closeSync(descriptor);
        }
        fsyncDirectory(path.dirname(this.lockPath));
        break;
      } catch (error) {
        if (!isFileExistsError(error)) throw error;
        if (removeStaleLock(this.lockPath, this.lockStaleMs)) continue;
        if (Date.now() >= deadline) {
          throw new Error(`Timed out waiting for Audit writer lock: ${this.lockPath}`);
        }
        sleepSync(LOCK_POLL_MS);
      }
    }

    try {
      return action();
    } finally {
      try {
        const lock = AuditLockSchema.parse(JSON.parse(fs.readFileSync(this.lockPath, "utf8")));
        if (lock.token === token) {
          fs.unlinkSync(this.lockPath);
          fsyncDirectory(path.dirname(this.lockPath));
        }
      } catch {
        // A dead writer's lock may already have been recovered.
      }
    }
  }
}

interface SanitizeState {
  entries: number;
  seen: WeakSet<object>;
}

function sanitizeMetadataValue(
  input: unknown,
  key: string,
  depth: number,
  state: SanitizeState,
): AuditMetadataValue {
  if (isSecretKey(key)) return REDACTED_VALUE;
  if (isRawContentKey(key)) return OMITTED_VALUE;
  if (input === null || typeof input === "boolean") return input;
  if (typeof input === "number") return Number.isFinite(input) ? input : OMITTED_VALUE;
  if (typeof input === "string") return sanitizeStringValue(input);
  if (typeof input !== "object") return OMITTED_VALUE;
  if (state.seen.has(input)) return OMITTED_VALUE;
  if (depth >= AUDIT_METADATA_MAX_DEPTH) return OMITTED_VALUE;
  state.seen.add(input);

  if (Array.isArray(input)) {
    const output: AuditMetadataValue[] = [];
    for (const item of input.slice(0, AUDIT_METADATA_MAX_ARRAY_ITEMS)) {
      if (state.entries >= AUDIT_METADATA_MAX_ENTRIES) break;
      state.entries += 1;
      output.push(sanitizeMetadataValue(item, key, depth + 1, state));
    }
    return output;
  }
  if (!isPlainRecord(input)) return OMITTED_VALUE;

  const output: Record<string, AuditMetadataValue> = {};
  for (const [rawKey, value] of Object.entries(input)) {
    if (state.entries >= AUDIT_METADATA_MAX_ENTRIES) break;
    state.entries += 1;
    const safeKey = uniqueMetadataKey(output, boundedMetadataKey(rawKey), rawKey);
    output[safeKey] = sanitizeMetadataValue(value, rawKey, depth + 1, state);
  }
  return output;
}

function sanitizeStringValue(value: string): string {
  const trimmed = value.trim();
  if (SECRET_VALUE_PATTERNS.some((pattern) => pattern.test(trimmed))) return REDACTED_VALUE;
  if (
    RAW_VALUE_PATTERN.test(trimmed) ||
    hasControlCharacters(trimmed) ||
    [...trimmed].length > AUDIT_METADATA_MAX_STRING_LENGTH
  ) {
    return OMITTED_VALUE;
  }
  return trimmed;
}

function hasControlCharacters(value: string): boolean {
  for (const character of value) {
    const codePoint = character.codePointAt(0) ?? 0;
    if (codePoint === 10 || codePoint === 13 || (codePoint < 32 && codePoint !== 9)) return true;
  }
  return false;
}

function boundedMetadataKey(key: string): string {
  const normalized = key
    .trim()
    .replace(/[^A-Za-z0-9_.-]+/g, "_")
    .replace(/^[_\-.]+|[_\-.]+$/g, "");
  const prefixed = /^[A-Za-z]/.test(normalized) ? normalized : `field_${normalized || "value"}`;
  if (prefixed.length <= AUDIT_METADATA_MAX_KEY_LENGTH) return prefixed;
  const suffix = sha256(key).slice(0, 12);
  return `${prefixed.slice(0, AUDIT_METADATA_MAX_KEY_LENGTH - suffix.length - 1)}_${suffix}`;
}

function uniqueMetadataKey(
  output: Readonly<Record<string, AuditMetadataValue>>,
  candidate: string,
  rawKey: string,
): string {
  if (!(candidate in output)) return candidate;
  const suffix = sha256(rawKey).slice(0, 8);
  const base = candidate.slice(0, AUDIT_METADATA_MAX_KEY_LENGTH - suffix.length - 1);
  let next = `${base}_${suffix}`;
  let counter = 1;
  while (next in output) {
    const counterSuffix = `_${counter}`;
    next = `${base.slice(0, AUDIT_METADATA_MAX_KEY_LENGTH - counterSuffix.length)}${counterSuffix}`;
    counter += 1;
  }
  return next;
}

function isSecretKey(key: string): boolean {
  return SECRET_KEY_PATTERN.test(key);
}

function isRawContentKey(key: string): boolean {
  return RAW_CONTENT_KEY_PATTERN.test(key);
}

function parseAuditLog(filePath: string): ParsedAuditLog {
  if (!fs.existsSync(filePath)) {
    return { events: [], fileBytes: 0, eventEndBytes: [], missingFinalNewline: false };
  }
  const source = fs.readFileSync(filePath);
  const events: AuditEvent[] = [];
  const eventEndBytes: number[] = [];
  let lineNumber = 0;
  let start = 0;
  for (let index = 0; index < source.length; index += 1) {
    if (source[index] !== 0x0a) continue;
    lineNumber += 1;
    const line = decodeUtf8(source.subarray(start, index), lineNumber, filePath);
    events.push(parseAuditLine(line, lineNumber, filePath));
    eventEndBytes.push(index + 1);
    start = index + 1;
  }

  if (start === source.length) {
    return {
      events,
      fileBytes: source.length,
      eventEndBytes,
      missingFinalNewline: false,
    };
  }

  lineNumber += 1;
  const finalBytes = source.subarray(start);
  try {
    const line = decodeUtf8(finalBytes, lineNumber, filePath);
    events.push(parseAuditLine(line, lineNumber, filePath));
    eventEndBytes.push(source.length);
    return {
      events,
      fileBytes: source.length,
      eventEndBytes,
      missingFinalNewline: true,
    };
  } catch {
    return {
      events,
      fileBytes: source.length,
      eventEndBytes,
      tornTail: {
        validBytes: start,
        discardedBytes: source.length - start,
      },
      missingFinalNewline: false,
    };
  }
}

function parseAuditLine(line: string, lineNumber: number, filePath: string): AuditEvent {
  if (!line.trim()) {
    throw new AuditRecordError(filePath, lineNumber, "Audit JSONL must not contain blank records.");
  }
  try {
    return AuditEventSchema.parse(JSON.parse(line));
  } catch (error) {
    throw new AuditRecordError(filePath, lineNumber, errorMessage(error));
  }
}

class AuditRecordError extends Error {
  readonly line: number;

  constructor(filePath: string, line: number, message: string) {
    super(`${path.basename(filePath)} line ${line} is corrupt: ${message}`);
    this.name = "AuditRecordError";
    this.line = line;
  }
}

function decodeUtf8(buffer: Buffer, line: number, filePath: string): string {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(buffer);
  } catch (error) {
    throw new AuditRecordError(filePath, line, `Invalid UTF-8: ${errorMessage(error)}`);
  }
}

function verifyParsedEventChain(events: readonly AuditEvent[]): AuditVerification {
  let previousHash = GENESIS_HASH;
  for (const [index, event] of events.entries()) {
    const expectedSequence = index + 1;
    if (event.sequence !== expectedSequence) {
      return verificationFailure(
        events.slice(0, index),
        "invalid_sequence",
        `Audit event sequence ${event.sequence} must be ${expectedSequence}.`,
        { line: index + 1, sequence: event.sequence },
      );
    }
    if (event.previousHash !== previousHash) {
      return verificationFailure(
        events.slice(0, index),
        "invalid_previous_hash",
        `Audit event ${event.sequence} does not reference the preceding event hash.`,
        { line: index + 1, sequence: event.sequence },
      );
    }
    const { eventHash: _eventHash, ...withoutHash } = event;
    const expectedHash = sha256(stableJson(withoutHash));
    if (event.eventHash !== expectedHash) {
      return verificationFailure(
        events.slice(0, index),
        "invalid_event_hash",
        `Audit event ${event.sequence} hash does not match its content.`,
        { line: index + 1, sequence: event.sequence },
      );
    }
    previousHash = event.eventHash;
  }
  return {
    ok: true,
    eventCount: events.length,
    headSequence: events.at(-1)?.sequence ?? 0,
    headHash: events.at(-1)?.eventHash ?? GENESIS_HASH,
    checkpointStatus: "not_applicable",
  };
}

function verifyCheckpoint(
  parsed: ParsedAuditLog,
  chain: AuditVerification,
  checkpoint: AuditHeadCheckpoint | undefined,
): AuditVerification {
  if (!checkpoint) {
    if (parsed.events.length === 0) {
      return { ...chain, checkpointStatus: "not_applicable" };
    }
    return {
      ...chain,
      ok: false,
      checkpointStatus: "missing",
      issue: {
        code: "checkpoint_missing",
        message: "Audit head checkpoint is missing for a non-empty log.",
      },
    };
  }
  if (checkpoint.sequence > parsed.events.length) {
    return {
      ...chain,
      ok: false,
      checkpointStatus: "mismatch",
      issue: {
        code: "checkpoint_ahead",
        message: `Audit checkpoint sequence ${checkpoint.sequence} is ahead of log sequence ${parsed.events.length}; the log tail may have been deleted.`,
      },
    };
  }
  const expectedHash =
    checkpoint.sequence === 0 ? GENESIS_HASH : parsed.events[checkpoint.sequence - 1]?.eventHash;
  if (checkpoint.eventHash !== expectedHash) {
    return {
      ...chain,
      ok: false,
      checkpointStatus: "mismatch",
      issue: {
        code: "checkpoint_hash_mismatch",
        message: `Audit checkpoint hash does not match log sequence ${checkpoint.sequence}.`,
      },
    };
  }
  const expectedBytes =
    checkpoint.sequence === 0 ? 0 : parsed.eventEndBytes[checkpoint.sequence - 1];
  if (checkpoint.fileBytes !== expectedBytes) {
    return {
      ...chain,
      ok: false,
      checkpointStatus: "mismatch",
      issue: {
        code: "checkpoint_size_mismatch",
        message: `Audit checkpoint byte offset ${checkpoint.fileBytes} does not match log sequence ${checkpoint.sequence}.`,
      },
    };
  }
  if (checkpoint.sequence < parsed.events.length) {
    return {
      ...chain,
      ok: false,
      checkpointStatus: "lagging",
      issue: {
        code: "checkpoint_lagging",
        message: `Audit log is ahead of checkpoint sequence ${checkpoint.sequence}; explicit recovery is required before accepting the tail.`,
      },
    };
  }
  return { ...chain, checkpointStatus: "matched" };
}

function verificationFailure(
  verifiedPrefix: readonly AuditEvent[],
  code: AuditVerificationIssueCode,
  message: string,
  location: Pick<AuditVerificationIssue, "line" | "sequence"> = {},
): AuditVerification {
  return {
    ok: false,
    eventCount: verifiedPrefix.length,
    headSequence: verifiedPrefix.at(-1)?.sequence ?? 0,
    headHash: verifiedPrefix.at(-1)?.eventHash ?? GENESIS_HASH,
    checkpointStatus: "not_applicable",
    issue: { code, message, ...location },
  };
}

function verificationFromReadError(error: unknown): AuditVerification {
  return {
    ok: false,
    eventCount: 0,
    headSequence: 0,
    headHash: GENESIS_HASH,
    checkpointStatus: "mismatch",
    issue: {
      code: "corrupt_record",
      message: errorMessage(error),
      ...(error instanceof AuditRecordError ? { line: error.line } : {}),
    },
  };
}

function filterAuditEvents(
  events: readonly AuditEvent[],
  query: z.output<typeof AuditQuerySchema>,
): AuditEvent[] {
  const filtered = events.filter((event) => {
    if (query.category && event.category !== query.category) return false;
    if (query.action && event.action !== query.action) return false;
    if (query.outcome && event.outcome !== query.outcome) return false;
    if (query.actorId && event.actor?.id !== query.actorId) return false;
    if (query.targetId && event.target?.id !== query.targetId) return false;
    if (query.sessionId && event.sessionId !== query.sessionId) return false;
    if (query.taskId && event.taskId !== query.taskId) return false;
    if (query.requestId && event.requestId !== query.requestId) return false;
    if (query.from && event.timestamp < query.from) return false;
    if (query.to && event.timestamp > query.to) return false;
    return true;
  });
  return query.reverse ? filtered.reverse() : filtered;
}

function appendFileDurably(filePath: string, content: string): void {
  rejectSymlink(filePath);
  const descriptor = fs.openSync(filePath, "a", 0o600);
  try {
    fs.writeFileSync(descriptor, content, "utf8");
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
  fs.chmodSync(filePath, 0o600);
}

function writeFileAtomically(filePath: string, content: string): void {
  rejectSymlink(filePath);
  const temporaryPath = `${filePath}.tmp-${process.pid}-${randomUUID()}`;
  let descriptor: number | undefined;
  try {
    descriptor = fs.openSync(temporaryPath, "wx", 0o600);
    fs.writeFileSync(descriptor, content, "utf8");
    fs.fsyncSync(descriptor);
    fs.closeSync(descriptor);
    descriptor = undefined;
    fs.renameSync(temporaryPath, filePath);
    fs.chmodSync(filePath, 0o600);
    fsyncDirectory(path.dirname(filePath));
  } finally {
    if (descriptor !== undefined) fs.closeSync(descriptor);
    if (fs.existsSync(temporaryPath)) fs.unlinkSync(temporaryPath);
  }
}

function fsyncFile(filePath: string): void {
  if (!fs.existsSync(filePath)) return;
  const descriptor = fs.openSync(filePath, "r+");
  try {
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
  fs.chmodSync(filePath, 0o600);
  fsyncDirectory(path.dirname(filePath));
}

function fsyncDirectory(directory: string): void {
  let descriptor: number | undefined;
  try {
    descriptor = fs.openSync(directory, "r");
    fs.fsyncSync(descriptor);
  } catch (error) {
    if (!isUnsupportedDirectoryFsyncError(error)) throw error;
  } finally {
    if (descriptor !== undefined) fs.closeSync(descriptor);
  }
}

function rejectSymlink(filePath: string): void {
  if (!fs.existsSync(filePath)) return;
  const stat = fs.lstatSync(filePath);
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error(`Audit path must be a regular file: ${filePath}`);
  }
}

export function removeStaleLock(lockPath: string, staleMs: number): boolean {
  try {
    const stat = fs.lstatSync(lockPath);
    if (stat.isSymbolicLink() || !stat.isFile()) {
      throw new Error(`Audit lock path must be a regular file: ${lockPath}`);
    }
    const age = Date.now() - stat.mtimeMs;
    let lock: { success: boolean; data?: AuditLock };
    try {
      lock = AuditLockSchema.safeParse(JSON.parse(fs.readFileSync(lockPath, "utf8")));
    } catch {
      // A concurrent writer may be mid-write; a half-written lock is not
      // stale evidence. Only a fully unreadable, old lock counts as stale.
      lock = { success: false };
    }
    if (lock.success && lock.data && processIsAlive(lock.data.pid)) return false;
    if (!lock.success && age < staleMs) return false;
    fs.unlinkSync(lockPath);
    fsyncDirectory(path.dirname(lockPath));
    return true;
  } catch (error) {
    if (isFileNotFoundError(error)) return true;
    throw error;
  }
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return !isNoSuchProcessError(error);
  }
}

function sleepSync(milliseconds: number): void {
  const signal = new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT));
  Atomics.wait(signal, 0, 0, milliseconds);
}

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function isPlainRecord(value: object): value is Record<string, unknown> {
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isFileExistsError(error: unknown): boolean {
  return errorCode(error) === "EEXIST";
}

function isFileNotFoundError(error: unknown): boolean {
  return errorCode(error) === "ENOENT";
}

function isNoSuchProcessError(error: unknown): boolean {
  return errorCode(error) === "ESRCH";
}

function isUnsupportedDirectoryFsyncError(error: unknown): boolean {
  return ["EINVAL", "ENOTSUP", "EBADF", "EISDIR"].includes(errorCode(error) ?? "");
}

function errorCode(error: unknown): string | undefined {
  return typeof error === "object" && error !== null && "code" in error
    ? String((error as { code?: unknown }).code)
    : undefined;
}
