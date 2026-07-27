import * as fs from "node:fs";
import { homedir } from "node:os";
import * as path from "node:path";
import { v4 as uuidv4 } from "uuid";
import { ZodError, z } from "zod";
import {
  type MessageChunk,
  MessageChunkSchema,
  type SessionData,
  SessionDataSchema,
  type SessionPermissionMode,
  type TransientSessionContextChip,
  type TransientSessionData,
  TransientSessionDataSchema,
} from "./types.js";

const SESSION_SCHEMA_VERSION = 1;
const DEFAULT_SESSIONS_DIR = path.join(homedir(), ".swarmx", "sessions");
const SESSION_INDEX_FILE = "sessions.index.jsonl";
const LEGACY_BACKUP_DIR = "legacy-json-backups";
const SESSION_LOCK_TIMEOUT_MS = 5_000;
const SESSION_LOCK_STALE_MS = 30_000;
const MIGRATION_ERROR_LIMIT = 600;

const LegacyDesktopMessageSchema = z.object({
  is_user: z.boolean(),
  content: z.string(),
  kind: z.enum(["Message", "Thinking", "ToolCall", "ToolResult"]).default("Message"),
  tool_name: z.string().optional(),
  tool_result: z.string().optional(),
  duration_ms: z.number().int().nonnegative().optional(),
});

const LegacyDesktopSessionSchema = z.object({
  id: z.string(),
  agent_instance_id: z.string().default(""),
  working_dir: z.string(),
  model_override: z.string().optional(),
  acp_session_id: z.string().nullish(),
  agent_runtime: z
    .enum(["claude_agent_acp", "codex_acp", "open_code", "hermes", "open_claw"])
    .nullish(),
  messages: z.array(LegacyDesktopMessageSchema),
  created_at: z.string(),
  updated_at: z.string(),
  title: z.string().nullish(),
  pinned: z.boolean().default(false),
  archived: z.boolean().default(false),
});

const LegacyAgentInstanceSchema = z.object({
  id: z.string(),
  label: z.string(),
  harness: z.string(),
  model: z.string().optional(),
});

type LegacyDesktopSession = z.infer<typeof LegacyDesktopSessionSchema>;
type LegacyAgentDescriptor = Pick<SessionData, "agentName" | "harness" | "model">;

type SessionMetadata = Omit<SessionData, "messages">;
type SessionSourceFormat = "json" | "jsonl";

interface SessionCreatedEvent {
  schemaVersion: typeof SESSION_SCHEMA_VERSION;
  type: "session_created";
  timestamp: string;
  session: SessionData;
}

interface MessagesAppendedEvent {
  schemaVersion: typeof SESSION_SCHEMA_VERSION;
  type: "messages_appended";
  timestamp: string;
  messages: MessageChunk[];
}

interface MessagesReplacedEvent {
  schemaVersion: typeof SESSION_SCHEMA_VERSION;
  type: "messages_replaced";
  timestamp: string;
  messages: MessageChunk[];
  reason?: "edit_last_user_message";
  replacedFromIndex?: number;
  replacedMessageCount?: number;
}

interface SessionUpdatedEvent {
  schemaVersion: typeof SESSION_SCHEMA_VERSION;
  type: "session_updated";
  timestamp: string;
  session: SessionMetadata;
}

type SessionEvent =
  | SessionCreatedEvent
  | MessagesAppendedEvent
  | MessagesReplacedEvent
  | SessionUpdatedEvent;

export interface SessionSummary extends SessionMetadata {
  messageCount: number;
}

interface SessionIndexEvent {
  schemaVersion: typeof SESSION_SCHEMA_VERSION;
  type: "session_indexed" | "session_deleted";
  timestamp: string;
  sessionId: string;
  sourceBytes?: number;
  sourceMtimeMs?: number;
  sourceFormat?: SessionSourceFormat;
  summary?: SessionSummary;
}

interface SessionCacheEntry {
  bytes: number;
  mtimeMs: number;
  session: SessionData;
  tornTail: boolean;
}

interface ParsedJsonl<T> {
  records: T[];
  tornTail: boolean;
}

export interface SessionMigrationOptions {
  sessionsDir?: string;
  dryRun?: boolean;
  backupDir?: string;
}

export type SessionMigrationStatus = "planned" | "migrated" | "skipped" | "failed";

export interface SessionMigrationEntry {
  id: string;
  sourcePath: string;
  targetPath: string;
  status: SessionMigrationStatus;
  backupPath?: string;
  error?: string;
}

export interface SessionMigrationResult {
  sessionsDir: string;
  backupDir?: string;
  discovered: number;
  planned: number;
  migrated: number;
  skipped: number;
  failed: number;
  sessions: SessionMigrationEntry[];
}

const sessionCache = new Map<string, SessionCacheEntry>();

function configuredSessionsDir(): string {
  return path.resolve(process.env.SWARMX_SESSIONS_DIR ?? DEFAULT_SESSIONS_DIR);
}

function ensureSessionsDir(sessionsDir = configuredSessionsDir()): string {
  if (!fs.existsSync(sessionsDir)) {
    fs.mkdirSync(sessionsDir, { recursive: true, mode: 0o700 });
  }
  return sessionsDir;
}

function sessionPaths(
  id: string,
  sessionsDir = configuredSessionsDir(),
): {
  json: string;
  jsonl: string;
} {
  if (!id || path.basename(id) !== id || id === "." || id === "..") {
    throw new Error("Session id must be a non-empty file-safe value.");
  }
  return {
    json: path.join(sessionsDir, `${id}.json`),
    jsonl: path.join(sessionsDir, `${id}.jsonl`),
  };
}

export interface SessionProjectContext {
  projectId?: string;
  cwd?: string;
  permissionMode?: SessionPermissionMode;
}

export interface EditSessionUserMessageInput {
  id: string;
  messageIndex: number;
  expectedMessages: MessageChunk[];
  content: string;
}

export interface ForkSessionInput {
  id: string;
  throughMessageIndex: number;
  expectedMessages: MessageChunk[];
}

export interface CreateTransientSessionForkInput {
  id: string;
  throughMessageIndex: number;
  expectedMessages: MessageChunk[];
  title?: string;
}

export interface PromoteTransientSessionForkInput {
  transient: TransientSessionData;
  title?: string;
}

export function createSession(
  agentName: string,
  harness: string,
  model?: string,
  project: SessionProjectContext = {},
): SessionData {
  ensureSessionsDir();
  const id = uuidv4();
  const now = new Date().toISOString();
  return SessionDataSchema.parse({
    id,
    title: "New Session",
    agentName,
    harness,
    model,
    ...(project.projectId ? { projectId: project.projectId } : {}),
    ...(project.cwd ? { cwd: project.cwd } : {}),
    ...(project.permissionMode ? { permissionMode: project.permissionMode } : {}),
    messages: [],
    createdAt: now,
    updatedAt: now,
  });
}

export function saveSession(session: SessionData): void {
  const sessionsDir = ensureSessionsDir();
  const paths = sessionPaths(session.id, sessionsDir);
  const now = new Date().toISOString();
  session.updatedAt = now;
  const next = SessionDataSchema.parse(session);

  withSessionLock(paths.jsonl, () => saveSessionLocked(next, paths, sessionsDir, now));
}

function saveSessionLocked(
  next: SessionData,
  paths: ReturnType<typeof sessionPaths>,
  sessionsDir: string,
  now: string,
): void {
  if (!fs.existsSync(paths.jsonl) && fs.existsSync(paths.json)) {
    const migration = migrateLegacySessionFile(paths.json, sessionsDir);
    if (migration.status === "failed") {
      throw new Error(migration.error ?? `Failed to migrate legacy Session ${next.id}.`);
    }
  }

  if (!fs.existsSync(paths.jsonl)) {
    createSessionLog(next, paths.jsonl);
    indexSession(next, paths.jsonl, "jsonl", sessionsDir);
    return;
  }

  const current = readSessionLog(paths.jsonl, { rejectTornTail: true });
  const events: SessionEvent[] = [];
  let effective = next;
  const metadataChanged = !sameValue(
    metadataWithoutUpdatedAt(sessionMetadata(current)),
    metadataWithoutUpdatedAt(sessionMetadata(next)),
  );

  if (isMessagePrefix(current.messages, next.messages)) {
    const appended = next.messages.slice(current.messages.length);
    if (appended.length > 0) {
      events.push(messagesAppendedEvent(appended, now));
    }
  } else if (isMessagePrefix(next.messages, current.messages)) {
    effective = SessionDataSchema.parse({ ...next, messages: current.messages });
  } else if (!sameValue(current.messages, next.messages)) {
    effective = SessionDataSchema.parse({
      ...next,
      externalAcpSession: undefined,
    });
    events.push(sessionUpdatedEvent(effective, now));
    events.push(messagesReplacedEvent(effective.messages, now));
  }

  if (
    (metadataChanged && !events.some((event) => event.type === "session_updated")) ||
    events.length === 0
  ) {
    events.push(sessionUpdatedEvent(effective, now));
  }

  appendSessionEvents(paths.jsonl, events);
  cacheSession(paths.jsonl, effective);
  indexSession(effective, paths.jsonl, "jsonl", sessionsDir);
}

export function loadSession(id: string): SessionData | null {
  return loadSessionFromDirectory(id, ensureSessionsDir());
}

export interface ListSessionsOptions {
  includeArchived?: boolean;
  sessionsDir?: string;
}

export function listSessionSummaries(options: ListSessionsOptions = {}): SessionSummary[] {
  const sessionsDir = ensureSessionsDir(
    path.resolve(options.sessionsDir ?? configuredSessionsDir()),
  );
  const index = readOrRebuildSessionIndex(sessionsDir);
  const summaries: SessionSummary[] = [];

  for (const source of discoverSessionSources(sessionsDir)) {
    const indexed = index.get(source.id);
    const sourceStat = fs.statSync(source.path);
    const sourceBytes = sourceStat.size;
    let summary = indexed?.summary;
    if (
      !indexed ||
      !summary ||
      indexed.sourceBytes !== sourceBytes ||
      indexed.sourceMtimeMs !== sourceStat.mtimeMs ||
      indexed.sourceFormat !== source.format
    ) {
      try {
        const session =
          source.format === "jsonl" ? readSessionLog(source.path) : readLegacySession(source.path);
        summary = sessionSummary(session);
        appendSessionIndexEvent(
          {
            schemaVersion: SESSION_SCHEMA_VERSION,
            type: "session_indexed",
            timestamp: new Date().toISOString(),
            sessionId: session.id,
            sourceBytes,
            sourceMtimeMs: sourceStat.mtimeMs,
            sourceFormat: source.format,
            summary,
          },
          sessionsDir,
        );
      } catch {
        continue;
      }
    }
    if (options.includeArchived || !summary.archivedAt) summaries.push(summary);
  }

  return sortSessions(summaries);
}

export function listSessions(options: ListSessionsOptions = {}): SessionData[] {
  const sessionsDir = path.resolve(options.sessionsDir ?? configuredSessionsDir());
  return listSessionSummaries({ includeArchived: true, sessionsDir })
    .flatMap((summary) => {
      const session = loadSessionFromDirectory(summary.id, sessionsDir);
      return session && (options.includeArchived || !session.archivedAt) ? [session] : [];
    })
    .sort(compareSessions);
}

function loadSessionFromDirectory(id: string, sessionsDir: string): SessionData | null {
  let paths: ReturnType<typeof sessionPaths>;
  try {
    paths = sessionPaths(id, sessionsDir);
  } catch {
    return null;
  }
  try {
    if (fs.existsSync(paths.jsonl)) return readSessionLog(paths.jsonl);
    if (fs.existsSync(paths.json)) return readLegacySession(paths.json);
  } catch {
    return null;
  }
  return null;
}

export function archiveProjectSessions(project: SessionProjectContext): number {
  if (!project.projectId && !project.cwd) return 0;
  const archivedAt = new Date().toISOString();
  let archived = 0;
  for (const summary of listSessionSummaries({ includeArchived: true })) {
    if (summary.archivedAt || !belongsToProject(summary, project)) continue;
    const session = loadSession(summary.id);
    if (!session) continue;
    session.archivedAt = archivedAt;
    saveSession(session);
    archived += 1;
  }
  return archived;
}

export function archiveSession(id: string): SessionData | null {
  const session = loadSession(id);
  if (!session) return null;
  if (!session.archivedAt) {
    session.archivedAt = new Date().toISOString();
    saveSession(session);
  }
  return session;
}

export function deleteSession(id: string): boolean {
  const sessionsDir = ensureSessionsDir();
  let paths: ReturnType<typeof sessionPaths>;
  try {
    paths = sessionPaths(id, sessionsDir);
  } catch {
    return false;
  }
  return withSessionLock(paths.jsonl, () => {
    let deleted = false;
    for (const filePath of [paths.jsonl, paths.json]) {
      if (!fs.existsSync(filePath)) continue;
      fs.unlinkSync(filePath);
      sessionCache.delete(filePath);
      deleted = true;
    }
    if (deleted) {
      appendSessionIndexEvent(
        {
          schemaVersion: SESSION_SCHEMA_VERSION,
          type: "session_deleted",
          timestamp: new Date().toISOString(),
          sessionId: id,
        },
        sessionsDir,
      );
    }
    return deleted;
  });
}

export function updateSessionTitle(id: string, title: string): boolean {
  const session = loadSession(id);
  if (!session) return false;
  session.title = title;
  saveSession(session);
  return true;
}

export function setSessionPinned(id: string, pinned: boolean): SessionData | null {
  const session = loadSession(id);
  if (!session) return null;
  session.pinned = pinned;
  saveSession(session);
  return session;
}

export function appendMessages(id: string, messages: MessageChunk[]): boolean {
  const sessionsDir = ensureSessionsDir();
  const paths = sessionPaths(id, sessionsDir);
  return withSessionLock(paths.jsonl, () => {
    if (!fs.existsSync(paths.jsonl) && fs.existsSync(paths.json)) {
      const migration = migrateLegacySessionFile(paths.json, sessionsDir);
      if (migration.status === "failed") {
        throw new Error(migration.error ?? `Failed to migrate legacy Session ${id}.`);
      }
    }
    if (!fs.existsSync(paths.jsonl)) return false;

    const current = readSessionLog(paths.jsonl, { rejectTornTail: true });
    const now = new Date().toISOString();
    const parsedMessages = messages.map((message) =>
      MessageChunkSchema.parse({
        ...message,
        createdAt: message.createdAt ?? now,
      }),
    );
    const next = SessionDataSchema.parse({
      ...current,
      messages: [...current.messages, ...parsedMessages],
      updatedAt: now,
    });
    const event =
      parsedMessages.length > 0
        ? messagesAppendedEvent(parsedMessages, now)
        : sessionUpdatedEvent(next, now);
    appendSessionEvents(paths.jsonl, [event]);
    cacheSession(paths.jsonl, next);
    indexSession(next, paths.jsonl, "jsonl", sessionsDir);
    return true;
  });
}

export function editSessionUserMessage(input: EditSessionUserMessageInput): SessionData | null {
  const sessionsDir = ensureSessionsDir();
  const paths = sessionPaths(input.id, sessionsDir);
  return withSessionLock(paths.jsonl, () => {
    if (!fs.existsSync(paths.jsonl) && fs.existsSync(paths.json)) {
      const migration = migrateLegacySessionFile(paths.json, sessionsDir);
      if (migration.status === "failed") {
        throw new Error(migration.error ?? `Failed to migrate legacy Session ${input.id}.`);
      }
    }
    if (!fs.existsSync(paths.jsonl)) return null;

    const current = readSessionLog(paths.jsonl, { rejectTornTail: true });
    const expectedMessages = input.expectedMessages.map((message) =>
      MessageChunkSchema.parse(message),
    );
    if (!sameValue(current.messages, expectedMessages)) {
      throw new Error("Session history changed before the message edit could be saved.");
    }
    if (
      !Number.isInteger(input.messageIndex) ||
      input.messageIndex < 0 ||
      input.messageIndex >= current.messages.length
    ) {
      throw new Error("Edited message index is outside the current Session history.");
    }

    const message = current.messages[input.messageIndex];
    if (message?.role !== "user" || message.kind !== "message") {
      throw new Error("Only user messages can be edited.");
    }
    if (input.messageIndex !== lastUserMessageIndex(current.messages)) {
      throw new Error("Only the latest user message can be edited.");
    }
    const content = input.content.trim();
    if (!content) throw new Error("Edited message content cannot be empty.");

    const now = new Date().toISOString();
    const replacedMessageCount = current.messages.length - input.messageIndex;
    const next = SessionDataSchema.parse({
      ...current,
      externalAcpSession: undefined,
      messages: [
        ...current.messages.slice(0, input.messageIndex),
        { ...message, content, createdAt: message.createdAt ?? now },
      ],
      updatedAt: now,
    });
    appendSessionEvents(paths.jsonl, [
      sessionUpdatedEvent(next, now),
      messagesReplacedEvent(next.messages, now, {
        reason: "edit_last_user_message",
        replacedFromIndex: input.messageIndex,
        replacedMessageCount,
      }),
    ]);
    cacheSession(paths.jsonl, next);
    indexSession(next, paths.jsonl, "jsonl", sessionsDir);
    return next;
  });
}

export function forkSession(input: ForkSessionInput): SessionData | null {
  const sessionsDir = ensureSessionsDir();
  const paths = sessionPaths(input.id, sessionsDir);
  return withSessionLock(paths.jsonl, () => {
    if (!fs.existsSync(paths.jsonl) && fs.existsSync(paths.json)) {
      const migration = migrateLegacySessionFile(paths.json, sessionsDir);
      if (migration.status === "failed") {
        throw new Error(migration.error ?? `Failed to migrate legacy Session ${input.id}.`);
      }
    }
    if (!fs.existsSync(paths.jsonl)) return null;

    const current = readSessionLog(paths.jsonl, { rejectTornTail: true });
    const expectedMessages = input.expectedMessages.map((message) =>
      MessageChunkSchema.parse(message),
    );
    if (!sameValue(current.messages, expectedMessages)) {
      throw new Error("Session history changed before the new chat could be created.");
    }
    if (
      !Number.isInteger(input.throughMessageIndex) ||
      input.throughMessageIndex < 0 ||
      input.throughMessageIndex >= current.messages.length
    ) {
      throw new Error("Fork message index is outside the current Session history.");
    }
    const checkpoint = current.messages[input.throughMessageIndex];
    if (checkpoint?.role !== "assistant" || checkpoint.kind !== "message") {
      throw new Error("A new chat can continue only from a completed assistant message.");
    }

    const now = new Date().toISOString();
    const forked = SessionDataSchema.parse({
      id: uuidv4(),
      title: continuedSessionTitle(current.title),
      forkedFrom: {
        sessionId: current.id,
        messageIndex: input.throughMessageIndex,
        createdAt: now,
      },
      ...(current.projectId ? { projectId: current.projectId } : {}),
      ...(current.cwd ? { cwd: current.cwd } : {}),
      agentName: current.agentName,
      harness: current.harness,
      ...(current.model ? { model: current.model } : {}),
      permissionMode: current.permissionMode,
      pinned: false,
      messages: current.messages.slice(0, input.throughMessageIndex + 1),
      createdAt: now,
      updatedAt: now,
    });
    saveSession(forked);
    return forked;
  });
}

export function createTransientSessionFork(
  input: CreateTransientSessionForkInput,
): TransientSessionData | null {
  const sessionsDir = ensureSessionsDir();
  const paths = sessionPaths(input.id, sessionsDir);
  return withSessionLock(paths.jsonl, () => {
    if (!fs.existsSync(paths.jsonl) && fs.existsSync(paths.json)) {
      const migration = migrateLegacySessionFile(paths.json, sessionsDir);
      if (migration.status === "failed") {
        throw new Error(migration.error ?? `Failed to migrate legacy Session ${input.id}.`);
      }
    }
    if (!fs.existsSync(paths.jsonl)) return null;

    const current = readSessionLog(paths.jsonl, { rejectTornTail: true });
    const expectedMessages = input.expectedMessages.map((message) =>
      MessageChunkSchema.parse(message),
    );
    if (!sameValue(current.messages, expectedMessages)) {
      throw new Error("Session history changed before the side chat could be created.");
    }
    if (
      !Number.isInteger(input.throughMessageIndex) ||
      input.throughMessageIndex < 0 ||
      input.throughMessageIndex >= current.messages.length
    ) {
      throw new Error("Side chat anchor is outside the current Session history.");
    }

    const now = new Date().toISOString();
    const anchorMessages = current.messages.slice(0, input.throughMessageIndex + 1);
    return TransientSessionDataSchema.parse({
      id: uuidv4(),
      parentSessionId: current.id,
      title: input.title?.trim() || nextSideChatTitle(1),
      anchor: {
        parentSessionId: current.id,
        messageIndex: input.throughMessageIndex,
        messageCount: anchorMessages.length,
        createdAt: now,
      },
      anchorMessages,
      messages: [],
      draft: "",
      attachments: [],
      contextChips: [],
      agentName: current.agentName,
      harness: current.harness,
      ...(current.model ? { model: current.model } : {}),
      ...(current.projectId ? { projectId: current.projectId } : {}),
      ...(current.cwd ? { cwd: current.cwd } : {}),
      permissionMode: current.permissionMode,
      runState: "idle",
      unread: false,
      createdAt: now,
      updatedAt: now,
    });
  });
}

export function appendTransientSessionMessages(
  transient: TransientSessionData,
  messages: MessageChunk[],
  options: {
    contextChips?: TransientSessionContextChip[];
    unread?: boolean;
  } = {},
): TransientSessionData {
  const current = TransientSessionDataSchema.parse(transient);
  const now = new Date().toISOString();
  const parsedMessages = messages.map((message) =>
    MessageChunkSchema.parse({
      ...message,
      createdAt: message.createdAt ?? now,
    }),
  );
  const contextChips = (options.contextChips ?? []).map((chip) => ({
    id: chip.id,
    text: chip.text.trim(),
    createdAt: chip.createdAt,
  }));
  const firstUserMessageIndex = parsedMessages.findIndex(
    (message) => message.role === "user" && message.kind === "message",
  );
  if (contextChips.length > 0 && firstUserMessageIndex >= 0) {
    const message = parsedMessages[firstUserMessageIndex];
    if (message) {
      parsedMessages[firstUserMessageIndex] = MessageChunkSchema.parse({
        ...message,
        structuredContent: {
          ...recordValue(message.structuredContent),
          sideChatContext: contextChips,
        },
      });
    }
  }
  return TransientSessionDataSchema.parse({
    ...current,
    messages: [...current.messages, ...parsedMessages],
    contextChips: contextChips.length > 0 ? [] : current.contextChips,
    unread: options.unread ?? current.unread,
    updatedAt: now,
  });
}

export function editTransientSessionUserMessage(
  transient: TransientSessionData,
  messageIndex: number,
  content: string,
): TransientSessionData {
  const current = TransientSessionDataSchema.parse(transient);
  if (
    !Number.isInteger(messageIndex) ||
    messageIndex < 0 ||
    messageIndex >= current.messages.length
  ) {
    throw new Error("Edited side chat message index is outside the transcript.");
  }
  const message = current.messages[messageIndex];
  if (message?.role !== "user" || message.kind !== "message") {
    throw new Error("Only side chat user messages can be edited.");
  }
  if (messageIndex !== lastUserMessageIndex(current.messages)) {
    throw new Error("Only the latest side chat user message can be edited.");
  }
  const nextContent = content.trim();
  if (!nextContent) throw new Error("Edited side chat message content cannot be empty.");
  return TransientSessionDataSchema.parse({
    ...current,
    messages: [...current.messages.slice(0, messageIndex), { ...message, content: nextContent }],
    updatedAt: new Date().toISOString(),
  });
}

export function transientSessionModelMessages(
  transient: TransientSessionData,
): Array<{ role: "user" | "assistant" | "system"; content: string }> {
  const current = TransientSessionDataSchema.parse(transient);
  return [...current.anchorMessages, ...current.messages].flatMap((message) => {
    if (
      message.kind !== "message" ||
      (message.role !== "user" && message.role !== "assistant" && message.role !== "system")
    ) {
      return [];
    }
    const chips = transientContextChips(message);
    const content =
      chips.length > 0 && message.role === "user"
        ? `${chips.map((chip) => `<side_context>\n${chip.text}\n</side_context>`).join("\n\n")}\n\n${message.content}`
        : message.content;
    return [{ role: message.role, content }];
  });
}

export function promoteTransientSessionFork(input: PromoteTransientSessionForkInput): SessionData {
  const transient = TransientSessionDataSchema.parse(input.transient);
  if (transient.runState !== "idle") {
    throw new Error("Stop the side chat before promoting it to a task.");
  }
  const now = new Date().toISOString();
  const promoted = SessionDataSchema.parse({
    id: uuidv4(),
    title: input.title?.trim() || promotedSideChatTitle(transient.title),
    forkedFrom: {
      sessionId: transient.parentSessionId,
      messageIndex: transient.anchor.messageIndex,
      createdAt: now,
    },
    ...(transient.projectId ? { projectId: transient.projectId } : {}),
    ...(transient.cwd ? { cwd: transient.cwd } : {}),
    agentName: transient.agentName,
    harness: transient.harness,
    ...(transient.model ? { model: transient.model } : {}),
    permissionMode: transient.permissionMode,
    pinned: false,
    messages: [...transient.anchorMessages, ...transient.messages],
    createdAt: now,
    updatedAt: now,
  });
  saveSession(promoted);
  return promoted;
}

export function migrateLegacySessions(
  options: SessionMigrationOptions = {},
): SessionMigrationResult {
  const sessionsDir = path.resolve(options.sessionsDir ?? configuredSessionsDir());
  if (!options.dryRun) ensureSessionsDir(sessionsDir);
  const legacyFiles = fs.existsSync(sessionsDir)
    ? fs
        .readdirSync(sessionsDir)
        .filter((entry) => entry.endsWith(".json"))
        .sort()
        .map((entry) => path.join(sessionsDir, entry))
    : [];
  const backupDir =
    !options.dryRun && legacyFiles.length > 0
      ? createMigrationBackupDir(sessionsDir, options.backupDir)
      : undefined;
  const entries = legacyFiles.map((sourcePath) => {
    if (options.dryRun) {
      return migrateLegacySessionFile(sourcePath, sessionsDir, {
        dryRun: true,
        backupDir,
      });
    }
    const id = path.basename(sourcePath, ".json");
    let targetPath: string;
    try {
      targetPath = sessionPaths(id, sessionsDir).jsonl;
    } catch (error) {
      return migrationFailure(id, sourcePath, "", error);
    }
    return withSessionLock(targetPath, () =>
      migrateLegacySessionFile(sourcePath, sessionsDir, { backupDir }),
    );
  });

  return {
    sessionsDir,
    ...(backupDir ? { backupDir } : {}),
    discovered: entries.length,
    planned: entries.filter((entry) => entry.status === "planned").length,
    migrated: entries.filter((entry) => entry.status === "migrated").length,
    skipped: entries.filter((entry) => entry.status === "skipped").length,
    failed: entries.filter((entry) => entry.status === "failed").length,
    sessions: entries,
  };
}

function migrateLegacySessionFile(
  sourcePath: string,
  sessionsDir: string,
  options: { dryRun?: boolean; backupDir?: string } = {},
): SessionMigrationEntry {
  const id = path.basename(sourcePath, ".json");
  let targetPath: string;
  try {
    targetPath = sessionPaths(id, sessionsDir).jsonl;
  } catch (error) {
    return migrationFailure(id, sourcePath, "", error);
  }

  try {
    const session = readLegacySession(sourcePath);
    if (fs.existsSync(targetPath)) {
      const existing = readSessionLog(targetPath, { rejectTornTail: true });
      if (!sameValue(existing, session)) {
        throw new Error("An existing JSONL Session has different data; legacy JSON was retained.");
      }
      if (options.dryRun) {
        return { id, sourcePath, targetPath, status: "skipped" };
      }
      indexSession(existing, targetPath, "jsonl", sessionsDir);
      const backupPath = backupLegacyFile(
        sourcePath,
        options.backupDir ?? createMigrationBackupDir(sessionsDir),
      );
      return { id, sourcePath, targetPath, backupPath, status: "skipped" };
    }

    if (options.dryRun) {
      return { id, sourcePath, targetPath, status: "planned" };
    }

    createSessionLog(session, targetPath);
    const replayed = readSessionLog(targetPath, {
      rejectTornTail: true,
      bypassCache: true,
    });
    if (!sameValue(replayed, session)) {
      throw new Error("Migrated JSONL did not replay to the original Session.");
    }
    indexSession(replayed, targetPath, "jsonl", sessionsDir);
    const backupPath = backupLegacyFile(
      sourcePath,
      options.backupDir ?? createMigrationBackupDir(sessionsDir),
    );
    return { id, sourcePath, targetPath, backupPath, status: "migrated" };
  } catch (error) {
    return migrationFailure(id, sourcePath, targetPath, error);
  }
}

function migrationFailure(
  id: string,
  sourcePath: string,
  targetPath: string,
  error: unknown,
): SessionMigrationEntry {
  return {
    id,
    sourcePath,
    targetPath,
    status: "failed",
    error: migrationErrorMessage(error),
  };
}

function migrationErrorMessage(error: unknown): string {
  let message: string;
  if (error instanceof ZodError) {
    const visible = error.issues.slice(0, 3).map((issue) => {
      const location = issue.path.length > 0 ? issue.path.join(".") : "<root>";
      return `${location}: ${issue.message}`;
    });
    const remaining = error.issues.length - visible.length;
    message = `Invalid legacy Session (${visible.join("; ")}${remaining > 0 ? `; +${remaining} more` : ""})`;
  } else {
    message = error instanceof Error ? error.message : String(error);
  }
  return message.length <= MIGRATION_ERROR_LIMIT
    ? message
    : `${message.slice(0, MIGRATION_ERROR_LIMIT - 1)}…`;
}

function createMigrationBackupDir(sessionsDir: string, requested?: string): string {
  if (requested) {
    const backupDir = path.resolve(requested);
    fs.mkdirSync(backupDir, { recursive: true, mode: 0o700 });
    return backupDir;
  }

  const root = path.join(sessionsDir, LEGACY_BACKUP_DIR);
  fs.mkdirSync(root, { recursive: true, mode: 0o700 });
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  let backupDir = path.join(root, timestamp);
  let suffix = 1;
  while (fs.existsSync(backupDir)) {
    backupDir = path.join(root, `${timestamp}-${suffix}`);
    suffix += 1;
  }
  fs.mkdirSync(backupDir, { mode: 0o700 });
  return backupDir;
}

function backupLegacyFile(sourcePath: string, backupDir: string): string {
  fs.mkdirSync(backupDir, { recursive: true, mode: 0o700 });
  const backupPath = path.join(backupDir, path.basename(sourcePath));
  if (fs.existsSync(backupPath)) {
    throw new Error(`Migration backup already exists: ${backupPath}`);
  }
  fs.renameSync(sourcePath, backupPath);
  return backupPath;
}

function withSessionLock<T>(rolloutPath: string, action: () => T): T {
  const lockPath = `${rolloutPath}.lock`;
  const token = uuidv4();
  const deadline = Date.now() + SESSION_LOCK_TIMEOUT_MS;

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
      if (!isFileExistsError(error)) throw error;
      if (removeStaleSessionLock(lockPath)) continue;
      if (Date.now() >= deadline) {
        throw new Error(`Timed out waiting for Session writer lock: ${lockPath}`);
      }
      sleepSync(10);
    }
  }

  try {
    return action();
  } finally {
    try {
      const lock = objectRecord(JSON.parse(fs.readFileSync(lockPath, "utf8")), "Session lock");
      if (lock.token === token) fs.unlinkSync(lockPath);
    } catch {
      // Another process may already have recovered a stale lock.
    }
  }
}

function removeStaleSessionLock(lockPath: string): boolean {
  try {
    const age = Date.now() - fs.statSync(lockPath).mtimeMs;
    const lock = objectRecord(JSON.parse(fs.readFileSync(lockPath, "utf8")), "Session lock");
    const pid = lock.pid;
    if (typeof pid === "number" && Number.isSafeInteger(pid) && pid > 0) {
      if (processIsAlive(pid)) return false;
      fs.unlinkSync(lockPath);
      return true;
    }
    if (age < SESSION_LOCK_STALE_MS) return false;
    fs.unlinkSync(lockPath);
    return true;
  } catch {
    try {
      if (Date.now() - fs.statSync(lockPath).mtimeMs < SESSION_LOCK_STALE_MS) return false;
      fs.unlinkSync(lockPath);
      return true;
    } catch {
      return false;
    }
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

function isFileExistsError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error as { code?: unknown }).code === "EEXIST"
  );
}

function isNoSuchProcessError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error as { code?: unknown }).code === "ESRCH"
  );
}

function createSessionLog(session: SessionData, targetPath: string): void {
  const event: SessionCreatedEvent = {
    schemaVersion: SESSION_SCHEMA_VERSION,
    type: "session_created",
    timestamp: session.createdAt,
    session,
  };
  const temporaryPath = `${targetPath}.tmp-${process.pid}-${uuidv4()}`;
  let linked = false;
  try {
    writeNewFileDurably(temporaryPath, jsonlLines([event]));
    fs.linkSync(temporaryPath, targetPath);
    linked = true;
    fs.chmodSync(targetPath, 0o600);
  } finally {
    if (fs.existsSync(temporaryPath)) fs.unlinkSync(temporaryPath);
  }
  if (!linked) throw new Error(`Failed to create Session log: ${targetPath}`);
  cacheSession(targetPath, session);
}

function appendSessionEvents(filePath: string, events: SessionEvent[]): void {
  if (events.length === 0) return;
  readSessionLog(filePath, { rejectTornTail: true });
  appendFileDurably(filePath, jsonlLines(events));
}

function readSessionLog(
  filePath: string,
  options: { rejectTornTail?: boolean; bypassCache?: boolean } = {},
): SessionData {
  const stat = fs.statSync(filePath);
  const bytes = stat.size;
  const cached = sessionCache.get(filePath);
  if (!options.bypassCache && cached?.bytes === bytes && cached.mtimeMs === stat.mtimeMs) {
    if (cached.tornTail && options.rejectTornTail) {
      throw new Error(`Session log has a torn final record and cannot be appended: ${filePath}`);
    }
    return SessionDataSchema.parse(cached.session);
  }

  const parsed = parseJsonlFile(filePath, parseSessionEvent);
  if (parsed.tornTail && options.rejectTornTail) {
    throw new Error(`Session log has a torn final record and cannot be appended: ${filePath}`);
  }
  const session = replaySessionEvents(parsed.records, filePath);
  assertSessionMatchesFile(session, filePath, ".jsonl");
  sessionCache.set(filePath, {
    bytes,
    mtimeMs: stat.mtimeMs,
    session,
    tornTail: parsed.tornTail,
  });
  return SessionDataSchema.parse(session);
}

function readLegacySession(filePath: string): SessionData {
  const input: unknown = JSON.parse(fs.readFileSync(filePath, "utf8"));
  const current = SessionDataSchema.safeParse(input);
  const session = current.success
    ? current.data
    : normalizeLegacyDesktopSession(input, path.dirname(filePath));
  assertSessionMatchesFile(session, filePath, ".json");
  return session;
}

function assertSessionMatchesFile(
  session: Pick<SessionData, "id">,
  filePath: string,
  extension: ".json" | ".jsonl",
): void {
  const expectedId = path.basename(filePath, extension);
  if (session.id !== expectedId) {
    throw new Error(
      `Session id "${session.id}" does not match its filename "${expectedId}${extension}".`,
    );
  }
}

function normalizeLegacyDesktopSession(input: unknown, sessionsDir: string): SessionData {
  const legacy = LegacyDesktopSessionSchema.parse(input);
  const agent = resolveLegacyAgent(legacy, sessionsDir);
  const createdAt = normalizeLegacyTimestamp(legacy.created_at, "created_at");
  const updatedAt = normalizeLegacyTimestamp(legacy.updated_at, "updated_at");
  return SessionDataSchema.parse({
    id: legacy.id,
    title: legacy.title || "New Session",
    ...(legacy.acp_session_id ? { acpSessionId: legacy.acp_session_id } : {}),
    cwd: legacy.working_dir,
    agentName: agent.agentName,
    harness: agent.harness,
    ...(legacy.model_override || agent.model
      ? { model: legacy.model_override ?? agent.model }
      : {}),
    pinned: legacy.pinned,
    messages: legacy.messages.map(normalizeLegacyMessage),
    ...(legacy.archived ? { archivedAt: updatedAt } : {}),
    createdAt,
    updatedAt,
  });
}

function normalizeLegacyMessage(message: z.infer<typeof LegacyDesktopMessageSchema>): MessageChunk {
  const kind = {
    Message: "message",
    Thinking: "thinking",
    ToolCall: "tool_call",
    ToolResult: "tool_result",
  }[message.kind] as MessageChunk["kind"];
  const structuredContent =
    message.tool_result !== undefined && message.tool_result !== message.content
      ? { legacyToolResult: message.tool_result }
      : undefined;
  return MessageChunkSchema.parse({
    role: message.is_user ? "user" : kind === "tool_result" ? "tool" : "assistant",
    content: message.content,
    kind,
    ...(message.tool_name ? { toolName: message.tool_name } : {}),
    ...(structuredContent ? { structuredContent } : {}),
    ...(message.duration_ms !== undefined ? { render: { durationMs: message.duration_ms } } : {}),
  });
}

function resolveLegacyAgent(
  legacy: LegacyDesktopSession,
  sessionsDir: string,
): LegacyAgentDescriptor {
  if (legacy.agent_runtime) return legacyRuntimeDescriptor(legacy.agent_runtime);

  const instances = readLegacyAgentInstances(sessionsDir);
  const instance =
    instances.find((candidate) => candidate.id === legacy.agent_instance_id) ?? instances[0];
  if (instance) {
    return {
      agentName: instance.label,
      harness: normalizeLegacyHarness(instance.harness),
      ...(instance.model ? { model: instance.model } : {}),
    };
  }
  return {
    agentName: legacy.agent_instance_id || "Legacy agent",
    harness: "swarmx",
  };
}

function readLegacyAgentInstances(
  sessionsDir: string,
): Array<z.infer<typeof LegacyAgentInstanceSchema>> {
  const instancesPath = path.join(path.dirname(sessionsDir), "instances.json");
  try {
    return z
      .array(LegacyAgentInstanceSchema)
      .parse(JSON.parse(fs.readFileSync(instancesPath, "utf8")));
  } catch {
    return [];
  }
}

function legacyRuntimeDescriptor(
  runtime: NonNullable<LegacyDesktopSession["agent_runtime"]>,
): LegacyAgentDescriptor {
  return {
    claude_agent_acp: { agentName: "Claude Code", harness: "claude_code" },
    codex_acp: { agentName: "Codex", harness: "codex" },
    open_code: { agentName: "OpenCode", harness: "opencode" },
    hermes: { agentName: "Hermes", harness: "hermes" },
    open_claw: { agentName: "OpenClaw", harness: "openclaw" },
  }[runtime];
}

function normalizeLegacyHarness(harness: string): string {
  return (
    {
      SwarmX: "swarmx",
      ClaudeCode: "claude_code",
      Codex: "codex",
      OpenCode: "opencode",
      Hermes: "hermes",
      OpenClaw: "openclaw",
    }[harness] ?? harness
  );
}

function normalizeLegacyTimestamp(value: string, field: string): string {
  const numeric = /^\d+$/.test(value) ? Number(value) : Number.NaN;
  const milliseconds = Number.isSafeInteger(numeric)
    ? value.length >= 13
      ? numeric
      : numeric * 1_000
    : Date.parse(value);
  const timestamp = new Date(milliseconds);
  if (!Number.isFinite(milliseconds) || Number.isNaN(timestamp.getTime())) {
    throw new Error(`Legacy Session ${field} is not a valid timestamp.`);
  }
  return timestamp.toISOString();
}

function parseSessionEvent(input: unknown): SessionEvent {
  const record = objectRecord(input, "Session event");
  if (record.schemaVersion !== SESSION_SCHEMA_VERSION) {
    throw new Error(`Unsupported Session schema version: ${String(record.schemaVersion)}`);
  }
  if (typeof record.timestamp !== "string" || !record.timestamp) {
    throw new Error("Session event timestamp must be a non-empty string.");
  }

  if (record.type === "session_created") {
    return {
      schemaVersion: SESSION_SCHEMA_VERSION,
      type: "session_created",
      timestamp: record.timestamp,
      session: SessionDataSchema.parse(record.session),
    };
  }
  if (record.type === "messages_appended" || record.type === "messages_replaced") {
    if (!Array.isArray(record.messages)) throw new Error(`${record.type} requires messages.`);
    const messages = record.messages.map((message) => MessageChunkSchema.parse(message));
    if (record.type === "messages_replaced" && record.reason !== undefined) {
      if (record.reason !== "edit_last_user_message") {
        throw new Error(`Unknown messages_replaced reason: ${String(record.reason)}`);
      }
      if (
        !Number.isInteger(record.replacedFromIndex) ||
        (record.replacedFromIndex as number) < 0 ||
        !Number.isInteger(record.replacedMessageCount) ||
        (record.replacedMessageCount as number) < 1
      ) {
        throw new Error("Edited messages_replaced events require a valid replaced range.");
      }
      return {
        schemaVersion: SESSION_SCHEMA_VERSION,
        type: "messages_replaced",
        timestamp: record.timestamp,
        messages,
        reason: record.reason,
        replacedFromIndex: record.replacedFromIndex as number,
        replacedMessageCount: record.replacedMessageCount as number,
      };
    }
    return {
      schemaVersion: SESSION_SCHEMA_VERSION,
      type: record.type,
      timestamp: record.timestamp,
      messages,
    };
  }
  if (record.type === "session_updated") {
    const metadata = parseSessionMetadata(record.session);
    return {
      schemaVersion: SESSION_SCHEMA_VERSION,
      type: "session_updated",
      timestamp: record.timestamp,
      session: metadata,
    };
  }
  throw new Error(`Unknown Session event type: ${String(record.type)}`);
}

function replaySessionEvents(events: SessionEvent[], filePath: string): SessionData {
  let session: SessionData | null = null;
  for (const [index, event] of events.entries()) {
    if (event.type === "session_created") {
      if (session) throw new Error(`Duplicate session_created record at ${filePath}:${index + 1}`);
      session = event.session;
      continue;
    }
    if (!session) {
      throw new Error(`Session event precedes session_created at ${filePath}:${index + 1}`);
    }
    if (event.type === "messages_appended") {
      session = SessionDataSchema.parse({
        ...session,
        messages: [...session.messages, ...event.messages],
        updatedAt: event.timestamp,
      });
    } else if (event.type === "messages_replaced") {
      session = SessionDataSchema.parse({
        ...session,
        messages: event.messages,
        updatedAt: event.timestamp,
      });
    } else {
      if (event.session.id !== session.id || event.session.createdAt !== session.createdAt) {
        throw new Error(`Session identity changed at ${filePath}:${index + 1}`);
      }
      session = SessionDataSchema.parse({
        ...event.session,
        messages: session.messages,
        updatedAt: event.timestamp,
      });
    }
  }
  if (!session) throw new Error(`Session log has no session_created record: ${filePath}`);
  return session;
}

function messagesAppendedEvent(messages: MessageChunk[], timestamp: string): MessagesAppendedEvent {
  return {
    schemaVersion: SESSION_SCHEMA_VERSION,
    type: "messages_appended",
    timestamp,
    messages,
  };
}

function messagesReplacedEvent(
  messages: MessageChunk[],
  timestamp: string,
  replacement: Pick<
    MessagesReplacedEvent,
    "reason" | "replacedFromIndex" | "replacedMessageCount"
  > = {},
): MessagesReplacedEvent {
  return {
    schemaVersion: SESSION_SCHEMA_VERSION,
    type: "messages_replaced",
    timestamp,
    messages,
    ...replacement,
  };
}

function sessionUpdatedEvent(session: SessionData, timestamp: string): SessionUpdatedEvent {
  return {
    schemaVersion: SESSION_SCHEMA_VERSION,
    type: "session_updated",
    timestamp,
    session: sessionMetadata(session),
  };
}

function sessionMetadata(session: SessionData): SessionMetadata {
  const { messages: _messages, ...metadata } = SessionDataSchema.parse(session);
  return metadata;
}

function parseSessionMetadata(input: unknown): SessionMetadata {
  return sessionMetadata(
    SessionDataSchema.parse({
      ...objectRecord(input, "Session metadata"),
      messages: [],
    }),
  );
}

function metadataWithoutUpdatedAt(metadata: SessionMetadata): Omit<SessionMetadata, "updatedAt"> {
  const { updatedAt: _updatedAt, ...rest } = metadata;
  return rest;
}

function cacheSession(filePath: string, session: SessionData): void {
  const stat = fs.statSync(filePath);
  sessionCache.set(filePath, {
    bytes: stat.size,
    mtimeMs: stat.mtimeMs,
    session: SessionDataSchema.parse(session),
    tornTail: false,
  });
}

function sessionSummary(session: SessionData): SessionSummary {
  return {
    ...sessionMetadata(session),
    messageCount: session.messages.length,
  };
}

function indexSession(
  session: SessionData,
  sourcePath: string,
  sourceFormat: SessionSourceFormat,
  sessionsDir: string,
): void {
  const sourceStat = fs.statSync(sourcePath);
  appendSessionIndexEvent(
    {
      schemaVersion: SESSION_SCHEMA_VERSION,
      type: "session_indexed",
      timestamp: new Date().toISOString(),
      sessionId: session.id,
      sourceBytes: sourceStat.size,
      sourceMtimeMs: sourceStat.mtimeMs,
      sourceFormat,
      summary: sessionSummary(session),
    },
    sessionsDir,
  );
}

function appendSessionIndexEvent(event: SessionIndexEvent, sessionsDir: string): void {
  const indexPath = path.join(sessionsDir, SESSION_INDEX_FILE);
  if (
    fs.existsSync(indexPath) &&
    fs.statSync(indexPath).size > 0 &&
    !fileEndsWithNewline(indexPath)
  ) {
    rebuildSessionIndex(sessionsDir);
  }
  appendFileDurably(indexPath, jsonlLines([event]));
}

function readOrRebuildSessionIndex(sessionsDir: string): Map<string, SessionIndexEvent> {
  const indexPath = path.join(sessionsDir, SESSION_INDEX_FILE);
  if (!fs.existsSync(indexPath)) return rebuildSessionIndex(sessionsDir);
  try {
    const parsed = parseJsonlFile(indexPath, parseSessionIndexEvent);
    if (parsed.tornTail) return rebuildSessionIndex(sessionsDir);
    return latestIndexEvents(parsed.records);
  } catch {
    return rebuildSessionIndex(sessionsDir);
  }
}

function rebuildSessionIndex(sessionsDir: string): Map<string, SessionIndexEvent> {
  const records: SessionIndexEvent[] = [];
  for (const source of discoverSessionSources(sessionsDir)) {
    try {
      const session =
        source.format === "jsonl" ? readSessionLog(source.path) : readLegacySession(source.path);
      const sourceStat = fs.statSync(source.path);
      records.push({
        schemaVersion: SESSION_SCHEMA_VERSION,
        type: "session_indexed",
        timestamp: new Date().toISOString(),
        sessionId: session.id,
        sourceBytes: sourceStat.size,
        sourceMtimeMs: sourceStat.mtimeMs,
        sourceFormat: source.format,
        summary: sessionSummary(session),
      });
    } catch {
      // A derived index can omit corrupt Sessions; canonical files remain untouched.
    }
  }

  const indexPath = path.join(sessionsDir, SESSION_INDEX_FILE);
  const temporaryPath = `${indexPath}.tmp-${process.pid}-${uuidv4()}`;
  writeNewFileDurably(temporaryPath, jsonlLines(records));
  fs.renameSync(temporaryPath, indexPath);
  fs.chmodSync(indexPath, 0o600);
  return latestIndexEvents(records);
}

function parseSessionIndexEvent(input: unknown): SessionIndexEvent {
  const record = objectRecord(input, "Session index event");
  if (record.schemaVersion !== SESSION_SCHEMA_VERSION) {
    throw new Error(`Unsupported Session index version: ${String(record.schemaVersion)}`);
  }
  if (
    (record.type !== "session_indexed" && record.type !== "session_deleted") ||
    typeof record.timestamp !== "string" ||
    typeof record.sessionId !== "string"
  ) {
    throw new Error("Invalid Session index event.");
  }
  if (record.type === "session_deleted") {
    return {
      schemaVersion: SESSION_SCHEMA_VERSION,
      type: "session_deleted",
      timestamp: record.timestamp,
      sessionId: record.sessionId,
    };
  }
  if (
    (record.sourceFormat !== "json" && record.sourceFormat !== "jsonl") ||
    typeof record.sourceBytes !== "number" ||
    !Number.isSafeInteger(record.sourceBytes) ||
    record.sourceBytes < 0 ||
    typeof record.sourceMtimeMs !== "number" ||
    !Number.isFinite(record.sourceMtimeMs) ||
    record.sourceMtimeMs < 0
  ) {
    throw new Error("Invalid indexed Session source.");
  }
  const summaryRecord = objectRecord(record.summary, "Session summary");
  const messageCount = summaryRecord.messageCount;
  if (typeof messageCount !== "number" || !Number.isSafeInteger(messageCount) || messageCount < 0) {
    throw new Error("Invalid Session message count.");
  }
  const { messageCount: _messageCount, ...metadata } = summaryRecord;
  const summary = {
    ...parseSessionMetadata(metadata),
    messageCount,
  };
  if (summary.id !== record.sessionId) {
    throw new Error("Indexed Session id does not match its summary.");
  }
  return {
    schemaVersion: SESSION_SCHEMA_VERSION,
    type: "session_indexed",
    timestamp: record.timestamp,
    sessionId: record.sessionId,
    sourceBytes: record.sourceBytes,
    sourceMtimeMs: record.sourceMtimeMs,
    sourceFormat: record.sourceFormat,
    summary,
  };
}

function latestIndexEvents(records: SessionIndexEvent[]): Map<string, SessionIndexEvent> {
  const latest = new Map<string, SessionIndexEvent>();
  for (const record of records) {
    if (record.type === "session_deleted") latest.delete(record.sessionId);
    else latest.set(record.sessionId, record);
  }
  return latest;
}

function discoverSessionSources(
  sessionsDir: string,
): Array<{ id: string; path: string; format: SessionSourceFormat }> {
  const sources = new Map<string, { id: string; path: string; format: SessionSourceFormat }>();
  for (const entry of fs.readdirSync(sessionsDir)) {
    if (entry === SESSION_INDEX_FILE) continue;
    const extension = path.extname(entry);
    if (extension !== ".json" && extension !== ".jsonl") continue;
    const id = entry.slice(0, -extension.length);
    if (!id) continue;
    const format: SessionSourceFormat = extension === ".jsonl" ? "jsonl" : "json";
    const existing = sources.get(id);
    if (!existing || format === "jsonl") {
      sources.set(id, { id, path: path.join(sessionsDir, entry), format });
    }
  }
  return [...sources.values()].sort((left, right) => left.id.localeCompare(right.id));
}

function parseJsonlFile<T>(filePath: string, parseRecord: (input: unknown) => T): ParsedJsonl<T> {
  const source = fs.readFileSync(filePath, "utf8");
  const terminated = source.endsWith("\n");
  const lines = source.split("\n");
  if (terminated) lines.pop();
  const records: T[] = [];

  for (const [index, line] of lines.entries()) {
    if (!line.trim()) continue;
    try {
      records.push(parseRecord(JSON.parse(line)));
    } catch (error) {
      const isTornTail = !terminated && index === lines.length - 1;
      if (isTornTail) return { records, tornTail: true };
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`${path.basename(filePath)} line ${index + 1} is corrupt: ${message}`);
    }
  }
  return { records, tornTail: false };
}

function writeNewFileDurably(filePath: string, content: string): void {
  const descriptor = fs.openSync(filePath, "wx", 0o600);
  try {
    if (content) fs.writeFileSync(descriptor, content, "utf8");
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
}

function appendFileDurably(filePath: string, content: string): void {
  const prefix =
    fs.existsSync(filePath) && fs.statSync(filePath).size > 0 && !fileEndsWithNewline(filePath)
      ? "\n"
      : "";
  const descriptor = fs.openSync(filePath, "a", 0o600);
  try {
    fs.writeFileSync(descriptor, `${prefix}${content}`, "utf8");
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
  fs.chmodSync(filePath, 0o600);
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

function jsonlLines(records: unknown[]): string {
  return records.map((record) => JSON.stringify(record)).join("\n") + (records.length ? "\n" : "");
}

function objectRecord(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function isMessagePrefix(current: MessageChunk[], next: MessageChunk[]): boolean {
  return (
    current.length <= next.length &&
    current.every((message, index) => sameValue(message, next[index]))
  );
}

function sameValue(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function lastUserMessageIndex(messages: MessageChunk[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.role === "user" && message.kind === "message") return index;
  }
  return -1;
}

function continuedSessionTitle(title: string): string {
  const normalized = title.trim() || "New Session";
  return `${normalized} (continued)`;
}

function nextSideChatTitle(index: number): string {
  return `Side chat ${index}`;
}

function promotedSideChatTitle(title: string): string {
  const normalized = title.trim() || "Side chat";
  return `${normalized} (promoted)`;
}

function recordValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function transientContextChips(message: MessageChunk): TransientSessionContextChip[] {
  const value = recordValue(message.structuredContent).sideChatContext;
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const parsed = z
      .object({
        id: z.string().min(1),
        text: z.string().min(1),
        createdAt: z.string().min(1),
      })
      .safeParse(item);
    return parsed.success ? [parsed.data] : [];
  });
}

function sortSessions<T extends Pick<SessionSummary, "pinned" | "updatedAt">>(sessions: T[]): T[] {
  return [...sessions].sort(compareSessions);
}

function compareSessions(
  a: Pick<SessionSummary, "pinned" | "updatedAt">,
  b: Pick<SessionSummary, "pinned" | "updatedAt">,
): number {
  return (
    Number(b.pinned) - Number(a.pinned) ||
    new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
  );
}

function belongsToProject(
  session: Pick<SessionMetadata, "projectId" | "cwd">,
  project: SessionProjectContext,
): boolean {
  if (project.projectId && session.projectId === project.projectId) return true;
  if (!project.cwd || !session.cwd) return false;
  return path.resolve(session.cwd) === path.resolve(project.cwd);
}
