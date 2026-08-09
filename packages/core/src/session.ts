import { createHash } from "node:crypto";
import * as fs from "node:fs";
import { homedir } from "node:os";
import * as path from "node:path";
import { v4 as uuidv4 } from "uuid";
import { z } from "zod";
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
const DEFAULT_SESSIONS_DIR = path.join(homedir(), ".swarmx", "projects");
const LEGACY_FLAT_SESSIONS_DIR = path.join(homedir(), ".swarmx", "sessions");
const RECENTS_DIRECTORY = "__recents__";
const SESSION_INDEX_FILE = "sessions-index.json";
const LEGACY_SESSION_INDEX_FILE = "sessions.index.jsonl";
const SESSION_LOCK_TIMEOUT_MS = 5_000;
const SESSION_LOCK_STALE_MS = 30_000;

type SessionMetadata = Omit<SessionData, "messages">;

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

interface SessionIndexEntry {
  sessionId: string;
  sourceBytes: number;
  sourceMtimeMs: number;
  sourceFormat: "jsonl";
  summary: SessionSummary;
}

interface SessionIndexDocument {
  version: typeof SESSION_SCHEMA_VERSION;
  entries: SessionIndexEntry[];
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

interface SessionSource {
  id: string;
  path: string;
  directory: string;
}

const sessionCache = new Map<string, SessionCacheEntry>();

function configuredSessionsDir(): string {
  return path.resolve(process.env.SWARMX_SESSIONS_DIR ?? DEFAULT_SESSIONS_DIR);
}

function configuredSessionRoots(): string[] {
  const configured = configuredSessionsDir();
  if (process.env.SWARMX_SESSIONS_DIR !== undefined) return [configured];
  return [configured, LEGACY_FLAT_SESSIONS_DIR];
}

function ensureSessionsDir(sessionsDir = configuredSessionsDir()): string {
  if (!fs.existsSync(sessionsDir)) {
    fs.mkdirSync(sessionsDir, { recursive: true, mode: 0o700 });
  }
  return sessionsDir;
}

function sessionPaths(
  id: string,
  sessionDirectory: string,
): {
  jsonl: string;
} {
  if (!id || path.basename(id) !== id || id === "." || id === "..") {
    throw new Error("Session id must be a non-empty file-safe value.");
  }
  return {
    jsonl: path.join(sessionDirectory, `${id}.jsonl`),
  };
}

function canonicalSessionDirectory(
  session: Pick<SessionData, "cwd" | "projectId">,
  sessionsDir = configuredSessionsDir(),
): string {
  return path.join(sessionsDir, sessionPartitionName(session));
}

function sessionPartitionName(session: Pick<SessionData, "cwd" | "projectId">): string {
  const cwd = session.cwd?.trim();
  if (cwd) {
    const normalized = path.resolve(cwd);
    const readable = normalized
      .replace(/[^a-zA-Z0-9._-]+/g, "-")
      .replace(/-{2,}/g, "-")
      .slice(0, 96);
    return `${readable || "root"}-${stableKey(normalized)}`;
  }
  const projectId = session.projectId?.trim();
  if (projectId) return `project-${stableKey(projectId)}`;
  return RECENTS_DIRECTORY;
}

function stableKey(value: string): string {
  return createHash("sha256").update(value).digest("hex").slice(0, 12);
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
  const now = new Date().toISOString();
  session.updatedAt = now;
  const next = SessionDataSchema.parse(session);
  const source = prepareSessionPathForWrite(next);

  withSessionLock(source.path, () =>
    saveSessionLocked(next, { jsonl: source.path }, source.directory, now),
  );
}

function saveSessionLocked(
  next: SessionData,
  paths: ReturnType<typeof sessionPaths>,
  sessionsDir: string,
  now: string,
): void {
  if (!fs.existsSync(paths.jsonl)) {
    createSessionLog(next, paths.jsonl);
    indexSession(next, paths.jsonl, sessionsDir);
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
  indexSession(effective, paths.jsonl, sessionsDir);
}

export function loadSession(id: string): SessionData | null {
  return loadSessionFromRoots(id, configuredSessionRoots());
}

export interface ListSessionsOptions {
  includeArchived?: boolean;
  sessionsDir?: string;
}

export function listSessionSummaries(options: ListSessionsOptions = {}): SessionSummary[] {
  const roots = options.sessionsDir
    ? [path.resolve(options.sessionsDir)]
    : configuredSessionRoots();
  const summaries = new Map<string, SessionSummary>();

  for (const partition of discoverSessionPartitions(roots)) {
    const index = readOrRebuildSessionIndex(partition);
    for (const source of discoverSessionSources(partition)) {
      if (summaries.has(source.id)) continue;
      const indexed = index.get(source.id);
      const sourceStat = fs.statSync(source.path);
      const sourceBytes = sourceStat.size;
      let summary = indexed?.summary;
      if (
        !indexed ||
        !summary ||
        indexed.sourceBytes !== sourceBytes ||
        indexed.sourceMtimeMs !== sourceStat.mtimeMs
      ) {
        try {
          const session = readSessionLog(source.path);
          summary = sessionSummary(session);
          upsertSessionIndexEntry(
            {
              sessionId: session.id,
              sourceBytes,
              sourceMtimeMs: sourceStat.mtimeMs,
              sourceFormat: "jsonl",
              summary,
            },
            partition,
          );
        } catch {
          continue;
        }
      }
      if (options.includeArchived || !summary.archivedAt) summaries.set(summary.id, summary);
    }
  }

  return sortSessions([...summaries.values()]);
}

export function listSessions(options: ListSessionsOptions = {}): SessionData[] {
  const roots = options.sessionsDir
    ? [path.resolve(options.sessionsDir)]
    : configuredSessionRoots();
  return listSessionSummaries({
    includeArchived: true,
    ...(options.sessionsDir ? { sessionsDir: options.sessionsDir } : {}),
  })
    .flatMap((summary) => {
      const session = loadSessionFromRoots(summary.id, roots);
      return session && (options.includeArchived || !session.archivedAt) ? [session] : [];
    })
    .sort(compareSessions);
}

function loadSessionFromRoots(id: string, roots: string[]): SessionData | null {
  try {
    const source = findSessionSource(id, roots);
    return source ? readSessionLog(source.path) : null;
  } catch {
    return null;
  }
}

function prepareSessionPathForWrite(session: SessionData): SessionSource {
  const sessionsDir = ensureSessionsDir();
  const sessionDirectory = canonicalSessionDirectory(session, sessionsDir);
  const targetPath = sessionPaths(session.id, sessionDirectory).jsonl;
  const layoutLockPath = path.join(sessionsDir, ".session-layout");

  return withSessionLock(layoutLockPath, () => {
    const existing = findSessionSources(session.id, configuredSessionRoots());
    if (existing.length > 1) {
      throw new Error(`Session ${session.id} exists in more than one Project directory.`);
    }

    const source = existing[0];
    if (source?.path === targetPath) return source;

    fs.mkdirSync(sessionDirectory, { recursive: true, mode: 0o700 });
    if (fs.existsSync(targetPath)) {
      throw new Error(`Session target already exists: ${targetPath}`);
    }
    if (source) {
      readSessionLog(source.path, { rejectTornTail: true });
      fs.renameSync(source.path, targetPath);
      fs.chmodSync(targetPath, 0o600);
      sessionCache.delete(source.path);
      rebuildSessionIndex(source.directory);
    }
    return { id: session.id, path: targetPath, directory: sessionDirectory };
  });
}

function findSessionSource(id: string, roots: string[]): SessionSource | null {
  return findSessionSources(id, roots)[0] ?? null;
}

function findSessionSources(id: string, roots: string[]): SessionSource[] {
  sessionPaths(id, ".");
  const sources: SessionSource[] = [];
  for (const directory of discoverSessionPartitions(roots)) {
    const filePath = sessionPaths(id, directory).jsonl;
    if (fs.existsSync(filePath)) sources.push({ id, path: filePath, directory });
  }
  return sources;
}

function discoverSessionPartitions(roots: string[]): string[] {
  const partitions: string[] = [];
  for (const root of roots) {
    if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) continue;
    const children = fs
      .readdirSync(root, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => path.join(root, entry.name))
      .sort((left, right) => left.localeCompare(right));
    for (const child of children) {
      if (
        discoverSessionSources(child).length > 0 ||
        fs.existsSync(path.join(child, SESSION_INDEX_FILE))
      ) {
        partitions.push(child);
      }
    }
    if (
      discoverSessionSources(root).length > 0 ||
      fs.existsSync(path.join(root, SESSION_INDEX_FILE))
    ) {
      partitions.push(root);
    }
  }
  return partitions;
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
  let source: SessionSource | null;
  try {
    source = findSessionSource(id, configuredSessionRoots());
  } catch {
    return false;
  }
  if (!source) return false;
  return withSessionLock(source.path, () => {
    const deleted = fs.existsSync(source.path);
    if (deleted) {
      fs.unlinkSync(source.path);
      sessionCache.delete(source.path);
      rebuildSessionIndex(source.directory);
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
  return (
    withSessionLog(id, (current, paths, sessionsDir) => {
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
      indexSession(next, paths.jsonl, sessionsDir);
      return true;
    }) ?? false
  );
}

function withSessionLog<T>(
  id: string,
  fn: (current: SessionData, paths: ReturnType<typeof sessionPaths>, sessionsDir: string) => T,
): T | null {
  const existing = findSessionSource(id, configuredSessionRoots());
  if (!existing) return null;
  const current = readSessionLog(existing.path, { rejectTornTail: true });
  const source = prepareSessionPathForWrite(current);
  return withSessionLock(source.path, () => {
    if (!fs.existsSync(source.path)) return null;
    const latest = readSessionLog(source.path, { rejectTornTail: true });
    return fn(latest, { jsonl: source.path }, source.directory);
  });
}

function withVerifiedSessionSnapshot<T>(
  id: string,
  expectedMessages: readonly MessageChunk[],
  index: number,
  errors: { mismatch: string; badIndex: string },
  fn: (current: SessionData, paths: ReturnType<typeof sessionPaths>, sessionsDir: string) => T,
): T | null {
  return withSessionLog(id, (current, paths, sessionsDir) => {
    const parsedExpected = expectedMessages.map((message) => MessageChunkSchema.parse(message));
    if (!sameValue(current.messages, parsedExpected)) {
      throw new Error(errors.mismatch);
    }
    if (!Number.isInteger(index) || index < 0 || index >= current.messages.length) {
      throw new Error(errors.badIndex);
    }
    return fn(current, paths, sessionsDir);
  });
}

export function editSessionUserMessage(input: EditSessionUserMessageInput): SessionData | null {
  return withVerifiedSessionSnapshot(
    input.id,
    input.expectedMessages,
    input.messageIndex,
    {
      mismatch: "Session history changed before the message edit could be saved.",
      badIndex: "Edited message index is outside the current Session history.",
    },
    (current, paths, sessionsDir) => {
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
      indexSession(next, paths.jsonl, sessionsDir);
      return next;
    },
  );
}

export function forkSession(input: ForkSessionInput): SessionData | null {
  return withVerifiedSessionSnapshot(
    input.id,
    input.expectedMessages,
    input.throughMessageIndex,
    {
      mismatch: "Session history changed before the new chat could be created.",
      badIndex: "Fork message index is outside the current Session history.",
    },
    (current) => {
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
        ...(current.builtinTools ? { builtinTools: current.builtinTools } : {}),
        permissionMode: current.permissionMode,
        pinned: false,
        messages: current.messages.slice(0, input.throughMessageIndex + 1),
        createdAt: now,
        updatedAt: now,
      });
      saveSession(forked);
      return forked;
    },
  );
}

export function createTransientSessionFork(
  input: CreateTransientSessionForkInput,
): TransientSessionData | null {
  return withVerifiedSessionSnapshot(
    input.id,
    input.expectedMessages,
    input.throughMessageIndex,
    {
      mismatch: "Session history changed before the side chat could be created.",
      badIndex: "Side chat anchor is outside the current Session history.",
    },
    (current) => {
      const now = new Date().toISOString();
      const anchorMessages = current.messages.slice(0, input.throughMessageIndex + 1);
      return TransientSessionDataSchema.parse({
        id: uuidv4(),
        parentSessionId: current.id,
        title: input.title?.trim() || "Side chat 1",
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
        ...(current.builtinTools ? { builtinTools: current.builtinTools } : {}),
        ...(current.projectId ? { projectId: current.projectId } : {}),
        ...(current.cwd ? { cwd: current.cwd } : {}),
        permissionMode: current.permissionMode,
        runState: "idle",
        unread: false,
        createdAt: now,
        updatedAt: now,
      });
    },
  );
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
    ...(transient.builtinTools ? { builtinTools: transient.builtinTools } : {}),
    permissionMode: transient.permissionMode,
    pinned: false,
    messages: [...transient.anchorMessages, ...transient.messages],
    createdAt: now,
    updatedAt: now,
  });
  saveSession(promoted);
  return promoted;
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
  assertSessionMatchesFile(session, filePath);
  sessionCache.set(filePath, {
    bytes,
    mtimeMs: stat.mtimeMs,
    session,
    tornTail: parsed.tornTail,
  });
  return SessionDataSchema.parse(session);
}

function assertSessionMatchesFile(session: Pick<SessionData, "id">, filePath: string): void {
  const expectedId = path.basename(filePath, ".jsonl");
  if (session.id !== expectedId) {
    throw new Error(
      `Session id "${session.id}" does not match its filename "${expectedId}.jsonl".`,
    );
  }
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

function indexSession(session: SessionData, sourcePath: string, sessionsDir: string): void {
  const sourceStat = fs.statSync(sourcePath);
  upsertSessionIndexEntry(
    {
      sessionId: session.id,
      sourceBytes: sourceStat.size,
      sourceMtimeMs: sourceStat.mtimeMs,
      sourceFormat: "jsonl",
      summary: sessionSummary(session),
    },
    sessionsDir,
  );
}

function upsertSessionIndexEntry(entry: SessionIndexEntry, sessionsDir: string): void {
  withSessionLock(path.join(sessionsDir, SESSION_INDEX_FILE), () => {
    const index = readSessionIndex(sessionsDir) ?? rebuildSessionIndexUnlocked(sessionsDir);
    index.set(entry.sessionId, entry);
    writeSessionIndex(index.values(), sessionsDir);
  });
}

function readOrRebuildSessionIndex(sessionsDir: string): Map<string, SessionIndexEntry> {
  return withSessionLock(path.join(sessionsDir, SESSION_INDEX_FILE), () => {
    return readSessionIndex(sessionsDir) ?? rebuildSessionIndexUnlocked(sessionsDir);
  });
}

function readSessionIndex(sessionsDir: string): Map<string, SessionIndexEntry> | null {
  const indexPath = path.join(sessionsDir, SESSION_INDEX_FILE);
  if (!fs.existsSync(indexPath)) return null;
  try {
    const document = objectRecord(JSON.parse(fs.readFileSync(indexPath, "utf8")), "Session index");
    if (document.version !== SESSION_SCHEMA_VERSION || !Array.isArray(document.entries)) {
      throw new Error("Unsupported Session index document.");
    }
    return new Map(
      document.entries.map((entry) => {
        const parsed = parseSessionIndexEntry(entry);
        return [parsed.sessionId, parsed];
      }),
    );
  } catch {
    return null;
  }
}

function rebuildSessionIndex(sessionsDir: string): Map<string, SessionIndexEntry> {
  return withSessionLock(path.join(sessionsDir, SESSION_INDEX_FILE), () =>
    rebuildSessionIndexUnlocked(sessionsDir),
  );
}

function rebuildSessionIndexUnlocked(sessionsDir: string): Map<string, SessionIndexEntry> {
  const entries: SessionIndexEntry[] = [];
  for (const source of discoverSessionSources(sessionsDir)) {
    try {
      const session = readSessionLog(source.path);
      const sourceStat = fs.statSync(source.path);
      entries.push({
        sessionId: session.id,
        sourceBytes: sourceStat.size,
        sourceMtimeMs: sourceStat.mtimeMs,
        sourceFormat: "jsonl",
        summary: sessionSummary(session),
      });
    } catch {
      // A derived index can omit corrupt Sessions; canonical files remain untouched.
    }
  }

  const index = new Map(entries.map((entry) => [entry.sessionId, entry]));
  writeSessionIndex(index.values(), sessionsDir);
  return index;
}

function writeSessionIndex(entries: Iterable<SessionIndexEntry>, sessionsDir: string): void {
  const indexPath = path.join(sessionsDir, SESSION_INDEX_FILE);
  const temporaryPath = `${indexPath}.tmp-${process.pid}-${uuidv4()}`;
  const document: SessionIndexDocument = {
    version: SESSION_SCHEMA_VERSION,
    entries: [...entries].sort((left, right) => left.sessionId.localeCompare(right.sessionId)),
  };
  writeNewFileDurably(temporaryPath, `${JSON.stringify(document, null, 2)}\n`);
  fs.renameSync(temporaryPath, indexPath);
  fs.chmodSync(indexPath, 0o600);
}

function parseSessionIndexEntry(input: unknown): SessionIndexEntry {
  const record = objectRecord(input, "Session index entry");
  if (typeof record.sessionId !== "string") {
    throw new Error("Invalid Session index entry.");
  }
  if (
    record.sourceFormat !== "jsonl" ||
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
    sessionId: record.sessionId,
    sourceBytes: record.sourceBytes,
    sourceMtimeMs: record.sourceMtimeMs,
    sourceFormat: record.sourceFormat,
    summary,
  };
}

function discoverSessionSources(sessionsDir: string): SessionSource[] {
  const sources: SessionSource[] = [];
  if (!fs.existsSync(sessionsDir) || !fs.statSync(sessionsDir).isDirectory()) return sources;
  for (const entry of fs.readdirSync(sessionsDir)) {
    if (entry === SESSION_INDEX_FILE || entry === LEGACY_SESSION_INDEX_FILE) continue;
    const extension = path.extname(entry);
    if (extension !== ".jsonl") continue;
    const id = entry.slice(0, -extension.length);
    if (!id) continue;
    sources.push({ id, path: path.join(sessionsDir, entry), directory: sessionsDir });
  }
  return sources.sort((left, right) => left.id.localeCompare(right.id));
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
