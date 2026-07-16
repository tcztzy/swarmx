import * as fs from "node:fs";
import { homedir } from "node:os";
import * as path from "node:path";
import { v4 as uuidv4 } from "uuid";
import { type MessageChunk, type SessionData, SessionDataSchema } from "./types.js";

const SESSIONS_DIR = path.join(homedir(), ".swarmx", "sessions");

function ensureSessionsDir(): string {
  if (!fs.existsSync(SESSIONS_DIR)) {
    fs.mkdirSync(SESSIONS_DIR, { recursive: true });
  }
  return SESSIONS_DIR;
}

export interface SessionProjectContext {
  projectId?: string;
  cwd?: string;
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
    messages: [],
    createdAt: now,
    updatedAt: now,
  });
}

export function saveSession(session: SessionData): void {
  ensureSessionsDir();
  session.updatedAt = new Date().toISOString();
  const filePath = path.join(SESSIONS_DIR, `${session.id}.json`);
  fs.writeFileSync(filePath, JSON.stringify(session, null, 2));
}

export function loadSession(id: string): SessionData | null {
  ensureSessionsDir();
  const filePath = path.join(SESSIONS_DIR, `${id}.json`);
  if (!fs.existsSync(filePath)) return null;
  try {
    const data = fs.readFileSync(filePath, "utf-8");
    const parsed = SessionDataSchema.safeParse(JSON.parse(data));
    return parsed.success ? parsed.data : null;
  } catch {
    return null;
  }
}

export interface ListSessionsOptions {
  includeArchived?: boolean;
}

export function listSessions(options: ListSessionsOptions = {}): SessionData[] {
  ensureSessionsDir();
  const sessions: SessionData[] = [];
  try {
    for (const entry of fs.readdirSync(SESSIONS_DIR)) {
      if (!entry.endsWith(".json")) continue;
      const filePath = path.join(SESSIONS_DIR, entry);
      try {
        const data = fs.readFileSync(filePath, "utf-8");
        const parsed = SessionDataSchema.safeParse(JSON.parse(data));
        if (parsed.success && (options.includeArchived || !parsed.data.archivedAt)) {
          sessions.push(parsed.data);
        }
      } catch {
        // skip corrupt files
      }
    }
  } catch {
    // directory doesn't exist
  }
  sessions.sort(
    (a, b) =>
      Number(b.pinned) - Number(a.pinned) ||
      new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
  );
  return sessions;
}

export function archiveProjectSessions(project: SessionProjectContext): number {
  if (!project.projectId && !project.cwd) return 0;
  const archivedAt = new Date().toISOString();
  let archived = 0;
  for (const session of listSessions({ includeArchived: true })) {
    if (session.archivedAt || !belongsToProject(session, project)) continue;
    session.archivedAt = archivedAt;
    saveSession(session);
    archived += 1;
  }
  return archived;
}

export function deleteSession(id: string): boolean {
  ensureSessionsDir();
  const filePath = path.join(SESSIONS_DIR, `${id}.json`);
  if (!fs.existsSync(filePath)) return false;
  fs.unlinkSync(filePath);
  return true;
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
  const session = loadSession(id);
  if (!session) return false;
  session.messages.push(...messages);
  saveSession(session);
  return true;
}

function belongsToProject(session: SessionData, project: SessionProjectContext): boolean {
  if (project.projectId && session.projectId === project.projectId) return true;
  if (!project.cwd || !session.cwd) return false;
  return path.resolve(session.cwd) === path.resolve(project.cwd);
}
