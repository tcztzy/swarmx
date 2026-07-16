import type { SessionInfo as AcpSessionInfo } from "@agentclientprotocol/sdk";
import { AcpClient, type AcpClientOptions } from "./acp.js";
import { HARNESSES } from "./harness.js";
import { listSessions as listLocalSessions, loadSession as loadLocalSession } from "./session.js";
import { type MessageChunk, type SessionData, SessionDataSchema } from "./types.js";

export type SessionGroupMode = "project" | "harness";
export type DiscoveredSessionSource = "local" | "acp";

export interface DiscoveredSession {
  id: string;
  title: string;
  projectId?: string;
  cwd: string;
  pinned?: boolean;
  updatedAt?: string;
  harnessId: string;
  harnessLabel: string;
  source: DiscoveredSessionSource;
}

export interface SessionDiscoveryError {
  harnessId: string;
  harnessLabel: string;
  message: string;
}

export interface SessionGroup {
  id: string;
  label: string;
  sessions: DiscoveredSession[];
}

export interface GroupedSessionsResult {
  mode: SessionGroupMode;
  groups: SessionGroup[];
  errors: SessionDiscoveryError[];
}

export interface ListGroupedSessionsOptions {
  mode?: SessionGroupMode;
  cwd?: string;
  harnessIds?: string[];
  timeoutMs?: number;
}

export interface LoadDiscoveredSessionOptions {
  createClient?: () => AcpSessionLoader;
  timeoutMs?: number;
}

interface AcpSessionLoader {
  loadSession(
    opts: AcpClientOptions,
    sessionId: string,
    cwd: string,
  ): Promise<{ messages: MessageChunk[] }>;
  stderrOutput(): string;
  kill(): void;
}

const DEFAULT_ACP_SESSION_HARNESSES = ["claude_code", "codex"] as const;
const DEFAULT_SESSION_TIMEOUT_MS = 30_000;
const UNGROUPED_PROJECT_ID = "__no_project__";
const UNGROUPED_PROJECT_LABEL = "No project";

export async function listGroupedSessions(
  options: ListGroupedSessionsOptions = {},
): Promise<GroupedSessionsResult> {
  const mode = options.mode ?? "harness";
  const sessions: DiscoveredSession[] = listLocalSessions().map(localSessionToDiscovered);
  const errors: SessionDiscoveryError[] = [];

  const harnessIds = options.harnessIds ?? [...DEFAULT_ACP_SESSION_HARNESSES];
  const acpResults = await Promise.all(
    harnessIds.map((harnessId) =>
      listAcpHarnessSessions(harnessId, {
        cwd: options.cwd,
        timeoutMs: options.timeoutMs ?? DEFAULT_SESSION_TIMEOUT_MS,
      }),
    ),
  );

  for (const result of acpResults) {
    sessions.push(...result.sessions);
    if (result.error) errors.push(result.error);
  }

  return {
    mode,
    groups: groupDiscoveredSessions(sortSessions(sessions), mode),
    errors,
  };
}

export function groupDiscoveredSessions(
  sessions: DiscoveredSession[],
  mode: SessionGroupMode,
): SessionGroup[] {
  const grouped = new Map<string, SessionGroup>();

  for (const session of sortSessions(sessions)) {
    const groupId =
      mode === "harness"
        ? session.harnessId
        : session.cwd.trim()
          ? session.cwd
          : UNGROUPED_PROJECT_ID;
    const groupLabel =
      mode === "harness"
        ? session.harnessLabel
        : session.cwd.trim()
          ? session.cwd
          : UNGROUPED_PROJECT_LABEL;

    const group = grouped.get(groupId);
    if (group) {
      group.sessions.push(session);
    } else {
      grouped.set(groupId, { id: groupId, label: groupLabel, sessions: [session] });
    }
  }

  return [...grouped.values()].map((group) => ({
    ...group,
    sessions: sortSessions(group.sessions),
  }));
}

export function acpSessionToDiscovered(
  session: AcpSessionInfo,
  harnessId: string,
): DiscoveredSession | null {
  const alternateSession = session as AcpSessionInfo & {
    session_id?: string | null;
    updated_at?: string | null;
  };
  const id = session.sessionId ?? alternateSession.session_id;
  if (!id) return null;

  const harness = HARNESSES[harnessId];
  const title = session.title?.trim() || id;

  return {
    id,
    title,
    cwd: session.cwd ?? "",
    updatedAt: session.updatedAt ?? alternateSession.updated_at ?? undefined,
    harnessId,
    harnessLabel: harness?.label ?? harnessId,
    source: "acp",
  };
}

export async function loadDiscoveredSession(
  session: DiscoveredSession,
  options: LoadDiscoveredSessionOptions = {},
): Promise<SessionData | null> {
  if (session.source === "local") {
    return loadLocalSession(session.id);
  }

  const harness = HARNESSES[session.harnessId];
  if (!harness || harness.enabled === false) {
    throw new Error(`Unknown harness: ${session.harnessId}`);
  }
  if (harness.backend.type !== "custom") {
    throw new Error(`Harness does not support ACP session loading: ${session.harnessId}`);
  }

  const client = options.createClient?.() ?? new AcpClient();
  const cwd = session.cwd.trim() || process.cwd();
  const opts: AcpClientOptions = {
    command: harness.backend.program,
    args: harness.backend.args ?? [],
    cwd,
  };

  try {
    const loaded = await withTimeout<{ messages: MessageChunk[] }>(
      client.loadSession(opts, session.id, cwd),
      options.timeoutMs ?? DEFAULT_SESSION_TIMEOUT_MS,
      `${harness.label} session loading timed out`,
    );
    return acpLoadedSessionToSessionData(session, loaded.messages);
  } catch (err) {
    throw new Error(acpErrorMessage(err, client.stderrOutput()));
  } finally {
    client.kill();
  }
}

export function acpLoadedSessionToSessionData(
  session: DiscoveredSession,
  messages: MessageChunk[],
): SessionData {
  const updatedAt = session.updatedAt ?? new Date().toISOString();

  return SessionDataSchema.parse({
    id: session.id,
    title: session.title || session.id,
    ...(session.projectId ? { projectId: session.projectId } : {}),
    ...(session.cwd ? { cwd: session.cwd } : {}),
    acpSessionId: session.id,
    agentName: session.harnessLabel || session.harnessId,
    harness: session.harnessId,
    messages,
    createdAt: updatedAt,
    updatedAt,
  });
}

function localSessionToDiscovered(session: SessionData): DiscoveredSession {
  const harness = HARNESSES[session.harness];

  return {
    id: session.id,
    title: session.title || "Untitled",
    ...(session.projectId ? { projectId: session.projectId } : {}),
    cwd: session.cwd ?? "",
    pinned: session.pinned,
    updatedAt: session.updatedAt,
    harnessId: session.harness,
    harnessLabel: harness?.label ?? session.harness,
    source: "local",
  };
}

async function listAcpHarnessSessions(
  harnessId: string,
  options: { cwd?: string; timeoutMs: number },
): Promise<{ sessions: DiscoveredSession[]; error?: SessionDiscoveryError }> {
  const harness = HARNESSES[harnessId];
  if (!harness || harness.enabled === false) {
    return {
      sessions: [],
      error: { harnessId, harnessLabel: harnessId, message: `Unknown harness: ${harnessId}` },
    };
  }
  if (harness.backend.type !== "custom") {
    return { sessions: [] };
  }

  const client = new AcpClient();
  const opts: AcpClientOptions = {
    command: harness.backend.program,
    args: harness.backend.args ?? [],
    cwd: options.cwd ?? process.cwd(),
  };

  try {
    const acpSessions = await withTimeout(
      client.listSessions(opts, options.cwd),
      options.timeoutMs,
      `${harness.label} session listing timed out`,
    );
    return {
      sessions: acpSessions
        .map((session) => acpSessionToDiscovered(session, harnessId))
        .filter((session): session is DiscoveredSession => session !== null),
    };
  } catch (err) {
    return {
      sessions: [],
      error: {
        harnessId,
        harnessLabel: harness.label,
        message: acpErrorMessage(err, client.stderrOutput()),
      },
    };
  } finally {
    client.kill();
  }
}

function acpErrorMessage(err: unknown, stderr: string): string {
  const message = err instanceof Error ? err.message : String(err);
  const normalized = message.includes("EPIPE")
    ? "ACP process exited before responding to session/list"
    : message;
  return stderr ? `${normalized}: ${stderr}` : normalized;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), timeoutMs);
    promise
      .then(resolve)
      .catch(reject)
      .finally(() => clearTimeout(timer));
  });
}

function sortSessions(sessions: DiscoveredSession[]): DiscoveredSession[] {
  return [...sessions].sort(
    (a, b) =>
      Number(Boolean(b.pinned)) - Number(Boolean(a.pinned)) ||
      timestamp(b.updatedAt) - timestamp(a.updatedAt),
  );
}

function timestamp(value?: string): number {
  if (!value) return 0;
  const time = Date.parse(value);
  return Number.isFinite(time) ? time : 0;
}
