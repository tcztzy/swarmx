import type {
  DesktopSessionData as SessionData,
  DesktopSessionSummary as SessionSummary,
} from "../../shared/desktop-api.js";
import { HARNESSES } from "./harness-presentation.js";
import { errorMessage, projectName } from "./text-utils.js";

export type SessionGroupMode = "project" | "harness";
export type ProjectOrganizationMode = "project" | "list";
export type ProjectSortMode = "priority" | "last-updated" | "manual";

export interface DiscoveredSession {
  id: string;
  title: string;
  projectId?: string;
  cwd: string;
  pinned?: boolean;
  updatedAt?: string;
  harnessId: string;
  harnessLabel: string;
  source: "local" | "acp";
}

export interface SessionGroup {
  id: string;
  label: string;
  sessions: DiscoveredSession[];
}

export interface ProjectData {
  id: string;
  name: string;
  cwd: string;
  pinned: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface ProjectSessionGroup extends SessionGroup {
  project?: ProjectData;
  cwd: string;
}

export interface ProjectPreviewState {
  projectId: string;
  top: number;
  left: number;
}

export interface SessionDiscoveryError {
  harnessId: string;
  harnessLabel: string;
  message: string;
}

export interface GroupedSessionsResult {
  mode: SessionGroupMode;
  groups: SessionGroup[];
  errors: SessionDiscoveryError[];
}

export interface SessionContextMenuState {
  session: DiscoveredSession;
  x: number;
  y: number;
}

const LOCAL_SESSION_PRELOAD_LIMIT = 24;
export const RECENTS_GROUP_ID = "__no_project__";

export function buildSessionErrors(
  discoveredErrors: SessionDiscoveryError[],
  localError: unknown,
  groupedError: unknown,
): SessionDiscoveryError[] {
  const errors = [...discoveredErrors];
  if (localError) {
    errors.push({
      harnessId: "local-sessions",
      harnessLabel: "Local Sessions",
      message: errorMessage(localError),
    });
  }
  if (groupedError) {
    errors.push({
      harnessId: "acp-sessions",
      harnessLabel: "ACP Sessions",
      message: errorMessage(groupedError),
    });
  }
  return errors;
}

export function sessionDetailKey(
  session: DiscoveredSession,
): readonly ["session:detail", string, string, string, string] {
  return ["session:detail", session.source, session.harnessId, session.id, session.cwd];
}

export function sessionCacheId(session: DiscoveredSession): string {
  return sessionDetailKey(session).join("\u001f");
}

function flattenSessions(groups: SessionGroup[]): DiscoveredSession[] {
  return groups.flatMap((group) => group.sessions);
}

export function preloadSessionCandidates(groups: SessionGroup[]): DiscoveredSession[] {
  return flattenSessions(groups)
    .filter((session) => session.source === "local")
    .slice(0, LOCAL_SESSION_PRELOAD_LIMIT);
}

export function mergeLocalSessionsIntoGroups(
  groups: SessionGroup[],
  sessions: SessionSummary[],
  mode: SessionGroupMode,
): SessionGroup[] {
  const externalSessions = groups
    .flatMap((group) => group.sessions)
    .filter((session) => session.source !== "local");
  const localSessions = sessions.map(localSessionToDiscovered);
  return groupDisplaySessions([...externalSessions, ...localSessions], mode);
}

export function mergeProjectsIntoSessionGroups(
  projects: ProjectData[],
  groups: SessionGroup[],
): ProjectSessionGroup[] {
  const remaining = new Set(groups.map((group) => group.id));
  const projectGroups = projects.map<ProjectSessionGroup>((project) => {
    const matchingGroups = groups.filter((group) =>
      group.sessions.some(
        (session) => session.projectId === project.id || sameProjectPath(session.cwd, project.cwd),
      ),
    );
    for (const group of matchingGroups) remaining.delete(group.id);
    return {
      id: project.id,
      label: project.name,
      cwd: project.cwd,
      project,
      sessions: sortDisplaySessions(matchingGroups.flatMap((group) => group.sessions)),
    };
  });

  const unmatched = groups
    .filter((group) => remaining.has(group.id))
    .flatMap<ProjectSessionGroup>((group) => {
      const sessions = group.sessions;
      if (sessions.length === 0) return [];
      return [
        {
          ...group,
          label: group.id === RECENTS_GROUP_ID ? "Recents" : projectDisplayName(group.label),
          cwd: group.id === RECENTS_GROUP_ID ? "" : group.label,
          sessions,
        },
      ];
    });
  return [...projectGroups, ...unmatched];
}

export function sortProjectSessionGroups(
  groups: ProjectSessionGroup[],
  mode: ProjectSortMode,
): ProjectSessionGroup[] {
  const projectGroups = groups.filter((group) => group.id !== RECENTS_GROUP_ID);
  const recentsGroups = groups.filter((group) => group.id === RECENTS_GROUP_ID);
  if (mode === "manual") return [...projectGroups, ...recentsGroups];
  return projectGroups
    .sort((left, right) => {
      if (mode === "priority" && Boolean(left.project?.pinned) !== Boolean(right.project?.pinned)) {
        return left.project?.pinned ? -1 : 1;
      }
      const timeDifference = projectSessionGroupTime(right) - projectSessionGroupTime(left);
      return timeDifference || left.label.localeCompare(right.label);
    })
    .concat(recentsGroups);
}

export function flattenProjectSessions(
  groups: ProjectSessionGroup[],
  mode: ProjectSortMode,
): DiscoveredSession[] {
  const entries = groups.flatMap((group) =>
    group.sessions.map((session) => ({
      session,
      pinned: Boolean(group.project?.pinned),
    })),
  );
  if (mode === "manual") return entries.map(({ session }) => session);
  return entries
    .sort((left, right) => {
      if (mode === "priority" && left.pinned !== right.pinned) return left.pinned ? -1 : 1;
      return sessionTime(right.session.updatedAt) - sessionTime(left.session.updatedAt);
    })
    .map(({ session }) => session);
}

function projectSessionGroupTime(group: ProjectSessionGroup): number {
  return Math.max(
    sessionTime(group.project?.updatedAt),
    ...group.sessions.map((session) => sessionTime(session.updatedAt)),
  );
}

export function filterSessionGroups(
  groups: ProjectSessionGroup[],
  query: string,
): ProjectSessionGroup[] {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) return groups;

  return groups.flatMap((group) => {
    if (group.label.toLowerCase().includes(normalizedQuery)) return [group];
    const sessions = group.sessions.filter((session) =>
      `${session.title} ${session.harnessLabel} ${session.cwd}`
        .toLowerCase()
        .includes(normalizedQuery),
    );
    return sessions.length > 0 ? [{ ...group, sessions }] : [];
  });
}

export function projectDisplayName(value: string): string {
  const normalized = value.trim().replace(/[\\/]+$/, "");
  const label = normalized.split(/[\\/]/).filter(Boolean).at(-1) ?? normalized;
  return label || "this project";
}

export function abbreviateHomePath(value: string): string {
  return value
    .replace(/^\/Users\/[^/]+(?=\/|$)/, "~")
    .replace(/^\/home\/[^/]+(?=\/|$)/, "~")
    .replace(/^[A-Za-z]:\\Users\\[^\\]+(?=\\|$)/, "~");
}

export function sameProjectPath(left?: string, right?: string): boolean {
  if (!left?.trim() || !right?.trim()) return false;
  const normalize = (value: string) => value.trim().replace(/[\\/]+$/, "");
  return normalize(left) === normalize(right);
}

export function navigationEntryKey(session: DiscoveredSession | null): string {
  return session ? sessionCacheId(session) : "__new_session__";
}

export function localSessionToDiscovered(session: SessionData | SessionSummary): DiscoveredSession {
  const harness = HARNESSES.find((item) => item.id === session.harness);

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

function groupDisplaySessions(
  sessions: DiscoveredSession[],
  mode: SessionGroupMode,
): SessionGroup[] {
  const grouped = new Map<string, SessionGroup>();

  for (const session of sortDisplaySessions(sessions)) {
    const project = session.cwd.trim();
    const groupId = mode === "harness" ? session.harnessId : project || RECENTS_GROUP_ID;
    const groupLabel = mode === "harness" ? session.harnessLabel : project || "Recents";
    const existing = grouped.get(groupId);
    if (existing) {
      existing.sessions.push(session);
    } else {
      grouped.set(groupId, { id: groupId, label: groupLabel, sessions: [session] });
    }
  }

  return [...grouped.values()];
}

function sortDisplaySessions(sessions: DiscoveredSession[]): DiscoveredSession[] {
  return [...sessions].sort(
    (a, b) =>
      Number(Boolean(b.pinned)) - Number(Boolean(a.pinned)) ||
      sessionTime(b.updatedAt) - sessionTime(a.updatedAt),
  );
}

export function isPlaceholderSessionTitle(title: string): boolean {
  return ["", "new session", "untitled"].includes(title.trim().toLocaleLowerCase());
}

function sessionTime(value?: string): number {
  if (!value) return 0;
  const time = Date.parse(value);
  return Number.isFinite(time) ? time : 0;
}

export function sessionMeta(session: DiscoveredSession, mode: SessionGroupMode): string {
  const date = formatSessionDate(session.updatedAt);
  if (mode === "project") {
    return `${session.harnessLabel} - ${date}`;
  }
  if (session.cwd.trim()) {
    return `${projectName(session.cwd)} - ${date}`;
  }
  return `${session.source === "local" ? "Local" : "ACP"} - ${date}`;
}

export function harnessLabel(id: string): string {
  return HARNESSES.find((harness) => harness.id === id)?.label ?? id;
}

function formatSessionDate(value?: string): string {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
