import {
  Bot,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Code2,
  GitBranch,
  Hammer,
  Loader2,
  type LucideIcon,
  MessageSquarePlus,
  Play,
  SendHorizontal,
  Sparkles,
  Terminal,
  Trash2,
  User,
  Workflow,
  Wrench,
  XCircle,
} from "lucide-react";
import type React from "react";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import type { SwarmxAPI } from "../../preload/index.js";
import { MessageContent } from "./message-content.js";

interface MessageChunk {
  role: string;
  content: string;
  kind: "message" | "thinking" | "tool_call" | "tool_result";
  agent?: string;
  swarmEvent?: string;
  toolName?: string;
}

interface SessionData {
  id: string;
  title: string;
  acpSessionId?: string;
  agentName: string;
  harness: string;
  model?: string;
  messages: MessageChunk[];
  createdAt: string;
  updatedAt: string;
}

type SessionGroupMode = "project" | "harness";

interface DiscoveredSession {
  id: string;
  title: string;
  cwd: string;
  updatedAt?: string;
  harnessId: string;
  harnessLabel: string;
  source: "local" | "acp";
}

interface SessionGroup {
  id: string;
  label: string;
  sessions: DiscoveredSession[];
}

interface SessionDiscoveryError {
  harnessId: string;
  harnessLabel: string;
  message: string;
}

interface GroupedSessionsResult {
  mode: SessionGroupMode;
  groups: SessionGroup[];
  errors: SessionDiscoveryError[];
}

interface HarnessOption {
  id: string;
  label: string;
  icon: LucideIcon;
}

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "secondary" | "ghost" | "destructive";
  size?: "sm" | "md" | "icon";
}

declare global {
  interface Window {
    swarmxAPI: SwarmxAPI;
  }
}

const api = window.swarmxAPI;
const LOCAL_SESSIONS_KEY = "sessions:local";
const GROUPED_SESSIONS_KEY = "sessions:grouped";
const SESSION_DEDUPING_INTERVAL_MS = 10_000;
const LOCAL_SESSION_PRELOAD_LIMIT = 24;

const HARNESSES: HarnessOption[] = [
  { id: "swarmx", label: "SwarmX", icon: Workflow },
  { id: "claude_code", label: "Claude Code", icon: Hammer },
  { id: "codex", label: "Codex", icon: Terminal },
  { id: "opencode", label: "OpenCode", icon: Code2 },
  { id: "hermes", label: "Hermes", icon: Sparkles },
  { id: "openclaw", label: "OpenClaw", icon: Wrench },
];

export function App() {
  const [sessionGroupMode, setSessionGroupMode] = useState<SessionGroupMode>("harness");
  const [currentSession, setCurrentSession] = useState<SessionData | null>(null);
  const [selectedDiscoveredSession, setSelectedDiscoveredSession] =
    useState<DiscoveredSession | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedHarness, setSelectedHarness] = useState("swarmx");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const chatRef = useRef<HTMLDivElement>(null);
  const preloadedSessionKeys = useRef(new Set<string>());
  const scrollStateRef = useRef<{ sessionId: string | null; messageCount: number }>({
    sessionId: null,
    messageCount: 0,
  });
  const { mutate: mutateSessionDetail } = useSWRConfig();
  const messageCount = currentSession?.messages.length ?? 0;

  const {
    data: sessions = [],
    error: localSessionsError,
    isLoading: localSessionsLoading,
    mutate: mutateLocalSessions,
  } = useSWR<SessionData[]>(
    LOCAL_SESSIONS_KEY,
    () => api.listSessions() as Promise<SessionData[]>,
    {
      dedupingInterval: SESSION_DEDUPING_INTERVAL_MS,
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );

  const {
    data: groupedSessions,
    error: groupedSessionsError,
    isLoading: groupedSessionsLoading,
  } = useSWR<GroupedSessionsResult>(
    GROUPED_SESSIONS_KEY,
    () => api.listGroupedSessions({ mode: "harness" }) as Promise<GroupedSessionsResult>,
    {
      dedupingInterval: SESSION_DEDUPING_INTERVAL_MS,
      keepPreviousData: true,
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );

  const activeHarness = useMemo(
    () => HARNESSES.find((harness) => harness.id === selectedHarness) ?? HARNESSES[0],
    [selectedHarness],
  );
  const sessionGroups = groupedSessions?.groups ?? [];
  const sessionErrors = useMemo(
    () =>
      buildSessionErrors(groupedSessions?.errors ?? [], localSessionsError, groupedSessionsError),
    [groupedSessions?.errors, localSessionsError, groupedSessionsError],
  );
  const sessionsLoading =
    (localSessionsLoading && sessions.length === 0) || (groupedSessionsLoading && !groupedSessions);
  const displayGroups = useMemo(
    () => mergeLocalSessionsIntoGroups(sessionGroups, sessions, sessionGroupMode),
    [sessionGroups, sessions, sessionGroupMode],
  );
  const selectedSessionKey = selectedDiscoveredSession
    ? sessionDetailKey(selectedDiscoveredSession)
    : null;
  const {
    data: selectedSessionData,
    error: selectedSessionError,
    isLoading: selectedSessionLoading,
  } = useSWR<SessionData | null>(
    selectedSessionKey,
    () =>
      selectedDiscoveredSession
        ? loadDiscoveredSessionDetail(selectedDiscoveredSession)
        : Promise.resolve(null),
    {
      keepPreviousData: false,
      revalidateIfStale: false,
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );
  const visibleSessionErrors = useMemo(() => {
    if (!selectedSessionError) return sessionErrors;
    return [
      ...sessionErrors,
      {
        harnessId: "session-load",
        harnessLabel: "Session Load",
        message: errorMessage(selectedSessionError),
      },
    ];
  }, [selectedSessionError, sessionErrors]);
  const runTitle = currentSession?.title || "SwarmX";
  const runSubtitle = currentSession
    ? `${currentSession.agentName} on ${harnessLabel(currentSession.harness)}`
    : `${activeHarness.label} ready`;

  const prefetchSession = useCallback(
    (session: DiscoveredSession) => {
      const cacheId = sessionCacheId(session);
      if (preloadedSessionKeys.current.has(cacheId)) return;
      preloadedSessionKeys.current.add(cacheId);
      void loadDiscoveredSessionDetail(session)
        .then((data) => {
          if (data) {
            void mutateSessionDetail(sessionDetailKey(session), data, {
              populateCache: true,
              revalidate: false,
            });
          }
        })
        .catch(() => {
          preloadedSessionKeys.current.delete(cacheId);
        });
    },
    [mutateSessionDetail],
  );

  useEffect(() => {
    for (const session of preloadSessionCandidates(displayGroups)) {
      prefetchSession(session);
    }
  }, [displayGroups, prefetchSession]);

  useEffect(() => {
    if (!selectedSessionData) return;
    setCurrentSession(selectedSessionData);
    setSelectedHarness(selectedSessionData.harness);
  }, [selectedSessionData]);

  useLayoutEffect(() => {
    const chat = chatRef.current;
    const sessionId = currentSession?.id ?? null;
    const previous = scrollStateRef.current;

    scrollStateRef.current = { sessionId, messageCount };

    if (!chat || messageCount === 0) return;

    const sessionChanged = sessionId !== previous.sessionId;
    const messageAdded = sessionId !== null && messageCount > previous.messageCount;

    chat.scrollTo({
      top: chat.scrollHeight,
      behavior: sessionChanged || !messageAdded ? "auto" : "smooth",
    });
  }, [currentSession?.id, messageCount]);

  const newSession = useCallback(async () => {
    const session = await api.createSession({
      agentName: "agent",
      harness: selectedHarness,
    });
    await api.saveSession(session);
    await mutateLocalSessions();
    setCurrentSession(session);
  }, [mutateLocalSessions, selectedHarness]);

  const selectSession = useCallback((session: DiscoveredSession) => {
    setSelectedDiscoveredSession(session);
  }, []);

  const deleteCurrentSession = useCallback(async () => {
    if (!currentSession) return;
    await api.deleteSession(currentSession.id);
    await mutateLocalSessions();
    setCurrentSession(null);
  }, [currentSession, mutateLocalSessions]);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setLoading(true);

    try {
      const userChunk: MessageChunk = {
        role: "user",
        content: text,
        kind: "message",
      };

      let session: SessionData;
      if (currentSession) {
        session = currentSession;
      } else {
        session = (await api.createSession({
          agentName: "agent",
          harness: selectedHarness,
        })) as SessionData;
        await api.saveSession(session);
      }

      const updatedMessages = [...session.messages, userChunk];
      setCurrentSession({ ...session, messages: updatedMessages });

      const result = await api.sendMessage({
        harnessId: selectedHarness,
        userText: text,
      });

      if (result.success && result.messages) {
        const responseMessages = result.messages as MessageChunk[];
        const updated = { ...session, messages: [...updatedMessages, ...responseMessages] };
        await api.saveSession(updated);
        setCurrentSession(updated);
      } else if (result.error) {
        const errorMsg: MessageChunk = {
          role: "system",
          content: `Error: ${result.error}`,
          kind: "message",
        };
        const updated = {
          ...session,
          messages: [...updatedMessages, errorMsg],
        };
        await api.saveSession(updated);
        setCurrentSession(updated);
      }

      await mutateLocalSessions();
    } finally {
      setLoading(false);
    }
  }, [input, loading, currentSession, selectedHarness, mutateLocalSessions]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        void sendMessage();
      }
    },
    [sendMessage],
  );

  return (
    <div className={cx("app-shell", !sidebarOpen && "app-shell--collapsed")}>
      <aside className="sidebar" aria-label="Sessions">
        <div className="sidebar__brand">
          <div className="brand-mark">
            <Workflow aria-hidden="true" />
          </div>
          <div className="brand-copy">
            <div className="brand-title">SwarmX</div>
            <div className="brand-subtitle">agent runtime</div>
          </div>
        </div>

        <div className="sidebar__controls">
          <label className="select-shell">
            <activeHarness.icon aria-hidden="true" />
            <select
              value={selectedHarness}
              onChange={(e) => setSelectedHarness(e.target.value)}
              className="select-control"
              aria-label="Harness"
            >
              {HARNESSES.map((harness) => (
                <option key={harness.id} value={harness.id}>
                  {harness.label}
                </option>
              ))}
            </select>
          </label>
          <Button onClick={newSession} size="sm">
            <MessageSquarePlus data-icon="inline-start" aria-hidden="true" />
            New
          </Button>
        </div>

        <div className="segmented-control" role="tablist" aria-label="Session grouping">
          <button
            type="button"
            role="tab"
            aria-selected={sessionGroupMode === "harness"}
            onClick={() => setSessionGroupMode("harness")}
            className={cx("segmented-control__item", sessionGroupMode === "harness" && "is-active")}
          >
            Harness
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={sessionGroupMode === "project"}
            onClick={() => setSessionGroupMode("project")}
            className={cx("segmented-control__item", sessionGroupMode === "project" && "is-active")}
          >
            Project
          </button>
        </div>

        <div className="session-scroll">
          {sessionsLoading && <div className="session-status">Loading sessions</div>}
          {!sessionsLoading && displayGroups.length === 0 && (
            <div className="session-status">No sessions</div>
          )}
          {displayGroups.map((group) => (
            <section key={group.id} className="session-group" aria-label={group.label}>
              <div className="session-group__header">
                <span>{group.label}</span>
                <span>{group.sessions.length}</span>
              </div>
              <div className="session-group__items">
                {group.sessions.map((session) => {
                  const isLocal = session.source === "local";
                  const isActive =
                    currentSession?.id === session.id &&
                    currentSession.harness === session.harnessId;
                  const isPending =
                    selectedSessionLoading &&
                    selectedDiscoveredSession !== null &&
                    sessionCacheId(selectedDiscoveredSession) === sessionCacheId(session);
                  return (
                    <button
                      type="button"
                      key={`${session.source}:${session.harnessId}:${session.id}`}
                      onFocus={() => prefetchSession(session)}
                      onPointerEnter={() => prefetchSession(session)}
                      onClick={() => {
                        selectSession(session);
                      }}
                      className={cx(
                        "session-item",
                        isActive && "is-active",
                        isPending && "is-loading",
                      )}
                    >
                      <span className="session-item__icon">
                        {isPending ? (
                          <Loader2 aria-hidden="true" />
                        ) : isLocal ? (
                          <Clock3 aria-hidden="true" />
                        ) : (
                          <GitBranch aria-hidden="true" />
                        )}
                      </span>
                      <span className="session-item__body">
                        <span className="session-item__title">{session.title || "Untitled"}</span>
                        <span className="session-item__meta">
                          {sessionMeta(session, sessionGroupMode)}
                        </span>
                      </span>
                    </button>
                  );
                })}
              </div>
            </section>
          ))}
          {visibleSessionErrors.map((error) => (
            <div key={error.harnessId} className="session-error">
              <XCircle aria-hidden="true" />
              <span>
                {error.harnessLabel}: {error.message}
              </span>
            </div>
          ))}
        </div>
      </aside>

      <main className="runtime">
        <header className="runtime__header">
          <div className="runtime__titlebar">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              title={sidebarOpen ? "Collapse sidebar" : "Open sidebar"}
              aria-label={sidebarOpen ? "Collapse sidebar" : "Open sidebar"}
            >
              {sidebarOpen ? (
                <ChevronLeft data-icon aria-hidden="true" />
              ) : (
                <ChevronRight data-icon aria-hidden="true" />
              )}
            </Button>
            <div className="runtime__title">
              <h1>{runTitle}</h1>
              <p>{runSubtitle}</p>
            </div>
          </div>

          <div className="runtime__actions">
            <Badge tone={loading || selectedSessionLoading ? "active" : "neutral"}>
              {loading || selectedSessionLoading ? (
                <Loader2 data-icon="inline-start" aria-hidden="true" />
              ) : (
                <Play data-icon="inline-start" aria-hidden="true" />
              )}
              {selectedSessionLoading ? "Loading" : loading ? "Running" : "Ready"}
            </Badge>
            <Badge tone="neutral">{messageCount} events</Badge>
            {visibleSessionErrors.length > 0 && (
              <Badge tone="danger">{visibleSessionErrors.length} alerts</Badge>
            )}
            {currentSession && (
              <Button variant="ghost" size="icon" onClick={deleteCurrentSession} title="Delete run">
                <Trash2 data-icon aria-hidden="true" />
              </Button>
            )}
          </div>
        </header>

        <div ref={chatRef} className="transcript-scroll">
          <div className="transcript">
            {!currentSession || currentSession.messages.length === 0 ? (
              <EmptyRun activeHarness={activeHarness} onStart={newSession} />
            ) : (
              currentSession.messages.map((msg) => <RunEvent key={messageKey(msg)} msg={msg} />)
            )}
            {loading && (
              <div className="run-event run-event--thinking">
                <div className="run-event__rail">
                  <Loader2 aria-hidden="true" />
                </div>
                <div className="run-event__card">
                  <div className="run-event__header">
                    <span>agent</span>
                    <span>thinking</span>
                  </div>
                  <div className="run-event__content">Thinking</div>
                </div>
              </div>
            )}
          </div>
        </div>

        <footer className="composer-dock">
          <div className="composer">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Message ${activeHarness.label}`}
              className="composer__textarea"
              rows={2}
              disabled={loading}
            />
            <div className="composer__footer">
              <div className="composer__meta">
                <activeHarness.icon aria-hidden="true" />
                <span>{activeHarness.label}</span>
              </div>
              <Button onClick={sendMessage} disabled={loading || !input.trim()}>
                {loading ? (
                  <Loader2 data-icon="inline-start" aria-hidden="true" />
                ) : (
                  <SendHorizontal data-icon="inline-start" aria-hidden="true" />
                )}
                Send
              </Button>
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}

function EmptyRun({
  activeHarness,
  onStart,
}: { activeHarness: HarnessOption; onStart: () => void }) {
  return (
    <div className="empty-run">
      <div className="empty-run__mark">
        <activeHarness.icon aria-hidden="true" />
      </div>
      <div className="empty-run__copy">
        <h2>Start a SwarmX run</h2>
        <p>{activeHarness.label} is selected.</p>
      </div>
      <div className="empty-run__actions">
        <Button onClick={onStart}>
          <MessageSquarePlus data-icon="inline-start" aria-hidden="true" />
          New run
        </Button>
      </div>
    </div>
  );
}

function RunEvent({ msg }: { msg: MessageChunk }) {
  const { icon: Icon, label, tone, meta } = messagePresentation(msg);

  return (
    <article className={cx("run-event", `run-event--${tone}`)}>
      <div className="run-event__rail">
        <Icon aria-hidden="true" />
      </div>
      <div className="run-event__card">
        <div className="run-event__header">
          <span>{label}</span>
          <span>{meta}</span>
        </div>
        {msg.toolName && (
          <div className="run-event__tool">
            <Code2 aria-hidden="true" />
            <span>{msg.toolName}</span>
          </div>
        )}
        {msg.swarmEvent && <div className="run-event__event">{msg.swarmEvent}</div>}
        <div className="run-event__content">
          <MessageContent kind={msg.kind} content={msg.content} />
        </div>
      </div>
    </article>
  );
}

function Button({
  children,
  className,
  variant = "default",
  size = "md",
  type = "button",
  ...props
}: ButtonProps) {
  return (
    <button
      type={type}
      className={cx("button", `button--${variant}`, `button--${size}`, className)}
      {...props}
    >
      {children}
    </button>
  );
}

function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: string }) {
  return <span className={cx("badge", `badge--${tone}`)}>{children}</span>;
}

function messagePresentation(msg: MessageChunk): {
  icon: LucideIcon;
  label: string;
  tone: string;
  meta: string;
} {
  if (msg.role === "user") {
    return { icon: User, label: "you", tone: "user", meta: "message" };
  }
  if (msg.role === "system") {
    return { icon: XCircle, label: "system", tone: "system", meta: msg.kind };
  }
  if (msg.kind === "thinking") {
    return { icon: Loader2, label: msg.agent ?? "agent", tone: "thinking", meta: "thinking" };
  }
  if (msg.kind === "tool_call") {
    return { icon: Terminal, label: msg.agent ?? "tool", tone: "tool", meta: "call" };
  }
  if (msg.kind === "tool_result") {
    return { icon: Code2, label: msg.agent ?? "tool", tone: "tool", meta: "result" };
  }
  return { icon: Bot, label: msg.agent ?? "assistant", tone: "assistant", meta: "message" };
}

function messageKey(msg: MessageChunk): string {
  return [msg.role, msg.kind, msg.agent ?? "", msg.toolName ?? "", msg.content].join("\u001f");
}

function buildSessionErrors(
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

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function loadDiscoveredSessionDetail(session: DiscoveredSession): Promise<SessionData | null> {
  return api.loadDiscoveredSession(session) as Promise<SessionData | null>;
}

function sessionDetailKey(
  session: DiscoveredSession,
): readonly ["session:detail", string, string, string, string] {
  return ["session:detail", session.source, session.harnessId, session.id, session.cwd];
}

function sessionCacheId(session: DiscoveredSession): string {
  return sessionDetailKey(session).join("\u001f");
}

function flattenSessions(groups: SessionGroup[]): DiscoveredSession[] {
  return groups.flatMap((group) => group.sessions);
}

function preloadSessionCandidates(groups: SessionGroup[]): DiscoveredSession[] {
  return flattenSessions(groups)
    .filter((session) => session.source === "local")
    .slice(0, LOCAL_SESSION_PRELOAD_LIMIT);
}

function mergeLocalSessionsIntoGroups(
  groups: SessionGroup[],
  sessions: SessionData[],
  mode: SessionGroupMode,
): SessionGroup[] {
  const externalSessions = groups
    .flatMap((group) => group.sessions)
    .filter((session) => session.source !== "local");
  const localSessions = sessions.map(localSessionToDiscovered);
  return groupDisplaySessions([...externalSessions, ...localSessions], mode);
}

function localSessionToDiscovered(session: SessionData): DiscoveredSession {
  const harness = HARNESSES.find((item) => item.id === session.harness);

  return {
    id: session.id,
    title: session.title || "Untitled",
    cwd: "",
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
    const groupId = mode === "harness" ? session.harnessId : project || "__no_project__";
    const groupLabel = mode === "harness" ? session.harnessLabel : project || "No project";
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
  return [...sessions].sort((a, b) => sessionTime(b.updatedAt) - sessionTime(a.updatedAt));
}

function sessionTime(value?: string): number {
  if (!value) return 0;
  const time = Date.parse(value);
  return Number.isFinite(time) ? time : 0;
}

function sessionMeta(session: DiscoveredSession, mode: SessionGroupMode): string {
  const date = formatSessionDate(session.updatedAt);
  if (mode === "project") {
    return `${session.harnessLabel} - ${date}`;
  }
  if (session.cwd.trim()) {
    return `${projectName(session.cwd)} - ${date}`;
  }
  return `${session.source === "local" ? "Local" : "ACP"} - ${date}`;
}

function harnessLabel(id: string): string {
  return HARNESSES.find((harness) => harness.id === id)?.label ?? id;
}

function projectName(cwd: string): string {
  const parts = cwd.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] ?? cwd;
}

function formatSessionDate(value?: string): string {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function cx(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}
