import { lazy, Suspense, useEffect, useState } from "react";
import { z } from "zod";
import { LanguageSchema, ProjectSchema } from "../settings.js";
import { HarnessPicker } from "./agent-controls.js";
import { projectFetch as fetch } from "./api.js";
import { ConversationSurface } from "./chat.js";
import { i18n, t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";
import { ProjectNav } from "./projects.js";
import { SettingsPage } from "./settings.js";
import { TracePanel } from "./trace.js";

const ResearchPanel = lazy(() =>
  import("./research.js").then(({ ResearchPanel }) => ({ default: ResearchPanel })),
);

const Sessions = z.array(
  z.object({
    sessionId: z.string(),
    title: z.string().nullish(),
    updatedAt: z.string().nullish(),
  }),
);
const Bootstrap = z.strictObject({
  agents: z.array(z.string()),
  defaultHarness: z.string(),
  language: LanguageSchema.nullable(),
  sessions: Sessions,
  sessionError: z.string().optional(),
  workspace: ProjectSchema,
  projects: z.array(ProjectSchema),
});

export function App() {
  useTranslation();
  const [bootstrap, setBootstrap] = useState<z.infer<typeof Bootstrap>>();
  const [error, setError] = useState<string>();
  const [sessions, setSessions] = useState<z.infer<typeof Sessions>>([]);
  const [selected, setSelected] = useState("");
  const [sessionRequest, setSessionRequest] = useState({ agentId: "swarm" });
  const { agentId } = sessionRequest;
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(() => matchMedia("(min-width: 768px)").matches);
  const [panel, setPanel] = useState<"assets" | "observe">();
  const [target, setTarget] = useState<{ artifactId?: string; projectId?: string }>();
  const [page, setPage] = useState<"chat" | "settings" | "project-settings">("chat");
  useEffect(() => {
    const open = (event: Event) => {
      setPage("chat");
      setPanel("assets");
      setTarget((event as CustomEvent).detail);
    };
    window.addEventListener("swarmx:open-research", open);
    return () => window.removeEventListener("swarmx:open-research", open);
  }, []);

  useEffect(() => {
    const { agentId } = sessionRequest;
    let current = true;
    setLoading(true);
    setError(undefined);
    const load = async () => {
      const response = await fetch(
        agentId === "swarm"
          ? "/api/v1/bootstrap"
          : `/api/v1/sessions?agent=${encodeURIComponent(agentId)}`,
      );
      if (!response.ok) throw new Error(await response.text());
      const data: unknown = await response.json();
      const bootstrap = agentId === "swarm" ? Bootstrap.parse(data) : undefined;
      const sessions = bootstrap?.sessions ?? Sessions.parse(data);
      if (!current) return;
      if (bootstrap !== undefined) {
        if (bootstrap.language) await i18n.changeLanguage(bootstrap.language);
        setBootstrap(bootstrap);
        setError(bootstrap.sessionError);
      }
      setSessions(sessions);
      setSelected((id) =>
        sessions.some((session) => session.sessionId === id) ? id : (sessions[0]?.sessionId ?? ""),
      );
    };
    void load()
      .catch((cause: unknown) => {
        if (current) setError(cause instanceof Error ? cause.message : String(cause));
      })
      .finally(() => {
        if (current) setLoading(false);
      });
    return () => {
      current = false;
    };
  }, [sessionRequest]);

  const select = (id: string) => {
    setPage("chat");
    setSelected(id);
    if (!matchMedia("(min-width: 768px)").matches) setSidebarOpen(false);
  };

  const create = async () => {
    setCreating(true);
    setError(undefined);
    try {
      const response = await fetch(`/api/v1/sessions?agent=${encodeURIComponent(agentId)}`, {
        method: "POST",
      });
      if (!response.ok) throw new Error(await response.text());
      const session = z.object({ sessionId: z.string() }).parse(await response.json());
      setSessions((items) => [{ ...session, title: t("新任务") }, ...items]);
      setQuery("");
      select(session.sessionId);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setCreating(false);
    }
  };

  const changeHarness = (id: string) => {
    setSessionRequest({ agentId: id });
    setSelected("");
    setSessions([]);
    setQuery("");
  };

  if (bootstrap === undefined) {
    return (
      <main className="grid h-dvh place-items-center bg-white text-neutral-900">
        <div className="flex max-w-md flex-col items-center gap-4 p-6 text-center">
          <Icon name="swarm" className="size-9" />
          {error === undefined ? (
            <p role="status" className="text-sm text-neutral-500">
              {t("正在连接 SwarmX…")}
            </p>
          ) : (
            <>
              <p role="alert" className="break-words text-sm">
                {error}
              </p>
              <button
                className="primary-button"
                type="button"
                onClick={() => setSessionRequest({ agentId })}
              >
                {t("重新连接")}
              </button>
            </>
          )}
        </div>
      </main>
    );
  }

  const title = sessions.find((session) => session.sessionId === selected)?.title ?? t("新任务");
  const harnessProps = {
    harness: agentId === "swarm" ? bootstrap.defaultHarness : agentId,
    harnesses: bootstrap.agents.filter((id) => id !== "swarm"),
    onHarnessChange: changeHarness,
  };
  const filtered = sessions.filter((session) =>
    (session.title ?? session.sessionId)
      .toLocaleLowerCase()
      .includes(query.trim().toLocaleLowerCase()),
  );
  const sidePanel = panel && (
    <Suspense
      fallback={
        <p role="status" className="p-6">
          {t("正在打开侧栏…")}
        </p>
      }
    >
      <ResearchPanel
        mode={panel ?? "assets"}
        canCompose={!!selected}
        target={target}
        onClose={() => setPanel(undefined)}
        trace={
          selected ? (
            <TracePanel />
          ) : (
            <p className="p-6 text-neutral-500">{t("任务开始后，执行轨迹会显示在这里。")}</p>
          )
        }
      />
    </Suspense>
  );

  return (
    <main className="relative flex h-dvh overflow-hidden bg-white text-sm text-neutral-900">
      {page !== "chat" && (
        <section className="flex min-w-0 flex-1 flex-col" aria-label={t("设置")}>
          <header className="flex h-16 shrink-0 items-center gap-3 border-b border-neutral-200 px-5">
            <button className="secondary-button" type="button" onClick={() => setPage("chat")}>
              <Icon name="sidebar" />
              {t("返回对话")}
            </button>
            <h1 className="font-medium">{page === "settings" ? t("通用设置") : t("项目设置")}</h1>
          </header>
          <SettingsPage
            project={page === "project-settings" ? bootstrap.workspace : undefined}
            sessionId={selected || undefined}
          />
        </section>
      )}
      <div hidden={page !== "chat"} className={page !== "chat" ? "hidden" : "flex min-w-0 flex-1"}>
        {sidebarOpen && (
          <>
            <button
              aria-label={t("关闭任务导航")}
              className="absolute inset-0 z-20 bg-black/20 md:hidden"
              type="button"
              onClick={() => setSidebarOpen(false)}
            />
            <aside
              id="task-sidebar"
              className="absolute inset-y-0 left-0 z-30 flex w-64 shrink-0 flex-col border-neutral-200/70 border-r bg-neutral-100 p-3 md:static"
            >
              <header className="flex h-12 items-center gap-2.5 px-2">
                <Icon name="swarm" className="size-6" />
                <strong className="text-base font-semibold tracking-tight">SwarmX</strong>
              </header>
              <button
                className="mt-3 flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2.5 text-left font-medium hover:bg-neutral-200 disabled:opacity-40"
                disabled={loading || creating}
                onClick={() => void create()}
                type="button"
              >
                <Icon name="compose" />
                {creating ? t("正在创建…") : t("新建任务")}
              </button>
              <label className="mt-2 flex items-center gap-2 rounded-lg bg-neutral-200/50 px-2.5 text-neutral-500 focus-within:ring-1 focus-within:ring-neutral-400">
                <Icon name="search" />
                <input
                  aria-label={t("搜索任务")}
                  className="min-w-0 flex-1 bg-transparent py-2 outline-none placeholder:text-neutral-500"
                  type="search"
                  placeholder={t("搜索任务")}
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
              </label>
              <ProjectNav
                projects={bootstrap.projects}
                current={bootstrap.workspace.id}
                onSettings={() => setPage("project-settings")}
              />
              <div className="mt-7 mb-2 flex items-center justify-between px-2 text-xs text-neutral-500">
                <span>{t("任务")}</span>
                <button
                  aria-label={t("刷新任务")}
                  title={t("刷新任务")}
                  className="icon-button size-7"
                  disabled={loading || creating}
                  onClick={() => setSessionRequest({ agentId })}
                  type="button"
                >
                  <Icon name="refresh" className={`size-3.5 ${loading ? "animate-spin" : ""}`} />
                </button>
              </div>
              <div className="mb-1 flex items-center gap-2 px-2.5 py-2 font-medium">
                <Icon name="folder" className="size-4 text-neutral-500" />
                <span className="truncate">{bootstrap.workspace.label}</span>
                <span className="ml-auto text-xs font-normal text-neutral-500">
                  {sessions.length}
                </span>
              </div>
              <nav
                aria-label={t("任务列表")}
                aria-busy={loading}
                className="min-h-0 flex-1 overflow-y-auto"
              >
                {filtered.map((session) => (
                  <button
                    aria-current={session.sessionId === selected ? "page" : undefined}
                    className="my-0.5 flex w-full items-center gap-2 rounded-lg py-2.5 pr-2.5 pl-8 text-left text-neutral-600 hover:bg-neutral-200/70 aria-[current=page]:bg-neutral-200 aria-[current=page]:text-neutral-950"
                    key={session.sessionId}
                    onClick={() => select(session.sessionId)}
                    title={session.title ?? session.sessionId}
                    type="button"
                  >
                    <span className="truncate">{session.title ?? session.sessionId}</span>
                    {session.updatedAt && (
                      <time
                        className="ml-auto shrink-0 text-[11px] text-neutral-500"
                        dateTime={session.updatedAt}
                      >
                        {new Date(session.updatedAt).toLocaleDateString(i18n.language, {
                          month: "numeric",
                          day: "numeric",
                        })}
                      </time>
                    )}
                  </button>
                ))}
                {filtered.length === 0 && (
                  <p className="px-3 py-5 text-xs leading-6 text-neutral-500">
                    {loading
                      ? t("正在加载任务…")
                      : query.trim()
                        ? t("没有匹配的任务")
                        : t("还没有任务。从一个问题开始吧。")}
                  </p>
                )}
              </nav>
              <footer className="mt-3 border-neutral-200 border-t pt-3">
                <button
                  className="workspace-nav w-full gap-2.5 py-3 text-left"
                  type="button"
                  aria-label={t("通用设置")}
                  onClick={() => setPage("settings")}
                >
                  <span className="grid size-8 shrink-0 place-items-center rounded-full bg-white text-xs font-semibold">
                    SX
                  </span>
                  <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                    <span className="font-medium">SwarmX</span>
                    <span className="text-[10px] text-neutral-500">{t("通用设置")}</span>
                  </span>
                  <Icon name="settings" />
                </button>
              </footer>
            </aside>
          </>
        )}
        <section className="flex min-w-0 flex-1 flex-col">
          <header className="flex h-16 shrink-0 items-center gap-3 border-neutral-100 border-b px-4">
            <button
              aria-controls="task-sidebar"
              aria-expanded={sidebarOpen}
              aria-label={sidebarOpen ? t("收起侧栏") : t("展开侧栏")}
              title={sidebarOpen ? t("收起侧栏") : t("展开侧栏")}
              className="icon-button"
              type="button"
              onClick={() => setSidebarOpen((open) => !open)}
            >
              <Icon name="sidebar" />
            </button>
            <span className="hidden truncate text-neutral-500 sm:inline">
              {bootstrap.workspace.label}
            </span>
            <span aria-hidden="true" className="hidden text-neutral-300 sm:inline">
              /
            </span>
            <h1 className="min-w-0 truncate font-medium">{title}</h1>
            <div className="ml-auto flex items-center gap-1">
              <button
                className="secondary-button border-transparent px-2.5 aria-expanded:bg-neutral-100"
                aria-label={t("科研资产")}
                aria-expanded={panel === "assets"}
                aria-controls="research-side-view"
                type="button"
                onClick={() => setPanel(panel === "assets" ? undefined : "assets")}
              >
                <Icon name="image" />
                <span className="hidden sm:inline">{t("科研资产")}</span>
              </button>
              <button
                className="secondary-button border-transparent px-2.5 aria-expanded:bg-neutral-100"
                aria-label={t("观测与溯源")}
                aria-expanded={panel === "observe"}
                aria-controls="research-side-view"
                type="button"
                onClick={() => setPanel(panel === "observe" ? undefined : "observe")}
              >
                <Icon name="trace" />
                <span className="hidden sm:inline">{t("观测与溯源")}</span>
              </button>
            </div>
          </header>
          {error !== undefined && (
            <p
              role="alert"
              className="mx-4 mt-3 rounded-lg border border-neutral-300 bg-neutral-50 px-4 py-3 text-sm break-words"
            >
              {error}
            </p>
          )}
          <div className="relative flex min-h-0 flex-1">
            {selected === "" ? (
              <>
                <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-5 p-8 text-center">
                  <Icon name="swarm" className="size-10" />
                  <h2 className="text-3xl font-semibold tracking-tight">
                    {t("让研究，从这里开始。")}
                  </h2>
                  <p className="max-w-md text-sm leading-7 text-neutral-500">
                    {t("梳理一个问题，探索一份数据，或继续你的实验。")}
                    <br />
                    {t("SwarmX 与你一起推进。")}
                  </p>
                  <button
                    className="primary-button mt-1 gap-2"
                    disabled={loading || creating}
                    onClick={() => void create()}
                    type="button"
                  >
                    <Icon name="plus" />
                    {loading ? t("正在加载任务…") : creating ? t("正在创建…") : t("开始一个任务")}
                  </button>
                  <span className="mt-2 flex items-center gap-1.5 text-xs text-neutral-400">
                    <Icon name="folder" className="size-3.5" />
                    {bootstrap.workspace.label}
                  </span>
                  <HarnessPicker {...harnessProps} disabled={creating} />
                </div>
                <div className={panel ? "research-side-view" : "hidden"} id="research-side-view">
                  {sidePanel}
                </div>
              </>
            ) : (
              <ConversationSurface
                key={`${agentId}:${selected}`}
                agentId={agentId}
                {...harnessProps}
                harnessDisabled={creating}
                threadId={selected}
                workspace={bootstrap.workspace.label}
                sidePanel={sidePanel}
                panelOpen={!!panel}
              />
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
