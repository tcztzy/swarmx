import { lazy, Suspense, useEffect, useState } from "react";
import type { z } from "zod";
import { BootstrapSchema, SessionCreateSchema, SessionListSchema } from "../bridge-contract.js";
import { HarnessPicker } from "./agent-controls.js";
import { bridge } from "./bridge.js";
import { ConversationSurface } from "./chat.js";
import { TooltipIconButton } from "./components/assistant-ui/elements/tooltip-icon-button.js";
import { Button } from "./components/ui/radix/button.js";
import { Input } from "./components/ui/radix/input.js";
import { i18n, t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";
import { SettingsPage } from "./settings.js";
import type { SourceReference } from "./source-inspection.js";
import { TracePanel } from "./trace.js";

const ObservePanel = lazy(() =>
  import("./observe.js").then(({ ObservePanel }) => ({ default: ObservePanel })),
);
const WorkPanel = lazy(() => import("./work.js").then(({ WorkPanel }) => ({ default: WorkPanel })));

type Sessions = z.infer<typeof SessionListSchema>;

export function App() {
  useTranslation();
  const [bootstrap, setBootstrap] = useState<z.infer<typeof BootstrapSchema>>();
  const [error, setError] = useState<string>();
  const [sessions, setSessions] = useState<Sessions>([]);
  const [selected, setSelected] = useState("");
  const [sessionRequest, setSessionRequest] = useState({ agentId: "swarm" });
  const { agentId } = sessionRequest;
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(() => matchMedia("(min-width: 768px)").matches);
  const [panel, setPanel] = useState<"observe" | "work">();
  const [target, setTarget] = useState<SourceReference>();
  const [page, setPage] = useState<"chat" | "settings">("chat");
  useEffect(() => {
    const open = (event: Event) => {
      setPage("chat");
      const detail = (event as CustomEvent<{ source?: SourceReference }>).detail;
      if (!detail?.source) return;
      setPanel("observe");
      setTarget(detail.source);
      setSidebarOpen(false);
    };
    window.addEventListener("swarmx:open-source", open);
    return () => window.removeEventListener("swarmx:open-source", open);
  }, []);

  useEffect(() => {
    const { agentId } = sessionRequest;
    let current = true;
    setLoading(true);
    setError(undefined);
    const load = async () => {
      if (agentId !== "swarm") {
        const listed = SessionListSchema.parse(await bridge().sessions.list({ agent: agentId }));
        if (!current) return;
        setSessions(listed);
        setSelected((id) =>
          listed.some((session) => session.sessionId === id) ? id : (listed[0]?.sessionId ?? ""),
        );
        return;
      }
      const bootstrap = BootstrapSchema.parse(await bridge().bootstrap());
      if (!current) return;
      if (bootstrap.language) await i18n.changeLanguage(bootstrap.language);
      setBootstrap(bootstrap);
      setError(bootstrap.sessionError);
      setSessions(bootstrap.sessions);
      setSelected((id) =>
        bootstrap.sessions.some((session) => session.sessionId === id)
          ? id
          : (bootstrap.sessions[0]?.sessionId ?? ""),
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
      const session = SessionCreateSchema.parse(await bridge().sessions.create({ agent: agentId }));
      setSessions((items) => [{ ...session, title: t("新任务") }, ...items]);
      setQuery("");
      select(session.sessionId);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setCreating(false);
    }
  };

  const refresh = () => setSessionRequest((request) => ({ agentId: request.agentId }));
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
              <Button
                variant="default"
                size="default"
                type="button"
                onClick={() => setSessionRequest({ agentId })}
              >
                {t("重新连接")}
              </Button>
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
      {panel === "work" && (
        <WorkPanel onClose={() => setPanel(undefined)} harnesses={harnessProps.harnesses} />
      )}
      {panel === "observe" && (
        <ObservePanel
          source={target}
          onClose={() => setPanel(undefined)}
          trace={
            selected ? (
              <TracePanel />
            ) : (
              <p className="p-6 text-neutral-500">{t("任务开始后，执行轨迹会显示在这里。")}</p>
            )
          }
        />
      )}
    </Suspense>
  );

  return (
    <main className="relative flex h-dvh overflow-hidden bg-white text-sm text-neutral-900">
      {page !== "chat" && (
        <section className="flex min-w-0 flex-1 flex-col" aria-label={t("设置")}>
          <header className="flex h-16 shrink-0 items-center gap-3 border-b border-neutral-200 px-5">
            <Button variant="outline" size="sm" type="button" onClick={() => setPage("chat")}>
              <Icon name="sidebar" />
              {t("返回对话")}
            </Button>
            <h1 className="font-medium">{t("设置")}</h1>
          </header>
          <SettingsPage sessionId={selected || undefined} />
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
                <Input
                  aria-label={t("搜索任务")}
                  className="min-w-0 flex-1 bg-transparent py-2 outline-none placeholder:text-neutral-500"
                  type="search"
                  placeholder={t("搜索任务")}
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
              </label>
              <div className="mt-7 mb-2 flex items-center justify-between px-2 text-xs text-neutral-500">
                <span>
                  {t("任务")} · {sessions.length}
                </span>
                <TooltipIconButton
                  tooltip={t("刷新任务")}
                  className="size-7"
                  disabled={loading || creating}
                  onClick={refresh}
                >
                  <Icon name="refresh" className={`size-3.5 ${loading ? "animate-spin" : ""}`} />
                </TooltipIconButton>
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
                  className="settings-nav w-full gap-2.5 py-3 text-left"
                  type="button"
                  aria-label={t("设置")}
                  onClick={() => setPage("settings")}
                >
                  <span className="grid size-8 shrink-0 place-items-center rounded-full bg-white text-xs font-semibold">
                    SX
                  </span>
                  <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                    <span className="font-medium">SwarmX</span>
                    <span className="text-[10px] text-neutral-500">{t("设置")}</span>
                  </span>
                  <Icon name="settings" />
                </button>
              </footer>
            </aside>
          </>
        )}
        <section className="flex min-w-0 flex-1 flex-col">
          <header className="conversation-header">
            <TooltipIconButton
              tooltip={sidebarOpen ? t("收起侧栏") : t("展开侧栏")}
              aria-controls="task-sidebar"
              aria-expanded={sidebarOpen}
              className="size-8"
              onClick={() => setSidebarOpen((open) => !open)}
            >
              <Icon name="sidebar" />
            </TooltipIconButton>
            <span aria-hidden="true" className="header-divider" />
            <h1 title={title}>{title}</h1>
            <div className="ml-auto flex items-center gap-1">
              <Button
                variant="outline"
                size="sm"
                className="border-transparent px-2.5 aria-expanded:bg-neutral-100"
                aria-label={t("长期工作")}
                aria-expanded={panel === "work"}
                aria-controls="observation-side-view"
                type="button"
                onClick={() => setPanel(panel === "work" ? undefined : "work")}
              >
                <Icon name="book" className="size-6" />
                <span className="hidden sm:inline">{t("长期工作")}</span>
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="border-transparent px-2.5 aria-expanded:bg-neutral-100"
                aria-label={t("观测与溯源")}
                aria-expanded={panel === "observe"}
                aria-controls="observation-side-view"
                type="button"
                onClick={() => {
                  setTarget(undefined);
                  setPanel(panel === "observe" && !target ? undefined : "observe");
                }}
              >
                <Icon name="eye" className="size-6" />
                <span className="hidden sm:inline">{t("观测与溯源")}</span>
              </Button>
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
                    {t("让工作，从这里开始。")}
                  </h2>
                  <p className="max-w-md text-sm leading-7 text-neutral-500">
                    {t("梳理一个问题，探索代码，或继续你的任务。")}
                    <br />
                    {t("SwarmX 与你一起推进。")}
                  </p>
                  <Button
                    variant="default"
                    size="default"
                    className="mt-1 gap-2"
                    disabled={loading || creating}
                    onClick={() => void create()}
                    type="button"
                  >
                    <Icon name="plus" />
                    {loading ? t("正在加载任务…") : creating ? t("正在创建…") : t("开始一个任务")}
                  </Button>
                  <span className="mt-2 flex items-center gap-1.5 text-xs text-neutral-400">
                    <Icon name="folder" className="size-3.5" />
                    {bootstrap.cwd}
                  </span>
                  <HarnessPicker {...harnessProps} disabled={creating} />
                </div>
                <div
                  className={panel ? "observation-side-view" : "hidden"}
                  id="observation-side-view"
                >
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
                sidePanel={sidePanel}
                panelOpen={!!panel}
                source={panel === "observe" ? target : undefined}
              />
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
