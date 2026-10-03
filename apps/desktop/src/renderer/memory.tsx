import { MarkerType, Position } from "@xyflow/react";
import { useCallback, useEffect, useState } from "react";
import { z } from "zod";
import { MemoryGraphSchema, MemorySettingsSchema, MemoryStatusSchema } from "../memory.js";
import { tool } from "./bridge.js";
import { Button } from "./components/ui/radix/button.js";
import { Input } from "./components/ui/radix/input.js";
import { NativeSelect } from "./components/ui/radix/native-select.js";
import { Textarea } from "./components/ui/radix/textarea.js";
import { i18n, t, useTranslation } from "./i18n.js";
import { GraphView } from "./research-graph.js";

const LoadedConcept = z.object({
  concepts: z.array(
    z.object({
      id: z.string(),
      body: z.string(),
      revision: z.string(),
      metadata: z.object({ title: z.string(), type: z.string() }).passthrough(),
    }),
  ),
  graph: MemoryGraphSchema,
});

export function MemorySettings({ sessionId }: { sessionId?: string | undefined }) {
  useTranslation();
  const [status, setStatus] = useState<z.infer<typeof MemoryStatusSchema>>();
  const [graph, setGraph] = useState<z.infer<typeof MemoryGraphSchema>>();
  const [loaded, setLoaded] = useState<z.infer<typeof LoadedConcept>>();
  const [selected, setSelected] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [query, setQuery] = useState("");
  const refreshStatus = useCallback(async (signal?: AbortSignal) => {
    try {
      setStatus(await tool("memory", { action: "memory_status", request: {} }, MemoryStatusSchema));
    } catch (cause) {
      if (!signal?.aborted) setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);
  useEffect(() => {
    const abort = new AbortController();
    void refreshStatus(abort.signal);
    return () => abort.abort();
  }, [refreshStatus]);
  useEffect(() => {
    if (status?.review.state !== "running") return;
    const abort = new AbortController();
    const timer = setInterval(() => void refreshStatus(abort.signal), 1500);
    return () => {
      abort.abort();
      clearInterval(timer);
    };
  }, [status?.review.state, refreshStatus]);
  async function perform(task: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    try {
      await task();
      await refreshStatus();
      if (graph)
        setGraph(await tool("memory", { action: "graph_memory", request: {} }, MemoryGraphSchema));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    if (!selected) return;
    setLoaded(undefined);
    void tool("memory", { action: "load_memory", request: { id: selected } }, LoadedConcept)
      .then(setLoaded)
      .catch((cause: Error) => setError(cause.message));
  }, [selected]);

  const levels = new Map<string, number>();
  for (const node of graph?.nodes ?? [])
    levels.set(
      node.id,
      Math.max(
        -1,
        ...(graph?.edges
          .filter(({ source }) => source === node.id)
          .map(({ target }) => levels.get(target) ?? 0) ?? []),
      ) + 1,
    );
  const rows = new Map<number, number>();
  const matches =
    graph?.nodes.filter((node) =>
      `${node.title} ${node.description} ${node.type}`
        .toLocaleLowerCase()
        .includes(query.trim().toLocaleLowerCase()),
    ) ?? [];
  const nodes = matches.slice(0, 200).map((node) => {
    const level = levels.get(node.id) ?? 0;
    const row = rows.get(level) ?? 0;
    rows.set(level, row + 1);
    return {
      id: node.id,
      position: { x: level * 260, y: row * 95 },
      selected: node.id === selected,
      sourcePosition: Position.Left,
      targetPosition: Position.Right,
      deletable: false,
      data: {
        label: (
          <div className="text-left">
            <small className={node.stale ? "text-amber-700" : "text-neutral-400"}>
              {node.type}
              {node.stale ? ` · ${t("需要复核")}` : ""}
            </small>
            <span className="block text-xs font-medium">{node.title}</span>
          </div>
        ),
      },
    };
  });
  const visible = new Set(nodes.map(({ id }) => id));

  return (
    <section className="space-y-5 border-t border-neutral-200 pt-6" aria-label={t("记忆与知识")}>
      <div>
        <h3 className="text-lg font-semibold">{t("记忆与知识")}</h3>
        <p className="mt-1 text-sm text-neutral-500">
          {t("Vault 保存带来源与依赖的知识。新任务自动加载记忆目录。")}
        </p>
      </div>
      {error && (
        <p role="alert" className="workbench-alert">
          {error}
        </p>
      )}
      {!status ? (
        <p role="status">{t("正在读取记忆…")}</p>
      ) : (
        <>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              const data = new FormData(event.currentTarget);
              void perform(() =>
                tool(
                  "memory",
                  {
                    action: "memory_configure",
                    request: {
                      enabled: data.has("enabled"),
                      autoReview: data.has("autoReview"),
                      writeApproval: data.has("writeApproval"),
                      reviewInterval: Number(data.get("reviewInterval")),
                      reviewHarness: data.get("reviewHarness"),
                    },
                  },
                  MemorySettingsSchema,
                ),
              );
            }}
          >
            <fieldset
              disabled={busy}
              className="space-y-3 rounded-xl border border-neutral-200 p-4"
            >
              <label className="flex items-center gap-2">
                <input type="checkbox" name="enabled" defaultChecked={status.settings.enabled} />
                {t("启用长期记忆")}
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  name="autoReview"
                  defaultChecked={status.settings.autoReview}
                />
                {t("对话后自动整理记忆")}
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  name="writeApproval"
                  defaultChecked={status.settings.writeApproval}
                />
                {t("保存前由我确认")}
              </label>
              <div className="grid grid-cols-2 gap-4">
                <label className="field-label">
                  {t("每多少次执行复盘")}
                  <Input
                    name="reviewInterval"
                    type="number"
                    min={1}
                    max={100}
                    required
                    defaultValue={status.settings.reviewInterval}
                  />
                </label>
                <label className="field-label">
                  {t("后台复盘运行时")}
                  <NativeSelect name="reviewHarness" defaultValue={status.settings.reviewHarness}>
                    <option value="codex">Codex</option>
                    <option value="claude">Claude</option>
                  </NativeSelect>
                </label>
              </div>
              <p className="text-xs text-neutral-500">
                {t(
                  "复盘使用已登录的模型并消耗用量。后台不能执行工具；提交的修改仍受记忆审批控制。会话快照在新任务中更新。",
                )}
              </p>
              <Button type="submit" variant="outline" size="sm">
                {t("保存记忆设置")}
              </Button>
            </fieldset>
          </form>
          <form
            className="space-y-2"
            onSubmit={(event) => {
              event.preventDefault();
              const content = new FormData(event.currentTarget).get("content");
              void perform(() =>
                tool(
                  "memory",
                  {
                    action: "update_core_memory",
                    request: { content, expectedRevision: status.note.revision },
                  },
                  z.unknown(),
                ),
              );
            }}
          >
            <label className="field-label">
              {t("用户偏好")}
              <Textarea
                name="content"
                rows={6}
                defaultValue={status.note.content}
                className="resize-y font-mono text-xs"
              />
            </label>
            <div className="flex items-center justify-between text-xs text-neutral-500">
              <span>
                {t("{{used}} / {{limit}} 字符", {
                  used: Array.from(status.note.content).length,
                  limit: status.note.limit,
                })}
              </span>
              <Button type="submit" variant="outline" size="sm" disabled={busy}>
                {t("保存笔记")}
              </Button>
            </div>
          </form>
          <div className="flex flex-wrap items-center gap-3">
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={
                busy || !sessionId || !status.settings.enabled || status.review.state === "running"
              }
              onClick={() =>
                void perform(() =>
                  tool("memory", { action: "memory_review", request: { sessionId } }, z.unknown()),
                )
              }
            >
              {status.review.state === "running" ? t("正在整理记忆…") : t("立即复盘当前对话")}
            </Button>
            <span role="status" className="text-xs text-neutral-500">
              {status.review.state === "completed"
                ? t("复盘完成：{{count}} 项修改", { count: Number(status.review.message) })
                : status.review.state === "failed"
                  ? `${t("复盘失败")}：${status.review.message}`
                  : ""}
            </span>
          </div>
          {status.review.summary && (
            <p className="text-sm text-neutral-500">{status.review.summary}</p>
          )}
          <div>
            <h4 className="mb-2 font-medium">
              {t("待确认的记忆")} <span className="text-neutral-400">{status.pending.length}</span>
            </h4>
            {status.pending.length === 0 ? (
              <p className="text-sm text-neutral-500">{t("没有待确认的修改。")}</p>
            ) : (
              status.pending.map((entry) => (
                <article
                  key={entry.id}
                  className="mb-3 space-y-2 rounded-xl border border-neutral-200 p-4"
                >
                  <p className="text-xs text-neutral-500">
                    {entry.origin === "review" ? t("后台复盘") : t("对话中的保存请求")} ·{" "}
                    {new Date(entry.createdAt).toLocaleString(i18n.language)}
                  </p>
                  <h5 className="text-sm font-medium">
                    {entry.resourcePath ??
                      entry.operation.request.title ??
                      (entry.operation.action === "update_core_memory"
                        ? t("用户偏好")
                        : entry.operation.request.id)}
                  </h5>
                  <pre className="max-h-72 overflow-auto whitespace-pre-wrap text-sm">
                    {entry.operation.request.content ??
                      entry.operation.request.body ??
                      (entry.operation.action === "deprecate_memory"
                        ? t("将该知识标记为不再使用。")
                        : t("更新知识的说明、来源或依赖。"))}
                  </pre>
                  <details className="text-xs text-neutral-500">
                    <summary className="cursor-pointer">{t("查看来源与依赖")}</summary>
                    <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap">
                      {JSON.stringify(entry.operation.request, null, 2)}
                    </pre>
                  </details>
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      disabled={busy}
                      onClick={() =>
                        void perform(() =>
                          tool(
                            "memory",
                            {
                              action: "memory_decide",
                              request: { id: entry.id, decision: "approve" },
                            },
                            z.unknown(),
                          ),
                        )
                      }
                    >
                      {t("批准保存")}
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      disabled={busy}
                      onClick={() =>
                        void perform(() =>
                          tool(
                            "memory",
                            {
                              action: "memory_decide",
                              request: { id: entry.id, decision: "reject" },
                            },
                            z.unknown(),
                          ),
                        )
                      }
                    >
                      {t("拒绝")}
                    </Button>
                  </div>
                </article>
              ))
            )}
          </div>
        </>
      )}
      <Button
        type="button"
        variant="outline"
        size="sm"
        disabled={busy}
        onClick={() =>
          void perform(async () =>
            setGraph(
              await tool("memory", { action: "graph_memory", request: {} }, MemoryGraphSchema),
            ),
          )
        }
      >
        {t("浏览 Vault 依赖图")}
      </Button>
      {graph && (
        <div className="space-y-3">
          <Input
            aria-label={t("搜索知识概念")}
            placeholder={t("搜索知识概念")}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          {graph.nodes.length === 0 ? (
            <p className="rounded-xl border border-dashed border-neutral-300 p-6 text-sm text-neutral-500">
              {t("Vault 还没有知识。可以在对话中要求记住研究方法，或在完成任务后运行复盘。")}
            </p>
          ) : (
            <>
              <div className="h-[420px] overflow-hidden rounded-xl border border-neutral-200">
                <GraphView
                  label={t("知识依赖图谱")}
                  graph={{
                    nodes,
                    count: matches.length,
                    edges: graph.edges
                      .filter(({ source, target }) => visible.has(source) && visible.has(target))
                      .map((edge) => ({
                        ...edge,
                        id: `${edge.source}:${edge.target}`,
                        label: t("依赖"),
                        markerEnd: { type: MarkerType.ArrowClosed },
                        type: "smoothstep",
                        style: { stroke: edge.stale ? "#b45309" : "#737373" },
                      })),
                  }}
                  onSelect={setSelected}
                />
              </div>
              <label className="field-label">
                {t("查看概念")}
                <NativeSelect
                  value={selected}
                  onChange={(event) => setSelected(event.target.value)}
                >
                  <option value="">{t("选择一个知识概念")}</option>
                  {matches.map((node) => (
                    <option key={node.id} value={node.id}>
                      {node.title}
                      {node.stale ? ` · ${t("需要复核")}` : ""}
                    </option>
                  ))}
                </NativeSelect>
              </label>
            </>
          )}
          {loaded && (
            <div className="space-y-3">
              <h4 className="font-medium">{t("知识与前置依赖")}</h4>
              {loaded.concepts.map((concept) => (
                <details
                  key={concept.id}
                  open={concept.id === selected}
                  className="rounded-lg border border-neutral-200 p-3"
                >
                  <summary className="cursor-pointer text-sm">
                    {concept.metadata.title} · {concept.metadata.type}
                  </summary>
                  <p className="mt-2 break-all font-mono text-[10px] text-neutral-400">
                    {concept.id} · {concept.revision}
                  </p>
                  <pre className="mt-3 whitespace-pre-wrap text-sm">{concept.body}</pre>
                </details>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
