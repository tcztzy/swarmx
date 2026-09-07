import { MarkerType, Position } from "@xyflow/react";
import { useEffect, useState } from "react";
import { z } from "zod";
import { MemoryGraphSchema, MemorySettingsSchema, MemoryStatusSchema } from "../memory.js";
import { api, jsonRequest } from "./api.js";
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

export function MemorySettings({
  sessionId,
  scope,
}: {
  sessionId?: string | undefined;
  scope: "user" | "project";
}) {
  useTranslation();
  const [status, setStatus] = useState<z.infer<typeof MemoryStatusSchema>>();
  const [graph, setGraph] = useState<z.infer<typeof MemoryGraphSchema>>();
  const [loaded, setLoaded] = useState<z.infer<typeof LoadedConcept>>();
  const [selected, setSelected] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [query, setQuery] = useState("");
  useEffect(() => {
    const abort = new AbortController();
    void api("/api/v1/memory", MemoryStatusSchema, { signal: abort.signal })
      .then(setStatus)
      .catch((cause: Error) => {
        if (!abort.signal.aborted) setError(cause.message);
      });
    return () => abort.abort();
  }, []);
  useEffect(() => {
    if (status?.review.state !== "running") return;
    const abort = new AbortController();
    const timer = setInterval(() => {
      void api("/api/v1/memory", MemoryStatusSchema, { signal: abort.signal })
        .then(setStatus)
        .catch((cause: Error) => {
          if (!abort.signal.aborted) setError(cause.message);
        });
    }, 1500);
    return () => {
      abort.abort();
      clearInterval(timer);
    };
  }, [status?.review.state]);
  async function perform(task: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    try {
      await task();
      setStatus(await api("/api/v1/memory", MemoryStatusSchema));
      if (graph) setGraph(await api("/api/v1/memory/graph", MemoryGraphSchema));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    if (!selected) return;
    const abort = new AbortController();
    setLoaded(undefined);
    void api(`/api/v1/memory/concept?id=${encodeURIComponent(selected)}`, LoadedConcept, {
      signal: abort.signal,
    })
      .then(setLoaded)
      .catch((cause: Error) => {
        if (!abort.signal.aborted) setError(cause.message);
      });
    return () => abort.abort();
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
        <h3 className="text-lg font-semibold">
          {scope === "user" ? t("用户偏好 · 跨工作区") : t("记忆与知识")}
        </h3>
        <p className="mt-1 text-sm text-neutral-500">
          {scope === "user"
            ? t("用户偏好在所有项目中共享，新任务会读取更新后的偏好。")
            : t("短记忆用于长期偏好，Vault 保存带来源与依赖的科研知识。新任务自动加载记忆目录。")}
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
          {scope === "project" && (
            <form
              onSubmit={(event) => {
                event.preventDefault();
                const data = new FormData(event.currentTarget);
                void perform(() =>
                  api(
                    "/api/v1/memory/settings",
                    MemorySettingsSchema,
                    jsonRequest(
                      {
                        enabled: data.has("enabled"),
                        autoReview: data.has("autoReview"),
                        writeApproval: data.has("writeApproval"),
                        reviewInterval: Number(data.get("reviewInterval")),
                        reviewHarness: data.get("reviewHarness"),
                      },
                      "PUT",
                    ),
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
                    {t("每多少次输入复盘")}
                    <input
                      name="reviewInterval"
                      type="number"
                      min={1}
                      max={100}
                      required
                      className="interaction-input"
                      defaultValue={status.settings.reviewInterval}
                    />
                  </label>
                  <label className="field-label">
                    {t("后台复盘运行时")}
                    <select
                      name="reviewHarness"
                      className="interaction-input"
                      defaultValue={status.settings.reviewHarness}
                    >
                      <option value="codex">Codex</option>
                      <option value="claude">Claude</option>
                    </select>
                  </label>
                </div>
                <p className="text-xs text-neutral-500">
                  {t(
                    "复盘使用已登录的模型并消耗用量。后台不能执行工具；提交的修改仍受记忆审批控制。会话快照在新任务中更新。",
                  )}
                </p>
                <button type="submit" className="secondary-button">
                  {t("保存记忆设置")}
                </button>
              </fieldset>
            </form>
          )}
          <div className="space-y-4">
            {status.notes
              .filter((note) => note.target === (scope === "user" ? "user" : "workspace"))
              .map((note) => (
                <form
                  key={`${note.target}:${note.revision}`}
                  className="space-y-2"
                  onSubmit={(event) => {
                    event.preventDefault();
                    const content = new FormData(event.currentTarget).get("content");
                    void perform(() =>
                      api(
                        "/api/v1/memory/notes",
                        z.unknown(),
                        jsonRequest(
                          { target: note.target, content, expectedRevision: note.revision },
                          "PUT",
                        ),
                      ),
                    );
                  }}
                >
                  <label className="field-label">
                    {note.target === "user" ? t("用户偏好 · 跨工作区") : t("工作区笔记")}
                    <textarea
                      name="content"
                      rows={6}
                      defaultValue={note.content}
                      className="interaction-input resize-y font-mono text-xs"
                    />
                  </label>
                  <div className="flex items-center justify-between text-xs text-neutral-500">
                    <span>
                      {t("{{used}} / {{limit}} 字符", {
                        used: Array.from(note.content).length,
                        limit: note.limit,
                      })}
                    </span>
                    <button type="submit" className="secondary-button" disabled={busy}>
                      {t("保存笔记")}
                    </button>
                  </div>
                </form>
              ))}
          </div>
          {scope === "project" && (
            <>
              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  className="secondary-button"
                  disabled={
                    busy ||
                    !sessionId ||
                    !status.settings.enabled ||
                    status.review.state === "running"
                  }
                  onClick={() =>
                    void perform(() =>
                      api("/api/v1/memory/review", z.unknown(), jsonRequest({ sessionId })),
                    )
                  }
                >
                  {status.review.state === "running" ? t("正在整理记忆…") : t("立即复盘当前对话")}
                </button>
                <span role="status" className="text-xs text-neutral-500">
                  {status.review.state === "completed"
                    ? t("复盘完成：{{count}} 项修改", { count: Number(status.review.message) })
                    : status.review.state === "failed"
                      ? `${t("复盘失败")}：${status.review.message}`
                      : ""}
                </span>
              </div>
              <div>
                <h4 className="mb-2 font-medium">
                  {t("待确认的记忆")}{" "}
                  <span className="text-neutral-400">{status.pending.length}</span>
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
                        {entry.operation.request.title ??
                          (entry.operation.request.target === "user"
                            ? t("用户偏好 · 跨工作区")
                            : entry.operation.request.target === "workspace"
                              ? t("工作区笔记")
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
                        <button
                          type="button"
                          className="primary-button"
                          disabled={busy}
                          onClick={() =>
                            void perform(() =>
                              api(
                                `/api/v1/memory/pending/${entry.id}`,
                                z.unknown(),
                                jsonRequest({ action: "approve" }),
                              ),
                            )
                          }
                        >
                          {t("批准保存")}
                        </button>
                        <button
                          type="button"
                          className="secondary-button"
                          disabled={busy}
                          onClick={() =>
                            void perform(() =>
                              api(
                                `/api/v1/memory/pending/${entry.id}`,
                                z.unknown(),
                                jsonRequest({ action: "reject" }),
                              ),
                            )
                          }
                        >
                          {t("拒绝")}
                        </button>
                      </div>
                    </article>
                  ))
                )}
              </div>
            </>
          )}
        </>
      )}
      {scope === "project" && (
        <>
          <button
            type="button"
            className="secondary-button"
            disabled={busy}
            onClick={() =>
              void perform(async () =>
                setGraph(await api("/api/v1/memory/graph", MemoryGraphSchema)),
              )
            }
          >
            {t("浏览 Vault 依赖图")}
          </button>
          {graph && (
            <div className="space-y-3">
              <input
                aria-label={t("搜索知识概念")}
                placeholder={t("搜索知识概念")}
                className="interaction-input"
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
                          .filter(
                            ({ source, target }) => visible.has(source) && visible.has(target),
                          )
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
                    <select
                      className="interaction-input"
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
                    </select>
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
        </>
      )}
    </section>
  );
}
