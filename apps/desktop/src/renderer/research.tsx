import {
  MAX_SCIENCE_IMPORT_BYTES,
  notebookExecutionSummarySchema,
  roCrateMetadataDocumentSchema,
  type ScienceArtifact,
  type ScienceNotebook,
  scienceArtifactSchema,
  scienceProjectExportSchema,
  scienceProjectSchema,
  scienceResearchRecordSchema,
  scienceWorkspaceSnapshotSchema,
} from "@swarmx/science/types";
import { type ReactNode, useEffect, useState } from "react";
import { z } from "zod";
import { api, download, jsonRequest, projectUrl, scienceTool } from "./api.js";
import { ArtifactPreview } from "./artifact-preview.js";
import { FigureStudio } from "./figure-studio.js";
import { i18n, t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";
import { ResearchGraph } from "./research-graph.js";

type Snapshot = z.infer<typeof scienceWorkspaceSnapshotSchema>;
type Studio = {
  outputPath?: string;
  previewArtifactId?: string;
  notebook?: ScienceNotebook | undefined;
  source?: string | undefined;
  inputs?: string[] | undefined;
};
const VIEWS = {
  trace: "执行轨迹",
  runs: "运行记录",
  graph: "RO-Crate 图谱",
  crate: "JSON-LD",
} as const;

export function ResearchPanel({
  mode,
  trace,
  target,
  onClose,
  canCompose = false,
}: {
  mode: "assets" | "observe";
  trace?: ReactNode;
  target?: { artifactId?: string; projectId?: string } | undefined;
  onClose(): void;
  canCompose?: boolean;
}) {
  useTranslation();
  const [snapshot, setSnapshot] = useState<Snapshot>();
  const [crate, setCrate] = useState<z.infer<typeof roCrateMetadataDocumentSchema>>();
  const [executions, setExecutions] = useState<z.infer<typeof notebookExecutionSummarySchema>[]>(
    [],
  );
  const [projectId, setProjectId] = useState("");
  const [selected, setSelected] = useState("");
  const [observeView, setView] = useState<keyof typeof VIEWS>("trace");
  const view = mode === "assets" ? "artifacts" : observeView;
  const [query, setQuery] = useState("");
  const [neighborhood, setNeighborhood] = useState(false);
  const [revision, setRevision] = useState(0);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [createForm, setCreateForm] = useState(false);
  const [questionForm, setQuestionForm] = useState(false);
  const [studio, setStudio] = useState<Studio>();
  const [notice, setNotice] = useState("");
  useEffect(() => {
    if (mode === "observe") setNotice("");
  }, [mode]);
  useEffect(() => {
    if (!target) return;
    if (target.projectId) setProjectId(target.projectId);
    if (target.artifactId) setSelected(`urn:uuid:${target.artifactId}`);
    setStudio(undefined);
  }, [target]);
  const ask = (text: string) => {
    window.dispatchEvent(new CustomEvent("swarmx:compose", { detail: text }));
    setNotice("已添加到对话草稿，请补充要求后发送。");
    if (!matchMedia("(min-width: 1024px)").matches) onClose();
  };
  const refresh = () => setRevision((value) => value + 1);
  useEffect(() => {
    const changed = () => setRevision((value) => value + 1);
    window.addEventListener("swarmx:science-changed", changed);
    return () => window.removeEventListener("swarmx:science-changed", changed);
  }, []);
  // biome-ignore lint/correctness/useExhaustiveDependencies: revision explicitly refreshes authoritative server state.
  useEffect(() => {
    const abort = new AbortController();
    void api("/api/v1/science", scienceWorkspaceSnapshotSchema, { signal: abort.signal })
      .then((value) => {
        setSnapshot(value);
        setProjectId((id) =>
          value.projects.some((project) => project.id === id) ? id : (value.projects[0]?.id ?? ""),
        );
      })
      .catch((cause: Error) => {
        if (!abort.signal.aborted) setError(cause.message);
      });
    return () => abort.abort();
  }, [revision]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: mutations refresh the current project's RO-Crate.
  useEffect(() => {
    setCrate(undefined);
    setExecutions([]);
    if (!projectId) return;
    const abort = new AbortController();
    void Promise.all([
      api(
        `/api/v1/research-object?project=${encodeURIComponent(projectId)}`,
        roCrateMetadataDocumentSchema,
        { signal: abort.signal },
      ),
      api(
        `/api/v1/notebook-executions?project=${encodeURIComponent(projectId)}`,
        z.array(notebookExecutionSummarySchema),
        { signal: abort.signal },
      ),
    ])
      .then(([document, executions]) => {
        setCrate(document);
        setExecutions(executions);
      })
      .catch((cause: Error) => {
        if (!abort.signal.aborted) setError(cause.message);
      });
    return () => abort.abort();
  }, [projectId, revision]);
  const perform = async (task: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await task();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  };
  const project = snapshot?.projects.find((project) => project.id === projectId);
  const artifacts =
    snapshot?.artifacts.filter((artifact) => artifact.projectId === projectId) ?? [];
  const notebooks =
    snapshot?.notebooks.filter((notebook) => notebook.projectId === projectId) ?? [];
  const records = snapshot?.records.filter((record) => record.projectId === projectId) ?? [];
  const artifactId = selected.replace(/^urn:uuid:/u, "");
  const artifact = artifacts.find((artifact) => artifact.id === artifactId);
  const notebook = notebooks.find(
    (notebook) =>
      notebook.id === artifactId ||
      notebook.cells.some((cell) => cell.outputArtifactIds.includes(artifactId)),
  );
  const cell =
    notebook?.cells.findLast(
      (cell) => cell.kind === "code" && cell.outputArtifactIds.includes(artifactId),
    ) ?? notebook?.cells.findLast((cell) => cell.kind === "code");
  const entity = crate?.["@graph"].find((entity) => entity["@id"] === selected);
  const visibleArtifacts = artifacts.filter((artifact) =>
    `${artifact.title} ${artifact.mime}`.toLocaleLowerCase().includes(query.toLocaleLowerCase()),
  );
  const edit = (artifact: ScienceArtifact) => {
    window.dispatchEvent(new CustomEvent("swarmx:open-research"));
    if (cell) {
      setStudio({
        notebook,
        source: cell.source,
        inputs: cell.inputArtifactIds,
        previewArtifactId: artifact.id,
      });
      return;
    }
    setStudio({
      previewArtifactId: artifact.id,
      outputPath: "figure.png",
      inputs: [artifact.id],
      source: `import os\nfrom PIL import Image, ImageEnhance\n\nimage = Image.open(os.environ["SWARMX_SCIENCE_INPUT_0"]).convert("RGB")\n# Edit contrast, size or crop here. The original remains unchanged.\nimage = ImageEnhance.Contrast(image).enhance(1.1)\nimage.save("figure.png")\nprint("Saved figure.png")`,
    });
  };
  return (
    <>
      {studio && project && (
        <div
          className={mode === "assets" ? "flex min-h-0 flex-1" : "hidden"}
          hidden={mode !== "assets"}
        >
          <FigureStudio
            projectId={project.id}
            {...studio}
            artifacts={artifacts}
            onClose={() => setStudio(undefined)}
            onResult={(id) => {
              if (id) setSelected(`urn:uuid:${id}`);
              refresh();
            }}
          />
        </div>
      )}
      <aside
        className={studio && mode === "assets" ? "hidden" : "flex min-h-0 flex-1 flex-col"}
        hidden={!!studio && mode === "assets"}
        aria-label={mode === "assets" ? t("科研资产侧栏") : t("观测与溯源侧栏")}
      >
        <header className="flex h-14 shrink-0 items-center gap-3 border-b border-neutral-200 px-4">
          <Icon name={mode === "assets" ? "image" : "trace"} />
          <h2 className="mr-auto font-medium">
            {mode === "assets" ? t("科研资产") : t("观测与溯源")}
          </h2>
          <button
            className="icon-button"
            type="button"
            aria-label={t("关闭侧栏")}
            onClick={onClose}
          >
            <Icon name="close" />
          </button>
        </header>
        {mode === "assets" && (
          <div className="flex flex-wrap items-center gap-2 border-b border-neutral-100 px-4 py-2">
            {canCompose && (
              <button
                className="secondary-button"
                type="button"
                onClick={() =>
                  ask(t("请根据当前研究数据生成科研图像，记录代码、输入、环境和成果。我的要求："))
                }
              >
                <Icon name="compose" />
                {t("在对话中生成")}
              </button>
            )}
            <button
              type="button"
              className="secondary-button ml-auto"
              onClick={() => setCreateForm((value) => !value)}
            >
              <Icon name="plus" />
              {t("管理研究集合")}
            </button>
            {project && (
              <button className="secondary-button" type="button" onClick={() => setStudio({})}>
                <Icon name="code" />
                {t("编写图像代码")}
              </button>
            )}
          </div>
        )}
        {mode === "observe" && (
          <div
            className="flex flex-wrap border-b border-neutral-200 px-3"
            role="tablist"
            aria-label={t("观测视图")}
          >
            {Object.entries(VIEWS).map(([key, label]) => (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={view === key}
                className="research-tab"
                onClick={() => setView(key as keyof typeof VIEWS)}
              >
                {t(label)}
              </button>
            ))}
          </div>
        )}
        {notice && (
          <p role="status" className="px-4 py-2 text-xs text-neutral-500">
            {t(notice)}
          </p>
        )}
        {error && (
          <p role="alert" className="workbench-alert mx-5 mb-3">
            {error}
          </p>
        )}
        {createForm && mode === "assets" && (
          <form
            className="mx-5 mb-4 flex flex-wrap items-end gap-3 rounded-xl border border-neutral-200 bg-neutral-50 p-5"
            onSubmit={(event) => {
              event.preventDefault();
              const title = new FormData(event.currentTarget).get("title");
              void perform(async () => {
                const project = await scienceTool(
                  "science_notebook",
                  "create_project",
                  { requestId: crypto.randomUUID(), title },
                  scienceProjectSchema,
                );
                setProjectId(project.id);
                setSelected("");
                setCreateForm(false);
              });
            }}
          >
            <label className="field-label min-w-52 flex-1">
              {t("研究集合名称")}
              <input
                required
                name="title"
                className="interaction-input"
                placeholder={t("例如：作物萌发数据分析")}
                maxLength={240}
              />
            </label>
            <button disabled={busy} className="primary-button" type="submit">
              {t("创建研究集合")}
            </button>
            <p className="w-full text-xs leading-6 text-neutral-500">
              {t("把问题、数据、分析和成果组织在一起。每个成果都可追溯到来源。")}
            </p>
          </form>
        )}
        {view === "trace" ? (
          <div className="min-h-0 flex-1 overflow-auto p-3">
            {trace}
            <p className="mt-4 text-xs text-neutral-500">
              {t("当前会话的运行与工具调用；科研来源可在 RO-Crate 中检查。")}
            </p>
          </div>
        ) : !snapshot ? (
          <p className="p-7" role="status">
            {t("正在加载研究记录…")}
          </p>
        ) : !project ? (
          <div className="grid flex-1 place-content-center gap-4 p-8 text-center text-neutral-400">
            <Icon name="graph" className="mx-auto size-12" />
            <p>{t("在对话中描述研究任务，助手创建的成果会显示在这里。")}</p>
          </div>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-3 border-b border-neutral-200 px-5 md:px-7">
              <span className="mr-auto text-xs text-neutral-500">{t("项目中的研究集合")}</span>
              <select
                className="ml-auto max-w-48 rounded-md border border-neutral-200 px-2 py-1 text-xs"
                aria-label={t("选择研究集合")}
                value={projectId}
                onChange={(event) => {
                  setProjectId(event.target.value);
                  setSelected("");
                  setQuery("");
                }}
              >
                {snapshot.projects.map((project) => (
                  <option key={project.id} value={project.id}>
                    {project.title}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="icon-button"
                aria-label={t("刷新研究记录")}
                onClick={refresh}
              >
                <Icon name="refresh" />
              </button>
            </div>
            <div className="flex flex-wrap items-center gap-3 border-b border-neutral-100 px-5 py-3 md:px-7">
              <label className="flex min-w-40 flex-1 items-center gap-2 text-neutral-400">
                <Icon name="search" />
                <input
                  type="search"
                  aria-label={t("搜索研究实体")}
                  placeholder={t("搜索成果、实体或类型…")}
                  className="w-full bg-transparent text-xs outline-none"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
              </label>
              {view === "graph" && (
                <label className="flex items-center gap-2 text-xs text-neutral-500">
                  <input
                    type="checkbox"
                    checked={neighborhood}
                    onChange={(event) => setNeighborhood(event.target.checked)}
                  />
                  {t("选中节点的邻居")}
                </label>
              )}
              <button
                className="text-xs text-neutral-500 hover:text-neutral-950"
                type="button"
                onClick={() => setQuestionForm((value) => !value)}
              >
                {t("记录研究问题")}
              </button>
              <label className="secondary-button cursor-pointer text-xs">
                <Icon name="plus" />
                {t("导入文件")}
                <input
                  className="sr-only"
                  type="file"
                  aria-label={t("导入研究文件")}
                  disabled={busy}
                  accept=".csv,.tsv,.json,.txt,.md,.png,.jpg,.jpeg,.webp,.gif,.pdf,.ipynb"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (!file) return;
                    void perform(async () => {
                      if (file.size > MAX_SCIENCE_IMPORT_BYTES)
                        throw new Error(t("导入文件不能超过 8 MiB。"));
                      const data = await new Promise<string>((resolve, reject) => {
                        const reader = new FileReader();
                        reader.onload = () => resolve(String(reader.result).split(",")[1] ?? "");
                        reader.onerror = () => reject(reader.error);
                        reader.readAsDataURL(file);
                      });
                      const artifact = await api(
                        "/api/v1/artifacts",
                        scienceArtifactSchema,
                        jsonRequest({
                          requestId: crypto.randomUUID(),
                          projectId,
                          name: file.name,
                          dataBase64: data,
                        }),
                      );
                      setSelected(`urn:uuid:${artifact.id}`);
                      refresh();
                    });
                    event.target.value = "";
                  }}
                />
              </label>
            </div>
            {questionForm && (
              <form
                className="mx-5 my-3 grid gap-3 rounded-xl border border-neutral-200 p-4"
                onSubmit={(event) => {
                  event.preventDefault();
                  const data = new FormData(event.currentTarget);
                  void perform(async () => {
                    await scienceTool(
                      "science_record",
                      "create_question",
                      {
                        requestId: crypto.randomUUID(),
                        projectId,
                        title: data.get("title"),
                        summary: data.get("summary"),
                        tags: [],
                      },
                      scienceResearchRecordSchema,
                    );
                    setQuestionForm(false);
                  });
                }}
              >
                <label className="field-label">
                  {t("研究问题")}
                  <input name="title" required className="interaction-input" />
                </label>
                <label className="field-label">
                  {t("背景与目的")}
                  <textarea name="summary" required className="interaction-input" />
                </label>
                <button className="primary-button justify-self-end" type="submit" disabled={busy}>
                  {t("保存研究问题")}
                </button>
              </form>
            )}
            <div className="min-h-0 flex-1 overflow-auto">
              <div className="min-w-0">
                {view === "artifacts" && !entity && (
                  <div className="space-y-7 p-5 md:p-7">
                    {visibleArtifacts.length > 0 ? (
                      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 2xl:grid-cols-3">
                        {visibleArtifacts.map((artifact) => (
                          <button
                            key={artifact.id}
                            type="button"
                            className="overflow-hidden rounded-xl border border-neutral-200 text-left transition-shadow hover:shadow-md aria-pressed:ring-1 aria-pressed:ring-neutral-800"
                            aria-pressed={artifact.id === artifactId}
                            onClick={() => setSelected(`urn:uuid:${artifact.id}`)}
                          >
                            <ArtifactPreview id={artifact.id} compact />
                            <div className="border-t border-neutral-100 px-4 py-3">
                              <p className="truncate font-medium">{artifact.title}</p>
                              <p className="mt-1 text-[11px] text-neutral-400">
                                {artifact.mime} · {(artifact.size / 1024).toFixed(1)} KiB
                              </p>
                            </div>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <div className="rounded-xl border border-dashed border-neutral-300 px-6 py-12 text-center">
                        <Icon name="image" className="mx-auto mb-4 size-8 text-neutral-400" />
                        <h3 className="font-medium">
                          {query ? t("没有匹配的成果") : t("让研究有迹可循")}
                        </h3>
                        <p className="mt-2 text-xs leading-6 text-neutral-500">
                          {t("在对话中发起分析，或导入数据。")}
                          <br />
                          {t("代码、环境与结果会一同记录。")}
                        </p>
                      </div>
                    )}
                    {notebooks.length > 0 && (
                      <div>
                        <p className="eyebrow mb-3">{t("分析笔记本")}</p>
                        {notebooks.map((notebook) => (
                          <button
                            className="entity-row"
                            type="button"
                            key={notebook.id}
                            onClick={() => {
                              setSelected(`urn:uuid:${notebook.id}`);
                              setStudio({
                                notebook,
                                source: notebook.cells.findLast((cell) => cell.kind === "code")
                                  ?.source,
                                inputs: notebook.cells.findLast((cell) => cell.kind === "code")
                                  ?.inputArtifactIds,
                              });
                            }}
                          >
                            <Icon name="code" />
                            <span className="truncate">{notebook.title}</span>
                            <span className="ml-auto text-xs text-neutral-400">
                              {t("{{count}} 次执行 · v{{revision}}", {
                                count: notebook.cells.filter((cell) => cell.kind === "code").length,
                                revision: notebook.revision,
                              })}
                            </span>
                          </button>
                        ))}
                      </div>
                    )}
                    {records.length > 0 && (
                      <div>
                        <p className="eyebrow mb-3">{t("研究记录")}</p>
                        {records
                          .filter((record) => `${record.title} ${record.summary}`.includes(query))
                          .map((record) => (
                            <button
                              className="entity-row"
                              type="button"
                              key={record.id}
                              onClick={() => setSelected(`urn:uuid:${record.id}`)}
                            >
                              <Icon name="book" />
                              <span className="truncate">{record.title}</span>
                              <span className="status-pill ml-auto">{record.kind}</span>
                            </button>
                          ))}
                      </div>
                    )}
                  </div>
                )}
                {view === "graph" && crate && (
                  <div className="h-[420px]">
                    <ResearchGraph
                      document={crate}
                      query={query}
                      selected={selected}
                      neighborhood={neighborhood}
                      onSelect={setSelected}
                    />
                  </div>
                )}
                {view === "runs" && (
                  <div className="p-5 md:p-7">
                    <p className="mb-4 text-xs leading-6 text-neutral-500">
                      {t("显示最近 100 次 Notebook 执行，以及项目实验运行。状态来自实际记录。")}
                    </p>
                    {executions
                      .filter((execution) =>
                        `${notebooks.find((notebook) => notebook.id === execution.notebookId)?.title} ${execution.status}`
                          .toLocaleLowerCase()
                          .includes(query.toLocaleLowerCase()),
                      )
                      .map((execution) => (
                        <details
                          key={execution.id}
                          className="mb-3 rounded-lg border border-neutral-200 p-4"
                        >
                          <summary className="flex cursor-pointer items-center gap-3 text-xs">
                            <Icon name="code" />
                            <span className="min-w-0 flex-1 truncate">
                              {
                                notebooks.find((notebook) => notebook.id === execution.notebookId)
                                  ?.title
                              }{" "}
                              · #{execution.executionCount}
                            </span>
                            <span className="status-pill">
                              {execution.status === "succeeded" ? t("成功") : t("失败")}
                            </span>
                            <span className="text-neutral-400">
                              {Math.round(execution.durationMs)} ms
                            </span>
                          </summary>
                          <pre className="code-panel mt-4">
                            {
                              notebooks
                                .find((notebook) => notebook.id === execution.notebookId)
                                ?.cells.find((cell) => cell.id === execution.cellId)?.source
                            }
                          </pre>
                          <pre className="code-panel mt-2">
                            {[execution.stdout.text, execution.stderr.text]
                              .filter(Boolean)
                              .join("\n") || t("无标准输出")}
                          </pre>
                          <details className="mt-3 text-xs">
                            <summary className="cursor-pointer">{t("运行环境与退出状态")}</summary>
                            <pre className="code-panel mt-2">
                              {JSON.stringify(
                                {
                                  environment: execution.environment,
                                  exitCode: execution.exitCode,
                                  signal: execution.signal,
                                },
                                null,
                                2,
                              )}
                            </pre>
                          </details>
                          {execution.artifact && (
                            <button
                              type="button"
                              className="secondary-button mt-3"
                              onClick={() => setSelected(`urn:uuid:${execution.artifact?.id}`)}
                            >
                              {t("查看输出成果")}
                            </button>
                          )}
                        </details>
                      ))}
                    {snapshot.runs
                      .filter(
                        (run) =>
                          run.projectId === projectId && `${run.id} ${run.status}`.includes(query),
                      )
                      .map((run) => (
                        <button
                          key={run.id}
                          className="entity-row"
                          type="button"
                          onClick={() => setSelected(`urn:uuid:${run.id}`)}
                        >
                          <Icon name="play" />
                          <span className="status-pill">
                            {run.status === "succeeded"
                              ? t("成功")
                              : run.status === "failed"
                                ? t("失败")
                                : run.status === "cancelled"
                                  ? t("已取消")
                                  : t("运行中")}
                          </span>
                          <time className="text-xs">
                            {new Date(run.startedAt).toLocaleString(i18n.language)}
                          </time>
                          <span className="ml-auto text-xs text-neutral-400">
                            {t("{{count}} 个输出", { count: run.artifactIds.length })}
                          </span>
                        </button>
                      ))}
                    {executions.length === 0 &&
                      !snapshot.runs.some((run) => run.projectId === projectId) && (
                        <p className="py-12 text-center text-neutral-400">
                          {t("尚无运行记录。生成图像或在对话中发起分析。")}
                        </p>
                      )}
                  </div>
                )}
                {view === "crate" && (
                  <div className="space-y-4 p-5 md:p-7">
                    <div className="flex items-center gap-3">
                      <div>
                        <h3 className="font-medium">{t("研究对象元数据")}</h3>
                        <p className="mt-1 text-xs text-neutral-500">
                          {t("JSON-LD · {{count}} 个实体 · 保留来源与关系", {
                            count: crate?.["@graph"].length ?? 0,
                          })}
                        </p>
                      </div>
                      <button
                        className="secondary-button ml-auto"
                        type="button"
                        disabled={busy}
                        onClick={() =>
                          void perform(async () => {
                            const exported = await scienceTool(
                              "science_export",
                              "project",
                              { requestId: crypto.randomUUID(), projectId },
                              scienceProjectExportSchema,
                            );
                            download("ro-crate-metadata.json", exported.content);
                          })
                        }
                      >
                        <Icon name="download" />
                        {t("导出 RO-Crate 元数据")}
                      </button>
                    </div>
                    <p className="text-xs leading-6 text-neutral-500">
                      {t(
                        "此文件包含研究对象描述；数据文件保存在本地成果库中，元数据导出不包含原始文件。",
                      )}
                    </p>
                    <pre className="code-panel max-h-none">{JSON.stringify(crate, null, 2)}</pre>
                  </div>
                )}
              </div>
              {entity && (
                <section
                  className="w-full border-neutral-200 border-t bg-white"
                  aria-label={t("科研成果详情")}
                >
                  {
                    <div className="space-y-5 p-5">
                      <div className="flex items-start gap-2">
                        <div className="min-w-0 flex-1">
                          <p className="eyebrow">{t("成果详情")}</p>
                          <h3 className="mt-2 break-words font-semibold">
                            {String(entity.name ?? entity["@id"])}
                          </h3>
                        </div>
                        <button
                          type="button"
                          className="icon-button"
                          aria-label={t("返回全部成果")}
                          onClick={() => setSelected("")}
                        >
                          <Icon name="close" />
                        </button>
                      </div>
                      {artifact && (
                        <>
                          <ArtifactPreview id={artifact.id} />
                          {canCompose && (
                            <button
                              className="primary-button w-full"
                              type="button"
                              onClick={() =>
                                ask(
                                  t(
                                    "请修改科研成果「{{title}}」（artifactId: {{id}}），保留原始成果并生成新版本。我的修改要求：",
                                    { title: artifact.title, id: artifact.id },
                                  ),
                                )
                              }
                            >
                              <Icon name="compose" />
                              {t("让助手修改")}
                            </button>
                          )}
                          <a
                            className="secondary-button w-full"
                            href={projectUrl(
                              `/api/v1/artifacts/${encodeURIComponent(artifact.id)}/content`,
                            )}
                            download
                          >
                            <Icon name="download" />
                            {t("下载原始成果")}
                          </a>
                          {(artifact.mime.startsWith("image/") || cell) && (
                            <button
                              className="secondary-button w-full"
                              type="button"
                              onClick={() => edit(artifact)}
                            >
                              <Icon name="compose" />
                              {t("编辑代码并生成新版本")}
                            </button>
                          )}
                          <dl className="metadata-list">
                            <dt>{t("内容哈希")}</dt>
                            <dd className="break-all font-mono text-[10px]">{artifact.digest}</dd>
                            <dt>{t("记录序号")}</dt>
                            <dd>{artifact.provenance.journalSeq}</dd>
                            <dt>{t("许可证")}</dt>
                            <dd>{artifact.license ?? t("尚未指定")}</dd>
                          </dl>
                        </>
                      )}
                      {cell && (
                        <details open>
                          <summary className="mb-2 cursor-pointer font-medium">
                            {t("生成代码")}
                          </summary>
                          <pre className="code-panel">{cell.source}</pre>
                        </details>
                      )}
                      {artifact && (
                        <details>
                          <summary className="cursor-pointer font-medium">{t("运行环境")}</summary>
                          <pre className="code-panel mt-2">
                            {JSON.stringify(artifact.environment, null, 2)}
                          </pre>
                        </details>
                      )}
                      <details open={!artifact}>
                        <summary className="cursor-pointer font-medium">{t("实体与来源")}</summary>
                        <pre className="code-panel mt-2">{JSON.stringify(entity, null, 2)}</pre>
                      </details>
                    </div>
                  }
                </section>
              )}
            </div>
          </>
        )}
      </aside>
    </>
  );
}
