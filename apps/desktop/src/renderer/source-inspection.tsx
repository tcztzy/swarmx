import { parseScienceResourceId } from "@swarmx/science/resource-id";
import { ScienceResourceResolver } from "@swarmx/science/resource-resolver";
import type {
  notebookExecutionSummarySchema,
  ScienceArtifact,
  ScienceNotebook,
  ScienceWorkspaceSnapshot,
} from "@swarmx/science/types";
import { useMemo, useState } from "react";
import type { z } from "zod";
import { ArtifactContentSchema } from "../bridge-contract.js";
import { ArtifactPreview } from "./artifact-preview.js";
import { bridge, download } from "./bridge.js";
import { Button } from "./components/ui/radix/button.js";
import { CodeBlock } from "./components/ui/radix/code-block.js";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "./components/ui/radix/collapsible.js";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/radix/tabs.js";
import { t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";

export type SourceReference = { resource: string; title?: string | undefined };
type Execution = z.infer<typeof notebookExecutionSummarySchema>;

export function CopyButton({
  value,
  label,
  compact = false,
}: {
  value: string;
  label: string;
  compact?: boolean;
}) {
  const [copiedValue, setCopiedValue] = useState("");
  const [error, setError] = useState("");
  const copied = copiedValue === value;
  return (
    <>
      <Button
        type="button"
        variant={compact ? "ghost" : "outline"}
        size={compact ? "icon-sm" : "sm"}
        aria-label={copied ? t("已复制") : label}
        title={label}
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(value);
            setCopiedValue(value);
            setError("");
          } catch (cause) {
            setError(cause instanceof Error ? cause.message : String(cause));
          }
        }}
      >
        <Icon name={copied ? "check" : "copy"} />
        {!compact && (copied ? t("已复制") : label)}
      </Button>
      {error && (
        <span role="alert" className="text-xs text-red-700">
          {error}
        </span>
      )}
    </>
  );
}

export function SourceInspection({
  source,
  snapshot,
  executions,
  onClose,
}: {
  source: SourceReference;
  snapshot: ScienceWorkspaceSnapshot | undefined;
  executions: Execution[];
  onClose(): void;
}) {
  const { t } = useTranslation();
  const resolved = useMemo(() => {
    if (!snapshot) return undefined;
    try {
      const parsed = parseScienceResourceId(source.resource);
      if (parsed.revision === null)
        throw new Error(t("此来源未指定固定版本，无法作为固定版本检查。"));
      const resource = new ScienceResourceResolver(snapshot).resolve(source.resource);
      if (resource.kind !== "artifact")
        throw new Error(t("此来源不是已登记成果，请在观测视图中检查其记录。"));
      return { artifact: resource.entity as ScienceArtifact };
    } catch (cause) {
      return { error: cause instanceof Error ? cause.message : String(cause) };
    }
  }, [snapshot, source.resource, t]);
  const artifact = resolved?.artifact;
  const notebooks =
    snapshot?.notebooks.filter((notebook) => notebook.projectId === artifact?.projectId) ?? [];
  const producers = executions.filter(
    (execution) => execution.artifact?.id === artifact?.id && execution.artifact !== null,
  );
  const inputIds = new Set([
    ...(artifact?.sourceEntityIds ?? []),
    ...producers.flatMap(
      (execution) =>
        execution.inputArtifactIds ??
        notebooks
          .find((notebook) => notebook.id === execution.notebookId)
          ?.cells.find((cell) => cell.id === execution.cellId)?.inputArtifactIds ??
        [],
    ),
  ]);
  const inputs =
    snapshot?.artifacts.filter(
      (input) =>
        input.projectId === artifact?.projectId &&
        input.id !== artifact?.id &&
        inputIds.has(input.id),
    ) ?? [];
  const related = executions.filter(
    (execution) =>
      !producers.includes(execution) &&
      notebooks.some((notebook) => notebook.id === execution.notebookId) &&
      (
        execution.inputArtifactIds ??
        notebooks
          .find((notebook) => notebook.id === execution.notebookId)
          ?.cells.find((cell) => cell.id === execution.cellId)?.inputArtifactIds ??
        []
      ).some((id) => inputs.some((input) => input.id === id)),
  );
  return (
    <aside className="source-inspection" aria-label={t("来源检查")}>
      <header className="source-heading">
        <div>
          <h2>{t("来源检查")}</h2>
          <p className="source-breadcrumb">
            <span>{t("已保存的发现")}</span>
            <Icon name="arrowRight" />
            <span>
              {artifact?.kind === "figure" ? t("已登记图像") : t("已登记成果")}
              {artifact && ` @${artifact.revision}`}
            </span>
          </p>
          <p className="source-description">{t("从发现的固定来源引用打开")}</p>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label={t("关闭侧栏")}
          onClick={onClose}
        >
          <Icon name="close" />
        </Button>
      </header>
      {resolved?.error && (
        <p role="alert" className="workbench-alert">
          {resolved.error}
        </p>
      )}
      {!snapshot && <p role="status">{t("正在加载来源…")}</p>}
      {artifact && (
        <>
          <section className="source-card">
            <h3 className="source-card-title">
              {source.title ?? artifact.title}
              <span className="source-revision">@{artifact.revision}</span>
            </h3>
            <div className="source-evidence">
              <div className="source-figure-preview">
                <ArtifactPreview id={artifact.id} compact />
              </div>
              <div className="source-identity">
                <h4>{t("固定来源版本")}</h4>
                {inputs.map((input) => (
                  <dl key={input.id}>
                    <dt>{t("输入成果标识")}</dt>
                    <dd className="source-input-id">{input.id}</dd>
                    <dt>{t("输入 SHA-256")}</dt>
                    <dd title={input.digest}>
                      {input.digest.replace(/^sha256:/u, "").slice(0, 12)}…
                    </dd>
                  </dl>
                ))}
                {inputs.length === 0 && (
                  <p className="text-sm text-neutral-500">{t("此成果没有已记录的输入来源。")}</p>
                )}
                <div className="source-actions">
                  {inputs.map((input) => (
                    <CopyButton
                      key={input.id}
                      value={input.digest}
                      label={
                        inputs.length === 1
                          ? t("复制完整哈希")
                          : `${t("复制完整哈希")} · ${input.title}`
                      }
                    />
                  ))}
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="source-original"
                    onClick={() =>
                      void bridge()
                        .science.artifactContent({ id: artifact.id })
                        .then((value) => {
                          const content = ArtifactContentSchema.parse(value);
                          download(content.name, content.bytes, content.mime);
                        })
                    }
                  >
                    <Icon name="external" />
                    {t("查看原始输出")}
                  </Button>
                </div>
              </div>
            </div>
          </section>
          <section className="source-card source-computations">
            <h3 className="source-card-title">{t("已记录的计算")}</h3>
            {[...producers, ...related].map((execution, index) => (
              <RecordedComputation
                key={execution.id}
                execution={execution}
                notebook={notebooks.find((notebook) => notebook.id === execution.notebookId)}
                initialOpen={index === 0}
                isProducer={producers.includes(execution)}
              />
            ))}
            {producers.length + related.length === 0 && (
              <p className="p-4 text-sm text-neutral-500">{t("没有与此来源关联的已记录计算。")}</p>
            )}
          </section>
        </>
      )}
    </aside>
  );
}

function RecordedComputation({
  execution,
  notebook,
  initialOpen,
  isProducer,
}: {
  execution: Execution;
  notebook: ScienceNotebook | undefined;
  initialOpen: boolean;
  isProducer: boolean;
}) {
  const [tab, setTab] = useState("output");
  const output = [execution.stdout.text, execution.stderr.text].filter(Boolean).join("\n");
  const tabs = { code: t("代码"), output: t("输出"), environment: t("运行环境") };
  return (
    <Collapsible defaultOpen={initialOpen} asChild>
      <article className="recorded-computation">
        <CollapsibleTrigger type="button" className="computation-summary group">
          <Icon name="chevron" className="size-4 group-data-[state=open]:rotate-90" />
          <span className="computation-name">
            <strong>{notebook?.title ?? execution.notebookId}</strong>
            <small>
              {isProducer
                ? execution.artifact?.kind === "figure"
                  ? t("绘图笔记本")
                  : t("来源笔记本")
                : t("关联输入笔记本")}{" "}
              · {execution.notebookId.slice(0, 8)}…
            </small>
          </span>
          <span className={`computation-status ${execution.status}`}>
            <Icon
              name={execution.status === "succeeded" ? "checkCircle" : "errorCircle"}
              className="size-6"
            />
            {execution.status === "succeeded" ? t("执行成功") : t("失败")}
          </span>
        </CollapsibleTrigger>
        <CollapsibleContent className="computation-body">
          <Tabs value={tab} onValueChange={setTab}>
            <div className="computation-toolbar">
              <TabsList variant="line" aria-label={t("计算详情")}>
                {(Object.keys(tabs) as (keyof typeof tabs)[]).map((key) => (
                  <TabsTrigger key={key} value={key}>
                    {tabs[key]}
                  </TabsTrigger>
                ))}
              </TabsList>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="inspect-code"
                onClick={() => setTab(tab === "code" ? "environment" : "code")}
              >
                <Icon name="external" />
                {t("检查代码和环境")}
              </Button>
            </div>
            {Object.keys(tabs).map((key) => (
              <TabsContent key={key} value={key}>
                <CodeBlock
                  title={tabs[key as keyof typeof tabs]}
                  viewportClassName="max-h-72 overflow-auto"
                >
                  <pre className="whitespace-pre-wrap break-words">
                    {tab === "code"
                      ? execution.source
                      : tab === "output"
                        ? readableOutput(output) || t("无标准输出")
                        : JSON.stringify(
                            {
                              ...execution.environment,
                              exitCode: execution.exitCode,
                              signal: execution.signal,
                            },
                            null,
                            2,
                          )}
                  </pre>
                </CodeBlock>
                {tab === "output" && (execution.stdout.truncated || execution.stderr.truncated) && (
                  <p className="text-xs text-neutral-500">{t("输出已截断，请检查原始记录。")}</p>
                )}
              </TabsContent>
            ))}
          </Tabs>
        </CollapsibleContent>
      </article>
    </Collapsible>
  );
}

function readableOutput(value: string): string {
  try {
    const output: unknown = JSON.parse(value);
    if (
      typeof output === "object" &&
      output !== null &&
      !Array.isArray(output) &&
      Object.values(output).every(
        (item) => item === null || ["string", "number", "boolean"].includes(typeof item),
      )
    ) {
      return Object.entries(output)
        .map(([key, item]) => `${key}: ${String(item)}`)
        .join("\n");
    }
    return JSON.stringify(output, null, 2);
  } catch {
    return value;
  }
}
