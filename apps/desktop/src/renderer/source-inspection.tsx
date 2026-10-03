import { useEffect, useState } from "react";
import { type ExecutionEvidence, ExecutionEvidenceSchema } from "../execution-record.js";
import { bridge } from "./bridge.js";
import { TooltipIconButton } from "./components/assistant-ui/elements/tooltip-icon-button.js";
import { Button } from "./components/ui/radix/button.js";
import { CodeBlock } from "./components/ui/radix/code-block.js";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "./components/ui/radix/collapsible.js";
import { ExportEvaluationButton } from "./evaluation-export.js";
import { t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";

export type SourceReference = { resource: string; title?: string | undefined };
type SourceInspectionProps = {
  source: SourceReference;
  onClose(): void;
};

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

export function SourceInspection({ source, onClose }: SourceInspectionProps) {
  const { t } = useTranslation();
  const [evidence, setEvidence] = useState<ExecutionEvidence>();
  const [error, setError] = useState("");
  useEffect(() => {
    setEvidence(undefined);
    setError("");
    if (!source.resource.startsWith("urn:swarmx:execution:")) {
      setError(t("此来源由外部应用管理。请复制引用，在原应用中检查。"));
      return;
    }
    let current = true;
    void bridge()
      .logs.evidence({ sources: [source.resource] })
      .then((value) => {
        const result = ExecutionEvidenceSchema.parse(value);
        if (current) setEvidence(result);
      })
      .catch((cause: unknown) => {
        if (current) setError(cause instanceof Error ? cause.message : String(cause));
      });
    return () => {
      current = false;
    };
  }, [source.resource, t]);
  const unknown = t("未记录");
  const elapsed = (milliseconds: number | null) =>
    milliseconds === null ? unknown : `${(milliseconds / 1000).toFixed(2)} s`;
  const statistics = evidence?.statistics;
  const outcomes = {
    completed: "已结束",
    error: "失败",
    cancelled: "已取消",
    incomplete: "未结束",
    other: "其他终态",
  };
  return (
    <aside className="source-inspection" aria-label={t("来源检查")}>
      <header className="source-heading">
        <div>
          <h2>{t("执行证据")}</h2>
          <p className="source-description">{source.title ?? t("来源引用")}</p>
        </div>
        <TooltipIconButton tooltip={t("关闭侧栏")} className="size-8" onClick={onClose}>
          <Icon name="close" />
        </TooltipIconButton>
      </header>
      <div className="mb-4 flex items-center gap-2">
        <code className="min-w-0 flex-1 break-all text-xs">{source.resource}</code>
        <CopyButton compact value={source.resource} label={t("复制来源引用")} />
      </div>
      {error && (
        <p role="alert" className="workbench-alert">
          {error}
        </p>
      )}
      {!evidence && !error && <p role="status">{t("正在加载来源…")}</p>}
      {evidence?.records.some(
        (record) =>
          `urn:swarmx:execution:${record.id}` === source.resource &&
          record.event.type === "CUSTOM" &&
          record.event.name === "swarmx.memory.review.started",
      ) && (
        <ExportEvaluationButton
          request={{ source: source.resource }}
          filename={`review-${source.resource.split(":").at(-1)}-ro-crate.zip`}
        />
      )}
      {statistics && (
        <section className="source-card source-identity">
          <h3 className="source-card-title">{t("引用到的执行")}</h3>
          <p>{t("{{count}} 个执行样本", { count: statistics.sampleCount })}</p>
          <p>
            {t(
              "{{completed}} 已结束 · {{error}} 失败 · {{cancelled}} 已取消 · {{incomplete}} 未结束 · {{other}} 其他",
              statistics,
            )}
          </p>
          <p className="text-sm text-neutral-500">
            {t("正常结束不代表任务正确；取消不计为成功。")}
          </p>
          <dl>
            <dt>{t("观测时间范围")}</dt>
            <dd>
              {statistics.window.startedAt ?? unknown} — {statistics.window.finishedAt ?? unknown}
            </dd>
            <dt>{t("墙钟时长中位数（含工具和等待）")}</dt>
            <dd>{elapsed(statistics.elapsed.medianMs)}</dd>
            <dt>{t("已记录时长的样本")}</dt>
            <dd>
              {statistics.elapsed.sampleCount} / {statistics.sampleCount}
            </dd>
            <dt>{t("输入 / 输出 Token")}</dt>
            <dd>
              {statistics.usage.inputTokens ?? unknown} / {statistics.usage.outputTokens ?? unknown}
            </dd>
            <dt>{t("已记录用量的样本")}</dt>
            <dd>
              {statistics.usage.sampleCount} / {statistics.sampleCount}
            </dd>
            <dt>{t("已记录费用（USD）")}</dt>
            <dd>{statistics.cost.usd ?? unknown}</dd>
            <dt>{t("已记录费用的样本")}</dt>
            <dd>
              {statistics.cost.sampleCount} / {statistics.sampleCount}
            </dd>
          </dl>
        </section>
      )}
      {evidence?.runs.map((run) => (
        <section key={run.runId} className="source-card source-identity">
          <h3 className="source-card-title">{run.task ?? t("执行任务")}</h3>
          <p>
            {t(outcomes[run.outcome])} · {t("墙钟时长")}: {elapsed(run.elapsedMs)}
          </p>
          <dl>
            <dt>{t("Harness")}</dt>
            <dd>{run.harness ?? unknown}</dd>
            <dt>{t("请求的模型")}</dt>
            <dd>{run.requestedModel ?? unknown}</dd>
            <dt>{t("请求的推理强度")}</dt>
            <dd>{run.requestedEffort ?? unknown}</dd>
            <dt>{t("自身费用（USD）")}</dt>
            <dd>{run.costUsd ?? unknown}</dd>
            <dt>{t("含下级的总费用（USD）")}</dt>
            <dd>{run.totalCostUsd ?? t("费用不完整")}</dd>
            <dt>{t("运行时报告的 Provider")}</dt>
            <dd>{run.provider ?? unknown}</dd>
            <dt>{t("Harness 版本")}</dt>
            <dd>{run.harnessVersion ?? unknown}</dd>
            <dt>{t("模型版本")}</dt>
            <dd>{run.modelVersion ?? unknown}</dd>
            <dt>{t("运行配置")}</dt>
            <dd>{run.profile ?? unknown}</dd>
            <dt>{t("用量与费用的记录依据")}</dt>
            <dd>{run.usageBasis ?? unknown}</dd>
          </dl>
        </section>
      ))}
      {evidence && (
        <section className="source-card source-computations">
          <h3 className="source-card-title">{t("引用的原始记录")}</h3>
          {evidence.records.map((record) => {
            const label =
              record.event.type === "RUN_STARTED"
                ? "原始输入"
                : record.event.type === "TEXT_MESSAGE_CHUNK" ||
                    record.event.type === "REASONING_MESSAGE_CHUNK"
                  ? "原始输出"
                  : record.event.type === "RUN_FINISHED" || record.event.type === "RUN_ERROR"
                    ? "原始终态"
                    : record.event.type === "CUSTOM" &&
                        record.event.name === "swarmx.memory.review.started"
                      ? "复盘快照"
                      : record.event.type === "CUSTOM" &&
                          record.event.name === "swarmx.memory.review.planned"
                        ? "复盘结论与评价者"
                        : record.event.type === "CUSTOM" &&
                            record.event.name === "swarmx.memory.review.response"
                          ? "复盘原始回复"
                          : "原始事件";
            const reference = `urn:swarmx:execution:${record.id}`;
            return (
              <Collapsible key={record.id} asChild>
                <article className="recorded-computation">
                  <CollapsibleTrigger className="computation-summary group">
                    <Icon name="chevron" className="size-4 group-data-[state=open]:rotate-90" />
                    <span className="computation-name">
                      <strong>{t(label)}</strong>
                      <small>{record.observedAt}</small>
                    </span>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="computation-body">
                    <div className="source-actions">
                      <code className="break-all text-xs">{reference}</code>
                      <CopyButton compact value={reference} label={t("复制来源引用")} />
                    </div>
                    <CodeBlock title={t(label)} viewportClassName="max-h-96 overflow-auto">
                      <pre className="whitespace-pre-wrap break-words">
                        {JSON.stringify(record, null, 2)}
                      </pre>
                    </CodeBlock>
                  </CollapsibleContent>
                </article>
              </Collapsible>
            );
          })}
        </section>
      )}
    </aside>
  );
}
