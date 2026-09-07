import type { ThreadMessage } from "@assistant-ui/react";
import { type SpanData, SpanPrimitive, SpanResource } from "@assistant-ui/react-o11y";
import { AuiConfig, AuiProvider, useAuiState } from "@assistant-ui/store";
import { useMemo } from "react";
import { t, useTranslation } from "./i18n.js";

export function TracePanel() {
  const { t } = useTranslation();
  const messages = useAuiState((state) => state.thread.messages);
  const spans = useMemo(
    () =>
      traceSpans(messages).map((span) => ({
        ...span,
        name:
          span.name === "Swarm response"
            ? t("Swarm 回复")
            : span.name === "Agent response"
              ? t("Agent 回复")
              : span.name,
      })),
    [messages, t],
  );
  const config = useMemo(() => AuiConfig({ span: SpanResource({ spans }) }), [spans]);
  if (spans.length === 0)
    return (
      <p className="px-3 py-10 text-center text-sm text-neutral-500">
        {t("任务开始后，执行轨迹会显示在这里。")}
      </p>
    );
  return (
    <details className="overflow-hidden rounded-xl border border-neutral-200 bg-white" open>
      <summary className="flex cursor-pointer justify-between px-3.5 py-2.5 font-bold text-neutral-700">
        <span>{t("运行与调用")}</span>
        <small className="font-normal text-neutral-500">
          {t("{{count}} 项", { count: spans.length })}
        </small>
      </summary>
      <AuiProvider config={config} extends={null}>
        <SpanPrimitive.Timeline
          className="overflow-auto border-neutral-200 border-t"
          paddingEnd={0.04}
        >
          <SpanPrimitive.Children>
            {({ span }) => (
              <SpanPrimitive.Root
                className="group grid min-h-11 min-w-[280px] grid-cols-[minmax(200px,3fr)_minmax(60px,1fr)] border-neutral-100 border-b"
                title={`${span.id} · ${span.type} · ${span.status}`}
              >
                <SpanPrimitive.Indent
                  baseIndent={8}
                  className="grid min-w-0 grid-cols-[12px_6px_minmax(60px,1fr)_auto] items-center gap-1.5 pr-1"
                  indentPerLevel={10}
                >
                  {span.hasChildren ? (
                    <SpanPrimitive.CollapseToggle
                      aria-label={t("折叠 {{name}}", { name: span.name })}
                      className="border-0 bg-transparent p-0 text-neutral-500 group-data-[collapsed=true]:-rotate-90"
                      type="button"
                    >
                      ▾
                    </SpanPrimitive.CollapseToggle>
                  ) : (
                    <span />
                  )}
                  <SpanPrimitive.StatusIndicator className="h-2 w-2 rounded-full bg-neutral-500 data-[span-status=failed]:rounded-none data-[span-status=running]:animate-pulse" />
                  <div className="min-w-0 py-1.5">
                    <SpanPrimitive.Name className="block truncate text-xs" />
                    <span className="text-[10px] text-neutral-500">
                      {t(
                        (
                          {
                            running: "运行中",
                            completed: "已完成",
                            failed: "失败",
                            skipped: "已停止",
                          } as Record<string, string>
                        )[span.status] ?? span.status,
                      )}
                    </span>
                  </div>
                  <time className="text-[10px] text-neutral-500">{duration(span.latencyMs)}</time>
                </SpanPrimitive.Indent>
                <div className="relative mx-2 my-5 rounded-full bg-neutral-200">
                  <SpanPrimitive.TimelineBar className="inset-y-0 rounded-full bg-black [--span-timeline-min-width:4px] data-[span-type=agent]:bg-neutral-500" />
                </div>
              </SpanPrimitive.Root>
            )}
          </SpanPrimitive.Children>
        </SpanPrimitive.Timeline>
      </AuiProvider>
    </details>
  );
}

export function traceSpans(messages: readonly ThreadMessage[]): SpanData[] {
  const spans: SpanData[] = [];
  append(messages, null);
  return spans;

  function append(items: readonly ThreadMessage[], parentSpanId: string | null): void {
    for (const message of items) {
      if (message.role !== "assistant") continue;
      const startedAt = message.createdAt.getTime();
      const status = messageStatus(message);
      const endedAt = status === "running" ? null : messageEnd(message, startedAt);
      const runId = `run:${message.id}`;
      spans.push({
        id: runId,
        parentSpanId,
        name: parentSpanId === null ? "Swarm response" : "Agent response",
        type: parentSpanId === null ? "run" : "agent",
        status,
        startedAt,
        endedAt,
        latencyMs: endedAt === null ? null : endedAt - startedAt,
      });
      for (const part of message.content) {
        if (part.type !== "tool-call") continue;
        const id = `tool:${part.toolCallId}`;
        const toolStartedAt = part.timing?.startedAt ?? startedAt;
        const toolStatus =
          part.isError === true
            ? "failed"
            : part.result !== undefined || part.timing?.completedAt !== undefined
              ? "completed"
              : status === "running"
                ? "running"
                : status === "failed"
                  ? "failed"
                  : "skipped";
        const toolEndedAt = toolStatus === "running" ? null : (part.timing?.completedAt ?? endedAt);
        spans.push({
          id,
          parentSpanId: runId,
          name: part.toolName,
          type: part.toolName === "swarm" ? "agent" : "tool",
          status: toolStatus,
          startedAt: toolStartedAt,
          endedAt: toolEndedAt,
          latencyMs: toolEndedAt === null ? null : Math.max(0, toolEndedAt - toolStartedAt),
        });
        if (part.messages !== undefined) append(part.messages, id);
      }
    }
  }
}

function messageStatus(message: Extract<ThreadMessage, { role: "assistant" }>): SpanData["status"] {
  if (message.status.type === "running" || message.status.type === "requires-action") {
    return "running";
  }
  if (message.status.type === "incomplete") {
    return message.status.reason === "cancelled" ? "skipped" : "failed";
  }
  return "completed";
}

function messageEnd(
  message: Extract<ThreadMessage, { role: "assistant" }>,
  startedAt: number,
): number {
  let endedAt = startedAt + (message.metadata.timing?.totalStreamTime ?? 0);
  for (const part of message.content) {
    if (part.type === "tool-call" && part.timing?.completedAt !== undefined) {
      endedAt = Math.max(endedAt, part.timing.completedAt);
    }
  }
  return endedAt;
}

function duration(milliseconds: number | null): string {
  if (milliseconds === null) return t("进行中");
  return milliseconds < 1_000
    ? `${String(Math.round(milliseconds))} ms`
    : `${(milliseconds / 1_000).toFixed(1)} s`;
}
