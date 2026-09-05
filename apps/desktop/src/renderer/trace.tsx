import type { ThreadMessage } from "@assistant-ui/react";
import { type SpanData, SpanPrimitive, SpanResource } from "@assistant-ui/react-o11y";
import { AuiConfig, AuiProvider, useAuiState } from "@assistant-ui/store";
import { useMemo } from "react";

export function TracePanel() {
  const messages = useAuiState((state) => state.thread.messages);
  const spans = useMemo(() => traceSpans(messages), [messages]);
  const config = useMemo(() => AuiConfig({ span: SpanResource({ spans }) }), [spans]);
  if (spans.length === 0) return null;
  return (
    <details
      className="mx-auto mb-7 max-w-5xl overflow-hidden rounded-xl border border-neutral-200 bg-white"
      open
    >
      <summary className="flex cursor-pointer justify-between px-3.5 py-2.5 font-bold text-neutral-700">
        <span>执行轨迹</span>
        <small className="font-normal text-neutral-500">{spans.length} spans</small>
      </summary>
      <AuiProvider config={config} extends={null}>
        <SpanPrimitive.Timeline
          className="max-h-60 overflow-auto border-neutral-200 border-t"
          paddingEnd={0.04}
        >
          <SpanPrimitive.Children>
            {({ span }) => (
              <SpanPrimitive.Root className="group grid min-h-8 min-w-[640px] grid-cols-[minmax(420px,3fr)_minmax(140px,2fr)] border-neutral-100 border-b">
                <SpanPrimitive.Indent
                  baseIndent={8}
                  className="grid min-w-0 grid-cols-[18px_8px_auto_minmax(90px,1fr)_auto_minmax(110px,auto)_auto] items-center gap-2 pr-2"
                  indentPerLevel={14}
                >
                  {span.hasChildren ? (
                    <SpanPrimitive.CollapseToggle
                      aria-label={`折叠 ${span.name}`}
                      className="border-0 bg-transparent p-0 text-neutral-500 group-data-[collapsed=true]:-rotate-90"
                      type="button"
                    >
                      ▾
                    </SpanPrimitive.CollapseToggle>
                  ) : (
                    <span className="w-4.5" />
                  )}
                  <SpanPrimitive.StatusIndicator className="h-2 w-2 rounded-full bg-neutral-500 data-[span-status=failed]:rounded-none data-[span-status=running]:animate-pulse" />
                  <SpanPrimitive.TypeBadge className="rounded border border-neutral-300 px-1 text-[10px] text-neutral-500" />
                  <SpanPrimitive.Name className="overflow-hidden text-ellipsis whitespace-nowrap text-xs" />
                  <span className="text-[10px] text-neutral-500">{span.status}</span>
                  <code
                    className="overflow-hidden text-ellipsis whitespace-nowrap text-[10px] text-neutral-500"
                    title={span.id}
                  >
                    {span.id}
                  </code>
                  <time className="text-[10px] text-neutral-500">{duration(span.latencyMs)}</time>
                </SpanPrimitive.Indent>
                <div className="relative m-3 rounded-full bg-neutral-200">
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
  if (milliseconds === null) return "进行中";
  return milliseconds < 1_000
    ? `${String(Math.round(milliseconds))} ms`
    : `${(milliseconds / 1_000).toFixed(1)} s`;
}
