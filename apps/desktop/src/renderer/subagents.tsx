import { EventType } from "@ag-ui/core";
import {
  AssistantRuntimeProvider,
  type ThreadMessageLike,
  ThreadPrimitive,
  useExternalStoreRuntime,
} from "@assistant-ui/react";
import { type ComponentProps, useEffect, useMemo, useState } from "react";
import { z } from "zod";
import { ExecutionPageSchema, type ExecutionRecord, type RunControl } from "../execution-record.js";
import { projectFetch as fetch } from "./api.js";
import { i18n, t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";

type MessageComponents = NonNullable<ComponentProps<typeof ThreadPrimitive.Messages>["components"]>;
const STATUS = {
  running: "运行中",
  waiting: "等待确认",
  stopping: "正在停止",
  stopped: "已停止",
  completed: "已完成",
  cancelled: "已结束 · 请求过停止",
  failed: "失败",
  unknown: "状态未知",
} as const;
const InteractionId = z.object({ id: z.string() });
const ControlFailure = z.object({ message: z.string() });
const SteeredInput = z.object({ text: z.string() });
const Delegation = z.object({ agentId: z.string() });
const ApiError = z.object({ error: z.string() });
const PAGE_SIZE = 200;

export function Subagents({
  sessionId,
  running,
  messageComponents,
}: {
  sessionId: string;
  running: boolean;
  messageComponents: MessageComponents;
}) {
  useTranslation();
  const [page, setPage] = useState<z.infer<typeof ExecutionPageSchema>>({
    events: [],
    nextAfter: 0,
    activeRunIds: [],
  });
  const [error, setError] = useState<string>();
  const [refreshCount, setRefreshCount] = useState(0);
  // biome-ignore lint/correctness/useExhaustiveDependencies: An explicit refresh restarts the journal read.
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let after = 0;
    let records: ExecutionRecord[] = [];
    setError(undefined);
    async function read() {
      try {
        let next: z.infer<typeof ExecutionPageSchema>;
        do {
          const query = new URLSearchParams({
            session: sessionId,
            descendants: "true",
            after: String(after),
            limit: String(PAGE_SIZE),
          });
          const response = await fetch(`/api/v1/logs?${query}`, { signal: controller.signal });
          if (!response.ok) throw new Error(ApiError.parse(await response.json()).error);
          next = ExecutionPageSchema.parse(await response.json());
          if (controller.signal.aborted) return;
          records = records.concat(next.events);
          after = next.nextAfter;
        } while (next.events.length === PAGE_SIZE);
        setPage({ ...next, events: records });
        const active = new Set(next.activeRunIds);
        if (running || records.some((record) => record.runId !== null && active.has(record.runId)))
          timer = setTimeout(() => void read(), 1000);
      } catch (cause) {
        if (!controller.signal.aborted)
          setError(cause instanceof Error ? cause.message : String(cause));
      }
    }
    void read();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [sessionId, running, refreshCount]);
  const runs = useMemo(
    () => subagentRuns(page.events, page.activeRunIds, sessionId),
    [page, sessionId],
  );
  if (runs.length === 0 && !error) return null;
  return (
    <section
      aria-label={t("子 Agent")}
      className="mx-auto mb-6 w-full max-w-3xl rounded-2xl border border-neutral-200 bg-white"
    >
      <header className="flex items-center gap-2 px-4 py-3 text-sm">
        <Icon name="swarm" className="size-4" />
        <h3 className="font-medium">{t("子 Agent")}</h3>
        <span className="text-xs text-neutral-500">
          {t("{{count}} / {{total}} 已完成", {
            count: runs.filter((run) => run.status === "completed").length,
            total: runs.length,
          })}
        </span>
        <button
          type="button"
          aria-label={t("刷新子 Agent")}
          title={t("刷新子 Agent")}
          className="icon-button ml-auto"
          onClick={() => setRefreshCount((value) => value + 1)}
        >
          <Icon name="refresh" className="size-3.5" />
        </button>
      </header>
      {error && (
        <p role="alert" className="px-4 pb-3 text-sm break-words">
          {t("子 Agent 日志加载失败：")}
          {error}
        </p>
      )}
      {runs.map((run) => (
        <SubagentCard
          key={run.id}
          run={run}
          messageComponents={messageComponents}
          unavailable={error !== undefined}
          onUpdate={() => setRefreshCount((value) => value + 1)}
        />
      ))}
    </section>
  );
}

function SubagentCard({
  run,
  messageComponents,
  unavailable,
  onUpdate,
}: {
  run: ReturnType<typeof subagentRuns>[number];
  messageComponents: MessageComponents;
  unavailable: boolean;
  onUpdate: () => void;
}) {
  useTranslation();
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string>();
  const [notice, setNotice] = useState<string>();
  async function control(command: RunControl) {
    setPending(true);
    setError(undefined);
    setNotice(undefined);
    try {
      const response = await fetch(`/api/v1/runs/${encodeURIComponent(run.id)}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(command),
      });
      if (!response.ok) throw new Error(ApiError.parse(await response.json()).error);
      if (command.action === "steer") setDraft("");
      setNotice(command.action === "steer" ? t("补充指令已发送") : t("已请求停止"));
      onUpdate();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setPending(false);
    }
  }
  return (
    <details
      className="subagent-card border-t border-neutral-200"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="flex items-start gap-3 px-4 py-3 hover:bg-neutral-50">
        <span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-full bg-neutral-100">
          <Icon name="swarm" className="size-3.5" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
            <span className="font-medium">{run.name}</span>
            <span className="text-xs text-neutral-500">{run.harness ?? t("Harness 未知")}</span>
          </div>
          <p className="mt-1 line-clamp-2 text-sm text-neutral-600">{run.task}</p>
          <p className="mt-1 text-xs text-neutral-500">
            {run.models.length > 0
              ? t("已报告模型：{{models}}", { models: run.models.join(", ") })
              : run.requestedModel
                ? t("请求模型：{{model}} · 实际模型未报告", { model: run.requestedModel })
                : t("模型未报告")}
          </p>
          {run.parent && (
            <p className="mt-1 text-xs text-neutral-500">
              {t("由 {{parent}} 委派", { parent: run.parent })}
            </p>
          )}
        </div>
        <span className="flex shrink-0 items-center gap-1.5 pt-1 text-xs" data-status={run.status}>
          <span
            aria-hidden
            className={`size-1.5 rounded-full ${run.status === "running" ? "animate-pulse bg-neutral-900" : run.status === "completed" ? "bg-neutral-600" : "border border-neutral-400"}`}
          />
          {t(STATUS[run.status])}
        </span>
        <Icon name="chevron" className="subagent-chevron mt-1 size-3.5 shrink-0 text-neutral-400" />
      </summary>
      {open && (
        <div className="space-y-4 border-t border-neutral-100 px-4 py-4 sm:pl-14">
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs text-neutral-500">
            <dt>{t("会话")}</dt>
            <dd className="break-all">{run.sessionId}</dd>
            <dt>{t("执行")}</dt>
            <dd className="break-all">{run.id}</dd>
            <dt>{t("开始")}</dt>
            <dd>
              <time dateTime={run.startedAt}>
                {new Date(run.startedAt).toLocaleString(i18n.language)}
              </time>
            </dd>
            {run.requestedModel && (
              <>
                <dt>{t("请求模型")}</dt>
                <dd>{run.requestedModel}</dd>
              </>
            )}
            {run.effort && (
              <>
                <dt>{t("请求 Thinking")}</dt>
                <dd>{run.effort}</dd>
              </>
            )}
          </dl>
          {run.failure && (
            <p role="alert" className="text-sm break-words">
              {run.failure}
            </p>
          )}
          {run.status === "unknown" && (
            <p className="text-sm text-neutral-500">
              {t("没有结束记录，当前 Host 也没有运行此任务，无法确认结果。")}
            </p>
          )}
          {run.status === "waiting" && (
            <p className="text-sm">{t("请在主对话中回答或取消此 Agent 的确认请求。")}</p>
          )}
          <div
            aria-label={t("{{name}} 本轮对话", { name: run.name })}
            className="max-h-96 overflow-y-auto rounded-xl border border-neutral-200 bg-neutral-50 px-3 pt-4"
          >
            <p className="mb-4 text-xs text-neutral-500">
              {t("本轮对话 · SwarmX 已记录的输入与输出")}
            </p>
            <RunConversation records={run.records} components={messageComponents} />
          </div>
          {run.active && (
            <form
              className="flex flex-wrap items-end gap-2"
              onSubmit={(event) => {
                event.preventDefault();
                void control({ action: "steer", text: draft.trim() });
              }}
            >
              <label className="min-w-0 flex-1 text-xs text-neutral-600">
                {t("补充指令")}
                <textarea
                  className="interaction-input mt-1 w-full resize-y"
                  aria-label={t("给 {{name}} 补充指令", { name: run.name })}
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder={t("调整这个子任务的方向…")}
                  rows={2}
                  maxLength={100000}
                  disabled={pending || unavailable || run.status !== "running"}
                />
              </label>
              <button
                className="primary-button"
                type="submit"
                disabled={pending || unavailable || run.status !== "running" || !draft.trim()}
              >
                {t("发送指令")}
              </button>
              <button
                className="icon-button border border-neutral-200"
                type="button"
                aria-label={t("停止 {{name}}", { name: run.name })}
                title={t("停止子 Agent")}
                disabled={pending || unavailable || run.status !== "running"}
                onClick={() => void control({ action: "cancel" })}
              >
                <Icon name="stop" className="size-3.5" />
              </button>
            </form>
          )}
          {notice && (
            <p role="status" className="text-xs text-neutral-500">
              {t(notice)}
            </p>
          )}
          {error && (
            <p role="alert" className="text-sm break-words">
              {error}
            </p>
          )}
          <details className="text-xs text-neutral-500">
            <summary className="py-1">
              {t("执行日志 · {{count}} 条", { count: run.records.length })}
            </summary>
            <div className="mt-2 max-h-80 overflow-auto">
              {run.records.map((record) => (
                <JournalRecord key={record.id} record={record} />
              ))}
            </div>
          </details>
        </div>
      )}
    </details>
  );
}

function JournalRecord({ record }: { record: ExecutionRecord }) {
  useTranslation();
  const [open, setOpen] = useState(false);
  return (
    <details
      className="border-t border-neutral-200 py-2"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="break-all font-mono">
        #{record.seq} {record.event.type}
        {record.event.type === EventType.CUSTOM ? ` · ${record.event.name}` : ""}
      </summary>
      {open && (
        <pre className="mt-2 overflow-x-auto rounded-lg bg-neutral-100 p-3 leading-5">
          {JSON.stringify(record, null, 2)}
        </pre>
      )}
    </details>
  );
}

function RunConversation({
  records,
  components,
}: {
  records: ExecutionRecord[];
  components: MessageComponents;
}) {
  useTranslation();
  const messages = useMemo(() => conversation(records), [records]);
  const runtime = useExternalStoreRuntime({
    messages,
    convertMessage: (message: ThreadMessageLike) => message,
    isDisabled: true,
    isRunning: false,
    onNew: async () => {
      throw new Error("Execution history is read-only.");
    },
  });
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root>
        <ThreadPrimitive.Messages components={components} />
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  );
}

function conversation(records: ExecutionRecord[]): ThreadMessageLike[] {
  const messages = new Map<
    string,
    {
      id: string;
      role: "user" | "assistant";
      createdAt: Date;
      content: [{ type: "text" | "reasoning"; text: string }];
    }
  >();
  function append(
    record: ExecutionRecord,
    id: string,
    text: string,
    role: "user" | "assistant",
    type: "text" | "reasoning" = "text",
  ) {
    const key = `${role}:${type}:${id}`;
    const existing = messages.get(key);
    if (existing) existing.content[0].text += text;
    else
      messages.set(key, {
        id: key,
        role,
        createdAt: new Date(record.observedAt),
        content: [{ type, text }],
      });
  }
  for (const record of records) {
    const event = record.event;
    if (event.type === EventType.RUN_STARTED) {
      for (const message of event.input?.messages ?? [])
        if (message.role === "user" && typeof message.content === "string")
          append(record, message.id, message.content, "user");
    } else if (event.type === EventType.TEXT_MESSAGE_CHUNK && event.messageId && event.delta) {
      append(record, event.messageId, event.delta, event.role === "user" ? "user" : "assistant");
    } else if (event.type === EventType.REASONING_MESSAGE_CHUNK && event.messageId && event.delta) {
      append(record, event.messageId, event.delta, "assistant", "reasoning");
    } else if (event.type === EventType.CUSTOM && event.name === "swarmx.input.steered") {
      append(
        record,
        record.id,
        t("补充指令请求：{{text}}", { text: SteeredInput.parse(event.value).text }),
        "user",
      );
    }
  }
  return [...messages.values()];
}

function subagentRuns(records: ExecutionRecord[], activeRunIds: string[], sessionId: string) {
  const byId = new Map(records.map((record) => [record.id, record]));
  const byRun = new Map<string, ExecutionRecord[]>();
  const targets = new Map<string | null, string>();
  for (const record of records) {
    if (record.runId !== null) {
      const group = byRun.get(record.runId);
      if (group) group.push(record);
      else byRun.set(record.runId, [record]);
    }
    if (record.event.type === EventType.TOOL_CALL_ARGS) {
      const cause = byId.get(record.causedBy ?? "");
      if (cause?.event.type === EventType.TOOL_CALL_START && cause.event.toolCallName === "swarm") {
        const call = JSON.parse(record.event.delta) as unknown;
        const parsed = Delegation.safeParse(call);
        if (parsed.success) targets.set(record.causedBy, parsed.data.agentId);
      }
    }
  }
  const starts = new Map(
    records
      .filter((record) => record.event.type === EventType.RUN_STARTED)
      .map((record) => [record.runId, record]),
  );
  const activeIds = new Set(activeRunIds);
  return [...starts.values()].flatMap((start) => {
    const cause = byId.get(start.causedBy ?? "");
    if (
      start.sessionId === sessionId ||
      start.sessionId === null ||
      start.runId === null ||
      cause?.event.type !== EventType.TOOL_CALL_START ||
      cause.event.toolCallName !== "swarm"
    )
      return [];
    const events = byRun.get(start.runId) ?? [];
    const terminal = events.findLast(
      ({ event }) => event.type === EventType.RUN_FINISHED || event.type === EventType.RUN_ERROR,
    )?.event;
    const active = !terminal && activeIds.has(start.runId);
    const requests = new Set<string>();
    let stopping: string | undefined;
    let controlFailure: string | undefined;
    for (const { event, id, causedBy } of events) {
      if (event.type !== EventType.CUSTOM) continue;
      if (event.name === "swarmx.interaction.requested")
        requests.add(InteractionId.parse(event.value).id);
      if (event.name === "swarmx.interaction.answered")
        requests.delete(InteractionId.parse(event.value).id);
      if (event.name === "swarmx.run.interrupt_requested") stopping = id;
      if (event.name === "swarmx.run.interrupt_requested" || event.name === "swarmx.input.steered")
        controlFailure = undefined;
      if (event.name === "swarmx.control.failed") {
        if (causedBy === stopping) stopping = undefined;
        controlFailure = ControlFailure.parse(event.value).message;
      }
    }
    const status: keyof typeof STATUS =
      terminal?.type === EventType.RUN_ERROR
        ? "failed"
        : terminal?.type === EventType.RUN_FINISHED
          ? terminal.result?.stopReason === "cancelled" ||
            (terminal.result?.stopReason === undefined &&
              terminal.result?.interruptionRequested === true)
            ? "cancelled"
            : terminal.result?.stopReason && terminal.result.stopReason !== "end_turn"
              ? "stopped"
              : "completed"
          : !active
            ? "unknown"
            : requests.size > 0
              ? "waiting"
              : stopping
                ? "stopping"
                : "running";
    const parent = starts.get(cause.runId);
    const stringAttribute = (key: string) => {
      const value = start.attributes[key];
      return typeof value === "string" ? value : undefined;
    };
    return [
      {
        id: start.runId,
        sessionId: start.sessionId,
        records: events,
        name: targets.get(start.causedBy) ?? stringAttribute("gen_ai.agent.name") ?? "Agent",
        parent:
          parent?.sessionId !== sessionId && parent
            ? (targets.get(parent.causedBy) ?? parent.sessionId)
            : null,
        task:
          start.event.type === EventType.RUN_STARTED
            ? start.event.input?.messages
                .filter((message) => message.role === "user")
                .map((message) => message.content)
                .join("\n")
            : "",
        startedAt: start.observedAt,
        harness: stringAttribute("swarmx.harness.name"),
        requestedModel: stringAttribute("gen_ai.request.model"),
        effort: stringAttribute("gen_ai.request.reasoning.level"),
        models: [
          ...new Set(
            events
              .map((record) => record.attributes["gen_ai.response.model"])
              .filter((value): value is string => typeof value === "string"),
          ),
        ],
        active,
        status,
        failure: terminal?.type === EventType.RUN_ERROR ? terminal.message : controlFailure,
      },
    ];
  });
}
