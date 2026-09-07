import { HttpAgent } from "@ag-ui/client";
import { MessageSchema } from "@ag-ui/core";
import {
  ActionBarPrimitive,
  type AssistantRuntime,
  AssistantRuntimeProvider,
  AuiIf,
  ComposerPrimitive,
  ExportedMessageRepository,
  MessagePrimitive,
  type ReasoningMessagePartProps,
  type ThreadHistoryAdapter,
  ThreadPrimitive,
  type ToolCallMessagePartProps,
  useAuiState,
} from "@assistant-ui/react";
import {
  type AgUiInterrupt,
  fromAgUiMessages,
  useAgUiInterrupts,
  useAgUiRuntime,
  useAgUiSubmitInterruptResponses,
} from "@assistant-ui/react-ag-ui";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import remarkGfm from "remark-gfm";
import { z } from "zod";
import { HarnessPicker, type HarnessProps, RunControls } from "./agent-controls.js";
import { projectFetch as fetch, projectUrl } from "./api.js";
import { t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";
import { Subagents } from "./subagents.js";

function Reasoning({ text, status }: ReasoningMessagePartProps) {
  useTranslation();
  return (
    <details className="activity-card" open={status.type === "running"}>
      <summary>
        <Icon name="chevron" className="activity-chevron size-3.5" />
        <span className={status.type === "running" ? "animate-pulse" : ""}>
          {status.type === "running" ? t("正在思考…") : t("思考过程")}
        </span>
      </summary>
      <div className="whitespace-pre-wrap border-neutral-200 border-l pl-4 text-sm leading-7 text-neutral-500">
        {text}
      </div>
    </details>
  );
}

function ToolCard({ toolName, args, result, isError, status }: ToolCallMessagePartProps) {
  useTranslation();
  const label = isError
    ? t("失败")
    : status.type === "running"
      ? t("运行中")
      : status.type === "requires-action"
        ? t("等待确认")
        : status.type === "incomplete"
          ? status.reason === "cancelled"
            ? t("已停止")
            : t("未完成")
          : result === undefined
            ? t("未完成")
            : t("完成");
  return (
    <details className="activity-card">
      <summary>
        <Icon name="chevron" className="activity-chevron size-3.5" />
        <Icon name="code" className="size-3.5" />
        <span className="truncate font-mono text-xs">{toolName}</span>
        <span className="ml-auto shrink-0 text-xs">{label}</span>
      </summary>
      <pre className="max-h-64 overflow-auto rounded-lg bg-neutral-100 p-3 text-xs leading-6">
        {JSON.stringify({ args, result }, null, 2)}
      </pre>
      {toolName.includes("science_") && result !== undefined && (
        <button
          className="secondary-button mt-2"
          type="button"
          onClick={() => {
            window.dispatchEvent(
              new CustomEvent("swarmx:open-research", { detail: scienceTarget(result) }),
            );
            window.dispatchEvent(new Event("swarmx:science-changed"));
          }}
        >
          <Icon name="graph" />
          {t("在侧栏中查看")}
        </button>
      )}
    </details>
  );
}

function UserMessage() {
  useTranslation();
  return (
    <MessagePrimitive.Root className="mx-auto mb-8 flex w-full max-w-3xl justify-end">
      <div className="max-w-[85%] rounded-2xl bg-neutral-100 px-4 py-2.5 text-[15px] leading-7 break-words">
        <MessagePrimitive.Parts
          components={{ Text: ({ text }) => <p className="whitespace-pre-wrap">{text}</p> }}
        />
      </div>
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  useTranslation();
  const hasText = useAuiState((state) =>
    state.message.content.some((part) => part.type === "text" && part.text.length > 0),
  );
  return (
    <MessagePrimitive.Root
      className={`mx-auto w-full max-w-3xl text-[15px] leading-7 ${hasText ? "mb-8" : "mb-2"}`}
    >
      <MessagePrimitive.Parts
        components={{
          Text: () => (
            <MarkdownTextPrimitive className="message-markdown" remarkPlugins={[remarkGfm]} />
          ),
          Reasoning,
          tools: { Fallback: ToolCard },
          Empty: () => (
            <AuiIf condition={(state) => state.message.status?.type === "running"}>
              <span role="status" className="animate-pulse text-sm text-neutral-500">
                {t("正在处理…")}
              </span>
            </AuiIf>
          ),
        }}
      />
      {hasText && (
        <ActionBarPrimitive.Root hideWhenRunning className="mt-3 flex items-center">
          <ActionBarPrimitive.Copy
            aria-label={t("复制回复")}
            title={t("复制回复")}
            className="icon-button group gap-1.5 text-xs"
          >
            <span className="group-data-[copied]:hidden">
              <Icon name="copy" className="size-3.5" />
            </span>
            <span className="hidden items-center gap-1 group-data-[copied]:flex">
              <Icon name="check" className="size-3.5" />
              {t("已复制")}
            </span>
          </ActionBarPrimitive.Copy>
        </ActionBarPrimitive.Root>
      )}
    </MessagePrimitive.Root>
  );
}

function InteractionForms() {
  useTranslation();
  const interrupts = useAgUiInterrupts();
  const submit = useAgUiSubmitInterruptResponses();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string>();
  const respond = async (responses: Parameters<typeof submit>[0]) => {
    setPending(true);
    setError(undefined);
    try {
      await submit(responses);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setPending(false);
    }
  };
  return (
    <>
      {interrupts.map((interrupt) => (
        <form
          className="mx-auto mb-6 w-full max-w-3xl rounded-2xl border border-neutral-300 bg-neutral-50 p-5"
          key={interrupt.id}
          onSubmit={(event) => {
            event.preventDefault();
            if (!event.currentTarget.reportValidity()) return;
            void respond([
              {
                interruptId: interrupt.id,
                status: "resolved",
                payload: formPayload(new FormData(event.currentTarget), interrupt.responseSchema),
              },
            ]);
          }}
        >
          <div className="mb-1 text-xs text-neutral-500">{t("需要你的确认")}</div>
          <h3 className="mb-4 font-medium">{interrupt.message ?? t("Agent 需要你的输入")}</h3>
          <fieldset disabled={pending} className="grid gap-4">
            {schemaFields(interrupt).map(([name, schema]) => (
              <InteractionField
                key={name}
                name={name}
                schema={schema}
                required={
                  Array.isArray(interrupt.responseSchema?.required) &&
                  interrupt.responseSchema.required.includes(name)
                }
              />
            ))}
            <div className="mt-1 flex gap-2">
              <button className="primary-button" type="submit">
                {pending ? t("正在提交…") : t("继续")}
              </button>
              <button
                className="rounded-lg border border-neutral-300 bg-white px-3 py-2 text-sm hover:bg-neutral-100"
                onClick={() => void respond([{ interruptId: interrupt.id, status: "cancelled" }])}
                type="button"
              >
                {t("取消")}
              </button>
            </div>
          </fieldset>
        </form>
      ))}
      {error !== undefined && (
        <p role="alert" className="mx-auto mb-4 max-w-3xl break-words text-sm">
          {error}
        </p>
      )}
    </>
  );
}

function InteractionField({
  name,
  schema,
  required,
}: {
  name: string;
  schema: JsonObject;
  required: boolean;
}) {
  useTranslation();
  const options = choices(schema.type === "array" ? object(schema.items) : schema);
  const label = typeof schema.title === "string" ? schema.title : name;
  if (schema.type === "boolean") {
    return (
      <label className="flex items-center gap-2">
        <input name={name} type="checkbox" className="size-4 accent-neutral-900" />
        {label}
      </label>
    );
  }
  if (options.length > 0) {
    return (
      <label className="grid gap-1.5 text-sm">
        {label}
        <select
          className="interaction-input"
          multiple={schema.type === "array"}
          name={name}
          required={required}
          defaultValue={schema.type === "array" ? [] : ""}
        >
          {schema.type !== "array" && <option value="">{t("请选择")}</option>}
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
    );
  }
  const type =
    schema.type === "number" || schema.type === "integer"
      ? "number"
      : ["date", "email", "url"].includes(String(schema.format))
        ? String(schema.format)
        : "text";
  return (
    <label className="grid gap-1.5 text-sm">
      {label}
      <input
        className="interaction-input"
        name={name}
        type={type}
        required={required}
        step={schema.type === "integer" ? 1 : "any"}
        min={typeof schema.minimum === "number" ? schema.minimum : undefined}
        max={typeof schema.maximum === "number" ? schema.maximum : undefined}
        minLength={typeof schema.minLength === "number" ? schema.minLength : undefined}
        maxLength={typeof schema.maxLength === "number" ? schema.maxLength : undefined}
      />
    </label>
  );
}

interface ConversationProps extends HarnessProps {
  harnessDisabled: boolean;
  threadId: string;
  agentId: string;
  workspace: string;
  sidePanel?: ReactNode;
  panelOpen?: boolean;
}

export function ConversationSurface(props: ConversationProps) {
  useTranslation();
  const { threadId, agentId } = props;
  const [error, setError] = useState<string>();
  const [historyReady, setHistoryReady] = useState(false);
  const agent = useMemo(
    () =>
      new HttpAgent({
        url: projectUrl(`/api/ag-ui?agent=${encodeURIComponent(agentId)}`),
        threadId,
      }),
    [threadId, agentId],
  );
  const history = useMemo<ThreadHistoryAdapter>(
    () => ({
      async load() {
        const response = await fetch(
          "/api/v1/sessions/" +
            encodeURIComponent(threadId) +
            "?agent=" +
            encodeURIComponent(agentId),
        );
        if (!response.ok) throw new Error(await response.text());
        const repository = ExportedMessageRepository.fromArray(
          fromAgUiMessages(MessageSchema.array().parse(await response.json())),
        );
        setHistoryReady(true);
        return repository;
      },
      async append() {},
    }),
    [threadId, agentId],
  );
  const runtime = useAgUiRuntime({
    agent,
    adapters: { history },
    showThinking: true,
    onError: (cause) => setError(cause.message),
  });
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ConversationContent
        {...props}
        runtime={runtime}
        error={error}
        historyReady={historyReady}
        onSend={() => setError(undefined)}
      />
    </AssistantRuntimeProvider>
  );
}

function ConversationContent({
  runtime,
  agentId,
  threadId,
  harness,
  harnesses,
  onHarnessChange,
  harnessDisabled,
  sidePanel,
  panelOpen,
  error,
  historyReady,
  onSend,
}: ConversationProps & {
  runtime: AssistantRuntime;
  error: string | undefined;
  historyReady: boolean;
  onSend: () => void;
}) {
  useTranslation();
  const running = useAuiState((state) => state.thread.isRunning);
  const loading = useAuiState((state) => state.thread.isLoading);
  const empty = useAuiState((state) => state.thread.messages.length === 0);
  const interrupts = useAgUiInterrupts();
  const blocked = !historyReady || interrupts.length > 0;
  const welcome = empty && historyReady;
  useEffect(() => {
    const draft = (event: Event) => {
      const text = z.string().parse((event as CustomEvent).detail);
      const previous = runtime.thread.composer.getState().text;
      runtime.thread.composer.setText(previous ? `${previous}\n\n${text}` : text);
    };
    window.addEventListener("swarmx:compose", draft);
    return () => window.removeEventListener("swarmx:compose", draft);
  }, [runtime]);
  return (
    <ThreadPrimitive.Root className="relative flex min-h-0 flex-1 overflow-hidden">
      <div className={`flex min-h-0 min-w-0 flex-1 flex-col ${welcome ? "justify-center" : ""}`}>
        <div className={`relative min-h-0 ${welcome ? "" : "flex-1"}`}>
          <ThreadPrimitive.Viewport className="h-full overflow-y-auto px-5 pt-7 pb-4 sm:px-8">
            {loading && (
              <p role="status" className="mx-auto max-w-3xl py-6 text-sm text-neutral-500">
                {t("正在加载历史记录…")}
              </p>
            )}
            {welcome && (
              <div className="mx-auto mb-5 max-w-3xl text-center">
                <Icon name="swarm" className="mx-auto mb-5 size-9" />
                <h2 className="text-3xl font-semibold tracking-tight">{t("今天想探索什么？")}</h2>
                <p className="mt-3 text-sm text-neutral-500">
                  {t("把问题交给 SwarmX，一起推进下一步。")}
                </p>
              </div>
            )}
            <ThreadPrimitive.Messages components={{ UserMessage, AssistantMessage }} />
            <Subagents
              key={threadId}
              sessionId={threadId}
              running={running || interrupts.length > 0}
              messageComponents={{ UserMessage, AssistantMessage }}
            />
            <InteractionForms />
          </ThreadPrimitive.Viewport>
          {!welcome && (
            <ThreadPrimitive.ScrollToBottom
              aria-label={t("滚动到最新消息")}
              className="absolute bottom-3 left-1/2 grid size-8 -translate-x-1/2 place-items-center rounded-full border border-neutral-200 bg-white shadow-sm disabled:invisible"
            >
              <Icon name="arrowDown" />
            </ThreadPrimitive.ScrollToBottom>
          )}
        </div>
        <div className="mx-auto w-full max-w-[52rem] shrink-0 px-5 pb-4 sm:px-8">
          {error !== undefined && (
            <p
              role="alert"
              className="mb-3 rounded-lg border border-neutral-300 bg-neutral-50 px-4 py-3 text-sm break-words"
            >
              {error}
            </p>
          )}
          <ComposerPrimitive.Root
            className="rounded-2xl border border-neutral-300 bg-white p-3 shadow-sm focus-within:border-neutral-400 focus-within:shadow-md"
            onSubmit={onSend}
          >
            <ComposerPrimitive.Input
              aria-label={t("发送消息")}
              className="max-h-48 min-h-20 w-full resize-none bg-transparent px-1 py-1.5 text-[15px] leading-6 outline-none placeholder:text-neutral-400 disabled:opacity-50"
              placeholder={
                interrupts.length > 0 ? t("请先完成上方确认…") : t("描述你的任务，或提出一个问题…")
              }
              rows={2}
              disabled={blocked}
            />
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <HarnessPicker
                harness={harness}
                harnesses={harnesses}
                onHarnessChange={onHarnessChange}
                disabled={harnessDisabled || running || interrupts.length > 0}
              />
              <div className="ml-auto flex min-w-0 items-center gap-2">
                <RunControls
                  runtime={runtime}
                  agentId={agentId}
                  threadId={threadId}
                  disabled={running || blocked}
                />
                {running ? (
                  <ComposerPrimitive.Cancel
                    aria-label={t("停止生成")}
                    title={t("停止生成")}
                    className="grid size-8 place-items-center rounded-full bg-neutral-900 text-white hover:bg-neutral-700"
                  >
                    <Icon name="stop" className="size-3.5 fill-current" />
                  </ComposerPrimitive.Cancel>
                ) : (
                  <ComposerPrimitive.Send
                    aria-label={t("发送消息")}
                    title={t("发送消息")}
                    disabled={blocked}
                    className="grid size-8 place-items-center rounded-full bg-neutral-900 text-white hover:bg-neutral-700 disabled:bg-neutral-200 disabled:text-neutral-400"
                  >
                    <Icon name="arrowUp" className="size-5" />
                  </ComposerPrimitive.Send>
                )}
              </div>
            </div>
          </ComposerPrimitive.Root>
          <div className="mt-2.5 flex justify-between gap-3 px-1 text-[11px] text-neutral-400">
            <span role="status">
              {running
                ? t("正在执行 · 切换任务会停止")
                : interrupts.length > 0
                  ? t("等待你的确认")
                  : t("本地工作区")}
            </span>
            <span>{t("Enter 发送 · Shift + Enter 换行")}</span>
          </div>
          {welcome && (
            <div className="mt-7 grid gap-2 sm:grid-cols-3">
              <ThreadPrimitive.Suggestion
                prompt={t("请梳理当前项目的研究目标、已有进展和下一步。")}
                send={false}
                className="suggestion-card"
              >
                <Icon name="book" />
                <span>{t("梳理研究思路")}</span>
              </ThreadPrimitive.Suggestion>
              <ThreadPrimitive.Suggestion
                prompt={t("请查看当前工作区的数据与分析代码，说明可以如何开展分析。")}
                send={false}
                className="suggestion-card"
              >
                <Icon name="code" />
                <span>{t("探索数据与代码")}</span>
              </ThreadPrimitive.Suggestion>
              <ThreadPrimitive.Suggestion
                prompt={t("请整理当前实验记录，列出主要发现和待验证的问题。")}
                send={false}
                className="suggestion-card"
              >
                <Icon name="trace" />
                <span>{t("整理实验记录")}</span>
              </ThreadPrimitive.Suggestion>
            </div>
          )}
        </div>
      </div>
      <div id="research-side-view" className={panelOpen ? "research-side-view" : "hidden"}>
        {sidePanel}
      </div>
    </ThreadPrimitive.Root>
  );
}

type JsonObject = Record<string, unknown>;

export function scienceTarget(result: unknown): { artifactId?: string; projectId?: string } {
  const payload =
    typeof result === "string"
      ? z
          .string()
          .transform((value, ctx) => {
            try {
              return JSON.parse(value) as unknown;
            } catch {
              ctx.addIssue({ code: "custom", message: "Not a JSON tool result" });
              return z.NEVER;
            }
          })
          .safeParse(result)
      : { success: true as const, data: result };
  if (!payload.success) return {};
  const envelope = z.object({ data: z.unknown() }).safeParse(payload.data);
  const data = envelope.success ? envelope.data.data : payload.data;
  const artifact = z
    .object({ artifact: z.object({ id: z.string(), projectId: z.string() }).nullable() })
    .safeParse(data);
  if (artifact.success && artifact.data.artifact)
    return { artifactId: artifact.data.artifact.id, projectId: artifact.data.artifact.projectId };
  const entity = z
    .object({ id: z.string(), kind: z.string(), projectId: z.string().optional() })
    .safeParse(data);
  if (!entity.success) return {};
  return entity.data.kind === "project"
    ? { projectId: entity.data.id }
    : {
        artifactId: entity.data.id,
        ...(entity.data.projectId ? { projectId: entity.data.projectId } : {}),
      };
}

function schemaFields(interrupt: AgUiInterrupt): Array<[string, JsonObject]> {
  const properties = object(interrupt.responseSchema?.properties);
  return Object.entries(properties).map(([name, schema]) => [name, object(schema)]);
}

function formPayload(data: FormData, schema: JsonObject | undefined): JsonObject {
  const result: JsonObject = {};
  const required = new Set(Array.isArray(schema?.required) ? schema.required : []);
  for (const [name, field] of Object.entries(object(schema?.properties))) {
    const definition = object(field);
    if (definition.type === "boolean") {
      result[name] = data.has(name);
    } else if (definition.type === "array") {
      const values = data.getAll(name).map(String);
      if (values.length > 0 || required.has(name)) result[name] = values;
    } else {
      const value = data.get(name);
      if (value !== null && (String(value) !== "" || required.has(name))) {
        result[name] =
          definition.type === "number" || definition.type === "integer"
            ? Number(value)
            : String(value);
      }
    }
  }
  return result;
}

function choices(schema: JsonObject): Array<{ value: string; label: string }> {
  if (Array.isArray(schema.enum)) {
    return schema.enum.map((value) => ({ value: String(value), label: String(value) }));
  }
  if (!Array.isArray(schema.oneOf)) return [];
  return schema.oneOf.map((entry) => {
    const option = object(entry);
    return {
      value: String(option.const),
      label: typeof option.title === "string" ? option.title : String(option.const),
    };
  });
}

function object(value: unknown): JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
}
