import { HttpAgent } from "@ag-ui/client";
import { MessageSchema } from "@ag-ui/core";
import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  ExportedMessageRepository,
  MessagePrimitive,
  type ReasoningMessagePartProps,
  type ThreadHistoryAdapter,
  ThreadPrimitive,
  type ToolCallMessagePartProps,
} from "@assistant-ui/react";
import {
  type AgUiInterrupt,
  fromAgUiMessages,
  useAgUiInterrupts,
  useAgUiRuntime,
  useAgUiSubmitInterruptResponses,
} from "@assistant-ui/react-ag-ui";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import { useMemo, useState } from "react";
import { TracePanel } from "./trace.js";

function Reasoning({ text, status }: ReasoningMessagePartProps) {
  return (
    <details
      className="my-3 rounded-lg border border-neutral-200 bg-white p-3"
      open={status.type === "running"}
    >
      <summary className="cursor-pointer font-semibold text-neutral-600">
        {status.type === "running" ? "正在思考…" : "推理"}
      </summary>
      <div>{text}</div>
    </details>
  );
}

function ToolCard({ toolName, args, result, isError, status }: ToolCallMessagePartProps) {
  return (
    <details className="my-3 rounded-lg border border-neutral-200 bg-white p-3">
      <summary className="cursor-pointer font-semibold text-neutral-600">
        {toolName} · {status.type === "running" ? "运行中" : isError ? "失败" : "完成"}
      </summary>
      <pre className="overflow-auto whitespace-pre-wrap text-xs">
        {JSON.stringify({ args, result }, null, 2)}
      </pre>
    </details>
  );
}

function UserMessage() {
  return (
    <MessagePrimitive.Root className="mx-auto mb-7 flex max-w-3xl justify-end leading-7">
      <MessagePrimitive.Parts
        components={{
          Text: ({ text }) => (
            <p className="m-0 max-w-[72%] rounded-2xl rounded-br-sm bg-neutral-200 px-4 py-3">
              {text}
            </p>
          ),
        }}
      />
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="mx-auto mb-7 max-w-3xl leading-7 [&_a]:underline [&_p]:my-3">
      <MessagePrimitive.Parts
        components={{
          Text: () => <MarkdownTextPrimitive />,
          Reasoning,
          tools: { Fallback: ToolCard },
          Empty: () => <span className="animate-pulse text-neutral-500">● ● ●</span>,
        }}
      />
    </MessagePrimitive.Root>
  );
}

function InteractionForms() {
  const interrupts = useAgUiInterrupts();
  const submit = useAgUiSubmitInterruptResponses();
  return interrupts.map((interrupt) => (
    <form
      className="mx-auto mb-5 grid max-w-3xl gap-3 rounded-xl border border-neutral-400 bg-neutral-50 p-4"
      key={interrupt.id}
      onSubmit={(event) => {
        event.preventDefault();
        void submit([
          {
            interruptId: interrupt.id,
            status: "resolved",
            payload: formPayload(new FormData(event.currentTarget), interrupt.responseSchema),
          },
        ]);
      }}
    >
      <strong>{interrupt.message ?? "Agent 需要你的输入"}</strong>
      {schemaFields(interrupt).map(([name, schema]) => (
        <InteractionField key={name} name={name} schema={schema} />
      ))}
      <div className="flex gap-2">
        <button className="rounded-lg bg-black px-3 py-2 text-white" type="submit">
          继续
        </button>
        <button
          className="rounded-lg bg-neutral-200 px-3 py-2 text-neutral-800"
          onClick={() => void submit([{ interruptId: interrupt.id, status: "cancelled" }])}
          type="button"
        >
          取消
        </button>
      </div>
    </form>
  ));
}

function InteractionField({ name, schema }: { name: string; schema: JsonObject }) {
  const options = choices(schema.type === "array" ? object(schema.items) : schema);
  const label = typeof schema.title === "string" ? schema.title : name;
  if (schema.type === "boolean") {
    return (
      <label className="grid gap-1">
        <input name={name} type="checkbox" /> {label}
      </label>
    );
  }
  if (options.length > 0) {
    return (
      <label className="grid gap-1">
        {label}
        <select
          className="min-h-9 rounded-lg border border-neutral-400 bg-white px-2 py-1.5"
          multiple={schema.type === "array"}
          name={name}
        >
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
    <label className="grid gap-1">
      {label}
      <input
        className="min-h-9 rounded-lg border border-neutral-400 bg-white px-2 py-1.5"
        name={name}
        type={type}
      />
    </label>
  );
}

export function ConversationSurface({
  threadId,
  title,
  agentId,
}: {
  threadId: string;
  title: string;
  agentId: string;
}) {
  const [error, setError] = useState<string>();
  const agent = useMemo(
    () => new HttpAgent({ url: `/api/ag-ui?agent=${agentId}`, threadId }),
    [threadId, agentId],
  );
  const history = useMemo<ThreadHistoryAdapter>(
    () => ({
      async load() {
        const response = await fetch(
          `/api/v1/sessions/${encodeURIComponent(threadId)}?agent=${agentId}`,
        );
        if (!response.ok) throw new Error(await response.text());
        return ExportedMessageRepository.fromArray(
          fromAgUiMessages(MessageSchema.array().parse(await response.json())),
        );
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
  const components = useMemo(() => ({ UserMessage, AssistantMessage }), []);
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root className="grid h-full min-w-0 grid-rows-[auto_minmax(0,1fr)_auto_auto] bg-white text-neutral-950">
        <header className="border-neutral-200 border-b bg-white/90 px-6 py-4 md:px-16 xl:px-24">
          <div>
            <small className="font-bold text-neutral-500 tracking-widest">SWARM AGENT</small>
            <h1 className="mt-1 text-xl font-semibold">{title}</h1>
          </div>
        </header>
        <ThreadPrimitive.Viewport className="overflow-y-auto px-6 pt-8 pb-30 md:px-16 xl:px-40">
          <ThreadPrimitive.Empty>
            <div className="mx-auto mt-[18vh] max-w-xl text-center">
              <span className="font-extrabold tracking-[0.12em]">SWARMX</span>
              <h2 className="mt-3 mb-2 font-serif text-4xl font-medium">从一个问题开始。</h2>
              <p className="text-neutral-500">递归 Swarm · 原生 Agent · 可追溯研究</p>
            </div>
          </ThreadPrimitive.Empty>
          <TracePanel />
          <ThreadPrimitive.Messages components={components} />
          <InteractionForms />
          <ThreadPrimitive.ScrollToBottom className="fixed right-7 bottom-24 h-9 w-9 rounded-lg border border-neutral-300 bg-white font-bold">
            ↓
          </ThreadPrimitive.ScrollToBottom>
        </ThreadPrimitive.Viewport>
        {error === undefined ? null : (
          <p className="mx-auto mb-2 w-[min(48rem,calc(100%-3rem))] font-medium text-black">
            错误：{error}
          </p>
        )}
        <ComposerPrimitive.Root className="mx-auto mb-6 flex w-[min(48rem,calc(100%-3rem))] gap-2 rounded-xl border border-neutral-300 bg-white p-2 shadow-lg shadow-neutral-200">
          <ComposerPrimitive.Input
            aria-label="发送消息"
            className="min-h-10 flex-1 resize-none border-0 p-2.5 outline-none"
            placeholder="给 SwarmX 一条指令…"
            rows={1}
          />
          <ComposerPrimitive.Send className="rounded-lg bg-black px-4 font-bold text-white disabled:cursor-not-allowed disabled:opacity-50">
            发送
          </ComposerPrimitive.Send>
          <ComposerPrimitive.Cancel className="rounded-lg bg-neutral-200 px-4 font-bold text-black disabled:cursor-not-allowed disabled:opacity-50">
            停止
          </ComposerPrimitive.Cancel>
        </ComposerPrimitive.Root>
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  );
}

type JsonObject = Record<string, unknown>;

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
