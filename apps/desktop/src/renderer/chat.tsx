import { FilterToolCallsMiddleware } from "@ag-ui/client";
import {
  ActionBarPrimitive,
  type AssistantRuntime,
  AssistantRuntimeProvider,
  ComposerPrimitive,
  ExportedMessageRepository,
  groupPartByType,
  MessagePrimitive,
  type ThreadHistoryAdapter,
  ThreadPrimitive,
  type ToolCallMessagePartProps,
  useAuiState,
} from "@assistant-ui/react";
import {
  fromAgUiMessages,
  useAgUiInterrupts,
  useAgUiRuntime,
  useAgUiSubmitInterruptResponses,
} from "@assistant-ui/react-ag-ui";
import { type ReactNode, useContext, useEffect, useMemo, useState } from "react";
import { z } from "zod";
import { HistoryMessagesSchema, type MessageActivity } from "../message-activity.js";
import { HarnessPicker, type HarnessProps, RunControls } from "./agent-controls.js";
import { IpcAgent } from "./agui.js";
import { bridge } from "./bridge.js";
import { CommentaryMessages, useMessageActivity } from "./commentary.js";
import { MarkdownText } from "./components/assistant-ui/elements/markdown-text.js";
import { TerminalBlock } from "./components/assistant-ui/elements/terminal-block.js";
import {
  ToolFallbackArgs,
  ToolFallbackContent,
  ToolFallbackError,
  ToolFallbackResult,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./components/assistant-ui/elements/tool-fallback.js";
import { Button } from "./components/ui/radix/button.js";
import { t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";
import { NativeInteractionForm } from "./interaction-form.js";
import { readConceptResult, SavedConcept } from "./saved-concept.js";
import type { SourceReference } from "./source-inspection.js";
import { Subagents } from "./subagents.js";
import { MessageToolGroup, readShell, ToolActivityContext } from "./tool-ui.js";

const CONTEXT_COMPACTION_TOOL = "contextCompaction";

function ToolCard({
  toolCallId,
  toolName,
  args,
  argsText,
  result,
  isError,
  status,
}: ToolCallMessagePartProps) {
  useTranslation();
  const native = useContext(ToolActivityContext)[toolCallId];
  const shell = readShell({ args, result });
  const cancelled = status.type === "incomplete" && status.reason === "cancelled";
  const failed =
    !cancelled &&
    (isError || native?.status === "failed" || (shell?.exitCode != null && shell.exitCode !== 0));
  const label = cancelled
    ? t("已停止")
    : failed
      ? t("失败")
      : status.type === "running"
        ? t("运行中")
        : status.type === "requires-action"
          ? t("等待确认")
          : status.type === "incomplete"
            ? t("未完成")
            : result === undefined
              ? t("未完成")
              : t("完成");
  const succeeded = !failed && status.type === "complete" && result !== undefined;
  if (shell)
    return (
      <TerminalBlock
        command={shell.command ? `$ ${shell.command}` : toolName}
        lines={[shell.output ?? ""]}
        visibleCount={1}
        done={status.type !== "running"}
        className="max-w-none [&>div:last-child]:max-h-60 [&>div:last-child]:overflow-auto [&>div:last-child]:whitespace-pre"
        status={
          <span
            role="status"
            className={`flex shrink-0 items-center gap-1 ${failed ? "text-red-600" : "text-muted-foreground"}`}
          >
            {shell.exitCode != null && <span>exit {shell.exitCode}</span>}
            <Icon name={succeeded ? "check" : failed ? "errorCircle" : "code"} />
            {succeeded ? t("命令成功") : label}
          </span>
        }
      />
    );
  const operation = z
    .object({ action: z.string() })
    .or(
      z
        .object({ arguments: z.object({ action: z.string() }) })
        .transform((value) => value.arguments),
    )
    .safeParse(args);
  const memory =
    toolName.includes("memory") &&
    operation.success &&
    ["search_memory", "read_memory", "load_memory"].includes(operation.data.action);
  return (
    <ToolFallbackRoot defaultOpen={status.type === "requires-action"}>
      <ToolFallbackTrigger
        toolName={memory ? t("记忆搜索与读取") : toolName}
        status={failed ? { type: "incomplete", reason: "error" } : status}
      >
        <span>{memory ? t("记忆搜索与读取") : toolName}</span> <span>{label}</span>
      </ToolFallbackTrigger>
      <ToolFallbackContent>
        <ToolFallbackError status={status} />
        <p className="text-muted-foreground text-xs font-medium">{t("工具参数")}</p>
        <ToolFallbackArgs argsText={argsText} />
        <ToolFallbackResult result={result} />
        {toolName.includes("science_") && result !== undefined && (
          <Button
            variant="outline"
            size="sm"
            className="mt-2"
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
          </Button>
        )}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
}

function UserMessage() {
  return (
    <MessagePrimitive.Root data-role="user" className="chat-message flex flex-col items-end">
      <div className="relative max-w-[80%]">
        <div className="user-bubble rounded-thread bg-muted px-4 py-2 text-[15px] wrap-break-word empty:hidden">
          <MessagePrimitive.Parts
            components={{ Text: ({ text }) => <p className="whitespace-pre-wrap">{text}</p> }}
          />
        </div>
      </div>
    </MessagePrimitive.Root>
  );
}

function AssistantMessage({ view = "all" }: { view?: "all" | "work" | "answer" | "commentary" }) {
  useTranslation();
  const hasText = useAuiState(
    (state) =>
      view !== "work" &&
      state.message.content.some((part) => part.type === "text" && part.text.length > 0),
  );
  const hasToolsOrStatus = useAuiState(
    (state) =>
      state.message.status?.type === "running" ||
      (view !== "answer" && state.message.content.some((part) => part.type === "tool-call")),
  );
  const conceptResult = useAuiState((state) => {
    const index = state.thread.messages.findIndex((message) => message.id === state.message.id);
    for (let i = index + 1; i < state.thread.messages.length; i++) {
      const later = state.thread.messages[i];
      if (!later || later.role === "user") break;
      if (
        later.role === "assistant" &&
        later.content.some((part) => part.type === "text" && part.text.length > 0)
      )
        return undefined;
    }
    for (let i = index; i >= 0; i--) {
      const message = state.thread.messages[i];
      if (!message || message.role === "user") break;
      for (let j = message.content.length - 1; j >= 0; j--) {
        const part = message.content[j];
        if (
          part?.type === "tool-call" &&
          part.toolName.includes("memory") &&
          !part.isError &&
          readConceptResult(part.result)
        )
          return part.result;
      }
    }
    return undefined;
  });
  const concept = readConceptResult(conceptResult);
  if (!hasText && !hasToolsOrStatus) return null;
  return (
    <MessagePrimitive.Root data-role="assistant" className="chat-message assistant-message">
      <div className="text-[15px] leading-relaxed wrap-break-word">
        <MessagePrimitive.GroupedParts
          groupBy={groupPartByType({
            "tool-call": ["group-tool"],
            data: ["group-tool"],
          })}
        >
          {({ part, children }) => {
            switch (part.type) {
              case "group-tool":
                return view === "answer" ? null : (
                  <MessageToolGroup indices={part.indices} running={part.status.type === "running"}>
                    {children}
                  </MessageToolGroup>
                );
              case "text":
                return view === "work" ? null : (
                  <MarkdownText
                    components={view === "commentary" ? { CodeHeader: () => null } : {}}
                  />
                );
              case "tool-call":
                return view === "answer" ? null : (part.toolUI ?? <ToolCard {...part} />);
              case "indicator":
                return view === "work" ? null : (
                  <span role="status" className="animate-pulse text-sm text-neutral-500">
                    {t("正在处理…")}
                  </span>
                );
              default:
                return null;
            }
          }}
        </MessagePrimitive.GroupedParts>
        {hasText && concept && <SavedConcept concept={concept} />}
      </div>
      {hasText && view !== "commentary" && (
        <div className="answer-actions mt-2 h-6">
          <ActionBarPrimitive.Root
            hideWhenRunning
            autohide="not-last"
            className="flex h-full items-center gap-1.5"
          >
            <ActionBarPrimitive.Copy
              aria-label={t("复制回复")}
              title={t("复制回复")}
              className="group p-1 text-muted-foreground/70 transition-colors hover:text-foreground disabled:pointer-events-none disabled:opacity-40"
            >
              <span className="group-data-[copied]:hidden">
                <Icon name="copy" />
              </span>
              <span className="hidden items-center gap-1 group-data-[copied]:flex">
                <Icon name="check" />
                <span className="sr-only">{t("已复制")}</span>
              </span>
            </ActionBarPrimitive.Copy>
          </ActionBarPrimitive.Root>
        </div>
      )}
    </MessagePrimitive.Root>
  );
}

function WorkMessage() {
  return <AssistantMessage view="work" />;
}

function FinalMessage() {
  return <AssistantMessage view="answer" />;
}

function CommentaryMessage() {
  return <AssistantMessage view="commentary" />;
}

function InteractionForms() {
  const interrupts = useAgUiInterrupts();
  const submit = useAgUiSubmitInterruptResponses();
  return interrupts.map((interrupt) => (
    <NativeInteractionForm
      key={interrupt.id}
      id={interrupt.id}
      title={interrupt.message ?? t("Agent 需要你的输入")}
      schema={interrupt.responseSchema ?? {}}
      onRespond={async (answer) => {
        await submit([
          {
            interruptId: interrupt.id,
            status: answer === undefined ? "cancelled" : "resolved",
            ...(answer === undefined ? {} : { payload: answer }),
          },
        ]);
      }}
    />
  ));
}

interface ConversationProps extends HarnessProps {
  harnessDisabled: boolean;
  threadId: string;
  agentId: string;
  sidePanel?: ReactNode;
  panelOpen?: boolean;
  source?: SourceReference | undefined;
}

export function ConversationSurface(props: ConversationProps) {
  useTranslation();
  const { threadId, agentId } = props;
  const [error, setError] = useState<string>();
  const [historyReady, setHistoryReady] = useState(false);
  const [retryingHistory, setRetryingHistory] = useState(false);
  const agent = useMemo(
    () =>
      new IpcAgent(agentId, threadId).use(
        new FilterToolCallsMiddleware({ disallowedToolCalls: [CONTEXT_COMPACTION_TOOL] }),
      ),
    [threadId, agentId],
  );
  useEffect(() => () => agent.abortRun(), [agent]);
  const { activity, tools, restore } = useMessageActivity(agent);
  const history = useMemo(
    () =>
      ({
        async load() {
          const messages = HistoryMessagesSchema.parse(
            await bridge().sessions.history({ agent: agentId, sessionId: threadId }),
          );
          restore(messages);
          const repository = ExportedMessageRepository.fromArray(
            fromAgUiMessages(messages, { showThinking: false })
              .map((message) => ({
                ...message,
                content:
                  typeof message.content === "string"
                    ? message.content
                    : message.content.filter(
                        (part) =>
                          part.type !== "tool-call" || part.toolName !== CONTEXT_COMPACTION_TOOL,
                      ),
              }))
              .filter((message) => message.content.length > 0),
          );
          setHistoryReady(true);
          return repository;
        },
        async append() {},
      }) satisfies ThreadHistoryAdapter,
    [threadId, agentId, restore],
  );
  const runtime = useAgUiRuntime({
    agent,
    adapters: { history },
    showThinking: false,
    onError: (cause) => setError(cause.message),
  });
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ToolActivityContext.Provider value={tools}>
        <ConversationContent
          {...props}
          runtime={runtime}
          activity={activity}
          error={error}
          historyReady={historyReady}
          retryingHistory={retryingHistory}
          onRetryHistory={async () => {
            setRetryingHistory(true);
            try {
              runtime.thread.import(await history.load());
              setError(undefined);
            } catch (cause) {
              setError(cause instanceof Error ? cause.message : String(cause));
            } finally {
              setRetryingHistory(false);
            }
          }}
          onSend={() => setError(undefined)}
        />
      </ToolActivityContext.Provider>
    </AssistantRuntimeProvider>
  );
}

function ConversationContent({
  runtime,
  activity,
  agentId,
  threadId,
  harness,
  harnesses,
  onHarnessChange,
  harnessDisabled,
  sidePanel,
  panelOpen,
  source,
  error,
  historyReady,
  retryingHistory,
  onRetryHistory,
  onSend,
}: ConversationProps & {
  runtime: AssistantRuntime;
  activity: Record<string, MessageActivity>;
  error: string | undefined;
  historyReady: boolean;
  retryingHistory: boolean;
  onRetryHistory: () => Promise<void>;
  onSend: () => void;
}) {
  useTranslation();
  const running = useAuiState((state) => state.thread.isRunning);
  const loading = useAuiState((state) => state.thread.isLoading);
  const empty = useAuiState((state) => state.thread.messages.length === 0);
  const interrupts = useAgUiInterrupts();
  const standaloneUsed = threadId.startsWith("dsh:") && !empty;
  const blocked = !historyReady || interrupts.length > 0 || standaloneUsed;
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
    <ThreadPrimitive.Root
      className={`conversation-layout relative flex min-h-0 flex-1 overflow-hidden ${panelOpen ? "has-side-panel" : ""}`}
    >
      <div className="conversation-column flex min-h-0 min-w-0 flex-1 flex-col">
        <ThreadPrimitive.Viewport
          turnAnchor="top"
          className={`conversation-viewport relative flex flex-1 flex-col overflow-y-auto ${welcome ? "justify-center" : ""}`}
        >
          {(loading || retryingHistory) && (
            <p role="status" className="mx-auto max-w-3xl py-6 text-sm text-neutral-500">
              {t("正在加载历史记录…")}
            </p>
          )}
          {welcome && (
            <div className="mx-auto mb-8 flex w-full max-w-(--thread-max-width) flex-col items-center text-center">
              <h2 className="text-2xl font-medium tracking-tight">{t("今天想探索什么？")}</h2>
            </div>
          )}
          <div className="mb-12 empty:hidden">
            <CommentaryMessages
              activity={activity}
              components={{
                UserMessage,
                AssistantMessage,
                WorkMessage,
                FinalMessage,
                CommentaryMessage,
              }}
            />
            <Subagents
              key={threadId}
              sessionId={threadId}
              running={running || interrupts.length > 0}
              messageComponents={{ UserMessage, AssistantMessage }}
            />
            <InteractionForms />
          </div>
          <ThreadPrimitive.ViewportFooter
            className={`conversation-composer ${welcome ? "" : "sticky bottom-0 mt-auto"}`}
          >
            {!welcome && (
              <ThreadPrimitive.ScrollToBottom
                aria-label={t("滚动到最新消息")}
                className="absolute -top-11 z-10 grid size-8 place-items-center self-center rounded-control border border-foreground/10 bg-background transition-colors hover:border-foreground/25 disabled:invisible"
              >
                <Icon name="arrowDown" />
              </ThreadPrimitive.ScrollToBottom>
            )}
            {error !== undefined && (
              <div
                role="alert"
                className="mb-3 rounded-lg border border-neutral-300 bg-neutral-50 px-4 py-3 text-sm break-words"
              >
                {!historyReady && <p className="mb-1 font-medium">{t("无法加载对话历史")}</p>}
                <p>{error}</p>
                {historyReady && error.includes("already has an active writer") && (
                  <p className="mt-1 text-neutral-600">
                    {t(
                      "该对话的写入权限正被另一个 Codex 实例持有，暂时无法在这里发送消息；历史记录仍可查看。",
                    )}
                  </p>
                )}
                {!historyReady && (
                  <>
                    <p className="mt-1 text-neutral-600">{t("重新加载后即可继续此对话。")}</p>
                    <Button
                      variant="outline"
                      size="sm"
                      className="mt-3"
                      disabled={retryingHistory}
                      onClick={onRetryHistory}
                      type="button"
                    >
                      {retryingHistory ? t("正在加载历史记录…") : t("重新加载历史")}
                    </Button>
                  </>
                )}
              </div>
            )}
            <ComposerPrimitive.Root
              className="chat-composer relative w-full rounded-thread border border-foreground/10 bg-muted/30 transition-colors focus-within:border-foreground/25"
              onSubmit={onSend}
            >
              <ComposerPrimitive.Input
                aria-label={t("发送消息")}
                className="composer-input field-sizing-content max-h-48 w-full resize-none bg-transparent px-4 pt-3 pb-2 text-base leading-6 placeholder:text-muted-foreground focus:outline-none"
                placeholder={
                  standaloneUsed
                    ? t("DSH 每个任务独立执行；请新建任务继续")
                    : interrupts.length > 0
                      ? t("请先完成上方确认…")
                      : source
                        ? t("询问这些记录…")
                        : t("描述你的任务，或提出一个问题…")
                }
                rows={1}
                disabled={blocked}
              />
              <div className="composer-toolbar flex flex-wrap items-center justify-between gap-2 px-2 pb-2">
                {source && (
                  <button
                    type="button"
                    className="composer-context"
                    onClick={() =>
                      window.dispatchEvent(
                        new CustomEvent("swarmx:compose", {
                          detail: `${source.title ?? t("来源引用")}: ${source.resource}`,
                        }),
                      )
                    }
                  >
                    <Icon name="attachment" className="size-6" />
                    <span>{t("添加来源上下文")}</span>
                  </button>
                )}
                <div className={`composer-run-controls ${source ? "ml-auto" : ""}`}>
                  <HarnessPicker
                    harness={harness}
                    harnesses={harnesses}
                    onHarnessChange={onHarnessChange}
                    disabled={harnessDisabled || running || interrupts.length > 0}
                  />
                  <RunControls
                    runtime={runtime}
                    agentId={agentId}
                    threadId={threadId}
                    disabled={running || blocked}
                  />
                </div>
                <div className="ml-auto flex min-w-0 items-center gap-2">
                  {running ? (
                    <ComposerPrimitive.Cancel
                      aria-label={t("停止生成")}
                      title={t("停止生成")}
                      className="grid size-7 place-items-center rounded-control bg-primary text-primary-foreground"
                    >
                      <Icon name="stop" className="size-3 fill-current" />
                    </ComposerPrimitive.Cancel>
                  ) : (
                    <ComposerPrimitive.Send
                      aria-label={t("发送消息")}
                      title={t("发送消息")}
                      disabled={blocked}
                      className="grid size-7 place-items-center rounded-control bg-primary text-primary-foreground transition-opacity disabled:opacity-40"
                    >
                      <Icon name="arrowUp" />
                    </ComposerPrimitive.Send>
                  )}
                </div>
              </div>
            </ComposerPrimitive.Root>
            <div className="composer-hint">
              <span role="status">
                {running
                  ? t("正在执行 · 切换任务会停止")
                  : interrupts.length > 0
                    ? t("等待你的确认")
                    : t("本地执行")}
              </span>
              <span>{t("Enter 发送 · Shift + Enter 换行")}</span>
            </div>
            {welcome && (
              <div className="mx-auto mt-3 flex max-w-[34rem] flex-wrap items-center justify-center gap-2">
                <ThreadPrimitive.Suggestion
                  prompt={t("请梳理当前研究目标、已有进展和下一步。")}
                  send={false}
                  className="suggestion-card"
                >
                  <span>{t("梳理研究思路")}</span>
                </ThreadPrimitive.Suggestion>
                <ThreadPrimitive.Suggestion
                  prompt={t("请查看当前目录的数据与分析代码，说明可以如何开展分析。")}
                  send={false}
                  className="suggestion-card"
                >
                  <span>{t("探索数据与代码")}</span>
                </ThreadPrimitive.Suggestion>
                <ThreadPrimitive.Suggestion
                  prompt={t("请整理当前实验记录，列出主要发现和待验证的问题。")}
                  send={false}
                  className="suggestion-card"
                >
                  <span>{t("整理实验记录")}</span>
                </ThreadPrimitive.Suggestion>
              </div>
            )}
          </ThreadPrimitive.ViewportFooter>
        </ThreadPrimitive.Viewport>
      </div>
      <div id="research-side-view" className={panelOpen ? "research-side-view" : "hidden"}>
        {sidePanel}
      </div>
    </ThreadPrimitive.Root>
  );
}

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
  const native = z
    .object({
      type: z.literal("mcpToolCall"),
      status: z.literal("completed"),
      result: z.object({ structuredContent: z.unknown() }),
      error: z.null(),
    })
    .safeParse(payload.data);
  const value = native.success ? native.data.result.structuredContent : payload.data;
  const envelope = z.object({ data: z.unknown() }).safeParse(value);
  const data = envelope.success ? envelope.data.data : value;
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
