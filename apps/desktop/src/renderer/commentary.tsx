import type { AbstractAgent } from "@ag-ui/client";
import { ThreadPrimitive, useAuiState } from "@assistant-ui/react";
import {
  type ComponentProps,
  type ComponentType,
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  type HistoryMessage,
  type MessageActivity,
  readMessageActivity,
  readToolActivity,
  type ToolActivity,
  type TurnTiming,
} from "../message-activity.js";
import {
  ReasoningContent,
  ReasoningRoot,
  ReasoningText,
  ReasoningTrigger,
} from "./components/assistant-ui/elements/reasoning.aui.js";
import { t } from "./i18n.js";
import { ToolGroup } from "./tool-ui.js";

export function useMessageActivity(agent: AbstractAgent) {
  const [activity, setActivity] = useState<Record<string, MessageActivity>>({});
  const [tools, setTools] = useState<Record<string, ToolActivity>>({});
  const turns = useRef(new Map<string, TurnTiming>());
  const restore = useCallback((messages: HistoryMessage[]) => {
    turns.current.clear();
    const restored: Record<string, MessageActivity> = {};
    const restoredTools: Record<string, ToolActivity> = {};
    for (const message of messages) {
      if (message._tool && message.role === "assistant") {
        for (const call of message.toolCalls ?? []) restoredTools[call.id] = message._tool;
      }
      if (!message._meta) continue;
      restored[message.id] = message._meta;
      const { turnId, startedAt, durationMs } = message._meta;
      turns.current.set(turnId, { turnId, startedAt, durationMs });
    }
    setActivity(restored);
    setTools(restoredTools);
  }, []);
  useEffect(() => {
    const subscription = agent.subscribe({
      onCustomEvent({ event }) {
        if (event.name !== "swarmx.activity") return;
        const tool = readToolActivity(event.value);
        if (tool) {
          const { toolCallId, ...metadata } = tool;
          setTools((previous) => ({
            ...previous,
            [toolCallId]: { ...previous[toolCallId], ...metadata },
          }));
        }
        const metadata = readMessageActivity(event.value);
        if (!metadata) return;
        const { messageId, phase, ...timing } = metadata;
        const turn = { ...turns.current.get(timing.turnId), ...timing };
        turns.current.set(turn.turnId, turn);
        setActivity((previous) => {
          if (messageId && phase) {
            if (previous[messageId]?.phase === phase && previous[messageId]?.turnId === turn.turnId)
              return previous;
            return { ...previous, [messageId]: { ...turn, phase } };
          }
          return Object.fromEntries(
            Object.entries(previous).map(([id, value]) => [
              id,
              value.turnId === turn.turnId ? { ...value, ...turn } : value,
            ]),
          );
        });
      },
    });
    return () => subscription.unsubscribe();
  }, [agent]);
  return { activity, tools, restore };
}

function Worked({
  timing,
  running,
  streaming,
  children,
}: {
  timing: MessageActivity;
  running: boolean;
  streaming: boolean;
  children: ReactNode;
}) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    if (!running || timing.durationMs != null || timing.startedAt == null) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [running, timing.durationMs, timing.startedAt]);
  const duration =
    timing.durationMs ??
    (running && timing.startedAt != null ? now - timing.startedAt * 1000 : undefined);
  const seconds = duration === undefined ? undefined : Math.max(0, Math.round(duration / 1000));
  const label =
    seconds === undefined
      ? "Worked"
      : `Worked for ${
          seconds >= 3600 ? `${Math.floor(seconds / 3600)}h ` : ""
        }${seconds >= 60 ? `${Math.floor(seconds / 60) % 60}m ` : ""}${seconds % 60}s`;
  if (!children)
    return seconds === undefined ? null : (
      <div className="worked-turn border-b pb-4 text-sm text-muted-foreground">{label}</div>
    );
  return (
    <ReasoningRoot streaming={streaming} variant="ghost" className="worked-turn">
      <ReasoningTrigger active={streaming}>{streaming ? t("正在处理…") : label}</ReasoningTrigger>
      <ReasoningContent>
        <ReasoningText>{children}</ReasoningText>
      </ReasoningContent>
    </ReasoningRoot>
  );
}

export function CommentaryMessages({
  activity,
  components,
}: {
  activity: Record<string, MessageActivity>;
  components: ComponentProps<typeof ThreadPrimitive.MessageByIndex>["components"] & {
    WorkMessage: ComponentType;
    FinalMessage: ComponentType;
    CommentaryMessage: ComponentType;
  };
}) {
  const messages = useAuiState((state) => state.thread.messages);
  const running = useAuiState((state) => state.thread.isRunning);
  const result: ReactNode[] = [];
  let first = 0;
  let turnId: string | undefined;
  const render = (start: number, end: number, view?: "work" | "answer") => {
    const nodes: ReactNode[] = [];
    const messageAt = (index: number) => (
      <ThreadPrimitive.MessageByIndex
        key={messages[index]?.id}
        index={index}
        components={
          view === "work"
            ? { ...components, AssistantMessage: components.WorkMessage }
            : view === "answer"
              ? { ...components, AssistantMessage: components.FinalMessage }
              : messages[index] && activity[messages[index].id]?.phase === "commentary"
                ? { ...components, AssistantMessage: components.CommentaryMessage }
                : components
        }
      />
    );
    const toolsOnly = (index: number) => {
      const message = messages[index];
      return (
        message?.role === "assistant" &&
        message.content.some((part) => part.type === "tool-call") &&
        message.content.every((part) => part.type === "tool-call" || part.type === "data")
      );
    };
    for (let index = start; index < end; index++) {
      if (view === "answer" || !toolsOnly(index)) {
        nodes.push(messageAt(index));
        continue;
      }
      const firstTool = index;
      while (index + 1 < end && toolsOnly(index + 1)) index++;
      const group = messages.slice(firstTool, index + 1);
      nodes.push(
        <ToolGroup
          key={`tools:${messages[firstTool]?.id}`}
          calls={group.flatMap((message) =>
            message.content.flatMap((part) => (part.type === "tool-call" ? [part] : [])),
          )}
          running={group.some((message) => message.status?.type === "running")}
        >
          {group.map((_, offset) => messageAt(firstTool + offset))}
        </ToolGroup>,
      );
    }
    return nodes;
  };
  for (const [index, message] of messages.entries()) {
    if (message.role === "user") {
      result.push(...render(first, index + 1));
      first = index + 1;
      turnId = undefined;
      continue;
    }
    const metadata = activity[message.id];
    if (metadata) {
      if (turnId !== undefined && turnId !== metadata.turnId) {
        result.push(...render(first, index));
        first = index;
      }
      turnId = metadata.turnId;
    }
    if (metadata?.phase !== "final_answer") continue;
    const hasWork =
      message.content.some((part) => part.type === "tool-call") ||
      messages
        .slice(first, index)
        .some((previous) =>
          previous.content.some(
            (part) =>
              part.type === "tool-call" ||
              (part.type === "text" &&
                part.text.length > 0 &&
                activity[previous.id]?.phase === "commentary" &&
                activity[previous.id]?.turnId === metadata.turnId),
          ),
        );
    if (!hasWork) result.push(...render(first, index));
    result.push(
      <Worked
        key={`work:${metadata.turnId}`}
        timing={metadata}
        running={running && message.status?.type === "running"}
        streaming={false}
      >
        {hasWork && (
          <>
            {render(first, index)}
            {message.content.some((part) => part.type === "tool-call") &&
              render(index, index + 1, "work")}
          </>
        )}
      </Worked>,
    );
    result.push(...render(index, index + 1, hasWork ? "answer" : undefined));
    first = index + 1;
  }
  const liveCommentary = messages
    .slice(first)
    .find(
      (message) =>
        activity[message.id]?.phase === "commentary" &&
        message.content.some((part) => part.type === "text" && part.text.length > 0),
    );
  const timing = liveCommentary && activity[liveCommentary.id];
  if (running && timing) {
    result.push(
      <Worked key={`work:${timing.turnId}`} timing={timing} running streaming>
        {render(first, messages.length)}
      </Worked>,
    );
  } else result.push(...render(first, messages.length));
  return result;
}
