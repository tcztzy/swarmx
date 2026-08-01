import type { ChatMessage, MessageChunk, SessionData } from "@swarmx/core";
import { normalizeMessageChunk } from "@swarmx/core";
import type { AgentChunkSender } from "./agent-chunk-publisher.js";

export function sessionChatMessages(session: SessionData | null): ChatMessage[] {
  if (!session) return [];
  return session.messages.flatMap((message): ChatMessage[] => {
    if (message.kind !== "message") return [];
    if (!isChatRole(message.role)) return [];
    return [
      {
        role: message.role,
        content: message.content,
        ...(message.attachments?.length ? { attachments: message.attachments } : {}),
      },
    ];
  });
}

export function timedMessages(
  messages: readonly MessageChunk[],
  startedAtMs: number,
  endedAtMs = Date.now(),
): MessageChunk[] {
  const startedAt = new Date(startedAtMs).toISOString();
  const endedAt = new Date(endedAtMs).toISOString();
  const durationMs = Math.max(1, endedAtMs - startedAtMs);
  return messages.map((message) => ({
    ...message,
    render: {
      ...(message.render ?? {}),
      startedAt: message.render?.startedAt ?? startedAt,
      endedAt: message.render?.endedAt ?? endedAt,
      durationMs: message.render?.durationMs ?? durationMs,
    },
  }));
}

export function interruptedMessages(
  messages: readonly MessageChunk[],
  startedAtMs: number,
  endedAtMs = Date.now(),
): MessageChunk[] {
  const timed = timedMessages(messages, startedAtMs, endedAtMs);
  const activities: Array<{
    invocationId?: string;
    toolName?: string;
    messageIndexes: number[];
    terminal: boolean;
  }> = [];

  timed.forEach((message, messageIndex) => {
    if (message.kind === "tool_call") {
      activities.push({
        invocationId: message.render?.invocationId,
        toolName: message.toolName,
        messageIndexes: [messageIndex],
        terminal: false,
      });
      return;
    }
    if (message.kind !== "tool_progress" && message.kind !== "tool_result") return;

    const invocationId = message.render?.invocationId;
    const exactMatch = invocationId
      ? activities.find((activity) => activity.invocationId === invocationId && !activity.terminal)
      : undefined;
    const fallbackCandidates = activities.filter(
      (activity) => activity.toolName === message.toolName && !activity.terminal,
    );
    const fallbackMatch =
      message.kind === "tool_result"
        ? fallbackCandidates[0]
        : fallbackCandidates[fallbackCandidates.length - 1];
    const activity = exactMatch ?? fallbackMatch;
    const terminal =
      message.kind === "tool_result" &&
      !["queued", "running"].includes(
        normalizeMessageChunk(message, { status: message.render?.status }).status,
      );

    if (activity) {
      activity.messageIndexes.push(messageIndex);
      activity.terminal = terminal;
      return;
    }
    activities.push({
      invocationId,
      toolName: message.toolName,
      messageIndexes: [messageIndex],
      terminal,
    });
  });

  const interruptedIndexes = new Set(
    activities
      .filter((activity) => !activity.terminal)
      .flatMap((activity) => activity.messageIndexes),
  );
  return timed.map((message, index) =>
    interruptedIndexes.has(index)
      ? {
          ...message,
          render: {
            ...(message.render ?? {}),
            status: "canceled",
          },
        }
      : message,
  );
}

export function publishSessionMessages(sender: AgentChunkSender, sessionId: string): void {
  if (!sender.isDestroyed()) sender.send("session:messages", { sessionId });
}

export function assertFinalAssistantMessage(messages: readonly MessageChunk[]): void {
  if (
    !messages.some(
      (message) =>
        message.kind === "message" &&
        message.role === "assistant" &&
        message.content.trim().length > 0,
    )
  ) {
    throw new Error("Agent run ended without a final assistant response.");
  }
}

function isChatRole(role: string): role is ChatMessage["role"] {
  return role === "user" || role === "assistant" || role === "system" || role === "tool";
}
