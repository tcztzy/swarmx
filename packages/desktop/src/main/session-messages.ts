import type { ChatMessage, MessageChunk, SessionData } from "@swarmx/core";
import type { AgentChunkSender } from "./agent-chunk-publisher.js";

export function sessionChatMessages(session: SessionData | null): ChatMessage[] {
  if (!session) return [];
  return session.messages.flatMap((message): ChatMessage[] => {
    if (message.kind !== "message") return [];
    if (!isChatRole(message.role)) return [];
    return [{ role: message.role, content: message.content }];
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
