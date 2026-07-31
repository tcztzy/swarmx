import type { DesktopMessageChunk as MessageChunk } from "../../shared/desktop-api.js";
import { isRecord } from "./text-utils.js";

export function withRequestTiming(
  messages: MessageChunk[],
  startedAt: string,
  endedAt: string,
): MessageChunk[] {
  const durationMs = Math.max(1, Date.parse(endedAt) - Date.parse(startedAt));
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

export function mergeStreamingMessage(
  messages: readonly MessageChunk[],
  incoming: MessageChunk,
): MessageChunk[] {
  const previous = messages.at(-1);
  if (
    incoming.kind === "tool_progress" &&
    previous?.kind === "tool_progress" &&
    previous.render?.invocationId === incoming.render?.invocationId
  ) {
    return [...messages.slice(0, -1), mergeToolProgress(previous, incoming)];
  }
  const mergeable = incoming.kind === "thinking" || incoming.kind === "message";
  if (
    mergeable &&
    previous?.kind === incoming.kind &&
    previous.role === incoming.role &&
    previous.agent === incoming.agent &&
    previous.toolName === incoming.toolName
  ) {
    return [
      ...messages.slice(0, -1),
      { ...previous, content: `${previous.content}${incoming.content}` },
    ];
  }
  return [...messages, incoming];
}

export function mergeToolProgress(
  previous: MessageChunk | undefined,
  incoming: MessageChunk,
): MessageChunk {
  if (!previous || toolProgressMode(incoming) === "replace") return incoming;
  const content = `${previous.content}${incoming.content}`;
  const previousStream = toolProgressStream(previous);
  const incomingStream = toolProgressStream(incoming);
  return {
    ...previous,
    ...incoming,
    content,
    structuredContent: {
      ...(isRecord(previous.structuredContent) ? previous.structuredContent : {}),
      ...(isRecord(incoming.structuredContent) ? incoming.structuredContent : {}),
      output: content,
      stream:
        previousStream && incomingStream && previousStream === incomingStream
          ? previousStream
          : "combined",
      mode: "append",
    },
  };
}

function toolProgressMode(message: MessageChunk): string | undefined {
  return isRecord(message.structuredContent) && typeof message.structuredContent.mode === "string"
    ? message.structuredContent.mode
    : undefined;
}

function toolProgressStream(message: MessageChunk): string | undefined {
  return isRecord(message.structuredContent) && typeof message.structuredContent.stream === "string"
    ? message.structuredContent.stream
    : undefined;
}

export function messageKey(msg: MessageChunk): string {
  return [msg.role, msg.kind, msg.agent ?? "", msg.toolName ?? "", msg.content].join("\u001f");
}
