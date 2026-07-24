import type { MessageChunk } from "@swarmx/core";

const TERMINAL_PROGRESS_BATCH_MS = 75;
const TERMINAL_PROGRESS_DELAY_MS = 250;
const TERMINAL_PROGRESS_MAX_BYTES = 1024 * 1024;
const TERMINAL_PROGRESS_MAX_LINES = 10_000;
const TERMINAL_PROGRESS_TRUNCATED = "\n[live output truncated]\n";

export interface AgentChunkSender {
  isDestroyed(): boolean;
  send(channel: string, payload: unknown): void;
}

export interface AgentChunkPublisher {
  (chunk: MessageChunk): void;
  close(): void;
}

interface PendingTerminalProgress {
  bytes: number;
  content: string;
  lines: number;
  mode: "append" | "replace";
  startedAt: number;
  streams: Set<string>;
  template: MessageChunk;
  timer?: ReturnType<typeof setTimeout>;
  truncated: boolean;
}

export function agentChunkPublisher(
  sender: AgentChunkSender,
  requestId: string,
): AgentChunkPublisher {
  const progress = new Map<string, PendingTerminalProgress>();
  const startedAt = new Map<string, number>();
  const send = (chunk: MessageChunk): void => {
    if (!sender.isDestroyed()) sender.send("agent:chunk", { requestId, chunk });
  };
  const discard = (invocationId: string): void => {
    const pending = progress.get(invocationId);
    if (pending?.timer) clearTimeout(pending.timer);
    progress.delete(invocationId);
    startedAt.delete(invocationId);
  };
  const flush = (invocationId: string): void => {
    const pending = progress.get(invocationId);
    if (!pending || !pending.content) return;
    if (pending.timer) clearTimeout(pending.timer);
    pending.timer = undefined;
    const structured = recordProperty(pending.template, "structuredContent");
    const stream = pending.streams.size === 1 ? [...pending.streams][0] : "combined";
    send({
      ...pending.template,
      content: pending.content,
      structuredContent: {
        ...structured,
        output: pending.content,
        stream,
        mode: pending.mode,
      },
    });
    pending.content = "";
    pending.mode = "append";
    pending.streams.clear();
  };
  const schedule = (invocationId: string, pending: PendingTerminalProgress): void => {
    if (pending.timer) return;
    const delay = Math.max(
      TERMINAL_PROGRESS_BATCH_MS,
      TERMINAL_PROGRESS_DELAY_MS - (Date.now() - pending.startedAt),
    );
    pending.timer = setTimeout(() => flush(invocationId), delay);
    pending.timer.unref?.();
  };

  const publish = ((chunk: MessageChunk) => {
    const invocationId = chunk.render?.invocationId;
    if (chunk.kind === "tool_call" && invocationId) {
      startedAt.set(invocationId, Date.now());
      send(chunk);
      return;
    }
    if (chunk.kind === "tool_progress") {
      if (!invocationId) return;
      const incomingMode =
        stringProperty(chunk.structuredContent, "mode") === "replace" ? "replace" : "append";
      const pending = progress.get(invocationId) ?? {
        bytes: 0,
        content: "",
        lines: 0,
        mode: incomingMode,
        startedAt: startedAt.get(invocationId) ?? Date.now(),
        streams: new Set<string>(),
        template: chunk,
        truncated: false,
      };
      progress.set(invocationId, pending);
      if (pending.truncated) return;
      if (incomingMode === "replace") {
        pending.bytes = 0;
        pending.content = "";
        pending.lines = 0;
        pending.mode = "replace";
        pending.streams.clear();
      }
      pending.template = chunk;
      const bounded = boundedTerminalProgress(pending, chunk.content);
      pending.content = incomingMode === "replace" ? bounded : `${pending.content}${bounded}`;
      pending.streams.add(stringProperty(chunk.structuredContent, "stream") ?? "combined");
      schedule(invocationId, pending);
      return;
    }
    if (invocationId && progress.has(invocationId)) {
      const pending = progress.get(invocationId);
      if (pending && Date.now() - pending.startedAt >= TERMINAL_PROGRESS_DELAY_MS) {
        flush(invocationId);
      }
      if (chunk.kind === "tool_result" && isTerminalToolChunk(chunk)) {
        discard(invocationId);
      }
    }
    send(chunk);
  }) as AgentChunkPublisher;

  publish.close = () => {
    for (const [invocationId, pending] of progress) {
      if (Date.now() - pending.startedAt >= TERMINAL_PROGRESS_DELAY_MS) flush(invocationId);
      discard(invocationId);
    }
  };
  return publish;
}

function boundedTerminalProgress(pending: PendingTerminalProgress, content: string): string {
  const remainingLines = Math.max(0, TERMINAL_PROGRESS_MAX_LINES - pending.lines);
  const lineBounded = takeLines(content, remainingLines);
  const remainingBytes = Math.max(0, TERMINAL_PROGRESS_MAX_BYTES - pending.bytes);
  const byteBounded = takeUtf8Bytes(lineBounded.value, remainingBytes);
  const truncated = lineBounded.truncated || byteBounded.truncated;
  pending.bytes += Buffer.byteLength(byteBounded.value);
  pending.lines += countNewlines(byteBounded.value);
  if (truncated) pending.truncated = true;
  return `${byteBounded.value}${truncated ? TERMINAL_PROGRESS_TRUNCATED : ""}`;
}

function takeLines(content: string, remainingLines: number): { value: string; truncated: boolean } {
  if (remainingLines <= 0) return { value: "", truncated: content.length > 0 };
  let lines = 0;
  for (let index = 0; index < content.length; index++) {
    if (content[index] !== "\n") continue;
    lines += 1;
    if (lines === remainingLines && index < content.length - 1) {
      return { value: content.slice(0, index + 1), truncated: true };
    }
  }
  return { value: content, truncated: false };
}

function takeUtf8Bytes(content: string, maxBytes: number): { value: string; truncated: boolean } {
  if (Buffer.byteLength(content) <= maxBytes) return { value: content, truncated: false };
  let low = 0;
  let high = content.length;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (Buffer.byteLength(content.slice(0, middle)) <= maxBytes) low = middle;
    else high = middle - 1;
  }
  return { value: content.slice(0, low), truncated: true };
}

function countNewlines(content: string): number {
  let count = 0;
  for (const character of content) if (character === "\n") count += 1;
  return count;
}

function isTerminalToolChunk(chunk: MessageChunk): boolean {
  return !["queued", "running"].includes(chunk.render?.status ?? "succeeded");
}

function stringProperty(value: unknown, key: string): string | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const property = (value as Record<string, unknown>)[key];
  return typeof property === "string" ? property : undefined;
}

function recordProperty(value: unknown, key: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const property = (value as Record<string, unknown>)[key];
  return property && typeof property === "object" && !Array.isArray(property)
    ? (property as Record<string, unknown>)
    : {};
}
