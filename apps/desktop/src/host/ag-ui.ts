import { randomUUID } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";
import {
  type AGUIEvent,
  EventSchemas,
  EventType,
  type Interrupt,
  type Message,
  type ResumeEntry,
  type RunAgentInput,
  RunAgentInputSchema,
} from "@ag-ui/core";
import { EventEncoder } from "@ag-ui/encoder";
import type { Interaction, NativeAgent, Observer } from "../agents/types.js";

export const parseAgUiInput = (raw: unknown): RunAgentInput => RunAgentInputSchema.parse(raw);

export class AgUiBridge {
  private readonly turns = new Map<string, Turn>();
  constructor(private readonly agent: NativeAgent) {}

  async handle(request: IncomingMessage, response: ServerResponse): Promise<void> {
    const input = parseAgUiInput(await readJson(request));
    const stream = new Projection(response, input.threadId, input.runId);
    const abort = () => {
      if (!stream.finished) void this.cancel(input.threadId);
    };
    request.once("aborted", abort);
    response.once("close", abort);
    try {
      let turn = this.turns.get(input.threadId);
      if (input.resume) {
        if (!turn) throw new Error("No pending interaction.");
        turn.projection = stream;
        turn.resume(input.resume);
      } else {
        if (turn) throw new Error("Session is busy.");
        const message = input.messages.at(-1);
        if (message?.role !== "user") throw new Error("An AG-UI run must end with a user message.");
        const text = messageText(message);
        if (!text.trim()) throw new Error("The user message cannot be empty.");
        turn = new Turn(stream, (observer) => this.agent.start(input.threadId, text, observer));
        this.turns.set(input.threadId, turn);
        void turn.completed.then(() => this.turns.delete(input.threadId));
      }
      const outcome = await Promise.race([
        turn.completed,
        turn.signal.promise.then((interrupt) => ({ kind: "interaction", interrupt }) as const),
      ]);
      turn.projection = undefined;
      if (outcome.kind === "interaction") stream.interrupt(outcome.interrupt);
      else if (outcome.kind === "error") stream.error(outcome.error);
      else stream.success();
    } catch (error) {
      stream.error(error);
    } finally {
      request.off("aborted", abort);
      response.off("close", abort);
    }
  }

  async cancel(id: string): Promise<void> {
    this.turns.get(id)?.cancelInteraction();
    await this.agent.interrupt(id);
  }
}

class Turn implements Observer {
  readonly completed: Promise<{ kind: "complete" } | { kind: "error"; error: unknown }>;
  signal = Promise.withResolvers<Interrupt>();
  private pending: { id: string; resolve(value: unknown): void } | undefined;

  constructor(
    public projection: Projection | undefined,
    start: (observer: Observer) => Promise<void>,
  ) {
    this.completed = Promise.resolve()
      .then(() => start(this))
      .then(
        () => ({ kind: "complete" }),
        (error: unknown) => ({ kind: "error", error }),
      );
  }
  text(id: string, text: string, role: "user" | "assistant" | "reasoning" = "assistant") {
    if (role !== "user") this.projection?.text(id, text, role);
  }
  tool(id: string, name: string, input: unknown, output?: unknown) {
    this.projection?.tool(id, name, input, output);
  }
  raw(event: unknown) {
    this.projection?.send({ type: EventType.CUSTOM, name: "native", value: event });
  }
  interact(request: Interaction): Promise<unknown> {
    if (this.pending) throw new Error("A native interaction is already pending.");
    const answer = Promise.withResolvers<unknown>();
    this.pending = { id: request.id, resolve: answer.resolve };
    this.signal.resolve({
      id: request.id,
      reason: "input_required",
      message: request.title,
      responseSchema: request.schema,
    });
    return answer.promise;
  }
  resume(entries: ResumeEntry[]) {
    if (!this.pending || entries.length !== 1 || entries[0]?.interruptId !== this.pending.id)
      throw new Error("AG-UI resume must answer the pending native interaction.");
    const entry = entries[0];
    const pending = this.pending;
    this.pending = undefined;
    this.signal = Promise.withResolvers<Interrupt>();
    pending.resolve(entry.status === "cancelled" ? undefined : entry.payload);
  }
  cancelInteraction() {
    this.pending?.resolve(undefined);
    this.pending = undefined;
  }
}

class Projection {
  private readonly encoder = new EventEncoder();
  private readonly tools = new Set<string>();
  private part: { id: string; role: "assistant" | "reasoning" } | undefined;
  finished = false;

  constructor(
    private readonly response: ServerResponse,
    private readonly threadId: string,
    private readonly runId: string,
  ) {
    response.writeHead(200, {
      "cache-control": "no-cache, no-store",
      connection: "keep-alive",
      "content-type": this.encoder.getContentType(),
      "x-accel-buffering": "no",
    });
    this.send({ type: EventType.RUN_STARTED, threadId, runId });
  }
  text(id: string, delta: string, role: "assistant" | "reasoning") {
    if (!delta) return;
    if (this.part?.id !== id || this.part.role !== role) {
      this.endPart();
      this.part = { id, role };
      if (role === "reasoning") {
        this.send({ type: EventType.REASONING_START, messageId: id });
        this.send({ type: EventType.REASONING_MESSAGE_START, messageId: id, role });
      } else this.send({ type: EventType.TEXT_MESSAGE_START, messageId: id, role });
    }
    this.send({
      type:
        role === "reasoning" ? EventType.REASONING_MESSAGE_CONTENT : EventType.TEXT_MESSAGE_CONTENT,
      messageId: id,
      delta,
    });
  }
  tool(id: string, name: string, input: unknown, output?: unknown) {
    this.endPart();
    if (!this.tools.has(id)) {
      this.tools.add(id);
      this.send({ type: EventType.TOOL_CALL_START, toolCallId: id, toolCallName: name });
      this.send({
        type: EventType.TOOL_CALL_ARGS,
        toolCallId: id,
        delta: JSON.stringify(input ?? {}),
      });
      this.send({ type: EventType.TOOL_CALL_END, toolCallId: id });
    }
    if (output !== undefined)
      this.send({
        type: EventType.TOOL_CALL_RESULT,
        messageId: randomUUID(),
        toolCallId: id,
        content: JSON.stringify(output),
      });
  }
  interrupt(interrupt: Interrupt) {
    this.endPart();
    this.send({
      type: EventType.RUN_FINISHED,
      threadId: this.threadId,
      runId: this.runId,
      outcome: { type: "interrupt", interrupts: [interrupt] },
    });
    this.close();
  }
  success() {
    this.endPart();
    this.send({
      type: EventType.RUN_FINISHED,
      threadId: this.threadId,
      runId: this.runId,
      outcome: { type: "success" },
    });
    this.close();
  }
  error(error: unknown) {
    this.endPart();
    this.send({
      type: EventType.RUN_ERROR,
      message: error instanceof Error ? error.message : String(error),
    });
    this.close();
  }
  private endPart() {
    if (!this.part) return;
    const { id, role } = this.part;
    if (role === "reasoning") {
      this.send({ type: EventType.REASONING_MESSAGE_END, messageId: id });
      this.send({ type: EventType.REASONING_END, messageId: id });
    } else this.send({ type: EventType.TEXT_MESSAGE_END, messageId: id });
    this.part = undefined;
  }
  send(event: AGUIEvent) {
    if (!this.finished && !this.response.destroyed)
      this.response.write(
        this.encoder.encodeSSE(EventSchemas.parse({ timestamp: Date.now(), ...event })),
      );
  }
  private close() {
    this.finished = true;
    this.response.end();
  }
}

export async function loadAgUiHistory(agent: NativeAgent, sessionId: string): Promise<Message[]> {
  const messages: Message[] = [];
  const byId = new Map<string, Message>();
  const tools = new Set<string>();
  await agent.read(sessionId, {
    text(id, text, role = "assistant") {
      const existing = byId.get(id);
      if (existing && typeof existing.content === "string") existing.content += text;
      else {
        const message = { id, role, content: text } as Message;
        byId.set(id, message);
        messages.push(message);
      }
    },
    tool(id, name, input, output) {
      if (!tools.has(id)) {
        tools.add(id);
        messages.push({
          id: `call:${id}`,
          role: "assistant",
          toolCalls: [
            { id, type: "function", function: { name, arguments: JSON.stringify(input ?? {}) } },
          ],
        });
      }
      if (output !== undefined)
        messages.push({
          id: `result:${id}`,
          role: "tool",
          toolCallId: id,
          content: JSON.stringify(output),
        });
    },
    raw() {},
    interact: async () => {
      throw new Error("History cannot request interaction.");
    },
  });
  return messages;
}

function messageText(message: Message): string {
  if (typeof message.content === "string") return message.content;
  if (!Array.isArray(message.content) || !message.content.every((part) => part.type === "text"))
    throw new Error("SwarmX accepts text-only AG-UI user messages.");
  return message.content.map((part) => part.text).join("");
}

async function readJson(request: IncomingMessage): Promise<unknown> {
  if (!(request.headers["content-type"] ?? "").startsWith("application/json"))
    throw new Error("Expected an application/json body.");
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request) {
    const bytes = Buffer.from(chunk);
    size += bytes.length;
    if (size > 1024 * 1024) throw new Error("AG-UI input is too large.");
    chunks.push(bytes);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}
