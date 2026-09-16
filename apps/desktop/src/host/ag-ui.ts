import { randomUUID } from "node:crypto";
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
import type { RunResult } from "@swarmx/swarm";
import type { Interaction, NativeAgent, Observer } from "../agents/types.js";
import { RunOptionsSchema } from "../agents/types.js";
import {
  type Activity,
  type HistoryMessage,
  type MessageActivity,
  readMessageActivity,
  readToolActivity,
  type ToolActivity,
} from "../message-activity.js";

export const parseAgUiInput = (raw: unknown): RunAgentInput => RunAgentInputSchema.parse(raw);

export interface AgUiSink {
  event(event: AGUIEvent): void;
}

export class AgUiBridge {
  private readonly turns = new Map<string, Turn>();
  constructor(private readonly agent: NativeAgent) {}

  async run(input: RunAgentInput, sink: AgUiSink): Promise<void> {
    const stream = new AgUiStream(sink, input.threadId, input.runId);
    try {
      let turn = this.turns.get(input.threadId);
      if (input.resume) {
        if (!turn) throw new Error("No pending interaction.");
        turn.stream = stream;
        turn.resume(input.resume);
      } else {
        if (turn) throw new Error("Session is busy.");
        const message = input.messages.at(-1);
        if (message?.role !== "user") throw new Error("An AG-UI run must end with a user message.");
        const text = messageText(message);
        if (!text.trim()) throw new Error("The user message cannot be empty.");
        const options = RunOptionsSchema.optional().parse(input.forwardedProps);
        turn = new Turn(stream, (observer) =>
          this.agent.start(input.threadId, text, observer, options),
        );
        this.turns.set(input.threadId, turn);
        void turn.completed.then(() => this.turns.delete(input.threadId));
      }
      const outcome = await Promise.race([
        turn.completed,
        turn.signal.promise.then((interrupt) => ({ kind: "interaction", interrupt }) as const),
      ]);
      turn.stream = undefined;
      if (outcome.kind === "interaction") stream.interrupt(outcome.interrupt);
      else if (outcome.kind === "error") stream.error(outcome.error);
      else stream.complete(outcome.result);
    } catch (error) {
      stream.error(error);
    }
  }

  async cancel(id: string): Promise<void> {
    const turn = this.turns.get(id);
    if (!turn) return;
    turn.cancelInteraction();
    await this.agent.interrupt(id);
  }
}

class Turn implements Observer {
  readonly completed: Promise<
    { kind: "complete"; result: RunResult } | { kind: "error"; error: unknown }
  >;
  signal = Promise.withResolvers<Interrupt>();
  private readonly pending: { interrupt: Interrupt; resolve(value: unknown): void }[] = [];

  constructor(
    public stream: AgUiStream | undefined,
    start: (observer: Observer) => Promise<RunResult>,
  ) {
    this.completed = Promise.resolve()
      .then(() => start(this))
      .then(
        (result) => ({ kind: "complete" as const, result }),
        (error: unknown) => ({ kind: "error" as const, error }),
      );
  }
  text(id: string, text: string, role: "user" | "assistant" | "reasoning" = "assistant") {
    if (role !== "user") this.stream?.text(id, text, role);
  }
  tool(id: string, name: string, input: unknown, output?: unknown) {
    this.stream?.tool(id, name, input, output);
  }
  activity(event: Activity) {
    this.stream?.send({ type: EventType.CUSTOM, name: "swarmx.activity", value: event });
  }
  raw(event: unknown) {
    this.stream?.send({ type: EventType.CUSTOM, name: "native", value: event });
  }
  interact(request: Interaction, signal?: AbortSignal): Promise<unknown> {
    if (signal?.aborted) return Promise.resolve(undefined);
    const answer = Promise.withResolvers<unknown>();
    const interrupt: Interrupt = {
      id: request.id,
      reason: "input_required",
      message: request.title,
      responseSchema: request.schema,
    };
    const pending = { interrupt, resolve: answer.resolve };
    const cancel = () => {
      const index = this.pending.indexOf(pending);
      if (index !== -1) this.pending.splice(index, 1);
      answer.resolve(undefined);
      if (index === 0) {
        this.signal = Promise.withResolvers<Interrupt>();
        if (this.pending[0]) this.signal.resolve(this.pending[0].interrupt);
      }
    };
    signal?.addEventListener("abort", cancel, { once: true });
    this.pending.push(pending);
    if (this.pending.length === 1) this.signal.resolve(interrupt);
    return answer.promise.finally(() => signal?.removeEventListener("abort", cancel));
  }
  resume(entries: ResumeEntry[]) {
    const pending = this.pending[0];
    if (!pending || entries.length !== 1 || entries[0]?.interruptId !== pending.interrupt.id)
      throw new Error("AG-UI resume must answer the pending native interaction.");
    const entry = entries[0];
    this.pending.shift();
    this.signal = Promise.withResolvers<Interrupt>();
    if (this.pending[0]) this.signal.resolve(this.pending[0].interrupt);
    pending.resolve(entry.status === "cancelled" ? undefined : entry.payload);
  }
  cancelInteraction() {
    for (const pending of this.pending.splice(0)) pending.resolve(undefined);
  }
}

class AgUiStream {
  private readonly tools = new Set<string>();
  private part: { id: string; role: "assistant" | "reasoning" } | undefined;
  finished = false;

  constructor(
    private readonly sink: AgUiSink,
    private readonly threadId: string,
    private readonly runId: string,
  ) {
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
    this.finished = true;
  }
  complete(result: RunResult) {
    this.endPart();
    this.send({
      type: EventType.RUN_FINISHED,
      threadId: this.threadId,
      runId: this.runId,
      result,
      ...(result.stopReason === "end_turn" ? { outcome: { type: "success" as const } } : {}),
    });
    this.finished = true;
  }
  error(error: unknown) {
    this.endPart();
    this.send({
      type: EventType.RUN_ERROR,
      message: error instanceof Error ? error.message : String(error),
    });
    this.finished = true;
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
    if (!this.finished) this.sink.event(EventSchemas.parse({ timestamp: Date.now(), ...event }));
  }
}

export async function loadAgUiHistory(
  agent: NativeAgent,
  sessionId: string,
): Promise<HistoryMessage[]> {
  const messages: Message[] = [];
  const activity = new Map<string, MessageActivity>();
  const toolActivity = new Map<string, ToolActivity>();
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
    activity(event) {
      const tool = readToolActivity(event);
      if (tool) {
        const { toolCallId, ...metadata } = tool;
        toolActivity.set(`call:${toolCallId}`, {
          ...toolActivity.get(`call:${toolCallId}`),
          ...metadata,
        });
      }
      const metadata = readMessageActivity(event);
      if (metadata?.messageId && metadata.phase) {
        const { messageId, phase, ...timing } = metadata;
        activity.set(messageId, { phase, ...timing });
      }
    },
    interact: async () => {
      throw new Error("History cannot request interaction.");
    },
  });
  return messages.map((message) => {
    const metadata = activity.get(message.id);
    const tool = toolActivity.get(message.id);
    return {
      ...message,
      ...(metadata ? { _meta: metadata } : {}),
      ...(tool ? { _tool: tool } : {}),
    };
  });
}

function messageText(message: Message): string {
  if (typeof message.content === "string") return message.content;
  if (!Array.isArray(message.content) || !message.content.every((part) => part.type === "text"))
    throw new Error("SwarmX accepts text-only AG-UI user messages.");
  return message.content.map((part) => part.text).join("");
}
