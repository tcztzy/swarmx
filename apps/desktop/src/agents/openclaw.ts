import { randomUUID } from "node:crypto";
import { GatewayClient } from "@openclaw/gateway-client";
import type {
  AgentEvent,
  ChatEvent,
  QuestionRequestedEvent,
  SessionsCreateResult,
} from "@openclaw/gateway-protocol";
import { z } from "zod";
import type { NativeAgent, Observer } from "./types.js";

export async function createOpenClaw(): Promise<NativeAgent> {
  const url = process.env.OPENCLAW_GATEWAY_URL;
  const token = process.env.OPENCLAW_GATEWAY_TOKEN;
  if (!url || !token)
    throw new Error("OpenClaw unavailable: set OPENCLAW_GATEWAY_URL and OPENCLAW_GATEWAY_TOKEN.");
  const ready = Promise.withResolvers<void>();
  const running = new Map<
    string,
    {
      observer: Observer;
      runId: string;
      streamed: boolean;
      done: ReturnType<typeof Promise.withResolvers<void>>;
    }
  >();
  const fail = (error: Error) => {
    client.stop();
    ready.reject(error);
    for (const turn of running.values()) turn.done.reject(error);
  };
  const client = new GatewayClient({
    url,
    token,
    clientName: "cli",
    mode: "cli",
    clientDisplayName: "SwarmX",
    scopes: ["operator.read", "operator.write", "operator.approvals"],
    onHelloOk: () => ready.resolve(),
    onConnectError: fail,
    onClose: (_code, reason) => fail(new Error(`OpenClaw disconnected: ${reason}`)),
    onEvent: (frame) => {
      void (async () => {
        if (frame.event === "chat") {
          const event = frame.payload as ChatEvent;
          const turn = running.get(event.sessionKey);
          if (!turn || event.runId !== turn.runId) return;
          turn.observer.raw(frame);
          if (event.state === "delta") {
            if (event.replace)
              throw new Error("OpenClaw replaced streamed text; reload native history.");
            turn.streamed = true;
            turn.observer.text(event.runId, event.deltaText);
          } else if (event.state === "final" || event.state === "aborted") {
            if (!turn.streamed && event.message)
              projectMessage(event.runId, event.message, turn.observer);
            turn.done.resolve();
          } else if (event.state === "error")
            turn.done.reject(new Error(event.errorMessage ?? "OpenClaw run failed."));
        } else if (frame.event === "agent") {
          const event = frame.payload as AgentEvent;
          const turn = [...running.values()].find((turn) => turn.runId === event.runId);
          if (!turn) return;
          turn.observer.raw(frame);
          const data = event.data;
          if (event.stream === "tool")
            turn.observer.tool(
              z.string().parse(data.toolCallId),
              z.string().parse(data.name),
              data.args,
              data.phase === "result" ? data.result : undefined,
            );
          if (event.stream === "reasoning" && typeof data.delta === "string")
            turn.observer.text(`${event.runId}:reasoning`, data.delta, "reasoning");
        } else if (frame.event === "exec.approval.requested") {
          const event = z
            .object({
              id: z.string(),
              request: z.object({ sessionKey: z.string(), command: z.string() }),
            })
            .parse(frame.payload);
          const turn = running.get(event.request.sessionKey);
          if (!turn) return;
          turn.observer.raw(frame);
          const answer = await turn.observer.interact({
            id: event.id,
            title: event.request.command,
            schema: {
              type: "object",
              properties: { decision: { type: "string", enum: ["allow-once", "deny"] } },
              required: ["decision"],
            },
          });
          await client.request("exec.approval.resolve", {
            id: event.id,
            decision:
              answer === undefined
                ? "deny"
                : z.object({ decision: z.enum(["allow-once", "deny"]) }).parse(answer).decision,
          });
        } else if (frame.event === "question.requested") {
          const event = frame.payload as QuestionRequestedEvent;
          const turn = event.sessionKey ? running.get(event.sessionKey) : undefined;
          if (!turn) return;
          turn.observer.raw(frame);
          const answer = await turn.observer.interact({
            id: event.id,
            title: "OpenClaw needs input",
            schema: {
              type: "object",
              properties: Object.fromEntries(
                event.questions.map((q) => [q.questionId, { type: "string", title: q.question }]),
              ),
              required: event.questions.map((q) => q.questionId),
            },
          });
          await client.request("question.resolve", {
            id: event.id,
            ...(answer === undefined
              ? { cancel: true }
              : {
                  answers: {
                    answers: Object.fromEntries(
                      Object.entries(z.record(z.string(), z.string()).parse(answer)).map(
                        ([key, text]) => [key, [text]],
                      ),
                    ),
                  },
                }),
          });
        }
      })().catch(fail);
    },
  });
  client.start();
  await ready.promise;
  return {
    name: "OpenClaw",
    async list() {
      const { sessions } = z
        .object({
          sessions: z.array(
            z.object({
              key: z.string(),
              displayName: z.string().optional(),
              derivedTitle: z.string().optional(),
              updatedAt: z.number().optional(),
            }),
          ),
        })
        .parse(await client.request("sessions.list", { includeDerivedTitles: true }));
      return sessions.map((row) => ({
        sessionId: row.key,
        title: row.displayName ?? row.derivedTitle ?? row.key,
        ...(row.updatedAt === undefined
          ? {}
          : { updatedAt: new Date(row.updatedAt).toISOString() }),
      }));
    },
    async create() {
      return (await client.request<SessionsCreateResult>("sessions.create", {})).key;
    },
    async read(id, observer) {
      const { messages } = z
        .object({ messages: z.array(z.unknown()) })
        .parse(await client.request("chat.history", { sessionKey: id }));
      for (const [index, message] of messages.entries())
        projectMessage(`${id}:${index}`, message, observer);
    },
    async start(id, text, observer) {
      if (running.has(id)) throw new Error("OpenClaw session is busy.");
      const turn = {
        observer,
        runId: randomUUID(),
        streamed: false,
        done: Promise.withResolvers<void>(),
      };
      running.set(id, turn);
      try {
        await Promise.all([
          turn.done.promise,
          client.request("chat.send", {
            sessionKey: id,
            message: text,
            idempotencyKey: turn.runId,
          }),
        ]);
      } finally {
        running.delete(id);
      }
    },
    async steer(id, text) {
      await client.request("chat.send", {
        sessionKey: id,
        message: text,
        idempotencyKey: randomUUID(),
        queueMode: "steer",
      });
    },
    async interrupt(id) {
      await client.request("chat.abort", { sessionKey: id });
    },
    async dispose() {
      await client.stopAndWait();
    },
  };
}

function projectMessage(id: string, raw: unknown, observer: Observer): void {
  observer.raw(raw);
  const row = z
    .object({
      role: z.string(),
      toolCallId: z.string().optional(),
      toolName: z.string().optional(),
      content: z.union([
        z.string(),
        z.array(
          z.object({
            type: z.string(),
            text: z.string().optional(),
            thinking: z.string().optional(),
            id: z.string().optional(),
            name: z.string().optional(),
            arguments: z.unknown().optional(),
          }),
        ),
      ]),
    })
    .parse(raw);
  if (row.role === "toolResult" && row.toolCallId) {
    observer.tool(row.toolCallId, row.toolName ?? "tool", undefined, row.content);
    return;
  }
  if (row.role !== "user" && row.role !== "assistant") return;
  if (typeof row.content === "string") observer.text(id, row.content, row.role);
  else
    for (const [index, block] of row.content.entries()) {
      if (block.text) observer.text(`${id}:${index}`, block.text, row.role);
      if (block.thinking) observer.text(`${id}:${index}`, block.thinking, "reasoning");
      if (block.type === "toolCall" && block.id && block.name)
        observer.tool(block.id, block.name, block.arguments);
    }
}
