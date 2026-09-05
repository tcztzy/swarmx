import { randomUUID } from "node:crypto";
import { z } from "zod";
import { rpcProcess } from "./rpc-process.js";
import type { AgentOptions, NativeAgent, Observer } from "./types.js";

const Session = z.object({ session_id: z.string(), stored_session_id: z.string() });
const Event = z.object({
  type: z.string(),
  session_id: z.string(),
  payload: z.record(z.string(), z.unknown()).optional(),
});

export async function createHermes(options: AgentOptions): Promise<NativeAgent> {
  const python = process.env.SWARMX_HERMES_PYTHON;
  if (!python)
    throw new Error(
      "Hermes unavailable: set SWARMX_HERMES_PYTHON to its installed Python environment.",
    );
  const sessions = new Map<string, string>();
  const running = new Map<
    string,
    {
      observer: Observer;
      messageId: string;
      streamed: boolean;
      done: ReturnType<typeof Promise.withResolvers<void>>;
    }
  >();
  const rpc = rpcProcess(
    python,
    ["-m", "tui_gateway.entry"],
    options.cwd,
    async (request) => {
      if (request.method !== "event") throw new Error(`Unknown Hermes request: ${request.method}`);
      const event = Event.parse(request.params);
      const turn = running.get(event.session_id);
      if (!turn) return;
      const data = event.payload ?? {};
      turn.observer.raw(request);
      switch (event.type) {
        case "message.start":
          turn.messageId = randomUUID();
          turn.streamed = false;
          break;
        case "message.delta":
          turn.streamed = true;
          turn.observer.text(turn.messageId, z.string().parse(data.text));
          break;
        case "thinking.delta":
        case "reasoning.delta":
          turn.observer.text(
            `${turn.messageId}:reasoning`,
            z.string().parse(data.text),
            "reasoning",
          );
          break;
        case "tool.start":
        case "tool.complete":
          turn.observer.tool(
            z.string().parse(data.tool_id),
            z.string().parse(data.name),
            data.args,
            event.type === "tool.complete" ? data : undefined,
          );
          break;
        case "approval.request": {
          const answer = await turn.observer.interact({
            id: z.string().parse(data.request_id),
            title: String(data.command ?? data.description ?? "Approve Hermes action?"),
            schema: {
              type: "object",
              properties: { choice: { type: "string", enum: ["once", "deny"] } },
              required: ["choice"],
            },
          });
          await rpc.request("approval.respond", {
            session_id: event.session_id,
            request_id: data.request_id,
            choice:
              answer === undefined
                ? "deny"
                : z.object({ choice: z.enum(["once", "deny"]) }).parse(answer).choice,
          });
          break;
        }
        case "clarify.request": {
          const answer = await turn.observer.interact({
            id: z.string().parse(data.request_id),
            title: String(data.question ?? "Hermes needs input"),
            schema: {
              type: "object",
              properties: { answer: { type: "string" } },
              required: ["answer"],
            },
          });
          await rpc.request("clarify.respond", {
            session_id: event.session_id,
            request_id: data.request_id,
            answer:
              answer === undefined ? "" : z.object({ answer: z.string() }).parse(answer).answer,
          });
          break;
        }
        case "message.complete":
          if (data.error) turn.done.reject(new Error(String(data.error)));
          else {
            if (!turn.streamed && data.text)
              turn.observer.text(turn.messageId, z.string().parse(data.text));
            turn.done.resolve();
          }
          break;
        case "error":
          turn.done.reject(new Error(z.string().parse(data.message)));
          break;
      }
    },
    (error) => {
      for (const turn of running.values()) turn.done.reject(error);
    },
  );
  async function session(id: string): Promise<string> {
    const live = sessions.get(id);
    if (live) return live;
    const resumed = Session.parse(await rpc.request("session.resume", { session_id: id }));
    sessions.set(id, resumed.session_id);
    return resumed.session_id;
  }
  return {
    name: "Hermes",
    async list() {
      const { sessions: rows } = z
        .object({
          sessions: z.array(
            z.object({
              id: z.string(),
              title: z.string().nullish(),
              preview: z.string().optional(),
            }),
          ),
        })
        .parse(await rpc.request("session.list", {}));
      return rows.map((row) => ({ sessionId: row.id, title: row.title ?? row.preview ?? row.id }));
    },
    async create() {
      const created = Session.parse(await rpc.request("session.create", { cwd: options.cwd }));
      sessions.set(created.stored_session_id, created.session_id);
      return created.stored_session_id;
    },
    async read(id, observer) {
      const history = z
        .object({
          messages: z.array(
            z
              .object({
                role: z.string(),
                text: z.string().optional(),
                name: z.string().optional(),
                args: z.unknown().optional(),
              })
              .passthrough(),
          ),
        })
        .parse(await rpc.request("session.history", { session_id: await session(id) }));
      for (const [index, row] of history.messages.entries()) {
        observer.raw(row);
        const key = `${id}:${index}`;
        if (row.role === "user" || row.role === "assistant")
          observer.text(key, row.text ?? "", row.role);
        if (row.role === "tool") observer.tool(key, row.name ?? "tool", row.args, row);
      }
    },
    async start(id, text, observer) {
      const live = await session(id);
      if (running.has(live)) throw new Error("Hermes session is busy.");
      const turn = {
        observer,
        messageId: randomUUID(),
        streamed: false,
        done: Promise.withResolvers<void>(),
      };
      running.set(live, turn);
      try {
        await Promise.all([
          turn.done.promise,
          rpc.request("prompt.submit", { session_id: live, text }),
        ]);
      } finally {
        running.delete(live);
      }
    },
    async steer(id, text) {
      await rpc.request("session.steer", { session_id: await session(id), text });
    },
    async interrupt(id) {
      await rpc.request("session.interrupt", { session_id: await session(id) });
    },
    dispose: () => rpc.dispose(),
  };
}
