import type { Session } from "@swarmx/swarm";
import { z } from "zod";
import type { ClientRequest } from "./generated/ClientRequest.js";
import type { ServerNotification } from "./generated/ServerNotification.js";
import type { ServerRequest } from "./generated/ServerRequest.js";
import type { ThreadItem } from "./generated/v2/ThreadItem.js";
import type { ThreadListResponse } from "./generated/v2/ThreadListResponse.js";
import type { ThreadReadResponse } from "./generated/v2/ThreadReadResponse.js";
import type { ThreadStartResponse } from "./generated/v2/ThreadStartResponse.js";
import type { TurnStartResponse } from "./generated/v2/TurnStartResponse.js";
import { rpcProcess } from "./rpc-process.js";
import type { AgentOptions, NativeAgent, Observer } from "./types.js";

export async function createCodex(options: AgentOptions): Promise<NativeAgent> {
  const running = new Map<
    string,
    {
      observer: Observer;
      ready: ReturnType<typeof Promise.withResolvers<string | undefined>>;
      streamed: Set<string>;
      done: ReturnType<typeof Promise.withResolvers<void>>;
    }
  >();
  const loaded = new Set<string>();
  const fresh = new Set<string>();
  const rpc = rpcProcess(
    "codex",
    ["app-server"],
    options.cwd,
    async (raw) => {
      const message = raw as ServerNotification | ServerRequest;
      const params = message.params;
      const turn =
        params && "threadId" in params && params.threadId
          ? running.get(params.threadId)
          : undefined;
      if (!turn) {
        if (raw.id !== undefined) throw new Error(`No active Codex turn for ${message.method}.`);
        return;
      }
      turn.observer.raw(message);
      if ("id" in message) return interaction(message, turn.observer);
      switch (message.method) {
        case "turn/started":
          turn.ready.resolve(message.params.turn.id);
          break;
        case "item/agentMessage/delta":
        case "item/reasoning/summaryTextDelta":
        case "item/reasoning/textDelta":
          turn.streamed.add(message.params.itemId);
          turn.observer.text(
            message.params.itemId,
            message.params.delta,
            message.method === "item/agentMessage/delta" ? "assistant" : "reasoning",
          );
          break;
        case "item/started":
          if (!["userMessage", "agentMessage", "reasoning"].includes(message.params.item.type))
            projectItem(message.params.item, turn.observer, false);
          break;
        case "item/completed":
          if (!turn.streamed.has(message.params.item.id))
            projectItem(message.params.item, turn.observer);
          break;
        case "turn/completed":
          if (message.params.turn.error)
            turn.done.reject(new Error(message.params.turn.error.message));
          else turn.done.resolve();
          break;
      }
    },
    (error) => {
      for (const turn of running.values()) turn.done.reject(error);
    },
  );
  async function request<T>(
    method: ClientRequest["method"],
    params: ClientRequest["params"],
  ): Promise<T> {
    return (await rpc.request(method, params ?? {})) as T;
  }
  try {
    await request("initialize", {
      clientInfo: { name: "swarmx", title: null, version: "0.1.0" },
      capabilities: { experimentalApi: true, requestAttestation: false },
    });
  } catch (error) {
    await rpc.dispose();
    throw error;
  }
  rpc.notify("initialized", {});
  const config = {
    mcp_servers: { swarmx: { url: options.mcp.url, http_headers: options.mcp.headers } },
  };
  return {
    name: "Codex",
    async list() {
      const sessions: Session[] = [];
      let cursor: string | null = null;
      do {
        const page: ThreadListResponse = await request("thread/list", { cwd: options.cwd, cursor });
        sessions.push(
          ...page.data.map((thread) => ({
            sessionId: thread.id,
            title: thread.name ?? thread.preview,
            updatedAt: new Date(thread.updatedAt * 1000).toISOString(),
          })),
        );
        cursor = page.nextCursor;
      } while (cursor);
      return sessions;
    },
    async create() {
      const { thread } = await request<ThreadStartResponse>("thread/start", {
        cwd: options.cwd,
        config,
      });
      loaded.add(thread.id);
      fresh.add(thread.id);
      return thread.id;
    },
    async read(id, observer) {
      if (fresh.has(id)) return;
      const { thread } = await request<ThreadReadResponse>("thread/read", {
        threadId: id,
        includeTurns: true,
      });
      if (thread.cwd !== options.cwd)
        throw new Error("Codex session belongs to another workspace.");
      for (const turn of thread.turns) for (const item of turn.items) projectItem(item, observer);
    },
    async start(id, text, observer) {
      if (running.has(id)) throw new Error("Codex session is busy.");
      const turn = {
        observer,
        streamed: new Set<string>(),
        done: Promise.withResolvers<void>(),
        ready: Promise.withResolvers<string | undefined>(),
      };
      running.set(id, turn);
      try {
        await Promise.all([
          turn.done.promise,
          (async () => {
            if (!loaded.has(id)) {
              await request("thread/resume", { threadId: id, cwd: options.cwd, config });
              loaded.add(id);
            }
            const started = await request<TurnStartResponse>("turn/start", {
              threadId: id,
              input: [{ type: "text", text, text_elements: [] }],
            });
            turn.ready.resolve(started.turn.id);
            fresh.delete(id);
          })(),
        ]);
      } finally {
        turn.ready.resolve(undefined);
        running.delete(id);
      }
    },
    async steer(id, text) {
      const turn = running.get(id);
      const turnId = await turn?.ready.promise;
      if (!turnId) throw new Error("No running Codex turn.");
      await request("turn/steer", {
        threadId: id,
        expectedTurnId: turnId,
        input: [{ type: "text", text, text_elements: [] }],
      });
    },
    async interrupt(id) {
      const turn = running.get(id);
      const turnId = await turn?.ready.promise;
      if (turnId) await request("turn/interrupt", { threadId: id, turnId });
    },
    dispose: () => rpc.dispose(),
  };
}

function projectItem(item: ThreadItem, observer: Observer, complete = true): void {
  observer.raw(item);
  if (item.type === "userMessage") {
    observer.text(
      item.id,
      item.content
        .filter((part) => part.type === "text")
        .map((part) => part.text)
        .join(""),
      "user",
    );
  } else if (item.type === "agentMessage") observer.text(item.id, item.text);
  else if (item.type === "reasoning") observer.text(item.id, item.summary.join("\n"), "reasoning");
  else
    observer.tool(
      item.id,
      "tool" in item ? item.tool : item.type,
      "arguments" in item ? item.arguments : item,
      complete ? item : undefined,
    );
}

async function interaction(message: ServerRequest, observer: Observer): Promise<unknown> {
  const id = String(message.id);
  switch (message.method) {
    case "item/commandExecution/requestApproval":
    case "item/fileChange/requestApproval": {
      const answer = await observer.interact({
        id,
        title: message.params.reason ?? "Approve Codex action?",
        schema: {
          type: "object",
          properties: { decision: { type: "string", enum: ["accept", "decline", "cancel"] } },
          required: ["decision"],
        },
      });
      return answer === undefined
        ? { decision: "cancel" }
        : z.object({ decision: z.enum(["accept", "decline", "cancel"]) }).parse(answer);
    }
    case "item/tool/requestUserInput": {
      const answer = await observer.interact({
        id,
        title: "Codex needs input",
        schema: {
          type: "object",
          properties: Object.fromEntries(
            message.params.questions.map((q) => [
              q.id,
              {
                type: "string",
                title: q.question,
                ...(q.options ? { examples: q.options.map((option) => option.label) } : {}),
              },
            ]),
          ),
          required: message.params.questions.map((q) => q.id),
        },
      });
      const values = answer === undefined ? {} : z.record(z.string(), z.string()).parse(answer);
      return {
        answers: Object.fromEntries(
          Object.entries(values).map(([key, value]) => [key, { answers: [value] }]),
        ),
      };
    }
    case "mcpServer/elicitation/request": {
      const request = message.params;
      if (request.mode !== "form") throw new Error("SwarmX supports form elicitations only.");
      const answer = await observer.interact({
        id,
        title: request.message,
        schema: request.requestedSchema as Record<string, unknown>,
      });
      return {
        action: answer === undefined ? "cancel" : "accept",
        content: answer ?? null,
        _meta: null,
      };
    }
    default:
      throw new Error(`Unsupported native Codex request: ${message.method}`);
  }
}
