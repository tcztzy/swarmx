import { randomUUID } from "node:crypto";
import { PassThrough } from "node:stream";
import {
  getSessionMessages,
  listSessions,
  type Options,
  type Query,
  query,
  type SDKUserMessage,
} from "@anthropic-ai/claude-agent-sdk";
import { ElicitResultSchema } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import type { AgentOptions, NativeAgent, Observer } from "./types.js";

export async function createClaude(options: AgentOptions): Promise<NativeAgent> {
  const fresh = new Set<string>();
  const running = new Map<string, { query: Query; input: PassThrough }>();
  return {
    name: "Claude",
    async list() {
      return (await listSessions({ dir: options.cwd })).map((session) => ({
        sessionId: session.sessionId,
        title: session.summary,
        updatedAt: new Date(session.lastModified).toISOString(),
      }));
    },
    async create() {
      const id = randomUUID();
      fresh.add(id);
      return id;
    },
    async read(id, observer) {
      if (fresh.has(id)) return;
      for (const row of await getSessionMessages(id, { dir: options.cwd })) {
        observer.raw(row);
        if (row.type !== "system") projectMessage(row.uuid, row.type, row.message, observer);
      }
    },
    async start(id, text, observer) {
      if (running.has(id)) throw new Error("Claude session is busy.");
      const input = new PassThrough({ objectMode: true });
      const nativeOptions: Options = {
        cwd: options.cwd,
        ...(fresh.has(id) ? { sessionId: id } : { resume: id }),
        settingSources: ["user", "project", "local"],
        includePartialMessages: true,
        mcpServers: { swarmx: { type: "http", ...options.mcp } },
        canUseTool: async (name, args, context) => {
          observer.tool(context.toolUseID, name, args);
          if (name === "AskUserQuestion") {
            const questions = z.array(z.object({ question: z.string() })).parse(args.questions);
            const answer = await observer.interact({
              id: context.toolUseID,
              title: "Claude needs input",
              schema: {
                type: "object",
                properties: Object.fromEntries(
                  questions.map((q) => [q.question, { type: "string", title: q.question }]),
                ),
                required: questions.map((q) => q.question),
              },
            });
            return answer === undefined
              ? { behavior: "deny", message: "User cancelled." }
              : {
                  behavior: "allow",
                  updatedInput: {
                    ...args,
                    answers: z.record(z.string(), z.string()).parse(answer),
                  },
                };
          }
          const answer = await observer.interact({
            id: context.toolUseID,
            title: context.title ?? `Allow ${name}?`,
            schema: {
              type: "object",
              properties: { allow: { type: "boolean" } },
              required: ["allow"],
            },
          });
          return answer !== undefined && z.object({ allow: z.boolean() }).parse(answer).allow
            ? { behavior: "allow", updatedInput: args }
            : { behavior: "deny", message: "User declined." };
        },
        onElicitation: async (request, context) => {
          if (!request.requestedSchema) throw new Error("SwarmX supports form elicitations only.");
          const answer = await observer.interact({
            id: context.requestId,
            title: request.message,
            schema: request.requestedSchema,
          });
          return ElicitResultSchema.parse(
            answer === undefined ? { action: "cancel" } : { action: "accept", content: answer },
          );
        },
      };
      const native = query({
        prompt: input as AsyncIterable<SDKUserMessage>,
        options: nativeOptions,
      });
      running.set(id, { query: native, input });
      input.write(userMessage(id, text));
      const messageIds = new Map<string, string>();
      const streamed = new Set<string>();
      let completed = false;
      try {
        for await (const message of native) {
          observer.raw(message);
          if (message.type === "stream_event") {
            const event = message.event;
            const parent = message.parent_tool_use_id ?? "";
            if (event.type === "message_start") messageIds.set(parent, event.message.id);
            if (event.type === "content_block_delta") {
              const messageId = messageIds.get(parent);
              if (!messageId) throw new Error("Claude stream has no message_start.");
              const key = `${messageId}:${event.index}`;
              if (event.delta.type === "text_delta") observer.text(key, event.delta.text);
              if (event.delta.type === "thinking_delta")
                observer.text(key, event.delta.thinking, "reasoning");
              streamed.add(messageId);
            }
          } else if (message.type === "assistant") {
            projectMessage(
              message.uuid,
              "assistant",
              message.message,
              observer,
              streamed.has(message.message.id),
            );
          } else if (message.type === "user") {
            projectMessage(message.uuid ?? randomUUID(), "user", message.message, observer, true);
          } else if (message.type === "result") {
            fresh.delete(id);
            completed = true;
            if (message.is_error)
              throw new Error(
                message.subtype === "success" ? message.result : message.errors.join("\n"),
              );
          } else if (
            message.type === "system" &&
            message.subtype === "session_state_changed" &&
            message.state === "idle" &&
            completed
          ) {
            break;
          }
        }
      } finally {
        running.delete(id);
        input.end();
        native.close();
      }
    },
    async steer(id, text) {
      const turn = running.get(id);
      if (!turn) throw new Error("No running Claude turn.");
      turn.input.write({ ...userMessage(id, text), priority: "now" } satisfies SDKUserMessage);
    },
    async interrupt(id) {
      await running.get(id)?.query.interrupt();
    },
    async dispose() {
      for (const turn of running.values()) turn.query.close();
    },
  };
}

function userMessage(id: string, text: string): SDKUserMessage {
  return {
    type: "user",
    session_id: id,
    message: { role: "user", content: text },
    parent_tool_use_id: null,
  };
}

const Block = z.object({
  type: z.string(),
  id: z.string().optional(),
  name: z.string().optional(),
  text: z.string().optional(),
  thinking: z.string().optional(),
  input: z.unknown().optional(),
  content: z.unknown().optional(),
  tool_use_id: z.string().optional(),
});
function projectMessage(
  id: string,
  role: "user" | "assistant",
  raw: unknown,
  observer: Observer,
  streamed = false,
) {
  const { content } = z.object({ content: z.union([z.string(), z.array(Block)]) }).parse(raw);
  if (typeof content === "string") {
    if (!streamed) observer.text(id, content, role);
    return;
  }
  for (const [index, part] of content.entries()) {
    if (!streamed && part.type === "text" && part.text)
      observer.text(`${id}:${index}`, part.text, role);
    if (!streamed && part.type === "thinking" && part.thinking)
      observer.text(`${id}:${index}`, part.thinking, "reasoning");
    if (part.type === "tool_use" && part.id && part.name)
      observer.tool(part.id, part.name, part.input);
    if (part.type === "tool_result" && part.tool_use_id)
      observer.tool(part.tool_use_id, "tool", undefined, part.content);
  }
}
