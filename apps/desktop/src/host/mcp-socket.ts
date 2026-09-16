import { chmodSync, mkdirSync, rmSync } from "node:fs";
import { createServer, type Socket } from "node:net";
import { dirname } from "node:path";
import { z } from "zod";
import type { ToolManifestEntry } from "./mcp.js";

const requestSchema = z.object({
  id: z.number(),
  token: z.string(),
  list: z.literal(true).optional(),
  tool: z.string().optional(),
  args: z.unknown().optional(),
});

export interface McpSocketHandler {
  readonly tools: readonly ToolManifestEntry[];
  readonly known: (token: string) => boolean;
  readonly invoke: (
    token: string,
    tool: string,
    args: unknown,
    signal: AbortSignal,
  ) => Promise<unknown>;
}

export interface McpSocketServer {
  readonly path: string;
  dispose(): Promise<void>;
}

export function startMcpSocket(path: string, handler: McpSocketHandler): Promise<McpSocketServer> {
  mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
  rmSync(path, { force: true });
  const server = createServer();
  const sockets = new Set<Socket>();
  const flights = new Map<Socket, AbortController>();
  server.on("connection", (socket) => {
    sockets.add(socket);
    flights.set(socket, new AbortController());
    let buffer = "";
    socket.on("data", (chunk: Buffer) => {
      buffer += chunk.toString("utf8");
      for (let index = buffer.indexOf("\n"); index >= 0; index = buffer.indexOf("\n")) {
        const line = buffer.slice(0, index);
        buffer = buffer.slice(index + 1);
        const request = requestSchema.safeParse(parse(line));
        if (!request.success) {
          socket.write(`${JSON.stringify({ id: null, ok: false, error: "Invalid request." })}\n`);
          continue;
        }
        void respond(socket, request.data);
      }
    });
    socket.on("error", () => undefined);
    socket.on("close", () => {
      flights.get(socket)?.abort(new Error("MCP bridge disconnected."));
      flights.delete(socket);
      sockets.delete(socket);
    });
  });
  const respond = async (socket: Socket, request: z.infer<typeof requestSchema>) => {
    const reply = (payload: Record<string, unknown>) =>
      socket.write(`${JSON.stringify({ id: request.id, ...payload })}\n`);
    try {
      if (!handler.known(request.token)) throw new Error("Unknown MCP execution credential.");
      if (request.list === true) {
        reply({ ok: true, value: handler.tools });
        return;
      }
      if (request.tool === undefined) throw new Error("MCP request requires a tool.");
      const signal = flights.get(socket)?.signal ?? AbortSignal.abort();
      const value = await handler.invoke(request.token, request.tool, request.args, signal);
      reply({ ok: true, value });
    } catch (error) {
      reply({ ok: false, error: error instanceof Error ? error.message : String(error) });
    }
  };
  return new Promise((done, reject) => {
    server.once("error", reject);
    server.listen(path, () => {
      chmodSync(path, 0o600);
      done({
        path,
        dispose: async () => {
          for (const controller of flights.values())
            controller.abort(new Error("MCP socket closed."));
          for (const socket of sockets) socket.destroy();
          await new Promise<void>((closed) => server.close(() => closed()));
          rmSync(path, { force: true });
        },
      });
    });
  });
}

function parse(line: string): unknown {
  try {
    return JSON.parse(line);
  } catch {
    return undefined;
  }
}
