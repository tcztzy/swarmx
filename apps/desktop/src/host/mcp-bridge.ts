import { connect, type Socket } from "node:net";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { createProductMcpServer, type ToolManifestEntry } from "./mcp.js";

interface Channel {
  request(payload: Record<string, unknown>, signal: AbortSignal): Promise<unknown>;
  close(): void;
}

function main(): Promise<void> {
  const path = process.env.SWARMX_MCP_SOCKET;
  const token = process.env.SWARMX_MCP_TOKEN;
  if (!path || !token) throw new Error("SWARMX_MCP_SOCKET and SWARMX_MCP_TOKEN are required.");
  return new Promise<void>((ready, fail) => {
    const socket = connect(path);
    socket.once("error", fail);
    socket.once("connect", () => {
      const channel = openChannel(socket);
      const list = z.array(
        z.object({
          name: z.string(),
          description: z.string(),
          inputSchema: z.record(z.string(), z.unknown()),
        }),
      );
      channel
        .request({ token, list: true }, AbortSignal.timeout(30_000))
        .then((manifest) => {
          const server = createProductMcpServer(
            list.parse(manifest) as ToolManifestEntry[],
            (name, args, signal) => channel.request({ token, tool: name, args }, signal),
          );
          return server.connect(new StdioServerTransport());
        })
        .then(ready, fail);
    });
  });
}

function openChannel(socket: Socket): Channel {
  const pending = new Map<number, { resolve(value: unknown): void; reject(error: Error): void }>();
  let buffer = "";
  let nextId = 1;
  socket.on("data", (chunk: Buffer) => {
    buffer += chunk.toString("utf8");
    for (let index = buffer.indexOf("\n"); index >= 0; index = buffer.indexOf("\n")) {
      const line = buffer.slice(0, index);
      buffer = buffer.slice(index + 1);
      const reply = z
        .object({
          id: z.number(),
          ok: z.boolean(),
          value: z.unknown().optional(),
          error: z.string().optional(),
        })
        .safeParse(parse(line));
      if (!reply.success) continue;
      const entry = pending.get(reply.data.id);
      if (!entry) continue;
      pending.delete(reply.data.id);
      if (reply.data.ok) entry.resolve(reply.data.value);
      else entry.reject(new Error(reply.data.error ?? "MCP bridge request failed."));
    }
  });
  socket.on("close", () => {
    for (const entry of pending.values())
      entry.reject(new Error("SwarmX Host tool bridge closed."));
    pending.clear();
  });
  return {
    request(payload, signal) {
      return new Promise((resolve, reject) => {
        const id = nextId++;
        pending.set(id, { resolve, reject });
        signal.addEventListener(
          "abort",
          () => {
            if (pending.delete(id))
              reject(
                signal.reason instanceof Error ? signal.reason : new Error("MCP call aborted."),
              );
          },
          { once: true },
        );
        socket.write(`${JSON.stringify({ id, ...payload })}\n`);
      });
    },
    close: () => socket.end(),
  };
}

function parse(line: string): unknown {
  try {
    return JSON.parse(line);
  } catch {
    return undefined;
  }
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
