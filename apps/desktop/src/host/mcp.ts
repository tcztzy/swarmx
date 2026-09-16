import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

export interface ToolManifestEntry {
  readonly name: string;
  readonly description: string;
  readonly inputSchema: Record<string, unknown>;
}

export type ProductToolHandler = (
  name: string,
  args: unknown,
  signal: AbortSignal,
  meta: Record<string, unknown> | undefined,
) => Promise<unknown>;

export function createProductMcpServer(
  tools: readonly ToolManifestEntry[],
  call: ProductToolHandler,
): McpServer {
  const server = new McpServer({ name: "swarmx-products", version: "3.3.0" });
  for (const tool of tools) {
    server.registerTool(
      tool.name,
      {
        description: tool.description,
        inputSchema: z.fromJSONSchema(tool.inputSchema as never),
      },
      async (args, extra) => result(await call(tool.name, args, extra.signal, extra._meta)),
    );
  }
  return server;
}

function result(value: unknown) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(value) }],
    structuredContent:
      typeof value === "object" && value !== null && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : { value },
  };
}
