import type { IncomingMessage, ServerResponse } from "node:http";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
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
) => Promise<unknown>;

export function createProductMcpServer(
  tools: readonly ToolManifestEntry[],
  call: ProductToolHandler,
): McpServer {
  const server = new McpServer({ name: "swarmx-products", version: "1.0.0" });
  for (const tool of tools) {
    server.registerTool(
      tool.name,
      {
        description: tool.description,
        inputSchema: z.fromJSONSchema(tool.inputSchema as never),
      },
      async (args, extra) => result(await call(tool.name, args, extra.signal)),
    );
  }
  return server;
}

export async function handleMcp(
  request: IncomingMessage,
  response: ServerResponse,
  tools: readonly ToolManifestEntry[],
  call: ProductToolHandler,
): Promise<void> {
  const server = createProductMcpServer(tools, call);
  const transport = new StreamableHTTPServerTransport();
  try {
    await server.connect(transport as never);
    await transport.handleRequest(request, response);
  } finally {
    await server.close();
  }
}

function result(value: unknown) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(value) }],
    structuredContent:
      typeof value === "object" && value !== null ? (value as Record<string, unknown>) : { value },
  };
}
