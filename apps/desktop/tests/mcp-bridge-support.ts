import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import type { ProductServices } from "../src/host/product-services.js";

const bridge = fileURLToPath(new URL("../src/host/mcp-bridge.ts", import.meta.url));

export function socketOf(products: ProductServices): string {
  return products.mcpSocket;
}

export function bridgeTransport(
  socket: string,
  token: string,
  args: readonly string[] = [],
): StdioClientTransport {
  return new StdioClientTransport({
    command: process.execPath,
    args: [...args, "--import", "tsx", bridge],
    stderr: "ignore",
    env: {
      ...(process.env as Record<string, string>),
      ELECTRON_RUN_AS_NODE: "1",
      SWARMX_MCP_SOCKET: socket,
      SWARMX_MCP_TOKEN: token,
    },
  });
}

export async function bridgeClient(socket: string, token: string): Promise<Client> {
  const client = new Client({ name: "swarmx-bridge-test", version: "1.0.0" });
  await client.connect(bridgeTransport(socket, token));
  return client;
}

export async function bridgeCall(
  socket: string,
  token: string,
  name: string,
  args: unknown,
): Promise<Awaited<ReturnType<Client["callTool"]>>> {
  const client = await bridgeClient(socket, token);
  try {
    return await client.callTool({ name, arguments: args as Record<string, unknown> });
  } finally {
    await client.close();
  }
}
