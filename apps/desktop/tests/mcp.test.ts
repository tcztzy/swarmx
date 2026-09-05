import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { describe, expect, it, vi } from "vitest";
import { createProductMcpServer } from "../src/host/mcp.js";

describe("ProductServices MCP", () => {
  it("uses the official MCP server and delegates to the single owner", async () => {
    const call = vi.fn(async () => ({ ok: true }));
    const server = createProductMcpServer(
      [
        {
          name: "swarm",
          description: "Call an ACP Swarm Agent.",
          inputSchema: { type: "object", properties: {}, additionalProperties: false },
        },
      ],
      call,
    );
    const client = new Client({ name: "swarmx-test", version: "0.1.0" });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    try {
      await Promise.all([client.connect(clientTransport), server.connect(serverTransport)]);
      await expect(client.listTools()).resolves.toMatchObject({ tools: [{ name: "swarm" }] });
      await client.callTool({ name: "swarm", arguments: {} });
      expect(call).toHaveBeenCalledOnce();
    } finally {
      await Promise.all([client.close(), server.close()]);
    }
  });
});
