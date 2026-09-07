import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { describe, expect, it, vi } from "vitest";
import { createProductMcpServer } from "../src/host/mcp.js";

describe("ProductServices MCP", () => {
  it.each([
    { value: { ok: true }, structured: { ok: true } },
    { value: ["swarm"], structured: { value: ["swarm"] } },
  ])(
    "uses the official MCP server and preserves product result $value",
    async ({ value, structured }) => {
      const call = vi.fn(async () => value);
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
        const metadata = {
          callId: "call",
          "x-codex-turn-metadata": { thread_id: "thread", turn_id: "turn" },
        };
        expect(
          await client.callTool({ name: "swarm", arguments: {}, _meta: metadata }),
        ).toMatchObject({
          structuredContent: structured,
          content: [{ type: "text", text: JSON.stringify(value) }],
        });
        expect(call).toHaveBeenCalledOnce();
        expect(call).toHaveBeenCalledWith("swarm", {}, expect.any(AbortSignal), metadata);
      } finally {
        await Promise.all([client.close(), server.close()]);
      }
    },
  );
});
