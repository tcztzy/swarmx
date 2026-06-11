import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import type { McpServerConfig } from "./types.js";

interface McpTool {
  name: string;
  description?: string;
  inputSchema: Record<string, unknown>;
}

export class McpManager {
  private clients: Map<string, Client> = new Map();
  private tools: McpTool[] = [];

  async addServer(name: string, config: McpServerConfig): Promise<void> {
    if (this.clients.has(name)) return;

    let transport: StdioClientTransport;

    if (config.type === "stdio" && config.command) {
      transport = new StdioClientTransport({
        command: config.command,
        args: config.args ?? [],
        env: config.env,
      });
    } else {
      throw new Error(
        `Unsupported MCP transport: ${config.type}. Currently only "stdio" is supported.`,
      );
    }

    const client = new Client({ name: "swarmx", version: "3.0.0" }, { capabilities: {} });

    await client.connect(transport);
    this.clients.set(name, client);

    const { tools } = await client.listTools();
    for (const tool of tools) {
      this.tools.push({
        name: tool.name,
        description: tool.description,
        inputSchema: tool.inputSchema as Record<string, unknown>,
      });
    }
  }

  async callTool(
    name: string,
    arguments_: Record<string, unknown>,
    _context?: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    for (const [, client] of this.clients) {
      try {
        const result = await client.callTool({
          name,
          arguments: arguments_,
        });
        return { result: result.content };
      } catch {
        // Try the next connected MCP client.
      }
    }
    throw new Error(`Tool ${name} not found or all servers failed`);
  }

  toolsForOpenai(): Array<{
    type: "function";
    function: {
      name: string;
      description: string;
      parameters: Record<string, unknown>;
    };
  }> {
    return this.tools.map((t) => ({
      type: "function" as const,
      function: {
        name: t.name,
        description: t.description ?? "",
        parameters: t.inputSchema,
      },
    }));
  }
}
