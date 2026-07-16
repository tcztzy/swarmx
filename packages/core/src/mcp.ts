import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { currentRequestSignal, throwIfCurrentRequestCancelled } from "./acp.js";
import type { McpServerConfig } from "./types.js";
import { SWARMX_VERSION } from "./version.js";

interface McpTool {
  name: string;
  description?: string;
  inputSchema: Record<string, unknown>;
}

export interface LocalMcpTool extends McpTool {
  call(arguments_: Record<string, unknown>): Promise<unknown>;
}

export class McpManager {
  private clients: Map<string, Client> = new Map();
  private tools: McpTool[] = [];
  private localTools = new Map<string, LocalMcpTool>();

  addLocalTools(tools: readonly LocalMcpTool[]): void {
    for (const tool of tools) {
      if (
        this.localTools.has(tool.name) ||
        this.tools.some((candidate) => candidate.name === tool.name)
      ) {
        throw new Error(`Duplicate MCP tool name: ${tool.name}`);
      }
      this.localTools.set(tool.name, tool);
      this.tools.push({
        name: tool.name,
        description: tool.description,
        inputSchema: tool.inputSchema,
      });
    }
  }

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

    const client = new Client({ name: "swarmx", version: SWARMX_VERSION }, { capabilities: {} });
    const signal = currentRequestSignal();
    const abortConnection = (): void => {
      void Promise.allSettled([client.close(), transport.close()]);
    };
    signal?.addEventListener("abort", abortConnection, { once: true });

    try {
      throwIfCurrentRequestCancelled();
      await client.connect(transport);
      throwIfCurrentRequestCancelled();
      const { tools } = await client.listTools(undefined, signal ? { signal } : undefined);
      throwIfCurrentRequestCancelled();
      this.clients.set(name, client);
      for (const tool of tools) {
        this.tools.push({
          name: tool.name,
          description: tool.description,
          inputSchema: tool.inputSchema as Record<string, unknown>,
        });
      }
    } catch (error) {
      await Promise.allSettled([client.close(), transport.close()]);
      throwIfCurrentRequestCancelled();
      throw error;
    } finally {
      signal?.removeEventListener("abort", abortConnection);
    }
  }

  async callTool(
    name: string,
    arguments_: Record<string, unknown>,
    _context?: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    const localTool = this.localTools.get(name);
    if (localTool) {
      const result = await localTool.call(arguments_);
      return result !== null && typeof result === "object" && !Array.isArray(result)
        ? (result as Record<string, unknown>)
        : { result };
    }

    for (const [, client] of this.clients) {
      try {
        const signal = currentRequestSignal();
        const result = await client.callTool(
          {
            name,
            arguments: arguments_,
          },
          undefined,
          signal ? { signal } : undefined,
        );
        return { result: result.content };
      } catch {
        throwIfCurrentRequestCancelled();
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

  async close(): Promise<void> {
    const clients = [...this.clients.values()];
    this.clients.clear();
    this.localTools.clear();
    this.tools = [];
    await Promise.allSettled(clients.map((client) => client.close()));
  }
}
