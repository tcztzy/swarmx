import { McpManager } from "./mcp.js";
import { type McpServerConfig, type ToolConfig, ToolConfigSchema } from "./types.js";

export class Tool {
  name: string;
  description?: string;
  parameters: Record<string, unknown>;
  returns?: Record<string, unknown>;
  instructions?: string;
  mcpServers: Map<string, McpServerConfig>;

  constructor(config: ToolConfig) {
    const parsed = ToolConfigSchema.parse(config);
    this.name = parsed.name;
    this.description = parsed.description;
    this.parameters = parsed.parameters ?? {};
    this.returns = parsed.returns;
    this.instructions = parsed.instructions;
    this.mcpServers = new Map(parsed.mcpServers ? Object.entries(parsed.mcpServers) : []);
  }

  async call(
    arguments_: Record<string, unknown>,
    context?: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    const manager = new McpManager();
    for (const [name, params] of this.mcpServers) {
      await manager.addServer(name, params);
    }
    return manager.callTool(this.name, arguments_, context);
  }
}
