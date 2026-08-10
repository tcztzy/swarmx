import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import {
  MEMORY_RUNTIME_MAX_RESPONSE_BYTES,
  MEMORY_RUNTIME_SERVER_NAME,
  MEMORY_RUNTIME_TOOL_NAME,
  type MemoryRuntimeRequest,
  MemoryRuntimeRequestSchema,
  type MemoryRuntimeResult,
  MemoryRuntimeToolResponseSchema,
} from "@swarmx/core/memory-runtime-protocol";
import { type MemoryRuntimeLaunchSpec, MemoryRuntimeLaunchSpecSchema } from "@swarmx/runtime";

interface MemoryRuntimeToolCallResult {
  content: Array<{ type: string; text?: string }>;
  structuredContent?: unknown;
  isError?: boolean;
}

export interface MemoryRuntimeConnection {
  serverInfo(): { name: string; version: string } | undefined;
  listTools(): Promise<Array<{ name: string }>>;
  callTool(input: {
    name: string;
    arguments: Record<string, unknown>;
  }): Promise<MemoryRuntimeToolCallResult>;
  close(): Promise<void>;
}

export interface MemoryRuntimeHostOptions {
  launch: MemoryRuntimeLaunchSpec;
  connect?: (launch: MemoryRuntimeLaunchSpec) => Promise<MemoryRuntimeConnection>;
}

export class MemoryRuntimeHost {
  private readonly launch: MemoryRuntimeLaunchSpec;
  private readonly connectRuntime: NonNullable<MemoryRuntimeHostOptions["connect"]>;
  private connectionPromise?: Promise<MemoryRuntimeConnection>;
  private connection?: MemoryRuntimeConnection;
  private closed = false;

  constructor(options: MemoryRuntimeHostOptions) {
    this.launch = MemoryRuntimeLaunchSpecSchema.parse(options.launch);
    this.connectRuntime = options.connect ?? connectMemoryRuntime;
  }

  async request<Request extends MemoryRuntimeRequest>(
    request: Request,
  ): Promise<MemoryRuntimeResult<Request>> {
    if (this.closed) throw new Error("Memory runtime host is closed.");
    const parsedRequest = MemoryRuntimeRequestSchema.parse(request);
    const connection = await this.connectPrivate();
    const response = await connection.callTool({
      name: MEMORY_RUNTIME_TOOL_NAME,
      arguments: parsedRequest,
    });
    const decoded = exactStructuredContent(response);
    const encoded = JSON.stringify(decoded);
    if (Buffer.byteLength(encoded, "utf8") > MEMORY_RUNTIME_MAX_RESPONSE_BYTES) {
      throw new Error("Memory runtime response exceeds the allowed size.");
    }
    const parsedResponse = MemoryRuntimeToolResponseSchema.safeParse(decoded);
    if (!parsedResponse.success) {
      throw new Error("Memory runtime returned an invalid response.");
    }
    if (parsedResponse.data.operation !== parsedRequest.operation) {
      throw new Error("Memory runtime response operation mismatch.");
    }
    if (!parsedResponse.data.ok) {
      throw new Error(`Memory runtime ${parsedResponse.data.error.code} failed.`);
    }
    if (response.isError) {
      throw new Error("Memory runtime returned a contradictory success response.");
    }
    return parsedResponse.data.result as MemoryRuntimeResult<Request>;
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    const connection = this.connection;
    this.connection = undefined;
    if (connection) await connection.close();
  }

  private async connectPrivate(): Promise<MemoryRuntimeConnection> {
    if (!this.connectionPromise) {
      this.connectionPromise = this.connectRuntime(this.launch)
        .then(async (connection) => {
          if (this.closed) {
            void connection.close();
            throw new Error("Memory runtime host is closed.");
          }
          const server = connection.serverInfo();
          const tools = await connection.listTools();
          if (
            server?.name !== MEMORY_RUNTIME_SERVER_NAME ||
            server.version !== this.launch.runtimeVersion ||
            tools.length !== 1 ||
            tools[0]?.name !== MEMORY_RUNTIME_TOOL_NAME
          ) {
            await connection.close();
            throw new Error("Unexpected private Memory MCP server surface.");
          }
          this.connection = connection;
          return connection;
        })
        .catch(() => {
          throw new Error("Memory runtime is unavailable.");
        });
    }
    return await this.connectionPromise;
  }
}

async function connectMemoryRuntime(
  launch: MemoryRuntimeLaunchSpec,
): Promise<MemoryRuntimeConnection> {
  const transport = new StdioClientTransport({
    command: launch.program,
    args: launch.args,
    cwd: launch.cwd,
    env: launch.env,
    stderr: "pipe",
  });
  const client = new Client(
    { name: "swarmx-mem-host", version: launch.runtimeVersion },
    { capabilities: {} },
  );
  let closed = false;
  try {
    await client.connect(transport);
  } catch (error) {
    await Promise.allSettled([client.close(), transport.close()]);
    throw error;
  }
  return {
    serverInfo() {
      return client.getServerVersion();
    },
    async listTools() {
      return (await client.listTools()).tools.map(({ name }) => ({ name }));
    },
    async callTool(input) {
      return (await client.callTool(input)) as MemoryRuntimeToolCallResult;
    },
    async close() {
      if (closed) return;
      closed = true;
      await Promise.allSettled([client.close(), transport.close()]);
    },
  };
}

function exactStructuredContent(result: MemoryRuntimeToolCallResult): unknown {
  if (result.content.length !== 1 || result.content[0]?.type !== "text") {
    throw new Error("Memory runtime returned an invalid response.");
  }
  const text = result.content[0].text;
  if (typeof text !== "string") throw new Error("Memory runtime returned an invalid response.");
  let decodedText: unknown;
  try {
    decodedText = JSON.parse(text);
  } catch {
    throw new Error("Memory runtime returned an invalid response.");
  }
  if (
    result.structuredContent === undefined ||
    JSON.stringify(decodedText) !== JSON.stringify(result.structuredContent)
  ) {
    throw new Error("Memory runtime returned contradictory structured content.");
  }
  return result.structuredContent;
}
