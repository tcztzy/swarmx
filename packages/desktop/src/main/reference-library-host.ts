import path from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import {
  type ReferenceLibraryBackend,
  type ReferenceLibraryRequest,
  ReferenceLibraryRequestSchema,
  type ReferenceLibraryResult,
  ReferenceLibraryResultSchema,
} from "@swarmx/core";
import { z } from "zod";

const MAX_REFERENCE_RESPONSE_BYTES = 96 * 1024;
const ReferenceLibraryLaunchSpecSchema = z
  .object({
    pythonPath: z.string().min(1).max(4_096),
    zimPath: z.string().min(1).max(4_096).optional(),
    zotero: z.boolean().optional(),
  })
  .strict()
  .superRefine((value, context) => {
    if (!path.isAbsolute(value.pythonPath)) {
      context.addIssue({ code: "custom", path: ["pythonPath"], message: "must be absolute" });
    }
    if (
      value.zimPath &&
      (!path.isAbsolute(value.zimPath) || path.extname(value.zimPath).toLowerCase() !== ".zim")
    ) {
      context.addIssue({
        code: "custom",
        path: ["zimPath"],
        message: "must be an absolute .zim path",
      });
    }
    if (!value.zimPath && value.zotero !== true) {
      context.addIssue({
        code: "custom",
        message: "at least one Reference source must be configured",
      });
    }
  });

type ReferenceLibraryLaunchSpec = z.infer<typeof ReferenceLibraryLaunchSpecSchema>;

interface ReferenceToolCallResult {
  content: Array<{ type: string; text?: string }>;
  structuredContent?: unknown;
  isError?: boolean;
}

export interface ReferenceLibraryConnection {
  serverInfo(): { name: string; version: string } | undefined;
  listTools(): Promise<Array<{ name: string }>>;
  callTool(input: {
    name: string;
    arguments: Record<string, unknown>;
  }): Promise<ReferenceToolCallResult>;
  close(): Promise<void>;
}

export interface ReferenceLibraryHostOptions extends ReferenceLibraryLaunchSpec {
  connect?: (launch: ReferenceLibraryLaunchSpec) => Promise<ReferenceLibraryConnection>;
}

export class ReferenceLibraryHost implements ReferenceLibraryBackend {
  private readonly launch: ReferenceLibraryLaunchSpec;
  private readonly connectModule: NonNullable<ReferenceLibraryHostOptions["connect"]>;
  private connectionPromise?: Promise<ReferenceLibraryConnection>;
  private connection?: ReferenceLibraryConnection;
  private closed = false;

  constructor(options: ReferenceLibraryHostOptions) {
    this.launch = ReferenceLibraryLaunchSpecSchema.parse({
      pythonPath: options.pythonPath,
      zimPath: options.zimPath,
      zotero: options.zotero,
    });
    this.connectModule = options.connect ?? connectReferenceLibrary;
  }

  async request(request: ReferenceLibraryRequest): Promise<ReferenceLibraryResult> {
    if (this.closed) throw new Error("Reference Library host is closed.");
    const parsedRequest = ReferenceLibraryRequestSchema.parse(request);
    const connection = await this.connectPrivate();
    const response = await connection.callTool({
      name: "swarmx_reference",
      arguments: { request: parsedRequest },
    });
    if (response.isError) throw new Error("Reference Library request failed.");
    const decoded = exactStructuredContent(response);
    if (Buffer.byteLength(JSON.stringify(decoded), "utf8") > MAX_REFERENCE_RESPONSE_BYTES) {
      throw new Error("Reference Library response exceeds the allowed size.");
    }
    const result = ReferenceLibraryResultSchema.parse(decoded);
    if (result.operation !== parsedRequest.operation) {
      throw new Error("Reference Library response operation mismatch.");
    }
    if (
      parsedRequest.source &&
      (result.operation === "status"
        ? result.sources.length !== 1 || result.sources[0]?.id !== parsedRequest.source
        : result.source !== parsedRequest.source)
    ) {
      throw new Error("Reference Library response source mismatch.");
    }
    return result;
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    const connection = this.connection;
    this.connection = undefined;
    if (connection) await connection.close();
  }

  private async connectPrivate(): Promise<ReferenceLibraryConnection> {
    if (!this.connectionPromise) {
      this.connectionPromise = this.connectModule(this.launch)
        .then(async (connection) => {
          const tools = await connection.listTools();
          if (
            this.closed ||
            connection.serverInfo()?.name !== "swarmx-ref" ||
            tools.length !== 1 ||
            tools[0]?.name !== "swarmx_reference"
          ) {
            await connection.close();
            throw new Error("Unexpected private Reference MCP server surface.");
          }
          this.connection = connection;
          return connection;
        })
        .catch(() => {
          throw new Error("Reference Library module is unavailable.");
        });
    }
    return await this.connectionPromise;
  }
}

async function connectReferenceLibrary(
  launch: ReferenceLibraryLaunchSpec,
): Promise<ReferenceLibraryConnection> {
  const args = ["-I", "-B", "-u", "-m", "swarmx.ref.server"];
  if (launch.zimPath) args.push("--zim", launch.zimPath);
  if (launch.zotero) args.push("--zotero");
  args.push("--stdio");
  const transport = new StdioClientTransport({
    command: launch.pythonPath,
    args,
    cwd: path.dirname(launch.zimPath ?? launch.pythonPath),
    env: {
      PATH: "",
      PYTHONDONTWRITEBYTECODE: "1",
      PYTHONUNBUFFERED: "1",
      PYTHONUTF8: "1",
    },
    stderr: "pipe",
  });
  const client = new Client({ name: "swarmx-ref-host", version: "1" }, { capabilities: {} });
  try {
    await client.connect(transport);
  } catch (error) {
    await Promise.allSettled([client.close(), transport.close()]);
    throw error;
  }
  let closed = false;
  return {
    serverInfo: () => client.getServerVersion(),
    async listTools() {
      return (await client.listTools()).tools.map(({ name }) => ({ name }));
    },
    async callTool(input) {
      return (await client.callTool(input)) as ReferenceToolCallResult;
    },
    async close() {
      if (closed) return;
      closed = true;
      await Promise.allSettled([client.close(), transport.close()]);
    },
  };
}

function exactStructuredContent(result: ReferenceToolCallResult): unknown {
  if (result.content.length !== 1 || result.content[0]?.type !== "text") {
    throw new Error("Reference Library returned invalid content.");
  }
  const text = result.content[0].text;
  if (typeof text !== "string" || result.structuredContent === undefined) {
    throw new Error("Reference Library returned invalid content.");
  }
  let decoded: unknown;
  try {
    decoded = JSON.parse(text);
  } catch {
    throw new Error("Reference Library returned invalid content.");
  }
  if (JSON.stringify(decoded) !== JSON.stringify(result.structuredContent)) {
    throw new Error("Reference Library returned contradictory structured content.");
  }
  return decoded;
}
