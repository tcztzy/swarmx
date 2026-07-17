import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { currentRequestSignal, throwIfCurrentRequestCancelled } from "./acp.js";
import type { McpServerConfig } from "./types.js";
import { SWARMX_VERSION } from "./version.js";

interface McpTool {
  name: string;
  description?: string;
  inputSchema: Record<string, unknown>;
  isEnabled?: () => boolean;
  serverName?: string;
  remoteName?: string;
}

export interface LocalMcpTool extends McpTool {
  kind?: "function";
  dispose?: () => Promise<void> | void;
  call(arguments_: Record<string, unknown>): Promise<unknown>;
}

export interface LocalTextTool {
  kind: "text";
  name: string;
  description?: string;
  isEnabled?: () => boolean;
  dispose?: () => Promise<void> | void;
  format?: {
    type: "grammar";
    syntax: "lark" | "regex";
    definition: string;
  };
  call(input: string): Promise<unknown>;
}

export type LocalTool = LocalMcpTool | LocalTextTool;

export interface LocalToolResult {
  content: string;
  structuredContent?: unknown;
  isError?: boolean;
}

export interface ToolExecutionResult {
  content: string;
  structuredContent?: unknown;
  isError: boolean;
}

const LOCAL_TOOL_RESULT = Symbol("swarmx.local-tool-result");

/** Separates model-facing tool text from client-facing structured output. */
export function localToolResult(
  content: string,
  structuredContent?: unknown,
  options: { isError?: boolean } = {},
): LocalToolResult {
  return {
    [LOCAL_TOOL_RESULT]: true,
    content,
    ...(structuredContent === undefined ? {} : { structuredContent }),
    ...(options.isError === undefined ? {} : { isError: options.isError }),
  } as LocalToolResult;
}

export type NativeLocalToolDefinition =
  | {
      type: "function";
      function: {
        name: string;
        description: string;
        parameters: Record<string, unknown>;
      };
    }
  | {
      type: "custom";
      name: string;
      description: string;
      format?: LocalTextTool["format"];
    };

export interface McpConnectionResult {
  client: Client;
  tools: Array<{
    name: string;
    description?: string;
    inputSchema: Record<string, unknown>;
  }>;
  close(): Promise<void>;
}

export interface McpManagerOptions {
  connectServer?: (
    name: string,
    config: McpServerConfig,
    signal: AbortSignal,
  ) => Promise<McpConnectionResult>;
  waitTimeoutMs?: number;
}

type McpServerState = "pending" | "connected" | "failed" | "needsAuth" | "disabled";

interface McpServerRecord {
  name: string;
  state: McpServerState;
  controller: AbortController;
  task: Promise<void>;
  error?: unknown;
  close?: () => Promise<void>;
}

const CLAUDE_MCP_WAIT_MS = 5_000;
const CLAUDE_TOOL_SEARCH_MAX_RESULTS = 20;

export class McpManager {
  private clients: Map<string, Client> = new Map();
  private tools: McpTool[] = [];
  private localTools = new Map<string, LocalMcpTool>();
  private localTextTools = new Map<string, LocalTextTool>();
  private localDisposers = new Set<() => Promise<void> | void>();
  private serverRecords = new Map<string, McpServerRecord>();
  private activatedMcpTools = new Set<string>();
  private claudeMcpDiscoveryEnabled = false;
  private claudeMcpResourcesEnabled = false;
  private closed = false;
  private readonly connectServer: NonNullable<McpManagerOptions["connectServer"]>;
  private readonly waitTimeoutMs: number;

  constructor(options: McpManagerOptions = {}) {
    this.connectServer = options.connectServer ?? connectMcpServer;
    this.waitTimeoutMs = options.waitTimeoutMs ?? CLAUDE_MCP_WAIT_MS;
  }

  addLocalTools(tools: readonly LocalTool[]): void {
    for (const tool of tools) {
      if (
        this.localTools.has(tool.name) ||
        this.localTextTools.has(tool.name) ||
        this.tools.some((candidate) => candidate.name === tool.name)
      ) {
        throw new Error(`Duplicate MCP tool name: ${tool.name}`);
      }
      if (tool.kind === "text") {
        this.localTextTools.set(tool.name, tool);
        if (tool.dispose) this.localDisposers.add(tool.dispose);
        continue;
      }
      this.localTools.set(tool.name, tool);
      if (tool.dispose) this.localDisposers.add(tool.dispose);
      this.tools.push({
        name: tool.name,
        description: tool.description,
        inputSchema: tool.inputSchema,
        ...(tool.isEnabled ? { isEnabled: tool.isEnabled } : {}),
      });
    }
  }

  addClaudeMcpResourceTools(): void {
    if (this.claudeMcpResourcesEnabled) return;
    this.claudeMcpResourcesEnabled = true;
    this.addLocalTools([
      {
        name: "ListMcpResourcesTool",
        description: "Lists resources available from connected MCP servers.",
        isEnabled: () => this.clients.size > 0,
        inputSchema: {
          type: "object",
          properties: {
            server: { type: "string", description: "Optional MCP server name to filter by." },
          },
        },
        call: async (input) => {
          const server = optionalMcpResourceText(input.server, "server");
          const resources = await this.listClaudeMcpResources(server);
          return localToolResult(
            resources.length === 0
              ? "No MCP resources found."
              : resources
                  .map((resource) => `${resource.server}: ${resource.uri} (${resource.name})`)
                  .join("\n"),
            resources,
          );
        },
      },
      {
        name: "ReadMcpResourceTool",
        description: "Reads a text resource from a connected MCP server.",
        isEnabled: () => this.clients.size > 0,
        inputSchema: {
          type: "object",
          properties: {
            server: { type: "string", description: "The MCP server name." },
            uri: { type: "string", description: "The resource URI to read." },
          },
          required: ["server", "uri"],
        },
        call: async (input) => {
          const result = await this.readClaudeMcpResource(
            requiredMcpResourceText(input.server, "server"),
            requiredMcpResourceText(input.uri, "uri"),
          );
          return localToolResult(
            result.error ??
              result.contents
                .map((content) => content.text)
                .filter(Boolean)
                .join("\n") ??
              "",
            result,
            { isError: result.error !== undefined },
          );
        },
      },
    ]);
  }

  addClaudeMcpDiscoveryTools(): void {
    if (this.claudeMcpDiscoveryEnabled || this.serverRecords.size === 0) return;
    this.claudeMcpDiscoveryEnabled = true;
    this.addLocalTools([
      {
        name: "ToolSearch",
        description:
          "Fetches full schema definitions for deferred tools so they can be called. Use select:<tool_name> for direct selection, or keywords to search.",
        isEnabled: () => this.deferredMcpTools().length > 0 || this.pendingServerNames().length > 0,
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description:
                'Query to find deferred tools. Use "select:<tool_name>" for direct selection, or keywords to search.',
            },
            max_results: {
              type: "number",
              default: 5,
              description: "Maximum number of results to return (default: 5)",
            },
          },
          required: ["query"],
        },
        call: async (input) => this.searchClaudeMcpTools(input),
      },
      {
        name: "WaitForMcpServers",
        description: [
          "Wait for MCP servers that are still connecting and whose tools are not",
          "yet in your tool list. Pass `servers` to wait for specific ones, or omit",
          "it to wait for all pending servers.",
          "",
          "If the user's request needs tools from a still-connecting server, call this",
          "tool to wait for it. Once it connects, its tools will be added to your tool",
          "list and you can use them directly. Returns ready=true when servers are",
          "ready, ready=false if they failed to connect, need authentication, or are",
          "disabled.",
          "",
          "You do not need to ask the user for confirmation to use this tool.",
        ].join("\n"),
        isEnabled: () => this.pendingServerNames().length > 0,
        inputSchema: {
          type: "object",
          properties: {
            servers: {
              type: "array",
              items: { type: "string" },
              description: "Server names to wait for (default: all pending)",
            },
          },
        },
        call: async (input) => this.waitForClaudeMcpServers(input),
      },
    ]);
  }

  private async listClaudeMcpResources(serverFilter?: string): Promise<
    Array<{
      uri: string;
      name: string;
      mimeType?: string;
      description?: string;
      server: string;
    }>
  > {
    if (serverFilter && !this.clients.has(serverFilter)) {
      throw new Error(`MCP server ${serverFilter} is not connected.`);
    }
    const clients = serverFilter
      ? ([[serverFilter, this.clients.get(serverFilter)]] as const)
      : [...this.clients.entries()];
    const resources: Array<{
      uri: string;
      name: string;
      mimeType?: string;
      description?: string;
      server: string;
    }> = [];
    for (const [server, client] of clients) {
      if (!client) continue;
      let cursor: string | undefined;
      const seenCursors = new Set<string>();
      try {
        do {
          const signal = currentRequestSignal();
          const page = await client.listResources(
            cursor === undefined ? undefined : { cursor },
            signal ? { signal } : undefined,
          );
          throwIfCurrentRequestCancelled();
          resources.push(
            ...page.resources.map((resource) => ({
              uri: resource.uri,
              name: resource.name,
              ...(resource.mimeType === undefined ? {} : { mimeType: resource.mimeType }),
              ...(resource.description === undefined ? {} : { description: resource.description }),
              server,
            })),
          );
          if (resources.length > 10_000) {
            throw new Error("MCP resource listing exceeded the 10000-item safety limit.");
          }
          cursor = page.nextCursor;
          if (cursor && seenCursors.has(cursor)) {
            throw new Error(`MCP server ${server} repeated resource cursor ${cursor}.`);
          }
          if (cursor) seenCursors.add(cursor);
        } while (cursor);
      } catch (error) {
        throwIfCurrentRequestCancelled();
        if (serverFilter) throw error;
      }
    }
    return resources;
  }

  private async readClaudeMcpResource(
    server: string,
    uri: string,
  ): Promise<{
    contents: Array<{ uri: string; mimeType?: string; text?: string }>;
    error?: string;
  }> {
    const client = this.clients.get(server);
    if (!client) throw new Error(`MCP server ${server} is not connected.`);
    try {
      const signal = currentRequestSignal();
      const resource = await client.readResource({ uri }, signal ? { signal } : undefined);
      throwIfCurrentRequestCancelled();
      const textContents = resource.contents.flatMap((content) =>
        "text" in content
          ? [
              {
                uri: content.uri,
                ...(content.mimeType === undefined ? {} : { mimeType: content.mimeType }),
                text: content.text,
              },
            ]
          : [],
      );
      const binaryCount = resource.contents.length - textContents.length;
      return {
        contents: textContents,
        ...(binaryCount === 0
          ? {}
          : {
              error: `${binaryCount} binary MCP resource ${binaryCount === 1 ? "item is" : "items are"} unsupported because no authorized file sink is configured.`,
            }),
      };
    } catch (error) {
      throwIfCurrentRequestCancelled();
      throw error;
    }
  }

  startServer(name: string, config: McpServerConfig): void {
    if (this.closed) throw new Error("MCP manager is closed.");
    if (this.serverRecords.has(name)) return;
    if (!name.trim()) throw new Error("MCP server name must be non-empty.");

    const controller = new AbortController();
    const requestSignal = currentRequestSignal();
    const abortFromRequest = (): void => controller.abort(requestSignal?.reason);
    requestSignal?.addEventListener("abort", abortFromRequest, { once: true });

    const record: McpServerRecord = {
      name,
      state: "pending",
      controller,
      task: Promise.resolve(),
    };
    this.serverRecords.set(name, record);
    record.task = this.connectServer(name, config, controller.signal)
      .then(async (connection) => {
        try {
          if (this.closed || controller.signal.aborted) {
            await connection.close();
            return;
          }
          const projected = connection.tools.map((tool) => ({
            name: claudeMcpToolName(name, tool.name),
            description: tool.description,
            inputSchema: tool.inputSchema,
            serverName: name,
            remoteName: tool.name,
          }));
          this.assertMcpToolNamesAvailable(projected);
          this.clients.set(name, connection.client);
          record.close = connection.close;
          this.tools.push(...projected);
          record.state = "connected";
        } catch (error) {
          await connection.close();
          throw error;
        }
      })
      .catch((error: unknown) => {
        record.error = error;
        record.state = "failed";
      })
      .finally(() => {
        requestSignal?.removeEventListener("abort", abortFromRequest);
      });
  }

  async addServer(name: string, config: McpServerConfig): Promise<void> {
    this.startServer(name, config);
    const record = this.serverRecords.get(name);
    if (!record) throw new Error(`MCP server ${name} was not started.`);
    await record.task;
    throwIfCurrentRequestCancelled();
    if (record.state !== "connected") {
      throw record.error instanceof Error
        ? record.error
        : new Error(`Failed to connect MCP server ${name}.`);
    }
  }

  private async searchClaudeMcpTools(input: Record<string, unknown>): Promise<LocalToolResult> {
    const query = requiredMcpResourceText(input.query, "query");
    if (query.length > 200) throw new Error("query must contain at most 200 characters.");
    const maxResults = toolSearchMaxResults(input.max_results);
    const candidates = this.deferredMcpTools();
    let matches: string[];

    const selection = query.match(/^select:(.+)$/i);
    if (selection) {
      const requested = selection[1]
        .split(",")
        .map((name) => name.trim())
        .filter(Boolean);
      matches = requested.flatMap((name) => {
        const match = candidates.find((tool) => tool.name.toLowerCase() === name.toLowerCase());
        return match ? [match.name] : [];
      });
      matches = [...new Set(matches)];
    } else {
      matches = searchMcpTools(candidates, query, maxResults);
    }

    for (const name of matches) this.activatedMcpTools.add(name);
    const pending = matches.length === 0 ? this.pendingServerNames() : [];
    const result = {
      matches,
      query,
      total_deferred_tools: candidates.length,
      ...(pending.length > 0 ? { pending_mcp_servers: pending } : {}),
    };
    return localToolResult(
      matches.length === 0
        ? pending.length > 0
          ? `No matching deferred tools found. Some MCP servers are still connecting: ${pending.join(", ")}. Their tools will become available shortly - try searching again.`
          : "No matching deferred tools found"
        : `Loaded deferred tools: ${matches.join(", ")}`,
      result,
    );
  }

  private async waitForClaudeMcpServers(input: Record<string, unknown>): Promise<LocalToolResult> {
    const requested = mcpServerNames(input.servers) ?? this.pendingServerNames();
    const records = requested.flatMap((name) => {
      const record = this.serverRecord(name);
      return record ? [record] : [];
    });
    const pendingTasks = records
      .filter((record) => record.state === "pending")
      .map((record) => record.task);
    if (pendingTasks.length > 0) {
      await settleUntil(pendingTasks, this.waitTimeoutMs, currentRequestSignal());
      throwIfCurrentRequestCancelled();
    }

    const connected: string[] = [];
    const failed: string[] = [];
    const stillPending: string[] = [];
    const needsAuth: string[] = [];
    const disabled: string[] = [];
    const unknown: string[] = [];
    for (const name of requested) {
      const record = this.serverRecord(name);
      if (!record) {
        unknown.push(name);
        continue;
      }
      if (record.state === "connected") connected.push(record.name);
      else if (record.state === "failed") failed.push(record.name);
      else if (record.state === "pending") stillPending.push(record.name);
      else if (record.state === "needsAuth") needsAuth.push(record.name);
      else disabled.push(record.name);
    }
    for (const serverName of connected) this.activateMcpServerTools(serverName);
    const result = {
      ready:
        failed.length === 0 &&
        stillPending.length === 0 &&
        needsAuth.length === 0 &&
        disabled.length === 0 &&
        unknown.length === 0,
      connected,
      failed,
      stillPending,
      needsAuth,
      disabled,
      unknown,
    };
    return localToolResult(claudeMcpWaitText(result), result);
  }

  private deferredMcpTools(): McpTool[] {
    return this.tools.filter((tool) => tool.serverName !== undefined);
  }

  private pendingServerNames(): string[] {
    return [...this.serverRecords.values()]
      .filter((record) => record.state === "pending")
      .map((record) => record.name);
  }

  private serverRecord(name: string): McpServerRecord | undefined {
    const normalized = name.toLowerCase();
    return [...this.serverRecords.values()].find(
      (record) => record.name.toLowerCase() === normalized,
    );
  }

  private activateMcpServerTools(serverName: string): void {
    for (const tool of this.tools) {
      if (tool.serverName === serverName) this.activatedMcpTools.add(tool.name);
    }
  }

  private assertMcpToolNamesAvailable(tools: readonly McpTool[]): void {
    const names = new Set<string>();
    for (const tool of tools) {
      if (
        names.has(tool.name) ||
        this.localTools.has(tool.name) ||
        this.localTextTools.has(tool.name) ||
        this.tools.some((candidate) => candidate.name === tool.name)
      ) {
        throw new Error(`Duplicate MCP tool name: ${tool.name}`);
      }
      names.add(tool.name);
    }
  }

  private isToolVisible(tool: McpTool): boolean {
    if (tool.isEnabled?.() === false) return false;
    return (
      tool.serverName === undefined ||
      !this.claudeMcpDiscoveryEnabled ||
      this.activatedMcpTools.has(tool.name)
    );
  }

  async callTool(
    name: string,
    arguments_: Record<string, unknown> | string,
    _context?: Record<string, unknown>,
  ): Promise<ToolExecutionResult> {
    const localTextTool = this.localTextTools.get(name);
    if (localTextTool) {
      if (localTextTool.isEnabled?.() === false) {
        throw new Error(`Tool ${name} is not currently available.`);
      }
      if (typeof arguments_ !== "string") {
        throw new Error(`Text tool ${name} requires freeform string input`);
      }
      return executionResult(await localTextTool.call(arguments_));
    }

    const localTool = this.localTools.get(name);
    if (localTool) {
      if (localTool.isEnabled?.() === false) {
        throw new Error(`Tool ${name} is not currently available.`);
      }
      if (typeof arguments_ === "string") {
        throw new Error(`Function tool ${name} requires object input`);
      }
      const result = await localTool.call(arguments_);
      return executionResult(result);
    }

    if (typeof arguments_ === "string") {
      throw new Error(`Tool ${name} does not accept freeform string input`);
    }

    const tool = this.tools.find((candidate) => candidate.name === name && candidate.serverName);
    if (!tool?.serverName || !tool.remoteName) {
      throw new Error(`Tool ${name} not found.`);
    }
    if (this.claudeMcpDiscoveryEnabled && !this.activatedMcpTools.has(name)) {
      throw new Error(`Tool ${name} is deferred. Load it with ToolSearch first.`);
    }
    const client = this.clients.get(tool.serverName);
    if (!client) throw new Error(`MCP server ${tool.serverName} is not connected.`);
    try {
      const signal = currentRequestSignal();
      const result = await client.callTool(
        {
          name: tool.remoteName,
          arguments: arguments_,
        },
        undefined,
        signal ? { signal } : undefined,
      );
      const structuredContent =
        "structuredContent" in result && result.structuredContent !== undefined
          ? result.structuredContent
          : { result: result.content };
      return {
        content: mcpModelContent(result.content),
        structuredContent,
        isError: result.isError === true,
      };
    } catch (error) {
      throwIfCurrentRequestCancelled();
      throw error;
    }
  }

  toolsForOpenai(): Array<{
    type: "function";
    function: {
      name: string;
      description: string;
      parameters: Record<string, unknown>;
    };
  }> {
    return this.tools
      .filter((tool) => this.isToolVisible(tool))
      .map((tool) => ({
        type: "function" as const,
        function: {
          name: tool.name,
          description: tool.description ?? "",
          parameters: tool.inputSchema,
        },
      }));
  }

  toolsForNative(): NativeLocalToolDefinition[] {
    return [
      ...this.toolsForOpenai(),
      ...[...this.localTextTools.values()]
        .filter((tool) => tool.isEnabled?.() !== false)
        .map((tool) => ({
          type: "custom" as const,
          name: tool.name,
          description: tool.description ?? "",
          ...(tool.format ? { format: tool.format } : {}),
        })),
    ];
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    const records = [...this.serverRecords.values()];
    for (const record of records) record.controller.abort(new Error("MCP manager closed."));
    await Promise.allSettled(records.map((record) => record.task));
    const recordClientNames = new Set(
      records.filter((record) => record.close).map((record) => record.name),
    );
    const closes = records.flatMap((record) => (record.close ? [record.close()] : []));
    for (const [name, client] of this.clients) {
      if (!recordClientNames.has(name)) closes.push(client.close());
    }
    const disposers = [...this.localDisposers];
    this.clients.clear();
    this.localTools.clear();
    this.localTextTools.clear();
    this.localDisposers.clear();
    this.serverRecords.clear();
    this.activatedMcpTools.clear();
    this.tools = [];
    await Promise.allSettled([
      ...closes,
      ...disposers.map((dispose) => Promise.resolve().then(dispose)),
    ]);
  }
}

async function connectMcpServer(
  _name: string,
  config: McpServerConfig,
  signal: AbortSignal,
): Promise<McpConnectionResult> {
  if (config.type !== "stdio" || !config.command) {
    throw new Error(
      `Unsupported MCP transport: ${config.type}. Currently only "stdio" is supported.`,
    );
  }
  if (signal.aborted) throw abortReason(signal);

  const transport = new StdioClientTransport({
    command: config.command,
    args: config.args ?? [],
    env: config.env,
  });
  const client = new Client({ name: "swarmx", version: SWARMX_VERSION }, { capabilities: {} });
  let closed = false;
  const close = async (): Promise<void> => {
    if (closed) return;
    closed = true;
    await Promise.allSettled([client.close(), transport.close()]);
  };
  const abortConnection = (): void => {
    void close();
  };
  signal.addEventListener("abort", abortConnection, { once: true });
  try {
    await client.connect(transport);
    if (signal.aborted) throw abortReason(signal);
    const response = await client.listTools(undefined, { signal });
    if (signal.aborted) throw abortReason(signal);
    return {
      client,
      tools: response.tools.map((tool) => ({
        name: tool.name,
        description: tool.description,
        inputSchema: tool.inputSchema as Record<string, unknown>,
      })),
      close,
    };
  } catch (error) {
    await close();
    throw error;
  } finally {
    signal.removeEventListener("abort", abortConnection);
  }
}

function claudeMcpToolName(serverName: string, toolName: string): string {
  return `mcp__${mcpToolNamePart(serverName, "server")}__${mcpToolNamePart(toolName, "tool")}`;
}

function mcpToolNamePart(value: string, kind: string): string {
  const name = value.trim().replace(/[^A-Za-z0-9_-]/g, "_");
  if (!name) throw new Error(`MCP ${kind} name cannot be projected as a model tool name.`);
  return name;
}

function toolSearchMaxResults(value: unknown): number {
  if (value === undefined) return 5;
  if (
    typeof value !== "number" ||
    !Number.isInteger(value) ||
    value < 1 ||
    value > CLAUDE_TOOL_SEARCH_MAX_RESULTS
  ) {
    throw new Error(`max_results must be an integer from 1 to ${CLAUDE_TOOL_SEARCH_MAX_RESULTS}.`);
  }
  return value;
}

function searchMcpTools(tools: readonly McpTool[], query: string, maxResults: number): string[] {
  const tokens = query
    .toLowerCase()
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean);
  const required = tokens
    .filter((token) => token.startsWith("+") && token.length > 1)
    .map((token) => token.slice(1));
  const terms = tokens.map((token) => (token.startsWith("+") ? token.slice(1) : token));
  return tools
    .flatMap((tool) => {
      const name = tool.name.toLowerCase();
      const document =
        `${name} ${tool.description ?? ""} ${JSON.stringify(tool.inputSchema)}`.toLowerCase();
      if (required.some((term) => !document.includes(term))) return [];
      let score = 0;
      for (const term of terms) {
        if (!term) continue;
        if (name === term) score += 100;
        else if (name.split(/[_-]+/).includes(term)) score += 12;
        else if (name.includes(term)) score += 6;
        if (document.includes(term)) score += 2;
      }
      return score > 0 ? [{ name: tool.name, score }] : [];
    })
    .sort((left, right) => right.score - left.score || left.name.localeCompare(right.name))
    .slice(0, maxResults)
    .map((match) => match.name);
}

function mcpServerNames(value: unknown): string[] | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value)) throw new Error("servers must be an array of server names.");
  const names: string[] = [];
  const seen = new Set<string>();
  for (const item of value) {
    if (typeof item !== "string" || !item.trim()) {
      throw new Error("servers must contain only non-empty server names.");
    }
    const name = item.trim();
    const normalized = name.toLowerCase();
    if (seen.has(normalized)) continue;
    seen.add(normalized);
    names.push(name);
  }
  return names;
}

async function settleUntil(
  tasks: readonly Promise<void>[],
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<void> {
  if (tasks.length === 0) return;
  await new Promise<void>((resolve, reject) => {
    let settled = false;
    const finish = (error?: unknown): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      if (error === undefined) resolve();
      else reject(error);
    };
    const abort = (): void => finish(abortReason(signal));
    const timer = setTimeout(() => finish(), timeoutMs);
    if (signal?.aborted) {
      finish(abortReason(signal));
      return;
    }
    signal?.addEventListener("abort", abort, { once: true });
    void Promise.allSettled(tasks).then(() => finish());
  });
}

function abortReason(signal?: AbortSignal): unknown {
  return signal?.reason instanceof Error ? signal.reason : new Error("Operation aborted.");
}

function claudeMcpWaitText(result: {
  ready: boolean;
  connected: string[];
  failed: string[];
  stillPending: string[];
  needsAuth: string[];
  disabled: string[];
  unknown: string[];
}): string {
  return [
    `ready: ${result.ready}`,
    result.connected.length > 0
      ? `Connected (their tools are now available - call them directly): ${result.connected.join(", ")}`
      : "",
    result.failed.length > 0 ? `Failed to connect: ${result.failed.join(", ")}` : "",
    result.stillPending.length > 0
      ? `Still connecting (try again or proceed without): ${result.stillPending.join(", ")}`
      : "",
    result.needsAuth.length > 0
      ? `Needs authentication (ask the user to configure MCP): ${result.needsAuth.join(", ")}`
      : "",
    result.disabled.length > 0
      ? `Disabled (ask the user to enable MCP): ${result.disabled.join(", ")}`
      : "",
    result.unknown.length > 0
      ? `Unknown (no MCP server with this name is configured): ${result.unknown.join(", ")}`
      : "",
  ]
    .filter(Boolean)
    .join("\n");
}

function executionResult(result: unknown): ToolExecutionResult {
  if (isLocalToolResult(result)) {
    return {
      content: result.content,
      ...(result.structuredContent === undefined
        ? {}
        : { structuredContent: result.structuredContent }),
      isError: result.isError === true,
    };
  }
  const structuredContent = recordResult(result);
  return { content: JSON.stringify(structuredContent), structuredContent, isError: false };
}

function isLocalToolResult(result: unknown): result is LocalToolResult {
  return (
    result !== null &&
    typeof result === "object" &&
    !Array.isArray(result) &&
    (result as Record<PropertyKey, unknown>)[LOCAL_TOOL_RESULT] === true &&
    typeof (result as { content?: unknown }).content === "string"
  );
}

function recordResult(result: unknown): Record<string, unknown> {
  return result !== null && typeof result === "object" && !Array.isArray(result)
    ? (result as Record<string, unknown>)
    : { result };
}

function requiredMcpResourceText(value: unknown, name: string): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${name} must be non-empty text.`);
  }
  return value;
}

function optionalMcpResourceText(value: unknown, name: string): string | undefined {
  if (value === undefined) return undefined;
  return requiredMcpResourceText(value, name);
}

function mcpModelContent(content: unknown): string {
  if (!Array.isArray(content)) return JSON.stringify(content) ?? String(content);
  const parts = content.flatMap((item): string[] => {
    if (item && typeof item === "object" && !Array.isArray(item)) {
      const record = item as Record<string, unknown>;
      if (record.type === "text" && typeof record.text === "string") return [record.text];
    }
    return [JSON.stringify(item) ?? String(item)];
  });
  return parts.join("\n");
}
