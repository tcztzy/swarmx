import path from "node:path";
import { Readable, Writable } from "node:stream";
import {
  type Agent as AcpAgent,
  AgentSideConnection,
  type AuthenticateRequest,
  type AuthenticateResponse,
  type CancelNotification,
  type CloseSessionRequest,
  type CloseSessionResponse,
  type InitializeRequest,
  type InitializeResponse,
  type ListSessionsRequest,
  type ListSessionsResponse,
  type LoadSessionRequest,
  type LoadSessionResponse,
  type McpServer,
  type NewSessionRequest,
  type NewSessionResponse,
  ndJsonStream,
  type PromptRequest,
  type PromptResponse,
  type ResumeSessionRequest,
  type ResumeSessionResponse,
  type SessionNotification,
  type SessionUpdate,
  type SetSessionModeRequest,
  type SetSessionModeResponse,
} from "@agentclientprotocol/sdk";
import type {
  McpServerConfig,
  MessageChunk,
  SessionData,
  SwarmConfig,
  SwarmNodeConfig,
} from "@swarmx/core";
import {
  appendMessages,
  cancelAcpRequest,
  createSession,
  listSessionSummaries as listSessionsFile,
  loadSession as loadSessionFile,
  RequestCancelledError,
  SWARMX_VERSION,
  Swarm,
  saveSession,
  withAcpRequest,
} from "@swarmx/core";

interface SessionState {
  cwd: string;
  mcpServers: Record<string, McpServerConfig>;
  swarmConfig?: SwarmConfig;
  sessionData: SessionData;
}

export interface SwarmExecutor {
  execute(
    arguments_: Record<string, unknown>,
    context?: Record<string, unknown>,
  ): Promise<MessageChunk[]>;
}

export interface SwarmXAgentOptions {
  createSwarm?: (config: SwarmConfig) => SwarmExecutor;
}

export class SwarmXAgent implements AcpAgent {
  private sessions = new Map<string, SessionState>();
  private conn: AgentSideConnection | null = null;
  private readonly createSwarm: (config: SwarmConfig) => SwarmExecutor;

  constructor(options: SwarmXAgentOptions = {}) {
    this.createSwarm = options.createSwarm ?? ((config) => new Swarm(config));
  }

  setConnection(conn: AgentSideConnection): void {
    this.conn = conn;
  }

  initialize = async (request: InitializeRequest): Promise<InitializeResponse> => {
    return {
      protocolVersion: request.protocolVersion,
      agentCapabilities: {
        loadSession: true,
        promptCapabilities: {
          image: false,
          audio: false,
          embeddedContext: false,
        },
        sessionCapabilities: {
          close: {},
          list: {},
          resume: {},
        },
      },
      agentInfo: {
        name: "swarmx",
        title: "SwarmX Agent Engine",
        version: SWARMX_VERSION,
      },
      authMethods: [],
    };
  };

  newSession = async (request: NewSessionRequest): Promise<NewSessionResponse> => {
    const cwd = normalizeAbsoluteCwd(request.cwd);
    const mcpServers = projectMcpServers(request.mcpServers, cwd);
    const sessionData = createSession("swarmx", "swarmx", undefined, { cwd });
    saveSession(sessionData);

    this.sessions.set(sessionData.id, {
      cwd,
      mcpServers,
      sessionData,
    });
    return { sessionId: sessionData.id };
  };

  loadSession = async (request: LoadSessionRequest): Promise<LoadSessionResponse> => {
    const { sessionId } = request;
    const sessionData = loadSessionFile(sessionId);
    if (!sessionData) {
      throw new Error(`Session ${sessionId} not found`);
    }

    const cwd = bindPersistedCwd(sessionData, request.cwd);
    this.sessions.set(sessionId, {
      cwd,
      mcpServers: projectMcpServers(request.mcpServers, cwd),
      sessionData,
    });

    for (const msg of sessionData.messages) {
      const update = buildSessionUpdate(msg);
      if (!update || !this.conn) continue;
      await this.conn.sessionUpdate({
        sessionId,
        update,
      });
    }

    return {};
  };

  listSessions = async (request: ListSessionsRequest): Promise<ListSessionsResponse> => {
    const cwd = request.cwd ? normalizeAbsoluteCwd(request.cwd) : undefined;
    const sessions = listSessionsFile().filter(
      (session) =>
        typeof session.cwd === "string" &&
        path.isAbsolute(session.cwd) &&
        (!cwd || path.normalize(session.cwd) === cwd),
    );
    return {
      sessions: sessions.map((s) => ({
        sessionId: s.id,
        cwd: path.normalize(s.cwd as string),
        title: s.title,
        updatedAt: s.updatedAt,
      })),
    };
  };

  prompt = async (request: PromptRequest): Promise<PromptResponse> => {
    const conn = this.conn;
    if (!conn) return { stopReason: "end_turn" };

    const session = this.sessions.get(request.sessionId);
    if (!session) return { stopReason: "cancelled" };

    const userText = projectPromptBlocks(request.prompt);
    const requestedSwarmConfig = request.prompt.find(
      (block) => block.type === "text" && block._meta?.swarmConfig,
    )?._meta?.swarmConfig as SwarmConfig | undefined;
    if (requestedSwarmConfig) session.swarmConfig = requestedSwarmConfig;

    if (!userText.trim()) {
      return { stopReason: "end_turn" };
    }

    const echoedMessageId = request.messageId ? { userMessageId: request.messageId } : {};
    try {
      return await withAcpRequest(acpRequestId(request.sessionId), async () => {
        appendSessionMessages(request.sessionId, [
          { role: "user", kind: "message", content: userText },
        ]);
        session.sessionData = requireSession(request.sessionId);

        const config = applySessionRuntime(
          session.swarmConfig ?? defaultSwarmConfig(),
          session.cwd,
          session.mcpServers,
        );
        const swarm = this.createSwarm(config);
        const result = await swarm.execute(
          { messages: session.sessionData.messages },
          { cwd: session.cwd, sessionId: request.sessionId },
        );

        for (const msg of result) {
          const update = buildSessionUpdate(msg);
          if (!update) continue;
          const notification: SessionNotification = {
            sessionId: request.sessionId,
            update,
          };
          await conn.sessionUpdate(notification);
        }

        appendSessionMessages(request.sessionId, result);
        session.sessionData = requireSession(request.sessionId);
        return {
          stopReason: "end_turn",
          ...echoedMessageId,
        };
      });
    } catch (err: unknown) {
      if (err instanceof RequestCancelledError) {
        return {
          stopReason: "cancelled",
          ...echoedMessageId,
        };
      }
      const errorMsg = err instanceof Error ? err.message : String(err);
      await conn.sessionUpdate({
        sessionId: request.sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: {
            type: "text",
            text: `[error] ${errorMsg}`,
            _meta: { agent: "system", status: "error" },
          },
        },
      });
      return {
        stopReason: "refusal",
        ...echoedMessageId,
      };
    }
  };

  cancel = async (params: CancelNotification): Promise<void> => {
    await cancelAcpRequest(acpRequestId(params.sessionId));
  };
  authenticate = async (_request: AuthenticateRequest): Promise<AuthenticateResponse> => {
    throw new Error("not supported");
  };
  setSessionMode = async (_request: SetSessionModeRequest): Promise<SetSessionModeResponse> => {
    return {};
  };
  resumeSession = async (request: ResumeSessionRequest): Promise<ResumeSessionResponse> => {
    const sessionData = loadSessionFile(request.sessionId);
    if (!sessionData) {
      throw new Error(`Session ${request.sessionId} not found`);
    }

    const cwd = bindPersistedCwd(sessionData, request.cwd);
    this.sessions.set(request.sessionId, {
      cwd,
      mcpServers: projectMcpServers(request.mcpServers ?? [], cwd),
      sessionData,
    });

    return {};
  };
  closeSession = async (request: CloseSessionRequest): Promise<CloseSessionResponse> => {
    this.sessions.delete(request.sessionId);
    return {};
  };
}

function defaultSwarmConfig(): SwarmConfig {
  return {
    name: "default",
    root: "agent",
    nodes: {
      agent: {
        kind: "agent",
        agent: {
          name: "agent",
          instructions: "You are a helpful AI assistant.",
        },
      },
    },
    edges: [],
  };
}

function normalizeAbsoluteCwd(cwd: string): string {
  if (!path.isAbsolute(cwd)) {
    throw new Error(`ACP Session working directory must be absolute: ${cwd}`);
  }
  return path.normalize(cwd);
}

function bindPersistedCwd(session: SessionData, requestedCwd: string): string {
  const cwd = normalizeAbsoluteCwd(requestedCwd);
  if (session.cwd) {
    const persistedCwd = normalizeAbsoluteCwd(session.cwd);
    if (persistedCwd !== cwd) {
      throw new Error(
        `ACP Session working directory mismatch: expected ${persistedCwd}, received ${cwd}.`,
      );
    }
    return persistedCwd;
  }
  session.cwd = cwd;
  saveSession(session);
  return cwd;
}

function projectMcpServers(
  servers: readonly McpServer[],
  cwd: string,
): Record<string, McpServerConfig> {
  const projected: Record<string, McpServerConfig> = {};
  for (const server of servers) {
    if (!("command" in server)) {
      throw new Error(
        `Unsupported ACP MCP transport "${server.type}" for "${server.name}". SwarmX ACP currently supports stdio MCP servers only.`,
      );
    }
    const name = server.name.trim();
    if (!name || projected[name]) {
      throw new Error(
        name ? `Duplicate ACP MCP server name: ${name}` : "ACP MCP server name is required.",
      );
    }
    if (!server.command.trim()) {
      throw new Error(`ACP MCP server "${name}" command is required.`);
    }
    const env: Record<string, string> = {};
    for (const variable of server.env) {
      if (Object.hasOwn(env, variable.name)) {
        throw new Error(`Duplicate environment variable "${variable.name}" for ACP MCP "${name}".`);
      }
      env[variable.name] = variable.value;
    }
    projected[name] = {
      type: "stdio",
      command: server.command,
      args: server.args,
      ...(Object.keys(env).length > 0 ? { env } : {}),
      cwd,
    };
  }
  return projected;
}

function projectPromptBlocks(blocks: PromptRequest["prompt"]): string {
  return blocks
    .map((block) => {
      switch (block.type) {
        case "text":
          return block.text;
        case "resource_link": {
          const details = [block.mimeType, block.size === undefined ? undefined : `${block.size} B`]
            .filter((value): value is string => typeof value === "string")
            .join(", ");
          return `[Resource: ${block.name}](${block.uri}${details ? `; ${details}` : ""})`;
        }
        case "resource":
          if ("text" in block.resource) {
            const mime = block.resource.mimeType ? `; ${block.resource.mimeType}` : "";
            return `[Embedded resource: ${block.resource.uri}${mime}]\n${block.resource.text}`;
          }
          return `[Embedded binary resource: ${block.resource.uri}${
            block.resource.mimeType ? `; ${block.resource.mimeType}` : ""
          }; ${block.resource.blob.length} base64 characters]`;
        case "image":
          return `[Image content: ${block.mimeType}; ${block.data.length} base64 characters${
            block.uri ? `; ${block.uri}` : ""
          }]`;
        case "audio":
          return `[Audio content: ${block.mimeType}; ${block.data.length} base64 characters]`;
        default:
          return "";
      }
    })
    .filter((text) => text.length > 0)
    .join("\n");
}

function applySessionRuntime(
  config: SwarmConfig,
  cwd: string,
  mcpServers: Record<string, McpServerConfig>,
): SwarmConfig {
  const applyAgent = (agent: NonNullable<SwarmConfig["queen"]>) => ({
    ...agent,
    process: { ...agent.process, currentDir: cwd },
    mcpServers: { ...agent.mcpServers, ...mcpServers },
  });
  const applyNode = (node: SwarmNodeConfig): SwarmNodeConfig => {
    if (node.kind === "agent") return { ...node, agent: applyAgent(node.agent) };
    if (node.kind === "tool") {
      return {
        ...node,
        tool: {
          ...node.tool,
          mcpServers: { ...node.tool.mcpServers, ...mcpServers },
        },
      };
    }
    return {
      ...node,
      swarm: applySessionRuntime(node.swarm, cwd, mcpServers),
    };
  };

  return {
    ...config,
    mcpServers: { ...config.mcpServers, ...mcpServers },
    ...(config.queen ? { queen: applyAgent(config.queen) } : {}),
    nodes: Object.fromEntries(
      Object.entries(config.nodes).map(([name, node]) => [name, applyNode(node)]),
    ),
  };
}

function appendSessionMessages(sessionId: string, messages: MessageChunk[]): void {
  if (messages.length === 0) return;
  if (!appendMessages(sessionId, messages)) {
    throw new Error(`Session ${sessionId} not found`);
  }
}

function requireSession(sessionId: string): SessionData {
  const session = loadSessionFile(sessionId);
  if (!session) throw new Error(`Session ${sessionId} not found`);
  return session;
}

function acpRequestId(sessionId: string): string {
  return `acp-server:${sessionId}`;
}

function buildSessionUpdate(msg: MessageChunk): SessionUpdate | null {
  const meta: Record<string, unknown> = {};
  if (msg.role && msg.role !== "assistant") meta.role = msg.role;
  if (msg.agent) meta.agent = msg.agent;
  if (msg.swarmEvent) meta.swarmEvent = msg.swarmEvent;
  if (msg.toolName) meta.toolName = msg.toolName;

  switch (msg.kind) {
    case "thinking":
      return {
        sessionUpdate: "agent_thought_chunk",
        content: { type: "text", text: msg.content },
      };
    case "tool_call":
      return {
        sessionUpdate: "tool_call",
        title: msg.toolName ?? "tool",
        toolCallId: msg.render?.invocationId ?? msg.toolName ?? "tool",
        rawInput: tryParseJson(msg.content),
      };
    case "tool_progress":
      return {
        sessionUpdate: "tool_call_update",
        toolCallId: msg.render?.invocationId ?? msg.toolName ?? "tool",
        _meta: {
          terminal_output_delta: {
            data: msg.content,
            terminal_id: msg.render?.invocationId ?? msg.toolName ?? "tool",
          },
        },
      };
    case "tool_result":
      return {
        sessionUpdate: "tool_call_update",
        title: msg.toolName ?? "tool",
        toolCallId: msg.render?.invocationId ?? msg.toolName ?? "tool",
        rawOutput: tryParseJson(msg.content),
        status: "completed",
      };
    case "message":
      return {
        sessionUpdate: msg.role === "user" ? "user_message_chunk" : "agent_message_chunk",
        content: {
          type: "text",
          text: msg.content,
          ...(Object.keys(meta).length > 0 ? { _meta: meta } : {}),
        },
      };
    default:
      return null;
  }
}

function tryParseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export function run(): void {
  const transport = ndJsonStream(Writable.toWeb(process.stdout), Readable.toWeb(process.stdin));

  const agent = new SwarmXAgent();

  const connection = new AgentSideConnection((conn) => {
    agent.setConnection(conn);
    return agent;
  }, transport);

  connection.closed.then(() => process.exit(0));
  process.on("SIGINT", () => process.exit(0));
  process.on("SIGTERM", () => process.exit(0));
}
