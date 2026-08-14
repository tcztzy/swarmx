import { createHash, randomUUID } from "node:crypto";
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
  AuditInput,
  AuditStore as AuditStoreType,
  McpServerConfig,
  MessageChunk,
  SessionData,
  SwarmConfig,
  SwarmNodeConfig,
} from "@swarmx/core";
import {
  AuditStore,
  appendMessages,
  createSession,
  listSessionSummaries as listSessionsFile,
  loadSession as loadSessionFile,
  SWARMX_VERSION,
  Swarm,
  saveSession,
} from "@swarmx/core";
import { cancelRequest, RequestCancelledError, withRequestScope } from "@swarmx/core/request-scope";

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
  audit?: Pick<AuditStoreType, "append">;
}

export class SwarmXAgent implements AcpAgent {
  private sessions = new Map<string, SessionState>();
  private conn: AgentSideConnection | null = null;
  private readonly createSwarm: (config: SwarmConfig) => SwarmExecutor;
  private readonly audit: Pick<AuditStoreType, "append"> | undefined;
  private readonly activePromptRequestIds = new Map<string, string>();

  constructor(options: SwarmXAgentOptions = {}) {
    this.createSwarm = options.createSwarm ?? ((config) => new Swarm(config));
    this.audit = options.audit;
  }

  setConnection(conn: AgentSideConnection): void {
    this.conn = conn;
  }

  initialize = async (request: InitializeRequest): Promise<InitializeResponse> => {
    const requestId = newAuditRequestId();
    this.recordAudit({
      category: "system",
      action: "acp.initialize",
      outcome: "attempted",
      requestId,
      metadata: {},
    });
    const response: InitializeResponse = {
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
    this.recordAudit({
      category: "system",
      action: "acp.initialize",
      outcome: "completed",
      requestId,
      metadata: {},
    });
    return response;
  };

  newSession = async (request: NewSessionRequest): Promise<NewSessionResponse> => {
    const requestId = newAuditRequestId();
    this.recordAudit({
      category: "session",
      action: "acp.session.new",
      outcome: "attempted",
      requestId,
      metadata: {},
    });

    let sessionData: SessionData;
    let cwd: string;
    let mcpServers: Record<string, McpServerConfig>;
    try {
      cwd = normalizeAbsoluteCwd(request.cwd);
      mcpServers = projectMcpServers(request.mcpServers, cwd);
      sessionData = createSession("swarmx", "swarmx", undefined, { cwd });
      saveSession(sessionData);

      this.sessions.set(sessionData.id, {
        cwd,
        mcpServers,
        sessionData,
      });
    } catch (error) {
      this.recordAudit({
        category: "session",
        action: "acp.session.new",
        outcome: "failed",
        requestId,
        metadata: {},
      });
      throw error;
    }

    this.recordAudit({
      category: "session",
      action: "acp.session.new",
      outcome: "completed",
      requestId,
      sessionId: auditSessionId(sessionData.id),
      target: auditSessionTarget(sessionData.id),
      metadata: {},
    });
    return { sessionId: sessionData.id };
  };

  loadSession = async (request: LoadSessionRequest): Promise<LoadSessionResponse> => {
    const { sessionId } = request;
    const auditedSessionId = auditSessionId(sessionId);
    const requestId = newAuditRequestId();
    this.recordAudit({
      category: "session",
      action: "acp.session.load",
      outcome: "attempted",
      requestId,
      sessionId: auditedSessionId,
      target: auditSessionTarget(sessionId),
      metadata: {},
    });
    const sessionData = loadSessionFile(sessionId);
    if (!sessionData) {
      this.recordAudit({
        category: "session",
        action: "acp.session.load",
        outcome: "denied",
        requestId,
        sessionId: auditedSessionId,
        target: auditSessionTarget(sessionId),
        metadata: { reason: "not_found" },
      });
      throw new Error(`Session ${sessionId} not found`);
    }

    try {
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
    } catch (error) {
      this.recordAudit({
        category: "session",
        action: "acp.session.load",
        outcome: "failed",
        requestId,
        sessionId: auditedSessionId,
        target: auditSessionTarget(sessionId),
        metadata: {},
      });
      throw error;
    }

    this.recordAudit({
      category: "session",
      action: "acp.session.load",
      outcome: "completed",
      requestId,
      sessionId: auditedSessionId,
      target: auditSessionTarget(sessionId),
      metadata: {},
    });
    return {};
  };

  listSessions = async (request: ListSessionsRequest): Promise<ListSessionsResponse> => {
    const requestId = newAuditRequestId();
    const audit = {
      category: "session" as const,
      action: "acp.session.list",
      requestId,
      metadata: { cwdScoped: Boolean(request.cwd) },
    };
    this.recordAudit({ ...audit, outcome: "attempted" });
    try {
      const cwd = request.cwd ? normalizeAbsoluteCwd(request.cwd) : undefined;
      const sessions = listSessionsFile().filter(
        (session) =>
          typeof session.cwd === "string" &&
          path.isAbsolute(session.cwd) &&
          (!cwd || path.normalize(session.cwd) === cwd),
      );
      const response = {
        sessions: sessions.map((s) => ({
          sessionId: s.id,
          cwd: path.normalize(s.cwd as string),
          title: s.title,
          updatedAt: s.updatedAt,
        })),
      };
      this.recordAudit({
        ...audit,
        outcome: "completed",
        metadata: { ...audit.metadata, sessionCount: response.sessions.length },
      });
      return response;
    } catch (error) {
      this.recordAudit({ ...audit, outcome: "failed" });
      throw error;
    }
  };

  prompt = async (request: PromptRequest): Promise<PromptResponse> => {
    const auditedSessionId = auditSessionId(request.sessionId);
    const requestId = newAuditRequestId();
    this.recordAudit({
      category: "session",
      action: "acp.prompt",
      outcome: "attempted",
      requestId,
      sessionId: auditedSessionId,
      target: auditSessionTarget(request.sessionId),
      metadata: {},
    });

    const conn = this.conn;
    if (!conn) {
      this.recordAudit({
        category: "session",
        action: "acp.prompt",
        outcome: "denied",
        requestId,
        sessionId: auditedSessionId,
        target: auditSessionTarget(request.sessionId),
        metadata: { reason: "connection_unavailable" },
      });
      return { stopReason: "end_turn" };
    }

    const session = this.sessions.get(request.sessionId);
    if (!session) {
      this.recordAudit({
        category: "session",
        action: "acp.prompt",
        outcome: "denied",
        requestId,
        sessionId: auditedSessionId,
        target: auditSessionTarget(request.sessionId),
        metadata: { reason: "session_unavailable" },
      });
      return { stopReason: "cancelled" };
    }

    const userText = projectPromptBlocks(request.prompt);
    const requestedSwarmConfig = request.prompt.find(
      (block) => block.type === "text" && block._meta?.swarmConfig,
    )?._meta?.swarmConfig as SwarmConfig | undefined;
    if (requestedSwarmConfig) session.swarmConfig = requestedSwarmConfig;

    if (!userText.trim()) {
      this.recordAudit({
        category: "session",
        action: "acp.prompt",
        outcome: "completed",
        requestId,
        sessionId: auditedSessionId,
        target: auditSessionTarget(request.sessionId),
        metadata: { skipped: true },
      });
      return { stopReason: "end_turn" };
    }

    this.activePromptRequestIds.set(request.sessionId, requestId);
    try {
      await withRequestScope(acpRequestId(request.sessionId), async () => {
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
      });
    } catch (err: unknown) {
      if (err instanceof RequestCancelledError) {
        this.recordAudit({
          category: "session",
          action: "acp.prompt",
          outcome: "cancelled",
          requestId,
          sessionId: auditedSessionId,
          target: auditSessionTarget(request.sessionId),
          metadata: {},
        });
        return {
          stopReason: "cancelled",
        };
      }
      this.recordAudit({
        category: "session",
        action: "acp.prompt",
        outcome: "failed",
        requestId,
        sessionId: auditedSessionId,
        target: auditSessionTarget(request.sessionId),
        metadata: {},
      });
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
      };
    } finally {
      if (this.activePromptRequestIds.get(request.sessionId) === requestId) {
        this.activePromptRequestIds.delete(request.sessionId);
      }
    }

    this.recordAudit({
      category: "session",
      action: "acp.prompt",
      outcome: "completed",
      requestId,
      sessionId: auditedSessionId,
      target: auditSessionTarget(request.sessionId),
      metadata: {},
    });
    return { stopReason: "end_turn" };
  };

  cancel = async (params: CancelNotification): Promise<void> => {
    const auditedSessionId = auditSessionId(params.sessionId);
    const requestId = newAuditRequestId();
    const promptRequestId = this.activePromptRequestIds.get(params.sessionId);
    this.recordAudit({
      category: "session",
      action: "acp.prompt",
      outcome: "cancel_requested",
      requestId,
      sessionId: auditedSessionId,
      target: auditSessionTarget(params.sessionId),
      metadata: promptRequestId ? { promptRequestId } : {},
    });
    if (!promptRequestId) {
      this.recordAudit({
        category: "session",
        action: "acp.prompt",
        outcome: "denied",
        requestId,
        sessionId: auditedSessionId,
        target: auditSessionTarget(params.sessionId),
        metadata: {},
      });
      return;
    }

    let cancelled: boolean;
    try {
      cancelled = await cancelRequest(acpRequestId(params.sessionId));
    } catch (error) {
      this.recordAudit({
        category: "session",
        action: "acp.prompt",
        outcome: "failed",
        requestId,
        sessionId: auditedSessionId,
        target: auditSessionTarget(params.sessionId),
        metadata: { promptRequestId },
      });
      throw error;
    }
    this.recordAudit({
      category: "session",
      action: "acp.prompt",
      outcome: cancelled ? "cancelled" : "denied",
      requestId,
      sessionId: auditedSessionId,
      target: auditSessionTarget(params.sessionId),
      metadata: { promptRequestId },
    });
  };
  authenticate = async (_request: AuthenticateRequest): Promise<AuthenticateResponse> => {
    const requestId = newAuditRequestId();
    this.recordAudit({
      category: "system",
      action: "acp.authenticate",
      outcome: "attempted",
      requestId,
      metadata: {},
    });
    this.recordAudit({
      category: "system",
      action: "acp.authenticate",
      outcome: "denied",
      requestId,
      metadata: { reason: "unsupported" },
    });
    throw new Error("not supported");
  };
  setSessionMode = async (_request: SetSessionModeRequest): Promise<SetSessionModeResponse> => {
    return {};
  };
  resumeSession = async (request: ResumeSessionRequest): Promise<ResumeSessionResponse> => {
    const auditedSessionId = auditSessionId(request.sessionId);
    const requestId = newAuditRequestId();
    const audit = {
      category: "session" as const,
      action: "acp.session.resume",
      requestId,
      sessionId: auditedSessionId,
      target: auditSessionTarget(request.sessionId),
      metadata: {},
    };
    this.recordAudit({ ...audit, outcome: "attempted" });
    const sessionData = loadSessionFile(request.sessionId);
    if (!sessionData) {
      this.recordAudit({
        ...audit,
        outcome: "denied",
        metadata: { reason: "not_found" },
      });
      throw new Error(`Session ${request.sessionId} not found`);
    }

    try {
      const cwd = bindPersistedCwd(sessionData, request.cwd);
      this.sessions.set(request.sessionId, {
        cwd,
        mcpServers: projectMcpServers(request.mcpServers ?? [], cwd),
        sessionData,
      });
    } catch (error) {
      this.recordAudit({ ...audit, outcome: "failed" });
      throw error;
    }

    this.recordAudit({ ...audit, outcome: "completed" });
    return {};
  };
  closeSession = async (request: CloseSessionRequest): Promise<CloseSessionResponse> => {
    const auditedSessionId = auditSessionId(request.sessionId);
    const requestId = newAuditRequestId();
    this.recordAudit({
      category: "session",
      action: "acp.session.close",
      outcome: "attempted",
      requestId,
      sessionId: auditedSessionId,
      target: auditSessionTarget(request.sessionId),
      metadata: {},
    });
    const existed = this.sessions.delete(request.sessionId);
    this.recordAudit({
      category: "session",
      action: "acp.session.close",
      outcome: "completed",
      requestId,
      sessionId: auditedSessionId,
      target: auditSessionTarget(request.sessionId),
      metadata: { existed },
    });
    return {};
  };

  private recordAudit(input: Omit<AuditInput, "actor">): void {
    this.audit?.append({
      ...input,
      actor: { kind: "service", id: "acp-server" },
    });
  }
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
      swarm: applySessionRuntime(node.swarm as SwarmConfig, cwd, mcpServers),
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

function newAuditRequestId(): string {
  return `acp:${randomUUID()}`;
}

function auditSessionId(sessionId: string): string {
  if (
    /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(sessionId)
  ) {
    return sessionId;
  }
  return `session:${createHash("sha256").update(sessionId).digest("hex")}`;
}

function auditSessionTarget(sessionId: string): NonNullable<AuditInput["target"]> {
  return { kind: "acp-session", id: auditSessionId(sessionId) };
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

  const agent = new SwarmXAgent({ audit: new AuditStore() });

  const connection = new AgentSideConnection((conn) => {
    agent.setConnection(conn);
    return agent;
  }, transport);

  connection.closed.then(() => process.exit(0));
  process.on("SIGINT", () => process.exit(0));
  process.on("SIGTERM", () => process.exit(0));
}
