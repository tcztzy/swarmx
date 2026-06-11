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
  type PromptRequest,
  type PromptResponse,
  type ResumeSessionRequest,
  type ResumeSessionResponse,
  type SessionNotification,
  type SessionUpdate,
  type SetSessionModeRequest,
  type SetSessionModeResponse,
  ndJsonStream,
} from "@agentclientprotocol/sdk";
import {
  Swarm,
  appendMessages,
  createSession,
  listSessions as listSessionsFile,
  loadSession as loadSessionFile,
  saveSession,
} from "@swarmx/core";
import type { MessageChunk, SessionData, SwarmConfig } from "@swarmx/core";

interface SessionState {
  cwd: string;
  mcpServers: McpServer[];
  swarmConfig?: SwarmConfig;
  sessionData?: SessionData;
}

export class SwarmXAgent implements AcpAgent {
  private sessions = new Map<string, SessionState>();
  private conn: AgentSideConnection | null = null;

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
          embeddedContext: true,
        },
      },
      agentInfo: {
        name: "swarmx",
        title: "SwarmX Agent Engine",
        version: "3.0.0",
      },
      authMethods: [],
    };
  };

  newSession = async (request: NewSessionRequest): Promise<NewSessionResponse> => {
    const sessionId = crypto.randomUUID().slice(0, 12);
    const sessionData = createSession("swarmx", "swarmx");
    sessionData.id = sessionId;
    saveSession(sessionData);

    this.sessions.set(sessionId, {
      cwd: request.cwd ?? process.cwd(),
      mcpServers: request.mcpServers ?? [],
      sessionData,
    });
    return { sessionId };
  };

  loadSession = async (request: LoadSessionRequest): Promise<LoadSessionResponse> => {
    const { sessionId } = request;
    const sessionData = loadSessionFile(sessionId);
    if (!sessionData) {
      throw new Error(`Session ${sessionId} not found`);
    }

    this.sessions.set(sessionId, {
      cwd: request.cwd ?? process.cwd(),
      mcpServers: request.mcpServers ?? [],
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

  listSessions = async (_request: ListSessionsRequest): Promise<ListSessionsResponse> => {
    const sessions = listSessionsFile();
    return {
      sessions: sessions.map((s) => ({
        sessionId: s.id,
        cwd: process.cwd(),
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

    let userText = "";
    let swarmConfig: SwarmConfig | undefined = session.swarmConfig;

    for (const block of request.prompt) {
      if (block.type === "text") {
        userText += block.text;
        if (block._meta?.swarmConfig) {
          swarmConfig = block._meta.swarmConfig as SwarmConfig;
          session.swarmConfig = swarmConfig;
        }
      }
    }

    if (!userText.trim()) {
      return { stopReason: "end_turn" };
    }

    try {
      const swarm = swarmConfig ? new Swarm(swarmConfig) : buildDefaultSwarm();

      const result = await swarm.execute({
        messages: [{ role: "user", content: userText }],
      });

      const updates: SessionUpdate[] = [];
      for (const msg of result) {
        const update = buildSessionUpdate(msg);
        if (!update) continue;
        updates.push(update);
        const notification: SessionNotification = {
          sessionId: request.sessionId,
          update,
        };
        await conn.sessionUpdate(notification);
      }

      if (session.sessionData) {
        appendMessages(session.sessionData.id, result);
      }
    } catch (err: unknown) {
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
    }

    return { stopReason: "end_turn" };
  };

  cancel = async (_params: CancelNotification): Promise<void> => {};
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

    this.sessions.set(request.sessionId, {
      cwd: request.cwd ?? process.cwd(),
      mcpServers: request.mcpServers ?? [],
      sessionData,
    });

    return {};
  };
  closeSession = async (request: CloseSessionRequest): Promise<CloseSessionResponse> => {
    this.sessions.delete(request.sessionId);
    return {};
  };
}

function buildDefaultSwarm(): Swarm {
  return new Swarm({
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
  });
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
        toolCallId: msg.toolName ?? "tool",
        rawInput: tryParseJson(msg.content),
      };
    case "tool_result":
      return {
        sessionUpdate: "tool_call_update",
        title: msg.toolName ?? "tool",
        toolCallId: msg.toolName ?? "tool",
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
