import OpenAI from "openai";
import { AcpClient } from "./acp.js";
import { McpManager } from "./mcp.js";
import {
  appendMessages,
  createSession,
  listSessions as listSessionsFile,
  loadSession as loadSessionFile,
  saveSession,
} from "./session.js";
import type {
  AgentBackend,
  AgentConfig,
  McpServerConfig,
  MessageChunk,
  ProcessOptions,
} from "./types.js";
import { AgentConfigSchema } from "./types.js";

interface SessionInfo {
  sessionId?: string;
  session_id?: string;
  cwd?: string;
  title?: string;
  updatedAt?: string;
  updated_at?: string;
}

type ChatMsg = OpenAI.Chat.Completions.ChatCompletionMessageParam;

interface AgentRuntimeOptions {
  createAcpClient?: () => AcpPromptClient;
}

interface AcpPromptClient {
  prompt(
    opts: {
      command: string;
      args: string[];
      cwd?: string;
      env?: Record<string, string>;
      clearEnv?: boolean;
    },
    userText: string,
    swarmConfig?: unknown,
    sessionId?: string,
    onChunk?: (chunk: MessageChunk) => void,
  ): Promise<{ messages: MessageChunk[] }>;
  stderrOutput?(): string;
}

export class Agent {
  name: string;
  description?: string;
  model: string;
  instructions: string;
  parameters: Record<string, unknown>;
  returns?: Record<string, unknown>;
  client: OpenAI;
  mcpServers: Map<string, McpServerConfig>;
  hooks: HookRef[];
  backend: AgentBackend;
  processOptions?: ProcessOptions;
  private mcp: McpManager | null = null;
  private createAcpClient: () => AcpPromptClient;

  constructor(config: AgentConfig, options: AgentRuntimeOptions = {}) {
    const parsed = AgentConfigSchema.parse(config);
    this.name = parsed.name;
    this.description = parsed.description;
    this.model = parsed.model ?? process.env.OPENAI_MODEL ?? "gpt-4o";
    this.instructions = parsed.instructions ?? "";
    this.parameters = parsed.parameters ?? {};
    this.returns = parsed.returns;
    this.mcpServers = new Map(parsed.mcpServers ? Object.entries(parsed.mcpServers) : []);
    this.hooks = (parsed.hooks ?? []).map((h) => new HookRef(h));
    this.backend = parsed.backend ?? { type: "swarmx" };
    this.processOptions = parsed.process;
    this.createAcpClient = options.createAcpClient ?? (() => new AcpClient());

    const clientConfig = (parsed.client ?? {}) as Record<string, unknown>;
    this.client = new OpenAI({
      apiKey: (clientConfig.apiKey as string) ?? process.env.OPENAI_API_KEY ?? "sk-no-key",
      baseURL: (clientConfig.baseUrl as string) ?? process.env.OPENAI_BASE_URL ?? undefined,
    });
  }

  toSwarmConfig(): Record<string, unknown> {
    return {
      name: this.name,
      root: this.name,
      nodes: {
        [this.name]: {
          name: this.name,
          description: this.description,
          model: this.model,
          instructions: this.instructions,
          mcpServers: Object.fromEntries(this.mcpServers),
          hooks: this.hooks,
        },
      },
      edges: [],
      parameters: this.parameters,
    };
  }

  // ── Native LLM call ───────────────────────────────────────────────────────

  async call(
    arguments_: Record<string, unknown>,
    _context?: Record<string, unknown>,
  ): Promise<{ messages: MessageChunk[] }> {
    if (this.backend.type === "echo") {
      return { messages: [this.echoMessage(arguments_)] };
    }
    if (this.backend.type === "custom") {
      return this.callAcp(arguments_);
    }

    await this.ensureMcpConnected();

    const messages = this.buildMessages(arguments_);
    const allChunks: MessageChunk[] = [];
    const maxSteps = 20;
    let steps = 0;

    while (steps < maxSteps) {
      steps++;

      const mcpTools = this.mcp?.toolsForOpenai() ?? [];

      const response = await this.client.chat.completions.create({
        model: this.model,
        messages,
        tools:
          mcpTools.length > 0
            ? (mcpTools as OpenAI.Chat.Completions.ChatCompletionTool[])
            : undefined,
      });

      const choice = response.choices[0];
      if (!choice) break;

      const { message: assistantMsg } = choice;

      if (assistantMsg.content) {
        messages.push({ role: "assistant", content: assistantMsg.content });
        allChunks.push({
          role: "assistant",
          content: assistantMsg.content,
          kind: "message",
          agent: this.name,
        });
      }

      const toolCalls = assistantMsg.tool_calls;
      if (toolCalls && toolCalls.length > 0) {
        messages.push({
          role: "assistant",
          content: assistantMsg.content,
          tool_calls: toolCalls,
        } as ChatMsg);

        for (const tc of toolCalls) {
          if (!("function" in tc)) continue;

          const toolName = tc.function.name;
          let toolArgs: Record<string, unknown>;
          try {
            toolArgs = JSON.parse(tc.function.arguments);
          } catch {
            toolArgs = {};
          }

          allChunks.push({
            role: "assistant",
            content: tc.function.arguments,
            kind: "tool_call",
            toolName,
            agent: this.name,
          });

          let toolResult: string;
          try {
            const result = await this.getMcp().callTool(toolName, toolArgs);
            toolResult = JSON.stringify(result);
          } catch (e) {
            toolResult = JSON.stringify({
              error: e instanceof Error ? e.message : String(e),
            });
          }

          allChunks.push({
            role: "tool",
            content: toolResult,
            kind: "tool_result",
            toolName,
            agent: this.name,
          });

          messages.push({
            role: "tool",
            content: toolResult,
            tool_call_id: tc.id,
          });
        }
      } else {
        break;
      }
    }

    return { messages: allChunks };
  }

  async callStream(
    arguments_: Record<string, unknown>,
    onChunk: (chunk: MessageChunk) => void,
  ): Promise<{ messages: MessageChunk[] }> {
    if (this.backend.type === "echo") {
      const message = this.echoMessage(arguments_);
      onChunk(message);
      return { messages: [message] };
    }
    if (this.backend.type === "custom") {
      return this.callAcp(arguments_, onChunk);
    }

    await this.ensureMcpConnected();

    const messages = this.buildMessages(arguments_);
    const allChunks: MessageChunk[] = [];
    const maxSteps = 20;
    let steps = 0;

    while (steps < maxSteps) {
      steps++;

      const mcpTools = this.mcp?.toolsForOpenai() ?? [];

      const stream = await this.client.chat.completions.create({
        model: this.model,
        messages,
        tools:
          mcpTools.length > 0
            ? (mcpTools as OpenAI.Chat.Completions.ChatCompletionTool[])
            : undefined,
        stream: true,
      });

      let content = "";
      const toolCallAcc = new Map<
        number,
        { id: string; function: { name: string; arguments: string } }
      >();

      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta;
        if (!delta) continue;

        if (delta.content) {
          content += delta.content;
          onChunk({
            role: "assistant",
            content: delta.content,
            kind: "message",
            agent: this.name,
          });
        }

        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            const existing = toolCallAcc.get(tc.index) ?? {
              id: "",
              function: { name: "", arguments: "" },
            };
            if (tc.id) existing.id = tc.id;
            if (tc.function?.name) existing.function.name += tc.function.name;
            if (tc.function?.arguments) existing.function.arguments += tc.function.arguments;
            toolCallAcc.set(tc.index, existing);
          }
        }

        if (chunk.choices[0]?.finish_reason) {
          break;
        }
      }

      if (content) {
        messages.push({ role: "assistant", content });
        allChunks.push({
          role: "assistant",
          content,
          kind: "message",
          agent: this.name,
        });
      }

      const toolCalls = Array.from(toolCallAcc.values()).filter((tc) => tc.function.name);

      if (toolCalls.length > 0) {
        const toolCallObjs = toolCalls.map((tc) => ({
          id: tc.id,
          type: "function" as const,
          function: {
            name: tc.function.name,
            arguments: tc.function.arguments,
          },
        }));

        messages.push({
          role: "assistant",
          content: content || null,
          tool_calls: toolCallObjs,
        } as ChatMsg);

        for (const tc of toolCallObjs) {
          onChunk({
            role: "assistant",
            content: tc.function.arguments,
            kind: "tool_call",
            toolName: tc.function.name,
            agent: this.name,
          });
          allChunks.push({
            role: "assistant",
            content: tc.function.arguments,
            kind: "tool_call",
            toolName: tc.function.name,
            agent: this.name,
          });

          let toolArgs: Record<string, unknown>;
          try {
            toolArgs = JSON.parse(tc.function.arguments);
          } catch {
            toolArgs = {};
          }

          let toolResult: string;
          try {
            const result = await this.getMcp().callTool(tc.function.name, toolArgs);
            toolResult = JSON.stringify(result);
          } catch (e) {
            toolResult = JSON.stringify({
              error: e instanceof Error ? e.message : String(e),
            });
          }

          const trChunk: MessageChunk = {
            role: "tool",
            content: toolResult,
            kind: "tool_result",
            toolName: tc.function.name,
            agent: this.name,
          };
          onChunk(trChunk);
          allChunks.push(trChunk);

          messages.push({
            role: "tool",
            content: toolResult,
            tool_call_id: tc.id,
          });
        }
      } else {
        break;
      }
    }

    return { messages: allChunks };
  }

  // ── Session management (native, file-based) ───────────────────────────────

  async newSession(_cwd?: string): Promise<string> {
    const session = createSession(this.name, "swarmx", this.model);
    saveSession(session);
    return session.id;
  }

  async callWithSession(
    arguments_: Record<string, unknown>,
    sessionId: string,
    _context?: Record<string, unknown>,
  ): Promise<{ messages: MessageChunk[] }> {
    const result = await this.call(arguments_);
    try {
      appendMessages(sessionId, result.messages);
    } catch (e) {
      console.warn(`Failed to append messages to session ${sessionId}: ${e}`);
    }
    return result;
  }

  async listSessions(_cwd?: string): Promise<SessionInfo[]> {
    const sessions = listSessionsFile();
    return sessions.map((s) => ({
      sessionId: s.id,
      session_id: s.id,
      cwd: "",
      title: s.title,
      updatedAt: s.updatedAt,
      updated_at: s.updatedAt,
    }));
  }

  async loadSession(
    sessionId: string,
    _cwd?: string,
  ): Promise<{ response: unknown; messages: MessageChunk[] }> {
    const session = loadSessionFile(sessionId);
    if (!session) throw new Error(`Session ${sessionId} not found`);
    return { response: {}, messages: session.messages };
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  private async ensureMcpConnected(): Promise<void> {
    if (this.mcp) return;
    this.mcp = new McpManager();
    for (const [name, config] of this.mcpServers) {
      try {
        await this.mcp.addServer(name, config);
      } catch (e) {
        console.warn(`Failed to connect MCP server ${name}: ${e}`);
      }
    }
  }

  private getMcp(): McpManager {
    if (!this.mcp) {
      throw new Error("MCP manager is not initialized");
    }
    return this.mcp;
  }

  private buildMessages(arguments_: Record<string, unknown>): ChatMsg[] {
    const msgs: ChatMsg[] = [];

    if (this.instructions) {
      msgs.push({ role: "system", content: this.instructions });
    }

    const raw = arguments_.messages as
      | Array<{
          role: string;
          content: string | null;
          tool_calls?: unknown[];
          tool_call_id?: string;
        }>
      | undefined;

    if (raw) {
      for (const m of raw) {
        if (m.tool_calls) {
          msgs.push({
            role: "assistant",
            content: m.content,
            tool_calls:
              m.tool_calls as OpenAI.Chat.Completions.ChatCompletionAssistantMessageParam["tool_calls"],
          } as ChatMsg);
        } else if (m.tool_call_id) {
          msgs.push({
            role: "tool",
            content: m.content ?? "",
            tool_call_id: m.tool_call_id,
          });
        } else if (m.role === "user" || m.role === "assistant" || m.role === "system") {
          msgs.push({
            role: m.role,
            content: m.content ?? "",
          });
        }
      }
    }

    return msgs;
  }

  private echoMessage(arguments_: Record<string, unknown>): MessageChunk {
    return {
      role: "assistant",
      content: latestUserContent(arguments_),
      kind: "message",
      agent: this.name,
    };
  }

  private async callAcp(
    arguments_: Record<string, unknown>,
    onChunk?: (chunk: MessageChunk) => void,
  ): Promise<{ messages: MessageChunk[] }> {
    if (this.backend.type !== "custom") {
      throw new Error(`Agent "${this.name}" backend is not an ACP custom backend.`);
    }

    const client = this.createAcpClient();
    try {
      const result = await client.prompt(
        {
          command: this.backend.program,
          args: this.backend.args ?? [],
          cwd: this.processOptions?.currentDir,
          env: this.processOptions?.env,
          clearEnv: this.processOptions?.clearEnv,
        },
        this.buildAcpPrompt(arguments_),
        undefined,
        undefined,
        onChunk,
      );
      return { messages: result.messages };
    } catch (error) {
      const stderr = client.stderrOutput?.();
      const detail = stderr ? ` Stderr: ${stderr}` : "";
      throw new Error(
        `ACP backend failed for agent "${this.name}": ${errorMessage(error)}.${detail}`,
      );
    }
  }

  private buildAcpPrompt(arguments_: Record<string, unknown>): string {
    const request = latestUserContent(arguments_);
    if (!this.instructions.trim()) return request;
    return `Agent instructions:\n${this.instructions.trim()}\n\nUser request:\n${request}`;
  }
}

function latestUserContent(arguments_: Record<string, unknown>): string {
  const raw = arguments_.messages as
    | Array<{
        role: string;
        content: string | null;
      }>
    | undefined;

  for (const message of [...(raw ?? [])].reverse()) {
    if (message.role === "user") {
      return message.content ?? "";
    }
  }

  return "";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

// ── HookRef ──────────────────────────────────────────────────────────────────

export class HookRef {
  onStart?: string;
  onEnd?: string;
  onHandoff?: string;
  onChunk?: string;

  constructor(config: {
    onStart?: string;
    onEnd?: string;
    onHandoff?: string;
    onChunk?: string;
  }) {
    this.onStart = config.onStart;
    this.onEnd = config.onEnd;
    this.onHandoff = config.onHandoff;
    this.onChunk = config.onChunk;
  }
}
