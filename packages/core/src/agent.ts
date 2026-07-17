import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import {
  AcpClient,
  RequestCancelledError,
  currentRequestSignal,
  throwIfCurrentRequestCancelled,
} from "./acp.js";
import { type LocalTool, McpManager } from "./mcp.js";
import { ModelApiModeSchema, ModelApiSchema } from "./model-api.js";
import type { ModelApi, ModelApiMode } from "./model-api.js";
import {
  type NativeProtocolContext,
  callAnthropicMessages,
  callOpenAIResponses,
} from "./native-model.js";
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
  ModelTokenUsage,
  ProcessOptions,
} from "./types.js";
import { AgentConfigSchema, ModelTokenUsageSchema } from "./types.js";
import { SWARMX_VERSION } from "./version.js";

const CODEX_RESPONSES_BASE_URL = "https://chatgpt.com/backend-api/codex";

interface SessionInfo {
  sessionId?: string;
  session_id?: string;
  cwd?: string;
  title?: string;
  updatedAt?: string;
  updated_at?: string;
}

type ChatMsg = OpenAI.Chat.Completions.ChatCompletionMessageParam;

export interface AgentRuntimeOptions {
  createAcpClient?: () => AcpPromptClient;
  createMcpManager?: () => McpManager;
  localTools?: readonly LocalTool[];
}

interface AcpPromptClient {
  prompt(
    opts: {
      command: string;
      args: string[];
      cwd?: string;
      env?: Record<string, string>;
      clearEnv?: boolean;
      model?: string;
      effort?: string;
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
  model?: string;
  instructions: string;
  parameters: Record<string, unknown>;
  returns?: Record<string, unknown>;
  client: OpenAI;
  anthropicClient: Anthropic;
  apiProtocol: ModelApi;
  apiMode: ModelApiMode;
  mcpServers: Map<string, McpServerConfig>;
  hooks: HookRef[];
  backend: AgentBackend;
  processOptions?: ProcessOptions;
  private mcp: McpManager | null = null;
  private createAcpClient: () => AcpPromptClient;
  private createMcpManager: () => McpManager;
  private localTools: readonly LocalTool[];
  private configuredModel?: string;
  private maxOutputTokens: number;

  constructor(config: AgentConfig, options: AgentRuntimeOptions = {}) {
    const parsed = AgentConfigSchema.parse(config);
    const clientConfig = (parsed.client ?? {}) as Record<string, unknown>;
    const runtimeEnv = parsed.process?.env ?? {};
    const hasExplicitRuntimeEnv = parsed.process?.env !== undefined;
    this.name = parsed.name;
    this.description = parsed.description;
    this.backend = parsed.backend ?? { type: "swarmx" };
    this.apiMode = nativeApiMode(clientConfig, runtimeEnv);
    this.apiProtocol = nativeApiProtocol(clientConfig, runtimeEnv, this.apiMode);
    if (this.apiMode === "codex_responses" && this.apiProtocol !== "openai_responses") {
      throw new Error('apiMode "codex_responses" requires apiProtocol "openai_responses".');
    }
    this.model =
      parsed.model ??
      (this.backend.type === "swarmx"
        ? nativeModelFromEnvironment(this.apiProtocol, runtimeEnv, hasExplicitRuntimeEnv)
        : undefined);
    this.configuredModel = parsed.model;
    this.instructions = parsed.instructions ?? "";
    this.parameters = parsed.parameters ?? {};
    this.returns = parsed.returns;
    this.mcpServers = new Map(parsed.mcpServers ? Object.entries(parsed.mcpServers) : []);
    this.hooks = (parsed.hooks ?? []).map((h) => new HookRef(h));
    this.processOptions = parsed.process;
    this.createAcpClient = options.createAcpClient ?? (() => new AcpClient());
    this.createMcpManager = options.createMcpManager ?? (() => new McpManager());
    this.localTools = options.localTools ?? [];
    this.maxOutputTokens = positiveInteger(clientConfig.maxOutputTokens) ?? 8192;

    const configuredApiKey = stringProperty(clientConfig, "apiKey");
    const configuredBaseUrl =
      stringProperty(clientConfig, "baseUrl") ?? stringProperty(clientConfig, "base_url");
    const configuredAccessToken =
      stringProperty(clientConfig, "accessToken") ??
      stringProperty(clientConfig, "access_token") ??
      configuredApiKey;
    const codexAccessToken =
      configuredAccessToken ??
      runtimeEnv.CODEX_ACCESS_TOKEN ??
      (hasExplicitRuntimeEnv ? undefined : process.env.CODEX_ACCESS_TOKEN);
    this.client = new OpenAI({
      apiKey:
        this.apiMode === "codex_responses"
          ? (codexAccessToken ?? "sk-no-key")
          : (configuredApiKey ??
            runtimeEnv.OPENAI_API_KEY ??
            (hasExplicitRuntimeEnv ? undefined : process.env.OPENAI_API_KEY) ??
            "sk-no-key"),
      baseURL:
        this.apiMode === "codex_responses"
          ? (configuredBaseUrl ??
            runtimeEnv.CODEX_BASE_URL ??
            (hasExplicitRuntimeEnv ? undefined : process.env.CODEX_BASE_URL) ??
            CODEX_RESPONSES_BASE_URL)
          : (configuredBaseUrl ??
            runtimeEnv.OPENAI_BASE_URL ??
            (hasExplicitRuntimeEnv ? undefined : process.env.OPENAI_BASE_URL) ??
            undefined),
      ...(this.apiMode === "codex_responses"
        ? { defaultHeaders: codexResponsesHeaders(codexAccessToken) }
        : {}),
    });
    const anthropicApiKey =
      configuredApiKey ??
      runtimeEnv.ANTHROPIC_API_KEY ??
      (hasExplicitRuntimeEnv ? undefined : process.env.ANTHROPIC_API_KEY);
    const anthropicAuthToken =
      stringProperty(clientConfig, "authToken") ??
      runtimeEnv.ANTHROPIC_AUTH_TOKEN ??
      (hasExplicitRuntimeEnv ? undefined : process.env.ANTHROPIC_AUTH_TOKEN);
    this.anthropicClient = new Anthropic({
      apiKey: anthropicApiKey ?? (anthropicAuthToken ? null : "sk-no-key"),
      authToken: anthropicAuthToken ?? null,
      baseURL:
        configuredBaseUrl ??
        runtimeEnv.ANTHROPIC_BASE_URL ??
        (hasExplicitRuntimeEnv ? undefined : process.env.ANTHROPIC_BASE_URL) ??
        undefined,
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
    onUsage?: (usage: ModelTokenUsage) => void,
  ): Promise<{ messages: MessageChunk[] }> {
    throwIfCurrentRequestCancelled();
    if (this.backend.type === "echo") {
      return { messages: [this.echoMessage(arguments_)] };
    }
    if (this.backend.type === "custom") {
      return this.callAcp(arguments_);
    }

    try {
      throwIfCurrentRequestCancelled();
      await this.ensureMcpConnected();
      throwIfCurrentRequestCancelled();

      if (this.apiProtocol === "anthropic") {
        return await callAnthropicMessages(this.nativeProtocolContext(onUsage), arguments_);
      }
      if (this.apiProtocol === "openai_responses") {
        return await callOpenAIResponses(this.nativeProtocolContext(onUsage), arguments_);
      }
      if (this.apiProtocol !== "openai_chat") {
        throw new Error(`SwarmX does not natively execute ${this.apiProtocol} Models.`);
      }

      const messages = this.buildMessages(arguments_);
      const allChunks: MessageChunk[] = [];
      const maxSteps = 20;
      let steps = 0;

      while (steps < maxSteps) {
        steps++;
        throwIfCurrentRequestCancelled();

        const mcpTools = this.mcp?.toolsForOpenai() ?? [];
        const reasoningEffort = this.chatReasoningEffort();

        const response = await this.client.chat.completions.create(
          {
            model: this.requiredNativeModel(),
            messages,
            ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
            tools:
              mcpTools.length > 0
                ? (mcpTools as OpenAI.Chat.Completions.ChatCompletionTool[])
                : undefined,
          },
          requestOptions(),
        );
        throwIfCurrentRequestCancelled();
        reportOpenAIChatUsage(response.usage, this.requiredNativeModel(), onUsage);

        const choice = response.choices[0];
        if (!choice) break;

        const { message: assistantMsg } = choice;
        const reasoningContent = stringProperty(assistantMsg, "reasoning_content");

        if (reasoningContent) {
          allChunks.push({
            role: "assistant",
            content: reasoningContent,
            kind: "thinking",
            agent: this.name,
          });
        }

        if (assistantMsg.content) {
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
            ...(reasoningContent ? { reasoning_content: reasoningContent } : {}),
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
            let structuredContent: unknown;
            try {
              throwIfCurrentRequestCancelled();
              const result = await this.getMcp().callTool(toolName, toolArgs);
              throwIfCurrentRequestCancelled();
              toolResult = result.content;
              structuredContent = result.structuredContent;
            } catch (e) {
              throwIfCurrentRequestCancelled();
              structuredContent = { error: e instanceof Error ? e.message : String(e) };
              toolResult = JSON.stringify(structuredContent);
            }

            allChunks.push({
              role: "tool",
              content: toolResult,
              kind: "tool_result",
              toolName,
              agent: this.name,
              ...(structuredContent === undefined ? {} : { structuredContent }),
            });

            messages.push({
              role: "tool",
              content: toolResult,
              tool_call_id: tc.id,
            });
          }
        } else {
          if (assistantMsg.content) {
            messages.push({ role: "assistant", content: assistantMsg.content });
          }
          break;
        }
      }

      throwIfCurrentRequestCancelled();
      return { messages: allChunks };
    } finally {
      await this.closeMcp();
    }
  }

  async callStream(
    arguments_: Record<string, unknown>,
    onChunk: (chunk: MessageChunk) => void,
    onUsage?: (usage: ModelTokenUsage) => void,
  ): Promise<{ messages: MessageChunk[] }> {
    throwIfCurrentRequestCancelled();
    if (this.backend.type === "echo") {
      const message = this.echoMessage(arguments_);
      onChunk(message);
      return { messages: [message] };
    }
    if (this.backend.type === "custom") {
      return this.callAcp(arguments_, onChunk);
    }

    try {
      throwIfCurrentRequestCancelled();
      await this.ensureMcpConnected();
      throwIfCurrentRequestCancelled();

      if (this.apiProtocol === "anthropic") {
        return await callAnthropicMessages(
          this.nativeProtocolContext(onUsage),
          arguments_,
          onChunk,
        );
      }
      if (this.apiProtocol === "openai_responses") {
        return await callOpenAIResponses(this.nativeProtocolContext(onUsage), arguments_, onChunk);
      }
      if (this.apiProtocol !== "openai_chat") {
        throw new Error(`SwarmX does not natively execute ${this.apiProtocol} Models.`);
      }

      const messages = this.buildMessages(arguments_);
      const allChunks: MessageChunk[] = [];
      const maxSteps = 20;
      let steps = 0;

      while (steps < maxSteps) {
        steps++;
        throwIfCurrentRequestCancelled();

        const mcpTools = this.mcp?.toolsForOpenai() ?? [];
        const reasoningEffort = this.chatReasoningEffort();

        const stream = await this.client.chat.completions.create(
          {
            model: this.requiredNativeModel(),
            messages,
            ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
            tools:
              mcpTools.length > 0
                ? (mcpTools as OpenAI.Chat.Completions.ChatCompletionTool[])
                : undefined,
            stream: true,
            stream_options: { include_usage: true },
          },
          requestOptions(),
        );
        throwIfCurrentRequestCancelled();

        let content = "";
        let reasoningContent = "";
        const toolCallAcc = new Map<
          number,
          { id: string; function: { name: string; arguments: string } }
        >();
        let streamedUsage: unknown;

        for await (const chunk of stream) {
          throwIfCurrentRequestCancelled();
          if (chunk.usage) streamedUsage = chunk.usage;
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

          const reasoningDelta = stringProperty(delta, "reasoning_content");
          if (reasoningDelta) {
            reasoningContent += reasoningDelta;
            onChunk({
              role: "assistant",
              content: reasoningDelta,
              kind: "thinking",
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
        }
        throwIfCurrentRequestCancelled();
        reportOpenAIChatUsage(streamedUsage, this.requiredNativeModel(), onUsage);

        if (reasoningContent) {
          allChunks.push({
            role: "assistant",
            content: reasoningContent,
            kind: "thinking",
            agent: this.name,
          });
        }

        if (content) {
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
            ...(reasoningContent ? { reasoning_content: reasoningContent } : {}),
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
            let structuredContent: unknown;
            try {
              throwIfCurrentRequestCancelled();
              const result = await this.getMcp().callTool(tc.function.name, toolArgs);
              throwIfCurrentRequestCancelled();
              toolResult = result.content;
              structuredContent = result.structuredContent;
            } catch (e) {
              throwIfCurrentRequestCancelled();
              structuredContent = { error: e instanceof Error ? e.message : String(e) };
              toolResult = JSON.stringify(structuredContent);
            }

            const trChunk: MessageChunk = {
              role: "tool",
              content: toolResult,
              kind: "tool_result",
              toolName: tc.function.name,
              agent: this.name,
              ...(structuredContent === undefined ? {} : { structuredContent }),
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
          if (content) messages.push({ role: "assistant", content });
          break;
        }
      }

      throwIfCurrentRequestCancelled();
      return { messages: allChunks };
    } finally {
      await this.closeMcp();
    }
  }

  // ── Session management (native, file-based) ───────────────────────────────

  async newSession(cwd?: string): Promise<string> {
    const session = createSession(this.name, "swarmx", this.model, cwd ? { cwd } : {});
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
      cwd: s.cwd ?? "",
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
    this.mcp = this.createMcpManager();
    this.mcp.addLocalTools(this.localTools);
    const isClaudeCodeProfile = this.localTools.some((tool) => tool.name === "Bash");
    if (isClaudeCodeProfile && this.mcpServers.size > 0) {
      for (const [name, config] of this.mcpServers) this.mcp.startServer(name, config);
      this.mcp.addClaudeMcpDiscoveryTools();
      this.mcp.addClaudeMcpResourceTools();
      return;
    }
    for (const [name, config] of this.mcpServers) {
      try {
        await this.mcp.addServer(name, config);
      } catch (e) {
        console.warn(`Failed to connect MCP server ${name}: ${e}`);
      }
    }
    if (isClaudeCodeProfile) {
      this.mcp.addClaudeMcpResourceTools();
    }
  }

  private async closeMcp(): Promise<void> {
    const mcp = this.mcp;
    this.mcp = null;
    await mcp?.close();
  }

  private getMcp(): McpManager {
    if (!this.mcp) {
      throw new Error("MCP manager is not initialized");
    }
    return this.mcp;
  }

  private nativeProtocolContext(onUsage?: (usage: ModelTokenUsage) => void): NativeProtocolContext {
    return {
      agentName: this.name,
      model: this.requiredNativeModel(),
      instructions: this.instructions,
      parameters: this.parameters,
      maxOutputTokens: this.maxOutputTokens,
      apiMode: this.apiMode,
      openai: this.client,
      anthropic: this.anthropicClient,
      tools: () => this.mcp?.toolsForNative() ?? [],
      callTool: (name, input) => this.getMcp().callTool(name, input),
      onUsage,
    };
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

  private chatReasoningEffort(): OpenAI.Chat.Completions.ChatCompletionReasoningEffort | undefined {
    const reasoning = this.parameters.reasoning;
    if (!reasoning || typeof reasoning !== "object" || Array.isArray(reasoning)) return undefined;
    const record = reasoning as Record<string, unknown>;
    const mapping = record.parameterMapping;
    if (!mapping || typeof mapping !== "object" || Array.isArray(mapping)) return undefined;
    const mappingRecord = mapping as Record<string, unknown>;
    if (
      record.control !== "effort_enum" ||
      mappingRecord.api !== "openai.chat.completions" ||
      mappingRecord.path !== "reasoning_effort" ||
      typeof record.effort !== "string" ||
      !OPENAI_CHAT_REASONING_EFFORTS.has(record.effort)
    ) {
      return undefined;
    }
    return record.effort as OpenAI.Chat.Completions.ChatCompletionReasoningEffort;
  }

  private requiredNativeModel(): string {
    if (!this.model) throw new Error(`Native agent "${this.name}" must resolve a model.`);
    return this.model;
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
          ...(this.configuredModel ? { model: this.configuredModel } : {}),
          ...(this.configuredReasoningEffort() ? { effort: this.configuredReasoningEffort() } : {}),
        },
        this.buildAcpPrompt(arguments_),
        undefined,
        undefined,
        onChunk,
      );
      throwIfCurrentRequestCancelled();
      return { messages: result.messages };
    } catch (error) {
      if (error instanceof RequestCancelledError) throw error;
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

  private configuredReasoningEffort(): string | undefined {
    const reasoning = this.parameters.reasoning;
    if (!reasoning || typeof reasoning !== "object" || Array.isArray(reasoning)) return undefined;
    const effort = (reasoning as Record<string, unknown>).effort;
    return typeof effort === "string" && effort.length > 0 ? effort : undefined;
  }
}

const OPENAI_CHAT_REASONING_EFFORTS = new Set([
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
]);

function nativeApiProtocol(
  clientConfig: Record<string, unknown>,
  runtimeEnv: Record<string, string>,
  apiMode: ModelApiMode,
): ModelApi {
  const configured = ModelApiSchema.safeParse(
    clientConfig.apiProtocol ?? runtimeEnv.SWARMX_MODEL_API,
  );
  if (configured.success) return configured.data;
  if (apiMode === "codex_responses") return "openai_responses";
  if (runtimeEnv.ANTHROPIC_MODEL && !runtimeEnv.OPENAI_MODEL) return "anthropic";
  return "openai_chat";
}

function nativeApiMode(
  clientConfig: Record<string, unknown>,
  runtimeEnv: Record<string, string>,
): ModelApiMode {
  const configured =
    clientConfig.apiMode ?? clientConfig.api_mode ?? runtimeEnv.SWARMX_API_MODE ?? "standard";
  return ModelApiModeSchema.parse(configured);
}

function nativeModelFromEnvironment(
  apiProtocol: ModelApi,
  runtimeEnv: Record<string, string>,
  hasExplicitRuntimeEnv: boolean,
): string | undefined {
  if (apiProtocol === "anthropic") {
    return (
      runtimeEnv.ANTHROPIC_MODEL ??
      (hasExplicitRuntimeEnv ? undefined : process.env.ANTHROPIC_MODEL)
    );
  }
  if (apiProtocol === "ollama") {
    return (
      runtimeEnv.OLLAMA_MODEL ?? (hasExplicitRuntimeEnv ? undefined : process.env.OLLAMA_MODEL)
    );
  }
  return (
    runtimeEnv.OPENAI_MODEL ??
    (hasExplicitRuntimeEnv ? undefined : process.env.OPENAI_MODEL) ??
    "gpt-4o"
  );
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : undefined;
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

function stringProperty(value: unknown, key: string): string | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const property = (value as Record<string, unknown>)[key];
  return typeof property === "string" && property.length > 0 ? property : undefined;
}

function reportOpenAIChatUsage(
  value: unknown,
  model: string,
  onUsage?: (usage: ModelTokenUsage) => void,
): void {
  if (!value || typeof value !== "object" || Array.isArray(value)) return;
  const usage = value as Record<string, unknown>;
  const promptDetails = objectProperty(usage.prompt_tokens_details);
  const completionDetails = objectProperty(usage.completion_tokens_details);
  const inputTokens = nonnegativeInteger(usage.prompt_tokens) ?? 0;
  const outputTokens = nonnegativeInteger(usage.completion_tokens) ?? 0;
  const totalTokens = nonnegativeInteger(usage.total_tokens) ?? inputTokens + outputTokens;
  if (totalTokens === 0) return;
  onUsage?.(
    ModelTokenUsageSchema.parse({
      inputTokens,
      outputTokens,
      reasoningTokens: nonnegativeInteger(completionDetails.reasoning_tokens) ?? 0,
      cachedInputTokens: nonnegativeInteger(promptDetails.cached_tokens) ?? 0,
      totalTokens,
      estimated: false,
      model,
      provider: "openai_chat",
    }),
  );
}

function objectProperty(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function nonnegativeInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : undefined;
}

function codexResponsesHeaders(accessToken: string | undefined): Record<string, string> {
  const accountId = accessToken ? chatGptAccountId(accessToken) : undefined;
  return {
    "User-Agent": `swarmx/${SWARMX_VERSION} (codex_responses)`,
    originator: "swarmx",
    ...(accountId ? { "ChatGPT-Account-ID": accountId } : {}),
  };
}

function chatGptAccountId(accessToken: string): string | undefined {
  const payload = accessToken.split(".")[1];
  if (!payload) return undefined;
  try {
    const claims = JSON.parse(Buffer.from(payload, "base64url").toString("utf8")) as Record<
      string,
      unknown
    >;
    const auth = claims["https://api.openai.com/auth"];
    return stringProperty(auth, "chatgpt_account_id");
  } catch {
    return undefined;
  }
}

function requestOptions(): { signal?: AbortSignal } | undefined {
  const signal = currentRequestSignal();
  return signal ? { signal } : undefined;
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
