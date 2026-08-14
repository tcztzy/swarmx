import { randomUUID } from "node:crypto";
import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import type { AcpPermissionHandler, AcpPromptInput } from "./acp.js";
import { AcpClient, AcpSessionUnavailableError } from "./acp.js";
import type {
  AgentContextEngine,
  CompiledContext,
  ContextCompilePhase,
  ContextWindowSource,
} from "./context-engine.js";
import {
  appendHookContext,
  dispatchHooks,
  Hook,
  type HookInvocation,
  type HookRuntimeOptions,
} from "./hook.js";
import type { LocalTool, LocalToolProgress } from "./local-tool-contracts.js";
import { McpManager, type McpServerContract } from "./mcp.js";
import {
  attachmentFallbackText,
  createInlineMediaLoader,
  type InlineMediaLoader,
  validateMediaAttachments,
} from "./media.js";
import type { ModelApi, ModelApiMode } from "./model-api.js";
import { ModelApiModeSchema, ModelApiSchema } from "./model-api.js";
import {
  appendProviderHostedWebSearchInstructions,
  callAnthropicMessages,
  callOpenAIResponses,
  type NativeProtocolContext,
  providerHostedWebSearchTool,
} from "./native-model.js";
import {
  appendGlobalMemoryInstructions,
  appendMemoryReflectionInstructions,
  appendPersonalMemoryInstructions,
  type GlobalMemorySnapshot,
  GlobalMemorySnapshotSchema,
  type MemoryReflectionDecision,
  MemoryReflectionDecisionSchema,
  type PersonalMemorySnapshot,
  PersonalMemorySnapshotSchema,
} from "./personal-memory.js";
import {
  appendProjectBootstrapInstructions,
  buildProjectBootstrapReceipt,
  PROJECT_BOOTSTRAP_TIMEOUT_MS,
  type ProjectBootstrapBinding,
  ProjectBootstrapBindingSchema,
  type ProjectBootstrapReceipt,
  type ProjectBootstrapSnapshot,
  parseProjectBootstrapResult,
} from "./project-bootstrap.js";
import {
  currentRequestSignal,
  RequestCancelledError,
  throwIfCurrentRequestCancelled,
} from "./request-scope.js";
import {
  appendMessages,
  createSession,
  listSessionSummaries as listSessionsFile,
  loadSession as loadSessionFile,
  saveSession,
} from "./session.js";
import {
  buildDeliveredInstructions,
  SkillDeliveryError,
  type SkillInstructionDelivery,
  SkillInstructionDeliverySchema,
} from "./skill-delivery.js";
import type {
  AgentBackend,
  AgentConfig,
  McpServerConfig,
  MediaAttachment,
  MessageChunk,
  ModelTokenUsage,
  ProcessOptions,
} from "./types.js";
import { AgentConfigSchema, ModelTokenUsageSchema } from "./types.js";
import { SWARMX_VERSION } from "./version.js";

const CODEX_RESPONSES_BASE_URL = "https://chatgpt.com/backend-api/codex";
export const REQUIRED_MCP_CONNECT_TIMEOUT_MS = 10_000;

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
  /**
   * Request-scoped, digest-verified `prompt_fragment` Skill deliveries. Content
   * must already be loaded and verified by the caller; the Agent re-verifies
   * the digest and appends each fragment to the model-visible instructions.
   * Persisted SwarmConfig and Skill files are never modified.
   */
  skillInstructions?: readonly SkillInstructionDelivery[];
  /**
   * Agent-name-scoped Skill deliveries. When present, only the named Agent
   * node receives the listed deliveries; the generic `skillInstructions`
   * option is ignored. This binds evolution to the target Agent instead of
   * leaking one delivery to every node in the swarm.
   */
  skillInstructionsByAgent?: Record<string, readonly SkillInstructionDelivery[]>;
  acpPermissionHandler?: AcpPermissionHandler;
  acpMode?: string;
  acpSessionId?: string;
  onAcpSessionId?: (sessionId: string | undefined) => void | Promise<void>;
  /** Resolves configured hook capability names through explicit host authority. */
  hook?: HookRuntimeOptions;
  /** Request-scoped, read-only Personal Memory for direct SwarmX execution only. */
  personalMemory?: PersonalMemorySnapshot;
  /** Request-scoped, read-only USER.md and MEMORY.md snapshot. */
  globalMemory?: GlobalMemorySnapshot;
  /** Session-scoped reminder to reflect after the current task. */
  memoryReflection?: MemoryReflectionDecision;
  /** Compiles one immutable, bounded model context before a native Provider request. */
  contextEngine?: AgentContextEngine;
  /** MCP server ids whose connection failure must stop native execution. */
  requiredMcpServers?: readonly string[];
  /** One verified, Project-bound service that supplies the execution-attempt bootstrap. */
  projectBootstrap?: ProjectBootstrapBinding;
  /** Publishes the concise bootstrap receipt without exposing the snapshot or Project root. */
  onProjectBootstrap?: (receipt: ProjectBootstrapReceipt) => void | Promise<void>;
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
      preferredMode?: string;
      requestPermission?: AcpPermissionHandler;
      onSessionId?: (sessionId: string) => void | Promise<void>;
    },
    input: AcpPromptInput,
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
  private acpPermissionHandler?: AcpPermissionHandler;
  private acpMode?: string;
  private acpSessionId?: string;
  private onAcpSessionId?: (sessionId: string | undefined) => void | Promise<void>;
  private hookRuntime?: HookRuntimeOptions;
  private configuredModel?: string;
  private maxOutputTokens: number;
  private readonly contextWindowTokens?: number;
  private readonly contextWindowSource: ContextWindowSource;
  private readonly providerHostedWebSearch: boolean;
  private readonly skillInstructions: readonly SkillInstructionDelivery[];
  private readonly contextEngine?: AgentContextEngine;
  private readonly requiredMcpServers: ReadonlySet<string>;
  private readonly projectBootstrap?: ProjectBootstrapBinding;
  private readonly onProjectBootstrap?: AgentRuntimeOptions["onProjectBootstrap"];

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
    const scopedDeliveries = options.skillInstructionsByAgent?.[parsed.name];
    const deliveries = scopedDeliveries ?? options.skillInstructions ?? [];
    this.skillInstructions = deliveries.map((delivery) =>
      SkillInstructionDeliverySchema.parse(delivery),
    );
    if (
      this.skillInstructions.length > 0 &&
      !["swarmx", "echo"].includes(parsed.backend?.type ?? "swarmx")
    ) {
      throw new SkillDeliveryError(
        "external_harness",
        `Skill prompt_fragment delivery is unsupported for backend "${parsed.backend?.type ?? "swarmx"}" on agent "${parsed.name}".`,
      );
    }
    const deliveredInstructions = buildDeliveredInstructions(
      parsed.instructions ?? "",
      this.skillInstructions,
    );
    const globalMemory = options.globalMemory
      ? GlobalMemorySnapshotSchema.parse(options.globalMemory)
      : undefined;
    const personalMemory =
      !globalMemory && options.personalMemory
        ? PersonalMemorySnapshotSchema.parse(options.personalMemory)
        : undefined;
    const memoryInstructions = globalMemory
      ? appendGlobalMemoryInstructions(deliveredInstructions, globalMemory)
      : personalMemory
        ? appendPersonalMemoryInstructions(deliveredInstructions, personalMemory)
        : deliveredInstructions;
    this.instructions = options.memoryReflection
      ? appendMemoryReflectionInstructions(
          memoryInstructions,
          MemoryReflectionDecisionSchema.parse(options.memoryReflection),
        )
      : memoryInstructions;
    this.parameters = parsed.parameters ?? {};
    this.returns = parsed.returns;
    this.mcpServers = new Map(parsed.mcpServers ? Object.entries(parsed.mcpServers) : []);
    this.hooks = (parsed.hooks ?? []).map((h) => new HookRef(h));
    this.processOptions = parsed.process;
    this.createAcpClient = options.createAcpClient ?? (() => new AcpClient());
    this.createMcpManager = options.createMcpManager ?? (() => new McpManager());
    this.localTools = options.localTools ?? [];
    this.acpPermissionHandler = options.acpPermissionHandler;
    this.acpMode = options.acpMode;
    this.acpSessionId = options.acpSessionId;
    this.onAcpSessionId = options.onAcpSessionId;
    this.hookRuntime = options.hook;
    this.contextEngine = options.contextEngine;
    this.requiredMcpServers = new Set(
      (options.requiredMcpServers ?? []).map((name) => {
        const normalized = name.trim();
        if (!normalized) throw new Error("Required MCP server names must be non-empty.");
        return normalized;
      }),
    );
    for (const name of this.requiredMcpServers) {
      if (!this.mcpServers.has(name)) {
        throw new Error(
          `Required MCP server "${name}" is not configured for Agent "${this.name}".`,
        );
      }
    }
    if (this.requiredMcpServers.size > 0 && this.backend.type !== "swarmx") {
      throw new Error(
        `Required MCP servers are unsupported for backend "${this.backend.type}"; use the direct SwarmX backend.`,
      );
    }
    this.projectBootstrap = options.projectBootstrap
      ? ProjectBootstrapBindingSchema.parse(options.projectBootstrap)
      : undefined;
    if (this.projectBootstrap) {
      if (!this.requiredMcpServers.has(this.projectBootstrap.capabilityId)) {
        throw new Error(
          `Project bootstrap service "${this.projectBootstrap.capabilityId}" must be a required MCP server.`,
        );
      }
      if (this.backend.type !== "swarmx") {
        throw new Error(`Project bootstrap is unsupported for backend "${this.backend.type}".`);
      }
    }
    this.onProjectBootstrap = options.onProjectBootstrap;
    this.maxOutputTokens = positiveInteger(clientConfig.maxOutputTokens) ?? 8192;
    this.contextWindowTokens = positiveInteger(clientConfig.contextWindowTokens);
    this.contextWindowSource = this.contextWindowTokens
      ? contextWindowSource(clientConfig.contextWindowSource)
      : "fallback_config";

    const configuredApiKey = stringProperty(clientConfig, "apiKey");
    const configuredBaseUrl =
      stringProperty(clientConfig, "baseUrl") ?? stringProperty(clientConfig, "base_url");
    const configuredAccessToken =
      stringProperty(clientConfig, "accessToken") ??
      stringProperty(clientConfig, "access_token") ??
      configuredApiKey;
    const hostedWebSearchPreference = booleanProperty(clientConfig, "providerHostedWebSearch");
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
    this.providerHostedWebSearch =
      hostedWebSearchPreference !== false &&
      ((this.apiProtocol === "openai_responses" &&
        this.client.apiKey !== "sk-no-key" &&
        isOfficialHostedResponsesEndpoint(this.client.baseURL)) ||
        (this.apiProtocol === "anthropic" &&
          Boolean(anthropicApiKey ?? anthropicAuthToken) &&
          isOfficialDeepseekAnthropicEndpoint(this.anthropicClient.baseURL)));
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
    context: Record<string, unknown> = {},
    onUsage?: (usage: ModelTokenUsage) => void,
  ): Promise<{ messages: MessageChunk[] }> {
    return this.runWithHooks(arguments_, context, undefined, (effectiveArguments) =>
      this.callUnchecked(effectiveArguments, context, onUsage),
    );
  }

  private async callUnchecked(
    arguments_: Record<string, unknown>,
    runtimeContext: Record<string, unknown>,
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
      const contextRequestId = this.contextEngine
        ? resolveContextRequestId(arguments_, runtimeContext)
        : undefined;
      let modelRequest = await this.compileModelRequest(
        arguments_,
        runtimeContext,
        contextRequestId,
      );
      throwIfCurrentRequestCancelled();
      await this.ensureMcpConnected();
      throwIfCurrentRequestCancelled();
      const projectBootstrap = await this.loadProjectBootstrap();
      const runInstructions = projectBootstrap
        ? appendProjectBootstrapInstructions(this.instructions, projectBootstrap)
        : this.instructions;
      if (this.contextEngine?.finalize || (this.contextEngine && projectBootstrap)) {
        modelRequest = await this.compileModelRequest(
          arguments_,
          runtimeContext,
          contextRequestId,
          "final",
          this.contextToolDefinitions(),
          runInstructions,
        );
        throwIfCurrentRequestCancelled();
      } else if (projectBootstrap) {
        modelRequest = {
          ...modelRequest,
          instructions: appendProjectBootstrapInstructions(
            modelRequest.instructions,
            projectBootstrap,
          ),
        };
      }

      if (this.apiProtocol === "anthropic") {
        return await callAnthropicMessages(
          this.nativeProtocolContext(onUsage, modelRequest.instructions),
          modelRequest.arguments,
        );
      }
      if (this.apiProtocol === "openai_responses") {
        return await callOpenAIResponses(
          this.nativeProtocolContext(onUsage, modelRequest.instructions),
          modelRequest.arguments,
        );
      }
      if (this.apiProtocol !== "openai_chat") {
        throw new Error(`SwarmX does not natively execute ${this.apiProtocol} Models.`);
      }

      const messages = await this.buildMessages(modelRequest.arguments, modelRequest.instructions);
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
            max_completion_tokens: this.maxOutputTokens,
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
              render: { invocationId: tc.id, status: "running" },
            });

            let toolResult: string;
            let structuredContent: unknown;
            let toolFailed = false;
            try {
              throwIfCurrentRequestCancelled();
              const result = await this.getMcp().callTool(toolName, toolArgs, {
                invocationId: tc.id,
              });
              throwIfCurrentRequestCancelled();
              toolResult = result.content;
              structuredContent = result.structuredContent;
              toolFailed = result.isError;
            } catch (e) {
              throwIfCurrentRequestCancelled();
              structuredContent = { error: e instanceof Error ? e.message : String(e) };
              toolResult = JSON.stringify(structuredContent);
              toolFailed = true;
            }

            allChunks.push({
              role: "tool",
              content: toolResult,
              kind: "tool_result",
              toolName,
              agent: this.name,
              render: {
                invocationId: tc.id,
                status: toolFailed ? "failed" : "succeeded",
              },
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
    context: Record<string, unknown> = {},
  ): Promise<{ messages: MessageChunk[] }> {
    return this.runWithHooks(arguments_, context, onChunk, (effectiveArguments, emitChunk) =>
      this.callStreamUnchecked(effectiveArguments, context, emitChunk ?? onChunk, onUsage),
    );
  }

  private async callStreamUnchecked(
    arguments_: Record<string, unknown>,
    runtimeContext: Record<string, unknown>,
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
      const contextRequestId = this.contextEngine
        ? resolveContextRequestId(arguments_, runtimeContext)
        : undefined;
      let modelRequest = await this.compileModelRequest(
        arguments_,
        runtimeContext,
        contextRequestId,
      );
      throwIfCurrentRequestCancelled();
      await this.ensureMcpConnected();
      throwIfCurrentRequestCancelled();
      const projectBootstrap = await this.loadProjectBootstrap();
      const runInstructions = projectBootstrap
        ? appendProjectBootstrapInstructions(this.instructions, projectBootstrap)
        : this.instructions;
      if (this.contextEngine?.finalize || (this.contextEngine && projectBootstrap)) {
        modelRequest = await this.compileModelRequest(
          arguments_,
          runtimeContext,
          contextRequestId,
          "final",
          this.contextToolDefinitions(),
          runInstructions,
        );
        throwIfCurrentRequestCancelled();
      } else if (projectBootstrap) {
        modelRequest = {
          ...modelRequest,
          instructions: appendProjectBootstrapInstructions(
            modelRequest.instructions,
            projectBootstrap,
          ),
        };
      }

      if (this.apiProtocol === "anthropic") {
        return await callAnthropicMessages(
          this.nativeProtocolContext(onUsage, modelRequest.instructions),
          modelRequest.arguments,
          onChunk,
        );
      }
      if (this.apiProtocol === "openai_responses") {
        return await callOpenAIResponses(
          this.nativeProtocolContext(onUsage, modelRequest.instructions),
          modelRequest.arguments,
          onChunk,
        );
      }
      if (this.apiProtocol !== "openai_chat") {
        throw new Error(`SwarmX does not natively execute ${this.apiProtocol} Models.`);
      }

      const messages = await this.buildMessages(modelRequest.arguments, modelRequest.instructions);
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
            max_completion_tokens: this.maxOutputTokens,
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
              render: { invocationId: tc.id, status: "running" },
            });
            allChunks.push({
              role: "assistant",
              content: tc.function.arguments,
              kind: "tool_call",
              toolName: tc.function.name,
              agent: this.name,
              render: { invocationId: tc.id, status: "running" },
            });

            let toolArgs: Record<string, unknown>;
            try {
              toolArgs = JSON.parse(tc.function.arguments);
            } catch {
              toolArgs = {};
            }

            let toolResult: string;
            let structuredContent: unknown;
            let toolFailed = false;
            try {
              throwIfCurrentRequestCancelled();
              const result = await this.getMcp().callTool(tc.function.name, toolArgs, {
                invocationId: tc.id,
                onProgress: (progress) =>
                  onChunk(toolProgressChunk(this.name, tc.function.name, tc.id, progress)),
              });
              throwIfCurrentRequestCancelled();
              toolResult = result.content;
              structuredContent = result.structuredContent;
              toolFailed = result.isError;
            } catch (e) {
              throwIfCurrentRequestCancelled();
              structuredContent = { error: e instanceof Error ? e.message : String(e) };
              toolResult = JSON.stringify(structuredContent);
              toolFailed = true;
            }

            const trChunk: MessageChunk = {
              role: "tool",
              content: toolResult,
              kind: "tool_result",
              toolName: tc.function.name,
              agent: this.name,
              render: {
                invocationId: tc.id,
                status: toolFailed ? "failed" : "succeeded",
              },
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

  async runHandoffHooks(
    arguments_: Record<string, unknown>,
    context: Record<string, unknown>,
    handoff: { source: string; target: string },
  ): Promise<string[]> {
    const result = await dispatchHooks(
      this.hooks,
      "onHandoff",
      this.hookInvocation(arguments_, context, { handoff }),
      this.hookRuntime,
    );
    return result.additionalContext;
  }

  private async runWithHooks(
    arguments_: Record<string, unknown>,
    context: Record<string, unknown>,
    onChunk: ((chunk: MessageChunk) => void) | undefined,
    run: (
      effectiveArguments: Record<string, unknown>,
      emitChunk?: (chunk: MessageChunk) => void,
    ) => Promise<{ messages: MessageChunk[] }>,
  ): Promise<{ messages: MessageChunk[] }> {
    let effectiveArguments = arguments_;
    let chunkHooks = Promise.resolve();
    let result: { messages: MessageChunk[] };

    try {
      const start = await dispatchHooks(
        this.hooks,
        "onStart",
        this.hookInvocation(effectiveArguments, context),
        this.hookRuntime,
      );
      effectiveArguments = appendHookContext(effectiveArguments, start.additionalContext);
      const emitChunk = onChunk
        ? (chunk: MessageChunk): void => {
            onChunk(chunk);
            chunkHooks = chunkHooks.then(async () => {
              await dispatchHooks(
                this.hooks,
                "onChunk",
                this.hookInvocation(effectiveArguments, context, { chunk }),
                this.hookRuntime,
              );
            });
          }
        : undefined;
      result = await run(effectiveArguments, emitChunk);
      await chunkHooks;
    } catch (error) {
      await this.runFailedEndHook(effectiveArguments, context, error);
      throw error;
    }

    await dispatchHooks(
      this.hooks,
      "onEnd",
      this.hookInvocation(effectiveArguments, context, {
        outcome: { status: "completed", messages: result.messages },
      }),
      this.hookRuntime,
    );
    return result;
  }

  private async runFailedEndHook(
    arguments_: Record<string, unknown>,
    context: Record<string, unknown>,
    error: unknown,
  ): Promise<void> {
    try {
      await dispatchHooks(
        this.hooks,
        "onEnd",
        this.hookInvocation(arguments_, context, {
          outcome: { status: "failed", error: errorMessage(error) },
        }),
        this.hookRuntime,
      );
    } catch (endError) {
      throw new AggregateError(
        [error, endError],
        `Agent "${this.name}" failed and its onEnd hook also failed.`,
      );
    }
  }

  private hookInvocation(
    arguments_: Record<string, unknown>,
    context: Record<string, unknown>,
    extra: Pick<HookInvocation, "chunk" | "handoff" | "outcome"> = {},
  ): Omit<HookInvocation, "event"> {
    return {
      scope: "agent",
      target: { name: this.name },
      arguments: arguments_,
      context,
      ...extra,
    };
  }

  private async ensureMcpConnected(): Promise<void> {
    if (this.mcp) return;
    const mcp = this.createMcpManager();
    this.mcp = mcp;
    mcp.addLocalTools([...this.localTools, ...(this.contextEngine?.tools ?? [])]);
    const isClaudeCodeProfile = this.localTools.some((tool) => tool.name === "Bash");
    if (isClaudeCodeProfile && this.mcpServers.size > 0) {
      for (const [name, config] of this.mcpServers) {
        mcp.startServer(name, config, this.mcpContract(name));
      }
      await Promise.all(
        [...this.requiredMcpServers].map(async (name) => {
          const config = this.mcpServers.get(name);
          if (!config) return;
          try {
            await mcp.addServer(name, config, this.mcpContract(name), {
              timeoutMs: REQUIRED_MCP_CONNECT_TIMEOUT_MS,
            });
          } catch (error) {
            throw requiredMcpError(name, error);
          }
        }),
      );
      mcp.addClaudeMcpDiscoveryTools();
      mcp.addClaudeMcpResourceTools();
      return;
    }
    await Promise.all(
      [...this.mcpServers].map(async ([name, config]) => {
        try {
          await mcp.addServer(name, config, this.mcpContract(name), {
            timeoutMs: REQUIRED_MCP_CONNECT_TIMEOUT_MS,
          });
        } catch (error) {
          if (this.requiredMcpServers.has(name)) throw requiredMcpError(name, error);
          console.warn(`Failed to connect MCP server ${name}: ${error}`);
        }
      }),
    );
    if (isClaudeCodeProfile) {
      mcp.addClaudeMcpResourceTools();
    }
  }

  private mcpContract(name: string): McpServerContract | undefined {
    const binding = this.projectBootstrap;
    if (!binding || binding.capabilityId !== name) return undefined;
    return {
      name: binding.serverName,
      version: binding.serverVersion,
      tools: binding.tools,
      hostOnlyTools: [binding.bootstrapTool],
    };
  }

  private async loadProjectBootstrap(): Promise<ProjectBootstrapSnapshot | undefined> {
    const binding = this.projectBootstrap;
    if (!binding) return undefined;
    try {
      const result = await this.getMcp().callServerTool(
        binding.capabilityId,
        binding.bootstrapTool,
        {
          schemaVersion: 1,
          projectId: binding.project.id,
          projectRoot: binding.project.root,
        },
        { timeoutMs: PROJECT_BOOTSTRAP_TIMEOUT_MS },
      );
      const snapshot = parseProjectBootstrapResult(result, binding.project.id);
      await this.onProjectBootstrap?.(buildProjectBootstrapReceipt(binding, snapshot));
      return snapshot;
    } catch (error) {
      throwIfCurrentRequestCancelled();
      throw new Error(
        `Required Project service "${binding.capabilityId}" bootstrap failed: ${errorMessage(error)}`,
      );
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

  private nativeProtocolContext(
    onUsage?: (usage: ModelTokenUsage) => void,
    instructions = this.instructions,
  ): NativeProtocolContext {
    return {
      agentName: this.name,
      model: this.requiredNativeModel(),
      instructions,
      parameters: this.parameters,
      maxOutputTokens: this.maxOutputTokens,
      apiMode: this.apiMode,
      openai: this.client,
      anthropic: this.anthropicClient,
      providerHostedWebSearch: this.providerHostedWebSearch,
      tools: () => this.mcp?.toolsForNative() ?? [],
      callTool: (name, input, context) => this.getMcp().callTool(name, input, context),
      onUsage,
    };
  }

  private async buildMessages(
    arguments_: Record<string, unknown>,
    instructions = this.instructions,
  ): Promise<ChatMsg[]> {
    const msgs: ChatMsg[] = [];
    const loadInline = createInlineMediaLoader();

    if (instructions) {
      msgs.push({ role: "system", content: instructions });
    }

    const raw = arguments_.messages as
      | Array<{
          role: string;
          content: string | null;
          tool_calls?: unknown[];
          tool_call_id?: string;
          attachments?: MediaAttachment[];
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
          if (m.role === "user" && m.attachments?.length) {
            msgs.push({
              role: "user",
              content: await openAIChatUserContent(m.content ?? "", m.attachments, loadInline),
            });
            continue;
          }
          msgs.push({
            role: m.role,
            content: m.content ?? "",
          });
        }
      }
    }

    return msgs;
  }

  private async compileModelRequest(
    arguments_: Record<string, unknown>,
    runtimeContext: Record<string, unknown>,
    requestId?: string,
    phase: ContextCompilePhase = "preflight",
    toolDefinitions: readonly unknown[] = [],
    baseInstructions = this.instructions,
  ): Promise<{
    arguments: Record<string, unknown>;
    instructions: string;
    compiled?: CompiledContext;
  }> {
    if (!this.contextEngine) {
      return { arguments: arguments_, instructions: this.instructions };
    }
    const resolvedRequestId = requestId ?? resolveContextRequestId(arguments_, runtimeContext);
    const compile =
      phase === "final" && this.contextEngine.finalize
        ? this.contextEngine.finalize.bind(this.contextEngine)
        : this.contextEngine.compile.bind(this.contextEngine);
    const budgetInstructions = this.providerHostedWebSearch
      ? appendProviderHostedWebSearchInstructions(baseInstructions)
      : baseInstructions;
    const hostedWebSearchTool = this.providerHostedWebSearch
      ? providerHostedWebSearchTool(this.apiProtocol)
      : undefined;
    const compiled = await compile({
      requestId: resolvedRequestId,
      agentName: this.name,
      modelVersion: this.requiredNativeModel(),
      instructions: budgetInstructions,
      arguments: arguments_,
      runtimeContext,
      ...(currentRequestSignal() ? { signal: currentRequestSignal() } : {}),
      requestBudget: {
        phase,
        ...(this.contextWindowTokens ? { contextWindowTokens: this.contextWindowTokens } : {}),
        reservedOutputTokens: this.maxOutputTokens,
        source: this.contextWindowSource,
        toolDefinitions: [
          ...(hostedWebSearchTool ? [hostedWebSearchTool] : []),
          ...toolDefinitions,
        ],
      },
    });
    if (phase === "final" || (!this.contextEngine.finalize && !this.projectBootstrap)) {
      await this.contextEngine.onCompiled?.(compiled.manifest);
    }
    const instructions = [baseInstructions.trim(), compiled.context.trim()]
      .filter(Boolean)
      .join("\n\n---\n\n");
    return {
      arguments: currentTurnArguments(arguments_),
      instructions,
      compiled,
    };
  }

  private contextToolDefinitions(): readonly unknown[] {
    if (!this.mcp) return [];
    return this.apiProtocol === "openai_chat"
      ? this.mcp.toolsForOpenai()
      : this.mcp.toolsForNative();
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
    const backend = this.backend;

    let client = this.createAcpClient();
    const prompt = (activeClient: AcpPromptClient, sessionId?: string) =>
      activeClient.prompt(
        {
          command: backend.program,
          args: backend.args ?? [],
          cwd: this.processOptions?.currentDir,
          env: this.processOptions?.env,
          clearEnv: this.processOptions?.clearEnv,
          ...(this.configuredModel ? { model: this.configuredModel } : {}),
          ...(this.configuredReasoningEffort() ? { effort: this.configuredReasoningEffort() } : {}),
          ...(this.acpMode ? { preferredMode: this.acpMode } : {}),
          ...(this.acpPermissionHandler ? { requestPermission: this.acpPermissionHandler } : {}),
          ...(!sessionId && this.onAcpSessionId ? { onSessionId: this.onAcpSessionId } : {}),
        },
        this.buildAcpPrompt(arguments_, !sessionId),
        undefined,
        sessionId,
        onChunk,
      );

    try {
      let result: Awaited<ReturnType<AcpPromptClient["prompt"]>>;
      try {
        result = await prompt(client, this.acpSessionId);
      } catch (error) {
        if (!this.acpSessionId || !(error instanceof AcpSessionUnavailableError)) throw error;
        await this.onAcpSessionId?.(undefined);
        client = this.createAcpClient();
        result = await prompt(client);
      }
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

  private buildAcpPrompt(
    arguments_: Record<string, unknown>,
    includeHistory: boolean,
  ): AcpPromptInput {
    const request = latestUserContent(arguments_);
    const history = includeHistory ? acpConversationHistory(arguments_) : "";
    const requestText = history
      ? `Conversation history from the canonical SwarmX Session:\n${history}\n\nCurrent user request:\n${request}`
      : `User request:\n${request}`;
    const text = this.instructions.trim()
      ? `Agent instructions:\n${this.instructions.trim()}\n\n${requestText}`
      : history
        ? requestText
        : request;
    const attachments = latestUserAttachments(arguments_);
    return { text, ...(attachments.length > 0 ? { attachments } : {}) };
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

function isOfficialDeepseekAnthropicEndpoint(value: string): boolean {
  try {
    const url = new URL(value);
    return (
      url.protocol === "https:" &&
      url.hostname === "api.deepseek.com" &&
      url.port === "" &&
      url.username === "" &&
      url.password === "" &&
      url.search === "" &&
      url.hash === "" &&
      url.pathname.replace(/\/+$/, "") === "/anthropic"
    );
  } catch {
    return false;
  }
}

function isOfficialHostedResponsesEndpoint(value: string): boolean {
  try {
    const url = new URL(value);
    if (
      url.protocol !== "https:" ||
      url.port !== "" ||
      url.username !== "" ||
      url.password !== "" ||
      url.search !== "" ||
      url.hash !== ""
    ) {
      return false;
    }
    const pathname = url.pathname.replace(/\/+$/, "");
    return (
      (url.hostname === "api.deepseek.com" && (pathname === "" || pathname === "/v1")) ||
      (url.hostname === "api.openai.com" && pathname === "/v1") ||
      (url.hostname === "chatgpt.com" && pathname === "/backend-api/codex")
    );
  } catch {
    return false;
  }
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

function contextWindowSource(value: unknown): ContextWindowSource {
  return value === "model" || value === "supply" || value === "client" ? value : "client";
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

function currentTurnArguments(arguments_: Record<string, unknown>): Record<string, unknown> {
  if (!Array.isArray(arguments_.messages)) return arguments_;
  const messages = arguments_.messages as Array<{ role?: unknown }>;
  let latestUserIndex = -1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "user") {
      latestUserIndex = index;
      break;
    }
  }
  if (latestUserIndex <= 0) return arguments_;
  return { ...arguments_, messages: messages.slice(latestUserIndex) };
}

function acpConversationHistory(arguments_: Record<string, unknown>): string {
  const raw = arguments_.messages as
    | Array<{
        role: string;
        content: string | null;
        attachments?: MediaAttachment[];
      }>
    | undefined;
  const messages = raw ?? [];
  let latestUserIndex = -1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "user") {
      latestUserIndex = index;
      break;
    }
  }
  if (latestUserIndex <= 0) return "";

  return messages
    .slice(0, latestUserIndex)
    .filter((message) => ["user", "assistant", "system", "tool"].includes(message.role))
    .map((message) => {
      const attachments = (message.attachments ?? [])
        .map(
          (attachment) =>
            `${attachment.name} (${attachment.mimeType}, ${attachment.sizeBytes} bytes)`,
        )
        .join(", ");
      const attachmentLine = attachments ? `\nAttachments (metadata only): ${attachments}` : "";
      return `[${message.role}]\n${message.content ?? ""}${attachmentLine}`;
    })
    .join("\n\n");
}

function latestUserAttachments(arguments_: Record<string, unknown>): MediaAttachment[] {
  const raw = arguments_.messages as
    | Array<{
        role: string;
        attachments?: MediaAttachment[];
      }>
    | undefined;
  for (const message of [...(raw ?? [])].reverse()) {
    if (message.role === "user") return message.attachments ?? [];
  }
  return [];
}

async function openAIChatUserContent(
  text: string,
  attachments: readonly MediaAttachment[],
  loadInline: InlineMediaLoader,
): Promise<OpenAI.Chat.Completions.ChatCompletionContentPart[]> {
  const validatedAttachments = validateMediaAttachments(attachments);
  const content: OpenAI.Chat.Completions.ChatCompletionContentPart[] = [];
  if (text) content.push({ type: "text", text });
  for (const attachment of validatedAttachments) {
    if (
      attachment.kind === "video" ||
      (attachment.kind === "audio" &&
        attachment.mimeType !== "audio/mpeg" &&
        attachment.mimeType !== "audio/wav")
    ) {
      content.push({ type: "text", text: attachmentFallbackText(attachment) });
      continue;
    }
    const loaded = await loadInline(attachment);
    if (attachment.kind === "image") {
      content.push({
        type: "image_url",
        image_url: {
          url: `data:${attachment.mimeType};base64,${loaded.base64}`,
          detail: "auto",
        },
      });
      continue;
    }
    if (attachment.kind === "audio") {
      content.push({
        type: "input_audio",
        input_audio: {
          data: loaded.base64,
          format: attachment.mimeType === "audio/wav" ? "wav" : "mp3",
        },
      });
      continue;
    }
    content.push({
      type: "file",
      file: {
        filename: attachment.name,
        file_data: `data:${attachment.mimeType};base64,${loaded.base64}`,
      },
    });
  }
  return content;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function resolveContextRequestId(
  arguments_: Record<string, unknown>,
  runtimeContext: Record<string, unknown>,
): string {
  return (
    stringProperty(arguments_, "requestId") ??
    stringProperty(runtimeContext, "requestId") ??
    randomUUID()
  );
}

function requiredMcpError(name: string, error: unknown): Error {
  return new Error(`Required MCP server "${name}" is unavailable: ${errorMessage(error)}`);
}

function stringProperty(value: unknown, key: string): string | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const property = (value as Record<string, unknown>)[key];
  return typeof property === "string" && property.length > 0 ? property : undefined;
}

function booleanProperty(value: unknown, key: string): boolean | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const property = (value as Record<string, unknown>)[key];
  return typeof property === "boolean" ? property : undefined;
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

function toolProgressChunk(
  agent: string,
  toolName: string,
  invocationId: string,
  progress: LocalToolProgress,
): MessageChunk {
  return {
    role: "tool",
    content: progress.content,
    kind: "tool_progress",
    toolName,
    agent,
    render: { invocationId, status: "running" },
    ...(progress.structuredContent === undefined
      ? {}
      : { structuredContent: progress.structuredContent }),
  };
}

function requestOptions(): { signal?: AbortSignal } | undefined {
  const signal = currentRequestSignal();
  return signal ? { signal } : undefined;
}

// ── HookRef ──────────────────────────────────────────────────────────────────

export class HookRef extends Hook {}
