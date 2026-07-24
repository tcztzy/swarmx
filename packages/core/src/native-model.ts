import type Anthropic from "@anthropic-ai/sdk";
import type {
  ContentBlockParam,
  Message,
  MessageCreateParamsBase,
  MessageParam,
  Tool,
  ToolResultBlockParam,
  ToolUseBlock,
} from "@anthropic-ai/sdk/resources/messages/messages";
import type OpenAI from "openai";
import type {
  Response,
  ResponseCustomToolCall,
  ResponseFunctionToolCall,
  ResponseInputItem,
  ResponseOutputItem,
  ResponseReasoningItem,
  Tool as ResponseTool,
} from "openai/resources/responses/responses";
import { currentRequestSignal, throwIfCurrentRequestCancelled } from "./acp.js";
import type {
  LocalToolCallContext,
  LocalToolProgress,
  NativeLocalToolDefinition,
  ToolExecutionResult,
} from "./mcp.js";
import type { ModelApiMode } from "./model-api.js";
import { ModelTokenUsageSchema } from "./types.js";
import type { MessageChunk, ModelTokenUsage } from "./types.js";

const MAX_TOOL_STEPS = 20;
const OPENAI_RESPONSES_EFFORTS = new Set([
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
]);
const ANTHROPIC_EFFORTS = new Set(["low", "medium", "high", "xhigh", "max"]);

export interface NativeProtocolContext {
  agentName: string;
  model: string;
  instructions: string;
  parameters: Record<string, unknown>;
  maxOutputTokens: number;
  apiMode: ModelApiMode;
  openai: OpenAI;
  anthropic: Anthropic;
  tools: NativeLocalToolDefinition[] | (() => NativeLocalToolDefinition[]);
  callTool(
    name: string,
    input: Record<string, unknown> | string,
    context?: LocalToolCallContext,
  ): Promise<ToolExecutionResult>;
  onUsage?: (usage: ModelTokenUsage) => void;
}

export async function callOpenAIResponses(
  context: NativeProtocolContext,
  arguments_: Record<string, unknown>,
  onChunk?: (chunk: MessageChunk) => void,
): Promise<{ messages: MessageChunk[] }> {
  const input = responseInput(arguments_);
  const reasoning = responseReasoning(context.parameters, context.apiMode);
  const allChunks: MessageChunk[] = [];

  for (let step = 0; step < MAX_TOOL_STEPS; step++) {
    throwIfCurrentRequestCancelled();
    const tools = responseTools(currentNativeTools(context.tools));
    const request = {
      model: context.model,
      input,
      ...(context.instructions.trim() ? { instructions: context.instructions } : {}),
      ...(reasoning ? { reasoning } : {}),
      ...(tools.length > 0 ? { tools } : {}),
      ...(context.apiMode === "codex_responses"
        ? {
            store: false,
            include: ["reasoning.encrypted_content" as const],
            ...(tools.length > 0
              ? { tool_choice: "auto" as const, parallel_tool_calls: true }
              : {}),
          }
        : {}),
    };
    const mustStream = context.apiMode === "codex_responses";
    const response =
      onChunk || mustStream
        ? await streamOpenAIResponse(context, request, onChunk ?? (() => {}))
        : await context.openai.responses.create(request, requestOptions());
    throwIfCurrentRequestCancelled();

    const responseChunks = responseChunksFromOutput(context.agentName, response.output);
    reportOpenAIResponsesUsage(context, response);
    const stepChunks =
      responseChunks.length > 0
        ? responseChunks
        : // A compatible endpoint may stream deltas without returning reconstructed output.
          streamedFallbackChunks(context.agentName, response);
    allChunks.push(...stepChunks);

    const toolCalls = response.output.filter(isResponseToolCall);
    if (toolCalls.length === 0) {
      if (stepChunks.some(isFinalAssistantChunk)) break;
      input.push(...responseReplayItems(response.output, context.apiMode));
      continue;
    }

    const outputs: ResponseInputItem[] = [];
    for (const call of toolCalls) {
      const name = responseToolCallName(call);
      const input = responseToolCallInput(call);
      const serializedInput = typeof input === "string" ? input : JSON.stringify(input);
      const invocationId = call.call_id;
      const toolCallChunk = toolCallMessage(context.agentName, name, serializedInput, invocationId);
      onChunk?.(toolCallChunk);
      allChunks.push(toolCallChunk);

      const result = await executeTool(context, name, input, invocationId, onChunk);
      const toolResultChunk = toolResultMessage(
        context.agentName,
        name,
        result.output,
        result.structuredContent,
        result.failed,
        invocationId,
      );
      onChunk?.(toolResultChunk);
      allChunks.push(toolResultChunk);
      outputs.push(
        call.type === "custom_tool_call"
          ? {
              type: "custom_tool_call_output",
              call_id: call.call_id,
              output: result.output,
            }
          : {
              type: "function_call_output",
              call_id: call.call_id,
              output: result.output,
            },
      );
    }
    input.push(...responseReplayItems(response.output, context.apiMode), ...outputs);
  }

  return { messages: allChunks };
}

function isFinalAssistantChunk(chunk: MessageChunk): boolean {
  return chunk.kind === "message" && chunk.role === "assistant" && chunk.content.trim().length > 0;
}

export async function callAnthropicMessages(
  context: NativeProtocolContext,
  arguments_: Record<string, unknown>,
  onChunk?: (chunk: MessageChunk) => void,
): Promise<{ messages: MessageChunk[] }> {
  const built = anthropicInput(context.instructions, arguments_);
  const messages = built.messages;
  const outputConfig = anthropicOutputConfig(context.parameters);
  const allChunks: MessageChunk[] = [];

  for (let step = 0; step < MAX_TOOL_STEPS; step++) {
    throwIfCurrentRequestCancelled();
    const tools = anthropicTools(currentNativeTools(context.tools));
    const request: MessageCreateParamsBase = {
      model: context.model,
      max_tokens: context.maxOutputTokens,
      messages,
      ...(built.system ? { system: built.system } : {}),
      ...(outputConfig ? { output_config: outputConfig } : {}),
      ...(tools.length > 0 ? { tools } : {}),
    };
    const response = onChunk
      ? await streamAnthropicMessage(context, request, onChunk)
      : await context.anthropic.messages.create({ ...request, stream: false }, requestOptions());
    throwIfCurrentRequestCancelled();
    reportAnthropicUsage(context, response);

    allChunks.push(...anthropicResponseChunks(context.agentName, response));
    const toolCalls = response.content.filter(isAnthropicToolUse);
    if (toolCalls.length === 0) break;

    messages.push({
      role: "assistant",
      content: response.content as unknown as ContentBlockParam[],
    });
    const results: ToolResultBlockParam[] = [];
    for (const call of toolCalls) {
      const serializedInput = jsonString(call.input);
      const invocationId = call.id;
      const toolCallChunk = toolCallMessage(
        context.agentName,
        call.name,
        serializedInput,
        invocationId,
      );
      onChunk?.(toolCallChunk);
      allChunks.push(toolCallChunk);

      const result = await executeTool(
        context,
        call.name,
        objectValue(call.input),
        invocationId,
        onChunk,
      );
      const toolResultChunk = toolResultMessage(
        context.agentName,
        call.name,
        result.output,
        result.structuredContent,
        result.failed,
        invocationId,
      );
      onChunk?.(toolResultChunk);
      allChunks.push(toolResultChunk);
      results.push({
        type: "tool_result",
        tool_use_id: call.id,
        content: result.output,
        ...(result.failed ? { is_error: true } : {}),
      });
    }
    messages.push({ role: "user", content: results });
  }

  return { messages: allChunks };
}

async function streamOpenAIResponse(
  context: NativeProtocolContext,
  request: Parameters<OpenAI["responses"]["create"]>[0],
  onChunk: (chunk: MessageChunk) => void,
): Promise<Response> {
  const stream = await context.openai.responses.create(
    { ...request, stream: true },
    requestOptions(),
  );
  let completed: Response | undefined;
  let streamedText = "";
  let streamedReasoning = "";
  const streamedOutputItems = new Map<number, ResponseOutputItem>();
  for await (const event of stream) {
    throwIfCurrentRequestCancelled();
    if (event.type === "response.output_text.delta") {
      streamedText += event.delta;
      onChunk(messageChunk(context.agentName, event.delta));
    } else if (
      event.type === "response.reasoning_text.delta" ||
      event.type === "response.reasoning_summary_text.delta"
    ) {
      streamedReasoning += event.delta;
      onChunk(thinkingChunk(context.agentName, event.delta));
    } else if (
      event.type === "response.output_item.added" ||
      event.type === "response.output_item.done"
    ) {
      streamedOutputItems.set(event.output_index, event.item);
    } else if (event.type === "response.function_call_arguments.delta") {
      const item = streamedOutputItems.get(event.output_index);
      if (item?.type === "function_call") {
        streamedOutputItems.set(event.output_index, {
          ...item,
          arguments: `${item.arguments}${event.delta}`,
        });
      }
    } else if (event.type === "response.function_call_arguments.done") {
      const item = streamedOutputItems.get(event.output_index);
      if (item?.type === "function_call") {
        streamedOutputItems.set(event.output_index, { ...item, arguments: event.arguments });
      }
    } else if (event.type === "response.completed") {
      completed = responseWithStreamedOutputItems(event.response, streamedOutputItems);
    } else if (event.type === "response.failed") {
      throw new Error(responseFailureMessage(event.response));
    } else if (event.type === "error") {
      throw new Error(event.message);
    }
  }
  if (!completed) throw new Error("OpenAI Responses stream ended without response.completed.");
  return Object.assign(completed, {
    __swarmxStreamedText: streamedText,
    __swarmxStreamedReasoning: streamedReasoning,
  });
}

function responseWithStreamedOutputItems(
  response: Response,
  streamedOutputItems: ReadonlyMap<number, ResponseOutputItem>,
): Response {
  if (streamedOutputItems.size === 0) return response;
  const output = [...response.output];
  const existingKeys = new Set(output.map(responseOutputItemKey));
  for (const [index, item] of [...streamedOutputItems].sort(([left], [right]) => left - right)) {
    const key = responseOutputItemKey(item);
    if (existingKeys.has(key)) continue;
    output.splice(Math.min(index, output.length), 0, item);
    existingKeys.add(key);
  }
  return { ...response, output };
}

function responseOutputItemKey(item: ResponseOutputItem): string {
  if ("id" in item && typeof item.id === "string") return `${item.type}:${item.id}`;
  if (item.type === "function_call") return `${item.type}:${item.call_id}`;
  return `${item.type}:${JSON.stringify(item)}`;
}

async function streamAnthropicMessage(
  context: NativeProtocolContext,
  request: MessageCreateParamsBase,
  onChunk: (chunk: MessageChunk) => void,
): Promise<Message> {
  const stream = context.anthropic.messages.stream(request, requestOptions());
  for await (const event of stream) {
    throwIfCurrentRequestCancelled();
    if (event.type !== "content_block_delta") continue;
    if (event.delta.type === "text_delta") {
      onChunk(messageChunk(context.agentName, event.delta.text));
    } else if (event.delta.type === "thinking_delta") {
      onChunk(thinkingChunk(context.agentName, event.delta.thinking));
    }
  }
  const message = await stream.finalMessage();
  throwIfCurrentRequestCancelled();
  return message;
}

function responseInput(arguments_: Record<string, unknown>): ResponseInputItem[] {
  return rawMessages(arguments_).flatMap((message): ResponseInputItem[] => {
    if (message.role === "tool" && message.tool_call_id) {
      return [
        {
          type: "function_call_output",
          call_id: message.tool_call_id,
          output: message.content ?? "",
        },
      ];
    }
    if (!["user", "assistant", "system"].includes(message.role)) return [];
    return [
      {
        role: message.role as "user" | "assistant" | "system",
        content: message.content ?? "",
      },
    ];
  });
}

function anthropicInput(
  instructions: string,
  arguments_: Record<string, unknown>,
): { system?: string; messages: MessageParam[] } {
  const system = [
    instructions.trim() || undefined,
    ...rawMessages(arguments_)
      .filter((message) => message.role === "system")
      .map((message) => message.content?.trim() || undefined),
  ]
    .filter((value): value is string => !!value)
    .join("\n\n");
  const messages = rawMessages(arguments_).flatMap((message): MessageParam[] => {
    if (message.role === "tool" && message.tool_call_id) {
      return [
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: message.tool_call_id,
              content: message.content ?? "",
            },
          ],
        },
      ];
    }
    if (message.role !== "user" && message.role !== "assistant") return [];
    return [{ role: message.role, content: message.content ?? "" }];
  });
  return { ...(system ? { system } : {}), messages };
}

function currentNativeTools(tools: NativeProtocolContext["tools"]): NativeLocalToolDefinition[] {
  return typeof tools === "function" ? tools() : tools;
}

function responseTools(tools: NativeLocalToolDefinition[]): ResponseTool[] {
  return tools.map((tool): ResponseTool => {
    if (tool.type === "custom") {
      return {
        type: "custom",
        name: tool.name,
        description: tool.description,
        ...(tool.format ? { format: tool.format } : {}),
      } as ResponseTool;
    }
    return {
      type: "function",
      name: tool.function.name,
      description: tool.function.description,
      parameters: tool.function.parameters,
      strict: false,
    };
  });
}

function anthropicTools(tools: NativeLocalToolDefinition[]): Tool[] {
  return tools.filter(isNativeFunctionTool).map((tool) => ({
    name: tool.function.name,
    description: tool.function.description,
    input_schema: tool.function.parameters as Tool.InputSchema,
  }));
}

function responseReasoning(
  parameters: Record<string, unknown>,
  apiMode: ModelApiMode,
): OpenAI.Reasoning | undefined {
  const effort = mappedReasoningEffort(parameters, "openai.responses", "reasoning.effort");
  const mappedEffort =
    effort && OPENAI_RESPONSES_EFFORTS.has(effort) ? (effort as OpenAI.ReasoningEffort) : undefined;
  if (apiMode === "codex_responses") {
    return { ...(mappedEffort ? { effort: mappedEffort } : {}), summary: "auto" };
  }
  return mappedEffort ? { effort: mappedEffort } : undefined;
}

function anthropicOutputConfig(
  parameters: Record<string, unknown>,
): MessageCreateParamsBase["output_config"] | undefined {
  const effort = mappedReasoningEffort(parameters, "anthropic.messages", "output_config.effort");
  if (!effort || !ANTHROPIC_EFFORTS.has(effort)) return undefined;
  return { effort: effort as "low" | "medium" | "high" | "xhigh" | "max" };
}

function mappedReasoningEffort(
  parameters: Record<string, unknown>,
  api: string,
  path: string,
): string | undefined {
  const reasoning = objectValue(parameters.reasoning);
  const mapping = objectValue(reasoning.parameterMapping);
  return reasoning.control === "effort_enum" &&
    mapping.api === api &&
    mapping.path === path &&
    typeof reasoning.effort === "string"
    ? reasoning.effort
    : undefined;
}

function responseChunksFromOutput(agentName: string, output: ResponseOutputItem[]): MessageChunk[] {
  return output.flatMap((item): MessageChunk[] => {
    if (item.type === "message") {
      return item.content.flatMap((content): MessageChunk[] => {
        if (content.type === "output_text" && content.text) {
          return [messageChunk(agentName, content.text)];
        }
        if (content.type === "refusal" && content.refusal) {
          return [messageChunk(agentName, content.refusal)];
        }
        return [];
      });
    }
    if (item.type === "reasoning") {
      const text = reasoningText(item);
      return text ? [thinkingChunk(agentName, text)] : [];
    }
    return [];
  });
}

function anthropicResponseChunks(agentName: string, response: Message): MessageChunk[] {
  return response.content.flatMap((block): MessageChunk[] => {
    if (block.type === "text" && block.text) return [messageChunk(agentName, block.text)];
    if (block.type === "thinking" && block.thinking) {
      return [thinkingChunk(agentName, block.thinking)];
    }
    return [];
  });
}

function reasoningText(item: ResponseReasoningItem): string {
  const content = item.content?.map((part) => part.text).filter(Boolean) ?? [];
  const summary = item.summary.map((part) => part.text).filter(Boolean);
  return (content.length > 0 ? content : summary).join("\n");
}

function streamedFallbackChunks(agentName: string, response: Response): MessageChunk[] {
  const record = response as Response & {
    __swarmxStreamedText?: string;
    __swarmxStreamedReasoning?: string;
  };
  return [
    record.__swarmxStreamedReasoning
      ? thinkingChunk(agentName, record.__swarmxStreamedReasoning)
      : undefined,
    record.__swarmxStreamedText ? messageChunk(agentName, record.__swarmxStreamedText) : undefined,
  ].filter((chunk): chunk is MessageChunk => !!chunk);
}

async function executeTool(
  context: NativeProtocolContext,
  name: string,
  input: Record<string, unknown> | string,
  invocationId: string,
  onChunk?: (chunk: MessageChunk) => void,
): Promise<{ output: string; structuredContent?: unknown; failed: boolean }> {
  try {
    throwIfCurrentRequestCancelled();
    const result = normalizeToolExecutionResult(
      await context.callTool(name, input, {
        invocationId,
        ...(onChunk
          ? {
              onProgress: (progress: LocalToolProgress) =>
                onChunk(toolProgressMessage(context.agentName, name, progress, invocationId)),
            }
          : {}),
      }),
    );
    throwIfCurrentRequestCancelled();
    return {
      output: result.content,
      ...(result.structuredContent === undefined
        ? {}
        : { structuredContent: result.structuredContent }),
      failed: result.isError,
    };
  } catch (error) {
    throwIfCurrentRequestCancelled();
    const structuredContent = { error: error instanceof Error ? error.message : String(error) };
    return { output: JSON.stringify(structuredContent), structuredContent, failed: true };
  }
}

function normalizeToolExecutionResult(result: unknown): ToolExecutionResult {
  if (
    result &&
    typeof result === "object" &&
    !Array.isArray(result) &&
    typeof (result as { content?: unknown }).content === "string"
  ) {
    const record = result as Partial<ToolExecutionResult> & { content: string };
    return {
      content: record.content,
      ...(record.structuredContent === undefined
        ? {}
        : { structuredContent: record.structuredContent }),
      isError: record.isError === true,
    };
  }
  const structuredContent =
    result && typeof result === "object" && !Array.isArray(result) ? result : { result };
  return {
    content: JSON.stringify(structuredContent),
    structuredContent,
    isError: false,
  };
}

function rawMessages(arguments_: Record<string, unknown>): Array<{
  role: string;
  content: string | null;
  tool_call_id?: string;
}> {
  const messages = arguments_.messages;
  if (!Array.isArray(messages)) return [];
  return messages.flatMap((message) => {
    if (!message || typeof message !== "object" || Array.isArray(message)) return [];
    const record = message as Record<string, unknown>;
    if (typeof record.role !== "string") return [];
    return [
      {
        role: record.role,
        content: typeof record.content === "string" ? record.content : null,
        ...(typeof record.tool_call_id === "string" ? { tool_call_id: record.tool_call_id } : {}),
      },
    ];
  });
}

function isResponseFunctionCall(item: ResponseOutputItem): item is ResponseFunctionToolCall {
  return item.type === "function_call";
}

function isResponseCustomToolCall(item: ResponseOutputItem): item is ResponseCustomToolCall {
  return item.type === "custom_tool_call";
}

function isResponseToolCall(
  item: ResponseOutputItem,
): item is ResponseFunctionToolCall | ResponseCustomToolCall {
  return isResponseFunctionCall(item) || isResponseCustomToolCall(item);
}

function responseToolCallName(call: ResponseFunctionToolCall | ResponseCustomToolCall): string {
  return call.name;
}

function responseToolCallInput(
  call: ResponseFunctionToolCall | ResponseCustomToolCall,
): Record<string, unknown> | string {
  return call.type === "custom_tool_call" ? call.input : parseObject(call.arguments);
}

function isNativeFunctionTool(
  tool: NativeLocalToolDefinition,
): tool is Extract<NativeLocalToolDefinition, { type: "function" }> {
  return tool.type === "function";
}

function responseReplayItems(
  output: ResponseOutputItem[],
  apiMode: ModelApiMode,
): ResponseInputItem[] {
  if (apiMode !== "codex_responses") return output as ResponseInputItem[];

  return output.flatMap((item): ResponseInputItem[] => {
    if (item.type === "reasoning" && item.encrypted_content) {
      return [
        {
          type: "reasoning",
          encrypted_content: item.encrypted_content,
          summary: item.summary,
        } as unknown as ResponseInputItem,
      ];
    }
    if (item.type === "function_call") {
      return [
        {
          type: "function_call",
          call_id: item.call_id,
          name: item.name,
          arguments: item.arguments,
        } as ResponseInputItem,
      ];
    }
    if (item.type === "custom_tool_call") {
      return [item as unknown as ResponseInputItem];
    }
    if (item.type === "message") {
      const content = item.content
        .flatMap((part) =>
          part.type === "output_text" ? [part.text] : part.type === "refusal" ? [part.refusal] : [],
        )
        .filter(Boolean)
        .join("\n");
      return content ? [{ role: "assistant", content }] : [];
    }
    return [];
  });
}

function isAnthropicToolUse(block: Message["content"][number]): block is ToolUseBlock {
  return block.type === "tool_use";
}

function messageChunk(agent: string, content: string): MessageChunk {
  return { role: "assistant", content, kind: "message", agent };
}

function thinkingChunk(agent: string, content: string): MessageChunk {
  return { role: "assistant", content, kind: "thinking", agent };
}

function toolCallMessage(
  agent: string,
  toolName: string,
  content: string,
  invocationId: string,
): MessageChunk {
  return {
    role: "assistant",
    content,
    kind: "tool_call",
    toolName,
    agent,
    render: { invocationId, status: "running" },
  };
}

function toolProgressMessage(
  agent: string,
  toolName: string,
  progress: LocalToolProgress,
  invocationId: string,
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

function toolResultMessage(
  agent: string,
  toolName: string,
  content: string,
  structuredContent?: unknown,
  failed = false,
  invocationId?: string,
): MessageChunk {
  return {
    role: "tool",
    content,
    kind: "tool_result",
    toolName,
    agent,
    render: {
      ...(invocationId ? { invocationId } : {}),
      status: failed ? "failed" : "succeeded",
    },
    ...(structuredContent === undefined ? {} : { structuredContent }),
  };
}

function parseObject(value: string): Record<string, unknown> {
  try {
    return objectValue(JSON.parse(value));
  } catch {
    return {};
  }
}

function objectValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function reportOpenAIResponsesUsage(context: NativeProtocolContext, response: Response): void {
  const usage = objectValue(response.usage);
  const inputDetails = objectValue(usage.input_tokens_details);
  const outputDetails = objectValue(usage.output_tokens_details);
  const inputTokens = nonnegativeInteger(usage.input_tokens);
  const outputTokens = nonnegativeInteger(usage.output_tokens);
  const totalTokens =
    nonnegativeInteger(usage.total_tokens) ?? (inputTokens ?? 0) + (outputTokens ?? 0);
  if (totalTokens === 0 && inputTokens === undefined && outputTokens === undefined) return;
  context.onUsage?.(
    ModelTokenUsageSchema.parse({
      inputTokens: inputTokens ?? 0,
      outputTokens: outputTokens ?? 0,
      reasoningTokens: nonnegativeInteger(outputDetails.reasoning_tokens) ?? 0,
      cachedInputTokens: nonnegativeInteger(inputDetails.cached_tokens) ?? 0,
      totalTokens,
      estimated: false,
      model: context.model,
      provider: "openai_responses",
    }),
  );
}

function reportAnthropicUsage(context: NativeProtocolContext, response: Message): void {
  const usage = objectValue(response.usage);
  const uncachedInputTokens = nonnegativeInteger(usage.input_tokens) ?? 0;
  const outputTokens = nonnegativeInteger(usage.output_tokens) ?? 0;
  const cachedInputTokens =
    (nonnegativeInteger(usage.cache_creation_input_tokens) ?? 0) +
    (nonnegativeInteger(usage.cache_read_input_tokens) ?? 0);
  const inputTokens = uncachedInputTokens + cachedInputTokens;
  if (inputTokens === 0 && outputTokens === 0 && cachedInputTokens === 0) return;
  context.onUsage?.(
    ModelTokenUsageSchema.parse({
      inputTokens,
      outputTokens,
      reasoningTokens: 0,
      cachedInputTokens,
      totalTokens: inputTokens + outputTokens,
      estimated: false,
      model: context.model,
      provider: "anthropic",
    }),
  );
}

function nonnegativeInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : undefined;
}

function jsonString(value: unknown): string {
  const serialized = JSON.stringify(value);
  return serialized ?? "{}";
}

function responseFailureMessage(response: Response): string {
  const error = response.error;
  return error?.message
    ? `OpenAI Responses request failed: ${error.message}`
    : `OpenAI Responses request ended with status ${response.status}.`;
}

function requestOptions(): { signal?: AbortSignal } | undefined {
  const signal = currentRequestSignal();
  return signal ? { signal } : undefined;
}
