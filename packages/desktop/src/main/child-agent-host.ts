import { randomUUID } from "node:crypto";
import { mergeModelTokenUsage } from "@swarmx/core";
import type { ChatMessage, MessageChunk, ModelTokenUsage } from "@swarmx/core";
import type { ClaudeAgentInvocation, ClaudeAgentResult } from "./workspace-tools.js";

const MAX_CHILD_AGENTS = 32;
const MAX_HISTORY_MESSAGES = 64;

export interface ClaudeChildAgentExecutionRequest {
  agentId: string;
  root: string;
  messages: ChatMessage[];
}

export interface ClaudeChildAgentExecutionResult {
  messages: MessageChunk[];
  usages: ModelTokenUsage[];
}

export interface ClaudeChildAgentHostOptions {
  parentModel: string;
  root: () => string;
  systemContext: (root: string) => string;
  execute: (request: ClaudeChildAgentExecutionRequest) => Promise<ClaudeChildAgentExecutionResult>;
}

export class ClaudeChildAgentHost {
  readonly #options: ClaudeChildAgentHostOptions;
  readonly #sessions = new Map<string, ChatMessage[]>();

  constructor(options: ClaudeChildAgentHostOptions) {
    this.#options = options;
  }

  async run(request: ClaudeAgentInvocation): Promise<ClaudeAgentResult> {
    if (
      request.model &&
      !this.#options.parentModel.toLowerCase().includes(request.model.toLowerCase())
    ) {
      throw new Error(
        `Model override ${request.model} has no configured route in this composition; omit model to inherit ${this.#options.parentModel}.`,
      );
    }
    if (request.subagentType && request.subagentType !== "general-purpose") {
      throw new Error(
        `Specialized agent type ${request.subagentType} is not configured; use general-purpose or omit subagent_type.`,
      );
    }

    const existing = request.resume ? this.#sessions.get(request.resume) : undefined;
    if (request.resume && !existing) {
      throw new Error(`No child agent ${request.resume} exists in the current parent request.`);
    }
    if (!request.resume && this.#sessions.size >= MAX_CHILD_AGENTS) {
      throw new Error(`This request has reached the ${MAX_CHILD_AGENTS} child-agent limit.`);
    }

    const agentId = request.resume ?? randomUUID();
    const root = this.#options.root();
    const systemContext: ChatMessage = {
      role: "system",
      content: this.#options.systemContext(root),
    };
    const history: ChatMessage[] = existing
      ? [systemContext, ...existing.slice(existing[0]?.role === "system" ? 1 : 0)]
      : [systemContext];
    const promptMessage: ChatMessage = { role: "user", content: request.prompt };
    const startedAt = Date.now();
    const execution = await this.#options.execute({
      agentId,
      root,
      messages: [...history, promptMessage],
    });
    const assistantMessages = execution.messages.filter(
      (message) =>
        message.kind === "message" &&
        message.role === "assistant" &&
        message.content.trim().length > 0,
    );
    if (assistantMessages.length === 0) {
      throw new Error("Child agent run ended without a final assistant response.");
    }
    const content = assistantMessages.map((message) => message.content).join("\n");
    const nextHistory: ChatMessage[] = [...history, promptMessage, { role: "assistant", content }];
    this.#sessions.set(
      agentId,
      nextHistory.length <= MAX_HISTORY_MESSAGES
        ? nextHistory
        : [nextHistory[0] as ChatMessage, ...nextHistory.slice(-(MAX_HISTORY_MESSAGES - 1))],
    );

    const usage = mergeModelTokenUsage(execution.usages);
    return {
      status: "completed",
      prompt: request.prompt,
      agentId,
      content: [{ type: "text", text: content }],
      totalToolUseCount: execution.messages.filter((message) => message.kind === "tool_call")
        .length,
      totalDurationMs: Math.max(0, Date.now() - startedAt),
      totalTokens: usage.totalTokens,
      usage: {
        input_tokens: usage.inputTokens,
        output_tokens: usage.outputTokens,
        cache_creation_input_tokens: null,
        cache_read_input_tokens: usage.cachedInputTokens,
        server_tool_use: null,
        service_tier: null,
        cache_creation: null,
      },
    };
  }

  close(): void {
    this.#sessions.clear();
  }
}
