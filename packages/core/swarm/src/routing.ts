import type { AgentOptions } from "@deepseek-ai/dsh-agent";
import type { LlmCallConfig } from "@deepseek-ai/dsh-llm";
import type { SwarmMember } from "./contracts.js";

export function memberAgentOptions(member: SwarmMember): AgentOptions | undefined {
  if (member.modelPolicy.source === "legacy-default") return undefined;
  const { provider, model, maxTokens } = member.modelPolicy;
  return {
    ...(provider ? { provider } : {}),
    ...(model ? { model } : {}),
    ...(maxTokens ? { maxTokens } : {}),
  };
}

/** Reassert durable per-member policy on every activation, including cold resume. */
export function applyMemberModelPolicy(config: LlmCallConfig, member: SwarmMember): LlmCallConfig {
  const options = memberAgentOptions(member);
  return options ? { ...config, ...options } : config;
}
