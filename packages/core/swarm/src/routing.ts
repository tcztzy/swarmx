import type { SwarmMember } from "./contracts.js";

export interface SwarmModelOptions {
  readonly provider?: string;
  readonly model?: string;
  readonly maxTokens?: number;
}

export function memberAgentOptions(member: SwarmMember): SwarmModelOptions | undefined {
  if (member.modelPolicy.source === "legacy-default") return undefined;
  const { provider, model, maxTokens } = member.modelPolicy;
  return {
    ...(provider ? { provider } : {}),
    ...(model ? { model } : {}),
    ...(maxTokens ? { maxTokens } : {}),
  };
}

/** Reassert durable per-member policy on every activation, including cold resume. */
export function applyMemberModelPolicy<Config extends object>(
  config: Config,
  member: SwarmMember,
): Config & SwarmModelOptions {
  const options = memberAgentOptions(member);
  return options ? { ...config, ...options } : config;
}
