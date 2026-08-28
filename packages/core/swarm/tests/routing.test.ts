import { describe, expect, it } from "vitest";
import { swarmMemberSchema } from "../src/contracts.js";
import { applyMemberModelPolicy, memberAgentOptions } from "../src/routing.js";

function member(name: string, modelPolicy: unknown) {
  return swarmMemberSchema.parse({
    id: `session-${name}`,
    name,
    role: "implementer",
    phase: "active",
    description: "Implementation member",
    createdAt: 100,
    modelPolicy,
  });
}

describe("V187 member model routing", () => {
  it("keeps two per-member provider/model/max-token policies distinct", () => {
    const cheap = member("cheap", {
      source: "requested",
      provider: "ollama",
      model: "qwen3:32b",
      maxTokens: 4_096,
    });
    const strong = member("strong", {
      source: "requested",
      provider: "openai",
      model: "gpt-5.6",
      maxTokens: 16_384,
    });
    expect(memberAgentOptions(cheap)).toEqual({
      provider: "ollama",
      model: "qwen3:32b",
      maxTokens: 4_096,
    });
    expect(memberAgentOptions(strong)).toEqual({
      provider: "openai",
      model: "gpt-5.6",
      maxTokens: 16_384,
    });
    expect(applyMemberModelPolicy({ provider: "lead", model: "lead-model" }, cheap)).toMatchObject({
      provider: "ollama",
      model: "qwen3:32b",
      maxTokens: 4_096,
    });
  });

  it("preserves deployment defaults for explicit legacy members", () => {
    const legacy = member("legacy", { source: "legacy-default" });
    const inherited = { provider: "deployment", model: "default", maxTokens: 2_048 };
    expect(memberAgentOptions(legacy)).toBeUndefined();
    expect(applyMemberModelPolicy(inherited, legacy)).toBe(inherited);
  });
});
