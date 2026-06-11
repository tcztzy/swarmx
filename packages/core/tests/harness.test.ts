import { describe, expect, it } from "vitest";
import { HARNESSES, getHarness, getHarnessList, providerEnvVars } from "../src/harness.js";
import type { ModelProvider } from "../src/harness.js";

describe("Harness", () => {
  it("has 6 registered harnesses", () => {
    expect(Object.keys(HARNESSES)).toHaveLength(6);
  });

  it("getHarness returns configured harness", () => {
    const h = getHarness("opencode");
    expect(h).toBeDefined();
    expect(h?.label).toBe("OpenCode");
    expect(h?.backend).toEqual({ type: "custom", program: "opencode", args: ["acp"] });
  });

  it("getHarness returns undefined for unknown", () => {
    expect(getHarness("unknown")).toBeUndefined();
  });

  it("getHarnessList returns all harnesses", () => {
    const list = getHarnessList();
    expect(list).toHaveLength(6);
    const labels = list.map((h) => h.label);
    expect(labels).toContain("Claude Code");
    expect(labels).toContain("Codex");
  });

  it("providerEnvVars sets Anthropic env", () => {
    const provider: ModelProvider = {
      kind: "anthropic",
      apiKey: "sk-ant-test",
      model: "claude-sonnet-4-20250514",
    };
    const env = providerEnvVars(provider);
    expect(env.ANTHROPIC_API_KEY).toBe("sk-ant-test");
    expect(env.OPENAI_MODEL).toBe("claude-sonnet-4-20250514");
  });

  it("providerEnvVars sets OpenAI env", () => {
    const provider: ModelProvider = {
      kind: "openai_chat",
      apiKey: "sk-test",
      baseUrl: "https://api.openai.com",
      model: "gpt-4",
    };
    const env = providerEnvVars(provider);
    expect(env.OPENAI_API_KEY).toBe("sk-test");
    expect(env.OPENAI_BASE_URL).toBe("https://api.openai.com");
    expect(env.OPENAI_MODEL).toBe("gpt-4");
  });

  it("providerEnvVars sets Ollama env", () => {
    const provider: ModelProvider = {
      kind: "ollama",
      baseUrl: "http://localhost:11434",
      model: "llama3",
    };
    const env = providerEnvVars(provider);
    expect(env.OLLAMA_HOST).toBe("http://localhost:11434");
  });

  it("ClaudeCode only compatible with Anthropic", () => {
    const h = getHarness("claude_code");
    expect(h).toBeDefined();
    if (!h) throw new Error("claude_code harness missing");
    expect(h.compatibleProviders).toEqual(["anthropic"]);
    expect(h.backend).toEqual({
      type: "custom",
      program: "bun",
      args: ["x", "--silent", "@agentclientprotocol/claude-agent-acp"],
    });
  });

  it("Codex ACP launches bunx silently", () => {
    const h = getHarness("codex");
    expect(h).toBeDefined();
    if (!h) throw new Error("codex harness missing");
    expect(h.backend).toEqual({
      type: "custom",
      program: "bun",
      args: ["x", "--silent", "@agentclientprotocol/codex-acp"],
    });
  });

  it("SwarmX compatible with Anthropic, OpenAI Chat, Ollama", () => {
    const h = getHarness("swarmx");
    expect(h).toBeDefined();
    if (!h) throw new Error("swarmx harness missing");
    expect(h.compatibleProviders).toContain("anthropic");
    expect(h.compatibleProviders).toContain("openai_chat");
    expect(h.compatibleProviders).toContain("ollama");
  });
});
