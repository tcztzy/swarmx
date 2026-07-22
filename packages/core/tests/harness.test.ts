import { describe, expect, it } from "vitest";
import {
  HARNESSES,
  getHarness,
  getHarnessList,
  harnessModelRuntimeEnv,
  harnessModelRuntimeModel,
} from "../src/harness.js";

describe("harness registry", () => {
  it("models model control independently of Provider", () => {
    expect(getHarness("swarmx")).toMatchObject({
      modelControl: "direct",
      modelCompatibility: "declared_apis",
      supportedModelApis: ["anthropic", "openai_responses", "openai_chat"],
    });
    expect(getHarness("swarmx")).not.toHaveProperty("compatibleProviders");
    expect(getHarness("swarmx")).not.toHaveProperty("modelSelection");
  });

  it.each(["claude_code", "codex", "pi", "opencode", "hermes"])(
    "%s exposes request-scoped ACP session model control",
    (id) => {
      expect(getHarness(id)).toMatchObject({
        modelControl: "session",
        modelCompatibility: "any",
      });
    },
  );

  it("V309 V494 launches built-in ACP adapters through pinned Node.js npx packages", () => {
    for (const harnessId of ["claude_code", "codex", "pi"]) {
      const backend = getHarness(harnessId)?.backend;
      expect(backend).toMatchObject({ type: "custom", program: "npx" });
      expect(backend?.type === "custom" ? backend.args : []).toContain("--yes");
      expect(backend?.type === "custom" ? backend.args : []).not.toContain("bun");
    }
    expect(getHarness("pi")?.backend).toEqual({
      type: "custom",
      program: "npx",
      args: ["--yes", "pi-acp@0.0.31"],
    });
    expect(getHarness("pi")?.passthroughEnv).toEqual(
      expect.arrayContaining(["HOME", "PI_CODING_AGENT_DIR", "PI_CODING_AGENT_SESSION_DIR"]),
    );
  });

  it("keeps unsupported model control explicit", () => {
    expect(getHarness("openclaw")).toMatchObject({
      modelControl: "unsupported",
      supportedModelApis: [],
    });
  });

  it("builds request-scoped bootstrap config without global file paths", () => {
    expect(harnessModelRuntimeEnv("claude_code", { modelId: "claude-opus-4-6" })).toEqual({
      ANTHROPIC_MODEL: "claude-opus-4-6",
      ANTHROPIC_CUSTOM_MODEL_OPTION: "claude-opus-4-6",
      CLAUDE_MODEL_CONFIG: '{"availableModels":["claude-opus-4-6"]}',
    });
    expect(
      harnessModelRuntimeEnv("codex", {
        modelId: "gpt-5.6",
        runtimeModel: "gateway/gpt-5.6",
        effort: "high",
      }),
    ).toEqual({
      CODEX_CONFIG: '{"model":"gateway/gpt-5.6","model_reasoning_effort":"high"}',
    });
    expect(harnessModelRuntimeEnv("opencode", { modelId: "anthropic/claude" })).toEqual({
      OPENCODE_CONFIG_CONTENT: '{"model":"anthropic/claude"}',
    });
    expect(() => harnessModelRuntimeEnv("openclaw", { modelId: "gpt-5" })).toThrow(
      /does not support request-scoped model selection/,
    );
  });

  it("uses the fixed Claude Code route for DeepSeek V4 Pro", () => {
    expect(
      harnessModelRuntimeModel("claude_code", {
        modelId: "deepseek-v4-pro",
        runtimeModel: "deepseek-v4-pro",
      }),
    ).toBe("deepseek-v4-pro[1m]");
    expect(
      harnessModelRuntimeEnv("claude_code", {
        modelId: "deepseek-v4-pro",
        effort: "max",
        env: { DEEPSEEK_API_KEY: "sk-deepseek-runtime" },
      }),
    ).toEqual({
      ANTHROPIC_BASE_URL: "https://api.deepseek.com/anthropic",
      ANTHROPIC_AUTH_TOKEN: "sk-deepseek-runtime",
      ANTHROPIC_MODEL: "deepseek-v4-pro[1m]",
      ANTHROPIC_DEFAULT_OPUS_MODEL: "deepseek-v4-pro[1m]",
      ANTHROPIC_DEFAULT_SONNET_MODEL: "deepseek-v4-pro[1m]",
      ANTHROPIC_DEFAULT_HAIKU_MODEL: "deepseek-v4-flash",
      CLAUDE_CODE_SUBAGENT_MODEL: "deepseek-v4-flash",
      CLAUDE_CODE_EFFORT_LEVEL: "max",
      CLAUDE_MODEL_CONFIG: '{"availableModels":["deepseek-v4-pro[1m]"]}',
    });
    expect(() =>
      harnessModelRuntimeEnv("claude_code", {
        modelId: "deepseek-v4-pro",
        effort: "max",
        env: {},
      }),
    ).toThrow(/requires env secret "DEEPSEEK_API_KEY"/);
  });

  it("filters disabled harnesses without changing registry metadata", () => {
    HARNESSES.openclaw.enabled = false;
    try {
      expect(getHarness("openclaw")).toBeUndefined();
      expect(getHarnessList().some((harness) => harness.label === "OpenClaw")).toBe(false);
    } finally {
      HARNESSES.openclaw.enabled = undefined;
    }
  });
});
