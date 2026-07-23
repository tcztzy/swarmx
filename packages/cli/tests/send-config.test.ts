import { describe, expect, it } from "vitest";
import { createSendSwarmConfig } from "../src/send-config.js";

describe("send command composition", () => {
  it("V335 serializes the selected Harness, runtime Model, and bootstrap env", () => {
    const config = createSendSwarmConfig({
      harnessId: "claude_code",
      model: "claude-opus-4-6",
      effort: "high",
      env: {},
    });

    expect(config.nodes.agent).toEqual({
      kind: "agent",
      agent: expect.objectContaining({
        model: "claude-opus-4-6",
        backend: expect.objectContaining({ type: "custom", program: "npx" }),
        process: {
          clearEnv: false,
          env: {
            ANTHROPIC_MODEL: "claude-opus-4-6",
            ANTHROPIC_CUSTOM_MODEL_OPTION: "claude-opus-4-6",
            CLAUDE_MODEL_CONFIG: '{"availableModels":["claude-opus-4-6"]}',
          },
        },
        parameters: { reasoning: { effort: "high" } },
      }),
    });
  });

  it("V335 rejects unknown and unsupported Harnesses", () => {
    expect(() => createSendSwarmConfig({ harnessId: "missing", model: "gpt-5" })).toThrow(
      /Unknown harness/,
    );
    expect(() => createSendSwarmConfig({ harnessId: "openclaw", model: "gpt-5" })).toThrow(
      /does not support request-scoped model selection/,
    );
  });

  it("V494 sends a provider-prefixed Pi Model through the pinned ACP adapter", () => {
    const config = createSendSwarmConfig({
      harnessId: "pi",
      model: "anthropic/claude-sonnet-4-20250514",
      effort: "high",
    });

    expect(config.nodes.agent).toEqual({
      kind: "agent",
      agent: expect.objectContaining({
        model: "anthropic/claude-sonnet-4-20250514",
        backend: {
          type: "custom",
          program: "npx",
          args: ["--yes", "pi-acp@0.0.31"],
        },
        parameters: { reasoning: { effort: "high" } },
      }),
    });
  });

  it("V502 sends a configured Kimi model alias through the native ACP entrypoint", () => {
    const config = createSendSwarmConfig({
      harnessId: "kimi",
      model: "kimi-managed",
      effort: "high",
    });

    expect(config.nodes.agent).toEqual({
      kind: "agent",
      agent: expect.objectContaining({
        model: "kimi-managed",
        backend: { type: "custom", program: "kimi", args: ["acp"] },
        parameters: { reasoning: { effort: "high" } },
      }),
    });
  });
});
