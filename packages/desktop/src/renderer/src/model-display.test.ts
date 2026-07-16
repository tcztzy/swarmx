import { describe, expect, it } from "vitest";
import {
  type ModelDisplayDescriptor,
  compareModelDisplayOrder,
  modelBrandPresentation,
  selectableModelReasoning,
} from "./model-display.js";

function model(modelId: string, label = modelId): ModelDisplayDescriptor {
  return { label, modelId, runtimeModel: modelId };
}

describe("compareModelDisplayOrder", () => {
  it("sorts GPT models by descending version and named variant priority", () => {
    const models = [
      model("gpt-5.6-luna"),
      model("gpt-5.5-sol"),
      model("gpt-5.6-terra"),
      model("gpt-6-luna"),
      model("gpt-5.6-sol"),
    ];

    expect(models.sort(compareModelDisplayOrder).map((item) => item.modelId)).toEqual([
      "gpt-6-luna",
      "gpt-5.6-sol",
      "gpt-5.6-terra",
      "gpt-5.6-luna",
      "gpt-5.5-sol",
    ]);
  });

  it("sorts Claude models by family priority and then descending version", () => {
    const models = [
      model("claude-haiku-6"),
      model("claude-sonnet-7"),
      model("claude-opus-4"),
      model("claude-fable-4"),
      model("claude-opus-5"),
      model("claude-mythos-4"),
      model("claude-sonnet-5"),
    ];

    expect(models.sort(compareModelDisplayOrder).map((item) => item.modelId)).toEqual([
      "claude-mythos-4",
      "claude-fable-4",
      "claude-opus-5",
      "claude-opus-4",
      "claude-sonnet-7",
      "claude-sonnet-5",
      "claude-haiku-6",
    ]);
  });
});

describe("modelBrandPresentation", () => {
  it.each([
    ["deepseek-v4-pro", "deepseek", "./provider-icons/deepseek.svg", "V4 Pro"],
    ["gpt-5.6-sol", "gpt", "./harness-icons/codex.svg", "5.6 Sol"],
    ["claude-fable-5", "claude", "./harness-icons/claude_code.svg", "Fable 5"],
  ] as const)("uses the brand icon and short label for %s", (modelId, brand, iconUrl, label) => {
    expect(modelBrandPresentation(model(modelId))).toEqual({ brand, iconUrl, label });
  });

  it("leaves unrecognized model families unchanged", () => {
    expect(modelBrandPresentation(model("qwen3-coder"))).toBeUndefined();
  });
});

describe("selectableModelReasoning", () => {
  it("hides none and ultra without remapping either value to max", () => {
    expect(
      selectableModelReasoning({
        supportedEfforts: ["none", "low", "medium", "high", "xhigh", "max", "ultra"],
        defaultEffort: "medium",
      }),
    ).toEqual({
      supportedEfforts: ["low", "medium", "high", "xhigh", "max"],
      defaultEffort: "medium",
    });
  });

  it("drops a hidden default instead of treating it as max", () => {
    expect(
      selectableModelReasoning({
        supportedEfforts: ["none", "low", "max", "ultra"],
        defaultEffort: "ultra",
      }),
    ).toEqual({ supportedEfforts: ["low", "max"] });
  });
});
