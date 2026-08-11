import { describe, expect, it } from "vitest";
import { BUILTIN_TOOL_CONTRACT_REVISION, resolveBuiltinToolStyle } from "../src/builtin-tools.js";

describe("built-in tool style resolution", () => {
  it("preserves a persisted binding before Settings or Model preferences", () => {
    expect(
      resolveBuiltinToolStyle({
        configuredStyle: "kimi_code",
        modelPreferredStyle: "claude_code",
        sessionBinding: {
          style: "codex",
          revision: BUILTIN_TOOL_CONTRACT_REVISION,
          source: "fallback",
        },
      }),
    ).toEqual({
      style: "codex",
      revision: BUILTIN_TOOL_CONTRACT_REVISION,
      source: "fallback",
    });
  });

  it("resolves concrete Settings, explicit Model metadata, then Codex fallback", () => {
    expect(
      resolveBuiltinToolStyle({
        configuredStyle: "kimi_code",
        modelPreferredStyle: "claude_code",
      }),
    ).toEqual({
      style: "kimi_code",
      revision: BUILTIN_TOOL_CONTRACT_REVISION,
      source: "settings",
    });
    expect(
      resolveBuiltinToolStyle({
        configuredStyle: "auto",
        modelPreferredStyle: "claude_code",
      }),
    ).toEqual({
      style: "claude_code",
      revision: BUILTIN_TOOL_CONTRACT_REVISION,
      source: "model",
    });
    expect(resolveBuiltinToolStyle({ configuredStyle: "auto" })).toEqual({
      style: "codex",
      revision: BUILTIN_TOOL_CONTRACT_REVISION,
      source: "fallback",
    });
  });
});
