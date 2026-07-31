import { describe, expect, it } from "vitest";
import type { ProviderUsageSnapshot } from "../../shared/desktop-api.js";
import { mergeProviderUsageSnapshot } from "./settings-workspace.js";

describe("Provider usage merging", () => {
  it("replaces refreshed rows without dropping untouched Providers or tool accounts", () => {
    const current: ProviderUsageSnapshot = {
      fetchedAt: "before",
      providers: [
        {
          source: "provider",
          sourceId: "deepseek",
          label: "DeepSeek",
          adapterId: "deepseek",
          status: "ready",
          meters: [],
        },
        {
          source: "provider",
          sourceId: "openai",
          label: "OpenAI",
          adapterId: "openai",
          status: "ready",
          meters: [],
        },
      ],
      toolAccounts: [
        {
          source: "tool_account",
          sourceId: "codex",
          label: "Codex",
          adapterId: "codex",
          status: "ready",
          meters: [],
        },
      ],
    };
    const targeted: ProviderUsageSnapshot = {
      fetchedAt: "after",
      providers: [{ ...current.providers[0], detail: "refreshed" }],
      toolAccounts: [],
    };

    expect(mergeProviderUsageSnapshot(current, targeted)).toEqual({
      fetchedAt: "after",
      providers: [{ ...current.providers[0], detail: "refreshed" }, current.providers[1]],
      toolAccounts: current.toolAccounts,
    });
  });
});
