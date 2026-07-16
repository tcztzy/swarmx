import { describe, expect, it } from "vitest";
import {
  parseHarnessDiscoveryRecord,
  parseHarnessInvocationMetadata,
  resolveHarnessSelector,
} from "../src/harness-management.js";

describe("harness management primitives", () => {
  it("parses discovery records without executing host probes", () => {
    expect(
      parseHarnessDiscoveryRecord({
        adapterId: "codex",
        displayName: "Codex",
        availability: "available",
        host: "local",
        commandPath: "/opt/homebrew/bin/codex",
        version: "0.22.0",
        installable: true,
        dependencyId: "codex-cli",
        statusNote: "ready",
      }),
    ).toMatchObject({
      adapterId: "codex",
      displayName: "Codex",
      availability: "available",
      installable: true,
      dependencyId: "codex-cli",
    });
  });

  it("resolves bare and Model-suffixed harness selectors", () => {
    expect(resolveHarnessSelector("@codex Plan the run", { knownHarnesses: ["codex"] })).toEqual({
      matched: true,
      source: "bare_harness",
      selector: "@codex",
      canonicalSelector: "@codex",
      harnessId: "codex",
      prompt: "Plan the run",
    });

    expect(
      resolveHarnessSelector("@claude:claude-sonnet-4-6 Inspect the repo", {
        knownHarnesses: ["codex", "claude"],
      }),
    ).toMatchObject({
      matched: true,
      source: "harness_model",
      selector: "@claude:claude-sonnet-4-6",
      canonicalSelector: "@claude:claude-sonnet-4-6",
      harnessId: "claude",
      modelId: "claude-sonnet-4-6",
      prompt: "Inspect the repo",
    });

    expect(resolveHarnessSelector("No selector", { knownHarnesses: ["codex"] })).toEqual({
      matched: false,
      source: "none",
      prompt: "No selector",
    });
  });

  it("resolves named agent aliases to canonical Harness-Model selectors", () => {
    expect(
      resolveHarnessSelector("@deepseek Analyze results", {
        knownHarnesses: ["codex", "claude"],
        agentAliases: [
          {
            alias: "deepseek",
            agentProfileId: "analysis-lead",
            harnessId: "claude",
            modelId: "deepseek-v4-pro",
            modelSupplyId: "deepseek-cloud",
          },
        ],
      }),
    ).toMatchObject({
      matched: true,
      source: "agent_alias",
      selector: "@deepseek",
      canonicalSelector: "@claude:deepseek-v4-pro",
      harnessId: "claude",
      modelId: "deepseek-v4-pro",
      modelSupplyId: "deepseek-cloud",
      agentProfileId: "analysis-lead",
      prompt: "Analyze results",
    });
  });

  it("fails unknown and ambiguous selectors instead of falling back", () => {
    expect(() =>
      resolveHarnessSelector("@unknown Do work", {
        knownHarnesses: ["codex"],
      }),
    ).toThrow(/Unknown harness or agent selector/);

    expect(() =>
      resolveHarnessSelector("@opencode:model Do work", {
        knownHarnesses: ["codex"],
      }),
    ).toThrow(/Unknown harness/);

    expect(() =>
      resolveHarnessSelector("@analysis Do work", {
        knownHarnesses: ["codex", "claude"],
        agentAliases: [
          { alias: "analysis", harnessId: "codex", modelId: "gpt-5" },
          { alias: "analysis", harnessId: "claude", modelId: "claude-sonnet-4-6" },
        ],
      }),
    ).toThrow(/Ambiguous agent selector/);
  });

  it("validates Harness-Model invocation metadata without Provider identity", () => {
    expect(
      parseHarnessInvocationMetadata({
        invocationId: "hinv_1",
        sessionId: "conv_1",
        triggerMessageId: "msg_1",
        contextPacketId: "ctx_1",
        harnessId: "claude",
        modelId: "deepseek-v4-pro",
        modelSupplyId: "deepseek-cloud",
        agentProfileId: "analysis-lead",
        canonicalSelector: "@claude:deepseek-v4-pro",
        externalSessionRef: "claude-session-1",
        status: "completed",
        startedAt: "2026-07-03T00:00:00.000Z",
        endedAt: "2026-07-03T00:01:00.000Z",
      }),
    ).toMatchObject({
      harnessId: "claude",
      modelId: "deepseek-v4-pro",
      modelSupplyId: "deepseek-cloud",
      canonicalSelector: "@claude:deepseek-v4-pro",
      status: "completed",
    });

    expect(() =>
      parseHarnessInvocationMetadata({
        invocationId: "hinv_secret",
        harnessId: "codex",
        modelId: "gpt-5",
        status: "started",
        metadata: {
          apiKey: "sk-test",
        },
      }),
    ).toThrow(/inline secret field.*apiKey/);

    expect(() =>
      parseHarnessDiscoveryRecord({
        adapterId: "codex",
        displayName: "Codex",
        availability: "available",
        metadata: {
          secretRef: "keychain:codex",
        },
      }),
    ).not.toThrow();

    expect(() =>
      parseHarnessInvocationMetadata({
        invocationId: "hinv_legacy_provider",
        harnessId: "claude",
        modelId: "deepseek-v4-pro",
        providerProfileId: "deepseek",
        status: "started",
      }),
    ).toThrow(/providerProfileId.*invalid/);
  });
});
