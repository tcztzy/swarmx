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

  it("resolves bare and provider-suffixed harness selectors", () => {
    expect(resolveHarnessSelector("@codex Plan the run", { knownAdapters: ["codex"] })).toEqual({
      matched: true,
      source: "bare_adapter",
      selector: "@codex",
      canonicalSelector: "@codex",
      adapterId: "codex",
      prompt: "Plan the run",
    });

    expect(
      resolveHarnessSelector("@claude:deepseek Inspect the repo", {
        knownAdapters: ["codex", "claude"],
      }),
    ).toMatchObject({
      matched: true,
      source: "adapter_provider",
      selector: "@claude:deepseek",
      canonicalSelector: "@claude:deepseek",
      adapterId: "claude",
      providerProfileId: "deepseek",
      prompt: "Inspect the repo",
    });

    expect(resolveHarnessSelector("No selector", { knownAdapters: ["codex"] })).toEqual({
      matched: false,
      source: "none",
      prompt: "No selector",
    });
  });

  it("resolves named agent aliases to canonical adapter/provider selectors", () => {
    expect(
      resolveHarnessSelector("@deepseek Analyze results", {
        knownAdapters: ["codex", "claude"],
        agentAliases: [
          {
            alias: "deepseek",
            agentProfileId: "analysis-lead",
            adapterId: "claude",
            providerProfileId: "deepseek",
          },
        ],
      }),
    ).toMatchObject({
      matched: true,
      source: "agent_alias",
      selector: "@deepseek",
      canonicalSelector: "@claude:deepseek",
      adapterId: "claude",
      providerProfileId: "deepseek",
      agentProfileId: "analysis-lead",
      prompt: "Analyze results",
    });
  });

  it("fails unknown and ambiguous selectors instead of falling back", () => {
    expect(() =>
      resolveHarnessSelector("@unknown Do work", {
        knownAdapters: ["codex"],
      }),
    ).toThrow(/Unknown harness or agent selector/);

    expect(() =>
      resolveHarnessSelector("@opencode:model Do work", {
        knownAdapters: ["codex"],
      }),
    ).toThrow(/Unknown harness adapter/);

    expect(() =>
      resolveHarnessSelector("@analysis Do work", {
        knownAdapters: ["codex", "claude"],
        agentAliases: [
          { alias: "analysis", adapterId: "codex" },
          { alias: "analysis", adapterId: "claude" },
        ],
      }),
    ).toThrow(/Ambiguous agent selector/);
  });

  it("validates invocation metadata without storing provider secrets", () => {
    expect(
      parseHarnessInvocationMetadata({
        invocationId: "hinv_1",
        sessionId: "conv_1",
        triggerMessageId: "msg_1",
        contextPacketId: "ctx_1",
        adapterId: "claude",
        providerProfileId: "deepseek",
        agentProfileId: "analysis-lead",
        canonicalSelector: "@claude:deepseek",
        externalSessionRef: "claude-session-1",
        status: "completed",
        startedAt: "2026-07-03T00:00:00.000Z",
        endedAt: "2026-07-03T00:01:00.000Z",
      }),
    ).toMatchObject({
      adapterId: "claude",
      providerProfileId: "deepseek",
      canonicalSelector: "@claude:deepseek",
      status: "completed",
    });

    expect(() =>
      parseHarnessInvocationMetadata({
        invocationId: "hinv_secret",
        adapterId: "codex",
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
  });
});
