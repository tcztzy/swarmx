import { describe, expect, it } from "vitest";
import {
  actionIntentFromDependencyPlan,
  assertActionConfirmed,
  createActionIntent,
  parseActionIntent,
  requiresExplicitConfirmation,
  sanitizeActionPayload,
} from "../src/actions.js";

describe("action intent primitives", () => {
  it("creates deterministic action intents and infers confirmation from risk", () => {
    const input = {
      kind: "plugin.install" as const,
      title: "Install GEEPilot plugin",
      source: { kind: "plugin_catalog" as const, id: "geepilot" },
      target: { kind: "extension" as const, id: "geepilot" },
      confirmationText: "Install the GEEPilot plugin.",
      payload: { host: "codex" },
      createdAt: "2026-07-03T00:00:00.000Z",
    };

    const first = createActionIntent(input);
    const second = createActionIntent(input);

    expect(first.actionId).toBe(second.actionId);
    expect(first.actionId).toMatch(/^act_/);
    expect(first).toMatchObject({
      kind: "plugin.install",
      confirmationRequired: true,
      risks: ["downloads_code", "writes_files", "executes_code"],
    });
  });

  it("requires confirmation for risky actions and accepts confirmed matches", () => {
    const intent = createActionIntent({
      kind: "command.rerun",
      title: "Rerun tests",
      source: { kind: "render_event", id: "rne_test_failure" },
      confirmationText: "Rerun the test command.",
    });

    expect(requiresExplicitConfirmation(intent.risks)).toBe(true);
    expect(() =>
      assertActionConfirmed(intent, {
        actionId: intent.actionId,
        confirmed: false,
      }),
    ).toThrow(/requires explicit confirmation/);
    expect(() =>
      assertActionConfirmed(intent, {
        actionId: "act_other",
        confirmed: true,
      }),
    ).toThrow(/does not match intent/);
    expect(
      assertActionConfirmed(intent, {
        actionId: intent.actionId,
        confirmed: true,
        confirmedAt: "2026-07-03T00:00:00.000Z",
      }),
    ).toMatchObject({
      intent: { actionId: intent.actionId },
      confirmation: { confirmed: true },
    });

    const revealRawPayload = createActionIntent({
      kind: "raw_payload.reveal",
      title: "Reveal raw tool payload",
      source: { kind: "render_event", id: "rne_tool_result" },
      confirmationText: "Reveal the raw host payload.",
    });

    expect(revealRawPayload).toMatchObject({
      confirmationRequired: true,
      risks: ["secrets"],
    });
  });

  it("does not require confirmation for read-only actions", () => {
    const intent = createActionIntent({
      kind: "marketplace.refresh",
      title: "Refresh marketplace metadata",
      source: { kind: "extension", id: "geepilot" },
      risks: ["read_only"],
      confirmationRequired: false,
    });

    expect(intent.confirmationRequired).toBe(false);
    expect(requiresExplicitConfirmation(intent.risks)).toBe(false);
    expect(
      assertActionConfirmed(intent, {
        actionId: intent.actionId,
        confirmed: false,
      }),
    ).toMatchObject({
      intent: { kind: "marketplace.refresh" },
    });
  });

  it("sanitizes payload secrets and rejects unsanitized action records", () => {
    expect(sanitizeActionPayload({ apiKey: "sk-test", nested: { password: "secret" } })).toEqual({
      apiKey: "[redacted]",
      nested: { password: "[redacted]" },
    });

    const intent = createActionIntent({
      kind: "plugin.enable",
      title: "Enable plugin",
      source: { kind: "plugin_catalog", id: "geepilot" },
      confirmationText: "Enable this plugin.",
      payload: {
        apiKey: "sk-test",
        secretRef: "keychain:geepilot",
      },
    });

    expect(intent.payload).toEqual({
      apiKey: "[redacted]",
      secretRef: "keychain:geepilot",
    });
    expect(JSON.stringify(intent)).not.toContain("sk-test");

    expect(() =>
      parseActionIntent({
        actionId: "act_bad",
        kind: "plugin.enable",
        title: "Enable plugin",
        source: { kind: "plugin_catalog", id: "geepilot" },
        confirmationRequired: true,
        confirmationText: "Enable plugin.",
        payload: { apiKey: "sk-test" },
      }),
    ).toThrow(/inline secret field.*apiKey/);
  });

  it("maps dependency install plans to explicit action intents", () => {
    const managed = actionIntentFromDependencyPlan({
      dependencyId: "ripgrep",
      action: "install_managed",
      reason: "missing managed binary",
      platform: "darwin-aarch64",
    });
    const external = actionIntentFromDependencyPlan({
      dependencyId: "codex-cli",
      action: "requires_user_action",
      reason: "vendor installer required",
    });
    const existing = actionIntentFromDependencyPlan({
      dependencyId: "uv",
      action: "use_existing",
      reason: "detected on PATH",
    });
    const unavailable = actionIntentFromDependencyPlan({
      dependencyId: "benchmark-assets",
      action: "unavailable",
      reason: "assets are not installed at startup",
    });

    expect(managed).toMatchObject({
      kind: "dependency.install_managed",
      confirmationRequired: true,
      risks: ["downloads_code", "writes_files", "executes_code"],
      payload: { dependencyId: "ripgrep", platform: "darwin-aarch64" },
    });
    expect(external).toMatchObject({
      kind: "dependency.open_installer",
      confirmationRequired: true,
    });
    expect(existing).toMatchObject({
      kind: "dependency.use_existing",
      confirmationRequired: false,
      risks: ["read_only"],
    });
    expect(unavailable).toMatchObject({
      kind: "dependency.unavailable",
      confirmationRequired: false,
      risks: ["read_only"],
    });
  });
});
