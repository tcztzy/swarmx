import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { ComposerPreferenceService } from "./composer-preferences.js";
import { DesktopSettingsStore } from "./settings-store.js";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true })));
});

describe("ComposerPreferenceService", () => {
  it("persists the latest Harness and one routed Model selection per Harness", async () => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-composer-preferences-"));
    roots.push(root);
    const store = new DesktopSettingsStore({ path: join(root, "settings.json") });
    const service = new ComposerPreferenceService(store);

    await store.update((settings) => ({
      ...settings,
      extensions: { ...settings.extensions, enabledPluginIds: ["paper-tools"] },
    }));
    await service.save({
      harnessId: "codex",
      modelId: "gpt-5.6-sol",
      modelSupplyId: "catalog:codex:gpt-5.6-sol",
      effort: "high",
    });
    await service.save({ harnessId: "swarmx", modelId: "claude-sonnet-5" });
    const preferences = await service.save({ harnessId: "codex" });

    expect(preferences).toEqual({
      lastHarnessId: "codex",
      selectionsByHarness: {
        codex: {
          modelId: "gpt-5.6-sol",
          modelSupplyId: "catalog:codex:gpt-5.6-sol",
          effort: "high",
        },
        swarmx: { modelId: "claude-sonnet-5" },
      },
    });
    expect((await store.read()).extensions.enabledPluginIds).toEqual(["paper-tools"]);
  });

  it("clears stale effort when the user selects a different Model", async () => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-composer-preferences-"));
    roots.push(root);
    const service = new ComposerPreferenceService(
      new DesktopSettingsStore({ path: join(root, "settings.json") }),
    );

    await service.save({ harnessId: "codex", modelId: "gpt-5.4", effort: "high" });
    await expect(
      service.save({ harnessId: "codex", modelSupplyId: "supply-without-model" }),
    ).rejects.toThrow(/require a model id/i);
    await expect(
      service.save({ harnessId: "codex", modelId: "gpt-5.6-sol" }),
    ).resolves.toMatchObject({
      selectionsByHarness: { codex: { modelId: "gpt-5.6-sol" } },
    });
  });
});
