import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { DesktopSettingsStore } from "./settings-store.js";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true })));
});

describe("DesktopSettingsStore", () => {
  it("serializes concurrent section mutations without lost updates", async () => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-settings-"));
    roots.push(root);
    const path = join(root, "settings.json");
    const store = new DesktopSettingsStore({ path });

    await Promise.all([
      store.update((settings) => ({
        ...settings,
        models: [
          {
            id: "model-one",
            runtimeModel: "model-one",
            apiProtocols: ["openai_responses"],
            capabilityIds: [],
          },
        ],
      })),
      store.update((settings) => ({
        ...settings,
        extensions: { ...settings.extensions, enabledPluginIds: ["paper-tools"] },
      })),
      store.update((settings) => ({
        ...settings,
        agents: [{ id: "agent-one", name: "Agent One" }],
      })),
    ]);

    const settings = await store.read();
    expect(settings.models.map((model) => model.id)).toEqual(["model-one"]);
    expect(settings.extensions.enabledPluginIds).toEqual(["paper-tools"]);
    expect(settings.agents.map((agent) => agent.id)).toEqual(["agent-one"]);
    expect(JSON.parse(await readFile(path, "utf8"))).toMatchObject({ schemaVersion: 1 });
  });
});
