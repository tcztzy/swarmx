import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { DesktopExtensionManager, loadMarketplaceCatalog } from "./extension-manager.js";
import { DesktopSettingsStore } from "./settings-store.js";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true })));
});

describe("DesktopExtensionManager", () => {
  it("persists marketplace sources, lifecycle receipts, and evolution policy", async () => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-extensions-"));
    roots.push(root);
    const manager = new DesktopExtensionManager(
      new DesktopSettingsStore({ path: join(root, "settings.json") }),
      () => "2026-07-14T10:00:00.000Z",
      async () => ({
        catalogDigest: "sha256:catalog",
        candidates: [
          {
            pluginId: "paper-tools",
            name: "Paper tools",
            trust: "verified",
            revision: {
              revisionId: "paper-tools@1.0.0",
              version: "1.0.0",
              contentDigest: "sha256:paper-tools-1",
              sourceId: "official",
            },
          },
        ],
      }),
    );
    await manager.saveSource({
      id: "official",
      name: "Official",
      kind: "remote_catalog",
      location: "https://plugins.swarmx.dev/catalog.json",
      trust: "verified",
    });
    await manager.refreshSource("official");
    const result = await manager.applyAction({
      action: "install",
      pluginId: "paper-tools",
      confirmed: true,
      candidate: {
        pluginId: "paper-tools",
        name: "Paper tools",
        trust: "verified",
        revision: {
          revisionId: "paper-tools@1.0.0",
          version: "1.0.0",
          contentDigest: "sha256:paper-tools-1",
          sourceId: "official",
        },
      },
    });
    const state = await manager.saveEvolutionPolicy({ enabled: true, promotionGate: "human" });

    expect(result.receipt.status).toBe("applied");
    expect(state).toMatchObject({
      sources: [{ id: "official" }],
      candidates: [{ pluginId: "paper-tools" }],
      installed: [{ pluginId: "paper-tools", enabled: true }],
      skillEvolutionEnabled: true,
      skillPromotionGate: "human",
    });
  });

  it("loads bounded local catalogs and binds candidates to the selected source", async () => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-extension-catalog-"));
    roots.push(root);
    const path = join(root, "catalog.json");
    await writeFile(
      path,
      JSON.stringify({
        schemaVersion: 1,
        plugins: [
          {
            pluginId: "paper-tools",
            name: "Paper tools",
            trust: "verified",
            revision: {
              revisionId: "paper-tools@1.1.0",
              version: "1.1.0",
              contentDigest: "sha256:paper-tools-2",
              sourceId: "publisher-id",
            },
          },
        ],
      }),
      "utf8",
    );

    const result = await loadMarketplaceCatalog({
      id: "local-catalog",
      name: "Local catalog",
      kind: "local_path",
      location: path,
      trust: "local",
    });

    expect(result.catalogDigest).toMatch(/^sha256:/);
    expect(result.candidates).toMatchObject([
      {
        pluginId: "paper-tools",
        trust: "local",
        revision: { sourceId: "local-catalog", version: "1.1.0" },
      },
    ]);
  });
});
