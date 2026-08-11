import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { MemoryRuntimeService } from "./memory-runtime-service.js";

const manifestPath = process.env.SWARMX_MEMORY_RUNTIME_MANIFEST;
const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })),
  );
});

describe.runIf(Boolean(manifestPath))("packaged Memory MCP integration", () => {
  it("runs versioned CRUD through the verified private server", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-mem-integration-"));
    temporaryRoots.push(root);
    const service = new MemoryRuntimeService({
      manifestPath: path.resolve(manifestPath as string),
      memoryRoot: path.join(root, "memory"),
    });

    try {
      const created = await service.create({
        title: "SwarmX",
        aliases: ["Swarm X"],
        content: "Version one links to [[Hermes Agent]].",
      });
      const updated = await service.update({
        id: created.id,
        expectedRevision: created.revision,
        content: "Version two links to [[Hermes Agent]].",
      });
      expect(await service.get(created.id)).toEqual(updated);
      expect(await service.search({ query: "version two", limit: 10 })).toEqual([updated]);
      const global = await service.getGlobalMemory();
      expect(global.user.content).toBeNull();
      const savedUser = await service.saveGlobalMemory({
        target: "user",
        expectedRevision: global.user.revision,
        content: "Prefers concise answers.",
      });
      expect(savedUser).toMatchObject({ fileName: "USER.md", revision: 1 });
      await expect(
        service.forgetGlobalMemory({
          target: "user",
          expectedRevision: savedUser.revision,
        }),
      ).resolves.toMatchObject({ content: null, revision: 2 });

      const history = await service.history({ id: created.id, limit: 10 });
      expect(history).toHaveLength(2);
      const createdVersion = history.find((entry) => entry.operation === "create");
      const updatedVersion = history.find((entry) => entry.operation === "update");
      if (!createdVersion || !updatedVersion) throw new Error("Missing Memory versions");
      await expect(
        service.getVersion({ id: created.id, version: createdVersion.version }),
      ).resolves.toMatchObject({ page: { content: "Version one links to [[Hermes Agent]]." } });
      await expect(
        service.diff({
          id: created.id,
          fromVersion: createdVersion.version,
          toVersion: updatedVersion.version,
        }),
      ).resolves.toMatchObject({ truncated: false });

      const restored = await service.restore({
        id: created.id,
        expectedRevision: updated.revision,
        version: createdVersion.version,
      });
      expect(restored).toMatchObject({ revision: 3, content: created.content });
      await service.delete({ id: restored.id, expectedRevision: restored.revision });
      expect(await service.list()).toEqual([]);
    } finally {
      await service.close();
    }
  }, 30_000);
});
