import { mkdtemp, rm, stat, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, it } from "vitest";
import { CoreMemory } from "../src/core-memory.js";

it("persists bounded Unicode user notes with revisions", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-core-memory-"));
  try {
    const memory = new CoreMemory(root);
    const empty = await memory.read();
    const saved = await memory.update({
      content: "中文研究偏好 🧪",
      expectedRevision: empty.revision,
    });
    expect(await new CoreMemory(root).read()).toEqual(saved);
    await expect(
      memory.update({
        content: "Overwrite",
        expectedRevision: empty.revision,
      }),
    ).rejects.toThrow("changed");
    await expect(
      memory.update({
        content: "🧪".repeat(1376),
        expectedRevision: saved.revision,
      }),
    ).rejects.toThrow("full");
    await expect(
      memory.update(
        { content: "Cancelled", expectedRevision: saved.revision },
        AbortSignal.abort(),
      ),
    ).rejects.toBeDefined();
    expect(await memory.read()).toEqual(saved);
    expect((await stat(join(root, "USER.md"))).mode & 0o777).toBe(0o600);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

it("rejects symlinked notes without changing their target", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-core-link-"));
  try {
    const target = join(root, "owner.txt");
    await writeFile(target, "Owner data");
    await symlink(target, join(root, "USER.md"));
    await expect(new CoreMemory(root).read()).rejects.toThrow("regular");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
