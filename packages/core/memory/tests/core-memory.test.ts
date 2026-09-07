import { mkdtemp, rm, stat, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, it } from "vitest";
import { CoreMemory } from "../src/core-memory.js";

it("persists bounded Unicode notes with revisions and isolates workspace notes", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-core-memory-"));
  try {
    const memory = new CoreMemory(root, "123456abcdef");
    const empty = await memory.read("workspace");
    const saved = await memory.update({
      target: "workspace",
      content: "中文研究偏好 🧪",
      expectedRevision: empty.revision,
    });
    expect(await new CoreMemory(root, "123456abcdef").read("workspace")).toEqual(saved);
    expect((await new CoreMemory(root, "abcdef123456").read("workspace")).content).toBe("");
    await expect(
      memory.update({
        target: "workspace",
        content: "Overwrite",
        expectedRevision: empty.revision,
      }),
    ).rejects.toThrow("changed");
    await expect(
      memory.update({
        target: "workspace",
        content: "🧪".repeat(2201),
        expectedRevision: saved.revision,
      }),
    ).rejects.toThrow("full");
    await expect(
      memory.update(
        { target: "workspace", content: "Cancelled", expectedRevision: saved.revision },
        AbortSignal.abort(),
      ),
    ).rejects.toBeDefined();
    expect(await memory.read("workspace")).toEqual(saved);
    const user = await memory.read("user");
    await memory.update({
      target: "user",
      content: "Prefer Chinese",
      expectedRevision: user.revision,
    });
    expect((await new CoreMemory(root, "abcdef123456").read("user")).content).toBe(
      "Prefer Chinese",
    );
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
    await expect(new CoreMemory(root, "123456abcdef").read("user")).rejects.toThrow("regular");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
