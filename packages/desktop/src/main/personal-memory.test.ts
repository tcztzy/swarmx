import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PERSONAL_MEMORY_MAX_CHARACTERS } from "@swarmx/core";
import { afterEach, describe, expect, it } from "vitest";
import { createPersonalMemoryAgentTool, PersonalMemoryService } from "./personal-memory.js";
import { DesktopSettingsStore } from "./settings-store.js";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("PersonalMemoryService", () => {
  it("persists, reloads, bounds, and explicitly forgets Personal Memory", async () => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-personal-memory-"));
    roots.push(root);
    const path = join(root, "settings.json");
    const store = new DesktopSettingsStore({ path });
    const service = new PersonalMemoryService(store, () => "2026-08-09T08:00:00.000Z");
    await store.update((settings) => ({
      ...settings,
      extensions: { ...settings.extensions, enabledPluginIds: ["paper-tools"] },
    }));

    await expect(service.get()).resolves.toEqual({
      status: "empty",
      maxCharacters: PERSONAL_MEMORY_MAX_CHARACTERS,
    });
    await expect(service.save({ content: "Prefer concise answers." })).resolves.toMatchObject({
      status: "saved",
      content: "Prefer concise answers.",
      characterCount: 23,
      updatedAt: "2026-08-09T08:00:00.000Z",
    });
    await expect(service.snapshot()).resolves.toMatchObject({
      content: "Prefer concise answers.",
      characterCount: 23,
    });
    expect(JSON.parse(await readFile(path, "utf8"))).toMatchObject({
      personalMemory: {
        content: "Prefer concise answers.",
        updatedAt: "2026-08-09T08:00:00.000Z",
      },
    });

    await expect(
      service.save({ content: "x".repeat(PERSONAL_MEMORY_MAX_CHARACTERS + 1) }),
    ).rejects.toThrow();
    await expect(service.save({ content: "invalid\u0000memory" })).rejects.toThrow();
    await expect(service.forget({ confirmed: false })).rejects.toThrow();
    await expect(service.forget({ confirmed: true })).resolves.toEqual({
      status: "empty",
      maxCharacters: PERSONAL_MEMORY_MAX_CHARACTERS,
    });
    expect((await store.read()).personalMemory).toBeNull();
    expect((await store.read()).extensions.enabledPluginIds).toEqual(["paper-tools"]);
  });

  it("lets an Agent propose save or forget only after explicit confirmation", async () => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-personal-memory-agent-"));
    roots.push(root);
    const service = new PersonalMemoryService(
      new DesktopSettingsStore({ path: join(root, "settings.json") }),
      () => "2026-08-09T08:00:00.000Z",
    );
    let approved = false;
    const audit: Array<{ operation: string; outcome: string; characterCount?: number }> = [];
    const tool = createPersonalMemoryAgentTool(service, {
      confirm: async () => approved,
      audit: (event) => audit.push(event),
    });
    if (tool.kind === "text") throw new Error("PersonalMemory must be a function tool");

    await expect(
      tool.call({ operation: "save", content: "Prefer concise answers." }),
    ).resolves.toMatchObject({
      structuredContent: { status: "denied", operation: "save" },
    });
    await expect(service.get()).resolves.toMatchObject({ status: "empty" });

    approved = true;
    await expect(
      tool.call({ operation: "save", content: "Prefer concise answers." }),
    ).resolves.toMatchObject({
      structuredContent: { status: "applied", operation: "save" },
    });
    await expect(service.get()).resolves.toMatchObject({
      status: "saved",
      content: "Prefer concise answers.",
    });
    await expect(tool.call({ operation: "forget" })).resolves.toMatchObject({
      structuredContent: { status: "applied", operation: "forget" },
    });
    await expect(service.get()).resolves.toMatchObject({ status: "empty" });
    expect(audit).toEqual([
      { operation: "save", outcome: "denied", characterCount: 23 },
      { operation: "save", outcome: "attempted", characterCount: 23 },
      { operation: "save", outcome: "completed", characterCount: 23 },
      { operation: "forget", outcome: "attempted" },
      { operation: "forget", outcome: "completed" },
    ]);
    expect(JSON.stringify(audit)).not.toContain("Prefer concise answers.");
  });

  it("fails closed and records no Memory body when Renderer confirmation is lost", async () => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-personal-memory-owner-lost-"));
    roots.push(root);
    const service = new PersonalMemoryService(
      new DesktopSettingsStore({ path: join(root, "settings.json") }),
    );
    const audit: Array<{ operation: string; outcome: string; characterCount?: number }> = [];
    const tool = createPersonalMemoryAgentTool(service, {
      confirm: async () => {
        throw new Error("Renderer window was closed.");
      },
      audit: (event) => audit.push(event),
    });
    if (tool.kind === "text") throw new Error("PersonalMemory must be a function tool");

    await expect(
      tool.call({ operation: "save", content: "Never persist without the owner." }),
    ).rejects.toThrow("Renderer window was closed");
    await expect(service.get()).resolves.toMatchObject({ status: "empty" });
    expect(audit).toEqual([{ operation: "save", outcome: "failed", characterCount: 32 }]);
    expect(JSON.stringify(audit)).not.toContain("Never persist without the owner.");
  });
});
