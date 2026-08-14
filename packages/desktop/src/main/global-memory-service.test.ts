import {
  createDefaultDesktopSettings,
  type DesktopSettingsDocument,
  type GlobalMemoryBackend,
  type GlobalMemoryFile,
  type GlobalMemoryTarget,
  globalMemoryState,
} from "@swarmx/core";
import { describe, expect, it, vi } from "vitest";
import { GlobalMemoryService } from "./global-memory-service.js";
import type { DesktopSettingsStoreLike } from "./settings-store.js";

const NOW = "2026-08-10T08:00:00.000Z";
const LEGACY_USER = {
  content: "Prefer concise answers.",
  updatedAt: "2026-08-09T08:00:00.000Z",
};

describe("GlobalMemoryService", () => {
  it("overlays legacy USER.md until an authoritative user save succeeds", async () => {
    const settings = settingsFixture({ personalMemory: LEGACY_USER });
    const memory = backendFixture();
    const service = new GlobalMemoryService(memory.backend, settings.store, () => NOW);

    await expect(service.get()).resolves.toMatchObject({
      user: { content: LEGACY_USER.content, revision: 0 },
      memory: { content: null },
      legacyUser: true,
    });
    await expect(service.getGlobalMemory()).resolves.toMatchObject({ legacyUser: true });

    await expect(
      service.save({
        target: "memory",
        content: "Prefer verified workflows.",
        expectedRevision: 0,
      }),
    ).resolves.toMatchObject({
      user: { content: LEGACY_USER.content },
      memory: { content: "Prefer verified workflows.", revision: 1 },
      legacyUser: true,
    });
    expect(settings.current().personalMemory).toEqual(LEGACY_USER);

    await expect(
      service.save({ target: "user", content: "Prefer evidence first." }),
    ).resolves.toMatchObject({
      user: { content: "Prefer evidence first.", revision: 1 },
      legacyUser: false,
    });
    expect(settings.current().personalMemory).toBeNull();
    expect(memory.backend.saveGlobalMemory).toHaveBeenNthCalledWith(1, {
      target: "memory",
      content: "Prefer verified workflows.",
      expectedRevision: 0,
    });
    expect(memory.backend.saveGlobalMemory).toHaveBeenNthCalledWith(2, {
      target: "user",
      content: "Prefer evidence first.",
      expectedRevision: 0,
    });

    await settings.store.update((current) => ({ ...current, personalMemory: LEGACY_USER }));
    await expect(service.get()).resolves.toMatchObject({
      user: { content: "Prefer evidence first." },
      legacyUser: false,
    });
    await expect(
      service.save({
        target: "user",
        content: "A stale client must not overwrite this.",
        expectedRevision: 0,
      }),
    ).rejects.toThrow("revision conflict");
    expect(settings.current().personalMemory).toEqual(LEGACY_USER);
    await expect(
      service.save({ target: "user", content: "Valid body", unexpected: true }),
    ).rejects.toThrow();
  });

  it("clears the legacy overlay and authoritative files through distinct forget paths", async () => {
    const settings = settingsFixture({ personalMemory: LEGACY_USER });
    const memory = backendFixture({
      user: globalFile("user", null, 2),
      memory: globalFile("memory", "Old operational note.", 4),
    });
    const service = new GlobalMemoryService(memory.backend, settings.store, () => NOW);

    await expect(service.forget({ target: "memory", confirmed: true })).resolves.toMatchObject({
      memory: { content: null, revision: 5 },
      legacyUser: true,
    });
    expect(settings.current().personalMemory).toEqual(LEGACY_USER);
    expect(memory.backend.forgetGlobalMemory).toHaveBeenLastCalledWith({
      target: "memory",
      expectedRevision: 4,
    });

    await expect(
      service.forget({ target: "user", confirmed: true, expectedRevision: 2 }),
    ).resolves.toMatchObject({ user: { content: null, revision: 2 }, legacyUser: false });
    expect(settings.current().personalMemory).toBeNull();
    expect(memory.backend.forgetGlobalMemory).toHaveBeenCalledTimes(1);

    await expect(
      service.forget({ target: "memory", confirmed: true, expectedRevision: 0 }),
    ).resolves.toMatchObject({ memory: { content: null, revision: 5 } });
    await expect(
      service.forget({ target: "user", confirmed: true, expectedRevision: 0 }),
    ).resolves.toMatchObject({ user: { content: null, revision: 2 } });
    expect(memory.backend.forgetGlobalMemory).toHaveBeenCalledTimes(1);

    memory.files.user = globalFile("user", "Authoritative profile.", 7);
    await settings.store.update((current) => ({ ...current, personalMemory: LEGACY_USER }));
    await expect(
      service.forget({ target: "user", confirmed: true, expectedRevision: 6 }),
    ).rejects.toThrow("revision conflict");
    await expect(
      service.forget({ target: "user", confirmed: true, expectedRevision: 7 }),
    ).resolves.toMatchObject({ user: { content: null, revision: 8 }, legacyUser: false });
    expect(settings.current().personalMemory).toBeNull();
    expect(memory.backend.forgetGlobalMemory).toHaveBeenLastCalledWith({
      target: "user",
      expectedRevision: 7,
    });

    await expect(
      service.forgetGlobalMemory({ target: "memory", expectedRevision: 4 }),
    ).rejects.toThrow("revision conflict");
    await expect(service.forget({ target: "memory", confirmed: false })).rejects.toThrow();
  });

  it("returns immutable snapshots for each source combination and null when empty", async () => {
    const settings = settingsFixture();
    const memory = backendFixture();
    const service = new GlobalMemoryService(memory.backend, settings.store, () => NOW);

    await expect(service.snapshot()).resolves.toBeNull();

    await settings.store.update((current) => ({ ...current, personalMemory: LEGACY_USER }));
    const legacySnapshot = await service.snapshot();
    expect(legacySnapshot).toMatchObject({
      source: "memory_files_with_legacy_user",
      user: { content: LEGACY_USER.content },
      memory: null,
    });
    expect(Object.isFrozen(legacySnapshot)).toBe(true);

    await settings.store.update((current) => ({ ...current, personalMemory: null }));
    memory.files.memory = globalFile("memory", "Cross-project lesson.", 1);
    await expect(service.snapshot()).resolves.toMatchObject({
      source: "memory_files",
      user: null,
      memory: { content: "Cross-project lesson." },
    });

    memory.files.user = globalFile("user", "Stable preference.", 1);
    await expect(service.snapshot()).resolves.toMatchObject({
      source: "memory_files",
      user: { content: "Stable preference." },
      memory: { content: "Cross-project lesson." },
    });
  });

  it("retains monotonic per-Session reflection cursors and only the newest thousand", async () => {
    const settings = settingsFixture();
    const memory = backendFixture();
    const service = new GlobalMemoryService(memory.backend, settings.store, () => NOW);

    await service.recordCompletedTurn({ sessionId: "session-a" });
    await service.recordCompletedTurn({
      sessionId: "session-a",
      reviewedThrough: 9,
      now: "2026-08-10T09:00:00.000Z",
    });
    await service.recordCompletedTurn({
      sessionId: "session-a",
      reviewedThrough: 2,
      now: "2026-08-10T10:00:00.000Z",
    });
    expect(settings.current().memoryReview.sessions["session-a"]).toEqual({
      reviewedUserTurns: 9,
      updatedAt: "2026-08-10T10:00:00.000Z",
    });

    await expect(
      service.reflectionDecision({
        sessionId: "session-a",
        userTurnCount: 10,
        userText: "continue",
      }),
    ).resolves.toEqual({ due: false, sessionId: "session-a", unreviewedUserTurns: 1 });
    await expect(
      service.reflectionDecision({
        sessionId: "session-a",
        userTurnCount: 10,
        userText: "continue",
        now: "2026-08-12T10:00:00.000Z",
      }),
    ).resolves.toMatchObject({ due: true, reason: "idle_tail", fromUserTurn: 10 });

    const defaultClockService = new GlobalMemoryService(memory.backend, settings.store);
    await expect(
      defaultClockService.reflectionDecision({
        sessionId: "session-new",
        userTurnCount: 1,
        userText: "Remember this.",
      }),
    ).resolves.toMatchObject({ due: true, reason: "explicit", fromUserTurn: 1 });

    const fullSessions = Object.fromEntries(
      Array.from({ length: 1_000 }, (_, index) => [
        `session-${index.toString().padStart(4, "0")}`,
        {
          reviewedUserTurns: index,
          updatedAt: new Date(Date.UTC(2025, 0, 1, 0, 0, index)).toISOString(),
        },
      ]),
    );
    const fullSettings = settingsFixture({ memoryReview: { sessions: fullSessions } });
    const retentionService = new GlobalMemoryService(memory.backend, fullSettings.store, () => NOW);
    await retentionService.recordCompletedTurn({
      sessionId: "session-newest",
      reviewedThrough: 1,
      now: NOW,
    });
    expect(Object.keys(fullSettings.current().memoryReview.sessions)).toHaveLength(1_000);
    expect(fullSettings.current().memoryReview.sessions["session-newest"]).toBeDefined();
    expect(fullSettings.current().memoryReview.sessions["session-0000"]).toBeUndefined();
  });
});

function settingsFixture(initial: Partial<DesktopSettingsDocument> = {}): {
  store: DesktopSettingsStoreLike;
  current(): DesktopSettingsDocument;
} {
  let current = createDefaultDesktopSettings(initial);
  return {
    store: {
      read: async () => current,
      update: async (mutation) => {
        current = await mutation(current);
        return current;
      },
    },
    current: () => current,
  };
}

function backendFixture(initial: Partial<Record<GlobalMemoryTarget, GlobalMemoryFile>> = {}): {
  backend: GlobalMemoryBackend;
  files: Record<GlobalMemoryTarget, GlobalMemoryFile>;
} {
  const files = {
    user: initial.user ?? globalFile("user", null, 0),
    memory: initial.memory ?? globalFile("memory", null, 0),
  };
  const backend: GlobalMemoryBackend = {
    getGlobalMemory: vi.fn(async () =>
      globalMemoryState({ user: files.user, memory: files.memory }),
    ),
    saveGlobalMemory: vi.fn(async (input) => {
      assertRevision(files[input.target], input.expectedRevision);
      const file = globalFile(input.target, input.content, input.expectedRevision + 1);
      files[input.target] = file;
      return file;
    }),
    forgetGlobalMemory: vi.fn(async (input) => {
      assertRevision(files[input.target], input.expectedRevision);
      const file = globalFile(input.target, null, input.expectedRevision + 1);
      files[input.target] = file;
      return file;
    }),
  };
  return { backend, files };
}

function assertRevision(file: GlobalMemoryFile, expectedRevision: number): void {
  if (file.revision !== expectedRevision) throw new Error("Global Memory revision conflict.");
}

function globalFile(
  target: GlobalMemoryTarget,
  content: string | null,
  revision: number,
): GlobalMemoryFile {
  return {
    target,
    fileName: target === "user" ? "USER.md" : "MEMORY.md",
    content,
    revision,
    updatedAt: content ? NOW : null,
  };
}
