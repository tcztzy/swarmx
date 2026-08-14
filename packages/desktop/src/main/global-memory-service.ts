import {
  createGlobalMemorySnapshot,
  type GlobalMemoryBackend,
  type GlobalMemoryDeleteInput,
  GlobalMemoryForgetInputSchema,
  GlobalMemorySaveInputSchema,
  type GlobalMemorySnapshot,
  type GlobalMemoryState,
  type GlobalMemoryWriteInput,
  globalMemoryState,
  type MemoryReflectionDecision,
  memoryReflectionDecision,
} from "@swarmx/core";
import { z } from "zod";
import type { DesktopSettingsStoreLike } from "./settings-store.js";

const GlobalMemoryObservedSaveInputSchema = GlobalMemorySaveInputSchema.extend({
  expectedRevision: z.number().int().nonnegative().optional(),
}).strict();
const GlobalMemoryObservedForgetInputSchema = GlobalMemoryForgetInputSchema.extend({
  expectedRevision: z.number().int().nonnegative().optional(),
}).strict();

export interface GlobalMemoryServiceLike extends GlobalMemoryBackend {
  get(): Promise<GlobalMemoryState>;
  save(input: unknown): Promise<GlobalMemoryState>;
  forget(input: unknown): Promise<GlobalMemoryState>;
  snapshot(): Promise<GlobalMemorySnapshot | null>;
  reflectionDecision(input: {
    sessionId: string;
    userTurnCount: number;
    userText: string;
    now?: string;
  }): Promise<MemoryReflectionDecision>;
  recordCompletedTurn(input: {
    sessionId: string;
    reviewedThrough?: number;
    now?: string;
  }): Promise<void>;
}

export class GlobalMemoryService implements GlobalMemoryServiceLike {
  readonly #backend: GlobalMemoryBackend;
  readonly #settings: DesktopSettingsStoreLike;
  readonly #now: () => string;

  constructor(
    backend: GlobalMemoryBackend,
    settings: DesktopSettingsStoreLike,
    now = () => new Date().toISOString(),
  ) {
    this.#backend = backend;
    this.#settings = settings;
    this.#now = now;
  }

  async get(): Promise<GlobalMemoryState> {
    const [stored, settings] = await Promise.all([
      this.#backend.getGlobalMemory(),
      this.#settings.read(),
    ]);
    return globalMemoryState({
      user: stored.user,
      memory: stored.memory,
      legacyUser: settings.personalMemory,
    });
  }

  async save(input: unknown): Promise<GlobalMemoryState> {
    const update = GlobalMemoryObservedSaveInputSchema.parse(input);
    const stored = await this.#backend.getGlobalMemory();
    await this.saveGlobalMemory({
      target: update.target,
      content: update.content,
      expectedRevision: update.expectedRevision ?? stored[update.target].revision,
    });
    return this.get();
  }

  async forget(input: unknown): Promise<GlobalMemoryState> {
    const update = GlobalMemoryObservedForgetInputSchema.parse(input);
    const [stored, settings] = await Promise.all([
      this.#backend.getGlobalMemory(),
      this.#settings.read(),
    ]);
    const file = stored[update.target];
    if (!file.content && (update.target !== "user" || !settings.personalMemory)) {
      return globalMemoryState({ user: stored.user, memory: stored.memory });
    }
    await this.forgetGlobalMemory({
      target: update.target,
      expectedRevision: update.expectedRevision ?? file.revision,
    });
    return this.get();
  }

  async getGlobalMemory(): Promise<GlobalMemoryState> {
    return this.get();
  }

  async saveGlobalMemory(input: GlobalMemoryWriteInput) {
    const file = await this.#backend.saveGlobalMemory(input);
    if (input.target === "user") await this.#clearLegacyUser();
    return file;
  }

  async forgetGlobalMemory(input: GlobalMemoryDeleteInput) {
    const [stored, settings] = await Promise.all([
      this.#backend.getGlobalMemory(),
      this.#settings.read(),
    ]);
    if (stored[input.target].revision !== input.expectedRevision) {
      throw new Error("Global Memory revision conflict.");
    }
    if (input.target === "user" && !stored.user.content && settings.personalMemory) {
      await this.#clearLegacyUser();
      return stored.user;
    }
    const file = await this.#backend.forgetGlobalMemory(input);
    if (input.target === "user") await this.#clearLegacyUser();
    return file;
  }

  async snapshot(): Promise<GlobalMemorySnapshot | null> {
    const [stored, settings] = await Promise.all([
      this.#backend.getGlobalMemory(),
      this.#settings.read(),
    ]);
    if (!stored.user.content && !stored.memory.content && !settings.personalMemory) return null;
    return createGlobalMemorySnapshot({
      user: stored.user,
      memory: stored.memory,
      legacyUser: settings.personalMemory,
    });
  }

  async reflectionDecision(input: {
    sessionId: string;
    userTurnCount: number;
    userText: string;
    now?: string;
  }): Promise<MemoryReflectionDecision> {
    const settings = await this.#settings.read();
    return memoryReflectionDecision({
      ...input,
      now: input.now ?? this.#now(),
      state: settings.memoryReview,
    });
  }

  async recordCompletedTurn(input: {
    sessionId: string;
    reviewedThrough?: number;
    now?: string;
  }): Promise<void> {
    const now = input.now ?? this.#now();
    await this.#settings.update((settings) => {
      const current = settings.memoryReview.sessions[input.sessionId];
      const sessions = {
        ...settings.memoryReview.sessions,
        [input.sessionId]: {
          reviewedUserTurns: Math.max(current?.reviewedUserTurns ?? 0, input.reviewedThrough ?? 0),
          updatedAt: now,
        },
      };
      const retained = Object.fromEntries(
        Object.entries(sessions)
          .sort((left, right) => right[1].updatedAt.localeCompare(left[1].updatedAt))
          .slice(0, 1_000),
      );
      return { ...settings, memoryReview: { sessions: retained } };
    });
  }

  async #clearLegacyUser(): Promise<void> {
    await this.#settings.update((settings) => ({ ...settings, personalMemory: null }));
  }
}
