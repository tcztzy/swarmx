import {
  createGlobalMemorySnapshot,
  createPersonalMemorySnapshot,
  type GlobalMemoryBackend,
  type GlobalMemoryDeleteInput,
  type GlobalMemoryForgetInput,
  GlobalMemoryForgetInputSchema,
  type GlobalMemorySaveInput,
  GlobalMemorySaveInputSchema,
  type GlobalMemorySnapshot,
  type GlobalMemoryState,
  type GlobalMemoryWriteInput,
  globalMemoryState,
  type LocalTool,
  localToolResult,
  type MemoryReflectionDecision,
  memoryReflectionDecision,
  type PersonalMemoryAgentMutation,
  PersonalMemoryAgentMutationSchema,
  PersonalMemoryForgetInputSchema,
  type PersonalMemorySaveInput,
  PersonalMemorySaveInputSchema,
  type PersonalMemorySnapshot,
  type PersonalMemoryState,
  personalMemoryState,
} from "@swarmx/core";
import { z } from "zod";
import type { DesktopSettingsStoreLike } from "./settings-store.js";

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
    const update: GlobalMemorySaveInput = GlobalMemorySaveInputSchema.parse(input);
    const stored = await this.#backend.getGlobalMemory();
    await this.saveGlobalMemory({
      target: update.target,
      content: update.content,
      expectedRevision: stored[update.target].revision,
    });
    return this.get();
  }

  async forget(input: unknown): Promise<GlobalMemoryState> {
    const update: GlobalMemoryForgetInput = GlobalMemoryForgetInputSchema.parse(input);
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
      expectedRevision: file.revision,
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

export interface PersonalMemoryServiceLike {
  get(): Promise<PersonalMemoryState>;
  save(input: unknown): Promise<PersonalMemoryState>;
  forget(input: unknown): Promise<PersonalMemoryState>;
  snapshot(): Promise<PersonalMemorySnapshot | null>;
}

export class PersonalMemoryService implements PersonalMemoryServiceLike {
  readonly #settings: DesktopSettingsStoreLike;
  readonly #now: () => string;

  constructor(settings: DesktopSettingsStoreLike, now = () => new Date().toISOString()) {
    this.#settings = settings;
    this.#now = now;
  }

  async get(): Promise<PersonalMemoryState> {
    return personalMemoryState((await this.#settings.read()).personalMemory);
  }

  async save(input: unknown): Promise<PersonalMemoryState> {
    const update: PersonalMemorySaveInput = PersonalMemorySaveInputSchema.parse(input);
    const settings = await this.#settings.update((current) => ({
      ...current,
      personalMemory: {
        content: update.content,
        updatedAt: this.#now(),
      },
    }));
    return personalMemoryState(settings.personalMemory);
  }

  async forget(input: unknown): Promise<PersonalMemoryState> {
    PersonalMemoryForgetInputSchema.parse(input);
    const settings = await this.#settings.update((current) => ({
      ...current,
      personalMemory: null,
    }));
    return personalMemoryState(settings.personalMemory);
  }

  async snapshot(): Promise<PersonalMemorySnapshot | null> {
    const record = (await this.#settings.read()).personalMemory;
    return record ? createPersonalMemorySnapshot(record) : null;
  }
}

export interface PersonalMemoryAgentToolAuditEvent {
  operation: PersonalMemoryAgentMutation["operation"];
  outcome: "denied" | "attempted" | "completed" | "failed";
  characterCount?: number;
}

export interface PersonalMemoryAgentToolOptions {
  confirm(mutation: PersonalMemoryAgentMutation): Promise<boolean>;
  audit(event: PersonalMemoryAgentToolAuditEvent): void;
}

export function createPersonalMemoryAgentTool(
  service: PersonalMemoryServiceLike,
  options: PersonalMemoryAgentToolOptions,
): LocalTool {
  return {
    name: "PersonalMemory",
    description:
      "Propose replacing or forgetting the user's Personal Memory. Every change requires explicit user confirmation and applies to future runs.",
    inputSchema: z.toJSONSchema(PersonalMemoryAgentMutationSchema) as Record<string, unknown>,
    async call(arguments_) {
      const mutation = PersonalMemoryAgentMutationSchema.parse(arguments_);
      const characterCount = mutation.operation === "save" ? mutation.content.length : undefined;
      let confirmed: boolean;
      try {
        confirmed = await options.confirm(mutation);
      } catch (error) {
        options.audit({
          operation: mutation.operation,
          outcome: "failed",
          ...(characterCount === undefined ? {} : { characterCount }),
        });
        throw error;
      }
      if (!confirmed) {
        options.audit({
          operation: mutation.operation,
          outcome: "denied",
          ...(characterCount === undefined ? {} : { characterCount }),
        });
        return localToolResult("The user declined the Personal Memory change.", {
          status: "denied",
          operation: mutation.operation,
        });
      }

      const auditBase = {
        operation: mutation.operation,
        ...(characterCount === undefined ? {} : { characterCount }),
      };
      options.audit({ ...auditBase, outcome: "attempted" });
      try {
        if (mutation.operation === "save") {
          await service.save({ content: mutation.content });
        } else {
          await service.forget({ confirmed: true });
        }
        options.audit({ ...auditBase, outcome: "completed" });
        return localToolResult(
          mutation.operation === "save"
            ? "Personal Memory was saved for future runs. The current run continues using its original snapshot."
            : "Personal Memory was forgotten for future runs. The current run continues using its original snapshot.",
          { status: "applied", operation: mutation.operation },
        );
      } catch (error) {
        options.audit({ ...auditBase, outcome: "failed" });
        throw error;
      }
    },
  };
}
