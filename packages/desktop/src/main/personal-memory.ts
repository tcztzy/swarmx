import {
  createPersonalMemorySnapshot,
  type LocalTool,
  localToolResult,
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
