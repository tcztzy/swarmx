import type { Agent, AgentCapabilities, PromptResponse, RunOptions } from "@swarmx/swarm";
import { z } from "zod";
import type { AgentPermissions, PermissionRequest } from "../permissions.js";
import type { ExecutionPolicy } from "../settings.js";

export const RunOptionsSchema = z
  .strictObject({
    modelName: z
      .string()
      .min(1)
      .max(512)
      .regex(/^[^\s-][^\s]*$/u)
      .optional(),
    reasoningEffort: z.string().min(1).max(64).optional(),
    mode: z.string().min(1).max(512).optional(),
  })
  .transform(({ modelName, reasoningEffort, mode }) => ({
    model: modelName,
    effort: reasoningEffort,
    ...(mode === undefined ? {} : { mode }),
  })) satisfies z.ZodType<RunOptions>;

export interface Interaction {
  readonly id: string;
  readonly title: string;
  readonly schema: Record<string, unknown>;
  readonly permission?: {
    readonly toolCall: import("@agentclientprotocol/sdk").RequestPermissionRequest["toolCall"];
    readonly options: import("@agentclientprotocol/sdk").PermissionOption[];
    readonly answers: Record<string, unknown>;
  };
}

export type EventAttributes = Record<string, string | number | boolean | null | undefined>;

/** Host callbacks; durable observation precedes each caller's projection. */
export interface Observer {
  readonly executionId?: string;
  execution?(context: {
    runId: string;
    parentRunId: string | null;
    causedBy: string | null;
    permissions?: AgentPermissions;
  }): void;
  text(id: string, text: string, role?: "user" | "assistant" | "reasoning"): void;
  tool(id: string, name: string, input: unknown, output?: unknown): void;
  raw(event: unknown, attributes?: EventAttributes): void;
  interact(request: Interaction, signal?: AbortSignal): Promise<unknown>;
}

export interface NativeRunOptions extends RunOptions {
  readonly permissions?: PermissionRequest | undefined;
  readonly instructions?: string;
}
export interface NativeAgent extends Omit<Agent<Observer>, "start" | "create"> {
  permissions?(sessionId: string): Promise<AgentPermissions>;
  restoreEmptySessions?(ids: readonly string[]): void;
  steer(sessionId: string, text: string, expectedRunId?: string): Promise<void>;
  interrupt(sessionId: string, expectedRunId?: string): Promise<void>;
  create(options?: Pick<NativeRunOptions, "instructions" | "permissions">): Promise<string>;
  start(
    sessionId: string,
    text: string,
    observer: Observer,
    options?: NativeRunOptions,
  ): Promise<PromptResponse>;
}

export const HARNESS_CAPABILITIES = {
  codex: {
    loadSession: true,
    sessionCapabilities: { list: {}, resume: {} },
    _meta: {
      swarmx: {
        version: 2,
        permissions: true,
        steer: true,
        emptySessionResume: false,
        activeRunResume: false,
        interactionResume: false,
      },
    },
  },
  claude: {
    loadSession: true,
    sessionCapabilities: { list: {}, resume: {} },
    _meta: {
      swarmx: {
        version: 2,
        permissions: true,
        steer: true,
        emptySessionResume: true,
        activeRunResume: false,
        interactionResume: false,
      },
    },
  },
  hermes: {
    loadSession: true,
    sessionCapabilities: { list: {}, resume: {} },
    _meta: {
      swarmx: {
        version: 2,
        permissions: true,
        steer: true,
        emptySessionResume: false,
        activeRunResume: false,
        interactionResume: false,
      },
    },
  },
  openclaw: {
    loadSession: true,
    sessionCapabilities: { list: {}, resume: {} },
    _meta: {
      swarmx: {
        version: 2,
        permissions: true,
        steer: true,
        emptySessionResume: false,
        activeRunResume: false,
        interactionResume: false,
      },
    },
  },
} as const satisfies Record<string, AgentCapabilities>;

export function memoryContextSuffix(instructions: string) {
  return `\n\n<swarmx-memory-context>\n${instructions}\n</swarmx-memory-context>`;
}

export interface AgentOptions {
  readonly cwd: string;
  readonly mcp: { readonly url: string; readonly headers: Record<string, string> };
  readonly executionPolicy?: () => ExecutionPolicy;
  readonly reviewOnly?: boolean;
  /** The child receives only its own revocable MCP credential. */
  readonly registerMcp?: (token: string) => {
    bind(sessionId: string, runId: string): void;
    dispose(): void;
  };
}
