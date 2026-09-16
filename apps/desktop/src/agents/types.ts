import type { Agent, AgentCapabilities, RunOptions, RunResult } from "@swarmx/swarm";
import { z } from "zod";
import type { ToolManifestEntry } from "../host/mcp.js";
import type { Activity } from "../message-activity.js";
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
  /** Deliver the response to the native caller without persisting its contents. */
  readonly sensitive?: boolean;
  readonly approval?: {
    readonly toolId: string;
    readonly choices: readonly {
      readonly id: string;
      readonly label: string;
      readonly kind: "allow_once" | "allow_always" | "reject_once" | "reject_always";
      readonly answer: unknown;
    }[];
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
  activity?(event: Activity): void;
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
  ): Promise<RunResult>;
}

export const HARNESS_CAPABILITIES = {
  pi: { history: true, list: true, resume: true, steer: true, emptySessionResume: true },
  codex: { history: true, list: true, resume: true, steer: true, emptySessionResume: false },
  claude: { history: true, list: true, resume: true, steer: true, emptySessionResume: true },
  hermes: { history: true, list: true, resume: true, steer: true, emptySessionResume: false },
  openclaw: { history: true, list: true, resume: true, steer: true, emptySessionResume: false },
  dsh: { history: true, list: true, resume: false, steer: false, emptySessionResume: false },
} as const satisfies Record<string, AgentCapabilities>;

export function memoryContextSuffix(instructions: string) {
  return `\n\n<swarmx-memory-context>\n${instructions}\n</swarmx-memory-context>`;
}

export interface AgentOptions {
  readonly cwd: string;
  readonly productHome?: string;
  readonly mcp: {
    readonly command: string;
    readonly args: readonly string[];
    readonly env: Record<string, string>;
  };
  readonly executionPolicy?: () => ExecutionPolicy;
  readonly reviewOnly?: boolean;
  readonly productTools?: {
    readonly definitions: readonly ToolManifestEntry[];
    call(name: string, args: unknown, callId: string, signal: AbortSignal): Promise<unknown>;
  };
  /** The child receives only its own revocable MCP credential. */
  readonly registerMcp?: (token: string) => {
    bind(sessionId: string, runId: string): void;
    dispose(): void;
  };
}
