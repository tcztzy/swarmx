import { z } from "zod";
import { type HookConfig, HookConfigSchema, type MessageChunk } from "./types.js";

export const HookEventSchema = z.enum(["onStart", "onChunk", "onHandoff", "onEnd"]);
export type HookEvent = z.infer<typeof HookEventSchema>;

export const HookResultSchema = z
  .object({
    continue: z.boolean().optional(),
    stopReason: z.string().min(1).max(2_000).optional(),
    additionalContext: z.string().min(1).max(20_000).optional(),
  })
  .strict()
  .superRefine((result, ctx) => {
    if (result.stopReason !== undefined && result.continue !== false) {
      ctx.addIssue({
        code: "custom",
        path: ["stopReason"],
        message: "stopReason requires continue to be false.",
      });
    }
  });

export type HookResult = z.infer<typeof HookResultSchema>;

export interface HookInvocation {
  event: HookEvent;
  scope: "agent" | "swarm";
  target: { name: string };
  arguments: Record<string, unknown>;
  context: Record<string, unknown>;
  chunk?: MessageChunk;
  handoff?: { source: string; target: string };
  outcome?: {
    status: "completed" | "failed";
    messages?: MessageChunk[];
    error?: string;
  };
}

export interface HookExecutorContext {
  signal: AbortSignal;
}

export type HookExecutor = (
  capability: string,
  input: HookInvocation,
  context: HookExecutorContext,
) => Promise<unknown> | unknown;

export interface HookRuntimeOptions {
  execute: HookExecutor;
  timeoutMs?: number;
}

export interface HookDispatchResult {
  additionalContext: string[];
}

export class HookExecutionError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "HookExecutionError";
  }
}

export class HookDeniedError extends HookExecutionError {
  constructor(capability: string, event: HookEvent, reason?: string) {
    super(`Hook "${capability}" denied ${event}${reason ? `: ${reason}` : "."}`);
    this.name = "HookDeniedError";
  }
}

const DEFAULT_HOOK_TIMEOUT_MS = 10_000;
const MAX_HOOK_TIMEOUT_MS = 600_000;
const MAX_ADDITIONAL_CONTEXT_CHARS = 50_000;

export class Hook {
  onStart?: string;
  onEnd?: string;
  onHandoff?: string;
  onChunk?: string;

  constructor(config?: HookConfig) {
    if (config) {
      const parsed = HookConfigSchema.parse(config);
      this.onStart = parsed.onStart;
      this.onEnd = parsed.onEnd;
      this.onHandoff = parsed.onHandoff;
      this.onChunk = parsed.onChunk;
    }
  }
}

export async function dispatchHooks(
  hooks: readonly Hook[],
  event: HookEvent,
  input: Omit<HookInvocation, "event">,
  runtime?: HookRuntimeOptions,
): Promise<HookDispatchResult> {
  const capabilities = hooks.flatMap((hook) => {
    const capability = hook[event];
    return capability ? [capability] : [];
  });
  if (capabilities.length === 0) return { additionalContext: [] };
  if (!runtime) {
    throw new HookExecutionError(
      `Configured ${event} hooks require an explicit host hook executor.`,
    );
  }

  const timeoutMs = hookTimeoutMs(runtime.timeoutMs);
  const settled = await Promise.allSettled(
    capabilities.map((capability) =>
      executeHook(capability, { ...input, event }, event, runtime.execute, timeoutMs),
    ),
  );
  const failed = settled.find(
    (result): result is PromiseRejectedResult => result.status === "rejected",
  );
  if (failed) throw failed.reason;

  const results = settled.map((result) => (result as PromiseFulfilledResult<HookResult>).value);
  for (let index = 0; index < results.length; index += 1) {
    const result = results[index];
    const capability = capabilities[index];
    if (!result || !capability) continue;
    if (
      (event === "onChunk" || event === "onEnd") &&
      (result.continue !== undefined ||
        result.stopReason !== undefined ||
        result.additionalContext !== undefined)
    ) {
      throw new HookExecutionError(
        `Hook "${capability}" returned control output for observational event ${event}.`,
      );
    }
  }

  const deniedIndex = results.findIndex((result) => result.continue === false);
  if (deniedIndex >= 0) {
    const result = results[deniedIndex];
    throw new HookDeniedError(capabilities[deniedIndex] ?? "unknown", event, result?.stopReason);
  }

  const additionalContext = results.flatMap((result) =>
    result.additionalContext ? [result.additionalContext] : [],
  );
  if (
    additionalContext.reduce((total, value) => total + value.length, 0) >
    MAX_ADDITIONAL_CONTEXT_CHARS
  ) {
    throw new HookExecutionError(
      `Combined ${event} additionalContext exceeds ${MAX_ADDITIONAL_CONTEXT_CHARS} characters.`,
    );
  }
  return { additionalContext };
}

export function appendHookContext(
  arguments_: Record<string, unknown>,
  additionalContext: readonly string[],
): Record<string, unknown> {
  if (additionalContext.length === 0) return arguments_;
  const messages = Array.isArray(arguments_.messages) ? [...arguments_.messages] : [];
  let insertionIndex = 0;
  while (
    insertionIndex < messages.length &&
    isRecord(messages[insertionIndex]) &&
    messages[insertionIndex].role === "system"
  ) {
    insertionIndex += 1;
  }
  messages.splice(
    insertionIndex,
    0,
    ...additionalContext.map((content) => ({ role: "system", content })),
  );
  return { ...arguments_, messages };
}

async function executeHook(
  capability: string,
  input: HookInvocation,
  event: HookEvent,
  execute: HookExecutor,
  timeoutMs: number,
): Promise<HookResult> {
  const controller = new AbortController();
  let timeout: ReturnType<typeof setTimeout> | undefined;
  const timeoutError = new HookExecutionError(
    `Hook "${capability}" timed out after ${timeoutMs}ms during ${event}.`,
  );
  const timeoutPromise = new Promise<never>((_resolve, reject) => {
    timeout = setTimeout(() => {
      controller.abort(timeoutError);
      reject(timeoutError);
    }, timeoutMs);
  });

  try {
    const isolatedInput = structuredClone(input);
    const raw = await Promise.race([
      Promise.resolve().then(() =>
        execute(capability, isolatedInput, { signal: controller.signal }),
      ),
      timeoutPromise,
    ]);
    return HookResultSchema.parse(raw === undefined ? {} : raw);
  } catch (error) {
    if (error instanceof HookExecutionError) throw error;
    throw new HookExecutionError(
      `Hook "${capability}" failed during ${event}: ${errorMessage(error)}`,
      { cause: error },
    );
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}

function hookTimeoutMs(value: number | undefined): number {
  const timeout = value ?? DEFAULT_HOOK_TIMEOUT_MS;
  if (!Number.isInteger(timeout) || timeout < 1 || timeout > MAX_HOOK_TIMEOUT_MS) {
    throw new HookExecutionError(
      `Hook timeout must be an integer from 1 to ${MAX_HOOK_TIMEOUT_MS} milliseconds.`,
    );
  }
  return timeout;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
