import { z } from "zod";
import { MemoryError } from "./errors.js";
import type { MemoryLintDiagnostic, MemoryResourceCheck } from "./lint.js";
import { type MemoryConcept, MemoryVault, normalizeCreateConceptRequest } from "./vault.js";

export const MEMORY_ACTIONS = [
  "search_memory",
  "read_memory",
  "create_memory",
  "update_memory",
  "deprecate_memory",
  "lint_memory",
] as const;

const RequestSchema = z.strictObject({
  action: z.enum(MEMORY_ACTIONS),
  request: z.unknown(),
});
const IdentifierSchema = z.strictObject({ id: z.string().min(1).max(1_024) });

export interface MemoryOperationContext {
  readonly actorId: string;
  readonly callId: string;
  readonly workspaceRoot: string;
  readonly signal: AbortSignal;
  approve(reason: string): Promise<string>;
}

export interface MemoryOperationResult {
  readonly action: (typeof MEMORY_ACTIONS)[number];
  readonly data: unknown;
  readonly diagnostics?: readonly MemoryLintDiagnostic[];
}

export async function executeMemoryOperation(
  vault: MemoryVault,
  raw: unknown,
  context: MemoryOperationContext,
): Promise<MemoryOperationResult> {
  context.signal.throwIfAborted();
  const input = parsed(RequestSchema, raw);
  const result = (data: unknown) => ({ action: input.action, data });
  const edited = async (concept: MemoryConcept) => ({
    ...result(concept),
    diagnostics: await vault.lint(context.workspaceRoot, {}, context.signal),
  });
  switch (input.action) {
    case "search_memory":
      return result(await vault.search(context.workspaceRoot, input.request as never));
    case "read_memory":
      return result(
        await vault.readConcept(context.workspaceRoot, parsed(IdentifierSchema, input.request).id),
      );
    case "lint_memory":
      return result(
        await vault.lint(context.workspaceRoot, input.request as never, context.signal),
      );
    case "create_memory":
      await approved(context, "Create one private memory concept.");
      return edited(
        await vault.createConcept(
          context.workspaceRoot,
          normalizeCreateConceptRequest(input.request as never),
          context.signal,
        ),
      );
    case "update_memory":
      await approved(context, "Update one private memory concept.");
      return edited(
        await vault.updateConcept(context.workspaceRoot, input.request as never, context.signal),
      );
    case "deprecate_memory":
      await approved(context, "Deprecate one private memory concept.");
      return edited(
        await vault.deprecateConcept(context.workspaceRoot, input.request as never, context.signal),
      );
  }
}

export interface Config {
  readonly root: string;
  readonly maxConceptBytes?: number;
  readonly maxSearchPages?: number;
  readonly checkResource?: MemoryResourceCheck;
}

export class MemoryService {
  readonly vault: MemoryVault;

  constructor(config: Config) {
    this.vault = new MemoryVault(config);
  }

  initialize(): Promise<void> {
    return this.vault.initialize();
  }

  execute(args: unknown, context: MemoryOperationContext) {
    return executeMemoryOperation(this.vault, args, context);
  }
}

async function approved(context: MemoryOperationContext, reason: string): Promise<void> {
  if ((await context.approve(reason)) !== "allowed-once") {
    throw new MemoryError("Memory change was not approved.", "AUTHORIZATION_REQUIRED");
  }
  context.signal.throwIfAborted();
}

function parsed<T>(schema: { parse(value: unknown): T }, value: unknown): T {
  try {
    return schema.parse(value);
  } catch (error) {
    throw new MemoryError("Invalid memory request.", "INVALID_REQUEST", {
      cause: error,
    });
  }
}

export default MemoryService;
