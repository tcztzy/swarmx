import { z } from "zod";
import { KnowledgeBaseError } from "./errors.js";
import { KnowledgeBaseVault, normalizeCreateConceptRequest } from "./vault.js";

export const KNOWLEDGE_BASE_ACTIONS = [
  "search_knowledge",
  "read_knowledge",
  "create_knowledge",
  "update_knowledge",
  "deprecate_knowledge",
] as const;

const RequestSchema = z.strictObject({
  action: z.enum(KNOWLEDGE_BASE_ACTIONS),
  request: z.unknown(),
});
const IdentifierSchema = z.strictObject({ id: z.string().min(1).max(1_024) });

export interface KnowledgeBaseOperationContext {
  readonly actorId: string;
  readonly callId: string;
  readonly workspaceRoot: string;
  readonly signal: AbortSignal;
  approve(reason: string): Promise<string>;
}

export interface KnowledgeBaseOperationResult {
  readonly action: (typeof KNOWLEDGE_BASE_ACTIONS)[number];
  readonly data: unknown;
}

export async function executeKnowledgeBaseOperation(
  vault: KnowledgeBaseVault,
  raw: unknown,
  context: KnowledgeBaseOperationContext,
): Promise<KnowledgeBaseOperationResult> {
  context.signal.throwIfAborted();
  const input = parsed(RequestSchema, raw);
  const result = (data: unknown) => ({ action: input.action, data });
  switch (input.action) {
    case "search_knowledge":
      return result(await vault.search(context.workspaceRoot, input.request as never));
    case "read_knowledge":
      return result(
        await vault.readConcept(context.workspaceRoot, parsed(IdentifierSchema, input.request).id),
      );
    case "create_knowledge":
      await approved(context, "Create one private knowledge-base concept.");
      return result(
        await vault.createConcept(
          context.workspaceRoot,
          normalizeCreateConceptRequest(input.request as never),
          context.signal,
        ),
      );
    case "update_knowledge":
      await approved(context, "Update one private knowledge-base concept.");
      return result(
        await vault.updateConcept(context.workspaceRoot, input.request as never, context.signal),
      );
    case "deprecate_knowledge":
      await approved(context, "Deprecate one private knowledge-base concept.");
      return result(
        await vault.deprecateConcept(context.workspaceRoot, input.request as never, context.signal),
      );
  }
}

export interface Config {
  readonly root: string;
  readonly maxConceptBytes?: number;
  readonly maxSearchPages?: number;
}

export class KnowledgeBaseService {
  readonly vault: KnowledgeBaseVault;

  constructor(config: Config) {
    this.vault = new KnowledgeBaseVault(config);
  }

  initialize(): Promise<void> {
    return this.vault.initialize();
  }

  execute(args: unknown, context: KnowledgeBaseOperationContext) {
    return executeKnowledgeBaseOperation(this.vault, args, context);
  }
}

async function approved(context: KnowledgeBaseOperationContext, reason: string): Promise<void> {
  if ((await context.approve(reason)) !== "allowed-once") {
    throw new KnowledgeBaseError(
      "Knowledge-base change was not approved.",
      "AUTHORIZATION_REQUIRED",
    );
  }
  context.signal.throwIfAborted();
}

function parsed<T>(schema: { parse(value: unknown): T }, value: unknown): T {
  try {
    return schema.parse(value);
  } catch (error) {
    throw new KnowledgeBaseError("Invalid knowledge-base request.", "INVALID_REQUEST", {
      cause: error,
    });
  }
}

export default KnowledgeBaseService;
