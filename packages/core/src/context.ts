import { createHash } from "node:crypto";
import { z } from "zod";

const ALLOWED_SECRET_REFERENCE_KEYS = new Set([
  "secretref",
  "secret_ref",
  "secretrefid",
  "secret_ref_id",
  "secretstatus",
  "secret_status",
  "credentialref",
  "credential_ref",
  "credentialrefs",
  "credential_refs",
  "credentialreferences",
  "credential_references",
]);

const FORBIDDEN_SECRET_KEY_PATTERN =
  /(api[_-]?key|access[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key)/i;

const idWithPrefix = (prefix: string) =>
  z.string().regex(new RegExp(`^${prefix}[A-Za-z0-9][A-Za-z0-9_-]*$`), `Must use ${prefix} prefix`);

export const ContextStrategySchema = z.enum([
  "auto",
  "checkpoint_tail",
  "microcompact",
  "full_tail",
  "isolated",
]);
export const ResolvedContextStrategySchema = z.enum([
  "checkpoint_tail",
  "microcompact",
  "full_tail",
  "isolated",
]);
export const ContextPacketModeSchema = z.enum(["thread_packet", "isolated"]);
export const ContextObjectKindSchema = z.enum([
  "instructions",
  "summary_checkpoint",
  "message_tail",
  "agent_invocations",
  "delegated_request",
]);

export const ContextObjectSchema = z
  .object({
    objectId: idWithPrefix("ctxo_"),
    kind: ContextObjectKindSchema,
    title: z.string().min(1),
    content: z.string(),
    sourceIds: z.array(z.string().min(1)).default([]),
    messageIds: z.array(z.string().min(1)).default([]),
    priority: z.number().int().default(0),
    originalBytes: z.number().int().nonnegative(),
    renderedBytes: z.number().int().nonnegative(),
    compressed: z.boolean().default(false),
    truncated: z.boolean().default(false),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const ContextModelRuntimeMetadataSchema = z
  .object({
    modelId: z.string().min(1),
    modelSupplyId: z.string().min(1).optional(),
    runtimeModel: z.string().min(1).optional(),
    apiType: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addAgentIdentityIssues);

export const ContextPacketMetadataSchema = z
  .object({
    packetId: idWithPrefix("ctxp_"),
    mode: ContextPacketModeSchema,
    conversationId: z.string().min(1),
    triggerMessageId: z.string().min(1).optional(),
    requestedStrategy: ContextStrategySchema,
    resolvedStrategy: ResolvedContextStrategySchema,
    latestSummaryCheckpointId: idWithPrefix("chk_").optional(),
    includedMessageIds: z.array(z.string().min(1)).default([]),
    promptBytes: z.number().int().nonnegative(),
    promptSha256: z.string().regex(/^[a-f0-9]{64}$/),
    includedObjectIds: z.array(idWithPrefix("ctxo_")).default([]),
    droppedObjectIds: z.array(idWithPrefix("ctxo_")).default([]),
    truncatedObjectIds: z.array(idWithPrefix("ctxo_")).default([]),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const ContextPacketSchema = z
  .object({
    metadata: ContextPacketMetadataSchema,
    objects: z.array(ContextObjectSchema).default([]),
    prompt: z.string(),
  })
  .passthrough()
  .superRefine((packet, ctx) => {
    addSecretIssues(packet, ctx);
    const promptBytes = byteLength(packet.prompt);
    const promptSha256 = sha256Hex(packet.prompt);
    if (packet.metadata.promptBytes !== promptBytes) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["metadata", "promptBytes"],
        message: "Context packet promptBytes must match rendered prompt.",
      });
    }
    if (packet.metadata.promptSha256 !== promptSha256) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["metadata", "promptSha256"],
        message: "Context packet promptSha256 must match rendered prompt.",
      });
    }
  });

export const SummaryCheckpointSchema = z
  .object({
    checkpointId: idWithPrefix("chk_"),
    conversationId: z.string().min(1),
    triggerMessageId: z.string().min(1).optional(),
    createdAt: z.string().min(1),
    source: z.string().min(1),
    requestedStrategy: ContextStrategySchema,
    resolvedStrategy: ResolvedContextStrategySchema,
    modelRuntime: ContextModelRuntimeMetadataSchema.optional(),
    coveredMessageIds: z.array(z.string().min(1)).default([]),
    includedMessageIds: z.array(z.string().min(1)).default([]),
    compressionPromptBytes: z.number().int().nonnegative(),
    compressionPromptSha256: z.string().regex(/^[a-f0-9]{64}$/),
    summary: z.string(),
  })
  .passthrough()
  .superRefine(addAgentIdentityIssues);

export const AgentInvocationContextMetadataSchema = z
  .object({
    triggerMessageId: z.string().min(1).optional(),
    harnessId: z.string().min(1),
    modelId: z.string().min(1),
    modelSupplyId: z.string().min(1).optional(),
    adapterId: z.string().min(1).optional(),
    contextStrategy: ResolvedContextStrategySchema,
    packet: ContextPacketMetadataSchema,
    externalSessionRef: z.string().min(1).optional(),
    trajectorySummary: z.string().optional(),
  })
  .passthrough()
  .superRefine(addAgentIdentityIssues);

export type ContextStrategy = z.infer<typeof ContextStrategySchema>;
export type ResolvedContextStrategy = z.infer<typeof ResolvedContextStrategySchema>;
export type ContextPacketMode = z.infer<typeof ContextPacketModeSchema>;
export type ContextObjectKind = z.infer<typeof ContextObjectKindSchema>;
export type ContextObject = z.infer<typeof ContextObjectSchema>;
export type ContextModelRuntimeMetadata = z.infer<typeof ContextModelRuntimeMetadataSchema>;
export type ContextPacketMetadata = z.infer<typeof ContextPacketMetadataSchema>;
export type ContextPacket = z.infer<typeof ContextPacketSchema>;
export type SummaryCheckpoint = z.infer<typeof SummaryCheckpointSchema>;
export type AgentInvocationContextMetadata = z.infer<typeof AgentInvocationContextMetadataSchema>;

export interface BuildContextPacketOptions {
  packetId: string;
  conversationId: string;
  triggerMessageId?: string;
  requestedStrategy: ContextStrategy;
  objects: unknown[];
  latestSummaryCheckpointId?: string;
  promptBudgetBytes?: number;
}

export function parseContextObject(input: unknown): ContextObject {
  return ContextObjectSchema.parse(input);
}

export function parseContextPacket(input: unknown): ContextPacket {
  return ContextPacketSchema.parse(input);
}

export function parseSummaryCheckpoint(input: unknown): SummaryCheckpoint {
  return SummaryCheckpointSchema.parse(input);
}

export function parseAgentInvocationContextMetadata(
  input: unknown,
): AgentInvocationContextMetadata {
  return AgentInvocationContextMetadataSchema.parse(input);
}

export function resolveContextStrategy(
  requested: ContextStrategy,
  options: { hasUsableCheckpoint: boolean },
): ResolvedContextStrategy {
  if (requested === "auto") {
    return options.hasUsableCheckpoint ? "checkpoint_tail" : "microcompact";
  }
  return requested;
}

export function buildContextPacket(options: BuildContextPacketOptions): ContextPacket {
  const objects = options.objects.map((object) => ContextObjectSchema.parse(object));
  const hasUsableCheckpoint =
    !!options.latestSummaryCheckpointId ||
    objects.some((object) => object.kind === "summary_checkpoint");
  const resolvedStrategy = resolveContextStrategy(options.requestedStrategy, {
    hasUsableCheckpoint,
  });
  const mode: ContextPacketMode = resolvedStrategy === "isolated" ? "isolated" : "thread_packet";
  const selectedObjects = selectContextObjects(objects, mode, options.promptBudgetBytes);
  const included = selectedObjects.included;
  const prompt = renderContextPrompt(included);

  const metadata = ContextPacketMetadataSchema.parse({
    packetId: options.packetId,
    mode,
    conversationId: options.conversationId,
    triggerMessageId: options.triggerMessageId,
    requestedStrategy: options.requestedStrategy,
    resolvedStrategy,
    latestSummaryCheckpointId:
      resolvedStrategy === "checkpoint_tail" ? options.latestSummaryCheckpointId : undefined,
    includedMessageIds: uniqueStrings(included.flatMap((object) => object.messageIds)),
    promptBytes: byteLength(prompt),
    promptSha256: sha256Hex(prompt),
    includedObjectIds: included.map((object) => object.objectId),
    droppedObjectIds: selectedObjects.dropped.map((object) => object.objectId),
    truncatedObjectIds: included
      .filter((object) => object.truncated)
      .map((object) => object.objectId),
  });

  return ContextPacketSchema.parse({
    metadata,
    objects: included,
    prompt,
  });
}

export function contextPromptSha256(prompt: string): string {
  return sha256Hex(prompt);
}

function selectContextObjects(
  objects: ContextObject[],
  mode: ContextPacketMode,
  promptBudgetBytes = Number.POSITIVE_INFINITY,
): { included: ContextObject[]; dropped: ContextObject[] } {
  const delegated = objects.filter((object) => object.kind === "delegated_request");
  if (delegated.length === 0) {
    throw new Error("Context packet requires a delegated_request object.");
  }

  if (mode === "isolated") {
    return {
      included: delegated,
      dropped: objects.filter((object) => object.kind !== "delegated_request"),
    };
  }

  const mandatoryIds = new Set(delegated.map((object) => object.objectId));
  const included = [...delegated];
  const dropped: ContextObject[] = [];
  let promptBytes = byteLength(renderContextPrompt(included));
  const candidates = objects
    .filter((object) => !mandatoryIds.has(object.objectId))
    .map((object, index) => ({ object, index }))
    .sort(
      (left, right) => right.object.priority - left.object.priority || left.index - right.index,
    );

  for (const { object } of candidates) {
    const nextPrompt = renderContextPrompt([...included, object]);
    const nextBytes = byteLength(nextPrompt);
    if (nextBytes <= promptBudgetBytes) {
      included.push(object);
      promptBytes = nextBytes;
    } else {
      dropped.push(object);
    }
  }

  if (promptBytes > promptBudgetBytes) {
    return { included, dropped };
  }

  return {
    included: restoreOriginalOrder(objects, included),
    dropped: restoreOriginalOrder(objects, dropped),
  };
}

function renderContextPrompt(objects: ContextObject[]): string {
  return objects.map((object) => `## ${object.title}\n\n${object.content}`).join("\n\n---\n\n");
}

function restoreOriginalOrder(
  objects: ContextObject[],
  selected: ContextObject[],
): ContextObject[] {
  const ids = new Set(selected.map((object) => object.objectId));
  return objects.filter((object) => ids.has(object.objectId));
}

function addSecretIssues(value: unknown, ctx: z.RefinementCtx): void {
  for (const issue of findInlineSecretKeys(value)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: issue.path,
      message: `Context records must not contain inline secret field "${issue.key}".`,
    });
  }
}

function addAgentIdentityIssues(value: unknown, ctx: z.RefinementCtx): void {
  addSecretIssues(value, ctx);
  if (!isObjectRecord(value)) return;
  for (const key of ["provider", "providerId", "providerProfileId", "model"]) {
    if (key in value) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: [key],
        message: `Context Agent identity must use modelId and optional modelSupplyId; field "${key}" is invalid.`,
      });
    }
  }
}

function findInlineSecretKeys(
  value: unknown,
  path: Array<string | number> = [],
): Array<{ key: string; path: Array<string | number> }> {
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => findInlineSecretKeys(item, [...path, index]));
  }
  if (!isObjectRecord(value)) return [];

  const issues: Array<{ key: string; path: Array<string | number> }> = [];
  for (const [key, child] of Object.entries(value)) {
    const normalizedKey = key.toLowerCase().replace(/[^a-z0-9_]/g, "");
    if (
      FORBIDDEN_SECRET_KEY_PATTERN.test(key) &&
      !ALLOWED_SECRET_REFERENCE_KEYS.has(normalizedKey)
    ) {
      issues.push({ key, path: [...path, key] });
    }
    issues.push(...findInlineSecretKeys(child, [...path, key]));
  }
  return issues;
}

function byteLength(value: string): number {
  return Buffer.byteLength(value, "utf8");
}

function sha256Hex(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
