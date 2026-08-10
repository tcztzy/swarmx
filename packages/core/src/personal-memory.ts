import { z } from "zod";
import { type MessageChunk, type SwarmConfig, SwarmConfigSchema } from "./types.js";

export const PERSONAL_MEMORY_MAX_CHARACTERS = 4_000;
const PERSONAL_MEMORY_SUMMARY_MAX_CHARACTERS = 160;

export const PersonalMemoryContentSchema = z
  .string()
  .max(PERSONAL_MEMORY_MAX_CHARACTERS)
  .refine((content) => content.trim().length > 0, "Personal Memory cannot be blank.")
  .refine(
    (content) => !hasUnsupportedControlCharacter(content),
    "Personal Memory contains unsupported control characters.",
  );

export const PersonalMemoryRecordSchema = z
  .object({
    content: PersonalMemoryContentSchema,
    updatedAt: z.iso.datetime(),
  })
  .strict();

export const PersonalMemorySaveInputSchema = z
  .object({ content: PersonalMemoryContentSchema })
  .strict();

export const PersonalMemoryForgetInputSchema = z.object({ confirmed: z.literal(true) }).strict();

export const PersonalMemoryAgentMutationSchema = z.discriminatedUnion("operation", [
  z
    .object({
      operation: z.literal("save"),
      content: PersonalMemoryContentSchema,
    })
    .strict(),
  z.object({ operation: z.literal("forget") }).strict(),
]);

export const PersonalMemorySnapshotSchema = z
  .object({
    source: z.literal("desktop_settings"),
    content: PersonalMemoryContentSchema,
    updatedAt: z.iso.datetime(),
    characterCount: z.number().int().positive().max(PERSONAL_MEMORY_MAX_CHARACTERS),
  })
  .strict()
  .superRefine((snapshot, context) => {
    if (snapshot.characterCount !== snapshot.content.length) {
      context.addIssue({
        code: "custom",
        path: ["characterCount"],
        message: "Personal Memory character count does not match its snapshot.",
      });
    }
  });

export const PersonalMemoryStateSchema = z.discriminatedUnion("status", [
  z
    .object({
      status: z.literal("empty"),
      maxCharacters: z.literal(PERSONAL_MEMORY_MAX_CHARACTERS),
    })
    .strict(),
  z
    .object({
      status: z.literal("saved"),
      content: PersonalMemoryContentSchema,
      updatedAt: z.iso.datetime(),
      characterCount: z.number().int().positive().max(PERSONAL_MEMORY_MAX_CHARACTERS),
      maxCharacters: z.literal(PERSONAL_MEMORY_MAX_CHARACTERS),
    })
    .strict(),
]);

export const PersonalMemoryExecutionPathSchema = z.enum([
  "direct_agent",
  "external_acp",
  "workflow",
]);

export const PersonalMemoryNotUsedReasonSchema = z.enum(["empty", "no_agent_nodes"]);

export const PersonalMemoryUseReceiptSchema = z.discriminatedUnion("status", [
  z
    .object({
      status: z.literal("used"),
      source: z.literal("desktop_settings"),
      executionPath: PersonalMemoryExecutionPathSchema,
      agentCount: z.number().int().positive().max(10_000),
      summary: z.string().min(1).max(PERSONAL_MEMORY_SUMMARY_MAX_CHARACTERS),
      updatedAt: z.iso.datetime(),
      characterCount: z.number().int().positive().max(PERSONAL_MEMORY_MAX_CHARACTERS),
      maxCharacters: z.literal(PERSONAL_MEMORY_MAX_CHARACTERS),
    })
    .strict(),
  z
    .object({
      status: z.literal("not_used"),
      source: z.literal("desktop_settings"),
      executionPath: PersonalMemoryExecutionPathSchema,
      agentCount: z.number().int().nonnegative().max(10_000),
      reason: PersonalMemoryNotUsedReasonSchema,
      summary: z.string().min(1).max(PERSONAL_MEMORY_SUMMARY_MAX_CHARACTERS),
      maxCharacters: z.literal(PERSONAL_MEMORY_MAX_CHARACTERS),
    })
    .strict(),
]);

export type PersonalMemoryRecord = z.infer<typeof PersonalMemoryRecordSchema>;
export type PersonalMemorySaveInput = z.infer<typeof PersonalMemorySaveInputSchema>;
export type PersonalMemoryForgetInput = z.infer<typeof PersonalMemoryForgetInputSchema>;
export type PersonalMemoryAgentMutation = z.infer<typeof PersonalMemoryAgentMutationSchema>;
export type PersonalMemorySnapshot = z.infer<typeof PersonalMemorySnapshotSchema>;
export type PersonalMemoryState = z.infer<typeof PersonalMemoryStateSchema>;
export type PersonalMemoryExecutionPath = z.infer<typeof PersonalMemoryExecutionPathSchema>;
export type PersonalMemoryNotUsedReason = z.infer<typeof PersonalMemoryNotUsedReasonSchema>;
export type PersonalMemoryUseReceipt = z.infer<typeof PersonalMemoryUseReceiptSchema>;

export function createPersonalMemorySnapshot(record: unknown): PersonalMemorySnapshot {
  const parsed = PersonalMemoryRecordSchema.parse(record);
  return Object.freeze(
    PersonalMemorySnapshotSchema.parse({
      source: "desktop_settings",
      content: parsed.content,
      updatedAt: parsed.updatedAt,
      characterCount: parsed.content.length,
    }),
  );
}

export function personalMemoryState(record: PersonalMemoryRecord | null): PersonalMemoryState {
  if (!record) {
    return PersonalMemoryStateSchema.parse({
      status: "empty",
      maxCharacters: PERSONAL_MEMORY_MAX_CHARACTERS,
    });
  }
  const parsed = PersonalMemoryRecordSchema.parse(record);
  return PersonalMemoryStateSchema.parse({
    status: "saved",
    content: parsed.content,
    updatedAt: parsed.updatedAt,
    characterCount: parsed.content.length,
    maxCharacters: PERSONAL_MEMORY_MAX_CHARACTERS,
  });
}

export function appendPersonalMemoryInstructions(
  instructions: string,
  snapshotInput: unknown,
): string {
  const snapshot = PersonalMemorySnapshotSchema.parse(snapshotInput);
  const memoryInstructions = [
    "Personal context: read-only Personal Memory snapshot from Desktop Settings.",
    "Use these durable user-authored facts and preferences when relevant.",
    "Keep this separate from Activity Profile, Session history, Project context, and Skills.",
    "Do not claim to update or forget this active snapshot; if the host provides a PersonalMemory tool, use it only to propose changes for future runs.",
    `Snapshot updated: ${snapshot.updatedAt}`,
    `Memory content (JSON string): ${JSON.stringify(snapshot.content)}`,
  ].join("\n");
  return [instructions.trim(), memoryInstructions].filter(Boolean).join("\n\n");
}

export function buildPersonalMemoryUseReceipt(input: {
  snapshot?: PersonalMemorySnapshot | null;
  executionPath: PersonalMemoryExecutionPath;
  agentCount: number;
}): PersonalMemoryUseReceipt {
  const executionPath = PersonalMemoryExecutionPathSchema.parse(input.executionPath);
  const agentCount = z.number().int().nonnegative().max(10_000).parse(input.agentCount);
  const reason: PersonalMemoryNotUsedReason | undefined =
    agentCount === 0 ? "no_agent_nodes" : input.snapshot ? undefined : "empty";
  if (reason) {
    return PersonalMemoryUseReceiptSchema.parse({
      status: "not_used",
      source: "desktop_settings",
      executionPath,
      agentCount,
      reason,
      summary: notUsedSummary(reason),
      maxCharacters: PERSONAL_MEMORY_MAX_CHARACTERS,
    });
  }
  if (!input.snapshot) throw new Error("Personal Memory snapshot is required when Memory is used.");
  const snapshot = PersonalMemorySnapshotSchema.parse(input.snapshot);
  return PersonalMemoryUseReceiptSchema.parse({
    status: "used",
    source: "desktop_settings",
    executionPath,
    agentCount,
    summary: summarizePersonalMemory(snapshot.content),
    updatedAt: snapshot.updatedAt,
    characterCount: snapshot.characterCount,
    maxCharacters: PERSONAL_MEMORY_MAX_CHARACTERS,
  });
}

export function personalMemoryReceiptMessage(receiptInput: unknown): MessageChunk {
  const receipt = PersonalMemoryUseReceiptSchema.parse(receiptInput);
  const content =
    receipt.status === "used"
      ? [
          "Personal Memory used",
          "Source: Settings → Personal Memory",
          `Execution: ${executionPathLabel(receipt.executionPath)} · ${receipt.agentCount} Agent${receipt.agentCount === 1 ? "" : "s"}`,
          `Snapshot: ${receipt.characterCount.toLocaleString("en-US")} / ${receipt.maxCharacters.toLocaleString("en-US")} characters · updated ${receipt.updatedAt}`,
          `Summary: ${receipt.summary}`,
        ].join("\n")
      : [
          "Personal Memory not used",
          "Source: Settings → Personal Memory",
          `Execution: ${executionPathLabel(receipt.executionPath)} · ${receipt.agentCount} Agent${receipt.agentCount === 1 ? "" : "s"}`,
          `Reason: ${receipt.summary}`,
        ].join("\n");
  return {
    role: "system",
    kind: "message",
    content,
    render: { source: "personal_memory_receipt" },
    structuredContent: { personalMemory: receipt },
  };
}

function summarizePersonalMemory(content: string): string {
  const normalized = content.replace(/\s+/gu, " ").trim();
  if (normalized.length <= PERSONAL_MEMORY_SUMMARY_MAX_CHARACTERS) return normalized;
  return `${normalized.slice(0, PERSONAL_MEMORY_SUMMARY_MAX_CHARACTERS - 1).trimEnd()}…`;
}

function notUsedSummary(reason: PersonalMemoryNotUsedReason): string {
  if (reason === "empty") return "No Personal Memory is saved.";
  return "This workflow has no Agent nodes that can receive Personal Memory.";
}

function executionPathLabel(path: PersonalMemoryExecutionPath): string {
  if (path === "direct_agent") return "Direct Agent";
  if (path === "external_acp") return "External ACP";
  return "Workflow";
}

export function countPersonalMemoryAgentTargets(configInput: unknown): number {
  return countAgentNodes(SwarmConfigSchema.parse(configInput));
}

function countAgentNodes(config: SwarmConfig): number {
  return Object.values(config.nodes).reduce((count, node) => {
    if (node.kind === "agent") return count + 1;
    if (node.kind === "swarm") {
      return count + countAgentNodes(SwarmConfigSchema.parse(node.swarm));
    }
    return count;
  }, 0);
}

function hasUnsupportedControlCharacter(content: string): boolean {
  return [...content].some((character) => {
    const codePoint = character.codePointAt(0);
    return (
      codePoint !== undefined &&
      ((codePoint < 32 && codePoint !== 9 && codePoint !== 10 && codePoint !== 13) ||
        codePoint === 127)
    );
  });
}
