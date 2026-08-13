import { z } from "zod";
import { stableHash, stableJson } from "./canonical-json.js";
import type { MessageChunk } from "./types.js";

export const PROJECT_BOOTSTRAP_MAX_BYTES = 8 * 1_024;
export const PROJECT_BOOTSTRAP_TIMEOUT_MS = 5_000;

export const ProjectBootstrapIdentifierSchema = z
  .string()
  .trim()
  .min(1)
  .max(256)
  .regex(
    /^[A-Za-z0-9][A-Za-z0-9._:@/+~-]*$/,
    "must be a safe identifier without whitespace or markup delimiters",
  );
const BoundedServiceTextSchema = z
  .string()
  .trim()
  .min(1)
  .max(128)
  .refine(hasNoControlCharacters, "must not contain control characters");
const ProjectStateStatusSchema = z.enum(["ready", "constrained", "blocked", "unknown"]);
const BoundedSnapshotRefSchema = z
  .string()
  .trim()
  .min(1)
  .max(128)
  .regex(
    /^[A-Za-z0-9][A-Za-z0-9._:@/+~-]*$/,
    "must be a safe reference without whitespace or markup delimiters",
  );

export const ProjectExecutionContextSchema = z
  .object({
    id: ProjectBootstrapIdentifierSchema,
    root: z
      .string()
      .trim()
      .min(1)
      .max(4_096)
      .refine(hasNoControlCharacters, "must not contain control characters")
      .refine(isAbsoluteProjectRoot, "must be an absolute filesystem path"),
  })
  .strict();

export const ProjectBootstrapContractSchema = z
  .object({
    serverName: BoundedServiceTextSchema,
    serverVersion: BoundedServiceTextSchema,
    bootstrapTool: BoundedServiceTextSchema,
    tools: z.array(BoundedServiceTextSchema).min(1).max(64),
  })
  .strict()
  .superRefine((value, context) => {
    if (new Set(value.tools).size !== value.tools.length) {
      context.addIssue({ code: "custom", path: ["tools"], message: "tool names must be unique" });
    }
    if (!value.tools.includes(value.bootstrapTool)) {
      context.addIssue({
        code: "custom",
        path: ["bootstrapTool"],
        message: "bootstrapTool must be included in tools",
      });
    }
  });

export const ProjectBootstrapBindingSchema = z
  .object({
    capabilityId: ProjectBootstrapIdentifierSchema,
    project: ProjectExecutionContextSchema,
    serverName: BoundedServiceTextSchema,
    serverVersion: BoundedServiceTextSchema,
    bootstrapTool: BoundedServiceTextSchema,
    tools: z.array(BoundedServiceTextSchema).min(1).max(64),
  })
  .strict()
  .superRefine((value, context) => {
    if (new Set(value.tools).size !== value.tools.length) {
      context.addIssue({ code: "custom", path: ["tools"], message: "tool names must be unique" });
    }
    if (!value.tools.includes(value.bootstrapTool)) {
      context.addIssue({
        code: "custom",
        path: ["bootstrapTool"],
        message: "bootstrapTool must be included in tools",
      });
    }
  });

export const ProjectBootstrapSnapshotSchema = z
  .object({
    schemaVersion: z.literal(1),
    projectId: ProjectBootstrapIdentifierSchema,
    registryRevision: ProjectBootstrapIdentifierSchema,
    activeRunRefs: z.array(BoundedSnapshotRefSchema).max(32),
    openDecisionRefs: z.array(BoundedSnapshotRefSchema).max(32),
    siteProfileVersion: BoundedSnapshotRefSchema.optional(),
    storageStatus: ProjectStateStatusSchema.optional(),
    quotaStatus: ProjectStateStatusSchema.optional(),
  })
  .strict()
  .superRefine((value, context) => {
    for (const field of ["activeRunRefs", "openDecisionRefs"] as const) {
      if (new Set(value[field]).size !== value[field].length) {
        context.addIssue({
          code: "custom",
          path: [field],
          message: `${field} must contain unique references`,
        });
      }
    }
    if (encodedBytes(stableJson(value)) > PROJECT_BOOTSTRAP_MAX_BYTES) {
      context.addIssue({
        code: "custom",
        message: `Project bootstrap exceeds the ${PROJECT_BOOTSTRAP_MAX_BYTES}-byte size limit`,
      });
    }
  });

export const ProjectBootstrapReceiptSchema = z
  .object({
    schemaVersion: z.literal(1),
    capabilityId: ProjectBootstrapIdentifierSchema,
    serverName: BoundedServiceTextSchema,
    serverVersion: BoundedServiceTextSchema,
    projectId: ProjectBootstrapIdentifierSchema,
    registryRevision: ProjectBootstrapIdentifierSchema,
    snapshotDigest: z.string().regex(/^[a-f0-9]{16}$/),
    activeRunCount: z.number().int().nonnegative().max(32),
    openDecisionCount: z.number().int().nonnegative().max(32),
    siteProfileVersion: BoundedServiceTextSchema.optional(),
    storageStatus: ProjectStateStatusSchema.optional(),
    quotaStatus: ProjectStateStatusSchema.optional(),
  })
  .strict();

export type ProjectExecutionContext = z.infer<typeof ProjectExecutionContextSchema>;
export type ProjectBootstrapContract = z.infer<typeof ProjectBootstrapContractSchema>;
export type ProjectBootstrapBinding = z.infer<typeof ProjectBootstrapBindingSchema>;
export type ProjectBootstrapSnapshot = z.infer<typeof ProjectBootstrapSnapshotSchema>;
export type ProjectBootstrapReceipt = z.infer<typeof ProjectBootstrapReceiptSchema>;

export function parseProjectBootstrapSnapshot(
  input: unknown,
  expectedProjectId: string,
): ProjectBootstrapSnapshot {
  const projectId = ProjectBootstrapIdentifierSchema.parse(expectedProjectId);
  const snapshot = ProjectBootstrapSnapshotSchema.parse(input);
  if (snapshot.projectId !== projectId) {
    throw new Error(
      `Project identity mismatch: expected "${projectId}", received "${snapshot.projectId}".`,
    );
  }
  return snapshot;
}

export function parseProjectBootstrapResult(
  result: {
    content: string;
    structuredContent?: unknown;
    isError: boolean;
    rawMcpContentBlocks?: readonly unknown[];
  },
  expectedProjectId: string,
): ProjectBootstrapSnapshot {
  if (result.isError) throw new Error("Project bootstrap tool failed.");
  if (!Array.isArray(result.rawMcpContentBlocks) || result.rawMcpContentBlocks.length !== 1) {
    throw new Error("Project bootstrap must return exactly one MCP content block.");
  }
  const contentBlock = result.rawMcpContentBlocks[0];
  if (!isTextContentBlock(contentBlock)) {
    throw new Error("Project bootstrap MCP content block must be text.");
  }
  if (contentBlock.text !== result.content) {
    throw new Error("Project bootstrap raw and normalized text content are contradictory.");
  }
  if (encodedBytes(result.content) > PROJECT_BOOTSTRAP_MAX_BYTES) {
    throw new Error(
      `Project bootstrap text exceeds the ${PROJECT_BOOTSTRAP_MAX_BYTES}-byte limit.`,
    );
  }
  if (result.structuredContent === undefined) {
    throw new Error("Project bootstrap returned no structured content.");
  }
  let decoded: unknown;
  try {
    decoded = JSON.parse(result.content);
  } catch {
    throw new Error("Project bootstrap returned invalid JSON text.");
  }
  if (stableJson(decoded) !== stableJson(result.structuredContent)) {
    throw new Error("Project bootstrap returned contradictory text and structured content.");
  }
  return parseProjectBootstrapSnapshot(result.structuredContent, expectedProjectId);
}

export function appendProjectBootstrapInstructions(
  instructions: string,
  snapshotInput: ProjectBootstrapSnapshot,
): string {
  const snapshot = parseProjectBootstrapSnapshot(snapshotInput, snapshotInput.projectId);
  const block = [
    "<project_bootstrap>",
    "Immutable Project snapshot for this run. Use the Project service for newer state; do not infer Project identity from paths or prior conversations.",
    stableJson(snapshot),
    "</project_bootstrap>",
  ].join("\n");
  return [instructions.trim(), block].filter(Boolean).join("\n\n---\n\n");
}

export function buildProjectBootstrapReceipt(
  bindingInput: ProjectBootstrapBinding,
  snapshotInput: ProjectBootstrapSnapshot,
): ProjectBootstrapReceipt {
  const binding = ProjectBootstrapBindingSchema.parse(bindingInput);
  const snapshot = parseProjectBootstrapSnapshot(snapshotInput, binding.project.id);
  return ProjectBootstrapReceiptSchema.parse({
    schemaVersion: 1,
    capabilityId: binding.capabilityId,
    serverName: binding.serverName,
    serverVersion: binding.serverVersion,
    projectId: snapshot.projectId,
    registryRevision: snapshot.registryRevision,
    snapshotDigest: stableHash(stableJson(snapshot)),
    activeRunCount: snapshot.activeRunRefs.length,
    openDecisionCount: snapshot.openDecisionRefs.length,
    siteProfileVersion: snapshot.siteProfileVersion,
    storageStatus: snapshot.storageStatus,
    quotaStatus: snapshot.quotaStatus,
  });
}

export function projectBootstrapReceiptMessage(
  receiptInput: ProjectBootstrapReceipt,
): MessageChunk {
  const receipt = ProjectBootstrapReceiptSchema.parse(receiptInput);
  const state = [
    `${receipt.activeRunCount} active Run${receipt.activeRunCount === 1 ? "" : "s"}`,
    `${receipt.openDecisionCount} open Decision${receipt.openDecisionCount === 1 ? "" : "s"}`,
    ...(receipt.storageStatus ? [`storage ${receipt.storageStatus}`] : []),
    ...(receipt.quotaStatus ? [`quota ${receipt.quotaStatus}`] : []),
  ].join(" · ");
  return {
    role: "system",
    kind: "message",
    content: [
      "Project bootstrap loaded",
      `Project: ${receipt.projectId} · registry ${receipt.registryRevision}`,
      `Service: ${receipt.serverName} ${receipt.serverVersion} · snapshot ${receipt.snapshotDigest}`,
      `State: ${state}`,
    ].join("\n"),
    render: { source: "project_bootstrap_receipt" },
    structuredContent: { projectBootstrap: receipt },
  };
}

function encodedBytes(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}

function isTextContentBlock(value: unknown): value is { type: "text"; text: string } {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    (value as { type?: unknown }).type === "text" &&
    typeof (value as { text?: unknown }).text === "string"
  );
}

function hasNoControlCharacters(value: string): boolean {
  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index);
    if (code <= 0x1f || code === 0x7f) return false;
  }
  return true;
}

function isAbsoluteProjectRoot(value: string): boolean {
  return value.startsWith("/") || value.startsWith("\\\\") || /^[A-Za-z]:[\\/]/.test(value);
}
