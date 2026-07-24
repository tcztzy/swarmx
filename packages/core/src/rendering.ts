import { z } from "zod";
import { MessageChunkSchema } from "./types.js";
import type { MessageChunk } from "./types.js";

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

const REDACTED_VALUE = "[redacted]";
const FORBIDDEN_SECRET_KEY_PATTERN =
  /(api[_-]?key|access[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key)/i;

const idWithPrefix = (prefix: string) =>
  z.string().regex(new RegExp(`^${prefix}[A-Za-z0-9][A-Za-z0-9_-]*$`), `Must use ${prefix} prefix`);

export const RenderEventKindSchema = z.enum([
  "message",
  "thinking",
  "tool_call",
  "tool_progress",
  "tool_result",
  "trace",
  "artifact",
  "agent_metadata",
]);

export const RenderEventStatusSchema = z.enum([
  "queued",
  "running",
  "succeeded",
  "failed",
  "canceled",
  "skipped",
  "completed",
]);

export const RenderArtifactKindSchema = z.enum([
  "file",
  "diff",
  "log",
  "screenshot",
  "image",
  "table",
  "html",
  "json",
  "terminal",
  "report",
  "evidence",
  "other",
]);

export const RenderArtifactReferenceSchema = z
  .object({
    artifactId: z.string().min(1).optional(),
    kind: RenderArtifactKindSchema.default("other"),
    title: z.string().min(1).optional(),
    path: z.string().min(1).optional(),
    url: z.string().min(1).optional(),
    mimeType: z.string().min(1).optional(),
    checksum: z.string().min(1).optional(),
    byteCount: z.number().int().nonnegative().optional(),
    truncated: z.boolean().optional(),
    metadata: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const RenderProvenanceSchema = z
  .object({
    host: z.string().min(1).optional(),
    adapter: z.string().min(1).optional(),
    pluginId: z.string().min(1).optional(),
    pluginName: z.string().min(1).optional(),
    mcpServer: z.string().min(1).optional(),
    marketplace: z.string().min(1).optional(),
    harnessId: z.string().min(1).optional(),
    modelId: z.string().min(1).optional(),
    modelSupplyId: z.string().min(1).optional(),
    agent: z.string().min(1).optional(),
    externalSessionRef: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addAgentProvenanceIssues);

export const NormalizedRenderEventSchema = z
  .object({
    eventId: idWithPrefix("rne_"),
    parentMessageId: z.string().min(1).optional(),
    invocationId: z.string().min(1).optional(),
    kind: RenderEventKindSchema,
    status: RenderEventStatusSchema,
    source: z.string().min(1),
    role: z.string().min(1).optional(),
    agent: z.string().min(1).optional(),
    title: z.string().min(1),
    summary: z.string(),
    content: z.string().optional(),
    toolName: z.string().min(1).optional(),
    input: z.unknown().optional(),
    output: z.unknown().optional(),
    artifacts: z.array(RenderArtifactReferenceSchema).default([]),
    startedAt: z.string().min(1).optional(),
    endedAt: z.string().min(1).optional(),
    durationMs: z.number().int().nonnegative().optional(),
    rawPayloadRef: z.string().min(1).optional(),
    provenance: RenderProvenanceSchema.default({}),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export type RenderEventKind = z.infer<typeof RenderEventKindSchema>;
export type RenderEventStatus = z.infer<typeof RenderEventStatusSchema>;
export type RenderArtifactKind = z.infer<typeof RenderArtifactKindSchema>;
export type RenderArtifactReference = z.infer<typeof RenderArtifactReferenceSchema>;
export type RenderProvenance = z.infer<typeof RenderProvenanceSchema>;
export type NormalizedRenderEvent = z.infer<typeof NormalizedRenderEventSchema>;

export interface NormalizeMessageChunkOptions {
  artifacts?: RenderArtifactReference[];
  parentMessageId?: string;
  invocationId?: string;
  source?: string;
  status?: RenderEventStatus;
  rawPayloadRef?: string;
  provenance?: RenderProvenance;
  startedAt?: string;
  endedAt?: string;
  durationMs?: number;
}

export function parseNormalizedRenderEvent(input: unknown): NormalizedRenderEvent {
  return NormalizedRenderEventSchema.parse(input);
}

export function sanitizeRenderPayload(input: unknown): unknown {
  if (Array.isArray(input)) return input.map(sanitizeRenderPayload);
  if (!isObjectRecord(input)) return input;

  return Object.fromEntries(
    Object.entries(input).map(([key, value]) => [
      key,
      isForbiddenSecretKey(key) ? REDACTED_VALUE : sanitizeRenderPayload(value),
    ]),
  );
}

export function normalizeMessageChunk(
  input: unknown,
  options: NormalizeMessageChunkOptions = {},
): NormalizedRenderEvent {
  const chunk = MessageChunkSchema.parse(input);
  const kind = chunk.kind;
  const status = options.status ?? defaultStatusForChunk(chunk);
  const source = options.source ?? "swarmx.message";
  const title = titleForChunk(chunk);
  const contentPayload =
    chunk.structuredContent === undefined
      ? parseContentPayload(chunk.content)
      : chunk.structuredContent;
  const sanitizedPayload = sanitizeRenderPayload(contentPayload);
  const eventInput = {
    eventId: deterministicRenderEventId({ chunk, options }),
    parentMessageId: options.parentMessageId,
    invocationId: options.invocationId,
    kind,
    status,
    source,
    role: chunk.role,
    agent: chunk.agent,
    title,
    summary: summarizeChunk(chunk, sanitizedPayload),
    content: kind === "message" || kind === "thinking" ? chunk.content : undefined,
    toolName: chunk.toolName,
    input: kind === "tool_call" ? sanitizedPayload : undefined,
    output: kind === "tool_progress" || kind === "tool_result" ? sanitizedPayload : undefined,
    artifacts: options.artifacts ?? [],
    rawPayloadRef: options.rawPayloadRef,
    provenance: sanitizeRenderPayload({
      ...(options.provenance ?? {}),
      agent: options.provenance?.agent ?? chunk.agent,
    }),
    startedAt: options.startedAt,
    endedAt: options.endedAt,
    durationMs: options.durationMs,
  };

  return NormalizedRenderEventSchema.parse(eventInput);
}

export function normalizeMessageChunks(
  chunks: unknown[],
  options: NormalizeMessageChunkOptions = {},
): NormalizedRenderEvent[] {
  return chunks.map((chunk, index) =>
    normalizeMessageChunk(chunk, {
      ...options,
      rawPayloadRef: options.rawPayloadRef ?? `message-chunk:${index}`,
    }),
  );
}

function defaultStatusForChunk(chunk: MessageChunk): RenderEventStatus {
  if (chunk.kind === "tool_call") return "running";
  if (chunk.kind === "tool_progress") return "running";
  if (chunk.kind === "tool_result") {
    const structuredFailure = explicitFailureStatus(chunk.structuredContent);
    if (structuredFailure !== undefined) return structuredFailure ? "failed" : "succeeded";
    return looksLikeFailure(chunk.content) ? "failed" : "succeeded";
  }
  return "completed";
}

function titleForChunk(chunk: MessageChunk): string {
  if (chunk.kind === "tool_call")
    return chunk.toolName ? `Tool call: ${chunk.toolName}` : "Tool call";
  if (chunk.kind === "tool_result") {
    return chunk.toolName ? `Tool result: ${chunk.toolName}` : "Tool result";
  }
  if (chunk.kind === "tool_progress") {
    return chunk.toolName ? `Tool progress: ${chunk.toolName}` : "Tool progress";
  }
  if (chunk.kind === "thinking") return "Thinking";
  return chunk.agent ? `Message from ${chunk.agent}` : "Message";
}

function summarizeChunk(chunk: MessageChunk, payload: unknown): string {
  if (
    chunk.kind === "tool_call" ||
    chunk.kind === "tool_progress" ||
    chunk.kind === "tool_result"
  ) {
    const summary = typeof payload === "string" ? payload : stableJson(payload);
    return summary.length > 180 ? `${summary.slice(0, 177)}...` : summary;
  }

  const text = chunk.content.replace(/\s+/g, " ").trim();
  if (!text) return chunk.kind;
  return text.length > 180 ? `${text.slice(0, 177)}...` : text;
}

function parseContentPayload(content: string): unknown {
  const trimmed = content.trim();
  if (!trimmed) return "";
  try {
    return JSON.parse(trimmed);
  } catch {
    return { text: content };
  }
}

function looksLikeFailure(content: string): boolean {
  const trimmed = content.trim();
  if (!trimmed) return false;
  try {
    const parsed = JSON.parse(trimmed);
    return explicitFailureStatus(parsed) ?? false;
  } catch {
    return /\b(error|failed|exception)\b/i.test(trimmed);
  }
}

function explicitFailureStatus(payload: unknown): boolean | undefined {
  if (!isObjectRecord(payload)) return undefined;

  const status = typeof payload.status === "string" ? payload.status.toLowerCase() : "";
  if (["failed", "error", "canceled", "cancelled", "declined"].includes(status)) return true;
  if (["succeeded", "success", "completed", "complete", "ok", "skipped"].includes(status)) {
    return false;
  }

  const isError = payload.isError ?? payload.is_error;
  if (typeof isError === "boolean") return isError;
  if (typeof payload.failed === "boolean") return payload.failed;

  const exitCode = payload.exit_code ?? payload.exitCode;
  if (typeof exitCode === "number" && Number.isFinite(exitCode)) return exitCode !== 0;
  if (typeof exitCode === "string" && /^-?\d+$/.test(exitCode.trim())) {
    return Number.parseInt(exitCode, 10) !== 0;
  }

  if ("error" in payload) {
    const error = payload.error;
    return error !== undefined && error !== null && error !== false && error !== "";
  }
  return undefined;
}

function deterministicRenderEventId(input: unknown): string {
  const digest = stableHash(stableJson(input));
  return `rne_${digest}`;
}

function addSecretIssues(value: unknown, ctx: z.RefinementCtx): void {
  for (const issue of findInlineSecrets(value)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: issue.path,
      message: `Render events must not contain inline secret field "${issue.key}".`,
    });
  }
}

function addAgentProvenanceIssues(value: unknown, ctx: z.RefinementCtx): void {
  addSecretIssues(value, ctx);
  if (!isObjectRecord(value)) return;
  for (const key of ["providerProfileId", "provider_profile_id", "model"]) {
    if (key in value) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: [key],
        message: `Agent provenance must use harnessId plus modelId; field "${key}" is invalid.`,
      });
    }
  }
}

function findInlineSecrets(
  value: unknown,
  path: Array<string | number> = [],
): Array<{ key: string; path: Array<string | number> }> {
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => findInlineSecrets(item, [...path, index]));
  }
  if (!isObjectRecord(value)) return [];

  const issues: Array<{ key: string; path: Array<string | number> }> = [];
  for (const [key, child] of Object.entries(value)) {
    if (isForbiddenSecretKey(key) && child !== REDACTED_VALUE) {
      issues.push({ key, path: [...path, key] });
    }
    issues.push(...findInlineSecrets(child, [...path, key]));
  }
  return issues;
}

function isForbiddenSecretKey(key: string): boolean {
  const normalizedKey = key.toLowerCase().replace(/[^a-z0-9_]/g, "");
  return (
    FORBIDDEN_SECRET_KEY_PATTERN.test(key) && !ALLOWED_SECRET_REFERENCE_KEYS.has(normalizedKey)
  );
}

function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  return `{${Object.entries(value)
    .filter(([, child]) => child !== undefined)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, child]) => `${JSON.stringify(key)}:${stableJson(child)}`)
    .join(",")}}`;
}

function stableHash(value: string): string {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  for (let index = 0; index < value.length; index++) {
    hash ^= BigInt(value.charCodeAt(index));
    hash = BigInt.asUintN(64, hash * prime);
  }
  return hash.toString(16).padStart(16, "0");
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
