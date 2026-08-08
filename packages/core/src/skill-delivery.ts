import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { z } from "zod";
import type { SkillDeliveryMode } from "./skill-variants.js";

const SHA256_DIGEST_PATTERN = /^sha256:[a-f0-9]{64}$/;
const MAX_SKILL_FRAGMENT_BYTES = 256 * 1024;
const MAX_TOTAL_DELIVERED_BYTES = 512 * 1024;

/**
 * A verified, request-scoped `prompt_fragment` Skill delivery. Content is loaded
 * from a trusted content-addressed artifact and digest-verified before it may
 * enter Agent instructions. It never mutates the persisted SwarmConfig or the
 * original Skill files.
 */
export const SkillInstructionDeliverySchema = z
  .object({
    skillId: z.string().min(1).max(256),
    variantId: z.string().min(1).max(256),
    revisionId: z.string().min(1).max(256),
    contentDigest: z.string().regex(SHA256_DIGEST_PATTERN),
    mode: z.literal("prompt_fragment"),
    content: z.string().max(MAX_SKILL_FRAGMENT_BYTES),
  })
  .strict()
  .superRefine((delivery, ctx) => {
    if (utf8ByteLength(delivery.content) > MAX_SKILL_FRAGMENT_BYTES) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["content"],
        message: `Skill fragment exceeds ${MAX_SKILL_FRAGMENT_BYTES} bytes.`,
      });
    }
    const digest = sha256Hex(delivery.content);
    if (`sha256:${digest}` !== delivery.contentDigest) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["contentDigest"],
        message: "Skill fragment content digest does not match its content.",
      });
    }
  });

export type SkillInstructionDelivery = z.infer<typeof SkillInstructionDeliverySchema>;

export interface LoadSkillFragmentOptions {
  expectedDigest: string;
  maxBytes?: number;
  mediaType?: string;
}

export const NATIVE_DIRECT_HARNESSES = new Set(["swarmx", "kimi_code", "opencode", "custom"]);

export class SkillDeliveryError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "SkillDeliveryError";
    this.code = code;
  }
}

/**
 * Reads Skill Markdown from a trusted content-addressed artifact (for example a
 * TaskRuntimeStore blob file) and verifies its SHA-256 digest and size bound.
 */
export async function loadSkillFragmentContent(
  contentPath: string,
  options: LoadSkillFragmentOptions,
): Promise<string> {
  const maxBytes = options.maxBytes ?? MAX_SKILL_FRAGMENT_BYTES;
  const stat = await readFile(contentPath);
  if (stat.byteLength > maxBytes) {
    throw new SkillDeliveryError(
      "oversized_artifact",
      `Skill content artifact is ${stat.byteLength} bytes; maximum is ${maxBytes}.`,
    );
  }
  const digest = sha256Hex(stat);
  if (`sha256:${digest}` !== options.expectedDigest) {
    throw new SkillDeliveryError(
      "digest_mismatch",
      "Skill content artifact digest does not match its declared digest.",
    );
  }
  return stat.toString("utf8");
}

export interface SkillDeliveryGuardInput {
  deliveryMode: SkillDeliveryMode | "unsupported";
  harnessId?: string;
  modelControl?: "direct" | "session" | "unsupported";
}

/**
 * Refuses Skill delivery when the injected semantics cannot be guaranteed:
 * native-plugin, rules-file, unsupported, or external ACP Harness delivery is
 * never silently approximated.
 */
export function assertPromptFragmentDeliverable(input: SkillDeliveryGuardInput): void {
  if (input.deliveryMode !== "prompt_fragment") {
    throw new SkillDeliveryError(
      "unsupported_delivery",
      `Skill delivery mode "${input.deliveryMode}" cannot be injected into model instructions.`,
    );
  }
  if (input.modelControl === "session") {
    throw new SkillDeliveryError(
      "external_harness",
      "External ACP Harnesses own their prompts; Skill prompt_fragment injection is unsupported.",
    );
  }
  if (input.modelControl === "unsupported") {
    throw new SkillDeliveryError(
      "unsupported_harness",
      `Harness "${input.harnessId ?? "unknown"}" has no defined prompt injection semantics.`,
    );
  }
}

export function parseSkillInstructionDelivery(input: unknown): SkillInstructionDelivery {
  return SkillInstructionDeliverySchema.parse(input);
}

/**
 * Appends verified Skill fragments to the base Agent instructions. The result is
 * the exact text that becomes the model-visible system/developer instructions.
 */
export function buildDeliveredInstructions(
  baseInstructions: string,
  deliveries: readonly SkillInstructionDelivery[],
): string {
  if (deliveries.length === 0) return baseInstructions;
  let totalBytes = utf8ByteLength(baseInstructions);
  const blocks: string[] = [];
  for (const delivery of deliveries) {
    const block = [
      `# Skill: ${delivery.skillId}`,
      `- variant: ${delivery.variantId}`,
      `- revision: ${delivery.revisionId}`,
      `- content digest: ${delivery.contentDigest}`,
      "",
      delivery.content.trim(),
    ].join("\n");
    totalBytes += utf8ByteLength(block) + 2;
    if (totalBytes > MAX_TOTAL_DELIVERED_BYTES) {
      throw new SkillDeliveryError(
        "oversized_instructions",
        "Delivered Skill instructions exceed the total instruction budget.",
      );
    }
    blocks.push(block);
  }
  return [baseInstructions, ...blocks].filter((block) => block.trim()).join("\n\n");
}

export function sha256Hex(input: string | Uint8Array): string {
  return createHash("sha256").update(input).digest("hex");
}

export function utf8ByteLength(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}
