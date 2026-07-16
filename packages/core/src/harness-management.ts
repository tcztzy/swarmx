import { z } from "zod";

const REDACTED_VALUE = "[redacted]";

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
  /(api[_-]?key|access[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key|smtp[_-]?password|telemetry[_-]?token|cluster[_-]?password|remote[_-]?compute[_-]?password)/i;

export const HarnessAvailabilitySchema = z.enum([
  "available",
  "missing",
  "installed",
  "unsupported",
  "failed",
  "unknown",
]);

export const HarnessHostScopeSchema = z.enum(["local", "server", "remote", "custom"]);

export const HarnessSelectorSourceSchema = z.enum([
  "none",
  "bare_harness",
  "harness_model",
  "agent_alias",
]);

export const HarnessInvocationStatusSchema = z.enum(["started", "completed", "failed", "canceled"]);

export const HarnessDiscoveryRecordSchema = z
  .object({
    adapterId: z.string().min(1),
    displayName: z.string().min(1),
    availability: HarnessAvailabilitySchema,
    host: HarnessHostScopeSchema.default("local"),
    commandPath: z.string().min(1).optional(),
    version: z.string().min(1).optional(),
    installable: z.boolean().default(false),
    dependencyId: z.string().min(1).optional(),
    statusNote: z.string().optional(),
    checkedAt: z.string().min(1).optional(),
    metadata: z.record(z.string(), z.unknown()).default({}),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const HarnessAgentAliasSchema = z
  .object({
    alias: z.string().min(1),
    agentProfileId: z.string().min(1).optional(),
    harnessId: z.string().min(1),
    modelId: z.string().min(1),
    modelSupplyId: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addHarnessIdentityIssues);

export const HarnessSelectorResolutionSchema = z
  .object({
    matched: z.boolean(),
    source: HarnessSelectorSourceSchema,
    selector: z.string().min(1).optional(),
    canonicalSelector: z.string().min(1).optional(),
    harnessId: z.string().min(1).optional(),
    modelId: z.string().min(1).optional(),
    modelSupplyId: z.string().min(1).optional(),
    agentProfileId: z.string().min(1).optional(),
    prompt: z.string(),
  })
  .passthrough()
  .superRefine(addHarnessIdentityIssues);

export const HarnessInvocationMetadataSchema = z
  .object({
    invocationId: z.string().min(1),
    sessionId: z.string().min(1).optional(),
    triggerMessageId: z.string().min(1).optional(),
    contextPacketId: z.string().min(1).optional(),
    harnessId: z.string().min(1),
    modelId: z.string().min(1),
    modelSupplyId: z.string().min(1).optional(),
    agentProfileId: z.string().min(1).optional(),
    canonicalSelector: z.string().min(1).optional(),
    externalSessionRef: z.string().min(1).optional(),
    status: HarnessInvocationStatusSchema,
    startedAt: z.string().min(1).optional(),
    endedAt: z.string().min(1).optional(),
    error: z.string().optional(),
    metadata: z.record(z.string(), z.unknown()).default({}),
  })
  .passthrough()
  .superRefine(addHarnessIdentityIssues);

export type HarnessAvailability = z.infer<typeof HarnessAvailabilitySchema>;
export type HarnessHostScope = z.infer<typeof HarnessHostScopeSchema>;
export type HarnessSelectorSource = z.infer<typeof HarnessSelectorSourceSchema>;
export type HarnessInvocationStatus = z.infer<typeof HarnessInvocationStatusSchema>;
export type HarnessDiscoveryRecord = z.infer<typeof HarnessDiscoveryRecordSchema>;
export type HarnessAgentAlias = z.infer<typeof HarnessAgentAliasSchema>;
export type HarnessSelectorResolution = z.infer<typeof HarnessSelectorResolutionSchema>;
export type HarnessInvocationMetadata = z.infer<typeof HarnessInvocationMetadataSchema>;

export interface ResolveHarnessSelectorOptions {
  knownHarnesses?: string[];
  agentAliases?: HarnessAgentAlias[];
}

export function parseHarnessDiscoveryRecord(input: unknown): HarnessDiscoveryRecord {
  return HarnessDiscoveryRecordSchema.parse(input);
}

export function parseHarnessInvocationMetadata(input: unknown): HarnessInvocationMetadata {
  return HarnessInvocationMetadataSchema.parse(input);
}

export function resolveHarnessSelector(
  input: string,
  options: ResolveHarnessSelectorOptions = {},
): HarnessSelectorResolution {
  const match = input.match(/^@([A-Za-z0-9_.-]+)(?::([A-Za-z0-9_.-]+))?(?:\s+|$)/);
  if (!match) {
    return HarnessSelectorResolutionSchema.parse({
      matched: false,
      source: "none",
      prompt: input,
    });
  }

  const selector = match[0].trim();
  const id = match[1] as string;
  const suffix = match[2];
  const prompt = input.slice(match[0].length);
  const knownHarnesses = new Set(options.knownHarnesses ?? []);

  if (knownHarnesses.has(id)) {
    return HarnessSelectorResolutionSchema.parse({
      matched: true,
      source: suffix ? "harness_model" : "bare_harness",
      selector,
      canonicalSelector: suffix ? `@${id}:${suffix}` : `@${id}`,
      harnessId: id,
      modelId: suffix,
      prompt,
    });
  }

  if (suffix) {
    throw new Error(`Unknown harness "${id}".`);
  }

  const aliases = (options.agentAliases ?? [])
    .map((alias) => HarnessAgentAliasSchema.parse(alias))
    .filter((alias) => alias.alias === id);
  if (aliases.length > 1) {
    throw new Error(`Ambiguous agent selector "@${id}".`);
  }
  if (aliases.length === 1) {
    const alias = aliases[0] as HarnessAgentAlias;
    return HarnessSelectorResolutionSchema.parse({
      matched: true,
      source: "agent_alias",
      selector,
      canonicalSelector: `@${alias.harnessId}:${alias.modelId}`,
      harnessId: alias.harnessId,
      modelId: alias.modelId,
      modelSupplyId: alias.modelSupplyId,
      agentProfileId: alias.agentProfileId,
      prompt,
    });
  }

  throw new Error(`Unknown harness or agent selector "@${id}".`);
}

function addHarnessIdentityIssues(value: unknown, ctx: z.RefinementCtx): void {
  addSecretIssues(value, ctx);
  if (!isObjectRecord(value)) return;
  for (const key of ["providerProfileId", "provider_profile_id", "adapterId", "adapter_id"]) {
    if (key in value) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: [key],
        message: `Harness Agent identity must use harnessId plus modelId; field "${key}" is invalid.`,
      });
    }
  }
  if (value.modelSupplyId && !value.modelId) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["modelSupplyId"],
      message: "A ModelSupply selection requires a Model id.",
    });
  }
}

function addSecretIssues(value: unknown, ctx: z.RefinementCtx): void {
  for (const issue of findInlineSecrets(value)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: issue.path,
      message: `Harness records must not contain inline secret field "${issue.key}".`,
    });
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

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
