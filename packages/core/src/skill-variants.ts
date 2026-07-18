import { z } from "zod";

const FORBIDDEN_SECRET_KEY_PATTERN =
  /(api[_-]?key|access[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key)/i;

export const SkillBindingModeSchema = z.enum(["off", "auto", "required"]);
export const SkillDeliveryModeSchema = z.enum([
  "prompt_fragment",
  "host_native_plugin",
  "rules_file",
  "unsupported",
]);
export const SkillVariantSourceSchema = z.enum(["upstream", "local", "evolved", "legacy"]);
export const SkillVariantStatusSchema = z.enum([
  "active",
  "candidate",
  "quarantined",
  "deprecated",
]);

export const SkillVariantTargetSchema = z
  .object({
    agentProfileIds: z.array(z.string().min(1)).default([]),
    modelIds: z.array(z.string().min(1)).default([]),
    modelFamilies: z.array(z.string().min(1)).default([]),
    modelPatterns: z.array(z.string().min(1)).default([]),
    modelCapabilities: z.array(z.string().min(1)).default([]),
    harnessIds: z.array(z.string().min(1)).default([]),
    softwareIds: z.array(z.string().min(1)).default([]),
    platforms: z.array(z.string().min(1)).default([]),
  })
  .passthrough()
  .default({})
  .superRefine(addSecretIssues);

export const SkillDeliverySchema = z
  .object({
    mode: SkillDeliveryModeSchema,
    host: z.string().min(1).optional(),
    contentRef: z.string().min(1).optional(),
    notes: z.string().optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const SkillVariantLineageSchema = z
  .object({
    source: SkillVariantSourceSchema,
    revisionId: z.string().min(1),
    parentRevisionId: z.string().min(1).optional(),
    upstreamRevisionId: z.string().min(1).optional(),
    contentDigest: z.string().min(1).optional(),
    optimizerId: z.string().min(1).optional(),
    optimizerVersion: z.string().min(1).optional(),
    optimizerConfigDigest: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const SkillVariantSchema = z
  .object({
    id: z.string().min(1),
    skillId: z.string().min(1),
    version: z.string().min(1).default("0.0.0"),
    description: z.string().optional(),
    priority: z.number().int().default(0),
    target: SkillVariantTargetSchema,
    delivery: SkillDeliverySchema,
    tokenEstimate: z.number().int().nonnegative().optional(),
    status: SkillVariantStatusSchema.default("active"),
    lineage: SkillVariantLineageSchema,
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const LogicalSkillSchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1).optional(),
    description: z.string().optional(),
    defaultVariantId: z.string().min(1).optional(),
    variants: z.array(SkillVariantSchema).min(1),
  })
  .passthrough()
  .superRefine((skill, ctx) => {
    addSecretIssues(skill, ctx);
    const ids = new Set<string>();
    for (const [index, variant] of skill.variants.entries()) {
      if (variant.skillId !== skill.id) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["variants", index, "skillId"],
          message: `Skill variant "${variant.id}" must belong to logical Skill "${skill.id}".`,
        });
      }
      if (ids.has(variant.id)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["variants", index, "id"],
          message: `Duplicate Skill variant id "${variant.id}".`,
        });
      }
      ids.add(variant.id);
    }
    if (skill.defaultVariantId && !ids.has(skill.defaultVariantId)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["defaultVariantId"],
        message: `Default Skill variant "${skill.defaultVariantId}" is not declared.`,
      });
    }
  });

export const HarnessSkillBindingSchema = z
  .object({
    skillId: z.string().min(1),
    mode: SkillBindingModeSchema.default("auto"),
    variantId: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine((binding, ctx) => {
    addSecretIssues(binding, ctx);
    if (binding.mode === "off" && binding.variantId) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["variantId"],
        message: "An off Skill binding cannot pin a variant.",
      });
    }
  });

export const SkillResolutionContextSchema = z
  .object({
    agentProfileId: z.string().min(1).optional(),
    modelId: z.string().min(1),
    modelFamily: z.string().min(1).optional(),
    modelCapabilities: z.array(z.string().min(1)).default([]),
    harnessId: z.string().min(1),
    softwareId: z.string().min(1).optional(),
    platform: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const ResolvedSkillBindingSchema = z
  .object({
    skillId: z.string().min(1),
    status: z.enum(["resolved", "off", "blocked"]),
    variantId: z.string().min(1).optional(),
    version: z.string().min(1).optional(),
    revisionId: z.string().min(1).optional(),
    contentDigest: z.string().min(1).optional(),
    delivery: SkillDeliverySchema.optional(),
    tokenEstimate: z.number().int().nonnegative().optional(),
    selectionReason: z.string().min(1),
    source: SkillVariantSourceSchema.optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const SkillEvaluationMetricsSchema = z
  .object({
    quality: z.number(),
    safety: z.number(),
    failureRate: z.number().min(0).max(1),
    latencyMs: z.number().nonnegative(),
    contextTokens: z.number().int().nonnegative(),
    costUsd: z.number().nonnegative().optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const SkillEvolutionCandidateSchema = z
  .object({
    id: z.string().min(1),
    skillId: z.string().min(1),
    variantId: z.string().min(1),
    revisionId: z.string().min(1),
    parentRevisionId: z.string().min(1),
    upstreamRevisionId: z.string().min(1).optional(),
    targetAgentId: z.string().min(1),
    targetModelFingerprint: z.string().min(1),
    optimizerId: z.string().min(1),
    optimizerVersion: z.string().min(1),
    optimizerConfigDigest: z.string().min(1),
    createdAt: z.string().datetime(),
    status: z.enum(["proposed", "evaluating", "staged", "rejected", "quarantined"]),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const SkillEvaluationRunSchema = z
  .object({
    id: z.string().min(1),
    candidateRevisionId: z.string().min(1),
    baselineRevisionId: z.string().min(1),
    targetAgentId: z.string().min(1),
    targetModelFingerprint: z.string().min(1),
    datasetDigest: z.string().min(1),
    seed: z.number().int().optional(),
    baseline: SkillEvaluationMetricsSchema,
    candidate: SkillEvaluationMetricsSchema,
    verdict: z.enum(["eligible", "rejected"]),
    completedAt: z.string().datetime(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const SkillPromotionDecisionSchema = z
  .object({
    id: z.string().min(1),
    candidateRevisionId: z.string().min(1),
    evaluationRunId: z.string().min(1),
    decision: z.enum(["promote", "reject", "quarantine", "canary", "rollback"]),
    gate: z.enum(["human", "policy"]),
    reason: z.string().min(1),
    decidedAt: z.string().datetime(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const HarnessProjectContextSchema = z
  .object({
    paths: z.array(z.string().min(1)).default([]),
    instructionFiles: z.array(z.string().min(1)).default([]),
    includeWorkspaceRules: z.boolean().default(true),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const HarnessDeliveryPolicySchema = z
  .object({
    unsupportedSkill: z.enum(["block", "skip"]).default("block"),
    requireContentDigest: z.boolean().default(true),
    allowHostNativePlugins: z.boolean().default(true),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const HarnessPermissionModeSchema = z.enum(["default", "plan", "restricted", "trusted"]);

export const HarnessToolAccessSchema = z.enum(["read", "write", "execute", "control"]);

export const HarnessToolPermissionDecisionSchema = z.enum(["allow", "ask", "deny"]);

export const HarnessPermissionLayerSourceSchema = z.enum([
  "managed",
  "project",
  "personal",
  "agent",
]);

export const HarnessPermissionPolicySchema = z
  .object({
    mode: HarnessPermissionModeSchema.default("default"),
    allowedTools: z.array(z.string().min(1)).default([]),
    deniedTools: z.array(z.string().min(1)).default([]),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const HarnessPermissionPolicyLayerSchema = z
  .object({
    id: z.string().min(1).max(128),
    source: HarnessPermissionLayerSourceSchema,
    label: z.string().min(1).max(160).optional(),
    mode: HarnessPermissionModeSchema.optional(),
    allowedTools: z.array(z.string().min(1).max(128)).max(256).default([]),
    deniedTools: z.array(z.string().min(1).max(128)).max(256).default([]),
    readOnly: z.boolean().default(true),
  })
  .strict()
  .superRefine((layer, ctx) => {
    addSecretIssues(layer, ctx);
    addDuplicateIssues(layer.allowedTools, "allowedTools", ctx);
    addDuplicateIssues(layer.deniedTools, "deniedTools", ctx);
    const denied = new Set(layer.deniedTools);
    for (const [index, toolName] of layer.allowedTools.entries()) {
      if (denied.has(toolName)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["allowedTools", index],
          message: `Tool "${toolName}" cannot be both allowed and denied in one layer.`,
        });
      }
    }
    if (layer.source === "project" && layer.allowedTools.length > 0) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["allowedTools"],
        message: "Project permission policy may restrict authority but cannot pre-approve tools.",
      });
    }
  });

export const ResolvedHarnessPermissionPolicySchema = z
  .object({
    policy: HarnessPermissionPolicySchema,
    layers: z.array(HarnessPermissionPolicyLayerSchema).max(16),
    modeSourceIds: z.array(z.string().min(1)).default([]),
    allowedToolSources: z.record(z.array(z.string().min(1))).default({}),
    deniedToolSources: z.record(z.array(z.string().min(1))).default({}),
  })
  .strict();

export const PermissionApprovalReceiptSchema = z
  .object({
    id: z.string().regex(/^prm_[a-zA-Z0-9_-]{8,}$/),
    createdAt: z.string().datetime(),
    source: z.enum(["direct", "acp"]),
    toolName: z.string().min(1).max(160),
    toolKind: z.string().min(1).max(80).optional(),
    decision: z.enum(["allowed", "rejected", "cancelled"]),
    optionKind: z.enum(["allow_once", "allow_always", "reject_once", "reject_always"]).optional(),
    policySourceIds: z.array(z.string().min(1).max(128)).max(16).default([]),
  })
  .strict()
  .superRefine((receipt, ctx) => {
    addSecretIssues(receipt, ctx);
    const values = [receipt.toolName, receipt.toolKind, ...receipt.policySourceIds].filter(
      (value): value is string => Boolean(value),
    );
    for (const value of values) {
      if (!FORBIDDEN_SECRET_KEY_PATTERN.test(value)) continue;
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Permission approval receipts must not contain secret-bearing values.",
      });
      break;
    }
  });

export const HarnessRecipeSchema = z
  .object({
    id: z.string().min(1),
    revisionId: z.string().min(1),
    name: z.string().min(1).optional(),
    softwareId: z.string().min(1),
    softwareVersion: z.string().min(1).optional(),
    skillBindings: z.array(HarnessSkillBindingSchema).default([]),
    mcpServerIds: z.array(z.string().min(1)).default([]),
    projectContext: HarnessProjectContextSchema.default({}),
    delivery: HarnessDeliveryPolicySchema.default({}),
    permissions: HarnessPermissionPolicySchema.default({}),
    pluginIds: z.array(z.string().min(1)).default([]),
    contentDigest: z.string().min(1).optional(),
    createdAt: z.string().datetime().optional(),
  })
  .passthrough()
  .superRefine((recipe, ctx) => {
    addSecretIssues(recipe, ctx);
    addDuplicateIssues(
      recipe.skillBindings.map((binding) => binding.skillId),
      "skillBindings",
      ctx,
    );
    addDuplicateIssues(recipe.mcpServerIds, "mcpServerIds", ctx);
  });

export type SkillBindingMode = z.infer<typeof SkillBindingModeSchema>;
export type SkillDeliveryMode = z.infer<typeof SkillDeliveryModeSchema>;
export type SkillVariant = z.infer<typeof SkillVariantSchema>;
export type LogicalSkill = z.infer<typeof LogicalSkillSchema>;
export type HarnessSkillBinding = z.infer<typeof HarnessSkillBindingSchema>;
export type HarnessPermissionMode = z.infer<typeof HarnessPermissionModeSchema>;
export type HarnessToolAccess = z.infer<typeof HarnessToolAccessSchema>;
export type HarnessToolPermissionDecision = z.infer<typeof HarnessToolPermissionDecisionSchema>;
export type HarnessPermissionPolicy = z.infer<typeof HarnessPermissionPolicySchema>;
export type HarnessPermissionLayerSource = z.infer<typeof HarnessPermissionLayerSourceSchema>;
export type HarnessPermissionPolicyLayer = z.infer<typeof HarnessPermissionPolicyLayerSchema>;
export type ResolvedHarnessPermissionPolicy = z.infer<typeof ResolvedHarnessPermissionPolicySchema>;
export type PermissionApprovalReceipt = z.infer<typeof PermissionApprovalReceiptSchema>;
export type SkillResolutionContext = z.infer<typeof SkillResolutionContextSchema>;
export type ResolvedSkillBinding = z.infer<typeof ResolvedSkillBindingSchema>;
export type SkillEvolutionCandidate = z.infer<typeof SkillEvolutionCandidateSchema>;
export type SkillEvaluationRun = z.infer<typeof SkillEvaluationRunSchema>;
export type SkillPromotionDecision = z.infer<typeof SkillPromotionDecisionSchema>;

export interface HarnessToolPermissionRequest {
  toolName: string;
  access: HarnessToolAccess;
}

export interface ResolvedHarnessToolPermission {
  decision: HarnessToolPermissionDecision;
  reason:
    | "explicit_deny"
    | "plan_read_only"
    | "explicit_allow"
    | "read_only"
    | HarnessPermissionMode;
  sourceIds: string[];
}

const PERMISSION_MODE_RANK: Record<HarnessPermissionMode, number> = {
  plan: 0,
  restricted: 1,
  default: 2,
  trusted: 3,
};

const PERMISSION_LAYER_RANK: Record<HarnessPermissionLayerSource, number> = {
  managed: 0,
  project: 1,
  personal: 2,
  agent: 3,
};

/** Merges authority ceilings without allowing repository content to grant authority. */
export function resolveHarnessPermissionLayers(input: unknown): ResolvedHarnessPermissionPolicy {
  const layers = z.array(HarnessPermissionPolicyLayerSchema).max(16).parse(input);
  const ids = new Set<string>();
  for (const layer of layers) {
    if (ids.has(layer.id)) throw new Error(`Duplicate permission layer id "${layer.id}".`);
    ids.add(layer.id);
  }
  const sortedLayers = [...layers].sort(
    (left, right) =>
      PERMISSION_LAYER_RANK[left.source] - PERMISSION_LAYER_RANK[right.source] ||
      left.id.localeCompare(right.id),
  );
  const declaredModes = sortedLayers.filter(
    (layer): layer is HarnessPermissionPolicyLayer & { mode: HarnessPermissionMode } =>
      layer.mode !== undefined,
  );
  const mode = declaredModes.reduce<HarnessPermissionMode>(
    (current, layer) =>
      PERMISSION_MODE_RANK[layer.mode] < PERMISSION_MODE_RANK[current] ? layer.mode : current,
    "trusted",
  );
  const effectiveMode = declaredModes.length > 0 ? mode : "default";
  const modeSourceIds = declaredModes
    .filter((layer) => layer.mode === effectiveMode)
    .map((layer) => layer.id);
  const allowedToolSources = collectPermissionToolSources(
    sortedLayers.filter((layer) => layer.source !== "project"),
    "allowedTools",
  );
  const deniedToolSources = collectPermissionToolSources(sortedLayers, "deniedTools");
  for (const toolName of Object.keys(deniedToolSources)) delete allowedToolSources[toolName];
  return ResolvedHarnessPermissionPolicySchema.parse({
    policy: {
      mode: effectiveMode,
      allowedTools: Object.keys(allowedToolSources).sort(),
      deniedTools: Object.keys(deniedToolSources).sort(),
    },
    layers: sortedLayers,
    modeSourceIds,
    allowedToolSources,
    deniedToolSources,
  });
}

/** Resolves direct host-tool authority without changing the enclosing OS sandbox. */
export function resolveHarnessToolPermission(
  policyInput: unknown,
  request: HarnessToolPermissionRequest,
): ResolvedHarnessToolPermission {
  const layered = ResolvedHarnessPermissionPolicySchema.safeParse(policyInput);
  const resolution = layered.success ? layered.data : undefined;
  const policy = resolution?.policy ?? HarnessPermissionPolicySchema.parse(policyInput);
  const toolName = z.string().min(1).parse(request.toolName);
  const access = HarnessToolAccessSchema.parse(request.access);

  if (policy.deniedTools.includes(toolName)) {
    return {
      decision: "deny",
      reason: "explicit_deny",
      sourceIds: resolution?.deniedToolSources[toolName] ?? [],
    };
  }
  if (policy.mode === "plan" && access !== "read") {
    return {
      decision: "deny",
      reason: "plan_read_only",
      sourceIds: resolution?.modeSourceIds ?? [],
    };
  }
  if (policy.allowedTools.includes(toolName)) {
    return {
      decision: "allow",
      reason: "explicit_allow",
      sourceIds: resolution?.allowedToolSources[toolName] ?? [],
    };
  }
  if (access === "read") {
    return { decision: "allow", reason: "read_only", sourceIds: [] };
  }
  if (policy.mode === "default") {
    return { decision: "ask", reason: "default", sourceIds: resolution?.modeSourceIds ?? [] };
  }
  if (policy.mode === "restricted") {
    return {
      decision: "deny",
      reason: "restricted",
      sourceIds: resolution?.modeSourceIds ?? [],
    };
  }
  return {
    decision: "allow",
    reason: policy.mode,
    sourceIds: resolution?.modeSourceIds ?? [],
  };
}

function collectPermissionToolSources(
  layers: HarnessPermissionPolicyLayer[],
  field: "allowedTools" | "deniedTools",
): Record<string, string[]> {
  const result: Record<string, string[]> = {};
  for (const layer of layers) {
    for (const toolName of layer[field]) {
      result[toolName] = [...(result[toolName] ?? []), layer.id];
    }
  }
  return result;
}
export type HarnessProjectContext = z.infer<typeof HarnessProjectContextSchema>;
export type HarnessDeliveryPolicy = z.infer<typeof HarnessDeliveryPolicySchema>;
export type HarnessRecipe = z.infer<typeof HarnessRecipeSchema>;

export interface LegacySkillCapabilityInput {
  id: string;
  name?: string;
  description?: string;
  path?: string;
  canonicalPath?: string;
  contentDigest?: string;
  tokenEstimate?: number;
  variants?: unknown[];
  defaultVariantId?: string;
  hostExposures?: Array<{ status?: string }>;
}

export function normalizeLogicalSkill(input: LegacySkillCapabilityInput): LogicalSkill {
  if (input.variants?.length) {
    return LogicalSkillSchema.parse(input);
  }
  const variantId = `${input.id}:default`;
  const contentRef = input.canonicalPath ?? input.path;
  const hostNative = input.hostExposures?.some((exposure) => exposure.status === "plugin") ?? false;
  const deliveryMode = contentRef
    ? "prompt_fragment"
    : hostNative
      ? "host_native_plugin"
      : "unsupported";
  return LogicalSkillSchema.parse({
    id: input.id,
    name: input.name,
    description: input.description,
    defaultVariantId: variantId,
    variants: [
      {
        id: variantId,
        skillId: input.id,
        version: "0.0.0-legacy",
        target: {},
        delivery: { mode: deliveryMode, contentRef },
        tokenEstimate: input.tokenEstimate,
        lineage: {
          source: "legacy",
          revisionId: variantId,
          contentDigest: input.contentDigest,
        },
      },
    ],
  });
}

export function resolveHarnessSkillBinding(
  skillInput: unknown,
  bindingInput: unknown,
  contextInput: unknown,
): ResolvedSkillBinding {
  const skill = LogicalSkillSchema.parse(skillInput);
  const binding = HarnessSkillBindingSchema.parse(bindingInput);
  const context = SkillResolutionContextSchema.parse(contextInput);
  if (binding.skillId !== skill.id) {
    return blocked(skill.id, `Binding targets unknown logical Skill "${binding.skillId}".`);
  }
  if (binding.mode === "off") {
    return ResolvedSkillBindingSchema.parse({
      skillId: skill.id,
      status: "off",
      selectionReason: "Harness Skill binding is off.",
    });
  }

  if (binding.variantId) {
    const pinned = skill.variants.find((variant) => variant.id === binding.variantId);
    if (!pinned)
      return blocked(skill.id, `Pinned Skill variant "${binding.variantId}" is missing.`);
    if (pinned.status !== "active" || !matchesTarget(pinned, context)) {
      return blocked(skill.id, `Pinned Skill variant "${binding.variantId}" is not compatible.`);
    }
    return resolved(skill.id, pinned, "Pinned variant selected explicitly.", binding.mode);
  }

  const candidates = skill.variants
    .filter((variant) => variant.status === "active" && matchesTarget(variant, context))
    .map((variant) => ({ variant, rank: variantRank(variant, skill, context) }))
    .sort((left, right) => right.rank[0] - left.rank[0] || right.rank[1] - left.rank[1]);
  const best = candidates[0];
  if (!best) {
    return binding.mode === "required"
      ? blocked(skill.id, "No compatible Skill variant is available.")
      : ResolvedSkillBindingSchema.parse({
          skillId: skill.id,
          status: "off",
          selectionReason: "Auto binding found no compatible Skill variant.",
        });
  }
  const tied = candidates.filter(
    (candidate) => candidate.rank[0] === best.rank[0] && candidate.rank[1] === best.rank[1],
  );
  if (tied.length > 1) {
    return blocked(
      skill.id,
      `Ambiguous Skill variants: ${tied.map((candidate) => candidate.variant.id).join(", ")}.`,
    );
  }
  return resolved(skill.id, best.variant, selectionReason(best.rank[0], context), binding.mode);
}

export function evaluateSkillCandidate(
  baselineInput: unknown,
  candidateInput: unknown,
): "eligible" | "rejected" {
  const baseline = SkillEvaluationMetricsSchema.parse(baselineInput);
  const candidate = SkillEvaluationMetricsSchema.parse(candidateInput);
  return candidate.quality > baseline.quality &&
    candidate.safety >= baseline.safety &&
    candidate.failureRate <= baseline.failureRate &&
    candidate.contextTokens <= baseline.contextTokens
    ? "eligible"
    : "rejected";
}

function resolved(
  skillId: string,
  variant: SkillVariant,
  reason: string,
  mode: SkillBindingMode,
): ResolvedSkillBinding {
  if (variant.delivery.mode === "unsupported") {
    return mode === "required"
      ? blocked(skillId, `Skill variant "${variant.id}" has no supported delivery.`)
      : ResolvedSkillBindingSchema.parse({
          skillId,
          status: "off",
          selectionReason: `Auto-selected Skill variant "${variant.id}" has no supported delivery.`,
        });
  }
  return ResolvedSkillBindingSchema.parse({
    skillId,
    status: "resolved",
    variantId: variant.id,
    version: variant.version,
    revisionId: variant.lineage.revisionId,
    contentDigest: variant.lineage.contentDigest,
    delivery: variant.delivery,
    tokenEstimate: variant.tokenEstimate,
    selectionReason: reason,
    source: variant.lineage.source,
  });
}

function blocked(skillId: string, selectionReason: string): ResolvedSkillBinding {
  return ResolvedSkillBindingSchema.parse({ skillId, status: "blocked", selectionReason });
}

function matchesTarget(variant: SkillVariant, context: SkillResolutionContext): boolean {
  const target = variant.target;
  return (
    matchesOptional(target.agentProfileIds, context.agentProfileId) &&
    matchesOptional(target.modelIds, context.modelId) &&
    matchesOptional(target.modelFamilies, context.modelFamily) &&
    matchesPatterns(target.modelPatterns, context.modelId) &&
    target.modelCapabilities.every((capability) =>
      context.modelCapabilities.includes(capability),
    ) &&
    matchesOptional(target.harnessIds, context.harnessId) &&
    matchesOptional(target.softwareIds, context.softwareId) &&
    matchesOptional(target.platforms, context.platform)
  );
}

function matchesOptional(expected: string[], actual: string | undefined): boolean {
  return expected.length === 0 || (actual !== undefined && expected.includes(actual));
}

function matchesPatterns(patterns: string[], value: string): boolean {
  return patterns.every((pattern) => {
    try {
      return new RegExp(pattern).test(value);
    } catch {
      return false;
    }
  });
}

function variantRank(
  variant: SkillVariant,
  skill: LogicalSkill,
  context: SkillResolutionContext,
): [number, number] {
  const target = variant.target;
  if (context.agentProfileId && target.agentProfileIds.includes(context.agentProfileId)) {
    return [600, variant.priority];
  }
  if (target.modelIds.includes(context.modelId) || target.modelPatterns.length > 0) {
    return [500, variant.priority];
  }
  if (context.modelFamily && target.modelFamilies.includes(context.modelFamily)) {
    return [400, variant.priority];
  }
  if (target.modelCapabilities.length > 0) return [350, variant.priority];
  if (
    target.harnessIds.includes(context.harnessId) ||
    (context.softwareId && target.softwareIds.includes(context.softwareId))
  ) {
    return [300, variant.priority];
  }
  if (context.platform && target.platforms.includes(context.platform))
    return [200, variant.priority];
  if (skill.defaultVariantId === variant.id || isUntargeted(target)) return [100, variant.priority];
  return [0, variant.priority];
}

function isUntargeted(target: z.infer<typeof SkillVariantTargetSchema>): boolean {
  return [
    target.agentProfileIds,
    target.modelIds,
    target.modelFamilies,
    target.modelPatterns,
    target.modelCapabilities,
    target.harnessIds,
    target.softwareIds,
    target.platforms,
  ].every((values) => values.length === 0);
}

function selectionReason(rank: number, context: SkillResolutionContext): string {
  if (rank >= 600) return `Exact Agent profile match for "${context.agentProfileId}".`;
  if (rank >= 500) return `Exact Model match for "${context.modelId}".`;
  if (rank >= 400) return `Model family match for "${context.modelFamily}".`;
  if (rank >= 350) return "Model capability match.";
  if (rank >= 300) return `Harness match for "${context.harnessId}".`;
  if (rank >= 200) return `Platform match for "${context.platform}".`;
  return "Default Skill variant selected.";
}

function addSecretIssues(value: unknown, ctx: z.RefinementCtx): void {
  visit(value, [], ctx);
}

function addDuplicateIssues(
  values: string[],
  path: "skillBindings" | "mcpServerIds" | "allowedTools" | "deniedTools",
  ctx: z.RefinementCtx,
): void {
  const seen = new Set<string>();
  values.forEach((value, index) => {
    if (seen.has(value)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: [path, index],
        message:
          path === "skillBindings" || path === "mcpServerIds"
            ? `Harness recipe contains duplicate ${path === "skillBindings" ? "Skill" : "MCP"} id "${value}".`
            : `Permission layer contains duplicate tool "${value}".`,
      });
    }
    seen.add(value);
  });
}

function visit(value: unknown, path: Array<string | number>, ctx: z.RefinementCtx): void {
  if (Array.isArray(value)) {
    value.forEach((item, index) => visit(item, [...path, index], ctx));
    return;
  }
  if (!value || typeof value !== "object") return;
  for (const [key, child] of Object.entries(value)) {
    if (FORBIDDEN_SECRET_KEY_PATTERN.test(key)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: [...path, key],
        message: `Skill metadata must not contain inline secret field "${key}".`,
      });
    }
    visit(child, [...path, key], ctx);
  }
}
