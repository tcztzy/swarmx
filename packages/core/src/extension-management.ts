import { z } from "zod";

const FORBIDDEN_SECRET_KEY_PATTERN =
  /(api[_-]?key|access[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key)/i;

export const ExtensionTrustSchema = z.enum(["builtin", "local", "verified", "untrusted"]);
export const ExtensionInstallStateSchema = z.enum([
  "available",
  "installed",
  "enabled",
  "disabled",
  "update_available",
  "blocked",
  "diverged",
  "conflict",
  "pinned",
]);
export const ExtensionActionKindSchema = z.enum([
  "refresh_source",
  "install",
  "update",
  "uninstall",
  "enable",
  "disable",
  "trust",
  "rollback",
]);

export const ExtensionMarketplaceSourceSchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1),
    kind: z.enum(["local_path", "remote_catalog", "host_native", "registry"]),
    location: z.string().min(1),
    trust: ExtensionTrustSchema.default("untrusted"),
    enabled: z.boolean().default(true),
    readOnly: z.boolean().default(false),
    refreshedAt: z.string().datetime().optional(),
    catalogDigest: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine((source, ctx) => {
    addSecretIssues(source, ctx);
    if (
      (source.kind === "remote_catalog" || source.kind === "registry") &&
      !isSafeRemoteLocation(source.location)
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["location"],
        message: "Remote Extension sources must use HTTPS and must not embed credentials.",
      });
    }
  });

export const ExtensionRevisionSchema = z
  .object({
    revisionId: z.string().min(1),
    version: z.string().min(1),
    contentDigest: z.string().min(1),
    sourceId: z.string().min(1),
    packageRef: z.string().min(1).optional(),
    publishedAt: z.string().datetime().optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const ExtensionCandidateSchema = z
  .object({
    pluginId: z.string().min(1),
    name: z.string().min(1),
    trust: ExtensionTrustSchema,
    revision: ExtensionRevisionSchema,
    description: z.string().optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const ExtensionMarketplaceCatalogSchema = z.preprocess(
  (input) => {
    if (!input || typeof input !== "object" || Array.isArray(input)) return input;
    const document = input as Record<string, unknown>;
    return {
      ...document,
      candidates: document.candidates ?? document.plugins ?? document.entries,
    };
  },
  z
    .object({
      schemaVersion: z.literal(1).default(1),
      candidates: z.array(ExtensionCandidateSchema).default([]),
      generatedAt: z.string().datetime().optional(),
    })
    .passthrough()
    .superRefine(addSecretIssues),
);

export const InstalledExtensionSchema = z
  .object({
    pluginId: z.string().min(1),
    name: z.string().min(1),
    state: ExtensionInstallStateSchema,
    enabled: z.boolean(),
    trust: ExtensionTrustSchema,
    currentRevision: ExtensionRevisionSchema.optional(),
    previousRevisions: z.array(ExtensionRevisionSchema).default([]),
    pinnedRevisionId: z.string().min(1).optional(),
    installedAt: z.string().datetime().optional(),
    updatedAt: z.string().datetime().optional(),
  })
  .passthrough()
  .superRefine((extension, ctx) => {
    addSecretIssues(extension, ctx);
    if (extension.state !== "available" && !extension.currentRevision) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["currentRevision"],
        message: "An installed Extension must retain its immutable current revision.",
      });
    }
  });

export const ExtensionActionRequestSchema = z
  .object({
    action: ExtensionActionKindSchema,
    pluginId: z.string().min(1),
    candidate: ExtensionCandidateSchema.optional(),
    confirmed: z.boolean().default(false),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const ExtensionActionPlanSchema = z
  .object({
    actionId: z.string().min(1),
    action: ExtensionActionKindSchema,
    pluginId: z.string().min(1),
    allowed: z.boolean(),
    requiresConfirmation: z.boolean(),
    confirmed: z.boolean(),
    reason: z.string().min(1),
    before: InstalledExtensionSchema.optional(),
    targetRevision: ExtensionRevisionSchema.optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const ExtensionActionReceiptSchema = z
  .object({
    actionId: z.string().min(1),
    action: ExtensionActionKindSchema,
    pluginId: z.string().min(1),
    status: z.enum(["applied", "rejected", "failed"]),
    before: InstalledExtensionSchema.optional(),
    after: InstalledExtensionSchema.optional(),
    appliedAt: z.string().datetime(),
    message: z.string().min(1),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export type ExtensionMarketplaceSource = z.infer<typeof ExtensionMarketplaceSourceSchema>;
export type ExtensionRevision = z.infer<typeof ExtensionRevisionSchema>;
export type ExtensionCandidate = z.infer<typeof ExtensionCandidateSchema>;
export type ExtensionMarketplaceCatalog = z.infer<typeof ExtensionMarketplaceCatalogSchema>;
export type InstalledExtension = z.infer<typeof InstalledExtensionSchema>;
export type ExtensionActionRequest = z.infer<typeof ExtensionActionRequestSchema>;
export type ExtensionActionPlan = z.infer<typeof ExtensionActionPlanSchema>;
export type ExtensionActionReceipt = z.infer<typeof ExtensionActionReceiptSchema>;

export function planExtensionAction(
  requestInput: unknown,
  installedInput?: unknown,
): ExtensionActionPlan {
  const request = ExtensionActionRequestSchema.parse(requestInput);
  const installed = installedInput ? InstalledExtensionSchema.parse(installedInput) : undefined;
  const requiresConfirmation = ["install", "update", "uninstall", "trust", "rollback"].includes(
    request.action,
  );
  const actionId = `${request.action}:${request.pluginId}:${request.candidate?.revision.revisionId ?? installed?.currentRevision?.revisionId ?? "none"}`;
  const reject = (reason: string) =>
    ExtensionActionPlanSchema.parse({
      actionId,
      action: request.action,
      pluginId: request.pluginId,
      allowed: false,
      requiresConfirmation,
      confirmed: request.confirmed,
      reason,
      before: installed,
      targetRevision: request.candidate?.revision,
    });

  if (request.candidate && request.candidate.pluginId !== request.pluginId) {
    return reject("The candidate does not belong to the requested Extension.");
  }
  if (request.action === "install" && (!request.candidate || installed)) {
    return reject(
      installed ? "The Extension is already installed." : "Install requires a candidate.",
    );
  }
  if (
    ["update", "uninstall", "enable", "disable", "trust", "rollback"].includes(request.action) &&
    !installed
  ) {
    return reject("The Extension is not installed.");
  }
  if (request.action === "update") {
    if (!request.candidate) return reject("Update requires an upstream candidate.");
    if (installed?.pinnedRevisionId)
      return reject("Pinned Extensions must be unpinned before update.");
    if (installed?.currentRevision?.contentDigest === request.candidate.revision.contentDigest) {
      return reject("The installed Extension already matches the upstream revision.");
    }
  }
  if (request.action === "uninstall" && installed?.trust === "builtin") {
    return reject("Built-in Extensions cannot be uninstalled.");
  }
  if (request.action === "rollback" && installed?.previousRevisions.length === 0) {
    return reject("No previous immutable revision is available for rollback.");
  }
  if (requiresConfirmation && !request.confirmed) {
    return reject("This Extension action requires explicit confirmation.");
  }

  return ExtensionActionPlanSchema.parse({
    actionId,
    action: request.action,
    pluginId: request.pluginId,
    allowed: true,
    requiresConfirmation,
    confirmed: request.confirmed,
    reason: "Extension action is ready to apply.",
    before: installed,
    targetRevision:
      request.action === "rollback"
        ? installed?.previousRevisions.at(-1)
        : request.candidate?.revision,
  });
}

export class ExtensionLifecycleManager {
  readonly #installed = new Map<string, InstalledExtension>();
  readonly #now: () => string;

  constructor(installed: unknown[] = [], now: () => string = () => new Date().toISOString()) {
    for (const item of installed) {
      const parsed = InstalledExtensionSchema.parse(item);
      this.#installed.set(parsed.pluginId, parsed);
    }
    this.#now = now;
  }

  list(): InstalledExtension[] {
    return [...this.#installed.values()].map((item) => InstalledExtensionSchema.parse(item));
  }

  plan(request: unknown): ExtensionActionPlan {
    const parsed = ExtensionActionRequestSchema.parse(request);
    return planExtensionAction(parsed, this.#installed.get(parsed.pluginId));
  }

  apply(requestInput: unknown): ExtensionActionReceipt {
    const request = ExtensionActionRequestSchema.parse(requestInput);
    const plan = this.plan(request);
    if (!plan.allowed) {
      return ExtensionActionReceiptSchema.parse({
        actionId: plan.actionId,
        action: plan.action,
        pluginId: plan.pluginId,
        status: "rejected",
        before: plan.before,
        appliedAt: this.#now(),
        message: plan.reason,
      });
    }

    const after = this.#applyPlan(plan, request);
    if (after) this.#installed.set(after.pluginId, after);
    else this.#installed.delete(plan.pluginId);
    return ExtensionActionReceiptSchema.parse({
      actionId: plan.actionId,
      action: plan.action,
      pluginId: plan.pluginId,
      status: "applied",
      before: plan.before,
      after,
      appliedAt: this.#now(),
      message: `Extension ${plan.action} applied.`,
    });
  }

  #applyPlan(
    plan: ExtensionActionPlan,
    request: ExtensionActionRequest,
  ): InstalledExtension | undefined {
    const now = this.#now();
    if (plan.action === "refresh_source") return plan.before;
    if (plan.action === "uninstall") return undefined;
    if (plan.action === "install") {
      const candidate = request.candidate;
      if (!candidate) throw new Error("An allowed install plan must include a candidate.");
      return InstalledExtensionSchema.parse({
        pluginId: candidate.pluginId,
        name: candidate.name,
        state: "enabled",
        enabled: true,
        trust: candidate.trust,
        currentRevision: candidate.revision,
        previousRevisions: [],
        installedAt: now,
        updatedAt: now,
      });
    }

    const before = plan.before;
    if (!before) throw new Error(`An allowed ${plan.action} plan must include installed state.`);
    if (plan.action === "enable" || plan.action === "disable") {
      const enabled = plan.action === "enable";
      return InstalledExtensionSchema.parse({
        ...before,
        enabled,
        state: enabled ? "enabled" : "disabled",
        updatedAt: now,
      });
    }
    if (plan.action === "trust") {
      return InstalledExtensionSchema.parse({ ...before, trust: "verified", updatedAt: now });
    }
    if (plan.action === "update") {
      const candidate = request.candidate;
      if (!candidate) throw new Error("An allowed update plan must include a candidate.");
      return InstalledExtensionSchema.parse({
        ...before,
        state: before.enabled ? "enabled" : "disabled",
        trust: candidate.trust,
        currentRevision: candidate.revision,
        previousRevisions: before.currentRevision
          ? [...before.previousRevisions, before.currentRevision]
          : before.previousRevisions,
        updatedAt: now,
      });
    }
    const target = plan.targetRevision;
    if (!target) throw new Error("An allowed rollback plan must include a target revision.");
    const remainsPinned = Boolean(before.pinnedRevisionId);
    return InstalledExtensionSchema.parse({
      ...before,
      state: remainsPinned ? "pinned" : before.enabled ? "enabled" : "disabled",
      pinnedRevisionId: remainsPinned ? target.revisionId : undefined,
      currentRevision: target,
      previousRevisions: before.previousRevisions.filter(
        (revision) => revision.revisionId !== target.revisionId,
      ),
      updatedAt: now,
    });
  }
}

function isSafeRemoteLocation(location: string): boolean {
  try {
    const url = new URL(location);
    return url.protocol === "https:" && !url.username && !url.password;
  } catch {
    return false;
  }
}

function addSecretIssues(value: unknown, ctx: z.RefinementCtx): void {
  visit(value, [], ctx);
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
        message: `Extension metadata must not contain inline secret field "${key}".`,
      });
    }
    visit(child, [...path, key], ctx);
  }
}
