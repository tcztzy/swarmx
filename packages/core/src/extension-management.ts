import { z } from "zod";
import { findInlineSecretFields } from "./secret-scanner.js";

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
  "revoke_trust",
  "grant_permissions",
  "rollback",
]);

const ExtensionPermissionIdSchema = z.string().trim().min(1).max(160);
export const ExtensionActionActorSchema = z.enum(["user", "system", "extension"]);
export const ExtensionAuthorityChangeSchema = z.enum(["none", "reduce", "expand"]);

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
    requestedPermissionIds: z.array(ExtensionPermissionIdSchema).default([]),
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
    requestedPermissionIds: z.array(ExtensionPermissionIdSchema).default([]),
    grantedPermissionIds: z.array(ExtensionPermissionIdSchema).default([]),
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
    const requested = new Set(extension.requestedPermissionIds);
    for (const permissionId of extension.grantedPermissionIds) {
      if (!requested.has(permissionId)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["grantedPermissionIds"],
          message: `Granted Extension permission "${permissionId}" is not requested by the installed revision.`,
        });
      }
    }
  });

export const ExtensionActionRequestSchema = z
  .object({
    action: ExtensionActionKindSchema,
    pluginId: z.string().min(1),
    candidate: ExtensionCandidateSchema.optional(),
    confirmed: z.boolean().default(false),
    permissionIds: z.array(ExtensionPermissionIdSchema).default([]),
    actor: ExtensionActionActorSchema.default("user"),
  })
  .passthrough()
  .superRefine((request, ctx) => {
    addSecretIssues(request, ctx);
    for (const key of Object.keys(request)) {
      if (isKernelAuthorityField(key)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: [key],
          message: `Extension actions cannot modify protected kernel policy field "${key}".`,
        });
      }
    }
  });

export const ExtensionActionPlanSchema = z
  .object({
    actionId: z.string().min(1),
    action: ExtensionActionKindSchema,
    pluginId: z.string().min(1),
    allowed: z.boolean(),
    requiresConfirmation: z.boolean(),
    confirmed: z.boolean(),
    reason: z.string().min(1),
    authorityChange: ExtensionAuthorityChangeSchema.default("none"),
    targetPermissionIds: z.array(ExtensionPermissionIdSchema).default([]),
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
export type ExtensionActionActor = z.infer<typeof ExtensionActionActorSchema>;
export type ExtensionAuthorityChange = z.infer<typeof ExtensionAuthorityChangeSchema>;

export interface ExtensionAuthorityAuditEvent {
  phase: "attempted" | "completed";
  action: "trust" | "revoke_trust" | "grant_permissions";
  pluginId: string;
  authorityChange: "reduce" | "expand";
  permissionIds: string[];
}

export type ExtensionAuthorityAudit = (event: ExtensionAuthorityAuditEvent) => void;

export function planExtensionAction(
  requestInput: unknown,
  installedInput?: unknown,
): ExtensionActionPlan {
  const request = ExtensionActionRequestSchema.parse(requestInput);
  const installed = installedInput ? InstalledExtensionSchema.parse(installedInput) : undefined;
  const targetPermissionIds = uniquePermissionIds(request.permissionIds);
  const currentPermissionIds = installed?.grantedPermissionIds ?? [];
  const authorityChange = permissionAuthorityChange(currentPermissionIds, targetPermissionIds);
  const trustChange = trustAuthorityChange(request.action, installed?.trust);
  const requiresConfirmation =
    ["install", "update", "uninstall", "trust", "revoke_trust", "rollback"].includes(
      request.action,
    ) ||
    (request.action === "grant_permissions" && authorityChange === "expand");
  const actionId = `${request.action}:${request.pluginId}:${
    request.candidate?.revision.revisionId ?? installed?.currentRevision?.revisionId ?? "none"
  }:${targetPermissionIds.join(",") || "none"}`;
  const reject = (reason: string) =>
    ExtensionActionPlanSchema.parse({
      actionId,
      action: request.action,
      pluginId: request.pluginId,
      allowed: false,
      requiresConfirmation,
      confirmed: request.confirmed,
      reason,
      authorityChange:
        request.action === "trust" || request.action === "revoke_trust"
          ? trustChange
          : request.action === "grant_permissions"
            ? authorityChange
            : "none",
      targetPermissionIds,
      before: installed,
      targetRevision: request.candidate?.revision,
    });

  if (request.candidate && request.candidate.pluginId !== request.pluginId) {
    return reject("The candidate does not belong to the requested Extension.");
  }
  if (request.candidate?.trust === "builtin" && request.actor !== "system") {
    return reject("Built-in trust is kernel-owned and cannot be claimed by an Extension package.");
  }
  if (
    request.actor === "extension" &&
    ["trust", "revoke_trust", "grant_permissions"].includes(request.action)
  ) {
    return reject("An Extension cannot change its own trust or kernel-owned authority.");
  }
  if (request.action === "install" && (!request.candidate || installed)) {
    return reject(
      installed ? "The Extension is already installed." : "Install requires a candidate.",
    );
  }
  if (
    [
      "update",
      "uninstall",
      "enable",
      "disable",
      "trust",
      "revoke_trust",
      "grant_permissions",
      "rollback",
    ].includes(request.action) &&
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
  if (request.action === "trust" && installed?.trust === "builtin") {
    return reject("Built-in Extension trust is kernel-owned and cannot be changed.");
  }
  if (request.action === "revoke_trust" && installed?.trust === "builtin") {
    return reject("Built-in Extensions cannot have kernel trust revoked.");
  }
  if (request.action === "enable" && installed?.trust === "untrusted") {
    return reject("An untrusted Extension cannot be enabled.");
  }
  if (request.action === "grant_permissions") {
    if (installed?.trust === "untrusted" && targetPermissionIds.length > 0) {
      return reject("Permissions can only be granted to a trusted Extension.");
    }
    const requested = new Set(installed?.requestedPermissionIds ?? []);
    const undeclared = targetPermissionIds.filter((permissionId) => !requested.has(permissionId));
    if (undeclared.length > 0) {
      return reject(`The installed revision did not request: ${undeclared.join(", ")}.`);
    }
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
    authorityChange:
      request.action === "trust" || request.action === "revoke_trust"
        ? trustChange
        : request.action === "grant_permissions"
          ? authorityChange
          : "none",
    targetPermissionIds,
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
  readonly #audit?: ExtensionAuthorityAudit;

  constructor(
    installed: unknown[] = [],
    now: () => string = () => new Date().toISOString(),
    audit?: ExtensionAuthorityAudit,
  ) {
    for (const item of installed) {
      const parsed = InstalledExtensionSchema.parse(item);
      this.#installed.set(parsed.pluginId, parsed);
    }
    this.#now = now;
    this.#audit = audit;
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

    const auditEvent = authorityAuditEvent(plan);
    if (auditEvent?.authorityChange === "expand" && !this.#audit) {
      throw new Error("Extension authority expansion requires an audit intent writer.");
    }
    if (auditEvent) this.#audit?.({ ...auditEvent, phase: "attempted" });
    const after = this.#applyPlan(plan, request);
    if (auditEvent) this.#audit?.({ ...auditEvent, phase: "completed" });
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
        requestedPermissionIds: uniquePermissionIds(candidate.requestedPermissionIds),
        grantedPermissionIds: [],
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
    if (plan.action === "revoke_trust") {
      return InstalledExtensionSchema.parse({
        ...before,
        trust: "untrusted",
        enabled: false,
        state: "disabled",
        grantedPermissionIds: [],
        updatedAt: now,
      });
    }
    if (plan.action === "grant_permissions") {
      return InstalledExtensionSchema.parse({
        ...before,
        grantedPermissionIds: plan.targetPermissionIds,
        updatedAt: now,
      });
    }
    if (plan.action === "update") {
      const candidate = request.candidate;
      if (!candidate) throw new Error("An allowed update plan must include a candidate.");
      const requestedPermissionIds = uniquePermissionIds(candidate.requestedPermissionIds);
      const trust = lowerExtensionTrust(before.trust, candidate.trust);
      const grantedPermissionIds =
        trust === "untrusted"
          ? []
          : before.grantedPermissionIds.filter((permissionId) =>
              requestedPermissionIds.includes(permissionId),
            );
      const enabled = trust === "untrusted" ? false : before.enabled;
      return InstalledExtensionSchema.parse({
        ...before,
        state: enabled ? "enabled" : "disabled",
        enabled,
        trust,
        requestedPermissionIds,
        grantedPermissionIds,
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

function uniquePermissionIds(permissionIds: readonly string[]): string[] {
  return [
    ...new Set(permissionIds.map((permissionId) => permissionId.trim()).filter(Boolean)),
  ].sort((left, right) => left.localeCompare(right));
}

function permissionAuthorityChange(
  currentInput: readonly string[],
  targetInput: readonly string[],
): "none" | "reduce" | "expand" {
  const current = new Set(currentInput);
  const target = new Set(targetInput);
  if ([...target].some((permissionId) => !current.has(permissionId))) return "expand";
  if ([...current].some((permissionId) => !target.has(permissionId))) return "reduce";
  return "none";
}

function lowerExtensionTrust(
  current: z.infer<typeof ExtensionTrustSchema>,
  candidate: z.infer<typeof ExtensionTrustSchema>,
): z.infer<typeof ExtensionTrustSchema> {
  const rank = { untrusted: 0, local: 1, verified: 2, builtin: 3 } as const;
  return rank[current] <= rank[candidate] ? current : candidate;
}

function authorityAuditEvent(
  plan: ExtensionActionPlan,
): Omit<ExtensionAuthorityAuditEvent, "phase"> | undefined {
  if (plan.authorityChange === "none") return undefined;
  if (
    plan.action !== "trust" &&
    plan.action !== "revoke_trust" &&
    plan.action !== "grant_permissions"
  ) {
    return undefined;
  }
  return {
    action: plan.action,
    pluginId: plan.pluginId,
    authorityChange: plan.authorityChange,
    permissionIds: plan.targetPermissionIds,
  };
}

function trustAuthorityChange(
  action: z.infer<typeof ExtensionActionKindSchema>,
  current: z.infer<typeof ExtensionTrustSchema> | undefined,
): "none" | "reduce" | "expand" {
  if (action === "trust") {
    return current === "verified" || current === "builtin" ? "none" : "expand";
  }
  if (action === "revoke_trust") return current === "untrusted" ? "none" : "reduce";
  return "none";
}

function isKernelAuthorityField(key: string): boolean {
  const normalized = key.replace(/[-_]/gu, "").toLowerCase();
  return [
    "approvalpolicy",
    "auditpolicy",
    "credentialpolicy",
    "credentialstore",
    "permissionpolicy",
    "trustpolicy",
    "truststate",
  ].includes(normalized);
}

function addSecretIssues(value: unknown, ctx: z.RefinementCtx): void {
  for (const issue of findInlineSecretFields(value)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: issue.path,
      message: `Extension metadata must not contain inline secret field "${issue.key}".`,
    });
  }
}
