import { z } from "zod";
import { stableHash, stableJson } from "./canonical-json.js";
import { ExtensionTrustSchema, InstalledExtensionSchema } from "./extension-management.js";

const IdentifierSchema = z.string().trim().min(1).max(160);
const CapabilitySchema = z
  .string()
  .trim()
  .min(3)
  .max(192)
  .regex(/^[A-Za-z][A-Za-z0-9_.-]*:[^\s]+$/u);

export const ExtensionHostObservationSchema = z
  .object({
    source: z
      .object({
        type: z.string().trim().min(1).max(64),
        locator: z.string().trim().min(1).max(4_096).optional(),
      })
      .strict(),
    contentDigest: z.string().trim().min(1).max(256),
  })
  .strict();

export const ExtensionRunPhaseSchema = z.enum(["inventory", "preflight", "runtime"]);

export const ExtensionCompositionDeclarationSchema = z
  .object({
    phase: z.string().trim().min(1).default("inventory"),
    provides: z.array(CapabilitySchema).default([]),
    requires: z.array(CapabilitySchema).default([]),
    conflicts: z.array(CapabilitySchema).default([]),
    tools: z.array(IdentifierSchema).default([]),
    before: z.array(IdentifierSchema).default([]),
    after: z.array(IdentifierSchema).default([]),
    orderSensitive: z.boolean().default(false),
  })
  .passthrough();

const PermissionInputSchema = z
  .object({
    id: IdentifierSchema,
    kind: z.string().trim().min(1).max(64),
    access: z.string().trim().min(1).max(64).default("read"),
    required: z.boolean().default(false),
  })
  .passthrough();

const BundleInputSchema = z
  .object({
    id: IdentifierSchema,
    name: z.string().trim().min(1).max(256),
    version: z.string().trim().min(1).max(128),
    enabled: z.boolean().optional(),
    trust: ExtensionTrustSchema.default("local"),
    source: z
      .object({
        type: z.string().trim().min(1).max(64),
        path: z.string().trim().min(1).max(4_096).optional(),
        marketplace: z.string().trim().min(1).max(256).optional(),
        package: z.string().trim().min(1).max(512).optional(),
      })
      .passthrough()
      .optional(),
    integrity: z.string().trim().min(1).max(256).optional(),
    hostObservation: ExtensionHostObservationSchema.optional(),
    composition: ExtensionCompositionDeclarationSchema.prefault({}),
    capabilities: z.record(z.string(), z.array(z.unknown())).default({}),
  })
  .passthrough();

export const ExtensionCompositionPreflightInputSchema = z
  .object({
    bundles: z.array(BundleInputSchema).max(1_024),
    selectedExtensionIds: z.array(IdentifierSchema).max(1_024).optional(),
    installedExtensions: z.array(InstalledExtensionSchema).max(1_024).default([]),
  })
  .strict();

export const ExtensionCompositionIssueCodeSchema = z.enum([
  "ambiguous_order",
  "dependency_cycle",
  "duplicate_capability",
  "duplicate_extension",
  "duplicate_tool",
  "executable_not_installed",
  "explicit_conflict",
  "extension_disabled",
  "integrity_mismatch",
  "invalid_phase",
  "missing_dependency",
  "missing_extension",
  "missing_order_target",
  "permission_missing",
  "permission_optional",
  "phase_violation",
  "protected_capability",
  "provider_conflict",
  "untrusted_executable",
  "untrusted_sensitive_permission",
  "unobserved_executable",
]);

export const ExtensionCompositionIssueSchema = z
  .object({
    severity: z.enum(["error", "warning"]),
    code: ExtensionCompositionIssueCodeSchema,
    message: z.string().min(1).max(1_024),
    extensionId: IdentifierSchema.optional(),
    relatedExtensionId: IdentifierSchema.optional(),
    capability: CapabilitySchema.optional(),
  })
  .strict();

export const ExtensionCompositionPreviewSchema = z
  .object({
    id: IdentifierSchema,
    version: z.string().min(1),
    loadReason: z.string().min(1),
    phase: ExtensionRunPhaseSchema,
    source: z
      .object({
        type: z.string().min(1),
        locator: z.string().min(1).optional(),
      })
      .strict(),
    integrity: z.string().min(1).optional(),
    trust: ExtensionTrustSchema,
    execution: z.enum(["declarative", "external_process"]),
    provides: z.array(CapabilitySchema),
    requires: z.array(CapabilitySchema),
    conflicts: z.array(CapabilitySchema),
    before: z.array(IdentifierSchema),
    after: z.array(IdentifierSchema),
    tools: z.array(IdentifierSchema),
    permissions: z
      .object({
        requested: z.array(IdentifierSchema),
        granted: z.array(IdentifierSchema),
        missing: z.array(IdentifierSchema),
      })
      .strict(),
  })
  .strict();

export const ExtensionCompositionPreflightSchema = z
  .object({
    status: z.enum(["ready", "blocked"]),
    fingerprint: z.string().regex(/^ecp_[a-f0-9]{16}$/),
    loadOrder: z.array(IdentifierSchema),
    extensions: z.array(ExtensionCompositionPreviewSchema),
    issues: z.array(ExtensionCompositionIssueSchema),
    sideEffects: z.array(z.string()).max(0),
  })
  .strict();

export type ExtensionRunPhase = z.infer<typeof ExtensionRunPhaseSchema>;
export type ExtensionHostObservation = z.infer<typeof ExtensionHostObservationSchema>;
export type ExtensionCompositionDeclaration = z.infer<typeof ExtensionCompositionDeclarationSchema>;
export type ExtensionCompositionIssue = z.infer<typeof ExtensionCompositionIssueSchema>;
export type ExtensionCompositionPreview = z.infer<typeof ExtensionCompositionPreviewSchema>;
export type ExtensionCompositionPreflight = z.infer<typeof ExtensionCompositionPreflightSchema>;

export const PROTECTED_EXTENSION_CAPABILITIES = [
  "kernel:approval-policy",
  "kernel:audit-policy",
  "kernel:composition-enforcement",
  "kernel:credential-store",
  "kernel:execution-authority",
  "kernel:extension-trust",
  "kernel:foreground-completion",
  "kernel:identity",
  "kernel:session-authority",
  "kernel:task-authority",
] as const;

const PROTECTED_CAPABILITIES = new Set<string>(PROTECTED_EXTENSION_CAPABILITIES);
const PHASE_RANK: Record<ExtensionRunPhase, number> = {
  inventory: 0,
  preflight: 1,
  runtime: 2,
};

const CAPABILITY_FIELDS: Readonly<Record<string, string>> = {
  software: "software",
  skills: "skill",
  mcpServers: "mcp",
  models: "model",
  modelSupplies: "model_supply",
  providers: "provider",
  harnesses: "harness",
  agents: "agent_profile",
  appConnectors: "connector",
  uiContributions: "ui",
  commands: "command",
  lspServers: "lsp",
  hooks: "hook",
  monitors: "monitor",
  outputStyles: "output_style",
  settings: "setting",
  assets: "asset",
  authPolicies: "auth_policy",
};

interface NormalizedNode {
  id: string;
  version: string;
  phase: ExtensionRunPhase;
  invalidPhase?: string;
  source: { type: string; locator?: string };
  integrity?: string;
  trust: z.infer<typeof ExtensionTrustSchema>;
  enabled: boolean;
  executable: boolean;
  orderSensitive: boolean;
  provides: string[];
  requires: string[];
  conflicts: string[];
  before: string[];
  after: string[];
  tools: string[];
  permissions: Array<z.infer<typeof PermissionInputSchema>>;
  grantedPermissions: string[];
  hostObservation?: ExtensionHostObservation;
}

export function preflightExtensionComposition(input: unknown): ExtensionCompositionPreflight {
  const parsed = ExtensionCompositionPreflightInputSchema.parse(input);
  const issues: ExtensionCompositionIssue[] = [];
  const installedById = new Map(
    parsed.installedExtensions.map((extension) => [extension.pluginId, extension]),
  );
  const grouped = new Map<string, Array<z.infer<typeof BundleInputSchema>>>();
  for (const bundle of parsed.bundles) {
    const items = grouped.get(bundle.id) ?? [];
    items.push(bundle);
    grouped.set(bundle.id, items);
  }

  const nodes = new Map<string, NormalizedNode>();
  for (const [id, candidates] of [...grouped.entries()].sort(([left], [right]) =>
    left.localeCompare(right),
  )) {
    const sorted = [...candidates].sort((left, right) =>
      stableJson(left).localeCompare(stableJson(right)),
    );
    if (sorted.length > 1) {
      issues.push({
        severity: "error",
        code: "duplicate_extension",
        extensionId: id,
        message: `Extension identity "${id}" is declared ${sorted.length} times.`,
      });
    }
    const primary = sorted[0];
    if (primary) nodes.set(id, normalizeNode(primary, installedById.get(id), issues));
  }

  const selectedIds = sortedUnique(parsed.selectedExtensionIds ?? [...nodes.keys()]);
  const loadedReasons = new Map<string, string>();
  for (const id of selectedIds) {
    if (!nodes.has(id)) {
      issues.push({
        severity: "error",
        code: "missing_extension",
        extensionId: id,
        message: `Selected Extension "${id}" is not present in inventory.`,
      });
      continue;
    }
    loadedReasons.set(id, "selected");
  }

  closeDependencies(nodes, loadedReasons, issues);
  const loaded = () =>
    [...loadedReasons.keys()].flatMap((id) => {
      const node = nodes.get(id);
      return node ? [node] : [];
    });
  validateLoadedNodes(loaded(), nodes, installedById, loadedReasons, issues);

  const { order, cycle } = topologicalOrder(loaded(), issues);
  if (cycle.length > 0) {
    issues.push({
      severity: "error",
      code: "dependency_cycle",
      extensionId: cycle[0],
      message: `Extension dependency/order cycle: ${cycle.join(" -> ")}.`,
    });
  }
  const orderedIds = [...order, ...cycle.filter((id) => !order.includes(id))];
  const previews = orderedIds.flatMap((id) => {
    const node = nodes.get(id);
    if (!node) return [];
    const requested = sortedUnique(node.permissions.map((permission) => permission.id));
    const granted = sortedUnique(
      node.grantedPermissions.filter((permission) => requested.includes(permission)),
    );
    const missing = sortedUnique(
      node.permissions
        .filter((permission) => permission.required && !granted.includes(permission.id))
        .map((permission) => permission.id),
    );
    return [
      {
        id: node.id,
        version: node.version,
        loadReason: loadedReasons.get(node.id) ?? "selected",
        phase: node.phase,
        source: node.source,
        ...(node.integrity ? { integrity: node.integrity } : {}),
        trust: node.trust,
        execution: node.executable ? ("external_process" as const) : ("declarative" as const),
        provides: node.provides,
        requires: node.requires,
        conflicts: node.conflicts,
        before: node.before,
        after: node.after,
        tools: node.tools,
        permissions: { requested, granted, missing },
      },
    ];
  });

  const sortedIssues = deduplicateIssues(issues).sort(compareIssues);
  const base = {
    status: sortedIssues.some((issue) => issue.severity === "error")
      ? ("blocked" as const)
      : ("ready" as const),
    loadOrder: orderedIds,
    extensions: previews,
    issues: sortedIssues,
    sideEffects: [] as string[],
  };
  return ExtensionCompositionPreflightSchema.parse({
    ...base,
    fingerprint: `ecp_${stableHash(stableJson(base))}`,
  });
}

function normalizeNode(
  bundle: z.infer<typeof BundleInputSchema>,
  installed: z.infer<typeof InstalledExtensionSchema> | undefined,
  issues: ExtensionCompositionIssue[],
): NormalizedNode {
  const parsedPhase = ExtensionRunPhaseSchema.safeParse(bundle.composition.phase);
  const phase = parsedPhase.success ? parsedPhase.data : "runtime";
  if (!parsedPhase.success) {
    issues.push({
      severity: "error",
      code: "invalid_phase",
      extensionId: bundle.id,
      message: `Extension "${bundle.id}" declares unsupported phase "${bundle.composition.phase}".`,
    });
  }
  const provides = [...bundle.composition.provides];
  for (const [field, prefix] of Object.entries(CAPABILITY_FIELDS)) {
    for (const item of bundle.capabilities[field] ?? []) {
      const id = recordIdentifier(item);
      if (id) provides.push(`${prefix}:${id}`);
    }
  }
  const tools = [
    ...bundle.composition.tools,
    ...(bundle.capabilities.mcpServers ?? []).flatMap((item) => {
      if (!isRecord(item) || !isRecord(item.projectService)) return [];
      const declared = item.projectService.tools;
      return Array.isArray(declared)
        ? declared.filter(
            (tool): tool is string => typeof tool === "string" && tool.trim().length > 0,
          )
        : [];
    }),
  ];
  const permissions = (bundle.capabilities.permissions ?? []).flatMap((permission) => {
    const parsed = PermissionInputSchema.safeParse(permission);
    return parsed.success ? [parsed.data] : [];
  });
  const trust = effectiveTrust(bundle.id, installed?.trust, bundle.hostObservation);
  const sourceLocator = bundle.source?.path ?? bundle.source?.package ?? bundle.source?.marketplace;
  const integrity =
    bundle.hostObservation?.contentDigest ??
    installed?.currentRevision?.contentDigest ??
    bundle.integrity;
  if (
    installed?.currentRevision &&
    bundle.hostObservation?.contentDigest &&
    installed.currentRevision.contentDigest !== bundle.hostObservation.contentDigest
  ) {
    issues.push({
      severity: "error",
      code: "integrity_mismatch",
      extensionId: bundle.id,
      message: `Extension "${bundle.id}" does not match its installed immutable revision digest.`,
    });
  }
  return {
    id: bundle.id,
    version: bundle.version,
    phase,
    ...(parsedPhase.success ? {} : { invalidPhase: bundle.composition.phase }),
    source: {
      type: bundle.hostObservation?.source.type ?? bundle.source?.type ?? "manifest",
      ...(bundle.hostObservation?.source.locator
        ? { locator: bundle.hostObservation.source.locator }
        : sourceLocator
          ? { locator: sourceLocator }
          : {}),
    },
    ...(integrity ? { integrity } : {}),
    trust,
    enabled:
      bundle.enabled !== false && (installed ? isEnabledInstalledExtension(installed) : true),
    executable: hasExecutableCapability(bundle.capabilities),
    orderSensitive: bundle.composition.orderSensitive,
    provides: sortedUnique(provides),
    requires: sortedUnique(bundle.composition.requires),
    conflicts: sortedUnique(bundle.composition.conflicts),
    before: sortedUnique(bundle.composition.before),
    after: sortedUnique(bundle.composition.after),
    tools: sortedUnique(tools),
    permissions,
    grantedPermissions: sortedUnique(installed?.grantedPermissionIds ?? []),
    ...(bundle.hostObservation ? { hostObservation: bundle.hostObservation } : {}),
  };
}

function closeDependencies(
  nodes: Map<string, NormalizedNode>,
  loadedReasons: Map<string, string>,
  issues: ExtensionCompositionIssue[],
): void {
  let changed = true;
  while (changed) {
    changed = false;
    const providers = capabilityOwners([...nodes.values()]);
    for (const id of [...loadedReasons.keys()].sort()) {
      const node = nodes.get(id);
      if (!node) continue;
      for (const requirement of node.requires) {
        const ownerIds = requirement.startsWith("extension:")
          ? [requirement.slice("extension:".length)].filter((ownerId) => nodes.has(ownerId))
          : (providers.get(requirement) ?? []);
        if (ownerIds.length === 0) {
          issues.push({
            severity: "error",
            code: "missing_dependency",
            extensionId: node.id,
            capability: requirement,
            message: `Extension "${node.id}" requires missing capability "${requirement}".`,
          });
          continue;
        }
        if (ownerIds.length > 1) {
          issues.push({
            severity: "error",
            code: "duplicate_capability",
            extensionId: ownerIds[0],
            relatedExtensionId: ownerIds[1],
            capability: requirement,
            message: `Required capability "${requirement}" has multiple owners: ${ownerIds.join(", ")}.`,
          });
          for (const ownerId of ownerIds) {
            if (!loadedReasons.has(ownerId)) {
              loadedReasons.set(ownerId, `ambiguous_requirement:${node.id}`);
              changed = true;
            }
          }
          continue;
        }
        const ownerId = ownerIds[0];
        if (!ownerId) continue;
        if (!loadedReasons.has(ownerId)) {
          loadedReasons.set(ownerId, `required_by:${node.id}`);
          changed = true;
        }
      }
      for (const relatedId of [...node.before, ...node.after]) {
        if (nodes.has(relatedId) && !loadedReasons.has(relatedId)) {
          loadedReasons.set(relatedId, `ordered_with:${node.id}`);
          changed = true;
        }
      }
    }
  }
}

function validateLoadedNodes(
  loaded: NormalizedNode[],
  allNodes: Map<string, NormalizedNode>,
  installedById: Map<string, z.infer<typeof InstalledExtensionSchema>>,
  loadedReasons: Map<string, string>,
  issues: ExtensionCompositionIssue[],
): void {
  const owners = capabilityOwners(loaded);
  for (const [capability, ownerIds] of [...owners.entries()].sort(([left], [right]) =>
    left.localeCompare(right),
  )) {
    if (PROTECTED_CAPABILITIES.has(capability)) {
      for (const extensionId of ownerIds) {
        if (loaded.find((node) => node.id === extensionId)?.trust === "builtin") continue;
        issues.push({
          severity: "error",
          code: "protected_capability",
          extensionId,
          capability,
          message: `Extension "${extensionId}" cannot replace protected capability "${capability}".`,
        });
      }
    }
    if (ownerIds.length > 1) {
      issues.push({
        severity: "error",
        code: capability.startsWith("provider:") ? "provider_conflict" : "duplicate_capability",
        extensionId: ownerIds[0],
        relatedExtensionId: ownerIds[1],
        capability,
        message: `Capability "${capability}" has multiple owners: ${ownerIds.join(", ")}.`,
      });
    }
  }

  const toolOwners = new Map<string, string[]>();
  for (const node of loaded) {
    for (const tool of node.tools) {
      const ids = toolOwners.get(tool) ?? [];
      ids.push(node.id);
      toolOwners.set(tool, ids);
    }
  }
  for (const [tool, ownerIds] of [...toolOwners.entries()].sort(([left], [right]) =>
    left.localeCompare(right),
  )) {
    if (ownerIds.length < 2) continue;
    issues.push({
      severity: "error",
      code: "duplicate_tool",
      extensionId: ownerIds[0],
      relatedExtensionId: ownerIds[1],
      capability: `tool:${tool}`,
      message: `Model-facing tool "${tool}" has multiple owners: ${ownerIds.join(", ")}.`,
    });
  }

  for (const node of loaded) {
    if (!node.enabled) {
      issues.push({
        severity: "error",
        code: "extension_disabled",
        extensionId: node.id,
        message: `Extension "${node.id}" is disabled or its trust was revoked.`,
      });
    }
    if (node.trust === "untrusted" && node.executable) {
      issues.push({
        severity: "error",
        code: "untrusted_executable",
        extensionId: node.id,
        message: `Untrusted Extension "${node.id}" cannot start executable code.`,
      });
    }
    const installed = installedById.get(node.id);
    if (node.executable && node.trust !== "builtin" && !isEnabledInstalledExtension(installed)) {
      issues.push({
        severity: "error",
        code: "executable_not_installed",
        extensionId: node.id,
        message: `Executable Extension "${node.id}" has no enabled installed revision.`,
      });
    }
    if (node.executable && node.trust !== "builtin" && !node.hostObservation) {
      issues.push({
        severity: "error",
        code: "unobserved_executable",
        extensionId: node.id,
        message: `Executable Extension "${node.id}" has no host-observed source and content digest.`,
      });
    }
    for (const capability of node.conflicts) {
      const conflictingOwners = capability.startsWith("extension:")
        ? [capability.slice("extension:".length)].filter((id) => loadedReasons.has(id))
        : (owners.get(capability) ?? []);
      for (const relatedExtensionId of conflictingOwners) {
        if (relatedExtensionId === node.id) continue;
        issues.push({
          severity: "error",
          code: "explicit_conflict",
          extensionId: node.id,
          relatedExtensionId,
          capability,
          message: `Extension "${node.id}" conflicts with "${relatedExtensionId}" through "${capability}".`,
        });
      }
    }
    for (const relatedId of [...node.before, ...node.after]) {
      if (allNodes.has(relatedId) && loadedReasons.has(relatedId)) continue;
      issues.push({
        severity: "error",
        code: "missing_order_target",
        extensionId: node.id,
        relatedExtensionId: relatedId,
        message: `Extension "${node.id}" orders against missing Extension "${relatedId}".`,
      });
    }
    const granted = new Set(node.grantedPermissions);
    for (const permission of node.permissions) {
      if (!granted.has(permission.id)) {
        issues.push({
          severity: permission.required ? "error" : "warning",
          code: permission.required ? "permission_missing" : "permission_optional",
          extensionId: node.id,
          message: `${permission.required ? "Required" : "Optional"} permission "${permission.id}" is not granted to Extension "${node.id}".`,
        });
      }
      if (node.trust === "untrusted" && isSensitivePermission(permission)) {
        issues.push({
          severity: "error",
          code: "untrusted_sensitive_permission",
          extensionId: node.id,
          message: `Untrusted Extension "${node.id}" cannot receive sensitive permission "${permission.id}".`,
        });
      }
    }
    if (installed?.currentRevision && node.integrity === undefined) {
      issues.push({
        severity: "error",
        code: "integrity_mismatch",
        extensionId: node.id,
        message: `Installed Extension "${node.id}" has no verifiable integrity digest.`,
      });
    }
  }
}

function topologicalOrder(
  nodes: NormalizedNode[],
  issues: ExtensionCompositionIssue[],
): { order: string[]; cycle: string[] } {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const owners = capabilityOwners(nodes);
  const edges = new Map(nodes.map((node) => [node.id, new Set<string>()]));
  const addEdge = (from: string, to: string): void => {
    if (from !== to && byId.has(from) && byId.has(to)) edges.get(from)?.add(to);
  };
  for (const node of nodes) {
    for (const requirement of node.requires) {
      const providerIds = requirement.startsWith("extension:")
        ? [requirement.slice("extension:".length)]
        : (owners.get(requirement) ?? []);
      if (providerIds.length === 1) {
        const providerId = providerIds[0];
        const provider = providerId ? byId.get(providerId) : undefined;
        if (provider && PHASE_RANK[provider.phase] > PHASE_RANK[node.phase]) {
          issues.push({
            severity: "error",
            code: "phase_violation",
            extensionId: node.id,
            relatedExtensionId: provider.id,
            capability: requirement,
            message: `Extension "${node.id}" cannot require later-phase capability "${requirement}" from "${provider.id}".`,
          });
        }
        if (provider) addEdge(provider.id, node.id);
      }
    }
    for (const target of node.before) addEdge(node.id, target);
    for (const target of node.after) addEdge(target, node.id);
  }
  for (const left of nodes) {
    for (const right of nodes) {
      if (PHASE_RANK[left.phase] < PHASE_RANK[right.phase]) addEdge(left.id, right.id);
    }
  }

  const indegree = new Map(nodes.map((node) => [node.id, 0]));
  for (const targets of edges.values()) {
    for (const target of targets) indegree.set(target, (indegree.get(target) ?? 0) + 1);
  }
  const remaining = new Set(nodes.map((node) => node.id));
  const order: string[] = [];
  let ambiguityReported = false;
  while (remaining.size > 0) {
    const ready = [...remaining]
      .filter((id) => indegree.get(id) === 0)
      .sort((left, right) => compareNodeIds(byId, left, right));
    if (ready.length === 0) break;
    const id = ready[0];
    if (!id) break;
    const firstPhase = byId.get(id)?.phase;
    const samePhase = ready.filter((id) => byId.get(id)?.phase === firstPhase);
    if (
      !ambiguityReported &&
      samePhase.length > 1 &&
      samePhase.some((id) => byId.get(id)?.orderSensitive)
    ) {
      ambiguityReported = true;
      issues.push({
        severity: "error",
        code: "ambiguous_order",
        extensionId: samePhase[0],
        relatedExtensionId: samePhase[1],
        message: `Order-sensitive Extensions require an explicit relation: ${samePhase.join(", ")}.`,
      });
    }
    remaining.delete(id);
    order.push(id);
    for (const target of edges.get(id) ?? []) {
      indegree.set(target, (indegree.get(target) ?? 1) - 1);
    }
  }
  return {
    order,
    cycle: [...remaining].sort((left, right) => compareNodeIds(byId, left, right)),
  };
}

function capabilityOwners(nodes: NormalizedNode[]): Map<string, string[]> {
  const owners = new Map<string, string[]>();
  for (const node of nodes) {
    for (const capability of node.provides) {
      const ids = owners.get(capability) ?? [];
      ids.push(node.id);
      owners.set(capability, ids.sort());
    }
  }
  return owners;
}

function hasExecutableCapability(capabilities: Record<string, unknown[]>): boolean {
  return (
    (capabilities.software ?? []).some((item) => isRecord(item) && item.command !== undefined) ||
    (capabilities.commands ?? []).some((item) => isRecord(item) && item.command !== undefined) ||
    (capabilities.hooks ?? []).some((item) => isRecord(item) && item.command !== undefined) ||
    (capabilities.lspServers ?? []).some((item) => isRecord(item) && item.command !== undefined) ||
    (capabilities.mcpServers ?? []).some((item) => {
      if (!isRecord(item)) return false;
      if (typeof item.entrypoint === "string" && item.entrypoint.trim()) return true;
      return isRecord(item.server) && item.server.type === "stdio";
    }) ||
    (capabilities.appConnectors ?? []).some(
      (item) => isRecord(item) && typeof item.entrypoint === "string" && item.entrypoint.trim(),
    ) ||
    (capabilities.harnesses ?? []).some(
      (item) => isRecord(item) && isRecord(item.backend) && item.backend.type === "custom",
    )
  );
}

function isSensitivePermission(permission: z.infer<typeof PermissionInputSchema>): boolean {
  return (
    ["network", "process", "secret", "hook", "monitor"].includes(permission.kind) ||
    ["write", "execute", "network", "admin", "custom"].includes(permission.access)
  );
}

function effectiveTrust(
  bundleId: string,
  installed: z.infer<typeof ExtensionTrustSchema> | undefined,
  observation: ExtensionHostObservation | undefined,
): z.infer<typeof ExtensionTrustSchema> {
  if (bundleId === "swarmx.builtin" && observation?.source.type === "builtin") {
    return "builtin";
  }
  if (!installed || installed === "builtin") return "untrusted";
  return installed;
}

function isEnabledInstalledExtension(
  installed: z.infer<typeof InstalledExtensionSchema> | undefined,
): boolean {
  return Boolean(
    installed?.enabled &&
      installed.currentRevision &&
      !new Set(["available", "disabled", "blocked", "diverged", "conflict"]).has(installed.state),
  );
}

function recordIdentifier(value: unknown): string | undefined {
  if (!isRecord(value)) return undefined;
  const id = value.id;
  return typeof id === "string" && id.trim() ? id.trim() : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sortedUnique(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter(Boolean))].sort((left, right) =>
    left.localeCompare(right),
  );
}

function compareNodes(left: NormalizedNode, right: NormalizedNode): number {
  return PHASE_RANK[left.phase] - PHASE_RANK[right.phase] || left.id.localeCompare(right.id);
}

function compareNodeIds(
  nodes: ReadonlyMap<string, NormalizedNode>,
  leftId: string,
  rightId: string,
): number {
  const left = nodes.get(leftId);
  const right = nodes.get(rightId);
  return left && right ? compareNodes(left, right) : leftId.localeCompare(rightId);
}

function compareIssues(left: ExtensionCompositionIssue, right: ExtensionCompositionIssue): number {
  return (
    left.severity.localeCompare(right.severity) ||
    left.code.localeCompare(right.code) ||
    (left.extensionId ?? "").localeCompare(right.extensionId ?? "") ||
    (left.relatedExtensionId ?? "").localeCompare(right.relatedExtensionId ?? "") ||
    (left.capability ?? "").localeCompare(right.capability ?? "") ||
    left.message.localeCompare(right.message)
  );
}

function deduplicateIssues(issues: ExtensionCompositionIssue[]): ExtensionCompositionIssue[] {
  const unique = new Map<string, ExtensionCompositionIssue>();
  for (const issue of issues) unique.set(stableJson(issue), issue);
  return [...unique.values()];
}
