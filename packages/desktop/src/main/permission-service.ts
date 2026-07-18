import { randomUUID } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import {
  type HarnessPermissionPolicy,
  type HarnessPermissionPolicyLayer,
  HarnessPermissionPolicyLayerSchema,
  type PermissionApprovalReceipt,
  PermissionApprovalReceiptSchema,
  type ResolvedHarnessPermissionPolicy,
  resolveHarnessPermissionLayers,
} from "@swarmx/core";
import type { DesktopSettingsStoreLike } from "./settings-store.js";

const MANAGED_PERMISSION_POLICY_ENV = "SWARMX_MANAGED_PERMISSION_POLICY";
const PROJECT_PERMISSION_POLICY_PATH = path.join(".swarmx", "permissions.json");
const MAX_POLICY_BYTES = 64 * 1024;
const MAX_RECEIPTS = 200;

export interface PermissionLayerStatus {
  id: string;
  source: HarnessPermissionPolicyLayer["source"];
  label: string;
  configured: boolean;
  readOnly: boolean;
  mode?: HarnessPermissionPolicyLayer["mode"];
  allowedTools: string[];
  deniedTools: string[];
  error?: string;
}

export interface DesktopPermissionStatus {
  personalPolicy: HarnessPermissionPolicyLayer;
  layers: PermissionLayerStatus[];
  effective?: ResolvedHarnessPermissionPolicy;
  blocked: boolean;
  projectPolicyPath: string;
  approvalReceipts: PermissionApprovalReceipt[];
}

export interface ResolveDesktopPermissionOptions {
  cwd?: string;
  agentId?: string;
  agentPolicy?: HarnessPermissionPolicy;
}

export interface RecordPermissionDecisionInput {
  source: PermissionApprovalReceipt["source"];
  toolName: string;
  toolKind?: string;
  decision: PermissionApprovalReceipt["decision"];
  optionKind?: PermissionApprovalReceipt["optionKind"];
  policySourceIds?: string[];
}

export interface PermissionServiceOptions {
  env?: Record<string, string | undefined>;
  now?: () => string;
  id?: () => string;
}

/** Owns secret-free permission policy loading and bounded approval history. */
export class PermissionService {
  readonly #settings: DesktopSettingsStoreLike;
  readonly #env: Record<string, string | undefined>;
  readonly #now: () => string;
  readonly #id: () => string;

  constructor(settings: DesktopSettingsStoreLike, options: PermissionServiceOptions = {}) {
    this.#settings = settings;
    this.#env = options.env ?? process.env;
    this.#now = options.now ?? (() => new Date().toISOString());
    this.#id = options.id ?? (() => `prm_${randomUUID()}`);
  }

  async status(options: ResolveDesktopPermissionOptions = {}): Promise<DesktopPermissionStatus> {
    const settings = await this.#settings.read();
    const managed = await this.#loadManagedPolicy();
    const project = await this.#loadProjectPolicy(options.cwd);
    const agent = options.agentPolicy
      ? permissionLayer({
          id: `agent:${options.agentId?.trim() || "selected"}`,
          source: "agent",
          label: options.agentId?.trim() ? `Agent · ${options.agentId.trim()}` : "Selected Agent",
          readOnly: false,
          policy: options.agentPolicy,
        })
      : undefined;
    const blocked = Boolean(managed.error || project.error);
    const configuredLayers = [
      ...(managed.layer ? [managed.layer] : []),
      ...(project.layer ? [project.layer] : []),
      settings.permissions.personalPolicy,
      ...(agent ? [agent] : []),
    ];
    return {
      personalPolicy: settings.permissions.personalPolicy,
      layers: [
        permissionLayerStatus("managed", "Managed policy", managed),
        permissionLayerStatus("project", "Project policy", project),
        permissionLayerStatus("personal", "Personal defaults", {
          layer: settings.permissions.personalPolicy,
        }),
        ...(agent
          ? [permissionLayerStatus("agent", agent.label ?? "Selected Agent", { layer: agent })]
          : []),
      ],
      ...(!blocked ? { effective: resolveHarnessPermissionLayers(configuredLayers) } : {}),
      blocked,
      projectPolicyPath: PROJECT_PERMISSION_POLICY_PATH,
      approvalReceipts: settings.permissions.approvalReceipts,
    };
  }

  async resolve(
    options: ResolveDesktopPermissionOptions,
  ): Promise<ResolvedHarnessPermissionPolicy> {
    if (!options.agentPolicy) throw new Error("Agent permission policy is required.");
    const status = await this.status(options);
    const invalid = status.layers.find((layer) => layer.error);
    if (invalid) throw new Error(`${invalid.label} is invalid: ${invalid.error}`);
    if (!status.effective) throw new Error("Effective permission policy is unavailable.");
    return status.effective;
  }

  async savePersonalPolicy(input: unknown): Promise<HarnessPermissionPolicyLayer> {
    const policy = HarnessPermissionPolicyLayerSchema.parse({
      ...(isRecord(input) ? input : {}),
      id: "personal",
      source: "personal",
      label: "Personal defaults",
      readOnly: false,
    });
    await this.#settings.update((settings) => ({
      ...settings,
      permissions: { ...settings.permissions, personalPolicy: policy },
    }));
    return policy;
  }

  async recordDecision(input: RecordPermissionDecisionInput): Promise<PermissionApprovalReceipt> {
    const receipt = PermissionApprovalReceiptSchema.parse({
      id: this.#id(),
      createdAt: this.#now(),
      source: input.source,
      toolName: input.toolName,
      ...(input.toolKind ? { toolKind: input.toolKind } : {}),
      decision: input.decision,
      ...(input.optionKind ? { optionKind: input.optionKind } : {}),
      policySourceIds: input.policySourceIds ?? [],
    });
    await this.#settings.update((settings) => ({
      ...settings,
      permissions: {
        ...settings.permissions,
        approvalReceipts: [receipt, ...settings.permissions.approvalReceipts].slice(
          0,
          MAX_RECEIPTS,
        ),
      },
    }));
    return receipt;
  }

  async #loadManagedPolicy(): Promise<LoadedPermissionLayer> {
    const source = this.#env[MANAGED_PERMISSION_POLICY_ENV]?.trim();
    if (!source) return {};
    if (Buffer.byteLength(source, "utf8") > MAX_POLICY_BYTES) {
      return { error: `policy exceeds ${MAX_POLICY_BYTES} bytes` };
    }
    try {
      return {
        layer: permissionLayer({
          id: "managed",
          source: "managed",
          label: "Managed policy",
          readOnly: true,
          policy: JSON.parse(source),
        }),
      };
    } catch (error) {
      return { error: safeErrorMessage(error) };
    }
  }

  async #loadProjectPolicy(cwd?: string): Promise<LoadedPermissionLayer> {
    if (!cwd) return {};
    const policyPath = path.join(cwd, PROJECT_PERMISSION_POLICY_PATH);
    try {
      const info = await stat(policyPath);
      if (!info.isFile()) return { error: "configured path is not a regular file" };
      if (info.size > MAX_POLICY_BYTES)
        return { error: `policy exceeds ${MAX_POLICY_BYTES} bytes` };
      return {
        layer: permissionLayer({
          id: "project",
          source: "project",
          label: "Project policy",
          readOnly: true,
          policy: JSON.parse(await readFile(policyPath, "utf8")),
        }),
      };
    } catch (error) {
      if (isMissingPathError(error)) return {};
      return { error: safeErrorMessage(error) };
    }
  }
}

interface LoadedPermissionLayer {
  layer?: HarnessPermissionPolicyLayer;
  error?: string;
}

function permissionLayer(input: {
  id: string;
  source: HarnessPermissionPolicyLayer["source"];
  label: string;
  readOnly: boolean;
  policy: unknown;
}): HarnessPermissionPolicyLayer {
  if (!isRecord(input.policy)) throw new Error("permission policy must be an object");
  return HarnessPermissionPolicyLayerSchema.parse({
    ...input.policy,
    id: input.id,
    source: input.source,
    label: input.label,
    readOnly: input.readOnly,
  });
}

function permissionLayerStatus(
  source: HarnessPermissionPolicyLayer["source"],
  label: string,
  loaded: LoadedPermissionLayer,
): PermissionLayerStatus {
  return {
    id: loaded.layer?.id ?? source,
    source,
    label,
    configured: Boolean(loaded.layer),
    readOnly: loaded.layer?.readOnly ?? source !== "personal",
    ...(loaded.layer?.mode ? { mode: loaded.layer.mode } : {}),
    allowedTools: loaded.layer?.allowedTools ?? [],
    deniedTools: loaded.layer?.deniedTools ?? [],
    ...(loaded.error ? { error: loaded.error } : {}),
  };
}

function isMissingPathError(error: unknown): boolean {
  return isRecord(error) && error.code === "ENOENT";
}

function safeErrorMessage(error: unknown): string {
  if (error instanceof SyntaxError) return "policy JSON is invalid";
  if (isRecord(error) && error.name === "ZodError") {
    const issues = Array.isArray(error.issues) ? error.issues : [];
    const messages = issues.flatMap((issue) =>
      isRecord(issue) && typeof issue.message === "string" ? [issue.message] : [],
    );
    if (messages.some((message) => /cannot pre-approve/i.test(message))) {
      return "Project policy may restrict authority but cannot pre-approve tools";
    }
    if (messages.some((message) => /both allowed and denied/i.test(message))) {
      return "policy contains conflicting allow and deny rules";
    }
    if (messages.some((message) => /duplicate tool/i.test(message))) {
      return "policy contains duplicate tool rules";
    }
    if (messages.some((message) => /secret/i.test(message))) {
      return "policy contains a forbidden secret-bearing field";
    }
    return "policy shape or values are invalid";
  }
  if (error instanceof Error) return error.message.slice(0, 240);
  return "policy could not be read";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
