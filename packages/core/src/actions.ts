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

const idWithPrefix = (prefix: string) =>
  z.string().regex(new RegExp(`^${prefix}[A-Za-z0-9][A-Za-z0-9_-]*$`), `Must use ${prefix} prefix`);

export const ActionKindSchema = z.enum([
  "marketplace.refresh",
  "plugin.install",
  "plugin.update",
  "plugin.uninstall",
  "plugin.enable",
  "plugin.disable",
  "harness.install",
  "dependency.install_managed",
  "dependency.open_installer",
  "dependency.use_existing",
  "dependency.unavailable",
  "mcp.start",
  "hook.enable",
  "command.rerun",
  "artifact.open",
  "raw_payload.reveal",
  "trust.change",
]);

export const ActionRiskSchema = z.enum([
  "read_only",
  "writes_settings",
  "writes_files",
  "downloads_code",
  "executes_code",
  "network",
  "secrets",
  "destructive",
  "trust_change",
]);

export const ActionHostSchema = z.enum(["local", "server", "remote", "custom"]);

export const ActionSourceKindSchema = z.enum([
  "extension",
  "plugin_catalog",
  "dependency",
  "harness",
  "render_event",
  "conversation",
  "user",
  "system",
]);

export const ActionSourceRefSchema = z
  .object({
    kind: ActionSourceKindSchema,
    id: z.string().min(1),
    label: z.string().min(1).optional(),
    path: z.string().min(1).optional(),
    url: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const ActionIntentSchema = z
  .object({
    actionId: idWithPrefix("act_"),
    kind: ActionKindSchema,
    title: z.string().min(1),
    description: z.string().optional(),
    host: ActionHostSchema.default("local"),
    source: ActionSourceRefSchema,
    target: ActionSourceRefSchema.optional(),
    risks: z.array(ActionRiskSchema).default([]),
    confirmationRequired: z.boolean(),
    confirmationText: z.string().min(1).optional(),
    payload: z.record(z.string(), z.unknown()).default({}),
    createdAt: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine((intent, ctx) => {
    addSecretIssues(intent, ctx);
    if (intent.confirmationRequired && !intent.confirmationText) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["confirmationText"],
        message: "Confirmation-required actions must provide confirmation text.",
      });
    }
    if (intent.risks.includes("read_only") && intent.risks.length > 1) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["risks"],
        message: "The read_only risk cannot be combined with mutating or sensitive risks.",
      });
    }
  });

export const ActionConfirmationSchema = z
  .object({
    actionId: idWithPrefix("act_"),
    confirmed: z.boolean(),
    confirmedAt: z.string().min(1).optional(),
    actor: z.string().min(1).optional(),
    acknowledgement: z.string().optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const ConfirmedActionSchema = z.object({
  intent: ActionIntentSchema,
  confirmation: ActionConfirmationSchema,
});

export type ActionKind = z.infer<typeof ActionKindSchema>;
export type ActionRisk = z.infer<typeof ActionRiskSchema>;
export type ActionHost = z.infer<typeof ActionHostSchema>;
export type ActionSourceKind = z.infer<typeof ActionSourceKindSchema>;
export type ActionSourceRef = z.infer<typeof ActionSourceRefSchema>;
export type ActionIntent = z.infer<typeof ActionIntentSchema>;
export type ActionConfirmation = z.infer<typeof ActionConfirmationSchema>;
export type ConfirmedAction = z.infer<typeof ConfirmedActionSchema>;

export type CreateActionIntentInput = {
  actionId?: string;
  kind: ActionKind;
  title: string;
  description?: string;
  host?: ActionHost;
  source: ActionSourceRef;
  target?: ActionSourceRef;
  createdAt?: string;
  confirmationRequired?: boolean;
  confirmationText?: string;
  payload?: Record<string, unknown>;
  risks?: ActionRisk[];
};

export function parseActionIntent(input: unknown): ActionIntent {
  return ActionIntentSchema.parse(input);
}

export function parseActionConfirmation(input: unknown): ActionConfirmation {
  return ActionConfirmationSchema.parse(input);
}

export function sanitizeActionPayload(input: unknown): unknown {
  if (Array.isArray(input)) return input.map(sanitizeActionPayload);
  if (!isObjectRecord(input)) return input;

  return Object.fromEntries(
    Object.entries(input).map(([key, value]) => [
      key,
      isForbiddenSecretKey(key) ? REDACTED_VALUE : sanitizeActionPayload(value),
    ]),
  );
}

export function createActionIntent(input: CreateActionIntentInput): ActionIntent {
  const risks = input.risks ?? defaultRisksForActionKind(input.kind);
  const withoutId = {
    kind: input.kind,
    title: input.title,
    description: input.description,
    host: input.host ?? "local",
    source: input.source,
    target: input.target,
    risks,
    confirmationRequired: input.confirmationRequired ?? requiresExplicitConfirmation(risks),
    confirmationText: input.confirmationText,
    payload: sanitizeActionPayload(input.payload ?? {}),
    createdAt: input.createdAt,
  };

  return ActionIntentSchema.parse({
    ...withoutId,
    actionId: input.actionId ?? `act_${stableHash(stableJson(withoutId))}`,
  });
}

export function requiresExplicitConfirmation(risksInput: unknown): boolean {
  const risks = z.array(ActionRiskSchema).parse(risksInput);
  return risks.some((risk) => risk !== "read_only");
}

export function assertActionConfirmed(
  intentInput: unknown,
  confirmationInput: unknown,
): ConfirmedAction {
  const intent = ActionIntentSchema.parse(intentInput);
  const confirmation = ActionConfirmationSchema.parse(confirmationInput);

  if (intent.actionId !== confirmation.actionId) {
    throw new Error(
      `Confirmation action id "${confirmation.actionId}" does not match intent "${intent.actionId}".`,
    );
  }
  if (intent.confirmationRequired && !confirmation.confirmed) {
    throw new Error(`Action "${intent.actionId}" requires explicit confirmation.`);
  }

  return ConfirmedActionSchema.parse({ intent, confirmation });
}

export function actionIntentFromDependencyPlan(input: {
  dependencyId: string;
  action: "use_existing" | "install_managed" | "requires_user_action" | "unavailable";
  reason: string;
  platform?: string;
}): ActionIntent {
  if (input.action === "install_managed") {
    return createActionIntent({
      kind: "dependency.install_managed",
      title: `Install ${input.dependencyId}`,
      description: input.reason,
      source: { kind: "dependency", id: input.dependencyId },
      risks: ["downloads_code", "writes_files", "executes_code"],
      confirmationText: `Install ${input.dependencyId} from the managed dependency manifest.`,
      payload: {
        dependencyId: input.dependencyId,
        platform: input.platform,
      },
    });
  }

  if (input.action === "requires_user_action") {
    return createActionIntent({
      kind: "dependency.open_installer",
      title: `Open installer for ${input.dependencyId}`,
      description: input.reason,
      source: { kind: "dependency", id: input.dependencyId },
      risks: ["downloads_code", "executes_code"],
      confirmationText: `Open an external installer or user-managed setup flow for ${input.dependencyId}.`,
      payload: {
        dependencyId: input.dependencyId,
      },
    });
  }

  return createActionIntent({
    kind: input.action === "use_existing" ? "dependency.use_existing" : "dependency.unavailable",
    title:
      input.action === "use_existing"
        ? `Use ${input.dependencyId}`
        : `${input.dependencyId} unavailable`,
    description: input.reason,
    source: { kind: "dependency", id: input.dependencyId },
    risks: ["read_only"],
    confirmationRequired: false,
    payload: {
      dependencyId: input.dependencyId,
      action: input.action,
    },
  });
}

function defaultRisksForActionKind(kind: ActionKind): ActionRisk[] {
  switch (kind) {
    case "marketplace.refresh":
    case "artifact.open":
    case "dependency.use_existing":
    case "dependency.unavailable":
      return ["read_only"];
    case "raw_payload.reveal":
      return ["secrets"];
    case "plugin.install":
    case "plugin.update":
    case "harness.install":
    case "dependency.install_managed":
      return ["downloads_code", "writes_files", "executes_code"];
    case "dependency.open_installer":
      return ["downloads_code", "executes_code"];
    case "plugin.uninstall":
      return ["destructive", "writes_files"];
    case "plugin.enable":
    case "plugin.disable":
      return ["writes_settings", "trust_change"];
    case "mcp.start":
    case "hook.enable":
    case "command.rerun":
      return ["executes_code"];
    case "trust.change":
      return ["trust_change", "writes_settings"];
  }
}

function addSecretIssues(value: unknown, ctx: z.RefinementCtx): void {
  for (const issue of findInlineSecrets(value)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: issue.path,
      message: `Action records must not contain inline secret field "${issue.key}".`,
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
