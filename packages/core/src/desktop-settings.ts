import { z } from "zod";
import { AgentProfileMetadataSchema } from "./agent-profiles.js";
import {
  ExtensionCandidateSchema,
  ExtensionMarketplaceSourceSchema,
  InstalledExtensionSchema,
} from "./extension-management.js";
import { ModelSchema, ModelSupplySchema } from "./model-capabilities.js";
import { ProviderProfileMetadataSchema } from "./providers.js";
import {
  type HarnessPermissionPolicyLayer,
  HarnessPermissionPolicyLayerSchema,
  type PermissionApprovalReceipt,
  PermissionApprovalReceiptSchema,
} from "./skill-variants.js";

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
  /(api[_-]?key|api[_-]?token|access[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key|smtp[_-]?password|telemetry[_-]?token|cluster[_-]?password|remote[_-]?compute[_-]?password|host[_-]?login|auth[_-]?token)/i;

export const DesktopRootConfigSchema = z.preprocess(
  normalizeDesktopRootConfigInput,
  z
    .object({
      root: z.string().min(1).optional(),
      legacyAppRoot: z.string().min(1).optional(),
    })
    .passthrough()
    .superRefine(addSecretIssues),
);

export const DesktopServerSettingsSchema = z.preprocess(
  normalizeDesktopServerSettingsInput,
  z
    .object({
      baseUrl: z.string().min(1).optional(),
      dataRoot: z.string().min(1).optional(),
      appRoot: z.string().min(1).optional(),
    })
    .passthrough()
    .superRefine(addSecretIssues),
);

export const DesktopUiStateSchema = z
  .object({
    locale: z.string().min(1).optional(),
    theme: z.enum(["system", "light", "dark"]).default("system"),
    lastView: z.string().min(1).optional(),
    sidebarCollapsed: z.boolean().optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const DesktopExtensionSettingsSchema = z
  .object({
    enabledPluginIds: z.array(z.string().min(1)).default([]),
    disabledPluginIds: z.array(z.string().min(1)).default([]),
    trustedSourceIds: z.array(z.string().min(1)).default([]),
    marketplaceSources: z.array(ExtensionMarketplaceSourceSchema).default([]),
    marketplaceCandidates: z.array(ExtensionCandidateSchema).default([]),
    installed: z.array(InstalledExtensionSchema).default([]),
    skillEvolutionEnabled: z.boolean().default(false),
    skillPromotionGate: z.enum(["human", "policy"]).default("human"),
  })
  .passthrough()
  .superRefine(addSecretIssues);

const PersonalPermissionPolicySchema = HarnessPermissionPolicyLayerSchema.superRefine(
  (policy, ctx) => {
    if (policy.source !== "personal") {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["source"],
        message: "Desktop personal permission policy must use the personal source.",
      });
    }
    if (policy.readOnly) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["readOnly"],
        message: "Desktop personal permission policy must remain editable.",
      });
    }
  },
);

export const DesktopPermissionProfileAvailabilitySchema = z
  .object({
    default: z.boolean().default(true),
    auto: z.boolean().default(true),
    trusted: z.boolean().default(true),
  })
  .strict();

export const DesktopPermissionSettingsSchema = z
  .object({
    personalPolicy: PersonalPermissionPolicySchema.default({
      id: "personal",
      source: "personal",
      label: "Personal defaults",
      allowedTools: [],
      deniedTools: [],
      readOnly: false,
    }),
    profileAvailability: DesktopPermissionProfileAvailabilitySchema.default({}),
    approvalReceipts: z.array(PermissionApprovalReceiptSchema).max(200).default([]),
  })
  .strict()
  .superRefine(addSecretIssues);

export const DesktopSettingsDocumentSchema: z.ZodType<
  DesktopSettingsDocument,
  z.ZodTypeDef,
  unknown
> = z.preprocess(
  normalizeDesktopSettingsDocumentInput,
  z
    .object({
      schemaVersion: z.literal(1).default(1),
      desktop: DesktopRootConfigSchema.default({}),
      server: DesktopServerSettingsSchema.default({}),
      ui: DesktopUiStateSchema.default({}),
      models: z.array(ModelSchema).default([]),
      modelSupplies: z.array(ModelSupplySchema).default([]),
      providers: z.array(ProviderProfileMetadataSchema).default([]),
      agents: z.array(AgentProfileMetadataSchema).default([]),
      extensions: DesktopExtensionSettingsSchema.default({}),
      permissions: DesktopPermissionSettingsSchema.default({}),
    })
    .passthrough()
    .superRefine(addSecretIssues),
);

export const DesktopRootSourceSchema = z.enum([
  "env_desktop_root",
  "settings_desktop_root",
  "env_legacy_app_root",
  "settings_legacy_app_root",
  "default",
  "unresolved",
]);

export const ServerDataRootSourceSchema = z.enum([
  "env_server_data_root",
  "settings_server_data_root",
]);

export const ResolvedDesktopRootSchema = z.object({
  desktopRoot: z.string().min(1).optional(),
  source: DesktopRootSourceSchema,
  legacyFallback: z.boolean().default(false),
  serverDataRoot: z.string().min(1).optional(),
  serverDataRootSource: ServerDataRootSourceSchema.optional(),
});

export const LocaleDirectionSchema = z.enum(["ltr", "rtl"]);

export const LocaleResourceSchema = z
  .object({
    id: z.string().min(1),
    label: z.string().min(1),
    nativeLabel: z.string().min(1).optional(),
    direction: LocaleDirectionSchema.default("ltr"),
    messages: z.record(z.string(), z.string()).default({}),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const LocaleRegistrySchema = z
  .object({
    defaultLocaleId: z.string().min(1),
    locales: z.array(LocaleResourceSchema).min(1),
  })
  .superRefine((registry, ctx) => {
    const seen = new Set<string>();
    for (const [index, locale] of registry.locales.entries()) {
      if (seen.has(locale.id)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["locales", index, "id"],
          message: `Duplicate locale id "${locale.id}".`,
        });
      }
      seen.add(locale.id);
    }
    if (!seen.has(registry.defaultLocaleId)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["defaultLocaleId"],
        message: `Default locale "${registry.defaultLocaleId}" is not registered.`,
      });
    }
  });

export const LocaleSelectionSourceSchema = z.enum(["requested", "settings", "env", "default"]);

export const ResolvedLocaleSelectionSchema = z.object({
  locale: LocaleResourceSchema,
  source: LocaleSelectionSourceSchema,
  requestedLocaleId: z.string().min(1).optional(),
  fallback: z.boolean(),
});

export type DesktopRootConfig = z.infer<typeof DesktopRootConfigSchema>;
export type DesktopServerSettings = z.infer<typeof DesktopServerSettingsSchema>;
export type DesktopUiState = z.infer<typeof DesktopUiStateSchema>;
export type DesktopExtensionSettings = z.infer<typeof DesktopExtensionSettingsSchema>;
export type DesktopPermissionProfileAvailability = z.infer<
  typeof DesktopPermissionProfileAvailabilitySchema
>;
export interface DesktopPermissionSettings {
  personalPolicy: HarnessPermissionPolicyLayer;
  profileAvailability: DesktopPermissionProfileAvailability;
  approvalReceipts: PermissionApprovalReceipt[];
}
export interface DesktopSettingsDocument {
  schemaVersion: 1;
  desktop: DesktopRootConfig;
  server: DesktopServerSettings;
  ui: DesktopUiState;
  models: Array<z.infer<typeof ModelSchema>>;
  modelSupplies: Array<z.infer<typeof ModelSupplySchema>>;
  providers: Array<z.infer<typeof ProviderProfileMetadataSchema>>;
  agents: Array<z.infer<typeof AgentProfileMetadataSchema>>;
  extensions: DesktopExtensionSettings;
  permissions: DesktopPermissionSettings;
  [key: string]: unknown;
}
export type DesktopRootSource = z.infer<typeof DesktopRootSourceSchema>;
export type ServerDataRootSource = z.infer<typeof ServerDataRootSourceSchema>;
export type ResolvedDesktopRoot = z.infer<typeof ResolvedDesktopRootSchema>;
export type LocaleDirection = z.infer<typeof LocaleDirectionSchema>;
export type LocaleResource = z.infer<typeof LocaleResourceSchema>;
export type LocaleRegistry = z.infer<typeof LocaleRegistrySchema>;
export type LocaleSelectionSource = z.infer<typeof LocaleSelectionSourceSchema>;
export type ResolvedLocaleSelection = z.infer<typeof ResolvedLocaleSelectionSchema>;

export interface ResolveDesktopRootOptions {
  settings?: unknown;
  env?: Record<string, string | undefined>;
  desktopRootEnvKeys?: string[];
  legacyAppRootEnvKeys?: string[];
  serverDataRootEnvKeys?: string[];
  defaultDesktopRoot?: string;
}

export interface ResolveLocaleSelectionOptions {
  registry: unknown;
  settings?: unknown;
  env?: Record<string, string | undefined>;
  envKeys?: string[];
  requestedLocaleId?: string;
}

export function parseDesktopSettingsDocument(input: unknown): DesktopSettingsDocument {
  return DesktopSettingsDocumentSchema.parse(input);
}

export function parseDesktopUiState(input: unknown): DesktopUiState {
  return DesktopUiStateSchema.parse(input);
}

export function createDefaultDesktopSettings(
  input: Partial<DesktopSettingsDocument> = {},
): DesktopSettingsDocument {
  return DesktopSettingsDocumentSchema.parse({
    schemaVersion: 1,
    desktop: {},
    server: {},
    ui: {},
    models: [],
    modelSupplies: [],
    providers: [],
    agents: [],
    extensions: {},
    permissions: {},
    ...input,
  });
}

export function createLocaleRegistry(input: unknown): LocaleRegistry {
  return LocaleRegistrySchema.parse(input);
}

export function resolveDesktopRoot(options: ResolveDesktopRootOptions = {}): ResolvedDesktopRoot {
  const settings = DesktopSettingsDocumentSchema.parse(options.settings ?? {});
  const env = options.env ?? {};
  const desktopRootEnvKeys = options.desktopRootEnvKeys ?? [
    "SWARMX_DESKTOP_ROOT",
    "GEEPILOT_DESKTOP_ROOT",
  ];
  const legacyAppRootEnvKeys = options.legacyAppRootEnvKeys ?? [
    "SWARMX_APP_ROOT",
    "GEEPILOT_APP_ROOT",
  ];
  const serverDataRootEnvKeys = options.serverDataRootEnvKeys ?? [
    "SWARMX_SERVER_DATA_ROOT",
    "GEEPILOT_SERVER_DATA_ROOT",
  ];

  const envDesktopRoot = firstEnvValue(env, desktopRootEnvKeys);
  const envLegacyAppRoot = firstEnvValue(env, legacyAppRootEnvKeys);
  const envServerDataRoot = firstEnvValue(env, serverDataRootEnvKeys);
  const serverDataRoot = envServerDataRoot.value ?? settings.server.dataRoot;
  const serverDataRootSource = envServerDataRoot.value
    ? "env_server_data_root"
    : settings.server.dataRoot
      ? "settings_server_data_root"
      : undefined;

  if (envDesktopRoot.value) {
    return ResolvedDesktopRootSchema.parse({
      desktopRoot: envDesktopRoot.value,
      source: "env_desktop_root",
      legacyFallback: false,
      serverDataRoot,
      serverDataRootSource,
    });
  }

  if (settings.desktop.root) {
    return ResolvedDesktopRootSchema.parse({
      desktopRoot: settings.desktop.root,
      source: "settings_desktop_root",
      legacyFallback: false,
      serverDataRoot,
      serverDataRootSource,
    });
  }

  if (envLegacyAppRoot.value) {
    return ResolvedDesktopRootSchema.parse({
      desktopRoot: envLegacyAppRoot.value,
      source: "env_legacy_app_root",
      legacyFallback: true,
      serverDataRoot,
      serverDataRootSource,
    });
  }

  const settingsLegacyAppRoot = settings.desktop.legacyAppRoot ?? settings.server.appRoot;
  if (settingsLegacyAppRoot) {
    return ResolvedDesktopRootSchema.parse({
      desktopRoot: settingsLegacyAppRoot,
      source: "settings_legacy_app_root",
      legacyFallback: true,
      serverDataRoot,
      serverDataRootSource,
    });
  }

  if (options.defaultDesktopRoot) {
    return ResolvedDesktopRootSchema.parse({
      desktopRoot: options.defaultDesktopRoot,
      source: "default",
      legacyFallback: false,
      serverDataRoot,
      serverDataRootSource,
    });
  }

  return ResolvedDesktopRootSchema.parse({
    source: "unresolved",
    legacyFallback: false,
    serverDataRoot,
    serverDataRootSource,
  });
}

export function resolveLocaleSelection(
  options: ResolveLocaleSelectionOptions,
): ResolvedLocaleSelection {
  const registry = LocaleRegistrySchema.parse(options.registry);
  const settings = DesktopSettingsDocumentSchema.parse(options.settings ?? {});
  const env = options.env ?? {};
  const envKeys = options.envKeys ?? ["SWARMX_LOCALE", "GEEPILOT_LOCALE", "LANG"];
  const candidates: Array<{ source: LocaleSelectionSource; localeId: string | undefined }> = [
    { source: "requested", localeId: options.requestedLocaleId },
    { source: "settings", localeId: settings.ui.locale },
    { source: "env", localeId: normalizeLocaleEnvValue(firstEnvValue(env, envKeys).value) },
    { source: "default", localeId: registry.defaultLocaleId },
  ];

  for (const candidate of candidates) {
    if (!candidate.localeId) continue;
    const locale = registry.locales.find((item) => item.id === candidate.localeId);
    if (locale) {
      return ResolvedLocaleSelectionSchema.parse({
        locale,
        source: candidate.source,
        requestedLocaleId: candidate.localeId,
        fallback: candidate.source === "default",
      });
    }
  }

  throw new Error(`Default locale "${registry.defaultLocaleId}" is not registered.`);
}

function normalizeDesktopSettingsDocumentInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  return {
    ...input,
    schemaVersion: input.schemaVersion ?? input.schema_version,
  };
}

function normalizeDesktopRootConfigInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  return {
    ...input,
    legacyAppRoot: input.legacyAppRoot ?? input.legacy_app_root,
  };
}

function normalizeDesktopServerSettingsInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  return {
    ...input,
    baseUrl: input.baseUrl ?? input.base_url,
    dataRoot: input.dataRoot ?? input.data_root,
    appRoot: input.appRoot ?? input.app_root,
  };
}

function firstEnvValue(
  env: Record<string, string | undefined>,
  keys: string[],
): { key?: string; value?: string } {
  for (const key of keys) {
    const value = env[key]?.trim();
    if (value) return { key, value };
  }
  return {};
}

function normalizeLocaleEnvValue(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const [locale] = value.split(".");
  return locale?.replace("_", "-");
}

function addSecretIssues(value: unknown, ctx: z.RefinementCtx): void {
  for (const issue of findInlineSecrets(value)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: issue.path,
      message: `Desktop settings records must not contain inline secret field "${issue.key}".`,
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
