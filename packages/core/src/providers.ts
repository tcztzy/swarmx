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
  /(api[_-]?key|access[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key|smtp[_-]?password|telemetry[_-]?token|ingest[_-]?token|cluster[_-]?password|remote[_-]?compute[_-]?password)/i;

export const ProviderKindSchema = z.enum([
  "anthropic",
  "openai_chat",
  "openai_responses",
  "ollama",
]);

export const ProviderApiCompatibilityModeSchema = z.enum(["auto", "native", "bridge"]);

export const ProviderApiCompatibilitySchema = z.preprocess(
  normalizeProviderApiCompatibilityInput,
  z
    .object({
      mode: ProviderApiCompatibilityModeSchema.default("auto"),
      baseUrl: z.string().min(1).optional(),
      targetKind: ProviderKindSchema.optional(),
    })
    .passthrough()
    .superRefine(addSecretIssues),
);

export const ProviderSecretSourceSchema = z.enum([
  "env",
  "local_keychain",
  "server_keychain",
  "prompt",
]);

export const ProviderSelectionSourceSchema = z.enum([
  "direct_prompt",
  "harness_selector",
  "agent_profile",
]);

export const ProviderSecretRefSchema = z
  .object({
    source: ProviderSecretSourceSchema,
    key: z.string().min(1),
    service: z.string().min(1).optional(),
    account: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const ProviderProfileMetadataSchema = z.preprocess(
  normalizeProviderProfileInput,
  z
    .object({
      id: z.string().min(1),
      presetId: z.string().min(1).optional(),
      displayName: z.string().min(1),
      description: z.string().optional(),
      kind: ProviderKindSchema,
      defaultModel: z.string().min(1).optional(),
      baseUrl: z.string().min(1).optional(),
      isDefault: z.boolean().default(false),
      harnessModelOverrides: z.record(z.string().min(1), z.string().min(1)).default({}),
      apiCompatibility: ProviderApiCompatibilitySchema.default({}),
      secretRef: ProviderSecretRefSchema.optional(),
      readOnly: z.boolean().optional(),
      metadata: z.record(z.string(), z.unknown()).default({}),
    })
    .passthrough()
    .superRefine(addSecretIssues),
);

export const ProviderSecretStatusSchema = z.preprocess(
  normalizeProviderSecretStatusInput,
  z
    .object({
      profileId: z.string().min(1),
      source: ProviderSecretSourceSchema,
      configured: z.boolean(),
      key: z.string().min(1).optional(),
      checkedAt: z.string().min(1).optional(),
      message: z.string().optional(),
    })
    .passthrough()
    .superRefine(addSecretStatusIssues),
);

export const ProviderSelectionSchema = z.preprocess(
  normalizeProviderSelectionInput,
  z
    .object({
      profileId: z.string().min(1).optional(),
      kind: ProviderKindSchema.optional(),
      model: z.string().min(1).optional(),
      harnessId: z.string().min(1).optional(),
      source: ProviderSelectionSourceSchema.default("direct_prompt"),
    })
    .passthrough()
    .superRefine(addSecretIssues),
);

export const ProviderPromptRequestSchema = z.preprocess(
  normalizeProviderPromptInput,
  z
    .object({
      requestId: z.string().min(1).optional(),
      profileId: z.string().min(1),
      userText: z.string().min(1),
      model: z.string().min(1).optional(),
      contextPacketId: z.string().min(1).optional(),
      parameters: z.record(z.string(), z.unknown()).default({}),
      metadata: z.record(z.string(), z.unknown()).default({}),
    })
    .passthrough()
    .superRefine(addSecretIssues),
);

export const ProviderRuntimeEnvSchema = z.object({
  profileId: z.string().min(1),
  kind: ProviderKindSchema,
  targetKind: ProviderKindSchema.optional(),
  model: z.string().min(1),
  baseUrl: z.string().min(1).optional(),
  apiCompatibility: ProviderApiCompatibilitySchema.default({}),
  bridgeEnabled: z.boolean().default(false),
  secretRef: ProviderSecretRefSchema.optional(),
  env: z.record(z.string(), z.string()).default({}),
  requiresSecret: z.boolean(),
  secretInjected: z.boolean(),
});

export type ProviderKind = z.infer<typeof ProviderKindSchema>;
export type ProviderApiCompatibilityMode = z.infer<typeof ProviderApiCompatibilityModeSchema>;
export type ProviderApiCompatibility = z.infer<typeof ProviderApiCompatibilitySchema>;
export type ProviderSecretSource = z.infer<typeof ProviderSecretSourceSchema>;
export type ProviderSelectionSource = z.infer<typeof ProviderSelectionSourceSchema>;
export type ProviderSecretRef = z.infer<typeof ProviderSecretRefSchema>;
export type ProviderProfileMetadata = z.infer<typeof ProviderProfileMetadataSchema>;
export type ProviderSecretStatus = z.infer<typeof ProviderSecretStatusSchema>;
export type ProviderSelection = z.infer<typeof ProviderSelectionSchema>;
export type ProviderPromptRequest = z.infer<typeof ProviderPromptRequestSchema>;
export type ProviderRuntimeEnv = z.infer<typeof ProviderRuntimeEnvSchema>;

export interface BuildProviderRuntimeEnvOptions {
  secretValue?: string;
  model?: string;
  harnessId?: string;
  targetKind?: ProviderKind;
  compatibleProviderKinds?: ProviderKind[];
  apiCompatibilityMode?: ProviderApiCompatibilityMode;
  bridgeBaseUrl?: string;
  downstreamSecretValue?: string;
}

export function parseProviderProfileMetadata(input: unknown): ProviderProfileMetadata {
  return ProviderProfileMetadataSchema.parse(input);
}

export function parseProviderSecretStatus(input: unknown): ProviderSecretStatus {
  return ProviderSecretStatusSchema.parse(input);
}

export function parseProviderPromptRequest(input: unknown): ProviderPromptRequest {
  return ProviderPromptRequestSchema.parse(input);
}

export function resolveProviderProfile(
  profilesInput: unknown[],
  selectionInput: unknown = {},
): ProviderProfileMetadata {
  const profiles = profilesInput.map((profile) => ProviderProfileMetadataSchema.parse(profile));
  const selection = ProviderSelectionSchema.parse(selectionInput);

  if (selection.profileId) {
    const matches = profiles.filter((profile) => profile.id === selection.profileId);
    if (matches.length === 1) return matches[0] as ProviderProfileMetadata;
    if (matches.length > 1) {
      throw new Error(`Ambiguous provider profile id "${selection.profileId}".`);
    }
    throw new Error(`Unknown provider profile id "${selection.profileId}".`);
  }

  const candidates = selection.kind
    ? profiles.filter((profile) => profile.kind === selection.kind)
    : profiles;
  const defaults = candidates.filter((profile) => profile.isDefault);
  if (defaults.length === 1) return defaults[0] as ProviderProfileMetadata;
  if (defaults.length > 1) {
    throw new Error("Ambiguous default provider profile selection.");
  }
  if (candidates.length === 1) return candidates[0] as ProviderProfileMetadata;
  if (candidates.length === 0) {
    throw new Error(
      selection.kind
        ? `No provider profile for kind "${selection.kind}".`
        : "No provider profiles.",
    );
  }
  throw new Error("Provider profile selection must be explicit.");
}

export function providerModelForSelection(
  profileInput: unknown,
  selectionInput: unknown = {},
): string {
  const profile = ProviderProfileMetadataSchema.parse(profileInput);
  const selection = ProviderSelectionSchema.parse(selectionInput);
  const harnessOverride = selection.harnessId
    ? profile.harnessModelOverrides[selection.harnessId]
    : undefined;
  const model = selection.model ?? harnessOverride ?? profile.defaultModel;
  if (!model) {
    throw new Error(`Provider profile "${profile.id}" must resolve a model.`);
  }
  return model;
}

export function buildProviderRuntimeEnv(
  profileInput: unknown,
  options: BuildProviderRuntimeEnvOptions = {},
): ProviderRuntimeEnv {
  const profile = ProviderProfileMetadataSchema.parse(profileInput);
  const model = providerModelForSelection(profile, {
    model: options.model,
    harnessId: options.harnessId,
  });
  const requiresSecret = !!profile.secretRef;

  if (requiresSecret && !options.secretValue) {
    throw new Error(`Provider profile "${profile.id}" requires a secret value for runtime use.`);
  }

  const apiCompatibility = ProviderApiCompatibilitySchema.parse({
    ...profile.apiCompatibility,
    mode: options.apiCompatibilityMode ?? profile.apiCompatibility.mode,
    baseUrl: options.bridgeBaseUrl ?? profile.apiCompatibility.baseUrl,
  });
  const targetKind = resolveRuntimeTargetKind(profile.kind, apiCompatibility, options);
  const bridgeEnabled = shouldUseProviderBridge(profile.kind, targetKind, apiCompatibility);
  const env = bridgeEnabled
    ? providerBridgeEnvVars({
        upstreamKind: profile.kind,
        targetKind,
        model,
        upstreamApiKey: options.secretValue,
        upstreamBaseUrl: profile.baseUrl,
        bridgeBaseUrl: apiCompatibility.baseUrl,
        downstreamSecretValue: options.downstreamSecretValue,
      })
    : providerEnvVars({
        kind: profile.kind,
        model,
        baseUrl: profile.baseUrl,
        apiKey: options.secretValue,
      });

  return ProviderRuntimeEnvSchema.parse({
    profileId: profile.id,
    kind: profile.kind,
    targetKind,
    model,
    baseUrl: profile.baseUrl,
    apiCompatibility,
    bridgeEnabled,
    secretRef: profile.secretRef,
    env,
    requiresSecret,
    secretInjected: !!options.secretValue,
  });
}

function providerBridgeEnvVars(provider: {
  upstreamKind: ProviderKind;
  targetKind: ProviderKind;
  model: string;
  upstreamApiKey?: string;
  upstreamBaseUrl?: string;
  bridgeBaseUrl?: string;
  downstreamSecretValue?: string;
}): Record<string, string> {
  const rootBaseUrl = normalizeBridgeRootBaseUrl(provider.bridgeBaseUrl);
  const model = `${providerRoutePrefix(provider.upstreamKind)}:${provider.model}`;
  return {
    ...bridgeUpstreamEnvVars(provider),
    ...bridgeDownstreamEnvVars({
      targetKind: provider.targetKind,
      model,
      rootBaseUrl,
      downstreamSecretValue: provider.downstreamSecretValue,
    }),
  };
}

export function providerEnvVars(provider: {
  kind: ProviderKind;
  model: string;
  apiKey?: string;
  baseUrl?: string;
}): Record<string, string> {
  const env: Record<string, string> = {};
  if (provider.model) env.OPENAI_MODEL = provider.model;

  switch (provider.kind) {
    case "anthropic":
      if (provider.apiKey) env.ANTHROPIC_API_KEY = provider.apiKey;
      if (provider.baseUrl) env.ANTHROPIC_BASE_URL = provider.baseUrl;
      break;
    case "openai_chat":
    case "openai_responses":
      if (provider.apiKey) env.OPENAI_API_KEY = provider.apiKey;
      if (provider.baseUrl) env.OPENAI_BASE_URL = provider.baseUrl;
      break;
    case "ollama":
      if (provider.baseUrl) env.OLLAMA_HOST = provider.baseUrl;
      break;
  }

  return env;
}

function resolveRuntimeTargetKind(
  providerKind: ProviderKind,
  apiCompatibility: ProviderApiCompatibility,
  options: BuildProviderRuntimeEnvOptions,
): ProviderKind {
  if (options.targetKind) return options.targetKind;
  if (apiCompatibility.targetKind) return apiCompatibility.targetKind;
  const compatibleProviderKinds = options.compatibleProviderKinds ?? [];
  if (compatibleProviderKinds.includes(providerKind)) return providerKind;
  if (apiCompatibility.mode !== "native" && compatibleProviderKinds.length > 0) {
    return compatibleProviderKinds[0] as ProviderKind;
  }
  return providerKind;
}

function shouldUseProviderBridge(
  providerKind: ProviderKind,
  targetKind: ProviderKind,
  apiCompatibility: ProviderApiCompatibility,
): boolean {
  if (apiCompatibility.mode === "native") return false;
  if (apiCompatibility.mode === "bridge") return true;
  return providerKind !== targetKind;
}

function bridgeUpstreamEnvVars(provider: {
  upstreamKind: ProviderKind;
  upstreamApiKey?: string;
  upstreamBaseUrl?: string;
}): Record<string, string> {
  const env: Record<string, string> = {
    YALLM_DEFAULT_PROVIDER: providerRoutePrefix(provider.upstreamKind),
  };

  switch (provider.upstreamKind) {
    case "anthropic":
      if (provider.upstreamApiKey) env.ANTHROPIC_API_KEY = provider.upstreamApiKey;
      if (provider.upstreamBaseUrl) env.ANTHROPIC_BASE_URL = provider.upstreamBaseUrl;
      break;
    case "openai_chat":
    case "openai_responses":
      if (provider.upstreamApiKey) env.OPENAI_API_KEY = provider.upstreamApiKey;
      if (provider.upstreamBaseUrl) env.OPENAI_BASE_URL = provider.upstreamBaseUrl;
      break;
    case "ollama":
      if (provider.upstreamBaseUrl) env.OLLAMA_BASE_URL = provider.upstreamBaseUrl;
      break;
  }

  return env;
}

function bridgeDownstreamEnvVars(provider: {
  targetKind: ProviderKind;
  model: string;
  rootBaseUrl: string;
  downstreamSecretValue?: string;
}): Record<string, string> {
  const downstreamSecretValue = provider.downstreamSecretValue ?? "sk-swarmx-bridge";
  switch (provider.targetKind) {
    case "anthropic":
      return {
        ANTHROPIC_AUTH_TOKEN: downstreamSecretValue,
        ANTHROPIC_BASE_URL: provider.rootBaseUrl,
        ANTHROPIC_MODEL: provider.model,
      };
    case "openai_chat":
    case "openai_responses":
      return {
        OPENAI_API_KEY: downstreamSecretValue,
        OPENAI_BASE_URL: `${provider.rootBaseUrl}/v1`,
        OPENAI_MODEL: provider.model,
      };
    case "ollama":
      return {
        OLLAMA_BASE_URL: provider.rootBaseUrl,
        OLLAMA_HOST: provider.rootBaseUrl,
        OLLAMA_MODEL: provider.model,
      };
  }
}

function providerRoutePrefix(kind: ProviderKind): string {
  switch (kind) {
    case "anthropic":
      return "anthropic";
    case "openai_chat":
    case "openai_responses":
      return "openai";
    case "ollama":
      return "ollama";
  }
}

function normalizeBridgeRootBaseUrl(baseUrl: string | undefined): string {
  const trimmed = (baseUrl ?? "http://127.0.0.1:4000").trim().replace(/\/+$/, "");
  return trimmed.endsWith("/v1") ? trimmed.slice(0, -3).replace(/\/+$/, "") : trimmed;
}

function normalizeProviderProfileInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  const {
    preset_id,
    display_name,
    default_model,
    base_url,
    is_default,
    harness_model_overrides,
    api_compatibility,
    apiBridge,
    api_bridge,
    bridge,
    translator,
    translation,
    secret_ref,
    label,
    name,
    model,
    ...rest
  } = input;

  return {
    ...rest,
    presetId: rest.presetId ?? preset_id,
    displayName: rest.displayName ?? display_name ?? label ?? name,
    defaultModel: rest.defaultModel ?? default_model ?? model,
    baseUrl: rest.baseUrl ?? base_url,
    isDefault: rest.isDefault ?? is_default,
    harnessModelOverrides: rest.harnessModelOverrides ?? harness_model_overrides,
    apiCompatibility:
      rest.apiCompatibility ??
      api_compatibility ??
      apiBridge ??
      api_bridge ??
      bridge ??
      translation ??
      translator,
    secretRef: rest.secretRef ?? secret_ref,
  };
}

function normalizeProviderApiCompatibilityInput(input: unknown): unknown {
  if (typeof input === "string" || typeof input === "boolean") {
    return { mode: normalizeProviderApiCompatibilityMode(input) };
  }
  if (!isObjectRecord(input)) return input;
  const { downstream_kind, target_kind, base_url, kind, translator, enabled, ...rest } = input;
  return {
    ...rest,
    mode:
      rest.mode ??
      normalizeProviderApiCompatibilityMode(
        kind ?? translator ?? (enabled === false ? "native" : undefined),
      ),
    targetKind: rest.targetKind ?? target_kind ?? downstream_kind,
    baseUrl: rest.baseUrl ?? base_url,
  };
}

function normalizeProviderApiCompatibilityMode(input: unknown): unknown {
  if (input === true) return "bridge";
  if (input === false) return "native";
  if (typeof input !== "string") return input;
  switch (input) {
    case "yallm":
    case "translated":
    case "translation":
      return "bridge";
    case "off":
    case "disabled":
      return "native";
    default:
      return input;
  }
}

function normalizeProviderSecretStatusInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  const { profile_id, checked_at, ...rest } = input;
  return {
    ...rest,
    profileId: rest.profileId ?? profile_id,
    checkedAt: rest.checkedAt ?? checked_at,
  };
}

function normalizeProviderSelectionInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  const { profile_id, harness_id, ...rest } = input;
  return {
    ...rest,
    profileId: rest.profileId ?? profile_id,
    harnessId: rest.harnessId ?? harness_id,
  };
}

function normalizeProviderPromptInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  const { request_id, profile_id, user_text, context_packet_id, ...rest } = input;
  return {
    ...rest,
    requestId: rest.requestId ?? request_id,
    profileId: rest.profileId ?? profile_id,
    userText: rest.userText ?? user_text,
    contextPacketId: rest.contextPacketId ?? context_packet_id,
  };
}

function addSecretStatusIssues(status: unknown, ctx: z.RefinementCtx): void {
  addSecretIssues(status, ctx);
  if (!isObjectRecord(status)) return;
  if ("value" in status || "secretValue" in status || "apiKeyValue" in status) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "Provider secret status records must not include secret values.",
    });
  }
}

function addSecretIssues(value: unknown, ctx: z.RefinementCtx): void {
  for (const issue of findInlineSecrets(value)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: issue.path,
      message: `Provider metadata must not contain inline secret field "${issue.key}".`,
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
