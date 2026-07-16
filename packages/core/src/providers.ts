import { z } from "zod";
import { ModelApiModeSchema, ModelApiSchema } from "./model-api.js";

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

export const ProviderApiSchema = ModelApiSchema;
/** @deprecated Provider kind is an API supply label, never a Model owner. */
export const ProviderKindSchema = ProviderApiSchema;
export const ProviderAuthModeSchema = z.enum(["api_key", "auth_token"]);
export const ProviderApiEntrypointsSchema = z
  .record(ProviderApiSchema, z.string().min(1))
  .default({});

export const ProviderApiCompatibilityModeSchema = z.enum(["auto", "native", "bridge"]);

export const ProviderApiCompatibilitySchema = z.preprocess(
  normalizeProviderApiCompatibilityInput,
  z
    .object({
      mode: ProviderApiCompatibilityModeSchema.default("auto"),
      baseUrl: z.string().min(1).optional(),
      targetApi: ProviderApiSchema.optional(),
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
      kind: ProviderApiSchema,
      apiMode: ModelApiModeSchema.optional(),
      baseUrl: z.string().min(1).optional(),
      apiEntrypoints: ProviderApiEntrypointsSchema,
      authMode: ProviderAuthModeSchema.default("api_key"),
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
      kind: ProviderApiSchema.optional(),
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
      modelId: z.string().min(1),
      runtimeModel: z.string().min(1).optional(),
      contextPacketId: z.string().min(1).optional(),
      parameters: z.record(z.string(), z.unknown()).default({}),
      metadata: z.record(z.string(), z.unknown()).default({}),
    })
    .passthrough()
    .superRefine(addSecretIssues),
);

export const ProviderRuntimeEnvSchema = z.object({
  profileId: z.string().min(1),
  kind: ProviderApiSchema,
  apiMode: ModelApiModeSchema,
  authMode: ProviderAuthModeSchema,
  targetApi: ProviderApiSchema.optional(),
  modelId: z.string().min(1),
  runtimeModel: z.string().min(1),
  baseUrl: z.string().min(1).optional(),
  apiEntrypoints: ProviderApiEntrypointsSchema,
  apiCompatibility: ProviderApiCompatibilitySchema.default({}),
  bridgeEnabled: z.boolean().default(false),
  secretRef: ProviderSecretRefSchema.optional(),
  env: z.record(z.string(), z.string()).default({}),
  requiresSecret: z.boolean(),
  secretInjected: z.boolean(),
});

export type ProviderKind = z.infer<typeof ProviderKindSchema>;
export type ProviderApi = z.infer<typeof ProviderApiSchema>;
export type ProviderAuthMode = z.infer<typeof ProviderAuthModeSchema>;
export type ProviderApiEntrypoints = z.infer<typeof ProviderApiEntrypointsSchema>;
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
  modelId: string;
  runtimeModel?: string;
  secretValue?: string;
  targetApi?: ProviderApi;
  apiCompatibility?: ProviderApiCompatibility;
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

export function buildProviderRuntimeEnv(
  profileInput: unknown,
  options: BuildProviderRuntimeEnvOptions,
): ProviderRuntimeEnv {
  const profile = ProviderProfileMetadataSchema.parse(profileInput);
  const modelId = options.modelId;
  const runtimeModel = options.runtimeModel ?? modelId;
  const requiresSecret = !!profile.secretRef;

  if (requiresSecret && !options.secretValue) {
    throw new Error(`Provider profile "${profile.id}" requires a secret value for runtime use.`);
  }

  const apiCompatibility = ProviderApiCompatibilitySchema.parse({
    ...options.apiCompatibility,
    mode: options.apiCompatibilityMode ?? options.apiCompatibility?.mode,
    baseUrl: options.bridgeBaseUrl ?? options.apiCompatibility?.baseUrl,
  });
  const targetApi = options.targetApi ?? apiCompatibility.targetApi ?? profile.kind;
  const apiMode = profile.apiMode ?? "standard";
  const nativeBaseUrl = profile.apiEntrypoints[targetApi];
  const nativeTargetAvailable = targetApi === profile.kind || nativeBaseUrl !== undefined;
  const runtimeBaseUrl = nativeBaseUrl ?? profile.baseUrl;
  const bridgeEnabled = shouldUseProviderBridge(
    profile.kind,
    targetApi,
    apiCompatibility,
    nativeTargetAvailable,
  );
  if (apiMode === "codex_responses" && targetApi !== "openai_responses") {
    throw new Error("codex_responses mode requires the openai_responses API protocol.");
  }
  if (apiMode === "codex_responses" && bridgeEnabled) {
    throw new Error("codex_responses mode does not support a protocol bridge.");
  }
  const env = bridgeEnabled
    ? providerBridgeEnvVars({
        upstreamKind: profile.kind,
        upstreamAuthMode: profile.authMode,
        targetApi,
        model: runtimeModel,
        upstreamApiKey: options.secretValue,
        upstreamBaseUrl: profile.baseUrl,
        bridgeBaseUrl: apiCompatibility.baseUrl,
        downstreamSecretValue: options.downstreamSecretValue,
      })
    : providerEnvVars({
        kind: targetApi,
        apiMode,
        authMode: profile.authMode,
        model: runtimeModel,
        baseUrl: runtimeBaseUrl,
        apiKey: options.secretValue,
      });

  return ProviderRuntimeEnvSchema.parse({
    profileId: profile.id,
    kind: profile.kind,
    apiMode,
    authMode: profile.authMode,
    targetApi,
    modelId,
    runtimeModel,
    baseUrl: runtimeBaseUrl,
    apiEntrypoints: profile.apiEntrypoints,
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
  upstreamAuthMode: ProviderAuthMode;
  targetApi: ProviderApi;
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
      targetApi: provider.targetApi,
      model,
      rootBaseUrl,
      downstreamSecretValue: provider.downstreamSecretValue,
    }),
  };
}

export function providerEnvVars(provider: {
  kind: ProviderKind;
  apiMode?: "standard" | "codex_responses";
  authMode?: ProviderAuthMode;
  model: string;
  apiKey?: string;
  baseUrl?: string;
}): Record<string, string> {
  const env: Record<string, string> = {};

  switch (provider.kind) {
    case "anthropic":
      env.ANTHROPIC_MODEL = provider.model;
      if (provider.apiKey) {
        if (provider.authMode === "auth_token") {
          env.ANTHROPIC_AUTH_TOKEN = provider.apiKey;
        } else {
          env.ANTHROPIC_API_KEY = provider.apiKey;
        }
      }
      if (provider.baseUrl) env.ANTHROPIC_BASE_URL = provider.baseUrl;
      break;
    case "openai_chat":
    case "openai_responses":
      env.OPENAI_MODEL = provider.model;
      if (provider.apiMode === "codex_responses") {
        env.SWARMX_API_MODE = "codex_responses";
        if (provider.apiKey) env.CODEX_ACCESS_TOKEN = provider.apiKey;
        if (provider.baseUrl) env.CODEX_BASE_URL = provider.baseUrl;
      } else {
        if (provider.apiKey) env.OPENAI_API_KEY = provider.apiKey;
        if (provider.baseUrl) env.OPENAI_BASE_URL = provider.baseUrl;
      }
      break;
    case "ollama":
      env.OLLAMA_MODEL = provider.model;
      if (provider.baseUrl) env.OLLAMA_HOST = provider.baseUrl;
      break;
  }

  return env;
}

function shouldUseProviderBridge(
  providerApi: ProviderApi,
  targetApi: ProviderApi,
  apiCompatibility: ProviderApiCompatibility,
  nativeTargetAvailable = providerApi === targetApi,
): boolean {
  if (apiCompatibility.mode === "native") return false;
  if (apiCompatibility.mode === "bridge") return true;
  return !nativeTargetAvailable;
}

function bridgeUpstreamEnvVars(provider: {
  upstreamKind: ProviderKind;
  upstreamAuthMode: ProviderAuthMode;
  upstreamApiKey?: string;
  upstreamBaseUrl?: string;
}): Record<string, string> {
  const env: Record<string, string> = {
    YALLM_DEFAULT_PROVIDER: providerRoutePrefix(provider.upstreamKind),
  };

  switch (provider.upstreamKind) {
    case "anthropic":
      if (provider.upstreamApiKey) {
        if (provider.upstreamAuthMode === "auth_token") {
          env.ANTHROPIC_AUTH_TOKEN = provider.upstreamApiKey;
        } else {
          env.ANTHROPIC_API_KEY = provider.upstreamApiKey;
        }
      }
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
  targetApi: ProviderApi;
  model: string;
  rootBaseUrl: string;
  downstreamSecretValue?: string;
}): Record<string, string> {
  const downstreamSecretValue = provider.downstreamSecretValue ?? "sk-swarmx-bridge";
  switch (provider.targetApi) {
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
  assertNoProviderOwnedModelFields(input);
  const {
    preset_id,
    display_name,
    base_url,
    api_entrypoints,
    api_mode,
    auth_mode,
    secret_ref,
    label,
    name,
    ...rest
  } = input;

  return {
    ...rest,
    presetId: rest.presetId ?? preset_id,
    displayName: rest.displayName ?? display_name ?? label ?? name,
    baseUrl: rest.baseUrl ?? base_url,
    apiEntrypoints: rest.apiEntrypoints ?? api_entrypoints,
    apiMode: rest.apiMode ?? api_mode,
    authMode: rest.authMode ?? auth_mode,
    secretRef: rest.secretRef ?? secret_ref,
  };
}

function normalizeProviderApiCompatibilityInput(input: unknown): unknown {
  if (typeof input === "string" || typeof input === "boolean") {
    return { mode: normalizeProviderApiCompatibilityMode(input) };
  }
  if (!isObjectRecord(input)) return input;
  const { downstream_kind, target_kind, target_api, base_url, kind, translator, enabled, ...rest } =
    input;
  return {
    ...rest,
    mode:
      rest.mode ??
      normalizeProviderApiCompatibilityMode(
        kind ?? translator ?? (enabled === false ? "native" : undefined),
      ),
    targetApi: rest.targetApi ?? target_api ?? rest.targetKind ?? target_kind ?? downstream_kind,
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
  const { profile_id, ...rest } = input;
  return {
    ...rest,
    profileId: rest.profileId ?? profile_id,
  };
}

function normalizeProviderPromptInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  const {
    request_id,
    profile_id,
    user_text,
    context_packet_id,
    model_id,
    runtime_model,
    model,
    ...rest
  } = input;
  return {
    ...rest,
    requestId: rest.requestId ?? request_id,
    profileId: rest.profileId ?? profile_id,
    userText: rest.userText ?? user_text,
    modelId: rest.modelId ?? model_id ?? model,
    runtimeModel: rest.runtimeModel ?? runtime_model,
    contextPacketId: rest.contextPacketId ?? context_packet_id,
  };
}

function assertNoProviderOwnedModelFields(input: Record<string, unknown>): void {
  const forbidden = new Set([
    "model",
    "models",
    "defaultmodel",
    "harnessmodeloverrides",
    "isdefault",
    "providerproduct",
    "apicompatibility",
    "apibridge",
    "bridge",
    "translator",
    "translation",
  ]);
  for (const key of Object.keys(input)) {
    if (forbidden.has(key.toLowerCase().replace(/[^a-z0-9]/g, ""))) {
      throw new Error(
        `Provider profile field "${key}" is invalid; Models and route compatibility belong to Model/ModelSupply.`,
      );
    }
  }
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
