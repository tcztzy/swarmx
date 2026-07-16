import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import { AgentDefinitionSourceSchema } from "./agent-profiles.js";
import { ContextPacketModeSchema, ContextStrategySchema } from "./context.js";
import { HARNESSES, harnessModelRuntimeEnv, harnessModelRuntimeModel } from "./harness.js";
import type { LocalMcpTool } from "./mcp.js";
import { ModelApiModeSchema, ModelApiSchema } from "./model-api.js";
import {
  ModelSchema as IndependentModelSchema,
  MODELS,
  ModelSupplySchema,
  modelCapabilityRegistry,
  normalizeModelReasoningEffort,
  resolveHarnessModelInventory,
  resolveModelReasoningCapability,
} from "./model-capabilities.js";
import type { Model, ModelSupply } from "./model-capabilities.js";
import {
  ProviderApiEntrypointsSchema,
  ProviderAuthModeSchema,
  buildProviderRuntimeEnv,
} from "./providers.js";
import { Swarm } from "./swarm.js";
import { AgentBackendSchema, McpServerConfigSchema } from "./types.js";
import type {
  AgentBackend,
  AgentConfig,
  ChatMessage,
  McpServerConfig,
  MessageChunk,
  SwarmConfig,
} from "./types.js";
import { SWARMX_VERSION } from "./version.js";

const MANIFEST_FILENAMES = new Set([
  "extension.json",
  "plugin.json",
  "swarmx.extension.json",
  "swarmx-extension.json",
]);

export const SWARMX_LOCAL_FILES_LSP_ID = "swarmx.local-files";
export const SWARMX_SKILLS_LSP_ID = "swarmx.skills";

const ALLOWED_SECRET_REFERENCE_KEYS = new Set([
  "secretref",
  "secretrefs",
  "secret_ref",
  "secret_refs",
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
  /(api[_-]?key|access[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key)/i;

const FORBIDDEN_INLINE_UI_PAYLOAD_KEYS = new Set([
  "code",
  "componentbody",
  "html",
  "iframe",
  "inlinecomponent",
  "inlinehtml",
  "inlinescript",
  "renderfunction",
  "script",
  "sourcecode",
  "template",
  "webview",
]);

export const ProviderKindSchema = ModelApiSchema;

export const ProviderApiCompatibilityModeSchema = z.enum(["auto", "native", "bridge"]);

export const ProviderApiCompatibilitySchema = z.preprocess(
  normalizeProviderApiCompatibilityInput,
  z
    .object({
      mode: ProviderApiCompatibilityModeSchema.default("auto"),
      baseUrl: z.string().min(1).optional(),
      targetApi: ModelApiSchema.optional(),
    })
    .passthrough()
    .superRefine(addInlineSecretIssues),
);

export const MarketplaceHostSchema = z.enum([
  "codex",
  "claude",
  "opencode",
  "swarmx",
  "local",
  "custom",
]);

export const ExtensionSourceSchema = z
  .object({
    type: z.enum(["builtin", "path", "marketplace", "plugin"]),
    path: z.string().optional(),
    marketplace: z.string().optional(),
    package: z.string().optional(),
  })
  .passthrough();

export const SoftwareCapabilitySchema = z
  .object({
    id: z.string().min(1).optional(),
    name: z.string().min(1),
    version: z.string().min(1).optional(),
    runner: z.string().min(1).optional(),
    command: z.array(z.string()).optional(),
    platform: z.string().optional(),
  })
  .passthrough();

export const SkillHostExposureStatusSchema = z.enum([
  "plugin",
  "rules_only",
  "unsupported",
  "unknown",
]);

export const SkillHostExposureSchema = z
  .object({
    host: MarketplaceHostSchema,
    status: SkillHostExposureStatusSchema.default("plugin"),
    manifestPath: z.string().min(1).optional(),
    marketplaceSourceId: z.string().min(1).optional(),
    rulesPath: z.string().min(1).optional(),
    package: z.string().min(1).optional(),
    readOnly: z.boolean().optional(),
    notes: z.string().optional(),
  })
  .passthrough()
  .superRefine(addInlineSecretIssues);

export const SkillCapabilitySchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1).optional(),
    path: z.string().min(1).optional(),
    canonicalPath: z.string().min(1).optional(),
    description: z.string().optional(),
    provenance: z.string().optional(),
    governanceRef: z.string().min(1).optional(),
    requiresGateSkillIds: z.array(z.string().min(1)).default([]),
    hostExposures: z.array(SkillHostExposureSchema).default([]),
    enabled: z.boolean().optional(),
    readOnly: z.boolean().optional(),
    sourcePluginId: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addInlineSecretIssues);

export const SkillHostCompatibilityIssueLevelSchema = z.enum(["error", "warning"]);

export const SkillHostCompatibilityIssueSchema = z
  .object({
    level: SkillHostCompatibilityIssueLevelSchema,
    code: z.string().min(1),
    message: z.string().min(1),
    skillId: z.string().min(1).optional(),
    host: MarketplaceHostSchema.optional(),
    path: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addInlineSecretIssues);

export const McpCapabilitySchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1).optional(),
    description: z.string().optional(),
    server: McpServerConfigSchema.optional(),
    provenance: z.string().optional(),
    scope: z.enum(["project", "user", "system", "server"]).optional(),
    enabled: z.boolean().optional(),
  })
  .passthrough();

export const ProviderProfileSchema = z.preprocess(
  normalizeProviderProfileInput,
  z
    .object({
      id: z.string().min(1),
      label: z.string().min(1),
      kind: ProviderKindSchema,
      apiMode: ModelApiModeSchema.optional(),
      baseUrl: z.string().min(1).optional(),
      apiEntrypoints: ProviderApiEntrypointsSchema,
      authMode: ProviderAuthModeSchema.default("api_key"),
      secretRef: z
        .object({
          source: z.enum(["env", "local_keychain", "server_keychain", "prompt"]),
          key: z.string().min(1),
        })
        .optional(),
      runtimeReady: z.boolean().optional(),
      runtimeNote: z.string().optional(),
      enabled: z.boolean().optional(),
      readOnly: z.boolean().optional(),
    })
    .passthrough(),
);

export const ModelProfileSchema = IndependentModelSchema;
export const ModelSupplyCapabilitySchema = ModelSupplySchema;

export const HarnessCapabilitySchema = z
  .object({
    id: z.string().min(1),
    runtimeHarnessId: z.string().min(1).optional(),
    label: z.string().min(1),
    icon: z.string().min(1).optional(),
    modelControl: z.enum(["direct", "session", "unsupported"]),
    modelCompatibility: z.enum(["declared_apis", "any"]),
    supportedModelApis: z.array(ModelApiSchema).default([]),
    requiresExplicitModelRoute: z.boolean().default(false),
    passthroughEnv: z.array(z.string()).default([]),
    backend: AgentBackendSchema,
    software: SoftwareCapabilitySchema.optional(),
    mcps: z.array(z.string()).default([]),
    skills: z.array(z.string()).default([]),
    projectFiles: z.array(z.string()).default([]),
    enabled: z.boolean().optional(),
    readOnly: z.boolean().optional(),
    source: ExtensionSourceSchema.optional(),
  })
  .passthrough();

export const AgentCompositionContextInputSchema = z
  .object({
    mode: z.string().min(1).optional(),
    strategy: z.string().min(1).optional(),
  })
  .passthrough();

export const AgentCompositionPermissionsInputSchema = z
  .object({
    tools: z.string().min(1).optional(),
    mcp: z.string().min(1).optional(),
    shell: z.string().min(1).optional(),
    mode: z.string().min(1).optional(),
  })
  .passthrough();

export const AgentCompositionVisualInputSchema = z
  .object({
    color: z.string().min(1).optional(),
    icon: z.string().min(1).optional(),
    label: z.string().min(1).optional(),
  })
  .passthrough();

export const AgentCompositionPluginSelectionSchema = z
  .object({
    pluginId: z.string().min(1),
    skills: z.array(z.string().min(1)).default([]),
    mcpServers: z.array(z.string().min(1)).default([]),
  })
  .passthrough();

export const AgentProfileSchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1),
    displayName: z.string().min(1).optional(),
    description: z.string().optional(),
    selector: z.string().min(1).optional(),
    enabled: z.boolean().optional(),
    aliases: z.array(z.string().min(1)).default([]),
    instructions: z.string().optional(),
    harnessId: z.string().min(1).optional(),
    modelId: z.string().min(1).optional(),
    nativeModel: z.string().min(1).optional(),
    modelSupplyId: z.string().min(1).optional(),
    skills: z.array(z.string()).default([]),
    mcpServers: z.array(z.string()).default([]),
    tools: z.array(z.string()).default([]),
    disallowedTools: z.array(z.string()).default([]),
    permissionMode: z.string().min(1).optional(),
    sandboxMode: z.string().min(1).optional(),
    nicknameCandidates: z.array(z.string().min(1)).default([]),
    maxTurns: z.number().int().positive().optional(),
    memory: z.string().min(1).optional(),
    effort: z.string().min(1).optional(),
    background: z.boolean().optional(),
    isolation: z.string().min(1).optional(),
    color: z.string().min(1).optional(),
    definition: AgentDefinitionSourceSchema.optional(),
    context: AgentCompositionContextInputSchema.optional(),
    permissions: AgentCompositionPermissionsInputSchema.optional(),
    visual: AgentCompositionVisualInputSchema.optional(),
    pluginIds: z.array(z.string()).default([]),
    readOnly: z.boolean().optional(),
    source: ExtensionSourceSchema.optional(),
  })
  .passthrough();

export const AppConnectorCapabilitySchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1),
    kind: z.string().min(1),
    entrypoint: z.string().min(1).optional(),
    readOnly: z.boolean().optional(),
  })
  .passthrough();

export const UiContributionKindSchema = z.enum([
  "navigation_item",
  "view",
  "panel",
  "settings_panel",
  "dashboard_widget",
  "composer_action",
  "message_action",
  "inspector_section",
  "toolbar_action",
  "menu_item",
  "status_item",
  "custom",
]);

export const UiContributionPlacementSchema = z.enum([
  "sidebar",
  "topbar",
  "workspace",
  "settings",
  "dashboard",
  "agent_detail",
  "run_detail",
  "message",
  "composer",
  "command_palette",
  "menu",
  "toolbar",
  "inspector",
  "extension_detail",
  "status_bar",
  "custom",
]);

export const UiContributionSchema = z
  .object({
    id: z.string().min(1),
    kind: UiContributionKindSchema,
    name: z.string().min(1),
    description: z.string().optional(),
    placement: UiContributionPlacementSchema,
    order: z.number().optional(),
    icon: z.string().min(1).optional(),
    route: z.string().min(1).optional(),
    target: z.string().min(1).optional(),
    componentRef: z.string().min(1).optional(),
    assetRef: z.string().min(1).optional(),
    commandId: z.string().min(1).optional(),
    settingIds: z.array(z.string().min(1)).default([]),
    permissionIds: z.array(z.string().min(1)).default([]),
    authPolicyIds: z.array(z.string().min(1)).default([]),
    sourcePluginId: z.string().min(1).optional(),
    readOnly: z.boolean().optional(),
    provenance: z.string().optional(),
  })
  .passthrough()
  .superRefine(addInlineSecretIssues)
  .superRefine(addInlineUiPayloadIssues);

export const CommandCapabilitySchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1).optional(),
    description: z.string().optional(),
    command: z.array(z.string().min(1)).optional(),
    scope: z.enum(["project", "user", "system", "server", "plugin"]).optional(),
    readOnly: z.boolean().optional(),
    provenance: z.string().optional(),
  })
  .passthrough();

export const LspCapabilitySchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1).optional(),
    description: z.string().optional(),
    languages: z.array(z.string().min(1)).default([]),
    languageIds: z.array(z.string().min(1)).default([]),
    /**
     * Mention token prefixes that should be routed to this server by a host
     * composer.
     */
    mentionPrefixes: z.array(z.string().min(1)).default([]),
    command: z.union([z.array(z.string().min(1)), z.string().min(1)]).optional(),
    args: z.array(z.string().min(1)).default([]),
    cwd: z.string().min(1).optional(),
    scope: z.enum(["project", "user", "system", "server", "plugin"]).optional(),
    readOnly: z.boolean().optional(),
    provenance: z.string().optional(),
  })
  .passthrough();

export const HookCapabilitySchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1).optional(),
    description: z.string().optional(),
    event: z.string().min(1),
    command: z.array(z.string().min(1)).optional(),
    readOnly: z.boolean().optional(),
    provenance: z.string().optional(),
  })
  .passthrough();

export const MonitorCapabilitySchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1).optional(),
    description: z.string().optional(),
    trigger: z.enum(["manual", "schedule", "file_change", "event", "custom"]).default("manual"),
    schedule: z.string().min(1).optional(),
    readOnly: z.boolean().optional(),
    provenance: z.string().optional(),
  })
  .passthrough();

export const OutputStyleCapabilitySchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1).optional(),
    description: z.string().optional(),
    path: z.string().min(1).optional(),
    readOnly: z.boolean().optional(),
    provenance: z.string().optional(),
  })
  .passthrough();

export const PluginSettingCapabilitySchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1).optional(),
    description: z.string().optional(),
    valueType: z.enum(["string", "number", "boolean", "json", "secret_ref"]).default("string"),
    required: z.boolean().default(false),
    defaultValue: z.unknown().optional(),
    readOnly: z.boolean().optional(),
    provenance: z.string().optional(),
  })
  .passthrough();

export const PluginAssetCapabilitySchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1).optional(),
    description: z.string().optional(),
    kind: z
      .enum(["icon", "image", "document", "template", "binary", "archive", "dataset", "other"])
      .default("other"),
    path: z.string().min(1).optional(),
    url: z.string().min(1).optional(),
    sha256: z.string().min(1).optional(),
    readOnly: z.boolean().optional(),
    provenance: z.string().optional(),
  })
  .passthrough();

export const PermissionDeclarationSchema = z
  .object({
    id: z.string().min(1),
    kind: z.enum([
      "filesystem",
      "network",
      "process",
      "mcp",
      "lsp",
      "hook",
      "monitor",
      "secret",
      "setting",
      "custom",
    ]),
    access: z.enum(["read", "write", "execute", "network", "admin", "custom"]).default("read"),
    target: z.string().min(1).optional(),
    reason: z.string().optional(),
    required: z.boolean().default(false),
  })
  .passthrough();

export const AuthPolicySchema = z
  .object({
    id: z.string().min(1),
    kind: z
      .enum([
        "none",
        "provider_secret",
        "oauth",
        "api_key",
        "env",
        "local_keychain",
        "server_keychain",
        "custom",
      ])
      .default("none"),
    required: z.boolean().default(false),
    description: z.string().optional(),
    secretRefs: z
      .array(
        z.object({
          source: z.enum(["env", "local_keychain", "server_keychain", "prompt"]),
          key: z.string().min(1),
        }),
      )
      .default([]),
  })
  .passthrough();

export const MarketplaceSourceSchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1),
    host: MarketplaceHostSchema.default("custom"),
    kind: z.enum(["local_path", "remote_catalog", "host_native", "registry"]).default("local_path"),
    path: z.string().min(1).optional(),
    url: z.string().min(1).optional(),
    package: z.string().min(1).optional(),
    enabled: z.boolean().default(true),
    readOnly: z.boolean().optional(),
    trust: z.enum(["builtin", "local", "verified", "untrusted"]).default("local"),
    description: z.string().optional(),
  })
  .passthrough();

export const PluginCatalogEntrySchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1),
    version: z.string().min(1).optional(),
    marketplaceSourceId: z.string().min(1).optional(),
    bundleId: z.string().min(1).optional(),
    hosts: z.array(MarketplaceHostSchema).default([]),
    source: ExtensionSourceSchema.optional(),
    trust: z.enum(["builtin", "local", "verified", "untrusted"]).default("local"),
    installState: z
      .enum(["available", "installed", "enabled", "disabled", "update_available", "blocked"])
      .default("available"),
    updateState: z.enum(["unknown", "current", "update_available", "blocked"]).default("unknown"),
    providesHarness: z.boolean().default(false),
    componentCounts: z
      .object({
        software: z.number().int().nonnegative().default(0),
        commands: z.number().int().nonnegative().default(0),
        skills: z.number().int().nonnegative().default(0),
        mcpServers: z.number().int().nonnegative().default(0),
        lspServers: z.number().int().nonnegative().default(0),
        agents: z.number().int().nonnegative().default(0),
        hooks: z.number().int().nonnegative().default(0),
        monitors: z.number().int().nonnegative().default(0),
        outputStyles: z.number().int().nonnegative().default(0),
        appConnectors: z.number().int().nonnegative().default(0),
        uiContributions: z.number().int().nonnegative().default(0),
        assets: z.number().int().nonnegative().default(0),
        settings: z.number().int().nonnegative().default(0),
        permissions: z.number().int().nonnegative().default(0),
        authPolicies: z.number().int().nonnegative().default(0),
      })
      .default({}),
    readOnly: z.boolean().optional(),
    description: z.string().optional(),
  })
  .passthrough();

export const ExtensionBundleSchema = z
  .object({
    schemaVersion: z.literal(1).default(1),
    id: z.string().min(1),
    name: z.string().min(1),
    version: z.string().min(1),
    description: z.string().optional(),
    trust: z.enum(["builtin", "local", "verified", "untrusted"]).default("local"),
    enabled: z.boolean().optional(),
    readOnly: z.boolean().optional(),
    source: ExtensionSourceSchema.optional(),
    capabilities: z
      .object({
        software: z.array(SoftwareCapabilitySchema).default([]),
        skills: z.array(SkillCapabilitySchema).default([]),
        mcpServers: z.array(McpCapabilitySchema).default([]),
        models: z.array(ModelProfileSchema).default([]),
        modelSupplies: z.array(ModelSupplyCapabilitySchema).default([]),
        providers: z.array(ProviderProfileSchema).default([]),
        harnesses: z.array(HarnessCapabilitySchema).default([]),
        agents: z.array(AgentProfileSchema).default([]),
        appConnectors: z.array(AppConnectorCapabilitySchema).default([]),
        uiContributions: z.array(UiContributionSchema).default([]),
        commands: z.array(CommandCapabilitySchema).default([]),
        lspServers: z.array(LspCapabilitySchema).default([]),
        hooks: z.array(HookCapabilitySchema).default([]),
        monitors: z.array(MonitorCapabilitySchema).default([]),
        outputStyles: z.array(OutputStyleCapabilitySchema).default([]),
        settings: z.array(PluginSettingCapabilitySchema).default([]),
        assets: z.array(PluginAssetCapabilitySchema).default([]),
        permissions: z.array(PermissionDeclarationSchema).default([]),
        authPolicies: z.array(AuthPolicySchema).default([]),
        marketplaceSources: z.array(MarketplaceSourceSchema).default([]),
        pluginCatalog: z.array(PluginCatalogEntrySchema).default([]),
      })
      .default({}),
  })
  .passthrough();

export const AgentCompositionSchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1).optional(),
    displayName: z.string().min(1).optional(),
    description: z.string().optional(),
    selector: z.string().min(1).optional(),
    enabled: z.boolean().optional(),
    agentProfileId: z.string().min(1).optional(),
    harnessId: z.string().min(1).optional(),
    modelId: z.string().min(1).optional(),
    modelSupplyId: z.string().min(1).optional(),
    effort: z.string().min(1).optional(),
    skills: z.array(z.string()).default([]),
    mcpServers: z.array(z.string()).default([]),
    pluginIds: z.array(z.string()).default([]),
    plugins: z.array(AgentCompositionPluginSelectionSchema).default([]),
    definition: AgentDefinitionSourceSchema.optional(),
    context: AgentCompositionContextInputSchema.optional(),
    memory: z.string().min(1).optional(),
    permissions: AgentCompositionPermissionsInputSchema.optional(),
    visual: AgentCompositionVisualInputSchema.optional(),
    host: z.enum(["local", "server"]).default("local"),
  })
  .passthrough();

export const AgentCompositionStatusSchema = z.enum([
  "draft",
  "ready",
  "disabled",
  "blocked",
  "running",
  "failed",
  "stale",
]);

export const AgentCompositionHealthStatusSchema = z.enum(["ready", "blocked"]);

export const AgentCompositionRequirementKindSchema = z.enum([
  "agent_profile",
  "harness",
  "model_supply",
  "provider_profile",
  "model",
  "plugin",
  "skill",
  "mcp_server",
  "context",
  "permission",
  "definition",
  "secret",
  "host",
]);

export const AgentCompositionRequirementStatusSchema = z.enum([
  "ok",
  "missing",
  "ambiguous",
  "disabled",
  "blocked",
  "unsupported",
  "unavailable",
  "unknown",
]);

export const AgentCompositionRequirementSchema = z
  .object({
    kind: AgentCompositionRequirementKindSchema,
    status: AgentCompositionRequirementStatusSchema,
    id: z.string().min(1).optional(),
    sourcePluginId: z.string().min(1).optional(),
    message: z.string().min(1),
  })
  .passthrough()
  .superRefine(addInlineSecretIssues);

export const AgentCompositionCapabilityRefSchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1).optional(),
    sourcePluginId: z.string().min(1).optional(),
    status: AgentCompositionRequirementStatusSchema.default("ok"),
  })
  .passthrough()
  .superRefine(addInlineSecretIssues);

export const AgentCompositionDefinitionSummarySchema = z
  .object({
    source: z.enum([
      "none",
      "inline",
      "local",
      "project",
      "user",
      "plugin",
      "host",
      "server",
      "imported",
    ]),
    path: z.string().min(1).optional(),
    pluginId: z.string().min(1).optional(),
    label: z.string().min(1).optional(),
    host: z.enum(["claude_code", "codex", "swarmx", "custom"]).optional(),
    format: z.enum(["claude_code", "codex"]).optional(),
    readOnly: z.boolean().optional(),
  })
  .passthrough()
  .superRefine(addInlineSecretIssues);

export const AgentCompositionContextSummarySchema = z
  .object({
    mode: ContextPacketModeSchema,
    strategy: ContextStrategySchema,
    memory: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addInlineSecretIssues);

export const AgentCompositionPermissionSummarySchema = z
  .object({
    tools: z.string().min(1).optional(),
    mcp: z.string().min(1).optional(),
    shell: z.string().min(1).optional(),
    mode: z.string().min(1).optional(),
    summary: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addInlineSecretIssues);

export const AgentCompositionVisualSummarySchema = z
  .object({
    label: z.string().min(1).optional(),
    color: z.string().min(1).optional(),
    icon: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addInlineSecretIssues);

export const AgentCompositionPlanSchema = z
  .object({
    id: z.string().min(1),
    agentId: z.string().min(1),
    agentProfileId: z.string().min(1).optional(),
    displayName: z.string().min(1),
    canonicalSelector: z.string().min(1),
    host: z.enum(["local", "server"]),
    status: AgentCompositionStatusSchema,
    healthStatus: AgentCompositionHealthStatusSchema,
    harnessId: z.string().min(1).optional(),
    harnessLabel: z.string().min(1).optional(),
    modelId: z.string().min(1).optional(),
    runtimeModel: z.string().min(1).optional(),
    modelSupplyId: z.string().min(1).optional(),
    supplyLabel: z.string().min(1).optional(),
    effort: z.string().min(1).optional(),
    definition: AgentCompositionDefinitionSummarySchema,
    pluginIds: z.array(z.string().min(1)).default([]),
    skills: z.array(AgentCompositionCapabilityRefSchema).default([]),
    mcpServers: z.array(AgentCompositionCapabilityRefSchema).default([]),
    context: AgentCompositionContextSummarySchema.optional(),
    permissions: AgentCompositionPermissionSummarySchema.optional(),
    visual: AgentCompositionVisualSummarySchema.optional(),
    requirements: z.array(AgentCompositionRequirementSchema).default([]),
  })
  .passthrough()
  .superRefine(addInlineSecretIssues);

export type ExtensionBundle = z.infer<typeof ExtensionBundleSchema>;
export type ExtensionSource = z.infer<typeof ExtensionSourceSchema>;
export type SoftwareCapability = z.infer<typeof SoftwareCapabilitySchema>;
export type SkillHostExposureStatus = z.infer<typeof SkillHostExposureStatusSchema>;
export type SkillHostExposure = z.infer<typeof SkillHostExposureSchema>;
export type SkillHostCompatibilityIssueLevel = z.infer<
  typeof SkillHostCompatibilityIssueLevelSchema
>;
export type SkillHostCompatibilityIssue = z.infer<typeof SkillHostCompatibilityIssueSchema>;
export type SkillCapability = z.infer<typeof SkillCapabilitySchema>;
export type McpCapability = z.infer<typeof McpCapabilitySchema>;
export type ModelProfile = z.infer<typeof ModelProfileSchema>;
export type ModelSupplyCapability = z.infer<typeof ModelSupplyCapabilitySchema>;
export type ProviderProfile = z.infer<typeof ProviderProfileSchema>;
export type ProviderApiCompatibility = z.infer<typeof ProviderApiCompatibilitySchema>;
export type HarnessCapability = z.infer<typeof HarnessCapabilitySchema>;
export type AgentProfile = z.infer<typeof AgentProfileSchema>;
export type AppConnectorCapability = z.infer<typeof AppConnectorCapabilitySchema>;
export type UiContributionKind = z.infer<typeof UiContributionKindSchema>;
export type UiContributionPlacement = z.infer<typeof UiContributionPlacementSchema>;
export type UiContribution = z.infer<typeof UiContributionSchema>;
export type CommandCapability = z.infer<typeof CommandCapabilitySchema>;
export type LspCapability = z.infer<typeof LspCapabilitySchema>;
export type HookCapability = z.infer<typeof HookCapabilitySchema>;
export type MonitorCapability = z.infer<typeof MonitorCapabilitySchema>;
export type OutputStyleCapability = z.infer<typeof OutputStyleCapabilitySchema>;
export type PluginSettingCapability = z.infer<typeof PluginSettingCapabilitySchema>;
export type PluginAssetCapability = z.infer<typeof PluginAssetCapabilitySchema>;
export type PermissionDeclaration = z.infer<typeof PermissionDeclarationSchema>;
export type AuthPolicy = z.infer<typeof AuthPolicySchema>;
export type MarketplaceHost = z.infer<typeof MarketplaceHostSchema>;
export type MarketplaceSource = z.infer<typeof MarketplaceSourceSchema>;
export type PluginCatalogEntry = z.infer<typeof PluginCatalogEntrySchema>;
export type AgentComposition = z.infer<typeof AgentCompositionSchema>;
export type AgentCompositionStatus = z.infer<typeof AgentCompositionStatusSchema>;
export type AgentCompositionHealthStatus = z.infer<typeof AgentCompositionHealthStatusSchema>;
export type AgentCompositionRequirementKind = z.infer<typeof AgentCompositionRequirementKindSchema>;
export type AgentCompositionRequirementStatus = z.infer<
  typeof AgentCompositionRequirementStatusSchema
>;
export type AgentCompositionRequirement = z.infer<typeof AgentCompositionRequirementSchema>;
export type AgentCompositionCapabilityRef = z.infer<typeof AgentCompositionCapabilityRefSchema>;
export type AgentCompositionDefinitionSummary = z.infer<
  typeof AgentCompositionDefinitionSummarySchema
>;
export type AgentCompositionContextSummary = z.infer<typeof AgentCompositionContextSummarySchema>;
export type AgentCompositionPermissionSummary = z.infer<
  typeof AgentCompositionPermissionSummarySchema
>;
export type AgentCompositionVisualSummary = z.infer<typeof AgentCompositionVisualSummarySchema>;
export type AgentCompositionPlan = z.infer<typeof AgentCompositionPlanSchema>;

export interface ExtensionLoadWarning {
  source: string;
  message: string;
}

export interface ExtensionInventory {
  bundles: ExtensionBundle[];
  software: SoftwareCapability[];
  skills: SkillCapability[];
  mcpServers: McpCapability[];
  models: ModelProfile[];
  modelSupplies: ModelSupplyCapability[];
  providers: ProviderProfile[];
  harnesses: HarnessCapability[];
  agents: AgentProfile[];
  appConnectors: AppConnectorCapability[];
  uiContributions: UiContribution[];
  commands: CommandCapability[];
  lspServers: LspCapability[];
  hooks: HookCapability[];
  monitors: MonitorCapability[];
  outputStyles: OutputStyleCapability[];
  settings: PluginSettingCapability[];
  assets: PluginAssetCapability[];
  permissions: PermissionDeclaration[];
  authPolicies: AuthPolicy[];
  marketplaceSources: MarketplaceSource[];
  pluginCatalog: PluginCatalogEntry[];
  warnings: ExtensionLoadWarning[];
}

export interface LoadExtensionInventoryOptions {
  roots?: string[];
  includeBuiltins?: boolean;
}

export interface ResolveAgentRuntimeEnvOptions {
  env?: NodeJS.ProcessEnv;
  providerSecrets?: Readonly<Record<string, string>>;
}

export interface ExecuteAgentCompositionOptions {
  inventory?: ExtensionInventory;
  inventoryOptions?: LoadExtensionInventoryOptions;
  env?: NodeJS.ProcessEnv;
  providerSecrets?: Readonly<Record<string, string>>;
  context?: Record<string, unknown>;
  cwd?: string;
  localTools?: readonly LocalMcpTool[];
  onChunk?: (chunk: MessageChunk) => void;
}

export interface ValidateSkillHostCompatibilityOptions {
  canonicalRoots?: string[];
  knownSkillIds?: string[];
  requireDotSlashLocalPathsForHosts?: MarketplaceHost[];
}

export function parseExtensionBundle(input: unknown, sourcePath?: string): ExtensionBundle {
  assertNoInlineSecrets(input);
  const bundle = ExtensionBundleSchema.parse(input);
  if (sourcePath && !bundle.source) {
    return { ...bundle, source: { type: "path", path: sourcePath } };
  }
  return bundle;
}

export function createExtensionInventory(
  bundles: ExtensionBundle[],
  warnings: ExtensionLoadWarning[] = [],
): ExtensionInventory {
  return {
    bundles,
    software: bundles.flatMap((bundle) => bundle.capabilities.software),
    skills: bundles.flatMap((bundle) => bundle.capabilities.skills),
    mcpServers: bundles.flatMap((bundle) => bundle.capabilities.mcpServers),
    models: bundles.flatMap((bundle) => bundle.capabilities.models),
    modelSupplies: bundles.flatMap((bundle) => bundle.capabilities.modelSupplies),
    providers: bundles.flatMap((bundle) => bundle.capabilities.providers),
    harnesses: bundles.flatMap((bundle) => bundle.capabilities.harnesses),
    agents: bundles.flatMap((bundle) => bundle.capabilities.agents),
    appConnectors: bundles.flatMap((bundle) => bundle.capabilities.appConnectors),
    uiContributions: bundles.flatMap((bundle) => bundle.capabilities.uiContributions),
    commands: bundles.flatMap((bundle) => bundle.capabilities.commands),
    lspServers: bundles.flatMap((bundle) => bundle.capabilities.lspServers),
    hooks: bundles.flatMap((bundle) => bundle.capabilities.hooks),
    monitors: bundles.flatMap((bundle) => bundle.capabilities.monitors),
    outputStyles: bundles.flatMap((bundle) => bundle.capabilities.outputStyles),
    settings: bundles.flatMap((bundle) => bundle.capabilities.settings),
    assets: bundles.flatMap((bundle) => bundle.capabilities.assets),
    permissions: bundles.flatMap((bundle) => bundle.capabilities.permissions),
    authPolicies: bundles.flatMap((bundle) => bundle.capabilities.authPolicies),
    marketplaceSources: bundles.flatMap((bundle) => bundle.capabilities.marketplaceSources),
    pluginCatalog: bundles.flatMap((bundle) => bundle.capabilities.pluginCatalog),
    warnings,
  };
}

export function validateSkillHostCompatibility(
  bundleInput: unknown,
  options: ValidateSkillHostCompatibilityOptions = {},
): SkillHostCompatibilityIssue[] {
  const bundle = parseExtensionBundle(bundleInput);
  const canonicalRoots = (options.canonicalRoots ?? []).map(normalizeManifestPath);
  const knownSkillIds = new Set([
    ...bundle.capabilities.skills.map((skill) => skill.id),
    ...(options.knownSkillIds ?? []),
  ]);
  const dotSlashHosts = new Set(options.requireDotSlashLocalPathsForHosts ?? []);
  const marketplaceById = new Map(
    bundle.capabilities.marketplaceSources.map((source) => [source.id, source]),
  );
  const issues: SkillHostCompatibilityIssue[] = [];

  for (const skill of bundle.capabilities.skills) {
    const canonicalPath = skill.canonicalPath ?? skill.path;
    if (
      canonicalPath &&
      canonicalRoots.length > 0 &&
      !isUnderAnyRoot(canonicalPath, canonicalRoots)
    ) {
      issues.push(
        skillHostIssue({
          code: "skill_path_outside_canonical_roots",
          skillId: skill.id,
          path: canonicalPath,
          message: `Skill "${skill.id}" uses path "${canonicalPath}" outside canonical roots ${canonicalRoots.join(", ")}.`,
        }),
      );
    }

    for (const gateSkillId of skill.requiresGateSkillIds) {
      if (gateSkillId === skill.id) {
        issues.push(
          skillHostIssue({
            code: "skill_gate_self_reference",
            skillId: skill.id,
            message: `Skill "${skill.id}" cannot require itself as a gate skill.`,
          }),
        );
      } else if (!knownSkillIds.has(gateSkillId)) {
        issues.push(
          skillHostIssue({
            code: "unknown_gate_skill",
            skillId: skill.id,
            message: `Skill "${skill.id}" requires unknown gate skill "${gateSkillId}".`,
          }),
        );
      }
    }

    for (const exposure of skill.hostExposures) {
      if (exposure.status === "plugin" && !canonicalPath) {
        issues.push(
          skillHostIssue({
            code: "plugin_exposure_without_skill_path",
            skillId: skill.id,
            host: exposure.host,
            message: `Skill "${skill.id}" exposes a ${exposure.host} plugin surface without a skill path.`,
          }),
        );
      }

      if (exposure.status === "rules_only" && exposure.manifestPath) {
        issues.push(
          skillHostIssue({
            code: "rules_only_manifest_claim",
            skillId: skill.id,
            host: exposure.host,
            path: exposure.manifestPath,
            message: `Rules-only exposure for "${skill.id}" on ${exposure.host} must not claim a plugin manifest path.`,
          }),
        );
      }

      if (exposure.marketplaceSourceId && !marketplaceById.has(exposure.marketplaceSourceId)) {
        issues.push(
          skillHostIssue({
            code: "unknown_marketplace_source",
            skillId: skill.id,
            host: exposure.host,
            message: `Skill "${skill.id}" references unknown marketplace source "${exposure.marketplaceSourceId}".`,
          }),
        );
      }

      if (dotSlashHosts.has(exposure.host)) {
        for (const localPath of [exposure.manifestPath, exposure.rulesPath].filter(
          (value): value is string => Boolean(value),
        )) {
          if (!localPath.startsWith("./")) {
            issues.push(
              skillHostIssue({
                code: "host_local_path_must_be_dot_slash",
                skillId: skill.id,
                host: exposure.host,
                path: localPath,
                message: `Local ${exposure.host} paths must be repository-root paths prefixed with "./".`,
              }),
            );
          }
        }
      }
    }
  }

  for (const source of bundle.capabilities.marketplaceSources) {
    if (
      dotSlashHosts.has(source.host) &&
      source.kind === "local_path" &&
      source.path &&
      !source.path.startsWith("./")
    ) {
      issues.push(
        skillHostIssue({
          code: "marketplace_local_path_must_be_dot_slash",
          host: source.host,
          path: source.path,
          message: `Local ${source.host} marketplace paths must be repository-root paths prefixed with "./".`,
        }),
      );
    }
  }

  return issues;
}

export async function loadExtensionInventory(
  options: LoadExtensionInventoryOptions = {},
): Promise<ExtensionInventory> {
  const roots = options.roots ?? extensionRootsFromEnv();
  const includeBuiltins = options.includeBuiltins ?? true;
  const bundles: ExtensionBundle[] = includeBuiltins ? [builtInExtensionBundle()] : [];
  const warnings: ExtensionLoadWarning[] = [];

  for (const root of roots) {
    for (const manifestPath of await discoverExtensionManifestPaths(root, warnings)) {
      try {
        const text = await readFile(manifestPath, "utf8");
        bundles.push(parseExtensionBundle(JSON.parse(text), manifestPath));
      } catch (error) {
        warnings.push({
          source: manifestPath,
          message: error instanceof Error ? error.message : String(error),
        });
      }
    }
  }

  return createExtensionInventory(bundles, warnings);
}

export function parseAgentCompositionPlan(input: unknown): AgentCompositionPlan {
  return AgentCompositionPlanSchema.parse(input);
}

export function resolveAgentCompositionPlan(
  compositionInput: unknown,
  inventory: ExtensionInventory,
): AgentCompositionPlan {
  assertNoInlineSecrets(compositionInput);
  const composition = AgentCompositionSchema.parse(compositionInput);
  const requirements: AgentCompositionRequirement[] = [];

  const profileResolution = composition.agentProfileId
    ? resolveForPlan(inventory.agents, composition.agentProfileId, "agent profile", "agent_profile")
    : undefined;
  if (profileResolution) requirements.push(profileResolution.requirement);
  const profile = profileResolution?.item;

  if (composition.enabled === false) {
    requirements.push({
      kind: "agent_profile",
      status: "disabled",
      id: composition.agentProfileId ?? composition.id,
      message: `Agent composition "${composition.id}" is disabled.`,
    });
  }
  if (profile?.enabled === false) {
    requirements.push({
      kind: "agent_profile",
      status: "disabled",
      id: profile.id,
      message: `Agent profile "${profile.id}" is disabled.`,
    });
  }

  const pluginSelections = composition.plugins ?? [];
  const selectedSkills = uniqueStrings([
    ...(profile?.skills ?? []),
    ...composition.skills,
    ...pluginSelections.flatMap((plugin) => plugin.skills),
  ]);
  const selectedMcpIds = uniqueStrings([
    ...(profile?.mcpServers ?? []),
    ...composition.mcpServers,
    ...pluginSelections.flatMap((plugin) => plugin.mcpServers),
  ]);

  const harnessId = composition.harnessId ?? profile?.harnessId;
  const harnessResolution = harnessId
    ? resolveForPlan(inventory.harnesses, harnessId, "harness", "harness")
    : missingRequirement<HarnessCapability>(
        "harness",
        "Agent composition must resolve one harness.",
      );
  requirements.push(harnessResolution.requirement);
  const harness = harnessResolution.item;

  const modelId = composition.modelId ?? profile?.modelId;
  const modelResolution = modelId
    ? resolveForPlan(inventory.models, modelId, "model", "model")
    : missingRequirement<ModelProfile>(
        "model",
        `Agent composition "${composition.id}" must resolve one Model.`,
      );
  requirements.push(modelResolution.requirement);
  const model = modelResolution.item;
  const requestedModelSupplyId = composition.modelSupplyId ?? profile?.modelSupplyId;
  const modelSupplyId =
    requestedModelSupplyId ??
    (model ? selectInternalModelSupply(inventory, model.id, harness?.id)?.id : undefined);
  const supplyResolution = modelSupplyId
    ? resolveForPlan(inventory.modelSupplies, modelSupplyId, "model supply", "model_supply")
    : undefined;
  if (supplyResolution) requirements.push(supplyResolution.requirement);
  const supply = supplyResolution?.item;
  if (supply && model && supply.modelId !== model.id) {
    requirements.push({
      kind: "model_supply",
      status: "blocked",
      id: supply.id,
      message: `Model supply "${supply.id}" does not supply Model "${model.id}".`,
    });
  }
  if (supply && harness && !modelSupplySupportsHarness(supply, harness)) {
    requirements.push({
      kind: "model_supply",
      status: "unsupported",
      id: supply.id,
      message: `Model supply "${supply.id}" cannot execute through Harness "${harness.id}".`,
    });
  }
  const providerResolution = supply
    ? resolveForPlan(
        inventory.providers,
        supply.providerProfileId,
        "provider profile",
        "provider_profile",
      )
    : undefined;
  if (providerResolution) requirements.push(providerResolution.requirement);
  const provider = providerResolution?.item;
  if (provider?.secretRef) {
    requirements.push({
      kind: "secret",
      status: "unknown",
      id: `${provider.secretRef.source}:${provider.secretRef.key}`,
      message: `Model supply "${supply?.id}" requires a ${provider.secretRef.source} secret at invocation time.`,
    });
  }

  const resolvedMatrixModel =
    harness && model
      ? resolveHarnessModelInventory({
          harnessId: harness.id,
          models: [model],
          supplies: supply ? [supply] : [],
          providers: provider ? [provider] : [],
          harnesses: [harness],
        })[0]
      : undefined;
  if (harness && model && !resolvedMatrixModel) {
    requirements.push({
      kind: "model",
      status: "unsupported",
      id: model.id,
      message: `Harness "${harness.id}" cannot select Model "${model.id}".`,
    });
  }
  const effort =
    harness && model && resolvedMatrixModel
      ? normalizeModelReasoningEffort(
          {
            harnessId: harness.id,
            modelId: model.id,
            apiProtocol: resolvedMatrixModel.apiProtocol,
            effort: composition.effort ?? profile?.effort,
          },
          modelRouteCapabilityRegistry(model, supply),
        )
      : undefined;
  const baseRuntimeModel =
    supply?.runtimeModel ?? resolvedMatrixModel?.runtimeModel ?? model?.runtimeModel;
  const adapterId = harness ? runtimeHarnessId(harness) : undefined;
  const runtimeModel =
    harness && model && baseRuntimeModel && adapterId && HARNESSES[adapterId]
      ? harnessModelRuntimeModel(adapterId, {
          modelId: model.id,
          runtimeModel: baseRuntimeModel,
        })
      : baseRuntimeModel;

  const skillResults = selectedSkills.map((id) =>
    resolveCapabilityRefForPlan("skill", id, inventory.skills, inventory),
  );
  const mcpResults = selectedMcpIds.map((id) =>
    resolveCapabilityRefForPlan("mcp_server", id, inventory.mcpServers, inventory),
  );
  for (const result of [...skillResults, ...mcpResults]) {
    requirements.push(result.requirement);
  }

  const pluginIds = uniqueStrings([
    ...(profile?.pluginIds ?? []),
    ...composition.pluginIds,
    ...pluginSelections.map((plugin) => plugin.pluginId),
    ...skillResults.flatMap((result) =>
      result.ref.sourcePluginId ? [result.ref.sourcePluginId] : [],
    ),
    ...mcpResults.flatMap((result) =>
      result.ref.sourcePluginId ? [result.ref.sourcePluginId] : [],
    ),
  ]);
  for (const pluginId of pluginIds) {
    requirements.push(pluginRequirementForPlan(pluginId, inventory));
  }

  const context = contextSummaryForPlan(composition, profile, requirements);
  const permissions = permissionSummaryForPlan(composition, profile, selectedMcpIds);
  if (permissions) {
    requirements.push({
      kind: "permission",
      status: "ok",
      id: permissions.mode ?? permissions.shell ?? "policy",
      message: "Resolved permission policy summary.",
    });
  }

  const displayName =
    composition.displayName ??
    composition.name ??
    profile?.displayName ??
    profile?.name ??
    composition.id;
  const definition = definitionSummaryForPlan(
    composition.definition ?? profile?.definition,
    profile,
  );
  const visual = visualSummaryForPlan(composition, profile, displayName);
  const status = compositionStatusForRequirements(requirements);

  return AgentCompositionPlanSchema.parse({
    id: composition.id,
    agentId:
      (harness?.id ?? harnessId) && modelId
        ? `${harness?.id ?? harnessId}:${modelId}`
        : (profile?.id ?? composition.agentProfileId ?? composition.id),
    agentProfileId: profile?.id,
    displayName,
    canonicalSelector: toCanonicalSelector(
      composition.selector ??
        profile?.selector ??
        profile?.aliases[0] ??
        profile?.id ??
        composition.id,
    ),
    host: composition.host,
    status,
    healthStatus: status === "ready" ? "ready" : "blocked",
    harnessId: harness?.id ?? harnessId,
    harnessLabel: harness?.label,
    modelId: model?.id ?? modelId,
    runtimeModel,
    modelSupplyId: supply?.id ?? modelSupplyId,
    supplyLabel: provider?.label ?? supply?.id,
    effort,
    definition,
    pluginIds,
    skills: skillResults.map((result) => result.ref),
    mcpServers: mcpResults.map((result) => result.ref),
    context,
    permissions,
    visual,
    requirements,
  });
}

export function resolveAgentComposition(
  compositionInput: unknown,
  inventory: ExtensionInventory,
): AgentConfig {
  const plan = resolveAgentCompositionPlan(compositionInput, inventory);
  assertPlanReadyForExecution(plan);
  const composition = AgentCompositionSchema.parse(compositionInput);
  const profile = resolveOptionalById(
    inventory.agents,
    composition.agentProfileId,
    "agent profile",
  );

  const harnessId = composition.harnessId ?? profile?.harnessId;
  if (!harnessId) {
    throw new Error(`Agent composition "${composition.id}" must resolve one harness.`);
  }
  const harness = resolveById(inventory.harnesses, harnessId, "harness");

  const modelId = composition.modelId ?? profile?.modelId;
  if (!modelId) throw new Error(`Agent composition "${composition.id}" must resolve one Model.`);
  const model = resolveById(inventory.models, modelId, "model");
  const supply = resolveOptionalById(inventory.modelSupplies, plan.modelSupplyId, "model supply");
  if (supply && supply.modelId !== model.id) {
    throw new Error(`Model supply "${supply.id}" does not supply Model "${model.id}".`);
  }
  const provider = supply
    ? resolveById(inventory.providers, supply.providerProfileId, "provider profile")
    : undefined;
  const matrixModel = resolveHarnessModelInventory({
    harnessId: harness.id,
    models: [model],
    supplies: supply ? [supply] : [],
    providers: provider ? [provider] : [],
    harnesses: [harness],
  })[0];
  if (!matrixModel) {
    throw new Error(`Harness "${harness.id}" cannot select Model "${model.id}".`);
  }
  const reasoning = matrixModel.reasoning;
  const reasoningRegistry = modelRouteCapabilityRegistry(model, supply);
  const effort = reasoning
    ? normalizeModelReasoningEffort(
        {
          harnessId: harness.id,
          modelId: model.id,
          apiProtocol: matrixModel.apiProtocol,
          effort: composition.effort ?? profile?.effort,
        },
        reasoningRegistry,
      )
    : undefined;
  const adapterId = runtimeHarnessId(harness);
  const baseRuntimeModel = supply?.runtimeModel ?? matrixModel.runtimeModel;
  const runtimeModel = HARNESSES[adapterId]
    ? harnessModelRuntimeModel(adapterId, {
        modelId: model.id,
        runtimeModel: baseRuntimeModel,
      })
    : baseRuntimeModel;

  const pluginSelections = composition.plugins ?? [];
  const selectedSkills = uniqueStrings([
    ...(profile?.skills ?? []),
    ...composition.skills,
    ...pluginSelections.flatMap((plugin) => plugin.skills),
  ]);
  const selectedMcpIds = uniqueStrings([
    ...(profile?.mcpServers ?? []),
    ...composition.mcpServers,
    ...pluginSelections.flatMap((plugin) => plugin.mcpServers),
  ]);
  const mcpServers = selectedMcpIds.reduce<Record<string, McpServerConfig>>((servers, id) => {
    const mcp = resolveById(inventory.mcpServers, id, "MCP server");
    if (mcp.server) servers[id] = mcp.server;
    return servers;
  }, {});

  const name = toAgentConfigName(`${harness.id}-${model.id}`);
  const pluginIds = plan.pluginIds;

  return {
    name,
    description: composition.description ?? profile?.description,
    model: runtimeModel,
    instructions: profile?.instructions ?? "",
    backend: harness.backend as AgentBackend,
    client: adapterId === "swarmx" ? { apiProtocol: matrixModel.apiProtocol } : undefined,
    mcpServers: Object.keys(mcpServers).length > 0 ? mcpServers : undefined,
    parameters: {
      ...(reasoning && effort
        ? {
            reasoning: {
              capabilityId: reasoning.capabilityId,
              control: reasoning.reasoningControl,
              effort,
              supportedEfforts: reasoning.supportedEfforts,
              parameterMapping: reasoning.parameterMapping,
              source: reasoning.source,
            },
          }
        : {}),
      extension: {
        compositionId: composition.id,
        agentProfileId: profile?.id,
        canonicalSelector: plan.canonicalSelector,
        harnessId: harness.id,
        modelId: model.id,
        modelSupplyId: supply?.id,
        host: composition.host,
        pluginIds,
        skills: selectedSkills,
        mcpServers: selectedMcpIds,
        definition: plan.definition,
        context: plan.context,
        permissions: plan.permissions,
        visual: plan.visual,
        planStatus: plan.status,
        profile: profile
          ? {
              tools: profile.tools,
              disallowedTools: profile.disallowedTools,
              permissionMode: profile.permissionMode,
              sandboxMode: profile.sandboxMode,
              nicknameCandidates: profile.nicknameCandidates,
              nativeModel: profile.nativeModel,
              maxTurns: profile.maxTurns,
              memory: profile.memory,
              ...(effort ? { effort } : {}),
              background: profile.background,
              isolation: profile.isolation,
              color: profile.color,
              definition: profile.definition,
            }
          : undefined,
      },
      harness: harnessDescriptor(harness),
    },
  };
}

export function resolveAgentCompositionRuntimeEnv(
  compositionInput: unknown,
  inventory: ExtensionInventory,
  options: ResolveAgentRuntimeEnvOptions = {},
): Record<string, string> {
  const plan = resolveAgentCompositionPlan(compositionInput, inventory);
  assertPlanReadyForExecution(plan);
  const composition = AgentCompositionSchema.parse(compositionInput);
  const profile = resolveOptionalById(
    inventory.agents,
    composition.agentProfileId,
    "agent profile",
  );
  const harnessId = composition.harnessId ?? profile?.harnessId;
  if (!harnessId) return {};
  const harness = resolveById(inventory.harnesses, harnessId, "harness");
  const sourceEnv = options.env ?? process.env;
  const passthroughEnv = selectRuntimeEnv(sourceEnv, harness.passthroughEnv);
  const modelId = composition.modelId ?? profile?.modelId;
  if (!modelId) return passthroughEnv;
  const model = resolveById(inventory.models, modelId, "model");
  const supply = resolveOptionalById(inventory.modelSupplies, plan.modelSupplyId, "model supply");
  const adapterId = runtimeHarnessId(harness);
  const matrixModel = resolveHarnessModelInventory({
    harnessId: harness.id,
    models: [model],
    supplies: supply ? [supply] : [],
    providers: supply
      ? [resolveById(inventory.providers, supply.providerProfileId, "provider profile")]
      : [],
    harnesses: [harness],
  })[0];
  if (!matrixModel) {
    throw new Error(`Harness "${harness.id}" cannot select Model "${model.id}".`);
  }
  const baseRuntimeModel = supply?.runtimeModel ?? matrixModel.runtimeModel;
  const runtimeModel = HARNESSES[adapterId]
    ? harnessModelRuntimeModel(adapterId, {
        modelId: model.id,
        runtimeModel: baseRuntimeModel,
      })
    : baseRuntimeModel;
  const harnessEnv = HARNESSES[adapterId]
    ? harnessModelRuntimeEnv(adapterId, {
        modelId: model.id,
        runtimeModel,
        effort: plan.effort,
        env: sourceEnv,
      })
    : {};
  if (!supply) return { ...passthroughEnv, ...harnessEnv };
  if (supply.modelId !== model.id) {
    throw new Error(`Model supply "${supply.id}" does not supply Model "${model.id}".`);
  }
  const provider = resolveById(inventory.providers, supply.providerProfileId, "provider profile");
  const secret = providerSecretValue(provider, sourceEnv, options.providerSecrets);
  const providerEnv = buildProviderRuntimeEnv(
    {
      id: provider.id,
      label: provider.label,
      kind: provider.kind,
      apiMode: provider.apiMode,
      baseUrl: provider.baseUrl,
      apiEntrypoints: provider.apiEntrypoints,
      authMode: provider.authMode,
      secretRef: provider.secretRef,
    },
    {
      modelId: model.id,
      runtimeModel,
      secretValue: secret,
      targetApi: matrixModel.apiProtocol,
      apiCompatibility: supply.apiCompatibility,
    },
  ).env;
  return { ...passthroughEnv, ...providerEnv, ...harnessEnv };
}

export async function executeAgentComposition(
  compositionInput: unknown,
  messages: ChatMessage[],
  options: ExecuteAgentCompositionOptions = {},
): Promise<MessageChunk[]> {
  const inventory = options.inventory ?? (await loadExtensionInventory(options.inventoryOptions));
  const agentConfig = resolveAgentComposition(compositionInput, inventory);
  const runtimeEnv = resolveAgentCompositionRuntimeEnv(compositionInput, inventory, {
    env: options.env,
    providerSecrets: options.providerSecrets,
  });
  const swarm = new Swarm(
    singleAgentSwarmConfig(agentConfigWithRuntimeEnv(agentConfig, runtimeEnv, options.cwd)),
    { agent: { localTools: options.localTools } },
  );
  return swarm.execute({ messages }, options.context, options.onChunk);
}

export function builtInExtensionBundle(): ExtensionBundle {
  const harnesses = Object.entries(HARNESSES).map(([id, harness]) => ({
    id,
    runtimeHarnessId: id,
    label: harness.label,
    icon: harness.icon,
    modelControl: harness.modelControl,
    modelCompatibility: harness.modelCompatibility,
    supportedModelApis: harness.supportedModelApis,
    requiresExplicitModelRoute: harness.requiresExplicitModelRoute ?? false,
    passthroughEnv: harness.passthroughEnv,
    backend: harness.backend,
    enabled: harness.enabled,
    software: softwareFromBackend(id, harness.backend),
    mcps: [],
    skills: [],
    projectFiles: [],
    readOnly: true,
    source: { type: "builtin" as const },
  }));

  return {
    schemaVersion: 1,
    id: "swarmx.builtin",
    name: "SwarmX Built-ins",
    version: SWARMX_VERSION,
    trust: "builtin",
    readOnly: true,
    source: { type: "builtin" },
    capabilities: {
      software: [],
      skills: [],
      mcpServers: [],
      models: MODELS.map((model) => ({ ...model, catalogSource: "builtin" })),
      modelSupplies: [],
      providers: [],
      harnesses,
      agents: [],
      appConnectors: [],
      uiContributions: [],
      commands: [],
      lspServers: [
        {
          id: SWARMX_LOCAL_FILES_LSP_ID,
          name: "SwarmX Local Files",
          description: "Workspace-local file path completions for @ references.",
          languages: ["markdown", "plaintext"],
          languageIds: ["markdown", "plaintext"],
          mentionPrefixes: ["@"],
          args: [],
          scope: "project",
          readOnly: true,
          provenance: "swarmx.builtin",
        },
        {
          id: SWARMX_SKILLS_LSP_ID,
          name: "SwarmX Skills",
          description: "Extension inventory skill completions for $ references.",
          languages: ["markdown", "plaintext"],
          languageIds: ["markdown", "plaintext"],
          mentionPrefixes: ["$"],
          args: [],
          scope: "project",
          readOnly: true,
          provenance: "swarmx.builtin",
        },
      ],
      hooks: [],
      monitors: [],
      outputStyles: [],
      settings: [],
      assets: [],
      permissions: [],
      authPolicies: [],
      marketplaceSources: [],
      pluginCatalog: [],
    },
  };
}

export function extensionRootsFromEnv(env: NodeJS.ProcessEnv = process.env): string[] {
  const raw = env.SWARMX_EXTENSION_PATHS ?? env.SWARMX_EXTENSION_ROOTS ?? "";
  return raw
    .split(path.delimiter)
    .map((item) => item.trim())
    .filter(Boolean);
}

async function discoverExtensionManifestPaths(
  root: string,
  warnings: ExtensionLoadWarning[],
): Promise<string[]> {
  try {
    const rootStat = await stat(root);
    if (rootStat.isFile()) return [root];
    if (!rootStat.isDirectory()) return [];

    const entries = await readdir(root, { withFileTypes: true });
    const manifestPaths: string[] = [];
    for (const entry of entries) {
      const entryPath = path.join(root, entry.name);
      if (entry.isFile() && isManifestFilename(entry.name)) {
        manifestPaths.push(entryPath);
      } else if (entry.isDirectory()) {
        for (const filename of MANIFEST_FILENAMES) {
          const candidate = path.join(entryPath, filename);
          try {
            if ((await stat(candidate)).isFile()) manifestPaths.push(candidate);
          } catch {
            // Missing conventional manifest names are expected.
          }
        }
      }
    }
    return uniqueStrings(manifestPaths);
  } catch (error) {
    warnings.push({
      source: root,
      message: error instanceof Error ? error.message : String(error),
    });
    return [];
  }
}

function isManifestFilename(filename: string): boolean {
  return MANIFEST_FILENAMES.has(filename) || filename.endsWith(".swarmx-extension.json");
}

function assertNoInlineSecrets(value: unknown, trail: string[] = []): void {
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertNoInlineSecrets(item, [...trail, String(index)]));
    return;
  }
  if (!value || typeof value !== "object") return;

  for (const [key, child] of Object.entries(value)) {
    const normalizedKey = key.toLowerCase().replace(/[^a-z0-9_]/g, "");
    if (
      FORBIDDEN_SECRET_KEY_PATTERN.test(key) &&
      !ALLOWED_SECRET_REFERENCE_KEYS.has(normalizedKey)
    ) {
      throw new Error(`Extension manifest must not contain inline secret field "${key}".`);
    }
    assertNoInlineSecrets(child, [...trail, key]);
  }
}

function normalizeProviderProfileInput(input: unknown): unknown {
  if (!input || typeof input !== "object" || Array.isArray(input)) return input;
  const record = input as Record<string, unknown>;
  assertNoProviderOwnedModelFields(record);
  const {
    display_name,
    base_url,
    api_entrypoints,
    api_mode,
    auth_mode,
    secret_ref,
    name,
    ...rest
  } = record;

  return {
    ...rest,
    label: rest.label ?? rest.displayName ?? display_name ?? name,
    baseUrl: rest.baseUrl ?? base_url,
    apiEntrypoints: rest.apiEntrypoints ?? api_entrypoints,
    apiMode: rest.apiMode ?? api_mode,
    authMode: rest.authMode ?? auth_mode,
    secretRef: rest.secretRef ?? secret_ref,
  };
}

function assertNoProviderOwnedModelFields(input: Record<string, unknown>): void {
  const forbidden = new Set([
    "model",
    "models",
    "defaultmodel",
    "harnessmodeloverrides",
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

function normalizeProviderApiCompatibilityInput(input: unknown): unknown {
  if (typeof input === "string" || typeof input === "boolean") {
    return { mode: normalizeProviderApiCompatibilityMode(input) };
  }
  if (!input || typeof input !== "object" || Array.isArray(input)) return input;
  const { downstream_kind, target_kind, target_api, base_url, kind, translator, enabled, ...rest } =
    input as Record<string, unknown>;

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

function addInlineSecretIssues(
  value: unknown,
  ctx: z.RefinementCtx,
  trail: Array<string | number> = [],
): void {
  if (Array.isArray(value)) {
    value.forEach((item, index) => addInlineSecretIssues(item, ctx, [...trail, index]));
    return;
  }
  if (!value || typeof value !== "object") return;

  for (const [key, child] of Object.entries(value)) {
    const normalizedKey = key.toLowerCase().replace(/[^a-z0-9_]/g, "");
    if (
      FORBIDDEN_SECRET_KEY_PATTERN.test(key) &&
      !ALLOWED_SECRET_REFERENCE_KEYS.has(normalizedKey)
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: [...trail, key],
        message: `Record must not contain inline secret field "${key}".`,
      });
    }
    addInlineSecretIssues(child, ctx, [...trail, key]);
  }
}

function addInlineUiPayloadIssues(
  value: unknown,
  ctx: z.RefinementCtx,
  trail: Array<string | number> = [],
): void {
  if (Array.isArray(value)) {
    value.forEach((item, index) => addInlineUiPayloadIssues(item, ctx, [...trail, index]));
    return;
  }
  if (!value || typeof value !== "object") return;

  for (const [key, child] of Object.entries(value)) {
    const normalizedKey = key.toLowerCase().replace(/[^a-z0-9]/g, "");
    if (FORBIDDEN_INLINE_UI_PAYLOAD_KEYS.has(normalizedKey)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: [...trail, key],
        message: `GUI contribution must not contain inline executable/render field "${key}".`,
      });
    }
    addInlineUiPayloadIssues(child, ctx, [...trail, key]);
  }
}

interface PlanResolution<T> {
  item?: T;
  requirement: AgentCompositionRequirement;
}

function missingRequirement<T>(
  kind: AgentCompositionRequirementKind,
  message: string,
  id?: string,
): PlanResolution<T> {
  return {
    requirement: {
      kind,
      status: "missing",
      id,
      message,
    },
  };
}

function resolveForPlan<T extends { id: string; enabled?: boolean }>(
  items: T[],
  id: string,
  label: string,
  kind: AgentCompositionRequirementKind,
): PlanResolution<T> {
  const matches = items.filter((item) => item.id === id);
  if (matches.length === 0) {
    return missingRequirement(kind, `Unknown ${label} id "${id}".`, id);
  }
  if (matches.length > 1) {
    return {
      item: matches[0],
      requirement: {
        kind,
        status: "ambiguous",
        id,
        message: `Ambiguous ${label} id "${id}".`,
      },
    };
  }

  const item = matches[0];
  if (item.enabled === false) {
    return {
      item,
      requirement: {
        kind,
        status: "disabled",
        id,
        message: `${label[0]?.toUpperCase() ?? "I"}${label.slice(1)} "${id}" is disabled.`,
      },
    };
  }

  return {
    item,
    requirement: {
      kind,
      status: "ok",
      id,
      message: `Resolved ${label} "${id}".`,
    },
  };
}

function resolveCapabilityRefForPlan<T extends { id: string; name?: string; enabled?: boolean }>(
  kind: "skill" | "mcp_server",
  id: string,
  items: T[],
  inventory: ExtensionInventory,
): { ref: AgentCompositionCapabilityRef; requirement: AgentCompositionRequirement } {
  const label = kind === "skill" ? "skill" : "MCP server";
  const sourcePluginId = sourcePluginIdForCapability(kind, id, inventory);
  const resolution = resolveForPlan(items, id, label, kind);
  const status = resolution.requirement.status;
  return {
    ref: {
      id,
      name: resolution.item?.name,
      sourcePluginId,
      status,
    },
    requirement: {
      ...resolution.requirement,
      sourcePluginId,
    },
  };
}

function sourcePluginIdForCapability(
  kind: "skill" | "mcp_server",
  id: string,
  inventory: ExtensionInventory,
): string | undefined {
  const capabilityKey = kind === "skill" ? "skills" : "mcpServers";
  return inventory.bundles.find((bundle) =>
    bundle.capabilities[capabilityKey].some((item) => item.id === id),
  )?.id;
}

function pluginRequirementForPlan(
  pluginId: string,
  inventory: ExtensionInventory,
): AgentCompositionRequirement {
  const bundles = inventory.bundles.filter((bundle) => bundle.id === pluginId);
  const catalogEntries = inventory.pluginCatalog.filter(
    (entry) => entry.id === pluginId || entry.bundleId === pluginId,
  );
  if (bundles.length === 0 && catalogEntries.length === 0) {
    return {
      kind: "plugin",
      status: "missing",
      id: pluginId,
      message: `Unknown plugin id "${pluginId}".`,
    };
  }
  if (bundles.length > 1 || catalogEntries.length > 1) {
    return {
      kind: "plugin",
      status: "ambiguous",
      id: pluginId,
      message: `Ambiguous plugin id "${pluginId}".`,
    };
  }

  const bundle = bundles[0];
  const catalogEntry = catalogEntries[0];
  const marketplaceSource = catalogEntry?.marketplaceSourceId
    ? inventory.marketplaceSources.find((source) => source.id === catalogEntry.marketplaceSourceId)
    : undefined;
  if (
    bundle?.enabled === false ||
    catalogEntry?.installState === "disabled" ||
    marketplaceSource?.enabled === false
  ) {
    return {
      kind: "plugin",
      status: "disabled",
      id: pluginId,
      message: `Plugin "${pluginId}" is disabled.`,
    };
  }
  if (catalogEntry?.installState === "blocked") {
    return {
      kind: "plugin",
      status: "blocked",
      id: pluginId,
      message: `Plugin "${pluginId}" is blocked.`,
    };
  }

  return {
    kind: "plugin",
    status: "ok",
    id: pluginId,
    message: `Resolved plugin "${pluginId}".`,
  };
}

function contextSummaryForPlan(
  composition: AgentComposition,
  profile: AgentProfile | undefined,
  requirements: AgentCompositionRequirement[],
): AgentCompositionContextSummary | undefined {
  const input = composition.context ?? profile?.context;
  const rawMode = input?.mode ?? "thread_packet";
  const rawStrategy = input?.strategy ?? "auto";
  const mode = ContextPacketModeSchema.safeParse(rawMode);
  const strategy = ContextStrategySchema.safeParse(rawStrategy);

  if (!mode.success) {
    requirements.push({
      kind: "context",
      status: "unsupported",
      id: rawMode,
      message: `Unsupported context mode "${rawMode}".`,
    });
  }
  if (!strategy.success) {
    requirements.push({
      kind: "context",
      status: "unsupported",
      id: rawStrategy,
      message: `Unsupported context strategy "${rawStrategy}".`,
    });
  }
  if (!mode.success || !strategy.success) return undefined;

  const context = {
    mode: mode.data,
    strategy: strategy.data,
    memory: composition.memory ?? profile?.memory,
  };
  requirements.push({
    kind: "context",
    status: "ok",
    id: `${context.mode}:${context.strategy}`,
    message: `Resolved context ${context.mode}/${context.strategy}.`,
  });
  return AgentCompositionContextSummarySchema.parse(context);
}

function permissionSummaryForPlan(
  composition: AgentComposition,
  profile: AgentProfile | undefined,
  selectedMcpIds: string[],
): AgentCompositionPermissionSummary {
  const input = composition.permissions ?? profile?.permissions;
  const mode = input?.mode ?? profile?.permissionMode;
  const tools =
    input?.tools ??
    (profile?.tools && profile.tools.length > 0
      ? `${profile.tools.length} allowed tools`
      : "inherit");
  const mcp = input?.mcp ?? (selectedMcpIds.length > 0 ? "selected" : "none");
  const shell = input?.shell ?? mode ?? "harness-policy";
  return AgentCompositionPermissionSummarySchema.parse({
    tools,
    mcp,
    shell,
    mode,
    summary: [mode ? `mode ${mode}` : undefined, `${mcp} MCP`, shell].filter(Boolean).join(" / "),
  });
}

function definitionSummaryForPlan(
  definition: AgentProfile["definition"] | AgentComposition["definition"],
  profile: AgentProfile | undefined,
): AgentCompositionDefinitionSummary {
  if (definition) {
    return AgentCompositionDefinitionSummarySchema.parse({
      source: definition.kind,
      path: definition.path,
      pluginId: definition.pluginId,
      label: definition.label,
      host: definition.host,
      format: definition.format,
      readOnly: definition.readOnly,
    });
  }
  return AgentCompositionDefinitionSummarySchema.parse({
    source: profile?.instructions ? "inline" : "none",
  });
}

function visualSummaryForPlan(
  composition: AgentComposition,
  profile: AgentProfile | undefined,
  displayName: string,
): AgentCompositionVisualSummary {
  const visual = composition.visual ?? profile?.visual;
  return AgentCompositionVisualSummarySchema.parse({
    label: visual?.label ?? displayName,
    color: visual?.color ?? profile?.color,
    icon: visual?.icon,
  });
}

function compositionStatusForRequirements(
  requirements: AgentCompositionRequirement[],
): AgentCompositionStatus {
  if (
    requirements.some(
      (requirement) => requirement.kind === "agent_profile" && requirement.status === "disabled",
    )
  ) {
    return "disabled";
  }
  if (
    requirements.some((requirement) =>
      ["ambiguous", "disabled", "blocked", "unsupported", "unavailable"].includes(
        requirement.status,
      ),
    )
  ) {
    return "blocked";
  }
  if (
    requirements.some(
      (requirement) => requirement.status === "missing" && requirement.id !== undefined,
    )
  ) {
    return "blocked";
  }
  if (requirements.some((requirement) => requirement.status === "missing")) return "draft";
  return "ready";
}

function assertPlanReadyForExecution(plan: AgentCompositionPlan): void {
  if (plan.status === "ready") return;
  const details = plan.requirements
    .filter((requirement) => requirement.status !== "ok" && requirement.status !== "unknown")
    .map((requirement) => requirement.message)
    .join("; ");
  throw new Error(
    `Agent composition "${plan.id}" is ${plan.status}: ${details || "not ready for execution"}`,
  );
}

/** Select hidden runtime routing only; Provider never changes Model compatibility or identity. */
function selectInternalModelSupply(
  inventory: ExtensionInventory,
  modelId: string,
  harnessId?: string,
): ModelSupplyCapability | undefined {
  const harness = harnessId
    ? inventory.harnesses.find((candidate) => candidate.id === harnessId)
    : undefined;
  const candidates = inventory.modelSupplies.filter(
    (supply) =>
      supply.modelId === modelId &&
      supply.enabled !== false &&
      (!harness || modelSupplySupportsHarness(supply, harness)),
  );
  const providerRank = (supply: ModelSupplyCapability): number => {
    const matches = inventory.providers.filter(
      (provider) => provider.id === supply.providerProfileId,
    );
    if (matches.length !== 1) return 4;
    const provider = matches[0];
    if (provider.enabled === false) return 3;
    const runtimeReady = (provider as { runtimeReady?: unknown }).runtimeReady;
    if (runtimeReady === true) return 0;
    if (runtimeReady === false) return 2;
    return 1;
  };
  return [...candidates].sort(
    (left, right) => providerRank(left) - providerRank(right) || left.id.localeCompare(right.id),
  )[0];
}

function modelRouteCapabilityRegistry(
  model: ModelProfile,
  supply: ModelSupplyCapability | undefined,
) {
  return modelCapabilityRegistry([
    supply?.reasoningCapabilities.length
      ? {
          ...model,
          reasoningCapabilities: [...model.reasoningCapabilities, ...supply.reasoningCapabilities],
        }
      : model,
  ]);
}

function resolveById<T extends { id: string }>(items: T[], id: string, label: string): T {
  const matches = items.filter((item) => item.id === id);
  if (matches.length === 1) return matches[0];
  if (matches.length > 1) throw new Error(`Ambiguous ${label} id "${id}".`);
  throw new Error(`Unknown ${label} id "${id}".`);
}

function resolveOptionalById<T extends { id: string }>(
  items: T[],
  id: string | undefined,
  label: string,
): T | undefined {
  return id ? resolveById(items, id, label) : undefined;
}

function singleAgentSwarmConfig(agentConfig: AgentConfig): SwarmConfig {
  return {
    name: agentConfig.name,
    root: agentConfig.name,
    nodes: {
      [agentConfig.name]: { kind: "agent", agent: agentConfig },
    },
    edges: [],
  };
}

function agentConfigWithRuntimeEnv(
  agentConfig: AgentConfig,
  runtimeEnv: Record<string, string>,
  cwd?: string,
): AgentConfig {
  return {
    ...agentConfig,
    process: {
      ...agentConfig.process,
      ...(cwd ? { currentDir: cwd } : {}),
      clearEnv: true,
      env: {
        ...(agentConfig.process?.env ?? {}),
        ...runtimeEnv,
      },
    },
  };
}

function selectRuntimeEnv(
  source: NodeJS.ProcessEnv,
  allowedKeys: readonly string[],
): Record<string, string> {
  const selected: Record<string, string> = {};
  for (const key of allowedKeys) {
    const value = source[key];
    if (value !== undefined) selected[key] = value;
  }
  return selected;
}

function providerSecretValue(
  provider: ProviderProfile,
  env: NodeJS.ProcessEnv,
  providerSecrets: Readonly<Record<string, string>> | undefined,
): string | undefined {
  const ref = provider.secretRef;
  if (!ref) return undefined;
  const override = providerSecrets?.[provider.id];
  if (override) return override;
  if (ref.source !== "env") {
    throw new Error(
      `Provider profile "${provider.id}" uses unsupported secret source "${ref.source}" in this runtime.`,
    );
  }

  const value = env[ref.key];
  if (!value) {
    throw new Error(`Provider profile "${provider.id}" requires env secret "${ref.key}".`);
  }
  return value;
}

function skillHostIssue(input: {
  level?: SkillHostCompatibilityIssueLevel;
  code: string;
  message: string;
  skillId?: string;
  host?: MarketplaceHost;
  path?: string;
}): SkillHostCompatibilityIssue {
  return SkillHostCompatibilityIssueSchema.parse({
    level: input.level ?? "error",
    code: input.code,
    message: input.message,
    skillId: input.skillId,
    host: input.host,
    path: input.path,
  });
}

function isUnderAnyRoot(skillPath: string, canonicalRoots: string[]): boolean {
  const normalized = normalizeManifestPath(skillPath);
  return canonicalRoots.some((root) => normalized === root || normalized.startsWith(root));
}

function normalizeManifestPath(value: string): string {
  const normalized = value.replace(/\\/g, "/").replace(/^\.\//, "");
  return normalized.endsWith("/") ? normalized : `${normalized}/`;
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

function toAgentConfigName(value: string): string {
  const normalized = value.replace(/[^A-Za-z0-9_]+/g, "_").replace(/^_+|_+$/g, "");
  if (/^[A-Za-z][A-Za-z0-9_]*$/.test(normalized)) return normalized;
  return `agent_${normalized || "profile"}`;
}

function toCanonicalSelector(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) return "@agent";
  return trimmed.startsWith("@") ? trimmed : `@${trimmed}`;
}

function harnessDescriptor(harness: HarnessCapability): Record<string, unknown> {
  return {
    runtimeHarnessId: harness.runtimeHarnessId,
    software: harness.software,
    mcps: harness.mcps,
    skills: harness.skills,
    projectFiles: harness.projectFiles,
  };
}

function runtimeHarnessId(harness: HarnessCapability): string {
  return harness.runtimeHarnessId ?? harness.id;
}

function modelSupplySupportsHarness(
  supply: ModelSupplyCapability,
  harness: HarnessCapability,
): boolean {
  if (harness.modelControl === "direct") return true;
  const adapterId = runtimeHarnessId(harness);
  if (harness.requiresExplicitModelRoute) {
    return Boolean(supply.runtimeModel) && supply.harnessIds?.includes(adapterId) === true;
  }
  return !supply.harnessIds || supply.harnessIds.includes(adapterId);
}

function softwareFromBackend(id: string, backend: AgentBackend): SoftwareCapability {
  if (backend.type === "custom") {
    return {
      name: backend.program,
      runner: backend.program,
      command: backend.args,
    };
  }

  return { name: id };
}
