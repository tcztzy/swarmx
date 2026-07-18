import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import {
  AgentProfileSchema,
  type DesktopSettingsDocument,
  type ExtensionInventory,
  HarnessCapabilitySchema,
  type Model,
  type ModelApi,
  ModelApiSchema,
  type ModelCapability,
  type ModelSupply,
  type ProviderAuthMode,
  ProviderAuthModeSchema,
  type ProviderProfile,
  parseModel,
  parseModelSupply,
} from "@swarmx/core";
import type { CodexAccessTokenProvider } from "./codex-auth.js";
import type { ProviderAuthStore } from "./provider-auth.js";
import { newApiAccountCredentialKey } from "./provider-auth.js";
import { queryCodexAppServerRequest } from "./provider-usage.js";
import { DesktopSettingsStore, type DesktopSettingsStoreLike } from "./settings-store.js";

const DEFAULT_DISCOVERY_TIMEOUT_MS = 8_000;
const MAX_ANTHROPIC_PAGES = 20;
const MAX_CODEX_MODEL_PAGES = 20;
const CODEX_PROVIDER_ID = "swarmx.local.codex";
const CLAUDE_ACP_RUNTIME_MODELS = new Set([
  "claude-haiku-4-5-20251001",
  "claude-opus-4-1-20250805",
  "claude-opus-4-5-20251101",
  "claude-opus-4-6",
  "claude-opus-4-7",
  "claude-opus-4-8",
  "claude-sonnet-4-5-20250929",
  "claude-sonnet-4-6",
  "claude-sonnet-5",
  "deepseek-v4-flash",
  "deepseek-v4-pro",
]);
const CODEX_ACP_RUNTIME_MODELS = new Set([
  "gpt-5.2",
  "gpt-5.4",
  "gpt-5.4-mini",
  "gpt-5.5",
  "gpt-5.6-luna",
  "gpt-5.6-sol",
  "gpt-5.6-terra",
]);
const CODEX_PROVIDER: ProviderProfile = {
  id: CODEX_PROVIDER_ID,
  label: "Codex",
  kind: "openai_responses",
  apiMode: "codex_responses",
  baseUrl: "https://chatgpt.com/backend-api/codex",
  apiEntrypoints: {},
  authMode: "auth_token",
  secretRef: { source: "env", key: "CODEX_ACCESS_TOKEN" },
  readOnly: true,
  catalogAdapter: "codex_app_server",
};

export interface ManualModelInput {
  id: string;
  label?: string;
  runtimeModel?: string;
  apiProtocol: ModelApi;
}

export interface UserProviderInput {
  id?: string;
  label: string;
  kind: ModelApi;
  baseUrl: string;
  authMode: ProviderAuthMode;
  usageAdapter?: "new_api";
  secret?: string;
  accountAccessToken?: string;
  accountUserId?: string;
  clearAccountAccess?: boolean;
}

export interface ModelCatalogProviderStatus {
  providerProfileId: string;
  label: string;
  status: "cached" | "ready" | "skipped" | "error";
  modelCount: number;
  fetchedAt?: string;
  error?: string;
}

export interface ModelCatalogMetadata {
  manualModelIds: string[];
  userProviderIds: string[];
  providers: ModelCatalogProviderStatus[];
  refreshedAt?: string;
}

export type ModelCatalogInventory = ExtensionInventory & {
  modelCatalog: ModelCatalogMetadata;
};

export interface ModelCatalogServiceOptions {
  settingsPath?: string;
  cachePath?: string;
  env?: NodeJS.ProcessEnv;
  fetch?: CatalogFetch;
  now?: () => Date;
  timeoutMs?: number;
  authStore?: ProviderAuthStore;
  includeCodex?: boolean;
  codexCommand?: string;
  codexModelReader?: CodexModelReader;
  codexAccessTokenProvider?: CodexAccessTokenProvider;
  settingsStore?: DesktopSettingsStoreLike;
}

type CodexModelReader = (cursor?: string) => Promise<unknown>;

interface CatalogHttpResponse {
  ok: boolean;
  status: number;
  statusText: string;
  json(): Promise<unknown>;
}

type CatalogFetch = (
  url: string,
  init: { method: "GET"; headers: Record<string, string>; signal: AbortSignal },
) => Promise<CatalogHttpResponse>;

interface ProviderDiscoveryCache {
  providerProfileId: string;
  fetchedAt: string;
  models: Model[];
  modelSupplies: ModelSupply[];
}

interface ModelCatalogCacheDocument {
  schemaVersion: 1;
  refreshedAt?: string;
  discoveries: ProviderDiscoveryCache[];
}

interface DiscoveredModelDescriptor {
  id: string;
  label?: string;
  runtimeModel?: string;
  group?: string;
  reasoning?: {
    apiProtocol: ModelApi;
    supportedEfforts: string[];
    defaultEffort?: string;
    sourceUrl?: string;
    sourceVersion?: string;
  };
}

export class ModelCatalogService {
  private readonly settingsStore: DesktopSettingsStoreLike;
  private readonly cachePath: string;
  private readonly env: NodeJS.ProcessEnv;
  private readonly fetch: CatalogFetch;
  private readonly now: () => Date;
  private readonly timeoutMs: number;
  private readonly authStore?: ProviderAuthStore;
  private readonly includeCodex: boolean;
  private readonly codexModelReader: CodexModelReader;
  private readonly codexAccessTokenProvider?: CodexAccessTokenProvider;

  constructor(options: ModelCatalogServiceOptions = {}) {
    const root = join(homedir(), ".swarmx");
    this.settingsStore =
      options.settingsStore ?? new DesktopSettingsStore({ path: options.settingsPath });
    this.cachePath = options.cachePath ?? join(root, "model-catalog-cache.json");
    this.env = options.env ?? process.env;
    this.fetch = options.fetch ?? ((url, init) => fetch(url, init));
    this.now = options.now ?? (() => new Date());
    this.timeoutMs = options.timeoutMs ?? DEFAULT_DISCOVERY_TIMEOUT_MS;
    this.authStore = options.authStore;
    this.includeCodex = options.includeCodex ?? true;
    this.codexAccessTokenProvider = options.codexAccessTokenProvider;
    const codexCommand = options.codexCommand ?? "codex";
    this.codexModelReader =
      options.codexModelReader ??
      ((cursor) =>
        queryCodexAppServerRequest(
          "model/list",
          { cursor: cursor ?? null, limit: 100, includeHidden: false },
          codexCommand,
          this.timeoutMs,
        ));
  }

  async list(inventory: ExtensionInventory): Promise<ModelCatalogInventory> {
    const [settings, cache] = await Promise.all([this.readSettings(), this.readCache()]);
    const providers = this.discoveryProviders(inventory, settings);
    return await this.mergeInventory(
      inventory,
      settings,
      cache,
      providers,
      cachedStatuses(cache, providers),
    );
  }

  async refresh(inventory: ExtensionInventory): Promise<ModelCatalogInventory> {
    const [settings, previousCache] = await Promise.all([this.readSettings(), this.readCache()]);
    const providers = this.discoveryProviders(inventory, settings);
    const previousByProvider = new Map(
      previousCache.discoveries.map((entry) => [entry.providerProfileId, entry]),
    );

    const results = await Promise.all(
      providers.map(
        async (
          provider,
        ): Promise<{
          cache?: ProviderDiscoveryCache;
          status: ModelCatalogProviderStatus;
        }> => {
          const secretResolution =
            stringProperty(provider, "catalogAdapter") === "codex_app_server"
              ? ({ ready: true } as const)
              : await this.resolveSecret(provider);
          if (!secretResolution.ready) {
            const previous = previousByProvider.get(provider.id);
            return {
              status: {
                providerProfileId: provider.id,
                label: provider.label,
                status: "skipped",
                modelCount: previous?.models.length ?? 0,
                fetchedAt: previous?.fetchedAt,
                error: secretResolution.error,
              },
            };
          }

          try {
            const secret = "value" in secretResolution ? secretResolution.value : undefined;
            const descriptors = await this.discoverProviderModels(provider, secret);
            const fetchedAt = this.now().toISOString();
            const cache = providerCache(provider, descriptors, fetchedAt);
            return {
              cache,
              status: {
                providerProfileId: provider.id,
                label: provider.label,
                status: "ready",
                modelCount: cache.models.length,
                fetchedAt,
              },
            };
          } catch (error) {
            const previous = previousByProvider.get(provider.id);
            return {
              status: {
                providerProfileId: provider.id,
                label: provider.label,
                status: "error",
                modelCount: previous?.models.length ?? 0,
                fetchedAt: previous?.fetchedAt,
                error: error instanceof Error ? error.message : String(error),
              },
            };
          }
        },
      ),
    );

    for (const result of results) {
      if (result.cache) previousByProvider.set(result.cache.providerProfileId, result.cache);
    }
    const refreshedAt = this.now().toISOString();
    const cache: ModelCatalogCacheDocument = {
      schemaVersion: 1,
      refreshedAt,
      discoveries: [...previousByProvider.values()].filter((entry) =>
        providers.some((provider) => provider.id === entry.providerProfileId),
      ),
    };
    await writeJsonAtomic(this.cachePath, cache);
    return await this.mergeInventory(
      inventory,
      settings,
      cache,
      providers,
      results.map((result) => result.status),
    );
  }

  async addManualModel(
    inventory: ExtensionInventory,
    input: ManualModelInput,
  ): Promise<ModelCatalogInventory> {
    const model = manualModel(input);
    await this.settingsStore.update((settings) => ({
      ...settings,
      models: [...settings.models.filter((candidate) => candidate.id !== model.id), model],
    }));
    return this.list(inventory);
  }

  async removeManualModel(
    inventory: ExtensionInventory,
    modelId: string,
  ): Promise<ModelCatalogInventory> {
    const normalizedId = modelId.trim();
    if (!normalizedId) throw new Error("Manual Model id is required.");
    await this.settingsStore.update((settings) => ({
      ...settings,
      models: settings.models.filter((model) => model.id !== normalizedId),
    }));
    return this.list(inventory);
  }

  async saveProvider(
    inventory: ExtensionInventory,
    input: UserProviderInput,
  ): Promise<ModelCatalogInventory> {
    const settings = await this.readSettings();
    const requestedId = input.id?.trim();
    const existing = requestedId
      ? settings.providers.find((provider) => provider.id === requestedId)
      : undefined;
    if (requestedId && !existing) {
      throw new Error(`User-managed Provider "${requestedId}" was not found.`);
    }
    if (existing?.readOnly) {
      throw new Error(`Provider "${existing.id}" is read-only.`);
    }

    const provider = normalizeUserProviderInput(
      input,
      existing?.id ?? createUserProviderId(input.label, inventory, settings),
      existing?.metadata,
    );
    const authStore = this.requireAuthStore();
    const previousSecret = existing ? await authStore.get(existing.id) : undefined;
    const accountCredentialKey = newApiAccountCredentialKey(provider.id);
    const previousAccountAccessToken = existing
      ? await authStore.get(accountCredentialKey)
      : undefined;
    const secret = input.secret?.trim();
    const accountAccessToken = input.accountAccessToken?.trim();
    if (!secret && !previousSecret) {
      throw new Error(
        provider.authMode === "auth_token" ? "Auth token is required." : "API key is required.",
      );
    }
    if (
      provider.accountUserId &&
      !accountAccessToken &&
      !previousAccountAccessToken &&
      !input.clearAccountAccess
    ) {
      throw new Error("New API account access token is required when a User ID is configured.");
    }
    let primaryCredentialChanged = false;
    let accountCredentialChanged = false;
    try {
      if (secret) {
        await authStore.set(provider.id, secret);
        primaryCredentialChanged = true;
      }
      if (accountAccessToken) {
        await authStore.set(accountCredentialKey, accountAccessToken);
        accountCredentialChanged = true;
      } else if (input.clearAccountAccess || provider.usageAdapter !== "new_api") {
        await authStore.delete(accountCredentialKey);
        accountCredentialChanged = true;
      }
      await this.settingsStore.update((current) => ({
        ...current,
        providers: [
          ...current.providers.filter((candidate) => candidate.id !== provider.id),
          {
            id: provider.id,
            displayName: provider.label,
            kind: provider.kind,
            baseUrl: provider.baseUrl,
            apiEntrypoints: provider.apiEntrypoints,
            authMode: provider.authMode,
            secretRef: {
              source: "local_keychain",
              key: provider.id,
              service: "swarmx-provider",
              account: provider.id,
            },
            readOnly: false,
            metadata: userProviderMetadata(existing?.metadata, provider),
          },
        ],
      }));
    } catch (error) {
      if (primaryCredentialChanged) {
        try {
          if (previousSecret) await authStore.set(provider.id, previousSecret);
          else await authStore.delete(provider.id);
        } catch {
          // Preserve the original settings failure while attempting both restores.
        }
      }
      if (accountCredentialChanged) {
        try {
          if (previousAccountAccessToken) {
            await authStore.set(accountCredentialKey, previousAccountAccessToken);
          } else {
            await authStore.delete(accountCredentialKey);
          }
        } catch {
          // Preserve the original settings failure while attempting both restores.
        }
      }
      throw error;
    }

    return this.list(inventory);
  }

  async removeProvider(
    inventory: ExtensionInventory,
    providerId: string,
  ): Promise<ModelCatalogInventory> {
    const id = providerId.trim();
    if (!id) throw new Error("Provider id is required.");
    const [settings, cache] = await Promise.all([this.readSettings(), this.readCache()]);
    const provider = settings.providers.find((candidate) => candidate.id === id);
    if (!provider) throw new Error(`User-managed Provider "${id}" was not found.`);
    if (provider.readOnly) throw new Error(`Provider "${id}" is read-only.`);

    await this.settingsStore.update((current) => ({
      ...current,
      providers: current.providers.filter((candidate) => candidate.id !== id),
    }));
    const authStore = this.requireAuthStore();
    await authStore.delete(provider.secretRef?.key ?? id);
    await authStore.delete(newApiAccountCredentialKey(id));
    await this.writeCache({
      ...cache,
      discoveries: cache.discoveries.filter((entry) => entry.providerProfileId !== id),
    });
    return this.list(inventory);
  }

  async runtimeSecretsForSupply(
    inventory: ExtensionInventory,
    modelSupplyId: string,
  ): Promise<Record<string, string>> {
    const supply = inventory.modelSupplies.find((candidate) => candidate.id === modelSupplyId);
    if (!supply) throw new Error(`Unknown Model supply "${modelSupplyId}".`);
    const provider = inventory.providers.find(
      (candidate) => candidate.id === supply.providerProfileId,
    );
    if (!provider) {
      throw new Error(`Provider "${supply.providerProfileId}" is unavailable.`);
    }
    if (provider.apiMode === "codex_responses") {
      if (!this.codexAccessTokenProvider) {
        throw new Error(
          "Codex subscription authentication is unavailable. Open Codex and sign in again.",
        );
      }
      return { [provider.id]: await this.codexAccessTokenProvider.resolve() };
    }
    const resolution = await this.resolveSecret(provider);
    if (!resolution.ready) throw new Error(resolution.error);
    return resolution.value ? { [provider.id]: resolution.value } : {};
  }

  private async discoverProviderModels(
    provider: ProviderProfile,
    secret: string | undefined,
  ): Promise<DiscoveredModelDescriptor[]> {
    if (stringProperty(provider, "catalogAdapter") === "codex_app_server") {
      return this.discoverCodexModels();
    }
    const discoveryApi = providerDiscoveryApi(provider);
    if (discoveryApi === "ollama") {
      const payload = await this.fetchJson(
        discoveryUrl(provider),
        providerHeaders(provider, secret),
      );
      return parseOllamaModels(payload);
    }
    if (discoveryApi === "anthropic") {
      return this.discoverAnthropicModels(provider, secret);
    }
    const url = discoveryUrl(provider);
    const payload = await this.fetchJson(url, providerHeaders(provider, secret));
    return parseOpenAiModels(payload, discoveryApi, url);
  }

  private async discoverCodexModels(): Promise<DiscoveredModelDescriptor[]> {
    const models = new Map<string, DiscoveredModelDescriptor>();
    let cursor: string | undefined;
    for (let page = 0; page < MAX_CODEX_MODEL_PAGES; page += 1) {
      const parsed = parseCodexModels(await this.codexModelReader(cursor));
      for (const model of parsed.models) models.set(model.id, model);
      if (!parsed.nextCursor) return [...models.values()];
      if (parsed.nextCursor === cursor) {
        throw new Error("Codex app-server returned invalid model/list pagination.");
      }
      cursor = parsed.nextCursor;
    }
    throw new Error("Codex app-server model/list exceeded the pagination limit.");
  }

  private async discoverAnthropicModels(
    provider: ProviderProfile,
    secret: string | undefined,
  ): Promise<DiscoveredModelDescriptor[]> {
    const models = new Map<string, DiscoveredModelDescriptor>();
    let afterId: string | undefined;
    for (let page = 0; page < MAX_ANTHROPIC_PAGES; page += 1) {
      const url = new URL(discoveryUrl(provider));
      url.searchParams.set("limit", "1000");
      if (afterId) url.searchParams.set("after_id", afterId);
      const payload = await this.fetchJson(url.toString(), providerHeaders(provider, secret));
      const parsed = parseAnthropicModels(payload);
      for (const model of parsed.models) models.set(model.id, model);
      if (!parsed.hasMore) break;
      if (!parsed.lastId || parsed.lastId === afterId) {
        throw new Error(`Provider "${provider.id}" returned invalid Models API pagination.`);
      }
      afterId = parsed.lastId;
    }
    return [...models.values()];
  }

  private async fetchJson(url: string, headers: Record<string, string>): Promise<unknown> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await this.fetch(url, {
        method: "GET",
        headers,
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(`Model discovery request failed with HTTP ${response.status}.`);
      }
      return await response.json();
    } catch (error) {
      if (controller.signal.aborted) {
        throw new Error(`Model discovery request timed out after ${this.timeoutMs}ms.`);
      }
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  private discoveryProviders(
    inventory: ExtensionInventory,
    settings: DesktopSettingsDocument,
  ): ProviderProfile[] {
    const configured = settings.providers.map((provider) => {
      const usageAdapter = newApiUsageAdapter(provider.metadata.usageAdapter, false);
      return normalizePersistedProvider({
        id: provider.id,
        label: provider.displayName,
        kind: provider.kind,
        baseUrl: provider.baseUrl,
        apiEntrypoints: provider.apiEntrypoints,
        authMode: provider.authMode,
        secretRef: provider.secretRef,
        readOnly: provider.readOnly,
        modelDiscoveryUrl: stringMetadata(provider.metadata.modelDiscoveryUrl),
        modelDiscoveryApi: modelApiMetadata(provider.metadata.modelDiscoveryApi),
        newApiAccountUserId: stringMetadata(provider.metadata.newApiAccountUserId),
        ...(usageAdapter ? { usageAdapter } : {}),
      });
    });
    const providers = mergeProviders(inventory.providers, configured).filter(
      (provider) => provider.enabled !== false,
    );
    if (
      this.includeCodex &&
      inventory.harnesses.some(
        (harness) =>
          (harness.id === "codex" || harness.id === "swarmx") && harness.enabled !== false,
      )
    ) {
      return mergeProviders(providers, [CODEX_PROVIDER]);
    }
    return providers;
  }

  private async mergeInventory(
    inventory: ExtensionInventory,
    settings: DesktopSettingsDocument,
    cache: ModelCatalogCacheDocument,
    discoveryProviders: ProviderProfile[],
    statuses: ModelCatalogProviderStatus[],
  ): Promise<ModelCatalogInventory> {
    const discoveredModels = cache.discoveries.flatMap((entry) => entry.models);
    const providerById = new Map(
      mergeProviders(inventory.providers, discoveryProviders).map((provider) => [
        provider.id,
        provider,
      ]),
    );
    const discoveredSupplies = cache.discoveries.flatMap((entry) =>
      entry.modelSupplies.map((supply) => {
        const provider = providerById.get(supply.providerProfileId);
        const harnessIds = catalogHarnessIds(
          provider,
          supply.apiCompatibility.targetApi ?? provider?.kind,
          supply.runtimeModel,
        );
        const { harnessIds: _cachedHarnessIds, ...cachedSupply } = supply;
        return parseModelSupply({ ...cachedSupply, ...(harnessIds ? { harnessIds } : {}) });
      }),
    );
    const configuredModelIds = new Set([
      ...discoveredModels.map((model) => model.id),
      ...settings.models.map((model) => model.id),
    ]);
    const declaredModels = inventory.models.filter(
      (model) =>
        stringProperty(model, "catalogSource") !== "builtin" || configuredModelIds.has(model.id),
    );
    const providers = await Promise.all(
      mergeProviders(inventory.providers, discoveryProviders).map((provider) =>
        this.providerWithReadiness(provider),
      ),
    );
    const customHarnesses = settings.agents.flatMap((agent) => {
      const recipe = agent.harnessRecipe;
      if (!recipe) return [];
      const software = inventory.harnesses.find((harness) => harness.id === recipe.softwareId);
      if (!software) return [];
      return [
        HarnessCapabilitySchema.parse({
          ...software,
          id: recipe.id,
          runtimeHarnessId: recipe.softwareId,
          label: recipe.name ?? `${agent.name} Harness`,
          skills: recipe.skillBindings
            .filter((binding) => binding.mode !== "off")
            .map((binding) => binding.skillId),
          mcps: recipe.mcpServerIds,
          projectFiles: [...recipe.projectContext.paths, ...recipe.projectContext.instructionFiles],
          readOnly: false,
          source: { type: "path", path: "~/.swarmx/settings.json" },
        }),
      ];
    });
    return {
      ...inventory,
      harnesses: [
        ...inventory.harnesses,
        ...customHarnesses.filter(
          (harness) => !inventory.harnesses.some((candidate) => candidate.id === harness.id),
        ),
      ],
      models: mergeModels(declaredModels, discoveredModels, settings.models),
      modelSupplies: mergeSupplies(inventory.modelSupplies, discoveredSupplies),
      providers,
      agents: [
        ...inventory.agents,
        ...settings.agents
          .filter((agent) => !inventory.agents.some((candidate) => candidate.id === agent.id))
          .map((agent) => {
            const { source, ...profile } = agent;
            return AgentProfileSchema.parse({
              ...profile,
              definition: source,
              ...(agent.harnessRecipe
                ? {
                    permissions: {
                      mode: agent.harnessRecipe.permissions.mode,
                      allowedTools: agent.harnessRecipe.permissions.allowedTools,
                      deniedTools: agent.harnessRecipe.permissions.deniedTools,
                    },
                  }
                : {}),
            });
          }),
      ],
      modelCatalog: {
        manualModelIds: settings.models.map((model) => model.id),
        userProviderIds: settings.providers
          .filter((provider) => provider.readOnly !== true)
          .map((provider) => provider.id),
        providers: statuses,
        refreshedAt: cache.refreshedAt,
      },
    };
  }

  private async readSettings(): Promise<DesktopSettingsDocument> {
    return this.settingsStore.read();
  }

  private async readCache(): Promise<ModelCatalogCacheDocument> {
    const input = await readJsonIfPresent(this.cachePath);
    return parseCache(input);
  }

  private writeCache(cache: ModelCatalogCacheDocument): Promise<void> {
    return writeJsonAtomic(this.cachePath, cache);
  }

  private async resolveSecret(
    provider: ProviderProfile,
  ): Promise<{ ready: true; value?: string } | { ready: false; error: string }> {
    if (!provider.secretRef) return { ready: true };
    if (provider.secretRef.source === "env") {
      const value = this.env[provider.secretRef.key];
      return value
        ? { ready: true, value }
        : {
            ready: false,
            error: `Environment secret ${provider.secretRef.key} is not set.`,
          };
    }
    if (provider.secretRef.source === "local_keychain") {
      if (!this.authStore) {
        return { ready: false, error: "Secure Provider credential storage is unavailable." };
      }
      try {
        const value = await this.authStore.get(provider.secretRef.key);
        return value
          ? { ready: true, value }
          : { ready: false, error: `Provider "${provider.id}" credential is not configured.` };
      } catch (error) {
        return {
          ready: false,
          error: error instanceof Error ? error.message : "Provider credential could not be read.",
        };
      }
    }
    return {
      ready: false,
      error: `Provider "${provider.id}" uses an unsupported credential source.`,
    };
  }

  private async providerWithReadiness(provider: ProviderProfile): Promise<ProviderProfile> {
    if (provider.apiMode === "codex_responses") {
      const ready = (await this.codexAccessTokenProvider?.available()) ?? false;
      return {
        ...provider,
        runtimeReady: ready,
        runtimeNote: ready
          ? undefined
          : "Codex sign-in is unavailable. Open Codex and sign in again.",
      };
    }
    const resolution = await this.resolveSecret(provider);
    const accountUserId = stringProperty(provider, "newApiAccountUserId");
    let accountAccessReady: boolean | undefined;
    if (stringProperty(provider, "usageAdapter") === "new_api" && accountUserId) {
      try {
        accountAccessReady = !!(await this.authStore?.get(newApiAccountCredentialKey(provider.id)));
      } catch {
        accountAccessReady = false;
      }
    }
    return {
      ...provider,
      runtimeReady: resolution.ready,
      ...(accountAccessReady === undefined ? {} : { accountAccessReady }),
      ...(!resolution.ready ? { runtimeNote: resolution.error } : { runtimeNote: undefined }),
    };
  }

  private requireAuthStore(): ProviderAuthStore {
    if (!this.authStore) {
      throw new Error("Secure Provider credential storage is unavailable.");
    }
    return this.authStore;
  }
}

function providerHeaders(provider: ProviderProfile, secret: string | undefined) {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (providerDiscoveryApi(provider) === "anthropic") {
    headers["anthropic-version"] = "2023-06-01";
    if (secret) {
      if (provider.authMode === "auth_token") headers.Authorization = `Bearer ${secret}`;
      else headers["x-api-key"] = secret;
    }
  } else if (secret) {
    headers.Authorization = `Bearer ${secret}`;
  }
  return headers;
}

function discoveryUrl(provider: ProviderProfile): string {
  const explicit = stringProperty(provider, "modelDiscoveryUrl");
  if (explicit) return explicit;
  const discoveryApi = providerDiscoveryApi(provider);
  const baseUrl = (
    provider.baseUrl ??
    (discoveryApi === "anthropic"
      ? "https://api.anthropic.com"
      : discoveryApi === "ollama"
        ? "http://127.0.0.1:11434"
        : "https://api.openai.com/v1")
  ).replace(/\/$/, "");
  if (discoveryApi === "ollama") {
    return baseUrl.endsWith("/api") ? `${baseUrl}/tags` : `${baseUrl}/api/tags`;
  }
  if (discoveryApi === "anthropic") {
    return baseUrl.endsWith("/v1") ? `${baseUrl}/models` : `${baseUrl}/v1/models`;
  }
  return baseUrl.endsWith("/v1") ? `${baseUrl}/models` : `${baseUrl}/v1/models`;
}

function providerDiscoveryApi(provider: ProviderProfile): ModelApi {
  return modelApiMetadata(stringProperty(provider, "modelDiscoveryApi")) ?? provider.kind;
}

function parseOpenAiModels(
  input: unknown,
  apiProtocol: ModelApi = "openai_chat",
  sourceUrl?: string,
): DiscoveredModelDescriptor[] {
  const record = asRecord(input, "OpenAI-compatible Models API response");
  if (!Array.isArray(record.data)) {
    throw new Error("OpenAI-compatible Models API response must contain data[].");
  }
  return uniqueDescriptors(
    record.data.flatMap((item) => {
      const model = optionalRecord(item);
      if (!model || typeof model.id !== "string") return [];
      const supportedEfforts = stringArray(
        model.supported_reasoning_efforts ?? model.supportedReasoningEfforts,
      );
      const defaultEffort = stringMetadata(
        model.default_reasoning_effort ?? model.defaultReasoningEffort,
      );
      return [
        {
          id: model.id,
          label: stringMetadata(model.display_name ?? model.displayName),
          group: stringMetadata(model.owned_by ?? model.group),
          ...(supportedEfforts.length > 0
            ? {
                reasoning: {
                  apiProtocol,
                  supportedEfforts,
                  ...(defaultEffort ? { defaultEffort } : {}),
                  ...(sourceUrl ? { sourceUrl } : {}),
                  sourceVersion: "Provider Models API advertised reasoning efforts",
                },
              }
            : {}),
        },
      ];
    }),
  );
}

function parseCodexModels(input: unknown): {
  models: DiscoveredModelDescriptor[];
  nextCursor?: string;
} {
  const record = asRecord(input, "Codex app-server model/list response");
  if (!Array.isArray(record.data)) {
    throw new Error("Codex app-server model/list response must contain data[].");
  }
  return {
    models: uniqueDescriptors(
      record.data.flatMap((item) => {
        const model = optionalRecord(item);
        if (!model || typeof model.id !== "string" || model.hidden === true) return [];
        const supportedEfforts = Array.isArray(model.supportedReasoningEfforts)
          ? uniqueStrings(
              model.supportedReasoningEfforts.flatMap((inputEffort) => {
                const effort = optionalRecord(inputEffort);
                return effort &&
                  typeof effort.reasoningEffort === "string" &&
                  effort.reasoningEffort.toLowerCase() !== "ultra"
                  ? [effort.reasoningEffort]
                  : [];
              }),
            )
          : [];
        const defaultEffort = stringMetadata(model.defaultReasoningEffort);
        return [
          {
            id: model.id,
            runtimeModel: stringMetadata(model.model) ?? model.id,
            label: stringMetadata(model.displayName),
            ...(supportedEfforts.length > 0
              ? {
                  reasoning: {
                    apiProtocol: "openai_responses" as const,
                    supportedEfforts,
                    ...(defaultEffort ? { defaultEffort } : {}),
                    sourceUrl: "https://developers.openai.com/codex/app-server/#list-models",
                    sourceVersion: "Codex app-server model/list response",
                  },
                }
              : {}),
          },
        ];
      }),
    ),
    nextCursor: stringMetadata(record.nextCursor),
  };
}

function parseAnthropicModels(input: unknown): {
  models: DiscoveredModelDescriptor[];
  hasMore: boolean;
  lastId?: string;
} {
  const record = asRecord(input, "Anthropic Models API response");
  if (!Array.isArray(record.data)) {
    throw new Error("Anthropic Models API response must contain data[].");
  }
  return {
    models: uniqueDescriptors(
      record.data.flatMap((item) => {
        const model = optionalRecord(item);
        if (!model || typeof model.id !== "string") return [];
        const effort = optionalRecord(optionalRecord(model.capabilities)?.effort);
        const supportedEfforts = effort?.supported
          ? ["low", "medium", "high", "xhigh", "max"].filter((level) => {
              const support = optionalRecord(effort[level]);
              return support?.supported === true;
            })
          : [];
        return [
          {
            id: model.id,
            label: typeof model.display_name === "string" ? model.display_name : undefined,
            ...(supportedEfforts.length > 0
              ? {
                  reasoning: {
                    apiProtocol: "anthropic" as const,
                    supportedEfforts,
                    ...(supportedEfforts.includes("high") ? { defaultEffort: "high" } : {}),
                    sourceVersion: "Anthropic Models API capability response",
                  },
                }
              : {}),
          },
        ];
      }),
    ),
    hasMore: record.has_more === true,
    lastId: typeof record.last_id === "string" ? record.last_id : undefined,
  };
}

function parseOllamaModels(input: unknown): DiscoveredModelDescriptor[] {
  const record = asRecord(input, "Ollama tags response");
  if (!Array.isArray(record.models)) {
    throw new Error("Ollama tags response must contain models[].");
  }
  return uniqueDescriptors(
    record.models.flatMap((item) => {
      const model = optionalRecord(item);
      const id =
        model && typeof model.name === "string"
          ? model.name
          : model && typeof model.model === "string"
            ? model.model
            : undefined;
      return id ? [{ id }] : [];
    }),
  );
}

function providerCache(
  provider: ProviderProfile,
  descriptors: DiscoveredModelDescriptor[],
  fetchedAt: string,
): ProviderDiscoveryCache {
  const apiProtocols = providerApiProtocols(provider);
  const models = mergeModels(
    descriptors.map((descriptor) => {
      const humanLabel = humanReadableModelLabel(descriptor.id);
      return parseModel({
        id: descriptor.id,
        label: humanLabel === descriptor.id ? (descriptor.label ?? descriptor.id) : humanLabel,
        runtimeModel: descriptor.runtimeModel ?? descriptor.id,
        apiProtocols,
        capabilityIds: [],
        reasoningCapabilities: [],
        readOnly: true,
        catalogSource: "provider",
        discoveredFrom: [provider.id],
      });
    }),
  );
  return {
    providerProfileId: provider.id,
    fetchedAt,
    models,
    modelSupplies: descriptors.flatMap((descriptor) =>
      apiProtocols.map((apiProtocol) => {
        const harnessIds = catalogHarnessIds(
          provider,
          apiProtocol,
          descriptor.runtimeModel ?? descriptor.id,
        );
        return parseModelSupply({
          id:
            `catalog:${encodeURIComponent(provider.id)}:${encodeURIComponent(descriptor.id)}` +
            `${apiProtocols.length === 1 ? "" : `:${apiProtocol}`}` +
            `${descriptor.group ? `:${encodeURIComponent(descriptor.group)}` : ""}`,
          modelId: descriptor.id,
          providerProfileId: provider.id,
          runtimeModel: descriptor.runtimeModel ?? descriptor.id,
          apiCompatibility: { mode: "native", targetApi: apiProtocol },
          ...(descriptor.group ? { providerGroup: descriptor.group } : {}),
          ...(harnessIds ? { harnessIds } : {}),
          ...(descriptor.reasoning
            ? {
                reasoningCapabilities: [
                  discoveredReasoningCapability(provider, descriptor, fetchedAt),
                ],
              }
            : {}),
          readOnly: true,
        });
      }),
    ),
  };
}

function catalogHarnessIds(
  provider: ProviderProfile | undefined,
  apiProtocol: ModelApi | undefined,
  runtimeModel: string | undefined,
): string[] | undefined {
  if (provider && stringProperty(provider, "catalogAdapter") === "codex_app_server") {
    return runtimeModel && CODEX_ACP_RUNTIME_MODELS.has(runtimeModel)
      ? ["codex", "swarmx"]
      : ["swarmx"];
  }
  return apiProtocol === "anthropic" && runtimeModel && CLAUDE_ACP_RUNTIME_MODELS.has(runtimeModel)
    ? ["claude_code"]
    : undefined;
}

function discoveredReasoningCapability(
  provider: ProviderProfile,
  descriptor: DiscoveredModelDescriptor,
  fetchedAt: string,
): ModelCapability {
  const reasoning = descriptor.reasoning;
  if (!reasoning) throw new Error("Discovered reasoning metadata is required.");
  const supportedEfforts = uniqueStrings(reasoning.supportedEfforts);
  const defaultEffort =
    reasoning.defaultEffort && supportedEfforts.includes(reasoning.defaultEffort)
      ? reasoning.defaultEffort
      : undefined;
  return {
    id: `catalog:${encodeURIComponent(provider.id)}:${encodeURIComponent(descriptor.id)}:${reasoning.apiProtocol}:effort`,
    apiProtocol: reasoning.apiProtocol,
    modelIds: [descriptor.id],
    reasoningControl: "effort_enum",
    supportedEfforts,
    ...(defaultEffort ? { defaultEffort } : {}),
    parameterMapping: reasoningParameterMapping(reasoning.apiProtocol),
    effortAliases: {},
    source: {
      url: reasoning.sourceUrl ?? discoveryUrl(provider),
      checkedAt: fetchedAt.slice(0, 10),
      applicability: `${provider.label} advertised effort values for ${descriptor.id}`,
      version: reasoning.sourceVersion ?? "Provider model capability response",
    },
  };
}

function reasoningParameterMapping(apiProtocol: ModelApi): { api: string; path: string } {
  switch (apiProtocol) {
    case "anthropic":
      return { api: "anthropic.messages", path: "output_config.effort" };
    case "openai_responses":
      return { api: "openai.responses", path: "reasoning.effort" };
    case "openai_chat":
      return { api: "openai.chat.completions", path: "reasoning_effort" };
    default:
      throw new Error(`Provider-advertised effort is unsupported for ${apiProtocol}.`);
  }
}

function providerApiProtocols(provider: ProviderProfile): ModelApi[] {
  const entrypoints = provider.apiEntrypoints ?? {};
  return uniqueStrings([
    provider.kind,
    ...ModelApiSchema.options.filter((api) => typeof entrypoints[api] === "string"),
  ]);
}

function normalizeUserProviderInput(
  input: UserProviderInput,
  id: string,
  existingMetadata?: Record<string, unknown>,
): {
  id: string;
  label: string;
  kind: ModelApi;
  baseUrl: string;
  apiEntrypoints: Partial<Record<ModelApi, string>>;
  authMode: ProviderAuthMode;
  usageAdapter?: "new_api";
  accountUserId?: string;
  modelDiscoveryUrl?: string;
  modelDiscoveryApi?: ModelApi;
} {
  const label = input.label.trim();
  if (!label) throw new Error("Provider name is required.");
  const baseUrl = input.baseUrl.trim().replace(/\/$/, "");
  let parsedUrl: URL;
  try {
    parsedUrl = new URL(baseUrl);
  } catch {
    throw new Error("Base URL must be a valid URL.");
  }
  if (!["http:", "https:"].includes(parsedUrl.protocol)) {
    throw new Error("Base URL must use http or https.");
  }
  if (parsedUrl.username || parsedUrl.password) {
    throw new Error("Base URL must not contain credentials.");
  }
  const kind = ModelApiSchema.parse(input.kind);
  const usageAdapter = newApiUsageAdapter(input.usageAdapter);
  const accountUserId = normalizeNewApiAccountUserId(
    input.clearAccountAccess
      ? undefined
      : input.accountUserId === undefined
        ? stringMetadata(existingMetadata?.newApiAccountUserId)
        : input.accountUserId,
  );
  const accountAccessToken = input.accountAccessToken?.trim();
  if (input.clearAccountAccess && accountAccessToken) {
    throw new Error("New API account access cannot be cleared and replaced at the same time.");
  }
  if (input.accountAccessToken?.trim() && usageAdapter !== "new_api") {
    throw new Error("New API account access requires the New API Usage API selection.");
  }
  if (input.accountUserId !== undefined && accountUserId && usageAdapter !== "new_api") {
    throw new Error("New API account access requires the New API Usage API selection.");
  }
  if (accountAccessToken && !accountUserId) {
    throw new Error("New API User ID is required for an account access token.");
  }
  const deepSeekRouting = officialDeepSeekRouting(parsedUrl, kind);
  const modelDiscovery =
    deepSeekRouting ?? (usageAdapter === "new_api" ? newApiModelDiscovery(parsedUrl) : undefined);
  return {
    id,
    label,
    kind,
    baseUrl: deepSeekRouting?.baseUrl ?? baseUrl,
    apiEntrypoints: deepSeekRouting?.apiEntrypoints ?? {},
    authMode: ProviderAuthModeSchema.parse(input.authMode),
    usageAdapter,
    ...(usageAdapter === "new_api" && accountUserId ? { accountUserId } : {}),
    ...(modelDiscovery
      ? {
          modelDiscoveryUrl: modelDiscovery.modelDiscoveryUrl,
          modelDiscoveryApi: modelDiscovery.modelDiscoveryApi,
        }
      : {}),
  };
}

function userProviderMetadata(
  existing: Record<string, unknown> | undefined,
  provider: ReturnType<typeof normalizeUserProviderInput>,
): Record<string, unknown> {
  const {
    usageAdapter: _usageAdapter,
    newApiAccountUserId: _newApiAccountUserId,
    modelDiscoveryUrl: _modelDiscoveryUrl,
    modelDiscoveryApi: _modelDiscoveryApi,
    ...rest
  } = existing ?? {};
  return {
    ...rest,
    managedBy: "swarmx-desktop",
    ...(provider.usageAdapter ? { usageAdapter: provider.usageAdapter } : {}),
    ...(provider.accountUserId ? { newApiAccountUserId: provider.accountUserId } : {}),
    ...(provider.modelDiscoveryUrl ? { modelDiscoveryUrl: provider.modelDiscoveryUrl } : {}),
    ...(provider.modelDiscoveryApi ? { modelDiscoveryApi: provider.modelDiscoveryApi } : {}),
  };
}

function normalizeNewApiAccountUserId(value: unknown): string | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  if (typeof value !== "string" || !/^\d+$/.test(value.trim())) {
    throw new Error("New API User ID must be a positive integer.");
  }
  const normalized = value.trim().replace(/^0+(?=\d)/, "");
  if (normalized === "0") throw new Error("New API User ID must be a positive integer.");
  return normalized;
}

function officialDeepSeekRouting(
  baseUrl: URL,
  preferredApi: ModelApi,
):
  | {
      baseUrl: string;
      apiEntrypoints: Partial<Record<ModelApi, string>>;
      modelDiscoveryUrl: string;
      modelDiscoveryApi: ModelApi;
    }
  | undefined {
  if (
    baseUrl.protocol !== "https:" ||
    baseUrl.hostname.toLowerCase() !== "api.deepseek.com" ||
    !["anthropic", "openai_chat"].includes(preferredApi)
  ) {
    return undefined;
  }
  const origin = baseUrl.origin;
  const apiEntrypoints = {
    anthropic: `${origin}/anthropic`,
    openai_chat: origin,
  } satisfies Partial<Record<ModelApi, string>>;
  return {
    baseUrl: apiEntrypoints[preferredApi as "anthropic" | "openai_chat"],
    apiEntrypoints,
    modelDiscoveryUrl: `${origin}/models`,
    modelDiscoveryApi: "openai_chat",
  };
}

function normalizePersistedProvider(provider: ProviderProfile): ProviderProfile {
  if (!provider.baseUrl) return provider;
  let parsedUrl: URL;
  try {
    parsedUrl = new URL(provider.baseUrl);
  } catch {
    return provider;
  }
  const routing = officialDeepSeekRouting(parsedUrl, provider.kind);
  const modelDiscovery =
    routing ??
    (stringProperty(provider, "usageAdapter") === "new_api"
      ? newApiModelDiscovery(parsedUrl)
      : undefined);
  if (!routing && !modelDiscovery) return provider;
  return {
    ...provider,
    ...(routing
      ? {
          baseUrl: routing.baseUrl,
          apiEntrypoints: routing.apiEntrypoints,
        }
      : {}),
    modelDiscoveryUrl: modelDiscovery?.modelDiscoveryUrl,
    modelDiscoveryApi: modelDiscovery?.modelDiscoveryApi,
  };
}

function newApiModelDiscovery(baseUrl: URL): {
  modelDiscoveryUrl: string;
  modelDiscoveryApi: ModelApi;
} {
  return {
    modelDiscoveryUrl: `${baseUrl.origin}/v1/models`,
    modelDiscoveryApi: "openai_chat",
  };
}

export function humanReadableModelLabel(modelId: string): string {
  const rules: Array<{
    prefix: string;
    brand?: string;
    family?: RegExp;
  }> = [
    { prefix: "claude-", family: /^(?:fable|mythos|opus|sonnet|haiku)-/i },
    { prefix: "gpt-", brand: "GPT" },
    { prefix: "deepseek-", brand: "DeepSeek" },
    { prefix: "gemini-", brand: "Gemini" },
    { prefix: "kimi-", brand: "Kimi" },
    { prefix: "glm-", brand: "GLM" },
  ];
  const rule = rules.find((candidate) => modelId.toLowerCase().startsWith(candidate.prefix));
  if (!rule) return modelId;
  const tail = modelId.slice(rule.prefix.length);
  if (!tail || !/^[a-z0-9.-]+$/i.test(tail) || (rule.family && !rule.family.test(tail))) {
    return modelId;
  }
  const tokens = tail.split("-").filter(Boolean);
  if (rule.prefix === "claude-" && /^\d+$/.test(tokens[1] ?? "") && /^\d+$/.test(tokens[2] ?? "")) {
    tokens.splice(1, 2, `${tokens[1]}.${tokens[2]}`);
  }
  const words = tokens.map(humanizeModelToken);
  if (words.length === 0) return modelId;
  return [...(rule.brand ? [rule.brand] : []), ...words].join(" ");
}

function humanizeModelToken(token: string): string {
  if (/^v\d+(?:\.\d+)?$/i.test(token)) return token.toUpperCase();
  if (/^\d+(?:\.\d+)?$/.test(token)) return token;
  if (/^[a-z]+\d+$/i.test(token)) {
    const match = /^([a-z]+)(\d+)$/i.exec(token);
    if (match?.[1]?.toLowerCase() === "v") return token.toUpperCase();
  }
  return token.charAt(0).toUpperCase() + token.slice(1).toLowerCase();
}

function newApiUsageAdapter(value: unknown, strict = true): "new_api" | undefined {
  if (value === undefined || value === "" || value === "automatic") return undefined;
  if (value === "new_api") return value;
  if (!strict) return undefined;
  throw new Error("Unsupported Provider Usage API selection.");
}

function createUserProviderId(
  label: string,
  inventory: ExtensionInventory,
  settings: DesktopSettingsDocument,
): string {
  const slug =
    label
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "") || "provider";
  const occupied = new Set([
    ...inventory.providers.map((provider) => provider.id),
    ...settings.providers.map((provider) => provider.id),
  ]);
  const base = `swarmx.user.${slug}`;
  if (!occupied.has(base)) return base;
  for (let suffix = 2; suffix < 10_000; suffix += 1) {
    const candidate = `${base}-${suffix}`;
    if (!occupied.has(candidate)) return candidate;
  }
  throw new Error("Could not allocate a unique Provider id.");
}

function manualModel(input: ManualModelInput): Model {
  const id = input.id.trim();
  const runtimeModel = input.runtimeModel?.trim() || id;
  const label = input.label?.trim() || id;
  return parseModel({
    id,
    label,
    runtimeModel,
    apiProtocols: [input.apiProtocol],
    capabilityIds: [],
    readOnly: false,
    catalogSource: "manual",
  });
}

function mergeModels(...groups: readonly Model[][]): Model[] {
  const models = new Map<string, Model>();
  for (const model of groups.flat()) {
    const existing = models.get(model.id);
    if (!existing) {
      models.set(model.id, parseModel(model));
      continue;
    }
    const existingSources = stringArrayProperty(existing, "catalogSources");
    const incomingSources = stringArrayProperty(model, "catalogSources");
    const reasoningCapabilities = new Map<string, ModelCapability>();
    for (const capability of [...existing.reasoningCapabilities, ...model.reasoningCapabilities]) {
      reasoningCapabilities.set(capability.id, capability);
    }
    const catalogSources = uniqueStrings([
      ...existingSources,
      ...incomingSources,
      ...optionalStringProperty(existing, "catalogSource"),
      ...optionalStringProperty(model, "catalogSource"),
    ]);
    models.set(
      model.id,
      parseModel({
        ...existing,
        ...model,
        apiProtocols: uniqueStrings([...existing.apiProtocols, ...model.apiProtocols]),
        capabilityIds: uniqueStrings([
          ...(existing.capabilityIds ?? []),
          ...(model.capabilityIds ?? []),
        ]),
        reasoningCapabilities: [...reasoningCapabilities.values()],
        catalogSources,
        readOnly: existing.readOnly !== false && model.readOnly !== false,
      }),
    );
  }
  return [...models.values()];
}

function mergeSupplies(...groups: readonly ModelSupply[][]): ModelSupply[] {
  const supplies = new Map<string, ModelSupply>();
  for (const supply of groups.flat()) supplies.set(supply.id, parseModelSupply(supply));
  return [...supplies.values()];
}

function mergeProviders(...groups: readonly ProviderProfile[][]): ProviderProfile[] {
  const providers = new Map<string, ProviderProfile>();
  for (const provider of groups.flat()) {
    providers.set(provider.id, { ...providers.get(provider.id), ...provider });
  }
  return [...providers.values()];
}

function cachedStatuses(
  cache: ModelCatalogCacheDocument,
  providers: ProviderProfile[],
): ModelCatalogProviderStatus[] {
  const cached = new Map(cache.discoveries.map((entry) => [entry.providerProfileId, entry]));
  return providers.map((provider) => {
    const entry = cached.get(provider.id);
    return {
      providerProfileId: provider.id,
      label: provider.label,
      status: entry ? "cached" : "skipped",
      modelCount: entry?.models.length ?? 0,
      fetchedAt: entry?.fetchedAt,
    };
  });
}

function parseCache(input: unknown): ModelCatalogCacheDocument {
  if (input === undefined) return { schemaVersion: 1, discoveries: [] };
  const record = asRecord(input, "Model catalog cache");
  if (record.schemaVersion !== 1 || !Array.isArray(record.discoveries)) {
    throw new Error("Unsupported Model catalog cache format.");
  }
  return {
    schemaVersion: 1,
    refreshedAt: typeof record.refreshedAt === "string" ? record.refreshedAt : undefined,
    discoveries: record.discoveries.map((inputEntry) => {
      const entry = asRecord(inputEntry, "Model catalog provider cache");
      if (
        typeof entry.providerProfileId !== "string" ||
        typeof entry.fetchedAt !== "string" ||
        !Array.isArray(entry.models) ||
        !Array.isArray(entry.modelSupplies)
      ) {
        throw new Error("Invalid Model catalog provider cache.");
      }
      return {
        providerProfileId: entry.providerProfileId,
        fetchedAt: entry.fetchedAt,
        models: entry.models.map(parseModel),
        modelSupplies: entry.modelSupplies.map(parseModelSupply),
      };
    }),
  };
}

async function readJsonIfPresent(path: string): Promise<unknown | undefined> {
  try {
    return JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    if (isNodeError(error, "ENOENT")) return undefined;
    throw error;
  }
}

async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = `${path}.tmp-${process.pid}-${Date.now()}`;
  await writeFile(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, { mode: 0o600 });
  await rename(temporaryPath, path);
}

function asRecord(input: unknown, label: string): Record<string, unknown> {
  const record = optionalRecord(input);
  if (!record) throw new Error(`${label} must be an object.`);
  return record;
}

function optionalRecord(input: unknown): Record<string, unknown> | undefined {
  return input && typeof input === "object" && !Array.isArray(input)
    ? (input as Record<string, unknown>)
    : undefined;
}

function uniqueDescriptors(models: DiscoveredModelDescriptor[]): DiscoveredModelDescriptor[] {
  const byId = new Map<string, DiscoveredModelDescriptor>();
  for (const model of models) {
    const id = model.id.trim();
    if (id) byId.set(`${id}\u0000${model.group ?? ""}`, { ...model, id });
  }
  return [...byId.values()];
}

function uniqueStrings<T extends string>(values: readonly T[]): T[] {
  return [...new Set(values)];
}

function optionalStringProperty(record: object, key: string): string[] {
  const value = (record as Record<string, unknown>)[key];
  return typeof value === "string" ? [value] : [];
}

function stringArrayProperty(record: object, key: string): string[] {
  const value = (record as Record<string, unknown>)[key];
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function stringProperty(record: object, key: string): string | undefined {
  const value = (record as Record<string, unknown>)[key];
  return typeof value === "string" && value.trim() ? value : undefined;
}

function stringMetadata(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? uniqueStrings(
        value.flatMap((item) => (typeof item === "string" && item.trim() ? [item.trim()] : [])),
      )
    : [];
}

function modelApiMetadata(value: unknown): ModelApi | undefined {
  const parsed = ModelApiSchema.safeParse(value);
  return parsed.success ? parsed.data : undefined;
}

function isNodeError(error: unknown, code: string): boolean {
  return error instanceof Error && "code" in error && error.code === code;
}
