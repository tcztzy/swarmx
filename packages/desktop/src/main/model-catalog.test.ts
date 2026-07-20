import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  builtInExtensionBundle,
  createDefaultDesktopSettings,
  createExtensionInventory,
  parseExtensionBundle,
  resolveHarnessModelInventory,
} from "@swarmx/core";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ModelCatalogService, humanReadableModelLabel } from "./model-catalog.js";
import {
  type ProviderAuthStore,
  newApiAccountCredentialKey,
  providerPoolCredentialKey,
} from "./provider-auth.js";
import { ProviderKeyUsageStore } from "./provider-key-pool.js";

const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((root) => rm(root, { recursive: true })));
});

describe("ModelCatalogService", () => {
  it("projects persisted Harness recipes into runnable inventory Harnesses", async () => {
    const paths = await catalogPaths();
    await writeFile(
      paths.settingsPath,
      JSON.stringify(
        createDefaultDesktopSettings({
          agents: [
            {
              id: "researcher",
              name: "Researcher",
              harnessId: "researcher-harness",
              harnessRecipe: {
                id: "researcher-harness",
                revisionId: "researcher-harness@1",
                softwareId: "swarmx",
                skillBindings: [{ skillId: "paper-review", mode: "auto" }],
                mcpServerIds: ["project-fs"],
                projectContext: { paths: ["docs"] },
              },
              modelId: "gpt-5",
            },
          ],
        }),
      ),
      "utf8",
    );

    const catalog = await new ModelCatalogService({ ...paths, includeCodex: false }).list(
      createExtensionInventory([builtInExtensionBundle()]),
    );

    expect(catalog.harnesses).toContainEqual(
      expect.objectContaining({
        id: "researcher-harness",
        runtimeHarnessId: "swarmx",
        label: "Researcher Harness",
        skills: ["paper-review"],
        mcps: ["project-fs"],
        projectFiles: ["docs"],
      }),
    );
    expect(catalog.agents).toContainEqual(
      expect.objectContaining({
        id: "researcher",
        harnessId: "researcher-harness",
        modelId: "gpt-5",
      }),
    );
  });

  it("V270 ignores ambient Provider variables without explicit connections", async () => {
    const paths = await catalogPaths();
    const fetch = vi.fn(async (url: string) =>
      url.includes("anthropic")
        ? response({ data: [], has_more: false })
        : url.includes("127.0.0.1")
          ? response({ models: [] })
          : response({ data: [] }),
    );
    const service = new ModelCatalogService({
      ...paths,
      env: {
        OPENAI_API_KEY: "ambient-openai",
        ANTHROPIC_API_KEY: "ambient-anthropic",
        DEEPSEEK_API_KEY: "ambient-deepseek",
        OLLAMA_HOST: "http://127.0.0.1:11434",
      },
      fetch,
    });

    const inventory = createExtensionInventory([]);
    const listed = await service.list(inventory);
    const catalog = await service.refresh(inventory);

    expect(fetch).not.toHaveBeenCalled();
    expect(listed.providers).toEqual([]);
    expect(listed.models).toEqual([]);
    expect(catalog.providers).toEqual([]);
    expect(catalog.models).toEqual([]);
    expect(catalog.modelSupplies).toEqual([]);
    expect(catalog.modelCatalog.providers).toEqual([]);
  });

  it("discovers Models from an explicit OpenAI-compatible Provider", async () => {
    const paths = await catalogPaths();
    const inventory = createExtensionInventory([
      providerBundle("explicit-openai", [
        {
          id: "explicit-openai",
          label: "OpenAI",
          kind: "openai_chat",
          baseUrl: "https://api.openai.com/v1",
          secretRef: { source: "env", key: "OPENAI_API_KEY" },
          readOnly: true,
        },
      ]),
    ]);
    const fetch = vi
      .fn()
      .mockResolvedValue(response({ data: [{ id: "remote-gpt" }, { id: "remote-gpt" }] }));
    const service = new ModelCatalogService({
      ...paths,
      env: { OPENAI_API_KEY: "sk-runtime-only" },
      fetch,
      now: fixedClock(),
    });

    const catalog = await service.refresh(inventory);

    expect(fetch).toHaveBeenCalledWith(
      "https://api.openai.com/v1/models",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer sk-runtime-only" }),
      }),
    );
    expect(catalog.models).toEqual([
      expect.objectContaining({
        id: "remote-gpt",
        runtimeModel: "remote-gpt",
        apiProtocols: ["openai_chat"],
      }),
    ]);
    expect(catalog.modelSupplies).toEqual([
      expect.objectContaining({
        modelId: "remote-gpt",
        providerProfileId: "explicit-openai",
      }),
    ]);
    expect(catalog.modelCatalog.providers).toEqual([
      expect.objectContaining({ label: "OpenAI", status: "ready", modelCount: 1 }),
    ]);
    expect(catalog.providers).toContainEqual(
      expect.objectContaining({
        id: "explicit-openai",
        label: "OpenAI",
        readOnly: true,
      }),
    );
    expect(await readFile(paths.cachePath, "utf8")).not.toContain("sk-runtime-only");
  });

  it("V482 treats a Custom Provider as one exact base URL plus /models", async () => {
    const paths = await catalogPaths();
    const authStore = new MemoryProviderAuthStore();
    const fetch = vi.fn().mockResolvedValue(response({ data: [{ id: "custom-model" }] }));
    const service = new ModelCatalogService({
      ...paths,
      env: {},
      authStore,
      fetch,
      now: fixedClock(),
    });
    const inventory = createExtensionInventory([]);

    const saved = await service.saveProvider(inventory, {
      label: "Custom Gateway",
      kind: "openai_chat",
      baseUrl: "https://gateway.example.test/shared/v1",
      authMode: "api_key",
      secret: "custom-only-key",
    });
    const providerId = saved.modelCatalog.userProviderIds[0] as string;

    expect(saved.providers[0]).toEqual(
      expect.objectContaining({
        id: providerId,
        baseUrl: "https://gateway.example.test/shared/v1",
        apiEntrypoints: {},
      }),
    );
    expect(saved.providers[0]).not.toHaveProperty("usageAdapter");
    expect(saved.providers[0]?.newApiAccountUserId).toBeUndefined();

    const refreshed = await service.refresh(inventory);
    expect(fetch).toHaveBeenCalledWith(
      "https://gateway.example.test/shared/v1/models",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer custom-only-key" }),
      }),
    );
    expect(refreshed.models).toContainEqual(expect.objectContaining({ id: "custom-model" }));
    expect(await readFile(paths.settingsPath, "utf8")).not.toContain("custom-only-key");
  });

  it("V483/V484 persists an OpenCode Go key pool and exposes local per-key usage", async () => {
    const paths = await catalogPaths();
    const authStore = new MemoryProviderAuthStore();
    const keyUsageStore = new ProviderKeyUsageStore({
      path: paths.keyUsagePath,
      now: fixedClock(),
    });
    const fetch = vi.fn().mockResolvedValue(response({ data: [{ id: "go-model" }] }));
    const service = new ModelCatalogService({
      ...paths,
      env: {},
      authStore,
      keyUsageStore,
      fetch,
      now: fixedClock(),
    });
    const inventory = createExtensionInventory([]);

    const saved = await service.saveProvider(inventory, {
      label: "OpenCode Go",
      kind: "openai_chat",
      baseUrl: "https://opencode.ai/zen/go",
      authMode: "api_key",
      secret: "go-primary",
      additionalApiKeys: [{ value: "go-second" }, { label: "Backup", value: "go-third" }],
    });
    const provider = saved.providers[0] as (typeof saved.providers)[number] & {
      runtimeKeySlots: Array<{ id: string; label: string; enabled: boolean }>;
      runtimeKeyUsage: Array<{ id: string; status: string; totalTokens: number }>;
    };
    const providerId = provider.id;
    const extraSlots = provider.runtimeKeySlots.filter((slot) => slot.id !== "primary");

    expect(provider).toEqual(
      expect.objectContaining({
        baseUrl: "https://opencode.ai/zen/go/v1",
        apiEntrypoints: {
          anthropic: "https://opencode.ai/zen/go",
          openai_chat: "https://opencode.ai/zen/go/v1",
        },
        modelDiscoveryUrl: "https://opencode.ai/zen/go/v1/models",
      }),
    );
    expect(provider.runtimeKeySlots).toHaveLength(3);
    expect(provider.runtimeKeyUsage.map((entry) => entry.status)).toEqual([
      "ready",
      "ready",
      "ready",
    ]);
    expect(await authStore.get(providerId)).toBe("go-primary");
    expect(
      await authStore.get(providerPoolCredentialKey(providerId, extraSlots[0]?.id ?? "")),
    ).toBe("go-second");
    expect(
      await authStore.get(providerPoolCredentialKey(providerId, extraSlots[1]?.id ?? "")),
    ).toBe("go-third");

    const refreshed = await service.refresh(inventory);
    expect(fetch).toHaveBeenCalledWith("https://opencode.ai/zen/go/v1/models", expect.any(Object));
    expect(refreshed.models).toContainEqual(
      expect.objectContaining({ id: "go-model", apiProtocols: ["anthropic", "openai_chat"] }),
    );
    const supply = refreshed.modelSupplies.find(
      (candidate) => candidate.apiCompatibility.targetApi === "openai_chat",
    );
    await expect(
      service.runtimeCredentialsForSupply(refreshed, supply?.id ?? "missing"),
    ).resolves.toEqual({
      providerId,
      pooled: true,
      candidates: [
        { id: "primary", value: "go-primary" },
        { id: extraSlots[0]?.id, value: "go-second" },
        { id: extraSlots[1]?.id, value: "go-third" },
      ],
    });

    await keyUsageStore.recordSuccess(providerId, extraSlots[0]?.id ?? "", {
      inputTokens: 10,
      outputTokens: 4,
      reasoningTokens: 0,
      cachedInputTokens: 0,
      totalTokens: 14,
      estimated: false,
    });
    const withUsage = await service.list(inventory);
    expect(
      (withUsage.providers[0] as typeof provider).runtimeKeyUsage.find(
        (entry) => entry.id === extraSlots[0]?.id,
      ),
    ).toEqual(expect.objectContaining({ totalTokens: 14, status: "ready" }));

    const removed = await service.saveProvider(inventory, {
      id: providerId,
      label: "OpenCode Go",
      kind: "anthropic",
      baseUrl: "https://opencode.ai/zen/go/v1",
      authMode: "api_key",
      removeApiKeyIds: [extraSlots[0]?.id ?? ""],
    });
    expect((removed.providers[0] as typeof provider).runtimeKeySlots).toHaveLength(2);
    expect(
      await authStore.get(providerPoolCredentialKey(providerId, extraSlots[0]?.id ?? "")),
    ).toBeUndefined();
    const settings = await readFile(paths.settingsPath, "utf8");
    expect(settings).not.toContain("go-primary");
    expect(settings).not.toContain("go-second");
    expect(settings).not.toContain("go-third");
  });

  it("V332 migrates cached Anthropic supplies into explicit Claude Code routes", async () => {
    const paths = await catalogPaths();
    await writeFile(
      paths.cachePath,
      JSON.stringify({
        schemaVersion: 1,
        discoveries: [
          {
            providerProfileId: "anthropic-cache",
            fetchedAt: "2026-07-15T00:00:00.000Z",
            models: [
              {
                id: "claude-opus-4-6",
                runtimeModel: "claude-opus-4-6",
                apiProtocols: ["anthropic"],
              },
              {
                id: "claude-fable-5",
                runtimeModel: "claude-fable-5",
                apiProtocols: ["anthropic"],
              },
            ],
            modelSupplies: [
              {
                id: "cached-anthropic-route",
                modelId: "claude-opus-4-6",
                providerProfileId: "anthropic-cache",
                runtimeModel: "claude-opus-4-6",
                apiCompatibility: { mode: "native", targetApi: "anthropic" },
              },
              {
                id: "stale-cached-anthropic-route",
                modelId: "claude-fable-5",
                providerProfileId: "anthropic-cache",
                runtimeModel: "claude-fable-5",
                apiCompatibility: { mode: "native", targetApi: "anthropic" },
                harnessIds: ["claude_code"],
              },
            ],
          },
        ],
      }),
      "utf8",
    );
    const inventory = createExtensionInventory([
      builtInExtensionBundle(),
      providerBundle("anthropic-cache", [
        {
          id: "anthropic-cache",
          label: "Anthropic Cache",
          kind: "anthropic",
          baseUrl: "https://api.anthropic.com",
        },
      ]),
    ]);

    const catalog = await new ModelCatalogService({ ...paths, includeCodex: false }).list(
      inventory,
    );

    expect(catalog.modelSupplies).toContainEqual(
      expect.objectContaining({
        id: "cached-anthropic-route",
        harnessIds: ["claude_code"],
      }),
    );
    expect(
      resolveHarnessModelInventory({
        harnessId: "claude_code",
        models: catalog.models,
        supplies: catalog.modelSupplies,
        providers: catalog.providers,
        harnesses: catalog.harnesses,
      }).map((model) => model.modelId),
    ).toContain("claude-opus-4-6");
    expect(
      resolveHarnessModelInventory({
        harnessId: "claude_code",
        models: catalog.models,
        supplies: catalog.modelSupplies,
        providers: catalog.providers,
        harnesses: catalog.harnesses,
      }).map((model) => model.modelId),
    ).not.toContain("claude-fable-5");
  });

  it("V288/V290 preserves OpenAI-compatible groups and humanizes common model ids", async () => {
    const paths = await catalogPaths();
    const inventory = createExtensionInventory([
      providerBundle("grouped-new-api", [
        {
          id: "packy",
          label: "Packy",
          kind: "anthropic",
          baseUrl: "https://www.packyapi.com",
          modelDiscoveryApi: "openai_chat",
          modelDiscoveryUrl: "https://www.packyapi.com/v1/models",
        },
      ]),
    ]);
    const service = new ModelCatalogService({
      ...paths,
      env: {},
      fetch: vi.fn().mockResolvedValue(
        response({
          data: [
            { id: "claude-fable-5", owned_by: "premium" },
            { id: "claude-fable-5", owned_by: "standard" },
            { id: "gpt-5.6-sol", owned_by: "premium" },
            { id: "deepseek-v4-pro", owned_by: "standard" },
            { id: "vendor_unmatched", owned_by: "other" },
          ],
        }),
      ),
      now: fixedClock(),
    });

    const catalog = await service.refresh(inventory);

    expect(catalog.models).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "claude-fable-5", label: "Fable 5" }),
        expect.objectContaining({ id: "gpt-5.6-sol", label: "GPT 5.6 Sol" }),
        expect.objectContaining({ id: "deepseek-v4-pro", label: "DeepSeek V4 Pro" }),
        expect.objectContaining({ id: "vendor_unmatched", label: "vendor_unmatched" }),
      ]),
    );
    expect(
      catalog.modelSupplies
        .filter((supply) => supply.modelId === "claude-fable-5")
        .map((supply) => supply.providerGroup)
        .sort(),
    ).toEqual(["premium", "standard"]);
    expect(
      catalog.modelSupplies
        .filter((supply) => supply.modelId === "claude-fable-5")
        .every((supply) => supply.harnessIds === undefined),
    ).toBe(true);
    expect(humanReadableModelLabel("claude-fable-5")).toBe("Fable 5");
    expect(humanReadableModelLabel("gpt-5.6-sol")).toBe("GPT 5.6 Sol");
    expect(humanReadableModelLabel("deepseek-v4-pro")).toBe("DeepSeek V4 Pro");
    expect(humanReadableModelLabel("vendor_unmatched")).toBe("vendor_unmatched");
  });

  it("V277 normalizes both official DeepSeek URL forms and keeps one shared secret", async () => {
    const paths = await catalogPaths();
    const authStore = new MemoryProviderAuthStore();
    const fetch = vi.fn().mockResolvedValue(response({ data: [{ id: "deepseek-chat" }] }));
    const service = new ModelCatalogService({
      ...paths,
      env: {},
      authStore,
      fetch,
      now: fixedClock(),
    });
    const inventory = createExtensionInventory([]);

    const saved = await service.saveProvider(inventory, {
      label: "DeepSeek",
      kind: "anthropic",
      baseUrl: "https://api.deepseek.com",
      authMode: "api_key",
      secret: "shared-deepseek-key",
    });

    const providerId = saved.modelCatalog.userProviderIds[0] as string;
    expect(saved.providers).toEqual([
      expect.objectContaining({
        id: providerId,
        kind: "anthropic",
        baseUrl: "https://api.deepseek.com/anthropic",
        apiEntrypoints: {
          anthropic: "https://api.deepseek.com/anthropic",
          openai_chat: "https://api.deepseek.com",
        },
      }),
    ]);
    expect(fetch).not.toHaveBeenCalled();
    expect(saved.models).toEqual([]);
    expect(saved.modelSupplies).toEqual([]);

    const refreshed = await service.refresh(inventory);
    expect(fetch).toHaveBeenLastCalledWith(
      "https://api.deepseek.com/models",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer shared-deepseek-key" }),
      }),
    );
    expect(refreshed.models).toEqual([
      expect.objectContaining({
        id: "deepseek-chat",
        apiProtocols: ["anthropic", "openai_chat"],
      }),
    ]);
    expect(refreshed.modelSupplies).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          modelId: "deepseek-chat",
          providerProfileId: providerId,
          apiCompatibility: { mode: "native", targetApi: "anthropic" },
          harnessIds: undefined,
        }),
        expect.objectContaining({
          modelId: "deepseek-chat",
          providerProfileId: providerId,
          apiCompatibility: { mode: "native", targetApi: "openai_chat" },
        }),
      ]),
    );
    expect(
      resolveHarnessModelInventory({
        harnessId: "claude_code",
        models: refreshed.models,
        supplies: refreshed.modelSupplies,
        providers: refreshed.providers,
      }).map((model) => model.modelId),
    ).not.toContain("deepseek-chat");
    expect(
      resolveHarnessModelInventory({
        harnessId: "opencode",
        models: refreshed.models,
        supplies: refreshed.modelSupplies,
        providers: refreshed.providers,
      }),
    ).toEqual([]);
    for (const supply of refreshed.modelSupplies) {
      expect(await service.runtimeSecretsForSupply(refreshed, supply.id)).toEqual({
        [providerId]: "shared-deepseek-key",
      });
    }

    const chatPreferred = await service.saveProvider(inventory, {
      id: providerId,
      label: "DeepSeek",
      kind: "openai_chat",
      baseUrl: "https://api.deepseek.com/anthropic/",
      authMode: "api_key",
    });
    expect(chatPreferred.providers[0]).toEqual(
      expect.objectContaining({
        kind: "openai_chat",
        baseUrl: "https://api.deepseek.com",
        apiEntrypoints: {
          anthropic: "https://api.deepseek.com/anthropic",
          openai_chat: "https://api.deepseek.com",
        },
      }),
    );
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(await authStore.get(providerId)).toBe("shared-deepseek-key");
  });

  it("V286 normalizes a legacy persisted DeepSeek Provider before discovery", async () => {
    const paths = await catalogPaths();
    const authStore = new MemoryProviderAuthStore();
    const providerId = "swarmx.user.deepseek";
    await authStore.set(providerId, "legacy-deepseek-key");
    await writeFile(
      paths.settingsPath,
      JSON.stringify(
        createDefaultDesktopSettings({
          providers: [
            {
              id: providerId,
              displayName: "DeepSeek",
              kind: "anthropic",
              baseUrl: "https://api.deepseek.com/anthropic",
              apiEntrypoints: {},
              authMode: "auth_token",
              secretRef: { source: "local_keychain", key: providerId },
              metadata: { managedBy: "swarmx-desktop" },
            },
          ],
        }),
      ),
      "utf8",
    );
    const fetch = vi.fn().mockResolvedValue(response({ data: [{ id: "deepseek-chat" }] }));

    const catalog = await new ModelCatalogService({
      ...paths,
      env: {},
      authStore,
      fetch,
      now: fixedClock(),
    }).refresh(createExtensionInventory([]));

    expect(fetch).toHaveBeenCalledWith(
      "https://api.deepseek.com/models",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer legacy-deepseek-key" }),
      }),
    );
    expect(catalog.providers[0]).toEqual(
      expect.objectContaining({
        baseUrl: "https://api.deepseek.com/anthropic",
        apiEntrypoints: {
          anthropic: "https://api.deepseek.com/anthropic",
          openai_chat: "https://api.deepseek.com",
        },
      }),
    );
  });

  it("does not apply official DeepSeek routing to a lookalike host", async () => {
    const paths = await catalogPaths();
    const fetch = vi.fn().mockResolvedValue(response({ data: [], has_more: false }));
    const service = new ModelCatalogService({
      ...paths,
      env: {},
      authStore: new MemoryProviderAuthStore(),
      fetch,
      now: fixedClock(),
    });

    const saved = await service.saveProvider(createExtensionInventory([]), {
      label: "DeepSeek Lookalike",
      kind: "anthropic",
      baseUrl: "https://api.deepseek.com.evil.test/anthropic",
      authMode: "api_key",
      secret: "lookalike-key",
    });

    expect(saved.providers[0]).toEqual(
      expect.objectContaining({
        baseUrl: "https://api.deepseek.com.evil.test/anthropic",
        apiEntrypoints: {},
      }),
    );
    expect(fetch).not.toHaveBeenCalled();
    await service.refresh(createExtensionInventory([]));
    expect(fetch).toHaveBeenLastCalledWith(
      "https://api.deepseek.com.evil.test/anthropic/models?limit=1000",
      expect.any(Object),
    );
  });

  it("paginates Anthropic discovery and parses Ollama tags", async () => {
    const paths = await catalogPaths();
    const inventory = createExtensionInventory([
      providerBundle("explicit-anthropic-ollama", [
        {
          id: "explicit-anthropic",
          label: "Anthropic",
          kind: "anthropic",
          baseUrl: "https://api.anthropic.com",
          secretRef: { source: "env", key: "ANTHROPIC_API_KEY" },
        },
        {
          id: "explicit-ollama",
          label: "Ollama",
          kind: "ollama",
          baseUrl: "http://127.0.0.1:11434",
        },
      ]),
    ]);
    const fetch = vi.fn(async (url: string, init: { headers: Record<string, string> }) => {
      if (url.includes("api.anthropic.com")) {
        expect(init.headers["x-api-key"]).toBe("sk-ant-runtime");
        return url.includes("after_id=claude-first")
          ? response({
              data: [{ id: "claude-second", display_name: "Claude Second" }],
              has_more: false,
            })
          : response({
              data: [{ id: "claude-first", display_name: "Claude First" }],
              has_more: true,
              last_id: "claude-first",
            });
      }
      return response({ models: [{ name: "qwen3:latest" }, { model: "devstral:latest" }] });
    });
    const service = new ModelCatalogService({
      ...paths,
      env: { ANTHROPIC_API_KEY: "sk-ant-runtime" },
      fetch,
      now: fixedClock(),
    });

    const catalog = await service.refresh(inventory);

    expect(fetch).toHaveBeenCalledTimes(3);
    expect(catalog.models.map((model) => model.id).sort()).toEqual([
      "claude-first",
      "claude-second",
      "devstral:latest",
      "qwen3:latest",
    ]);
    expect(catalog.models.find((model) => model.id === "claude-second")?.label).toBe(
      "Claude Second",
    );
    expect(catalog.models.find((model) => model.id === "qwen3:latest")?.apiProtocols).toEqual([
      "ollama",
    ]);
  });

  it("preserves the last successful Provider cache when refresh fails", async () => {
    const paths = await catalogPaths();
    const inventory = createExtensionInventory([
      providerBundle("explicit-deepseek", [
        {
          id: "explicit-deepseek",
          label: "DeepSeek",
          kind: "openai_chat",
          baseUrl: "https://api.deepseek.com",
          modelDiscoveryUrl: "https://api.deepseek.com/models",
          secretRef: { source: "env", key: "DEEPSEEK_API_KEY" },
        },
      ]),
    ]);
    const ready = new ModelCatalogService({
      ...paths,
      env: { DEEPSEEK_API_KEY: "sk-deepseek" },
      fetch: vi.fn().mockResolvedValue(response({ data: [{ id: "deepseek-live" }] })),
      now: fixedClock(),
    });
    await ready.refresh(inventory);
    const failing = new ModelCatalogService({
      ...paths,
      env: { DEEPSEEK_API_KEY: "sk-deepseek" },
      fetch: vi.fn().mockRejectedValue(new Error("network offline")),
      now: fixedClock("2026-07-12T09:00:00.000Z"),
    });

    const catalog = await failing.refresh(inventory);

    expect(catalog.models.map((model) => model.id)).toContain("deepseek-live");
    expect(catalog.modelCatalog.providers).toEqual([
      expect.objectContaining({
        status: "error",
        modelCount: 1,
        error: "network offline",
      }),
    ]);
  });

  it("V282 loads the disk catalog after restart without Provider discovery", async () => {
    const paths = await catalogPaths();
    const inventory = createExtensionInventory([
      providerBundle("explicit-openai", [
        {
          id: "explicit-openai",
          label: "OpenAI",
          kind: "openai_chat",
          baseUrl: "https://api.openai.com/v1",
          secretRef: { source: "env", key: "OPENAI_API_KEY" },
        },
      ]),
    ]);
    const initialFetch = vi.fn().mockResolvedValue(
      response({
        data: [{ id: "gpt-cached" }],
      }),
    );
    await new ModelCatalogService({
      ...paths,
      env: { OPENAI_API_KEY: "sk-openai" },
      fetch: initialFetch,
      now: fixedClock(),
    }).refresh(inventory);
    const restartFetch = vi.fn().mockRejectedValue(new Error("must not fetch on list"));

    const catalog = await new ModelCatalogService({
      ...paths,
      env: { OPENAI_API_KEY: "sk-openai" },
      fetch: restartFetch,
      now: fixedClock("2026-07-13T08:00:00.000Z"),
    }).list(inventory);

    expect(initialFetch).toHaveBeenCalledTimes(1);
    expect(restartFetch).not.toHaveBeenCalled();
    expect(catalog.models).toEqual([expect.objectContaining({ id: "gpt-cached" })]);
    expect(catalog.modelCatalog.providers).toEqual([
      expect.objectContaining({ status: "cached", modelCount: 1 }),
    ]);
    expect(catalog.modelCatalog.refreshedAt).toBe("2026-07-12T08:00:00.000Z");
  });

  it("V287/V291/V337 routes only proven Codex subscription models to Codex", async () => {
    const paths = await catalogPaths();
    const inventory = createExtensionInventory([builtInExtensionBundle()]);
    const codexModelReader = vi.fn().mockResolvedValue({
      data: [
        {
          id: "gpt-5.4",
          model: "gpt-5.4",
          displayName: "GPT 5.4",
          hidden: false,
          supportedReasoningEfforts: [
            { reasoningEffort: "none", description: "Disabled" },
            { reasoningEffort: "low", description: "Fast" },
            { reasoningEffort: "max", description: "Maximum" },
            { reasoningEffort: "ultra", description: "Codex-only display tier" },
          ],
          defaultReasoningEffort: "max",
        },
        {
          id: "gpt-5.3-codex-spark",
          model: "gpt-5.3-codex-spark",
          displayName: "GPT 5.3 Codex Spark",
          hidden: false,
          supportedReasoningEfforts: [],
          defaultReasoningEffort: "medium",
        },
        {
          id: "hidden-codex-model",
          model: "hidden-codex-model",
          displayName: "Hidden",
          hidden: true,
          supportedReasoningEfforts: [],
          defaultReasoningEffort: "medium",
        },
      ],
      nextCursor: null,
    });
    const service = new ModelCatalogService({
      ...paths,
      env: {},
      codexModelReader,
      codexAccessTokenProvider: {
        available: vi.fn().mockResolvedValue(true),
        resolve: vi.fn().mockResolvedValue("subscription-access-token"),
      },
      now: fixedClock(),
    });

    const catalog = await service.refresh(inventory);

    expect(codexModelReader).toHaveBeenCalledTimes(1);
    expect(catalog.providers).toContainEqual(
      expect.objectContaining({
        id: "swarmx.local.codex",
        label: "Codex",
        apiMode: "codex_responses",
        baseUrl: "https://chatgpt.com/backend-api/codex",
        runtimeReady: true,
        catalogAdapter: "codex_app_server",
      }),
    );
    expect(catalog.models).toContainEqual(
      expect.objectContaining({
        id: "gpt-5.4",
        label: "GPT 5.4",
        runtimeModel: "gpt-5.4",
      }),
    );
    expect(catalog.models.some((model) => model.id === "hidden-codex-model")).toBe(false);
    expect(catalog.modelSupplies).toContainEqual(
      expect.objectContaining({
        modelId: "gpt-5.4",
        harnessIds: ["codex", "swarmx"],
        reasoningCapabilities: [
          expect.objectContaining({
            supportedEfforts: ["none", "low", "max"],
            defaultEffort: "max",
          }),
        ],
      }),
    );
    expect(catalog.modelSupplies).toContainEqual(
      expect.objectContaining({
        modelId: "gpt-5.3-codex-spark",
        harnessIds: ["swarmx"],
      }),
    );
    expect(
      resolveHarnessModelInventory({
        harnessId: "swarmx",
        models: catalog.models,
        supplies: catalog.modelSupplies,
        providers: catalog.providers,
        harnesses: catalog.harnesses,
      }).some((model) => model.modelId === "gpt-5.4"),
    ).toBe(true);
    expect(
      resolveHarnessModelInventory({
        harnessId: "codex",
        models: catalog.models,
        supplies: catalog.modelSupplies,
        providers: catalog.providers,
        harnesses: catalog.harnesses,
      }),
    ).toContainEqual(
      expect.objectContaining({
        modelId: "gpt-5.4",
        supplies: [
          expect.objectContaining({
            reasoning: expect.objectContaining({
              supportedEfforts: ["none", "low", "max"],
              defaultEffort: "max",
            }),
          }),
        ],
      }),
    );
    expect(
      resolveHarnessModelInventory({
        harnessId: "codex",
        models: catalog.models,
        supplies: catalog.modelSupplies,
        providers: catalog.providers,
        harnesses: catalog.harnesses,
      }).map((model) => model.modelId),
    ).not.toContain("gpt-5.3-codex-spark");

    const restartReader = vi.fn().mockRejectedValue(new Error("must not query on list"));
    const reloaded = await new ModelCatalogService({
      ...paths,
      env: {},
      codexModelReader: restartReader,
      codexAccessTokenProvider: {
        available: vi.fn().mockResolvedValue(true),
        resolve: vi.fn().mockResolvedValue("subscription-access-token"),
      },
    }).list(inventory);
    expect(restartReader).not.toHaveBeenCalled();
    expect(reloaded.models.some((model) => model.id === "gpt-5.4")).toBe(true);
    expect(await readFile(paths.cachePath, "utf8")).not.toContain('"ultra"');
    const supply = catalog.modelSupplies.find((candidate) => candidate.modelId === "gpt-5.4");
    expect(supply).toBeDefined();
    await expect(
      service.runtimeSecretsForSupply(catalog, supply?.id ?? "missing"),
    ).resolves.toEqual({ "swarmx.local.codex": "subscription-access-token" });
  });

  it("persists manual Models without a Provider and removes only the manual record", async () => {
    const paths = await catalogPaths();
    const inventory = createExtensionInventory([]);
    const service = new ModelCatalogService({ ...paths, env: {} });

    const added = await service.addManualModel(inventory, {
      id: "custom-model",
      label: "Custom Model",
      runtimeModel: "vendor/custom-model-v2",
      apiProtocol: "openai_responses",
    });

    expect(added.models).toEqual([
      expect.objectContaining({
        id: "custom-model",
        label: "Custom Model",
        runtimeModel: "vendor/custom-model-v2",
        apiProtocols: ["openai_responses"],
      }),
    ]);
    expect(added.modelSupplies).toEqual([]);
    expect(added.modelCatalog.manualModelIds).toEqual(["custom-model"]);
    const reloaded = await new ModelCatalogService({ ...paths, env: {} }).list(inventory);
    expect(reloaded.models.map((model) => model.id)).toEqual(["custom-model"]);

    const removed = await service.removeManualModel(inventory, "custom-model");
    expect(removed.models).toEqual([]);
    expect(removed.modelCatalog.manualModelIds).toEqual([]);
  });

  it("V271 saves a Provider Usage API selection and removes it when reset", async () => {
    const paths = await catalogPaths();
    const authStore = new MemoryProviderAuthStore();
    const fetch = vi.fn().mockResolvedValue(
      response({
        data: [
          { id: "claude-proxy-model", display_name: "Claude Proxy Model", owned_by: "default" },
        ],
        has_more: false,
      }),
    );
    const service = new ModelCatalogService({
      ...paths,
      env: {},
      authStore,
      fetch,
      now: fixedClock(),
    });
    const inventory = createExtensionInventory([]);

    const saved = await service.saveProvider(inventory, {
      label: "Anthropic Proxy",
      kind: "anthropic",
      baseUrl: "https://proxy.example.test/anthropic/",
      authMode: "auth_token",
      usageAdapter: "new_api",
      secret: "secret-auth-token",
    });

    const providerId = saved.modelCatalog.userProviderIds[0];
    expect(providerId).toBe("swarmx.user.anthropic-proxy");
    expect(saved.providers).toEqual([
      expect.objectContaining({
        id: providerId,
        label: "Anthropic Proxy",
        kind: "anthropic",
        baseUrl: "https://proxy.example.test/anthropic",
        authMode: "auth_token",
        usageAdapter: "new_api",
        runtimeReady: true,
      }),
    ]);
    expect(fetch).not.toHaveBeenCalled();

    const refreshed = await service.refresh(inventory);
    expect(fetch).toHaveBeenCalledWith(
      "https://proxy.example.test/v1/models",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer secret-auth-token" }),
      }),
    );
    expect(fetch.mock.calls[0]?.[1].headers).not.toHaveProperty("x-api-key");
    const persistedSettings = await readFile(paths.settingsPath, "utf8");
    expect(persistedSettings).not.toContain("secret-auth-token");
    expect(JSON.parse(persistedSettings).providers).toEqual([
      expect.objectContaining({
        id: providerId,
        authMode: "auth_token",
        secretRef: expect.objectContaining({ source: "local_keychain", key: providerId }),
        metadata: expect.objectContaining({
          managedBy: "swarmx-desktop",
          usageAdapter: "new_api",
        }),
      }),
    ]);
    const supplyId = refreshed.modelSupplies[0]?.id;
    expect(supplyId).toBeTruthy();
    expect(refreshed.modelSupplies[0]?.providerGroup).toBe("default");
    expect(await service.runtimeSecretsForSupply(refreshed, supplyId as string)).toEqual({
      [providerId as string]: "secret-auth-token",
    });
    expect(JSON.stringify(saved)).not.toContain("secret-auth-token");

    const updated = await service.saveProvider(inventory, {
      id: providerId,
      label: "Anthropic Gateway",
      kind: "anthropic",
      baseUrl: "https://gateway.example.test",
      authMode: "auth_token",
    });
    expect(updated.providers[0]).toEqual(
      expect.objectContaining({
        id: providerId,
        label: "Anthropic Gateway",
        baseUrl: "https://gateway.example.test",
      }),
    );
    expect(updated.providers[0]).not.toHaveProperty("usageAdapter");
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(JSON.parse(await readFile(paths.settingsPath, "utf8")).providers[0].metadata).toEqual({
      managedBy: "swarmx-desktop",
    });
    expect(await authStore.get(providerId as string)).toBe("secret-auth-token");
  });

  it("V278 stores New API account access separately and removes both credentials", async () => {
    const paths = await catalogPaths();
    const authStore = new MemoryProviderAuthStore();
    const service = new ModelCatalogService({
      ...paths,
      env: {},
      authStore,
      fetch: vi.fn().mockResolvedValue(response({ data: [] })),
      now: fixedClock(),
    });
    const inventory = createExtensionInventory([]);

    const saved = await service.saveProvider(inventory, {
      label: "New API Gateway",
      kind: "openai_chat",
      baseUrl: "https://new-api.example.test/v1",
      authMode: "api_key",
      usageAdapter: "new_api",
      secret: "primary-api-key",
      accountAccessToken: "account-access-token",
      accountUserId: "00042",
    });

    const providerId = saved.modelCatalog.userProviderIds[0] as string;
    const accountKey = newApiAccountCredentialKey(providerId);
    expect(accountKey).not.toBe(providerId);
    expect(await authStore.get(providerId)).toBe("primary-api-key");
    expect(await authStore.get(accountKey)).toBe("account-access-token");
    expect(saved.providers[0]).toEqual(
      expect.objectContaining({
        usageAdapter: "new_api",
        newApiAccountUserId: "42",
        accountAccessReady: true,
      }),
    );
    const persisted = await readFile(paths.settingsPath, "utf8");
    expect(persisted).not.toContain("primary-api-key");
    expect(persisted).not.toContain("account-access-token");
    expect(JSON.parse(persisted).providers[0].metadata).toEqual({
      managedBy: "swarmx-desktop",
      usageAdapter: "new_api",
      newApiAccountUserId: "42",
      modelDiscoveryUrl: "https://new-api.example.test/v1/models",
      modelDiscoveryApi: "openai_chat",
    });
    expect(JSON.stringify(saved)).not.toContain("primary-api-key");
    expect(JSON.stringify(saved)).not.toContain("account-access-token");

    await service.removeProvider(inventory, providerId);

    expect(await authStore.get(providerId)).toBeUndefined();
    expect(await authStore.get(accountKey)).toBeUndefined();
  });

  it("requires a New API user id for a separate account access token", async () => {
    const paths = await catalogPaths();
    const service = new ModelCatalogService({
      ...paths,
      env: {},
      authStore: new MemoryProviderAuthStore(),
      fetch: vi.fn().mockResolvedValue(response({ data: [] })),
    });

    await expect(
      service.saveProvider(createExtensionInventory([]), {
        label: "Incomplete New API",
        kind: "openai_chat",
        baseUrl: "https://new-api.example.test/v1",
        authMode: "api_key",
        usageAdapter: "new_api",
        secret: "primary-api-key",
        accountAccessToken: "orphan-account-token",
      }),
    ).rejects.toThrow(/User ID is required/);
  });

  it("V280 serially rolls back primary and account credentials when settings fail", async () => {
    const paths = await catalogPaths();
    await writeFile(paths.settingsPath, JSON.stringify({ schemaVersion: 1 }), "utf8");
    const authStore = new OverlapDetectingAuthStore(async (key) => {
      if (key.endsWith(":new-api-account")) {
        await rm(paths.settingsPath);
        await mkdir(paths.settingsPath);
      }
    });
    const service = new ModelCatalogService({
      ...paths,
      env: {},
      authStore,
      fetch: vi.fn().mockResolvedValue(response({ data: [] })),
    });

    await expect(
      service.saveProvider(createExtensionInventory([]), {
        label: "Rollback Gateway",
        kind: "openai_chat",
        baseUrl: "https://new-api.example.test/v1",
        authMode: "api_key",
        usageAdapter: "new_api",
        secret: "new-primary-token",
        accountAccessToken: "new-account-token",
        accountUserId: "42",
      }),
    ).rejects.toThrow();

    expect(authStore.overlapObserved).toBe(false);
    expect(authStore.entries()).toEqual([]);
  });

  it("removes Provider settings, auth, and supply cache without deleting a manual Model", async () => {
    const paths = await catalogPaths();
    const authStore = new MemoryProviderAuthStore();
    const service = new ModelCatalogService({
      ...paths,
      env: {},
      authStore,
      fetch: vi
        .fn()
        .mockResolvedValue(
          response({ data: [{ id: "shared-model", display_name: "Discovered label" }] }),
        ),
      now: fixedClock(),
    });
    const inventory = createExtensionInventory([]);
    await service.addManualModel(inventory, {
      id: "shared-model",
      label: "Independent Model",
      apiProtocol: "anthropic",
    });
    const saved = await service.saveProvider(inventory, {
      label: "Removable Provider",
      kind: "anthropic",
      baseUrl: "https://remove.example.test",
      authMode: "api_key",
      secret: "remove-me",
    });
    const providerId = saved.modelCatalog.userProviderIds[0] as string;
    await service.refresh(inventory);

    const removed = await service.removeProvider(inventory, providerId);

    expect(removed.providers).toEqual([]);
    expect(removed.modelSupplies).toEqual([]);
    expect(removed.models).toEqual([
      expect.objectContaining({ id: "shared-model", label: "Independent Model" }),
    ]);
    expect(removed.modelCatalog.userProviderIds).toEqual([]);
    expect(await authStore.get(providerId)).toBeUndefined();
    expect(await readFile(paths.settingsPath, "utf8")).not.toContain(providerId);
    expect(await readFile(paths.cachePath, "utf8")).not.toContain(providerId);
  });

  it("merges discovered and built-in metadata by stable Model id", async () => {
    const paths = await catalogPaths();
    const inventory = createExtensionInventory([
      builtInExtensionBundle(),
      providerBundle("explicit-openai-merge", [
        {
          id: "explicit-openai",
          label: "OpenAI",
          kind: "openai_chat",
          baseUrl: "https://api.openai.com/v1",
          secretRef: { source: "env", key: "OPENAI_API_KEY" },
        },
      ]),
    ]);
    const service = new ModelCatalogService({
      ...paths,
      env: { OPENAI_API_KEY: "sk-openai" },
      fetch: vi.fn().mockResolvedValue(response({ data: [{ id: "gpt-5" }] })),
      now: fixedClock(),
      includeCodex: false,
    });

    const catalog = await service.refresh(inventory);
    const gpt5 = catalog.models.filter((model) => model.id === "gpt-5");

    expect(gpt5).toHaveLength(1);
    expect(gpt5[0]?.apiProtocols).toEqual(
      expect.arrayContaining(["openai_chat", "openai_responses"]),
    );
    expect(gpt5[0]?.capabilityIds.length).toBeGreaterThan(0);
    expect(gpt5[0]?.id).not.toContain("openai");
  });

  it("does not display undiscovered built-in capability entries as catalog Models", async () => {
    const paths = await catalogPaths();
    const service = new ModelCatalogService({ ...paths, env: {}, includeCodex: false });

    const catalog = await service.list(createExtensionInventory([builtInExtensionBundle()]));

    expect(catalog.models).toEqual([]);
    expect(catalog.modelCatalog.providers).toEqual([]);
  });

  it("keeps extension discovery failures bounded and secret-safe", async () => {
    const paths = await catalogPaths();
    const inventory = createExtensionInventory([
      parseExtensionBundle({
        id: "custom-provider",
        name: "Custom Provider",
        version: "1.0.0",
        capabilities: {
          providers: [
            {
              id: "custom-openai",
              label: "Custom OpenAI",
              kind: "openai_chat",
              baseUrl: "https://models.example.test/v1",
              secretRef: { source: "env", key: "CUSTOM_API_KEY" },
            },
          ],
        },
      }),
    ]);
    const fetch = vi.fn(
      (_url: string, init: { signal: AbortSignal }) =>
        new Promise<ReturnType<typeof response>>((_resolve, reject) => {
          init.signal.addEventListener("abort", () => reject(new Error("aborted")));
        }),
    );
    const service = new ModelCatalogService({
      ...paths,
      env: { CUSTOM_API_KEY: "sk-custom-runtime" },
      fetch,
      timeoutMs: 5,
    });

    const catalog = await service.refresh(inventory);

    expect(catalog.modelCatalog.providers).toEqual([
      expect.objectContaining({ status: "error", error: expect.stringContaining("timed out") }),
    ]);
    expect(JSON.stringify(catalog)).not.toContain("sk-custom-runtime");
  });
});

async function catalogPaths() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-model-catalog-"));
  temporaryRoots.push(root);
  return {
    settingsPath: join(root, "settings.json"),
    cachePath: join(root, "model-catalog-cache.json"),
    keyUsagePath: join(root, "provider-key-usage.json"),
  };
}

function response(payload: unknown) {
  return {
    ok: true,
    status: 200,
    statusText: "OK",
    json: async () => payload,
  };
}

function fixedClock(value = "2026-07-12T08:00:00.000Z") {
  return () => new Date(value);
}

function providerBundle(id: string, providers: unknown[]) {
  return parseExtensionBundle({
    id,
    name: id,
    version: "1.0.0",
    capabilities: { providers },
  });
}

class MemoryProviderAuthStore implements ProviderAuthStore {
  private readonly values = new Map<string, string>();

  async get(key: string): Promise<string | undefined> {
    return this.values.get(key);
  }

  async set(key: string, value: string): Promise<void> {
    this.values.set(key, value);
  }

  async delete(key: string): Promise<void> {
    this.values.delete(key);
  }
}

class OverlapDetectingAuthStore implements ProviderAuthStore {
  readonly values = new Map<string, string>();
  overlapObserved = false;
  private mutationsInFlight = 0;

  constructor(private readonly afterSet: (key: string) => Promise<void>) {}

  async get(key: string): Promise<string | undefined> {
    return this.values.get(key);
  }

  async set(key: string, value: string): Promise<void> {
    await this.mutate(() => this.values.set(key, value));
    await this.afterSet(key);
  }

  async delete(key: string): Promise<void> {
    await this.mutate(() => this.values.delete(key));
  }

  entries(): Array<[string, string]> {
    return [...this.values.entries()];
  }

  private async mutate(operation: () => void): Promise<void> {
    this.mutationsInFlight += 1;
    if (this.mutationsInFlight > 1) this.overlapObserved = true;
    try {
      await new Promise((resolve) => setTimeout(resolve, 5));
      operation();
    } finally {
      this.mutationsInFlight -= 1;
    }
  }
}
