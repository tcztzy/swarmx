import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { join } from "node:path";
import {
  type ExtensionActionReceipt,
  ExtensionActionRequestSchema,
  type ExtensionCandidate,
  ExtensionLifecycleManager,
  ExtensionMarketplaceCatalogSchema,
  type ExtensionMarketplaceSource,
  ExtensionMarketplaceSourceSchema,
} from "@swarmx/core";
import type { DesktopSettingsStoreLike } from "./settings-store.js";

const DEFAULT_CATALOG_LIMIT_BYTES = 2 * 1024 * 1024;
const DEFAULT_CATALOG_TIMEOUT_MS = 8_000;

export interface LoadedMarketplaceCatalog {
  candidates: ExtensionCandidate[];
  catalogDigest: string;
}

export type MarketplaceCatalogLoader = (
  source: ExtensionMarketplaceSource,
) => Promise<LoadedMarketplaceCatalog>;

export interface ExtensionManagementState {
  sources: ExtensionMarketplaceSource[];
  candidates: ExtensionCandidate[];
  installed: ReturnType<ExtensionLifecycleManager["list"]>;
  skillEvolutionEnabled: boolean;
  skillPromotionGate: "human" | "policy";
}

export class DesktopExtensionManager {
  readonly #settings: DesktopSettingsStoreLike;
  readonly #now: () => string;
  readonly #loadCatalog: MarketplaceCatalogLoader;

  constructor(
    settings: DesktopSettingsStoreLike,
    now = () => new Date().toISOString(),
    loadCatalog: MarketplaceCatalogLoader = loadMarketplaceCatalog,
  ) {
    this.#settings = settings;
    this.#now = now;
    this.#loadCatalog = loadCatalog;
  }

  async state(): Promise<ExtensionManagementState> {
    const extensions = (await this.#settings.read()).extensions;
    return {
      sources: extensions.marketplaceSources,
      candidates: extensions.marketplaceCandidates,
      installed: extensions.installed,
      skillEvolutionEnabled: extensions.skillEvolutionEnabled,
      skillPromotionGate: extensions.skillPromotionGate,
    };
  }

  async saveSource(input: unknown): Promise<ExtensionManagementState> {
    const source = ExtensionMarketplaceSourceSchema.parse(input);
    await this.#settings.update((settings) => ({
      ...settings,
      extensions: {
        ...settings.extensions,
        marketplaceSources: [
          ...settings.extensions.marketplaceSources.filter((item) => item.id !== source.id),
          source,
        ],
        trustedSourceIds:
          source.trust === "verified" || source.trust === "builtin"
            ? unique([...settings.extensions.trustedSourceIds, source.id])
            : settings.extensions.trustedSourceIds.filter((id) => id !== source.id),
      },
    }));
    return this.state();
  }

  async refreshSource(idInput: string): Promise<ExtensionManagementState> {
    const id = idInput.trim();
    if (!id) throw new Error("Marketplace source id is required.");
    const source = (await this.#settings.read()).extensions.marketplaceSources.find(
      (item) => item.id === id,
    );
    if (!source) throw new Error(`Marketplace source "${id}" was not found.`);
    if (!source.enabled) throw new Error(`Marketplace source "${id}" is disabled.`);
    const loaded = await this.#loadCatalog(source);
    await this.#settings.update((settings) => {
      if (!settings.extensions.marketplaceSources.some((item) => item.id === id)) {
        throw new Error(`Marketplace source "${id}" was removed while it was refreshing.`);
      }
      return {
        ...settings,
        extensions: {
          ...settings.extensions,
          marketplaceSources: settings.extensions.marketplaceSources.map((item) =>
            item.id === id
              ? {
                  ...item,
                  refreshedAt: this.#now(),
                  catalogDigest: loaded.catalogDigest,
                }
              : item,
          ),
          marketplaceCandidates: [
            ...settings.extensions.marketplaceCandidates.filter(
              (candidate) => candidate.revision.sourceId !== id,
            ),
            ...loaded.candidates,
          ],
        },
      };
    });
    return this.state();
  }

  async removeSource(idInput: string): Promise<ExtensionManagementState> {
    const id = idInput.trim();
    if (!id) throw new Error("Marketplace source id is required.");
    await this.#settings.update((settings) => {
      const source = settings.extensions.marketplaceSources.find((item) => item.id === id);
      if (source?.readOnly) throw new Error(`Marketplace source "${id}" is read-only.`);
      return {
        ...settings,
        extensions: {
          ...settings.extensions,
          marketplaceSources: settings.extensions.marketplaceSources.filter(
            (item) => item.id !== id,
          ),
          marketplaceCandidates: settings.extensions.marketplaceCandidates.filter(
            (candidate) => candidate.revision.sourceId !== id,
          ),
          trustedSourceIds: settings.extensions.trustedSourceIds.filter(
            (sourceId) => sourceId !== id,
          ),
        },
      };
    });
    return this.state();
  }

  async applyAction(input: unknown): Promise<{
    receipt: ExtensionActionReceipt;
    state: ExtensionManagementState;
  }> {
    const request = ExtensionActionRequestSchema.parse(input);
    let receipt!: ExtensionActionReceipt;
    await this.#settings.update((settings) => {
      const manager = new ExtensionLifecycleManager(settings.extensions.installed, this.#now);
      receipt = manager.apply(request);
      if (receipt.status !== "applied") return settings;
      const enabledPluginIds = manager
        .list()
        .filter((plugin) => plugin.enabled)
        .map((plugin) => plugin.pluginId);
      const disabledPluginIds = manager
        .list()
        .filter((plugin) => !plugin.enabled)
        .map((plugin) => plugin.pluginId);
      return {
        ...settings,
        extensions: {
          ...settings.extensions,
          installed: manager.list(),
          enabledPluginIds,
          disabledPluginIds,
        },
      };
    });
    return { receipt, state: await this.state() };
  }

  async saveEvolutionPolicy(input: {
    enabled: boolean;
    promotionGate: "human" | "policy";
  }): Promise<ExtensionManagementState> {
    await this.#settings.update((settings) => ({
      ...settings,
      extensions: {
        ...settings.extensions,
        skillEvolutionEnabled: input.enabled,
        skillPromotionGate: input.promotionGate,
      },
    }));
    return this.state();
  }
}

export async function loadMarketplaceCatalog(
  sourceInput: unknown,
  options: {
    fetchImpl?: typeof fetch;
    maxBytes?: number;
    timeoutMs?: number;
  } = {},
): Promise<LoadedMarketplaceCatalog> {
  const source = ExtensionMarketplaceSourceSchema.parse(sourceInput);
  const maxBytes = options.maxBytes ?? DEFAULT_CATALOG_LIMIT_BYTES;
  let content: string;
  if (source.kind === "local_path") {
    const sourceStat = await stat(source.location);
    const catalogPath = sourceStat.isDirectory()
      ? join(source.location, "catalog.json")
      : source.location;
    const catalogStat = sourceStat.isDirectory() ? await stat(catalogPath) : sourceStat;
    if (catalogStat.size > maxBytes) {
      throw new Error(`Extension catalog exceeds the ${maxBytes}-byte limit.`);
    }
    content = await readFile(catalogPath, "utf8");
  } else if (source.kind === "remote_catalog" || source.kind === "registry") {
    const controller = new AbortController();
    const timeout = setTimeout(
      () => controller.abort(),
      options.timeoutMs ?? DEFAULT_CATALOG_TIMEOUT_MS,
    );
    try {
      const response = await (options.fetchImpl ?? fetch)(source.location, {
        method: "GET",
        redirect: "error",
        signal: controller.signal,
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error(`Extension catalog request failed with HTTP ${response.status}.`);
      }
      content = await readBoundedResponse(response, maxBytes);
    } finally {
      clearTimeout(timeout);
    }
  } else {
    throw new Error(`Marketplace source kind "${source.kind}" cannot be refreshed by the host.`);
  }

  let document: unknown;
  try {
    document = JSON.parse(content);
  } catch {
    throw new Error("Extension catalog is not valid JSON.");
  }
  const catalog = ExtensionMarketplaceCatalogSchema.parse(document);
  const candidates = catalog.candidates.map((candidate) => ({
    ...candidate,
    trust: effectiveCandidateTrust(source.trust, candidate.trust),
    revision: { ...candidate.revision, sourceId: source.id },
  }));
  return {
    candidates,
    catalogDigest: `sha256:${createHash("sha256").update(content).digest("hex")}`,
  };
}

async function readBoundedResponse(response: Response, maxBytes: number): Promise<string> {
  const declaredLength = Number(response.headers.get("content-length"));
  if (Number.isFinite(declaredLength) && declaredLength > maxBytes) {
    throw new Error(`Extension catalog exceeds the ${maxBytes}-byte limit.`);
  }
  if (!response.body) return "";
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maxBytes) {
      await reader.cancel();
      throw new Error(`Extension catalog exceeds the ${maxBytes}-byte limit.`);
    }
    chunks.push(value);
  }
  return Buffer.concat(chunks.map((chunk) => Buffer.from(chunk))).toString("utf8");
}

function effectiveCandidateTrust(
  sourceTrust: ExtensionMarketplaceSource["trust"],
  candidateTrust: ExtensionCandidate["trust"],
): ExtensionCandidate["trust"] {
  if (candidateTrust === "untrusted" || sourceTrust === "untrusted") return "untrusted";
  if (sourceTrust === "local") return "local";
  if (sourceTrust === "verified") return "verified";
  return "builtin";
}

function unique(values: string[]): string[] {
  return [...new Set(values)];
}
