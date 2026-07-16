import { createHash } from "node:crypto";
import type { Dirent } from "node:fs";
import { readFile, readdir, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { extname, join, resolve } from "node:path";
import {
  type AgentProfile,
  type AgentProfileMetadata,
  AgentProfileMetadataSchema,
  AgentProfileSchema,
  type HarnessRecipe,
  HarnessRecipeSchema,
  createAgentProfileFromDefinition,
  parseNativeAgentDefinition,
} from "@swarmx/core";
import type { DesktopSettingsStoreLike } from "./settings-store.js";

const MAX_NATIVE_AGENT_FILE_BYTES = 1024 * 1024;

export interface DiscoverNativeAgentsOptions {
  workspaceRoot: string;
  homeDir?: string;
}

export interface NativeAgentDiscoveryWarning {
  source: string;
  message: string;
}

export interface NativeAgentDiscoveryResult {
  agents: AgentProfile[];
  warnings: NativeAgentDiscoveryWarning[];
}

export interface SaveCustomAgentOptions {
  reservedAgentIds?: string[];
}

export class CustomAgentService {
  readonly #settings: DesktopSettingsStoreLike;
  readonly #now: () => string;

  constructor(settings: DesktopSettingsStoreLike, now = () => new Date().toISOString()) {
    this.#settings = settings;
    this.#now = now;
  }

  async list(): Promise<AgentProfileMetadata[]> {
    return (await this.#settings.read()).agents;
  }

  async discoverNative(options: DiscoverNativeAgentsOptions): Promise<NativeAgentDiscoveryResult> {
    const workspaceRoot = resolve(options.workspaceRoot);
    const homeRoot = resolve(options.homeDir ?? homedir());
    const locations: NativeAgentLocation[] = [
      nativeAgentLocation("codex", "user", join(homeRoot, ".codex", "agents")),
      nativeAgentLocation("claude_code", "user", join(homeRoot, ".claude", "agents")),
      nativeAgentLocation("codex", "project", join(workspaceRoot, ".codex", "agents")),
      nativeAgentLocation("claude_code", "project", join(workspaceRoot, ".claude", "agents")),
    ];
    const selected = new Map<string, DiscoveredNativeAgent>();
    const warnings: NativeAgentDiscoveryWarning[] = [];

    for (const location of locations) {
      let entries: Dirent[];
      try {
        entries = await readdir(location.directory, { withFileTypes: true });
      } catch (error) {
        if (isMissingPathError(error)) continue;
        warnings.push({ source: location.directory, message: errorMessage(error) });
        continue;
      }

      for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
        if (!entry.isFile() || extname(entry.name).toLowerCase() !== location.extension) continue;
        const filePath = join(location.directory, entry.name);
        try {
          const file = await stat(filePath);
          if (file.size > MAX_NATIVE_AGENT_FILE_BYTES) {
            throw new Error("Native Agent definition exceeds the 1 MiB read limit.");
          }
          const definition = parseNativeAgentDefinition(await readFile(filePath, "utf8"), {
            format: location.format,
            source: {
              kind: location.scope,
              path: filePath,
              label: `${nativeHostLabel(location.host)} ${location.scope} Agent`,
              readOnly: true,
            },
          });
          const metadata = createAgentProfileFromDefinition(definition, {
            id: nativeAgentId(location.host, definition.frontmatter.name ?? entry.name),
            harnessId: location.host,
            readOnly: true,
          });
          const { source, ...profile } = metadata;
          const agent = AgentProfileSchema.parse({ ...profile, definition: source });
          const key = `${location.host}\u0000${agent.name.trim().toLowerCase()}`;
          const current = selected.get(key);
          if (current && current.priority === location.priority) {
            warnings.push({
              source: filePath,
              message: `Duplicate ${nativeHostLabel(location.host)} ${location.scope} Agent name "${agent.name}"; kept ${current.path}.`,
            });
            continue;
          }
          if (!current || location.priority > current.priority) {
            selected.set(key, {
              agent,
              path: filePath,
              priority: location.priority,
            });
          }
        } catch (error) {
          warnings.push({ source: filePath, message: errorMessage(error) });
        }
      }
    }

    return {
      agents: [...selected.values()]
        .map((entry) => entry.agent)
        .sort((left, right) => left.id.localeCompare(right.id)),
      warnings,
    };
  }

  async save(input: unknown, options: SaveCustomAgentOptions = {}): Promise<AgentProfileMetadata> {
    const requested = AgentProfileMetadataSchema.parse(input);
    if (requested.readOnly) throw new Error("Custom Agent profiles cannot be read-only.");
    if (!requested.harnessRecipe) {
      throw new Error("Custom Agent requires a reproducible Harness recipe.");
    }
    const requestedRecipe = requested.harnessRecipe;
    if (!requested.modelId) throw new Error("Custom Agent requires a Model.");
    if (options.reservedAgentIds?.includes(requested.id)) {
      throw new Error(
        `Agent profile "${requested.id}" is provided by an Extension and is read-only.`,
      );
    }

    let saved!: AgentProfileMetadata;
    await this.#settings.update((settings) => {
      const existing = settings.agents.find((agent) => agent.id === requested.id);
      if (existing?.readOnly) throw new Error(`Agent profile "${requested.id}" is read-only.`);
      const harnessRecipe = this.#revisionFor(requestedRecipe, existing?.harnessRecipe);
      const history = existing?.harnessRecipe
        ? uniqueRecipes([
            ...existing.harnessRecipeHistory,
            ...(existing.harnessRecipe.revisionId === harnessRecipe.revisionId
              ? []
              : [existing.harnessRecipe]),
          ])
        : [];
      saved = AgentProfileMetadataSchema.parse({
        ...requested,
        harnessId: harnessRecipe.id,
        harnessRecipe,
        harnessRecipeHistory: history,
        readOnly: false,
        source: { kind: "local", label: "Custom Agent", readOnly: false },
      });
      return {
        ...settings,
        agents: [...settings.agents.filter((agent) => agent.id !== saved.id), saved],
      };
    });
    return saved;
  }

  async remove(idInput: string): Promise<boolean> {
    const id = idInput.trim();
    if (!id) throw new Error("Custom Agent id is required.");
    let removed = false;
    await this.#settings.update((settings) => {
      const existing = settings.agents.find((agent) => agent.id === id);
      if (existing?.readOnly) throw new Error(`Agent profile "${id}" is read-only.`);
      removed = Boolean(existing);
      return { ...settings, agents: settings.agents.filter((agent) => agent.id !== id) };
    });
    return removed;
  }

  #revisionFor(requested: HarnessRecipe, existing?: HarnessRecipe): HarnessRecipe {
    const content = canonicalRecipeContent(requested);
    const contentDigest = `sha256:${createHash("sha256").update(content).digest("hex")}`;
    if (existing?.contentDigest === contentDigest) return existing;
    return HarnessRecipeSchema.parse({
      ...requested,
      revisionId: `${requested.id}@${contentDigest.slice(7, 19)}`,
      contentDigest,
      createdAt: this.#now(),
    });
  }
}

type NativeAgentHost = "claude_code" | "codex";
type NativeAgentScope = "project" | "user";

interface NativeAgentLocation {
  host: NativeAgentHost;
  format: NativeAgentHost;
  scope: NativeAgentScope;
  directory: string;
  extension: ".md" | ".toml";
  priority: number;
}

interface DiscoveredNativeAgent {
  agent: AgentProfile;
  path: string;
  priority: number;
}

function nativeAgentLocation(
  host: NativeAgentHost,
  scope: NativeAgentScope,
  directory: string,
): NativeAgentLocation {
  return {
    host,
    format: host,
    scope,
    directory,
    extension: host === "codex" ? ".toml" : ".md",
    priority: scope === "project" ? 1 : 0,
  };
}

function nativeAgentId(host: NativeAgentHost, name: string): string {
  const slug = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `native:${host}:${slug || "agent"}`;
}

function nativeHostLabel(host: NativeAgentHost): string {
  return host === "codex" ? "Codex" : "Claude Code";
}

function isMissingPathError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error as { code?: unknown }).code === "ENOENT"
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function canonicalRecipeContent(recipe: HarnessRecipe): string {
  const {
    revisionId: _revisionId,
    contentDigest: _contentDigest,
    createdAt: _createdAt,
    ...content
  } = recipe;
  return JSON.stringify(sortRecord(content));
}

function sortRecord(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortRecord);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => [key, sortRecord(child)]),
  );
}

function uniqueRecipes(recipes: HarnessRecipe[]): HarnessRecipe[] {
  const byRevision = new Map<string, HarnessRecipe>();
  for (const recipe of recipes) byRevision.set(recipe.revisionId, recipe);
  return [...byRevision.values()];
}
