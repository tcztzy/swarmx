import { realpath } from "node:fs/promises";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import type { Context } from "@deepseek-ai/cordis";
import type { Agent } from "@deepseek-ai/dsh-agent";
import type {
  SkillCandidate,
  SkillDefinition,
  SkillLookupOptions,
  SkillProvider,
  SkillProviderControl,
  SkillProviderObservation,
} from "@deepseek-ai/dsh-skill";
import { FileSystemSkillProvider } from "@deepseek-ai/dsh-skill-filesystem";
import { type WikiSkillTarget, wikiSkillTargetSchema } from "./contracts.js";
import { WikiSkillError } from "./errors.js";
import { type WikiSkillStore, wikiSkillTargetKey } from "./store.js";

export interface WikiSkillProviderConfig {
  readonly watch?: boolean;
}

function agentTarget(agent: Agent): WikiSkillTarget | undefined {
  const candidate = {
    model: agent.options.model,
    preset: agent.session.header.agentPreset,
  };
  const parsed = wikiSkillTargetSchema.safeParse(candidate);
  return parsed.success ? parsed.data : undefined;
}

function contains(root: string, path: string): boolean {
  const child = relative(root, path);
  return child !== ".." && !child.startsWith(`..${sep}`) && !isAbsolute(child);
}

export class TargetedWikiSkillProvider implements SkillProvider {
  readonly name = "wikiskill-active";
  private delegate: FileSystemSkillProvider | undefined;
  private initialized = false;
  private target: WikiSkillTarget | undefined;
  private targetKey: string | undefined;
  private disposed = false;

  constructor(
    private readonly ctx: Context,
    private readonly control: SkillProviderControl,
    private readonly store: WikiSkillStore,
    private readonly agent: Agent,
    private readonly config: WikiSkillProviderConfig = {},
  ) {}

  async refreshTarget(): Promise<void> {
    if (this.disposed) {
      throw new WikiSkillError("WikiSkill provider is disposed", "WIKISKILL_IO_ERROR");
    }
    const target = agentTarget(this.agent);
    const targetKey = target === undefined ? undefined : wikiSkillTargetKey(target);
    if (this.initialized && targetKey === this.targetKey) return;
    const initialized = this.initialized;
    this.initialized = true;
    const previous = this.delegate;
    this.target = target;
    this.targetKey = targetKey;
    this.delegate =
      target === undefined
        ? undefined
        : new FileSystemSkillProvider(this.ctx, this.control, {
            customSkillDirs: [this.store.activeRoot(target)],
            includeDefaultRoots: false,
            providerName: this.name,
            watch: this.config.watch ?? true,
          });
    await previous?.dispose();
    if (initialized) this.control.invalidate();
  }

  async list(
    options: SkillLookupOptions,
  ): Promise<readonly SkillCandidate[] | SkillProviderObservation> {
    await this.refreshTarget();
    const delegate = this.delegate;
    const target = this.target;
    if (delegate === undefined || target === undefined) return [];
    const output = await delegate.list(options);
    const candidates = Array.isArray(output) ? output : output.candidates;
    const active: SkillCandidate[] = [];
    for (const candidate of candidates) {
      options.signal?.throwIfAborted();
      if (await this.isActiveCandidate(target, candidate)) active.push(candidate);
    }
    return Array.isArray(output) ? active : { ...output, candidates: active };
  }

  async get(
    candidate: SkillCandidate,
    options: SkillLookupOptions,
  ): Promise<SkillDefinition | undefined> {
    await this.refreshTarget();
    const delegate = this.delegate;
    const target = this.target;
    if (
      delegate === undefined ||
      target === undefined ||
      !(await this.isActiveCandidate(target, candidate))
    ) {
      return undefined;
    }
    const definition = await delegate.get(candidate, options);
    if (definition === undefined || !(await this.isActiveCandidate(target, candidate))) {
      return undefined;
    }
    return definition;
  }

  private async isActiveCandidate(
    target: WikiSkillTarget,
    candidate: SkillCandidate,
  ): Promise<boolean> {
    if (candidate.path === undefined) return false;
    const root = this.store.activeRoot(target);
    if (resolve(candidate.path) !== join(root, candidate.name, "SKILL.md")) return false;
    let storeRoot: string;
    let targetRoot: string;
    let skillPath: string;
    try {
      [storeRoot, targetRoot, skillPath] = await Promise.all([
        realpath(this.store.root),
        realpath(root),
        realpath(candidate.path),
      ]);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
      throw error;
    }
    if (!contains(storeRoot, targetRoot) || !contains(targetRoot, skillPath)) return false;
    return (await this.store.readActive(target, candidate.name)) !== undefined;
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await this.delegate?.dispose();
    this.delegate = undefined;
  }
}

export function registerWikiSkillProvider(
  agentCtx: Context,
  store: WikiSkillStore,
  config: WikiSkillProviderConfig = {},
): () => void {
  const agent = agentCtx.agent;
  if (agent === undefined) {
    throw new WikiSkillError(
      "WikiSkill provider requires an exact DSH Agent scope",
      "WIKISKILL_INVALID_REQUEST",
    );
  }
  let provider: TargetedWikiSkillProvider;
  const unregister = agentCtx.skills.registerProvider((control) => {
    provider = new TargetedWikiSkillProvider(agentCtx, control, store, agent, config);
    return provider;
  });
  const removeListener = agentCtx.on("agent/pre-step", async ({ agent: subject }, next) => {
    if (subject === agent) await provider.refreshTarget();
    return next();
  });
  return () => {
    removeListener();
    unregister();
  };
}
