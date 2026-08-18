import { type Context, Service } from "@deepseek-ai/cordis";
import { AcpClient } from "./acp.js";
import type { HarnessCatalog, HarnessConfig, HarnessModelLaunchOptions } from "./harness.js";
import type {
  HarnessApprovalResolver,
  HarnessClient,
  HarnessLaunchRequest,
  HarnessLaunchSpec,
} from "./harness-client.js";
import type {
  BuildProviderRuntimeEnvOptions,
  ProviderApi,
  ProviderProfileMetadata,
  ProviderRuntimeEnv,
} from "./providers.js";
import { buildProviderRuntimeEnv, ProviderProfileMetadataSchema } from "./providers.js";
import {
  TASK_GUIDANCE_CATALOG,
  type TaskGuidanceCatalog,
  TaskGuidanceCatalogSchema,
  type TaskGuidanceRecord,
  TaskGuidanceRecordSchema,
  type TaskGuidanceSource,
  TaskGuidanceSourceSchema,
  validateTaskGuidanceTargets,
} from "./task-guidance.js";

export interface ProviderConnector {
  id: string;
  kinds: readonly ProviderApi[];
  /** Higher priority connectors win over the built-in connector for a kind. */
  priority?: number;
  buildRuntimeEnv(
    profile: ProviderProfileMetadata,
    options: BuildProviderRuntimeEnvOptions,
  ): ProviderRuntimeEnv;
}

export interface HarnessConnector {
  id: string;
  config: HarnessConfig;
  resolveRuntimeModel?(
    options: Pick<HarnessModelLaunchOptions, "modelId" | "runtimeModel">,
  ): string;
  resolveModelRuntimeEnv?(options: HarnessModelLaunchOptions): Record<string, string>;
}

export type HarnessTransportFactory = (
  request: HarnessLaunchRequest,
  launch: HarnessLaunchSpec,
) => HarnessClient;

/** DSH plugin registry for Provider supply connectors. */
export class ProviderConnectorService extends Service {
  private readonly connectors = new Map<string, ProviderConnector>();

  constructor(ctx: Context) {
    super(ctx, "providerConnectors");
  }

  register(connector: ProviderConnector): () => void {
    if (!connector.id.trim()) throw new Error("Provider connector id must be non-empty.");
    if (connector.kinds.length === 0) {
      throw new Error(`Provider connector "${connector.id}" must declare at least one API kind.`);
    }
    return this.ctx.effect(
      function* (this: ProviderConnectorService) {
        if (this.connectors.has(connector.id)) {
          throw new Error(`Provider connector "${connector.id}" is already registered.`);
        }
        this.connectors.set(connector.id, connector);
        yield () => {
          if (this.connectors.get(connector.id) === connector) this.connectors.delete(connector.id);
        };
      }.bind(this),
      `providerConnectors.register(${JSON.stringify(connector.id)})`,
    );
  }

  list(): ProviderConnector[] {
    return [...this.connectors.values()];
  }

  buildRuntimeEnv(
    profileInput: unknown,
    options: BuildProviderRuntimeEnvOptions,
  ): ProviderRuntimeEnv {
    const profile = ProviderProfileMetadataSchema.parse(profileInput);
    const matches = this.list().filter((connector) => connector.kinds.includes(profile.kind));
    if (matches.length === 0) return buildProviderRuntimeEnv(profile, options);
    const priority = Math.max(...matches.map((connector) => connector.priority ?? 0));
    const selected = matches.filter((connector) => (connector.priority ?? 0) === priority);
    if (selected.length > 1) {
      throw new Error(
        `Ambiguous provider connectors for kind "${profile.kind}": ${selected
          .map((connector) => connector.id)
          .join(", ")}.`,
      );
    }
    const selectedConnector = selected[0];
    if (!selectedConnector) {
      throw new Error(`No provider connector was selected for kind "${profile.kind}".`);
    }
    return selectedConnector.buildRuntimeEnv(profile, options);
  }
}

/** DSH plugin registry for Harness catalogs and request-scoped model routes. */
export class HarnessConnectorService extends Service {
  private readonly connectors = new Map<string, HarnessConnector>();

  constructor(ctx: Context) {
    super(ctx, "harnessConnectors");
  }

  register(connector: HarnessConnector): () => void {
    if (!connector.id.trim()) throw new Error("Harness connector id must be non-empty.");
    return this.ctx.effect(
      function* (this: HarnessConnectorService) {
        if (this.connectors.has(connector.id)) {
          throw new Error(`Harness connector "${connector.id}" is already registered.`);
        }
        this.connectors.set(connector.id, connector);
        yield () => {
          if (this.connectors.get(connector.id) === connector) this.connectors.delete(connector.id);
        };
      }.bind(this),
      `harnessConnectors.register(${JSON.stringify(connector.id)})`,
    );
  }

  get(id: string): HarnessConnector | undefined {
    return this.connectors.get(id);
  }

  list(): HarnessConnector[] {
    return [...this.connectors.values()];
  }

  resolveRuntimeModel(
    id: string,
    options: Pick<HarnessModelLaunchOptions, "modelId" | "runtimeModel">,
  ): string {
    const connector = this.require(id);
    return connector.resolveRuntimeModel?.(options) ?? options.runtimeModel ?? options.modelId;
  }

  resolveModelRuntimeEnv(id: string, options: HarnessModelLaunchOptions): Record<string, string> {
    return this.require(id).resolveModelRuntimeEnv?.(options) ?? {};
  }

  private require(id: string): HarnessConnector {
    const connector = this.get(id);
    if (!connector || connector.config.enabled === false) {
      throw new Error(`Unknown harness: ${id}`);
    }
    return connector;
  }
}

/** Generic DSH plugin registry for named Swarm execution strategies. */
export class SwarmStrategyService<T = unknown> extends Service {
  private readonly strategies = new Map<string, T>();

  constructor(ctx: Context) {
    super(ctx, "swarmStrategies");
  }

  register(id: string, strategy: T): () => void {
    if (!id.trim()) throw new Error("Swarm strategy id must be non-empty.");
    return this.ctx.effect(
      function* (this: SwarmStrategyService<T>) {
        if (this.strategies.has(id)) {
          throw new Error(`Swarm strategy "${id}" is already registered.`);
        }
        this.strategies.set(id, strategy);
        yield () => {
          if (this.strategies.get(id) === strategy) this.strategies.delete(id);
        };
      }.bind(this),
      `swarmStrategies.register(${JSON.stringify(id)})`,
    );
  }

  get(id: string): T | undefined {
    return this.strategies.get(id);
  }

  listIds(): string[] {
    return [...this.strategies.keys()];
  }
}

export interface TaskGuidanceContribution {
  sources?: readonly TaskGuidanceSource[];
  records: readonly TaskGuidanceRecord[];
}

/**
 * DSH plugin registry for source-dated Model/Harness/Agent task guidance.
 * The browser-safe static catalog remains the baseline; plugin contributions
 * are merged in registration order and validated against the live Harness
 * registry and the static Model registry.
 */
export class TaskGuidanceService extends Service {
  static inject = ["harnessConnectors"];

  private readonly contributions = new Map<string, TaskGuidanceContribution>();

  constructor(ctx: Context) {
    super(ctx, "taskGuidance");
  }

  register(id: string, contribution: TaskGuidanceContribution): () => void {
    if (!id.trim()) throw new Error("Task guidance contribution id must be non-empty.");
    const sources = contribution.sources?.map((source) => TaskGuidanceSourceSchema.parse(source));
    const records = contribution.records.map((record) => TaskGuidanceRecordSchema.parse(record));
    return this.ctx.effect(
      function* (this: TaskGuidanceService) {
        if (this.contributions.has(id)) {
          throw new Error(`Task guidance contribution "${id}" is already registered.`);
        }
        this.contributions.set(id, { sources, records });
        yield () => {
          if (this.contributions.get(id)?.records === records) this.contributions.delete(id);
        };
      }.bind(this),
      `taskGuidance.register(${JSON.stringify(id)})`,
    );
  }

  catalog(): TaskGuidanceCatalog {
    const sources = [...TASK_GUIDANCE_CATALOG.sources];
    const records = [...TASK_GUIDANCE_CATALOG.records];
    for (const contribution of this.contributions.values()) {
      sources.push(...(contribution.sources ?? []));
      records.push(...contribution.records);
    }
    return validateTaskGuidanceTargets(
      TaskGuidanceCatalogSchema.parse({ schemaVersion: 1, sources, records }),
      this.harnessCatalog(),
    );
  }

  private harnessCatalog(): HarnessCatalog {
    return {
      listHarnesses: () =>
        this.ctx.harnessConnectors
          .list()
          .filter((connector) => connector.config.enabled !== false)
          .map((connector) => ({ id: connector.id, config: connector.config })),
      getHarness: (id) => {
        const connector = this.ctx.harnessConnectors.get(id);
        return connector?.config.enabled === false ? undefined : connector?.config;
      },
      resolveRuntimeModel: (id, options) =>
        this.ctx.harnessConnectors.resolveRuntimeModel(id, options),
      resolveModelRuntimeEnv: (id, options) =>
        this.ctx.harnessConnectors.resolveModelRuntimeEnv(id, options),
    };
  }
}

/** DSH plugin registry for Harness wire transports. */
export class HarnessTransportService extends Service {
  static inject = ["acpLaunchers"];

  private readonly transports = new Map<string, HarnessTransportFactory>();

  constructor(ctx: Context) {
    super(ctx, "harnessTransports");
  }

  register(id: string, factory: HarnessTransportFactory): () => void {
    if (!id.trim()) throw new Error("Harness transport id must be non-empty.");
    return this.ctx.effect(
      function* (this: HarnessTransportService) {
        if (this.transports.has(id)) {
          throw new Error(`Harness transport "${id}" is already registered.`);
        }
        this.transports.set(id, factory);
        yield () => {
          if (this.transports.get(id) === factory) this.transports.delete(id);
        };
      }.bind(this),
      `harnessTransports.register(${JSON.stringify(id)})`,
    );
  }

  createClient(request: HarnessLaunchRequest): HarnessClient {
    const launch = this.ctx.acpLaunchers.resolve(request);
    const factory = this.transports.get(request.transport ?? request.command);
    if (factory) return factory(request, launch);
    return new AcpClient({ resolveLaunch: () => launch });
  }
}

/**
 * DSH plugin registry for Harness permission resolvers. Codex and other
 * non-ACP transports consume the highest-priority registered resolver; a DSH
 * host may register a bridge to its own permission plugin here.
 */
export class HarnessPermissionService extends Service {
  private readonly resolvers = new Map<
    string,
    { resolver: HarnessApprovalResolver; priority: number }
  >();

  constructor(ctx: Context) {
    super(ctx, "harnessPermissions");
  }

  register(id: string, resolver: HarnessApprovalResolver, priority = 0): () => void {
    if (!id.trim()) throw new Error("Harness permission resolver id must be non-empty.");
    return this.ctx.effect(
      function* (this: HarnessPermissionService) {
        if (this.resolvers.has(id)) {
          throw new Error(`Harness permission resolver "${id}" is already registered.`);
        }
        this.resolvers.set(id, { resolver, priority });
        yield () => {
          if (this.resolvers.get(id)?.resolver === resolver) this.resolvers.delete(id);
        };
      }.bind(this),
      `harnessPermissions.register(${JSON.stringify(id)})`,
    );
  }

  resolve(): HarnessApprovalResolver | undefined {
    const entries = [...this.resolvers.values()];
    if (entries.length === 0) return undefined;
    const priority = Math.max(...entries.map((entry) => entry.priority));
    const selected = entries.filter((entry) => entry.priority === priority);
    if (selected.length > 1) {
      throw new Error("Ambiguous harness permission resolvers at the same priority.");
    }
    return selected[0]?.resolver;
  }
}

declare module "@deepseek-ai/cordis" {
  interface Context {
    providerConnectors: ProviderConnectorService;
    harnessConnectors: HarnessConnectorService;
    harnessPermissions: HarnessPermissionService;
    harnessTransports: HarnessTransportService;
    swarmStrategies: SwarmStrategyService;
    taskGuidance: TaskGuidanceService;
  }
}
