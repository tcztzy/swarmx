import { randomUUID } from "node:crypto";
import { Context, type Plugin, Service } from "@deepseek-ai/cordis";
import {
  type AcpLaunchRequest,
  type AcpLaunchResolver,
  type AcpLaunchSpec,
  type CodexHarnessPluginConfig,
  codexHarnessPlugin,
} from "@swarmx/codex";
import { AcpClient, type AcpClientOptions } from "./acp.js";
import { Agent, type AgentRuntimeOptions } from "./agent.js";
import {
  type HarnessConnector,
  HarnessConnectorService,
  HarnessPermissionService,
  HarnessTransportService,
  type ProviderConnector,
  ProviderConnectorService,
  SwarmStrategyService,
  TaskGuidanceService,
} from "./dsh-plugin.js";
import {
  HARNESSES,
  type HarnessCatalog,
  type HarnessCatalogEntry,
  type HarnessModelLaunchOptions,
  harnessModelRuntimeEnv,
  harnessModelRuntimeModel,
} from "./harness.js";
import type {
  HarnessClient,
  HarnessLaunchRequest,
  HarnessSessionClient,
} from "./harness-client.js";
import { McpManager, type McpManagerOptions } from "./mcp.js";
import {
  type BuildProviderRuntimeEnvOptions,
  buildProviderRuntimeEnv,
  type ProviderRuntimeEnv,
} from "./providers.js";
import {
  type DiscoveredSession,
  type GroupedSessionsResult,
  type ListGroupedSessionsOptions,
  type LoadDiscoveredSessionOptions,
  listGroupedSessions,
  loadDiscoveredSession,
} from "./session-discovery.js";
import { Swarm, type SwarmRuntimeOptions } from "./swarm.js";
import {
  getTaskGuidanceForAgent,
  getTaskGuidanceForHarness,
  getTaskGuidanceForModel,
  getTaskGuidanceForTask,
  type TaskGuidanceRecord,
  type TaskGuidanceTaskFamily,
} from "./task-guidance.js";
import {
  type AgentConfig,
  AgentConfigSchema,
  type EvalRunResult,
  type MessageChunk,
  type ModelTokenUsage,
  type SessionData,
  type SwarmConfig,
  SwarmConfigSchema,
} from "./types.js";

export type CoreAgentRuntimeOptions = Omit<
  AgentRuntimeOptions,
  "createAcpClient" | "createHarnessClient" | "createMcpManager"
>;

export interface CoreSwarmRuntimeOptions extends Omit<SwarmRuntimeOptions, "agent"> {
  agent?: CoreAgentRuntimeOptions;
}

export interface CoreAgentExecution {
  call(
    arguments_: Record<string, unknown>,
    context?: Record<string, unknown>,
    onUsage?: (usage: ModelTokenUsage) => void,
  ): Promise<{ messages: MessageChunk[] }>;
}

export interface CoreSwarmExecution {
  readonly name: string;
  readonly root: string;
  readonly models: readonly { id: string; object: "model" }[];
  execute(
    arguments_: Record<string, unknown>,
    context?: Record<string, unknown>,
    onChunk?: (chunk: MessageChunk) => void,
    onUsage?: (usage: ModelTokenUsage) => void,
  ): Promise<MessageChunk[]>;
  executeForEval(
    arguments_: Record<string, unknown>,
    context?: Record<string, unknown>,
    onUsage?: (usage: ModelTokenUsage) => void,
  ): Promise<EvalRunResult>;
  listAllSessions(cwd?: string): Promise<Array<{ agent: string; sessions: unknown[] }>>;
}

export interface SwarmStrategy {
  readonly id: string;
  prepare(
    config: SwarmConfig,
    options: CoreSwarmRuntimeOptions,
    resources: CoreRequestResources,
  ): CoreSwarmExecution;
}

interface CoreRequestResources {
  createHarnessClient(request: HarnessLaunchRequest): HarnessClient;
  createMcpManager(): McpManager;
}

/** Cordis service that owns effect-scoped Harness command providers. */
export class AcpLauncherService extends Service {
  private readonly resolvers = new Map<string, AcpLaunchResolver>();

  constructor(ctx: Context) {
    super(ctx, "acpLaunchers");
  }

  register(command: string, resolver: AcpLaunchResolver): () => void {
    return this.ctx.effect(
      function* (this: AcpLauncherService) {
        if (!command.trim()) throw new Error("ACP launcher command must be non-empty.");
        if (this.resolvers.has(command)) {
          throw new Error(`ACP launcher command "${command}" is already registered.`);
        }
        this.resolvers.set(command, resolver);
        yield () => {
          if (this.resolvers.get(command) === resolver) this.resolvers.delete(command);
        };
      }.bind(this),
      `acpLaunchers.register(${JSON.stringify(command)})`,
    );
  }

  resolve(request: AcpLaunchRequest): AcpLaunchSpec {
    const resolver = this.resolvers.get(request.command);
    return resolver
      ? resolver(request)
      : { command: request.command, args: [...request.args], env: {} };
  }
}

/** Creates Harness clients whose child processes are owned by the caller Fiber. */
export class AcpRuntimeService extends Service {
  static inject = ["harnessTransports"];

  constructor(ctx: Context) {
    super(ctx, "acpRuntime");
  }

  createClient(request?: HarnessLaunchRequest): HarnessClient {
    const client = request ? this.ctx.harnessTransports.createClient(request) : new AcpClient();
    this.ctx.effect(() => () => client.kill?.(), "acpRuntime.createClient");
    return client;
  }

  createClientFor(options: AcpClientOptions): HarnessSessionClient {
    return this.createClient({
      command: options.command,
      args: options.args,
      ...(options.transport ? { transport: options.transport } : {}),
    }) as HarnessSessionClient;
  }
}

/** Creates MCP managers whose transports are owned by the caller Fiber. */
export class McpRuntimeService extends Service {
  constructor(ctx: Context) {
    super(ctx, "mcpRuntime");
  }

  createManager(options?: McpManagerOptions): McpManager {
    const manager = new McpManager(options);
    this.ctx.effect(() => () => manager.close(), "mcpRuntime.createManager");
    return manager;
  }
}

/** Sole constructor and request-Fiber boundary for Swarm and Agent execution. */
export class SwarmRuntimeService extends Service {
  static inject = ["acpRuntime", "mcpRuntime", "swarmStrategies"];

  constructor(ctx: Context) {
    super(ctx, "swarmRuntime");
  }

  prepare(configInput: SwarmConfig, options: CoreSwarmRuntimeOptions = {}): CoreSwarmExecution {
    const config = SwarmConfigSchema.parse(configInput) as SwarmConfig;
    const strategy = this.swarmStrategy(config);
    return {
      name: config.name,
      root: config.root,
      models: describeModels(config),
      execute: (arguments_, context, onChunk, onUsage) =>
        this.runInRequestFiber((resources) =>
          strategy
            .prepare(config, options, resources)
            .execute(arguments_, context, onChunk, onUsage),
        ),
      executeForEval: (arguments_, context, onUsage) =>
        this.runInRequestFiber((resources) =>
          strategy.prepare(config, options, resources).executeForEval(arguments_, context, onUsage),
        ),
      listAllSessions: (cwd) =>
        this.runInRequestFiber((resources) =>
          strategy.prepare(config, options, resources).listAllSessions(cwd),
        ),
    };
  }

  prepareAgent(
    configInput: AgentConfig,
    options: CoreAgentRuntimeOptions = {},
  ): CoreAgentExecution {
    const config = AgentConfigSchema.parse(configInput);
    return {
      call: (arguments_, context, onUsage) =>
        this.runInRequestFiber((resources) =>
          createRuntimeAgent(config, options, resources).call(arguments_, context, onUsage),
        ),
    };
  }

  private swarmStrategy(config: SwarmConfig): SwarmStrategy {
    const id = config.strategy ?? "dag";
    const strategy = this.ctx.swarmStrategies.get(id) as SwarmStrategy | undefined;
    if (!strategy) throw new Error(`Unknown swarm strategy "${id}".`);
    return strategy;
  }

  private async runInRequestFiber<T>(run: (resources: CoreRequestResources) => Promise<T>) {
    let result: T | undefined;
    const fiber = this.ctx.plugin({
      name: `swarmx-request-${randomUUID()}`,
      inject: ["harnessTransports", "mcpRuntime"],
      async apply(ctx) {
        result = await run({
          createHarnessClient: (request) => ctx.harnessTransports.createClient(request),
          createMcpManager: () => ctx.mcpRuntime.createManager(),
        });
      },
    });
    try {
      await fiber;
      return result as T;
    } finally {
      await fiber.dispose();
    }
  }
}

function createRuntimeSwarm(
  config: SwarmConfig,
  options: CoreSwarmRuntimeOptions,
  resources: CoreRequestResources,
): Swarm {
  return new Swarm(config, {
    ...options,
    agent: runtimeAgentOptions(options.agent, resources),
  });
}

function createRuntimeAgent(
  config: AgentConfig,
  options: CoreAgentRuntimeOptions,
  resources: CoreRequestResources,
): Agent {
  return new Agent(config, runtimeAgentOptions(options, resources));
}

function runtimeAgentOptions(
  options: CoreAgentRuntimeOptions = {},
  resources: CoreRequestResources,
): AgentRuntimeOptions {
  return {
    ...options,
    createHarnessClient: resources.createHarnessClient,
    createMcpManager: resources.createMcpManager,
  };
}

export const builtinProviderConnector: ProviderConnector = {
  id: "swarmx-builtin-provider",
  kinds: ["anthropic", "openai_chat", "openai_responses", "ollama"],
  buildRuntimeEnv: (profile, options) => buildProviderRuntimeEnv(profile, options),
};

/** Cordis plugin that contributes the built-in Provider supply connectors. */
export const builtinProviderConnectorPlugin = {
  name: "swarmx-builtin-provider-connectors",
  inject: ["providerConnectors"],
  apply(ctx: Context) {
    ctx.providerConnectors.register(builtinProviderConnector);
  },
} satisfies Plugin.Object;

export const builtinHarnessConnectors = Object.entries(HARNESSES).map(
  ([id, config]): HarnessConnector => ({
    id,
    config,
    resolveRuntimeModel: (options) => harnessModelRuntimeModel(id, options),
    resolveModelRuntimeEnv: (options) => harnessModelRuntimeEnv(id, options),
  }),
);

/** Cordis plugin that contributes the built-in Harness catalog. */
export const builtinHarnessConnectorPlugin = {
  name: "swarmx-builtin-harness-connectors",
  inject: ["harnessConnectors"],
  apply(ctx: Context) {
    for (const connector of builtinHarnessConnectors) {
      ctx.harnessConnectors.register(connector);
    }
  },
} satisfies Plugin.Object;

/** Cordis plugin that contributes the default DAG Swarm execution strategy. */
export const dagSwarmStrategyPlugin = {
  name: "swarmx-dag-swarm-strategy",
  inject: ["swarmStrategies"],
  apply(ctx: Context) {
    ctx.swarmStrategies.register("dag", {
      id: "dag",
      prepare: (
        config: SwarmConfig,
        options: CoreSwarmRuntimeOptions,
        resources: CoreRequestResources,
      ) => dagExecution(config, options, resources),
    });
  },
} satisfies Plugin.Object;

/** Built-in fail-closed permission resolver; DSH hosts override with priority. */
export const failClosedHarnessPermissionPlugin = {
  name: "swarmx-fail-closed-harness-permissions",
  inject: ["harnessPermissions"],
  apply(ctx: Context) {
    ctx.harnessPermissions.register("swarmx-fail-closed", async () => ({
      outcome: "cancelled" as const,
    }));
  },
} satisfies Plugin.Object;

function dagExecution(
  config: SwarmConfig,
  options: CoreSwarmRuntimeOptions,
  resources: CoreRequestResources,
): CoreSwarmExecution {
  const swarm = createRuntimeSwarm(config, options, resources);
  return {
    name: swarm.name,
    root: swarm.root,
    models: describeModels(config),
    execute: (arguments_, context, onChunk, onUsage) =>
      swarm.execute(arguments_, context, onChunk, onUsage),
    executeForEval: (arguments_, context, onUsage) =>
      swarm.executeForEval(arguments_, context, onUsage),
    listAllSessions: (cwd) => swarm.listAllSessions(cwd),
  };
}

declare module "@deepseek-ai/cordis" {
  interface Context {
    acpRuntime: AcpRuntimeService;
    mcpRuntime: McpRuntimeService;
    swarmRuntime: SwarmRuntimeService;
  }
}

export interface CoreRuntimeOptions {
  codex?: CodexHarnessPluginConfig;
  /** Additional DSH plugins loaded after the built-in runtime plugins. */
  plugins?: readonly Plugin[];
}

/**
 * One DSH plugin that installs every Core runtime Service, the built-in
 * Provider/Harness/Swarm plugins, and first-party Codex into an existing
 * Cordis Context. A DeepSeek Harness host can load this directly.
 */
export const coreRuntimePlugin = {
  name: "swarmx-core-runtime",
  async apply(ctx: Context, options: CoreRuntimeOptions = {}) {
    await ctx.plugin(AcpLauncherService);
    await ctx.plugin(HarnessTransportService);
    await ctx.plugin(AcpRuntimeService);
    await ctx.plugin(McpRuntimeService);
    await ctx.plugin(ProviderConnectorService);
    await ctx.plugin(HarnessConnectorService);
    await ctx.plugin(TaskGuidanceService);
    await ctx.plugin(HarnessPermissionService);
    await ctx.plugin(SwarmStrategyService);
    await ctx.plugin(SwarmRuntimeService);
    await ctx.plugin(builtinProviderConnectorPlugin);
    await ctx.plugin(builtinHarnessConnectorPlugin);
    await ctx.plugin(failClosedHarnessPermissionPlugin);
    await ctx.plugin(dagSwarmStrategyPlugin);
    await ctx.plugin(codexHarnessPlugin, options.codex ?? {});
    for (const plugin of options.plugins ?? []) {
      await ctx.plugin(plugin);
    }
  },
} satisfies Plugin.Object<CoreRuntimeOptions>;

export interface CoreRuntime {
  readonly context: Context;
  readonly harnessCatalog: HarnessCatalog;
  resolveAcpLaunch(request: AcpLaunchRequest): AcpLaunchSpec;
  prepareSwarm(config: SwarmConfig, options?: CoreSwarmRuntimeOptions): CoreSwarmExecution;
  prepareAgent(config: AgentConfig, options?: CoreAgentRuntimeOptions): CoreAgentExecution;
  resolveProviderRuntimeEnv(
    profile: unknown,
    options: BuildProviderRuntimeEnvOptions,
  ): ProviderRuntimeEnv;
  listHarnessConnectors(): HarnessConnector[];
  getHarnessConnector(id: string): HarnessConnector | undefined;
  resolveHarnessModelRuntimeEnv(
    id: string,
    options: HarnessModelLaunchOptions,
  ): Record<string, string>;
  listSwarmStrategies(): string[];
  getTaskGuidanceForModel(modelId: string): TaskGuidanceRecord[];
  getTaskGuidanceForHarness(harnessId: string): TaskGuidanceRecord[];
  getTaskGuidanceForAgent(
    harnessId: string,
    modelId: string,
    taskFamily?: TaskGuidanceTaskFamily,
  ): TaskGuidanceRecord[];
  getTaskGuidanceForTask(taskFamily: TaskGuidanceTaskFamily): TaskGuidanceRecord[];
  listGroupedSessions(
    options?: Omit<
      ListGroupedSessionsOptions,
      "createClient" | "createClientFor" | "harnessCatalog"
    >,
  ): Promise<GroupedSessionsResult>;
  loadDiscoveredSession(
    session: DiscoveredSession,
    options?: Omit<
      LoadDiscoveredSessionOptions,
      "createClient" | "createClientFor" | "harnessCatalog"
    >,
  ): Promise<SessionData | null>;
  dispose(): Promise<void>;
}

/** Boot the one canonical Core Cordis container and its first-party services. */
export async function createCoreRuntime(options: CoreRuntimeOptions = {}): Promise<CoreRuntime> {
  const context = new Context();
  try {
    await context.plugin(coreRuntimePlugin, options);
  } catch (error) {
    await context.fiber.dispose();
    throw error;
  }

  let disposed = false;
  const assertActive = (): void => {
    if (disposed) throw new Error("Core runtime is disposed.");
  };
  const harnessCatalog: HarnessCatalog = {
    listHarnesses(): HarnessCatalogEntry[] {
      assertActive();
      return context.harnessConnectors
        .list()
        .filter((connector) => connector.config.enabled !== false)
        .map((connector) => ({ id: connector.id, config: connector.config }));
    },
    getHarness(id: string) {
      assertActive();
      const connector = context.harnessConnectors.get(id);
      return connector?.config.enabled === false ? undefined : connector?.config;
    },
    resolveRuntimeModel(id, modelOptions) {
      assertActive();
      return context.harnessConnectors.resolveRuntimeModel(id, modelOptions);
    },
    resolveModelRuntimeEnv(id, modelOptions) {
      assertActive();
      return context.harnessConnectors.resolveModelRuntimeEnv(id, modelOptions);
    },
  };
  return {
    context,
    harnessCatalog,
    resolveAcpLaunch(request) {
      assertActive();
      return context.acpLaunchers.resolve(request);
    },
    prepareSwarm(config, runtimeOptions) {
      assertActive();
      return context.swarmRuntime.prepare(config, runtimeOptions);
    },
    prepareAgent(config, runtimeOptions) {
      assertActive();
      return context.swarmRuntime.prepareAgent(config, runtimeOptions);
    },
    resolveProviderRuntimeEnv(profile, runtimeOptions) {
      assertActive();
      return context.providerConnectors.buildRuntimeEnv(profile, runtimeOptions);
    },
    listHarnessConnectors() {
      assertActive();
      return context.harnessConnectors.list();
    },
    getHarnessConnector(id) {
      assertActive();
      return context.harnessConnectors.get(id);
    },
    resolveHarnessModelRuntimeEnv(id, modelOptions) {
      assertActive();
      return context.harnessConnectors.resolveModelRuntimeEnv(id, modelOptions);
    },
    listSwarmStrategies() {
      assertActive();
      return context.swarmStrategies.listIds();
    },
    getTaskGuidanceForModel(modelId) {
      assertActive();
      return getTaskGuidanceForModel(modelId, context.taskGuidance.catalog());
    },
    getTaskGuidanceForHarness(harnessId) {
      assertActive();
      return getTaskGuidanceForHarness(harnessId, context.taskGuidance.catalog());
    },
    getTaskGuidanceForAgent(harnessId, modelId, taskFamily) {
      assertActive();
      return getTaskGuidanceForAgent(
        harnessId,
        modelId,
        taskFamily,
        context.taskGuidance.catalog(),
      );
    },
    getTaskGuidanceForTask(taskFamily) {
      assertActive();
      return getTaskGuidanceForTask(taskFamily, context.taskGuidance.catalog());
    },
    listGroupedSessions(options = {}) {
      assertActive();
      return runWithAcpRequestFiber(context, "session-list", (createClientFor) =>
        listGroupedSessions({ ...options, harnessCatalog, createClientFor }),
      );
    },
    loadDiscoveredSession(session, options = {}) {
      assertActive();
      return runWithAcpRequestFiber(context, "session-load", (createClientFor) =>
        loadDiscoveredSession(session, { ...options, harnessCatalog, createClientFor }),
      );
    },
    async dispose() {
      if (disposed) return;
      disposed = true;
      await context.fiber.dispose();
    },
  };
}

async function runWithAcpRequestFiber<T>(
  context: Context,
  label: string,
  run: (createClientFor: (options: AcpClientOptions) => HarnessSessionClient) => Promise<T>,
): Promise<T> {
  let result: T | undefined;
  const fiber = context.plugin({
    name: `swarmx-${label}-${randomUUID()}`,
    inject: ["acpRuntime"],
    async apply(ctx) {
      result = await run((options) => ctx.acpRuntime.createClientFor(options));
    },
  });
  try {
    await fiber;
    return result as T;
  } finally {
    await fiber.dispose();
  }
}

function describeModels(config: SwarmConfig): Array<{ id: string; object: "model" }> {
  const models = Object.entries(config.nodes)
    .filter(([, node]) => node.kind === "agent")
    .map(([id]) => ({ id, object: "model" as const }));
  if (config.queen) models.push({ id: config.queen.name, object: "model" });
  return models;
}
