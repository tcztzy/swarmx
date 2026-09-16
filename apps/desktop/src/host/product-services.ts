import { createHash, randomBytes, randomUUID } from "node:crypto";
import { realpath, stat } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { EventType } from "@ag-ui/core";
import { MemoryService } from "@swarmx/memory";
import {
  createScienceToolDefinitions,
  parseScienceResourceId,
  type ScienceAttachmentStore,
  ScienceCore,
  ScienceError,
  type ScienceToolDefinition,
} from "@swarmx/science";
import { createSwarm } from "@swarmx/swarm";
import { z } from "zod";
import { AGENT_IDS, type AgentId, selectedAgent } from "../agent.js";
import type { AgentOptions, NativeAgent } from "../agents/types.js";
import {
  type AgentPermissions,
  HarnessSchema,
  intersectPermissions,
  narrowPermissions,
  type PermissionRequest,
  PermissionRequestSchema,
  policyPermissions,
} from "../permissions.js";
import { ExecutionPolicySchema } from "../settings.js";
import { A2AEndpoints, SwarmA2AExecutor } from "./a2a.js";
import { AgUiBridge } from "./ag-ui.js";
import { AgentRegistry, bindAgent } from "./agent-registry.js";
import { publicCapabilities } from "./capabilities.js";
import { ExecutionJournal } from "./execution-journal.js";
import type { ToolManifestEntry } from "./mcp.js";
import { AgentMemory, HOST_MEMORY_ACTIONS, MEMORY_AUTHORING_RULES } from "./memory.js";
import { reviewMemory } from "./memory-review.js";
import { NodeScienceProcessRuntime } from "./process-runner.js";
import { recordedAgent } from "./recorded-agent.js";
import { ResearchEnvironment } from "./research-environment.js";
import { SettingsStore } from "./settings-store.js";

const Id = z.string().min(1).max(2_048);
const Text = z.string().min(1).max(100_000);
const SwarmCall = z.discriminatedUnion("action", [
  z.strictObject({
    action: z.literal("create"),
    id: Id,
    leadAgentId: Id,
    permissions: PermissionRequestSchema.optional(),
  }),
  z.strictObject({ action: z.literal("status"), id: Id.optional() }),
  z.strictObject({ action: z.literal("capabilities"), agentId: Id }),
  z.strictObject({ action: z.literal("new_session"), agentId: Id }),
  z.strictObject({
    action: z.literal("send_message"),
    agentId: Id,
    sessionId: Id.optional(),
    text: Text,
    model: z
      .string()
      .min(1)
      .max(512)
      .regex(/^[^\s-][^\s]*$/u)
      .optional(),
    effort: z.string().min(1).max(64).optional(),
    permissions: PermissionRequestSchema.optional(),
  }),
  z.strictObject({ action: z.literal("cancel"), agentId: Id, sessionId: Id }),
]);

export interface SwarmRecord {
  readonly id: string;
  readonly leadAgentId: string;
  readonly permissions: AgentPermissions;
}

export interface ProductServicesOptions {
  readonly productHome: string;
  readonly cwd: string;
  readonly agents?: AgentRegistry;
  readonly scienceConfig?: ConstructorParameters<typeof ScienceCore>[1];
}

export class ProductServices {
  readonly directoryKey: string;
  readonly mcpSocket: string;
  readonly memory: MemoryService;
  readonly learning: AgentMemory;
  readonly science: ScienceCore;
  readonly journal: ExecutionJournal;
  readonly settings: SettingsStore;
  readonly environment: ResearchEnvironment;
  readonly toolManifest: readonly ToolManifestEntry[];
  readonly agents: AgentRegistry;

  private readonly a2a = new A2AEndpoints();
  private readonly boundAgents = new Map<string, NativeAgent>();
  private readonly members = new Map<string, NativeAgent>();
  private readonly ownsAgents: boolean;
  private readonly bridges = new Map<string, AgUiBridge>();
  private readonly scienceDefinitions: ReadonlyMap<string, ScienceToolDefinition>;
  private readonly scienceDisposers: Array<() => Promise<void>> = [];
  private readonly swarms = new Map<string, SwarmRecord>();
  private agentOptions?: AgentOptions;
  private closed = false;
  private readonly shutdown = new AbortController();
  private readonly toolOperations = new Set<Promise<unknown>>();
  readonly mcpExecutions = new Map<string, { sessionId: string; runId: string } | null>();
  private permissionChecks = 0;

  private constructor(readonly options: ProductServicesOptions) {
    this.agents = options.agents ?? new AgentRegistry();
    this.ownsAgents = options.agents === undefined;
    this.directoryKey = createHash("sha256").update(options.cwd).digest("hex").slice(0, 12);
    this.mcpSocket = join(options.productHome, "mcp", `${randomBytes(6).toString("hex")}.sock`);
    this.settings = new SettingsStore(options.productHome);
    this.environment = new ResearchEnvironment(
      this.settings,
      options.cwd,
      join(options.productHome, "science", "artifacts", "v1", "staging"),
    );
    this.journal = new ExecutionJournal(join(options.productHome, "logs"), this.directoryKey);
    const attachments: ScienceAttachmentStore = {
      saveImage: async ({ data, mediaType, name }) => ({
        attachmentId: `swarmx-inline:${createHash("sha256").update(data).digest("hex")}`,
        mediaType,
        bytes: data.byteLength,
        ...(name === undefined ? {} : { name }),
        inlineData: Buffer.from(data).toString("base64"),
      }),
    };
    this.science = new ScienceCore(
      {
        subprocess: this.environment,
        documentSubprocess: new NodeScienceProcessRuntime(),
        onDispose: (dispose) => this.scienceDisposers.push(dispose),
      },
      {
        ...options.scienceConfig,
        root: join(options.productHome, "science"),
        notebookRuntime: "isolated",
      },
      () => ({ key: this.directoryKey, root: options.cwd }),
    );
    this.memory = new MemoryService({
      root: join(options.productHome, "memory"),
      checkResource: (resource) => {
        try {
          parseScienceResourceId(resource);
          this.science.headResource(this.directoryKey, { id: resource });
          return undefined;
        } catch (error) {
          if (!(error instanceof ScienceError)) throw error;
          if (error.code === "INVALID_RESOURCE_ID") {
            return {
              ruleId: "source.invalid",
              severity: "error",
              message: "Invalid Science resource address.",
            };
          }
          if (
            ["RESOURCE_NOT_FOUND", "RESOURCE_KIND_MISMATCH", "RESOURCE_REVISION_MISMATCH"].includes(
              error.code,
            )
          ) {
            return {
              ruleId: "source.unresolved",
              severity: "warning",
              message:
                "Science source is unavailable in this directory or its revision has changed.",
            };
          }
          throw error;
        }
      },
    });
    const definitions = createScienceToolDefinitions(this.science, attachments);
    this.learning = new AgentMemory(
      options,
      this.memory,
      this.journal,
      this.settings,
      async (prompt, signal, harness) => {
        if (!this.agentOptions) throw new Error("Agents are not attached for memory review.");
        return reviewMemory(this.agentOptions, prompt, signal, harness);
      },
    );
    this.scienceDefinitions = new Map(definitions.map((tool) => [tool.name, tool]));
    this.toolManifest = [
      ...definitions.map((tool) => ({
        name: tool.name,
        description: tool.description,
        inputSchema: (tool.mcpParameters ?? tool.parameters) as Record<string, unknown>,
      })),
      {
        name: "memory",
        description: `Persistent learning: read_core_memory {}; update_core_memory {content,expectedRevision} replaces bounded user notes. search_sessions {query?,sessionId?,limit?} recalls original conversations. Vault: search_memory {query,limit?,includeDeprecated?}; read_memory/load_memory {id} (load includes prerequisites); graph_memory {}; lint_memory {id?,now?}; UI actions for the desktop app: memory_status {}, memory_configure {settings}, memory_review {sessionId,focus?}, memory_decide {id,decision}. create_memory {title,description,type,body,sources?,dependencies?:[{id,revision}]}; update_memory {id,expectedRevision,body?,dependencies?,title?,description?,sources?,status?}; deprecate_memory {id,expectedRevision}. Store reusable procedures as Playbook concepts with explicit prerequisites. Search/read before updating. Never invent evidence. Pending writes are not yet saved; only the user can approve them in Settings. ${MEMORY_AUTHORING_RULES}`,
        inputSchema: {
          type: "object",
          additionalProperties: false,
          properties: {
            action: { type: "string", enum: [...HOST_MEMORY_ACTIONS] },
            request: { type: "object" },
          },
          required: ["action", "request"],
        },
      },
      {
        name: "swarm",
        description: "Create or call recursive Swarms and native Agents.",
        inputSchema: swarmSchema,
      },
    ];
  }

  static async create(options: ProductServicesOptions): Promise<ProductServices> {
    const cwd = await realpath(options.cwd);
    if (!(await stat(cwd)).isDirectory()) throw new Error("Working directory is not a directory.");
    const services = new ProductServices({ ...options, cwd });
    try {
      await services.memory.initialize();
      await services.learning.core.initialize();
      return services;
    } catch (error) {
      await services.dispose();
      throw error;
    }
  }

  async attachAgents(
    origin: string,
    agent?: NativeAgent,
    selected = selectedAgent(),
  ): Promise<void> {
    this.a2a.attach(origin);
    const agentOptions: AgentOptions = {
      cwd: this.options.cwd,
      productHome: this.options.productHome,
      mcp: {
        command: process.execPath,
        args: [fileURLToPath(new URL("./mcp-bridge.js", import.meta.url))],
        env: {
          ELECTRON_RUN_AS_NODE: "1",
          SWARMX_MCP_SOCKET: this.mcpSocket,
        },
      },
      executionPolicy: () => ({ ...this.settings.read().policy, ...this.currentPermissions() }),
      productTools: {
        definitions: this.toolManifest,
        call: (name, args, callId, signal) => {
          const context = this.journal.scope.getStore();
          if (!context?.sessionId) throw new Error("Product tools require an active execution.");
          return this.callTool(name, args, {
            actorId: "pi",
            callId,
            signal,
            sessionId: context.sessionId,
            runId: context.runId,
          });
        },
      },
      registerMcp: (token) => {
        this.mcpExecutions.set(token, null);
        return {
          bind: (sessionId, runId) => {
            const active = this.journal.activeSession(sessionId);
            if (active?.runId !== runId) throw new Error("MCP requires an active Host execution.");
            this.mcpExecutions.set(token, { sessionId, runId });
          },
          dispose: () => {
            this.mcpExecutions.delete(token);
          },
        };
      },
    };
    this.agentOptions = agentOptions;
    if (agent)
      this.boundAgents.set(
        selected,
        this.protectAgent(
          selected,
          recordedAgent(this.journal, selected, bindAgent(agent, agentOptions), this.learning),
        ),
      );
    await this.createSwarm("swarm", selected);
  }

  get rootAgent(): NativeAgent {
    const agent = this.members.get("swarm");
    if (!agent) throw new Error("Swarm is not attached.");
    return agent;
  }

  updatePolicy(raw: unknown) {
    if (this.busy) throw new Error("Stop active executions before changing permissions.");
    const policy = ExecutionPolicySchema.parse(raw);
    return this.settings.write({ ...this.settings.read(), policy });
  }

  get busy(): boolean {
    return (
      this.permissionChecks > 0 ||
      this.toolOperations.size > 0 ||
      this.environment.busy ||
      this.learning.busy ||
      this.journal.activeRuns().length > 0
    );
  }

  get availableAgents() {
    return [
      "swarm",
      ...AGENT_IDS.filter((id) => this.currentPermissions().harnesses[id] !== undefined),
    ];
  }

  get defaultHarness() {
    const swarm = this.swarms.get("swarm");
    if (!swarm) throw new Error("Swarm is not attached.");
    return swarm.leadAgentId;
  }

  async agent(id: string): Promise<NativeAgent> {
    const existing = this.members.get(id);
    if (existing) return existing;
    const nativeId = selectedAgent(id);
    const attached = this.boundAgents.get(nativeId);
    if (attached) {
      this.members.set(id, attached);
      return attached;
    }
    const binding = this.agentOptions;
    if (!binding) throw new Error("Agents are not attached.");
    const bound = bindAgent(await this.agents.agent(nativeId, binding), binding);
    bound.restoreEmptySessions?.(this.journal.emptySessions(nativeId));
    const protectedAgent = this.protectAgent(
      nativeId,
      recordedAgent(this.journal, nativeId, bound, this.learning),
    );
    this.boundAgents.set(nativeId, protectedAgent);
    this.members.set(id, protectedAgent);
    return protectedAgent;
  }

  async agUi(id: string): Promise<AgUiBridge> {
    let bridge = this.bridges.get(id);
    if (!bridge) {
      bridge = new AgUiBridge(await this.agent(id));
      this.bridges.set(id, bridge);
    }
    return bridge;
  }

  async cancelAgUi(id: string, threadId: string): Promise<void> {
    await this.bridges.get(id)?.cancel(threadId);
  }

  listSwarms(): SwarmRecord[] {
    return structuredClone([...this.swarms.values()]);
  }

  a2aCard(id: string): unknown {
    return this.a2a.card(id);
  }

  hasA2A(id: string): boolean {
    return this.a2a.has(id);
  }

  handleA2A(id: string, body: Record<string, unknown>, version: string): Promise<unknown> {
    return this.a2a.handle(id, body, version);
  }

  async callTool(
    name: string,
    args: unknown,
    context: {
      readonly actorId: string;
      readonly callId: string;
      readonly signal: AbortSignal;
      readonly sessionId?: string | undefined;
      readonly runId?: string | undefined;
    },
  ): Promise<unknown> {
    this.assertOpen();
    context = { ...context, signal: AbortSignal.any([context.signal, this.shutdown.signal]) };
    const operation = this.journal.tool(name, args, context, async () => {
      context.signal.throwIfAborted();
      const science = this.scienceDefinitions.get(name);
      if (name === "memory" || science !== undefined) {
        const readOnlyMemory =
          name === "memory" && z.object({ action: z.string() }).parse(args).action;
        const readOnly =
          name === "science_query" ||
          Boolean(
            readOnlyMemory &&
              [
                "read_core_memory",
                "search_sessions",
                "search_memory",
                "read_memory",
                "load_memory",
                "graph_memory",
                "lint_memory",
              ].includes(readOnlyMemory),
          );
        const grant =
          `${name === "memory" ? "memory" : "science"}.${readOnly ? "read" : "write"}` as const;
        if (!this.currentPermissions().tools.includes(grant))
          throw new Error(`Product tool permission "${grant}" is required.`);
      }
      if (science !== undefined) return science.invoke(args, context);
      if (name === "swarm") return this.callSwarm(args, context.signal);
      if (name === "memory") {
        return this.learning.call(args, context);
      }
      throw new Error(`Unknown SwarmX product tool "${name}".`);
    });
    this.toolOperations.add(operation);
    try {
      return await operation;
    } finally {
      this.toolOperations.delete(operation);
    }
  }

  async dispose(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    this.shutdown.abort(new Error("Product services are closing."));

    const results = await Promise.allSettled([
      this.environment.close(),
      this.learning.close(),
      ...(this.ownsAgents ? [this.agents.dispose()] : []),
    ]);
    await Promise.allSettled([...this.toolOperations]);
    results.push(
      ...(await Promise.allSettled(this.scienceDisposers.splice(0).map((dispose) => dispose()))),
    );
    this.journal.close();
    const failure = results.find((result) => result.status === "rejected");
    if (failure?.status === "rejected") throw failure.reason;
  }

  private currentPermissions(): AgentPermissions {
    const policy = policyPermissions(this.settings.read().policy);
    const parent = this.journal.scope.getStore()?.permissions;
    return parent ? intersectPermissions(policy, parent) : policy;
  }

  private protectAgent(id: string, agent: NativeAgent): NativeAgent {
    const admissions = new Map<string, Set<AbortController>>();
    const assertRun = (sessionId: string, expectedRunId?: string) => {
      if (
        expectedRunId !== undefined &&
        this.journal.activeSession(sessionId)?.runId !== expectedRunId
      )
        throw new Error("This execution is no longer active. Refresh its status.");
    };
    const run = async <T>(
      request: PermissionRequest | undefined,
      execute: (permissions: AgentPermissions) => Promise<T>,
      sessionId?: string,
    ) => {
      this.assertOpen();
      this.permissionChecks += 1;
      try {
        const captured = id === "swarm" ? undefined : this.swarms.get(id)?.permissions;
        const current = captured
          ? intersectPermissions(this.currentPermissions(), captured)
          : this.currentPermissions();
        const harness = HarnessSchema.safeParse(id);
        const resolve = (ceiling: AgentPermissions) => {
          const saved =
            sessionId === undefined ? undefined : this.journal.sessionPermissions(sessionId);
          const permissions = narrowPermissions(
            saved ? intersectPermissions(ceiling, saved) : ceiling,
            request,
          );
          if (harness.success) {
            const models = permissions.harnesses[harness.data];
            if (models === undefined || models?.length === 0)
              throw new Error(`Harness "${id}" is not permitted.`);
          }
          return permissions;
        };
        const permissions = resolve(current);
        const context = this.journal.scope.getStore() ?? {
          sessionId: null,
          runId: randomUUID(),
          causedBy: null,
          attributes: {},
        };
        return await this.journal.scope.run({ ...context, permissions }, () =>
          execute(permissions),
        );
      } finally {
        this.permissionChecks -= 1;
      }
    };
    return {
      name: agent.name,
      capabilities: agent.capabilities,
      permissions: (session) =>
        run(undefined, async (permissions) => agent.permissions?.(session) ?? permissions, session),
      models: async (session) =>
        run(
          undefined,
          async (permissions) => {
            const catalog = await agent.models(session);
            const harness = HarnessSchema.safeParse(id);
            const allowed = harness.success ? permissions.harnesses[harness.data] : null;
            if (!allowed) return catalog;
            return {
              ...catalog,
              models: catalog.models.filter((model) => allowed.includes(model.id)),
              current:
                catalog.current.model && allowed.includes(catalog.current.model)
                  ? catalog.current
                  : { ...(catalog.current.mode ? { mode: catalog.current.mode } : {}) },
            };
          },
          session,
        ),
      list: async () => run(undefined, () => agent.list()),
      create: async (options) =>
        run(options?.permissions, () =>
          agent.create(
            options?.instructions === undefined
              ? undefined
              : { instructions: options.instructions },
          ),
        ),
      read: async (session, observer) =>
        run(undefined, () => agent.read(session, observer), session),
      start: async (session, text, observer, options) => {
        this.journal.assertCurrentPermissions(session);
        const admission = new AbortController();
        const pending = admissions.get(session) ?? new Set<AbortController>();
        pending.add(admission);
        admissions.set(session, pending);
        try {
          return await run(
            options?.permissions,
            async (permissions) => {
              if (admission.signal.aborted) return { stopReason: "cancelled" };
              const harness = HarnessSchema.safeParse(id);
              const allowed = harness.success ? permissions.harnesses[harness.data] : null;
              if (allowed && (!options?.model || !allowed.includes(options.model)))
                throw new Error(`An explicit permitted model is required for harness "${id}".`);
              if (
                harness.success &&
                (options?.model !== undefined ||
                  options?.effort !== undefined ||
                  options?.mode !== undefined)
              ) {
                const catalog = await agent.models(session);
                if (admission.signal.aborted) return { stopReason: "cancelled" };
                const model = catalog.models.find(
                  (model) => model.id === (options?.model ?? catalog.current.model),
                );
                if (options?.model !== undefined && !model)
                  throw new Error("Select an advertised native model.");
                if (
                  options?.effort !== undefined &&
                  !model?.efforts.some((effort) => effort.id === options.effort)
                )
                  throw new Error("Select an advertised native reasoning effort.");
                if (
                  options?.mode !== undefined &&
                  !catalog.modes?.some((mode) => mode.id === options.mode)
                )
                  throw new Error("Select an advertised native mode.");
              }
              const { permissions: _permissions, ...selection } = options ?? {};
              return agent.start(session, text, observer, harness.success ? selection : options);
            },
            session,
          );
        } finally {
          pending.delete(admission);
          if (!pending.size) admissions.delete(session);
        }
      },
      steer: (session, text, expectedRunId) =>
        run(
          undefined,
          async (permissions) => {
            assertRun(session, expectedRunId);
            const active = this.journal.activeSession(session)?.permissions;
            if (active) narrowPermissions(permissions, active);
            return agent.steer(session, text);
          },
          session,
        ),
      interrupt: async (session, expectedRunId) => {
        assertRun(session, expectedRunId);
        for (const admission of admissions.get(session) ?? []) admission.abort();
        await agent.interrupt(session);
      },
      dispose: () => agent.dispose(),
    };
  }

  private async createSwarm(
    id: string,
    leadAgentId: string,
    request?: PermissionRequest,
  ): Promise<void> {
    if (this.members.has(id) || AGENT_IDS.includes(id as AgentId))
      throw new Error(`Agent "${id}" already exists.`);
    const permissions = narrowPermissions(this.currentPermissions(), request);
    const lead = await this.agent(leadAgentId);
    this.journal.append(this.journal.scope.getStore() ?? null, {
      type: EventType.CUSTOM,
      name: "swarmx.swarm.created",
      value: { id, leadAgentId, permissions },
    });
    this.swarms.set(id, { id, leadAgentId, permissions });
    const swarm = this.protectAgent(id, createSwarm(id, lead));
    this.members.set(id, swarm);
    this.a2a.add(id, id, new SwarmA2AExecutor(swarm, this.journal, id));
  }

  private async callSwarm(raw: unknown, signal: AbortSignal): Promise<unknown> {
    const call = SwarmCall.parse(raw);
    if (!["status", "cancel"].includes(call.action) && !this.currentPermissions().delegation)
      throw new Error("This execution does not have delegation permission.");
    if (call.action === "create") {
      await this.createSwarm(call.id, call.leadAgentId, call.permissions);
      return this.swarms.get(call.id);
    }
    if (call.action === "status")
      return call.id === undefined ? this.listSwarms() : this.swarms.get(call.id);
    const agent = await this.agent(call.agentId);
    if (call.action === "capabilities")
      return publicCapabilities(agent.capabilities, !!agent.permissions);
    if (call.action === "new_session") return { sessionId: await agent.create() };
    if (call.action === "cancel") {
      await agent.interrupt(call.sessionId);
      return { sessionId: call.sessionId, cancellationRequested: true };
    }
    const sessionId = call.sessionId ?? (await agent.create({ permissions: call.permissions }));
    const interact = this.journal.scope.getStore()?.interact;
    const text: string[] = [];
    signal.throwIfAborted();
    let cancellation: Promise<void> | undefined;
    const abort = () => {
      cancellation = agent.interrupt(sessionId);
      // Observe immediately; the operation's finally block propagates cancellation failures.
      void cancellation.catch(() => {});
    };
    signal.addEventListener("abort", abort, { once: true });
    try {
      const result = await agent.start(
        sessionId,
        call.text,
        {
          text: (_id, value, role = "assistant") => {
            if (role === "assistant") text.push(value);
          },
          tool() {},
          raw() {},
          interact: async (request, signal) => {
            if (!interact)
              throw new Error("Delegated Agent needs a connected user for interaction.");
            return interact(
              {
                ...request,
                id: `${sessionId}:${request.id}`,
                title: `${call.agentId} · ${sessionId} — ${request.title}`,
              },
              signal,
            );
          },
        },
        { model: call.model, effort: call.effort, permissions: call.permissions },
      );
      return { sessionId, text: text.join(""), result };
    } finally {
      signal.removeEventListener("abort", abort);
      await cancellation;
    }
  }

  private assertOpen(): void {
    if (this.closed) throw new Error("Product services are closed.");
  }
}

const swarmSchema: Record<string, unknown> = {
  oneOf: [
    {
      type: "object",
      additionalProperties: false,
      properties: {
        action: { const: "create" },
        id: { type: "string" },
        leadAgentId: { type: "string" },
        permissions: z.toJSONSchema(PermissionRequestSchema),
      },
      required: ["action", "id", "leadAgentId"],
    },
    {
      type: "object",
      additionalProperties: false,
      properties: {
        action: { enum: ["status", "capabilities", "new_session", "send_message", "cancel"] },
        id: { type: "string" },
        agentId: { type: "string" },
        sessionId: { type: "string" },
        text: { type: "string" },
        model: { type: "string" },
        effort: { type: "string" },
        permissions: z.toJSONSchema(PermissionRequestSchema),
      },
      required: ["action"],
    },
  ],
};
