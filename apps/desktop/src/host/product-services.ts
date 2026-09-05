import { createHash } from "node:crypto";
import { lstatSync } from "node:fs";
import { mkdir, rename, rmdir } from "node:fs/promises";
import { join } from "node:path";
import { DvcService } from "@swarmx/dvc";
import { MEMORY_ACTIONS, MemoryService } from "@swarmx/memory";
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
import { AGENT_IDS, type AgentId, loadAgent, selectedAgent } from "../agent.js";
import type { AgentOptions, NativeAgent } from "../agents/types.js";
import { A2AEndpoints, SwarmA2AExecutor } from "./a2a.js";
import { AgUiBridge } from "./ag-ui.js";
import type { ToolManifestEntry } from "./mcp.js";
import { NodeProcessRunner, NodeScienceProcessRuntime } from "./process-runner.js";

const Id = z.string().min(1).max(2_048);
const Text = z.string().min(1).max(100_000);
const SwarmCall = z.discriminatedUnion("action", [
  z.strictObject({ action: z.literal("create"), id: Id, leadAgentId: Id }),
  z.strictObject({ action: z.literal("status"), id: Id.optional() }),
  z.strictObject({ action: z.literal("new_session"), agentId: Id }),
  z.strictObject({
    action: z.literal("send_message"),
    agentId: Id,
    sessionId: Id.optional(),
    text: Text,
  }),
  z.strictObject({ action: z.literal("cancel"), agentId: Id, sessionId: Id }),
]);
const MemoryCall = z.strictObject({
  action: z.enum(MEMORY_ACTIONS),
  request: z.unknown(),
  approved: z.boolean().optional(),
});

export interface Workspace {
  readonly id: string;
  readonly label: string;
  readonly root: string;
}

export interface SwarmRecord {
  readonly id: string;
  readonly leadAgentId: string;
}

export interface ProductServicesOptions {
  readonly productHome: string;
  readonly workspace: Workspace;
  readonly scienceConfig?: ConstructorParameters<typeof ScienceCore>[1];
}

export class ProductServices {
  readonly dvc = new DvcService(new NodeProcessRunner());
  readonly memory: MemoryService;
  readonly science: ScienceCore;
  readonly toolManifest: readonly ToolManifestEntry[];

  private readonly a2a = new A2AEndpoints();
  private readonly agents = new Map<string, NativeAgent>();
  private readonly loading = new Map<AgentId, Promise<NativeAgent>>();
  private readonly bridges = new Map<string, AgUiBridge>();
  private readonly scienceDefinitions: ReadonlyMap<string, ScienceToolDefinition>;
  private readonly scienceDisposers: Array<() => Promise<void>> = [];
  private readonly swarms = new Map<string, SwarmRecord>();
  private agentOptions?: AgentOptions;
  private closed = false;

  private constructor(readonly options: ProductServicesOptions) {
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
        subprocess: new NodeScienceProcessRuntime(),
        onDispose: (dispose) => this.scienceDisposers.push(dispose),
      },
      { ...options.scienceConfig, root: join(options.productHome, "science") },
      () => ({ key: options.workspace.id, root: options.workspace.root }),
    );
    this.memory = new MemoryService({
      root: join(options.productHome, "memory", "vault"),
      checkResource: (resource) => {
        try {
          parseScienceResourceId(resource);
          this.science.headResource(options.workspace.id, { id: resource });
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
                "Science source is unavailable in this workspace or its revision has changed.",
            };
          }
          throw error;
        }
      },
    });
    const definitions = createScienceToolDefinitions(this.science, attachments);
    this.scienceDefinitions = new Map(definitions.map((tool) => [tool.name, tool]));
    this.toolManifest = [
      ...definitions.map((tool) => ({
        name: tool.name,
        description: tool.description,
        inputSchema: (tool.mcpParameters ?? tool.parameters) as Record<string, unknown>,
      })),
      {
        name: "memory",
        description:
          "Search, read, curate, or lint the private SwarmX memory. lint_memory accepts optional id and ISO datetime now; search_memory accepts includeDeprecated.",
        inputSchema: {
          type: "object",
          additionalProperties: false,
          properties: {
            action: { type: "string", enum: [...MEMORY_ACTIONS] },
            request: { type: "object" },
            approved: { type: "boolean" },
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
    const previousDirectory = join(options.productHome, "knowledge-base");
    const previousVault = join(previousDirectory, "vault");
    if (lstatSync(previousVault, { throwIfNoEntry: false })) {
      const directory = join(options.productHome, "memory");
      const vault = join(directory, "vault");
      if (lstatSync(vault, { throwIfNoEntry: false })) {
        throw new Error(
          "Both previous and current Memory vaults exist; consolidate them before starting SwarmX.",
        );
      }
      await mkdir(directory, { mode: 0o700, recursive: true });
      await rename(previousVault, vault);
      await rmdir(previousDirectory);
    }
    const services = new ProductServices(options);
    await services.memory.initialize();
    return services;
  }

  async attachAgents(
    origin: string,
    token: string,
    agent?: NativeAgent,
    selected = selectedAgent(),
  ): Promise<void> {
    this.a2a.attach(origin);
    this.agentOptions = {
      cwd: this.options.workspace.root,
      mcp: { url: `${origin}/mcp`, headers: { authorization: `Bearer ${token}` } },
    };
    if (agent) this.agents.set(selected, agent);
    await this.createSwarm("swarm", selected);
  }

  get rootAgent(): NativeAgent {
    const agent = this.agents.get("swarm");
    if (!agent) throw new Error("Swarm is not attached.");
    return agent;
  }

  get availableAgents() {
    return ["swarm", ...AGENT_IDS];
  }

  async agent(id: string): Promise<NativeAgent> {
    const existing = this.agents.get(id);
    if (existing) return existing;
    const nativeId = selectedAgent(id);
    if (!this.agentOptions) throw new Error("Agents are not attached.");
    let pending = this.loading.get(nativeId);
    if (!pending) {
      pending = loadAgent(nativeId, this.agentOptions);
      this.loading.set(nativeId, pending);
    }
    const agent = await pending;
    this.agents.set(id, agent);
    return agent;
  }

  async agUi(id: string): Promise<AgUiBridge> {
    let bridge = this.bridges.get(id);
    if (!bridge) {
      bridge = new AgUiBridge(await this.agent(id));
      this.bridges.set(id, bridge);
    }
    return bridge;
  }

  listSwarms(): SwarmRecord[] {
    return [...this.swarms.values()].map((record) => ({ ...record }));
  }

  a2aCard(id: string): unknown {
    return this.a2a.card(id);
  }

  handleA2A(id: string, body: Record<string, unknown>, version: string): Promise<unknown> {
    return this.a2a.handle(id, body, version);
  }

  async callTool(
    name: string,
    args: unknown,
    context: { readonly actorId: string; readonly callId: string; readonly signal: AbortSignal },
  ): Promise<unknown> {
    this.assertOpen();
    const science = this.scienceDefinitions.get(name);
    if (science !== undefined) return science.invoke(args, context);
    if (name === "swarm") return this.callSwarm(args, context.signal);
    if (name === "memory") {
      const call = MemoryCall.parse(args);
      return this.memory.execute(
        { action: call.action, request: call.request },
        {
          ...context,
          workspaceRoot: this.options.workspace.root,
          approve: async () => (call.approved === true ? "allowed-once" : "rejected"),
        },
      );
    }
    throw new Error(`Unknown SwarmX product tool "${name}".`);
  }

  async dispose(): Promise<void> {
    if (this.closed) return;
    this.closed = true;

    const results = await Promise.allSettled([
      this.dvc.close(),
      ...[...this.agents.entries()]
        .filter(([id]) => !this.swarms.has(id))
        .map(([, agent]) => agent.dispose()),
      ...this.scienceDisposers.splice(0).map((dispose) => dispose()),
    ]);
    const failure = results.find((result) => result.status === "rejected");
    if (failure?.status === "rejected") throw failure.reason;
  }

  private async createSwarm(id: string, leadAgentId: string): Promise<void> {
    if (this.agents.has(id) || AGENT_IDS.includes(id as AgentId))
      throw new Error(`Agent "${id}" already exists.`);
    const swarm = createSwarm(id, await this.agent(leadAgentId));
    this.agents.set(id, swarm);
    this.swarms.set(id, { id, leadAgentId });
    this.a2a.add(id, id, new SwarmA2AExecutor(swarm));
  }

  private async callSwarm(raw: unknown, signal: AbortSignal): Promise<unknown> {
    const call = SwarmCall.parse(raw);
    if (call.action === "create") {
      await this.createSwarm(call.id, call.leadAgentId);
      return this.swarms.get(call.id);
    }
    if (call.action === "status")
      return call.id === undefined ? this.listSwarms() : this.swarms.get(call.id);
    const agent = await this.agent(call.agentId);
    if (call.action === "new_session") return { sessionId: await agent.create() };
    if (call.action === "cancel") {
      await agent.interrupt(call.sessionId);
      return { sessionId: call.sessionId, cancelled: true };
    }
    const sessionId = call.sessionId ?? (await agent.create());
    const text: string[] = [];
    signal.throwIfAborted();
    const abort = () => void agent.interrupt(sessionId);
    signal.addEventListener("abort", abort, { once: true });
    try {
      await agent.start(sessionId, call.text, {
        text: (_id, value, role = "assistant") => {
          if (role === "assistant") text.push(value);
        },
        tool() {},
        raw() {},
        interact: async () => {
          throw new Error(
            "Delegated Agent needs user interaction; open its native session in the browser.",
          );
        },
      });
      return { sessionId, text: text.join("") };
    } finally {
      signal.removeEventListener("abort", abort);
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
      },
      required: ["action", "id", "leadAgentId"],
    },
    {
      type: "object",
      additionalProperties: false,
      properties: {
        action: { enum: ["status", "new_session", "send_message", "cancel"] },
        id: { type: "string" },
        agentId: { type: "string" },
        sessionId: { type: "string" },
        text: { type: "string" },
      },
      required: ["action"],
    },
  ],
};
