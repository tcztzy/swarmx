import { createHash, randomUUID } from "node:crypto";
import { realpathSync } from "node:fs";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { pathToFileURL } from "node:url";
import { Context } from "@deepseek-ai/cordis";
import LocalSubprocessRuntime from "@deepseek-ai/dsh-subprocess-local";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import { ListToolsRequestSchema, ToolSchema } from "@modelcontextprotocol/sdk/types.js";
import { loadImage } from "@napi-rs/canvas";
import { createPkbToolDefinition, executePkbOperation, PkbVault } from "@swarmx/dsh-pkb";
import { ScienceCore } from "@swarmx/dsh-science/core";
import { createScienceToolDefinitions } from "@swarmx/dsh-science/tools";
import {
  isMutatingMemberTool,
  type KnowledgeCommitter,
  leadToolGuard,
  memberAgentOptions,
  memberToolGuard,
  OwnerKnowledgeCommitter,
  SwarmCoordinator,
  SwarmError,
  SwarmJournal,
  SwarmMemberStartupError,
  type SwarmRuntimeAdapter,
} from "@swarmx/dsh-swarm";
import { createSwarmToolDefinition } from "@swarmx/dsh-swarm/tools";
import { z } from "zod";
import { RuntimeBridgeClient, RuntimeBridgeMemberStartupError } from "../bridge.js";
import type { ConversationItem, ConversationSnapshot, ConversationSummary } from "../contracts.js";
import { MAX_SCIENCE_CARRIER_CONFIG_BYTES, parseScienceCarrierConfig } from "../science-config.js";
import { CodexMemberBindingConflictError, CodexMemberBindingStore } from "./member-bindings.js";

const EnvironmentSchema = z.object({
  SWARMX_BRIDGE_TOKEN: z.string().min(1),
  SWARMX_BRIDGE_URL: z.string().url(),
  SWARMX_HOME: z.string().min(1),
  SWARMX_SCIENCE_CONFIG: z.string().max(MAX_SCIENCE_CARRIER_CONFIG_BYTES).optional(),
  SWARMX_WORKSPACE_ID: z.string().min(1),
  SWARMX_WORKSPACE_LABEL: z.string().min(1),
  SWARMX_WORKSPACE_ROOT: z.string().min(1),
});
const CodexMcpMetadataSchema = z.object({ threadId: z.string().min(1).max(512) }).passthrough();
const InlineImageSchema = z
  .object({
    attachmentId: z.string().startsWith("swarmx-mcp-inline:"),
    mediaType: z.enum(["image/png", "image/jpeg", "image/webp", "image/gif"]),
    bytes: z
      .number()
      .int()
      .positive()
      .max(2 * 1024 * 1024),
    name: z.string().max(512).optional(),
    inlineData: z.string().max(3 * 1024 * 1024),
  })
  .passthrough();
const MAX_CONVERSATION_SEARCH_CANDIDATES = 32;
const MAX_CONVERSATION_SEARCH_BYTES = 1024 * 1024;
const MAX_CONVERSATION_ITEM_CHARACTERS = 16_000;
const MEMBER_CREATION_TIMEOUT_MS = 30_000;
const MEMBER_CLEANUP_TIMEOUT_MS = 30_000;

interface ProductMcpHost {
  server: McpServer;
  dispose(): Promise<void>;
}

interface ProductToolDefinition {
  name: string;
  description: string;
  parameters: unknown;
  mcpParameters?: unknown;
  execute(arguments_: unknown, execution: ProductToolExecution): Promise<unknown>;
}

interface ProductToolExecution {
  readonly actorId: string;
  readonly callId: string;
  readonly conversationId?: string;
  readonly identityKind: "native-thread" | "session" | "call";
  readonly signal: AbortSignal;
}

interface ProductToolIdentity {
  readonly authorityScope: string;
  readonly bridge: RuntimeBridgeClient;
  readonly nonce: string;
  readonly releaseCallActor: (actorId: string) => void;
  readonly workspaceId: string;
  invocation: number;
}

export async function createProductMcpHost(
  rawEnvironment: Readonly<Record<string, string | undefined>> = process.env,
): Promise<ProductMcpHost> {
  const environment = EnvironmentSchema.parse(rawEnvironment);
  const scienceConfig = parseScienceCarrierConfig(environment.SWARMX_SCIENCE_CONFIG);
  const workspaceRoot = realpathSync.native(environment.SWARMX_WORKSPACE_ROOT);
  const bridge = new RuntimeBridgeClient(
    environment.SWARMX_BRIDGE_URL,
    environment.SWARMX_BRIDGE_TOKEN,
  );
  const context = new Context();
  new LocalSubprocessRuntime(context);
  const scienceDisposers: Array<() => Promise<void>> = [];
  let disposeServer: () => Promise<void> = async () => {};
  let disposeSwarm: () => Promise<void> = async () => {};
  try {
    const science = new ScienceCore(
      {
        subprocess: context.subprocess,
        onDispose: (dispose) => scienceDisposers.push(dispose),
      },
      { ...scienceConfig, root: join(environment.SWARMX_HOME, "science") },
      () => ({
        key: environment.SWARMX_WORKSPACE_ID,
        root: workspaceRoot,
      }),
    );
    const vault = new PkbVault({ root: join(environment.SWARMX_HOME, "pkb", "vault") });
    await vault.initialize();
    const archive = new BridgeConversationArchive(bridge, vault, environment.SWARMX_WORKSPACE_ID);
    const server = new McpServer({ name: "swarmx", version: "0.1.0" });
    disposeServer = () => server.close();
    const approval = {
      request: async (input: { reason: string; signal: AbortSignal }) => {
        const response = await server.server.elicitInput(
          {
            mode: "form",
            message: input.reason,
            requestedSchema: {
              type: "object",
              properties: {
                confirm: {
                  type: "boolean",
                  title: "Confirm",
                  description: "Allow this one PKB change?",
                  default: false,
                },
              },
              required: ["confirm"],
            },
          },
          { signal: input.signal },
        );
        return response.action === "accept" && response.content?.confirm === true
          ? "allowed-once"
          : response.action === "cancel"
            ? "cancelled"
            : "rejected";
      },
    };
    const swarm = new CodexSwarmHost(
      bridge,
      workspaceRoot,
      environment.SWARMX_WORKSPACE_ID,
      join(environment.SWARMX_HOME, "swarm"),
      new OwnerKnowledgeCommitter({
        science,
        pkb: { vault },
        workspaceRoot: () => workspaceRoot,
        approval: {
          request: ({ reason, signal }) => approval.request({ reason, signal }),
        },
      }),
    );
    disposeSwarm = () => swarm.dispose();
    const identity: ProductToolIdentity = {
      authorityScope: swarm.authorityScope,
      bridge,
      nonce: randomUUID(),
      releaseCallActor: (actorId) => swarm.releaseCallActor(actorId),
      workspaceId: environment.SWARMX_WORKSPACE_ID,
      invocation: 0,
    };
    const scienceDefinitions = createScienceToolDefinitions(science, {
      saveImage: async ({ data, mediaType, name }) => {
        const dimensions = await imageDimensions(data, mediaType);
        return {
          attachmentId: `swarmx-mcp-inline:${createHash("sha256").update(data).digest("hex")}`,
          mediaType,
          bytes: data.byteLength,
          ...dimensions,
          ...(name === undefined ? {} : { name }),
          inlineData: Buffer.from(data).toString("base64"),
        } as never;
      },
    });
    const pkbDefinition = createPkbToolDefinition({
      approval: approval as never,
      archive,
      vault,
    });
    const swarmDefinition = createSwarmToolDefinition(swarm.service);
    const definitions: ProductToolDefinition[] = [
      ...scienceDefinitions.map(
        (definition): ProductToolDefinition => ({
          name: definition.name,
          description: definition.description,
          parameters: definition.parameters,
          ...(definition.mcpParameters === undefined
            ? {}
            : { mcpParameters: definition.mcpParameters }),
          execute: (arguments_, execution) =>
            swarm.invokeProductTool(definition.name, execution, (active) =>
              definition.invoke(arguments_, active),
            ),
        }),
      ),
      {
        name: pkbDefinition.name,
        description: pkbDefinition.description,
        parameters: pkbDefinition.parameters,
        execute: (arguments_, execution) =>
          swarm.invokeProductTool(
            pkbDefinition.name,
            execution,
            (active) =>
              executePkbOperation({ archive, vault }, arguments_, {
                ...active,
                workspaceRoot,
                approve: (reason) => approval.request({ reason, signal: active.signal }),
              }),
            arguments_,
          ),
      },
      {
        name: swarmDefinition.name,
        description: swarmDefinition.description,
        parameters: swarmDefinition.parameters,
        execute: async (arguments_, execution) => {
          const action =
            typeof arguments_ === "object" &&
            arguments_ !== null &&
            !Array.isArray(arguments_) &&
            "action" in arguments_
              ? arguments_.action
              : undefined;
          if (execution.identityKind === "call" && action !== "status") {
            throw new Error(
              "Durable Codex Swarm actions require exact native Thread or transport session metadata.",
            );
          }
          const actor = swarm.actorFor(execution.actorId, execution.conversationId);
          await swarm.initialize(actor);
          return swarmDefinition.invoke(arguments_, {
            ...execution,
            actor,
          });
        },
      },
    ];
    for (const tool of definitions) {
      registerDefinition(server, tool, identity);
    }
    const publishedTools = definitions.map((definition) =>
      ToolSchema.parse({
        name: definition.name,
        description: definition.description,
        inputSchema: definition.mcpParameters ?? definition.parameters,
      }),
    );
    server.server.setRequestHandler(ListToolsRequestSchema, () => ({ tools: publishedTools }));
    return {
      server,
      dispose: () =>
        disposeInOrder([
          () => server.close(),
          () => swarm.dispose(),
          ...scienceDisposers,
          () => context.fiber.dispose(),
        ]),
    };
  } catch (cause) {
    try {
      await disposeInOrder([
        disposeServer,
        disposeSwarm,
        ...scienceDisposers,
        () => context.fiber.dispose(),
      ]);
    } catch (cleanupError) {
      throw new AggregateError([cause, cleanupError], "Codex product MCP startup failed");
    }
    throw cause;
  }
}

export async function disposeInOrder(disposers: readonly (() => Promise<void>)[]): Promise<void> {
  const failures: unknown[] = [];
  for (const dispose of disposers) {
    try {
      await dispose();
    } catch (error) {
      failures.push(error);
    }
  }
  if (failures.length > 0) throw failures[0];
}

export async function connectProductMcpTransport(
  server: Pick<McpServer, "connect">,
  transport: Transport,
  dispose: () => Promise<void>,
): Promise<void> {
  try {
    await server.connect(transport);
  } catch (cause) {
    try {
      await dispose();
    } catch (cleanupError) {
      throw new AggregateError(
        [cause, cleanupError],
        "Codex product MCP transport startup and cleanup failed",
      );
    }
    throw cause;
  }
}

function registerDefinition(
  server: McpServer,
  definition: ProductToolDefinition,
  identity: ProductToolIdentity,
): void {
  server.registerTool(
    definition.name,
    {
      description: definition.description,
      inputSchema: z.fromJSONSchema(
        (definition.mcpParameters ?? definition.parameters) as Parameters<
          typeof z.fromJSONSchema
        >[0],
      ),
    },
    async (arguments_, extra) => {
      const requestId = String(extra.requestId);
      const callId = `mcp:${createHash("sha256")
        .update(JSON.stringify([identity.nonce, typeof extra.requestId, requestId]))
        .digest("hex")}`;
      identity.invocation += 1;
      const nativeThreadId = CodexMcpMetadataSchema.safeParse(extra._meta).data?.threadId;
      const conversationId = nativeThreadId === undefined ? undefined : `codex:${nativeThreadId}`;
      if (conversationId !== undefined) {
        const snapshot = await identity.bridge.request<ConversationSnapshot>(
          { action: "read", conversationId },
          extra.signal,
        );
        if (
          snapshot.conversationId !== conversationId ||
          snapshot.workspace.id !== identity.workspaceId
        ) {
          throw new Error("Codex native Thread does not belong to this MCP workspace.");
        }
        if (snapshot.archived) throw new Error("Codex native Thread is archived.");
      }
      const identityKind =
        nativeThreadId !== undefined
          ? ("native-thread" as const)
          : extra.sessionId !== undefined
            ? ("session" as const)
            : ("call" as const);
      const execution = {
        actorId:
          nativeThreadId !== undefined
            ? `codex-mcp-thread:${createHash("sha256")
                .update(JSON.stringify([identity.authorityScope, nativeThreadId]))
                .digest("hex")}`
            : extra.sessionId === undefined
              ? `codex-mcp-call:${identity.nonce}:${String(identity.invocation)}:${typeof extra.requestId}:${requestId}`
              : `codex-mcp-session:${createHash("sha256")
                  .update(JSON.stringify([identity.authorityScope, extra.sessionId]))
                  .digest("hex")}`,
        callId,
        ...(conversationId === undefined ? {} : { conversationId }),
        identityKind,
        signal: extra.signal,
      };
      try {
        const value = await definition.execute(arguments_, execution);
        const rendered = mcpContent(value);
        return { content: rendered };
      } finally {
        if (identityKind === "call") identity.releaseCallActor(execution.actorId);
      }
    },
  );
}

function mcpContent(value: unknown) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return [{ type: "text" as const, text: JSON.stringify(value) }];
  }
  const data = "data" in value ? value.data : undefined;
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    return [{ type: "text" as const, text: JSON.stringify(value) }];
  }
  const parsed = InlineImageSchema.safeParse("attachment" in data ? data.attachment : undefined);
  if (!parsed.success) {
    return [{ type: "text" as const, text: JSON.stringify(value) }];
  }
  const { inlineData, attachmentId: _attachmentId, ...metadata } = parsed.data;
  const sanitized = {
    ...value,
    data: { ...data, attachment: metadata },
  };
  return [
    { type: "text" as const, text: JSON.stringify(sanitized) },
    { type: "image" as const, data: inlineData, mimeType: parsed.data.mediaType },
  ];
}

async function imageDimensions(
  data: Uint8Array,
  mediaType: "image/png" | "image/jpeg" | "image/webp" | "image/gif",
): Promise<{ width: number; height: number }> {
  const buffer = Buffer.from(data);
  const signatureMatches =
    (mediaType === "image/png" &&
      buffer.length >= 8 &&
      buffer.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) ||
    (mediaType === "image/gif" &&
      buffer.length >= 6 &&
      ["GIF87a", "GIF89a"].includes(buffer.subarray(0, 6).toString("ascii"))) ||
    (mediaType === "image/jpeg" &&
      buffer.length >= 3 &&
      buffer.subarray(0, 3).equals(Buffer.from([0xff, 0xd8, 0xff]))) ||
    (mediaType === "image/webp" &&
      buffer.length >= 12 &&
      buffer.subarray(0, 4).toString("ascii") === "RIFF" &&
      buffer.subarray(8, 12).toString("ascii") === "WEBP");
  if (!signatureMatches) {
    throw new Error(`Science produced malformed ${mediaType} image data.`);
  }
  let decoded: Awaited<ReturnType<typeof loadImage>>;
  try {
    decoded = await loadImage(buffer);
  } catch (cause) {
    throw new Error(`Science produced malformed ${mediaType} image data.`, { cause });
  }
  const dimensions = { width: decoded.width, height: decoded.height };
  if (
    !Number.isSafeInteger(dimensions.width) ||
    !Number.isSafeInteger(dimensions.height) ||
    dimensions.width <= 0 ||
    dimensions.height <= 0
  ) {
    throw new Error(`Science produced malformed ${mediaType} image data.`);
  }
  return dimensions;
}

type CoordinatorAgent = Parameters<SwarmCoordinator["create"]>[0];

interface CodexSwarmActor {
  id: string;
  status: "running" | "idle";
  origin: "lead" | "member";
  options?: { provider?: string; model?: string; maxTokens?: number };
  conversationId?: string;
  activeTurnId?: string;
  idle?: Promise<void>;
  cancel(input: { kind: "hook"; reason: string }): Promise<void>;
  whenIdle(): Promise<void>;
}

class CodexMemberLifecycleError extends Error {}

class CodexSwarmHost {
  readonly service: ReturnType<CodexSwarmHost["createService"]>;
  readonly authorityScope: string;
  private readonly journal: SwarmJournal;
  private readonly coordinator: SwarmCoordinator;
  private readonly actors = new Map<string, CodexSwarmActor>();
  private readonly bindingStore: CodexMemberBindingStore;
  private readonly lifetime = new AbortController();
  private readonly operations = new Set<Promise<unknown>>();
  private readonly initializations = new Map<string, Promise<void>>();
  private closed = false;

  constructor(
    private readonly bridge: RuntimeBridgeClient,
    private readonly workspaceRoot: string,
    private readonly workspaceId: string,
    journalRoot: string,
    knowledge: KnowledgeCommitter,
  ) {
    this.journal = new SwarmJournal(journalRoot, { mode: "client" });
    this.authorityScope = this.journal.workspaceKey(this.workspaceRoot);
    try {
      this.bindingStore = new CodexMemberBindingStore(this.journal, this.authorityScope);
      const runtime: SwarmRuntimeAdapter = {
        followupWithoutParent: true,
        exact: (candidate) => this.currentActor(String(candidate.id)) === candidate,
        getActor: (id) => this.currentActor(id) as unknown as CoordinatorAgent | undefined,
        isSubagent: (candidate) => (candidate as unknown as CodexSwarmActor).origin === "member",
        modelOptions: (candidate) => (candidate as unknown as CodexSwarmActor).options ?? {},
        workspaceKey: () => this.authorityScope,
        inject: (target, content) => this.deliver(target as unknown as CodexSwarmActor, content),
        followup: async (_parent, targetId, content, _senderId, signal) => {
          await this.deliver(this.requireActor(targetId), content, signal);
        },
        followupRoot: (target, content) =>
          this.deliver(target as unknown as CodexSwarmActor, content),
        interrupt: (_parent, targetId) => this.interrupt(this.requireActor(targetId)),
        stopContinuable: async (_parent, targetId) => {
          const binding = this.bindingStore.get(targetId);
          if (binding === undefined) return;
          await this.bridge.request(
            {
              action: "archive",
              conversationId: binding.conversationId,
              memberId: targetId,
            },
            this.lifetime.signal,
          );
          if (!this.bindingStore.retireForArchive(String(_parent.id), binding)) {
            this.bindingStore.release(binding);
          }
          this.actors.delete(targetId);
        },
        startContinuable: async (_parent, request) => {
          if (
            request.agentOptions?.provider !== undefined &&
            request.agentOptions.provider !== "codex" &&
            request.agentOptions.provider !== "openai"
          ) {
            throw new SwarmMemberStartupError(
              `Codex Swarm cannot route provider "${request.agentOptions.provider}". Use "codex", "openai", or omit provider.`,
              "absent",
            );
          }
          const activeSignal = AbortSignal.any([this.lifetime.signal, request.signal]);
          let creation: {
            claim: "archive_required" | "created" | "existing" | "unclaimed";
            conversation: ConversationSummary;
          };
          try {
            creation = await this.bridge.request(
              {
                action: "create_member",
                teamId: String(_parent.id),
                memberId: request.childId,
                ...(request.agentOptions?.model === undefined
                  ? {}
                  : { model: request.agentOptions.model }),
              },
              AbortSignal.any([
                this.lifetime.signal,
                AbortSignal.timeout(MEMBER_CREATION_TIMEOUT_MS),
              ]),
            );
          } catch (cause) {
            if (
              cause instanceof RuntimeBridgeMemberStartupError &&
              cause.handleState === "absent"
            ) {
              throw new SwarmMemberStartupError(cause.message, "absent", { cause });
            }
            throw cause;
          }
          const conversation = creation.conversation;
          let claimed = false;
          try {
            const actor = this.actor(request.childId, "member", conversation.conversationId, {
              ...(request.agentOptions?.provider === undefined
                ? {}
                : { provider: request.agentOptions.provider }),
              ...(request.agentOptions?.model === undefined
                ? {}
                : { model: request.agentOptions.model }),
              ...(request.agentOptions?.maxTokens === undefined
                ? {}
                : { maxTokens: request.agentOptions.maxTokens }),
            });
            this.actors.set(actor.id, actor);
            const binding = { id: actor.id, conversationId: conversation.conversationId };
            const claim =
              creation.claim === "unclaimed"
                ? this.bindingStore.claimProvisioning(String(_parent.id), binding)
                : creation.claim;
            if (creation.claim !== "unclaimed") {
              const durable = this.bindingStore.get(actor.id);
              if (durable?.conversationId !== conversation.conversationId) {
                throw new CodexMemberBindingConflictError(
                  "Codex Swarm root member claim is no longer current.",
                  "handle",
                );
              }
            }
            claimed = true;
            if (claim === "archive_required") {
              await this.bridge.request(
                {
                  action: "archive",
                  conversationId: conversation.conversationId,
                  memberId: request.childId,
                },
                AbortSignal.timeout(MEMBER_CLEANUP_TIMEOUT_MS),
              );
              if (!this.bindingStore.retireForArchive(String(_parent.id), binding)) {
                throw new Error("Codex Swarm archive acknowledgement became stale.");
              }
              claimed = false;
              throw new CodexMemberBindingConflictError(
                "Swarm archive started while the member Thread was provisioning.",
                "handle",
              );
            }
            activeSignal.throwIfAborted();
            const started = await this.bridge.request<{ turnId: string }>(
              {
                action: "send",
                conversationId: conversation.conversationId,
                text: request.prompt,
              },
              activeSignal,
            );
            actor.activeTurnId = started.turnId;
            actor.status = "running";
            this.startWatcher(actor);
            return actor.id;
          } catch (cause) {
            this.actors.delete(request.childId);
            if (
              !claimed &&
              cause instanceof CodexMemberBindingConflictError &&
              cause.kind === "handle"
            ) {
              throw new SwarmMemberStartupError(cause.message, "absent", { cause });
            }
            try {
              await this.bridge.request(
                {
                  action: "archive",
                  conversationId: conversation.conversationId,
                  memberId: request.childId,
                },
                AbortSignal.timeout(MEMBER_CLEANUP_TIMEOUT_MS),
              );
              if (claimed) {
                const binding = {
                  id: request.childId,
                  conversationId: conversation.conversationId,
                };
                if (!this.bindingStore.retireForArchive(String(_parent.id), binding)) {
                  this.bindingStore.release(binding);
                }
              }
            } catch (cleanupError) {
              throw new AggregateError(
                [cause, cleanupError],
                "Codex Swarm member provisioning and rollback failed",
              );
            }
            throw new SwarmMemberStartupError(
              cause instanceof Error ? cause.message : "Codex Swarm member startup failed",
              "absent",
              { cause },
            );
          }
        },
      };
      this.coordinator = new SwarmCoordinator(
        this.journal,
        runtime,
        {
          maxMembers: 8,
          maxMessageBytes: 16 * 1024,
          maxPendingMessagesPerMember: 32,
          maxTasks: 256,
          quiescenceTimeoutMs: 30_000,
        },
        knowledge,
      );
      this.service = this.createService();
    } catch (error) {
      this.journal.close();
      throw error;
    }
  }

  async initialize(actor: CoordinatorAgent): Promise<void> {
    const actorId = String(actor.id);
    const existing = this.initializations.get(actorId);
    if (existing !== undefined) return existing;
    const initialization = this.run(undefined, async () => {
      const current = actor as unknown as CodexSwarmActor;
      if (current.origin === "member") {
        await this.synchronizeObserved(current);
        await this.coordinator.recoverMember(current as unknown as CoordinatorAgent);
        return;
      }
      if (!this.coordinator.isLeadIdentity(actorId)) return;
      await this.synchronize(current);
      for (const memberActor of this.hydrateTeamActors(actorId)) {
        try {
          await this.synchronizeObserved(memberActor);
        } catch (error) {
          if (!(error instanceof CodexMemberLifecycleError)) throw error;
        }
      }
      await this.coordinator.recoverMember(current as unknown as CoordinatorAgent);
    });
    this.initializations.set(actorId, initialization);
    try {
      await initialization;
    } finally {
      if (this.initializations.get(actorId) === initialization)
        this.initializations.delete(actorId);
    }
  }

  async invokeProductTool(
    toolName: string,
    execution: ProductToolExecution,
    invoke: (active: ProductToolExecution) => Promise<unknown>,
    toolArguments?: unknown,
  ): Promise<unknown> {
    return this.run(execution.signal, async (signal) => {
      const mutating = isMutatingMemberTool(toolName, toolArguments);
      const actor = this.actorFor(execution.actorId, execution.conversationId);
      await this.initialize(actor);
      const team = this.journal.findByParticipant(String(actor.id));
      if (team === undefined || team.workspaceKey !== this.authorityScope) {
        return invoke({ ...execution, signal });
      }
      if (team.archiveStartedAt !== undefined) {
        throw new SwarmError("Swarm archive is in progress", "SWARM_ARCHIVED");
      }
      const profile = team.members.find((member) => member.id === String(actor.id));
      const member = this.coordinator.isMemberIdentity(String(actor.id));
      const lead =
        this.coordinator.isLeadIdentity(String(actor.id)) ||
        (team.phase === "archived" && team.id === String(actor.id) && profile?.role === "lead");
      if (!lead && !member) {
        throw new SwarmError("Swarm participant is inactive", "SWARM_UNAUTHORIZED");
      }
      const memberRole =
        profile?.role === undefined || profile.role === "lead" ? "legacy" : profile.role;
      const denied = member
        ? memberToolGuard(
            actor,
            this.coordinator,
            { agent: actor, mutating, name: toolName },
            memberRole,
          )
        : leadToolGuard(actor, this.coordinator, { agent: actor, mutating, name: toolName });
      if (denied !== undefined) {
        if (member) this.coordinator.recordRoleToolViolation(actor, toolName);
        throw new SwarmError(denied, "SWARM_UNAUTHORIZED");
      }

      const active = { ...execution, signal };
      const effect = mutating
        ? this.coordinator.beginToolEffect(actor, execution.callId, toolName)
        : undefined;
      let settled = false;
      try {
        const value = await invoke(active);
        if (effect !== undefined) {
          const aborted = signal.aborted;
          this.coordinator.settleToolEffect(actor, effect.id, {
            status: aborted ? "uncertain" : "succeeded",
            ...(aborted
              ? {}
              : {
                  resultDigest: `sha256:${createHash("sha256")
                    .update(JSON.stringify(value) ?? "null")
                    .digest("hex")}`,
                }),
          });
          settled = true;
          signal.throwIfAborted();
        }
        return value;
      } catch (cause) {
        if (effect !== undefined && !settled) {
          try {
            this.coordinator.settleToolEffect(actor, effect.id, { status: "uncertain" });
          } catch (settlementError) {
            throw new AggregateError(
              [cause, settlementError],
              "Codex Team Tool execution and effect settlement failed",
            );
          }
        }
        throw cause;
      }
    });
  }

  actorFor(id: string, conversationId?: string): CoordinatorAgent {
    if (this.closed) throw new Error("Codex Swarm MCP is closed.");
    if (conversationId !== undefined) {
      const binding = this.bindingStore.findByConversation(conversationId);
      if (binding !== undefined) {
        return this.bindMemberActor(
          binding.id,
          binding.conversationId,
        ) as unknown as CoordinatorAgent;
      }
    }
    let actor = this.actors.get(id);
    if (actor === undefined) {
      actor = this.actor(id, "lead", conversationId);
      this.actors.set(id, actor);
    } else if (conversationId !== undefined) {
      if (actor.conversationId !== undefined && actor.conversationId !== conversationId) {
        throw new Error(`Codex Swarm actor "${id}" changed its native conversation identity.`);
      }
      actor.conversationId = conversationId;
    }
    return actor as unknown as CoordinatorAgent;
  }

  releaseCallActor(id: string): void {
    const actor = this.actors.get(id);
    if (actor?.origin === "lead" && actor.conversationId === undefined) {
      this.actors.delete(id);
    }
  }

  async dispose(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    this.lifetime.abort(new Error("Codex Swarm MCP disposed."));
    await Promise.allSettled([...this.operations]);
    await Promise.allSettled([...this.actors.values()].map((actor) => actor.idle).filter(Boolean));
    this.journal.close();
  }

  private createService() {
    const mutate = <T>(
      agent: CoordinatorAgent,
      signal: AbortSignal | undefined,
      operation: (activeSignal: AbortSignal) => Promise<T> | T,
    ) =>
      this.run(signal, async (activeSignal) => {
        const actor = agent as unknown as CodexSwarmActor;
        if (actor.origin === "lead") this.hydrateTeamActors(String(agent.id));
        await operation(activeSignal);
        activeSignal.throwIfAborted();
        return this.coordinator.snapshot(agent);
      });
    return {
      create: (
        agent: CoordinatorAgent,
        request: Parameters<SwarmCoordinator["create"]>[1],
        signal?: AbortSignal,
      ) => mutate(agent, signal, () => this.coordinator.create(agent, request)),
      snapshot: (agent: CoordinatorAgent) =>
        this.run(undefined, () => this.coordinator.snapshot(agent)),
      addMember: (
        agent: CoordinatorAgent,
        request: Parameters<SwarmCoordinator["addMember"]>[1],
        signal?: AbortSignal,
      ) => mutate(agent, signal, (active) => this.coordinator.addMember(agent, request, active)),
      sendMessage: (
        agent: CoordinatorAgent,
        request: Parameters<SwarmCoordinator["sendMessage"]>[1],
        signal?: AbortSignal,
      ) => this.run(signal, (active) => this.coordinator.sendMessage(agent, request, active)),
      createTask: (
        agent: CoordinatorAgent,
        request: Parameters<SwarmCoordinator["createTask"]>[1],
        signal?: AbortSignal,
      ) => mutate(agent, signal, () => this.coordinator.createTask(agent, request)),
      updateTask: (
        agent: CoordinatorAgent,
        request: Parameters<SwarmCoordinator["updateTask"]>[1],
        signal?: AbortSignal,
      ) => mutate(agent, signal, () => this.coordinator.updateTask(agent, request)),
      submitTask: (
        agent: CoordinatorAgent,
        request: Parameters<SwarmCoordinator["submitTask"]>[1],
        signal?: AbortSignal,
      ) => mutate(agent, signal, () => this.coordinator.submitTask(agent, request)),
      startVerification: (
        agent: CoordinatorAgent,
        request: Parameters<SwarmCoordinator["startVerification"]>[1],
        signal?: AbortSignal,
      ) => mutate(agent, signal, () => this.coordinator.startVerification(agent, request)),
      recordVerdict: (
        agent: CoordinatorAgent,
        request: Parameters<SwarmCoordinator["recordVerdict"]>[1],
        signal?: AbortSignal,
      ) => mutate(agent, signal, () => this.coordinator.recordVerdict(agent, request)),
      recordMonitorFinding: (
        agent: CoordinatorAgent,
        request: Parameters<SwarmCoordinator["recordSemanticFinding"]>[1],
        signal?: AbortSignal,
      ) => mutate(agent, signal, () => this.coordinator.recordSemanticFinding(agent, request)),
      escalateTask: (
        agent: CoordinatorAgent,
        request: Parameters<SwarmCoordinator["escalateTask"]>[1],
        signal?: AbortSignal,
      ) => mutate(agent, signal, () => this.coordinator.escalateTask(agent, request)),
      reassignTask: (
        agent: CoordinatorAgent,
        request: Parameters<SwarmCoordinator["reassignTask"]>[1],
        signal?: AbortSignal,
      ) => mutate(agent, signal, () => this.coordinator.reassignTask(agent, request)),
      interruptMember: (
        agent: CoordinatorAgent,
        request: Parameters<SwarmCoordinator["interruptMember"]>[1],
        signal?: AbortSignal,
      ) => mutate(agent, signal, () => this.coordinator.interruptMember(agent, request)),
      admitKnowledge: (
        agent: CoordinatorAgent,
        request: Parameters<SwarmCoordinator["admitKnowledge"]>[1],
        callId: string,
        signal?: AbortSignal,
      ) =>
        this.run(signal, (active) =>
          this.coordinator.admitKnowledge(agent, request, { callId, signal: active }),
        ),
      resolveEffect: (
        agent: CoordinatorAgent,
        request: Parameters<SwarmCoordinator["resolveEffect"]>[1],
        signal?: AbortSignal,
      ) => mutate(agent, signal, () => this.coordinator.resolveEffect(agent, request)),
      waitForChange: (
        agent: CoordinatorAgent,
        request: { afterRevision: number; timeoutMs?: number },
        signal: AbortSignal,
      ) =>
        this.run(signal, async (active) => {
          const deadline = Date.now() + (request.timeoutMs ?? 30_000);
          for (;;) {
            active.throwIfAborted();
            const snapshot = await this.coordinator.snapshot(agent);
            if (snapshot.revision > request.afterRevision || Date.now() >= deadline)
              return snapshot;
            await delay(Math.min(250, Math.max(1, deadline - Date.now())), undefined, {
              signal: active,
            });
          }
        }),
      archive: (agent: CoordinatorAgent, signal?: AbortSignal) =>
        mutate(agent, signal, () => this.coordinator.archive(agent)),
    };
  }

  private async run<T>(
    signal: AbortSignal | undefined,
    operation: (activeSignal: AbortSignal) => Promise<T> | T,
  ): Promise<T> {
    if (this.closed) throw new Error("Codex Swarm MCP is closed.");
    const activeSignal =
      signal === undefined ? this.lifetime.signal : AbortSignal.any([this.lifetime.signal, signal]);
    const owned = Promise.resolve().then(() => {
      activeSignal.throwIfAborted();
      return operation(activeSignal);
    });
    this.operations.add(owned);
    try {
      return await owned;
    } finally {
      this.operations.delete(owned);
    }
  }

  private actorByMemberId(id: string): CodexSwarmActor | undefined {
    const current = this.currentActor(id);
    if (current !== undefined) return current;
    if (!z.string().uuid().safeParse(id).success) return undefined;
    const binding = this.bindingStore.get(id);
    if (binding === undefined || !this.coordinator.isMemberIdentity(id)) return undefined;
    return this.bindMemberActor(binding.id, binding.conversationId);
  }

  private hydrateTeamActors(participantId: string): CodexSwarmActor[] {
    const team = this.journal.findByParticipant(participantId);
    if (team?.phase !== "active" || team.workspaceKey !== this.authorityScope) return [];
    return team.members.flatMap((member) => {
      if (
        member.role === "lead" ||
        (member.phase !== "active" && member.phase !== "provisioning")
      ) {
        return [];
      }
      const actor = this.actorByMemberId(member.id);
      return actor === undefined ? [] : [actor];
    });
  }

  private bindMemberActor(id: string, conversationId: string): CodexSwarmActor {
    const binding = this.bindingStore.get(id);
    if (binding?.conversationId !== conversationId || !this.coordinator.isMemberIdentity(id)) {
      this.actors.delete(id);
      throw new Error(`Codex Swarm member "${id}" is unavailable.`);
    }
    const current = this.currentActor(id);
    if (current !== undefined) {
      if (current.origin !== "member" || current.conversationId !== conversationId) {
        throw new Error(`Codex Swarm member "${id}" changed its native Thread identity.`);
      }
      return current;
    }
    const team = this.journal.findByParticipant(id);
    const member = team?.members.find((candidate) => candidate.id === id);
    if (
      team?.phase !== "active" ||
      team.workspaceKey !== this.authorityScope ||
      member === undefined ||
      member.role === "lead" ||
      (member.phase !== "active" && member.phase !== "provisioning")
    ) {
      throw new Error(`Codex Swarm member "${id}" is unavailable.`);
    }
    const actor = this.actor(id, "member", conversationId, memberAgentOptions(member));
    this.actors.set(id, actor);
    return actor;
  }

  private currentActor(id: string): CodexSwarmActor | undefined {
    const actor = this.actors.get(id);
    if (actor?.origin !== "member") return actor;
    const binding = this.bindingStore.get(id);
    if (
      binding?.conversationId !== actor.conversationId ||
      !this.coordinator.isMemberIdentity(id)
    ) {
      this.actors.delete(id);
      return undefined;
    }
    return actor;
  }

  private actor(
    id: string,
    origin: "lead" | "member",
    conversationId?: string,
    options?: CodexSwarmActor["options"],
  ): CodexSwarmActor {
    const actor: CodexSwarmActor = {
      id,
      origin,
      status: "idle",
      ...(conversationId === undefined ? {} : { conversationId }),
      ...(options === undefined ? {} : { options }),
      cancel: () => this.interrupt(actor),
      whenIdle: () => actor.idle ?? Promise.resolve(),
    };
    return actor;
  }

  private requireActor(id: string): CodexSwarmActor {
    const actor = this.actorByMemberId(id);
    if (actor === undefined) throw new Error(`Codex Swarm member "${id}" is unavailable.`);
    return actor;
  }

  private requireConversation(actor: CodexSwarmActor): string {
    if (actor.conversationId === undefined) {
      throw new Error(`Codex Swarm actor "${actor.id}" has no native conversation handle.`);
    }
    return actor.conversationId;
  }

  private async deliver(
    actor: CodexSwarmActor,
    content: string,
    signal?: AbortSignal,
  ): Promise<void> {
    const conversationId = this.requireConversation(actor);
    const activeSignal =
      signal === undefined ? this.lifetime.signal : AbortSignal.any([this.lifetime.signal, signal]);
    if (actor.status !== "running" || actor.activeTurnId === undefined) {
      await this.synchronizeObserved(actor, activeSignal);
    }
    if (actor.status === "running" && actor.activeTurnId !== undefined) {
      try {
        await this.bridge.request(
          {
            action: "steer",
            conversationId,
            turnId: actor.activeTurnId,
            text: content,
          },
          activeSignal,
        );
      } catch (error) {
        return this.rethrowAfterLifecycleObservation(actor, error);
      }
      return;
    }
    let started: { turnId: string };
    try {
      started = await this.bridge.request<{ turnId: string }>(
        { action: "send", conversationId, text: content },
        activeSignal,
      );
    } catch (error) {
      return this.rethrowAfterLifecycleObservation(actor, error);
    }
    actor.activeTurnId = started.turnId;
    actor.status = "running";
    this.startWatcher(actor);
  }

  private async interrupt(actor: CodexSwarmActor): Promise<void> {
    if (actor.conversationId === undefined) return;
    if (actor.activeTurnId === undefined || actor.status !== "running") {
      await this.synchronizeObserved(actor, this.lifetime.signal);
    }
    if (actor.activeTurnId === undefined || actor.status !== "running") return;
    try {
      await this.bridge.request(
        {
          action: "interrupt",
          conversationId: actor.conversationId,
          turnId: actor.activeTurnId,
        },
        this.lifetime.signal,
      );
    } catch (error) {
      return this.rethrowAfterLifecycleObservation(actor, error);
    }
  }

  private async synchronizeObserved(actor: CodexSwarmActor, signal?: AbortSignal): Promise<void> {
    try {
      await this.synchronize(actor, signal);
    } catch (error) {
      if (error instanceof CodexMemberLifecycleError && actor.origin === "member") {
        await this.coordinator.recordMemberLifecycleFailure(actor.id);
      }
      throw error;
    }
  }

  private async rethrowAfterLifecycleObservation(
    actor: CodexSwarmActor,
    cause: unknown,
  ): Promise<never> {
    try {
      await this.synchronizeObserved(actor, this.lifetime.signal);
    } catch (error) {
      if (error instanceof CodexMemberLifecycleError) throw error;
    }
    throw cause;
  }

  private async synchronize(actor: CodexSwarmActor, signal?: AbortSignal): Promise<void> {
    if (actor.conversationId === undefined) return;
    const activeSignal =
      signal === undefined ? this.lifetime.signal : AbortSignal.any([this.lifetime.signal, signal]);
    const snapshot = await this.bridge.request<ConversationSnapshot>(
      { action: "read", conversationId: actor.conversationId },
      activeSignal,
    );
    this.assertActorSnapshot(actor, snapshot);
    const running = snapshot.turns.find((turn) => turn.status === "running");
    if (running === undefined) {
      actor.status = "idle";
      delete actor.activeTurnId;
      return;
    }
    actor.status = "running";
    actor.activeTurnId = running.id;
    this.startWatcher(actor);
  }

  private startWatcher(actor: CodexSwarmActor): void {
    if (actor.idle !== undefined) return;
    let handled: Promise<void>;
    handled = this.watchIdle(actor)
      .then(undefined, async () => {
        if (!this.lifetime.signal.aborted) {
          await this.coordinator.recordMemberLifecycleFailure(actor.id);
        }
      })
      .catch(() => undefined)
      .finally(() => {
        if (actor.idle === handled) delete actor.idle;
      });
    actor.idle = handled;
  }

  private async watchIdle(actor: CodexSwarmActor): Promise<void> {
    if (actor.conversationId === undefined) return;
    for (;;) {
      this.lifetime.signal.throwIfAborted();
      const snapshot = await this.bridge.request<ConversationSnapshot>(
        { action: "read", conversationId: actor.conversationId },
        this.lifetime.signal,
      );
      this.assertActorSnapshot(actor, snapshot);
      const running = snapshot.turns.find((turn) => turn.status === "running");
      if (running === undefined) {
        actor.status = "idle";
        delete actor.activeTurnId;
        return;
      }
      actor.activeTurnId = running.id;
      await delay(250, undefined, { signal: this.lifetime.signal });
    }
  }

  private assertActorSnapshot(actor: CodexSwarmActor, snapshot: ConversationSnapshot): void {
    const conversationId = this.requireConversation(actor);
    if (snapshot.conversationId !== conversationId) {
      throw new CodexMemberLifecycleError(
        `Codex Swarm actor "${actor.id}" changed its native Thread identity.`,
      );
    }
    if (snapshot.workspace.id !== this.workspaceId) {
      throw new CodexMemberLifecycleError(
        `Codex Swarm actor "${actor.id}" moved outside its authorized workspace.`,
      );
    }
    if (snapshot.archived) {
      throw new CodexMemberLifecycleError(
        `Codex Swarm actor "${actor.id}" native Thread is archived.`,
      );
    }
  }
}

class BridgeConversationArchive {
  constructor(
    private readonly bridge: RuntimeBridgeClient,
    private readonly vault: PkbVault,
    private readonly workspaceId: string,
  ) {}

  async search(
    _cwd: string,
    request: { query: string; limit?: number; scope?: "all" | "workspace" },
    signal?: AbortSignal,
  ) {
    const conversations = await this.bridge.request<ConversationSummary[]>(
      { action: "list" },
      signal,
    );
    const query = request.query.toLocaleLowerCase();
    const eligible = conversations.filter(
      (conversation) => request.scope === "all" || conversation.workspace.id === this.workspaceId,
    );
    const candidates = eligible.slice(0, MAX_CONVERSATION_SEARCH_CANDIDATES);
    const matches: ReturnType<typeof indexedItems> = [];
    let scannedBytes = 0;
    let byteLimitReached = false;
    scan: for (const conversation of candidates) {
      signal?.throwIfAborted();
      const snapshot = await this.bridge.request<ConversationSnapshot>(
        { action: "read", conversationId: conversation.conversationId },
        signal,
      );
      for (const indexed of indexedItems(snapshot)) {
        const text = boundedSemanticText(indexed.item);
        const bytes = Buffer.byteLength(text);
        if (scannedBytes + bytes > MAX_CONVERSATION_SEARCH_BYTES) {
          byteLimitReached = true;
          break scan;
        }
        scannedBytes += bytes;
        if (text.toLocaleLowerCase().includes(query)) matches.push(indexed);
      }
    }
    const items = matches
      .sort((left, right) => right.item.createdAt - left.item.createdAt)
      .slice(0, request.limit ?? 20)
      .map(({ conversationId, item, seq }) => ({
        eventTime: new Date(item.createdAt).toISOString(),
        eventType: item.type,
        locator: { seq, sessionId: conversationId },
        snippet: Array.from(boundedSemanticText(item)).slice(0, 800).join(""),
        trust: "untrusted-evidence" as const,
      }));
    const diagnostics = [
      ...(eligible.length > candidates.length
        ? [
            `Conversation search inspected the newest ${String(MAX_CONVERSATION_SEARCH_CANDIDATES)} eligible Threads.`,
          ]
        : []),
      ...(byteLimitReached
        ? [`Conversation search stopped at ${String(MAX_CONVERSATION_SEARCH_BYTES)} bytes.`]
        : []),
    ];
    return { diagnostics, items };
  }

  async read(
    _cwd: string,
    locator: { sessionId: string; seq: number; allAuthorized?: boolean },
    signal?: AbortSignal,
  ) {
    const snapshot = await this.bridge.request<ConversationSnapshot>(
      { action: "read", conversationId: locator.sessionId },
      signal,
    );
    if (!locator.allAuthorized && snapshot.workspace.id !== this.workspaceId) {
      throw new Error("Conversation belongs to another workspace.");
    }
    const selected = indexedItems(snapshot).find(({ seq }) => seq === locator.seq);
    if (selected === undefined) throw new Error("Conversation event not found.");
    return {
      eventTime: new Date(selected.item.createdAt).toISOString(),
      eventType: selected.item.type,
      locator: { seq: selected.seq, sessionId: snapshot.conversationId },
      text: boundedSemanticText(selected.item),
      trust: "untrusted-evidence" as const,
    };
  }

  async capture(
    cwd: string,
    locator: { sessionId: string; seq: number; allAuthorized?: boolean },
    signal?: AbortSignal,
  ) {
    const evidence = await this.read(cwd, locator, signal);
    return this.vault.saveConversationExcerpt(
      cwd,
      {
        eventTime: Date.parse(evidence.eventTime),
        eventType: evidence.eventType,
        seq: evidence.locator.seq,
        sessionId: evidence.locator.sessionId,
        text: evidence.text,
      },
      signal,
    );
  }
}

function indexedItems(snapshot: ConversationSnapshot): Array<{
  conversationId: string;
  item: ConversationItem;
  seq: number;
}> {
  const seen = new Map<number, string>();
  return snapshot.turns
    .flatMap((turn) => turn.items)
    .map((item) => {
      const seq = stableItemSequence(item.id);
      const previous = seen.get(seq);
      if (previous !== undefined && previous !== item.id) {
        throw new Error("Codex conversation item locator collision.");
      }
      if (previous === item.id) {
        throw new Error("Codex conversation contains a duplicate native item identity.");
      }
      seen.set(seq, item.id);
      return { conversationId: snapshot.conversationId, item, seq };
    });
}

function stableItemSequence(itemId: string): number {
  return Number.parseInt(createHash("sha256").update(itemId).digest("hex").slice(0, 13), 16);
}

function boundedSemanticText(item: ConversationItem): string {
  return Array.from(semanticText(item)).slice(0, MAX_CONVERSATION_ITEM_CHARACTERS).join("");
}

function semanticText(item: ConversationItem): string {
  if (item.type === "tool") return `${item.name}\n${item.summary ?? ""}`.trim();
  if (item.type === "error") return item.message;
  return item.text;
}

async function main(): Promise<void> {
  const host = await createProductMcpHost();
  const transport = new StdioServerTransport(process.stdin, process.stdout, {
    maxBufferSize: 2 * 1024 * 1024,
  });
  let disposal: Promise<void> | undefined;
  const dispose = () => (disposal ??= host.dispose());
  const close = () => {
    void dispose().catch((error: unknown) => {
      process.stderr.write(
        `swarmx-mcp cleanup: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
      );
      process.exitCode = 1;
    });
  };
  process.once("SIGINT", close);
  process.once("SIGTERM", close);
  process.stdin.once("end", close);
  process.stdin.once("close", close);
  await connectProductMcpTransport(host.server, transport, dispose);
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  void main().catch((error: unknown) => {
    process.stderr.write(
      `swarmx-mcp: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
    );
    process.exitCode = 1;
  });
}
