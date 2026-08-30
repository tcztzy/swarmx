import { createHash } from "node:crypto";
import { realpathSync } from "node:fs";
import { basename } from "node:path";
import { z } from "zod";
import { ApprovalRegistry } from "../approval.js";
import type {
  ApprovalDecision,
  ApprovalKind,
  ApprovalRequest,
  ApprovalResponse,
  ConversationItem,
  ConversationRuntime,
  ConversationSnapshot,
  ConversationSummary,
  CreateConversationRequest,
  ForkConversationRequest,
  InterruptTurnRequest,
  ReviseConversationRequest,
  RuntimeEvent,
  RuntimeEventListener,
  StartTurnRequest,
  SteerTurnRequest,
  TurnStatus,
  WorkspaceSummary,
} from "../contracts.js";
import { CodexRpcError } from "./connection.js";

const IdSchema = z.string().min(1).max(512);
const NativeItemSchema = z
  .object({
    id: IdSchema,
    type: z.string().min(1).max(128),
  })
  .passthrough();
const NativeTurnSchema = z.object({
  id: IdSchema,
  status: z.enum(["completed", "interrupted", "failed", "inProgress"]),
  startedAt: z.number().int().nullable().optional(),
  completedAt: z.number().int().nullable().optional(),
  error: z.unknown().nullable().optional(),
  items: z.array(NativeItemSchema).max(100_000),
});
const NativeThreadSchema = z.object({
  id: IdSchema,
  cwd: z.string().min(1).max(32_768),
  name: z.string().max(10_000).nullable().optional(),
  preview: z.string().max(100_000).optional().default(""),
  createdAt: z.number().int(),
  updatedAt: z.number().int(),
  archived: z.boolean().optional(),
  historyMode: z.enum(["legacy", "paginated"]).optional().default("legacy"),
  threadSource: z.string().max(512).nullable().optional(),
  turns: z.array(NativeTurnSchema).max(100_000).optional().default([]),
});
const ThreadResponseSchema = z.object({ thread: NativeThreadSchema });
const ThreadListResponseSchema = z.object({
  data: z.array(NativeThreadSchema).max(1000),
  nextCursor: z.string().nullable().optional(),
});
const ThreadIdParamsSchema = z.object({ threadId: IdSchema });
const ThreadStartedParamsSchema = z.object({ thread: z.object({ id: IdSchema }) });
const ThreadStatusParamsSchema = ThreadIdParamsSchema.extend({
  status: z.object({ type: z.string().min(1).max(128) }).passthrough(),
});
const RpcIdSchema = z.union([z.string().min(1).max(512), z.number().int()]);
const ServerRequestResolvedParamsSchema = ThreadIdParamsSchema.extend({ requestId: RpcIdSchema });
const TurnStartResponseSchema = z.object({
  turn: z.object({ id: IdSchema }),
});
const DeltaParamsSchema = z.object({
  threadId: IdSchema,
  turnId: IdSchema,
  itemId: IdSchema,
  delta: z.string().max(2 * 1024 * 1024),
});
const TurnNotificationSchema = z.object({
  threadId: IdSchema,
  turn: NativeTurnSchema,
});
const ItemNotificationSchema = z.object({
  threadId: IdSchema,
  turnId: IdSchema,
  item: NativeItemSchema,
  completedAtMs: z.number().int().optional(),
  startedAtMs: z.number().int().optional(),
});
const UserInputQuestionSchema = z.object({
  id: IdSchema,
  header: z.string().max(100).optional(),
  question: z.string().min(1).max(10_000),
  options: z
    .array(z.object({ label: z.string().min(1).max(1_000) }).passthrough())
    .max(20)
    .nullable()
    .optional(),
});
const ElicitationLabelSchema = z.string().max(10_000);
const ElicitationStringSchema = z
  .strictObject({
    type: z.literal("string"),
    title: ElicitationLabelSchema.optional(),
    description: ElicitationLabelSchema.optional(),
    minLength: z.number().int().nonnegative().optional(),
    maxLength: z.number().int().nonnegative().optional(),
    format: z.enum(["email", "uri", "date", "date-time"]).optional(),
    default: z.string().optional(),
  })
  .refine(
    (value) =>
      value.minLength === undefined ||
      value.maxLength === undefined ||
      value.minLength <= value.maxLength,
    { message: "Elicitation string minLength cannot exceed maxLength." },
  );
const ElicitationNumberSchema = z
  .strictObject({
    type: z.enum(["number", "integer"]),
    title: ElicitationLabelSchema.optional(),
    description: ElicitationLabelSchema.optional(),
    minimum: z.number().finite().optional(),
    maximum: z.number().finite().optional(),
    default: z.number().finite().optional(),
  })
  .refine(
    (value) =>
      value.minimum === undefined || value.maximum === undefined || value.minimum <= value.maximum,
    { message: "Elicitation minimum cannot exceed maximum." },
  );
const ElicitationBooleanSchema = z.strictObject({
  type: z.literal("boolean"),
  title: ElicitationLabelSchema.optional(),
  description: ElicitationLabelSchema.optional(),
  default: z.boolean().optional(),
});
const ElicitationUntitledEnumSchema = z.strictObject({
  type: z.literal("string"),
  title: ElicitationLabelSchema.optional(),
  description: ElicitationLabelSchema.optional(),
  enum: z.array(z.string()).min(1).max(100),
  default: z.string().optional(),
});
const ElicitationLegacyEnumSchema = ElicitationUntitledEnumSchema.extend({
  enumNames: z.array(ElicitationLabelSchema).min(1).max(100),
});
const ElicitationTitledEnumSchema = z.strictObject({
  type: z.literal("string"),
  title: ElicitationLabelSchema.optional(),
  description: ElicitationLabelSchema.optional(),
  oneOf: z
    .array(z.strictObject({ const: z.string(), title: ElicitationLabelSchema }))
    .min(1)
    .max(100),
  default: z.string().optional(),
});
const ElicitationUntitledMultiSelectSchema = z
  .strictObject({
    type: z.literal("array"),
    title: ElicitationLabelSchema.optional(),
    description: ElicitationLabelSchema.optional(),
    minItems: z.number().int().nonnegative().optional(),
    maxItems: z.number().int().nonnegative().optional(),
    items: z.strictObject({
      type: z.literal("string"),
      enum: z.array(z.string()).min(1).max(100),
    }),
    default: z.array(z.string()).max(100).optional(),
  })
  .refine(
    (value) =>
      value.minItems === undefined ||
      value.maxItems === undefined ||
      value.minItems <= value.maxItems,
    { message: "Elicitation minItems cannot exceed maxItems." },
  );
const ElicitationTitledMultiSelectSchema = z
  .strictObject({
    type: z.literal("array"),
    title: ElicitationLabelSchema.optional(),
    description: ElicitationLabelSchema.optional(),
    minItems: z.number().int().nonnegative().optional(),
    maxItems: z.number().int().nonnegative().optional(),
    items: z.strictObject({
      anyOf: z
        .array(z.strictObject({ const: z.string(), title: ElicitationLabelSchema }))
        .min(1)
        .max(100),
    }),
    default: z.array(z.string()).max(100).optional(),
  })
  .refine(
    (value) =>
      value.minItems === undefined ||
      value.maxItems === undefined ||
      value.minItems <= value.maxItems,
    { message: "Elicitation minItems cannot exceed maxItems." },
  );
const ElicitationPropertySchema = z.union([
  ElicitationLegacyEnumSchema,
  ElicitationTitledEnumSchema,
  ElicitationUntitledMultiSelectSchema,
  ElicitationTitledMultiSelectSchema,
  ElicitationBooleanSchema,
  ElicitationStringSchema,
  ElicitationNumberSchema,
  ElicitationUntitledEnumSchema,
]);
const ElicitationPropertyNameSchema = z
  .string()
  .min(1)
  .max(200)
  .refine(
    (value) => value !== "__proto__" && value !== "prototype" && value !== "constructor",
    "Elicitation property name is unsafe.",
  );
const RequestedSchemaSchema = z
  .strictObject({
    $schema: z.string().min(1).max(2_048).url().optional(),
    type: z.literal("object"),
    properties: z.record(ElicitationPropertyNameSchema, ElicitationPropertySchema),
    required: z.array(ElicitationPropertyNameSchema).max(100).optional(),
  })
  .superRefine((value, context) => {
    const keys = Object.keys(value.properties);
    if (keys.length > 100) {
      context.addIssue({ code: "custom", message: "Elicitation form exceeds 100 fields." });
    }
    const required = value.required ?? [];
    if (new Set(required).size !== required.length) {
      context.addIssue({ code: "custom", message: "Elicitation required fields must be unique." });
    }
    for (const key of required) {
      if (!Object.hasOwn(value.properties, key)) {
        context.addIssue({
          code: "custom",
          message: `Elicitation required field "${key}" is not declared.`,
        });
      }
    }
    for (const [key, property] of Object.entries(value.properties)) {
      const schema = property as Record<string, unknown>;
      const itemSchema =
        schema.items !== null && typeof schema.items === "object"
          ? (schema.items as Record<string, unknown>)
          : undefined;
      const optionSets = [
        { label: "enum", path: ["enum"], values: arrayOfStrings(schema.enum) },
        {
          label: "oneOf",
          path: ["oneOf"],
          values: arrayOfRecords(schema.oneOf).map((entry) => entry.const),
        },
        {
          label: "array enum",
          path: ["items", "enum"],
          values: arrayOfStrings(itemSchema?.enum),
        },
        {
          label: "array anyOf",
          path: ["items", "anyOf"],
          values: arrayOfRecords(itemSchema?.anyOf).map((entry) => entry.const),
        },
      ];
      for (const options of optionSets) {
        if (options.values.length > 0 && new Set(options.values).size !== options.values.length) {
          context.addIssue({
            code: "custom",
            path: ["properties", key, ...options.path],
            message: `Elicitation ${options.label} values must be unique.`,
          });
        }
      }
      if (schema.enumNames !== undefined) {
        const enumValues = arrayOfStrings(schema.enum);
        const enumNames = arrayOfStrings(schema.enumNames);
        if (enumNames.length !== enumValues.length) {
          context.addIssue({
            code: "custom",
            path: ["properties", key, "enumNames"],
            message: "Elicitation enumNames must match the enum length.",
          });
        }
      }
      if (schema.default !== undefined) {
        const defaultSchema = z.fromJSONSchema(schema as Parameters<typeof z.fromJSONSchema>[0]);
        if (!defaultSchema.safeParse(schema.default).success) {
          context.addIssue({
            code: "custom",
            path: ["properties", key, "default"],
            message: "Elicitation default must satisfy its field schema.",
          });
        }
      }
    }
  });
const ApprovalParamsSchema = z
  .object({
    threadId: IdSchema,
    turnId: IdSchema.optional(),
    itemId: IdSchema.optional(),
    approvalId: IdSchema.nullable().optional(),
    command: z.string().max(100_000).nullable().optional(),
    reason: z.string().max(100_000).nullable().optional(),
    serverName: z.string().max(512).optional(),
    questions: z.array(UserInputQuestionSchema).max(3).optional(),
    permissions: z.record(z.string(), z.unknown()).optional(),
    mode: z.string().max(32).optional(),
    message: z.string().max(100_000).optional(),
    requestedSchema: RequestedSchemaSchema.optional(),
    availableDecisions: z.array(z.unknown()).max(20).nullable().optional(),
  })
  .passthrough();

const THREAD_LIST_PAGE_SIZE = 100;
const MAX_THREAD_LIST_PAGES = 10;
const MAX_LISTED_THREADS = 1000;
const INTERACTIVE_SOURCE_KINDS = ["cli", "vscode", "appServer"] as const;
const ALL_SOURCE_KINDS = [
  "cli",
  "vscode",
  "exec",
  "appServer",
  "subAgent",
  "subAgentReview",
  "subAgentCompact",
  "subAgentThreadSpawn",
  "subAgentOther",
  "unknown",
] as const;
const MEMBER_THREAD_SOURCE_PREFIX = "swarmx-member:";

type NativeThread = z.infer<typeof NativeThreadSchema>;
type NativeTurn = z.infer<typeof NativeTurnSchema>;
type NativeItem = z.infer<typeof NativeItemSchema>;
type UnsequencedRuntimeEvent = RuntimeEvent extends infer Event
  ? Event extends RuntimeEvent
    ? Omit<Event, "seq" | "runtime">
    : never
  : never;

export interface CodexRpcClient {
  request(method: string, params?: Record<string, unknown>): Promise<unknown>;
  onNotification(method: string, handler: (params: unknown) => void): () => void;
  onRequest(
    method: string,
    handler: (params: Record<string, unknown>, requestId: string | number) => Promise<unknown>,
  ): () => void;
  dispose(): Promise<void>;
}

export interface CodexRuntimeOptions {
  paginatedHistory?: boolean;
}

export class CodexConversationRuntime implements ConversationRuntime {
  readonly kind = "codex" as const;
  private readonly listeners = new Set<RuntimeEventListener>();
  private readonly approvals = new ApprovalRegistry();
  private readonly disposers: Array<() => void> = [];
  private readonly workspaces = new Map<string, WorkspaceSummary>();
  private readonly loadedThreads = new Set<string>();
  private readonly loadingThreads = new Map<string, Promise<void>>();
  private readonly transientThreads = new Map<string, NativeThread>();
  private readonly approvalRequests = new Map<string, ApprovalRequest>();
  private readonly elicitationSchemas = new Map<string, z.infer<typeof RequestedSchemaSchema>>();
  private seq = 0;
  private disposed = false;

  constructor(
    private readonly rpc: CodexRpcClient,
    private readonly options: CodexRuntimeOptions = {},
  ) {
    this.listen("item/agentMessage/delta", (params) => this.emitDelta(params, "assistant_message"));
    this.listen("item/reasoning/textDelta", (params) => this.emitDelta(params, "reasoning"));
    this.listen("item/completed", (params) => this.emitCompleted(params));
    this.listen("turn/started", (params) => this.emitTurn(params));
    this.listen("turn/completed", (params) => this.emitTurn(params));
    this.listen("thread/started", (params) => {
      this.loadedThreads.add(ThreadStartedParamsSchema.parse(params).thread.id);
    });
    this.listen("thread/status/changed", (params) => {
      const value = ThreadStatusParamsSchema.parse(params);
      if (value.status.type === "notLoaded") {
        this.clearThread(value.threadId, "Codex thread unloaded.");
      }
    });
    this.listen("thread/closed", (params) => {
      this.clearThread(ThreadIdParamsSchema.parse(params).threadId, "Codex thread closed.");
    });
    this.listen("thread/deleted", (params) => {
      this.clearThread(ThreadIdParamsSchema.parse(params).threadId, "Codex thread deleted.");
    });
    this.listen("thread/archived", (params) => {
      const { threadId } = ThreadIdParamsSchema.parse(params);
      this.clearThread(threadId, "Codex thread archived.");
    });
    this.listen("serverRequest/resolved", (params) => this.resolveServerRequest(params));
    this.handleApproval("item/commandExecution/requestApproval", "command");
    this.handleApproval("item/fileChange/requestApproval", "file_change");
    this.handleApproval("item/permissions/requestApproval", "permissions");
    this.handleApproval("mcpServer/elicitation/request", "elicitation");
    this.handleApproval("item/tool/requestUserInput", "user_input");
  }

  async list(signal?: AbortSignal): Promise<ConversationSummary[]> {
    throwIfAborted(signal);
    const threads: NativeThread[] = [];
    const seenThreads = new Set<string>();
    const seenCursors = new Set<string>();
    let pageCount = 0;
    let cursor: string | undefined;
    while (threads.length < MAX_LISTED_THREADS) {
      const response = ThreadListResponseSchema.parse(
        await this.rpc.request("thread/list", {
          limit: Math.min(THREAD_LIST_PAGE_SIZE, MAX_LISTED_THREADS - threads.length),
          archived: false,
          sortKey: "updated_at",
          sortDirection: "desc",
          sourceKinds: INTERACTIVE_SOURCE_KINDS,
          ...(cursor === undefined ? {} : { cursor }),
        }),
      );
      pageCount += 1;
      throwIfAborted(signal);
      for (const thread of response.data) {
        if (seenThreads.has(thread.id)) continue;
        seenThreads.add(thread.id);
        threads.push({ ...thread, archived: false });
        if (threads.length === MAX_LISTED_THREADS) break;
      }
      const nextCursor = response.nextCursor ?? null;
      if (nextCursor === null || threads.length === MAX_LISTED_THREADS) break;
      if (pageCount === MAX_THREAD_LIST_PAGES) {
        throw new Error(
          `Codex thread/list exceeded the ${String(MAX_THREAD_LIST_PAGES)} page limit.`,
        );
      }
      if (seenCursors.has(nextCursor)) {
        throw new Error(`Codex thread/list repeated cursor "${nextCursor}".`);
      }
      seenCursors.add(nextCursor);
      cursor = nextCursor;
    }
    const merged = new Map<string, NativeThread>();
    for (const thread of threads) merged.set(thread.id, thread);
    for (const thread of this.transientThreads.values()) {
      if (!merged.has(thread.id)) merged.set(thread.id, thread);
    }
    return [...merged.values()]
      .sort((left, right) => right.updatedAt - left.updatedAt)
      .slice(0, MAX_LISTED_THREADS)
      .map((thread) => this.summary(thread));
  }

  async create(
    request: CreateConversationRequest,
    signal?: AbortSignal,
  ): Promise<ConversationSummary> {
    return this.createThread(request, signal);
  }

  async createProvisionedMember(
    request: CreateConversationRequest,
    provisioningId: string,
    signal?: AbortSignal,
  ): Promise<ConversationSummary> {
    return this.createThread(request, signal, `${MEMBER_THREAD_SOURCE_PREFIX}${provisioningId}`);
  }

  private async createThread(
    request: CreateConversationRequest,
    signal?: AbortSignal,
    threadSource?: string,
  ): Promise<ConversationSummary> {
    throwIfAborted(signal);
    const workspaceRoot = canonicalWorkspaceRoot(request.workspace.root);
    this.workspaces.set(workspaceRoot, {
      id: request.workspace.id,
      label: request.workspace.label,
    });
    const response = ThreadResponseSchema.parse(
      await this.rpc.request("thread/start", {
        cwd: request.workspace.root,
        ...(request.model === undefined ? {} : { model: request.model }),
        approvalPolicy: "on-request",
        ...(threadSource === undefined ? {} : { threadSource }),
        ...(this.options.paginatedHistory === true ? { historyMode: "paginated" } : {}),
      }),
    );
    const created = { ...response.thread, archived: false };
    assertThreadWorkspace(created, workspaceRoot, "created");
    this.loadedThreads.add(created.id);
    this.transientThreads.set(created.id, created);
    throwIfAborted(signal);
    return this.summary(created);
  }

  async read(conversationId: string, signal?: AbortSignal): Promise<ConversationSnapshot> {
    return this.snapshot(await this.readThread(conversationId, signal));
  }

  async start(request: StartTurnRequest, signal?: AbortSignal): Promise<{ turnId: string }> {
    throwIfAborted(signal);
    if (!request.text) throw new Error("Cannot send an empty Codex message.");
    const threadId = nativeId("codex", request.conversationId);
    await this.ensureLoaded(threadId, signal);
    throwIfAborted(signal);
    const response = TurnStartResponseSchema.parse(
      await this.rpc.request("turn/start", {
        threadId,
        input: [{ type: "text", text: request.text }],
        approvalPolicy: "on-request",
      }),
    );
    this.transientThreads.delete(threadId);
    throwIfAborted(signal);
    return { turnId: qualifiedId("codex", response.turn.id) };
  }

  async steer(request: SteerTurnRequest, signal?: AbortSignal): Promise<void> {
    throwIfAborted(signal);
    await this.rpc.request("turn/steer", {
      threadId: nativeId("codex", request.conversationId),
      expectedTurnId: nativeId("codex", request.turnId),
      input: [{ type: "text", text: request.text }],
    });
    throwIfAborted(signal);
  }

  async interrupt(request: InterruptTurnRequest, signal?: AbortSignal): Promise<void> {
    throwIfAborted(signal);
    await this.rpc.request("turn/interrupt", {
      threadId: nativeId("codex", request.conversationId),
      turnId: nativeId("codex", request.turnId),
    });
    this.rejectConversationApprovals(request.conversationId, "Codex turn interrupted.");
    throwIfAborted(signal);
  }

  async revise(
    request: ReviseConversationRequest,
    signal?: AbortSignal,
  ): Promise<ConversationSummary> {
    if (!request.text) throw new Error("Cannot revise a Codex conversation with an empty message.");
    const source = await this.readThread(request.conversationId, signal);
    const boundary = this.turnBoundary(source, request.beforeTurnId, request.conversationId);
    const selected = source.turns[boundary];
    if (selected === undefined) throw new Error("Codex turn boundary invariant failed.");
    if (selected.status === "inProgress") {
      throw new Error(`Cannot revise running Codex turn "${request.beforeTurnId}".`);
    }

    if (source.historyMode === "paginated" && boundary === source.turns.length - 1) {
      await this.ensureLoaded(source.id, signal);
      const reverted = ThreadResponseSchema.parse(
        await this.rpc.request("thread/revert", {
          threadId: source.id,
          beforeTurnId: selected.id,
        }),
      ).thread;
      assertThreadWorkspace(reverted, canonicalWorkspaceRoot(source.cwd), "reverted");
      this.loadedThreads.add(reverted.id);
      try {
        const replacement = {
          conversationId: qualifiedId(this.kind, reverted.id),
          text: request.text,
        };
        if (signal === undefined) await this.start(replacement);
        else await this.start(replacement, signal);
      } catch (error) {
        throw new Error(
          `Codex history was reverted, but the replacement turn could not start. The workspace was not rolled back; keep the draft and submit it again. ${errorMessage(error)}`,
          { cause: error },
        );
      }
      return this.summary(reverted);
    }

    const child = await this.forkThreadBefore(source, boundary, signal);
    const replacement = {
      conversationId: qualifiedId(this.kind, child.id),
      text: request.text,
    };
    if (signal === undefined) await this.start(replacement);
    else await this.start(replacement, signal);
    return this.summary(child);
  }

  async fork(request: ForkConversationRequest, signal?: AbortSignal): Promise<ConversationSummary> {
    const source = await this.readThread(request.conversationId, signal);
    const boundary = this.turnBoundary(source, request.beforeTurnId, request.conversationId);
    if (source.turns[boundary]?.status === "inProgress") {
      throw new Error(`Cannot fork before running Codex turn "${request.beforeTurnId}".`);
    }
    return this.summary(await this.forkThreadBefore(source, boundary, signal));
  }

  async archive(conversationId: string, signal?: AbortSignal): Promise<void> {
    throwIfAborted(signal);
    await this.rpc.request("thread/archive", {
      threadId: nativeId("codex", conversationId),
    });
    this.clearThread(nativeId("codex", conversationId), "Codex thread archived.");
    throwIfAborted(signal);
  }

  async retireProvisionedMember(
    conversationId: string,
    provisioningId: string,
    signal?: AbortSignal,
  ): Promise<void> {
    throwIfAborted(signal);
    const threadId = nativeId("codex", conversationId);
    let thread: NativeThread;
    try {
      thread = await this.readThread(conversationId, signal);
    } catch (error) {
      if (!isThreadNotLoaded(error, threadId)) throw error;
      this.clearThread(threadId, "Codex provisioning Thread is already absent.");
      return;
    }
    if (thread.archived === true) {
      this.clearThread(threadId, "Codex provisioning Thread is already archived.");
      return;
    }
    if (thread.turns.length === 0) {
      const expectedSource = `${MEMBER_THREAD_SOURCE_PREFIX}${provisioningId}`;
      if (thread.threadSource !== expectedSource) {
        throw new Error(
          `Refusing to delete unmaterialized Codex Thread "${threadId}" without its exact provisioning identity.`,
        );
      }
      await this.rpc.request("thread/delete", { threadId });
      this.clearThread(threadId, "Codex provisioning Thread was deleted before materialization.");
      throwIfAborted(signal);
      return;
    }
    await this.archive(conversationId, signal);
  }

  subscribe(listener: RuntimeEventListener): () => void {
    this.assertOpen();
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  async respondToApproval(response: ApprovalResponse): Promise<void> {
    this.assertOpen();
    const requestedSchema = this.elicitationSchemas.get(approvalIdentity(response));
    if (
      requestedSchema !== undefined &&
      (response.decision === "accept" || response.decision === "accept_for_session")
    ) {
      if (response.form === undefined) {
        throw new Error(`Approval "${response.approvalId}" requires form content.`);
      }
      z.fromJSONSchema(requestedSchema as Parameters<typeof z.fromJSONSchema>[0]).parse(
        response.form,
      );
    }
    this.approvals.respond(response);
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    for (const dispose of this.disposers.splice(0)) dispose();
    this.approvals.dispose();
    this.approvalRequests.clear();
    this.elicitationSchemas.clear();
    this.loadedThreads.clear();
    this.loadingThreads.clear();
    this.transientThreads.clear();
    this.listeners.clear();
    await this.rpc.dispose();
  }

  private async readThread(conversationId: string, signal?: AbortSignal): Promise<NativeThread> {
    throwIfAborted(signal);
    const threadId = nativeId("codex", conversationId);
    let nativeResponse: unknown;
    try {
      nativeResponse = await this.rpc.request("thread/read", {
        threadId,
        includeTurns: true,
      });
    } catch (error) {
      if (!isUnmaterializedThreadRead(error, threadId)) throw error;
      nativeResponse = await this.rpc.request("thread/read", {
        threadId,
        includeTurns: false,
      });
    }
    const response = ThreadResponseSchema.parse(nativeResponse);
    throwIfAborted(signal);
    const archived =
      response.thread.archived === true ||
      (await this.isThreadArchived(response.thread.id, signal));
    return { ...response.thread, archived };
  }

  private async isThreadArchived(threadId: string, signal?: AbortSignal): Promise<boolean> {
    const seenCursors = new Set<string>();
    let cursor: string | undefined;
    for (let page = 1; page <= MAX_THREAD_LIST_PAGES; page += 1) {
      const response = ThreadListResponseSchema.parse(
        await this.rpc.request("thread/list", {
          limit: THREAD_LIST_PAGE_SIZE,
          archived: true,
          sortKey: "updated_at",
          sortDirection: "desc",
          sourceKinds: ALL_SOURCE_KINDS,
          ...(cursor === undefined ? {} : { cursor }),
        }),
      );
      throwIfAborted(signal);
      if (response.data.some((thread) => thread.id === threadId)) return true;
      const nextCursor = response.nextCursor ?? null;
      if (nextCursor === null) return false;
      if (page === MAX_THREAD_LIST_PAGES) {
        throw new Error(
          `Codex archived thread lookup exceeded the ${String(MAX_THREAD_LIST_PAGES)} page limit.`,
        );
      }
      if (seenCursors.has(nextCursor)) {
        throw new Error(`Codex thread/list repeated cursor "${nextCursor}".`);
      }
      seenCursors.add(nextCursor);
      cursor = nextCursor;
    }
    throw new Error("Codex archived thread lookup did not terminate.");
  }

  private async ensureLoaded(threadId: string, signal?: AbortSignal): Promise<void> {
    if (this.loadedThreads.has(threadId)) return;
    throwIfAborted(signal);
    let loading = this.loadingThreads.get(threadId);
    if (loading === undefined) {
      loading = this.resumeThread(threadId);
      this.loadingThreads.set(threadId, loading);
    }
    try {
      await loading;
    } finally {
      if (this.loadingThreads.get(threadId) === loading) this.loadingThreads.delete(threadId);
    }
    throwIfAborted(signal);
  }

  private async resumeThread(threadId: string): Promise<void> {
    const response = ThreadResponseSchema.parse(
      await this.rpc.request("thread/resume", { threadId }),
    );
    if (response.thread.id !== threadId) {
      throw new Error(
        `Codex thread/resume returned "${response.thread.id}" for requested thread "${threadId}".`,
      );
    }
    this.loadedThreads.add(threadId);
  }

  private clearThread(threadId: string, reason: string): void {
    this.loadedThreads.delete(threadId);
    this.transientThreads.delete(threadId);
    this.rejectConversationApprovals(qualifiedId(this.kind, threadId), reason);
  }

  private rejectConversationApprovals(conversationId: string, reason: string): void {
    for (const approval of this.approvals.rejectConversation(this.kind, conversationId, reason)) {
      this.emitApprovalResolved(approval);
    }
  }

  private resolveServerRequest(params: unknown): void {
    const value = ServerRequestResolvedParamsSchema.parse(params);
    const key = rpcIdKey(value.requestId);
    const approval = this.approvalRequests.get(key);
    if (approval === undefined) return;
    if (nativeId(this.kind, approval.conversationId) !== value.threadId) {
      throw new Error(
        `Codex resolved request "${String(value.requestId)}" for a mismatched thread.`,
      );
    }
    this.approvalRequests.delete(key);
    if (this.approvals.reject(approval, "Approval was resolved by Codex App Server.")) {
      this.emitApprovalResolved(approval);
    }
  }

  private emitApprovalResolved(approval: ApprovalRequest): void {
    this.emit({
      type: "approval_resolved",
      conversationId: approval.conversationId,
      turnId: approval.turnId,
      itemId: approval.itemId,
      approvalId: approval.approvalId,
    });
  }

  private turnBoundary(thread: NativeThread, beforeTurnId: string, conversationId: string): number {
    const nativeTurnId = nativeId("codex", beforeTurnId);
    const boundary = thread.turns.findIndex((turn) => turn.id === nativeTurnId);
    if (boundary < 0) {
      throw new Error(`Turn "${beforeTurnId}" is not present in "${conversationId}".`);
    }
    return boundary;
  }

  private async forkThreadBefore(
    source: NativeThread,
    boundary: number,
    signal?: AbortSignal,
  ): Promise<NativeThread> {
    const beforeTurnId = source.turns[boundary]?.id;
    if (beforeTurnId === undefined) throw new Error("Codex fork boundary invariant failed.");
    const nativeResponse = await this.rpc.request("thread/fork", {
      threadId: source.id,
      beforeTurnId,
      ephemeral: false,
    });
    throwIfAborted(signal);
    const child = { ...ThreadResponseSchema.parse(nativeResponse).thread, archived: false };
    assertThreadWorkspace(child, canonicalWorkspaceRoot(source.cwd), "forked");
    this.loadedThreads.add(child.id);
    return child;
  }

  private snapshot(thread: NativeThread): ConversationSnapshot {
    const conversationId = qualifiedId(this.kind, thread.id);
    return {
      runtime: this.kind,
      conversationId,
      workspace: this.workspace(thread.cwd),
      title: thread.name ?? (thread.preview || "New conversation"),
      archived: thread.archived ?? false,
      turns: thread.turns.map((turn) => ({
        id: qualifiedId(this.kind, turn.id),
        status: turnStatus(turn.status),
        items: this.items(turn),
      })),
      approvals: this.approvals.list(this.kind, conversationId),
    };
  }

  private summary(thread: NativeThread): ConversationSummary {
    return {
      runtime: this.kind,
      conversationId: qualifiedId(this.kind, thread.id),
      workspace: this.workspace(thread.cwd),
      title: thread.name ?? (thread.preview || "New conversation"),
      archived: thread.archived ?? false,
      updatedAt: thread.updatedAt * 1000,
    };
  }

  private items(turn: NativeTurn): ConversationItem[] {
    const createdAt = (turn.startedAt ?? 0) * 1000;
    const items = turn.items.flatMap((item) => {
      const projected = projectItem(
        item,
        qualifiedId(this.kind, turn.id),
        createdAt,
        turn.status === "inProgress",
      );
      return projected === undefined ? [] : [projected];
    });
    const errorMessage = turnError(turn.error);
    if (errorMessage !== undefined) {
      items.push({
        type: "error",
        id: qualifiedId(this.kind, `${turn.id}:error`),
        turnId: qualifiedId(this.kind, turn.id),
        message: errorMessage,
        createdAt: (turn.completedAt ?? turn.startedAt ?? 0) * 1000,
      });
    }
    return items;
  }

  private workspace(root: string): WorkspaceSummary {
    const canonical = canonicalWorkspaceRoot(root);
    const known = this.workspaces.get(canonical);
    if (known !== undefined) return known;
    const workspace = {
      id: createHash("sha256").update(canonical).digest("hex").slice(0, 24),
      label: basename(canonical) || "workspace",
    };
    this.workspaces.set(canonical, workspace);
    return workspace;
  }

  private listen(method: string, handler: (params: unknown) => void): void {
    this.disposers.push(this.rpc.onNotification(method, handler));
  }

  private emitDelta(params: unknown, itemType: "assistant_message" | "reasoning"): void {
    const value = DeltaParamsSchema.parse(params);
    this.emit({
      type: "item_delta",
      conversationId: qualifiedId(this.kind, value.threadId),
      turnId: qualifiedId(this.kind, value.turnId),
      itemId: qualifiedId(this.kind, value.itemId),
      delta: value.delta,
      itemType,
    });
  }

  private emitCompleted(params: unknown): void {
    const value = ItemNotificationSchema.parse(params);
    const turnId = qualifiedId(this.kind, value.turnId);
    const item = projectItem(value.item, turnId, value.completedAtMs ?? value.startedAtMs ?? 0);
    if (item !== undefined) {
      this.emit({
        type: "item_completed",
        conversationId: qualifiedId(this.kind, value.threadId),
        turnId,
        item,
      });
    }
  }

  private emitTurn(params: unknown): void {
    const value = TurnNotificationSchema.parse(params);
    this.emit({
      type: "turn_status",
      conversationId: qualifiedId(this.kind, value.threadId),
      turnId: qualifiedId(this.kind, value.turn.id),
      status: turnStatus(value.turn.status),
    });
  }

  private emit(event: UnsequencedRuntimeEvent): void {
    const complete = { ...event, seq: ++this.seq, runtime: this.kind } as RuntimeEvent;
    for (const listener of this.listeners) listener(complete);
  }

  private handleApproval(method: string, kind: ApprovalKind): void {
    this.disposers.push(
      this.rpc.onRequest(method, async (params, requestId) => {
        const value = ApprovalParamsSchema.parse(params);
        if (kind === "elicitation" && value.mode !== "form") {
          throw new Error("Codex supports MCP elicitation mode=form only.");
        }
        if (kind === "elicitation" && value.requestedSchema === undefined) {
          throw new Error("Codex MCP form elicitation requires requestedSchema.");
        }
        const conversationId = qualifiedId(this.kind, value.threadId);
        const nativeApprovalId = value.approvalId ?? `${method}:${String(requestId)}`;
        const approval: ApprovalRequest = {
          runtime: this.kind,
          conversationId,
          turnId: qualifiedId(this.kind, value.turnId ?? `request:${String(requestId)}`),
          itemId: qualifiedId(this.kind, value.itemId ?? `request:${String(requestId)}`),
          approvalId: qualifiedId(this.kind, nativeApprovalId),
          kind,
          prompt: approvalPrompt(kind, value),
          choices: approvalChoices(kind, value.availableDecisions),
          ...approvalQuestions(kind, value),
        };
        const requestKey = rpcIdKey(requestId);
        if (this.approvalRequests.has(requestKey)) {
          throw new Error(`Codex request id "${String(requestId)}" is already pending.`);
        }
        const identity = approvalIdentity(approval);
        if (kind === "elicitation" && value.requestedSchema !== undefined) {
          if (this.elicitationSchemas.has(identity)) {
            throw new Error(`Codex approval "${approval.approvalId}" is already pending.`);
          }
          this.elicitationSchemas.set(identity, value.requestedSchema);
        }
        this.approvalRequests.set(requestKey, approval);
        const pending = this.approvals.request(approval);
        this.emit({ type: "approval_requested", ...approval });
        try {
          const response = await pending;
          return approvalResult(kind, response, value.permissions);
        } finally {
          this.approvalRequests.delete(requestKey);
          this.elicitationSchemas.delete(identity);
        }
      }),
    );
  }

  private assertOpen(): void {
    if (this.disposed) throw new Error("Codex runtime is disposed.");
  }
}

function isUnmaterializedThreadRead(error: unknown, threadId: string): boolean {
  return (
    error instanceof CodexRpcError &&
    error.code === -32600 &&
    error.data === undefined &&
    error.message ===
      `thread ${threadId} is not materialized yet; includeTurns is unavailable before first user message`
  );
}

function qualifiedId(runtime: "codex", id: string): string {
  if (!id) throw new Error("Codex returned an empty native id.");
  return `${runtime}:${id}`;
}

function nativeId(runtime: "codex", id: string): string {
  const prefix = `${runtime}:`;
  if (!id.startsWith(prefix) || id.length === prefix.length) {
    throw new Error(`Expected a ${runtime}-qualified id, received "${id}".`);
  }
  return id.slice(prefix.length);
}

function isThreadNotLoaded(error: unknown, threadId: string): boolean {
  return (
    error instanceof CodexRpcError &&
    error.code === -32600 &&
    error.data === undefined &&
    error.message === `thread not loaded: ${threadId}`
  );
}

function rpcIdKey(id: string | number): string {
  return `${typeof id}:${String(id)}`;
}

function approvalIdentity(request: ApprovalRequest | ApprovalResponse): string {
  const identity = JSON.stringify([
    request.runtime,
    request.conversationId,
    request.turnId,
    request.itemId,
    request.approvalId,
  ]);
  if (identity === undefined) throw new Error("Approval identity could not be encoded.");
  return identity;
}

function turnStatus(status: NativeTurn["status"]): TurnStatus {
  return status === "inProgress" ? "running" : status;
}

function projectItem(
  item: NativeItem,
  turnId: string,
  createdAt: number,
  provisional = false,
): ConversationItem | undefined {
  const id = qualifiedId("codex", item.id);
  if (item.type === "userMessage") {
    return {
      type: "user_message",
      id,
      turnId,
      text: userText(item.content),
      createdAt,
    };
  }
  if (item.type === "agentMessage") {
    return {
      type: "assistant_message",
      id,
      turnId,
      text: stringField(item.text),
      createdAt,
      ...(provisional ? { provisional: true } : {}),
    };
  }
  if (item.type === "reasoning") {
    return {
      type: "reasoning",
      id,
      turnId,
      text: stringList(item.summary).concat(stringList(item.content)).join("\n"),
      createdAt,
      ...(provisional ? { provisional: true } : {}),
    };
  }
  const toolName = toolItemName(item.type);
  if (toolName !== undefined) {
    return {
      type: "tool",
      id,
      turnId,
      name: toolName,
      status: toolStatus(item.status),
      ...(typeof item.aggregatedOutput === "string"
        ? { summary: item.aggregatedOutput.slice(0, 100_000) }
        : {}),
      createdAt,
    };
  }
  return undefined;
}

function userText(content: unknown): string {
  if (!Array.isArray(content)) return "";
  return content
    .flatMap((entry) => {
      if (entry === null || typeof entry !== "object") return [];
      const record = entry as Record<string, unknown>;
      return (record.type === "text" || record.type === "input_text") &&
        typeof record.text === "string"
        ? [record.text]
        : [];
    })
    .join("\n");
}

function stringField(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function stringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string")
    : [];
}

function toolItemName(type: string): string | undefined {
  switch (type) {
    case "commandExecution":
      return "command";
    case "fileChange":
      return "file_change";
    case "mcpToolCall":
      return "mcp";
    case "webSearch":
      return "web_search";
    case "imageGeneration":
      return "image_generation";
    case "collabAgentToolCall":
      return "agent";
    default:
      return undefined;
  }
}

function toolStatus(value: unknown): "running" | "completed" | "failed" {
  return value === "completed"
    ? "completed"
    : value === "failed" || value === "declined"
      ? "failed"
      : "running";
}

function turnError(value: unknown): string | undefined {
  if (value === null || value === undefined) return undefined;
  if (typeof value === "string") return value;
  if (typeof value === "object" && "message" in value) {
    const message = (value as { message?: unknown }).message;
    if (typeof message === "string") return message;
  }
  return "Codex turn failed.";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function approvalPrompt(kind: ApprovalKind, value: z.infer<typeof ApprovalParamsSchema>): string {
  const detail = value.reason ?? value.command ?? value.message ?? value.serverName;
  return (detail ?? `Codex requested ${kind.replaceAll("_", " ")}.`).slice(0, 100_000);
}

function approvalChoices(
  kind: ApprovalKind,
  availableDecisions: unknown[] | null | undefined,
): readonly ApprovalDecision[] {
  if (kind === "user_input") return ["submit", "cancel"];
  if (kind === "elicitation") return ["accept", "decline", "cancel"];
  if (kind === "command" && availableDecisions != null) {
    const choices = availableDecisions.flatMap<ApprovalDecision>((decision) => {
      switch (decision) {
        case "accept":
        case "decline":
        case "cancel":
          return [decision];
        case "acceptForSession":
          return ["accept_for_session"];
        default:
          return [];
      }
    });
    if (choices.length === 0) {
      throw new Error("Codex command approval offers no supported decision.");
    }
    return [...new Set(choices)];
  }
  return ["accept", "accept_for_session", "decline", "cancel"];
}

function approvalQuestions(
  kind: ApprovalKind,
  value: z.infer<typeof ApprovalParamsSchema>,
): { questions?: readonly import("../contracts.js").ApprovalQuestion[] } {
  if (kind === "user_input") {
    return {
      questions: (value.questions ?? []).map((question) => ({
        id: question.id,
        prompt: question.question,
        ...(question.header === undefined ? {} : { header: question.header }),
        ...(question.options == null
          ? {}
          : { options: question.options.map((option) => option.label) }),
      })),
    };
  }
  if (kind !== "elicitation" || value.requestedSchema === undefined) return {};
  const required = new Set(value.requestedSchema.required ?? []);
  return {
    questions: Object.entries(value.requestedSchema.properties).map(([id, raw]) => {
      const schema = raw as Record<string, unknown>;
      const itemSchema =
        schema.items !== null && typeof schema.items === "object"
          ? (schema.items as Record<string, unknown>)
          : undefined;
      const directOptions = arrayOfStrings(schema.enum);
      const titledOptions = arrayOfRecords(schema.oneOf).map((entry) => entry.const);
      const itemOptions = arrayOfStrings(itemSchema?.enum);
      const titledItemOptions = arrayOfRecords(itemSchema?.anyOf).map((entry) => entry.const);
      const options = firstNonEmpty(directOptions, titledOptions, itemOptions, titledItemOptions);
      const type =
        schema.type === "array"
          ? "string_array"
          : schema.type === "boolean" || schema.type === "number" || schema.type === "integer"
            ? schema.type
            : "string";
      return {
        id,
        type,
        prompt:
          typeof schema.description === "string"
            ? schema.description
            : typeof schema.title === "string"
              ? schema.title
              : id,
        required: required.has(id),
        ...(typeof schema.title === "string" ? { header: schema.title } : {}),
        ...(isApprovalDefault(schema.default) ? { defaultValue: schema.default } : {}),
        ...(options === undefined ? {} : { options }),
        ...(schema.type === "array" ? { multiSelect: true } : {}),
        ...numberProperty(schema, "minimum"),
        ...numberProperty(schema, "maximum"),
        ...numberProperty(schema, "minLength"),
        ...numberProperty(schema, "maxLength"),
        ...numberProperty(schema, "minItems"),
        ...numberProperty(schema, "maxItems"),
        ...(schema.format === "date" ||
        schema.format === "uri" ||
        schema.format === "email" ||
        schema.format === "date-time"
          ? { format: schema.format }
          : {}),
      };
    }),
  };
}

function arrayOfStrings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string")
    : [];
}

function arrayOfRecords(value: unknown): Array<{ const: string }> {
  return Array.isArray(value)
    ? value.filter(
        (entry): entry is { const: string } =>
          entry !== null &&
          typeof entry === "object" &&
          "const" in entry &&
          typeof entry.const === "string",
      )
    : [];
}

function firstNonEmpty(...values: string[][]): string[] | undefined {
  return values.find((value) => value.length > 0);
}

function isApprovalDefault(value: unknown): value is string | boolean | number | string[] {
  return (
    typeof value === "string" ||
    typeof value === "boolean" ||
    typeof value === "number" ||
    (Array.isArray(value) && value.every((entry) => typeof entry === "string"))
  );
}

function numberProperty(
  schema: Record<string, unknown>,
  key: "minimum" | "maximum" | "minLength" | "maxLength" | "minItems" | "maxItems",
): Partial<Record<typeof key, number>> {
  return typeof schema[key] === "number" ? { [key]: schema[key] } : {};
}

function approvalResult(
  kind: ApprovalKind,
  response: ApprovalResponse,
  requestedPermissions: Record<string, unknown> | undefined,
): unknown {
  const { decision } = response;
  if (kind === "command" || kind === "file_change") {
    return { decision: decision === "accept_for_session" ? "acceptForSession" : decision };
  }
  if (kind === "permissions") {
    return {
      permissions:
        decision === "accept" || decision === "accept_for_session"
          ? (requestedPermissions ?? {})
          : {},
      scope: decision === "accept_for_session" ? "session" : "turn",
    };
  }
  if (kind === "elicitation") {
    return {
      action: decision === "accept_for_session" ? "accept" : decision,
      ...(decision === "accept" || decision === "accept_for_session"
        ? { content: response.form ?? {} }
        : {}),
    };
  }
  return {
    answers: Object.fromEntries(
      Object.entries(response.answers ?? {}).map(([id, answers]) => [id, { answers }]),
    ),
  };
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  signal?.throwIfAborted();
}

function canonicalWorkspaceRoot(root: string): string {
  try {
    return realpathSync.native(root);
  } catch (cause) {
    throw new Error(`Codex Thread workspace "${root}" cannot be canonicalized.`, { cause });
  }
}

function assertThreadWorkspace(
  thread: NativeThread,
  expectedCanonicalRoot: string,
  operation: "created" | "forked" | "reverted",
): void {
  if (canonicalWorkspaceRoot(thread.cwd) !== expectedCanonicalRoot) {
    throw new Error(
      `Codex ${operation} Thread workspace "${thread.cwd}" does not match expected workspace "${expectedCanonicalRoot}".`,
    );
  }
}
