import { createHash, randomUUID } from "node:crypto";
import { basename } from "node:path";
import type { Context } from "@deepseek-ai/cordis";
import type { Agent, AgentHandle } from "@deepseek-ai/dsh-agent";
import type { AgentPresets } from "@deepseek-ai/dsh-agent-presets";
import { type ContentBlock, createUserMessage } from "@deepseek-ai/dsh-llm";
import { type Session, type SessionEvent, SessionId } from "@deepseek-ai/dsh-session";
import type { SessionQueryEngine } from "@deepseek-ai/dsh-session-query";
import type { WorkspaceRegistry } from "@deepseek-ai/dsh-workspace";
import type {
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

export type DshRuntimeHost = Context & {
  agents: Context["agents"];
  agentPresets: AgentPresets;
  sessionQuery: SessionQueryEngine;
  workspaceRegistry: WorkspaceRegistry;
};

type UnsequencedRuntimeEvent = RuntimeEvent extends infer Event
  ? Event extends RuntimeEvent
    ? Omit<Event, "seq" | "runtime">
    : never
  : never;

const MAX_LISTED_CONVERSATIONS = 1_000;

export class DshConversationRuntime implements ConversationRuntime {
  readonly kind = "dsh" as const;
  private readonly listeners = new Set<RuntimeEventListener>();
  private readonly disposers: Array<() => unknown> = [];
  private readonly handles = new Map<string, AgentHandle>();
  private readonly workspaces = new Map<string, WorkspaceSummary>();
  private readonly nextTurns = new Map<string, number>();
  private readonly archiving = new Set<string>();
  private seq = 0;
  private disposed = false;

  constructor(private readonly ctx: DshRuntimeHost) {
    this.disposers.push(
      ctx.on("session/event", (session, event) =>
        this.emitSessionEvent(session as unknown as Session, event as unknown as SessionEvent),
      ),
    );
  }

  async list(signal?: AbortSignal): Promise<ConversationSummary[]> {
    signal?.throwIfAborted();
    const records = await this.ctx.sessionQuery.listSessions(signal);
    const archived = new Set(this.ctx.workspaceRegistry.archivedSessionIds.map(String));
    const visible = records
      .filter(({ header }) => !archived.has(String(header.id)))
      .slice(0, MAX_LISTED_CONVERSATIONS);
    const observed = await this.ctx.sessionQuery.readTitleSnapshots(
      visible.map(({ header }) => header.id),
      signal,
    );
    signal?.throwIfAborted();
    const summaries = visible.map(({ header }, index) => {
      const result = observed[index];
      const source = result?.status === "fulfilled" ? result.value.session : header;
      const title = result?.status === "fulfilled" ? result.value.title : undefined;
      return {
        runtime: this.kind,
        conversationId: conversationId(source.id),
        workspace: this.workspace(source.cwd),
        title: title?.title ?? "New conversation",
        archived: false,
        updatedAt: title?.updatedAt ?? source.createdAt,
      } satisfies ConversationSummary;
    });
    return summaries.sort((left, right) => right.updatedAt - left.updatedAt);
  }

  async create(
    request: CreateConversationRequest,
    signal?: AbortSignal,
  ): Promise<ConversationSummary> {
    signal?.throwIfAborted();
    this.workspaces.set(request.workspace.root, {
      id: request.workspace.id,
      label: request.workspace.label,
    });
    return this.createAgent({
      root: request.workspace.root,
      workspace: request.workspace,
      ...(request.model === undefined ? {} : { model: request.model }),
      ...(signal === undefined ? {} : { signal }),
    });
  }

  async read(conversation: string, signal?: AbortSignal): Promise<ConversationSnapshot> {
    signal?.throwIfAborted();
    const source = await this.ctx.sessionQuery.readSession(sessionId(conversation));
    const title = await this.ctx.sessionQuery.readTitle(source.session.id, signal);
    const archived = this.ctx.workspaceRegistry.archivedSessionIds.some(
      (id) => String(id) === String(source.session.id),
    );
    signal?.throwIfAborted();
    return projectSession(
      source.session,
      source.events,
      title?.title ?? "New conversation",
      archived,
      (cwd) => this.workspace(cwd),
    );
  }

  async start(request: StartTurnRequest, signal?: AbortSignal): Promise<{ turnId: string }> {
    signal?.throwIfAborted();
    if (!request.text) throw new Error("Cannot send an empty DSH message.");
    const agent = await this.ensureAgent(request.conversationId, signal);
    this.assertNotArchived(agent.id);
    this.assertNotArchiving(agent.id);
    signal?.throwIfAborted();
    const nativeSessionId = String(agent.id);
    const observed = nextTurn(agent.session.events);
    const turn = Math.max(observed, this.nextTurns.get(nativeSessionId) ?? observed);
    this.nextTurns.set(nativeSessionId, turn + 1);
    agent.followup(
      createUserMessage({
        content: [{ type: "text", text: request.text }],
        source: { kind: "user" },
      }),
    );
    return { turnId: turnId(nativeSessionId, turn) };
  }

  async steer(request: SteerTurnRequest, signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    const agent = await this.ensureAgent(request.conversationId, signal);
    this.assertNotArchived(agent.id);
    this.assertNotArchiving(agent.id);
    signal?.throwIfAborted();
    const open = openTurn(agent.session.events);
    const requested = nativeTurnId(String(agent.id), request.turnId);
    if (open !== requested) {
      throw new Error(
        `DSH turn "${request.turnId}" is not the active turn for "${request.conversationId}".`,
      );
    }
    agent.steer(
      createUserMessage({
        content: [{ type: "text", text: request.text }],
        source: { kind: "user" },
      }),
    );
  }

  async interrupt(request: InterruptTurnRequest, signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    const agent = await this.ensureAgent(request.conversationId, signal);
    this.assertNotArchived(agent.id);
    this.assertNotArchiving(agent.id);
    signal?.throwIfAborted();
    const requested = nativeTurnId(String(agent.id), request.turnId);
    if (openTurn(agent.session.events) !== requested) {
      throw new Error(`DSH turn "${request.turnId}" is not active.`);
    }
    agent.cancel({ kind: "user" });
  }

  async revise(
    request: ReviseConversationRequest,
    signal?: AbortSignal,
  ): Promise<ConversationSummary> {
    if (!request.text) throw new Error("Cannot revise a DSH conversation with an empty message.");
    const child = await this.fork(request, signal);
    const replacement = { conversationId: child.conversationId, text: request.text };
    if (signal === undefined) await this.start(replacement);
    else await this.start(replacement, signal);
    return child;
  }

  async fork(request: ForkConversationRequest, signal?: AbortSignal): Promise<ConversationSummary> {
    signal?.throwIfAborted();
    const sourceId = sessionId(request.conversationId);
    this.assertNotArchived(sourceId);
    this.assertNotArchiving(sourceId);
    const source = await this.ctx.sessionQuery.readSession(sourceId);
    signal?.throwIfAborted();
    this.assertNotArchived(sourceId);
    this.assertNotArchiving(sourceId);
    const selectedTurn = nativeTurnId(String(source.session.id), request.beforeTurnId);
    const start = source.events.find(
      (event) => event.type === "turn/start" && event.data.turn === selectedTurn,
    );
    if (start === undefined) {
      throw new Error(
        `Turn "${request.beforeTurnId}" is not present in "${request.conversationId}".`,
      );
    }
    const end = source.events.find(
      (event) => event.type === "turn/end" && event.data.turn === selectedTurn,
    );
    if (end === undefined) {
      throw new Error(`Cannot fork before running DSH turn "${request.beforeTurnId}".`);
    }
    if (source.session.cwd === undefined) {
      throw new Error("Cannot fork a DSH conversation without a workspace directory.");
    }
    const workspace = this.workspace(source.session.cwd);
    const seed = start.seq === 0 ? undefined : source.events.slice(0, start.seq);
    return this.createAgent({
      root: source.session.cwd,
      workspace,
      ...(source.session.agentPreset === undefined ? {} : { preset: source.session.agentPreset }),
      ...(seed === undefined
        ? {}
        : { seed, parentSession: source.session.id, seedLength: seed.length }),
      ...(signal === undefined ? {} : { signal }),
    });
  }

  async archive(conversation: string, signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted();
    const id = sessionId(conversation);
    this.assertNotArchived(id);
    this.assertNotArchiving(id);
    this.archiving.add(String(id));
    try {
      const live = this.handles.get(String(id))?.agent ?? this.ctx.agents.get(id);
      const events = live?.session.events ?? (await this.ctx.sessionQuery.readSession(id)).events;
      signal?.throwIfAborted();
      if (openTurn(events) !== undefined) {
        throw new Error(`Cannot archive running DSH conversation "${conversation}".`);
      }
      await this.ctx.workspaceRegistry.archiveSession(id);
      signal?.throwIfAborted();
    } finally {
      this.archiving.delete(String(id));
    }
  }

  subscribe(listener: RuntimeEventListener): () => void {
    this.assertOpen();
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  async respondToApproval(_response: ApprovalResponse): Promise<void> {
    this.assertOpen();
    throw new Error("DSH approvals are handled by the native DSH Web approval channel.");
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    for (const dispose of this.disposers.splice(0)) dispose();
    this.listeners.clear();
    await Promise.all([...this.handles.values()].map((handle) => handle.dispose()));
    this.handles.clear();
  }

  private async createAgent(options: {
    root: string;
    workspace: WorkspaceSummary;
    model?: string;
    preset?: string;
    seed?: readonly SessionEvent[];
    parentSession?: SessionId;
    seedLength?: number;
    signal?: AbortSignal;
  }): Promise<ConversationSummary> {
    const id = SessionId(randomUUID());
    const preset = options.preset ?? this.ctx.agentPresets.defaultId;
    const handle = await this.ctx.agents.create({
      sessionId: id,
      meta: {
        cwd: options.root,
        agentPreset: preset,
        ...(options.parentSession === undefined ? {} : { parentSession: options.parentSession }),
        ...(options.seedLength === undefined ? {} : { seedLength: options.seedLength }),
      },
      ...(options.seed === undefined ? {} : { seed: options.seed }),
      ...(options.model === undefined ? {} : { agentOptions: { model: options.model } }),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      setup: async (agentCtx) => {
        await this.ctx.agentPresets.mount(agentCtx, preset);
      },
    });
    try {
      options.signal?.throwIfAborted();
      const workspace = await this.ctx.workspaceRegistry.create(
        options.root,
        options.workspace.label,
      );
      options.signal?.throwIfAborted();
      await workspace.attachSession(id);
      options.signal?.throwIfAborted();
    } catch (error) {
      await handle.dispose();
      throw error;
    }
    this.handles.set(String(id), handle);
    this.workspaces.set(options.root, options.workspace);
    return {
      runtime: this.kind,
      conversationId: conversationId(id),
      workspace: options.workspace,
      title: "New conversation",
      archived: false,
      updatedAt: Date.now(),
    };
  }

  private async ensureAgent(conversation: string, signal?: AbortSignal): Promise<Agent> {
    const id = sessionId(conversation);
    this.assertNotArchived(id);
    this.assertNotArchiving(id);
    const owned = this.handles.get(String(id));
    if (owned !== undefined) return owned.agent;
    const live = this.ctx.agents.get(id);
    if (live !== undefined) return live;
    const source = await this.ctx.sessionQuery.readSession(id);
    const preset = source.session.agentPreset ?? this.ctx.agentPresets.defaultId;
    const handle = await this.ctx.agents.resume({
      resumeSessionId: id,
      ...(signal === undefined ? {} : { signal }),
      setup: async (agentCtx) => {
        await this.ctx.agentPresets.mount(agentCtx, preset);
      },
    });
    if (signal?.aborted === true) {
      await handle.dispose();
      signal.throwIfAborted();
    }
    if (this.isArchived(id)) {
      await handle.dispose();
      this.assertNotArchived(id);
    }
    this.assertNotArchiving(id);
    this.handles.set(String(id), handle);
    return handle.agent;
  }

  private assertNotArchived(id: SessionId): void {
    if (this.isArchived(id)) {
      throw new Error(`DSH conversation "${conversationId(id)}" is archived.`);
    }
  }

  private assertNotArchiving(id: SessionId): void {
    if (this.archiving.has(String(id))) {
      throw new Error(`DSH conversation "${conversationId(id)}" is being archived.`);
    }
  }

  private isArchived(id: SessionId): boolean {
    return this.ctx.workspaceRegistry.archivedSessionIds.some(
      (archivedId) => String(archivedId) === String(id),
    );
  }

  private workspace(cwd: string | undefined): WorkspaceSummary {
    if (cwd === undefined) return { id: "ungrouped", label: "Ungrouped" };
    const known = this.workspaces.get(cwd);
    if (known !== undefined) return known;
    const registered = this.ctx.workspaceRegistry
      .list()
      .find((workspace) => workspace.path === cwd);
    const workspace =
      registered === undefined
        ? {
            id: createHash("sha256").update(cwd).digest("hex").slice(0, 24),
            label: basename(cwd) || "workspace",
          }
        : { id: String(registered.id), label: registered.title };
    this.workspaces.set(cwd, workspace);
    return workspace;
  }

  private emitSessionEvent(session: Session, event: SessionEvent): void {
    const conversation = conversationId(session.id);
    if (event.type === "turn/start") {
      this.emit({
        type: "turn_status",
        conversationId: conversation,
        turnId: turnId(String(session.id), event.data.turn),
        status: "running",
      });
      return;
    }
    if (event.type === "turn/end") {
      this.emit({
        type: "turn_status",
        conversationId: conversation,
        turnId: turnId(String(session.id), event.data.turn),
        status: endStatus(event.data.reason.kind),
      });
      return;
    }
    if (event.type === "assistant/chunk") {
      const chunk = event.data.chunk;
      if (chunk.type !== "text-delta" && chunk.type !== "reasoning-delta") return;
      this.emit({
        type: "item_delta",
        conversationId: conversation,
        turnId: turnId(String(session.id), event.data.turn),
        itemId: assistantItemId(String(session.id), event.data.turn, event.data.step, chunk.type),
        itemType: chunk.type === "text-delta" ? "assistant_message" : "reasoning",
        delta: chunk.text,
      });
      return;
    }
    if (event.type === "user/message" && event.surfaceOp === "append") {
      if (event.data.source.kind !== "user") return;
      const turn = openTurnAt(session.events, event.seq);
      if (turn === undefined) return;
      this.emit({
        type: "item_completed",
        conversationId: conversation,
        turnId: turnId(String(session.id), turn),
        item: {
          type: "user_message",
          id: itemId(event.data.id),
          turnId: turnId(String(session.id), turn),
          text: contentText(event.data.content, "text"),
          createdAt: event.time,
        },
      });
      return;
    }
    if (event.type === "assistant/message" && event.surfaceOp === "append") {
      for (const item of assistantItems(String(session.id), event)) {
        this.emit({
          type: "item_completed",
          conversationId: conversation,
          turnId: item.turnId,
          item,
        });
      }
      return;
    }
    if (event.type === "tool/result") {
      this.emit({
        type: "item_completed",
        conversationId: conversation,
        turnId: turnId(String(session.id), event.data.turn),
        item: {
          type: "tool",
          id: itemId(String(event.data.message.content[0]?.toolCallId ?? event.data.message.id)),
          turnId: turnId(String(session.id), event.data.turn),
          name: "tool",
          status: event.data.error === undefined ? "completed" : "failed",
          summary: contentText(event.data.message.content[0]?.content ?? [], "text"),
          createdAt: event.time,
        },
      });
    }
  }

  private emit(event: UnsequencedRuntimeEvent): void {
    const complete = { ...event, seq: ++this.seq, runtime: this.kind } as RuntimeEvent;
    for (const listener of this.listeners) listener(complete);
  }

  private assertOpen(): void {
    if (this.disposed) throw new Error("DSH runtime is disposed.");
  }
}

function projectSession(
  header: { id: SessionId; cwd?: string; createdAt: number },
  events: readonly SessionEvent[],
  title: string,
  archived: boolean,
  workspace: (cwd: string | undefined) => WorkspaceSummary,
): ConversationSnapshot {
  const turns = new Map<number, { id: string; status: TurnStatus; items: ConversationItem[] }>();
  const tools = new Map<string, ConversationItem & { type: "tool" }>();
  let activeTurn: number | undefined;
  for (const event of events) {
    if (event.type === "turn/start") {
      activeTurn = event.data.turn;
      turns.set(activeTurn, {
        id: turnId(String(header.id), activeTurn),
        status: "running",
        items: [],
      });
      continue;
    }
    if (event.type === "turn/end") {
      const turn = turns.get(event.data.turn);
      if (turn !== undefined) turn.status = endStatus(event.data.reason.kind);
      if (activeTurn === event.data.turn) activeTurn = undefined;
      continue;
    }
    if (event.type === "user/message" && event.surfaceOp === "append") {
      if (activeTurn !== undefined && event.data.source.kind === "user") {
        turns.get(activeTurn)?.items.push({
          type: "user_message",
          id: itemId(event.data.id),
          turnId: turnId(String(header.id), activeTurn),
          text: contentText(event.data.content, "text"),
          createdAt: event.time,
        });
      }
      continue;
    }
    if (event.type === "assistant/message" && event.surfaceOp === "append") {
      turns.get(event.data.turn)?.items.push(...assistantItems(String(header.id), event));
      continue;
    }
    if (event.type === "tool/call") {
      const item: ConversationItem & { type: "tool" } = {
        type: "tool",
        id: itemId(String(event.data.callId)),
        turnId: turnId(String(header.id), event.data.turn),
        name: event.data.name,
        status: "running",
        createdAt: event.time,
      };
      tools.set(String(event.data.callId), item);
      turns.get(event.data.turn)?.items.push(item);
      continue;
    }
    if (event.type === "tool/result") {
      const callId = String(event.data.message.content[0]?.toolCallId ?? "");
      const tool = tools.get(callId);
      if (tool !== undefined) {
        tool.status = event.data.error === undefined ? "completed" : "failed";
        tool.summary = contentText(event.data.message.content[0]?.content ?? [], "text");
      }
    }
  }
  return {
    runtime: "dsh",
    conversationId: conversationId(header.id),
    workspace: workspace(header.cwd),
    title,
    archived,
    turns: [...turns.values()],
  };
}

function assistantItems(
  nativeSessionId: string,
  event: Extract<SessionEvent, { type: "assistant/message" }>,
): ConversationItem[] {
  const nativeTurn = event.data.turn;
  const commonTurnId = turnId(nativeSessionId, nativeTurn);
  const items: ConversationItem[] = [];
  const reasoning = contentText(event.data.message.content, "reasoning");
  if (reasoning) {
    items.push({
      type: "reasoning",
      id: assistantItemId(nativeSessionId, nativeTurn, event.data.step, "reasoning-delta"),
      turnId: commonTurnId,
      text: reasoning,
      createdAt: event.time,
    });
  }
  const text = contentText(event.data.message.content, "text");
  if (text) {
    items.push({
      type: "assistant_message",
      id: assistantItemId(nativeSessionId, nativeTurn, event.data.step, "text-delta"),
      turnId: commonTurnId,
      text,
      createdAt: event.time,
    });
  }
  return items;
}

function contentText(blocks: readonly ContentBlock[], type: "text" | "reasoning"): string {
  return blocks
    .filter((block): block is Extract<ContentBlock, { type: typeof type }> => block.type === type)
    .map((block) => block.text)
    .join("\n");
}

function nextTurn(events: readonly SessionEvent[]): number {
  let highest = 0;
  for (const event of events) {
    if (event.type === "turn/start") highest = Math.max(highest, event.data.turn);
  }
  return highest + 1;
}

function openTurn(events: readonly SessionEvent[]): number | undefined {
  return openTurnAt(events, events.length);
}

function openTurnAt(events: readonly SessionEvent[], beforeSeq: number): number | undefined {
  let open: number | undefined;
  for (const event of events) {
    if (event.seq >= beforeSeq) break;
    if (event.type === "turn/start") open = event.data.turn;
    else if (event.type === "turn/end" && event.data.turn === open) open = undefined;
  }
  return open;
}

function endStatus(kind: string): TurnStatus {
  if (kind === "completed" || kind === "max-tokens") return "completed";
  if (kind === "aborted" || kind === "interrupted") return "interrupted";
  return "failed";
}

function conversationId(id: SessionId | string): string {
  return `dsh:${String(id)}`;
}

function sessionId(id: string): SessionId {
  if (!id.startsWith("dsh:") || id.length === 4) {
    throw new Error(`Expected a dsh-qualified conversation id, received "${id}".`);
  }
  return SessionId(id.slice(4));
}

function turnId(nativeSessionId: string, turn: number): string {
  return `dsh:${nativeSessionId}:turn:${String(turn)}`;
}

function nativeTurnId(nativeSessionId: string, id: string): number {
  const prefix = `dsh:${nativeSessionId}:turn:`;
  if (!id.startsWith(prefix)) throw new Error(`Turn "${id}" does not belong to DSH session.`);
  const turn = Number(id.slice(prefix.length));
  if (!Number.isSafeInteger(turn) || turn < 1) throw new Error(`Invalid DSH turn id "${id}".`);
  return turn;
}

function itemId(id: string): string {
  return `dsh:${id}`;
}

function assistantItemId(
  nativeSessionId: string,
  turn: number,
  step: number,
  type: "text-delta" | "reasoning-delta",
): string {
  return itemId(
    `${nativeSessionId}:${type === "text-delta" ? "assistant" : "reasoning"}:${String(turn)}:${String(step)}`,
  );
}
