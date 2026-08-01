import { randomUUID } from "node:crypto";
import type {
  MessageChunk,
  SessionData,
  TransientSessionContextChip,
  TransientSessionData,
} from "@swarmx/core";
import {
  appendTransientSessionMessages,
  createTransientSessionFork,
  editTransientSessionUserMessage,
  promoteTransientSessionFork,
  transientSessionModelMessages,
} from "@swarmx/core";

export interface SideChatParentState {
  parentSessionId: string;
  activeSideChatId: string | null;
  paneHidden: boolean;
  chats: TransientSessionData[];
}

export interface CreateSideChatInput {
  parentSessionId: string;
  throughMessageIndex: number;
  expectedMessages: MessageChunk[];
  title?: string;
}

export interface UpdateSideChatInput {
  parentSessionId: string;
  sideChatId: string;
  draft?: string;
  attachments?: string[];
  title?: string;
  unread?: boolean;
}

interface MutableParentState {
  activeSideChatId: string | null;
  paneHidden: boolean;
  chats: Map<string, TransientSessionData>;
}

export class SideChatService {
  private readonly parents = new Map<string, MutableParentState>();

  list(parentSessionId: string): SideChatParentState {
    const state = this.parents.get(parentSessionId);
    return {
      parentSessionId,
      activeSideChatId: state?.activeSideChatId ?? null,
      paneHidden: state?.paneHidden ?? true,
      chats: state ? [...state.chats.values()].map(cloneSideChat) : [],
    };
  }

  create(input: CreateSideChatInput): TransientSessionData {
    const state = this.parent(input.parentSessionId);
    const transient = createTransientSessionFork({
      id: input.parentSessionId,
      throughMessageIndex: input.throughMessageIndex,
      expectedMessages: input.expectedMessages,
      title: input.title?.trim() || `Side chat ${state.chats.size + 1}`,
    });
    if (!transient) throw new Error(`Session "${input.parentSessionId}" was not found.`);
    state.chats.set(transient.id, transient);
    state.activeSideChatId = transient.id;
    state.paneHidden = false;
    return cloneSideChat(transient);
  }

  update(input: UpdateSideChatInput): TransientSessionData {
    const state = this.requireParent(input.parentSessionId);
    const current = this.requireChat(state, input.sideChatId);
    const next = {
      ...current,
      ...(input.draft !== undefined ? { draft: input.draft } : {}),
      ...(input.attachments !== undefined ? { attachments: [...input.attachments] } : {}),
      ...(input.title?.trim() ? { title: input.title.trim() } : {}),
      ...(input.unread !== undefined ? { unread: input.unread } : {}),
      updatedAt: new Date().toISOString(),
    };
    state.chats.set(current.id, next);
    return cloneSideChat(next);
  }

  activate(parentSessionId: string, sideChatId: string): SideChatParentState {
    const state = this.requireParent(parentSessionId);
    const current = this.requireChat(state, sideChatId);
    state.activeSideChatId = current.id;
    state.paneHidden = false;
    state.chats.set(current.id, { ...current, unread: false });
    return this.list(parentSessionId);
  }

  setPaneHidden(parentSessionId: string, hidden: boolean): SideChatParentState {
    const state = this.requireParent(parentSessionId);
    state.paneHidden = hidden;
    return this.list(parentSessionId);
  }

  addContext(parentSessionId: string, sideChatId: string, text: string): TransientSessionData {
    const state = this.requireParent(parentSessionId);
    const current = this.requireChat(state, sideChatId);
    const normalized = text.replace(/\s+/g, " ").trim();
    if (!normalized) throw new Error("Selected side chat context cannot be empty.");
    const now = new Date().toISOString();
    const chip: TransientSessionContextChip = {
      id: randomUUID(),
      text: normalized.slice(0, 8_000),
      createdAt: now,
    };
    const next = {
      ...current,
      contextChips: [...current.contextChips, chip],
      updatedAt: now,
    };
    state.chats.set(current.id, next);
    state.activeSideChatId = current.id;
    state.paneHidden = false;
    return cloneSideChat(next);
  }

  beginRun(
    parentSessionId: string,
    sideChatId: string,
    requestId: string,
    userText: string,
    replaceMessageIndex?: number,
  ): TransientSessionData {
    const state = this.requireParent(parentSessionId);
    const current = this.requireChat(state, sideChatId);
    if (current.runState !== "idle") throw new Error("This side chat is already running.");
    const content = userText.trim();
    if (!content) throw new Error("Side chat message cannot be empty.");
    const withUser =
      replaceMessageIndex === undefined
        ? appendTransientSessionMessages(current, [{ role: "user", content, kind: "message" }], {
            contextChips: current.contextChips,
          })
        : editTransientSessionUserMessage(current, replaceMessageIndex, content);
    const next = {
      ...withUser,
      draft: "",
      attachments: [],
      runState: "running" as const,
      requestId,
      unread: false,
    };
    state.chats.set(current.id, next);
    return cloneSideChat(next);
  }

  finishRun(
    parentSessionId: string,
    sideChatId: string,
    requestId: string,
    messages: MessageChunk[],
    options: { unread: boolean },
  ): TransientSessionData {
    const state = this.requireParent(parentSessionId);
    const current = this.requireChat(state, sideChatId);
    if (current.requestId !== requestId) {
      throw new Error("Side chat response does not match its active request.");
    }
    const completed = appendTransientSessionMessages(current, messages, {
      unread: options.unread,
    });
    const next = {
      ...completed,
      runState: "idle" as const,
      requestId: undefined,
      unread: options.unread,
    };
    state.chats.set(current.id, next);
    return cloneSideChat(next);
  }

  markStopping(parentSessionId: string, sideChatId: string, requestId: string): void {
    const state = this.requireParent(parentSessionId);
    const current = this.requireChat(state, sideChatId);
    if (current.requestId !== requestId || current.runState !== "running") return;
    state.chats.set(current.id, { ...current, runState: "stopping" });
  }

  markRunning(parentSessionId: string, sideChatId: string, requestId: string): void {
    const state = this.requireParent(parentSessionId);
    const current = this.requireChat(state, sideChatId);
    if (current.requestId !== requestId || current.runState !== "stopping") return;
    state.chats.set(current.id, { ...current, runState: "running" });
  }

  edit(
    parentSessionId: string,
    sideChatId: string,
    messageIndex: number,
    content: string,
  ): TransientSessionData {
    const state = this.requireParent(parentSessionId);
    const current = this.requireChat(state, sideChatId);
    if (current.runState !== "idle") throw new Error("Stop the side chat before editing it.");
    const next = editTransientSessionUserMessage(current, messageIndex, content);
    state.chats.set(current.id, next);
    return cloneSideChat(next);
  }

  modelMessages(
    parentSessionId: string,
    sideChatId: string,
  ): Array<{
    role: "user" | "assistant" | "system";
    content: string;
  }> {
    return transientSessionModelMessages(
      this.requireChat(this.requireParent(parentSessionId), sideChatId),
    );
  }

  promote(parentSessionId: string, sideChatId: string): SessionData {
    const state = this.requireParent(parentSessionId);
    return promoteTransientSessionFork({
      transient: this.requireChat(state, sideChatId),
    });
  }

  delete(parentSessionId: string, sideChatId: string): SideChatParentState {
    const state = this.requireParent(parentSessionId);
    const current = this.requireChat(state, sideChatId);
    if (current.runState !== "idle") throw new Error("Stop the side chat before deleting it.");
    state.chats.delete(sideChatId);
    if (state.activeSideChatId === sideChatId) {
      state.activeSideChatId = [...state.chats.keys()].at(-1) ?? null;
    }
    if (state.chats.size === 0) state.paneHidden = true;
    return this.list(parentSessionId);
  }

  isParentRunning(parentSessionId: string): boolean {
    const state = this.parents.get(parentSessionId);
    return state ? [...state.chats.values()].some((chat) => chat.runState !== "idle") : false;
  }

  clearParent(parentSessionId: string): void {
    this.parents.delete(parentSessionId);
  }

  clear(): void {
    this.parents.clear();
  }

  private parent(parentSessionId: string): MutableParentState {
    const existing = this.parents.get(parentSessionId);
    if (existing) return existing;
    const state: MutableParentState = {
      activeSideChatId: null,
      paneHidden: true,
      chats: new Map(),
    };
    this.parents.set(parentSessionId, state);
    return state;
  }

  private requireParent(parentSessionId: string): MutableParentState {
    const state = this.parents.get(parentSessionId);
    if (!state) throw new Error(`No side chats exist for Session "${parentSessionId}".`);
    return state;
  }

  private requireChat(state: MutableParentState, sideChatId: string): TransientSessionData {
    const chat = state.chats.get(sideChatId);
    if (!chat) throw new Error(`Unknown side chat: ${sideChatId}`);
    return chat;
  }
}

function cloneSideChat(chat: TransientSessionData): TransientSessionData {
  return structuredClone(chat);
}
