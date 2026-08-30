import { createHash } from "node:crypto";
import type {
  ConversationRuntime,
  RuntimeEventListener,
  RuntimeKind,
  TurnStatus,
} from "./contracts.js";

const MAX_ACTIVE_COMPLETED_ITEMS = 10_000;
const MAX_TERMINAL_TURNS = 10_000;

/** Peer native conversation adapters available beneath the DSH Web host. */
export class ConversationRuntimeRegistry {
  readonly defaultKind: RuntimeKind;
  private readonly adapters = new Map<RuntimeKind, ConversationRuntime>();
  private readonly listeners = new Set<RuntimeEventListener>();
  private readonly adapterSequences = new Map<RuntimeKind, number>();
  private readonly completedItems = new Map<string, string>();
  private readonly completedItemsByTurn = new Map<string, Set<string>>();
  private readonly terminalTurns = new Map<string, TurnStatus>();
  private subscriptions: Array<() => void> | undefined;
  private seq = 0;
  private disposed: Promise<void> | undefined;

  constructor(adapters: readonly ConversationRuntime[], defaultKind: RuntimeKind) {
    for (const adapter of adapters) {
      if (this.adapters.has(adapter.kind)) {
        throw new Error(`Conversation runtime "${adapter.kind}" is registered more than once.`);
      }
      this.adapters.set(adapter.kind, adapter);
    }
    if (!this.adapters.has(defaultKind)) {
      throw new Error(`Default conversation runtime "${defaultKind}" is not registered.`);
    }
    this.defaultKind = defaultKind;
  }

  kinds(): RuntimeKind[] {
    return [...this.adapters.keys()];
  }

  runtime(kind: RuntimeKind): ConversationRuntime {
    const runtime = this.adapters.get(kind);
    if (runtime === undefined) {
      throw new RuntimeNotRegisteredError(kind);
    }
    return runtime;
  }

  subscribe(listener: RuntimeEventListener): () => void {
    if (this.disposed !== undefined) throw new Error("Conversation runtime registry is disposed.");
    this.listeners.add(listener);
    if (this.subscriptions === undefined) this.startSubscriptions();
    let active = true;
    return () => {
      if (!active) return;
      active = false;
      this.listeners.delete(listener);
      if (this.listeners.size === 0) this.stopSubscriptions();
    };
  }

  dispose(): Promise<void> {
    this.stopSubscriptions();
    this.listeners.clear();
    this.disposed ??= Promise.allSettled(
      [...this.adapters.values()].reverse().map((runtime) => runtime.dispose()),
    ).then((results) => {
      const failure = results.find(
        (result): result is PromiseRejectedResult => result.status === "rejected",
      );
      if (failure !== undefined) throw failure.reason;
    });
    return this.disposed;
  }

  private startSubscriptions(): void {
    const subscriptions: Array<() => void> = [];
    try {
      for (const runtime of this.adapters.values()) {
        subscriptions.push(
          runtime.subscribe((event) => {
            this.assertAdapterEvent(runtime.kind, event);
            if (!this.acceptEvent(event)) return;
            const sequenced = { ...event, seq: ++this.seq };
            for (const listener of this.listeners) listener(sequenced);
          }),
        );
      }
      this.subscriptions = subscriptions;
    } catch (error) {
      for (const unsubscribe of subscriptions.reverse()) unsubscribe();
      throw error;
    }
  }

  private stopSubscriptions(): void {
    if (this.subscriptions === undefined) return;
    for (const unsubscribe of this.subscriptions.reverse()) unsubscribe();
    this.subscriptions = undefined;
  }

  private assertAdapterEvent(
    runtime: RuntimeKind,
    event: Parameters<RuntimeEventListener>[0],
  ): void {
    if (event.runtime !== runtime) {
      throw new Error(
        `Conversation runtime "${runtime}" emitted an event qualified as "${event.runtime}".`,
      );
    }
    const previous = this.adapterSequences.get(runtime) ?? 0;
    if (!Number.isSafeInteger(event.seq) || event.seq <= previous) {
      throw new Error(
        `Conversation runtime "${runtime}" event sequence ${String(event.seq)} must be greater than ${String(previous)}.`,
      );
    }
    this.adapterSequences.set(runtime, event.seq);
  }

  private acceptEvent(event: Parameters<RuntimeEventListener>[0]): boolean {
    const eventTurnKey = turnKey(event.runtime, event.conversationId, event.turnId);
    if (event.type === "item_delta") {
      if (this.terminalTurns.has(eventTurnKey)) return false;
      return !this.completedItems.has(itemKey(event.runtime, event.conversationId, event.itemId));
    }
    if (event.type === "turn_status") {
      return this.acceptTurnStatus(eventTurnKey, event.status);
    }
    if (event.type !== "item_completed") return true;
    if (event.item.turnId !== event.turnId) {
      throw new Error(`Completed item "${event.item.id}" has a mismatched turn id.`);
    }
    if (this.terminalTurns.has(eventTurnKey)) return false;
    const key = itemKey(event.runtime, event.conversationId, event.item.id);
    const digest = createHash("sha256").update(JSON.stringify(event.item)).digest("hex");
    const previous = this.completedItems.get(key);
    if (previous === digest) return false;
    if (previous !== undefined) {
      throw new Error(`Contradictory completed item "${event.item.id}".`);
    }
    if (this.completedItems.size >= MAX_ACTIVE_COMPLETED_ITEMS) {
      throw new Error(
        `Conversation runtime active completed-item limit ${String(MAX_ACTIVE_COMPLETED_ITEMS)} exceeded.`,
      );
    }
    this.completedItems.set(key, digest);
    const turnItems = this.completedItemsByTurn.get(eventTurnKey) ?? new Set<string>();
    turnItems.add(key);
    this.completedItemsByTurn.set(eventTurnKey, turnItems);
    return true;
  }

  private acceptTurnStatus(key: string, status: TurnStatus): boolean {
    const terminal = this.terminalTurns.get(key);
    if (status === "running") {
      if (terminal !== undefined) {
        throw new Error(`Terminal conversation turn cannot return to running status.`);
      }
      return true;
    }
    if (terminal === status) return false;
    if (terminal !== undefined) {
      throw new Error(`Conversation turn has contradictory terminal statuses.`);
    }
    if (this.terminalTurns.size >= MAX_TERMINAL_TURNS) {
      throw new Error(
        `Conversation runtime terminal-turn limit ${String(MAX_TERMINAL_TURNS)} exceeded.`,
      );
    }
    this.terminalTurns.set(key, status);
    for (const item of this.completedItemsByTurn.get(key) ?? []) this.completedItems.delete(item);
    this.completedItemsByTurn.delete(key);
    return true;
  }
}

function itemKey(runtime: RuntimeKind, conversationId: string, itemId: string): string {
  return JSON.stringify([runtime, conversationId, itemId]);
}

function turnKey(runtime: RuntimeKind, conversationId: string, turnId: string): string {
  return JSON.stringify([runtime, conversationId, turnId]);
}

export class RuntimeNotRegisteredError extends Error {
  readonly status = 404;

  constructor(readonly runtimeKind: RuntimeKind) {
    super(`Conversation runtime "${runtimeKind}" is not registered.`);
    this.name = "RuntimeNotRegisteredError";
  }
}
