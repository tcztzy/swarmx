import type { SessionId } from "@deepseek-ai/dsh-session/types";

export type SideViewMode = "inspect" | "workbench";

export type JsonValue =
  | null
  | boolean
  | number
  | string
  | readonly JsonValue[]
  | { readonly [key: string]: JsonValue };

/** Serializable routing descriptor retained by the generic Side View service. */
export interface SideViewEntry {
  readonly id: string;
  readonly kind: string;
  readonly title: string;
  readonly mode: SideViewMode;
  readonly payload: JsonValue;
}

export interface SideViewSnapshot {
  readonly entries: readonly SideViewEntry[];
  readonly activeId: string | null;
}

export interface SideViewContentOwnerProps {
  readonly entry: SideViewEntry;
}

export interface TurnTailItemOwnerProps {
  readonly turn: number;
}

declare module "@deepseek-ai/cordis" {
  interface Context {
    sideView: ISideView;
  }
}

declare module "@deepseek-ai/dsh-client-ui-slots" {
  interface SlotMap {
    "side-view.content": {
      kind: "keyed";
      scope: "session";
      owner: SideViewContentOwnerProps;
    };
    "conversation.chat.turnTail.items": {
      kind: "list";
      scope: "session";
      owner: TurnTailItemOwnerProps;
    };
  }
}

export interface ISideView {
  open(sessionId: SessionId, entry: SideViewEntry): void;
  activate(sessionId: SessionId, entryId: string): void;
  close(sessionId: SessionId, entryId: string): void;
  dismiss(sessionId: SessionId): void;
  getSnapshot(sessionId: SessionId): SideViewSnapshot;
  subscribe(sessionId: SessionId, listener: () => void): () => void;
}

interface SideViewLayout {
  openDetails(preferredWidth?: number): void;
  closeDetails(): void;
}

const WORKBENCH_DETAILS_WIDTH = 880;

function preferredWidth(entry: SideViewEntry): number | undefined {
  return entry.mode === "workbench" ? WORKBENCH_DETAILS_WIDTH : undefined;
}

const EMPTY_ENTRIES: readonly SideViewEntry[] = Object.freeze([]);
const EMPTY_SNAPSHOT: SideViewSnapshot = Object.freeze({
  entries: EMPTY_ENTRIES,
  activeId: null,
});

function assertJsonValue(value: unknown, seen: Set<object>): asserts value is JsonValue {
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number") {
    if (Number.isFinite(value)) return;
    throw new TypeError("Side View entry payload must be JSON-serializable");
  }
  if (typeof value !== "object") {
    throw new TypeError("Side View entry payload must be JSON-serializable");
  }
  if (seen.has(value)) {
    throw new TypeError("Side View entry payload must be JSON-serializable");
  }
  seen.add(value);
  if (Array.isArray(value)) {
    for (const item of value) assertJsonValue(item, seen);
  } else {
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new TypeError("Side View entry payload must be JSON-serializable");
    }
    for (const item of Object.values(value)) assertJsonValue(item, seen);
  }
  seen.delete(value);
}

function freezeJson(value: JsonValue): JsonValue {
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) {
    return Object.freeze(value.map((item) => freezeJson(item)));
  }
  return Object.freeze(
    Object.fromEntries(Object.entries(value).map(([key, item]) => [key, freezeJson(item)])),
  );
}

function normalizeEntry(entry: SideViewEntry): SideViewEntry {
  if (entry.id.trim().length === 0 || entry.kind.trim().length === 0) {
    throw new TypeError("Side View entry id and kind must be non-empty");
  }
  if (entry.title.trim().length === 0) {
    throw new TypeError("Side View entry title must be non-empty");
  }
  assertJsonValue(entry.payload, new Set());
  return Object.freeze({
    id: entry.id,
    kind: entry.kind,
    title: entry.title,
    mode: entry.mode,
    payload: freezeJson(structuredClone(entry.payload)),
  });
}

/** Client-only tab state and layout orchestration; content remains slot-owned. */
export class SideViewController implements ISideView {
  private disposed = false;
  private readonly listeners = new Map<SessionId, Set<() => void>>();
  private readonly sessions = new Map<SessionId, SideViewSnapshot>();

  constructor(private readonly layout: SideViewLayout) {}

  open(sessionId: SessionId, entry: SideViewEntry): void {
    this.assertActive();
    const nextEntry = normalizeEntry(entry);
    const current = this.getSnapshot(sessionId);
    const index = current.entries.findIndex((candidate) => candidate.id === nextEntry.id);
    const entries =
      index < 0
        ? [...current.entries, nextEntry]
        : current.entries.map((candidate, candidateIndex) =>
            candidateIndex === index ? nextEntry : candidate,
          );
    this.write(sessionId, entries, nextEntry.id);
    this.layout.openDetails(preferredWidth(nextEntry));
  }

  activate(sessionId: SessionId, entryId: string): void {
    this.assertActive();
    const current = this.getSnapshot(sessionId);
    if (!current.entries.some((entry) => entry.id === entryId)) {
      throw new Error(`Side View entry ${entryId} is not open in Session ${String(sessionId)}`);
    }
    if (current.activeId !== entryId) this.write(sessionId, current.entries, entryId);
    const entry = current.entries.find((candidate) => candidate.id === entryId);
    this.layout.openDetails(entry === undefined ? undefined : preferredWidth(entry));
  }

  close(sessionId: SessionId, entryId: string): void {
    this.assertActive();
    const current = this.getSnapshot(sessionId);
    const index = current.entries.findIndex((entry) => entry.id === entryId);
    if (index < 0) return;
    const entries = current.entries.filter((entry) => entry.id !== entryId);
    const activeId =
      current.activeId !== entryId
        ? current.activeId
        : (entries[Math.min(index, entries.length - 1)]?.id ?? null);
    this.write(sessionId, entries, activeId);
    if (entries.length === 0) this.layout.closeDetails();
  }

  dismiss(_sessionId: SessionId): void {
    this.assertActive();
    this.layout.closeDetails();
  }

  getSnapshot(sessionId: SessionId): SideViewSnapshot {
    if (this.disposed) return EMPTY_SNAPSHOT;
    return this.sessions.get(sessionId) ?? EMPTY_SNAPSHOT;
  }

  subscribe(sessionId: SessionId, listener: () => void): () => void {
    if (this.disposed) return () => {};
    let listeners = this.listeners.get(sessionId);
    if (listeners === undefined) {
      listeners = new Set();
      this.listeners.set(sessionId, listeners);
    }
    listeners.add(listener);
    return () => {
      listeners?.delete(listener);
      if (listeners?.size === 0) this.listeners.delete(sessionId);
    };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const listeners = [...this.listeners.values()].flatMap((group) => [...group]);
    this.sessions.clear();
    this.listeners.clear();
    this.layout.closeDetails();
    for (const listener of listeners) listener();
  }

  private assertActive(): void {
    if (this.disposed) throw new Error("Side View controller is disposed");
  }

  private write(
    sessionId: SessionId,
    entries: readonly SideViewEntry[],
    activeId: string | null,
  ): void {
    const snapshot = Object.freeze({ entries: Object.freeze([...entries]), activeId });
    if (entries.length === 0) this.sessions.delete(sessionId);
    else this.sessions.set(sessionId, snapshot);
    for (const listener of this.listeners.get(sessionId) ?? []) listener();
  }
}
