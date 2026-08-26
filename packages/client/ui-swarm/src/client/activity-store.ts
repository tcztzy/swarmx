import type { SessionId } from "@deepseek-ai/dsh-client-runtime/client";
import type { SwarmUiSnapshot } from "@swarmx/dsh-swarm/contracts";

export interface SwarmActivityRemote {
  load(sessionId: SessionId, signal: AbortSignal): Promise<SwarmUiSnapshot>;
  wait(sessionId: SessionId, afterRevision: number, signal: AbortSignal): Promise<SwarmUiSnapshot>;
}

export type SwarmActivityState =
  | { readonly kind: "loading" }
  | { readonly kind: "error"; readonly error: Error }
  | { readonly kind: "ready"; readonly snapshot: SwarmUiSnapshot };

interface SessionActivity {
  readonly controller: AbortController;
  readonly listeners: Set<() => void>;
  state: SwarmActivityState;
}

const LOADING: SwarmActivityState = Object.freeze({ kind: "loading" });

/** One cancellable revision wait loop per rendered Session, shared by header and Side View. */
export class SwarmActivityStore {
  private readonly sessions = new Map<SessionId, SessionActivity>();
  private disposed = false;

  constructor(private readonly remote: SwarmActivityRemote) {}

  getSnapshot(sessionId: SessionId): SwarmActivityState {
    return this.sessions.get(sessionId)?.state ?? LOADING;
  }

  subscribe(sessionId: SessionId, listener: () => void): () => void {
    if (this.disposed) return () => undefined;
    let activity = this.sessions.get(sessionId);
    if (!activity) {
      activity = {
        controller: new AbortController(),
        listeners: new Set(),
        state: LOADING,
      };
      this.sessions.set(sessionId, activity);
      void this.run(sessionId, activity);
    }
    activity.listeners.add(listener);
    return () => {
      activity?.listeners.delete(listener);
      if (activity?.listeners.size !== 0) return;
      activity?.controller.abort();
      this.sessions.delete(sessionId);
    };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const activity of this.sessions.values()) activity.controller.abort();
    this.sessions.clear();
  }

  private publish(activity: SessionActivity, state: SwarmActivityState): void {
    activity.state = state;
    for (const listener of activity.listeners) listener();
  }

  private async run(sessionId: SessionId, activity: SessionActivity): Promise<void> {
    const { signal } = activity.controller;
    try {
      let snapshot = await this.remote.load(sessionId, signal);
      if (signal.aborted) return;
      this.publish(activity, { kind: "ready", snapshot });
      while (!signal.aborted && snapshot.kind !== "archived") {
        snapshot = await this.remote.wait(sessionId, snapshot.revision, signal);
        if (signal.aborted) return;
        this.publish(activity, { kind: "ready", snapshot });
      }
    } catch (error) {
      if (signal.aborted) return;
      this.publish(activity, {
        error: error instanceof Error ? error : new Error("Swarm activity could not be read"),
        kind: "error",
      });
    }
  }
}
