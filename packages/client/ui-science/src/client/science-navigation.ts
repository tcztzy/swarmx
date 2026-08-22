import type { SessionId } from "@deepseek-ai/dsh-client-runtime/client";

export interface ScienceWorkbenchTarget {
  readonly kind: "artifact";
  readonly artifactId: string;
  readonly projectId: string;
  readonly surface: "artifacts";
}

/** Retained per-Session deep link used when rc.8 cannot programmatically switch view tabs. */
export class ScienceNavigationController {
  private disposed = false;
  private readonly listeners = new Map<SessionId, Set<() => void>>();
  private readonly mounted = new Map<SessionId, number>();
  private readonly targets = new Map<SessionId, ScienceWorkbenchTarget>();

  open(sessionId: SessionId, target: ScienceWorkbenchTarget): boolean {
    this.assertActive();
    const current = this.targets.get(sessionId);
    if (
      current?.kind !== target.kind ||
      current.artifactId !== target.artifactId ||
      current.projectId !== target.projectId ||
      current.surface !== target.surface
    ) {
      this.targets.set(sessionId, Object.freeze({ ...target }));
      this.notify(sessionId);
    }
    return (this.mounted.get(sessionId) ?? 0) > 0;
  }

  getSnapshot(sessionId: SessionId): ScienceWorkbenchTarget | null {
    if (this.disposed) return null;
    return this.targets.get(sessionId) ?? null;
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

  mount(sessionId: SessionId): () => void {
    this.assertActive();
    this.mounted.set(sessionId, (this.mounted.get(sessionId) ?? 0) + 1);
    return () => {
      const remaining = (this.mounted.get(sessionId) ?? 1) - 1;
      if (remaining <= 0) this.mounted.delete(sessionId);
      else this.mounted.set(sessionId, remaining);
    };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const listeners = [...this.listeners.values()].flatMap((group) => [...group]);
    this.targets.clear();
    this.mounted.clear();
    this.listeners.clear();
    for (const listener of listeners) listener();
  }

  private assertActive(): void {
    if (this.disposed) throw new Error("Science navigation controller is disposed");
  }

  private notify(sessionId: SessionId): void {
    for (const listener of this.listeners.get(sessionId) ?? []) listener();
  }
}
