import { cancelAcpRequest, withAcpRequest } from "@swarmx/core";

export interface RequestOwner {
  readonly id: number;
  once(event: "destroyed", listener: () => void): unknown;
  removeListener(event: "destroyed", listener: () => void): unknown;
}

interface ActiveDesktopRequest {
  owner: RequestOwner;
  token: symbol;
  onDestroyed: () => void;
}

/**
 * Owns renderer request IDs in the main process. Ownership prevents one window
 * from stopping another, and token-checked cleanup prevents an old request's
 * finally block from deleting newer state.
 */
export class DesktopRequestRegistry {
  private readonly active = new Map<string, ActiveDesktopRequest>();

  async run<T>(owner: RequestOwner, requestId: string, run: () => Promise<T>): Promise<T> {
    if (this.active.has(requestId)) {
      throw new Error(`Request "${requestId}" is already active.`);
    }

    const token = Symbol(requestId);
    const entry: ActiveDesktopRequest = {
      owner,
      token,
      onDestroyed: () => {
        void this.cancelEntry(requestId, token);
      },
    };
    this.active.set(requestId, entry);
    owner.once("destroyed", entry.onDestroyed);

    try {
      return await withAcpRequest(requestId, run);
    } finally {
      if (this.active.get(requestId)?.token === token) this.active.delete(requestId);
      owner.removeListener("destroyed", entry.onDestroyed);
    }
  }

  async cancel(owner: RequestOwner, requestId: string): Promise<boolean> {
    const entry = this.active.get(requestId);
    if (!entry || entry.owner !== owner) return false;
    return cancelAcpRequest(requestId);
  }

  async cancelAll(owner: RequestOwner): Promise<number> {
    const entries = [...this.active.entries()].filter(([, entry]) => entry.owner === owner);
    const results = await Promise.all(
      entries.map(([requestId, entry]) => this.cancelEntry(requestId, entry.token)),
    );
    return results.filter(Boolean).length;
  }

  private async cancelEntry(requestId: string, token: symbol): Promise<boolean> {
    if (this.active.get(requestId)?.token !== token) return false;
    return cancelAcpRequest(requestId);
  }
}
