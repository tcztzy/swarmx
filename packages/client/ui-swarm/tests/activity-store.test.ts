import type { SessionId } from "@deepseek-ai/dsh-client-runtime/client";
import { describe, expect, it, vi } from "vitest";
import { SwarmActivityStore } from "../src/client/activity-store.js";

const inactive = { kind: "inactive" as const, revision: 0 as const };

describe("V171 Swarm activity subscription", () => {
  it("shares one cancellable wait loop for all renderers of one Session", async () => {
    let waitSignal: AbortSignal | undefined;
    const load = vi.fn(async () => inactive);
    const wait = vi.fn(
      async (_sessionId: SessionId, _afterRevision: number, signal: AbortSignal) => {
        waitSignal = signal;
        return new Promise<typeof inactive>((_resolve, reject) =>
          signal.addEventListener("abort", () => reject(signal.reason), { once: true }),
        );
      },
    );
    const store = new SwarmActivityStore({ load, wait });
    const sessionId = "session-lead" as SessionId;
    const first = store.subscribe(sessionId, vi.fn());
    const second = store.subscribe(sessionId, vi.fn());
    await vi.waitFor(() => expect(wait).toHaveBeenCalledOnce());

    expect(load).toHaveBeenCalledOnce();
    expect(waitSignal?.aborted).toBe(false);
    first();
    expect(waitSignal?.aborted).toBe(false);
    second();
    expect(waitSignal?.aborted).toBe(true);
    store.dispose();
  });
});
