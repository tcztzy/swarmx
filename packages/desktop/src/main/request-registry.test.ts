import { EventEmitter } from "node:events";
import { RequestCancelledError, currentRequestSignal } from "@swarmx/core";
import { describe, expect, it } from "vitest";
import { DesktopRequestRegistry } from "./request-registry.js";

describe("DesktopRequestRegistry", () => {
  it("only lets the owning renderer cancel a request and cleans terminal state", async () => {
    const registry = new DesktopRequestRegistry();
    const owner = new FakeOwner(1);
    const stranger = new FakeOwner(2);
    const gate = deferred<void>();
    const run = registry.run(owner, "desktop-owned-cancel", () => gate.promise);

    await expect(registry.cancel(stranger, "desktop-owned-cancel")).resolves.toBe(false);
    await expect(registry.cancel(owner, "desktop-owned-cancel")).resolves.toBe(true);
    await expect(registry.cancel(owner, "desktop-owned-cancel")).resolves.toBe(true);
    gate.resolve();
    await expect(run).rejects.toBeInstanceOf(RequestCancelledError);
    await expect(registry.cancel(owner, "desktop-owned-cancel")).resolves.toBe(false);
    expect(owner.listenerCount("destroyed")).toBe(0);
  });

  it("cancels every request owned by a destroyed window", async () => {
    const registry = new DesktopRequestRegistry();
    const owner = new FakeOwner(3);
    const first = registry.run(owner, "desktop-close-first", waitForCancellation);
    const second = registry.run(owner, "desktop-close-second", waitForCancellation);

    owner.emit("destroyed");

    await expect(first).rejects.toBeInstanceOf(RequestCancelledError);
    await expect(second).rejects.toBeInstanceOf(RequestCancelledError);
    await expect(registry.cancelAll(owner)).resolves.toBe(0);
    expect(owner.listenerCount("destroyed")).toBe(0);
  });

  it("isolates rapid requests and rejects concurrent ID reuse", async () => {
    const registry = new DesktopRequestRegistry();
    const owner = new FakeOwner(4);
    const firstGate = deferred<string>();
    const secondGate = deferred<string>();
    const first = registry.run(owner, "desktop-rapid-first", () => firstGate.promise);
    const second = registry.run(owner, "desktop-rapid-second", () => secondGate.promise);

    await expect(
      registry.run(owner, "desktop-rapid-first", async () => "duplicate"),
    ).rejects.toThrow("already active");
    await expect(registry.cancel(owner, "desktop-rapid-first")).resolves.toBe(true);

    secondGate.resolve("second");
    await expect(second).resolves.toBe("second");
    firstGate.resolve("first");
    await expect(first).rejects.toBeInstanceOf(RequestCancelledError);
    await expect(registry.cancel(owner, "desktop-rapid-second")).resolves.toBe(false);
  });

  it("cleans ownership after failures", async () => {
    const registry = new DesktopRequestRegistry();
    const owner = new FakeOwner(5);

    await expect(
      registry.run(owner, "desktop-failure-cleanup", async () => {
        throw new Error("boom");
      }),
    ).rejects.toThrow("boom");
    await expect(registry.cancel(owner, "desktop-failure-cleanup")).resolves.toBe(false);
    expect(owner.listenerCount("destroyed")).toBe(0);
  });

  it("tracks whether a session has an active request", async () => {
    const registry = new DesktopRequestRegistry();
    const owner = new FakeOwner(6);
    const gate = deferred<string>();
    const run = registry.runForSession(
      {
        owner,
        requestId: "session-request",
        sessionId: "session-1",
      },
      () => gate.promise,
    );

    expect(registry.isSessionActive("session-1")).toBe(true);
    expect(registry.isSessionActive("session-2")).toBe(false);

    gate.resolve("done");
    await expect(run).resolves.toBe("done");
    expect(registry.isSessionActive("session-1")).toBe(false);
  });
});

class FakeOwner extends EventEmitter {
  constructor(readonly id: number) {
    super();
  }
}

function waitForCancellation(): Promise<never> {
  const signal = currentRequestSignal();
  if (!signal) throw new Error("Missing request signal.");
  return new Promise<never>((_resolve, reject) => {
    if (signal.aborted) {
      reject(signal.reason);
      return;
    }
    signal.addEventListener("abort", () => reject(signal.reason), { once: true });
  });
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T | PromiseLike<T>) => void;
} {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}
