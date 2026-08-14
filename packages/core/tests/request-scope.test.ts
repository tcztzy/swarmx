import { describe, expect, it } from "vitest";
import {
  RequestCancelledError as AcpRequestCancelledError,
  cancelAcpRequest,
  withAcpRequest,
} from "../src/acp.js";
import {
  cancelRequest,
  currentRequestSignal,
  RequestCancelledError,
  registerCurrentRequestParticipant,
  throwIfCurrentRequestCancelled,
  withRequestScope,
} from "../src/request-scope.js";

describe("request scope", () => {
  it("reports cancellation with and without a request id", () => {
    expect(new RequestCancelledError().message).toBe("Request was cancelled.");
    expect(new RequestCancelledError("run_1").message).toBe('Request "run_1" was cancelled.');
  });

  it("preserves normal results and failures while releasing request ids", async () => {
    await expect(withRequestScope("normal", async () => "done")).resolves.toBe("done");
    await expect(withRequestScope("normal", async () => "reused")).resolves.toBe("reused");

    const failure = new Error("failed");
    await expect(
      withRequestScope("failure", async () => {
        throw failure;
      }),
    ).rejects.toBe(failure);
    await expect(cancelRequest("failure")).resolves.toBe(false);
    expect(currentRequestSignal()).toBeUndefined();
    expect(registerCurrentRequestParticipant({ cancel: async () => {}, cleanup: () => {} })).toBe(
      undefined,
    );
  });

  it("validates exclusive bounded request ids", async () => {
    await expect(withRequestScope("", async () => undefined)).rejects.toThrow("non-empty");
    await expect(withRequestScope("   ", async () => undefined)).rejects.toThrow("non-empty");
    await expect(withRequestScope(42 as unknown as string, async () => undefined)).rejects.toThrow(
      "non-empty",
    );
    await expect(withRequestScope("x".repeat(257), async () => undefined)).rejects.toThrow(
      "at most 256",
    );

    const release = deferred<void>();
    const active = withRequestScope("exclusive", () => release.promise);
    await expect(withRequestScope("exclusive", async () => undefined)).rejects.toThrow(
      "already active",
    );
    release.resolve();
    await expect(active).resolves.toBeUndefined();
  });

  it("aborts synchronously and cancels each participant once", async () => {
    const ready = deferred<void>();
    const work = deferred<void>();
    const participantCancellation = deferred<void>();
    let signal: AbortSignal | undefined;
    let cancelCalls = 0;
    let cleanupCalls = 0;
    const run = withRequestScope("cancel-once", async () => {
      signal = currentRequestSignal();
      registerCurrentRequestParticipant({
        async cancel() {
          cancelCalls += 1;
          await participantCancellation.promise;
        },
        cleanup() {
          cleanupCalls += 1;
        },
      });
      ready.resolve();
      await work.promise;
      throwIfCurrentRequestCancelled();
      return "late success";
    });

    await ready.promise;
    const firstCancellation = cancelRequest("cancel-once");
    expect(signal?.aborted).toBe(true);
    expect(signal?.reason).toBeInstanceOf(RequestCancelledError);
    expect(() => throwIfCurrentRequestCancelled()).not.toThrow();
    const repeatedCancellation = cancelRequest("cancel-once");
    expect(cancelCalls).toBe(1);

    participantCancellation.resolve();
    await expect(firstCancellation).resolves.toBe(true);
    await expect(repeatedCancellation).resolves.toBe(true);
    work.resolve();
    await expect(run).rejects.toBe(signal?.reason);
    expect(cleanupCalls).toBe(1);
    await expect(cancelRequest("cancel-once")).resolves.toBe(false);
  });

  it("settles participant cancellation failures and honors unregister", async () => {
    const ready = deferred<void>();
    const work = deferred<void>();
    let failedCancelCalls = 0;
    let removedCancelCalls = 0;
    let removedCleanupCalls = 0;
    const run = withRequestScope("participants", async () => {
      registerCurrentRequestParticipant({
        cancel() {
          failedCancelCalls += 1;
          throw new Error("participant failed");
        },
        cleanup() {},
      });
      const removed = registerCurrentRequestParticipant({
        async cancel() {
          removedCancelCalls += 1;
        },
        cleanup() {
          removedCleanupCalls += 1;
        },
      });
      removed?.unregister();
      ready.resolve();
      await work.promise;
    });

    await ready.promise;
    await expect(cancelRequest("participants")).resolves.toBe(true);
    expect(failedCancelCalls).toBe(1);
    expect(removedCancelCalls).toBe(0);
    work.resolve();
    await expect(run).rejects.toBeInstanceOf(RequestCancelledError);
    expect(removedCleanupCalls).toBe(0);
  });

  it("isolates concurrent and nested async scopes", async () => {
    const firstReady = deferred<void>();
    const firstWork = deferred<void>();
    let firstSignal: AbortSignal | undefined;
    const first = withRequestScope("first", async () => {
      firstSignal = currentRequestSignal();
      firstReady.resolve();
      await firstWork.promise;
    });
    const second = withRequestScope("second", async () => {
      const outerSignal = currentRequestSignal();
      const innerSignal = await withRequestScope("inner", async () => currentRequestSignal());
      expect(innerSignal).not.toBe(outerSignal);
      expect(currentRequestSignal()).toBe(outerSignal);
      return outerSignal;
    });

    await firstReady.promise;
    const secondSignal = await second;
    await expect(cancelRequest("first")).resolves.toBe(true);
    expect(firstSignal?.aborted).toBe(true);
    expect(secondSignal?.aborted).toBe(false);
    firstWork.resolve();
    await expect(first).rejects.toBeInstanceOf(RequestCancelledError);
  });

  it("keeps ACP compatibility aliases on the same scope and error identity", async () => {
    expect(AcpRequestCancelledError).toBe(RequestCancelledError);

    const legacyReady = deferred<void>();
    const legacyWork = deferred<void>();
    const legacy = withAcpRequest("legacy", async () => {
      legacyReady.resolve();
      await legacyWork.promise;
    });
    await legacyReady.promise;
    await expect(cancelRequest("legacy")).resolves.toBe(true);
    legacyWork.resolve();
    await expect(legacy).rejects.toBeInstanceOf(RequestCancelledError);

    const genericReady = deferred<void>();
    const genericWork = deferred<void>();
    const generic = withRequestScope("generic", async () => {
      genericReady.resolve();
      await genericWork.promise;
    });
    await genericReady.promise;
    await expect(cancelAcpRequest("generic")).resolves.toBe(true);
    genericWork.resolve();
    await expect(generic).rejects.toBeInstanceOf(AcpRequestCancelledError);
  });
});

function deferred<T>(): {
  promise: Promise<T>;
  resolve(value: T | PromiseLike<T>): void;
} {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}
