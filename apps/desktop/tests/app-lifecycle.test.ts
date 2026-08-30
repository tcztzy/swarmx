import { describe, expect, it, vi } from "vitest";
import {
  acquirePrimaryInstance,
  activateOwnedResource,
  createBeforeQuitHandler,
  disposeAfterFailure,
  disposeOwnedResource,
  onceFailureReporter,
} from "../src/app-lifecycle.js";

function deferred(): {
  promise: Promise<void>;
  resolve(): void;
  reject(error: unknown): void;
} {
  let resolve!: () => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<void>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

describe("application quit coordination", () => {
  it("disposes an activated platform when initial window creation fails", async () => {
    const startupError = new Error("window creation failed");
    const dispose = vi.fn(async () => undefined);

    await expect(
      activateOwnedResource(Promise.resolve({ dispose }), () => {
        throw startupError;
      }),
    ).rejects.toBe(startupError);
    expect(dispose).toHaveBeenCalledOnce();
  });

  it("preserves both surface and platform cleanup failures", async () => {
    const surfaceError = new Error("window creation failed");
    const cleanupError = new Error("platform cleanup failed");

    const failure = await disposeAfterFailure(
      {
        dispose: vi.fn(async () => {
          throw cleanupError;
        }),
      },
      surfaceError,
    ).catch((error: unknown) => error);

    expect(failure).toBeInstanceOf(AggregateError);
    expect(failure).toMatchObject({
      errors: [surfaceError, cleanupError],
      cause: surfaceError,
    });
  });

  it("reports one fatal error when startup and quit observe the same rejection", () => {
    const report = vi.fn();
    const reportOnce = onceFailureReporter(report);
    const failure = new Error("platform boot failed");

    reportOnce(failure);
    reportOnce(failure);

    expect(report).toHaveBeenCalledOnce();
    expect(report).toHaveBeenCalledWith(failure);
  });

  it("lets only the primary product-state owner boot and focuses it on a second launch", () => {
    let secondInstance: (() => void) | undefined;
    const application = {
      requestSingleInstanceLock: vi.fn(() => true),
      quit: vi.fn(),
      on: vi.fn((_event: "second-instance", listener: () => void) => {
        secondInstance = listener;
      }),
    };
    const window = {
      focus: vi.fn(),
      isMinimized: vi.fn(() => true),
      restore: vi.fn(),
      show: vi.fn(),
    };

    expect(acquirePrimaryInstance(application, () => [window])).toBe(true);
    secondInstance?.();

    expect(application.quit).not.toHaveBeenCalled();
    expect(window.restore).toHaveBeenCalledOnce();
    expect(window.show).toHaveBeenCalledOnce();
    expect(window.focus).toHaveBeenCalledOnce();

    const secondary = {
      requestSingleInstanceLock: vi.fn(() => false),
      quit: vi.fn(),
      on: vi.fn(),
    };
    expect(acquirePrimaryInstance(secondary, () => [])).toBe(false);
    expect(secondary.quit).toHaveBeenCalledOnce();
    expect(secondary.on).not.toHaveBeenCalled();
  });

  it("holds every pending quit event and resumes application quit exactly once", async () => {
    const cleanup = deferred();
    const dispose = vi.fn(() => cleanup.promise);
    const quit = vi.fn();
    const reportFailure = vi.fn();
    const handleBeforeQuit = createBeforeQuitHandler({ dispose, quit, reportFailure });
    const firstEvent = { preventDefault: vi.fn() };
    const repeatedEvent = { preventDefault: vi.fn() };

    handleBeforeQuit(firstEvent);
    handleBeforeQuit(repeatedEvent);

    expect(firstEvent.preventDefault).toHaveBeenCalledOnce();
    expect(repeatedEvent.preventDefault).toHaveBeenCalledOnce();
    expect(dispose).toHaveBeenCalledOnce();
    expect(quit).not.toHaveBeenCalled();

    cleanup.resolve();
    await cleanup.promise;
    await Promise.resolve();

    expect(quit).toHaveBeenCalledOnce();
    expect(reportFailure).not.toHaveBeenCalled();

    const resumedQuitEvent = { preventDefault: vi.fn() };
    handleBeforeQuit(resumedQuitEvent);
    expect(resumedQuitEvent.preventDefault).not.toHaveBeenCalled();
    expect(quit).toHaveBeenCalledOnce();
  });

  it("waits for an in-flight platform boot before disposing an exit-time owner", async () => {
    let resolveBoot!: (resource: { dispose(): Promise<void> }) => void;
    const boot = new Promise<{ dispose(): Promise<void> }>((resolve) => {
      resolveBoot = resolve;
    });
    const dispose = vi.fn(async () => undefined);
    const cleanup = disposeOwnedResource(
      () => undefined,
      () => boot,
    );
    expect(dispose).not.toHaveBeenCalled();

    resolveBoot({ dispose });
    await cleanup;

    expect(dispose).toHaveBeenCalledOnce();
  });

  it("reports rejected cleanup without resuming quit", async () => {
    const cleanup = deferred();
    const failure = new Error("cleanup failed");
    const quit = vi.fn();
    const reportFailure = vi.fn();
    const handleBeforeQuit = createBeforeQuitHandler({
      dispose: () => cleanup.promise,
      quit,
      reportFailure,
    });
    const event = { preventDefault: vi.fn() };

    handleBeforeQuit(event);
    cleanup.reject(failure);
    await expect(cleanup.promise).rejects.toBe(failure);
    await Promise.resolve();

    expect(event.preventDefault).toHaveBeenCalledOnce();
    expect(reportFailure).toHaveBeenCalledOnce();
    expect(reportFailure).toHaveBeenCalledWith(failure);
    expect(quit).not.toHaveBeenCalled();
  });

  it("reports a synchronous cleanup failure", () => {
    const failure = new Error("synchronous cleanup failure");
    const quit = vi.fn();
    const reportFailure = vi.fn();
    const handleBeforeQuit = createBeforeQuitHandler({
      dispose: () => {
        throw failure;
      },
      quit,
      reportFailure,
    });
    const event = { preventDefault: vi.fn() };

    handleBeforeQuit(event);

    expect(event.preventDefault).toHaveBeenCalledOnce();
    expect(reportFailure).toHaveBeenCalledWith(failure);
    expect(quit).not.toHaveBeenCalled();
  });
});
