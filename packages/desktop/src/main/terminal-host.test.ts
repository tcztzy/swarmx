import type { IDisposable, IPty } from "node-pty";
import { describe, expect, it, vi } from "vitest";
import {
  type TerminalAuditCallback,
  type TerminalAuditEvent,
  TerminalHost,
  type TerminalOwner,
  type TerminalProcessFactory,
} from "./terminal-host.js";

describe("TerminalHost", () => {
  it("starts the user's login shell in the requested project and forwards PTY events", () => {
    const process = new FakePty();
    const factory = fakeFactory(process);
    const owner = fakeOwner(7);
    const host = new TerminalHost(factory, "darwin", {
      SHELL: "/bin/zsh",
      HOME: "/Users/test",
    });

    const created = host.create(owner, { cwd: "/workspace/swarmx", cols: 120, rows: 32 });

    expect(factory.spawn).toHaveBeenCalledWith(
      "/bin/zsh",
      ["-l"],
      expect.objectContaining({
        cwd: "/workspace/swarmx",
        cols: 120,
        rows: 32,
        name: "xterm-256color",
        env: expect.objectContaining({ TERM: "xterm-256color", TERM_PROGRAM: "SwarmX" }),
      }),
    );

    process.emitData("ready\r\n");
    expect(owner.send).toHaveBeenCalledWith("terminal:data", {
      id: created.id,
      data: "ready\r\n",
    });

    process.emitExit({ exitCode: 0, signal: 1 });
    expect(owner.send).toHaveBeenCalledWith("terminal:exit", {
      id: created.id,
      exitCode: 0,
      signal: 1,
    });
  });

  it("allows only the owning renderer to write, resize, or kill a terminal", () => {
    const process = new FakePty();
    const owner = fakeOwner(3);
    const host = new TerminalHost(fakeFactory(process), "darwin", { SHELL: "/bin/zsh" });
    const { id } = host.create(owner, { cwd: "/workspace" });

    expect(host.write(4, id, "blocked")).toBe(false);
    expect(host.resize(4, id, 90, 30)).toBe(false);
    expect(host.kill(4, id)).toBe(false);
    expect(process.write).not.toHaveBeenCalled();
    expect(process.resize).not.toHaveBeenCalled();
    expect(process.kill).not.toHaveBeenCalled();

    expect(host.write(3, id, "pwd\r")).toBe(true);
    expect(host.resize(3, id, 90, 30)).toBe(true);
    expect(process.write).toHaveBeenCalledWith("pwd\r");
    expect(process.resize).toHaveBeenCalledWith(90, 30);
    expect(host.kill(3, id)).toBe(true);
    expect(process.kill).toHaveBeenCalledOnce();
  });

  it("kills every terminal owned by a renderer when that renderer exits", () => {
    const first = new FakePty();
    const second = new FakePty();
    const factory = {
      spawn: vi.fn().mockReturnValueOnce(first).mockReturnValueOnce(second),
    } satisfies TerminalProcessFactory;
    const host = new TerminalHost(factory, "linux", { SHELL: "/bin/bash" });

    host.create(fakeOwner(1), { cwd: "/workspace/a" });
    const other = host.create(fakeOwner(2), { cwd: "/workspace/b" });
    host.cleanupOwner(1);

    expect(first.kill).toHaveBeenCalledOnce();
    expect(second.kill).not.toHaveBeenCalled();
    expect(host.write(2, other.id, "still alive")).toBe(true);
  });

  it("continues owner cleanup after one Terminal release fails", () => {
    const first = new FakePty();
    const second = new FakePty();
    const cleanupError = new Error("first Terminal cleanup failed");
    first.kill.mockImplementation(() => {
      throw cleanupError;
    });
    const factory = {
      spawn: vi.fn().mockReturnValueOnce(first).mockReturnValueOnce(second),
    } satisfies TerminalProcessFactory;
    const host = new TerminalHost(factory, "linux", { SHELL: "/bin/bash" });
    const owner = fakeOwner(5);
    host.create(owner, { id: "first-terminal", cwd: "/workspace/a" });
    host.create(owner, { id: "second-terminal", cwd: "/workspace/b" });

    expect(() => host.cleanupOwner(owner.id)).toThrow(cleanupError);

    expect(first.kill).toHaveBeenCalledOnce();
    expect(second.kill).toHaveBeenCalledOnce();
    expect(host.kill(owner.id, "first-terminal")).toBe(false);
    expect(host.kill(owner.id, "second-terminal")).toBe(false);
  });

  it("audits terminal operations before their side effects and correlates owner and terminal", () => {
    const process = new FakePty();
    const events: TerminalAuditEvent[] = [];
    const timeline: string[] = [];
    const factory = {
      spawn: vi.fn(() => {
        timeline.push("spawn");
        return process;
      }),
    } satisfies TerminalProcessFactory;
    const audit: TerminalAuditCallback = (event) => {
      events.push({ ...event });
      timeline.push(`${event.operation}:${event.phase}:${event.outcome ?? ""}`);
    };
    process.write.mockImplementation(() => timeline.push("pty:write"));
    process.resize.mockImplementation(() => timeline.push("pty:resize"));
    process.kill.mockImplementation(() => timeline.push("pty:kill"));
    const host = new TerminalHost(factory, "darwin", { SHELL: "/bin/zsh" }, audit);

    const created = host.create(fakeOwner(17), {
      id: "terminal-17",
      cwd: "/workspace",
      cols: 90,
      rows: 30,
    });
    host.write(17, created.id, "pwd\r");
    host.resize(17, created.id, 100, 40);
    host.kill(17, created.id);

    expect(timeline).toEqual([
      "create:attempt:",
      "spawn",
      "create:outcome:succeeded",
      "write:attempt:",
      "pty:write",
      "write:outcome:succeeded",
      "resize:attempt:",
      "pty:resize",
      "resize:outcome:succeeded",
      "close:attempt:",
      "pty:kill",
      "close:outcome:succeeded",
    ]);
    expect(events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          operation: "create",
          phase: "outcome",
          outcome: "succeeded",
          terminalId: "terminal-17",
          ownerId: 17,
          pid: 42,
        }),
        expect.objectContaining({
          operation: "write",
          phase: "attempt",
          terminalId: "terminal-17",
          ownerId: 17,
          byteCount: 4,
        }),
        expect.objectContaining({
          operation: "close",
          phase: "attempt",
          terminalId: "terminal-17",
          ownerId: 17,
          closeReason: "user_kill",
        }),
      ]),
    );
  });

  it("marks only the invoke operation after its semantic audit sink succeeds", () => {
    const process = new FakePty();
    const events: TerminalAuditEvent[] = [];
    const recordSemanticAudit = vi.fn();
    const host = new TerminalHost(
      fakeFactory(process),
      "darwin",
      { SHELL: "/bin/zsh" },
      collectAudit(events),
    );

    const { id } = host.create(
      fakeOwner(18),
      { id: "terminal-18", cwd: "/workspace" },
      recordSemanticAudit,
    );
    expect(recordSemanticAudit).toHaveBeenCalledOnce();

    const markedAfterCreate = recordSemanticAudit.mock.calls.length;
    process.emitExit({ exitCode: 0 });
    expect(recordSemanticAudit).toHaveBeenCalledTimes(markedAfterCreate);
    expect(events.at(-1)).toMatchObject({ operation: "exit", outcome: "succeeded" });

    const blockedMarker = vi.fn();
    const blockedHost = new TerminalHost(
      fakeFactory(new FakePty()),
      "darwin",
      { SHELL: "/bin/zsh" },
      () => {
        throw new Error("audit unavailable");
      },
    );
    expect(() =>
      blockedHost.create(fakeOwner(19), { id: `${id}-blocked`, cwd: "/workspace" }, blockedMarker),
    ).toThrow("audit unavailable");
    expect(blockedMarker).not.toHaveBeenCalled();

    const failedOutcomeMarker = vi.fn();
    const failedOutcomeHost = new TerminalHost(
      fakeFactory(new FakePty()),
      "darwin",
      { SHELL: "/bin/zsh" },
      (event) => {
        if (event.phase === "outcome") throw new Error("outcome audit unavailable");
      },
    );
    expect(() =>
      failedOutcomeHost.create(
        fakeOwner(20),
        { id: "invalid-outcome", cwd: "" },
        failedOutcomeMarker,
      ),
    ).toThrow("outcome audit unavailable");
    expect(failedOutcomeMarker).not.toHaveBeenCalled();
  });

  it("releases a spawned PTY when event subscription setup fails", () => {
    const brokenProcess = new FakePty();
    const healthyProcess = new FakePty();
    const dataSubscription = { dispose: vi.fn() };
    const setupError = new Error("exit subscription unavailable");
    vi.spyOn(brokenProcess, "onData").mockReturnValue(dataSubscription);
    vi.spyOn(brokenProcess, "onExit").mockImplementation(() => {
      throw setupError;
    });
    const factory = {
      spawn: vi.fn().mockReturnValueOnce(brokenProcess).mockReturnValueOnce(healthyProcess),
    } satisfies TerminalProcessFactory;
    const events: TerminalAuditEvent[] = [];
    const marker = vi.fn();
    const owner = fakeOwner(20);
    const host = new TerminalHost(factory, "darwin", { SHELL: "/bin/zsh" }, collectAudit(events));

    expect(() => host.create(owner, { id: "partial-terminal", cwd: "/workspace" }, marker)).toThrow(
      setupError,
    );
    expect(dataSubscription.dispose).toHaveBeenCalledOnce();
    expect(brokenProcess.kill).toHaveBeenCalledOnce();
    expect(owner.send).not.toHaveBeenCalled();
    expect(marker).toHaveBeenCalledOnce();
    expect(events).toEqual([
      expect.objectContaining({ operation: "create", phase: "attempt" }),
      expect.objectContaining({
        operation: "create",
        phase: "outcome",
        outcome: "failed",
        reason: "create_failed",
      }),
    ]);

    expect(host.create(owner, { id: "partial-terminal", cwd: "/workspace" })).toMatchObject({
      id: "partial-terminal",
      pid: 42,
    });
  });

  it("preserves the setup error while attempting every failed-create cleanup", () => {
    const process = new FakePty();
    const setupError = new Error("secret setup failure");
    const dataSubscription = {
      dispose: vi.fn(() => {
        throw new Error("secret dispose failure");
      }),
    };
    vi.spyOn(process, "onData").mockReturnValue(dataSubscription);
    vi.spyOn(process, "onExit").mockImplementation(() => {
      throw setupError;
    });
    process.kill.mockImplementation(() => {
      throw new Error("secret kill failure");
    });
    const warning = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const events: TerminalAuditEvent[] = [];
    const host = new TerminalHost(
      fakeFactory(process),
      "darwin",
      { SHELL: "/bin/zsh" },
      collectAudit(events),
    );

    try {
      expect(() =>
        host.create(fakeOwner(21), { id: "secret-terminal-id", cwd: "/secret/cwd" }),
      ).toThrow(setupError);
      expect(dataSubscription.dispose).toHaveBeenCalledOnce();
      expect(process.kill).toHaveBeenCalledOnce();
      expect(warning).toHaveBeenCalledWith(
        "Failed to fully release a partially created Terminal process.",
      );
      expect(JSON.stringify(warning.mock.calls)).not.toContain("secret");
      expect(events.at(-1)).toMatchObject({
        operation: "create",
        outcome: "failed",
        reason: "create_failed",
      });
    } finally {
      warning.mockRestore();
    }
  });

  it("audits cleanup and dispose outcomes for each affected terminal", () => {
    const first = new FakePty();
    const second = new FakePty();
    const third = new FakePty();
    const events: TerminalAuditEvent[] = [];
    const factory = {
      spawn: vi
        .fn()
        .mockReturnValueOnce(first)
        .mockReturnValueOnce(second)
        .mockReturnValueOnce(third),
    } satisfies TerminalProcessFactory;
    const host = new TerminalHost(factory, "linux", { SHELL: "/bin/bash" }, collectAudit(events));

    host.create(fakeOwner(1), { id: "cleanup-terminal", cwd: "/workspace/a" });
    host.create(fakeOwner(2), { id: "dispose-terminal", cwd: "/workspace/b" });
    host.create(fakeOwner(3), { id: "exit-terminal", cwd: "/workspace/c" });
    events.length = 0;

    host.cleanupOwner(1);
    host.dispose();

    expect(events).toEqual([
      {
        operation: "close",
        phase: "attempt",
        terminalId: "cleanup-terminal",
        ownerId: 1,
        closeReason: "owner_cleanup",
      },
      {
        operation: "close",
        phase: "outcome",
        outcome: "succeeded",
        terminalId: "cleanup-terminal",
        ownerId: 1,
        closeReason: "owner_cleanup",
      },
      {
        operation: "close",
        phase: "attempt",
        terminalId: "dispose-terminal",
        ownerId: 2,
        closeReason: "app_dispose",
      },
      {
        operation: "close",
        phase: "outcome",
        outcome: "succeeded",
        terminalId: "dispose-terminal",
        ownerId: 2,
        closeReason: "app_dispose",
      },
      {
        operation: "close",
        phase: "attempt",
        terminalId: "exit-terminal",
        ownerId: 3,
        closeReason: "app_dispose",
      },
      {
        operation: "close",
        phase: "outcome",
        outcome: "succeeded",
        terminalId: "exit-terminal",
        ownerId: 3,
        closeReason: "app_dispose",
      },
    ]);
  });

  it("records exit outcomes without retaining terminal content, cwd, or environment", () => {
    const process = new FakePty();
    const events: TerminalAuditEvent[] = [];
    const host = new TerminalHost(
      fakeFactory(process),
      "darwin",
      { SHELL: "/bin/zsh", API_TOKEN: "environment-secret" },
      collectAudit(events),
    );
    const input = "stdin-secret-密钥\r";
    const { id } = host.create(fakeOwner(8), {
      id: "safe-terminal-id",
      cwd: "/workspace/cwd-secret",
    });

    host.write(8, id, input);
    process.emitData("stdout-secret\r\n");
    process.emitExit({ exitCode: 0 });

    expect(events).toContainEqual({
      operation: "exit",
      phase: "outcome",
      terminalId: "safe-terminal-id",
      ownerId: 8,
      outcome: "succeeded",
      exitCode: 0,
      signal: null,
    });
    expect(events).toContainEqual(
      expect.objectContaining({
        operation: "write",
        phase: "attempt",
        byteCount: Buffer.byteLength(input, "utf8"),
      }),
    );
    const serialized = JSON.stringify(events);
    expect(serialized).not.toContain(input);
    expect(serialized).not.toContain("stdout-secret");
    expect(serialized).not.toContain("cwd-secret");
    expect(serialized).not.toContain("environment-secret");
    expect(serialized).not.toContain("API_TOKEN");
  });

  it("fails closed when an attempt audit cannot be recorded", () => {
    const process = new FakePty();
    const factory = fakeFactory(process);
    let blockedOperation: TerminalAuditEvent["operation"] | undefined = "create";
    let blockedCloseReason: TerminalAuditEvent["closeReason"];
    const audit: TerminalAuditCallback = (event) => {
      if (
        event.operation === blockedOperation &&
        event.phase === "attempt" &&
        (!blockedCloseReason || event.closeReason === blockedCloseReason)
      ) {
        throw new Error(
          `audit unavailable for ${event.operation}${event.closeReason ? `:${event.closeReason}` : ""}`,
        );
      }
    };
    const host = new TerminalHost(factory, "darwin", { SHELL: "/bin/zsh" }, audit);

    expect(() => host.create(fakeOwner(12), { id: "blocked-create", cwd: "/workspace" })).toThrow(
      "audit unavailable for create",
    );
    expect(factory.spawn).not.toHaveBeenCalled();

    blockedOperation = undefined;
    const { id } = host.create(fakeOwner(12), { id: "active-terminal", cwd: "/workspace" });

    blockedOperation = "write";
    expect(() => host.write(12, id, "secret input")).toThrow("audit unavailable for write");
    expect(process.write).not.toHaveBeenCalled();

    blockedOperation = "resize";
    expect(() => host.resize(12, id, 90, 30)).toThrow("audit unavailable for resize");
    expect(process.resize).not.toHaveBeenCalled();

    blockedOperation = "close";
    blockedCloseReason = "user_kill";
    expect(() => host.kill(12, id)).toThrow("audit unavailable for close:user_kill");
    expect(process.kill).not.toHaveBeenCalled();

    blockedCloseReason = "owner_cleanup";
    expect(() => host.cleanupOwner(12)).toThrow("audit unavailable for close:owner_cleanup");
    expect(process.kill).not.toHaveBeenCalled();

    blockedCloseReason = "app_dispose";
    expect(() => host.dispose()).toThrow("audit unavailable for close:app_dispose");
    expect(process.kill).not.toHaveBeenCalled();
  });

  it("records the close reason when a close is rejected or the PTY kill fails", () => {
    const missingEvents: TerminalAuditEvent[] = [];
    const missingHost = new TerminalHost(
      fakeFactory(new FakePty()),
      "darwin",
      { SHELL: "/bin/zsh" },
      collectAudit(missingEvents),
    );

    expect(missingHost.kill(42, "missing-terminal")).toBe(false);
    expect(missingEvents).toEqual([
      {
        operation: "close",
        phase: "attempt",
        terminalId: "missing-terminal",
        ownerId: 42,
        closeReason: "user_kill",
      },
      {
        operation: "close",
        phase: "outcome",
        outcome: "rejected",
        reason: "not_owned_or_missing",
        terminalId: "missing-terminal",
        ownerId: 42,
        closeReason: "user_kill",
      },
    ]);

    const process = new FakePty();
    const failedEvents: TerminalAuditEvent[] = [];
    const host = new TerminalHost(
      fakeFactory(process),
      "darwin",
      { SHELL: "/bin/zsh" },
      collectAudit(failedEvents),
    );
    host.create(fakeOwner(43), { id: "failed-close", cwd: "/workspace" });
    failedEvents.length = 0;
    process.kill.mockImplementation(() => {
      throw new Error("kill failed with secret detail");
    });

    expect(() => host.cleanupOwner(43)).toThrow("kill failed with secret detail");
    expect(failedEvents).toEqual([
      {
        operation: "close",
        phase: "attempt",
        terminalId: "failed-close",
        ownerId: 43,
        closeReason: "owner_cleanup",
      },
      {
        operation: "close",
        phase: "outcome",
        outcome: "failed",
        reason: "operation_failed",
        terminalId: "failed-close",
        ownerId: 43,
        closeReason: "owner_cleanup",
      },
    ]);
    expect(JSON.stringify(failedEvents)).not.toContain("secret detail");
  });

  it("throws explicitly when an outcome audit fails after the side effect", () => {
    const process = new FakePty();
    let failWriteOutcome = false;
    const audit: TerminalAuditCallback = (event) => {
      if (failWriteOutcome && event.operation === "write" && event.phase === "outcome") {
        throw new Error("write outcome audit failed");
      }
    };
    const host = new TerminalHost(fakeFactory(process), "darwin", { SHELL: "/bin/zsh" }, audit);
    const { id } = host.create(fakeOwner(23), { id: "terminal-23", cwd: "/workspace" });

    failWriteOutcome = true;
    expect(() => host.write(23, id, "pwd\r")).toThrow("write outcome audit failed");
    expect(process.write).toHaveBeenCalledWith("pwd\r");
  });

  it("releases a created PTY when its success outcome audit cannot commit", () => {
    const failed = new FakePty();
    const healthy = new FakePty();
    const factory = {
      spawn: vi.fn().mockReturnValueOnce(failed).mockReturnValueOnce(healthy),
    } satisfies TerminalProcessFactory;
    let failCreateOutcome = true;
    const host = new TerminalHost(factory, "darwin", { SHELL: "/bin/zsh" }, (event) => {
      if (failCreateOutcome && event.operation === "create" && event.phase === "outcome") {
        throw new Error("create outcome audit failed");
      }
    });

    expect(() => host.create(fakeOwner(24), { id: "terminal-24", cwd: "/workspace" })).toThrow(
      "create outcome audit failed",
    );
    expect(failed.kill).toHaveBeenCalledOnce();

    failCreateOutcome = false;
    expect(host.create(fakeOwner(24), { id: "terminal-24", cwd: "/workspace" })).toMatchObject({
      id: "terminal-24",
      pid: 42,
    });
  });

  it("releases every resource when an explicit close step fails", () => {
    const process = new FakePty();
    const dataError = new Error("data subscription cleanup failed");
    const dataSubscription = {
      dispose: vi.fn(() => {
        throw dataError;
      }),
    };
    const exitSubscription = { dispose: vi.fn() };
    vi.spyOn(process, "onData").mockReturnValue(dataSubscription);
    vi.spyOn(process, "onExit").mockReturnValue(exitSubscription);
    const events: TerminalAuditEvent[] = [];
    const host = new TerminalHost(
      fakeFactory(process),
      "darwin",
      { SHELL: "/bin/zsh" },
      collectAudit(events),
    );
    const owner = fakeOwner(25);
    host.create(owner, { id: "terminal-25", cwd: "/workspace" });
    events.length = 0;

    expect(() => host.kill(owner.id, "terminal-25")).toThrow(dataError);

    expect(dataSubscription.dispose).toHaveBeenCalledOnce();
    expect(exitSubscription.dispose).toHaveBeenCalledOnce();
    expect(process.kill).toHaveBeenCalledOnce();
    expect(events).toEqual([
      expect.objectContaining({ operation: "close", phase: "attempt" }),
      expect.objectContaining({
        operation: "close",
        phase: "outcome",
        outcome: "failed",
        reason: "operation_failed",
      }),
    ]);
    expect(host.kill(owner.id, "terminal-25")).toBe(false);
  });

  it("releases a naturally exited Terminal even when its audit outcome fails", () => {
    const exited = new FakePty();
    const replacement = new FakePty();
    const dataSubscription = { dispose: vi.fn() };
    const exitSubscription = { dispose: vi.fn() };
    let exitListener: ((event: { exitCode: number; signal?: number }) => void) | undefined;
    vi.spyOn(exited, "onData").mockReturnValue(dataSubscription);
    vi.spyOn(exited, "onExit").mockImplementation((listener) => {
      exitListener = listener;
      return exitSubscription;
    });
    const factory = {
      spawn: vi.fn().mockReturnValueOnce(exited).mockReturnValueOnce(replacement),
    } satisfies TerminalProcessFactory;
    const exitAuditError = new Error("exit audit unavailable");
    let failExitAudit = true;
    const host = new TerminalHost(factory, "darwin", { SHELL: "/bin/zsh" }, (event) => {
      if (failExitAudit && event.operation === "exit") throw exitAuditError;
    });
    const owner = fakeOwner(26);
    host.create(owner, { id: "terminal-26", cwd: "/workspace" });

    expect(() => exitListener?.({ exitCode: 1, signal: 15 })).toThrow(exitAuditError);

    expect(dataSubscription.dispose).toHaveBeenCalledOnce();
    expect(exitSubscription.dispose).toHaveBeenCalledOnce();
    expect(owner.send).toHaveBeenCalledWith("terminal:exit", {
      id: "terminal-26",
      exitCode: 1,
      signal: 15,
    });
    failExitAudit = false;
    expect(host.create(owner, { id: "terminal-26", cwd: "/workspace" })).toMatchObject({
      id: "terminal-26",
      pid: 42,
    });
  });

  it("records a sanitized failed outcome when a PTY side effect throws", () => {
    const process = new FakePty();
    const events: TerminalAuditEvent[] = [];
    const host = new TerminalHost(
      fakeFactory(process),
      "darwin",
      { SHELL: "/bin/zsh" },
      collectAudit(events),
    );
    const { id } = host.create(fakeOwner(31), { id: "terminal-31", cwd: "/workspace" });
    events.length = 0;
    process.write.mockImplementation(() => {
      throw new Error("process failure containing secret input");
    });

    expect(() => host.write(31, id, "secret input")).toThrow(
      "process failure containing secret input",
    );
    expect(events).toEqual([
      {
        operation: "write",
        phase: "attempt",
        terminalId: "terminal-31",
        ownerId: 31,
        byteCount: 12,
      },
      {
        operation: "write",
        phase: "outcome",
        outcome: "failed",
        reason: "operation_failed",
        terminalId: "terminal-31",
        ownerId: 31,
        byteCount: 12,
      },
    ]);
    expect(JSON.stringify(events)).not.toContain("secret input");
  });
});

function collectAudit(events: TerminalAuditEvent[]): TerminalAuditCallback {
  return (event) => events.push({ ...event });
}

function fakeFactory(process: IPty) {
  return { spawn: vi.fn(() => process) } satisfies TerminalProcessFactory;
}

function fakeOwner(id: number) {
  return {
    id,
    send: vi.fn<(channel: string, value: unknown) => void>(),
    isDestroyed: () => false,
  } satisfies TerminalOwner;
}

class FakePty implements IPty {
  readonly pid = 42;
  readonly cols = 80;
  readonly rows = 24;
  readonly process = "shell";
  handleFlowControl = false;
  readonly write = vi.fn();
  readonly resize = vi.fn();
  readonly clear = vi.fn();
  readonly kill = vi.fn();
  readonly pause = vi.fn();
  readonly resume = vi.fn();
  #dataListeners = new Set<(data: string) => void>();
  #exitListeners = new Set<(event: { exitCode: number; signal?: number }) => void>();

  readonly onData = (listener: (data: string) => void): IDisposable => {
    this.#dataListeners.add(listener);
    return { dispose: () => this.#dataListeners.delete(listener) };
  };

  readonly onExit = (
    listener: (event: { exitCode: number; signal?: number }) => void,
  ): IDisposable => {
    this.#exitListeners.add(listener);
    return { dispose: () => this.#exitListeners.delete(listener) };
  };

  emitData(data: string): void {
    for (const listener of this.#dataListeners) listener(data);
  }

  emitExit(event: { exitCode: number; signal?: number }): void {
    for (const listener of this.#exitListeners) listener(event);
  }
}
