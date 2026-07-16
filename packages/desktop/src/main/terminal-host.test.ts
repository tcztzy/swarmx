import type { IDisposable, IPty } from "node-pty";
import { describe, expect, it, vi } from "vitest";
import { TerminalHost, type TerminalOwner, type TerminalProcessFactory } from "./terminal-host.js";

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
});

function fakeFactory(process: IPty) {
  return { spawn: vi.fn(() => process) } satisfies TerminalProcessFactory;
}

function fakeOwner(id: number): TerminalOwner & { send: ReturnType<typeof vi.fn> } {
  return { id, send: vi.fn(), isDestroyed: () => false };
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
