/** @vitest-environment jsdom */

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { DesktopTerminalApi } from "../../shared/desktop-api.js";
import { useTerminalController } from "./terminal-controller.js";

const xterm = vi.hoisted(() => {
  const instances: MockTerminal[] = [];
  class MockTerminal {
    cols = 80;
    rows = 24;
    options: { theme?: unknown } = {};
    inputListener: ((data: string) => void) | undefined;
    readonly loadAddon = vi.fn();
    readonly open = vi.fn();
    readonly write = vi.fn();
    readonly writeln = vi.fn();
    readonly focus = vi.fn();
    readonly reset = vi.fn();
    readonly dispose = vi.fn();
    readonly inputDispose = vi.fn();

    constructor() {
      instances.push(this);
    }

    onData(listener: (data: string) => void) {
      this.inputListener = listener;
      return { dispose: this.inputDispose };
    }

    emitInput(data: string): void {
      this.inputListener?.(data);
    }
  }
  return { instances, MockTerminal };
});

const fit = vi.hoisted(() => {
  const instances: MockFitAddon[] = [];
  class MockFitAddon {
    readonly fit = vi.fn();

    constructor() {
      instances.push(this);
    }
  }
  return { instances, MockFitAddon };
});

vi.mock("@xterm/xterm", () => ({ Terminal: xterm.MockTerminal }));
vi.mock("@xterm/addon-fit", () => ({ FitAddon: fit.MockFitAddon }));

interface MockApi extends DesktopTerminalApi {
  createTerminal: ReturnType<typeof vi.fn>;
  writeTerminal: ReturnType<typeof vi.fn>;
  resizeTerminal: ReturnType<typeof vi.fn>;
  killTerminal: ReturnType<typeof vi.fn>;
  onTerminalData: ReturnType<typeof vi.fn>;
  onTerminalExit: ReturnType<typeof vi.fn>;
  emitData(event: { id: string; data: string }): void;
  emitExit(event: { id: string; exitCode: number; signal?: number }): void;
  removeData: ReturnType<typeof vi.fn>;
  removeExit: ReturnType<typeof vi.fn>;
}

const resizeObservers: Array<{
  callback: ResizeObserverCallback;
  disconnect: ReturnType<typeof vi.fn>;
}> = [];
const getTheme = () => ({ background: "#000000" });
let themeListener: (() => void) | undefined;
const removeThemeListener = vi.fn();

beforeEach(() => {
  xterm.instances.length = 0;
  fit.instances.length = 0;
  resizeObservers.length = 0;
  themeListener = undefined;
  removeThemeListener.mockClear();
  vi.stubGlobal(
    "ResizeObserver",
    class {
      readonly disconnect = vi.fn();

      constructor(readonly callback: ResizeObserverCallback) {
        resizeObservers.push(this);
      }

      observe(): void {}
    },
  );
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    callback(0);
    return 1;
  });
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: vi.fn(() => ({
      matches: false,
      addEventListener: vi.fn((_event, listener) => {
        themeListener = listener;
      }),
      removeEventListener: removeThemeListener,
    })),
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("useTerminalController", () => {
  it("starts only while active and deduplicates visible resizes", async () => {
    const api = createApi();
    const view = render(<Harness api={api} active={false} />);
    expect(api.createTerminal).not.toHaveBeenCalled();

    view.rerender(<Harness api={api} active />);
    await waitFor(() => expect(api.createTerminal).toHaveBeenCalledOnce());
    expect(screen.getByTestId("status").textContent).toBe("running");
    xterm.instances[0]?.emitInput("direct");
    expect(api.writeTerminal).toHaveBeenCalledWith(expect.any(String), "direct");
    act(() => themeListener?.());
    expect(xterm.instances[0]?.options.theme).toEqual(getTheme());

    const viewport = screen.getByTestId("viewport");
    Object.defineProperties(viewport, {
      offsetWidth: { configurable: true, value: 640 },
      offsetHeight: { configurable: true, value: 320 },
    });
    act(() => resizeObservers.at(-1)?.callback([], resizeObservers.at(-1) as never));
    act(() => resizeObservers.at(-1)?.callback([], resizeObservers.at(-1) as never));
    expect(api.resizeTerminal).toHaveBeenCalledOnce();

    view.rerender(<Harness api={api} active={false} />);
    act(() => resizeObservers.at(-1)?.callback([], resizeObservers.at(-1) as never));
    expect(api.resizeTerminal).toHaveBeenCalledOnce();
    expect(api.killTerminal).not.toHaveBeenCalled();
  });

  it("kills a deferred create after unmount without activating or flushing it", async () => {
    const pending = deferred<{ id: string; pid: number }>();
    const api = createApi({ createTerminal: vi.fn(() => pending.promise) });
    const view = render(<Harness api={api} active />);
    await waitFor(() => expect(api.createTerminal).toHaveBeenCalledOnce());
    const id = api.createTerminal.mock.calls[0]?.[0].id as string;
    xterm.instances[0]?.emitInput("queued input");

    view.unmount();
    expect(api.killTerminal).not.toHaveBeenCalled();
    await act(() => pending.resolve({ id, pid: 42 }));

    await waitFor(() => expect(api.killTerminal).toHaveBeenCalledWith(id));
    expect(api.killTerminal).toHaveBeenCalledOnce();
    expect(api.writeTerminal).not.toHaveBeenCalled();
    expect(xterm.instances[0]?.focus).not.toHaveBeenCalled();
    expect(xterm.instances[0]?.inputDispose).toHaveBeenCalledOnce();
    expect(xterm.instances[0]?.dispose).toHaveBeenCalledOnce();
    expect(api.removeData).toHaveBeenCalledOnce();
    expect(api.removeExit).toHaveBeenCalledOnce();
    expect(resizeObservers[0]?.disconnect).toHaveBeenCalledOnce();
    expect(removeThemeListener).toHaveBeenCalledOnce();
  });

  it("does not focus a Terminal that becomes hidden while creation is pending", async () => {
    const pending = deferred<{ id: string; pid: number }>();
    const api = createApi({ createTerminal: vi.fn(() => pending.promise) });
    const view = render(<Harness api={api} active />);
    await waitFor(() => expect(api.createTerminal).toHaveBeenCalledOnce());
    const id = api.createTerminal.mock.calls[0]?.[0].id as string;

    view.rerender(<Harness api={api} active={false} />);
    await act(() => pending.resolve({ id, pid: 42 }));

    await waitFor(() => expect(screen.getByTestId("status").textContent).toBe("running"));
    expect(xterm.instances[0]?.focus).not.toHaveBeenCalled();
    expect(api.killTerminal).not.toHaveBeenCalled();
  });

  it("isolates StrictMode generations and keeps only the current create live", async () => {
    const first = deferred<{ id: string; pid: number }>();
    const second = deferred<{ id: string; pid: number }>();
    const api = createApi({
      createTerminal: vi
        .fn()
        .mockImplementationOnce(() => first.promise)
        .mockImplementationOnce(() => second.promise),
    });
    render(
      <StrictMode>
        <Harness api={api} active />
      </StrictMode>,
    );
    await waitFor(() => expect(api.createTerminal).toHaveBeenCalledTimes(2));
    expect(xterm.instances[0]?.inputDispose).toHaveBeenCalledOnce();
    expect(xterm.instances[0]?.dispose).toHaveBeenCalledOnce();
    expect(resizeObservers[0]?.disconnect).toHaveBeenCalledOnce();
    const firstId = api.createTerminal.mock.calls[0]?.[0].id as string;
    const secondId = api.createTerminal.mock.calls[1]?.[0].id as string;

    await act(() => first.resolve({ id: firstId, pid: 41 }));
    await waitFor(() => expect(api.killTerminal).toHaveBeenCalledWith(firstId));
    await act(() => second.resolve({ id: secondId, pid: 42 }));

    await waitFor(() => expect(screen.getByTestId("status").textContent).toBe("running"));
    expect(api.killTerminal).not.toHaveBeenCalledWith(secondId);
    expect(xterm.instances.at(-1)?.focus).toHaveBeenCalledOnce();
  });

  it("serializes a live restart and does not let visibility changes start another PTY", async () => {
    const pendingKill = deferred<{ killed: boolean }>();
    const api = createApi({ killTerminal: vi.fn(() => pendingKill.promise) });
    const view = render(<Harness api={api} active />);
    await waitFor(() => expect(screen.getByTestId("status").textContent).toBe("running"));

    fireEvent.click(screen.getByRole("button", { name: "restart" }));
    fireEvent.click(screen.getByRole("button", { name: "restart" }));
    view.rerender(<Harness api={api} active={false} />);
    view.rerender(<Harness api={api} active />);

    await waitFor(() => expect(api.killTerminal).toHaveBeenCalledOnce());
    expect(api.createTerminal).toHaveBeenCalledOnce();
    await act(() => pendingKill.resolve({ killed: true }));

    await waitFor(() => expect(api.createTerminal).toHaveBeenCalledTimes(2));
    expect(xterm.instances[0]?.reset).toHaveBeenCalledOnce();
    expect(screen.getByTestId("status").textContent).toBe("running");
  });

  it("keeps the live PTY tracked when restart cleanup fails and permits retry", async () => {
    const restartError = new Error("restart cleanup failed");
    const api = createApi({
      killTerminal: vi
        .fn()
        .mockRejectedValueOnce(restartError)
        .mockResolvedValueOnce({ killed: true }),
    });
    render(<Harness api={api} active />);
    await waitFor(() => expect(screen.getByTestId("status").textContent).toBe("running"));
    const firstId = api.createTerminal.mock.calls[0]?.[0].id as string;

    fireEvent.click(screen.getByRole("button", { name: "restart" }));
    await waitFor(() => expect(screen.getByTestId("status").textContent).toBe("error"));
    expect(xterm.instances[0]?.writeln).toHaveBeenCalledWith(
      "\r\nUnable to restart terminal: restart cleanup failed",
    );
    expect(api.createTerminal).toHaveBeenCalledOnce();
    xterm.instances[0]?.emitInput("still live");
    expect(api.writeTerminal).toHaveBeenCalledWith(firstId, "still live");

    fireEvent.click(screen.getByRole("button", { name: "restart" }));
    await waitFor(() => expect(api.createTerminal).toHaveBeenCalledTimes(2));
    expect(api.killTerminal).toHaveBeenCalledTimes(2);
    expect(screen.getByTestId("status").textContent).toBe("running");
  });

  it("ignores a failed restart cleanup after the controller unmounts", async () => {
    const pendingKill = deferred<{ killed: boolean }>();
    const api = createApi({ killTerminal: vi.fn(() => pendingKill.promise) });
    const view = render(<Harness api={api} active />);
    await waitFor(() => expect(screen.getByTestId("status").textContent).toBe("running"));
    const terminal = xterm.instances[0];

    fireEvent.click(screen.getByRole("button", { name: "restart" }));
    await waitFor(() => expect(api.killTerminal).toHaveBeenCalledOnce());
    view.unmount();
    await act(() => pendingKill.reject(new Error("late cleanup failure")));

    expect(terminal?.writeln).not.toHaveBeenCalledWith(
      expect.stringContaining("Unable to restart terminal"),
    );
    expect(api.createTerminal).toHaveBeenCalledOnce();
  });

  it("keeps a created PTY tracked when its buffered write fails", async () => {
    const pendingCreate = deferred<{ id: string; pid: number }>();
    const api = createApi({
      createTerminal: vi.fn(() => pendingCreate.promise),
      writeTerminal: vi.fn(async () => {
        throw new Error("write failed");
      }),
    });
    const view = render(<Harness api={api} active />);
    await waitFor(() => expect(api.createTerminal).toHaveBeenCalledOnce());
    const id = api.createTerminal.mock.calls[0]?.[0].id as string;
    xterm.instances[0]?.emitInput("queued");

    await act(() => pendingCreate.resolve({ id, pid: 42 }));
    await waitFor(() => expect(screen.getByTestId("status").textContent).toBe("error"));
    expect(xterm.instances[0]?.writeln).toHaveBeenCalledWith(
      "\r\nUnable to write to terminal: write failed",
    );
    expect(api.killTerminal).not.toHaveBeenCalled();

    view.unmount();
    expect(api.killTerminal).toHaveBeenCalledWith(id);
  });

  it("keeps a live PTY tracked when a direct write fails", async () => {
    const api = createApi({
      writeTerminal: vi.fn(async () => {
        throw new Error("direct write failed");
      }),
    });
    const view = render(<Harness api={api} active />);
    await waitFor(() => expect(screen.getByTestId("status").textContent).toBe("running"));
    const id = api.createTerminal.mock.calls[0]?.[0].id as string;

    xterm.instances[0]?.emitInput("direct");

    await waitFor(() => expect(screen.getByTestId("status").textContent).toBe("error"));
    expect(xterm.instances[0]?.writeln).toHaveBeenCalledWith(
      "\r\nUnable to write to terminal: direct write failed",
    );
    view.unmount();
    expect(api.killTerminal).toHaveBeenCalledWith(id);
  });

  it("replaces a live PTY when its working directory changes", async () => {
    const api = createApi();
    const view = render(<Harness key="/workspace/a" api={api} active cwd="/workspace/a" />);
    await waitFor(() => expect(api.createTerminal).toHaveBeenCalledOnce());
    const firstId = api.createTerminal.mock.calls[0]?.[0].id as string;

    view.rerender(<Harness key="/workspace/b" api={api} active cwd="/workspace/b" />);

    await waitFor(() => expect(api.createTerminal).toHaveBeenCalledTimes(2));
    expect(api.killTerminal).toHaveBeenCalledWith(firstId);
    expect(api.createTerminal.mock.calls[1]?.[0]).toMatchObject({ cwd: "/workspace/b" });
  });

  it("buffers input, filters owner events, and exposes an explicit restart", async () => {
    const pending = deferred<{ id: string; pid: number }>();
    const api = createApi({ createTerminal: vi.fn(() => pending.promise) });
    render(<Harness api={api} active />);
    await waitFor(() => expect(api.createTerminal).toHaveBeenCalledOnce());
    const id = api.createTerminal.mock.calls[0]?.[0].id as string;
    const terminal = xterm.instances[0];
    terminal?.emitInput("queued");
    api.emitData({ id: "other", data: "ignored" });
    expect(terminal?.write).not.toHaveBeenCalled();

    await act(() => pending.resolve({ id, pid: 42 }));
    await waitFor(() => expect(api.writeTerminal).toHaveBeenCalledWith(id, "queued"));
    api.emitData({ id, data: "output" });
    expect(terminal?.write).toHaveBeenCalledWith("output");
    act(() => api.emitExit({ id: "other", exitCode: 1 }));
    expect(screen.getByTestId("status").textContent).toBe("running");
    act(() => api.emitExit({ id, exitCode: 0 }));
    expect(screen.getByTestId("status").textContent).toBe("exited");
    expect(terminal?.writeln).toHaveBeenCalledWith("\r\n[Process exited with code 0]");

    fireEvent.click(screen.getByRole("button", { name: "restart" }));
    await waitFor(() => expect(api.createTerminal).toHaveBeenCalledTimes(2));
    expect(terminal?.reset).toHaveBeenCalledOnce();
  });

  it("sanitizes create errors, clears pending input, and permits retry", async () => {
    const api = createApi({
      createTerminal: vi
        .fn()
        .mockRejectedValueOnce(new Error("\u001b[31msecret\u0007 detail"))
        .mockResolvedValueOnce({ id: "retry", pid: 42 }),
    });
    render(<Harness api={api} active />);
    await waitFor(() => expect(screen.getByTestId("status").textContent).toBe("error"));
    expect(xterm.instances[0]?.writeln).toHaveBeenCalledWith(
      "\r\nUnable to start terminal: [31msecret  detail",
    );

    fireEvent.click(screen.getByRole("button", { name: "restart" }));
    await waitFor(() => expect(screen.getByTestId("status").textContent).toBe("running"));
    expect(api.writeTerminal).not.toHaveBeenCalled();
  });
});

function Harness({
  api,
  active,
  cwd = "/workspace",
}: {
  api: DesktopTerminalApi;
  active: boolean;
  cwd?: string;
}) {
  const controller = useTerminalController({ api, active, cwd, getTheme });
  return (
    <>
      <div ref={controller.viewportRef} data-testid="viewport" />
      <output data-testid="status">{controller.status}</output>
      <button type="button" onClick={() => void controller.startNewTerminal()}>
        restart
      </button>
    </>
  );
}

function createApi(overrides: Partial<DesktopTerminalApi> = {}): MockApi {
  let dataListener: Parameters<DesktopTerminalApi["onTerminalData"]>[0] = () => undefined;
  let exitListener: Parameters<DesktopTerminalApi["onTerminalExit"]>[0] = () => undefined;
  const removeData = vi.fn();
  const removeExit = vi.fn();
  const api = {
    createTerminal: vi.fn(async ({ id }) => ({ id, pid: 42 })),
    writeTerminal: vi.fn(async () => ({ written: true })),
    resizeTerminal: vi.fn(async () => ({ resized: true })),
    killTerminal: vi.fn(async () => ({ killed: true })),
    onTerminalData: vi.fn((listener) => {
      dataListener = listener;
      return removeData;
    }),
    onTerminalExit: vi.fn((listener) => {
      exitListener = listener;
      return removeExit;
    }),
    ...overrides,
  } satisfies DesktopTerminalApi;
  return Object.assign(api, {
    emitData: (event: { id: string; data: string }) => dataListener(event),
    emitExit: (event: { id: string; exitCode: number; signal?: number }) => exitListener(event),
    removeData,
    removeExit,
  }) as MockApi;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}
