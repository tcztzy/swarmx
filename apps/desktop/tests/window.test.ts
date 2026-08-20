import { beforeEach, describe, expect, it, vi } from "vitest";

const electronMocks = vi.hoisted(() => ({
  loadURL: vi.fn((_url: string) => Promise.resolve()),
  show: vi.fn(),
}));

vi.mock("electron", () => ({
  BrowserWindow: class {
    webContents = {
      handlers: [] as ((event: { preventDefault: () => void }, url: string) => void)[],
      openHandler: undefined as ((details: { url: string }) => unknown) | undefined,
      session: {
        permissionCheckHandler: undefined as ((...args: unknown[]) => boolean) | undefined,
        permissionRequestHandler: undefined as
          | ((
              webContents: unknown,
              permission: string,
              callback: (allowed: boolean) => void,
            ) => void)
          | undefined,
        setPermissionCheckHandler(handler: (...args: unknown[]) => boolean) {
          this.permissionCheckHandler = handler;
        },
        setPermissionRequestHandler(
          handler: (
            webContents: unknown,
            permission: string,
            callback: (allowed: boolean) => void,
          ) => void,
        ) {
          this.permissionRequestHandler = handler;
        },
      },
      on(_event: string, handler: (event: { preventDefault: () => void }, url: string) => void) {
        this.handlers.push(handler);
      },
      setWindowOpenHandler(handler: (details: { url: string }) => unknown) {
        this.openHandler = handler;
      },
    };
    once() {}
    show() {
      electronMocks.show();
    }
    isDestroyed() {
      return false;
    }
    loadURL(url: string) {
      return electronMocks.loadURL(url);
    }
  },
  shell: { openExternal: vi.fn(() => Promise.resolve()) },
}));

const { createWindow } = await import("../src/window.js");
const { shell } = await import("electron");

beforeEach(() => {
  vi.mocked(shell.openExternal).mockClear();
  electronMocks.loadURL.mockReset();
  electronMocks.loadURL.mockResolvedValue(undefined);
  electronMocks.show.mockClear();
});

/** Drive the single `will-navigate` listener the window installs. */
function navigate(window: ReturnType<typeof createWindow>, url: string): boolean {
  let prevented = false;
  const contents = window.webContents as unknown as {
    handlers: ((event: { preventDefault: () => void }, url: string) => void)[];
  };
  for (const handler of contents.handlers) {
    handler({ preventDefault: () => (prevented = true) }, url);
  }
  return prevented;
}

describe("navigation fence", () => {
  it("allows navigation within the harness origin", () => {
    const window = createWindow("http://127.0.0.1:5173");
    expect(navigate(window, "http://127.0.0.1:5173/sessions/abc")).toBe(false);
    expect(shell.openExternal).not.toHaveBeenCalled();
  });

  it("cancels navigation to another origin and opens it externally", () => {
    const window = createWindow("http://127.0.0.1:5173");
    expect(navigate(window, "https://example.com/phish")).toBe(true);
    expect(shell.openExternal).toHaveBeenCalledWith("https://example.com/phish");
  });

  it("treats another loopback port as a foreign origin", () => {
    const window = createWindow("http://127.0.0.1:5173");
    expect(navigate(window, "http://127.0.0.1:9999/")).toBe(true);
  });

  it("blocks local-file navigation without handing it to the os", () => {
    const window = createWindow("http://127.0.0.1:5173");
    expect(navigate(window, "file:///etc/passwd")).toBe(true);
    expect(shell.openExternal).not.toHaveBeenCalled();
  });

  it("blocks malformed navigation without throwing", () => {
    const window = createWindow("http://127.0.0.1:5173");
    expect(() => navigate(window, "not a url")).not.toThrow();
    expect(navigate(window, "not a url")).toBe(true);
    expect(shell.openExternal).not.toHaveBeenCalled();
  });

  it("denies every new-window request and hands the url to the os", () => {
    const window = createWindow("http://127.0.0.1:5173");
    const contents = window.webContents as unknown as {
      openHandler: (details: { url: string }) => { action: string };
    };
    expect(contents.openHandler({ url: "https://docs.example.com" })).toEqual({ action: "deny" });
    expect(shell.openExternal).toHaveBeenCalledWith("https://docs.example.com");
  });

  it("denies non-web new-window protocols without handing them to the os", () => {
    const window = createWindow("http://127.0.0.1:5173");
    const contents = window.webContents as unknown as {
      openHandler: (details: { url: string }) => { action: string };
    };
    expect(contents.openHandler({ url: "javascript:alert(1)" })).toEqual({ action: "deny" });
    expect(contents.openHandler({ url: "file:///etc/passwd" })).toEqual({ action: "deny" });
    expect(shell.openExternal).not.toHaveBeenCalled();
  });

  it("denies renderer permission checks and requests", () => {
    const window = createWindow("http://127.0.0.1:5173");
    const session = (
      window.webContents as unknown as {
        session: {
          permissionCheckHandler: (...args: unknown[]) => boolean;
          permissionRequestHandler: (
            webContents: unknown,
            permission: string,
            callback: (allowed: boolean) => void,
          ) => void;
        };
      }
    ).session;
    const callback = vi.fn();
    expect(session.permissionCheckHandler(undefined, "media")).toBe(false);
    session.permissionRequestHandler(undefined, "media", callback);
    expect(callback).toHaveBeenCalledWith(false);
  });
});

describe("initial load", () => {
  it("reports a failure and shows the window", async () => {
    electronMocks.loadURL.mockRejectedValueOnce(new Error("connection refused"));
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);

    createWindow("http://127.0.0.1:5173");

    await vi.waitFor(() => expect(electronMocks.show).toHaveBeenCalledOnce());
    expect(stderr).toHaveBeenCalledWith(
      expect.stringContaining("failed to load the Harness surface: connection refused"),
    );
    stderr.mockRestore();
  });
});
