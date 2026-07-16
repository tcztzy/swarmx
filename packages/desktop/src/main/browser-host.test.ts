import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  BrowserHost,
  type BrowserNavigationHistory,
  type BrowserOwner,
  type BrowserPermissionSession,
  type BrowserView,
  type BrowserViewFactory,
  type BrowserWebContents,
  type BrowserWindowHost,
  MAX_BROWSER_COORDINATE,
  MAX_BROWSER_DIMENSION,
  normalizeBrowserUrl,
} from "./browser-host.js";

vi.mock("electron", () => ({
  BrowserWindow: { fromWebContents: vi.fn() },
  WebContentsView: vi.fn(),
}));

describe("normalizeBrowserUrl", () => {
  it("normalizes web addresses and turns plain text into a search", () => {
    expect(normalizeBrowserUrl("example.com/docs")).toBe("https://example.com/docs");
    expect(normalizeBrowserUrl("localhost:4173")).toBe("http://localhost:4173/");
    expect(normalizeBrowserUrl("https://example.com/a?q=1")).toBe("https://example.com/a?q=1");
    expect(normalizeBrowserUrl("review the release notes")).toBe(
      "https://www.google.com/search?q=review%20the%20release%20notes",
    );
  });

  it("rejects empty and privileged URL schemes", () => {
    expect(() => normalizeBrowserUrl("  ")).toThrow("required");
    expect(() => normalizeBrowserUrl("file:///Users/test/.ssh/id_ed25519")).toThrow(
      "Unsupported browser URL protocol: file:",
    );
    expect(() => normalizeBrowserUrl("javascript:alert(1)")).toThrow(
      "Unsupported browser URL protocol: javascript:",
    );
  });
});

describe("BrowserHost", () => {
  beforeEach(() => vi.clearAllMocks());

  it("creates an isolated sandboxed view, denies permissions and popups, and clamps bounds", () => {
    const harness = createHarness();
    const owner = fakeOwner(7);

    const state = harness.host.create(owner, {
      url: "example.com",
      visible: false,
      bounds: {
        x: -20.9,
        y: MAX_BROWSER_COORDINATE + 20,
        width: MAX_BROWSER_DIMENSION + 500,
        height: 0,
      },
    });

    expect(state).toMatchObject({
      id: "browser-1",
      url: "https://example.com/",
      loading: true,
    });
    expect(harness.factory.create).toHaveBeenCalledWith({
      webPreferences: expect.objectContaining({
        sandbox: true,
        contextIsolation: true,
        nodeIntegration: false,
        webSecurity: true,
        allowRunningInsecureContent: false,
        partition: "persist:browser-test",
        webviewTag: false,
      }),
    });
    expect(harness.window.contentView.addChildView).toHaveBeenCalledWith(harness.view);
    expect(harness.view.setVisible).toHaveBeenCalledWith(false);
    expect(harness.view.setBounds).toHaveBeenCalledWith({
      x: 0,
      y: MAX_BROWSER_COORDINATE,
      width: MAX_BROWSER_DIMENSION,
      height: 1,
    });
    expect(harness.contents.windowOpenHandler?.()).toEqual({ action: "deny" });
    expect(harness.session.permissionCheckHandler?.()).toBe(false);
    const permissionResult = vi.fn();
    harness.session.permissionRequestHandler?.({}, "geolocation", permissionResult);
    expect(permissionResult).toHaveBeenCalledWith(false);
    expect(owner.send).toHaveBeenCalledWith(
      "browser:state",
      expect.objectContaining({ id: "browser-1", url: "https://example.com/" }),
    );
  });

  it("publishes navigation state and prevents page navigation outside HTTP(S)", async () => {
    const harness = createHarness();
    const owner = fakeOwner(4);
    const created = harness.host.create(owner);

    const pending = harness.host.navigate(owner.id, created.id, "https://docs.example.com");
    harness.contents.title = "Example docs";
    harness.contents.url = "https://docs.example.com/start";
    harness.contents.navigationHistory.backAvailable = true;
    harness.contents.emit("page-title-updated", {}, "Example docs", true);
    harness.contents.emit("did-navigate", {}, "https://docs.example.com/start", 200, "OK");
    harness.contents.emit("did-stop-loading");
    await pending;

    expect(owner.send).toHaveBeenLastCalledWith("browser:state", {
      id: created.id,
      url: "https://docs.example.com/start",
      title: "Example docs",
      loading: false,
      canGoBack: true,
      canGoForward: false,
    });

    const preventDefault = vi.fn();
    harness.contents.emit("will-frame-navigate", {
      url: "file:///etc/passwd",
      isMainFrame: true,
      preventDefault,
    });
    expect(preventDefault).toHaveBeenCalledOnce();
    expect(owner.send).toHaveBeenLastCalledWith(
      "browser:state",
      expect.objectContaining({ error: expect.stringContaining("only HTTP and HTTPS") }),
    );

    await expect(harness.host.navigate(owner.id, created.id, "data:text/html,bad")).rejects.toThrow(
      "Unsupported browser URL protocol: data:",
    );
  });

  it("scopes controls to the owning renderer and reports load failures", async () => {
    const harness = createHarness();
    const owner = fakeOwner(2);
    const { id } = harness.host.create(owner);
    harness.contents.url = "https://example.com/";
    harness.contents.navigationHistory.backAvailable = true;
    harness.contents.navigationHistory.forwardAvailable = true;

    expect(harness.host.back(99, id)).toBe(false);
    expect(harness.host.forward(99, id)).toBe(false);
    expect(harness.host.reload(99, id)).toBe(false);
    expect(harness.host.setVisible(99, id, false)).toBe(false);
    expect(harness.host.setBounds(99, id, { x: 0, y: 0, width: 10, height: 10 })).toBe(false);

    harness.contents.emit("did-navigate", {}, harness.contents.url, 200, "OK");
    expect(harness.host.back(owner.id, id)).toBe(true);
    expect(harness.host.forward(owner.id, id)).toBe(true);
    expect(harness.host.reload(owner.id, id)).toBe(true);
    expect(harness.contents.navigationHistory.goBack).toHaveBeenCalledOnce();
    expect(harness.contents.navigationHistory.goForward).toHaveBeenCalledOnce();
    expect(harness.contents.reload).toHaveBeenCalledOnce();

    harness.contents.loadError = new Error("DNS lookup failed");
    const state = await harness.host.navigate(owner.id, id, "https://missing.example");
    expect(state?.error).toBe("DNS lookup failed");
    expect(state?.loading).toBe(false);
  });

  it("detaches and closes only views owned by the renderer being cleaned up", () => {
    const first = createHarness(["first", "second"]);
    const firstOwner = fakeOwner(1);
    const otherOwner = fakeOwner(2);
    const firstState = first.host.create(firstOwner);
    const secondState = first.host.create(otherOwner);
    const secondView = first.views[1];

    expect(first.host.destroy(otherOwner.id, firstState.id)).toBe(false);
    first.host.cleanupOwner(firstOwner.id);

    expect(first.window.contentView.removeChildView).toHaveBeenCalledWith(first.view);
    expect(first.contents.close).toHaveBeenCalledWith({ waitForBeforeUnload: false });
    expect(secondView.webContents.close).not.toHaveBeenCalled();
    expect(first.host.getState(firstOwner.id, firstState.id)).toBeNull();
    expect(first.host.getState(otherOwner.id, secondState.id)).not.toBeNull();

    expect(first.host.destroy(otherOwner.id, secondState.id)).toBe(true);
    expect(secondView.webContents.close).toHaveBeenCalledWith({ waitForBeforeUnload: false });
  });
});

function createHarness(ids: string[] = ["browser-1"]) {
  const views = ids.map(() => new FakeView());
  const window = new FakeWindow();
  const factory = {
    create: vi.fn().mockImplementation(() => {
      const view = views[factory.create.mock.calls.length - 1];
      if (!view) throw new Error("No fake browser view is available.");
      return view;
    }),
    windowForOwner: vi.fn(() => window),
  } satisfies BrowserViewFactory;
  let nextId = 0;
  const host = new BrowserHost(factory, "persist:browser-test", () => ids[nextId++] ?? "extra");
  return {
    host,
    factory,
    window,
    views,
    view: views[0],
    contents: views[0].webContents,
    session: views[0].webContents.session,
  };
}

function fakeOwner(id: number): BrowserOwner & { send: ReturnType<typeof vi.fn> } {
  return { id, send: vi.fn(), isDestroyed: () => false };
}

class FakeWindow implements BrowserWindowHost {
  readonly contentView = {
    addChildView: vi.fn<(view: BrowserView) => void>(),
    removeChildView: vi.fn<(view: BrowserView) => void>(),
  };
  readonly isDestroyed = vi.fn(() => false);
}

class FakeView implements BrowserView {
  readonly webContents = new FakeWebContents();
  readonly setBounds = vi.fn();
  readonly setVisible = vi.fn();
}

class FakePermissionSession implements BrowserPermissionSession {
  permissionCheckHandler: ((...args: unknown[]) => boolean) | null = null;
  permissionRequestHandler:
    | ((
        webContents: unknown,
        permission: string,
        callback: (granted: boolean) => void,
        details?: unknown,
      ) => void)
    | null = null;

  setPermissionCheckHandler(handler: ((...args: unknown[]) => boolean) | null): void {
    this.permissionCheckHandler = handler;
  }

  setPermissionRequestHandler(
    handler:
      | ((
          webContents: unknown,
          permission: string,
          callback: (granted: boolean) => void,
          details?: unknown,
        ) => void)
      | null,
  ): void {
    this.permissionRequestHandler = handler;
  }
}

class FakeHistory implements BrowserNavigationHistory {
  backAvailable = false;
  forwardAvailable = false;
  readonly canGoBack = vi.fn(() => this.backAvailable);
  readonly canGoForward = vi.fn(() => this.forwardAvailable);
  readonly goBack = vi.fn();
  readonly goForward = vi.fn();
}

class FakeWebContents implements BrowserWebContents {
  readonly navigationHistory = new FakeHistory();
  readonly session = new FakePermissionSession();
  title = "";
  url = "";
  destroyed = false;
  loadError: Error | null = null;
  windowOpenHandler: ((...args: unknown[]) => { action: "allow" | "deny" }) | null = null;
  readonly listeners = new Map<string, Set<(...args: unknown[]) => void>>();
  readonly reload = vi.fn();
  readonly close = vi.fn((_options?: { waitForBeforeUnload?: boolean }) => {
    this.destroyed = true;
    this.emit("destroyed");
  });
  readonly loadURL = vi.fn(async (url: string) => {
    this.url = url;
    if (this.loadError) throw this.loadError;
  });

  getTitle(): string {
    return this.title;
  }

  getURL(): string {
    return this.url;
  }

  isDestroyed(): boolean {
    return this.destroyed;
  }

  on(event: string, listener: (...args: unknown[]) => void): this {
    const listeners = this.listeners.get(event) ?? new Set();
    listeners.add(listener);
    this.listeners.set(event, listeners);
    return this;
  }

  setWindowOpenHandler(handler: (...args: unknown[]) => { action: "allow" | "deny" }): void {
    this.windowOpenHandler = handler;
  }

  emit(event: string, ...args: unknown[]): void {
    for (const listener of this.listeners.get(event) ?? []) listener(...args);
  }
}
