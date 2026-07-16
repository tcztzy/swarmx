import { randomUUID } from "node:crypto";
import { BrowserWindow, type WebContents, WebContentsView, type WebPreferences } from "electron";

export const DEFAULT_BROWSER_PARTITION = "persist:swarmx-browser";
export const DEFAULT_BROWSER_SEARCH_URL = "https://www.google.com/search?q=";
export const MAX_BROWSER_COORDINATE = 100_000;
export const MAX_BROWSER_DIMENSION = 16_384;

const BLOCKED_NAVIGATION_ERROR = "Navigation blocked: only HTTP and HTTPS URLs are allowed.";

export interface BrowserBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface BrowserState {
  id: string;
  url: string;
  title: string;
  loading: boolean;
  canGoBack: boolean;
  canGoForward: boolean;
  error?: string;
}

export interface BrowserOwner {
  id: number;
  isDestroyed?(): boolean;
  send(channel: string, value: unknown): void;
}

export interface CreateBrowserRequest {
  id?: string;
  url?: string;
  bounds?: BrowserBounds;
  visible?: boolean;
}

export interface BrowserPermissionSession {
  setPermissionCheckHandler(handler: ((...args: unknown[]) => boolean) | null): void;
  setPermissionRequestHandler(
    handler:
      | ((
          webContents: unknown,
          permission: string,
          callback: (granted: boolean) => void,
          details?: unknown,
        ) => void)
      | null,
  ): void;
}

export interface BrowserNavigationHistory {
  canGoBack(): boolean;
  canGoForward(): boolean;
  goBack(): void;
  goForward(): void;
}

export interface BrowserWebContents {
  readonly navigationHistory: BrowserNavigationHistory;
  readonly session: BrowserPermissionSession;
  close(options?: { waitForBeforeUnload?: boolean }): void;
  getTitle(): string;
  getURL(): string;
  isDestroyed(): boolean;
  loadURL(url: string): Promise<unknown>;
  on(event: string, listener: (...args: unknown[]) => void): unknown;
  reload(): void;
  setWindowOpenHandler(handler: (...args: unknown[]) => { action: "allow" | "deny" }): void;
}

export interface BrowserView {
  readonly webContents: BrowserWebContents;
  setBounds(bounds: BrowserBounds): void;
  setVisible(visible: boolean): void;
}

export interface BrowserWindowHost {
  readonly contentView: {
    addChildView(view: BrowserView): void;
    removeChildView(view: BrowserView): void;
  };
  isDestroyed?(): boolean;
}

export interface BrowserViewFactory {
  create(options: { webPreferences: WebPreferences }): BrowserView;
  windowForOwner(owner: BrowserOwner): BrowserWindowHost | null;
}

interface BrowserEntry {
  id: string;
  owner: BrowserOwner;
  window: BrowserWindowHost;
  view: BrowserView;
  state: BrowserState;
  pendingUrl?: string;
  navigationSequence: number;
}

interface BrowserStatePatch {
  url?: string;
  title?: string;
  loading?: boolean;
  error?: string;
}

const electronBrowserViewFactory: BrowserViewFactory = {
  create: (options) => new WebContentsView(options) as unknown as BrowserView,
  windowForOwner: (owner) =>
    BrowserWindow.fromWebContents(owner as unknown as WebContents) as BrowserWindowHost | null,
};

/**
 * Owns sandboxed WebContentsViews and keeps them scoped to the renderer that
 * created them. IPC handlers should always pass event.sender.id to mutating methods.
 */
export class BrowserHost {
  readonly #entries = new Map<string, BrowserEntry>();
  readonly #securedSessions = new WeakSet<object>();

  constructor(
    private readonly factory: BrowserViewFactory = electronBrowserViewFactory,
    private readonly partition = DEFAULT_BROWSER_PARTITION,
    private readonly idFactory: () => string = randomUUID,
  ) {
    if (!partition.trim()) throw new Error("Browser partition is required.");
  }

  create(owner: BrowserOwner, request: CreateBrowserRequest = {}): BrowserState {
    if (owner.isDestroyed?.()) throw new Error("Browser owner is no longer available.");
    const window = this.factory.windowForOwner(owner);
    if (!window || window.isDestroyed?.()) {
      throw new Error("Browser owner window is no longer available.");
    }

    const id = request.id?.trim() || this.idFactory();
    if (this.#entries.has(id)) throw new Error("Browser id is already active.");
    const initialUrl = request.url === undefined ? undefined : normalizeBrowserUrl(request.url);
    const view = this.factory.create({
      webPreferences: {
        allowRunningInsecureContent: false,
        contextIsolation: true,
        devTools: false,
        navigateOnDragDrop: false,
        nodeIntegration: false,
        nodeIntegrationInSubFrames: false,
        nodeIntegrationInWorker: false,
        partition: this.partition,
        safeDialogs: true,
        sandbox: true,
        webSecurity: true,
        webviewTag: false,
      },
    });

    this.#secure(view.webContents);
    try {
      window.contentView.addChildView(view);
      if (request.bounds) view.setBounds(normalizeBrowserBounds(request.bounds));
      view.setVisible(request.visible ?? true);
    } catch (error) {
      if (!view.webContents.isDestroyed()) view.webContents.close({ waitForBeforeUnload: false });
      throw error;
    }

    const entry: BrowserEntry = {
      id,
      owner,
      window,
      view,
      navigationSequence: 0,
      state: {
        id,
        url: safeString(() => view.webContents.getURL()),
        title: safeString(() => view.webContents.getTitle()),
        loading: false,
        canGoBack: safeBoolean(() => view.webContents.navigationHistory.canGoBack()),
        canGoForward: safeBoolean(() => view.webContents.navigationHistory.canGoForward()),
      },
    };
    this.#entries.set(id, entry);
    this.#bindEvents(entry);

    if (initialUrl) void this.#load(entry, initialUrl);
    else this.#emit(entry);
    return copyBrowserState(entry.state);
  }

  getState(ownerId: number, id: string): BrowserState | null {
    const entry = this.#ownedEntry(ownerId, id);
    return entry ? copyBrowserState(entry.state) : null;
  }

  async navigate(ownerId: number, id: string, input: string): Promise<BrowserState | null> {
    const entry = this.#ownedEntry(ownerId, id);
    if (!entry) return null;
    return this.#load(entry, normalizeBrowserUrl(input));
  }

  back(ownerId: number, id: string): boolean {
    const entry = this.#ownedEntry(ownerId, id);
    if (!entry || !safeBoolean(() => entry.view.webContents.navigationHistory.canGoBack())) {
      return false;
    }
    this.#markLoading(entry);
    entry.view.webContents.navigationHistory.goBack();
    return true;
  }

  forward(ownerId: number, id: string): boolean {
    const entry = this.#ownedEntry(ownerId, id);
    if (!entry || !safeBoolean(() => entry.view.webContents.navigationHistory.canGoForward())) {
      return false;
    }
    this.#markLoading(entry);
    entry.view.webContents.navigationHistory.goForward();
    return true;
  }

  reload(ownerId: number, id: string): boolean {
    const entry = this.#ownedEntry(ownerId, id);
    if (!entry || !isAllowedBrowserUrl(entry.state.url)) return false;
    this.#markLoading(entry);
    entry.view.webContents.reload();
    return true;
  }

  setBounds(ownerId: number, id: string, bounds: BrowserBounds): boolean {
    const entry = this.#ownedEntry(ownerId, id);
    if (!entry) return false;
    entry.view.setBounds(normalizeBrowserBounds(bounds));
    return true;
  }

  setVisible(ownerId: number, id: string, visible: boolean): boolean {
    const entry = this.#ownedEntry(ownerId, id);
    if (!entry) return false;
    entry.view.setVisible(visible === true);
    return true;
  }

  destroy(ownerId: number, id: string): boolean {
    const entry = this.#ownedEntry(ownerId, id);
    if (!entry) return false;
    this.#close(entry);
    return true;
  }

  cleanupOwner(ownerId: number): void {
    for (const entry of this.#entries.values()) {
      if (entry.owner.id === ownerId) this.#close(entry);
    }
  }

  dispose(): void {
    for (const entry of this.#entries.values()) this.#close(entry);
  }

  #secure(webContents: BrowserWebContents): void {
    webContents.setWindowOpenHandler(() => ({ action: "deny" }));
    const permissionSession = webContents.session;
    if (!this.#securedSessions.has(permissionSession)) {
      this.#securedSessions.add(permissionSession);
      permissionSession.setPermissionCheckHandler(() => false);
      permissionSession.setPermissionRequestHandler((_webContents, _permission, callback) =>
        callback(false),
      );
    }
  }

  #bindEvents(entry: BrowserEntry): void {
    const { webContents } = entry.view;
    const guardNavigation = (...args: unknown[]): void => {
      const details = asNavigationEvent(args[0]);
      const legacyUrl = typeof args[1] === "string" ? args[1] : undefined;
      const url = details?.url ?? legacyUrl;
      if (!url || isAllowedBrowserUrl(url)) return;
      details?.preventDefault?.();
      if (details?.isMainFrame !== false) {
        ++entry.navigationSequence;
        entry.pendingUrl = undefined;
        this.#update(entry, { loading: false, error: BLOCKED_NAVIGATION_ERROR });
      }
    };

    webContents.on("will-frame-navigate", guardNavigation);
    webContents.on("will-navigate", guardNavigation);
    webContents.on("will-redirect", guardNavigation);
    webContents.on("will-attach-webview", (event) => asPreventableEvent(event)?.preventDefault());
    webContents.on("did-start-loading", () => {
      this.#update(entry, { loading: true, error: undefined });
    });
    webContents.on("did-stop-loading", () => {
      entry.pendingUrl = undefined;
      this.#update(entry, { loading: false });
    });
    webContents.on("did-navigate", (_event, url) => {
      entry.pendingUrl = undefined;
      this.#update(entry, {
        ...(typeof url === "string" ? { url } : {}),
        error: undefined,
      });
    });
    webContents.on("did-navigate-in-page", (_event, url, isMainFrame) => {
      if (isMainFrame === false) return;
      this.#update(entry, typeof url === "string" ? { url } : {});
    });
    webContents.on("page-title-updated", (_event, title) => {
      this.#update(entry, typeof title === "string" ? { title } : {});
    });
    webContents.on(
      "did-fail-load",
      (_event, errorCode, errorDescription, validatedUrl, isMainFrame) => {
        if (isMainFrame === false || errorCode === -3) return;
        entry.pendingUrl = undefined;
        this.#update(entry, {
          ...(typeof validatedUrl === "string" && validatedUrl ? { url: validatedUrl } : {}),
          loading: false,
          error:
            typeof errorDescription === "string" && errorDescription
              ? errorDescription
              : "Page failed to load.",
        });
      },
    );
    webContents.on("render-process-gone", (_event, details) => {
      const reason = asRenderProcessDetails(details)?.reason;
      this.#update(entry, {
        loading: false,
        error: reason ? `Browser renderer exited: ${reason}.` : "Browser renderer exited.",
      });
    });
    webContents.on("destroyed", () => {
      if (this.#entries.get(entry.id) !== entry) return;
      this.#entries.delete(entry.id);
      if (!entry.window.isDestroyed?.()) entry.window.contentView.removeChildView(entry.view);
    });
  }

  async #load(entry: BrowserEntry, url: string): Promise<BrowserState> {
    const sequence = ++entry.navigationSequence;
    entry.pendingUrl = url;
    this.#update(entry, { url, loading: true, error: undefined });
    try {
      await entry.view.webContents.loadURL(url);
      if (this.#entries.get(entry.id) === entry && entry.navigationSequence === sequence) {
        entry.pendingUrl = undefined;
        this.#update(entry, { loading: false, error: undefined });
      }
    } catch (error) {
      if (this.#entries.get(entry.id) === entry && entry.navigationSequence === sequence) {
        entry.pendingUrl = undefined;
        this.#update(entry, { loading: false, error: errorMessage(error) });
      }
    }
    return copyBrowserState(entry.state);
  }

  #markLoading(entry: BrowserEntry): void {
    ++entry.navigationSequence;
    entry.pendingUrl = undefined;
    this.#update(entry, { loading: true, error: undefined });
  }

  #update(entry: BrowserEntry, patch: BrowserStatePatch): void {
    if (this.#entries.get(entry.id) !== entry || entry.view.webContents.isDestroyed()) return;
    const actualUrl = safeString(() => entry.view.webContents.getURL());
    const actualTitle = safeString(() => entry.view.webContents.getTitle());
    const error = Object.hasOwn(patch, "error") ? patch.error : entry.state.error;
    entry.state = {
      id: entry.id,
      url: patch.url ?? entry.pendingUrl ?? (actualUrl || entry.state.url),
      title: patch.title ?? (actualTitle || entry.state.title),
      loading: patch.loading ?? entry.state.loading,
      canGoBack: safeBoolean(() => entry.view.webContents.navigationHistory.canGoBack()),
      canGoForward: safeBoolean(() => entry.view.webContents.navigationHistory.canGoForward()),
      ...(error ? { error } : {}),
    };
    this.#emit(entry);
  }

  #emit(entry: BrowserEntry): void {
    if (entry.owner.isDestroyed?.()) return;
    try {
      entry.owner.send("browser:state", copyBrowserState(entry.state));
    } catch {
      // The renderer can disappear between isDestroyed() and send().
    }
  }

  #ownedEntry(ownerId: number, id: string): BrowserEntry | undefined {
    const entry = this.#entries.get(id);
    if (!entry || entry.owner.id !== ownerId || entry.owner.isDestroyed?.()) return undefined;
    return entry;
  }

  #close(entry: BrowserEntry): void {
    if (this.#entries.get(entry.id) !== entry) return;
    this.#entries.delete(entry.id);
    ++entry.navigationSequence;
    if (!entry.window.isDestroyed?.()) entry.window.contentView.removeChildView(entry.view);
    if (!entry.view.webContents.isDestroyed()) {
      entry.view.webContents.close({ waitForBeforeUnload: false });
    }
  }
}

export function normalizeBrowserBounds(bounds: BrowserBounds): BrowserBounds {
  return {
    x: clampInteger(bounds.x, 0, 0, MAX_BROWSER_COORDINATE),
    y: clampInteger(bounds.y, 0, 0, MAX_BROWSER_COORDINATE),
    width: clampInteger(bounds.width, 1, 1, MAX_BROWSER_DIMENSION),
    height: clampInteger(bounds.height, 1, 1, MAX_BROWSER_DIMENSION),
  };
}

export function normalizeBrowserUrl(input: string): string {
  const value = input.trim();
  if (!value) throw new Error("Browser URL or search query is required.");

  if (/^https?:\/\//i.test(value)) return assertAllowedBrowserUrl(value);
  if (value.startsWith("//")) return assertAllowedBrowserUrl(`https:${value}`);
  if (looksLikeBareHost(value)) {
    const local = /^(?:localhost|127(?:\.\d{1,3}){3}|\[::1\])(?::\d+)?(?:[/?#]|$)/i.test(value);
    return assertAllowedBrowserUrl(`${local ? "http" : "https"}://${value}`);
  }

  const scheme = /^([a-z][a-z\d+.-]*):/i.exec(value)?.[1];
  if (scheme) throw new Error(`Unsupported browser URL protocol: ${scheme.toLowerCase()}:`);
  return `${DEFAULT_BROWSER_SEARCH_URL}${encodeURIComponent(value)}`;
}

export function isAllowedBrowserUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return (url.protocol === "http:" || url.protocol === "https:") && Boolean(url.hostname);
  } catch {
    return false;
  }
}

function assertAllowedBrowserUrl(value: string): string {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error("Browser URL is invalid.");
  }
  if (!isAllowedBrowserUrl(url.href)) throw new Error(BLOCKED_NAVIGATION_ERROR);
  return url.href;
}

function looksLikeBareHost(value: string): boolean {
  if (/\s/.test(value)) return false;
  try {
    const url = new URL(`https://${value}`);
    const hostname = url.hostname;
    return (
      hostname === "localhost" ||
      hostname === "::1" ||
      /^\d{1,3}(?:\.\d{1,3}){3}$/.test(hostname) ||
      hostname.includes(".")
    );
  } catch {
    return false;
  }
}

function clampInteger(value: number, fallback: number, minimum: number, maximum: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(maximum, Math.max(minimum, Math.floor(value)));
}

function safeString(read: () => string): string {
  try {
    return read();
  } catch {
    return "";
  }
}

function safeBoolean(read: () => boolean): boolean {
  try {
    return read();
  } catch {
    return false;
  }
}

function copyBrowserState(state: BrowserState): BrowserState {
  return { ...state };
}

function errorMessage(error: unknown): string {
  return error instanceof Error && error.message
    ? error.message
    : String(error || "Page failed to load.");
}

function asPreventableEvent(value: unknown): { preventDefault(): void } | undefined {
  if (!value || typeof value !== "object") return undefined;
  const event = value as { preventDefault?: unknown };
  return typeof event.preventDefault === "function"
    ? (event as { preventDefault(): void })
    : undefined;
}

function asNavigationEvent(
  value: unknown,
): { url?: string; isMainFrame?: boolean; preventDefault?(): void } | undefined {
  if (!value || typeof value !== "object") return undefined;
  return value as { url?: string; isMainFrame?: boolean; preventDefault?(): void };
}

function asRenderProcessDetails(value: unknown): { reason?: string } | undefined {
  return value && typeof value === "object" ? (value as { reason?: string }) : undefined;
}
