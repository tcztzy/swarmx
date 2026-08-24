/**
 * The application window. The renderer loads the harness's own web surface over
 * loopback HTTP, so it is ordinary remote content: no preload, no Node
 * integration, context isolation on. Every privileged capability it needs is
 * already reachable through the harness's `/api` transport, which its client
 * plugins speak natively.
 */
import { BrowserWindow, shell, type WebContents } from "electron";

/** Initial window size; the harness UI is a desktop-width layout. */
const WIDTH = 1280;
const HEIGHT = 860;

/**
 * Create the main window and load the harness surface.
 *
 * Navigation is fenced to the harness origin: in-page navigation elsewhere is
 * cancelled and new-window requests are handed to the OS browser, so a link in
 * a model response cannot repoint the application at another origin.
 */
export function createWindow(url: string): BrowserWindow {
  const window = new BrowserWindow({
    width: WIDTH,
    height: HEIGHT,
    show: false,
    title: "SwarmX",
    backgroundColor: "#000000",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  const origin = new URL(url).origin;
  fenceNavigation(window, origin);
  setRendererPermissionPolicy(window, origin);
  window.once("ready-to-show", () => window.show());
  void window.loadURL(url).catch((error: unknown) => {
    process.stderr.write(
      `swarmx: failed to load the Harness surface: ${error instanceof Error ? error.message : String(error)}\n`,
    );
    if (!window.isDestroyed()) window.show();
  });
  return window;
}

/** Keep the window on `origin`; send anything else to the OS browser. */
function fenceNavigation(window: BrowserWindow, origin: string): void {
  const { webContents } = window;
  webContents.on("will-navigate", (event, target) => {
    const external = webUrl(target);
    if (external?.origin === origin) return;
    event.preventDefault();
    if (external !== undefined) openExternal(target);
  });
  webContents.setWindowOpenHandler(({ url: target }) => {
    const external = webUrl(target);
    if (external !== undefined) openExternal(target);
    return { action: "deny" };
  });
}

/** Parse the only URL schemes model-rendered content may hand to the OS. */
function webUrl(target: string): URL | undefined {
  try {
    const url = new URL(target);
    return url.protocol === "http:" || url.protocol === "https:" ? url : undefined;
  } catch {
    return undefined;
  }
}

/** Hand a validated Web URL to the OS without creating an unhandled rejection. */
function openExternal(target: string): void {
  void shell.openExternal(target).catch((error: unknown) => {
    process.stderr.write(
      `swarmx: failed to open external link: ${error instanceof Error ? error.message : String(error)}\n`,
    );
  });
}

/** Match one permission requester to the app-owned window, origin, and main frame. */
function isHarnessClipboardWrite(
  window: BrowserWindow,
  origin: string,
  webContents: WebContents | null,
  permission: string,
  requestingUrl: string,
  isMainFrame: boolean,
): boolean {
  if (
    webContents !== window.webContents ||
    permission !== "clipboard-sanitized-write" ||
    !isMainFrame
  ) {
    return false;
  }
  try {
    return new URL(requestingUrl).origin === origin;
  } catch {
    return false;
  }
}

/** Allow same-origin text writes required by Copy; deny every other renderer permission. */
function setRendererPermissionPolicy(window: BrowserWindow, origin: string): void {
  const { session } = window.webContents;
  session.setPermissionCheckHandler((webContents, permission, requestingOrigin, details) =>
    isHarnessClipboardWrite(
      window,
      origin,
      webContents,
      permission,
      requestingOrigin,
      details.isMainFrame,
    ),
  );
  session.setPermissionRequestHandler((webContents, permission, callback, details) =>
    callback(
      isHarnessClipboardWrite(
        window,
        origin,
        webContents,
        permission,
        details.requestingUrl,
        details.isMainFrame,
      ),
    ),
  );
}
