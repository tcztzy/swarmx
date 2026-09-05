import { BrowserWindow, shell, type WebContents } from "electron";

const WIDTH = 1280;
const HEIGHT = 860;

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
      `swarmx: failed to load the SwarmX surface: ${error instanceof Error ? error.message : String(error)}\n`,
    );
    if (!window.isDestroyed()) window.show();
  });
  return window;
}

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

function webUrl(target: string): URL | undefined {
  try {
    const url = new URL(target);
    return url.protocol === "http:" || url.protocol === "https:" ? url : undefined;
  } catch {
    return undefined;
  }
}

function openExternal(target: string): void {
  void shell.openExternal(target).catch((error: unknown) => {
    process.stderr.write(
      `swarmx: failed to open external link: ${error instanceof Error ? error.message : String(error)}\n`,
    );
  });
}

function isRendererClipboardWrite(
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

function setRendererPermissionPolicy(window: BrowserWindow, origin: string): void {
  const { session } = window.webContents;
  session.setPermissionCheckHandler((webContents, permission, requestingOrigin, details) =>
    isRendererClipboardWrite(
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
      isRendererClipboardWrite(
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
