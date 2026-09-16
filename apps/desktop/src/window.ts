import { fileURLToPath } from "node:url";
import { BrowserWindow, shell, type WebContents } from "electron";

const WIDTH = 1280;
const HEIGHT = 860;

export function createWindow(renderer: string): BrowserWindow {
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
      preload: fileURLToPath(new URL("../preload.cjs", import.meta.url)),
    },
  });
  const location = new URL(renderer);
  const local = (target: URL) =>
    location.protocol === "file:" ? target.protocol === "file:" : target.origin === location.origin;
  fenceNavigation(window, local);
  setRendererPermissionPolicy(window, local);
  window.once("ready-to-show", () => window.show());
  void window.loadURL(renderer).catch((error: unknown) => {
    process.stderr.write(
      `swarmx: failed to load the SwarmX surface: ${error instanceof Error ? error.message : String(error)}\n`,
    );
    if (!window.isDestroyed()) window.show();
  });
  return window;
}

function fenceNavigation(window: BrowserWindow, local: (target: URL) => boolean): void {
  const { webContents } = window;
  webContents.on("will-navigate", (event, target) => {
    const destination = new URL(target);
    if (local(destination)) return;
    event.preventDefault();
    if (destination.protocol === "http:" || destination.protocol === "https:") openExternal(target);
  });
  webContents.setWindowOpenHandler(({ url: target }) => {
    const destination = webUrl(target);
    if (destination !== undefined) openExternal(target);
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
  local: (target: URL) => boolean,
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
    return local(new URL(requestingUrl));
  } catch {
    return false;
  }
}

function setRendererPermissionPolicy(window: BrowserWindow, local: (target: URL) => boolean): void {
  const { session } = window.webContents;
  session.setPermissionCheckHandler((webContents, permission, requestingOrigin, details) =>
    isRendererClipboardWrite(
      window,
      local,
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
        local,
        webContents,
        permission,
        details.requestingUrl,
        details.isMainFrame,
      ),
    ),
  );
}
