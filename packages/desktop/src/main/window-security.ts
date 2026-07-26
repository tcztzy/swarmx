import type { IpcMainInvokeEvent, WebContents, WebPreferences } from "electron";

export interface RendererIpcEvent {
  readonly sender: {
    readonly mainFrame: IpcMainInvokeEvent["senderFrame"];
  };
  readonly senderFrame: IpcMainInvokeEvent["senderFrame"];
}

export type OpenExternalUrl = (url: string) => Promise<unknown>;

export function secureMainWindowWebPreferences(preload: string): WebPreferences {
  return {
    preload,
    contextIsolation: true,
    nodeIntegration: false,
    sandbox: true,
    webSecurity: true,
    webviewTag: false,
  };
}

export function isTrustedRendererUrl(candidate: string, rendererUrl: string): boolean {
  try {
    const actual = new URL(candidate);
    const expected = new URL(rendererUrl);
    return (
      actual.protocol === expected.protocol &&
      actual.origin === expected.origin &&
      actual.pathname === expected.pathname &&
      actual.username === expected.username &&
      actual.password === expected.password
    );
  } catch {
    return false;
  }
}

export function isTrustedRendererIpcEvent(event: RendererIpcEvent, rendererUrl: string): boolean {
  const frame = event.senderFrame;
  return Boolean(
    frame && frame === event.sender.mainFrame && isTrustedRendererUrl(frame.url, rendererUrl),
  );
}

export function isSafeExternalUrl(candidate: string): boolean {
  try {
    const url = new URL(candidate);
    return (
      (url.protocol === "http:" || url.protocol === "https:") &&
      url.username === "" &&
      url.password === ""
    );
  } catch {
    return false;
  }
}

export function installMainWindowNavigationGuards(
  webContents: WebContents,
  rendererUrl: string,
  openExternal: OpenExternalUrl,
): void {
  const rejectNavigation = (event: { preventDefault(): void }, targetUrl: string) => {
    if (isTrustedRendererUrl(targetUrl, rendererUrl)) return;
    event.preventDefault();
    if (isSafeExternalUrl(targetUrl)) void openExternal(targetUrl).catch(() => undefined);
  };

  webContents.on("will-navigate", rejectNavigation);
  webContents.on("will-redirect", rejectNavigation);
  webContents.on("will-attach-webview", (event) => event.preventDefault());
  webContents.setWindowOpenHandler(({ url }) => {
    if (!isTrustedRendererUrl(url, rendererUrl) && isSafeExternalUrl(url)) {
      void openExternal(url).catch(() => undefined);
    }
    return { action: "deny" };
  });
}
