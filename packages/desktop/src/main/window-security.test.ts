import { EventEmitter } from "node:events";
import { readFile } from "node:fs/promises";
import type { WebContents } from "electron";
import { describe, expect, it, vi } from "vitest";
import {
  installMainWindowNavigationGuards,
  isSafeExternalUrl,
  isTrustedRendererIpcEvent,
  isTrustedRendererUrl,
  secureMainWindowWebPreferences,
} from "./window-security.js";

const RENDERER_URL = "file:///Applications/SwarmX/renderer/index.html";

describe("Desktop window security", () => {
  it("V549 enables renderer sandboxing and ships a restrictive CSP", async () => {
    expect(secureMainWindowWebPreferences("/app/preload.mjs")).toEqual({
      preload: "/app/preload.mjs",
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
      webviewTag: false,
    });
    const html = await readFile(new URL("../renderer/index.html", import.meta.url), "utf8");
    expect(html).toContain(`default-src 'self'`);
    expect(html).toContain(`script-src 'self'`);
    expect(html).toContain(`object-src 'none'`);
    expect(html).toContain(`base-uri 'none'`);
  });

  it("V547 trusts only the configured renderer entry and safe external HTTP(S) URLs", () => {
    expect(isTrustedRendererUrl(`${RENDERER_URL}#task-1`, RENDERER_URL)).toBe(true);
    expect(
      isTrustedRendererUrl("file:///Applications/SwarmX/renderer/other.html", RENDERER_URL),
    ).toBe(false);
    expect(isTrustedRendererUrl("https://example.com", RENDERER_URL)).toBe(false);
    expect(isSafeExternalUrl("https://example.com/docs")).toBe(true);
    expect(isSafeExternalUrl("javascript:alert(1)")).toBe(false);
    expect(isSafeExternalUrl("https://user:secret@example.com")).toBe(false);
  });

  it("V548 authorizes only the configured main frame", () => {
    const mainFrame = { url: RENDERER_URL };
    expect(
      isTrustedRendererIpcEvent({ sender: { mainFrame }, senderFrame: mainFrame }, RENDERER_URL),
    ).toBe(true);
    expect(
      isTrustedRendererIpcEvent(
        {
          sender: { mainFrame },
          senderFrame: { url: RENDERER_URL },
        },
        RENDERER_URL,
      ),
    ).toBe(false);
    expect(
      isTrustedRendererIpcEvent(
        {
          sender: { mainFrame },
          senderFrame: { url: "https://example.com" },
        },
        RENDERER_URL,
      ),
    ).toBe(false);
  });

  it("V547 blocks navigation, redirects, popups, and webviews before opening safe links externally", async () => {
    const contents = new EventEmitter() as EventEmitter & {
      setWindowOpenHandler: ReturnType<typeof vi.fn>;
    };
    contents.setWindowOpenHandler = vi.fn((handler) => {
      Object.assign(contents, { windowOpenHandler: handler });
    });
    const openExternal = vi.fn(async () => undefined);
    installMainWindowNavigationGuards(
      contents as unknown as WebContents,
      RENDERER_URL,
      openExternal,
    );
    const trustedEvent = { preventDefault: vi.fn() };
    const externalEvent = { preventDefault: vi.fn() };
    const unsafeEvent = { preventDefault: vi.fn() };
    const webviewEvent = { preventDefault: vi.fn() };

    contents.emit("will-navigate", trustedEvent, `${RENDERER_URL}#settings`);
    contents.emit("will-redirect", externalEvent, "https://example.com/docs");
    contents.emit("will-navigate", unsafeEvent, "javascript:alert(1)");
    contents.emit("will-attach-webview", webviewEvent);
    const windowOpenHandler = (
      contents as typeof contents & {
        windowOpenHandler: (details: { url: string }) => { action: string };
      }
    ).windowOpenHandler;

    expect(trustedEvent.preventDefault).not.toHaveBeenCalled();
    expect(externalEvent.preventDefault).toHaveBeenCalledOnce();
    expect(unsafeEvent.preventDefault).toHaveBeenCalledOnce();
    expect(webviewEvent.preventDefault).toHaveBeenCalledOnce();
    expect(windowOpenHandler({ url: "https://example.org" })).toEqual({ action: "deny" });
    expect(windowOpenHandler({ url: "file:///tmp/secret" })).toEqual({ action: "deny" });
    await vi.waitFor(() =>
      expect(openExternal.mock.calls.map(([url]) => url)).toEqual([
        "https://example.com/docs",
        "https://example.org",
      ]),
    );
  });
});
