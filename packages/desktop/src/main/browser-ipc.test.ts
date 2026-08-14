import { describe, expect, it, vi } from "vitest";
import { BrowserInvokeContracts, DESKTOP_BROWSER_LIMITS } from "../shared/ipc-contracts/browser.js";
import {
  type BrowserIpcHost,
  createDesktopBrowserHost,
  registerBrowserIpc,
  toDesktopBrowserState,
} from "./browser-ipc.js";
import { createDesktopIpcRegistrar, createSemanticAuditReceipt } from "./ipc-router.js";

const state = {
  id: "browser-1",
  url: "https://example.com/",
  title: "Example",
  loading: false,
  canGoBack: false,
  canGoForward: false,
};

describe("Browser IPC router", () => {
  it("registers all contracts, binds owners before create, and preserves action semantics", async () => {
    const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>();
    const registrar = createDesktopIpcRegistrar({
      registerAuthorized: (channel, handler) => handlers.set(channel, handler as never),
      auditPolicy: (channel) =>
        BrowserInvokeContracts[channel as keyof typeof BrowserInvokeContracts].audit,
    });
    const order: string[] = [];
    const host = {
      create: vi.fn((owner, request) => {
        order.push("create");
        expect(owner).toMatchObject({ id: 41 });
        expect(request).toBeUndefined();
        return state;
      }),
      navigate: vi.fn(async () => ({ ...state, loading: true })),
      back: vi.fn(() => false),
      forward: vi.fn(() => true),
      reload: vi.fn(() => false),
      getState: vi.fn(() => state),
      setBounds: vi.fn(() => false),
      setVisible: vi.fn(() => true),
      destroy: vi.fn(() => false),
    } satisfies BrowserIpcHost;
    const ensureInteractiveOwner = vi.fn(() => order.push("bind"));

    registerBrowserIpc(registrar, host, ensureInteractiveOwner);
    const event = { sender: { id: 41 } };
    const invoke = (channel: string, ...args: unknown[]) =>
      handlers.get(channel)?.(event, createSemanticAuditReceipt(), ...args);

    expect([...handlers.keys()]).toEqual(Object.keys(BrowserInvokeContracts));
    expect(invoke("browser:create")).toEqual(state);
    expect(order).toEqual(["bind", "create"]);
    await expect(
      invoke("browser:navigate", { id: "browser-1", url: "release notes" }),
    ).resolves.toEqual({ ...state, loading: true });
    expect(invoke("browser:back", { id: "browser-1" })).toEqual(state);
    expect(invoke("browser:forward", { id: "browser-1" })).toEqual(state);
    expect(invoke("browser:reload", { id: "browser-1" })).toEqual(state);
    expect(
      invoke("browser:setBounds", {
        id: "browser-1",
        bounds: {
          x: Number.NaN,
          y: Number.POSITIVE_INFINITY,
          width: Number.NEGATIVE_INFINITY,
          height: Number.NaN,
        },
      }),
    ).toEqual({ updated: false });
    expect(invoke("browser:setVisible", { id: "browser-1", visible: false })).toEqual({
      updated: true,
    });
    expect(invoke("browser:destroy", { id: "browser-1" })).toEqual({ destroyed: false });
    expect(host.back).toHaveBeenCalledWith(41, "browser-1");
    expect(host.forward).toHaveBeenCalledWith(41, "browser-1");
    expect(host.reload).toHaveBeenCalledWith(41, "browser-1");
    expect(host.getState).toHaveBeenCalledTimes(3);
    expect(host.setBounds).toHaveBeenCalledWith(41, "browser-1", {
      x: Number.NaN,
      y: Number.POSITIVE_INFINITY,
      width: Number.NEGATIVE_INFINITY,
      height: Number.NaN,
    });
  });

  it("rejects invalid input before effects and preserves missing-view failure differences", async () => {
    const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>();
    const registrar = createDesktopIpcRegistrar({
      registerAuthorized: (channel, handler) => handlers.set(channel, handler as never),
      auditPolicy: (channel) =>
        BrowserInvokeContracts[channel as keyof typeof BrowserInvokeContracts].audit,
    });
    const host = {
      create: vi.fn(() => state),
      navigate: vi.fn(async () => null),
      back: vi.fn(() => false),
      forward: vi.fn(() => false),
      reload: vi.fn(() => false),
      getState: vi.fn(() => null),
      setBounds: vi.fn(() => false),
      setVisible: vi.fn(() => false),
      destroy: vi.fn(() => false),
    } satisfies BrowserIpcHost;
    const ensureInteractiveOwner = vi.fn();
    registerBrowserIpc(registrar, host, ensureInteractiveOwner);
    const event = { sender: { id: 42 } };

    expect(() =>
      handlers.get("browser:navigate")?.(event, createSemanticAuditReceipt(), {
        id: "browser-1",
        url: "https://example.com",
        rawCredential: "secret",
      }),
    ).toThrow(/arguments failed validation/i);
    expect(host.navigate).not.toHaveBeenCalled();
    await expect(
      handlers.get("browser:navigate")?.(event, createSemanticAuditReceipt(), {
        id: "browser-1",
        url: "https://example.com",
      }),
    ).rejects.toThrow("Browser view is not available.");
    expect(() =>
      handlers.get("browser:back")?.(event, createSemanticAuditReceipt(), { id: "browser-1" }),
    ).toThrow("Browser view is not available.");
    expect(
      handlers.get("browser:setVisible")?.(event, createSemanticAuditReceipt(), {
        id: "browser-1",
        visible: true,
      }),
    ).toEqual({ updated: false });
    expect(
      handlers.get("browser:destroy")?.(event, createSemanticAuditReceipt(), { id: "browser-1" }),
    ).toEqual({
      destroyed: false,
    });
  });

  it("projects and validates bounded invoke and event state without leaking Host fields", () => {
    expect(createDesktopBrowserHost()).toBeDefined();
    const projected = toDesktopBrowserState({
      ...state,
      url: `https://example.com/${"界".repeat(DESKTOP_BROWSER_LIMITS.urlBytes)}`,
      title: "界".repeat(DESKTOP_BROWSER_LIMITS.titleBytes),
      error: "界".repeat(DESKTOP_BROWSER_LIMITS.errorBytes),
      hostOnly: "not transported",
    } as never);

    expect(Buffer.byteLength(projected.url)).toBeLessThanOrEqual(DESKTOP_BROWSER_LIMITS.urlBytes);
    expect(Buffer.byteLength(projected.title)).toBeLessThanOrEqual(
      DESKTOP_BROWSER_LIMITS.titleBytes,
    );
    expect(Buffer.byteLength(projected.error ?? "")).toBeLessThanOrEqual(
      DESKTOP_BROWSER_LIMITS.errorBytes,
    );
    expect(projected).not.toHaveProperty("hostOnly");
    expect(toDesktopBrowserState(state)).toEqual(state);
    expect(() => toDesktopBrowserState({ ...state, id: "" })).toThrow();
  });
});
