import { Buffer } from "node:buffer";
import type { IpcMainInvokeEvent } from "electron";
import {
  BrowserInvokeContracts,
  DESKTOP_BROWSER_LIMITS,
  type DesktopBrowserState,
  DesktopBrowserStateSchema,
} from "../shared/ipc-contracts/browser.js";
import {
  BrowserHost,
  type BrowserOwner,
  type BrowserState,
  type CreateBrowserRequest,
} from "./browser-host.js";
import type { DesktopIpcRegistrar } from "./ipc-router.js";

export interface BrowserIpcHost {
  create(owner: BrowserOwner, request?: CreateBrowserRequest): BrowserState;
  navigate(ownerId: number, id: string, input: string): Promise<BrowserState | null>;
  back(ownerId: number, id: string): boolean;
  forward(ownerId: number, id: string): boolean;
  reload(ownerId: number, id: string): boolean;
  getState(ownerId: number, id: string): BrowserState | null;
  setBounds(ownerId: number, id: string, bounds: BrowserStateBounds): boolean;
  setVisible(ownerId: number, id: string, visible: boolean): boolean;
  destroy(ownerId: number, id: string): boolean;
}

type BrowserStateBounds = Parameters<BrowserHost["setBounds"]>[2];
type InteractiveOwner = IpcMainInvokeEvent["sender"];

export function createDesktopBrowserHost(): BrowserHost {
  return new BrowserHost(undefined, undefined, undefined, toDesktopBrowserState);
}

export function registerBrowserIpc(
  registrar: DesktopIpcRegistrar,
  host: BrowserIpcHost,
  ensureInteractiveOwner: (owner: InteractiveOwner) => void,
): void {
  const requiredState = (ownerId: number, id: string): DesktopBrowserState => {
    const state = host.getState(ownerId, id);
    if (!state) throw new Error("Browser view is not available.");
    return toDesktopBrowserState(state);
  };

  registrar.register(
    "browser:create",
    BrowserInvokeContracts["browser:create"],
    (event, [request]) => {
      ensureInteractiveOwner(event.sender);
      return toDesktopBrowserState(host.create(event.sender, request));
    },
  );
  registrar.register(
    "browser:navigate",
    BrowserInvokeContracts["browser:navigate"],
    async (event, [{ id, url }]) => {
      const state = await host.navigate(event.sender.id, id, url);
      if (!state) throw new Error("Browser view is not available.");
      return toDesktopBrowserState(state);
    },
  );
  registrar.register("browser:back", BrowserInvokeContracts["browser:back"], (event, [{ id }]) => {
    host.back(event.sender.id, id);
    return requiredState(event.sender.id, id);
  });
  registrar.register(
    "browser:forward",
    BrowserInvokeContracts["browser:forward"],
    (event, [{ id }]) => {
      host.forward(event.sender.id, id);
      return requiredState(event.sender.id, id);
    },
  );
  registrar.register(
    "browser:reload",
    BrowserInvokeContracts["browser:reload"],
    (event, [{ id }]) => {
      host.reload(event.sender.id, id);
      return requiredState(event.sender.id, id);
    },
  );
  registrar.register(
    "browser:setBounds",
    BrowserInvokeContracts["browser:setBounds"],
    (event, [{ id, bounds }]) => ({ updated: host.setBounds(event.sender.id, id, bounds) }),
  );
  registrar.register(
    "browser:setVisible",
    BrowserInvokeContracts["browser:setVisible"],
    (event, [{ id, visible }]) => ({ updated: host.setVisible(event.sender.id, id, visible) }),
  );
  registrar.register(
    "browser:destroy",
    BrowserInvokeContracts["browser:destroy"],
    (event, [{ id }]) => ({ destroyed: host.destroy(event.sender.id, id) }),
  );
}

export function toDesktopBrowserState(state: BrowserState): DesktopBrowserState {
  return DesktopBrowserStateSchema.parse({
    id: state.id,
    url: truncateUtf8(state.url, DESKTOP_BROWSER_LIMITS.urlBytes),
    title: truncateUtf8(state.title, DESKTOP_BROWSER_LIMITS.titleBytes),
    loading: state.loading,
    canGoBack: state.canGoBack,
    canGoForward: state.canGoForward,
    ...(state.error ? { error: truncateUtf8(state.error, DESKTOP_BROWSER_LIMITS.errorBytes) } : {}),
  });
}

function truncateUtf8(value: string, maximumBytes: number): string {
  const buffer = Buffer.from(value);
  if (buffer.length <= maximumBytes) return value;
  let length = maximumBytes;
  while (length > 0 && (buffer[length] & 0xc0) === 0x80) length -= 1;
  return buffer.subarray(0, length).toString("utf8");
}
