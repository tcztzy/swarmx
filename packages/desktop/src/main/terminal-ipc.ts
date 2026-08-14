import type { IpcMainInvokeEvent } from "electron";
import {
  type DesktopTerminalDataEvent,
  DesktopTerminalDataEventSchema,
  type DesktopTerminalExitEvent,
  DesktopTerminalExitEventSchema,
  TerminalInvokeContracts,
} from "../shared/ipc-contracts/terminal.js";
import type { DesktopIpcRegistrar } from "./ipc-router.js";
import type {
  CreateTerminalRequest,
  TerminalOwner,
  TerminalSemanticAuditMarker,
} from "./terminal-host.js";

export interface TerminalIpcHost {
  create(
    owner: TerminalOwner,
    request: CreateTerminalRequest,
    recordSemanticAudit?: TerminalSemanticAuditMarker,
  ): { id: string; pid: number };
  write(
    ownerId: number,
    id: string,
    data: string,
    recordSemanticAudit?: TerminalSemanticAuditMarker,
  ): boolean;
  resize(
    ownerId: number,
    id: string,
    cols: number,
    rows: number,
    recordSemanticAudit?: TerminalSemanticAuditMarker,
  ): boolean;
  kill(ownerId: number, id: string, recordSemanticAudit?: TerminalSemanticAuditMarker): boolean;
}

type InteractiveOwner = IpcMainInvokeEvent["sender"];

export function registerTerminalIpc(
  registrar: DesktopIpcRegistrar,
  host: TerminalIpcHost,
  ensureInteractiveOwner: (owner: InteractiveOwner) => void,
): void {
  registrar.register(
    "terminal:create",
    TerminalInvokeContracts["terminal:create"],
    (event, [request], receipt) => {
      const created = host.create(
        projectedTerminalOwner(event.sender),
        request,
        receipt.recordSemanticAudit,
      );
      ensureInteractiveOwner(event.sender);
      return created;
    },
  );
  registrar.register(
    "terminal:write",
    TerminalInvokeContracts["terminal:write"],
    (event, [{ id, data }], receipt) => ({
      written: host.write(event.sender.id, id, data, receipt.recordSemanticAudit),
    }),
  );
  registrar.register(
    "terminal:resize",
    TerminalInvokeContracts["terminal:resize"],
    (event, [{ id, cols, rows }], receipt) => ({
      resized: host.resize(event.sender.id, id, cols, rows, receipt.recordSemanticAudit),
    }),
  );
  registrar.register(
    "terminal:kill",
    TerminalInvokeContracts["terminal:kill"],
    (event, [{ id }], receipt) => ({
      killed: host.kill(event.sender.id, id, receipt.recordSemanticAudit),
    }),
  );
}

export function toDesktopTerminalDataEvent(value: unknown): DesktopTerminalDataEvent {
  return DesktopTerminalDataEventSchema.parse(value);
}

export function toDesktopTerminalExitEvent(value: unknown): DesktopTerminalExitEvent {
  return DesktopTerminalExitEventSchema.parse(value);
}

function projectedTerminalOwner(owner: InteractiveOwner): TerminalOwner {
  return {
    id: owner.id,
    isDestroyed: () => owner.isDestroyed(),
    send: (channel, value) => {
      if (channel === "terminal:data") {
        sendTerminalEvent(owner, channel, toDesktopTerminalDataEvent(value));
        return;
      }
      if (channel === "terminal:exit") {
        sendTerminalEvent(owner, channel, toDesktopTerminalExitEvent(value));
        return;
      }
      throw new Error(`Unsupported Terminal event channel: ${channel}`);
    },
  };
}

function sendTerminalEvent(owner: InteractiveOwner, channel: string, value: unknown): void {
  try {
    owner.send(channel, value);
  } catch {
    // The owner can disappear after isDestroyed() but before Electron sends.
  }
}
