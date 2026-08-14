import { describe, expect, it, vi } from "vitest";
import {
  TerminalEventContracts,
  TerminalInvokeContracts,
} from "../shared/ipc-contracts/terminal.js";
import { createDesktopIpcRegistrar, createSemanticAuditReceipt } from "./ipc-router.js";
import {
  registerTerminalIpc,
  type TerminalIpcHost,
  toDesktopTerminalDataEvent,
  toDesktopTerminalExitEvent,
} from "./terminal-ipc.js";

describe("Terminal IPC router", () => {
  it("registers all contracts and preserves Host-before-owner-binding create order", () => {
    const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>();
    const registrar = createDesktopIpcRegistrar({
      registerAuthorized: (channel, handler) => handlers.set(channel, handler as never),
      auditPolicy: (channel) =>
        TerminalInvokeContracts[channel as keyof typeof TerminalInvokeContracts].audit,
    });
    const order: string[] = [];
    const hostMarkers = new Map<string, (() => void) | undefined>();
    const host = {
      create: vi.fn((owner, request, recordSemanticAudit) => {
        order.push("create");
        hostMarkers.set("terminal:create", recordSemanticAudit);
        expect(owner).toMatchObject({ id: 41 });
        expect(request).toEqual({ id: "terminal-1", cwd: "/workspace", cols: 80, rows: 24 });
        recordSemanticAudit?.();
        return { id: "terminal-1", pid: 42 };
      }),
      write: vi.fn((_ownerId, _id, _data, recordSemanticAudit) => {
        hostMarkers.set("terminal:write", recordSemanticAudit);
        recordSemanticAudit?.();
        return true;
      }),
      resize: vi.fn((_ownerId, _id, _cols, _rows, recordSemanticAudit) => {
        hostMarkers.set("terminal:resize", recordSemanticAudit);
        recordSemanticAudit?.();
        return false;
      }),
      kill: vi.fn((_ownerId, _id, recordSemanticAudit) => {
        hostMarkers.set("terminal:kill", recordSemanticAudit);
        recordSemanticAudit?.();
        return true;
      }),
    } satisfies TerminalIpcHost;
    const ensureInteractiveOwner = vi.fn(() => order.push("bind"));

    registerTerminalIpc(registrar, host, ensureInteractiveOwner);
    const event = { sender: { id: 41 } };
    const receipts = new Map<string, ReturnType<typeof createSemanticAuditReceipt>>();
    const invoke = (channel: string, ...args: unknown[]) => {
      const receipt = createSemanticAuditReceipt();
      receipts.set(channel, receipt);
      return handlers.get(channel)?.(event, receipt, ...args);
    };

    expect([...handlers.keys()]).toEqual(Object.keys(TerminalInvokeContracts));
    expect(
      invoke("terminal:create", {
        id: "terminal-1",
        cwd: "/workspace",
        cols: 80,
        rows: 24,
      }),
    ).toEqual({ id: "terminal-1", pid: 42 });
    expect(order).toEqual(["create", "bind"]);
    expect(invoke("terminal:write", { id: "terminal-1", data: "pwd\r" })).toEqual({
      written: true,
    });
    expect(
      invoke("terminal:resize", {
        id: "terminal-1",
        cols: Number.NaN,
        rows: Number.POSITIVE_INFINITY,
      }),
    ).toEqual({ resized: false });
    expect(invoke("terminal:kill", { id: "terminal-1" })).toEqual({ killed: true });
    expect(host.write).toHaveBeenCalledWith(41, "terminal-1", "pwd\r", expect.any(Function));
    expect(host.resize).toHaveBeenCalledWith(
      41,
      "terminal-1",
      Number.NaN,
      Number.POSITIVE_INFINITY,
      expect.any(Function),
    );
    expect(host.kill).toHaveBeenCalledWith(41, "terminal-1", expect.any(Function));
    for (const channel of Object.keys(TerminalInvokeContracts)) {
      expect(receipts.get(channel)?.semanticAuditRecorded).toBe(true);
      expect(hostMarkers.get(channel)).toBe(receipts.get(channel)?.recordSemanticAudit);
    }
  });

  it("rejects invalid arguments before Host effects and does not bind a failed create", () => {
    const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>();
    const registrar = createDesktopIpcRegistrar({
      registerAuthorized: (channel, handler) => handlers.set(channel, handler as never),
      auditPolicy: (channel) =>
        TerminalInvokeContracts[channel as keyof typeof TerminalInvokeContracts].audit,
    });
    const host = {
      create: vi.fn(() => {
        throw new Error("spawn failed");
      }),
      write: vi.fn(() => true),
      resize: vi.fn(() => true),
      kill: vi.fn(() => true),
    } satisfies TerminalIpcHost;
    const ensureInteractiveOwner = vi.fn();
    registerTerminalIpc(registrar, host, ensureInteractiveOwner);
    const event = { sender: { id: 42 } };

    expect(() =>
      handlers.get("terminal:write")?.(event, createSemanticAuditReceipt(), {
        id: "terminal-1",
        data: "secret",
        rawCredential: "not transported",
      }),
    ).toThrow(/arguments failed validation/i);
    expect(host.write).not.toHaveBeenCalled();
    expect(() =>
      handlers.get("terminal:create")?.(event, createSemanticAuditReceipt(), {
        id: "terminal-1",
        cwd: "/workspace",
      }),
    ).toThrow("spawn failed");
    expect(ensureInteractiveOwner).not.toHaveBeenCalled();
  });

  it("strictly projects Main-published data and exit events before sending", () => {
    const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>();
    const registrar = createDesktopIpcRegistrar({
      registerAuthorized: (channel, handler) => handlers.set(channel, handler as never),
      auditPolicy: (channel) =>
        TerminalInvokeContracts[channel as keyof typeof TerminalInvokeContracts].audit,
    });
    let projectedOwner: Parameters<TerminalIpcHost["create"]>[0] | undefined;
    const host = {
      create: vi.fn((owner) => {
        projectedOwner = owner;
        return { id: "terminal-1", pid: 42 };
      }),
      write: vi.fn(() => true),
      resize: vi.fn(() => true),
      kill: vi.fn(() => true),
    } satisfies TerminalIpcHost;
    const send = vi.fn<(channel: string, value: unknown) => void>();
    registerTerminalIpc(registrar, host, vi.fn());
    handlers.get("terminal:create")?.(
      { sender: { id: 43, isDestroyed: () => false, send } },
      createSemanticAuditReceipt(),
      { id: "terminal-1", cwd: "/workspace" },
    );
    expect(projectedOwner?.isDestroyed?.()).toBe(false);

    projectedOwner?.send("terminal:data", { id: "terminal-1", data: "raw" });
    projectedOwner?.send("terminal:exit", { id: "terminal-1", exitCode: 0 });
    expect(send).toHaveBeenNthCalledWith(1, "terminal:data", {
      id: "terminal-1",
      data: "raw",
    });
    expect(send).toHaveBeenNthCalledWith(2, "terminal:exit", {
      id: "terminal-1",
      exitCode: 0,
    });
    expect(() =>
      projectedOwner?.send("terminal:data", {
        id: "terminal-1",
        data: "raw",
        cwd: "/secret",
      }),
    ).toThrow();
    expect(() =>
      projectedOwner?.send("terminal:exit", {
        id: "terminal-1",
        exitCode: Number.POSITIVE_INFINITY,
      }),
    ).toThrow();
    expect(() => projectedOwner?.send("terminal:unknown", {})).toThrow(
      "Unsupported Terminal event channel",
    );
    expect(send).toHaveBeenCalledTimes(2);

    send.mockImplementation(() => {
      throw new Error("renderer disappeared");
    });
    expect(() =>
      projectedOwner?.send("terminal:data", { id: "terminal-1", data: "late" }),
    ).not.toThrow();
    expect(() =>
      projectedOwner?.send("terminal:data", { id: "terminal-1", data: "late", cwd: "/secret" }),
    ).toThrow();

    expect(toDesktopTerminalDataEvent({ id: "terminal-1", data: "raw" })).toEqual({
      id: "terminal-1",
      data: "raw",
    });
    expect(toDesktopTerminalExitEvent({ id: "terminal-1", exitCode: 0 })).toEqual({
      id: "terminal-1",
      exitCode: 0,
    });
    expect(() =>
      toDesktopTerminalDataEvent({ id: "terminal-1", data: "raw", cwd: "/secret" } as never),
    ).toThrow();
    expect(() =>
      TerminalEventContracts["terminal:exit"].payload.parse({
        id: "terminal-1",
        exitCode: 0,
        environment: "secret",
      }),
    ).toThrow();
  });
});
