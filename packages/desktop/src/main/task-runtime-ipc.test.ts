import { TaskWorkItemSchema } from "@swarmx/core/task-runtime";
import { describe, expect, it, vi } from "vitest";
import { TaskRuntimeInvokeContracts } from "../shared/ipc-contracts/task-runtime.js";
import { createDesktopIpcRegistrar, createSemanticAuditReceipt } from "./ipc-router.js";
import { registerTaskRuntimeIpc, type TaskRuntimeIpcSupervisor } from "./task-runtime-ipc.js";

const REQUEST_ID = "00000000-0000-4000-8000-000000000001";
const workItem = TaskWorkItemSchema.parse({
  id: "awi_detached",
  status: "queued",
  executor: { backend: "test", operation: "test.echo" },
  createdAt: "2026-08-13T00:00:00.000Z",
  updatedAt: "2026-08-13T00:00:00.000Z",
});

describe("Task Runtime IPC router", () => {
  it("registers the read/control contracts and constructs only allowed commands", async () => {
    const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>();
    const registrar = createDesktopIpcRegistrar({
      registerAuthorized: (channel, handler) => handlers.set(channel, handler as never),
      auditPolicy: (channel) =>
        TaskRuntimeInvokeContracts[channel as keyof typeof TaskRuntimeInvokeContracts].audit,
    });
    const request = vi.fn<TaskRuntimeIpcSupervisor["request"]>(async (command) => {
      if (command.operation === "list") {
        return {
          requestId: REQUEST_ID,
          ok: true,
          operation: "list",
          workItems: [workItem],
          approvals: [],
          activeWorkItemIds: [workItem.id],
        };
      }
      return {
        requestId: REQUEST_ID,
        ok: true,
        operation: command.operation as "cancel" | "decide",
        workItem,
      };
    });
    registerTaskRuntimeIpc(registrar, { request });
    const invoke = (channel: string, ...args: unknown[]) =>
      handlers.get(channel)?.({}, createSemanticAuditReceipt(), ...args);

    expect([...handlers.keys()]).toEqual(Object.keys(TaskRuntimeInvokeContracts));
    await expect(invoke("taskRuntime:list")).resolves.toMatchObject({ operation: "list" });
    await expect(
      invoke("taskRuntime:cancel", { workItemId: "awi_detached", reason: "No longer needed." }),
    ).resolves.toMatchObject({ operation: "cancel" });
    await expect(
      invoke("taskRuntime:decide", {
        approvalId: "apr_detached",
        status: "approved",
        decidedBy: "desktop-user",
        reason: "Reviewed by the user.",
        response: { accepted: true },
      }),
    ).resolves.toMatchObject({ operation: "decide" });
    expect(request.mock.calls).toEqual([
      [{ operation: "list" }],
      [{ operation: "cancel", workItemId: "awi_detached", reason: "No longer needed." }],
      [
        {
          operation: "decide",
          approvalId: "apr_detached",
          status: "approved",
          decidedBy: "desktop-user",
          reason: "Reviewed by the user.",
          response: { accepted: true },
        },
      ],
    ]);
    expect(request).not.toHaveBeenCalledWith(expect.objectContaining({ operation: "run" }));
  });

  it("rejects invalid controls before Supervisor effects and validates operation drift", async () => {
    const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>();
    const registrar = createDesktopIpcRegistrar({
      registerAuthorized: (channel, handler) => handlers.set(channel, handler as never),
      auditPolicy: (channel) =>
        TaskRuntimeInvokeContracts[channel as keyof typeof TaskRuntimeInvokeContracts].audit,
    });
    const request = vi.fn<TaskRuntimeIpcSupervisor["request"]>(async (command) => ({
      requestId: REQUEST_ID,
      ok: true,
      operation:
        command.operation === "list"
          ? "cancel"
          : command.operation === "cancel"
            ? "decide"
            : "cancel",
      workItem,
    }));
    registerTaskRuntimeIpc(registrar, { request });
    const invoke = (channel: string, ...args: unknown[]) =>
      handlers.get(channel)?.({}, createSemanticAuditReceipt(), ...args);

    expect(() =>
      invoke("taskRuntime:cancel", { workItemId: "awi_detached", program: "/bin/sh" }),
    ).toThrow(/arguments failed validation/i);
    const overriddenCommands = [
      { operation: "list" },
      { operation: "ping" },
      {
        operation: "create",
        workItem: {
          id: "awi_injected",
          backend: "test",
          operation: "test.echo",
          input: {},
        },
      },
      {
        operation: "run",
        workItemId: "awi_detached",
        launch: {
          backendId: "test",
          program: "/bin/sh",
          args: [],
          cwd: "/tmp",
          env: {},
          environmentDigest: `sha256:${"a".repeat(64)}`,
        },
        grants: [],
      },
    ];
    for (const command of overriddenCommands) {
      expect(() => invoke("taskRuntime:cancel", command)).toThrow(/arguments failed validation/i);
      expect(() => invoke("taskRuntime:decide", command)).toThrow(/arguments failed validation/i);
    }
    expect(request).not.toHaveBeenCalled();
    await expect(invoke("taskRuntime:list")).rejects.toThrow(/unexpected cancel response/i);
    await expect(invoke("taskRuntime:cancel", { workItemId: "awi_detached" })).rejects.toThrow(
      /unexpected decide response/i,
    );
    await expect(
      invoke("taskRuntime:decide", {
        approvalId: "apr_detached",
        status: "approved",
        decidedBy: "desktop-user",
      }),
    ).rejects.toThrow(/unexpected cancel response/i);
  });
});
