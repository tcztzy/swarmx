import { describe, expect, it, vi } from "vitest";
import { ProjectInvokeContracts } from "../shared/ipc-contracts/project.js";
import { createDesktopIpcRegistrar, createSemanticAuditReceipt } from "./ipc-router.js";
import { registerProjectIpc } from "./project-ipc.js";
import type { ProjectServiceLike } from "./project-service.js";

const project = {
  id: "project-1",
  name: "Project One",
  cwd: "/workspace/project-one",
  pinned: false,
  createdAt: "2026-08-13T00:00:00.000Z",
  updatedAt: "2026-08-13T00:00:00.000Z",
};

describe("Project IPC router", () => {
  it("registers every contract and validates values around Project effects", async () => {
    const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>();
    const registrar = createDesktopIpcRegistrar({
      registerAuthorized: (channel, handler) => handlers.set(channel, handler as never),
      auditPolicy: (channel) =>
        ProjectInvokeContracts[channel as keyof typeof ProjectInvokeContracts].audit,
    });
    const service = {
      list: vi.fn(() => []),
      addExisting: vi.fn(async () => null),
      createScratch: vi.fn(async () => null),
      setPinned: vi.fn((_id: string, pinned: boolean) => ({ ...project, pinned })),
      rename: vi.fn((_id: string, name: string) => ({ ...project, name })),
      reveal: vi.fn((id: string) => id === "project-1"),
      archiveTasks: vi.fn((id: string) => (id === "project-1" ? 2 : 0)),
      remove: vi.fn((id: string) => id === "project-1"),
    } satisfies ProjectServiceLike;

    registerProjectIpc(registrar, service);
    const invoke = (channel: string, ...args: unknown[]) =>
      handlers.get(channel)?.({}, createSemanticAuditReceipt(), ...args);

    expect(handlers.size).toBe(Object.keys(ProjectInvokeContracts).length);
    expect(invoke("project:list")).toEqual([]);
    await expect(invoke("project:addExisting")).resolves.toBeNull();
    await expect(invoke("project:createScratch")).resolves.toBeNull();
    expect(invoke("project:setPinned", { id: "project-1", pinned: true })).toMatchObject({
      id: "project-1",
      pinned: true,
    });
    expect(invoke("project:rename", { id: "project-1", name: "Renamed" })).toMatchObject({
      id: "project-1",
      name: "Renamed",
    });
    expect(invoke("project:reveal", { id: "project-1" })).toBe(true);
    expect(invoke("project:archiveTasks", { id: "project-1" })).toBe(2);
    expect(invoke("project:remove", { id: "project-1" })).toBe(true);

    expect(() => invoke("project:setPinned", { id: "project-1", pinned: "yes" })).toThrow(
      /arguments failed validation/i,
    );
    expect(service.setPinned).toHaveBeenCalledTimes(1);

    service.list.mockReturnValueOnce([{ ...project, pinned: undefined }] as never);
    expect(() => invoke("project:list")).toThrow(/result failed validation/i);
  });
});
