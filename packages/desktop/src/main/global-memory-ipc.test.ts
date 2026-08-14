import { describe, expect, it, vi } from "vitest";
import { GlobalMemoryInvokeContracts } from "../shared/ipc-contracts/global-memory.js";
import { type GlobalMemoryIpcService, registerGlobalMemoryIpc } from "./global-memory-ipc.js";
import { createDesktopIpcRegistrar, createSemanticAuditReceipt } from "./ipc-router.js";

const emptyMemory = {
  user: {
    target: "user" as const,
    fileName: "USER.md" as const,
    content: null,
    revision: 0,
    updatedAt: null,
  },
  memory: {
    target: "memory" as const,
    fileName: "MEMORY.md" as const,
    content: null,
    revision: 0,
    updatedAt: null,
  },
  legacyUser: false,
  maxCharacters: { user: 4_000 as const, memory: 4_000 as const },
};

describe("Global Memory IPC router", () => {
  it("registers every contract and validates values around Memory effects", async () => {
    const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>();
    const registrar = createDesktopIpcRegistrar({
      registerAuthorized: (channel, handler) => handlers.set(channel, handler as never),
      auditPolicy: (channel) =>
        GlobalMemoryInvokeContracts[channel as keyof typeof GlobalMemoryInvokeContracts].audit,
    });
    const service = {
      get: vi.fn(async () => emptyMemory),
      save: vi.fn(async () => emptyMemory),
      forget: vi.fn(async () => emptyMemory),
    } satisfies GlobalMemoryIpcService;

    registerGlobalMemoryIpc(registrar, service);
    const invoke = (channel: string, ...args: unknown[]) =>
      handlers.get(channel)?.({}, createSemanticAuditReceipt(), ...args);

    expect([...handlers.keys()]).toEqual(Object.keys(GlobalMemoryInvokeContracts));
    await expect(invoke("personalMemory:get")).resolves.toEqual(emptyMemory);
    await expect(
      invoke("personalMemory:save", { target: "user", content: "Prefer concise answers." }),
    ).resolves.toEqual(emptyMemory);
    await expect(
      invoke("personalMemory:forget", {
        target: "memory",
        confirmed: true,
        expectedRevision: 4,
      }),
    ).resolves.toEqual(emptyMemory);
    expect(service.save).toHaveBeenCalledWith({
      target: "user",
      content: "Prefer concise answers.",
    });
    expect(service.forget).toHaveBeenCalledWith({
      target: "memory",
      confirmed: true,
      expectedRevision: 4,
    });

    expect(() =>
      invoke("personalMemory:save", {
        target: "user",
        content: "secret",
        rawCredential: "not transported",
      }),
    ).toThrow(/arguments failed validation/i);
    expect(service.save).toHaveBeenCalledTimes(1);

    service.get.mockResolvedValueOnce({ ...emptyMemory, legacyUser: undefined } as never);
    await expect(invoke("personalMemory:get")).rejects.toThrow(/result failed validation/i);
  });
});
