import { describe, expect, it } from "vitest";
import { GlobalMemoryInvokeContracts } from "./global-memory.js";
import { DesktopEventContractRegistry, DesktopInvokeContractRegistry } from "./index.js";

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

describe("Global Memory IPC contracts", () => {
  it("owns the three compatibility-named invokes with stable audit policies", () => {
    expect(Object.keys(GlobalMemoryInvokeContracts)).toEqual([
      "personalMemory:get",
      "personalMemory:save",
      "personalMemory:forget",
    ]);
    expect(
      Object.fromEntries(
        Object.entries(GlobalMemoryInvokeContracts).map(([channel, contract]) => [
          channel,
          contract.audit,
        ]),
      ),
    ).toEqual({
      "personalMemory:get": "failure_only",
      "personalMemory:save": "intent_outcome",
      "personalMemory:forget": "intent_outcome",
    });
    expect(
      Object.keys(DesktopInvokeContractRegistry).filter((channel) =>
        channel.startsWith("personalMemory:"),
      ),
    ).toEqual(Object.keys(GlobalMemoryInvokeContracts));
    expect(
      Object.keys(DesktopEventContractRegistry).filter((channel) =>
        channel.startsWith("personalMemory:"),
      ),
    ).toEqual([]);
  });

  it("accepts strict Global Memory inputs and rejects legacy Personal Memory shapes", () => {
    expect(GlobalMemoryInvokeContracts["personalMemory:get"].args.parse([])).toEqual([]);
    expect(
      GlobalMemoryInvokeContracts["personalMemory:save"].args.parse([
        { target: "user", content: "Prefer concise answers.", expectedRevision: 2 },
      ]),
    ).toEqual([{ target: "user", content: "Prefer concise answers.", expectedRevision: 2 }]);
    expect(
      GlobalMemoryInvokeContracts["personalMemory:forget"].args.parse([
        { target: "memory", confirmed: true, expectedRevision: 3 },
      ]),
    ).toEqual([{ target: "memory", confirmed: true, expectedRevision: 3 }]);
    expect(
      GlobalMemoryInvokeContracts["personalMemory:save"].args.safeParse([
        { target: "user", content: "Compatibility client" },
      ]).success,
    ).toBe(true);
    expect(
      GlobalMemoryInvokeContracts["personalMemory:save"].args.safeParse([
        { content: "legacy shape" },
      ]).success,
    ).toBe(false);
    expect(
      GlobalMemoryInvokeContracts["personalMemory:forget"].args.safeParse([
        { target: "user", confirmed: false },
      ]).success,
    ).toBe(false);
    expect(
      GlobalMemoryInvokeContracts["personalMemory:save"].args.safeParse([
        { target: "user", content: "secret", rawCredential: "not transported" },
      ]).success,
    ).toBe(false);
  });

  it("requires the complete strict Global Memory state on every result", () => {
    expect(GlobalMemoryInvokeContracts["personalMemory:get"].result.parse(emptyMemory)).toEqual(
      emptyMemory,
    );
    expect(
      GlobalMemoryInvokeContracts["personalMemory:get"].result.safeParse({
        ...emptyMemory,
        legacyUser: undefined,
      }).success,
    ).toBe(false);
    expect(
      GlobalMemoryInvokeContracts["personalMemory:save"].result.safeParse({
        ...emptyMemory,
        user: { ...emptyMemory.user, fileName: "MEMORY.md" },
      }).success,
    ).toBe(false);
    expect(
      GlobalMemoryInvokeContracts["personalMemory:forget"].result.safeParse({
        ...emptyMemory,
        plaintextCredential: "secret",
      }).success,
    ).toBe(false);
  });
});
