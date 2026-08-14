import { TaskWorkItemSchema } from "@swarmx/core/task-runtime";
import { describe, expect, it } from "vitest";
import { DesktopEventContractRegistry, DesktopInvokeContractRegistry } from "./index.js";
import { TaskRuntimeInvokeContracts } from "./task-runtime.js";

const REQUEST_ID = "00000000-0000-4000-8000-000000000001";
const workItem = TaskWorkItemSchema.parse({
  id: "awi_detached",
  status: "queued",
  executor: { backend: "test", operation: "test.echo" },
  createdAt: "2026-08-13T00:00:00.000Z",
  updatedAt: "2026-08-13T00:00:00.000Z",
});

describe("Task Runtime IPC contracts", () => {
  it("owns only list, cancel, and decide without launch authority", () => {
    expect(Object.keys(TaskRuntimeInvokeContracts)).toEqual([
      "taskRuntime:list",
      "taskRuntime:cancel",
      "taskRuntime:decide",
    ]);
    expect(
      Object.fromEntries(
        Object.entries(TaskRuntimeInvokeContracts).map(([channel, contract]) => [
          channel,
          contract.audit,
        ]),
      ),
    ).toEqual({
      "taskRuntime:list": "failure_only",
      "taskRuntime:cancel": "intent_outcome",
      "taskRuntime:decide": "intent_outcome",
    });
    expect(
      Object.keys(DesktopInvokeContractRegistry).filter((channel) =>
        channel.startsWith("taskRuntime:"),
      ),
    ).toEqual(Object.keys(TaskRuntimeInvokeContracts));
    expect(
      Object.keys(DesktopEventContractRegistry).filter((channel) =>
        channel.startsWith("taskRuntime:"),
      ),
    ).toEqual([]);
    expect("taskRuntime:run" in DesktopInvokeContractRegistry).toBe(false);
  });

  it("validates strict control inputs while preserving bounded worker responses", () => {
    expect(TaskRuntimeInvokeContracts["taskRuntime:list"].args.parse([])).toEqual([]);
    expect(
      TaskRuntimeInvokeContracts["taskRuntime:cancel"].args.parse([
        { workItemId: "awi_detached", reason: "No longer needed." },
      ]),
    ).toEqual([{ workItemId: "awi_detached", reason: "No longer needed." }]);
    expect(
      TaskRuntimeInvokeContracts["taskRuntime:decide"].args.parse([
        {
          approvalId: "apr_detached",
          status: "approved",
          decidedBy: "desktop-user",
          response: { accepted: true },
        },
      ]),
    ).toEqual([
      {
        approvalId: "apr_detached",
        status: "approved",
        decidedBy: "desktop-user",
        response: { accepted: true },
      },
    ]);
    expect(
      TaskRuntimeInvokeContracts["taskRuntime:cancel"].args.safeParse([
        { workItemId: "awi_detached", program: "/bin/sh" },
      ]).success,
    ).toBe(false);
    expect(
      TaskRuntimeInvokeContracts["taskRuntime:decide"].args.safeParse([
        {
          approvalId: "apr_detached",
          status: "requested",
          decidedBy: "desktop-user",
        },
      ]).success,
    ).toBe(false);
    expect(
      TaskRuntimeInvokeContracts["taskRuntime:decide"].args.safeParse([
        {
          approvalId: "apr_detached",
          status: "approved",
          decidedBy: "desktop-user",
          response: { apiKey: "inline-secret" },
        },
      ]).success,
    ).toBe(false);
  });

  it("validates operation-specific strict Supervisor receipts", () => {
    expect(
      TaskRuntimeInvokeContracts["taskRuntime:list"].result.parse({
        requestId: REQUEST_ID,
        ok: true,
        operation: "list",
        workItems: [workItem],
        approvals: [],
        activeWorkItemIds: [workItem.id],
      }),
    ).toMatchObject({ operation: "list", workItems: [workItem] });
    expect(
      TaskRuntimeInvokeContracts["taskRuntime:cancel"].result.parse({
        requestId: REQUEST_ID,
        ok: true,
        operation: "cancel",
        workItem,
      }),
    ).toMatchObject({ operation: "cancel", workItem });
    expect(
      TaskRuntimeInvokeContracts["taskRuntime:decide"].result.safeParse({
        requestId: REQUEST_ID,
        ok: true,
        operation: "cancel",
        workItem,
      }).success,
    ).toBe(false);
    expect(
      TaskRuntimeInvokeContracts["taskRuntime:list"].result.safeParse({
        requestId: "not-a-uuid",
        ok: true,
        operation: "list",
        workItems: [],
        approvals: [],
        activeWorkItemIds: [],
      }).success,
    ).toBe(false);
    expect(
      TaskRuntimeInvokeContracts["taskRuntime:cancel"].result.safeParse({
        requestId: REQUEST_ID,
        ok: true,
        operation: "cancel",
        workItem,
        token: "not transported",
      }).success,
    ).toBe(false);
  });

  it("rejects oversized, deeply nested, cyclic, and excessively broad responses", () => {
    const decision = TaskRuntimeInvokeContracts["taskRuntime:decide"].args;
    const input = (response: unknown) => [
      {
        approvalId: "apr_detached",
        status: "approved" as const,
        decidedBy: "desktop-user",
        response,
      },
    ];
    const deeplyNested: unknown[] = [];
    let cursor = deeplyNested;
    for (let depth = 0; depth < 33; depth += 1) {
      const child: unknown[] = [];
      cursor.push(child);
      cursor = child;
    }
    const cyclic: unknown[] = [];
    cyclic.push(cyclic);

    expect(decision.safeParse(input("x".repeat(256 * 1024))).success).toBe(false);
    expect(decision.safeParse(input("\u0000".repeat(50_000))).success).toBe(false);
    expect(decision.safeParse(input({ count: 1, enabled: false })).success).toBe(true);
    expect(decision.safeParse(input(null)).success).toBe(true);
    expect(decision.safeParse(input({ ["k".repeat(256 * 1024)]: null })).success).toBe(false);
    const nearlyFullKey = "k".repeat(256 * 1024 - 6);
    expect(decision.safeParse(input({ [nearlyFullKey]: [] })).success).toBe(false);
    expect(decision.safeParse(input({ [nearlyFullKey]: {} })).success).toBe(false);
    expect(() => decision.safeParse(input(deeplyNested))).not.toThrow();
    expect(decision.safeParse(input(deeplyNested)).success).toBe(false);
    expect(() => decision.safeParse(input(cyclic))).not.toThrow();
    expect(decision.safeParse(input(cyclic)).success).toBe(false);
    expect(decision.safeParse(input(Array.from({ length: 10_000 }, () => null))).success).toBe(
      false,
    );
    expect(
      decision.safeParse(
        input(
          Object.fromEntries(Array.from({ length: 10_000 }, (_, index) => [`k${index}`, null])),
        ),
      ).success,
    ).toBe(false);

    const shared = { accepted: true };
    expect(decision.safeParse(input({ left: shared, right: shared })).success).toBe(true);
    expect(decision.safeParse(input(Object.create({ inherited: true }))).success).toBe(true);

    const hostile = new Proxy(
      {},
      {
        ownKeys() {
          throw new Error("enumeration failed");
        },
      },
    );
    expect(() => decision.safeParse(input(hostile))).not.toThrow();
    expect(decision.safeParse(input(hostile)).success).toBe(false);

    let reads = 0;
    const stateful: Record<string, unknown> = {};
    Object.defineProperty(stateful, "value", {
      enumerable: true,
      get() {
        reads += 1;
        return reads === 1 ? null : stateful;
      },
    });
    const statefulResult = decision.safeParse(input(stateful));
    expect(statefulResult.success).toBe(true);
    if (statefulResult.success) expect(statefulResult.data[0].response).toEqual({ value: null });
    expect(reads).toBe(1);

    let lengthReads = 0;
    const statefulArray = new Proxy([], {
      get(target, key, receiver) {
        if (key !== "length") return Reflect.get(target, key, receiver);
        lengthReads += 1;
        if (lengthReads > 1) throw new Error("length read twice");
        return 0;
      },
    });
    expect(decision.safeParse(input(statefulArray)).success).toBe(true);
    expect(lengthReads).toBe(1);
    const invalidLength = new Proxy([], {
      get(target, key, receiver) {
        return key === "length" ? Number.POSITIVE_INFINITY : Reflect.get(target, key, receiver);
      },
    });
    expect(decision.safeParse(input(invalidLength)).success).toBe(false);
  });
});
