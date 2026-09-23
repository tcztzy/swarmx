import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, it, vi } from "vitest";
import { HARNESS_CAPABILITIES, type NativeAgent } from "../src/agents/types.js";
import { currentAgentBinding } from "../src/host/agent-registry.js";
import { HostOperations } from "../src/host/operations.js";
import { ProductServices } from "../src/host/product-services.js";
import { WorkConfigurationSchema } from "../src/work.js";

it.each([false, true])(
  "validates prepared instructions before reserving a runtime (oversize=%s)",
  async (oversize) => {
    const cwd = await mkdtemp(join(tmpdir(), "swarmx-managed-size-"));
    const products = await ProductServices.create({ cwd, productHome: join(cwd, "home") });
    products.settings.writeMemory({ autoReview: false });
    const goal = "g".repeat(16_000);
    const criteria = "c".repeat(16_000);
    const native: NativeAgent = {
      name: "local preparation fixture",
      capabilities: HARNESS_CAPABILITIES.pi,
      models: async () => ({
        models: [{ id: "fixture", name: "Fixture", efforts: [] }],
        current: {},
      }),
      create: async () => "pi:fixture",
      list: async () => [],
      read: async () => {},
      start: vi.fn(async (_session, text, observer) => {
        expect(text).toContain(goal);
        expect(text).toContain(criteria);
        expect(text).toContain("<swarmx-preparation>");
        observer.raw({}, { "swarmx.usage.cost_usd": 0 });
        return { stopReason: "end_turn" };
      }),
      steer: async () => {},
      interrupt: async () => {},
      dispose: async () => {},
    };
    try {
      await products.attachAgents("http://localhost", native, "pi");
      products.work.createCycle({ id: "cycle", project: "test", budgetUsd: 10 });
      products.work.createItem({
        id: "item",
        cycleId: "cycle",
        goal,
        criteria,
        criteriaVersion: "v1",
        taskClass: "analysis",
        mode: "managed",
        supervisor: { harness: "pi", model: "fixture" },
        runtime: { budgetUsd: 1 },
      });
      if (oversize)
        vi.spyOn(products, "callTool").mockResolvedValueOnce({ knowledge: "k".repeat(100_000) });
      const execution = products.runWork("item", new AbortController().signal);
      if (oversize) {
        await expect(execution).rejects.toThrow();
        expect(native.start).not.toHaveBeenCalled();
        expect(products.work.snapshot("cycle")).toMatchObject({
          items: [{ state: "queued" }],
          reservations: [],
          balance: { heldUsd: 0, remainingUsd: 10 },
        });
      } else {
        await execution;
        expect(native.start).toHaveBeenCalledTimes(1);
        expect(products.work.snapshot("cycle")).toMatchObject({
          reservations: [{ outcome: "completed" }],
          balance: { heldUsd: 0, remainingUsd: 10 },
        });
      }
    } finally {
      await products.dispose();
      await rm(cwd, { recursive: true, force: true });
    }
  },
);

it("keeps presets and temporary agent choices separate from runtime limits", () => {
  expect(
    WorkConfigurationSchema.safeParse({ harness: "pi", model: "worker", profile: "sdk-minimal" })
      .success,
  ).toBe(false);
  expect(
    WorkConfigurationSchema.safeParse({ harness: "dsh", model: "worker", profile: "sdk-minimal" })
      .success,
  ).toBe(true);
  expect(WorkConfigurationSchema.parse({ harness: "pi", model: "worker", effort: "high" })).toEqual(
    { id: "temporary", harness: "pi", model: "worker", effort: "high" },
  );
  expect(
    WorkConfigurationSchema.safeParse({ harness: "pi", model: "worker", version: "1" }).success,
  ).toBe(false);
  expect(
    WorkConfigurationSchema.safeParse({ harness: "pi", model: "worker", reserveUsd: 2 }).success,
  ).toBe(false);
});

it.each(["stop", "timeout"])(
  "propagates %s through an MCP caller while child creation is pending",
  async (cancel) => {
    const cwd = await mkdtemp(join(tmpdir(), "swarmx-managed-cancel-"));
    const products = await ProductServices.create({ cwd, productHome: join(cwd, "home") });
    products.settings.writeMemory({ autoReview: false });
    const rootReady = Promise.withResolvers<void>();
    const childCreating = Promise.withResolvers<void>();
    const releaseRoot = Promise.withResolvers<void>();
    const releaseChild = Promise.withResolvers<void>();
    const interrupted = Promise.withResolvers<void>();
    let sessions = 0;
    let stopped = false;
    const native: NativeAgent = {
      name: "local cancellation fixture",
      capabilities: HARNESS_CAPABILITIES.pi,
      models: async () => ({
        models: [{ id: "fixture", name: "Fixture", efforts: [] }],
        current: {},
      }),
      async create() {
        if (++sessions === 1) return "pi:root";
        childCreating.resolve();
        await releaseChild.promise;
        return "pi:child";
      },
      list: async () => [],
      read: async () => {},
      start: vi.fn(async (session) => {
        if (session === "pi:root") {
          rootReady.resolve();
          await releaseRoot.promise;
        }
        return { stopReason: stopped ? "cancelled" : "end_turn" };
      }),
      interrupt: async () => {
        stopped = true;
        interrupted.resolve();
      },
      steer: async () => {},
      dispose: async () => {},
    };
    try {
      await products.attachAgents("http://localhost", native, "pi");
      products.work.createCycle({ id: "cycle", project: "test", budgetUsd: 2 });
      products.work.createItem({
        id: "item",
        cycleId: "cycle",
        goal: "Analyze",
        criteria: "Verify",
        criteriaVersion: "1",
        taskClass: "test",
        mode: "managed",
        supervisor: { harness: "pi", model: "fixture" },
        runtime: { budgetUsd: 2, ...(cancel === "timeout" ? { timeoutMs: 200 } : {}) },
      });
      const operations = new HostOperations({
        products,
        signal: new AbortController().signal,
        origin: "http://localhost",
        token: "fixture",
        dispose: () => products.dispose(),
      });
      const running = operations.workCommand({ action: "start", workId: "item" });
      await rootReady.promise;
      const runId = products.journal.activeSession("pi:root")?.runId;
      assert.ok(runId);
      // Socket callbacks carry session/run credentials, not the Agent's AsyncLocalStorage scope.
      expect(products.journal.scope.getStore()).toBeUndefined();
      const context = {
        actorId: "mcp",
        sessionId: "pi:root",
        runId,
        signal: new AbortController().signal,
      };
      const prepared = (await products.callTool(
        "swarm",
        { action: "prepare", task: "Child", queries: ["test"] },
        { ...context, callId: randomUUID() },
      )) as { preparationId: string };
      const child = products.callTool(
        "swarm",
        {
          action: "send_message",
          agentId: "pi",
          model: "fixture",
          text: "Child",
          preparationId: prepared.preparationId,
          reason: "Check result",
        },
        { ...context, callId: randomUUID() },
      );
      const outcome = child.then(
        () => "dispatched",
        () => "cancelled",
      );
      await childCreating.promise;
      if (cancel === "stop") await operations.workCommand({ action: "stop", workId: "item" });
      await interrupted.promise;
      releaseChild.resolve();
      expect(await outcome).toBe("cancelled");
      expect(native.start).toHaveBeenCalledTimes(1);
      releaseRoot.resolve();
      await running;
      expect(products.work.snapshot("cycle").reservations[0]?.outcome).toBe("cancelled");
    } finally {
      releaseChild.resolve();
      releaseRoot.resolve();
      await products.dispose();
      await rm(cwd, { recursive: true, force: true });
    }
  },
);

it("prepares the supervisor before dispatch and lets it follow up on a child result in one shared runtime", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "swarmx-managed-"));
  const products = await ProductServices.create({ cwd, productHome: join(cwd, "home") });
  products.settings.writeMemory({ autoReview: false });
  const calls: string[] = [];
  const native: NativeAgent = {
    name: "local supervised work fixture",
    capabilities: HARNESS_CAPABILITIES.pi,
    models: async () => ({
      models: ["supervisor", "worker"].map((id) => ({ id, name: id, efforts: [] })),
      current: {},
    }),
    create: async () => `pi:${randomUUID()}`,
    list: async () => [],
    read: async () => {},
    async start(_id, text, observer, options) {
      calls.push(`${options?.model}:${text.startsWith("Follow up") ? "follow-up" : "start"}`);
      if (options?.model === "supervisor") {
        expect(text).toContain("<swarmx-preparation>");
        expect(text).toContain("Independently check the result");
        const tools = currentAgentBinding().productTools;
        assert.ok(tools);
        for (const task of ["First analysis", "Follow up: correct the missing control"]) {
          const prepare = (await tools.call(
            "swarm",
            { action: "prepare", task, queries: ["analysis"] },
            randomUUID(),
            new AbortController().signal,
          )) as { preparationId: string };
          const result = await tools.call(
            "swarm",
            {
              action: "send_message",
              agentId: "pi",
              model: "worker",
              text: task,
              reason: "Check and improve the returned analysis",
              preparationId: prepare.preparationId,
            },
            randomUUID(),
            new AbortController().signal,
          );
          expect(result).toMatchObject({ text: "observed output" });
        }
      }
      observer.text("answer", "observed output");
      observer.raw(
        {},
        { "swarmx.usage.cost_usd": 1, "swarmx.usage.cost_source": "native-estimate" },
      );
      return { stopReason: "end_turn" };
    },
    steer: async () => {},
    interrupt: async () => {},
    dispose: async () => {},
  };
  try {
    await products.attachAgents("http://localhost", native, "pi");
    products.work.createCycle({
      id: "cycle",
      project: "analysis",
      budgetUsd: 10,
      configurations: [{ id: "worker", harness: "pi", model: "worker" }],
    });
    products.work.createItem({
      id: "item",
      cycleId: "cycle",
      goal: "Analyze",
      criteria: "Independently check the result",
      criteriaVersion: "1",
      taskClass: "analysis",
      mode: "managed",
      supervisor: { harness: "pi", model: "supervisor" },
      runtime: { budgetUsd: 4, timeoutMs: 10_000 },
    });
    await products.runWork("item", new AbortController().signal);
    expect(calls).toEqual(["supervisor:start", "worker:start", "worker:follow-up"]);
    const snapshot = products.work.snapshot("cycle");
    expect(snapshot.balance).toMatchObject({ spentUsd: 3, heldUsd: 0, remainingUsd: 7 });
    expect(snapshot.reservations.map((row) => row.reservedUsd)).toEqual([4, 0, 0]);
    expect(snapshot.items[0]?.state).toBe("awaiting-acceptance");
    expect(snapshot.feedback).toEqual([]);
  } finally {
    await products.dispose();
    await rm(cwd, { recursive: true, force: true });
  }
});
