import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import { z } from "zod";
import { HARNESS_CAPABILITIES, type NativeAgent, type Observer } from "../src/agents/types.js";
import { ProductServices, type ProductServicesOptions } from "../src/host/product-services.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
const signal = () => new AbortController().signal;
const sink: Observer = { text() {}, tool() {}, raw() {}, interact: async () => undefined };
const configuration = {
  id: "registered",
  harness: "codex",
  model: "fixture/research",
};

async function tool(services: ProductServices, name: string, args: unknown) {
  return services.callTool(name, args, { actorId: "mcp", callId: randomUUID(), signal: signal() });
}

async function fixture(
  execute: (services: ProductServices, text: string) => Promise<void>,
  referenceProvider?: ProductServicesOptions["referenceProvider"],
) {
  const cwd = await mkdtemp(join(tmpdir(), "swarmx-work-boundaries-"));
  const services = await ProductServices.create({
    cwd,
    productHome: join(cwd, "home"),
    ...(referenceProvider === undefined ? {} : { referenceProvider }),
  });
  cleanups.push(async () => {
    await services.dispose();
    await rm(cwd, { recursive: true, force: true });
  });
  services.settings.writeMemory({ autoReview: false });
  let sessions = 0;
  const start = vi.fn<NativeAgent["start"]>(async (_session, text, observer) => {
    await execute(services, text);
    observer.text("answer", "Task output requires independent acceptance.");
    observer.raw(
      { synthetic: true },
      {
        "swarmx.usage.cost_usd": 0.4,
        "swarmx.usage.cost_source": "native-estimate",
        "swarmx.usage.scope": "run-total",
        "swarmx.usage.coverage": "complete",
      },
    );
    return { stopReason: "end_turn" };
  });
  const native: NativeAgent = {
    name: "work boundary fixture",
    capabilities: HARNESS_CAPABILITIES.codex,
    models: async () => ({
      models: [{ id: configuration.model, name: "Research fixture", efforts: [] }],
      current: {},
    }),
    list: async () => [],
    create: async () => `codex:boundary-${++sessions}`,
    read: async () => {},
    start,
    steer: async () => {},
    interrupt: async () => {},
    dispose: async () => {},
  };
  await services.attachAgents("http://localhost", native, "codex");
  services.work.createCycle({
    id: "cycle",
    project: "research",
    budgetUsd: 1,
    concurrency: 3,
    configurations: [configuration],
  });
  services.work.createItem({
    id: "first",
    cycleId: "cycle",
    goal: "Parent task",
    criteria: "Independently verify output",
    criteriaVersion: "criteria-v1",
    taskClass: "research",
    runtime: { budgetUsd: 0.4 },
  });
  return { cwd, services, start };
}

it("rejects native self-acceptance and administration while retaining the unaccepted runtime result", async () => {
  const { services } = await fixture(async (products) => {
    await expect(tool(services, "work", { action: "accept", accepted: true })).rejects.toThrow();
    expect(() => products.work.accept({})).toThrow("trusted Host caller");
    expect(() => products.work.createCycle({})).toThrow("trusted Host caller");
    await expect(products.runWork("first", signal())).rejects.toThrow("trusted Host caller");
    const status = await tool(services, "work", { action: "status" });
    expect(status).toMatchObject({ item: { id: "first", state: "running" }, heldUsd: 0.4 });
  });
  await services.runWork("first", signal());
  const snapshot = services.work.snapshot("cycle");
  expect(snapshot.items[0]?.state).toBe("awaiting-acceptance");
  expect(snapshot.feedback).toEqual([]);
  expect(snapshot.reservations).toHaveLength(1);
});

it("requires exact-task preparation and atomically shares the parent budget across child calls", async () => {
  const childStarted = Promise.withResolvers<void>();
  const releaseChild = Promise.withResolvers<void>();
  let childCount = 0;
  const { services, start } = await fixture(async (products, text) => {
    if (text === "Child task") {
      if (++childCount === 2) childStarted.resolve();
      await releaseChild.promise;
      return;
    }
    const send = {
      action: "send_message",
      agentId: "codex",
      model: configuration.model,
      text: "Child task",
      reason: "Use the registered child configuration.",
    };
    await expect(tool(services, "swarm", send)).rejects.toThrow("swarm.prepare");
    expect(products.work.snapshot("cycle").reservations).toHaveLength(1);
    const preparation = z.object({ preparationId: z.string() }).parse(
      await tool(services, "swarm", {
        action: "prepare",
        task: "Child task",
        queries: ["research"],
      }),
    );
    await expect(
      tool(services, "swarm", { ...send, text: "Different child task", ...preparation }),
    ).rejects.toThrow("exact task");
    const children = [1, 2].map(() => tool(services, "swarm", { ...send, ...preparation }));
    try {
      await childStarted.promise;
      const snapshot = products.work.snapshot("cycle");
      expect(snapshot.reservations).toHaveLength(3);
      expect(snapshot.reservations.slice(1)).toEqual([
        expect.objectContaining({ runtimeId: snapshot.reservations[0]?.id, reservedUsd: 0 }),
        expect.objectContaining({ runtimeId: snapshot.reservations[0]?.id, reservedUsd: 0 }),
      ]);
      expect(snapshot.balance.heldUsd).toBe(1.5);
      expect(start).toHaveBeenCalledTimes(3);
    } finally {
      releaseChild.resolve();
      await Promise.all(children);
    }
  });
  services.work.setBudget({ cycleId: "cycle", expectedBudgetUsd: 1, budgetUsd: 2 });
  await services.runWork("first", signal(), undefined, { runtime: { budgetUsd: 1.5 } });
  const snapshot = services.work.snapshot("cycle");
  expect(
    snapshot.reservations.map(({ purpose, workId, cycleId }) => ({ purpose, workId, cycleId })),
  ).toEqual([
    { purpose: "execution", workId: "first", cycleId: "cycle" },
    { purpose: "delegation", workId: "first", cycleId: "cycle" },
    { purpose: "delegation", workId: "first", cycleId: "cycle" },
  ]);
  expect(snapshot.reservations.every(({ runIds }) => runIds.length === 1)).toBe(true);
  expect(snapshot.balance.spentUsd).toBeCloseTo(1.2);
  expect(snapshot.balance.heldUsd).toBe(0);
});

it("prevents a managed native session from escaping its original work identity", async () => {
  const { services, start } = await fixture(async () => {});
  const completed = await services.runWork("first", signal());
  assert.ok("result" in completed);
  const { sessionId } = z.object({ sessionId: z.string() }).parse(completed.result);
  const agent = await services.agent("codex");
  await expect(
    agent.start(sessionId, "Resume outside work", sink, { model: configuration.model }),
  ).rejects.toThrow("original work identity");
  await expect(
    services.journal.scope.run(
      {
        sessionId: null,
        runId: randomUUID(),
        causedBy: null,
        attributes: { "swarmx.work.item_id": "another-work" },
      },
      () => agent.start(sessionId, "Reuse for another goal", sink, { model: configuration.model }),
    ),
  ).rejects.toThrow("original work identity");
  expect(start).toHaveBeenCalledTimes(1);
  expect(services.work.snapshot("cycle").reservations).toHaveLength(1);
});

it.each([true, false])(
  "preserves provider exact revisions and science.read permission (granted=%s)",
  async (granted) => {
    const logicalId = "sx:a/evidence";
    const evidence = { id: `${logicalId}@1`, revision: "1" };
    const referenceProvider = {
      scheme: "sx:",
      requiredPermissions: ["science.read"] as const,
      resolve: vi.fn(() => ({ id: logicalId, exactId: evidence.id, revision: evidence.revision })),
      checkResource: vi.fn(() => undefined),
    };
    const { services } = await fixture(async () => {
      if (!granted) {
        await expect(
          tool(services, "work", { action: "submit", artifacts: [evidence] }),
        ).rejects.toThrow("Science read permission");
        return;
      }
      await expect(
        tool(services, "work", { action: "submit", artifacts: [{ ...evidence, id: logicalId }] }),
      ).rejects.toThrow(/exact.*resource ID/iu);
      await expect(
        tool(services, "work", { action: "submit", artifacts: [{ ...evidence, revision: "2" }] }),
      ).rejects.toThrow("matching revision");
      await expect(
        tool(services, "work", { action: "submit", artifacts: [evidence] }),
      ).resolves.toMatchObject({ artifacts: [evidence] });
    }, referenceProvider);
    if (!granted) {
      const { policy } = services.settings.read();
      services.updatePolicy({
        ...policy,
        tools: policy.tools.filter((permission) => permission !== "science.read"),
      });
    }
    const execution = await services.runWork("first", signal());
    assert.ok(execution.reservation);
    const snapshot = services.work.snapshot("cycle");
    expect(snapshot.reservations[0]?.artifacts).toEqual(granted ? [evidence] : []);
    expect(referenceProvider.resolve).toHaveBeenCalledTimes(granted ? 3 : 0);
    expect(referenceProvider.checkResource).not.toHaveBeenCalled();
    if (granted)
      expect(() =>
        services.work.accept({
          id: "mismatched-report",
          attemptId: execution.reservation.id,
          criteriaVersion: "criteria-v1",
          verdict: "passed",
          accepted: true,
          fraction: 1,
          layer: "behavior",
          source: "validator",
          evaluator: "test",
          evaluatorVersion: "v1",
          report: "Checked a different artifact version",
          artifacts: [{ ...evidence, revision: "2" }],
        }),
      ).toThrow("pin the submitted artifact revisions");
  },
);

it.each([{ artifacts: [] }, { artifacts: [{ id: "sx:a/evidence@1", revision: "1" }] }])(
  "fails closed without a trusted evidence provider (artifacts=$artifacts)",
  async ({ artifacts }) => {
    const { services } = await fixture(async () => {
      await expect(tool(services, "work", { action: "submit", artifacts })).rejects.toThrow(
        /provider.*configured|configured.*provider/iu,
      );
    });
    await services.runWork("first", signal());
    expect(services.work.snapshot("cycle").reservations[0]?.artifacts).toEqual([]);
  },
);

it.each(["missing-base-grant", "missing-provider-grant", "unsupported-scheme"] as const)(
  "rejects %s before invoking any provider method",
  async (failure) => {
    const evidence = {
      id: failure === "unsupported-scheme" ? "other:a/evidence@1" : "sx:a/evidence@1",
      revision: "1",
    };
    const referenceProvider = {
      scheme: "sx:",
      requiredPermissions:
        failure === "missing-base-grant" ? [] : (["science.read", "science.write"] as const),
      resolve: vi.fn(() => ({ id: "sx:a/evidence", exactId: evidence.id, revision: "1" })),
      checkResource: vi.fn(() => undefined),
    };
    const { services } = await fixture(async () => {
      await expect(
        tool(services, "work", { action: "submit", artifacts: [evidence] }),
      ).rejects.toThrow(failure === "unsupported-scheme" ? /scheme|supported/iu : /permission/iu);
    }, referenceProvider);
    const removed = failure === "missing-base-grant" ? "science.read" : "science.write";
    if (failure !== "unsupported-scheme") {
      const { policy } = services.settings.read();
      services.updatePolicy({
        ...policy,
        tools: policy.tools.filter((permission) => permission !== removed),
      });
    }
    await services.runWork("first", signal());
    expect(referenceProvider.resolve).not.toHaveBeenCalled();
    expect(referenceProvider.checkResource).not.toHaveBeenCalled();
    expect(services.work.snapshot("cycle").reservations[0]?.artifacts).toEqual([]);
  },
);

it.each(["exact-id", "revision", "exception"] as const)(
  "rejects provider %s failure without partially submitting evidence",
  async (failure) => {
    const artifacts = [
      { id: "sx:a/first@1", revision: "1" },
      { id: "sx:a/second@1", revision: "1" },
    ];
    const providerError = new Error("Trusted reference store is unavailable.");
    const referenceProvider = {
      scheme: "sx:",
      requiredPermissions: ["science.read"] as const,
      resolve: vi.fn((id: string) => {
        if (id === artifacts[0]?.id) return { id: "sx:a/first", exactId: id, revision: "1" };
        if (failure === "exception") throw providerError;
        return {
          id: "sx:a/second",
          exactId: failure === "exact-id" ? "sx:a/different@1" : id,
          revision: failure === "revision" ? "2" : "1",
        };
      }),
      checkResource: vi.fn(() => undefined),
    };
    const { services } = await fixture(async () => {
      const submission = tool(services, "work", { action: "submit", artifacts });
      if (failure === "exception") await expect(submission).rejects.toBe(providerError);
      else await expect(submission).rejects.toThrow(/exact.*matching revision/iu);
    }, referenceProvider);
    await services.runWork("first", signal());
    expect(referenceProvider.resolve).toHaveBeenCalledTimes(2);
    expect(referenceProvider.checkResource).not.toHaveBeenCalled();
    expect(services.work.snapshot("cycle").reservations[0]?.artifacts).toEqual([]);
  },
);

it("submits exact evidence from a custom provider scheme when every required grant is present", async () => {
  const evidence = { id: "bio:dataset/evidence@v3", revision: "v3" };
  const referenceProvider = {
    scheme: "bio:",
    requiredPermissions: ["memory.read", "science.write"] as const,
    resolve: vi.fn(() => ({ id: "bio:dataset/evidence", exactId: evidence.id, revision: "v3" })),
    checkResource: vi.fn(() => undefined),
  };
  const { services } = await fixture(async () => {
    await expect(
      tool(services, "work", { action: "submit", artifacts: [evidence] }),
    ).resolves.toMatchObject({ artifacts: [evidence] });
  }, referenceProvider);
  await services.runWork("first", signal());
  expect(referenceProvider.resolve).toHaveBeenCalledExactlyOnceWith(evidence.id);
  expect(referenceProvider.checkResource).not.toHaveBeenCalled();
  expect(services.work.snapshot("cycle").reservations[0]?.artifacts).toEqual([evidence]);
});
