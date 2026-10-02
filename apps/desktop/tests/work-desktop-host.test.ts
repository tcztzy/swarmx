import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { afterEach, expect, it, vi } from "vitest";
import { HARNESS_CAPABILITIES, type NativeAgent } from "../src/agents/types.js";
import { HostOperations } from "../src/host/operations.js";
import { ProductServices } from "../src/host/product-services.js";
import type { SwarmXHost } from "../src/host/server.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
const cycle = {
  id: "cycle",
  project: "analysis",
  budgetUsd: 5,
  concurrency: 2,
  configurations: [{ id: "local", harness: "codex", model: "fixture" }],
};
const item = {
  id: "first",
  cycleId: "cycle",
  goal: "Check the provided data",
  criteria: "Match the independent count",
  criteriaVersion: "v1",
  taskClass: "analysis",
  runtime: { budgetUsd: 1 },
};

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-work-desktop-"));
  cleanups.push(() => rm(root, { recursive: true, force: true }));
  const options = { cwd: root, productHome: join(root, "home") };
  const products = await ProductServices.create(options);
  products.settings.writeMemory({ autoReview: false });
  let sessions = 0;
  const native: NativeAgent = {
    name: "desktop local fixture",
    capabilities: HARNESS_CAPABILITIES.codex,
    create: vi.fn(async () => `codex:${++sessions}`),
    list: async () => [],
    read: async () => {},
    models: vi.fn(async () => ({
      models: [{ id: "fixture", name: "Fixture", efforts: [] }],
      current: {},
    })),
    start: vi.fn(async (_id, _prompt, observer) => {
      observer.text("answer", "Observed count: 3");
      observer.raw(
        { type: "fixture-usage" },
        {
          "swarmx.usage.cost_usd": 0.4,
          "swarmx.usage.cost_source": "native-estimate",
          "swarmx.usage.coverage": "complete",
          "swarmx.usage.basis": "Synthetic local fixture, no paid call",
        },
      );
      return { stopReason: "end_turn" };
    }),
    interrupt: vi.fn(async () => {}),
    steer: async () => {},
    dispose: async () => {},
  };
  await products.attachAgents("http://localhost", native, "codex");
  const shutdown = new AbortController();
  const host: SwarmXHost = {
    products,
    signal: shutdown.signal,
    origin: "http://localhost",
    token: "fixture",
    async dispose() {
      shutdown.abort(new Error("Fixture Host closed"));
      await products.dispose();
    },
  };
  cleanups.push(() => host.dispose());
  const operations = new HostOperations(host);
  const seed = async () => {
    await operations.workCommand({ action: "createCycle", request: cycle });
    await operations.workCommand({ action: "createItem", request: item });
  };
  return { products, native, options, host, shutdown, operations, seed };
}

it("validates desktop commands and restores goals, budgets and acceptance without dispatch on reopen", async () => {
  const { products, native, options, host, operations, seed } = await fixture();
  expect(await operations.workRead()).toEqual({
    cycles: [],
    snapshot: null,
    activeWorkIds: [],
    interactions: [],
  });
  await expect(operations.workRead({ cycleId: "missing" })).rejects.toThrow("Unknown work cycle");
  await expect(operations.workRead({ directory: "/other" })).rejects.toThrow();
  for (const invalid of [
    { action: "createCycle", request: { ...cycle, budgetUsd: -1 } },
    { action: "createCycle", request: { ...cycle, budgetUsd: Number.NaN } },
    { action: "createCycle", request: { ...cycle, directory: "/other" } },
    { action: "start", workId: "first", bypass: true },
  ])
    await expect(operations.workCommand(invalid)).rejects.toThrow();
  expect(native.start).not.toHaveBeenCalled();
  await seed();
  await operations.workCommand({
    action: "setBudget",
    request: { cycleId: "cycle", expectedBudgetUsd: 5, budgetUsd: 7 },
  });
  await expect(
    operations.workCommand({
      action: "setBudget",
      request: { cycleId: "cycle", expectedBudgetUsd: 5, budgetUsd: 8 },
    }),
  ).rejects.toThrow("budget changed");
  const ran = await operations.workCommand({ action: "start", workId: "first" });
  const attempt = ran.snapshot?.reservations[0];
  if (!attempt) throw new Error("Missing execution attempt");
  expect(ran.snapshot?.items[0]?.state).toBe("awaiting-acceptance");
  const request = {
    id: "user-feedback",
    attemptId: attempt.id,
    criteriaVersion: "v1",
    verdict: "passed",
    accepted: true,
    fraction: 1,
    report: "I checked the provided data and its count.",
  };
  for (const forged of [
    { source: "validator" },
    { evaluator: "trusted-validator" },
    { layer: "behavior" },
    { evaluatorVersion: "v2" },
  ])
    await expect(
      operations.workCommand({ action: "accept", request: { ...request, ...forged } }),
    ).rejects.toThrow();
  await expect(
    operations.workCommand({ action: "accept", request: { ...request, criteriaVersion: "stale" } }),
  ).rejects.toThrow("criteria revision");
  await expect(
    operations.workCommand({
      action: "accept",
      request: { ...request, artifacts: [{ id: "not-submitted", revision: "v1" }] },
    }),
  ).rejects.toThrow("submitted artifact revisions");
  const accepted = await operations.workCommand({ action: "accept", request });
  expect(accepted.snapshot?.items[0]?.state).toBe("accepted");
  expect(accepted.snapshot?.feedback[0]).toMatchObject({
    source: "user",
    layer: "user",
    evaluator: "desktop-user",
    evaluatorVersion: "v1",
  });
  await expect(
    operations.workCommand({
      action: "setBudget",
      request: { cycleId: "cycle", expectedBudgetUsd: 7, budgetUsd: 0.3 },
    }),
  ).rejects.toThrow("spent and reserved");
  await host.dispose();
  const reopened = await ProductServices.create(options);
  const reopenedHost = {
    ...host,
    products: reopened,
    signal: new AbortController().signal,
    dispose: () => reopened.dispose(),
  };
  cleanups.push(() => reopened.dispose());
  const restored = await new HostOperations(reopenedHost).workRead({ cycleId: "cycle" });
  expect(restored).toEqual(accepted);
  expect(native.start).toHaveBeenCalledTimes(1);
  expect(products.options.cwd).toBe(reopened.options.cwd);
});

it("pins corrections and revised criteria and keeps uncertain execution separate from its invoice", async () => {
  const { products, operations, seed } = await fixture();
  await seed();
  const reserved = products.work.reserve("first", () => true).reservation;
  if (!reserved) throw new Error("Missing reservation");
  products.work.finish(reserved.id);
  const budget = {
    action: "setBudget",
    request: { cycleId: "cycle", expectedBudgetUsd: 5, budgetUsd: 6 },
  };
  const revise = {
    action: "revise",
    request: {
      id: "first",
      expectedCriteriaVersion: "v1",
      criteriaVersion: "v2",
      goal: "Check updated data",
      criteria: "Use the corrected inclusion rule",
    },
  };
  await expect(operations.workCommand(budget)).rejects.toThrow("unfinished executions");
  await expect(operations.workCommand(revise)).rejects.toThrow("Stop running work");
  await operations.workCommand({
    action: "reconcileCharge",
    request: {
      id: "invoice",
      reservationId: reserved.id,
      costUsd: 0.2,
      source: "invoice",
      reference: "Fixture invoice for stopped execution",
    },
  });
  await expect(operations.workCommand(budget)).rejects.toThrow("unfinished executions");
  await operations.workCommand({
    action: "reconcileOutcome",
    request: {
      reservationId: reserved.id,
      outcome: "cancelled",
      reference: "Operator confirmed no execution remains",
    },
  });
  const accepted = await operations.workCommand({
    action: "accept",
    request: {
      id: "limited-result",
      attemptId: reserved.id,
      criteriaVersion: "v1",
      verdict: "insufficient-evidence",
      accepted: true,
      fraction: 1,
      report: "The requested deliverable was a justified limitation report.",
    },
  });
  expect(accepted.snapshot?.items[0]?.state).toBe("accepted");
  const correction = {
    id: "correction",
    attemptId: reserved.id,
    criteriaVersion: "v1",
    verdict: "failed",
    accepted: false,
    fraction: 0,
    report: "The limitation report omitted required evidence.",
  };
  await expect(operations.workCommand({ action: "accept", request: correction })).rejects.toThrow(
    "supersede",
  );
  await operations.workCommand({
    action: "accept",
    request: { ...correction, supersedes: "limited-result" },
  });
  await expect(
    operations.workCommand({
      action: "revise",
      request: { ...revise.request, expectedCriteriaVersion: "old" },
    }),
  ).rejects.toThrow("criteria revision changed");
  const revised = await operations.workCommand(revise);
  expect(revised.snapshot?.items[0]).toMatchObject({
    state: "queued",
    criteriaVersion: "v2",
    firstAcceptedAt: expect.any(String),
  });
  expect(revised.snapshot?.feedback).toHaveLength(2);
  await operations.workCommand(budget);
  expect((await operations.workRead()).snapshot?.balance).toMatchObject({
    budgetUsd: 6,
    spentUsd: 0.2,
    heldUsd: 0,
  });
});

it("rejects double starts and keeps a stopped run active until the native execution finishes", async () => {
  const { native, operations, seed } = await fixture();
  await seed();
  const entered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  vi.mocked(native.start).mockImplementation(async () => {
    entered.resolve();
    await release.promise;
    return { stopReason: "cancelled" };
  });
  const running = operations.workCommand({ action: "start", workId: "first" });
  try {
    await entered.promise;
    expect((await operations.workRead()).activeWorkIds).toEqual(["first"]);
    await expect(operations.workCommand({ action: "start", workId: "first" })).rejects.toThrow(
      "already active",
    );
    await expect(
      operations.workCommand({
        action: "setBudget",
        request: { cycleId: "cycle", expectedBudgetUsd: 5, budgetUsd: 6 },
      }),
    ).rejects.toThrow("Stop active work");
    await expect(
      operations.workCommand({
        action: "revise",
        request: {
          id: "first",
          expectedCriteriaVersion: "v1",
          criteriaVersion: "v2",
          goal: "Changed",
          criteria: "Changed",
        },
      }),
    ).rejects.toThrow("Stop active work");
    const stopping = await operations.workCommand({ action: "stop", workId: "first" });
    expect(stopping.activeWorkIds).toEqual(["first"]);
    await vi.waitFor(() => expect(native.interrupt).toHaveBeenCalledTimes(1));
  } finally {
    release.resolve();
    await running;
  }
  const stopped = await operations.workRead();
  expect(stopped.activeWorkIds).toEqual([]);
  expect(stopped.snapshot?.executions[0]?.outcome).toBe("cancelled");
  expect(stopped.snapshot?.feedback).toEqual([]);
  expect(native.start).toHaveBeenCalledTimes(1);
});

it("does not dispatch after Stop returns while the native model catalog is still loading", async () => {
  const { native, operations, seed } = await fixture();
  await seed();
  const entered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  vi.mocked(native.models).mockImplementation(async () => {
    entered.resolve();
    await release.promise;
    return { models: [{ id: "fixture", name: "Fixture", efforts: [] }], current: {} };
  });
  const running = operations.workCommand({ action: "start", workId: "first" });
  try {
    await entered.promise;
    const stopped = await operations.workCommand({ action: "stop", workId: "first" });
    expect(stopped.activeWorkIds).toEqual(["first"]);
    expect(native.start).not.toHaveBeenCalled();
  } finally {
    release.resolve();
    await running;
  }
  expect(native.start).not.toHaveBeenCalled();
  expect((await operations.workRead()).snapshot?.reservations[0]).toMatchObject({
    outcome: "cancelled-before-dispatch",
    runIds: [],
    finishedAt: expect.any(String),
  });
});

it.each(["answer", "cancel", "stop", "shutdown"] as const)(
  "routes native confirmation %s without persisting sensitive answers or allowing stale responses",
  async (action) => {
    const { products, native, shutdown, operations, seed } = await fixture();
    await seed();
    const release = Promise.withResolvers<void>();
    const answered = Promise.withResolvers<unknown>();
    const effect = vi.fn();
    const request = {
      id: "native-question",
      title: "Enter the private response",
      schema: { type: "string" },
      sensitive: true,
    };
    vi.mocked(native.start).mockImplementation(async (_id, _prompt, observer) => {
      const answer = await observer.interact(request);
      answered.resolve(answer);
      if (answer !== undefined) effect(answer);
      await release.promise;
      return { stopReason: answer === undefined ? "cancelled" : "end_turn" };
    });
    const running = operations.workCommand({ action: "start", workId: "first" }).then(
      (value) => ({ value, error: undefined }),
      (error: unknown) => ({ value: undefined, error }),
    );
    const interactionId = `codex:1:${request.id}`;
    let asserted = false;
    try {
      await vi.waitFor(async () =>
        expect((await operations.workRead()).interactions).toEqual([
          {
            workId: "first",
            id: interactionId,
            title: `codex · codex:1 — ${request.title}`,
            schema: request.schema,
          },
        ]),
      );
      await expect(
        operations.workCommand({
          action: "respond",
          workId: "first",
          interactionId: "stale",
          answer: "wrong",
        }),
      ).rejects.toThrow("no longer pending");
      if (action === "shutdown") shutdown.abort(new Error("Fixture Host closed"));
      else if (action === "stop") await operations.workCommand({ action: "stop", workId: "first" });
      else
        await operations.workCommand({
          action: "respond",
          workId: "first",
          interactionId,
          ...(action === "cancel" ? { cancel: true } : { answer: "fixture-private-response" }),
        });
      expect(await answered.promise).toBe(
        action === "answer" ? "fixture-private-response" : undefined,
      );
      await expect(
        operations.workCommand({
          action: "respond",
          workId: "first",
          interactionId,
          answer: "late",
        }),
      ).rejects.toThrow();
      if (action !== "shutdown") expect((await operations.workRead()).interactions).toEqual([]);
      if (action === "answer")
        expect(effect).toHaveBeenCalledExactlyOnceWith("fixture-private-response");
      else expect(effect).not.toHaveBeenCalled();
      asserted = true;
    } finally {
      if (!asserted) shutdown.abort(new Error("Test cleanup"));
      release.resolve();
      const result = await running;
      if (asserted) {
        if (action === "shutdown") expect(String(result.error)).toContain("Fixture Host closed");
        else expect(result.error).toBeUndefined();
      }
    }
    expect(JSON.stringify(products.work.records())).not.toContain("fixture-private-response");
    expect(
      products.work
        .records()
        .some(
          ({ event }) =>
            event.type === EventType.CUSTOM &&
            event.name === "swarmx.interaction.answered" &&
            event.value.redacted === true,
        ),
    ).toBe(true);
    expect(native.start).toHaveBeenCalledTimes(1);
  },
);
