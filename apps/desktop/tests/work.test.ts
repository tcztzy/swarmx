import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { afterEach, expect, it, vi } from "vitest";
import { HARNESS_CAPABILITIES, type NativeAgent } from "../src/agents/types.js";
import { ExecutionJournal } from "../src/host/execution-journal.js";
import { HostOperations } from "../src/host/operations.js";
import { ProductServices } from "../src/host/product-services.js";
import { WorkManager } from "../src/host/work.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
const configurations = [
  { id: "cheap", harness: "codex", model: "cheap" },
  { id: "strong", harness: "codex", model: "strong" },
];
const signal = () => new AbortController().signal;

async function fixture(budgetUsd = 10, concurrency = 3) {
  const root = await mkdtemp(join(tmpdir(), "swarmx-work-"));
  const services = await ProductServices.create({ cwd: root, productHome: join(root, "home") });
  services.settings.writeMemory({ ...services.settings.readMemory(), autoReview: false });
  let counter = 0;
  const agent: NativeAgent = {
    name: "local deterministic provider",
    capabilities: HARNESS_CAPABILITIES.codex,
    create: vi.fn(async () => `codex:${++counter}`),
    list: async () => [],
    read: async () => {},
    models: vi.fn(async () => ({
      models: configurations.map(({ model }) => ({ id: model, name: model, efforts: [] })),
      current: {},
    })),
    start: vi.fn(async (_id, _text, observer, options) => {
      observer.text("answer", options?.model === "cheap" ? "incorrect" : "correct");
      observer.raw(
        { type: "fixture-usage" },
        {
          "gen_ai.usage.input_tokens": 12,
          "gen_ai.usage.output_tokens": 3,
          "swarmx.usage.cost_usd": options?.model === "cheap" ? 0.5 : 1.5,
          "swarmx.usage.basis": "deterministic fixture, not a provider bill",
          "swarmx.usage.coverage": "complete",
          "swarmx.usage.cost_source": "native-estimate",
        },
      );
      return { stopReason: "end_turn" };
    }),
    interrupt: vi.fn(async () => {}),
    steer: async () => {},
    dispose: async () => {},
  };
  await services.attachAgents("http://localhost", agent, "codex");
  services.work.createCycle({
    id: "cycle",
    project: "research",
    budgetUsd,
    concurrency,
    reviewReserveUsd: 0.5,
    configurations,
  });
  const add = (id: string, extra = {}) =>
    services.work.createItem({
      id,
      cycleId: "cycle",
      goal: `Analyze ${id}`,
      criteria: "Answer must match independent computation",
      criteriaVersion: "v1",
      taskClass: "analysis",
      runtime: { budgetUsd: 1 },
      ...extra,
    });
  cleanups.push(async () => {
    await services.dispose();
    await rm(root, { recursive: true, force: true });
  });
  return { root, services, agent, add };
}

function feedback(attemptId: string, passed = false, extra = {}) {
  return {
    id: `feedback-${attemptId}`,
    attemptId,
    criteriaVersion: "v1",
    verdict: passed ? "passed" : "failed",
    accepted: passed,
    fraction: passed ? 1 : 0,
    layer: "behavior",
    source: "validator",
    evaluator: "independent-test",
    evaluatorVersion: "v1",
    report: passed ? "Verified expected content" : "Content differs from expected",
    ...extra,
  };
}

it("uses persisted feedback and a shared budget across two actual ProductServices executions", async () => {
  const { services, agent, add } = await fixture();
  add("first");
  add("second");
  const first = await services.runWork("first", signal());
  expect(first.reservation?.configuration?.id).toBe("cheap");
  expect(services.work.item("first").state).toBe("awaiting-acceptance");
  const firstId = first.reservation?.id ?? "";
  services.work.accept(feedback(firstId));
  const second = await services.runWork("second", signal());
  expect(second.reservation?.configuration?.id).toBe("strong");
  expect(second.decision.evidence.find((row) => row.configurationId === "cheap")).toMatchObject({
    samples: 1,
    accepted: 0,
  });
  services.work.accept(feedback(second.reservation?.id ?? "", true));
  expect(services.work.item("second")).toMatchObject({
    state: "accepted",
    firstAcceptedAt: expect.any(String),
  });
  const state = services.work.snapshot("cycle");
  expect(state.balance).toMatchObject({ spentUsd: 2, heldUsd: 0, remainingUsd: 8 });
  expect(state.reservations.every((row) => row.runIds.length === 1)).toBe(true);
  expect(agent.start).toHaveBeenCalledTimes(2);
  const prepare = services.work
    .records()
    .find(
      (record) =>
        record.event.type === EventType.TOOL_CALL_RESULT &&
        record.event.content.includes('"work":'),
    );
  expect(prepare).toBeDefined();
});

it("atomically reserves across independent database connections and cannot spend a shared balance twice", async () => {
  const { services, root, add } = await fixture(1, 4);
  add("one");
  add("two");
  const other = new WorkManager(
    join(root, "home", "work"),
    services.directoryKey,
    services.journal,
  );
  try {
    const [a, b] = await Promise.all([
      Promise.resolve().then(() => services.work.reserve("one", () => true)),
      Promise.resolve().then(() => other.reserve("two", () => true)),
    ]);
    expect([a, b].filter(({ reservation }) => reservation)).toHaveLength(1);
    expect(other.snapshot("cycle").balance).toMatchObject({ remainingUsd: 0, heldUsd: 1 });
  } finally {
    other.close();
  }
});

it("admits reservations that exactly consume a decimal USD budget", async () => {
  const { services } = await fixture();
  services.work.createCycle({
    id: "decimal",
    project: "research",
    budgetUsd: 0.3,
    concurrency: 2,
    configurations,
  });
  for (const id of ["cheap", "strong"])
    services.work.createItem({
      id,
      cycleId: "decimal",
      goal: "Analyze",
      criteria: "Check",
      criteriaVersion: "v1",
      taskClass: "analysis",
      policy: "fixed",
      fixedConfiguration: id,
      runtime: { budgetUsd: id === "cheap" ? 0.1 : 0.2 },
    });
  expect(services.work.reserve("cheap", () => true).reservation).not.toBeNull();
  expect(services.work.reserve("strong", () => true).reservation).not.toBeNull();
  expect(services.work.snapshot("decimal").balance.remainingUsd).toBe(0);
});

it("enforces concurrency before native dispatch and leaves budget blocked for unknown cost", async () => {
  const { services, agent, add } = await fixture(10, 1);
  add("one");
  add("two");
  const started = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  vi.mocked(agent.start).mockImplementation(async () => {
    started.resolve();
    await release.promise;
    return { stopReason: "end_turn" };
  });
  const first = services.runWork("one", signal());
  await started.promise;
  const second = await services.runWork("two", signal());
  expect(second.decision.action).toBe("defer");
  expect(agent.start).toHaveBeenCalledTimes(1);
  release.resolve();
  await first;
  const state = services.work.snapshot("cycle");
  expect(state.reservations[0]).toMatchObject({
    costUsd: null,
    coverage: "unknown",
    state: "uncertain",
  });
  expect(state.balance).toMatchObject({ heldUsd: 1, remainingUsd: 9 });
  await expect(services.runWork("one", signal())).rejects.toThrow("unresolved");
});

it("keeps unknown outcomes on restart and reconciles a bill idempotently without rerunning", async () => {
  const { services, root, agent, add } = await fixture();
  add("one");
  const reserved = services.work.reserve("one", () => true).reservation;
  expect(reserved).not.toBeNull();
  services.work.finish(
    reserved?.id ?? "",
    new Error("Process ended before dispatch acknowledgement"),
  );
  const reopened = new WorkManager(
    join(root, "home", "work"),
    services.directoryKey,
    services.journal,
  );
  try {
    expect(reopened.snapshot("cycle").items[0]?.state).toBe("blocked");
    expect(agent.start).not.toHaveBeenCalled();
    const bill = {
      id: "invoice-1",
      reservationId: reserved?.id,
      costUsd: 0.2,
      source: "invoice",
      reference: "Trusted reconciliation found one partial provider call",
    };
    reopened.reconcileCharge(bill);
    reopened.reconcileCharge(bill);
    reopened.reconcile();
    expect(reopened.snapshot("cycle").balance).toMatchObject({ spentUsd: 0.2, heldUsd: 0.8 });
    expect(() => reopened.reconcileCharge({ ...bill, costUsd: 0.1 })).toThrow("idempotency");
    await expect(services.runWork("one", signal())).rejects.toThrow("unresolved");
    reopened.reconcileOutcome({
      reservationId: reserved?.id,
      outcome: "cancelled",
      reference: "Operator confirmed the interrupted provider execution has stopped",
    });
    expect(reopened.snapshot("cycle").balance.heldUsd).toBe(0);
    const continued = await services.runWork("one", signal());
    expect(continued.reservation?.workId).toBe("one");
    expect(services.work.snapshot("cycle").reservations).toHaveLength(2);
  } finally {
    reopened.close();
  }
});

it("late corrected acceptance remains effective and does not manufacture runtime success", async () => {
  const { services, add } = await fixture();
  add("one");
  add("two");
  const first = await services.runWork("one", signal());
  const id = first.reservation?.id ?? "";
  services.work.accept(feedback(id, true));
  const firstAcceptedAt = services.work.item("one").firstAcceptedAt;
  expect(() => services.work.accept(feedback(id, false, { id: "corrected" }))).toThrow("supersede");
  services.work.accept(
    feedback(id, false, {
      id: "corrected",
      supersedes: `feedback-${id}`,
      intervention: "revision",
    }),
  );
  expect(services.work.item("one")).toMatchObject({
    state: "awaiting-acceptance",
    firstAcceptedAt,
  });
  const next = await services.runWork("two", signal());
  expect(next.reservation?.configuration?.id).toBe("strong");
  expect(services.work.snapshot("cycle").reservations[0]?.outcome).toBe("completed");
});

it("retains historical criteria feedback without accepting work against a revised contract", async () => {
  const { services, add } = await fixture();
  add("one");
  const attempt = await services.runWork("one", signal());
  services.work.revise({
    id: "one",
    expectedCriteriaVersion: "v1",
    criteriaVersion: "v2",
    goal: "New data",
    criteria: "Recompute new data",
  });
  services.work.accept(feedback(attempt.reservation?.id ?? "", true));
  expect(services.work.item("one").state).toBe("queued");
  expect(() =>
    services.work.revise({
      id: "one",
      expectedCriteriaVersion: "v1",
      criteriaVersion: "v3",
      goal: "stale",
      criteria: "stale",
    }),
  ).toThrow("revision changed");
});

it("accepts an evidence-insufficient deliverable only through trusted content acceptance", async () => {
  const { services, add } = await fixture();
  add("one");
  const result = await services.runWork("one", signal());
  const id = result.reservation?.id ?? "";
  expect(() => services.work.accept(feedback(id, true, { layer: "structure" }))).toThrow(
    "Structure",
  );
  services.work.accept(
    feedback(id, false, {
      accepted: true,
      verdict: "insufficient-evidence",
      fraction: 1,
      report: "Correctly withheld causal conclusion because the required control was not collected",
    }),
  );
  expect(services.work.item("one").state).toBe("accepted");
  expect(() => services.work.accept(feedback(id, true))).toThrow("idempotency");
});

it("does not label cancellation as quality failure, but retains reported incurred costs", async () => {
  const { services, agent, add } = await fixture();
  add("one");
  vi.mocked(agent.start).mockImplementation(async (_id, _text, observer) => {
    observer.raw(
      {},
      { "swarmx.usage.cost_usd": 0.1, "swarmx.usage.basis": "fixture cancellation bill" },
    );
    return { stopReason: "cancelled" };
  });
  await services.runWork("one", signal());
  const state = services.work.snapshot("cycle");
  expect(state.reservations[0]).toMatchObject({ outcome: "cancelled", costUsd: 0.1 });
  expect(state.items[0]?.state).toBe("awaiting-acceptance");
  expect(state.feedback).toEqual([]);
});

it("cancels during native catalog preparation with zero later native starts", async () => {
  const { services, agent, add } = await fixture();
  add("one");
  const paused = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  vi.mocked(agent.models).mockImplementation(async () => {
    paused.resolve();
    await release.promise;
    return { models: [{ id: "cheap", name: "cheap", efforts: [] }], current: {} };
  });
  const controller = new AbortController();
  const running = services.runWork("one", controller.signal);
  await paused.promise;
  controller.abort();
  await vi.waitFor(() => expect(agent.interrupt).toHaveBeenCalled());
  release.resolve();
  const result = await running;
  expect(result).toMatchObject({ result: { result: { stopReason: "cancelled" } } });
  expect(agent.start).not.toHaveBeenCalled();
  expect(services.work.snapshot("cycle").items[0]?.state).toBe("blocked");
});

it("accounts a shared review once and preserves pending learning when its reservation is unaffordable", async () => {
  const { services, add } = await fixture(1.1);
  add("one");
  await services.runWork("one", signal());
  const first = services.work.snapshot("cycle").reservations[0];
  const review = services.work.reserveReview(first?.runIds ?? []);
  expect(review).toMatchObject({ purpose: "memory-review", reservedUsd: 0.5 });
  expect(() => services.work.reserveReview(first?.runIds ?? [])).toThrow("affordable");
  services.work.finish(review?.id ?? "");
  const invoice = {
    id: "learning-charge",
    reservationId: review?.id,
    costUsd: 0.3,
    source: "invoice",
    reference: "Synthetic fixture review bill",
  };
  services.work.reconcileCharge(invoice);
  services.work.reconcileCharge(invoice);
  expect(services.work.snapshot("cycle").balance.spentUsd).toBeCloseTo(0.8);
});

it("shares no work or feedback with another directory, even in the same product home", async () => {
  const { root, add } = await fixture();
  add("one");
  const otherJournal = new ExecutionJournal(join(root, "home", "logs"), "other-directory");
  const other = new WorkManager(join(root, "home", "work"), "other-directory", otherJournal);
  try {
    expect(() => other.item("one")).toThrow("Unknown");
    other.createCycle({ id: "cycle", project: "other", budgetUsd: 2, configurations });
    expect(other.snapshot("cycle").items).toEqual([]);
  } finally {
    other.close();
    otherJournal.close();
  }
});

it("honors dependency, priority, deadline and current permission constraints", async () => {
  const { services, add } = await fixture();
  add("dependency");
  add("blocked", { dependencies: ["dependency"], priority: 50 });
  add("urgent", { priority: 20 });
  const operations = new HostOperations({
    products: services,
    signal: signal(),
    origin: "http://localhost",
    token: "fixture",
    dispose: () => services.dispose(),
  });
  const next = await operations.workCommand({ action: "startNext", cycleId: "cycle" });
  expect(next.snapshot?.reservations[0]?.workId).toBe("urgent");
  await expect(services.runWork("blocked", signal())).rejects.toThrow("dependencies");
  add("expired", { deadline: "2020-01-01T00:00:00.000Z" });
  expect((await services.runWork("expired", signal())).decision.action).toBe("stop");
  services.updatePolicy({ ...services.settings.read().policy, harnesses: { codex: ["strong"] } });
  const onlyStrong = await services.runWork("dependency", signal());
  expect(onlyStrong.reservation?.configuration?.id).toBe("strong");
});
