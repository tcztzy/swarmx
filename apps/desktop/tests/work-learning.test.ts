import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { afterEach, expect, it, vi } from "vitest";
import { loadAgent } from "../src/agent.js";
import { HARNESS_CAPABILITIES, type NativeAgent } from "../src/agents/types.js";
import { ProductServices } from "../src/host/product-services.js";

vi.mock("../src/agent.js", async (original) => ({
  ...(await original<typeof import("../src/agent.js")>()),
  loadAgent: vi.fn(),
}));

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
  vi.clearAllMocks();
});

it("charges restricted background reviews once and uses acceptance arriving after review for the next work item", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-work-learning-"));
  const services = await ProductServices.create({ cwd: root, productHome: join(root, "home") });
  cleanups.push(async () => {
    await services.dispose();
    await rm(root, { recursive: true, force: true });
  });
  services.settings.writeMemory({
    ...services.settings.readMemory(),
    autoReview: true,
    reviewInterval: 1,
    reviewHarness: "codex",
  });
  const configurations = [
    { id: "cheap", harness: "codex", model: "cheap" },
    { id: "strong", harness: "codex", model: "strong" },
  ];
  let session = 0;
  const agent: NativeAgent = {
    name: "deterministic work provider",
    capabilities: HARNESS_CAPABILITIES.codex,
    create: async () => `codex:main-${++session}`,
    list: async () => [],
    read: async () => {},
    models: async () => ({
      models: configurations.map(({ model }) => ({ id: model, name: model, efforts: [] })),
      current: {},
    }),
    start: vi.fn(async (_id, _prompt, observer, selected) => {
      observer.text("answer", selected?.model === "cheap" ? "wrong result" : "correct result");
      observer.raw(
        { type: "main-usage" },
        {
          "gen_ai.usage.input_tokens": 10,
          "gen_ai.usage.output_tokens": 2,
          "swarmx.usage.cost_usd": selected?.model === "cheap" ? 0.5 : 1.5,
          "swarmx.usage.cost_source": "native-estimate",
          "swarmx.usage.coverage": "complete",
          "swarmx.usage.basis": "Offline deterministic fixture; synthetic USD; no billed call",
        },
      );
      return { stopReason: "end_turn" };
    }),
    interrupt: async () => {},
    steer: async () => {},
    dispose: async () => {},
  };
  const rawReview = {
    type: "review-usage",
    response: "review-response-identity",
    usage: { input: 8, output: 3, additionalNativeField: ["preserved"] },
  };
  const reviewer: NativeAgent = {
    ...agent,
    name: "deterministic restricted reviewer",
    capabilities: HARNESS_CAPABILITIES.codex,
    create: async () => `codex:review-${++session}`,
    start: vi.fn(async (_id, _prompt, observer) => {
      observer.raw(rawReview, {
        "gen_ai.response.model": "review-reported-model",
        "gen_ai.usage.input_tokens": 8,
        "gen_ai.usage.output_tokens": 3,
        "swarmx.usage.cost_usd": 0.25,
        "swarmx.usage.cost_source": "native-estimate",
        "swarmx.usage.coverage": "complete",
        "swarmx.usage.basis": "Offline deterministic fixture; synthetic USD; no billed call",
      });
      observer.text("review", '{"summary":"Fixture review completed.","operations":[]}');
      return { stopReason: "end_turn" };
    }),
  };
  vi.mocked(loadAgent).mockResolvedValue(reviewer);
  await services.attachAgents("http://localhost", agent, "codex");
  services.work.createCycle({
    id: "cycle",
    project: "research",
    budgetUsd: 4,
    concurrency: 1,
    reviewReserveUsd: 0.5,
    configurations,
  });
  for (const id of ["first", "second"]) {
    services.work.createItem({
      id,
      cycleId: "cycle",
      goal: `Analyze ${id}`,
      taskClass: "analysis",
      criteria: "Match the independent computation",
      criteriaVersion: "v1",
      runtime: { budgetUsd: 1 },
    });
  }

  const first = await services.runWork("first", new AbortController().signal);
  expect(first.reservation?.configuration?.id).toBe("cheap");
  await vi.waitFor(async () =>
    expect((await services.learning.status()).review.state).toBe("completed"),
  );
  expect(reviewer.start).toHaveBeenCalledTimes(1);
  const firstSnapshot = services.work.snapshot("cycle");
  expect(firstSnapshot.reservations).toHaveLength(2);
  expect(firstSnapshot.balance).toMatchObject({ spentUsd: 0.75, heldUsd: 0, remainingUsd: 3.25 });
  expect(firstSnapshot.feedback).toEqual([]);
  expect(services.work.item("first").state).toBe("awaiting-acceptance");
  const execution = firstSnapshot.reservations.find(({ purpose }) => purpose === "execution");
  const review = firstSnapshot.reservations.find(({ purpose }) => purpose === "memory-review");
  expect(review).toMatchObject({
    workId: "first",
    cycleId: "cycle",
    costUsd: 0.25,
    state: "settled",
  });
  expect(review?.runIds).toHaveLength(1);
  expect(review?.runIds[0]).not.toBe(execution?.runIds[0]);
  const records = services.work.records();
  const reviewStart = records.find(
    ({ runId, event }) => runId === review?.runIds[0] && event.type === EventType.RUN_STARTED,
  );
  expect(reviewStart?.attributes).toMatchObject({
    "swarmx.work.reservation_id": review?.id,
    "swarmx.work.item_id": "first",
    "swarmx.work.cycle_id": "cycle",
    "swarmx.memory.review.source_run_ids": JSON.stringify(execution?.runIds),
    "swarmx.memory.review_eligible": false,
    "swarmx.execution.purpose": "memory-review",
  });
  expect(records.find(({ id }) => id === reviewStart?.causedBy)?.event).toMatchObject({
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.started",
  });
  expect(
    records.find(({ runId, event }) => runId === review?.runIds[0] && event.type === EventType.RAW)
      ?.event,
  ).toMatchObject({ event: rawReview });
  expect(services.journal.pendingLearningRuns()).toEqual([]);
  const restricted = vi.mocked(loadAgent).mock.calls[0]?.[1];
  expect(restricted?.reviewOnly).toBe(true);
  expect(restricted?.executionPolicy?.()).toMatchObject({ tools: [], delegation: false });

  services.work.accept({
    id: "late-content-feedback",
    attemptId: execution?.id,
    criteriaVersion: "v1",
    verdict: "failed",
    accepted: false,
    fraction: 0,
    layer: "behavior",
    source: "validator",
    evaluator: "independent-computation",
    evaluatorVersion: "v1",
    report: "Returned value differs from the expected value",
  });
  await vi.waitFor(async () => {
    expect(reviewer.start).toHaveBeenCalledTimes(2);
    expect((await services.learning.status()).review.state).toBe("completed");
  });
  const feedbackReview = vi.mocked(reviewer.start).mock.calls[1]?.[1];
  expect(feedbackReview).toContain("late-content-feedback");
  expect(feedbackReview).toContain("Returned value differs from the expected value");
  const second = await services.runWork("second", new AbortController().signal);
  expect(second.reservation?.configuration?.id).toBe("strong");
  expect(
    second.decision.evidence.find(({ configurationId }) => configurationId === "cheap"),
  ).toMatchObject({ samples: 1, accepted: 0, feedbackIds: ["late-content-feedback"] });
  await vi.waitFor(async () => {
    expect(reviewer.start).toHaveBeenCalledTimes(3);
    expect((await services.learning.status()).review.state).toBe("completed");
  });
  services.work.reconcile();
  services.work.reconcile();
  services.learning.resume();
  await vi.waitFor(async () =>
    expect((await services.learning.status()).review.state).toBe("completed"),
  );
  const final = services.work.snapshot("cycle");
  expect(final.reservations).toHaveLength(5);
  expect(final.reservations.filter(({ purpose }) => purpose === "memory-review")).toHaveLength(3);
  expect(
    final.reservations.every(({ runIds, state }) => runIds.length === 1 && state === "settled"),
  ).toBe(true);
  expect(final.balance).toMatchObject({ spentUsd: 2.75, heldUsd: 0, remainingUsd: 1.25 });
  expect(reviewer.start).toHaveBeenCalledTimes(3);
  expect(agent.start).toHaveBeenCalledTimes(2);
  expect(services.journal.pendingLearningRuns()).toEqual([]);
});
