import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { afterEach, expect, it, vi } from "vitest";
import type { ExecutionContext } from "../src/host/execution-journal.js";
import { AgentMemory, type MemoryReviewer } from "../src/host/memory.js";
import { ProductServices } from "../src/host/product-services.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
const unchanged =
  '{"summary":"Preserve the corrected acceptance evidence without unsupported generalization.","operations":[]}';
const permissions = {
  tools: ["memory.read", "memory.write"] as ("memory.read" | "memory.write")[],
  harnesses: { codex: ["permitted-reviewer"] },
  delegation: true,
};

async function fixture(reviewer = vi.fn<MemoryReviewer>().mockResolvedValue(unchanged)) {
  const root = await mkdtemp(join(tmpdir(), "swarmx-feedback-review-"));
  const options = { cwd: root, productHome: join(root, "home") };
  const products = await ProductServices.create(options);
  products.settings.writeMemory({ reviewInterval: 100 });
  const memory = new AgentMemory(
    options,
    products.memory,
    products.journal,
    products.settings,
    reviewer,
  );
  cleanups.push(async () => {
    await memory.close();
    await products.dispose();
    await rm(root, { recursive: true, force: true });
  });
  const scope: ExecutionContext = {
    sessionId: "codex:original",
    runId: randomUUID(),
    causedBy: null,
    attributes: {
      "swarmx.harness.name": "codex",
      "swarmx.memory.review_eligible": true,
      "swarmx.memory.review_permissions": JSON.stringify(permissions),
      "swarmx.work.item_id": "analysis",
      "swarmx.work.cycle_id": "cycle",
    },
  };
  const started = products.journal.append(scope, {
    type: EventType.RUN_STARTED,
    threadId: scope.sessionId ?? "",
    runId: scope.runId,
    input: {
      threadId: scope.sessionId ?? "",
      runId: scope.runId,
      messages: [
        {
          id: randomUUID(),
          role: "user",
          content:
            "Estimate the included sample mean. Acceptance (v1): Match current included rows.",
        },
      ],
      tools: [],
      context: [],
      state: {},
      forwardedProps: {},
    },
  });
  const terminal = products.journal.append(scope, {
    type: EventType.RUN_FINISHED,
    threadId: scope.sessionId ?? "",
    runId: scope.runId,
    result: { stopReason: "end_turn" },
  });
  products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.finished",
    value: { state: "completed", terminalIds: [terminal.id], message: "0", sessionId: null },
  });
  const feedback = (
    verdict: "passed" | "failed",
    supersedes?: string,
    eligible = true,
    report = "Independent content check corrected the earlier result.",
    artifacts: { id: string; revision: string }[] = [],
  ) => {
    const value = {
      id: randomUUID(),
      attemptId: "attempt",
      criteriaVersion: "v1",
      verdict,
      accepted: verdict === "passed",
      fraction: verdict === "passed" ? 1 : 0,
      layer: "behavior",
      source: "validator",
      evaluator: "numeric-check",
      evaluatorVersion: "v1",
      report,
      artifacts,
      intervention: "none",
      recordedAt: new Date().toISOString(),
      ...(supersedes ? { supersedes } : {}),
    };
    return products.journal.append(
      {
        ...scope,
        causedBy: started.id,
        attributes: { ...scope.attributes, "swarmx.memory.review_eligible": eligible },
      },
      {
        type: EventType.CUSTOM,
        name: "swarmx.work.feedback",
        value: {
          workId: "analysis",
          cycleId: "cycle",
          feedback: value,
          attemptId: "attempt",
          runIds: [scope.runId],
          configuration: null,
        },
      },
    );
  };
  return { options, products, memory, reviewer, feedback, scope };
}

it("reviews delayed acceptance and a later correction after the original execution was acknowledged", async () => {
  const { products, memory, reviewer, feedback } = await fixture();
  const first = feedback("passed");
  expect(products.journal.pendingLearningRuns()).toEqual([first]);
  memory.resume();
  await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(1));
  await vi.waitFor(() => expect(products.journal.pendingLearningRuns()).toEqual([]));
  expect(reviewer.mock.calls[0]?.[0]).toContain(first.id);
  expect(reviewer.mock.calls[0]?.[0]).toContain("Estimate the included sample mean");
  const correction = feedback(
    "failed",
    first.event.type === EventType.CUSTOM ? first.event.value.feedback.id : "",
  );
  memory.resume();
  await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(2));
  await vi.waitFor(() => expect(products.journal.pendingLearningRuns()).toEqual([]));
  expect(reviewer.mock.calls[1]?.[0]).toContain(correction.id);
  expect(reviewer.mock.calls[1]?.[3]).toEqual(permissions);
  await memory.close();
  expect(reviewer).toHaveBeenCalledTimes(2);
});

it("does not acknowledge correction feedback arriving while an earlier snapshot is being reviewed", async () => {
  const gate = Promise.withResolvers<string>();
  const reviewer = vi
    .fn<MemoryReviewer>()
    .mockReturnValueOnce(gate.promise)
    .mockResolvedValue(unchanged);
  const { products, memory, feedback } = await fixture(reviewer);
  const first = feedback("failed");
  memory.resume();
  await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(1));
  const second = feedback("passed");
  memory.resume();
  expect(products.journal.pendingLearningRuns()).toEqual([first, second]);
  expect(reviewer.mock.calls[0]?.[0]).not.toContain(second.id);
  gate.resolve(unchanged);
  await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(2));
  await vi.waitFor(() => expect(products.journal.pendingLearningRuns()).toEqual([]));
  const receipts = products.journal
    .read({ limit: 1000 })
    .events.filter(
      ({ event }) =>
        event.type === EventType.CUSTOM &&
        event.name === "swarmx.memory.review.finished" &&
        event.value.jobId,
    );
  expect(
    receipts.map(({ event }) => event.type === EventType.CUSTOM && event.value.terminalIds),
  ).toEqual([[first.id], [second.id]]);
  expect(reviewer.mock.calls[1]?.[0]).toContain(second.id);
});

it("keeps feedback arriving during snapshot preparation for the next exact batch", async () => {
  const { products, memory, reviewer, feedback } = await fixture();
  const entered = Promise.withResolvers<void>();
  const gate = Promise.withResolvers<void>();
  vi.spyOn(memory.resources, "snapshot").mockImplementationOnce(async () => {
    entered.resolve();
    await gate.promise;
    return [];
  });
  const first = feedback("failed");
  memory.resume();
  await entered.promise;
  const second = feedback("passed");
  memory.resume();
  gate.resolve();
  await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(2));
  await vi.waitFor(() => expect(products.journal.pendingLearningRuns()).toEqual([]));
  expect(reviewer.mock.calls[0]?.[0]).toContain(first.id);
  expect(reviewer.mock.calls[0]?.[0]).not.toContain(second.id);
  expect(reviewer.mock.calls[1]?.[0]).toContain(second.id);
});

it("saves feedback-grounded experience that later selection can retrieve", async () => {
  const { products, memory, reviewer, feedback } = await fixture();
  const source = feedback("failed");
  const completed = Promise.withResolvers<void>();
  const append = products.journal.append.bind(products.journal);
  vi.spyOn(products.journal, "append").mockImplementation((...args) => {
    const record = append(...args);
    if (
      record.event.type === EventType.CUSTOM &&
      record.event.name === "swarmx.memory.review.finished" &&
      record.event.value.jobId
    ) {
      if (record.event.value.state === "failed") {
        completed.reject(new Error(String(record.event.value.message)));
      } else if (record.event.value.terminalIds?.includes(source.id)) {
        completed.resolve();
      }
    }
    return record;
  });
  reviewer.mockResolvedValue(
    JSON.stringify({
      summary: "Retain one independently checked failure.",
      operations: [
        {
          action: "create_memory",
          request: {
            title: "Included Mean Acceptance",
            description: "One checked analysis result.",
            type: "Finding",
            tags: ["agent-selection"],
            body: "The supplied analysis failed its included-row content check. This observation is limited to the recorded task and evaluator revision.",
            evaluation: {
              kind: "observation",
              task: "Included-row mean",
              criteria: "Match the current included rows.",
              evidence: [`urn:swarmx:execution:${source.id}`],
              limitations: "One task; no general model ranking.",
            },
          },
        },
      ],
    }),
  );
  memory.resume();
  await completed.promise;
  expect(products.journal.pendingLearningRuns()).toEqual([]);
  const selection = await memory.selection(["Included Mean"], new AbortController().signal);
  expect(selection.loaded.flatMap(({ concepts }) => concepts)).toContainEqual(
    expect.objectContaining({
      metadata: expect.objectContaining({
        swarmx_evaluation: expect.objectContaining({
          evidence: [`urn:swarmx:execution:${source.id}`],
        }),
      }),
      evaluation: expect.objectContaining({ status: "referenced" }),
    }),
  );
});

it("retains feedback through disabled learning and restart without widening original grants", async () => {
  const first = await fixture();
  first.products.settings.writeMemory({ enabled: false, autoReview: false, reviewInterval: 100 });
  const source = first.feedback("failed");
  first.memory.resume();
  expect(first.reviewer).not.toHaveBeenCalled();
  await first.memory.close();
  await first.products.dispose();
  const products = await ProductServices.create(first.options);
  const reviewer = vi.fn<MemoryReviewer>().mockResolvedValue(unchanged);
  const memory = new AgentMemory(
    first.options,
    products.memory,
    products.journal,
    products.settings,
    reviewer,
  );
  cleanups.push(async () => {
    await memory.close();
    await products.dispose();
  });
  memory.resume();
  expect(reviewer).not.toHaveBeenCalled();
  expect(products.journal.pendingLearningRuns().map(({ id }) => id)).toEqual([source.id]);
  products.settings.writeMemory({ enabled: true, autoReview: true, reviewInterval: 100 });
  memory.resume();
  await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(1));
  await vi.waitFor(() => expect(products.journal.pendingLearningRuns()).toEqual([]));
  expect(reviewer.mock.calls[0]?.[3]).toEqual(permissions);
});

it("does not turn ineligible feedback or review output into new learning work", async () => {
  const { products, memory, reviewer, feedback, scope } = await fixture();
  feedback("failed", undefined, false);
  products.journal.append(scope, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.response",
    value: { text: "A review may not manufacture acceptance." },
  });
  products.journal.append(scope, {
    type: EventType.RAW,
    event: { type: "CUSTOM", name: "swarmx.work.feedback", value: { accepted: true } },
  });
  memory.resume();
  await memory.close();
  expect(products.journal.pendingLearningRuns()).toEqual([]);
  expect(reviewer).not.toHaveBeenCalled();
});

it("rejects Agent-originated acceptance and publication actions at the Host boundary", async () => {
  const { products, scope } = await fixture();
  const accept = vi.spyOn(products.work, "accept");
  await products.journal.scope.run(scope, async () => {
    for (const action of [
      "accept",
      "flushFeedback",
      "createCycle",
      "reconcileCharge",
      "reconcileOutcome",
    ])
      await expect(
        products.callTool(
          "work",
          { action },
          {
            actorId: "native",
            callId: randomUUID(),
            signal: new AbortController().signal,
          },
        ),
      ).rejects.toThrow();
  });
  expect(accept).not.toHaveBeenCalled();
  products.journal.scope.run(scope, () => {
    for (const operation of [
      () => products.work.accept({}),
      () => products.work.createCycle({}),
      () => products.work.revise({}),
      () => products.work.reconcileCharge({}),
      () => products.work.reconcileOutcome({}),
      () => products.work.flushFeedback(),
    ])
      expect(operation).toThrow("trusted Host caller");
  });
  expect(products.journal.pendingLearningRuns()).toEqual([]);
});

it("bounds feedback batches chronologically without acknowledging the remainder", async () => {
  const { products, feedback } = await fixture();
  const sources = Array.from({ length: 101 }, () => feedback("passed"));
  expect(products.journal.pendingLearningRuns()).toEqual(sources.slice(0, 100));
  products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.finished",
    value: {
      state: "completed",
      terminalIds: sources.slice(0, 100).map(({ id }) => id),
      message: "0",
      sessionId: null,
    },
  });
  expect(products.journal.pendingLearningRuns()).toEqual(sources.slice(100));
});

it("drains large valid feedback across complete bounded batches", async () => {
  const { products, memory, reviewer, feedback } = await fixture();
  const sources = Array.from({ length: 3 }, () =>
    feedback("failed", undefined, true, "x".repeat(16_000)),
  );
  memory.resume();
  await vi.waitFor(() => expect(products.journal.pendingLearningRuns()).toEqual([]));
  expect(reviewer).toHaveBeenCalledTimes(2);
  const acknowledgements = products.journal
    .read({ limit: 1000 })
    .events.flatMap(({ event }) =>
      event.type === EventType.CUSTOM &&
      event.name === "swarmx.memory.review.finished" &&
      event.value.jobId
        ? event.value.terminalIds
        : [],
    );
  expect(acknowledgements).toEqual(sources.map(({ id }) => id));
  for (const [prompt] of reviewer.mock.calls) {
    const snapshot = JSON.parse(prompt.split("\n\nSnapshot: ")[1] ?? "null");
    expect(JSON.stringify(snapshot.evidence.records).length).toBeLessThanOrEqual(40_000);
    for (const source of sources)
      if (snapshot.evidence.records.some((record: { id: string }) => record.id === source.id))
        expect(snapshot.evidence.records).toContainEqual(source);
  }
});

it("retains one individually oversized feedback source with an explicit failure", async () => {
  const { products, memory, reviewer, feedback } = await fixture();
  const source = feedback(
    "failed",
    undefined,
    true,
    "x".repeat(16_000),
    Array.from({ length: 100 }, (_, index) => ({
      id: `${index}-${"a".repeat(200)}`,
      revision: "b".repeat(64),
    })),
  );
  memory.resume();
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("failed"));
  expect(reviewer).not.toHaveBeenCalled();
  expect(products.journal.pendingLearningRuns().map(({ id }) => id)).toEqual([source.id]);
  expect((await memory.status()).review.message).toContain("omitted queued source");
});
