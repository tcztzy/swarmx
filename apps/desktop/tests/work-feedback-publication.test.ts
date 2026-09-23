import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { afterEach, expect, it, vi } from "vitest";
import { ExecutionJournal } from "../src/host/execution-journal.js";
import { WorkManager } from "../src/host/work.js";
import { policyPermissions } from "../src/permissions.js";
import { DEFAULT_POLICY } from "../src/settings.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-feedback-publication-"));
  const journal = new ExecutionJournal(join(root, "logs"), "workspace");
  const changed = vi.fn();
  const work = new WorkManager(root, "workspace", journal, changed);
  cleanups.push(async () => {
    work.close();
    journal.close();
    await rm(root, { recursive: true, force: true });
  });
  work.createCycle({
    id: "cycle",
    project: "research",
    budgetUsd: 10,
    configurations: [{ id: "route", harness: "pi", model: "fixture" }],
  });
  work.createItem({
    id: "item",
    cycleId: "cycle",
    goal: "Compute the included mean",
    criteria: "Match the independent result",
    criteriaVersion: "v1",
    taskClass: "mean",
    runtime: { budgetUsd: 1 },
  });
  const attempt = work.reserve("item", () => true).reservation;
  if (!attempt) throw new Error("Missing reservation");
  const context = {
    sessionId: "pi:session",
    runId: "run",
    causedBy: null,
    attributes: {
      ...work.attributes(attempt),
      "swarmx.memory.review_eligible": true,
      "swarmx.memory.review_permissions": JSON.stringify(policyPermissions(DEFAULT_POLICY)),
    },
  };
  const started = journal.append(context, {
    type: EventType.RUN_STARTED,
    threadId: context.sessionId,
    runId: context.runId,
  });
  journal.append(
    { ...context, causedBy: started.id },
    {
      type: EventType.RUN_FINISHED,
      threadId: context.sessionId,
      runId: context.runId,
      result: { stopReason: "end_turn" },
    },
    { "swarmx.usage.cost_usd": 0.1 },
  );
  work.finish(attempt.id);
  const feedback = {
    id: "feedback",
    attemptId: attempt.id,
    criteriaVersion: "v1",
    verdict: "failed",
    accepted: false,
    fraction: 0,
    layer: "behavior",
    source: "validator",
    evaluator: "mean-check",
    evaluatorVersion: "v1",
    report: "Incorrect included-row count",
  };
  return { root, journal, work, changed, attempt, started, feedback };
}

it("publishes immutable feedback once with original execution identity and grants", async () => {
  const { journal, work, changed, started, feedback, attempt } = await fixture();
  const saved = work.accept(feedback);
  work.accept(feedback);
  work.flushFeedback();
  const events = journal
    .read()
    .events.filter(
      ({ event }) => event.type === EventType.CUSTOM && event.name === "swarmx.work.feedback",
    );
  expect(events).toHaveLength(1);
  expect(events[0]).toMatchObject({
    sessionId: "pi:session",
    runId: "run",
    causedBy: started.id,
    attributes: { "swarmx.work.item_id": "item", "swarmx.memory.review_eligible": true },
    event: {
      type: EventType.CUSTOM,
      name: "swarmx.work.feedback",
      value: {
        workId: "item",
        cycleId: "cycle",
        attemptId: attempt.id,
        runIds: ["run"],
        feedback: saved,
      },
    },
  });
  expect(changed).toHaveBeenCalledTimes(1);
});

it("replays an acceptance saved before interrupted journal publication on restart", async () => {
  const { root, journal, work, feedback } = await fixture();
  const append = journal.append.bind(journal);
  const failure = vi
    .spyOn(journal, "append")
    .mockImplementation((context, event, attributes, key) => {
      if (event.type === EventType.CUSTOM && event.name === "swarmx.work.feedback")
        throw new Error("Interrupted publication");
      return append(context, event, attributes, key);
    });
  expect(() => work.accept(feedback)).toThrow("Interrupted publication");
  expect(work.snapshot("cycle").feedback).toHaveLength(1);
  failure.mockRestore();
  const recovered = new WorkManager(root, "workspace", journal);
  try {
    recovered.accept(feedback);
    expect(
      journal
        .read()
        .events.filter(
          ({ event }) => event.type === EventType.CUSTOM && event.name === "swarmx.work.feedback",
        ),
    ).toHaveLength(1);
  } finally {
    recovered.close();
  }
});

it("adds a new correction event without modifying the original acceptance evidence", async () => {
  const { journal, work, feedback } = await fixture();
  work.accept(feedback);
  work.accept({
    ...feedback,
    id: "correction",
    supersedes: "feedback",
    verdict: "passed",
    accepted: true,
    fraction: 1,
    report: "Correct after applying the specified inclusion rule",
  });
  expect(
    journal
      .read()
      .events.filter(
        ({ event }) => event.type === EventType.CUSTOM && event.name === "swarmx.work.feedback",
      )
      .map(({ event }) => event.type === EventType.CUSTOM && event.value.feedback.verdict),
  ).toEqual(["failed", "passed"]);
});

it("serializes identical keyed journal writes across connections and rejects changed payloads", async () => {
  const { root, journal } = await fixture();
  const other = new ExecutionJournal(join(root, "logs"), "workspace");
  const foreign = new ExecutionJournal(join(root, "logs"), "other");
  const event = {
    type: EventType.CUSTOM,
    timestamp: 1,
    name: "test.feedback",
    value: { answer: 1 },
  } as const;
  try {
    const first = journal.append(null, event, {}, "feedback-key");
    expect(other.append(null, event, {}, "feedback-key").id).toBe(first.id);
    expect(() =>
      other.append(null, { ...event, value: { answer: 2 } }, {}, "feedback-key"),
    ).toThrow("idempotency");
    expect(foreign.append(null, event, {}, "feedback-key").id).not.toBe(first.id);
  } finally {
    other.close();
    foreign.close();
  }
});
