import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { afterEach, expect, it } from "vitest";
import { type ExecutionContext, ExecutionJournal } from "../src/host/execution-journal.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-learning-journal-"));
  const journal = new ExecutionJournal(root, "workspace-a");
  cleanups.push(async () => {
    journal.close();
    await rm(root, { recursive: true, force: true });
  });
  return { root, journal };
}

function context(runId: string) {
  return {
    sessionId: `session:${runId}`,
    runId,
    causedBy: null,
    attributes: {},
  } satisfies ExecutionContext;
}

function run(journal: ExecutionJournal, id: string, eligible = true, failed = false) {
  const scope = context(id);
  journal.append(
    scope,
    { type: EventType.RUN_STARTED, threadId: scope.sessionId, runId: id },
    { "swarmx.memory.review_eligible": eligible },
  );
  return journal.append(
    scope,
    failed
      ? { type: EventType.RUN_ERROR, message: "Provider unavailable" }
      : {
          type: EventType.RUN_FINISHED,
          threadId: scope.sessionId,
          runId: id,
          result: { stopReason: "end_turn" },
        },
  );
}

it("persists eligible terminal work and acknowledges only the exact successful review subset", async () => {
  const { root, journal } = await fixture();
  const first = run(journal, "first");
  run(journal, "ineligible", false);
  const failed = run(journal, "failed", true, true);
  journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.finished",
    value: { state: "failed", terminalIds: [first.id, failed.id] },
  });
  const concurrent = run(journal, "concurrent");
  expect(journal.pendingLearningRuns()).toEqual([first, failed, concurrent]);
  journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.finished",
    value: { state: "completed", terminalIds: [first.id] },
  });
  journal.close();
  const reopened = new ExecutionJournal(root, "workspace-a");
  const other = new ExecutionJournal(root, "workspace-b");
  try {
    expect(reopened.pendingLearningRuns()).toEqual([failed, concurrent]);
    expect(other.pendingLearningRuns()).toEqual([]);
    other.append(null, {
      type: EventType.CUSTOM,
      name: "swarmx.memory.review.finished",
      value: { state: "completed", terminalIds: [failed.id] },
    });
    expect(reopened.pendingLearningRuns()).toEqual([failed, concurrent]);
  } finally {
    reopened.close();
    other.close();
  }
});

it("bounds pending terminal batches without consuming later executions", async () => {
  const { journal } = await fixture();
  const terminals = Array.from({ length: 101 }, (_, i) => run(journal, String(i)));
  expect(journal.pendingLearningRuns()).toEqual(terminals.slice(0, 100));
  journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.finished",
    value: { state: "completed", terminalIds: terminals.slice(0, 100).map(({ id }) => id) },
  });
  expect(journal.pendingLearningRuns()).toEqual(terminals.slice(100));
});

it("resumes the first unfinished review job and reads only its directory-scoped memory events", async () => {
  const { root, journal } = await fixture();
  const first = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.queued",
    value: { terminalIds: [] },
  });
  const second = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.queued",
    value: { terminalIds: [] },
  });
  const plan = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.plan",
    value: { jobId: first.id, operations: [] },
  });
  const failure = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.finished",
    value: { jobId: first.id, state: "failed" },
  });
  journal.append(null, {
    type: EventType.CUSTOM,
    name: "unrelated",
    value: { jobId: first.id },
  });
  journal.close();
  const reopened = new ExecutionJournal(root, "workspace-a");
  const other = new ExecutionJournal(root, "workspace-b");
  try {
    expect(reopened.pendingMemoryReview()).toEqual(first);
    expect(reopened.memoryJobEvents(first.id)).toEqual([plan, failure]);
    expect(reopened.memoryJobEvents(second.id)).toEqual([]);
    expect(other.pendingMemoryReview()).toBeUndefined();
    expect(other.memoryJobEvents(first.id)).toEqual([]);
    reopened.append(null, {
      type: EventType.CUSTOM,
      name: "swarmx.memory.review.finished",
      value: { jobId: first.id, state: "completed" },
    });
    expect(reopened.pendingMemoryReview()).toEqual(second);
  } finally {
    reopened.close();
    other.close();
  }
});

it("supersedes a failed review without acknowledging its original execution evidence", async () => {
  const { root, journal } = await fixture();
  const terminal = run(journal, "retry");
  const first = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.queued",
    value: { terminalIds: [terminal.id] },
  });
  journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.finished",
    value: { jobId: first.id, state: "failed" },
  });
  const replacement = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.queued",
    value: { terminalIds: [terminal.id] },
  });
  const other = new ExecutionJournal(root, "workspace-b");
  try {
    other.append(null, {
      type: EventType.CUSTOM,
      name: "swarmx.memory.review.superseded",
      value: { jobId: first.id },
    });
    expect(journal.pendingMemoryReview()).toEqual(first);
  } finally {
    other.close();
  }
  journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.superseded",
    value: { jobId: first.id },
  });
  expect(journal.pendingMemoryReview()).toEqual(replacement);
  expect(journal.pendingLearningRuns()).toEqual([terminal]);
});

it("retains full run evidence and causal delegation choices while reporting oversized omissions", async () => {
  const { root, journal } = await fixture();
  const parent = context("parent");
  const call = journal.append(parent, {
    type: EventType.TOOL_CALL_START,
    toolCallId: "delegate",
    toolCallName: "swarm",
  });
  const delegation = { ...parent, causedBy: call.id };
  const args = journal.append(delegation, {
    type: EventType.TOOL_CALL_ARGS,
    toolCallId: "delegate",
    delta: JSON.stringify({ action: "send_message", reason: "Observed writing quality" }),
  });
  const child = {
    ...context("child"),
    causedBy: call.id,
    attributes: { "swarmx.harness.name": "pi", "gen_ai.request.model": "provider/model" },
  };
  const started = journal.append(child, {
    type: EventType.RUN_STARTED,
    threadId: child.sessionId,
    runId: child.runId,
  });
  const resources = journal.append(child, {
    type: EventType.CUSTOM,
    name: "swarmx.learning.resources",
    value: [{ id: "writer", kind: "agent", path: "agents/writer.md", revision: "sha256:observed" }],
  });
  const oversized = journal.append(child, {
    type: EventType.TEXT_MESSAGE_CHUNK,
    messageId: "long",
    role: "assistant",
    delta: "x".repeat(40_001),
  });
  const text = journal.append(child, {
    type: EventType.TEXT_MESSAGE_CHUNK,
    messageId: "answer",
    role: "assistant",
    delta: "Completed",
  });
  const terminal = journal.append(child, { type: EventType.RUN_ERROR, message: "Rate limited" });
  const result = journal.append(delegation, {
    type: EventType.TOOL_CALL_RESULT,
    toolCallId: "delegate",
    messageId: "delegation-result",
    content: JSON.stringify({ result: "failed" }),
  });
  journal.append(child, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.started",
    value: { snapshot: "Never learn from a review's own output" },
  });
  run(journal, "unrelated");
  const evidence = journal.learningEvidence(["child"]);
  expect(evidence.records).toEqual([call, args, started, resources, text, terminal, result]);
  expect(evidence.omitted).toEqual([oversized.id]);
  expect(evidence.omittedCount).toBe(1);
  expect(JSON.stringify(evidence.records).length).toBeLessThanOrEqual(40_000);
  expect(journal.learningEvidence([])).toMatchObject({
    records: [],
    omitted: [],
    omittedCount: 0,
    runs: [],
    statistics: { sampleCount: 0 },
  });
  const other = new ExecutionJournal(root, "workspace-b");
  try {
    expect(other.learningEvidence(["child"])).toMatchObject({
      records: [],
      omitted: [],
      omittedCount: 0,
    });
  } finally {
    other.close();
  }
});

it("caps omission diagnostics while retaining the exact omitted event count", async () => {
  const { journal } = await fixture();
  const scope = context("large");
  const omitted = Array.from({ length: 110 }, (_, i) =>
    journal.append(scope, {
      type: EventType.TEXT_MESSAGE_CHUNK,
      messageId: String(i),
      role: "assistant",
      delta: "x".repeat(40_001),
    }),
  );
  expect(journal.learningEvidence([scope.runId])).toMatchObject({
    records: [],
    omitted: omitted.slice(0, 100).map(({ id }) => id),
    omittedCount: 110,
  });
});
