import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { EventType } from "@ag-ui/core";
import { afterEach, expect, it, vi } from "vitest";
import { ExecutionEvidenceSchema, type ExecutionRecord } from "../src/execution-record.js";
import {
  type ExecutionContext,
  ExecutionJournal,
  ExecutionSourceError,
} from "../src/host/execution-journal.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
  vi.useRealTimers();
});

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-execution-evidence-"));
  const journal = new ExecutionJournal(root, "workspace-a");
  cleanups.push(async () => {
    journal.close();
    await rm(root, { recursive: true, force: true });
  });
  return { root, journal };
}

const source = (record: ExecutionRecord) => `urn:swarmx:execution:${record.id}`;
const context = (runId: string): ExecutionContext & { sessionId: string } => ({
  sessionId: `session:${runId}`,
  runId,
  causedBy: null,
  attributes: {
    "swarmx.harness.name": "dsh",
    "gen_ai.request.model": "requested-provider/requested",
  },
});

function start(journal: ExecutionJournal, scope: ReturnType<typeof context>) {
  return journal.append(scope, {
    type: EventType.RUN_STARTED,
    threadId: scope.sessionId,
    runId: scope.runId,
    input: {
      threadId: scope.sessionId,
      runId: scope.runId,
      messages: [{ id: "task", role: "user", content: `Actual task ${scope.runId}` }],
      tools: [],
      context: [],
      state: {},
      forwardedProps: { profile: "sdk-minimal" },
    },
  });
}

function finish(
  journal: ExecutionJournal,
  scope: ReturnType<typeof context>,
  stopReason = "end_turn",
  attributes = {},
) {
  return journal.append(
    scope,
    {
      type: EventType.RUN_FINISHED,
      threadId: scope.sessionId,
      runId: scope.runId,
      result: { stopReason },
    },
    attributes,
  );
}

it("resolves exact immutable sources after reopen and rejects foreign, missing and malformed sources", async () => {
  const { root, journal } = await fixture();
  const record = start(journal, context("one"));
  journal.close();
  const reopened = new ExecutionJournal(root, "workspace-a");
  const foreign = new ExecutionJournal(root, "workspace-b");
  try {
    expect(reopened.resolveSource(source(record))).toEqual(record);
    expect(() => foreign.resolveSource(source(record))).toThrow(ExecutionSourceError);
    expect(() => reopened.resolveSource(`urn:swarmx:execution:${randomUUID()}`)).toThrow(
      ExecutionSourceError,
    );
    for (const invalid of [
      "not-a-source",
      `${source(record)}?run=one`,
      `URN:swarmx:execution:${record.id}`,
      "urn:swarmx:execution:../one",
    ])
      expect(() => reopened.resolveSource(invalid)).toThrow(ExecutionSourceError);
    expect(() => reopened.evidence([])).toThrow();
    expect(() => reopened.evidence(Array(65).fill(source(record)))).toThrow();
    reopened.close();
    expect(() => reopened.resolveSource(source(record))).toThrow();
    try {
      reopened.resolveSource(source(record));
    } catch (error) {
      expect(error).not.toBeInstanceOf(ExecutionSourceError);
    }
  } finally {
    reopened.close();
    foreign.close();
  }
});

it("reads the exact stored JSON text without normalizing whitespace or unknown fields", async () => {
  const { root, journal } = await fixture();
  const record = journal.append(null, {
    type: EventType.RAW,
    event: { nativeFutureField: { text: "原文\nCafé", count: 7 } },
  });
  const id = randomUUID();
  const raw = JSON.stringify(
    { ...record, id, seq: record.seq + 1, futureEnvelope: "retain" },
    null,
    2,
  );
  const database = new DatabaseSync(journal.databasePath);
  try {
    database
      .prepare("INSERT INTO execution_events (seq,id,workspace_id,record_json) VALUES (?,?,?,?)")
      .run(record.seq + 1, id, "workspace-a", raw);
  } finally {
    database.close();
  }
  const reference = `urn:swarmx:execution:${id}`;
  expect(journal.sourceText(reference)).toBe(raw);
  expect(journal.resolveSource(reference).event).toMatchObject({
    event: { nativeFutureField: { text: "原文\nCafé", count: 7 } },
  });
  expect(JSON.stringify(journal.resolveSource(reference))).not.toBe(raw);
  const foreign = new ExecutionJournal(root, "workspace-b");
  try {
    expect(() => foreign.sourceText(reference)).toThrow(ExecutionSourceError);
    expect(() => journal.sourceText("urn:swarmx:execution:invalid")).toThrow(ExecutionSourceError);
  } finally {
    foreign.close();
  }
});

it("reads only decisions for the selected proposals in the current directory", async () => {
  const { root, journal } = await fixture();
  const proposalIds = [randomUUID(), randomUUID()];
  const first = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.accepted",
    value: { proposalId: proposalIds[0] },
  });
  journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.rejected",
    value: { proposalId: randomUUID() },
  });
  journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.saved",
    value: { proposalId: proposalIds[0] },
  });
  const last = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.rejected",
    value: { proposalId: proposalIds[1] },
  });
  const foreign = new ExecutionJournal(root, "workspace-b");
  try {
    foreign.append(null, {
      type: EventType.CUSTOM,
      name: "swarmx.memory.accepted",
      value: { proposalId: proposalIds[0] },
    });
    expect(journal.memoryDecisions(proposalIds)).toEqual([first, last]);
    expect(journal.memoryDecisions([])).toEqual([]);
  } finally {
    foreign.close();
  }
});

it("counts distinct started executions with explicit outcomes and reports only known terminal quantities", async () => {
  const { journal } = await fixture();
  vi.useFakeTimers({ toFake: ["Date"] });
  const base = Date.UTC(2026, 8, 20);
  const sources: string[] = [];
  const cases = [
    ["end_turn", 1_000],
    ["error", 3_000],
    ["cancelled", 4_000],
    ["incomplete", 0],
    ["max_tokens", 9_000],
  ] as const;
  for (const [index, [reason, duration]] of cases.entries()) {
    const scope = context(String(index));
    vi.setSystemTime(base + index * 100_000);
    const started = start(journal, scope);
    sources.push(source(started));
    journal.append(
      scope,
      { type: EventType.RAW, event: { cumulative: true } },
      {
        "gen_ai.usage.input_tokens": 9_999,
        "gen_ai.usage.output_tokens": 9_999,
        "swarmx.usage.cost_usd": 99,
      },
    );
    vi.setSystemTime(base + index * 100_000 + duration);
    if (reason === "incomplete") continue;
    const terminal =
      reason === "error"
        ? journal.append(scope, { type: EventType.RUN_ERROR, message: "Network failed" })
        : finish(
            journal,
            scope,
            reason,
            index
              ? {}
              : {
                  "gen_ai.response.model": "reported-model",
                  "gen_ai.provider.name": "reported-provider",
                  "swarmx.agent.version": "native-version",
                  "swarmx.model.version": "model-version",
                  "gen_ai.usage.input_tokens": 20,
                  "gen_ai.usage.output_tokens": 10,
                  "swarmx.usage.cost_usd": 0.03,
                  "swarmx.usage.basis": "Native reported usage; native estimated cost",
                },
          );
    sources.push(source(terminal));
  }
  const evidence = journal.evidence([...sources, ...sources]);
  expect(ExecutionEvidenceSchema.parse(evidence)).toEqual(evidence);
  expect(evidence.runs[0]).toMatchObject({
    task: "Actual task 0",
    harness: "dsh",
    requestedModel: "requested-provider/requested",
    provider: "reported-provider",
    harnessVersion: "native-version",
    modelVersion: "model-version",
    profile: "sdk-minimal",
    elapsedMs: 1_000,
    inputTokens: 20,
    outputTokens: 10,
    costUsd: 0.03,
    usageBasis: "Native reported usage; native estimated cost",
  });
  expect(evidence.runs[1]).toMatchObject({
    provider: null,
    inputTokens: null,
    outputTokens: null,
    costUsd: null,
  });
  expect(evidence.runs[3]).toMatchObject({
    outcome: "incomplete",
    finishedAt: null,
    elapsedMs: null,
  });
  expect(evidence.statistics).toEqual({
    recipe: "swarmx.execution.v1",
    scope: "cited executions",
    runIds: ["0", "1", "2", "3", "4"],
    window: {
      startedAt: new Date(base).toISOString(),
      finishedAt: new Date(base + 409_000).toISOString(),
    },
    sampleCount: 5,
    completed: 1,
    error: 1,
    cancelled: 1,
    incomplete: 1,
    other: 1,
    elapsed: { sampleCount: 4, medianMs: 3_500 },
    usage: { sampleCount: 1, inputTokens: 20, outputTokens: 10 },
    cost: { sampleCount: 1, usd: 0.03, complete: false },
    tools: { callCount: 0, usd: 0, unpricedCalls: 0 },
  });
  expect(journal.evidence([...sources].reverse()).statistics).toEqual(evidence.statistics);
});

it("keeps snapshot evidence frozen and never fetches omitted or later transcripts", async () => {
  const { journal } = await fixture();
  const scope = context("frozen");
  const started = start(journal, scope);
  const oversized = journal.append(scope, {
    type: EventType.TEXT_MESSAGE_CHUNK,
    messageId: "large",
    role: "assistant",
    delta: "x".repeat(40_001),
  });
  const saved = journal.learningEvidence([scope.runId]);
  expect(saved.omitted).toEqual([oversized.id]);
  const snapshot = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.started",
    value: { snapshot: { evidence: saved } },
  });
  const terminal = finish(journal, scope);
  journal.append(scope, { type: EventType.RAW, event: { mustNotAppear: true } });
  const evidence = journal.evidence([source(snapshot)]);
  expect(evidence.records).toEqual([started, snapshot]);
  expect(evidence.runs).toEqual(saved.runs);
  expect(evidence.statistics).toEqual(saved.statistics);
  expect(evidence.statistics).toMatchObject({
    sampleCount: 1,
    incomplete: 1,
    completed: 0,
    usage: { sampleCount: 0, inputTokens: null, outputTokens: null },
    cost: { sampleCount: 0, usd: null },
  });
  const current = journal.evidence([source(started)]);
  expect(current.records).toEqual([started]);
  expect(current.runs[0]?.sources).toEqual([source(started), source(terminal)]);
  expect(current.statistics).toMatchObject({ sampleCount: 1, completed: 1, incomplete: 0 });
});

it.each([
  { children: 1, taskLength: 41_000 },
  { children: 8, taskLength: 9_000 },
])("does not call truncated descendant costs complete (%j)", async ({ children, taskLength }) => {
  const { journal } = await fixture();
  const parent = context("parent");
  const started = start(journal, parent);
  for (let index = 0; index < children; index++) {
    const child = {
      ...context(`child-${index}`),
      causedBy: started.id,
      attributes: { ...parent.attributes, "swarmx.execution.parent_run_id": parent.runId },
    };
    journal.append(child, {
      type: EventType.RUN_STARTED,
      threadId: child.sessionId,
      runId: child.runId,
      input: {
        threadId: child.sessionId,
        runId: child.runId,
        messages: [{ id: "task", role: "user", content: "x".repeat(taskLength) }],
        tools: [],
        context: [],
        state: {},
        forwardedProps: {},
      },
    });
    finish(journal, child, "end_turn", { "swarmx.usage.cost_usd": 2 });
  }
  const terminal = finish(journal, parent, "end_turn", { "swarmx.usage.cost_usd": 1 });
  const saved = journal.learningEvidence([parent.runId], [terminal.id]);
  expect(saved.omittedCount).toBeGreaterThan(0);
  expect(saved.statistics.cost.complete).toBe(false);
  expect(saved.runs).not.toHaveLength(0);
  expect(saved.runs.every((run) => run.totalCostUsd === null && !run.totalCostComplete)).toBe(true);
  const snapshot = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.started",
    value: { snapshot: { evidence: saved } },
  });
  const recalled = journal.evidence([source(snapshot)]);
  expect(recalled.runs).toEqual(saved.runs);
  expect(recalled.statistics).toEqual(saved.statistics);
  expect(recalled.records).toEqual([...saved.records, snapshot]);
  expect(journal.evidence([source(started)]).runs[0]).toMatchObject({
    totalCostUsd: 1 + children * 2,
    totalCostComplete: true,
  });
});

it("summarizes only the requested model and preserves native attributes in the original record", async () => {
  const { journal } = await fixture();
  const scope = context("unknown-model");
  start(journal, scope);
  const terminal = finish(journal, scope, "end_turn", {
    "gen_ai.response.model": null,
    "swarmx.agent.model": "configured-model",
    "gen_ai.usage.input_tokens": 0,
    "gen_ai.usage.output_tokens": 0,
    "swarmx.usage.cost_usd": 0,
  });
  const evidence = journal.evidence([source(terminal)]);
  expect(evidence.runs[0]).toMatchObject({
    requestedModel: "requested-provider/requested",
  });
  expect(evidence.runs[0]).not.toHaveProperty("reportedModel");
  expect(evidence.records[0]?.attributes["gen_ai.response.model"]).toBeNull();
  expect(evidence.records[0]?.attributes["swarmx.agent.model"]).toBe("configured-model");
  expect(evidence.statistics).toMatchObject({
    usage: { sampleCount: 1, inputTokens: 0, outputTokens: 0 },
    cost: { sampleCount: 1, usd: 0 },
  });
});

it("uses dispatch effort without inferring a level from later native records", async () => {
  const { journal } = await fixture();
  const cases = [
    ["high", "medium", "high"],
    ["high", undefined, "high"],
    [undefined, "off", null],
    [undefined, undefined, null],
    ["low", null, "low"],
    [3, false, null],
  ] as const;
  for (const [index, [requested, reported, requestedEffort]] of cases.entries()) {
    const scope = context(`effort-${index}`);
    scope.attributes["gen_ai.request.reasoning.level"] = requested;
    start(journal, scope);
    journal.append(
      scope,
      { type: EventType.RAW, event: { type: "unprojected-config" } },
      {
        "swarmx.agent.effort": "raw-only-level",
      },
    );
    const terminal = finish(journal, scope, "end_turn", {
      "gen_ai.request.reasoning.level": "later-request-attribute",
      "swarmx.agent.effort": reported,
    });
    const evidence = journal.evidence([source(terminal)]);
    expect(evidence.runs).toHaveLength(1);
    expect(evidence.runs[0]).toMatchObject({ requestedEffort });
    expect(evidence.runs[0]).not.toHaveProperty("reportedEffort");
    expect(evidence.records[0]?.attributes["swarmx.agent.effort"]).toBe(reported);
  }
});

it("counts independent tool charges once and leaves unpriced tool calls visible", async () => {
  const { journal } = await fixture();
  const scope = context("tools");
  const started = start(journal, scope);
  journal.append(scope, {
    type: EventType.TOOL_CALL_START,
    toolCallId: "priced",
    toolCallName: "cloud-compute",
  });
  for (let index = 0; index < 2; index++)
    journal.append(
      scope,
      {
        type: EventType.TOOL_CALL_RESULT,
        toolCallId: "priced",
        messageId: `result-${index}`,
        content: "{}",
      },
      { "swarmx.tool.cost_usd": 2 },
    );
  journal.append(scope, {
    type: EventType.TOOL_CALL_CHUNK,
    toolCallId: "local",
    toolCallName: "bash",
    delta: "{}",
  });
  finish(journal, scope, "end_turn", { "swarmx.usage.cost_usd": 1 });
  const evidence = journal.evidence([source(started)]);
  expect(evidence.runs[0]).toMatchObject({ costUsd: 1, totalCostUsd: 3, totalCostComplete: true });
  expect(evidence.statistics).toMatchObject({
    tools: { callCount: 2, usd: 2, unpricedCalls: 1 },
    cost: { sampleCount: 1, usd: 3, complete: true },
  });
});

it("shows only the reviewer plan from the cited attempt without changing its frozen statistics", async () => {
  const { journal } = await fixture();
  const scope = context("reviewed");
  const started = start(journal, scope);
  const queued = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.queued",
    value: { runIds: [scope.runId] },
  });
  const saved = journal.learningEvidence([scope.runId]);
  const first = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.started",
    value: { jobId: queued.id, snapshot: { evidence: saved } },
  });
  const firstResponse = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.response",
    value: {
      jobId: queued.id,
      source: source(first),
      text: "Invalid JSON response",
      reviewer: { model: "first-reviewer" },
    },
  });
  journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.finished",
    value: { jobId: queued.id, state: "failed" },
  });
  const terminal = finish(journal, scope);
  const retryEvidence = journal.learningEvidence([scope.runId]);
  const retry = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.started",
    value: { jobId: queued.id, snapshot: { evidence: retryEvidence } },
  });
  journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.response",
    value: { jobId: queued.id, source: source(first), text: "Wrong attempt", reviewer: {} },
  });
  const response = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.response",
    value: {
      jobId: queued.id,
      source: source(retry),
      text: '{\n  "summary": "原始回复"\n}',
      reviewer: { model: "reported-reviewer" },
    },
  });
  journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.planned",
    value: { jobId: randomUUID(), reviewer: { model: "unrelated-reviewer" } },
  });
  const plan = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.planned",
    value: {
      jobId: queued.id,
      reviewer: { harness: "codex", model: "reported-reviewer", version: "reported-version" },
    },
  });
  const firstResult = journal.evidence([source(first)]);
  expect(firstResult.records).toEqual([started, first, firstResponse]);
  expect(firstResult.statistics).toEqual(saved.statistics);
  const retryResult = journal.evidence([source(retry)]);
  expect(retryResult.records).toEqual([started, terminal, retry, response, plan]);
  expect(retryResult.statistics).toEqual(retryEvidence.statistics);
});

it("excludes parent tool context and lifecycle without a start from execution denominators", async () => {
  const { journal } = await fixture();
  const parent = context("parent");
  const parentStart = start(journal, parent);
  const tool = journal.append(
    { ...parent, causedBy: parentStart.id },
    {
      type: EventType.TOOL_CALL_START,
      toolCallId: "delegate",
      toolCallName: "swarm",
    },
  );
  const child = { ...context("child"), causedBy: tool.id };
  start(journal, child);
  finish(journal, child, "cancelled");
  const orphan = finish(journal, context("orphan"));
  const saved = journal.learningEvidence(["child"]);
  expect(saved.records).toContainEqual(parentStart);
  expect(saved.statistics).toMatchObject({
    runIds: ["child"],
    sampleCount: 1,
    cancelled: 1,
    error: 0,
  });
  const job = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.queued",
    value: { runIds: ["child"] },
  });
  const snapshot = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.started",
    value: {
      jobId: job.id,
      snapshot: {
        evidence: {
          ...saved,
          statistics: { ...saved.statistics, runIds: ["parent", "child"] },
        },
      },
    },
  });
  expect(journal.evidence([source(snapshot)]).statistics).toEqual(saved.statistics);
  const legacy = journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.started",
    value: { jobId: job.id, snapshot: { evidence: { records: saved.records } } },
  });
  expect(journal.evidence([source(legacy)]).statistics).toEqual(saved.statistics);
  expect(journal.evidence([source(orphan)]).statistics).toMatchObject({
    runIds: [],
    sampleCount: 0,
  });
});

it("rejects snapshots that alter source records or embed records from another directory", async () => {
  const { root, journal } = await fixture();
  const record = start(journal, context("real"));
  const foreign = new ExecutionJournal(root, "workspace-b");
  try {
    const other = start(foreign, context("foreign"));
    for (const entry of [
      { ...record, attributes: { "gen_ai.response.model": "fabricated" } },
      other,
    ]) {
      const snapshot = journal.append(null, {
        type: EventType.CUSTOM,
        name: "swarmx.memory.review.started",
        value: { snapshot: { evidence: { records: [entry] } } },
      });
      expect(() => journal.evidence([source(snapshot)])).toThrow(ExecutionSourceError);
    }
  } finally {
    foreign.close();
  }
});
