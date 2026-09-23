import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createCodex } from "../src/agents/codex.js";
import type { NativeAgent } from "../src/agents/types.js";
import { ExecutionJournal } from "../src/host/execution-journal.js";
import { reviewMemory } from "../src/host/memory-review.js";
import { DEFAULT_POLICY } from "../src/settings.js";

const native = vi.hoisted(() => ({
  name: "review",
  capabilities: { history: true, list: true, resume: true, steer: true, emptySessionResume: false },
  models: vi.fn(async () => ({
    models: [{ id: "small", name: "Small", efforts: [] }],
    current: {},
  })),
  list: vi.fn(async () => []),
  read: vi.fn(async () => {}),
  steer: vi.fn(async () => {}),
  create: vi.fn(async () => "review"),
  start: vi.fn<NativeAgent["start"]>(async () => ({ stopReason: "end_turn" })),
  interrupt: vi.fn<NativeAgent["interrupt"]>(async () => {}),
  dispose: vi.fn(async () => {}),
}));
vi.mock("../src/agents/codex.js", () => ({ createCodex: vi.fn(async () => native) }));
vi.mock("../src/agents/claude.js", () => ({ createClaude: vi.fn(async () => native) }));
let root: string;
let journal: ExecutionJournal;
beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "swarmx-review-"));
  journal = new ExecutionJournal(root, "/research");
});
afterEach(async () => {
  journal.close();
  await rm(root, { recursive: true, force: true });
  vi.resetAllMocks();
});
const options = { cwd: "/research", mcp: { command: "node", args: ["/bridge.js"], env: {} } };

it("obeys harness/model admission for background reviews", async () => {
  await expect(
    reviewMemory(
      { ...options, executionPolicy: () => ({ ...DEFAULT_POLICY, harnesses: {} }) },
      "Review",
      new AbortController().signal,
      "codex",
      undefined,
      journal,
    ),
  ).rejects.toThrow("not permitted");
  expect(native.create).not.toHaveBeenCalled();
  await reviewMemory(
    { ...options, executionPolicy: () => ({ ...DEFAULT_POLICY, harnesses: { codex: ["small"] } }) },
    "Review",
    new AbortController().signal,
    "codex",
    undefined,
    journal,
  );
  expect(native.start).toHaveBeenCalledWith(expect.anything(), "Review", expect.anything(), {
    model: "small",
  });
  const restricted = vi.mocked(createCodex).mock.calls[0]?.[0];
  expect(restricted?.reviewOnly).toBe(true);
  expect(restricted?.executionPolicy?.()).toMatchObject({ tools: [], delegation: false });
});

it.each(["codex", "claude"] as const)(
  "accepts assistant text only and rejects tool calls in %s reviews",
  async (harness) => {
    native.start.mockImplementationOnce(async (_id, _prompt, observer) => {
      observer.text("user", "Untrusted input", "user");
      observer.text("reason", "Hidden reasoning", "reasoning");
      observer.text("answer", '{"operations":[]}');

      return { stopReason: "end_turn" as const };
    });
    expect(
      await reviewMemory(
        options,
        "Review",
        new AbortController().signal,
        harness,
        undefined,
        journal,
      ),
    ).toBe('{"operations":[]}');
    native.start.mockImplementationOnce(async (_id, _prompt, observer) => {
      await observer.tool("write", "shell", {});
      return { stopReason: "end_turn" as const };
    });
    await expect(
      reviewMemory(options, "Review", new AbortController().signal, harness, undefined, journal),
    ).rejects.toThrow("tool call");
    expect(native.dispose).toHaveBeenCalledTimes(2);
  },
);

it("reports requested and native reviewer identity without inferring missing values", async () => {
  native.start.mockImplementationOnce(async (_id, _prompt, observer) => {
    observer.raw(
      { native: "response" },
      {
        "gen_ai.response.model": "reported-model",
        "swarmx.harness.version": "observed-version",
      },
    );
    observer.text("answer", '{"operations":[]}');
    return { stopReason: "end_turn" };
  });
  const identity = vi.fn();
  await reviewMemory(options, "Review", new AbortController().signal, "codex", identity, journal);
  expect(identity.mock.calls).toEqual([
    [{ "gen_ai.request.model": null }],
    [{ "gen_ai.response.model": "reported-model", "swarmx.harness.version": "observed-version" }],
  ]);
});

it("interrupts cancellation, disposes the runtime and propagates interrupt failures", async () => {
  const controller = new AbortController();
  const finished = Promise.withResolvers<void>();
  native.start.mockImplementation(async () => {
    await finished.promise;
    return { stopReason: "end_turn" as const };
  });
  native.interrupt.mockImplementation(async () => {
    finished.resolve();
    throw new Error("Native interrupt failed");
  });
  const reviewing = reviewMemory(options, "Review", controller.signal, "codex", undefined, journal);
  const rejected = expect(reviewing).rejects.toThrow("Native interrupt failed");
  await vi.waitFor(() => expect(native.start).toHaveBeenCalledTimes(1));
  controller.abort(new Error("Shutdown"));
  await rejected;
  expect(native.dispose).toHaveBeenCalledTimes(1);
});

it("records review lifecycle and unmodified native usage without creating recursive review work", async () => {
  const raw = { type: "native_result", billing: { futureField: [1, 2] } };
  native.start.mockImplementationOnce(async (_id, _prompt, observer) => {
    observer.raw(raw, {
      "gen_ai.response.model": "reported",
      "gen_ai.usage.input_tokens": 12,
      "gen_ai.usage.output_tokens": 3,
      "swarmx.usage.cost_usd": 0.02,
      "swarmx.usage.basis": "Fixture native report",
    });
    observer.text("answer", '{"operations":[]}');
    return { stopReason: "end_turn" };
  });
  await reviewMemory(options, "Review", new AbortController().signal, "codex", undefined, journal);
  const records = journal.read({}).events;
  const started = records.find(({ event }) => event.type === EventType.RUN_STARTED);
  expect(started).toBeDefined();
  expect(started?.attributes).toMatchObject({
    "swarmx.execution.purpose": "memory-review",
    "swarmx.memory.review_eligible": false,
  });
  expect(journal.pendingLearningRuns()).toEqual([]);
  expect(records.find(({ event }) => event.type === EventType.RAW)?.event).toMatchObject({
    event: raw,
  });
  expect(journal.evidence([`urn:swarmx:execution:${started?.id}`]).runs[0]).toMatchObject({
    outcome: "completed",
    requestedModel: null,
    inputTokens: 12,
    outputTokens: 3,
    costUsd: 0.02,
    totalCostUsd: 0.02,
    totalCostComplete: true,
  });
});

it("does not dispatch after cancellation while a review session is being created", async () => {
  const created = Promise.withResolvers<string>();
  native.create.mockReturnValueOnce(created.promise);
  const controller = new AbortController();
  const reviewing = reviewMemory(options, "Review", controller.signal, "codex", undefined, journal);
  const rejected = expect(reviewing).rejects.toThrow("Stopped");
  await vi.waitFor(() => expect(native.create).toHaveBeenCalledOnce());
  controller.abort(new Error("Stopped"));
  created.resolve("review");
  await rejected;
  expect(native.start).not.toHaveBeenCalled();
  expect(native.dispose).toHaveBeenCalledOnce();
  expect(
    journal.read({}).events.filter(({ event }) => event.type === EventType.RUN_STARTED),
  ).toEqual([]);
});

it("preserves observed usage when an active review is cancelled", async () => {
  const finished = Promise.withResolvers<void>();
  native.start.mockImplementationOnce(async (_id, _prompt, observer) => {
    observer.raw(
      { billed: true },
      {
        "gen_ai.usage.input_tokens": 8,
        "gen_ai.usage.output_tokens": 2,
        "swarmx.usage.cost_usd": 0.01,
      },
    );
    await finished.promise;
    return { stopReason: "cancelled" };
  });
  native.interrupt.mockImplementationOnce(async () => finished.resolve());
  const controller = new AbortController();
  const reviewing = reviewMemory(options, "Review", controller.signal, "codex", undefined, journal);
  const rejected = expect(reviewing).rejects.toThrow("Stopped");
  await vi.waitFor(() => expect(native.start).toHaveBeenCalledOnce());
  controller.abort(new Error("Stopped"));
  await rejected;
  const terminal = journal
    .read({})
    .events.find(({ event }) => event.type === EventType.RUN_FINISHED);
  expect(journal.evidence([`urn:swarmx:execution:${terminal?.id}`]).runs[0]).toMatchObject({
    outcome: "cancelled",
    inputTokens: 8,
    outputTokens: 2,
    costUsd: 0.01,
  });
});
