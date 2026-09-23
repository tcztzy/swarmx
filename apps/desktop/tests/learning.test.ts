import { randomUUID } from "node:crypto";
import { mkdtemp, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { afterEach, expect, it, vi } from "vitest";
import { HARNESS_CAPABILITIES, type NativeAgent, type Observer } from "../src/agents/types.js";
import { AgentMemory, type MemoryReviewer } from "../src/host/memory.js";
import { ProductServices } from "../src/host/product-services.js";
import { recordedAgent } from "../src/host/recorded-agent.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
const sink: Observer = { text() {}, tool() {}, raw() {}, interact: async () => undefined };
const unchanged =
  '{"summary":"No durable change is supported by these executions.","operations":[]}';

async function fixture(reviewer = vi.fn<MemoryReviewer>().mockResolvedValue(unchanged)) {
  const root = await mkdtemp(join(tmpdir(), "swarmx-learning-queue-"));
  const options = { cwd: root, productHome: join(root, "product") };
  const products = await ProductServices.create(options);
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
  const start = vi.fn<NativeAgent["start"]>(async (_id, _text, observer) => {
    observer.text("answer", "Observed result");
    return { stopReason: "end_turn" };
  });
  const agent = recordedAgent(
    products.journal,
    "codex",
    {
      name: "fixture",
      capabilities: HARNESS_CAPABILITIES.codex,
      models: async () => ({ models: [], current: {} }),
      list: async () => [],
      create: async () => "codex:fixture",
      read: async () => {},
      start,
      steer: async () => {},
      interrupt: async () => {},
      dispose: async () => {},
    },
    memory,
  );
  return { options, products, memory, reviewer, start, agent };
}

it("counts short child sessions together instead of waiting for ten turns in each", async () => {
  const { products, memory, reviewer, agent } = await fixture();
  products.settings.writeMemory({ ...products.settings.readMemory(), reviewInterval: 2 });
  await agent.start("codex:first", "First child", sink);
  expect(reviewer).not.toHaveBeenCalled();
  await agent.start("codex:second", "Second child", sink);
  await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(1));
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("completed"));
  expect(reviewer.mock.calls[0]?.[0]).toContain("First child");
  expect(reviewer.mock.calls[0]?.[0]).toContain("Second child");
});

it("retains the original reviewer response and its source when JSON parsing fails", async () => {
  const text = ' \r\n```json\r\n{"summary":"原始回复", "operations":[]}\r\n``` \r\n';
  const reviewer = vi.fn<MemoryReviewer>(
    async (_prompt, _signal, _harness, _permissions, reportIdentity) => {
      reportIdentity({
        "gen_ai.request.model": "requested-reviewer",
        "gen_ai.response.model": "reported-reviewer",
        "gen_ai.provider.name": "reported-provider",
        "swarmx.harness.version": "reported-version",
      });
      return text;
    },
  );
  const { products, memory, agent } = await fixture(reviewer);
  products.settings.writeMemory({ ...products.settings.readMemory(), reviewInterval: 1 });
  await agent.start("codex:invalid-review", "Preserve even a malformed review", sink);
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("failed"));
  const started = products.journal.memoryEvent("swarmx.memory.review.started");
  const response = products.journal.memoryEvent("swarmx.memory.review.response");
  if (started?.event.type !== EventType.CUSTOM || response?.event.type !== EventType.CUSTOM)
    throw new Error("Missing review source or original response");
  expect(response.seq).toBeGreaterThan(started.seq);
  expect(response.event.value).toEqual({
    jobId: started.event.value.jobId,
    source: `urn:swarmx:execution:${started.id}`,
    text,
    reviewer: {
      harness: products.settings.readMemory().reviewHarness,
      requestedModel: "requested-reviewer",
      provider: "reported-provider",
      version: "reported-version",
    },
  });
  expect(started.event.value.reviewer).not.toHaveProperty("model");
  expect(started.event.value.prompt).toContain("Use requested settings as the Agent identity");
  expect(started.event.value.prompt).not.toContain("native-reported effort");
  expect(started.event.value.prompt).not.toContain("distinguish requested settings");
  expect(products.journal.memoryEvent("swarmx.memory.review.planned")).toBeUndefined();
  expect(products.journal.pendingLearningRuns()).toHaveLength(1);
});

it("drains completions arriving during a paused review without acknowledging them early", async () => {
  const gate = Promise.withResolvers<string>();
  const reviewer = vi
    .fn<MemoryReviewer>()
    .mockReturnValueOnce(gate.promise)
    .mockResolvedValue(unchanged);
  const { products, memory, agent } = await fixture(reviewer);
  products.settings.writeMemory({ ...products.settings.readMemory(), reviewInterval: 1 });
  await agent.start("codex:first", "First child", sink);
  await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(1));
  await agent.start("codex:second", "Second child", sink);
  await agent.start("codex:third", "Third child", sink);
  expect(reviewer).toHaveBeenCalledTimes(1);
  gate.resolve(unchanged);
  await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(2));
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("completed"));
  expect(reviewer.mock.calls[1]?.[0]).toContain("Second child");
  expect(reviewer.mock.calls[1]?.[0]).toContain("Third child");
  expect(products.journal.pendingLearningRuns()).toEqual([]);
});

it("retains failed work across a reopened journal and uses the original reviewer grants", async () => {
  const first = await fixture(vi.fn<MemoryReviewer>().mockRejectedValue(new Error("offline")));
  const permissions = {
    tools: ["memory.read", "memory.write"] as const,
    delegation: true,
    harnesses: { codex: ["permitted-model"] },
  };
  first.products.settings.writeMemory({
    ...first.products.settings.readMemory(),
    reviewInterval: 1,
  });
  await first.products.journal.scope.run(
    {
      sessionId: null,
      runId: randomUUID(),
      causedBy: null,
      attributes: {},
      permissions: { ...permissions, tools: [...permissions.tools] },
    },
    () => first.agent.start("codex:one", "Keep this failure evidence", sink),
  );
  await vi.waitFor(async () => expect((await first.memory.status()).review.state).toBe("failed"));
  expect(first.products.journal.pendingLearningRuns()).toHaveLength(1);
  expect(first.reviewer).toHaveBeenCalledTimes(1);
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
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("completed"));
  expect(reviewer.mock.calls[0]?.[0]).toContain("Keep this failure evidence");
  expect(reviewer.mock.calls[0]?.[3]).toEqual(permissions);
  expect(products.journal.pendingLearningRuns()).toEqual([]);
  expect((await memory.status()).review.summary).toContain("No durable change");
});

it("resumes the saved operation plan after a write/receipt crash without asking the model again", async () => {
  const first = await fixture();
  first.products.settings.writeMemory({
    ...first.products.settings.readMemory(),
    reviewInterval: 1,
  });
  first.reviewer.mockImplementation(async (prompt) => {
    const snapshot = JSON.parse(prompt.split("\n\nSnapshot: ")[1] ?? "null");
    const answer = snapshot.evidence.records.find(
      (record: { event: { type: string } }) => record.event.type === EventType.TEXT_MESSAGE_CHUNK,
    );
    return JSON.stringify({
      summary: "Save an observed provider incident.",
      operations: [
        {
          action: "create_memory",
          request: {
            title: "Provider Incident",
            description: "One scoped observation.",
            type: "Finding",
            tags: ["agent-selection"],
            body: "A single observed provider incident; not a general quality ranking.",
            evaluation: {
              kind: "observation",
              task: "Provider observation",
              criteria: "Retain the observed provider response.",
              evidence: [`urn:swarmx:execution:${answer.id}`],
              limitations: "One execution, without a controlled comparison.",
            },
          },
        },
      ],
    });
  });
  const append = first.products.journal.append.bind(first.products.journal);
  const failedReview = Promise.withResolvers<void>();
  let failed = false;
  vi.spyOn(first.products.journal, "append").mockImplementation((context, event, attributes) => {
    if (!failed && event.type === EventType.CUSTOM && event.name === "swarmx.memory.saved") {
      failed = true;
      throw new Error("crash after the file was saved");
    }
    const recorded = append(context, event, attributes);
    if (event.type === EventType.CUSTOM && event.name === "swarmx.memory.review.finished")
      failedReview.resolve();
    return recorded;
  });
  await first.agent.start("codex:one", "Observe provider behavior", sink);
  await failedReview.promise;
  await first.memory.close();
  expect((await first.memory.status()).review.state).toBe("failed");
  const before = await first.products.memory.vault.readConcept("provider-incident.md");
  const beforeStat = await stat(join(first.options.productHome, "memory", "provider-incident.md"));
  await first.products.dispose();
  const products = await ProductServices.create(first.options);
  const reviewer = vi.fn<MemoryReviewer>();
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
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("completed"));
  expect(reviewer).not.toHaveBeenCalled();
  expect((await products.memory.vault.readConcept(before.id)).revision).toBe(before.revision);
  expect(
    (await stat(join(first.options.productHome, "memory", "provider-incident.md"))).mtimeMs,
  ).toBe(beforeStat.mtimeMs);
  expect(products.journal.pendingMemoryReview()).toBeUndefined();
  expect(products.journal.pendingLearningRuns()).toEqual([]);
});

it("keeps partial writes and retries only unfinished operations from the same plan", async () => {
  const { products, memory, reviewer, agent } = await fixture();
  products.settings.writeMemory({ ...products.settings.readMemory(), reviewInterval: 1 });
  reviewer.mockResolvedValue(
    JSON.stringify({
      summary: "Two independent observations.",
      operations: ["First finding", "Second finding"].map((title) => ({
        action: "create_memory",
        request: {
          title,
          description: "Observed evidence.",
          type: "Finding",
          body: "Keep the observed task context.",
        },
      })),
    }),
  );
  const create = products.memory.vault.createConcept.bind(products.memory.vault);
  let failSecond = true;
  const writes = vi
    .spyOn(products.memory.vault, "createConcept")
    .mockImplementation(async (request, signal) => {
      if (request.title === "Second finding" && failSecond) {
        failSecond = false;
        throw new Error("disk full");
      }
      return create(request, signal);
    });
  await agent.start("codex:one", "Produce evidence", sink);
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("failed"));
  const firstRevision = (await products.memory.vault.readConcept("first-finding.md")).revision;
  memory.resume();
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("completed"));
  expect(reviewer).toHaveBeenCalledTimes(1);
  expect(writes.mock.calls.map(([request]) => request.title)).toEqual([
    "First finding",
    "Second finding",
    "Second finding",
  ]);
  expect((await products.memory.vault.readConcept("first-finding.md")).revision).toBe(
    firstRevision,
  );
  expect(products.journal.pendingLearningRuns()).toEqual([]);
});

it("does not dispatch a reviewer after shutdown during asynchronous snapshot preparation", async () => {
  const { memory, products, reviewer, agent } = await fixture();
  products.settings.writeMemory({ ...products.settings.readMemory(), reviewInterval: 1 });
  const graph = products.memory.vault.graph.bind(products.memory.vault);
  const gate = Promise.withResolvers<void>();
  const entered = Promise.withResolvers<void>();
  vi.spyOn(products.memory.vault, "graph").mockImplementation(async () => {
    entered.resolve();
    await gate.promise;
    return graph();
  });
  await agent.start("codex:one", "Task", sink);
  await entered.promise;
  const closing = memory.close();
  gate.resolve();
  await closing;
  expect(reviewer).not.toHaveBeenCalled();
  expect(products.journal.pendingLearningRuns()).toHaveLength(1);
  expect(products.journal.pendingMemories()).toEqual([]);
});

it("does not publish a late model response after shutdown", async () => {
  const gate = Promise.withResolvers<string>();
  const { memory, products, reviewer, agent } = await fixture(
    vi.fn<MemoryReviewer>().mockReturnValue(gate.promise),
  );
  products.settings.writeMemory({ ...products.settings.readMemory(), reviewInterval: 1 });
  await agent.start("codex:one", "Task", sink);
  await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(1));
  const closing = memory.close();
  gate.resolve(unchanged);
  await closing;
  expect(products.journal.memoryEvent("swarmx.memory.review.planned")).toBeUndefined();
  expect(products.journal.pendingLearningRuns()).toHaveLength(1);
  expect((await memory.status()).review.state).toBe("failed");
});

it("leaves disallowed runs out of automatic learning", async () => {
  const { memory, products, reviewer, agent } = await fixture();
  products.settings.writeMemory({ ...products.settings.readMemory(), reviewInterval: 1 });
  await products.journal.scope.run(
    {
      sessionId: null,
      runId: randomUUID(),
      causedBy: null,
      attributes: {},
      permissions: { tools: ["memory.read"], harnesses: { codex: null }, delegation: true },
    },
    () => agent.start("codex:read-only", "Private task", sink),
  );
  await memory.close();
  expect(reviewer).not.toHaveBeenCalled();
  expect(products.journal.pendingLearningRuns()).toEqual([]);
});

it("preserves another session's manual review and focus while a review is busy", async () => {
  const gate = Promise.withResolvers<string>();
  const reviewer = vi
    .fn<MemoryReviewer>()
    .mockReturnValueOnce(gate.promise)
    .mockResolvedValue(unchanged);
  const { memory, agent } = await fixture(reviewer);
  await agent.start("codex:first", "First conversation", sink);
  await agent.start("codex:second", "Second conversation", sink);
  memory.review("codex:first", "First focus");
  await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(1));
  memory.review("codex:second", "Second focus");
  gate.resolve(unchanged);
  await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(2));
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("completed"));
  expect(reviewer.mock.calls[1]?.[0]).toContain("Second focus");
  expect(reviewer.mock.calls[1]?.[0]).toContain("Second conversation");
});

it("cancels before native dispatch when Stop arrives during resource snapshot preparation", async () => {
  const { memory, agent, start } = await fixture();
  const gate = Promise.withResolvers<void>();
  const entered = Promise.withResolvers<void>();
  vi.spyOn(memory.resources, "snapshot").mockImplementation(async (signal) => {
    entered.resolve();
    await gate.promise;
    signal.throwIfAborted();
    return [];
  });
  const running = agent.start("codex:preparing", "Stop before dispatch", sink);
  await entered.promise;
  await agent.interrupt("codex:preparing");
  gate.resolve();
  await expect(running).resolves.toMatchObject({ stopReason: "cancelled" });
  expect(start).not.toHaveBeenCalled();
  await memory.close();
  expect(start).not.toHaveBeenCalled();
});

it("allows a fresh manual review to resolve a failed plan's revision conflict", async () => {
  const gate = Promise.withResolvers<string>();
  const reviewer = vi
    .fn<MemoryReviewer>()
    .mockReturnValueOnce(gate.promise)
    .mockResolvedValue(unchanged);
  const { memory, agent, products } = await fixture(reviewer);
  products.settings.writeMemory({ ...products.settings.readMemory(), reviewInterval: 1 });
  const note = await memory.core.read();
  await agent.start("codex:one", "Learn this preference", sink);
  await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(1));
  await memory.core.update({
    expectedRevision: note.revision,
    content: "User edited this meanwhile.",
  });
  gate.resolve(
    JSON.stringify({
      summary: "Save the earlier preference.",
      operations: [
        {
          action: "update_core_memory",
          request: { expectedRevision: note.revision, content: "Outdated preference" },
        },
      ],
    }),
  );
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("failed"));
  const failed = products.journal.pendingMemoryReview();
  memory.review("codex:one");
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("completed"));
  expect(reviewer).toHaveBeenCalledTimes(2);
  expect(reviewer.mock.calls[1]?.[0]).toContain("User edited this meanwhile.");
  expect((await memory.core.read()).content).toBe("User edited this meanwhile.");
  expect(products.journal.memoryEvent("swarmx.memory.review.superseded")?.event).toMatchObject({
    value: { jobId: failed?.id },
  });
  expect(products.journal.pendingLearningRuns()).toEqual([]);
});

it.each(["error", "cancelled"])(
  "reviews %s outcomes with their observed route and evidence",
  async (outcome) => {
    const { start, agent, reviewer, memory } = await fixture();
    start.mockImplementationOnce(async () => {
      if (outcome === "error") throw new Error("provider unavailable");
      return { stopReason: "cancelled" };
    });
    const run = agent.start("codex:failed", "Try writing", sink, { model: "requested-model" });
    if (outcome === "error") await expect(run).rejects.toThrow("provider unavailable");
    else await expect(run).resolves.toMatchObject({ stopReason: "cancelled" });
    await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(1));
    await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("completed"));
    expect(reviewer.mock.calls[0]?.[0]).toContain("requested-model");
    expect(reviewer.mock.calls[0]?.[0]).toContain(
      outcome === "error" ? "provider unavailable" : "cancelled",
    );
  },
);
