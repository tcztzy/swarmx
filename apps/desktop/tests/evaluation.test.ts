import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { afterEach, expect, it, vi } from "vitest";
import { ExecutionJournal } from "../src/host/execution-journal.js";
import { AgentMemory, type MemoryReviewer } from "../src/host/memory.js";
import { ProductServices } from "../src/host/product-services.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
const context = { actorId: "model", callId: "evaluation", signal: new AbortController().signal };
const request = {
  title: "Research Agent Selection",
  type: "Finding",
  description: "Observed research task behavior",
  tags: ["agent-selection"],
  body: "The answer preserved the requested limitations on this task.",
};
const source = (id: string) => `urn:swarmx:execution:${id}`;
const evaluation = (id: string) => ({
  kind: "judgment",
  task: "Research writing",
  criteria: "Preserve the user's stated limitations.",
  evidence: [source(id)],
  counterEvidence: [],
  limitations: "One local task; not a controlled comparison or a general model ranking.",
});

async function fixture(
  reviewer: MemoryReviewer = async () => '{"summary":"No change","operations":[]}',
) {
  const root = await mkdtemp(join(tmpdir(), "swarmx-evaluation-"));
  const options = { productHome: join(root, "product"), cwd: root };
  const products = await ProductServices.create(options);
  products.settings.writeMemory({ autoReview: false });
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
  const scope = {
    sessionId: "codex:evidence",
    runId: randomUUID(),
    causedBy: null,
    attributes: { "swarmx.harness.name": "codex", "gen_ai.request.model": "requested" },
  };
  const started = products.journal.append(scope, {
    type: EventType.RUN_STARTED,
    threadId: scope.sessionId,
    runId: scope.runId,
    input: {
      threadId: scope.sessionId,
      runId: scope.runId,
      messages: [{ id: "prompt", role: "user", content: "Preserve my research limitations." }],
      tools: [],
      context: [],
      state: {},
    },
  });
  const answer = products.journal.append(
    { ...scope, causedBy: started.id },
    {
      type: EventType.TEXT_MESSAGE_CHUNK,
      messageId: "answer",
      role: "assistant",
      delta: "The limitations are preserved in the revised draft.",
    },
  );
  products.journal.append(
    { ...scope, causedBy: started.id },
    {
      type: EventType.RUN_FINISHED,
      threadId: scope.sessionId,
      runId: scope.runId,
      result: { stopReason: "end_turn" },
    },
  );
  return { root, options, products, memory, scope, started, answer };
}

it("requires structured evidence for new selection writes before staging", async () => {
  const { memory, products } = await fixture();
  products.settings.writeMemory({ writeApproval: true });
  await expect(memory.call({ action: "create_memory", request }, context)).rejects.toThrow(
    "evaluation",
  );
  expect((await memory.status()).pending).toEqual([]);
  expect((await products.memory.vault.search({ query: request.title })).items).toEqual([]);
});

it("rejects invented and foreign execution references before a write", async () => {
  const { memory, options } = await fixture();
  const foreign = new ExecutionJournal(join(options.productHome, "logs"), "other-workspace");
  const record = foreign.append(null, {
    type: EventType.CUSTOM,
    name: "foreign",
    value: "private",
  });
  foreign.close();
  for (const id of [randomUUID(), record.id]) {
    await expect(
      memory.call(
        { action: "create_memory", request: { ...request, evaluation: evaluation(id) } },
        context,
      ),
    ).rejects.toThrow();
  }
  expect((await memory.status()).pending).toEqual([]);
});

it("requires evidence when revising a legacy selection note even if its tag is removed", async () => {
  const { memory, products } = await fixture();
  const concept = await products.memory.vault.createConcept(request);
  await expect(
    memory.call(
      {
        action: "update_memory",
        request: {
          id: concept.id,
          expectedRevision: concept.revision,
          tags: [],
          body: "Unsupported replacement",
        },
      },
      context,
    ),
  ).rejects.toThrow("evaluation");
  expect((await products.memory.vault.readConcept(concept.id)).revision).toBe(concept.revision);
});

it("returns scoped evidence to the lead and labels old or foreign evaluations unverified", async () => {
  const { memory, products, answer, options } = await fixture();
  await memory.call(
    { action: "create_memory", request: { ...request, evaluation: evaluation(answer.id) } },
    context,
  );
  await products.memory.vault.createConcept({ ...request, title: "Legacy Agent Selection" });
  const selection = await memory.selection(["research"], context.signal);
  const concepts = selection.loaded.flatMap((group) => group.concepts);
  expect(concepts.find(({ id }) => id === "research-agent-selection.md")).toMatchObject({
    evaluation: {
      status: "referenced",
      statistics: { sampleCount: 1, completed: 1 },
      runs: [{ requestedModel: "requested", totalCostUsd: null, totalCostComplete: false }],
    },
  });
  expect(concepts.find(({ id }) => id === "legacy-agent-selection.md")).toMatchObject({
    evaluation: { status: "unverified" },
  });
  await mkdir(join(options.cwd, "other"));
  const other = await ProductServices.create({ ...options, cwd: join(options.cwd, "other") });
  try {
    const read = await other.learning.call(
      { action: "read_memory", request: { id: "research-agent-selection.md" } },
      context,
    );
    expect(read).toMatchObject({ data: { evaluation: { status: "unverified" } } });
  } finally {
    await other.dispose();
  }
});

it("binds each reviewed evaluation to inspected original events and records the reviewer prompt", async () => {
  const reviewer = vi.fn<MemoryReviewer>(
    async (prompt, _signal, _harness, _permissions, identity) => {
      identity({
        "gen_ai.request.model": "configured-review-model",
        "swarmx.agent.model": "configured-fallback",
        "gen_ai.response.model": null,
        "swarmx.harness.version": "review-runtime",
      });
      const snapshot = JSON.parse(prompt.split("\n\nSnapshot: ")[1] ?? "null");
      const answer = snapshot.evidence.records.find(
        (record: { event: { type: string } }) => record.event.type === EventType.TEXT_MESSAGE_CHUNK,
      );
      return JSON.stringify({
        summary: "Preserved the requested limitations in this single observed output.",
        operations: [
          { action: "create_memory", request: { ...request, evaluation: evaluation(answer.id) } },
        ],
      });
    },
  );
  const { memory, products, scope } = await fixture(reviewer);
  memory.review(scope.sessionId);
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("completed"));
  const concept = await products.memory.vault.readConcept("research-agent-selection.md");
  expect(concept.metadata.swarmx_evaluation?.review).toMatch(/^urn:swarmx:execution:/u);
  const snapshot = products.journal.memoryEvent("swarmx.memory.review.started");
  expect(snapshot?.event).toMatchObject({
    value: {
      prompt: reviewer.mock.calls[0]?.[0],
      promptRevision: expect.stringMatching(/^sha256:/u),
      reviewer: { harness: "codex" },
    },
  });
  const planned = products.journal.memoryEvent("swarmx.memory.review.planned");
  expect(planned?.event).toMatchObject({
    value: {
      reviewer: {
        harness: "codex",
        requestedModel: "configured-review-model",
        version: "review-runtime",
      },
    },
  });
  if (planned?.event.type !== EventType.CUSTOM) throw new Error("Missing review plan");
  expect(planned.event.value.reviewer).not.toHaveProperty("model");
});

it.each(["omitted", "previous-review"])(
  "rejects %s evidence in a new review plan before saving",
  async (kind) => {
    let evidenceId = "";
    const { memory, products, scope } = await fixture(async () =>
      JSON.stringify({
        summary: "A purported improvement",
        operations: [
          { action: "create_memory", request: { ...request, evaluation: evaluation(evidenceId) } },
        ],
      }),
    );
    const event =
      kind === "omitted"
        ? {
            type: EventType.TEXT_MESSAGE_CHUNK as const,
            messageId: "large",
            role: "assistant" as const,
            delta: "x".repeat(45_000),
          }
        : {
            type: EventType.CUSTOM as const,
            name: "swarmx.memory.review.started",
            value: { summary: "Prior speculation" },
          };
    evidenceId = products.journal.append(scope, event).id;
    memory.review(scope.sessionId);
    await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("failed"));
    expect(products.journal.memoryEvent("swarmx.memory.review.planned")).toBeUndefined();
    expect((await products.memory.vault.search({ query: request.title })).items).toEqual([]);
  },
);
