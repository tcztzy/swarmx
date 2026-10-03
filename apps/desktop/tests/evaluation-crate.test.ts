import { createHash, randomUUID } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { roCrateMetadataDocumentSchema } from "@swarmx/evidence";
import { afterEach, expect, it } from "vitest";
import { EvaluationCrateSchema } from "../src/evaluation-crate.js";
import { ProductServices } from "../src/host/product-services.js";

const close: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const fn of close.splice(0).reverse()) await fn();
});
const context = () => ({
  actorId: "renderer",
  callId: randomUUID(),
  signal: new AbortController().signal,
});
const urn = (id: string) => `urn:swarmx:execution:${id}`;
const hash = (content: string) => createHash("sha256").update(content).digest("hex");

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-crate-"));
  const products = await ProductServices.create({ cwd: root, productHome: join(root, "product") });
  products.settings.writeMemory({ autoReview: false });
  close.push(async () => {
    await products.dispose();
    await rm(root, { force: true, recursive: true });
  });
  const scope = {
    runId: randomUUID(),
    sessionId: "codex:crate",
    causedBy: null,
    attributes: {
      "swarmx.harness.name": "codex",
      "gen_ai.request.model": "requested-model",
      "gen_ai.request.reasoning.level": "low",
    },
  };
  const started = products.journal.append(scope, {
    type: EventType.RUN_STARTED,
    threadId: scope.sessionId,
    runId: scope.runId,
    input: {
      threadId: scope.sessionId,
      runId: scope.runId,
      messages: [{ id: "prompt", role: "user", content: "Preserve the limitations." }],
      tools: [],
      context: [],
      state: {},
    },
  });
  const answer = products.journal.append(scope, {
    type: EventType.TEXT_MESSAGE_CHUNK,
    messageId: "answer",
    role: "assistant",
    delta: "原始回答\nLimitations preserved.",
  });
  products.journal.append(
    { ...scope, attributes: { ...scope.attributes, "swarmx.agent.effort": "high" } },
    {
      type: EventType.RUN_FINISHED,
      threadId: scope.sessionId,
      runId: scope.runId,
      result: { stopReason: "end_turn" },
    },
  );
  const evaluation = {
    kind: "judgment" as const,
    task: "Writing",
    criteria: "Preserve limitations",
    evidence: [urn(answer.id)],
    counterEvidence: [urn(started.id)],
    limitations: "Single task, no comparison.",
  };
  const concept = await products.memory.vault.createConcept({
    title: "Writing evaluation",
    description: "One scoped observation",
    type: "Finding",
    tags: ["agent-selection"],
    body: "The original limitation remains.",
    evaluation,
  });
  const exportCrate = async (request: unknown) => {
    const result = (await products.callTool(
      "memory",
      { action: "export_evaluation", request },
      context(),
    )) as { data: unknown };
    return EvaluationCrateSchema.parse(result.data);
  };
  return { root, products, scope, started, answer, concept, evaluation, exportCrate };
}

it("exports exact pinned evidence as a deterministic Attached RO-Crate without mutating knowledge", async () => {
  const { products, root, concept, answer, scope, exportCrate } = await fixture();
  const request = { id: concept.id, expectedRevision: concept.revision };
  const first = await exportCrate(request);
  expect(await exportCrate(request)).toEqual(first);
  expect(roCrateMetadataDocumentSchema.safeParse(first.metadata).success).toBe(true);
  const files = new Map(first.files.map(({ path, content }) => [path, content]));
  expect(files.get("evaluation.md")).toBe(
    await readFile(join(root, "product/memory", concept.id), "utf8"),
  );
  expect(hash(files.get("evaluation.md") ?? "")).toBe(concept.revision.slice(7));
  expect(files.get(`records/${answer.id}.json`)).toBe(products.journal.sourceText(urn(answer.id)));
  for (const file of first.files) {
    expect(first.metadata["@graph"].find((entity) => entity["@id"] === file.path)).toMatchObject({
      "@type": "File",
      sha256: hash(file.content),
      contentSize: String(Buffer.byteLength(file.content)),
    });
  }
  const graphIds = new Set(first.metadata["@graph"].map((entity) => entity["@id"]));
  const checkReferences = (value: unknown): void => {
    if (Array.isArray(value)) {
      value.forEach(checkReferences);
      return;
    }
    if (!value || typeof value !== "object") return;
    for (const [key, item] of Object.entries(value)) {
      if (key === "@id" && typeof item === "string" && !/^https?:/u.test(item))
        expect(graphIds.has(item), item).toBe(true);
      else checkReferences(item);
    }
  };
  checkReferences(first.metadata["@graph"]);
  expect(
    first.metadata["@graph"]
      .filter((entity) => entity["@type"] === "File")
      .map((entity) => entity["@id"])
      .sort(),
  ).toEqual([...files.keys()].sort());
  expect(first.metadata["@graph"].find((entity) => entity["@id"] === "./")).toMatchObject({
    "@type": "Dataset",
  });
  expect(first.metadata["@graph"].some((entity) => entity["@type"] === "Review")).toBe(true);
  const statistics = JSON.parse(files.get("statistics.json") ?? "null");
  expect(statistics.statistics).toMatchObject({
    sampleCount: 1,
    completed: 1,
    cost: { sampleCount: 0, usd: null },
  });
  expect(statistics.runs).toMatchObject([{ requestedEffort: "low" }]);
  expect(statistics.runs[0]).not.toHaveProperty("reportedEffort");
  const software = first.metadata["@graph"].find(
    (entity) => entity["@id"] === `#software-${scope.runId}`,
  );
  expect(JSON.parse(String(software?.description))).toMatchObject({
    requestedEffort: "low",
  });
  expect((await products.memory.vault.readConcept(concept.id)).revision).toBe(concept.revision);
});

it.each(["different request", "different body"])(
  "rejects a concept falsely attributed to a review with %s",
  async (mismatch) => {
    const { products, scope, evaluation, exportCrate } = await fixture();
    const job = products.journal.append(null, {
      type: EventType.CUSTOM,
      name: "swarmx.memory.review.queued",
      value: { runIds: [scope.runId] },
    });
    const review = products.journal.append(null, {
      type: EventType.CUSTOM,
      name: "swarmx.memory.review.started",
      value: {
        jobId: job.id,
        snapshot: { evidence: products.journal.learningEvidence([scope.runId]), resources: [] },
      },
    });
    const request = {
      requestId: randomUUID(),
      title: "Attributed evaluation",
      description: "Scoped finding",
      type: "Finding" as const,
      body: "Original conclusion.",
      evaluation: { ...evaluation, review: urn(review.id) },
    };
    products.journal.append(null, {
      type: EventType.CUSTOM,
      name: "swarmx.memory.review.planned",
      value: {
        jobId: job.id,
        summary: "Observed one execution.",
        operations: [{ action: "create_memory", request }],
      },
    });
    const concept = await products.memory.vault.createConcept({
      ...request,
      ...(mismatch === "different request"
        ? { requestId: randomUUID() }
        : { body: "A different conclusion." }),
    });
    await expect(
      exportCrate({ id: concept.id, expectedRevision: concept.revision }),
    ).rejects.toThrow("did not generate");
  },
);

it("exports the actual generated concept and a body-preserving update at their recorded request IDs", async () => {
  const { products, scope, evaluation, exportCrate } = await fixture();
  const job = products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.queued",
    value: { runIds: [scope.runId] },
  });
  const first = products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.started",
    value: {
      jobId: job.id,
      snapshot: { evidence: products.journal.learningEvidence([scope.runId]), resources: [] },
    },
  });
  const request = {
    requestId: randomUUID(),
    title: "Generated evaluation",
    description: "Scoped finding",
    type: "Finding" as const,
    body: "\n  Original conclusion.\n",
    evaluation: { ...evaluation, review: urn(first.id) },
  };
  products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.planned",
    value: {
      jobId: job.id,
      summary: "Observed execution.",
      operations: [{ action: "create_memory", request }],
    },
  });
  const concept = await products.memory.vault.createConcept(request);
  const created = await exportCrate({ id: concept.id, expectedRevision: concept.revision });
  expect(
    created.metadata["@graph"].find((entity) => entity["@id"] === "#review")?.result,
  ).toContainEqual({ "@id": "#evaluation" });
  const second = products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.started",
    value: {
      jobId: job.id,
      snapshot: {
        evidence: products.journal.learningEvidence([scope.runId]),
        resources: [],
        concepts: [concept],
      },
    },
  });
  const update = {
    id: concept.id,
    expectedRevision: concept.revision,
    requestId: randomUUID(),
    evaluation: {
      ...evaluation,
      review: urn(second.id),
      limitations: "Updated uncertainty, same body.",
    },
  };
  products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.planned",
    value: {
      jobId: job.id,
      summary: "Refined uncertainty.",
      operations: [{ action: "update_memory", request: update }],
    },
  });
  const updated = await products.memory.vault.updateConcept(update);
  await expect(
    exportCrate({ id: updated.id, expectedRevision: updated.revision }),
  ).resolves.toBeTruthy();
});

it("exports a pending resource proposal as applied only after its real approval and validator succeed", async () => {
  const { products, root, scope, evaluation, exportCrate } = await fixture();
  await mkdir(join(root, ".swarmx"));
  await writeFile(join(root, "writer.md"), "Original instructions.\n");
  await writeFile(
    join(root, ".swarmx/learning.json"),
    JSON.stringify({
      resources: [
        {
          id: "writer",
          kind: "agent",
          path: "writer.md",
          validate: [
            process.execPath,
            "-e",
            "process.exit(require('node:fs').readFileSync(process.argv[1], 'utf8').includes('improved') ? 0 : 1)",
          ],
        },
      ],
    }),
  );
  const resources = await products.learning.resources.snapshot(new AbortController().signal);
  const resource = resources[0];
  if (!resource) throw new Error("Missing resource fixture");
  const job = products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.queued",
    value: { runIds: [scope.runId] },
  });
  const review = products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.started",
    value: {
      jobId: job.id,
      snapshot: { evidence: products.journal.learningEvidence([scope.runId]), resources },
    },
  });
  const operation = {
    action: "update_resource" as const,
    request: {
      id: resource.id,
      expectedRevision: resource.expectedRevision,
      content: "Preserve limitations in improved instructions.\n",
      evaluation: { ...evaluation, review: urn(review.id) },
    },
  };
  products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.planned",
    value: { jobId: job.id, summary: "Propose clearer instructions.", operations: [operation] },
  });
  const proposal = products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.proposed",
    value: { origin: "review", operation, resource, jobId: job.id, operationIndex: 0 },
  });
  products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.finished",
    value: { jobId: job.id, state: "completed" },
  });
  const pending = await exportCrate({ source: urn(review.id) });
  expect(pending.metadata["@graph"].find((entity) => entity["@id"] === "#update-0")).toMatchObject({
    actionStatus: { "@id": "https://schema.org/PotentialActionStatus" },
  });
  expect(await readFile(join(root, "writer.md"), "utf8")).toBe(resource.content);
  await products.learning.decide(proposal.id, "approve");
  const approved = await exportCrate({ source: urn(review.id) });
  expect(approved.metadata["@graph"].find((entity) => entity["@id"] === "#update-0")).toMatchObject(
    { actionStatus: { "@id": "https://schema.org/CompletedActionStatus" } },
  );
  expect(await readFile(join(root, "writer.md"), "utf8")).toBe(operation.request.content);
  expect(approved.files.find((file) => file.path === "resources/0-original.md")?.content).toBe(
    resource.content,
  );
  expect(approved.files.find((file) => file.path === "resources/0-candidate.md")?.content).toBe(
    operation.request.content,
  );
  const decision = products.journal.memoryDecisions([proposal.id])[0];
  expect(approved.files.some((file) => file.path === `records/${decision?.id}.json`)).toBe(true);
});

it("keeps a successful no-op review isolated from a later attempt of the same job", async () => {
  const { products, scope, exportCrate } = await fixture();
  const job = products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.queued",
    value: { runIds: [scope.runId] },
  });
  const start = () =>
    products.journal.append(null, {
      type: EventType.CUSTOM,
      name: "swarmx.memory.review.started",
      value: {
        jobId: job.id,
        snapshot: { evidence: products.journal.learningEvidence([scope.runId]), resources: [] },
      },
    });
  const first = start();
  const response = (source: string, text: string) =>
    products.journal.append(null, {
      type: EventType.CUSTOM,
      name: "swarmx.memory.review.response",
      value: { jobId: job.id, source, text, reviewer: { harness: "codex" } },
    });
  response(urn(first.id), '{"summary":"Insufficient evidence.","operations":[]}');
  products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.planned",
    value: { jobId: job.id, summary: "Insufficient evidence.", operations: [] },
  });
  products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.finished",
    value: { jobId: job.id, state: "completed" },
  });
  const before = await exportCrate({ source: urn(first.id) });
  const next = start();
  response(urn(next.id), "Later attempt response.");
  products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.planned",
    value: { jobId: job.id, summary: "Later attempt.", operations: [] },
  });
  const result = await exportCrate({ source: urn(first.id) });
  expect(result).toEqual(before);
  expect(result.metadata["@graph"].find((entity) => entity["@id"] === "#review")).toMatchObject({
    actionStatus: { "@id": "https://schema.org/CompletedActionStatus" },
    description: expect.stringContaining("Insufficient evidence."),
  });
  expect(result.metadata["@graph"].some((entity) => entity["@type"] === "UpdateAction")).toBe(
    false,
  );
});

it("rejects an export over 8 MiB in UTF-8 bytes without changing the selected concept", async () => {
  const { products, scope, evaluation, exportCrate } = await fixture();
  const raw = products.journal.append(scope, {
    type: EventType.RAW,
    event: { text: "界".repeat(3_000_000) },
  });
  const concept = await products.memory.vault.createConcept({
    title: "Large evidence",
    description: "An observed output",
    type: "Finding",
    body: "The record is too large to package.",
    evaluation: { ...evaluation, evidence: [urn(raw.id)] },
  });
  await expect(exportCrate({ id: concept.id, expectedRevision: concept.revision })).rejects.toThrow(
    "8 MiB",
  );
  expect((await products.memory.vault.readConcept(concept.id)).revision).toBe(concept.revision);
});

it("requires memory.read and rejects stale revisions and foreign sources", async () => {
  const { products, concept, exportCrate } = await fixture();
  await expect(
    exportCrate({ id: concept.id, expectedRevision: `sha256:${"0".repeat(64)}` }),
  ).rejects.toThrow("revision");
  await expect(exportCrate({ source: urn(randomUUID()) })).rejects.toThrow();
  products.updatePolicy({ ...products.settings.read().policy, tools: ["memory.read"] });
  await expect(
    exportCrate({ id: concept.id, expectedRevision: concept.revision }),
  ).resolves.toBeTruthy();
  products.updatePolicy({ ...products.settings.read().policy, tools: [] });
  await expect(exportCrate({ id: concept.id, expectedRevision: concept.revision })).rejects.toThrow(
    "memory.read",
  );
});

it("exports a failed review's exact input and invalid original response without inventing a plan", async () => {
  const { products, scope, exportCrate } = await fixture();
  const job = products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.queued",
    value: { runIds: [scope.runId] },
  });
  const prompt = "Evaluate this evidence.\n原始提示";
  const review = products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.started",
    value: {
      jobId: job.id,
      snapshot: { evidence: products.journal.learningEvidence([scope.runId]), resources: [] },
      prompt,
      promptRevision: `sha256:${hash(prompt)}`,
    },
  });
  products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.response",
    value: {
      jobId: job.id,
      source: urn(review.id),
      text: "not JSON\n保留原文",
      reviewer: { harness: "codex", model: null, version: null },
    },
  });
  products.journal.append(null, {
    type: EventType.CUSTOM,
    name: "swarmx.memory.review.finished",
    value: { jobId: job.id, state: "failed", message: "Invalid JSON" },
  });
  const result = await exportCrate({ source: urn(review.id) });
  const files = new Map(result.files.map(({ path, content }) => [path, content]));
  expect(files.get("review/prompt.txt")).toBe(prompt);
  expect(files.get("review/response.txt")).toBe("not JSON\n保留原文");
  expect(result.metadata["@graph"].find((entity) => entity["@id"] === "#review")).toMatchObject({
    "@type": "CreateAction",
    actionStatus: { "@id": "https://schema.org/FailedActionStatus" },
  });
  expect(result.metadata["@graph"].some((entity) => entity["@type"] === "UpdateAction")).toBe(
    false,
  );
});
