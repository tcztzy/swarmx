import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import type { AgentId } from "../src/agent.js";
import { HARNESS_CAPABILITIES, type NativeAgent } from "../src/agents/types.js";
import { ProductServices } from "../src/host/product-services.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});

async function fixture(harness: AgentId = "codex") {
  const root = await mkdtemp(join(tmpdir(), "swarmx-selection-"));
  const products = await ProductServices.create({ productHome: join(root, "product"), cwd: root });
  let next = 0;
  const native: NativeAgent = {
    name: harness,
    capabilities: HARNESS_CAPABILITIES[harness],
    create: vi.fn(async () => `${harness}:${++next}`),
    models: vi.fn(async () => ({
      models: harness === "dsh" ? [] : [{ id: "small", name: "Small", efforts: [] }],
      current: {},
    })),
    start: vi.fn(async () => ({ stopReason: "end_turn" as const })),
    list: async () => [],
    read: async () => {},
    steer: async () => {},
    interrupt: async () => {},
    dispose: async () => {},
  };
  await products.attachAgents("http://localhost", native, harness);
  cleanups.push(async () => {
    await products.dispose();
    await rm(root, { recursive: true, force: true });
  });
  const parent = { sessionId: "codex:lead", runId: "lead-run", causedBy: null, attributes: {} };
  const call = (args: unknown) =>
    products.callTool("swarm", args, {
      actorId: "test",
      callId: `call-${++next}`,
      signal: new AbortController().signal,
    });
  return { products, native, call, parent };
}

it("prepares admitted candidates, bundled knowledge and complete private evidence before choosing", async () => {
  const { products, native, call } = await fixture();
  products.updatePolicy({ ...products.settings.read().policy, harnesses: { codex: ["small"] } });
  const note = await products.learning.core.read();
  await products.learning.core.update({
    content: "Prefer inexpensive providers",
    expectedRevision: note.revision,
  });
  const evidence = await products.memory.vault.createConcept({
    title: "Provider incident",
    description: "Observed latency",
    type: "Finding",
    body: "The measured request took 90 seconds.",
  });
  const choice = await products.memory.vault.createConcept({
    title: "Provider selection",
    description: "OpenCode experience",
    type: "Finding",
    tags: ["agent-selection"],
    body: "Avoid this OpenCode deployment for time-sensitive writing.",
    dependencies: [{ id: evidence.id, revision: evidence.revision }],
  });
  const prepared = await call({ action: "prepare", task: "Write a paper", queries: ["OpenCode"] });
  expect(prepared).toMatchObject({
    action: "prepare",
    task: "Write a paper",
    preparationId: expect.any(String),
    candidates: [{ agentId: "codex", allowedModels: ["small"] }],
    knowledge: {
      content: expect.stringContaining("sdk-minimal"),
      revision: expect.stringMatching(/^sha256:/u),
    },
    memory: {
      status: "available",
      note: { content: "Prefer inexpensive providers" },
      loaded: [
        expect.objectContaining({
          concepts: [
            expect.objectContaining({ id: evidence.id, body: evidence.body }),
            expect.objectContaining({ id: choice.id, body: choice.body }),
          ],
        }),
      ],
    },
  });
  expect(native.models).not.toHaveBeenCalled();
  expect(native.create).not.toHaveBeenCalled();
  await expect(call({ action: "models", agentId: "codex" })).resolves.toMatchObject({
    models: [{ id: "small" }],
  });
});

it.each(["disabled", "not_permitted"])(
  "prepares without reading private memory when %s",
  async (status) => {
    const { products, call } = await fixture();
    if (status === "disabled")
      products.settings.writeMemory({ ...products.settings.readMemory(), enabled: false });
    else products.updatePolicy({ ...products.settings.read().policy, tools: [] });
    const read = vi.spyOn(products.learning.core, "read");
    const search = vi.spyOn(products.memory.vault, "search");
    const result = await call({ action: "prepare", task: "Write", queries: ["writing"] });
    expect(result).toMatchObject({
      memory: { status },
      knowledge: { content: expect.any(String) },
    });
    expect(read).not.toHaveBeenCalled();
    expect(search).not.toHaveBeenCalled();
  },
);

it("refreshes selection memory and reports stale dependencies and oversized omissions", async () => {
  const { products, call } = await fixture();
  const evidence = await products.memory.vault.createConcept({
    title: "Deployment latency",
    description: "Measured request latency",
    type: "Finding",
    body: "First measurement.",
  });
  const choice = await products.memory.vault.createConcept({
    title: "Writing route",
    description: "Writing provider choice",
    type: "Finding",
    tags: ["agent-selection"],
    body: "Use the deployment when latency permits.",
    dependencies: [{ id: evidence.id, revision: evidence.revision }],
  });
  const prepare = { action: "prepare", task: "Write", queries: ["Writing route"] };
  await call(prepare);
  const updated = await products.memory.vault.updateConcept({
    id: evidence.id,
    expectedRevision: evidence.revision,
    body: "A newer measurement invalidates the old comparison.",
  });
  const note = await products.learning.core.read();
  await products.learning.core.update({
    content: "New cost preference",
    expectedRevision: note.revision,
  });
  const result = await call(prepare);
  expect(result).toMatchObject({
    memory: {
      note: { content: "New cost preference" },
      loaded: [
        expect.objectContaining({
          concepts: [
            expect.objectContaining({ id: updated.id, revision: updated.revision }),
            expect.objectContaining({ id: choice.id }),
          ],
          graph: expect.objectContaining({
            nodes: expect.arrayContaining([
              expect.objectContaining({ id: choice.id, stale: true }),
            ]),
          }),
        }),
      ],
    },
  });
  const oversized = await products.memory.vault.createConcept({
    title: "Large route report",
    description: "Long observations",
    type: "Finding",
    body: "Evidence ".repeat(6000),
  });
  await expect(call({ ...prepare, queries: ["Large route report"] })).resolves.toMatchObject({
    memory: { omitted: [oversized.id] },
  });
});

it("requires completed same-run task preparation and a reason before creating a delegated session", async () => {
  const { products, native, call, parent } = await fixture();
  const send = {
    action: "send_message",
    agentId: "codex",
    text: "Write the methods",
    model: "small",
  };
  let preparationId = "";
  await products.journal.scope.run(parent, async () => {
    await expect(call(send)).rejects.toThrow("swarm.prepare");
    expect(native.create).not.toHaveBeenCalled();
    const prepared = (await call({ action: "prepare", task: send.text, queries: ["writing"] })) as {
      preparationId: string;
    };
    preparationId = prepared.preparationId;
    await expect(call({ ...send, preparationId })).rejects.toThrow("reason");
    await expect(
      call({ ...send, preparationId, reason: " ", text: "Different task" }),
    ).rejects.toThrow();
    await expect(
      call({ ...send, preparationId, reason: "Fits the writing task", text: "Different task" }),
    ).rejects.toThrow("swarm.prepare");
    expect(native.create).not.toHaveBeenCalled();
    await expect(
      call({
        ...send,
        preparationId,
        reason: "User requested this model; the provider note is not applicable.",
      }),
    ).resolves.toMatchObject({ result: { stopReason: "end_turn" } });
  });
  expect(native.start).toHaveBeenCalledOnce();
  const records = products.journal.read({ run: parent.runId }).events;
  expect(JSON.stringify(records)).toContain(
    "User requested this model; the provider note is not applicable.",
  );
  await products.journal.scope.run({ ...parent, runId: "next-turn" }, async () => {
    await expect(call({ ...send, preparationId, reason: "Reused old context" })).rejects.toThrow(
      "swarm.prepare",
    );
  });
  expect(native.start).toHaveBeenCalledOnce();
});

it("does not dispatch after cancelled preparation and preserves unscoped external calls", async () => {
  const { products, native, call, parent } = await fixture();
  const pause = Promise.withResolvers<void>();
  const entered = Promise.withResolvers<void>();
  const search = products.memory.vault.search.bind(products.memory.vault);
  vi.spyOn(products.memory.vault, "search").mockImplementation(async (request) => {
    entered.resolve();
    await pause.promise;
    return search(request);
  });
  const controller = new AbortController();
  const preparation = products.journal.scope.run(parent, () =>
    products.callTool(
      "swarm",
      { action: "prepare", task: "Work", queries: ["writing"] },
      { actorId: "test", callId: "paused", signal: controller.signal },
    ),
  );
  await entered.promise;
  const preparationId = products.journal
    .read({ run: parent.runId })
    .events.find(
      ({ event }) => event.type === "TOOL_CALL_START" && event.toolCallId === "paused",
    )?.id;
  expect(preparationId).toBeDefined();
  const send = {
    action: "send_message",
    agentId: "codex",
    text: "Work",
    preparationId,
    reason: "The selected model fits this task.",
  };
  await products.journal.scope.run(parent, async () => {
    await expect(call(send)).rejects.toThrow("swarm.prepare");
  });
  expect(native.create).not.toHaveBeenCalled();
  expect(native.start).not.toHaveBeenCalled();
  controller.abort();
  pause.resolve();
  await expect(preparation).rejects.toThrow();
  await products.journal.scope.run(parent, async () => {
    await expect(call(send)).rejects.toThrow("swarm.prepare");
  });
  expect(native.create).not.toHaveBeenCalled();
  expect(native.start).not.toHaveBeenCalled();
  await expect(
    call({ action: "send_message", agentId: "codex", text: "External work" }),
  ).resolves.toMatchObject({ result: { stopReason: "end_turn" } });
});

it("passes DSH choices through Host admission without inventing a model catalog", async () => {
  const { products, native, call } = await fixture("dsh");
  products.updatePolicy({
    ...products.settings.read().policy,
    harnesses: { dsh: ["deepseek-official/example-model"] },
  });
  const request = {
    action: "send_message",
    agentId: "swarm",
    text: "Work",
    model: "deepseek-official/example-model",
    effort: "high",
    profile: "sdk-minimal",
  };
  await expect(call(request)).resolves.toMatchObject({ result: { stopReason: "end_turn" } });
  expect(native.start).toHaveBeenCalledWith(
    expect.any(String),
    "Work",
    expect.any(Object),
    expect.objectContaining({ model: request.model, effort: "high", profile: "sdk-minimal" }),
  );
  await expect(call({ ...request, model: "other/example-model" })).rejects.toThrow(
    "permitted model",
  );
  expect(native.start).toHaveBeenCalledOnce();
});

it("rejects DSH profiles on another harness", async () => {
  const { native, call } = await fixture();
  await expect(
    call({ action: "send_message", agentId: "swarm", text: "Work", profile: "sdk-minimal" }),
  ).rejects.toThrow("DSH");
  expect(native.start).not.toHaveBeenCalled();
});
