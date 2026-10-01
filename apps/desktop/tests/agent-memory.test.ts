import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import type { NativeAgent, Observer } from "../src/agents/types.js";
import { HARNESS_CAPABILITIES } from "../src/agents/types.js";
import { ExecutionJournal } from "../src/host/execution-journal.js";
import { AgentMemory, type MemoryReviewer } from "../src/host/memory.js";
import { ProductServices } from "../src/host/product-services.js";
import { recordedAgent } from "../src/host/recorded-agent.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
const sink: Observer = { text() {}, tool() {}, raw() {}, interact: async () => undefined };
const context = { actorId: "model", callId: "test-call", signal: new AbortController().signal };
async function fixture(
  reviewer: MemoryReviewer = async () => '{"summary":"No new evidence.","operations":[]}',
) {
  const root = await mkdtemp(join(tmpdir(), "swarmx-learning-"));
  const options = {
    productHome: join(root, "product"),
    cwd: root,
  };
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
  const start = vi.fn<NativeAgent["start"]>(async (_session, _text, observer) => {
    observer.text("reply", "科研", "assistant");
    observer.text("reply", "方案 verified", "assistant");
    return { stopReason: "end_turn" };
  });
  const native: NativeAgent = {
    name: "Fixture",
    capabilities: HARNESS_CAPABILITIES.codex,
    models: async () => ({ models: [], current: {} }),
    list: async () => [],
    create: async () => "codex:session",
    read: async () => {},
    start,
    steer: async () => {},
    interrupt: async () => {},
    dispose: async () => {},
  };
  return {
    root,
    options,
    products,
    memory,
    start,
    agent: recordedAgent(products.journal, "codex", native, memory),
  };
}

it("loads the full Memory guide on demand and for reviews without injecting it into sessions", async () => {
  const request = {
    title: "ふるさと納税",
    description: "A checklist for Japan's local government donation program.",
    type: "Playbook",
    tags: ["Japan", "taxation"],
    body: "# ふるさと納税\n\n先核对官方资料。Keep the source links. 手順を確認する。",
  };
  const reviewer = vi.fn<MemoryReviewer>().mockResolvedValue(
    JSON.stringify({
      summary: "Save the demonstrated workflow.",
      operations: [{ action: "create_memory", request }],
    }),
  );
  const { agent, start, memory, products } = await fixture(reviewer);
  const indexSnapshot = vi.spyOn(products.memory.vault, "indexSnapshot");
  await agent.start("codex:authoring", "Save the research", sink);
  const instructions = start.mock.calls[0]?.[3]?.instructions;
  const description = products.toolManifest.find((tool) => tool.name === "memory")?.description;
  expect(instructions).toContain("read_memory_guide");
  expect(instructions).not.toContain("Store durable user- or research-specific knowledge");
  expect(description).toContain("read_memory_guide");
  expect(description).not.toContain("Store durable user- or research-specific knowledge");
  expect(indexSnapshot).not.toHaveBeenCalled();
  const guide = (await memory.call({ action: "read_memory_guide", request: {} }, context)) as {
    action: string;
    data: string;
  };
  expect(guide.action).toBe("read_memory_guide");
  expect(guide.data).toBe(
    await readFile(new URL(import.meta.resolve("@swarmx/memory/skills/memory/SKILL.md")), "utf8"),
  );
  await expect(
    memory.call({ action: "read_memory_guide", request: { extra: true } }, context),
  ).rejects.toThrow();
  memory.review("codex:authoring");
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("completed"));

  for (const fullGuide of [guide.data, reviewer.mock.calls[0]?.[0]]) {
    expect(fullGuide).toContain(
      "Store durable user- or research-specific knowledge that adds value beyond public sources",
    );
    expect(fullGuide).toContain(
      "if there is no durable added value, do not save an encyclopedia summary",
    );
    expect(fullGuide).toContain(
      "Do not put migration notes, curation history, source-scope bookkeeping or self-commentary in memory content",
    );
    expect(fullGuide).toContain(
      "Do not create standalone current concepts or first-level index/navigation/disambiguation entries for merged or obsolete topics",
    );
    expect(fullGuide).toContain(
      "These are Memory operating rules, not user preferences; do not copy them into core notes or vault concepts",
    );
    expect(fullGuide).toContain("Use concise, unambiguous concept titles");
    expect(fullGuide).toContain("without hashes, UUIDs or timestamps");
    expect(fullGuide).toContain("Store concept pages directly at the memory root");
    expect(fullGuide).toContain("Do not add folders");
    expect(fullGuide).toContain("no separate Markdown change log");
    expect(fullGuide).toContain("Write natural-language Memory metadata in American English");
    expect(fullGuide).toContain("language community, culture or institution");
    expect(fullGuide).toContain("法定节假日调休, ふるさと納税, 전세");
    expect(fullGuide).toContain("Body content may use any language or mix languages");
    expect(fullGuide).toContain("Preserve IDs, URLs, hashes, timestamps");
  }

  const [entry] = (await products.memory.vault.search({ query: request.title })).items;
  expect(entry).toBeDefined();
  if (!entry) throw new Error("The reviewed concept is missing.");
  const concept = await products.memory.vault.readConcept(entry.id);
  expect(concept.metadata).toMatchObject({
    title: request.title,
    description: request.description,
    tags: request.tags,
  });
  expect(concept.body.trim()).toBe(request.body);
});

it("freezes notes per session across reloads while new sessions receive updates", async () => {
  const { agent, start, memory, products, options } = await fixture();
  const empty = await memory.core.read();
  await memory.call(
    {
      action: "update_core_memory",
      request: { content: "Prefer Chinese", expectedRevision: empty.revision },
    },
    { actorId: "user", callId: "note-one", signal: new AbortController().signal },
  );
  await agent.start("codex:one", "First task", sink);
  const first = start.mock.calls[0]?.[3]?.instructions;
  expect(first).toContain("Prefer Chinese");
  const saved = await memory.core.read();
  await memory.call(
    {
      action: "update_core_memory",
      request: { content: "Prefer English", expectedRevision: saved.revision },
    },
    { actorId: "user", callId: "note-two", signal: new AbortController().signal },
  );
  const reopened = new AgentMemory(
    options,
    products.memory,
    products.journal,
    products.settings,
    async () => '{"summary":"No new evidence.","operations":[]}',
  );
  expect(await reopened.snapshot("codex:one")).toBe(first);
  await agent.start("codex:two", "Second task", sink);
  expect(start.mock.calls[1]?.[3]?.instructions).toContain("Prefer English");
  expect(products.journal.recall({ query: "First task" })[0]?.text).toBe("First task");
  await reopened.close();
});

it("recalls complete streamed Chinese messages with original event references and directory isolation", async () => {
  const { agent, products, options } = await fixture();
  await agent.start("codex:one", "记住科研方案", sink);
  const matches = products.journal.recall({ query: "科研方案" });
  expect(matches.map(({ text }) => text)).toEqual(["科研方案 verified", "记住科研方案"]);
  expect(products.journal.recall({ query: "科研" })).toHaveLength(2);
  expect(products.journal.recall({ query: '" OR malicious*' })).toEqual([]);
  const events = products.journal.read().events;
  expect(matches.every(({ eventId }) => events.some(({ id }) => id === eventId))).toBe(true);
  const other = new ExecutionJournal(join(options.productHome, "logs"), "other");
  try {
    expect(other.recall({ query: "科研" })).toEqual([]);
  } finally {
    other.close();
  }
  expect(products.journal.recall({ query: "科研方案" })).toEqual(matches);
});

it("stages writes durably, rejects self-approval, and retains a conflicting proposal for review", async () => {
  const { products, memory, options } = await fixture();
  products.settings.writeMemory({ ...products.settings.readMemory(), writeApproval: true });
  const empty = await memory.core.read();
  const operation = {
    action: "update_core_memory",
    request: { content: "Prefer Chinese", expectedRevision: empty.revision },
  };
  await expect(memory.call({ ...operation, approved: true }, context)).rejects.toThrow();
  const proposed = await memory.call(operation, context);
  expect(proposed).toMatchObject({ staged: true });
  expect((await memory.core.read()).content).toBe("");
  const reopened = new AgentMemory(
    options,
    products.memory,
    products.journal,
    products.settings,
    async () => '{"summary":"No new evidence.","operations":[]}',
  );
  const pending = (await reopened.status()).pending[0];
  if (!pending) throw new Error("Missing pending memory change");
  await reopened.decide(pending.id, "approve");
  expect((await memory.core.read()).content).toBe("Prefer Chinese");
  expect((await reopened.status()).pending).toEqual([]);
  await expect(reopened.decide(pending.id, "approve")).rejects.toThrow("not found");
  await memory.call(operation, context);
  const conflict = (await memory.status()).pending[0];
  if (!conflict) throw new Error("Missing conflicting memory change");
  await expect(memory.decide(conflict.id, "approve")).rejects.toThrow("changed");
  expect((await memory.status()).pending).toHaveLength(1);
  await memory.decide(conflict.id, "reject");
  await reopened.close();
});

it("triggers a real review boundary at the configured threshold and records validated writes", async () => {
  const reviewer = vi.fn<MemoryReviewer>();
  const { agent, memory, products } = await fixture(reviewer);
  const note = await memory.core.read();
  reviewer.mockResolvedValue(
    JSON.stringify({
      summary: "Save an explicit preference.",
      operations: [
        {
          action: "update_core_memory",
          request: {
            content: "Use reproducible environments",
            expectedRevision: note.revision,
          },
        },
      ],
    }),
  );
  products.settings.writeMemory({ ...products.settings.readMemory(), reviewInterval: 2 });
  await agent.start("codex:one", "First", sink);
  expect(reviewer).not.toHaveBeenCalled();
  await agent.start("codex:one", "Second", sink);
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("completed"));
  expect(reviewer).toHaveBeenCalledTimes(1);
  expect((await memory.core.read()).content).toBe("Use reproducible environments");
  expect(products.journal.memoryEvent("swarmx.memory.saved")?.event).toMatchObject({
    value: { origin: "review" },
  });
});

it("surfaces invalid review output and never treats a cancelled review as a successful save", async () => {
  const reviewer = vi.fn<MemoryReviewer>().mockResolvedValue("not JSON");
  const { agent, memory } = await fixture(reviewer);
  await agent.start("codex:one", "Work", sink);
  memory.review("codex:one");
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("failed"));
  expect((await memory.core.read()).content).toBe("");
  reviewer.mockImplementation(
    async (_prompt, signal) =>
      new Promise((_resolve, reject) =>
        signal.addEventListener("abort", () => reject(signal.reason), { once: true }),
      ),
  );
  memory.review("codex:one");
  await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(2));
  await memory.close();
  expect((await memory.status()).review.state).toBe("failed");
});
