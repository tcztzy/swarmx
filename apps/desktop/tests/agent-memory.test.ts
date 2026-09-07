import { mkdtemp, rm } from "node:fs/promises";
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
async function fixture(reviewer: MemoryReviewer = async () => '{"operations":[]}') {
  const root = await mkdtemp(join(tmpdir(), "swarmx-learning-"));
  const options = {
    productHome: join(root, "product"),
    workspace: { id: "123456abcdef", label: "Research", root },
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

it("freezes notes per session across reloads while new sessions receive updates", async () => {
  const { agent, start, memory, products, options } = await fixture();
  const empty = await memory.core.read("user");
  await memory.edit({
    action: "update_core_memory",
    request: { target: "user", content: "Prefer Chinese", expectedRevision: empty.revision },
  });
  await agent.start("codex:one", "First task", sink);
  const first = start.mock.calls[0]?.[3]?.instructions;
  expect(first).toContain("Prefer Chinese");
  const saved = await memory.core.read("user");
  await memory.edit({
    action: "update_core_memory",
    request: { target: "user", content: "Prefer English", expectedRevision: saved.revision },
  });
  const reopened = new AgentMemory(
    options,
    products.memory,
    products.journal,
    products.settings,
    async () => '{"operations":[]}',
  );
  expect(await reopened.snapshot("codex:one")).toBe(first);
  await agent.start("codex:two", "Second task", sink);
  expect(start.mock.calls[1]?.[3]?.instructions).toContain("Prefer English");
  expect(products.journal.recall({ query: "First task" })[0]?.text).toBe("First task");
  await reopened.close();
});

it("recalls complete streamed Chinese messages with original event references and workspace isolation", async () => {
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
  const empty = await memory.core.read("user");
  const operation = {
    action: "update_core_memory",
    request: { target: "user", content: "Prefer Chinese", expectedRevision: empty.revision },
  };
  await expect(memory.call({ ...operation, approved: true }, context)).rejects.toThrow();
  const proposed = await memory.call(operation, context);
  expect(proposed).toMatchObject({ staged: true });
  expect((await memory.core.read("user")).content).toBe("");
  const reopened = new AgentMemory(
    options,
    products.memory,
    products.journal,
    products.settings,
    async () => '{"operations":[]}',
  );
  const pending = (await reopened.status()).pending[0];
  if (!pending) throw new Error("Missing pending memory change");
  await reopened.decide(pending.id, "approve");
  expect((await memory.core.read("user")).content).toBe("Prefer Chinese");
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
  const note = await memory.core.read("workspace");
  reviewer.mockResolvedValue(
    JSON.stringify({
      operations: [
        {
          action: "update_core_memory",
          request: {
            target: "workspace",
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
  expect((await memory.core.read("workspace")).content).toBe("Use reproducible environments");
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
  expect((await memory.core.read("user")).content).toBe("");
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
