import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import { HARNESS_CAPABILITIES, type NativeAgent, type Observer } from "../src/agents/types.js";
import { ResourceSnapshotSchema } from "../src/host/learning-resources.js";
import { AgentMemory, type MemoryReviewer } from "../src/host/memory.js";
import { ProductServices } from "../src/host/product-services.js";
import { recordedAgent } from "../src/host/recorded-agent.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
const sink: Observer = { text() {}, tool() {}, raw() {}, interact: async () => undefined };
const original = "---\nname: research\ndescription: Research method\n---\nRead evidence.\n";
const updated =
  "---\nname: research\ndescription: Research method\n---\nCheck original evidence before reporting a finding.\n";

function resourcePlan(prompt: string) {
  const input = prompt.split("\n\nSnapshot: ")[1];
  assert.ok(input);
  const [resource] = ResourceSnapshotSchema.array().parse(JSON.parse(input).resources);
  assert.ok(resource);
  return {
    summary: "The observed task supports checking original evidence before reporting findings.",
    operations: [
      {
        action: "update_resource",
        request: { id: resource.id, expectedRevision: resource.expectedRevision, content: updated },
      },
    ],
  };
}

async function fixture(
  reviewer = vi.fn<MemoryReviewer>(async (prompt) => JSON.stringify(resourcePlan(prompt))),
  validator = `
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
const content = await readFile(process.argv[2], 'utf8');
assert.match(content, /Check original evidence before reporting a finding/);
await writeFile('validated', content);
`,
) {
  const root = await mkdtemp(join(tmpdir(), "swarmx-learning-evolution-"));
  const options = { cwd: root, productHome: join(root, "product") };
  await mkdir(join(root, ".swarmx"));
  await mkdir(join(root, ".pi/skills/research"), { recursive: true });
  const target = join(root, ".pi/skills/research/SKILL.md");
  await writeFile(target, original);
  await writeFile(join(root, ".swarmx/validate.mjs"), validator);
  await writeFile(
    join(root, ".swarmx/learning.json"),
    JSON.stringify({
      resources: [
        {
          id: "research",
          kind: "skill",
          path: ".pi/skills/research/SKILL.md",
          validate: [process.execPath, ".swarmx/validate.mjs"],
        },
      ],
    }),
  );
  const products = await ProductServices.create(options);
  products.settings.writeMemory({ reviewInterval: 1 });
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
  const native: NativeAgent = {
    name: "fixture",
    capabilities: HARNESS_CAPABILITIES.codex,
    models: async () => ({ models: [], current: {} }),
    list: async () => [],
    create: async () => "codex:evolution",
    read: async () => {},
    start: async (_id, _text, observer) => {
      observer.text("answer", "Checked the original source and corrected the unsupported finding.");
      return { stopReason: "end_turn" };
    },
    steer: async () => {},
    interrupt: async () => {},
    dispose: async () => {},
  };
  const agent = recordedAgent(products.journal, "codex", native, memory);
  return { root, target, products, memory, reviewer, agent };
}

it("one execution review updates a real skill and selection memory, then records a reasoned no-op", async () => {
  const noChange = "The second execution supplies no additional durable learning.";
  const reviewer = vi
    .fn<MemoryReviewer>()
    .mockImplementationOnce(async (prompt) =>
      JSON.stringify({
        ...resourcePlan(prompt),
        operations: [
          ...resourcePlan(prompt).operations,
          {
            action: "create_memory",
            request: {
              title: "Fixture Agent Selection",
              description: "Observed fixture task behavior",
              type: "Finding",
              body: "The fixture Codex agent checked the original evidence on this research task.",
              tags: ["agent-selection"],
            },
          },
        ],
      }),
    )
    .mockResolvedValue(JSON.stringify({ summary: noChange, operations: [] }));
  const { root, target, products, memory, agent } = await fixture(reviewer);
  await agent.start("codex:first", "Correct the unsupported finding", sink);
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("completed"));
  expect(await readFile(target, "utf8")).toBe(updated);
  expect(await readFile(join(root, "validated"), "utf8")).toBe(updated);
  const selection = await memory.selection(["research"], new AbortController().signal);
  expect(JSON.stringify(selection.loaded)).toContain("Fixture Agent Selection");
  expect(JSON.stringify(selection.loaded)).toContain("Source execution snapshot");
  expect(products.journal.pendingLearningRuns()).toEqual([]);
  await agent.start("codex:second", "Confirm the corrected finding", sink);
  await vi.waitFor(async () =>
    expect((await memory.status()).review).toMatchObject({
      state: "completed",
      summary: noChange,
      message: "0",
    }),
  );
  expect(reviewer).toHaveBeenCalledTimes(2);
  expect(await readFile(target, "utf8")).toBe(updated);
});

it("stages resource updates until approval, and rejects approval after filesystem authority is revoked", async () => {
  const { target, products, memory, agent } = await fixture();
  products.settings.writeMemory({ reviewInterval: 1, writeApproval: true });
  await agent.start("codex:stage", "Improve the research procedure", sink);
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("completed"));
  const [pending] = (await memory.status()).pending;
  assert.ok(pending);
  expect(pending.operation.action).toBe("update_resource");
  expect(await readFile(target, "utf8")).toBe(original);
  const settings = products.settings.read();
  products.settings.write({ ...settings, policy: { ...settings.policy, filesystem: "read-only" } });
  await expect(memory.decide(pending.id, "approve")).rejects.toThrow("workspace-write");
  expect((await memory.status()).pending.map(({ id }) => id)).toEqual([pending.id]);
  expect(await readFile(target, "utf8")).toBe(original);
  products.settings.write(settings);
  await memory.decide(pending.id, "approve");
  expect(await readFile(target, "utf8")).toBe(updated);
  expect((await memory.status()).pending).toEqual([]);
});

it("disabling memory aborts a paused validator and retains the unfinished review job", async () => {
  const { root, target, products, memory, agent } = await fixture(
    undefined,
    `
import { writeFile } from 'node:fs/promises';
await writeFile('validator-started', 'yes');
setInterval(() => {}, 1000);
`,
  );
  await agent.start("codex:paused", "Improve the research procedure", sink);
  await expect
    .poll(async () => readFile(join(root, "validator-started"), "utf8").catch(() => ""))
    .toBe("yes");
  const queued = products.journal.pendingMemoryReview();
  assert.ok(queued);
  await memory.call(
    {
      action: "memory_configure",
      request: { ...products.settings.readMemory(), enabled: false },
    },
    { actorId: "renderer", callId: randomUUID(), signal: new AbortController().signal },
  );
  await vi.waitFor(async () => expect((await memory.status()).review.state).toBe("failed"));
  expect(await readFile(target, "utf8")).toBe(original);
  expect(products.journal.pendingMemoryReview()?.id).toBe(queued.id);
  expect(products.journal.pendingLearningRuns()).toHaveLength(1);
  expect(
    products.journal
      .memoryJobEvents(queued.id)
      .some(({ event }) => event.type === "CUSTOM" && event.name === "swarmx.memory.saved"),
  ).toBe(false);
});
