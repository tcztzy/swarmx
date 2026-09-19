import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it } from "vitest";
import { ExecutionJournal } from "../src/host/execution-journal.js";

const cleanups: Array<() => Promise<void>> = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
const parent = { sessionId: "lead", runId: "turn", causedBy: null, attributes: {} };
const caller = { actorId: "lead", callId: "prepare" };
const task = "Write the paper abstract.";

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-preparation-"));
  const journal = new ExecutionJournal(root, "workspace");
  cleanups.push(async () => {
    journal.close();
    await rm(root, { recursive: true, force: true });
  });
  return { root, journal };
}

it("requires completed preparation for the exact task, session, run and directory", async () => {
  const { root, journal } = await fixture();
  const paused = Promise.withResolvers<void>();
  let preparationId = "";
  const preparation = journal.scope.run(parent, () =>
    journal.tool("swarm", { action: "prepare", task }, caller, async () => {
      preparationId = journal.scope.getStore()?.causedBy ?? "";
      await paused.promise;
      return { action: "prepare", task, preparationId };
    }),
  );
  expect(preparationId).not.toBe("");
  expect(journal.hasDelegationPreparation(preparationId, "lead", "turn", task)).toBe(false);
  paused.resolve();
  await preparation;
  expect(journal.hasDelegationPreparation(preparationId, "lead", "turn", task)).toBe(true);
  expect(journal.hasDelegationPreparation(preparationId, "lead", "next-turn", task)).toBe(false);
  expect(journal.hasDelegationPreparation(preparationId, "sibling", "turn", task)).toBe(false);
  expect(journal.hasDelegationPreparation(preparationId, "lead", "turn", `${task} More.`)).toBe(
    false,
  );
  expect(journal.hasDelegationPreparation("unknown", "lead", "turn", task)).toBe(false);
  const reopened = new ExecutionJournal(root, "workspace");
  const other = new ExecutionJournal(root, "other-workspace");
  try {
    expect(reopened.hasDelegationPreparation(preparationId, "lead", "turn", task)).toBe(true);
    expect(other.hasDelegationPreparation(preparationId, "lead", "turn", task)).toBe(false);
  } finally {
    reopened.close();
    other.close();
  }
});

it("does not accept failed preparation or a different product tool/action", async () => {
  const { journal } = await fixture();
  let failedId = "";
  await expect(
    journal.scope.run(parent, () =>
      journal.tool("swarm", { action: "prepare", task }, caller, async () => {
        failedId = journal.scope.getStore()?.causedBy ?? "";
        throw new Error("Memory read failed");
      }),
    ),
  ).rejects.toThrow("Memory read failed");
  expect(journal.hasDelegationPreparation(failedId, "lead", "turn", task)).toBe(false);
  for (const [name, action] of [
    ["memory", "prepare"],
    ["swarm", "status"],
  ]) {
    const { preparationId } = await journal.scope.run(parent, () =>
      journal.tool(name, { action, task }, caller, async () => ({
        action,
        task,
        preparationId: journal.scope.getStore()?.causedBy ?? "",
      })),
    );
    expect(journal.hasDelegationPreparation(preparationId, "lead", "turn", task)).toBe(false);
  }
});
