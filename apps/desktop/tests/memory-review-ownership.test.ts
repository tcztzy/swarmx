import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { once } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { afterEach, expect, it, vi } from "vitest";
import { ExecutionJournal } from "../src/host/execution-journal.js";
import { AgentMemory, type MemoryReviewer } from "../src/host/memory.js";
import { ProductServices } from "../src/host/product-services.js";

const unchanged =
  '{"summary":"Observed execution does not warrant a durable change.","operations":[]}';
const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});

async function fixture(firstReviewer: MemoryReviewer, secondReviewer = firstReviewer) {
  const root = await mkdtemp(join(tmpdir(), "swarmx-review-ownership-"));
  cleanups.push(() => rm(root, { recursive: true, force: true }));
  const options = { cwd: root, productHome: join(root, "product") };
  const consumers = [];
  for (const reviewer of [firstReviewer, secondReviewer]) {
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
    });
    consumers.push({ products, memory });
  }
  const [first, second] = consumers;
  if (!first || !second) throw new Error("Missing review consumer");
  first.products.settings.writeMemory({
    ...first.products.settings.readMemory(),
    reviewInterval: 1,
  });
  const context = {
    sessionId: "codex:observed",
    runId: randomUUID(),
    causedBy: null,
    attributes: {
      "swarmx.memory.review_eligible": true,
      "swarmx.memory.review_permissions": JSON.stringify(first.memory.reviewPermissions()),
    },
  };
  first.products.journal.append(context, {
    type: EventType.RUN_STARTED,
    threadId: context.sessionId,
    runId: context.runId,
  });
  const terminal = first.products.journal.append(context, {
    type: EventType.RUN_FINISHED,
    threadId: context.sessionId,
    runId: context.runId,
    result: { stopReason: "end_turn" },
  });
  return { first, second, terminal };
}

it("lets only one Host review the same durable job across separate connections", async () => {
  const gate = Promise.withResolvers<string>();
  const reviewer = vi.fn<MemoryReviewer>(() => gate.promise);
  const { first, second, terminal } = await fixture(reviewer);
  try {
    first.memory.resume();
    await vi.waitFor(() => expect(reviewer).toHaveBeenCalledTimes(1));
    const queued = first.products.journal.pendingMemoryReview();
    if (!queued) throw new Error("Missing persistent review job");
    second.memory.resume();
    await Promise.resolve();
    await vi.waitFor(() =>
      expect(!second.memory.busy || reviewer.mock.calls.length > 1).toBe(true),
    );
    expect(reviewer).toHaveBeenCalledTimes(1);
    expect(second.products.journal.pendingLearningRuns().map(({ id }) => id)).toEqual([
      terminal.id,
    ]);
    gate.resolve(unchanged);
    await vi.waitFor(() => expect(first.products.journal.pendingMemoryReview()).toBeUndefined());
    second.memory.resume();
    await Promise.resolve();
    await vi.waitFor(() => expect(second.memory.busy).toBe(false));
    expect(reviewer).toHaveBeenCalledTimes(1);
    expect(second.products.journal.pendingLearningRuns()).toEqual([]);
    expect(
      first.products.journal
        .memoryJobEvents(queued.id)
        .filter(
          ({ event }) =>
            event.type === EventType.CUSTOM &&
            event.name === "swarmx.memory.review.finished" &&
            event.value.state === "completed",
        ),
    ).toHaveLength(1);
  } finally {
    gate.resolve(unchanged);
  }
});

it("holds ownership during cancellation cleanup then lets another Host retry the saved job", async () => {
  const gate = Promise.withResolvers<string>();
  const firstReviewer = vi.fn<MemoryReviewer>(() => gate.promise);
  const secondReviewer = vi.fn<MemoryReviewer>().mockResolvedValue(unchanged);
  const { first, second, terminal } = await fixture(firstReviewer, secondReviewer);
  try {
    first.memory.resume();
    await vi.waitFor(() => expect(firstReviewer).toHaveBeenCalledTimes(1));
    const queued = first.products.journal.pendingMemoryReview();
    if (!queued) throw new Error("Missing persistent review job");
    const closing = first.memory.close();
    expect(firstReviewer.mock.calls[0]?.[1].aborted).toBe(true);
    second.memory.resume();
    await Promise.resolve();
    await vi.waitFor(() =>
      expect(!second.memory.busy || secondReviewer.mock.calls.length > 0).toBe(true),
    );
    expect(secondReviewer).not.toHaveBeenCalled();
    expect(second.products.journal.pendingMemoryReview()?.id).toBe(queued.id);
    gate.resolve(unchanged);
    await closing;
    expect(first.products.journal.memoryEvent("swarmx.memory.review.planned")).toBeUndefined();
    expect(first.products.journal.pendingLearningRuns().map(({ id }) => id)).toEqual([terminal.id]);
    await first.products.dispose();
    second.memory.resume();
    await vi.waitFor(() => expect(secondReviewer).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(second.products.journal.pendingMemoryReview()).toBeUndefined());
    expect(second.products.journal.pendingLearningRuns()).toEqual([]);
    expect(
      second.products.journal
        .memoryJobEvents(queued.id)
        .flatMap(({ event }) =>
          event.type === EventType.CUSTOM && event.name === "swarmx.memory.review.finished"
            ? [event.value.state]
            : [],
        ),
    ).toEqual(["failed", "completed"]);
    expect(secondReviewer.mock.calls[0]?.[3]).toEqual(firstReviewer.mock.calls[0]?.[3]);
  } finally {
    gate.resolve(unchanged);
  }
});

it("releases a killed process's review lock without preventing journal appends", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-review-process-"));
  const directoryKey = "/research/owned";
  const journal = new ExecutionJournal(root, directoryKey);
  const otherDirectory = new ExecutionJournal(root, "/research/independent");
  const child = spawn(
    process.execPath,
    [
      "--import",
      "tsx",
      "--input-type=module",
      "-e",
      `import { ExecutionJournal } from ${JSON.stringify(new URL("../src/host/execution-journal.ts", import.meta.url).href)};
const journal = new ExecutionJournal(process.argv[1], process.argv[2]);
if (!journal.tryReviewLock()) throw new Error("Failed to acquire child review lock");
process.stdout.write("locked\\n");
setInterval(() => {}, 1000);`,
      root,
      directoryKey,
    ],
    { stdio: ["ignore", "pipe", "pipe"] },
  );
  const exited = once(child, "exit");
  let stderr = "";
  child.stderr.setEncoding("utf8").on("data", (chunk: string) => {
    stderr += chunk;
  });
  let release: (() => void) | undefined;
  let releaseOther: (() => void) | undefined;
  let reopened: ExecutionJournal | undefined;
  try {
    const ready = await Promise.race([
      once(child.stdout, "data").then(([chunk]) => String(chunk)),
      exited.then(([code, signal]) => {
        throw new Error(`Lock holder exited: ${code}/${signal}: ${stderr}`);
      }),
    ]);
    expect(ready).toBe("locked\n");
    expect(journal.tryReviewLock()).toBeUndefined();
    releaseOther = otherDirectory.tryReviewLock();
    expect(releaseOther).toBeTypeOf("function");
    const before = journal.append(null, {
      type: EventType.CUSTOM,
      name: "fixture.review-held",
      value: { observed: true },
    });
    expect(journal.read({}).events.map(({ id }) => id)).toEqual([before.id]);
    expect(child.kill("SIGKILL")).toBe(true);
    expect(await exited).toEqual([null, "SIGKILL"]);
    reopened = new ExecutionJournal(root, directoryKey);
    release = reopened.tryReviewLock();
    expect(release).toBeTypeOf("function");
    const after = reopened.append(null, {
      type: EventType.CUSTOM,
      name: "fixture.review-recovered",
      value: { observed: true },
    });
    expect(journal.read({}).events.map(({ id }) => id)).toEqual([before.id, after.id]);
  } finally {
    if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
    await exited;
    release?.();
    releaseOther?.();
    reopened?.close();
    journal.close();
    otherDirectory.close();
    await rm(root, { recursive: true, force: true });
  }
}, 10_000);
