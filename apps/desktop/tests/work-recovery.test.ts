import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { afterEach, expect, it, vi } from "vitest";
import { HARNESS_CAPABILITIES, type NativeAgent } from "../src/agents/types.js";
import { type ExecutionContext, ExecutionJournal } from "../src/host/execution-journal.js";
import { HostOperations } from "../src/host/operations.js";
import { ProductServices } from "../src/host/product-services.js";
import { WorkManager } from "../src/host/work.js";
import type { WorkAttempt } from "../src/work.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
const configurations = [{ id: "registered", harness: "codex", model: "fixture" }];
const item = (id: string, extra = {}) => ({
  id,
  cycleId: "cycle",
  goal: `Analyze ${id}`,
  criteria: "Verify independently",
  criteriaVersion: "v1",
  taskClass: "analysis",
  runtime: { budgetUsd: 1 },
  ...extra,
});
const invoice = (reservationId: string) => ({
  id: `invoice-${reservationId}`,
  reservationId,
  costUsd: 0.2,
  source: "invoice",
  reference: "Trusted invoice for observed consumption",
});
const confirmation = (reservationId: string) => ({
  reservationId,
  outcome: "cancelled",
  reference: "Operator verified that the provider process has stopped",
});

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-work-recovery-"));
  const handles: { journal: ExecutionJournal; work: WorkManager; closed: boolean }[] = [];
  const connect = () => {
    const journal = new ExecutionJournal(join(root, "logs"), "directory");
    const work = new WorkManager(join(root, "work"), "directory", journal);
    const handle = { journal, work, closed: false };
    handles.push(handle);
    return handle;
  };
  const close = (handle: ReturnType<typeof connect>) => {
    handle.work.close();
    handle.journal.close();
    handle.closed = true;
  };
  const first = connect();
  first.work.createCycle({
    id: "cycle",
    project: "research",
    budgetUsd: 10,
    concurrency: 1,
    reviewReserveUsd: 0.25,
    configurations,
  });
  first.work.createItem(item("interrupted", { priority: 100 }));
  first.work.createItem(item("ready"));
  const reserve = () => {
    const reservation = first.work.reserve("interrupted", () => true).reservation;
    assert.ok(reservation);
    return reservation;
  };
  const start = (reservation: WorkAttempt) => {
    const context: ExecutionContext = {
      sessionId: `codex:${randomUUID()}`,
      runId: randomUUID(),
      causedBy: null,
      attributes: first.work.attributes(reservation),
    };
    first.journal.append(context, {
      type: EventType.RUN_STARTED,
      threadId: context.sessionId ?? "",
      runId: context.runId,
    });
    first.journal.activate(context);
    return context;
  };
  cleanups.push(async () => {
    for (const handle of handles.reverse()) if (!handle.closed) close(handle);
    await rm(root, { recursive: true, force: true });
  });
  return { ...first, first, reserve, start, connect, close };
}

it("retains the concurrency slot and money for an unfinished native execution after restart", async () => {
  const { first, reserve, start, connect, close } = await fixture();
  const reservation = reserve();
  start(reservation);
  close(first);
  const restarted = connect();
  const snapshot = restarted.work.snapshot("cycle");
  expect(snapshot.reservations[0]).toMatchObject({
    state: "uncertain",
    outcome: "incomplete",
    finishedAt: null,
    costUsd: null,
  });
  expect(snapshot.items[0]?.state).toBe("blocked");
  expect(snapshot.balance).toMatchObject({ heldUsd: 1, remainingUsd: 9 });
  expect(restarted.journal.activeRuns()).toEqual([]);
  expect(restarted.work.reserve("ready", () => true)).toMatchObject({
    reservation: null,
    decision: { action: "defer", reason: "Cycle concurrency is occupied." },
  });
});

it("refreshes native liveness before applying an invoice and cannot settle an active reserved record", async () => {
  const { work, reserve, start } = await fixture();
  const reservation = reserve();
  start(reservation);
  expect(() => work.reconcileCharge(invoice(reservation.id))).toThrow("execution");
  expect(work.snapshot("cycle").reservations[0]).toMatchObject({ state: "running", costUsd: null });
  expect(work.reserve("ready", () => true).reservation).toBeNull();
});

it.each(["invoice-first", "outcome-first"])(
  "keeps cost and stopped-process confirmation independent (%s)",
  async (order) => {
    const { first, reserve, start, connect, close } = await fixture();
    const reservation = reserve();
    start(reservation);
    close(first);
    const { work, journal } = connect();
    work.reconcile();
    if (order === "invoice-first") {
      work.reconcileCharge(invoice(reservation.id));
      const billed = work.snapshot("cycle");
      expect(billed.reservations[0]).toMatchObject({
        state: "uncertain",
        finishedAt: null,
        outcome: "incomplete",
        costUsd: 0.2,
      });
      expect(work.reserve("ready", () => true).reservation).toBeNull();
    }
    const confirmed = work.reconcileOutcome(confirmation(reservation.id));
    expect(confirmed).toMatchObject({
      outcome: "confirmed-cancelled",
      finishedAt: expect.any(String),
    });
    if (order === "outcome-first") {
      expect(work.snapshot("cycle").balance).toMatchObject({ heldUsd: 1, remainingUsd: 9 });
      work.reconcileCharge(invoice(reservation.id));
    }
    work.reconcileCharge(invoice(reservation.id));
    expect(work.reconcileOutcome(confirmation(reservation.id)).finishedAt).toBe(
      confirmed.finishedAt,
    );
    expect(() =>
      work.reconcileOutcome({ ...confirmation(reservation.id), outcome: "completed" }),
    ).toThrow("conflict");
    expect(work.snapshot("cycle").reservations[0]).toMatchObject({
      state: "settled",
      outcome: "confirmed-cancelled",
      costUsd: 0.2,
    });
    expect(work.snapshot("cycle").feedback).toEqual([]);
    expect(work.snapshot("cycle").executions[0]?.outcome).toBe("incomplete");
    expect(
      journal
        .read({ limit: 100 })
        .events.filter(({ event }) => event.type === EventType.RUN_FINISHED),
    ).toEqual([]);
    expect(work.reserve("ready", () => true).reservation).not.toBeNull();
  },
);

it("refuses outcome confirmation during reservation preparation or a live native execution", async () => {
  const { work, reserve, start } = await fixture();
  const reservation = reserve();
  expect(() => work.reconcileOutcome(confirmation(reservation.id))).toThrow("execution");
  start(reservation);
  expect(() => work.reconcileOutcome(confirmation(reservation.id))).toThrow("execution");
  expect(work.snapshot("cycle").reservations[0]?.finishedAt).toBeNull();
});

it("preserves outcome confirmation identity when a delayed native usage report arrives", async () => {
  const { first, reserve, start, connect, close } = await fixture();
  const reservation = reserve();
  const context = start(reservation);
  close(first);
  const { work, journal } = connect();
  work.reconcile();
  const confirmed = work.reconcileOutcome(confirmation(reservation.id));
  journal.append(
    context,
    {
      type: EventType.RUN_FINISHED,
      threadId: context.sessionId ?? "",
      runId: context.runId,
      result: { stopReason: "cancelled" },
    },
    { "swarmx.usage.cost_usd": 0.3, "swarmx.usage.cost_source": "native-estimate" },
  );
  expect(work.reconcileOutcome(confirmation(reservation.id))).toMatchObject({
    finishedAt: confirmed.finishedAt,
    outcome: "confirmed-cancelled",
    report: confirmed.report,
    costUsd: 0.3,
    state: "settled",
  });
  expect(() =>
    work.reconcileOutcome({
      ...confirmation(reservation.id),
      reference: "Different operator report",
    }),
  ).toThrow("conflict");
  expect(work.snapshot("cycle").feedback).toEqual([]);
});

it("background review cannot occupy a slot held by an unresolved restarted execution", async () => {
  const { first, reserve, start, connect, close } = await fixture();
  const reservation = reserve();
  const context = start(reservation);
  close(first);
  const { work } = connect();
  expect(() => work.reserveReview([context.runId])).toThrow("occupies");
  work.reconcileOutcome(confirmation(reservation.id));
  expect(work.reserveReview([context.runId])).toMatchObject({
    purpose: "memory-review",
    reservedUsd: 0.25,
  });
});

it("releases only the concurrency slot when a native terminal result has unknown cost", async () => {
  const { work, journal, reserve, start } = await fixture();
  const reservation = reserve();
  const context = start(reservation);
  journal.append(context, {
    type: EventType.RUN_FINISHED,
    threadId: context.sessionId ?? "",
    runId: context.runId,
    result: { stopReason: "cancelled" },
  });
  journal.deactivate(context.sessionId ?? "");
  work.finish(reservation.id);
  expect(work.snapshot("cycle").reservations[0]).toMatchObject({
    state: "uncertain",
    outcome: "cancelled",
    finishedAt: expect.any(String),
    costUsd: null,
  });
  expect(work.snapshot("cycle").balance.heldUsd).toBe(1);
  expect(work.reserve("ready", () => true).reservation).not.toBeNull();
});

it("does not turn missing dispatch acknowledgement or its invoice into a finished execution", async () => {
  const { work, reserve } = await fixture();
  const reservation = reserve();
  work.finish(reservation.id, new Error("Dispatch acknowledgement was lost"));
  expect(work.snapshot("cycle").reservations[0]).toMatchObject({
    outcome: "dispatch-unconfirmed",
    finishedAt: null,
  });
  work.reconcileCharge(invoice(reservation.id));
  expect(work.reserve("ready", () => true).reservation).toBeNull();
  work.reconcileOutcome(confirmation(reservation.id));
  expect(work.reserve("ready", () => true).reservation).not.toBeNull();
});

it.each(["Bash", "swarm"])(
  "does not parse native %s argument fragments as Host dispatch records",
  async (toolCallName) => {
    const { work, journal, reserve, start } = await fixture();
    const reservation = reserve();
    const context = start(reservation);
    const toolCallId = randomUUID();
    const nativeTool = journal.append(context, {
      type: EventType.TOOL_CALL_START,
      toolCallId,
      toolCallName,
    });
    for (const delta of ['{"command":', '"pwd"}'])
      journal.append(
        { ...context, causedBy: nativeTool.id },
        { type: EventType.TOOL_CALL_ARGS, toolCallId, delta },
      );
    journal.append(context, {
      type: EventType.RUN_FINISHED,
      threadId: context.sessionId ?? "",
      runId: context.runId,
      result: { stopReason: "end_turn" },
    });
    journal.deactivate(context.sessionId ?? "");
    expect(() => work.finish(reservation.id)).not.toThrow();
    expect(work.snapshot("cycle").reservations[0]).toMatchObject({
      outcome: "completed",
      finishedAt: expect.any(String),
      costUsd: null,
    });
  },
);

async function products() {
  const cwd = await mkdtemp(join(tmpdir(), "swarmx-work-recovery-host-"));
  const services = await ProductServices.create({ cwd, productHome: join(cwd, "home") });
  services.settings.writeMemory({ autoReview: false });
  let sessions = 0;
  const native: NativeAgent = {
    name: "recovery fixture",
    capabilities: HARNESS_CAPABILITIES.codex,
    create: async () => `codex:${++sessions}`,
    list: async () => [],
    read: async () => {},
    models: vi.fn(async () => ({
      models: [{ id: "fixture", name: "Fixture", efforts: [] }],
      current: {},
    })),
    start: vi.fn(async () => ({ stopReason: "end_turn" })),
    interrupt: vi.fn(async () => {}),
    steer: async () => {},
    dispose: async () => {},
  };
  await services.attachAgents("http://localhost", native, "codex");
  services.work.createCycle({
    id: "cycle",
    project: "research",
    budgetUsd: 10,
    concurrency: 3,
    configurations: [
      ...configurations,
      { ...configurations[0], id: "not-permitted", model: "not-permitted" },
    ],
  });
  services.updatePolicy({ ...services.settings.read().policy, harnesses: { codex: ["fixture"] } });
  cleanups.push(async () => {
    await services.dispose();
    await rm(cwd, { recursive: true, force: true });
  });
  return { services, native };
}

it("records confirmed cancellation before native dispatch without inventing zero setup cost", async () => {
  const { services, native } = await products();
  services.work.createItem(item("cancelled"));
  const entered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  vi.mocked(native.models).mockImplementation(async () => {
    entered.resolve();
    await release.promise;
    return { models: [{ id: "fixture", name: "Fixture", efforts: [] }], current: {} };
  });
  const controller = new AbortController();
  const work = services.runWork("cancelled", controller.signal);
  await entered.promise;
  controller.abort();
  await vi.waitFor(() => expect(native.interrupt).toHaveBeenCalled());
  release.resolve();
  await work;
  expect(native.start).not.toHaveBeenCalled();
  const snapshot = services.work.snapshot("cycle");
  expect(snapshot.reservations[0]).toMatchObject({
    outcome: "cancelled-before-dispatch",
    finishedAt: expect.any(String),
    costUsd: null,
    runIds: [],
  });
  expect(snapshot.balance.heldUsd).toBe(1);
  expect(snapshot.feedback).toEqual([]);
});

it("skips unresolved, expired and inadmissible earlier queue items to reach ready work", async () => {
  const { services, native } = await products();
  services.work.createItem(item("unresolved", { priority: 100 }));
  services.work.createItem(item("expired", { priority: 90, deadline: "2020-01-01T00:00:00.000Z" }));
  services.work.createItem(
    item("inadmissible", { priority: 80, policy: "fixed", fixedConfiguration: "not-permitted" }),
  );
  services.work.createItem(item("ready", { priority: 1 }));
  const reservation = services.work.reserve("unresolved", () => true).reservation;
  assert.ok(reservation);
  services.work.finish(reservation.id, new Error("Unknown previous dispatch"));
  const operations = new HostOperations({
    products: services,
    signal: new AbortController().signal,
    origin: "http://localhost",
    token: "fixture",
    dispose: () => services.dispose(),
  });
  const result = await operations.workCommand({ action: "startNext", cycleId: "cycle" });
  expect(result.snapshot?.reservations.at(-1)?.workId).toBe("ready");
  expect(native.start).toHaveBeenCalledTimes(1);
  expect(services.work.item("unresolved").state).toBe("blocked");
});

it.each(["manual", "managed"] as const)(
  "skips an unavailable explicit %s Agent when starting the next item",
  async (mode) => {
    const { services, native } = await products();
    services.work.createItem(
      item("unavailable", {
        priority: 10,
        mode,
        [mode === "managed" ? "supervisor" : "configuration"]: {
          harness: "codex",
          model: "not-permitted",
        },
      }),
    );
    services.work.createItem(item("ready", { priority: 1 }));
    const operations = new HostOperations({
      products: services,
      signal: new AbortController().signal,
      origin: "http://localhost",
      token: "fixture",
      dispose: () => services.dispose(),
    });
    await expect(
      operations.workCommand({ action: "start", workId: "unavailable" }),
    ).rejects.toThrow("The selected Agent configuration is not permitted.");
    const result = await operations.workCommand({ action: "startNext", cycleId: "cycle" });
    expect(result.snapshot?.reservations.map(({ workId }) => workId)).toEqual(["ready"]);
    expect(native.start).toHaveBeenCalledTimes(1);
  },
);
