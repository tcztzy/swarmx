import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { EventType } from "@ag-ui/core";
import { afterEach, expect, it, vi } from "vitest";
import { z } from "zod";
import { HARNESS_CAPABILITIES, type NativeAgent, type Observer } from "../src/agents/types.js";
import { ExecutionJournal } from "../src/host/execution-journal.js";
import { ProductServices } from "../src/host/product-services.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
const signal = () => new AbortController().signal;
const sink: Observer = { text() {}, tool() {}, raw() {}, interact: async () => undefined };
const configuration = {
  id: "registered",
  harness: "codex",
  model: "fixture/research",
};

async function tool(services: ProductServices, name: string, args: unknown) {
  return services.callTool(name, args, { actorId: "mcp", callId: randomUUID(), signal: signal() });
}

async function fixture(execute: (services: ProductServices, text: string) => Promise<void>) {
  const cwd = await mkdtemp(join(tmpdir(), "swarmx-work-boundaries-"));
  const services = await ProductServices.create({
    cwd,
    productHome: join(cwd, "home"),
  });
  cleanups.push(async () => {
    await services.dispose();
    await rm(cwd, { recursive: true, force: true });
  });
  services.settings.writeMemory({ autoReview: false });
  let sessions = 0;
  const start = vi.fn<NativeAgent["start"]>(async (_session, text, observer) => {
    await execute(services, text);
    observer.text("answer", "Task output requires independent acceptance.");
    observer.raw(
      { synthetic: true },
      {
        "swarmx.usage.cost_usd": 0.4,
        "swarmx.usage.cost_source": "native-estimate",
        "swarmx.usage.scope": "run-total",
        "swarmx.usage.coverage": "complete",
      },
    );
    return { stopReason: "end_turn" };
  });
  const native: NativeAgent = {
    name: "work boundary fixture",
    capabilities: HARNESS_CAPABILITIES.codex,
    models: async () => ({
      models: [{ id: configuration.model, name: "Research fixture", efforts: [] }],
      current: {},
    }),
    list: async () => [],
    create: async () => `codex:boundary-${++sessions}`,
    read: async () => {},
    start,
    steer: async () => {},
    interrupt: async () => {},
    dispose: async () => {},
  };
  await services.attachAgents("http://localhost", native, "codex");
  services.work.createCycle({
    id: "cycle",
    project: "research",
    budgetUsd: 1,
    concurrency: 3,
    configurations: [configuration],
  });
  services.work.createItem({
    id: "first",
    cycleId: "cycle",
    goal: "Parent task",
    criteria: "Independently verify output",
    criteriaVersion: "criteria-v1",
    taskClass: "research",
    runtime: { budgetUsd: 0.4 },
  });
  return { cwd, services, start };
}

it("rejects native self-acceptance and administration while retaining the unaccepted runtime result", async () => {
  const { services } = await fixture(async (products) => {
    await expect(tool(services, "work", { action: "accept", accepted: true })).rejects.toThrow();
    expect(() => products.work.accept({})).toThrow("trusted Host caller");
    expect(() => products.work.createCycle({})).toThrow("trusted Host caller");
    await expect(products.runWork("first", signal())).rejects.toThrow("trusted Host caller");
    const status = await tool(services, "work", { action: "status" });
    expect(status).toMatchObject({ item: { id: "first", state: "running" }, heldUsd: 0.4 });
  });
  await services.runWork("first", signal());
  const snapshot = services.work.snapshot("cycle");
  expect(snapshot.items[0]?.state).toBe("awaiting-acceptance");
  expect(snapshot.feedback).toEqual([]);
  expect(snapshot.reservations).toHaveLength(1);
});

it("requires exact-task preparation and atomically shares the parent budget across child calls", async () => {
  const childStarted = Promise.withResolvers<void>();
  const releaseChild = Promise.withResolvers<void>();
  let childCount = 0;
  const { services, start } = await fixture(async (products, text) => {
    if (text === "Child task") {
      if (++childCount === 2) childStarted.resolve();
      await releaseChild.promise;
      return;
    }
    const send = {
      action: "send_message",
      agentId: "codex",
      model: configuration.model,
      text: "Child task",
      reason: "Use the registered child configuration.",
    };
    await expect(tool(services, "swarm", send)).rejects.toThrow("swarm.prepare");
    expect(products.work.snapshot("cycle").reservations).toHaveLength(1);
    const preparation = z.object({ preparationId: z.string() }).parse(
      await tool(services, "swarm", {
        action: "prepare",
        task: "Child task",
        queries: ["research"],
      }),
    );
    await expect(
      tool(services, "swarm", { ...send, text: "Different child task", ...preparation }),
    ).rejects.toThrow("exact task");
    const children = [1, 2].map(() => tool(services, "swarm", { ...send, ...preparation }));
    try {
      await childStarted.promise;
      const snapshot = products.work.snapshot("cycle");
      expect(snapshot.reservations).toHaveLength(3);
      expect(snapshot.reservations.slice(1)).toEqual([
        expect.objectContaining({ runtimeId: snapshot.reservations[0]?.id, reservedUsd: 0 }),
        expect.objectContaining({ runtimeId: snapshot.reservations[0]?.id, reservedUsd: 0 }),
      ]);
      expect(snapshot.balance.heldUsd).toBe(1.5);
      expect(start).toHaveBeenCalledTimes(3);
    } finally {
      releaseChild.resolve();
      await Promise.all(children);
    }
  });
  services.work.setBudget({ cycleId: "cycle", expectedBudgetUsd: 1, budgetUsd: 2 });
  await services.runWork("first", signal(), undefined, { runtime: { budgetUsd: 1.5 } });
  const snapshot = services.work.snapshot("cycle");
  expect(
    snapshot.reservations.map(({ purpose, workId, cycleId }) => ({ purpose, workId, cycleId })),
  ).toEqual([
    { purpose: "execution", workId: "first", cycleId: "cycle" },
    { purpose: "delegation", workId: "first", cycleId: "cycle" },
    { purpose: "delegation", workId: "first", cycleId: "cycle" },
  ]);
  expect(snapshot.reservations.every(({ runIds }) => runIds.length === 1)).toBe(true);
  expect(snapshot.balance.spentUsd).toBeCloseTo(1.2);
  expect(snapshot.balance.heldUsd).toBe(0);
});

it("prevents a managed native session from escaping its original work identity", async () => {
  const { services, start } = await fixture(async () => {});
  const completed = await services.runWork("first", signal());
  assert.ok("result" in completed);
  const { sessionId } = z.object({ sessionId: z.string() }).parse(completed.result);
  const agent = await services.agent("codex");
  await expect(
    agent.start(sessionId, "Resume outside work", sink, { model: configuration.model }),
  ).rejects.toThrow("original work identity");
  await expect(
    services.journal.scope.run(
      {
        sessionId: null,
        runId: randomUUID(),
        causedBy: null,
        attributes: { "swarmx.work.item_id": "another-work" },
      },
      () => agent.start(sessionId, "Reuse for another goal", sink, { model: configuration.model }),
    ),
  ).rejects.toThrow("original work identity");
  expect(start).toHaveBeenCalledTimes(1);
  expect(services.work.snapshot("cycle").reservations).toHaveLength(1);
});

function recordedSource(services: ProductServices) {
  const scope = services.journal.scope.getStore();
  assert.ok(scope, "An active recorded execution is required.");
  const record = services.journal.append(scope, {
    type: EventType.CUSTOM,
    name: "fixture.artifact.observed",
    value: { claim: "An Agent-reported domain result, not independent verification." },
  });
  return `urn:swarmx:execution:${record.id}`;
}

it("requires an active managed Work context for Host submissions", async () => {
  const { services } = await fixture(async () => {});
  await expect(tool(services, "work", { action: "submit", artifacts: [] })).rejects.toThrow(
    "No managed work is attached",
  );
  expect(services.work.snapshot("cycle").reservations).toEqual([]);
});

it("discovers bounded same-runtime provenance through Work status without Memory or science grants", async () => {
  const { services } = await fixture(async (products) => {
    const scope = products.journal.scope.getStore();
    assert.ok(scope);
    const oldestSource = recordedSource(products);
    for (let index = 0; index < 105; index++) recordedSource(products);
    const source = recordedSource(products);
    const foreignSources = [
      { ...scope.attributes, "swarmx.work.runtime_id": "foreign-runtime" },
      { ...scope.attributes, "swarmx.work.item_id": "foreign-work" },
    ].map((attributes) => {
      const record = products.journal.append(
        { ...scope, attributes },
        { type: EventType.CUSTOM, name: "fixture.foreign.evidence", value: {} },
      );
      return `urn:swarmx:execution:${record.id}`;
    });
    const noRun = products.journal.append(
      null,
      { type: EventType.CUSTOM, name: "fixture.no-run", value: {} },
      scope.attributes,
    );
    foreignSources.push(`urn:swarmx:execution:${noRun.id}`);
    const result = await tool(products, "work", { action: "status" });
    const { submissionEvidence } = z
      .object({
        submissionEvidence: z.object({
          sources: z.array(
            z.strictObject({
              source: z.string(),
              runId: z.string(),
              type: z.string(),
              observedAt: z.string(),
            }),
          ),
          truncated: z.boolean(),
        }),
      })
      .parse(result);
    expect(submissionEvidence.truncated).toBe(true);
    expect(submissionEvidence.sources).toHaveLength(100);
    const discovered = submissionEvidence.sources.find((entry) => entry.source === source);
    expect(discovered).toEqual({
      source,
      runId: scope.runId,
      type: EventType.CUSTOM,
      observedAt: products.journal.resolveSource(source).observedAt,
    });
    expect(submissionEvidence.sources.map((entry) => entry.source)).not.toContain(oldestSource);
    for (const foreignSource of foreignSources)
      expect(submissionEvidence.sources.map((entry) => entry.source)).not.toContain(foreignSource);
    expect(JSON.stringify(result)).not.toContain("An Agent-reported domain result");
    assert.ok(discovered);
    await expect(
      tool(products, "work", {
        action: "submit",
        artifacts: [{ id: "claim", revision: "v1", evidence: [discovered.source] }],
      }),
    ).resolves.toMatchObject({
      artifacts: [{ id: "claim", revision: "v1", evidence: [source] }],
    });
  });
  services.updatePolicy({ ...services.settings.read().policy, tools: [] });
  await services.runWork("first", signal());
  expect(services.work.snapshot("cycle").items[0]?.state).toBe("awaiting-acceptance");
});

it.each([
  "sx:a/evidence@1",
  "bio:dataset/evidence@v3",
  "https://example.org/result",
  "opaque claim",
])(
  "records opaque %s claims with current-runtime provenance and no science.read grant",
  async (id) => {
    let artifacts: { id: string; revision: string; evidence: string[] }[] = [];
    const { services } = await fixture(async (products) => {
      artifacts = [{ id, revision: "independent-revision", evidence: [recordedSource(products)] }];
      await expect(tool(products, "work", { action: "submit", artifacts })).resolves.toMatchObject({
        artifacts,
      });
      await expect(tool(products, "work", { action: "accept", accepted: true })).rejects.toThrow();
      expect(() => products.work.accept({})).toThrow("trusted Host caller");
      expect(products.work.snapshot("cycle").feedback).toEqual([]);
    });
    const { policy } = services.settings.read();
    services.updatePolicy({
      ...policy,
      tools: policy.tools.filter((permission) => permission !== "science.read"),
    });
    await services.runWork("first", signal());
    const snapshot = services.work.snapshot("cycle");
    expect(snapshot.reservations[0]?.artifacts).toEqual(artifacts);
    expect(snapshot.items[0]?.state).toBe("awaiting-acceptance");
    expect(snapshot.feedback).toEqual([]);
  },
);

const provenanceFailures = [
  "missing-evidence",
  "empty-evidence",
  "malformed-evidence",
  "missing-record",
  "foreign-directory",
  "foreign-work",
  "foreign-runtime",
  "missing-run",
] as const;

it.each(provenanceFailures)(
  "rejects %s in both Host and manager without partially replacing submitted artifacts",
  async (failure) => {
    let retained: { id: string; revision: string; evidence: string[] }[] = [];
    const { services } = await fixture(async (products) => {
      const scope = products.journal.scope.getStore();
      assert.ok(scope);
      const attemptId = scope.attributes["swarmx.work.reservation_id"];
      assert.equal(typeof attemptId, "string");
      const validSource = recordedSource(products);
      retained = [{ id: "retained artifact", revision: "original", evidence: [validSource] }];
      await tool(products, "work", { action: "submit", artifacts: retained });
      let evidence: string[] | undefined;
      if (failure === "empty-evidence") evidence = [];
      else if (failure === "malformed-evidence") evidence = ["urn:swarmx:execution:invalid"];
      else if (failure === "missing-record") evidence = [`urn:swarmx:execution:${randomUUID()}`];
      else if (failure !== "missing-evidence") {
        const journal =
          failure === "foreign-directory"
            ? new ExecutionJournal(join(products.options.productHome, "logs"), "other-directory")
            : products.journal;
        try {
          const attributes = {
            ...scope.attributes,
            ...(failure === "foreign-work" ? { "swarmx.work.item_id": "another-work" } : {}),
            ...(failure === "foreign-runtime" ? { "swarmx.work.runtime_id": randomUUID() } : {}),
          };
          const record = journal.append(
            failure === "missing-run" ? null : { ...scope, causedBy: null, attributes },
            { type: EventType.CUSTOM, name: "fixture.foreign.evidence", value: {} },
            attributes,
          );
          evidence = [`urn:swarmx:execution:${record.id}`];
        } finally {
          if (journal !== products.journal) journal.close();
        }
      }
      const artifacts = [
        { id: "would replace retained artifact", revision: "new", evidence: [validSource] },
        {
          id: "unverified domain claim",
          revision: "claimed revision",
          ...(evidence === undefined ? {} : { evidence }),
        },
      ];
      await expect(tool(products, "work", { action: "submit", artifacts })).rejects.toThrow();
      expect(products.work.snapshot("cycle").reservations[0]?.artifacts).toEqual(retained);
      expect(() => products.work.submit(attemptId, artifacts)).toThrow();
      expect(products.work.snapshot("cycle").reservations[0]?.artifacts).toEqual(retained);
    });
    await services.runWork("first", signal());
    expect(services.work.snapshot("cycle").reservations[0]?.artifacts).toEqual(retained);
    expect(services.work.snapshot("cycle").feedback).toEqual([]);
  },
);

it("accepts child-run evidence from the same managed runtime without science.read", async () => {
  let childSource = "";
  let parentRun = "";
  let childRun = "";
  const { services } = await fixture(async (products, text) => {
    const scope = products.journal.scope.getStore();
    assert.ok(scope);
    if (text === "Observe delegated result") {
      childRun = scope.runId;
      childSource = recordedSource(products);
      return;
    }
    parentRun = scope.runId;
    const preparation = z.object({ preparationId: z.string() }).parse(
      await tool(products, "swarm", {
        action: "prepare",
        task: "Observe delegated result",
        queries: ["research"],
      }),
    );
    await tool(products, "swarm", {
      action: "send_message",
      agentId: "codex",
      model: configuration.model,
      text: "Observe delegated result",
      reason: "Observe a result within the same managed Work runtime.",
      ...preparation,
    });
    expect(childRun).not.toBe(parentRun);
    await expect(
      tool(products, "work", {
        action: "submit",
        artifacts: [{ id: "delegated claim", revision: "v1", evidence: [childSource] }],
      }),
    ).resolves.toMatchObject({
      artifacts: [{ id: "delegated claim", revision: "v1", evidence: [childSource] }],
    });
  });
  const { policy } = services.settings.read();
  services.updatePolicy({
    ...policy,
    tools: policy.tools.filter((permission) => permission !== "science.read"),
  });
  await services.runWork("first", signal(), undefined, { runtime: { budgetUsd: 1 } });
  const snapshot = services.work.snapshot("cycle");
  expect(snapshot.reservations).toHaveLength(2);
  expect(snapshot.reservations[0]?.artifacts).toEqual([
    { id: "delegated claim", revision: "v1", evidence: [childSource] },
  ]);
  expect(snapshot.items[0]?.state).toBe("awaiting-acceptance");
  expect(snapshot.feedback).toEqual([]);
});

it("requires independent acceptance to pin the exact submitted artifact, revision and evidence", async () => {
  let artifact = { id: "claimed result", revision: "v1", evidence: [] as string[] };
  let otherSource = "";
  const { services } = await fixture(async (products) => {
    artifact = { ...artifact, evidence: [recordedSource(products), recordedSource(products)] };
    otherSource = recordedSource(products);
    await tool(products, "work", { action: "submit", artifacts: [artifact] });
  });
  const execution = await services.runWork("first", signal());
  assert.ok(execution.reservation);
  const feedback = {
    id: "independent-report",
    attemptId: execution.reservation.id,
    criteriaVersion: "criteria-v1",
    verdict: "passed",
    accepted: true,
    fraction: 1,
    layer: "behavior",
    source: "validator",
    evaluator: "test",
    evaluatorVersion: "v1",
    report: "Independently assessed the claimed result against the acceptance criteria.",
    artifacts: [artifact],
  };
  for (const mismatch of [
    { ...artifact, id: "another claim" },
    { ...artifact, revision: "v2" },
    { ...artifact, evidence: [otherSource] },
    { ...artifact, evidence: [...artifact.evidence].reverse() },
    { id: artifact.id, revision: artifact.revision },
  ]) {
    expect(() => services.work.accept({ ...feedback, artifacts: [mismatch] })).toThrow(
      "pin the submitted artifact revisions",
    );
    expect(services.work.snapshot("cycle").feedback).toEqual([]);
  }
  expect(services.work.accept(feedback)).toMatchObject(feedback);
  expect(services.work.snapshot("cycle").items[0]?.state).toBe("accepted");
  expect(() => services.work.submit(execution.reservation.id, [artifact])).toThrow(
    "Accepted submission is immutable",
  );
});

it("allows empty submissions while legacy artifacts stay readable without invented evidence", async () => {
  const { cwd, services } = await fixture(async (products) => {
    await expect(
      tool(products, "work", { action: "submit", artifacts: [] }),
    ).resolves.toMatchObject({ artifacts: [] });
  });
  const execution = await services.runWork("first", signal());
  assert.ok(execution.reservation);
  const artifact = { id: "sx:historical/result@1", revision: "1" };
  const database = new DatabaseSync(join(cwd, "home", "work", "work.sqlite"));
  try {
    database
      .prepare(
        "UPDATE work_state SET state = json_set(state, '$.reservations[0].artifacts', json(?)) WHERE workspace = ?",
      )
      .run(JSON.stringify([artifact]), services.directoryKey);
  } finally {
    database.close();
  }
  expect(services.work.snapshot("cycle").reservations[0]?.artifacts).toEqual([artifact]);
  const reopened = await ProductServices.create({ cwd, productHome: join(cwd, "home") });
  try {
    expect(reopened.work.snapshot("cycle").reservations[0]?.artifacts).toEqual([artifact]);
    expect(() => reopened.work.submit(execution.reservation.id, [artifact])).toThrow();
    expect(reopened.work.snapshot("cycle").reservations[0]?.artifacts).toEqual([artifact]);
  } finally {
    await reopened.dispose();
  }
});
