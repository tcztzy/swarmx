import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { expect, it, vi } from "vitest";
import { HARNESS_CAPABILITIES, type NativeAgent } from "../src/agents/types.js";
import { ProductServices } from "../src/host/product-services.js";

it("keeps feedback, cost and dispatch effort separate for two configurations of the same model", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-work-effort-"));
  const services = await ProductServices.create({ cwd: root, productHome: join(root, "home") });
  services.settings.writeMemory({ autoReview: false });
  const configurations = [
    {
      id: "low",
      harness: "pi",
      model: "fixture-model",
      effort: "low",
    },
    {
      id: "high",
      harness: "pi",
      model: "fixture-model",
      effort: "high",
    },
  ];
  const agent: NativeAgent = {
    name: "Offline effort fixture",
    capabilities: HARNESS_CAPABILITIES.pi,
    create: async () => `pi:${randomUUID()}`,
    list: async () => [],
    read: async () => {},
    models: async () => ({
      models: [
        {
          id: "fixture-model",
          name: "Fixture",
          efforts: [
            { id: "low", name: "Low" },
            { id: "high", name: "High" },
          ],
        },
      ],
      current: {},
    }),
    start: vi.fn(async (_id, _text, observer, options) => {
      const high = options?.effort === "high";
      observer.text("answer", high ? "5" : "4");
      observer.raw(
        { type: "offline-usage" },
        {
          "swarmx.agent.effort": options?.effort,
          "gen_ai.usage.input_tokens": 10,
          "gen_ai.usage.output_tokens": high ? 8 : 2,
          "swarmx.usage.cost_usd": high ? 0.8 : 0.2,
          "swarmx.usage.basis": "Offline deterministic fixture; synthetic USD; no billed call",
          "swarmx.usage.coverage": "complete",
          "swarmx.usage.cost_source": "native-estimate",
        },
      );
      return { stopReason: "end_turn" };
    }),
    interrupt: async () => {},
    steer: async () => {},
    dispose: async () => {},
  };
  try {
    await services.attachAgents("http://localhost", agent, "pi");
    services.work.createCycle({ id: "cycle", project: "analysis", budgetUsd: 4, configurations });
    for (const id of ["first", "second", "next"])
      services.work.createItem({
        id,
        cycleId: "cycle",
        goal: "Compute 2 + 3",
        criteria: "Return exactly the independently computed sum",
        criteriaVersion: "v1",
        taskClass: "arithmetic",
        runtime: { budgetUsd: 1 },
        ...(id === "first" ? { configuration: configurations[0] } : {}),
      });

    const first = await services.runWork("first", new AbortController().signal);
    expect(first).toMatchObject({ result: { text: "4" } });
    expect(first.reservation?.configuration).toEqual(configurations[0]);
    const firstFeedback = services.work.accept({
      id: "low-feedback",
      attemptId: first.reservation?.id,
      criteriaVersion: "v1",
      verdict: "failed",
      accepted: false,
      fraction: 0,
      layer: "behavior",
      source: "validator",
      evaluator: "independent-arithmetic",
      evaluatorVersion: "v1",
      report: "2 + 3 = 5; observed 4.",
    });
    const second = await services.runWork("second", new AbortController().signal);
    expect(second).toMatchObject({ result: { text: "5" } });
    expect(second.reservation?.configuration).toEqual(configurations[1]);
    expect(second.decision.evidence).toEqual([
      expect.objectContaining({
        configurationId: "low",
        samples: 1,
        accepted: 0,
        observedCostUsd: 0.2,
        costSamples: 1,
        feedbackIds: [firstFeedback.id],
      }),
      expect.objectContaining({
        configurationId: "high",
        samples: 0,
        accepted: 0,
        observedCostUsd: null,
        costSamples: 0,
        feedbackIds: [],
      }),
    ]);
    const secondFeedback = services.work.accept({
      id: "high-feedback",
      attemptId: second.reservation?.id,
      criteriaVersion: "v1",
      verdict: "passed",
      accepted: true,
      fraction: 1,
      layer: "behavior",
      source: "validator",
      evaluator: "independent-arithmetic",
      evaluatorVersion: "v1",
      report: "2 + 3 = 5; observed 5.",
    });
    const prepared = await services.journal.scope.run(
      {
        sessionId: null,
        runId: randomUUID(),
        causedBy: null,
        attributes: { "swarmx.work.item_id": "next" },
      },
      () =>
        services.callTool(
          "swarm",
          { action: "prepare", task: "Compute 2 + 3", queries: ["arithmetic"] },
          { actorId: "fixture", callId: randomUUID(), signal: new AbortController().signal },
        ),
    );
    expect(prepared).toMatchObject({
      work: {
        candidates: configurations,
        evidence: [
          expect.objectContaining({
            configurationId: "low",
            samples: 1,
            accepted: 0,
            meanFraction: 0,
            observedCostUsd: 0.2,
            costSamples: 1,
            feedbackIds: [firstFeedback.id],
          }),
          expect.objectContaining({
            configurationId: "high",
            samples: 1,
            accepted: 1,
            meanFraction: 1,
            observedCostUsd: 0.8,
            costSamples: 1,
            feedbackIds: [secondFeedback.id],
          }),
        ],
      },
    });
    expect(agent.start).toHaveBeenCalledTimes(2);
    expect(agent.start).toHaveBeenNthCalledWith(
      1,
      expect.any(String),
      expect.any(String),
      expect.any(Object),
      expect.objectContaining({ model: "fixture-model", effort: "low" }),
    );
    expect(agent.start).toHaveBeenNthCalledWith(
      2,
      expect.any(String),
      expect.any(String),
      expect.any(Object),
      expect.objectContaining({ model: "fixture-model", effort: "high" }),
    );
    const starts = services.work
      .records()
      .filter(({ event }) => event.type === EventType.RUN_STARTED);
    expect(starts.map(({ attributes }) => attributes["gen_ai.request.reasoning.level"])).toEqual([
      "low",
      "high",
    ]);
    const snapshot = services.work.snapshot("cycle");
    expect(snapshot.balance).toMatchObject({ spentUsd: 1, heldUsd: 0, remainingUsd: 3 });
    expect(snapshot.reservations).toEqual([
      expect.objectContaining({
        id: first.reservation?.id,
        costUsd: 0.2,
        configuration: configurations[0],
      }),
      expect.objectContaining({
        id: second.reservation?.id,
        costUsd: 0.8,
        configuration: configurations[1],
      }),
    ]);
  } finally {
    await services.dispose();
    await rm(root, { recursive: true, force: true });
  }
});
