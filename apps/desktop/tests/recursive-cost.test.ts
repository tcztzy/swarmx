import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { expect, it } from "vitest";
import { z } from "zod";
import { HARNESS_CAPABILITIES, type NativeAgent } from "../src/agents/types.js";
import { currentAgentBinding } from "../src/host/agent-registry.js";
import { ProductServices } from "../src/host/product-services.js";

it.each([false, true])(
  "counts real three-level Host delegation once and compares subtree costs (missing leaf cost: %s)",
  async (missing) => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-recursive-cost-"));
    const services = await ProductServices.create({ cwd: root, productHome: join(root, "home") });
    services.settings.writeMemory({ autoReview: false });
    const configurations = ["A", "B", "C"].map((id) => ({
      id,
      harness: "pi",
      model: `fixture/${id}`,
    }));
    const native: NativeAgent = {
      name: "Local recursive cost fixture",
      capabilities: HARNESS_CAPABILITIES.pi,
      create: async () => `pi:${randomUUID()}`,
      list: async () => [],
      read: async () => {},
      models: async () => ({
        models: configurations.map(({ model }) => ({ id: model, name: model, efforts: [] })),
        current: {},
      }),
      start: async (_session, _text, observer, options) => {
        const level = options?.model === "fixture/C" ? 3 : options?.model === "fixture/B" ? 2 : 1;
        if (level < 3) {
          const child = level === 1 ? "B" : "C";
          const tools = currentAgentBinding().productTools;
          assert.ok(tools);
          const call = (args: unknown) =>
            tools.call("swarm", args, randomUUID(), new AbortController().signal);
          const prepared = z
            .object({ preparationId: z.string() })
            .parse(
              await call({ action: "prepare", task: child, queries: ["local recursive cost"] }),
            );
          const request = {
            action: "send_message",
            agentId: "pi",
            model: `fixture/${child}`,
            text: child,
            preparationId: prepared.preparationId,
            reason: "Use the requested local child configuration.",
          };
          await call(request);
          if (missing && level === 2)
            await expect(call(request)).rejects.toThrow("Finished child cost is unknown");
        }
        observer.tool("local-command", "bash", { command: "true" });
        observer.tool("local-command", "bash", { command: "true" }, { exitCode: 0 });
        observer.raw(
          { type: "fixture-usage", nativeInternalCostIncluded: true },
          {
            "swarmx.usage.cost_usd": missing && level === 3 ? null : level,
            "swarmx.usage.cost_source": "native-estimate",
            "swarmx.usage.scope": "native-query",
            "swarmx.usage.basis": "Local fixture; synthetic USD; includes native internal calls",
          },
        );
        observer.text("answer", `${level}`);
        return { stopReason: "end_turn" };
      },
      interrupt: async () => {},
      steer: async () => {},
      dispose: async () => {},
    };
    try {
      await services.attachAgents("http://localhost", native, "pi");
      services.work.createCycle({
        id: "cycle",
        project: "recursive accounting",
        budgetUsd: 30,
        concurrency: 3,
        configurations,
      });
      for (const id of ["first", "next"])
        services.work.createItem({
          id,
          cycleId: "cycle",
          goal: "A",
          criteria: "Complete all three levels",
          criteriaVersion: "v1",
          taskClass: "recursive",
        });
      await services.runWork("first", new AbortController().signal);
      const records = services.work.records();
      const starts = records.filter(({ event }) => event.type === EventType.RUN_STARTED);
      expect(starts).toHaveLength(3);
      const parent = starts[0];
      assert.ok(parent);
      const evidence = services.journal.evidence([`urn:swarmx:execution:${parent.id}`]);
      expect(evidence.runs.map(({ costUsd }) => costUsd)).toEqual([1, 2, missing ? null : 3]);
      expect(evidence.runs.map(({ totalCostUsd }) => totalCostUsd)).toEqual(
        missing ? [null, null, null] : [6, 5, 3],
      );
      expect(evidence.runs.map(({ totalCostComplete }) => totalCostComplete)).toEqual(
        Array(3).fill(!missing),
      );
      expect(evidence.runs.map(({ parentRunId }) => parentRunId)).toEqual([
        null,
        starts[0]?.runId,
        starts[1]?.runId,
      ]);
      expect(evidence.statistics.cost).toEqual({
        sampleCount: missing ? 2 : 3,
        usd: missing ? 3 : 6,
        complete: !missing,
      });
      const callCount = missing ? 8 : 7;
      expect(evidence.statistics.tools).toEqual({ callCount, usd: 0, unpricedCalls: callCount });
      const repeated = services.journal.evidence(
        starts.map(({ id }) => `urn:swarmx:execution:${id}`),
      );
      expect(repeated.statistics.cost).toEqual(evidence.statistics.cost);
      assert.ok(parent.runId);
      const saved = services.journal.learningEvidence([parent.runId]);
      expect(saved.omittedCount).toBeGreaterThan(0);
      expect(saved.runs.map(({ totalCostUsd }) => totalCostUsd)).toEqual([null, null, null]);
      expect(saved.runs.map(({ totalCostComplete }) => totalCostComplete)).toEqual([
        false,
        false,
        false,
      ]);
      const snapshot = services.journal.append(null, {
        type: EventType.CUSTOM,
        name: "swarmx.memory.review.started",
        value: { snapshot: { evidence: saved } },
      });
      expect(
        services.journal.evidence([`urn:swarmx:execution:${snapshot.id}`]).statistics.cost,
      ).toEqual({
        ...evidence.statistics.cost,
        complete: false,
      });
      const status = services.work.snapshot("cycle");
      expect(status.balance).toMatchObject({
        spentUsd: missing ? 3 : 6,
        heldUsd: missing ? 27 : 0,
      });
      expect(status.tools).toEqual({
        callCount: callCount + 2,
        usd: 0,
        unpricedCalls: callCount + 2,
      });
      const prepared = services.work.prepare("next", () => true);
      expect(prepared.evidence.map(({ observedCostUsd }) => observedCostUsd)).toEqual([
        missing ? null : 6,
        null,
        null,
      ]);
      for (const [attributes, cost] of [
        [{ "swarmx.work.item_id": "first" }, 0.5],
        [{ "swarmx.work.cycle_id": "cycle" }, 0.125],
        [parent.attributes, 0.25],
      ] as const) {
        const context = {
          sessionId: null,
          runId: randomUUID(),
          causedBy: null,
          attributes,
        };
        const toolCallId = randomUUID();
        services.journal.append(context, {
          type: EventType.TOOL_CALL_START,
          toolCallId,
          toolCallName: "local-resource",
        });
        for (let index = 0; index < 2; index++)
          services.journal.append(
            context,
            {
              type: EventType.TOOL_CALL_RESULT,
              toolCallId,
              messageId: randomUUID(),
              content: "{}",
            },
            { "swarmx.tool.cost_usd": cost },
          );
      }
      const priced = services.work.snapshot("cycle");
      expect(priced.tools.usd).toBe(0.875);
      expect(priced.balance.spentUsd).toBe((missing ? 3 : 6) + 0.875);
      expect(priced.balance.heldUsd).toBe(missing ? 26.75 : 0);
      const leaf = priced.reservations.find(
        ({ configuration }) => configuration?.model === "fixture/C",
      );
      assert.ok(leaf);
      services.work.reconcileCharge({
        id: "leaf-invoice",
        reservationId: leaf.id,
        costUsd: 4,
        source: "invoice",
        reference: "Local fixture bill replaces the C estimate; no external charge",
      });
      expect(services.work.snapshot("cycle").balance).toMatchObject({
        spentUsd: 7.875,
        heldUsd: 0,
      });
      expect(services.work.prepare("next", () => true).evidence[0]?.observedCostUsd).toBe(7.25);
    } finally {
      await services.dispose();
      await rm(root, { recursive: true, force: true });
    }
  },
);
