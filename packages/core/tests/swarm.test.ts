import { describe, expect, it } from "vitest";
import { Swarm, SwarmNode } from "../src/swarm.js";
import type { SwarmConfig } from "../src/types.js";

describe("Swarm", () => {
  const twoNodeSwarm: SwarmConfig = {
    name: "test",
    root: "agent_a",
    nodes: {
      agent_a: {
        kind: "agent",
        agent: { name: "agent_a", instructions: "First agent" },
      },
      agent_b: {
        kind: "agent",
        agent: { name: "agent_b", instructions: "Second agent" },
      },
    },
    edges: [{ source: "agent_a", target: "agent_b" }],
  };

  it("constructs from config", () => {
    const swarm = new Swarm(twoNodeSwarm);
    expect(swarm.name).toBe("test");
    expect(swarm.root).toBe("agent_a");
    expect(swarm.nodes.size).toBe(2);
    expect(swarm.edges).toHaveLength(1);
  });

  it("rebuilds predecessor graph correctly", () => {
    const swarm = new Swarm(twoNodeSwarm);
    const { predecessors } = swarm.rebuildGraphs();

    expect(predecessors.get("agent_a")?.size).toBe(0);
    expect(predecessors.get("agent_b")?.has("agent_a")).toBe(true);
  });

  it("conditioned edges do not create predecessors", () => {
    const swarm = new Swarm({
      name: "test",
      root: "a",
      nodes: {
        a: { kind: "agent", agent: { name: "a" } },
        b: { kind: "agent", agent: { name: "b" } },
      },
      edges: [{ source: "a", target: "b", condition: "false" }],
    });

    const { predecessors } = swarm.rebuildGraphs();
    expect(predecessors.get("b")?.size).toBe(0);
  });

  it("throws on unknown root", async () => {
    const swarm = new Swarm({
      name: "test",
      root: "nonexistent",
      nodes: {},
      edges: [],
    });

    await expect(swarm.execute({ messages: [{ role: "user", content: "hi" }] })).rejects.toThrow(
      /Root node/,
    );
  });

  it("runs swarm and source-agent handoff hooks", async () => {
    const invocations: Array<{
      capability: string;
      event: string;
      scope: string;
      handoff?: { source: string; target: string };
      arguments: Record<string, unknown>;
      outcome?: { status: string };
    }> = [];
    const swarm = new Swarm(
      {
        name: "hooked_swarm",
        root: "a",
        hooks: [
          {
            onStart: "swarm.start",
            onHandoff: "swarm.handoff",
            onEnd: "swarm.end",
          },
        ],
        nodes: {
          a: {
            kind: "agent",
            agent: {
              name: "a",
              backend: { type: "echo" },
              hooks: [{ onHandoff: "agent.handoff" }],
            },
          },
          b: { kind: "agent", agent: { name: "b", backend: { type: "echo" } } },
        },
        edges: [{ source: "a", target: "b" }],
      },
      {
        hook: {
          execute: async (capability, input) => {
            invocations.push({
              capability,
              event: input.event,
              scope: input.scope,
              handoff: input.handoff,
              arguments: input.arguments,
              outcome: input.outcome,
            });
            if (capability === "swarm.start") return { additionalContext: "start context" };
            if (capability === "swarm.handoff") {
              return { additionalContext: "handoff context" };
            }
          },
        },
      },
    );

    const messages = await swarm.execute({
      messages: [{ role: "user", content: "hello" }],
    });

    expect(messages).toHaveLength(2);
    expect(invocations.map(({ capability }) => capability)).toEqual([
      "swarm.start",
      "agent.handoff",
      "swarm.handoff",
      "swarm.end",
    ]);
    expect(invocations[1]).toMatchObject({
      event: "onHandoff",
      scope: "agent",
      handoff: { source: "a", target: "b" },
    });
    expect(invocations[2]).toMatchObject({
      event: "onHandoff",
      scope: "swarm",
      handoff: { source: "a", target: "b" },
    });
    expect(invocations[3]?.arguments.messages).toEqual([
      { role: "system", content: "start context" },
      { role: "system", content: "handoff context" },
      { role: "user", content: "hello" },
    ]);
    expect(invocations[3]?.outcome).toEqual({ status: "completed", messages });
  });

  it("runs onEnd with a failed outcome when onStart denies execution", async () => {
    const events: Array<{ capability: string; status?: string }> = [];
    const swarm = new Swarm(
      {
        name: "hooked_swarm",
        root: "agent",
        hooks: [{ onStart: "swarm.start", onEnd: "swarm.end" }],
        nodes: {
          agent: { kind: "agent", agent: { name: "agent", backend: { type: "echo" } } },
        },
        edges: [],
      },
      {
        hook: {
          execute: async (capability, input) => {
            events.push({ capability, status: input.outcome?.status });
            if (capability === "swarm.start") {
              return { continue: false, stopReason: "workflow blocked" };
            }
          },
        },
      },
    );

    await expect(swarm.execute({ messages: [{ role: "user", content: "hello" }] })).rejects.toThrow(
      /workflow blocked/,
    );
    expect(events).toEqual([
      { capability: "swarm.start", status: undefined },
      { capability: "swarm.end", status: "failed" },
    ]);
  });

  it("fails when the workflow step bound leaves a scheduled node unsettled", async () => {
    const names = Array.from({ length: 101 }, (_, index) => `agent_${index}`);
    const nodes = Object.fromEntries(
      names.map((name) => [
        name,
        { kind: "agent" as const, agent: { name, backend: { type: "echo" as const } } },
      ]),
    );
    const edges = names.slice(1).map((target, index) => ({
      source: names[index] as string,
      target,
    }));
    const swarm = new Swarm({
      name: "bounded_swarm",
      root: names[0] as string,
      nodes,
      edges,
    });

    await expect(
      swarm.execute({ messages: [{ role: "user", content: "continue" }] }),
    ).rejects.toThrow(/did not settle within 100 workflow steps/i);
  });

  it("swarm node name access", () => {
    const node = new SwarmNode({
      kind: "agent",
      agent: { name: "test_agent" },
    });
    expect(node.name).toBe("test_agent");
    expect(node.kind).toBe("agent");

    const toolNode = new SwarmNode({
      kind: "tool",
      tool: { name: "test_tool" },
    });
    expect(toolNode.name).toBe("test_tool");
    expect(toolNode.kind).toBe("tool");
  });
});
