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
