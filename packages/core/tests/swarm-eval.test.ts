import { describe, expect, it } from "vitest";
import { Swarm } from "../src/swarm.js";
import { EvalRunResultSchema, type MessageChunk, type SwarmConfig } from "../src/types.js";

function stubAgent(swarm: Swarm, nodeName: string, content: string): void {
  const agent = swarm.nodes.get(nodeName)?.agent;
  if (!agent) throw new Error(`Missing agent ${nodeName}`);
  agent.call = async (): Promise<{ messages: MessageChunk[] }> => ({
    messages: [
      {
        role: "assistant",
        content,
        kind: "message",
        agent: nodeName,
      },
    ],
  });
}

describe("Swarm eval execution", () => {
  const twoAgentConfig: SwarmConfig = {
    name: "eval_test",
    root: "agent_a",
    nodes: {
      agent_a: {
        kind: "agent",
        agent: { name: "agent_a", instructions: "First" },
      },
      agent_b: {
        kind: "agent",
        agent: { name: "agent_b", instructions: "Second" },
      },
    },
    edges: [{ source: "agent_a", target: "agent_b" }],
  };

  it("returns schema-valid output, messages, trace, and metrics", async () => {
    const swarm = new Swarm(twoAgentConfig);
    stubAgent(swarm, "agent_a", "first answer");
    stubAgent(swarm, "agent_b", "second answer");

    const result = await swarm.executeForEval({
      messages: [{ role: "user", content: "hello" }],
    });

    expect(EvalRunResultSchema.parse(result)).toEqual(result);
    expect(result.error).toBeNull();
    expect(result.output).toBe("first answer\nsecond answer");
    expect(result.messages.map((message) => message.content)).toEqual([
      "first answer",
      "second answer",
    ]);
    expect(result.trace).toMatchObject([
      {
        swarm: "eval_test",
        node: "agent_a",
        kind: "agent",
        step: 1,
        status: "completed",
        messageCount: 1,
      },
      {
        swarm: "eval_test",
        node: "agent_b",
        kind: "agent",
        step: 2,
        status: "completed",
        messageCount: 1,
      },
    ]);
    expect(new Set(result.trace.map((event) => event.runId)).size).toBe(1);
    expect(result.metrics).toEqual({
      steps: 2,
      messages: 2,
      toolCalls: 0,
      toolResults: 0,
    });
  });

  it("captures runtime errors in the eval result", async () => {
    const swarm = new Swarm(twoAgentConfig);
    const agent = swarm.nodes.get("agent_a")?.agent;
    if (!agent) throw new Error("Missing agent");
    agent.call = async (): Promise<{ messages: MessageChunk[] }> => {
      throw new Error("model unavailable");
    };

    const result = await swarm.executeForEval({
      messages: [{ role: "user", content: "hello" }],
    });

    expect(EvalRunResultSchema.parse(result)).toEqual(result);
    expect(result.output).toBe("");
    expect(result.messages).toEqual([]);
    expect(result.error).toBe("model unavailable");
    expect(result.trace).toMatchObject([
      {
        node: "agent_a",
        kind: "agent",
        status: "failed",
        messageCount: 0,
        error: "model unavailable",
      },
    ]);
    expect(result.metrics.steps).toBe(1);
  });
});
