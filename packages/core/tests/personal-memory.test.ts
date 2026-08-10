import { describe, expect, it } from "vitest";
import { Agent } from "../src/agent.js";
import {
  appendPersonalMemoryInstructions,
  buildPersonalMemoryUseReceipt,
  countPersonalMemoryAgentTargets,
  createPersonalMemorySnapshot,
  PERSONAL_MEMORY_MAX_CHARACTERS,
  PersonalMemorySaveInputSchema,
  personalMemoryReceiptMessage,
} from "../src/personal-memory.js";

describe("Personal Memory", () => {
  const updatedAt = "2026-08-09T08:00:00.000Z";

  it("accepts bounded visible text and rejects blank, oversized, control, and extra input", () => {
    expect(PersonalMemorySaveInputSchema.parse({ content: "Prefer concise answers." })).toEqual({
      content: "Prefer concise answers.",
    });
    expect(() => PersonalMemorySaveInputSchema.parse({ content: "   \n" })).toThrow();
    expect(() =>
      PersonalMemorySaveInputSchema.parse({
        content: "x".repeat(PERSONAL_MEMORY_MAX_CHARACTERS + 1),
      }),
    ).toThrow();
    expect(() => PersonalMemorySaveInputSchema.parse({ content: "hidden\u0000text" })).toThrow();
    expect(() =>
      PersonalMemorySaveInputSchema.parse({ content: "valid", unexpected: true }),
    ).toThrow();
  });

  it("injects one immutable read-only snapshot into direct and ACP Agent instructions", async () => {
    const snapshot = createPersonalMemorySnapshot({
      content: "Prefer concise answers.\nUse TypeScript for examples.",
      updatedAt,
    });
    const instructions = appendPersonalMemoryInstructions("Follow project policy.", snapshot);

    expect(Object.isFrozen(snapshot)).toBe(true);
    expect(instructions).toContain("Follow project policy.");
    expect(instructions).toContain("read-only Personal Memory snapshot");
    expect(instructions).toContain(JSON.stringify(snapshot.content));

    const agent = new Agent(
      { name: "memory_agent", backend: { type: "echo" }, instructions: "Follow project policy." },
      { personalMemory: snapshot },
    );
    expect(agent.instructions).toBe(instructions);

    const prompts: string[] = [];
    const external = new Agent(
      { name: "external_memory_agent", backend: { type: "custom", program: "test-acp" } },
      {
        personalMemory: snapshot,
        createAcpClient: () => ({
          async prompt(_options, input) {
            prompts.push(input.text);
            return {
              messages: [{ role: "assistant", kind: "message", content: "done" }],
            };
          },
        }),
      },
    );
    await external.call({ messages: [{ role: "user", content: "hello" }] });
    expect(prompts).toHaveLength(1);
    expect(prompts[0]).toContain(JSON.stringify(snapshot.content));
  });

  it("builds path-specific bounded receipts and counts nested workflow Agent consumers", () => {
    const snapshot = createPersonalMemorySnapshot({
      content: `Prefer concise answers. ${"detail ".repeat(80)}`,
      updatedAt,
    });
    const used = buildPersonalMemoryUseReceipt({
      snapshot,
      executionPath: "external_acp",
      agentCount: 1,
    });
    const noAgents = buildPersonalMemoryUseReceipt({
      snapshot,
      executionPath: "workflow",
      agentCount: 0,
    });
    const usedMessage = personalMemoryReceiptMessage(used);

    expect(used).toMatchObject({
      status: "used",
      source: "desktop_settings",
      executionPath: "external_acp",
      agentCount: 1,
      updatedAt,
      characterCount: snapshot.characterCount,
    });
    expect(used.summary.length).toBeLessThanOrEqual(160);
    expect(JSON.stringify(used)).not.toContain(snapshot.content);
    expect(usedMessage).toMatchObject({
      role: "system",
      kind: "message",
      render: { source: "personal_memory_receipt" },
    });
    expect(usedMessage.content).toContain("Personal Memory used");
    expect(usedMessage.content).toContain("External ACP");
    expect(noAgents).toMatchObject({
      status: "not_used",
      reason: "no_agent_nodes",
      executionPath: "workflow",
      source: "desktop_settings",
    });
    expect(
      countPersonalMemoryAgentTargets({
        name: "nested",
        root: "direct",
        edges: [],
        nodes: {
          direct: { kind: "agent", agent: { name: "direct" } },
          nested: {
            kind: "swarm",
            swarm: {
              name: "inner",
              root: "external",
              edges: [],
              nodes: {
                external: {
                  kind: "agent",
                  agent: {
                    name: "external",
                    backend: { type: "custom", program: "test-acp" },
                  },
                },
              },
            },
          },
        },
      }),
    ).toBe(2);
  });
});
