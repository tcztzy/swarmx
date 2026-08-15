import { describe, expect, it } from "vitest";
import { Agent } from "../src/agent.js";
import {
  appendGlobalMemoryInstructions,
  appendMemoryReflectionInstructions,
  appendPersonalMemoryInstructions,
  buildGlobalMemoryUseReceipt,
  buildPersonalMemoryUseReceipt,
  countPersonalMemoryAgentTargets,
  createGlobalMemorySnapshot,
  createPersonalMemorySnapshot,
  GLOBAL_MEMORY_MAX_CHARACTERS,
  GlobalMemorySaveInputSchema,
  MemoryReviewStateSchema,
  memoryReflectionDecision,
  PERSONAL_MEMORY_MAX_CHARACTERS,
  PersonalMemorySaveInputSchema,
  personalMemoryReceiptMessage,
} from "../src/personal-memory.js";

describe("Global Memory", () => {
  const updatedAt = "2026-08-12T08:00:00.000Z";

  it("keeps USER.md and MEMORY.md distinct in one frozen global snapshot", () => {
    const snapshot = createGlobalMemorySnapshot({
      user: {
        target: "user",
        fileName: "USER.md",
        content: "Prefers concise Chinese responses.",
        revision: 2,
        updatedAt,
      },
      memory: {
        target: "memory",
        fileName: "MEMORY.md",
        content: "SwarmX research findings belong in entity pages.",
        revision: 3,
        updatedAt,
      },
    });

    expect(Object.isFrozen(snapshot)).toBe(true);
    expect(appendGlobalMemoryInstructions("Base policy.", snapshot)).toContain(
      "USER.md (read-only global user context)",
    );
    expect(appendGlobalMemoryInstructions("Base policy.", snapshot)).toContain(
      "MEMORY.md (read-only global operational context)",
    );
    expect(
      buildGlobalMemoryUseReceipt({
        snapshot,
        executionPath: "direct_agent",
        agentCount: 1,
      }),
    ).toMatchObject({
      status: "used",
      files: [
        { target: "user", characterCount: 34 },
        { target: "memory", characterCount: 48 },
      ],
    });
    const agent = new Agent(
      { name: "global_memory_agent", backend: { type: "echo" }, instructions: "Base policy." },
      { globalMemory: snapshot },
    );
    expect(agent.instructions).toContain("USER.md (read-only global user context)");
    expect(agent.instructions).toContain("MEMORY.md (read-only global operational context)");
    expect(
      buildGlobalMemoryUseReceipt({
        snapshot: null,
        executionPath: "direct_agent",
        agentCount: 1,
        unavailable: true,
      }),
    ).toMatchObject({ status: "not_used", reason: "unavailable" });
  });

  it("bounds each global file and preserves legacy USER.md migration input", () => {
    expect(() =>
      createGlobalMemorySnapshot({
        user: {
          target: "user",
          fileName: "USER.md",
          content: "x".repeat(GLOBAL_MEMORY_MAX_CHARACTERS.user + 1),
          revision: 1,
          updatedAt,
        },
        memory: null,
      }),
    ).toThrow();
    expect(() =>
      createGlobalMemorySnapshot({
        user: null,
        memory: {
          target: "memory",
          fileName: "MEMORY.md",
          content: "😀".repeat(2_001),
          revision: 1,
          updatedAt,
        },
      }),
    ).toThrow();
    expect(
      createGlobalMemorySnapshot({
        user: null,
        memory: null,
        legacyUser: { content: "Legacy preference.", updatedAt },
      }),
    ).toMatchObject({
      user: { target: "user", fileName: "USER.md", content: "Legacy preference.", revision: 0 },
      source: "memory_files_with_legacy_user",
    });
  });

  it("rejects credentials before they enter USER.md or MEMORY.md", () => {
    expect(
      GlobalMemorySaveInputSchema.safeParse({
        target: "memory",
        content: "password = live-secret-value",
      }).success,
    ).toBe(false);
    expect(
      GlobalMemorySaveInputSchema.safeParse({
        target: "user",
        content: "Reference https://user:password@example.test/private",
      }).success,
    ).toBe(false);
  });

  it("keeps ten-turn review cursors isolated by Session across restarts", () => {
    const state = MemoryReviewStateSchema.parse({
      sessions: {
        session_a: { reviewedUserTurns: 6, updatedAt },
        session_b: { reviewedUserTurns: 9, updatedAt },
      },
    });
    expect(
      memoryReflectionDecision({
        sessionId: "session_a",
        userTurnCount: 10,
        state,
        userText: "Continue SwarmX architecture work.",
        now: updatedAt,
      }),
    ).toMatchObject({ due: false, unreviewedUserTurns: 4 });
    expect(
      memoryReflectionDecision({
        sessionId: "session_b",
        userTurnCount: 10,
        state,
        userText: "Finish Hermes research.",
        now: updatedAt,
      }),
    ).toMatchObject({ due: false, unreviewedUserTurns: 1 });
    const due = memoryReflectionDecision({
      sessionId: "session_a",
      userTurnCount: 16,
      state,
      userText: "Continue SwarmX architecture work.",
      now: updatedAt,
    });
    expect(due).toMatchObject({
      due: true,
      reason: "interval",
      sessionId: "session_a",
      fromUserTurn: 7,
      throughUserTurn: 16,
    });
    expect(appendMemoryReflectionInstructions("Base policy.", due)).toContain(
      "Only review Session session_a",
    );
    expect(appendMemoryReflectionInstructions("Base policy.", due)).not.toContain("session_b");
  });

  it("triggers explicit requests immediately and idle tails without global aggregation", () => {
    const state = MemoryReviewStateSchema.parse({ sessions: {} });
    expect(
      memoryReflectionDecision({
        sessionId: "session_explicit",
        userTurnCount: 1,
        state,
        userText: "请记住我偏好使用 TypeScript。",
        now: updatedAt,
      }),
    ).toMatchObject({ due: true, reason: "explicit", fromUserTurn: 1, throughUserTurn: 1 });
    expect(
      memoryReflectionDecision({
        sessionId: "session_idle",
        userTurnCount: 3,
        state: MemoryReviewStateSchema.parse({
          sessions: {
            session_idle: {
              reviewedUserTurns: 0,
              updatedAt: "2026-08-10T08:00:00.000Z",
            },
          },
        }),
        userText: "Resume the research.",
        now: updatedAt,
      }),
    ).toMatchObject({ due: true, reason: "idle_tail", sessionId: "session_idle" });
  });
});

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
