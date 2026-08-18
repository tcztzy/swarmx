import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AuditInput, ChatMessage, MessageChunk } from "@swarmx/core";
import { afterEach, describe, expect, it, vi } from "vitest";

const originalArgv = [...process.argv];
const originalExitCode = process.exitCode;

afterEach(() => {
  process.argv.splice(0, process.argv.length, ...originalArgv);
  process.exitCode = originalExitCode;
  vi.doUnmock("@swarmx/core");
  vi.doUnmock("node:readline");
  vi.restoreAllMocks();
  vi.resetModules();
});

describe("CLI entry surfaces", () => {
  it("renders persisted Session summaries and the complete built-in Harness inventory", async () => {
    const lines: string[] = [];
    vi.spyOn(console, "log").mockImplementation((...values) => lines.push(values.join(" ")));
    vi.doMock("@swarmx/core", async (importOriginal) => {
      const actual = await importOriginal<typeof import("@swarmx/core")>();
      return {
        ...actual,
        AuditStore: noOpAuditStore(),
        listSessionSummaries: () => [
          {
            id: "session_12345678",
            title: "Context audit",
            harness: "swarmx",
            messageCount: 4,
          },
        ],
      };
    });

    await runCli("sessions");
    expect(lines).toContain("[session_] Context audit (swarmx) - 4 messages");

    lines.length = 0;
    vi.resetModules();
    await runCli("harnesses");
    const { HARNESSES } = await import("@swarmx/core");
    for (const [id, harness] of Object.entries(HARNESSES)) {
      expect(lines).toContain(`${id}: ${harness.label}`);
    }
  }, 30_000);

  it("routes sessions timeline to the safe causal projector", async () => {
    const writes: string[] = [];
    vi.spyOn(process.stdout, "write").mockImplementation((value) => {
      writes.push(String(value));
      return true;
    });
    vi.doMock("@swarmx/core", async (importOriginal) => {
      const actual = await importOriginal<typeof import("@swarmx/core")>();
      return {
        ...actual,
        AuditStore: class {
          append(input: AuditInput): AuditInput {
            return input;
          }

          queryReadOnly(): [] {
            return [];
          }
        },
        readSessionTimelineSource: () => ({
          sessionId: "session-1",
          projectId: "project-1",
          tornTail: false,
          records: [
            {
              sequence: 1,
              type: "session_created",
              timestamp: "2026-08-14T00:00:00.000Z",
              messages: [],
            },
            {
              sequence: 2,
              type: "messages_appended",
              timestamp: "2026-08-14T00:00:01.000Z",
              requestId: "request-1",
              messages: [{ role: "user", kind: "message", content: "private prompt" }],
            },
          ],
        }),
      };
    });

    await runCli("sessions", "timeline", "session-1", "--json");

    expect(JSON.parse(writes.join(""))).toMatchObject({
      authority: "derived_diagnostic_projection",
      sessionId: "session-1",
      turns: [{ correlationId: "request-1" }],
    });
    expect(writes.join("")).not.toContain("private prompt");
  });

  it("replays prior user and assistant turns on each REPL request", async () => {
    const executeInputs: ChatMessage[][] = [];
    let lineHandler: ((line: string) => Promise<void>) | undefined;
    vi.spyOn(console, "log").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(process.stdout, "write").mockImplementation(() => true);
    vi.doMock("@swarmx/core", async (importOriginal) => {
      const actual = await importOriginal<typeof import("@swarmx/core")>();
      return {
        ...actual,
        AuditStore: noOpAuditStore(),
        createCoreRuntime: async () => ({
          prepareSwarm: () =>
            new (class {
              readonly name = "repl-test";
              readonly root = "agent";

              async execute(input: { messages: ChatMessage[] }): Promise<MessageChunk[]> {
                executeInputs.push(structuredClone(input.messages));
                const user = [...input.messages]
                  .reverse()
                  .find((message) => message.role === "user");
                return [
                  {
                    role: "assistant",
                    content: `answer:${user?.content ?? ""}`,
                    kind: "message",
                    agent: "agent",
                  },
                ];
              }
            })(),
          dispose: vi.fn(),
        }),
      };
    });
    vi.doMock("node:readline", () => ({
      createInterface: () => ({
        prompt: vi.fn(),
        on(event: string, handler: (line: string) => Promise<void>) {
          if (event === "line") lineHandler = handler;
          return this;
        },
      }),
    }));

    await runCli("repl");
    if (!lineHandler) throw new Error("REPL line handler was not registered.");
    await lineHandler("first");
    await lineHandler("second");

    expect(executeInputs).toEqual([
      [{ role: "user", content: "first" }],
      [
        { role: "user", content: "first" },
        { role: "assistant", content: "answer:first" },
        { role: "user", content: "second" },
      ],
    ]);
  });

  it("routes --context-suite through the context evaluator and prints only its report", async () => {
    const directory = mkdtempSync(join(tmpdir(), "swarmx-cli-context-"));
    const suitePath = join(directory, "suite.json");
    writeFileSync(suitePath, JSON.stringify(contextSuiteInput()));
    const writes: string[] = [];
    vi.spyOn(process.stdout, "write").mockImplementation((value) => {
      writes.push(String(value));
      return true;
    });
    const runContextEvaluation = vi.fn().mockResolvedValue({
      records: [],
      report: {
        schemaVersion: 2,
        suiteId: "cli_entry_context_v2",
        suiteHash: `sha256:${"a".repeat(64)}`,
        scorerVersion: "context_eval_scorer_v2",
        completedRounds: 1,
        totalRuns: 0,
        leaderboard: [],
        nextCandidates: [],
        candidateComparisons: [],
        completedAt: "2026-08-12T00:00:00.000Z",
      },
    });
    vi.doMock("@swarmx/core", async (importOriginal) => {
      const actual = await importOriginal<typeof import("@swarmx/core")>();
      return {
        ...actual,
        AuditStore: noOpAuditStore(),
        runContextEvaluation,
        createCoreRuntime: async () => ({
          prepareAgent: vi.fn(),
          dispose: vi.fn(),
        }),
      };
    });

    await runCli("eval-run", "--context-suite", suitePath, "--pretty");

    expect(runContextEvaluation).toHaveBeenCalledWith(
      expect.objectContaining({
        suite: expect.objectContaining({ suiteId: "cli_entry_context_v2" }),
      }),
    );
    expect(JSON.parse(writes.join(""))).toMatchObject({
      suiteId: "cli_entry_context_v2",
      totalRuns: 0,
    });
  });
});

async function runCli(...arguments_: string[]): Promise<void> {
  process.argv.splice(0, process.argv.length, process.execPath, "swarmx-cli", ...arguments_);
  await import("../src/cli.js");
  await new Promise((resolve) => setTimeout(resolve, 10));
}

function noOpAuditStore(): new () => { append(input: AuditInput): AuditInput } {
  return class {
    append(input: AuditInput): AuditInput {
      return input;
    }
  };
}

function contextSuiteInput(): unknown {
  return {
    schemaVersion: 2,
    suiteId: "cli_entry_context_v2",
    description: "Commander routing smoke suite.",
    provenance: {
      collectedAt: "2026-08-12",
      split: "development",
      exposureRisk: "public",
      source: "repository-authored",
      retirementPolicy: "Retire leaked cases.",
    },
    agents: [
      {
        agentId: "model_a",
        continuation: {
          name: "agent",
          model: "test-model",
          client: {
            apiProtocol: "openai_responses",
            contextWindowTokens: 4096,
            maxOutputTokens: 512,
          },
        },
      },
    ],
    cases: [
      {
        caseId: "case_a",
        objective: "Apply the pending change.",
        difficulty: "easy",
        history: [{ role: "user", kind: "message", content: "Apply the pending change." }],
        currentUserMessage: "Continue.",
        environment: {
          initialState: { done: false },
          goalState: { done: true },
          actions: [
            {
              actionId: "complete",
              description: "Complete the change.",
              effects: { done: true },
            },
          ],
        },
        scoring: {
          requiredActionIds: ["complete"],
        },
        provenance: {
          familyId: "case_a",
          source: "repository-authored",
          collectedAt: "2026-08-12",
          split: "development",
          exposureRisk: "public",
        },
      },
    ],
    matrix: { profiles: ["baseline_full"] },
    search: { rounds: 1 },
    baselineProfile: "baseline_full",
    maxRuns: 5,
  };
}
