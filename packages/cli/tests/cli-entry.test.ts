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
        Swarm: class {
          readonly name = "repl-test";
          readonly root = "agent";

          async execute(input: { messages: ChatMessage[] }): Promise<MessageChunk[]> {
            executeInputs.push(structuredClone(input.messages));
            const user = [...input.messages].reverse().find((message) => message.role === "user");
            return [
              {
                role: "assistant",
                content: `answer:${user?.content ?? ""}`,
                kind: "message",
                agent: "agent",
              },
            ];
          }
        },
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
});

async function runCli(...arguments_: string[]): Promise<void> {
  process.argv.splice(0, process.argv.length, process.execPath, "swarmx-cli", ...arguments_);
  await import("../src/cli.js");
}

function noOpAuditStore(): new () => { append(input: AuditInput): AuditInput } {
  return class {
    append(input: AuditInput): AuditInput {
      return input;
    }
  };
}
