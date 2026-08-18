import { existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type AuditInput, AuditStore } from "@swarmx/core";
import { afterEach, describe, expect, it, vi } from "vitest";
import { runAuditCommand } from "../src/audit-command.js";

const roots: string[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("audit command", () => {
  it("verifies and filters compact events without exposing omitted content", () => {
    const { store } = fixture();
    store.append({
      category: "permission",
      action: "tool.decision",
      outcome: "completed",
      requestId: "req_123",
      metadata: { decision: "allowed", prompt: "private prompt", password: "secret" },
    });
    store.append({ category: "system", action: "ipc.request", outcome: "completed" });

    const verified = runAuditCommand({ verify: true }, store);
    expect(verified.exitCode).toBe(0);
    expect(verified.output).toContain("Audit chain verified: 2 events");

    const listed = runAuditCommand(
      { category: "permission", requestId: "req_123", json: true },
      store,
    );
    expect(listed.events).toHaveLength(1);
    expect(listed.output).toContain('"decision": "allowed"');
    expect(listed.output).not.toContain("private prompt");
    expect(listed.output).not.toContain('"password": "secret"');
  });

  it("exports verified JSONL with restrictive permissions and audits the export", () => {
    const { root, store } = fixture();
    store.append({ category: "task", action: "run.start", outcome: "attempted" });
    const output = join(root, "export.jsonl");

    const result = runAuditCommand({ output }, store);

    expect(result.exitCode).toBe(0);
    expect(result.exportedTo).toBe(output);
    expect(existsSync(output)).toBe(true);
    expect(statSync(output).mode & 0o777).toBe(0o600);
    expect(readFileSync(output, "utf8")).toContain('"action":"audit.export"');
    expect(store.query({ action: "audit.export" }).map((event) => event.outcome)).toEqual([
      "attempted",
      "completed",
    ]);
  });

  it("refuses query and export when chain verification fails", () => {
    const { root, store } = fixture();
    store.append({ category: "task", action: "run.start" });
    const source = readFileSync(store.filePath, "utf8");
    writeFileSync(store.filePath, source.replace('"category":"task"', '"category":"tool"'));
    const output = join(root, "must-not-exist.jsonl");

    const result = runAuditCommand({ output }, store);

    expect(result.exitCode).toBe(1);
    expect(result.output).toContain("Audit verification failed");
    expect(existsSync(output)).toBe(false);
  });
});

describe("CLI agent run audit inputs", () => {
  it("uses one action with compact send, eval, and REPL surfaces", async () => {
    const originalArgv = [...process.argv];
    const originalExitCode = process.exitCode;
    const appended: AuditInput[] = [];
    let executionError: Error | undefined;
    let lineHandler: ((line: string) => Promise<void>) | undefined;
    const consoleLog = vi.spyOn(console, "log").mockImplementation(() => {});
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    const stdoutWrite = vi.spyOn(process.stdout, "write").mockImplementation(() => true);

    vi.doMock("@swarmx/core", async (importOriginal) => {
      const actual = await importOriginal<typeof import("@swarmx/core")>();
      return {
        ...actual,
        AuditStore: class {
          append(input: AuditInput): AuditInput {
            appended.push(input);
            return input;
          }
        },
        createCoreRuntime: async () => ({
          prepareSwarm: () => ({
            name: "test",
            root: "agent",
            async execute(): Promise<[]> {
              if (executionError) throw executionError;
              return [];
            },
            async executeForEval() {
              if (executionError) throw executionError;
              return {
                output: "",
                messages: [],
                trace: [],
                error: null,
                metrics: { steps: 0, messages: 0, toolCalls: 0, toolResults: 0 },
              };
            },
          }),
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

    const runCli = async (...args: string[]): Promise<AuditInput[]> => {
      appended.length = 0;
      vi.resetModules();
      process.argv.splice(0, process.argv.length, process.execPath, "swarmx-cli", ...args);
      await import("../src/cli.js");
      await vi.waitFor(() => {
        expect(appended.some((event) => event.outcome !== "attempted")).toBe(true);
      });
      return appended.filter((event) => event.action === "agent.run");
    };

    try {
      const send = await runCli("send", "hello", "--model", "test-model");
      expect(send.map((event) => event.outcome)).toEqual(["attempted", "completed"]);
      expect(send[0]?.metadata).toEqual({
        surface: "cli_send",
        hasConfig: false,
        customHarness: false,
        modelSpecified: true,
        effortSpecified: false,
        resolvesEvolvedSkills: false,
      });
      expect(send[1]?.metadata).toEqual({ surface: "cli_send" });

      const evaluation = await runCli("eval-run", "hello");
      expect(evaluation.map((event) => event.outcome)).toEqual(["attempted", "completed"]);
      expect(evaluation[0]?.metadata).toEqual({
        surface: "eval",
        hasInlineMessage: true,
        hasConfig: false,
        hasInputFile: false,
        hasSkillDelivery: false,
        resolvesEvolvedSkills: false,
        contextSuite: false,
        writesContextJsonl: false,
        hasAblationProfile: false,
        hasMemorySnapshot: false,
      });
      expect(evaluation[1]?.metadata).toEqual({ surface: "eval" });

      executionError = new TypeError("failed turn");
      appended.length = 0;
      vi.resetModules();
      process.argv.splice(0, process.argv.length, process.execPath, "swarmx-cli", "repl");
      await import("../src/cli.js");
      expect(lineHandler).toBeTypeOf("function");
      await lineHandler?.("hello");
      const repl = appended.filter((event) => event.action === "agent.run");
      expect(repl.map((event) => event.outcome)).toEqual(["attempted", "failed"]);
      expect(repl[0]?.metadata).toEqual({ surface: "repl" });
      expect(repl[1]?.metadata).toEqual({ surface: "repl", errorType: "TypeError" });
    } finally {
      process.argv.splice(0, process.argv.length, ...originalArgv);
      process.exitCode = originalExitCode;
      consoleLog.mockRestore();
      consoleError.mockRestore();
      stdoutWrite.mockRestore();
      vi.doUnmock("@swarmx/core");
      vi.doUnmock("node:readline");
      vi.resetModules();
    }
  });
});

function fixture(): { root: string; store: AuditStore } {
  const root = mkdtempSync(join(tmpdir(), "swarmx-cli-audit-"));
  roots.push(root);
  return { root, store: new AuditStore({ filePath: join(root, "events.jsonl") }) };
}
