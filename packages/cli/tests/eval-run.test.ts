import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { buildEvalArguments, runEval } from "../src/eval-run.js";

describe("eval-run helpers", () => {
  it("builds chat arguments from a message", () => {
    expect(buildEvalArguments("hello", {})).toEqual({
      messages: [{ role: "user", content: "hello" }],
    });
  });

  it("prefers structured input JSON over the positional message", () => {
    expect(
      buildEvalArguments("ignored", {
        inputJson: '{"messages":[{"role":"user","content":"from json"}],"caseId":"case-1"}',
      }),
    ).toEqual({
      messages: [{ role: "user", content: "from json" }],
      caseId: "case-1",
    });
  });

  it("returns a schema-valid JSON result when Swarm execution fails", async () => {
    const dir = mkdtempSync(join(tmpdir(), "swarmx-eval-run-"));
    const configPath = join(dir, "swarm.json");
    writeFileSync(
      configPath,
      JSON.stringify({
        name: "bad_eval",
        root: "missing",
        nodes: {},
        edges: [],
      }),
    );

    const result = await runEval("hello", { config: configPath });

    expect(result.output).toBe("");
    expect(result.messages).toEqual([]);
    expect(result.trace).toEqual([]);
    expect(result.error).toMatch(/Root node/);
    expect(result.metrics).toEqual({
      steps: 0,
      messages: 0,
      toolCalls: 0,
      toolResults: 0,
    });
  });

  it("runs deterministic echo backend samples without model credentials", async () => {
    const dir = mkdtempSync(join(tmpdir(), "swarmx-echo-eval-run-"));
    const configPath = join(dir, "swarm.json");
    writeFileSync(
      configPath,
      JSON.stringify({
        name: "echo_eval",
        root: "echo_agent",
        nodes: {
          echo_agent: {
            kind: "agent",
            agent: {
              name: "echo_agent",
              backend: { type: "echo" },
            },
          },
        },
        edges: [],
      }),
    );

    const result = await runEval("deterministic answer", { config: configPath });

    expect(result.error).toBeNull();
    expect(result.output).toBe("deterministic answer");
    expect(result.messages).toHaveLength(1);
    expect(result.trace).toMatchObject([
      {
        swarm: "echo_eval",
        node: "echo_agent",
        kind: "agent",
        step: 1,
        status: "completed",
        messageCount: 1,
      },
    ]);
    expect(result.metrics).toEqual({
      steps: 1,
      messages: 1,
      toolCalls: 0,
      toolResults: 0,
    });
  });
});
