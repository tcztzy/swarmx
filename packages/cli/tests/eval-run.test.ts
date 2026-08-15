import { mkdtempSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  type ContextEvaluationExecutor,
  estimateContextEvaluationMaxRuns,
  type SwarmConfig,
} from "@swarmx/core";
import { describe, expect, it, vi } from "vitest";
import {
  buildEvalArguments,
  evalSwarmOptions,
  formatContextEvaluationError,
  formatContextEvaluationReport,
  loadContextEvaluationSuite,
  runContextEvalSuite,
  runEval,
} from "../src/eval-run.js";

describe("eval-run skill delivery binding", () => {
  it("refuses a multi-agent config without --skill-delivery-agent", async () => {
    const config: SwarmConfig = {
      name: "two",
      root: "a",
      nodes: {
        a: { kind: "agent", agent: { name: "a" } },
        b: { kind: "agent", agent: { name: "b" } },
      },
      edges: [],
    };
    await expect(
      evalSwarmOptions(
        {
          skillDelivery: JSON.stringify({
            skillId: "s",
            variantId: "s:v",
            revisionId: "r_v",
            contentDigest:
              "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            mode: "prompt_fragment",
          }),
          skillContentPath: "/tmp/skill.md",
        },
        config,
      ),
    ).rejects.toThrow(/requires --skill-delivery-agent/i);
  });

  it("binds an explicit delivery to a single named agent", async () => {
    const config: SwarmConfig = {
      name: "two",
      root: "a",
      nodes: {
        a: { kind: "agent", agent: { name: "a" } },
        b: { kind: "agent", agent: { name: "b" } },
      },
      edges: [],
    };
    const { mkdtemp, writeFile } = await import("node:fs/promises");
    const { tmpdir } = await import("node:os");
    const root = await mkdtemp(`${tmpdir()}/swarmx-eval-agent-`);
    const content = "# Skill\n";
    const { createHash } = await import("node:crypto");
    const digest = createHash("sha256").update(content).digest("hex");
    await writeFile(`${root}/skill.md`, content, "utf8");
    const options = await evalSwarmOptions(
      {
        skillDelivery: JSON.stringify({
          skillId: "s",
          variantId: "s:v",
          revisionId: "r_v",
          contentDigest: `sha256:${digest}`,
          mode: "prompt_fragment",
        }),
        skillContentPath: `${root}/skill.md`,
        skillDeliveryAgent: "b",
      },
      config,
    );
    expect(options.agent?.skillInstructionsByAgent).toBeDefined();
    expect(Object.keys(options.agent?.skillInstructionsByAgent ?? {})).toEqual(["b"]);
  });
});

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
      contextTokens: 0,
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
      contextTokens: 0,
    });
  });
});

describe("eval-run context suite", () => {
  it("keeps the checked-in harness and paper fixture at its declared run bound", () => {
    const suitePath = fileURLToPath(
      new URL("../../../evals/context/smoke-suite.json", import.meta.url),
    );
    const suite = loadContextEvaluationSuite(suitePath);

    expect(suite.cases).toHaveLength(5);
    expect(suite.agents).toHaveLength(2);
    expect(suite.matrix.profiles).toHaveLength(10);
    expect(estimateContextEvaluationMaxRuns(suite)).toBe(100);
    expect(suite.maxRuns).toBe(100);
  });

  it("strictly loads a versioned context-evaluation suite", () => {
    const dir = mkdtempSync(join(tmpdir(), "swarmx-context-suite-"));
    const suitePath = join(dir, "suite.json");
    writeFileSync(suitePath, JSON.stringify(contextSuiteInput()));

    expect(loadContextEvaluationSuite(suitePath)).toMatchObject({
      schemaVersion: 2,
      suiteId: "cli_context_smoke_v2",
    });

    writeFileSync(suitePath, JSON.stringify({ ...contextSuiteInput(), unknownField: true }));
    expect(() => loadContextEvaluationSuite(suitePath)).toThrow(/unknownField|unrecognized/i);
  });

  it("runs paired evaluation and exclusively writes content-free JSONL", async () => {
    const dir = mkdtempSync(join(tmpdir(), "swarmx-context-run-"));
    const suitePath = join(dir, "suite.json");
    const jsonlPath = join(dir, "runs.jsonl");
    writeFileSync(suitePath, JSON.stringify(contextSuiteInput()));
    const executor: ContextEvaluationExecutor = vi.fn(async () => ({
      output: "TOP_SECRET_RESPONSE retained TOKEN-CLI.",
      finalState: { inspected: true, patched: true, protected: "unchanged" },
      actions: [
        {
          sequence: 1,
          actionId: "apply_patch",
          status: "completed",
          recovery: false,
        },
      ],
      usage: {
        continuation: contextUsage("test-model", 100, 20),
        summary: contextUsage("summary-model", 0, 0),
      },
      completionTimeMs: 12,
      costUsd: 0.00012,
    }));

    const result = await runContextEvalSuite(
      undefined,
      { contextSuite: suitePath, contextJsonl: jsonlPath },
      { executor, now: () => new Date("2026-08-12T00:00:00.000Z") },
    );

    expect(result.report.totalRuns).toBe(1);
    expect(executor).toHaveBeenCalledTimes(1);
    const jsonl = readFileSync(jsonlPath, "utf8");
    expect(jsonl).not.toContain("TOP_SECRET_HISTORY");
    expect(jsonl).not.toContain("TOP_SECRET_RESPONSE");
    expect(jsonl).not.toContain("TOKEN-CLI");
    expect(JSON.parse(jsonl.trim())).toMatchObject({
      recordType: "context_evaluation_run",
      suiteId: "cli_context_smoke_v2",
    });
    expect(statSync(jsonlPath).mode & 0o777).toBe(0o600);
    expect(JSON.parse(formatContextEvaluationReport(result.report, true))).toMatchObject({
      suiteId: "cli_context_smoke_v2",
      totalRuns: 1,
    });

    const blockedExecutor: ContextEvaluationExecutor = vi.fn();
    await expect(
      runContextEvalSuite(
        undefined,
        { contextSuite: suitePath, contextJsonl: jsonlPath },
        { executor: blockedExecutor },
      ),
    ).rejects.toThrow(/exist|exclusive/i);
    expect(blockedExecutor).not.toHaveBeenCalled();
  });

  it("rejects context-suite option mixing before evaluation", async () => {
    const executor: ContextEvaluationExecutor = vi.fn();
    await expect(
      runContextEvalSuite(
        "do not mix",
        { contextSuite: "/tmp/suite.json", inputJson: "{}" },
        { executor },
      ),
    ).rejects.toThrow(/cannot be combined/i);
    await expect(
      runContextEvalSuite(undefined, { contextJsonl: "/tmp/runs.jsonl" }, { executor }),
    ).rejects.toThrow(/requires --context-suite/i);
    expect(executor).not.toHaveBeenCalled();

    const formatted = formatContextEvaluationError(new Error("TOP_SECRET_PROVIDER_DETAIL"));
    expect(formatted).not.toContain("TOP_SECRET_PROVIDER_DETAIL");
    expect(JSON.parse(formatted)).toMatchObject({
      recordType: "context_evaluation_error",
      failure: { kind: "infrastructure_failure", code: "provider_or_runtime_failure" },
    });
  });
});

function contextSuiteInput(): unknown {
  return {
    schemaVersion: 2,
    suiteId: "cli_context_smoke_v2",
    description: "CLI context evaluation smoke suite.",
    provenance: {
      collectedAt: "2026-08-12",
      split: "development",
      exposureRisk: "public",
      source: "repository-authored",
      retirementPolicy: "Retire leaked or non-discriminating cases.",
    },
    agents: [
      {
        agentId: "model_a",
        continuation: {
          name: "continuation_agent",
          model: "test-model",
          client: {
            apiProtocol: "openai_responses",
            contextWindowTokens: 4_096,
            maxOutputTokens: 512,
          },
        },
      },
    ],
    cases: [
      {
        caseId: "durable_constraint",
        objective: "Continue the pending patch.",
        difficulty: "medium",
        history: [
          {
            role: "user",
            kind: "message",
            content: "TOP_SECRET_HISTORY Retain TOKEN-CLI after the patch.",
          },
        ],
        currentUserMessage: "Continue from the prior state.",
        environment: {
          initialState: { inspected: true, patched: false, protected: "unchanged" },
          goalState: { patched: true },
          immutableStateKeys: ["protected"],
          actions: [
            {
              actionId: "apply_patch",
              description: "Apply the pending safe patch.",
              requires: { inspected: true },
              effects: { patched: true },
            },
          ],
        },
        scoring: {
          requiredOutputContains: ["TOKEN-CLI"],
          forbiddenOutputContains: [],
          requiredActionIds: ["apply_patch"],
          forbiddenActionIds: [],
          requiredRecoveryActionIds: [],
          maxBlockedActions: 0,
          maxRepeatedActions: 0,
        },
        provenance: {
          familyId: "durable_constraint",
          source: "repository-authored",
          collectedAt: "2026-08-12",
          split: "development",
          exposureRisk: "public",
        },
      },
    ],
    matrix: {
      profiles: ["baseline_full"],
      repetitionSeeds: [7],
      summaryFailureMode: "error",
    },
    search: { rounds: 1 },
    baselineProfile: "baseline_full",
    maxRuns: 5,
  };
}

function contextUsage(model: string, inputTokens: number, outputTokens: number) {
  return {
    inputTokens,
    outputTokens,
    reasoningTokens: 0,
    cachedInputTokens: 0,
    totalTokens: inputTokens + outputTokens,
    estimated: false,
    model,
  };
}
