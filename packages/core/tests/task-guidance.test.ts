import { describe, expect, it } from "vitest";
import { HARNESSES } from "../src/harness.js";
import { MODELS } from "../src/model-capabilities.js";
import {
  getTaskGuidanceForAgent,
  getTaskGuidanceForHarness,
  getTaskGuidanceForModel,
  getTaskGuidanceForTask,
  TASK_GUIDANCE_CATALOG,
  TaskGuidanceCatalogSchema,
  TaskGuidanceRecordSchema,
} from "../src/task-guidance.js";

describe("source-dated Task guidance", () => {
  it("ships a fully referenced 2026-08-12 evidence snapshot", () => {
    expect(TaskGuidanceCatalogSchema.parse(TASK_GUIDANCE_CATALOG)).toBeDefined();
    expect(TASK_GUIDANCE_CATALOG.sources.map((source) => source.id)).toEqual([
      "livebench-2026-06-25",
      "terminal-bench-2.1",
      "swe-bench-verified",
      "bfcl-v4-2025.12.17",
    ]);

    const sourceIds = new Set(TASK_GUIDANCE_CATALOG.sources.map((source) => source.id));
    for (const source of TASK_GUIDANCE_CATALOG.sources) {
      expect(source.checkedAt).toBe("2026-08-12");
      expect(source.url).toMatch(/^https:\/\//);
    }
    for (const record of TASK_GUIDANCE_CATALOG.records) {
      expect(TaskGuidanceRecordSchema.parse(record)).toBeDefined();
      expect(record.reviewedAt).toBe("2026-08-12");
      expect(record.limitations.length).toBeGreaterThan(0);
      expect(record.evidence.every((item) => sourceIds.has(item.sourceId))).toBe(true);
      expect(record.verdict).not.toBe("unsupported");
    }
  });

  it("targets only built-in Model and Harness identities", () => {
    const modelIds = new Set(MODELS.map((model) => model.id));
    const harnessIds = new Set(Object.keys(HARNESSES));

    for (const record of TASK_GUIDANCE_CATALOG.records) {
      if (record.target.kind === "model" || record.target.kind === "agent") {
        expect(modelIds.has(record.target.modelId)).toBe(true);
      }
      if (record.target.kind === "harness" || record.target.kind === "agent") {
        expect(harnessIds.has(record.target.harnessId)).toBe(true);
      }
    }
  });

  it("rejects duplicate ids and dangling evidence references", () => {
    const source = {
      id: "source",
      kind: "benchmark_leaderboard",
      title: "Benchmark",
      publisher: "Publisher",
      url: "https://example.test/leaderboard",
      version: "v1",
      checkedAt: "2026-08-12",
      dynamic: true,
    } as const;
    const record = {
      id: "record",
      target: { kind: "model", modelId: "model" },
      taskFamilies: ["general"],
      verdict: "suitable",
      confidence: "medium",
      summary: "Scoped positive evidence.",
      reviewedAt: "2026-08-12",
      conditions: { benchmarkConfiguration: "Test configuration." },
      evidence: [
        {
          sourceId: "source",
          scope: "model",
          benchmarkTask: "overall",
          metric: "score",
          value: 1,
          unit: "points",
          evaluatedModel: "Model",
        },
      ],
      limitations: ["Test limitation."],
    } as const;

    expect(() =>
      TaskGuidanceCatalogSchema.parse({
        schemaVersion: 1,
        sources: [source, source],
        records: [record, record],
      }),
    ).toThrow(/Duplicate/i);
    expect(() =>
      TaskGuidanceCatalogSchema.parse({
        schemaVersion: 1,
        sources: [source],
        records: [
          {
            ...record,
            evidence: [{ ...record.evidence[0], sourceId: "missing" }],
          },
        ],
      }),
    ).toThrow(/Unknown Task guidance source/i);
  });

  it("requires an evaluated Harness for whole-system evidence", () => {
    expect(() =>
      TaskGuidanceRecordSchema.parse({
        id: "invalid-agent-evidence",
        target: { kind: "agent", harnessId: "codex", modelId: "gpt-5.5" },
        taskFamilies: ["terminal_work"],
        verdict: "suitable",
        confidence: "medium",
        summary: "Invalid evidence.",
        reviewedAt: "2026-08-12",
        conditions: { benchmarkConfiguration: "Test configuration." },
        evidence: [
          {
            sourceId: "terminal-bench-2.1",
            scope: "agent_model",
            benchmarkTask: "terminal-bench@2.1",
            metric: "accuracy",
            value: 83.1,
            unit: "percent",
            evaluatedModel: "GPT-5.5",
          },
        ],
        limitations: ["Test limitation."],
      }),
    ).toThrow(/Harness/i);
  });

  it("queries exact layers without synthesizing a score", () => {
    expect(
      getTaskGuidanceForModel("gpt-5.6-sol").every((record) => record.target.kind === "model"),
    ).toBe(true);
    expect(
      getTaskGuidanceForHarness("codex").every((record) => record.target.kind === "harness"),
    ).toBe(true);

    const terminal = getTaskGuidanceForAgent("codex", "gpt-5.5", "terminal_work");
    expect(terminal.map((record) => record.target.kind)).toEqual(["agent", "harness"]);
    expect(terminal[0]).toMatchObject({ verdict: "preferred", confidence: "medium" });
    expect(terminal[0]?.limitations.join(" ")).toMatch(/direct Codex app-server transport/i);

    expect(getTaskGuidanceForAgent("pi", "gpt-5.5", "terminal_work")).toEqual([]);
  });

  it("sorts task guidance deterministically and treats missing records as unrated", () => {
    const general = getTaskGuidanceForTask("general");
    const firstSuitable = general.findIndex((record) => record.verdict === "suitable");
    expect(firstSuitable).toBeGreaterThan(0);
    expect(general.slice(0, firstSuitable).every((record) => record.verdict === "preferred")).toBe(
      true,
    );
    expect(getTaskGuidanceForModel("deepseek-reasoner")).toEqual([]);
  });
});
