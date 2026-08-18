import { z } from "zod";
import { type HarnessCatalog, staticHarnessCatalog } from "./harness.js";
import { MODELS } from "./model-capabilities.js";

export const TaskGuidanceTaskFamilySchema = z.enum([
  "general",
  "reasoning",
  "coding",
  "agentic_coding",
  "repository_coding",
  "terminal_work",
  "tool_use",
  "mathematics",
  "data_analysis",
  "language",
  "instruction_following",
]);

export const TaskGuidanceVerdictSchema = z.enum(["preferred", "suitable", "weak"]);
export const TaskGuidanceConfidenceSchema = z.enum(["low", "medium", "high"]);
export const TaskGuidanceEvidenceScopeSchema = z.enum(["model", "agent_model"]);

export const TaskGuidanceSourceSchema = z
  .object({
    id: z.string().min(1),
    kind: z.literal("benchmark_leaderboard"),
    title: z.string().min(1),
    publisher: z.string().min(1),
    url: z.string().url(),
    version: z.string().min(1),
    publishedAt: z.string().date().optional(),
    updatedAt: z.string().date().optional(),
    checkedAt: z.string().date(),
    dynamic: z.boolean(),
  })
  .strict();

export const TaskGuidanceTargetSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("model"), modelId: z.string().min(1) }).strict(),
  z.object({ kind: z.literal("harness"), harnessId: z.string().min(1) }).strict(),
  z
    .object({
      kind: z.literal("agent"),
      harnessId: z.string().min(1),
      modelId: z.string().min(1),
    })
    .strict(),
]);

export const TaskGuidanceConditionsSchema = z
  .object({
    reasoningEffort: z.string().min(1).optional(),
    apiMode: z.string().min(1).optional(),
    benchmarkHarness: z.string().min(1).optional(),
    benchmarkConfiguration: z.string().min(1),
  })
  .strict();

export const TaskGuidanceEvidenceSchema = z
  .object({
    sourceId: z.string().min(1),
    scope: TaskGuidanceEvidenceScopeSchema,
    benchmarkTask: z.string().min(1),
    metric: z.string().min(1),
    value: z.number().finite().nonnegative(),
    unit: z.enum(["points", "percent"]),
    rank: z.number().int().positive().optional(),
    evaluatedModel: z.string().min(1),
    evaluatedHarness: z.string().min(1).optional(),
    reasoningEffort: z.string().min(1).optional(),
    apiMode: z.string().min(1).optional(),
    evaluatedAt: z.string().date().optional(),
    note: z.string().min(1).optional(),
  })
  .strict()
  .superRefine((evidence, ctx) => {
    if (evidence.scope === "agent_model" && !evidence.evaluatedHarness) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["evaluatedHarness"],
        message: "Whole-system Agent x Model evidence must identify the evaluated Harness.",
      });
    }
    if (evidence.scope === "model" && evidence.evaluatedHarness) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["evaluatedHarness"],
        message: "Model-scoped evidence cannot identify a Harness.",
      });
    }
  });

export const TaskGuidanceRecordSchema = z
  .object({
    id: z.string().min(1),
    target: TaskGuidanceTargetSchema,
    taskFamilies: z.array(TaskGuidanceTaskFamilySchema).min(1),
    verdict: TaskGuidanceVerdictSchema,
    confidence: TaskGuidanceConfidenceSchema,
    summary: z.string().min(1),
    reviewedAt: z.string().date(),
    conditions: TaskGuidanceConditionsSchema,
    evidence: z.array(TaskGuidanceEvidenceSchema).min(1),
    limitations: z.array(z.string().min(1)).min(1),
  })
  .strict()
  .superRefine((record, ctx) => {
    if (new Set(record.taskFamilies).size !== record.taskFamilies.length) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["taskFamilies"],
        message: "Task guidance task families must be unique.",
      });
    }
  });

export const TaskGuidanceCatalogSchema = z
  .object({
    schemaVersion: z.literal(1),
    sources: z.array(TaskGuidanceSourceSchema).min(1),
    records: z.array(TaskGuidanceRecordSchema),
  })
  .strict()
  .superRefine((catalog, ctx) => {
    const sourceIds = new Set<string>();
    for (const [index, source] of catalog.sources.entries()) {
      if (sourceIds.has(source.id)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["sources", index, "id"],
          message: `Duplicate Task guidance source id "${source.id}".`,
        });
      }
      sourceIds.add(source.id);
    }

    const recordIds = new Set<string>();
    for (const [recordIndex, record] of catalog.records.entries()) {
      if (recordIds.has(record.id)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["records", recordIndex, "id"],
          message: `Duplicate Task guidance record id "${record.id}".`,
        });
      }
      recordIds.add(record.id);
      for (const [evidenceIndex, evidence] of record.evidence.entries()) {
        if (!sourceIds.has(evidence.sourceId)) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ["records", recordIndex, "evidence", evidenceIndex, "sourceId"],
            message: `Unknown Task guidance source id "${evidence.sourceId}".`,
          });
        }
      }
    }
  });

export type TaskGuidanceTaskFamily = z.infer<typeof TaskGuidanceTaskFamilySchema>;
export type TaskGuidanceVerdict = z.infer<typeof TaskGuidanceVerdictSchema>;
export type TaskGuidanceConfidence = z.infer<typeof TaskGuidanceConfidenceSchema>;
export type TaskGuidanceEvidenceScope = z.infer<typeof TaskGuidanceEvidenceScopeSchema>;
export type TaskGuidanceSource = z.infer<typeof TaskGuidanceSourceSchema>;
export type TaskGuidanceTarget = z.infer<typeof TaskGuidanceTargetSchema>;
export type TaskGuidanceConditions = z.infer<typeof TaskGuidanceConditionsSchema>;
export type TaskGuidanceEvidence = z.infer<typeof TaskGuidanceEvidenceSchema>;
export type TaskGuidanceRecord = z.infer<typeof TaskGuidanceRecordSchema>;
export type TaskGuidanceCatalog = z.infer<typeof TaskGuidanceCatalogSchema>;

const REVIEWED_AT = "2026-08-12";
const LIVEBENCH_SOURCE_ID = "livebench-2026-06-25";
const TERMINAL_BENCH_SOURCE_ID = "terminal-bench-2.1";
const SWE_BENCH_SOURCE_ID = "swe-bench-verified";
const BFCL_SOURCE_ID = "bfcl-v4-2025.12.17";

const MODEL_BENCHMARK_LIMITATION =
  "Model-level benchmark evidence does not measure any SwarmX Harness, Provider route, or Project workload.";
const UPSTREAM_CODEX_LIMITATION =
  "The leaderboard evaluated the upstream native Harness; SwarmX runs the repository-owned direct Codex app-server transport, whose exact version and behavior were not benchmarked.";
const UPSTREAM_ACP_LIMITATION =
  "The leaderboard evaluated the upstream native Harness; SwarmX runs an ACP adapter whose exact version and behavior were not benchmarked.";
const WHOLE_SYSTEM_LIMITATION =
  "The result measures the submitted Agent x Model system and cannot be attributed to the bare Model.";

function liveBenchEvidence(
  evaluatedModel: string,
  benchmarkTask: string,
  value: number,
  reasoningEffort: string,
  rank?: number,
): TaskGuidanceEvidence {
  return {
    sourceId: LIVEBENCH_SOURCE_ID,
    scope: "model",
    benchmarkTask,
    metric: benchmarkTask === "overall" ? "overall category mean" : "category score",
    value,
    unit: "points",
    ...(rank ? { rank } : {}),
    evaluatedModel,
    reasoningEffort,
  };
}

function terminalBenchEvidence(
  evaluatedHarness: string,
  evaluatedModel: string,
  value: number,
  rank: number,
  reasoningEffort: string,
  evaluatedAt: string,
): TaskGuidanceEvidence {
  return {
    sourceId: TERMINAL_BENCH_SOURCE_ID,
    scope: "agent_model",
    benchmarkTask: "terminal-bench@2.1",
    metric: "accuracy",
    value,
    unit: "percent",
    rank,
    evaluatedHarness,
    evaluatedModel,
    reasoningEffort,
    evaluatedAt,
  };
}

function sweBenchEvidence(
  evaluatedHarness: string,
  evaluatedModel: string,
  value: number,
  reasoningEffort: string | undefined,
  evaluatedAt: string,
): TaskGuidanceEvidence {
  return {
    sourceId: SWE_BENCH_SOURCE_ID,
    scope: "agent_model",
    benchmarkTask: "SWE-bench Verified",
    metric: "resolved",
    value,
    unit: "percent",
    evaluatedHarness,
    evaluatedModel,
    ...(reasoningEffort ? { reasoningEffort } : {}),
    evaluatedAt,
  };
}

const RAW_TASK_GUIDANCE_CATALOG = {
  schemaVersion: 1,
  sources: [
    {
      id: LIVEBENCH_SOURCE_ID,
      kind: "benchmark_leaderboard",
      title: "LiveBench leaderboard",
      publisher: "LiveBench",
      url: "https://livebench.ai/",
      version: "LiveBench-2026-06-25",
      publishedAt: "2026-06-25",
      checkedAt: REVIEWED_AT,
      dynamic: true,
    },
    {
      id: TERMINAL_BENCH_SOURCE_ID,
      kind: "benchmark_leaderboard",
      title: "Terminal-Bench 2.1 leaderboard",
      publisher: "Terminal-Bench",
      url: "https://www.tbench.ai/leaderboard/terminal-bench/2.1",
      version: "terminal-bench@2.1 leaderboard snapshot checked 2026-08-12",
      publishedAt: "2026-05-06",
      checkedAt: REVIEWED_AT,
      dynamic: true,
    },
    {
      id: SWE_BENCH_SOURCE_ID,
      kind: "benchmark_leaderboard",
      title: "SWE-bench Verified leaderboard",
      publisher: "SWE-bench",
      url: "https://www.swebench.com/",
      version: "SWE-bench Verified leaderboard snapshot checked 2026-08-12",
      checkedAt: REVIEWED_AT,
      dynamic: true,
    },
    {
      id: BFCL_SOURCE_ID,
      kind: "benchmark_leaderboard",
      title: "Berkeley Function Calling Leaderboard V4",
      publisher: "UC Berkeley Gorilla",
      url: "https://gorilla.cs.berkeley.edu/leaderboard",
      version: "BFCL V4 commit f7cf735; bfcl-eval 2025.12.17",
      updatedAt: "2026-04-12",
      checkedAt: REVIEWED_AT,
      dynamic: true,
    },
  ],
  records: [
    {
      id: "model-gpt-5.6-sol-general",
      target: { kind: "model", modelId: "gpt-5.6-sol" },
      taskFamilies: ["general"],
      verdict: "preferred",
      confidence: "high",
      summary: "Leading broad objective quality in the LiveBench 2026-06-25 snapshot.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "max",
        benchmarkConfiguration: "GPT-5.6 Sol Max Effort on LiveBench-2026-06-25",
      },
      evidence: [liveBenchEvidence("GPT-5.6 Sol", "overall", 82.4, "max", 1)],
      limitations: [MODEL_BENCHMARK_LIMITATION],
    },
    {
      id: "model-gpt-5.6-sol-breadth",
      target: { kind: "model", modelId: "gpt-5.6-sol" },
      taskFamilies: ["reasoning", "coding", "mathematics", "data_analysis", "language"],
      verdict: "suitable",
      confidence: "high",
      summary: "Strong objective scores across reasoning, coding, mathematics, data, and language.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "max",
        benchmarkConfiguration: "GPT-5.6 Sol Max Effort on LiveBench-2026-06-25",
      },
      evidence: [
        liveBenchEvidence("GPT-5.6 Sol", "reasoning", 91.7, "max"),
        liveBenchEvidence("GPT-5.6 Sol", "coding", 83.9, "max"),
        liveBenchEvidence("GPT-5.6 Sol", "mathematics", 96.2, "max"),
        liveBenchEvidence("GPT-5.6 Sol", "data_analysis", 79.8, "max"),
        liveBenchEvidence("GPT-5.6 Sol", "language", 87.7, "max"),
      ],
      limitations: [MODEL_BENCHMARK_LIMITATION],
    },
    {
      id: "model-claude-fable-5-general",
      target: { kind: "model", modelId: "claude-fable-5" },
      taskFamilies: ["general"],
      verdict: "preferred",
      confidence: "high",
      summary: "Second-highest broad objective score in the LiveBench 2026-06-25 snapshot.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "max",
        benchmarkConfiguration: "Claude Fable 5 Max Effort on LiveBench-2026-06-25",
      },
      evidence: [liveBenchEvidence("Claude Fable 5", "overall", 80.8, "max", 2)],
      limitations: [MODEL_BENCHMARK_LIMITATION],
    },
    {
      id: "model-claude-fable-5-breadth",
      target: { kind: "model", modelId: "claude-fable-5" },
      taskFamilies: [
        "reasoning",
        "coding",
        "mathematics",
        "data_analysis",
        "language",
        "instruction_following",
      ],
      verdict: "suitable",
      confidence: "high",
      summary: "Strong objective category scores, especially coding, mathematics, and language.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "max",
        benchmarkConfiguration: "Claude Fable 5 Max Effort on LiveBench-2026-06-25",
      },
      evidence: [
        liveBenchEvidence("Claude Fable 5", "reasoning", 89.7, "max"),
        liveBenchEvidence("Claude Fable 5", "coding", 86, "max"),
        liveBenchEvidence("Claude Fable 5", "mathematics", 96, "max"),
        liveBenchEvidence("Claude Fable 5", "data_analysis", 80.5, "max"),
        liveBenchEvidence("Claude Fable 5", "language", 90.7, "max"),
        liveBenchEvidence("Claude Fable 5", "instruction_following", 75.8, "max"),
      ],
      limitations: [MODEL_BENCHMARK_LIMITATION],
    },
    {
      id: "model-gpt-5.5-general",
      target: { kind: "model", modelId: "gpt-5.5" },
      taskFamilies: ["general"],
      verdict: "preferred",
      confidence: "high",
      summary: "Third-highest broad objective score in the LiveBench 2026-06-25 snapshot.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "xhigh",
        benchmarkConfiguration: "GPT-5.5 Thinking xHigh Effort on LiveBench-2026-06-25",
      },
      evidence: [liveBenchEvidence("GPT-5.5 Thinking", "overall", 79.9, "xhigh", 3)],
      limitations: [MODEL_BENCHMARK_LIMITATION],
    },
    {
      id: "model-gpt-5.5-breadth",
      target: { kind: "model", modelId: "gpt-5.5" },
      taskFamilies: ["reasoning", "coding", "mathematics", "data_analysis"],
      verdict: "suitable",
      confidence: "high",
      summary: "Strong reasoning, coding, mathematics, and data-analysis category results.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "xhigh",
        benchmarkConfiguration: "GPT-5.5 Thinking xHigh Effort on LiveBench-2026-06-25",
      },
      evidence: [
        liveBenchEvidence("GPT-5.5 Thinking", "reasoning", 89.7, "xhigh"),
        liveBenchEvidence("GPT-5.5 Thinking", "coding", 82.1, "xhigh"),
        liveBenchEvidence("GPT-5.5 Thinking", "mathematics", 95.9, "xhigh"),
        liveBenchEvidence("GPT-5.5 Thinking", "data_analysis", 81.6, "xhigh"),
      ],
      limitations: [MODEL_BENCHMARK_LIMITATION],
    },
    {
      id: "model-gpt-5.6-terra-general",
      target: { kind: "model", modelId: "gpt-5.6-terra" },
      taskFamilies: ["general"],
      verdict: "preferred",
      confidence: "high",
      summary: "Fourth-highest broad objective score in the LiveBench 2026-06-25 snapshot.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "max",
        benchmarkConfiguration: "GPT-5.6 Terra Max Effort on LiveBench-2026-06-25",
      },
      evidence: [liveBenchEvidence("GPT-5.6 Terra", "overall", 79.8, "max", 4)],
      limitations: [MODEL_BENCHMARK_LIMITATION],
    },
    {
      id: "model-gpt-5.6-terra-agentic",
      target: { kind: "model", modelId: "gpt-5.6-terra" },
      taskFamilies: ["reasoning", "agentic_coding", "mathematics", "data_analysis"],
      verdict: "suitable",
      confidence: "high",
      summary: "Strong objective reasoning and agentic-coding results at max effort.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "max",
        benchmarkConfiguration: "GPT-5.6 Terra Max Effort on LiveBench-2026-06-25",
      },
      evidence: [
        liveBenchEvidence("GPT-5.6 Terra", "reasoning", 90.6, "max"),
        liveBenchEvidence("GPT-5.6 Terra", "agentic_coding", 68, "max"),
        liveBenchEvidence("GPT-5.6 Terra", "mathematics", 94.9, "max"),
        liveBenchEvidence("GPT-5.6 Terra", "data_analysis", 79.3, "max"),
      ],
      limitations: [MODEL_BENCHMARK_LIMITATION],
    },
    {
      id: "model-claude-opus-4.8-general",
      target: { kind: "model", modelId: "claude-opus-4-8" },
      taskFamilies: ["general"],
      verdict: "preferred",
      confidence: "high",
      summary: "Fifth-highest broad objective score in the LiveBench 2026-06-25 snapshot.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "xhigh",
        benchmarkConfiguration: "Claude 4.8 Opus Thinking xHigh Effort on LiveBench-2026-06-25",
      },
      evidence: [liveBenchEvidence("Claude 4.8 Opus Thinking", "overall", 78.9, "xhigh", 5)],
      limitations: [MODEL_BENCHMARK_LIMITATION],
    },
    {
      id: "model-claude-opus-4.8-breadth",
      target: { kind: "model", modelId: "claude-opus-4-8" },
      taskFamilies: ["reasoning", "coding", "agentic_coding", "mathematics", "data_analysis"],
      verdict: "suitable",
      confidence: "high",
      summary: "Strong reasoning, coding, mathematics, and data-analysis category results.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "xhigh",
        benchmarkConfiguration: "Claude 4.8 Opus Thinking xHigh Effort on LiveBench-2026-06-25",
      },
      evidence: [
        liveBenchEvidence("Claude 4.8 Opus Thinking", "reasoning", 89.7, "xhigh"),
        liveBenchEvidence("Claude 4.8 Opus Thinking", "coding", 79.3, "xhigh"),
        liveBenchEvidence("Claude 4.8 Opus Thinking", "agentic_coding", 56.1, "xhigh"),
        liveBenchEvidence("Claude 4.8 Opus Thinking", "mathematics", 95.3, "xhigh"),
        liveBenchEvidence("Claude 4.8 Opus Thinking", "data_analysis", 78.3, "xhigh"),
      ],
      limitations: [MODEL_BENCHMARK_LIMITATION],
    },
    {
      id: "model-gpt-5.6-luna-coding",
      target: { kind: "model", modelId: "gpt-5.6-luna" },
      taskFamilies: ["general", "reasoning", "coding", "data_analysis"],
      verdict: "suitable",
      confidence: "high",
      summary: "Positive broad, reasoning, coding, and data-analysis results at max effort.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "max",
        benchmarkConfiguration: "GPT-5.6 Luna Max Effort on LiveBench-2026-06-25",
      },
      evidence: [
        liveBenchEvidence("GPT-5.6 Luna", "overall", 74.3, "max"),
        liveBenchEvidence("GPT-5.6 Luna", "reasoning", 85.6, "max"),
        liveBenchEvidence("GPT-5.6 Luna", "coding", 82.9, "max"),
        liveBenchEvidence("GPT-5.6 Luna", "data_analysis", 78, "max"),
      ],
      limitations: [MODEL_BENCHMARK_LIMITATION],
    },
    {
      id: "model-claude-sonnet-5-coding",
      target: { kind: "model", modelId: "claude-sonnet-5" },
      taskFamilies: ["reasoning", "coding", "mathematics"],
      verdict: "suitable",
      confidence: "high",
      summary: "Positive reasoning, coding, and mathematics results at xhigh effort.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "xhigh",
        benchmarkConfiguration: "Claude Sonnet 5 xHigh Effort on LiveBench-2026-06-25",
      },
      evidence: [
        liveBenchEvidence("Claude Sonnet 5", "reasoning", 88.7, "xhigh"),
        liveBenchEvidence("Claude Sonnet 5", "coding", 80.7, "xhigh"),
        liveBenchEvidence("Claude Sonnet 5", "mathematics", 92.9, "xhigh"),
      ],
      limitations: [MODEL_BENCHMARK_LIMITATION],
    },
    {
      id: "model-deepseek-v4-pro-reasoning",
      target: { kind: "model", modelId: "deepseek-v4-pro" },
      taskFamilies: ["reasoning", "mathematics", "data_analysis"],
      verdict: "suitable",
      confidence: "high",
      summary:
        "Positive reasoning, mathematics, and data-analysis results in the open-model route.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        benchmarkConfiguration: "DeepSeek V4 Pro open on LiveBench-2026-06-25",
      },
      evidence: [
        liveBenchEvidence("DeepSeek V4 Pro open", "reasoning", 82.7, "reported by benchmark"),
        liveBenchEvidence("DeepSeek V4 Pro open", "mathematics", 90.7, "reported by benchmark"),
        liveBenchEvidence("DeepSeek V4 Pro open", "data_analysis", 74.5, "reported by benchmark"),
      ],
      limitations: [MODEL_BENCHMARK_LIMITATION],
    },
    {
      id: "model-claude-opus-4.5-tool-use",
      target: { kind: "model", modelId: "claude-opus-4-5" },
      taskFamilies: ["tool_use"],
      verdict: "preferred",
      confidence: "high",
      summary:
        "Ranked first for overall BFCL V4 accuracy in the checked snapshot using native function calling.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        apiMode: "native function calling",
        benchmarkConfiguration: "Claude-Opus-4-5-20251101 (FC) on BFCL V4",
      },
      evidence: [
        {
          sourceId: BFCL_SOURCE_ID,
          scope: "model",
          benchmarkTask: "BFCL V4 overall",
          metric: "overall accuracy",
          value: 77.47,
          unit: "percent",
          rank: 1,
          evaluatedModel: "Claude-Opus-4-5-20251101",
          apiMode: "FC",
        },
      ],
      limitations: [
        "BFCL V4 measures its function-calling, agentic web-search, memory, and format-sensitivity tasks; it does not exercise SwarmX Project tools.",
      ],
    },
    {
      id: "model-claude-opus-4.5-repository",
      target: { kind: "model", modelId: "claude-opus-4-5" },
      taskFamilies: ["repository_coding"],
      verdict: "suitable",
      confidence: "medium",
      summary: "Positive real-repository issue-resolution evidence under a submitted coding Agent.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "medium",
        benchmarkHarness: "live-SWE-agent",
        benchmarkConfiguration: "live-SWE-agent with Claude 4.5 Opus medium on SWE-bench Verified",
      },
      evidence: [
        sweBenchEvidence("live-SWE-agent", "Claude 4.5 Opus", 79.2, "medium", "2025-12-15"),
      ],
      limitations: [WHOLE_SYSTEM_LIMITATION],
    },
    {
      id: "model-claude-opus-4.6-repository",
      target: { kind: "model", modelId: "claude-opus-4-6" },
      taskFamilies: ["repository_coding"],
      verdict: "suitable",
      confidence: "medium",
      summary: "Positive real-repository issue-resolution evidence under mini-SWE-agent.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        benchmarkHarness: "mini-SWE-agent",
        benchmarkConfiguration: "mini-SWE-agent with Claude 4.6 Opus on SWE-bench Verified",
      },
      evidence: [
        sweBenchEvidence("mini-SWE-agent", "Claude 4.6 Opus", 75.6, undefined, "2026-02-17"),
      ],
      limitations: [WHOLE_SYSTEM_LIMITATION],
    },
    {
      id: "model-gpt-5-repository",
      target: { kind: "model", modelId: "gpt-5" },
      taskFamilies: ["repository_coding"],
      verdict: "suitable",
      confidence: "medium",
      summary: "Positive real-repository issue-resolution evidence under a submitted coding Agent.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        benchmarkHarness: "Prometheus-v1.2.1",
        benchmarkConfiguration: "Prometheus-v1.2.1 with GPT-5 on SWE-bench Verified",
      },
      evidence: [sweBenchEvidence("Prometheus-v1.2.1", "GPT-5", 74.4, undefined, "2025-10-15")],
      limitations: [WHOLE_SYSTEM_LIMITATION],
    },
    {
      id: "harness-codex-terminal",
      target: { kind: "harness", harnessId: "codex" },
      taskFamilies: ["terminal_work"],
      verdict: "suitable",
      confidence: "medium",
      summary: "Multiple upstream Codex submissions show positive terminal-task evidence.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        benchmarkHarness: "Codex",
        benchmarkConfiguration: "Verified upstream Codex submissions on terminal-bench@2.1",
      },
      evidence: [
        terminalBenchEvidence("Codex", "GPT-5.5", 83.1, 2, "xhigh", "2026-05-01"),
        terminalBenchEvidence("Codex", "GPT-5.6 Terra", 78.4, 6, "max", "2026-07-11"),
        terminalBenchEvidence("Codex", "GPT-5.6 Luna", 75.7, 9, "max", "2026-07-11"),
      ],
      limitations: [UPSTREAM_CODEX_LIMITATION, WHOLE_SYSTEM_LIMITATION],
    },
    {
      id: "harness-claude-code-terminal",
      target: { kind: "harness", harnessId: "claude_code" },
      taskFamilies: ["terminal_work"],
      verdict: "suitable",
      confidence: "medium",
      summary: "Multiple upstream Claude Code submissions show positive terminal-task evidence.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        benchmarkHarness: "Claude Code",
        benchmarkConfiguration: "Verified upstream Claude Code submissions on terminal-bench@2.1",
      },
      evidence: [
        terminalBenchEvidence("Claude Code", "Fable 5", 83.8, 1, "xhigh", "2026-06-07"),
        terminalBenchEvidence("Claude Code", "Opus 4.8", 78.9, 5, "high", "2026-07-09"),
        terminalBenchEvidence("Claude Code", "Sonnet 5", 74.6, 10, "high", "2026-07-09"),
      ],
      limitations: [UPSTREAM_ACP_LIMITATION, WHOLE_SYSTEM_LIMITATION],
    },
    {
      id: "agent-claude-code-fable-5-terminal",
      target: { kind: "agent", harnessId: "claude_code", modelId: "claude-fable-5" },
      taskFamilies: ["terminal_work"],
      verdict: "preferred",
      confidence: "medium",
      summary: "The matching upstream Agent x Model label ranked first on Terminal-Bench 2.1.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "xhigh",
        benchmarkHarness: "Claude Code",
        benchmarkConfiguration: "Claude Code with Fable 5 xhigh on terminal-bench@2.1",
      },
      evidence: [terminalBenchEvidence("Claude Code", "Fable 5", 83.8, 1, "xhigh", "2026-06-07")],
      limitations: [UPSTREAM_ACP_LIMITATION],
    },
    {
      id: "agent-codex-gpt-5.5-terminal",
      target: { kind: "agent", harnessId: "codex", modelId: "gpt-5.5" },
      taskFamilies: ["terminal_work"],
      verdict: "preferred",
      confidence: "medium",
      summary: "The matching upstream Agent x Model label ranked second on Terminal-Bench 2.1.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "xhigh",
        benchmarkHarness: "Codex",
        benchmarkConfiguration: "Codex with GPT-5.5 xhigh on terminal-bench@2.1",
      },
      evidence: [terminalBenchEvidence("Codex", "GPT-5.5", 83.1, 2, "xhigh", "2026-05-01")],
      limitations: [UPSTREAM_CODEX_LIMITATION],
    },
    {
      id: "agent-claude-code-opus-4.8-terminal",
      target: { kind: "agent", harnessId: "claude_code", modelId: "claude-opus-4-8" },
      taskFamilies: ["terminal_work"],
      verdict: "preferred",
      confidence: "medium",
      summary: "The matching upstream Agent x Model label ranked fifth on Terminal-Bench 2.1.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "high",
        benchmarkHarness: "Claude Code",
        benchmarkConfiguration: "Claude Code with Opus 4.8 high on terminal-bench@2.1",
      },
      evidence: [terminalBenchEvidence("Claude Code", "Opus 4.8", 78.9, 5, "high", "2026-07-09")],
      limitations: [UPSTREAM_ACP_LIMITATION],
    },
    {
      id: "agent-codex-gpt-5.6-terra-terminal",
      target: { kind: "agent", harnessId: "codex", modelId: "gpt-5.6-terra" },
      taskFamilies: ["terminal_work"],
      verdict: "suitable",
      confidence: "medium",
      summary: "Positive upstream Agent x Model terminal-task evidence at max effort.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "max",
        benchmarkHarness: "Codex",
        benchmarkConfiguration: "Codex with GPT-5.6 Terra max on terminal-bench@2.1",
      },
      evidence: [terminalBenchEvidence("Codex", "GPT-5.6 Terra", 78.4, 6, "max", "2026-07-11")],
      limitations: [UPSTREAM_CODEX_LIMITATION],
    },
    {
      id: "agent-codex-gpt-5.6-luna-terminal",
      target: { kind: "agent", harnessId: "codex", modelId: "gpt-5.6-luna" },
      taskFamilies: ["terminal_work"],
      verdict: "suitable",
      confidence: "medium",
      summary: "Positive upstream Agent x Model terminal-task evidence at max effort.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "max",
        benchmarkHarness: "Codex",
        benchmarkConfiguration: "Codex with GPT-5.6 Luna max on terminal-bench@2.1",
      },
      evidence: [terminalBenchEvidence("Codex", "GPT-5.6 Luna", 75.7, 9, "max", "2026-07-11")],
      limitations: [UPSTREAM_CODEX_LIMITATION],
    },
    {
      id: "agent-claude-code-sonnet-5-terminal",
      target: { kind: "agent", harnessId: "claude_code", modelId: "claude-sonnet-5" },
      taskFamilies: ["terminal_work"],
      verdict: "suitable",
      confidence: "medium",
      summary: "Positive upstream Agent x Model terminal-task evidence at high effort.",
      reviewedAt: REVIEWED_AT,
      conditions: {
        reasoningEffort: "high",
        benchmarkHarness: "Claude Code",
        benchmarkConfiguration: "Claude Code with Sonnet 5 high on terminal-bench@2.1",
      },
      evidence: [terminalBenchEvidence("Claude Code", "Sonnet 5", 74.6, 10, "high", "2026-07-09")],
      limitations: [UPSTREAM_ACP_LIMITATION],
    },
  ],
} as const;

export function parseTaskGuidanceCatalog(input: unknown): TaskGuidanceCatalog {
  return TaskGuidanceCatalogSchema.parse(input);
}

export const TASK_GUIDANCE_CATALOG: TaskGuidanceCatalog = validateTaskGuidanceTargets(
  parseTaskGuidanceCatalog(RAW_TASK_GUIDANCE_CATALOG),
);

export function getTaskGuidanceForModel(
  modelId: string,
  catalog: TaskGuidanceCatalog = TASK_GUIDANCE_CATALOG,
): TaskGuidanceRecord[] {
  return catalog.records
    .filter((record) => record.target.kind === "model" && record.target.modelId === modelId)
    .sort(compareGuidance);
}

export function getTaskGuidanceForHarness(
  harnessId: string,
  catalog: TaskGuidanceCatalog = TASK_GUIDANCE_CATALOG,
): TaskGuidanceRecord[] {
  return catalog.records
    .filter((record) => record.target.kind === "harness" && record.target.harnessId === harnessId)
    .sort(compareGuidance);
}

export function getTaskGuidanceForAgent(
  harnessId: string,
  modelId: string,
  taskFamily?: TaskGuidanceTaskFamily,
  catalog: TaskGuidanceCatalog = TASK_GUIDANCE_CATALOG,
): TaskGuidanceRecord[] {
  const parsedTaskFamily = taskFamily ? TaskGuidanceTaskFamilySchema.parse(taskFamily) : undefined;
  return catalog.records
    .filter((record) => {
      if (parsedTaskFamily && !record.taskFamilies.includes(parsedTaskFamily)) return false;
      switch (record.target.kind) {
        case "agent":
          return record.target.harnessId === harnessId && record.target.modelId === modelId;
        case "model":
          return record.target.modelId === modelId;
        case "harness":
          return record.target.harnessId === harnessId;
      }
      return false;
    })
    .sort(compareLayeredGuidance);
}

export function getTaskGuidanceForTask(
  taskFamily: TaskGuidanceTaskFamily,
  catalog: TaskGuidanceCatalog = TASK_GUIDANCE_CATALOG,
): TaskGuidanceRecord[] {
  const parsedTaskFamily = TaskGuidanceTaskFamilySchema.parse(taskFamily);
  return catalog.records
    .filter((record) => record.taskFamilies.includes(parsedTaskFamily))
    .sort(compareGuidance);
}

export function validateTaskGuidanceTargets(
  catalog: TaskGuidanceCatalog,
  harnessCatalog: HarnessCatalog = staticHarnessCatalog,
): TaskGuidanceCatalog {
  const modelIds = new Set(MODELS.map((model) => model.id));
  const harnessIds = new Set(harnessCatalog.listHarnesses().map((entry) => entry.id));
  for (const record of catalog.records) {
    if (
      (record.target.kind === "model" || record.target.kind === "agent") &&
      !modelIds.has(record.target.modelId)
    ) {
      throw new Error(
        `Built-in Task guidance record "${record.id}" references unknown Model "${record.target.modelId}".`,
      );
    }
    if (
      (record.target.kind === "harness" || record.target.kind === "agent") &&
      !harnessIds.has(record.target.harnessId)
    ) {
      throw new Error(
        `Built-in Task guidance record "${record.id}" references unknown Harness "${record.target.harnessId}".`,
      );
    }
  }
  return catalog;
}

const VERDICT_ORDER: Record<TaskGuidanceVerdict, number> = {
  preferred: 0,
  suitable: 1,
  weak: 2,
};

const CONFIDENCE_ORDER: Record<TaskGuidanceConfidence, number> = {
  high: 0,
  medium: 1,
  low: 2,
};

const TARGET_ORDER: Record<TaskGuidanceTarget["kind"], number> = {
  agent: 0,
  model: 1,
  harness: 2,
};

function compareGuidance(left: TaskGuidanceRecord, right: TaskGuidanceRecord): number {
  return (
    VERDICT_ORDER[left.verdict] - VERDICT_ORDER[right.verdict] ||
    CONFIDENCE_ORDER[left.confidence] - CONFIDENCE_ORDER[right.confidence] ||
    TARGET_ORDER[left.target.kind] - TARGET_ORDER[right.target.kind] ||
    left.id.localeCompare(right.id)
  );
}

function compareLayeredGuidance(left: TaskGuidanceRecord, right: TaskGuidanceRecord): number {
  return (
    TARGET_ORDER[left.target.kind] - TARGET_ORDER[right.target.kind] || compareGuidance(left, right)
  );
}
