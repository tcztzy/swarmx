import type { CoreSwarmExecution } from "./core-runtime.js";
import type { SkillInstructionDelivery } from "./skill-delivery.js";
import { evaluateSkillCandidateVerdict, skillEvaluationGateDigest } from "./skill-evolution.js";
import {
  type SkillEvaluationGate,
  type SkillEvaluationManifest,
  SkillEvaluationManifestSchema,
  type SkillEvaluationMetrics,
  type SkillEvaluationSample,
  type SkillEvaluationSampleRun,
} from "./skill-variants.js";
import type { ModelTokenUsage } from "./types.js";

export interface PairedSkillEvaluationCase {
  caseId: string;
  input: string;
  target?: string;
  expectedOutputContains?: string;
  safetyFlag?: string;
}

export type SkillEvaluationSwarmFactory = (
  delivery: SkillInstructionDelivery,
) => Pick<CoreSwarmExecution, "executeForEval">;

export interface RunPairedSkillEvaluationOptions {
  evaluationId: string;
  candidateId: string;
  holdoutContentDigest: string;
  holdoutCaseCount: number;
  baselineDelivery: SkillInstructionDelivery;
  candidateDelivery: SkillInstructionDelivery;
  cases: readonly PairedSkillEvaluationCase[];
  createSwarm: SkillEvaluationSwarmFactory;
  evaluatorId: string;
  scorerFingerprint: string;
  runtimeFingerprint: string;
  seed: number;
  gate: SkillEvaluationGate;
  estimateCostUsd?: (usage: ModelTokenUsage) => number;
}

export interface RunPairedSkillEvaluationResult {
  samples: SkillEvaluationSample[];
  manifest: SkillEvaluationManifest;
}

function seededShuffle<T>(input: readonly T[], seed: number): T[] {
  const values = [...input];
  let state = seed >>> 0;
  const next = (): number => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x1_0000_0000;
  };
  for (let index = values.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(next() * (index + 1));
    [values[index], values[swap]] = [values[swap], values[index]];
  }
  return values;
}

/**
 * Paired baseline/candidate evaluation through the same real SwarmX path. The
 * only difference between the two executions is the request-scoped Skill
 * instruction delivery; the model, tool policy, and budget are identical.
 * The per-case execution order is seeded-randomized and actually followed, so
 * caching, rate limits, and stateful models cannot systematically favor one
 * side.
 */
export async function runPairedSkillEvaluation(
  options: RunPairedSkillEvaluationOptions,
): Promise<RunPairedSkillEvaluationResult> {
  const cases = [...options.cases];
  const samples: SkillEvaluationSample[] = [];
  for (const caseItem of cases) {
    const candidateRanFirst = seededShuffle([false, true], sampleSeed(options.seed, caseItem))[0];
    const firstRun = await runCase(
      options.createSwarm,
      candidateRanFirst ? options.candidateDelivery : options.baselineDelivery,
      caseItem,
      options.estimateCostUsd,
    );
    const secondRun = await runCase(
      options.createSwarm,
      candidateRanFirst ? options.baselineDelivery : options.candidateDelivery,
      caseItem,
      options.estimateCostUsd,
    );
    const baseline = candidateRanFirst ? secondRun : firstRun;
    const candidate = candidateRanFirst ? firstRun : secondRun;
    samples.push({ caseId: caseItem.caseId, baseline, candidate, candidateRanFirst });
  }
  const manifest = aggregateSkillEvaluation({ ...options, samples });
  return { samples, manifest };
}

async function runCase(
  createSwarm: SkillEvaluationSwarmFactory,
  delivery: SkillInstructionDelivery,
  caseItem: PairedSkillEvaluationCase,
  estimateCostUsd: ((usage: ModelTokenUsage) => number) | undefined,
): Promise<SkillEvaluationSampleRun> {
  const startedAt = Date.now();
  let contextTokens = 0;
  let costUsd: number | undefined;
  const swarm = createSwarm(delivery);
  const usageSink: (usage: ModelTokenUsage) => void = (usage) => {
    contextTokens += usage.totalTokens;
    if (estimateCostUsd) {
      costUsd = (costUsd ?? 0) + estimateCostUsd(usage);
    }
  };
  let output = "";
  let failed = false;
  try {
    const result = await swarm.executeForEval(
      { messages: [{ role: "user", content: caseItem.input }] },
      undefined,
      usageSink,
    );
    output = result.output ?? "";
    failed = Boolean(result.error);
  } catch {
    failed = true;
  }
  const latencyMs = Date.now() - startedAt;
  const passed = !failed && scorePassed(caseItem, output);
  const safetyPassed = !failed && !scoreSafetyViolation(caseItem, output);
  return { passed, safetyPassed, contextTokens, latencyMs, costUsd, failed };
}

function sampleSeed(seed: number, caseItem: PairedSkillEvaluationCase): number {
  let hash = seed >>> 0;
  for (const char of caseItem.caseId) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }
  return hash;
}

export function scorePassed(caseItem: PairedSkillEvaluationCase, output: string): boolean {
  const normalized = output.trim();
  if (caseItem.target?.trim()) {
    return normalized.includes(caseItem.target.trim());
  }
  if (caseItem.expectedOutputContains) {
    return normalized.includes(caseItem.expectedOutputContains);
  }
  return normalized.length > 0;
}

export function scoreSafetyViolation(caseItem: PairedSkillEvaluationCase, output: string): boolean {
  if (!caseItem.safetyFlag) return false;
  return output.includes(caseItem.safetyFlag);
}

interface AggregateSkillEvaluationInput
  extends Omit<RunPairedSkillEvaluationOptions, "createSwarm"> {
  samples: SkillEvaluationSample[];
}

export function aggregateSkillEvaluation(
  input: AggregateSkillEvaluationInput,
): SkillEvaluationManifest {
  const samples = input.samples;
  const baseline = aggregateMetrics(samples.map((sample) => sample.baseline));
  const candidate = aggregateMetrics(samples.map((sample) => sample.candidate));
  const gateDigest = skillEvaluationGateDigest(input.gate);
  const verdict = evaluateSkillCandidateVerdict({
    baseline,
    candidate,
    samples,
    gate: input.gate,
    gateDigest,
  });
  return SkillEvaluationManifestSchema.parse({
    schemaVersion: 1,
    evaluationId: input.evaluationId,
    candidateId: input.candidateId,
    candidateRevisionId: input.candidateDelivery.revisionId,
    baselineRevisionId: input.baselineDelivery.revisionId,
    holdoutContentRef: input.holdoutContentDigest,
    holdoutContentDigest: input.holdoutContentDigest,
    holdoutCaseCount: input.holdoutCaseCount,
    evaluatorId: input.evaluatorId,
    scorerFingerprint: input.scorerFingerprint,
    runtimeFingerprint: input.runtimeFingerprint,
    seed: input.seed,
    sampleCount: samples.length,
    samplesRef: undefined,
    baseline,
    candidate,
    verdict: verdict.verdict,
    reasons: verdict.reasons,
    gate: input.gate,
    completedAt: new Date().toISOString(),
  });
}

function aggregateMetrics(runs: readonly SkillEvaluationSampleRun[]): SkillEvaluationMetrics {
  const count = runs.length;
  const failed = runs.filter((run) => run.failed).length;
  const quality = count > 0 ? runs.filter((run) => run.passed && !run.failed).length / count : 0;
  const safety =
    count > 0 ? runs.filter((run) => run.safetyPassed && !run.failed).length / count : 0;
  const latencyMs = count > 0 ? runs.reduce((total, run) => total + run.latencyMs, 0) / count : 0;
  const contextTokens =
    count > 0 ? Math.round(runs.reduce((total, run) => total + run.contextTokens, 0) / count) : 0;
  const costUsd =
    count > 0 ? runs.reduce((total, run) => total + (run.costUsd ?? 0), 0) / count : undefined;
  return {
    quality,
    safety,
    failureRate: count > 0 ? failed / count : 1,
    latencyMs,
    contextTokens,
    costUsd,
  };
}
