import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import type { SkillInstructionDelivery } from "../src/skill-delivery.js";
import {
  aggregateSkillEvaluation,
  type PairedSkillEvaluationCase,
  runPairedSkillEvaluation,
  scorePassed,
  scoreSafetyViolation,
} from "../src/skill-evaluation.js";
import { Swarm } from "../src/swarm.js";
import type { SwarmConfig } from "../src/types.js";

const CONFIG: SwarmConfig = {
  name: "eval",
  root: "agent",
  nodes: {
    agent: {
      kind: "agent",
      agent: {
        name: "agent",
        model: "gpt-test",
        client: { apiKey: "sk-test" },
        instructions: "You are a helpful assistant.",
      },
    },
  },
  edges: [],
};

function delivery(keyword: string, digestHex: string): SkillInstructionDelivery {
  const content = `# Skill\n\nUse \`${keyword}\` in your answer.`;
  return {
    skillId: "skill",
    variantId: "skill:default",
    revisionId: `r_${digestHex}`,
    contentDigest: `sha256:${createHash("sha256").update(content).digest("hex")}`,
    mode: "prompt_fragment",
    content,
  };
}

const observedInstructions: string[] = [];

function fakeCreate(body: {
  messages?: Array<{ role?: string; content?: string }>;
}): Promise<unknown> {
  const system = body.messages?.find((message) => message.role === "system");
  const instructions = system?.content ?? "";
  observedInstructions.push(instructions);
  const keyword = /`([A-Za-z0-9][A-Za-z0-9._-]*)`/.exec(instructions)?.[1];
  return Promise.resolve({
    id: "chatcmpl-eval",
    model: "gpt-test",
    object: "chat.completion",
    created: 0,
    choices: [
      {
        index: 0,
        finish_reason: "stop",
        message: { role: "assistant", content: keyword ?? "nope" },
      },
    ],
    usage: { prompt_tokens: 100, completion_tokens: 10, total_tokens: 110 },
  });
}

function createEvalSwarm(deliveryInput: SkillInstructionDelivery): Swarm {
  const swarm = new Swarm(CONFIG, { agent: { skillInstructions: [deliveryInput] } });
  for (const node of swarm.nodes.values()) {
    if (node.kind === "agent" && node.agent) {
      Object.defineProperty(node.agent.client.chat.completions, "create", {
        configurable: true,
        value: fakeCreate,
      });
    }
  }
  return swarm;
}

const cases: PairedSkillEvaluationCase[] = [1, 2, 3, 4].map((index) => ({
  caseId: `h${index}`,
  input: `question ${index}`,
  target: "parrot",
}));

describe("runPairedSkillEvaluation", () => {
  it("runs baseline and candidate through the same swarm path with different instructions", async () => {
    observedInstructions.length = 0;
    const result = await runPairedSkillEvaluation({
      evaluationId: "ske_eval1",
      candidateId: "skc_candidate1",
      holdoutContentDigest: `sha256:${"1".repeat(64)}`,
      holdoutCaseCount: cases.length,
      baselineDelivery: delivery("nope", "a".repeat(64)),
      candidateDelivery: delivery("parrot", "b".repeat(64)),
      cases,
      createSwarm: createEvalSwarm,
      evaluatorId: "test",
      scorerFingerprint: "test.v1",
      runtimeFingerprint: "test.runtime",
      seed: 3,
      gate: { minSampleCount: 4, minQualityImprovement: 0.1, minImprovedRatio: 0.5 },
    });
    expect(result.manifest.verdict).toBe("eligible");
    expect(result.manifest.baseline.quality).toBe(0);
    expect(result.manifest.candidate.quality).toBe(1);
    expect(result.manifest.candidateRevisionId).toBe(`r_${"b".repeat(64)}`);
    expect(result.manifest.baselineRevisionId).toBe(`r_${"a".repeat(64)}`);
    const instructions = new Set(observedInstructions);
    expect([...instructions].some((text) => text.includes("parrot"))).toBe(true);
    expect([...instructions].some((text) => !text.includes("parrot"))).toBe(true);
  }, 20_000);

  it("keeps the per-case order deterministic for a fixed paired seed", async () => {
    const run = async () => {
      const result = await runPairedSkillEvaluation({
        evaluationId: "ske_eval2",
        candidateId: "skc_candidate2",
        holdoutContentDigest: `sha256:${"2".repeat(64)}`,
        holdoutCaseCount: cases.length,
        baselineDelivery: delivery("nope", "a".repeat(64)),
        candidateDelivery: delivery("parrot", "b".repeat(64)),
        cases,
        createSwarm: createEvalSwarm,
        evaluatorId: "test",
        scorerFingerprint: "test.v1",
        runtimeFingerprint: "test.runtime",
        seed: 3,
        gate: { minSampleCount: 4, minQualityImprovement: 0.1, minImprovedRatio: 0.5 },
      });
      return result.samples.map((sample) => sample.candidateRanFirst).join(",");
    };
    const first = await run();
    const second = await run();
    expect(first).toBe(second);
    expect(first.split(",")).toHaveLength(4);
  }, 20_000);
});

describe("scorePassed and scoreSafetyViolation", () => {
  it("scores targets, expected substrings, and safety flags deterministically", () => {
    const caseItem: PairedSkillEvaluationCase = {
      caseId: "c1",
      input: "x",
      target: "parrot",
      safetyFlag: "UNSAFE",
    };
    expect(scorePassed(caseItem, "The answer is parrot")).toBe(true);
    expect(scorePassed(caseItem, "nope")).toBe(false);
    expect(scoreSafetyViolation(caseItem, "UNSAFE content")).toBe(true);
    expect(scoreSafetyViolation(caseItem, "parrot")).toBe(false);
  });
});

describe("aggregateSkillEvaluation", () => {
  it("rejects when candidate samples regress even with a quality gain", () => {
    const samples = [
      {
        caseId: "h1",
        baseline: {
          passed: false,
          safetyPassed: true,
          contextTokens: 100,
          latencyMs: 10,
          failed: false,
        },
        candidate: {
          passed: true,
          safetyPassed: false,
          contextTokens: 100,
          latencyMs: 10,
          failed: false,
        },
        candidateRanFirst: false,
      },
      {
        caseId: "h2",
        baseline: {
          passed: false,
          safetyPassed: true,
          contextTokens: 100,
          latencyMs: 10,
          failed: false,
        },
        candidate: {
          passed: true,
          safetyPassed: true,
          contextTokens: 100,
          latencyMs: 10,
          failed: false,
        },
        candidateRanFirst: false,
      },
      {
        caseId: "h3",
        baseline: {
          passed: false,
          safetyPassed: true,
          contextTokens: 100,
          latencyMs: 10,
          failed: false,
        },
        candidate: {
          passed: true,
          safetyPassed: true,
          contextTokens: 100,
          latencyMs: 10,
          failed: false,
        },
        candidateRanFirst: false,
      },
      {
        caseId: "h4",
        baseline: {
          passed: false,
          safetyPassed: true,
          contextTokens: 100,
          latencyMs: 10,
          failed: false,
        },
        candidate: {
          passed: true,
          safetyPassed: true,
          contextTokens: 100,
          latencyMs: 10,
          failed: false,
        },
        candidateRanFirst: false,
      },
    ];
    const manifest = aggregateSkillEvaluation({
      evaluationId: "ske_eval3",
      candidateId: "skc_candidate3",
      holdoutContentDigest: `sha256:${"3".repeat(64)}`,
      holdoutCaseCount: cases.length,
      baselineDelivery: delivery("nope", "a".repeat(64)),
      candidateDelivery: delivery("parrot", "b".repeat(64)),
      cases,
      evaluatorId: "test",
      scorerFingerprint: "test.v1",
      runtimeFingerprint: "test.runtime",
      seed: 3,
      gate: { minSampleCount: 4, minQualityImprovement: 0.1, minImprovedRatio: 0.5 },
      samples,
    });
    expect(manifest.verdict).toBe("rejected");
    expect(manifest.candidate.safety).toBeLessThan(manifest.baseline.safety);
  });
});
