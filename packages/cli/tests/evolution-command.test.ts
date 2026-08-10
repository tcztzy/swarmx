import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { canonicalSkillOptimizerConfig } from "@swarmx/core";
import { afterEach, describe, expect, it } from "vitest";
import {
  createEvolutionCliContext,
  evolutionLaunchSpec,
  launchDigestForWorker,
  parseDatasetRecords,
  resolveActiveSkillDeliveriesForAgent,
  runEvolutionStatus,
} from "../src/evolution-command.js";

const WORKER_PATH = fileURLToPath(new URL("../../../src/swarmx/worker.py", import.meta.url));
const temporaryRoots: string[] = [];

afterEach(async () => {
  while (temporaryRoots.length > 0) {
    const root = temporaryRoots.pop();
    if (root) await rmTree(root);
  }
});

async function temporaryRoot(): Promise<string> {
  const root = await mkdtemp(path.join(tmpdir(), "swarmx-cli-evolution-"));
  temporaryRoots.push(root);
  return root;
}

describe("parseDatasetRecords", () => {
  it("parses JSONL dataset records", () => {
    const records = parseDatasetRecords('{"id":"t1","input":"q"}\n{"id":"t2","input":"q2"}\n');
    expect(records).toHaveLength(2);
    expect(records[1]).toMatchObject({ id: "t2" });
  });

  it("rejects non-object lines", () => {
    expect(() => parseDatasetRecords('["not","an","object"]\n')).toThrow(/JSON objects/);
  });
});

describe("launchDigestForWorker", () => {
  it("derives a stable sha256 digest from the worker source", () => {
    const first = launchDigestForWorker(WORKER_PATH, "python3");
    const second = launchDigestForWorker(WORKER_PATH, "python3");
    expect(first).toMatch(/^sha256:[a-f0-9]{64}$/);
    expect(second).toBe(first);
  });
});

describe("evolutionLaunchSpec", () => {
  it("sets a sanitized worker environment and artifact root", () => {
    const spec = evolutionLaunchSpec({
      workerPath: WORKER_PATH,
      python: "python3",
      digest: `sha256:${"0".repeat(64)}`,
      cwd: "/tmp/swarmx-evolution-worker",
    });
    expect(spec.artifactRoot).toBe("/tmp/swarmx-evolution-worker");
    expect(spec.env.SWARMX_EVOLUTION_PATH).toBeUndefined();
    expect(spec.args).toContain(WORKER_PATH);
  });
});

describe("runEvolutionStatus", () => {
  it("reports an empty ledger", async () => {
    const root = await temporaryRoot();
    const output = runEvolutionStatus({ evolutionRoot: path.join(root, "evolution") });
    expect(output).toContain("No skill evolution records");
  });

  it("reports active pointers and candidates", async () => {
    const root = await temporaryRoot();
    const context = createEvolutionCliContext({
      taskRoot: path.join(root, "task"),
      evolutionRoot: path.join(root, "evolution"),
    });
    const taskStore = context.controlService.store;
    const baselineRef = taskStore.putBytes(Buffer.from("# Skill\n", "utf8")).ref;
    const trainRef = taskStore.putJson([
      { id: "t1", input: "q", target: "parrot", keyword: "parrot" },
    ]).ref;
    const devRef = taskStore.putJson([
      { id: "d1", input: "q", target: "parrot", keyword: "parrot" },
    ]).ref;
    const digest = launchDigestForWorker(WORKER_PATH, "python3");
    const request = {
      schemaVersion: 1,
      skillId: "math-coach",
      variantId: "math-coach:default",
      parentRevisionId: `r_${"a".repeat(64)}`,
      parentRevisionDigest: baselineRef,
      baselineContentRef: baselineRef,
      baselineContentDigest: baselineRef,
      targetAgentId: "agent",
      targetModelFingerprint: "model",
      trainDataset: {
        role: "train",
        contentRef: trainRef,
        contentDigest: trainRef,
        caseCount: 1,
        format: "swarmx.eval.jsonl",
      },
      devDataset: {
        role: "dev",
        contentRef: devRef,
        contentDigest: devRef,
        caseCount: 1,
        format: "swarmx.eval.jsonl",
      },
      optimizer: {
        optimizerId: "deterministic.v1",
        optimizerVersion: "1",
        environmentDigest: digest,
        configDigest: canonicalSkillOptimizerConfig({
          optimizerId: "deterministic.v1",
          seed: 0,
          proposer: "none",
          budget: { maxModelCalls: 0, maxTokens: 0, maxWallTimeMs: 60000 },
        }),
        seed: 0,
      },
      budget: { maxWallTimeMs: 60000, maxModelCalls: 0, maxTokens: 0, maxArtifactBytes: 262144 },
      proposer: "none",
      requestedBy: "cli-test",
    };
    const { workItem, grant } = context.service.createOptimizationWorkItem({
      request,
      launch: evolutionLaunchSpec({
        workerPath: WORKER_PATH,
        python: "python3",
        digest,
        cwd: path.join(root, "worker"),
      }),
    });
    const workerDir = path.join(root, "worker");
    await import("node:fs/promises").then((fs) => fs.mkdir(workerDir, { recursive: true }));
    await context.controlService.runWorkItem(workItem.id, {
      launch: evolutionLaunchSpec({
        workerPath: WORKER_PATH,
        python: "python3",
        digest,
        cwd: workerDir,
      }),
      grants: [grant],
    });
    context.service.ingestCandidate({ workItemId: workItem.id });
    const output = runEvolutionStatus({ evolutionRoot: path.join(root, "evolution") });
    expect(output).toContain("math-coach");
    expect(output).toContain("proposed");
  }, 30_000);
});

async function rmTree(root: string): Promise<void> {
  const { rm } = await import("node:fs/promises");
  await rm(root, { recursive: true, force: true });
}

describe("resolveActiveSkillDeliveriesForAgent", () => {
  it("returns the promoted revision for new executions and the baseline after rollback", async () => {
    const root = await temporaryRoot();
    const taskRoot = path.join(root, "task");
    const evolutionRoot = path.join(root, "evolution");
    const { AuditStore } = await import("@swarmx/core");
    const context = createEvolutionCliContext({
      taskRoot,
      evolutionRoot,
      audit: new AuditStore({ filePath: path.join(root, "audit", "audit.jsonl") }),
    });
    const taskStore = context.controlService?.store;
    if (!taskStore) throw new Error("missing task store");
    const baseline = "# Math Coach Skill\n\nAnswer the user's question.";
    const baselineRef = taskStore.putBytes(Buffer.from(baseline, "utf8")).ref;
    const trainRef = taskStore.putJson([
      { id: "t1", input: "q1", target: "parrot", keyword: "parrot" },
      { id: "t2", input: "q2", target: "parrot", keyword: "parrot" },
    ]).ref;
    const devRef = taskStore.putJson([
      { id: "d1", input: "q3", target: "parrot", keyword: "parrot" },
    ]).ref;
    const digest = launchDigestForWorker(WORKER_PATH, "python3");
    const request = {
      schemaVersion: 1,
      skillId: "math-coach",
      variantId: "math-coach:default",
      parentRevisionId: `r_${"a".repeat(64)}`,
      parentRevisionDigest: baselineRef,
      baselineContentRef: baselineRef,
      baselineContentDigest: baselineRef,
      targetAgentId: "swarmx:model-x",
      targetModelFingerprint: "model-x@v1",
      trainDataset: {
        role: "train",
        contentRef: trainRef,
        contentDigest: trainRef,
        caseCount: 2,
        format: "swarmx.eval.jsonl",
      },
      devDataset: {
        role: "dev",
        contentRef: devRef,
        contentDigest: devRef,
        caseCount: 1,
        format: "swarmx.eval.jsonl",
      },
      optimizer: {
        optimizerId: "deterministic.v1",
        optimizerVersion: "1",
        environmentDigest: digest,
        configDigest: canonicalSkillOptimizerConfig({
          optimizerId: "deterministic.v1",
          seed: 0,
          proposer: "none",
          budget: { maxModelCalls: 0, maxTokens: 0, maxWallTimeMs: 60000 },
        }),
        seed: 0,
      },
      budget: { maxWallTimeMs: 60000, maxModelCalls: 0, maxTokens: 0, maxArtifactBytes: 262144 },
      proposer: "none",
      requestedBy: "cli-test",
    };
    const workerDir = path.join(root, "worker");
    const { mkdir } = await import("node:fs/promises");
    await mkdir(workerDir, { recursive: true });
    const launch = evolutionLaunchSpec({
      workerPath: WORKER_PATH,
      python: "python3",
      digest,
      cwd: workerDir,
    });
    const { workItem, grant } = context.service.createOptimizationWorkItem({ request, launch });
    await context.controlService?.runWorkItem(workItem.id, { launch, grants: [grant] });
    const candidate = context.service.ingestCandidate({ workItemId: workItem.id });

    const beforePromotion = await resolveActiveSkillDeliveriesForAgent({
      bindings: [{ skillId: "math-coach", variantId: "math-coach:default" }],
      agentName: "agent",
      targetAgentId: "swarmx:model-x",
      evolutionRoot,
      taskRoot,
    });
    expect(beforePromotion).toEqual({});

    const holdout = [1, 2, 3, 4]
      .map((index) =>
        JSON.stringify({ caseId: `h${index}`, input: `question ${index}`, target: "parrot" }),
      )
      .join("\n");
    const { Swarm } = await import("@swarmx/core");
    const evalConfig = {
      name: "skill-eval",
      root: "agent",
      nodes: {
        agent: {
          kind: "agent",
          agent: { name: "agent", model: "model-x", client: { apiKey: "sk-test" } },
        },
      },
      edges: [],
    };
    const fakeModelCreate = (body: { messages?: Array<{ role?: string; content?: string }> }) => {
      const system = body.messages?.find((message) => message.role === "system");
      const instructions = system?.content ?? "";
      const keyword = /`([A-Za-z0-9][A-Za-z0-9._-]*)`/.exec(instructions)?.[1];
      return Promise.resolve({
        id: "chatcmpl-test",
        model: "model-x",
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
    };
    await context.service.evaluateCandidate({
      candidateId: candidate.candidateId,
      holdoutContent: holdout,
      createSwarm: (delivery) => {
        const swarm = new Swarm(evalConfig, {
          agent: { skillInstructions: [delivery] },
        });
        for (const node of swarm.nodes.values()) {
          if (node.kind === "agent" && node.agent) {
            Object.defineProperty(node.agent.client.chat.completions, "create", {
              configurable: true,
              value: fakeModelCreate,
            });
          }
        }
        return swarm;
      },
      evaluatorId: "cli-test",
      scorerFingerprint: "test.v1",
      runtimeFingerprint: "test.runtime",
      seed: 5,
      gate: { minSampleCount: 4, minQualityImprovement: 0.1, minImprovedRatio: 0.5 },
    });
    context.service.promote({
      candidateId: candidate.candidateId,
      actor: "cli-test",
      reason: "ok",
    });

    const afterPromotion = await resolveActiveSkillDeliveriesForAgent({
      bindings: [{ skillId: "math-coach", variantId: "math-coach:default" }],
      agentName: "agent",
      targetAgentId: "swarmx:model-x",
      evolutionRoot,
      taskRoot,
    });
    const deliveries = afterPromotion.agent ?? [];
    expect(deliveries).toHaveLength(1);
    expect(deliveries[0].revisionId).toBe(candidate.revisionId);
    expect(deliveries[0].contentDigest).toBe(candidate.contentDigest);
    expect(deliveries[0].content).not.toBe(baseline);

    context.service.rollback({
      skillId: "math-coach",
      targetRevisionId: request.parentRevisionId,
      actor: "cli-test",
      reason: "restore baseline",
    });
    const afterRollback = await resolveActiveSkillDeliveriesForAgent({
      bindings: [{ skillId: "math-coach", variantId: "math-coach:default" }],
      agentName: "agent",
      targetAgentId: "swarmx:model-x",
      evolutionRoot,
      taskRoot,
    });
    const rolledBack = afterRollback.agent ?? [];
    expect(rolledBack).toHaveLength(1);
    expect(rolledBack[0].revisionId).toBe(request.parentRevisionId);
    expect(rolledBack[0].content).toBe(baseline);
  }, 60_000);

  it("refuses deliveries whose variant or target agent do not match the promoted candidate", async () => {
    const root = await temporaryRoot();
    const taskRoot = path.join(root, "task");
    const evolutionRoot = path.join(root, "evolution");
    const { AuditStore } = await import("@swarmx/core");
    const context = createEvolutionCliContext({
      taskRoot,
      evolutionRoot,
      audit: new AuditStore({ filePath: path.join(root, "audit", "audit.jsonl") }),
    });
    const taskStore = context.controlService?.store;
    if (!taskStore) throw new Error("missing task store");
    const baseline = "# Math Coach Skill\n\nAnswer the user's question.";
    const baselineRef = taskStore.putBytes(Buffer.from(baseline, "utf8")).ref;
    const trainRef = taskStore.putJson([
      { id: "t1", input: "q1", target: "parrot", keyword: "parrot" },
    ]).ref;
    const devRef = taskStore.putJson([
      { id: "d1", input: "q2", target: "parrot", keyword: "parrot" },
    ]).ref;
    const digest = launchDigestForWorker(WORKER_PATH, "python3");
    const request = {
      schemaVersion: 1,
      skillId: "math-coach",
      variantId: "math-coach:default",
      parentRevisionId: `r_${"a".repeat(64)}`,
      parentRevisionDigest: baselineRef,
      baselineContentRef: baselineRef,
      baselineContentDigest: baselineRef,
      targetAgentId: "swarmx:model-x",
      targetModelFingerprint: "model-x@v1",
      trainDataset: {
        role: "train",
        contentRef: trainRef,
        contentDigest: trainRef,
        caseCount: 1,
        format: "swarmx.eval.jsonl",
      },
      devDataset: {
        role: "dev",
        contentRef: devRef,
        contentDigest: devRef,
        caseCount: 1,
        format: "swarmx.eval.jsonl",
      },
      optimizer: {
        optimizerId: "deterministic.v1",
        optimizerVersion: "1",
        environmentDigest: digest,
        configDigest: canonicalSkillOptimizerConfig({
          optimizerId: "deterministic.v1",
          seed: 0,
          proposer: "none",
          budget: { maxModelCalls: 0, maxTokens: 0, maxWallTimeMs: 60000 },
        }),
        seed: 0,
      },
      budget: { maxWallTimeMs: 60000, maxModelCalls: 0, maxTokens: 0, maxArtifactBytes: 262144 },
      proposer: "none",
      requestedBy: "cli-test",
    };
    const workerDir = path.join(root, "worker");
    const { mkdir } = await import("node:fs/promises");
    await mkdir(workerDir, { recursive: true });
    const launch = evolutionLaunchSpec({
      workerPath: WORKER_PATH,
      python: "python3",
      digest,
      cwd: workerDir,
    });
    const { workItem, grant } = context.service.createOptimizationWorkItem({ request, launch });
    await context.controlService?.runWorkItem(workItem.id, { launch, grants: [grant] });
    const candidate = context.service.ingestCandidate({ workItemId: workItem.id });
    const { Swarm } = await import("@swarmx/core");
    const holdout = [1, 2, 3, 4]
      .map((index) =>
        JSON.stringify({ caseId: `h${index}`, input: `question ${index}`, target: "parrot" }),
      )
      .join("\n");
    const evalConfig = {
      name: "skill-eval",
      root: "agent",
      nodes: {
        agent: {
          kind: "agent",
          agent: { name: "agent", model: "model-x", client: { apiKey: "sk-test" } },
        },
      },
      edges: [],
    };
    const fakeModelCreate = (body: { messages?: Array<{ role?: string; content?: string }> }) => {
      const system = body.messages?.find((message) => message.role === "system");
      const instructions = system?.content ?? "";
      const keyword = /`([A-Za-z0-9][A-Za-z0-9._-]*)`/.exec(instructions)?.[1];
      return Promise.resolve({
        id: "chatcmpl-test",
        model: "model-x",
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
    };
    await context.service.evaluateCandidate({
      candidateId: candidate.candidateId,
      holdoutContent: holdout,
      createSwarm: (delivery) => {
        const swarm = new Swarm(evalConfig, {
          agent: { skillInstructions: [delivery] },
        });
        for (const node of swarm.nodes.values()) {
          if (node.kind === "agent" && node.agent) {
            Object.defineProperty(node.agent.client.chat.completions, "create", {
              configurable: true,
              value: fakeModelCreate,
            });
          }
        }
        return swarm;
      },
      evaluatorId: "cli-test",
      scorerFingerprint: "test.v1",
      runtimeFingerprint: "test.runtime",
      seed: 6,
      gate: { minSampleCount: 4, minQualityImprovement: 0.1, minImprovedRatio: 0.5 },
    });
    context.service.promote({
      candidateId: candidate.candidateId,
      actor: "cli-test",
      reason: "ok",
    });
    await expect(
      resolveActiveSkillDeliveriesForAgent({
        bindings: [{ skillId: "math-coach", variantId: "wrong-variant" }],
        agentName: "agent",
        targetAgentId: "swarmx:model-x",
        evolutionRoot,
        taskRoot,
      }),
    ).rejects.toThrow(/wrong-variant/i);
    await expect(
      resolveActiveSkillDeliveriesForAgent({
        bindings: [{ skillId: "math-coach", variantId: "math-coach:default" }],
        agentName: "agent",
        targetAgentId: "someone-else",
        evolutionRoot,
        taskRoot,
      }),
    ).rejects.toThrow(/targets agent/i);
  }, 60_000);

  it("rejects a digest mismatch on the active pointer content", async () => {
    const root = await temporaryRoot();
    const taskRoot = path.join(root, "task");
    const evolutionRoot = path.join(root, "evolution");
    const { SkillEvolutionStore, TaskRuntimeStore } = await import("@swarmx/core");
    const ledger = new SkillEvolutionStore({ rootDir: evolutionRoot });
    const taskStore = new TaskRuntimeStore({ rootDir: taskRoot });
    const contentRef = taskStore.putBytes(Buffer.from("# Skill\n", "utf8")).ref;
    const request = {
      schemaVersion: 1,
      skillId: "math-coach",
      variantId: "math-coach:default",
      parentRevisionId: `r_${"a".repeat(64)}`,
      parentRevisionDigest: contentRef,
      baselineContentRef: contentRef,
      baselineContentDigest: contentRef,
      targetAgentId: "agent",
      targetModelFingerprint: "model",
      trainDataset: {
        role: "train",
        contentRef: `sha256:${"1".repeat(64)}`,
        contentDigest: `sha256:${"1".repeat(64)}`,
        caseCount: 1,
        format: "swarmx.eval.jsonl",
      },
      devDataset: {
        role: "dev",
        contentRef: `sha256:${"2".repeat(64)}`,
        contentDigest: `sha256:${"2".repeat(64)}`,
        caseCount: 1,
        format: "swarmx.eval.jsonl",
      },
      optimizer: {
        optimizerId: "deterministic.v1",
        optimizerVersion: "1",
        environmentDigest: `sha256:${"4".repeat(64)}`,
        configDigest: `sha256:${"5".repeat(64)}`,
        seed: 0,
      },
      budget: { maxWallTimeMs: 60000, maxModelCalls: 0, maxTokens: 0, maxArtifactBytes: 262144 },
      proposer: "none",
      requestedBy: "test",
    };
    const now = new Date().toISOString();
    const anchor = {
      schemaVersion: 1,
      recordId: "evl_0",
      kind: "optimization_requested",
      timestamp: now,
      idempotencyKey: "k0",
      payload: { requestId: "svr_1", requestedBy: "test", request },
    };
    const candidate = {
      schemaVersion: 1,
      recordId: "evl_1",
      kind: "candidate_created",
      timestamp: now,
      idempotencyKey: "k1",
      payload: {
        manifest: {
          schemaVersion: 1,
          candidateId: "skc_candidate1",
          skillId: "math-coach",
          variantId: "math-coach:default",
          revisionId: `r_${"c".repeat(64)}`,
          parentRevisionId: `r_${"a".repeat(64)}`,
          parentRevisionDigest: contentRef,
          contentRef: contentRef,
          contentDigest: contentRef,
          contentSizeBytes: 10,
          mediaType: "text/markdown",
          targetAgentId: "agent",
          targetModelFingerprint: "model",
          optimizer: request.optimizer,
          trainDatasetDigest: `sha256:${"1".repeat(64)}`,
          devDatasetDigest: `sha256:${"2".repeat(64)}`,
          staticChecks: {
            contentDigestVerified: true,
            parentRevisionDigestMatches: true,
            lineageMatchesRequest: true,
            instructionDeltaPresent: true,
            sizeWithinBudget: true,
            deliverySupported: true,
            secretScan: { passed: true, findings: [] },
          },
          createdAt: now,
          status: "proposed",
        },
      },
    };
    const evaluating = {
      schemaVersion: 1,
      recordId: "evl_1b",
      kind: "candidate_status_changed",
      timestamp: now,
      idempotencyKey: "k1b",
      payload: {
        candidateId: "skc_candidate1",
        from: "proposed",
        to: "evaluating",
        reason: "eval",
      },
    };
    const evaluation = {
      schemaVersion: 1,
      recordId: "evl_2",
      kind: "evaluation_recorded",
      timestamp: now,
      idempotencyKey: "k2",
      payload: {
        evaluationId: "ske_eval1",
        manifest: {
          schemaVersion: 1,
          evaluationId: "ske_eval1",
          candidateId: "skc_candidate1",
          candidateRevisionId: `r_${"c".repeat(64)}`,
          baselineRevisionId: `r_${"a".repeat(64)}`,
          holdoutContentRef: `sha256:${"3".repeat(64)}`,
          holdoutContentDigest: `sha256:${"3".repeat(64)}`,
          holdoutCaseCount: 4,
          evaluatorId: "test",
          scorerFingerprint: "test.v1",
          runtimeFingerprint: "test.runtime",
          seed: 1,
          sampleCount: 4,
          samplesRef: `sha256:${"6".repeat(64)}`,
          baseline: { quality: 0.25, safety: 1, failureRate: 0, latencyMs: 10, contextTokens: 100 },
          candidate: {
            quality: 0.75,
            safety: 1,
            failureRate: 0,
            latencyMs: 10,
            contextTokens: 100,
          },
          verdict: "eligible",
          reasons: ["gate:g"],
          gate: { minSampleCount: 4, minQualityImprovement: 0.05, minImprovedRatio: 0.5 },
          completedAt: now,
        },
      },
    };
    const promotion = {
      schemaVersion: 1,
      recordId: "evl_3",
      kind: "promotion_recorded",
      timestamp: now,
      idempotencyKey: "k3",
      payload: {
        receipt: {
          schemaVersion: 1,
          receiptId: "skp_test1",
          skillId: "math-coach",
          decision: "promote",
          gate: "human",
          candidateId: "skc_candidate1",
          candidateRevisionId: `r_${"c".repeat(64)}`,
          parentRevisionId: `r_${"a".repeat(64)}`,
          evaluationRunId: "ske_eval1",
          casExpectedRevisionId: `r_${"a".repeat(64)}`,
          previousRevisionId: null,
          newRevisionId: `r_${"c".repeat(64)}`,
          actor: "test",
          reason: "ok",
          idempotencyKey: "k3",
          decidedAt: now,
        },
        promotedRevision: {
          revisionId: `r_${"c".repeat(64)}`,
          contentRef: contentRef,
          contentDigest: `sha256:${"0".repeat(64)}`,
        },
      },
    };
    const staged = {
      schemaVersion: 1,
      recordId: "evl_2b",
      kind: "candidate_status_changed",
      timestamp: now,
      idempotencyKey: "k2b",
      payload: {
        candidateId: "skc_candidate1",
        from: "evaluating",
        to: "staged",
        reason: "eligible",
        evaluationRunId: "ske_eval1",
      },
    };
    expect(() =>
      ledger.append([anchor, candidate, evaluating, evaluation, staged, promotion]),
    ).toThrow(/PROMOTED_CONTENT_MISMATCH|coordinates do not match/i);
    expect(ledger.state().activePointers["math-coach"]).toBeUndefined();
  }, 30_000);
});
