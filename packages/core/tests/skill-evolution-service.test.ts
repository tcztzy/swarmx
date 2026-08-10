import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

const WORKER_PATH = fileURLToPath(new URL("../../../src/swarmx/worker.py", import.meta.url));
const EVOLUTION_PYTHON = fileURLToPath(new URL("../../../.venv/bin/python", import.meta.url));

import { type AuditInput, AuditStore } from "../src/audit.js";
import type { SkillInstructionDelivery } from "../src/skill-delivery.js";
import {
  canonicalSkillOptimizerConfig,
  SkillEvolutionCasError,
  skillCandidateRevisionId,
} from "../src/skill-evolution.js";
import {
  assertEvalSafeSwarmConfig,
  createSkillEvolutionCapabilityGateway,
  SkillEvolutionPolicyGateClosedError,
  SkillEvolutionService,
  SkillEvolutionServiceError,
} from "../src/skill-evolution-service.js";
import { SkillEvolutionStore } from "../src/skill-evolution-store.js";
import {
  type SkillEvaluationGate,
  type SkillOptimizationRequest,
  SkillPromotionReceiptSchema,
} from "../src/skill-variants.js";
import { Swarm } from "../src/swarm.js";
import { AppAttachedTaskControlService } from "../src/task-control-service.js";
import { TaskRuntimeStore } from "../src/task-runtime-store.js";
import type { TaskWorkerLaunchSpec } from "../src/task-worker-process.js";
import type { SwarmConfig } from "../src/types.js";

const temporaryRoots: string[] = [];

afterEach(async () => {
  while (temporaryRoots.length > 0) {
    const root = temporaryRoots.pop();
    if (root) await rmTree(root);
  }
});

async function temporaryRoot(prefix: string): Promise<string> {
  const root = await mkdtemp(path.join(tmpdir(), prefix));
  temporaryRoots.push(root);
  return root;
}

async function evolutionLaunchSpec(cwd: string, digest: string): Promise<TaskWorkerLaunchSpec> {
  const workerSource = await readFile(WORKER_PATH);
  const workerSha256 = createHash("sha256").update(workerSource).digest("hex");
  const canonical = JSON.stringify({ schemaVersion: 1, workerSha256, pythonLabel: "python3" });
  const computed = `sha256:${createHash("sha256").update(canonical).digest("hex")}`;
  if (computed !== digest) {
    throw new Error(`Launch digest mismatch: ${computed} != ${digest}`);
  }
  return {
    backendId: "python",
    program: process.env.SWARMX_TEST_PYTHON ?? "python3",
    args: ["-I", "-B", "-u", WORKER_PATH, "--environment-digest", digest],
    cwd,
    env: {
      PATH: process.env.PATH ?? "",
      PYTHONDONTWRITEBYTECODE: "1",
      PYTHONUNBUFFERED: "1",
      PYTHONUTF8: "1",
    },
    environmentDigest: digest,
    artifactRoot: cwd,
  };
}

const BASELINE = "# Math Coach Skill\n\nAnswer the user's question.";

interface EvolutionFixture {
  root: string;
  taskStore: TaskRuntimeStore;
  controlService: AppAttachedTaskControlService;
  service: SkillEvolutionService;
  audit: AuditStore;
  baselineRef: string;
  trainRef: string;
  devRef: string;
  launch: TaskWorkerLaunchSpec;
  digest: string;
}

async function createEvolutionFixture(
  options: { audit?: AuditStore } = {},
): Promise<EvolutionFixture> {
  const root = await temporaryRoot("swarmx-evolution-e2e-");
  const audit =
    options.audit ?? new AuditStore({ filePath: path.join(root, "audit", "audit.jsonl") });
  const taskStore = new TaskRuntimeStore({ rootDir: path.join(root, "task-runtime") });
  const gateway = createSkillEvolutionCapabilityGateway({ taskStore });
  const controlService = new AppAttachedTaskControlService({
    store: taskStore,
    capabilityGateway: gateway,
    ownerId: "controller:evolution-e2e",
  });
  const service = new SkillEvolutionService({
    ledger: new SkillEvolutionStore({ rootDir: path.join(root, "evolution") }),
    controlService,
    audit,
  });
  const baselineRef = taskStore.putBytes(Buffer.from(BASELINE, "utf8")).ref;
  const trainRecords = [
    { id: "t1", input: "q1", target: "parrot", keyword: "parrot" },
    { id: "t2", input: "q2", target: "parrot", keyword: "parrot" },
  ];
  const devRecords = [{ id: "d1", input: "q3", target: "parrot", keyword: "parrot" }];
  const trainRef = taskStore.putJson(trainRecords).ref;
  const devRef = taskStore.putJson(devRecords).ref;
  const workerDirectory = path.join(root, "worker");
  await mkdir(workerDirectory, { recursive: true });
  const workerSource = await readFile(WORKER_PATH);
  const workerSha256 = createHash("sha256").update(workerSource).digest("hex");
  const digest = `sha256:${createHash("sha256")
    .update(JSON.stringify({ schemaVersion: 1, workerSha256, pythonLabel: "python3" }))
    .digest("hex")}`;
  const launch = await evolutionLaunchSpec(workerDirectory, digest);
  return {
    root,
    taskStore,
    controlService,
    service,
    audit,
    baselineRef,
    trainRef,
    devRef,
    launch,
    digest,
  };
}

function optimizationRequest(
  fixture: EvolutionFixture,
  overrides: Partial<Record<string, unknown>> = {},
): SkillOptimizationRequest {
  const base: SkillOptimizationRequest = {
    schemaVersion: 1,
    skillId: "math-coach",
    variantId: "math-coach:default",
    parentRevisionId: `r_${"a".repeat(64)}`,
    parentRevisionDigest: fixture.baselineRef,
    baselineContentRef: fixture.baselineRef,
    baselineContentDigest: fixture.baselineRef,
    targetAgentId: "swarmx:model-x",
    targetModelFingerprint: "model-x@v1",
    trainDataset: {
      role: "train",
      contentRef: fixture.trainRef,
      contentDigest: fixture.trainRef,
      caseCount: 2,
      format: "swarmx.eval.jsonl",
    },
    devDataset: {
      role: "dev",
      contentRef: fixture.devRef,
      contentDigest: fixture.devRef,
      caseCount: 1,
      format: "swarmx.eval.jsonl",
    },
    optimizer: {
      optimizerId: "deterministic.v1",
      optimizerVersion: "1",
      environmentDigest: fixture.digest,
      configDigest: "",
      seed: 7,
    },
    budget: {
      maxWallTimeMs: 120_000,
      maxModelCalls: 0,
      maxTokens: 0,
      maxArtifactBytes: 256 * 1024,
    },
    proposer: "none",
    requestedBy: "e2e-test",
  };
  const merged = { ...base, ...overrides } as SkillOptimizationRequest;
  return {
    ...merged,
    optimizer: {
      ...merged.optimizer,
      configDigest: canonicalSkillOptimizerConfig({
        optimizerId: merged.optimizer.optimizerId,
        seed: merged.optimizer.seed,
        proposer: merged.proposer,
        budget: merged.budget,
      }),
    },
  };
}

const EVAL_CONFIG: SwarmConfig = {
  name: "skill-eval",
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

const observedInstructions: string[] = [];

function fakeModelCreate(body: {
  messages?: Array<{ role?: string; content?: string }>;
  model?: string;
}): Promise<unknown> {
  const system = body.messages?.find((message) => message.role === "system");
  const instructions = system?.content ?? "";
  observedInstructions.push(instructions);
  const keyword = /`([A-Za-z0-9][A-Za-z0-9._-]*)`/.exec(instructions)?.[1];
  return Promise.resolve({
    id: "chatcmpl-test",
    model: body.model ?? "gpt-test",
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

function createEvalSwarm(delivery: SkillInstructionDelivery): Swarm {
  const swarm = new Swarm(EVAL_CONFIG, { agent: { skillInstructions: [delivery] } });
  for (const node of swarm.nodes.values()) {
    if (node.kind === "agent" && node.agent) {
      Object.defineProperty(node.agent.client.chat.completions, "create", {
        configurable: true,
        value: fakeModelCreate,
      });
    }
  }
  return swarm;
}

const gate: SkillEvaluationGate = {
  minSampleCount: 4,
  minQualityImprovement: 0.1,
  minImprovedRatio: 0.5,
};

function holdout(keyword: string): string {
  return [1, 2, 3, 4]
    .map((index) =>
      JSON.stringify({
        caseId: `h${index}`,
        input: `question ${index}`,
        target: keyword,
        safetyFlag: "unsafe-token",
      }),
    )
    .join("\n");
}

describe("SkillEvolutionService end-to-end closed loop", () => {
  it("evolves, evaluates, stages, promotes with CAS, and delivers the new revision", async () => {
    const fixture = await createEvolutionFixture();
    const request = optimizationRequest(fixture);
    const { workItem, grant } = fixture.service.createOptimizationWorkItem({
      request,
      launch: fixture.launch,
      requestedBy: "e2e",
    });
    await fixture.controlService.runWorkItem(workItem.id, {
      launch: fixture.launch,
      grants: [grant],
    });
    const candidate = fixture.service.ingestCandidate({ workItemId: workItem.id });
    expect(candidate.status).toBe("proposed");
    expect(candidate.skillId).toBe("math-coach");
    expect(candidate.parentRevisionId).toBe(request.parentRevisionId);
    expect(candidate.optimizer.optimizerId).toBe("deterministic.v1");
    expect(candidate.staticChecks.instructionDeltaPresent).toBe(true);
    const candidateContent = fixture.taskStore.readBytes(candidate.contentRef).toString("utf8");
    expect(candidateContent).toContain("parrot");
    expect(candidateContent).not.toBe(BASELINE);
    expect(candidate.revisionId).toBe(skillCandidateRevisionId(candidate.contentDigest));

    const activeBefore = fixture.service.state().activePointers["math-coach"];
    expect(activeBefore).toBeUndefined();

    const manifest = await fixture.service.evaluateCandidate({
      candidateId: candidate.candidateId,
      holdoutContent: holdout("parrot"),
      createSwarm: createEvalSwarm,
      evaluatorId: "e2e",
      scorerFingerprint: "e2e.deterministic.v1",
      runtimeFingerprint: "e2e.runtime.v1",
      seed: 11,
      gate,
    });
    expect(manifest.verdict).toBe("eligible");
    expect(fixture.service.state().candidates[candidate.candidateId]?.status).toBe("staged");

    const receipt = fixture.service.promote({
      candidateId: candidate.candidateId,
      actor: "e2e-operator",
      reason: "Evaluation passed.",
    });
    SkillPromotionReceiptSchema.parse(receipt);
    const pointer = fixture.service.state().activePointers["math-coach"];
    expect(pointer?.revisionId).toBe(candidate.revisionId);
    expect(pointer?.contentRef).toBe(candidate.contentRef);

    const delivered = fixture.service.resolveActiveSkillDelivery({
      skillId: "math-coach",
      variantId: "math-coach:default",
      modelControl: "direct",
    });
    expect(delivered.revisionId).toBe(candidate.revisionId);
    expect(delivered.contentDigest).toBe(candidate.contentDigest);
    expect(delivered.content).toBe(candidateContent);
  }, 30_000);

  it("rejects a candidate whose instructions regress safety", async () => {
    const fixture = await createEvolutionFixture();
    const badTrainRef = fixture.taskStore.putJson([
      { id: "t1", input: "q1", target: "unsafe-token", keyword: "unsafe-token" },
      { id: "t2", input: "q2", target: "unsafe-token", keyword: "unsafe-token" },
    ]).ref;
    const request = optimizationRequest(fixture, {
      trainDataset: {
        role: "train",
        contentRef: badTrainRef,
        contentDigest: badTrainRef,
        caseCount: 2,
        format: "swarmx.eval.jsonl",
      },
    });
    const { workItem, grant } = fixture.service.createOptimizationWorkItem({
      request,
      launch: fixture.launch,
    });
    await fixture.controlService.runWorkItem(workItem.id, {
      launch: fixture.launch,
      grants: [grant],
    });
    const candidate = fixture.service.ingestCandidate({ workItemId: workItem.id });
    const manifest = await fixture.service.evaluateCandidate({
      candidateId: candidate.candidateId,
      holdoutContent: holdout("parrot"),
      createSwarm: createEvalSwarm,
      evaluatorId: "e2e",
      scorerFingerprint: "e2e.deterministic.v1",
      runtimeFingerprint: "e2e.runtime.v1",
      seed: 12,
      gate,
    });
    expect(manifest.verdict).toBe("rejected");
    expect(manifest.candidate.safety).toBeLessThan(manifest.baseline.safety);
    expect(fixture.service.state().candidates[candidate.candidateId]?.status).toBe("rejected");
    expect(() =>
      fixture.service.promote({
        candidateId: candidate.candidateId,
        actor: "e2e",
        reason: "should fail",
      }),
    ).toThrow(SkillEvolutionServiceError);
    expect(fixture.service.state().activePointers["math-coach"]).toBeUndefined();
  }, 30_000);

  it("quarantines a candidate with secret-bearing content", async () => {
    const fixture = await createEvolutionFixture();
    const secretBaseline = "# Skill\n\napi_key=sk-live-12345\n";
    const secretRef = fixture.taskStore.putBytes(Buffer.from(secretBaseline, "utf8")).ref;
    const secretTrainRef = fixture.taskStore.putJson([
      { id: "t1", input: "q", target: "k", keyword: "k" },
    ]).ref;
    const request = optimizationRequest(fixture, {
      baselineContentRef: secretRef,
      baselineContentDigest: secretRef,
      parentRevisionDigest: secretRef,
      trainDataset: {
        role: "train",
        contentRef: secretTrainRef,
        contentDigest: secretTrainRef,
        caseCount: 1,
        format: "swarmx.eval.jsonl",
      },
    });
    const { workItem, grant } = fixture.service.createOptimizationWorkItem({
      request,
      launch: fixture.launch,
    });
    await fixture.controlService.runWorkItem(workItem.id, {
      launch: fixture.launch,
      grants: [grant],
    });
    const candidate = fixture.service.ingestCandidate({ workItemId: workItem.id });
    expect(fixture.service.state().candidates[candidate.candidateId]?.status).toBe("quarantined");
    expect(candidate.staticChecks.secretScan.passed).toBe(false);
    await expect(
      fixture.service.evaluateCandidate({
        candidateId: candidate.candidateId,
        holdoutContent: holdout("k"),
        createSwarm: createEvalSwarm,
        evaluatorId: "e2e",
        scorerFingerprint: "e2e",
        runtimeFingerprint: "e2e",
        seed: 13,
        gate,
      }),
    ).rejects.toThrow(/expected proposed/);
  }, 30_000);

  it("rejects stale-parent promotion with compare-and-swap", async () => {
    const fixture = await createEvolutionFixture();
    const request = optimizationRequest(fixture);
    const { workItem, grant } = fixture.service.createOptimizationWorkItem({
      request,
      launch: fixture.launch,
    });
    await fixture.controlService.runWorkItem(workItem.id, {
      launch: fixture.launch,
      grants: [grant],
    });
    const first = fixture.service.ingestCandidate({ workItemId: workItem.id });
    await fixture.service.evaluateCandidate({
      candidateId: first.candidateId,
      holdoutContent: holdout("parrot"),
      createSwarm: createEvalSwarm,
      evaluatorId: "e2e",
      scorerFingerprint: "e2e",
      runtimeFingerprint: "e2e",
      seed: 14,
      gate,
    });
    fixture.service.promote({ candidateId: first.candidateId, actor: "e2e", reason: "ok" });

    const secondTrainRef = fixture.taskStore.putJson([
      { id: "s1", input: "q1", target: "parrot", keyword: "parrot" },
      { id: "s2", input: "q2", target: "parrot", keyword: "parrot" },
    ]).ref;
    const second = optimizationRequest(fixture, {
      trainDataset: {
        role: "train",
        contentRef: secondTrainRef,
        contentDigest: secondTrainRef,
        caseCount: 2,
        format: "swarmx.eval.jsonl",
      },
    });
    const { workItem: workItem2, grant: grant2 } = fixture.service.createOptimizationWorkItem({
      request: second,
      launch: fixture.launch,
    });
    await fixture.controlService.runWorkItem(workItem2.id, {
      launch: fixture.launch,
      grants: [grant2],
    });
    const secondCandidate = fixture.service.ingestCandidate({ workItemId: workItem2.id });
    await fixture.service.evaluateCandidate({
      candidateId: secondCandidate.candidateId,
      holdoutContent: holdout("parrot"),
      createSwarm: createEvalSwarm,
      evaluatorId: "e2e",
      scorerFingerprint: "e2e",
      runtimeFingerprint: "e2e",
      seed: 15,
      gate,
    });
    expect(fixture.service.state().candidates[secondCandidate.candidateId]?.status).toBe("staged");
    expect(() =>
      fixture.service.promote({
        candidateId: secondCandidate.candidateId,
        actor: "e2e",
        reason: "stale parent",
      }),
    ).toThrow(SkillEvolutionCasError);
    expect(fixture.service.state().activePointers["math-coach"]?.revisionId).toBe(first.revisionId);
  }, 40_000);

  it("rolls back to the retained baseline revision", async () => {
    const fixture = await createEvolutionFixture();
    const request = optimizationRequest(fixture);
    const { workItem, grant } = fixture.service.createOptimizationWorkItem({
      request,
      launch: fixture.launch,
    });
    await fixture.controlService.runWorkItem(workItem.id, {
      launch: fixture.launch,
      grants: [grant],
    });
    const candidate = fixture.service.ingestCandidate({ workItemId: workItem.id });
    await fixture.service.evaluateCandidate({
      candidateId: candidate.candidateId,
      holdoutContent: holdout("parrot"),
      createSwarm: createEvalSwarm,
      evaluatorId: "e2e",
      scorerFingerprint: "e2e",
      runtimeFingerprint: "e2e",
      seed: 16,
      gate,
    });
    fixture.service.promote({ candidateId: candidate.candidateId, actor: "e2e", reason: "ok" });
    expect(fixture.service.state().activePointers["math-coach"]?.revisionId).toBe(
      candidate.revisionId,
    );

    fixture.service.rollback({
      skillId: "math-coach",
      targetRevisionId: request.parentRevisionId,
      actor: "e2e",
      reason: "restore baseline",
    });
    const pointer = fixture.service.state().activePointers["math-coach"];
    expect(pointer?.revisionId).toBe(request.parentRevisionId);
    expect(pointer?.contentRef).toBe(fixture.baselineRef);
    const delivered = fixture.service.resolveActiveSkillDelivery({
      skillId: "math-coach",
      variantId: "math-coach:default",
      modelControl: "direct",
    });
    expect(delivered.content).toBe(BASELINE);
    expect(delivered.revisionId).toBe(request.parentRevisionId);
  }, 30_000);

  it("promotion affects only new execution snapshots", async () => {
    const fixture = await createEvolutionFixture();
    const request = optimizationRequest(fixture);
    const { workItem, grant } = fixture.service.createOptimizationWorkItem({
      request,
      launch: fixture.launch,
    });
    await fixture.controlService.runWorkItem(workItem.id, {
      launch: fixture.launch,
      grants: [grant],
    });
    const candidate = fixture.service.ingestCandidate({ workItemId: workItem.id });
    await fixture.service.evaluateCandidate({
      candidateId: candidate.candidateId,
      holdoutContent: holdout("parrot"),
      createSwarm: createEvalSwarm,
      evaluatorId: "e2e",
      scorerFingerprint: "e2e",
      runtimeFingerprint: "e2e",
      seed: 17,
      gate,
    });

    const beforePromotion: string[] = [];
    const captureCreate = (body: { messages?: Array<{ role?: string; content?: string }> }) => {
      const system = body.messages?.find((message) => message.role === "system");
      beforePromotion.push(system?.content ?? "");
      return fakeModelCreate(body);
    };
    const oldSwarm = createEvalSwarm({
      skillId: "math-coach",
      variantId: "math-coach:default",
      revisionId: request.parentRevisionId,
      contentDigest: fixture.baselineRef,
      mode: "prompt_fragment",
      content: BASELINE,
    });
    for (const node of oldSwarm.nodes.values()) {
      if (node.kind === "agent" && node.agent) {
        Object.defineProperty(node.agent.client.chat.completions, "create", {
          configurable: true,
          value: captureCreate,
        });
      }
    }
    const oldResult = await oldSwarm.executeForEval({
      messages: [{ role: "user", content: "question 1" }],
    });
    expect(oldResult.output).toBe("nope");
    expect(beforePromotion[0]).toContain(request.parentRevisionId);

    fixture.service.promote({ candidateId: candidate.candidateId, actor: "e2e", reason: "ok" });

    const newSwarm = createEvalSwarm(
      fixture.service.resolveActiveSkillDelivery({
        skillId: "math-coach",
        variantId: "math-coach:default",
        modelControl: "direct",
      }),
    );
    const newResult = await newSwarm.executeForEval({
      messages: [{ role: "user", content: "question 1" }],
    });
    expect(newResult.output).toBe("parrot");
  }, 30_000);

  it("records audit intent before the promotion effect and fails closed", async () => {
    class ThrowingAuditStore extends AuditStore {
      throwOnAppend = false;
      override append(input: AuditInput | readonly AuditInput[]): void {
        if (this.throwOnAppend) throw new Error("audit authority unavailable");
        super.append(input);
      }
    }
    const auditRoot = await temporaryRoot("swarmx-evolution-audit-");
    const audit = new ThrowingAuditStore({ filePath: path.join(auditRoot, "audit.jsonl") });
    const fixture = await createEvolutionFixture({ audit });
    const request = optimizationRequest(fixture);
    const { workItem, grant } = fixture.service.createOptimizationWorkItem({
      request,
      launch: fixture.launch,
    });
    await fixture.controlService.runWorkItem(workItem.id, {
      launch: fixture.launch,
      grants: [grant],
    });
    const candidate = fixture.service.ingestCandidate({ workItemId: workItem.id });
    await fixture.service.evaluateCandidate({
      candidateId: candidate.candidateId,
      holdoutContent: holdout("parrot"),
      createSwarm: createEvalSwarm,
      evaluatorId: "e2e",
      scorerFingerprint: "e2e",
      runtimeFingerprint: "e2e",
      seed: 18,
      gate,
    });
    const promotionsBefore = fixture.service.state().promotionReceipts.length;
    audit.throwOnAppend = true;
    expect(() =>
      fixture.service.promote({ candidateId: candidate.candidateId, actor: "e2e", reason: "x" }),
    ).toThrow("audit authority unavailable");
    audit.throwOnAppend = false;
    expect(fixture.service.state().activePointers["math-coach"]).toBeUndefined();
    expect(fixture.service.state().promotionReceipts.length).toBe(promotionsBefore);
    expect(audit.query({ action: "skill.evolution.promote" }).length).toBe(0);

    fixture.service.promote({ candidateId: candidate.candidateId, actor: "e2e", reason: "ok" });
    const events = audit.query({ action: "skill.evolution.promote" });
    const attempted = events.filter((event) => event.outcome === "attempted");
    const completed = events.filter((event) => event.outcome === "completed");
    expect(attempted.length).toBe(1);
    expect(completed.length).toBe(1);
    expect(attempted[0].metadata.candidateId).toBe(candidate.candidateId);
    expect(attempted[0].sequence).toBeLessThan(completed[0].sequence);
  }, 30_000);

  it("fails closed for the policy promotion gate", async () => {
    const fixture = await createEvolutionFixture();
    const request = optimizationRequest(fixture);
    const { workItem, grant } = fixture.service.createOptimizationWorkItem({
      request,
      launch: fixture.launch,
    });
    await fixture.controlService.runWorkItem(workItem.id, {
      launch: fixture.launch,
      grants: [grant],
    });
    const candidate = fixture.service.ingestCandidate({ workItemId: workItem.id });
    await fixture.service.evaluateCandidate({
      candidateId: candidate.candidateId,
      holdoutContent: holdout("parrot"),
      createSwarm: createEvalSwarm,
      evaluatorId: "e2e",
      scorerFingerprint: "e2e",
      runtimeFingerprint: "e2e",
      seed: 19,
      gate,
    });
    expect(() =>
      fixture.service.promote({
        candidateId: candidate.candidateId,
        actor: "policy-bot",
        reason: "auto",
        gate: "policy",
      }),
    ).toThrow(SkillEvolutionPolicyGateClosedError);
    expect(fixture.service.state().activePointers["math-coach"]).toBeUndefined();
  }, 30_000);

  it("rejects overlapping train/dev case ids when creating the optimization", async () => {
    const fixture = await createEvolutionFixture();
    const overlappingRef = fixture.taskStore.putJson([
      { id: "t1", input: "q1", target: "parrot", keyword: "parrot" },
    ]).ref;
    const request = optimizationRequest(fixture, {
      devDataset: {
        role: "dev",
        contentRef: overlappingRef,
        contentDigest: overlappingRef,
        caseCount: 1,
        format: "swarmx.eval.jsonl",
      },
    });
    expect(() =>
      fixture.service.createOptimizationWorkItem({ request, launch: fixture.launch }),
    ).toThrow(/share case ids/);
  });

  it("rejects a hidden holdout that reuses train/dev case ids", async () => {
    const fixture = await createEvolutionFixture();
    const request = optimizationRequest(fixture);
    const { workItem, grant } = fixture.service.createOptimizationWorkItem({
      request,
      launch: fixture.launch,
    });
    await fixture.controlService.runWorkItem(workItem.id, {
      launch: fixture.launch,
      grants: [grant],
    });
    const candidate = fixture.service.ingestCandidate({ workItemId: workItem.id });
    const contaminatedHoldout = [
      JSON.stringify({ caseId: "t1", input: "q", target: "parrot" }),
      JSON.stringify({ caseId: "h2", input: "q", target: "parrot" }),
    ].join("\n");
    await expect(
      fixture.service.evaluateCandidate({
        candidateId: candidate.candidateId,
        holdoutContent: contaminatedHoldout,
        createSwarm: createEvalSwarm,
        evaluatorId: "e2e",
        scorerFingerprint: "e2e",
        runtimeFingerprint: "e2e",
        seed: 20,
        gate,
      }),
    ).rejects.toThrow(/shares case ids with train\/dev/);
  }, 30_000);

  it("denies capability calls for artifacts outside the granted request", async () => {
    const fixture = await createEvolutionFixture();
    const request = optimizationRequest(fixture);
    const { workItem } = fixture.service.createOptimizationWorkItem({
      request,
      launch: fixture.launch,
    });
    const gateway = fixture.service.createCapabilityGateway();
    const outcome = await gateway.invoke({
      workItem,
      call: {
        protocolVersion: 1,
        messageId: "cap:1",
        direction: "worker_to_host",
        type: "capability_call",
        workItemId: workItem.id,
        runId: "run_evolution_deny",
        leaseId: "lease_evolution_deny",
        fencingToken: 1,
        sequence: 0,
        emittedAt: new Date().toISOString(),
        callId: "cap:test",
        grantId: "gnt_evolve_1",
        capabilityId: "skill_evolution",
        operation: "read_artifact",
        idempotencyKey: "cap:test:1",
        arguments: { ref: `sha256:${"9".repeat(64)}` },
      },
    });
    expect(outcome.status).toBe("failed");
    expect(outcome.status === "failed" && outcome.error.code).toBe("artifact_not_granted");
  });

  it("runs the GEPA sidecar through the worker and round-trips the exported candidate", async () => {
    const dspyAvailable = await import("node:child_process").then(({ execFileSync }) => {
      for (const program of [
        process.env.SWARMX_EVOLUTION_PYTHON ?? EVOLUTION_PYTHON,
        process.env.SWARMX_TEST_PYTHON ?? "python3",
      ]) {
        try {
          execFileSync(program, ["-c", "import dspy"], {
            stdio: "ignore",
            env: { ...process.env, PYTHONPATH: "" },
          });
          return true;
        } catch {
          // Try the next candidate interpreter.
        }
      }
      return false;
    });
    if (!dspyAvailable) {
      return; // Requires the locked evolution dependency group; skipped honestly.
    }
    const fixture = await createEvolutionFixture();
    const workerSource = await readFile(WORKER_PATH);
    const workerSha256 = createHash("sha256").update(workerSource).digest("hex");
    const gepaDigest = `sha256:${createHash("sha256")
      .update(JSON.stringify({ schemaVersion: 1, workerSha256, pythonLabel: "evolution" }))
      .digest("hex")}`;
    const request = optimizationRequest(fixture, {
      optimizer: {
        optimizerId: "dspy.gepa.v1",
        optimizerVersion: "1",
        environmentDigest: gepaDigest,
        configDigest: "",
        seed: 7,
      },
      proposer: "deterministic",
      budget: {
        maxWallTimeMs: 120_000,
        maxModelCalls: 24,
        maxTokens: 2000,
        maxArtifactBytes: 256 * 1024,
      },
    });
    request.optimizer.configDigest = canonicalSkillOptimizerConfig({
      optimizerId: "dspy.gepa.v1",
      seed: 7,
      proposer: "deterministic",
      budget: request.budget,
    });
    const workerDir = path.join(fixture.root, "gepa-worker");
    await mkdir(workerDir, { recursive: true });
    const launch: TaskWorkerLaunchSpec = {
      backendId: "python",
      program: process.env.SWARMX_EVOLUTION_PYTHON ?? EVOLUTION_PYTHON,
      args: ["-I", "-B", "-u", WORKER_PATH, "--environment-digest", gepaDigest],
      cwd: workerDir,
      env: {
        PATH: process.env.PATH ?? "",
        PYTHONDONTWRITEBYTECODE: "1",
        PYTHONUNBUFFERED: "1",
        PYTHONUTF8: "1",
        SWARMX_EVOLUTION_PATH: path.dirname(WORKER_PATH),
      },
      environmentDigest: gepaDigest,
      artifactRoot: workerDir,
    };
    const { workItem, grant } = fixture.service.createOptimizationWorkItem({
      request,
      launch,
    });
    await fixture.controlService.runWorkItem(workItem.id, { launch, grants: [grant] });
    const candidate = fixture.service.ingestCandidate({ workItemId: workItem.id });
    expect(candidate.status).toBe("proposed");
    expect(candidate.optimizer.optimizerId).toBe("dspy.gepa.v1");
    const candidateContent = fixture.taskStore.readBytes(candidate.contentRef).toString("utf8");
    expect(candidateContent).not.toBe(BASELINE);
    expect(candidateContent).toContain("parrot");
    expect(candidate.revisionId).toBe(skillCandidateRevisionId(candidate.contentDigest));
  }, 300_000);
});

describe("assertEvalSafeSwarmConfig", () => {
  const safe = {
    name: "safe",
    root: "agent",
    nodes: { agent: { kind: "agent", agent: { name: "agent" } } },
    edges: [],
  };
  it("accepts a single native agent without side-effect surfaces", () => {
    expect(() => assertEvalSafeSwarmConfig(safe)).not.toThrow();
  });

  it("rejects queen agents, hooks, MCP servers, tool nodes, and external backends", () => {
    expect(() =>
      assertEvalSafeSwarmConfig({
        ...safe,
        queen: { name: "queen", instructions: "x" },
      }),
    ).toThrow(/queen/i);
    expect(() => assertEvalSafeSwarmConfig({ ...safe, hooks: [{ event: "pre_run" }] })).toThrow(
      /hooks/i,
    );
    expect(() =>
      assertEvalSafeSwarmConfig({
        ...safe,
        mcpServers: { srv: { url: "http://x" } },
      }),
    ).toThrow(/MCP/i);
    expect(() =>
      assertEvalSafeSwarmConfig({
        name: "t",
        root: "tool",
        nodes: { tool: { kind: "tool", tool: { name: "t" } } },
        edges: [],
      }),
    ).toThrow(/only agent nodes/i);
    expect(() =>
      assertEvalSafeSwarmConfig({
        name: "a",
        root: "agent",
        nodes: {
          agent: {
            kind: "agent",
            agent: { name: "agent", backend: { type: "custom", program: "x" } },
          },
        },
        edges: [],
      }),
    ).toThrow(/backend/i);
  });
});

async function rmTree(root: string): Promise<void> {
  const { rm } = await import("node:fs/promises");
  await rm(root, { recursive: true, force: true });
}

describe("skill evolution safety gates", () => {
  it("rejects dspy.gepa.v1 with a zero model budget instead of defaulting", async () => {
    const fixture = await createEvolutionFixture();
    const workerSource = await readFile(WORKER_PATH);
    const workerSha256 = createHash("sha256").update(workerSource).digest("hex");
    const gepaDigest = `sha256:${createHash("sha256")
      .update(JSON.stringify({ schemaVersion: 1, workerSha256, pythonLabel: "evolution" }))
      .digest("hex")}`;
    const request = optimizationRequest(fixture, {
      optimizer: {
        optimizerId: "dspy.gepa.v1",
        optimizerVersion: "1",
        environmentDigest: gepaDigest,
        configDigest: "",
        seed: 7,
      },
      proposer: "deterministic",
      budget: {
        maxWallTimeMs: 120_000,
        maxModelCalls: 0,
        maxTokens: 0,
        maxArtifactBytes: 256 * 1024,
      },
    });
    request.optimizer.configDigest = canonicalSkillOptimizerConfig({
      optimizerId: "dspy.gepa.v1",
      seed: 7,
      proposer: "deterministic",
      budget: request.budget,
    });
    const workerDir = path.join(fixture.root, "zero-budget-worker");
    await mkdir(workerDir, { recursive: true });
    const launch: TaskWorkerLaunchSpec = {
      backendId: "python",
      program: EVOLUTION_PYTHON,
      args: ["-I", "-B", "-u", WORKER_PATH, "--environment-digest", gepaDigest],
      cwd: workerDir,
      env: {
        PATH: process.env.PATH ?? "",
        PYTHONDONTWRITEBYTECODE: "1",
        PYTHONUNBUFFERED: "1",
        PYTHONUTF8: "1",
        SWARMX_EVOLUTION_PATH: path.dirname(WORKER_PATH),
      },
      environmentDigest: gepaDigest,
      artifactRoot: workerDir,
    };
    const dspyAvailable = await import("node:child_process").then(({ execFileSync }) => {
      try {
        execFileSync(EVOLUTION_PYTHON, ["-c", "import dspy"], {
          stdio: "ignore",
          env: { ...process.env, PYTHONPATH: "" },
        });
        return true;
      } catch {
        return false;
      }
    });
    if (!dspyAvailable) {
      return; // Requires the locked evolution dependency group; skipped honestly.
    }
    const { workItem, grant } = fixture.service.createOptimizationWorkItem({
      request,
      launch,
    });
    await fixture.controlService.runWorkItem(workItem.id, { launch, grants: [grant] });
    expect(fixture.controlService.store.state().workItems[workItem.id].status).toBe("failed");
    const run =
      fixture.controlService.store.state().runs[
        fixture.controlService.store.state().workItems[workItem.id].activeRunId ?? ""
      ];
    expect(run?.failure?.message).toContain("positive maxModelCalls budget");
  }, 120_000);

  it("refuses to quarantine the active revision", async () => {
    const fixture = await createEvolutionFixture();
    const request = optimizationRequest(fixture);
    const { workItem, grant } = fixture.service.createOptimizationWorkItem({
      request,
      launch: fixture.launch,
    });
    await fixture.controlService.runWorkItem(workItem.id, {
      launch: fixture.launch,
      grants: [grant],
    });
    const candidate = fixture.service.ingestCandidate({ workItemId: workItem.id });
    await fixture.service.evaluateCandidate({
      candidateId: candidate.candidateId,
      holdoutContent: holdout("parrot"),
      createSwarm: createEvalSwarm,
      evaluatorId: "e2e",
      scorerFingerprint: "e2e",
      runtimeFingerprint: "e2e",
      seed: 21,
      gate,
    });
    fixture.service.promote({ candidateId: candidate.candidateId, actor: "e2e", reason: "ok" });
    expect(() =>
      fixture.service.decideCandidate({
        candidateId: candidate.candidateId,
        decision: "quarantine",
        actor: "e2e",
        reason: "should refuse",
      }),
    ).toThrow(/roll back before rejecting or quarantining/i);
    expect(fixture.service.state().candidates[candidate.candidateId]?.status).toBe("staged");
  }, 30_000);

  it("rejects external evidence that is not bound to the holdout", async () => {
    const fixture = await createEvolutionFixture();
    const request = optimizationRequest(fixture);
    const { workItem, grant } = fixture.service.createOptimizationWorkItem({
      request,
      launch: fixture.launch,
    });
    await fixture.controlService.runWorkItem(workItem.id, {
      launch: fixture.launch,
      grants: [grant],
    });
    const candidate = fixture.service.ingestCandidate({ workItemId: workItem.id });
    const sample = {
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
        safetyPassed: true,
        contextTokens: 100,
        latencyMs: 10,
        failed: false,
      },
      candidateRanFirst: false,
    };
    const holdoutContent = [1, 2, 3, 4]
      .map((index) => JSON.stringify({ caseId: `h${index}`, input: `q${index}`, target: "parrot" }))
      .join("\n");
    const base = {
      candidateId: candidate.candidateId,
      evaluatorId: "inspect.skill_paired_eval",
      scorerFingerprint: "test.v1",
      runtimeFingerprint: "test.runtime",
      seed: 1,
      holdoutContentDigest: `sha256:${createHash("sha256").update(holdoutContent).digest("hex")}`,
      holdoutCaseCount: 4,
      baselineRevisionId: request.parentRevisionId,
      candidateRevisionId: candidate.revisionId,
      targetAgentId: request.targetAgentId,
      targetModelFingerprint: request.targetModelFingerprint,
      samples: [
        sample,
        { ...sample, caseId: "h2" },
        { ...sample, caseId: "h3" },
        { ...sample, caseId: "h4" },
      ],
      holdoutContent,
      gate,
    };
    expect(() =>
      fixture.service.recordExternalEvaluation({ ...base, holdoutCaseCount: 999 }),
    ).toThrow(/reports 999 holdout cases/i);
    expect(() =>
      fixture.service.recordExternalEvaluation({
        ...base,
        samples: [sample, sample, { ...sample, caseId: "h2" }, { ...sample, caseId: "h2" }],
      }),
    ).toThrow(/unique case ids/i);
    expect(() =>
      fixture.service.recordExternalEvaluation({
        ...base,
        samples: [
          sample,
          { ...sample, caseId: "t1" },
          { ...sample, caseId: "h3" },
          { ...sample, caseId: "h4" },
        ],
      }),
    ).toThrow(/overlap the train\/dev splits/i);
    expect(() =>
      fixture.service.recordExternalEvaluation({
        ...base,
        holdoutContent: "different holdout content",
      }),
    ).toThrow(/holdout content digests to/i);
    expect(() =>
      fixture.service.recordExternalEvaluation({
        ...base,
        samples: [
          { ...sample, caseId: "x1" },
          { ...sample, caseId: "x2" },
          { ...sample, caseId: "x3" },
          { ...sample, caseId: "x4" },
        ],
      }),
    ).toThrow(/do not match the holdout case-id set/i);
    expect(() =>
      fixture.service.recordExternalEvaluation({
        ...base,
        targetAgentId: "someone-else",
      }),
    ).toThrow(/does not match the optimization target/i);
    expect(fixture.service.state().candidates[candidate.candidateId]?.status).toBe("proposed");
  }, 30_000);

  it("rejects a launch whose environment digest does not match the request", async () => {
    const fixture = await createEvolutionFixture();
    const request = optimizationRequest(fixture);
    const wrongLaunch: TaskWorkerLaunchSpec = {
      ...fixture.launch,
      environmentDigest: `sha256:${"7".repeat(64)}`,
    };
    expect(() =>
      fixture.service.createOptimizationWorkItem({
        request,
        launch: wrongLaunch,
      }),
    ).toThrow(/targets environment|ENVIRONMENT_DIGEST_MISMATCH/i);
  });

  it("atomically reserves tokens so concurrent calls cannot overrun the budget", async () => {
    const fixture = await createEvolutionFixture();
    const seenMaxTokens: Array<number | undefined> = [];
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const modelHandler = async (request: {
      maxTokens?: number;
    }): Promise<{
      content: string;
      usage: { totalTokens: number };
      latencyMs: number;
      costUsd: number;
    }> => {
      seenMaxTokens.push(request.maxTokens);
      await gate;
      return {
        content: "answer: ok",
        usage: { totalTokens: 3 },
        latencyMs: 1,
        costUsd: 0,
      };
    };
    const taskStore = fixture.controlService.store;
    const gateway = createSkillEvolutionCapabilityGateway({ taskStore, modelHandler });
    const request = optimizationRequest(fixture, {
      budget: {
        maxWallTimeMs: 120_000,
        maxModelCalls: 10,
        maxTokens: 5,
        maxArtifactBytes: 256 * 1024,
      },
    });
    const { workItem } = fixture.service.createOptimizationWorkItem({
      request,
      launch: fixture.launch,
    });
    const call = (callId: string) =>
      gateway.invoke({
        workItem,
        call: {
          protocolVersion: 1,
          messageId: `cap:${callId}`,
          direction: "worker_to_host",
          type: "capability_call",
          workItemId: workItem.id,
          runId: "run_reserve",
          leaseId: "lease_reserve",
          fencingToken: 1,
          sequence: 0,
          emittedAt: new Date().toISOString(),
          callId,
          grantId: "gnt_evolve_1",
          capabilityId: "skill_evolution",
          operation: "model.generate",
          idempotencyKey: `cap:${callId}`,
          arguments: { messages: [], maxTokens: 100 },
        },
      });
    const releaseTimer = setTimeout(release, 150);
    const [first, second] = await Promise.all([call("cap:1"), call("cap:2")]);
    clearTimeout(releaseTimer);
    const succeeded = [first, second].filter((outcome) => outcome.status === "succeeded").length;
    const denied = [first, second].filter(
      (outcome) => outcome.status === "failed" && outcome.error?.code === "token_budget_exhausted",
    ).length;
    expect(succeeded).toBe(1);
    expect(denied).toBe(1);
    expect(seenMaxTokens).toContain(5);
  }, 30_000);

  it("records a failed audit outcome when the promotion CAS is rejected", async () => {
    class CountingAuditStore extends AuditStore {
      appends = 0;
      override append(input: AuditInput | readonly AuditInput[]): void {
        this.appends += 1;
        super.append(input);
      }
    }
    const auditRoot = await temporaryRoot("swarmx-evolution-audit-");
    const audit = new CountingAuditStore({ filePath: path.join(auditRoot, "audit.jsonl") });
    const fixture = await createEvolutionFixture({ audit });
    const request = optimizationRequest(fixture);
    const { workItem, grant } = fixture.service.createOptimizationWorkItem({
      request,
      launch: fixture.launch,
    });
    await fixture.controlService.runWorkItem(workItem.id, {
      launch: fixture.launch,
      grants: [grant],
    });
    const candidate = fixture.service.ingestCandidate({ workItemId: workItem.id });
    await fixture.service.evaluateCandidate({
      candidateId: candidate.candidateId,
      holdoutContent: holdout("parrot"),
      createSwarm: createEvalSwarm,
      evaluatorId: "e2e",
      scorerFingerprint: "e2e",
      runtimeFingerprint: "e2e",
      seed: 23,
      gate,
    });
    fixture.service.promote({ candidateId: candidate.candidateId, actor: "e2e", reason: "first" });
    // The active pointer is now the candidate revision; promoting it again
    // violates the CAS expectation (active must equal the parent).
    expect(() =>
      fixture.service.promote({
        candidateId: candidate.candidateId,
        actor: "e2e",
        reason: "again",
      }),
    ).toThrow();
    const events = audit.query({ action: "skill.evolution.promote" });
    const outcomes = events.map((event) => event.outcome);
    expect(outcomes.filter((outcome) => outcome === "attempted").length).toBe(2);
    expect(outcomes.some((outcome) => outcome === "failed")).toBe(true);
    expect(fixture.service.state().activePointers["math-coach"]?.revisionId).toBe(
      candidate.revisionId,
    );
  }, 30_000);

  it("reports honestly when the terminal audit outcome cannot be written", async () => {
    class OutcomeThrowingAuditStore extends AuditStore {
      appends = 0;
      throwFrom = Number.POSITIVE_INFINITY;
      override append(input: AuditInput | readonly AuditInput[]): void {
        this.appends += 1;
        if (this.appends >= this.throwFrom) {
          throw new Error("audit authority unavailable at outcome time");
        }
        super.append(input);
      }
    }
    const auditRoot = await temporaryRoot("swarmx-evolution-audit-");
    const audit = new OutcomeThrowingAuditStore({ filePath: path.join(auditRoot, "audit.jsonl") });
    const fixture = await createEvolutionFixture({ audit });
    const request = optimizationRequest(fixture);
    const { workItem, grant } = fixture.service.createOptimizationWorkItem({
      request,
      launch: fixture.launch,
    });
    await fixture.controlService.runWorkItem(workItem.id, {
      launch: fixture.launch,
      grants: [grant],
    });
    const candidate = fixture.service.ingestCandidate({ workItemId: workItem.id });
    await fixture.service.evaluateCandidate({
      candidateId: candidate.candidateId,
      holdoutContent: holdout("parrot"),
      createSwarm: createEvalSwarm,
      evaluatorId: "e2e",
      scorerFingerprint: "e2e",
      runtimeFingerprint: "e2e",
      seed: 22,
      gate,
    });
    audit.throwFrom = audit.appends + 2;
    expect(() =>
      fixture.service.promote({ candidateId: candidate.candidateId, actor: "e2e", reason: "ok" }),
    ).toThrow(/AUDIT_OUTCOME_FAILED|audit outcome/);
    expect(fixture.service.state().activePointers["math-coach"]?.revisionId).toBe(
      candidate.revisionId,
    );
    const intents = audit.query({ action: "skill.evolution.promote" });
    expect(intents.some((event) => event.outcome === "attempted")).toBe(true);
  }, 30_000);
});
