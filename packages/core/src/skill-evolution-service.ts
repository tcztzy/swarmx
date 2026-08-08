import { createHash, randomUUID } from "node:crypto";
import { z } from "zod";
import type { AuditInput, AuditStore } from "./audit.js";
import {
  assertPromptFragmentDeliverable,
  SkillDeliveryError,
  type SkillInstructionDelivery,
} from "./skill-delivery.js";
import {
  aggregateSkillEvaluation,
  type PairedSkillEvaluationCase,
  type RunPairedSkillEvaluationOptions,
  runPairedSkillEvaluation,
} from "./skill-evaluation.js";
import {
  activePointerRevisionId,
  retainedSkillRevisions,
  type SkillCandidateStatus,
  type SkillEvolutionState,
  skillCandidateRevisionId,
  skillCandidateStaticChecksPassed,
} from "./skill-evolution.js";
import type { SkillEvolutionStore } from "./skill-evolution-store.js";
import {
  type SkillCandidateManifest,
  type SkillCandidateSecretScan,
  type SkillCandidateStaticChecks,
  type SkillEvaluationGate,
  type SkillEvaluationManifest,
  SkillEvaluationManifestSchema,
  type SkillEvaluationSample,
  SkillEvaluationSampleSchema,
  type SkillOptimizationRequest,
  SkillOptimizationRequestSchema,
  type SkillPromotionReceipt,
  SkillPromotionReceiptSchema,
} from "./skill-variants.js";
import type {
  AppAttachedTaskControlService,
  TaskCapabilityGateway,
  TaskCapabilityGatewayContext,
} from "./task-control-service.js";
import type { TaskRuntimeState, TaskWorkItem } from "./task-runtime.js";
import type { TaskRuntimeStore } from "./task-runtime-store.js";
import type { TaskWorkerCapabilityOutcome, TaskWorkerLaunchSpec } from "./task-worker-process.js";
import type { TaskWorkerCapabilityGrant } from "./task-worker-protocol.js";
import type { ModelTokenUsage, SwarmConfig } from "./types.js";
import { SwarmConfigSchema } from "./types.js";

export interface SkillEvolutionModelGenerateRequest {
  model?: string;
  messages: Array<{ role: string; content: string }>;
  temperature?: number;
  maxTokens?: number;
}

export interface SkillEvolutionModelGenerateResult {
  content: string;
  usage: ModelTokenUsage;
  latencyMs: number;
  costUsd?: number;
}

export type SkillEvolutionModelHandler = (
  request: SkillEvolutionModelGenerateRequest,
) => Promise<SkillEvolutionModelGenerateResult>;

export interface SkillEvolutionServiceOptions {
  ledger: SkillEvolutionStore;
  /** Attach for optimization/evaluation/delivery operations; absent for ledger-only commands. */
  controlService?: AppAttachedTaskControlService;
  audit: AuditStore;
  now?: () => Date;
  modelHandler?: SkillEvolutionModelHandler;
  capabilityTimeoutMs?: number;
}

export interface CreateOptimizationWorkItemInput {
  request: unknown;
  launch: TaskWorkerLaunchSpec;
  requestedBy?: string;
}

export interface CreateOptimizationWorkItemResult {
  workItem: TaskWorkItem;
  grant: TaskWorkerCapabilityGrant;
}

export interface IngestCandidateInput {
  workItemId: string;
}

export interface EvaluateCandidateInput {
  candidateId: string;
  holdoutContent: string;
  createSwarm: RunPairedSkillEvaluationOptions["createSwarm"];
  evaluatorId: string;
  scorerFingerprint: string;
  runtimeFingerprint: string;
  seed: number;
  gate: SkillEvaluationGate;
  estimateCostUsd?: (usage: ModelTokenUsage) => number;
}

export const SkillExternalEvaluationEvidenceSchema = z
  .object({
    evaluatorId: z.string().min(1).max(256),
    scorerFingerprint: z.string().min(1).max(512),
    runtimeFingerprint: z.string().min(1).max(512),
    seed: z.number().int().nonnegative(),
    holdoutContentDigest: z.string().regex(/^sha256:[a-f0-9]{64}$/),
    holdoutCaseCount: z.number().int().positive(),
    baselineRevisionId: z.string().min(1).max(512),
    candidateRevisionId: z.string().min(1).max(512),
    targetAgentId: z.string().min(1).max(256),
    targetModelFingerprint: z.string().min(1).max(512),
    samples: z.array(SkillEvaluationSampleSchema).min(1),
  })
  .strict();

export interface RecordExternalEvaluationInput {
  candidateId: string;
  evaluatorId: string;
  scorerFingerprint: string;
  runtimeFingerprint: string;
  seed: number;
  holdoutContentDigest: string;
  holdoutCaseCount: number;
  baselineRevisionId: string;
  candidateRevisionId: string;
  targetAgentId: string;
  targetModelFingerprint: string;
  samples: SkillEvaluationSample[];
  gate: SkillEvaluationGate;
  /** The actual holdout artifact content; its digest and case-id set are verified. */
  holdoutContent: string;
  /**
   * Host-computed config digest in the form `swarmx.inspect.config:<sha256>`.
   * When provided, the evidence runtimeFingerprint must equal it so a
   * self-reported fingerprint cannot be accepted.
   */
  hostConfigFingerprint?: string;
}

export interface PromoteSkillInput {
  candidateId: string;
  actor: string;
  reason: string;
  gate?: "human" | "policy";
}

export interface RollbackSkillInput {
  skillId: string;
  targetRevisionId: string;
  actor: string;
  reason: string;
  casExpectedRevisionId?: string;
}

export interface DecideCandidateInput {
  candidateId: string;
  decision: "reject" | "quarantine";
  actor: string;
  reason: string;
}

export interface ResolveActiveSkillDeliveryInput {
  skillId: string;
  variantId: string;
  harnessId?: string;
  modelControl: "direct" | "session" | "unsupported";
  baselineContentRef?: string;
  baselineContentDigest?: string;
  baselineRevisionId?: string;
}

export class SkillEvolutionServiceError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "SkillEvolutionServiceError";
    this.code = code;
  }
}

/**
 * Restricts evaluation swarms to direct native agents without MCP servers,
 * hooks, queen agents, or tool nodes, so a paired evaluation cannot reach
 * real external side effects. Tool/ACP behavior is out of scope for the
 * deterministic skill gate.
 */
export function assertEvalSafeSwarmConfig(config: unknown): SwarmConfig {
  const parsed = SwarmConfigSchema.parse(config);
  if (parsed.queen) {
    throw new SkillEvolutionServiceError(
      "EVAL_CONFIG_UNSAFE",
      "Evaluation swarms must not define a queen agent.",
    );
  }
  if (parsed.mcpServers && Object.keys(parsed.mcpServers).length > 0) {
    throw new SkillEvolutionServiceError(
      "EVAL_CONFIG_UNSAFE",
      "Evaluation swarms must not configure MCP servers.",
    );
  }
  if (parsed.hooks && parsed.hooks.length > 0) {
    throw new SkillEvolutionServiceError(
      "EVAL_CONFIG_UNSAFE",
      "Evaluation swarms must not configure hooks.",
    );
  }
  for (const [name, node] of Object.entries(parsed.nodes)) {
    if (node.kind !== "agent") {
      throw new SkillEvolutionServiceError(
        "EVAL_CONFIG_UNSAFE",
        `Evaluation swarms may contain only agent nodes; "${name}" is "${node.kind}".`,
      );
    }
    const backend = node.agent.backend?.type ?? "swarmx";
    if (backend !== "swarmx" && backend !== "echo") {
      throw new SkillEvolutionServiceError(
        "EVAL_CONFIG_UNSAFE",
        `Evaluation agent "${name}" uses backend "${backend}"; only direct native execution is supported.`,
      );
    }
    if (node.agent.mcpServers && Object.keys(node.agent.mcpServers).length > 0) {
      throw new SkillEvolutionServiceError(
        "EVAL_CONFIG_UNSAFE",
        `Evaluation agent "${name}" must not configure MCP servers.`,
      );
    }
  }
  return parsed;
}

export class SkillEvolutionPolicyGateClosedError extends SkillEvolutionServiceError {
  constructor() {
    super(
      "POLICY_GATE_CLOSED",
      "Policy promotion is fail-closed until canary and drift monitoring exist; only the human gate is available.",
    );
  }
}

/**
 * Owns the skill self-improvement loop: optimization WorkItems through the
 * durable task runtime, immutable candidate ingestion, paired evaluation,
 * static and evaluation gates, human promotion with compare-and-swap, and
 * rollback. DSPy only proposes; evaluation provides evidence; this service
 * decides.
 */
export class SkillEvolutionService {
  readonly ledger: SkillEvolutionStore;
  readonly controlService?: AppAttachedTaskControlService;
  readonly audit: AuditStore;
  private readonly now: () => Date;
  private readonly modelHandler?: SkillEvolutionModelHandler;
  private readonly capabilityTimeoutMs: number;

  constructor(options: SkillEvolutionServiceOptions) {
    this.ledger = options.ledger;
    this.controlService = options.controlService;
    this.audit = options.audit;
    this.now = options.now ?? (() => new Date());
    this.modelHandler = options.modelHandler;
    this.capabilityTimeoutMs = options.capabilityTimeoutMs ?? 60_000;
  }

  state(): SkillEvolutionState {
    return this.ledger.state();
  }

  private requireControlService(): AppAttachedTaskControlService {
    if (!this.controlService) {
      throw new SkillEvolutionServiceError(
        "CONTROL_SERVICE_UNAVAILABLE",
        "This operation requires the durable task runtime control service.",
      );
    }
    return this.controlService;
  }

  createOptimizationWorkItem(
    input: CreateOptimizationWorkItemInput,
  ): CreateOptimizationWorkItemResult {
    const request = SkillOptimizationRequestSchema.parse(input.request);
    if (request.optimizer.environmentDigest !== input.launch.environmentDigest) {
      throw new SkillEvolutionServiceError(
        "ENVIRONMENT_DIGEST_MISMATCH",
        `The optimization request targets environment ${request.optimizer.environmentDigest}, but the launch uses ${input.launch.environmentDigest}.`,
      );
    }
    const taskStore = this.requireControlService().store;
    for (const ref of [
      request.baselineContentRef,
      request.trainDataset.contentRef,
      request.devDataset.contentRef,
    ]) {
      taskStore.readBytes(ref);
    }
    const trainIds = this.datasetCaseIds(request.trainDataset.contentRef);
    const devIds = this.datasetCaseIds(request.devDataset.contentRef);
    const overlap = [...trainIds].filter((caseId) => devIds.has(caseId));
    if (overlap.length > 0) {
      throw new SkillEvolutionServiceError(
        "DATASET_OVERLAP",
        `Train and dev datasets share case ids: ${overlap.slice(0, 8).join(", ")}.`,
      );
    }
    const timestamp = this.timestamp();
    const requestId = `svr_${randomUUID().replaceAll("-", "")}`;
    this.appendLedger({
      kind: "optimization_requested",
      timestamp,
      idempotencyKey: `evolution:request:${requestId}`,
      payload: {
        requestId,
        request,
        requestedBy: input.requestedBy ?? "unknown",
      },
    });
    const workItem = this.requireControlService().createWorkItem({
      backend: "python",
      operation: "swarmx.evolve_skill",
      input: { ...request, requestId },
      priority: 10,
      owner: input.requestedBy ?? "skill-evolution",
      budget: {
        wallTimeMs: request.budget.maxWallTimeMs,
        maxArtifactBytes: request.budget.maxArtifactBytes,
        capabilityCalls: {
          "skill_evolution:read_artifact": 64,
          "skill_evolution:model.generate": request.budget.maxModelCalls ?? 1_000,
        },
      },
      maxAttempts: 1,
    });
    const grant: TaskWorkerCapabilityGrant = {
      grantId: `gnt_evolve_${workItem.id.slice(4)}`,
      capabilityId: "skill_evolution",
      operations: ["read_artifact", "model.generate"],
    };
    return { workItem, grant };
  }

  ingestCandidate(input: IngestCandidateInput): SkillCandidateManifest {
    const taskStore = this.requireControlService().store;
    const state = taskStore.state();
    const workItem = state.workItems[input.workItemId];
    if (!workItem) {
      throw new SkillEvolutionServiceError(
        "UNKNOWN_WORK_ITEM",
        `Work item "${input.workItemId}" does not exist.`,
      );
    }
    if (workItem.status !== "succeeded") {
      throw new SkillEvolutionServiceError(
        "WORK_ITEM_NOT_SUCCEEDED",
        `Work item "${input.workItemId}" is ${workItem.status}; expected succeeded.`,
      );
    }
    const rawInput = taskStore.readJson(workItem.inputRef as string) as Record<string, unknown>;
    const { requestId: _requestId, ...requestInput } = rawInput;
    const request = SkillOptimizationRequestSchema.parse(requestInput);
    const candidateArtifacts = candidateArtifactsFor(state, workItem);
    if (candidateArtifacts.length === 0) {
      throw new SkillEvolutionServiceError(
        "CANDIDATE_ARTIFACT_MISSING",
        `Work item "${input.workItemId}" produced no skill_candidate artifact.`,
      );
    }
    const artifact = candidateArtifacts[0];
    const content = taskStore.readBytes(artifactRefFor(artifact.uri));
    const contentDigest = `sha256:${sha256(content)}`;
    if (artifact.sha256 && contentDigest !== `sha256:${artifact.sha256}`) {
      throw new SkillEvolutionServiceError(
        "CANDIDATE_DIGEST_MISMATCH",
        `Candidate artifact digest does not match its receipt: ${artifact.artifactId}`,
      );
    }
    const revisionId = skillCandidateRevisionId(contentDigest);
    const workerManifest = this.workerCandidateManifest(workItem);
    const checks = this.computeStaticChecks(
      content,
      contentDigest,
      request,
      request.budget.maxArtifactBytes ?? 256 * 1024,
      workerManifest,
    );
    const secretScan = scanForSecrets(content.toString("utf8"));
    const checksWithSecretScan = { ...checks, secretScan };
    const passed = skillCandidateStaticChecksPassed(checksWithSecretScan);
    const status: SkillCandidateStatus = passed
      ? "proposed"
      : secretScan.passed
        ? "rejected"
        : "quarantined";
    void status;
    const manifest: SkillCandidateManifest = {
      schemaVersion: 1,
      candidateId: `skc_${randomUUID().replaceAll("-", "").slice(0, 16)}`,
      skillId: request.skillId,
      variantId: request.variantId,
      revisionId,
      parentRevisionId: request.parentRevisionId,
      parentRevisionDigest: request.baselineContentDigest,
      contentRef: artifactRefFor(artifact.uri),
      contentDigest,
      contentSizeBytes: content.byteLength,
      mediaType: artifact.mediaType ?? "text/markdown",
      targetAgentId: request.targetAgentId,
      targetModelFingerprint: request.targetModelFingerprint,
      optimizer: request.optimizer,
      trainDatasetDigest: request.trainDataset.contentDigest,
      devDatasetDigest: request.devDataset.contentDigest,
      staticChecks: checksWithSecretScan,
      createdAt: this.timestamp(),
      status: "proposed",
    };
    this.appendLedger({
      kind: "candidate_created",
      timestamp: this.timestamp(),
      idempotencyKey: `evolution:candidate:${manifest.candidateId}`,
      payload: { manifest, workItemId: workItem.id },
    });
    if (status !== "proposed") {
      this.appendLedger({
        kind: "candidate_status_changed",
        timestamp: this.timestamp(),
        idempotencyKey: `evolution:candidate:status:${manifest.candidateId}:${status}`,
        payload: {
          candidateId: manifest.candidateId,
          from: "proposed",
          to: status,
          reason:
            status === "quarantined"
              ? "Candidate content failed the secret scan."
              : "Candidate failed static checks.",
        },
      });
    }
    return manifest;
  }

  async evaluateCandidate(input: EvaluateCandidateInput): Promise<SkillEvaluationManifest> {
    const candidate = this.requireCandidate(input.candidateId);
    if (candidate.status !== "proposed" && candidate.status !== "evaluating") {
      throw new SkillEvolutionServiceError(
        "CANDIDATE_NOT_EVALUATABLE",
        `Candidate "${input.candidateId}" is ${candidate.status}; expected proposed.`,
      );
    }
    if (!skillCandidateStaticChecksPassed(candidate.manifest.staticChecks)) {
      throw new SkillEvolutionServiceError(
        "CANDIDATE_STATIC_FAILURE",
        `Candidate "${input.candidateId}" did not pass static checks.`,
      );
    }
    const manifest = candidate.manifest;
    const request = this.requestForCandidate(manifest);
    const evaluationId = `ske_${randomUUID().replaceAll("-", "").slice(0, 16)}`;
    this.appendLedger({
      kind: "candidate_status_changed",
      timestamp: this.timestamp(),
      idempotencyKey: `evolution:candidate:status:${input.candidateId}:evaluating:${evaluationId}`,
      payload: {
        candidateId: input.candidateId,
        from: candidate.status,
        to: "evaluating",
        reason: `Evaluation ${evaluationId} started.`,
      },
    });
    const baselineDelivery = this.deliveryForRevision(
      manifest.skillId,
      manifest.variantId,
      request.parentRevisionId,
      request.baselineContentRef,
      request.baselineContentDigest,
    );
    const candidateDelivery = this.deliveryForRevision(
      manifest.skillId,
      manifest.variantId,
      manifest.revisionId,
      manifest.contentRef,
      manifest.contentDigest,
    );
    const cases = this.parseHoldoutCases(input.holdoutContent, request);
    const holdoutDigest = `sha256:${sha256(input.holdoutContent)}`;
    const { samples, manifest: aggregated } = await runPairedSkillEvaluation({
      ...input,
      evaluationId,
      candidateId: input.candidateId,
      holdoutContentDigest: holdoutDigest,
      holdoutCaseCount: cases.length,
      cases,
      baselineDelivery,
      candidateDelivery,
    });
    return this.finalizeEvaluation({
      candidateId: input.candidateId,
      evaluationId,
      baselineDelivery,
      candidateDelivery,
      samples,
      holdoutDigest,
      holdoutCaseCount: cases.length,
      evaluatorId: input.evaluatorId,
      scorerFingerprint: input.scorerFingerprint,
      runtimeFingerprint: input.runtimeFingerprint,
      seed: input.seed,
      gate: input.gate,
      aggregated,
    });
  }

  /**
   * Records evaluation evidence produced by an independent evaluator (for
   * example the Inspect paired adapter) without re-running the executions.
   * The evidence must name the same baseline and candidate revisions and the
   * hidden holdout digest; the gate verdict is computed here in Core.
   */
  recordExternalEvaluation(input: RecordExternalEvaluationInput): SkillEvaluationManifest {
    const validated = SkillExternalEvaluationEvidenceSchema.parse({
      evaluatorId: input.evaluatorId,
      scorerFingerprint: input.scorerFingerprint,
      runtimeFingerprint: input.runtimeFingerprint,
      seed: input.seed,
      holdoutContentDigest: input.holdoutContentDigest,
      holdoutCaseCount: input.holdoutCaseCount,
      baselineRevisionId: input.baselineRevisionId,
      candidateRevisionId: input.candidateRevisionId,
      targetAgentId: input.targetAgentId,
      targetModelFingerprint: input.targetModelFingerprint,
      samples: input.samples,
    });
    if (validated.samples.length < input.gate.minSampleCount) {
      throw new SkillEvolutionServiceError(
        "EVIDENCE_INSUFFICIENT",
        `Evidence has ${validated.samples.length} samples; the gate requires at least ${input.gate.minSampleCount}.`,
      );
    }
    if (input.hostConfigFingerprint !== undefined) {
      if (validated.runtimeFingerprint !== input.hostConfigFingerprint) {
        throw new SkillEvolutionServiceError(
          "EVIDENCE_CONFIG_FINGERPRINT_MISMATCH",
          `Evidence runtimeFingerprint "${validated.runtimeFingerprint}" does not match the host-computed config fingerprint "${input.hostConfigFingerprint}".`,
        );
      }
    }
    if (validated.samples.length !== validated.holdoutCaseCount) {
      throw new SkillEvolutionServiceError(
        "EVIDENCE_COUNT_MISMATCH",
        `Evidence reports ${validated.holdoutCaseCount} holdout cases but contains ${validated.samples.length} samples.`,
      );
    }
    const caseIds = validated.samples.map((sample) => sample.caseId);
    if (new Set(caseIds).size !== caseIds.length) {
      throw new SkillEvolutionServiceError(
        "EVIDENCE_DUPLICATE_CASE_IDS",
        "Evidence samples must have unique case ids.",
      );
    }
    const candidate = this.requireCandidate(input.candidateId);
    if (candidate.status !== "proposed" && candidate.status !== "evaluating") {
      throw new SkillEvolutionServiceError(
        "CANDIDATE_NOT_EVALUATABLE",
        `Candidate "${input.candidateId}" is ${candidate.status}; expected proposed.`,
      );
    }
    if (!skillCandidateStaticChecksPassed(candidate.manifest.staticChecks)) {
      throw new SkillEvolutionServiceError(
        "CANDIDATE_STATIC_FAILURE",
        `Candidate "${input.candidateId}" did not pass static checks.`,
      );
    }
    const manifest = candidate.manifest;
    const request = this.requestForCandidate(manifest);
    if (validated.baselineRevisionId !== request.parentRevisionId) {
      throw new SkillEvolutionServiceError(
        "EVIDENCE_REVISION_MISMATCH",
        `Evidence baseline ${validated.baselineRevisionId} does not match ${request.parentRevisionId}.`,
      );
    }
    if (validated.candidateRevisionId !== manifest.revisionId) {
      throw new SkillEvolutionServiceError(
        "EVIDENCE_REVISION_MISMATCH",
        `Evidence candidate ${validated.candidateRevisionId} does not match ${manifest.revisionId}.`,
      );
    }
    if (validated.targetAgentId !== request.targetAgentId) {
      throw new SkillEvolutionServiceError(
        "EVIDENCE_TARGET_MISMATCH",
        `Evidence target agent "${validated.targetAgentId}" does not match the optimization target "${request.targetAgentId}".`,
      );
    }
    if (validated.targetModelFingerprint !== request.targetModelFingerprint) {
      throw new SkillEvolutionServiceError(
        "EVIDENCE_MODEL_MISMATCH",
        `Evidence target model fingerprint "${validated.targetModelFingerprint}" does not match the optimization target "${request.targetModelFingerprint}".`,
      );
    }
    const overlap = caseIds.filter(
      (caseId) =>
        this.datasetCaseIds(request.trainDataset.contentRef).has(caseId) ||
        this.datasetCaseIds(request.devDataset.contentRef).has(caseId),
    );
    if (overlap.length > 0) {
      throw new SkillEvolutionServiceError(
        "EVIDENCE_HOLDOUT_OVERLAP",
        `Evidence cases overlap the train/dev splits: ${overlap.slice(0, 8).join(", ")}.`,
      );
    }
    if (input.holdoutContent !== undefined) {
      const actualDigest = `sha256:${sha256(input.holdoutContent)}`;
      if (actualDigest !== validated.holdoutContentDigest) {
        throw new SkillEvolutionServiceError(
          "EVIDENCE_HOLDOUT_DIGEST_MISMATCH",
          `The supplied holdout content digests to ${actualDigest}, not the claimed ${validated.holdoutContentDigest}.`,
        );
      }
    }
    // The evidence case-id set must be exactly the actual holdout case-id set.
    const holdoutCases = parseEvalCases(input.holdoutContent);
    const holdoutIds = new Set(holdoutCases.map((caseItem) => caseItem.caseId));
    if (holdoutIds.size !== holdoutCases.length) {
      throw new SkillEvolutionServiceError(
        "HOLDOUT_DUPLICATE_CASE_IDS",
        "The holdout content contains duplicate case ids.",
      );
    }
    if (holdoutCases.length !== validated.holdoutCaseCount) {
      throw new SkillEvolutionServiceError(
        "HOLDOUT_COUNT_MISMATCH",
        `The holdout content has ${holdoutCases.length} cases; evidence claims ${validated.holdoutCaseCount}.`,
      );
    }
    const evidenceIds = new Set(caseIds);
    if (
      evidenceIds.size !== holdoutIds.size ||
      ![...evidenceIds].every((id) => holdoutIds.has(id))
    ) {
      throw new SkillEvolutionServiceError(
        "EVIDENCE_HOLDOUT_SET_MISMATCH",
        "Evidence case ids do not match the holdout case-id set exactly.",
      );
    }
    const evaluationId = `ske_${randomUUID().replaceAll("-", "").slice(0, 16)}`;
    this.appendLedger({
      kind: "candidate_status_changed",
      timestamp: this.timestamp(),
      idempotencyKey: `evolution:candidate:status:${input.candidateId}:evaluating:${evaluationId}`,
      payload: {
        candidateId: input.candidateId,
        from: candidate.status,
        to: "evaluating",
        reason: `External evaluation ${evaluationId} started.`,
      },
    });
    const baselineDelivery = this.deliveryForRevision(
      manifest.skillId,
      manifest.variantId,
      request.parentRevisionId,
      request.baselineContentRef,
      request.baselineContentDigest,
    );
    const candidateDelivery = this.deliveryForRevision(
      manifest.skillId,
      manifest.variantId,
      manifest.revisionId,
      manifest.contentRef,
      manifest.contentDigest,
    );
    const aggregated = aggregateSkillEvaluation({
      evaluationId,
      candidateId: input.candidateId,
      holdoutContentDigest: validated.holdoutContentDigest,
      holdoutCaseCount: validated.holdoutCaseCount,
      baselineDelivery,
      candidateDelivery,
      cases: [],
      evaluatorId: validated.evaluatorId,
      scorerFingerprint: validated.scorerFingerprint,
      runtimeFingerprint: validated.runtimeFingerprint,
      seed: validated.seed,
      gate: input.gate,
      samples: validated.samples,
    });
    return this.finalizeEvaluation({
      candidateId: input.candidateId,
      evaluationId,
      baselineDelivery,
      candidateDelivery,
      samples: validated.samples,
      holdoutDigest: validated.holdoutContentDigest,
      holdoutCaseCount: validated.holdoutCaseCount,
      evaluatorId: validated.evaluatorId,
      scorerFingerprint: validated.scorerFingerprint,
      runtimeFingerprint: validated.runtimeFingerprint,
      seed: validated.seed,
      gate: input.gate,
      aggregated,
    });
  }

  private finalizeEvaluation(input: {
    candidateId: string;
    evaluationId: string;
    baselineDelivery: SkillInstructionDelivery;
    candidateDelivery: SkillInstructionDelivery;
    samples: SkillEvaluationSample[];
    holdoutDigest: string;
    holdoutCaseCount: number;
    evaluatorId: string;
    scorerFingerprint: string;
    runtimeFingerprint: string;
    seed: number;
    gate: SkillEvaluationGate;
    aggregated: SkillEvaluationManifest;
  }): SkillEvaluationManifest {
    const samplesBlob = this.ledger.putJson(input.samples);
    const finalManifest = SkillEvaluationManifestSchema.parse({
      ...input.aggregated,
      samplesRef: samplesBlob.ref,
      holdoutContentRef: input.holdoutDigest,
    });
    const targetStatus: SkillCandidateStatus =
      finalManifest.verdict === "eligible" ? "staged" : "rejected";
    this.appendLedger([
      {
        kind: "evaluation_recorded",
        timestamp: this.timestamp(),
        idempotencyKey: `evolution:evaluation:${input.evaluationId}`,
        payload: { manifest: finalManifest, evaluationId: input.evaluationId },
      },
      {
        kind: "candidate_status_changed",
        timestamp: this.timestamp(),
        idempotencyKey: `evolution:candidate:status:${input.candidateId}:${targetStatus}`,
        payload: {
          candidateId: input.candidateId,
          from: "evaluating",
          to: targetStatus,
          reason:
            finalManifest.verdict === "eligible"
              ? `Evaluation ${input.evaluationId} is eligible.`
              : `Evaluation ${input.evaluationId} rejected: ${finalManifest.reasons.join("; ")}`,
          evaluationRunId: input.evaluationId,
        },
      },
    ]);
    return finalManifest;
  }

  promote(input: PromoteSkillInput): SkillPromotionReceipt {
    const candidate = this.requireCandidate(input.candidateId);
    const gate = input.gate ?? "human";
    if (gate === "policy") {
      throw new SkillEvolutionPolicyGateClosedError();
    }
    if (candidate.status !== "staged") {
      throw new SkillEvolutionServiceError(
        "CANDIDATE_NOT_STAGED",
        `Candidate "${input.candidateId}" is ${candidate.status}; expected staged.`,
      );
    }
    const manifest = candidate.manifest;
    const evaluation = this.requireEligibleEvaluation(manifest);
    const currentRevision = activePointerRevisionId(this.state(), manifest.skillId);
    const casExpectedRevisionId = manifest.parentRevisionId;
    const requestId = `svp_${randomUUID().replaceAll("-", "")}`;
    this.auditIntent("skill.evolution.promote", {
      requestId,
      actor: input.actor,
      metadata: {
        skillId: manifest.skillId,
        candidateId: manifest.candidateId,
        candidateRevisionId: manifest.revisionId,
        parentRevisionId: manifest.parentRevisionId,
        evaluationRunId: evaluation.evaluationId,
        casExpectedRevisionId,
        gate,
      },
      target: { kind: "skill", id: manifest.skillId },
    });
    const receipt = SkillPromotionReceiptSchema.parse({
      schemaVersion: 1,
      receiptId: `skp_${randomUUID().replaceAll("-", "").slice(0, 16)}`,
      skillId: manifest.skillId,
      decision: "promote",
      gate,
      candidateId: manifest.candidateId,
      candidateRevisionId: manifest.revisionId,
      parentRevisionId: manifest.parentRevisionId,
      evaluationRunId: evaluation.evaluationId,
      casExpectedRevisionId,
      previousRevisionId: currentRevision,
      newRevisionId: manifest.revisionId,
      actor: input.actor,
      reason: input.reason,
      idempotencyKey: `evolution:promote:${manifest.candidateId}`,
      decidedAt: this.timestamp(),
    });
    try {
      this.appendLedger({
        kind: "promotion_recorded",
        timestamp: this.timestamp(),
        idempotencyKey: receipt.idempotencyKey,
        payload: {
          receipt,
          promotedRevision: {
            revisionId: manifest.revisionId,
            contentRef: manifest.contentRef,
            contentDigest: manifest.contentDigest,
          },
        },
      });
    } catch (error) {
      try {
        this.auditOutcome("skill.evolution.promote", "failed", requestId, {
          skillId: manifest.skillId,
          candidateId: manifest.candidateId,
          casExpectedRevisionId,
          errorType: error instanceof Error ? error.name : "Error",
        });
      } catch {
        // The audit authority is unavailable; the intent event is durable.
      }
      throw error;
    }
    this.auditOutcomeOrFailClosed("skill.evolution.promote", requestId, {
      skillId: manifest.skillId,
      candidateId: manifest.candidateId,
      previousRevisionId: currentRevision,
      newRevisionId: manifest.revisionId,
      receiptId: receipt.receiptId,
    });
    return receipt;
  }

  rollback(input: RollbackSkillInput): SkillPromotionReceipt {
    const currentRevision = activePointerRevisionId(this.state(), input.skillId);
    if (currentRevision === null) {
      throw new SkillEvolutionServiceError(
        "NO_ACTIVE_REVISION",
        `Skill "${input.skillId}" has no evolved active revision to roll back.`,
      );
    }
    if (input.targetRevisionId === currentRevision) {
      throw new SkillEvolutionServiceError(
        "ROLLBACK_TO_ACTIVE",
        `Skill "${input.skillId}" is already at revision "${input.targetRevisionId}".`,
      );
    }
    const retained = retainedSkillRevisions(this.state(), input.skillId);
    if (!retained[input.targetRevisionId]) {
      throw new SkillEvolutionServiceError(
        "UNKNOWN_ROLLBACK_TARGET",
        `Revision "${input.targetRevisionId}" is not a retained revision of Skill "${input.skillId}".`,
      );
    }
    const casExpectedRevisionId = input.casExpectedRevisionId ?? currentRevision;
    const requestId = `svr_${randomUUID().replaceAll("-", "")}`;
    this.auditIntent("skill.evolution.rollback", {
      requestId,
      actor: input.actor,
      metadata: {
        skillId: input.skillId,
        targetRevisionId: input.targetRevisionId,
        casExpectedRevisionId,
        gate: "human",
      },
      target: { kind: "skill", id: input.skillId },
    });
    const receipt = SkillPromotionReceiptSchema.parse({
      schemaVersion: 1,
      receiptId: `skp_${randomUUID().replaceAll("-", "").slice(0, 16)}`,
      skillId: input.skillId,
      decision: "rollback",
      gate: "human",
      casExpectedRevisionId,
      previousRevisionId: currentRevision,
      newRevisionId: input.targetRevisionId,
      actor: input.actor,
      reason: input.reason,
      idempotencyKey: `evolution:rollback:${input.skillId}:${input.targetRevisionId}`,
      decidedAt: this.timestamp(),
    });
    try {
      this.appendLedger({
        kind: "promotion_recorded",
        timestamp: this.timestamp(),
        idempotencyKey: receipt.idempotencyKey,
        payload: { receipt },
      });
    } catch (error) {
      try {
        this.auditOutcome("skill.evolution.rollback", "failed", requestId, {
          skillId: input.skillId,
          casExpectedRevisionId,
          errorType: error instanceof Error ? error.name : "Error",
        });
      } catch {
        // The audit authority is unavailable; the intent event is durable.
      }
      throw error;
    }
    this.auditOutcomeOrFailClosed("skill.evolution.rollback", requestId, {
      skillId: input.skillId,
      previousRevisionId: currentRevision,
      newRevisionId: input.targetRevisionId,
      receiptId: receipt.receiptId,
    });
    return receipt;
  }

  decideCandidate(input: DecideCandidateInput): void {
    const candidate = this.requireCandidate(input.candidateId);
    const target: SkillCandidateStatus = input.decision === "reject" ? "rejected" : "quarantined";
    const activeRevision = activePointerRevisionId(this.state(), candidate.manifest.skillId);
    if (activeRevision === candidate.manifest.revisionId) {
      throw new SkillEvolutionServiceError(
        "ACTIVE_REVISION_NOT_DECIDABLE",
        `Candidate "${input.candidateId}" is the active revision; roll back before rejecting or quarantining it.`,
      );
    }
    const requestId = `svc_${randomUUID().replaceAll("-", "")}`;
    this.auditIntent("skill.evolution.decide", {
      requestId,
      actor: input.actor,
      metadata: {
        candidateId: input.candidateId,
        decision: input.decision,
        from: candidate.status,
      },
    });
    this.appendLedger({
      kind: "candidate_status_changed",
      timestamp: this.timestamp(),
      idempotencyKey: `evolution:candidate:status:${input.candidateId}:${target}`,
      payload: {
        candidateId: input.candidateId,
        from: candidate.status,
        to: target,
        reason: input.reason,
      },
    });
    this.auditOutcome("skill.evolution.decide", "completed", requestId, {
      candidateId: input.candidateId,
      to: target,
    });
  }

  resolveActiveSkillDelivery(input: ResolveActiveSkillDeliveryInput): SkillInstructionDelivery {
    assertPromptFragmentDeliverable({
      deliveryMode: "prompt_fragment",
      harnessId: input.harnessId,
      modelControl: input.modelControl,
    });
    const pointer = this.state().activePointers[input.skillId];
    const contentRef = pointer?.contentRef ?? input.baselineContentRef;
    const contentDigest = pointer?.contentDigest ?? input.baselineContentDigest;
    const revisionId = pointer?.revisionId ?? input.baselineRevisionId;
    if (!contentRef || !contentDigest || !revisionId) {
      throw new SkillEvolutionServiceError(
        "NO_ACTIVE_DELIVERY",
        `Skill "${input.skillId}" has no active evolved revision and no baseline was supplied.`,
      );
    }
    return this.deliveryForRevision(
      input.skillId,
      input.variantId,
      revisionId,
      contentRef,
      contentDigest,
    );
  }

  createCapabilityGateway(): TaskCapabilityGateway {
    return createSkillEvolutionCapabilityGateway({
      taskStore: this.requireControlService().store,
      modelHandler: this.modelHandler,
      capabilityTimeoutMs: this.capabilityTimeoutMs,
    });
  }

  private datasetCaseIds(contentRef: string): Set<string> {
    const records = this.requireControlService().store.readJson(contentRef);
    return parseDatasetCaseIds(records);
  }

  private parseHoldoutCases(
    content: string,
    request: SkillOptimizationRequest,
  ): PairedSkillEvaluationCase[] {
    const trainIds = this.datasetCaseIds(request.trainDataset.contentRef);
    const devIds = this.datasetCaseIds(request.devDataset.contentRef);
    const cases = parseEvalCases(content);
    const overlap = cases.filter(
      (caseItem) => trainIds.has(caseItem.caseId) || devIds.has(caseItem.caseId),
    );
    if (overlap.length > 0) {
      throw new SkillEvolutionServiceError(
        "HOLDOUT_OVERLAP",
        `Holdout shares case ids with train/dev: ${overlap
          .slice(0, 8)
          .map((caseItem) => caseItem.caseId)
          .join(", ")}.`,
      );
    }
    return cases;
  }

  private workerCandidateManifest(workItem: TaskWorkItem): Record<string, unknown> | undefined {
    const state = this.requireControlService().store.state();
    const run = workItem.activeRunId ? state.runs[workItem.activeRunId] : undefined;
    if (!run?.resultRef) return undefined;
    const result = this.requireControlService().store.readJson(run.resultRef) as {
      candidateManifest?: unknown;
    };
    return typeof result.candidateManifest === "object" && result.candidateManifest !== null
      ? (result.candidateManifest as Record<string, unknown>)
      : undefined;
  }

  private computeStaticChecks(
    content: Buffer,
    contentDigest: string,
    request: SkillOptimizationRequest,
    maxArtifactBytes: number,
    workerManifest: Record<string, unknown> | undefined,
  ): SkillCandidateStaticChecks {
    if (workerManifest === undefined) {
      return {
        contentDigestVerified: true,
        parentRevisionDigestMatches: false,
        lineageMatchesRequest: false,
        instructionDeltaPresent:
          contentDigest !== request.baselineContentDigest && content.length > 0,
        sizeWithinBudget: content.byteLength <= maxArtifactBytes,
        deliverySupported: false,
        secretScan: { passed: true, findings: [] },
      };
    }
    const workerParent = workerManifest.parentRevisionId;
    const workerParentDigest = workerManifest.parentRevisionDigest;
    const workerOptimizer = workerManifest.optimizer as
      | { optimizerId?: unknown; optimizerVersion?: unknown; configDigest?: unknown }
      | undefined;
    const lineageMatchesRequest =
      workerManifest.skillId === request.skillId &&
      workerManifest.variantId === request.variantId &&
      workerParent === request.parentRevisionId &&
      workerParentDigest === request.baselineContentDigest &&
      workerOptimizer?.optimizerId === request.optimizer.optimizerId &&
      workerOptimizer?.optimizerVersion === request.optimizer.optimizerVersion &&
      workerOptimizer?.configDigest === request.optimizer.configDigest;
    return {
      contentDigestVerified: true,
      parentRevisionDigestMatches:
        workerManifest.parentRevisionDigest === request.baselineContentDigest,
      lineageMatchesRequest,
      instructionDeltaPresent:
        contentDigest !== request.baselineContentDigest && content.length > 0,
      sizeWithinBudget: content.byteLength <= maxArtifactBytes,
      deliverySupported: workerManifest.mediaType === "text/markdown",
      secretScan: { passed: true, findings: [] },
    };
  }

  private deliveryForRevision(
    skillId: string,
    variantId: string,
    revisionId: string,
    contentRef: string,
    contentDigest: string,
  ): SkillInstructionDelivery {
    const content = this.requireControlService().store.readBytes(contentRef).toString("utf8");
    if (`sha256:${sha256(content)}` !== contentDigest) {
      throw new SkillDeliveryError(
        "digest_mismatch",
        `Content digest mismatch for revision "${revisionId}".`,
      );
    }
    return {
      skillId,
      variantId,
      revisionId,
      contentDigest,
      mode: "prompt_fragment",
      content,
    };
  }

  private requestForCandidate(manifest: SkillCandidateManifest): SkillOptimizationRequest {
    for (const entry of Object.values(this.state().optimizationRequests)) {
      if (
        entry.request.skillId === manifest.skillId &&
        entry.request.parentRevisionId === manifest.parentRevisionId &&
        entry.request.trainDataset.contentDigest === manifest.trainDatasetDigest &&
        entry.request.devDataset.contentDigest === manifest.devDatasetDigest
      ) {
        return entry.request;
      }
    }
    throw new SkillEvolutionServiceError(
      "REQUEST_NOT_FOUND",
      `No optimization request matches candidate "${manifest.candidateId}".`,
    );
  }

  private requireCandidate(candidateId: string): {
    manifest: SkillCandidateManifest;
    status: SkillCandidateStatus;
  } {
    const candidate = this.state().candidates[candidateId];
    if (!candidate) {
      throw new SkillEvolutionServiceError(
        "UNKNOWN_CANDIDATE",
        `Skill candidate "${candidateId}" does not exist.`,
      );
    }
    return candidate;
  }

  private requireEligibleEvaluation(manifest: SkillCandidateManifest): SkillEvaluationManifest {
    const evaluations = Object.values(this.state().evaluations)
      .filter((evaluation) => evaluation.candidateId === manifest.candidateId)
      .sort((left, right) => left.completedAt.localeCompare(right.completedAt));
    const latest = evaluations.at(-1);
    if (!latest?.verdict || latest.verdict !== "eligible") {
      throw new SkillEvolutionServiceError(
        "NO_ELIGIBLE_EVALUATION",
        `Candidate "${manifest.candidateId}" has no eligible evaluation.`,
      );
    }
    return latest;
  }

  private appendLedger(input: unknown | readonly unknown[]): void {
    const records = (Array.isArray(input) ? input : [input]) as Array<{
      kind: string;
      timestamp: string;
      idempotencyKey: string;
      payload: Record<string, unknown>;
    }>;
    this.ledger.append(
      records.map((record) => ({
        schemaVersion: 1,
        recordId: `evl_${randomUUID().replaceAll("-", "")}`,
        kind: record.kind,
        timestamp: record.timestamp,
        idempotencyKey: record.idempotencyKey,
        payload: record.payload,
      })),
    );
  }

  private auditIntent(
    action: string,
    input: {
      requestId: string;
      actor: string;
      metadata: Record<string, string | null | number | boolean>;
      target?: { kind: string; id: string };
    },
  ): void {
    this.audit.append(
      auditInput(action, "attempted", {
        requestId: input.requestId,
        actor: input.actor,
        metadata: input.metadata,
        target: input.target,
      }),
    );
  }

  private auditOutcome(
    action: string,
    outcome: "completed" | "failed" | "denied",
    requestId: string,
    metadata: Record<string, string | null | number | boolean>,
  ): void {
    this.audit.append(
      auditInput(action, outcome, { requestId, actor: "skill-evolution", metadata }),
    );
  }

  /**
   * Terminal audit outcome for an effect that already happened. If the write
   * fails, one best-effort `failed` outcome is attempted and the failure is
   * surfaced honestly; the durable intent event remains for verification.
   */
  private auditOutcomeOrFailClosed(
    action: string,
    requestId: string,
    metadata: Record<string, string | null | number | boolean>,
  ): void {
    try {
      this.auditOutcome(action, "completed", requestId, metadata);
    } catch {
      try {
        this.auditOutcome(action, "failed", requestId, {
          ...metadata,
          note: "The completed outcome could not be recorded after the effect applied.",
        });
      } catch {
        // The audit authority is fully unavailable; the intent event is durable.
      }
      throw new SkillEvolutionServiceError(
        "AUDIT_OUTCOME_FAILED",
        `${action} applied, but its audit outcome could not be recorded; verify the ledger before continuing.`,
      );
    }
  }

  private timestamp(): string {
    return this.now().toISOString();
  }
}

export interface SkillEvolutionCapabilityGatewayOptions {
  taskStore: TaskRuntimeStore;
  modelHandler?: SkillEvolutionModelHandler;
  capabilityTimeoutMs?: number;
}

/**
 * Grant-checked capability gateway for the skill evolution worker. `read_artifact`
 * is authorized only for the content-addressed refs named in the WorkItem's
 * optimization request; `model.generate` resolves credentials inside the host
 * model handler and never passes them to the worker.
 */
export function createSkillEvolutionCapabilityGateway(
  options: SkillEvolutionCapabilityGatewayOptions,
): TaskCapabilityGateway {
  // In-process atomic reservations per work item. Capability calls are handled
  // sequentially by the single-threaded control loop, so checking and
  // reserving without an await is atomic within one host; committed durable
  // receipts cover the budget across restarts.
  const reservations = new Map<string, Map<string, number>>();
  return {
    invoke: async (context: TaskCapabilityGatewayContext): Promise<TaskWorkerCapabilityOutcome> => {
      const call = context.call;
      if (call.capabilityId !== "skill_evolution") {
        return failed("unsupported_capability", `Unsupported capability "${call.capabilityId}".`);
      }
      if (call.operation === "read_artifact") {
        return readArtifactCapability(options, context);
      }
      if (call.operation === "model.generate") {
        return generateCapability(options, context, reservations);
      }
      return failed("unsupported_operation", `Unsupported operation "${call.operation}".`);
    },
  };
}

function readArtifactCapability(
  options: SkillEvolutionCapabilityGatewayOptions,
  context: TaskCapabilityGatewayContext,
): TaskWorkerCapabilityOutcome {
  const ref = (context.call.arguments as { ref?: unknown })?.ref;
  if (typeof ref !== "string" || !/^sha256:[a-f0-9]{64}$/.test(ref)) {
    return failed("bad_arguments", "read_artifact requires a content-addressed ref.");
  }
  if (!allowedArtifactRefs(options, context.workItem).has(ref)) {
    return failed(
      "artifact_not_granted",
      "The requested artifact is not granted to this optimization work.",
    );
  }
  let bytes: Buffer;
  try {
    bytes = options.taskStore.readBytes(ref);
  } catch {
    return failed("artifact_unavailable", "The requested artifact is unavailable.");
  }
  return {
    status: "succeeded",
    value: {
      ref,
      contentType: "text/markdown",
      content: bytes.toString("utf8"),
      sizeBytes: bytes.byteLength,
    },
    artifactIds: [],
  };
}

async function generateCapability(
  options: SkillEvolutionCapabilityGatewayOptions,
  context: TaskCapabilityGatewayContext,
  reservations: Map<string, Map<string, number>>,
): Promise<TaskWorkerCapabilityOutcome> {
  if (!options.modelHandler) {
    return failed(
      "model_handler_unavailable",
      "No model handler is attached to this skill evolution service.",
    );
  }
  const arguments_ = context.call.arguments as {
    model?: unknown;
    messages?: unknown;
    temperature?: unknown;
    maxTokens?: unknown;
  };
  if (!Array.isArray(arguments_.messages)) {
    return failed("bad_arguments", "model.generate requires a messages array.");
  }
  const committed = remainingTokenBudget(options, context.workItem);
  if (committed === 0) {
    return failed(
      "token_budget_exhausted",
      "The granted model token budget is zero; model calls are denied before dispatch.",
    );
  }
  const reservationMap = workItemReservationMap(reservations, context.workItem.id);
  const reservedTotal = reservationTotal(reservationMap);
  const available = committed === undefined ? undefined : committed - reservedTotal;
  if (available !== undefined && available <= 0) {
    return failed(
      "token_budget_exhausted",
      "The granted model token budget is fully reserved by in-flight calls.",
    );
  }
  const requestedMaxTokens =
    typeof arguments_.maxTokens === "number" && arguments_.maxTokens > 0
      ? arguments_.maxTokens
      : undefined;
  // Always pass an explicit maxTokens when a budget exists, clamped to what
  // is available after durable usage and in-flight reservations.
  const grantedMaxTokens =
    available === undefined
      ? requestedMaxTokens
      : Math.min(requestedMaxTokens ?? available, available);
  if (grantedMaxTokens !== undefined) {
    reservationMap.set(context.call.callId, grantedMaxTokens);
  }
  const request = {
    model: typeof arguments_.model === "string" ? arguments_.model : undefined,
    messages: arguments_.messages as Array<{ role: string; content: string }>,
    temperature: typeof arguments_.temperature === "number" ? arguments_.temperature : undefined,
    maxTokens: grantedMaxTokens,
  };
  try {
    const result = await withTimeout(
      options.modelHandler(request),
      options.capabilityTimeoutMs ?? 60_000,
    );
    return {
      status: "succeeded",
      value: {
        content: result.content,
        usage: { totalTokens: result.usage.totalTokens },
        latencyMs: result.latencyMs,
        costUsd: result.costUsd ?? 0,
      },
      artifactIds: [],
    };
  } catch (error) {
    return failed("model_generation_failed", boundedErrorMessage(error));
  } finally {
    if (grantedMaxTokens !== undefined) {
      reservationMap.delete(context.call.callId);
      if (reservationMap.size === 0) reservations.delete(context.workItem.id);
    }
  }
}

function workItemReservationMap(
  reservations: Map<string, Map<string, number>>,
  workItemId: string,
): Map<string, number> {
  let map = reservations.get(workItemId);
  if (!map) {
    map = new Map<string, number>();
    reservations.set(workItemId, map);
  }
  return map;
}

function reservationTotal(map: Map<string, number>): number {
  let total = 0;
  for (const amount of map.values()) total += amount;
  return total;
}

/**
 * Remaining token budget for a work item derived from durable committed
 * receipts: the granted request budget minus tokens already consumed by
 * committed `model.generate` outcomes. Zero grants deny every call before
 * dispatch; exhausted grants are denied before the next paid call.
 */
function remainingTokenBudget(
  options: SkillEvolutionCapabilityGatewayOptions,
  workItem: TaskWorkItem,
): number | undefined {
  const input = options.taskStore.readJson(workItem.inputRef as string) as {
    budget?: { maxTokens?: number };
  };
  const maxTokens = input.budget?.maxTokens;
  if (typeof maxTokens !== "number" || maxTokens < 0) return undefined;
  const state = options.taskStore.state();
  const current = state.workItems[workItem.id];
  let usedTokens = 0;
  for (const receiptId of current?.sideEffectReceiptIds ?? []) {
    const receipt = state.sideEffectReceipts[receiptId];
    if (receipt?.status !== "committed" || !receipt.detailRef) continue;
    if (receipt.effectKind !== "skill_evolution:model.generate") continue;
    const detail = options.taskStore.readJson(receipt.detailRef) as {
      value?: { usage?: { totalTokens?: number } };
    };
    usedTokens += detail.value?.usage?.totalTokens ?? 0;
  }
  return Math.max(0, maxTokens - usedTokens);
}

function allowedArtifactRefs(
  options: SkillEvolutionCapabilityGatewayOptions,
  workItem: TaskWorkItem,
): Set<string> {
  const input = options.taskStore.readJson(workItem.inputRef as string) as {
    baselineContentRef?: string;
    trainDataset?: { contentRef?: string };
    devDataset?: { contentRef?: string };
  };
  const refs = new Set<string>();
  if (input.baselineContentRef) refs.add(input.baselineContentRef);
  if (input.trainDataset?.contentRef) refs.add(input.trainDataset.contentRef);
  if (input.devDataset?.contentRef) refs.add(input.devDataset.contentRef);
  return refs;
}

export function parseEvalCases(content: string): PairedSkillEvaluationCase[] {
  const cases: PairedSkillEvaluationCase[] = [];
  for (const [index, rawLine] of content.split("\n").entries()) {
    const line = rawLine.trim();
    if (!line) continue;
    const record = JSON.parse(line) as {
      id?: unknown;
      caseId?: unknown;
      input?: unknown;
      target?: unknown;
      expectedOutputContains?: unknown;
      safetyFlag?: unknown;
    };
    const caseId = typeof record.caseId === "string" ? record.caseId : String(record.id ?? index);
    if (typeof record.input !== "string") {
      throw new SkillEvolutionServiceError(
        "BAD_HOLDOUT",
        `Holdout case "${caseId}" has no string input.`,
      );
    }
    cases.push({
      caseId,
      input: record.input,
      target: typeof record.target === "string" ? record.target : undefined,
      expectedOutputContains:
        typeof record.expectedOutputContains === "string"
          ? record.expectedOutputContains
          : undefined,
      safetyFlag: typeof record.safetyFlag === "string" ? record.safetyFlag : undefined,
    });
  }
  return cases;
}

function parseDatasetCaseIds(records: unknown): Set<string> {
  const ids = new Set<string>();
  for (const record of Array.isArray(records) ? records : []) {
    const entry = record as { id?: unknown; caseId?: unknown };
    const caseId = typeof entry.caseId === "string" ? entry.caseId : String(entry.id ?? "");
    if (caseId) ids.add(caseId);
  }
  return ids;
}

function candidateArtifactsFor(
  state: TaskRuntimeState,
  workItem: TaskWorkItem,
): Array<{ artifactId: string; uri: string; mediaType?: string; sha256: string }> {
  const artifactIds = new Set<string>(workItem.artifactIds);
  for (const runId of workItem.runIds) {
    const run = state.runs[runId];
    for (const artifactId of run?.artifactIds ?? []) artifactIds.add(artifactId);
  }
  return [...artifactIds]
    .map((artifactId) => state.artifacts[artifactId])
    .filter((artifact) => artifact?.kind === "skill_candidate")
    .map((artifact) => ({
      artifactId: artifact.artifactId,
      uri: artifact.uri,
      mediaType: artifact.mediaType,
      sha256: artifact.sha256 ?? "",
    }));
}

export function artifactRefFor(uri: string): string {
  const basename = uri.split("/").at(-1) ?? "";
  const match = /^(?:sha256:)?([a-f0-9]{64})(?:\.blob)?$/.exec(decodeURIComponent(basename));
  if (match) return `sha256:${match[1]}`;
  throw new SkillEvolutionServiceError("BAD_ARTIFACT_URI", `Cannot resolve artifact uri "${uri}".`);
}

export function scanForSecrets(content: string): SkillCandidateSecretScan {
  const findings: string[] = [];
  const secretValuePattern =
    /(?:api[_-]?key|access[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key)/i;
  for (const [index, line] of content.split("\n").entries()) {
    if (secretValuePattern.test(line)) {
      findings.push(`line ${index + 1}`);
    }
  }
  return { passed: findings.length === 0, findings: findings.slice(0, 8) };
}

function auditInput(
  action: string,
  outcome: "attempted" | "completed" | "failed" | "denied",
  input: {
    requestId: string;
    actor: string;
    metadata: Record<string, string | null | number | boolean>;
    target?: { kind: string; id: string };
  },
): AuditInput {
  return {
    category: "extension",
    action,
    outcome,
    actor: { kind: "user", id: input.actor },
    target: input.target,
    requestId: input.requestId,
    metadata: input.metadata,
  };
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Capability call timed out after ${timeoutMs}ms.`));
    }, timeoutMs);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

function boundedErrorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return message.slice(0, 4_096);
}

function failed(code: string, message: string): TaskWorkerCapabilityOutcome {
  return { status: "failed", error: { code, message, retryable: false } };
}

function sha256(input: string | Uint8Array): string {
  return createHash("sha256").update(input).digest("hex");
}
