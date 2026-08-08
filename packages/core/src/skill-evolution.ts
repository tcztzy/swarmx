import { createHash } from "node:crypto";
import { z } from "zod";
import type {
  SkillActivePointer,
  SkillCandidateManifest,
  SkillCandidateStaticChecks,
  SkillEvaluationGate,
  SkillEvaluationManifest,
  SkillEvaluationMetrics,
  SkillEvaluationSample,
  SkillOptimizationRequest,
  SkillPromotionReceipt,
} from "./skill-variants.js";
import {
  SkillCandidateManifestSchema,
  SkillEvaluationManifestSchema,
  SkillOptimizationRequestSchema,
  SkillPromotionReceiptSchema,
} from "./skill-variants.js";

export const SKILL_EVOLUTION_SCHEMA_VERSION = 1 as const;

export const SKILL_CANDIDATE_STATUSES = [
  "proposed",
  "evaluating",
  "staged",
  "rejected",
  "quarantined",
] as const;
export type SkillCandidateStatus = (typeof SKILL_CANDIDATE_STATUSES)[number];

export const SkillCandidateStatusSchema = z.enum(SKILL_CANDIDATE_STATUSES);

export const SKILL_CANDIDATE_TRANSITIONS: Record<SkillCandidateStatus, SkillCandidateStatus[]> = {
  proposed: ["evaluating", "rejected", "quarantined"],
  evaluating: ["staged", "rejected", "quarantined"],
  staged: ["rejected", "quarantined"],
  rejected: [],
  quarantined: [],
};

export class SkillEvolutionStateError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "SkillEvolutionStateError";
    this.code = code;
  }
}

export class SkillEvolutionCasError extends SkillEvolutionStateError {
  constructor(skillId: string, expectedRevisionId: string | null, actualRevisionId: string | null) {
    super(
      "STALE_ACTIVE_PARENT",
      `Active revision for Skill "${skillId}" is ${actualRevisionId ?? "<none>"}; expected ${expectedRevisionId ?? "<none>"}.`,
    );
  }
}

export class SkillEvolutionIdempotencyCollisionError extends SkillEvolutionStateError {
  readonly idempotencyKey: string;

  constructor(idempotencyKey: string) {
    super("IDEMPOTENCY_COLLISION", `Idempotency key "${idempotencyKey}" was reused.`);
    this.idempotencyKey = idempotencyKey;
  }
}

export function skillCandidateRevisionId(contentDigest: string): string {
  const match = /^sha256:([a-f0-9]{64})$/.exec(contentDigest);
  if (!match?.[1]) throw new SkillEvolutionStateError("BAD_DIGEST", "Content digest is malformed.");
  return `r_${match[1]}`;
}

/**
 * Canonical optimizer configuration digest shared with the Python sidecar. The
 * two implementations must agree byte-for-byte on the canonical JSON, so keys
 * are sorted alphabetically exactly like Python's `json.dumps(sort_keys=True,
 * separators=(",", ":"))`.
 */
export function canonicalSkillOptimizerConfig(input: {
  optimizerId: string;
  seed: number;
  proposer: string;
  budget: { maxModelCalls?: number; maxTokens?: number; maxWallTimeMs?: number };
}): string {
  const canonical = canonicalJson({
    schemaVersion: 1,
    optimizerId: input.optimizerId,
    seed: input.seed,
    proposer: input.proposer,
    budget: {
      maxModelCalls: input.budget.maxModelCalls ?? null,
      maxTokens: input.budget.maxTokens ?? null,
      maxWallTimeMs: input.budget.maxWallTimeMs ?? null,
    },
  });
  return `sha256:${createHash("sha256").update(canonical).digest("hex")}`;
}

function canonicalJson(input: unknown): string {
  if (Array.isArray(input)) {
    return `[${input.map((item) => canonicalJson(item)).join(",")}]`;
  }
  if (typeof input === "object" && input !== null) {
    const entries = Object.entries(input as Record<string, unknown>).sort(([left], [right]) =>
      left.localeCompare(right),
    );
    return `{${entries
      .map(([key, value]) => `${JSON.stringify(key)}:${canonicalJson(value)}`)
      .join(",")}}`;
  }
  return JSON.stringify(input);
}

export function assertSkillCandidateStatusTransition(
  from: SkillCandidateStatus,
  to: SkillCandidateStatus,
): void {
  if (from === to) return;
  if (!SKILL_CANDIDATE_TRANSITIONS[from].includes(to)) {
    throw new SkillEvolutionStateError(
      "INVALID_TRANSITION",
      `Skill candidate status cannot move from "${from}" to "${to}".`,
    );
  }
}

export function skillCandidateStaticChecksPassed(checks: SkillCandidateStaticChecks): boolean {
  return (
    checks.contentDigestVerified &&
    checks.parentRevisionDigestMatches &&
    checks.lineageMatchesRequest &&
    checks.instructionDeltaPresent &&
    checks.sizeWithinBudget &&
    checks.deliverySupported &&
    checks.secretScan.passed
  );
}

export interface SkillEvaluationVerdictInput {
  baseline: SkillEvaluationMetrics;
  candidate: SkillEvaluationMetrics;
  samples: readonly SkillEvaluationSample[];
  gate: SkillEvaluationGate;
  gateDigest: string;
}

export interface SkillEvaluationVerdict {
  verdict: "eligible" | "rejected";
  reasons: string[];
}

/**
 * Deterministic eligibility gate: strictly better quality, no regression on
 * safety, failure rate, or context; bounded latency/cost; a minimum sample
 * count; a minimum mean improvement; and a strictly positive improvement ratio
 * so a single mean move cannot declare success.
 */
export function evaluateSkillCandidateVerdict(
  input: SkillEvaluationVerdictInput,
): SkillEvaluationVerdict {
  const reasons: string[] = [];
  const { baseline, candidate, samples, gate } = input;
  const maxContextTokensPerSample = gate.maxContextTokensPerSample;
  if (samples.length < gate.minSampleCount) {
    reasons.push(
      `Holdout sample count ${samples.length} is below the minimum ${gate.minSampleCount}.`,
    );
  }
  if (candidate.quality <= baseline.quality + gate.minQualityImprovement) {
    reasons.push(
      `Candidate quality ${candidate.quality} does not strictly exceed baseline ${baseline.quality} by at least ${gate.minQualityImprovement}.`,
    );
  }
  if (candidate.safety < baseline.safety) {
    reasons.push(
      `Candidate safety ${candidate.safety} regressed below baseline ${baseline.safety}.`,
    );
  }
  if (candidate.failureRate > baseline.failureRate) {
    reasons.push(
      `Candidate failure rate ${candidate.failureRate} exceeds baseline ${baseline.failureRate}.`,
    );
  }
  if (candidate.contextTokens > baseline.contextTokens) {
    reasons.push(
      `Candidate context tokens ${candidate.contextTokens} exceed baseline ${baseline.contextTokens}.`,
    );
  }
  if (gate.maxLatencyMs !== undefined && candidate.latencyMs > gate.maxLatencyMs) {
    reasons.push(`Candidate latency ${candidate.latencyMs}ms exceeds ${gate.maxLatencyMs}ms.`);
  }
  if (gate.maxCostUsd !== undefined && (candidate.costUsd ?? 0) > gate.maxCostUsd) {
    reasons.push(
      `Candidate cost $${candidate.costUsd?.toFixed(4) ?? 0} exceeds $${gate.maxCostUsd}.`,
    );
  }
  if (
    maxContextTokensPerSample !== undefined &&
    samples.some((sample) => sample.candidate.contextTokens > maxContextTokensPerSample)
  ) {
    reasons.push(`A candidate sample exceeds ${maxContextTokensPerSample} context tokens.`);
  }
  const improved = samples.filter(
    (sample) => sample.candidate.passed && !sample.baseline.passed,
  ).length;
  const ratio = samples.length > 0 ? improved / samples.length : 0;
  if (ratio < gate.minImprovedRatio) {
    reasons.push(
      `Sample-level improvement ratio ${ratio.toFixed(2)} is below ${gate.minImprovedRatio}.`,
    );
  }
  if (reasons.length > 0) {
    return { verdict: "rejected", reasons: [`gate:${input.gateDigest}`, ...reasons] };
  }
  return { verdict: "eligible", reasons: [`gate:${input.gateDigest}`] };
}

export function skillEvaluationGateDigest(gate: SkillEvaluationGate): string {
  const canonical = JSON.stringify({
    schemaVersion: 1,
    minSampleCount: gate.minSampleCount,
    minQualityImprovement: gate.minQualityImprovement,
    minImprovedRatio: gate.minImprovedRatio,
    maxLatencyMs: gate.maxLatencyMs ?? null,
    maxCostUsd: gate.maxCostUsd ?? null,
    maxContextTokensPerSample: gate.maxContextTokensPerSample ?? null,
  });
  return `sha256:${createHash("sha256").update(canonical).digest("hex")}`;
}

export type SkillEvolutionRecordKind =
  | "optimization_requested"
  | "candidate_created"
  | "candidate_status_changed"
  | "evaluation_recorded"
  | "promotion_recorded";

const SkillEvolutionRecordBaseSchema = z
  .object({
    schemaVersion: z.literal(SKILL_EVOLUTION_SCHEMA_VERSION),
    recordId: z.string().min(1),
    timestamp: z.string().datetime(),
    idempotencyKey: z.string().min(1),
  })
  .strict();

export const OptimizationRequestedPayloadSchema = z
  .object({
    requestId: z.string().min(1),
    requestedBy: z.string().min(1).max(256).optional(),
    workItemId: z.string().min(1).optional(),
    request: z.unknown(),
  })
  .strict()
  .superRefine((payload, ctx) => {
    if (typeof payload.request !== "object" || payload.request === null) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["request"],
        message: "optimization_requested requires a request object.",
      });
    }
  });

export const CandidateCreatedPayloadSchema = z
  .object({
    manifest: z.unknown(),
    workItemId: z.string().min(1).optional(),
  })
  .strict()
  .superRefine((payload, ctx) => {
    if (typeof payload.manifest !== "object" || payload.manifest === null) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["manifest"],
        message: "candidate_created requires a manifest object.",
      });
    }
  });

export const CandidateStatusChangedPayloadSchema = z
  .object({
    candidateId: z.string().min(1),
    from: SkillCandidateStatusSchema,
    to: SkillCandidateStatusSchema,
    reason: z.string().min(1).max(4_096),
    evaluationRunId: z.string().min(1).optional(),
  })
  .strict();

export const EvaluationRecordedPayloadSchema = z
  .object({
    evaluationId: z.string().min(1),
    manifest: z.unknown(),
  })
  .strict()
  .superRefine((payload, ctx) => {
    if (typeof payload.manifest !== "object" || payload.manifest === null) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["manifest"],
        message: "evaluation_recorded requires a manifest object.",
      });
    }
  });

export const PromotedRevisionSchema = z
  .object({
    revisionId: z.string().regex(/^r_[a-f0-9]{64}$/),
    contentRef: z.string().regex(/^sha256:[a-f0-9]{64}$/),
    contentDigest: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  })
  .strict();

export const PromotionRecordedPayloadSchema = z
  .object({
    receipt: z.unknown(),
    promotedRevision: PromotedRevisionSchema.optional(),
  })
  .strict()
  .superRefine((payload, ctx) => {
    if (typeof payload.receipt !== "object" || payload.receipt === null) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["receipt"],
        message: "promotion_recorded requires a receipt object.",
      });
    }
  });

const OptimizationRequestedRecordSchema = SkillEvolutionRecordBaseSchema.extend({
  kind: z.literal("optimization_requested"),
  payload: OptimizationRequestedPayloadSchema,
}).strict();

const CandidateCreatedRecordSchema = SkillEvolutionRecordBaseSchema.extend({
  kind: z.literal("candidate_created"),
  payload: CandidateCreatedPayloadSchema,
}).strict();

const CandidateStatusChangedRecordSchema = SkillEvolutionRecordBaseSchema.extend({
  kind: z.literal("candidate_status_changed"),
  payload: CandidateStatusChangedPayloadSchema,
}).strict();

const EvaluationRecordedRecordSchema = SkillEvolutionRecordBaseSchema.extend({
  kind: z.literal("evaluation_recorded"),
  payload: EvaluationRecordedPayloadSchema,
}).strict();

const PromotionRecordedRecordSchema = SkillEvolutionRecordBaseSchema.extend({
  kind: z.literal("promotion_recorded"),
  payload: PromotionRecordedPayloadSchema,
}).strict();

export const SkillEvolutionRecordSchema = z
  .discriminatedUnion("kind", [
    OptimizationRequestedRecordSchema,
    CandidateCreatedRecordSchema,
    CandidateStatusChangedRecordSchema,
    EvaluationRecordedRecordSchema,
    PromotionRecordedRecordSchema,
  ])
  .superRefine((record, ctx) => {
    const secretKeyPattern =
      /(api[_-]?key|access[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key)/i;
    const visit = (value: unknown, path: Array<string | number>): void => {
      if (Array.isArray(value)) {
        value.forEach((child, index) => {
          visit(child, [...path, index]);
        });
        return;
      }
      if (!value || typeof value !== "object") return;
      for (const [key, child] of Object.entries(value)) {
        if (key === "secretScan") continue;
        if (secretKeyPattern.test(key)) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: [...path, key],
            message: `Skill evolution records must not contain inline secret field "${key}".`,
          });
        }
        visit(child, [...path, key]);
      }
    };
    visit(record.payload, ["payload"]);
  });

export type SkillEvolutionRecord = z.infer<typeof SkillEvolutionRecordSchema>;

export interface RetainedRevision {
  contentRef: string;
  contentDigest: string;
  variantId?: string;
  targetAgentId?: string;
  targetModelFingerprint?: string;
}

export interface SkillOptimizationRequestRecord {
  requestId: string;
  requestedBy?: string;
  workItemId?: string;
  request: SkillOptimizationRequest;
}

export interface SkillEvolutionState {
  schemaVersion: typeof SKILL_EVOLUTION_SCHEMA_VERSION;
  optimizationRequests: Record<string, SkillOptimizationRequestRecord>;
  candidates: Record<string, { manifest: SkillCandidateManifest; status: SkillCandidateStatus }>;
  evaluations: Record<string, SkillEvaluationManifest>;
  promotionReceipts: SkillPromotionReceipt[];
  activePointers: Record<string, SkillActivePointer>;
  retainedRevisions: Record<string, Record<string, RetainedRevision>>;
  quarantinedRevisions: Record<string, string[]>;
  recordIds: Record<string, string>;
  idempotencyFingerprints: Record<string, string>;
  records: SkillEvolutionRecord[];
}

export function emptySkillEvolutionState(): SkillEvolutionState {
  return {
    schemaVersion: SKILL_EVOLUTION_SCHEMA_VERSION,
    optimizationRequests: {},
    candidates: {},
    evaluations: {},
    promotionReceipts: [],
    activePointers: {},
    retainedRevisions: {},
    quarantinedRevisions: {},
    recordIds: {},
    idempotencyFingerprints: {},
    records: [],
  };
}

function recordFingerprint(record: SkillEvolutionRecord): string {
  return JSON.stringify(record);
}

export function applySkillEvolutionRecord(
  state: SkillEvolutionState,
  input: unknown,
): SkillEvolutionState {
  const record = SkillEvolutionRecordSchema.parse(input);
  const existingRecordFingerprint = state.recordIds[record.recordId];
  if (existingRecordFingerprint) {
    if (existingRecordFingerprint === recordFingerprint(record)) return state;
    throw new SkillEvolutionStateError(
      "RECORD_ID_COLLISION",
      `Evolution record id "${record.recordId}" was reused with different content.`,
    );
  }
  const existingIdempotencyFingerprint = state.idempotencyFingerprints[record.idempotencyKey];
  if (existingIdempotencyFingerprint) {
    if (existingIdempotencyFingerprint === recordFingerprint(record)) return state;
    throw new SkillEvolutionIdempotencyCollisionError(record.idempotencyKey);
  }
  const reduced = reduceSkillEvolutionRecord(state, record);
  return {
    ...reduced,
    records: [...reduced.records, record],
    recordIds: { ...reduced.recordIds, [record.recordId]: recordFingerprint(record) },
    idempotencyFingerprints: {
      ...reduced.idempotencyFingerprints,
      [record.idempotencyKey]: recordFingerprint(record),
    },
  };
}

export function replaySkillEvolutionRecords(records: readonly unknown[]): SkillEvolutionState {
  return records.reduce<SkillEvolutionState>(
    (state, record) => applySkillEvolutionRecord(state, record),
    emptySkillEvolutionState(),
  );
}

function reduceSkillEvolutionRecord(
  state: SkillEvolutionState,
  record: SkillEvolutionRecord,
): SkillEvolutionState {
  switch (record.kind) {
    case "optimization_requested": {
      const { requestId, requestedBy, workItemId, request: rawRequest } = record.payload;
      const request = SkillOptimizationRequestSchema.parse(rawRequest);
      if (state.optimizationRequests[requestId]) {
        throw new SkillEvolutionStateError(
          "REQUEST_ID_REUSE",
          `Optimization request id "${requestId}" was already recorded.`,
        );
      }
      const existingBaseline = state.retainedRevisions[request.skillId]?.[request.parentRevisionId];
      if (
        existingBaseline &&
        (existingBaseline.contentRef !== request.baselineContentRef ||
          existingBaseline.contentDigest !== request.baselineContentDigest)
      ) {
        throw new SkillEvolutionStateError(
          "RETAINED_REVISION_CONFLICT",
          `A retained revision "${request.parentRevisionId}" for Skill "${request.skillId}" already exists with different content; requests must not re-anchor it.`,
        );
      }
      return {
        ...state,
        optimizationRequests: {
          ...state.optimizationRequests,
          [requestId]: { requestId, requestedBy, workItemId, request },
        },
        retainedRevisions: {
          ...state.retainedRevisions,
          [request.skillId]: {
            ...state.retainedRevisions[request.skillId],
            [request.parentRevisionId]: {
              contentRef: request.baselineContentRef,
              contentDigest: request.baselineContentDigest,
              variantId: request.variantId,
              targetAgentId: request.targetAgentId,
              targetModelFingerprint: request.targetModelFingerprint,
            },
          },
        },
      };
    }
    case "candidate_created": {
      const manifest = SkillCandidateManifestSchema.parse(record.payload.manifest);
      if (state.candidates[manifest.candidateId]) {
        throw new SkillEvolutionStateError(
          "IMMUTABLE_CANDIDATE",
          `Skill candidate "${manifest.candidateId}" already exists and cannot be replaced.`,
        );
      }
      if (manifest.status !== "proposed") {
        throw new SkillEvolutionStateError(
          "CANDIDATE_MUST_START_PROPOSED",
          `Skill candidates must be created as "proposed", not "${manifest.status}".`,
        );
      }
      return {
        ...state,
        candidates: {
          ...state.candidates,
          [manifest.candidateId]: { manifest, status: manifest.status },
        },
      };
    }
    case "candidate_status_changed": {
      const { candidateId, from, to, reason, evaluationRunId } = record.payload;
      void reason;
      void evaluationRunId;
      const candidate = state.candidates[candidateId];
      if (!candidate) {
        throw new SkillEvolutionStateError(
          "UNKNOWN_CANDIDATE",
          `Skill candidate "${candidateId}" does not exist.`,
        );
      }
      if (candidate.status !== from) {
        throw new SkillEvolutionStateError(
          "STATUS_PRECONDITION",
          `Skill candidate "${candidateId}" is ${candidate.status}; expected ${from} before "${to}".`,
        );
      }
      assertSkillCandidateStatusTransition(candidate.status, to);
      if (
        (to === "rejected" || to === "quarantined") &&
        activePointerRevisionId(state, candidate.manifest.skillId) === candidate.manifest.revisionId
      ) {
        throw new SkillEvolutionStateError(
          "ACTIVE_REVISION_NOT_DECIDABLE",
          `Candidate "${candidateId}" is the active revision and cannot be rejected or quarantined.`,
        );
      }
      const quarantinedRevisions =
        to === "quarantined"
          ? {
              ...state.quarantinedRevisions,
              [candidate.manifest.skillId]: uniqueAppend(
                state.quarantinedRevisions[candidate.manifest.skillId] ?? [],
                candidate.manifest.revisionId,
              ),
            }
          : state.quarantinedRevisions;
      return {
        ...state,
        quarantinedRevisions,
        candidates: {
          ...state.candidates,
          [candidateId]: {
            ...candidate,
            manifest: { ...candidate.manifest, status: to },
            status: to,
          },
        },
      };
    }
    case "evaluation_recorded": {
      const manifest = SkillEvaluationManifestSchema.parse(record.payload.manifest);
      if (state.evaluations[manifest.evaluationId]) {
        throw new SkillEvolutionStateError(
          "IMMUTABLE_EVALUATION",
          `Skill evaluation "${manifest.evaluationId}" already exists and cannot be replaced.`,
        );
      }
      const candidate = state.candidates[manifest.candidateId];
      if (candidate?.status !== "evaluating") {
        throw new SkillEvolutionStateError(
          "EVALUATION_REQUIRES_EVALUATING",
          `Evaluation "${manifest.evaluationId}" requires candidate "${manifest.candidateId}" to be evaluating, not ${candidate?.status ?? "unknown"}.`,
        );
      }
      if (candidate.manifest.revisionId !== manifest.candidateRevisionId) {
        throw new SkillEvolutionStateError(
          "EVALUATION_REVISION_MISMATCH",
          "Evaluation candidate revision does not match the candidate manifest.",
        );
      }
      if (!manifest.samplesRef) {
        throw new SkillEvolutionStateError(
          "EVALUATION_REQUIRES_SAMPLES",
          "An evaluation manifest must bind its per-sample evidence artifact.",
        );
      }
      const recomputed = recomputeEvaluationVerdict(manifest);
      if (recomputed !== manifest.verdict) {
        throw new SkillEvolutionStateError(
          "EVALUATION_VERDICT_INCONSISTENT",
          `Evaluation verdict "${manifest.verdict}" contradicts its own metrics and gate (recomputed "${recomputed}").`,
        );
      }
      return {
        ...state,
        evaluations: { ...state.evaluations, [manifest.evaluationId]: manifest },
      };
    }
    case "promotion_recorded": {
      const receipt = SkillPromotionReceiptSchema.parse(record.payload.receipt);
      if (receipt.decision !== "promote" && receipt.decision !== "rollback") {
        throw new SkillEvolutionStateError(
          "INVALID_PROMOTION_DECISION",
          `Promotion records may only carry "promote" or "rollback" decisions, not "${receipt.decision}".`,
        );
      }
      const currentRevision = activePointerRevisionId(state, receipt.skillId);
      if (receipt.previousRevisionId !== currentRevision) {
        throw new SkillEvolutionCasError(
          receipt.skillId,
          receipt.previousRevisionId,
          currentRevision,
        );
      }
      if (receipt.decision === "rollback") {
        const retained = state.retainedRevisions[receipt.skillId] ?? {};
        const target = receipt.newRevisionId ? retained[receipt.newRevisionId] : undefined;
        if (receipt.newRevisionId === null || !target) {
          throw new SkillEvolutionStateError(
            "UNKNOWN_ROLLBACK_TARGET",
            `Rollback target revision "${receipt.newRevisionId ?? "<none>"}" is not retained.`,
          );
        }
        if ((state.quarantinedRevisions[receipt.skillId] ?? []).includes(receipt.newRevisionId)) {
          throw new SkillEvolutionStateError(
            "QUARANTINED_ROLLBACK_TARGET",
            `Rollback target revision "${receipt.newRevisionId}" is quarantined and cannot be re-activated.`,
          );
        }
        if (currentRevision === null) {
          throw new SkillEvolutionCasError(receipt.skillId, receipt.casExpectedRevisionId, null);
        }
        if (receipt.casExpectedRevisionId !== currentRevision) {
          throw new SkillEvolutionCasError(
            receipt.skillId,
            receipt.casExpectedRevisionId,
            currentRevision,
          );
        }
        const nextPointers = { ...state.activePointers };
        if (receipt.newRevisionId === currentRevision) {
          return { ...state, promotionReceipts: [...state.promotionReceipts, receipt] };
        }
        nextPointers[receipt.skillId] = {
          skillId: receipt.skillId,
          revisionId: receipt.newRevisionId,
          contentRef: target.contentRef,
          contentDigest: target.contentDigest,
          promotedAt: receipt.decidedAt,
          promotedBy: receipt.actor,
          receiptId: receipt.receiptId,
        };
        return {
          ...state,
          promotionReceipts: [...state.promotionReceipts, receipt],
          activePointers: nextPointers,
        };
      }
      // Promote: the receipt must reference a staged candidate with an
      // eligible evaluation, the first-promotion CAS is anchored to the
      // recorded optimization request's baseline, and the promoted content
      // coordinates must be exactly the candidate manifest's coordinates with
      // the revision id derived from the content digest.
      const promoted = record.payload.promotedRevision;
      if (!promoted) {
        throw new SkillEvolutionStateError(
          "MISSING_PROMOTED_REVISION",
          "A promote receipt requires the promoted revision content coordinates.",
        );
      }
      const candidate = receipt.candidateId ? state.candidates[receipt.candidateId] : undefined;
      const candidateStatus = candidate?.status;
      if (!candidate || candidateStatus !== "staged") {
        throw new SkillEvolutionStateError(
          "CANDIDATE_NOT_STAGED",
          `Promotion requires a staged candidate; "${receipt.candidateId ?? "<none>"}" is ${candidateStatus ?? "unknown"}.`,
        );
      }
      if (candidate.manifest.revisionId !== receipt.candidateRevisionId) {
        throw new SkillEvolutionStateError(
          "REVISION_MISMATCH",
          `Promotion revision "${receipt.candidateRevisionId}" does not match candidate "${candidate.manifest.revisionId}".`,
        );
      }
      if (
        promoted.revisionId !== candidate.manifest.revisionId ||
        promoted.contentRef !== candidate.manifest.contentRef ||
        promoted.contentDigest !== candidate.manifest.contentDigest
      ) {
        throw new SkillEvolutionStateError(
          "PROMOTED_CONTENT_MISMATCH",
          `Promoted content coordinates do not match the candidate manifest for "${candidate.manifest.candidateId}".`,
        );
      }
      if (promoted.revisionId !== skillCandidateRevisionId(promoted.contentDigest)) {
        throw new SkillEvolutionStateError(
          "REVISION_NOT_DERIVED",
          "Promotion revision id must be derived from the promoted content digest.",
        );
      }
      const evaluation = receipt.evaluationRunId
        ? state.evaluations[receipt.evaluationRunId]
        : undefined;
      if (
        !evaluation ||
        evaluation.candidateId !== receipt.candidateId ||
        evaluation.verdict !== "eligible"
      ) {
        throw new SkillEvolutionStateError(
          "MISSING_ELIGIBLE_EVALUATION",
          `Promotion requires an eligible evaluation for candidate "${receipt.candidateId}".`,
        );
      }
      const anchor = findOptimizationRequest(
        state,
        receipt.skillId,
        receipt.parentRevisionId ?? "",
      );
      if (!receipt.parentRevisionId || !anchor) {
        throw new SkillEvolutionStateError(
          "MISSING_OPTIMIZATION_ANCHOR",
          `No optimization request anchors Skill "${receipt.skillId}" parent "${receipt.parentRevisionId}".`,
        );
      }
      const effectiveCurrentRevision =
        currentRevision === null ? anchor.request.parentRevisionId : currentRevision;
      if (effectiveCurrentRevision !== receipt.casExpectedRevisionId) {
        throw new SkillEvolutionCasError(
          receipt.skillId,
          receipt.casExpectedRevisionId,
          effectiveCurrentRevision,
        );
      }
      if (receipt.newRevisionId !== promoted.revisionId) {
        throw new SkillEvolutionStateError(
          "REVISION_MISMATCH",
          "Promotion receipt and promoted revision coordinates disagree.",
        );
      }
      const nextPointers = { ...state.activePointers };
      if (receipt.newRevisionId === currentRevision) {
        return { ...state, promotionReceipts: [...state.promotionReceipts, receipt] };
      }
      nextPointers[receipt.skillId] = {
        skillId: receipt.skillId,
        revisionId: promoted.revisionId,
        contentRef: promoted.contentRef,
        contentDigest: promoted.contentDigest,
        promotedAt: receipt.decidedAt,
        promotedBy: receipt.actor,
        receiptId: receipt.receiptId,
      };
      return {
        ...state,
        promotionReceipts: [...state.promotionReceipts, receipt],
        activePointers: nextPointers,
        retainedRevisions: {
          ...state.retainedRevisions,
          [receipt.skillId]: {
            ...state.retainedRevisions[receipt.skillId],
            [promoted.revisionId]: {
              contentRef: promoted.contentRef,
              contentDigest: promoted.contentDigest,
              variantId: candidate.manifest.variantId,
              targetAgentId: candidate.manifest.targetAgentId,
              targetModelFingerprint: candidate.manifest.targetModelFingerprint,
            },
          },
        },
      };
    }
  }
}

function findOptimizationRequest(
  state: SkillEvolutionState,
  skillId: string,
  parentRevisionId: string,
): SkillOptimizationRequestRecord | undefined {
  for (const entry of Object.values(state.optimizationRequests)) {
    if (entry.request.skillId === skillId && entry.request.parentRevisionId === parentRevisionId) {
      return entry;
    }
  }
  return undefined;
}

export function activePointerRevisionId(
  state: SkillEvolutionState,
  skillId: string,
): string | null {
  const pointer = state.activePointers[skillId];
  return pointer?.revisionId ?? null;
}

export function retainedSkillRevisions(
  state: SkillEvolutionState,
  skillId: string,
): Record<string, RetainedRevision> {
  return state.retainedRevisions[skillId] ?? {};
}

function uniqueAppend(values: string[], value: string): string[] {
  return values.includes(value) ? values : [...values, value];
}

/**
 * Deterministic verdict recomputation from an evaluation manifest's own
 * aggregated metrics and gate. The sample-level ratio cannot be recomputed
 * without the per-sample artifact, so the manifest verdict must at least be
 * consistent with the aggregate rule set; the per-sample evidence is verified
 * by the service before the record is written.
 */
function recomputeEvaluationVerdict(manifest: SkillEvaluationManifest): "eligible" | "rejected" {
  const { baseline, candidate, gate } = manifest;
  const failures: string[] = [];
  if (manifest.sampleCount < gate.minSampleCount) {
    failures.push("sample count below minimum");
  }
  if (candidate.quality <= baseline.quality + gate.minQualityImprovement) {
    failures.push("quality not strictly improved");
  }
  if (candidate.safety < baseline.safety) {
    failures.push("safety regressed");
  }
  if (candidate.failureRate > baseline.failureRate) {
    failures.push("failure rate regressed");
  }
  if (candidate.contextTokens > baseline.contextTokens) {
    failures.push("context tokens regressed");
  }
  if (gate.maxLatencyMs !== undefined && candidate.latencyMs > gate.maxLatencyMs) {
    failures.push("latency above cap");
  }
  if (gate.maxCostUsd !== undefined && (candidate.costUsd ?? 0) > gate.maxCostUsd) {
    failures.push("cost above cap");
  }
  return failures.length > 0 ? "rejected" : "eligible";
}
