import { z } from "zod";
import { ExecutionRunSummarySchema } from "./execution-record.js";
import { HarnessSchema } from "./permissions.js";

const Id = z.string().min(1).max(256);
const Text = z.string().min(1).max(16_000);
const Money = z.number().finite().nonnegative();
export const WorkConfigurationSchema = z
  .strictObject({
    id: Id.default("temporary"),
    harness: HarnessSchema,
    model: Id,
    effort: Id.optional(),
    profile: z.enum(["sdk", "sdk-minimal"]).optional(),
  })
  .refine(
    (configuration) => configuration.profile === undefined || configuration.harness === "dsh",
    {
      message: "Profile selection is only supported by DSH.",
      path: ["profile"],
    },
  );
export type WorkConfiguration = z.infer<typeof WorkConfigurationSchema>;
export const WorkRuntimeSchema = z.strictObject({
  budgetUsd: Money.positive().optional(),
  timeoutMs: z.number().int().positive().max(2_147_483_647).optional(),
});
export const WorkRunSchema = z.strictObject({
  mode: z.enum(["manual", "managed"]).optional(),
  configuration: WorkConfigurationSchema.optional(),
  supervisor: WorkConfigurationSchema.optional(),
  runtime: WorkRuntimeSchema.optional(),
});
export const WorkCycleSchema = z.strictObject({
  id: Id,
  project: Id,
  budgetUsd: Money,
  concurrency: z.number().int().min(1).max(64).default(1),
  reviewReserveUsd: Money.default(0),
  configurations: z.array(WorkConfigurationSchema).max(32).default([]),
});
export const CreateWorkItemSchema = z.strictObject({
  id: Id,
  cycleId: Id,
  goal: Text,
  criteria: Text,
  criteriaVersion: Id,
  taskClass: Id,
  priority: z.number().int().default(0),
  value: Money.default(1),
  risk: z.enum(["ordinary", "high"]).default("ordinary"),
  deadline: z.iso.datetime().optional(),
  dependencies: z.array(Id).max(100).default([]),
  policy: z.enum(["fixed", "cheap", "adaptive"]).default("adaptive"),
  fixedConfiguration: Id.optional(),
  mode: z.enum(["manual", "managed"]).default("manual"),
  configuration: WorkConfigurationSchema.optional(),
  supervisor: WorkConfigurationSchema.optional(),
  runtime: WorkRuntimeSchema.default({}),
});
const WorkEvidenceSchema = z
  .array(z.templateLiteral(["urn:swarmx:execution:", z.uuid()]))
  .max(64)
  .refine(
    (sources) => new Set(sources).size === sources.length,
    "Evidence sources must be unique.",
  );
export const WorkArtifactSchema = z.strictObject({
  id: Id,
  revision: Id,
  evidence: WorkEvidenceSchema.optional(),
});
export const WorkArtifactSubmissionSchema = WorkArtifactSchema.extend({
  evidence: WorkEvidenceSchema.refine(
    (sources) => sources.length > 0,
    "Execution evidence is required.",
  ),
});
export const WorkItemSchema = CreateWorkItemSchema.extend({
  createdAt: z.iso.datetime(),
  state: z.enum(["queued", "running", "awaiting-acceptance", "accepted", "blocked"]),
  blockedReason: z.string().nullable(),
  firstAcceptedAt: z.iso.datetime().nullable(),
});
export type WorkItem = z.infer<typeof WorkItemSchema>;
export const WorkAttemptSchema = z.strictObject({
  id: Id,
  workId: Id,
  cycleId: Id,
  runtimeId: Id,
  configuration: WorkConfigurationSchema.nullable(),
  purpose: z.enum(["execution", "delegation", "memory-review"]),
  reservedUsd: Money,
  costUsd: Money.nullable(),
  costSource: z.enum(["native-estimate", "price-snapshot", "invoice", "unknown"]),
  coverage: z.enum(["complete", "partial", "unknown"]),
  state: z.enum(["reserved", "running", "settled", "uncertain"]),
  outcome: z.string().nullable(),
  createdAt: z.iso.datetime(),
  finishedAt: z.iso.datetime().nullable(),
  runIds: z.array(Id),
  artifacts: z.array(WorkArtifactSchema),
  criteriaVersion: Id,
  report: z.string().nullable(),
});
export type WorkAttempt = z.infer<typeof WorkAttemptSchema>;
export const WorkFeedbackInputSchema = z.strictObject({
  id: Id,
  attemptId: Id,
  criteriaVersion: Id,
  verdict: z.enum(["passed", "failed", "partial", "insufficient-evidence"]),
  accepted: z.boolean(),
  fraction: z.number().min(0).max(1),
  layer: z.enum(["structure", "behavior", "scientific-evidence", "user"]),
  source: z.enum(["user", "validator"]),
  evaluator: Id,
  evaluatorVersion: Id,
  report: Text,
  artifacts: z.array(WorkArtifactSchema).max(100).default([]),
  supersedes: Id.optional(),
  intervention: z.enum(["none", "revision", "takeover"]).default("none"),
});
export const WorkFeedbackSchema = WorkFeedbackInputSchema.extend({ recordedAt: z.iso.datetime() });
export type WorkFeedback = z.infer<typeof WorkFeedbackSchema>;
export const WorkSelectionEvidenceSchema = z.strictObject({
  configurationId: Id,
  samples: z.number().int(),
  accepted: z.number().int(),
  meanFraction: z.number().nullable(),
  observedCostUsd: Money.nullable(),
  costSamples: z.number().int(),
  feedbackIds: z.array(Id),
  latestFeedbackAt: z.iso.datetime().nullable(),
  meanElapsedMs: z.number().nullable(),
  executionOutcomes: z.array(z.string()),
});
export type SelectionEvidence = z.infer<typeof WorkSelectionEvidenceSchema>;
export const WorkDecisionSchema = z.strictObject({
  id: Id,
  workId: Id,
  at: z.iso.datetime(),
  policyVersion: Id,
  remainingUsd: z.number(),
  candidates: z.array(WorkConfigurationSchema),
  evidence: z.array(WorkSelectionEvidenceSchema),
  configurationId: Id.nullable(),
  action: z.enum(["execute", "defer", "stop"]),
  reason: Text,
});
export const WorkChargeSchema = z.strictObject({
  id: Id,
  reservationId: Id,
  costUsd: Money,
  source: z.literal("invoice"),
  reference: Text,
});
export type WorkCycle = z.infer<typeof WorkCycleSchema>;
export const ReviseWorkSchema = z.strictObject({
  id: Id,
  expectedCriteriaVersion: Id,
  criteriaVersion: Id,
  goal: Text,
  criteria: Text,
});
export const SetWorkBudgetSchema = z.strictObject({
  cycleId: Id,
  expectedBudgetUsd: Money,
  budgetUsd: Money,
});
export const ReconcileWorkOutcomeSchema = z.strictObject({
  reservationId: Id,
  outcome: z.enum(["cancelled", "failed", "completed"]),
  reference: Text,
});
export const WorkSnapshotSchema = z.strictObject({
  cycle: WorkCycleSchema,
  items: z.array(WorkItemSchema),
  reservations: z.array(WorkAttemptSchema),
  feedback: z.array(WorkFeedbackSchema),
  executions: z.array(ExecutionRunSummarySchema),
  outcomes: z.strictObject({
    acceptedItems: z.number().int().nonnegative(),
    acceptedValue: Money,
    partialValue: Money,
    interventions: z.number().int().nonnegative(),
    valueBasis: z.string(),
  }),
  decisions: z.array(WorkDecisionSchema),
  tools: z.strictObject({
    callCount: z.number().int().nonnegative(),
    usd: Money,
    unpricedCalls: z.number().int().nonnegative(),
  }),
  balance: z.strictObject({
    budgetUsd: Money,
    spentUsd: Money,
    heldUsd: Money,
    remainingUsd: z.number().finite(),
    costCoverage: z.strictObject({
      reported: z.number().int().nonnegative(),
      total: z.number().int().nonnegative(),
      completeExecutions: z.number().int().nonnegative(),
    }),
    enforcement: z.string(),
  }),
  coverageGaps: z.array(z.string()),
});
export type WorkSnapshot = z.infer<typeof WorkSnapshotSchema>;
export const WorkReadRequestSchema = z.strictObject({ cycleId: Id.optional() });
export const WorkReadSchema = z.strictObject({
  cycles: z.array(WorkCycleSchema),
  snapshot: WorkSnapshotSchema.nullable(),
  activeWorkIds: z.array(Id),
  interactions: z.array(
    z.strictObject({
      workId: Id,
      id: z.string().min(1),
      title: z.string(),
      schema: z.record(z.string(), z.unknown()),
    }),
  ),
});
export type WorkRead = z.infer<typeof WorkReadSchema>;
export const WorkCommandSchema = z.discriminatedUnion("action", [
  z.strictObject({ action: z.literal("createCycle"), request: WorkCycleSchema }),
  z.strictObject({ action: z.literal("createItem"), request: CreateWorkItemSchema }),
  z.strictObject({ action: z.literal("setBudget"), request: SetWorkBudgetSchema }),
  z.strictObject({ action: z.literal("revise"), request: ReviseWorkSchema }),
  z.strictObject({
    action: z.literal("accept"),
    request: WorkFeedbackInputSchema.omit({
      layer: true,
      source: true,
      evaluator: true,
      evaluatorVersion: true,
    }),
  }),
  z.strictObject({ action: z.literal("start"), workId: Id, options: WorkRunSchema.optional() }),
  z.strictObject({ action: z.literal("startNext"), cycleId: Id }),
  z.strictObject({ action: z.literal("stop"), workId: Id }),
  z.strictObject({
    action: z.literal("respond"),
    workId: Id,
    interactionId: z.string().min(1),
    answer: z.unknown().optional(),
    cancel: z.boolean().optional(),
  }),
  z.strictObject({ action: z.literal("reconcileCharge"), request: WorkChargeSchema }),
  z.strictObject({ action: z.literal("reconcileOutcome"), request: ReconcileWorkOutcomeSchema }),
]);
