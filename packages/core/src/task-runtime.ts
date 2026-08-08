import { z } from "zod";
import { stableJson } from "./canonical-json.js";

export const TASK_RUNTIME_SCHEMA_VERSION = 1 as const;

export const TASK_RUNTIME_DELIVERY_SEMANTICS = Object.freeze({
  delivery: "at_least_once" as const,
  exactlyOnce: false as const,
  externalEffects: "idempotency_key_with_durable_receipt" as const,
});

const prefixedId = (prefix: string) =>
  z.string().regex(new RegExp(`^${prefix}[A-Za-z0-9][A-Za-z0-9_-]*$`), `Must use ${prefix} prefix`);

const TimestampSchema = z.string().datetime();
const WorkItemIdSchema = prefixedId("awi_");
const RunIdSchema = prefixedId("run_");
const EventIdSchema = prefixedId("evt_");
const LeaseIdSchema = prefixedId("lease_");
const CheckpointIdSchema = prefixedId("ckp_");
const ArtifactIdSchema = prefixedId("art_");
const ApprovalIdSchema = prefixedId("apr_");
const SessionLinkIdSchema = prefixedId("slnk_");
const ReceiptIdSchema = prefixedId("rcpt_");
const Sha256DigestSchema = z.string().regex(/^sha256:[a-f0-9]{64}$/);

export const TaskRuntimeSemanticsSchema = z
  .object({
    delivery: z.literal("at_least_once"),
    exactlyOnce: z.literal(false),
    externalEffects: z.literal("idempotency_key_with_durable_receipt"),
  })
  .strict();

export const TaskExecutorSchema = z
  .object({
    backend: z.string().min(1),
    operation: z.string().min(1),
  })
  .strict();

export const TaskWorkItemStatusSchema = z.enum([
  "queued",
  "leased",
  "running",
  "blocked",
  "needs_human",
  "failed",
  "succeeded",
  "canceled",
  "superseded",
]);

export const TaskRunStatusSchema = z.enum([
  "created",
  "leased",
  "running",
  "cancel_requested",
  "needs_human",
  "interrupted",
  "failed",
  "succeeded",
  "canceled",
]);

export const TaskBudgetSchema = z
  .object({
    wallTimeMs: z.number().int().positive().optional(),
    maxArtifactBytes: z.number().int().nonnegative().optional(),
    maxCheckpoints: z.number().int().nonnegative().optional(),
    maxProgressEvents: z.number().int().nonnegative().optional(),
    capabilityCalls: z.record(z.string().min(1), z.number().int().nonnegative()).optional(),
  })
  .strict();

export const TaskBudgetUsageSchema = z
  .object({
    wallTimeMs: z.number().int().nonnegative().default(0),
    artifactBytes: z.number().int().nonnegative().default(0),
    checkpoints: z.number().int().nonnegative().default(0),
    progressEvents: z.number().int().nonnegative().default(0),
    capabilityCalls: z.record(z.string().min(1), z.number().int().nonnegative()).default({}),
  })
  .strict();

export const TaskRetryStateSchema = z
  .object({
    attemptsStarted: z.number().int().nonnegative().default(0),
    maxAttempts: z.number().int().positive().default(1),
    nextAttemptAt: TimestampSchema.optional(),
    lastFailure: z.string().min(1).optional(),
  })
  .strict();

export const TaskLeaseSchema = z
  .object({
    leaseId: LeaseIdSchema,
    workItemId: WorkItemIdSchema,
    runId: RunIdSchema,
    workerId: z.string().min(1),
    fencingToken: z.number().int().positive(),
    acquiredAt: TimestampSchema,
    heartbeatAt: TimestampSchema,
    expiresAt: TimestampSchema,
    budgetSnapshot: TaskBudgetSchema.optional(),
  })
  .strict()
  .superRefine((lease, ctx) => {
    const acquiredAt = Date.parse(lease.acquiredAt);
    const heartbeatAt = Date.parse(lease.heartbeatAt);
    const expiresAt = Date.parse(lease.expiresAt);
    if (heartbeatAt < acquiredAt) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["heartbeatAt"],
        message: "Lease heartbeat cannot precede acquisition.",
      });
    }
    if (expiresAt <= heartbeatAt) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["expiresAt"],
        message: "Lease expiry must follow its latest heartbeat.",
      });
    }
  });

export const TaskLeaseClaimSchema = z
  .object({
    leaseId: LeaseIdSchema,
    fencingToken: z.number().int().positive(),
  })
  .strict();

export const TaskCancellationSchema = z
  .object({
    status: z.enum(["requested", "acknowledged"]),
    requestedAt: TimestampSchema,
    acknowledgedAt: TimestampSchema.optional(),
    reason: z.string().min(1).optional(),
  })
  .strict()
  .superRefine((cancellation, ctx) => {
    if (cancellation.status === "acknowledged" && !cancellation.acknowledgedAt) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["acknowledgedAt"],
        message: "Acknowledged cancellation requires acknowledgedAt.",
      });
    }
    if (cancellation.status === "requested" && cancellation.acknowledgedAt) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["acknowledgedAt"],
        message: "Requested cancellation cannot already be acknowledged.",
      });
    }
  });

export const TaskFailureSchema = z
  .object({
    occurredAt: TimestampSchema,
    message: z.string().min(1),
    code: z.string().min(1).optional(),
    retryable: z.boolean().default(false),
  })
  .strict();

export const TaskProgressSchema = z
  .object({
    sequence: z.number().int().nonnegative(),
    recordedAt: TimestampSchema,
    message: z.string().min(1).optional(),
    completedUnits: z.number().nonnegative().optional(),
    totalUnits: z.number().positive().optional(),
    detailsRef: z.string().min(1).optional(),
  })
  .strict()
  .superRefine((progress, ctx) => {
    if (
      progress.completedUnits !== undefined &&
      progress.totalUnits !== undefined &&
      progress.completedUnits > progress.totalUnits
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["completedUnits"],
        message: "Progress cannot exceed total units.",
      });
    }
  });

export const TaskCheckpointSchema = z
  .object({
    checkpointId: CheckpointIdSchema,
    workItemId: WorkItemIdSchema,
    runId: RunIdSchema,
    sequence: z.number().int().nonnegative(),
    createdAt: TimestampSchema,
    resumeRef: z.string().min(1),
    checksum: Sha256DigestSchema.optional(),
    environmentDigest: Sha256DigestSchema,
    parentCheckpointId: CheckpointIdSchema.optional(),
    artifactIds: z.array(ArtifactIdSchema).default([]),
  })
  .strict();

export const TaskArtifactReferenceSchema = z
  .object({
    artifactId: ArtifactIdSchema,
    workItemId: WorkItemIdSchema,
    runId: RunIdSchema.optional(),
    kind: z.string().min(1),
    uri: z.string().min(1),
    createdAt: TimestampSchema,
    mediaType: z.string().min(1).optional(),
    sha256: z
      .string()
      .regex(/^[a-f0-9]{64}$/)
      .optional(),
    sizeBytes: z.number().int().nonnegative().optional(),
    immutable: z.boolean().default(true),
  })
  .strict();

export const TaskApprovalStatusSchema = z.enum(["requested", "approved", "rejected", "waived"]);

export const TaskApprovalSchema = z
  .object({
    approvalId: ApprovalIdSchema,
    workItemId: WorkItemIdSchema,
    runId: RunIdSchema.optional(),
    kind: z.string().min(1),
    status: TaskApprovalStatusSchema,
    requestedAt: TimestampSchema,
    requestedBy: z.string().min(1).optional(),
    requestRef: z.string().min(1).optional(),
    decidedAt: TimestampSchema.optional(),
    decidedBy: z.string().min(1).optional(),
    decisionRef: z.string().min(1).optional(),
    reason: z.string().min(1).optional(),
  })
  .strict()
  .superRefine((approval, ctx) => {
    const decided = approval.status !== "requested";
    if (decided && (!approval.decidedAt || !approval.decidedBy)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["decidedAt"],
        message: "A decided approval requires decidedAt and decidedBy.",
      });
    }
    if (!decided && (approval.decidedAt || approval.decidedBy || approval.decisionRef)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["decidedAt"],
        message: "A requested approval cannot include a decision.",
      });
    }
    if (approval.decidedAt && Date.parse(approval.decidedAt) < Date.parse(approval.requestedAt)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["decidedAt"],
        message: "Approval decision cannot precede its request.",
      });
    }
  });

export const TaskScheduleCadenceSchema = z
  .object({
    kind: z.literal("interval"),
    everySeconds: z.number().int().positive(),
  })
  .strict();

export const TaskScheduleSchema = z
  .object({
    scheduleId: z.string().min(1),
    enabled: z.boolean().default(true),
    cadence: TaskScheduleCadenceSchema,
    lastTriggeredAt: TimestampSchema.optional(),
    nextDueAt: TimestampSchema.optional(),
  })
  .strict();

export const TaskScheduleDecisionSchema = z
  .object({
    scheduleId: z.string().min(1),
    due: z.boolean(),
    disabled: z.boolean(),
    now: TimestampSchema,
    dueAt: TimestampSchema.optional(),
    nextDueAt: TimestampSchema.optional(),
    idempotencyKey: z.string().min(1).optional(),
  })
  .strict();

export const TaskSessionLinkSchema = z
  .object({
    linkId: SessionLinkIdSchema,
    workItemId: WorkItemIdSchema,
    sessionId: z.string().min(1),
    role: z.enum(["creator", "observer"]),
    linkedAt: TimestampSchema,
    unlinkedAt: TimestampSchema.optional(),
  })
  .strict()
  .superRefine((link, ctx) => {
    if (link.unlinkedAt && Date.parse(link.unlinkedAt) < Date.parse(link.linkedAt)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["unlinkedAt"],
        message: "Session unlink cannot precede its link.",
      });
    }
  });

export const TaskSideEffectReceiptStatusSchema = z.enum([
  "uncertain",
  "not_committed",
  "committed",
]);

export const TaskSideEffectReceiptSchema = z
  .object({
    receiptId: ReceiptIdSchema,
    workItemId: WorkItemIdSchema,
    runId: RunIdSchema,
    effectKind: z.string().min(1),
    effectIdempotencyKey: z.string().min(1),
    status: TaskSideEffectReceiptStatusSchema,
    recordedAt: TimestampSchema,
    externalRef: z.string().min(1).optional(),
    detailRef: z.string().min(1).optional(),
    delivery: z.literal("at_least_once").default("at_least_once"),
    exactlyOnce: z.literal(false).default(false),
  })
  .strict();

export const TaskWorkItemSchema = z
  .object({
    id: WorkItemIdSchema,
    status: TaskWorkItemStatusSchema,
    executor: TaskExecutorSchema,
    priority: z.number().int().default(0),
    createdAt: TimestampSchema,
    updatedAt: TimestampSchema,
    revision: z.number().int().nonnegative().default(0),
    inputRef: z.string().min(1).optional(),
    owner: z.string().min(1).optional(),
    budget: TaskBudgetSchema.optional(),
    budgetUsage: TaskBudgetUsageSchema.default({
      wallTimeMs: 0,
      artifactBytes: 0,
      checkpoints: 0,
      progressEvents: 0,
      capabilityCalls: {},
    }),
    retry: TaskRetryStateSchema.default({ attemptsStarted: 0, maxAttempts: 1 }),
    lease: TaskLeaseSchema.optional(),
    lastFencingToken: z.number().int().nonnegative().default(0),
    runIds: z.array(RunIdSchema).default([]),
    activeRunId: RunIdSchema.optional(),
    latestCheckpointId: CheckpointIdSchema.optional(),
    artifactIds: z.array(ArtifactIdSchema).default([]),
    approvalIds: z.array(ApprovalIdSchema).default([]),
    sessionLinkIds: z.array(SessionLinkIdSchema).default([]),
    sideEffectReceiptIds: z.array(ReceiptIdSchema).default([]),
    cancellation: TaskCancellationSchema.optional(),
    blockedReason: z.string().min(1).optional(),
    needsHumanReason: z.string().min(1).optional(),
    supersededBy: WorkItemIdSchema.optional(),
  })
  .strict()
  .superRefine((workItem, ctx) => {
    if (Date.parse(workItem.updatedAt) < Date.parse(workItem.createdAt)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["updatedAt"],
        message: "Work item updatedAt cannot precede createdAt.",
      });
    }
    if (["leased", "running"].includes(workItem.status) && !workItem.lease) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["lease"],
        message: `${workItem.status} work item requires a lease.`,
      });
    }
    if (workItem.lease && workItem.lease.workItemId !== workItem.id) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["lease", "workItemId"],
        message: "Lease workItemId must match its work item.",
      });
    }
    if (workItem.activeRunId && !workItem.runIds.includes(workItem.activeRunId)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["activeRunId"],
        message: "activeRunId must appear in runIds.",
      });
    }
    if (workItem.lease && workItem.activeRunId !== workItem.lease.runId) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["activeRunId"],
        message: "Active run must match the lease run.",
      });
    }
    if (workItem.retry.attemptsStarted > workItem.retry.maxAttempts) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["retry", "attemptsStarted"],
        message: "Started attempts cannot exceed maxAttempts.",
      });
    }
  });

export const TaskRunSchema = z
  .object({
    runId: RunIdSchema,
    workItemId: WorkItemIdSchema,
    executor: TaskExecutorSchema,
    status: TaskRunStatusSchema,
    attempt: z.number().int().positive(),
    createdAt: TimestampSchema,
    startedAt: TimestampSchema.optional(),
    endedAt: TimestampSchema.optional(),
    environmentDigest: Sha256DigestSchema,
    resumeFromCheckpointId: CheckpointIdSchema.optional(),
    leaseId: LeaseIdSchema.optional(),
    fencingToken: z.number().int().positive().optional(),
    latestCheckpointId: CheckpointIdSchema.optional(),
    artifactIds: z.array(ArtifactIdSchema).default([]),
    sideEffectReceiptIds: z.array(ReceiptIdSchema).default([]),
    lastProgress: TaskProgressSchema.optional(),
    cancellation: TaskCancellationSchema.optional(),
    failure: TaskFailureSchema.optional(),
    resultRef: z.string().min(1).optional(),
  })
  .strict()
  .superRefine((run, ctx) => {
    const leaseRequired = ["leased", "running", "cancel_requested"].includes(run.status);
    if (leaseRequired && (!run.leaseId || !run.fencingToken)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["leaseId"],
        message: `${run.status} run requires a lease and fencing token.`,
      });
    }
    const terminal = ["interrupted", "failed", "succeeded", "canceled", "needs_human"].includes(
      run.status,
    );
    if (terminal && !run.endedAt) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["endedAt"],
        message: `${run.status} run requires endedAt.`,
      });
    }
    if (["interrupted", "failed"].includes(run.status) && !run.failure) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["failure"],
        message: `${run.status} run requires failure details.`,
      });
    }
    if (run.status === "cancel_requested" && run.cancellation?.status !== "requested") {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["cancellation"],
        message: "cancel_requested run requires a requested cancellation.",
      });
    }
    if (run.status === "canceled" && run.cancellation?.status !== "acknowledged") {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["cancellation"],
        message: "Canceled run requires acknowledged cancellation.",
      });
    }
  });

const RuntimeEventBaseShape = {
  schemaVersion: z.literal(TASK_RUNTIME_SCHEMA_VERSION),
  eventId: EventIdSchema,
  timestamp: TimestampSchema,
  source: z.string().min(1),
  idempotencyKey: z.string().min(1),
  workItemId: WorkItemIdSchema,
};

const RuntimeRunEventBaseShape = {
  ...RuntimeEventBaseShape,
  runId: RunIdSchema,
};

const WorkItemCreatedEventSchema = z
  .object({
    ...RuntimeEventBaseShape,
    eventType: z.literal("work_item_created"),
    payload: z.object({ workItem: TaskWorkItemSchema }).strict(),
  })
  .strict();

const RunCreatedEventSchema = z
  .object({
    ...RuntimeRunEventBaseShape,
    eventType: z.literal("run_created"),
    payload: z.object({ run: TaskRunSchema }).strict(),
  })
  .strict();

const LeaseAcquiredEventSchema = z
  .object({
    ...RuntimeRunEventBaseShape,
    eventType: z.literal("lease_acquired"),
    payload: z.object({ lease: TaskLeaseSchema }).strict(),
  })
  .strict();

const LeaseHeartbeatEventSchema = z
  .object({
    ...RuntimeRunEventBaseShape,
    eventType: z.literal("lease_heartbeat"),
    payload: z
      .object({
        claim: TaskLeaseClaimSchema,
        heartbeatAt: TimestampSchema,
        expiresAt: TimestampSchema,
      })
      .strict(),
  })
  .strict();

const LeaseExpiredEventSchema = z
  .object({
    ...RuntimeRunEventBaseShape,
    eventType: z.literal("lease_expired"),
    payload: z
      .object({
        claim: TaskLeaseClaimSchema,
        expiredAt: TimestampSchema,
        reason: z.string().min(1).optional(),
      })
      .strict(),
  })
  .strict();

const RunStartedEventSchema = z
  .object({
    ...RuntimeRunEventBaseShape,
    eventType: z.literal("run_started"),
    payload: z.object({ claim: TaskLeaseClaimSchema, startedAt: TimestampSchema }).strict(),
  })
  .strict();

const ProgressRecordedEventSchema = z
  .object({
    ...RuntimeRunEventBaseShape,
    eventType: z.literal("progress_recorded"),
    payload: z.object({ claim: TaskLeaseClaimSchema, progress: TaskProgressSchema }).strict(),
  })
  .strict();

const CheckpointRecordedEventSchema = z
  .object({
    ...RuntimeRunEventBaseShape,
    eventType: z.literal("checkpoint_recorded"),
    payload: z.object({ claim: TaskLeaseClaimSchema, checkpoint: TaskCheckpointSchema }).strict(),
  })
  .strict();

const ArtifactRecordedEventSchema = z
  .object({
    ...RuntimeRunEventBaseShape,
    eventType: z.literal("artifact_recorded"),
    payload: z
      .object({ claim: TaskLeaseClaimSchema, artifact: TaskArtifactReferenceSchema })
      .strict(),
  })
  .strict();

const NeedsHumanEventSchema = z
  .object({
    ...RuntimeRunEventBaseShape,
    eventType: z.literal("needs_human"),
    payload: z
      .object({
        claim: TaskLeaseClaimSchema,
        requestedAt: TimestampSchema,
        reason: z.string().min(1),
        approvalId: ApprovalIdSchema.optional(),
      })
      .strict(),
  })
  .strict();

const ApprovalRecordedEventSchema = z
  .object({
    ...RuntimeEventBaseShape,
    eventType: z.literal("approval_recorded"),
    payload: z.object({ approval: TaskApprovalSchema }).strict(),
  })
  .strict();

const SessionLinkedEventSchema = z
  .object({
    ...RuntimeEventBaseShape,
    eventType: z.literal("session_linked"),
    payload: z.object({ link: TaskSessionLinkSchema }).strict(),
  })
  .strict();

const SessionUnlinkedEventSchema = z
  .object({
    ...RuntimeEventBaseShape,
    eventType: z.literal("session_unlinked"),
    payload: z.object({ linkId: SessionLinkIdSchema, unlinkedAt: TimestampSchema }).strict(),
  })
  .strict();

const CancelRequestedEventSchema = z
  .object({
    ...RuntimeRunEventBaseShape,
    eventType: z.literal("cancel_requested"),
    payload: z
      .object({ requestedAt: TimestampSchema, reason: z.string().min(1).optional() })
      .strict(),
  })
  .strict();

const CancelAcknowledgedEventSchema = z
  .object({
    ...RuntimeRunEventBaseShape,
    eventType: z.literal("cancel_acknowledged"),
    payload: z
      .object({
        claim: TaskLeaseClaimSchema,
        acknowledgedAt: TimestampSchema,
        reason: z.string().min(1).optional(),
      })
      .strict(),
  })
  .strict();

const RetryScheduledEventSchema = z
  .object({
    ...RuntimeEventBaseShape,
    eventType: z.literal("retry_scheduled"),
    payload: z
      .object({
        scheduledAt: TimestampSchema,
        nextAttemptAt: TimestampSchema,
        reason: z.string().min(1),
        resumeFromCheckpointId: CheckpointIdSchema.optional(),
        approvalId: ApprovalIdSchema.optional(),
      })
      .strict(),
  })
  .strict();

const SideEffectReceiptRecordedEventSchema = z
  .object({
    ...RuntimeRunEventBaseShape,
    eventType: z.literal("side_effect_receipt_recorded"),
    payload: z
      .object({ claim: TaskLeaseClaimSchema, receipt: TaskSideEffectReceiptSchema })
      .strict(),
  })
  .strict();

const RunCompletedEventSchema = z
  .object({
    ...RuntimeRunEventBaseShape,
    eventType: z.literal("run_completed"),
    payload: z
      .object({
        claim: TaskLeaseClaimSchema,
        completedAt: TimestampSchema,
        resultRef: z.string().min(1).optional(),
      })
      .strict(),
  })
  .strict();

const RunFailedEventSchema = z
  .object({
    ...RuntimeRunEventBaseShape,
    eventType: z.literal("run_failed"),
    payload: z.object({ claim: TaskLeaseClaimSchema, failure: TaskFailureSchema }).strict(),
  })
  .strict();

const WorkItemBlockedEventSchema = z
  .object({
    ...RuntimeEventBaseShape,
    eventType: z.literal("work_item_blocked"),
    payload: z.object({ reason: z.string().min(1) }).strict(),
  })
  .strict();

const WorkItemSupersededEventSchema = z
  .object({
    ...RuntimeEventBaseShape,
    eventType: z.literal("work_item_superseded"),
    payload: z.object({ supersededBy: WorkItemIdSchema.optional() }).strict(),
  })
  .strict();

export const TaskRuntimeEventSchema = z.discriminatedUnion("eventType", [
  WorkItemCreatedEventSchema,
  RunCreatedEventSchema,
  LeaseAcquiredEventSchema,
  LeaseHeartbeatEventSchema,
  LeaseExpiredEventSchema,
  RunStartedEventSchema,
  ProgressRecordedEventSchema,
  CheckpointRecordedEventSchema,
  ArtifactRecordedEventSchema,
  NeedsHumanEventSchema,
  ApprovalRecordedEventSchema,
  SessionLinkedEventSchema,
  SessionUnlinkedEventSchema,
  CancelRequestedEventSchema,
  CancelAcknowledgedEventSchema,
  RetryScheduledEventSchema,
  SideEffectReceiptRecordedEventSchema,
  RunCompletedEventSchema,
  RunFailedEventSchema,
  WorkItemBlockedEventSchema,
  WorkItemSupersededEventSchema,
]);

export type TaskRuntimeSemantics = z.infer<typeof TaskRuntimeSemanticsSchema>;
export type TaskExecutor = z.infer<typeof TaskExecutorSchema>;
export type TaskWorkItemStatus = z.infer<typeof TaskWorkItemStatusSchema>;
export type TaskRunStatus = z.infer<typeof TaskRunStatusSchema>;
export type TaskBudget = z.infer<typeof TaskBudgetSchema>;
export type TaskBudgetUsage = z.infer<typeof TaskBudgetUsageSchema>;
export type TaskRetryState = z.infer<typeof TaskRetryStateSchema>;
export type TaskLease = z.infer<typeof TaskLeaseSchema>;
export type TaskLeaseClaim = z.infer<typeof TaskLeaseClaimSchema>;
export type TaskCancellation = z.infer<typeof TaskCancellationSchema>;
export type TaskFailure = z.infer<typeof TaskFailureSchema>;
export type TaskProgress = z.infer<typeof TaskProgressSchema>;
export type TaskCheckpoint = z.infer<typeof TaskCheckpointSchema>;
export type TaskArtifactReference = z.infer<typeof TaskArtifactReferenceSchema>;
export type TaskApprovalStatus = z.infer<typeof TaskApprovalStatusSchema>;
export type TaskApproval = z.infer<typeof TaskApprovalSchema>;
export type TaskSchedule = z.infer<typeof TaskScheduleSchema>;
export type TaskScheduleDecision = z.infer<typeof TaskScheduleDecisionSchema>;
export type TaskSessionLink = z.infer<typeof TaskSessionLinkSchema>;
export type TaskSideEffectReceiptStatus = z.infer<typeof TaskSideEffectReceiptStatusSchema>;
export type TaskSideEffectReceipt = z.infer<typeof TaskSideEffectReceiptSchema>;
export type TaskWorkItem = z.infer<typeof TaskWorkItemSchema>;
export type TaskRun = z.infer<typeof TaskRunSchema>;
export type TaskRuntimeEvent = z.infer<typeof TaskRuntimeEventSchema>;

export const TaskRuntimeStateSchema = z
  .object({
    schemaVersion: z.literal(TASK_RUNTIME_SCHEMA_VERSION),
    semantics: TaskRuntimeSemanticsSchema,
    workItems: z.record(WorkItemIdSchema, TaskWorkItemSchema),
    runs: z.record(RunIdSchema, TaskRunSchema),
    checkpoints: z.record(CheckpointIdSchema, TaskCheckpointSchema),
    artifacts: z.record(ArtifactIdSchema, TaskArtifactReferenceSchema),
    approvals: z.record(ApprovalIdSchema, TaskApprovalSchema),
    sessionLinks: z.record(SessionLinkIdSchema, TaskSessionLinkSchema),
    sideEffectReceipts: z.record(ReceiptIdSchema, TaskSideEffectReceiptSchema),
    events: z.array(TaskRuntimeEventSchema),
    eventFingerprints: z.record(EventIdSchema, z.string().min(1)),
    idempotencyFingerprints: z.record(z.string().min(1), z.string().min(1)),
  })
  .strict();

export type TaskRuntimeState = z.infer<typeof TaskRuntimeStateSchema>;

type DistributiveCreateEventInput<T> = T extends TaskRuntimeEvent
  ? Omit<T, "schemaVersion" | "eventId"> & {
      schemaVersion?: typeof TASK_RUNTIME_SCHEMA_VERSION;
      eventId?: string;
    }
  : never;

export type CreateTaskRuntimeEventInput = DistributiveCreateEventInput<TaskRuntimeEvent>;

export class TaskRuntimeInvariantError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "TaskRuntimeInvariantError";
    this.code = code;
  }
}

export class TaskRuntimeIdempotencyCollisionError extends TaskRuntimeInvariantError {
  readonly idempotencyKey: string;
  readonly existingFingerprint: string;
  readonly incomingFingerprint: string;

  constructor(idempotencyKey: string, existingFingerprint: string, incomingFingerprint: string) {
    super(
      "IDEMPOTENCY_COLLISION",
      `Idempotency key "${idempotencyKey}" was reused for a different runtime event.`,
    );
    this.name = "TaskRuntimeIdempotencyCollisionError";
    this.idempotencyKey = idempotencyKey;
    this.existingFingerprint = existingFingerprint;
    this.incomingFingerprint = incomingFingerprint;
  }
}

export function createTaskRuntimeEvent(input: CreateTaskRuntimeEventInput): TaskRuntimeEvent {
  const schemaVersion = input.schemaVersion ?? TASK_RUNTIME_SCHEMA_VERSION;
  const eventId = input.eventId ?? deterministicId("evt_", { ...input, schemaVersion });
  return TaskRuntimeEventSchema.parse({ ...input, schemaVersion, eventId });
}

export function emptyTaskRuntimeState(): TaskRuntimeState {
  return {
    schemaVersion: TASK_RUNTIME_SCHEMA_VERSION,
    semantics: { ...TASK_RUNTIME_DELIVERY_SEMANTICS },
    workItems: {},
    runs: {},
    checkpoints: {},
    artifacts: {},
    approvals: {},
    sessionLinks: {},
    sideEffectReceipts: {},
    events: [],
    eventFingerprints: {},
    idempotencyFingerprints: {},
  };
}

export function replayTaskRuntimeEvents(events: readonly unknown[]): TaskRuntimeState {
  return events.reduce<TaskRuntimeState>(
    (state, event) => applyTaskRuntimeEvent(state, event),
    emptyTaskRuntimeState(),
  );
}

export function applyTaskRuntimeEvent(state: TaskRuntimeState, input: unknown): TaskRuntimeState {
  const event = TaskRuntimeEventSchema.parse(input);
  const fingerprint = runtimeEventFingerprint(event);
  const existingEventFingerprint = state.eventFingerprints[event.eventId];
  if (existingEventFingerprint) {
    if (existingEventFingerprint === fingerprint) return state;
    throw new TaskRuntimeInvariantError(
      "EVENT_ID_COLLISION",
      `Event id "${event.eventId}" was reused for a different runtime event.`,
    );
  }

  const existingIdempotencyFingerprint = state.idempotencyFingerprints[event.idempotencyKey];
  if (existingIdempotencyFingerprint) {
    if (existingIdempotencyFingerprint === fingerprint) return state;
    throw new TaskRuntimeIdempotencyCollisionError(
      event.idempotencyKey,
      existingIdempotencyFingerprint,
      fingerprint,
    );
  }

  const reduced = reduceTaskRuntimeEvent(state, event);
  return {
    ...reduced,
    events: [...reduced.events, event],
    eventFingerprints: {
      ...reduced.eventFingerprints,
      [event.eventId]: fingerprint,
    },
    idempotencyFingerprints: {
      ...reduced.idempotencyFingerprints,
      [event.idempotencyKey]: fingerprint,
    },
  };
}

export function expiredTaskLeases(state: TaskRuntimeState, now: string): TaskLease[] {
  const nowMs = parseTimestamp(now);
  return Object.values(state.workItems)
    .flatMap((workItem) => (workItem.lease ? [workItem.lease] : []))
    .filter((lease) => Date.parse(lease.expiresAt) <= nowMs)
    .sort((left, right) => left.leaseId.localeCompare(right.leaseId));
}

export function isTaskWorkItemRunnable(workItemInput: unknown, now: string): boolean {
  const workItem = TaskWorkItemSchema.parse(workItemInput);
  if (workItem.status !== "queued" || workItem.cancellation) return false;
  return (
    !workItem.retry.nextAttemptAt || Date.parse(workItem.retry.nextAttemptAt) <= parseTimestamp(now)
  );
}

export function evaluateTaskSchedule(scheduleInput: unknown, now: string): TaskScheduleDecision {
  const schedule = TaskScheduleSchema.parse(scheduleInput);
  const nowMs = parseTimestamp(now);
  if (!schedule.enabled) {
    return TaskScheduleDecisionSchema.parse({
      scheduleId: schedule.scheduleId,
      due: false,
      disabled: true,
      now,
    });
  }

  const dueAt =
    schedule.nextDueAt ??
    (schedule.lastTriggeredAt
      ? addSeconds(schedule.lastTriggeredAt, schedule.cadence.everySeconds)
      : now);
  const due = nowMs >= Date.parse(dueAt);
  return TaskScheduleDecisionSchema.parse({
    scheduleId: schedule.scheduleId,
    due,
    disabled: false,
    now,
    dueAt,
    nextDueAt: due ? addSeconds(now, schedule.cadence.everySeconds) : dueAt,
    idempotencyKey: due ? `schedule:${schedule.scheduleId}:${dueAt}` : undefined,
  });
}

function reduceTaskRuntimeEvent(
  state: TaskRuntimeState,
  event: TaskRuntimeEvent,
): TaskRuntimeState {
  switch (event.eventType) {
    case "work_item_created":
      return applyWorkItemCreated(state, event);
    case "run_created":
      return applyRunCreated(state, event);
    case "lease_acquired":
      return applyLeaseAcquired(state, event);
    case "lease_heartbeat":
      return applyLeaseHeartbeat(state, event);
    case "lease_expired":
      return applyLeaseExpired(state, event);
    case "run_started":
      return applyRunStarted(state, event);
    case "progress_recorded":
      return applyProgressRecorded(state, event);
    case "checkpoint_recorded":
      return applyCheckpointRecorded(state, event);
    case "artifact_recorded":
      return applyArtifactRecorded(state, event);
    case "needs_human":
      return applyNeedsHuman(state, event);
    case "approval_recorded":
      return applyApprovalRecorded(state, event);
    case "session_linked":
      return applySessionLinked(state, event);
    case "session_unlinked":
      return applySessionUnlinked(state, event);
    case "cancel_requested":
      return applyCancelRequested(state, event);
    case "cancel_acknowledged":
      return applyCancelAcknowledged(state, event);
    case "retry_scheduled":
      return applyRetryScheduled(state, event);
    case "side_effect_receipt_recorded":
      return applySideEffectReceiptRecorded(state, event);
    case "run_completed":
      return applyRunCompleted(state, event);
    case "run_failed":
      return applyRunFailed(state, event);
    case "work_item_blocked":
      return applyWorkItemBlocked(state, event);
    case "work_item_superseded":
      return applyWorkItemSuperseded(state, event);
  }
}

function applyWorkItemCreated(
  state: TaskRuntimeState,
  event: z.infer<typeof WorkItemCreatedEventSchema>,
): TaskRuntimeState {
  const workItem = event.payload.workItem;
  assertEqual(event.workItemId, workItem.id, "WORK_ITEM_ID_MISMATCH", "Event and payload differ.");
  if (state.workItems[workItem.id]) {
    invariant("WORK_ITEM_EXISTS", `Work item "${workItem.id}" already exists.`);
  }
  if (workItem.lease || workItem.activeRunId || workItem.runIds.length > 0) {
    invariant("INVALID_INITIAL_WORK_ITEM", "A new work item cannot already own a run or lease.");
  }
  if (workItem.status !== "queued") {
    invariant("INVALID_INITIAL_WORK_ITEM", "A new work item must be queued.");
  }
  return { ...state, workItems: { ...state.workItems, [workItem.id]: workItem } };
}

function applyRunCreated(
  state: TaskRuntimeState,
  event: z.infer<typeof RunCreatedEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const run = event.payload.run;
  assertEqual(event.runId, run.runId, "RUN_ID_MISMATCH", "Event and payload differ.");
  assertEqual(workItem.id, run.workItemId, "WORK_ITEM_ID_MISMATCH", "Run belongs elsewhere.");
  if (state.runs[run.runId]) invariant("RUN_EXISTS", `Run "${run.runId}" already exists.`);
  if (!isTaskWorkItemRunnable(workItem, event.timestamp) || workItem.activeRunId) {
    invariant("WORK_ITEM_NOT_RUNNABLE", `Work item "${workItem.id}" is not ready for a run.`);
  }
  if (run.status !== "created") invariant("INVALID_INITIAL_RUN", "A new run must be created.");
  const expectedAttempt = workItem.retry.attemptsStarted + 1;
  if (run.attempt !== expectedAttempt || run.attempt > workItem.retry.maxAttempts) {
    invariant(
      "INVALID_RUN_ATTEMPT",
      `Run attempt ${run.attempt} must be ${expectedAttempt} and within the retry budget.`,
    );
  }
  if (
    run.executor.backend !== workItem.executor.backend ||
    run.executor.operation !== workItem.executor.operation
  ) {
    invariant("EXECUTOR_MISMATCH", "Run executor must match its work item.");
  }
  if (run.resumeFromCheckpointId) {
    const checkpoint = state.checkpoints[run.resumeFromCheckpointId];
    if (!checkpoint || checkpoint.workItemId !== workItem.id) {
      invariant("INVALID_RESUME_CHECKPOINT", "Resume checkpoint is missing or belongs elsewhere.");
    }
    if (checkpoint.environmentDigest !== run.environmentDigest) {
      invariant(
        "INVALID_RESUME_CHECKPOINT",
        "Resume checkpoint environment must match the new run environment.",
      );
    }
  }
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    runIds: unique([...workItem.runIds, run.runId]),
  });
  return {
    ...state,
    workItems: { ...state.workItems, [workItem.id]: updatedWorkItem },
    runs: { ...state.runs, [run.runId]: run },
  };
}

function applyLeaseAcquired(
  state: TaskRuntimeState,
  event: z.infer<typeof LeaseAcquiredEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const run = requireRun(state, event.runId, workItem.id);
  const lease = event.payload.lease;
  assertEqual(lease.workItemId, workItem.id, "WORK_ITEM_ID_MISMATCH", "Lease belongs elsewhere.");
  assertEqual(lease.runId, run.runId, "RUN_ID_MISMATCH", "Lease belongs to another run.");
  if (
    workItem.status !== "queued" ||
    run.status !== "created" ||
    workItem.lease ||
    workItem.activeRunId
  ) {
    invariant(
      "LEASE_NOT_ACQUIRABLE",
      "Only an unleased queued work item with a created run may lease.",
    );
  }
  if (lease.fencingToken <= workItem.lastFencingToken) {
    invariant("STALE_FENCING_TOKEN", "Lease fencing token must increase monotonically.");
  }
  if (Date.parse(event.timestamp) >= Date.parse(lease.expiresAt)) {
    invariant("LEASE_EXPIRED", "A lease must still be live when it is acquired.");
  }
  const updatedRun = TaskRunSchema.parse({
    ...run,
    status: "leased",
    leaseId: lease.leaseId,
    fencingToken: lease.fencingToken,
  });
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    status: "leased",
    lease,
    activeRunId: run.runId,
    lastFencingToken: lease.fencingToken,
    retry: { ...workItem.retry, attemptsStarted: run.attempt, nextAttemptAt: undefined },
  });
  return replaceWorkItemAndRun(state, updatedWorkItem, updatedRun);
}

function applyLeaseHeartbeat(
  state: TaskRuntimeState,
  event: z.infer<typeof LeaseHeartbeatEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const run = requireRun(state, event.runId, workItem.id);
  const lease = requireLeaseClaim(workItem, run, event.payload.claim);
  if (Date.parse(event.payload.heartbeatAt) < Date.parse(lease.heartbeatAt)) {
    invariant("STALE_HEARTBEAT", "Lease heartbeat cannot move backwards.");
  }
  if (Date.parse(event.payload.heartbeatAt) >= Date.parse(lease.expiresAt)) {
    invariant("LEASE_EXPIRED", "An expired lease cannot be renewed.");
  }
  const renewedLease = TaskLeaseSchema.parse({
    ...lease,
    heartbeatAt: event.payload.heartbeatAt,
    expiresAt: event.payload.expiresAt,
  });
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, { lease: renewedLease });
  return { ...state, workItems: { ...state.workItems, [workItem.id]: updatedWorkItem } };
}

function applyLeaseExpired(
  state: TaskRuntimeState,
  event: z.infer<typeof LeaseExpiredEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const run = requireRun(state, event.runId, workItem.id);
  const lease = requireLeaseClaim(workItem, run, event.payload.claim);
  if (Date.parse(event.payload.expiredAt) < Date.parse(lease.expiresAt)) {
    invariant("LEASE_NOT_EXPIRED", "Lease expiry event precedes the lease deadline.");
  }
  if (run.status === "cancel_requested" && run.cancellation?.status === "requested") {
    const cancellation = TaskCancellationSchema.parse({
      ...run.cancellation,
      status: "acknowledged",
      acknowledgedAt: event.payload.expiredAt,
      reason:
        event.payload.reason ??
        run.cancellation.reason ??
        "Cancellation completed on lease expiry.",
    });
    const updatedRun = TaskRunSchema.parse({
      ...run,
      status: "canceled",
      endedAt: event.payload.expiredAt,
      cancellation,
    });
    const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
      status: "canceled",
      lease: undefined,
      cancellation,
      budgetUsage: withRunWallTime(workItem, run, event.payload.expiredAt),
    });
    return replaceWorkItemAndRun(state, updatedWorkItem, updatedRun);
  }
  const uncertainEffects = run.sideEffectReceiptIds.some(
    (receiptId) => state.sideEffectReceipts[receiptId]?.status === "uncertain",
  );
  const pendingApproval = workItem.approvalIds.some(
    (approvalId) =>
      state.approvals[approvalId]?.runId === run.runId &&
      state.approvals[approvalId]?.status === "requested",
  );
  const needsHuman = uncertainEffects || pendingApproval;
  const reason =
    event.payload.reason ??
    (uncertainEffects
      ? "Lease expired with an uncertain external side effect."
      : pendingApproval
        ? "Lease expired while awaiting a persisted human decision."
        : "Lease expired before the run reached a terminal state.");
  const updatedRun = TaskRunSchema.parse({
    ...run,
    status: "interrupted",
    endedAt: event.payload.expiredAt,
    failure: {
      occurredAt: event.payload.expiredAt,
      message: reason,
      code: uncertainEffects
        ? "LEASE_EXPIRED_UNCERTAIN_EFFECT"
        : pendingApproval
          ? "LEASE_EXPIRED_AWAITING_HUMAN"
          : "LEASE_EXPIRED",
      retryable: !needsHuman,
    },
  });
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    status: needsHuman ? "needs_human" : "failed",
    lease: undefined,
    activeRunId: run.runId,
    needsHumanReason: needsHuman ? reason : undefined,
    retry: { ...workItem.retry, lastFailure: reason },
    budgetUsage: withRunWallTime(workItem, run, event.payload.expiredAt),
  });
  return replaceWorkItemAndRun(state, updatedWorkItem, updatedRun);
}

function applyRunStarted(
  state: TaskRuntimeState,
  event: z.infer<typeof RunStartedEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const run = requireRun(state, event.runId, workItem.id);
  const lease = requireLeaseClaim(workItem, run, event.payload.claim);
  if (run.status !== "leased" || workItem.status !== "leased") {
    invariant("RUN_NOT_STARTABLE", "Only a leased run may start.");
  }
  if (Date.parse(event.payload.startedAt) >= Date.parse(lease.expiresAt)) {
    invariant("LEASE_EXPIRED", "Run cannot start after lease expiry.");
  }
  const updatedRun = TaskRunSchema.parse({
    ...run,
    status: "running",
    startedAt: event.payload.startedAt,
  });
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, { status: "running" });
  return replaceWorkItemAndRun(state, updatedWorkItem, updatedRun);
}

function applyProgressRecorded(
  state: TaskRuntimeState,
  event: z.infer<typeof ProgressRecordedEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const run = requireRun(state, event.runId, workItem.id);
  requireLeaseClaim(workItem, run, event.payload.claim, event.payload.progress.recordedAt);
  requireActiveRun(run, ["running", "cancel_requested"]);
  if (run.lastProgress && event.payload.progress.sequence <= run.lastProgress.sequence) {
    invariant("STALE_PROGRESS", "Progress sequence must increase.");
  }
  if (
    workItem.budget?.maxProgressEvents !== undefined &&
    workItem.budgetUsage.progressEvents + 1 > workItem.budget.maxProgressEvents
  ) {
    invariant("BUDGET_EXCEEDED", "Progress event budget is exhausted.");
  }
  const updatedRun = TaskRunSchema.parse({ ...run, lastProgress: event.payload.progress });
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    budgetUsage: {
      ...workItem.budgetUsage,
      progressEvents: workItem.budgetUsage.progressEvents + 1,
    },
  });
  return replaceWorkItemAndRun(state, updatedWorkItem, updatedRun);
}

function applyCheckpointRecorded(
  state: TaskRuntimeState,
  event: z.infer<typeof CheckpointRecordedEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const run = requireRun(state, event.runId, workItem.id);
  requireLeaseClaim(workItem, run, event.payload.claim, event.payload.checkpoint.createdAt);
  requireActiveRun(run, ["running", "cancel_requested"]);
  const checkpoint = event.payload.checkpoint;
  assertEqual(
    checkpoint.workItemId,
    workItem.id,
    "WORK_ITEM_ID_MISMATCH",
    "Checkpoint belongs elsewhere.",
  );
  assertEqual(checkpoint.runId, run.runId, "RUN_ID_MISMATCH", "Checkpoint belongs to another run.");
  if (checkpoint.environmentDigest !== run.environmentDigest) {
    invariant(
      "CHECKPOINT_ENVIRONMENT_MISMATCH",
      "Checkpoint environment must match its authoritative run.",
    );
  }
  if (state.checkpoints[checkpoint.checkpointId]) {
    invariant("CHECKPOINT_EXISTS", `Checkpoint "${checkpoint.checkpointId}" already exists.`);
  }
  const previousInRun = run.latestCheckpointId
    ? state.checkpoints[run.latestCheckpointId]
    : undefined;
  const expectedParentId = previousInRun?.checkpointId ?? run.resumeFromCheckpointId;
  if (previousInRun && checkpoint.sequence <= previousInRun.sequence) {
    invariant("STALE_CHECKPOINT", "Checkpoint sequence must increase.");
  }
  if (checkpoint.parentCheckpointId !== expectedParentId) {
    invariant("CHECKPOINT_PARENT_MISMATCH", "Checkpoint parent must be the prior run checkpoint.");
  }
  for (const artifactId of checkpoint.artifactIds) {
    const artifact = state.artifacts[artifactId];
    if (!artifact || artifact.workItemId !== workItem.id) {
      invariant(
        "UNKNOWN_ARTIFACT",
        `Checkpoint artifact "${artifactId}" is missing or belongs elsewhere.`,
      );
    }
  }
  if (
    workItem.budget?.maxCheckpoints !== undefined &&
    workItem.budgetUsage.checkpoints + 1 > workItem.budget.maxCheckpoints
  ) {
    invariant("BUDGET_EXCEEDED", "Checkpoint budget is exhausted.");
  }
  const updatedRun = TaskRunSchema.parse({ ...run, latestCheckpointId: checkpoint.checkpointId });
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    latestCheckpointId: checkpoint.checkpointId,
    budgetUsage: {
      ...workItem.budgetUsage,
      checkpoints: workItem.budgetUsage.checkpoints + 1,
    },
  });
  return {
    ...replaceWorkItemAndRun(state, updatedWorkItem, updatedRun),
    checkpoints: { ...state.checkpoints, [checkpoint.checkpointId]: checkpoint },
  };
}

function applyArtifactRecorded(
  state: TaskRuntimeState,
  event: z.infer<typeof ArtifactRecordedEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const run = requireRun(state, event.runId, workItem.id);
  requireLeaseClaim(workItem, run, event.payload.claim, event.payload.artifact.createdAt);
  requireActiveRun(run, ["running", "cancel_requested"]);
  const artifact = event.payload.artifact;
  assertEqual(
    artifact.workItemId,
    workItem.id,
    "WORK_ITEM_ID_MISMATCH",
    "Artifact belongs elsewhere.",
  );
  if (artifact.runId && artifact.runId !== run.runId) {
    invariant("RUN_ID_MISMATCH", "Artifact belongs to another run.");
  }
  if (state.artifacts[artifact.artifactId]) {
    invariant("ARTIFACT_EXISTS", `Artifact "${artifact.artifactId}" already exists.`);
  }
  if (
    workItem.budget?.maxArtifactBytes !== undefined &&
    workItem.budgetUsage.artifactBytes + (artifact.sizeBytes ?? 0) >
      workItem.budget.maxArtifactBytes
  ) {
    invariant("BUDGET_EXCEEDED", "Artifact byte budget is exhausted.");
  }
  const updatedRun = TaskRunSchema.parse({
    ...run,
    artifactIds: unique([...run.artifactIds, artifact.artifactId]),
  });
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    artifactIds: unique([...workItem.artifactIds, artifact.artifactId]),
    budgetUsage: {
      ...workItem.budgetUsage,
      artifactBytes: workItem.budgetUsage.artifactBytes + (artifact.sizeBytes ?? 0),
    },
  });
  return {
    ...replaceWorkItemAndRun(state, updatedWorkItem, updatedRun),
    artifacts: { ...state.artifacts, [artifact.artifactId]: artifact },
  };
}

function applyNeedsHuman(
  state: TaskRuntimeState,
  event: z.infer<typeof NeedsHumanEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const run = requireRun(state, event.runId, workItem.id);
  requireLeaseClaim(workItem, run, event.payload.claim, event.payload.requestedAt);
  requireActiveRun(run, ["running", "cancel_requested"]);
  if (event.payload.approvalId && !state.approvals[event.payload.approvalId]) {
    invariant("UNKNOWN_APPROVAL", `Approval "${event.payload.approvalId}" does not exist.`);
  }
  const updatedRun = TaskRunSchema.parse({
    ...run,
    status: "needs_human",
    endedAt: event.payload.requestedAt,
  });
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    status: "needs_human",
    lease: undefined,
    needsHumanReason: event.payload.reason,
    budgetUsage: withRunWallTime(workItem, run, event.payload.requestedAt),
  });
  return replaceWorkItemAndRun(state, updatedWorkItem, updatedRun);
}

function applyApprovalRecorded(
  state: TaskRuntimeState,
  event: z.infer<typeof ApprovalRecordedEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const approval = event.payload.approval;
  assertEqual(
    approval.workItemId,
    workItem.id,
    "WORK_ITEM_ID_MISMATCH",
    "Approval belongs elsewhere.",
  );
  if (approval.runId) requireRun(state, approval.runId, workItem.id);
  const existing = state.approvals[approval.approvalId];
  if (existing) {
    if (existing.status !== "requested" || approval.status === "requested") {
      invariant(
        "INVALID_APPROVAL_UPDATE",
        "Approval decisions are immutable after being recorded.",
      );
    }
    if (
      existing.workItemId !== approval.workItemId ||
      existing.runId !== approval.runId ||
      existing.kind !== approval.kind ||
      existing.requestedAt !== approval.requestedAt ||
      existing.requestedBy !== approval.requestedBy ||
      existing.requestRef !== approval.requestRef
    ) {
      invariant("APPROVAL_ID_COLLISION", "Approval identity fields cannot change.");
    }
  }
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    approvalIds: unique([...workItem.approvalIds, approval.approvalId]),
  });
  return {
    ...state,
    workItems: { ...state.workItems, [workItem.id]: updatedWorkItem },
    approvals: { ...state.approvals, [approval.approvalId]: approval },
  };
}

function applySessionLinked(
  state: TaskRuntimeState,
  event: z.infer<typeof SessionLinkedEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const link = event.payload.link;
  assertEqual(
    link.workItemId,
    workItem.id,
    "WORK_ITEM_ID_MISMATCH",
    "Session link belongs elsewhere.",
  );
  if (link.unlinkedAt) invariant("INVALID_SESSION_LINK", "A new session link must be active.");
  if (state.sessionLinks[link.linkId]) {
    invariant("SESSION_LINK_EXISTS", `Session link "${link.linkId}" already exists.`);
  }
  const duplicate = Object.values(state.sessionLinks).find(
    (candidate) =>
      candidate.workItemId === link.workItemId &&
      candidate.sessionId === link.sessionId &&
      !candidate.unlinkedAt,
  );
  if (duplicate) invariant("SESSION_ALREADY_LINKED", "Session already observes this work item.");
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    sessionLinkIds: unique([...workItem.sessionLinkIds, link.linkId]),
  });
  return {
    ...state,
    workItems: { ...state.workItems, [workItem.id]: updatedWorkItem },
    sessionLinks: { ...state.sessionLinks, [link.linkId]: link },
  };
}

function applySessionUnlinked(
  state: TaskRuntimeState,
  event: z.infer<typeof SessionUnlinkedEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const link = state.sessionLinks[event.payload.linkId];
  if (!link || link.workItemId !== workItem.id || link.unlinkedAt) {
    invariant("UNKNOWN_SESSION_LINK", "Active session link does not exist.");
  }
  const updatedLink = TaskSessionLinkSchema.parse({
    ...link,
    unlinkedAt: event.payload.unlinkedAt,
  });
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    sessionLinkIds: workItem.sessionLinkIds.filter((linkId) => linkId !== link.linkId),
  });
  return {
    ...state,
    workItems: { ...state.workItems, [workItem.id]: updatedWorkItem },
    sessionLinks: { ...state.sessionLinks, [link.linkId]: updatedLink },
  };
}

function applyCancelRequested(
  state: TaskRuntimeState,
  event: z.infer<typeof CancelRequestedEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const run = requireRun(state, event.runId, workItem.id);
  if (workItem.activeRunId !== run.runId || !["leased", "running"].includes(run.status)) {
    invariant("RUN_NOT_CANCELABLE", "Only the active leased or running run can be canceled.");
  }
  const cancellation = TaskCancellationSchema.parse({
    status: "requested",
    requestedAt: event.payload.requestedAt,
    reason: event.payload.reason,
  });
  const updatedRun = TaskRunSchema.parse({ ...run, status: "cancel_requested", cancellation });
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, { cancellation });
  return replaceWorkItemAndRun(state, updatedWorkItem, updatedRun);
}

function applyCancelAcknowledged(
  state: TaskRuntimeState,
  event: z.infer<typeof CancelAcknowledgedEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const run = requireRun(state, event.runId, workItem.id);
  requireLeaseClaim(workItem, run, event.payload.claim, event.payload.acknowledgedAt);
  if (run.status !== "cancel_requested" || !run.cancellation) {
    invariant("CANCEL_NOT_REQUESTED", "Cancellation must be requested before acknowledgement.");
  }
  const cancellation = TaskCancellationSchema.parse({
    ...run.cancellation,
    status: "acknowledged",
    acknowledgedAt: event.payload.acknowledgedAt,
    reason: event.payload.reason ?? run.cancellation.reason,
  });
  const updatedRun = TaskRunSchema.parse({
    ...run,
    status: "canceled",
    endedAt: event.payload.acknowledgedAt,
    cancellation,
  });
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    status: "canceled",
    lease: undefined,
    cancellation,
    budgetUsage: withRunWallTime(workItem, run, event.payload.acknowledgedAt),
  });
  return replaceWorkItemAndRun(state, updatedWorkItem, updatedRun);
}

function applyRetryScheduled(
  state: TaskRuntimeState,
  event: z.infer<typeof RetryScheduledEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  if (!["failed", "blocked", "needs_human"].includes(workItem.status)) {
    invariant("RETRY_NOT_SCHEDULABLE", "Only failed, blocked, or human-gated work may retry.");
  }
  if (workItem.cancellation) {
    invariant("CANCEL_REQUESTED", "Canceled work cannot be scheduled for retry.");
  }
  if (workItem.retry.attemptsStarted >= workItem.retry.maxAttempts) {
    invariant("RETRY_BUDGET_EXHAUSTED", "Retry attempt budget is exhausted.");
  }
  if (Date.parse(event.payload.nextAttemptAt) < Date.parse(event.payload.scheduledAt)) {
    invariant("INVALID_RETRY_SCHEDULE", "Retry cannot be due before it is scheduled.");
  }
  if (workItem.status === "needs_human") {
    const approvalId = event.payload.approvalId;
    const approval = approvalId ? state.approvals[approvalId] : undefined;
    if (!approval || !["approved", "waived"].includes(approval.status)) {
      invariant("APPROVAL_REQUIRED", "Human-gated work requires an approved or waived decision.");
    }
  }
  const priorRun = workItem.activeRunId ? state.runs[workItem.activeRunId] : undefined;
  if (priorRun?.failure && !priorRun.failure.retryable) {
    const approvalId = event.payload.approvalId;
    const approval = approvalId ? state.approvals[approvalId] : undefined;
    if (!approval || !["approved", "waived"].includes(approval.status)) {
      invariant(
        "APPROVAL_REQUIRED",
        "A non-retryable failure requires an approved or waived decision.",
      );
    }
  }
  if (event.payload.resumeFromCheckpointId) {
    const checkpoint = state.checkpoints[event.payload.resumeFromCheckpointId];
    if (!checkpoint || checkpoint.workItemId !== workItem.id) {
      invariant("INVALID_RESUME_CHECKPOINT", "Retry checkpoint is missing or belongs elsewhere.");
    }
  }
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    status: "queued",
    activeRunId: undefined,
    lease: undefined,
    cancellation: undefined,
    blockedReason: undefined,
    needsHumanReason: undefined,
    retry: {
      ...workItem.retry,
      nextAttemptAt: event.payload.nextAttemptAt,
      lastFailure: event.payload.reason,
    },
    latestCheckpointId: event.payload.resumeFromCheckpointId ?? workItem.latestCheckpointId,
  });
  return { ...state, workItems: { ...state.workItems, [workItem.id]: updatedWorkItem } };
}

function applySideEffectReceiptRecorded(
  state: TaskRuntimeState,
  event: z.infer<typeof SideEffectReceiptRecordedEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const run = requireRun(state, event.runId, workItem.id);
  requireLeaseClaim(workItem, run, event.payload.claim, event.payload.receipt.recordedAt);
  requireActiveRun(run, ["running", "cancel_requested"]);
  const receipt = event.payload.receipt;
  assertEqual(
    receipt.workItemId,
    workItem.id,
    "WORK_ITEM_ID_MISMATCH",
    "Receipt belongs elsewhere.",
  );
  assertEqual(receipt.runId, run.runId, "RUN_ID_MISMATCH", "Receipt belongs to another run.");
  const duplicateEffect = Object.values(state.sideEffectReceipts).find(
    (candidate) =>
      candidate.effectIdempotencyKey === receipt.effectIdempotencyKey &&
      candidate.receiptId !== receipt.receiptId,
  );
  if (duplicateEffect) {
    invariant(
      "SIDE_EFFECT_RECEIPT_COLLISION",
      "One side-effect key cannot have multiple receipts.",
    );
  }
  const existing = state.sideEffectReceipts[receipt.receiptId];
  if (existing) {
    if (
      existing.workItemId !== receipt.workItemId ||
      existing.runId !== receipt.runId ||
      existing.effectKind !== receipt.effectKind ||
      existing.effectIdempotencyKey !== receipt.effectIdempotencyKey
    ) {
      invariant("SIDE_EFFECT_RECEIPT_COLLISION", "Receipt identity fields cannot change.");
    }
    if (existing.status === "committed" && receipt.status !== "committed") {
      invariant(
        "SIDE_EFFECT_STATUS_REGRESSION",
        "Committed side effects cannot become uncertain or not committed.",
      );
    }
  }
  const updatedRun = TaskRunSchema.parse({
    ...run,
    sideEffectReceiptIds: unique([...run.sideEffectReceiptIds, receipt.receiptId]),
  });
  const capabilityCallCount = workItem.budgetUsage.capabilityCalls[receipt.effectKind] ?? 0;
  const capabilityCallLimit = workItem.budget?.capabilityCalls?.[receipt.effectKind];
  if (
    !existing &&
    capabilityCallLimit !== undefined &&
    capabilityCallCount + 1 > capabilityCallLimit
  ) {
    invariant(
      "BUDGET_EXCEEDED",
      `Capability call budget for "${receipt.effectKind}" is exhausted.`,
    );
  }
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    sideEffectReceiptIds: unique([...workItem.sideEffectReceiptIds, receipt.receiptId]),
    budgetUsage: existing
      ? workItem.budgetUsage
      : {
          ...workItem.budgetUsage,
          capabilityCalls: {
            ...workItem.budgetUsage.capabilityCalls,
            [receipt.effectKind]: capabilityCallCount + 1,
          },
        },
  });
  return {
    ...replaceWorkItemAndRun(state, updatedWorkItem, updatedRun),
    sideEffectReceipts: { ...state.sideEffectReceipts, [receipt.receiptId]: receipt },
  };
}

function applyRunCompleted(
  state: TaskRuntimeState,
  event: z.infer<typeof RunCompletedEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const run = requireRun(state, event.runId, workItem.id);
  requireLeaseClaim(workItem, run, event.payload.claim, event.payload.completedAt);
  if (run.status !== "running")
    invariant("RUN_NOT_COMPLETABLE", "Only a running run may complete.");
  if (hasUncertainSideEffects(state, run)) {
    invariant(
      "UNCERTAIN_SIDE_EFFECT",
      "A run with an uncertain external side effect cannot complete.",
    );
  }
  const updatedRun = TaskRunSchema.parse({
    ...run,
    status: "succeeded",
    endedAt: event.payload.completedAt,
    resultRef: event.payload.resultRef,
  });
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    status: "succeeded",
    lease: undefined,
    budgetUsage: withRunWallTime(workItem, run, event.payload.completedAt),
  });
  return replaceWorkItemAndRun(state, updatedWorkItem, updatedRun);
}

function applyRunFailed(
  state: TaskRuntimeState,
  event: z.infer<typeof RunFailedEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  const run = requireRun(state, event.runId, workItem.id);
  requireLeaseClaim(workItem, run, event.payload.claim, event.payload.failure.occurredAt);
  requireActiveRun(run, ["running", "cancel_requested"]);
  const uncertainEffects = hasUncertainSideEffects(state, run);
  const failure = TaskFailureSchema.parse({
    ...event.payload.failure,
    retryable: uncertainEffects ? false : event.payload.failure.retryable,
  });
  const updatedRun = TaskRunSchema.parse({
    ...run,
    status: "failed",
    endedAt: failure.occurredAt,
    failure,
  });
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    status: uncertainEffects ? "needs_human" : "failed",
    lease: undefined,
    needsHumanReason: uncertainEffects
      ? "Run failed with an uncertain external side effect."
      : undefined,
    retry: { ...workItem.retry, lastFailure: failure.message },
    budgetUsage: withRunWallTime(workItem, run, failure.occurredAt),
  });
  return replaceWorkItemAndRun(state, updatedWorkItem, updatedRun);
}

function applyWorkItemBlocked(
  state: TaskRuntimeState,
  event: z.infer<typeof WorkItemBlockedEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  if (workItem.lease) {
    invariant("ACTIVE_WORK_ITEM", "Active work must stop before it is blocked.");
  }
  if (["succeeded", "canceled", "superseded"].includes(workItem.status)) {
    invariant("TERMINAL_WORK_ITEM", "A terminal work item cannot be blocked.");
  }
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    status: "blocked",
    lease: undefined,
    blockedReason: event.payload.reason,
  });
  return { ...state, workItems: { ...state.workItems, [workItem.id]: updatedWorkItem } };
}

function applyWorkItemSuperseded(
  state: TaskRuntimeState,
  event: z.infer<typeof WorkItemSupersededEventSchema>,
): TaskRuntimeState {
  const workItem = requireWorkItem(state, event.workItemId);
  if (["leased", "running"].includes(workItem.status) || workItem.lease) {
    invariant("ACTIVE_WORK_ITEM", "Active work must be canceled before it is superseded.");
  }
  if (event.payload.supersededBy && !state.workItems[event.payload.supersededBy]) {
    invariant("UNKNOWN_SUPERSEDING_WORK_ITEM", "Superseding work item does not exist.");
  }
  const updatedWorkItem = updateWorkItem(workItem, event.timestamp, {
    status: "superseded",
    supersededBy: event.payload.supersededBy,
  });
  return { ...state, workItems: { ...state.workItems, [workItem.id]: updatedWorkItem } };
}

function requireWorkItem(state: TaskRuntimeState, workItemId: string): TaskWorkItem {
  const workItem = state.workItems[workItemId];
  if (!workItem) invariant("UNKNOWN_WORK_ITEM", `Unknown work item "${workItemId}".`);
  return workItem;
}

function requireRun(state: TaskRuntimeState, runId: string, workItemId: string): TaskRun {
  const run = state.runs[runId];
  if (!run) invariant("UNKNOWN_RUN", `Unknown run "${runId}".`);
  if (run.workItemId !== workItemId)
    invariant("RUN_ID_MISMATCH", "Run belongs to another work item.");
  return run;
}

function requireLeaseClaim(
  workItem: TaskWorkItem,
  run: TaskRun,
  claim: TaskLeaseClaim,
  occurredAt?: string,
): TaskLease {
  const lease = workItem.lease;
  if (
    !lease ||
    lease.runId !== run.runId ||
    lease.leaseId !== claim.leaseId ||
    lease.fencingToken !== claim.fencingToken ||
    run.leaseId !== claim.leaseId ||
    run.fencingToken !== claim.fencingToken
  ) {
    invariant("STALE_LEASE_CLAIM", "Worker event does not hold the active fenced lease.");
  }
  if (occurredAt && Date.parse(occurredAt) >= Date.parse(lease.expiresAt)) {
    invariant("LEASE_EXPIRED", "Worker event occurred after lease expiry.");
  }
  return lease;
}

function requireActiveRun(run: TaskRun, statuses: TaskRunStatus[]): void {
  if (!statuses.includes(run.status)) {
    invariant("RUN_NOT_ACTIVE", `Run "${run.runId}" is ${run.status}.`);
  }
}

function hasUncertainSideEffects(state: TaskRuntimeState, run: TaskRun): boolean {
  return run.sideEffectReceiptIds.some(
    (receiptId) => state.sideEffectReceipts[receiptId]?.status === "uncertain",
  );
}

function updateWorkItem(
  workItem: TaskWorkItem,
  timestamp: string,
  patch: Partial<TaskWorkItem>,
): TaskWorkItem {
  return TaskWorkItemSchema.parse({
    ...workItem,
    ...patch,
    id: workItem.id,
    updatedAt: timestamp,
    revision: workItem.revision + 1,
  });
}

function withRunWallTime(workItem: TaskWorkItem, run: TaskRun, endedAt: string): TaskBudgetUsage {
  if (!run.startedAt) return workItem.budgetUsage;
  const elapsed = Math.max(0, Date.parse(endedAt) - Date.parse(run.startedAt));
  return {
    ...workItem.budgetUsage,
    wallTimeMs: workItem.budgetUsage.wallTimeMs + elapsed,
  };
}

function replaceWorkItemAndRun(
  state: TaskRuntimeState,
  workItem: TaskWorkItem,
  run: TaskRun,
): TaskRuntimeState {
  return {
    ...state,
    workItems: { ...state.workItems, [workItem.id]: workItem },
    runs: { ...state.runs, [run.runId]: run },
  };
}

function runtimeEventFingerprint(event: TaskRuntimeEvent): string {
  const { eventId: _eventId, ...content } = event;
  return stableJson(content);
}

function deterministicId(prefix: string, value: unknown): string {
  return `${prefix}${portableHash(stableJson(value)).slice(0, 20)}`;
}

function portableHash(value: string): string {
  const hashes = [0x811c9dc5, 0x9e3779b1, 0x85ebca77, 0xc2b2ae3d];
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    hashes[0] = Math.imul((hashes[0] ?? 0) ^ code, 0x01000193);
    hashes[1] = Math.imul((hashes[1] ?? 0) ^ code, 0x27d4eb2d);
    hashes[2] = Math.imul((hashes[2] ?? 0) ^ code, 0x165667b1);
    hashes[3] = Math.imul((hashes[3] ?? 0) ^ code, 0x85ebca6b);
  }
  return hashes.map((hash) => (hash >>> 0).toString(16).padStart(8, "0")).join("");
}

function parseTimestamp(timestamp: string): number {
  TimestampSchema.parse(timestamp);
  return Date.parse(timestamp);
}

function addSeconds(timestamp: string, seconds: number): string {
  return new Date(parseTimestamp(timestamp) + seconds * 1_000).toISOString();
}

function unique(values: string[]): string[] {
  return Array.from(new Set(values));
}

function assertEqual(actual: string, expected: string, code: string, message: string): void {
  if (actual !== expected) invariant(code, message);
}

function invariant(code: string, message: string): never {
  throw new TaskRuntimeInvariantError(code, message);
}
