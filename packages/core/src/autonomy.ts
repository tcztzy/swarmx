import { createHash } from "node:crypto";
import { z } from "zod";

const ALLOWED_SECRET_REFERENCE_KEYS = new Set([
  "secretref",
  "secret_ref",
  "secretrefid",
  "secret_ref_id",
  "secretstatus",
  "secret_status",
  "credentialref",
  "credential_ref",
  "credentialrefs",
  "credential_refs",
  "credentialreferences",
  "credential_references",
]);

const FORBIDDEN_SECRET_KEY_PATTERN =
  /(api[_-]?key|access[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key)/i;
const FORBIDDEN_RUNTIME_RAW_KEY_PATTERN =
  /(raw[_-]?(issue|terminal|validator|analysis|data|prompt|response|model|tool)|issue[_-]?body|terminal[_-]?output|std(out|err)|validator[_-]?output|analysis[_-]?output|data[_-]?output)/i;

const idWithPrefix = (prefix: string) =>
  z.string().regex(new RegExp(`^${prefix}[A-Za-z0-9][A-Za-z0-9_-]*$`), `Must use ${prefix} prefix`);

export const AutonomyWorkClassSchema = z.enum(["project_iteration", "analysis_execution"]);
export const AutonomyLevelSchema = z.enum(["A0", "A1", "A2", "A3", "A4"]);
export const AutonomyWorkItemStatusSchema = z.enum([
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
export const AutonomyTriggerTypeSchema = z.enum([
  "schedule_tick",
  "manual_request",
  "issue_created",
  "bug_reported",
  "validation_failed",
  "report_feedback_received",
  "analysis_finding_created",
  "file_changed",
  "dependency_update_available",
]);
export const EngineeringLifecyclePrimaryStateSchema = z.enum([
  "intake",
  "triage",
  "proposal",
  "discussion",
  "specification",
  "implementation",
  "validation",
  "report",
  "close",
]);
export const EngineeringLifecycleSideStateSchema = z.enum([
  "needs_human",
  "blocked",
  "rejected",
  "deferred",
  "superseded",
]);
export const EngineeringLifecycleStateSchema = z.enum([
  ...EngineeringLifecyclePrimaryStateSchema.options,
  ...EngineeringLifecycleSideStateSchema.options,
]);
export const EngineeringIntakeSourceTypeSchema = z.enum([
  "new_requirement",
  "bug_report",
  "failing_validation",
  "report_feedback",
  "scheduled_review",
  "analysis_finding",
  "manual_request",
]);
export const EngineeringProposalStatusSchema = z.enum([
  "draft",
  "discussion",
  "accepted",
  "rejected",
  "superseded",
]);
export const EngineeringApprovalStatusSchema = z.enum([
  "requested",
  "approved",
  "rejected",
  "waived",
]);
export const AutonomyAgentRunStatusSchema = z.enum([
  "planned",
  "running",
  "succeeded",
  "failed",
  "skipped",
  "needs_human",
  "canceled",
]);
export const AutonomyWorkflowDecisionKindSchema = z.enum([
  "continue",
  "dispatch",
  "close",
  "block",
  "needs_human",
  "defer",
  "reject",
]);
export const AutonomyWorkflowDecisionStatusSchema = z.enum(["accepted", "rejected", "needs_human"]);
export const AutonomyRuntimeEventTypeSchema = z.enum([
  ...AutonomyTriggerTypeSchema.options,
  "trigger_rejected",
  "work_item_created",
  "work_item_updated",
  "deterministic_transition",
  "lease_acquired",
  "lease_renewed",
  "lease_expired",
  "lease_released",
  "command_dag_selected",
  "command_started",
  "command_succeeded",
  "command_failed",
  "validator_started",
  "validator_passed",
  "validator_failed",
  "validator_skipped",
  "validator_waived",
  "evidence_packet_created",
  "agent_run_written",
  "workflow_decision_recorded",
  "report_created",
  "artifact_created",
  "feedback_received",
  "state_transition_rejected",
]);
export const ValidatorStatusSchema = z.enum(["passed", "failed", "skipped", "waived"]);
export const ArtifactKindSchema = z.enum([
  "raw_input",
  "derived_intermediate",
  "generated_figure",
  "generated_table",
  "summary_report",
  "evidence_packet",
  "report",
  "artifact",
  "other",
]);
export const AutonomySchedulerEventTypeSchema = z.enum([
  "work_item_created",
  "work_item_leased",
  "work_item_started",
  "work_item_completed",
  "work_item_failed",
  "work_item_blocked",
  "work_item_canceled",
  "work_item_superseded",
  "discovery_completed",
  "proposal_written",
  "agent_run_written",
  "workflow_decision_recorded",
  "report_created",
  "artifact_created",
  "feedback_recorded",
]);
export const AutonomyFeedbackActionSchema = z.enum([
  "accept",
  "reject",
  "redirect",
  "approve",
  "stop",
  "remember",
  "spec",
  "paper",
  "benchmark",
]);
export const AutonomyFeedbackRouteSchema = z.enum([
  "none",
  "work_item",
  "memory",
  "spec",
  "paper",
  "benchmark",
]);
export const AutonomyDashboardKindSchema = z.enum(["report_artifact", "live_local", "server_live"]);
export const AutonomyWakeupRoleSchema = z.enum(["app", "server"]);
export const AutonomyWakeupAdapterSchema = z.enum([
  "desktop_timer",
  "server_timer",
  "launchd",
  "systemd",
  "cron",
  "codex_automation",
]);
export const AutonomyDaemonRunStatusSchema = z.enum([
  "queued",
  "leased",
  "running",
  "needs_human",
  "failed",
  "succeeded",
  "canceled",
]);
export const AutonomyCircuitBreakerStatusSchema = z.enum(["allow", "needs_human", "blocked"]);

export const DEFAULT_REPORT_CADENCE_HOURS = 24;
export const DEFAULT_DASHBOARD_REFRESH_SECONDS = 60;
export const DEFAULT_CIRCUIT_BREAKER_FAILURES = 3;

export const AutonomySourceRefSchema = z
  .object({
    kind: z.string().min(1),
    id: z.string().min(1),
    url: z.string().optional(),
    title: z.string().optional(),
  })
  .passthrough();

export const AutonomyTriggerRecordSchema = z
  .object({
    triggerId: idWithPrefix("trg_"),
    triggerType: AutonomyTriggerTypeSchema,
    timestamp: z.string().min(1),
    source: z.string().min(1),
    idempotencyKey: z.string().min(1),
    scheduleId: z.string().min(1).optional(),
    workItemId: idWithPrefix("awi_").optional(),
    runId: idWithPrefix("run_").optional(),
    feedbackId: idWithPrefix("fbk_").optional(),
    sourceRefs: z.array(AutonomySourceRefSchema).default([]),
    payload: z.record(z.string(), z.unknown()).default({}),
  })
  .passthrough()
  .superRefine(addRuntimeRecordIssues);

export const EngineeringApprovalRecordSchema = z
  .object({
    approvalId: idWithPrefix("apr_"),
    status: EngineeringApprovalStatusSchema,
    actor: z.string().min(1).optional(),
    decidedAt: z.string().min(1).optional(),
    reason: z.string().min(1).optional(),
    sourceRefs: z.array(AutonomySourceRefSchema).default([]),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const EngineeringIntakeRecordSchema = z
  .object({
    intakeId: idWithPrefix("int_"),
    sourceId: z.string().min(1),
    sourceType: EngineeringIntakeSourceTypeSchema,
    receivedAt: z.string().min(1),
    title: z.string().min(1),
    summary: z.string().min(1),
    requester: z.string().min(1).optional(),
    sourceRefs: z.array(AutonomySourceRefSchema).default([]),
    reportId: idWithPrefix("rpt_").optional(),
    validationId: z.string().min(1).optional(),
    autonomyLevel: AutonomyLevelSchema,
    initialState: EngineeringLifecycleStateSchema.default("intake"),
    workItemId: idWithPrefix("awi_").optional(),
    risk: z.enum(["low", "medium", "high", "safety_review"]).default("medium"),
    metadata: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const EngineeringProposalRecordSchema = z
  .object({
    proposalId: idWithPrefix("prp_"),
    workItemId: idWithPrefix("awi_").optional(),
    status: EngineeringProposalStatusSchema.default("draft"),
    problem: z.string().min(1),
    affectedSurfaces: z.array(z.string().min(1)).default([]),
    alternatives: z.array(z.string().min(1)).default([]),
    compatibility: z.string().min(1).optional(),
    securityPrivacy: z.string().min(1).optional(),
    validationPlan: z.array(z.string().min(1)).min(1),
    migrationRollback: z.string().min(1).optional(),
    approvals: z.array(EngineeringApprovalRecordSchema).default([]),
    sourceRefs: z.array(AutonomySourceRefSchema).default([]),
    createdAt: z.string().min(1).optional(),
    updatedAt: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AutonomyTransitionDecisionStatusSchema = z.enum(["allowed", "rejected"]);
export const AutonomyTransitionPreconditionKindSchema = z.enum([
  "current_state",
  "status_edge",
  "approval",
  "validator",
  "budget",
  "autonomy_policy",
  "lease",
  "idempotency",
  "custom",
]);
export const AutonomyTransitionPreconditionStatusSchema = z.enum([
  "passed",
  "failed",
  "missing",
  "unknown",
  "waived",
]);
export const AutonomyTransitionIdempotencyStatusSchema = z.enum(["new", "duplicate", "unknown"]);

export const AutonomyTransitionPreconditionSchema = z
  .object({
    kind: AutonomyTransitionPreconditionKindSchema,
    id: z.string().min(1).optional(),
    status: AutonomyTransitionPreconditionStatusSchema.default("passed"),
    required: z.boolean().default(true),
    message: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addRuntimeRecordIssues);

export const AutonomyTransitionDecisionSchema = z
  .object({
    decisionId: idWithPrefix("dec_"),
    workItemId: idWithPrefix("awi_"),
    runId: idWithPrefix("run_").optional(),
    from: AutonomyWorkItemStatusSchema,
    to: AutonomyWorkItemStatusSchema,
    status: AutonomyTransitionDecisionStatusSchema,
    idempotencyKey: z.string().min(1),
    idempotencyStatus: AutonomyTransitionIdempotencyStatusSchema,
    preconditions: z.array(AutonomyTransitionPreconditionSchema).default([]),
    missingRequirements: z.array(z.string().min(1)).default([]),
    reason: z.string().min(1),
    patch: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough()
  .superRefine(addRuntimeRecordIssues);

export const EngineeringLifecycleTransitionDecisionSchema = z
  .object({
    decisionId: idWithPrefix("dec_"),
    workItemId: idWithPrefix("awi_").optional(),
    from: EngineeringLifecycleStateSchema,
    to: EngineeringLifecycleStateSchema,
    status: AutonomyTransitionDecisionStatusSchema,
    currentStateMatched: z.boolean(),
    validEdge: z.boolean(),
    requiredEvidenceIds: z.array(z.string().min(1)).default([]),
    presentEvidenceIds: z.array(z.string().min(1)).default([]),
    missingEvidenceIds: z.array(z.string().min(1)).default([]),
    requiredApprovalIds: z.array(idWithPrefix("apr_")).default([]),
    presentApprovalIds: z.array(idWithPrefix("apr_")).default([]),
    missingApprovalIds: z.array(idWithPrefix("apr_")).default([]),
    validatorGate: z.lazy(() => ValidatorGateDecisionSchema).optional(),
    reason: z.string().min(1),
    nextWorkflow: z.lazy(() => AutonomyWorkflowStateSchema).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AutonomyAgentRunRecordSchema = z
  .object({
    agentRunId: idWithPrefix("agt_"),
    workItemId: idWithPrefix("awi_"),
    runId: idWithPrefix("run_").optional(),
    workflowKind: z.string().min(1),
    stage: z.string().min(1),
    role: z.string().min(1),
    status: AutonomyAgentRunStatusSchema,
    harnessId: z.string().min(1),
    modelId: z.string().min(1),
    modelSupplyId: z.string().min(1).optional(),
    adapter: z.string().min(1).optional(),
    agentProfileId: z.string().min(1).optional(),
    startedAt: z.string().min(1).optional(),
    endedAt: z.string().min(1).optional(),
    durationMs: z.number().int().nonnegative().optional(),
    inputRefs: z.array(AutonomySourceRefSchema).default([]),
    outputRefs: z.array(AutonomySourceRefSchema).default([]),
    artifactIds: z.array(idWithPrefix("art_")).default([]),
    evidenceIds: z.array(idWithPrefix("evp_")).default([]),
    summary: z.string().min(1).optional(),
    resultRef: z.string().min(1).optional(),
    errorRef: z.string().min(1).optional(),
    failureSummary: z.string().min(1).optional(),
    metadata: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough()
  .superRefine(addAgentRunIdentityIssues);

export const AutonomyWorkflowDecisionRecordSchema = z
  .object({
    decisionId: idWithPrefix("dec_"),
    workItemId: idWithPrefix("awi_"),
    runId: idWithPrefix("run_").optional(),
    workflowKind: z.string().min(1),
    currentStage: z.string().min(1),
    nextStage: z.string().min(1).optional(),
    status: AutonomyWorkflowDecisionStatusSchema,
    decisionKind: AutonomyWorkflowDecisionKindSchema,
    agentRunIds: z.array(idWithPrefix("agt_")).default([]),
    evidenceIds: z.array(idWithPrefix("evp_")).default([]),
    artifactIds: z.array(idWithPrefix("art_")).default([]),
    reason: z.string().min(1),
    requiredHumanAction: z.string().min(1).optional(),
    nextWorkflow: z.lazy(() => AutonomyWorkflowStateSchema).optional(),
    metadata: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough()
  .superRefine(addRuntimeRecordIssues);

export const ValidatorGateDecisionStatusSchema = z.enum(["passed", "failed", "blocked"]);

export const ValidatorGateDecisionSchema = z
  .object({
    decisionId: idWithPrefix("dec_"),
    manifestId: idWithPrefix("val_"),
    gateId: z.string().min(1),
    status: ValidatorGateDecisionStatusSchema,
    requiredValidatorIds: z.array(z.string().min(1)).default([]),
    missingValidatorIds: z.array(z.string().min(1)).default([]),
    passedValidatorIds: z.array(z.string().min(1)).default([]),
    failedValidatorIds: z.array(z.string().min(1)).default([]),
    waivedValidatorIds: z.array(z.string().min(1)).default([]),
    skippedValidatorIds: z.array(z.string().min(1)).default([]),
    reason: z.string().min(1),
  })
  .passthrough()
  .superRefine(addRuntimeRecordIssues);

export const AutonomyReplayRecordSchema = z
  .object({
    replayId: idWithPrefix("rpl_"),
    eventLogPath: z.string().min(1).default("autonomy/runtime/events.jsonl"),
    recordedAt: z.string().min(1).optional(),
    eventCount: z.number().int().nonnegative(),
    appliedEventIds: z.array(idWithPrefix("evt_")).default([]),
    rejectedEventIds: z.array(idWithPrefix("evt_")).default([]),
    workItemIds: z.array(idWithPrefix("awi_")).default([]),
    workItemStatusCounts: z.record(AutonomyWorkItemStatusSchema, z.number().int().nonnegative()),
    stateHash: z.string().min(1),
    missingExternalRefs: z.array(AutonomySourceRefSchema).default([]),
  })
  .passthrough()
  .superRefine(addRuntimeRecordIssues);

export const AutonomyBudgetSchema = z
  .object({
    wallTimeMinutes: z.number().int().positive().optional(),
    modelCalls: z.number().int().nonnegative().optional(),
    tokens: z.number().int().nonnegative().optional(),
    externalApiCalls: z.number().int().nonnegative().optional(),
    remoteComputeSubmissions: z.number().int().nonnegative().optional(),
    retries: z.number().int().nonnegative().optional(),
    diskOutputBytes: z.number().int().nonnegative().optional(),
    reportIntervalHours: z.number().positive().optional(),
  })
  .passthrough();

export const AutonomyBudgetUsageSchema = z
  .object({
    used: AutonomyBudgetSchema.default({}),
    limit: AutonomyBudgetSchema.optional(),
    remaining: AutonomyBudgetSchema.optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AutonomyScheduleCadenceSchema = z
  .object({
    kind: z.enum(["interval"]).default("interval"),
    everySeconds: z.number().int().positive(),
  })
  .passthrough();

export const AutonomyScheduleStateSchema = z
  .object({
    scheduleId: z.string().min(1),
    kind: z.enum(["heartbeat", "report", "wakeup", "custom"]).default("custom"),
    enabled: z.boolean().default(true),
    role: AutonomyWakeupRoleSchema.optional(),
    cadence: AutonomyScheduleCadenceSchema,
    lastTriggeredAt: z.string().min(1).optional(),
    nextDueAt: z.string().min(1).optional(),
    timezone: z.string().min(1).optional(),
    metadata: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AutonomyScheduleDecisionSchema = z
  .object({
    scheduleId: z.string().min(1),
    due: z.boolean(),
    disabled: z.boolean().default(false),
    now: z.string().min(1),
    dueAt: z.string().min(1).optional(),
    nextDueAt: z.string().min(1).optional(),
    idempotencyKey: z.string().min(1).optional(),
    reason: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AutonomyScheduleTriggerSchema = z
  .object({
    triggerId: idWithPrefix("trg_"),
    triggerType: AutonomyTriggerTypeSchema.default("schedule_tick"),
    scheduleId: z.string().min(1),
    dueAt: z.string().min(1),
    emittedAt: z.string().min(1),
    idempotencyKey: z.string().min(1),
    metadata: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AutonomyBlockerSchema = z
  .object({
    id: z.string().min(1),
    kind: z.string().min(1),
    reason: z.string().min(1),
    approvalRequired: z.boolean().default(false),
  })
  .passthrough();

export const AutonomyVerificationOutcomeSchema = z
  .object({
    id: z.string().min(1),
    command: z.string().min(1).optional(),
    status: ValidatorStatusSchema,
    summary: z.string().min(1).optional(),
    artifactIds: z.array(idWithPrefix("art_")).default([]),
    evidenceIds: z.array(idWithPrefix("evp_")).default([]),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AutonomyHumanDecisionPromptSchema = z
  .object({
    promptId: z.string().min(1),
    question: z.string().min(1),
    actions: z.array(AutonomyFeedbackActionSchema).min(1),
    targetRefs: z.array(AutonomySourceRefSchema).default([]),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AutonomyReportMetadataSchema = z
  .object({
    reportId: idWithPrefix("rpt_"),
    period: z
      .object({
        startedAt: z.string().min(1),
        endedAt: z.string().min(1),
      })
      .passthrough(),
    runner: z.string().min(1).optional(),
    codeVersion: z.string().min(1).optional(),
    runIds: z.array(idWithPrefix("run_")).default([]),
    attemptedWorkItemIds: z.array(idWithPrefix("awi_")).default([]),
    completedWorkItemIds: z.array(idWithPrefix("awi_")).default([]),
    failedWorkItemIds: z.array(idWithPrefix("awi_")).default([]),
    blockedWorkItemIds: z.array(idWithPrefix("awi_")).default([]),
    deferredWorkItemIds: z.array(idWithPrefix("awi_")).default([]),
    changedFiles: z.array(z.string().min(1)).default([]),
    artifactIds: z.array(idWithPrefix("art_")).default([]),
    evidenceIds: z.array(idWithPrefix("evp_")).default([]),
    verification: z.array(AutonomyVerificationOutcomeSchema).default([]),
    budget: AutonomyBudgetUsageSchema.optional(),
    decisions: z.array(z.string().min(1)).default([]),
    risks: z.array(z.string().min(1)).default([]),
    limitations: z.array(z.string().min(1)).default([]),
    nextWork: z.array(z.string().min(1)).default([]),
    humanPrompts: z.array(AutonomyHumanDecisionPromptSchema).max(3).default([]),
    path: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AutonomyDashboardMetadataSchema = z
  .object({
    artifactId: idWithPrefix("art_"),
    kind: AutonomyDashboardKindSchema,
    reportId: idWithPrefix("rpt_").optional(),
    runId: idWithPrefix("run_").optional(),
    schedulerRoot: z.string().min(1).optional(),
    path: z.string().min(1).optional(),
    url: z.string().min(1).optional(),
    refreshSeconds: z.number().int().positive().default(DEFAULT_DASHBOARD_REFRESH_SECONDS),
    selfContained: z.boolean().default(true),
    externalResources: z.array(z.string().min(1)).default([]),
    authoritative: z.literal(false).default(false),
    metadata: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough()
  .superRefine((dashboard, ctx) => {
    addSecretIssues(dashboard, ctx);
    if (dashboard.kind !== "server_live" && dashboard.url) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["url"],
        message: "Local autonomy dashboards must not depend on hosted URLs.",
      });
    }
    if (dashboard.kind !== "server_live" && !dashboard.selfContained) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["selfContained"],
        message: "Local autonomy dashboards must be self-contained.",
      });
    }
    if (dashboard.kind !== "server_live" && dashboard.externalResources.length > 0) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["externalResources"],
        message: "Local autonomy dashboards must not require external resources.",
      });
    }
  });

export const AutonomyFeedbackRecordSchema = z
  .object({
    feedbackId: idWithPrefix("fbk_"),
    reportId: idWithPrefix("rpt_"),
    action: AutonomyFeedbackActionSchema,
    timestamp: z.string().min(1),
    actor: z.string().min(1).optional(),
    targetRefs: z.array(AutonomySourceRefSchema).default([]),
    comment: z.string().optional(),
    route: z
      .object({
        kind: AutonomyFeedbackRouteSchema,
        targetId: z.string().min(1).optional(),
        metadata: z.record(z.string(), z.unknown()).optional(),
      })
      .passthrough()
      .default({ kind: "none" }),
    resultingRefs: z.array(AutonomySourceRefSchema).default([]),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AutonomyWakeupStateSchema = z
  .object({
    role: AutonomyWakeupRoleSchema,
    adapter: AutonomyWakeupAdapterSchema,
    statePath: z.string().min(1).optional(),
    tickLockPath: z.string().min(1).default("autonomy/tick.lock"),
    desiredCadenceSeconds: z.number().int().positive(),
    installed: z.boolean().default(false),
    running: z.boolean().default(false),
    lastInstalledAt: z.string().min(1).optional(),
    lastHeartbeatAt: z.string().min(1).optional(),
    nextDueAt: z.string().min(1).optional(),
    lastError: z.string().optional(),
    metadata: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AutonomyDaemonRunMetadataSchema = z
  .object({
    runId: z.string().min(1),
    source: z.string().min(1),
    sourceKey: z.string().min(1),
    repository: z.string().min(1).optional(),
    issueNumber: z.number().int().positive().optional(),
    issueUrl: z.string().min(1).optional(),
    title: z.string().min(1),
    requester: z.string().min(1).optional(),
    labels: z.array(z.string().min(1)).default([]),
    kind: z.string().min(1),
    priority: z.number().int().default(0),
    selectedAdapter: z.string().min(1).optional(),
    status: AutonomyDaemonRunStatusSchema,
    attemptCount: z.number().int().nonnegative().default(0),
    maxAttempts: z.number().int().positive().optional(),
    leaseOwner: z.string().min(1).optional(),
    leaseId: z.string().min(1).optional(),
    leaseExpiresAt: z.string().min(1).optional(),
    agentSessionId: z.string().min(1).optional(),
    resultRef: z.string().min(1).optional(),
    lastError: z.string().optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AutonomyCircuitBreakerDecisionSchema = z
  .object({
    decisionId: idWithPrefix("dec_"),
    workItemId: idWithPrefix("awi_").optional(),
    status: AutonomyCircuitBreakerStatusSchema,
    consecutiveFailures: z.number().int().nonnegative(),
    maxFailures: z.number().int().positive().default(DEFAULT_CIRCUIT_BREAKER_FAILURES),
    failureSignature: z.string().min(1).optional(),
    nextStatus: z.enum(["needs_human", "blocked"]).optional(),
    reason: z.string().min(1),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AutonomyRetryStateSchema = z
  .object({
    count: z.number().int().nonnegative().default(0),
    maxAttempts: z.number().int().positive().optional(),
    lastAttemptStatus: z.string().optional(),
    lastFailureSignature: z.string().optional(),
  })
  .default({ count: 0 });

export const AutonomyLeaseSchema = z
  .object({
    leaseId: z.string().min(1),
    runId: idWithPrefix("run_"),
    actor: z.string().min(1),
    startedAt: z.string().min(1),
    expiresAt: z.string().min(1),
    renewedAt: z.string().min(1).optional(),
    budgetSnapshot: AutonomyBudgetSchema.optional(),
  })
  .passthrough();

export const AutonomyWorkflowStateSchema = z
  .object({
    kind: z.string().min(1),
    stage: z.string().min(1),
    agentRunIds: z.array(idWithPrefix("agt_")).default([]),
    decisionId: idWithPrefix("dec_").optional(),
  })
  .passthrough();

export const AnalysisWorkItemDetailsSchema = z
  .object({
    question: z.string().min(1).optional(),
    inputData: z.array(AutonomySourceRefSchema).default([]),
    plannedCommands: z.array(z.string().min(1)).default([]),
    expectedArtifacts: z.array(z.string().min(1)).default([]),
    environment: z.record(z.string(), z.unknown()).optional(),
    parameters: z.record(z.string(), z.unknown()).optional(),
    randomSeeds: z.record(z.string(), z.union([z.string(), z.number()])).optional(),
    validationChecks: z.array(z.string().min(1)).default([]),
    interpretationBoundary: z
      .enum(["benchmark", "paper", "memory", "spec", "human_review", "report_only"])
      .default("human_review"),
  })
  .passthrough();

export const AutonomyWorkItemSchema = z
  .object({
    id: idWithPrefix("awi_"),
    class: AutonomyWorkClassSchema,
    type: z.string().min(1),
    status: AutonomyWorkItemStatusSchema,
    priority: z.number().int().default(0),
    autonomyLevel: AutonomyLevelSchema,
    owner: z.string().optional(),
    reviewOwner: z.string().optional(),
    sourceRefs: z.array(AutonomySourceRefSchema).default([]),
    nextAction: z.string().optional(),
    requiredEvidence: z.array(z.string().min(1)).default([]),
    budget: AutonomyBudgetSchema.optional(),
    blockers: z.array(AutonomyBlockerSchema).default([]),
    retry: AutonomyRetryStateSchema,
    lease: AutonomyLeaseSchema.optional(),
    lastReport: z
      .object({
        reportId: idWithPrefix("rpt_").optional(),
        timestamp: z.string().min(1).optional(),
      })
      .optional(),
    workflow: AutonomyWorkflowStateSchema.optional(),
    analysis: AnalysisWorkItemDetailsSchema.optional(),
    metadata: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AutonomyRuntimeEventSchema = z
  .object({
    eventId: idWithPrefix("evt_"),
    eventType: AutonomyRuntimeEventTypeSchema,
    timestamp: z.string().min(1),
    source: z.string().min(1),
    idempotencyKey: z.string().min(1),
    workItemId: idWithPrefix("awi_").optional(),
    runId: idWithPrefix("run_").optional(),
    previousState: AutonomyWorkItemStatusSchema.optional(),
    nextState: AutonomyWorkItemStatusSchema.optional(),
    commandId: z.string().min(1).optional(),
    validatorId: z.string().min(1).optional(),
    payload: z.record(z.string(), z.unknown()).default({}),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const ArtifactReferenceSchema = z
  .object({
    id: z.string().min(1).optional(),
    kind: ArtifactKindSchema.default("other"),
    path: z.string().min(1).optional(),
    url: z.string().min(1).optional(),
    checksum: z.string().min(1).optional(),
    version: z.string().min(1).optional(),
    immutable: z.boolean().optional(),
    metadata: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough();

export const CommandSpecSchema = z
  .object({
    program: z.string().min(1),
    args: z.array(z.string()).default([]),
    cwd: z.string().min(1).optional(),
    env: z.record(z.string(), z.string()).optional(),
  })
  .passthrough();

export const CommandRetryPolicySchema = z
  .object({
    maxAttempts: z.number().int().positive().default(1),
    backoffSeconds: z.number().int().nonnegative().optional(),
    rerunRequiresApproval: z.boolean().default(false),
  })
  .passthrough();

export const CommandArtifactPolicySchema = z
  .object({
    overwrite: z.enum(["never", "if_authorized", "always"]).default("never"),
    expectedArtifacts: z.array(ArtifactReferenceSchema).default([]),
    maxOutputBytes: z.number().int().nonnegative().optional(),
  })
  .passthrough();

export const CommandDagNodeSchema = z
  .object({
    nodeId: z.string().min(1),
    command: CommandSpecSchema.optional(),
    operationId: z.string().min(1).optional(),
    inputs: z.array(ArtifactReferenceSchema).default([]),
    outputs: z.array(ArtifactReferenceSchema).default([]),
    dependencies: z.array(z.string().min(1)).default([]),
    requirements: z
      .object({
        tools: z.array(z.string()).default([]),
        environment: z.array(z.string()).default([]),
        software: z.array(z.string()).default([]),
      })
      .passthrough()
      .default({}),
    timeoutSeconds: z.number().int().positive().optional(),
    retry: CommandRetryPolicySchema.default({ maxAttempts: 1 }),
    validators: z.array(z.string().min(1)).default([]),
    artifactPolicy: CommandArtifactPolicySchema.default({}),
    idempotency: z
      .object({
        key: z.string().min(1),
        rerunRequiresApproval: z.boolean().default(false),
        rationale: z.string().optional(),
      })
      .passthrough()
      .optional(),
  })
  .passthrough()
  .superRefine((node, ctx) => {
    if (!node.command && !node.operationId) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Command DAG node must declare either command or operationId.",
      });
    }
    if (node.command && node.operationId) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Command DAG node cannot declare both command and operationId.",
      });
    }
  });

export const CommandDagSchema = z
  .object({
    dagId: idWithPrefix("dag_"),
    name: z.string().min(1).optional(),
    description: z.string().optional(),
    nodes: z.array(CommandDagNodeSchema).min(1),
    metadata: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough()
  .superRefine((dag, ctx) => {
    addSecretIssues(dag, ctx);
    validateDagStructure(dag, ctx);
  });

export const ValidatorDefinitionSchema = z
  .object({
    id: z.string().min(1),
    kind: z.enum([
      "command",
      "schema",
      "checksum",
      "golden_report",
      "policy",
      "secret_scan",
      "manual_approval",
      "custom",
    ]),
    label: z.string().min(1).optional(),
    command: CommandSpecSchema.optional(),
    required: z.boolean().default(true),
    timeoutSeconds: z.number().int().positive().optional(),
    waiverAllowed: z.boolean().default(false),
    metadata: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough();

export const ValidatorManifestSchema = z
  .object({
    manifestId: idWithPrefix("val_"),
    name: z.string().min(1),
    validators: z.array(ValidatorDefinitionSchema).min(1),
    gates: z.record(z.string(), z.array(z.string())).default({}),
    metadata: z.record(z.string(), z.unknown()).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const ValidatorOutcomeSchema = z
  .object({
    validatorId: z.string().min(1),
    status: ValidatorStatusSchema,
    startedAt: z.string().min(1).optional(),
    endedAt: z.string().min(1).optional(),
    exitCode: z.number().int().optional(),
    output: z.string().optional(),
    artifacts: z.array(ArtifactReferenceSchema).default([]),
    waiverReason: z.string().optional(),
  })
  .passthrough();

export const CommandRunSummarySchema = z
  .object({
    nodeId: z.string().min(1).optional(),
    command: CommandSpecSchema.optional(),
    operationId: z.string().min(1).optional(),
    startedAt: z.string().min(1).optional(),
    endedAt: z.string().min(1).optional(),
    status: z.enum(["succeeded", "failed", "skipped"]).optional(),
    exitCode: z.number().int().optional(),
    artifacts: z.array(ArtifactReferenceSchema).default([]),
  })
  .passthrough();

export const EvidencePacketSchema = z
  .object({
    evidenceId: idWithPrefix("evp_"),
    workItemId: idWithPrefix("awi_"),
    runId: idWithPrefix("run_"),
    workspace: z
      .object({
        root: z.string().min(1).optional(),
        gitCommit: z.string().min(1).optional(),
        snapshot: z.string().min(1).optional(),
      })
      .passthrough(),
    inputs: z.array(ArtifactReferenceSchema).default([]),
    commands: z.array(CommandRunSummarySchema).default([]),
    parameters: z.record(z.string(), z.unknown()).default({}),
    environment: z.record(z.string(), z.unknown()).default({}),
    artifacts: z.array(ArtifactReferenceSchema).default([]),
    validation: z.array(ValidatorOutcomeSchema).default([]),
    limitations: z.array(z.string()).default([]),
    observations: z.array(z.string()).default([]),
    conclusions: z.array(z.string()).default([]),
    confidence: z.enum(["low", "medium", "high", "review_required"]).default("review_required"),
    followUp: z.array(z.string()).default([]),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export type AutonomyWorkClass = z.infer<typeof AutonomyWorkClassSchema>;
export type AutonomyLevel = z.infer<typeof AutonomyLevelSchema>;
export type AutonomyWorkItemStatus = z.infer<typeof AutonomyWorkItemStatusSchema>;
export type AutonomyTriggerType = z.infer<typeof AutonomyTriggerTypeSchema>;
export type AutonomyRuntimeEventType = z.infer<typeof AutonomyRuntimeEventTypeSchema>;
export type EngineeringLifecyclePrimaryState = z.infer<
  typeof EngineeringLifecyclePrimaryStateSchema
>;
export type EngineeringLifecycleSideState = z.infer<typeof EngineeringLifecycleSideStateSchema>;
export type EngineeringLifecycleState = z.infer<typeof EngineeringLifecycleStateSchema>;
export type EngineeringIntakeSourceType = z.infer<typeof EngineeringIntakeSourceTypeSchema>;
export type EngineeringProposalStatus = z.infer<typeof EngineeringProposalStatusSchema>;
export type EngineeringApprovalStatus = z.infer<typeof EngineeringApprovalStatusSchema>;
export type AutonomyAgentRunStatus = z.infer<typeof AutonomyAgentRunStatusSchema>;
export type AutonomyWorkflowDecisionKind = z.infer<typeof AutonomyWorkflowDecisionKindSchema>;
export type AutonomyWorkflowDecisionStatus = z.infer<typeof AutonomyWorkflowDecisionStatusSchema>;
export type EngineeringApprovalRecord = z.infer<typeof EngineeringApprovalRecordSchema>;
export type EngineeringIntakeRecord = z.infer<typeof EngineeringIntakeRecordSchema>;
export type EngineeringProposalRecord = z.infer<typeof EngineeringProposalRecordSchema>;
export type AutonomyWorkItem = z.infer<typeof AutonomyWorkItemSchema>;
export type AutonomyRuntimeEvent = z.infer<typeof AutonomyRuntimeEventSchema>;
export type AutonomyTriggerRecord = z.infer<typeof AutonomyTriggerRecordSchema>;
export type AutonomyTransitionDecisionStatus = z.infer<
  typeof AutonomyTransitionDecisionStatusSchema
>;
export type AutonomyTransitionPreconditionKind = z.infer<
  typeof AutonomyTransitionPreconditionKindSchema
>;
export type AutonomyTransitionPreconditionStatus = z.infer<
  typeof AutonomyTransitionPreconditionStatusSchema
>;
export type AutonomyTransitionIdempotencyStatus = z.infer<
  typeof AutonomyTransitionIdempotencyStatusSchema
>;
export type AutonomyTransitionPrecondition = z.infer<typeof AutonomyTransitionPreconditionSchema>;
export type AutonomyTransitionDecision = z.infer<typeof AutonomyTransitionDecisionSchema>;
export type EngineeringLifecycleTransitionDecision = z.infer<
  typeof EngineeringLifecycleTransitionDecisionSchema
>;
export type AutonomyAgentRunRecord = z.infer<typeof AutonomyAgentRunRecordSchema>;
export type AutonomyWorkflowDecisionRecord = z.infer<typeof AutonomyWorkflowDecisionRecordSchema>;
export type ValidatorGateDecisionStatus = z.infer<typeof ValidatorGateDecisionStatusSchema>;
export type ValidatorGateDecision = z.infer<typeof ValidatorGateDecisionSchema>;
export type AutonomyReplayRecord = z.infer<typeof AutonomyReplayRecordSchema>;
export type AutonomySchedulerEventType = z.infer<typeof AutonomySchedulerEventTypeSchema>;
export type AutonomyFeedbackAction = z.infer<typeof AutonomyFeedbackActionSchema>;
export type AutonomyFeedbackRoute = z.infer<typeof AutonomyFeedbackRouteSchema>;
export type AutonomyDashboardKind = z.infer<typeof AutonomyDashboardKindSchema>;
export type AutonomyWakeupRole = z.infer<typeof AutonomyWakeupRoleSchema>;
export type AutonomyWakeupAdapter = z.infer<typeof AutonomyWakeupAdapterSchema>;
export type AutonomyDaemonRunStatus = z.infer<typeof AutonomyDaemonRunStatusSchema>;
export type AutonomyCircuitBreakerStatus = z.infer<typeof AutonomyCircuitBreakerStatusSchema>;
export type AutonomyBudgetUsage = z.infer<typeof AutonomyBudgetUsageSchema>;
export type AutonomyScheduleState = z.infer<typeof AutonomyScheduleStateSchema>;
export type AutonomyScheduleDecision = z.infer<typeof AutonomyScheduleDecisionSchema>;
export type AutonomyScheduleTrigger = z.infer<typeof AutonomyScheduleTriggerSchema>;
export type AutonomyVerificationOutcome = z.infer<typeof AutonomyVerificationOutcomeSchema>;
export type AutonomyHumanDecisionPrompt = z.infer<typeof AutonomyHumanDecisionPromptSchema>;
export type AutonomyReportMetadata = z.infer<typeof AutonomyReportMetadataSchema>;
export type AutonomyDashboardMetadata = z.infer<typeof AutonomyDashboardMetadataSchema>;
export type AutonomyFeedbackRecord = z.infer<typeof AutonomyFeedbackRecordSchema>;
export type AutonomyWakeupState = z.infer<typeof AutonomyWakeupStateSchema>;
export type AutonomyDaemonRunMetadata = z.infer<typeof AutonomyDaemonRunMetadataSchema>;
export type AutonomyCircuitBreakerDecision = z.infer<typeof AutonomyCircuitBreakerDecisionSchema>;
export type AutonomyWorkflowState = z.infer<typeof AutonomyWorkflowStateSchema>;
export type CommandDag = z.infer<typeof CommandDagSchema>;
export type ValidatorManifest = z.infer<typeof ValidatorManifestSchema>;
export type ValidatorOutcome = z.infer<typeof ValidatorOutcomeSchema>;
export type EvidencePacket = z.infer<typeof EvidencePacketSchema>;

export interface AutonomyReplayState {
  workItems: Record<string, AutonomyWorkItem>;
  events: AutonomyRuntimeEvent[];
  rejectedEvents: AutonomyRuntimeEvent[];
  appliedEventIds: string[];
  appliedIdempotencyKeys: string[];
}

export type CreateAutonomyRuntimeEventInput = Omit<AutonomyRuntimeEvent, "eventId" | "payload"> & {
  eventId?: string;
  payload?: Record<string, unknown>;
};

export interface EvaluateAutonomyTransitionOptions {
  workItem: unknown;
  to: AutonomyWorkItemStatus;
  from?: AutonomyWorkItemStatus;
  runId?: string;
  idempotencyKey: string;
  appliedIdempotencyKeys?: string[];
  preconditions?: unknown[];
  patch?: Record<string, unknown>;
  decisionId?: string;
}

export interface CreateAutonomyTransitionRuntimeEventOptions {
  decision: unknown;
  timestamp: string;
  source?: string;
}

export interface CreateAutonomyAgentRunRuntimeEventOptions {
  agentRun: unknown;
  timestamp: string;
  source?: string;
  idempotencyKey?: string;
}

export interface CreateAutonomyWorkflowDecisionRuntimeEventOptions {
  decision: unknown;
  timestamp: string;
  source?: string;
  idempotencyKey?: string;
}

export interface CreateAutonomyReplayRecordOptions {
  eventLogPath?: string;
  recordedAt?: string;
  missingExternalRefs?: unknown[];
  replayId?: string;
}

export interface EvaluateEngineeringLifecycleTransitionOptions {
  workItem?: unknown;
  currentState?: EngineeringLifecycleState;
  to: EngineeringLifecycleState;
  workItemId?: string;
  requiredEvidenceIds?: string[];
  presentEvidenceIds?: string[];
  requiredApprovalIds?: string[];
  approvals?: unknown[];
  validatorGate?: unknown;
  decisionId?: string;
}

const VALID_STATUS_TRANSITIONS: Record<AutonomyWorkItemStatus, readonly AutonomyWorkItemStatus[]> =
  {
    queued: ["queued", "leased", "blocked", "needs_human", "canceled", "superseded"],
    leased: ["leased", "queued", "running", "blocked", "failed", "needs_human", "canceled"],
    running: ["running", "succeeded", "failed", "blocked", "needs_human", "canceled"],
    blocked: ["blocked", "queued", "needs_human", "canceled", "superseded"],
    needs_human: ["needs_human", "queued", "blocked", "canceled", "superseded"],
    failed: ["failed", "queued", "needs_human", "canceled", "superseded"],
    succeeded: ["succeeded", "superseded"],
    canceled: ["canceled", "superseded"],
    superseded: ["superseded"],
  };

const VALID_ENGINEERING_LIFECYCLE_TRANSITIONS: Record<
  EngineeringLifecycleState,
  readonly EngineeringLifecycleState[]
> = {
  intake: ["intake", "triage", "needs_human", "rejected", "deferred", "blocked"],
  triage: ["triage", "proposal", "needs_human", "rejected", "deferred", "blocked"],
  proposal: [
    "proposal",
    "discussion",
    "specification",
    "needs_human",
    "rejected",
    "deferred",
    "blocked",
  ],
  discussion: [
    "discussion",
    "proposal",
    "specification",
    "needs_human",
    "rejected",
    "deferred",
    "blocked",
  ],
  specification: [
    "specification",
    "implementation",
    "needs_human",
    "rejected",
    "deferred",
    "blocked",
  ],
  implementation: ["implementation", "validation", "needs_human", "blocked", "deferred"],
  validation: ["validation", "implementation", "report", "needs_human", "blocked", "deferred"],
  report: ["report", "close", "implementation", "needs_human", "blocked", "deferred"],
  close: ["close", "superseded"],
  needs_human: [
    "needs_human",
    "triage",
    "proposal",
    "discussion",
    "specification",
    "implementation",
    "validation",
    "report",
    "rejected",
    "deferred",
    "blocked",
  ],
  blocked: [
    "blocked",
    "triage",
    "proposal",
    "discussion",
    "specification",
    "implementation",
    "validation",
    "needs_human",
    "rejected",
    "deferred",
  ],
  rejected: ["rejected", "superseded"],
  deferred: ["deferred", "triage", "proposal", "needs_human", "rejected", "superseded"],
  superseded: ["superseded"],
};

export const AUTONOMY_PATHS = Object.freeze({
  root: "autonomy",
  workItems: "autonomy/work-items.jsonl",
  runtimeEvents: "autonomy/runtime/events.jsonl",
  discovery: "autonomy/discovery.json",
  inbox: "autonomy/inbox",
  schedules: "autonomy/schedules.json",
  feedback: "autonomy/feedback.jsonl",
  liveDashboard: "autonomy/artifacts/latest.html",
  wakeup: "autonomy/wakeup.json",
  serverWakeup: "autonomy/wakeup-server.json",
  tickLock: "autonomy/tick.lock",
});

const DATED_PATHS = {
  run: { dir: "runs", ext: "jsonl", prefix: "run_" },
  proposal: { dir: "proposals", ext: "md", prefix: "prp_" },
  agentRun: { dir: "agent-runs", ext: "json", prefix: "agt_" },
  report: { dir: "reports", ext: "md", prefix: "rpt_" },
  artifact: { dir: "artifacts", ext: "html", prefix: "art_" },
  evidence: { dir: "evidence", ext: "json", prefix: "evp_" },
} as const;

export type AutonomyDatedPathKind = keyof typeof DATED_PATHS;

export function parseAutonomyWorkItem(input: unknown): AutonomyWorkItem {
  return AutonomyWorkItemSchema.parse(input);
}

export function parseAutonomyRuntimeEvent(input: unknown): AutonomyRuntimeEvent {
  return AutonomyRuntimeEventSchema.parse(input);
}

export function parseAutonomyTriggerRecord(input: unknown): AutonomyTriggerRecord {
  return AutonomyTriggerRecordSchema.parse(input);
}

export function parseEngineeringIntakeRecord(input: unknown): EngineeringIntakeRecord {
  return EngineeringIntakeRecordSchema.parse(input);
}

export function parseEngineeringProposalRecord(input: unknown): EngineeringProposalRecord {
  return EngineeringProposalRecordSchema.parse(input);
}

export function parseAutonomyTransitionDecision(input: unknown): AutonomyTransitionDecision {
  return AutonomyTransitionDecisionSchema.parse(input);
}

export function parseEngineeringLifecycleTransitionDecision(
  input: unknown,
): EngineeringLifecycleTransitionDecision {
  return EngineeringLifecycleTransitionDecisionSchema.parse(input);
}

export function parseAutonomyAgentRunRecord(input: unknown): AutonomyAgentRunRecord {
  return AutonomyAgentRunRecordSchema.parse(input);
}

export function parseAutonomyWorkflowDecisionRecord(
  input: unknown,
): AutonomyWorkflowDecisionRecord {
  return AutonomyWorkflowDecisionRecordSchema.parse(input);
}

export function parseValidatorGateDecision(input: unknown): ValidatorGateDecision {
  return ValidatorGateDecisionSchema.parse(input);
}

export function parseAutonomyReplayRecord(input: unknown): AutonomyReplayRecord {
  return AutonomyReplayRecordSchema.parse(input);
}

export function parseAutonomyScheduleState(input: unknown): AutonomyScheduleState {
  return AutonomyScheduleStateSchema.parse(input);
}

export function parseAutonomyScheduleDecision(input: unknown): AutonomyScheduleDecision {
  return AutonomyScheduleDecisionSchema.parse(input);
}

export function parseAutonomyScheduleTrigger(input: unknown): AutonomyScheduleTrigger {
  return AutonomyScheduleTriggerSchema.parse(input);
}

export function parseAutonomyReportMetadata(input: unknown): AutonomyReportMetadata {
  return AutonomyReportMetadataSchema.parse(input);
}

export function parseAutonomyDashboardMetadata(input: unknown): AutonomyDashboardMetadata {
  return AutonomyDashboardMetadataSchema.parse(input);
}

export function parseAutonomyFeedbackRecord(input: unknown): AutonomyFeedbackRecord {
  return AutonomyFeedbackRecordSchema.parse(input);
}

export function parseAutonomyWakeupState(input: unknown): AutonomyWakeupState {
  return AutonomyWakeupStateSchema.parse(input);
}

export function parseAutonomyDaemonRunMetadata(input: unknown): AutonomyDaemonRunMetadata {
  return AutonomyDaemonRunMetadataSchema.parse(input);
}

export function parseAutonomyCircuitBreakerDecision(
  input: unknown,
): AutonomyCircuitBreakerDecision {
  return AutonomyCircuitBreakerDecisionSchema.parse(input);
}

export function parseCommandDag(input: unknown): CommandDag {
  return CommandDagSchema.parse(input);
}

export function parseValidatorManifest(input: unknown): ValidatorManifest {
  return ValidatorManifestSchema.parse(input);
}

export function parseEvidencePacket(input: unknown): EvidencePacket {
  return EvidencePacketSchema.parse(input);
}

export function createAutonomyRuntimeEvent(
  input: CreateAutonomyRuntimeEventInput,
): AutonomyRuntimeEvent {
  return AutonomyRuntimeEventSchema.parse({
    ...input,
    payload: input.payload ?? {},
    eventId: input.eventId ?? deterministicEventId(input),
  });
}

export function runtimeEventFromTrigger(input: unknown): AutonomyRuntimeEvent {
  const trigger = AutonomyTriggerRecordSchema.parse(input);
  return createAutonomyRuntimeEvent({
    eventType: trigger.triggerType,
    timestamp: trigger.timestamp,
    source: trigger.source,
    idempotencyKey: trigger.idempotencyKey,
    workItemId: trigger.workItemId,
    runId: trigger.runId,
    payload: {
      triggerId: trigger.triggerId,
      triggerType: trigger.triggerType,
      scheduleId: trigger.scheduleId,
      feedbackId: trigger.feedbackId,
      sourceRefs: trigger.sourceRefs,
      payload: trigger.payload,
    },
  });
}

export function evaluateAutonomyTransition(
  options: EvaluateAutonomyTransitionOptions,
): AutonomyTransitionDecision {
  const workItem = AutonomyWorkItemSchema.parse(options.workItem);
  const from = options.from ?? workItem.status;
  const currentStateMatches = workItem.status === from;
  const validEdge = canTransitionWorkItemStatus(workItem.status, options.to);
  const duplicateIdempotency =
    options.appliedIdempotencyKeys?.includes(options.idempotencyKey) ?? false;
  const callerPreconditions = (options.preconditions ?? []).map((precondition) =>
    AutonomyTransitionPreconditionSchema.parse(precondition),
  );
  const preconditions = [
    AutonomyTransitionPreconditionSchema.parse({
      kind: "current_state",
      status: currentStateMatches ? "passed" : "failed",
      message: currentStateMatches
        ? `Current state is ${workItem.status}.`
        : `Current state is ${workItem.status}, not ${from}.`,
    }),
    AutonomyTransitionPreconditionSchema.parse({
      kind: "status_edge",
      status: validEdge ? "passed" : "failed",
      message: validEdge
        ? `Transition ${workItem.status} -> ${options.to} is valid.`
        : `Transition ${workItem.status} -> ${options.to} is not valid.`,
    }),
    AutonomyTransitionPreconditionSchema.parse({
      kind: "idempotency",
      status: duplicateIdempotency ? "failed" : "passed",
      message: duplicateIdempotency
        ? `Idempotency key "${options.idempotencyKey}" was already applied.`
        : `Idempotency key "${options.idempotencyKey}" is new.`,
    }),
    ...callerPreconditions,
  ];
  const missingRequirements = preconditions
    .filter((precondition) => precondition.required)
    .filter((precondition) => !["passed", "waived"].includes(precondition.status))
    .map((precondition) => precondition.message ?? transitionPreconditionLabel(precondition));
  const status: AutonomyTransitionDecisionStatus =
    missingRequirements.length === 0 ? "allowed" : "rejected";
  const decisionInput = {
    workItemId: workItem.id,
    runId: options.runId,
    from,
    to: options.to,
    status,
    idempotencyKey: options.idempotencyKey,
    idempotencyStatus: duplicateIdempotency ? "duplicate" : "new",
    preconditions,
    missingRequirements,
    reason:
      status === "allowed"
        ? `Transition ${from} -> ${options.to} allowed.`
        : missingRequirements.join("; "),
    patch: options.patch,
  } satisfies Omit<AutonomyTransitionDecision, "decisionId">;

  return AutonomyTransitionDecisionSchema.parse({
    ...decisionInput,
    decisionId: options.decisionId ?? deterministicId("dec_", decisionInput),
  });
}

export function createAutonomyTransitionRuntimeEvent(
  options: CreateAutonomyTransitionRuntimeEventOptions,
): AutonomyRuntimeEvent {
  const decision = AutonomyTransitionDecisionSchema.parse(options.decision);
  if (decision.status === "allowed") {
    return createAutonomyRuntimeEvent({
      eventType: "deterministic_transition",
      timestamp: options.timestamp,
      source: options.source ?? "swarmx.runtime",
      idempotencyKey: decision.idempotencyKey,
      workItemId: decision.workItemId,
      runId: decision.runId,
      previousState: decision.from,
      nextState: decision.to,
      payload: {
        decisionId: decision.decisionId,
        patch: decision.patch,
      },
    });
  }

  return createAutonomyRuntimeEvent({
    eventType: "state_transition_rejected",
    timestamp: options.timestamp,
    source: options.source ?? "swarmx.runtime",
    idempotencyKey: decision.idempotencyKey,
    workItemId: decision.workItemId,
    runId: decision.runId,
    payload: {
      decisionId: decision.decisionId,
      requestedFrom: decision.from,
      requestedTo: decision.to,
      reason: decision.reason,
      missingRequirements: decision.missingRequirements,
    },
  });
}

export function createAutonomyAgentRunRuntimeEvent(
  options: CreateAutonomyAgentRunRuntimeEventOptions,
): AutonomyRuntimeEvent {
  const agentRun = AutonomyAgentRunRecordSchema.parse(options.agentRun);
  return createAutonomyRuntimeEvent({
    eventType: "agent_run_written",
    timestamp: options.timestamp,
    source: options.source ?? "swarmx.runtime",
    idempotencyKey: options.idempotencyKey ?? `agent-run:${agentRun.agentRunId}`,
    workItemId: agentRun.workItemId,
    runId: agentRun.runId,
    payload: {
      agentRunId: agentRun.agentRunId,
      workflowKind: agentRun.workflowKind,
      stage: agentRun.stage,
      role: agentRun.role,
      status: agentRun.status,
      artifactIds: agentRun.artifactIds,
      evidenceIds: agentRun.evidenceIds,
      resultRef: agentRun.resultRef,
      errorRef: agentRun.errorRef,
      summary: agentRun.summary,
    },
  });
}

export function createAutonomyWorkflowDecisionRuntimeEvent(
  options: CreateAutonomyWorkflowDecisionRuntimeEventOptions,
): AutonomyRuntimeEvent {
  const decision = AutonomyWorkflowDecisionRecordSchema.parse(options.decision);
  return createAutonomyRuntimeEvent({
    eventType: "workflow_decision_recorded",
    timestamp: options.timestamp,
    source: options.source ?? "swarmx.runtime",
    idempotencyKey: options.idempotencyKey ?? `workflow-decision:${decision.decisionId}`,
    workItemId: decision.workItemId,
    runId: decision.runId,
    payload: {
      decisionId: decision.decisionId,
      workflowKind: decision.workflowKind,
      currentStage: decision.currentStage,
      nextStage: decision.nextStage,
      status: decision.status,
      decisionKind: decision.decisionKind,
      agentRunIds: decision.agentRunIds,
      artifactIds: decision.artifactIds,
      evidenceIds: decision.evidenceIds,
      nextWorkflow: decision.nextWorkflow,
      reason: decision.reason,
    },
  });
}

export function linkAgentStageToWorkflowState(
  workflowInput: unknown,
  agentRunInput: unknown,
  decisionInput?: unknown,
): AutonomyWorkflowState {
  const agentRun = AutonomyAgentRunRecordSchema.parse(agentRunInput);
  const decision =
    decisionInput === undefined
      ? undefined
      : AutonomyWorkflowDecisionRecordSchema.parse(decisionInput);
  const base = workflowInput
    ? AutonomyWorkflowStateSchema.parse(workflowInput)
    : AutonomyWorkflowStateSchema.parse({
        kind: agentRun.workflowKind,
        stage: agentRun.stage,
      });

  if (base.kind !== agentRun.workflowKind) {
    throw new Error(
      `Workflow kind "${base.kind}" does not match agent run "${agentRun.workflowKind}".`,
    );
  }
  if (decision && decision.workflowKind !== base.kind) {
    throw new Error(
      `Workflow kind "${base.kind}" does not match decision "${decision.workflowKind}".`,
    );
  }

  return AutonomyWorkflowStateSchema.parse({
    ...base,
    stage: decision?.nextWorkflow?.stage ?? decision?.nextStage ?? base.stage,
    agentRunIds: uniqueStrings([
      ...base.agentRunIds,
      agentRun.agentRunId,
      ...(decision?.agentRunIds ?? []),
    ]),
    decisionId: decision?.decisionId ?? base.decisionId,
  });
}

export function evaluateEngineeringLifecycleTransition(
  options: EvaluateEngineeringLifecycleTransitionOptions,
): EngineeringLifecycleTransitionDecision {
  const workItem = options.workItem ? AutonomyWorkItemSchema.parse(options.workItem) : undefined;
  const currentState = EngineeringLifecycleStateSchema.parse(
    options.currentState ?? workItem?.workflow?.stage ?? "intake",
  );
  const to = EngineeringLifecycleStateSchema.parse(options.to);
  const validEdge = canTransitionEngineeringLifecycleState(currentState, to);
  const currentStateMatched =
    !options.workItem || !workItem?.workflow?.stage || workItem.workflow.stage === currentState;
  const requiredEvidenceIds = options.requiredEvidenceIds ?? [];
  const presentEvidenceIds = uniqueStrings(options.presentEvidenceIds ?? []);
  const missingEvidenceIds = requiredEvidenceIds.filter((id) => !presentEvidenceIds.includes(id));
  const approvals = (options.approvals ?? []).map((approval) =>
    EngineeringApprovalRecordSchema.parse(approval),
  );
  const presentApprovalIds = approvals
    .filter((approval) => ["approved", "waived"].includes(approval.status))
    .map((approval) => approval.approvalId);
  const requiredApprovalIds = options.requiredApprovalIds ?? [];
  const missingApprovalIds = requiredApprovalIds.filter((id) => !presentApprovalIds.includes(id));
  const validatorGate = options.validatorGate
    ? ValidatorGateDecisionSchema.parse(options.validatorGate)
    : undefined;
  const validatorGateOk = !validatorGate || validatorGate.status === "passed";
  const failures = [
    currentStateMatched ? undefined : `Current lifecycle state is not ${currentState}.`,
    validEdge ? undefined : `Lifecycle transition ${currentState} -> ${to} is not valid.`,
    ...missingEvidenceIds.map((id) => `Missing evidence ${id}.`),
    ...missingApprovalIds.map((id) => `Missing approval ${id}.`),
    validatorGateOk
      ? undefined
      : `Validator gate ${validatorGate?.gateId} is ${validatorGate?.status}.`,
  ].filter((failure): failure is string => Boolean(failure));
  const status: AutonomyTransitionDecisionStatus = failures.length === 0 ? "allowed" : "rejected";
  const decisionInput = {
    workItemId: workItem?.id ?? options.workItemId,
    from: currentState,
    to,
    status,
    currentStateMatched,
    validEdge,
    requiredEvidenceIds,
    presentEvidenceIds,
    missingEvidenceIds,
    requiredApprovalIds,
    presentApprovalIds,
    missingApprovalIds,
    validatorGate,
    reason:
      status === "allowed"
        ? `Lifecycle transition ${currentState} -> ${to} allowed.`
        : failures.join("; "),
    nextWorkflow:
      status === "allowed"
        ? engineeringLifecycleWorkflowState(to, workItem?.workflow?.agentRunIds ?? [])
        : undefined,
  } satisfies Omit<EngineeringLifecycleTransitionDecision, "decisionId">;

  return EngineeringLifecycleTransitionDecisionSchema.parse({
    ...decisionInput,
    decisionId: options.decisionId ?? deterministicId("dec_", decisionInput),
  });
}

export function canTransitionEngineeringLifecycleState(
  from: EngineeringLifecycleState,
  to: EngineeringLifecycleState,
): boolean {
  return VALID_ENGINEERING_LIFECYCLE_TRANSITIONS[from].includes(to);
}

export function engineeringLifecycleWorkflowState(
  stage: EngineeringLifecycleState,
  agentRunIds: string[] = [],
): AutonomyWorkflowState {
  return AutonomyWorkflowStateSchema.parse({
    kind: "engineering",
    stage,
    agentRunIds,
  });
}

export function evaluateValidatorGate(
  manifestInput: unknown,
  outcomesInput: unknown[],
  gateId = "completion",
): ValidatorGateDecision {
  const manifest = ValidatorManifestSchema.parse(manifestInput);
  const outcomes = outcomesInput.map((outcome) => ValidatorOutcomeSchema.parse(outcome));
  const definitionsById = new Map(
    manifest.validators.map((validator) => [validator.id, validator]),
  );
  const outcomesById = new Map(outcomes.map((outcome) => [outcome.validatorId, outcome]));
  const requiredValidatorIds = uniqueStrings(
    manifest.gates[gateId] ??
      manifest.validators
        .filter((validator) => validator.required)
        .map((validator) => validator.id),
  );
  const missingValidatorIds: string[] = [];
  const passedValidatorIds: string[] = [];
  const failedValidatorIds: string[] = [];
  const waivedValidatorIds: string[] = [];
  const skippedValidatorIds: string[] = [];

  for (const validatorId of requiredValidatorIds) {
    const definition = definitionsById.get(validatorId);
    const outcome = outcomesById.get(validatorId);
    if (!definition || !outcome) {
      missingValidatorIds.push(validatorId);
      continue;
    }

    if (outcome.status === "passed") {
      passedValidatorIds.push(validatorId);
      continue;
    }
    if (outcome.status === "waived" && definition.waiverAllowed) {
      waivedValidatorIds.push(validatorId);
      continue;
    }
    if (outcome.status === "skipped") {
      skippedValidatorIds.push(validatorId);
      continue;
    }
    failedValidatorIds.push(validatorId);
  }

  const status: ValidatorGateDecisionStatus =
    missingValidatorIds.length > 0
      ? "blocked"
      : failedValidatorIds.length > 0 || skippedValidatorIds.length > 0
        ? "failed"
        : "passed";
  const decisionInput = {
    manifestId: manifest.manifestId,
    gateId,
    status,
    requiredValidatorIds,
    missingValidatorIds,
    passedValidatorIds,
    failedValidatorIds,
    waivedValidatorIds,
    skippedValidatorIds,
    reason: validatorGateReason(status, {
      missingValidatorIds,
      failedValidatorIds,
      skippedValidatorIds,
    }),
  } satisfies Omit<ValidatorGateDecision, "decisionId">;

  return ValidatorGateDecisionSchema.parse({
    ...decisionInput,
    decisionId: deterministicId("dec_", decisionInput),
  });
}

export function createAutonomyReplayRecord(
  state: AutonomyReplayState,
  options: CreateAutonomyReplayRecordOptions = {},
): AutonomyReplayRecord {
  const workItems = Object.values(state.workItems)
    .map((workItem) => AutonomyWorkItemSchema.parse(workItem))
    .sort((left, right) => left.id.localeCompare(right.id));
  const workItemStatusCounts = Object.fromEntries(
    AutonomyWorkItemStatusSchema.options.map((status) => [status, 0]),
  ) as Record<AutonomyWorkItemStatus, number>;
  for (const workItem of workItems) {
    workItemStatusCounts[workItem.status] += 1;
  }
  const appliedEventIds = [...state.appliedEventIds].sort();
  const rejectedEventIds = state.rejectedEvents.map((event) => event.eventId).sort();
  const missingExternalRefs = (options.missingExternalRefs ?? []).map((ref) =>
    AutonomySourceRefSchema.parse(ref),
  );
  const stateHash = `sha256:${createHash("sha256")
    .update(stableJson({ appliedEventIds, rejectedEventIds, workItems }))
    .digest("hex")}`;
  const recordInput = {
    eventLogPath: options.eventLogPath ?? AUTONOMY_PATHS.runtimeEvents,
    recordedAt: options.recordedAt,
    eventCount: state.events.length,
    appliedEventIds,
    rejectedEventIds,
    workItemIds: workItems.map((workItem) => workItem.id),
    workItemStatusCounts,
    stateHash,
    missingExternalRefs,
  } satisfies Omit<AutonomyReplayRecord, "replayId">;

  return AutonomyReplayRecordSchema.parse({
    ...recordInput,
    replayId: options.replayId ?? deterministicId("rpl_", recordInput),
  });
}

export function evaluateAutonomySchedule(
  scheduleInput: unknown,
  now: string,
): AutonomyScheduleDecision {
  const schedule = AutonomyScheduleStateSchema.parse(scheduleInput);
  const nowMs = parseAutonomyTimestamp(now);
  if (!schedule.enabled) {
    return AutonomyScheduleDecisionSchema.parse({
      scheduleId: schedule.scheduleId,
      due: false,
      disabled: true,
      now,
      reason: "schedule disabled",
    });
  }

  const dueAt = schedule.nextDueAt
    ? schedule.nextDueAt
    : schedule.lastTriggeredAt
      ? addSecondsIso(schedule.lastTriggeredAt, schedule.cadence.everySeconds)
      : now;
  const dueAtMs = parseAutonomyTimestamp(dueAt);
  const due = nowMs >= dueAtMs;
  const nextDueAt = due ? addSecondsIso(now, schedule.cadence.everySeconds) : dueAt;
  return AutonomyScheduleDecisionSchema.parse({
    scheduleId: schedule.scheduleId,
    due,
    now,
    dueAt,
    nextDueAt,
    idempotencyKey: due ? `schedule:${schedule.scheduleId}:${dueAt}` : undefined,
    reason: due ? "schedule due" : "schedule not due",
  });
}

export function createAutonomyScheduleTrigger(
  scheduleInput: unknown,
  now: string,
  triggerId?: string,
): AutonomyScheduleTrigger {
  const decision = evaluateAutonomySchedule(scheduleInput, now);
  if (!decision.due || !decision.dueAt || !decision.idempotencyKey) {
    throw new Error(`Schedule "${decision.scheduleId}" is not due.`);
  }
  return AutonomyScheduleTriggerSchema.parse({
    triggerId: triggerId ?? deterministicId("trg_", decision),
    triggerType: "schedule_tick",
    scheduleId: decision.scheduleId,
    dueAt: decision.dueAt,
    emittedAt: now,
    idempotencyKey: decision.idempotencyKey,
  });
}

export function defaultReportSchedule(scheduleId = "daily-report"): AutonomyScheduleState {
  return AutonomyScheduleStateSchema.parse({
    scheduleId,
    kind: "report",
    cadence: { everySeconds: DEFAULT_REPORT_CADENCE_HOURS * 60 * 60 },
  });
}

export function wakeupStatePath(role: AutonomyWakeupRole): string {
  return role === "server" ? AUTONOMY_PATHS.serverWakeup : AUTONOMY_PATHS.wakeup;
}

export function evaluateCircuitBreaker(input: {
  workItemId?: string;
  consecutiveFailures: number;
  maxFailures?: number;
  failureSignature?: string;
}): AutonomyCircuitBreakerDecision {
  const maxFailures = input.maxFailures ?? DEFAULT_CIRCUIT_BREAKER_FAILURES;
  const tripped = input.consecutiveFailures >= maxFailures;
  return AutonomyCircuitBreakerDecisionSchema.parse({
    decisionId: deterministicId("dec_", { ...input, maxFailures }),
    workItemId: input.workItemId,
    status: tripped ? "needs_human" : "allow",
    consecutiveFailures: input.consecutiveFailures,
    maxFailures,
    failureSignature: input.failureSignature,
    nextStatus: tripped ? "needs_human" : undefined,
    reason: tripped
      ? `Circuit breaker tripped after ${input.consecutiveFailures} consecutive failures.`
      : "Circuit breaker not tripped.",
  });
}

export function emptyAutonomyReplayState(): AutonomyReplayState {
  return {
    workItems: {},
    events: [],
    rejectedEvents: [],
    appliedEventIds: [],
    appliedIdempotencyKeys: [],
  };
}

export function replayAutonomyEvents(events: unknown[]): AutonomyReplayState {
  return events.reduce<AutonomyReplayState>(
    (state, event) => applyAutonomyRuntimeEvent(state, event),
    emptyAutonomyReplayState(),
  );
}

export function applyAutonomyRuntimeEvent(
  state: AutonomyReplayState,
  input: unknown,
): AutonomyReplayState {
  const event = AutonomyRuntimeEventSchema.parse(input);
  if (state.appliedIdempotencyKeys.includes(event.idempotencyKey)) {
    return state;
  }

  const next = markEventApplied(state, event);
  if (event.eventType === "work_item_created") {
    return applyWorkItemCreated(next, event);
  }

  if (event.workItemId && (event.nextState || payloadPatch(event.payload))) {
    return applyWorkItemTransition(next, event);
  }

  return next;
}

export function canTransitionWorkItemStatus(
  from: AutonomyWorkItemStatus,
  to: AutonomyWorkItemStatus,
): boolean {
  return VALID_STATUS_TRANSITIONS[from].includes(to);
}

export function autonomyDatedPath(
  kind: AutonomyDatedPathKind,
  id: string,
  timestamp: string,
): string {
  const spec = DATED_PATHS[kind];
  if (!id.startsWith(spec.prefix)) {
    throw new Error(`${kind} id must use ${spec.prefix} prefix.`);
  }
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    throw new Error(`Invalid autonomy timestamp "${timestamp}".`);
  }
  const year = String(date.getUTCFullYear()).padStart(4, "0");
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  return `autonomy/${spec.dir}/${year}/${month}/${day}/${id}.${spec.ext}`;
}

function applyWorkItemCreated(
  state: AutonomyReplayState,
  event: AutonomyRuntimeEvent,
): AutonomyReplayState {
  const rawWorkItem = isObjectRecord(event.payload) ? event.payload.workItem : undefined;
  if (!rawWorkItem) {
    return rejectEvent(state, event, "work_item_created requires payload.workItem.");
  }

  const workItem = AutonomyWorkItemSchema.parse(rawWorkItem);
  if (state.workItems[workItem.id]) {
    return rejectEvent(state, event, `Work item "${workItem.id}" already exists.`);
  }

  return {
    ...state,
    workItems: {
      ...state.workItems,
      [workItem.id]: workItem,
    },
  };
}

function applyWorkItemTransition(
  state: AutonomyReplayState,
  event: AutonomyRuntimeEvent,
): AutonomyReplayState {
  const workItemId = event.workItemId;
  if (!workItemId) return state;

  const current = state.workItems[workItemId];
  if (!current) {
    return rejectEvent(state, event, `Unknown work item "${workItemId}".`);
  }

  const nextStatus = event.nextState ?? current.status;
  if (event.previousState && current.status !== event.previousState) {
    return rejectEvent(
      state,
      event,
      `Work item "${workItemId}" is "${current.status}", not "${event.previousState}".`,
    );
  }
  if (!canTransitionWorkItemStatus(current.status, nextStatus)) {
    return rejectEvent(
      state,
      event,
      `Invalid work item transition ${current.status} -> ${nextStatus}.`,
    );
  }

  const patch = payloadPatch(event.payload) ?? {};
  const nextLease =
    "lease" in patch || nextStatus === "leased" || nextStatus === "running"
      ? (patch.lease ?? current.lease)
      : undefined;
  const updated = AutonomyWorkItemSchema.parse({
    ...current,
    ...patch,
    id: current.id,
    status: nextStatus,
    lease: nextLease,
  });

  return {
    ...state,
    workItems: {
      ...state.workItems,
      [workItemId]: updated,
    },
  };
}

function rejectEvent(
  state: AutonomyReplayState,
  event: AutonomyRuntimeEvent,
  reason: string,
): AutonomyReplayState {
  return {
    ...state,
    rejectedEvents: [
      ...state.rejectedEvents,
      createAutonomyRuntimeEvent({
        eventType: "state_transition_rejected",
        timestamp: event.timestamp,
        source: "swarmx.runtime",
        idempotencyKey: `reject:${event.idempotencyKey}`,
        workItemId: event.workItemId,
        runId: event.runId,
        previousState: event.previousState,
        nextState: event.nextState,
        payload: {
          rejectedEventId: event.eventId,
          reason,
        },
      }),
    ],
  };
}

function markEventApplied(
  state: AutonomyReplayState,
  event: AutonomyRuntimeEvent,
): AutonomyReplayState {
  return {
    workItems: { ...state.workItems },
    events: [...state.events, event],
    rejectedEvents: [...state.rejectedEvents],
    appliedEventIds: [...state.appliedEventIds, event.eventId],
    appliedIdempotencyKeys: [...state.appliedIdempotencyKeys, event.idempotencyKey],
  };
}

function payloadPatch(payload: Record<string, unknown>): Record<string, unknown> | undefined {
  const patch = payload.patch;
  return isObjectRecord(patch) ? patch : undefined;
}

function deterministicEventId(input: CreateAutonomyRuntimeEventInput): string {
  const digest = createHash("sha256")
    .update(stableJson({ ...input, payload: input.payload ?? {} }))
    .digest("hex")
    .slice(0, 20);
  return `evt_${digest}`;
}

function deterministicId(prefix: string, input: unknown): string {
  const digest = createHash("sha256").update(stableJson(input)).digest("hex").slice(0, 20);
  return `${prefix}${digest}`;
}

function parseAutonomyTimestamp(timestamp: string): number {
  const value = Date.parse(timestamp);
  if (Number.isNaN(value)) throw new Error(`Invalid autonomy timestamp "${timestamp}".`);
  return value;
}

function addSecondsIso(timestamp: string, seconds: number): string {
  return new Date(parseAutonomyTimestamp(timestamp) + seconds * 1000).toISOString();
}

function validateDagStructure(dag: z.infer<typeof CommandDagSchema>, ctx: z.RefinementCtx): void {
  const seen = new Map<string, number>();
  for (const [index, node] of dag.nodes.entries()) {
    if (seen.has(node.nodeId)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["nodes", index, "nodeId"],
        message: `Duplicate command DAG node "${node.nodeId}".`,
      });
    }
    seen.set(node.nodeId, index);
  }

  for (const [index, node] of dag.nodes.entries()) {
    for (const dependency of node.dependencies) {
      if (!seen.has(dependency)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["nodes", index, "dependencies"],
          message: `Unknown command DAG dependency "${dependency}".`,
        });
      }
      if (dependency === node.nodeId) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["nodes", index, "dependencies"],
          message: `Command DAG node "${node.nodeId}" cannot depend on itself.`,
        });
      }
    }
  }

  const visiting = new Set<string>();
  const visited = new Set<string>();
  const byId = new Map(dag.nodes.map((node) => [node.nodeId, node]));

  const visit = (nodeId: string): boolean => {
    if (visiting.has(nodeId)) return true;
    if (visited.has(nodeId)) return false;
    const node = byId.get(nodeId);
    if (!node) return false;
    visiting.add(nodeId);
    for (const dependency of node.dependencies) {
      if (visit(dependency)) return true;
    }
    visiting.delete(nodeId);
    visited.add(nodeId);
    return false;
  };

  for (const node of dag.nodes) {
    if (visit(node.nodeId)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["nodes"],
        message: "Command DAG must be acyclic.",
      });
      return;
    }
  }
}

function addSecretIssues(value: unknown, ctx: z.RefinementCtx): void {
  for (const issue of findInlineSecretKeys(value)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: issue.path,
      message: `Autonomy records must not contain inline secret field "${issue.key}".`,
    });
  }
}

function addRuntimeRecordIssues(value: unknown, ctx: z.RefinementCtx): void {
  addSecretIssues(value, ctx);
  for (const issue of findRuntimeRawKeys(value)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: issue.path,
      message: `Runtime records must not contain raw content field "${issue.key}".`,
    });
  }
}

function addAgentRunIdentityIssues(value: unknown, ctx: z.RefinementCtx): void {
  addRuntimeRecordIssues(value, ctx);
  if (!isObjectRecord(value)) return;
  for (const key of ["providerProfileId", "provider_profile_id", "model"]) {
    if (key in value) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: [key],
        message: `Agent-run identity must use harnessId plus modelId; field "${key}" is invalid.`,
      });
    }
  }
}

function findInlineSecretKeys(
  value: unknown,
  path: Array<string | number> = [],
): Array<{ key: string; path: Array<string | number> }> {
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => findInlineSecretKeys(item, [...path, index]));
  }
  if (!isObjectRecord(value)) return [];

  const issues: Array<{ key: string; path: Array<string | number> }> = [];
  for (const [key, child] of Object.entries(value)) {
    const normalizedKey = key.toLowerCase().replace(/[^a-z0-9_]/g, "");
    if (
      FORBIDDEN_SECRET_KEY_PATTERN.test(key) &&
      !ALLOWED_SECRET_REFERENCE_KEYS.has(normalizedKey)
    ) {
      issues.push({ key, path: [...path, key] });
    }
    issues.push(...findInlineSecretKeys(child, [...path, key]));
  }
  return issues;
}

function findRuntimeRawKeys(
  value: unknown,
  path: Array<string | number> = [],
): Array<{ key: string; path: Array<string | number> }> {
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => findRuntimeRawKeys(item, [...path, index]));
  }
  if (!isObjectRecord(value)) return [];

  const issues: Array<{ key: string; path: Array<string | number> }> = [];
  for (const [key, child] of Object.entries(value)) {
    if (FORBIDDEN_RUNTIME_RAW_KEY_PATTERN.test(key)) {
      issues.push({ key, path: [...path, key] });
    }
    issues.push(...findRuntimeRawKeys(child, [...path, key]));
  }
  return issues;
}

function transitionPreconditionLabel(precondition: AutonomyTransitionPrecondition): string {
  return precondition.id
    ? `${precondition.kind}:${precondition.id} ${precondition.status}`
    : `${precondition.kind} ${precondition.status}`;
}

function validatorGateReason(
  status: ValidatorGateDecisionStatus,
  ids: {
    missingValidatorIds: string[];
    failedValidatorIds: string[];
    skippedValidatorIds: string[];
  },
): string {
  if (status === "passed") return "Validator gate passed.";
  if (ids.missingValidatorIds.length > 0) {
    return `Validator gate blocked by missing validators: ${ids.missingValidatorIds.join(", ")}.`;
  }
  const failed = [...ids.failedValidatorIds, ...ids.skippedValidatorIds];
  return `Validator gate failed: ${failed.join(", ")}.`;
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values));
}

function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  return `{${Object.entries(value)
    .filter(([, child]) => child !== undefined)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, child]) => `${JSON.stringify(key)}:${stableJson(child)}`)
    .join(",")}}`;
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
