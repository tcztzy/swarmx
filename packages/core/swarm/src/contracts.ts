import { z } from "zod";

export const MEMBER_NAME_PATTERN = /^[a-z][a-z0-9-]{0,31}$/u;
export const TASK_ID_PATTERN = /^task-[1-9][0-9]*$/u;
export const MODEL_ROUTE_PATTERN = /^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,127}$/u;

const sessionIdSchema = z.string().min(1).max(500);
const timestampSchema = z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER);
const boundedNameSchema = z.string().trim().min(1).max(100);
const memberNameSchema = z.string().regex(MEMBER_NAME_PATTERN);
const taskIdSchema = z.string().regex(TASK_ID_PATTERN);
const revisionSchema = z.number().int().positive().max(Number.MAX_SAFE_INTEGER);
const uuidSchema = z.string().uuid();
const sha256Schema = z.string().regex(/^sha256:[a-f0-9]{64}$/u);
const attemptIdSchema = z.string().min(1).max(200);
const boundedSummarySchema = z.string().trim().min(1).max(2_000);
const auditReferenceSchema = z
  .string()
  .trim()
  .min(1)
  .max(2_048)
  .refine(
    (value) =>
      !value.includes("\0") &&
      !value.startsWith("/") &&
      !value.startsWith("\\\\") &&
      !/^[a-z]:[\\/]/iu.test(value) &&
      !/^file:/iu.test(value),
    "Audit references must not contain absolute host paths",
  );

export const swarmRoleSchema = z.enum([
  "lead",
  "legacy",
  "researcher",
  "implementer",
  "monitor",
  "verifier",
]);

export const agentOptionsSchema = z.strictObject({
  provider: z.string().regex(MODEL_ROUTE_PATTERN).optional(),
  model: z.string().regex(MODEL_ROUTE_PATTERN).optional(),
  maxTokens: z.number().int().positive().max(1_000_000).optional(),
});

export const memberModelPolicySchema = z.strictObject({
  source: z.enum(["legacy-default", "requested", "observed"]),
  provider: z.string().regex(MODEL_ROUTE_PATTERN).optional(),
  model: z.string().regex(MODEL_ROUTE_PATTERN).optional(),
  maxTokens: z.number().int().positive().max(1_000_000).optional(),
});

export const attemptBudgetSchema = z.strictObject({
  maxWallMs: z
    .number()
    .int()
    .min(100)
    .max(7 * 24 * 60 * 60 * 1_000)
    .optional(),
  maxTurns: z.number().int().positive().max(10_000).optional(),
  maxInputTokens: z.number().int().positive().max(1_000_000_000).optional(),
  maxOutputTokens: z.number().int().positive().max(1_000_000_000).optional(),
  warningFraction: z.number().min(0.5).max(0.99).default(0.8),
});

const currentSwarmMemberSchema = z.strictObject({
  id: sessionIdSchema,
  name: memberNameSchema,
  role: swarmRoleSchema,
  phase: z.enum(["provisioning", "active", "failed", "retired"]),
  description: z.string().trim().min(1).max(500),
  createdAt: timestampSchema,
  modelPolicy: memberModelPolicySchema.default({ source: "legacy-default" }),
  budget: attemptBudgetSchema.optional(),
  error: z.string().min(1).max(1_000).optional(),
});

export const swarmMemberSchema = z.preprocess((value) => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return value;
  const input = value as Record<string, unknown>;
  return input.role === "member" ? { ...input, role: "legacy" } : input;
}, currentSwarmMemberSchema);

export const acceptanceCriteriaSchema = z.strictObject({
  summary: boundedSummarySchema,
  requiredChecks: z.array(z.string().trim().min(1).max(200)).max(32).default([]),
  expectedArtifacts: z.array(z.string().trim().min(1).max(200)).max(32).default([]),
  rubric: z.string().trim().min(1).max(4_000).optional(),
});

export const clientSafeLocatorSchema = z.discriminatedUnion("kind", [
  z.strictObject({
    kind: z.literal("science_entity"),
    label: z.string().trim().min(1).max(200),
    entityId: uuidSchema,
  }),
  z.strictObject({
    kind: z.literal("reference"),
    label: z.string().trim().min(1).max(200),
    resource: auditReferenceSchema,
    digest: sha256Schema.optional(),
  }),
]);

export const taskSubmissionSchema = z.strictObject({
  id: uuidSchema,
  attemptId: attemptIdSchema,
  summary: boundedSummarySchema,
  artifactLocators: z.array(clientSafeLocatorSchema).max(32),
  evidenceDigests: z.array(sha256Schema).max(64),
  submittedAt: timestampSchema,
});

export const verificationCheckSchema = z.strictObject({
  name: z.string().trim().min(1).max(200),
  status: z.enum(["pass", "fail", "skipped", "unknown"]),
  digest: sha256Schema.optional(),
});

export const taskVerificationSchema = z.strictObject({
  verifierId: sessionIdSchema,
  verifierName: memberNameSchema.or(z.literal("lead")),
  submissionId: uuidSchema,
  attemptId: attemptIdSchema,
  verdict: z.enum(["pass", "fail", "uncertain", "escalate"]),
  mode: z.enum(["independent", "degraded"]),
  checkResults: z.array(verificationCheckSchema).min(1).max(64),
  rationale: z.string().trim().min(1).max(4_000),
  recordedAt: timestampSchema,
});

export const budgetStateSchema = z.enum(["within", "warning", "exhausted", "unknown"]);

export const attemptUsageSchema = z.strictObject({
  availability: z.enum(["known", "unknown"]),
  inputTokens: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
  outputTokens: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
  cacheReadTokens: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
  cacheWriteTokens: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
  turns: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
  toolCalls: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
});

export const attemptActorUsageSchema = z.strictObject({
  phase: z.enum(["implementation", "verification"]),
  memberName: memberNameSchema.or(z.literal("lead")),
  role: swarmRoleSchema,
  modelPolicy: memberModelPolicySchema,
  observedModel: agentOptionsSchema.pick({ provider: true, model: true }).optional(),
  budget: attemptBudgetSchema.optional(),
  usage: attemptUsageSchema,
  startedAt: timestampSchema,
  endedAt: timestampSchema.optional(),
});

export const swarmAttemptSchema = z.strictObject({
  id: attemptIdSchema,
  revision: revisionSchema.default(1),
  taskId: taskIdSchema,
  taskRevision: revisionSchema,
  ownerId: sessionIdSchema,
  memberName: memberNameSchema.or(z.literal("lead")),
  role: swarmRoleSchema,
  modelPolicy: memberModelPolicySchema,
  observedModel: agentOptionsSchema.pick({ provider: true, model: true }).optional(),
  budget: attemptBudgetSchema.optional(),
  budgetState: budgetStateSchema,
  status: z.enum([
    "active",
    "submitted",
    "verifying",
    "accepted",
    "rejected",
    "escalated",
    "failed",
    "released",
    "interrupted",
    "budget_exhausted",
  ]),
  usage: attemptUsageSchema,
  actors: z.array(attemptActorUsageSchema).max(16).default([]),
  startedAt: timestampSchema,
  lastProgressAt: timestampSchema,
  submittedAt: timestampSchema.optional(),
  verifiedAt: timestampSchema.optional(),
  endedAt: timestampSchema.optional(),
  wallMs: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER).optional(),
  submission: taskSubmissionSchema.optional(),
  verification: taskVerificationSchema.optional(),
  terminalReason: z.string().trim().min(1).max(500).optional(),
  warningCodes: z.array(z.string().trim().min(1).max(100)).max(32),
});

export const monitorFindingCodeSchema = z.enum([
  "attempt_wall_warning",
  "attempt_wall_exhausted",
  "attempt_turns_exhausted",
  "attempt_input_tokens_exhausted",
  "attempt_output_tokens_exhausted",
  "attempt_stalled",
  "write_attempt_stalled",
  "mailbox_near_limit",
  "mailbox_limit_reached",
  "member_lifecycle_failure",
  "role_tool_violation",
  "submission_missing_artifact",
  "submission_missing_evidence",
  "verification_repeated_failure",
  "usage_unknown",
  "usage_unattributed",
  "semantic_submission_concern",
  "semantic_conclusion_conflict",
  "semantic_monitor_delivery_failed",
]);

export const monitorFindingSchema = z.strictObject({
  id: uuidSchema,
  dedupeKey: z.string().trim().min(1).max(500),
  severity: z.enum(["info", "warning", "block", "escalate"]),
  code: monitorFindingCodeSchema,
  subject: z.strictObject({
    kind: z.enum(["member", "task", "attempt", "team"]),
    id: z.string().trim().min(1).max(500),
  }),
  summary: z.string().trim().min(1).max(500),
  action: z.enum(["none", "notify", "interrupt", "needs_attention", "lead_review"]),
  recordedAt: timestampSchema,
});

export const swarmTaskSchema = z.strictObject({
  id: taskIdSchema,
  sequence: z.number().int().positive().max(Number.MAX_SAFE_INTEGER),
  revision: revisionSchema,
  subject: z.string().trim().min(1).max(200),
  description: z.string().trim().min(1).max(8_000),
  kind: z.enum(["read", "write", "knowledge"]),
  status: z.enum([
    "pending",
    "in_progress",
    "submitted",
    "verifying",
    "completed",
    "rejected",
    "escalated",
    "failed",
    "cancelled",
    "needs_attention",
  ]),
  ownerId: sessionIdSchema.optional(),
  attemptId: attemptIdSchema.optional(),
  verifierId: sessionIdSchema.optional(),
  acceptance: acceptanceCriteriaSchema.optional(),
  submission: taskSubmissionSchema.optional(),
  verification: taskVerificationSchema.optional(),
  verificationStartedById: sessionIdSchema.optional(),
  verificationStartedAt: timestampSchema.optional(),
  escalationReason: z.string().trim().min(1).max(500).optional(),
  blockedBy: z.array(taskIdSchema).max(32),
  writeScopes: z.array(z.string().trim().min(1).max(1_024)).max(32),
  createdAt: timestampSchema,
  updatedAt: timestampSchema,
});

export const swarmMessageSchema = z.strictObject({
  id: z.string().min(1).max(200),
  sequence: z.number().int().positive().max(Number.MAX_SAFE_INTEGER),
  senderId: sessionIdSchema,
  senderName: memberNameSchema,
  targetId: sessionIdSchema,
  delivery: z.enum(["quiet", "wakeup"]),
  content: z.string().min(1).max(65_536),
  createdAt: timestampSchema,
  deliveryStartedAt: timestampSchema.optional(),
  deliveredAt: timestampSchema.optional(),
});

export const swarmEffectSchema = z.strictObject({
  id: uuidSchema,
  revision: revisionSchema,
  callId: z.string().min(1).max(500),
  taskId: taskIdSchema,
  taskRevision: revisionSchema,
  attemptId: z.string().min(1).max(200),
  ownerId: sessionIdSchema,
  toolName: z.string().trim().min(1).max(200),
  status: z.enum(["started", "succeeded", "uncertain", "observed", "absent"]),
  createdAt: timestampSchema,
  updatedAt: timestampSchema,
  resultDigest: sha256Schema.optional(),
  verification: z
    .strictObject({
      kind: z.enum(["tool_postcondition", "operator_observation"]),
      reference: auditReferenceSchema,
      digest: sha256Schema.optional(),
      verifiedAt: timestampSchema,
    })
    .optional(),
});

export const evidenceSourceSchema = z.discriminatedUnion("kind", [
  z.strictObject({
    kind: z.literal("science_entity"),
    entityId: uuidSchema,
  }),
  z.strictObject({
    kind: z.literal("reference"),
    resource: auditReferenceSchema,
    title: z.string().trim().min(1).max(500).optional(),
    digest: sha256Schema.optional(),
  }),
]);

export const knowledgeVerificationSchema = z.strictObject({
  status: z.literal("verified"),
  method: z.enum(["reproduced", "source_reviewed", "operator_confirmed"]),
  verifiedAt: timestampSchema,
});

const scienceEvidenceTargetSchema = z.strictObject({
  kind: z.literal("science_evidence"),
  projectId: uuidSchema,
  claimId: uuidSchema,
  relation: z.enum(["supports", "refutes"]),
  title: z.string().trim().min(1).max(500),
  summary: z.string().trim().min(1).max(20_000),
  tags: z.array(z.string().trim().min(1).max(100)).max(64),
});

const pkbConceptTargetSchema = z.strictObject({
  kind: z.literal("pkb_concept"),
  scope: z.enum(["global", "workspace"]),
  title: z.string().trim().min(1).max(500),
  description: z.string().trim().min(1).max(500),
  type: z.string().trim().min(1).max(120),
  body: z.string().trim().min(1).max(65_536),
  tags: z.array(z.string().trim().min(1).max(80)).max(32).optional(),
  aliases: z.array(z.string().trim().min(1).max(200)).max(32).optional(),
  status: z.enum(["draft", "stable"]).optional(),
});

export const admitKnowledgeRequestSchema = z.strictObject({
  admissionId: uuidSchema,
  taskId: taskIdSchema,
  expectedRevision: revisionSchema,
  attemptId: z.string().min(1).max(200),
  sources: z.array(evidenceSourceSchema).min(1).max(32),
  verification: knowledgeVerificationSchema,
  target: z.discriminatedUnion("kind", [scienceEvidenceTargetSchema, pkbConceptTargetSchema]),
});

export const knowledgeCommitReceiptSchema = z.discriminatedUnion("kind", [
  z.strictObject({
    kind: z.literal("science_evidence"),
    entityId: uuidSchema,
    journalSequence: z.number().int().positive().max(Number.MAX_SAFE_INTEGER),
  }),
  z.strictObject({
    kind: z.literal("pkb_concept"),
    conceptId: z.string().min(1).max(1_024),
    revision: sha256Schema,
  }),
]);

export const swarmKnowledgeAdmissionSchema = z.strictObject({
  id: uuidSchema,
  revision: revisionSchema,
  taskId: taskIdSchema,
  taskRevision: revisionSchema,
  attemptId: z.string().min(1).max(200),
  requestHash: sha256Schema,
  targetKind: z.enum(["science_evidence", "pkb_concept"]),
  sources: z.array(evidenceSourceSchema).min(1).max(32),
  verification: knowledgeVerificationSchema,
  status: z.enum(["started", "uncertain", "committed"]),
  createdAt: timestampSchema,
  updatedAt: timestampSchema,
  receipt: knowledgeCommitReceiptSchema.optional(),
});

export const swarmTeamStateSchema = z.strictObject({
  id: sessionIdSchema,
  revision: revisionSchema,
  name: boundedNameSchema,
  workspaceKey: z.string().regex(/^swarmx--[0-9a-f]{64}$/u),
  phase: z.enum(["active", "archived"]),
  createdAt: timestampSchema,
  updatedAt: timestampSchema,
  archivedAt: timestampSchema.optional(),
  members: z.array(swarmMemberSchema).min(1).max(64),
  tasks: z.array(swarmTaskSchema).max(2_048),
  messages: z.array(swarmMessageSchema).max(4_096),
  effects: z.array(swarmEffectSchema).max(4_096),
  admissions: z.array(swarmKnowledgeAdmissionSchema).max(2_048),
  attempts: z.array(swarmAttemptSchema).max(4_096).default([]),
  findings: z.array(monitorFindingSchema).max(2_048).default([]),
});

export const createSwarmRequestSchema = z.strictObject({
  name: boundedNameSchema,
});

export const addSwarmMemberRequestSchema = z.strictObject({
  name: memberNameSchema,
  description: z.string().trim().min(1).max(500),
  prompt: z.string().trim().min(1).max(16_000),
  role: swarmRoleSchema.exclude(["lead"]).default("legacy"),
  agentOptions: agentOptionsSchema.optional(),
  budget: attemptBudgetSchema.optional(),
});

export const sendSwarmMessageRequestSchema = z.strictObject({
  target: memberNameSchema.or(z.literal("lead")),
  content: z.string().trim().min(1).max(65_536),
  delivery: z.enum(["quiet", "wakeup"]),
  idempotencyKey: uuidSchema.optional(),
});

export const createSwarmTaskRequestSchema = z.strictObject({
  subject: z.string().trim().min(1).max(200),
  description: z.string().trim().min(1).max(8_000),
  kind: z.enum(["read", "write", "knowledge"]),
  assignedTo: memberNameSchema.optional(),
  verifier: memberNameSchema.optional(),
  blockedBy: z.array(taskIdSchema).max(32).default([]),
  writeScopes: z.array(z.string().trim().min(1).max(1_024)).max(32).default([]),
  acceptance: acceptanceCriteriaSchema.optional(),
});

export const updateSwarmTaskRequestSchema = z.strictObject({
  taskId: taskIdSchema,
  expectedRevision: revisionSchema,
  attemptId: z.string().min(1).max(200),
  action: z.enum(["complete", "fail", "release"]),
});

export const submitSwarmTaskRequestSchema = z.strictObject({
  taskId: taskIdSchema,
  expectedRevision: revisionSchema,
  attemptId: attemptIdSchema,
  summary: boundedSummarySchema,
  artifactLocators: z.array(clientSafeLocatorSchema).max(32),
  evidenceDigests: z.array(sha256Schema).max(64),
});

export const startSwarmVerificationRequestSchema = z.strictObject({
  taskId: taskIdSchema,
  expectedRevision: revisionSchema,
  attemptId: attemptIdSchema,
  submissionId: uuidSchema,
});

export const recordSwarmVerdictRequestSchema = z.strictObject({
  taskId: taskIdSchema,
  expectedRevision: revisionSchema,
  attemptId: attemptIdSchema,
  submissionId: uuidSchema,
  verdict: z.enum(["pass", "fail", "uncertain", "escalate"]),
  checkResults: z.array(verificationCheckSchema).min(1).max(64),
  rationale: z.string().trim().min(1).max(4_000),
});

export const escalateSwarmTaskRequestSchema = z.strictObject({
  taskId: taskIdSchema,
  expectedRevision: revisionSchema,
  attemptId: attemptIdSchema,
  submissionId: uuidSchema.optional(),
  reason: z.string().trim().min(1).max(500),
});

export const recordSemanticFindingRequestSchema = z.strictObject({
  triggerId: uuidSchema,
  severity: z.enum(["info", "warning", "escalate"]),
  code: z.enum(["semantic_submission_concern", "semantic_conclusion_conflict"]),
  subject: z.strictObject({
    kind: z.enum(["task", "team"]),
    id: z.string().trim().min(1).max(500),
  }),
  summary: z.string().trim().min(1).max(500),
  action: z.enum(["none", "notify", "lead_review"]),
});

export const resolveSwarmEffectRequestSchema = z.strictObject({
  effectId: uuidSchema,
  taskId: taskIdSchema,
  expectedRevision: revisionSchema,
  attemptId: z.string().min(1).max(200),
  resolution: z.enum(["observed", "absent"]),
  verification: z.strictObject({
    kind: z.enum(["tool_postcondition", "operator_observation"]),
    reference: auditReferenceSchema,
    digest: sha256Schema.optional(),
    verifiedAt: timestampSchema,
  }),
});

export const reassignSwarmTaskRequestSchema = z.strictObject({
  taskId: taskIdSchema,
  expectedRevision: revisionSchema,
  target: memberNameSchema,
});

export const interruptSwarmMemberRequestSchema = z.strictObject({
  target: memberNameSchema,
});

export const waitForSwarmChangeRequestSchema = z.strictObject({
  afterRevision: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
  timeoutMs: z.number().int().min(10_000).max(60_000).default(30_000),
});

export const swarmMemberViewSchema = z.strictObject({
  name: memberNameSchema,
  role: swarmRoleSchema,
  status: z.enum(["running", "idle", "inactive", "provisioning", "failed", "retired"]),
  description: z.string().min(1).max(500),
  modelLabel: z.string().trim().min(1).max(260).default("deployment default"),
  budgetState: budgetStateSchema.default("unknown"),
});

export const attemptUsageSummarySchema = attemptUsageSchema.extend({
  wallMs: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
});

export const swarmTaskViewSchema = z.strictObject({
  id: taskIdSchema,
  revision: revisionSchema,
  subject: z.string().min(1).max(200),
  description: z.string().min(1).max(8_000),
  kind: z.enum(["read", "write", "knowledge"]),
  status: swarmTaskSchema.shape.status,
  ownerName: memberNameSchema.or(z.literal("lead")).optional(),
  verifierName: memberNameSchema.or(z.literal("lead")).optional(),
  attemptId: attemptIdSchema.optional(),
  blockedBy: z.array(taskIdSchema).max(32),
  writeScopes: z.array(z.string().min(1).max(1_024)).max(32),
  ready: z.boolean(),
  acceptance: acceptanceCriteriaSchema.optional(),
  submission: taskSubmissionSchema.optional(),
  verification: taskVerificationSchema.optional(),
  budgetState: budgetStateSchema.default("unknown"),
  usage: attemptUsageSummarySchema.optional(),
  escalationReason: z.string().trim().min(1).max(500).optional(),
});

export const swarmEffectViewSchema = swarmEffectSchema.pick({
  id: true,
  taskId: true,
  attemptId: true,
  toolName: true,
  status: true,
});

export const swarmAdmissionViewSchema = swarmKnowledgeAdmissionSchema.pick({
  id: true,
  taskId: true,
  attemptId: true,
  targetKind: true,
  status: true,
  receipt: true,
});

export const swarmFindingViewSchema = monitorFindingSchema.pick({
  severity: true,
  code: true,
  summary: true,
  action: true,
  recordedAt: true,
});

const inactiveSnapshotSchema = z.strictObject({
  kind: z.literal("inactive"),
  revision: z.literal(0),
});

const visibleSnapshotFields = {
  revision: revisionSchema,
  name: boundedNameSchema,
  role: swarmRoleSchema,
  memberName: memberNameSchema.or(z.literal("lead")),
  members: z.array(swarmMemberViewSchema).max(64),
  tasks: z.array(swarmTaskViewSchema).max(2_048),
  pendingMessages: z.number().int().nonnegative().max(4_096),
  findings: z.array(swarmFindingViewSchema).max(100).default([]),
  updatedAt: timestampSchema,
} as const;

export const swarmSnapshotSchema = z.discriminatedUnion("kind", [
  inactiveSnapshotSchema,
  z.strictObject({
    kind: z.literal("active"),
    ...visibleSnapshotFields,
    effects: z.array(swarmEffectViewSchema).max(4_096),
    admissions: z.array(swarmAdmissionViewSchema).max(2_048),
  }),
  z.strictObject({
    kind: z.literal("archived"),
    ...visibleSnapshotFields,
    effects: z.array(swarmEffectViewSchema).max(4_096),
    admissions: z.array(swarmAdmissionViewSchema).max(2_048),
  }),
]);

export const swarmUiTaskSchema = z.strictObject({
  id: taskIdSchema,
  revision: revisionSchema,
  subject: z.string().min(1).max(200),
  kind: z.enum(["read", "write", "knowledge"]),
  status: swarmTaskSchema.shape.status,
  ownerName: memberNameSchema.or(z.literal("lead")).optional(),
  verifierName: memberNameSchema.or(z.literal("lead")).optional(),
  blockedBy: z.array(taskIdSchema).max(32),
  ready: z.boolean(),
  budgetState: budgetStateSchema.default("unknown"),
  usage: attemptUsageSummarySchema.optional(),
  submission: z
    .strictObject({
      summary: boundedSummarySchema,
      artifactCount: z.number().int().nonnegative().max(32),
      evidenceCount: z.number().int().nonnegative().max(64),
      submittedAt: timestampSchema,
    })
    .optional(),
  verification: z
    .strictObject({
      verifierName: memberNameSchema.or(z.literal("lead")),
      verdict: z.enum(["pass", "fail", "uncertain", "escalate"]),
      mode: z.enum(["independent", "degraded"]),
      checkResults: z.array(verificationCheckSchema).max(64),
      rationale: z.string().trim().min(1).max(4_000),
      recordedAt: timestampSchema,
    })
    .optional(),
  escalationReason: z.string().trim().min(1).max(500).optional(),
});
const visibleUiSnapshotFields = {
  ...visibleSnapshotFields,
  tasks: z.array(swarmUiTaskSchema).max(2_048),
} as const;
export const swarmUiSnapshotSchema = z.discriminatedUnion("kind", [
  inactiveSnapshotSchema,
  z.strictObject({
    kind: z.literal("active"),
    ...visibleUiSnapshotFields,
  }),
  z.strictObject({
    kind: z.literal("archived"),
    ...visibleUiSnapshotFields,
  }),
]);

export type AddSwarmMemberRequest = z.infer<typeof addSwarmMemberRequestSchema>;
export type AdmitKnowledgeRequest = z.infer<typeof admitKnowledgeRequestSchema>;
export type CreateSwarmRequest = z.infer<typeof createSwarmRequestSchema>;
export type CreateSwarmTaskRequest = z.infer<typeof createSwarmTaskRequestSchema>;
export type InterruptSwarmMemberRequest = z.infer<typeof interruptSwarmMemberRequestSchema>;
export type EscalateSwarmTaskRequest = z.infer<typeof escalateSwarmTaskRequestSchema>;
export type RecordSwarmVerdictRequest = z.infer<typeof recordSwarmVerdictRequestSchema>;
export type RecordSemanticFindingRequest = z.infer<typeof recordSemanticFindingRequestSchema>;
export type ReassignSwarmTaskRequest = z.infer<typeof reassignSwarmTaskRequestSchema>;
export type ResolveSwarmEffectRequest = z.infer<typeof resolveSwarmEffectRequestSchema>;
export type SendSwarmMessageRequest = z.infer<typeof sendSwarmMessageRequestSchema>;
export type SwarmMember = z.infer<typeof swarmMemberSchema>;
export type SwarmRole = z.infer<typeof swarmRoleSchema>;
export type SwarmAttempt = z.infer<typeof swarmAttemptSchema>;
export type MonitorFinding = z.infer<typeof monitorFindingSchema>;
export type AttemptBudget = z.infer<typeof attemptBudgetSchema>;
export type AttemptUsage = z.infer<typeof attemptUsageSchema>;
export type SwarmEffect = z.infer<typeof swarmEffectSchema>;
export type SwarmKnowledgeAdmission = z.infer<typeof swarmKnowledgeAdmissionSchema>;
export type EvidenceSource = z.infer<typeof evidenceSourceSchema>;
export type KnowledgeCommitReceipt = z.infer<typeof knowledgeCommitReceiptSchema>;
export type SwarmMessage = z.infer<typeof swarmMessageSchema>;
export type SwarmSnapshot = z.infer<typeof swarmSnapshotSchema>;
export type SwarmTask = z.infer<typeof swarmTaskSchema>;
export type SwarmTaskView = z.infer<typeof swarmTaskViewSchema>;
export type SwarmTeamState = z.infer<typeof swarmTeamStateSchema>;
export type SwarmUiSnapshot = z.infer<typeof swarmUiSnapshotSchema>;
export type StartSwarmVerificationRequest = z.infer<typeof startSwarmVerificationRequestSchema>;
export type SubmitSwarmTaskRequest = z.infer<typeof submitSwarmTaskRequestSchema>;
export type UpdateSwarmTaskRequest = z.infer<typeof updateSwarmTaskRequestSchema>;
export type WaitForSwarmChangeRequest = z.infer<typeof waitForSwarmChangeRequestSchema>;
