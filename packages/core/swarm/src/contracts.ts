import { z } from "zod";

export const MEMBER_NAME_PATTERN = /^[a-z][a-z0-9-]{0,31}$/u;
export const TASK_ID_PATTERN = /^task-[1-9][0-9]*$/u;

const sessionIdSchema = z.string().min(1).max(500);
const timestampSchema = z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER);
const boundedNameSchema = z.string().trim().min(1).max(100);
const memberNameSchema = z.string().regex(MEMBER_NAME_PATTERN);
const taskIdSchema = z.string().regex(TASK_ID_PATTERN);
const revisionSchema = z.number().int().positive().max(Number.MAX_SAFE_INTEGER);

export const swarmMemberSchema = z.strictObject({
  id: sessionIdSchema,
  name: memberNameSchema,
  role: z.enum(["lead", "member"]),
  phase: z.enum(["provisioning", "active", "failed", "retired"]),
  description: z.string().trim().min(1).max(500),
  createdAt: timestampSchema,
  error: z.string().min(1).max(1_000).optional(),
});

export const swarmTaskSchema = z.strictObject({
  id: taskIdSchema,
  sequence: z.number().int().positive().max(Number.MAX_SAFE_INTEGER),
  revision: revisionSchema,
  subject: z.string().trim().min(1).max(200),
  description: z.string().trim().min(1).max(8_000),
  kind: z.enum(["read", "write"]),
  status: z.enum(["pending", "in_progress", "completed", "failed", "cancelled", "needs_attention"]),
  ownerId: sessionIdSchema.optional(),
  attemptId: z.string().min(1).max(200).optional(),
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
  deliveredAt: timestampSchema.optional(),
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
});

export const createSwarmRequestSchema = z.strictObject({
  name: boundedNameSchema,
});

export const addSwarmMemberRequestSchema = z.strictObject({
  name: memberNameSchema,
  description: z.string().trim().min(1).max(500),
  prompt: z.string().trim().min(1).max(16_000),
});

export const sendSwarmMessageRequestSchema = z.strictObject({
  target: memberNameSchema.or(z.literal("lead")),
  content: z.string().trim().min(1).max(65_536),
  delivery: z.enum(["quiet", "wakeup"]),
});

export const createSwarmTaskRequestSchema = z.strictObject({
  subject: z.string().trim().min(1).max(200),
  description: z.string().trim().min(1).max(8_000),
  kind: z.enum(["read", "write"]),
  assignedTo: memberNameSchema.optional(),
  blockedBy: z.array(taskIdSchema).max(32).default([]),
  writeScopes: z.array(z.string().trim().min(1).max(1_024)).max(32).default([]),
});

export const updateSwarmTaskRequestSchema = z.strictObject({
  taskId: taskIdSchema,
  expectedRevision: revisionSchema,
  attemptId: z.string().min(1).max(200),
  action: z.enum(["complete", "fail", "release"]),
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
  role: z.enum(["lead", "member"]),
  status: z.enum(["running", "idle", "inactive", "provisioning", "failed", "retired"]),
  description: z.string().min(1).max(500),
});

export const swarmTaskViewSchema = z.strictObject({
  id: taskIdSchema,
  revision: revisionSchema,
  subject: z.string().min(1).max(200),
  description: z.string().min(1).max(8_000),
  kind: z.enum(["read", "write"]),
  status: swarmTaskSchema.shape.status,
  ownerName: memberNameSchema.or(z.literal("lead")).optional(),
  attemptId: z.string().min(1).max(200).optional(),
  blockedBy: z.array(taskIdSchema).max(32),
  writeScopes: z.array(z.string().min(1).max(1_024)).max(32),
  ready: z.boolean(),
});

const inactiveSnapshotSchema = z.strictObject({
  kind: z.literal("inactive"),
  revision: z.literal(0),
});

const visibleSnapshotFields = {
  revision: revisionSchema,
  name: boundedNameSchema,
  role: z.enum(["lead", "member"]),
  memberName: memberNameSchema.or(z.literal("lead")),
  members: z.array(swarmMemberViewSchema).max(64),
  tasks: z.array(swarmTaskViewSchema).max(2_048),
  pendingMessages: z.number().int().nonnegative().max(4_096),
  updatedAt: timestampSchema,
} as const;

export const swarmSnapshotSchema = z.discriminatedUnion("kind", [
  inactiveSnapshotSchema,
  z.strictObject({ kind: z.literal("active"), ...visibleSnapshotFields }),
  z.strictObject({ kind: z.literal("archived"), ...visibleSnapshotFields }),
]);

export const swarmUiTaskSchema = swarmTaskViewSchema.omit({
  attemptId: true,
  description: true,
  writeScopes: true,
});
export const swarmUiSnapshotSchema = z.discriminatedUnion("kind", [
  inactiveSnapshotSchema,
  z.strictObject({
    kind: z.literal("active"),
    ...visibleSnapshotFields,
    tasks: z.array(swarmUiTaskSchema).max(2_048),
  }),
  z.strictObject({
    kind: z.literal("archived"),
    ...visibleSnapshotFields,
    tasks: z.array(swarmUiTaskSchema).max(2_048),
  }),
]);

export type AddSwarmMemberRequest = z.infer<typeof addSwarmMemberRequestSchema>;
export type CreateSwarmRequest = z.infer<typeof createSwarmRequestSchema>;
export type CreateSwarmTaskRequest = z.infer<typeof createSwarmTaskRequestSchema>;
export type InterruptSwarmMemberRequest = z.infer<typeof interruptSwarmMemberRequestSchema>;
export type ReassignSwarmTaskRequest = z.infer<typeof reassignSwarmTaskRequestSchema>;
export type SendSwarmMessageRequest = z.infer<typeof sendSwarmMessageRequestSchema>;
export type SwarmMember = z.infer<typeof swarmMemberSchema>;
export type SwarmMessage = z.infer<typeof swarmMessageSchema>;
export type SwarmSnapshot = z.infer<typeof swarmSnapshotSchema>;
export type SwarmTask = z.infer<typeof swarmTaskSchema>;
export type SwarmTaskView = z.infer<typeof swarmTaskViewSchema>;
export type SwarmTeamState = z.infer<typeof swarmTeamStateSchema>;
export type SwarmUiSnapshot = z.infer<typeof swarmUiSnapshotSchema>;
export type UpdateSwarmTaskRequest = z.infer<typeof updateSwarmTaskRequestSchema>;
export type WaitForSwarmChangeRequest = z.infer<typeof waitForSwarmChangeRequestSchema>;
