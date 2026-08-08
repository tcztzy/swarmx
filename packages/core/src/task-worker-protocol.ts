import { z } from "zod";

export const TASK_WORKER_PROTOCOL_VERSION = 1 as const;
export const TASK_WORKER_MAX_JSONL_LINE_BYTES = 1024 * 1024;

const SAFE_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]*$/;
const SHA256_PATTERN = /^sha256:[a-f0-9]{64}$/;
const MAX_ID_LENGTH = 160;
const MAX_SHORT_TEXT_LENGTH = 4_096;
const MAX_SUMMARY_LENGTH = 16_384;

const SafeIdSchema = z.string().min(1).max(MAX_ID_LENGTH).regex(SAFE_ID_PATTERN);
const TimestampSchema = z.string().datetime();
const Sha256Schema = z.string().regex(SHA256_PATTERN);
const WorkItemIdSchema = z.string().regex(/^awi_[A-Za-z0-9][A-Za-z0-9_-]*$/);
const RunIdSchema = z.string().regex(/^run_[A-Za-z0-9][A-Za-z0-9_-]*$/);
const LeaseIdSchema = z.string().regex(/^lease_[A-Za-z0-9][A-Za-z0-9_-]*$/);
const CheckpointIdSchema = z.string().regex(/^ckp_[A-Za-z0-9][A-Za-z0-9_-]*$/);
const ArtifactIdSchema = z.string().regex(/^art_[A-Za-z0-9][A-Za-z0-9_-]*$/);
const ReceiptIdSchema = z.string().regex(/^rcpt_[A-Za-z0-9][A-Za-z0-9_-]*$/);

export type TaskWorkerJsonValue =
  | null
  | boolean
  | number
  | string
  | TaskWorkerJsonValue[]
  | { [key: string]: TaskWorkerJsonValue };

export const TaskWorkerJsonValueSchema: z.ZodType<TaskWorkerJsonValue> = z.lazy(() =>
  z.union([
    z.null(),
    z.boolean(),
    z.number().finite(),
    z.string(),
    z.array(TaskWorkerJsonValueSchema),
    z.record(z.string(), TaskWorkerJsonValueSchema),
  ]),
);

/** JSON payloads that may cross the worker boundary or enter durable task blobs. */
export const TaskWorkerPayloadSchema = TaskWorkerJsonValueSchema.superRefine(addInlineSecretIssues);

export const TaskWorkerMessageDirectionSchema = z.enum(["host_to_worker", "worker_to_host"]);

export const TaskWorkerFeatureSchema = z.enum([
  "heartbeat",
  "progress",
  "checkpoint",
  "artifact",
  "needs_human",
  "cancel",
  "capability_gateway",
]);

export const TaskWorkerArtifactReferenceSchema = z
  .object({
    artifactId: ArtifactIdSchema,
    kind: z.string().min(1).max(128),
    relativePath: z.string().min(1).max(4_096),
    sha256: Sha256Schema,
    sizeBytes: z.number().int().nonnegative(),
    mediaType: z.string().min(1).max(256).optional(),
    metadata: z.record(z.string(), TaskWorkerJsonValueSchema).optional(),
  })
  .strict()
  .superRefine((artifact, ctx) => {
    if (!isSafeRelativePath(artifact.relativePath)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["relativePath"],
        message: "Worker artifact paths must be safe relative paths.",
      });
    }
  });

export const TaskWorkerCheckpointSchema = z
  .object({
    checkpointId: CheckpointIdSchema,
    format: z.string().min(1).max(128),
    formatVersion: z.number().int().positive(),
    environmentDigest: Sha256Schema,
    state: TaskWorkerJsonValueSchema.optional(),
    artifact: TaskWorkerArtifactReferenceSchema.optional(),
  })
  .strict()
  .superRefine((checkpoint, ctx) => {
    const representations = Number(checkpoint.state !== undefined) + Number(!!checkpoint.artifact);
    if (representations !== 1) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["state"],
        message: "A checkpoint must contain exactly one inline state or artifact reference.",
      });
    }
  });

export const TaskWorkerCapabilityGrantSchema = z
  .object({
    grantId: SafeIdSchema,
    capabilityId: SafeIdSchema,
    operations: z.array(SafeIdSchema).min(1).max(64),
    expiresAt: TimestampSchema.optional(),
  })
  .strict()
  .superRefine((grant, ctx) => addDuplicateIssues(grant.operations, ctx, ["operations"]));

export const TaskWorkerHumanDecisionSchema = z
  .object({
    approvalId: z.string().regex(/^apr_[A-Za-z0-9][A-Za-z0-9_-]*$/),
    status: z.enum(["approved", "rejected", "waived"]),
    decidedAt: TimestampSchema,
    decidedBy: z.string().min(1).max(256),
    reason: z.string().min(1).max(MAX_SUMMARY_LENGTH).optional(),
    response: TaskWorkerPayloadSchema.optional(),
  })
  .strict();

const MessageBaseShape = {
  protocolVersion: z.literal(TASK_WORKER_PROTOCOL_VERSION),
  messageId: SafeIdSchema,
};

const RunMessageShape = {
  workItemId: WorkItemIdSchema,
  runId: RunIdSchema,
  leaseId: LeaseIdSchema,
  fencingToken: z.number().int().positive(),
};

const WorkerRunEventShape = {
  ...MessageBaseShape,
  direction: z.literal("worker_to_host"),
  ...RunMessageShape,
  sequence: z.number().int().nonnegative(),
  emittedAt: TimestampSchema,
};

export const TaskWorkerHelloMessageSchema = protocolMessageSchema({
  ...MessageBaseShape,
  direction: z.literal("worker_to_host"),
  type: z.literal("hello"),
  worker: z
    .object({
      instanceId: SafeIdSchema,
      backendId: SafeIdSchema,
      backendVersion: z.string().min(1).max(128),
      language: z.string().min(1).max(64),
      languageVersion: z.string().min(1).max(128),
      environmentDigest: Sha256Schema,
    })
    .strict(),
  supportedProtocolVersions: z.array(z.literal(TASK_WORKER_PROTOCOL_VERSION)).min(1).max(8),
  operations: z.array(SafeIdSchema).min(1).max(256),
  features: z.array(TaskWorkerFeatureSchema).max(TaskWorkerFeatureSchema.options.length),
});

export const TaskWorkerCapabilitiesMessageSchema = protocolMessageSchema({
  ...MessageBaseShape,
  direction: z.literal("host_to_worker"),
  type: z.literal("capabilities"),
  helloMessageId: SafeIdSchema,
  selectedProtocolVersion: z.literal(TASK_WORKER_PROTOCOL_VERSION),
  enabledFeatures: z.array(TaskWorkerFeatureSchema).max(TaskWorkerFeatureSchema.options.length),
  grants: z.array(TaskWorkerCapabilityGrantSchema).max(256),
  limits: z
    .object({
      maxJsonlLineBytes: z.number().int().positive().max(TASK_WORKER_MAX_JSONL_LINE_BYTES),
      heartbeatIntervalMs: z.number().int().positive(),
      heartbeatTimeoutMs: z.number().int().positive(),
      maxArtifactBytes: z.number().int().nonnegative(),
    })
    .strict(),
});

export const TaskWorkerStartMessageSchema = protocolMessageSchema({
  ...MessageBaseShape,
  direction: z.literal("host_to_worker"),
  type: z.literal("start"),
  ...RunMessageShape,
  attempt: z.number().int().positive(),
  operation: z
    .object({
      name: SafeIdSchema,
      input: TaskWorkerJsonValueSchema,
    })
    .strict(),
  environmentDigest: Sha256Schema,
  resumeFrom: TaskWorkerCheckpointSchema.optional(),
  humanDecisions: z.array(TaskWorkerHumanDecisionSchema).max(256).optional(),
  capabilityGrantIds: z.array(SafeIdSchema).max(256),
  budget: z
    .object({
      wallTimeMs: z.number().int().positive().optional(),
      outputBytes: z.number().int().nonnegative().optional(),
      capabilityCalls: z
        .record(z.string().min(1).max(128), z.number().int().nonnegative())
        .optional(),
    })
    .strict()
    .optional(),
}).superRefine((message, ctx) => {
  if (message.resumeFrom && message.resumeFrom.environmentDigest !== message.environmentDigest) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["resumeFrom", "environmentDigest"],
      message: "Resume checkpoint environment must match the start environment.",
    });
  }
});

export const TaskWorkerHeartbeatMessageSchema = protocolMessageSchema({
  ...WorkerRunEventShape,
  type: z.literal("heartbeat"),
});

export const TaskWorkerProgressMessageSchema = protocolMessageSchema({
  ...WorkerRunEventShape,
  type: z.literal("progress"),
  message: z.string().min(1).max(MAX_SHORT_TEXT_LENGTH).optional(),
  fraction: z.number().min(0).max(1).optional(),
  counters: z.record(z.string().min(1).max(128), z.number().finite()).optional(),
});

export const TaskWorkerCheckpointMessageSchema = protocolMessageSchema({
  ...WorkerRunEventShape,
  type: z.literal("checkpoint"),
  idempotencyKey: z.string().min(1).max(512),
  checkpoint: TaskWorkerCheckpointSchema,
});

export const TaskWorkerArtifactMessageSchema = protocolMessageSchema({
  ...WorkerRunEventShape,
  type: z.literal("artifact"),
  idempotencyKey: z.string().min(1).max(512),
  artifact: TaskWorkerArtifactReferenceSchema,
});

export const TaskWorkerNeedsHumanMessageSchema = protocolMessageSchema({
  ...WorkerRunEventShape,
  type: z.literal("needs_human"),
  idempotencyKey: z.string().min(1).max(512),
  request: z
    .object({
      requestId: SafeIdSchema,
      kind: z.enum(["approval", "question"]),
      prompt: z.string().min(1).max(MAX_SUMMARY_LENGTH),
      options: z
        .array(
          z
            .object({
              optionId: SafeIdSchema,
              label: z.string().min(1).max(256),
              description: z.string().min(1).max(MAX_SHORT_TEXT_LENGTH).optional(),
            })
            .strict(),
        )
        .max(32),
      checkpointId: CheckpointIdSchema.optional(),
    })
    .strict(),
});

export const TaskWorkerCompleteMessageSchema = protocolMessageSchema({
  ...WorkerRunEventShape,
  type: z.literal("complete"),
  idempotencyKey: z.string().min(1).max(512),
  summary: z.string().max(MAX_SUMMARY_LENGTH).optional(),
  result: TaskWorkerJsonValueSchema.optional(),
  artifactIds: z.array(ArtifactIdSchema).max(1_024),
  checkpointId: CheckpointIdSchema.optional(),
});

export const TaskWorkerFailMessageSchema = protocolMessageSchema({
  ...WorkerRunEventShape,
  type: z.literal("fail"),
  idempotencyKey: z.string().min(1).max(512),
  failure: z
    .object({
      code: SafeIdSchema,
      message: z.string().min(1).max(MAX_SUMMARY_LENGTH),
      retryable: z.boolean(),
      details: TaskWorkerJsonValueSchema.optional(),
    })
    .strict(),
  checkpointId: CheckpointIdSchema.optional(),
});

export const TaskWorkerCancelMessageSchema = protocolMessageSchema({
  ...MessageBaseShape,
  direction: z.literal("host_to_worker"),
  type: z.literal("cancel"),
  ...RunMessageShape,
  requestedAt: TimestampSchema,
  mode: z.enum(["cancel", "interrupt"]),
  reason: z.string().min(1).max(MAX_SHORT_TEXT_LENGTH),
  graceMs: z.number().int().nonnegative(),
});

export const TaskWorkerCanceledMessageSchema = protocolMessageSchema({
  ...WorkerRunEventShape,
  type: z.literal("canceled"),
  idempotencyKey: z.string().min(1).max(512),
  mode: z.enum(["cancel", "interrupt"]),
  reason: z.string().max(MAX_SHORT_TEXT_LENGTH).optional(),
  checkpointId: CheckpointIdSchema.optional(),
});

export const TaskWorkerCapabilityCallMessageSchema = protocolMessageSchema({
  ...WorkerRunEventShape,
  type: z.literal("capability_call"),
  callId: SafeIdSchema,
  grantId: SafeIdSchema,
  capabilityId: SafeIdSchema,
  operation: SafeIdSchema,
  idempotencyKey: z.string().min(1).max(512),
  arguments: TaskWorkerJsonValueSchema,
});

const CapabilitySuccessSchema = z
  .object({
    status: z.literal("succeeded"),
    value: TaskWorkerJsonValueSchema.optional(),
    artifactIds: z.array(ArtifactIdSchema).max(1_024),
    receipt: z
      .object({
        receiptId: ReceiptIdSchema,
        idempotencyKey: z.string().min(1).max(512),
        externalRef: z.string().min(1).max(4_096).optional(),
      })
      .strict()
      .optional(),
  })
  .strict();

const CapabilityFailureSchema = z
  .object({
    status: z.enum(["failed", "unknown"]),
    error: z
      .object({
        code: SafeIdSchema,
        message: z.string().min(1).max(MAX_SUMMARY_LENGTH),
        retryable: z.boolean(),
      })
      .strict(),
  })
  .strict();

export const TaskWorkerCapabilityOutcomeSchema = z.discriminatedUnion("status", [
  CapabilitySuccessSchema,
  CapabilityFailureSchema,
]);

export const TaskWorkerCapabilityResultMessageSchema = protocolMessageSchema({
  ...MessageBaseShape,
  direction: z.literal("host_to_worker"),
  type: z.literal("capability_result"),
  ...RunMessageShape,
  callId: SafeIdSchema,
  grantId: SafeIdSchema,
  capabilityId: SafeIdSchema,
  outcome: TaskWorkerCapabilityOutcomeSchema,
});

export const TaskWorkerHostToWorkerMessageSchema = z.discriminatedUnion("type", [
  TaskWorkerCapabilitiesMessageSchema,
  TaskWorkerStartMessageSchema,
  TaskWorkerCancelMessageSchema,
  TaskWorkerCapabilityResultMessageSchema,
]);

export const TaskWorkerControlMessageSchema = TaskWorkerHostToWorkerMessageSchema;

export const TaskWorkerWorkerToHostMessageSchema = z.discriminatedUnion("type", [
  TaskWorkerHelloMessageSchema,
  TaskWorkerHeartbeatMessageSchema,
  TaskWorkerProgressMessageSchema,
  TaskWorkerCheckpointMessageSchema,
  TaskWorkerArtifactMessageSchema,
  TaskWorkerNeedsHumanMessageSchema,
  TaskWorkerCompleteMessageSchema,
  TaskWorkerFailMessageSchema,
  TaskWorkerCapabilityCallMessageSchema,
  TaskWorkerCanceledMessageSchema,
]);

export const TaskWorkerEventMessageSchema = TaskWorkerWorkerToHostMessageSchema;

export const TaskWorkerProtocolMessageSchema = z.discriminatedUnion("type", [
  TaskWorkerCapabilitiesMessageSchema,
  TaskWorkerStartMessageSchema,
  TaskWorkerCancelMessageSchema,
  TaskWorkerCapabilityResultMessageSchema,
  TaskWorkerHelloMessageSchema,
  TaskWorkerHeartbeatMessageSchema,
  TaskWorkerProgressMessageSchema,
  TaskWorkerCheckpointMessageSchema,
  TaskWorkerArtifactMessageSchema,
  TaskWorkerNeedsHumanMessageSchema,
  TaskWorkerCompleteMessageSchema,
  TaskWorkerFailMessageSchema,
  TaskWorkerCapabilityCallMessageSchema,
  TaskWorkerCanceledMessageSchema,
]);

export type TaskWorkerMessageDirection = z.infer<typeof TaskWorkerMessageDirectionSchema>;
export type TaskWorkerFeature = z.infer<typeof TaskWorkerFeatureSchema>;
export type TaskWorkerArtifactReference = z.infer<typeof TaskWorkerArtifactReferenceSchema>;
export type TaskWorkerCheckpoint = z.infer<typeof TaskWorkerCheckpointSchema>;
export type TaskWorkerCapabilityGrant = z.infer<typeof TaskWorkerCapabilityGrantSchema>;
export type TaskWorkerHumanDecision = z.infer<typeof TaskWorkerHumanDecisionSchema>;
export type TaskWorkerHelloMessage = z.infer<typeof TaskWorkerHelloMessageSchema>;
export type TaskWorkerCapabilitiesMessage = z.infer<typeof TaskWorkerCapabilitiesMessageSchema>;
export type TaskWorkerStartMessage = z.infer<typeof TaskWorkerStartMessageSchema>;
export type TaskWorkerHeartbeatMessage = z.infer<typeof TaskWorkerHeartbeatMessageSchema>;
export type TaskWorkerProgressMessage = z.infer<typeof TaskWorkerProgressMessageSchema>;
export type TaskWorkerCheckpointMessage = z.infer<typeof TaskWorkerCheckpointMessageSchema>;
export type TaskWorkerArtifactMessage = z.infer<typeof TaskWorkerArtifactMessageSchema>;
export type TaskWorkerNeedsHumanMessage = z.infer<typeof TaskWorkerNeedsHumanMessageSchema>;
export type TaskWorkerCompleteMessage = z.infer<typeof TaskWorkerCompleteMessageSchema>;
export type TaskWorkerFailMessage = z.infer<typeof TaskWorkerFailMessageSchema>;
export type TaskWorkerCancelMessage = z.infer<typeof TaskWorkerCancelMessageSchema>;
export type TaskWorkerCanceledMessage = z.infer<typeof TaskWorkerCanceledMessageSchema>;
export type TaskWorkerCapabilityCallMessage = z.infer<typeof TaskWorkerCapabilityCallMessageSchema>;
export type TaskWorkerCapabilityResultMessage = z.infer<
  typeof TaskWorkerCapabilityResultMessageSchema
>;
export type TaskWorkerHostToWorkerMessage = z.infer<typeof TaskWorkerHostToWorkerMessageSchema>;
export type TaskWorkerWorkerToHostMessage = z.infer<typeof TaskWorkerWorkerToHostMessageSchema>;
export type TaskWorkerControlMessage = z.infer<typeof TaskWorkerControlMessageSchema>;
export type TaskWorkerEventMessage = z.infer<typeof TaskWorkerEventMessageSchema>;
export type TaskWorkerProtocolMessage = z.infer<typeof TaskWorkerProtocolMessageSchema>;

export function parseTaskWorkerMessage(
  input: unknown,
  expectedDirection?: TaskWorkerMessageDirection,
): TaskWorkerProtocolMessage {
  if (expectedDirection === "host_to_worker") {
    return TaskWorkerHostToWorkerMessageSchema.parse(input);
  }
  if (expectedDirection === "worker_to_host") {
    return TaskWorkerWorkerToHostMessageSchema.parse(input);
  }
  return TaskWorkerProtocolMessageSchema.parse(input);
}

export function parseTaskWorkerJsonlLine(
  line: string,
  expectedDirection: TaskWorkerMessageDirection,
): TaskWorkerProtocolMessage {
  if (!line || line.includes("\n") || line.includes("\r")) {
    throw new Error("Task worker protocol input must contain exactly one non-empty JSONL line.");
  }
  assertJsonlLineSize(line);
  let input: unknown;
  try {
    input = JSON.parse(line);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Invalid task worker protocol JSON: ${message}`);
  }
  return parseTaskWorkerMessage(input, expectedDirection);
}

export function parseTaskWorkerControlLine(line: string): TaskWorkerControlMessage {
  return TaskWorkerControlMessageSchema.parse(parseTaskWorkerJsonlLine(line, "host_to_worker"));
}

export function parseTaskWorkerEventLine(line: string): TaskWorkerEventMessage {
  return TaskWorkerEventMessageSchema.parse(parseTaskWorkerJsonlLine(line, "worker_to_host"));
}

export function parseTaskWorkerJsonl(
  input: string,
  expectedDirection: TaskWorkerMessageDirection,
): TaskWorkerProtocolMessage[] {
  const lines = input.split("\n");
  if (lines.at(-1) === "") lines.pop();
  if (lines.length === 0) throw new Error("Task worker protocol JSONL input is empty.");

  return lines.map((rawLine, index) => {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (!line) throw new Error(`Task worker protocol JSONL line ${index + 1} is empty.`);
    try {
      return parseTaskWorkerJsonlLine(line, expectedDirection);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Invalid task worker protocol JSONL line ${index + 1}: ${message}`);
    }
  });
}

export function taskWorkerJsonlLine(input: unknown): string {
  const message = TaskWorkerProtocolMessageSchema.parse(input);
  const line = JSON.stringify(message);
  assertJsonlLineSize(line);
  return `${line}\n`;
}

export const serializeTaskWorkerMessage = taskWorkerJsonlLine;

export function assertTaskWorkerCapabilityCallAllowed(
  input: unknown,
  grantsInput: readonly unknown[],
  now = new Date(),
): TaskWorkerCapabilityCallMessage {
  const call = TaskWorkerCapabilityCallMessageSchema.parse(input);
  const grants = grantsInput.map((grant) => TaskWorkerCapabilityGrantSchema.parse(grant));
  const grant = grants.find((candidate) => candidate.grantId === call.grantId);
  if (!grant) throw new Error(`Unknown task worker capability grant "${call.grantId}".`);
  if (grant.capabilityId !== call.capabilityId) {
    throw new Error(
      `Capability grant "${call.grantId}" does not authorize "${call.capabilityId}".`,
    );
  }
  if (!grant.operations.includes(call.operation)) {
    throw new Error(
      `Capability grant "${call.grantId}" does not authorize operation "${call.operation}".`,
    );
  }
  if (grant.expiresAt && Date.parse(grant.expiresAt) <= now.getTime()) {
    throw new Error(`Capability grant "${call.grantId}" has expired.`);
  }
  return call;
}

function protocolMessageSchema<T extends z.ZodRawShape>(shape: T): z.ZodObject<T> {
  return z.object(shape).strict().superRefine(addProtocolMessageIssues);
}

function addProtocolMessageIssues(value: unknown, ctx: z.RefinementCtx): void {
  addInlineSecretIssues(value, ctx);

  const encoded = JSON.stringify(value);
  if (utf8ByteLength(encoded) > TASK_WORKER_MAX_JSONL_LINE_BYTES) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: `Task worker protocol message exceeds ${TASK_WORKER_MAX_JSONL_LINE_BYTES} bytes.`,
    });
  }
}

function addInlineSecretIssues(value: unknown, ctx: z.RefinementCtx): void {
  for (const issue of findInlineSecretKeys(value)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: issue.path,
      message: `Task worker protocol messages must not contain inline secret field "${issue.key}".`,
    });
  }
}

function findInlineSecretKeys(
  value: unknown,
  path: Array<string | number> = [],
): Array<{ key: string; path: Array<string | number> }> {
  if (Array.isArray(value)) {
    return value.flatMap((child, index) => findInlineSecretKeys(child, [...path, index]));
  }
  if (!isRecord(value)) return [];

  const issues: Array<{ key: string; path: Array<string | number> }> = [];
  for (const [key, child] of Object.entries(value)) {
    if (isInlineSecretKey(key)) issues.push({ key, path: [...path, key] });
    issues.push(...findInlineSecretKeys(child, [...path, key]));
  }
  return issues;
}

function isInlineSecretKey(key: string): boolean {
  const normalized = key.toLowerCase().replace(/[^a-z0-9]/g, "");
  if (
    [
      "secretref",
      "secretrefs",
      "secretrefid",
      "secretstatus",
      "credentialref",
      "credentialrefs",
      "credentialrefid",
      "credentialstatus",
      "tokenref",
      "tokenrefs",
      "tokenrefid",
      "authref",
      "authrefs",
      "authrefid",
      "fencingtoken",
    ].includes(normalized)
  ) {
    return false;
  }
  return (
    normalized === "token" ||
    normalized === "authorization" ||
    normalized === "cookie" ||
    normalized.includes("apikey") ||
    normalized.includes("accesstoken") ||
    normalized.includes("authtoken") ||
    normalized.includes("bearertoken") ||
    normalized.includes("password") ||
    normalized.includes("passwd") ||
    normalized.includes("privatekey") ||
    normalized.includes("clientsecret") ||
    normalized.includes("secretaccesskey") ||
    normalized === "jwt" ||
    normalized === "idtoken" ||
    normalized === "refreshtoken" ||
    normalized === "sessiontoken" ||
    normalized.endsWith("token") ||
    normalized === "secret" ||
    normalized.endsWith("secret") ||
    normalized.includes("secretvalue") ||
    normalized === "credential" ||
    normalized.includes("credentialvalue")
  );
}

function assertJsonlLineSize(line: string): void {
  const bytes = utf8ByteLength(line);
  if (bytes > TASK_WORKER_MAX_JSONL_LINE_BYTES) {
    throw new Error(
      `Task worker protocol JSONL line is ${bytes} bytes; maximum is ${TASK_WORKER_MAX_JSONL_LINE_BYTES}.`,
    );
  }
}

function utf8ByteLength(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}

function isSafeRelativePath(value: string): boolean {
  if (value.startsWith("/") || value.startsWith("\\") || /^[A-Za-z]:[\\/]/.test(value)) {
    return false;
  }
  const segments = value.replaceAll("\\", "/").split("/");
  return segments.every((segment) => segment !== "" && segment !== "." && segment !== "..");
}

function addDuplicateIssues(
  values: readonly string[],
  ctx: z.RefinementCtx,
  path: Array<string | number>,
): void {
  const seen = new Set<string>();
  for (const [index, value] of values.entries()) {
    if (seen.has(value)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: [...path, index],
        message: `Duplicate value "${value}".`,
      });
    }
    seen.add(value);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
