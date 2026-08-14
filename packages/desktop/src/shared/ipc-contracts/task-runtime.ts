import {
  TaskApprovalSchema,
  TaskApprovalStatusSchema,
  TaskWorkItemSchema,
} from "@swarmx/core/task-runtime";
import {
  type TaskWorkerJsonValue,
  TaskWorkerPayloadSchema,
} from "@swarmx/core/task-worker-protocol";
import { z } from "zod";
import type { DesktopInvokeContract } from "./base.js";

const TASK_DECISION_MAX_RESPONSE_BYTES = 256 * 1024;
const TASK_DECISION_MAX_RESPONSE_DEPTH = 32;
const TASK_DECISION_MAX_RESPONSE_NODES = 10_000;
const TASK_DECISION_TEXT_ENCODER = new TextEncoder();
const TASK_DECISION_RESPONSE_REJECTED = Symbol("task-decision-response-rejected");
const TaskRuntimeRequestIdSchema = z.uuid();
const TaskRuntimeWorkItemIdSchema = TaskWorkItemSchema.shape.id;
const TaskRuntimeApprovalIdSchema = TaskApprovalSchema.shape.approvalId;

const DesktopTaskRuntimeDecisionResponseSchema = z
  .unknown()
  .transform<unknown>((value, context) => {
    const snapshot = boundedTaskDecisionResponseSnapshot(value);
    if (snapshot !== TASK_DECISION_RESPONSE_REJECTED) return snapshot;
    addTaskDecisionResponseIssue(context);
    return z.NEVER;
  })
  .pipe(TaskWorkerPayloadSchema);

export const DesktopTaskRuntimeCancelInputSchema = z
  .object({
    workItemId: TaskRuntimeWorkItemIdSchema,
    reason: z.string().min(1).max(512).optional(),
  })
  .strict();

export const DesktopTaskRuntimeDecisionInputSchema = z
  .object({
    approvalId: TaskRuntimeApprovalIdSchema,
    status: TaskApprovalStatusSchema.exclude(["requested"]),
    decidedBy: z.string().min(1).max(256),
    reason: z.string().min(1).max(512).optional(),
    response: DesktopTaskRuntimeDecisionResponseSchema.optional(),
  })
  .strict();

export const DesktopTaskRuntimeListResultSchema = z
  .object({
    requestId: TaskRuntimeRequestIdSchema,
    ok: z.literal(true),
    operation: z.literal("list"),
    workItems: z.array(TaskWorkItemSchema).max(10_000),
    approvals: z.array(TaskApprovalSchema).max(10_000),
    activeWorkItemIds: z.array(TaskRuntimeWorkItemIdSchema).max(10_000),
  })
  .strict();

const taskRuntimeWorkItemResultSchema = <Operation extends "cancel" | "decide">(
  operation: Operation,
) =>
  z
    .object({
      requestId: TaskRuntimeRequestIdSchema,
      ok: z.literal(true),
      operation: z.literal(operation),
      workItem: TaskWorkItemSchema,
    })
    .strict();

export const DesktopTaskRuntimeCancelResultSchema = taskRuntimeWorkItemResultSchema("cancel");
export const DesktopTaskRuntimeDecisionResultSchema = taskRuntimeWorkItemResultSchema("decide");
export const DesktopTaskRuntimeWorkItemResultSchema = z.discriminatedUnion("operation", [
  DesktopTaskRuntimeCancelResultSchema,
  DesktopTaskRuntimeDecisionResultSchema,
]);

export type DesktopTaskRuntimeCancelInput = z.infer<typeof DesktopTaskRuntimeCancelInputSchema>;
export type DesktopTaskRuntimeDecisionInput = z.infer<typeof DesktopTaskRuntimeDecisionInputSchema>;
export type DesktopTaskRuntimeListResult = z.infer<typeof DesktopTaskRuntimeListResultSchema>;
export type DesktopTaskRuntimeWorkItemResult = z.infer<
  typeof DesktopTaskRuntimeWorkItemResultSchema
>;

export const TaskRuntimeInvokeContracts = {
  "taskRuntime:list": {
    kind: "invoke",
    args: z.tuple([]),
    result: DesktopTaskRuntimeListResultSchema,
    audit: "failure_only",
  },
  "taskRuntime:cancel": {
    kind: "invoke",
    args: z.tuple([DesktopTaskRuntimeCancelInputSchema]),
    result: DesktopTaskRuntimeCancelResultSchema,
    audit: "intent_outcome",
  },
  "taskRuntime:decide": {
    kind: "invoke",
    args: z.tuple([DesktopTaskRuntimeDecisionInputSchema]),
    result: DesktopTaskRuntimeDecisionResultSchema,
    audit: "intent_outcome",
  },
} as const satisfies Record<string, DesktopInvokeContract>;

export interface DesktopTaskRuntimeApi {
  listTaskWorkItems(): Promise<DesktopTaskRuntimeListResult>;
  cancelTaskWorkItem(
    input: DesktopTaskRuntimeCancelInput,
  ): Promise<DesktopTaskRuntimeWorkItemResult>;
  decideTaskApproval(
    input: DesktopTaskRuntimeDecisionInput,
  ): Promise<DesktopTaskRuntimeWorkItemResult>;
}

function boundedTaskDecisionResponseSnapshot(
  root: unknown,
): TaskWorkerJsonValue | typeof TASK_DECISION_RESPONSE_REJECTED {
  type Container = TaskWorkerJsonValue[] | Record<string, TaskWorkerJsonValue>;
  type Frame =
    | {
        kind: "value";
        value: unknown;
        depth: number;
        parent: Container | null;
        key: string | number | null;
      }
    | { kind: "leave"; value: object };
  const pending: Frame[] = [{ kind: "value", value: root, depth: 0, parent: null, key: null }];
  const ancestors = new WeakSet<object>();
  let snapshot: TaskWorkerJsonValue | typeof TASK_DECISION_RESPONSE_REJECTED =
    TASK_DECISION_RESPONSE_REJECTED;
  let queuedValues = 1;
  let nodes = 0;
  let bytes = 0;
  const addBytes = (count: number): boolean => {
    bytes += count;
    return bytes <= TASK_DECISION_MAX_RESPONSE_BYTES;
  };

  try {
    while (pending.length > 0) {
      const frame = pending.pop() as Frame;
      if (frame.kind === "leave") {
        ancestors.delete(frame.value);
        continue;
      }
      queuedValues -= 1;
      nodes += 1;
      if (frame.depth > TASK_DECISION_MAX_RESPONSE_DEPTH) return TASK_DECISION_RESPONSE_REJECTED;

      const primitiveBytes = taskDecisionPrimitiveBytes(frame.value, bytes);
      if (primitiveBytes !== null) {
        if (!addBytes(primitiveBytes)) return TASK_DECISION_RESPONSE_REJECTED;
        assignTaskDecisionSnapshot(frame, frame.value as TaskWorkerJsonValue);
        continue;
      }
      if (frame.value === null || typeof frame.value !== "object") {
        return TASK_DECISION_RESPONSE_REJECTED;
      }
      if (ancestors.has(frame.value)) return TASK_DECISION_RESPONSE_REJECTED;
      ancestors.add(frame.value);
      pending.push({ kind: "leave", value: frame.value });

      if (Array.isArray(frame.value)) {
        const length = frame.value.length;
        if (!Number.isSafeInteger(length) || length < 0) {
          return TASK_DECISION_RESPONSE_REJECTED;
        }
        if (nodes + queuedValues + length > TASK_DECISION_MAX_RESPONSE_NODES) {
          return TASK_DECISION_RESPONSE_REJECTED;
        }
        if (!addBytes(2 + Math.max(0, length - 1))) {
          return TASK_DECISION_RESPONSE_REJECTED;
        }
        const output: TaskWorkerJsonValue[] = [];
        assignTaskDecisionSnapshot(frame, output);
        queuedValues += length;
        for (let index = length - 1; index >= 0; index -= 1) {
          pending.push({
            kind: "value",
            value: frame.value[index],
            depth: frame.depth + 1,
            parent: output,
            key: index,
          });
        }
        continue;
      }

      if (!addBytes(2)) return TASK_DECISION_RESPONSE_REJECTED;
      const output = Object.create(null) as Record<string, TaskWorkerJsonValue>;
      assignTaskDecisionSnapshot(frame, output);
      const children: Array<[string, unknown]> = [];
      let ownCount = 0;
      for (const key in frame.value) {
        if (!Object.hasOwn(frame.value, key)) continue;
        ownCount += 1;
        if (nodes + queuedValues + ownCount > TASK_DECISION_MAX_RESPONSE_NODES) {
          return TASK_DECISION_RESPONSE_REJECTED;
        }
        const keyBytes = taskDecisionStringBytes(key, bytes);
        if (keyBytes === null || !addBytes(keyBytes + (ownCount === 1 ? 1 : 2))) {
          return TASK_DECISION_RESPONSE_REJECTED;
        }
        children.push([key, (frame.value as Record<string, unknown>)[key]]);
      }
      queuedValues += ownCount;
      for (let index = children.length - 1; index >= 0; index -= 1) {
        const [key, value] = children[index] as [string, unknown];
        pending.push({
          kind: "value",
          value,
          depth: frame.depth + 1,
          parent: output,
          key,
        });
      }
    }
    return snapshot;
  } catch {
    return TASK_DECISION_RESPONSE_REJECTED;
  }

  function assignTaskDecisionSnapshot(
    frame: Extract<Frame, { kind: "value" }>,
    value: TaskWorkerJsonValue,
  ): void {
    if (frame.parent === null) {
      snapshot = value;
    } else if (Array.isArray(frame.parent)) {
      frame.parent[frame.key as number] = value;
    } else {
      frame.parent[frame.key as string] = value;
    }
  }
}

function taskDecisionPrimitiveBytes(value: unknown, usedBytes: number): number | null {
  if (typeof value === "string") return taskDecisionStringBytes(value, usedBytes);
  if (value === null) return 4;
  if (typeof value === "boolean") return value ? 4 : 5;
  if (typeof value === "number" && Number.isFinite(value)) return JSON.stringify(value).length;
  return null;
}

function taskDecisionStringBytes(value: string, usedBytes: number): number | null {
  const remaining = TASK_DECISION_MAX_RESPONSE_BYTES - usedBytes;
  if (value.length + 2 > remaining) return null;
  return TASK_DECISION_TEXT_ENCODER.encode(JSON.stringify(value)).byteLength;
}

function addTaskDecisionResponseIssue(context: z.RefinementCtx): void {
  context.addIssue({
    code: "custom",
    message: "Task Runtime decision response exceeds the transport limit.",
  });
}
