import { type AGUIEvent, EventSchemas } from "@ag-ui/core";
import { z } from "zod";

export const ExecutionAttributes = z.record(
  z.string(),
  z.union([z.string(), z.number(), z.boolean(), z.null()]),
);
const RecordEnvelope = z.object({
  schemaVersion: z.literal(1),
  seq: z.number().int().positive(),
  id: z.string(),
  observedAt: z.string(),
  workspaceId: z.string(),
  sessionId: z.string().nullable(),
  runId: z.string().nullable(),
  causedBy: z.string().nullable(),
  attributes: ExecutionAttributes,
  event: z.unknown(),
});
export type ExecutionRecord = z.infer<typeof RecordEnvelope> & { event: AGUIEvent };
export const ExecutionRecordSchema: z.ZodType<ExecutionRecord> = RecordEnvelope.transform(
  (record) => ({ ...record, event: EventSchemas.parse(record.event) }),
);

export const ExecutionPageSchema = z.object({
  events: z.array(ExecutionRecordSchema),
  nextAfter: z.number().int().nonnegative(),
  activeRunIds: z.array(z.string()),
});

export const ExecutionRunSummarySchema = z.object({
  runId: z.string(),
  sessionId: z.string().nullable(),
  task: z.string().nullable(),
  harness: z.string().nullable(),
  requestedModel: z.string().nullable(),
  reportedModel: z.string().nullable(),
  provider: z.string().nullable(),
  harnessVersion: z.string().nullable(),
  modelVersion: z.string().nullable(),
  profile: z.string().nullable(),
  startedAt: z.string().nullable(),
  finishedAt: z.string().nullable(),
  outcome: z.enum(["completed", "error", "cancelled", "incomplete", "other"]),
  elapsedMs: z.number().nonnegative().nullable(),
  inputTokens: z.number().int().nonnegative().nullable(),
  outputTokens: z.number().int().nonnegative().nullable(),
  costUsd: z.number().nonnegative().nullable(),
  usageBasis: z.string().nullable(),
  sources: z.array(z.string()),
});
export type ExecutionRunSummary = z.infer<typeof ExecutionRunSummarySchema>;

export const ExecutionStatisticsSchema = z.object({
  recipe: z.literal("swarmx.execution.v1"),
  scope: z.literal("cited executions"),
  runIds: z.array(z.string()),
  window: z.object({ startedAt: z.string().nullable(), finishedAt: z.string().nullable() }),
  sampleCount: z.number().int().nonnegative(),
  completed: z.number().int().nonnegative(),
  error: z.number().int().nonnegative(),
  cancelled: z.number().int().nonnegative(),
  incomplete: z.number().int().nonnegative(),
  other: z.number().int().nonnegative(),
  elapsed: z.object({
    sampleCount: z.number().int().nonnegative(),
    medianMs: z.number().nullable(),
  }),
  usage: z.object({
    sampleCount: z.number().int().nonnegative(),
    inputTokens: z.number().int().nonnegative().nullable(),
    outputTokens: z.number().int().nonnegative().nullable(),
  }),
  cost: z.object({ sampleCount: z.number().int().nonnegative(), usd: z.number().nullable() }),
});
export type ExecutionStatistics = z.infer<typeof ExecutionStatisticsSchema>;

export const ExecutionEvidenceSchema = z.object({
  records: z.array(ExecutionRecordSchema),
  runs: z.array(ExecutionRunSummarySchema),
  statistics: ExecutionStatisticsSchema,
});
export type ExecutionEvidence = z.infer<typeof ExecutionEvidenceSchema>;

export const RunControlSchema = z.discriminatedUnion("action", [
  z.strictObject({ action: z.literal("steer"), text: z.string().trim().min(1).max(100_000) }),
  z.strictObject({ action: z.literal("cancel") }),
]);
export type RunControl = z.infer<typeof RunControlSchema>;
