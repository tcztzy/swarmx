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

export const RunControlSchema = z.discriminatedUnion("action", [
  z.strictObject({ action: z.literal("steer"), text: z.string().trim().min(1).max(100_000) }),
  z.strictObject({ action: z.literal("cancel") }),
]);
export type RunControl = z.infer<typeof RunControlSchema>;
