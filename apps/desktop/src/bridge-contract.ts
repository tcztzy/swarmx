import { z } from "zod";
import { ExecutionPageSchema, RunControlSchema } from "./execution-record.js";
import { HistoryMessagesSchema } from "./message-activity.js";
import { LanguageSchema, SettingsSchema } from "./settings.js";

export const LogsQuerySchema = z
  .strictObject({
    after: z.coerce.number().int().nonnegative().default(0),
    limit: z.coerce.number().int().min(1).max(1000).default(200),
    session: z.string().min(1).max(2048).optional(),
    run: z.string().min(1).max(2048).optional(),
    descendants: z
      .enum(["true", "false"])
      .default("false")
      .transform((value) => value === "true"),
  })
  .refine(
    (query) => !query.descendants || query.session !== undefined,
    "Descendants require a session.",
  );

export const LogsEvidencePayloadSchema = z.strictObject({
  sources: z
    .array(z.templateLiteral(["urn:swarmx:execution:", z.uuid()]))
    .min(1)
    .max(64),
});

export const ToolCallPayloadSchema = z.strictObject({
  requestId: z.string().uuid(),
  name: z.string().min(1).max(64),
  args: z.unknown(),
});
export const ToolCancelPayloadSchema = z.strictObject({ requestId: z.string().uuid() });
export const LanguageWritePayloadSchema = z.strictObject({ language: LanguageSchema });
export const AgentPayloadSchema = z.strictObject({ agent: z.string().min(1).max(64) });
export const HistoryPayloadSchema = z.strictObject({
  agent: z.string().min(1).max(64),
  sessionId: z.string().min(1).max(2048),
});
export const ModelsPayloadSchema = z.strictObject({
  agent: z.string().min(1).max(64),
  session: z.string().min(1).max(2048).optional(),
});
export const RunControlPayloadSchema = z.strictObject({
  runId: z.string().min(1).max(2048),
  command: RunControlSchema,
});
export const AgUiStartPayloadSchema = z.strictObject({
  agent: z.string().min(1).max(64),
  input: z.unknown(),
});
export const AgUiCancelPayloadSchema = z.strictObject({
  agent: z.string().min(1).max(64),
  threadId: z.string().min(1).max(2048),
});

export const SessionListSchema = z.array(
  z.object({
    sessionId: z.string(),
    title: z.string().nullish(),
    updatedAt: z.string().nullish(),
  }),
);
export const SessionCreateSchema = z.object({ sessionId: z.string() });
export const BootstrapSchema = z.strictObject({
  agents: z.array(z.string()),
  defaultHarness: z.string(),
  language: LanguageSchema.nullable(),
  sessions: SessionListSchema,
  sessionError: z.string().optional(),
  cwd: z.string().min(1),
});
export const SettingsResponseSchema = SettingsSchema.extend({ cwd: z.string().min(1) });
export const HistoryResponseSchema = z.discriminatedUnion("supported", [
  z.strictObject({ supported: z.literal(true), messages: HistoryMessagesSchema }),
  z.strictObject({ supported: z.literal(false) }),
]);
export const ExecutionPageResponseSchema = ExecutionPageSchema;
export const AgUiEventMessageSchema = z.object({
  threadId: z.string(),
  event: z.unknown().optional(),
  error: z.string().optional(),
  done: z.boolean().optional(),
});
export type AgUiEventMessage = z.infer<typeof AgUiEventMessageSchema>;

/** Preserves native error detail for renderer alerts. */
export function actionableMessage(error: unknown): string {
  let message = error instanceof Error ? error.message : String(error);
  const detail = z
    .object({
      data: z.union([
        z.string(),
        z.object({ message: z.string() }).transform((value) => value.message),
        z.object({ details: z.string() }).transform((value) => value.details),
      ]),
    })
    .safeParse(error);
  if (detail.success && detail.data.data) message += `: ${detail.data.data}`;
  return message;
}
