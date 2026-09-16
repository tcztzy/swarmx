import { type Message, MessageSchema } from "@ag-ui/core";
import { z } from "zod";

const PhaseSchema = z.enum(["commentary", "final_answer"]);
const TurnTimingSchema = z.object({
  turnId: z.string(),
  startedAt: z.number().nonnegative().nullish(),
  durationMs: z.number().nonnegative().nullish(),
});
export const MessageActivitySchema = TurnTimingSchema.extend({ phase: PhaseSchema });
export type MessageActivity = z.infer<typeof MessageActivitySchema>;
export type TurnTiming = z.infer<typeof TurnTimingSchema>;
const ToolActivitySchema = z.object({
  kind: z.string().optional(),
  status: z.enum(["pending", "in_progress", "completed", "failed"]).optional(),
});
export type ToolActivity = z.infer<typeof ToolActivitySchema>;
export type HistoryMessage = Message & { _meta?: MessageActivity; _tool?: ToolActivity };

export const HistoryMessagesSchema = z
  .array(
    z.looseObject({
      _meta: MessageActivitySchema.optional(),
      _tool: ToolActivitySchema.optional(),
    }),
  )
  .transform((messages): HistoryMessage[] =>
    messages.map((message) => ({
      ...MessageSchema.parse(message),
      ...(message._meta ? { _meta: message._meta } : {}),
      ...(message._tool ? { _tool: message._tool } : {}),
    })),
  );

const MessageEventSchema = TurnTimingSchema.extend({
  type: z.literal("message"),
  messageId: z.string().optional(),
  phase: PhaseSchema.optional(),
});
const ToolEventSchema = ToolActivitySchema.extend({
  type: z.literal("tool"),
  toolCallId: z.string(),
});
export const ActivitySchema = z.discriminatedUnion("type", [MessageEventSchema, ToolEventSchema]);
export type Activity = z.infer<typeof ActivitySchema>;

export function readMessageActivity(event: unknown) {
  const parsed = MessageEventSchema.safeParse(event);
  if (!parsed.success) return undefined;
  const { type: _type, ...activity } = parsed.data;
  return activity;
}

export function readToolActivity(event: unknown) {
  const parsed = ToolEventSchema.safeParse(event);
  if (!parsed.success) return undefined;
  const { type: _type, ...activity } = parsed.data;
  return activity;
}
