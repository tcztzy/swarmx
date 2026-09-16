import { z } from "zod";

export const MemorySettingsSchema = z.strictObject({
  enabled: z.boolean().default(true),
  autoReview: z.boolean().default(true),
  writeApproval: z.boolean().default(false),
  reviewInterval: z.number().int().min(1).max(100).default(10),
  reviewHarness: z.enum(["codex", "claude"]).default("codex"),
});
export const MemoryNoteSchema = z.strictObject({
  content: z.string(),
  revision: z.string(),
  limit: z.number(),
});
export const MemoryGraphSchema = z.strictObject({
  nodes: z.array(
    z.strictObject({
      id: z.string(),
      revision: z.string(),
      title: z.string(),
      description: z.string(),
      type: z.string(),
      status: z.enum(["draft", "stable", "deprecated"]),
      stale: z.boolean(),
    }),
  ),
  edges: z.array(
    z.strictObject({
      source: z.string(),
      target: z.string(),
      revision: z.string(),
      stale: z.boolean(),
    }),
  ),
});
export const MemoryStatusSchema = z.strictObject({
  settings: MemorySettingsSchema,
  note: MemoryNoteSchema,
  pending: z.array(
    z.strictObject({
      id: z.string().uuid(),
      createdAt: z.string(),
      origin: z.enum(["agent", "review"]),
      sessionId: z.string().nullable(),
      operation: z.object({
        action: z.enum([
          "create_memory",
          "update_memory",
          "deprecate_memory",
          "update_core_memory",
        ]),
        request: z
          .object({
            title: z.string().optional(),
            content: z.string().optional(),
            body: z.string().optional(),
            id: z.string().optional(),
          })
          .passthrough(),
      }),
    }),
  ),
  review: z.strictObject({
    state: z.enum(["idle", "running", "completed", "failed"]),
    message: z.string(),
    sessionId: z.string().nullable(),
  }),
});
