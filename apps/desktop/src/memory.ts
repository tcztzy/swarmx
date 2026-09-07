import { z } from "zod";

export const MemorySettingsSchema = z.strictObject({
  enabled: z.boolean().default(true),
  autoReview: z.boolean().default(true),
  writeApproval: z.boolean().default(false),
  reviewInterval: z.number().int().min(1).max(100).default(10),
  reviewHarness: z.enum(["codex", "claude"]).default("codex"),
});
export const MemoryNoteSchema = z.strictObject({
  target: z.enum(["user", "workspace"]),
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
      scope: z.enum(["global", "workspace"]),
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
  notes: z.array(MemoryNoteSchema),
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
            target: z.enum(["user", "workspace"]).optional(),
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
