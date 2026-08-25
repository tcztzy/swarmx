import { z } from "zod";

const objectIdSchema = z.string().regex(/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/u);
const boundedPathSchema = z.string().min(1).max(4_096);

export const gitUiEntrySchema = z.strictObject({
  kind: z.enum(["ordinary", "renamed", "unmerged", "untracked"]),
  path: boundedPathSchema,
  previousPath: boundedPathSchema.optional(),
  index: z.string().length(1),
  worktree: z.string().length(1),
});

export const gitUiRepositorySnapshotSchema = z.strictObject({
  kind: z.literal("repository"),
  version: z.string().min(1).max(100),
  objectFormat: z.string().min(1).max(20),
  head: objectIdSchema,
  branch: z.string().min(1).max(1_024).nullable(),
  upstream: z.string().min(1).max(1_024).nullable(),
  ahead: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER).nullable(),
  behind: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER).nullable(),
  clean: z.boolean(),
  staged: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
  unstaged: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
  untracked: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
  conflicted: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
  truncated: z.boolean(),
  entries: z.array(gitUiEntrySchema).max(5_000),
});

export const gitUiSnapshotSchema = z.discriminatedUnion("kind", [
  gitUiRepositorySnapshotSchema,
  z.strictObject({
    kind: z.literal("not-repository"),
    message: z.string().min(1).max(200),
  }),
  z.strictObject({
    kind: z.literal("unavailable"),
    message: z.string().min(1).max(200),
  }),
]);

export type GitUiEntry = z.infer<typeof gitUiEntrySchema>;
export type GitUiRepositorySnapshot = z.infer<typeof gitUiRepositorySnapshotSchema>;
export type GitUiSnapshot = z.infer<typeof gitUiSnapshotSchema>;
