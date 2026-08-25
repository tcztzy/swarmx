import { z } from "zod";

const boundedCountSchema = z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER);
const digestSchema = z.string().regex(/^sha256:[0-9a-f]{64}$/u);
const objectIdSchema = z.string().regex(/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/u);
const relativeRootSchema = z
  .string()
  .min(1)
  .max(4_096)
  .refine(
    (value) =>
      value === "." ||
      (!value.startsWith("/") &&
        !/^[A-Za-z]:[\\/]/u.test(value) &&
        !value.split(/[\\/]/u).includes("..")),
    "DVC root must be repository-relative",
  );

export const dvcUiStatusCategorySchema = z.strictObject({
  name: z.string().min(1).max(100),
  count: boundedCountSchema,
});

export const dvcUiStatusSummarySchema = z.strictObject({
  categories: z.array(dvcUiStatusCategorySchema).max(100),
  digest: digestSchema,
  entries: boundedCountSchema,
});

export const dvcUiGitSnapshotSchema = z.strictObject({
  version: z.string().min(1).max(100),
  objectFormat: z.string().min(1).max(20),
  head: objectIdSchema,
  branch: z.string().min(1).max(1_024).nullable(),
  upstream: z.string().min(1).max(1_024).nullable(),
  ahead: boundedCountSchema.nullable(),
  behind: boundedCountSchema.nullable(),
  clean: z.boolean(),
  staged: boundedCountSchema,
  unstaged: boundedCountSchema,
  untracked: boundedCountSchema,
  conflicted: boundedCountSchema,
});

export const dvcUiInspectionSchema = z.strictObject({
  data: dvcUiStatusSummarySchema,
  dvcLockDigest: digestSchema.nullable(),
  dvcYamlDigest: digestSchema.nullable(),
  git: dvcUiGitSnapshotSchema,
  root: relativeRootSchema,
  pipeline: dvcUiStatusSummarySchema,
  version: z.string().min(1).max(100),
});

export const dvcUiSnapshotSchema = z.discriminatedUnion("kind", [
  z.strictObject({
    kind: z.literal("project"),
    inspection: dvcUiInspectionSchema,
  }),
  z.strictObject({
    kind: z.literal("not-project"),
    message: z.string().min(1).max(200),
  }),
  z.strictObject({
    kind: z.literal("unavailable"),
    message: z.string().min(1).max(200),
  }),
]);

export type DvcUiInspection = z.infer<typeof dvcUiInspectionSchema>;
export type DvcUiSnapshot = z.infer<typeof dvcUiSnapshotSchema>;
export type DvcUiStatusSummary = z.infer<typeof dvcUiStatusSummarySchema>;
