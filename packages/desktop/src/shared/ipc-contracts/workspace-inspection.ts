import { z } from "zod";
import type { DesktopInvokeContract } from "./base.js";

const MAX_ROOT_LENGTH = 32_768;
const MAX_PATH_LENGTH = 4_096;
const MAX_ERROR_LENGTH = 8_192;
const MAX_REVIEW_FILES = 200;
const MAX_PATCH_LENGTH = 256 * 1024;
const MAX_REVIEW_PATCH_BYTES = 2 * 1024 * 1024;
const MAX_DIRECTORY_ENTRIES = 500;
const MAX_PREVIEW_LENGTH = 1024 * 1024;
const UTF8_ENCODER = new TextEncoder();

const WorkspaceRootSchema = z.string().min(1).max(MAX_ROOT_LENGTH);
const WorkspaceCwdSchema = z
  .string()
  .max(MAX_ROOT_LENGTH)
  .refine((value) => !value.includes("\0"));
const WorkspacePathSchema = z
  .string()
  .max(MAX_PATH_LENGTH)
  .refine((value) => !value.includes("\0"));
const WorkspaceFilePathSchema = WorkspacePathSchema.refine((value) => value.length > 0);

export const DesktopWorkspaceReviewFileSchema = z
  .object({
    path: WorkspaceFilePathSchema,
    previousPath: WorkspaceFilePathSchema.optional(),
    status: z.string().length(2),
    patch: z.string().max(MAX_PATCH_LENGTH),
    binary: z.boolean(),
    additions: z.number().int().nonnegative(),
    deletions: z.number().int().nonnegative(),
    truncated: z.boolean(),
    error: z.string().min(1).max(MAX_ERROR_LENGTH).optional(),
  })
  .strict();

export const DesktopWorkspaceReviewSnapshotSchema = z
  .object({
    root: WorkspaceRootSchema,
    branch: z.string().max(MAX_PATH_LENGTH).nullable(),
    isRepository: z.boolean(),
    files: z.array(DesktopWorkspaceReviewFileSchema).max(MAX_REVIEW_FILES),
    truncated: z.boolean(),
    error: z.string().min(1).max(MAX_ERROR_LENGTH).optional(),
  })
  .strict()
  .superRefine((snapshot, context) => {
    let patchBytes = 0;
    for (const [index, file] of snapshot.files.entries()) {
      patchBytes += UTF8_ENCODER.encode(file.patch).byteLength;
      if (patchBytes > MAX_REVIEW_PATCH_BYTES) {
        context.addIssue({
          code: "custom",
          path: ["files", index, "patch"],
          message: "total patch content exceeds the Desktop transport limit",
        });
        break;
      }
    }
  });

export const DesktopWorkspaceDirectoryEntrySchema = z
  .object({
    name: z.string().min(1).max(MAX_PATH_LENGTH),
    path: WorkspaceFilePathSchema,
    kind: z.enum(["directory", "file", "symlink", "other"]),
    size: z.number().int().nonnegative().optional(),
  })
  .strict();

export const DesktopWorkspaceDirectoryListingSchema = z
  .object({
    root: WorkspaceRootSchema,
    path: WorkspacePathSchema,
    entries: z.array(DesktopWorkspaceDirectoryEntrySchema).max(MAX_DIRECTORY_ENTRIES),
    truncated: z.boolean(),
  })
  .strict();

export const DesktopWorkspaceFilePreviewSchema = z
  .object({
    root: WorkspaceRootSchema,
    path: WorkspaceFilePathSchema,
    content: z.string().max(MAX_PREVIEW_LENGTH),
    size: z.number().int().nonnegative(),
    binary: z.boolean(),
    truncated: z.boolean(),
  })
  .strict();

export type DesktopWorkspaceReviewFile = z.infer<typeof DesktopWorkspaceReviewFileSchema>;
export type DesktopWorkspaceReviewSnapshot = z.infer<typeof DesktopWorkspaceReviewSnapshotSchema>;
export type DesktopWorkspaceDirectoryEntry = z.infer<typeof DesktopWorkspaceDirectoryEntrySchema>;
export type DesktopWorkspaceDirectoryListing = z.infer<
  typeof DesktopWorkspaceDirectoryListingSchema
>;
export type DesktopWorkspaceFilePreview = z.infer<typeof DesktopWorkspaceFilePreviewSchema>;

const WorkspaceContextSchema = z.object({ cwd: WorkspaceCwdSchema.optional() }).strict();

export const WorkspaceInspectionInvokeContracts = {
  "workspace:root": {
    kind: "invoke",
    args: z.tuple([]),
    result: WorkspaceRootSchema,
    audit: "intent_outcome",
  },
  "workspace:review": {
    kind: "invoke",
    args: z.tuple([WorkspaceContextSchema]),
    result: DesktopWorkspaceReviewSnapshotSchema,
    audit: "intent_outcome",
  },
  "workspace:listDirectory": {
    kind: "invoke",
    args: z.tuple([
      z.object({ path: WorkspacePathSchema, cwd: WorkspaceCwdSchema.optional() }).strict(),
    ]),
    result: DesktopWorkspaceDirectoryListingSchema,
    audit: "intent_outcome",
  },
  "workspace:readFile": {
    kind: "invoke",
    args: z.tuple([
      z.object({ path: WorkspaceFilePathSchema, cwd: WorkspaceCwdSchema.optional() }).strict(),
    ]),
    result: DesktopWorkspaceFilePreviewSchema,
    audit: "intent_outcome",
  },
} as const satisfies Record<string, DesktopInvokeContract>;

export interface DesktopWorkspaceInspectionApi {
  workspaceRoot(): Promise<string>;
  getWorkspaceReview(cwd?: string): Promise<DesktopWorkspaceReviewSnapshot>;
  listWorkspaceDirectory(path?: string, cwd?: string): Promise<DesktopWorkspaceDirectoryListing>;
  readWorkspaceFile(path: string, cwd?: string): Promise<DesktopWorkspaceFilePreview>;
}
