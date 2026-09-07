import { z } from "zod";
import { HarnessAccessSchema, ToolGrantSchema } from "./permissions.js";

export const ProjectSchema = z.strictObject({
  id: z
    .string()
    .min(1)
    .max(128)
    .regex(/^[a-zA-Z0-9_-]+$/u),
  label: z.string().trim().min(1).max(120),
  root: z.string().min(1).max(4096),
});
export const ProjectCatalogSchema = z
  .strictObject({
    projects: z.array(ProjectSchema),
    activeId: z.string().nullable(),
  })
  .refine(
    ({ projects, activeId }) =>
      new Set(projects.map(({ id }) => id)).size === projects.length &&
      new Set(projects.map(({ root }) => root)).size === projects.length &&
      (activeId === null ? projects.length === 0 : projects.some(({ id }) => id === activeId)),
    "Invalid project catalog",
  );
export const AddProjectSchema = ProjectSchema.pick({ label: true, root: true });

export const LanguageSchema = z.enum(["zh", "en"]);

export const ExecutionPolicySchema = z.strictObject({
  harnesses: HarnessAccessSchema.optional(),
  delegation: z.boolean().optional(),
  filesystem: z.enum(["read-only", "workspace-write"]),
  tools: z.array(ToolGrantSchema),
  cpus: z.number().int().min(1).max(32),
  memoryMb: z.number().int().min(256).max(65536),
  timeoutSeconds: z.number().int().min(5).max(3600),
});
export const EnvironmentSchema = z.strictObject({
  imageId: z.string().regex(/^sha256:[a-f0-9]{64}$/u),
  recipeDigest: z.string().regex(/^sha256:[a-f0-9]{64}$/u),
  platform: z.enum(["linux/amd64", "linux/arm64"]),
  createdAt: z.string().datetime(),
  pythonVersion: z.string(),
  packages: z.array(z.string()).max(1000),
});
export const WorkspaceSettingsSchema = z.strictObject({
  policy: ExecutionPolicySchema,
  environment: EnvironmentSchema.nullable(),
});
export const EnvironmentStatusSchema = z.strictObject({
  state: z.enum(["missing", "ready", "building", "failed"]),
  log: z.string(),
  environment: EnvironmentSchema.nullable(),
  activeProcesses: z.number().int().nonnegative(),
});
export type ExecutionPolicy = z.infer<typeof ExecutionPolicySchema>;
export type ResearchEnvironment = z.infer<typeof EnvironmentSchema>;
export type WorkspaceSettings = z.infer<typeof WorkspaceSettingsSchema>;
export const DEFAULT_POLICY: ExecutionPolicy = {
  filesystem: "workspace-write",
  tools: [...ToolGrantSchema.options],
  cpus: 2,
  memoryMb: 2048,
  timeoutSeconds: 300,
};
