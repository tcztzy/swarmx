import { z } from "zod";
import { HarnessAccessSchema, ToolGrantSchema } from "./permissions.js";

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
export const SettingsSchema = z.strictObject({
  policy: ExecutionPolicySchema,
  // Preserve retired environment metadata in saved settings without interpreting or executing it.
  environment: z.json().nullable(),
});
export type ExecutionPolicy = z.infer<typeof ExecutionPolicySchema>;
export type Settings = z.infer<typeof SettingsSchema>;
export const DEFAULT_POLICY: ExecutionPolicy = {
  filesystem: "workspace-write",
  tools: [...ToolGrantSchema.options],
  cpus: 2,
  memoryMb: 2048,
  timeoutSeconds: 300,
};
