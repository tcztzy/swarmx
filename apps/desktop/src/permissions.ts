import { z } from "zod";

export const HarnessSchema = z.enum(["codex", "claude", "hermes", "openclaw"]);
export const HarnessAccessSchema = z.partialRecord(
  HarnessSchema,
  z.array(z.string().min(1).max(512)).nullable(),
);
export const ToolGrantSchema = z.enum([
  "memory.read",
  "memory.write",
  "science.read",
  "science.write",
]);
export const PermissionRequestSchema = z.strictObject({
  tools: z.array(ToolGrantSchema).optional(),
  harnesses: HarnessAccessSchema.optional(),
  delegation: z.boolean().optional(),
});
export type PermissionRequest = z.infer<typeof PermissionRequestSchema>;
export const AgentPermissionsSchema = PermissionRequestSchema.required();
export type AgentPermissions = z.infer<typeof AgentPermissionsSchema>;

export function projectPermissions(policy: PermissionRequest): AgentPermissions {
  return {
    tools: policy.tools ?? [...ToolGrantSchema.options],
    harnesses: policy.harnesses ?? { codex: null, claude: null, hermes: null, openclaw: null },
    delegation: policy.delegation ?? true,
  };
}

export function intersectPermissions(a: AgentPermissions, b: AgentPermissions): AgentPermissions {
  const harnesses: AgentPermissions["harnesses"] = {};
  for (const id of HarnessSchema.options) {
    const left = a.harnesses[id];
    const right = b.harnesses[id];
    if (left === undefined || right === undefined) continue;
    harnesses[id] =
      left === null ? right : right === null ? left : left.filter((m) => right.includes(m));
  }
  return {
    tools: a.tools.filter((tool) => b.tools.includes(tool)),
    harnesses,
    delegation: a.delegation && b.delegation,
  };
}

export function narrowPermissions(
  parent: AgentPermissions,
  request?: PermissionRequest,
): AgentPermissions {
  if (!request) return parent;
  const requested: AgentPermissions = {
    tools: request.tools ?? parent.tools,
    harnesses: request.harnesses ?? parent.harnesses,
    delegation: request.delegation ?? parent.delegation,
  };
  if (requested.tools.some((tool) => !parent.tools.includes(tool)))
    throw new Error("Cannot grant product tool permission beyond the parent.");
  if (!parent.delegation && requested.delegation)
    throw new Error("Cannot grant delegation permission beyond the parent.");
  for (const id of HarnessSchema.options) {
    const models = requested.harnesses[id];
    const allowed = parent.harnesses[id];
    if (
      models !== undefined &&
      (allowed === undefined ||
        (allowed !== null && (models === null || models.some((m) => !allowed.includes(m)))))
    )
      throw new Error(`Cannot grant harness/model permission for "${id}" beyond the parent.`);
  }
  return structuredClone(requested);
}
