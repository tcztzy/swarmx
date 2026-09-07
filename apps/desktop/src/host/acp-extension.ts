import { z } from "zod";
import {
  AgentPermissionsSchema,
  narrowPermissions,
  type PermissionRequest,
  PermissionRequestSchema,
  projectPermissions,
} from "../permissions.js";

export const SwarmxCapability = z.object({ version: z.literal(2), permissions: z.literal(true) });
export const SwarmxRequest = z.strictObject({
  version: z.literal(2),
  permissions: PermissionRequestSchema.optional(),
});

/** Callers requiring a ceiling must verify both negotiation and the session acknowledgement. */
export function acknowledgedPermissions(
  meta: Record<string, unknown> | null | undefined,
  requested: PermissionRequest,
) {
  const { permissions } = z
    .object({ version: z.literal(2), permissions: AgentPermissionsSchema })
    .parse(meta?.swarmx);
  return narrowPermissions(projectPermissions(requested), permissions);
}
