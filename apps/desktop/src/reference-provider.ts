import type { z } from "zod";
import type { ToolGrantSchema } from "./permissions.js";

/** Borrowed, workspace-bound resolver supplied only by trusted Host configuration. */
export interface ReferenceProvider {
  readonly scheme: string;
  readonly requiredPermissions: readonly z.infer<typeof ToolGrantSchema>[];
  resolve(id: string): { id: string; exactId: string; revision: string };
  checkResource(id: string):
    | undefined
    | {
        ruleId: "source.invalid" | "source.unresolved";
        severity: "error" | "warning";
        message: string;
      };
}
