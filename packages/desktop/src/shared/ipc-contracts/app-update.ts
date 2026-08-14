import { z } from "zod";
import type { DesktopEventContract, DesktopInvokeContract } from "./base.js";

export const DesktopUpdatePhaseSchema = z.enum([
  "hidden",
  "available",
  "downloading",
  "installing",
  "restarting",
]);

export const DesktopUpdateStateSchema = z
  .object({
    phase: DesktopUpdatePhaseSchema,
    currentVersion: z.string().min(1).max(128),
    latestVersion: z.string().min(1).max(128).optional(),
    progress: z.number().finite().min(0).max(100).optional(),
    error: z.string().max(1_024).optional(),
  })
  .strict();

export type DesktopUpdatePhase = z.infer<typeof DesktopUpdatePhaseSchema>;
export type DesktopUpdateState = z.infer<typeof DesktopUpdateStateSchema>;

export const AppUpdateInvokeContracts = {
  "appUpdate:getState": {
    kind: "invoke",
    args: z.tuple([]),
    result: DesktopUpdateStateSchema,
    audit: "failure_only",
  },
  "appUpdate:install": {
    kind: "invoke",
    args: z.tuple([]),
    result: DesktopUpdateStateSchema,
    audit: "intent_outcome",
  },
} as const satisfies Record<string, DesktopInvokeContract>;

export const AppUpdateEventContracts = {
  "appUpdate:state": {
    kind: "event",
    payload: DesktopUpdateStateSchema,
  },
} as const satisfies Record<string, DesktopEventContract>;
