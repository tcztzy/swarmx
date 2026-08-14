import {
  GlobalMemoryForgetInputSchema,
  GlobalMemorySaveInputSchema,
  GlobalMemoryStateSchema,
} from "@swarmx/core/personal-memory";
import { z } from "zod";
import type { DesktopInvokeContract } from "./base.js";

export const DesktopGlobalMemoryStateSchema = GlobalMemoryStateSchema.extend({
  legacyUser: z.boolean(),
}).strict();

export const DesktopGlobalMemorySaveInputSchema = GlobalMemorySaveInputSchema.extend({
  expectedRevision: z.number().int().nonnegative().optional(),
}).strict();

export const DesktopGlobalMemoryForgetInputSchema = GlobalMemoryForgetInputSchema.extend({
  expectedRevision: z.number().int().nonnegative().optional(),
}).strict();

export type DesktopGlobalMemoryState = z.infer<typeof DesktopGlobalMemoryStateSchema>;
export type DesktopGlobalMemorySaveInput = z.infer<typeof DesktopGlobalMemorySaveInputSchema>;
export type DesktopGlobalMemoryForgetInput = z.infer<typeof DesktopGlobalMemoryForgetInputSchema>;

export const GlobalMemoryInvokeContracts = {
  "personalMemory:get": {
    kind: "invoke",
    args: z.tuple([]),
    result: DesktopGlobalMemoryStateSchema,
    audit: "failure_only",
  },
  "personalMemory:save": {
    kind: "invoke",
    args: z.tuple([DesktopGlobalMemorySaveInputSchema]),
    result: DesktopGlobalMemoryStateSchema,
    audit: "intent_outcome",
  },
  "personalMemory:forget": {
    kind: "invoke",
    args: z.tuple([DesktopGlobalMemoryForgetInputSchema]),
    result: DesktopGlobalMemoryStateSchema,
    audit: "intent_outcome",
  },
} as const satisfies Record<string, DesktopInvokeContract>;

export interface DesktopGlobalMemoryApi {
  getPersonalMemory(): Promise<DesktopGlobalMemoryState>;
  savePersonalMemory(input: DesktopGlobalMemorySaveInput): Promise<DesktopGlobalMemoryState>;
  forgetPersonalMemory(input: DesktopGlobalMemoryForgetInput): Promise<DesktopGlobalMemoryState>;
}
