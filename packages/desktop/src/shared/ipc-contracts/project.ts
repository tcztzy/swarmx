import { ProjectDataSchema as CoreProjectDataSchema } from "@swarmx/core/project-contracts";
import { z } from "zod";
import type { DesktopInvokeContract } from "./base.js";

const ProjectIdSchema = z.string().min(1);

export const DesktopProjectDataSchema = CoreProjectDataSchema.extend({
  pinned: z.boolean(),
}).strict();
export type DesktopProjectData = z.infer<typeof DesktopProjectDataSchema>;

const ProjectIdInputSchema = z.object({ id: ProjectIdSchema }).strict();

export const ProjectInvokeContracts = {
  "project:list": {
    kind: "invoke",
    args: z.tuple([]),
    result: z.array(DesktopProjectDataSchema),
    audit: "intent_outcome",
  },
  "project:addExisting": {
    kind: "invoke",
    args: z.tuple([]),
    result: DesktopProjectDataSchema.nullable(),
    audit: "intent_outcome",
  },
  "project:createScratch": {
    kind: "invoke",
    args: z.tuple([]),
    result: DesktopProjectDataSchema.nullable(),
    audit: "intent_outcome",
  },
  "project:setPinned": {
    kind: "invoke",
    args: z.tuple([z.object({ id: ProjectIdSchema, pinned: z.boolean() }).strict()]),
    result: DesktopProjectDataSchema,
    audit: "intent_outcome",
  },
  "project:rename": {
    kind: "invoke",
    args: z.tuple([z.object({ id: ProjectIdSchema, name: z.string().min(1) }).strict()]),
    result: DesktopProjectDataSchema,
    audit: "intent_outcome",
  },
  "project:reveal": {
    kind: "invoke",
    args: z.tuple([ProjectIdInputSchema]),
    result: z.boolean(),
    audit: "intent_outcome",
  },
  "project:archiveTasks": {
    kind: "invoke",
    args: z.tuple([ProjectIdInputSchema]),
    result: z.number().int().nonnegative(),
    audit: "intent_outcome",
  },
  "project:remove": {
    kind: "invoke",
    args: z.tuple([ProjectIdInputSchema]),
    result: z.boolean(),
    audit: "intent_outcome",
  },
} as const satisfies Record<string, DesktopInvokeContract>;
