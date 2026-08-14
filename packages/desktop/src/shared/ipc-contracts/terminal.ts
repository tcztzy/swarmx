import { z } from "zod";
import type { DesktopEventContract, DesktopInvokeContract } from "./base.js";

const TerminalInputNumberSchema = z.custom<number>((value) => typeof value === "number", {
  message: "Expected a number",
});
const TerminalIdSchema = z.string();
const TerminalResultIdSchema = z.string().min(1);
const TerminalOutputIntegerSchema = z.number().int();

export const DesktopTerminalCreateInputSchema = z
  .object({
    id: TerminalIdSchema,
    cwd: z.string(),
    cols: TerminalInputNumberSchema.optional(),
    rows: TerminalInputNumberSchema.optional(),
  })
  .strict();

export const DesktopTerminalCreateResultSchema = z
  .object({
    id: TerminalResultIdSchema,
    pid: TerminalOutputIntegerSchema.nonnegative(),
  })
  .strict();

export const DesktopTerminalDataEventSchema = z
  .object({
    id: TerminalResultIdSchema,
    data: z.string(),
  })
  .strict();

export const DesktopTerminalExitEventSchema = z
  .object({
    id: TerminalResultIdSchema,
    exitCode: TerminalOutputIntegerSchema,
    signal: TerminalOutputIntegerSchema.optional(),
  })
  .strict();

const TerminalWrittenResultSchema = z.object({ written: z.boolean() }).strict();
const TerminalResizedResultSchema = z.object({ resized: z.boolean() }).strict();
const TerminalKilledResultSchema = z.object({ killed: z.boolean() }).strict();

export type DesktopTerminalCreateInput = z.infer<typeof DesktopTerminalCreateInputSchema>;
export type DesktopTerminalCreateResult = z.infer<typeof DesktopTerminalCreateResultSchema>;
export type DesktopTerminalDataEvent = z.infer<typeof DesktopTerminalDataEventSchema>;
export type DesktopTerminalExitEvent = z.infer<typeof DesktopTerminalExitEventSchema>;

export const TerminalInvokeContracts = {
  "terminal:create": {
    kind: "invoke",
    args: z.tuple([DesktopTerminalCreateInputSchema]),
    result: DesktopTerminalCreateResultSchema,
    audit: "semantic_only",
  },
  "terminal:write": {
    kind: "invoke",
    args: z.tuple([z.object({ id: TerminalIdSchema, data: z.string() }).strict()]),
    result: TerminalWrittenResultSchema,
    audit: "semantic_only",
  },
  "terminal:resize": {
    kind: "invoke",
    args: z.tuple([
      z
        .object({
          id: TerminalIdSchema,
          cols: TerminalInputNumberSchema,
          rows: TerminalInputNumberSchema,
        })
        .strict(),
    ]),
    result: TerminalResizedResultSchema,
    audit: "semantic_only",
  },
  "terminal:kill": {
    kind: "invoke",
    args: z.tuple([z.object({ id: TerminalIdSchema }).strict()]),
    result: TerminalKilledResultSchema,
    audit: "semantic_only",
  },
} as const satisfies Record<string, DesktopInvokeContract>;

export const TerminalEventContracts = {
  "terminal:data": {
    kind: "event",
    payload: DesktopTerminalDataEventSchema,
  },
  "terminal:exit": {
    kind: "event",
    payload: DesktopTerminalExitEventSchema,
  },
} as const satisfies Record<string, DesktopEventContract>;

export interface DesktopTerminalApi {
  createTerminal(params: DesktopTerminalCreateInput): Promise<DesktopTerminalCreateResult>;
  writeTerminal(id: string, data: string): Promise<{ written: boolean }>;
  resizeTerminal(id: string, cols: number, rows: number): Promise<{ resized: boolean }>;
  killTerminal(id: string): Promise<{ killed: boolean }>;
  onTerminalData(listener: (event: DesktopTerminalDataEvent) => void): () => void;
  onTerminalExit(listener: (event: DesktopTerminalExitEvent) => void): () => void;
}
