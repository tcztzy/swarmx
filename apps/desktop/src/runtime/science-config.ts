import {
  MAX_TYPST_PDF_BYTES,
  MAX_TYPST_SOURCE_BYTES,
  type Config as ScienceConfig,
} from "@swarmx/dsh-science/core";
import { z } from "zod";

const CommandSchema = z.string().max(4_096);
export const MAX_SCIENCE_CARRIER_CONFIG_BYTES = 64 * 1024;
const ScienceCarrierConfigSchema = z
  .object({
    embedArtifactMetadata: z.boolean().optional(),
    maxArtifactBytes: z.number().int().min(1).max(Number.MAX_SAFE_INTEGER).optional(),
    maxCellOutputBytes: z.number().int().min(1).max(1_000_000).optional(),
    maxExportBytes: z.number().int().min(1).max(10_000_000).optional(),
    maxNotebookDocumentBytes: z.number().int().min(1).max(10_000_000).optional(),
    notebookRuntime: z.enum(["isolated", "jupymcp"]).optional(),
    processGraceMs: z.number().int().min(1).max(60_000).optional(),
    jupymcpArgs: z.array(z.string().max(4_096)).max(128).optional(),
    jupymcpCommand: CommandSchema.optional(),
    jupymcpRequestTimeoutMs: z.number().int().min(1_000).max(3_600_000).optional(),
    pythonCommand: CommandSchema.optional(),
    typstCommand: CommandSchema.optional(),
    writingPreviewRuntimeCommand: CommandSchema.optional(),
    typstInitialCompileTimeoutMs: z.number().int().min(100).max(60_000).optional(),
    typstMaxDiagnosticsBytes: z
      .number()
      .int()
      .min(1_024)
      .max(1024 * 1024)
      .optional(),
    typstMaxPdfBytes: z.number().int().min(1_024).max(MAX_TYPST_PDF_BYTES).optional(),
    typstMaxSourceBytes: z.number().int().min(1_024).max(MAX_TYPST_SOURCE_BYTES).optional(),
  })
  .strict()
  .superRefine((value, context) => {
    if (Buffer.byteLength(JSON.stringify(value)) > MAX_SCIENCE_CARRIER_CONFIG_BYTES) {
      context.addIssue({
        code: "custom",
        message: `Science carrier configuration exceeds ${String(MAX_SCIENCE_CARRIER_CONFIG_BYTES)} bytes.`,
      });
    }
  });

type ParsedScienceCarrierConfig = z.infer<typeof ScienceCarrierConfigSchema>;
export type ScienceCarrierConfig = {
  readonly [Key in keyof ParsedScienceCarrierConfig]?: Exclude<
    ParsedScienceCarrierConfig[Key],
    undefined
  >;
};

export function projectScienceCarrierConfig(config: ScienceConfig): ScienceCarrierConfig {
  const { root: _root, ...candidate } = config;
  const known = ScienceCarrierConfigSchema.keyof().options;
  return definedScienceCarrierConfig(
    ScienceCarrierConfigSchema.parse(
      Object.fromEntries(known.flatMap((key) => (key in candidate ? [[key, candidate[key]]] : []))),
    ),
  );
}

export function parseScienceCarrierConfig(raw: string | undefined): ScienceCarrierConfig {
  if (raw === undefined) return {};
  try {
    if (Buffer.byteLength(raw) > MAX_SCIENCE_CARRIER_CONFIG_BYTES) {
      throw new Error("Science carrier configuration exceeds the byte limit.");
    }
    return definedScienceCarrierConfig(ScienceCarrierConfigSchema.parse(JSON.parse(raw)));
  } catch (cause) {
    throw new Error("SWARMX_SCIENCE_CONFIG is not a valid bounded Science configuration.", {
      cause,
    });
  }
}

export function serializeScienceCarrierConfig(config: ScienceCarrierConfig): string {
  return JSON.stringify(ScienceCarrierConfigSchema.parse(config));
}

function definedScienceCarrierConfig(config: ParsedScienceCarrierConfig): ScienceCarrierConfig {
  return Object.fromEntries(
    Object.entries(config).filter(([, value]) => value !== undefined),
  ) as ScienceCarrierConfig;
}
