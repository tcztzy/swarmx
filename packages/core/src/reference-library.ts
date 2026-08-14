import { z } from "zod";
import { type LocalTool, localToolResult } from "./local-tool-contracts.js";

export const MAX_REFERENCE_QUERY_CHARS = 256;
export const MAX_REFERENCE_SEARCH_RESULTS = 20;
export const MAX_REFERENCE_PAGE_CHARS = 32_000;
export const MAX_REFERENCE_PATH_CHARS = 4_096;
export const ReferenceSourceIdSchema = z.enum(["zim", "zotero"]);

const ReferenceQuerySchema = z
  .string()
  .trim()
  .min(1)
  .max(MAX_REFERENCE_QUERY_CHARS)
  .refine((value) => !value.includes("\0"), "Reference query cannot contain NUL bytes");
const ReferencePathSchema = z
  .string()
  .trim()
  .min(1)
  .max(MAX_REFERENCE_PATH_CHARS)
  .refine((value) => !value.includes("\0"), "Reference path cannot contain NUL bytes");

export const ReferenceLibraryRequestSchema = z.discriminatedUnion("operation", [
  z
    .object({
      operation: z.literal("status"),
      source: ReferenceSourceIdSchema.optional(),
    })
    .strict(),
  z
    .object({
      operation: z.literal("search"),
      source: ReferenceSourceIdSchema.optional(),
      query: ReferenceQuerySchema,
      limit: z.number().int().min(1).max(MAX_REFERENCE_SEARCH_RESULTS).default(10),
    })
    .strict(),
  z
    .object({
      operation: z.literal("get"),
      source: ReferenceSourceIdSchema.optional(),
      path: ReferencePathSchema,
      maxChars: z
        .number()
        .int()
        .min(1)
        .max(MAX_REFERENCE_PAGE_CHARS)
        .default(MAX_REFERENCE_PAGE_CHARS),
    })
    .strict(),
]);

const LegacyZimReferenceSourceSchema = z
  .object({
    fileName: z.string().min(1).max(2_048),
    fileSize: z.number().int().nonnegative(),
    title: z.string().max(2_048).nullable().optional(),
    language: z.string().max(2_048).nullable().optional(),
    date: z.string().max(2_048).nullable().optional(),
    description: z.string().max(2_048).nullable().optional(),
  })
  .strict();

const ZimReferenceSourceSchema = LegacyZimReferenceSourceSchema.extend({
  id: z.literal("zim"),
  kind: z.literal("zim"),
  name: z.string().min(1).max(2_048),
});

const ZoteroReferenceSourceSchema = z
  .object({
    id: z.literal("zotero"),
    kind: z.literal("zotero"),
    name: z.string().min(1).max(2_048),
    endpoint: z.url().max(MAX_REFERENCE_PATH_CHARS),
  })
  .strict();

const ReferenceSourceSchema = z.union([ZimReferenceSourceSchema, ZoteroReferenceSourceSchema]);

const ReferenceMatchSchema = z
  .object({
    source: ReferenceSourceIdSchema,
    path: ReferencePathSchema,
    title: z.string().max(2_048),
    url: z.url().max(MAX_REFERENCE_PATH_CHARS).optional(),
    snippet: z.string().max(2_048).optional(),
  })
  .strict();

export const ReferenceLibraryResultSchema = z.discriminatedUnion("operation", [
  z
    .object({
      operation: z.literal("status"),
      sources: z.array(ReferenceSourceSchema).min(1).max(2),
      source: LegacyZimReferenceSourceSchema.optional(),
    })
    .strict(),
  z
    .object({
      operation: z.literal("search"),
      source: ReferenceSourceIdSchema,
      query: ReferenceQuerySchema,
      mode: z.enum(["full_text", "suggestion", "zotero"]),
      estimatedMatches: z.number().int().nonnegative(),
      matches: z.array(ReferenceMatchSchema).max(MAX_REFERENCE_SEARCH_RESULTS),
    })
    .strict(),
  z
    .object({
      operation: z.literal("get"),
      source: ReferenceSourceIdSchema,
      path: ReferencePathSchema,
      title: z.string().max(2_048),
      mimeType: z.string().min(1).max(128),
      text: z.string().max(MAX_REFERENCE_PAGE_CHARS),
      truncated: z.boolean(),
      url: z.url().max(MAX_REFERENCE_PATH_CHARS).optional(),
    })
    .strict(),
]);

export type ReferenceLibraryRequest = z.infer<typeof ReferenceLibraryRequestSchema>;
export type ReferenceLibraryResult = z.infer<typeof ReferenceLibraryResultSchema>;
export type ReferenceSourceId = z.infer<typeof ReferenceSourceIdSchema>;

export interface ReferenceLibraryBackend {
  request(request: ReferenceLibraryRequest): Promise<ReferenceLibraryResult>;
}

export class ReferenceLibraryUnavailableError extends Error {
  constructor() {
    super("Reference Library is not configured on this execution path.");
    this.name = "ReferenceLibraryUnavailableError";
  }
}

export function createReferenceLibraryAgentTool(backend: ReferenceLibraryBackend): LocalTool {
  return {
    name: "ReferenceLibrary",
    description:
      "Read configured objective ZIM and Zotero sources. Zotero requests must explicitly select their source. This tool cannot search the Web, download, fetch arbitrary URLs, read Zotero attachments/full text, mutate a source, or write Memory.",
    inputSchema: z.toJSONSchema(ReferenceLibraryRequestSchema) as Record<string, unknown>,
    async call(arguments_) {
      const request = ReferenceLibraryRequestSchema.parse(arguments_);
      try {
        const result = ReferenceLibraryResultSchema.parse(await backend.request(request));
        if (result.operation !== request.operation) {
          throw new Error("Reference Library response operation mismatch.");
        }
        if (
          request.source &&
          (result.operation === "status"
            ? result.sources.length !== 1 || result.sources[0]?.id !== request.source
            : result.source !== request.source)
        ) {
          throw new Error("Reference Library response source mismatch.");
        }
        return localToolResult(JSON.stringify({ status: "ok", ...result }), {
          status: "ok",
          ...result,
        });
      } catch (error) {
        if (error instanceof ReferenceLibraryUnavailableError) {
          return localToolResult(error.message, {
            status: "unsupported",
            operation: request.operation,
          });
        }
        throw error;
      }
    },
  };
}
