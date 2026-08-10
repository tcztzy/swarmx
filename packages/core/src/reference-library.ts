import { z } from "zod";
import { type LocalTool, localToolResult } from "./mcp.js";

export const MAX_REFERENCE_QUERY_CHARS = 256;
export const MAX_REFERENCE_SEARCH_RESULTS = 20;
export const MAX_REFERENCE_PAGE_CHARS = 32_000;
export const MAX_REFERENCE_PATH_CHARS = 4_096;

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
  z.object({ operation: z.literal("status") }).strict(),
  z
    .object({
      operation: z.literal("search"),
      query: ReferenceQuerySchema,
      limit: z.number().int().min(1).max(MAX_REFERENCE_SEARCH_RESULTS).default(10),
    })
    .strict(),
  z
    .object({
      operation: z.literal("get"),
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

const ReferenceSourceSchema = z
  .object({
    fileName: z.string().min(1).max(2_048),
    fileSize: z.number().int().nonnegative(),
    title: z.string().max(2_048).nullable().optional(),
    language: z.string().max(2_048).nullable().optional(),
    date: z.string().max(2_048).nullable().optional(),
    description: z.string().max(2_048).nullable().optional(),
  })
  .strict();

const ReferenceMatchSchema = z
  .object({
    path: ReferencePathSchema,
    title: z.string().max(2_048),
  })
  .strict();

export const ReferenceLibraryResultSchema = z.discriminatedUnion("operation", [
  z.object({ operation: z.literal("status"), source: ReferenceSourceSchema }).strict(),
  z
    .object({
      operation: z.literal("search"),
      query: ReferenceQuerySchema,
      mode: z.enum(["full_text", "suggestion"]),
      estimatedMatches: z.number().int().nonnegative(),
      matches: z.array(ReferenceMatchSchema).max(MAX_REFERENCE_SEARCH_RESULTS),
    })
    .strict(),
  z
    .object({
      operation: z.literal("get"),
      path: ReferencePathSchema,
      title: z.string().max(2_048),
      mimeType: z.string().min(1).max(128),
      text: z.string().max(MAX_REFERENCE_PAGE_CHARS),
      truncated: z.boolean(),
    })
    .strict(),
]);

export type ReferenceLibraryRequest = z.infer<typeof ReferenceLibraryRequestSchema>;
export type ReferenceLibraryResult = z.infer<typeof ReferenceLibraryResultSchema>;

export interface ReferenceLibraryBackend {
  request(request: ReferenceLibraryRequest): Promise<ReferenceLibraryResult>;
}

export class ReferenceLibraryUnavailableError extends Error {
  constructor() {
    super("Offline Reference Library is not configured on this execution path.");
    this.name = "ReferenceLibraryUnavailableError";
  }
}

export function createReferenceLibraryAgentTool(backend: ReferenceLibraryBackend): LocalTool {
  return {
    name: "ReferenceLibrary",
    description:
      "Read the configured objective offline reference archive. Status, search, and bounded plaintext article reads are available; this tool cannot download, create, update, delete, or write Memory.",
    inputSchema: z.toJSONSchema(ReferenceLibraryRequestSchema) as Record<string, unknown>,
    async call(arguments_) {
      const request = ReferenceLibraryRequestSchema.parse(arguments_);
      try {
        const result = ReferenceLibraryResultSchema.parse(await backend.request(request));
        if (result.operation !== request.operation) {
          throw new Error("Reference Library response operation mismatch.");
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
