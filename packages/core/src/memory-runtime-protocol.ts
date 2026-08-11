import { z } from "zod";
import {
  GlobalMemoryDeleteInputSchema,
  GlobalMemoryWriteInputSchema,
  MAX_MEMORY_PAGES,
  MAX_MEMORY_SEARCH_RESULTS,
  MAX_MEMORY_VERSIONS,
  MEMORY_SCHEMA_VERSION,
  MemoryCreateInputSchema,
  MemoryDeleteInputSchema,
  MemoryDiffInputSchema,
  MemoryDiffSchema,
  MemoryDocumentSchema,
  MemoryGetVersionInputSchema,
  MemoryHistoryInputSchema,
  MemoryPageSchema,
  MemoryPageSummarySchema,
  MemoryRestoreInputSchema,
  MemorySearchInputSchema,
  MemoryUpdateInputSchema,
  MemoryVersionSchema,
  MemoryVersionSummarySchema,
} from "./memory.js";
import { GlobalMemoryFileSchema } from "./personal-memory.js";

export const MEMORY_RUNTIME_PROTOCOL_VERSION = 1 as const;
export const MEMORY_RUNTIME_SERVER_NAME = "swarmx-mem" as const;
export const MEMORY_RUNTIME_TOOL_NAME = "swarmx_memory" as const;
export const MEMORY_RUNTIME_MAX_RESPONSE_BYTES = 12 * 1024 * 1024;

const RequestBase = {
  protocolVersion: z.literal(MEMORY_RUNTIME_PROTOCOL_VERSION),
};

const MemoryRuntimePageSetSchema = z
  .array(MemoryPageSchema)
  .max(MAX_MEMORY_PAGES)
  .superRefine((pages, context) => {
    const document = MemoryDocumentSchema.safeParse({
      schemaVersion: MEMORY_SCHEMA_VERSION,
      generation: 0,
      pages,
    });
    if (!document.success) {
      context.addIssue({
        code: "custom",
        message: "Memory page set exceeds its aggregate bounds or contains conflicts",
      });
    }
  });

export const MemoryRuntimeRequestSchema = z.discriminatedUnion("operation", [
  z.object({ ...RequestBase, operation: z.literal("list") }).strict(),
  z
    .object({
      ...RequestBase,
      operation: z.literal("get"),
      id: MemoryGetVersionInputSchema.shape.id,
    })
    .strict(),
  z
    .object({
      ...RequestBase,
      operation: z.literal("search"),
      ...MemorySearchInputSchema.shape,
    })
    .strict(),
  z.object({ ...RequestBase, operation: z.literal("snapshot") }).strict(),
  z.object({ ...RequestBase, operation: z.literal("global_get") }).strict(),
  z
    .object({
      ...RequestBase,
      operation: z.literal("global_save"),
      ...GlobalMemoryWriteInputSchema.shape,
    })
    .strict(),
  z
    .object({
      ...RequestBase,
      operation: z.literal("global_forget"),
      ...GlobalMemoryDeleteInputSchema.shape,
    })
    .strict(),
  z
    .object({
      ...RequestBase,
      operation: z.literal("create"),
      ...MemoryCreateInputSchema.shape,
    })
    .strict(),
  z
    .object({
      ...RequestBase,
      operation: z.literal("update"),
      ...MemoryUpdateInputSchema.shape,
    })
    .strict()
    .refine(
      (input) =>
        input.title !== undefined || input.aliases !== undefined || input.content !== undefined,
      { message: "Memory update must change at least one field" },
    ),
  z
    .object({
      ...RequestBase,
      operation: z.literal("delete"),
      ...MemoryDeleteInputSchema.shape,
    })
    .strict(),
  z
    .object({
      ...RequestBase,
      operation: z.literal("history"),
      ...MemoryHistoryInputSchema.shape,
    })
    .strict(),
  z
    .object({
      ...RequestBase,
      operation: z.literal("get_version"),
      ...MemoryGetVersionInputSchema.shape,
    })
    .strict(),
  z
    .object({ ...RequestBase, operation: z.literal("diff"), ...MemoryDiffInputSchema.shape })
    .strict()
    .refine((input) => input.fromVersion !== input.toVersion, {
      message: "Memory diff versions must differ",
      path: ["toVersion"],
    }),
  z
    .object({
      ...RequestBase,
      operation: z.literal("restore"),
      ...MemoryRestoreInputSchema.shape,
    })
    .strict(),
]);

export const MemoryRuntimeOperationSchema = MemoryRuntimeRequestSchema.options.reduce(
  (schema, option) => schema.or(option.shape.operation),
  z.never() as z.ZodType<string>,
);

const MemoryRuntimeMutationResultSchema = z
  .object({
    page: MemoryPageSchema,
    version: MemoryVersionSummarySchema.shape.version,
  })
  .strict();

const SuccessBase = {
  protocolVersion: z.literal(MEMORY_RUNTIME_PROTOCOL_VERSION),
  ok: z.literal(true),
};

const MemoryRuntimeSuccessResponseSchema = z.discriminatedUnion("operation", [
  z
    .object({
      ...SuccessBase,
      operation: z.literal("list"),
      result: z.object({ pages: z.array(MemoryPageSummarySchema).max(MAX_MEMORY_PAGES) }).strict(),
    })
    .strict(),
  z
    .object({
      ...SuccessBase,
      operation: z.literal("get"),
      result: z.object({ page: MemoryPageSchema.nullable() }).strict(),
    })
    .strict(),
  z
    .object({
      ...SuccessBase,
      operation: z.literal("search"),
      result: z
        .object({ pages: z.array(MemoryPageSchema).max(MAX_MEMORY_SEARCH_RESULTS) })
        .strict(),
    })
    .strict(),
  z
    .object({
      ...SuccessBase,
      operation: z.literal("snapshot"),
      result: z
        .object({
          generation: z.number().int().nonnegative(),
          pages: MemoryRuntimePageSetSchema,
        })
        .strict(),
    })
    .strict(),
  z
    .object({
      ...SuccessBase,
      operation: z.literal("global_get"),
      result: z
        .object({
          user: GlobalMemoryFileSchema,
          memory: GlobalMemoryFileSchema,
        })
        .strict(),
    })
    .strict(),
  z
    .object({
      ...SuccessBase,
      operation: z.literal("global_save"),
      result: z
        .object({
          file: GlobalMemoryFileSchema,
          version: MemoryVersionSummarySchema.shape.version,
        })
        .strict(),
    })
    .strict(),
  z
    .object({
      ...SuccessBase,
      operation: z.literal("global_forget"),
      result: z
        .object({
          file: GlobalMemoryFileSchema,
          version: MemoryVersionSummarySchema.shape.version,
        })
        .strict(),
    })
    .strict(),
  z
    .object({
      ...SuccessBase,
      operation: z.literal("create"),
      result: MemoryRuntimeMutationResultSchema,
    })
    .strict(),
  z
    .object({
      ...SuccessBase,
      operation: z.literal("update"),
      result: MemoryRuntimeMutationResultSchema,
    })
    .strict(),
  z
    .object({
      ...SuccessBase,
      operation: z.literal("delete"),
      result: MemoryRuntimeMutationResultSchema,
    })
    .strict(),
  z
    .object({
      ...SuccessBase,
      operation: z.literal("restore"),
      result: MemoryRuntimeMutationResultSchema,
    })
    .strict(),
  z
    .object({
      ...SuccessBase,
      operation: z.literal("history"),
      result: z
        .object({ versions: z.array(MemoryVersionSummarySchema).max(MAX_MEMORY_VERSIONS) })
        .strict(),
    })
    .strict(),
  z
    .object({
      ...SuccessBase,
      operation: z.literal("get_version"),
      result: z.object({ version: MemoryVersionSchema }).strict(),
    })
    .strict(),
  z
    .object({
      ...SuccessBase,
      operation: z.literal("diff"),
      result: z.object({ diff: MemoryDiffSchema }).strict(),
    })
    .strict(),
]);

export const MemoryRuntimeErrorCodeSchema = z.enum([
  "not_found",
  "conflict",
  "invalid_input",
  "corrupt",
  "busy",
  "unsupported",
  "internal",
]);

const MemoryRuntimeErrorResponseSchema = z
  .object({
    protocolVersion: z.literal(MEMORY_RUNTIME_PROTOCOL_VERSION),
    operation: z.enum([
      "list",
      "get",
      "search",
      "snapshot",
      "global_get",
      "global_save",
      "global_forget",
      "create",
      "update",
      "delete",
      "history",
      "get_version",
      "diff",
      "restore",
    ]),
    ok: z.literal(false),
    error: z
      .object({
        code: MemoryRuntimeErrorCodeSchema,
        message: z.string().min(1).max(512),
      })
      .strict(),
  })
  .strict();

export const MemoryRuntimeToolResponseSchema = z.union([
  MemoryRuntimeSuccessResponseSchema,
  MemoryRuntimeErrorResponseSchema,
]);

export type MemoryRuntimeRequest = z.infer<typeof MemoryRuntimeRequestSchema>;
export type MemoryRuntimeOperation = MemoryRuntimeRequest["operation"];
export type MemoryRuntimeToolResponse = z.infer<typeof MemoryRuntimeToolResponseSchema>;
type MemoryRuntimeSuccessResponse = Extract<MemoryRuntimeToolResponse, { ok: true }>;
export type MemoryRuntimeResult<Request extends MemoryRuntimeRequest> = Extract<
  MemoryRuntimeSuccessResponse,
  { operation: Request["operation"] }
>["result"];
