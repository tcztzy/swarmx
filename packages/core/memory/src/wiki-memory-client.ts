import { isAbsolute } from "node:path";
import { z } from "zod";
import { MemoryError } from "./errors.js";

const value = z.string().trim().min(1).max(4_096);
const scopeSchema = z
  .string()
  .min(1)
  .max(4_096)
  .refine((path) => isAbsolute(path) && !path.includes("\0"));
const filtersSchema = z.strictObject({
  atom_type: value.optional(),
  project_module: value.optional(),
  area: value.optional(),
  language: value.optional(),
  task_type: value.optional(),
  error_pattern: value.optional(),
  tags: value.optional(),
  subject: z.union([value, z.array(value).max(64)]).optional(),
});
const searchSchema = z.strictObject({
  query: value.max(1_000),
  datasets: z.array(value).max(64).optional(),
  filters: filtersSchema.optional(),
  scoreThreshold: z.number().finite().min(0).max(1).optional(),
  maxResults: z.number().int().min(1).max(50).default(8),
  maxChars: z.number().int().min(80).max(20_000).default(600),
  sections: z.array(z.enum(["frontmatter", "body"])).optional(),
});
const writeSchema = z.strictObject({
  target: value,
  write: z.strictObject({
    datasetId: value,
    name: value.max(180),
    text: z.string().trim().min(20).max(200_000),
    path: value.max(500).optional(),
    metadata: filtersSchema
      .partial()
      .extend({ priority: z.enum(["P0", "P1", "P2"]).optional() })
      .optional(),
  }),
  userRequested: z.boolean().optional(),
});
const searchResultSchema = z
  .object({
    query: z.string(),
    totalRecords: z.number().int().nonnegative(),
    records: z
      .array(
        z
          .object({
            datasetId: z.string(),
            documentId: z.string(),
            documentName: z.string(),
            score: z.number().finite(),
            priority: z.string(),
            content: z.string().optional(),
          })
          .passthrough(),
      )
      .max(50),
  })
  .passthrough();
const writeResultSchema = z
  .object({
    ok: z.literal(true),
    documentId: value,
  })
  .passthrough();
const envelopeSchema = z
  .object({
    isError: z.boolean().optional(),
    content: z.array(z.object({ type: z.literal("text"), text: z.string() })).length(1),
  })
  .passthrough();
const MAX_RESPONSE_BYTES = 2 * 1024 * 1024;
const REFUSALS = new Set([
  "write-gate-refused",
  "quality-judge-unavailable",
  "quality-judge-rejected",
  "duplicate-suspected",
  "inline-body-too-large",
]);

export type WikiMemorySearchRequest = z.input<typeof searchSchema>;
export type WikiMemoryWriteRequest = z.input<typeof writeSchema>;
export type WikiMemorySearchResult = z.output<typeof searchResultSchema>;
export type WikiMemoryWriteResult = z.output<typeof writeResultSchema>;

export interface WikiMemoryTransport {
  callTool(
    request: { name: "search_memory" | "write_memory"; arguments: Record<string, unknown> },
    signal: AbortSignal,
  ): Promise<unknown>;
}

export class WikiMemoryError extends MemoryError {
  constructor(
    message: string,
    readonly outcome: "refused" | "unknown",
    readonly response?: unknown,
    options?: ErrorOptions,
  ) {
    super(message, "IO_ERROR", options);
    this.name = "WikiMemoryError";
  }
}

export class WikiMemoryClient {
  private readonly scopes: string[];

  constructor(
    config: { scopes: readonly string[] },
    private readonly transport: WikiMemoryTransport,
  ) {
    this.scopes = z.array(scopeSchema).min(1).max(64).parse(config.scopes);
  }

  search(input: WikiMemorySearchRequest, signal: AbortSignal): Promise<WikiMemorySearchResult> {
    return this.query(input, signal);
  }

  read(input: WikiMemorySearchRequest, signal: AbortSignal): Promise<WikiMemorySearchResult> {
    return this.query(
      {
        ...input,
        maxChars: input.maxChars ?? 2_000,
        sections: input.sections ?? ["frontmatter", "body"],
      },
      signal,
    );
  }

  async write(input: WikiMemoryWriteRequest, signal: AbortSignal): Promise<WikiMemoryWriteResult> {
    signal.throwIfAborted();
    const { userRequested, ...request } = writeSchema.parse(input);
    return this.call(
      "write_memory",
      {
        ...request,
        scopes: [...this.scopes],
        ...(userRequested === undefined ? {} : { gate: { userRequested } }),
      },
      writeResultSchema,
      signal,
    );
  }

  private async query(input: WikiMemorySearchRequest, signal: AbortSignal) {
    signal.throwIfAborted();
    const request = searchSchema.parse(input);
    const frontmatterOnly =
      request.sections?.includes("frontmatter") && !request.sections.includes("body");
    const schema = frontmatterOnly
      ? searchResultSchema
      : searchResultSchema.refine(
          (result) => result.records.every((record) => record.content !== undefined),
          "Body search results must include content.",
        );
    return this.call(
      "search_memory",
      { ...request, scopes: [...this.scopes], fullContent: false },
      schema,
      signal,
    );
  }

  private async call<T>(
    name: "search_memory" | "write_memory",
    args: Record<string, unknown>,
    schema: { parse(value: unknown): T },
    signal: AbortSignal,
  ): Promise<T> {
    let response: unknown;
    try {
      response = await this.transport.callTool({ name, arguments: args }, signal);
      signal.throwIfAborted();
      const envelope = envelopeSchema.parse(response);
      const text = envelope.content[0]?.text ?? "";
      if (Buffer.byteLength(text) > MAX_RESPONSE_BYTES) throw new Error("Response exceeds 2 MiB.");
      const payload: unknown = JSON.parse(text);
      if (
        typeof payload === "object" &&
        payload !== null &&
        (("ok" in payload && payload.ok === false) || ("error" in payload && payload.error))
      ) {
        const refused =
          "error" in payload && typeof payload.error === "string" && REFUSALS.has(payload.error);
        throw new WikiMemoryError(
          "Wiki memory returned a failed request.",
          refused ? "refused" : "unknown",
          payload,
        );
      }
      if (envelope.isError) {
        throw new WikiMemoryError("Wiki memory tool returned an error.", "unknown", response);
      }
      return schema.parse(payload);
    } catch (error) {
      if (error instanceof WikiMemoryError) throw error;
      throw new WikiMemoryError(
        "Wiki memory response is unavailable or invalid; inspect before retrying a write.",
        "unknown",
        response,
        { cause: error },
      );
    }
  }
}
