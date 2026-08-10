import { z } from "zod";
import { type LocalTool, localToolResult } from "./mcp.js";
import {
  MAX_MEMORY_ENTITY_ALIASES,
  MAX_MEMORY_LINK_MARKERS as MAX_MEMORY_PAGE_LINK_MARKERS,
  MemoryEntityIdSchema,
  MemoryEntityNameSchema,
  MemoryGraphEdgeSchema,
  MemoryLinkDiagnosticSchema,
  MemoryLinkGraphBuilder,
  normalizeMemoryEntityKey,
} from "./memory-links.js";

export const MEMORY_SCHEMA_VERSION = 1;
export const MAX_MEMORY_PAGES = 2_048;
export const MAX_MEMORY_PAGE_CHARS = 64_000;
export const MAX_MEMORY_TOTAL_CHARS = 8_000_000;
export const MAX_MEMORY_TOTAL_LINK_MARKERS = 10_000;
export const MAX_MEMORY_SEARCH_RESULTS = 50;
export const MAX_MEMORY_VERSIONS = 100;
export const MAX_MEMORY_DIFF_CHARS = 128_000;
const TimestampSchema = z.string().datetime({ offset: true });
const MemoryContentSchema = z
  .string()
  .max(MAX_MEMORY_PAGE_CHARS)
  .refine((value) => !value.includes("\0"), "Memory content cannot contain NUL bytes");

export const MemoryPageSchema = z
  .object({
    id: MemoryEntityIdSchema,
    title: MemoryEntityNameSchema,
    aliases: z.array(MemoryEntityNameSchema).max(MAX_MEMORY_ENTITY_ALIASES),
    content: MemoryContentSchema,
    revision: z.number().int().positive(),
    createdAt: TimestampSchema,
    updatedAt: TimestampSchema,
  })
  .strict();

export const MemoryPageSummarySchema = MemoryPageSchema.omit({ content: true });

export const MemoryDocumentSchema = z
  .object({
    schemaVersion: z.literal(MEMORY_SCHEMA_VERSION),
    generation: z.number().int().nonnegative(),
    pages: z.array(MemoryPageSchema).max(MAX_MEMORY_PAGES),
  })
  .strict()
  .superRefine((document, context) => {
    const ids = new Set<string>();
    const nameOwners = new Map<string, string>();
    let totalContentChars = 0;
    let totalLinkMarkers = 0;

    for (const [pageIndex, page] of document.pages.entries()) {
      if (ids.has(page.id)) {
        context.addIssue({
          code: "custom",
          message: `Duplicate Memory page id: ${page.id}`,
          path: ["pages", pageIndex, "id"],
        });
      }
      ids.add(page.id);
      totalContentChars += page.content.length;
      const linkMarkers = countLinkMarkers(page.content);
      totalLinkMarkers += linkMarkers;
      if (linkMarkers > MAX_MEMORY_PAGE_LINK_MARKERS) {
        context.addIssue({
          code: "custom",
          message: `A Memory page can contain at most ${MAX_MEMORY_PAGE_LINK_MARKERS} link markers`,
          path: ["pages", pageIndex, "content"],
        });
      }

      for (const [nameIndex, name] of [page.title, ...page.aliases].entries()) {
        const key = normalizeMemoryEntityKey(name);
        const owner = nameOwners.get(key);
        if (owner) {
          context.addIssue({
            code: "custom",
            message:
              owner === page.id
                ? `Duplicate Memory name on page ${page.id}: ${name}`
                : `Memory name "${name}" conflicts with page ${owner}`,
            path:
              nameIndex === 0
                ? ["pages", pageIndex, "title"]
                : ["pages", pageIndex, "aliases", nameIndex - 1],
          });
        } else {
          nameOwners.set(key, page.id);
        }
      }
    }

    if (totalContentChars > MAX_MEMORY_TOTAL_CHARS) {
      context.addIssue({
        code: "custom",
        message: `Memory content can contain at most ${MAX_MEMORY_TOTAL_CHARS} characters`,
        path: ["pages"],
      });
    }
    if (totalLinkMarkers > MAX_MEMORY_TOTAL_LINK_MARKERS) {
      context.addIssue({
        code: "custom",
        message: `Memory can contain at most ${MAX_MEMORY_TOTAL_LINK_MARKERS} link markers`,
        path: ["pages"],
      });
    }
  });

export const MemoryCreateInputSchema = z
  .object({
    title: MemoryEntityNameSchema,
    aliases: z.array(MemoryEntityNameSchema).max(MAX_MEMORY_ENTITY_ALIASES).optional(),
    content: MemoryContentSchema,
  })
  .strict();

export const MemoryUpdateInputSchema = z
  .object({
    id: MemoryEntityIdSchema,
    expectedRevision: z.number().int().positive(),
    title: MemoryEntityNameSchema.optional(),
    aliases: z.array(MemoryEntityNameSchema).max(MAX_MEMORY_ENTITY_ALIASES).optional(),
    content: MemoryContentSchema.optional(),
  })
  .strict()
  .refine(
    (input) =>
      input.title !== undefined || input.aliases !== undefined || input.content !== undefined,
    { message: "Memory update must change at least one field" },
  );

export const MemoryDeleteInputSchema = z
  .object({
    id: MemoryEntityIdSchema,
    expectedRevision: z.number().int().positive(),
  })
  .strict();

export const MemorySearchInputSchema = z
  .object({
    query: z.string().trim().min(1).max(256),
    limit: z.number().int().min(1).max(MAX_MEMORY_SEARCH_RESULTS).default(20),
  })
  .strict();

export const MemoryVersionIdSchema = z.string().regex(/^[a-f0-9]{40}$/);

export const MemoryHistoryInputSchema = z
  .object({
    id: MemoryEntityIdSchema,
    limit: z.number().int().min(1).max(MAX_MEMORY_VERSIONS).default(20),
  })
  .strict();

export const MemoryGetVersionInputSchema = z
  .object({
    id: MemoryEntityIdSchema,
    version: MemoryVersionIdSchema,
  })
  .strict();

export const MemoryDiffInputSchema = z
  .object({
    id: MemoryEntityIdSchema,
    fromVersion: MemoryVersionIdSchema,
    toVersion: MemoryVersionIdSchema,
  })
  .strict()
  .refine((input) => input.fromVersion !== input.toVersion, {
    message: "Memory diff versions must differ",
    path: ["toVersion"],
  });

export const MemoryRestoreInputSchema = z
  .object({
    id: MemoryEntityIdSchema,
    expectedRevision: z.number().int().positive(),
    version: MemoryVersionIdSchema,
  })
  .strict();

export const MemoryVersionSummarySchema = z
  .object({
    version: MemoryVersionIdSchema,
    revision: z.number().int().positive(),
    operation: z.enum(["create", "update", "delete", "restore"]),
    committedAt: TimestampSchema,
  })
  .strict();

export const MemoryVersionSchema = MemoryVersionSummarySchema.extend({
  page: MemoryPageSchema,
  deleted: z.boolean(),
}).strict();

export const MemoryDiffSchema = z
  .object({
    id: MemoryEntityIdSchema,
    fromVersion: MemoryVersionIdSchema,
    toVersion: MemoryVersionIdSchema,
    unifiedDiff: z.string().max(MAX_MEMORY_DIFF_CHARS),
    truncated: z.boolean(),
  })
  .strict();

export const MemoryAgentInputSchema = z
  .discriminatedUnion("operation", [
    z.object({ operation: z.literal("list") }).strict(),
    z.object({ operation: z.literal("get"), id: MemoryEntityIdSchema }).strict(),
    z
      .object({
        operation: z.literal("search"),
        query: z.string().trim().min(1).max(256),
        limit: z.number().int().min(1).max(MAX_MEMORY_SEARCH_RESULTS).optional(),
      })
      .strict(),
    z.object({ operation: z.literal("graph") }).strict(),
    z
      .object({
        operation: z.literal("create"),
        title: MemoryEntityNameSchema,
        aliases: z.array(MemoryEntityNameSchema).max(MAX_MEMORY_ENTITY_ALIASES).optional(),
        content: MemoryContentSchema,
      })
      .strict(),
    z
      .object({
        operation: z.literal("update"),
        id: MemoryEntityIdSchema,
        expectedRevision: z.number().int().positive(),
        title: MemoryEntityNameSchema.optional(),
        aliases: z.array(MemoryEntityNameSchema).max(MAX_MEMORY_ENTITY_ALIASES).optional(),
        content: MemoryContentSchema.optional(),
      })
      .strict(),
    z
      .object({
        operation: z.literal("delete"),
        id: MemoryEntityIdSchema,
        expectedRevision: z.number().int().positive(),
      })
      .strict(),
    z
      .object({
        operation: z.literal("history"),
        id: MemoryEntityIdSchema,
        limit: z.number().int().min(1).max(MAX_MEMORY_VERSIONS).optional(),
      })
      .strict(),
    z
      .object({
        operation: z.literal("get_version"),
        id: MemoryEntityIdSchema,
        version: MemoryVersionIdSchema,
      })
      .strict(),
    z
      .object({
        operation: z.literal("diff"),
        id: MemoryEntityIdSchema,
        fromVersion: MemoryVersionIdSchema,
        toVersion: MemoryVersionIdSchema,
      })
      .strict(),
    z
      .object({
        operation: z.literal("restore"),
        id: MemoryEntityIdSchema,
        expectedRevision: z.number().int().positive(),
        version: MemoryVersionIdSchema,
      })
      .strict(),
  ])
  .superRefine((input, context) => {
    if (
      input.operation === "update" &&
      input.title === undefined &&
      input.aliases === undefined &&
      input.content === undefined
    ) {
      context.addIssue({
        code: "custom",
        message: "Memory update must change at least one field",
      });
    }
    if (input.operation === "diff" && input.fromVersion === input.toVersion) {
      context.addIssue({
        code: "custom",
        message: "Memory diff versions must differ",
        path: ["toVersion"],
      });
    }
  });

export const MemoryGraphDiagnosticSchema = z
  .object({
    sourceEntityId: MemoryEntityIdSchema,
    diagnostic: MemoryLinkDiagnosticSchema,
  })
  .strict();

export const MemoryGraphSchema = z
  .object({
    generation: z.number().int().nonnegative(),
    pages: z.array(MemoryPageSummarySchema).max(MAX_MEMORY_PAGES),
    edges: z.array(MemoryGraphEdgeSchema).max(MAX_MEMORY_TOTAL_LINK_MARKERS),
    diagnostics: z.array(MemoryGraphDiagnosticSchema).max(MAX_MEMORY_TOTAL_LINK_MARKERS),
  })
  .strict();

export type MemoryPage = z.infer<typeof MemoryPageSchema>;
export type MemoryPageSummary = z.infer<typeof MemoryPageSummarySchema>;
export type MemoryDocument = z.infer<typeof MemoryDocumentSchema>;
export type MemoryCreateInput = z.infer<typeof MemoryCreateInputSchema>;
export type MemoryUpdateInput = z.infer<typeof MemoryUpdateInputSchema>;
export type MemoryDeleteInput = z.infer<typeof MemoryDeleteInputSchema>;
export type MemorySearchInput = z.input<typeof MemorySearchInputSchema>;
export type MemoryHistoryInput = z.input<typeof MemoryHistoryInputSchema>;
export type MemoryGetVersionInput = z.infer<typeof MemoryGetVersionInputSchema>;
export type MemoryDiffInput = z.infer<typeof MemoryDiffInputSchema>;
export type MemoryRestoreInput = z.infer<typeof MemoryRestoreInputSchema>;
export type MemoryVersionSummary = z.infer<typeof MemoryVersionSummarySchema>;
export type MemoryVersion = z.infer<typeof MemoryVersionSchema>;
export type MemoryDiff = z.infer<typeof MemoryDiffSchema>;
export type MemoryGraph = z.infer<typeof MemoryGraphSchema>;
export type MemoryAgentInput = z.infer<typeof MemoryAgentInputSchema>;
export type MemoryAgentMutation = Extract<
  MemoryAgentInput,
  { operation: "create" | "update" | "delete" | "restore" }
>;

type MaybePromise<T> = T | Promise<T>;

export interface MemoryBackend {
  create(input: MemoryCreateInput): MaybePromise<MemoryPage>;
  get(id: string): MaybePromise<MemoryPage | null>;
  list(): MaybePromise<MemoryPageSummary[]>;
  search(input: MemorySearchInput): MaybePromise<MemoryPage[]>;
  update(input: MemoryUpdateInput): MaybePromise<MemoryPage>;
  delete(input: MemoryDeleteInput): MaybePromise<MemoryPage>;
  graph(): MaybePromise<MemoryGraph>;
  history(input: MemoryHistoryInput): MaybePromise<MemoryVersionSummary[]>;
  getVersion(input: MemoryGetVersionInput): MaybePromise<MemoryVersion>;
  diff(input: MemoryDiffInput): MaybePromise<MemoryDiff>;
  restore(input: MemoryRestoreInput): MaybePromise<MemoryPage>;
}

type MemoryAgentBackend = Omit<MemoryBackend, "history" | "getVersion" | "diff" | "restore"> &
  Partial<Pick<MemoryBackend, "history" | "getVersion" | "diff" | "restore">>;

export function buildMemoryGraph(generation: number, pages: readonly MemoryPage[]): MemoryGraph {
  const parsedPages = pages.map((page) => MemoryPageSchema.parse(page));
  const entities = parsedPages.map((page) => ({
    id: page.id,
    title: page.title,
    aliases: page.aliases,
  }));
  const edges: MemoryGraph["edges"] = [];
  const diagnostics: MemoryGraph["diagnostics"] = [];
  if (entities.length === 0) {
    return MemoryGraphSchema.parse({
      generation,
      pages: [],
      edges,
      diagnostics,
    });
  }
  const graphBuilder = new MemoryLinkGraphBuilder(entities);
  for (const page of parsedPages) {
    const projection = graphBuilder.build({
      sourceEntityId: page.id,
      markdown: page.content,
    });
    edges.push(...projection.edges);
    diagnostics.push(
      ...projection.diagnostics.map((diagnostic) => ({
        sourceEntityId: page.id,
        diagnostic,
      })),
    );
  }
  return MemoryGraphSchema.parse({
    generation,
    pages: parsedPages.map((page) => summarizePage(page)).sort(comparePageSummaries),
    edges,
    diagnostics,
  });
}

export interface MemoryAgentToolAuditEvent {
  operation: MemoryAgentMutation["operation"];
  outcome: "denied" | "attempted" | "completed" | "failed";
  pageId?: string;
  expectedRevision?: number;
  characterCount?: number;
}

export interface MemoryAgentToolOptions {
  confirm(mutation: MemoryAgentMutation): Promise<boolean>;
  audit(event: MemoryAgentToolAuditEvent): void;
}

export function createMemoryAgentTool(
  store: MemoryAgentBackend,
  options: MemoryAgentToolOptions,
): LocalTool {
  return {
    name: "Memory",
    description:
      "Read and maintain the user's durable, versioned Markdown Memory. List, get, search, graph, history, version reads, and diffs are read-only. Create, update, delete, and restore always require explicit user confirmation.",
    inputSchema: z.toJSONSchema(MemoryAgentInputSchema) as Record<string, unknown>,
    async call(arguments_) {
      const input = MemoryAgentInputSchema.parse(arguments_);
      switch (input.operation) {
        case "list": {
          const pages = await store.list();
          return memoryReadResult("list", { pages });
        }
        case "get": {
          const page = await store.get(input.id);
          if (!page) {
            return localToolResult(`Memory page not found: ${input.id}`, {
              status: "not_found",
              operation: "get",
              id: input.id,
            });
          }
          return memoryReadResult("get", { page });
        }
        case "search": {
          const pages = await store.search({ query: input.query, limit: input.limit });
          return memoryReadResult("search", { pages });
        }
        case "graph": {
          const graph = await store.graph();
          return memoryReadResult("graph", { graph });
        }
        case "history": {
          if (!store.history) return memoryVersioningUnavailable("history");
          const versions = await store.history({ id: input.id, limit: input.limit });
          return memoryReadResult("history", { versions });
        }
        case "get_version": {
          if (!store.getVersion) return memoryVersioningUnavailable("get_version");
          const version = await store.getVersion({ id: input.id, version: input.version });
          return memoryReadResult("get_version", { version });
        }
        case "diff": {
          if (!store.diff) return memoryVersioningUnavailable("diff");
          const diff = await store.diff(input);
          return memoryReadResult("diff", { diff });
        }
        case "create":
        case "update":
        case "delete":
        case "restore":
          return applyMemoryAgentMutation(store, input, options);
      }
    },
  };
}

function memoryReadResult(
  operation: "list" | "get" | "search" | "graph" | "history" | "get_version" | "diff",
  value: object,
) {
  const result = { status: "ok", operation, ...value };
  return localToolResult(JSON.stringify(result), result);
}

function memoryVersioningUnavailable(operation: "history" | "get_version" | "diff") {
  return localToolResult("Memory versioning is unavailable on this execution path.", {
    status: "unsupported",
    operation,
  });
}

async function applyMemoryAgentMutation(
  store: MemoryAgentBackend,
  mutation: MemoryAgentMutation,
  options: MemoryAgentToolOptions,
) {
  const auditBase = memoryMutationAuditBase(mutation);
  let confirmed: boolean;
  try {
    confirmed = await options.confirm(mutation);
  } catch (error) {
    options.audit({ ...auditBase, outcome: "failed" });
    throw error;
  }
  if (!confirmed) {
    options.audit({ ...auditBase, outcome: "denied" });
    return localToolResult("The user declined the Memory change.", {
      status: "denied",
      operation: mutation.operation,
    });
  }

  options.audit({ ...auditBase, outcome: "attempted" });
  try {
    let page: MemoryPage;
    if (mutation.operation === "create") {
      const { operation: _operation, ...input } = mutation;
      page = await store.create(input);
    } else if (mutation.operation === "update") {
      const { operation: _operation, ...input } = mutation;
      page = await store.update(input);
    } else if (mutation.operation === "delete") {
      const { operation: _operation, ...input } = mutation;
      page = await store.delete(input);
    } else {
      if (!store.restore) return memoryVersioningUnavailableMutation(options, auditBase);
      const { operation: _operation, ...input } = mutation;
      page = await store.restore(input);
    }
    const summary = summarizePage(page);
    options.audit({ ...auditBase, outcome: "completed", pageId: page.id });
    const result = { status: "applied", operation: mutation.operation, page: summary };
    return localToolResult(
      `Memory ${mutation.operation} completed for page ${page.id} at revision ${page.revision}.`,
      result,
    );
  } catch (error) {
    options.audit({ ...auditBase, outcome: "failed" });
    throw error;
  }
}

function memoryVersioningUnavailableMutation(
  options: MemoryAgentToolOptions,
  auditBase: Omit<MemoryAgentToolAuditEvent, "outcome">,
) {
  options.audit({ ...auditBase, outcome: "failed" });
  return localToolResult("Memory version restore is unavailable on this execution path.", {
    status: "unsupported",
    operation: "restore",
  });
}

function memoryMutationAuditBase(
  mutation: MemoryAgentMutation,
): Omit<MemoryAgentToolAuditEvent, "outcome"> {
  return {
    operation: mutation.operation,
    ...(mutation.operation === "create" ? {} : { pageId: mutation.id }),
    ...(mutation.operation === "create" ? {} : { expectedRevision: mutation.expectedRevision }),
    ...((mutation.operation === "create" || mutation.operation === "update") &&
    mutation.content !== undefined
      ? { characterCount: mutation.content.length }
      : {}),
  };
}

function summarizePage(page: MemoryPage): MemoryPageSummary {
  const { content: _content, ...summary } = page;
  return summary;
}

function comparePageSummaries(left: MemoryPageSummary, right: MemoryPageSummary): number {
  return compareText(left.title, right.title) || compareText(left.id, right.id);
}

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function countLinkMarkers(value: string): number {
  let count = 0;
  let offset = 0;
  while (true) {
    const markerOffset = value.indexOf("[[", offset);
    if (markerOffset === -1) return count;
    count += 1;
    offset = markerOffset + 2;
  }
}
