import { createHash } from "node:crypto";
import { LineCounter, parseDocument, stringify } from "yaml";
import { z } from "zod";
import { MemoryError } from "./errors.js";
import { inspectMarkdown, positionAt } from "./markdown-body.js";

export const DEFAULT_MAX_CONCEPT_BYTES = 128 * 1024;

export const memoryDateTimeSchema = z.iso.datetime({ offset: true });
const sourceId = /^[a-z0-9][a-z0-9_-]{0,63}$/u;
const workspaceKey = /^[a-f0-9]{12}$/u;

const generatedSchema = z
  .object({
    at: memoryDateTimeSchema,
    by: z.string().trim().min(1).max(160),
  })
  .passthrough();

const sourceSchema = z
  .object({
    author: z.string().trim().min(1).max(200).optional(),
    id: z.string().regex(sourceId).optional(),
    last_modified: memoryDateTimeSchema.optional(),
    resource: z.string().trim().min(1).max(2_048),
    title: z.string().trim().min(1).max(500).optional(),
  })
  .passthrough();

const metadataSchema = z
  .object({
    aliases: z.array(z.string().trim().min(1).max(200)).max(32).optional(),
    description: z.string().trim().min(1).max(500),
    generated: generatedSchema,
    sources: z.array(sourceSchema).max(32).optional(),
    status: z.enum(["draft", "stable", "deprecated"]).optional(),
    stale_after: memoryDateTimeSchema.optional(),
    verified: z.union([generatedSchema, z.array(generatedSchema).min(1)]).optional(),
    swarmx_scope: z.enum(["global", "workspace"]),
    swarmx_workspace: z.string().regex(workspaceKey).optional(),
    tags: z.array(z.string().trim().min(1).max(80)).max(32).optional(),
    title: z.string().trim().min(1).max(500),
    type: z.string().trim().min(1).max(120),
  })
  .passthrough();

export type MemorySource = z.infer<typeof sourceSchema>;
export type MemoryConceptMetadata = z.infer<typeof metadataSchema> & {
  status: "draft" | "stable" | "deprecated";
  sources: MemorySource[];
  tags: string[];
};

export interface ParsedConcept {
  readonly body: string;
  readonly metadata: MemoryConceptMetadata;
  readonly revision: string;
}

function invalid(message: string, cause?: unknown): MemoryError {
  return new MemoryError(message, "INVALID_CONCEPT", cause === undefined ? undefined : { cause });
}

function normalizeMetadata(input: unknown): MemoryConceptMetadata {
  let parsed: z.infer<typeof metadataSchema>;
  try {
    parsed = metadataSchema.parse(input);
  } catch (error) {
    throw invalid("Invalid memory concept frontmatter", error);
  }

  if (parsed.swarmx_scope === "workspace" && parsed.swarmx_workspace === undefined) {
    throw invalid("Workspace memory concept requires swarmx_workspace");
  }
  if (parsed.swarmx_scope === "global" && parsed.swarmx_workspace !== undefined) {
    throw invalid("Global memory concept must not carry swarmx_workspace");
  }

  return {
    ...parsed,
    sources: parsed.sources ?? [],
    status: parsed.status ?? "stable",
    tags: parsed.tags ?? [],
  };
}

export function conceptRevision(source: string | Uint8Array): string {
  return `sha256:${createHash("sha256").update(source).digest("hex")}`;
}

export function parseConcept(
  input: string | Uint8Array,
  maxBytes: number = DEFAULT_MAX_CONCEPT_BYTES,
): ParsedConcept {
  if (Buffer.byteLength(input) > maxBytes) {
    throw invalid(`memory concept exceeds ${String(maxBytes)} bytes`);
  }
  const source = decodeConcept(input);
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/u.exec(source);
  if (match?.[1] === undefined || match[2] === undefined) {
    throw invalid("memory concept requires one leading YAML frontmatter block");
  }

  const lineCounter = new LineCounter();
  const document = parseDocument(match[1], { uniqueKeys: true, lineCounter, prettyErrors: false });
  if (document.errors.length > 0) {
    throw new MemoryError("Invalid memory concept YAML", "INVALID_CONCEPT", {
      issues: document.errors.map((error) => {
        const position = lineCounter.linePos(error.pos[0]);
        return {
          ruleId: "document.yaml",
          severity: "error",
          line: position.line + 1,
          column: position.col,
          message: error.message,
        };
      }),
    });
  }
  let value: unknown;
  try {
    value = document.toJS({ maxAliasCount: 0 });
  } catch (error) {
    throw invalid("Invalid memory concept YAML value", error);
  }
  let metadata: MemoryConceptMetadata;
  try {
    metadata = normalizeMetadata(value);
  } catch (error) {
    if (!(error instanceof MemoryError) || !(error.cause instanceof z.ZodError)) throw error;
    throw new MemoryError(error.message, error.code, {
      cause: error.cause,
      issues: error.cause.issues.map((issue) => {
        const node = document.getIn(issue.path, true) as { range?: number[] } | undefined;
        const position = lineCounter.linePos(node?.range?.[0] ?? 0);
        return {
          ruleId: "metadata.invalid",
          severity: "error",
          line: position.line + 1,
          column: position.col,
          message: `${issue.path.join(".")}: ${issue.message}`,
        };
      }),
    });
  }
  const ids = new Set<string>();
  for (const [index, entry] of metadata.sources.entries()) {
    if (entry.id === undefined) continue;
    if (ids.has(entry.id)) {
      const node = document.getIn(["sources", index, "id"], true) as { range: number[] };
      const position = lineCounter.linePos(node.range[0] ?? 0);
      throw new MemoryError("Duplicate memory source ID.", "INVALID_CONCEPT", {
        issues: [
          {
            ruleId: "source.duplicate",
            severity: "error",
            line: position.line + 1,
            column: position.col,
            message: `Duplicate source ID '${entry.id}'.`,
          },
        ],
      });
    }
    ids.add(entry.id);
  }
  const body = match[2];
  const bodyLine = positionAt(source, source.length - body.length).line;
  if (body.trim().length === 0) throw invalid("Memory concept body must not be empty.");
  const { issues } = inspectMarkdown(body);
  if (issues.length > 0) {
    throw new MemoryError("Invalid memory Markdown.", "INVALID_CONCEPT", {
      issues: issues.map((issue) => ({ ...issue, line: issue.line + bodyLine - 1 })),
    });
  }
  return { body, metadata, revision: conceptRevision(input) };
}

export function decodeConcept(source: string | Uint8Array): string {
  if (typeof source === "string") return source;
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(source);
  } catch (error) {
    if (!(error instanceof TypeError)) throw error;
    throw new MemoryError("Memory document is not valid UTF-8.", "INVALID_CONCEPT", {
      cause: error,
      issues: [
        {
          ruleId: "document.encoding",
          severity: "error",
          line: 1,
          column: 1,
          message: "Memory document is not valid UTF-8.",
        },
      ],
    });
  }
}

export function renderConcept(metadata: MemoryConceptMetadata, body: string): string {
  const normalized = normalizeMetadata(metadata);
  const yaml = stringify(normalized, { lineWidth: 0 }).trimEnd();
  const rendered = `---\n${yaml}\n---\n\n${body.trim()}\n`;
  parseConcept(rendered);
  return rendered;
}
