import { createHash } from "node:crypto";
import { parseDocument, stringify } from "yaml";
import { z } from "zod";
import { PkbError } from "./errors.js";

export const DEFAULT_MAX_CONCEPT_BYTES = 128 * 1024;

const isoDateTime = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/u;
const sourceId = /^[a-z0-9][a-z0-9_-]{0,63}$/u;
const workspaceKey = /^[a-f0-9]{12}$/u;

const generatedSchema = z
  .object({
    at: z.string().regex(isoDateTime),
    by: z.string().trim().min(1).max(160),
  })
  .passthrough();

const sourceSchema = z
  .object({
    author: z.string().trim().min(1).max(200).optional(),
    id: z.string().regex(sourceId).optional(),
    last_modified: z.string().regex(isoDateTime).optional(),
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
    swarmx_scope: z.enum(["global", "workspace"]),
    swarmx_workspace: z.string().regex(workspaceKey).optional(),
    tags: z.array(z.string().trim().min(1).max(80)).max(32).optional(),
    title: z.string().trim().min(1).max(500),
    type: z.string().trim().min(1).max(120),
  })
  .passthrough();

export type PkbSource = z.infer<typeof sourceSchema>;
export type PkbConceptMetadata = z.infer<typeof metadataSchema> & {
  status: "draft" | "stable" | "deprecated";
  sources: PkbSource[];
  tags: string[];
};

export interface ParsedConcept {
  readonly body: string;
  readonly metadata: PkbConceptMetadata;
  readonly revision: string;
}

function invalid(message: string, cause?: unknown): PkbError {
  return new PkbError(message, "INVALID_CONCEPT", cause === undefined ? undefined : { cause });
}

function regexEscape(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}

function validatePortableBody(body: string, sources: readonly PkbSource[]): void {
  if (/\[\[[^\]]+\]\]/u.test(body)) {
    throw invalid("PKB concepts must use standard Markdown links, not Wikilinks");
  }
  if (/^\s*\^[a-z0-9-]+\s*$/imu.test(body)) {
    throw invalid("PKB concepts must not use Obsidian block references");
  }
  if (/<\/?(?:script|iframe|object|embed)\b/iu.test(body)) {
    throw invalid("PKB concepts must not contain executable embedded HTML");
  }

  const ids = new Set<string>();
  for (const source of sources) {
    if (source.id !== undefined) {
      if (ids.has(source.id)) throw invalid(`Duplicate PKB source id '${source.id}'`);
      ids.add(source.id);
    }
    if (!source.resource.includes("/references/conversations/")) continue;
    if (source.id === undefined) {
      throw invalid("Conversation sources require a stable source id");
    }
    const escaped = regexEscape(source.id);
    if (!new RegExp(`\\[\\^${escaped}\\]`, "u").test(body)) {
      throw invalid(`Conversation source '${source.id}' has no body citation`);
    }
    if (!new RegExp(`^\\[\\^${escaped}\\]:\\s+`, "mu").test(body)) {
      throw invalid(`Conversation source '${source.id}' has no footnote definition`);
    }
  }
}

function normalizeMetadata(input: unknown): PkbConceptMetadata {
  let parsed: z.infer<typeof metadataSchema>;
  try {
    parsed = metadataSchema.parse(input);
  } catch (error) {
    throw invalid("Invalid PKB concept frontmatter", error);
  }

  if (parsed.swarmx_scope === "workspace" && parsed.swarmx_workspace === undefined) {
    throw invalid("Workspace PKB concept requires swarmx_workspace");
  }
  if (parsed.swarmx_scope === "global" && parsed.swarmx_workspace !== undefined) {
    throw invalid("Global PKB concept must not carry swarmx_workspace");
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
  source: string,
  maxBytes: number = DEFAULT_MAX_CONCEPT_BYTES,
): ParsedConcept {
  if (Buffer.byteLength(source, "utf8") > maxBytes) {
    throw invalid(`PKB concept exceeds ${String(maxBytes)} bytes`);
  }
  const match = /^---\n([\s\S]*?)\n---\n([\s\S]*)$/u.exec(source);
  if (match?.[1] === undefined || match[2] === undefined) {
    throw invalid("PKB concept requires one leading YAML frontmatter block");
  }

  const document = parseDocument(match[1], { uniqueKeys: true });
  if (document.errors.length > 0) {
    throw invalid("Invalid PKB concept YAML", document.errors[0]);
  }
  let value: unknown;
  try {
    value = document.toJS({ maxAliasCount: 0 });
  } catch (error) {
    throw invalid("Invalid PKB concept YAML value", error);
  }
  const metadata = normalizeMetadata(value);
  validatePortableBody(match[2], metadata.sources);
  return { body: match[2], metadata, revision: conceptRevision(source) };
}

export function renderConcept(metadata: PkbConceptMetadata, body: string): string {
  const normalized = normalizeMetadata(metadata);
  validatePortableBody(body, normalized.sources);
  const yaml = stringify(normalized, { lineWidth: 0 }).trimEnd();
  const rendered = `---\n${yaml}\n---\n\n${body.trim()}\n`;
  parseConcept(rendered);
  return rendered;
}
