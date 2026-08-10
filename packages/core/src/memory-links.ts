import { z } from "zod";

export const MAX_MEMORY_MARKDOWN_CHARS = 1_000_000;
export const MAX_MEMORY_GRAPH_ENTITIES = 10_000;
export const MAX_MEMORY_ENTITY_ALIASES = 32;
export const MAX_MEMORY_LINK_MARKERS = 2_048;

const MAX_ENTITY_TEXT_CHARS = 256;
const MAX_LINK_CONTENT_CHARS = 512;

function hasControlCharacters(value: string): boolean {
  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index);
    if (code <= 0x1f || code === 0x7f) return true;
  }
  return false;
}

function hasMemoryLinkNameDelimiter(value: string): boolean {
  return value.includes("[") || value.includes("]") || value.includes("|") || value.includes("#");
}

export const MemoryEntityIdSchema = z
  .string()
  .trim()
  .min(1)
  .max(MAX_ENTITY_TEXT_CHARS)
  .refine((value) => !hasControlCharacters(value), "Entity id cannot contain control characters");

export const MemoryEntityNameSchema = z
  .string()
  .trim()
  .min(1)
  .max(MAX_ENTITY_TEXT_CHARS)
  .refine(
    (value) => !hasControlCharacters(value) && !hasMemoryLinkNameDelimiter(value),
    "Entity names cannot contain double-bracket link delimiters or control characters",
  );

const MemoryLinkTargetTextSchema = z.string().max(MAX_LINK_CONTENT_CHARS);
const MemoryLinkPartSchema = z.string().trim().min(1).max(MAX_ENTITY_TEXT_CHARS);

export const MemoryEntitySchema = z
  .object({
    id: MemoryEntityIdSchema,
    title: MemoryEntityNameSchema,
    aliases: z.array(MemoryEntityNameSchema).max(MAX_MEMORY_ENTITY_ALIASES).optional(),
  })
  .strict();

export const MemoryEntityRegistrySchema = z
  .array(MemoryEntitySchema)
  .min(1)
  .max(MAX_MEMORY_GRAPH_ENTITIES)
  .superRefine((entities, context) => {
    const ids = new Set<string>();
    for (const [index, entity] of entities.entries()) {
      if (ids.has(entity.id)) {
        context.addIssue({
          code: "custom",
          message: `Duplicate entity id: ${entity.id}`,
          path: [index, "id"],
        });
      }
      ids.add(entity.id);
    }
  });

const LinkedMarkdownSchema = z
  .string()
  .max(MAX_MEMORY_MARKDOWN_CHARS)
  .superRefine((markdown, context) => {
    let count = 0;
    let offset = 0;
    while (true) {
      const markerOffset = markdown.indexOf("[[", offset);
      if (markerOffset === -1) break;
      count += 1;
      if (count > MAX_MEMORY_LINK_MARKERS) {
        context.addIssue({
          code: "custom",
          message: `Markdown can contain at most ${MAX_MEMORY_LINK_MARKERS} link markers`,
        });
        return;
      }
      offset = markerOffset + 2;
    }
  });

export const MemoryLinkGraphProjectionInputSchema = z
  .object({
    sourceEntityId: MemoryEntityIdSchema,
    markdown: LinkedMarkdownSchema,
  })
  .strict();

export const MemoryGraphBuildInputSchema = MemoryLinkGraphProjectionInputSchema.extend({
  entities: MemoryEntityRegistrySchema,
}).superRefine((input, context) => {
  if (!input.entities.some((entity) => entity.id === input.sourceEntityId)) {
    context.addIssue({
      code: "custom",
      message: "sourceEntityId must identify an entity in the registry",
      path: ["sourceEntityId"],
    });
  }
});

export const MemoryLinkOccurrenceSchema = z
  .object({
    targetText: MemoryLinkTargetTextSchema,
    alias: MemoryLinkPartSchema.optional(),
    heading: MemoryLinkPartSchema.optional(),
    embedded: z.boolean(),
    start: z.number().int().nonnegative(),
    end: z.number().int().positive(),
  })
  .strict()
  .refine((occurrence) => occurrence.end > occurrence.start, {
    message: "Occurrence end must be after start",
    path: ["end"],
  });

export const MemoryGraphEdgeSchema = z
  .object({
    kind: z.literal("memory_link"),
    source: MemoryEntityIdSchema,
    target: MemoryEntityIdSchema,
    occurrences: z.array(MemoryLinkOccurrenceSchema).min(1).max(MAX_MEMORY_LINK_MARKERS),
  })
  .strict()
  .refine((edge) => edge.source !== edge.target, {
    message: "Memory graph edges must connect two different entities",
    path: ["target"],
  });

export const MemoryLinkDiagnosticKindSchema = z.enum([
  "unresolved",
  "ambiguous",
  "self_reference",
  "malformed",
]);

export const MemoryLinkDiagnosticSchema = z
  .object({
    kind: MemoryLinkDiagnosticKindSchema,
    targetText: MemoryLinkTargetTextSchema,
    candidateEntityIds: z.array(MemoryEntityIdSchema).max(MAX_MEMORY_GRAPH_ENTITIES).optional(),
    start: z.number().int().nonnegative(),
    end: z.number().int().positive(),
  })
  .strict()
  .refine((diagnostic) => diagnostic.end > diagnostic.start, {
    message: "Diagnostic end must be after start",
    path: ["end"],
  });

export const MemoryGraphBuildResultSchema = z
  .object({
    edges: z.array(MemoryGraphEdgeSchema).max(MAX_MEMORY_LINK_MARKERS),
    diagnostics: z.array(MemoryLinkDiagnosticSchema).max(MAX_MEMORY_LINK_MARKERS),
  })
  .strict();

export type MemoryEntity = z.infer<typeof MemoryEntitySchema>;
export type MemoryGraphBuildInput = z.infer<typeof MemoryGraphBuildInputSchema>;
export type MemoryLinkOccurrence = z.infer<typeof MemoryLinkOccurrenceSchema>;
export type MemoryGraphEdge = z.infer<typeof MemoryGraphEdgeSchema>;
export type MemoryLinkDiagnosticKind = z.infer<typeof MemoryLinkDiagnosticKindSchema>;
export type MemoryLinkDiagnostic = z.infer<typeof MemoryLinkDiagnosticSchema>;
export type MemoryGraphBuildResult = z.infer<typeof MemoryGraphBuildResultSchema>;
export type MemoryLinkGraphProjectionInput = z.infer<typeof MemoryLinkGraphProjectionInputSchema>;

interface ParsedMemoryLink extends MemoryLinkOccurrence {
  kind: "link";
}

interface MalformedMemoryLink {
  kind: "malformed";
  targetText: string;
  start: number;
  end: number;
}

type ScannedMemoryLink = ParsedMemoryLink | MalformedMemoryLink;

export function normalizeMemoryEntityKey(value: string): string {
  return value.trim().normalize("NFC").replace(/\.md$/i, "").toLocaleLowerCase("en-US");
}

function markerIsEscaped(line: string, markerStart: number): boolean {
  let backslashes = 0;
  for (let index = markerStart - 1; index >= 0 && line[index] === "\\"; index--) {
    backslashes += 1;
  }
  return backslashes % 2 === 1;
}

function findClosingBackticks(line: string, from: number, length: number): number {
  for (let index = from; index < line.length; index++) {
    if (line[index] !== "`") continue;
    let runLength = 1;
    while (line[index + runLength] === "`") runLength += 1;
    if (runLength === length) return index;
    index += runLength - 1;
  }
  return -1;
}

function malformedLink(targetText: string, start: number, end: number): MalformedMemoryLink {
  return {
    kind: "malformed",
    targetText: targetText.slice(0, MAX_LINK_CONTENT_CHARS),
    start,
    end,
  };
}

function parseMemoryLink(
  content: string,
  embedded: boolean,
  start: number,
  end: number,
): ScannedMemoryLink {
  if (content.length > MAX_LINK_CONTENT_CHARS || content.includes("[") || content.includes("]")) {
    return malformedLink(content, start, end);
  }

  const aliasDivider = content.indexOf("|");
  if (aliasDivider !== -1 && content.indexOf("|", aliasDivider + 1) !== -1) {
    return malformedLink(content, start, end);
  }

  const destination = (aliasDivider === -1 ? content : content.slice(0, aliasDivider)).trim();
  const aliasText = aliasDivider === -1 ? undefined : content.slice(aliasDivider + 1).trim();
  const headingDivider = destination.indexOf("#");
  const targetText = (
    headingDivider === -1 ? destination : destination.slice(0, headingDivider)
  ).trim();
  const headingText =
    headingDivider === -1 ? undefined : destination.slice(headingDivider + 1).trim();

  const alias = aliasText === undefined ? undefined : MemoryLinkPartSchema.safeParse(aliasText);
  const heading =
    headingText === undefined ? undefined : MemoryLinkPartSchema.safeParse(headingText);
  if (
    (targetText.length === 0 && headingText === undefined) ||
    targetText.length > MAX_ENTITY_TEXT_CHARS ||
    (alias !== undefined && !alias.success) ||
    (heading !== undefined && !heading.success)
  ) {
    return malformedLink(targetText, start, end);
  }

  return {
    kind: "link",
    targetText,
    ...(alias?.success ? { alias: alias.data } : {}),
    ...(heading?.success ? { heading: heading.data } : {}),
    embedded,
    start,
    end,
  };
}

function scanLine(line: string, lineOffset: number): ScannedMemoryLink[] {
  const links: ScannedMemoryLink[] = [];
  for (let index = 0; index < line.length; ) {
    if (line[index] === "`") {
      let runLength = 1;
      while (line[index + runLength] === "`") runLength += 1;
      const closing = findClosingBackticks(line, index + runLength, runLength);
      index = closing === -1 ? index + runLength : closing + runLength;
      continue;
    }

    const embedded = line[index] === "!" && line.slice(index + 1, index + 3) === "[[";
    const plain = line.slice(index, index + 2) === "[[";
    if (!embedded && !plain) {
      index += 1;
      continue;
    }

    const markerStart = index;
    const bracketStart = embedded ? index + 1 : index;
    if (markerIsEscaped(line, markerStart)) {
      index = bracketStart + 2;
      continue;
    }

    const closing = line.indexOf("]]", bracketStart + 2);
    if (closing === -1) {
      links.push(
        malformedLink(
          line.slice(bracketStart + 2),
          lineOffset + markerStart,
          lineOffset + line.length,
        ),
      );
      break;
    }

    const end = closing + 2;
    links.push(
      parseMemoryLink(
        line.slice(bracketStart + 2, closing),
        embedded,
        lineOffset + markerStart,
        lineOffset + end,
      ),
    );
    index = end;
  }
  return links;
}

function scanMemoryLinks(markdown: string): ScannedMemoryLink[] {
  const links: ScannedMemoryLink[] = [];
  let fence: { character: "`" | "~"; length: number } | undefined;
  let lineStart = 0;

  while (lineStart <= markdown.length) {
    const newline = markdown.indexOf("\n", lineStart);
    const lineEnd = newline === -1 ? markdown.length : newline;
    const rawLine = markdown.slice(lineStart, lineEnd);
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    const fenceMatch = /^ {0,3}(`{3,}|~{3,})/.exec(line);

    if (fence) {
      if (fenceMatch) {
        const marker = fenceMatch[1] as string;
        const suffix = line.slice((fenceMatch.index ?? 0) + fenceMatch[0].length);
        if (
          marker[0] === fence.character &&
          marker.length >= fence.length &&
          suffix.trim().length === 0
        ) {
          fence = undefined;
        }
      }
    } else if (fenceMatch) {
      const marker = fenceMatch[1] as string;
      fence = { character: marker[0] as "`" | "~", length: marker.length };
    } else {
      links.push(...scanLine(line, lineStart));
    }

    if (newline === -1) break;
    lineStart = newline + 1;
  }

  return links;
}

/**
 * Projects one registered entity's double-bracket Markdown links into directed knowledge
 * edges. Offsets use JavaScript UTF-16 string indices and `end` is exclusive.
 */
function indexMemoryEntities(entities: MemoryEntity[]): Map<string, Set<string>> {
  const entityIdsByKey = new Map<string, Set<string>>();
  for (const entity of entities) {
    for (const name of [entity.title, ...(entity.aliases ?? [])]) {
      const key = normalizeMemoryEntityKey(name);
      const ids = entityIdsByKey.get(key) ?? new Set<string>();
      ids.add(entity.id);
      entityIdsByKey.set(key, ids);
    }
  }
  return entityIdsByKey;
}

function projectMemoryLinkEdges(
  sourceEntity: MemoryEntity,
  markdown: string,
  entityIdsByKey: Map<string, Set<string>>,
): MemoryGraphBuildResult {
  const edgesByTarget = new Map<string, MemoryGraphEdge>();
  const diagnostics: MemoryLinkDiagnostic[] = [];
  for (const scanned of scanMemoryLinks(markdown)) {
    if (scanned.kind === "malformed") {
      diagnostics.push(scanned);
      continue;
    }

    const targetKey = normalizeMemoryEntityKey(scanned.targetText);
    const candidateEntityIds =
      targetKey.length === 0
        ? [sourceEntity.id]
        : [...(entityIdsByKey.get(targetKey) ?? [])].sort((left, right) =>
            left < right ? -1 : left > right ? 1 : 0,
          );

    if (candidateEntityIds.length === 0) {
      diagnostics.push({
        kind: "unresolved",
        targetText: scanned.targetText,
        start: scanned.start,
        end: scanned.end,
      });
      continue;
    }
    if (candidateEntityIds.length > 1) {
      diagnostics.push({
        kind: "ambiguous",
        targetText: scanned.targetText,
        candidateEntityIds,
        start: scanned.start,
        end: scanned.end,
      });
      continue;
    }

    const target = candidateEntityIds[0] as string;
    if (target === sourceEntity.id) {
      diagnostics.push({
        kind: "self_reference",
        targetText: scanned.targetText,
        candidateEntityIds: [target],
        start: scanned.start,
        end: scanned.end,
      });
      continue;
    }

    const occurrence: MemoryLinkOccurrence = {
      targetText: scanned.targetText,
      ...(scanned.alias === undefined ? {} : { alias: scanned.alias }),
      ...(scanned.heading === undefined ? {} : { heading: scanned.heading }),
      embedded: scanned.embedded,
      start: scanned.start,
      end: scanned.end,
    };
    const edge = edgesByTarget.get(target);
    if (edge) {
      edge.occurrences.push(occurrence);
    } else {
      edgesByTarget.set(target, {
        kind: "memory_link",
        source: sourceEntity.id,
        target,
        occurrences: [occurrence],
      });
    }
  }

  return MemoryGraphBuildResultSchema.parse({
    edges: [...edgesByTarget.values()],
    diagnostics,
  });
}

export class MemoryLinkGraphBuilder {
  private readonly entities: MemoryEntity[];
  private readonly entityById: Map<string, MemoryEntity>;
  private readonly entityIdsByKey: Map<string, Set<string>>;

  constructor(entities: MemoryEntity[]) {
    this.entities = MemoryEntityRegistrySchema.parse(entities);
    this.entityById = new Map(this.entities.map((entity) => [entity.id, entity]));
    this.entityIdsByKey = indexMemoryEntities(this.entities);
  }

  build(input: MemoryLinkGraphProjectionInput): MemoryGraphBuildResult {
    const parsed = MemoryLinkGraphProjectionInputSchema.parse(input);
    const sourceEntity = this.entityById.get(parsed.sourceEntityId);
    if (!sourceEntity) {
      MemoryEntityIdSchema.refine((id) => this.entityById.has(id), {
        message: "sourceEntityId must identify an entity in the registry",
      }).parse(parsed.sourceEntityId);
      throw new Error("Validated source entity is missing");
    }
    return projectMemoryLinkEdges(sourceEntity, parsed.markdown, this.entityIdsByKey);
  }
}

export function buildMemoryLinkEdges(input: MemoryGraphBuildInput): MemoryGraphBuildResult {
  const parsed = MemoryGraphBuildInputSchema.parse(input);
  return new MemoryLinkGraphBuilder(parsed.entities).build({
    sourceEntityId: parsed.sourceEntityId,
    markdown: parsed.markdown,
  });
}
