import { z } from "zod";
import {
  artifactKindSchema,
  figureLibrarySchema,
  researchRecordKindSchema,
  researchRecordStatusSchema,
  type ScienceArtifact,
  type ScienceArtifactPreview,
  type ScienceDocument,
  type ScienceExperiment,
  type ScienceFigure,
  type ScienceNotebook,
  type ScienceProject,
  type ScienceResearchRecord,
  type ScienceRun,
  scienceDocumentFormatSchema,
  scienceRelationTypeSchema,
} from "./contracts.js";
import { ScienceError } from "./errors.js";
import { formatScienceResourceId, type ScienceResourceKind } from "./resource-id.js";
import type {
  ResolvedScienceResource,
  ScienceResourceRef,
  ScienceResourceResolver,
} from "./resource-resolver.js";

export const MAX_RESOURCE_BATCH_HEAD = 50;
export const MAX_RESOURCE_LIST_ITEMS = 20;
export const MAX_RESOURCE_NEIGHBORS = 100;
export const MAX_RESOURCE_SELECT_COLUMNS = 32;
export const MAX_RESOURCE_SELECT_ROWS = 100;
export const MAX_RESOURCE_SELECT_TEXT_CHARS = 16 * 1024;
export const MAX_RESOURCE_SELECT_SOURCE_BYTES = 64 * 1024;

const MAX_METADATA_TEXT_CHARS = 500;
const MAX_TABLE_CELL_CHARS = 1_000;
const resourceIdSchema = z.string().min(1).max(1_024);

export const scienceResourceRefSchema = z.strictObject({
  id: resourceIdSchema,
  exactId: resourceIdSchema,
  kind: z.enum([
    "project",
    "notebook",
    "artifact",
    "document",
    "figure",
    "record",
    "experiment",
    "run",
  ]),
  title: z.string().min(1).max(240),
  revision: z.number().int().positive(),
  digest: z
    .string()
    .regex(/^sha256:[0-9a-f]{64}$/u)
    .nullable(),
});

export const scienceResourceCapabilitySchema = z.enum(["get", "select", "neighbors"]);

export const scienceResourceHeadRequestSchema = z.strictObject({ id: resourceIdSchema });

export const scienceResourceBatchHeadRequestSchema = z.strictObject({
  ids: z.array(resourceIdSchema).min(1).max(MAX_RESOURCE_BATCH_HEAD),
});

export const scienceResourceGetRequestSchema = z.strictObject({
  id: resourceIdSchema,
  projection: z.literal("metadata"),
});

const selectedColumnsSchema = z
  .array(z.string().max(4_096))
  .min(1)
  .max(MAX_RESOURCE_SELECT_COLUMNS)
  .refine((columns) => new Set(columns).size === columns.length, "selected columns must be unique");

export const scienceResourceSelectRequestSchema = z.discriminatedUnion("format", [
  z.strictObject({
    id: resourceIdSchema,
    format: z.literal("table"),
    offset: z.number().int().nonnegative().default(0),
    limit: z.number().int().min(1).max(MAX_RESOURCE_SELECT_ROWS).default(25),
    columns: selectedColumnsSchema.optional(),
  }),
  z.strictObject({
    id: resourceIdSchema,
    format: z.literal("text"),
    offset: z.number().int().nonnegative().default(0),
    limit: z.number().int().min(1).max(MAX_RESOURCE_SELECT_TEXT_CHARS).default(4_096),
  }),
]);

export const scienceResourceRelationSchema = z.enum([
  ...scienceRelationTypeSchema.options,
  "contains",
  "has_hypothesis",
  "has_run",
]);

export const scienceResourceNeighborsRequestSchema = z.strictObject({
  id: resourceIdSchema,
  relations: z
    .array(scienceResourceRelationSchema)
    .min(1)
    .max(16)
    .refine((relations) => new Set(relations).size === relations.length, "relations must be unique")
    .optional(),
  limit: z.number().int().min(1).max(MAX_RESOURCE_NEIGHBORS).default(50),
});

export const scienceResourceHeadSchema = z.strictObject({
  ref: scienceResourceRefSchema,
  summary: z.string().min(1).max(300),
  capabilities: z.array(scienceResourceCapabilitySchema).min(1).max(3),
});

export const scienceResourceBatchHeadSchema = z.strictObject({
  heads: z.array(scienceResourceHeadSchema).min(1).max(MAX_RESOURCE_BATCH_HEAD),
});

const boundedStringsSchema = z.strictObject({
  items: z.array(z.string().max(500)).max(MAX_RESOURCE_LIST_ITEMS),
  total: z.number().int().nonnegative(),
  truncated: z.boolean(),
});
const boundedRefsSchema = z.strictObject({
  items: z.array(scienceResourceRefSchema).max(MAX_RESOURCE_LIST_ITEMS),
  total: z.number().int().nonnegative(),
  truncated: z.boolean(),
});

export const scienceResourceMetadataSchema = z.discriminatedUnion("kind", [
  z.strictObject({
    kind: z.literal("project"),
    counts: z.strictObject({
      notebooks: z.number().int().nonnegative(),
      artifacts: z.number().int().nonnegative(),
      documents: z.number().int().nonnegative(),
      figures: z.number().int().nonnegative(),
      records: z.number().int().nonnegative(),
      experiments: z.number().int().nonnegative(),
      runs: z.number().int().nonnegative(),
    }),
    createdAt: z.number().int().nonnegative(),
    updatedAt: z.number().int().nonnegative(),
  }),
  z.strictObject({
    kind: z.literal("notebook"),
    cellCount: z.number().int().nonnegative(),
    inputArtifactCount: z.number().int().nonnegative(),
    outputArtifactCount: z.number().int().nonnegative(),
    createdAt: z.number().int().nonnegative(),
    updatedAt: z.number().int().nonnegative(),
  }),
  z.strictObject({
    kind: z.literal("artifact"),
    artifactKind: artifactKindSchema,
    mime: z.string().min(1).max(200),
    size: z.number().int().nonnegative(),
    license: z.string().max(200).nullable(),
    runRef: scienceResourceRefSchema.nullable(),
    sourceRefs: boundedRefsSchema,
    createdAt: z.number().int().nonnegative(),
    updatedAt: z.number().int().nonnegative(),
  }),
  z.strictObject({
    kind: z.literal("document"),
    format: scienceDocumentFormatSchema,
    contentRevision: z.number().int().positive(),
    sourceHash: z.string().regex(/^sha256:[0-9a-f]{64}$/u),
    diagnosticCount: z.number().int().nonnegative(),
    proposalCount: z.number().int().nonnegative(),
    createdAt: z.number().int().nonnegative(),
    updatedAt: z.number().int().nonnegative(),
  }),
  z.strictObject({
    kind: z.literal("figure"),
    library: figureLibrarySchema,
    codeRevision: z.number().int().positive(),
    codeHash: z.string().regex(/^sha256:[0-9a-f]{64}$/u),
    artifactRef: scienceResourceRefSchema.nullable(),
    objectCount: z.number().int().nonnegative(),
    proposalCount: z.number().int().nonnegative(),
    createdAt: z.number().int().nonnegative(),
    updatedAt: z.number().int().nonnegative(),
  }),
  z.strictObject({
    kind: z.literal("record"),
    recordKind: researchRecordKindSchema,
    status: researchRecordStatusSchema,
    summary: z.string().max(MAX_METADATA_TEXT_CHARS),
    tags: boundedStringsSchema,
    sourceRefs: boundedRefsSchema,
    createdAt: z.number().int().nonnegative(),
    updatedAt: z.number().int().nonnegative(),
  }),
  z.strictObject({
    kind: z.literal("experiment"),
    status: z.enum(["defined", "active", "completed"]),
    summary: z.string().max(MAX_METADATA_TEXT_CHARS),
    tags: boundedStringsSchema,
    hypothesisRefs: boundedRefsSchema,
    runRefs: boundedRefsSchema,
    createdAt: z.number().int().nonnegative(),
    updatedAt: z.number().int().nonnegative(),
  }),
  z.strictObject({
    kind: z.literal("run"),
    status: z.enum(["running", "succeeded", "failed", "cancelled"]),
    experimentRef: scienceResourceRefSchema.nullable(),
    metricKeys: boundedStringsSchema,
    artifactRefs: boundedRefsSchema,
    startedAt: z.number().int().nonnegative(),
    finishedAt: z.number().int().nonnegative().nullable(),
  }),
]);

export const scienceResourceGetResultSchema = z.strictObject({
  ref: scienceResourceRefSchema,
  projection: z.literal("metadata"),
  metadata: scienceResourceMetadataSchema,
});

const selectedColumnSchema = z.strictObject({
  id: z.string().regex(/^column-\d+$/u),
  name: z.string().max(500),
  type: z.enum(["string", "number", "boolean"]),
});
const selectedScalarSchema = z.union([
  z.string().max(MAX_TABLE_CELL_CHARS),
  z.number(),
  z.boolean(),
  z.null(),
]);

export const scienceResourceSelectResultSchema = z.discriminatedUnion("kind", [
  z.strictObject({
    ref: scienceResourceRefSchema,
    kind: z.literal("table"),
    columns: z.array(selectedColumnSchema).max(MAX_RESOURCE_SELECT_COLUMNS),
    rows: z
      .array(z.array(selectedScalarSchema).max(MAX_RESOURCE_SELECT_COLUMNS))
      .max(MAX_RESOURCE_SELECT_ROWS),
    total: z.number().int().nonnegative(),
    offset: z.number().int().nonnegative(),
    returned: z.number().int().nonnegative().max(MAX_RESOURCE_SELECT_ROWS),
    truncated: z.boolean(),
    nextOffset: z.number().int().nonnegative().nullable(),
  }),
  z.strictObject({
    ref: scienceResourceRefSchema,
    kind: z.literal("text"),
    text: z.string().max(MAX_RESOURCE_SELECT_TEXT_CHARS),
    total: z.number().int().nonnegative(),
    offset: z.number().int().nonnegative(),
    returned: z.number().int().nonnegative().max(MAX_RESOURCE_SELECT_TEXT_CHARS),
    truncated: z.boolean(),
    nextOffset: z.number().int().nonnegative().nullable(),
  }),
  z.strictObject({
    ref: scienceResourceRefSchema,
    kind: z.literal("unavailable"),
    reason: z.enum(["unsupported", "too-large"]),
  }),
]);

export const scienceResourceNeighborSchema = z.strictObject({
  relation: scienceResourceRelationSchema,
  direction: z.enum(["incoming", "outgoing"]),
  target: scienceResourceRefSchema,
});

export const scienceResourceNeighborsResultSchema = z.strictObject({
  ref: scienceResourceRefSchema,
  neighbors: z.array(scienceResourceNeighborSchema).max(MAX_RESOURCE_NEIGHBORS),
  total: z.number().int().nonnegative(),
  truncated: z.boolean(),
});

export type ScienceResourceHeadRequest = z.infer<typeof scienceResourceHeadRequestSchema>;
export type ScienceResourceBatchHeadRequest = z.infer<typeof scienceResourceBatchHeadRequestSchema>;
export type ScienceResourceGetRequest = z.infer<typeof scienceResourceGetRequestSchema>;
export type ScienceResourceSelectRequest = z.infer<typeof scienceResourceSelectRequestSchema>;
export type ScienceResourceNeighborsRequest = z.infer<typeof scienceResourceNeighborsRequestSchema>;
export type ScienceResourceHead = z.infer<typeof scienceResourceHeadSchema>;
export type ScienceResourceBatchHead = z.infer<typeof scienceResourceBatchHeadSchema>;
export type ScienceResourceGetResult = z.infer<typeof scienceResourceGetResultSchema>;
export type ScienceResourceSelectResult = z.infer<typeof scienceResourceSelectResultSchema>;
export type ScienceResourceNeighborsResult = z.infer<typeof scienceResourceNeighborsResultSchema>;

function truncate(value: string, max = MAX_METADATA_TEXT_CHARS): string {
  return value.length <= max ? value : `${value.slice(0, max - 3)}...`;
}

function boundedStrings(values: readonly string[]) {
  const items = values.slice(0, MAX_RESOURCE_LIST_ITEMS).map((value) => truncate(value));
  return { items, total: values.length, truncated: values.length > items.length };
}

function typedRef(
  resolver: ScienceResourceResolver,
  kind: ScienceResourceKind,
  entityId: string,
): ScienceResourceRef | null {
  try {
    return resolver.resolve(formatScienceResourceId(kind, entityId)).ref;
  } catch (error) {
    if (
      error instanceof ScienceError &&
      (error.code === "RESOURCE_NOT_FOUND" || error.code === "RESOURCE_KIND_MISMATCH")
    ) {
      return null;
    }
    throw error;
  }
}

function boundedRefs(
  resolver: ScienceResourceResolver,
  entityIds: readonly string[],
  kind?: ScienceResourceKind,
) {
  const items: ScienceResourceRef[] = [];
  for (const entityId of entityIds) {
    const ref = kind
      ? typedRef(resolver, kind, entityId)
      : resolver.resolveUntypedEntityId(entityId)?.ref;
    if (ref && items.length < MAX_RESOURCE_LIST_ITEMS) items.push(ref);
  }
  return { items, total: entityIds.length, truncated: entityIds.length > items.length };
}

function optionalRef(
  resolver: ScienceResourceResolver,
  kind: ScienceResourceKind,
  entityId: string | null,
): ScienceResourceRef | null {
  return entityId === null ? null : typedRef(resolver, kind, entityId);
}

function projectEntityCount(
  resolver: ScienceResourceResolver,
  projectId: string,
  kind: ScienceResourceKind,
): number {
  const snapshot = resolver.snapshot;
  if (kind === "notebook")
    return snapshot.notebooks.filter((item) => item.projectId === projectId).length;
  if (kind === "artifact")
    return snapshot.artifacts.filter((item) => item.projectId === projectId).length;
  if (kind === "document")
    return snapshot.documents.filter((item) => item.projectId === projectId).length;
  if (kind === "figure")
    return snapshot.figures.filter((item) => item.projectId === projectId).length;
  if (kind === "record")
    return snapshot.records.filter((item) => item.projectId === projectId).length;
  if (kind === "experiment")
    return snapshot.experiments.filter((item) => item.projectId === projectId).length;
  if (kind === "run") return snapshot.runs.filter((item) => item.projectId === projectId).length;
  return 0;
}

function supportsSelect(entity: ScienceArtifact): boolean {
  return (
    entity.size <= MAX_RESOURCE_SELECT_SOURCE_BYTES &&
    (entity.mime.startsWith("text/") || entity.mime === "application/json")
  );
}

function summary(resolver: ScienceResourceResolver, resource: ResolvedScienceResource): string {
  const entity = resource.entity;
  if (resource.kind === "project") {
    const project = entity as ScienceProject;
    const count = (
      ["notebook", "artifact", "document", "figure", "record", "experiment", "run"] as const
    ).reduce((total, kind) => total + projectEntityCount(resolver, project.id, kind), 0);
    return `Science project, ${count} resources`;
  }
  if (resource.kind === "notebook")
    return `Science notebook, ${(entity as ScienceNotebook).cells.length} cells`;
  if (resource.kind === "artifact") {
    const artifact = entity as ScienceArtifact;
    return `${artifact.kind} artifact, ${artifact.mime}, ${artifact.size} bytes`;
  }
  if (resource.kind === "document") {
    const document = entity as ScienceDocument;
    return `${document.format} document, content revision ${document.contentRevision}`;
  }
  if (resource.kind === "figure") {
    const figure = entity as ScienceFigure;
    return `${figure.library} figure source, code revision ${figure.codeRevision}`;
  }
  if (resource.kind === "record") {
    const record = entity as ScienceResearchRecord;
    return `${record.kind} record, ${record.status}`;
  }
  if (resource.kind === "experiment") {
    const experiment = entity as ScienceExperiment;
    return `Science experiment, ${experiment.status}, ${experiment.runIds.length} runs`;
  }
  const run = entity as ScienceRun;
  return `Experiment run, ${run.status}, ${run.artifactIds.length} artifacts`;
}

export function scienceResourceHead(
  resolver: ScienceResourceResolver,
  request: ScienceResourceHeadRequest,
): ScienceResourceHead {
  const resource = resolver.resolve(request.id);
  const capabilities: z.infer<typeof scienceResourceCapabilitySchema>[] = ["get"];
  if (resource.kind === "artifact" && supportsSelect(resource.entity as ScienceArtifact)) {
    capabilities.push("select");
  }
  capabilities.push("neighbors");
  return scienceResourceHeadSchema.parse({
    ref: resource.ref,
    summary: summary(resolver, resource),
    capabilities,
  });
}

export function scienceResourceBatchHead(
  resolver: ScienceResourceResolver,
  request: ScienceResourceBatchHeadRequest,
): ScienceResourceBatchHead {
  const cache = new Map<string, ScienceResourceHead>();
  const heads = request.ids.map((id) => {
    const cached = cache.get(id);
    if (cached) return cached;
    const head = scienceResourceHead(resolver, { id });
    cache.set(id, head);
    return head;
  });
  return scienceResourceBatchHeadSchema.parse({ heads });
}

export function scienceResourceMetadata(
  resolver: ScienceResourceResolver,
  request: ScienceResourceGetRequest,
): ScienceResourceGetResult {
  const resource = resolver.resolve(request.id);
  const entity = resource.entity;
  let metadata: z.input<typeof scienceResourceMetadataSchema>;
  if (resource.kind === "project") {
    const project = entity as ScienceProject;
    metadata = {
      kind: "project",
      counts: {
        notebooks: projectEntityCount(resolver, project.id, "notebook"),
        artifacts: projectEntityCount(resolver, project.id, "artifact"),
        documents: projectEntityCount(resolver, project.id, "document"),
        figures: projectEntityCount(resolver, project.id, "figure"),
        records: projectEntityCount(resolver, project.id, "record"),
        experiments: projectEntityCount(resolver, project.id, "experiment"),
        runs: projectEntityCount(resolver, project.id, "run"),
      },
      createdAt: project.createdAt,
      updatedAt: project.updatedAt,
    };
  } else if (resource.kind === "notebook") {
    const notebook = entity as ScienceNotebook;
    metadata = {
      kind: "notebook",
      cellCount: notebook.cells.length,
      inputArtifactCount: notebook.cells.reduce(
        (count, cell) => count + (cell.inputArtifactIds?.length ?? 0),
        0,
      ),
      outputArtifactCount: notebook.cells.reduce(
        (count, cell) => count + cell.outputArtifactIds.length,
        0,
      ),
      createdAt: notebook.createdAt,
      updatedAt: notebook.updatedAt,
    };
  } else if (resource.kind === "artifact") {
    const artifact = entity as ScienceArtifact;
    metadata = {
      kind: "artifact",
      artifactKind: artifact.kind,
      mime: artifact.mime,
      size: artifact.size,
      license: artifact.license,
      runRef: optionalRef(resolver, "run", artifact.runId),
      sourceRefs: boundedRefs(resolver, artifact.sourceEntityIds),
      createdAt: artifact.createdAt,
      updatedAt: artifact.updatedAt,
    };
  } else if (resource.kind === "document") {
    const document = entity as ScienceDocument;
    const sourceHash = document.revisions.at(-1)?.sourceHash;
    if (!sourceHash)
      throw new ScienceError("Document has no current source digest", "RESOURCE_INDEX_CONFLICT");
    metadata = {
      kind: "document",
      format: document.format,
      contentRevision: document.contentRevision,
      sourceHash,
      diagnosticCount: document.diagnostics.length,
      proposalCount: document.proposals.length,
      createdAt: document.createdAt,
      updatedAt: document.updatedAt,
    };
  } else if (resource.kind === "figure") {
    const figure = entity as ScienceFigure;
    const codeHash = figure.revisions.at(-1)?.codeHash;
    if (!codeHash)
      throw new ScienceError("Figure has no current code digest", "RESOURCE_INDEX_CONFLICT");
    metadata = {
      kind: "figure",
      library: figure.library,
      codeRevision: figure.codeRevision,
      codeHash,
      artifactRef: optionalRef(resolver, "artifact", figure.artifactId),
      objectCount: figure.objects.length,
      proposalCount: figure.proposals.length,
      createdAt: figure.createdAt,
      updatedAt: figure.updatedAt,
    };
  } else if (resource.kind === "record") {
    const record = entity as ScienceResearchRecord;
    metadata = {
      kind: "record",
      recordKind: record.kind,
      status: record.status,
      summary: truncate(record.summary),
      tags: boundedStrings(record.tags),
      sourceRefs: boundedRefs(resolver, record.sourceEntityIds),
      createdAt: record.createdAt,
      updatedAt: record.updatedAt,
    };
  } else if (resource.kind === "experiment") {
    const experiment = entity as ScienceExperiment;
    metadata = {
      kind: "experiment",
      status: experiment.status,
      summary: truncate(experiment.summary),
      tags: boundedStrings(experiment.tags),
      hypothesisRefs: boundedRefs(resolver, experiment.hypothesisIds, "record"),
      runRefs: boundedRefs(resolver, experiment.runIds, "run"),
      createdAt: experiment.createdAt,
      updatedAt: experiment.updatedAt,
    };
  } else {
    const run = entity as ScienceRun;
    metadata = {
      kind: "run",
      status: run.status,
      experimentRef: optionalRef(resolver, "experiment", run.experimentId),
      metricKeys: boundedStrings(Object.keys(run.metrics).sort()),
      artifactRefs: boundedRefs(resolver, run.artifactIds, "artifact"),
      startedAt: run.startedAt,
      finishedAt: run.finishedAt,
    };
  }
  return scienceResourceGetResultSchema.parse({
    ref: resource.ref,
    projection: "metadata",
    metadata,
  });
}

function verifiedPreview(resource: ResolvedScienceResource, preview: ScienceArtifactPreview): void {
  if (resource.kind !== "artifact") {
    throw new ScienceError("Resource selection requires an artifact ID", "RESOURCE_KIND_MISMATCH");
  }
  const artifact = resource.entity as ScienceArtifact;
  if (preview.artifactId !== artifact.id || preview.digest !== artifact.digest) {
    throw new ScienceError(
      "Artifact preview no longer matches the addressed resource",
      "ARTIFACT_SOURCE_CHANGED",
    );
  }
}

export function scienceResourceSelect(
  resolver: ScienceResourceResolver,
  request: ScienceResourceSelectRequest,
  preview: ScienceArtifactPreview,
): ScienceResourceSelectResult {
  const resource = resolver.resolve(request.id);
  verifiedPreview(resource, preview);
  if (preview.kind === "unavailable") {
    return scienceResourceSelectResultSchema.parse({
      ref: resource.ref,
      kind: "unavailable",
      reason: preview.reason,
    });
  }
  if (request.format === "text") {
    if (preview.kind !== "text") {
      return scienceResourceSelectResultSchema.parse({
        ref: resource.ref,
        kind: "unavailable",
        reason: "unsupported",
      });
    }
    const text = preview.text.slice(request.offset, request.offset + request.limit);
    const nextOffset =
      request.offset + text.length < preview.text.length ? request.offset + text.length : null;
    return scienceResourceSelectResultSchema.parse({
      ref: resource.ref,
      kind: "text",
      text,
      total: preview.text.length,
      offset: request.offset,
      returned: text.length,
      truncated: nextOffset !== null,
      nextOffset,
    });
  }
  if (preview.kind !== "table") {
    return scienceResourceSelectResultSchema.parse({
      ref: resource.ref,
      kind: "unavailable",
      reason: "unsupported",
    });
  }
  if (
    request.offset > preview.rows.length ||
    (request.offset === preview.rows.length && request.offset < preview.rowCount)
  ) {
    throw new ScienceError(
      "Requested rows are outside the bounded artifact preview",
      "RESOURCE_SELECTION_OUT_OF_RANGE",
    );
  }
  const indices = request.columns
    ? request.columns.map((name) => {
        const matches = preview.columns.flatMap((column, index) =>
          column.name === name ? [index] : [],
        );
        if (matches.length === 0)
          throw new ScienceError(
            `Artifact column '${name}' was not found`,
            "RESOURCE_COLUMN_NOT_FOUND",
          );
        if (matches.length > 1)
          throw new ScienceError(
            `Artifact column '${name}' is ambiguous`,
            "RESOURCE_COLUMN_AMBIGUOUS",
          );
        return matches[0] as number;
      })
    : preview.columns.slice(0, MAX_RESOURCE_SELECT_COLUMNS).map((_column, index) => index);
  let valueTruncated = false;
  const columns = indices.map((index) => {
    const column = preview.columns[index];
    if (!column)
      throw new ScienceError("Artifact column index is invalid", "RESOURCE_INDEX_CONFLICT");
    const name = truncate(column.name, 500);
    if (name !== column.name) valueTruncated = true;
    return { ...column, name };
  });
  const sourceRows = preview.rows.slice(request.offset, request.offset + request.limit);
  const rows = sourceRows.map((row) =>
    indices.map((index) => {
      const value = row[index] ?? null;
      if (typeof value !== "string" || value.length <= MAX_TABLE_CELL_CHARS) return value;
      valueTruncated = true;
      return truncate(value, MAX_TABLE_CELL_CHARS);
    }),
  );
  const end = request.offset + rows.length;
  const nextOffset = end < preview.rows.length ? end : null;
  return scienceResourceSelectResultSchema.parse({
    ref: resource.ref,
    kind: "table",
    columns,
    rows,
    total: preview.rowCount,
    offset: request.offset,
    returned: rows.length,
    truncated:
      preview.truncated ||
      valueTruncated ||
      nextOffset !== null ||
      indices.length < preview.columns.length,
    nextOffset,
  });
}

type Neighbor = z.infer<typeof scienceResourceNeighborSchema>;

export function scienceResourceNeighbors(
  resolver: ScienceResourceResolver,
  request: ScienceResourceNeighborsRequest,
): ScienceResourceNeighborsResult {
  const resource = resolver.resolve(request.id);
  const candidates: Neighbor[] = [];
  const add = (
    relation: z.infer<typeof scienceResourceRelationSchema>,
    direction: Neighbor["direction"],
    entityId: string,
  ): void => {
    const target = resolver.resolveUntypedEntityId(entityId);
    if (target) candidates.push({ relation, direction, target: target.ref });
  };
  const addTyped = (
    relation: z.infer<typeof scienceResourceRelationSchema>,
    direction: Neighbor["direction"],
    kind: ScienceResourceKind,
    entityId: string,
  ): void => {
    const target = typedRef(resolver, kind, entityId);
    if (target) candidates.push({ relation, direction, target });
  };
  const entity = resource.entity;
  if (resource.kind === "project") {
    const projectId = resource.entityId;
    for (const member of resolver.snapshot.notebooks)
      if (member.projectId === projectId) addTyped("contains", "outgoing", "notebook", member.id);
    for (const member of resolver.snapshot.artifacts)
      if (member.projectId === projectId) addTyped("contains", "outgoing", "artifact", member.id);
    for (const member of resolver.snapshot.documents)
      if (member.projectId === projectId) addTyped("contains", "outgoing", "document", member.id);
    for (const member of resolver.snapshot.figures)
      if (member.projectId === projectId) addTyped("contains", "outgoing", "figure", member.id);
    for (const member of resolver.snapshot.records)
      if (member.projectId === projectId) addTyped("contains", "outgoing", "record", member.id);
    for (const member of resolver.snapshot.experiments)
      if (member.projectId === projectId) addTyped("contains", "outgoing", "experiment", member.id);
    for (const member of resolver.snapshot.runs)
      if (member.projectId === projectId) addTyped("contains", "outgoing", "run", member.id);
  } else {
    addTyped(
      "contains",
      "incoming",
      "project",
      (entity as Exclude<typeof entity, ScienceProject>).projectId,
    );
  }
  if (resource.kind === "artifact") {
    const artifact = entity as ScienceArtifact;
    for (const sourceId of artifact.sourceEntityIds) add("derived_from", "outgoing", sourceId);
    if (artifact.runId) addTyped("produces", "incoming", "run", artifact.runId);
  } else if (resource.kind === "record") {
    for (const sourceId of (entity as ScienceResearchRecord).sourceEntityIds)
      add("derived_from", "outgoing", sourceId);
  } else if (resource.kind === "figure") {
    const artifactId = (entity as ScienceFigure).artifactId;
    if (artifactId) addTyped("uses", "outgoing", "artifact", artifactId);
  } else if (resource.kind === "experiment") {
    const experiment = entity as ScienceExperiment;
    for (const hypothesisId of experiment.hypothesisIds)
      addTyped("has_hypothesis", "outgoing", "record", hypothesisId);
    for (const runId of experiment.runIds) addTyped("has_run", "outgoing", "run", runId);
  } else if (resource.kind === "run") {
    const run = entity as ScienceRun;
    addTyped("has_run", "incoming", "experiment", run.experimentId);
    for (const artifactId of run.artifactIds)
      addTyped("produces", "outgoing", "artifact", artifactId);
  }
  for (const relation of resolver.snapshot.relations) {
    if (relation.fromId === resource.entityId) add(relation.type, "outgoing", relation.toId);
    if (relation.toId === resource.entityId) add(relation.type, "incoming", relation.fromId);
  }
  const unique = new Map(
    candidates.map((candidate) => [
      `${candidate.relation}\0${candidate.direction}\0${candidate.target.id}`,
      candidate,
    ]),
  );
  const allowed = request.relations ? new Set(request.relations) : null;
  const filtered = [...unique.values()].filter(
    (candidate) => allowed?.has(candidate.relation) ?? true,
  );
  const neighbors = filtered.slice(0, request.limit);
  return scienceResourceNeighborsResultSchema.parse({
    ref: resource.ref,
    neighbors,
    total: filtered.length,
    truncated: filtered.length > neighbors.length,
  });
}
