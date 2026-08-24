import { z } from "zod";

const entityIdSchema = z.string().min(1).max(200);
const titleSchema = z.string().trim().min(1).max(160);
const sha256DigestSchema = z.string().regex(/^sha256:[0-9a-f]{64}$/u);
const environmentSchema = z
  .record(z.string().min(1).max(100), z.string().max(1_000))
  .superRefine((environment, context) => {
    if (Object.keys(environment).length > 64) {
      context.addIssue({ code: "custom", message: "environment may contain at most 64 entries" });
    }
  });
const sourceEntityIdsSchema = z
  .array(entityIdSchema)
  .max(100)
  .refine((ids) => new Set(ids).size === ids.length, "source entity ids must be unique");
const documentNameSchema = z
  .string()
  .min(1)
  .max(240)
  .refine((name) => !name.startsWith("/") && !name.includes("\\") && !name.includes("\0"), {
    message: "document name must be a relative logical name",
  })
  .refine(
    (name) =>
      name.split("/").every((segment) => segment.length > 0 && segment !== "." && segment !== ".."),
    { message: "document name may not contain empty or traversal segments" },
  )
  .refine((name) => /\.(?:typ|typst|tex|md|bib)$/u.test(name), {
    message: "document name must end in .typ, .typst, .tex, .md, or .bib",
  });
const documentSourceSchema = z.string().max(500_000);
const figureCodeSchema = z.string().min(1).max(200_000);
const summarySchema = z.string().trim().min(1).max(4_000);
const tagsSchema = z
  .array(z.string().trim().min(1).max(80))
  .max(32)
  .refine((tags) => new Set(tags).size === tags.length, "tags must be unique");

export const RO_CRATE_CONTEXT = "https://w3id.org/ro/crate/1.3/context" as const;
export const RO_CRATE_PROFILE = "https://w3id.org/ro/crate/1.3" as const;
export const RO_CRATE_FILENAME = "ro-crate-metadata.json" as const;
export const RO_CRATE_MEDIA_TYPE = "application/ld+json" as const;
export const RO_CRATE_FORMAT = "ro-crate@1.3" as const;

export function roCrateEntityId(entityId: string): string {
  return `urn:uuid:${entityId}`;
}

const roCrateIdSchema = z.string().trim().min(1).max(4_096);
const roCrateReferenceSchema = z.strictObject({ "@id": roCrateIdSchema });
const roCrateReferencesSchema = z.array(roCrateReferenceSchema).max(5_000);
const roCrateTypeSchema = z.union([
  z.string().trim().min(1).max(1_000),
  z
    .array(z.string().trim().min(1).max(1_000))
    .min(1)
    .max(16)
    .refine((types) => new Set(types).size === types.length, "RO-Crate types must be unique"),
]);

export const roCrateEntitySchema = z
  .object({
    "@id": roCrateIdSchema,
    "@type": roCrateTypeSchema,
    about: roCrateReferenceSchema.optional(),
    actionStatus: roCrateReferenceSchema.optional(),
    additionalType: z.union([z.string().max(1_000), roCrateReferenceSchema]).optional(),
    bestRating: z.number().finite().optional(),
    citation: roCrateReferencesSchema.optional(),
    conformsTo: z.union([roCrateReferenceSchema, roCrateReferencesSchema]).optional(),
    contentSize: z.string().max(100).optional(),
    creativeWorkStatus: z.string().max(200).optional(),
    dateCreated: z.iso.datetime().optional(),
    dateModified: z.iso.datetime().optional(),
    datePublished: z.iso.datetime().optional(),
    description: z.string().max(20_000).optional(),
    encodingFormat: z.string().max(500).optional(),
    endTime: z.iso.datetime().optional(),
    hasPart: roCrateReferencesSchema.optional(),
    identifier: z.string().max(500).optional(),
    instrument: z.union([roCrateReferenceSchema, roCrateReferencesSchema]).optional(),
    isBasedOn: roCrateReferencesSchema.optional(),
    isPartOf: roCrateReferenceSchema.optional(),
    itemReviewed: roCrateReferenceSchema.optional(),
    keywords: z.array(z.string().max(500)).max(100).optional(),
    license: z.union([z.string().max(2_000), roCrateReferenceSchema]).optional(),
    name: z.string().max(1_000).optional(),
    object: z.union([roCrateReferenceSchema, roCrateReferencesSchema]).optional(),
    ratingValue: z.number().finite().optional(),
    result: z.union([roCrateReferenceSchema, roCrateReferencesSchema]).optional(),
    reviewRating: roCrateReferenceSchema.optional(),
    sha256: z
      .string()
      .regex(/^[0-9a-f]{64}$/u)
      .optional(),
    startTime: z.iso.datetime().optional(),
    text: z.string().max(100_000).optional(),
    version: z.union([z.string().max(100), z.number().finite()]).optional(),
    worstRating: z.number().finite().optional(),
  })
  .catchall(z.json());

function includesRoCrateType(entity: z.infer<typeof roCrateEntitySchema>, type: string): boolean {
  const types = entity["@type"];
  return Array.isArray(types) ? types.includes(type) : types === type;
}

function roCrateReferenceIds(
  value: z.infer<typeof roCrateReferenceSchema> | z.infer<typeof roCrateReferencesSchema>,
): readonly string[] {
  return (Array.isArray(value) ? value : [value]).map((reference) => reference["@id"]);
}

function validateRoCrateReferences(
  value: unknown,
  entityIds: ReadonlySet<string>,
  context: z.RefinementCtx,
  path: readonly (string | number)[],
): void {
  if (Array.isArray(value)) {
    value.forEach((item, index) => {
      validateRoCrateReferences(item, entityIds, context, [...path, index]);
    });
    return;
  }
  if (typeof value !== "object" || value === null) return;
  const object = value as Record<string, unknown>;
  if (typeof object["@id"] === "string") {
    if (Object.keys(object).length !== 1) {
      context.addIssue({
        code: "custom",
        message: "RO-Crate entities must be flattened into @graph",
        path: [...path],
      });
      return;
    }
    const id = object["@id"];
    if ((id.startsWith("#") || id.startsWith("urn:uuid:")) && !entityIds.has(id)) {
      context.addIssue({
        code: "custom",
        message: `RO-Crate local reference '${id}' has no entity`,
        path: [...path, "@id"],
      });
    }
    return;
  }
  for (const [key, child] of Object.entries(object)) {
    if (key !== "@id" && key !== "@type") {
      validateRoCrateReferences(child, entityIds, context, [...path, key]);
    }
  }
}

export const roCrateMetadataDocumentSchema = z
  .strictObject({
    "@context": z.literal(RO_CRATE_CONTEXT),
    "@graph": z.array(roCrateEntitySchema).min(2).max(5_000),
  })
  .superRefine((document, context) => {
    const entities = new Map<string, z.infer<typeof roCrateEntitySchema>>();
    for (const [index, entity] of document["@graph"].entries()) {
      if (entities.has(entity["@id"])) {
        context.addIssue({
          code: "custom",
          message: `RO-Crate entity id '${entity["@id"]}' is duplicated`,
          path: ["@graph", index, "@id"],
        });
      }
      entities.set(entity["@id"], entity);
    }
    const descriptor = entities.get(RO_CRATE_FILENAME);
    if (
      !descriptor ||
      !includesRoCrateType(descriptor, "CreativeWork") ||
      !descriptor.about ||
      !descriptor.conformsTo ||
      !roCrateReferenceIds(descriptor.conformsTo).includes(RO_CRATE_PROFILE)
    ) {
      context.addIssue({
        code: "custom",
        message: "RO-Crate Metadata Descriptor is missing or invalid",
        path: ["@graph"],
      });
      return;
    }
    const entityIds = new Set(entities.keys());
    for (const [index, entity] of document["@graph"].entries()) {
      for (const [key, value] of Object.entries(entity)) {
        if (key !== "@id" && key !== "@type") {
          validateRoCrateReferences(value, entityIds, context, ["@graph", index, key]);
        }
      }
    }
    const root = entities.get(descriptor.about["@id"]);
    if (
      !root ||
      !includesRoCrateType(root, "Dataset") ||
      !root.name ||
      !root.description ||
      !root.datePublished ||
      !root.license ||
      !root.hasPart
    ) {
      context.addIssue({
        code: "custom",
        message: "RO-Crate Root Data Entity is missing required metadata",
        path: ["@graph"],
      });
    }
  });

export const notebookMimeDataSchema = z.strictObject({
  mime: z.string().trim().min(1).max(200),
  data: z.string().max(1_400_000),
  encoding: z.enum(["utf8", "base64"]),
  truncated: z.boolean(),
});

export const notebookOutputBlockSchema = z.discriminatedUnion("type", [
  z.strictObject({
    type: z.literal("stream"),
    name: z.enum(["stdout", "stderr"]),
    text: z.string().max(1_000_000),
    truncated: z.boolean(),
  }),
  z.strictObject({
    type: z.enum(["display_data", "execute_result"]),
    data: z.array(notebookMimeDataSchema).min(1).max(32),
  }),
  z.strictObject({
    type: z.literal("error"),
    name: z.string().min(1).max(200),
    message: z.string().max(1_000_000),
    truncated: z.boolean(),
  }),
]);

export const provenanceReceiptSchema = z.strictObject({
  eventId: entityIdSchema,
  journalSeq: z.number().int().positive(),
  sessionId: entityIdSchema,
});

export const notebookCellSchema = z.strictObject({
  id: entityIdSchema,
  kind: z.enum(["markdown", "code", "output", "visualization", "ai-interaction"]),
  source: z.string(),
  executionCount: z.number().int().nonnegative().nullable(),
  executionTimeMs: z.number().nonnegative().nullable(),
  inputArtifactIds: z.array(entityIdSchema).max(4).optional(),
  outputArtifactIds: z.array(entityIdSchema),
  runtimeEnvironment: environmentSchema,
  relatedClaimIds: z.array(entityIdSchema),
  relatedExperimentIds: z.array(entityIdSchema),
  outputs: z.array(notebookOutputBlockSchema).max(256).default([]),
});

export const scienceProjectSchema = z.strictObject({
  id: entityIdSchema,
  kind: z.literal("project"),
  title: titleSchema,
  createdAt: z.number().int().nonnegative(),
  updatedAt: z.number().int().nonnegative(),
  revision: z.number().int().positive(),
  provenance: provenanceReceiptSchema,
});

export const scienceNotebookSchema = z.strictObject({
  id: entityIdSchema,
  projectId: entityIdSchema,
  kind: z.literal("notebook"),
  title: titleSchema,
  cells: z.array(notebookCellSchema),
  createdAt: z.number().int().nonnegative(),
  updatedAt: z.number().int().nonnegative(),
  revision: z.number().int().positive(),
  provenance: provenanceReceiptSchema,
});

export const artifactKindSchema = z.enum([
  "code",
  "notebook",
  "figure",
  "pdf",
  "dataset",
  "log",
  "model",
]);

export const MAX_SCIENCE_IMPORT_BYTES = 8 * 1024 * 1024;

const SCIENCE_IMPORT_TYPES = {
  ".csv": { kind: "dataset", mime: "text/csv" },
  ".tsv": { kind: "dataset", mime: "text/tab-separated-values" },
  ".json": { kind: "dataset", mime: "application/json" },
  ".txt": { kind: "log", mime: "text/plain" },
  ".md": { kind: "code", mime: "text/markdown" },
  ".xlsx": {
    kind: "dataset",
    mime: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  },
  ".pdf": { kind: "pdf", mime: "application/pdf" },
  ".png": { kind: "figure", mime: "image/png" },
  ".jpg": { kind: "figure", mime: "image/jpeg" },
  ".jpeg": { kind: "figure", mime: "image/jpeg" },
  ".gif": { kind: "figure", mime: "image/gif" },
  ".webp": { kind: "figure", mime: "image/webp" },
} as const;

export type ScienceImportType = (typeof SCIENCE_IMPORT_TYPES)[keyof typeof SCIENCE_IMPORT_TYPES];

/** Infer trusted artifact metadata from one path-free logical filename. */
export function scienceImportType(name: string): ScienceImportType | undefined {
  if (
    name.length === 0 ||
    name.length > 160 ||
    name !== name.trim() ||
    name.includes("/") ||
    name.includes("\\") ||
    name.includes("\0")
  ) {
    return undefined;
  }
  const extension = name.slice(name.lastIndexOf(".")).toLocaleLowerCase();
  return SCIENCE_IMPORT_TYPES[extension as keyof typeof SCIENCE_IMPORT_TYPES];
}

export const scienceArtifactSchema = z.strictObject({
  id: entityIdSchema,
  projectId: entityIdSchema,
  kind: artifactKindSchema,
  title: titleSchema,
  digest: sha256DigestSchema,
  mime: z.string().trim().min(1).max(200),
  size: z.number().int().nonnegative(),
  creator: z.strictObject({ kind: z.literal("session"), sessionId: entityIdSchema }),
  runId: entityIdSchema.nullable(),
  environment: environmentSchema,
  license: z.string().trim().min(1).max(200).nullable(),
  sourceEntityIds: sourceEntityIdsSchema,
  createdAt: z.number().int().nonnegative(),
  updatedAt: z.number().int().nonnegative(),
  revision: z.number().int().positive(),
  provenance: provenanceReceiptSchema,
});

export const scienceDocumentFormatSchema = z.enum(["typst", "latex", "markdown", "bibtex"]);

export const writingDiagnosticSchema = z.strictObject({
  code: z.enum([
    "claim-needs-evidence",
    "figure-reference-missing",
    "unbalanced-delimiter",
    "unbalanced-environment",
  ]),
  scope: z.enum(["structural", "scientific"]),
  severity: z.enum(["error", "warning"]),
  message: z.string().min(1).max(500),
  start: z.number().int().nonnegative(),
  end: z.number().int().nonnegative(),
});

export const documentRevisionSchema = z.strictObject({
  revision: z.number().int().positive(),
  contentRevision: z.number().int().positive(),
  sourceHash: sha256DigestSchema,
  previousSourceHash: sha256DigestSchema.nullable(),
  reason: z.enum(["created", "proposal-created", "proposal-accepted", "proposal-rejected"]),
  proposalId: entityIdSchema.nullable(),
  provenance: provenanceReceiptSchema,
});

export const documentPatchProposalSchema = z.strictObject({
  id: entityIdSchema,
  selection: z.strictObject({
    start: z.number().int().nonnegative(),
    end: z.number().int().positive(),
  }),
  originalText: documentSourceSchema,
  proposedText: documentSourceSchema,
  instruction: z.string().trim().min(1).max(2_000),
  reasoning: z.strictObject({
    classification: z.literal("proposal"),
    summary: z.string().trim().min(1).max(4_000),
  }),
  status: z.enum(["pending", "accepted", "rejected"]),
  createdAt: z.number().int().nonnegative(),
  resolvedAt: z.number().int().nonnegative().nullable(),
  createdProvenance: provenanceReceiptSchema,
  resolvedProvenance: provenanceReceiptSchema.nullable(),
});

export const scienceDocumentSchema = z.strictObject({
  id: entityIdSchema,
  projectId: entityIdSchema,
  kind: z.literal("document"),
  name: documentNameSchema,
  format: scienceDocumentFormatSchema,
  content: documentSourceSchema,
  revision: z.number().int().positive(),
  contentRevision: z.number().int().positive(),
  proposals: z.array(documentPatchProposalSchema).max(1_000),
  revisions: z.array(documentRevisionSchema).min(1).max(2_001),
  diagnostics: z.array(writingDiagnosticSchema).max(200),
  validation: z.strictObject({
    structural: z.literal("checked"),
    compilation: z.literal("not-run"),
  }),
  createdAt: z.number().int().nonnegative(),
  updatedAt: z.number().int().nonnegative(),
  provenance: provenanceReceiptSchema,
});

export const figureLibrarySchema = z.enum(["matplotlib", "seaborn", "ggplot2", "plotly"]);

function containsControlCharacter(value: string): boolean {
  return [...value].some((character) => (character.codePointAt(0) ?? 0) < 0x20);
}

const workspaceSourcePathSchema = z
  .string()
  .min(1)
  .max(4_096)
  .refine(
    (path) =>
      !path.startsWith("/") &&
      !/^[a-z]:[\\/]/iu.test(path) &&
      !path.includes("\\") &&
      !path.includes("\0") &&
      path.split("/").every((segment) => segment.length > 0 && segment !== "." && segment !== ".."),
    "workspace source must be a traversal-free relative path",
  );
const s3SourceUriSchema = z
  .string()
  .min(8)
  .max(4_096)
  .regex(
    /^s3:\/\/[a-z0-9](?:[a-z0-9.-]{1,61}[a-z0-9])\/[^\s?#]+$/u,
    "S3 source must be a credential-free s3://bucket/key URI",
  )
  .refine((uri) => !containsControlCharacter(uri), "S3 source may not contain control characters");
const sourceVersionIdSchema = z
  .string()
  .min(1)
  .max(1_024)
  .refine(
    (versionId) => !containsControlCharacter(versionId),
    "source version id may not contain control characters",
  );

export const figureSourceReferenceInputSchema = z.discriminatedUnion("kind", [
  z.strictObject({
    kind: z.literal("workspace"),
    relativePath: workspaceSourcePathSchema,
    digest: sha256DigestSchema.optional(),
  }),
  z.strictObject({
    kind: z.literal("artifact"),
    artifactId: entityIdSchema,
  }),
  z.strictObject({
    kind: z.literal("s3"),
    uri: s3SourceUriSchema,
    versionId: sourceVersionIdSchema.optional(),
    digest: sha256DigestSchema.optional(),
  }),
]);

const figureSourceReferencesInputSchema = z
  .array(figureSourceReferenceInputSchema)
  .max(32)
  .refine(
    (sources) => new Set(sources.map((source) => JSON.stringify(source))).size === sources.length,
    "figure source references must be unique",
  );

const artifactFigureCodeSchema = figureCodeSchema.refine(
  (code) => !/["'`](?:\/(?!\/)|[a-z]:[\\/]|\\\\)/iu.test(code),
  "Artifact metadata code may not contain an absolute filesystem path literal",
);

export const registerReproducibilityMetadataInputSchema = z.strictObject({
  library: figureLibrarySchema,
  code: artifactFigureCodeSchema,
  sources: figureSourceReferencesInputSchema,
});

export const notebookReproducibilityMetadataInputSchema = z.strictObject({
  library: figureLibrarySchema,
  sources: figureSourceReferencesInputSchema,
});

const normalizedFigureSourceReferenceSchema = z.discriminatedUnion("kind", [
  z.strictObject({
    kind: z.literal("workspace"),
    relativePath: workspaceSourcePathSchema,
    digest: sha256DigestSchema,
  }),
  z.strictObject({
    kind: z.literal("artifact"),
    artifactId: entityIdSchema,
    digest: sha256DigestSchema,
  }),
  z.strictObject({
    kind: z.literal("s3"),
    uri: s3SourceUriSchema,
    versionId: sourceVersionIdSchema.optional(),
    digest: sha256DigestSchema.optional(),
  }),
]);

export const figureReproducibilityMetadataSchema = z.strictObject({
  schema: z.literal("dsh-science.figure-provenance"),
  version: z.literal(1),
  generationId: z.string().uuid(),
  generator: z.strictObject({
    library: figureLibrarySchema,
    code: artifactFigureCodeSchema,
    codeHash: sha256DigestSchema,
  }),
  sources: z.array(normalizedFigureSourceReferenceSchema).max(32),
  environment: environmentSchema,
});

export const figureObjectKindSchema = z.enum([
  "axis",
  "legend",
  "annotation",
  "line",
  "point",
  "image-layer",
  "data-series",
]);

export const figureObjectSchema = z.strictObject({
  id: entityIdSchema,
  kind: figureObjectKindSchema,
  label: titleSchema,
  codeRange: z.strictObject({
    start: z.number().int().nonnegative(),
    end: z.number().int().positive(),
  }),
});

export const figureRevisionSchema = z.strictObject({
  revision: z.number().int().positive(),
  codeRevision: z.number().int().positive(),
  codeHash: sha256DigestSchema,
  previousCodeHash: sha256DigestSchema.nullable(),
  reason: z.enum(["created", "proposal-created", "proposal-accepted", "proposal-rejected"]),
  proposalId: entityIdSchema.nullable(),
  provenance: provenanceReceiptSchema,
});

export const figureCodeProposalSchema = z.strictObject({
  id: entityIdSchema,
  objectIds: z.array(entityIdSchema).min(1).max(50),
  selection: z.strictObject({
    start: z.number().int().nonnegative(),
    end: z.number().int().positive(),
  }),
  originalCode: figureCodeSchema,
  proposedCode: figureCodeSchema,
  instruction: z.string().trim().min(1).max(2_000),
  reasoning: z.strictObject({
    classification: z.literal("proposal"),
    summary: z.string().trim().min(1).max(4_000),
  }),
  status: z.enum(["pending", "accepted", "rejected"]),
  createdAt: z.number().int().nonnegative(),
  resolvedAt: z.number().int().nonnegative().nullable(),
  createdProvenance: provenanceReceiptSchema,
  resolvedProvenance: provenanceReceiptSchema.nullable(),
});

export const scienceFigureSchema = z.strictObject({
  id: entityIdSchema,
  projectId: entityIdSchema,
  kind: z.literal("figure"),
  title: titleSchema,
  library: figureLibrarySchema,
  code: figureCodeSchema,
  artifactId: entityIdSchema.nullable(),
  objects: z.array(figureObjectSchema).min(1).max(200),
  revision: z.number().int().positive(),
  codeRevision: z.number().int().positive(),
  proposals: z.array(figureCodeProposalSchema).max(200),
  revisions: z.array(figureRevisionSchema).min(1).max(401),
  createdAt: z.number().int().nonnegative(),
  updatedAt: z.number().int().nonnegative(),
  provenance: provenanceReceiptSchema,
});

export const researchRecordKindSchema = z.enum([
  "question",
  "hypothesis",
  "claim",
  "evidence",
  "decision",
  "review",
  "open-question",
]);

export const researchRecordStatusSchema = z.enum([
  "open",
  "proposed",
  "supported",
  "refuted",
  "accepted",
  "rejected",
]);

export const scienceResearchRecordSchema = z.strictObject({
  id: entityIdSchema,
  projectId: entityIdSchema,
  kind: researchRecordKindSchema,
  title: titleSchema,
  summary: summarySchema,
  status: researchRecordStatusSchema,
  tags: tagsSchema,
  sourceEntityIds: sourceEntityIdsSchema,
  createdAt: z.number().int().nonnegative(),
  updatedAt: z.number().int().nonnegative(),
  revision: z.number().int().positive(),
  provenance: provenanceReceiptSchema,
});

export const scienceRelationTypeSchema = z.enum([
  "supports",
  "refutes",
  "derived_from",
  "uses",
  "produces",
  "reproduces",
  "supersedes",
  "conflicts_with",
  "branch_of",
  "motivated_by",
]);

export const scienceRelationSchema = z.strictObject({
  id: entityIdSchema,
  projectId: entityIdSchema,
  fromId: entityIdSchema,
  toId: entityIdSchema,
  type: scienceRelationTypeSchema,
  createdAt: z.number().int().nonnegative(),
  provenance: provenanceReceiptSchema,
});

export const linkEvidenceResultSchema = z.strictObject({
  evidence: scienceResearchRecordSchema,
  relation: scienceRelationSchema,
  provenance: provenanceReceiptSchema,
});

export const scienceExperimentSchema = z.strictObject({
  id: entityIdSchema,
  projectId: entityIdSchema,
  kind: z.literal("experiment"),
  title: titleSchema,
  summary: summarySchema,
  protocol: z.string().trim().min(1).max(100_000),
  hypothesisIds: sourceEntityIdsSchema,
  runIds: sourceEntityIdsSchema,
  status: z.enum(["defined", "active", "completed"]),
  tags: tagsSchema,
  createdAt: z.number().int().nonnegative(),
  updatedAt: z.number().int().nonnegative(),
  revision: z.number().int().positive(),
  provenance: provenanceReceiptSchema,
});

const metricsSchema = z
  .record(z.string().trim().min(1).max(100), z.number().finite())
  .superRefine((metrics, context) => {
    if (Object.keys(metrics).length > 100) {
      context.addIssue({ code: "custom", message: "metrics may contain at most 100 entries" });
    }
  });

export const scienceRunSchema = z.strictObject({
  id: entityIdSchema,
  projectId: entityIdSchema,
  experimentId: entityIdSchema,
  kind: z.literal("run"),
  status: z.enum(["running", "succeeded", "failed", "cancelled"]),
  environment: environmentSchema,
  metrics: metricsSchema,
  artifactIds: sourceEntityIdsSchema,
  notes: z.string().max(20_000),
  startedAt: z.number().int().nonnegative(),
  finishedAt: z.number().int().nonnegative().nullable(),
  revision: z.number().int().positive(),
  provenance: provenanceReceiptSchema,
});

export const runMutationSchema = z.strictObject({
  experiment: scienceExperimentSchema,
  run: scienceRunSchema,
  provenance: provenanceReceiptSchema,
});

export const runComparisonSchema = z.strictObject({
  experimentId: entityIdSchema,
  baselineRunId: entityIdSchema,
  runIds: z.array(entityIdSchema).min(2).max(10),
  classification: z.literal("inference"),
  deltas: z.array(
    z.strictObject({
      metric: z.string().trim().min(1).max(100),
      values: z.array(z.number().finite()).min(2).max(10),
    }),
  ),
});

export const projectExportCountsSchema = z.strictObject({
  projects: z.literal(1),
  notebooks: z.number().int().nonnegative(),
  artifacts: z.number().int().nonnegative(),
  documents: z.number().int().nonnegative(),
  figures: z.number().int().nonnegative(),
  records: z.number().int().nonnegative(),
  relations: z.number().int().nonnegative(),
  experiments: z.number().int().nonnegative(),
  runs: z.number().int().nonnegative(),
});

const legacyProjectExportRecordSchema = z.strictObject({
  id: entityIdSchema,
  projectId: entityIdSchema,
  kind: z.literal("export"),
  format: z.literal("dsh-science-project@1"),
  digest: sha256DigestSchema,
  bytes: z.number().int().nonnegative(),
  counts: projectExportCountsSchema,
  createdAt: z.number().int().nonnegative(),
  revision: z.literal(1),
  provenance: provenanceReceiptSchema,
});

const roCrateProjectExportRecordSchema = z.strictObject({
  id: entityIdSchema,
  projectId: entityIdSchema,
  kind: z.literal("export"),
  format: z.literal(RO_CRATE_FORMAT),
  filename: z.literal(RO_CRATE_FILENAME),
  mediaType: z.literal(RO_CRATE_MEDIA_TYPE),
  digest: sha256DigestSchema,
  bytes: z.number().int().nonnegative(),
  counts: projectExportCountsSchema,
  createdAt: z.number().int().nonnegative(),
  revision: z.literal(1),
  provenance: provenanceReceiptSchema,
});

export const projectExportRecordSchema = z.discriminatedUnion("format", [
  legacyProjectExportRecordSchema,
  roCrateProjectExportRecordSchema,
]);

const legacyScienceProjectExportSchema = legacyProjectExportRecordSchema.extend({
  classification: z.literal("fact"),
  content: z.string().max(10_000_000),
});

const roCrateScienceProjectExportSchema = roCrateProjectExportRecordSchema.extend({
  classification: z.literal("fact"),
  content: z.string().max(10_000_000),
});

export const scienceProjectExportSchema = z.discriminatedUnion("format", [
  legacyScienceProjectExportSchema,
  roCrateScienceProjectExportSchema,
]);

export const createProjectRequestSchema = z.strictObject({
  requestId: z.string().uuid(),
  title: titleSchema,
});

export const createNotebookRequestSchema = z.strictObject({
  requestId: z.string().uuid(),
  projectId: entityIdSchema,
  title: titleSchema,
});

export const createDocumentRequestSchema = z.strictObject({
  requestId: z.string().uuid(),
  projectId: entityIdSchema,
  name: documentNameSchema,
  content: documentSourceSchema,
});

export const createFigureRequestSchema = z.strictObject({
  requestId: z.string().uuid(),
  projectId: entityIdSchema,
  title: titleSchema,
  library: figureLibrarySchema,
  code: figureCodeSchema,
  artifactId: entityIdSchema.nullable(),
});

const researchRecordRequestBase = {
  requestId: z.string().uuid(),
  projectId: entityIdSchema,
  title: titleSchema,
  summary: summarySchema,
  tags: tagsSchema,
};

export const createQuestionRequestSchema = z.strictObject(researchRecordRequestBase);

export const createHypothesisRequestSchema = z.strictObject({
  ...researchRecordRequestBase,
  questionId: entityIdSchema,
});

export const recordClaimRequestSchema = z.strictObject({
  ...researchRecordRequestBase,
  hypothesisId: entityIdSchema.nullable(),
  status: z.enum(["proposed", "supported", "refuted", "accepted", "rejected"]),
});

export const linkEvidenceRequestSchema = z.strictObject({
  ...researchRecordRequestBase,
  claimId: entityIdSchema,
  relation: z.enum(["supports", "refutes"]),
  sourceEntityIds: sourceEntityIdsSchema,
});

export const defineExperimentRequestSchema = z.strictObject({
  ...researchRecordRequestBase,
  hypothesisIds: sourceEntityIdsSchema,
  protocol: z.string().trim().min(1).max(100_000),
});

export const startRunRequestSchema = z.strictObject({
  requestId: z.string().uuid(),
  experimentId: entityIdSchema,
  expectedRevision: z.number().int().positive(),
  environment: environmentSchema,
});

export const finishRunRequestSchema = z.strictObject({
  requestId: z.string().uuid(),
  runId: entityIdSchema,
  expectedRevision: z.number().int().positive(),
  status: z.enum(["succeeded", "failed", "cancelled"]),
  metrics: metricsSchema,
  artifactIds: sourceEntityIdsSchema,
  notes: z.string().max(20_000),
});

export const compareRunsRequestSchema = z.strictObject({
  runIds: z
    .array(entityIdSchema)
    .min(2)
    .max(10)
    .refine((ids) => new Set(ids).size === ids.length, "run ids must be unique"),
});

export const exportProjectRequestSchema = z.strictObject({
  requestId: z.string().uuid(),
  projectId: entityIdSchema,
});

export const getResearchObjectRequestSchema = z.strictObject({
  projectId: entityIdSchema,
});

const proposeDocumentPatchRequestSchema = z
  .strictObject({
    requestId: z.string().uuid(),
    documentId: entityIdSchema,
    expectedRevision: z.number().int().positive(),
    action: z.literal("propose"),
    selection: z.strictObject({
      start: z.number().int().nonnegative(),
      end: z.number().int().positive(),
    }),
    proposedText: documentSourceSchema,
    instruction: z.string().trim().min(1).max(2_000),
    reasoning: z.string().trim().min(1).max(4_000),
  })
  .refine((request) => request.selection.start < request.selection.end, {
    message: "document selection must be non-empty",
    path: ["selection"],
  });

const resolveDocumentPatchRequestSchema = z.strictObject({
  requestId: z.string().uuid(),
  documentId: entityIdSchema,
  expectedRevision: z.number().int().positive(),
  action: z.enum(["accept", "reject"]),
  proposalId: entityIdSchema,
});

export const modifyDocumentRequestSchema = z.discriminatedUnion("action", [
  proposeDocumentPatchRequestSchema,
  resolveDocumentPatchRequestSchema,
]);

const proposeFigureCodeRequestSchema = z.strictObject({
  requestId: z.string().uuid(),
  figureId: entityIdSchema,
  expectedRevision: z.number().int().positive(),
  action: z.literal("propose"),
  objectIds: z
    .array(entityIdSchema)
    .min(1)
    .max(50)
    .refine((ids) => new Set(ids).size === ids.length, "figure object ids must be unique"),
  proposedCode: figureCodeSchema,
  instruction: z.string().trim().min(1).max(2_000),
  reasoning: z.string().trim().min(1).max(4_000),
});

const resolveFigureCodeRequestSchema = z.strictObject({
  requestId: z.string().uuid(),
  figureId: entityIdSchema,
  expectedRevision: z.number().int().positive(),
  action: z.enum(["accept", "reject"]),
  proposalId: entityIdSchema,
});

export const modifyFigureCodeRequestSchema = z.discriminatedUnion("action", [
  proposeFigureCodeRequestSchema,
  resolveFigureCodeRequestSchema,
]);

export const registerArtifactRequestSchema = z
  .strictObject({
    requestId: z.string().uuid(),
    projectId: entityIdSchema,
    relativePath: z.string().min(1).max(4_096),
    kind: artifactKindSchema,
    title: titleSchema,
    mime: z.string().trim().min(1).max(200),
    runId: entityIdSchema.nullable(),
    environment: environmentSchema,
    license: z.string().trim().min(1).max(200).nullable(),
    sourceEntityIds: sourceEntityIdsSchema,
    reproducibilityMetadata: z
      .union([z.literal(false), registerReproducibilityMetadataInputSchema])
      .optional(),
  })
  .superRefine((request, context) => {
    if (
      request.reproducibilityMetadata !== undefined &&
      request.reproducibilityMetadata !== false &&
      (request.kind !== "figure" ||
        !["image/png", "image/svg+xml", "application/pdf"].includes(request.mime))
    ) {
      context.addIssue({
        code: "custom",
        message: "Reproducibility metadata requires a PNG, SVG, or PDF figure artifact",
        path: ["reproducibilityMetadata"],
      });
    }
  });

const canonicalBase64Shape = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/u;

export const importArtifactRequestSchema = z.strictObject({
  requestId: z.string().uuid(),
  projectId: entityIdSchema,
  name: z
    .string()
    .min(1)
    .max(160)
    .refine((name) => scienceImportType(name) !== undefined, {
      message: "imported artifact name must be one supported basename",
    }),
  dataBase64: z
    .string()
    .min(4)
    .max(Math.ceil(MAX_SCIENCE_IMPORT_BYTES / 3) * 4)
    .regex(canonicalBase64Shape, "imported artifact bytes must use canonical base64 shape"),
});

export const previewArtifactRequestSchema = z.strictObject({
  artifactId: entityIdSchema,
});

export const MAX_TYPST_SOURCE_BYTES = 1024 * 1024;
export const MAX_TYPST_PDF_BYTES = 32 * 1024 * 1024;

export const typstRelativePathSchema = z
  .string()
  .min(1)
  .max(4_096)
  .refine(
    (path) =>
      !path.startsWith("/") &&
      !/^[a-z]:[\\/]/iu.test(path) &&
      !path.includes("\\") &&
      !path.includes("\0") &&
      path.split("/").every((segment) => segment.length > 0 && segment !== "." && segment !== ".."),
    "Typst paper path must be traversal-free and workspace-relative",
  )
  .refine((path) => /\.(?:typ|typst)$/iu.test(path), "Typst paper path must end in .typ or .typst");

export const previewTypstDocumentRequestSchema = z.strictObject({
  relativePath: typstRelativePathSchema,
});

export const resolveTypstSourceAtPointRequestSchema = z.strictObject({
  relativePath: typstRelativePathSchema,
  pdfRevision: sha256DigestSchema,
  page: z.number().int().positive().max(100_000),
  x: z.number().finite().min(0).max(1),
  y: z.number().finite().min(0).max(1),
});

export const updateTypstSourceRequestSchema = z.strictObject({
  relativePath: typstRelativePathSchema,
  expectedSourceRevision: sha256DigestSchema,
  source: z.string().max(MAX_TYPST_SOURCE_BYTES),
});

export const typstSourceUpdateSchema = z.strictObject({
  relativePath: typstRelativePathSchema,
  title: titleSchema,
  source: z.string().max(MAX_TYPST_SOURCE_BYTES),
  sourceRevision: sha256DigestSchema,
});

export const typstSourceTargetSchema = z.strictObject({
  relativePath: typstRelativePathSchema,
  title: titleSchema,
  source: z.string().max(MAX_TYPST_SOURCE_BYTES),
  sourceRevision: sha256DigestSchema,
  offset: z.number().int().nonnegative().max(MAX_TYPST_SOURCE_BYTES),
});

export const typstDocumentPreviewSchema = z.strictObject({
  relativePath: typstRelativePathSchema,
  title: titleSchema,
  source: z.string().max(MAX_TYPST_SOURCE_BYTES),
  sourceRevision: sha256DigestSchema,
  status: z.enum(["compiling", "ready", "stale", "error", "unavailable"]),
  diagnostics: z.array(z.string().max(4_096)).max(100),
  pdfBase64: z
    .string()
    .max(Math.ceil(MAX_TYPST_PDF_BYTES / 3) * 4)
    .regex(canonicalBase64Shape, "Typst PDF bytes must use canonical base64 shape")
    .nullable(),
  pdfRevision: sha256DigestSchema.nullable(),
  pdfSourceRevision: sha256DigestSchema.nullable(),
  pdfSize: z.number().int().nonnegative().max(MAX_TYPST_PDF_BYTES).nullable(),
  compiledAt: z.number().int().nonnegative().nullable(),
});

const normalizedPdfRectSchema = z
  .strictObject({
    x: z.number().finite().min(0).max(1),
    y: z.number().finite().min(0).max(1),
    width: z.number().finite().positive().max(1),
    height: z.number().finite().positive().max(1),
  })
  .refine((rect) => rect.x + rect.width <= 1 && rect.y + rect.height <= 1, {
    message: "Paper annotation rectangle must remain inside the PDF page",
  });

const paperAnnotationIdentitySchema = {
  version: z.literal(1),
  id: entityIdSchema,
  relativePath: typstRelativePathSchema,
  title: titleSchema,
  sourceRevision: sha256DigestSchema,
  pdfRevision: sha256DigestSchema,
  page: z.number().int().positive().max(100_000),
  rect: normalizedPdfRectSchema,
  comment: z.string().trim().min(1).max(2_000),
  createdAt: z.number().int().nonnegative(),
};

export const sciencePaperAnnotationSchema = z.discriminatedUnion("kind", [
  z.strictObject({
    ...paperAnnotationIdentitySchema,
    kind: z.literal("text"),
    selectedText: z.string().trim().min(1).max(8_000),
  }),
  z.strictObject({
    ...paperAnnotationIdentitySchema,
    kind: z.literal("figure-point"),
    figureIndex: z.number().int().nonnegative().max(10_000),
    x: z.number().finite().min(0).max(1),
    y: z.number().finite().min(0).max(1),
  }),
]);

export const scienceImageAnnotationSchema = z.strictObject({
  version: z.literal(1),
  id: entityIdSchema,
  artifactId: entityIdSchema,
  projectId: entityIdSchema,
  title: titleSchema,
  digest: sha256DigestSchema,
  mime: z.enum(["image/png", "image/jpeg", "image/gif", "image/webp"]),
  x: z.number().finite().min(0).max(1),
  y: z.number().finite().min(0).max(1),
  comment: z.string().trim().min(1).max(2_000),
  createdAt: z.number().int().nonnegative(),
});

const artifactPreviewIdentitySchema = {
  artifactId: entityIdSchema,
  digest: sha256DigestSchema,
  mime: z.string().trim().min(1).max(200),
  size: z.number().int().nonnegative(),
};

export const scienceArtifactPreviewSchema = z.discriminatedUnion("kind", [
  z.strictObject({
    kind: z.literal("table"),
    ...artifactPreviewIdentitySchema,
    columns: z
      .array(
        z.strictObject({
          id: z.string().regex(/^column-\d+$/u),
          name: z.string().max(4_096),
          type: z.enum(["string", "number", "boolean"]),
        }),
      )
      .max(256),
    rows: z
      .array(
        z.array(z.union([z.string().max(64 * 1024), z.number(), z.boolean(), z.null()])).max(256),
      )
      .max(500),
    rowCount: z.number().int().nonnegative(),
    truncated: z.boolean(),
  }),
  z.strictObject({
    kind: z.literal("text"),
    ...artifactPreviewIdentitySchema,
    text: z.string().max(64 * 1024),
  }),
  z.strictObject({
    kind: z.literal("image"),
    ...artifactPreviewIdentitySchema,
    dataUrl: z.string().max(2_800_000),
  }),
  z.strictObject({
    kind: z.literal("unavailable"),
    ...artifactPreviewIdentitySchema,
    reason: z.enum(["unsupported", "too-large"]),
  }),
]);

export const executeNotebookCellRequestSchema = z.strictObject({
  requestId: z.string().uuid(),
  notebookId: entityIdSchema,
  source: z.string().max(100_000),
  inputArtifactIds: z
    .array(entityIdSchema)
    .max(4)
    .refine((ids) => new Set(ids).size === ids.length, "input artifact ids must be unique")
    .optional(),
  outputArtifact: z
    .strictObject({
      relativePath: z.string().min(1).max(4_096),
      kind: artifactKindSchema,
      title: titleSchema,
      mime: z.string().trim().min(1).max(200),
      license: z.string().trim().min(1).max(200).nullable(),
      reproducibilityMetadata: z
        .union([z.literal(false), notebookReproducibilityMetadataInputSchema])
        .optional(),
    })
    .superRefine((artifact, context) => {
      if (
        artifact.reproducibilityMetadata !== undefined &&
        artifact.reproducibilityMetadata !== false &&
        (artifact.kind !== "figure" ||
          !["image/png", "image/svg+xml", "application/pdf"].includes(artifact.mime))
      ) {
        context.addIssue({
          code: "custom",
          message: "Reproducibility metadata requires a PNG, SVG, or PDF figure artifact",
          path: ["reproducibilityMetadata"],
        });
      }
    })
    .nullable(),
});

export const notebookExecutionOutputSchema = z.strictObject({
  text: z.string().max(1_000_000),
  truncated: z.boolean(),
});

export const notebookExecutionSchema = z.strictObject({
  id: entityIdSchema,
  notebookId: entityIdSchema,
  cellId: entityIdSchema,
  executionCount: z.number().int().positive(),
  status: z.enum(["succeeded", "failed"]),
  stdout: notebookExecutionOutputSchema,
  stderr: notebookExecutionOutputSchema,
  outputs: z.array(notebookOutputBlockSchema).max(256).default([]),
  exitCode: z.number().int().nullable(),
  signal: z.string().max(32).nullable(),
  durationMs: z.number().int().nonnegative(),
  environment: environmentSchema,
  inputArtifactIds: z.array(entityIdSchema).max(4).optional(),
  artifact: scienceArtifactSchema.nullable(),
  notebook: scienceNotebookSchema,
  provenance: provenanceReceiptSchema,
});

const literatureYearFilterSchema = z
  .strictObject({
    from: z.number().int().min(1000).max(3000).optional(),
    to: z.number().int().min(1000).max(3000).optional(),
  })
  .refine((value) => value.from === undefined || value.to === undefined || value.from <= value.to, {
    message: "literature year range is inverted",
  });

export const literatureSearchRequestSchema = z.strictObject({
  query: z.string().trim().min(1).max(500),
  limit: z.number().int().min(1).max(20).default(10),
  filters: z
    .strictObject({
      years: literatureYearFilterSchema.optional(),
      entryTypes: z
        .array(
          z
            .string()
            .trim()
            .toLowerCase()
            .regex(/^[a-z][a-z0-9_-]{0,39}$/u),
        )
        .min(1)
        .max(16)
        .optional(),
    })
    .optional(),
});

export const literatureMatchFieldSchema = z.enum([
  "title",
  "authors",
  "abstract",
  "keywords",
  "venue",
  "identifier",
]);

export const literatureWorkSchema = z.strictObject({
  citationKey: z.string().min(1).max(200),
  sourceItemKey: z.string().regex(/^[A-Z0-9]{8}$/u),
  entryType: z.string().regex(/^[a-z][a-z0-9_-]{0,39}$/u),
  title: z.string().min(1).max(1_000),
  authors: z.array(z.string().min(1).max(500)).max(100),
  year: z.number().int().min(1000).max(3000).nullable(),
  venue: z.string().max(1_000).nullable(),
  doi: z.string().max(500).nullable(),
  url: z.string().max(2_048).nullable(),
  abstract: z.string().max(4_000).nullable(),
  keywords: z.array(z.string().min(1).max(500)).max(100),
  score: z.number().int().nonnegative(),
  matchedFields: z.array(literatureMatchFieldSchema).max(6),
  bibtex: z
    .string()
    .min(1)
    .max(64 * 1024),
});

export const bibliographySnapshotSchema = z.strictObject({
  source: z.literal("zotero"),
  format: z.literal("bibtex"),
  digest: sha256DigestSchema,
  entryCount: z.number().int().nonnegative().max(500),
  sourceVersion: z.string().max(100).nullable(),
});

export const literatureSearchResultSchema = z.strictObject({
  source: z.literal("zotero"),
  ranking: z.literal("zotero-local-v1"),
  query: z.string().min(1).max(500),
  totalCandidates: z.number().int().nonnegative().max(500),
  snapshot: bibliographySnapshotSchema,
  results: z.array(literatureWorkSchema).max(20),
});

export const scienceWorkspaceSnapshotSchema = z.strictObject({
  projects: z.array(scienceProjectSchema),
  notebooks: z.array(scienceNotebookSchema),
  artifacts: z.array(scienceArtifactSchema),
  documents: z.array(scienceDocumentSchema),
  figures: z.array(scienceFigureSchema),
  records: z.array(scienceResearchRecordSchema),
  relations: z.array(scienceRelationSchema),
  experiments: z.array(scienceExperimentSchema),
  runs: z.array(scienceRunSchema),
  exports: z.array(projectExportRecordSchema),
});

export type ProvenanceReceipt = z.infer<typeof provenanceReceiptSchema>;
export type NotebookMimeData = z.infer<typeof notebookMimeDataSchema>;
export type NotebookOutputBlock = z.infer<typeof notebookOutputBlockSchema>;
export type NotebookCell = z.infer<typeof notebookCellSchema>;
export type ScienceProject = z.infer<typeof scienceProjectSchema>;
export type ScienceNotebook = z.infer<typeof scienceNotebookSchema>;
export type ArtifactKind = z.infer<typeof artifactKindSchema>;
export type ScienceArtifact = z.infer<typeof scienceArtifactSchema>;
export type ScienceDocumentFormat = z.infer<typeof scienceDocumentFormatSchema>;
export type WritingDiagnostic = z.infer<typeof writingDiagnosticSchema>;
export type DocumentRevision = z.infer<typeof documentRevisionSchema>;
export type DocumentPatchProposal = z.infer<typeof documentPatchProposalSchema>;
export type ScienceDocument = z.infer<typeof scienceDocumentSchema>;
export type FigureLibrary = z.infer<typeof figureLibrarySchema>;
export type FigureSourceReferenceInput = z.infer<typeof figureSourceReferenceInputSchema>;
export type RegisterReproducibilityMetadataInput = z.infer<
  typeof registerReproducibilityMetadataInputSchema
>;
export type NotebookReproducibilityMetadataInput = z.infer<
  typeof notebookReproducibilityMetadataInputSchema
>;
export type FigureReproducibilityMetadata = z.infer<typeof figureReproducibilityMetadataSchema>;
export type FigureObjectKind = z.infer<typeof figureObjectKindSchema>;
export type FigureObject = z.infer<typeof figureObjectSchema>;
export type FigureRevision = z.infer<typeof figureRevisionSchema>;
export type FigureCodeProposal = z.infer<typeof figureCodeProposalSchema>;
export type ScienceFigure = z.infer<typeof scienceFigureSchema>;
export type ResearchRecordKind = z.infer<typeof researchRecordKindSchema>;
export type ResearchRecordStatus = z.infer<typeof researchRecordStatusSchema>;
export type ScienceResearchRecord = z.infer<typeof scienceResearchRecordSchema>;
export type ScienceRelationType = z.infer<typeof scienceRelationTypeSchema>;
export type ScienceRelation = z.infer<typeof scienceRelationSchema>;
export type LinkEvidenceResult = z.infer<typeof linkEvidenceResultSchema>;
export type ScienceExperiment = z.infer<typeof scienceExperimentSchema>;
export type ScienceRun = z.infer<typeof scienceRunSchema>;
export type RunMutation = z.infer<typeof runMutationSchema>;
export type RunComparison = z.infer<typeof runComparisonSchema>;
export type ProjectExportCounts = z.infer<typeof projectExportCountsSchema>;
export type ProjectExportRecord = z.infer<typeof projectExportRecordSchema>;
export type ScienceProjectExport = z.infer<typeof scienceProjectExportSchema>;
export type RoCrateEntity = z.infer<typeof roCrateEntitySchema>;
export type RoCrateMetadataDocument = z.infer<typeof roCrateMetadataDocumentSchema>;
export type CreateProjectRequest = z.infer<typeof createProjectRequestSchema>;
export type CreateNotebookRequest = z.infer<typeof createNotebookRequestSchema>;
export type CreateDocumentRequest = z.infer<typeof createDocumentRequestSchema>;
export type ModifyDocumentRequest = z.infer<typeof modifyDocumentRequestSchema>;
export type CreateFigureRequest = z.infer<typeof createFigureRequestSchema>;
export type ModifyFigureCodeRequest = z.infer<typeof modifyFigureCodeRequestSchema>;
export type CreateQuestionRequest = z.infer<typeof createQuestionRequestSchema>;
export type CreateHypothesisRequest = z.infer<typeof createHypothesisRequestSchema>;
export type RecordClaimRequest = z.infer<typeof recordClaimRequestSchema>;
export type LinkEvidenceRequest = z.infer<typeof linkEvidenceRequestSchema>;
export type DefineExperimentRequest = z.infer<typeof defineExperimentRequestSchema>;
export type StartRunRequest = z.infer<typeof startRunRequestSchema>;
export type FinishRunRequest = z.infer<typeof finishRunRequestSchema>;
export type CompareRunsRequest = z.infer<typeof compareRunsRequestSchema>;
export type ExportProjectRequest = z.infer<typeof exportProjectRequestSchema>;
export type GetResearchObjectRequest = z.infer<typeof getResearchObjectRequestSchema>;
export type RegisterArtifactRequest = z.infer<typeof registerArtifactRequestSchema>;
export type ImportArtifactRequest = z.infer<typeof importArtifactRequestSchema>;
export type PreviewArtifactRequest = z.infer<typeof previewArtifactRequestSchema>;
export type ScienceArtifactPreview = z.infer<typeof scienceArtifactPreviewSchema>;
export type ScienceImageAnnotation = z.infer<typeof scienceImageAnnotationSchema>;
export type PreviewTypstDocumentRequest = z.infer<typeof previewTypstDocumentRequestSchema>;
export type ResolveTypstSourceAtPointRequest = z.infer<
  typeof resolveTypstSourceAtPointRequestSchema
>;
export type UpdateTypstSourceRequest = z.infer<typeof updateTypstSourceRequestSchema>;
export type TypstSourceUpdate = z.infer<typeof typstSourceUpdateSchema>;
export type TypstSourceTarget = z.infer<typeof typstSourceTargetSchema>;
export type TypstDocumentPreview = z.infer<typeof typstDocumentPreviewSchema>;
export type SciencePaperAnnotation = z.infer<typeof sciencePaperAnnotationSchema>;
export type ExecuteNotebookCellRequest = z.infer<typeof executeNotebookCellRequestSchema>;
export type NotebookExecutionOutput = z.infer<typeof notebookExecutionOutputSchema>;
export type NotebookExecution = z.infer<typeof notebookExecutionSchema>;
export type LiteratureSearchRequest = z.infer<typeof literatureSearchRequestSchema>;
export type LiteratureMatchField = z.infer<typeof literatureMatchFieldSchema>;
export type LiteratureWork = z.infer<typeof literatureWorkSchema>;
export type BibliographySnapshot = z.infer<typeof bibliographySnapshotSchema>;
export type LiteratureSearchResult = z.infer<typeof literatureSearchResultSchema>;
export type ScienceWorkspaceSnapshot = z.infer<typeof scienceWorkspaceSnapshotSchema>;
