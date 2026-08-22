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
  .refine((name) => /\.(?:typ|tex|md|bib)$/u.test(name), {
    message: "document name must end in .typ, .tex, .md, or .bib",
  });
const documentSourceSchema = z.string().max(500_000);
const figureCodeSchema = z.string().min(1).max(200_000);
const summarySchema = z.string().trim().min(1).max(4_000);
const tagsSchema = z
  .array(z.string().trim().min(1).max(80))
  .max(32)
  .refine((tags) => new Set(tags).size === tags.length, "tags must be unique");

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

export const projectExportRecordSchema = z.strictObject({
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

export const scienceProjectExportSchema = projectExportRecordSchema.extend({
  classification: z.literal("fact"),
  content: z.string().max(10_000_000),
});

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

export const registerArtifactRequestSchema = z.strictObject({
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

export const traceProvenanceRequestSchema = z.strictObject({
  entityId: entityIdSchema,
  maxDepth: z.number().int().nonnegative().max(20),
});

export const provenanceEntitySchema = z.strictObject({
  id: entityIdSchema,
  kind: z.enum([
    "project",
    "notebook",
    "artifact",
    "document",
    "figure",
    "question",
    "hypothesis",
    "claim",
    "evidence",
    "decision",
    "review",
    "open-question",
    "experiment",
    "run",
    "export",
  ]),
  title: titleSchema,
});

export const provenanceRelationSchema = z.strictObject({
  fromId: entityIdSchema,
  toId: entityIdSchema,
  type: scienceRelationTypeSchema,
});

export const provenanceEventSchema = z.strictObject({
  eventId: entityIdSchema,
  journalSeq: z.number().int().positive(),
  entityId: entityIdSchema,
  operation: z.enum([
    "project/created",
    "notebook/created",
    "notebook/cell-executed",
    "artifact/registered",
    "document/created",
    "document/modified",
    "figure/created",
    "figure/modified",
    "question/created",
    "hypothesis/created",
    "claim/recorded",
    "evidence/linked",
    "experiment/defined",
    "run/started",
    "run/finished",
    "project/exported",
  ]),
  occurredAt: z.number().int().nonnegative(),
  sessionId: entityIdSchema,
});

export const provenanceTraceSchema = z.strictObject({
  rootId: entityIdSchema,
  entities: z.array(provenanceEntitySchema).max(200),
  relations: z.array(provenanceRelationSchema).max(400),
  events: z.array(provenanceEventSchema).max(200),
  truncated: z.boolean(),
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
export type RegisterArtifactRequest = z.infer<typeof registerArtifactRequestSchema>;
export type ImportArtifactRequest = z.infer<typeof importArtifactRequestSchema>;
export type PreviewArtifactRequest = z.infer<typeof previewArtifactRequestSchema>;
export type ScienceArtifactPreview = z.infer<typeof scienceArtifactPreviewSchema>;
export type ExecuteNotebookCellRequest = z.infer<typeof executeNotebookCellRequestSchema>;
export type NotebookExecutionOutput = z.infer<typeof notebookExecutionOutputSchema>;
export type NotebookExecution = z.infer<typeof notebookExecutionSchema>;
export type TraceProvenanceRequest = z.infer<typeof traceProvenanceRequestSchema>;
export type ProvenanceEntity = z.infer<typeof provenanceEntitySchema>;
export type ProvenanceRelation = z.infer<typeof provenanceRelationSchema>;
export type ProvenanceEvent = z.infer<typeof provenanceEventSchema>;
export type ProvenanceTrace = z.infer<typeof provenanceTraceSchema>;
export type ScienceWorkspaceSnapshot = z.infer<typeof scienceWorkspaceSnapshotSchema>;
