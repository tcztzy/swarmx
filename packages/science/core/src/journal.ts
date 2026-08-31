import { createHash, randomUUID } from "node:crypto";
import { chmodSync, mkdirSync } from "node:fs";
import { isAbsolute, join } from "node:path";
import { DatabaseSync, type StatementSync } from "node:sqlite";
import type { CapturedArtifactObject } from "./artifact-store.js";
import {
  type CompareRunsRequest,
  type CreateDocumentRequest,
  type CreateFigureRequest,
  type CreateHypothesisRequest,
  type CreateNotebookRequest,
  type CreateProjectRequest,
  type CreateQuestionRequest,
  type DefineExperimentRequest,
  type ExecuteNotebookCellRequest,
  type ExportProjectRequest,
  type FinishRunRequest,
  type LinkEvidenceRequest,
  type LinkEvidenceResult,
  linkEvidenceResultSchema,
  type ModifyDocumentRequest,
  type ModifyFigureCodeRequest,
  type NotebookExecution,
  notebookExecutionSchema,
  type ProjectExportCounts,
  type ProjectExportRecord,
  projectExportRecordSchema,
  type RecordClaimRequest,
  type RegisterArtifactRequest,
  RO_CRATE_FILENAME,
  RO_CRATE_FORMAT,
  RO_CRATE_MEDIA_TYPE,
  type RunComparison,
  type RunMutation,
  runComparisonSchema,
  runMutationSchema,
  type ScienceArtifact,
  type ScienceDocument,
  type ScienceExperiment,
  type ScienceFigure,
  type ScienceNotebook,
  type ScienceProject,
  type ScienceRelation,
  type ScienceResearchRecord,
  type ScienceRun,
  type ScienceWorkspaceSnapshot,
  type StartRunRequest,
  scienceArtifactSchema,
  scienceDocumentSchema,
  scienceExperimentSchema,
  scienceFigureSchema,
  scienceNotebookSchema,
  scienceProjectSchema,
  scienceRelationSchema,
  scienceResearchRecordSchema,
  scienceRunSchema,
} from "./contracts.js";
import { ScienceError } from "./errors.js";
import { codeHash, inferFigureObjects, remapFigureObjects } from "./figure.js";
import { analyzeDocument, documentFormat, sourceHash } from "./writing.js";

const DATABASE_NAME = "science.sqlite";
const MIGRATION_VERSION = 5;
const MAX_DOCUMENT_SOURCE_LENGTH = 500_000;
const MAX_DOCUMENT_PROPOSALS = 1_000;
const MAX_FIGURE_PROPOSALS = 200;

interface JournalRow {
  occurred_at: number;
  payload_json: string;
  seq: number;
  type: string;
  workspace_key: string;
}

interface RequestRow {
  payload_json: string;
  request_hash: string;
  type: string;
}

interface ProjectRow {
  created_at: number;
  event_id: string;
  event_seq: number;
  id: string;
  revision: number;
  session_id: string;
  title: string;
  updated_at: number;
}

interface NotebookRow extends ProjectRow {
  cells_json: string;
  project_id: string;
}

interface ArtifactRow extends ProjectRow {
  creator_session_id: string;
  digest: string;
  environment_json: string;
  kind: string;
  license: string | null;
  mime: string;
  project_id: string;
  run_id: string | null;
  size: number;
  source_entity_ids_json: string;
}

interface DocumentRow {
  document_json: string;
}

interface FigureRow {
  figure_json: string;
}

interface RecordRow {
  record_json: string;
}

interface RelationRow {
  relation_json: string;
}

interface ExperimentRow {
  experiment_json: string;
}

interface RunRow {
  run_json: string;
}

interface ExportRow {
  export_json: string;
}

export interface SettledNotebookCell {
  readonly capturedArtifact: CapturedArtifactObject | undefined;
  readonly durationMs: number;
  readonly environment: Record<string, string>;
  readonly exitCode: number | null;
  readonly outputs: NotebookExecution["outputs"];
  readonly signal: string | null;
  readonly status: NotebookExecution["status"];
  readonly stderr: { readonly text: string; readonly truncated: boolean };
  readonly stdout: { readonly text: string; readonly truncated: boolean };
}

function notebookOutputText(outputs: NotebookExecution["outputs"]): string {
  return outputs
    .flatMap((output) => {
      if (output.type === "stream") return [output.text];
      if (output.type === "error") return [output.message];
      const preferred =
        output.data.find((item) => item.mime === "text/plain") ??
        output.data.find((item) => item.encoding === "utf8");
      if (preferred) return [preferred.data];
      const first = output.data[0];
      return first ? [`[${first.mime} output]`] : [];
    })
    .join("\n");
}

function requestHash(type: string, workspaceKey: string, request: object): string {
  return createHash("sha256").update(JSON.stringify({ request, type, workspaceKey })).digest("hex");
}

function ensureOpen(open: boolean): void {
  if (!open) {
    throw new ScienceError("The science journal is closed", "SCIENCE_CLOSED");
  }
}

function projectFromRow(row: ProjectRow): ScienceProject {
  return scienceProjectSchema.parse({
    id: row.id,
    kind: "project",
    title: row.title,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    revision: row.revision,
    provenance: {
      eventId: row.event_id,
      journalSeq: row.event_seq,
      sessionId: row.session_id,
    },
  });
}

function notebookFromRow(row: NotebookRow): ScienceNotebook {
  return scienceNotebookSchema.parse({
    id: row.id,
    projectId: row.project_id,
    kind: "notebook",
    title: row.title,
    cells: JSON.parse(row.cells_json),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    revision: row.revision,
    provenance: {
      eventId: row.event_id,
      journalSeq: row.event_seq,
      sessionId: row.session_id,
    },
  });
}

function artifactFromRow(row: ArtifactRow): ScienceArtifact {
  return scienceArtifactSchema.parse({
    id: row.id,
    projectId: row.project_id,
    kind: row.kind,
    title: row.title,
    digest: row.digest,
    mime: row.mime,
    size: row.size,
    creator: { kind: "session", sessionId: row.creator_session_id },
    runId: row.run_id,
    environment: JSON.parse(row.environment_json),
    license: row.license,
    sourceEntityIds: JSON.parse(row.source_entity_ids_json),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    revision: row.revision,
    provenance: {
      eventId: row.event_id,
      journalSeq: row.event_seq,
      sessionId: row.session_id,
    },
  });
}

function documentFromRow(row: DocumentRow): ScienceDocument {
  return scienceDocumentSchema.parse(JSON.parse(row.document_json));
}

function figureFromRow(row: FigureRow): ScienceFigure {
  return scienceFigureSchema.parse(JSON.parse(row.figure_json));
}

function recordFromRow(row: RecordRow): ScienceResearchRecord {
  return scienceResearchRecordSchema.parse(JSON.parse(row.record_json));
}

function relationFromRow(row: RelationRow): ScienceRelation {
  return scienceRelationSchema.parse(JSON.parse(row.relation_json));
}

function experimentFromRow(row: ExperimentRow): ScienceExperiment {
  return scienceExperimentSchema.parse(JSON.parse(row.experiment_json));
}

function runFromRow(row: RunRow): ScienceRun {
  return scienceRunSchema.parse(JSON.parse(row.run_json));
}

function exportFromRow(row: ExportRow): ProjectExportRecord {
  return projectExportRecordSchema.parse(JSON.parse(row.export_json));
}

/** Durable append-only science facts plus rebuildable workspace projections. */
export class ScienceJournal {
  readonly databasePath: string;

  private readonly database: DatabaseSync;
  private open = true;

  constructor(root: string) {
    if (!isAbsolute(root)) {
      throw new ScienceError("Science storage root must be absolute", "INVALID_STORAGE_ROOT");
    }

    mkdirSync(root, { recursive: true, mode: 0o700 });
    chmodSync(root, 0o700);
    this.databasePath = join(root, DATABASE_NAME);
    this.database = new DatabaseSync(this.databasePath);
    chmodSync(this.databasePath, 0o600);
    this.database.exec(`
      PRAGMA foreign_keys = ON;
      PRAGMA synchronous = FULL;
      PRAGMA busy_timeout = 5000;
      PRAGMA journal_mode = WAL;
    `);
    this.migrate();
    this.rebuildProjections();
  }

  createProject(
    workspaceKey: string,
    sessionId: string,
    request: CreateProjectRequest,
  ): ScienceProject {
    ensureOpen(this.open);
    const hash = requestHash("project/created", workspaceKey, request);
    const existing = this.findRequest(request.requestId);
    if (existing) {
      return this.resolveIdempotent(existing, hash, "project/created", scienceProjectSchema);
    }

    const entity = this.transaction(() => {
      const eventId = randomUUID();
      const journalSeq = this.nextSequence();
      const now = Date.now();
      const project = scienceProjectSchema.parse({
        id: randomUUID(),
        kind: "project",
        title: request.title,
        createdAt: now,
        updatedAt: now,
        revision: 1,
        provenance: { eventId, journalSeq, sessionId },
      });
      this.appendEvent(
        journalSeq,
        eventId,
        request.requestId,
        hash,
        workspaceKey,
        "project/created",
        now,
        project,
      );
      this.projectStatement().run(
        project.id,
        workspaceKey,
        project.title,
        project.createdAt,
        project.updatedAt,
        project.revision,
        eventId,
        journalSeq,
        sessionId,
      );
      return project;
    });
    return entity;
  }

  createNotebook(
    workspaceKey: string,
    sessionId: string,
    request: CreateNotebookRequest,
  ): ScienceNotebook {
    ensureOpen(this.open);
    const hash = requestHash("notebook/created", workspaceKey, request);
    const existing = this.findRequest(request.requestId);
    if (existing) {
      return this.resolveIdempotent(existing, hash, "notebook/created", scienceNotebookSchema);
    }
    this.requireProject(workspaceKey, request.projectId);

    return this.transaction(() => {
      const eventId = randomUUID();
      const journalSeq = this.nextSequence();
      const now = Date.now();
      const notebook = scienceNotebookSchema.parse({
        id: randomUUID(),
        projectId: request.projectId,
        kind: "notebook",
        title: request.title,
        cells: [],
        createdAt: now,
        updatedAt: now,
        revision: 1,
        provenance: { eventId, journalSeq, sessionId },
      });
      this.appendEvent(
        journalSeq,
        eventId,
        request.requestId,
        hash,
        workspaceKey,
        "notebook/created",
        now,
        notebook,
      );
      this.notebookStatement().run(
        notebook.id,
        notebook.projectId,
        workspaceKey,
        notebook.title,
        JSON.stringify(notebook.cells),
        notebook.createdAt,
        notebook.updatedAt,
        notebook.revision,
        eventId,
        journalSeq,
        sessionId,
      );
      return notebook;
    });
  }

  createDocument(
    workspaceKey: string,
    sessionId: string,
    request: CreateDocumentRequest,
  ): ScienceDocument {
    ensureOpen(this.open);
    const hash = requestHash("document/created", workspaceKey, request);
    const existing = this.findRequest(request.requestId);
    if (existing) {
      return this.resolveIdempotent(existing, hash, "document/created", scienceDocumentSchema);
    }
    this.requireProject(workspaceKey, request.projectId);

    return this.transaction(() => {
      const eventId = randomUUID();
      const journalSeq = this.nextSequence();
      const now = Date.now();
      const provenance = { eventId, journalSeq, sessionId };
      const contentHash = sourceHash(request.content);
      const format = documentFormat(request.name);
      const document = scienceDocumentSchema.parse({
        id: randomUUID(),
        projectId: request.projectId,
        kind: "document",
        name: request.name,
        format,
        content: request.content,
        revision: 1,
        contentRevision: 1,
        proposals: [],
        revisions: [
          {
            revision: 1,
            contentRevision: 1,
            sourceHash: contentHash,
            previousSourceHash: null,
            reason: "created",
            proposalId: null,
            provenance,
          },
        ],
        diagnostics: analyzeDocument(request.content, format),
        validation: { structural: "checked", compilation: "not-run" },
        createdAt: now,
        updatedAt: now,
        provenance,
      });
      this.appendEvent(
        journalSeq,
        eventId,
        request.requestId,
        hash,
        workspaceKey,
        "document/created",
        now,
        document,
      );
      this.documentStatement().run(
        document.id,
        document.projectId,
        workspaceKey,
        JSON.stringify(document),
        document.createdAt,
      );
      return document;
    });
  }

  createFigure(
    workspaceKey: string,
    sessionId: string,
    request: CreateFigureRequest,
  ): ScienceFigure {
    ensureOpen(this.open);
    const hash = requestHash("figure/created", workspaceKey, request);
    const existing = this.findRequest(request.requestId);
    if (existing) {
      return this.resolveIdempotent(existing, hash, "figure/created", scienceFigureSchema);
    }
    this.requireProject(workspaceKey, request.projectId);
    if (request.artifactId) {
      const artifact = this.database
        .prepare(
          "SELECT kind, project_id FROM science_artifacts WHERE id = ? AND workspace_key = ?",
        )
        .get(request.artifactId, workspaceKey) as
        | { readonly kind: string; readonly project_id: string }
        | undefined;
      if (artifact?.kind !== "figure" || artifact.project_id !== request.projectId) {
        throw new ScienceError(
          "Figure artifact not found in this project and workspace",
          "FIGURE_ARTIFACT_INVALID",
        );
      }
    }

    return this.transaction(() => {
      const eventId = randomUUID();
      const journalSeq = this.nextSequence();
      const now = Date.now();
      const provenance = { eventId, journalSeq, sessionId };
      const figure = scienceFigureSchema.parse({
        id: randomUUID(),
        projectId: request.projectId,
        kind: "figure",
        title: request.title,
        library: request.library,
        code: request.code,
        artifactId: request.artifactId,
        objects: inferFigureObjects(request.code, request.library, randomUUID),
        revision: 1,
        codeRevision: 1,
        proposals: [],
        revisions: [
          {
            revision: 1,
            codeRevision: 1,
            codeHash: codeHash(request.code),
            previousCodeHash: null,
            reason: "created",
            proposalId: null,
            provenance,
          },
        ],
        createdAt: now,
        updatedAt: now,
        provenance,
      });
      this.appendEvent(
        journalSeq,
        eventId,
        request.requestId,
        hash,
        workspaceKey,
        "figure/created",
        now,
        figure,
      );
      this.figureStatement().run(
        figure.id,
        figure.projectId,
        workspaceKey,
        JSON.stringify(figure),
        figure.createdAt,
      );
      return figure;
    });
  }

  createQuestion(
    workspaceKey: string,
    sessionId: string,
    request: CreateQuestionRequest,
  ): ScienceResearchRecord {
    return this.createResearchRecord(workspaceKey, sessionId, request, {
      eventType: "question/created",
      kind: "question",
      sourceEntityIds: [],
      status: "open",
    });
  }

  createHypothesis(
    workspaceKey: string,
    sessionId: string,
    request: CreateHypothesisRequest,
  ): ScienceResearchRecord {
    this.requireResearchRecord(workspaceKey, request.projectId, request.questionId, "question");
    return this.createResearchRecord(workspaceKey, sessionId, request, {
      eventType: "hypothesis/created",
      kind: "hypothesis",
      sourceEntityIds: [request.questionId],
      status: "proposed",
      relation: { toId: request.questionId, type: "motivated_by" },
    });
  }

  recordClaim(
    workspaceKey: string,
    sessionId: string,
    request: RecordClaimRequest,
  ): ScienceResearchRecord {
    if (request.hypothesisId) {
      this.requireResearchRecord(
        workspaceKey,
        request.projectId,
        request.hypothesisId,
        "hypothesis",
      );
    }
    return this.createResearchRecord(workspaceKey, sessionId, request, {
      eventType: "claim/recorded",
      kind: "claim",
      sourceEntityIds: request.hypothesisId ? [request.hypothesisId] : [],
      status: request.status,
      ...(request.hypothesisId
        ? { relation: { toId: request.hypothesisId, type: "derived_from" as const } }
        : {}),
    });
  }

  linkEvidence(
    workspaceKey: string,
    sessionId: string,
    request: LinkEvidenceRequest,
  ): LinkEvidenceResult {
    ensureOpen(this.open);
    const hash = requestHash("evidence/linked", workspaceKey, request);
    const existing = this.findRequest(request.requestId);
    if (existing)
      return this.resolveIdempotent(existing, hash, "evidence/linked", linkEvidenceResultSchema);
    this.requireProject(workspaceKey, request.projectId);
    this.requireResearchRecord(workspaceKey, request.projectId, request.claimId, "claim");
    for (const sourceId of request.sourceEntityIds) {
      if (!this.entityExists(workspaceKey, sourceId)) {
        throw new ScienceError(
          "Evidence source entity not found in this workspace",
          "RESEARCH_ENTITY_NOT_FOUND",
        );
      }
    }

    return this.transaction(() => {
      const eventId = randomUUID();
      const journalSeq = this.nextSequence();
      const now = Date.now();
      const provenance = { eventId, journalSeq, sessionId };
      const evidence = scienceResearchRecordSchema.parse({
        id: randomUUID(),
        projectId: request.projectId,
        kind: "evidence",
        title: request.title,
        summary: request.summary,
        status: "accepted",
        tags: request.tags,
        sourceEntityIds: request.sourceEntityIds,
        createdAt: now,
        updatedAt: now,
        revision: 1,
        provenance,
      });
      const relation = scienceRelationSchema.parse({
        id: randomUUID(),
        projectId: request.projectId,
        fromId: evidence.id,
        toId: request.claimId,
        type: request.relation,
        createdAt: now,
        provenance,
      });
      const result = linkEvidenceResultSchema.parse({ evidence, relation, provenance });
      this.appendEvent(
        journalSeq,
        eventId,
        request.requestId,
        hash,
        workspaceKey,
        "evidence/linked",
        now,
        result,
      );
      this.recordStatement().run(
        evidence.id,
        evidence.projectId,
        workspaceKey,
        JSON.stringify(evidence),
        evidence.createdAt,
      );
      this.relationStatement().run(
        relation.id,
        relation.projectId,
        workspaceKey,
        JSON.stringify(relation),
        relation.createdAt,
      );
      return result;
    });
  }

  defineExperiment(
    workspaceKey: string,
    sessionId: string,
    request: DefineExperimentRequest,
  ): ScienceExperiment {
    ensureOpen(this.open);
    const hash = requestHash("experiment/defined", workspaceKey, request);
    const existing = this.findRequest(request.requestId);
    if (existing) {
      return this.resolveIdempotent(existing, hash, "experiment/defined", scienceExperimentSchema);
    }
    this.requireProject(workspaceKey, request.projectId);
    for (const hypothesisId of request.hypothesisIds) {
      this.requireResearchRecord(workspaceKey, request.projectId, hypothesisId, "hypothesis");
    }

    return this.transaction(() => {
      const eventId = randomUUID();
      const journalSeq = this.nextSequence();
      const now = Date.now();
      const experiment = scienceExperimentSchema.parse({
        id: randomUUID(),
        projectId: request.projectId,
        kind: "experiment",
        title: request.title,
        summary: request.summary,
        protocol: request.protocol,
        hypothesisIds: request.hypothesisIds,
        runIds: [],
        status: "defined",
        tags: request.tags,
        createdAt: now,
        updatedAt: now,
        revision: 1,
        provenance: { eventId, journalSeq, sessionId },
      });
      this.appendEvent(
        journalSeq,
        eventId,
        request.requestId,
        hash,
        workspaceKey,
        "experiment/defined",
        now,
        experiment,
      );
      this.experimentStatement().run(
        experiment.id,
        experiment.projectId,
        workspaceKey,
        JSON.stringify(experiment),
        experiment.createdAt,
      );
      return experiment;
    });
  }

  startRun(workspaceKey: string, sessionId: string, request: StartRunRequest): RunMutation {
    ensureOpen(this.open);
    const hash = requestHash("run/started", workspaceKey, request);
    const existing = this.findRequest(request.requestId);
    if (existing) return this.resolveIdempotent(existing, hash, "run/started", runMutationSchema);
    const row = this.database
      .prepare("SELECT experiment_json FROM science_experiments WHERE id = ? AND workspace_key = ?")
      .get(request.experimentId, workspaceKey) as unknown as ExperimentRow | undefined;
    if (!row)
      throw new ScienceError("Experiment not found in this workspace", "EXPERIMENT_NOT_FOUND");
    const previous = experimentFromRow(row);
    if (previous.revision !== request.expectedRevision) {
      throw new ScienceError("Experiment revision does not match", "REVISION_CONFLICT");
    }

    return this.transaction(() => {
      const eventId = randomUUID();
      const journalSeq = this.nextSequence();
      const now = Date.now();
      const provenance = { eventId, journalSeq, sessionId };
      const run = scienceRunSchema.parse({
        id: randomUUID(),
        projectId: previous.projectId,
        experimentId: previous.id,
        kind: "run",
        status: "running",
        environment: request.environment,
        metrics: {},
        artifactIds: [],
        notes: "",
        startedAt: now,
        finishedAt: null,
        revision: 1,
        provenance,
      });
      const experiment = scienceExperimentSchema.parse({
        ...previous,
        runIds: [...previous.runIds, run.id],
        status: "active",
        updatedAt: now,
        revision: previous.revision + 1,
        provenance,
      });
      const result = runMutationSchema.parse({ experiment, run, provenance });
      this.appendEvent(
        journalSeq,
        eventId,
        request.requestId,
        hash,
        workspaceKey,
        "run/started",
        now,
        result,
      );
      this.updateExperimentProjection(workspaceKey, experiment);
      this.runStatement().run(
        run.id,
        run.projectId,
        run.experimentId,
        workspaceKey,
        JSON.stringify(run),
        run.startedAt,
      );
      return result;
    });
  }

  finishRun(workspaceKey: string, sessionId: string, request: FinishRunRequest): ScienceRun {
    ensureOpen(this.open);
    const hash = requestHash("run/finished", workspaceKey, request);
    const existing = this.findRequest(request.requestId);
    if (existing) return this.resolveIdempotent(existing, hash, "run/finished", scienceRunSchema);
    const row = this.database
      .prepare("SELECT run_json FROM science_runs WHERE id = ? AND workspace_key = ?")
      .get(request.runId, workspaceKey) as unknown as RunRow | undefined;
    if (!row) throw new ScienceError("Run not found in this workspace", "RUN_NOT_FOUND");
    const previous = runFromRow(row);
    if (previous.revision !== request.expectedRevision || previous.status !== "running") {
      throw new ScienceError("Run revision or lifecycle state does not match", "REVISION_CONFLICT");
    }
    for (const artifactId of request.artifactIds) {
      const artifact = this.database
        .prepare(
          "SELECT project_id, run_id FROM science_artifacts WHERE id = ? AND workspace_key = ?",
        )
        .get(artifactId, workspaceKey) as
        | { readonly project_id: string; readonly run_id: string | null }
        | undefined;
      if (
        !artifact ||
        artifact.project_id !== previous.projectId ||
        (artifact.run_id !== null && artifact.run_id !== previous.id)
      ) {
        throw new ScienceError("Run artifact is not owned by this run", "RUN_ARTIFACT_INVALID");
      }
    }

    return this.transaction(() => {
      const eventId = randomUUID();
      const journalSeq = this.nextSequence();
      const now = Date.now();
      const run = scienceRunSchema.parse({
        ...previous,
        status: request.status,
        metrics: request.metrics,
        artifactIds: request.artifactIds,
        notes: request.notes,
        finishedAt: now,
        revision: previous.revision + 1,
        provenance: { eventId, journalSeq, sessionId },
      });
      this.appendEvent(
        journalSeq,
        eventId,
        request.requestId,
        hash,
        workspaceKey,
        "run/finished",
        now,
        run,
      );
      this.updateRunProjection(workspaceKey, run);
      return run;
    });
  }

  prepareProjectExport(
    workspaceKey: string,
    request: ExportProjectRequest,
  ): ProjectExportRecord | undefined {
    ensureOpen(this.open);
    const hash = requestHash("project/exported", workspaceKey, request);
    const existing = this.findRequest(request.requestId);
    if (existing) {
      return this.resolveIdempotent(existing, hash, "project/exported", projectExportRecordSchema);
    }
    this.requireProject(workspaceKey, request.projectId);
    return undefined;
  }

  recordProjectExport(
    workspaceKey: string,
    sessionId: string,
    request: ExportProjectRequest,
    object: CapturedArtifactObject,
    counts: ProjectExportCounts,
  ): ProjectExportRecord {
    ensureOpen(this.open);
    const hash = requestHash("project/exported", workspaceKey, request);
    const existing = this.findRequest(request.requestId);
    if (existing) {
      return this.resolveIdempotent(existing, hash, "project/exported", projectExportRecordSchema);
    }
    return this.transaction(() => {
      const eventId = randomUUID();
      const journalSeq = this.nextSequence();
      const now = Date.now();
      const exported = projectExportRecordSchema.parse({
        id: randomUUID(),
        projectId: request.projectId,
        kind: "export",
        format: RO_CRATE_FORMAT,
        filename: RO_CRATE_FILENAME,
        mediaType: RO_CRATE_MEDIA_TYPE,
        digest: object.digest,
        bytes: object.size,
        counts,
        createdAt: now,
        revision: 1,
        provenance: { eventId, journalSeq, sessionId },
      });
      this.appendEvent(
        journalSeq,
        eventId,
        request.requestId,
        hash,
        workspaceKey,
        "project/exported",
        now,
        exported,
      );
      this.exportStatement().run(
        exported.id,
        exported.projectId,
        workspaceKey,
        JSON.stringify(exported),
        exported.createdAt,
      );
      return exported;
    });
  }

  modifyDocument(
    workspaceKey: string,
    sessionId: string,
    request: ModifyDocumentRequest,
  ): ScienceDocument {
    ensureOpen(this.open);
    const hash = requestHash("document/modified", workspaceKey, request);
    const existing = this.findRequest(request.requestId);
    if (existing) {
      return this.resolveIdempotent(existing, hash, "document/modified", scienceDocumentSchema);
    }

    return this.transaction(() => {
      const row = this.database
        .prepare("SELECT document_json FROM science_documents WHERE id = ? AND workspace_key = ?")
        .get(request.documentId, workspaceKey) as unknown as DocumentRow | undefined;
      if (!row) {
        throw new ScienceError("Document not found in this workspace", "DOCUMENT_NOT_FOUND");
      }
      const previous = documentFromRow(row);
      if (previous.revision !== request.expectedRevision) {
        throw new ScienceError(
          `Document revision ${request.expectedRevision} is stale; current revision is ${previous.revision}`,
          "REVISION_CONFLICT",
        );
      }
      if (request.action === "propose" && previous.proposals.length >= MAX_DOCUMENT_PROPOSALS) {
        throw new ScienceError("Document proposal history is full", "DOCUMENT_HISTORY_FULL");
      }

      const eventId = randomUUID();
      const journalSeq = this.nextSequence();
      const now = Date.now();
      const provenance = { eventId, journalSeq, sessionId };
      const previousHash = sourceHash(previous.content);
      let content = previous.content;
      let contentRevision = previous.contentRevision;
      let reason: "proposal-created" | "proposal-accepted" | "proposal-rejected";
      let proposalId: string;
      let proposals = previous.proposals;

      if (request.action === "propose") {
        if (request.selection.end > previous.content.length) {
          throw new ScienceError("Document selection exceeds the stored source", "INVALID_REQUEST");
        }
        proposalId = randomUUID();
        reason = "proposal-created";
        proposals = [
          ...previous.proposals,
          {
            id: proposalId,
            selection: request.selection,
            originalText: previous.content.slice(request.selection.start, request.selection.end),
            proposedText: request.proposedText,
            instruction: request.instruction,
            reasoning: { classification: "proposal" as const, summary: request.reasoning },
            status: "pending" as const,
            createdAt: now,
            resolvedAt: null,
            createdProvenance: provenance,
            resolvedProvenance: null,
          },
        ];
      } else {
        proposalId = request.proposalId;
        const proposal = previous.proposals.find((candidate) => candidate.id === proposalId);
        if (!proposal) {
          throw new ScienceError("Document proposal not found", "PROPOSAL_NOT_FOUND");
        }
        if (proposal.status !== "pending") {
          throw new ScienceError("Document proposal is no longer pending", "PROPOSAL_NOT_PENDING");
        }
        if (
          previous.content.slice(proposal.selection.start, proposal.selection.end) !==
          proposal.originalText
        ) {
          throw new ScienceError("Document source changed under the proposal", "REVISION_CONFLICT");
        }
        reason = request.action === "accept" ? "proposal-accepted" : "proposal-rejected";
        if (request.action === "accept") {
          content = `${previous.content.slice(0, proposal.selection.start)}${proposal.proposedText}${previous.content.slice(proposal.selection.end)}`;
          if (content.length > MAX_DOCUMENT_SOURCE_LENGTH) {
            throw new ScienceError(
              "Accepted document source exceeds the size limit",
              "INVALID_REQUEST",
            );
          }
          contentRevision += 1;
        }
        proposals = previous.proposals.map((candidate) =>
          candidate.id === proposalId
            ? {
                ...candidate,
                status: request.action === "accept" ? ("accepted" as const) : ("rejected" as const),
                resolvedAt: now,
                resolvedProvenance: provenance,
              }
            : candidate,
        );
      }

      const currentHash = sourceHash(content);
      const document = scienceDocumentSchema.parse({
        ...previous,
        content,
        revision: previous.revision + 1,
        contentRevision,
        proposals,
        revisions: [
          ...previous.revisions,
          {
            revision: previous.revision + 1,
            contentRevision,
            sourceHash: currentHash,
            previousSourceHash: previousHash,
            reason,
            proposalId,
            provenance,
          },
        ],
        diagnostics: analyzeDocument(content, previous.format),
        validation: { structural: "checked", compilation: "not-run" },
        updatedAt: now,
        provenance,
      });
      this.appendEvent(
        journalSeq,
        eventId,
        request.requestId,
        hash,
        workspaceKey,
        "document/modified",
        now,
        document,
      );
      this.updateDocumentProjection(workspaceKey, document);
      return document;
    });
  }

  modifyFigureCode(
    workspaceKey: string,
    sessionId: string,
    request: ModifyFigureCodeRequest,
  ): ScienceFigure {
    ensureOpen(this.open);
    const hash = requestHash("figure/modified", workspaceKey, request);
    const existing = this.findRequest(request.requestId);
    if (existing) {
      return this.resolveIdempotent(existing, hash, "figure/modified", scienceFigureSchema);
    }

    return this.transaction(() => {
      const row = this.database
        .prepare("SELECT figure_json FROM science_figures WHERE id = ? AND workspace_key = ?")
        .get(request.figureId, workspaceKey) as unknown as FigureRow | undefined;
      if (!row) {
        throw new ScienceError("Figure not found in this workspace", "FIGURE_NOT_FOUND");
      }
      const previous = figureFromRow(row);
      if (previous.revision !== request.expectedRevision) {
        throw new ScienceError(
          `Figure revision ${request.expectedRevision} is stale; current revision is ${previous.revision}`,
          "REVISION_CONFLICT",
        );
      }
      if (request.action === "propose" && previous.proposals.length >= MAX_FIGURE_PROPOSALS) {
        throw new ScienceError("Figure proposal history is full", "FIGURE_HISTORY_FULL");
      }

      const eventId = randomUUID();
      const journalSeq = this.nextSequence();
      const now = Date.now();
      const provenance = { eventId, journalSeq, sessionId };
      const previousHash = codeHash(previous.code);
      let code = previous.code;
      let codeRevision = previous.codeRevision;
      let objects = previous.objects;
      let reason: "proposal-created" | "proposal-accepted" | "proposal-rejected";
      let proposalId: string;
      let proposals = previous.proposals;

      if (request.action === "propose") {
        const selectedIds = new Set(request.objectIds);
        const selected = previous.objects.filter((object) => selectedIds.has(object.id));
        if (selected.length !== selectedIds.size) {
          throw new ScienceError(
            "One or more semantic figure objects were not found",
            "FIGURE_OBJECT_NOT_FOUND",
          );
        }
        const selection = {
          start: Math.min(...selected.map((object) => object.codeRange.start)),
          end: Math.max(...selected.map((object) => object.codeRange.end)),
        };
        const overlaps = previous.objects.filter(
          (object) =>
            object.codeRange.start < selection.end && object.codeRange.end > selection.start,
        );
        if (overlaps.some((object) => !selectedIds.has(object.id))) {
          throw new ScienceError(
            "Selected figure objects overlap an unselected semantic object",
            "FIGURE_SELECTION_AMBIGUOUS",
          );
        }
        proposalId = randomUUID();
        reason = "proposal-created";
        proposals = [
          ...previous.proposals,
          {
            id: proposalId,
            objectIds: request.objectIds,
            selection,
            originalCode: previous.code.slice(selection.start, selection.end),
            proposedCode: request.proposedCode,
            instruction: request.instruction,
            reasoning: { classification: "proposal" as const, summary: request.reasoning },
            status: "pending" as const,
            createdAt: now,
            resolvedAt: null,
            createdProvenance: provenance,
            resolvedProvenance: null,
          },
        ];
      } else {
        proposalId = request.proposalId;
        const proposal = previous.proposals.find((candidate) => candidate.id === proposalId);
        if (!proposal) {
          throw new ScienceError("Figure proposal not found", "PROPOSAL_NOT_FOUND");
        }
        if (proposal.status !== "pending") {
          throw new ScienceError("Figure proposal is no longer pending", "PROPOSAL_NOT_PENDING");
        }
        if (
          previous.code.slice(proposal.selection.start, proposal.selection.end) !==
          proposal.originalCode
        ) {
          throw new ScienceError("Figure code changed under the proposal", "REVISION_CONFLICT");
        }
        reason = request.action === "accept" ? "proposal-accepted" : "proposal-rejected";
        if (request.action === "accept") {
          code = `${previous.code.slice(0, proposal.selection.start)}${proposal.proposedCode}${previous.code.slice(proposal.selection.end)}`;
          objects = remapFigureObjects(
            previous.objects,
            new Set(proposal.objectIds),
            proposal.selection,
            proposal.proposedCode.length,
          );
          codeRevision += 1;
        }
        proposals = previous.proposals.map((candidate) =>
          candidate.id === proposalId
            ? {
                ...candidate,
                status: request.action === "accept" ? ("accepted" as const) : ("rejected" as const),
                resolvedAt: now,
                resolvedProvenance: provenance,
              }
            : candidate,
        );
      }

      const figure = scienceFigureSchema.parse({
        ...previous,
        code,
        objects,
        revision: previous.revision + 1,
        codeRevision,
        proposals,
        revisions: [
          ...previous.revisions,
          {
            revision: previous.revision + 1,
            codeRevision,
            codeHash: codeHash(code),
            previousCodeHash: previousHash,
            reason,
            proposalId,
            provenance,
          },
        ],
        updatedAt: now,
        provenance,
      });
      this.appendEvent(
        journalSeq,
        eventId,
        request.requestId,
        hash,
        workspaceKey,
        "figure/modified",
        now,
        figure,
      );
      this.updateFigureProjection(workspaceKey, figure);
      return figure;
    });
  }

  prepareNotebookExecution(
    workspaceKey: string,
    request: ExecuteNotebookCellRequest,
  ): NotebookExecution | undefined {
    ensureOpen(this.open);
    const hash = requestHash("notebook/cell-executed", workspaceKey, request);
    const existing = this.findRequest(request.requestId);
    if (existing) {
      return this.resolveIdempotent(
        existing,
        hash,
        "notebook/cell-executed",
        notebookExecutionSchema,
      );
    }
    const notebook = this.database
      .prepare("SELECT id FROM science_notebooks WHERE id = ? AND workspace_key = ?")
      .get(request.notebookId, workspaceKey);
    if (!notebook) {
      throw new ScienceError("Notebook not found in this workspace", "NOTEBOOK_NOT_FOUND");
    }
    return undefined;
  }

  recordNotebookExecution(
    workspaceKey: string,
    sessionId: string,
    request: ExecuteNotebookCellRequest,
    settled: SettledNotebookCell,
  ): NotebookExecution {
    ensureOpen(this.open);
    const hash = requestHash("notebook/cell-executed", workspaceKey, request);
    const existing = this.findRequest(request.requestId);
    if (existing) {
      return this.resolveIdempotent(
        existing,
        hash,
        "notebook/cell-executed",
        notebookExecutionSchema,
      );
    }
    if ((request.outputArtifact === null) !== (settled.capturedArtifact === undefined)) {
      throw new ScienceError("Notebook artifact capture did not settle", "ARTIFACT_IO_FAILED");
    }

    return this.transaction(() => {
      const row = this.database
        .prepare(
          "SELECT id, project_id, title, cells_json, created_at, updated_at, revision, event_id, event_seq, session_id FROM science_notebooks WHERE id = ? AND workspace_key = ?",
        )
        .get(request.notebookId, workspaceKey) as unknown as NotebookRow | undefined;
      if (!row) {
        throw new ScienceError("Notebook not found in this workspace", "NOTEBOOK_NOT_FOUND");
      }
      const previous = notebookFromRow(row);
      const eventId = randomUUID();
      const journalSeq = this.nextSequence();
      const now = Date.now();
      const executionId = randomUUID();
      const executionCount =
        previous.cells.reduce((highest, cell) => Math.max(highest, cell.executionCount ?? 0), 0) +
        1;
      const artifact =
        request.outputArtifact && settled.capturedArtifact
          ? scienceArtifactSchema.parse({
              id: randomUUID(),
              projectId: previous.projectId,
              kind: request.outputArtifact.kind,
              title: request.outputArtifact.title,
              digest: settled.capturedArtifact.digest,
              mime: request.outputArtifact.mime,
              size: settled.capturedArtifact.size,
              creator: { kind: "session", sessionId },
              runId: executionId,
              environment: settled.environment,
              license: request.outputArtifact.license,
              sourceEntityIds: [previous.id, ...(request.inputArtifactIds ?? [])],
              createdAt: now,
              updatedAt: now,
              revision: 1,
              provenance: { eventId, journalSeq, sessionId },
            })
          : null;
      const cellId = randomUUID();
      const commonCell = {
        executionCount,
        executionTimeMs: settled.durationMs,
        inputArtifactIds: request.inputArtifactIds ?? [],
        outputArtifactIds: artifact ? [artifact.id] : [],
        runtimeEnvironment: settled.environment,
        relatedClaimIds: [],
        relatedExperimentIds: [],
      };
      const outputSource =
        [settled.stdout.text, settled.stderr.text.length > 0 ? settled.stderr.text : undefined]
          .filter((value): value is string => value !== undefined && value.length > 0)
          .join(settled.stdout.text.length > 0 ? "\n" : "") || notebookOutputText(settled.outputs);
      const notebook = scienceNotebookSchema.parse({
        ...previous,
        cells: [
          ...previous.cells,
          { ...commonCell, id: cellId, kind: "code", outputs: [], source: request.source },
          {
            ...commonCell,
            id: randomUUID(),
            kind: "output",
            outputs: settled.outputs,
            source: outputSource,
          },
        ],
        updatedAt: now,
        revision: previous.revision + 1,
        provenance: { eventId, journalSeq, sessionId },
      });
      const execution = notebookExecutionSchema.parse({
        id: executionId,
        notebookId: notebook.id,
        cellId,
        executionCount,
        status: settled.status,
        stdout: settled.stdout,
        stderr: settled.stderr,
        outputs: settled.outputs,
        exitCode: settled.exitCode,
        signal: settled.signal,
        durationMs: settled.durationMs,
        environment: settled.environment,
        inputArtifactIds: request.inputArtifactIds ?? [],
        artifact,
        notebook,
        provenance: { eventId, journalSeq, sessionId },
      });
      this.appendEvent(
        journalSeq,
        eventId,
        request.requestId,
        hash,
        workspaceKey,
        "notebook/cell-executed",
        now,
        execution,
      );
      this.updateNotebookProjection(workspaceKey, notebook);
      if (artifact) this.insertArtifactProjection(workspaceKey, artifact);
      return execution;
    });
  }

  registerArtifact(
    workspaceKey: string,
    sessionId: string,
    request: RegisterArtifactRequest,
    capture: () => CapturedArtifactObject,
    identity?: object,
  ): ScienceArtifact;
  registerArtifact(
    workspaceKey: string,
    sessionId: string,
    request: RegisterArtifactRequest,
    capture: () => Promise<CapturedArtifactObject>,
    identity?: object,
  ): Promise<ScienceArtifact>;
  registerArtifact(
    workspaceKey: string,
    sessionId: string,
    request: RegisterArtifactRequest,
    capture: () => CapturedArtifactObject | Promise<CapturedArtifactObject>,
    identity: object = request,
  ): ScienceArtifact | Promise<ScienceArtifact> {
    ensureOpen(this.open);
    const hash = requestHash("artifact/registered", workspaceKey, identity);
    const existing = this.findRequest(request.requestId);
    if (existing) {
      return this.resolveIdempotent(existing, hash, "artifact/registered", scienceArtifactSchema);
    }
    this.requireProject(workspaceKey, request.projectId);
    for (const sourceEntityId of request.sourceEntityIds) {
      if (!this.entityExists(workspaceKey, sourceEntityId)) {
        throw new ScienceError(
          "Provenance source entity not found in this workspace",
          "PROVENANCE_ENTITY_NOT_FOUND",
        );
      }
    }

    const commit = (captured: CapturedArtifactObject): ScienceArtifact => {
      ensureOpen(this.open);
      return this.transaction(() => {
        const eventId = randomUUID();
        const journalSeq = this.nextSequence();
        const now = Date.now();
        const artifact = scienceArtifactSchema.parse({
          id: randomUUID(),
          projectId: request.projectId,
          kind: request.kind,
          title: request.title,
          digest: captured.digest,
          mime: request.mime,
          size: captured.size,
          creator: { kind: "session", sessionId },
          runId: request.runId,
          environment: request.environment,
          license: request.license,
          sourceEntityIds: request.sourceEntityIds,
          createdAt: now,
          updatedAt: now,
          revision: 1,
          provenance: { eventId, journalSeq, sessionId },
        });
        this.appendEvent(
          journalSeq,
          eventId,
          request.requestId,
          hash,
          workspaceKey,
          "artifact/registered",
          now,
          artifact,
        );
        this.insertArtifactProjection(workspaceKey, artifact);
        return artifact;
      });
    };
    const captured = capture();
    return captured instanceof Promise ? captured.then(commit) : commit(captured);
  }

  compareRuns(workspaceKey: string, request: CompareRunsRequest): RunComparison {
    ensureOpen(this.open);
    const runs = request.runIds.map((runId) => {
      const row = this.database
        .prepare("SELECT run_json FROM science_runs WHERE id = ? AND workspace_key = ?")
        .get(runId, workspaceKey) as unknown as RunRow | undefined;
      if (!row) throw new ScienceError("Run not found in this workspace", "RUN_NOT_FOUND");
      return runFromRow(row);
    });
    const baseline = runs[0];
    const experimentId = baseline?.experimentId;
    if (
      !experimentId ||
      runs.some(
        (run) => run.experimentId !== experimentId || run.status === "running" || !run.finishedAt,
      )
    ) {
      throw new ScienceError(
        "Run comparison requires completed runs from one experiment",
        "RUN_COMPARISON_INVALID",
      );
    }
    const metrics = Object.keys(baseline.metrics)
      .filter((metric) => runs.every((run) => metric in run.metrics))
      .sort();
    return runComparisonSchema.parse({
      experimentId,
      baselineRunId: baseline.id,
      runIds: runs.map((run) => run.id),
      classification: "inference",
      deltas: metrics.map((metric) => {
        const baselineValue = baseline.metrics[metric];
        if (baselineValue === undefined) throw new Error("Baseline metric disappeared");
        return {
          metric,
          values: runs.map((run) => {
            const value = run.metrics[metric];
            if (value === undefined) throw new Error("Compared metric disappeared");
            return value - baselineValue;
          }),
        };
      }),
    });
  }

  getWorkspace(workspaceKey: string): ScienceWorkspaceSnapshot {
    ensureOpen(this.open);
    const projects = this.database
      .prepare(
        "SELECT id, title, created_at, updated_at, revision, event_id, event_seq, session_id FROM science_projects WHERE workspace_key = ? ORDER BY created_at, id",
      )
      .all(workspaceKey)
      .map((row) => projectFromRow(row as unknown as ProjectRow));
    const notebooks = this.database
      .prepare(
        "SELECT id, project_id, title, cells_json, created_at, updated_at, revision, event_id, event_seq, session_id FROM science_notebooks WHERE workspace_key = ? ORDER BY created_at, id",
      )
      .all(workspaceKey)
      .map((row) => notebookFromRow(row as unknown as NotebookRow));
    const artifacts = this.database
      .prepare(
        "SELECT id, project_id, kind, title, digest, mime, size, creator_session_id, run_id, environment_json, license, source_entity_ids_json, created_at, updated_at, revision, event_id, event_seq, session_id FROM science_artifacts WHERE workspace_key = ? ORDER BY created_at, id",
      )
      .all(workspaceKey)
      .map((row) => artifactFromRow(row as unknown as ArtifactRow));
    const documents = this.database
      .prepare(
        "SELECT document_json FROM science_documents WHERE workspace_key = ? ORDER BY created_at, id",
      )
      .all(workspaceKey)
      .map((row) => documentFromRow(row as unknown as DocumentRow));
    const figures = this.database
      .prepare(
        "SELECT figure_json FROM science_figures WHERE workspace_key = ? ORDER BY created_at, id",
      )
      .all(workspaceKey)
      .map((row) => figureFromRow(row as unknown as FigureRow));
    const records = this.database
      .prepare(
        "SELECT record_json FROM science_records WHERE workspace_key = ? ORDER BY created_at, id",
      )
      .all(workspaceKey)
      .map((row) => recordFromRow(row as unknown as RecordRow));
    const relations = this.database
      .prepare(
        "SELECT relation_json FROM science_relations WHERE workspace_key = ? ORDER BY created_at, id",
      )
      .all(workspaceKey)
      .map((row) => relationFromRow(row as unknown as RelationRow));
    const experiments = this.database
      .prepare(
        "SELECT experiment_json FROM science_experiments WHERE workspace_key = ? ORDER BY created_at, id",
      )
      .all(workspaceKey)
      .map((row) => experimentFromRow(row as unknown as ExperimentRow));
    const runs = this.database
      .prepare("SELECT run_json FROM science_runs WHERE workspace_key = ? ORDER BY started_at, id")
      .all(workspaceKey)
      .map((row) => runFromRow(row as unknown as RunRow));
    const exports = this.database
      .prepare(
        "SELECT export_json FROM science_exports WHERE workspace_key = ? ORDER BY created_at, id",
      )
      .all(workspaceKey)
      .map((row) => exportFromRow(row as unknown as ExportRow));
    return {
      projects,
      notebooks,
      artifacts,
      documents,
      figures,
      records,
      relations,
      experiments,
      runs,
      exports,
    };
  }

  journalCount(): number {
    ensureOpen(this.open);
    const row = this.database.prepare("SELECT COUNT(*) AS count FROM science_journal").get() as {
      count: number;
    };
    return row.count;
  }

  close(): void {
    if (!this.open) return;
    this.open = false;
    this.database.close();
  }

  private migrate(): void {
    this.database.exec(`
      CREATE TABLE IF NOT EXISTS science_migrations (
        version INTEGER PRIMARY KEY,
        applied_at INTEGER NOT NULL
      ) STRICT;
    `);
    const applied = new Set(
      (
        this.database.prepare("SELECT version FROM science_migrations ORDER BY version").all() as {
          version: number;
        }[]
      ).map((row) => row.version),
    );
    const newest = Math.max(0, ...applied);
    if (newest > MIGRATION_VERSION) {
      throw new Error(`Science database version ${newest} is newer than supported`);
    }
    for (let version = 1; version <= MIGRATION_VERSION; version += 1) {
      if (applied.has(version)) continue;
      this.database.exec("BEGIN IMMEDIATE");
      try {
        this.applyMigration(version);
        this.database
          .prepare("INSERT INTO science_migrations(version, applied_at) VALUES (?, ?)")
          .run(version, Date.now());
        this.database.exec("COMMIT");
      } catch (error) {
        this.database.exec("ROLLBACK");
        throw error;
      }
    }
  }

  private applyMigration(version: number): void {
    if (version === 1) {
      this.database.exec(`
        CREATE TABLE IF NOT EXISTS science_journal (
          seq INTEGER PRIMARY KEY AUTOINCREMENT,
          event_id TEXT NOT NULL UNIQUE,
          request_id TEXT NOT NULL UNIQUE,
          request_hash TEXT NOT NULL,
          workspace_key TEXT NOT NULL,
          type TEXT NOT NULL,
          occurred_at INTEGER NOT NULL,
          payload_json TEXT NOT NULL
        ) STRICT;
        CREATE INDEX IF NOT EXISTS science_journal_workspace_idx
          ON science_journal(workspace_key, seq);

        CREATE TABLE IF NOT EXISTS science_projects (
          id TEXT PRIMARY KEY,
          workspace_key TEXT NOT NULL,
          title TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          revision INTEGER NOT NULL,
          event_id TEXT NOT NULL UNIQUE,
          event_seq INTEGER NOT NULL UNIQUE,
          session_id TEXT NOT NULL
        ) STRICT;
        CREATE INDEX IF NOT EXISTS science_projects_workspace_idx
          ON science_projects(workspace_key, created_at, id);

        CREATE TABLE IF NOT EXISTS science_notebooks (
          id TEXT PRIMARY KEY,
          project_id TEXT NOT NULL REFERENCES science_projects(id),
          workspace_key TEXT NOT NULL,
          title TEXT NOT NULL,
          cells_json TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          revision INTEGER NOT NULL,
          event_id TEXT NOT NULL UNIQUE,
          event_seq INTEGER NOT NULL UNIQUE,
          session_id TEXT NOT NULL
        ) STRICT;
        CREATE INDEX IF NOT EXISTS science_notebooks_workspace_idx
          ON science_notebooks(workspace_key, created_at, id);
      `);
      return;
    }
    if (version === 2) {
      this.database.exec(`
        CREATE TABLE IF NOT EXISTS science_artifacts (
          id TEXT PRIMARY KEY,
          project_id TEXT NOT NULL REFERENCES science_projects(id),
          workspace_key TEXT NOT NULL,
          kind TEXT NOT NULL,
          title TEXT NOT NULL,
          digest TEXT NOT NULL,
          mime TEXT NOT NULL,
          size INTEGER NOT NULL,
          creator_session_id TEXT NOT NULL,
          run_id TEXT,
          environment_json TEXT NOT NULL,
          license TEXT,
          source_entity_ids_json TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          revision INTEGER NOT NULL,
          event_id TEXT NOT NULL UNIQUE,
          event_seq INTEGER NOT NULL UNIQUE,
          session_id TEXT NOT NULL
        ) STRICT;
        CREATE INDEX IF NOT EXISTS science_artifacts_workspace_idx
          ON science_artifacts(workspace_key, created_at, id);
        CREATE INDEX IF NOT EXISTS science_artifacts_digest_idx
          ON science_artifacts(digest);
      `);
      return;
    }
    if (version === 3) {
      this.database.exec(`
        CREATE TABLE IF NOT EXISTS science_documents (
          id TEXT PRIMARY KEY,
          project_id TEXT NOT NULL REFERENCES science_projects(id),
          workspace_key TEXT NOT NULL,
          document_json TEXT NOT NULL,
          created_at INTEGER NOT NULL
        ) STRICT;
        CREATE INDEX IF NOT EXISTS science_documents_workspace_idx
          ON science_documents(workspace_key, created_at, id);
      `);
      return;
    }
    if (version === 4) {
      this.database.exec(`
        CREATE TABLE IF NOT EXISTS science_figures (
          id TEXT PRIMARY KEY,
          project_id TEXT NOT NULL REFERENCES science_projects(id),
          workspace_key TEXT NOT NULL,
          figure_json TEXT NOT NULL,
          created_at INTEGER NOT NULL
        ) STRICT;
        CREATE INDEX IF NOT EXISTS science_figures_workspace_idx
          ON science_figures(workspace_key, created_at, id);
      `);
      return;
    }
    if (version === 5) {
      this.database.exec(`
        CREATE TABLE IF NOT EXISTS science_records (
          id TEXT PRIMARY KEY,
          project_id TEXT NOT NULL REFERENCES science_projects(id),
          workspace_key TEXT NOT NULL,
          record_json TEXT NOT NULL,
          created_at INTEGER NOT NULL
        ) STRICT;
        CREATE INDEX IF NOT EXISTS science_records_workspace_idx
          ON science_records(workspace_key, created_at, id);

        CREATE TABLE IF NOT EXISTS science_relations (
          id TEXT PRIMARY KEY,
          project_id TEXT NOT NULL REFERENCES science_projects(id),
          workspace_key TEXT NOT NULL,
          relation_json TEXT NOT NULL,
          created_at INTEGER NOT NULL
        ) STRICT;
        CREATE INDEX IF NOT EXISTS science_relations_workspace_idx
          ON science_relations(workspace_key, created_at, id);

        CREATE TABLE IF NOT EXISTS science_experiments (
          id TEXT PRIMARY KEY,
          project_id TEXT NOT NULL REFERENCES science_projects(id),
          workspace_key TEXT NOT NULL,
          experiment_json TEXT NOT NULL,
          created_at INTEGER NOT NULL
        ) STRICT;
        CREATE INDEX IF NOT EXISTS science_experiments_workspace_idx
          ON science_experiments(workspace_key, created_at, id);

        CREATE TABLE IF NOT EXISTS science_runs (
          id TEXT PRIMARY KEY,
          project_id TEXT NOT NULL REFERENCES science_projects(id),
          experiment_id TEXT NOT NULL REFERENCES science_experiments(id),
          workspace_key TEXT NOT NULL,
          run_json TEXT NOT NULL,
          started_at INTEGER NOT NULL
        ) STRICT;
        CREATE INDEX IF NOT EXISTS science_runs_workspace_idx
          ON science_runs(workspace_key, started_at, id);

        CREATE TABLE IF NOT EXISTS science_exports (
          id TEXT PRIMARY KEY,
          project_id TEXT NOT NULL REFERENCES science_projects(id),
          workspace_key TEXT NOT NULL,
          export_json TEXT NOT NULL,
          created_at INTEGER NOT NULL
        ) STRICT;
        CREATE INDEX IF NOT EXISTS science_exports_workspace_idx
          ON science_exports(workspace_key, created_at, id);
      `);
      return;
    }
    throw new Error(`Unsupported science migration ${version}`);
  }

  private rebuildProjections(): void {
    const rows = this.database
      .prepare(
        "SELECT seq, type, occurred_at, payload_json, workspace_key FROM science_journal ORDER BY seq",
      )
      .all() as unknown as JournalRow[];
    this.transaction(() => {
      this.database.exec(
        "DELETE FROM science_runs; DELETE FROM science_experiments; DELETE FROM science_relations; DELETE FROM science_records; DELETE FROM science_exports; DELETE FROM science_figures; DELETE FROM science_documents; DELETE FROM science_artifacts; DELETE FROM science_notebooks; DELETE FROM science_projects;",
      );
      for (const row of rows) {
        if (row.type === "project/created") {
          const project = scienceProjectSchema.parse(JSON.parse(row.payload_json));
          this.projectStatement().run(
            project.id,
            row.workspace_key,
            project.title,
            project.createdAt,
            project.updatedAt,
            project.revision,
            project.provenance.eventId,
            project.provenance.journalSeq,
            project.provenance.sessionId,
          );
          continue;
        }
        if (row.type === "notebook/created") {
          const notebook = scienceNotebookSchema.parse(JSON.parse(row.payload_json));
          this.notebookStatement().run(
            notebook.id,
            notebook.projectId,
            row.workspace_key,
            notebook.title,
            JSON.stringify(notebook.cells),
            notebook.createdAt,
            notebook.updatedAt,
            notebook.revision,
            notebook.provenance.eventId,
            notebook.provenance.journalSeq,
            notebook.provenance.sessionId,
          );
          continue;
        }
        if (row.type === "notebook/cell-executed") {
          const execution = notebookExecutionSchema.parse(JSON.parse(row.payload_json));
          this.updateNotebookProjection(row.workspace_key, execution.notebook);
          if (execution.artifact) {
            this.insertArtifactProjection(row.workspace_key, execution.artifact);
          }
          continue;
        }
        if (row.type === "artifact/registered") {
          const artifact = scienceArtifactSchema.parse(JSON.parse(row.payload_json));
          this.insertArtifactProjection(row.workspace_key, artifact);
          continue;
        }
        if (row.type === "document/created") {
          const document = scienceDocumentSchema.parse(JSON.parse(row.payload_json));
          this.documentStatement().run(
            document.id,
            document.projectId,
            row.workspace_key,
            JSON.stringify(document),
            document.createdAt,
          );
          continue;
        }
        if (row.type === "document/modified") {
          const document = scienceDocumentSchema.parse(JSON.parse(row.payload_json));
          this.updateDocumentProjection(row.workspace_key, document);
          continue;
        }
        if (row.type === "figure/created") {
          const figure = scienceFigureSchema.parse(JSON.parse(row.payload_json));
          this.figureStatement().run(
            figure.id,
            figure.projectId,
            row.workspace_key,
            JSON.stringify(figure),
            figure.createdAt,
          );
          continue;
        }
        if (row.type === "figure/modified") {
          const figure = scienceFigureSchema.parse(JSON.parse(row.payload_json));
          this.updateFigureProjection(row.workspace_key, figure);
          continue;
        }
        if (
          row.type === "question/created" ||
          row.type === "hypothesis/created" ||
          row.type === "claim/recorded"
        ) {
          const payload = JSON.parse(row.payload_json) as {
            readonly record: unknown;
            readonly relation: unknown | null;
          };
          const record = scienceResearchRecordSchema.parse(payload.record);
          this.recordStatement().run(
            record.id,
            record.projectId,
            row.workspace_key,
            JSON.stringify(record),
            record.createdAt,
          );
          if (payload.relation) {
            const relation = scienceRelationSchema.parse(payload.relation);
            this.relationStatement().run(
              relation.id,
              relation.projectId,
              row.workspace_key,
              JSON.stringify(relation),
              relation.createdAt,
            );
          }
          continue;
        }
        if (row.type === "evidence/linked") {
          const result = linkEvidenceResultSchema.parse(JSON.parse(row.payload_json));
          this.recordStatement().run(
            result.evidence.id,
            result.evidence.projectId,
            row.workspace_key,
            JSON.stringify(result.evidence),
            result.evidence.createdAt,
          );
          this.relationStatement().run(
            result.relation.id,
            result.relation.projectId,
            row.workspace_key,
            JSON.stringify(result.relation),
            result.relation.createdAt,
          );
          continue;
        }
        if (row.type === "experiment/defined") {
          const experiment = scienceExperimentSchema.parse(JSON.parse(row.payload_json));
          this.experimentStatement().run(
            experiment.id,
            experiment.projectId,
            row.workspace_key,
            JSON.stringify(experiment),
            experiment.createdAt,
          );
          continue;
        }
        if (row.type === "run/started") {
          const result = runMutationSchema.parse(JSON.parse(row.payload_json));
          this.updateExperimentProjection(row.workspace_key, result.experiment);
          this.runStatement().run(
            result.run.id,
            result.run.projectId,
            result.run.experimentId,
            row.workspace_key,
            JSON.stringify(result.run),
            result.run.startedAt,
          );
          continue;
        }
        if (row.type === "run/finished") {
          const run = scienceRunSchema.parse(JSON.parse(row.payload_json));
          this.updateRunProjection(row.workspace_key, run);
          continue;
        }
        if (row.type === "project/exported") {
          const exported = projectExportRecordSchema.parse(JSON.parse(row.payload_json));
          this.exportStatement().run(
            exported.id,
            exported.projectId,
            row.workspace_key,
            JSON.stringify(exported),
            exported.createdAt,
          );
          continue;
        }
        throw new Error(`Unsupported science journal event '${row.type}' at sequence ${row.seq}`);
      }
    });
  }

  private createResearchRecord(
    workspaceKey: string,
    sessionId: string,
    request: {
      readonly requestId: string;
      readonly projectId: string;
      readonly title: string;
      readonly summary: string;
      readonly tags: readonly string[];
    },
    options: {
      readonly eventType: "question/created" | "hypothesis/created" | "claim/recorded";
      readonly kind: "question" | "hypothesis" | "claim";
      readonly sourceEntityIds: readonly string[];
      readonly status: ScienceResearchRecord["status"];
      readonly relation?: {
        readonly toId: string;
        readonly type: "motivated_by" | "derived_from";
      };
    },
  ): ScienceResearchRecord {
    ensureOpen(this.open);
    const hash = requestHash(options.eventType, workspaceKey, request);
    const existing = this.findRequest(request.requestId);
    if (existing) return this.resolveIdempotentRecord(existing, hash, options.eventType);
    this.requireProject(workspaceKey, request.projectId);

    return this.transaction(() => {
      const eventId = randomUUID();
      const journalSeq = this.nextSequence();
      const now = Date.now();
      const provenance = { eventId, journalSeq, sessionId };
      const record = scienceResearchRecordSchema.parse({
        id: randomUUID(),
        projectId: request.projectId,
        kind: options.kind,
        title: request.title,
        summary: request.summary,
        status: options.status,
        tags: request.tags,
        sourceEntityIds: options.sourceEntityIds,
        createdAt: now,
        updatedAt: now,
        revision: 1,
        provenance,
      });
      const relation = options.relation
        ? scienceRelationSchema.parse({
            id: randomUUID(),
            projectId: request.projectId,
            fromId: record.id,
            toId: options.relation.toId,
            type: options.relation.type,
            createdAt: now,
            provenance,
          })
        : null;
      const payload = { record, relation };
      this.appendEvent(
        journalSeq,
        eventId,
        request.requestId,
        hash,
        workspaceKey,
        options.eventType,
        now,
        payload,
      );
      this.recordStatement().run(
        record.id,
        record.projectId,
        workspaceKey,
        JSON.stringify(record),
        record.createdAt,
      );
      if (relation) {
        this.relationStatement().run(
          relation.id,
          relation.projectId,
          workspaceKey,
          JSON.stringify(relation),
          relation.createdAt,
        );
      }
      return record;
    });
  }

  private requireProject(workspaceKey: string, projectId: string): void {
    const project = this.database
      .prepare("SELECT id FROM science_projects WHERE id = ? AND workspace_key = ?")
      .get(projectId, workspaceKey);
    if (!project)
      throw new ScienceError("Project not found in this workspace", "PROJECT_NOT_FOUND");
  }

  private requireResearchRecord(
    workspaceKey: string,
    projectId: string,
    recordId: string,
    kind: ScienceResearchRecord["kind"],
  ): ScienceResearchRecord {
    const row = this.database
      .prepare("SELECT record_json FROM science_records WHERE id = ? AND workspace_key = ?")
      .get(recordId, workspaceKey) as unknown as RecordRow | undefined;
    const record = row ? recordFromRow(row) : undefined;
    if (!record || record.projectId !== projectId || record.kind !== kind) {
      throw new ScienceError(
        `${kind} not found in this project and workspace`,
        "RESEARCH_ENTITY_NOT_FOUND",
      );
    }
    return record;
  }

  private findRequest(requestId: string): RequestRow | undefined {
    return this.database
      .prepare("SELECT type, request_hash, payload_json FROM science_journal WHERE request_id = ?")
      .get(requestId) as unknown as RequestRow | undefined;
  }

  private resolveIdempotent<T>(
    existing: RequestRow,
    hash: string,
    type: string,
    schema: { parse(value: unknown): T },
  ): T {
    if (existing.type !== type || existing.request_hash !== hash) {
      throw new ScienceError(
        "Idempotency key was already used for a different science mutation",
        "IDEMPOTENCY_CONFLICT",
      );
    }
    return schema.parse(JSON.parse(existing.payload_json));
  }

  private resolveIdempotentRecord(
    existing: RequestRow,
    hash: string,
    type: "question/created" | "hypothesis/created" | "claim/recorded",
  ): ScienceResearchRecord {
    if (existing.type !== type || existing.request_hash !== hash) {
      throw new ScienceError(
        "Idempotency key was already used for a different science mutation",
        "IDEMPOTENCY_CONFLICT",
      );
    }
    const payload = JSON.parse(existing.payload_json) as { readonly record: unknown };
    return scienceResearchRecordSchema.parse(payload.record);
  }

  private entityExists(workspaceKey: string, entityId: string): boolean {
    const row = this.database
      .prepare(`
        SELECT 1 AS found FROM science_projects WHERE workspace_key = ? AND id = ?
        UNION ALL
        SELECT 1 AS found FROM science_notebooks WHERE workspace_key = ? AND id = ?
        UNION ALL
        SELECT 1 AS found FROM science_artifacts WHERE workspace_key = ? AND id = ?
        UNION ALL
        SELECT 1 AS found FROM science_documents WHERE workspace_key = ? AND id = ?
        UNION ALL
        SELECT 1 AS found FROM science_figures WHERE workspace_key = ? AND id = ?
        UNION ALL
        SELECT 1 AS found FROM science_records WHERE workspace_key = ? AND id = ?
        UNION ALL
        SELECT 1 AS found FROM science_experiments WHERE workspace_key = ? AND id = ?
        UNION ALL
        SELECT 1 AS found FROM science_runs WHERE workspace_key = ? AND id = ?
        UNION ALL
        SELECT 1 AS found FROM science_exports WHERE workspace_key = ? AND id = ?
        LIMIT 1
      `)
      .get(
        workspaceKey,
        entityId,
        workspaceKey,
        entityId,
        workspaceKey,
        entityId,
        workspaceKey,
        entityId,
        workspaceKey,
        entityId,
        workspaceKey,
        entityId,
        workspaceKey,
        entityId,
        workspaceKey,
        entityId,
        workspaceKey,
        entityId,
      );
    return row !== undefined;
  }

  private nextSequence(): number {
    const row = this.database
      .prepare("SELECT COALESCE(MAX(seq), 0) + 1 AS seq FROM science_journal")
      .get() as { seq: number };
    return row.seq;
  }

  private appendEvent(
    sequence: number,
    eventId: string,
    requestId: string,
    hash: string,
    workspaceKey: string,
    type: string,
    occurredAt: number,
    payload: object,
  ): void {
    this.database
      .prepare(
        "INSERT INTO science_journal(seq, event_id, request_id, request_hash, workspace_key, type, occurred_at, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
      )
      .run(
        sequence,
        eventId,
        requestId,
        hash,
        workspaceKey,
        type,
        occurredAt,
        JSON.stringify(payload),
      );
  }

  private projectStatement(): StatementSync {
    return this.database.prepare(
      "INSERT INTO science_projects(id, workspace_key, title, created_at, updated_at, revision, event_id, event_seq, session_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    );
  }

  private notebookStatement(): StatementSync {
    return this.database.prepare(
      "INSERT INTO science_notebooks(id, project_id, workspace_key, title, cells_json, created_at, updated_at, revision, event_id, event_seq, session_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    );
  }

  private artifactStatement(): StatementSync {
    return this.database.prepare(
      "INSERT INTO science_artifacts(id, project_id, workspace_key, kind, title, digest, mime, size, creator_session_id, run_id, environment_json, license, source_entity_ids_json, created_at, updated_at, revision, event_id, event_seq, session_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    );
  }

  private documentStatement(): StatementSync {
    return this.database.prepare(
      "INSERT INTO science_documents(id, project_id, workspace_key, document_json, created_at) VALUES (?, ?, ?, ?, ?)",
    );
  }

  private figureStatement(): StatementSync {
    return this.database.prepare(
      "INSERT INTO science_figures(id, project_id, workspace_key, figure_json, created_at) VALUES (?, ?, ?, ?, ?)",
    );
  }

  private recordStatement(): StatementSync {
    return this.database.prepare(
      "INSERT INTO science_records(id, project_id, workspace_key, record_json, created_at) VALUES (?, ?, ?, ?, ?)",
    );
  }

  private relationStatement(): StatementSync {
    return this.database.prepare(
      "INSERT INTO science_relations(id, project_id, workspace_key, relation_json, created_at) VALUES (?, ?, ?, ?, ?)",
    );
  }

  private experimentStatement(): StatementSync {
    return this.database.prepare(
      "INSERT INTO science_experiments(id, project_id, workspace_key, experiment_json, created_at) VALUES (?, ?, ?, ?, ?)",
    );
  }

  private runStatement(): StatementSync {
    return this.database.prepare(
      "INSERT INTO science_runs(id, project_id, experiment_id, workspace_key, run_json, started_at) VALUES (?, ?, ?, ?, ?, ?)",
    );
  }

  private exportStatement(): StatementSync {
    return this.database.prepare(
      "INSERT INTO science_exports(id, project_id, workspace_key, export_json, created_at) VALUES (?, ?, ?, ?, ?)",
    );
  }

  private insertArtifactProjection(workspaceKey: string, artifact: ScienceArtifact): void {
    this.artifactStatement().run(
      artifact.id,
      artifact.projectId,
      workspaceKey,
      artifact.kind,
      artifact.title,
      artifact.digest,
      artifact.mime,
      artifact.size,
      artifact.creator.sessionId,
      artifact.runId,
      JSON.stringify(artifact.environment),
      artifact.license,
      JSON.stringify(artifact.sourceEntityIds),
      artifact.createdAt,
      artifact.updatedAt,
      artifact.revision,
      artifact.provenance.eventId,
      artifact.provenance.journalSeq,
      artifact.provenance.sessionId,
    );
  }

  private updateNotebookProjection(workspaceKey: string, notebook: ScienceNotebook): void {
    const result = this.database
      .prepare(
        "UPDATE science_notebooks SET title = ?, cells_json = ?, updated_at = ?, revision = ?, event_id = ?, event_seq = ?, session_id = ? WHERE id = ? AND workspace_key = ?",
      )
      .run(
        notebook.title,
        JSON.stringify(notebook.cells),
        notebook.updatedAt,
        notebook.revision,
        notebook.provenance.eventId,
        notebook.provenance.journalSeq,
        notebook.provenance.sessionId,
        notebook.id,
        workspaceKey,
      );
    if (result.changes !== 1) {
      throw new ScienceError("Notebook not found in this workspace", "NOTEBOOK_NOT_FOUND");
    }
  }

  private updateDocumentProjection(workspaceKey: string, document: ScienceDocument): void {
    const result = this.database
      .prepare("UPDATE science_documents SET document_json = ? WHERE id = ? AND workspace_key = ?")
      .run(JSON.stringify(document), document.id, workspaceKey);
    if (result.changes !== 1) {
      throw new ScienceError("Document not found in this workspace", "DOCUMENT_NOT_FOUND");
    }
  }

  private updateFigureProjection(workspaceKey: string, figure: ScienceFigure): void {
    const result = this.database
      .prepare("UPDATE science_figures SET figure_json = ? WHERE id = ? AND workspace_key = ?")
      .run(JSON.stringify(figure), figure.id, workspaceKey);
    if (result.changes !== 1) {
      throw new ScienceError("Figure not found in this workspace", "FIGURE_NOT_FOUND");
    }
  }

  private updateExperimentProjection(workspaceKey: string, experiment: ScienceExperiment): void {
    const result = this.database
      .prepare(
        "UPDATE science_experiments SET experiment_json = ? WHERE id = ? AND workspace_key = ?",
      )
      .run(JSON.stringify(experiment), experiment.id, workspaceKey);
    if (result.changes !== 1) {
      throw new ScienceError("Experiment not found in this workspace", "EXPERIMENT_NOT_FOUND");
    }
  }

  private updateRunProjection(workspaceKey: string, run: ScienceRun): void {
    const result = this.database
      .prepare("UPDATE science_runs SET run_json = ? WHERE id = ? AND workspace_key = ?")
      .run(JSON.stringify(run), run.id, workspaceKey);
    if (result.changes !== 1) {
      throw new ScienceError("Run not found in this workspace", "RUN_NOT_FOUND");
    }
  }

  private transaction<T>(operation: () => T): T {
    this.database.exec("BEGIN IMMEDIATE");
    try {
      const result = operation();
      this.database.exec("COMMIT");
      return result;
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
  }
}
