import { createHash } from "node:crypto";
import { realpathSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Context } from "@deepseek-ai/cordis";
import { SessionId } from "@deepseek-ai/dsh-session";
import { TypertRemoteService } from "@deepseek-ai/dsh-typert-protocol";
import s from "@deepseek-ai/schemastery";
import { isArtifactMetadataMime } from "./artifact-metadata.js";
import { ArtifactStore } from "./artifact-store.js";
import {
  type CompareRunsRequest,
  type CreateDocumentRequest,
  type CreateFigureRequest,
  type CreateHypothesisRequest,
  type CreateNotebookRequest,
  type CreateProjectRequest,
  type CreateQuestionRequest,
  compareRunsRequestSchema,
  createDocumentRequestSchema,
  createFigureRequestSchema,
  createHypothesisRequestSchema,
  createNotebookRequestSchema,
  createProjectRequestSchema,
  createQuestionRequestSchema,
  type DefineExperimentRequest,
  defineExperimentRequestSchema,
  type ExecuteNotebookCellRequest,
  type ExportProjectRequest,
  executeNotebookCellRequestSchema,
  exportProjectRequestSchema,
  type FigureLibrary,
  type FigureReproducibilityMetadata,
  type FigureSourceReferenceInput,
  type FinishRunRequest,
  figureReproducibilityMetadataSchema,
  finishRunRequestSchema,
  type GetResearchObjectRequest,
  getResearchObjectRequestSchema,
  type ImportArtifactRequest,
  importArtifactRequestSchema,
  type LinkEvidenceRequest,
  type LinkEvidenceResult,
  type LiteratureSearchRequest,
  type LiteratureSearchResult,
  linkEvidenceRequestSchema,
  MAX_SCIENCE_IMPORT_BYTES,
  MAX_TYPST_PDF_BYTES,
  MAX_TYPST_SOURCE_BYTES,
  type ModifyDocumentRequest,
  type ModifyFigureCodeRequest,
  modifyDocumentRequestSchema,
  modifyFigureCodeRequestSchema,
  type NotebookExecution,
  type PreviewArtifactRequest,
  type PreviewTypstDocumentRequest,
  type ProjectExportCounts,
  previewArtifactRequestSchema,
  previewTypstDocumentRequestSchema,
  type RecordClaimRequest,
  type RegisterArtifactRequest,
  type ResolveTypstSourceAtPointRequest,
  type RoCrateMetadataDocument,
  type RunComparison,
  type RunMutation,
  recordClaimRequestSchema,
  registerArtifactRequestSchema,
  resolveTypstSourceAtPointRequestSchema,
  type ScienceArtifact,
  type ScienceArtifactPreview,
  type ScienceDocument,
  type ScienceExperiment,
  type ScienceFigure,
  type ScienceNotebook,
  type ScienceProject,
  type ScienceProjectExport,
  type ScienceResearchRecord,
  type ScienceRun,
  type ScienceWorkspaceSnapshot,
  type StartRunRequest,
  scienceImportType,
  scienceProjectExportSchema,
  startRunRequestSchema,
  type TypstDocumentPreview,
  type TypstSourceTarget,
  type TypstSourceUpdate,
  typstDocumentPreviewSchema,
  type UpdateTypstSourceRequest,
  updateTypstSourceRequestSchema,
} from "./contracts.js";
import { ScienceError } from "./errors.js";
import { codeHash } from "./figure.js";
import { ScienceJournal } from "./journal.js";
import { JupyMcpRuntime } from "./jupymcp-runtime.js";
import { LiteratureSearchRuntime, ZoteroBibliographySource } from "./literature.js";
import { PythonRuntime } from "./python-runtime.js";
import { createResearchObject } from "./research-object.js";
import { tabularPreview } from "./tabular-preview.js";
import { TypstPreviewRuntime } from "./typst-preview.js";

export {
  ARTIFACT_METADATA_KEYWORD,
  ARTIFACT_METADATA_MIMES,
  type ArtifactMetadataMime,
  countPdfMetadataRecords,
  countPngMetadataChunks,
  countSvgMetadataRecords,
  extractArtifactMetadata,
  extractPdfMetadata,
  extractPngMetadata,
  extractSvgMetadata,
  MAX_ARTIFACT_METADATA_BYTES,
} from "./artifact-metadata.js";
export { ArtifactStore } from "./artifact-store.js";
export * from "./bibliography.js";
export * from "./contracts.js";
export { runScienceDemo, type ScienceDemoResult } from "./demo.js";
export * from "./errors.js";
export { ScienceJournal } from "./journal.js";
export * from "./literature.js";
export { createResearchObject } from "./research-object.js";
export interface Config {
  readonly embedArtifactMetadata?: boolean;
  readonly root: string;
  readonly maxArtifactBytes?: number;
  readonly maxCellOutputBytes?: number;
  readonly maxExportBytes?: number;
  readonly maxNotebookDocumentBytes?: number;
  readonly notebookRuntime?: "isolated" | "jupymcp";
  readonly processGraceMs?: number;
  readonly jupymcpArgs?: readonly string[];
  readonly jupymcpCommand?: string;
  readonly jupymcpRequestTimeoutMs?: number;
  readonly pythonCommand?: string;
  readonly typstCommand?: string;
  readonly writingPreviewRuntimeCommand?: string;
  readonly typstInitialCompileTimeoutMs?: number;
  readonly typstMaxDiagnosticsBytes?: number;
  readonly typstMaxPdfBytes?: number;
  readonly typstMaxSourceBytes?: number;
}

const DEFAULT_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024;
const DEFAULT_MAX_CELL_OUTPUT_BYTES = 256 * 1024;
const DEFAULT_MAX_EXPORT_BYTES = 5 * 1024 * 1024;
const DEFAULT_MAX_NOTEBOOK_DOCUMENT_BYTES = 5 * 1024 * 1024;
const MAX_TEXT_PREVIEW_BYTES = 64 * 1024;
const MAX_IMAGE_PREVIEW_BYTES = 2 * 1024 * 1024;
const MAX_NOTEBOOK_INPUT_BYTES = 32 * 1024 * 1024;
const IMAGE_PREVIEW_MIME = new Set(["image/png", "image/jpeg", "image/gif", "image/webp"]);
const DEFAULT_PROCESS_GRACE_MS = 2_000;
const DEFAULT_JUPYMCP_COMMAND = "jupymcp";
const DEFAULT_JUPYMCP_REQUEST_TIMEOUT_MS = 60_000;
const DEFAULT_PYTHON_COMMAND = "python3";
const DEFAULT_TYPST_COMMAND = "typst";
const DEFAULT_WRITING_PREVIEW_RUNTIME_COMMAND = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "bin",
  `${process.platform}-${process.arch}`,
  process.platform === "win32"
    ? "swarmx-writing-preview-runtime.exe"
    : "swarmx-writing-preview-runtime",
);
const DEFAULT_TYPST_INITIAL_COMPILE_TIMEOUT_MS = 15_000;
const DEFAULT_TYPST_MAX_DIAGNOSTICS_BYTES = 64 * 1024;
const DEFAULT_TYPST_MAX_PDF_BYTES = 32 * 1024 * 1024;
const DEFAULT_TYPST_MAX_SOURCE_BYTES = 1024 * 1024;
const REDACTED_METADATA = "[redacted]";
const SECRET_ENVIRONMENT_KEY = /api.?key|credential|password|private.?key|secret|token/iu;
const ABSOLUTE_PATH_VALUE = /^(?:\/|[a-z]:[\\/]|\\\\)/iu;

declare module "@deepseek-ai/cordis" {
  interface Context {
    science: ScienceService;
  }
}

function abortIfRequested(signal?: AbortSignal): void {
  signal?.throwIfAborted();
}

function parseRequest<T>(schema: { parse(value: unknown): T }, value: unknown): T {
  try {
    return schema.parse(value);
  } catch (error) {
    throw new ScienceError("Invalid science request", "INVALID_REQUEST", { cause: error });
  }
}

function redactEnvironment(environment: Readonly<Record<string, string>>): Record<string, string> {
  return Object.fromEntries(
    Object.entries(environment).map(([key, value]) => [
      key,
      SECRET_ENVIRONMENT_KEY.test(key) || ABSOLUTE_PATH_VALUE.test(value)
        ? REDACTED_METADATA
        : value,
    ]),
  );
}

type NormalizedFigureSource = FigureReproducibilityMetadata["sources"][number];

function inferFigureLibrary(code: string): FigureLibrary | undefined {
  if (/(?:library\s*\(\s*["']ggplot2|ggplot\s*\()/iu.test(code)) return "ggplot2";
  if (/(?:\bseaborn\b|\bsns\.)/iu.test(code)) return "seaborn";
  if (/(?:\bplotly\b|\bpx\.|\bgo\.)/iu.test(code)) return "plotly";
  if (/(?:\bmatplotlib\b|\bplt\.)/iu.test(code)) return "matplotlib";
  return undefined;
}

function normalizedSourceKey(source: NormalizedFigureSource): string {
  if (source.kind === "artifact") return `artifact:${source.artifactId}`;
  if (source.kind === "workspace") {
    return `workspace:${source.relativePath}:${source.digest}`;
  }
  return `s3:${source.uri}:${source.versionId ?? ""}:${source.digest ?? ""}`;
}

function uniqueNormalizedSources(
  sources: readonly NormalizedFigureSource[],
): NormalizedFigureSource[] {
  const unique = new Map(sources.map((source) => [normalizedSourceKey(source), source]));
  if (unique.size > 32) {
    throw new ScienceError("Artifact metadata may contain at most 32 sources", "INVALID_REQUEST");
  }
  return [...unique.values()];
}

function normalizeFigureSources(
  sources: readonly FigureSourceReferenceInput[],
  artifacts: readonly ScienceArtifact[],
  fingerprintWorkspaceSource: (relativePath: string) => `sha256:${string}`,
): NormalizedFigureSource[] {
  const normalized = sources.map((source): NormalizedFigureSource => {
    if (source.kind === "workspace") {
      const digest = fingerprintWorkspaceSource(source.relativePath);
      if (source.digest && source.digest !== digest) {
        throw new ScienceError(
          "Workspace source digest does not match its current bytes",
          "ARTIFACT_SOURCE_CHANGED",
        );
      }
      return { ...source, digest };
    }
    if (source.kind !== "artifact") return source;
    const artifact = artifacts.find((candidate) => candidate.id === source.artifactId);
    if (!artifact) {
      throw new ScienceError("Artifact source not found in this workspace", "ARTIFACT_NOT_FOUND");
    }
    return { kind: "artifact", artifactId: artifact.id, digest: artifact.digest };
  });
  return uniqueNormalizedSources(normalized);
}

function artifactMetadataDocument(input: {
  readonly code: string;
  readonly environment: Readonly<Record<string, string>>;
  readonly generationId: string;
  readonly library: FigureLibrary;
  readonly sources: readonly NormalizedFigureSource[];
}): FigureReproducibilityMetadata {
  return parseRequest(figureReproducibilityMetadataSchema, {
    schema: "dsh-science.figure-provenance",
    version: 1,
    generationId: input.generationId,
    generator: {
      library: input.library,
      code: input.code,
      codeHash: codeHash(input.code),
    },
    sources: input.sources,
    environment: input.environment,
  });
}

/** Workspace-scoped science journal exposed to Host and strict Client Remote callers. */
export class ScienceService extends TypertRemoteService {
  static inject = ["sessions", "subprocess"];
  static Config = s.object({
    embedArtifactMetadata: s.boolean().default(true),
    root: s.string().required(),
    maxArtifactBytes: s.natural().min(1).default(DEFAULT_MAX_ARTIFACT_BYTES),
    maxCellOutputBytes: s.natural().min(1).max(1_000_000).default(DEFAULT_MAX_CELL_OUTPUT_BYTES),
    maxExportBytes: s.natural().min(1).max(10_000_000).default(DEFAULT_MAX_EXPORT_BYTES),
    maxNotebookDocumentBytes: s
      .natural()
      .min(1)
      .max(10_000_000)
      .default(DEFAULT_MAX_NOTEBOOK_DOCUMENT_BYTES),
    notebookRuntime: s.union(["jupymcp", "isolated"]).default("jupymcp"),
    processGraceMs: s.natural().min(1).max(60_000).default(DEFAULT_PROCESS_GRACE_MS),
    jupymcpArgs: s.array(s.string()).default([]),
    jupymcpCommand: s.string().default(DEFAULT_JUPYMCP_COMMAND),
    jupymcpRequestTimeoutMs: s
      .natural()
      .min(1_000)
      .max(3_600_000)
      .default(DEFAULT_JUPYMCP_REQUEST_TIMEOUT_MS),
    pythonCommand: s.string().default(DEFAULT_PYTHON_COMMAND),
    typstCommand: s.string().default(DEFAULT_TYPST_COMMAND),
    writingPreviewRuntimeCommand: s.string().default(DEFAULT_WRITING_PREVIEW_RUNTIME_COMMAND),
    typstInitialCompileTimeoutMs: s
      .natural()
      .min(100)
      .max(60_000)
      .default(DEFAULT_TYPST_INITIAL_COMPILE_TIMEOUT_MS),
    typstMaxDiagnosticsBytes: s
      .natural()
      .min(1_024)
      .max(1024 * 1024)
      .default(DEFAULT_TYPST_MAX_DIAGNOSTICS_BYTES),
    typstMaxPdfBytes: s
      .natural()
      .min(1_024)
      .max(MAX_TYPST_PDF_BYTES)
      .default(DEFAULT_TYPST_MAX_PDF_BYTES),
    typstMaxSourceBytes: s
      .natural()
      .min(1_024)
      .max(MAX_TYPST_SOURCE_BYTES)
      .default(DEFAULT_TYPST_MAX_SOURCE_BYTES),
  });

  private readonly artifacts: ArtifactStore;
  private readonly embedArtifactMetadata: boolean;
  private readonly executions = new Map<
    string,
    { readonly fingerprint: string; readonly promise: Promise<NotebookExecution> }
  >();
  private readonly journal: ScienceJournal;
  private readonly literature: LiteratureSearchRuntime;
  private readonly registrations = new Map<
    string,
    { readonly fingerprint: string; readonly promise: Promise<ScienceArtifact> }
  >();
  private readonly maxExportBytes: number;
  private readonly notebookRuntime: JupyMcpRuntime | PythonRuntime;
  private readonly notebookRuntimeKind: "isolated" | "jupymcp";
  private readonly typstRuntime: TypstPreviewRuntime;

  constructor(ctx: Context, config: Config) {
    super(ctx, "science");
    this.journal = new ScienceJournal(config.root);
    this.literature = new LiteratureSearchRuntime(config.root, new ZoteroBibliographySource());
    this.artifacts = new ArtifactStore(
      config.root,
      config.maxArtifactBytes ?? DEFAULT_MAX_ARTIFACT_BYTES,
    );
    this.embedArtifactMetadata = config.embedArtifactMetadata ?? true;
    this.notebookRuntimeKind = config.notebookRuntime ?? "jupymcp";
    this.notebookRuntime =
      this.notebookRuntimeKind === "jupymcp"
        ? new JupyMcpRuntime({
            args: config.jupymcpArgs ?? [],
            command: config.jupymcpCommand ?? DEFAULT_JUPYMCP_COMMAND,
            maxNotebookBytes:
              config.maxNotebookDocumentBytes ?? DEFAULT_MAX_NOTEBOOK_DOCUMENT_BYTES,
            maxOutputBytes: config.maxCellOutputBytes ?? DEFAULT_MAX_CELL_OUTPUT_BYTES,
            requestTimeoutMs: config.jupymcpRequestTimeoutMs ?? DEFAULT_JUPYMCP_REQUEST_TIMEOUT_MS,
          })
        : new PythonRuntime(ctx.subprocess, {
            command: config.pythonCommand ?? DEFAULT_PYTHON_COMMAND,
            graceMs: config.processGraceMs ?? DEFAULT_PROCESS_GRACE_MS,
            maxOutputBytes: config.maxCellOutputBytes ?? DEFAULT_MAX_CELL_OUTPUT_BYTES,
          });
    this.maxExportBytes = config.maxExportBytes ?? DEFAULT_MAX_EXPORT_BYTES;
    this.typstRuntime = new TypstPreviewRuntime(ctx.subprocess, {
      command: config.typstCommand ?? DEFAULT_TYPST_COMMAND,
      runtimeCommand:
        config.writingPreviewRuntimeCommand ?? DEFAULT_WRITING_PREVIEW_RUNTIME_COMMAND,
      graceMs: config.processGraceMs ?? DEFAULT_PROCESS_GRACE_MS,
      initialCompileTimeoutMs:
        config.typstInitialCompileTimeoutMs ?? DEFAULT_TYPST_INITIAL_COMPILE_TIMEOUT_MS,
      maxDiagnosticsBytes: config.typstMaxDiagnosticsBytes ?? DEFAULT_TYPST_MAX_DIAGNOSTICS_BYTES,
      maxPdfBytes: config.typstMaxPdfBytes ?? DEFAULT_TYPST_MAX_PDF_BYTES,
      maxSourceBytes: config.typstMaxSourceBytes ?? DEFAULT_TYPST_MAX_SOURCE_BYTES,
    });
    ctx.effect(
      () => async () => {
        try {
          await Promise.all([this.notebookRuntime.close(), this.typstRuntime.close()]);
        } finally {
          this.journal.close();
        }
      },
      "dsh-science: close runtime and journal",
    );
  }

  createProject(
    sessionId: SessionId,
    request: CreateProjectRequest,
    signal?: AbortSignal,
  ): ScienceProject {
    abortIfRequested(signal);
    const parsed = parseRequest(createProjectRequestSchema, request);
    return this.journal.createProject(this.workspace(sessionId).key, sessionId, parsed);
  }

  createQuestion(
    sessionId: SessionId,
    request: CreateQuestionRequest,
    signal?: AbortSignal,
  ): ScienceResearchRecord {
    abortIfRequested(signal);
    const parsed = parseRequest(createQuestionRequestSchema, request);
    return this.journal.createQuestion(this.workspace(sessionId).key, sessionId, parsed);
  }

  createHypothesis(
    sessionId: SessionId,
    request: CreateHypothesisRequest,
    signal?: AbortSignal,
  ): ScienceResearchRecord {
    abortIfRequested(signal);
    const parsed = parseRequest(createHypothesisRequestSchema, request);
    return this.journal.createHypothesis(this.workspace(sessionId).key, sessionId, parsed);
  }

  recordClaim(
    sessionId: SessionId,
    request: RecordClaimRequest,
    signal?: AbortSignal,
  ): ScienceResearchRecord {
    abortIfRequested(signal);
    const parsed = parseRequest(recordClaimRequestSchema, request);
    return this.journal.recordClaim(this.workspace(sessionId).key, sessionId, parsed);
  }

  linkEvidence(
    sessionId: SessionId,
    request: LinkEvidenceRequest,
    signal?: AbortSignal,
  ): LinkEvidenceResult {
    abortIfRequested(signal);
    const parsed = parseRequest(linkEvidenceRequestSchema, request);
    return this.journal.linkEvidence(this.workspace(sessionId).key, sessionId, parsed);
  }

  defineExperiment(
    sessionId: SessionId,
    request: DefineExperimentRequest,
    signal?: AbortSignal,
  ): ScienceExperiment {
    abortIfRequested(signal);
    const parsed = parseRequest(defineExperimentRequestSchema, request);
    return this.journal.defineExperiment(this.workspace(sessionId).key, sessionId, parsed);
  }

  startRun(sessionId: SessionId, request: StartRunRequest, signal?: AbortSignal): RunMutation {
    abortIfRequested(signal);
    const parsed = parseRequest(startRunRequestSchema, request);
    return this.journal.startRun(this.workspace(sessionId).key, sessionId, {
      ...parsed,
      environment: redactEnvironment(parsed.environment),
    });
  }

  finishRun(sessionId: SessionId, request: FinishRunRequest, signal?: AbortSignal): ScienceRun {
    abortIfRequested(signal);
    const parsed = parseRequest(finishRunRequestSchema, request);
    return this.journal.finishRun(this.workspace(sessionId).key, sessionId, parsed);
  }

  compareRuns(
    sessionId: SessionId,
    request: CompareRunsRequest,
    signal?: AbortSignal,
  ): RunComparison {
    abortIfRequested(signal);
    const parsed = parseRequest(compareRunsRequestSchema, request);
    return this.journal.compareRuns(this.workspace(sessionId).key, parsed);
  }

  exportProject(
    sessionId: SessionId,
    request: ExportProjectRequest,
    signal?: AbortSignal,
  ): ScienceProjectExport {
    abortIfRequested(signal);
    const parsed = parseRequest(exportProjectRequestSchema, request);
    const workspace = this.workspace(sessionId);
    const existing = this.journal.prepareProjectExport(workspace.key, parsed);
    if (existing) {
      return scienceProjectExportSchema.parse({
        ...existing,
        classification: "fact",
        content: this.artifacts.readText(existing.digest, this.maxExportBytes, signal),
      });
    }
    const snapshot = this.journal.getWorkspace(workspace.key);
    const project = snapshot.projects.find((candidate) => candidate.id === parsed.projectId);
    if (!project)
      throw new ScienceError("Project not found in this workspace", "PROJECT_NOT_FOUND");
    const bundle = {
      notebooks: snapshot.notebooks.filter((entity) => entity.projectId === project.id),
      artifacts: snapshot.artifacts.filter((entity) => entity.projectId === project.id),
      documents: snapshot.documents.filter((entity) => entity.projectId === project.id),
      figures: snapshot.figures.filter((entity) => entity.projectId === project.id),
      records: snapshot.records.filter((entity) => entity.projectId === project.id),
      relations: snapshot.relations.filter((entity) => entity.projectId === project.id),
      experiments: snapshot.experiments.filter((entity) => entity.projectId === project.id),
      runs: snapshot.runs.filter((entity) => entity.projectId === project.id),
    };
    const counts: ProjectExportCounts = {
      projects: 1,
      notebooks: bundle.notebooks.length,
      artifacts: bundle.artifacts.length,
      documents: bundle.documents.length,
      figures: bundle.figures.length,
      records: bundle.records.length,
      relations: bundle.relations.length,
      experiments: bundle.experiments.length,
      runs: bundle.runs.length,
    };
    const content = `${JSON.stringify(createResearchObject(snapshot, project.id), null, 2)}\n`;
    const object = this.artifacts.publishText(content, this.maxExportBytes, signal);
    abortIfRequested(signal);
    const exported = this.journal.recordProjectExport(
      workspace.key,
      sessionId,
      parsed,
      object,
      counts,
    );
    return scienceProjectExportSchema.parse({ ...exported, classification: "fact", content });
  }

  createDocument(
    sessionId: SessionId,
    request: CreateDocumentRequest,
    signal?: AbortSignal,
  ): ScienceDocument {
    abortIfRequested(signal);
    const parsed = parseRequest(createDocumentRequestSchema, request);
    return this.journal.createDocument(this.workspace(sessionId).key, sessionId, parsed);
  }

  createFigure(
    sessionId: SessionId,
    request: CreateFigureRequest,
    signal?: AbortSignal,
  ): ScienceFigure {
    abortIfRequested(signal);
    const parsed = parseRequest(createFigureRequestSchema, request);
    return this.journal.createFigure(this.workspace(sessionId).key, sessionId, parsed);
  }

  createNotebook(
    sessionId: SessionId,
    request: CreateNotebookRequest,
    signal?: AbortSignal,
  ): ScienceNotebook {
    abortIfRequested(signal);
    const parsed = parseRequest(createNotebookRequestSchema, request);
    return this.journal.createNotebook(this.workspace(sessionId).key, sessionId, parsed);
  }

  executeNotebookCell(
    sessionId: SessionId,
    request: ExecuteNotebookCellRequest,
    signal?: AbortSignal,
  ): Promise<NotebookExecution> {
    abortIfRequested(signal);
    const parsed = parseRequest(executeNotebookCellRequestSchema, request);
    const workspace = this.workspace(sessionId);
    const committed = this.journal.prepareNotebookExecution(workspace.key, parsed);
    if (committed) return Promise.resolve(committed);

    const key = `${workspace.key}:${parsed.requestId}`;
    const fingerprint = JSON.stringify(parsed);
    const active = this.executions.get(key);
    if (active) {
      if (active.fingerprint !== fingerprint) {
        throw new ScienceError(
          "Idempotency key is already executing a different science mutation",
          "IDEMPOTENCY_CONFLICT",
        );
      }
      return active.promise;
    }
    const promise = this.executeNotebookCellOnce(workspace, sessionId, parsed, signal).finally(() =>
      this.executions.delete(key),
    );
    this.executions.set(key, { fingerprint, promise });
    return promise;
  }

  async registerArtifact(
    sessionId: SessionId,
    request: RegisterArtifactRequest,
    signal?: AbortSignal,
  ): Promise<ScienceArtifact> {
    abortIfRequested(signal);
    const parsed = parseRequest(registerArtifactRequestSchema, request);
    const workspace = this.workspace(sessionId);
    const metadataArtifactIds =
      this.embedArtifactMetadata && parsed.reproducibilityMetadata
        ? parsed.reproducibilityMetadata.sources.flatMap((source) =>
            source.kind === "artifact" ? [source.artifactId] : [],
          )
        : [];
    const redacted = parseRequest(registerArtifactRequestSchema, {
      ...parsed,
      environment: redactEnvironment(parsed.environment),
      sourceEntityIds: [...new Set([...parsed.sourceEntityIds, ...metadataArtifactIds])],
    });
    const key = `${workspace.key}\0${redacted.requestId}`;
    const fingerprint = createHash("sha256").update(JSON.stringify(redacted)).digest("hex");
    const active = this.registrations.get(key);
    if (active) {
      if (active.fingerprint !== fingerprint) {
        throw new ScienceError(
          "Idempotency key is already registering a different science artifact",
          "IDEMPOTENCY_CONFLICT",
        );
      }
      return active.promise;
    }
    const promise = Promise.resolve(
      this.journal.registerArtifact(workspace.key, sessionId, redacted, async () => {
        const normalizedSources =
          this.embedArtifactMetadata && redacted.reproducibilityMetadata
            ? normalizeFigureSources(
                redacted.reproducibilityMetadata.sources,
                this.journal.getWorkspace(workspace.key).artifacts,
                (relativePath) =>
                  this.artifacts.fingerprint(workspace.root, relativePath, signal).digest,
              )
            : [];
        const metadata =
          this.embedArtifactMetadata && redacted.reproducibilityMetadata
            ? artifactMetadataDocument({
                code: redacted.reproducibilityMetadata.code,
                environment: redacted.environment,
                generationId: redacted.requestId,
                library: redacted.reproducibilityMetadata.library,
                sources: normalizedSources,
              })
            : undefined;
        if (metadata && !isArtifactMetadataMime(redacted.mime)) {
          throw new ScienceError("Artifact metadata MIME is unsupported", "INVALID_REQUEST");
        }
        const captureMetadata =
          metadata && isArtifactMetadataMime(redacted.mime)
            ? { metadata, mime: redacted.mime }
            : undefined;
        return this.artifacts.capture(
          workspace.root,
          redacted.relativePath,
          signal,
          captureMetadata,
        );
      }),
    ).finally(() => this.registrations.delete(key));
    this.registrations.set(key, { fingerprint, promise });
    return promise;
  }

  importArtifact(
    sessionId: SessionId,
    request: ImportArtifactRequest,
    signal?: AbortSignal,
  ): ScienceArtifact {
    abortIfRequested(signal);
    const parsed = parseRequest(importArtifactRequestSchema, request);
    const importedType = scienceImportType(parsed.name);
    if (!importedType) {
      throw new ScienceError("Imported artifact type is unsupported", "INVALID_REQUEST");
    }
    const bytes = Buffer.from(parsed.dataBase64, "base64");
    if (
      bytes.byteLength === 0 ||
      bytes.byteLength > MAX_SCIENCE_IMPORT_BYTES ||
      bytes.toString("base64") !== parsed.dataBase64
    ) {
      throw new ScienceError("Imported artifact bytes are not canonical base64", "INVALID_REQUEST");
    }
    abortIfRequested(signal);
    const contentDigest = `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
    const registration: RegisterArtifactRequest = {
      requestId: parsed.requestId,
      projectId: parsed.projectId,
      relativePath: parsed.name,
      kind: importedType.kind,
      title: parsed.name,
      mime: importedType.mime,
      runId: null,
      environment: {},
      license: null,
      sourceEntityIds: [],
    };
    const workspace = this.workspace(sessionId);
    return this.journal.registerArtifact(
      workspace.key,
      sessionId,
      registration,
      () => this.artifacts.publishBytes(bytes, MAX_SCIENCE_IMPORT_BYTES, signal),
      {
        requestId: parsed.requestId,
        projectId: parsed.projectId,
        name: parsed.name,
        kind: importedType.kind,
        mime: importedType.mime,
        contentDigest,
      },
    );
  }

  searchLiterature(
    sessionId: SessionId,
    request: LiteratureSearchRequest,
    signal?: AbortSignal,
  ): Promise<LiteratureSearchResult> {
    abortIfRequested(signal);
    return this.literature.search(this.workspace(sessionId).key, request, signal);
  }

  getWorkspace(sessionId: SessionId, signal?: AbortSignal): ScienceWorkspaceSnapshot {
    abortIfRequested(signal);
    return this.journal.getWorkspace(this.workspace(sessionId).key);
  }

  getResearchObject(
    sessionId: SessionId,
    request: GetResearchObjectRequest,
    signal?: AbortSignal,
  ): RoCrateMetadataDocument {
    abortIfRequested(signal);
    const parsed = parseRequest(getResearchObjectRequestSchema, request);
    return createResearchObject(
      this.journal.getWorkspace(this.workspace(sessionId).key),
      parsed.projectId,
    );
  }

  previewArtifact(
    sessionId: SessionId,
    request: PreviewArtifactRequest,
    signal?: AbortSignal,
  ): ScienceArtifactPreview {
    abortIfRequested(signal);
    const parsed = parseRequest(previewArtifactRequestSchema, request);
    const artifact = this.journal
      .getWorkspace(this.workspace(sessionId).key)
      .artifacts.find((candidate) => candidate.id === parsed.artifactId);
    if (!artifact) {
      throw new ScienceError("Artifact not found in this workspace", "ARTIFACT_NOT_FOUND");
    }
    const identity = {
      artifactId: artifact.id,
      digest: artifact.digest,
      mime: artifact.mime,
      size: artifact.size,
    };
    if (IMAGE_PREVIEW_MIME.has(artifact.mime)) {
      if (artifact.size > MAX_IMAGE_PREVIEW_BYTES) {
        return { kind: "unavailable", ...identity, reason: "too-large" };
      }
      const bytes = this.artifacts.readBytes(artifact.digest, MAX_IMAGE_PREVIEW_BYTES, signal);
      return {
        kind: "image",
        ...identity,
        dataUrl: `data:${artifact.mime};base64,${Buffer.from(bytes).toString("base64")}`,
      };
    }
    if (artifact.mime.startsWith("text/") || artifact.mime === "application/json") {
      if (artifact.size > MAX_TEXT_PREVIEW_BYTES) {
        return { kind: "unavailable", ...identity, reason: "too-large" };
      }
      const text = this.artifacts.readText(artifact.digest, MAX_TEXT_PREVIEW_BYTES, signal);
      const table = tabularPreview(artifact.mime, text);
      return table
        ? { ...table, ...identity }
        : {
            kind: "text",
            ...identity,
            text,
          };
    }
    return { kind: "unavailable", ...identity, reason: "unsupported" };
  }

  async previewTypstDocument(
    sessionId: SessionId,
    request: PreviewTypstDocumentRequest,
    signal?: AbortSignal,
  ): Promise<TypstDocumentPreview> {
    abortIfRequested(signal);
    const parsed = parseRequest(previewTypstDocumentRequestSchema, request);
    const workspace = this.workspace(sessionId);
    return typstDocumentPreviewSchema.parse(
      await this.typstRuntime.preview({
        workspaceKey: workspace.key,
        workspaceRoot: workspace.root,
        relativePath: parsed.relativePath,
        ...(signal === undefined ? {} : { signal }),
      }),
    );
  }

  async updateTypstSource(
    sessionId: SessionId,
    request: UpdateTypstSourceRequest,
    signal?: AbortSignal,
  ): Promise<TypstSourceUpdate> {
    abortIfRequested(signal);
    const parsed = parseRequest(updateTypstSourceRequestSchema, request);
    const workspace = this.workspace(sessionId);
    return this.typstRuntime.updateSource({
      workspaceKey: workspace.key,
      workspaceRoot: workspace.root,
      ...parsed,
      ...(signal === undefined ? {} : { signal }),
    });
  }

  async resolveTypstSourceAtPoint(
    sessionId: SessionId,
    request: ResolveTypstSourceAtPointRequest,
    signal?: AbortSignal,
  ): Promise<TypstSourceTarget | null> {
    abortIfRequested(signal);
    const parsed = parseRequest(resolveTypstSourceAtPointRequestSchema, request);
    const workspace = this.workspace(sessionId);
    return this.typstRuntime.resolveSourceAtPoint({
      workspaceKey: workspace.key,
      workspaceRoot: workspace.root,
      ...parsed,
      ...(signal === undefined ? {} : { signal }),
    });
  }

  modifyDocument(
    sessionId: SessionId,
    request: ModifyDocumentRequest,
    signal?: AbortSignal,
  ): ScienceDocument {
    abortIfRequested(signal);
    const parsed = parseRequest(modifyDocumentRequestSchema, request);
    return this.journal.modifyDocument(this.workspace(sessionId).key, sessionId, parsed);
  }

  modifyFigureCode(
    sessionId: SessionId,
    request: ModifyFigureCodeRequest,
    signal?: AbortSignal,
  ): ScienceFigure {
    abortIfRequested(signal);
    const parsed = parseRequest(modifyFigureCodeRequestSchema, request);
    return this.journal.modifyFigureCode(this.workspace(sessionId).key, sessionId, parsed);
  }

  journalCount(): number {
    return this.journal.journalCount();
  }

  private async executeNotebookCellOnce(
    workspace: { key: string; root: string },
    sessionId: SessionId,
    request: ExecuteNotebookCellRequest,
    signal?: AbortSignal,
  ): Promise<NotebookExecution> {
    const workspaceSnapshot = this.journal.getWorkspace(workspace.key);
    const inputArtifacts = (request.inputArtifactIds ?? []).map((artifactId) => {
      const artifact = workspaceSnapshot.artifacts.find((candidate) => candidate.id === artifactId);
      if (!artifact) {
        throw new ScienceError("Artifact not found in this workspace", "ARTIFACT_NOT_FOUND");
      }
      return artifact;
    });
    const requestedMetadata = request.outputArtifact?.reproducibilityMetadata;
    const shouldEmbedMetadata =
      this.embedArtifactMetadata &&
      request.outputArtifact?.kind === "figure" &&
      isArtifactMetadataMime(request.outputArtifact.mime) &&
      requestedMetadata !== false;
    const metadataLibrary = shouldEmbedMetadata
      ? (requestedMetadata?.library ?? inferFigureLibrary(request.source))
      : undefined;
    if (shouldEmbedMetadata && !metadataLibrary) {
      throw new ScienceError(
        "Figure metadata requires an explicit or inferable plotting library",
        "INVALID_REQUEST",
      );
    }
    const explicitMetadataSources =
      shouldEmbedMetadata && requestedMetadata
        ? normalizeFigureSources(
            requestedMetadata.sources,
            workspaceSnapshot.artifacts,
            (relativePath) =>
              this.artifacts.fingerprint(workspace.root, relativePath, signal).digest,
          )
        : [];
    const inputArtifactIds = new Set(inputArtifacts.map((artifact) => artifact.id));
    if (
      explicitMetadataSources.some(
        (source) => source.kind === "artifact" && !inputArtifactIds.has(source.artifactId),
      )
    ) {
      throw new ScienceError(
        "Notebook metadata may reference only declared input artifacts",
        "INVALID_REQUEST",
      );
    }
    const metadataSources = uniqueNormalizedSources([
      ...inputArtifacts.map(
        (artifact): NormalizedFigureSource => ({
          kind: "artifact",
          artifactId: artifact.id,
          digest: artifact.digest,
        }),
      ),
      ...explicitMetadataSources,
    ]);
    if (shouldEmbedMetadata && metadataLibrary) {
      artifactMetadataDocument({
        code: request.source,
        environment: {},
        generationId: request.requestId,
        library: metadataLibrary,
        sources: metadataSources,
      });
    }
    const inputBytes = inputArtifacts.reduce((total, artifact) => total + artifact.size, 0);
    if (inputBytes > MAX_NOTEBOOK_INPUT_BYTES) {
      throw new ScienceError("Notebook inputs exceed the 32 MiB limit", "ARTIFACT_TOO_LARGE");
    }
    const materialized =
      inputArtifacts.length > 0
        ? this.artifacts.materializeInputs(inputArtifacts, MAX_NOTEBOOK_INPUT_BYTES, signal)
        : undefined;
    try {
      const inputEnvironment = Object.fromEntries(
        (materialized?.paths ?? []).map((path, index) => [`DSH_SCIENCE_INPUT_${index}`, path]),
      );
      const process =
        this.notebookRuntimeKind === "jupymcp"
          ? await (this.notebookRuntime as JupyMcpRuntime).execute(
              {
                inputEnvironment,
                notebookId: request.notebookId,
                source: request.source,
                workspaceKey: workspace.key,
                workspaceRoot: workspace.root,
              },
              signal,
            )
          : await (this.notebookRuntime as PythonRuntime)
              .execute(workspace.root, request.source, signal, inputEnvironment)
              .then((isolated) => ({
                ...isolated,
                outputs: [
                  ...(isolated.stdout.text.length > 0
                    ? [
                        {
                          type: "stream" as const,
                          name: "stdout" as const,
                          text: isolated.stdout.text,
                          truncated: isolated.stdout.truncated,
                        },
                      ]
                    : []),
                  ...(isolated.stderr.text.length > 0
                    ? [
                        {
                          type: "stream" as const,
                          name: "stderr" as const,
                          text: isolated.stderr.text,
                          truncated: isolated.stderr.truncated,
                        },
                      ]
                    : []),
                ],
                status:
                  isolated.outcome.exitCode === 0 && isolated.outcome.signal === null
                    ? ("succeeded" as const)
                    : ("failed" as const),
              }));
      signal?.throwIfAborted();
      for (const source of metadataSources) {
        if (source.kind !== "workspace") continue;
        const currentDigest = this.artifacts.fingerprint(
          workspace.root,
          source.relativePath,
          signal,
        ).digest;
        if (currentDigest !== source.digest) {
          throw new ScienceError(
            "Workspace source changed during Figure generation",
            "ARTIFACT_SOURCE_CHANGED",
          );
        }
      }
      const environment = redactEnvironment(process.environment);
      const metadata =
        shouldEmbedMetadata && metadataLibrary
          ? artifactMetadataDocument({
              code: request.source,
              environment,
              generationId: request.requestId,
              library: metadataLibrary,
              sources: metadataSources,
            })
          : undefined;
      const capturedArtifact = request.outputArtifact
        ? await this.artifacts.capture(
            workspace.root,
            request.outputArtifact.relativePath,
            signal,
            metadata && isArtifactMetadataMime(request.outputArtifact.mime)
              ? { metadata, mime: request.outputArtifact.mime }
              : undefined,
          )
        : undefined;
      signal?.throwIfAborted();
      return this.journal.recordNotebookExecution(workspace.key, sessionId, request, {
        capturedArtifact,
        durationMs: process.durationMs,
        environment,
        exitCode: "outcome" in process ? process.outcome.exitCode : process.exitCode,
        outputs: process.outputs,
        signal: "outcome" in process ? process.outcome.signal : process.signal,
        status: process.status,
        stderr: process.stderr,
        stdout: process.stdout,
      });
    } finally {
      materialized?.dispose();
    }
  }

  private workspace(sessionId: SessionId): { key: string; root: string } {
    const session = this.ctx.sessions.get(SessionId(sessionId));
    if (!session) {
      throw new ScienceError("Live session not found", "SESSION_NOT_FOUND");
    }
    const cwd = session.header.cwd;
    if (!cwd) {
      throw new ScienceError(
        "The live session has no workspace directory",
        "WORKSPACE_UNAVAILABLE",
      );
    }
    try {
      const canonical = realpathSync.native(cwd);
      return {
        key: createHash("sha256").update(canonical).digest("hex"),
        root: canonical,
      };
    } catch (error) {
      throw new ScienceError(
        "The live session workspace cannot be resolved",
        "WORKSPACE_UNAVAILABLE",
        { cause: error },
      );
    }
  }
}

export default ScienceService;
