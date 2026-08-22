import { createHash } from "node:crypto";
import { realpathSync } from "node:fs";
import type { Context } from "@deepseek-ai/cordis";
import { SessionId } from "@deepseek-ai/dsh-session";
import { TypertRemoteService } from "@deepseek-ai/dsh-typert-protocol";
import s from "@deepseek-ai/schemastery";
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
  type FinishRunRequest,
  finishRunRequestSchema,
  type ImportArtifactRequest,
  importArtifactRequestSchema,
  type LinkEvidenceRequest,
  type LinkEvidenceResult,
  linkEvidenceRequestSchema,
  MAX_SCIENCE_IMPORT_BYTES,
  type ModifyDocumentRequest,
  type ModifyFigureCodeRequest,
  modifyDocumentRequestSchema,
  modifyFigureCodeRequestSchema,
  type NotebookExecution,
  type PreviewArtifactRequest,
  type ProjectExportCounts,
  type ProvenanceTrace,
  previewArtifactRequestSchema,
  type RecordClaimRequest,
  type RegisterArtifactRequest,
  type RunComparison,
  type RunMutation,
  recordClaimRequestSchema,
  registerArtifactRequestSchema,
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
  type TraceProvenanceRequest,
  traceProvenanceRequestSchema,
} from "./contracts.js";
import { ScienceError } from "./errors.js";
import { ScienceJournal } from "./journal.js";
import { JupyMcpRuntime } from "./jupymcp-runtime.js";
import { PythonRuntime } from "./python-runtime.js";
import { tabularPreview } from "./tabular-preview.js";

export { ArtifactStore } from "./artifact-store.js";
export * from "./contracts.js";
export { runScienceDemo, type ScienceDemoResult } from "./demo.js";
export * from "./errors.js";
export { ScienceJournal } from "./journal.js";

export interface Config {
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

/** Workspace-scoped science journal exposed to Host and strict Client Remote callers. */
export class ScienceService extends TypertRemoteService {
  static inject = ["sessions", "subprocess"];
  static Config = s.object({
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
  });

  private readonly artifacts: ArtifactStore;
  private readonly executions = new Map<
    string,
    { readonly fingerprint: string; readonly promise: Promise<NotebookExecution> }
  >();
  private readonly journal: ScienceJournal;
  private readonly maxExportBytes: number;
  private readonly notebookRuntime: JupyMcpRuntime | PythonRuntime;
  private readonly notebookRuntimeKind: "isolated" | "jupymcp";

  constructor(ctx: Context, config: Config) {
    super(ctx, "science");
    this.journal = new ScienceJournal(config.root);
    this.artifacts = new ArtifactStore(
      config.root,
      config.maxArtifactBytes ?? DEFAULT_MAX_ARTIFACT_BYTES,
    );
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
    ctx.effect(
      () => async () => {
        try {
          await this.notebookRuntime.close();
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
      format: "dsh-science-project@1" as const,
      project,
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
    const content = `${JSON.stringify(bundle, null, 2)}\n`;
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

  registerArtifact(
    sessionId: SessionId,
    request: RegisterArtifactRequest,
    signal?: AbortSignal,
  ): ScienceArtifact {
    abortIfRequested(signal);
    const parsed = parseRequest(registerArtifactRequestSchema, request);
    const workspace = this.workspace(sessionId);
    const redacted = {
      ...parsed,
      environment: redactEnvironment(parsed.environment),
    };
    return this.journal.registerArtifact(workspace.key, sessionId, redacted, () =>
      this.artifacts.capture(workspace.root, redacted.relativePath, signal),
    );
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

  traceProvenance(
    sessionId: SessionId,
    request: TraceProvenanceRequest,
    signal?: AbortSignal,
  ): ProvenanceTrace {
    abortIfRequested(signal);
    const parsed = parseRequest(traceProvenanceRequestSchema, request);
    return this.journal.traceProvenance(this.workspace(sessionId).key, parsed);
  }

  getWorkspace(sessionId: SessionId, signal?: AbortSignal): ScienceWorkspaceSnapshot {
    abortIfRequested(signal);
    return this.journal.getWorkspace(this.workspace(sessionId).key);
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
      const capturedArtifact = request.outputArtifact
        ? this.artifacts.capture(workspace.root, request.outputArtifact.relativePath, signal)
        : undefined;
      signal?.throwIfAborted();
      return this.journal.recordNotebookExecution(workspace.key, sessionId, request, {
        capturedArtifact,
        durationMs: process.durationMs,
        environment: redactEnvironment(process.environment),
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
