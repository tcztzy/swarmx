import type { SessionId } from "@deepseek-ai/dsh-session";
import type { RemoteResult, TypertRemoteContribution } from "@deepseek-ai/dsh-typert-protocol";
import type {
  CompareRunsRequest,
  CreateDocumentRequest,
  CreateFigureRequest,
  CreateHypothesisRequest,
  CreateNotebookRequest,
  CreateProjectRequest,
  CreateQuestionRequest,
  DefineExperimentRequest,
  ExecuteNotebookCellRequest,
  ExportProjectRequest,
  FinishRunRequest,
  ImportArtifactRequest,
  LinkEvidenceRequest,
  LinkEvidenceResult,
  ModifyDocumentRequest,
  ModifyFigureCodeRequest,
  NotebookExecution,
  PreviewArtifactRequest,
  ProvenanceTrace,
  RecordClaimRequest,
  RegisterArtifactRequest,
  RunComparison,
  RunMutation,
  ScienceArtifact,
  ScienceArtifactPreview,
  ScienceDocument,
  ScienceExperiment,
  ScienceFigure,
  ScienceNotebook,
  ScienceProject,
  ScienceProjectExport,
  ScienceResearchRecord,
  ScienceRun,
  ScienceWorkspaceSnapshot,
  StartRunRequest,
  TraceProvenanceRequest,
} from "./contracts.js";
import { SCIENCE_INVOCATIONS } from "./remote-contract.js";

interface ScienceRemoteNamespace {
  compareRuns: (
    sessionId: SessionId,
    request: CompareRunsRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<RunComparison>>;
  createHypothesis: (
    sessionId: SessionId,
    request: CreateHypothesisRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceResearchRecord>>;
  createDocument: (
    sessionId: SessionId,
    request: CreateDocumentRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceDocument>>;
  createFigure: (
    sessionId: SessionId,
    request: CreateFigureRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceFigure>>;
  createNotebook: (
    sessionId: SessionId,
    request: CreateNotebookRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceNotebook>>;
  createProject: (
    sessionId: SessionId,
    request: CreateProjectRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceProject>>;
  createQuestion: (
    sessionId: SessionId,
    request: CreateQuestionRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceResearchRecord>>;
  defineExperiment: (
    sessionId: SessionId,
    request: DefineExperimentRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceExperiment>>;
  executeNotebookCell: (
    sessionId: SessionId,
    request: ExecuteNotebookCellRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<NotebookExecution>>;
  exportProject: (
    sessionId: SessionId,
    request: ExportProjectRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceProjectExport>>;
  finishRun: (
    sessionId: SessionId,
    request: FinishRunRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceRun>>;
  getWorkspace: (
    sessionId: SessionId,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceWorkspaceSnapshot>>;
  importArtifact: (
    sessionId: SessionId,
    request: ImportArtifactRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceArtifact>>;
  linkEvidence: (
    sessionId: SessionId,
    request: LinkEvidenceRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<LinkEvidenceResult>>;
  modifyDocument: (
    sessionId: SessionId,
    request: ModifyDocumentRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceDocument>>;
  modifyFigureCode: (
    sessionId: SessionId,
    request: ModifyFigureCodeRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceFigure>>;
  previewArtifact: (
    sessionId: SessionId,
    request: PreviewArtifactRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceArtifactPreview>>;
  recordClaim: (
    sessionId: SessionId,
    request: RecordClaimRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceResearchRecord>>;
  registerArtifact: (
    sessionId: SessionId,
    request: RegisterArtifactRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ScienceArtifact>>;
  startRun: (
    sessionId: SessionId,
    request: StartRunRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<RunMutation>>;
  traceProvenance: (
    sessionId: SessionId,
    request: TraceProvenanceRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<ProvenanceTrace>>;
}

declare module "@deepseek-ai/dsh-typert-protocol" {
  interface TypertRemoteMap {
    "science/compareRuns": ScienceRemoteNamespace["compareRuns"];
    "science/createHypothesis": ScienceRemoteNamespace["createHypothesis"];
    "science/createDocument": ScienceRemoteNamespace["createDocument"];
    "science/createFigure": ScienceRemoteNamespace["createFigure"];
    "science/createNotebook": ScienceRemoteNamespace["createNotebook"];
    "science/createProject": ScienceRemoteNamespace["createProject"];
    "science/createQuestion": ScienceRemoteNamespace["createQuestion"];
    "science/defineExperiment": ScienceRemoteNamespace["defineExperiment"];
    "science/executeNotebookCell": ScienceRemoteNamespace["executeNotebookCell"];
    "science/exportProject": ScienceRemoteNamespace["exportProject"];
    "science/finishRun": ScienceRemoteNamespace["finishRun"];
    "science/getWorkspace": ScienceRemoteNamespace["getWorkspace"];
    "science/importArtifact": ScienceRemoteNamespace["importArtifact"];
    "science/linkEvidence": ScienceRemoteNamespace["linkEvidence"];
    "science/modifyDocument": ScienceRemoteNamespace["modifyDocument"];
    "science/modifyFigureCode": ScienceRemoteNamespace["modifyFigureCode"];
    "science/previewArtifact": ScienceRemoteNamespace["previewArtifact"];
    "science/recordClaim": ScienceRemoteNamespace["recordClaim"];
    "science/registerArtifact": ScienceRemoteNamespace["registerArtifact"];
    "science/startRun": ScienceRemoteNamespace["startRun"];
    "science/traceProvenance": ScienceRemoteNamespace["traceProvenance"];
  }

  interface TypertRemoteNamespaceMap {
    science: ScienceRemoteNamespace;
  }
}

export const TYPERT_REMOTE: TypertRemoteContribution = Object.freeze({
  package: "@swarmx/dsh-science",
  descriptors: SCIENCE_INVOCATIONS,
});

export default TYPERT_REMOTE;
