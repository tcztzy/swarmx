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
  GetResearchObjectRequest,
  ImportArtifactRequest,
  LinkEvidenceRequest,
  LinkEvidenceResult,
  LiteratureSearchRequest,
  LiteratureSearchResult,
  ModifyDocumentRequest,
  ModifyFigureCodeRequest,
  NotebookExecution,
  PreviewArtifactRequest,
  PreviewTypstDocumentRequest,
  RecordClaimRequest,
  RegisterArtifactRequest,
  ResolveTypstSourceAtPointRequest,
  RoCrateMetadataDocument,
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
  StartRunRequest,
  TypstDocumentPreview,
  TypstSourceTarget,
  TypstSourceUpdate,
  UpdateTypstSourceRequest,
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
  getResearchObject: (
    sessionId: SessionId,
    request: GetResearchObjectRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<RoCrateMetadataDocument>>;
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
  previewTypstDocument: (
    sessionId: SessionId,
    request: PreviewTypstDocumentRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<TypstDocumentPreview>>;
  resolveTypstSourceAtPoint: (
    sessionId: SessionId,
    request: ResolveTypstSourceAtPointRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<TypstSourceTarget | null>>;
  searchLiterature: (
    sessionId: SessionId,
    request: LiteratureSearchRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<LiteratureSearchResult>>;
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
  updateTypstSource: (
    sessionId: SessionId,
    request: UpdateTypstSourceRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<TypstSourceUpdate>>;
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
    "science/getResearchObject": ScienceRemoteNamespace["getResearchObject"];
    "science/importArtifact": ScienceRemoteNamespace["importArtifact"];
    "science/linkEvidence": ScienceRemoteNamespace["linkEvidence"];
    "science/modifyDocument": ScienceRemoteNamespace["modifyDocument"];
    "science/modifyFigureCode": ScienceRemoteNamespace["modifyFigureCode"];
    "science/previewArtifact": ScienceRemoteNamespace["previewArtifact"];
    "science/previewTypstDocument": ScienceRemoteNamespace["previewTypstDocument"];
    "science/resolveTypstSourceAtPoint": ScienceRemoteNamespace["resolveTypstSourceAtPoint"];
    "science/searchLiterature": ScienceRemoteNamespace["searchLiterature"];
    "science/recordClaim": ScienceRemoteNamespace["recordClaim"];
    "science/registerArtifact": ScienceRemoteNamespace["registerArtifact"];
    "science/startRun": ScienceRemoteNamespace["startRun"];
    "science/updateTypstSource": ScienceRemoteNamespace["updateTypstSource"];
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
