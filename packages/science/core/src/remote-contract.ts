import type { InvocationDescriptor } from "@deepseek-ai/dsh-typert-protocol";
import { z } from "zod";
import {
  compareRunsRequestSchema,
  createDocumentRequestSchema,
  createFigureRequestSchema,
  createHypothesisRequestSchema,
  createNotebookRequestSchema,
  createProjectRequestSchema,
  createQuestionRequestSchema,
  defineExperimentRequestSchema,
  executeNotebookCellRequestSchema,
  exportProjectRequestSchema,
  finishRunRequestSchema,
  importArtifactRequestSchema,
  linkEvidenceRequestSchema,
  linkEvidenceResultSchema,
  modifyDocumentRequestSchema,
  modifyFigureCodeRequestSchema,
  notebookExecutionSchema,
  previewArtifactRequestSchema,
  provenanceTraceSchema,
  recordClaimRequestSchema,
  registerArtifactRequestSchema,
  runComparisonSchema,
  runMutationSchema,
  scienceArtifactPreviewSchema,
  scienceArtifactSchema,
  scienceDocumentSchema,
  scienceExperimentSchema,
  scienceFigureSchema,
  scienceNotebookSchema,
  scienceProjectExportSchema,
  scienceProjectSchema,
  scienceResearchRecordSchema,
  scienceRunSchema,
  scienceWorkspaceSnapshotSchema,
  startRunRequestSchema,
  traceProvenanceRequestSchema,
} from "./contracts.js";

const sessionIdSchema = z.string().min(1).max(200);

function parameter(
  name: string,
  typeSymbol: string,
  schema: z.ZodType,
): InvocationDescriptor["parameters"][number] {
  return {
    name,
    wire: name,
    source: "json",
    codec: { mode: "strict", typeSymbol, schema },
  };
}

function descriptor(
  method: string,
  parameters: InvocationDescriptor["parameters"],
  typeSymbol: string,
  schema: z.ZodType,
): InvocationDescriptor {
  return {
    id: `@swarmx/dsh-science#science/${method}`,
    service: "science",
    namespace: "science",
    method,
    invocation: { kind: "direct" },
    parameters,
    cancellation: { parameter: "signal" },
    result: { mode: "strict", typeSymbol, schema },
  };
}

export const SCIENCE_INVOCATIONS = Object.freeze([
  descriptor(
    "compareRuns",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#CompareRunsRequest",
        compareRunsRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#RunComparison",
    runComparisonSchema,
  ),
  descriptor(
    "createHypothesis",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#CreateHypothesisRequest",
        createHypothesisRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#ScienceResearchRecord",
    scienceResearchRecordSchema,
  ),
  descriptor(
    "createDocument",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#CreateDocumentRequest",
        createDocumentRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#ScienceDocument",
    scienceDocumentSchema,
  ),
  descriptor(
    "createFigure",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#CreateFigureRequest",
        createFigureRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#ScienceFigure",
    scienceFigureSchema,
  ),
  descriptor(
    "createNotebook",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#CreateNotebookRequest",
        createNotebookRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#ScienceNotebook",
    scienceNotebookSchema,
  ),
  descriptor(
    "createProject",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#CreateProjectRequest",
        createProjectRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#ScienceProject",
    scienceProjectSchema,
  ),
  descriptor(
    "createQuestion",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#CreateQuestionRequest",
        createQuestionRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#ScienceResearchRecord",
    scienceResearchRecordSchema,
  ),
  descriptor(
    "defineExperiment",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#DefineExperimentRequest",
        defineExperimentRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#ScienceExperiment",
    scienceExperimentSchema,
  ),
  descriptor(
    "executeNotebookCell",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#ExecuteNotebookCellRequest",
        executeNotebookCellRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#NotebookExecution",
    notebookExecutionSchema,
  ),
  descriptor(
    "exportProject",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#ExportProjectRequest",
        exportProjectRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#ScienceProjectExport",
    scienceProjectExportSchema,
  ),
  descriptor(
    "finishRun",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter("request", "@swarmx/dsh-science/types#FinishRunRequest", finishRunRequestSchema),
    ],
    "@swarmx/dsh-science/types#ScienceRun",
    scienceRunSchema,
  ),
  descriptor(
    "getWorkspace",
    [parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema)],
    "@swarmx/dsh-science/types#ScienceWorkspaceSnapshot",
    scienceWorkspaceSnapshotSchema,
  ),
  descriptor(
    "importArtifact",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#ImportArtifactRequest",
        importArtifactRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#ScienceArtifact",
    scienceArtifactSchema,
  ),
  descriptor(
    "linkEvidence",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#LinkEvidenceRequest",
        linkEvidenceRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#LinkEvidenceResult",
    linkEvidenceResultSchema,
  ),
  descriptor(
    "modifyDocument",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#ModifyDocumentRequest",
        modifyDocumentRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#ScienceDocument",
    scienceDocumentSchema,
  ),
  descriptor(
    "modifyFigureCode",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#ModifyFigureCodeRequest",
        modifyFigureCodeRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#ScienceFigure",
    scienceFigureSchema,
  ),
  descriptor(
    "previewArtifact",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#PreviewArtifactRequest",
        previewArtifactRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#ScienceArtifactPreview",
    scienceArtifactPreviewSchema,
  ),
  descriptor(
    "recordClaim",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#RecordClaimRequest",
        recordClaimRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#ScienceResearchRecord",
    scienceResearchRecordSchema,
  ),
  descriptor(
    "registerArtifact",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#RegisterArtifactRequest",
        registerArtifactRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#ScienceArtifact",
    scienceArtifactSchema,
  ),
  descriptor(
    "startRun",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter("request", "@swarmx/dsh-science/types#StartRunRequest", startRunRequestSchema),
    ],
    "@swarmx/dsh-science/types#RunMutation",
    runMutationSchema,
  ),
  descriptor(
    "traceProvenance",
    [
      parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
      parameter(
        "request",
        "@swarmx/dsh-science/types#TraceProvenanceRequest",
        traceProvenanceRequestSchema,
      ),
    ],
    "@swarmx/dsh-science/types#ProvenanceTrace",
    provenanceTraceSchema,
  ),
] satisfies readonly InvocationDescriptor[]);
