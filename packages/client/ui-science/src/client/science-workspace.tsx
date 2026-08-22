import {
  type ExecuteNotebookCellRequest,
  type FigureLibrary,
  MAX_SCIENCE_IMPORT_BYTES,
  type ModifyDocumentRequest,
  type ModifyFigureCodeRequest,
  type NotebookExecution,
  type RunMutation,
  type ScienceArtifact,
  type ScienceDocument,
  type ScienceExperiment,
  type ScienceFigure,
  type ScienceNotebook,
  type ScienceProject,
  type ScienceProjectExport,
  type ScienceRun,
  type ScienceWorkspaceSnapshot,
  scienceImportType,
} from "@swarmx/dsh-science/types";
import {
  type DragEvent,
  type FormEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { NotebookOutputArea } from "./notebook-output.js";
import type { ScienceWorkbenchTarget } from "./science-navigation.js";
import css from "./science-workspace.module.css";

export type ScienceDestination = "notebook" | "writing" | "figures" | "research" | "experiments";

export { scienceImportType };

const DESTINATIONS: readonly { readonly id: ScienceDestination; readonly label: string }[] = [
  { id: "notebook", label: "Notebook" },
  { id: "writing", label: "Writing" },
  { id: "figures", label: "Figures" },
  { id: "research", label: "Research Map" },
  { id: "experiments", label: "Experiments" },
];

type ClientDocumentMutation = ModifyDocumentRequest extends infer Request
  ? Request extends { requestId: string }
    ? Omit<Request, "requestId">
    : never
  : never;

type ClientFigureMutation = ModifyFigureCodeRequest extends infer Request
  ? Request extends { requestId: string }
    ? Omit<Request, "requestId">
    : never
  : never;

export type NotebookOutputArtifact = NonNullable<ExecuteNotebookCellRequest["outputArtifact"]>;

export interface ScienceArtifactAnalysisPlan {
  readonly source: string;
  readonly outputArtifact: NotebookOutputArtifact;
}

export type ScienceFileImportStatus =
  | { readonly state: "idle" }
  | { readonly state: "importing"; readonly completed: number; readonly total: number }
  | { readonly state: "error"; readonly message: string };

const SCIENCE_FILE_ACCEPT = ".csv,.tsv,.json,.txt,.md,.xlsx,.pdf,.png,.jpg,.jpeg,.gif,.webp";

function artifactExtension(title: string): string {
  return title.slice(title.lastIndexOf(".")).toLocaleLowerCase();
}

function analysisBody(artifact: ScienceArtifact): readonly string[] {
  const extension = artifactExtension(artifact.title);
  if (extension === ".csv" || extension === ".tsv") {
    return [
      "import csv",
      'delimiter = "\\t" if input_path.suffix.lower() == ".tsv" else ","',
      'with input_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:',
      "    rows = []",
      "    reader = csv.reader(stream, delimiter=delimiter)",
      "    for index, row in enumerate(reader):",
      "        if index < 6:",
      "            rows.append(row)",
      "        else:",
      "            break",
      'analysis = {"format": "tabular", "columns": rows[0] if rows else [], "preview": rows[1:], "previewRowCount": max(0, len(rows) - 1)}',
    ];
  }
  if (extension === ".json") {
    return [
      'data = json.loads(input_path.read_text(encoding="utf-8"))',
      'analysis = {"format": "json", "rootType": type(data).__name__, "itemCount": len(data) if hasattr(data, "__len__") else None, "keys": list(data)[:20] if isinstance(data, dict) else []}',
    ];
  }
  if (extension === ".txt" || extension === ".md") {
    return [
      'text = input_path.read_text(encoding="utf-8", errors="replace")',
      'analysis = {"format": "text", "characters": len(text), "lines": len(text.splitlines()), "preview": text[:2000]}',
    ];
  }
  if (extension === ".xlsx") {
    return [
      "import zipfile",
      "import xml.etree.ElementTree as ET",
      "with zipfile.ZipFile(input_path) as archive:",
      '    workbook = ET.fromstring(archive.read("xl/workbook.xml"))',
      '    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}',
      '    sheets = [node.attrib.get("name", "") for node in workbook.findall("main:sheets/main:sheet", namespace)]',
      'analysis = {"format": "xlsx", "sheets": sheets, "sheetCount": len(sheets), "bytes": input_path.stat().st_size}',
    ];
  }
  if (extension === ".pdf") {
    return [
      "import re",
      "content = input_path.read_bytes()",
      'analysis = {"format": "pdf", "bytes": len(content), "pages": len(re.findall(rb"/Type\\s*/Page(?!s)", content)), "header": content[:16].decode("latin-1", errors="replace")}',
    ];
  }
  return [
    "content = input_path.read_bytes()",
    `analysis = {"format": ${JSON.stringify(extension.slice(1) || "binary")}, "bytes": len(content), "signature": content[:16].hex()}`,
  ];
}

/** Build one path-free, deterministic inspection cell for a registered artifact. */
export function scienceArtifactAnalysisPlan(
  artifact: ScienceArtifact,
): ScienceArtifactAnalysisPlan {
  const safeId = artifact.id.replace(/[^A-Za-z0-9_-]+/gu, "_").slice(0, 120) || "artifact";
  const titleWithoutExtension = artifact.title.slice(0, artifact.title.lastIndexOf("."));
  const safeTitle =
    titleWithoutExtension.replace(/[^A-Za-z0-9_-]+/gu, "-").slice(0, 140) || "artifact";
  const relativePath = `.dsh-science/analysis/${safeId}.json`;
  const source = [
    "import json, os",
    "from pathlib import Path",
    'input_path = Path(os.environ["DSH_SCIENCE_INPUT_0"])',
    ...analysisBody(artifact),
    `analysis["source"] = ${JSON.stringify(artifact.title)}`,
    `output_path = Path(${JSON.stringify(relativePath)})`,
    "output_path.parent.mkdir(parents=True, exist_ok=True)",
    'output_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")',
    "print(json.dumps(analysis, ensure_ascii=False, indent=2))",
  ].join("\n");
  return {
    source,
    outputArtifact: {
      relativePath,
      kind: "dataset",
      title: `${safeTitle}-analysis.json`,
      mime: "application/json",
      license: null,
    },
  };
}

async function fileBase64(file: File, signal?: AbortSignal): Promise<string> {
  signal?.throwIfAborted();
  if (file.size === 0 || file.size > MAX_SCIENCE_IMPORT_BYTES) {
    throw new Error(`Each imported file must contain 1–${MAX_SCIENCE_IMPORT_BYTES} bytes.`);
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  signal?.throwIfAborted();
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 32 * 1024) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 32 * 1024));
  }
  return btoa(binary);
}

const NOTEBOOK_OUTPUT_TYPES = {
  ".csv": { kind: "dataset", mime: "text/csv" },
  ".tsv": { kind: "dataset", mime: "text/tab-separated-values" },
  ".json": { kind: "dataset", mime: "application/json" },
  ".png": { kind: "figure", mime: "image/png" },
  ".jpg": { kind: "figure", mime: "image/jpeg" },
  ".jpeg": { kind: "figure", mime: "image/jpeg" },
  ".gif": { kind: "figure", mime: "image/gif" },
  ".webp": { kind: "figure", mime: "image/webp" },
  ".pdf": { kind: "pdf", mime: "application/pdf" },
  ".py": { kind: "code", mime: "text/x-python" },
  ".r": { kind: "code", mime: "text/x-r-source" },
  ".ipynb": { kind: "notebook", mime: "application/x-ipynb+json" },
  ".log": { kind: "log", mime: "text/plain" },
  ".txt": { kind: "log", mime: "text/plain" },
} as const satisfies Record<string, Pick<NotebookOutputArtifact, "kind" | "mime">>;

/** Convert a bounded workspace-relative path into Host capture metadata. */
export function notebookOutputArtifact(value: string): NotebookOutputArtifact | null | undefined {
  const relativePath = value.trim();
  if (relativePath.length === 0) return null;
  if (
    relativePath.length > 4_096 ||
    relativePath.startsWith("/") ||
    /^[a-z]:/iu.test(relativePath) ||
    relativePath.includes("\\") ||
    relativePath.includes("\0")
  ) {
    return undefined;
  }
  const segments = relativePath.split("/");
  if (segments.some((segment) => segment.length === 0 || segment === "." || segment === "..")) {
    return undefined;
  }
  const title = segments.at(-1);
  if (title === undefined || title.length > 160) return undefined;
  const extensionIndex = title.lastIndexOf(".");
  const extension = title.slice(extensionIndex).toLocaleLowerCase();
  const outputType = NOTEBOOK_OUTPUT_TYPES[extension as keyof typeof NOTEBOOK_OUTPUT_TYPES];
  if (outputType === undefined) return undefined;
  return { relativePath, title, ...outputType, license: null };
}

export interface SourceSelection {
  readonly start: number;
  readonly end: number;
}

export function textareaSelection(
  textarea: Pick<HTMLTextAreaElement, "selectionStart" | "selectionEnd">,
): SourceSelection | null {
  return textarea.selectionStart < textarea.selectionEnd
    ? { start: textarea.selectionStart, end: textarea.selectionEnd }
    : null;
}

export function toggleFigureObjectSelection(
  current: readonly string[],
  objectId: string,
  additive: boolean,
): string[] {
  if (!additive) return [objectId];
  return current.includes(objectId)
    ? current.filter((candidate) => candidate !== objectId)
    : [...current, objectId];
}

export function settleScienceMutation<T>(
  mutation: Promise<T>,
  onSuccess: (value: T) => void,
  onFailure: (error: unknown) => void,
): void {
  void mutation.then(onSuccess, onFailure);
}

export function matchesResearchSearch(
  record: Pick<
    ScienceWorkspaceSnapshot["records"][number],
    "id" | "kind" | "title" | "summary" | "tags"
  >,
  search: string,
): boolean {
  const normalized = search.trim().toLocaleLowerCase();
  if (normalized.length === 0) return true;
  if (normalized.includes(record.id.toLocaleLowerCase())) return true;
  return [record.id, record.kind, record.title, record.summary, ...record.tags].some((value) =>
    value.toLocaleLowerCase().includes(normalized),
  );
}

/** Keep Notebook selection inside the refreshed workspace with deterministic fallback. */
export function resolveActiveNotebookId(
  notebooks: readonly ScienceNotebook[],
  preferredId: string | null,
): string | null {
  if (preferredId !== null && notebooks.some((notebook) => notebook.id === preferredId)) {
    return preferredId;
  }
  return notebooks[0]?.id ?? null;
}

export type ScienceWorkspaceModel =
  | { readonly status: "loading" }
  | { readonly status: "error"; readonly message: string }
  | { readonly status: "ready"; readonly workspace: ScienceWorkspaceSnapshot };

export interface ScienceWorkspaceViewProps {
  readonly model: ScienceWorkspaceModel;
  readonly projectTitle: string;
  readonly notebookTitle: string;
  readonly cellSource: string;
  readonly outputCapturePath: string;
  readonly documentName: string;
  readonly documentContent: string;
  readonly documentSelection: SourceSelection | null;
  readonly proposalText: string;
  readonly proposalInstruction: string;
  readonly proposalReasoning: string;
  readonly figureTitle: string;
  readonly figureLibrary: FigureLibrary;
  readonly figureCode: string;
  readonly selectedFigureObjectIds: readonly string[];
  readonly figureProposalCode: string;
  readonly figureProposalInstruction: string;
  readonly figureProposalReasoning: string;
  readonly researchSearch: string;
  readonly experimentTitle: string;
  readonly experimentSummary: string;
  readonly experimentProtocol: string;
  readonly focusedArtifactId: string | null;
  readonly activeNotebookId: string | null;
  readonly activeDestination: ScienceDestination;
  readonly newNotebookOpen: boolean;
  readonly cellComposerOpen: boolean;
  readonly fileImportStatus: ScienceFileImportStatus;
  readonly onProjectTitleChange: (value: string) => void;
  readonly onNotebookTitleChange: (value: string) => void;
  readonly onActiveNotebookChange: (notebookId: string) => void;
  readonly onDestinationChange: (destination: ScienceDestination) => void;
  readonly onNewNotebookOpenChange: (open: boolean) => void;
  readonly onCellComposerOpenChange: (open: boolean) => void;
  readonly onCellSourceChange: (value: string) => void;
  readonly onOutputCapturePathChange: (value: string) => void;
  readonly onDocumentNameChange: (value: string) => void;
  readonly onDocumentContentChange: (value: string) => void;
  readonly onDocumentSelectionChange: (value: SourceSelection | null) => void;
  readonly onProposalTextChange: (value: string) => void;
  readonly onProposalInstructionChange: (value: string) => void;
  readonly onProposalReasoningChange: (value: string) => void;
  readonly onFigureTitleChange: (value: string) => void;
  readonly onFigureLibraryChange: (value: FigureLibrary) => void;
  readonly onFigureCodeChange: (value: string) => void;
  readonly onFigureObjectToggle: (objectId: string, additive: boolean) => void;
  readonly onFigureProposalCodeChange: (value: string) => void;
  readonly onFigureProposalInstructionChange: (value: string) => void;
  readonly onFigureProposalReasoningChange: (value: string) => void;
  readonly onResearchSearchChange: (value: string) => void;
  readonly onExperimentTitleChange: (value: string) => void;
  readonly onExperimentSummaryChange: (value: string) => void;
  readonly onExperimentProtocolChange: (value: string) => void;
  readonly onCreateProject: () => void;
  readonly onCreateNotebook: () => void;
  readonly onExecuteCell: () => void;
  readonly onCreateDocument: () => void;
  readonly onProposeDocumentPatch: () => void;
  readonly onResolveDocumentPatch: (proposalId: string, action: "accept" | "reject") => void;
  readonly onCreateFigure: () => void;
  readonly onProposeFigurePatch: () => void;
  readonly onResolveFigurePatch: (proposalId: string, action: "accept" | "reject") => void;
  readonly onDefineExperiment: () => void;
  readonly onStartRun: (experimentId: string, expectedRevision: number) => void;
  readonly onFinishRun: (runId: string, expectedRevision: number) => void;
  readonly onExportProject: () => void;
  readonly onImportFiles: (files: readonly File[]) => void;
  readonly onAnalyzeArtifact: (artifact: ScienceArtifact) => void;
  readonly onOpenArtifact: (artifact: ScienceArtifact) => void;
  readonly onRetry: () => void;
}

export interface ScienceWorkspaceInjected {
  readonly loadWorkspace: (signal?: AbortSignal) => Promise<ScienceWorkspaceSnapshot>;
  readonly createProject: (title: string, signal?: AbortSignal) => Promise<ScienceProject>;
  readonly createNotebook: (
    projectId: string,
    title: string,
    signal?: AbortSignal,
  ) => Promise<ScienceNotebook>;
  readonly executeCell: (
    notebookId: string,
    source: string,
    outputArtifact: NotebookOutputArtifact | null,
    inputArtifactIds?: readonly string[],
    signal?: AbortSignal,
  ) => Promise<NotebookExecution>;
  readonly importArtifact: (
    projectId: string,
    name: string,
    dataBase64: string,
    signal?: AbortSignal,
  ) => Promise<ScienceArtifact>;
  readonly createDocument: (
    projectId: string,
    name: string,
    content: string,
    signal?: AbortSignal,
  ) => Promise<ScienceDocument>;
  readonly modifyDocument: (
    request: ClientDocumentMutation,
    signal?: AbortSignal,
  ) => Promise<ScienceDocument>;
  readonly createFigure: (
    projectId: string,
    title: string,
    library: FigureLibrary,
    code: string,
    signal?: AbortSignal,
  ) => Promise<ScienceFigure>;
  readonly modifyFigureCode: (
    request: ClientFigureMutation,
    signal?: AbortSignal,
  ) => Promise<ScienceFigure>;
  readonly defineExperiment: (
    projectId: string,
    title: string,
    summary: string,
    protocol: string,
    signal?: AbortSignal,
  ) => Promise<ScienceExperiment>;
  readonly startRun: (
    experimentId: string,
    expectedRevision: number,
    signal?: AbortSignal,
  ) => Promise<RunMutation>;
  readonly finishRun: (
    runId: string,
    expectedRevision: number,
    signal?: AbortSignal,
  ) => Promise<ScienceRun>;
  readonly exportProject: (
    projectId: string,
    signal?: AbortSignal,
  ) => Promise<ScienceProjectExport>;
  readonly openArtifact: (artifact: ScienceArtifact) => void;
  readonly navigation: {
    readonly getSnapshot: () => ScienceWorkbenchTarget | null;
    readonly subscribe: (listener: () => void) => () => void;
    readonly mount: () => () => void;
  };
}

function submit(action: () => void) {
  return (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    action();
  };
}

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KiB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MiB`;
}

export function scienceArtifactDomId(artifactId: string): string {
  return `science-artifact-${encodeURIComponent(artifactId)}`;
}

/** Pure presentation boundary covering loading, empty, error, and populated states. */
export function ScienceWorkspaceView({
  model,
  projectTitle,
  notebookTitle,
  cellSource,
  outputCapturePath,
  documentName,
  documentContent,
  documentSelection,
  proposalText,
  proposalInstruction,
  proposalReasoning,
  figureTitle,
  figureLibrary,
  figureCode,
  selectedFigureObjectIds,
  figureProposalCode,
  figureProposalInstruction,
  figureProposalReasoning,
  researchSearch,
  experimentTitle,
  experimentSummary,
  experimentProtocol,
  focusedArtifactId,
  activeNotebookId,
  activeDestination,
  newNotebookOpen,
  cellComposerOpen,
  fileImportStatus,
  onProjectTitleChange,
  onNotebookTitleChange,
  onActiveNotebookChange,
  onDestinationChange,
  onNewNotebookOpenChange,
  onCellComposerOpenChange,
  onCellSourceChange,
  onOutputCapturePathChange,
  onDocumentNameChange,
  onDocumentContentChange,
  onDocumentSelectionChange,
  onProposalTextChange,
  onProposalInstructionChange,
  onProposalReasoningChange,
  onFigureTitleChange,
  onFigureLibraryChange,
  onFigureCodeChange,
  onFigureObjectToggle,
  onFigureProposalCodeChange,
  onFigureProposalInstructionChange,
  onFigureProposalReasoningChange,
  onResearchSearchChange,
  onExperimentTitleChange,
  onExperimentSummaryChange,
  onExperimentProtocolChange,
  onCreateProject,
  onCreateNotebook,
  onExecuteCell,
  onCreateDocument,
  onProposeDocumentPatch,
  onResolveDocumentPatch,
  onCreateFigure,
  onProposeFigurePatch,
  onResolveFigurePatch,
  onDefineExperiment,
  onStartRun,
  onFinishRun,
  onExportProject,
  onImportFiles,
  onAnalyzeArtifact,
  onOpenArtifact,
  onRetry,
}: ScienceWorkspaceViewProps) {
  const workspace = model.status === "ready" ? model.workspace : undefined;
  const firstProject = workspace?.projects[0];
  const firstDocument = workspace?.documents[0];
  const firstFigure = workspace?.figures[0];
  const resolvedNotebookId = resolveActiveNotebookId(workspace?.notebooks ?? [], activeNotebookId);
  const activeNotebook = workspace?.notebooks.find(
    (notebook) => notebook.id === resolvedNotebookId,
  );
  const visibleRecords =
    workspace?.records.filter((record) => matchesResearchSearch(record, researchSearch)) ?? [];
  const outputArtifact = notebookOutputArtifact(outputCapturePath);
  const notebookCreationVisible = newNotebookOpen || workspace?.notebooks.length === 0;
  const cellComposerVisible = cellComposerOpen || activeNotebook?.cells.length === 0;
  const activeDestinationLabel =
    DESTINATIONS.find((destination) => destination.id === activeDestination)?.label ?? "Notebook";

  return (
    <main
      className={css.workspace}
      data-science-shell="true"
      aria-labelledby="science-project-heading"
    >
      <div className={css.shell}>
        <aside className={css.projectRail}>
          <header className={css.projectHeader}>
            <span>Science project</span>
            <strong id="science-project-heading">
              {firstProject?.title ?? "Untitled project"}
            </strong>
          </header>

          <button
            type="button"
            className={css.newAction}
            aria-expanded={notebookCreationVisible}
            aria-controls="science-new-notebook"
            onClick={() => {
              onDestinationChange("notebook");
              onNewNotebookOpenChange(!notebookCreationVisible);
            }}
          >
            New
          </button>

          <nav className={css.projectNavigation} aria-label="Science project navigation">
            <p>Workspaces</p>
            <ul>
              {DESTINATIONS.map((destination) => (
                <li key={destination.id}>
                  <button
                    type="button"
                    data-destination={destination.id}
                    aria-current={destination.id === activeDestination ? "page" : undefined}
                    onClick={() => onDestinationChange(destination.id)}
                  >
                    <span>{destination.label}</span>
                  </button>
                </li>
              ))}
            </ul>
          </nav>

          <section className={css.fileRail} aria-labelledby="science-files-title">
            <h2 id="science-files-title">
              <span>Files</span>
              <small>{workspace?.artifacts.length ?? 0}</small>
            </h2>
            {firstProject && (
              <div
                className={css.fileImport}
                data-importing={fileImportStatus.state === "importing" || undefined}
                onDragOver={(event: DragEvent<HTMLDivElement>) => event.preventDefault()}
                onDrop={(event: DragEvent<HTMLDivElement>) => {
                  event.preventDefault();
                  onImportFiles(Array.from(event.dataTransfer.files));
                }}
              >
                <input
                  id="science-file-import"
                  type="file"
                  multiple
                  accept={SCIENCE_FILE_ACCEPT}
                  aria-label="Import research files"
                  disabled={fileImportStatus.state === "importing"}
                  onChange={(event) => {
                    onImportFiles(Array.from(event.currentTarget.files ?? []));
                    event.currentTarget.value = "";
                  }}
                />
                <label htmlFor="science-file-import">Import files</label>
                <span>Drop files here</span>
                {fileImportStatus.state === "importing" && (
                  <small role="status" aria-live="polite">
                    Importing {fileImportStatus.completed + 1} of {fileImportStatus.total}…
                  </small>
                )}
                {fileImportStatus.state === "error" && (
                  <small role="alert">{fileImportStatus.message}</small>
                )}
              </div>
            )}
            {workspace !== undefined && workspace.artifacts.length > 0 ? (
              <ul>
                {workspace.artifacts.map((artifact) => (
                  <li
                    key={artifact.id}
                    id={scienceArtifactDomId(artifact.id)}
                    tabIndex={-1}
                    data-focused={artifact.id === focusedArtifactId || undefined}
                  >
                    <div className={css.fileActions}>
                      <button
                        type="button"
                        aria-label={`Open artifact details: ${artifact.title}`}
                        onClick={() => onOpenArtifact(artifact)}
                      >
                        <span data-artifact-summary="true">
                          <strong>{artifact.title}</strong>
                          <small>
                            {artifact.kind} · {formatBytes(artifact.size)} · Journal #
                            {artifact.provenance.journalSeq}
                          </small>
                        </span>
                      </button>
                      <button
                        type="button"
                        className={css.analyzeFile}
                        aria-label={`Analyze ${artifact.title} in Notebook`}
                        disabled={model.status !== "ready"}
                        onClick={() => onAnalyzeArtifact(artifact)}
                      >
                        Analyze
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p>Import data or generate a file from Notebook.</p>
            )}
          </section>

          <footer className={css.localStatus}>
            <span>Local workspace</span>
            <small>
              {firstProject ? `Journal #${firstProject.provenance.journalSeq}` : "Private"}
            </small>
          </footer>
        </aside>

        <section className={css.stage} aria-label={`${activeDestinationLabel} workbench`}>
          <header className={css.stageHeader}>
            <div>
              <span>{firstProject?.title ?? "Science"}</span>
              <strong>{activeDestinationLabel}</strong>
            </div>
            <small>{model.status === "ready" ? "Saved locally" : "Science journal"}</small>
          </header>

          <div className={css.stageBody}>
            {model.status === "loading" && (
              <section className={css.notice} role="status" aria-live="polite">
                Loading science journal…
              </section>
            )}

            {model.status === "error" && (
              <section className={css.notice} role="alert">
                <p>{model.message}</p>
                <button type="button" onClick={onRetry}>
                  Retry
                </button>
              </section>
            )}

            {workspace !== undefined && workspace.projects.length === 0 && (
              <section className={css.emptyProject} aria-labelledby="science-empty-title">
                <p className={css.eyebrow}>Start a research workspace</p>
                <h2 id="science-empty-title">No research project yet</h2>
                <p>
                  Create the durable project that will own notebooks and future scientific
                  artifacts.
                </p>
                <form onSubmit={submit(onCreateProject)}>
                  <label htmlFor="science-project-title">Project title</label>
                  <div className={css.formRow}>
                    <input
                      id="science-project-title"
                      value={projectTitle}
                      maxLength={160}
                      required
                      onChange={(event) => onProjectTitleChange(event.currentTarget.value)}
                    />
                    <button type="submit" disabled={projectTitle.trim().length === 0}>
                      Create project
                    </button>
                  </div>
                </form>
              </section>
            )}

            {workspace !== undefined && firstProject !== undefined && (
              <div className={css.columns}>
                {activeDestination === "notebook" && (
                  <section
                    className={`${css.panel} ${css.notebookWorkspace}`}
                    aria-labelledby="science-notebooks-title"
                  >
                    <header className={css.notebookHeader}>
                      <div>
                        <p className={css.eyebrow}>Research activity</p>
                        <h2 id="science-notebooks-title">{activeNotebook?.title ?? "Notebooks"}</h2>
                      </div>
                      {activeNotebook && (
                        <span>Journal #{activeNotebook.provenance.journalSeq}</span>
                      )}
                    </header>

                    {workspace.notebooks.length === 0 ? (
                      <p>No notebook yet. Start the first reproducible workspace.</p>
                    ) : (
                      <ul className={css.notebookTabs} aria-label="Notebooks">
                        {workspace.notebooks.map((notebook) => (
                          <li key={notebook.id}>
                            <button
                              type="button"
                              data-notebook-id={notebook.id}
                              aria-pressed={notebook.id === resolvedNotebookId}
                              onClick={() => onActiveNotebookChange(notebook.id)}
                            >
                              {notebook.title}
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}

                    {notebookCreationVisible && (
                      <form
                        id="science-new-notebook"
                        className={css.newNotebookForm}
                        onSubmit={submit(onCreateNotebook)}
                      >
                        <label htmlFor="science-notebook-title">New notebook</label>
                        <div className={css.formRow}>
                          <input
                            id="science-notebook-title"
                            value={notebookTitle}
                            maxLength={160}
                            placeholder="Notebook title"
                            required
                            onChange={(event) => onNotebookTitleChange(event.currentTarget.value)}
                          />
                          <button type="submit" disabled={notebookTitle.trim().length === 0}>
                            Create
                          </button>
                          {workspace.notebooks.length > 0 && (
                            <button
                              type="button"
                              className={css.secondaryAction}
                              onClick={() => onNewNotebookOpenChange(false)}
                            >
                              Cancel
                            </button>
                          )}
                        </div>
                      </form>
                    )}

                    {activeNotebook !== undefined && activeNotebook.cells.length > 0 ? (
                      <ol className={css.cells} aria-label="Executed notebook cells">
                        {activeNotebook.cells.map((cell) => (
                          <li key={cell.id} data-kind={cell.kind}>
                            <span>
                              {cell.kind} · execution {cell.executionCount ?? "—"}
                            </span>
                            {cell.kind === "output" && cell.outputs.length > 0 ? (
                              <NotebookOutputArea outputs={cell.outputs} />
                            ) : (
                              <pre>{cell.source}</pre>
                            )}
                          </li>
                        ))}
                      </ol>
                    ) : (
                      workspace.notebooks.length > 0 && (
                        <div className={css.emptyActivity}>
                          <strong>Ready for the first run</strong>
                          <p>
                            Write Python below. Every execution remains linked to its artifacts.
                          </p>
                        </div>
                      )
                    )}

                    {workspace.notebooks.length > 0 && (
                      <>
                        <div className={css.cellAction}>
                          <button
                            type="button"
                            aria-expanded={cellComposerVisible}
                            aria-controls="science-cell-composer"
                            onClick={() => onCellComposerOpenChange(!cellComposerVisible)}
                          >
                            {cellComposerVisible ? "Close Python cell" : "New Python cell"}
                          </button>
                        </div>
                        {cellComposerVisible && (
                          <form
                            id="science-cell-composer"
                            className={css.cellComposer}
                            onSubmit={submit(onExecuteCell)}
                          >
                            <label htmlFor="science-cell-source">Python cell</label>
                            <textarea
                              id="science-cell-source"
                              value={cellSource}
                              maxLength={100_000}
                              required
                              rows={5}
                              spellCheck={false}
                              placeholder="Ask the notebook to compute, analyze, or generate a file…"
                              onChange={(event) => onCellSourceChange(event.currentTarget.value)}
                            />
                            <div className={css.captureRow}>
                              <div>
                                <label htmlFor="science-output-artifact-path">
                                  Capture file (optional)
                                </label>
                                <input
                                  id="science-output-artifact-path"
                                  value={outputCapturePath}
                                  maxLength={4_096}
                                  placeholder="results.csv"
                                  spellCheck={false}
                                  aria-invalid={outputArtifact === undefined || undefined}
                                  aria-describedby="science-output-artifact-help"
                                  onChange={(event) =>
                                    onOutputCapturePathChange(event.currentTarget.value)
                                  }
                                />
                              </div>
                              <button
                                type="submit"
                                disabled={
                                  cellSource.trim().length === 0 || outputArtifact === undefined
                                }
                              >
                                Run Python
                              </button>
                            </div>
                            <p
                              id="science-output-artifact-help"
                              className={css.validationStatus}
                              role={outputArtifact === undefined ? "alert" : undefined}
                            >
                              {outputArtifact === null
                                ? "Leave empty for stdout-only execution."
                                : outputArtifact === undefined
                                  ? "Use a safe relative CSV, JSON, image, PDF, notebook, code, or log path."
                                  : `Will capture as ${outputArtifact.kind} · ${outputArtifact.mime}`}
                            </p>
                          </form>
                        )}
                      </>
                    )}
                  </section>
                )}

                {activeDestination === "writing" && (
                  <section
                    className={`${css.panel} ${css.writingPanel}`}
                    aria-labelledby="science-writing-title"
                  >
                    <h2 id="science-writing-title">Writing Studio</h2>
                    {firstDocument === undefined ? (
                      <form onSubmit={submit(onCreateDocument)}>
                        <p>Create a local Typst, LaTeX, Markdown, or BibTeX source document.</p>
                        <label htmlFor="science-document-name">Document name</label>
                        <input
                          id="science-document-name"
                          value={documentName}
                          maxLength={240}
                          required
                          placeholder="paper.typ"
                          onChange={(event) => onDocumentNameChange(event.currentTarget.value)}
                        />
                        <label htmlFor="science-document-source">Source</label>
                        <textarea
                          id="science-document-source"
                          value={documentContent}
                          maxLength={500_000}
                          rows={12}
                          spellCheck={false}
                          onChange={(event) => onDocumentContentChange(event.currentTarget.value)}
                        />
                        <button type="submit" disabled={documentName.trim().length === 0}>
                          Create document
                        </button>
                      </form>
                    ) : (
                      <div className={css.writingGrid}>
                        <div>
                          <p className={css.documentMeta}>
                            <strong>{firstDocument.name}</strong>
                            <span>
                              {firstDocument.format} · revision {firstDocument.revision} · Journal #
                              {firstDocument.provenance.journalSeq}
                            </span>
                          </p>
                          <label htmlFor="science-document-source">Source selection</label>
                          <textarea
                            id="science-document-source"
                            value={documentContent}
                            readOnly
                            rows={16}
                            spellCheck={false}
                            onSelect={(event) =>
                              onDocumentSelectionChange(textareaSelection(event.currentTarget))
                            }
                          />
                          <p className={css.validationStatus}>
                            Structure checked · Compilation not run
                          </p>
                          <p aria-live="polite">
                            {documentSelection
                              ? `Selected UTF-16 range ${documentSelection.start}–${documentSelection.end}`
                              : "Select source text to prepare an AI patch."}
                          </p>
                          <form onSubmit={submit(onProposeDocumentPatch)}>
                            <label htmlFor="science-proposal-text">Proposed replacement</label>
                            <textarea
                              id="science-proposal-text"
                              value={proposalText}
                              maxLength={500_000}
                              required
                              rows={5}
                              onChange={(event) => onProposalTextChange(event.currentTarget.value)}
                            />
                            <label htmlFor="science-proposal-instruction">Instruction</label>
                            <input
                              id="science-proposal-instruction"
                              value={proposalInstruction}
                              maxLength={2_000}
                              required
                              onChange={(event) =>
                                onProposalInstructionChange(event.currentTarget.value)
                              }
                            />
                            <label htmlFor="science-proposal-reasoning">Reasoning summary</label>
                            <textarea
                              id="science-proposal-reasoning"
                              value={proposalReasoning}
                              maxLength={4_000}
                              required
                              rows={3}
                              onChange={(event) =>
                                onProposalReasoningChange(event.currentTarget.value)
                              }
                            />
                            <button
                              type="submit"
                              disabled={
                                documentSelection === null ||
                                proposalText.length === 0 ||
                                proposalInstruction.trim().length === 0 ||
                                proposalReasoning.trim().length === 0
                              }
                            >
                              Propose change
                            </button>
                          </form>
                        </div>
                        <aside aria-label="Writing review">
                          <h3>Checks</h3>
                          {firstDocument.diagnostics.length === 0 ? (
                            <p>No structural or scientific warnings.</p>
                          ) : (
                            <ul className={css.checks}>
                              {firstDocument.diagnostics.map((diagnostic) => (
                                <li
                                  key={`${diagnostic.code}:${diagnostic.start}:${diagnostic.end}`}
                                >
                                  <strong>{diagnostic.scope}</strong>
                                  <span>{diagnostic.message}</span>
                                </li>
                              ))}
                            </ul>
                          )}
                          <h3>Patch proposals</h3>
                          {firstDocument.proposals.length === 0 ? (
                            <p>No proposal yet.</p>
                          ) : (
                            <ul className={css.proposals}>
                              {firstDocument.proposals.map((proposal) => (
                                <li key={proposal.id}>
                                  <span>{proposal.status}</span>
                                  <del>{proposal.originalText}</del>
                                  <ins>{proposal.proposedText}</ins>
                                  <p>{proposal.reasoning.summary}</p>
                                  {proposal.status === "pending" && (
                                    <div className={css.proposalActions}>
                                      <button
                                        type="button"
                                        onClick={() =>
                                          onResolveDocumentPatch(proposal.id, "accept")
                                        }
                                      >
                                        Accept
                                      </button>
                                      <button
                                        type="button"
                                        onClick={() =>
                                          onResolveDocumentPatch(proposal.id, "reject")
                                        }
                                      >
                                        Reject
                                      </button>
                                    </div>
                                  )}
                                </li>
                              ))}
                            </ul>
                          )}
                        </aside>
                      </div>
                    )}
                  </section>
                )}

                {activeDestination === "figures" && (
                  <section
                    className={`${css.panel} ${css.writingPanel}`}
                    aria-labelledby="science-figure-title"
                  >
                    <h2 id="science-figure-title">Figure Studio</h2>
                    {firstFigure === undefined ? (
                      <form onSubmit={submit(onCreateFigure)}>
                        <p>Create a code-linked semantic figure for local scientific editing.</p>
                        <label htmlFor="science-figure-name">Figure title</label>
                        <input
                          id="science-figure-name"
                          value={figureTitle}
                          maxLength={160}
                          required
                          onChange={(event) => onFigureTitleChange(event.currentTarget.value)}
                        />
                        <label htmlFor="science-figure-library">Plotting library</label>
                        <select
                          id="science-figure-library"
                          value={figureLibrary}
                          onChange={(event) =>
                            onFigureLibraryChange(event.currentTarget.value as FigureLibrary)
                          }
                        >
                          <option value="matplotlib">matplotlib</option>
                          <option value="seaborn">seaborn</option>
                          <option value="ggplot2">ggplot2</option>
                          <option value="plotly">plotly</option>
                        </select>
                        <label htmlFor="science-figure-code">Figure code</label>
                        <textarea
                          id="science-figure-code"
                          value={figureCode}
                          maxLength={200_000}
                          required
                          rows={10}
                          spellCheck={false}
                          onChange={(event) => onFigureCodeChange(event.currentTarget.value)}
                        />
                        <button
                          type="submit"
                          disabled={
                            figureTitle.trim().length === 0 || figureCode.trim().length === 0
                          }
                        >
                          Create figure
                        </button>
                      </form>
                    ) : (
                      <div className={css.writingGrid}>
                        <div>
                          <p className={css.documentMeta}>
                            <strong>{firstFigure.title}</strong>
                            <span>
                              {firstFigure.library} · revision {firstFigure.revision} · Journal #
                              {firstFigure.provenance.journalSeq}
                            </span>
                          </p>
                          <div
                            className={css.figureCanvas}
                            role="img"
                            aria-label="Semantic figure canvas"
                          >
                            {firstFigure.objects.map((object) => (
                              <button
                                key={object.id}
                                type="button"
                                aria-pressed={selectedFigureObjectIds.includes(object.id)}
                                onClick={(event) =>
                                  onFigureObjectToggle(
                                    object.id,
                                    event.shiftKey || event.metaKey || event.ctrlKey,
                                  )
                                }
                              >
                                <strong>{object.label}</strong>
                                <span>{object.kind}</span>
                              </button>
                            ))}
                          </div>
                          <p>
                            Click one object, or Shift-click to brush-select multiple linked
                            objects.
                          </p>
                          <label htmlFor="science-figure-code">Figure code</label>
                          <textarea
                            id="science-figure-code"
                            value={figureCode}
                            readOnly
                            rows={12}
                            spellCheck={false}
                          />
                          <form onSubmit={submit(onProposeFigurePatch)}>
                            <label htmlFor="science-figure-proposal-code">Proposed code</label>
                            <textarea
                              id="science-figure-proposal-code"
                              value={figureProposalCode}
                              maxLength={200_000}
                              required
                              rows={5}
                              onChange={(event) =>
                                onFigureProposalCodeChange(event.currentTarget.value)
                              }
                            />
                            <label htmlFor="science-figure-proposal-instruction">Instruction</label>
                            <input
                              id="science-figure-proposal-instruction"
                              value={figureProposalInstruction}
                              maxLength={2_000}
                              required
                              onChange={(event) =>
                                onFigureProposalInstructionChange(event.currentTarget.value)
                              }
                            />
                            <label htmlFor="science-figure-proposal-reasoning">
                              Reasoning summary
                            </label>
                            <textarea
                              id="science-figure-proposal-reasoning"
                              value={figureProposalReasoning}
                              maxLength={4_000}
                              required
                              rows={3}
                              onChange={(event) =>
                                onFigureProposalReasoningChange(event.currentTarget.value)
                              }
                            />
                            <button
                              type="submit"
                              disabled={
                                selectedFigureObjectIds.length === 0 ||
                                figureProposalCode.trim().length === 0 ||
                                figureProposalInstruction.trim().length === 0 ||
                                figureProposalReasoning.trim().length === 0
                              }
                            >
                              Propose figure patch
                            </button>
                          </form>
                        </div>
                        <aside aria-label="Figure review">
                          <h3>Code patches</h3>
                          {firstFigure.proposals.length === 0 ? (
                            <p>No figure patch yet.</p>
                          ) : (
                            <ul className={css.proposals}>
                              {firstFigure.proposals.map((proposal) => (
                                <li key={proposal.id}>
                                  <span>{proposal.status}</span>
                                  <del>{proposal.originalCode}</del>
                                  <ins>{proposal.proposedCode}</ins>
                                  <p>{proposal.reasoning.summary}</p>
                                  {proposal.status === "pending" && (
                                    <div className={css.proposalActions}>
                                      <button
                                        type="button"
                                        onClick={() => onResolveFigurePatch(proposal.id, "accept")}
                                      >
                                        Accept
                                      </button>
                                      <button
                                        type="button"
                                        onClick={() => onResolveFigurePatch(proposal.id, "reject")}
                                      >
                                        Reject
                                      </button>
                                    </div>
                                  )}
                                </li>
                              ))}
                            </ul>
                          )}
                        </aside>
                      </div>
                    )}
                  </section>
                )}

                {activeDestination === "research" && (
                  <section
                    className={`${css.panel} ${css.writingPanel}`}
                    aria-labelledby="science-research-map-title"
                  >
                    <h2 id="science-research-map-title">Research Map</h2>
                    <label htmlFor="science-research-search">
                      Search or paste an entity locator
                    </label>
                    <input
                      id="science-research-search"
                      type="search"
                      value={researchSearch}
                      placeholder="Question, claim, evidence, or entity id"
                      onChange={(event) => onResearchSearchChange(event.currentTarget.value)}
                    />
                    {workspace.records.length === 0 ? (
                      <p>
                        No research facts yet. Science agent tools can record the first question.
                      </p>
                    ) : visibleRecords.length === 0 ? (
                      <p role="status">No Research Map entity matches this locator.</p>
                    ) : (
                      <ul className={css.mapGrid} aria-label="Research entities">
                        {visibleRecords.map((record) => (
                          <li key={record.id} data-kind={record.kind}>
                            <span>{record.kind}</span>
                            <strong>{record.title}</strong>
                            <p>{record.summary}</p>
                            <small>
                              {record.status} · Journal #{record.provenance.journalSeq} ·{" "}
                              {record.id}
                            </small>
                          </li>
                        ))}
                      </ul>
                    )}
                    {workspace.relations.length > 0 && (
                      <ul className={css.relationList} aria-label="Research relations">
                        {workspace.relations.map((relation) => (
                          <li key={relation.id}>
                            <code>{relation.fromId}</code>
                            <strong>{relation.type}</strong>
                            <code>{relation.toId}</code>
                          </li>
                        ))}
                      </ul>
                    )}
                  </section>
                )}

                {activeDestination === "experiments" && (
                  <section
                    className={`${css.panel} ${css.writingPanel}`}
                    aria-labelledby="science-experiments-title"
                  >
                    <div className={css.sectionHeading}>
                      <div>
                        <h2 id="science-experiments-title">Experiments</h2>
                        <p>
                          Define reproducible work, track Runs, then export the complete project.
                        </p>
                      </div>
                      <button type="button" onClick={onExportProject}>
                        Export project JSON
                      </button>
                    </div>
                    {workspace.exports.length > 0 && (
                      <p role="status">
                        Latest export {workspace.exports.at(-1)?.digest} ·{" "}
                        {workspace.exports.at(-1)?.bytes} B
                      </p>
                    )}
                    {workspace.experiments.length > 0 && (
                      <ul className={css.experimentList} aria-label="Experiment ledger">
                        {workspace.experiments.map((experiment) => (
                          <li key={experiment.id}>
                            <div>
                              <strong>{experiment.title}</strong>
                              <span>
                                {experiment.status} · {experiment.runIds.length} Runs · revision{" "}
                                {experiment.revision}
                              </span>
                            </div>
                            <button
                              type="button"
                              onClick={() => onStartRun(experiment.id, experiment.revision)}
                            >
                              Start Run
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                    {workspace.runs.length > 0 && (
                      <ul className={css.experimentList} aria-label="Experiment runs">
                        {workspace.runs.map((run) => (
                          <li key={run.id}>
                            <div>
                              <strong>Run {run.id.slice(0, 8)}</strong>
                              <span>
                                {run.status} · revision {run.revision}
                              </span>
                            </div>
                            {run.status === "running" && (
                              <button
                                type="button"
                                onClick={() => onFinishRun(run.id, run.revision)}
                              >
                                Finish succeeded
                              </button>
                            )}
                          </li>
                        ))}
                      </ul>
                    )}
                    <form onSubmit={submit(onDefineExperiment)}>
                      <label htmlFor="science-experiment-title">Experiment title</label>
                      <input
                        id="science-experiment-title"
                        value={experimentTitle}
                        maxLength={160}
                        required
                        onChange={(event) => onExperimentTitleChange(event.currentTarget.value)}
                      />
                      <label htmlFor="science-experiment-summary">Summary</label>
                      <textarea
                        id="science-experiment-summary"
                        value={experimentSummary}
                        maxLength={4_000}
                        required
                        rows={3}
                        onChange={(event) => onExperimentSummaryChange(event.currentTarget.value)}
                      />
                      <label htmlFor="science-experiment-protocol">Protocol</label>
                      <textarea
                        id="science-experiment-protocol"
                        value={experimentProtocol}
                        maxLength={100_000}
                        required
                        rows={5}
                        spellCheck={false}
                        onChange={(event) => onExperimentProtocolChange(event.currentTarget.value)}
                      />
                      <button
                        type="submit"
                        disabled={
                          experimentTitle.trim().length === 0 ||
                          experimentSummary.trim().length === 0 ||
                          experimentProtocol.trim().length === 0
                        }
                      >
                        Define experiment
                      </button>
                    </form>
                  </section>
                )}
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Science service unavailable";
}

function downloadProjectExport(exported: ScienceProjectExport): void {
  const url = URL.createObjectURL(
    new Blob([exported.content], { type: "application/json;charset=utf-8" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `dsh-science-${exported.projectId}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

/** Live Science Workspace bound to one conversation session by the slot registry. */
export function ScienceWorkspace({
  loadWorkspace,
  createProject,
  createNotebook,
  executeCell,
  importArtifact,
  createDocument,
  modifyDocument,
  createFigure,
  modifyFigureCode,
  defineExperiment,
  startRun,
  finishRun,
  exportProject,
  openArtifact,
  navigation,
}: ScienceWorkspaceInjected) {
  const [model, setModel] = useState<ScienceWorkspaceModel>({ status: "loading" });
  const [projectTitle, setProjectTitle] = useState("");
  const [notebookTitle, setNotebookTitle] = useState("");
  const [activeNotebookId, setActiveNotebookId] = useState<string | null>(null);
  const [activeDestination, setActiveDestination] = useState<ScienceDestination>("notebook");
  const [newNotebookOpen, setNewNotebookOpen] = useState(false);
  const [cellComposerOpen, setCellComposerOpen] = useState(false);
  const [cellSource, setCellSource] = useState("");
  const [outputCapturePath, setOutputCapturePath] = useState("");
  const [fileImportStatus, setFileImportStatus] = useState<ScienceFileImportStatus>({
    state: "idle",
  });
  const [documentName, setDocumentName] = useState("paper.typ");
  const [documentContent, setDocumentContent] = useState("");
  const [documentSelection, setDocumentSelection] = useState<SourceSelection | null>(null);
  const [proposalText, setProposalText] = useState("");
  const [proposalInstruction, setProposalInstruction] = useState("");
  const [proposalReasoning, setProposalReasoning] = useState("");
  const [figureTitle, setFigureTitle] = useState("");
  const [figureLibrary, setFigureLibrary] = useState<FigureLibrary>("matplotlib");
  const [figureCode, setFigureCode] = useState("");
  const [selectedFigureObjectIds, setSelectedFigureObjectIds] = useState<string[]>([]);
  const [figureProposalCode, setFigureProposalCode] = useState("");
  const [figureProposalInstruction, setFigureProposalInstruction] = useState("");
  const [figureProposalReasoning, setFigureProposalReasoning] = useState("");
  const [researchSearch, setResearchSearch] = useState("");
  const [experimentTitle, setExperimentTitle] = useState("");
  const [experimentSummary, setExperimentSummary] = useState("");
  const [experimentProtocol, setExperimentProtocol] = useState("");
  const activeExecution = useRef<AbortController>();
  const activeFileImport = useRef<AbortController>();
  const workbenchTarget = useSyncExternalStore(
    navigation.subscribe,
    navigation.getSnapshot,
    navigation.getSnapshot,
  );

  useEffect(() => navigation.mount(), [navigation]);

  useEffect(() => {
    if (model.status !== "ready" || workbenchTarget?.kind !== "artifact") return;
    const element = document.getElementById(scienceArtifactDomId(workbenchTarget.artifactId));
    element?.scrollIntoView({ block: "center" });
    element?.focus({ preventScroll: true });
  }, [model.status, workbenchTarget]);

  useEffect(
    () => () => {
      activeExecution.current?.abort();
      activeFileImport.current?.abort();
    },
    [],
  );

  const load = useCallback(
    async (signal?: AbortSignal, preferredNotebookId: string | null = null) => {
      setModel({ status: "loading" });
      try {
        const workspace = await loadWorkspace(signal);
        setModel({ status: "ready", workspace });
        setActiveNotebookId((current) =>
          resolveActiveNotebookId(workspace.notebooks, preferredNotebookId ?? current),
        );
        const document = workspace.documents[0];
        if (document) {
          setDocumentName(document.name);
          setDocumentContent(document.content);
        }
        setDocumentSelection(null);
        const figure = workspace.figures[0];
        if (figure) {
          setFigureTitle(figure.title);
          setFigureLibrary(figure.library);
          setFigureCode(figure.code);
        }
        setSelectedFigureObjectIds([]);
      } catch (error) {
        if (signal?.aborted) return;
        setModel({ status: "error", message: errorMessage(error) });
      }
    },
    [loadWorkspace],
  );

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const addProject = useCallback(() => {
    const title = projectTitle.trim();
    if (!title) return;
    setModel({ status: "loading" });
    void createProject(title).then(
      () => {
        setProjectTitle("");
        void load();
      },
      (error) => setModel({ status: "error", message: errorMessage(error) }),
    );
  }, [createProject, load, projectTitle]);

  const addNotebook = useCallback(() => {
    const title = notebookTitle.trim();
    const projectId = model.status === "ready" ? model.workspace.projects[0]?.id : undefined;
    if (!title || !projectId) return;
    setModel({ status: "loading" });
    void createNotebook(projectId, title).then(
      (notebook) => {
        setNotebookTitle("");
        setNewNotebookOpen(false);
        void load(undefined, notebook.id);
      },
      (error) => setModel({ status: "error", message: errorMessage(error) }),
    );
  }, [createNotebook, load, model, notebookTitle]);

  const runCell = useCallback(() => {
    const source = cellSource;
    const outputArtifact = notebookOutputArtifact(outputCapturePath);
    const notebookId =
      model.status === "ready"
        ? resolveActiveNotebookId(model.workspace.notebooks, activeNotebookId)
        : null;
    if (!source.trim() || !notebookId || outputArtifact === undefined) return;
    const controller = new AbortController();
    activeExecution.current?.abort();
    activeExecution.current = controller;
    setModel({ status: "loading" });
    void executeCell(notebookId, source, outputArtifact, [], controller.signal).then(
      () => {
        if (activeExecution.current === controller) activeExecution.current = undefined;
        setCellSource("");
        setOutputCapturePath("");
        setCellComposerOpen(false);
        void load();
      },
      (error) => {
        if (activeExecution.current === controller) activeExecution.current = undefined;
        if (controller.signal.aborted) return;
        setModel({ status: "error", message: errorMessage(error) });
      },
    );
  }, [activeNotebookId, cellSource, executeCell, load, model, outputCapturePath]);

  const importResearchFiles = useCallback(
    (files: readonly File[]) => {
      const projectId = model.status === "ready" ? model.workspace.projects[0]?.id : undefined;
      if (!projectId || files.length === 0) return;
      const invalid = files.find(
        (file) =>
          scienceImportType(file.name) === undefined ||
          file.size === 0 ||
          file.size > MAX_SCIENCE_IMPORT_BYTES,
      );
      if (invalid) {
        setFileImportStatus({
          state: "error",
          message: `Cannot import ${invalid.name}. Use a supported file up to 8 MiB.`,
        });
        return;
      }
      const controller = new AbortController();
      activeFileImport.current?.abort();
      activeFileImport.current = controller;
      setFileImportStatus({ state: "importing", completed: 0, total: files.length });
      void (async () => {
        try {
          for (const [index, file] of files.entries()) {
            controller.signal.throwIfAborted();
            setFileImportStatus({ state: "importing", completed: index, total: files.length });
            const dataBase64 = await fileBase64(file, controller.signal);
            await importArtifact(projectId, file.name, dataBase64, controller.signal);
          }
          if (activeFileImport.current === controller) activeFileImport.current = undefined;
          setFileImportStatus({ state: "idle" });
          await load();
        } catch (error) {
          if (activeFileImport.current === controller) activeFileImport.current = undefined;
          if (controller.signal.aborted) return;
          setFileImportStatus({ state: "error", message: errorMessage(error) });
        }
      })();
    },
    [importArtifact, load, model],
  );

  const analyzeArtifact = useCallback(
    (artifact: ScienceArtifact) => {
      if (model.status !== "ready") return;
      const project = model.workspace.projects[0];
      if (!project || artifact.projectId !== project.id) return;
      const selectedNotebookId = resolveActiveNotebookId(
        model.workspace.notebooks,
        activeNotebookId,
      );
      const plan = scienceArtifactAnalysisPlan(artifact);
      const controller = new AbortController();
      activeExecution.current?.abort();
      activeExecution.current = controller;
      setActiveDestination("notebook");
      setCellComposerOpen(false);
      setModel({ status: "loading" });
      void (async () => {
        try {
          const notebookId =
            selectedNotebookId ??
            (await createNotebook(project.id, "File analysis", controller.signal)).id;
          setActiveNotebookId(notebookId);
          await executeCell(
            notebookId,
            plan.source,
            plan.outputArtifact,
            [artifact.id],
            controller.signal,
          );
          if (activeExecution.current === controller) activeExecution.current = undefined;
          await load(undefined, notebookId);
        } catch (error) {
          if (activeExecution.current === controller) activeExecution.current = undefined;
          if (controller.signal.aborted) return;
          setModel({ status: "error", message: errorMessage(error) });
        }
      })();
    },
    [activeNotebookId, createNotebook, executeCell, load, model],
  );

  const addDocument = useCallback(() => {
    const name = documentName.trim();
    const projectId = model.status === "ready" ? model.workspace.projects[0]?.id : undefined;
    if (!name || !projectId) return;
    setModel({ status: "loading" });
    void createDocument(projectId, name, documentContent).then(
      () => void load(),
      (error) => setModel({ status: "error", message: errorMessage(error) }),
    );
  }, [createDocument, documentContent, documentName, load, model]);

  const proposeDocumentPatch = useCallback(() => {
    const document = model.status === "ready" ? model.workspace.documents[0] : undefined;
    if (!document || !documentSelection) return;
    setModel({ status: "loading" });
    void modifyDocument({
      documentId: document.id,
      expectedRevision: document.revision,
      action: "propose",
      selection: documentSelection,
      proposedText: proposalText,
      instruction: proposalInstruction,
      reasoning: proposalReasoning,
    }).then(
      () => {
        setProposalText("");
        setProposalInstruction("");
        setProposalReasoning("");
        void load();
      },
      (error) => setModel({ status: "error", message: errorMessage(error) }),
    );
  }, [
    documentSelection,
    load,
    model,
    modifyDocument,
    proposalInstruction,
    proposalReasoning,
    proposalText,
  ]);

  const resolveDocumentPatch = useCallback(
    (proposalId: string, action: "accept" | "reject") => {
      const document = model.status === "ready" ? model.workspace.documents[0] : undefined;
      if (!document) return;
      setModel({ status: "loading" });
      void modifyDocument({
        documentId: document.id,
        expectedRevision: document.revision,
        action,
        proposalId,
      }).then(
        () => void load(),
        (error) => setModel({ status: "error", message: errorMessage(error) }),
      );
    },
    [load, model, modifyDocument],
  );

  const addFigure = useCallback(() => {
    const title = figureTitle.trim();
    const projectId = model.status === "ready" ? model.workspace.projects[0]?.id : undefined;
    if (!title || !figureCode.trim() || !projectId) return;
    setModel({ status: "loading" });
    settleScienceMutation(
      createFigure(projectId, title, figureLibrary, figureCode),
      () => void load(),
      (error) => setModel({ status: "error", message: errorMessage(error) }),
    );
  }, [createFigure, figureCode, figureLibrary, figureTitle, load, model]);

  const toggleFigureObject = useCallback((objectId: string, additive: boolean) => {
    setSelectedFigureObjectIds((current) =>
      toggleFigureObjectSelection(current, objectId, additive),
    );
  }, []);

  const proposeFigurePatch = useCallback(() => {
    const figure = model.status === "ready" ? model.workspace.figures[0] : undefined;
    if (!figure || selectedFigureObjectIds.length === 0) return;
    setModel({ status: "loading" });
    void modifyFigureCode({
      figureId: figure.id,
      expectedRevision: figure.revision,
      action: "propose",
      objectIds: selectedFigureObjectIds,
      proposedCode: figureProposalCode,
      instruction: figureProposalInstruction,
      reasoning: figureProposalReasoning,
    }).then(
      () => {
        setFigureProposalCode("");
        setFigureProposalInstruction("");
        setFigureProposalReasoning("");
        void load();
      },
      (error) => setModel({ status: "error", message: errorMessage(error) }),
    );
  }, [
    figureProposalCode,
    figureProposalInstruction,
    figureProposalReasoning,
    load,
    model,
    modifyFigureCode,
    selectedFigureObjectIds,
  ]);

  const resolveFigurePatch = useCallback(
    (proposalId: string, action: "accept" | "reject") => {
      const figure = model.status === "ready" ? model.workspace.figures[0] : undefined;
      if (!figure) return;
      setModel({ status: "loading" });
      void modifyFigureCode({
        figureId: figure.id,
        expectedRevision: figure.revision,
        action,
        proposalId,
      }).then(
        () => void load(),
        (error) => setModel({ status: "error", message: errorMessage(error) }),
      );
    },
    [load, model, modifyFigureCode],
  );

  const addExperiment = useCallback(() => {
    const projectId = model.status === "ready" ? model.workspace.projects[0]?.id : undefined;
    const title = experimentTitle.trim();
    const summary = experimentSummary.trim();
    const protocol = experimentProtocol.trim();
    if (!projectId || !title || !summary || !protocol) return;
    setModel({ status: "loading" });
    settleScienceMutation(
      defineExperiment(projectId, title, summary, protocol),
      () => {
        setExperimentTitle("");
        setExperimentSummary("");
        setExperimentProtocol("");
        void load();
      },
      (error) => setModel({ status: "error", message: errorMessage(error) }),
    );
  }, [defineExperiment, experimentProtocol, experimentSummary, experimentTitle, load, model]);

  const beginRun = useCallback(
    (experimentId: string, expectedRevision: number) => {
      setModel({ status: "loading" });
      settleScienceMutation(
        startRun(experimentId, expectedRevision),
        () => void load(),
        (error) => setModel({ status: "error", message: errorMessage(error) }),
      );
    },
    [load, startRun],
  );

  const completeRun = useCallback(
    (runId: string, expectedRevision: number) => {
      setModel({ status: "loading" });
      settleScienceMutation(
        finishRun(runId, expectedRevision),
        () => void load(),
        (error) => setModel({ status: "error", message: errorMessage(error) }),
      );
    },
    [finishRun, load],
  );

  const exportCurrentProject = useCallback(() => {
    const projectId = model.status === "ready" ? model.workspace.projects[0]?.id : undefined;
    if (!projectId) return;
    settleScienceMutation(
      exportProject(projectId),
      (exported) => {
        downloadProjectExport(exported);
        void load();
      },
      (error) => setModel({ status: "error", message: errorMessage(error) }),
    );
  }, [exportProject, load, model]);

  return (
    <ScienceWorkspaceView
      model={model}
      projectTitle={projectTitle}
      notebookTitle={notebookTitle}
      cellSource={cellSource}
      outputCapturePath={outputCapturePath}
      documentName={documentName}
      documentContent={documentContent}
      documentSelection={documentSelection}
      proposalText={proposalText}
      proposalInstruction={proposalInstruction}
      proposalReasoning={proposalReasoning}
      figureTitle={figureTitle}
      figureLibrary={figureLibrary}
      figureCode={figureCode}
      selectedFigureObjectIds={selectedFigureObjectIds}
      figureProposalCode={figureProposalCode}
      figureProposalInstruction={figureProposalInstruction}
      figureProposalReasoning={figureProposalReasoning}
      researchSearch={researchSearch}
      experimentTitle={experimentTitle}
      experimentSummary={experimentSummary}
      experimentProtocol={experimentProtocol}
      focusedArtifactId={workbenchTarget?.kind === "artifact" ? workbenchTarget.artifactId : null}
      activeNotebookId={activeNotebookId}
      activeDestination={activeDestination}
      newNotebookOpen={newNotebookOpen}
      cellComposerOpen={cellComposerOpen}
      fileImportStatus={fileImportStatus}
      onProjectTitleChange={setProjectTitle}
      onNotebookTitleChange={setNotebookTitle}
      onActiveNotebookChange={setActiveNotebookId}
      onDestinationChange={setActiveDestination}
      onNewNotebookOpenChange={setNewNotebookOpen}
      onCellComposerOpenChange={setCellComposerOpen}
      onCellSourceChange={setCellSource}
      onOutputCapturePathChange={setOutputCapturePath}
      onDocumentNameChange={setDocumentName}
      onDocumentContentChange={setDocumentContent}
      onDocumentSelectionChange={setDocumentSelection}
      onProposalTextChange={setProposalText}
      onProposalInstructionChange={setProposalInstruction}
      onProposalReasoningChange={setProposalReasoning}
      onFigureTitleChange={setFigureTitle}
      onFigureLibraryChange={setFigureLibrary}
      onFigureCodeChange={setFigureCode}
      onFigureObjectToggle={toggleFigureObject}
      onFigureProposalCodeChange={setFigureProposalCode}
      onFigureProposalInstructionChange={setFigureProposalInstruction}
      onFigureProposalReasoningChange={setFigureProposalReasoning}
      onResearchSearchChange={setResearchSearch}
      onExperimentTitleChange={setExperimentTitle}
      onExperimentSummaryChange={setExperimentSummary}
      onExperimentProtocolChange={setExperimentProtocol}
      onCreateProject={addProject}
      onCreateNotebook={addNotebook}
      onExecuteCell={runCell}
      onCreateDocument={addDocument}
      onProposeDocumentPatch={proposeDocumentPatch}
      onResolveDocumentPatch={resolveDocumentPatch}
      onCreateFigure={addFigure}
      onProposeFigurePatch={proposeFigurePatch}
      onResolveFigurePatch={resolveFigurePatch}
      onDefineExperiment={addExperiment}
      onStartRun={beginRun}
      onFinishRun={completeRun}
      onExportProject={exportCurrentProject}
      onImportFiles={importResearchFiles}
      onAnalyzeArtifact={analyzeArtifact}
      onOpenArtifact={openArtifact}
      onRetry={() => void load()}
    />
  );
}
