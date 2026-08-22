import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { ScienceWorkspaceSnapshot } from "../../../science/core/src/contracts.js";
import {
  matchesResearchSearch,
  notebookOutputArtifact,
  resolveActiveNotebookId,
  ScienceWorkspaceView,
  scienceArtifactAnalysisPlan,
  scienceArtifactDomId,
  scienceImportType,
  settleScienceMutation,
  textareaSelection,
  toggleFigureObjectSelection,
} from "../src/client/science-workspace.js";

const emptyWorkspace: ScienceWorkspaceSnapshot = {
  projects: [],
  notebooks: [],
  artifacts: [],
  documents: [],
  figures: [],
  records: [],
  relations: [],
  experiments: [],
  runs: [],
  exports: [],
};

function render(
  model: Parameters<typeof ScienceWorkspaceView>[0]["model"],
  focusedArtifactId: string | null = null,
  activeNotebookId: string | null = null,
  activeDestination: Parameters<typeof ScienceWorkspaceView>[0]["activeDestination"] = "notebook",
  newNotebookOpen = false,
  cellComposerOpen = false,
): string {
  return renderToStaticMarkup(
    <ScienceWorkspaceView
      model={model}
      projectTitle=""
      notebookTitle=""
      cellSource=""
      documentName="paper.typ"
      documentContent=""
      documentSelection={null}
      proposalText=""
      proposalInstruction=""
      proposalReasoning=""
      figureTitle=""
      figureLibrary="matplotlib"
      figureCode=""
      selectedFigureObjectIds={[]}
      figureProposalCode=""
      figureProposalInstruction=""
      figureProposalReasoning=""
      researchSearch=""
      experimentTitle=""
      experimentSummary=""
      experimentProtocol=""
      focusedArtifactId={focusedArtifactId}
      activeNotebookId={activeNotebookId}
      activeDestination={activeDestination}
      newNotebookOpen={newNotebookOpen}
      cellComposerOpen={cellComposerOpen}
      fileImportStatus={{ state: "idle" }}
      outputCapturePath=""
      onProjectTitleChange={vi.fn()}
      onNotebookTitleChange={vi.fn()}
      onActiveNotebookChange={vi.fn()}
      onDestinationChange={vi.fn()}
      onNewNotebookOpenChange={vi.fn()}
      onCellComposerOpenChange={vi.fn()}
      onOutputCapturePathChange={vi.fn()}
      onCellSourceChange={vi.fn()}
      onDocumentNameChange={vi.fn()}
      onDocumentContentChange={vi.fn()}
      onDocumentSelectionChange={vi.fn()}
      onProposalTextChange={vi.fn()}
      onProposalInstructionChange={vi.fn()}
      onProposalReasoningChange={vi.fn()}
      onFigureTitleChange={vi.fn()}
      onFigureLibraryChange={vi.fn()}
      onFigureCodeChange={vi.fn()}
      onFigureObjectToggle={vi.fn()}
      onFigureProposalCodeChange={vi.fn()}
      onFigureProposalInstructionChange={vi.fn()}
      onFigureProposalReasoningChange={vi.fn()}
      onResearchSearchChange={vi.fn()}
      onExperimentTitleChange={vi.fn()}
      onExperimentSummaryChange={vi.fn()}
      onExperimentProtocolChange={vi.fn()}
      onCreateProject={vi.fn()}
      onCreateNotebook={vi.fn()}
      onExecuteCell={vi.fn()}
      onCreateDocument={vi.fn()}
      onProposeDocumentPatch={vi.fn()}
      onResolveDocumentPatch={vi.fn()}
      onCreateFigure={vi.fn()}
      onProposeFigurePatch={vi.fn()}
      onResolveFigurePatch={vi.fn()}
      onDefineExperiment={vi.fn()}
      onStartRun={vi.fn()}
      onFinishRun={vi.fn()}
      onExportProject={vi.fn()}
      onImportFiles={vi.fn()}
      onAnalyzeArtifact={vi.fn()}
      onOpenArtifact={vi.fn()}
      onRetry={vi.fn()}
    />,
  );
}

describe("V23 Science Workspace states", () => {
  it("keeps all five first-level destinations visible while loading", () => {
    const markup = render({ status: "loading" });

    expect(markup).toContain('data-science-shell="true"');
    expect(markup).toContain('aria-label="Science project navigation"');
    expect(markup).toContain('role="status"');
    for (const label of ["Notebook", "Writing", "Figures", "Research Map", "Experiments"]) {
      expect(markup).toContain(label);
    }
    expect(markup).toContain("Files");
    expect(markup).not.toContain("Local-first scientific IDE");
    expect(markup).not.toContain("Next slice");
  });

  it("V45 routes a rejected mutation only to the retryable error path", async () => {
    const success = vi.fn();
    const failure = vi.fn();

    settleScienceMutation(Promise.reject(new Error("figure failed")), success, failure);
    await Promise.resolve();

    expect(success).not.toHaveBeenCalled();
    expect(failure).toHaveBeenCalledOnce();
    expect(failure).toHaveBeenCalledWith(expect.objectContaining({ message: "figure failed" }));
  });

  it("renders an accessible empty project creation flow", () => {
    const markup = render({ status: "ready", workspace: emptyWorkspace });

    expect(markup).toContain("No research project yet");
    expect(markup).toContain('for="science-project-title"');
    expect(markup).toContain('id="science-project-title"');
    expect(markup).toContain("Create project");
  });

  it("renders a retryable error without hiding the product destinations", () => {
    const markup = render({ status: "error", message: "Science service unavailable" });

    expect(markup).toContain('role="alert"');
    expect(markup).toContain("Science service unavailable");
    expect(markup).toContain("Retry");
    expect(markup).toContain("Notebook");
  });

  it("renders created project and notebook provenance without host paths", () => {
    const workspace: ScienceWorkspaceSnapshot = {
      projects: [
        {
          id: "project-1",
          kind: "project",
          title: "Protein folding",
          createdAt: 1,
          updatedAt: 1,
          revision: 1,
          provenance: { eventId: "event-1", journalSeq: 1, sessionId: "session-1" },
        },
      ],
      notebooks: [
        {
          id: "notebook-1",
          projectId: "project-1",
          kind: "notebook",
          title: "Baseline",
          cells: [
            {
              id: "cell-1",
              kind: "code",
              source: "print(42)",
              executionCount: 1,
              executionTimeMs: 12,
              outputArtifactIds: [],
              outputs: [],
              runtimeEnvironment: { pythonVersion: "3.12.1" },
              relatedClaimIds: [],
              relatedExperimentIds: [],
            },
            {
              id: "cell-2",
              kind: "output",
              source: "42\n",
              executionCount: 1,
              executionTimeMs: 12,
              outputArtifactIds: [],
              outputs: [
                {
                  type: "execute_result",
                  data: [
                    {
                      mime: "text/plain",
                      data: "42\n",
                      encoding: "utf8",
                      truncated: false,
                    },
                  ],
                },
              ],
              runtimeEnvironment: { pythonVersion: "3.12.1" },
              relatedClaimIds: [],
              relatedExperimentIds: [],
            },
          ],
          createdAt: 2,
          updatedAt: 2,
          revision: 1,
          provenance: { eventId: "event-2", journalSeq: 2, sessionId: "session-1" },
        },
      ],
      artifacts: [
        {
          id: "artifact-1",
          projectId: "project-1",
          kind: "dataset",
          title: "analysis.csv",
          digest: `sha256:${"a".repeat(64)}`,
          mime: "text/csv",
          size: 128,
          creator: { kind: "session", sessionId: "session-1" },
          runId: null,
          environment: {},
          license: null,
          sourceEntityIds: ["notebook-1"],
          createdAt: 3,
          updatedAt: 3,
          revision: 1,
          provenance: { eventId: "event-3", journalSeq: 3, sessionId: "session-1" },
        },
      ],
      documents: [
        {
          id: "document-1",
          projectId: "project-1",
          kind: "document",
          name: "paper.typ",
          format: "typst",
          content: "Our model improves accuracy.",
          revision: 2,
          contentRevision: 1,
          proposals: [
            {
              id: "proposal-1",
              selection: { start: 10, end: 18 },
              originalText: "improves",
              proposedText: "improves accuracy by 4.2%",
              instruction: "Quantify the result",
              reasoning: {
                classification: "proposal",
                summary: "The claim needs measured evidence.",
              },
              status: "pending",
              createdAt: 4,
              resolvedAt: null,
              createdProvenance: { eventId: "event-4", journalSeq: 4, sessionId: "session-1" },
              resolvedProvenance: null,
            },
          ],
          revisions: [
            {
              revision: 1,
              contentRevision: 1,
              sourceHash: `sha256:${"b".repeat(64)}`,
              previousSourceHash: null,
              reason: "created",
              proposalId: null,
              provenance: { eventId: "event-3", journalSeq: 3, sessionId: "session-1" },
            },
            {
              revision: 2,
              contentRevision: 1,
              sourceHash: `sha256:${"b".repeat(64)}`,
              previousSourceHash: `sha256:${"b".repeat(64)}`,
              reason: "proposal-created",
              proposalId: "proposal-1",
              provenance: { eventId: "event-4", journalSeq: 4, sessionId: "session-1" },
            },
          ],
          diagnostics: [
            {
              code: "claim-needs-evidence",
              scope: "scientific",
              severity: "warning",
              message: "Scientific claim lacks quantitative evidence.",
              start: 10,
              end: 18,
            },
          ],
          validation: { structural: "checked", compilation: "not-run" },
          createdAt: 3,
          updatedAt: 4,
          provenance: { eventId: "event-4", journalSeq: 4, sessionId: "session-1" },
        },
      ],
      figures: [
        {
          id: "figure-1",
          projectId: "project-1",
          kind: "figure",
          title: "Accuracy curve",
          library: "matplotlib",
          code: 'plt.plot(x, y)\nplt.legend(loc="best")',
          artifactId: "artifact-1",
          objects: [
            { id: "line-1", kind: "line", label: "line 1", codeRange: { start: 0, end: 14 } },
            {
              id: "legend-1",
              kind: "legend",
              label: "legend 1",
              codeRange: { start: 15, end: 38 },
            },
          ],
          revision: 2,
          codeRevision: 1,
          proposals: [
            {
              id: "figure-proposal-1",
              objectIds: ["legend-1"],
              selection: { start: 15, end: 38 },
              originalCode: 'plt.legend(loc="best")',
              proposedCode: "plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8)",
              instruction: "Move the legend outside",
              reasoning: {
                classification: "proposal",
                summary: "The legend overlaps the curve.",
              },
              status: "pending",
              createdAt: 5,
              resolvedAt: null,
              createdProvenance: { eventId: "event-5", journalSeq: 5, sessionId: "session-1" },
              resolvedProvenance: null,
            },
          ],
          revisions: [
            {
              revision: 1,
              codeRevision: 1,
              codeHash: `sha256:${"c".repeat(64)}`,
              previousCodeHash: null,
              reason: "created",
              proposalId: null,
              provenance: { eventId: "event-4", journalSeq: 4, sessionId: "session-1" },
            },
            {
              revision: 2,
              codeRevision: 1,
              codeHash: `sha256:${"c".repeat(64)}`,
              previousCodeHash: `sha256:${"c".repeat(64)}`,
              reason: "proposal-created",
              proposalId: "figure-proposal-1",
              provenance: { eventId: "event-5", journalSeq: 5, sessionId: "session-1" },
            },
          ],
          createdAt: 4,
          updatedAt: 5,
          provenance: { eventId: "event-5", journalSeq: 5, sessionId: "session-1" },
        },
      ],
      records: [
        {
          id: "question-1",
          projectId: "project-1",
          kind: "question",
          title: "Does accuracy improve?",
          summary: "Compare the held-out cohort.",
          status: "open",
          tags: ["primary"],
          sourceEntityIds: [],
          createdAt: 6,
          updatedAt: 6,
          revision: 1,
          provenance: { eventId: "event-6", journalSeq: 6, sessionId: "session-1" },
        },
        {
          id: "claim-1",
          projectId: "project-1",
          kind: "claim",
          title: "Accuracy improves by 4.2%",
          summary: "Held-out accuracy exceeds baseline.",
          status: "supported",
          tags: [],
          sourceEntityIds: ["question-1"],
          createdAt: 7,
          updatedAt: 7,
          revision: 1,
          provenance: { eventId: "event-7", journalSeq: 7, sessionId: "session-1" },
        },
      ],
      relations: [
        {
          id: "relation-1",
          projectId: "project-1",
          fromId: "claim-1",
          toId: "question-1",
          type: "derived_from",
          createdAt: 7,
          provenance: { eventId: "event-7", journalSeq: 7, sessionId: "session-1" },
        },
      ],
      experiments: [
        {
          id: "experiment-1",
          projectId: "project-1",
          kind: "experiment",
          title: "Accuracy benchmark",
          summary: "Repeat the held-out evaluation.",
          protocol: "python evaluate.py",
          hypothesisIds: [],
          runIds: ["run-1"],
          status: "active",
          tags: [],
          createdAt: 8,
          updatedAt: 9,
          revision: 2,
          provenance: { eventId: "event-9", journalSeq: 9, sessionId: "session-1" },
        },
      ],
      runs: [
        {
          id: "run-1",
          projectId: "project-1",
          experimentId: "experiment-1",
          kind: "run",
          status: "running",
          environment: { seed: "1" },
          metrics: {},
          artifactIds: [],
          notes: "",
          startedAt: 9,
          finishedAt: null,
          revision: 1,
          provenance: { eventId: "event-9", journalSeq: 9, sessionId: "session-1" },
        },
      ],
      exports: [
        {
          id: "export-1",
          projectId: "project-1",
          kind: "export",
          format: "dsh-science-project@1",
          digest: `sha256:${"d".repeat(64)}`,
          bytes: 4096,
          counts: {
            projects: 1,
            notebooks: 1,
            artifacts: 1,
            documents: 1,
            figures: 1,
            records: 2,
            relations: 1,
            experiments: 1,
            runs: 1,
          },
          createdAt: 10,
          revision: 1,
          provenance: { eventId: "event-10", journalSeq: 10, sessionId: "session-1" },
        },
      ],
    };
    const markup = (["notebook", "writing", "figures", "research", "experiments"] as const)
      .map((destination) =>
        render({ status: "ready", workspace }, "artifact-1", "notebook-1", destination),
      )
      .join("\n");

    expect(markup).toContain("Protein folding");
    expect(markup).toContain("Baseline");
    expect(markup).toContain("Journal #2");
    expect(markup).toContain("analysis.csv");
    expect(markup).toContain("128 B");
    expect(markup).toContain("New Python cell");
    expect(markup).not.toContain('id="science-cell-source"');
    expect(markup).toContain("print(42)");
    expect(markup).toContain("42\n");
    expect(markup).toContain("Writing Studio");
    expect(markup).toContain('for="science-document-source"');
    expect(markup).toContain('id="science-document-source"');
    expect(markup).toContain("Structure checked · Compilation not run");
    expect(markup).toContain("Scientific claim lacks quantitative evidence.");
    expect(markup).toContain("improves accuracy by 4.2%");
    expect(markup).toContain("Accept");
    expect(markup).toContain("Reject");
    expect(markup).toContain("Figure Studio");
    expect(markup).toContain('role="img"');
    expect(markup).toContain("Semantic figure canvas");
    expect(markup).toContain('aria-pressed="false"');
    expect(markup).toContain("legend 1");
    expect(markup).toContain("plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8)");
    expect(markup).toContain("Research Map");
    expect(markup).toContain('id="science-research-search"');
    expect(markup).toContain("Accuracy improves by 4.2%");
    expect(markup).toContain("derived_from");
    expect(markup).toContain("Accuracy benchmark");
    expect(markup).toContain("Start Run");
    expect(markup).toContain("Finish succeeded");
    expect(markup).toContain("Export project JSON");
    expect(markup).toContain(`sha256:${"d".repeat(64)}`);
    expect(markup).not.toContain("/Users/");
    expect(markup).toContain("Open artifact details");
    expect(markup).toContain('data-artifact-summary="true"');
    expect(markup).toContain(`id="${scienceArtifactDomId("artifact-1")}"`);
    expect(markup).toContain('data-focused="true"');
  });

  it("V56 keeps one accessible active Notebook and renders only its cells", () => {
    const project = {
      id: "project-1",
      kind: "project" as const,
      title: "Benchmark",
      createdAt: 1,
      updatedAt: 1,
      revision: 1,
      provenance: { eventId: "event-1", journalSeq: 1, sessionId: "session-1" },
    };
    const notebook = (id: string, title: string, source: string, journalSeq: number) => ({
      id,
      projectId: project.id,
      kind: "notebook" as const,
      title,
      cells: [
        {
          id: `cell-${id}`,
          kind: "code" as const,
          source,
          executionCount: 1,
          executionTimeMs: 4,
          outputArtifactIds: [],
          outputs: [],
          runtimeEnvironment: {},
          relatedClaimIds: [],
          relatedExperimentIds: [],
        },
      ],
      createdAt: journalSeq,
      updatedAt: journalSeq,
      revision: 1,
      provenance: {
        eventId: `event-${journalSeq}`,
        journalSeq,
        sessionId: "session-1",
      },
    });
    const first = notebook("notebook-1", "Baseline", "print('baseline')", 2);
    const second = notebook("notebook-2", "Live Demo", "print('selected')", 3);
    const workspace: ScienceWorkspaceSnapshot = {
      ...emptyWorkspace,
      projects: [project],
      notebooks: [first, second],
    };

    expect(resolveActiveNotebookId(workspace.notebooks, null)).toBe(first.id);
    expect(resolveActiveNotebookId(workspace.notebooks, second.id)).toBe(second.id);
    expect(resolveActiveNotebookId([first], second.id)).toBe(first.id);
    expect(resolveActiveNotebookId([], second.id)).toBeNull();

    const markup = render({ status: "ready", workspace }, null, second.id);
    expect(markup).toContain('data-notebook-id="notebook-2" aria-pressed="true"');
    expect(markup).toContain('data-notebook-id="notebook-1" aria-pressed="false"');
    expect(markup).toContain("print(&#x27;selected&#x27;)");
    expect(markup).not.toContain("print(&#x27;baseline&#x27;)");
    expect(markup).toContain('aria-expanded="false"');
    expect(markup).not.toContain('id="science-notebook-title"');
    expect(markup).not.toContain('id="science-cell-source"');

    const createMarkup = render({ status: "ready", workspace }, null, second.id, "notebook", true);
    expect(createMarkup).toContain('aria-expanded="true"');
    expect(createMarkup).toContain('id="science-notebook-title"');

    const cellMarkup = render(
      { status: "ready", workspace },
      null,
      second.id,
      "notebook",
      false,
      true,
    );
    expect(cellMarkup).toContain('id="science-cell-source"');
  });

  it("V57 exposes one active first-level workbench and renders only its studio body", () => {
    const workspace: ScienceWorkspaceSnapshot = {
      ...emptyWorkspace,
      projects: [
        {
          id: "project-1",
          kind: "project",
          title: "Benchmark",
          createdAt: 1,
          updatedAt: 1,
          revision: 1,
          provenance: { eventId: "event-1", journalSeq: 1, sessionId: "session-1" },
        },
      ],
    };
    const destinations = {
      notebook: "science-notebooks-title",
      writing: "science-writing-title",
      figures: "science-figure-title",
      research: "science-research-map-title",
      experiments: "science-experiments-title",
    } as const;

    for (const [destination, sectionId] of Object.entries(destinations)) {
      const markup = render(
        { status: "ready", workspace },
        null,
        null,
        destination as keyof typeof destinations,
      );
      expect(markup.match(/aria-current="page"/g)).toHaveLength(1);
      expect(markup).toContain(`data-destination="${destination}"`);
      expect(markup).toContain(`id="${sectionId}"`);
      for (const otherId of Object.values(destinations).filter((id) => id !== sectionId)) {
        expect(markup).not.toContain(`id="${otherId}"`);
      }
    }
  });

  it("V58 derives bounded capture metadata only from safe supported relative paths", () => {
    expect(notebookOutputArtifact("")).toBeNull();
    expect(notebookOutputArtifact("results/live-demo.csv")).toEqual({
      relativePath: "results/live-demo.csv",
      kind: "dataset",
      title: "live-demo.csv",
      mime: "text/csv",
      license: null,
    });
    expect(notebookOutputArtifact("figure.PNG")).toEqual(
      expect.objectContaining({ kind: "figure", mime: "image/png", title: "figure.PNG" }),
    );
    for (const invalid of [
      "/tmp/result.csv",
      "../result.csv",
      "a/../../result.csv",
      "result.exe",
    ]) {
      expect(notebookOutputArtifact(invalid)).toBeUndefined();
    }
  });

  it("V68/V69 exposes accessible Files import and deterministic Notebook analysis actions", () => {
    const artifact = {
      id: "artifact-1",
      projectId: "project-1",
      kind: "dataset" as const,
      title: "measurements.csv",
      digest: `sha256:${"a".repeat(64)}` as const,
      mime: "text/csv",
      size: 24,
      creator: { kind: "session" as const, sessionId: "session-1" },
      runId: null,
      environment: {},
      license: null,
      sourceEntityIds: [],
      createdAt: 2,
      updatedAt: 2,
      revision: 1,
      provenance: { eventId: "event-2", journalSeq: 2, sessionId: "session-1" },
    };
    const workspace: ScienceWorkspaceSnapshot = {
      ...emptyWorkspace,
      projects: [
        {
          id: "project-1",
          kind: "project",
          title: "Imported data",
          createdAt: 1,
          updatedAt: 1,
          revision: 1,
          provenance: { eventId: "event-1", journalSeq: 1, sessionId: "session-1" },
        },
      ],
      artifacts: [artifact],
    };

    const markup = render({ status: "ready", workspace });
    expect(markup).toContain('type="file"');
    expect(markup).toContain('id="science-file-import"');
    expect(markup).toContain('aria-label="Import research files"');
    expect(markup).toContain("Drop files here");
    expect(markup).toContain('aria-label="Analyze measurements.csv in Notebook"');

    expect(scienceImportType("DATA.XLSX")).toEqual({
      kind: "dataset",
      mime: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    });
    expect(scienceImportType("folder/data.csv")).toBeUndefined();
    expect(scienceImportType("program.exe")).toBeUndefined();
    const plan = scienceArtifactAnalysisPlan(artifact);
    expect(plan.source).toContain('os.environ["DSH_SCIENCE_INPUT_0"]');
    expect(plan.source).toContain("csv");
    expect(plan.outputArtifact).toMatchObject({
      kind: "dataset",
      title: "measurements-analysis.json",
      mime: "application/json",
    });
  });

  it("uses textarea-compatible UTF-16 offsets for a source selection", () => {
    expect(
      textareaSelection({
        selectionStart: 3,
        selectionEnd: 11,
      }),
    ).toEqual({ start: 3, end: 11 });
    expect(textareaSelection({ selectionStart: 4, selectionEnd: 4 })).toBeNull();
  });

  it("supports single-object and additive brush selection", () => {
    expect(toggleFigureObjectSelection(["line-1"], "legend-1", false)).toEqual(["legend-1"]);
    expect(toggleFigureObjectSelection(["line-1"], "legend-1", true)).toEqual([
      "line-1",
      "legend-1",
    ]);
    expect(toggleFigureObjectSelection(["line-1", "legend-1"], "legend-1", true)).toEqual([
      "line-1",
    ]);
  });

  it("finds Research Map records by title, tag, id, or pasted tool locator", () => {
    const record = {
      id: "claim-42",
      kind: "claim" as const,
      title: "Recovery improves",
      summary: "Supported by Run 9.",
      tags: ["primary"],
    };

    expect(matchesResearchSearch(record, "recovery")).toBe(true);
    expect(matchesResearchSearch(record, "primary")).toBe(true);
    expect(matchesResearchSearch(record, "claim-42")).toBe(true);
    expect(matchesResearchSearch(record, '{"entityId":"claim-42"}')).toBe(true);
    expect(matchesResearchSearch(record, "unrelated")).toBe(false);
  });
});
