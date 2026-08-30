import { randomUUID } from "node:crypto";
import type { SessionId } from "@deepseek-ai/dsh-session";
import type {
  RoCrateMetadataDocument,
  ScienceDocument,
  ScienceExperiment,
  ScienceFigure,
  ScienceNotebook,
  ScienceProject,
  ScienceProjectExport,
  ScienceResearchRecord,
  ScienceRun,
} from "./contracts.js";
import type { ScienceCore } from "./core.js";

export interface ScienceDemoResult {
  readonly project: ScienceProject;
  readonly notebook: ScienceNotebook;
  readonly experiment: ScienceExperiment;
  readonly run: ScienceRun;
  readonly figure: ScienceFigure;
  readonly document: ScienceDocument;
  readonly claim: ScienceResearchRecord;
  readonly evidence: ScienceResearchRecord;
  readonly researchObject: RoCrateMetadataDocument;
  readonly exported: ScienceProjectExport;
}

/** One runnable, local-only tour through the public Science service. */
export async function runScienceDemo(
  science: ScienceCore,
  sessionId: SessionId,
  signal?: AbortSignal,
): Promise<ScienceDemoResult> {
  const project = science.createProject(
    sessionId,
    { requestId: randomUUID(), title: "Recovery treatment study" },
    signal,
  );
  let notebook = science.createNotebook(
    sessionId,
    { requestId: randomUUID(), projectId: project.id, title: "Recovery analysis" },
    signal,
  );
  const execution = await science.executeNotebookCell(
    sessionId,
    {
      requestId: randomUUID(),
      notebookId: notebook.id,
      source: [
        "from pathlib import Path",
        'Path("recovery.csv").write_text("group,days\\nbaseline,10\\ntreatment,8.8\\n")',
        'Path("recovery.svg").write_text("<svg xmlns=\\"http://www.w3.org/2000/svg\\" width=\\"240\\" height=\\"120\\"><path d=\\"M20 90 L110 40 L210 28\\" stroke=\\"#1f7658\\" fill=\\"none\\"/></svg>")',
        'print("mean improvement: 12%")',
      ].join("\n"),
      outputArtifact: {
        relativePath: "recovery.csv",
        kind: "dataset",
        title: "Recovery dataset",
        mime: "text/csv",
        license: "CC0-1.0",
      },
    },
    signal,
  );
  notebook = execution.notebook;
  const question = science.createQuestion(
    sessionId,
    {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Does treatment shorten recovery?",
      summary: "Compare treatment and baseline recovery time.",
      tags: ["recovery"],
    },
    signal,
  );
  const hypothesis = science.createHypothesis(
    sessionId,
    {
      requestId: randomUUID(),
      projectId: project.id,
      questionId: question.id,
      title: "Treatment reduces mean recovery time",
      summary: "The treatment cohort recovers faster than baseline.",
      tags: ["primary"],
    },
    signal,
  );
  const experiment = science.defineExperiment(
    sessionId,
    {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Recovery comparison",
      summary: "Reproduce the cohort mean comparison.",
      hypothesisIds: [hypothesis.id],
      protocol: "Execute the Recovery analysis notebook with the bundled local dataset.",
      tags: ["demo"],
    },
    signal,
  );
  const started = science.startRun(
    sessionId,
    {
      requestId: randomUUID(),
      experimentId: experiment.id,
      expectedRevision: experiment.revision,
      environment: { seed: "2026", runtime: "python" },
    },
    signal,
  );
  const runDataset = await science.registerArtifact(
    sessionId,
    {
      requestId: randomUUID(),
      projectId: project.id,
      relativePath: "recovery.csv",
      kind: "dataset",
      title: "Run recovery dataset",
      mime: "text/csv",
      runId: started.run.id,
      environment: { seed: "2026" },
      license: "CC0-1.0",
      sourceEntityIds: [notebook.id, started.run.id],
    },
    signal,
  );
  const run = science.finishRun(
    sessionId,
    {
      requestId: randomUUID(),
      runId: started.run.id,
      expectedRevision: started.run.revision,
      status: "succeeded",
      metrics: { improvementPercent: 12, treatmentMeanDays: 8.8 },
      artifactIds: [runDataset.id],
      notes: "Local demo run completed.",
    },
    signal,
  );
  const figureArtifact = await science.registerArtifact(
    sessionId,
    {
      requestId: randomUUID(),
      projectId: project.id,
      relativePath: "recovery.svg",
      kind: "figure",
      title: "Recovery curve",
      mime: "image/svg+xml",
      runId: run.id,
      environment: { renderer: "svg" },
      license: "CC0-1.0",
      sourceEntityIds: [run.id],
    },
    signal,
  );
  let figure = science.createFigure(
    sessionId,
    {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Recovery time by cohort",
      library: "matplotlib",
      code: 'plt.plot(days, label="Treatment")\nplt.legend(loc="best")\nplt.xlabel("Cohort")',
      artifactId: figureArtifact.id,
    },
    signal,
  );
  const legend = figure.objects.find((object) => object.kind === "legend");
  if (!legend) throw new Error("Demo figure did not infer a legend");
  figure = science.modifyFigureCode(
    sessionId,
    {
      requestId: randomUUID(),
      figureId: figure.id,
      expectedRevision: figure.revision,
      action: "propose",
      objectIds: [legend.id],
      proposedCode: "plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8)",
      instruction: "Move the legend outside and reduce its size.",
      reasoning: "The legend overlaps the recovery curve.",
    },
    signal,
  );
  const figureProposal = figure.proposals.at(-1);
  if (!figureProposal) throw new Error("Demo figure patch was not proposed");
  figure = science.modifyFigureCode(
    sessionId,
    {
      requestId: randomUUID(),
      figureId: figure.id,
      expectedRevision: figure.revision,
      action: "accept",
      proposalId: figureProposal.id,
    },
    signal,
  );
  let document = science.createDocument(
    sessionId,
    {
      requestId: randomUUID(),
      projectId: project.id,
      name: "paper.typ",
      content: "= Results\n\nThe treatment improved recovery.",
    },
    signal,
  );
  const originalText = "The treatment improved recovery.";
  const selectionStart = document.content.indexOf(originalText);
  document = science.modifyDocument(
    sessionId,
    {
      requestId: randomUUID(),
      documentId: document.id,
      expectedRevision: document.revision,
      action: "propose",
      selection: { start: selectionStart, end: selectionStart + originalText.length },
      proposedText: "The treatment reduced mean recovery time by 12% in the demo cohort.",
      instruction: "Quantify the result.",
      reasoning: "The Run records the measured improvement.",
    },
    signal,
  );
  const writingProposal = document.proposals.at(-1);
  if (!writingProposal) throw new Error("Demo writing patch was not proposed");
  document = science.modifyDocument(
    sessionId,
    {
      requestId: randomUUID(),
      documentId: document.id,
      expectedRevision: document.revision,
      action: "accept",
      proposalId: writingProposal.id,
    },
    signal,
  );
  const claim = science.recordClaim(
    sessionId,
    {
      requestId: randomUUID(),
      projectId: project.id,
      hypothesisId: hypothesis.id,
      title: "Treatment improves recovery by 12%",
      summary: "The completed local Run reports a 12% improvement.",
      status: "supported",
      tags: ["result"],
    },
    signal,
  );
  const linked = science.linkEvidence(
    sessionId,
    {
      requestId: randomUUID(),
      projectId: project.id,
      claimId: claim.id,
      relation: "supports",
      title: "Recovery Run and figure",
      summary: "Run metrics and the edited figure support the claim.",
      sourceEntityIds: [run.id, figure.id, document.id],
      tags: ["demo"],
    },
    signal,
  );
  const researchObject = science.getResearchObject(sessionId, { projectId: project.id }, signal);
  const exported = science.exportProject(
    sessionId,
    { requestId: randomUUID(), projectId: project.id },
    signal,
  );
  return {
    project,
    notebook,
    experiment: started.experiment,
    run,
    figure,
    document,
    claim,
    evidence: linked.evidence,
    researchObject,
    exported,
  };
}
