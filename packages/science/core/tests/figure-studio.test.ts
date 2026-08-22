import { randomUUID } from "node:crypto";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import type { ScienceError } from "../src/index.js";
import { createScienceFixture, type ScienceFixture } from "./fixture.js";

const fixtures: ScienceFixture[] = [];

afterEach(async () => {
  await Promise.all(fixtures.splice(0).map((fixture) => fixture.dispose()));
});

async function setup() {
  const fixture = await createScienceFixture();
  fixtures.push(fixture);
  const project = fixture.context.science.createProject(fixture.sessionA, {
    requestId: randomUUID(),
    title: "Semantic figures",
  });
  return { fixture, project };
}

const libraryCases = [
  {
    library: "matplotlib" as const,
    code: 'plt.plot(x, y)\nplt.legend(loc="best")\nplt.xlabel("Time")',
    kinds: ["line", "legend", "axis"],
  },
  {
    library: "seaborn" as const,
    code: 'sns.scatterplot(data=df, x="x", y="y")\nplt.legend()',
    kinds: ["point", "legend"],
  },
  {
    library: "ggplot2" as const,
    code: 'ggplot(df, aes(x, y)) + geom_point() +\n  theme(legend.position="bottom")',
    kinds: ["point", "legend"],
  },
  {
    library: "plotly" as const,
    code: 'fig.add_scatter(x=x, y=y, mode="lines")\nfig.update_layout(legend={})',
    kinds: ["line", "legend"],
  },
];

describe("T17 Science Figure Studio", () => {
  it.each(libraryCases)(
    "creates a $library figure with bounded semantic objects",
    async (sample) => {
      const { fixture, project } = await setup();

      const figure = fixture.context.science.createFigure(fixture.sessionA, {
        requestId: randomUUID(),
        projectId: project.id,
        title: `${sample.library} result`,
        library: sample.library,
        code: sample.code,
        artifactId: null,
      });

      expect(figure).toMatchObject({
        projectId: project.id,
        kind: "figure",
        title: `${sample.library} result`,
        library: sample.library,
        code: sample.code,
        artifactId: null,
        revision: 1,
        codeRevision: 1,
        proposals: [],
      });
      expect(figure.objects.map((object) => object.kind)).toEqual(sample.kinds);
      expect(
        figure.objects.every(
          (object) =>
            object.codeRange.start < object.codeRange.end &&
            object.codeRange.end <= sample.code.length,
        ),
      ).toBe(true);
      expect(fixture.context.science.getWorkspace(fixture.sessionA).figures).toEqual([figure]);
    },
  );

  it("links only a workspace-local registered figure artifact", async () => {
    const { fixture, project } = await setup();
    writeFileSync(join(fixture.workspaceA, "figure.png"), "rendered figure bytes");
    const artifact = fixture.context.science.registerArtifact(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      relativePath: "figure.png",
      kind: "figure",
      title: "Rendered result",
      mime: "image/png",
      runId: null,
      environment: {},
      license: null,
      sourceEntityIds: [],
    });

    const figure = fixture.context.science.createFigure(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Registered result",
      library: "matplotlib",
      code: "plt.plot(x, y)",
      artifactId: artifact.id,
    });
    expect(figure.artifactId).toBe(artifact.id);
    const dataset = fixture.context.science.registerArtifact(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      relativePath: "figure.png",
      kind: "dataset",
      title: "Not a rendered figure",
      mime: "application/octet-stream",
      runId: null,
      environment: {},
      license: null,
      sourceEntityIds: [],
    });
    expect(() =>
      fixture.context.science.createFigure(fixture.sessionA, {
        requestId: randomUUID(),
        projectId: project.id,
        title: "Wrong artifact kind",
        library: "matplotlib",
        code: "plt.plot(x, y)",
        artifactId: dataset.id,
      }),
    ).toThrowError(
      expect.objectContaining<Partial<ScienceError>>({ code: "FIGURE_ARTIFACT_INVALID" }),
    );

    expect(() =>
      fixture.context.science.createFigure(fixture.sessionB, {
        requestId: randomUUID(),
        projectId: project.id,
        title: "Cross-workspace result",
        library: "matplotlib",
        code: "plt.plot(x, y)",
        artifactId: artifact.id,
      }),
    ).toThrowError(expect.objectContaining<Partial<ScienceError>>({ code: "PROJECT_NOT_FOUND" }));
  });

  it("proposes an object-linked matplotlib patch without changing code", async () => {
    const { fixture, project } = await setup();
    const code = 'plt.plot(x, y)\nplt.legend(loc="best")\nplt.xlabel("Time")';
    const figure = fixture.context.science.createFigure(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Legend edit",
      library: "matplotlib",
      code,
      artifactId: null,
    });
    const legend = figure.objects.find((object) => object.kind === "legend");
    if (!legend) throw new Error("Expected an inferred legend object");

    const proposed = fixture.context.science.modifyFigureCode(fixture.sessionA, {
      requestId: randomUUID(),
      figureId: figure.id,
      expectedRevision: figure.revision,
      action: "propose",
      objectIds: [legend.id],
      proposedCode: "plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8)",
      instruction: "Move the legend outside and reduce size",
      reasoning: "The legend overlaps the data series.",
    });

    expect(proposed.code).toBe(code);
    expect(proposed.revision).toBe(2);
    expect(proposed.codeRevision).toBe(1);
    expect(proposed.proposals[0]).toMatchObject({
      objectIds: [legend.id],
      selection: legend.codeRange,
      originalCode: 'plt.legend(loc="best")',
      proposedCode: "plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8)",
      status: "pending",
      reasoning: { classification: "proposal", summary: "The legend overlaps the data series." },
    });
  });

  it("honors a pre-aborted figure mutation before appending", async () => {
    const { fixture, project } = await setup();
    const controller = new AbortController();
    controller.abort();

    expect(() =>
      fixture.context.science.createFigure(
        fixture.sessionA,
        {
          requestId: randomUUID(),
          projectId: project.id,
          title: "Cancelled figure",
          library: "matplotlib",
          code: "plt.plot(x, y)",
          artifactId: null,
        },
        controller.signal,
      ),
    ).toThrowError(expect.objectContaining({ name: "AbortError" }));
    expect(fixture.context.science.journalCount()).toBe(1);
  });

  it("accepts a semantic patch and shifts later object ranges by its UTF-16 delta", async () => {
    const { fixture, project } = await setup();
    const code = 'plt.plot(x, y)\nplt.legend(loc="best")\nplt.xlabel("Time")';
    const figure = fixture.context.science.createFigure(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Range remap",
      library: "matplotlib",
      code,
      artifactId: null,
    });
    const legend = figure.objects.find((object) => object.kind === "legend");
    const axis = figure.objects.find((object) => object.kind === "axis");
    if (!legend || !axis) throw new Error("Expected legend and axis objects");
    const replacement = "plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8)";
    const proposed = fixture.context.science.modifyFigureCode(fixture.sessionA, {
      requestId: randomUUID(),
      figureId: figure.id,
      expectedRevision: 1,
      action: "propose",
      objectIds: [legend.id],
      proposedCode: replacement,
      instruction: "Move the legend",
      reasoning: "The canvas selection identifies the legend call.",
    });
    const proposalId = proposed.proposals[0]?.id;
    if (!proposalId) throw new Error("Expected a figure patch proposal");
    const acceptedRequest = {
      requestId: randomUUID(),
      figureId: figure.id,
      expectedRevision: proposed.revision,
      action: "accept" as const,
      proposalId,
    };

    const accepted = fixture.context.science.modifyFigureCode(fixture.sessionA, acceptedRequest);
    const repeated = fixture.context.science.modifyFigureCode(fixture.sessionA, acceptedRequest);
    const acceptedLegend = accepted.objects.find((object) => object.id === legend.id);
    const acceptedAxis = accepted.objects.find((object) => object.id === axis.id);
    const delta = replacement.length - (legend.codeRange.end - legend.codeRange.start);

    expect(repeated).toEqual(accepted);
    expect(accepted.code).toContain(replacement);
    expect(accepted.codeRevision).toBe(2);
    expect(acceptedLegend?.codeRange).toEqual({
      start: legend.codeRange.start,
      end: legend.codeRange.start + replacement.length,
    });
    expect(acceptedAxis?.codeRange.start).toBe(axis.codeRange.start + delta);
    expect(accepted.proposals[0]?.status).toBe("accepted");
    expect(fixture.context.science.journalCount()).toBe(4);
    const trace = fixture.context.science.traceProvenance(fixture.sessionA, {
      entityId: accepted.id,
      maxDepth: 20,
    });
    expect(trace.entities.map(({ id, kind }) => ({ id, kind }))).toEqual([
      { id: accepted.id, kind: "figure" },
      { id: project.id, kind: "project" },
    ]);
    expect(trace.events.map((event) => event.operation)).toEqual([
      "project/created",
      "figure/created",
      "figure/modified",
      "figure/modified",
    ]);
  });

  it("rejects a multi-object patch that crosses an unselected semantic object", async () => {
    const { fixture, project } = await setup();
    const figure = fixture.context.science.createFigure(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Ambiguous brush",
      library: "matplotlib",
      code: 'plt.plot(x, y)\nplt.legend()\nplt.xlabel("Time")',
      artifactId: null,
    });
    const line = figure.objects.find((object) => object.kind === "line");
    const axis = figure.objects.find((object) => object.kind === "axis");
    if (!line || !axis) throw new Error("Expected line and axis objects");

    expect(() =>
      fixture.context.science.modifyFigureCode(fixture.sessionA, {
        requestId: randomUUID(),
        figureId: figure.id,
        expectedRevision: 1,
        action: "propose",
        objectIds: [line.id, axis.id],
        proposedCode: "# replace selected objects",
        instruction: "Change separated objects",
        reasoning: "The brush skipped the legend between them.",
      }),
    ).toThrowError(
      expect.objectContaining<Partial<ScienceError>>({ code: "FIGURE_SELECTION_AMBIGUOUS" }),
    );
  });

  it("applies a ggplot2 semantic patch and rejects stale revisions", async () => {
    const { fixture, project } = await setup();
    const figure = fixture.context.science.createFigure(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      title: "ggplot legend",
      library: "ggplot2",
      code: 'ggplot(df, aes(x, y)) + geom_point() +\n  theme(legend.position="right")',
      artifactId: null,
    });
    const legend = figure.objects.find((object) => object.kind === "legend");
    if (!legend) throw new Error("Expected a ggplot2 legend object");
    const proposed = fixture.context.science.modifyFigureCode(fixture.sessionA, {
      requestId: randomUUID(),
      figureId: figure.id,
      expectedRevision: 1,
      action: "propose",
      objectIds: [legend.id],
      proposedCode: '  theme(legend.position="bottom", legend.text=element_text(size=8))',
      instruction: "Move the legend below",
      reasoning: "The horizontal layout has more room below the chart.",
    });
    const proposalId = proposed.proposals[0]?.id;
    if (!proposalId) throw new Error("Expected a ggplot2 proposal");

    expect(() =>
      fixture.context.science.modifyFigureCode(fixture.sessionA, {
        requestId: randomUUID(),
        figureId: figure.id,
        expectedRevision: 1,
        action: "accept",
        proposalId,
      }),
    ).toThrowError(expect.objectContaining<Partial<ScienceError>>({ code: "REVISION_CONFLICT" }));
  });

  it("replays figure creation and accepted code revisions", async () => {
    const { fixture, project } = await setup();
    const figure = fixture.context.science.createFigure(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Replayable figure",
      library: "plotly",
      code: "fig.add_scatter(x=x, y=y)",
      artifactId: null,
    });
    const object = figure.objects[0];
    if (!object) throw new Error("Expected an inferred plotly object");
    const proposed = fixture.context.science.modifyFigureCode(fixture.sessionA, {
      requestId: randomUUID(),
      figureId: figure.id,
      expectedRevision: 1,
      action: "propose",
      objectIds: [object.id],
      proposedCode: 'fig.add_scatter(x=x, y=y, mode="lines+markers")',
      instruction: "Show markers",
      reasoning: "Markers expose each observed sample.",
    });
    const proposalId = proposed.proposals[0]?.id;
    if (!proposalId) throw new Error("Expected a replayable figure proposal");
    const accepted = fixture.context.science.modifyFigureCode(fixture.sessionA, {
      requestId: randomUUID(),
      figureId: figure.id,
      expectedRevision: 2,
      action: "accept",
      proposalId,
    });
    await fixture.scienceFiber.dispose();
    const database = new DatabaseSync(fixture.databasePath);
    database.exec("DELETE FROM science_figures");
    database.close();

    await fixture.remount();

    expect(fixture.context.science.getWorkspace(fixture.sessionA).figures).toEqual([accepted]);
  });

  it("migrates a v3 database before accepting figure facts", async () => {
    const { fixture, project } = await setup();
    await fixture.scienceFiber.dispose();
    const v3 = new DatabaseSync(fixture.databasePath);
    v3.exec("DELETE FROM science_migrations WHERE version = 4; DROP TABLE science_figures;");
    v3.close();

    await fixture.remount();
    const figure = fixture.context.science.createFigure(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Migrated figure",
      library: "seaborn",
      code: "sns.scatterplot(data=df, x='x', y='y')",
      artifactId: null,
    });
    const migrated = new DatabaseSync(fixture.databasePath, { readOnly: true });
    const version = migrated
      .prepare("SELECT MAX(version) AS version FROM science_migrations")
      .get() as {
      version: number;
    };
    migrated.close();

    expect(version.version).toBe(5);
    expect(fixture.context.science.getWorkspace(fixture.sessionA).figures).toEqual([figure]);
  });
});
