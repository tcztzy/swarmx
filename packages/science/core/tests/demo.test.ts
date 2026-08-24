import { afterEach, describe, expect, it } from "vitest";
import { runScienceDemo } from "../src/demo.js";
import { createScienceFixture, type ScienceFixture } from "./fixture.js";

const fixtures: ScienceFixture[] = [];

afterEach(async () => {
  await Promise.all(fixtures.splice(0).map((fixture) => fixture.dispose()));
});

describe("T18 runnable Science IDE demo", () => {
  it("completes notebook, experiment, figure, writing, Research Object, and export", async () => {
    const fixture = await createScienceFixture();
    fixtures.push(fixture);

    const demo = await runScienceDemo(fixture.context.science, fixture.sessionA);

    expect(demo.run).toMatchObject({ status: "succeeded", metrics: { improvementPercent: 12 } });
    expect(demo.figure.code).toContain("bbox_to_anchor");
    expect(demo.document.content).toContain("12%");
    expect(demo.claim.status).toBe("supported");
    expect(demo.researchObject["@graph"].map((entity) => entity["@id"])).toContain(
      `urn:uuid:${demo.run.id}`,
    );
    expect(demo.exported.counts).toMatchObject({
      projects: 1,
      notebooks: 1,
      figures: 1,
      documents: 1,
      experiments: 1,
      runs: 1,
    });
    expect(demo.exported.content).not.toContain(fixture.workspaceA);
  });
});
