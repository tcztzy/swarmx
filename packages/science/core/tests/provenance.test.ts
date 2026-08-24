import { randomUUID } from "node:crypto";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { roCrateEntityId, type ScienceError } from "../src/index.js";
import { createScienceFixture, type ScienceFixture } from "./fixture.js";

const fixtures: ScienceFixture[] = [];

afterEach(async () => {
  await Promise.all(fixtures.splice(0).map((fixture) => fixture.dispose()));
});

describe("Science RO-Crate provenance", () => {
  it("projects artifact lineage through standard references without exposing host paths", async () => {
    const current = await createScienceFixture();
    fixtures.push(current);
    const project = current.context.science.createProject(current.sessionA, {
      requestId: randomUUID(),
      title: "Lineage project",
    });
    const notebook = current.context.science.createNotebook(current.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Source notebook",
    });
    const relativePath = "lineage-secret.bin";
    writeFileSync(join(current.workspaceA, relativePath), "lineage bytes");
    const artifact = await current.context.science.registerArtifact(current.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      relativePath,
      kind: "model",
      title: "Fitted model",
      mime: "application/octet-stream",
      runId: null,
      environment: {},
      license: null,
      sourceEntityIds: [notebook.id],
    });

    const researchObject = current.context.science.getResearchObject(current.sessionA, {
      projectId: project.id,
    });
    const artifactEntity = researchObject["@graph"].find(
      (entity) => entity["@id"] === roCrateEntityId(artifact.id),
    );
    expect(artifactEntity).toMatchObject({
      "@type": ["MediaObject"],
      name: "Fitted model",
      isBasedOn: [{ "@id": roCrateEntityId(notebook.id) }],
      isPartOf: { "@id": roCrateEntityId(project.id) },
    });
    expect(researchObject["@graph"]).toContainEqual(
      expect.objectContaining({
        "@type": "CreateAction",
        name: "Register research artifact",
        object: [{ "@id": roCrateEntityId(notebook.id) }],
        result: [{ "@id": roCrateEntityId(artifact.id) }],
      }),
    );
    expect(JSON.stringify(researchObject)).not.toContain(current.workspaceA);
    expect(JSON.stringify(researchObject)).not.toContain(relativePath);
  });

  it("rejects cross-workspace source links and Research Object roots", async () => {
    const current = await createScienceFixture();
    fixtures.push(current);
    const projectA = current.context.science.createProject(current.sessionA, {
      requestId: randomUUID(),
      title: "Workspace A",
    });
    const projectB = current.context.science.createProject(current.sessionB, {
      requestId: randomUUID(),
      title: "Workspace B",
    });
    writeFileSync(join(current.workspaceA, "cross.bin"), "cross workspace");

    await expect(
      current.context.science.registerArtifact(current.sessionA, {
        requestId: randomUUID(),
        projectId: projectA.id,
        relativePath: "cross.bin",
        kind: "code",
        title: "Cross source",
        mime: "text/plain",
        runId: null,
        environment: {},
        license: null,
        sourceEntityIds: [projectB.id],
      }),
    ).rejects.toMatchObject(
      expect.objectContaining<Partial<ScienceError>>({ code: "PROVENANCE_ENTITY_NOT_FOUND" }),
    );
    expect(() =>
      current.context.science.getResearchObject(current.sessionA, {
        projectId: projectB.id,
      }),
    ).toThrowError(expect.objectContaining<Partial<ScienceError>>({ code: "PROJECT_NOT_FOUND" }));
    expect(current.context.science.journalCount()).toBe(2);
  });

  it("links a document and its revision facts to the owning project", async () => {
    const current = await createScienceFixture();
    fixtures.push(current);
    const project = current.context.science.createProject(current.sessionA, {
      requestId: randomUUID(),
      title: "Writing lineage",
    });
    const document = current.context.science.createDocument(current.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      name: "paper.typ",
      content: "= Result",
    });
    const revised = current.context.science.modifyDocument(current.sessionA, {
      requestId: randomUUID(),
      documentId: document.id,
      expectedRevision: 1,
      action: "propose",
      selection: { start: 2, end: 8 },
      proposedText: "Results",
      instruction: "Use a plural heading",
      reasoning: "The section reports multiple findings.",
    });

    const researchObject = current.context.science.getResearchObject(current.sessionA, {
      projectId: project.id,
    });
    expect(researchObject["@graph"]).toContainEqual(
      expect.objectContaining({
        "@id": roCrateEntityId(revised.id),
        "@type": "DigitalDocument",
        isPartOf: { "@id": roCrateEntityId(project.id) },
      }),
    );
    expect(researchObject["@graph"]).toContainEqual(
      expect.objectContaining({
        "@type": "UpdateAction",
        name: "Update writing document",
        result: [{ "@id": roCrateEntityId(revised.id) }],
      }),
    );
  });
});
