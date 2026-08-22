import { randomUUID } from "node:crypto";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { ScienceError } from "../src/index.js";
import { createScienceFixture, type ScienceFixture } from "./fixture.js";

const fixtures: ScienceFixture[] = [];

afterEach(async () => {
  await Promise.all(fixtures.splice(0).map((fixture) => fixture.dispose()));
});

describe("Science provenance trace", () => {
  it("traces bounded artifact lineage from Journal facts without exposing host paths", async () => {
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
    const artifact = current.context.science.registerArtifact(current.sessionA, {
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

    const trace = current.context.science.traceProvenance(current.sessionA, {
      entityId: artifact.id,
      maxDepth: 20,
    });
    expect(trace.rootId).toBe(artifact.id);
    expect(trace.entities.map(({ id, kind }) => ({ id, kind }))).toEqual([
      { id: artifact.id, kind: "artifact" },
      { id: notebook.id, kind: "notebook" },
      { id: project.id, kind: "project" },
    ]);
    expect(trace.relations).toEqual([
      { fromId: artifact.id, toId: notebook.id, type: "derived_from" },
      { fromId: notebook.id, toId: project.id, type: "derived_from" },
    ]);
    expect(trace.events.map((event) => event.operation)).toEqual([
      "project/created",
      "notebook/created",
      "artifact/registered",
    ]);
    expect(trace.truncated).toBe(false);
    expect(JSON.stringify(trace)).not.toContain(current.workspaceA);
    expect(JSON.stringify(trace)).not.toContain(relativePath);

    const bounded = current.context.science.traceProvenance(current.sessionA, {
      entityId: artifact.id,
      maxDepth: 1,
    });
    expect(bounded.entities.map((entity) => entity.id)).toEqual([artifact.id, notebook.id]);
    expect(bounded.truncated).toBe(true);
  });

  it("rejects cross-workspace source links and trace roots", async () => {
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

    expect(() =>
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
    ).toThrowError(
      expect.objectContaining<Partial<ScienceError>>({ code: "PROVENANCE_ENTITY_NOT_FOUND" }),
    );
    expect(() =>
      current.context.science.traceProvenance(current.sessionA, {
        entityId: projectB.id,
        maxDepth: 20,
      }),
    ).toThrowError(
      expect.objectContaining<Partial<ScienceError>>({ code: "PROVENANCE_ENTITY_NOT_FOUND" }),
    );
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

    const trace = current.context.science.traceProvenance(current.sessionA, {
      entityId: revised.id,
      maxDepth: 20,
    });

    expect(trace.entities.map(({ id, kind }) => ({ id, kind }))).toEqual([
      { id: document.id, kind: "document" },
      { id: project.id, kind: "project" },
    ]);
    expect(trace.relations).toEqual([
      { fromId: document.id, toId: project.id, type: "derived_from" },
    ]);
    expect(trace.events.map((event) => event.operation)).toEqual([
      "project/created",
      "document/created",
      "document/modified",
    ]);
  });
});
