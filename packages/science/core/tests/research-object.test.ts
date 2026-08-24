import { randomUUID } from "node:crypto";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  type RoCrateEntity,
  type RoCrateMetadataDocument,
  roCrateEntityId,
  roCrateMetadataDocumentSchema,
} from "../src/index.js";
import { createScienceFixture, type ScienceFixture } from "./fixture.js";

const fixtures: ScienceFixture[] = [];

afterEach(async () => {
  await Promise.all(fixtures.splice(0).map((fixture) => fixture.dispose()));
});

function graphEntity(crate: RoCrateMetadataDocument, id: string): RoCrateEntity {
  const entity = crate["@graph"].find((candidate) => candidate["@id"] === id);
  if (!entity) throw new Error(`Missing RO-Crate entity ${id}`);
  return entity;
}

function entityTypes(entity: RoCrateEntity): readonly string[] {
  const type = entity["@type"];
  return Array.isArray(type) ? type : [type];
}

describe("RO-Crate 1.3 Research Object", () => {
  it("projects one Science project into standard entities and provenance actions", async () => {
    const fixture = await createScienceFixture();
    fixtures.push(fixture);
    const project = fixture.context.science.createProject(fixture.sessionA, {
      requestId: randomUUID(),
      title: "Recovery study",
    });
    const notebook = fixture.context.science.createNotebook(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Analysis",
    });
    writeFileSync(join(fixture.workspaceA, "result.txt"), "result");
    const artifact = await fixture.context.science.registerArtifact(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      relativePath: "result.txt",
      kind: "log",
      title: "Result log",
      mime: "text/plain",
      runId: null,
      environment: { API_TOKEN: "never-export" },
      license: null,
      sourceEntityIds: [notebook.id],
    });
    const question = fixture.context.science.createQuestion(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Does recovery improve?",
      summary: "Compare recovery against baseline.",
      tags: ["recovery"],
    });
    const hypothesis = fixture.context.science.createHypothesis(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      questionId: question.id,
      title: "Recovery improves",
      summary: "Recovery is faster after treatment.",
      tags: [],
    });
    const claim = fixture.context.science.recordClaim(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      hypothesisId: hypothesis.id,
      title: "Recovery improved by 12%",
      summary: "Observed in the held-out cohort.",
      status: "supported",
      tags: [],
    });
    const linked = fixture.context.science.linkEvidence(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      claimId: claim.id,
      relation: "supports",
      title: "Held-out cohort analysis",
      summary: "The confidence interval excludes baseline.",
      sourceEntityIds: [artifact.id],
      tags: ["statistical"],
    });

    const crate = fixture.context.science.getResearchObject(fixture.sessionA, {
      projectId: project.id,
    });
    expect(roCrateMetadataDocumentSchema.parse(crate)).toEqual(crate);
    expect(crate["@context"]).toBe("https://w3id.org/ro/crate/1.3/context");

    const rootId = roCrateEntityId(project.id);
    expect(graphEntity(crate, "ro-crate-metadata.json")).toMatchObject({
      "@type": "CreativeWork",
      about: { "@id": rootId },
      conformsTo: { "@id": "https://w3id.org/ro/crate/1.3" },
    });
    expect(graphEntity(crate, rootId)).toMatchObject({
      "@type": "Dataset",
      name: "Recovery study",
      license: "All rights reserved",
      hasPart: expect.arrayContaining([
        { "@id": roCrateEntityId(notebook.id) },
        { "@id": roCrateEntityId(artifact.id) },
        { "@id": roCrateEntityId(claim.id) },
      ]),
    });
    expect(entityTypes(graphEntity(crate, roCrateEntityId(notebook.id)))).toContain(
      "SoftwareSourceCode",
    );
    expect(graphEntity(crate, roCrateEntityId(artifact.id))).toMatchObject({
      "@type": expect.arrayContaining(["MediaObject"]),
      encodingFormat: "text/plain",
      sha256: artifact.digest.slice("sha256:".length),
      isBasedOn: [{ "@id": roCrateEntityId(notebook.id) }],
    });
    expect(entityTypes(graphEntity(crate, roCrateEntityId(question.id)))).toContain("Question");
    expect(graphEntity(crate, roCrateEntityId(hypothesis.id))).toMatchObject({
      "@type": "CreativeWork",
      additionalType: "Hypothesis",
      isBasedOn: [{ "@id": roCrateEntityId(question.id) }],
    });
    expect(entityTypes(graphEntity(crate, roCrateEntityId(claim.id)))).toContain("Claim");
    expect(graphEntity(crate, roCrateEntityId(linked.evidence.id))).toMatchObject({
      "@type": "Review",
      itemReviewed: { "@id": roCrateEntityId(claim.id) },
      reviewRating: { "@id": `#rating-${linked.evidence.id}` },
    });
    expect(graphEntity(crate, `#rating-${linked.evidence.id}`)).toMatchObject({
      "@type": "Rating",
      name: "supports",
      ratingValue: 1,
      bestRating: 1,
      worstRating: -1,
    });
    expect(graphEntity(crate, `#action-${artifact.provenance.eventId}`)).toMatchObject({
      "@type": "CreateAction",
      object: [{ "@id": roCrateEntityId(notebook.id) }],
      result: [{ "@id": roCrateEntityId(artifact.id) }],
    });

    const serialized = JSON.stringify(crate);
    expect(serialized).not.toContain("fromId");
    expect(serialized).not.toContain("toId");
    expect(serialized).not.toContain("sessionId");
    expect(serialized).not.toContain("never-export");
    expect(serialized).not.toContain(fixture.workspaceA);
  });

  it("rejects duplicate ids, dangling local references, and cross-workspace projects", async () => {
    const fixture = await createScienceFixture();
    fixtures.push(fixture);
    const project = fixture.context.science.createProject(fixture.sessionA, {
      requestId: randomUUID(),
      title: "Bounded crate",
    });
    const crate = fixture.context.science.getResearchObject(fixture.sessionA, {
      projectId: project.id,
    });
    expect(() =>
      roCrateMetadataDocumentSchema.parse({
        ...crate,
        "@graph": [...crate["@graph"], crate["@graph"][0]],
      }),
    ).toThrow();
    expect(() =>
      roCrateMetadataDocumentSchema.parse({
        ...crate,
        "@graph": crate["@graph"].map((entity) =>
          entity["@id"] === roCrateEntityId(project.id)
            ? { ...entity, hasPart: [{ "@id": "urn:uuid:missing" }] }
            : entity,
        ),
      }),
    ).toThrow(/has no entity/u);
    expect(() =>
      roCrateMetadataDocumentSchema.parse({
        ...crate,
        "@graph": crate["@graph"].map((entity) =>
          entity["@id"] === "ro-crate-metadata.json"
            ? { ...entity, conformsTo: undefined }
            : entity,
        ),
      }),
    ).toThrow(/Metadata Descriptor/u);
    expect(() =>
      roCrateMetadataDocumentSchema.parse({
        ...crate,
        "@graph": crate["@graph"].map((entity) =>
          entity["@id"] === roCrateEntityId(project.id)
            ? {
                ...entity,
                hasPart: [{ "@id": "urn:uuid:inline", "@type": "CreativeWork" }],
              }
            : entity,
        ),
      }),
    ).toThrow();
    expect(() =>
      fixture.context.science.getResearchObject(fixture.sessionB, { projectId: project.id }),
    ).toThrowError(expect.objectContaining({ code: "PROJECT_NOT_FOUND" }));
  });
});
