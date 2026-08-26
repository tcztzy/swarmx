import { randomUUID } from "node:crypto";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import { createScienceFixture, type ScienceFixture } from "./fixture.js";

const fixtures: ScienceFixture[] = [];

afterEach(async () => {
  await Promise.all(fixtures.splice(0).map((fixture) => fixture.dispose()));
});

async function setup() {
  const fixture = await createScienceFixture({ maxExportBytes: 256_000 });
  fixtures.push(fixture);
  const project = fixture.context.science.createProject(fixture.sessionA, {
    requestId: randomUUID(),
    title: "Reproducible treatment study",
  });
  return { fixture, project };
}

describe("T18 research facts and experiment ledger", () => {
  it("records a typed question, hypothesis, claim, and supporting evidence", async () => {
    const { fixture, project } = await setup();
    const question = fixture.context.science.createQuestion(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Does treatment improve recovery?",
      summary: "Compare recovery time with the baseline cohort.",
      tags: ["clinical", "recovery"],
    });
    const hypothesis = fixture.context.science.createHypothesis(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      questionId: question.id,
      title: "Treatment shortens recovery",
      summary: "Median recovery time is lower after treatment.",
      tags: ["primary"],
    });
    const claim = fixture.context.science.recordClaim(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      hypothesisId: hypothesis.id,
      title: "Recovery improved by 12%",
      summary: "Observed effect in the held-out cohort.",
      status: "proposed",
      tags: [],
    });
    const linked = fixture.context.science.linkEvidence(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      claimId: claim.id,
      relation: "supports",
      title: "Held-out cohort analysis",
      summary: "Confidence interval excludes the baseline median.",
      sourceEntityIds: [hypothesis.id],
      tags: ["statistical"],
    });

    expect(question).toMatchObject({ kind: "question", status: "open", revision: 1 });
    expect(hypothesis).toMatchObject({
      kind: "hypothesis",
      sourceEntityIds: [question.id],
    });
    expect(claim).toMatchObject({ kind: "claim", sourceEntityIds: [hypothesis.id] });
    expect(linked).toMatchObject({
      evidence: { kind: "evidence", sourceEntityIds: [hypothesis.id] },
      relation: { fromId: linked.evidence.id, toId: claim.id, type: "supports" },
    });
    const workspace = fixture.context.science.getWorkspace(fixture.sessionA);
    expect(workspace.records).toHaveLength(4);
    expect(workspace.relations).toHaveLength(3);
    expect(workspace.relations.map((relation) => relation.type)).toEqual(
      expect.arrayContaining(["motivated_by", "derived_from", "supports"]),
    );

    const researchObject = fixture.context.science.getResearchObject(fixture.sessionA, {
      projectId: project.id,
    });
    expect(researchObject["@graph"]).toContainEqual(
      expect.objectContaining({
        "@id": `urn:uuid:${linked.evidence.id}`,
        "@type": "Review",
        itemReviewed: { "@id": `urn:uuid:${claim.id}` },
      }),
    );
    expect(researchObject["@graph"]).toContainEqual(
      expect.objectContaining({
        "@type": "CreateAction",
        result: [{ "@id": `urn:uuid:${linked.evidence.id}` }],
      }),
    );
  });

  it("runs one experiment lifecycle, compares completed runs, and survives replay", async () => {
    const { fixture, project } = await setup();
    const experiment = fixture.context.science.defineExperiment(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Recovery benchmark",
      summary: "Two seeded repetitions.",
      hypothesisIds: [],
      protocol: "python analysis.py --seed $SEED",
      tags: ["benchmark"],
    });
    const firstStarted = fixture.context.science.startRun(fixture.sessionA, {
      requestId: randomUUID(),
      experimentId: experiment.id,
      expectedRevision: 1,
      environment: { seed: "1", API_TOKEN: "must-not-persist", cwd: fixture.workspaceA },
    });
    const first = fixture.context.science.finishRun(fixture.sessionA, {
      requestId: randomUUID(),
      runId: firstStarted.run.id,
      expectedRevision: 1,
      status: "succeeded",
      metrics: { accuracy: 0.8, loss: 0.4 },
      artifactIds: [],
      notes: "baseline",
    });
    const secondStarted = fixture.context.science.startRun(fixture.sessionA, {
      requestId: randomUUID(),
      experimentId: experiment.id,
      expectedRevision: firstStarted.experiment.revision,
      environment: { seed: "2" },
    });
    const second = fixture.context.science.finishRun(fixture.sessionA, {
      requestId: randomUUID(),
      runId: secondStarted.run.id,
      expectedRevision: 1,
      status: "succeeded",
      metrics: { accuracy: 0.86, loss: 0.35 },
      artifactIds: [],
      notes: "replication",
    });

    expect(first.environment).toMatchObject({
      seed: "1",
      API_TOKEN: "[redacted]",
      cwd: "[redacted]",
    });
    expect(first.finishedAt).not.toBeNull();
    expect(second.revision).toBe(2);
    const comparison = fixture.context.science.compareRuns(fixture.sessionA, {
      runIds: [first.id, second.id],
    });
    expect(comparison).toEqual(
      expect.objectContaining({
        experimentId: experiment.id,
        baselineRunId: first.id,
        classification: "inference",
        deltas: [
          { metric: "accuracy", values: [0, 0.05999999999999994] },
          { metric: "loss", values: [0, -0.050000000000000044] },
        ],
      }),
    );
    expect(fixture.context.science.journalCount()).toBe(6);
    const researchObject = fixture.context.science.getResearchObject(fixture.sessionA, {
      projectId: project.id,
    });
    expect(researchObject["@graph"]).toContainEqual(
      expect.objectContaining({
        "@id": `urn:uuid:${first.id}`,
        "@type": "CreateAction",
        object: { "@id": `urn:uuid:${experiment.id}` },
        instrument: { "@id": "https://github.com/tcztzy/swarmx" },
        actionStatus: { "@id": "https://schema.org/CompletedActionStatus" },
      }),
    );
    expect(JSON.stringify(researchObject)).not.toContain("must-not-persist");

    await fixture.scienceFiber.dispose();
    await fixture.remount();
    expect(fixture.context.science.getWorkspace(fixture.sessionA)).toMatchObject({
      experiments: [{ id: experiment.id, revision: 3, runIds: [first.id, second.id] }],
      runs: [
        { id: first.id, status: "succeeded", revision: 2 },
        { id: second.id, status: "succeeded", revision: 2 },
      ],
    });
    const database = new DatabaseSync(fixture.databasePath);
    expect(
      database.prepare("SELECT MAX(version) AS version FROM science_migrations").get(),
    ).toEqual({ version: 5 });
    database.close();
  });

  it("rejects stale lifecycle revisions, cross-workspace links, and pre-aborted writes", async () => {
    const { fixture, project } = await setup();
    const experiment = fixture.context.science.defineExperiment(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Revision gate",
      summary: "Exact revision lifecycle.",
      hypothesisIds: [],
      protocol: "record only",
      tags: [],
    });
    fixture.context.science.startRun(fixture.sessionA, {
      requestId: randomUUID(),
      experimentId: experiment.id,
      expectedRevision: 1,
      environment: {},
    });
    expect(() =>
      fixture.context.science.startRun(fixture.sessionA, {
        requestId: randomUUID(),
        experimentId: experiment.id,
        expectedRevision: 1,
        environment: {},
      }),
    ).toThrowError(expect.objectContaining({ code: "REVISION_CONFLICT" }));

    const foreignProject = fixture.context.science.createProject(fixture.sessionB, {
      requestId: randomUUID(),
      title: "Foreign",
    });
    expect(() =>
      fixture.context.science.defineExperiment(fixture.sessionB, {
        requestId: randomUUID(),
        projectId: foreignProject.id,
        title: "Foreign experiment",
        summary: "Cross-link rejected.",
        hypothesisIds: [experiment.id],
        protocol: "none",
        tags: [],
      }),
    ).toThrowError(expect.objectContaining({ code: "RESEARCH_ENTITY_NOT_FOUND" }));

    const controller = new AbortController();
    controller.abort();
    expect(() =>
      fixture.context.science.defineExperiment(
        fixture.sessionA,
        {
          requestId: randomUUID(),
          projectId: project.id,
          title: "Cancelled",
          summary: "Must not append.",
          hypothesisIds: [],
          protocol: "none",
          tags: [],
        },
        controller.signal,
      ),
    ).toThrowError(expect.objectContaining({ name: "AbortError" }));
    expect(fixture.context.science.journalCount()).toBe(4);
  });
});

describe("T18 project export", () => {
  it("stores deterministic bytes outside Journal and repeats the exact payload", async () => {
    const { fixture, project } = await setup();
    fixture.context.science.createNotebook(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      title: "Analysis",
    });
    const request = { requestId: randomUUID(), projectId: project.id };
    const researchObject = fixture.context.science.getResearchObject(fixture.sessionA, {
      projectId: project.id,
    });

    const exported = fixture.context.science.exportProject(fixture.sessionA, request);
    const repeated = fixture.context.science.exportProject(fixture.sessionA, request);
    expect(repeated).toEqual(exported);
    expect(exported).toMatchObject({
      format: "ro-crate@1.3",
      filename: "ro-crate-metadata.json",
      mediaType: "application/ld+json",
      projectId: project.id,
      classification: "fact",
      digest: expect.stringMatching(/^sha256:[0-9a-f]{64}$/u),
      bytes: expect.any(Number),
      counts: { projects: 1, notebooks: 1 },
    });
    expect(JSON.parse(exported.content)).toEqual(researchObject);
    expect(exported.content).not.toContain(fixture.workspaceA);
    const database = new DatabaseSync(fixture.databasePath);
    const row = database
      .prepare("SELECT type, payload_json FROM science_journal WHERE type = 'project/exported'")
      .get() as { type: string; payload_json: string };
    expect(row.type).toBe("project/exported");
    expect(row.payload_json).not.toContain(exported.content);
    expect(fixture.context.science.getWorkspace(fixture.sessionA).exports).toHaveLength(1);
    database.close();
  });

  it("replays immutable pre-RO-Crate export records without making them the current format", async () => {
    const { fixture, project } = await setup();
    const exported = fixture.context.science.exportProject(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
    });
    if (exported.format !== "ro-crate@1.3") throw new Error("Expected current export format");
    const legacy = {
      id: exported.id,
      projectId: exported.projectId,
      kind: exported.kind,
      format: "dsh-science-project@1",
      digest: exported.digest,
      bytes: exported.bytes,
      counts: exported.counts,
      createdAt: exported.createdAt,
      revision: exported.revision,
      provenance: exported.provenance,
    };
    const database = new DatabaseSync(fixture.databasePath);
    database
      .prepare("UPDATE science_journal SET payload_json = ? WHERE type = 'project/exported'")
      .run(JSON.stringify(legacy));
    database.close();

    await fixture.scienceFiber.dispose();
    await fixture.remount();
    expect(fixture.context.science.getWorkspace(fixture.sessionA).exports).toEqual([legacy]);

    const next = fixture.context.science.exportProject(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
    });
    expect(next.format).toBe("ro-crate@1.3");
  });
});
