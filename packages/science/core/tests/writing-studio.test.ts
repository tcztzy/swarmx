import { randomUUID } from "node:crypto";
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
    title: "Evidence-aware writing",
  });
  return { fixture, project };
}

describe("T16 Science Writing Studio", () => {
  it("creates a bounded Typst document with structural and scientific diagnostics", async () => {
    const { fixture, project } = await setup();
    const content = [
      "= Results",
      "Our model significantly improves accuracy.",
      "See @fig:missing for details.",
      "#emph[unfinished",
    ].join("\n");

    const document = fixture.context.science.createDocument(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      name: "paper.typ",
      content,
    });

    expect(document).toMatchObject({
      projectId: project.id,
      kind: "document",
      name: "paper.typ",
      format: "typst",
      content,
      revision: 1,
      contentRevision: 1,
      proposals: [],
    });
    expect(document.revisions).toHaveLength(1);
    expect(document.revisions[0]).toMatchObject({
      revision: 1,
      contentRevision: 1,
      reason: "created",
      proposalId: null,
    });
    expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual(
      expect.arrayContaining([
        "claim-needs-evidence",
        "figure-reference-missing",
        "unbalanced-delimiter",
      ]),
    );
    expect(document.diagnostics.every((diagnostic) => diagnostic.end <= content.length)).toBe(true);
    expect(document.diagnostics.some((diagnostic) => diagnostic.scope === "compilation")).toBe(
      false,
    );
    expect(fixture.context.science.getWorkspace(fixture.sessionA).documents).toEqual([document]);
    expect(fixture.context.science.journalCount()).toBe(2);
  });

  it.each(["paper.typ", "paper.tex", "notes.md", "references.bib"])(
    "accepts the supported logical document name %s",
    async (name) => {
      const { fixture, project } = await setup();

      const document = fixture.context.science.createDocument(fixture.sessionA, {
        requestId: randomUUID(),
        projectId: project.id,
        name,
        content: "Scientific source",
      });

      expect(document.name).toBe(name);
    },
  );

  it("reports LaTeX environment and figure-reference structure without claiming compilation", async () => {
    const { fixture, project } = await setup();
    const document = fixture.context.science.createDocument(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      name: "paper.tex",
      content: String.raw`\begin{figure}\caption{Result}\label{fig:present}\ref{fig:missing}`,
    });

    expect(document.diagnostics.map((diagnostic) => diagnostic.code)).toEqual(
      expect.arrayContaining(["unbalanced-environment", "figure-reference-missing"]),
    );
    expect(document.validation).toEqual({ structural: "checked", compilation: "not-run" });
  });

  it.each(["../paper.typ", "/tmp/paper.tex", "draft.docx", "chapter\\paper.typ"])(
    "rejects unsafe or unsupported logical document name %s",
    async (name) => {
      const { fixture, project } = await setup();

      expect(() =>
        fixture.context.science.createDocument(fixture.sessionA, {
          requestId: randomUUID(),
          projectId: project.id,
          name,
          content: "Scientific source",
        }),
      ).toThrowError(expect.objectContaining<Partial<ScienceError>>({ code: "INVALID_REQUEST" }));
      expect(fixture.context.science.journalCount()).toBe(1);
    },
  );

  it("proposes a UTF-16 source patch without changing the stored source", async () => {
    const { fixture, project } = await setup();
    const content = "😀 Our model improves performance.";
    const document = fixture.context.science.createDocument(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      name: "paper.tex",
      content,
    });
    const start = content.indexOf("improves");
    const end = start + "improves".length;

    const proposed = fixture.context.science.modifyDocument(fixture.sessionA, {
      requestId: randomUUID(),
      documentId: document.id,
      expectedRevision: document.revision,
      action: "propose",
      selection: { start, end },
      proposedText: "improves mean accuracy by 4.2% (95% CI 3.8–4.6)",
      instruction: "Quantify the claim",
      reasoning: "The current claim has no statistical support.",
    });

    expect(proposed.content).toBe(content);
    expect(proposed.revision).toBe(2);
    expect(proposed.contentRevision).toBe(1);
    expect(proposed.proposals).toHaveLength(1);
    expect(proposed.proposals[0]).toMatchObject({
      selection: { start, end },
      originalText: "improves",
      proposedText: "improves mean accuracy by 4.2% (95% CI 3.8–4.6)",
      instruction: "Quantify the claim",
      reasoning: {
        classification: "proposal",
        summary: "The current claim has no statistical support.",
      },
      status: "pending",
      resolvedAt: null,
      resolvedProvenance: null,
    });
  });

  it("accepts or rejects only a pending proposal at the exact document revision", async () => {
    const { fixture, project } = await setup();
    const document = fixture.context.science.createDocument(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      name: "paper.md",
      content: "The result is better.",
    });
    const proposal = fixture.context.science.modifyDocument(fixture.sessionA, {
      requestId: randomUUID(),
      documentId: document.id,
      expectedRevision: 1,
      action: "propose",
      selection: { start: 14, end: 20 },
      proposedText: "4.2% better",
      instruction: "Add the measured effect",
      reasoning: "A quantified effect is auditable.",
    });
    const proposalId = proposal.proposals[0]?.id;
    expect(proposalId).toBeDefined();
    if (!proposalId) throw new Error("Expected a pending proposal");

    expect(() =>
      fixture.context.science.modifyDocument(fixture.sessionA, {
        requestId: randomUUID(),
        documentId: document.id,
        expectedRevision: 1,
        action: "accept",
        proposalId,
      }),
    ).toThrowError(expect.objectContaining<Partial<ScienceError>>({ code: "REVISION_CONFLICT" }));

    const acceptedRequest = {
      requestId: randomUUID(),
      documentId: document.id,
      expectedRevision: proposal.revision,
      action: "accept" as const,
      proposalId,
    };
    const accepted = fixture.context.science.modifyDocument(fixture.sessionA, acceptedRequest);
    const repeated = fixture.context.science.modifyDocument(fixture.sessionA, acceptedRequest);

    expect(repeated).toEqual(accepted);
    expect(accepted.content).toBe("The result is 4.2% better.");
    expect(accepted.revision).toBe(3);
    expect(accepted.contentRevision).toBe(2);
    expect(accepted.proposals[0]).toMatchObject({
      status: "accepted",
      resolvedProvenance: accepted.provenance,
    });
    expect(accepted.revisions.at(-1)).toMatchObject({
      revision: 3,
      contentRevision: 2,
      reason: "proposal-accepted",
      proposalId,
    });
    expect(fixture.context.science.journalCount()).toBe(4);

    const second = fixture.context.science.modifyDocument(fixture.sessionA, {
      requestId: randomUUID(),
      documentId: document.id,
      expectedRevision: accepted.revision,
      action: "propose",
      selection: { start: 0, end: 3 },
      proposedText: "This",
      instruction: "Clarify",
      reasoning: "The antecedent can be explicit.",
    });
    const secondProposalId = second.proposals.at(-1)?.id;
    if (!secondProposalId) throw new Error("Expected a second pending proposal");
    const rejected = fixture.context.science.modifyDocument(fixture.sessionA, {
      requestId: randomUUID(),
      documentId: document.id,
      expectedRevision: second.revision,
      action: "reject",
      proposalId: secondProposalId,
    });
    expect(rejected.content).toBe(accepted.content);
    expect(rejected.contentRevision).toBe(accepted.contentRevision);
    expect(rejected.proposals.at(-1)?.status).toBe("rejected");
  });

  it("isolates document ownership by live workspace", async () => {
    const { fixture, project } = await setup();
    const document = fixture.context.science.createDocument(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      name: "paper.typ",
      content: "= Results",
    });

    expect(() =>
      fixture.context.science.modifyDocument(fixture.sessionB, {
        requestId: randomUUID(),
        documentId: document.id,
        expectedRevision: 1,
        action: "propose",
        selection: { start: 0, end: 1 },
        proposedText: "#",
        instruction: "Change heading syntax",
        reasoning: "This request belongs to another workspace.",
      }),
    ).toThrowError(expect.objectContaining<Partial<ScienceError>>({ code: "DOCUMENT_NOT_FOUND" }));
  });

  it("honors pre-aborted create and modify requests before appending", async () => {
    const { fixture, project } = await setup();
    const controller = new AbortController();
    controller.abort();

    expect(() =>
      fixture.context.science.createDocument(
        fixture.sessionA,
        {
          requestId: randomUUID(),
          projectId: project.id,
          name: "cancelled.typ",
          content: "= Cancelled",
        },
        controller.signal,
      ),
    ).toThrowError(expect.objectContaining({ name: "AbortError" }));
    expect(fixture.context.science.journalCount()).toBe(1);
  });

  it("migrates a v2 database before accepting document facts", async () => {
    const { fixture, project } = await setup();
    await fixture.scienceFiber.dispose();
    const v2 = new DatabaseSync(fixture.databasePath);
    v2.exec(
      "DELETE FROM science_migrations WHERE version >= 3; DROP TABLE science_figures; DROP TABLE science_documents;",
    );
    v2.close();

    await fixture.remount();
    const document = fixture.context.science.createDocument(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      name: "migrated.md",
      content: "# Migrated",
    });
    const migrated = new DatabaseSync(fixture.databasePath, { readOnly: true });
    const version = migrated
      .prepare("SELECT MAX(version) AS version FROM science_migrations")
      .get() as {
      version: number;
    };
    migrated.close();

    expect(version.version).toBe(5);
    expect(fixture.context.science.getWorkspace(fixture.sessionA).documents).toEqual([document]);
  });

  it("rebuilds the document projection from create and revision facts", async () => {
    const { fixture, project } = await setup();
    const document = fixture.context.science.createDocument(fixture.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      name: "paper.typ",
      content: "= Result",
    });
    const proposed = fixture.context.science.modifyDocument(fixture.sessionA, {
      requestId: randomUUID(),
      documentId: document.id,
      expectedRevision: 1,
      action: "propose",
      selection: { start: 2, end: 8 },
      proposedText: "Results",
      instruction: "Use the conventional section title",
      reasoning: "The section contains multiple findings.",
    });
    const proposalId = proposed.proposals[0]?.id;
    if (!proposalId) throw new Error("Expected a replayable proposal");
    const accepted = fixture.context.science.modifyDocument(fixture.sessionA, {
      requestId: randomUUID(),
      documentId: document.id,
      expectedRevision: 2,
      action: "accept",
      proposalId,
    });
    await fixture.scienceFiber.dispose();
    const database = new DatabaseSync(fixture.databasePath);
    database.exec("DELETE FROM science_documents");
    database.close();

    await fixture.remount();

    expect(fixture.context.science.getWorkspace(fixture.sessionA).documents).toEqual([accepted]);
  });
});
