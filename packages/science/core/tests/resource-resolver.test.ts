import { describe, expect, it } from "vitest";
import type { ScienceWorkspaceSnapshot } from "../src/contracts.js";
import { formatScienceResourceId } from "../src/resource-id.js";
import { ScienceResourceResolver } from "../src/resource-resolver.js";

const DIGEST_A = `sha256:${"a".repeat(64)}` as const;
const DIGEST_B = `sha256:${"b".repeat(64)}` as const;
const PROVENANCE = {
  eventId: "event-secret",
  journalSeq: 7,
  sessionId: "session-secret",
};

function workspace(): ScienceWorkspaceSnapshot {
  return {
    projects: [
      {
        id: "project-id",
        kind: "project",
        title: "Project",
        createdAt: 1,
        updatedAt: 2,
        revision: 2,
        provenance: PROVENANCE,
      },
    ],
    notebooks: [
      {
        id: "notebook-id",
        projectId: "project-id",
        kind: "notebook",
        title: "Notebook",
        cells: [],
        createdAt: 1,
        updatedAt: 2,
        revision: 3,
        provenance: PROVENANCE,
      },
    ],
    artifacts: [
      {
        id: "artifact-id",
        projectId: "project-id",
        kind: "dataset",
        title: "Dataset",
        digest: DIGEST_A,
        mime: "text/csv",
        size: 20,
        creator: { kind: "session", sessionId: "session-secret" },
        runId: "run-id",
        environment: { API_TOKEN: "secret-environment" },
        license: "MIT",
        sourceEntityIds: ["notebook-id"],
        createdAt: 1,
        updatedAt: 2,
        revision: 4,
        provenance: PROVENANCE,
      },
    ],
    documents: [
      {
        id: "document-id",
        projectId: "project-id",
        kind: "document",
        name: "paper.typ",
        format: "typst",
        content: "= private source",
        revision: 5,
        contentRevision: 2,
        proposals: [],
        revisions: [
          {
            revision: 5,
            contentRevision: 2,
            sourceHash: DIGEST_B,
            previousSourceHash: DIGEST_A,
            reason: "proposal-accepted",
            proposalId: "proposal-id",
            provenance: PROVENANCE,
          },
        ],
        diagnostics: [],
        validation: { structural: "checked", compilation: "not-run" },
        createdAt: 1,
        updatedAt: 2,
        provenance: PROVENANCE,
      },
    ],
    figures: [
      {
        id: "figure-id",
        projectId: "project-id",
        kind: "figure",
        title: "Figure",
        library: "matplotlib",
        code: "private plotting code",
        artifactId: "artifact-id",
        objects: [
          { id: "object-id", kind: "axis", label: "Axis", codeRange: { start: 0, end: 1 } },
        ],
        revision: 6,
        codeRevision: 2,
        proposals: [],
        revisions: [
          {
            revision: 6,
            codeRevision: 2,
            codeHash: DIGEST_A,
            previousCodeHash: DIGEST_B,
            reason: "proposal-accepted",
            proposalId: "proposal-id",
            provenance: PROVENANCE,
          },
        ],
        createdAt: 1,
        updatedAt: 2,
        provenance: PROVENANCE,
      },
    ],
    records: [
      {
        id: "record-id",
        projectId: "project-id",
        kind: "claim",
        title: "Claim",
        summary: "private summary",
        status: "supported",
        tags: [],
        sourceEntityIds: ["artifact-id"],
        createdAt: 1,
        updatedAt: 2,
        revision: 7,
        provenance: PROVENANCE,
      },
    ],
    relations: [],
    experiments: [
      {
        id: "experiment-id",
        projectId: "project-id",
        kind: "experiment",
        title: "Experiment",
        summary: "private experiment summary",
        protocol: "private protocol",
        hypothesisIds: [],
        runIds: ["run-id"],
        status: "active",
        tags: [],
        createdAt: 1,
        updatedAt: 2,
        revision: 8,
        provenance: PROVENANCE,
      },
    ],
    runs: [
      {
        id: "run-id",
        projectId: "project-id",
        experimentId: "experiment-id",
        kind: "run",
        status: "succeeded",
        environment: { PRIVATE_PATH: "/host/secret" },
        metrics: { accuracy: 0.9 },
        artifactIds: ["artifact-id"],
        notes: "private run notes",
        startedAt: 1,
        finishedAt: 2,
        revision: 9,
        provenance: PROVENANCE,
      },
    ],
    exports: [],
  };
}

describe("workspace-scoped Science Resource Resolver", () => {
  it("resolves every resource kind with current revision and trusted digest", () => {
    const resolver = new ScienceResourceResolver(workspace());
    const cases = [
      ["project", "project-id", "Project", 2, null],
      ["notebook", "notebook-id", "Notebook", 3, null],
      ["artifact", "artifact-id", "Dataset", 4, DIGEST_A],
      ["document", "document-id", "paper.typ", 5, DIGEST_B],
      ["figure", "figure-id", "Figure", 6, DIGEST_A],
      ["record", "record-id", "Claim", 7, null],
      ["experiment", "experiment-id", "Experiment", 8, null],
      ["run", "run-id", "Experiment run", 9, null],
    ] as const;

    for (const [kind, entityId, title, revision, digest] of cases) {
      const logical = formatScienceResourceId(kind, entityId);
      const resolved = resolver.resolve(logical);
      expect(resolved.ref).toEqual({
        id: logical,
        exactId: `${logical}@${revision}`,
        kind,
        title,
        revision,
        digest,
      });
      expect(resolver.resolve(`${logical}@${revision}`).ref).toEqual(resolved.ref);
    }
  });

  it("distinguishes malformed, absent, wrong-kind, and stale exact addresses", () => {
    const resolver = new ScienceResourceResolver(workspace());

    expect(() => resolver.resolve("artifact-id")).toThrowError(
      expect.objectContaining({ code: "INVALID_RESOURCE_ID" }),
    );
    expect(() => resolver.resolve(formatScienceResourceId("artifact", "missing"))).toThrowError(
      expect.objectContaining({ code: "RESOURCE_NOT_FOUND" }),
    );
    expect(() => resolver.resolve(formatScienceResourceId("notebook", "artifact-id"))).toThrowError(
      expect.objectContaining({ code: "RESOURCE_KIND_MISMATCH" }),
    );
    expect(() =>
      resolver.resolve(formatScienceResourceId("artifact", "artifact-id", 3)),
    ).toThrowError(expect.objectContaining({ code: "RESOURCE_REVISION_MISMATCH" }));
  });

  it("does not discover an entity from another workspace", () => {
    const foreign = workspace().artifacts[0];
    const local = workspace();
    const resolver = new ScienceResourceResolver({ ...local, artifacts: [] });

    expect(() =>
      resolver.resolve(formatScienceResourceId("artifact", foreign?.id ?? "missing")),
    ).toThrowError(expect.objectContaining({ code: "RESOURCE_NOT_FOUND" }));
  });

  it("keeps same raw IDs unambiguous through typed addresses and rejects same-kind collisions", () => {
    const current = workspace();
    const project = current.projects[0];
    const notebook = current.notebooks[0];
    const artifact = current.artifacts[0];
    if (!project || !notebook || !artifact) throw new Error("fixture incomplete");
    const shared = "shared-id";
    const resolver = new ScienceResourceResolver({
      ...current,
      projects: [{ ...project, id: shared }],
      notebooks: [{ ...notebook, id: shared, projectId: shared }],
    });

    expect(resolver.resolve(formatScienceResourceId("project", shared)).ref.kind).toBe("project");
    expect(resolver.resolve(formatScienceResourceId("notebook", shared)).ref.kind).toBe("notebook");
    expect(
      () =>
        new ScienceResourceResolver({
          ...current,
          artifacts: [artifact, artifact],
        }),
    ).toThrowError(expect.objectContaining({ code: "RESOURCE_INDEX_CONFLICT" }));
  });

  it("returns only the path-free client ref by default", () => {
    const ref = new ScienceResourceResolver(workspace()).resolve(
      formatScienceResourceId("artifact", "artifact-id"),
    ).ref;
    const serialized = JSON.stringify(ref);

    expect(serialized).not.toContain("session-secret");
    expect(serialized).not.toContain("secret-environment");
    expect(serialized).not.toContain("/host/secret");
    expect(serialized).not.toContain("event-secret");
  });
});
