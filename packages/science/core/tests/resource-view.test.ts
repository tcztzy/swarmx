import { describe, expect, it } from "vitest";
import type { ScienceArtifactPreview, ScienceWorkspaceSnapshot } from "../src/contracts.js";
import { formatScienceResourceId } from "../src/resource-id.js";
import { ScienceResourceResolver } from "../src/resource-resolver.js";
import {
  scienceResourceBatchHead,
  scienceResourceBatchHeadRequestSchema,
  scienceResourceGetRequestSchema,
  scienceResourceHead,
  scienceResourceHeadRequestSchema,
  scienceResourceMetadata,
  scienceResourceNeighbors,
  scienceResourceNeighborsRequestSchema,
  scienceResourceSelect,
  scienceResourceSelectRequestSchema,
} from "../src/resource-view.js";

const DIGEST = `sha256:${"a".repeat(64)}` as const;
const provenance = { eventId: "event-private", journalSeq: 1, sessionId: "session-private" };

function workspace(): ScienceWorkspaceSnapshot {
  const artifacts = Array.from({ length: 25 }, (_, index) => ({
    id: `artifact-${index}`,
    projectId: "project",
    kind: "dataset" as const,
    title: `Artifact ${index}`,
    digest: DIGEST,
    mime: "text/csv",
    size: 100,
    creator: { kind: "session" as const, sessionId: "session-private" },
    runId: index === 0 ? "run" : null,
    environment: { TOKEN: "private-environment" },
    license: null,
    sourceEntityIds: index === 0 ? ["notebook"] : [],
    createdAt: 1,
    updatedAt: 2,
    revision: 1,
    provenance,
  }));
  return {
    projects: [
      {
        id: "project",
        kind: "project",
        title: "Project",
        createdAt: 1,
        updatedAt: 2,
        revision: 1,
        provenance,
      },
    ],
    notebooks: [
      {
        id: "notebook",
        projectId: "project",
        kind: "notebook",
        title: "Notebook",
        cells: [
          {
            id: "cell-private",
            kind: "code",
            source: "private notebook source",
            executionCount: 1,
            executionTimeMs: 1,
            inputArtifactIds: ["artifact-1"],
            outputArtifactIds: ["artifact-0"],
            runtimeEnvironment: { TOKEN: "private-cell-environment" },
            relatedClaimIds: [],
            relatedExperimentIds: [],
            outputs: [],
          },
        ],
        createdAt: 1,
        updatedAt: 2,
        revision: 2,
        provenance,
      },
    ],
    artifacts,
    documents: [
      {
        id: "document",
        projectId: "project",
        kind: "document",
        name: "paper.typ",
        format: "typst",
        content: "private document source",
        revision: 3,
        contentRevision: 2,
        proposals: [],
        revisions: [
          {
            revision: 3,
            contentRevision: 2,
            sourceHash: DIGEST,
            previousSourceHash: null,
            reason: "created",
            proposalId: null,
            provenance,
          },
        ],
        diagnostics: [],
        validation: { structural: "checked", compilation: "not-run" },
        createdAt: 1,
        updatedAt: 2,
        provenance,
      },
    ],
    figures: [
      {
        id: "figure",
        projectId: "project",
        kind: "figure",
        title: "Figure",
        library: "matplotlib",
        code: "private figure code",
        artifactId: "artifact-0",
        objects: [{ id: "axis", kind: "axis", label: "Axis", codeRange: { start: 0, end: 1 } }],
        revision: 2,
        codeRevision: 1,
        proposals: [],
        revisions: [
          {
            revision: 2,
            codeRevision: 1,
            codeHash: DIGEST,
            previousCodeHash: null,
            reason: "created",
            proposalId: null,
            provenance,
          },
        ],
        createdAt: 1,
        updatedAt: 2,
        provenance,
      },
    ],
    records: [
      {
        id: "record",
        projectId: "project",
        kind: "claim",
        title: "Claim",
        summary: "s".repeat(4_000),
        status: "supported",
        tags: Array.from({ length: 32 }, (_, index) => `tag-${index}`),
        sourceEntityIds: [...artifacts.map((artifact) => artifact.id), "missing-source"],
        createdAt: 1,
        updatedAt: 2,
        revision: 1,
        provenance,
      },
      {
        id: "hypothesis",
        projectId: "project",
        kind: "hypothesis",
        title: "Hypothesis",
        summary: "Hypothesis summary",
        status: "proposed",
        tags: [],
        sourceEntityIds: [],
        createdAt: 1,
        updatedAt: 2,
        revision: 1,
        provenance,
      },
    ],
    relations: [
      {
        id: "relation",
        projectId: "project",
        fromId: "record",
        toId: "hypothesis",
        type: "supports",
        createdAt: 2,
        provenance,
      },
    ],
    experiments: [
      {
        id: "experiment",
        projectId: "project",
        kind: "experiment",
        title: "Experiment",
        summary: "Experiment summary",
        protocol: "private experiment protocol",
        hypothesisIds: ["hypothesis"],
        runIds: ["run"],
        status: "active",
        tags: [],
        createdAt: 1,
        updatedAt: 2,
        revision: 2,
        provenance,
      },
    ],
    runs: [
      {
        id: "run",
        projectId: "project",
        experimentId: "experiment",
        kind: "run",
        status: "succeeded",
        environment: { TOKEN: "private-run-environment" },
        metrics: Object.fromEntries(
          Array.from({ length: 30 }, (_, index) => [`metric-${index}`, index]),
        ),
        artifactIds: artifacts.map((artifact) => artifact.id),
        notes: "private run notes",
        startedAt: 1,
        finishedAt: 2,
        revision: 2,
        provenance,
      },
    ],
    exports: [],
  } as ScienceWorkspaceSnapshot;
}

describe("bounded Science resource views", () => {
  it("returns a minimal head and preserves batch order and duplicates", () => {
    const resolver = new ScienceResourceResolver(workspace());
    const artifactId = formatScienceResourceId("artifact", "artifact-0");
    const recordId = formatScienceResourceId("record", "record");
    const head = scienceResourceHead(
      resolver,
      scienceResourceHeadRequestSchema.parse({ id: artifactId }),
    );
    const batch = scienceResourceBatchHead(
      resolver,
      scienceResourceBatchHeadRequestSchema.parse({ ids: [recordId, artifactId, recordId] }),
    );

    expect(head).toMatchObject({
      ref: { id: artifactId, exactId: `${artifactId}@1`, kind: "artifact", digest: DIGEST },
      capabilities: ["get", "select", "neighbors"],
    });
    expect(batch.heads.map((item) => item.ref.id)).toEqual([recordId, artifactId, recordId]);
    expect(JSON.stringify(head)).not.toContain("private-environment");
    expect(() =>
      scienceResourceBatchHeadRequestSchema.parse({ ids: Array(51).fill(recordId) }),
    ).toThrow();
  });

  it("returns bounded metadata without full document, notebook, figure, protocol, or environment", () => {
    const resolver = new ScienceResourceResolver(workspace());
    const ids = [
      formatScienceResourceId("notebook", "notebook"),
      formatScienceResourceId("document", "document"),
      formatScienceResourceId("figure", "figure"),
      formatScienceResourceId("experiment", "experiment"),
    ];
    const results = ids.map((id) =>
      scienceResourceMetadata(
        resolver,
        scienceResourceGetRequestSchema.parse({ id, projection: "metadata" }),
      ),
    );
    const serialized = JSON.stringify(results);

    expect(serialized).not.toContain("private notebook source");
    expect(serialized).not.toContain("private document source");
    expect(serialized).not.toContain("private figure code");
    expect(serialized).not.toContain("private experiment protocol");
    expect(serialized).not.toContain("private-cell-environment");
    expect(results.every((result) => result.ref.exactId.endsWith(`@${result.ref.revision}`))).toBe(
      true,
    );
  });

  it("truncates tags, source refs, metric keys, and artifact refs", () => {
    const resolver = new ScienceResourceResolver(workspace());
    const record = scienceResourceMetadata(
      resolver,
      scienceResourceGetRequestSchema.parse({
        id: formatScienceResourceId("record", "record"),
        projection: "metadata",
      }),
    );
    const run = scienceResourceMetadata(
      resolver,
      scienceResourceGetRequestSchema.parse({
        id: formatScienceResourceId("run", "run"),
        projection: "metadata",
      }),
    );

    expect(record.metadata).toMatchObject({
      kind: "record",
      summary: `${"s".repeat(497)}...`,
      tags: { total: 32, truncated: true },
      sourceRefs: { total: 26, truncated: true },
    });
    expect(record.metadata.kind === "record" ? record.metadata.tags.items : []).toHaveLength(20);
    expect(run.metadata).toMatchObject({
      kind: "run",
      metricKeys: { total: 30, truncated: true },
      artifactRefs: { total: 25, truncated: true },
    });
  });

  it("slices table columns and rows in requested order and rejects unknown columns and limits", () => {
    const resolver = new ScienceResourceResolver(workspace());
    const id = formatScienceResourceId("artifact", "artifact-0");
    const preview: ScienceArtifactPreview = {
      kind: "table",
      artifactId: "artifact-0",
      digest: DIGEST,
      mime: "text/csv",
      size: 100,
      columns: [
        { id: "column-0", name: "a", type: "number" },
        { id: "column-1", name: "b", type: "number" },
      ],
      rows: [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
      rowCount: 3,
      truncated: false,
    };
    const selected = scienceResourceSelect(
      resolver,
      scienceResourceSelectRequestSchema.parse({
        id,
        format: "table",
        offset: 1,
        limit: 1,
        columns: ["b", "a"],
      }),
      preview,
    );

    expect(selected).toMatchObject({
      kind: "table",
      columns: [{ name: "b" }, { name: "a" }],
      rows: [[4, 3]],
      total: 3,
      offset: 1,
      returned: 1,
      truncated: true,
      nextOffset: 2,
    });
    expect(() =>
      scienceResourceSelect(
        resolver,
        scienceResourceSelectRequestSchema.parse({
          id,
          format: "table",
          columns: ["missing"],
        }),
        preview,
      ),
    ).toThrowError(expect.objectContaining({ code: "RESOURCE_COLUMN_NOT_FOUND" }));
    expect(() =>
      scienceResourceSelectRequestSchema.parse({ id, format: "table", limit: 101 }),
    ).toThrow();
  });

  it("bounds text windows and returns explicit unsupported and too-large states", () => {
    const resolver = new ScienceResourceResolver(workspace());
    const id = formatScienceResourceId("artifact", "artifact-0");
    const request = scienceResourceSelectRequestSchema.parse({
      id,
      format: "text",
      offset: 1,
      limit: 16_384,
    });
    const text = scienceResourceSelect(resolver, request, {
      kind: "text",
      artifactId: "artifact-0",
      digest: DIGEST,
      mime: "text/plain",
      size: 20_000,
      text: "x".repeat(20_000),
    });
    const unavailable = scienceResourceSelect(resolver, request, {
      kind: "unavailable",
      artifactId: "artifact-0",
      digest: DIGEST,
      mime: "text/plain",
      size: 70_000,
      reason: "too-large",
    });
    const unsupported = scienceResourceSelect(
      resolver,
      scienceResourceSelectRequestSchema.parse({ id, format: "table" }),
      {
        kind: "text",
        artifactId: "artifact-0",
        digest: DIGEST,
        mime: "text/plain",
        size: 4,
        text: "text",
      },
    );

    expect(text.kind === "text" ? text.text.length : 0).toBe(16_384);
    expect(text).toMatchObject({ returned: 16_384, truncated: true, nextOffset: 16_385 });
    expect(unavailable).toMatchObject({ kind: "unavailable", reason: "too-large" });
    expect(unsupported).toMatchObject({ kind: "unavailable", reason: "unsupported" });
  });

  it("returns explicit directed neighbors, applies filters/limits, and skips dangling refs", () => {
    const resolver = new ScienceResourceResolver(workspace());
    const record = scienceResourceNeighbors(
      resolver,
      scienceResourceNeighborsRequestSchema.parse({
        id: formatScienceResourceId("record", "record"),
        relations: ["supports", "derived_from"],
        limit: 2,
      }),
    );
    const experiment = scienceResourceNeighbors(
      resolver,
      scienceResourceNeighborsRequestSchema.parse({
        id: formatScienceResourceId("experiment", "experiment"),
        limit: 100,
      }),
    );

    expect(record.neighbors).toHaveLength(2);
    expect(record.neighbors[0]).toMatchObject({
      relation: "derived_from",
      direction: "outgoing",
      target: { kind: "artifact" },
    });
    expect(record.total).toBe(26);
    expect(record.truncated).toBe(true);
    expect(experiment.neighbors).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          relation: "has_hypothesis",
          target: expect.objectContaining({ kind: "record" }),
        }),
        expect.objectContaining({
          relation: "has_run",
          target: expect.objectContaining({ kind: "run" }),
        }),
      ]),
    );
    expect(() =>
      scienceResourceNeighborsRequestSchema.parse({ id: "sx:p/project", limit: 101 }),
    ).toThrow();
  });
});
