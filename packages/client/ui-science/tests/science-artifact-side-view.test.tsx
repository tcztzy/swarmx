import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type {
  RoCrateMetadataDocument,
  ScienceArtifact,
  ScienceImageAnnotation,
} from "../../../science/core/src/contracts.js";
import {
  normalizedImagePoint,
  SCIENCE_ARTIFACT_ACTIONS,
  ScienceArtifactPreviewView,
  ScienceArtifactProvenanceView,
  ScienceArtifactSideView,
  scienceArtifactSideViewEntry,
} from "../src/client/science-artifact-side-view.js";

vi.mock("@deepseek-ai/dsh-client-ui-primitives", () => ({
  IconBranchOutline16: () => <span data-icon="branch" />,
  IconEllipsisOutline16: () => <span data-icon="ellipsis" />,
  IconFullscreenOutline16: () => <span data-icon="fullscreen" />,
  IconEditOutline16: () => <span data-icon="edit" />,
  IconSendOutline16: () => <span data-icon="send" />,
  IconTrashOutline16: () => <span data-icon="trash" />,
  Menu: ({ anchor }: { readonly anchor: React.ReactNode }) => <>{anchor}</>,
}));

const artifact: ScienceArtifact = {
  id: "artifact-1",
  projectId: "project-1",
  kind: "figure",
  title: "umap.png",
  digest: `sha256:${"a".repeat(64)}`,
  mime: "image/png",
  size: 2048,
  creator: { kind: "session", sessionId: "session-1" },
  runId: "run-1",
  environment: { python: "3.12" },
  license: null,
  sourceEntityIds: ["notebook-1"],
  createdAt: 1,
  updatedAt: 1,
  revision: 1,
  provenance: { eventId: "event-1", journalSeq: 42, sessionId: "session-1" },
};

const annotation: ScienceImageAnnotation = {
  version: 1,
  id: "annotation-1",
  artifactId: "artifact-1",
  projectId: "project-1",
  title: "umap.png",
  digest: `sha256:${"a".repeat(64)}`,
  mime: "image/png",
  x: 0.25,
  y: 0.75,
  comment: "Why is this cluster separated?",
  createdAt: 1_787_371_200_000,
};

describe("V50 Science artifact Side View", () => {
  it("builds one bounded metadata-only keyed entry", () => {
    expect(scienceArtifactSideViewEntry(artifact, [annotation])).toEqual({
      id: "science-artifact:artifact-1",
      kind: "science-artifact",
      title: "umap.png",
      mode: "workbench",
      payload: {
        artifactId: "artifact-1",
        projectId: "project-1",
        artifactKind: "figure",
        title: "umap.png",
        digest: artifact.digest,
        mime: "image/png",
        size: 2048,
        runId: "run-1",
        sourceEntityIds: ["notebook-1"],
        revision: 1,
        journalSeq: 42,
        annotations: [annotation],
      },
    });
    expect(JSON.stringify(scienceArtifactSideViewEntry(artifact))).not.toContain("/Users/");
  });

  it("renders artifact identity, provenance, and no host path", () => {
    const markup = renderToStaticMarkup(
      <ScienceArtifactSideView
        entry={scienceArtifactSideViewEntry(artifact)}
        loadPreview={() => new Promise(() => undefined)}
        loadResearchObject={() => new Promise(() => undefined)}
        updateAnnotations={vi.fn()}
        addAnnotationToConversation={() => true}
      />,
    );

    expect(markup).toContain("umap.png");
    expect(markup).toContain("image/png");
    expect(markup).toContain("Journal #42");
    expect(markup).toContain('data-artifact-file-view="true"');
    expect(markup).toContain("File details");
    expect(markup).toContain('aria-label="Open fullscreen"');
    expect(markup).toContain('aria-label="More actions"');
    expect(SCIENCE_ARTIFACT_ACTIONS.map(({ id }) => id)).toContain("provenance");
    expect(markup).not.toContain("Open in Science");
    expect(markup).not.toContain("/Users/");
  });

  it("renders a bounded current-artifact provenance surface", () => {
    const researchObject: RoCrateMetadataDocument = {
      "@context": "https://w3id.org/ro/crate/1.3/context",
      "@graph": [
        {
          "@id": "ro-crate-metadata.json",
          "@type": "CreativeWork",
          about: { "@id": "urn:uuid:project-1" },
          conformsTo: { "@id": "https://w3id.org/ro/crate/1.3" },
        },
        {
          "@id": "urn:uuid:project-1",
          "@type": "Dataset",
          name: "Project",
          description: "Research Object for Project",
          datePublished: "2026-08-24T00:00:00.000Z",
          license: "All rights reserved",
          hasPart: [{ "@id": "urn:uuid:artifact-1" }, { "@id": "urn:uuid:notebook-1" }],
        },
        {
          "@id": "urn:uuid:artifact-1",
          "@type": ["MediaObject", "ImageObject"],
          name: "umap.png",
          isBasedOn: [{ "@id": "urn:uuid:notebook-1" }],
        },
        { "@id": "urn:uuid:notebook-1", "@type": "SoftwareSourceCode", name: "Analysis" },
        {
          "@id": "#action-event-1",
          "@type": "CreateAction",
          name: "Register research artifact",
          identifier: "science-journal:42",
          object: [{ "@id": "urn:uuid:notebook-1" }],
          result: [{ "@id": "urn:uuid:artifact-1" }],
        },
      ],
    };

    const markup = renderToStaticMarkup(
      <ScienceArtifactProvenanceView
        artifactId="artifact-1"
        researchObject={researchObject}
        onClose={vi.fn()}
      />,
    );
    expect(markup).toContain("Provenance");
    expect(markup).toContain("Back to artifact");
    expect(markup).toContain("Analysis");
    expect(markup).toContain("is based on");
    expect(markup).toContain("Journal #42");
    expect(markup).not.toContain("/Users/");
  });

  it("normalizes only points inside the fitted image content", () => {
    const rect = { left: 10, top: 20, width: 200, height: 200 };
    expect(
      normalizedImagePoint({
        clientX: 110,
        clientY: 120,
        rect,
        naturalWidth: 400,
        naturalHeight: 200,
      }),
    ).toEqual({ x: 0.5, y: 0.5 });
    expect(
      normalizedImagePoint({
        clientX: 110,
        clientY: 30,
        rect,
        naturalWidth: 400,
        naturalHeight: 200,
      }),
    ).toBeNull();
  });

  it("renders bounded table, text, image, unavailable, and error preview states", () => {
    const tableMarkup = renderToStaticMarkup(
      <ScienceArtifactPreviewView
        preview={{
          kind: "table",
          artifactId: "artifact-1",
          digest: artifact.digest,
          mime: "text/csv",
          size: 24,
          columns: [
            { id: "column-0", name: "sample", type: "string" },
            { id: "column-1", name: "value", type: "number" },
          ],
          rows: [
            ["A", 42],
            ["B", null],
          ],
          rowCount: 2,
          truncated: false,
        }}
      />,
    );
    expect(tableMarkup).toContain('data-science-table-grid="true"');
    expect(tableMarkup).toContain("2 rows × 2 columns");
    expect(tableMarkup).not.toContain("&quot;sample&quot;");
    expect(
      renderToStaticMarkup(
        <ScienceArtifactPreviewView
          preview={{
            kind: "text",
            artifactId: "artifact-1",
            digest: artifact.digest,
            mime: "text/plain",
            size: 5,
            text: "hello",
          }}
        />,
      ),
    ).toContain("hello");
    expect(
      renderToStaticMarkup(
        <ScienceArtifactPreviewView
          preview={{
            kind: "image",
            artifactId: "artifact-1",
            digest: artifact.digest,
            mime: "image/png",
            size: 4,
            dataUrl: "data:image/png;base64,iVBORw==",
          }}
        />,
      ),
    ).toContain('src="data:image/png;base64,iVBORw=="');
    expect(
      renderToStaticMarkup(
        <ScienceArtifactPreviewView
          preview={{
            kind: "unavailable",
            artifactId: "artifact-1",
            digest: artifact.digest,
            mime: "application/octet-stream",
            size: 100,
            reason: "unsupported",
          }}
        />,
      ),
    ).toContain("Preview unavailable");
    expect(
      renderToStaticMarkup(<ScienceArtifactPreviewView preview={new Error("failed")} />),
    ).toContain("failed");
  });
});
