import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ScienceArtifact } from "../../../science/core/src/contracts.js";
import {
  ScienceArtifactPreviewView,
  ScienceArtifactSideView,
  scienceArtifactSideViewEntry,
  scienceArtifactWorkbenchTarget,
} from "../src/client/science-artifact-side-view.js";

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

describe("V50 Science artifact Side View", () => {
  it("builds one bounded metadata-only keyed entry", () => {
    expect(scienceArtifactSideViewEntry(artifact)).toEqual({
      id: "science-artifact:artifact-1",
      kind: "science-artifact",
      title: "umap.png",
      mode: "inspect",
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
      },
    });
    expect(JSON.stringify(scienceArtifactSideViewEntry(artifact))).not.toContain("/Users/");
  });

  it("renders artifact identity, provenance, and no host path", () => {
    const markup = renderToStaticMarkup(
      <ScienceArtifactSideView
        entry={scienceArtifactSideViewEntry(artifact)}
        loadPreview={() => new Promise(() => undefined)}
        openInScience={() => false}
      />,
    );

    expect(markup).toContain("umap.png");
    expect(markup).toContain("image/png");
    expect(markup).toContain("Journal #42");
    expect(markup).toContain('data-artifact-file-view="true"');
    expect(markup).toContain("File details");
    expect(markup.indexOf("Preview")).toBeLessThan(markup.indexOf("File details"));
    expect(markup).not.toContain("/Users/");
  });

  it("keeps the same locator for the fullscreen Science fallback", () => {
    expect(scienceArtifactWorkbenchTarget(scienceArtifactSideViewEntry(artifact))).toEqual({
      kind: "artifact",
      artifactId: "artifact-1",
      projectId: "project-1",
      surface: "artifacts",
    });
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
