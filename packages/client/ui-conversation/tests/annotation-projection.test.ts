import { describe, expect, it } from "vitest";
import { annotationCopyText, projectAnnotatedText } from "../src/client/annotation-projection.js";

const annotation = {
  type: "comment",
  id: "annotation-1",
  comment: "State the confidence interval.",
  created_at: 1_787_371_200_000,
  target: {
    type: "document_text",
    relative_path: "papers/main.typ",
    title: "main.typ",
    source_revision: `sha256:${"a".repeat(64)}`,
    render_revision: `sha256:${"b".repeat(64)}`,
    page: 2,
    rect: { x: 0.1, y: 0.2, width: 0.6, height: 0.04 },
    text: "The result improves by four percent.",
  },
} as const;

describe("V101 annotation transcript projection", () => {
  it("removes protocol markup while preserving the Markdown body", () => {
    const text = `Please **revise** this. <annotation>${JSON.stringify(annotation)}</annotation>`;
    const projected = projectAnnotatedText(text);
    expect(projected.body).toBe("Please **revise** this.");
    expect(projected.annotations).toEqual([annotation]);
    expect(projected.invalidCount).toBe(0);
    expect(annotationCopyText(projected)).toContain("> State the confidence interval.");
    expect(annotationCopyText(projected)).toContain("Please **revise** this.");
    expect(annotationCopyText(projected)).not.toContain("<annotation>");
  });

  it("projects persisted legacy paper payloads without rewriting them", () => {
    const legacy = {
      version: 1,
      id: "paper-annotation-1",
      kind: "text",
      relativePath: "papers/main.typ",
      title: "main.typ",
      sourceRevision: `sha256:${"a".repeat(64)}`,
      pdfRevision: `sha256:${"b".repeat(64)}`,
      page: 2,
      rect: { x: 0.1, y: 0.2, width: 0.6, height: 0.04 },
      selectedText: "The result improves by four percent.",
      comment: "State the confidence interval.",
      createdAt: 1_787_371_200_000,
    };
    const text = `<science-paper-annotation>${JSON.stringify(legacy)}</science-paper-annotation>\nUse the workspace-relative Typst source and this rendered PDF context to make the requested revision. Re-open the paper preview after editing; do not infer a host path.`;
    const projected = projectAnnotatedText(text);
    expect(projected.body).toBe("");
    expect(projected.annotations).toEqual([{ ...annotation, id: "paper-annotation-1" }]);
    expect(text).toContain("science-paper-annotation");
  });

  it("hides malformed payloads instead of exposing raw JSON", () => {
    const projected = projectAnnotatedText("<annotation>{bad}</annotation> Continue here.");
    expect(projected.body).toBe("Continue here.");
    expect(projected.annotations).toEqual([]);
    expect(projected.invalidCount).toBe(1);
  });
});
