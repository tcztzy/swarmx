import { describe, expect, it, vi } from "vitest";
import type {
  ScienceImageAnnotation,
  SciencePaperAnnotation,
} from "../../../science/core/src/contracts.js";
import {
  annotationReferenceInsert,
  annotationReferenceSource,
  insertAnnotationReference,
} from "../../ui-conversation/src/client/annotation-reference.js";
import {
  imageCommentAnnotation,
  paperCommentAnnotation,
} from "../src/client/annotation-reference.js";

const imageAnnotation: ScienceImageAnnotation = {
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

const paperAnnotation: SciencePaperAnnotation = {
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
  comment: "State the confidence interval too.",
  createdAt: 1_787_371_200_000,
};

describe("V81 V92 generic annotation references", () => {
  it("maps image and paper context into one comment union", () => {
    expect(imageCommentAnnotation(imageAnnotation)).toMatchObject({
      type: "comment",
      target: { type: "image_point", point: { x: 0.25, y: 0.75 } },
    });
    expect(paperCommentAnnotation(paperAnnotation)).toMatchObject({
      type: "comment",
      target: {
        type: "document_text",
        relative_path: "papers/main.typ",
        text: "The result improves by four percent.",
      },
    });
  });

  it("uses one annotation source and one hidden model payload", async () => {
    const annotation = paperCommentAnnotation(paperAnnotation);
    const source = annotationReferenceSource();
    const insert = annotationReferenceInsert(annotation);
    const serialized = await source.codec?.serialize(insert.ref, new AbortController().signal);
    expect(source.name).toBe("annotation");
    expect(insert.source).toBe("annotation");
    expect(serialized).toMatch(/^<dsh-annotation>.*<\/dsh-annotation>$/u);
    expect(serialized).toContain("The result improves by four percent.");
    expect(serialized).not.toContain("science-paper-annotation");
    expect(serialized).not.toContain("data:application/pdf");
    expect(serialized).not.toContain("/Users/");
  });

  it("inserts one occurrence while preserving the existing draft", () => {
    let state = {
      draft: "Please revise",
      draftRev: 4,
      phase: "plain",
      occurrences: [],
      imageIds: [],
      queue: [],
    };
    const insertReference = vi.fn(() => true);
    const input = {
      state: { getSnapshot: () => state },
      setDraft: vi.fn((draft: string) => {
        state = { ...state, draft, draftRev: state.draftRev + 1 };
      }),
      insertReference,
    };
    const conversation = { input: { for: vi.fn(() => input) } };
    const sessions = { binding: vi.fn(() => ({ ctx: {} })) };
    const annotation = paperCommentAnnotation(paperAnnotation);

    expect(insertAnnotationReference(conversation, sessions, "session-1", annotation)).toBe(true);
    expect(input.setDraft).not.toHaveBeenCalled();
    expect(insertReference).toHaveBeenCalledWith(annotationReferenceInsert(annotation), {
      start: 13,
      end: 13,
      draftRev: 4,
    });
  });
});
