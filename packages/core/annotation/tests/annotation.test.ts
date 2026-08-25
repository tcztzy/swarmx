import { describe, expect, it } from "vitest";
import {
  annotationSchema,
  commentAnnotationSchema,
  messageQuoteAnnotationSchema,
  openAIResponseAnnotationSchema,
} from "../src/index.js";

const openAIAnnotations = [
  {
    type: "file_citation",
    file_id: "file-1",
    filename: "paper.pdf",
    index: 12,
  },
  {
    type: "url_citation",
    start_index: 4,
    end_index: 18,
    title: "Source",
    url: "https://example.com/source",
  },
  {
    type: "container_file_citation",
    container_id: "container-1",
    file_id: "file-2",
    filename: "result.csv",
    start_index: 20,
    end_index: 31,
  },
  { type: "file_path", file_id: "file-3", index: 42 },
] as const;

describe("V100 annotation superset", () => {
  it("accepts every current OpenAI Responses annotation unchanged", () => {
    for (const value of openAIAnnotations) {
      expect(annotationSchema.parse(value)).toEqual(value);
      expect(openAIResponseAnnotationSchema.parse(value)).toEqual(value);
    }
  });

  it("preserves provider-added fields on known OpenAI variants", () => {
    const value = { ...openAIAnnotations[0], provider_extension: { page: 7 } };
    expect(annotationSchema.parse(value)).toEqual(value);
  });

  it("adds comment targets without changing an OpenAI branch", () => {
    const value = {
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
    expect(commentAnnotationSchema.parse(value)).toEqual(value);
    expect(annotationSchema.parse(value)).toEqual(value);
  });

  it("rejects malformed official ranges and out-of-bounds SwarmX geometry", () => {
    expect(() => annotationSchema.parse({ ...openAIAnnotations[1], start_index: 19 })).toThrow();
    expect(() =>
      annotationSchema.parse({
        type: "comment",
        id: "annotation-1",
        comment: "Look here",
        created_at: 1,
        target: {
          type: "image_point",
          artifact_id: "artifact-1",
          project_id: "project-1",
          title: "plot.png",
          digest: `sha256:${"a".repeat(64)}`,
          mime: "image/png",
          point: { x: 1.1, y: 0.5 },
        },
      }),
    ).toThrow();
  });

  it("keeps a message quote source-addressed for future cross-session insertion", () => {
    const value = {
      type: "message_quote",
      id: "quote-1",
      created_at: 1_787_371_200_000,
      target: {
        type: "message_text",
        session_id: "source-session",
        message_seq: 42,
        message_id: "message-42",
        role: "assistant",
        text: "The selected answer fragment.",
      },
      comment: "Compare this with the earlier constraint.",
    } as const;
    expect(messageQuoteAnnotationSchema.parse(value)).toEqual(value);
    expect(annotationSchema.parse(value)).toEqual(value);
    expect(() =>
      messageQuoteAnnotationSchema.parse({
        ...value,
        target: { ...value.target, text: "" },
      }),
    ).toThrow();
    const { comment: _comment, ...withoutComment } = value;
    expect(messageQuoteAnnotationSchema.parse(withoutComment)).toEqual(withoutComment);
  });
});
