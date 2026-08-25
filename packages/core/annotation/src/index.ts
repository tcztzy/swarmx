import { z } from "zod";

const boundedString = z.string().min(1).max(4_096);
const safeIndex = z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER);
const sha256DigestSchema = z.string().regex(/^sha256:[0-9a-f]{64}$/u);
const pointSchema = z.strictObject({
  x: z.number().finite().min(0).max(1),
  y: z.number().finite().min(0).max(1),
});
const rectSchema = z
  .strictObject({
    x: z.number().finite().min(0).max(1),
    y: z.number().finite().min(0).max(1),
    width: z.number().finite().positive().max(1),
    height: z.number().finite().positive().max(1),
  })
  .refine((rect) => rect.x + rect.width <= 1 && rect.y + rect.height <= 1, {
    message: "annotation rectangle must remain inside its target",
  });

export const openAIFileCitationAnnotationSchema = z.looseObject({
  type: z.literal("file_citation"),
  file_id: boundedString,
  filename: boundedString,
  index: safeIndex,
});

export const openAIURLCitationAnnotationSchema = z
  .looseObject({
    type: z.literal("url_citation"),
    start_index: safeIndex,
    end_index: safeIndex,
    title: z.string().max(8_000),
    url: z.url().max(16_384),
  })
  .refine((annotation) => annotation.start_index <= annotation.end_index, {
    message: "URL citation range must be ordered",
  });

export const openAIContainerFileCitationAnnotationSchema = z
  .looseObject({
    type: z.literal("container_file_citation"),
    container_id: boundedString,
    file_id: boundedString,
    filename: boundedString,
    start_index: safeIndex,
    end_index: safeIndex,
  })
  .refine((annotation) => annotation.start_index <= annotation.end_index, {
    message: "container file citation range must be ordered",
  });

export const openAIFilePathAnnotationSchema = z.looseObject({
  type: z.literal("file_path"),
  file_id: boundedString,
  index: safeIndex,
});

export const openAIResponseAnnotationSchema = z.union([
  openAIFileCitationAnnotationSchema,
  openAIURLCitationAnnotationSchema,
  openAIContainerFileCitationAnnotationSchema,
  openAIFilePathAnnotationSchema,
]);

const documentTargetIdentity = {
  relative_path: z.string().min(1).max(4_096),
  title: z.string().trim().min(1).max(160),
  source_revision: sha256DigestSchema,
  render_revision: sha256DigestSchema,
  page: z.number().int().positive().max(100_000),
  rect: rectSchema,
};

export const annotationTargetSchema = z.discriminatedUnion("type", [
  z.strictObject({
    type: z.literal("document_text"),
    ...documentTargetIdentity,
    text: z.string().trim().min(1).max(8_000),
  }),
  z.strictObject({
    type: z.literal("document_region"),
    ...documentTargetIdentity,
    region_index: z.number().int().nonnegative().max(10_000),
    point: pointSchema,
  }),
  z.strictObject({
    type: z.literal("image_point"),
    artifact_id: boundedString,
    project_id: boundedString,
    title: z.string().trim().min(1).max(160),
    digest: sha256DigestSchema,
    mime: z.enum(["image/png", "image/jpeg", "image/gif", "image/webp"]),
    point: pointSchema,
  }),
]);

export const commentAnnotationSchema = z.strictObject({
  type: z.literal("comment"),
  id: boundedString,
  comment: z.string().trim().min(1).max(2_000),
  created_at: z.number().int().nonnegative(),
  target: annotationTargetSchema,
});

export const messageTextTargetSchema = z.strictObject({
  type: z.literal("message_text"),
  session_id: boundedString,
  message_seq: safeIndex,
  message_id: boundedString.optional(),
  role: z.enum(["user", "steering", "assistant"]),
  text: z.string().trim().min(1).max(8_000),
});

export const messageQuoteAnnotationSchema = z.strictObject({
  type: z.literal("message_quote"),
  id: boundedString,
  created_at: z.number().int().nonnegative(),
  target: messageTextTargetSchema,
  comment: z.string().trim().min(1).max(2_000).optional(),
});

/** OpenAI Responses output annotations plus SwarmX-authored comment targets. */
export const annotationSchema = z.union([
  openAIFileCitationAnnotationSchema,
  openAIURLCitationAnnotationSchema,
  openAIContainerFileCitationAnnotationSchema,
  openAIFilePathAnnotationSchema,
  commentAnnotationSchema,
  messageQuoteAnnotationSchema,
]);

export type OpenAIResponseAnnotation = z.infer<typeof openAIResponseAnnotationSchema>;
export type AnnotationTarget = z.infer<typeof annotationTargetSchema>;
export type CommentAnnotation = z.infer<typeof commentAnnotationSchema>;
export type MessageTextTarget = z.infer<typeof messageTextTargetSchema>;
export type MessageQuoteAnnotation = z.infer<typeof messageQuoteAnnotationSchema>;
export type Annotation = z.infer<typeof annotationSchema>;
