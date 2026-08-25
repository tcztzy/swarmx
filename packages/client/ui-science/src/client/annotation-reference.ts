import type { CommentAnnotation } from "@swarmx/annotation";
import { commentAnnotationSchema } from "@swarmx/annotation";
import type { ScienceImageAnnotation, SciencePaperAnnotation } from "@swarmx/dsh-science/types";

export function imageCommentAnnotation(annotation: ScienceImageAnnotation): CommentAnnotation {
  return commentAnnotationSchema.parse({
    type: "comment",
    id: annotation.id,
    comment: annotation.comment,
    created_at: annotation.createdAt,
    target: {
      type: "image_point",
      artifact_id: annotation.artifactId,
      project_id: annotation.projectId,
      title: annotation.title,
      digest: annotation.digest,
      mime: annotation.mime,
      point: { x: annotation.x, y: annotation.y },
    },
  });
}

export function paperCommentAnnotation(annotation: SciencePaperAnnotation): CommentAnnotation {
  const targetIdentity = {
    relative_path: annotation.relativePath,
    title: annotation.title,
    source_revision: annotation.sourceRevision,
    render_revision: annotation.pdfRevision,
    page: annotation.page,
    rect: annotation.rect,
  };
  return commentAnnotationSchema.parse({
    type: "comment",
    id: annotation.id,
    comment: annotation.comment,
    created_at: annotation.createdAt,
    target:
      annotation.kind === "text"
        ? { type: "document_text", ...targetIdentity, text: annotation.selectedText }
        : {
            type: "document_region",
            ...targetIdentity,
            region_index: annotation.figureIndex,
            point: { x: annotation.x, y: annotation.y },
          },
  });
}
