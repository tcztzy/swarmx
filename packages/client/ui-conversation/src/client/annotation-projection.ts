import type { Annotation, CommentAnnotation } from "@swarmx/annotation";
import { annotationSchema } from "@swarmx/annotation";

const LEGACY_PAPER_INSTRUCTION =
  "Use the workspace-relative Typst source and this rendered PDF context to make the requested revision. Re-open the paper preview after editing; do not infer a host path.";
const LEGACY_IMAGE_INSTRUCTION =
  "Use science_query with this exact action and request before discussing the referenced image point.";
const TAG = /<(annotation|science-paper-annotation|science-image-annotation)>([\s\S]*?)<\/\1>/gu;

export interface AnnotatedTextProjection {
  readonly body: string;
  readonly annotations: readonly Annotation[];
  readonly invalidCount: number;
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function legacyPaper(value: unknown): Annotation | null {
  const input = record(value);
  if (input === null) return null;
  const identity = {
    relative_path: input.relativePath,
    title: input.title,
    source_revision: input.sourceRevision,
    render_revision: input.pdfRevision,
    page: input.page,
    rect: input.rect,
  };
  const target =
    input.kind === "text"
      ? { type: "document_text", ...identity, text: input.selectedText }
      : input.kind === "figure-point"
        ? {
            type: "document_region",
            ...identity,
            region_index: input.figureIndex,
            point: { x: input.x, y: input.y },
          }
        : null;
  if (target === null) return null;
  const parsed = annotationSchema.safeParse({
    type: "comment",
    id: input.id,
    comment: input.comment,
    created_at: input.createdAt,
    target,
  });
  return parsed.success ? parsed.data : null;
}

function legacyImage(value: unknown): Annotation | null {
  const envelope = record(value);
  const input = record(envelope?.request ?? value);
  if (input === null) return null;
  const parsed = annotationSchema.safeParse({
    type: "comment",
    id: input.id,
    comment: input.comment,
    created_at: input.createdAt,
    target: {
      type: "image_point",
      artifact_id: input.artifactId,
      project_id: input.projectId,
      title: input.title,
      digest: input.digest,
      mime: input.mime,
      point: { x: input.x, y: input.y },
    },
  });
  return parsed.success ? parsed.data : null;
}

function parsePayload(tag: string, payload: string): Annotation | null {
  try {
    const value: unknown = JSON.parse(payload);
    if (tag === "science-paper-annotation") return legacyPaper(value);
    if (tag === "science-image-annotation") return legacyImage(value);
    const parsed = annotationSchema.safeParse(value);
    return parsed.success ? parsed.data : null;
  } catch {
    return null;
  }
}

/** Strip model-only transport markup and recover bounded annotations for display. */
export function projectAnnotatedText(text: string): AnnotatedTextProjection {
  const annotations: Annotation[] = [];
  let invalidCount = 0;
  const withoutTags = text.replace(TAG, (_match, tag: string, payload: string) => {
    const annotation = parsePayload(tag, payload);
    if (annotation === null) invalidCount += 1;
    else annotations.push(annotation);
    return "";
  });
  const body = withoutTags
    .replaceAll(LEGACY_PAPER_INSTRUCTION, "")
    .replaceAll(LEGACY_IMAGE_INSTRUCTION, "")
    .replace(/[ \t]+\n/gu, "\n")
    .replace(/\n{3,}/gu, "\n\n")
    .trim();
  return { body, annotations, invalidCount };
}

export interface AnnotationPresentation {
  readonly title: string;
  readonly meta: string | null;
  readonly quote: string | null;
  readonly comment: string | null;
  readonly href: string | null;
}

export function annotationPresentation(annotation: Annotation): AnnotationPresentation {
  if (annotation.type === "file_citation") {
    return {
      title: annotation.filename,
      meta: "File citation",
      quote: null,
      comment: null,
      href: null,
    };
  }
  if (annotation.type === "url_citation") {
    return {
      title: annotation.title || annotation.url,
      meta: "Web citation",
      quote: null,
      comment: null,
      href: annotation.url,
    };
  }
  if (annotation.type === "container_file_citation") {
    return {
      title: annotation.filename,
      meta: "Container file citation",
      quote: null,
      comment: null,
      href: null,
    };
  }
  if (annotation.type === "file_path") {
    return {
      title: annotation.file_id,
      meta: "File path",
      quote: null,
      comment: null,
      href: null,
    };
  }
  return commentPresentation(annotation);
}

function commentPresentation(annotation: CommentAnnotation): AnnotationPresentation {
  const target = annotation.target;
  if (target.type === "document_text") {
    return {
      title: target.title,
      meta: `Page ${target.page}`,
      quote: target.text,
      comment: annotation.comment,
      href: null,
    };
  }
  if (target.type === "document_region") {
    return {
      title: target.title,
      meta: `Page ${target.page} · Region ${target.region_index + 1}`,
      quote: null,
      comment: annotation.comment,
      href: null,
    };
  }
  return {
    title: target.title,
    meta: "Image point",
    quote: null,
    comment: annotation.comment,
    href: null,
  };
}

/** Human-readable clipboard projection; never emits protocol JSON. */
export function annotationCopyText(projection: AnnotatedTextProjection): string {
  const annotations = projection.annotations.map((annotation) => {
    const view = annotationPresentation(annotation);
    return [
      `> [Annotation] ${view.title}${view.meta === null ? "" : ` · ${view.meta}`}`,
      ...(view.quote === null ? [] : [`> “${view.quote}”`]),
      ...(view.comment === null ? [] : [`> ${view.comment}`]),
    ].join("\n");
  });
  return [...annotations, ...(projection.body === "" ? [] : [projection.body])].join("\n\n");
}
