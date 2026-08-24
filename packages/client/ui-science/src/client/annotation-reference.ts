import type { ISessions, SessionId } from "@deepseek-ai/dsh-client-runtime/client";
import type { IConversation } from "@deepseek-ai/dsh-client-ui-conversation/client";
import type {
  InputTriggerSource,
  ReferenceInsert,
} from "@deepseek-ai/dsh-client-ui-input-trigger/client";
import type { Annotation, CommentAnnotation } from "@swarmx/annotation";
import { annotationSchema, commentAnnotationSchema } from "@swarmx/annotation";
import type { ScienceImageAnnotation, SciencePaperAnnotation } from "@swarmx/dsh-science/types";

export const ANNOTATION_REFERENCE_SOURCE = "annotation";

function safeJson(value: unknown): string {
  return JSON.stringify(value)
    .replaceAll("&", "\\u0026")
    .replaceAll("<", "\\u003c")
    .replaceAll(">", "\\u003e");
}

export function encodeAnnotationReference(annotation: Annotation): string {
  return JSON.stringify(annotationSchema.parse(annotation));
}

export function decodeAnnotationReference(ref: string): Annotation {
  try {
    return annotationSchema.parse(JSON.parse(ref));
  } catch (error) {
    throw new Error("Invalid annotation reference", { cause: error });
  }
}

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

function short(value: string, maximum = 72): string {
  return value.length <= maximum ? value : `${value.slice(0, maximum - 1)}…`;
}

function annotationLabel(annotation: Annotation): string {
  if (annotation.type === "file_citation") return annotation.filename;
  if (annotation.type === "url_citation") return annotation.title || annotation.url;
  if (annotation.type === "container_file_citation") return annotation.filename;
  if (annotation.type === "file_path") return annotation.file_id;
  const target = annotation.target;
  const context =
    target.type === "document_text"
      ? `${target.title} · p.${target.page} · ${target.text}`
      : target.type === "document_region"
        ? `${target.title} · p.${target.page} · Region ${target.region_index + 1}`
        : target.title;
  return short(`${context} · ${annotation.comment}`);
}

export function annotationReferenceInsert(annotation: Annotation): ReferenceInsert {
  const parsed = annotationSchema.parse(annotation);
  const label = annotationLabel(parsed);
  return {
    source: ANNOTATION_REFERENCE_SOURCE,
    ref: encodeAnnotationReference(parsed),
    label,
    appearance: "file",
    clipboardText: `@${label}`,
  };
}

export function annotationReferenceSource(): InputTriggerSource {
  return {
    trigger: "@",
    name: ANNOTATION_REFERENCE_SOURCE,
    showGroupTitle: false,
    candidates: async () => [],
    onPick: () => undefined,
    codec: {
      clipboardText: (ref) =>
        annotationReferenceInsert(decodeAnnotationReference(ref)).clipboardText,
      async serialize(ref, signal) {
        signal.throwIfAborted();
        return `<annotation>${safeJson(decodeAnnotationReference(ref))}</annotation>`;
      },
    },
  };
}

export function insertAnnotationReference(
  conversation: Pick<IConversation, "input">,
  sessions: Pick<ISessions, "binding">,
  sessionId: SessionId,
  annotation: Annotation,
): boolean {
  const binding = sessions.binding(sessionId);
  if (!binding) return false;
  const input = conversation.input.for(binding.ctx);
  const before = input.state.getSnapshot();
  if (before.phase === "adjudicating" || before.phase === "submitting") return false;
  const separator = before.draft.length > 0 && !/\s$/u.test(before.draft) ? " " : "";
  const triggerStart = before.draft.length + separator.length;
  const withTrigger = `${before.draft}${separator}@`;
  input.setDraft(withTrigger);
  const staged = input.state.getSnapshot();
  const inserted = input.insertReference(annotationReferenceInsert(annotation), {
    start: triggerStart,
    end: triggerStart + 1,
    draftRev: staged.draftRev,
  });
  if (!inserted && input.state.getSnapshot().draft === withTrigger) input.setDraft(before.draft);
  return inserted;
}
