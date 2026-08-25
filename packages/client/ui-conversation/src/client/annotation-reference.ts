import type { ISessions, SessionId } from "@deepseek-ai/dsh-client-runtime/client";
import type { IConversation } from "@deepseek-ai/dsh-client-ui-conversation/client";
import type {
  InputTriggerSource,
  ReferenceInsert,
} from "@deepseek-ai/dsh-client-ui-input-trigger/client";
import type { Annotation, MessageQuoteAnnotation, MessageTextTarget } from "@swarmx/annotation";
import { annotationSchema, messageQuoteAnnotationSchema } from "@swarmx/annotation";

declare module "@deepseek-ai/dsh-client-ui-input-trigger/client" {
  interface ReferenceInsert {
    readonly placement?: "inline" | "detached";
  }
}

export const ANNOTATION_REFERENCE_SOURCE = "annotation";
export const MAX_COMPOSER_ANNOTATIONS = 32;

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

export interface MessageQuoteInput {
  readonly id: string;
  readonly createdAt: number;
  readonly sourceSessionId: string;
  readonly messageSeq: number;
  readonly messageId?: string;
  readonly role: MessageTextTarget["role"];
  readonly text: string;
  readonly comment?: string;
}

export function messageQuoteAnnotation(input: MessageQuoteInput): MessageQuoteAnnotation {
  const comment = input.comment?.trim();
  return messageQuoteAnnotationSchema.parse({
    type: "message_quote",
    id: input.id,
    created_at: input.createdAt,
    target: {
      type: "message_text",
      session_id: input.sourceSessionId,
      message_seq: input.messageSeq,
      ...(input.messageId === undefined ? {} : { message_id: input.messageId }),
      role: input.role,
      text: input.text,
    },
    ...(comment === undefined || comment === "" ? {} : { comment }),
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
  if (annotation.type === "message_quote") return short(annotation.target.text);
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
    placement: "detached",
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
        return `<dsh-annotation>${safeJson(decodeAnnotationReference(ref))}</dsh-annotation>`;
      },
    },
  };
}

function sessionInput(
  conversation: Pick<IConversation, "input">,
  sessions: Pick<ISessions, "binding">,
  sessionId: SessionId,
) {
  const binding = sessions.binding(sessionId);
  return binding === undefined ? null : conversation.input.for(binding.ctx);
}

export function insertAnnotationReference(
  conversation: Pick<IConversation, "input">,
  sessions: Pick<ISessions, "binding">,
  destinationSessionId: SessionId,
  annotation: Annotation,
): boolean {
  const input = sessionInput(conversation, sessions, destinationSessionId);
  if (input === null) return false;
  const state = input.state.getSnapshot();
  if (state.phase === "adjudicating" || state.phase === "submitting") return false;
  if (
    state.occurrences.filter(({ source }) => source === ANNOTATION_REFERENCE_SOURCE).length >=
    MAX_COMPOSER_ANNOTATIONS
  ) {
    return false;
  }
  return input.insertReference(annotationReferenceInsert(annotation), {
    start: state.draft.length,
    end: state.draft.length,
    draftRev: state.draftRev,
  });
}

export function replaceAnnotationReference(
  conversation: Pick<IConversation, "input">,
  sessions: Pick<ISessions, "binding">,
  destinationSessionId: SessionId,
  occurrenceId: number,
  annotation: Annotation,
): boolean {
  const input = sessionInput(conversation, sessions, destinationSessionId);
  return input?.replaceReference(occurrenceId, annotationReferenceInsert(annotation)) ?? false;
}

export function removeAnnotationReference(
  conversation: Pick<IConversation, "input">,
  sessions: Pick<ISessions, "binding">,
  destinationSessionId: SessionId,
  occurrenceId: number,
): boolean {
  const input = sessionInput(conversation, sessions, destinationSessionId);
  return input?.removeReference(occurrenceId) ?? false;
}
