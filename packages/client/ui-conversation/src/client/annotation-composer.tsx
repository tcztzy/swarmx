import type { ChatSnapshot } from "@deepseek-ai/dsh-client-ui-chat/client";
import {
  IconCloseOutline16,
  IconEditOutline16,
  Tooltip,
} from "@deepseek-ai/dsh-client-ui-primitives";
import type { PropsLocale, PropsRuntime } from "@deepseek-ai/dsh-client-ui-slots";
import type { Annotation } from "@swarmx/annotation";
import { type KeyboardEvent, type PointerEvent, useEffect, useMemo, useRef, useState } from "react";
import css from "./annotation-composer.module.css";
import type { ANNOTATION_LOCALE_NS, AnnotationTranslate } from "./annotation-locales.js";
import { type AnnotationPresentation, annotationPresentation } from "./annotation-projection.js";
import {
  ANNOTATION_REFERENCE_SOURCE,
  decodeAnnotationReference,
  MAX_COMPOSER_ANNOTATIONS,
  messageQuoteAnnotation,
} from "./annotation-reference.js";
import {
  annotationNoteKeyAction,
  type MessageSelectionCandidate,
  messageSelectionTarget,
  readMessageSelection,
  shouldRequestAnnotationNote,
} from "./message-selection.js";

export interface AnnotationOccurrence {
  readonly occurrenceId: number;
  readonly source: string;
  readonly ref: string;
  readonly placement?: "inline" | "detached" | undefined;
}

export interface ComposerAnnotation {
  readonly occurrenceId: number;
  readonly annotation: Annotation;
  readonly presentation: AnnotationPresentation;
}

export function composerAnnotations(
  occurrences: readonly AnnotationOccurrence[],
  t?: AnnotationTranslate,
): readonly ComposerAnnotation[] {
  const annotations: ComposerAnnotation[] = [];
  for (const occurrence of occurrences) {
    if (occurrence.source !== ANNOTATION_REFERENCE_SOURCE || occurrence.placement !== "detached") {
      continue;
    }
    try {
      const annotation = decodeAnnotationReference(occurrence.ref);
      annotations.push({
        occurrenceId: occurrence.occurrenceId,
        annotation,
        presentation: annotationPresentation(annotation, t),
      });
    } catch {}
  }
  return annotations;
}

function withComment(annotation: Annotation, value: string): Annotation | null {
  const comment = value.trim();
  if (annotation.type === "message_quote") {
    const { comment: _comment, ...rest } = annotation;
    return comment === "" ? rest : { ...rest, comment };
  }
  if (annotation.type !== "comment" || comment === "") return null;
  return { ...annotation, comment };
}

function annotationComment(annotation: Annotation): string | null {
  if (annotation.type === "message_quote") return annotation.comment ?? "";
  if (annotation.type === "comment") return annotation.comment;
  return null;
}

function stopSelectionLoss(event: PointerEvent<HTMLElement>): void {
  event.preventDefault();
}

function focusEditor(element: HTMLTextAreaElement | null): void {
  element?.focus();
}

interface AnnotationTrayProps {
  readonly annotations: readonly ComposerAnnotation[];
  readonly open: boolean;
  readonly editingId: number | null;
  readonly editValue: string;
  readonly onToggle: () => void;
  readonly onBeginEdit: (item: ComposerAnnotation) => void;
  readonly onEditValue: (value: string) => void;
  readonly onCommitEdit: () => void;
  readonly onCancelEdit: () => void;
  readonly onRemove: (occurrenceId: number) => void;
  readonly t: AnnotationTranslate;
}

export function AnnotationTray({
  annotations,
  open,
  editingId,
  editValue,
  onToggle,
  onBeginEdit,
  onEditValue,
  onCommitEdit,
  onCancelEdit,
  onRemove,
  t,
}: AnnotationTrayProps) {
  if (annotations.length === 0) return null;
  const countLabel = t(annotations.length === 1 ? "tray.countOne" : "tray.countMany", {
    count: annotations.length,
  });
  return (
    <div className={css.tray} data-annotation-tray>
      <button
        type="button"
        className={css.countButton}
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={onToggle}
      >
        {countLabel}
      </button>
      {open && (
        <section className={css.list} role="dialog" aria-label={t("tray.dialog")}>
          {annotations.map((item, itemIndex) => {
            const index = itemIndex + 1;
            const comment = annotationComment(item.annotation);
            const editing = editingId === item.occurrenceId;
            return (
              <article className={css.item} key={item.occurrenceId}>
                <span className={css.itemIndex}>{index}.</span>
                <div className={css.itemBody}>
                  <span className={css.itemLabel}>
                    {item.annotation.type === "message_quote"
                      ? t("tray.selectedText")
                      : item.presentation.title}
                  </span>
                  {item.presentation.quote !== null && (
                    <blockquote>{item.presentation.quote}</blockquote>
                  )}
                  {item.presentation.quote === null && item.presentation.meta !== null && (
                    <p>{item.presentation.meta}</p>
                  )}
                  {!editing && item.presentation.comment !== null && (
                    <p className={css.comment}>{item.presentation.comment}</p>
                  )}
                  {editing && (
                    <textarea
                      ref={focusEditor}
                      className={css.inlineEditor}
                      aria-label={t("tray.editLabel")}
                      value={editValue}
                      maxLength={2_000}
                      onChange={(event) => onEditValue(event.currentTarget.value)}
                      onKeyDown={(event) => {
                        const action = annotationNoteKeyAction(
                          event.key,
                          event.shiftKey,
                          event.nativeEvent.isComposing,
                        );
                        if (action === null) return;
                        event.preventDefault();
                        if (action === "submit") onCommitEdit();
                        else onCancelEdit();
                      }}
                    />
                  )}
                </div>
                <div className={css.itemActions}>
                  {comment !== null && (
                    <Tooltip label={t("tray.edit", { index })} side="top">
                      <button
                        type="button"
                        aria-label={t("tray.edit", { index })}
                        onClick={() => onBeginEdit(item)}
                      >
                        <IconEditOutline16 size={15} />
                      </button>
                    </Tooltip>
                  )}
                  <Tooltip label={t("tray.remove", { index })} side="top">
                    <button
                      type="button"
                      aria-label={t("tray.remove", { index })}
                      onClick={() => onRemove(item.occurrenceId)}
                    >
                      <IconCloseOutline16 size={15} />
                    </button>
                  </Tooltip>
                </div>
              </article>
            );
          })}
        </section>
      )}
    </div>
  );
}

interface AnnotationComposerInjected {
  readonly addAnnotation: (annotation: Annotation) => boolean;
  readonly replaceAnnotation: (occurrenceId: number, annotation: Annotation) => boolean;
  readonly removeAnnotation: (occurrenceId: number) => boolean;
}

type AnnotationComposerProps = PropsRuntime<"conversation.input.dock"> &
  PropsLocale<typeof ANNOTATION_LOCALE_NS> &
  AnnotationComposerInjected;

function selectionStyle(candidate: MessageSelectionCandidate) {
  return { left: candidate.left, top: candidate.top };
}

export function AnnotationComposer({
  sessionId,
  useChat,
  input,
  addAnnotation,
  replaceAnnotation,
  removeAnnotation,
  t,
}: AnnotationComposerProps) {
  const chat = useChat((snapshot: ChatSnapshot) => snapshot);
  const annotationT = t as AnnotationTranslate;
  const annotations = useMemo(
    () => composerAnnotations(input.occurrences, annotationT),
    [annotationT, input.occurrences],
  );
  const [candidate, setCandidate] = useState<MessageSelectionCandidate | null>(null);
  const [noteOpen, setNoteOpen] = useState(false);
  const [note, setNote] = useState("");
  const [trayOpen, setTrayOpen] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editValue, setEditValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const trayRef = useRef<HTMLDivElement | null>(null);
  const noteRef = useRef<HTMLTextAreaElement | null>(null);
  const target =
    candidate === null
      ? null
      : messageSelectionTarget({ sessionId, chat }, candidate.nodeKey, candidate.text);

  useEffect(() => {
    const update = () => {
      if (noteOpen) return;
      setCandidate(readMessageSelection(window.getSelection()));
      setError(null);
    };
    const dismiss = () => {
      if (!noteOpen) setCandidate(null);
    };
    document.addEventListener("pointerup", update);
    document.addEventListener("keyup", update);
    document.addEventListener("scroll", dismiss, true);
    window.addEventListener("resize", dismiss);
    return () => {
      document.removeEventListener("pointerup", update);
      document.removeEventListener("keyup", update);
      document.removeEventListener("scroll", dismiss, true);
      window.removeEventListener("resize", dismiss);
    };
  }, [noteOpen]);

  useEffect(() => {
    if (!trayOpen) return;
    const closeOutside = (event: globalThis.PointerEvent) => {
      const node = event.target;
      if (node instanceof Node && trayRef.current?.contains(node) !== true) setTrayOpen(false);
    };
    document.addEventListener("pointerdown", closeOutside);
    return () => document.removeEventListener("pointerdown", closeOutside);
  }, [trayOpen]);

  useEffect(() => {
    if (noteOpen) noteRef.current?.focus();
  }, [noteOpen]);

  function clearSelection(): void {
    window.getSelection()?.removeAllRanges();
    setCandidate(null);
    setNoteOpen(false);
    setNote("");
    setError(null);
  }

  function commitCandidate(comment?: string): void {
    if (target === null) return;
    if (annotations.length >= MAX_COMPOSER_ANNOTATIONS) {
      setError(annotationT("error.limit", { count: MAX_COMPOSER_ANNOTATIONS }));
      return;
    }
    const inserted = addAnnotation(
      messageQuoteAnnotation({
        id: crypto.randomUUID(),
        createdAt: Date.now(),
        sourceSessionId: target.session_id,
        messageSeq: target.message_seq,
        ...(target.message_id === undefined ? {} : { messageId: target.message_id }),
        role: target.role,
        text: target.text,
        ...(comment === undefined ? {} : { comment }),
      }),
    );
    if (inserted) clearSelection();
    else setError(annotationT("error.add"));
  }

  function addSelection(): void {
    if (shouldRequestAnnotationNote(annotations.length)) {
      setNoteOpen(true);
      return;
    }
    commitCandidate();
  }

  function beginEdit(item: ComposerAnnotation): void {
    const comment = annotationComment(item.annotation);
    if (comment === null) return;
    setEditingId(item.occurrenceId);
    setEditValue(comment);
  }

  function commitEdit(): void {
    const item = annotations.find(({ occurrenceId }) => occurrenceId === editingId);
    if (item === undefined) return;
    const updated = withComment(item.annotation, editValue);
    if (updated === null || !replaceAnnotation(item.occurrenceId, updated)) return;
    setEditingId(null);
    setEditValue("");
  }

  function remove(occurrenceId: number): void {
    if (!removeAnnotation(occurrenceId)) return;
    if (editingId === occurrenceId) {
      setEditingId(null);
      setEditValue("");
    }
  }

  return (
    <div ref={trayRef} className={css.root}>
      <AnnotationTray
        annotations={annotations}
        open={trayOpen}
        editingId={editingId}
        editValue={editValue}
        onToggle={() => setTrayOpen((open) => !open)}
        onBeginEdit={beginEdit}
        onEditValue={setEditValue}
        onCommitEdit={commitEdit}
        onCancelEdit={() => {
          setEditingId(null);
          setEditValue("");
        }}
        onRemove={remove}
        t={annotationT}
      />
      {candidate !== null && target !== null && !noteOpen && (
        <div
          className={css.selectionAction}
          data-placement={candidate.placement}
          style={selectionStyle(candidate)}
          role="toolbar"
          aria-label={annotationT("selection.add")}
          onPointerDown={stopSelectionLoss}
        >
          <button type="button" onClick={addSelection}>
            {annotationT("selection.add")}
          </button>
        </div>
      )}
      {candidate !== null && target !== null && noteOpen && (
        <form
          className={css.noteEditor}
          data-placement={candidate.placement}
          style={selectionStyle(candidate)}
          onPointerDown={(event) => event.stopPropagation()}
          onSubmit={(event) => {
            event.preventDefault();
            commitCandidate(note);
          }}
        >
          <label htmlFor={`annotation-note-${target.message_seq}`}>
            {annotationT("selection.noteLabel")}
          </label>
          <textarea
            ref={noteRef}
            id={`annotation-note-${target.message_seq}`}
            value={note}
            maxLength={2_000}
            placeholder={annotationT("selection.notePlaceholder")}
            onChange={(event) => setNote(event.currentTarget.value)}
            onKeyDown={(event: KeyboardEvent<HTMLTextAreaElement>) => {
              const action = annotationNoteKeyAction(
                event.key,
                event.shiftKey,
                event.nativeEvent.isComposing,
              );
              if (action === null) return;
              event.preventDefault();
              if (action === "submit") commitCandidate(note);
              else clearSelection();
            }}
          />
          <div className={css.noteActions}>
            <button type="button" onClick={clearSelection}>
              {annotationT("selection.cancel")}
            </button>
            <button type="submit">{annotationT("selection.confirm")}</button>
          </div>
        </form>
      )}
      {error !== null && (
        <p className={css.error} role="status">
          {error}
        </p>
      )}
    </div>
  );
}
