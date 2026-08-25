import { describe, expect, it, vi } from "vitest";
import {
  annotationReferenceInsert,
  annotationReferenceSource,
  insertAnnotationReference,
  messageQuoteAnnotation,
  removeAnnotationReference,
  replaceAnnotationReference,
} from "../src/client/annotation-reference.js";

const quote = messageQuoteAnnotation({
  id: "quote-1",
  createdAt: 1_787_371_200_000,
  sourceSessionId: "source-session",
  messageSeq: 42,
  messageId: "message-42",
  role: "assistant",
  text: "The selected answer fragment.",
});

describe("V126 V127 message quote references", () => {
  it("uses the DSH annotation codec and a detached composer occurrence", async () => {
    const source = annotationReferenceSource();
    const insert = annotationReferenceInsert(quote);
    const serialized = await source.codec?.serialize(insert.ref, new AbortController().signal);

    expect(source.name).toBe("annotation");
    expect(insert).toMatchObject({ source: "annotation", placement: "detached" });
    expect(serialized).toMatch(/^<dsh-annotation>.*<\/dsh-annotation>$/u);
    expect(serialized).toContain('"session_id":"source-session"');
    expect(serialized).toContain("The selected answer fragment.");
  });

  it("keeps source identity when inserting into a different destination session", () => {
    let state = {
      draft: "Compare this",
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
    const sessions = {
      binding: vi.fn((sessionId: string) =>
        sessionId === "destination-session" ? { ctx: {} } : undefined,
      ),
    };

    expect(insertAnnotationReference(conversation, sessions, "destination-session", quote)).toBe(
      true,
    );
    expect(sessions.binding).toHaveBeenCalledWith("destination-session");
    expect(input.setDraft).not.toHaveBeenCalled();
    expect(insertReference).toHaveBeenCalledWith(annotationReferenceInsert(quote), {
      start: 12,
      end: 12,
      draftRev: 4,
    });
    expect(quote.target.session_id).toBe("source-session");
  });

  it("edits and removes one detached occurrence by its stable identity", () => {
    const replaceReference = vi.fn(() => true);
    const removeReference = vi.fn(() => true);
    const input = { replaceReference, removeReference };
    const conversation = { input: { for: vi.fn(() => input) } };
    const sessions = { binding: vi.fn(() => ({ ctx: {} })) };
    const revised = messageQuoteAnnotation({
      id: "quote-1",
      createdAt: 1_787_371_200_000,
      sourceSessionId: "source-session",
      messageSeq: 42,
      role: "assistant",
      text: "The selected answer fragment.",
      comment: "Focus on this constraint.",
    });

    expect(
      replaceAnnotationReference(conversation, sessions, "destination-session", 9, revised),
    ).toBe(true);
    expect(replaceReference).toHaveBeenCalledWith(9, annotationReferenceInsert(revised));
    expect(removeAnnotationReference(conversation, sessions, "destination-session", 9)).toBe(true);
    expect(removeReference).toHaveBeenCalledWith(9);
  });
});
