import { describe, expect, it } from "vitest";
import {
  annotationNoteKeyAction,
  messageSelectionTarget,
  selectionPopoverPosition,
  shouldRequestAnnotationNote,
} from "../src/client/message-selection.js";

function sessionWith(node: unknown) {
  return {
    sessionId: "session-1",
    chat: { nodes: { get: () => node } },
  } as never;
}

describe("V126 message selection", () => {
  it("derives durable user and finalized assistant locators", () => {
    expect(
      messageSelectionTarget(
        sessionWith({ kind: "user", data: { seq: 7 } }),
        "user-key",
        " Selected user text ",
      ),
    ).toEqual({
      type: "message_text",
      session_id: "session-1",
      message_seq: 7,
      role: "user",
      text: "Selected user text",
    });
    expect(
      messageSelectionTarget(
        sessionWith({
          kind: "assistant-step",
          data: {
            finalNode: { seq: 18, messageId: "assistant-18" },
          },
        }),
        "assistant-key",
        "Selected answer text",
      ),
    ).toEqual({
      type: "message_text",
      session_id: "session-1",
      message_seq: 18,
      message_id: "assistant-18",
      role: "assistant",
      text: "Selected answer text",
    });
  });

  it("rejects non-message and streaming-only rows", () => {
    expect(
      messageSelectionTarget(
        sessionWith({ kind: "tool-call", data: {} }),
        "tool-key",
        "tool output",
      ),
    ).toBeNull();
    expect(
      messageSelectionTarget(
        sessionWith({ kind: "assistant-step", data: {} }),
        "assistant-key",
        "partial output",
      ),
    ).toBeNull();
  });

  it("positions the action above a selection and flips below near the viewport edge", () => {
    expect(
      selectionPopoverPosition(
        { left: 100, top: 100, right: 220, bottom: 120 },
        { width: 800, height: 600 },
      ),
    ).toEqual({ left: 160, top: 92, placement: "above" });
    expect(
      selectionPopoverPosition(
        { left: -20, top: 8, right: 20, bottom: 28 },
        { width: 320, height: 600 },
      ),
    ).toEqual({ left: 76, top: 36, placement: "below" });
  });

  it("captures the first selection directly and requests an optional note thereafter", () => {
    expect(shouldRequestAnnotationNote(0)).toBe(false);
    expect(shouldRequestAnnotationNote(1)).toBe(true);
    expect(annotationNoteKeyAction("Enter", false, false)).toBe("submit");
    expect(annotationNoteKeyAction("Enter", true, false)).toBeNull();
    expect(annotationNoteKeyAction("Escape", false, false)).toBe("cancel");
  });
});
