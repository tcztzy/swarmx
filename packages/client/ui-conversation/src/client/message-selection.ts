import type { ChatSnapshot } from "@deepseek-ai/dsh-client-ui-chat/client";
import type { SessionId } from "@deepseek-ai/dsh-session/types";
import type { MessageTextTarget } from "@swarmx/annotation";

const MAX_MESSAGE_SELECTION = 8_000;
const POPOVER_HALF_WIDTH = 76;
const POPOVER_GAP = 8;
const POPOVER_HEIGHT = 44;

interface RectEdges {
  readonly left: number;
  readonly top: number;
  readonly right: number;
  readonly bottom: number;
}

export interface SelectionPopoverPosition {
  readonly left: number;
  readonly top: number;
  readonly placement: "above" | "below";
}

export interface MessageSelectionCandidate extends SelectionPopoverPosition {
  readonly nodeKey: string;
  readonly text: string;
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export function messageSelectionTarget(
  session: { readonly sessionId: SessionId; readonly chat: Pick<ChatSnapshot, "nodes"> },
  nodeKey: string,
  selectedText: string,
): MessageTextTarget | null {
  const node = session.chat.nodes.get(nodeKey);
  const data = record(node?.data);
  const text = selectedText.trim().slice(0, MAX_MESSAGE_SELECTION);
  if (node === undefined || data === null || text === "") return null;

  if (node.kind === "user") {
    if (typeof data.seq !== "number") return null;
    return {
      type: "message_text",
      session_id: session.sessionId,
      message_seq: data.seq,
      role: "user",
      text,
    };
  }
  if (node.kind === "steering") {
    if (typeof data.seq !== "number") return null;
    return {
      type: "message_text",
      session_id: session.sessionId,
      message_seq: data.seq,
      ...(typeof data.messageId === "string" ? { message_id: data.messageId } : {}),
      role: "steering",
      text,
    };
  }
  if (node.kind !== "assistant-step") return null;
  const finalNode = record(data.finalNode);
  if (finalNode === null || typeof finalNode.seq !== "number") return null;
  return {
    type: "message_text",
    session_id: session.sessionId,
    message_seq: finalNode.seq,
    ...(typeof finalNode.messageId === "string" ? { message_id: finalNode.messageId } : {}),
    role: "assistant",
    text,
  };
}

export function selectionPopoverPosition(
  rect: RectEdges,
  viewport: { readonly width: number; readonly height: number },
): SelectionPopoverPosition {
  const left = Math.min(
    Math.max((rect.left + rect.right) / 2, POPOVER_HALF_WIDTH),
    Math.max(POPOVER_HALF_WIDTH, viewport.width - POPOVER_HALF_WIDTH),
  );
  if (rect.top >= POPOVER_HEIGHT + POPOVER_GAP) {
    return { left, top: rect.top - POPOVER_GAP, placement: "above" };
  }
  return {
    left,
    top: Math.min(rect.bottom + POPOVER_GAP, viewport.height - POPOVER_GAP),
    placement: "below",
  };
}

function rowOf(node: Node | null): HTMLElement | null {
  const element = node instanceof Element ? node : node?.parentElement;
  return element?.closest<HTMLElement>("[data-chat-flow-key]") ?? null;
}

export function readMessageSelection(
  selection: Selection | null,
): MessageSelectionCandidate | null {
  if (selection === null || selection.isCollapsed || selection.rangeCount !== 1) return null;
  const range = selection.getRangeAt(0);
  const startRow = rowOf(range.startContainer);
  const endRow = rowOf(range.endContainer);
  if (startRow === null || startRow !== endRow) return null;
  const kind = startRow.dataset.chatFlowKind;
  if (kind !== "user" && kind !== "steering" && kind !== "assistant-step") return null;
  const nodeKey = startRow.dataset.chatFlowKey;
  const text = selection.toString().trim().slice(0, MAX_MESSAGE_SELECTION);
  const rect = range.getBoundingClientRect();
  if (nodeKey === undefined || text === "" || (rect.width <= 0 && rect.height <= 0)) return null;
  return {
    nodeKey,
    text,
    ...selectionPopoverPosition(rect, { width: window.innerWidth, height: window.innerHeight }),
  };
}

export function shouldRequestAnnotationNote(existingCount: number): boolean {
  return existingCount > 0;
}

export function annotationNoteKeyAction(
  key: string,
  shiftKey: boolean,
  composing: boolean,
): "submit" | "cancel" | null {
  if (composing) return null;
  if (key === "Escape") return "cancel";
  return key === "Enter" && !shiftKey ? "submit" : null;
}
