/** Resolve the user-authored text that opened one DSH turn. */

import type { ConversationNode, UserMessageNode } from "@deepseek-ai/dsh-client-runtime/client";
import type { TimelineTurn } from "./fork-boundary.js";

/** Exact DSH conversation-node union consumed by the lookup. */
export type LookupNode = ConversationNode;

/**
 * Find the first ordinary user message inside one turn's event boundaries.
 * Steering messages have their own node kind and are intentionally excluded.
 */
export function turnTextOf(nodes: readonly LookupNode[], turn: TimelineTurn): string | undefined {
  const start = turn.start?.seq;
  if (start === undefined) return undefined;
  const end = turn.end?.seq ?? Number.POSITIVE_INFINITY;
  const opening = nodes.find(
    (node): node is UserMessageNode => node.kind === "user" && node.seq > start && node.seq < end,
  );
  if (opening === undefined) return undefined;
  const text = plainText(opening.content);
  return text === "" ? undefined : text;
}

/** Concatenate the text blocks of a message, dropping images and other forms. */
export function plainText(
  content: readonly { readonly type: string; readonly text?: string }[],
): string {
  return content
    .filter((block) => block.type === "text")
    .map((block) => block.text ?? "")
    .join("")
    .trim();
}
