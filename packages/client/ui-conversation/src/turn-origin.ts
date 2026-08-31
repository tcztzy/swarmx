/** Resolve the user-authored text that opened one DSH turn. */

import type { ChatNode, ChatSnapshot } from "@deepseek-ai/dsh-client-ui-chat/client";

/**
 * Find the first ordinary user message indexed under one turn.
 * Steering messages have their own node kind and are intentionally excluded.
 */
export function turnTextOf(
  chat: Pick<ChatSnapshot, "locations" | "nodes">,
  turn: number,
): string | undefined {
  for (const key of chat.locations.getTurn(turn)) {
    const node = chat.nodes.get(key) as ChatNode | undefined;
    if (node?.kind !== "user") continue;
    const text = plainText(node.data.content);
    if (text !== "") return text;
  }
  return undefined;
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
