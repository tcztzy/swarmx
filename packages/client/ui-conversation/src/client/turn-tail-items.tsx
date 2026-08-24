import type { PropsRenderSlots } from "@deepseek-ai/dsh-client-ui-slots";

interface TurnTailItemsProps extends PropsRenderSlots<"conversation.chat.turnTail.items"> {
  readonly matched: number;
}

/** Render only explicitly registered additions beneath a completed Chat turn. */
export function TurnTailItems({ matched, renderSlot }: TurnTailItemsProps) {
  return renderSlot("conversation.chat.turnTail.items", { turn: matched });
}
