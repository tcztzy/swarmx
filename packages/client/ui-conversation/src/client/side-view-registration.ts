import type { Context } from "@deepseek-ai/cordis";
import type { TurnTailOwnerProps } from "@deepseek-ai/dsh-client-ui-chat/client";
import type {} from "@deepseek-ai/dsh-client-ui-layout/client";
import type {} from "@deepseek-ai/dsh-client-ui-renderer/client";
import { SideViewController } from "./side-view.js";
import { SideViewPanel } from "./side-view-panel.js";
import { TurnTailItems } from "./turn-tail-items.js";

/** Install the generic right-column shell and its first Tool content route. */
export function registerSideView(ctx: Context): SideViewController {
  const sideView = new SideViewController(ctx.layout);
  ctx.effect(() => {
    const disposeService = ctx.reflect.provide("sideView", sideView);
    return () => {
      sideView.dispose();
      disposeService();
    };
  }, "dsh-ui-conversation: generic Side View service");

  ctx.slots.inject("details", () =>
    ctx.slots.register(
      {
        name: "details",
        priority: -10,
        children: {
          "side-view.content": { kind: "keyed", scope: "session" },
        },
        inject: () => ({ sideView }),
      },
      SideViewPanel,
    ),
  );
  ctx.slots.inject("conversation.chat.turnTail", () =>
    ctx.slots.register(
      {
        name: "conversation.chat.turnTail",
        priority: Number.MAX_SAFE_INTEGER,
        select: (owner: TurnTailOwnerProps) => owner.turn.turn,
        children: {
          "conversation.chat.turnTail.items": { kind: "list", scope: "session" },
        },
      },
      TurnTailItems,
    ),
  );
  return sideView;
}
