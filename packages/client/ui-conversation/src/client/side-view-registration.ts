import type { ClientContext } from "@deepseek-ai/dsh-client-runtime/client";
import type { TurnTailOwnerProps } from "@deepseek-ai/dsh-client-ui-conversation/client";
import type {} from "@deepseek-ai/dsh-client-ui-layout/client";
import { SideViewController } from "./side-view.js";
import { SideViewPanel } from "./side-view-panel.js";
import { TurnTailItems } from "./turn-tail-items.js";

/** Install the generic right-column shell and its first Tool content route. */
export function registerSideView(ctx: ClientContext): SideViewController {
  const sideView = new SideViewController(ctx.layout);
  ctx.effect(() => {
    const disposeService = ctx.reflect.provide("sideView", sideView);
    return () => {
      sideView.dispose();
      disposeService();
    };
  }, "dsh-ui-conversation: generic Side View service");

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
  );
  ctx.slots.inject("conversation.chat.turnTail", () =>
    ctx.slots.register(
      {
        name: "conversation.chat.turnTail",
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
