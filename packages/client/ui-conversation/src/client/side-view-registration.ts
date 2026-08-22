import type { ClientContext, ToolCallBlock } from "@deepseek-ai/dsh-client-runtime/client";
import type { TurnTailOwnerProps } from "@deepseek-ai/dsh-client-ui-conversation/client";
import type {} from "@deepseek-ai/dsh-client-ui-layout/client";
import { SideViewController } from "./side-view.js";
import { SideViewPanel } from "./side-view-panel.js";
import { ToolDetailsAction, ToolSideViewContent, toolSideViewEntry } from "./tool-side-view.js";

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
  ctx.slots.inject("side-view.content", () =>
    ctx.slots.register(
      {
        name: "side-view.content",
        key: "tool",
        children: {
          "side-view.tool.actions": { kind: "list", scope: "session" },
        },
      },
      ToolSideViewContent,
    ),
  );
  ctx.slots.inject("conversation.chat.turnTail", () =>
    ctx.slots.register(
      {
        name: "conversation.chat.turnTail",
        select: (owner: TurnTailOwnerProps) => owner.turn.turn,
        inject: (sessionId) => ({
          openTool: (block: ToolCallBlock) => sideView.open(sessionId, toolSideViewEntry(block)),
        }),
      },
      ToolDetailsAction,
    ),
  );
  return sideView;
}
