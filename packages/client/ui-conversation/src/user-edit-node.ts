/** A tiny Chat contribution that places Edit immediately after a user message. */

import type {
  ChatConversationViewNode,
  ConversationLocation,
  ConversationNodeContext,
  ConversationNodeDefinition,
} from "@deepseek-ai/dsh-client-runtime/client";
import type {} from "@deepseek-ai/dsh-client-ui-conversation/client";
import { plainText } from "./turn-origin.js";

/** Renderer payload for one user-message Edit icon. */
export interface UserEditNodeData {
  readonly turn: number;
  readonly text: string;
}

declare module "@deepseek-ai/dsh-client-ui-conversation/client" {
  interface ChatNodeDataMap {
    /** Edit action associated with an ordinary user-authored message. */
    "swarmx-user-edit": UserEditNodeData;
  }
}

interface UserEditState {
  readonly seq: number;
  readonly text: string;
}

function contextLocation(context: ConversationNodeContext): ConversationLocation {
  return context.start?.location ?? context.matches[0]?.location ?? { kind: "unresolved" };
}

/** User-message action Definition using DSH's public Conversation registry. */
export const userEditDefinition: ConversationNodeDefinition<UserEditState> = {
  kind: "swarmx-user-edit",
  target: "chat",
  match: (event) =>
    event.type === "user/message" &&
    event.surfaceOp === "append" &&
    event.data.source.kind === "user"
      ? { id: String(event.data.id), role: "start" }
      : null,
  start: (_context, match) => {
    if (match.event.type !== "user/message") {
      throw new Error("swarmx-user-edit start requires user/message");
    }
    return { seq: match.event.seq, text: plainText(match.event.data.content) };
  },
  update: (context) => context.state,
  buildViewNode: (context): ChatConversationViewNode | null => {
    const state = context.state;
    if (state === undefined || state.text === "") return null;
    const location = contextLocation(context);
    if (location.kind !== "turn" && location.kind !== "step") return null;
    return {
      key: context.key,
      kind: "swarmx-user-edit",
      id: context.id,
      target: "chat",
      anchorSeq: state.seq + 0.01,
      location,
      visibility: "visible",
      data: { turn: location.turn.turn, text: state.text },
    };
  },
};
