/**
 * Conversation extensions for DeepSeek Harness: generic Side View plus terminal
 * failure Retry and user-authored message Edit.
 *
 * Neither action mutates history. Later turns fork at the preceding completed
 * boundary; the first turn starts in a fresh sibling because DSH cannot fork
 * an empty completed-turn prefix.
 *
 * Nothing in DSH is modified: user Edit is a derived conversation node and
 * terminal errors enter through the published turn-tail extension chain.
 * @module @swarmx/dsh-ui-conversation/client
 */
import type { Context } from "@deepseek-ai/cordis";
import type {} from "@deepseek-ai/dsh-api-session-controller/client";
import type {} from "@deepseek-ai/dsh-client-locale/client";
// Type-only: pulls the Chat SlotMap and ChatNodeDataMap merges.
import type {} from "@deepseek-ai/dsh-client-ui-chat/client";
import type {} from "@deepseek-ai/dsh-client-ui-conversation/client";
import type {} from "@deepseek-ai/dsh-client-ui-input-trigger/client";
import type {} from "@deepseek-ai/dsh-client-ui-layout/client";
import type {} from "@deepseek-ai/dsh-client-ui-renderer/client";
import type { SessionId } from "@deepseek-ai/dsh-session/types";
import type { Annotation } from "@swarmx/annotation";
import { selectFailedTurn } from "../error-turn.js";
import { userEditDefinition } from "../user-edit-node.js";
import { FailedTurnAction, UserEditAction } from "./actions.js";
import { AnnotationComposer } from "./annotation-composer.js";
import {
  ANNOTATION_LOCALE_NS,
  type AnnotationLocaleKey,
  en as annotationEn,
  zh as annotationZh,
} from "./annotation-locales.js";
import {
  annotationReferenceSource,
  insertAnnotationReference,
  removeAnnotationReference,
  replaceAnnotationReference,
} from "./annotation-reference.js";
import {
  AnnotationSteeringMessageView,
  AnnotationUserMessageView,
} from "./annotation-user-message.js";
import { createSibling, RerunController } from "./controller.js";
import { registerPeerRuntimeConversation } from "./runtime-conversation.js";
import { registerSideView } from "./side-view-registration.js";
import type { RerunActionsInjected } from "./slots.js";

export {
  ANNOTATION_REFERENCE_SOURCE,
  annotationReferenceInsert,
  annotationReferenceSource,
  decodeAnnotationReference,
  encodeAnnotationReference,
  insertAnnotationReference,
  MAX_COMPOSER_ANNOTATIONS,
  messageQuoteAnnotation,
  removeAnnotationReference,
  replaceAnnotationReference,
} from "./annotation-reference.js";
export { RerunController } from "./controller.js";
export {
  type ISideView,
  type JsonValue,
  type SideViewContentOwnerProps,
  SideViewController,
  type SideViewEntry,
  type SideViewMode,
  type SideViewSnapshot,
  type TurnTailItemOwnerProps,
} from "./side-view.js";
export type { RerunActionsInjected } from "./slots.js";

declare module "@deepseek-ai/dsh-client-ui-slots" {
  interface LocaleNamespaceMap {
    "swarmx.annotation": AnnotationLocaleKey;
  }
}

/** Services required before this plugin can register its entries. */
export const inject = [
  "slots",
  "sessions",
  "workspaces",
  "conversation",
  "uiConversation",
  "layout",
  "inputTriggers",
  "locale",
];

/**
 * Register the failed-turn Retry and user-message Edit entries.
 * @param ctx - client root context.
 */
export function apply(ctx: Context): void {
  registerPeerRuntimeConversation(ctx);
  registerSideView(ctx);
  ctx.effect(
    () =>
      ctx.locale.register(ANNOTATION_LOCALE_NS, {
        zh: annotationZh,
        en: annotationEn,
      }),
    "dsh-ui-conversation: annotation dictionaries",
  );
  const annotationT = ctx.locale.bind(ANNOTATION_LOCALE_NS);
  const disposeAnnotationSource = ctx.inputTriggers.registerSource(annotationReferenceSource());
  ctx.effect(() => disposeAnnotationSource, "dsh-ui-conversation: annotation reference source");
  ctx.slots.inject("conversation.chat.node", () =>
    ctx.slots.register(
      {
        name: "conversation.chat.node",
        key: "user",
        priority: -10,
        locale: "chat",
        inject: () => ({ annotationT }),
      },
      AnnotationUserMessageView,
    ),
  );
  ctx.slots.inject("conversation.chat.node", () =>
    ctx.slots.register(
      {
        name: "conversation.chat.node",
        key: "steering",
        priority: -10,
        locale: "chat",
        inject: () => ({ annotationT }),
      },
      AnnotationSteeringMessageView,
    ),
  );
  ctx.slots.inject("conversation.input.dock", () =>
    ctx.slots.register(
      {
        name: "conversation.input.dock",
        id: "swarmx-annotations",
        order: -20,
        locale: ANNOTATION_LOCALE_NS,
        inject: (sessionId: SessionId) => ({
          addAnnotation: (annotation: Annotation) =>
            insertAnnotationReference(ctx.conversation, ctx.sessions, sessionId, annotation),
          replaceAnnotation: (occurrenceId: number, annotation: Annotation) =>
            replaceAnnotationReference(
              ctx.conversation,
              ctx.sessions,
              sessionId,
              occurrenceId,
              annotation,
            ),
          removeAnnotation: (occurrenceId: number) =>
            removeAnnotationReference(ctx.conversation, ctx.sessions, sessionId, occurrenceId),
        }),
      },
      AnnotationComposer,
    ),
  );
  const controllers = new Map<SessionId, RerunController>();
  const controllerFor = (sessionId: SessionId): RerunController => {
    let controller = controllers.get(sessionId);
    if (controller === undefined) {
      controller = new RerunController(
        {
          sessions: {
            createSibling: (sourceId) => createSibling(ctx, sourceId),
            fork: (opts) => ctx.sessions.fork(opts),
            open: (id) => ctx.sessions.open(id),
            prompt: async (id, text) => {
              const session = ctx.sessions.binding(id)?.session;
              if (session === undefined) {
                throw new Error(`session ${String(id)} is not available`);
              }
              const result = await session.prompt([{ type: "text", text }], "queue");
              if (!result.ok) throw result.error;
            },
          },
          snapshot: (id) => {
            const binding = ctx.sessions.binding(id);
            if (binding === undefined) return undefined;
            const chat = ctx.uiConversation.binding(binding).target("chat").getSnapshot();
            return chat === undefined
              ? undefined
              : { chat, hasMore: binding.session.getSnapshot().hasMore };
          },
          setDraft: (id, text) => {
            const scoped = ctx.sessions.binding(id)?.ctx;
            if (scoped !== undefined) ctx.conversation.input.for(scoped).setDraft(text);
          },
        },
        sessionId,
      );
      controllers.set(sessionId, controller);
    }
    return controller;
  };

  const face = (sessionId: SessionId): RerunActionsInjected => {
    const controller = controllerFor(sessionId);
    return {
      canRerun: (turn) => controller.canRerun(turn),
      rerun: (turn) => controller.rerun(turn),
      beginEdit: (turn, text) => controller.beginEdit(turn, text),
    };
  };

  ctx.uiConversation.events.register(userEditDefinition);
  ctx.slots.inject("conversation.chat.node", () =>
    ctx.slots.register(
      {
        name: "conversation.chat.node",
        key: "swarmx-user-edit",
        inject: face,
      },
      UserEditAction,
    ),
  );
  ctx.slots.inject("conversation.chat.turnTail", () =>
    ctx.slots.register(
      {
        name: "conversation.chat.turnTail",
        select: selectFailedTurn,
        inject: face,
      },
      FailedTurnAction,
    ),
  );
  ctx.effect(() => () => controllers.clear(), "dsh-ui-conversation: controller cache");
}
