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
import type { ConnectionHandle } from "@deepseek-ai/dsh-client-connection/client";
import type {} from "@deepseek-ai/dsh-client-locale/client";
import type { ClientContext, SessionId } from "@deepseek-ai/dsh-client-runtime/client";
// Type-only: pulls the ui-conversation SlotMap and ChatNodeDataMap merges.
import type {} from "@deepseek-ai/dsh-client-ui-conversation/client";
import type {} from "@deepseek-ai/dsh-client-ui-input-trigger/client";
import type {} from "@deepseek-ai/dsh-client-ui-layout/client";
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
import { RerunController } from "./controller.js";
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
  "conversationEvents",
  "connection",
  "layout",
  "inputTriggers",
  "locale",
];

/** Maximum wait for the session-added stream to reach the client runtime. */
const SESSION_READY_TIMEOUT_MS = 5_000;

/** Wait until a directly created sibling is addressable through ctx.sessions. */
async function waitForSession(ctx: ClientContext, sessionId: SessionId): Promise<void> {
  if (ctx.sessions.binding(sessionId) !== undefined) return;
  await new Promise<void>((resolve, reject) => {
    let stopped = false;
    let stop = () => {};
    const timer = setTimeout(() => {
      if (stopped) return;
      stopped = true;
      stop();
      reject(new Error(`session ${String(sessionId)} was created but did not reach the client`));
    }, SESSION_READY_TIMEOUT_MS);
    const check = () => {
      if (ctx.sessions.binding(sessionId) === undefined || stopped) return;
      stopped = true;
      clearTimeout(timer);
      stop();
      resolve();
    };
    stop = ctx.sessions.list.subscribe(check);
    if (stopped) stop();
    else check();
  });
}

/** Create an empty session in the source workspace, adopting its cwd when needed. */
async function createSibling(ctx: ClientContext, sourceId: SessionId): Promise<SessionId> {
  const source = ctx.sessions.list.getSnapshot().byId[sourceId];
  if (source === undefined) throw new Error(`source session ${String(sourceId)} is unavailable`);
  const workspaces = ctx.workspaces.list.getSnapshot().items;
  let workspace = workspaces.find((candidate) => candidate.sessionIds.includes(sourceId));
  if (workspace === undefined) {
    if (source.cwd === undefined) {
      throw new Error(`source session ${String(sourceId)} has no workspace or working directory`);
    }
    workspace = await ctx.workspaces.create({ path: source.cwd });
  }
  const connection = ctx.get("connection") as ConnectionHandle;
  const response = await connection.api.sessions.create({
    workspaceId: workspace.workspaceId,
    ...(source.agentPreset === undefined ? {} : { agentPreset: source.agentPreset }),
  });
  if (!response.result.ok) {
    throw new Error(
      `fresh session failed: ${response.result.error.code}: ${response.result.error.message}`,
    );
  }
  const sessionId = response.result.value.sessionId;
  await waitForSession(ctx, sessionId);
  return sessionId;
}

/**
 * Register the failed-turn Retry and user-message Edit entries.
 * @param ctx - client root context.
 */
export function apply(ctx: ClientContext): void {
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
        locale: "conversation",
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
        locale: "conversation",
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
          addAnnotation: (annotation) =>
            insertAnnotationReference(ctx.conversation, ctx.sessions, sessionId, annotation),
          replaceAnnotation: (occurrenceId, annotation) =>
            replaceAnnotationReference(
              ctx.conversation,
              ctx.sessions,
              sessionId,
              occurrenceId,
              annotation,
            ),
          removeAnnotation: (occurrenceId) =>
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
            prompt: (id, text) =>
              ctx.sessions.binding(id)?.session.prompt([{ type: "text", text }], "queue") ??
              Promise.reject(new Error(`session ${String(id)} is not available`)),
            binding: (id) => ctx.sessions.binding(id),
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

  ctx.conversationEvents.register(userEditDefinition);
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
