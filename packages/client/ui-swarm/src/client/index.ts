import type { Context } from "@deepseek-ai/cordis";
import type {} from "@deepseek-ai/dsh-api-gateway/client";
import type {} from "@deepseek-ai/dsh-client-locale/client";
import type {} from "@deepseek-ai/dsh-client-ui-renderer/client";
import type { SessionId } from "@deepseek-ai/dsh-session/types";
import { TYPERT_REMOTE } from "@swarmx/dsh-swarm/remote";
import { SwarmActivityStore } from "./activity-store.js";
import { en, SWARM_LOCALE_NS, type SwarmLocaleKey, zh } from "./swarm-locales.js";
import { SwarmActivityView, SwarmHeaderAction, swarmSideViewEntry } from "./swarm-view.js";

export const inject = ["locale", "remote", "sideView", "slots"];

declare module "@deepseek-ai/dsh-client-ui-slots" {
  interface LocaleNamespaceMap {
    "swarmx.swarm": SwarmLocaleKey;
  }
}

function remoteValue<T>(result: { ok: true; value: T } | { ok: false; error: Error }): T {
  if (result.ok) return result.value;
  throw result.error;
}

/** Mount the strict read-only Swarm Remote and reuse the generic per-Session Side View. */
export async function apply(ctx: Context): Promise<() => Promise<void>> {
  const disposeLocale = ctx.locale.register(SWARM_LOCALE_NS, { en, zh });
  const t = ctx.locale.bind(SWARM_LOCALE_NS);
  const disposeRemote = await ctx.remote.$mount(TYPERT_REMOTE);
  const stores = new Set<SwarmActivityStore>();
  ctx.inject(["remote.swarm"], (swarmCtx) => {
    const store = new SwarmActivityStore({
      load: (sessionId, signal) =>
        swarmCtx.remote.swarm.uiSnapshot(sessionId, signal).then(remoteValue),
      wait: (sessionId, afterRevision, signal) =>
        swarmCtx.remote.swarm
          .waitUi(sessionId, { afterRevision, timeoutMs: 30_000 }, signal)
          .then(remoteValue),
    });
    stores.add(store);
    const disposeHeader = swarmCtx.slots.inject("conversation.session.header.actions", () =>
      swarmCtx.slots.register(
        {
          name: "conversation.session.header.actions",
          id: "swarmx-swarm-activity",
          order: -9,
          inject: (sessionId: SessionId) => ({
            store,
            t,
            open: (snapshot: Parameters<typeof swarmSideViewEntry>[0]) =>
              swarmCtx.sideView.open(sessionId, swarmSideViewEntry(snapshot, t)),
          }),
        },
        SwarmHeaderAction,
      ),
    );
    const disposeView = swarmCtx.slots.inject("side-view.content", () =>
      swarmCtx.slots.register(
        {
          name: "side-view.content",
          key: "swarm-activity",
          inject: (sessionId: SessionId) => ({ sessionId, store, t }),
        },
        SwarmActivityView,
      ),
    );
    return () => {
      disposeView();
      disposeHeader();
      store.dispose();
      stores.delete(store);
    };
  });
  return async () => {
    for (const store of stores) store.dispose();
    stores.clear();
    disposeLocale();
    await disposeRemote();
  };
}
