import type { Context } from "@deepseek-ai/cordis";
import type {} from "@deepseek-ai/dsh-api-gateway/client";
import type { SessionId } from "@deepseek-ai/dsh-client-runtime/client";
import { TYPERT_REMOTE } from "@swarmx/dsh-swarm/remote";
import { SwarmActivityStore } from "./activity-store.js";
import { SwarmActivityView, SwarmHeaderAction, swarmSideViewEntry } from "./swarm-view.js";

export const inject = ["remote", "sideView", "slots"];

function remoteValue<T>(
  result: { ok: true; value: T } | { ok: false; error: { message: string } },
): T {
  if (result.ok) return result.value;
  throw new Error(result.error.message);
}

/** Mount the strict read-only Swarm Remote and reuse the generic per-Session Side View. */
export async function apply(ctx: Context): Promise<() => Promise<void>> {
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
            open: (snapshot: Parameters<typeof swarmSideViewEntry>[0]) =>
              swarmCtx.sideView.open(sessionId, swarmSideViewEntry(snapshot)),
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
          inject: (sessionId: SessionId) => ({ sessionId, store }),
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
    await disposeRemote();
  };
}
