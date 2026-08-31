import type { Context } from "@deepseek-ai/cordis";
import type {} from "@deepseek-ai/dsh-api-gateway/client";
import type {} from "@deepseek-ai/dsh-client-ui-renderer/client";
import type { SessionId } from "@deepseek-ai/dsh-session/types";
import type { DvcUiSnapshot } from "@swarmx/dsh-ui-dvc/contracts";
import type {} from "@swarmx/dsh-ui-dvc/remote";
import type { GitUiSnapshot } from "../contracts.js";
import { GIT_UI_REMOTE } from "../remote.js";
import {
  VersionControlHeaderAction,
  type VersionControlSnapshot,
  VersionControlView,
  versionControlSideViewEntry,
} from "./version-control-view.js";

export const inject = ["remote", "sideView", "slots"];

function remoteValue<T>(result: { ok: true; value: T } | { ok: false; error: Error }): T {
  if (result.ok) return result.value;
  throw result.error;
}

/** Mount the read-only Git Remote and the shared Git/DVC Version Control presentation. */
export async function apply(ctx: Context): Promise<() => Promise<void>> {
  const disposeRemote = await ctx.remote.$mount(GIT_UI_REMOTE);
  ctx.inject(["remote.gitUi", "remote.dvcUi"], (versionControlCtx) => {
    const loadGit = (sessionId: SessionId, signal?: AbortSignal): Promise<GitUiSnapshot> =>
      versionControlCtx.remote.gitUi.snapshot(sessionId, signal).then(remoteValue);
    const loadDvc = (sessionId: SessionId, signal?: AbortSignal): Promise<DvcUiSnapshot> =>
      versionControlCtx.remote.dvcUi.snapshot(sessionId, signal).then(remoteValue);
    const load = async (
      sessionId: SessionId,
      signal?: AbortSignal,
    ): Promise<VersionControlSnapshot> => {
      const [git, dvc] = await Promise.all([
        loadGit(sessionId, signal),
        loadDvc(sessionId, signal).catch((error: unknown) => {
          if (signal?.aborted) throw error;
          return null;
        }),
      ]);
      return { git, dvc: dvc?.kind === "project" ? dvc.inspection : null };
    };

    versionControlCtx.slots.inject("conversation.session.header.actions", () =>
      versionControlCtx.slots.register(
        {
          name: "conversation.session.header.actions",
          id: "swarmx-version-control",
          order: -10,
          inject: (sessionId: SessionId) => ({
            load: (signal?: AbortSignal) => load(sessionId, signal),
            open: (snapshot: VersionControlSnapshot) =>
              versionControlCtx.sideView.open(sessionId, versionControlSideViewEntry(snapshot)),
          }),
        },
        VersionControlHeaderAction,
      ),
    );
    return versionControlCtx.slots.inject("side-view.content", () =>
      versionControlCtx.slots.register(
        {
          name: "side-view.content",
          key: "version-control",
          inject: (sessionId: SessionId) => ({
            load: (signal?: AbortSignal) => load(sessionId, signal),
          }),
        },
        VersionControlView,
      ),
    );
  });
  return async () => disposeRemote();
}
