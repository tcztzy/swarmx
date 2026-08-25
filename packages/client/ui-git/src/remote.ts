import type { SessionId } from "@deepseek-ai/dsh-session";
import type { RemoteResult, TypertRemoteContribution } from "@deepseek-ai/dsh-typert-protocol";
import type { GitUiSnapshot } from "./contracts.js";
import { GIT_UI_INVOCATIONS } from "./remote-contract.js";

interface GitUiRemoteNamespace {
  snapshot(sessionId: SessionId, signal?: AbortSignal): Promise<RemoteResult<GitUiSnapshot>>;
}

declare module "@deepseek-ai/dsh-typert-protocol" {
  interface TypertRemoteMap {
    "gitUi/snapshot": GitUiRemoteNamespace["snapshot"];
  }

  interface TypertRemoteNamespaceMap {
    gitUi: GitUiRemoteNamespace;
  }
}

export const GIT_UI_REMOTE: TypertRemoteContribution = Object.freeze({
  package: "@swarmx/dsh-ui-git",
  descriptors: GIT_UI_INVOCATIONS,
});

export default GIT_UI_REMOTE;
