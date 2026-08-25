import type { SessionId } from "@deepseek-ai/dsh-session";
import type { RemoteResult, TypertRemoteContribution } from "@deepseek-ai/dsh-typert-protocol";
import type { DvcUiSnapshot } from "./contracts.js";
import { DVC_UI_INVOCATIONS } from "./remote-contract.js";

interface DvcUiRemoteNamespace {
  snapshot(sessionId: SessionId, signal?: AbortSignal): Promise<RemoteResult<DvcUiSnapshot>>;
}

declare module "@deepseek-ai/dsh-typert-protocol" {
  interface TypertRemoteMap {
    "dvcUi/snapshot": DvcUiRemoteNamespace["snapshot"];
  }

  interface TypertRemoteNamespaceMap {
    dvcUi: DvcUiRemoteNamespace;
  }
}

export const DVC_UI_REMOTE: TypertRemoteContribution = Object.freeze({
  package: "@swarmx/dsh-ui-dvc",
  descriptors: DVC_UI_INVOCATIONS,
});

export default DVC_UI_REMOTE;
