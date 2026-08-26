import type { SessionId } from "@deepseek-ai/dsh-session";
import type { RemoteResult, TypertRemoteContribution } from "@deepseek-ai/dsh-typert-protocol";
import type { SwarmUiSnapshot, WaitForSwarmChangeRequest } from "./contracts.js";
import { SWARM_INVOCATIONS } from "./remote-contract.js";

interface SwarmRemoteNamespace {
  uiSnapshot: (
    sessionId: SessionId,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<SwarmUiSnapshot>>;
  waitUi: (
    sessionId: SessionId,
    request: WaitForSwarmChangeRequest,
    signal?: AbortSignal,
  ) => Promise<RemoteResult<SwarmUiSnapshot>>;
}

declare module "@deepseek-ai/dsh-typert-protocol" {
  interface TypertRemoteMap {
    "swarm/uiSnapshot": SwarmRemoteNamespace["uiSnapshot"];
    "swarm/waitUi": SwarmRemoteNamespace["waitUi"];
  }

  interface TypertRemoteNamespaceMap {
    swarm: SwarmRemoteNamespace;
  }
}

export const TYPERT_REMOTE: TypertRemoteContribution = Object.freeze({
  package: "@swarmx/dsh-swarm",
  descriptors: SWARM_INVOCATIONS,
});

export default TYPERT_REMOTE;
