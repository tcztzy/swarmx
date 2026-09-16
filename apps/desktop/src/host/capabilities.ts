import type { AgentCapabilities } from "@swarmx/swarm";

/** Preserve the existing public capability response at HTTP, tool and ACP boundaries. */
export function publicCapabilities(capabilities: AgentCapabilities, permissions: boolean) {
  return {
    loadSession: capabilities.history,
    sessionCapabilities: {
      ...(capabilities.list ? { list: {} } : {}),
      ...(capabilities.resume ? { resume: {} } : {}),
    },
    _meta: {
      swarmx: {
        version: 2,
        permissions,
        steer: capabilities.steer,
        emptySessionResume: capabilities.emptySessionResume,
        activeRunResume: false,
        interactionResume: false,
      },
    },
  };
}
