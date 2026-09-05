import type { AgentOptions, NativeAgent, Observer } from "./agents/types.js";

export const AGENT_IDS = ["codex", "claude", "hermes", "openclaw"] as const;
export type AgentId = (typeof AGENT_IDS)[number];

export function selectedAgent(value = process.env.SWARMX_AGENT ?? "codex"): AgentId {
  if (!AGENT_IDS.includes(value as AgentId)) throw new Error(`Unknown Agent "${value}".`);
  return value as AgentId;
}

const loaders = {
  codex: async (options: AgentOptions) => (await import("./agents/codex.js")).createCodex(options),
  claude: async (options: AgentOptions) =>
    (await import("./agents/claude.js")).createClaude(options),
  hermes: async (options: AgentOptions) =>
    (await import("./agents/hermes.js")).createHermes(options),
  openclaw: async (_options: AgentOptions) =>
    (await import("./agents/openclaw.js")).createOpenClaw(),
};

export async function loadAgent(id: AgentId, options: AgentOptions): Promise<NativeAgent> {
  const agent = scopeSessions(id, await loaders[id](options));
  try {
    await agent.list();
    return agent;
  } catch (error) {
    await agent.dispose();
    throw error;
  }
}

/** Browser/external session ids cannot be reused against another native Agent. */
export function scopeSessions(id: string, agent: NativeAgent): NativeAgent {
  const known = new Set<string>();
  const remember = (native: string) => {
    const scoped = `${id}:${native}`;
    known.add(scoped);
    return scoped;
  };
  const nativeId = (session: string) => {
    if (!known.has(session))
      throw new Error(`Session "${session}" does not belong to Agent "${id}".`);
    return session.slice(id.length + 1);
  };
  return {
    name: agent.name,
    list: async () =>
      (await agent.list()).map((session) => ({
        ...session,
        sessionId: remember(session.sessionId),
      })),
    create: async () => remember(await agent.create()),
    read: (session: string, observer: Observer) => agent.read(nativeId(session), observer),
    start: (session, text, observer) => agent.start(nativeId(session), text, observer),
    steer: (session, text) => agent.steer(nativeId(session), text),
    interrupt: (session) => agent.interrupt(nativeId(session)),
    dispose: () => agent.dispose(),
  };
}
