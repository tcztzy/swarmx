import type { AgentOptions, NativeAgent, Observer } from "./agents/types.js";
import { HarnessSchema } from "./permissions.js";

export const AGENT_IDS = HarnessSchema.options;
export type AgentId = (typeof AGENT_IDS)[number];

export function selectedAgent(value = process.env.SWARMX_AGENT ?? "codex"): AgentId {
  if (!AGENT_IDS.includes(value as AgentId)) throw new Error(`Unknown Agent "${value}".`);
  return value as AgentId;
}

export async function loadAgent(id: AgentId, options: AgentOptions): Promise<NativeAgent> {
  const { createAcpHarness } = await import("./agents/acp-harness.js");
  const agent = scopeSessions(id, await createAcpHarness(id, options));
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
  const restore = agent.restoreEmptySessions?.bind(agent);
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
    capabilities: agent.capabilities,
    ...(restore
      ? {
          restoreEmptySessions(ids: readonly string[]) {
            restore(
              ids.map((session) => {
                if (!session.startsWith(`${id}:`))
                  throw new Error("Invalid restored session owner.");
                remember(session.slice(id.length + 1));
                return nativeId(session);
              }),
            );
          },
        }
      : {}),
    models: (session) => agent.models(session === undefined ? undefined : nativeId(session)),
    list: async () =>
      (await agent.list()).map((session) => ({
        ...session,
        sessionId: remember(session.sessionId),
      })),
    create: async (options) => remember(await agent.create(options)),
    read: (session: string, observer: Observer) => agent.read(nativeId(session), observer),
    start: (session, text, observer, options) =>
      agent.start(nativeId(session), text, observer, options),
    steer: (session, text) => agent.steer(nativeId(session), text),
    interrupt: (session) => agent.interrupt(nativeId(session)),
    dispose: () => agent.dispose(),
  };
}
