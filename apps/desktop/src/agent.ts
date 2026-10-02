import type { AgentOptions, NativeAgent, Observer } from "./agents/types.js";
import { HarnessSchema } from "./permissions.js";

// Keep retired ids in persisted schemas, but never advertise or load them as runtimes.
export const AGENT_IDS = HarnessSchema.options.filter((id) => id !== "pi");
export type AgentId = (typeof AGENT_IDS)[number];

export function selectedAgent(
  value = process.env.SWARMX_AGENT ?? (process.env.SWARMX_ACP_AGENT ? "acp" : "codex"),
): AgentId {
  if (value === "pi")
    throw new Error(
      "The built-in Pi agent has been retired. Select Codex or configure an external ACP agent with SWARMX_AGENT=acp and SWARMX_ACP_AGENT. Existing Pi authentication and session files are unchanged.",
    );
  if (!AGENT_IDS.includes(value as AgentId)) throw new Error(`Unknown Agent "${value}".`);
  return value as AgentId;
}

export async function loadAgent(id: AgentId, options: AgentOptions): Promise<NativeAgent> {
  id = selectedAgent(id);
  const native =
    id === "codex"
      ? await (await import("./agents/codex.js")).createCodex(options)
      : id === "claude"
        ? await (await import("./agents/claude.js")).createClaude(options)
        : id === "hermes"
          ? await (await import("./agents/hermes.js")).createHermes(options)
          : id === "openclaw"
            ? await (await import("./agents/openclaw.js")).createOpenClaw(options)
            : id === "acp"
              ? await (await import("./agents/external-acp.js")).createExternalAcp(options)
              : await (await import("./agents/dsh.js")).createDsh(options);
  const agent = scopeSessions(id, native);
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
