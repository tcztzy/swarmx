import { AsyncLocalStorage } from "node:async_hooks";
import { type AgentId, loadAgent } from "../agent.js";
import type { AgentOptions, NativeAgent } from "../agents/types.js";

const bindings = new AsyncLocalStorage<AgentOptions>();

/** The execution binding of the Agent call currently executing. */
export function currentAgentBinding(): AgentOptions {
  const binding = bindings.getStore();
  if (!binding) throw new Error("Agent used outside an execution binding.");
  return binding;
}

/** Binds one shared native Agent to an execution. Every call resolves that execution's cwd, MCP and policy. */
export function bindAgent(agent: NativeAgent, binding: AgentOptions): NativeAgent {
  const scoped = <T>(task: () => T): T => bindings.run(binding, task);
  const bound: NativeAgent = {
    name: agent.name,
    capabilities: agent.capabilities,
    models: (sessionId) =>
      scoped(() => (sessionId === undefined ? agent.models() : agent.models(sessionId))),
    list: () => scoped(() => agent.list()),
    create: (options) =>
      scoped(() => (options === undefined ? agent.create() : agent.create(options))),
    read: (sessionId, observer) => scoped(() => agent.read(sessionId, observer)),
    start: (sessionId, text, observer, options) =>
      scoped(() =>
        options === undefined
          ? agent.start(sessionId, text, observer)
          : agent.start(sessionId, text, observer, options),
      ),
    steer: (sessionId, text, expectedRunId) =>
      scoped(() =>
        expectedRunId === undefined
          ? agent.steer(sessionId, text)
          : agent.steer(sessionId, text, expectedRunId),
      ),
    interrupt: (sessionId, expectedRunId) =>
      scoped(() =>
        expectedRunId === undefined
          ? agent.interrupt(sessionId)
          : agent.interrupt(sessionId, expectedRunId),
      ),
    // The host-level registry owns the shared runtime; an execution never disposes it.
    dispose: async () => {},
  };
  const permissions = agent.permissions;
  if (permissions) bound.permissions = (sessionId) => scoped(() => permissions(sessionId));
  const restoreEmptySessions = agent.restoreEmptySessions;
  if (restoreEmptySessions)
    bound.restoreEmptySessions = (ids) => scoped(() => restoreEmptySessions(ids));
  return bound;
}

export type AgentLoader = (id: AgentId, options: AgentOptions) => Promise<NativeAgent>;

/**
 * Host-level ownership of native Agent runtimes: one lazy instance per harness, shared by every
 * execution. Execution directories, policies and MCP endpoints enter through per-call bindings.
 */
export class AgentRegistry {
  private readonly loaded = new Map<AgentId, NativeAgent>();
  private readonly loading = new Map<AgentId, Promise<NativeAgent>>();
  private closed = false;

  constructor(private readonly load: AgentLoader = (id, options) => loadAgent(id, options)) {}

  agent(id: AgentId, binding: AgentOptions): Promise<NativeAgent> {
    if (this.closed) throw new Error("The Agent registry is closed.");
    const existing = this.loaded.get(id);
    if (existing) return Promise.resolve(existing);
    const pending = this.loading.get(id);
    if (pending) return pending;
    const options = new Proxy({} as AgentOptions, {
      get: (_target, key) => {
        if (key === "reviewOnly") return undefined;
        return (currentAgentBinding() as unknown as Record<PropertyKey, unknown>)[key];
      },
    });
    const operation = Promise.resolve(bindings.run(binding, () => this.load(id, options)))
      .then((agent) => {
        this.loaded.set(id, agent);
        return agent;
      })
      .finally(() => {
        this.loading.delete(id);
      });
    this.loading.set(id, operation);
    return operation;
  }

  async dispose(): Promise<void> {
    this.closed = true;
    const agents = [...this.loaded.values()];
    this.loaded.clear();
    this.loading.clear();
    await Promise.allSettled(agents.map((agent) => agent.dispose()));
  }
}
