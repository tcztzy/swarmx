import type { LocalTool } from "./local-tool-contracts.js";
import type { MessageChunk } from "./types.js";

export type BuiltinHarnessExecutionMode = "call" | "stream";

export interface BuiltinHarnessPromptContext {
  agentName: string;
  mode: BuiltinHarnessExecutionMode;
  arguments: Record<string, unknown>;
  runtimeContext: Record<string, unknown>;
}

export interface BuiltinHarnessTurn extends BuiltinHarnessPromptContext {
  instructions: string;
  tools: readonly LocalTool[];
  emitChunk?: (chunk: MessageChunk) => void;
}

export interface BuiltinHarnessResult {
  messages: MessageChunk[];
}

export interface BuiltinHarnessTurnUpdate {
  arguments?: Record<string, unknown>;
  runtimeContext?: Record<string, unknown>;
  instructions?: string;
  tools?: readonly LocalTool[];
  emitChunk?: (chunk: MessageChunk) => void;
}

export type BuiltinHarnessNext = (
  update?: BuiltinHarnessTurnUpdate,
) => Promise<BuiltinHarnessResult>;

export type BuiltinHarnessAgentLoop = (
  turn: Readonly<BuiltinHarnessTurn>,
  next: BuiltinHarnessNext,
) => Promise<BuiltinHarnessResult>;

export type BuiltinHarnessPromptSection = (
  context: Readonly<BuiltinHarnessPromptContext>,
) => string;

export interface BuiltinHarnessPluginContext {
  addPromptSection(id: string, render: BuiltinHarnessPromptSection): () => void;
  addTool(tool: LocalTool): () => void;
  useAgentLoop(id: string, middleware: BuiltinHarnessAgentLoop): () => void;
}

export interface BuiltinHarnessPlugin {
  id: string;
  setup(context: BuiltinHarnessPluginContext): undefined | (() => void);
}

interface ScopedRegistration<T> {
  id: string;
  value: T;
}

interface MountedPlugin {
  cleanup: () => void;
}

/**
 * Ordered, reversible composition kernel for the direct `swarmx` Harness.
 * External ACP Harnesses never enter this runtime.
 */
export class BuiltinHarness {
  private readonly plugins = new Map<string, MountedPlugin>();
  private readonly promptSections: ScopedRegistration<BuiltinHarnessPromptSection>[] = [];
  private readonly tools: ScopedRegistration<LocalTool>[] = [];
  private readonly agentLoops: ScopedRegistration<BuiltinHarnessAgentLoop>[] = [];

  mount(plugin: BuiltinHarnessPlugin): () => void {
    const pluginId = normalizedId(plugin.id, "plugin");
    if (this.plugins.has(pluginId)) {
      throw new Error(`Built-in Harness plugin "${pluginId}" is already mounted.`);
    }

    const unregister: Array<() => void> = [];
    const context: BuiltinHarnessPluginContext = {
      addPromptSection: (id, render) => {
        const remove = this.register(this.promptSections, id, render, "prompt section");
        unregister.push(remove);
        return remove;
      },
      addTool: (tool) => {
        const remove = this.register(this.tools, tool.name, tool, "tool");
        unregister.push(remove);
        return remove;
      },
      useAgentLoop: (id, middleware) => {
        const remove = this.register(this.agentLoops, id, middleware, "Agent-loop middleware");
        unregister.push(remove);
        return remove;
      },
    };

    let pluginCleanup: undefined | (() => void);
    try {
      pluginCleanup = plugin.setup(context);
    } catch (error) {
      for (const remove of unregister.reverse()) remove();
      throw error;
    }

    let mounted = true;
    const cleanup = () => {
      if (!mounted) return;
      mounted = false;
      let cleanupError: unknown;
      try {
        pluginCleanup?.();
      } catch (error) {
        cleanupError = error;
      } finally {
        for (const remove of unregister.reverse()) remove();
        this.plugins.delete(pluginId);
      }
      if (cleanupError) throw cleanupError;
    };
    this.plugins.set(pluginId, { cleanup });
    return cleanup;
  }

  unmount(pluginId: string): void {
    this.plugins.get(normalizedId(pluginId, "plugin"))?.cleanup();
  }

  close(): void {
    const errors: unknown[] = [];
    for (const plugin of [...this.plugins.values()].reverse()) {
      try {
        plugin.cleanup();
      } catch (error) {
        errors.push(error);
      }
    }
    if (errors.length > 0) {
      throw new AggregateError(errors, "One or more built-in Harness plugins failed to clean up.");
    }
  }

  async run(
    turn: BuiltinHarnessTurn,
    defaultLoop: (turn: Readonly<BuiltinHarnessTurn>) => Promise<BuiltinHarnessResult>,
  ): Promise<BuiltinHarnessResult> {
    const prepared = this.prepareTurn(turn);
    const middlewares = this.agentLoops.map((entry) => entry.value);

    const dispatch = async (
      index: number,
      current: Readonly<BuiltinHarnessTurn>,
    ): Promise<BuiltinHarnessResult> => {
      const middleware = middlewares[index];
      if (!middleware) return defaultLoop(current);

      let delegated = false;
      return middleware(current, async (update = {}) => {
        if (delegated) {
          throw new Error("Built-in Harness Agent-loop middleware called next() more than once.");
        }
        delegated = true;
        return dispatch(index + 1, freezeTurn({ ...current, ...update }));
      });
    };

    return dispatch(0, prepared);
  }

  private prepareTurn(turn: BuiltinHarnessTurn): Readonly<BuiltinHarnessTurn> {
    const context: BuiltinHarnessPromptContext = {
      agentName: turn.agentName,
      mode: turn.mode,
      arguments: turn.arguments,
      runtimeContext: turn.runtimeContext,
    };
    const sections = this.promptSections
      .map((entry) => entry.value(context).trim())
      .filter(Boolean);
    const instructions = [turn.instructions.trim(), ...sections]
      .filter(Boolean)
      .join("\n\n---\n\n");

    const tools = [...turn.tools];
    const names = new Set(tools.map((tool) => normalizedId(tool.name, "tool")));
    for (const entry of this.tools) {
      if (names.has(entry.id)) {
        throw new Error(`Built-in Harness tool "${entry.id}" conflicts with a base tool.`);
      }
      names.add(entry.id);
      tools.push(entry.value);
    }

    return freezeTurn({ ...turn, instructions, tools });
  }

  private register<T>(
    registrations: ScopedRegistration<T>[],
    id: string,
    value: T,
    kind: string,
  ): () => void {
    const normalized = normalizedId(id, kind);
    if (registrations.some((entry) => entry.id === normalized)) {
      throw new Error(`Built-in Harness ${kind} "${normalized}" is already registered.`);
    }
    const registration = { id: normalized, value };
    registrations.push(registration);
    let registered = true;
    return () => {
      if (!registered) return;
      registered = false;
      const index = registrations.indexOf(registration);
      if (index >= 0) registrations.splice(index, 1);
    };
  }
}

function freezeTurn(turn: BuiltinHarnessTurn): Readonly<BuiltinHarnessTurn> {
  const toolNames = new Set<string>();
  for (const tool of turn.tools) {
    const name = normalizedId(tool.name, "tool");
    if (toolNames.has(name)) {
      throw new Error(`Built-in Harness turn contains duplicate tool "${name}".`);
    }
    toolNames.add(name);
  }
  return Object.freeze({ ...turn, tools: Object.freeze([...turn.tools]) });
}

function normalizedId(value: string, kind: string): string {
  const normalized = value.trim();
  if (!normalized) throw new Error(`Built-in Harness ${kind} id must be non-empty.`);
  return normalized;
}
