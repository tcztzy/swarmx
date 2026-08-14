import { randomUUID } from "node:crypto";
import { Agent, type AgentRuntimeOptions } from "./agent.js";
import { Edge } from "./edge.js";
import {
  appendHookContext,
  dispatchHooks,
  Hook,
  type HookInvocation,
  type HookRuntimeOptions,
} from "./hook.js";
import { Tool } from "./tool.js";
import {
  type EvalRunResult,
  EvalRunResultSchema,
  type EvalTraceEvent,
  type McpServerConfig,
  type MessageChunk,
  type ModelTokenUsage,
  type SwarmConfig,
  SwarmConfigSchema,
  type SwarmNodeConfig,
  SwarmNodeConfigSchema,
} from "./types.js";

const MAX_STEPS = 100;

export interface SwarmRuntimeOptions {
  agent?: AgentRuntimeOptions;
  hook?: HookRuntimeOptions;
}

interface EvalTraceCollector {
  runId: string;
  events: EvalTraceEvent[];
  nextStep: number;
}

export class SwarmNode {
  kind: "agent" | "tool" | "swarm";
  agent?: Agent;
  tool?: Tool;
  swarm?: Swarm;

  constructor(config: SwarmNodeConfig, options: SwarmRuntimeOptions = {}) {
    const parsed = SwarmNodeConfigSchema.parse(config);
    this.kind = parsed.kind;
    if (parsed.kind === "agent") {
      this.agent = new Agent(parsed.agent, {
        ...options.agent,
        hook: options.agent?.hook ?? options.hook,
      });
    } else if (parsed.kind === "tool") {
      this.tool = new Tool(parsed.tool);
    } else {
      this.swarm = new Swarm(parsed.swarm as SwarmConfig, options);
    }
  }

  get name(): string {
    if (this.agent) return this.agent.name;
    if (this.tool) return this.tool.name;
    if (this.swarm) return this.swarm.name;
    throw new Error(`Invalid swarm node kind "${this.kind}"`);
  }
}

export class Swarm {
  name: string;
  description?: string;
  parameters: Record<string, unknown>;
  returns?: Record<string, unknown>;
  mcpServers: Map<string, McpServerConfig>;
  queen?: Agent;
  nodes: Map<string, SwarmNode>;
  edges: Edge[];
  root: string;
  hooks: Hook[];
  private hookRuntime?: HookRuntimeOptions;

  constructor(config: SwarmConfig, options: SwarmRuntimeOptions = {}) {
    const parsed = SwarmConfigSchema.parse(config) as SwarmConfig;
    this.name = parsed.name;
    this.description = parsed.description;
    this.parameters = parsed.parameters ?? {};
    this.returns = parsed.returns;
    this.mcpServers = new Map(parsed.mcpServers ? Object.entries(parsed.mcpServers) : []);
    this.queen = parsed.queen ? new Agent(parsed.queen, options.agent) : undefined;
    this.nodes = new Map(
      Object.entries(parsed.nodes).map(([k, v]) => [k, new SwarmNode(v, options)]),
    );
    this.edges = (parsed.edges ?? []).map((e) => new Edge(e));
    this.root = parsed.root;
    this.hooks = (parsed.hooks ?? []).map((h) => new Hook(h));
    this.hookRuntime = options.hook;

    this.validateDag();
  }

  private detectCycle(edges: Array<{ source: string; target: string }>): string[] | null {
    const adj = new Map<string, string[]>();
    for (const e of edges) {
      const targets = adj.get(e.source) ?? [];
      targets.push(e.target);
      adj.set(e.source, targets);
    }

    const WHITE = 1;
    const GRAY = 2;
    const BLACK = 3;
    const color = new Map<string, number>();
    const path: string[] = [];

    function dfs(node: string): string[] | null {
      color.set(node, GRAY);
      path.push(node);

      for (const next of adj.get(node) ?? []) {
        const c = color.get(next);
        if (c === GRAY) {
          const idx = path.indexOf(next);
          return [...path.slice(idx), next];
        }
        if (c === undefined || c === WHITE) {
          const result = dfs(next);
          if (result) return result;
        }
      }

      path.pop();
      color.set(node, BLACK);
      return null;
    }

    for (const node of adj.keys()) {
      if (!color.has(node) || color.get(node) === WHITE) {
        const cycle = dfs(node);
        if (cycle) return cycle;
      }
    }

    return null;
  }

  public validateDag(): void {
    const unconditional = this.edges
      .filter((e) => !e.condition)
      .map((e) => ({ source: e.source, target: e.target }));
    const cycle = this.detectCycle(unconditional);
    if (cycle) {
      throw new Error(`Unconditional cycle detected in swarm "${this.name}": ${cycle.join(" → ")}`);
    }

    const allEdges = this.edges.map((e) => ({
      source: e.source,
      target: e.target,
    }));
    const condCycle = this.detectCycle(allEdges);
    if (condCycle) {
      console.warn(
        `Warning: Conditional cycle detected in swarm "${this.name}": ${condCycle.join(" → ")}. Ensure at least one edge condition can break the loop.`,
      );
    }
  }

  /**
   * Execute the swarm DAG starting from root.
   * Uses topological traversal respecting CEL edge conditions.
   */
  async execute(
    arguments_: Record<string, unknown>,
    context?: Record<string, unknown>,
    onChunk?: (chunk: MessageChunk) => void,
    onUsage?: (usage: ModelTokenUsage) => void,
  ): Promise<MessageChunk[]> {
    return this.executeInternal(arguments_, context, undefined, onChunk, onUsage);
  }

  async executeForEval(
    arguments_: Record<string, unknown>,
    context?: Record<string, unknown>,
    onUsage?: (usage: ModelTokenUsage) => void,
  ): Promise<EvalRunResult> {
    const trace: EvalTraceCollector = {
      runId: randomUUID(),
      events: [],
      nextStep: 1,
    };
    let messages: MessageChunk[] = [];
    let error: string | null = null;
    let contextTokens = 0;

    try {
      messages = await this.executeInternal(arguments_, context, trace, undefined, (usage) => {
        contextTokens += usage.totalTokens;
        onUsage?.(usage);
      });
    } catch (err) {
      error = errorMessage(err);
    }

    return EvalRunResultSchema.parse({
      output: messagesToEvalOutput(messages),
      messages,
      trace: [...trace.events].sort((a, b) => a.step - b.step),
      error,
      metrics: { ...buildEvalMetrics(messages, trace.events), contextTokens },
    });
  }

  private async executeInternal(
    arguments_: Record<string, unknown>,
    context?: Record<string, unknown>,
    trace?: EvalTraceCollector,
    onChunk?: (chunk: MessageChunk) => void,
    onUsage?: (usage: ModelTokenUsage) => void,
  ): Promise<MessageChunk[]> {
    const ctx = { ...(context ?? {}) };
    let effectiveArguments = arguments_;
    let chunkHooks = Promise.resolve();
    let newMessages: MessageChunk[];

    try {
      if (!this.nodes.has(this.root)) {
        throw new Error(`Root node "${this.root}" not found in swarm "${this.name}"`);
      }
      const start = await dispatchHooks(
        this.hooks,
        "onStart",
        this.hookInvocation(effectiveArguments, ctx),
        this.hookRuntime,
      );
      effectiveArguments = appendHookContext(effectiveArguments, start.additionalContext);
      const emitChunk = onChunk
        ? (chunk: MessageChunk): void => {
            onChunk(chunk);
            chunkHooks = chunkHooks.then(async () => {
              await dispatchHooks(
                this.hooks,
                "onChunk",
                this.hookInvocation(effectiveArguments, ctx, { chunk }),
                this.hookRuntime,
              );
            });
          }
        : undefined;
      newMessages = await this.executeDag(
        effectiveArguments,
        ctx,
        trace,
        emitChunk,
        onUsage,
        async (node, source, target) => {
          const [agentContext, swarmContext] = await Promise.all([
            node.agent?.runHandoffHooks(effectiveArguments, ctx, { source, target }) ?? [],
            dispatchHooks(
              this.hooks,
              "onHandoff",
              this.hookInvocation(effectiveArguments, ctx, {
                handoff: { source, target },
              }),
              this.hookRuntime,
            ).then((result) => result.additionalContext),
          ]);
          effectiveArguments = appendHookContext(effectiveArguments, [
            ...agentContext,
            ...swarmContext,
          ]);
          return effectiveArguments;
        },
      );
      await chunkHooks;
    } catch (error) {
      await this.runFailedEndHook(effectiveArguments, ctx, error);
      throw error;
    }

    await dispatchHooks(
      this.hooks,
      "onEnd",
      this.hookInvocation(effectiveArguments, ctx, {
        outcome: { status: "completed", messages: newMessages },
      }),
      this.hookRuntime,
    );
    return newMessages;
  }

  private async executeDag(
    initialArguments: Record<string, unknown>,
    context: Record<string, unknown>,
    trace: EvalTraceCollector | undefined,
    onChunk: ((chunk: MessageChunk) => void) | undefined,
    onUsage: ((usage: ModelTokenUsage) => void) | undefined,
    onHandoff: (
      node: SwarmNode,
      source: string,
      target: string,
    ) => Promise<Record<string, unknown>>,
  ): Promise<MessageChunk[]> {
    let effectiveArguments = initialArguments;
    const newMessages: MessageChunk[] = [];
    const { predecessors } = this.rebuildGraphs();
    const visited = new Set<string>();
    const scheduled = new Set<string>();
    const queue: string[] = [this.root];
    scheduled.add(this.root);

    let steps = 0;
    while (queue.length > 0 && steps < MAX_STEPS) {
      const nodeName = queue.pop();
      if (!nodeName) break;
      const node = this.nodes.get(nodeName);
      if (!node) throw new Error(`Node "${nodeName}" not found`);

      const nodeMessages = await this.runNode(
        nodeName,
        node,
        effectiveArguments,
        context,
        trace,
        onChunk,
        onUsage,
      );
      visited.add(nodeName);
      if (nodeMessages.length > 0) newMessages.push(...nodeMessages);

      for (const edge of this.edges) {
        if (edge.source !== nodeName || !edge.evaluate(context)) continue;
        for (const target of edge.resolveTargets(context)) {
          if (!this.nodes.has(target)) {
            throw new Error(`Unknown target "${target}" in swarm "${this.name}"`);
          }
          if (visited.has(target) || scheduled.has(target)) continue;
          const required = predecessors.get(target) ?? new Set();
          if (!isSubset(required, visited)) continue;

          effectiveArguments = await onHandoff(node, nodeName, target);
          queue.push(target);
          scheduled.add(target);
        }
      }
      steps++;
    }
    if (queue.length > 0) {
      throw new Error(
        `Swarm "${this.name}" did not settle within ${MAX_STEPS} workflow steps; ${queue.length} scheduled node(s) remain.`,
      );
    }
    return newMessages;
  }

  private async runFailedEndHook(
    arguments_: Record<string, unknown>,
    context: Record<string, unknown>,
    error: unknown,
  ): Promise<void> {
    try {
      await dispatchHooks(
        this.hooks,
        "onEnd",
        this.hookInvocation(arguments_, context, {
          outcome: { status: "failed", error: errorMessage(error) },
        }),
        this.hookRuntime,
      );
    } catch (endError) {
      throw new AggregateError(
        [error, endError],
        `Swarm "${this.name}" failed and its onEnd hook also failed.`,
      );
    }
  }

  private hookInvocation(
    arguments_: Record<string, unknown>,
    context: Record<string, unknown>,
    extra: Pick<HookInvocation, "chunk" | "handoff" | "outcome"> = {},
  ): Omit<HookInvocation, "event"> {
    return {
      scope: "swarm",
      target: { name: this.name },
      arguments: arguments_,
      context,
      ...extra,
    };
  }

  private async runNode(
    nodeName: string,
    node: SwarmNode,
    arguments_: Record<string, unknown>,
    context: Record<string, unknown>,
    trace?: EvalTraceCollector,
    onChunk?: (chunk: MessageChunk) => void,
    onUsage?: (usage: ModelTokenUsage) => void,
  ): Promise<MessageChunk[]> {
    const startedAt = new Date().toISOString();
    const step = trace?.nextStep ?? 0;
    if (trace) {
      trace.nextStep++;
    }

    try {
      const messages = await this.runNodeUnchecked(
        node,
        arguments_,
        context,
        trace,
        onChunk,
        onUsage,
      );
      if (trace) {
        trace.events.push({
          runId: trace.runId,
          swarm: this.name,
          node: nodeName,
          kind: node.kind,
          step,
          startedAt,
          endedAt: new Date().toISOString(),
          status: "completed",
          messageCount: messages.length,
        });
      }
      return messages;
    } catch (err) {
      if (trace) {
        trace.events.push({
          runId: trace.runId,
          swarm: this.name,
          node: nodeName,
          kind: node.kind,
          step,
          startedAt,
          endedAt: new Date().toISOString(),
          status: "failed",
          messageCount: 0,
          error: errorMessage(err),
        });
      }
      throw err;
    }
  }

  private async runNodeUnchecked(
    node: SwarmNode,
    arguments_: Record<string, unknown>,
    context: Record<string, unknown>,
    trace?: EvalTraceCollector,
    onChunk?: (chunk: MessageChunk) => void,
    onUsage?: (usage: ModelTokenUsage) => void,
  ): Promise<MessageChunk[]> {
    switch (node.kind) {
      case "agent": {
        if (!node.agent) return [];
        const result = onChunk
          ? await node.agent.callStream(arguments_, onChunk, onUsage, context)
          : await node.agent.call(arguments_, context, onUsage);
        const messages = result.messages as MessageChunk[] | undefined;
        return messages ?? [];
      }
      case "tool": {
        if (!node.tool) return [];
        const result = await node.tool.call(arguments_, context);
        return [
          {
            role: "tool",
            content: JSON.stringify(result),
            kind: "message",
          },
        ];
      }
      case "swarm": {
        if (!node.swarm) return [];
        return node.swarm.executeInternal(arguments_, context, trace, onChunk, onUsage);
      }
    }
  }

  rebuildGraphs(): {
    predecessors: Map<string, Set<string>>;
  } {
    const predecessors = new Map<string, Set<string>>();

    for (const name of this.nodes.keys()) {
      predecessors.set(name, new Set());
    }

    for (const edge of this.edges) {
      if (!edge.condition && this.nodes.has(edge.target)) {
        const preds = predecessors.get(edge.target);
        if (!preds) continue;
        preds.add(edge.source);
      }
    }

    return { predecessors };
  }

  async listAllSessions(cwd?: string): Promise<Array<{ agent: string; sessions: SessionInfo[] }>> {
    const results: Array<{ agent: string; sessions: SessionInfo[] }> = [];

    for (const [name, node] of this.nodes) {
      if (node.kind === "agent" && node.agent) {
        try {
          const sessions = await node.agent.listSessions(cwd);
          results.push({ agent: name, sessions });
        } catch (e) {
          console.warn(`Failed to list sessions for agent ${name}: ${e}`);
        }
      }
    }

    if (this.queen) {
      try {
        const sessions = await this.queen.listSessions(cwd);
        results.push({ agent: this.queen.name, sessions });
      } catch (e) {
        console.warn(`Failed to list sessions for queen agent: ${e}`);
      }
    }

    return results;
  }
}

function isSubset(set: Set<string>, superset: Set<string>): boolean {
  for (const item of set) {
    if (!superset.has(item)) return false;
  }
  return true;
}

function messagesToEvalOutput(messages: MessageChunk[]): string {
  const assistantMessages = messages.filter(
    (message) => message.kind === "message" && message.role === "assistant",
  );
  const source =
    assistantMessages.length > 0
      ? assistantMessages
      : messages.filter((message) => message.kind === "message");
  return source
    .map((message) => message.content)
    .filter(Boolean)
    .join("\n");
}

function buildEvalMetrics(
  messages: MessageChunk[],
  trace: EvalTraceEvent[],
): EvalRunResult["metrics"] {
  return {
    steps: trace.length,
    messages: messages.length,
    toolCalls: messages.filter((message) => message.kind === "tool_call").length,
    toolResults: messages.filter((message) => message.kind === "tool_result").length,
  };
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

interface SessionInfo {
  sessionId?: string;
  session_id?: string;
  cwd?: string;
  title?: string;
  updatedAt?: string;
  updated_at?: string;
}
