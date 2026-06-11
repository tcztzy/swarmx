import { Agent } from "./agent.js";
import { Edge } from "./edge.js";
import { Hook } from "./hook.js";
import { Tool } from "./tool.js";
import {
  type McpServerConfig,
  type MessageChunk,
  type SwarmConfig,
  SwarmConfigSchema,
  type SwarmNodeConfig,
  SwarmNodeConfigSchema,
} from "./types.js";

const MAX_STEPS = 100;

export class SwarmNode {
  kind: "agent" | "tool" | "swarm";
  agent?: Agent;
  tool?: Tool;
  swarm?: Swarm;

  constructor(config: SwarmNodeConfig) {
    const parsed = SwarmNodeConfigSchema.parse(config);
    this.kind = parsed.kind;
    if (parsed.kind === "agent") {
      this.agent = new Agent(parsed.agent);
    } else if (parsed.kind === "tool") {
      this.tool = new Tool(parsed.tool);
    } else {
      this.swarm = new Swarm(parsed.swarm);
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

  constructor(config: SwarmConfig) {
    const parsed = SwarmConfigSchema.parse(config);
    this.name = parsed.name;
    this.description = parsed.description;
    this.parameters = parsed.parameters ?? {};
    this.returns = parsed.returns;
    this.mcpServers = new Map(parsed.mcpServers ? Object.entries(parsed.mcpServers) : []);
    this.queen = parsed.queen ? new Agent(parsed.queen) : undefined;
    this.nodes = new Map(Object.entries(parsed.nodes).map(([k, v]) => [k, new SwarmNode(v)]));
    this.edges = (parsed.edges ?? []).map((e) => new Edge(e));
    this.root = parsed.root;
    this.hooks = (parsed.hooks ?? []).map((h) => new Hook(h));

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
  ): Promise<MessageChunk[]> {
    const ctx = { ...(context ?? {}) };
    const newMessages: MessageChunk[] = [];

    if (!this.nodes.has(this.root)) {
      throw new Error(`Root node "${this.root}" not found in swarm "${this.name}"`);
    }

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

      const nodeMessages = await this.runNode(node, arguments_, ctx);
      visited.add(nodeName);

      if (nodeMessages.length > 0) {
        newMessages.push(...nodeMessages);
      }

      for (const edge of this.edges) {
        if (edge.source !== nodeName) continue;
        if (!edge.evaluate(ctx)) continue;

        const targets = edge.resolveTargets(ctx);
        for (const target of targets) {
          if (!this.nodes.has(target)) {
            throw new Error(`Unknown target "${target}" in swarm "${this.name}"`);
          }
          if (visited.has(target) || scheduled.has(target)) continue;

          const required = predecessors.get(target) ?? new Set();
          if (!isSubset(required, visited)) continue;

          queue.push(target);
          scheduled.add(target);
        }
      }

      steps++;
    }

    return newMessages;
  }

  private async runNode(
    node: SwarmNode,
    arguments_: Record<string, unknown>,
    context: Record<string, unknown>,
  ): Promise<MessageChunk[]> {
    switch (node.kind) {
      case "agent": {
        if (!node.agent) return [];
        const result = await node.agent.call(arguments_, context);
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
        return node.swarm.execute(arguments_, context);
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

interface SessionInfo {
  sessionId?: string;
  session_id?: string;
  cwd?: string;
  title?: string;
  updatedAt?: string;
  updated_at?: string;
}
