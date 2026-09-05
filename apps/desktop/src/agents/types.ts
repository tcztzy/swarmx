import type { Agent } from "@swarmx/swarm";

export interface Interaction {
  readonly id: string;
  readonly title: string;
  readonly schema: Record<string, unknown>;
}

/** Transient UI callbacks, not a transport or a persisted transcript. */
export interface Observer {
  text(id: string, text: string, role?: "user" | "assistant" | "reasoning"): void;
  tool(id: string, name: string, input: unknown, output?: unknown): void;
  raw(event: unknown): void;
  interact(request: Interaction): Promise<unknown>;
}

export type NativeAgent = Agent<Observer>;

export interface AgentOptions {
  readonly cwd: string;
  readonly mcp: { readonly url: string; readonly headers: Record<string, string> };
}
