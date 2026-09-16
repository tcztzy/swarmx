export interface Session {
  readonly sessionId: string;
  readonly title?: string;
  readonly updatedAt?: string;
}

export interface RunOptions {
  readonly model?: string | undefined;
  readonly effort?: string | undefined;
  readonly mode?: string | undefined;
}

export interface ModelCatalog {
  readonly modes?:
    | { readonly id: string; readonly name: string; readonly description?: string | undefined }[]
    | undefined;
  readonly models: {
    readonly id: string;
    readonly name: string;
    readonly description?: string | undefined;
    readonly efforts: { readonly id: string; readonly name: string }[];
    readonly defaultEffort?: string | undefined;
  }[];
  readonly current: RunOptions;
}

export interface AgentCapabilities {
  readonly history: boolean;
  readonly list: boolean;
  readonly resume: boolean;
  readonly steer: boolean;
  readonly emptySessionResume: boolean;
}

/** Stable public outcome values also used by persisted execution records. */
export interface RunResult {
  readonly stopReason: "end_turn" | "cancelled" | "max_tokens" | "max_turn_requests" | "refusal";
}

/** Direct native execution boundary, shared by callers and recursive composition. */
export interface Agent<Observer> {
  readonly name: string;
  readonly capabilities: AgentCapabilities;
  models(sessionId?: string): Promise<ModelCatalog>;
  list(): Promise<Session[]>;
  create(): Promise<string>;
  read(sessionId: string, observer: Observer): Promise<void>;
  start(
    sessionId: string,
    text: string,
    observer: Observer,
    options?: RunOptions,
  ): Promise<RunResult>;
  steer(sessionId: string, text: string): Promise<void>;
  interrupt(sessionId: string): Promise<void>;
  dispose(): Promise<void>;
}

/** A Swarm borrows its lead; the owner remains responsible for disposing the native runtime. */
export function createSwarm<T extends Agent<never>>(name: string, lead: T): T {
  return {
    ...lead,
    name,
    models: lead.models.bind(lead),
    list: lead.list.bind(lead),
    create: lead.create.bind(lead),
    read: lead.read.bind(lead),
    start: lead.start.bind(lead),
    steer: lead.steer.bind(lead),
    interrupt: lead.interrupt.bind(lead),
    dispose: async () => {},
  };
}
