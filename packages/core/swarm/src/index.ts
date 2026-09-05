export interface Session {
  readonly sessionId: string;
  readonly title?: string;
  readonly updatedAt?: string;
}

/** In-process composition only; native events and interactions belong to the caller. */
export interface Agent<Observer> {
  readonly name: string;
  list(): Promise<Session[]>;
  create(): Promise<string>;
  read(sessionId: string, observer: Observer): Promise<void>;
  start(sessionId: string, text: string, observer: Observer): Promise<void>;
  steer(sessionId: string, text: string): Promise<void>;
  interrupt(sessionId: string): Promise<void>;
  dispose(): Promise<void>;
}

export function createSwarm<O>(name: string, lead: Agent<O>): Agent<O> {
  return {
    name,
    list: () => lead.list(),
    create: () => lead.create(),
    read: (id, observer) => lead.read(id, observer),
    start: (id, text, observer) => lead.start(id, text, observer),
    steer: (id, text) => lead.steer(id, text),
    interrupt: (id) => lead.interrupt(id),
    dispose: () => lead.dispose(),
  };
}
