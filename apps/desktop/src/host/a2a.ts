import { A2A_PROTOCOL_VERSION, AgentCard, type Artifact, type Task, TaskState } from "@a2a-js/sdk";
import {
  AgentEvent,
  type AgentExecutor,
  DefaultRequestHandler,
  type ExecutionEventBus,
  InMemoryTaskStore,
  JsonRpcTransportHandler,
  type RequestContext,
  ServerCallContext,
  validateVersion,
} from "@a2a-js/sdk/server";
import type { NativeAgent } from "../agents/types.js";

interface Endpoint {
  readonly card: AgentCard;
  readonly handler: JsonRpcTransportHandler;
}

/** Optional A2A 1.0 endpoints. Swarm internals never call this table. */
export class A2AEndpoints {
  private readonly endpoints = new Map<string, Endpoint>();
  private origin?: string;

  attach(origin: string): void {
    this.origin = origin;
  }

  add(id: string, name: string, executor: AgentExecutor): void {
    const origin = this.origin;
    if (origin === undefined) throw new Error("A2A endpoints are not attached.");
    if (this.endpoints.has(id)) throw new Error(`A2A Agent "${id}" already exists.`);
    const card = agentCard(`${origin}/a2a/${encodeURIComponent(id)}`, name);
    this.endpoints.set(id, {
      card,
      handler: new JsonRpcTransportHandler(
        new DefaultRequestHandler(card, new InMemoryTaskStore(), executor),
      ),
    });
  }

  card(id: string): unknown {
    return AgentCard.toJSON(this.get(id).card);
  }

  async handle(id: string, body: Record<string, unknown>, version: string): Promise<unknown> {
    const endpoint = this.get(id);
    validateVersion(version, endpoint.card, "JSONRPC");
    const result = await endpoint.handler.handle(
      body,
      new ServerCallContext({ requestedVersion: version }),
    );
    if (Symbol.asyncIterator in result) throw new Error("Streaming A2A ingress is not enabled.");
    return result;
  }

  private get(id: string): Endpoint {
    const endpoint = this.endpoints.get(id);
    if (endpoint === undefined) throw new Error(`A2A Agent "${id}" was not found.`);
    return endpoint;
  }
}

/** Adapts the optional external A2A task lifecycle to one native Swarm. */
export class SwarmA2AExecutor implements AgentExecutor {
  private readonly contexts = new Map<string, string>();
  private readonly active = new Map<string, { sessionId: string; cancelled: boolean }>();

  constructor(private readonly agent: NativeAgent) {}

  async execute(context: RequestContext, events: ExecutionEventBus): Promise<void> {
    try {
      const sessionId = this.contexts.get(context.contextId) ?? (await this.agent.create());
      this.contexts.set(context.contextId, sessionId);
      const active = { sessionId, cancelled: false };
      this.active.set(context.taskId, active);
      events.publish(AgentEvent.task(task(context, TaskState.TASK_STATE_WORKING)));
      const text: string[] = [];
      await this.agent.start(sessionId, input(context), {
        text: (_id, value, role = "assistant") => {
          if (role === "assistant") text.push(value);
        },
        tool() {},
        raw() {},
        interact: async () => {
          throw new Error(
            "This A2A entry point cannot answer interactive native requests. Use ACP or the browser.",
          );
        },
      });
      const state = active.cancelled
        ? TaskState.TASK_STATE_CANCELED
        : TaskState.TASK_STATE_COMPLETED;
      if (text.length > 0)
        events.publish(AgentEvent.artifactUpdate(artifact(context, text.join(""))));
      events.publish(AgentEvent.statusUpdate(statusUpdate(context, state)));
    } catch (error) {
      events.publish(
        AgentEvent.statusUpdate({
          ...statusUpdate(context, TaskState.TASK_STATE_FAILED),
          metadata: { error: error instanceof Error ? error.message : String(error) },
        }),
      );
    } finally {
      this.active.delete(context.taskId);
      events.finished();
    }
  }

  async cancelTask(taskId: string, _events: ExecutionEventBus): Promise<void> {
    const active = this.active.get(taskId);
    if (active === undefined) throw new Error(`A2A task "${taskId}" is not running.`);
    active.cancelled = true;
    await this.agent.interrupt(active.sessionId);
  }
}

function input(context: RequestContext): string {
  if (context.userMessage.parts.some((part) => part.content?.$case !== "text"))
    throw new Error("SwarmX accepts text-only A2A messages.");
  const text = context.userMessage.parts
    .flatMap((part) => (part.content?.$case === "text" ? [part.content.value] : []))
    .join("");
  if (text.trim() === "") throw new Error("A2A input must contain text.");
  return text;
}

function task(context: RequestContext, state: TaskState): Task {
  return {
    id: context.taskId,
    contextId: context.contextId,
    status: { state, message: undefined, timestamp: new Date().toISOString() },
    artifacts: [],
    history: [],
    metadata: undefined,
  };
}

function statusUpdate(context: RequestContext, state: TaskState) {
  return {
    taskId: context.taskId,
    contextId: context.contextId,
    status: { state, message: undefined, timestamp: new Date().toISOString() },
    metadata: undefined,
  };
}

function artifact(context: RequestContext, text: string) {
  const artifact: Artifact = {
    artifactId: `${context.taskId}:answer`,
    name: "answer",
    description: "",
    parts: [
      {
        content: { $case: "text", value: text },
        filename: "",
        mediaType: "text/plain",
        metadata: undefined,
      },
    ],
    extensions: [],
    metadata: undefined,
  };
  return {
    taskId: context.taskId,
    contextId: context.contextId,
    artifact,
    append: false,
    lastChunk: true,
    metadata: undefined,
  };
}

function agentCard(url: string, name: string): AgentCard {
  return {
    name,
    description: "A2A gateway to a SwarmX Agent",
    supportedInterfaces: [
      { url, protocolBinding: "JSONRPC", protocolVersion: A2A_PROTOCOL_VERSION, tenant: "" },
    ],
    provider: { organization: "SwarmX", url: "https://github.com/blackscience/swarmx" },
    version: "1.0.0",
    capabilities: { streaming: false, pushNotifications: false, extensions: [] },
    securitySchemes: {
      bearer: {
        scheme: {
          $case: "httpAuthSecurityScheme",
          value: { description: "SwarmX Host token", scheme: "Bearer", bearerFormat: "opaque" },
        },
      },
    },
    securityRequirements: [{ schemes: { bearer: { list: [] } } }],
    defaultInputModes: ["text/plain"],
    defaultOutputModes: ["text/plain"],
    skills: [],
    signatures: [],
  };
}
