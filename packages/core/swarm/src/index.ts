import type { Context } from "@deepseek-ai/cordis";
import type { Agent } from "@deepseek-ai/dsh-agent";
import { createUserMessage } from "@deepseek-ai/dsh-llm";
import { SessionId } from "@deepseek-ai/dsh-session";
import type {} from "@deepseek-ai/dsh-subagent";
import { TypertRemoteService } from "@deepseek-ai/dsh-typert-protocol";
import s from "@deepseek-ai/schemastery";
import { memberToolGuard } from "./capabilities.js";
import {
  type AddSwarmMemberRequest,
  type CreateSwarmRequest,
  type CreateSwarmTaskRequest,
  type InterruptSwarmMemberRequest,
  type ReassignSwarmTaskRequest,
  type SendSwarmMessageRequest,
  type SwarmSnapshot,
  type SwarmUiSnapshot,
  swarmUiSnapshotSchema,
  type UpdateSwarmTaskRequest,
  type WaitForSwarmChangeRequest,
  waitForSwarmChangeRequestSchema,
} from "./contracts.js";
import {
  SwarmCoordinator,
  type SwarmCoordinatorConfig,
  type SwarmRuntimeAdapter,
} from "./coordinator.js";
import { SwarmError } from "./errors.js";
import { SwarmJournal } from "./journal.js";

export * from "./capabilities.js";
export * from "./contracts.js";
export * from "./coordinator.js";
export * from "./errors.js";
export * from "./journal.js";

export interface Config extends Partial<SwarmCoordinatorConfig> {
  readonly provider?: string;
  readonly root: string;
}

const DEFAULT_CONFIG: SwarmCoordinatorConfig = {
  maxMembers: 8,
  maxMessageBytes: 16 * 1_024,
  maxPendingMessagesPerMember: 32,
  maxTasks: 256,
  quiescenceTimeoutMs: 30_000,
};

interface Waiter {
  reject(error: unknown): void;
  wake(): void;
}

declare module "@deepseek-ai/cordis" {
  interface Context {
    swarm: SwarmService;
  }
}

function uiProjection(snapshot: SwarmSnapshot): SwarmUiSnapshot {
  if (snapshot.kind === "inactive") return snapshot;
  return swarmUiSnapshotSchema.parse({
    ...snapshot,
    tasks: snapshot.tasks.map(
      ({ attemptId: _attemptId, description: _description, writeScopes: _writeScopes, ...task }) =>
        task,
    ),
  });
}

function parsedWaitRequest(request: WaitForSwarmChangeRequest): WaitForSwarmChangeRequest {
  try {
    return waitForSwarmChangeRequestSchema.parse(request);
  } catch (cause) {
    throw new SwarmError("Invalid swarm wait request", "SWARM_INVALID_REQUEST", { cause });
  }
}

/** Host-owned DSH swarm service; every mutating method requires an exact live Agent. */
export class SwarmService extends TypertRemoteService {
  static inject = ["agents", "subagents"];
  static Config = s.object({
    root: s.string().required(),
    provider: s.string().default("spawn"),
    maxMembers: s.natural().min(2).max(64).default(DEFAULT_CONFIG.maxMembers),
    maxMessageBytes: s.natural().min(1).max(65_536).default(DEFAULT_CONFIG.maxMessageBytes),
    maxPendingMessagesPerMember: s
      .natural()
      .min(1)
      .max(64)
      .default(DEFAULT_CONFIG.maxPendingMessagesPerMember),
    maxTasks: s.natural().min(1).max(2_048).default(DEFAULT_CONFIG.maxTasks),
    quiescenceTimeoutMs: s
      .natural()
      .min(100)
      .max(60_000)
      .default(DEFAULT_CONFIG.quiescenceTimeoutMs),
  });

  private readonly coordinator: SwarmCoordinator;
  private readonly journal: SwarmJournal;
  private readonly lifetime = new AbortController();
  private readonly operations = new Set<Promise<unknown>>();
  private readonly waiters = new Set<Waiter>();
  private closed = false;

  constructor(ctx: Context, config: Config) {
    super(ctx, "swarm");
    this.journal = new SwarmJournal(config.root);
    this.journal.recoverInterruptedTasks(Date.now());
    for (const team of this.journal.list()) {
      if (team.phase === "archived") continue;
      for (const member of team.members.filter((candidate) => candidate.phase === "provisioning")) {
        this.journal.append(team.id, {
          type: "member/updated",
          data: {
            ...member,
            error: "Host restarted before member provisioning completed",
            phase: "failed",
          },
        });
      }
    }

    const provider = config.provider ?? "spawn";
    const runtime: SwarmRuntimeAdapter = {
      exact: (agent) => ctx.agents.get(agent.id) === agent,
      getAgent: (id) => ctx.agents.get(SessionId(id)),
      workspaceKey: (agent) => this.journal.workspaceKey(agent.session.header.cwd),
      inject: (target, content, senderId) => {
        target.inject(
          createUserMessage({
            content: [{ type: "text", text: content }],
            source: {
              form: "relay",
              kind: "coordinator",
              senderSessionId: SessionId(senderId),
            },
          }),
        );
      },
      followup: async (parent, targetId, content, senderId, signal) => {
        await ctx.subagents.followup(
          parent,
          SessionId(targetId),
          [{ type: "text", text: content }],
          {
            signal: signal ?? this.lifetime.signal,
            source: {
              form: "relay",
              kind: "coordinator",
              senderSessionId: SessionId(senderId),
            },
          },
        );
      },
      followupRoot: (target, content, senderId) => {
        target.followup(
          createUserMessage({
            content: [{ type: "text", text: content }],
            source: {
              form: "relay",
              kind: "coordinator",
              senderSessionId: SessionId(senderId),
            },
          }),
        );
      },
      interrupt: (parent, targetId) => {
        ctx.subagents.interrupt(SessionId(targetId), { agent: parent, kind: "ancestor" });
      },
      stopContinuable: async (parent, targetId) => {
        await ctx.subagents.drainContinuableChildren(parent, [SessionId(targetId)]);
      },
      startContinuable: async (parent, request) => {
        const started = await ctx.subagents.startContinuable({
          childId: SessionId(request.childId),
          label: request.description,
          provider,
          request: {
            maxDepth: 1,
            parent,
            persona: `You are Swarm member ${request.name}: ${request.description}. Work only on assigned Swarm tasks. Use the swarm tool for messages and task settlement. Do not delegate or access PKB.`,
            prompt: [{ type: "text", text: request.prompt }],
          },
          signal: request.signal,
        });
        return started.childId;
      },
    };
    this.coordinator = new SwarmCoordinator(this.journal, runtime, {
      maxMembers: config.maxMembers ?? DEFAULT_CONFIG.maxMembers,
      maxMessageBytes: config.maxMessageBytes ?? DEFAULT_CONFIG.maxMessageBytes,
      maxPendingMessagesPerMember:
        config.maxPendingMessagesPerMember ?? DEFAULT_CONFIG.maxPendingMessagesPerMember,
      maxTasks: config.maxTasks ?? DEFAULT_CONFIG.maxTasks,
      quiescenceTimeoutMs: config.quiescenceTimeoutMs ?? DEFAULT_CONFIG.quiescenceTimeoutMs,
    });

    ctx.effect(
      () =>
        ctx.subagents.registerContinuableSetup((childCtx: Context) => {
          const member = childCtx.agent;
          if (!member || !this.coordinator.isMemberIdentity(member.id)) return () => undefined;
          return childCtx.tools.guard((execution) =>
            memberToolGuard(member, this.coordinator, execution),
          );
        }),
      "dsh-swarm: guard continuable member capabilities",
    );
    ctx.on("agent/created", ({ agent }) => {
      if (!this.coordinator.isMemberIdentity(agent.id)) return;
      return this.track(async () => {
        await this.coordinator.recoverMember(agent);
        this.notify();
      });
    });
    ctx.effect(() => () => this.dispose(), "dsh-swarm: drain runtime and close journal");
  }

  create(agent: Agent, request: CreateSwarmRequest, signal?: AbortSignal): Promise<SwarmSnapshot> {
    return this.mutate(agent, signal, async () => {
      await this.coordinator.create(agent, request);
    });
  }

  snapshot(agent: Agent): Promise<SwarmSnapshot> {
    return this.run(() => this.coordinator.snapshot(agent));
  }

  addMember(
    agent: Agent,
    request: AddSwarmMemberRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot> {
    return this.mutate(agent, signal, async (operationSignal) => {
      await this.coordinator.addMember(agent, request, operationSignal);
    });
  }

  sendMessage(agent: Agent, request: SendSwarmMessageRequest, signal?: AbortSignal) {
    return this.run(async (operationSignal) => {
      const sent = await this.coordinator.sendMessage(agent, request, operationSignal);
      this.notify();
      return sent;
    }, signal);
  }

  createTask(
    agent: Agent,
    request: CreateSwarmTaskRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot> {
    return this.mutate(agent, signal, async () => {
      await this.coordinator.createTask(agent, request);
    });
  }

  updateTask(
    agent: Agent,
    request: UpdateSwarmTaskRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot> {
    return this.mutate(agent, signal, async () => {
      await this.coordinator.updateTask(agent, request);
    });
  }

  reassignTask(
    agent: Agent,
    request: ReassignSwarmTaskRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot> {
    return this.mutate(agent, signal, async () => {
      await this.coordinator.reassignTask(agent, request);
    });
  }

  interruptMember(
    agent: Agent,
    request: InterruptSwarmMemberRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot> {
    return this.mutate(agent, signal, async () => {
      await this.coordinator.interruptMember(agent, request);
    });
  }

  waitForChange(
    agent: Agent,
    input: WaitForSwarmChangeRequest,
    signal: AbortSignal,
  ): Promise<SwarmSnapshot> {
    return this.run(async (operationSignal) => {
      const request = parsedWaitRequest(input);
      const current = await this.coordinator.snapshot(agent);
      if (current.revision > request.afterRevision) return current;
      return new Promise<SwarmSnapshot>((resolve, reject) => {
        let timer: ReturnType<typeof setTimeout> | undefined;
        const cleanup = () => {
          if (timer) clearTimeout(timer);
          operationSignal.removeEventListener("abort", aborted);
          this.waiters.delete(waiter);
        };
        const settle = () => {
          cleanup();
          void this.coordinator.snapshot(agent).then(resolve, reject);
        };
        const aborted = () => {
          cleanup();
          reject(operationSignal.reason);
        };
        const waiter: Waiter = { reject, wake: settle };
        this.waiters.add(waiter);
        operationSignal.addEventListener("abort", aborted, { once: true });
        timer = setTimeout(settle, request.timeoutMs);
      });
    }, signal);
  }

  archive(agent: Agent, signal?: AbortSignal): Promise<SwarmSnapshot> {
    return this.mutate(agent, signal, async () => {
      await this.coordinator.archive(agent);
    });
  }

  async uiSnapshot(sessionId: SessionId, signal?: AbortSignal): Promise<SwarmUiSnapshot> {
    signal?.throwIfAborted();
    return uiProjection(await this.snapshot(this.exactAgent(sessionId)));
  }

  async waitUi(
    sessionId: SessionId,
    request: WaitForSwarmChangeRequest,
    signal: AbortSignal,
  ): Promise<SwarmUiSnapshot> {
    return uiProjection(await this.waitForChange(this.exactAgent(sessionId), request, signal));
  }

  private exactAgent(sessionId: SessionId): Agent {
    const agent = this.ctx.agents.get(SessionId(sessionId));
    if (!agent) throw new SwarmError("Live swarm session not found", "SWARM_UNAUTHORIZED");
    return agent;
  }

  private async mutate(
    agent: Agent,
    signal: AbortSignal | undefined,
    operation: (signal: AbortSignal) => Promise<void>,
  ): Promise<SwarmSnapshot> {
    return this.run(async (operationSignal) => {
      await operation(operationSignal);
      this.notify();
      return this.coordinator.snapshot(agent);
    }, signal);
  }

  private run<T>(
    operation: (signal: AbortSignal) => Promise<T>,
    callerSignal?: AbortSignal,
  ): Promise<T> {
    if (this.closed)
      return Promise.reject(new SwarmError("Swarm service is closed", "SWARM_CLOSED"));
    const signal = callerSignal
      ? AbortSignal.any([callerSignal, this.lifetime.signal])
      : this.lifetime.signal;
    signal.throwIfAborted();
    return this.track(() => operation(signal));
  }

  private track<T>(operation: () => Promise<T>): Promise<T> {
    const promise = operation();
    this.operations.add(promise);
    void promise.finally(() => this.operations.delete(promise)).catch(() => undefined);
    return promise;
  }

  private notify(): void {
    for (const waiter of [...this.waiters]) waiter.wake();
  }

  private async dispose(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    this.lifetime.abort(new SwarmError("Swarm service is closing", "SWARM_CLOSED"));
    for (const waiter of [...this.waiters]) waiter.reject(this.lifetime.signal.reason);
    this.waiters.clear();
    await Promise.allSettled([...this.operations]);
    const drains = this.journal.list().flatMap((team) => {
      const lead = this.ctx.agents.get(SessionId(team.id));
      if (!lead) return [];
      const children = team.members
        .filter((member) => member.role === "member")
        .map((member) => SessionId(member.id));
      return [this.ctx.subagents.drainContinuableChildren(lead, children)];
    });
    const settled = await Promise.allSettled(drains);
    this.journal.recoverInterruptedTasks(Date.now());
    this.journal.close();
    const failures = settled.flatMap((result) =>
      result.status === "rejected" ? [result.reason] : [],
    );
    if (failures.length > 0) throw new AggregateError(failures, "Failed to drain swarm members");
  }
}

export default SwarmService;
