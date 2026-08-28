import { createHash } from "node:crypto";
import type { Context } from "@deepseek-ai/cordis";
import type { Agent } from "@deepseek-ai/dsh-agent";
import { createUserMessage } from "@deepseek-ai/dsh-llm";
import { type SessionEvent, SessionId } from "@deepseek-ai/dsh-session";
import type {} from "@deepseek-ai/dsh-subagent";
import type { ToolExecutionResult } from "@deepseek-ai/dsh-tools";
import { TypertRemoteService } from "@deepseek-ai/dsh-typert-protocol";
import s from "@deepseek-ai/schemastery";
import { isMutatingMemberTool, leadToolGuard, memberToolGuard } from "./capabilities.js";
import {
  type AddSwarmMemberRequest,
  type AdmitKnowledgeRequest,
  type CreateSwarmRequest,
  type CreateSwarmTaskRequest,
  type EscalateSwarmTaskRequest,
  type InterruptSwarmMemberRequest,
  type ReassignSwarmTaskRequest,
  type RecordSemanticFindingRequest,
  type RecordSwarmVerdictRequest,
  type ResolveSwarmEffectRequest,
  type SendSwarmMessageRequest,
  type StartSwarmVerificationRequest,
  type SubmitSwarmTaskRequest,
  type SwarmRole,
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
import { type KnowledgeOwners, OwnerKnowledgeCommitter } from "./knowledge.js";
import { redactSwarmText } from "./privacy.js";
import { applyMemberModelPolicy } from "./routing.js";

export * from "./capabilities.js";
export * from "./contracts.js";
export * from "./coordinator.js";
export * from "./errors.js";
export * from "./journal.js";
export * from "./knowledge.js";
export * from "./monitor.js";
export * from "./routing.js";
export * from "./team-policy.js";
export * from "./verification-model.js";

export interface Config extends Partial<SwarmCoordinatorConfig> {
  readonly monitorStallMs?: number;
  readonly semanticMonitor?: boolean;
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

const DEFAULT_MONITOR_STALL_MS = 5 * 60_000;

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
  const { effects: _effects, admissions: _admissions, ...visible } = snapshot;
  return swarmUiSnapshotSchema.parse({
    ...visible,
    name: redactSwarmText(snapshot.name, 100),
    members: snapshot.members.map((member) => ({
      ...member,
      description: redactSwarmText(member.description, 500),
    })),
    tasks: snapshot.tasks.map((task) => ({
      id: task.id,
      revision: task.revision,
      subject: redactSwarmText(task.subject, 200),
      kind: task.kind,
      status: task.status,
      ownerName: task.ownerName,
      verifierName: task.verifierName,
      blockedBy: task.blockedBy,
      ready: task.ready,
      budgetState: task.budgetState,
      usage: task.usage,
      escalationReason: task.escalationReason
        ? redactSwarmText(task.escalationReason, 500)
        : undefined,
      ...(task.submission
        ? {
            submission: {
              summary: redactSwarmText(task.submission.summary, 2_000),
              artifactCount: task.submission.artifactLocators.length,
              evidenceCount: task.submission.evidenceDigests.length,
              submittedAt: task.submission.submittedAt,
            },
          }
        : {}),
      ...(task.verification
        ? {
            verification: {
              verifierName: task.verification.verifierName,
              verdict: task.verification.verdict,
              mode: task.verification.mode,
              checkResults: task.verification.checkResults.map((check) => ({
                ...check,
                name: redactSwarmText(check.name, 200),
              })),
              rationale: redactSwarmText(task.verification.rationale, 4_000),
              recordedAt: task.verification.recordedAt,
            },
          }
        : {}),
    })),
    findings: (snapshot.findings ?? []).map((finding) => ({
      ...finding,
      summary: redactSwarmText(finding.summary, 500),
    })),
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
  static inject = ["agents", "approval", "pkb", "science", "subagents"];
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
    monitorStallMs: s.natural().min(1_000).max(86_400_000).default(DEFAULT_MONITOR_STALL_MS),
    semanticMonitor: s.boolean().default(false),
  });

  private readonly coordinator: SwarmCoordinator;
  private readonly journal: SwarmJournal;
  private readonly lifetime = new AbortController();
  private readonly operations = new Set<Promise<unknown>>();
  private readonly waiters = new Set<Waiter>();
  private readonly leadBoundaries = new Map<string, () => void>();
  private readonly intentionalStops = new Set<string>();
  private readonly monitorStallMs: number;
  private readonly semanticMonitor: boolean;
  private monitorTimer: ReturnType<typeof setTimeout> | undefined;
  private monitoring = false;
  private closed = false;

  constructor(ctx: Context, config: Config) {
    super(ctx, "swarm");
    this.journal = new SwarmJournal(config.root);
    this.journal.recoverUncertainIntents(Date.now());
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
        const agentOptions = request.agentOptions
          ? {
              ...(request.agentOptions.provider ? { provider: request.agentOptions.provider } : {}),
              ...(request.agentOptions.model ? { model: request.agentOptions.model } : {}),
              ...(request.agentOptions.maxTokens
                ? { maxTokens: request.agentOptions.maxTokens }
                : {}),
            }
          : undefined;
        const started = await ctx.subagents.startContinuable({
          childId: SessionId(request.childId),
          label: request.description,
          provider,
          request: {
            maxDepth: 1,
            parent,
            ...(agentOptions ? { agentOptions } : {}),
            persona: `You are Swarm ${request.role} ${request.name}: ${request.description}. Work only within this role and assigned Swarm tasks. Use the swarm tool for bounded coordination. Do not delegate or access PKB. Submit implementation evidence; only an authorized verifier or lead may accept it.`,
            prompt: [{ type: "text", text: request.prompt }],
          },
          signal: request.signal,
        });
        return started.childId;
      },
    };
    this.coordinator = new SwarmCoordinator(
      this.journal,
      runtime,
      {
        maxMembers: config.maxMembers ?? DEFAULT_CONFIG.maxMembers,
        maxMessageBytes: config.maxMessageBytes ?? DEFAULT_CONFIG.maxMessageBytes,
        maxPendingMessagesPerMember:
          config.maxPendingMessagesPerMember ?? DEFAULT_CONFIG.maxPendingMessagesPerMember,
        maxTasks: config.maxTasks ?? DEFAULT_CONFIG.maxTasks,
        quiescenceTimeoutMs: config.quiescenceTimeoutMs ?? DEFAULT_CONFIG.quiescenceTimeoutMs,
      },
      new OwnerKnowledgeCommitter(ctx as unknown as KnowledgeOwners),
    );
    this.monitorStallMs = config.monitorStallMs ?? DEFAULT_MONITOR_STALL_MS;
    this.semanticMonitor = config.semanticMonitor ?? false;
    for (const team of this.journal.list()) {
      const lead = ctx.agents.get(SessionId(team.id));
      if (lead && team.phase === "active") this.ensureLeadBoundary(lead);
    }

    ctx.effect(
      () =>
        ctx.subagents.registerContinuableSetup((childCtx: Context) => {
          const member = childCtx.agent;
          if (!member || !this.coordinator.isMemberIdentity(member.id)) return () => undefined;
          const profile = this.coordinator.memberProfileBySessionId(member.id);
          const disposePolicy = childCtx.on("agent/request", async (_payload, next) =>
            applyMemberModelPolicy(await next(), profile),
          );
          const disposeBoundary = this.registerEffectBoundary(childCtx, member, true);
          return () => {
            disposeBoundary();
            disposePolicy();
          };
        }),
      "dsh-swarm: guard continuable member capabilities",
    );
    ctx.on("agent/created", ({ agent }) => {
      if (this.coordinator.isLeadIdentity(agent.id)) {
        this.ensureLeadBoundary(agent);
        return;
      }
      if (!this.coordinator.isMemberIdentity(agent.id)) return;
      return this.track(async () => {
        await this.coordinator.recoverMember(agent);
        this.monitorAndArm();
        this.notify();
      });
    });
    ctx.on("agent/status", ({ agent }) => {
      if (
        !this.coordinator.isLeadIdentity(agent.id) &&
        !this.coordinator.isMemberIdentity(agent.id)
      ) {
        return;
      }
      this.monitorAndArm();
    });
    ctx.on("agent/disposed", ({ agent }) => {
      if (this.closed || this.intentionalStops.has(agent.id)) return;
      return this.track(async () => {
        await this.coordinator.recordMemberLifecycleFailure(agent.id);
        this.monitorAndArm();
        this.notify();
      });
    });
    ctx.on("session/event", (session, event) => this.recordSessionEvent(session.id, event));
    this.monitorAndArm();
    ctx.effect(() => () => this.dispose(), "dsh-swarm: drain runtime and close journal");
  }

  create(agent: Agent, request: CreateSwarmRequest, signal?: AbortSignal): Promise<SwarmSnapshot> {
    return this.mutate(agent, signal, async () => {
      await this.coordinator.create(agent, request);
      this.ensureLeadBoundary(agent);
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
      this.monitorAndArm();
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

  submitTask(
    agent: Agent,
    request: SubmitSwarmTaskRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot> {
    return this.mutate(agent, signal, async () => {
      await this.coordinator.submitTask(agent, request);
      this.queueSemanticMonitor(agent, "submission", request.taskId);
    });
  }

  startVerification(
    agent: Agent,
    request: StartSwarmVerificationRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot> {
    return this.mutate(agent, signal, async () => {
      await this.coordinator.startVerification(agent, request);
    });
  }

  recordVerdict(
    agent: Agent,
    request: RecordSwarmVerdictRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot> {
    return this.mutate(agent, signal, async () => {
      await this.coordinator.recordVerdict(agent, request);
      if (request.verdict !== "pass") {
        this.queueSemanticMonitor(agent, "verification_failure", request.taskId);
      }
    });
  }

  recordMonitorFinding(
    agent: Agent,
    request: RecordSemanticFindingRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot> {
    return this.mutate(agent, signal, async () => {
      this.coordinator.recordSemanticFinding(agent, request);
    });
  }

  escalateTask(
    agent: Agent,
    request: EscalateSwarmTaskRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot> {
    return this.mutate(agent, signal, async () => {
      await this.coordinator.escalateTask(agent, request);
    });
  }

  admitKnowledge(
    agent: Agent,
    request: AdmitKnowledgeRequest,
    callId: string,
    signal?: AbortSignal,
  ) {
    return this.run(async (operationSignal) => {
      try {
        return await this.coordinator.admitKnowledge(agent, request, {
          callId,
          signal: operationSignal,
        });
      } finally {
        this.monitorAndArm();
        this.notify();
      }
    }, signal);
  }

  resolveEffect(
    agent: Agent,
    request: ResolveSwarmEffectRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot> {
    return this.mutate(agent, signal, async () => {
      this.coordinator.resolveEffect(agent, request);
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
      const team = this.journal.findByParticipant(agent.id);
      for (const member of team?.members ?? []) {
        if (member.role !== "lead") this.intentionalStops.add(member.id);
      }
      try {
        await this.coordinator.archive(agent);
        this.leadBoundaries.get(agent.id)?.();
        this.leadBoundaries.delete(agent.id);
      } finally {
        for (const member of team?.members ?? []) this.intentionalStops.delete(member.id);
      }
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

  private ensureLeadBoundary(lead: Agent): void {
    if (this.leadBoundaries.has(lead.id)) return;
    this.leadBoundaries.set(lead.id, this.registerEffectBoundary(lead.ctx, lead, false));
  }

  private registerEffectBoundary(scope: Context, agent: Agent, member: boolean): () => void {
    const effects = new Map<string, string>();
    const disposeGuard = scope.tools.guard((execution) => {
      const denied = member
        ? memberToolGuard(
            agent,
            this.coordinator,
            execution,
            this.coordinator.memberProfile(agent).role as Exclude<SwarmRole, "lead">,
          )
        : leadToolGuard(agent, this.coordinator, execution);
      if (denied && member) {
        this.coordinator.recordRoleToolViolation(agent, execution.name);
        this.monitorAndArm();
        this.notify();
      }
      return denied;
    });
    const disposeExecution = scope.on("tools/execute", async (execution, next) => {
      if (execution.agent !== agent || execution.parent || !isMutatingMemberTool(execution.name)) {
        return next();
      }
      const effect = this.coordinator.beginToolEffect(agent, execution.callId, execution.name);
      effects.set(execution.callId, effect.id);
      return next();
    });
    const disposeResult = scope.on(
      "tools/result",
      (execution, result: Readonly<ToolExecutionResult>) => {
        const effectId = effects.get(execution.callId);
        if (!effectId) return;
        effects.delete(execution.callId);
        const digest = `sha256:${createHash("sha256")
          .update(JSON.stringify(result))
          .digest("hex")}`;
        this.coordinator.settleToolEffect(agent, effectId, {
          status: result.isError ? "uncertain" : "succeeded",
          ...(result.isError ? {} : { resultDigest: digest }),
        });
      },
    );
    return () => {
      disposeResult();
      disposeExecution();
      disposeGuard();
    };
  }

  private async mutate(
    agent: Agent,
    signal: AbortSignal | undefined,
    operation: (signal: AbortSignal) => Promise<void>,
  ): Promise<SwarmSnapshot> {
    return this.run(async (operationSignal) => {
      try {
        await operation(operationSignal);
      } finally {
        this.monitorAndArm();
        this.notify();
      }
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

  private recordSessionEvent(sessionId: SessionId, event: SessionEvent): void {
    if (
      this.closed ||
      (!this.coordinator.isLeadIdentity(sessionId) && !this.coordinator.isMemberIdentity(sessionId))
    ) {
      return;
    }
    const agent = this.ctx.agents.get(sessionId);
    if (!agent) return;
    let changed = false;
    if (event.type === "turn/start") {
      changed = this.coordinator.recordUsage(agent, { turns: 1 }) !== undefined;
    } else if (event.type === "tool/call") {
      changed = this.coordinator.recordUsage(agent, { toolCalls: 1 }) !== undefined;
    } else if (event.type === "assistant/message") {
      changed =
        this.coordinator.recordUsage(agent, {
          ...(event.data.usage ? { usage: event.data.usage } : {}),
          observedModel: {
            provider: event.data.message.source.provider,
            model: event.data.message.source.model,
          },
        }) !== undefined;
    }
    if (!changed) return;
    this.monitorAndArm();
    this.notify();
  }

  private monitorAndArm(): void {
    if (this.closed || this.monitoring) return;
    this.monitoring = true;
    try {
      const changedTeams = this.coordinator.runMonitor(Date.now(), this.monitorStallMs);
      if (changedTeams.length > 0) {
        this.notify();
        if (this.semanticMonitor) {
          for (const teamId of changedTeams) {
            void this.track(() =>
              this.coordinator
                .triggerSemanticMonitor(
                  teamId,
                  "deterministic_finding",
                  undefined,
                  this.lifetime.signal,
                )
                .then(() => {
                  this.monitorAndArm();
                  this.notify();
                }),
            ).catch(() => {
              if (this.closed) return;
              this.coordinator.recordSemanticMonitorDeliveryFailure(teamId);
              this.monitorAndArm();
              this.notify();
            });
          }
        }
      }
      if (this.monitorTimer) clearTimeout(this.monitorTimer);
      const deadline = this.coordinator.nextMonitorAt(Date.now(), this.monitorStallMs);
      this.monitorTimer = deadline
        ? setTimeout(() => this.monitorAndArm(), Math.min(2_147_483_647, deadline - Date.now()))
        : undefined;
    } finally {
      this.monitoring = false;
    }
  }

  private queueSemanticMonitor(
    agent: Agent,
    trigger: "submission" | "verification_failure",
    taskId: string,
  ): void {
    if (!this.semanticMonitor) return;
    const teamId = this.coordinator.teamId(agent);
    void this.track(() =>
      this.coordinator
        .triggerSemanticMonitor(teamId, trigger, taskId, this.lifetime.signal)
        .then(() => {
          this.monitorAndArm();
          this.notify();
        }),
    ).catch(() => {
      if (this.closed) return;
      this.coordinator.recordSemanticMonitorDeliveryFailure(teamId);
      this.monitorAndArm();
      this.notify();
    });
  }

  private async dispose(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    if (this.monitorTimer) clearTimeout(this.monitorTimer);
    this.monitorTimer = undefined;
    this.lifetime.abort(new SwarmError("Swarm service is closing", "SWARM_CLOSED"));
    for (const waiter of [...this.waiters]) waiter.reject(this.lifetime.signal.reason);
    this.waiters.clear();
    for (const dispose of this.leadBoundaries.values()) dispose();
    this.leadBoundaries.clear();
    await Promise.allSettled([...this.operations]);
    const drains = this.journal.list().flatMap((team) => {
      const lead = this.ctx.agents.get(SessionId(team.id));
      if (!lead) return [];
      const children = team.members
        .filter((member) => member.role !== "lead")
        .map((member) => SessionId(member.id));
      return [this.ctx.subagents.drainContinuableChildren(lead, children)];
    });
    const settled = await Promise.allSettled(drains);
    this.journal.recoverUncertainIntents(Date.now());
    this.journal.recoverInterruptedTasks(Date.now());
    this.journal.close();
    const failures = settled.flatMap((result) =>
      result.status === "rejected" ? [result.reason] : [],
    );
    if (failures.length > 0) throw new AggregateError(failures, "Failed to drain swarm members");
  }
}

export default SwarmService;
