import { createHash, randomUUID } from "node:crypto";
import type { Agent } from "@deepseek-ai/dsh-agent";
import type { TokenUsage } from "@deepseek-ai/dsh-llm";
import { SessionId } from "@deepseek-ai/dsh-session";
import {
  type AddSwarmMemberRequest,
  type AdmitKnowledgeRequest,
  type AttemptBudget,
  addSwarmMemberRequestSchema,
  admitKnowledgeRequestSchema,
  type CreateSwarmRequest,
  type CreateSwarmTaskRequest,
  createSwarmRequestSchema,
  createSwarmTaskRequestSchema,
  type EscalateSwarmTaskRequest,
  escalateSwarmTaskRequestSchema,
  type InterruptSwarmMemberRequest,
  interruptSwarmMemberRequestSchema,
  type KnowledgeCommitReceipt,
  knowledgeCommitReceiptSchema,
  MODEL_ROUTE_PATTERN,
  type ReassignSwarmTaskRequest,
  type RecordSemanticFindingRequest,
  type RecordSwarmVerdictRequest,
  type ResolveSwarmEffectRequest,
  reassignSwarmTaskRequestSchema,
  recordSemanticFindingRequestSchema,
  recordSwarmVerdictRequestSchema,
  resolveSwarmEffectRequestSchema,
  type SendSwarmMessageRequest,
  type StartSwarmVerificationRequest,
  type SubmitSwarmTaskRequest,
  type SwarmAttempt,
  type SwarmEffect,
  type SwarmMember,
  type SwarmMessage,
  type SwarmRole,
  type SwarmSnapshot,
  type SwarmTask,
  type SwarmTeamState,
  sendSwarmMessageRequestSchema,
  startSwarmVerificationRequestSchema,
  submitSwarmTaskRequestSchema,
  swarmSnapshotSchema,
  type UpdateSwarmTaskRequest,
  updateSwarmTaskRequestSchema,
} from "./contracts.js";
import { SwarmError } from "./errors.js";
import type { SwarmJournal } from "./journal.js";
import { evaluateSwarmMonitor } from "./monitor.js";
import { redactSwarmText } from "./privacy.js";

export interface StartSwarmMemberRequest {
  childId: string;
  name: string;
  description: string;
  prompt: string;
  role: Exclude<SwarmRole, "lead">;
  agentOptions?: AddSwarmMemberRequest["agentOptions"];
  budget?: AttemptBudget;
  signal: AbortSignal;
}

export interface SwarmRuntimeAdapter {
  exact(agent: Agent): boolean;
  getAgent(id: string): Agent | undefined;
  workspaceKey(agent: Agent): string;
  inject(target: Agent, content: string, senderId: string): void;
  followup(
    parent: Agent,
    targetId: string,
    content: string,
    senderId: string,
    signal?: AbortSignal,
  ): Promise<void>;
  followupRoot(target: Agent, content: string, senderId: string): void;
  interrupt(parent: Agent, targetId: string): void;
  stopContinuable(parent: Agent, targetId: string): Promise<void>;
  startContinuable(parent: Agent, request: StartSwarmMemberRequest): Promise<string>;
}

export interface SwarmCoordinatorConfig {
  maxMembers: number;
  maxMessageBytes: number;
  maxPendingMessagesPerMember: number;
  maxTasks: number;
  quiescenceTimeoutMs: number;
}

export interface SentSwarmMessage {
  id: string;
  status: "delivered" | "queued" | "uncertain";
}

export interface KnowledgeCommitContext {
  readonly callId: string;
  readonly signal: AbortSignal;
}

export interface KnowledgeCommitter {
  commit(
    lead: Agent,
    request: AdmitKnowledgeRequest,
    context: KnowledgeCommitContext,
  ): Promise<KnowledgeCommitReceipt>;
}

function requestHash(value: unknown): `sha256:${string}` {
  return `sha256:${createHash("sha256").update(JSON.stringify(value)).digest("hex")}`;
}

function deterministicUuid(value: string): string {
  const hex = createHash("sha256").update(value).digest("hex").slice(0, 32);
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-4${hex.slice(13, 16)}-8${hex.slice(17, 20)}-${hex.slice(20)}`;
}

function modelPolicy(
  source: "legacy-default" | "requested" | "observed",
  options:
    | {
        readonly provider?: string | undefined;
        readonly model?: string | undefined;
        readonly maxTokens?: number | undefined;
      }
    | undefined,
): SwarmMember["modelPolicy"] {
  const provider = options?.provider;
  const model = options?.model;
  const maxTokens = options?.maxTokens;
  return {
    source,
    ...(provider && MODEL_ROUTE_PATTERN.test(provider) ? { provider } : {}),
    ...(model && MODEL_ROUTE_PATTERN.test(model) ? { model } : {}),
    ...(maxTokens && Number.isSafeInteger(maxTokens) && maxTokens > 0 && maxTokens <= 1_000_000
      ? { maxTokens }
      : {}),
  };
}

function emptyUsage(): SwarmAttempt["usage"] {
  return {
    availability: "unknown",
    inputTokens: 0,
    outputTokens: 0,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    turns: 0,
    toolCalls: 0,
  };
}

function budgetState(attempt: SwarmAttempt, now: number): SwarmAttempt["budgetState"] {
  const actor = attempt.actors.find((candidate) => candidate.endedAt === undefined);
  const budget = actor?.budget ?? attempt.budget;
  const usage = actor?.usage ?? attempt.usage;
  const startedAt = actor?.startedAt ?? attempt.startedAt;
  if (!budget) return usage.availability === "known" ? "within" : "unknown";
  const input = usage.inputTokens + usage.cacheReadTokens + usage.cacheWriteTokens;
  const exhausted =
    (budget.maxWallMs !== undefined && now - startedAt > budget.maxWallMs) ||
    (budget.maxTurns !== undefined && usage.turns > budget.maxTurns) ||
    (budget.maxInputTokens !== undefined && input > budget.maxInputTokens) ||
    (budget.maxOutputTokens !== undefined && usage.outputTokens > budget.maxOutputTokens);
  if (exhausted) return "exhausted";
  const warning =
    (budget.maxWallMs !== undefined &&
      now - startedAt >= budget.maxWallMs * budget.warningFraction) ||
    (budget.maxTurns !== undefined && usage.turns >= budget.maxTurns * budget.warningFraction) ||
    (budget.maxInputTokens !== undefined &&
      input >= budget.maxInputTokens * budget.warningFraction) ||
    (budget.maxOutputTokens !== undefined &&
      usage.outputTokens >= budget.maxOutputTokens * budget.warningFraction);
  if (warning) return "warning";
  if (
    usage.availability === "unknown" &&
    (budget.maxInputTokens !== undefined || budget.maxOutputTokens !== undefined)
  ) {
    return "unknown";
  }
  return "within";
}

function modelLabel(policy: SwarmMember["modelPolicy"]): string {
  if (policy.provider && policy.model) return `${policy.provider}/${policy.model}`;
  return policy.model ?? policy.provider ?? "deployment default";
}

function canExecuteTask(role: SwarmRole): boolean {
  return role === "lead" || role === "legacy" || role === "researcher" || role === "implementer";
}

function canCreateTask(role: SwarmRole): boolean {
  return role === "lead" || role === "legacy" || role === "researcher";
}

function withoutAttempt(task: SwarmTask): Omit<SwarmTask, "attemptId"> {
  const { attemptId: _attemptId, ...rest } = task;
  return rest;
}

function exactStatus(agent: Agent | undefined): "running" | "idle" | "inactive" {
  if (!agent) return "inactive";
  return agent.status === "running" ? "running" : "idle";
}

function parse<T>(schema: { parse(input: unknown): T }, input: unknown): T {
  try {
    return schema.parse(input);
  } catch (cause) {
    throw new SwarmError("Invalid swarm request", "SWARM_INVALID_REQUEST", { cause });
  }
}

function requireActive(team: SwarmTeamState): void {
  if (team.phase === "archived") {
    throw new SwarmError("Swarm is archived", "SWARM_ARCHIVED");
  }
}

function hasUnresolvedEffect(team: SwarmTeamState, taskId: string): boolean {
  return team.effects.some(
    (effect) =>
      effect.taskId === taskId && (effect.status === "started" || effect.status === "uncertain"),
  );
}

function formatDelivery(senderName: string, content: string): string {
  return `<swarm-message from="${senderName}">\n${content}\n</swarm-message>`;
}

function formatAssignment(task: SwarmTask): string {
  return [
    `<swarm-task id="${task.id}" revision="${task.revision}" attempt="${task.attemptId}">`,
    `Subject: ${task.subject}`,
    `Kind: ${task.kind}`,
    `Write scopes: ${task.writeScopes.join(", ") || "none"}`,
    task.description,
    "Submit results and evidence through submit_task with this exact revision and attempt. Submission is not acceptance.",
    "</swarm-task>",
  ].join("\n");
}

async function waitForIdle(agent: Agent, timeoutMs: number): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      agent.whenIdle(),
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(
          () => reject(new SwarmError("Swarm member did not become idle", "SWARM_INVALID_REQUEST")),
          timeoutMs,
        );
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** Durable authority, scheduling, task fencing, and mailbox policy for one DSH process. */
export class SwarmCoordinator {
  private readonly scheduling = new Map<string, Promise<void>>();

  constructor(
    private readonly journal: SwarmJournal,
    private readonly runtime: SwarmRuntimeAdapter,
    private readonly config: SwarmCoordinatorConfig,
    private readonly knowledge?: KnowledgeCommitter,
  ) {}

  async create(agent: Agent, input: CreateSwarmRequest): Promise<SwarmTeamState> {
    this.requireExact(agent);
    if (agent.session.header.origin === "subagent") {
      throw new SwarmError("Subagents cannot create nested swarms", "SWARM_UNAUTHORIZED");
    }
    if (this.journal.findByParticipant(agent.id)) {
      throw new SwarmError("Agent already belongs to a swarm", "SWARM_INVALID_REQUEST");
    }
    const request = parse(createSwarmRequestSchema, input);
    const now = Date.now();
    return this.journal.append(agent.id, {
      type: "team/created",
      data: {
        createdAt: now,
        lead: {
          createdAt: now,
          description: "Swarm lead",
          id: agent.id,
          name: "lead",
          phase: "active",
          role: "lead",
          modelPolicy: modelPolicy("observed", agent.options),
        },
        name: request.name,
        workspaceKey: this.runtime.workspaceKey(agent),
      },
    });
  }

  async addMember(
    agent: Agent,
    input: AddSwarmMemberRequest,
    signal: AbortSignal = new AbortController().signal,
  ): Promise<SwarmMember> {
    const team = this.requireLead(agent);
    requireActive(team);
    const request = parse(addSwarmMemberRequestSchema, input);
    if (team.members.length >= this.config.maxMembers) {
      throw new SwarmError("Swarm member limit reached", "SWARM_LIMIT");
    }
    if (team.members.some((member) => member.name === request.name)) {
      throw new SwarmError("Swarm member name already exists", "SWARM_INVALID_REQUEST");
    }

    const member: SwarmMember = {
      createdAt: Date.now(),
      description: request.description,
      id: SessionId(randomUUID()),
      name: request.name,
      phase: "provisioning",
      role: request.role,
      modelPolicy: request.agentOptions
        ? modelPolicy("requested", request.agentOptions)
        : modelPolicy("legacy-default", undefined),
      ...(request.budget ? { budget: request.budget } : {}),
    };
    this.journal.append(team.id, { type: "member/updated", data: member });
    let startedId: string | undefined;
    try {
      startedId = await this.runtime.startContinuable(agent, {
        childId: member.id,
        description: member.description,
        name: member.name,
        prompt: request.prompt,
        role: request.role,
        ...(request.agentOptions ? { agentOptions: request.agentOptions } : {}),
        ...(request.budget ? { budget: request.budget } : {}),
        signal,
      });
      if (startedId !== member.id) {
        throw new SwarmError(
          "Runtime changed the reserved member identity",
          "SWARM_INVALID_REQUEST",
        );
      }
      const active: SwarmMember = { ...member, phase: "active" };
      this.journal.append(team.id, { type: "member/updated", data: active });
      await this.kick(team.id);
      return active;
    } catch (cause) {
      const failed: SwarmMember = {
        ...member,
        error: cause instanceof Error ? cause.message.slice(0, 1_000) : "Member startup failed",
        phase: "failed",
      };
      const failures: unknown[] = [cause];
      try {
        this.journal.append(team.id, { type: "member/updated", data: failed });
      } catch (recordError) {
        failures.push(recordError);
      }
      if (startedId) {
        try {
          await this.runtime.stopContinuable(agent, startedId);
        } catch (stopError) {
          failures.push(stopError);
        }
      }
      if (failures.length > 1) {
        throw new AggregateError(failures, "Swarm member provisioning failed");
      }
      throw cause;
    }
  }

  async snapshot(agent: Agent): Promise<SwarmSnapshot> {
    this.requireExact(agent);
    const team = this.journal.findByParticipant(agent.id);
    if (!team) return { kind: "inactive", revision: 0 };
    const caller = team.members.find((member) => member.id === agent.id);
    if (!caller) return { kind: "inactive", revision: 0 };
    const nameById = new Map(team.members.map((member) => [member.id, member.name]));
    const completed = new Set(
      team.tasks.filter((task) => task.status === "completed").map((task) => task.id),
    );
    const latestAttemptForTask = (taskId: string) =>
      team.attempts.filter((attempt) => attempt.taskId === taskId).at(-1);
    const latestAttemptForMember = (memberId: string) =>
      team.attempts.filter((attempt) => attempt.ownerId === memberId).at(-1);
    return swarmSnapshotSchema.parse({
      kind: team.phase,
      memberName: caller.name,
      members: team.members.map((member) => ({
        budgetState: latestAttemptForMember(member.id)?.budgetState ?? "unknown",
        description: member.description,
        modelLabel: modelLabel(member.modelPolicy),
        name: member.name,
        role: member.role,
        status:
          member.phase === "active" ? exactStatus(this.runtime.getAgent(member.id)) : member.phase,
      })),
      name: team.name,
      pendingMessages: team.messages.filter(
        (message) => message.targetId === caller.id && message.deliveredAt === undefined,
      ).length,
      revision: team.revision,
      role: caller.role,
      tasks: team.tasks.map((task) => {
        const attempt = latestAttemptForTask(task.id);
        return {
          blockedBy: task.blockedBy,
          budgetState: attempt?.budgetState ?? "unknown",
          description: task.description,
          id: task.id,
          kind: task.kind,
          ownerName: task.ownerId ? nameById.get(task.ownerId) : undefined,
          verifierName: task.verifierId ? nameById.get(task.verifierId) : undefined,
          ready: task.blockedBy.every((dependency) => completed.has(dependency)),
          revision: task.revision,
          status: task.status,
          subject: task.subject,
          ...(task.acceptance ? { acceptance: task.acceptance } : {}),
          ...(task.submission ? { submission: task.submission } : {}),
          ...(task.verification ? { verification: task.verification } : {}),
          ...(task.escalationReason ? { escalationReason: task.escalationReason } : {}),
          ...(attempt
            ? {
                usage: {
                  ...attempt.usage,
                  wallMs: attempt.wallMs ?? Math.max(0, Date.now() - attempt.startedAt),
                },
              }
            : {}),
          writeScopes: task.writeScopes,
          ...(task.attemptId && (caller.role === "lead" || task.ownerId === caller.id)
            ? { attemptId: task.attemptId }
            : {}),
        };
      }),
      effects: team.effects
        .filter((effect) => effect.status !== "succeeded")
        .map(({ id, taskId, attemptId, toolName, status }) => ({
          id,
          taskId,
          attemptId,
          toolName,
          status,
        })),
      admissions: team.admissions.map(({ id, taskId, attemptId, targetKind, status, receipt }) => ({
        id,
        taskId,
        attemptId,
        targetKind,
        status,
        ...(receipt ? { receipt } : {}),
      })),
      findings: team.findings
        .slice(-100)
        .map(({ severity, code, summary, action, recordedAt }) => ({
          severity,
          code,
          summary,
          action,
          recordedAt,
        })),
      updatedAt: team.updatedAt,
    });
  }

  memberByName(agent: Agent, name: string): Agent {
    const team = this.requireMember(agent);
    const member = team.members.find((candidate) => candidate.name === name);
    if (!member) throw new SwarmError("Swarm member not found", "SWARM_MEMBER_NOT_FOUND");
    const memberAgent = this.runtime.getAgent(member.id);
    if (!memberAgent) {
      throw new SwarmError("Swarm member is inactive", "SWARM_MEMBER_NOT_FOUND");
    }
    return memberAgent;
  }

  task(agent: Agent, taskId: string): SwarmTask {
    const team = this.requireMember(agent);
    const task = team.tasks.find((candidate) => candidate.id === taskId);
    if (!task) throw new SwarmError("Swarm task not found", "SWARM_TASK_NOT_FOUND");
    return structuredClone(task);
  }

  async createTask(agent: Agent, input: CreateSwarmTaskRequest): Promise<SwarmTask> {
    const team = this.requireMember(agent);
    requireActive(team);
    const caller = this.findMember(team, agent.id);
    if (!canCreateTask(caller.role)) {
      throw new SwarmError("This Swarm role cannot create tasks", "SWARM_UNAUTHORIZED");
    }
    const request = parse(createSwarmTaskRequestSchema, input);
    if (team.tasks.length >= this.config.maxTasks) {
      throw new SwarmError("Swarm task limit reached", "SWARM_LIMIT");
    }
    if (request.kind !== "write" && request.writeScopes.length > 0) {
      throw new SwarmError("Only write tasks may declare write scopes", "SWARM_INVALID_REQUEST");
    }
    if (request.kind === "write" && request.writeScopes.length === 0) {
      throw new SwarmError("Write tasks must declare write scopes", "SWARM_INVALID_REQUEST");
    }
    for (const dependency of request.blockedBy) {
      if (!team.tasks.some((task) => task.id === dependency)) {
        throw new SwarmError("Swarm task dependency not found", "SWARM_TASK_DEPENDENCY");
      }
    }
    const owner = request.assignedTo
      ? team.members.find(
          (member) => member.name === request.assignedTo && member.phase === "active",
        )
      : undefined;
    if (request.assignedTo && !owner) {
      throw new SwarmError("Assigned swarm member not found", "SWARM_MEMBER_NOT_FOUND");
    }
    if (owner && !canExecuteTask(owner.role)) {
      throw new SwarmError("This Swarm role cannot own implementation tasks", "SWARM_UNAUTHORIZED");
    }
    const verifier = request.verifier
      ? team.members.find(
          (member) =>
            member.name === request.verifier &&
            member.phase === "active" &&
            (member.role === "verifier" || member.role === "lead"),
        )
      : undefined;
    if (request.verifier && !verifier) {
      throw new SwarmError("Assigned Swarm verifier not found", "SWARM_MEMBER_NOT_FOUND");
    }
    const sequence = Math.max(0, ...team.tasks.map((task) => task.sequence)) + 1;
    const now = Date.now();
    const task: SwarmTask = {
      blockedBy: request.blockedBy,
      createdAt: now,
      description: request.description,
      id: `task-${sequence}`,
      kind: request.kind,
      ownerId: owner?.id,
      verifierId: verifier?.id,
      revision: 1,
      sequence,
      status: "pending",
      subject: request.subject,
      updatedAt: now,
      writeScopes: request.writeScopes,
      ...(request.acceptance ? { acceptance: request.acceptance } : {}),
    };
    this.journal.append(team.id, { type: "task/updated", data: task });
    await this.kick(team.id);
    return this.task(agent, task.id);
  }

  async updateTask(agent: Agent, input: UpdateSwarmTaskRequest): Promise<SwarmTask> {
    const team = this.requireMember(agent);
    requireActive(team);
    const request = parse(updateSwarmTaskRequestSchema, input);
    const task = this.findTask(team, request.taskId);
    if (
      task.kind === "knowledge" &&
      team.admissions.some(
        (admission) => admission.taskId === task.id && admission.status === "started",
      )
    ) {
      throw new SwarmError("Knowledge admission is in progress", "SWARM_INVALID_REQUEST");
    }
    if (task.kind === "write" && hasUnresolvedEffect(team, task.id)) {
      throw new SwarmError(
        "Resolve the uncertain Tool effect before settling this write task",
        "SWARM_EFFECT_UNCERTAIN",
      );
    }
    if (task.kind === "knowledge" && request.action === "complete") {
      throw new SwarmError(
        "Knowledge tasks complete only through evidence admission",
        "SWARM_UNAUTHORIZED",
      );
    }
    if (task.revision !== request.expectedRevision) {
      throw new SwarmError("Swarm task revision is stale", "SWARM_STALE_REVISION");
    }
    if (task.attemptId !== request.attemptId) {
      throw new SwarmError("Swarm task attempt is stale", "SWARM_STALE_ATTEMPT");
    }
    if (task.status !== "in_progress" || task.ownerId !== agent.id) {
      throw new SwarmError("Only the active task owner may update it", "SWARM_UNAUTHORIZED");
    }
    if (!canExecuteTask(this.findMember(team, agent.id).role)) {
      throw new SwarmError(
        "This Swarm role cannot settle implementation tasks",
        "SWARM_UNAUTHORIZED",
      );
    }
    if (request.action === "complete") {
      throw new SwarmError(
        "Implementation tasks must be submitted and independently verified",
        "SWARM_VERIFICATION_REQUIRED",
      );
    }
    const status = request.action === "fail" ? "failed" : "pending";
    const next = {
      ...withoutAttempt(task),
      ...(request.action === "release" ? { ownerId: undefined } : {}),
      revision: task.revision + 1,
      status,
      updatedAt: Date.now(),
    } satisfies SwarmTask;
    const attempt = team.attempts.find((candidate) => candidate.id === request.attemptId);
    if (attempt) {
      const now = Date.now();
      this.journal.append(team.id, {
        type: "attempt/ended",
        data: {
          task: next,
          attempt: {
            ...attempt,
            revision: attempt.revision + 1,
            status: request.action === "fail" ? "failed" : "released",
            endedAt: now,
            wallMs: Math.max(0, now - attempt.startedAt),
            lastProgressAt: now,
            terminalReason:
              request.action === "fail" ? "Owner reported failure" : "Owner released task",
          },
        },
      });
    } else {
      this.journal.append(team.id, { type: "task/updated", data: next });
    }
    await this.kick(team.id);
    return this.task(agent, task.id);
  }

  async submitTask(agent: Agent, input: SubmitSwarmTaskRequest): Promise<SwarmTask> {
    const team = this.requireMember(agent);
    requireActive(team);
    const request = parse(submitSwarmTaskRequestSchema, input);
    const task = this.findTask(team, request.taskId);
    if (task.revision !== request.expectedRevision) {
      throw new SwarmError("Swarm task revision is stale", "SWARM_STALE_REVISION");
    }
    if (task.attemptId !== request.attemptId) {
      throw new SwarmError("Swarm task attempt is stale", "SWARM_STALE_ATTEMPT");
    }
    if (task.kind === "knowledge") {
      throw new SwarmError("Knowledge tasks use evidence admission", "SWARM_UNAUTHORIZED");
    }
    if (task.status !== "in_progress" || task.ownerId !== agent.id) {
      throw new SwarmError("Only the active task owner may submit it", "SWARM_UNAUTHORIZED");
    }
    if (!canExecuteTask(this.findMember(team, agent.id).role)) {
      throw new SwarmError(
        "This Swarm role cannot submit implementation tasks",
        "SWARM_UNAUTHORIZED",
      );
    }
    if (task.kind === "write" && hasUnresolvedEffect(team, task.id)) {
      throw new SwarmError(
        "Resolve the uncertain Tool effect before submitting this write task",
        "SWARM_EFFECT_UNCERTAIN",
      );
    }
    const attempt = this.findAttempt(team, request.attemptId);
    if (attempt.status !== "active" || attempt.ownerId !== agent.id) {
      throw new SwarmError("Swarm task attempt is stale", "SWARM_STALE_ATTEMPT");
    }
    const now = Date.now();
    const submission = {
      id: randomUUID(),
      attemptId: request.attemptId,
      summary: request.summary,
      artifactLocators: request.artifactLocators,
      evidenceDigests: request.evidenceDigests,
      submittedAt: now,
    };
    const nextTask: SwarmTask = {
      ...task,
      revision: task.revision + 1,
      status: "submitted",
      submission,
      updatedAt: now,
    };
    const nextAttempt: SwarmAttempt = {
      ...attempt,
      revision: attempt.revision + 1,
      status: "submitted",
      submittedAt: now,
      wallMs: Math.max(0, now - attempt.startedAt),
      lastProgressAt: now,
      submission,
      budgetState: budgetState(attempt, now),
      actors: attempt.actors.map((actor) =>
        actor.phase === "implementation" && actor.endedAt === undefined
          ? { ...actor, endedAt: now }
          : actor,
      ),
    };
    this.journal.append(team.id, {
      type: "task/submitted",
      data: { task: nextTask, attempt: nextAttempt },
    });
    return this.task(agent, task.id);
  }

  async startVerification(agent: Agent, input: StartSwarmVerificationRequest): Promise<SwarmTask> {
    const team = this.requireMember(agent);
    requireActive(team);
    const request = parse(startSwarmVerificationRequestSchema, input);
    const task = this.findTask(team, request.taskId);
    this.requireCurrentSubmission(task, request);
    if (task.status !== "submitted") {
      throw new SwarmError("Swarm task is not awaiting verification", "SWARM_INVALID_REQUEST");
    }
    const verifier = this.findMember(team, agent.id);
    const isLead = agent.id === team.id;
    if ((!isLead && task.verifierId !== agent.id) || (!isLead && verifier.role !== "verifier")) {
      throw new SwarmError(
        "Only the exact assigned verifier or lead may verify",
        "SWARM_UNAUTHORIZED",
      );
    }
    if (!isLead && task.ownerId === agent.id) {
      throw new SwarmError("Implementers cannot verify their own submission", "SWARM_UNAUTHORIZED");
    }
    const attempt = this.findAttempt(team, request.attemptId);
    if (attempt.status !== "submitted") {
      throw new SwarmError("Swarm task attempt is stale", "SWARM_STALE_ATTEMPT");
    }
    const now = Date.now();
    const nextTask: SwarmTask = {
      ...task,
      revision: task.revision + 1,
      status: "verifying",
      verificationStartedById: agent.id,
      verificationStartedAt: now,
      updatedAt: now,
    };
    const verifierUsage = emptyUsage();
    const nextAttempt: SwarmAttempt = {
      ...attempt,
      revision: attempt.revision + 1,
      status: "verifying",
      budgetState:
        verifier.budget?.maxInputTokens !== undefined ||
        verifier.budget?.maxOutputTokens !== undefined
          ? "unknown"
          : "within",
      lastProgressAt: now,
      actors: [
        ...attempt.actors,
        {
          phase: "verification",
          memberName: verifier.name,
          role: verifier.role,
          modelPolicy: verifier.modelPolicy,
          ...(verifier.budget ? { budget: verifier.budget } : {}),
          usage: verifierUsage,
          startedAt: now,
        },
      ],
    };
    this.journal.append(team.id, {
      type: "verification/started",
      data: { task: nextTask, attempt: nextAttempt },
    });
    return this.task(agent, task.id);
  }

  async recordVerdict(agent: Agent, input: RecordSwarmVerdictRequest): Promise<SwarmTask> {
    const team = this.requireMember(agent);
    requireActive(team);
    const request = parse(recordSwarmVerdictRequestSchema, input);
    const task = this.findTask(team, request.taskId);
    this.requireCurrentSubmission(task, request);
    if (task.status !== "verifying" || task.verificationStartedById !== agent.id) {
      throw new SwarmError(
        "Only the exact active verifier may record a verdict",
        "SWARM_UNAUTHORIZED",
      );
    }
    const verifier = this.findMember(team, agent.id);
    const selfVerification = task.ownerId === agent.id;
    if (selfVerification && agent.id !== team.id) {
      throw new SwarmError("Implementers cannot verify their own submission", "SWARM_UNAUTHORIZED");
    }
    const attempt = this.findAttempt(team, request.attemptId);
    if (attempt.status !== "verifying") {
      throw new SwarmError("Swarm task attempt is stale", "SWARM_STALE_ATTEMPT");
    }
    const now = Date.now();
    const verification = {
      verifierId: agent.id,
      verifierName: verifier.name,
      submissionId: request.submissionId,
      attemptId: request.attemptId,
      verdict: request.verdict,
      mode: selfVerification ? ("degraded" as const) : ("independent" as const),
      checkResults: request.checkResults,
      rationale: request.rationale,
      recordedAt: now,
    };
    const status =
      request.verdict === "pass"
        ? "completed"
        : request.verdict === "fail"
          ? "rejected"
          : "escalated";
    const { attemptId: _attemptId, ...taskWithoutAttempt } = task;
    const nextTask: SwarmTask = {
      ...taskWithoutAttempt,
      revision: task.revision + 1,
      status,
      verification,
      ...(status === "escalated" ? { escalationReason: request.rationale } : {}),
      updatedAt: now,
    };
    const attemptStatus =
      request.verdict === "pass"
        ? "accepted"
        : request.verdict === "fail"
          ? "rejected"
          : "escalated";
    const nextAttempt: SwarmAttempt = {
      ...attempt,
      revision: attempt.revision + 1,
      status: attemptStatus,
      verification,
      verifiedAt: now,
      endedAt: now,
      wallMs: Math.max(0, now - attempt.startedAt),
      lastProgressAt: now,
      terminalReason:
        request.verdict === "pass" ? "Verification passed" : `Verification ${request.verdict}`,
      actors: attempt.actors.map((actor) =>
        actor.phase === "verification" && actor.endedAt === undefined
          ? { ...actor, endedAt: now }
          : actor,
      ),
    };
    this.journal.append(team.id, {
      type: status === "escalated" ? "task/escalated" : "verification/recorded",
      data: { task: nextTask, attempt: nextAttempt },
    });
    await this.kick(team.id);
    return this.task(agent, task.id);
  }

  async escalateTask(agent: Agent, input: EscalateSwarmTaskRequest): Promise<SwarmTask> {
    const team = this.requireMember(agent);
    requireActive(team);
    const request = parse(escalateSwarmTaskRequestSchema, input);
    const task = this.findTask(team, request.taskId);
    if (task.revision !== request.expectedRevision) {
      throw new SwarmError("Swarm task revision is stale", "SWARM_STALE_REVISION");
    }
    if (task.attemptId !== request.attemptId) {
      throw new SwarmError("Swarm task attempt is stale", "SWARM_STALE_ATTEMPT");
    }
    if (task.submission && task.submission.id !== request.submissionId) {
      throw new SwarmError("Swarm task submission is stale", "SWARM_STALE_SUBMISSION");
    }
    if (agent.id !== team.id && agent.id !== task.ownerId && agent.id !== task.verifierId) {
      throw new SwarmError(
        "Only the task owner, verifier, or lead may escalate",
        "SWARM_UNAUTHORIZED",
      );
    }
    const attempt = this.findAttempt(team, request.attemptId);
    const now = Date.now();
    const { attemptId: _attemptId, ...taskWithoutAttempt } = task;
    const nextTask: SwarmTask = {
      ...taskWithoutAttempt,
      revision: task.revision + 1,
      status: "escalated",
      escalationReason: request.reason,
      updatedAt: now,
    };
    const nextAttempt: SwarmAttempt = {
      ...attempt,
      revision: attempt.revision + 1,
      status: "escalated",
      endedAt: now,
      wallMs: Math.max(0, now - attempt.startedAt),
      lastProgressAt: now,
      terminalReason: request.reason,
      actors: attempt.actors.map((actor) =>
        actor.endedAt === undefined ? { ...actor, endedAt: now } : actor,
      ),
    };
    this.journal.append(team.id, {
      type: "task/escalated",
      data: { task: nextTask, attempt: nextAttempt },
    });
    return this.task(agent, task.id);
  }

  recordSemanticFinding(agent: Agent, input: RecordSemanticFindingRequest): void {
    const team = this.requireMember(agent);
    requireActive(team);
    const member = this.findMember(team, agent.id);
    if (member.role !== "monitor") {
      throw new SwarmError(
        "Only the exact semantic monitor may record this finding",
        "SWARM_UNAUTHORIZED",
      );
    }
    const request = parse(recordSemanticFindingRequestSchema, input);
    if (request.subject.kind === "team" && request.subject.id !== "team") {
      throw new SwarmError("Semantic monitor Team subject is stale", "SWARM_INVALID_REQUEST");
    }
    if (
      request.subject.kind === "task" &&
      !team.tasks.some((task) => task.id === request.subject.id)
    ) {
      throw new SwarmError("Semantic monitor task subject is stale", "SWARM_TASK_NOT_FOUND");
    }
    const dedupeKey = `${request.code}:${request.subject.kind}:${request.subject.id}:${request.triggerId}`;
    if (team.findings.some((finding) => finding.dedupeKey === dedupeKey)) return;
    this.journal.append(team.id, {
      type: "monitor/finding-recorded",
      data: {
        id: randomUUID(),
        dedupeKey,
        severity: request.severity,
        code: request.code,
        subject: request.subject,
        summary: request.summary,
        action: request.action,
        recordedAt: Date.now(),
      },
    });
  }

  recordRoleToolViolation(agent: Agent, toolName: string): void {
    const team = this.requireMember(agent);
    requireActive(team);
    const member = this.findMember(team, agent.id);
    if (member.role !== "monitor" && member.role !== "verifier") return;
    const boundedTool = toolName.trim().slice(0, 200) || "unknown";
    const dedupeKey = `role_tool_violation:member:${member.name}:${boundedTool}`;
    if (team.findings.some((finding) => finding.dedupeKey === dedupeKey)) return;
    this.journal.append(team.id, {
      type: "monitor/finding-recorded",
      data: {
        id: randomUUID(),
        dedupeKey,
        severity: "block",
        code: "role_tool_violation",
        subject: { kind: "member", id: member.name },
        summary: `Read-only ${member.role} attempted a forbidden Tool.`,
        action: "lead_review",
        recordedAt: Date.now(),
      },
    });
  }

  async recordMemberLifecycleFailure(sessionId: string): Promise<void> {
    const team = this.journal.findByParticipant(sessionId);
    if (team?.phase !== "active") return;
    const member = team.members.find((candidate) => candidate.id === sessionId);
    if (!member || member.role === "lead" || member.phase !== "active") return;
    this.journal.append(team.id, {
      type: "member/updated",
      data: {
        ...member,
        phase: "failed",
        error: "Continuable member exited unexpectedly",
      },
    });
    await this.revokeAttempts(team.id, sessionId);
  }

  recordSemanticMonitorDeliveryFailure(teamId: string): void {
    const team = this.journal.get(teamId);
    if (team?.phase !== "active") return;
    const dedupeKey = `semantic_monitor_delivery_failed:team:${team.name}:${team.revision}`;
    this.journal.append(team.id, {
      type: "monitor/finding-recorded",
      data: {
        id: randomUUID(),
        dedupeKey,
        severity: "warning",
        code: "semantic_monitor_delivery_failed",
        subject: { kind: "team", id: team.name },
        summary: "The optional semantic monitor event could not be delivered.",
        action: "notify",
        recordedAt: Date.now(),
      },
    });
  }

  async triggerSemanticMonitor(
    teamId: string,
    trigger: "submission" | "verification_failure" | "deterministic_finding",
    taskId?: string,
    signal?: AbortSignal,
  ): Promise<boolean> {
    const team = this.journal.get(teamId);
    if (team?.phase !== "active") return false;
    const monitor = team.members.find(
      (member) => member.role === "monitor" && member.phase === "active",
    );
    const lead = this.runtime.getAgent(team.id);
    if (!monitor || !lead) return false;
    const task = taskId ? team.tasks.find((candidate) => candidate.id === taskId) : undefined;
    const attempt = task
      ? team.attempts.filter((candidate) => candidate.taskId === task.id).at(-1)
      : undefined;
    const triggerId = deterministicUuid(
      `${team.id}:${trigger}:${task?.id ?? "team"}:${task?.revision ?? team.revision}`,
    );
    const content = JSON.stringify({
      kind: "swarm-semantic-monitor-event",
      triggerId,
      trigger,
      task: task
        ? {
            id: task.id,
            subject: task.subject,
            status: task.status,
            acceptance: task.acceptance
              ? {
                  summary: redactSwarmText(task.acceptance.summary, 500),
                  requiredChecks: task.acceptance.requiredChecks
                    .slice(0, 12)
                    .map((check) => redactSwarmText(check, 100)),
                  expectedArtifacts: task.acceptance.expectedArtifacts
                    .slice(0, 12)
                    .map((artifact) => redactSwarmText(artifact, 100)),
                }
              : undefined,
            submission: task.submission
              ? {
                  summary: redactSwarmText(task.submission.summary, 500),
                  artifactCount: task.submission.artifactLocators.length,
                  evidenceCount: task.submission.evidenceDigests.length,
                }
              : undefined,
            verification: task.verification
              ? {
                  verdict: task.verification.verdict,
                  checks: task.verification.checkResults.slice(0, 16).map(({ name, status }) => ({
                    name: redactSwarmText(name, 100),
                    status,
                  })),
                }
              : undefined,
            budgetState: attempt?.budgetState ?? "unknown",
            usage: attempt?.usage ?? null,
          }
        : undefined,
      outputContract: {
        action: "record_monitor_finding",
        request: {
          triggerId,
          code: ["semantic_submission_concern", "semantic_conclusion_conflict"],
          subject: task ? { kind: "task", id: task.id } : { kind: "team", id: "team" },
          severity: ["info", "warning", "escalate"],
          action: ["none", "notify", "lead_review"],
          summary:
            "bounded finding; do not include commands, paths, credentials, or transcript text",
        },
      },
    });
    await this.sendMessage(
      lead,
      {
        target: monitor.name,
        content,
        delivery: "wakeup",
        idempotencyKey: triggerId,
      },
      signal,
    );
    return true;
  }

  teamId(agent: Agent): string {
    return this.requireMember(agent).id;
  }

  async reassignTask(agent: Agent, input: ReassignSwarmTaskRequest): Promise<SwarmTask> {
    const team = this.requireLead(agent);
    requireActive(team);
    const request = parse(reassignSwarmTaskRequestSchema, input);
    const task = this.findTask(team, request.taskId);
    if (task.revision !== request.expectedRevision) {
      throw new SwarmError("Swarm task revision is stale", "SWARM_STALE_REVISION");
    }
    if (task.status === "completed" || task.status === "cancelled") {
      throw new SwarmError("Finished swarm tasks cannot be reassigned", "SWARM_INVALID_REQUEST");
    }
    if (
      team.admissions.some(
        (admission) => admission.taskId === task.id && admission.status === "started",
      )
    ) {
      throw new SwarmError("Knowledge admission is in progress", "SWARM_INVALID_REQUEST");
    }
    if (task.kind === "write" && hasUnresolvedEffect(team, task.id)) {
      throw new SwarmError(
        "Resolve the uncertain Tool effect before reassigning this write task",
        "SWARM_EFFECT_UNCERTAIN",
      );
    }
    const target = team.members.find(
      (member) => member.name === request.target && member.phase === "active",
    );
    if (!target) throw new SwarmError("Swarm member not found", "SWARM_MEMBER_NOT_FOUND");
    if (!canExecuteTask(target.role)) {
      throw new SwarmError("This Swarm role cannot own implementation tasks", "SWARM_UNAUTHORIZED");
    }

    const activeActorId = task.status === "verifying" ? task.verificationStartedById : task.ownerId;
    if ((task.status === "in_progress" || task.status === "verifying") && activeActorId) {
      const oldOwner = this.runtime.getAgent(activeActorId);
      if (oldOwner) {
        this.runtime.interrupt(agent, oldOwner.id);
        await waitForIdle(oldOwner, this.config.quiescenceTimeoutMs);
      }
      const current = this.findTask(this.requireLead(agent), task.id);
      if (current.revision !== request.expectedRevision || current.attemptId !== task.attemptId) {
        throw new SwarmError("Swarm task changed during reassignment", "SWARM_STALE_REVISION");
      }
    }
    const {
      attemptId: _attemptId,
      submission: _submission,
      verification: _verification,
      verificationStartedById: _verificationStartedById,
      verificationStartedAt: _verificationStartedAt,
      escalationReason: _escalationReason,
      ...retryableTask
    } = task;
    const next = {
      ...retryableTask,
      ownerId: target.id,
      revision: task.revision + 1,
      status: "pending",
      updatedAt: Date.now(),
    } satisfies SwarmTask;
    const attempt = task.attemptId
      ? team.attempts.find((candidate) => candidate.id === task.attemptId)
      : undefined;
    if (attempt && ["active", "submitted", "verifying"].includes(attempt.status)) {
      const now = Date.now();
      this.journal.append(team.id, {
        type: "attempt/ended",
        data: {
          task: next,
          attempt: {
            ...attempt,
            revision: attempt.revision + 1,
            status: "interrupted",
            endedAt: now,
            wallMs: Math.max(0, now - attempt.startedAt),
            lastProgressAt: now,
            terminalReason: "Lead reassigned task",
            actors: attempt.actors.map((actor) =>
              actor.endedAt === undefined ? { ...actor, endedAt: now } : actor,
            ),
          },
        },
      });
    } else {
      this.journal.append(team.id, { type: "task/updated", data: next });
    }
    await this.kick(team.id);
    return this.task(agent, task.id);
  }

  async sendMessage(
    agent: Agent,
    input: SendSwarmMessageRequest,
    signal?: AbortSignal,
  ): Promise<SentSwarmMessage> {
    const team = this.requireMember(agent);
    requireActive(team);
    if (this.findMember(team, agent.id).role === "monitor") {
      throw new SwarmError("Monitor role cannot send peer messages", "SWARM_UNAUTHORIZED");
    }
    const request = parse(sendSwarmMessageRequestSchema, input);
    if (Buffer.byteLength(request.content, "utf8") > this.config.maxMessageBytes) {
      throw new SwarmError("Swarm message is too large", "SWARM_LIMIT");
    }
    const sender = team.members.find((member) => member.id === agent.id) as SwarmMember;
    const target = team.members.find((member) => member.name === request.target);
    if (!target) throw new SwarmError("Swarm member not found", "SWARM_MEMBER_NOT_FOUND");
    if (target.id === agent.id) {
      throw new SwarmError("Cannot send a swarm message to yourself", "SWARM_INVALID_REQUEST");
    }
    if (request.idempotencyKey) {
      const existing = team.messages.find((candidate) => candidate.id === request.idempotencyKey);
      if (existing) {
        if (
          existing.senderId !== agent.id ||
          existing.targetId !== target.id ||
          existing.delivery !== request.delivery ||
          existing.content !== request.content
        ) {
          throw new SwarmError(
            "Swarm message idempotency key conflicts with another message",
            "SWARM_MESSAGE_CONFLICT",
          );
        }
        return {
          id: existing.id,
          status:
            existing.deliveredAt !== undefined
              ? "delivered"
              : existing.deliveryStartedAt !== undefined
                ? "uncertain"
                : "queued",
        };
      }
    }
    const pending = team.messages.filter(
      (message) => message.targetId === target.id && message.deliveredAt === undefined,
    ).length;
    if (pending >= this.config.maxPendingMessagesPerMember) {
      throw new SwarmError("Swarm mailbox limit reached", "SWARM_LIMIT");
    }
    const message: SwarmMessage = {
      content: request.content,
      createdAt: Date.now(),
      delivery: request.delivery,
      id: request.idempotencyKey ?? randomUUID(),
      senderId: agent.id,
      senderName: sender.name,
      sequence: Math.max(0, ...team.messages.map((candidate) => candidate.sequence)) + 1,
      targetId: target.id,
    };
    this.journal.append(team.id, { type: "message/queued", data: message });
    const delivered = await this.deliver(team, message, signal);
    const current = this.journal
      .get(team.id)
      ?.messages.find((candidate) => candidate.id === message.id);
    return {
      id: message.id,
      status: delivered
        ? "delivered"
        : current?.deliveryStartedAt !== undefined
          ? "uncertain"
          : "queued",
    };
  }

  async recoverMember(agent: Agent): Promise<number> {
    const team = this.requireMember(agent);
    if (team.phase === "archived") return 0;
    let delivered = 0;
    for (const message of team.messages
      .filter(
        (candidate) =>
          candidate.targetId === agent.id &&
          candidate.deliveredAt === undefined &&
          candidate.deliveryStartedAt === undefined,
      )
      .sort((left, right) => left.sequence - right.sequence)) {
      if (await this.deliver(team, message)) delivered += 1;
    }
    await this.kick(team.id);
    return delivered;
  }

  async interruptMember(agent: Agent, input: InterruptSwarmMemberRequest): Promise<void> {
    const team = this.requireLead(agent);
    requireActive(team);
    const request = parse(interruptSwarmMemberRequestSchema, input);
    const target = team.members.find(
      (member) => member.name === request.target && member.role !== "lead",
    );
    if (!target) throw new SwarmError("Swarm member not found", "SWARM_MEMBER_NOT_FOUND");
    const targetAgent = this.runtime.getAgent(target.id);
    if (targetAgent) {
      this.runtime.interrupt(agent, target.id);
      await waitForIdle(targetAgent, this.config.quiescenceTimeoutMs);
    }
    await this.revokeAttempts(team.id, target.id);
  }

  async archive(agent: Agent): Promise<SwarmTeamState> {
    const team = this.requireLead(agent);
    requireActive(team);
    for (const member of team.members.filter((candidate) => candidate.role !== "lead")) {
      const child = this.runtime.getAgent(member.id);
      if (!child) continue;
      this.runtime.interrupt(agent, child.id);
      await waitForIdle(child, this.config.quiescenceTimeoutMs);
    }
    for (const member of team.members) await this.revokeAttempts(team.id, member.id);
    for (const member of team.members.filter((candidate) => candidate.role !== "lead")) {
      this.journal.append(team.id, {
        type: "member/updated",
        data: { ...member, phase: "retired" },
      });
      await this.runtime.stopContinuable(agent, member.id);
    }
    return this.journal.append(team.id, {
      type: "team/archived",
      data: { archivedAt: Date.now() },
    });
  }

  hasActiveWriteAttempt(agent: Agent): boolean {
    if (!this.runtime.exact(agent)) return false;
    const team = this.journal.findByParticipant(agent.id);
    return Boolean(
      team?.tasks.some(
        (task) =>
          task.kind === "write" && task.status === "in_progress" && task.ownerId === agent.id,
      ),
    );
  }

  beginToolEffect(agent: Agent, callId: string, toolName: string): SwarmEffect {
    const team = this.requireMember(agent);
    requireActive(team);
    const task = team.tasks.find(
      (candidate) =>
        candidate.kind === "write" &&
        candidate.status === "in_progress" &&
        candidate.ownerId === agent.id &&
        candidate.attemptId !== undefined,
    );
    if (!task?.attemptId) {
      throw new SwarmError(
        "Workspace mutation requires an active write attempt",
        "SWARM_UNAUTHORIZED",
      );
    }
    if (
      team.effects.some((effect) => effect.attemptId === task.attemptId && effect.callId === callId)
    ) {
      throw new SwarmError(
        "This Tool call already entered the effect boundary",
        "SWARM_DUPLICATE_EFFECT",
      );
    }
    if (hasUnresolvedEffect(team, task.id)) {
      throw new SwarmError(
        "Verify the previous uncertain Tool effect before another mutation",
        "SWARM_EFFECT_UNCERTAIN",
      );
    }
    const now = Date.now();
    const effect: SwarmEffect = {
      id: randomUUID(),
      revision: 1,
      callId,
      taskId: task.id,
      taskRevision: task.revision,
      attemptId: task.attemptId,
      ownerId: agent.id,
      toolName,
      status: "started",
      createdAt: now,
      updatedAt: now,
    };
    this.journal.append(team.id, { type: "effect/updated", data: effect });
    return structuredClone(effect);
  }

  settleToolEffect(
    agent: Agent,
    effectId: string,
    outcome: { readonly status: "succeeded" | "uncertain"; readonly resultDigest?: string },
  ): SwarmEffect {
    const team = this.requireMember(agent);
    const effect = team.effects.find((candidate) => candidate.id === effectId);
    if (!effect || effect.ownerId !== agent.id || effect.status !== "started") {
      throw new SwarmError("Swarm Tool effect is not active", "SWARM_UNAUTHORIZED");
    }
    const current = this.findTask(team, effect.taskId);
    const stillCurrent =
      current.status === "in_progress" &&
      current.ownerId === agent.id &&
      current.attemptId === effect.attemptId;
    const next: SwarmEffect = {
      ...effect,
      revision: effect.revision + 1,
      status: stillCurrent ? outcome.status : "uncertain",
      updatedAt: Date.now(),
      ...(stillCurrent && outcome.resultDigest
        ? { resultDigest: outcome.resultDigest as `sha256:${string}` }
        : {}),
    };
    this.journal.append(team.id, { type: "effect/updated", data: next });
    return structuredClone(next);
  }

  resolveEffect(agent: Agent, input: ResolveSwarmEffectRequest): SwarmEffect {
    const team = this.requireLead(agent);
    requireActive(team);
    const request = parse(resolveSwarmEffectRequestSchema, input);
    const task = this.findTask(team, request.taskId);
    if (task.revision !== request.expectedRevision) {
      throw new SwarmError("Swarm task revision is stale", "SWARM_STALE_REVISION");
    }
    const effect = team.effects.find((candidate) => candidate.id === request.effectId);
    if (
      !effect ||
      effect.taskId !== task.id ||
      effect.attemptId !== request.attemptId ||
      effect.status !== "uncertain"
    ) {
      throw new SwarmError("Uncertain Swarm Tool effect not found", "SWARM_STALE_ATTEMPT");
    }
    const next: SwarmEffect = {
      ...effect,
      revision: effect.revision + 1,
      status: request.resolution,
      updatedAt: Date.now(),
      verification: request.verification,
    };
    this.journal.append(team.id, { type: "effect/updated", data: next });
    return structuredClone(next);
  }

  async admitKnowledge(
    agent: Agent,
    input: AdmitKnowledgeRequest,
    context: KnowledgeCommitContext,
  ): Promise<KnowledgeCommitReceipt> {
    const team = this.requireLead(agent);
    requireActive(team);
    const request = parse(admitKnowledgeRequestSchema, input);
    const task = this.findTask(team, request.taskId);
    if (task.revision !== request.expectedRevision) {
      throw new SwarmError("Swarm task revision is stale", "SWARM_STALE_REVISION");
    }
    if (
      task.kind !== "knowledge" ||
      task.status !== "in_progress" ||
      task.attemptId !== request.attemptId
    ) {
      throw new SwarmError(
        "Evidence admission requires the active knowledge attempt",
        "SWARM_STALE_ATTEMPT",
      );
    }
    if (
      request.target.kind === "science_evidence" &&
      request.sources.some((source) => source.kind !== "science_entity")
    ) {
      throw new SwarmError(
        "Science evidence admission requires Science entity sources",
        "SWARM_INVALID_REQUEST",
      );
    }
    if (!this.knowledge) {
      throw new SwarmError("Knowledge owner services are unavailable", "SWARM_CLOSED");
    }

    const hash = requestHash({
      admissionId: request.admissionId,
      sources: request.sources,
      target: request.target,
      verification: request.verification,
    });
    const existing = team.admissions.find((admission) => admission.id === request.admissionId);
    if (existing?.requestHash !== undefined && existing.requestHash !== hash) {
      throw new SwarmError(
        "Knowledge admission id was reused for different content",
        "SWARM_ADMISSION_CONFLICT",
      );
    }
    if (existing?.status === "started") {
      throw new SwarmError("Knowledge admission outcome is uncertain", "SWARM_EFFECT_UNCERTAIN");
    }
    if (existing?.status === "committed" && existing.receipt) {
      this.completeKnowledgeTask(team.id, task);
      return structuredClone(existing.receipt);
    }

    const now = Date.now();
    const started = existing
      ? {
          ...existing,
          attemptId: request.attemptId,
          revision: existing.revision + 1,
          status: "started" as const,
          taskRevision: task.revision,
          updatedAt: now,
        }
      : {
          id: request.admissionId,
          revision: 1,
          taskId: task.id,
          taskRevision: task.revision,
          attemptId: request.attemptId,
          requestHash: hash,
          targetKind: request.target.kind,
          sources: request.sources,
          verification: request.verification,
          status: "started" as const,
          createdAt: now,
          updatedAt: now,
        };
    this.journal.append(team.id, { type: "knowledge/admission-updated", data: started });

    let receipt: KnowledgeCommitReceipt;
    try {
      receipt = knowledgeCommitReceiptSchema.parse(
        await this.knowledge.commit(agent, request, context),
      );
    } catch (cause) {
      this.journal.append(team.id, {
        type: "knowledge/admission-updated",
        data: {
          ...started,
          revision: started.revision + 1,
          status: "uncertain",
          updatedAt: Date.now(),
        },
      });
      throw cause;
    }
    this.journal.append(team.id, {
      type: "knowledge/admission-updated",
      data: {
        ...started,
        receipt,
        revision: started.revision + 1,
        status: "committed",
        updatedAt: Date.now(),
      },
    });
    this.completeKnowledgeTask(team.id, task);
    return structuredClone(receipt);
  }

  isMemberIdentity(sessionId: string): boolean {
    const team = this.journal.findByParticipant(sessionId);
    return Boolean(
      team?.phase === "active" &&
        team.members.some(
          (member) =>
            member.id === sessionId &&
            member.role !== "lead" &&
            (member.phase === "active" || member.phase === "provisioning"),
        ),
    );
  }

  isLeadIdentity(sessionId: string): boolean {
    const team = this.journal.findByParticipant(sessionId);
    return Boolean(team?.phase === "active" && team.id === sessionId);
  }

  memberProfile(agent: Agent): SwarmMember {
    const team = this.requireMember(agent);
    return structuredClone(this.findMember(team, agent.id));
  }

  memberProfileBySessionId(sessionId: string): SwarmMember {
    const team = this.journal.findByParticipant(sessionId);
    const member = team?.members.find((candidate) => candidate.id === sessionId);
    if (team?.phase !== "active" || !member || member.role === "lead") {
      throw new SwarmError("Swarm member not found", "SWARM_MEMBER_NOT_FOUND");
    }
    return structuredClone(member);
  }

  recordUsage(
    agent: Agent,
    delta: {
      readonly usage?: TokenUsage;
      readonly turns?: number;
      readonly toolCalls?: number;
      readonly observedModel?: { readonly provider?: string; readonly model?: string };
    },
  ): SwarmAttempt | undefined {
    const team = this.requireMember(agent);
    const task = team.tasks.find(
      (candidate) =>
        (candidate.status === "in_progress" && candidate.ownerId === agent.id) ||
        (candidate.status === "submitted" && candidate.ownerId === agent.id) ||
        (candidate.status === "verifying" && candidate.verificationStartedById === agent.id),
    );
    if (!task?.attemptId) return undefined;
    const attempt = this.findAttempt(team, task.attemptId);
    const now = Date.now();
    const applyDelta = (usage: SwarmAttempt["usage"]): SwarmAttempt["usage"] => ({
      availability: delta.usage ? "known" : usage.availability,
      inputTokens: usage.inputTokens + (delta.usage?.inputTokens ?? 0),
      outputTokens: usage.outputTokens + (delta.usage?.outputTokens ?? 0),
      cacheReadTokens: usage.cacheReadTokens + (delta.usage?.cacheReadTokens ?? 0),
      cacheWriteTokens: usage.cacheWriteTokens + (delta.usage?.cacheWriteTokens ?? 0),
      turns: usage.turns + (delta.turns ?? 0),
      toolCalls: usage.toolCalls + (delta.toolCalls ?? 0),
    });
    const usage = applyDelta(attempt.usage);
    const actors = attempt.actors.map((actor) =>
      actor.memberName === this.findMember(team, agent.id).name &&
      (actor.endedAt === undefined ||
        (task.status === "submitted" && actor.phase === "implementation"))
        ? {
            ...actor,
            usage: applyDelta(actor.usage),
            ...(delta.observedModel ? { observedModel: delta.observedModel } : {}),
          }
        : actor,
    );
    const next: SwarmAttempt = {
      ...attempt,
      revision: attempt.revision + 1,
      usage,
      actors,
      lastProgressAt: now,
      ...(delta.observedModel ? { observedModel: delta.observedModel } : {}),
    };
    next.budgetState = budgetState(next, now);
    this.journal.append(team.id, { type: "attempt/usage-recorded", data: next });
    return structuredClone(next);
  }

  runMonitor(now: number, stallMs: number): string[] {
    const changedTeams = new Set<string>();
    for (const team of this.journal.list().filter((candidate) => candidate.phase === "active")) {
      const runningMemberIds = new Set(
        team.members
          .filter((member) => this.runtime.getAgent(member.id)?.status === "running")
          .map((member) => member.id),
      );
      const drafts = evaluateSwarmMonitor(team, {
        now,
        stallMs,
        maxPendingMessagesPerMember: this.config.maxPendingMessagesPerMember,
        runningMemberIds,
      });
      for (const draft of drafts) {
        const exhaustion = draft.code.endsWith("_exhausted");
        if (draft.subject.kind === "attempt" && exhaustion) {
          const current = this.journal.get(team.id);
          const attempt = current?.attempts.find((candidate) => candidate.id === draft.subject.id);
          const task = attempt
            ? current?.tasks.find((candidate) => candidate.id === attempt.taskId)
            : undefined;
          if (
            current &&
            attempt &&
            task?.attemptId === attempt.id &&
            (attempt.status === "active" || attempt.status === "verifying")
          ) {
            const actor = attempt.actors.find((candidate) => candidate.endedAt === undefined);
            const actorMember = actor
              ? current.members.find((member) => member.name === actor.memberName)
              : undefined;
            this.runtime.getAgent(actorMember?.id ?? attempt.ownerId)?.cancel({
              kind: "hook",
              reason: `Swarm budget exhausted: ${draft.code}`,
            });
            const {
              attemptId: _attemptId,
              verificationStartedById: _verificationStartedById,
              verificationStartedAt: _verificationStartedAt,
              ...taskWithoutAttempt
            } = task;
            this.journal.append(team.id, {
              type: "attempt/ended",
              data: {
                task: {
                  ...taskWithoutAttempt,
                  revision: task.revision + 1,
                  status: "needs_attention",
                  escalationReason: draft.summary,
                  updatedAt: now,
                },
                attempt: {
                  ...attempt,
                  revision: attempt.revision + 1,
                  budgetState: "exhausted",
                  status: "budget_exhausted",
                  endedAt: now,
                  wallMs: Math.max(0, now - attempt.startedAt),
                  lastProgressAt: now,
                  terminalReason: draft.summary,
                  actors: attempt.actors.map((candidate) =>
                    candidate.endedAt === undefined ? { ...candidate, endedAt: now } : candidate,
                  ),
                },
              },
            });
          }
        } else if (draft.subject.kind === "attempt" && draft.code === "attempt_wall_warning") {
          const current = this.journal.get(team.id);
          const attempt = current?.attempts.find((candidate) => candidate.id === draft.subject.id);
          if (attempt && (attempt.status === "active" || attempt.status === "verifying")) {
            this.journal.append(team.id, {
              type: "attempt/budget-warning",
              data: {
                ...attempt,
                revision: attempt.revision + 1,
                budgetState: "warning",
                warningCodes: attempt.warningCodes.includes(draft.code)
                  ? attempt.warningCodes
                  : [...attempt.warningCodes, draft.code],
              },
            });
          }
        }
        this.journal.append(team.id, {
          type: "monitor/finding-recorded",
          data: { ...draft, id: randomUUID() },
        });
        changedTeams.add(team.id);
      }
    }
    return [...changedTeams];
  }

  nextMonitorAt(now: number, stallMs: number): number | undefined {
    let next: number | undefined;
    const include = (candidate: number) => {
      if (candidate <= now) candidate = now + 1;
      next = next === undefined ? candidate : Math.min(next, candidate);
    };
    for (const team of this.journal.list().filter((candidate) => candidate.phase === "active")) {
      for (const attempt of team.attempts) {
        if (!["active", "submitted", "verifying"].includes(attempt.status)) continue;
        const actor = attempt.actors.find((candidate) => candidate.endedAt === undefined);
        const budget = actor?.budget ?? attempt.budget;
        const startedAt = actor?.startedAt ?? attempt.startedAt;
        const actorMember = actor
          ? team.members.find((member) => member.name === actor.memberName)
          : undefined;
        const actorRunning =
          this.runtime.getAgent(actorMember?.id ?? attempt.ownerId)?.status === "running";
        const suffix = `${actor?.phase ?? "implementation"}:${actor?.memberName ?? attempt.memberName}`;
        const findingKeys = new Set(team.findings.map((finding) => finding.dedupeKey));
        if (budget?.maxWallMs !== undefined && attempt.status !== "submitted") {
          const subject = `attempt:${attempt.id}`;
          if (!findingKeys.has(`attempt_wall_warning:${subject}:${suffix}`)) {
            include(startedAt + budget.maxWallMs * budget.warningFraction);
          }
          if (!findingKeys.has(`attempt_wall_exhausted:${subject}:${suffix}`)) {
            include(startedAt + budget.maxWallMs + 1);
          }
        }
        const stalled = [...findingKeys].some(
          (key) =>
            (key.startsWith(`attempt_stalled:attempt:${attempt.id}:`) ||
              key.startsWith(`write_attempt_stalled:attempt:${attempt.id}:`)) &&
            key.endsWith(attempt.status),
        );
        if (actorRunning && !stalled) include(attempt.lastProgressAt + stallMs + 1);
      }
    }
    return next;
  }

  private requireExact(agent: Agent): void {
    if (!this.runtime.exact(agent)) {
      throw new SwarmError("Agent authority is not exact", "SWARM_UNAUTHORIZED");
    }
  }

  private completeKnowledgeTask(teamId: string, task: SwarmTask): void {
    const current = this.journal.get(teamId);
    const active = current ? this.findTask(current, task.id) : undefined;
    if (
      active?.kind !== "knowledge" ||
      active.status !== "in_progress" ||
      active.attemptId !== task.attemptId
    ) {
      throw new SwarmError("Knowledge task changed during admission", "SWARM_STALE_ATTEMPT");
    }
    const now = Date.now();
    const completed: SwarmTask = {
      ...withoutAttempt(active),
      revision: active.revision + 1,
      status: "completed",
      updatedAt: now,
    };
    const attempt = current?.attempts.find((candidate) => candidate.id === active.attemptId);
    if (attempt) {
      this.journal.append(teamId, {
        type: "attempt/ended",
        data: {
          task: completed,
          attempt: {
            ...attempt,
            revision: attempt.revision + 1,
            status: "accepted",
            endedAt: now,
            wallMs: Math.max(0, now - attempt.startedAt),
            lastProgressAt: now,
            terminalReason: "Knowledge admitted by owner",
            actors: attempt.actors.map((actor) =>
              actor.endedAt === undefined ? { ...actor, endedAt: now } : actor,
            ),
          },
        },
      });
    } else {
      this.journal.append(teamId, { type: "task/updated", data: completed });
    }
  }

  private requireMember(agent: Agent): SwarmTeamState {
    this.requireExact(agent);
    const team = this.journal.findByParticipant(agent.id);
    if (!team) throw new SwarmError("Agent does not belong to a swarm", "SWARM_NOT_FOUND");
    return team;
  }

  private requireLead(agent: Agent): SwarmTeamState {
    const team = this.requireMember(agent);
    if (team.id !== agent.id) {
      throw new SwarmError("Only the swarm lead may perform this action", "SWARM_UNAUTHORIZED");
    }
    return team;
  }

  private findTask(team: SwarmTeamState, taskId: string): SwarmTask {
    const task = team.tasks.find((candidate) => candidate.id === taskId);
    if (!task) throw new SwarmError("Swarm task not found", "SWARM_TASK_NOT_FOUND");
    return task;
  }

  private findMember(team: SwarmTeamState, memberId: string): SwarmMember {
    const member = team.members.find((candidate) => candidate.id === memberId);
    if (!member) throw new SwarmError("Swarm member not found", "SWARM_MEMBER_NOT_FOUND");
    return member;
  }

  private findAttempt(team: SwarmTeamState, attemptId: string): SwarmAttempt {
    const attempt = team.attempts.find((candidate) => candidate.id === attemptId);
    if (!attempt) throw new SwarmError("Swarm task attempt is stale", "SWARM_STALE_ATTEMPT");
    return attempt;
  }

  private requireCurrentSubmission(
    task: SwarmTask,
    request: {
      readonly expectedRevision: number;
      readonly attemptId: string;
      readonly submissionId: string;
    },
  ): void {
    if (task.revision !== request.expectedRevision) {
      throw new SwarmError("Swarm task revision is stale", "SWARM_STALE_REVISION");
    }
    if (
      task.attemptId !== request.attemptId ||
      !task.submission ||
      task.submission.attemptId !== request.attemptId
    ) {
      throw new SwarmError("Swarm task attempt is stale", "SWARM_STALE_ATTEMPT");
    }
    if (task.submission.id !== request.submissionId) {
      throw new SwarmError("Swarm task submission is stale", "SWARM_STALE_SUBMISSION");
    }
  }

  private async deliver(
    team: SwarmTeamState,
    message: SwarmMessage,
    signal?: AbortSignal,
  ): Promise<boolean> {
    const target = this.runtime.getAgent(message.targetId);
    if (!target) return false;
    if (message.deliveryStartedAt !== undefined) return false;
    this.journal.append(team.id, {
      type: "message/delivery-started",
      data: { deliveryStartedAt: Date.now(), messageId: message.id },
    });
    const content = formatDelivery(message.senderName, message.content);
    try {
      if (message.delivery === "quiet") {
        this.runtime.inject(target, content, message.senderId);
      } else if (target.id === team.id) {
        this.runtime.followupRoot(target, content, message.senderId);
      } else {
        const lead = this.runtime.getAgent(team.id);
        if (!lead) return false;
        await this.runtime.followup(lead, target.id, content, message.senderId, signal);
      }
      this.journal.append(team.id, {
        type: "message/delivered",
        data: { deliveredAt: Date.now(), messageId: message.id },
      });
      return true;
    } catch {
      return false;
    }
  }

  private async revokeAttempts(teamId: string, ownerId: string): Promise<void> {
    const team = this.journal.get(teamId);
    if (!team || team.phase === "archived") return;
    for (const task of team.tasks.filter(
      (candidate) =>
        (candidate.status === "in_progress" && candidate.ownerId === ownerId) ||
        (candidate.status === "verifying" && candidate.verificationStartedById === ownerId),
    )) {
      const now = Date.now();
      const {
        attemptId: _attemptId,
        verificationStartedById: _verificationStartedById,
        verificationStartedAt: _verificationStartedAt,
        ...taskWithoutAttempt
      } = task;
      const nextTask: SwarmTask = {
        ...taskWithoutAttempt,
        revision: task.revision + 1,
        status: "needs_attention",
        updatedAt: now,
      };
      const attempt = task.attemptId
        ? team.attempts.find((candidate) => candidate.id === task.attemptId)
        : undefined;
      if (attempt && (attempt.status === "active" || attempt.status === "verifying")) {
        this.journal.append(team.id, {
          type: "attempt/ended",
          data: {
            task: nextTask,
            attempt: {
              ...attempt,
              revision: attempt.revision + 1,
              status: "interrupted",
              endedAt: now,
              wallMs: Math.max(0, now - attempt.startedAt),
              lastProgressAt: now,
              terminalReason: "Host revoked the active attempt",
              actors: attempt.actors.map((actor) =>
                actor.endedAt === undefined ? { ...actor, endedAt: now } : actor,
              ),
            },
          },
        });
      } else {
        this.journal.append(team.id, { type: "task/updated", data: nextTask });
      }
    }
  }

  private async kick(teamId: string): Promise<void> {
    const previous = this.scheduling.get(teamId) ?? Promise.resolve();
    let current: Promise<void>;
    current = previous
      .catch(() => undefined)
      .then(() => this.schedule(teamId))
      .finally(() => {
        if (this.scheduling.get(teamId) === current) this.scheduling.delete(teamId);
      });
    this.scheduling.set(teamId, current);
    await current;
  }

  private async schedule(teamId: string): Promise<void> {
    const team = this.journal.get(teamId);
    if (!team || team.phase === "archived") return;
    const completed = new Set(
      team.tasks.filter((task) => task.status === "completed").map((task) => task.id),
    );
    let hasWriter = team.tasks.some(
      (task) => task.kind === "write" && task.status === "in_progress",
    );
    const occupied = new Set(
      team.tasks
        .filter((task) => task.status === "in_progress" && task.ownerId)
        .map((task) => task.ownerId as string),
    );
    const available = team.tasks.filter((task) => task.status === "pending");

    for (const member of team.members.filter(
      (candidate) =>
        candidate.phase === "active" &&
        (canExecuteTask(candidate.role) || available.some((task) => task.ownerId === candidate.id)),
    )) {
      if (occupied.has(member.id) || !this.runtime.getAgent(member.id)) continue;
      const candidates = available.filter(
        (task) =>
          task.status === "pending" &&
          (task.ownerId === member.id ||
            (canExecuteTask(member.role) &&
              member.role !== "lead" &&
              task.ownerId === undefined)) &&
          task.blockedBy.every((dependency) => completed.has(dependency)) &&
          !(task.kind === "write" && hasUnresolvedEffect(team, task.id)),
      );
      const task = candidates.find((candidate) => candidate.kind !== "write" || !hasWriter);
      if (!task) continue;
      available.splice(
        available.findIndex((candidate) => candidate.id === task.id),
        1,
      );
      const now = Date.now();
      const active: SwarmTask = {
        ...task,
        attemptId: randomUUID(),
        ownerId: member.id,
        revision: task.revision + 1,
        status: "in_progress",
        updatedAt: now,
      };
      const usage = emptyUsage();
      const attempt: SwarmAttempt = {
        id: active.attemptId as string,
        revision: 1,
        taskId: active.id,
        taskRevision: active.revision,
        ownerId: member.id,
        memberName: member.name,
        role: member.role,
        modelPolicy: member.modelPolicy,
        ...(member.budget ? { budget: member.budget } : {}),
        budgetState:
          member.budget?.maxInputTokens !== undefined ||
          member.budget?.maxOutputTokens !== undefined
            ? "unknown"
            : "within",
        status: "active",
        usage,
        actors: [
          {
            phase: "implementation",
            memberName: member.name,
            role: member.role,
            modelPolicy: member.modelPolicy,
            ...(member.budget ? { budget: member.budget } : {}),
            usage,
            startedAt: now,
          },
        ],
        startedAt: now,
        lastProgressAt: now,
        warningCodes: [],
      };
      this.journal.append(team.id, { type: "attempt/started", data: { task: active, attempt } });
      occupied.add(member.id);
      if (active.kind === "write") hasWriter = true;
      const lead = this.runtime.getAgent(team.id);
      if (!lead) {
        await this.revokeAttempts(team.id, member.id);
        occupied.delete(member.id);
        if (active.kind === "write") hasWriter = false;
        continue;
      }
      try {
        if (member.id === team.id) {
          this.runtime.followupRoot(lead, formatAssignment(active), lead.id);
        } else {
          await this.runtime.followup(lead, member.id, formatAssignment(active), lead.id);
        }
      } catch {
        await this.revokeAttempts(team.id, member.id);
        occupied.delete(member.id);
        if (active.kind === "write") hasWriter = false;
      }
    }
  }
}
