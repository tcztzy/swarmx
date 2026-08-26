import { randomUUID } from "node:crypto";
import type { Agent } from "@deepseek-ai/dsh-agent";
import { SessionId } from "@deepseek-ai/dsh-session";
import {
  type AddSwarmMemberRequest,
  addSwarmMemberRequestSchema,
  type CreateSwarmRequest,
  type CreateSwarmTaskRequest,
  createSwarmRequestSchema,
  createSwarmTaskRequestSchema,
  type InterruptSwarmMemberRequest,
  interruptSwarmMemberRequestSchema,
  type ReassignSwarmTaskRequest,
  reassignSwarmTaskRequestSchema,
  type SendSwarmMessageRequest,
  type SwarmMember,
  type SwarmMessage,
  type SwarmSnapshot,
  type SwarmTask,
  type SwarmTeamState,
  sendSwarmMessageRequestSchema,
  swarmSnapshotSchema,
  type UpdateSwarmTaskRequest,
  updateSwarmTaskRequestSchema,
} from "./contracts.js";
import { SwarmError } from "./errors.js";
import type { SwarmJournal } from "./journal.js";

export interface StartSwarmMemberRequest {
  childId: string;
  name: string;
  description: string;
  prompt: string;
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
  status: "delivered" | "queued";
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
    "Report completion, failure, or release through the swarm tool with this exact revision and attempt.",
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
      role: "member",
    };
    this.journal.append(team.id, { type: "member/updated", data: member });
    let startedId: string | undefined;
    try {
      startedId = await this.runtime.startContinuable(agent, {
        childId: member.id,
        description: member.description,
        name: member.name,
        prompt: request.prompt,
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
    return swarmSnapshotSchema.parse({
      kind: team.phase,
      memberName: caller.name,
      members: team.members.map((member) => ({
        description: member.description,
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
      tasks: team.tasks.map((task) => ({
        blockedBy: task.blockedBy,
        description: task.description,
        id: task.id,
        kind: task.kind,
        ownerName: task.ownerId ? nameById.get(task.ownerId) : undefined,
        ready: task.blockedBy.every((dependency) => completed.has(dependency)),
        revision: task.revision,
        status: task.status,
        subject: task.subject,
        writeScopes: task.writeScopes,
        ...(task.attemptId && (caller.role === "lead" || task.ownerId === caller.id)
          ? { attemptId: task.attemptId }
          : {}),
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
    const request = parse(createSwarmTaskRequestSchema, input);
    if (team.tasks.length >= this.config.maxTasks) {
      throw new SwarmError("Swarm task limit reached", "SWARM_LIMIT");
    }
    if (request.kind === "read" && request.writeScopes.length > 0) {
      throw new SwarmError("Read tasks cannot declare write scopes", "SWARM_INVALID_REQUEST");
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
    const sequence = Math.max(0, ...team.tasks.map((task) => task.sequence)) + 1;
    const now = Date.now();
    const task: SwarmTask = {
      blockedBy: request.blockedBy,
      createdAt: now,
      description: request.description,
      id: `task-${sequence}`,
      kind: request.kind,
      ownerId: owner?.id,
      revision: 1,
      sequence,
      status: "pending",
      subject: request.subject,
      updatedAt: now,
      writeScopes: request.writeScopes,
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
    if (task.revision !== request.expectedRevision) {
      throw new SwarmError("Swarm task revision is stale", "SWARM_STALE_REVISION");
    }
    if (task.attemptId !== request.attemptId) {
      throw new SwarmError("Swarm task attempt is stale", "SWARM_STALE_ATTEMPT");
    }
    if (task.status !== "in_progress" || task.ownerId !== agent.id) {
      throw new SwarmError("Only the active task owner may update it", "SWARM_UNAUTHORIZED");
    }
    const status =
      request.action === "complete"
        ? "completed"
        : request.action === "fail"
          ? "failed"
          : "pending";
    const next = {
      ...withoutAttempt(task),
      ...(request.action === "release" ? { ownerId: undefined } : {}),
      revision: task.revision + 1,
      status,
      updatedAt: Date.now(),
    } satisfies SwarmTask;
    this.journal.append(team.id, { type: "task/updated", data: next });
    await this.kick(team.id);
    return this.task(agent, task.id);
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
    const target = team.members.find(
      (member) => member.name === request.target && member.phase === "active",
    );
    if (!target) throw new SwarmError("Swarm member not found", "SWARM_MEMBER_NOT_FOUND");

    if (task.status === "in_progress" && task.ownerId) {
      const oldOwner = this.runtime.getAgent(task.ownerId);
      if (oldOwner) {
        this.runtime.interrupt(agent, oldOwner.id);
        await waitForIdle(oldOwner, this.config.quiescenceTimeoutMs);
      }
      const current = this.findTask(this.requireLead(agent), task.id);
      if (current.revision !== request.expectedRevision || current.attemptId !== task.attemptId) {
        throw new SwarmError("Swarm task changed during reassignment", "SWARM_STALE_REVISION");
      }
    }
    const next = {
      ...withoutAttempt(task),
      ownerId: target.id,
      revision: task.revision + 1,
      status: "pending",
      updatedAt: Date.now(),
    } satisfies SwarmTask;
    this.journal.append(team.id, { type: "task/updated", data: next });
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
      id: randomUUID(),
      senderId: agent.id,
      senderName: sender.name,
      sequence: Math.max(0, ...team.messages.map((candidate) => candidate.sequence)) + 1,
      targetId: target.id,
    };
    this.journal.append(team.id, { type: "message/queued", data: message });
    const delivered = await this.deliver(team, message, signal);
    return { id: message.id, status: delivered ? "delivered" : "queued" };
  }

  async recoverMember(agent: Agent): Promise<number> {
    const team = this.requireMember(agent);
    if (team.phase === "archived") return 0;
    let delivered = 0;
    for (const message of team.messages
      .filter((candidate) => candidate.targetId === agent.id && candidate.deliveredAt === undefined)
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
      (member) => member.name === request.target && member.role === "member",
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
    for (const member of team.members.filter((candidate) => candidate.role === "member")) {
      const child = this.runtime.getAgent(member.id);
      if (!child) continue;
      this.runtime.interrupt(agent, child.id);
      await waitForIdle(child, this.config.quiescenceTimeoutMs);
    }
    for (const member of team.members) await this.revokeAttempts(team.id, member.id);
    for (const member of team.members.filter((candidate) => candidate.role === "member")) {
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

  isMemberIdentity(sessionId: string): boolean {
    const team = this.journal.findByParticipant(sessionId);
    return Boolean(
      team?.phase === "active" &&
        team.members.some(
          (member) =>
            member.id === sessionId &&
            member.role === "member" &&
            (member.phase === "active" || member.phase === "provisioning"),
        ),
    );
  }

  private requireExact(agent: Agent): void {
    if (!this.runtime.exact(agent)) {
      throw new SwarmError("Agent authority is not exact", "SWARM_UNAUTHORIZED");
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

  private async deliver(
    team: SwarmTeamState,
    message: SwarmMessage,
    signal?: AbortSignal,
  ): Promise<boolean> {
    const target = this.runtime.getAgent(message.targetId);
    if (!target) return false;
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
      (candidate) => candidate.status === "in_progress" && candidate.ownerId === ownerId,
    )) {
      this.journal.append(team.id, {
        type: "task/updated",
        data: {
          ...withoutAttempt(task),
          revision: task.revision + 1,
          status: "needs_attention",
          updatedAt: Date.now(),
        },
      });
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
      (candidate) => candidate.role === "member" && candidate.phase === "active",
    )) {
      if (occupied.has(member.id) || !this.runtime.getAgent(member.id)) continue;
      const candidates = available.filter(
        (task) =>
          task.status === "pending" &&
          (task.ownerId === member.id || task.ownerId === undefined) &&
          task.blockedBy.every((dependency) => completed.has(dependency)),
      );
      const task = candidates.find((candidate) => candidate.kind === "read" || !hasWriter);
      if (!task) continue;
      available.splice(
        available.findIndex((candidate) => candidate.id === task.id),
        1,
      );
      const active: SwarmTask = {
        ...task,
        attemptId: randomUUID(),
        ownerId: member.id,
        revision: task.revision + 1,
        status: "in_progress",
        updatedAt: Date.now(),
      };
      this.journal.append(team.id, { type: "task/updated", data: active });
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
        await this.runtime.followup(lead, member.id, formatAssignment(active), lead.id);
      } catch {
        await this.revokeAttempts(team.id, member.id);
        occupied.delete(member.id);
        if (active.kind === "write") hasWriter = false;
      }
    }
  }
}
