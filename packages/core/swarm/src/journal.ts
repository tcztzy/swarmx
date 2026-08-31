import { createHash, randomBytes } from "node:crypto";
import { chmodSync, existsSync, mkdirSync, realpathSync } from "node:fs";
import { isAbsolute, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { z } from "zod";
import {
  knowledgeCommitReceiptSchema,
  legacySwarmKnowledgeAdmissionSchema,
  type MonitorFinding,
  monitorFindingSchema,
  type ResolveSwarmEffectRequest,
  SWARM_PROVISIONING_INTERRUPTED_ERROR,
  type SwarmAttempt,
  type SwarmEffect,
  type SwarmKnowledgeAdmission,
  type SwarmMember,
  type SwarmMessage,
  type SwarmTask,
  type SwarmTeamState,
  swarmAttemptSchema,
  swarmEffectSchema,
  swarmKnowledgeAdmissionSchema,
  swarmMemberSchema,
  swarmMessageSchema,
  swarmTaskSchema,
  swarmTeamStateSchema,
} from "./contracts.js";
import { SwarmError } from "./errors.js";

const DATABASE_NAME = "swarm.sqlite";
const MIGRATION_VERSION = 5;
const LEGACY_EVENT_CUTOFF_SETTING = "swarm_v5_legacy_event_cutoff";
const workspaceKeySchema = z.string().regex(/^swarmx--[0-9a-f]{64}$/u);
const memberBindingSchema = z.strictObject({
  workspaceKey: workspaceKeySchema,
  runtime: z.string().min(1).max(64),
  memberId: z.string().uuid(),
  handle: z.string().min(1).max(2_048),
});

const createdEventDataSchema = z.strictObject({
  createdAt: z.number().int().nonnegative(),
  lead: swarmMemberSchema,
  name: z.string().trim().min(1).max(100),
  workspaceKey: workspaceKeySchema,
});
const archivedEventDataSchema = z.strictObject({
  archivedAt: z.number().int().nonnegative(),
});
const archiveStartedEventDataSchema = z.strictObject({
  archiveStartedAt: z.number().int().nonnegative(),
});
const deliveredEventDataSchema = z.strictObject({
  messageId: z.string().min(1).max(200),
  deliveredAt: z.number().int().nonnegative(),
});
const deliveryStartedEventDataSchema = z.strictObject({
  messageId: z.string().min(1).max(200),
  deliveryStartedAt: z.number().int().nonnegative(),
});
const attemptTransitionEventDataSchema = z.strictObject({
  task: swarmTaskSchema,
  attempt: swarmAttemptSchema,
});

export type SwarmEvent =
  | { type: "team/created"; data: z.infer<typeof createdEventDataSchema> }
  | { type: "team/archive-started"; data: z.infer<typeof archiveStartedEventDataSchema> }
  | { type: "team/archived"; data: z.infer<typeof archivedEventDataSchema> }
  | { type: "member/updated"; data: SwarmMember }
  | { type: "task/updated"; data: SwarmTask }
  | { type: "effect/updated"; data: SwarmEffect }
  | { type: "knowledge/admission-updated"; data: SwarmKnowledgeAdmission }
  | { type: "message/queued"; data: SwarmMessage }
  | { type: "message/delivery-started"; data: z.infer<typeof deliveryStartedEventDataSchema> }
  | { type: "message/delivered"; data: z.infer<typeof deliveredEventDataSchema> }
  | { type: "attempt/started"; data: z.infer<typeof attemptTransitionEventDataSchema> }
  | { type: "attempt/usage-recorded"; data: SwarmAttempt }
  | { type: "attempt/budget-warning"; data: SwarmAttempt }
  | { type: "task/submitted"; data: z.infer<typeof attemptTransitionEventDataSchema> }
  | { type: "verification/started"; data: z.infer<typeof attemptTransitionEventDataSchema> }
  | { type: "verification/recorded"; data: z.infer<typeof attemptTransitionEventDataSchema> }
  | { type: "task/escalated"; data: z.infer<typeof attemptTransitionEventDataSchema> }
  | { type: "attempt/ended"; data: z.infer<typeof attemptTransitionEventDataSchema> }
  | { type: "monitor/finding-recorded"; data: MonitorFinding };

interface EventRow {
  seq: number;
  team_id: string;
  revision: number;
  type: SwarmEvent["type"];
  payload_json: string;
}

interface TeamRow {
  snapshot_json: string;
}

interface MemberBindingRow {
  workspace_key: string;
  runtime: string;
  member_id: string;
  handle: string;
}

interface MessageLedgerRow {
  snapshot_json: string;
}

export type SwarmMemberBinding = z.infer<typeof memberBindingSchema>;

export interface SwarmJournalOptions {
  readonly mode?: "client" | "owner";
}

export interface QueuedSwarmMessage {
  readonly created: boolean;
  readonly message: SwarmMessage;
}

export interface SettledKnowledgeAdmission {
  readonly admission: SwarmKnowledgeAdmission;
  readonly committed: boolean;
}

export type ProvisioningMemberBindingClaim = "created" | "existing" | "archive_required";

function clone<T>(value: T): T {
  return structuredClone(value);
}

function assertOpen(open: boolean): void {
  if (!open) throw new SwarmError("Swarm journal is closed", "SWARM_CLOSED");
}

function parseEvent(type: SwarmEvent["type"], value: unknown): SwarmEvent {
  switch (type) {
    case "team/created":
      return { type, data: createdEventDataSchema.parse(value) };
    case "team/archive-started":
      return { type, data: archiveStartedEventDataSchema.parse(value) };
    case "team/archived":
      return { type, data: archivedEventDataSchema.parse(value) };
    case "member/updated":
      return { type, data: swarmMemberSchema.parse(value) };
    case "task/updated":
      return { type, data: swarmTaskSchema.parse(value) };
    case "effect/updated":
      return { type, data: swarmEffectSchema.parse(value) };
    case "knowledge/admission-updated":
      return { type, data: swarmKnowledgeAdmissionSchema.parse(value) };
    case "message/queued":
      return { type, data: swarmMessageSchema.parse(value) };
    case "message/delivery-started":
      return { type, data: deliveryStartedEventDataSchema.parse(value) };
    case "message/delivered":
      return { type, data: deliveredEventDataSchema.parse(value) };
    case "attempt/started":
    case "task/submitted":
    case "verification/started":
    case "verification/recorded":
    case "task/escalated":
    case "attempt/ended":
      return { type, data: attemptTransitionEventDataSchema.parse(value) };
    case "attempt/usage-recorded":
    case "attempt/budget-warning":
      return { type, data: swarmAttemptSchema.parse(value) };
    case "monitor/finding-recorded":
      return { type, data: monitorFindingSchema.parse(value) };
  }
}

function parseLegacyEvent(type: SwarmEvent["type"], value: unknown): SwarmEvent {
  if (type !== "knowledge/admission-updated") return parseEvent(type, value);
  const admission = legacySwarmKnowledgeAdmissionSchema.parse(value);
  const receiptMatches =
    admission.receipt === undefined || admission.receipt.kind === admission.targetKind;
  if (receiptMatches && (admission.status !== "committed" || admission.receipt !== undefined)) {
    return { type, data: swarmKnowledgeAdmissionSchema.parse(admission) };
  }
  const { receipt: _receipt, ...rest } = admission;
  return {
    type,
    data: swarmKnowledgeAdmissionSchema.parse({
      ...rest,
      ...(admission.status === "committed" ? { status: "uncertain" } : {}),
    }),
  };
}

function eventTimestamp(event: SwarmEvent): number {
  switch (event.type) {
    case "team/created":
      return event.data.createdAt;
    case "team/archive-started":
      return event.data.archiveStartedAt;
    case "team/archived":
      return event.data.archivedAt;
    case "member/updated":
      return event.data.createdAt;
    case "task/updated":
      return event.data.updatedAt;
    case "effect/updated":
      return event.data.updatedAt;
    case "knowledge/admission-updated":
      return event.data.updatedAt;
    case "message/queued":
      return event.data.createdAt;
    case "message/delivery-started":
      return event.data.deliveryStartedAt;
    case "message/delivered":
      return event.data.deliveredAt;
    case "attempt/started":
    case "task/submitted":
    case "verification/started":
    case "verification/recorded":
    case "task/escalated":
    case "attempt/ended":
      return event.data.task.updatedAt;
    case "attempt/usage-recorded":
    case "attempt/budget-warning":
      return event.data.lastProgressAt;
    case "monitor/finding-recorded":
      return event.data.recordedAt;
  }
}

function replaceMember(state: SwarmTeamState, member: SwarmMember): SwarmMember[] {
  const existing = state.members.find((candidate) => candidate.id === member.id);
  const nameOwner = state.members.find((candidate) => candidate.name === member.name);
  if (nameOwner && nameOwner.id !== member.id) {
    throw new SwarmError(
      `Swarm member name already exists: ${member.name}`,
      "SWARM_INVALID_REQUEST",
    );
  }
  if (
    existing &&
    (existing.name !== member.name ||
      existing.role !== member.role ||
      existing.createdAt !== member.createdAt ||
      JSON.stringify(existing.modelPolicy) !== JSON.stringify(member.modelPolicy) ||
      JSON.stringify(existing.budget) !== JSON.stringify(member.budget))
  ) {
    throw new SwarmError("Swarm member identity is immutable", "SWARM_INVALID_REQUEST");
  }
  return existing
    ? state.members.map((candidate) => (candidate.id === member.id ? member : candidate))
    : [...state.members, member];
}

function replaceTask(state: SwarmTeamState, task: SwarmTask): SwarmTask[] {
  const existing = state.tasks.find((candidate) => candidate.id === task.id);
  const sequenceOwner = state.tasks.find((candidate) => candidate.sequence === task.sequence);
  if (sequenceOwner && sequenceOwner.id !== task.id) {
    throw new SwarmError("Swarm task sequence already exists", "SWARM_INVALID_REQUEST");
  }
  if (!existing && task.revision !== 1) {
    throw new SwarmError("A new swarm task must start at revision 1", "SWARM_STALE_REVISION");
  }
  if (existing) {
    if (existing.sequence !== task.sequence || existing.createdAt !== task.createdAt) {
      throw new SwarmError("Swarm task identity is immutable", "SWARM_INVALID_REQUEST");
    }
    if (task.revision !== existing.revision + 1) {
      throw new SwarmError("Swarm task revision is stale", "SWARM_STALE_REVISION");
    }
  }
  return existing
    ? state.tasks.map((candidate) => (candidate.id === task.id ? task : candidate))
    : [...state.tasks, task];
}

function replaceAttempt(state: SwarmTeamState, attempt: SwarmAttempt): SwarmAttempt[] {
  const existing = state.attempts.find((candidate) => candidate.id === attempt.id);
  if (!existing && attempt.revision !== 1) {
    throw new SwarmError("A new swarm attempt must start at revision 1", "SWARM_STALE_REVISION");
  }
  if (existing) {
    if (
      existing.taskId !== attempt.taskId ||
      existing.taskRevision !== attempt.taskRevision ||
      existing.ownerId !== attempt.ownerId ||
      existing.memberName !== attempt.memberName ||
      existing.role !== attempt.role ||
      existing.startedAt !== attempt.startedAt ||
      JSON.stringify(existing.modelPolicy) !== JSON.stringify(attempt.modelPolicy) ||
      JSON.stringify(existing.budget) !== JSON.stringify(attempt.budget)
    ) {
      throw new SwarmError("Swarm attempt identity is immutable", "SWARM_INVALID_REQUEST");
    }
    if (attempt.revision !== existing.revision + 1) {
      throw new SwarmError("Swarm attempt revision is stale", "SWARM_STALE_REVISION");
    }
  }
  return existing
    ? state.attempts.map((candidate) => (candidate.id === attempt.id ? attempt : candidate))
    : [...state.attempts, attempt];
}

function appendFinding(state: SwarmTeamState, finding: MonitorFinding): MonitorFinding[] {
  if (state.findings.some((candidate) => candidate.dedupeKey === finding.dedupeKey)) {
    throw new SwarmError("Swarm monitor finding already exists", "SWARM_INVALID_REQUEST");
  }
  const findings = [...state.findings, finding];
  return findings.length > 2_048 ? findings.slice(-2_048) : findings;
}

function replaceEffect(state: SwarmTeamState, effect: SwarmEffect): SwarmEffect[] {
  const existing = state.effects.find((candidate) => candidate.id === effect.id);
  const callOwner = state.effects.find(
    (candidate) => candidate.attemptId === effect.attemptId && candidate.callId === effect.callId,
  );
  if (callOwner && callOwner.id !== effect.id) {
    throw new SwarmError("Swarm effect call already exists", "SWARM_DUPLICATE_EFFECT");
  }
  if (!existing && effect.revision !== 1) {
    throw new SwarmError("A new swarm effect must start at revision 1", "SWARM_STALE_REVISION");
  }
  if (existing) {
    if (
      existing.callId !== effect.callId ||
      existing.taskId !== effect.taskId ||
      existing.attemptId !== effect.attemptId ||
      existing.ownerId !== effect.ownerId ||
      existing.toolName !== effect.toolName ||
      existing.createdAt !== effect.createdAt
    ) {
      throw new SwarmError("Swarm effect identity is immutable", "SWARM_INVALID_REQUEST");
    }
    if (effect.revision !== existing.revision + 1) {
      throw new SwarmError("Swarm effect revision is stale", "SWARM_STALE_REVISION");
    }
  }
  return existing
    ? state.effects.map((candidate) => (candidate.id === effect.id ? effect : candidate))
    : [...state.effects, effect];
}

function replaceAdmission(
  state: SwarmTeamState,
  admission: SwarmKnowledgeAdmission,
): SwarmKnowledgeAdmission[] {
  const existing = state.admissions.find((candidate) => candidate.id === admission.id);
  if (!existing && admission.revision !== 1) {
    throw new SwarmError(
      "A new knowledge admission must start at revision 1",
      "SWARM_STALE_REVISION",
    );
  }
  if (existing) {
    if (
      existing.requestHash !== admission.requestHash ||
      existing.taskId !== admission.taskId ||
      existing.targetKind !== admission.targetKind ||
      existing.createdAt !== admission.createdAt
    ) {
      throw new SwarmError("Knowledge admission identity is immutable", "SWARM_ADMISSION_CONFLICT");
    }
    if (admission.revision !== existing.revision + 1) {
      throw new SwarmError("Knowledge admission revision is stale", "SWARM_STALE_REVISION");
    }
  }
  return existing
    ? state.admissions.map((candidate) => (candidate.id === admission.id ? admission : candidate))
    : [...state.admissions, admission];
}

function queueMessage(state: SwarmTeamState, message: SwarmMessage): SwarmMessage[] {
  const existing = state.messages.find((candidate) => candidate.id === message.id);
  if (existing) {
    if (
      existing.senderId === message.senderId &&
      existing.targetId === message.targetId &&
      existing.delivery === message.delivery &&
      existing.content === message.content
    ) {
      return state.messages;
    }
    throw new SwarmError("Swarm message id already exists", "SWARM_MESSAGE_CONFLICT");
  }
  if (
    state.messages.some(
      (candidate) =>
        candidate.targetId === message.targetId && candidate.sequence === message.sequence,
    )
  ) {
    throw new SwarmError("Swarm message sequence already exists", "SWARM_INVALID_REQUEST");
  }
  const messages = [...state.messages, message];
  while (messages.length > 4_096) {
    const delivered = messages.findIndex((candidate) => candidate.deliveredAt !== undefined);
    if (delivered < 0) {
      throw new SwarmError("Swarm mailbox projection limit reached", "SWARM_LIMIT");
    }
    messages.splice(delivered, 1);
  }
  return messages;
}

function deliverMessage(
  state: SwarmTeamState,
  data: z.infer<typeof deliveredEventDataSchema>,
): SwarmMessage[] {
  const existing = state.messages.find((message) => message.id === data.messageId);
  if (!existing) {
    throw new SwarmError("Swarm message not found", "SWARM_INVALID_REQUEST");
  }
  if (existing.deliveredAt !== undefined) return state.messages;
  return state.messages.map((message) =>
    message.id === data.messageId ? { ...message, deliveredAt: data.deliveredAt } : message,
  );
}

function startMessageDelivery(
  state: SwarmTeamState,
  data: z.infer<typeof deliveryStartedEventDataSchema>,
): SwarmMessage[] {
  const existing = state.messages.find((message) => message.id === data.messageId);
  if (!existing) {
    throw new SwarmError("Swarm message not found", "SWARM_INVALID_REQUEST");
  }
  if (existing.deliveryStartedAt !== undefined) return state.messages;
  return state.messages.map((message) =>
    message.id === data.messageId
      ? { ...message, deliveryStartedAt: data.deliveryStartedAt }
      : message,
  );
}

const NON_TERMINAL_TASK_STATUSES = new Set(["in_progress", "submitted", "verifying"]);
const NON_TERMINAL_ATTEMPT_STATUSES = new Set(["active", "submitted", "verifying"]);

function isArchiveCleanupEvent(current: SwarmTeamState, event: SwarmEvent): boolean {
  switch (event.type) {
    case "team/archived":
    case "message/delivered":
      return true;
    case "member/updated": {
      const member = current.members.find((candidate) => candidate.id === event.data.id);
      if (member === undefined) return false;
      if (event.data.phase !== "active" && event.data.phase !== "provisioning") return true;
      return (
        member.phase === "provisioning" &&
        member.runtimeReadyAt === undefined &&
        event.data.phase === "provisioning" &&
        event.data.runtimeReadyAt !== undefined &&
        event.data.name === member.name &&
        event.data.role === member.role &&
        event.data.description === member.description &&
        event.data.createdAt === member.createdAt &&
        event.data.error === member.error &&
        JSON.stringify(event.data.modelPolicy) === JSON.stringify(member.modelPolicy) &&
        JSON.stringify(event.data.budget) === JSON.stringify(member.budget)
      );
    }
    case "task/updated":
      return (
        current.tasks.some((task) => task.id === event.data.id) &&
        !NON_TERMINAL_TASK_STATUSES.has(event.data.status)
      );
    case "attempt/ended":
      return (
        !NON_TERMINAL_TASK_STATUSES.has(event.data.task.status) &&
        !NON_TERMINAL_ATTEMPT_STATUSES.has(event.data.attempt.status)
      );
    case "effect/updated":
      return (
        current.effects.some((effect) => effect.id === event.data.id) &&
        event.data.status !== "started"
      );
    case "knowledge/admission-updated":
      return (
        current.admissions.some((admission) => admission.id === event.data.id) &&
        event.data.status !== "started"
      );
    default:
      return false;
  }
}

function applyEvent(
  teamId: string,
  current: SwarmTeamState | undefined,
  revision: number,
  event: SwarmEvent,
): SwarmTeamState {
  if (event.type === "team/created") {
    if (current) throw new SwarmError("Swarm already exists", "SWARM_INVALID_REQUEST");
    if (revision !== 1 || event.data.lead.id !== teamId || event.data.lead.role !== "lead") {
      throw new SwarmError("Invalid swarm lead identity", "SWARM_INVALID_REQUEST");
    }
    return swarmTeamStateSchema.parse({
      id: teamId,
      revision,
      name: event.data.name,
      workspaceKey: event.data.workspaceKey,
      phase: "active",
      createdAt: event.data.createdAt,
      updatedAt: event.data.createdAt,
      members: [event.data.lead],
      tasks: [],
      messages: [],
      effects: [],
      admissions: [],
      attempts: [],
      findings: [],
    });
  }

  if (!current) throw new SwarmError("Swarm not found", "SWARM_NOT_FOUND");
  if (revision !== current.revision + 1) {
    throw new SwarmError("Swarm revision is stale", "SWARM_STALE_REVISION");
  }
  if (current.phase === "archived") {
    throw new SwarmError("Swarm is archived", "SWARM_ARCHIVED");
  }
  if (
    current.archiveStartedAt !== undefined &&
    event.type !== "team/archive-started" &&
    !isArchiveCleanupEvent(current, event)
  ) {
    throw new SwarmError("Swarm archive is in progress", "SWARM_ARCHIVED");
  }

  const updatedAt = eventTimestamp(event);
  switch (event.type) {
    case "team/archive-started":
      if (current.archiveStartedAt !== undefined) {
        throw new SwarmError("Swarm archive is already in progress", "SWARM_INVALID_REQUEST");
      }
      return swarmTeamStateSchema.parse({
        ...current,
        revision,
        archiveStartedAt: event.data.archiveStartedAt,
        updatedAt,
      });
    case "team/archived": {
      if (
        current.archiveStartedAt === undefined ||
        current.members.some((member) => member.role !== "lead" && member.phase !== "retired") ||
        current.tasks.some((task) => NON_TERMINAL_TASK_STATUSES.has(task.status)) ||
        current.attempts.some((attempt) => NON_TERMINAL_ATTEMPT_STATUSES.has(attempt.status)) ||
        current.effects.some((effect) => effect.status === "started") ||
        current.admissions.some((admission) => admission.status === "started")
      ) {
        throw new SwarmError("Swarm archive has not drained", "SWARM_INVALID_REQUEST");
      }
      const { archiveStartedAt: _archiveStartedAt, ...rest } = current;
      return swarmTeamStateSchema.parse({
        ...rest,
        revision,
        phase: "archived",
        archivedAt: event.data.archivedAt,
        updatedAt,
      });
    }
    case "member/updated":
      return swarmTeamStateSchema.parse({
        ...current,
        revision,
        members: replaceMember(current, event.data),
        updatedAt,
      });
    case "task/updated":
      return swarmTeamStateSchema.parse({
        ...current,
        revision,
        tasks: replaceTask(current, event.data),
        updatedAt,
      });
    case "effect/updated":
      return swarmTeamStateSchema.parse({
        ...current,
        revision,
        effects: replaceEffect(current, event.data),
        updatedAt,
      });
    case "knowledge/admission-updated":
      return swarmTeamStateSchema.parse({
        ...current,
        revision,
        admissions: replaceAdmission(current, event.data),
        updatedAt,
      });
    case "message/queued":
      return {
        ...current,
        revision,
        messages: queueMessage(current, event.data),
        updatedAt,
      };
    case "message/delivery-started":
      return {
        ...current,
        revision,
        messages: startMessageDelivery(current, event.data),
        updatedAt,
      };
    case "message/delivered":
      return {
        ...current,
        revision,
        messages: deliverMessage(current, event.data),
        updatedAt,
      };
    case "attempt/started":
    case "task/submitted":
    case "verification/started":
    case "verification/recorded":
    case "task/escalated":
    case "attempt/ended": {
      const tasks = replaceTask(current, event.data.task);
      return swarmTeamStateSchema.parse({
        ...current,
        revision,
        tasks,
        attempts: replaceAttempt({ ...current, tasks }, event.data.attempt),
        updatedAt,
      });
    }
    case "attempt/usage-recorded":
    case "attempt/budget-warning":
      return swarmTeamStateSchema.parse({
        ...current,
        revision,
        attempts: replaceAttempt(current, event.data),
        updatedAt,
      });
    case "monitor/finding-recorded":
      return swarmTeamStateSchema.parse({
        ...current,
        revision,
        findings: appendFinding(current, event.data),
        updatedAt,
      });
  }
}

function applyLegacyArchivedEvent(
  teamId: string,
  current: SwarmTeamState | undefined,
  revision: number,
  archivedAt: number,
): SwarmTeamState {
  if (current === undefined) throw new SwarmError("Swarm not found", "SWARM_NOT_FOUND");
  if (revision !== current.revision + 1) {
    throw new SwarmError("Swarm revision is stale", "SWARM_STALE_REVISION");
  }
  if (current.phase === "archived") {
    throw new SwarmError("Swarm is archived", "SWARM_ARCHIVED");
  }
  const { archiveStartedAt: _archiveStartedAt, ...rest } = current;
  return swarmTeamStateSchema.parse({
    ...rest,
    id: teamId,
    revision,
    phase: "archived",
    archivedAt,
    updatedAt: archivedAt,
  });
}

function advanceIgnoredLegacyMessageEvent(
  current: SwarmTeamState | undefined,
  revision: number,
  event: SwarmEvent,
): SwarmTeamState {
  if (current === undefined) throw new SwarmError("Swarm not found", "SWARM_NOT_FOUND");
  if (revision !== current.revision + 1) {
    throw new SwarmError("Swarm revision is stale", "SWARM_STALE_REVISION");
  }
  if (current.phase === "archived") {
    throw new SwarmError("Swarm is archived", "SWARM_ARCHIVED");
  }
  if (current.archiveStartedAt !== undefined && event.type === "message/queued") {
    throw new SwarmError("Swarm archive is in progress", "SWARM_ARCHIVED");
  }
  return { ...current, revision, updatedAt: eventTimestamp(event) };
}

/** Owner-only append-only event log with rebuildable swarm projections. */
export class SwarmJournal {
  readonly databasePath: string;

  private readonly database: DatabaseSync;
  private open = true;

  constructor(root: string, options: SwarmJournalOptions = {}) {
    if (!isAbsolute(root)) {
      throw new SwarmError("Swarm storage root must be absolute", "SWARM_INVALID_REQUEST");
    }
    const mode = options.mode ?? "owner";
    if (mode === "owner") {
      mkdirSync(root, { recursive: true, mode: 0o700 });
      chmodSync(root, 0o700);
    }
    this.databasePath = join(root, DATABASE_NAME);
    if (mode === "client" && !existsSync(this.databasePath)) {
      throw new SwarmError(
        "Swarm storage has not been initialized by the platform owner",
        "SWARM_CLOSED",
      );
    }
    this.database = new DatabaseSync(this.databasePath);
    try {
      if (mode === "owner") chmodSync(this.databasePath, 0o600);
      this.database.exec(`
        PRAGMA foreign_keys = ON;
        PRAGMA synchronous = FULL;
        PRAGMA busy_timeout = 5000;
      `);
      if (mode === "owner") {
        this.database.exec("PRAGMA journal_mode = WAL");
        this.transaction(() => {
          this.migrate();
          this.ensureWorkspaceSalt();
          this.rebuildProjectionsInTransaction();
        });
      } else {
        this.verifyClientStorage();
      }
    } catch (cause) {
      this.database.close();
      this.open = false;
      if (mode === "client") {
        throw new SwarmError("Swarm storage is not ready for an auxiliary client", "SWARM_CLOSED", {
          cause,
        });
      }
      throw cause;
    }
  }

  append(teamId: string, input: SwarmEvent): SwarmTeamState {
    assertOpen(this.open);
    const event = parseEvent(input.type, input.data);
    return this.transaction(() => clone(this.appendEvent(teamId, this.readTeam(teamId), event)));
  }

  reserveMember(
    teamId: string,
    input: SwarmMember,
    maxMembers: number,
    authorize?: () => boolean,
  ): SwarmTeamState {
    assertOpen(this.open);
    const member = swarmMemberSchema.parse(input);
    const limit = z.number().int().positive().max(64).parse(maxMembers);
    if (member.role === "lead" || member.phase !== "provisioning") {
      throw new SwarmError("Invalid provisioning member", "SWARM_INVALID_REQUEST");
    }
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      if (current === undefined) throw new SwarmError("Swarm not found", "SWARM_NOT_FOUND");
      if (current.phase !== "active" || current.archiveStartedAt !== undefined) {
        throw new SwarmError("Swarm is archived", "SWARM_ARCHIVED");
      }
      if (authorize !== undefined && !authorize()) {
        throw new SwarmError("Swarm lead identity is stale", "SWARM_UNAUTHORIZED");
      }
      if (current.members.length >= limit) {
        throw new SwarmError("Swarm member limit reached", "SWARM_LIMIT");
      }
      return clone(this.appendEvent(teamId, current, { type: "member/updated", data: member }));
    });
  }

  activateProvisioningMember(
    teamId: string,
    input: SwarmMember,
    authorize?: () => boolean,
  ): SwarmTeamState {
    assertOpen(this.open);
    const member = swarmMemberSchema.parse(input);
    if (member.phase !== "active") {
      throw new SwarmError("Invalid active member", "SWARM_INVALID_REQUEST");
    }
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      const existing = current?.members.find((candidate) => candidate.id === member.id);
      if (
        current?.phase !== "active" ||
        current.archiveStartedAt !== undefined ||
        existing?.phase !== "provisioning"
      ) {
        throw new SwarmError("Swarm member provisioning is no longer active", "SWARM_ARCHIVED");
      }
      if (authorize !== undefined && !authorize()) {
        throw new SwarmError("Swarm lead identity is stale", "SWARM_UNAUTHORIZED");
      }
      return clone(this.appendEvent(teamId, current, { type: "member/updated", data: member }));
    });
  }

  acknowledgeProvisioningMember(teamId: string, memberId: string, readyAt: number): SwarmMember {
    assertOpen(this.open);
    const id = z.string().min(1).max(512).parse(memberId);
    const runtimeReadyAt = z.number().int().nonnegative().parse(readyAt);
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      const member = current?.members.find((candidate) => candidate.id === id);
      if (current?.phase !== "active" || member?.phase !== "provisioning") {
        throw new SwarmError("Swarm member is no longer provisioning", "SWARM_ARCHIVED");
      }
      if (member.runtimeReadyAt !== undefined) return clone(member);
      const acknowledged = swarmMemberSchema.parse({ ...member, runtimeReadyAt });
      this.appendEvent(teamId, current, { type: "member/updated", data: acknowledged });
      return clone(acknowledged);
    });
  }

  beginArchive(
    teamId: string,
    archiveStartedAt: number,
    authorize?: () => boolean,
  ): SwarmTeamState {
    assertOpen(this.open);
    const startedAt = z.number().int().nonnegative().parse(archiveStartedAt);
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      if (current === undefined) throw new SwarmError("Swarm not found", "SWARM_NOT_FOUND");
      if (current.phase === "archived") {
        throw new SwarmError("Swarm is archived", "SWARM_ARCHIVED");
      }
      if (authorize !== undefined && !authorize()) {
        throw new SwarmError("Swarm lead identity is stale", "SWARM_UNAUTHORIZED");
      }
      if (current.archiveStartedAt !== undefined) return clone(current);
      return clone(
        this.appendEvent(teamId, current, {
          type: "team/archive-started",
          data: { archiveStartedAt: startedAt },
        }),
      );
    });
  }

  settleProvisioningMemberWithoutBinding(teamId: string, memberId: string): boolean {
    assertOpen(this.open);
    const id = z.string().min(1).max(512).parse(memberId);
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      const member = current?.members.find((candidate) => candidate.id === id);
      if (
        current?.phase !== "active" ||
        current.archiveStartedAt === undefined ||
        member?.role === "lead" ||
        member?.phase !== "provisioning"
      ) {
        return false;
      }
      const binding = this.database
        .prepare("SELECT 1 FROM swarm_member_bindings WHERE member_id = ? LIMIT 1")
        .get(id);
      if (binding !== undefined) return false;
      this.appendEvent(teamId, current, {
        type: "member/updated",
        data: { ...member, phase: "retired" },
      });
      return true;
    });
  }

  retireMemberForArchive(teamId: string, memberId: string): boolean {
    assertOpen(this.open);
    const id = z.string().min(1).max(512).parse(memberId);
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      const member = current?.members.find((candidate) => candidate.id === id);
      if (
        current?.phase !== "active" ||
        current.archiveStartedAt === undefined ||
        member === undefined ||
        member.role === "lead"
      ) {
        return false;
      }
      if (member.phase === "retired") return true;
      if (member.phase === "provisioning") return false;
      const binding = this.database
        .prepare("SELECT 1 FROM swarm_member_bindings WHERE member_id = ? LIMIT 1")
        .get(id);
      if (binding !== undefined) return false;
      this.appendEvent(teamId, current, {
        type: "member/updated",
        data: { ...member, phase: "retired" },
      });
      return true;
    });
  }

  retireBoundMemberForArchive(teamId: string, input: SwarmMemberBinding): boolean {
    assertOpen(this.open);
    const binding = memberBindingSchema.parse(input);
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      const member = current?.members.find((candidate) => candidate.id === binding.memberId);
      if (
        current?.phase !== "active" ||
        current.archiveStartedAt === undefined ||
        current.workspaceKey !== binding.workspaceKey ||
        member === undefined ||
        member.role === "lead"
      ) {
        return false;
      }
      const owner = this.database
        .prepare(
          `SELECT workspace_key, runtime, member_id, handle
           FROM swarm_member_bindings
           WHERE workspace_key = ? AND runtime = ? AND member_id = ?`,
        )
        .get(binding.workspaceKey, binding.runtime, binding.memberId) as
        | MemberBindingRow
        | undefined;
      if (
        owner === undefined ||
        owner.handle !== binding.handle ||
        owner.runtime !== binding.runtime ||
        owner.workspace_key !== binding.workspaceKey
      ) {
        return false;
      }
      if (member.phase !== "retired") {
        this.appendEvent(teamId, current, {
          type: "member/updated",
          data: { ...member, phase: "retired" },
        });
      }
      const released = this.database
        .prepare(
          `DELETE FROM swarm_member_bindings
           WHERE workspace_key = ? AND runtime = ? AND member_id = ? AND handle = ?`,
        )
        .run(binding.workspaceKey, binding.runtime, binding.memberId, binding.handle);
      if (Number(released.changes) !== 1) {
        throw new SwarmError("Swarm member binding changed during archive", "SWARM_STALE_REVISION");
      }
      return true;
    });
  }

  finishArchive(teamId: string, archivedAt: number): SwarmTeamState | undefined {
    assertOpen(this.open);
    const completedAt = z.number().int().nonnegative().parse(archivedAt);
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      if (current === undefined) throw new SwarmError("Swarm not found", "SWARM_NOT_FOUND");
      if (current.phase === "archived") return clone(current);
      if (
        current.archiveStartedAt === undefined ||
        current.members.some((member) => member.role !== "lead" && member.phase !== "retired") ||
        current.tasks.some((task) => NON_TERMINAL_TASK_STATUSES.has(task.status)) ||
        current.attempts.some((attempt) => NON_TERMINAL_ATTEMPT_STATUSES.has(attempt.status)) ||
        current.effects.some((effect) => effect.status === "started") ||
        current.admissions.some((admission) => admission.status === "started")
      ) {
        return undefined;
      }
      for (const member of current.members.filter((candidate) => candidate.role !== "lead")) {
        const binding = this.database
          .prepare("SELECT 1 FROM swarm_member_bindings WHERE member_id = ? LIMIT 1")
          .get(member.id);
        if (binding !== undefined) return undefined;
      }
      return clone(
        this.appendEvent(teamId, current, {
          type: "team/archived",
          data: { archivedAt: completedAt },
        }),
      );
    });
  }

  settleArchiveIntents(teamId: string, now: number): SwarmTeamState {
    assertOpen(this.open);
    const settledAt = z.number().int().nonnegative().parse(now);
    return this.transaction(() => {
      let current = this.readTeam(teamId);
      if (current?.phase !== "active" || current.archiveStartedAt === undefined) {
        throw new SwarmError("Swarm archive is not in progress", "SWARM_INVALID_REQUEST");
      }
      for (const effect of current.effects.filter((candidate) => candidate.status === "started")) {
        current = this.appendEvent(teamId, current, {
          type: "effect/updated",
          data: {
            ...effect,
            revision: effect.revision + 1,
            status: "uncertain",
            updatedAt: settledAt,
          },
        });
      }
      for (const admission of current.admissions.filter(
        (candidate) => candidate.status === "started",
      )) {
        current = this.appendEvent(teamId, current, {
          type: "knowledge/admission-updated",
          data: {
            ...admission,
            revision: admission.revision + 1,
            status: "uncertain",
            updatedAt: settledAt,
          },
        });
      }
      return clone(current);
    });
  }

  tryStartAttempt(
    teamId: string,
    input: z.infer<typeof attemptTransitionEventDataSchema>,
    authorize?: () => boolean,
  ): boolean {
    assertOpen(this.open);
    const data = attemptTransitionEventDataSchema.parse(input);
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      if (
        current?.phase !== "active" ||
        current.archiveStartedAt !== undefined ||
        (authorize !== undefined && !authorize())
      ) {
        return false;
      }
      const task = current.tasks.find((candidate) => candidate.id === data.task.id);
      const member = current.members.find((candidate) => candidate.id === data.attempt.ownerId);
      if (
        task?.status !== "pending" ||
        task.revision + 1 !== data.task.revision ||
        data.task.status !== "in_progress" ||
        data.task.ownerId !== data.attempt.ownerId ||
        data.task.attemptId !== data.attempt.id ||
        data.attempt.taskId !== task.id ||
        data.attempt.taskRevision !== data.task.revision ||
        data.attempt.status !== "active" ||
        member?.phase !== "active" ||
        (task.ownerId !== undefined && task.ownerId !== member.id) ||
        current.tasks.some(
          (candidate) => candidate.status === "in_progress" && candidate.ownerId === member.id,
        ) ||
        task.blockedBy.some(
          (dependency) =>
            current.tasks.find((candidate) => candidate.id === dependency)?.status !== "completed",
        ) ||
        (task.kind === "write" &&
          current.tasks.some(
            (candidate) => candidate.kind === "write" && candidate.status === "in_progress",
          )) ||
        current.effects.some(
          (effect) =>
            effect.taskId === task.id &&
            (effect.status === "started" || effect.status === "uncertain"),
        )
      ) {
        return false;
      }
      this.appendEvent(teamId, current, { type: "attempt/started", data });
      return true;
    });
  }

  beginToolEffect(
    teamId: string,
    actorId: string,
    input: SwarmEffect,
    authorize?: () => boolean,
  ): SwarmEffect {
    assertOpen(this.open);
    const effect = swarmEffectSchema.parse(input);
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      const member = current?.members.find((candidate) => candidate.id === actorId);
      const task = current?.tasks.find((candidate) => candidate.id === effect.taskId);
      if (
        current?.phase !== "active" ||
        current.archiveStartedAt !== undefined ||
        member?.phase !== "active" ||
        (authorize !== undefined && !authorize()) ||
        task?.kind !== "write" ||
        task.status !== "in_progress" ||
        task.ownerId !== actorId ||
        task.attemptId !== effect.attemptId ||
        task.revision !== effect.taskRevision ||
        effect.ownerId !== actorId ||
        effect.status !== "started"
      ) {
        throw new SwarmError(
          "Workspace mutation requires an active write attempt",
          "SWARM_UNAUTHORIZED",
        );
      }
      if (
        current.effects.some(
          (candidate) =>
            candidate.attemptId === effect.attemptId && candidate.callId === effect.callId,
        )
      ) {
        throw new SwarmError(
          "This Tool call already entered the effect boundary",
          "SWARM_DUPLICATE_EFFECT",
        );
      }
      if (
        current.effects.some(
          (candidate) =>
            candidate.taskId === task.id &&
            (candidate.status === "started" || candidate.status === "uncertain"),
        )
      ) {
        throw new SwarmError(
          "Verify the previous uncertain Tool effect before another mutation",
          "SWARM_EFFECT_UNCERTAIN",
        );
      }
      this.appendEvent(teamId, current, { type: "effect/updated", data: effect });
      return clone(effect);
    });
  }

  settleToolEffect(
    teamId: string,
    actorId: string,
    effectId: string,
    outcome: { readonly status: "succeeded" | "uncertain"; readonly resultDigest?: string },
    authorize?: () => boolean,
  ): SwarmEffect {
    assertOpen(this.open);
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      const effect = current?.effects.find((candidate) => candidate.id === effectId);
      if (current === undefined || effect?.ownerId !== actorId || effect.status !== "started") {
        throw new SwarmError("Swarm Tool effect is not active", "SWARM_UNAUTHORIZED");
      }
      const task = current.tasks.find((candidate) => candidate.id === effect.taskId);
      const member = current.members.find((candidate) => candidate.id === actorId);
      const stillCurrent =
        current.phase === "active" &&
        current.archiveStartedAt === undefined &&
        member?.phase === "active" &&
        (authorize === undefined || authorize()) &&
        task?.status === "in_progress" &&
        task.ownerId === actorId &&
        task.attemptId === effect.attemptId &&
        task.revision === effect.taskRevision;
      const next = swarmEffectSchema.parse({
        ...effect,
        revision: effect.revision + 1,
        status: stillCurrent ? outcome.status : "uncertain",
        updatedAt: Date.now(),
        ...(stillCurrent && outcome.resultDigest !== undefined
          ? { resultDigest: outcome.resultDigest }
          : {}),
      });
      this.appendEvent(teamId, current, { type: "effect/updated", data: next });
      return clone(next);
    });
  }

  resolveToolEffect(
    teamId: string,
    actorId: string,
    request: ResolveSwarmEffectRequest,
    authorize?: () => boolean,
  ): SwarmEffect {
    assertOpen(this.open);
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      if (current === undefined) {
        throw new SwarmError("Swarm not found", "SWARM_NOT_FOUND");
      }
      const lead = current.members.find(
        (member) => member.id === actorId && member.role === "lead" && member.phase === "active",
      );
      if (
        current.id !== actorId ||
        lead === undefined ||
        (authorize !== undefined && !authorize())
      ) {
        throw new SwarmError("Swarm lead authority is stale", "SWARM_UNAUTHORIZED");
      }
      if (current.phase !== "active" || current.archiveStartedAt !== undefined) {
        throw new SwarmError("Swarm is archived", "SWARM_ARCHIVED");
      }
      const task = current.tasks.find((candidate) => candidate.id === request.taskId);
      if (task === undefined) {
        throw new SwarmError("Swarm task not found", "SWARM_TASK_NOT_FOUND");
      }
      if (task.revision !== request.expectedRevision) {
        throw new SwarmError("Swarm task revision is stale", "SWARM_STALE_REVISION");
      }
      const effect = current.effects.find((candidate) => candidate.id === request.effectId);
      if (
        effect === undefined ||
        effect.taskId !== task.id ||
        effect.attemptId !== request.attemptId ||
        effect.status !== "uncertain"
      ) {
        throw new SwarmError("Uncertain Swarm Tool effect not found", "SWARM_STALE_ATTEMPT");
      }
      const next = swarmEffectSchema.parse({
        ...effect,
        revision: effect.revision + 1,
        status: request.resolution,
        updatedAt: Date.now(),
        verification: request.verification,
      });
      this.appendEvent(teamId, current, { type: "effect/updated", data: next });
      return clone(next);
    });
  }

  beginKnowledgeAdmission(
    teamId: string,
    actorId: string,
    input: SwarmKnowledgeAdmission,
    authorize?: () => boolean,
  ): SwarmKnowledgeAdmission {
    assertOpen(this.open);
    const admission = swarmKnowledgeAdmissionSchema.parse(input);
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      const lead = current?.members.find(
        (member) => member.id === actorId && member.role === "lead",
      );
      const task = current?.tasks.find((candidate) => candidate.id === admission.taskId);
      const existing = current?.admissions.find((candidate) => candidate.id === admission.id);
      if (
        current?.phase !== "active" ||
        current.archiveStartedAt !== undefined ||
        current.id !== actorId ||
        lead?.phase !== "active" ||
        (authorize !== undefined && !authorize()) ||
        task?.kind !== "knowledge" ||
        task.status !== "in_progress" ||
        task.attemptId !== admission.attemptId ||
        task.revision !== admission.taskRevision ||
        admission.status !== "started" ||
        (existing !== undefined && existing.status !== "uncertain") ||
        current.admissions.some(
          (candidate) =>
            candidate.id !== admission.id &&
            candidate.taskId === admission.taskId &&
            ["started", "uncertain", "committed"].includes(candidate.status),
        )
      ) {
        throw new SwarmError(
          "Knowledge admission does not own the current task attempt",
          "SWARM_STALE_ATTEMPT",
        );
      }
      this.appendEvent(teamId, current, {
        type: "knowledge/admission-updated",
        data: admission,
      });
      return clone(admission);
    });
  }

  settleKnowledgeAdmissionUncertain(
    teamId: string,
    actorId: string,
    admissionId: string,
    authorize?: () => boolean,
  ): SwarmKnowledgeAdmission {
    assertOpen(this.open);
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      const admission = current?.admissions.find((candidate) => candidate.id === admissionId);
      if (
        current === undefined ||
        current.id !== actorId ||
        admission?.status !== "started" ||
        (authorize !== undefined && !authorize())
      ) {
        throw new SwarmError("Knowledge admission is not active", "SWARM_UNAUTHORIZED");
      }
      const uncertain = swarmKnowledgeAdmissionSchema.parse({
        ...admission,
        revision: admission.revision + 1,
        status: "uncertain",
        updatedAt: Date.now(),
      });
      this.appendEvent(teamId, current, {
        type: "knowledge/admission-updated",
        data: uncertain,
      });
      return clone(uncertain);
    });
  }

  commitKnowledgeAdmission(
    teamId: string,
    actorId: string,
    admissionId: string,
    receipt: NonNullable<SwarmKnowledgeAdmission["receipt"]>,
    authorize?: () => boolean,
  ): SettledKnowledgeAdmission {
    assertOpen(this.open);
    const ownerReceipt = knowledgeCommitReceiptSchema.parse(receipt);
    return this.transaction(() => {
      let current = this.readTeam(teamId);
      const admission = current?.admissions.find((candidate) => candidate.id === admissionId);
      if (current === undefined || current.id !== actorId || admission?.status !== "started") {
        throw new SwarmError("Knowledge admission is not active", "SWARM_UNAUTHORIZED");
      }
      const task = current.tasks.find((candidate) => candidate.id === admission.taskId);
      const stillCurrent =
        current.phase === "active" &&
        current.archiveStartedAt === undefined &&
        (authorize === undefined || authorize()) &&
        task?.kind === "knowledge" &&
        task.status === "in_progress" &&
        task.attemptId === admission.attemptId &&
        task.revision === admission.taskRevision;
      const settled = swarmKnowledgeAdmissionSchema.parse({
        ...admission,
        receipt: ownerReceipt,
        revision: admission.revision + 1,
        status: stillCurrent ? "committed" : "uncertain",
        updatedAt: Date.now(),
      });
      current = this.appendEvent(teamId, current, {
        type: "knowledge/admission-updated",
        data: settled,
      });
      if (!stillCurrent || task === undefined) {
        return { admission: clone(settled), committed: false };
      }
      const now = Date.now();
      const { attemptId: _attemptId, ...taskWithoutAttempt } = task;
      const completed = swarmTaskSchema.parse({
        ...taskWithoutAttempt,
        revision: task.revision + 1,
        status: "completed",
        updatedAt: now,
      });
      const attempt = current.attempts.find((candidate) => candidate.id === task.attemptId);
      if (attempt === undefined) {
        this.appendEvent(teamId, current, { type: "task/updated", data: completed });
      } else {
        this.appendEvent(teamId, current, {
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
      }
      return { admission: clone(settled), committed: true };
    });
  }

  queueMessage(
    teamId: string,
    input: Omit<SwarmMessage, "sequence">,
    maxPendingMessages: number,
    authorize?: () => boolean,
  ): QueuedSwarmMessage {
    assertOpen(this.open);
    const limit = z.number().int().positive().parse(maxPendingMessages);
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      if (current === undefined) {
        throw new SwarmError("Swarm not found", "SWARM_NOT_FOUND");
      }
      if (current.phase !== "active") {
        throw new SwarmError("Swarm is archived", "SWARM_ARCHIVED");
      }
      if (authorize !== undefined && !authorize()) {
        throw new SwarmError("Swarm actor identity is stale", "SWARM_UNAUTHORIZED");
      }
      const sender = current.members.find((member) => member.id === input.senderId);
      if (
        sender?.phase !== "active" ||
        sender.name !== input.senderName ||
        sender.role === "monitor"
      ) {
        throw new SwarmError("Swarm sender is unavailable", "SWARM_UNAUTHORIZED");
      }
      const existing = this.readMessageLedger(teamId, input.id);
      if (existing !== undefined) {
        if (
          existing.senderId !== input.senderId ||
          existing.targetId !== input.targetId ||
          existing.delivery !== input.delivery ||
          existing.content !== input.content
        ) {
          throw new SwarmError(
            "Swarm message idempotency key conflicts with another message",
            "SWARM_MESSAGE_CONFLICT",
          );
        }
        return { created: false, message: clone(existing) };
      }
      const target = current.members.find((member) => member.id === input.targetId);
      if (target?.phase !== "active") {
        throw new SwarmError("Swarm target is unavailable", "SWARM_MEMBER_NOT_FOUND");
      }
      if (target.id === sender.id) {
        throw new SwarmError("Cannot send a swarm message to yourself", "SWARM_INVALID_REQUEST");
      }
      const pending = current.messages.filter(
        (message) => message.targetId === input.targetId && message.deliveredAt === undefined,
      ).length;
      if (pending >= limit) throw new SwarmError("Swarm mailbox limit reached", "SWARM_LIMIT");
      const message = swarmMessageSchema.parse({
        ...input,
        sequence:
          Number(
            (
              this.database
                .prepare(
                  `SELECT MAX(sequence) AS sequence
                   FROM swarm_message_ledger
                   WHERE team_id = ? AND target_id = ?`,
                )
                .get(teamId, input.targetId) as { sequence: number | null }
            ).sequence ?? 0,
          ) + 1,
      });
      const event = parseEvent("message/queued", message);
      this.appendEvent(teamId, current, event);
      return { created: true, message: clone(message) };
    });
  }

  claimMessageDelivery(
    teamId: string,
    messageId: string,
    deliveryStartedAt: number,
    authorize?: () => boolean,
  ): boolean {
    assertOpen(this.open);
    const data = deliveryStartedEventDataSchema.parse({ messageId, deliveryStartedAt });
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      const existing = this.readMessageLedger(teamId, data.messageId);
      if (existing === undefined) {
        throw new SwarmError("Swarm message not found", "SWARM_INVALID_REQUEST");
      }
      if (existing.deliveryStartedAt !== undefined || existing.deliveredAt !== undefined) {
        return false;
      }
      if (current?.phase !== "active") {
        throw new SwarmError("Swarm is archived", "SWARM_ARCHIVED");
      }
      if (authorize !== undefined && !authorize()) {
        throw new SwarmError("Swarm target identity is stale", "SWARM_UNAUTHORIZED");
      }
      const target = current.members.find((member) => member.id === existing.targetId);
      if (target?.phase !== "active") {
        throw new SwarmError("Swarm target is unavailable", "SWARM_MEMBER_NOT_FOUND");
      }
      this.appendEvent(teamId, current, parseEvent("message/delivery-started", data));
      return true;
    });
  }

  get(teamId: string): SwarmTeamState | undefined {
    assertOpen(this.open);
    const state = this.readTeam(teamId);
    return state ? clone(state) : undefined;
  }

  list(): SwarmTeamState[] {
    assertOpen(this.open);
    const rows = this.database
      .prepare("SELECT snapshot_json FROM swarm_teams ORDER BY created_at, team_id")
      .all() as unknown as TeamRow[];
    return rows.map((row) => clone(swarmTeamStateSchema.parse(JSON.parse(row.snapshot_json))));
  }

  findByParticipant(sessionId: string): SwarmTeamState | undefined {
    return this.list().find((team) => team.members.some((member) => member.id === sessionId));
  }

  listMemberBindings(workspaceKey: string, runtime: string): SwarmMemberBinding[] {
    assertOpen(this.open);
    const scope = workspaceKeySchema.parse(workspaceKey);
    const runtimeName = z.string().min(1).max(64).parse(runtime);
    const rows = this.database
      .prepare(
        `SELECT workspace_key, runtime, member_id, handle
         FROM swarm_member_bindings
         WHERE workspace_key = ? AND runtime = ?
         ORDER BY member_id`,
      )
      .all(scope, runtimeName) as unknown as MemberBindingRow[];
    return rows.map((row) =>
      memberBindingSchema.parse({
        workspaceKey: row.workspace_key,
        runtime: row.runtime,
        memberId: row.member_id,
        handle: row.handle,
      }),
    );
  }

  claimMemberBinding(input: SwarmMemberBinding): "created" | "existing" {
    assertOpen(this.open);
    const binding = memberBindingSchema.parse(input);
    return this.transaction(() => this.claimMemberBindingRow(binding));
  }

  claimProvisioningMemberBinding(
    teamId: string,
    input: SwarmMemberBinding,
  ): ProvisioningMemberBindingClaim {
    assertOpen(this.open);
    const id = z.string().min(1).max(512).parse(teamId);
    const binding = memberBindingSchema.parse(input);
    return this.transaction(() => {
      const team = this.readTeam(id);
      const member = team?.members.find((candidate) => candidate.id === binding.memberId);
      if (
        team?.phase !== "active" ||
        team.workspaceKey !== binding.workspaceKey ||
        member?.role === "lead" ||
        member?.phase !== "provisioning"
      ) {
        throw new SwarmError(
          "Swarm member is not reserved for native provisioning",
          "SWARM_UNAUTHORIZED",
        );
      }
      const result = this.claimMemberBindingRow(binding);
      return team.archiveStartedAt === undefined ? result : "archive_required";
    });
  }

  releaseMemberBinding(input: SwarmMemberBinding): boolean {
    assertOpen(this.open);
    const binding = memberBindingSchema.parse(input);
    const result = this.database
      .prepare(
        `DELETE FROM swarm_member_bindings
         WHERE workspace_key = ? AND runtime = ? AND member_id = ? AND handle = ?`,
      )
      .run(binding.workspaceKey, binding.runtime, binding.memberId, binding.handle);
    return Number(result.changes) === 1;
  }

  workspaceKey(cwd: string | undefined): string {
    assertOpen(this.open);
    if (!cwd) throw new SwarmError("Agent workspace is unavailable", "SWARM_INVALID_REQUEST");
    let canonical: string;
    try {
      canonical = realpathSync.native(cwd);
    } catch (cause) {
      throw new SwarmError("Agent workspace cannot be resolved", "SWARM_INVALID_REQUEST", {
        cause,
      });
    }
    const row = this.database
      .prepare("SELECT value FROM swarm_settings WHERE key = 'workspace_salt'")
      .get() as { value: string } | undefined;
    if (!row) throw new SwarmError("Swarm workspace salt is unavailable", "SWARM_CLOSED");
    const digest = createHash("sha256")
      .update(row.value)
      .update("\0")
      .update(canonical)
      .digest("hex");
    return `swarmx--${digest}`;
  }

  rebuildProjections(): void {
    assertOpen(this.open);
    this.transaction(() => this.rebuildProjectionsInTransaction());
  }

  recoverInterruptedTasks(now: number, workspaceKey?: string): number {
    assertOpen(this.open);
    const interrupted = this.list().flatMap((team) =>
      team.phase === "active" && (workspaceKey === undefined || team.workspaceKey === workspaceKey)
        ? team.tasks
            .filter((task) => ["in_progress", "submitted", "verifying"].includes(task.status))
            .map((task) => ({
              teamId: team.id,
              task,
              attempt: task.attemptId
                ? team.attempts.find((candidate) => candidate.id === task.attemptId)
                : undefined,
            }))
        : [],
    );
    for (const { teamId, task, attempt } of interrupted) {
      const { attemptId: _attemptId, ...rest } = task;
      const recoveredTask = {
        ...rest,
        revision: task.revision + 1,
        status: "needs_attention" as const,
        updatedAt: now,
      };
      if (attempt && ["active", "submitted", "verifying"].includes(attempt.status)) {
        this.append(teamId, {
          type: "attempt/ended",
          data: {
            task: recoveredTask,
            attempt: {
              ...attempt,
              revision: attempt.revision + 1,
              status: "interrupted",
              endedAt: now,
              wallMs: Math.max(0, now - attempt.startedAt),
              terminalReason: "Host recovered a non-terminal attempt",
              lastProgressAt: now,
              actors: attempt.actors.map((actor) =>
                actor.endedAt === undefined ? { ...actor, endedAt: now } : actor,
              ),
            },
          },
        });
      } else {
        this.append(teamId, { type: "task/updated", data: recoveredTask });
      }
    }
    return interrupted.length;
  }

  recoverUncertainIntents(now: number, workspaceKey?: string): number {
    assertOpen(this.open);
    const started = this.list().flatMap((team) =>
      team.phase === "active" && (workspaceKey === undefined || team.workspaceKey === workspaceKey)
        ? [
            ...team.effects
              .filter((effect) => effect.status === "started")
              .map((effect) => ({ kind: "effect" as const, teamId: team.id, value: effect })),
            ...team.admissions
              .filter((admission) => admission.status === "started")
              .map((admission) => ({
                kind: "admission" as const,
                teamId: team.id,
                value: admission,
              })),
          ]
        : [],
    );
    for (const item of started) {
      if (item.kind === "effect") {
        this.append(item.teamId, {
          type: "effect/updated",
          data: {
            ...item.value,
            revision: item.value.revision + 1,
            status: "uncertain",
            updatedAt: now,
          },
        });
      } else {
        this.append(item.teamId, {
          type: "knowledge/admission-updated",
          data: {
            ...item.value,
            revision: item.value.revision + 1,
            status: "uncertain",
            updatedAt: now,
          },
        });
      }
    }
    return started.length;
  }

  recoverProvisioningMembers(
    error = SWARM_PROVISIONING_INTERRUPTED_ERROR,
    workspaceKey?: string,
  ): number {
    assertOpen(this.open);
    const interrupted = this.list().flatMap((team) =>
      team.phase === "active" &&
      team.archiveStartedAt === undefined &&
      (workspaceKey === undefined || team.workspaceKey === workspaceKey)
        ? team.members
            .filter((member) => member.phase === "provisioning")
            .map((member) => ({ teamId: team.id, member }))
        : [],
    );
    for (const { teamId, member } of interrupted) {
      this.append(teamId, {
        type: "member/updated",
        data: { ...member, error, phase: "failed" },
      });
    }
    return interrupted.length;
  }

  close(): void {
    if (!this.open) return;
    this.open = false;
    this.database.close();
  }

  private migrate(): void {
    this.database.exec(`
      CREATE TABLE IF NOT EXISTS swarm_migrations (
        version INTEGER PRIMARY KEY,
        applied_at INTEGER NOT NULL
      ) STRICT;
    `);
    const rows = this.database
      .prepare("SELECT version FROM swarm_migrations ORDER BY version")
      .all() as { version: number }[];
    const applied = new Set(rows.map((row) => row.version));
    const newest = Math.max(0, ...applied);
    if (newest > MIGRATION_VERSION) {
      throw new Error(`Swarm database version ${newest} is newer than supported`);
    }
    for (let version = 1; version <= MIGRATION_VERSION; version += 1) {
      if (applied.has(version)) continue;
      if (version === 1) {
        this.database.exec(`
          CREATE TABLE swarm_events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            occurred_at INTEGER NOT NULL,
            UNIQUE(team_id, revision)
          ) STRICT;
          CREATE INDEX swarm_events_team_idx ON swarm_events(team_id, seq);
          CREATE TABLE swarm_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
          ) STRICT;
          CREATE TABLE swarm_teams (
            team_id TEXT PRIMARY KEY,
            revision INTEGER NOT NULL,
            phase TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            snapshot_json TEXT NOT NULL
          ) STRICT;
        `);
      }
      if (version === 3) {
        this.database.exec(`
          CREATE TABLE swarm_attempts (
            team_id TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            started_at INTEGER NOT NULL,
            snapshot_json TEXT NOT NULL,
            PRIMARY KEY(team_id, attempt_id)
          ) STRICT;
          CREATE INDEX swarm_attempts_task_idx ON swarm_attempts(team_id, task_id, started_at);
        `);
      }
      if (version === 4) {
        this.database.exec(`
          CREATE TABLE swarm_member_bindings (
            workspace_key TEXT NOT NULL,
            runtime TEXT NOT NULL,
            member_id TEXT NOT NULL,
            handle TEXT NOT NULL,
            PRIMARY KEY(workspace_key, runtime, member_id),
            UNIQUE(runtime, handle)
          ) STRICT;
        `);
      }
      if (version === 5) {
        this.database.exec(`
          CREATE TABLE swarm_message_ledger (
            team_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            snapshot_json TEXT NOT NULL,
            PRIMARY KEY(team_id, message_id)
          ) STRICT;
        `);
        const cutoff = this.database
          .prepare("SELECT COALESCE(MAX(seq), 0) AS seq FROM swarm_events")
          .get() as {
          seq: number;
        };
        this.database
          .prepare("INSERT INTO swarm_settings(key, value) VALUES (?, ?)")
          .run(LEGACY_EVENT_CUTOFF_SETTING, String(cutoff.seq));
      }
      this.database
        .prepare("INSERT INTO swarm_migrations(version, applied_at) VALUES (?, ?)")
        .run(version, Date.now());
    }
    this.database
      .prepare("INSERT OR IGNORE INTO swarm_settings(key, value) VALUES (?, '0')")
      .run(LEGACY_EVENT_CUTOFF_SETTING);
  }

  private readTeam(teamId: string): SwarmTeamState | undefined {
    const row = this.database
      .prepare("SELECT snapshot_json FROM swarm_teams WHERE team_id = ?")
      .get(teamId) as TeamRow | undefined;
    return row ? swarmTeamStateSchema.parse(JSON.parse(row.snapshot_json)) : undefined;
  }

  private ensureWorkspaceSalt(): void {
    this.database
      .prepare("INSERT OR IGNORE INTO swarm_settings(key, value) VALUES ('workspace_salt', ?)")
      .run(randomBytes(32).toString("hex"));
  }

  private rebuildProjectionsInTransaction(): void {
    this.database.exec("DELETE FROM swarm_attempts");
    this.database.exec("DELETE FROM swarm_message_ledger");
    this.database.exec("DELETE FROM swarm_teams");
    const cutoffRow = this.database
      .prepare("SELECT value FROM swarm_settings WHERE key = ?")
      .get(LEGACY_EVENT_CUTOFF_SETTING) as { value: string } | undefined;
    const legacyEventCutoff = Number(cutoffRow?.value);
    if (!Number.isSafeInteger(legacyEventCutoff) || legacyEventCutoff < 0) {
      throw new Error("Swarm legacy event cutoff is invalid");
    }
    const states = new Map<string, SwarmTeamState>();
    const ignoredDuplicateMessages = new Set<string>();
    const rows = this.database
      .prepare("SELECT seq, team_id, revision, type, payload_json FROM swarm_events ORDER BY seq")
      .all() as unknown as EventRow[];
    for (const row of rows) {
      const legacy = row.seq <= legacyEventCutoff;
      const event = (legacy ? parseLegacyEvent : parseEvent)(
        row.type,
        JSON.parse(row.payload_json),
      );
      const current = states.get(row.team_id);
      const messageId =
        event.type === "message/queued"
          ? event.data.id
          : event.type === "message/delivery-started" || event.type === "message/delivered"
            ? event.data.messageId
            : undefined;
      const messageKey = messageId === undefined ? undefined : `${row.team_id}\0${messageId}`;
      let next: SwarmTeamState;
      if (legacy && event.type === "team/archived" && current?.archiveStartedAt === undefined) {
        next = applyLegacyArchivedEvent(row.team_id, current, row.revision, event.data.archivedAt);
      } else if (
        legacy &&
        event.type === "message/queued" &&
        this.readMessageLedger(row.team_id, event.data.id) !== undefined
      ) {
        if (messageKey !== undefined) ignoredDuplicateMessages.add(messageKey);
        next = advanceIgnoredLegacyMessageEvent(current, row.revision, event);
      } else if (
        legacy &&
        messageKey !== undefined &&
        ignoredDuplicateMessages.has(messageKey) &&
        (event.type === "message/delivery-started" || event.type === "message/delivered")
      ) {
        next = advanceIgnoredLegacyMessageEvent(current, row.revision, event);
      } else {
        next = applyEvent(row.team_id, current, row.revision, event);
        this.writeMessageLedgerEvent(row.team_id, event);
      }
      states.set(row.team_id, next);
    }
    for (const state of states.values()) this.writeTeam(state);
  }

  private verifyClientStorage(): void {
    const rows = this.database
      .prepare("SELECT version FROM swarm_migrations ORDER BY version")
      .all() as { version: number }[];
    const newest = Math.max(0, ...rows.map((row) => row.version));
    if (newest !== MIGRATION_VERSION) {
      throw new Error(
        `Swarm database version ${newest} is not the required version ${MIGRATION_VERSION}`,
      );
    }
    const salt = this.database
      .prepare("SELECT value FROM swarm_settings WHERE key = 'workspace_salt'")
      .get() as { value: string } | undefined;
    if (salt === undefined) throw new Error("Swarm workspace salt is unavailable");
    const cutoff = this.database
      .prepare("SELECT value FROM swarm_settings WHERE key = ?")
      .get(LEGACY_EVENT_CUTOFF_SETTING) as { value: string } | undefined;
    if (
      cutoff === undefined ||
      !Number.isSafeInteger(Number(cutoff.value)) ||
      Number(cutoff.value) < 0
    ) {
      throw new Error("Swarm legacy event cutoff is unavailable");
    }
    this.database.prepare("SELECT 1 FROM swarm_member_bindings LIMIT 1").get();
    this.database.prepare("SELECT 1 FROM swarm_message_ledger LIMIT 1").get();
  }

  private readMessageLedger(teamId: string, messageId: string): SwarmMessage | undefined {
    const row = this.database
      .prepare(
        `SELECT snapshot_json
         FROM swarm_message_ledger
         WHERE team_id = ? AND message_id = ?`,
      )
      .get(teamId, messageId) as MessageLedgerRow | undefined;
    return row === undefined ? undefined : swarmMessageSchema.parse(JSON.parse(row.snapshot_json));
  }

  private writeMessageLedgerEvent(teamId: string, event: SwarmEvent): void {
    if (event.type === "message/queued") {
      this.database
        .prepare(
          `INSERT OR IGNORE INTO swarm_message_ledger(
             team_id,
             message_id,
             target_id,
             sequence,
             snapshot_json
           ) VALUES (?, ?, ?, ?, ?)`,
        )
        .run(
          teamId,
          event.data.id,
          event.data.targetId,
          event.data.sequence,
          JSON.stringify(event.data),
        );
      return;
    }
    if (event.type !== "message/delivery-started" && event.type !== "message/delivered") return;
    const message = this.readMessageLedger(teamId, event.data.messageId);
    if (message === undefined) {
      throw new SwarmError("Swarm message ledger entry is unavailable", "SWARM_INVALID_REQUEST");
    }
    const updated =
      event.type === "message/delivery-started"
        ? { ...message, deliveryStartedAt: event.data.deliveryStartedAt }
        : { ...message, deliveredAt: event.data.deliveredAt };
    this.database
      .prepare(
        `UPDATE swarm_message_ledger
         SET snapshot_json = ?
         WHERE team_id = ? AND message_id = ?`,
      )
      .run(JSON.stringify(swarmMessageSchema.parse(updated)), teamId, event.data.messageId);
  }

  private appendEvent(
    teamId: string,
    current: SwarmTeamState | undefined,
    event: SwarmEvent,
  ): SwarmTeamState {
    if (event.type === "message/queued" && this.readMessageLedger(teamId, event.data.id)) {
      throw new SwarmError("Swarm message id already exists", "SWARM_MESSAGE_CONFLICT");
    }
    const revision = (current?.revision ?? 0) + 1;
    const next = applyEvent(teamId, current, revision, event);
    this.database
      .prepare(
        `INSERT INTO swarm_events(team_id, revision, type, payload_json, occurred_at)
         VALUES (?, ?, ?, ?, ?)`,
      )
      .run(teamId, revision, event.type, JSON.stringify(event.data), eventTimestamp(event));
    this.writeMessageLedgerEvent(teamId, event);
    this.writeTeam(next);
    return next;
  }

  private claimMemberBindingRow(binding: SwarmMemberBinding): "created" | "existing" {
    const memberOwner = this.database
      .prepare(
        `SELECT workspace_key, runtime, member_id, handle
         FROM swarm_member_bindings
         WHERE workspace_key = ? AND runtime = ? AND member_id = ?`,
      )
      .get(binding.workspaceKey, binding.runtime, binding.memberId) as MemberBindingRow | undefined;
    const handleOwner = this.database
      .prepare(
        `SELECT workspace_key, runtime, member_id, handle
         FROM swarm_member_bindings
         WHERE runtime = ? AND handle = ?`,
      )
      .get(binding.runtime, binding.handle) as MemberBindingRow | undefined;
    if (memberOwner !== undefined) {
      if (
        memberOwner.handle === binding.handle &&
        handleOwner?.workspace_key === binding.workspaceKey &&
        handleOwner.member_id === binding.memberId
      ) {
        return "existing";
      }
      if (handleOwner !== undefined) {
        throw new SwarmError(
          "Swarm runtime handle already belongs to another member",
          "SWARM_INVALID_REQUEST",
        );
      }
      throw new SwarmError(
        "Swarm member already belongs to another runtime handle",
        "SWARM_INVALID_REQUEST",
      );
    }
    if (handleOwner !== undefined) {
      throw new SwarmError(
        "Swarm runtime handle already belongs to another member",
        "SWARM_INVALID_REQUEST",
      );
    }
    this.database
      .prepare(
        `INSERT INTO swarm_member_bindings(workspace_key, runtime, member_id, handle)
         VALUES (?, ?, ?, ?)`,
      )
      .run(binding.workspaceKey, binding.runtime, binding.memberId, binding.handle);
    return "created";
  }

  private writeTeam(state: SwarmTeamState): void {
    this.database
      .prepare(
        `INSERT INTO swarm_teams(team_id, revision, phase, created_at, snapshot_json)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT(team_id) DO UPDATE SET
           revision = excluded.revision,
           phase = excluded.phase,
           created_at = excluded.created_at,
           snapshot_json = excluded.snapshot_json`,
      )
      .run(state.id, state.revision, state.phase, state.createdAt, JSON.stringify(state));
    this.database.prepare("DELETE FROM swarm_attempts WHERE team_id = ?").run(state.id);
    const insert = this.database.prepare(
      `INSERT INTO swarm_attempts(team_id, attempt_id, task_id, started_at, snapshot_json)
       VALUES (?, ?, ?, ?, ?)`,
    );
    for (const attempt of state.attempts) {
      insert.run(state.id, attempt.id, attempt.taskId, attempt.startedAt, JSON.stringify(attempt));
    }
  }

  private transaction<T>(operation: () => T): T {
    this.database.exec("BEGIN IMMEDIATE");
    try {
      const result = operation();
      this.database.exec("COMMIT");
      return result;
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
  }
}
