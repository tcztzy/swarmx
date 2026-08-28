import { createHash, randomBytes } from "node:crypto";
import { chmodSync, mkdirSync, realpathSync } from "node:fs";
import { isAbsolute, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { z } from "zod";
import {
  type MonitorFinding,
  monitorFindingSchema,
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
const MIGRATION_VERSION = 3;

const createdEventDataSchema = z.strictObject({
  createdAt: z.number().int().nonnegative(),
  lead: swarmMemberSchema,
  name: z.string().trim().min(1).max(100),
  workspaceKey: z.string().regex(/^swarmx--[0-9a-f]{64}$/u),
});
const archivedEventDataSchema = z.strictObject({
  archivedAt: z.number().int().nonnegative(),
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
  team_id: string;
  revision: number;
  type: SwarmEvent["type"];
  payload_json: string;
}

interface TeamRow {
  snapshot_json: string;
}

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

function eventTimestamp(event: SwarmEvent): number {
  switch (event.type) {
    case "team/created":
      return event.data.createdAt;
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
    if (
      existing.sequence !== task.sequence ||
      existing.createdAt !== task.createdAt ||
      existing.id !== task.id
    ) {
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
  if (state.messages.some((candidate) => candidate.sequence === message.sequence)) {
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

  const updatedAt = eventTimestamp(event);
  switch (event.type) {
    case "team/archived":
      return swarmTeamStateSchema.parse({
        ...current,
        revision,
        phase: "archived",
        archivedAt: event.data.archivedAt,
        updatedAt,
      });
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
      return swarmTeamStateSchema.parse({
        ...current,
        revision,
        messages: queueMessage(current, event.data),
        updatedAt,
      });
    case "message/delivery-started":
      return swarmTeamStateSchema.parse({
        ...current,
        revision,
        messages: startMessageDelivery(current, event.data),
        updatedAt,
      });
    case "message/delivered":
      return swarmTeamStateSchema.parse({
        ...current,
        revision,
        messages: deliverMessage(current, event.data),
        updatedAt,
      });
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

/** Owner-only append-only event log with rebuildable swarm projections. */
export class SwarmJournal {
  readonly databasePath: string;

  private readonly database: DatabaseSync;
  private open = true;

  constructor(root: string) {
    if (!isAbsolute(root)) {
      throw new SwarmError("Swarm storage root must be absolute", "SWARM_INVALID_REQUEST");
    }
    mkdirSync(root, { recursive: true, mode: 0o700 });
    chmodSync(root, 0o700);
    this.databasePath = join(root, DATABASE_NAME);
    this.database = new DatabaseSync(this.databasePath);
    chmodSync(this.databasePath, 0o600);
    this.database.exec(`
      PRAGMA foreign_keys = ON;
      PRAGMA synchronous = FULL;
      PRAGMA busy_timeout = 5000;
      PRAGMA journal_mode = WAL;
    `);
    this.migrate();
    this.ensureWorkspaceSalt();
    this.rebuildProjections();
  }

  append(teamId: string, input: SwarmEvent): SwarmTeamState {
    assertOpen(this.open);
    const event = parseEvent(input.type, input.data);
    return this.transaction(() => {
      const current = this.readTeam(teamId);
      const revision = (current?.revision ?? 0) + 1;
      const next = applyEvent(teamId, current, revision, event);
      this.database
        .prepare(
          `INSERT INTO swarm_events(team_id, revision, type, payload_json, occurred_at)
           VALUES (?, ?, ?, ?, ?)`,
        )
        .run(teamId, revision, event.type, JSON.stringify(event.data), eventTimestamp(event));
      this.writeTeam(next);
      return clone(next);
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
    this.transaction(() => {
      this.database.exec("DELETE FROM swarm_attempts");
      this.database.exec("DELETE FROM swarm_teams");
      const states = new Map<string, SwarmTeamState>();
      const rows = this.database
        .prepare("SELECT team_id, revision, type, payload_json FROM swarm_events ORDER BY seq")
        .all() as unknown as EventRow[];
      for (const row of rows) {
        const event = parseEvent(row.type, JSON.parse(row.payload_json));
        const next = applyEvent(row.team_id, states.get(row.team_id), row.revision, event);
        states.set(row.team_id, next);
      }
      for (const state of states.values()) this.writeTeam(state);
    });
  }

  recoverInterruptedTasks(now: number): number {
    assertOpen(this.open);
    const interrupted = this.list().flatMap((team) =>
      team.phase === "active"
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

  recoverUncertainIntents(now: number): number {
    assertOpen(this.open);
    const started = this.list().flatMap((team) =>
      team.phase === "active"
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
      this.transaction(() => {
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
        this.database
          .prepare("INSERT INTO swarm_migrations(version, applied_at) VALUES (?, ?)")
          .run(version, Date.now());
      });
    }
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
