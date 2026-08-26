import { createHash, randomBytes } from "node:crypto";
import { chmodSync, mkdirSync, realpathSync } from "node:fs";
import { isAbsolute, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { z } from "zod";
import {
  type SwarmMember,
  type SwarmMessage,
  type SwarmTask,
  type SwarmTeamState,
  swarmMemberSchema,
  swarmMessageSchema,
  swarmTaskSchema,
  swarmTeamStateSchema,
} from "./contracts.js";
import { SwarmError } from "./errors.js";

const DATABASE_NAME = "swarm.sqlite";
const MIGRATION_VERSION = 1;

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

export type SwarmEvent =
  | { type: "team/created"; data: z.infer<typeof createdEventDataSchema> }
  | { type: "team/archived"; data: z.infer<typeof archivedEventDataSchema> }
  | { type: "member/updated"; data: SwarmMember }
  | { type: "task/updated"; data: SwarmTask }
  | { type: "message/queued"; data: SwarmMessage }
  | { type: "message/delivered"; data: z.infer<typeof deliveredEventDataSchema> };

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
    case "message/queued":
      return { type, data: swarmMessageSchema.parse(value) };
    case "message/delivered":
      return { type, data: deliveredEventDataSchema.parse(value) };
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
    case "message/queued":
      return event.data.createdAt;
    case "message/delivered":
      return event.data.deliveredAt;
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
      existing.createdAt !== member.createdAt)
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

function queueMessage(state: SwarmTeamState, message: SwarmMessage): SwarmMessage[] {
  if (state.messages.some((candidate) => candidate.id === message.id)) {
    throw new SwarmError("Swarm message id already exists", "SWARM_INVALID_REQUEST");
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
    case "message/queued":
      return swarmTeamStateSchema.parse({
        ...current,
        revision,
        messages: queueMessage(current, event.data),
        updatedAt,
      });
    case "message/delivered":
      return swarmTeamStateSchema.parse({
        ...current,
        revision,
        messages: deliverMessage(current, event.data),
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
            .filter((task) => task.status === "in_progress")
            .map((task) => ({ teamId: team.id, task }))
        : [],
    );
    for (const { teamId, task } of interrupted) {
      const { attemptId: _attemptId, ...rest } = task;
      this.append(teamId, {
        type: "task/updated",
        data: {
          ...rest,
          revision: task.revision + 1,
          status: "needs_attention",
          updatedAt: now,
        },
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
