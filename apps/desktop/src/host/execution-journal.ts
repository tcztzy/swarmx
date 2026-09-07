import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";
import { chmodSync, mkdirSync } from "node:fs";
import { isAbsolute, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { type AGUIEvent, EventSchemas, EventType } from "@ag-ui/core";
import { z } from "zod";
import type { EventAttributes, NativeAgent, Observer } from "../agents/types.js";
import {
  ExecutionAttributes,
  type ExecutionRecord,
  ExecutionRecordSchema,
} from "../execution-record.js";
import {
  type AgentPermissions,
  AgentPermissionsSchema,
  intersectPermissions,
  ToolGrantSchema,
} from "../permissions.js";

export interface ExecutionContext {
  readonly permissions?: AgentPermissions;
  readonly sessionId: string | null;
  readonly runId: string;
  readonly causedBy: string | null;
  readonly attributes: EventAttributes;
  readonly agent?: NativeAgent;
  readonly interact?: Observer["interact"] | undefined;
  readonly cancelInteractions?: () => void;
  pendingInteractions?: number;
}

export class ExecutionJournal {
  readonly databasePath: string;
  readonly scope = new AsyncLocalStorage<ExecutionContext>();
  private readonly database: DatabaseSync;
  private readonly active = new Map<string, ExecutionContext>();
  private closed = false;

  constructor(
    root: string,
    private readonly workspaceId: string,
  ) {
    if (!isAbsolute(root)) throw new Error("Execution journal root must be absolute.");
    mkdirSync(root, { recursive: true, mode: 0o700 });
    chmodSync(root, 0o700);
    this.databasePath = join(root, "execution.sqlite");
    this.database = new DatabaseSync(this.databasePath);
    chmodSync(this.databasePath, 0o600);
    try {
      this.database.exec(
        "PRAGMA busy_timeout = 5000; PRAGMA journal_mode = WAL; PRAGMA synchronous = FULL; PRAGMA foreign_keys = ON;",
      );
      const { user_version: version } = this.database.prepare("PRAGMA user_version").get() as {
        user_version: number;
      };
      if (version !== 0 && version !== 1)
        throw new Error(`Unsupported execution journal schema: ${version}`);
      this.database.exec(`
        BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS execution_events (
          seq INTEGER PRIMARY KEY,
          id TEXT NOT NULL UNIQUE,
          workspace_id TEXT NOT NULL,
          session_id TEXT,
          run_id TEXT,
          caused_by TEXT REFERENCES execution_events(id),
          record_json TEXT NOT NULL CHECK(json_valid(record_json))
        ) STRICT;
        CREATE INDEX IF NOT EXISTS execution_workspace ON execution_events(workspace_id, seq);
        CREATE INDEX IF NOT EXISTS execution_session ON execution_events(workspace_id, session_id, seq);
        CREATE INDEX IF NOT EXISTS execution_run ON execution_events(workspace_id, run_id, seq);
        CREATE INDEX IF NOT EXISTS execution_cause ON execution_events(workspace_id, caused_by);
        CREATE INDEX IF NOT EXISTS execution_custom ON execution_events(workspace_id, json_extract(record_json, '$.event.name'), seq);
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_messages USING fts5(
          workspace_id UNINDEXED, session_id UNINDEXED, run_id UNINDEXED,
          role UNINDEXED, event_id UNINDEXED, text, tokenize='trigram'
        );
        CREATE TRIGGER IF NOT EXISTS execution_no_update BEFORE UPDATE ON execution_events
          BEGIN SELECT RAISE(ABORT, 'Execution journal is append-only'); END;
        CREATE TRIGGER IF NOT EXISTS execution_no_delete BEFORE DELETE ON execution_events
          BEGIN SELECT RAISE(ABORT, 'Execution journal is append-only'); END;
        CREATE TRIGGER IF NOT EXISTS execution_cause_workspace BEFORE INSERT ON execution_events
          WHEN NEW.caused_by IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM execution_events WHERE id = NEW.caused_by AND workspace_id = NEW.workspace_id
          ) BEGIN SELECT RAISE(ABORT, 'Execution cause belongs to another workspace or is missing'); END;
        PRAGMA user_version = 1;
        COMMIT;
      `);
    } catch (error) {
      this.database.close();
      throw error;
    }
  }

  append(
    context: ExecutionContext | null,
    event: AGUIEvent,
    attributes: EventAttributes = {},
  ): ExecutionRecord {
    const observedAt = new Date();
    const data = EventSchemas.parse({ timestamp: observedAt.getTime(), ...event });
    z.json().parse(data);
    const attribution = ExecutionAttributes.parse(
      Object.fromEntries(
        Object.entries({ ...context?.attributes, ...attributes }).filter(
          ([, value]) => value !== undefined,
        ),
      ),
    );
    this.database.exec("BEGIN IMMEDIATE");
    try {
      const { seq } = this.database
        .prepare("SELECT COALESCE(MAX(seq), 0) + 1 AS seq FROM execution_events")
        .get() as { seq: number };
      const record: ExecutionRecord = {
        schemaVersion: 1,
        seq,
        id: randomUUID(),
        observedAt: observedAt.toISOString(),
        workspaceId: this.workspaceId,
        sessionId: context?.sessionId ?? null,
        runId: context?.runId ?? null,
        causedBy: context?.causedBy ?? null,
        attributes: attribution,
        event: data,
      };
      const json = JSON.stringify(record);
      this.database
        .prepare(
          "INSERT INTO execution_events(seq, id, workspace_id, session_id, run_id, caused_by, record_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .run(
          seq,
          record.id,
          record.workspaceId,
          record.sessionId,
          record.runId,
          record.causedBy,
          json,
        );
      this.database.exec("COMMIT");
      return JSON.parse(json) as ExecutionRecord;
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
  }

  read({
    after = 0,
    limit = 200,
    session,
    run,
    descendants = false,
  }: {
    after?: number;
    limit?: number;
    session?: string | undefined;
    run?: string | undefined;
    descendants?: boolean;
  } = {}): { events: ExecutionRecord[]; nextAfter: number } {
    const rows = this.database
      .prepare(`
      WITH RECURSIVE related(id) AS (
        SELECT id FROM execution_events WHERE workspace_id = ? AND session_id = ?
        UNION
        SELECT child.id FROM execution_events child JOIN related parent ON child.caused_by = parent.id
        WHERE child.workspace_id = ? AND ?
      )
      SELECT record_json FROM execution_events
      WHERE workspace_id = ? AND seq > ? AND (? IS NULL OR id IN (SELECT id FROM related)) AND (? IS NULL OR run_id = ?)
      ORDER BY seq LIMIT ?
    `)
      .all(
        this.workspaceId,
        session ?? null,
        this.workspaceId,
        Number(descendants),
        this.workspaceId,
        after,
        session ?? null,
        run ?? null,
        run ?? null,
        limit,
      ) as { record_json: string }[];
    const events = rows.map(({ record_json }) =>
      ExecutionRecordSchema.parse(JSON.parse(record_json)),
    );
    return { events, nextAfter: events.at(-1)?.seq ?? after };
  }

  activeSession(sessionId: string): ExecutionContext | undefined {
    return this.active.get(sessionId);
  }

  memoryEvent(name: string, sessionId?: string) {
    const row = this.database
      .prepare(`SELECT record_json FROM execution_events
      WHERE workspace_id = ? AND json_extract(record_json, '$.event.name') = ?
      AND (? IS NULL OR session_id = ?) ORDER BY seq DESC LIMIT 1`)
      .get(this.workspaceId, name, sessionId ?? null, sessionId ?? null) as
      | { record_json: string }
      | undefined;
    return row ? ExecutionRecordSchema.parse(JSON.parse(row.record_json)) : undefined;
  }

  pendingMemories() {
    const rows = this.database
      .prepare(`SELECT proposal.record_json FROM execution_events proposal
      WHERE proposal.workspace_id = ? AND json_extract(proposal.record_json, '$.event.name') = 'swarmx.memory.proposed'
      AND NOT EXISTS (SELECT 1 FROM execution_events decision WHERE decision.workspace_id = proposal.workspace_id
        AND json_extract(decision.record_json, '$.event.name') IN ('swarmx.memory.accepted', 'swarmx.memory.rejected')
        AND json_extract(decision.record_json, '$.event.value.proposalId') = proposal.id)
      ORDER BY proposal.seq LIMIT 100`)
      .all(this.workspaceId) as { record_json: string }[];
    return rows.map(({ record_json }) => ExecutionRecordSchema.parse(JSON.parse(record_json)));
  }

  completedTurns(sessionId: string, after = 0) {
    const row = this.database
      .prepare(`SELECT COUNT(*) AS count FROM execution_events
      WHERE workspace_id = ? AND session_id = ? AND seq > ?
      AND json_extract(record_json, '$.event.type') = 'RUN_FINISHED'
      AND COALESCE(json_extract(record_json, '$.event.result.interruptionRequested'), 0) = 0
      AND COALESCE(json_extract(record_json, '$.event.result.stopReason'), 'end_turn') = 'end_turn'`)
      .get(this.workspaceId, sessionId, after) as { count: number };
    return row.count;
  }

  emptySessions(harness: string): string[] {
    const rows = this.database
      .prepare(`SELECT DISTINCT created.session_id FROM execution_events created
      WHERE created.workspace_id = ? AND json_extract(created.record_json, '$.event.name') = 'swarmx.session.created'
      AND json_extract(created.record_json, '$.attributes."swarmx.harness.name"') = ?
      AND NOT EXISTS (SELECT 1 FROM execution_events started WHERE started.workspace_id = created.workspace_id
        AND started.session_id = created.session_id AND json_extract(started.record_json, '$.event.type') = 'RUN_STARTED')`)
      .all(this.workspaceId, harness) as { session_id: string }[];
    return rows.map((row) => row.session_id);
  }

  conversationBindings(agentId: string): Map<string, string> {
    const rows = this.database
      .prepare(`SELECT record_json FROM execution_events
      WHERE workspace_id = ? AND json_extract(record_json, '$.event.name') = 'swarmx.a2a.context.bound'
      AND json_extract(record_json, '$.event.value.agentId') = ? ORDER BY seq`)
      .all(this.workspaceId, agentId) as { record_json: string }[];
    return new Map(
      rows.map(({ record_json }) => {
        const record = ExecutionRecordSchema.parse(JSON.parse(record_json));
        const { contextId, sessionId } = z
          .object({ value: z.object({ contextId: z.string(), sessionId: z.string() }) })
          .parse(record.event).value;
        return [contextId, sessionId];
      }),
    );
  }

  memoryToolEvidence(
    sessionId: string,
  ): { eventId: string; runId: string | null; event: AGUIEvent }[] {
    const rows = this.database
      .prepare(`SELECT record_json FROM execution_events
      WHERE workspace_id = ? AND session_id = ?
      AND json_extract(record_json, '$.event.type') IN ('TOOL_CALL_CHUNK', 'TOOL_CALL_RESULT')
      ORDER BY seq DESC LIMIT 30`)
      .all(this.workspaceId, sessionId) as { record_json: string }[];
    let remaining = 20_000;
    return rows
      .filter(({ record_json }) => {
        if (record_json.length > remaining) return false;
        remaining -= record_json.length;
        return true;
      })
      .reverse()
      .map(({ record_json }) => {
        const record = ExecutionRecordSchema.parse(JSON.parse(record_json));
        return { eventId: record.id, runId: record.runId, event: record.event };
      });
  }

  recall(raw: unknown) {
    const input = z
      .strictObject({
        query: z.string().trim().min(1).max(200).optional(),
        sessionId: z.string().min(1).max(2048).optional(),
        limit: z.number().int().min(1).max(50).default(20),
      })
      .parse(raw);
    // This is a rebuildable search projection. Original events remain authoritative.
    this.database
      .prepare(`INSERT INTO memory_messages(rowid, workspace_id, session_id, run_id, role, event_id, text)
      SELECT seq, workspace_id, session_id, run_id, 'user', id,
        json_extract(record_json, '$.event.input.messages[0].content') FROM execution_events e
      WHERE workspace_id = ? AND json_extract(record_json, '$.event.type') = 'RUN_STARTED'
      AND NOT EXISTS (SELECT 1 FROM memory_messages m WHERE m.rowid = e.seq)`)
      .run(this.workspaceId);
    this.database
      .prepare(`WITH messages AS (
      SELECT MIN(seq) AS first_seq, workspace_id, session_id, run_id, GROUP_CONCAT(delta, '') AS text FROM (
        SELECT seq, id, workspace_id, session_id, run_id, json_extract(record_json, '$.event.messageId') AS message_id,
          json_extract(record_json, '$.event.delta') AS delta FROM execution_events e
        WHERE workspace_id = ? AND json_extract(record_json, '$.event.type') = 'TEXT_MESSAGE_CHUNK'
          AND json_extract(record_json, '$.event.role') = 'assistant'
          AND EXISTS (SELECT 1 FROM execution_events completed WHERE completed.run_id = e.run_id
            AND json_extract(completed.record_json, '$.event.type') IN ('RUN_FINISHED', 'RUN_ERROR')) ORDER BY seq
      ) GROUP BY workspace_id, session_id, run_id, message_id)
      INSERT INTO memory_messages(rowid, workspace_id, session_id, run_id, role, event_id, text)
      SELECT m.first_seq, m.workspace_id, m.session_id, m.run_id, 'assistant', original.id, m.text
      FROM messages m JOIN execution_events original ON original.seq = m.first_seq
      WHERE NOT EXISTS (SELECT 1 FROM memory_messages stored WHERE stored.rowid = m.first_seq)`)
      .run(this.workspaceId);
    const query = input.query ?? "";
    const useFts = Array.from(query).length >= 3;
    const rows = this.database
      .prepare(`SELECT session_id, run_id, role, event_id, text FROM memory_messages
      WHERE workspace_id = ? AND (? IS NULL OR session_id = ?)
        AND ${useFts ? "text MATCH ?" : "instr(lower(text), lower(?)) > 0"}
      ORDER BY rowid DESC LIMIT ?`)
      .all(
        this.workspaceId,
        input.sessionId ?? null,
        input.sessionId ?? null,
        useFts ? `"${query.replaceAll('"', '""')}"` : query,
        input.limit,
      ) as {
      session_id: string;
      run_id: string;
      role: string;
      event_id: string;
      text: string;
    }[];
    return rows.map((row) => ({
      sessionId: row.session_id,
      runId: row.run_id,
      role: row.role,
      eventId: row.event_id,
      text: row.text,
    }));
  }

  sessionIds(): string[] {
    return (
      this.database
        .prepare(
          `SELECT DISTINCT session_id FROM execution_events
           WHERE workspace_id = ? AND session_id IS NOT NULL AND (
             json_extract(record_json, '$.event.type') = 'RUN_STARTED' OR
             json_extract(record_json, '$.event.name') = 'swarmx.session.created'
           )`,
        )
        .all(this.workspaceId) as { session_id: string }[]
    ).map(({ session_id }) => session_id);
  }

  sessionPermissions(sessionId: string): AgentPermissions | undefined {
    const rows = this.database
      .prepare(`
      SELECT CASE WHEN json_extract(record_json, '$.event.type') = 'RUN_STARTED'
        THEN json_extract(record_json, '$.event.input.forwardedProps.permissions')
        ELSE json_extract(record_json, '$.event.value.permissions') END AS permissions
      FROM execution_events WHERE workspace_id = ? AND session_id = ? AND (
        (json_extract(record_json, '$.event.type') = 'RUN_STARTED'
          AND json_type(record_json, '$.event.input.forwardedProps.permissions') IS NOT NULL) OR
        (json_extract(record_json, '$.event.name') = 'swarmx.session.created'
          AND json_type(record_json, '$.event.value.permissions') IS NOT NULL)
      ) ORDER BY seq
    `)
      .all(this.workspaceId, sessionId) as { permissions: string }[];
    let permissions: AgentPermissions | undefined;
    for (const row of rows) {
      const raw: unknown = JSON.parse(row.permissions);
      const legacy = AgentPermissionsSchema.omit({ tools: true })
        .extend({ filesystem: z.enum(["read-only", "workspace-write"]) })
        .safeParse(raw);
      const grant = legacy.success
        ? {
            harnesses: legacy.data.harnesses,
            delegation: legacy.data.delegation,
            tools: ToolGrantSchema.options.filter(
              (tool) => legacy.data.filesystem !== "read-only" || tool.endsWith(".read"),
            ),
          }
        : AgentPermissionsSchema.parse(raw);
      permissions = permissions ? intersectPermissions(permissions, grant) : grant;
    }
    return permissions;
  }

  assertCurrentPermissions(sessionId: string): void {
    const legacy = this.database
      .prepare(`SELECT 1 FROM execution_events WHERE workspace_id = ? AND session_id = ? AND (
        json_type(record_json, '$.event.input.forwardedProps.permissions.filesystem') IS NOT NULL OR
        json_type(record_json, '$.event.value.permissions.filesystem') IS NOT NULL
      ) LIMIT 1`)
      .get(this.workspaceId, sessionId);
    if (legacy)
      throw new Error(
        "This conversation has legacy filesystem permissions. Its history remains available; create a new conversation and select its native mode. See docs/permissions.md.",
      );
  }

  sessionMode(sessionId: string): string | undefined {
    const row = this.database
      .prepare(`SELECT COALESCE(
          json_extract(record_json, '$.attributes."swarmx.native.mode"'),
          json_extract(record_json, '$.event.input.forwardedProps.mode')) AS mode
        FROM execution_events WHERE workspace_id = ? AND session_id = ?
        AND ((json_extract(record_json, '$.event.type') = 'RUN_STARTED'
          AND json_type(record_json, '$.event.input.forwardedProps.mode') = 'text') OR
          (json_extract(record_json, '$.event.type') = 'RAW'
          AND json_type(record_json, '$.attributes."swarmx.native.mode"') = 'text'))
        ORDER BY seq DESC LIMIT 1`)
      .get(this.workspaceId, sessionId) as { mode: string } | undefined;
    return row?.mode;
  }

  activeRuns(): ExecutionContext[] {
    return [...this.active.values()];
  }

  activate(context: ExecutionContext): void {
    if (context.sessionId === null) throw new Error("An Agent run requires a session.");
    if (this.active.has(context.sessionId)) throw new Error("Session is busy.");
    this.active.set(context.sessionId, context);
  }

  deactivate(sessionId: string): void {
    this.active.delete(sessionId);
  }

  async tool<T>(
    name: string,
    args: unknown,
    caller: {
      actorId: string;
      callId: string;
      sessionId?: string | undefined;
      runId?: string | undefined;
    },
    execute: () => Promise<T>,
  ): Promise<T> {
    const parent =
      caller.sessionId === undefined ? this.scope.getStore() : this.activeSession(caller.sessionId);
    if (caller.sessionId !== undefined && !parent)
      throw new Error("MCP session has no active SwarmX execution.");
    if (caller.sessionId !== undefined && caller.runId !== parent?.runId)
      throw new Error("MCP execution does not match the active session run.");
    const context: ExecutionContext = {
      ...(parent?.permissions ? { permissions: parent.permissions } : {}),
      sessionId: parent?.sessionId ?? null,
      runId: parent?.runId ?? randomUUID(),
      causedBy: parent?.causedBy ?? null,
      attributes: { ...parent?.attributes, "swarmx.actor.id": caller.actorId },
      interact: parent?.interact,
    };
    const started = this.append(context, {
      type: EventType.TOOL_CALL_START,
      toolCallId: caller.callId,
      toolCallName: name,
    });
    const scope = { ...context, causedBy: started.id };
    this.append(scope, {
      type: EventType.TOOL_CALL_ARGS,
      toolCallId: caller.callId,
      delta: JSON.stringify(args ?? null),
    });
    this.append(scope, { type: EventType.TOOL_CALL_END, toolCallId: caller.callId });
    return this.scope.run(scope, async () => {
      try {
        const value = await execute();
        this.append(scope, {
          type: EventType.TOOL_CALL_RESULT,
          toolCallId: caller.callId,
          messageId: randomUUID(),
          content: JSON.stringify(value ?? null),
        });
        return value;
      } catch (error) {
        this.append(scope, {
          type: EventType.CUSTOM,
          name: "swarmx.tool.failed",
          value: {
            toolCallId: caller.callId,
            message: error instanceof Error ? error.message : String(error),
          },
        });
        throw error;
      }
    });
  }

  close(): void {
    if (this.closed) return;
    this.database.close();
    this.closed = true;
  }
}
