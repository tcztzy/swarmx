import { mkdtempSync, statSync } from "node:fs";
import { rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import { SwarmJournal } from "../src/journal.js";

const roots: string[] = [];

function fixture() {
  const root = mkdtempSync(join(tmpdir(), "swarmx-swarm-journal-"));
  roots.push(root);
  return { journal: new SwarmJournal(root), root };
}

function createTeam(journal: SwarmJournal, teamId = "session-lead") {
  return journal.append(teamId, {
    type: "team/created",
    data: {
      createdAt: 100,
      lead: {
        createdAt: 100,
        description: "Team lead",
        id: teamId,
        name: "lead",
        phase: "active",
        role: "lead",
      },
      name: "Research team",
      workspaceKey: `swarmx--${"a".repeat(64)}`,
    },
  });
}

function createLegacyDatabase(root: string, version: 1 | 2 | 3 | 4) {
  const databasePath = join(root, "swarm.sqlite");
  const database = new DatabaseSync(databasePath);
  database.exec(`
    CREATE TABLE swarm_migrations (version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL) STRICT;
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
    CREATE TABLE swarm_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT;
    INSERT INTO swarm_settings(key, value) VALUES ('workspace_salt', '${"a".repeat(64)}');
    CREATE TABLE swarm_teams (
      team_id TEXT PRIMARY KEY,
      revision INTEGER NOT NULL,
      phase TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      snapshot_json TEXT NOT NULL
    ) STRICT;
  `);
  for (let applied = 1; applied <= version; applied += 1) {
    database
      .prepare("INSERT INTO swarm_migrations(version, applied_at) VALUES (?, ?)")
      .run(applied, applied);
  }
  if (version >= 3) {
    database.exec(`
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
  if (version >= 4) {
    database.exec(`
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
  return { database, databasePath };
}

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { force: true, recursive: true })));
});

describe("Swarm journal", () => {
  it("V194: migrates a v1 event fixture through v5 with explicit legacy defaults", () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-swarm-journal-v1-"));
    roots.push(root);
    const { database, databasePath } = createLegacyDatabase(root, 1);
    database
      .prepare(
        `INSERT INTO swarm_events(team_id, revision, type, payload_json, occurred_at)
         VALUES (?, 1, 'team/created', ?, 100)`,
      )
      .run(
        "session-lead",
        JSON.stringify({
          createdAt: 100,
          lead: {
            createdAt: 100,
            description: "Team lead",
            id: "session-lead",
            name: "lead",
            phase: "active",
            role: "lead",
          },
          name: "Legacy team",
          workspaceKey: `swarmx--${"b".repeat(64)}`,
        }),
      );
    database
      .prepare(
        `INSERT INTO swarm_events(team_id, revision, type, payload_json, occurred_at)
         VALUES (?, 2, 'member/updated', ?, 101)`,
      )
      .run(
        "session-lead",
        JSON.stringify({
          createdAt: 101,
          description: "Old member",
          id: "session-old",
          name: "old",
          phase: "active",
          role: "member",
        }),
      );
    database.close();

    const migrated = new SwarmJournal(root);
    expect(migrated.get("session-lead")?.members[1]).toMatchObject({
      role: "legacy",
      modelPolicy: { source: "legacy-default" },
    });
    expect(migrated.get("session-lead")?.attempts).toEqual([]);
    migrated.close();

    const inspected = new DatabaseSync(databasePath, { readOnly: true });
    expect(
      inspected.prepare("SELECT version FROM swarm_migrations ORDER BY version").all(),
    ).toEqual([{ version: 1 }, { version: 2 }, { version: 3 }, { version: 4 }, { version: 5 }]);
    expect(
      inspected
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'swarm_attempts'")
        .get(),
    ).toEqual({ name: "swarm_attempts" });
    inspected.close();
  });

  it.each([2, 3, 4] as const)("V194: migrates a v%s database directly through v5", (version) => {
    const root = mkdtempSync(join(tmpdir(), `swarmx-swarm-journal-v${String(version)}-`));
    roots.push(root);
    const { database, databasePath } = createLegacyDatabase(root, version);
    database.close();

    const migrated = new SwarmJournal(root);
    expect(migrated.list()).toEqual([]);
    migrated.close();

    const inspected = new DatabaseSync(databasePath, { readOnly: true });
    expect(
      inspected.prepare("SELECT version FROM swarm_migrations ORDER BY version").all(),
    ).toEqual([{ version: 1 }, { version: 2 }, { version: 3 }, { version: 4 }, { version: 5 }]);
    expect(
      inspected
        .prepare(
          "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'swarm_member_bindings'",
        )
        .get(),
    ).toEqual({ name: "swarm_member_bindings" });
    expect(
      inspected
        .prepare(
          "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'swarm_message_ledger'",
        )
        .get(),
    ).toEqual({ name: "swarm_message_ledger" });
    inspected.close();
  });

  it("V173/V194: replays a pre-fence archived v4 Team during migration", () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-swarm-journal-v4-archive-"));
    roots.push(root);
    const { database } = createLegacyDatabase(root, 4);
    const insert = database.prepare(
      `INSERT INTO swarm_events(team_id, revision, type, payload_json, occurred_at)
       VALUES ('session-lead', ?, ?, ?, ?)`,
    );
    insert.run(
      1,
      "team/created",
      JSON.stringify({
        createdAt: 100,
        lead: {
          createdAt: 100,
          description: "Team lead",
          id: "session-lead",
          name: "lead",
          phase: "active",
          role: "lead",
        },
        name: "Legacy archived team",
        workspaceKey: `swarmx--${"c".repeat(64)}`,
      }),
      100,
    );
    insert.run(
      2,
      "member/updated",
      JSON.stringify({
        createdAt: 101,
        description: "Legacy active member",
        id: "session-member",
        name: "member",
        phase: "active",
        role: "member",
      }),
      101,
    );
    insert.run(3, "team/archived", JSON.stringify({ archivedAt: 102 }), 102);
    database.close();

    const migrated = new SwarmJournal(root);
    expect(migrated.get("session-lead")).toMatchObject({
      archivedAt: 102,
      phase: "archived",
      members: expect.arrayContaining([
        expect.objectContaining({ id: "session-member", phase: "active" }),
      ]),
    });
    migrated.close();
  });

  it("V179/V194: migrates legacy committed admissions without fabricating receipts", () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-swarm-journal-v4-admission-"));
    roots.push(root);
    const { database } = createLegacyDatabase(root, 4);
    const insert = database.prepare(
      `INSERT INTO swarm_events(team_id, revision, type, payload_json, occurred_at)
       VALUES ('session-lead', ?, ?, ?, ?)`,
    );
    insert.run(
      1,
      "team/created",
      JSON.stringify({
        createdAt: 100,
        lead: {
          createdAt: 100,
          description: "Team lead",
          id: "session-lead",
          name: "lead",
          phase: "active",
          role: "lead",
        },
        name: "Legacy admission team",
        workspaceKey: `swarmx--${"e".repeat(64)}`,
      }),
      100,
    );
    const admission = {
      id: "21000000-0000-4000-8000-000000000001",
      revision: 1,
      taskId: "task-1",
      taskRevision: 1,
      attemptId: "attempt-1",
      requestHash: `sha256:${"a".repeat(64)}`,
      targetKind: "pkb_concept",
      sources: [{ kind: "reference", resource: "docs/source.md" }],
      verification: {
        status: "verified",
        method: "source_reviewed",
        verifiedAt: 101,
      },
      status: "committed",
      createdAt: 101,
      updatedAt: 101,
    };
    insert.run(2, "knowledge/admission-updated", JSON.stringify(admission), 101);
    insert.run(
      3,
      "knowledge/admission-updated",
      JSON.stringify({
        ...admission,
        id: "21000000-0000-4000-8000-000000000002",
        receipt: {
          kind: "science_evidence",
          entityId: "22000000-0000-4000-8000-000000000001",
          journalSequence: 1,
        },
      }),
      102,
    );
    database.close();

    const migrated = new SwarmJournal(root);
    const admissions = migrated.get("session-lead")?.admissions;
    expect(admissions).toEqual([
      expect.objectContaining({
        id: "21000000-0000-4000-8000-000000000001",
        status: "uncertain",
      }),
      expect.objectContaining({
        id: "21000000-0000-4000-8000-000000000002",
        status: "uncertain",
      }),
    ]);
    expect(admissions?.[0]).not.toHaveProperty("receipt");
    expect(admissions?.[1]).not.toHaveProperty("receipt");
    migrated.close();
  });

  it("V194: rejects a database newer than the supported journal version", () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-swarm-journal-newer-"));
    roots.push(root);
    const { database } = createLegacyDatabase(root, 3);
    database.prepare("INSERT INTO swarm_migrations(version, applied_at) VALUES (6, 6)").run();
    database.close();

    expect(() => new SwarmJournal(root)).toThrow("version 6 is newer than supported");
  });

  it("V194/V224: auxiliary clients require initialized v5 storage and preserve projections", () => {
    const { journal, root } = fixture();
    const team = createTeam(journal);
    journal.close();
    const databasePath = join(root, "swarm.sqlite");
    const marker = 424_242;
    const database = new DatabaseSync(databasePath);
    database
      .prepare("UPDATE swarm_teams SET created_at = ? WHERE team_id = ?")
      .run(marker, team.id);
    database.close();

    const client = new SwarmJournal(root, { mode: "client" });
    expect(client.get(team.id)?.id).toBe(team.id);
    client.close();
    const inspected = new DatabaseSync(databasePath, { readOnly: true });
    expect(
      inspected.prepare("SELECT created_at FROM swarm_teams WHERE team_id = ?").get(team.id),
    ).toEqual({ created_at: marker });
    inspected.close();

    const uninitialized = mkdtempSync(join(tmpdir(), "swarmx-swarm-client-uninitialized-"));
    roots.push(uninitialized);
    expect(() => new SwarmJournal(uninitialized, { mode: "client" })).toThrow(
      "not been initialized by the platform owner",
    );
  });

  it("V167/V180: atomically queues idempotent messages and grants one delivery claim", () => {
    const { journal, root } = fixture();
    const team = createTeam(journal);
    for (const [id, name] of [
      ["member-1", "one"],
      ["member-2", "two"],
    ] as const) {
      journal.append(team.id, {
        type: "member/updated",
        data: {
          createdAt: 101,
          description: name,
          id,
          name,
          phase: "active",
          role: "legacy",
          modelPolicy: { source: "legacy-default" },
        },
      });
    }
    const second = new SwarmJournal(root, { mode: "client" });
    const input = {
      content: "one delivery",
      createdAt: 101,
      delivery: "quiet" as const,
      id: "60000000-0000-4000-8000-000000000001",
      senderId: team.id,
      senderName: "lead",
      targetId: "member-1",
    };

    expect(journal.queueMessage(team.id, input, 2)).toMatchObject({
      created: true,
      message: { sequence: 1 },
    });
    expect(second.queueMessage(team.id, input, 2)).toMatchObject({
      created: false,
      message: { sequence: 1 },
    });
    expect(
      second.queueMessage(
        team.id,
        { ...input, id: "60000000-0000-4000-8000-000000000002", content: "next" },
        2,
      ),
    ).toMatchObject({ created: true, message: { sequence: 2 } });
    expect(
      second.queueMessage(
        team.id,
        {
          ...input,
          id: "60000000-0000-4000-8000-000000000003",
          content: "other target",
          targetId: "member-2",
        },
        2,
      ),
    ).toMatchObject({ created: true, message: { sequence: 1 } });
    expect(journal.claimMessageDelivery(team.id, input.id, 102)).toBe(true);
    expect(second.claimMessageDelivery(team.id, input.id, 103)).toBe(false);
    expect(second.get(team.id)?.messages[0]).toMatchObject({ deliveryStartedAt: 102 });
    journal.close();
    second.close();
  });

  it("V163/V167: rejects stale senders and failed targets inside the queue transaction", () => {
    const { journal, root } = fixture();
    const team = createTeam(journal);
    const target = {
      createdAt: 101,
      description: "target",
      id: "member-1",
      name: "target",
      phase: "active" as const,
      role: "legacy" as const,
      modelPolicy: { source: "legacy-default" as const },
    };
    journal.append(team.id, { type: "member/updated", data: target });
    const second = new SwarmJournal(root, { mode: "client" });
    const input = {
      content: "must remain authorized",
      createdAt: 102,
      delivery: "quiet" as const,
      id: "60000000-0000-4000-8000-000000000004",
      senderId: team.id,
      senderName: "lead",
      targetId: target.id,
    };

    expect(() => journal.queueMessage(team.id, input, 2, () => false)).toThrow(
      "actor identity is stale",
    );
    expect(journal.queueMessage(team.id, input, 2, () => true)).toMatchObject({ created: true });
    expect(() => journal.claimMessageDelivery(team.id, input.id, 103, () => false)).toThrow(
      "target identity is stale",
    );
    second.append(team.id, {
      type: "member/updated",
      data: { ...target, error: "native Thread archived", phase: "failed" },
    });
    expect(journal.queueMessage(team.id, input, 2, () => true)).toMatchObject({ created: false });
    expect(() => journal.claimMessageDelivery(team.id, input.id, 104, () => true)).toThrow(
      "target is unavailable",
    );
    expect(() =>
      journal.queueMessage(
        team.id,
        { ...input, id: "60000000-0000-4000-8000-000000000005" },
        2,
        () => true,
      ),
    ).toThrow("target is unavailable");
    expect(journal.get(team.id)?.messages).toEqual([
      expect.objectContaining({ id: "60000000-0000-4000-8000-000000000004" }),
    ]);
    journal.close();
    second.close();
  });

  it("V167/V180: retains idempotency and target sequence outside the bounded projection", () => {
    const { journal, root } = fixture();
    const team = createTeam(journal);
    const target = {
      createdAt: 101,
      description: "target",
      id: "member-ledger",
      name: "ledger",
      phase: "active" as const,
      role: "legacy" as const,
      modelPolicy: { source: "legacy-default" as const },
    };
    journal.append(team.id, { type: "member/updated", data: target });
    const first = {
      content: "message-1",
      createdAt: 102,
      delivery: "quiet" as const,
      id: "70000000-0000-4000-8000-000000000001",
      senderId: team.id,
      senderName: "lead",
      targetId: target.id,
    };
    expect(journal.queueMessage(team.id, first, 1).message.sequence).toBe(1);
    expect(journal.claimMessageDelivery(team.id, first.id, 10_001)).toBe(true);
    journal.append(team.id, {
      type: "message/delivered",
      data: { messageId: first.id, deliveredAt: 20_001 },
    });
    const databasePath = journal.databasePath;
    journal.close();

    const database = new DatabaseSync(databasePath);
    const row = database
      .prepare("SELECT snapshot_json FROM swarm_teams WHERE team_id = ?")
      .get(team.id) as { snapshot_json: string };
    const projected = JSON.parse(row.snapshot_json) as { messages: unknown[] };
    const last = {
      ...first,
      content: "projected-last",
      createdAt: 4_200,
      id: "70000000-0000-4000-8000-000000004097",
      sequence: 4_097,
    };
    projected.messages = [last];
    database
      .prepare("UPDATE swarm_teams SET snapshot_json = ? WHERE team_id = ?")
      .run(JSON.stringify(projected), team.id);
    database
      .prepare(
        `INSERT INTO swarm_message_ledger(team_id, message_id, target_id, sequence, snapshot_json)
         VALUES (?, ?, ?, ?, ?)`,
      )
      .run(team.id, last.id, last.targetId, last.sequence, JSON.stringify(last));
    database.close();

    const reopened = new SwarmJournal(root, { mode: "client" });
    expect(reopened.queueMessage(team.id, first, 1)).toMatchObject({
      created: false,
      message: { deliveredAt: 20_001, sequence: 1 },
    });
    expect(
      reopened.queueMessage(
        team.id,
        {
          ...first,
          content: "after-trim",
          createdAt: 5_000,
          id: "70000000-0000-4000-8000-000000009999",
        },
        2,
      ).message.sequence,
    ).toBe(4_098);
    reopened.close();
  });

  it("V167/V180/V194: earliest legacy message ownership wins v4 replay after trimming", () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-swarm-journal-v4-mailbox-"));
    roots.push(root);
    const { database } = createLegacyDatabase(root, 4);
    const insert = database.prepare(
      `INSERT INTO swarm_events(team_id, revision, type, payload_json, occurred_at)
         VALUES ('session-lead', ?, ?, ?, ?)`,
    );
    let revision = 1;
    insert.run(
      revision,
      "team/created",
      JSON.stringify({
        createdAt: 100,
        lead: {
          createdAt: 100,
          description: "Team lead",
          id: "session-lead",
          name: "lead",
          phase: "active",
          role: "lead",
        },
        name: "Legacy mailbox team",
        workspaceKey: `swarmx--${"d".repeat(64)}`,
      }),
      100,
    );
    revision += 1;
    insert.run(
      revision,
      "member/updated",
      JSON.stringify({
        createdAt: 101,
        description: "Mailbox target",
        id: "session-target",
        name: "target",
        phase: "active",
        role: "member",
      }),
      101,
    );
    const firstId = "70000000-0000-4000-8000-000000000001";
    database.exec("BEGIN");
    for (let index = 1; index <= 4_097; index += 1) {
      const messageId = `70000000-0000-4000-8000-${String(index).padStart(12, "0")}`;
      revision += 1;
      insert.run(
        revision,
        "message/queued",
        JSON.stringify({
          content: `message-${String(index)}`,
          createdAt: 101 + index,
          delivery: "quiet",
          id: messageId,
          senderId: "session-lead",
          senderName: "lead",
          targetId: "session-target",
          sequence: index,
        }),
        101 + index,
      );
      revision += 1;
      insert.run(
        revision,
        "message/delivered",
        JSON.stringify({ messageId, deliveredAt: 10_000 + index }),
        10_000 + index,
      );
    }
    revision += 1;
    insert.run(
      revision,
      "message/queued",
      JSON.stringify({
        content: "message-1",
        createdAt: 30_000,
        delivery: "quiet",
        id: firstId,
        senderId: "session-lead",
        senderName: "lead",
        targetId: "session-target",
        sequence: 4_098,
      }),
      30_000,
    );
    database.exec("COMMIT");
    database.close();

    const migrated = new SwarmJournal(root);
    expect(migrated.get("session-lead")?.messages).toHaveLength(4_096);
    expect(migrated.get("session-lead")?.messages.some((message) => message.id === firstId)).toBe(
      false,
    );
    expect(
      migrated.queueMessage(
        "session-lead",
        {
          content: "message-1",
          createdAt: 102,
          delivery: "quiet",
          id: firstId,
          senderId: "session-lead",
          senderName: "lead",
          targetId: "session-target",
        },
        2,
      ),
    ).toMatchObject({ created: false, message: { deliveredAt: 10_001, sequence: 1 } });
    expect(
      migrated.queueMessage(
        "session-lead",
        {
          content: "new-message",
          createdAt: 31_000,
          delivery: "quiet",
          id: "70000000-0000-4000-8000-000000009999",
          senderId: "session-lead",
          senderName: "lead",
          targetId: "session-target",
        },
        2,
      ).message.sequence,
    ).toBe(4_098);
    migrated.close();
  }, 30_000);

  it("V189/V192: commits attempt events with their projection and rebuilds economics", () => {
    const { journal, root } = fixture();
    const team = createTeam(journal);
    const task = {
      attemptId: "10000000-0000-4000-8000-000000000001",
      blockedBy: [],
      createdAt: 101,
      description: "Inspect evidence",
      id: "task-1" as const,
      kind: "read" as const,
      ownerId: team.id,
      revision: 1,
      sequence: 1,
      status: "in_progress" as const,
      subject: "Inspect",
      updatedAt: 101,
      writeScopes: [],
    };
    const attempt = {
      id: task.attemptId,
      revision: 1,
      taskId: task.id,
      taskRevision: task.revision,
      ownerId: team.id,
      memberName: "lead" as const,
      role: "lead" as const,
      modelPolicy: { source: "observed" as const, provider: "openai", model: "gpt-5.6" },
      budgetState: "within" as const,
      status: "active" as const,
      usage: {
        availability: "known" as const,
        inputTokens: 100,
        outputTokens: 20,
        cacheReadTokens: 10,
        cacheWriteTokens: 0,
        turns: 1,
        toolCalls: 2,
      },
      actors: [
        {
          phase: "implementation" as const,
          memberName: "lead" as const,
          role: "lead" as const,
          modelPolicy: { source: "observed" as const, provider: "openai", model: "gpt-5.6" },
          observedModel: { provider: "openai", model: "gpt-5.6" },
          usage: {
            availability: "known" as const,
            inputTokens: 100,
            outputTokens: 20,
            cacheReadTokens: 10,
            cacheWriteTokens: 0,
            turns: 1,
            toolCalls: 2,
          },
          startedAt: 101,
        },
      ],
      startedAt: 101,
      lastProgressAt: 110,
      warningCodes: [],
      observedModel: { provider: "openai", model: "gpt-5.6" },
    };
    journal.append(team.id, { type: "attempt/started", data: { task, attempt } });
    expect(journal.get(team.id)?.attempts[0]).toMatchObject({
      id: attempt.id,
      usage: { inputTokens: 100, outputTokens: 20 },
    });
    journal.rebuildProjections();
    expect(journal.get(team.id)?.attempts[0]).toEqual(expect.objectContaining(attempt));
    journal.close();

    const database = new DatabaseSync(join(root, "swarm.sqlite"), { readOnly: true });
    expect(database.prepare("SELECT COUNT(*) AS count FROM swarm_attempts").get()).toEqual({
      count: 1,
    });
    const eventJson = JSON.stringify(
      database.prepare("SELECT type, payload_json FROM swarm_events ORDER BY seq").all(),
    );
    expect(eventJson).not.toMatch(/prompt|assistant response|credential|\/Users\//iu);
    database.close();
  });

  it("V163: commits events and projections atomically and rebuilds from the event log", () => {
    const { journal, root } = fixture();
    const created = createTeam(journal);
    expect(created.revision).toBe(1);

    journal.append(created.id, {
      type: "member/updated",
      data: {
        createdAt: 101,
        description: "Read-only analyst",
        id: "session-analyst",
        name: "analyst",
        phase: "active",
        role: "member",
      },
    });
    journal.append(created.id, {
      type: "task/updated",
      data: {
        attemptId: "attempt-1",
        blockedBy: [],
        createdAt: 102,
        description: "Inspect evidence",
        id: "task-1",
        kind: "read",
        ownerId: "session-analyst",
        revision: 1,
        sequence: 1,
        status: "in_progress",
        subject: "Inspect",
        updatedAt: 102,
        writeScopes: [],
      },
    });
    expect(journal.get(created.id)?.revision).toBe(3);

    journal.rebuildProjections();
    expect(journal.get(created.id)).toMatchObject({
      id: created.id,
      members: [{ id: created.id }, { id: "session-analyst" }],
      tasks: [{ id: "task-1", status: "in_progress" }],
    });
    expect(statSync(root).mode & 0o777).toBe(0o700);
    expect(statSync(join(root, "swarm.sqlite")).mode & 0o777).toBe(0o600);
    const workspaceKey = journal.workspaceKey(root);
    expect(workspaceKey).toMatch(/^swarmx--[0-9a-f]{64}$/u);
    expect(workspaceKey).not.toContain(root);
    journal.close();
    const reopened = new SwarmJournal(root);
    expect(reopened.workspaceKey(root)).toBe(workspaceKey);
    reopened.close();
  });

  it("V168: revokes every interrupted attempt instead of replaying work", () => {
    const { journal } = fixture();
    const team = createTeam(journal);
    journal.append(team.id, {
      type: "task/updated",
      data: {
        attemptId: "attempt-before-crash",
        blockedBy: [],
        createdAt: 101,
        description: "May already have changed files",
        id: "task-1",
        kind: "write",
        ownerId: team.id,
        revision: 1,
        sequence: 1,
        status: "in_progress",
        subject: "Mutate",
        updatedAt: 101,
        writeScopes: ["src"],
      },
    });
    expect(journal.recoverInterruptedTasks(200)).toBe(1);
    const recovered = journal.get(team.id)?.tasks[0];
    expect(recovered).not.toHaveProperty("attemptId");
    expect(recovered).toMatchObject({
      revision: 2,
      status: "needs_attention",
      updatedAt: 200,
    });
    expect(journal.recoverInterruptedTasks(300)).toBe(0);
    journal.close();
  });

  it("V194: recovers a verifying ledger attempt to needs_attention without replay", () => {
    const { journal } = fixture();
    const team = createTeam(journal);
    const attemptId = "10000000-0000-4000-8000-000000000001";
    const submission = {
      id: "20000000-0000-4000-8000-000000000001",
      attemptId,
      summary: "Bounded submission",
      artifactLocators: [],
      evidenceDigests: [],
      submittedAt: 110,
    };
    const task = {
      attemptId,
      blockedBy: [],
      createdAt: 101,
      description: "Verify after restart",
      id: "task-1" as const,
      kind: "read" as const,
      ownerId: team.id,
      revision: 1,
      sequence: 1,
      status: "in_progress" as const,
      subject: "Verify",
      updatedAt: 101,
      writeScopes: [],
    };
    const usage = {
      availability: "unknown" as const,
      inputTokens: 0,
      outputTokens: 0,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
      turns: 0,
      toolCalls: 0,
    };
    const attempt = {
      id: attemptId,
      revision: 1,
      taskId: task.id,
      taskRevision: task.revision,
      ownerId: team.id,
      memberName: "lead" as const,
      role: "lead" as const,
      modelPolicy: { source: "legacy-default" as const },
      budgetState: "unknown" as const,
      status: "active" as const,
      usage,
      actors: [
        {
          phase: "implementation" as const,
          memberName: "lead" as const,
          role: "lead" as const,
          modelPolicy: { source: "legacy-default" as const },
          usage,
          startedAt: 101,
        },
      ],
      startedAt: 101,
      lastProgressAt: 101,
      warningCodes: [],
    };
    journal.append(team.id, { type: "attempt/started", data: { task, attempt } });
    const submittedTask = {
      ...task,
      revision: 2,
      status: "submitted" as const,
      submission,
      updatedAt: 110,
    };
    const submittedAttempt = {
      ...attempt,
      revision: 2,
      status: "submitted" as const,
      submission,
      submittedAt: 110,
      lastProgressAt: 110,
      actors: attempt.actors.map((actor) => ({ ...actor, endedAt: 110 })),
    };
    journal.append(team.id, {
      type: "task/submitted",
      data: { task: submittedTask, attempt: submittedAttempt },
    });
    journal.append(team.id, {
      type: "verification/started",
      data: {
        task: {
          ...submittedTask,
          revision: 3,
          status: "verifying",
          verificationStartedById: team.id,
          verificationStartedAt: 120,
          updatedAt: 120,
        },
        attempt: {
          ...submittedAttempt,
          revision: 3,
          status: "verifying",
          lastProgressAt: 120,
          actors: [
            ...submittedAttempt.actors,
            {
              phase: "verification",
              memberName: "lead",
              role: "lead",
              modelPolicy: { source: "legacy-default" },
              usage,
              startedAt: 120,
            },
          ],
        },
      },
    });

    expect(journal.recoverInterruptedTasks(200)).toBe(1);
    expect(journal.get(team.id)?.tasks[0]).toMatchObject({ status: "needs_attention" });
    expect(journal.get(team.id)?.tasks[0]).not.toHaveProperty("attemptId");
    expect(journal.get(team.id)?.attempts[0]).toMatchObject({
      status: "interrupted",
      terminalReason: "Host recovered a non-terminal attempt",
    });
    journal.close();
  });

  it("V175/V179: recovers started effects and admissions as uncertain without replay", () => {
    const { journal, root } = fixture();
    const team = createTeam(journal);
    journal.append(team.id, {
      type: "effect/updated",
      data: {
        id: "10000000-0000-4000-8000-000000000001",
        revision: 1,
        callId: "call-before-crash",
        taskId: "task-1",
        taskRevision: 2,
        attemptId: "attempt-before-crash",
        ownerId: "session-member",
        toolName: "write",
        status: "started",
        createdAt: 110,
        updatedAt: 110,
      },
    });
    journal.append(team.id, {
      type: "effect/updated",
      data: {
        id: "10000000-0000-4000-8000-000000000002",
        revision: 1,
        callId: "call-settled-before-crash",
        taskId: "task-1",
        taskRevision: 2,
        attemptId: "settled-before-crash",
        ownerId: "session-member",
        toolName: "write",
        status: "succeeded",
        createdAt: 115,
        updatedAt: 115,
        resultDigest: `sha256:${"c".repeat(64)}`,
      },
    });
    journal.append(team.id, {
      type: "knowledge/admission-updated",
      data: {
        id: "20000000-0000-4000-8000-000000000002",
        revision: 1,
        taskId: "task-2",
        taskRevision: 2,
        attemptId: "admitted-before-crash",
        requestHash: `sha256:${"d".repeat(64)}`,
        targetKind: "science_evidence",
        sources: [{ kind: "science_entity", entityId: "30000000-0000-4000-8000-000000000001" }],
        verification: {
          status: "verified",
          method: "reproduced",
          verifiedAt: 100,
        },
        status: "committed",
        receipt: {
          kind: "science_evidence",
          entityId: "40000000-0000-4000-8000-000000000001",
          journalSequence: 7,
        },
        createdAt: 125,
        updatedAt: 125,
      },
    });
    journal.append(team.id, {
      type: "knowledge/admission-updated",
      data: {
        id: "20000000-0000-4000-8000-000000000001",
        revision: 1,
        taskId: "task-2",
        taskRevision: 2,
        attemptId: "knowledge-before-crash",
        requestHash: `sha256:${"a".repeat(64)}`,
        targetKind: "pkb_concept",
        sources: [{ kind: "reference", resource: "https://example.test/source" }],
        verification: {
          status: "verified",
          method: "source_reviewed",
          verifiedAt: 100,
        },
        status: "started",
        createdAt: 120,
        updatedAt: 120,
      },
    });
    journal.close();

    const recovered = new SwarmJournal(root);
    expect(recovered.recoverUncertainIntents(200)).toBe(2);
    expect(
      recovered
        .get(team.id)
        ?.effects.find((effect) => effect.id === "10000000-0000-4000-8000-000000000001"),
    ).toMatchObject({
      revision: 2,
      status: "uncertain",
    });
    expect(
      recovered
        .get(team.id)
        ?.admissions.find((admission) => admission.id === "20000000-0000-4000-8000-000000000001"),
    ).toMatchObject({
      revision: 2,
      status: "uncertain",
    });
    expect(
      recovered
        .get(team.id)
        ?.effects.find((effect) => effect.id === "10000000-0000-4000-8000-000000000002")?.status,
    ).toBe("succeeded");
    expect(
      recovered
        .get(team.id)
        ?.admissions.find((admission) => admission.id === "20000000-0000-4000-8000-000000000002")
        ?.status,
    ).toBe("committed");
    expect(recovered.recoverUncertainIntents(300)).toBe(0);
    recovered.close();
  });

  it("V163/V173: keeps archived history readable and rejects later events", () => {
    const { journal } = fixture();
    const team = createTeam(journal);
    journal.beginArchive(team.id, 199);
    const archived = journal.finishArchive(team.id, 200);
    expect(archived?.phase).toBe("archived");
    expect(journal.get(team.id)?.phase).toBe("archived");
    expect(() =>
      journal.append(team.id, {
        type: "member/updated",
        data: {
          createdAt: 201,
          description: "Too late",
          id: "session-late",
          name: "late",
          phase: "active",
          role: "member",
        },
      }),
    ).toThrow(/archived/iu);
    journal.close();
  });

  it("V164/V173: persists an archive fence across clients and finalizes only after exact cleanup", () => {
    const { journal, root } = fixture();
    const team = createTeam(journal);
    const member = {
      createdAt: 101,
      description: "in-flight member",
      id: "80000000-0000-4000-8000-000000000001",
      name: "in-flight",
      phase: "provisioning" as const,
      role: "legacy" as const,
      modelPolicy: { source: "legacy-default" as const },
    };
    journal.reserveMember(team.id, member, 2);
    const second = new SwarmJournal(root, { mode: "client" });
    second.beginArchive(team.id, 102);
    expect(() =>
      journal.reserveMember(
        team.id,
        {
          ...member,
          id: "80000000-0000-4000-8000-000000000002",
          name: "too-late",
        },
        3,
      ),
    ).toThrow(/archiv/iu);
    expect(() =>
      journal.activateProvisioningMember(team.id, { ...member, phase: "active" }),
    ).toThrow(/no longer active/iu);
    expect(
      journal.claimProvisioningMemberBinding(team.id, {
        workspaceKey: team.workspaceKey,
        runtime: "codex",
        memberId: member.id,
        handle: "codex:in-flight",
      }),
    ).toBe("archive_required");
    expect(second.finishArchive(team.id, 103)).toBeUndefined();
    second.close();
    journal.close();

    const recovered = new SwarmJournal(root);
    expect(recovered.get(team.id)?.archiveStartedAt).toBe(102);
    expect(() =>
      recovered.append(team.id, {
        type: "task/updated",
        data: {
          blockedBy: [],
          createdAt: 104,
          description: "too late",
          id: "task-1",
          kind: "read",
          revision: 1,
          sequence: 1,
          status: "pending",
          subject: "Too late",
          updatedAt: 104,
          writeScopes: [],
        },
      }),
    ).toThrow(/archive is in progress/iu);
    expect(
      recovered.retireBoundMemberForArchive(team.id, {
        workspaceKey: team.workspaceKey,
        runtime: "codex",
        memberId: member.id,
        handle: "codex:in-flight",
      }),
    ).toBe(true);
    const archived = recovered.finishArchive(team.id, 105);
    expect(archived).toMatchObject({ phase: "archived" });
    expect(archived?.members.find((candidate) => candidate.id === member.id)?.phase).toBe(
      "retired",
    );
    expect(recovered.listMemberBindings(team.workspaceKey, "codex")).toEqual([]);
    recovered.rebuildProjections();
    expect(recovered.get(team.id)).toMatchObject({ phase: "archived" });
    recovered.close();
  });
});
