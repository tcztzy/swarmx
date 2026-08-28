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

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { force: true, recursive: true })));
});

describe("Swarm journal", () => {
  it("V194: migrates a v1 event fixture to v3 with explicit legacy defaults", () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-swarm-journal-v1-"));
    roots.push(root);
    const databasePath = join(root, "swarm.sqlite");
    const database = new DatabaseSync(databasePath);
    database.exec(`
      CREATE TABLE swarm_migrations (version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL) STRICT;
      INSERT INTO swarm_migrations(version, applied_at) VALUES (1, 1);
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
    ).toEqual([{ version: 1 }, { version: 2 }, { version: 3 }]);
    expect(
      inspected
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'swarm_attempts'")
        .get(),
    ).toEqual({ name: "swarm_attempts" });
    inspected.close();
  });

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
    const archived = journal.append(team.id, { type: "team/archived", data: { archivedAt: 200 } });
    expect(archived.phase).toBe("archived");
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
});
