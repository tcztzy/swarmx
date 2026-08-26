import { mkdtempSync, statSync } from "node:fs";
import { rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
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
