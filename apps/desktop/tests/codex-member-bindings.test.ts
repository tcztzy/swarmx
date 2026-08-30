import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { SwarmJournal } from "@swarmx/dsh-swarm";
import { afterEach, describe, expect, it } from "vitest";
import { CodexMemberBindingStore } from "../src/runtime/codex/member-bindings.js";

const roots: string[] = [];
const workspaceA = `swarmx--${"a".repeat(64)}`;
const workspaceB = `swarmx--${"b".repeat(64)}`;

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("Codex Swarm member bindings", () => {
  it("keeps concurrent per-member claims and exact conditional releases", () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-bindings-"));
    roots.push(root);
    const firstJournal = new SwarmJournal(root);
    const secondJournal = new SwarmJournal(root);
    const first = new CodexMemberBindingStore(firstJournal, workspaceA);
    const second = new CodexMemberBindingStore(secondJournal, workspaceA);
    const firstId = randomUUID();
    const secondId = randomUUID();

    expect(first.claim({ id: firstId, conversationId: "codex:thread-1" })).toBe("created");
    expect(second.claim({ id: secondId, conversationId: "codex:thread-2" })).toBe("created");
    expect(first.claim({ id: firstId, conversationId: "codex:thread-1" })).toBe("existing");
    expect(first.list()).toEqual(
      expect.arrayContaining([
        { id: firstId, conversationId: "codex:thread-1" },
        { id: secondId, conversationId: "codex:thread-2" },
      ]),
    );

    expect(first.release({ id: firstId, conversationId: "codex:thread-1" })).toBe(true);
    expect(first.claim({ id: firstId, conversationId: "codex:thread-new" })).toBe("created");
    expect(second.release({ id: firstId, conversationId: "codex:thread-1" })).toBe(false);
    expect(second.get(firstId)).toEqual({ id: firstId, conversationId: "codex:thread-new" });

    firstJournal.rebuildProjections();
    expect(second.list()).toHaveLength(2);
    firstJournal.close();
    secondJournal.close();
    expect(statSync(join(root, "swarm.sqlite")).mode & 0o777).toBe(0o600);
  });

  it("rejects member, native Thread, and cross-workspace identity conflicts", () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-binding-conflict-"));
    roots.push(root);
    const journal = new SwarmJournal(root);
    const first = new CodexMemberBindingStore(journal, workspaceA);
    const second = new CodexMemberBindingStore(journal, workspaceB);
    const memberId = randomUUID();
    first.claim({ id: memberId, conversationId: "codex:thread-shared" });

    expect(() => first.claim({ id: memberId, conversationId: "codex:thread-other" })).toThrow(
      "member already belongs",
    );
    expect(() => first.claim({ id: randomUUID(), conversationId: "codex:thread-shared" })).toThrow(
      "native Thread already belongs",
    );
    expect(() => second.claim({ id: randomUUID(), conversationId: "codex:thread-shared" })).toThrow(
      "native Thread already belongs",
    );
    journal.close();
  });

  it("rejects malformed persisted rows on startup", () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-binding-invalid-"));
    roots.push(root);
    const journal = new SwarmJournal(root);
    const databasePath = journal.databasePath;
    journal.close();
    const database = new DatabaseSync(databasePath);
    database
      .prepare(
        `INSERT INTO swarm_member_bindings(workspace_key, runtime, member_id, handle)
         VALUES (?, 'codex', ?, '')`,
      )
      .run(workspaceA, randomUUID());
    database.close();

    const reopened = new SwarmJournal(root);
    expect(() => new CodexMemberBindingStore(reopened, workspaceA)).toThrow(
      "Codex Swarm member binding store is invalid",
    );
    reopened.close();
  });
});
