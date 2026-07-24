import * as fs from "node:fs";
import { homedir } from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  appendMessages,
  archiveProjectSessions,
  archiveSession,
  createSession,
  deleteSession,
  listSessions,
  loadSession,
  saveSession,
  setSessionPinned,
} from "../src/session.js";
import type { MessageChunk } from "../src/types.js";

const sessionsDir = path.join(homedir(), ".swarmx", "sessions");

describe("Session", () => {
  const savedIds: string[] = [];

  afterEach(() => {
    for (const id of savedIds) {
      deleteSession(id);
    }
  });

  it("creates a session with auto-generated id", () => {
    const session = createSession("agent", "swarmx", "gpt-4");
    savedIds.push(session.id);

    expect(session.id).toBeTruthy();
    expect(session.agentName).toBe("agent");
    expect(session.harness).toBe("swarmx");
    expect(session.model).toBe("gpt-4");
    expect(session.permissionMode).toBe("inherit");
    expect(session.messages).toEqual([]);
    expect(session.pinned).toBe(false);
    expect(session.createdAt).toBeTruthy();
  });

  it("persists project identity and working directory", () => {
    const session = createSession("agent", "swarmx", "gpt-4", {
      projectId: "project-1",
      cwd: "/workspace/project-1",
      permissionMode: "auto",
    });
    savedIds.push(session.id);
    saveSession(session);

    expect(loadSession(session.id)).toMatchObject({
      projectId: "project-1",
      cwd: "/workspace/project-1",
      permissionMode: "auto",
    });
  });

  it("V457 migrates legacy sessions to inherit and rejects unsupported overrides", () => {
    fs.mkdirSync(sessionsDir, { recursive: true });
    const legacy = {
      id: "legacy-permission-session",
      title: "Legacy",
      agentName: "agent",
      harness: "swarmx",
      pinned: false,
      messages: [],
      createdAt: "2026-07-18T00:00:00.000Z",
      updatedAt: "2026-07-18T00:00:00.000Z",
    };
    savedIds.push(legacy.id);
    fs.writeFileSync(path.join(sessionsDir, `${legacy.id}.json`), JSON.stringify(legacy), "utf8");
    expect(loadSession(legacy.id)?.permissionMode).toBe("inherit");

    expect(() => saveSession({ ...legacy, permissionMode: "restricted" } as never)).toThrow();
  });

  it("saves and loads a session", () => {
    const session = createSession("test", "opencode");
    savedIds.push(session.id);

    saveSession(session);
    const loaded = loadSession(session.id);
    expect(loaded).not.toBeNull();
    if (!loaded) throw new Error("saved session did not load");
    expect(loaded.id).toBe(session.id);
    expect(loaded.agentName).toBe("test");
  });

  it("lists all sessions", () => {
    const s1 = createSession("a", "swarmx");
    const s2 = createSession("b", "claude_code");
    savedIds.push(s1.id, s2.id);

    saveSession(s1);
    saveSession(s2);

    const all = listSessions();
    const ids = all.map((s) => s.id);
    expect(ids).toContain(s1.id);
    expect(ids).toContain(s2.id);
  });

  it("persists pin state and lists pinned sessions first", () => {
    const olderPinned = createSession("pinned", "swarmx");
    const newerUnpinned = createSession("recent", "swarmx");
    savedIds.push(olderPinned.id, newerUnpinned.id);
    olderPinned.updatedAt = "2026-01-01T00:00:00.000Z";
    newerUnpinned.updatedAt = "2026-01-02T00:00:00.000Z";
    saveSession(olderPinned);
    saveSession(newerUnpinned);

    expect(setSessionPinned(olderPinned.id, true)).toMatchObject({ pinned: true });
    expect(loadSession(olderPinned.id)?.pinned).toBe(true);
    expect(listSessions().filter((session) => savedIds.includes(session.id))[0]?.id).toBe(
      olderPinned.id,
    );
  });

  it("archives every task in a project without deleting its persisted history", () => {
    const projectSession = createSession("a", "swarmx", undefined, {
      projectId: "project-archive",
      cwd: "/workspace/archive",
    });
    const otherSession = createSession("b", "swarmx", undefined, {
      projectId: "project-other",
      cwd: "/workspace/other",
    });
    savedIds.push(projectSession.id, otherSession.id);
    saveSession(projectSession);
    saveSession(otherSession);

    expect(
      archiveProjectSessions({ projectId: "project-archive", cwd: "/workspace/archive" }),
    ).toBe(1);
    expect(listSessions().map((session) => session.id)).not.toContain(projectSession.id);
    expect(listSessions().map((session) => session.id)).toContain(otherSession.id);
    expect(loadSession(projectSession.id)?.archivedAt).toBeTruthy();
    expect(listSessions({ includeArchived: true }).map((session) => session.id)).toContain(
      projectSession.id,
    );
  });

  it("archives one task without deleting its persisted history", () => {
    const session = createSession("archive", "swarmx");
    savedIds.push(session.id);
    session.messages.push({ role: "user", content: "keep me", kind: "message" });
    saveSession(session);

    const archived = archiveSession(session.id);

    expect(archived?.archivedAt).toBeTruthy();
    expect(loadSession(session.id)?.messages).toEqual(session.messages);
    expect(listSessions().map((candidate) => candidate.id)).not.toContain(session.id);
    expect(listSessions({ includeArchived: true }).map((candidate) => candidate.id)).toContain(
      session.id,
    );
  });

  it("deletes a session", () => {
    const session = createSession("del", "swarmx");
    saveSession(session);

    const deleted = deleteSession(session.id);
    expect(deleted).toBe(true);
    expect(loadSession(session.id)).toBeNull();
  });

  it("appends messages to session", () => {
    const session = createSession("chat", "opencode");
    savedIds.push(session.id);
    saveSession(session);

    const msgs: MessageChunk[] = [
      { role: "user", content: "hello", kind: "message" },
      { role: "assistant", content: "hi there", kind: "message" },
    ];

    appendMessages(session.id, msgs);

    const loaded = loadSession(session.id);
    expect(loaded).not.toBeNull();
    if (!loaded) throw new Error("session with appended messages did not load");
    expect(loaded.messages).toHaveLength(2);
    expect(loaded.messages[0].content).toBe("hello");
    expect(loaded.messages[1].content).toBe("hi there");
  });

  it("returns null for nonexistent session", () => {
    expect(loadSession("nonexistent-id")).toBeNull();
  });

  it("returns false for deleting nonexistent session", () => {
    expect(deleteSession("nonexistent-id")).toBe(false);
  });

  it("returns null for archiving a nonexistent session", () => {
    expect(archiveSession("nonexistent-id")).toBeNull();
  });
});
