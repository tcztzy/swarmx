import * as fs from "node:fs";
import { homedir } from "node:os";
import * as path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  appendMessages,
  createSession,
  deleteSession,
  listSessions,
  loadSession,
  saveSession,
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
    expect(session.messages).toEqual([]);
    expect(session.createdAt).toBeTruthy();
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
});
