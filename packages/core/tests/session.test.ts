import * as fs from "node:fs";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import {
  appendMessages,
  appendTransientSessionMessages,
  archiveProjectSessions,
  archiveSession,
  createSession,
  createTransientSessionFork,
  deleteSession,
  editSessionUserMessage,
  editTransientSessionUserMessage,
  forkSession,
  listSessionSummaries,
  listSessions,
  loadSession,
  promoteTransientSessionFork,
  saveSession,
  setSessionPinned,
  transientSessionModelMessages,
  updateSessionTitle,
} from "../src/session.js";
import type { MessageChunk } from "../src/types.js";

const sessionsDir = fs.mkdtempSync(path.join(tmpdir(), "swarmx-session-tests-"));
const originalSessionsDir = process.env.SWARMX_SESSIONS_DIR;

function filesUnder(root: string): string[] {
  if (!fs.existsSync(root)) return [];
  return fs
    .readdirSync(root, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => path.join(entry.parentPath, entry.name))
    .sort();
}

function sessionJsonlPath(root: string, id: string): string {
  const matches = filesUnder(root).filter((filePath) => path.basename(filePath) === `${id}.jsonl`);
  if (matches.length !== 1 || !matches[0]) {
    throw new Error(`Expected exactly one JSONL file for Session ${id}, found ${matches.length}.`);
  }
  return matches[0];
}

describe("Session", () => {
  const savedIds: string[] = [];

  beforeAll(() => {
    process.env.SWARMX_SESSIONS_DIR = sessionsDir;
  });

  afterEach(() => {
    for (const id of savedIds) {
      deleteSession(id);
    }
  });

  afterAll(() => {
    if (originalSessionsDir === undefined) {
      Reflect.deleteProperty(process.env, "SWARMX_SESSIONS_DIR");
    } else process.env.SWARMX_SESSIONS_DIR = originalSessionsDir;
    fs.rmSync(sessionsDir, { recursive: true, force: true });
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

  it("stores Sessions in working-directory partitions with a per-Project index", () => {
    const first = createSession("agent", "swarmx", undefined, {
      projectId: "project-a",
      cwd: "/workspace/project-a",
    });
    const second = createSession("agent", "swarmx", undefined, {
      projectId: "project-b",
      cwd: "/workspace/project-b",
    });
    const sameProject = createSession("agent", "swarmx", undefined, {
      projectId: "replacement-bookmark-id",
      cwd: "/workspace/project-a",
    });
    savedIds.push(first.id, second.id, sameProject.id);

    saveSession(first);
    saveSession(second);
    saveSession(sameProject);

    const firstDirectory = path.dirname(sessionJsonlPath(sessionsDir, first.id));
    const secondDirectory = path.dirname(sessionJsonlPath(sessionsDir, second.id));
    const sameProjectDirectory = path.dirname(sessionJsonlPath(sessionsDir, sameProject.id));
    expect(firstDirectory).not.toBe(sessionsDir);
    expect(firstDirectory).not.toBe(secondDirectory);
    expect(sameProjectDirectory).toBe(firstDirectory);
    expect(fs.existsSync(path.join(firstDirectory, "sessions-index.json"))).toBe(true);
    expect(fs.existsSync(path.join(secondDirectory, "sessions-index.json"))).toBe(true);
    expect(
      JSON.parse(fs.readFileSync(path.join(firstDirectory, "sessions-index.json"), "utf8")),
    ).toMatchObject({
      version: 1,
      entries: expect.arrayContaining([
        expect.objectContaining({ sessionId: first.id }),
        expect.objectContaining({ sessionId: sameProject.id }),
      ]),
    });
    expect(loadSession(first.id)?.cwd).toBe("/workspace/project-a");
    expect(loadSession(second.id)?.cwd).toBe("/workspace/project-b");
  });

  it("stores projectless Sessions in the Recents partition", () => {
    const session = createSession("agent", "swarmx");
    savedIds.push(session.id);
    saveSession(session);

    expect(path.basename(path.dirname(sessionJsonlPath(sessionsDir, session.id)))).toBe(
      "__recents__",
    );
  });

  it("keeps a flat JSONL readable and relocates it only when mutated", () => {
    const session = createSession("legacy-flat", "swarmx", undefined, {
      projectId: "legacy-project",
      cwd: "/workspace/legacy-project",
    });
    savedIds.push(session.id);
    const flatPath = path.join(sessionsDir, `${session.id}.jsonl`);
    fs.writeFileSync(
      flatPath,
      `${JSON.stringify({
        schemaVersion: 1,
        type: "session_created",
        timestamp: session.createdAt,
        session,
      })}\n`,
      "utf8",
    );

    expect(listSessionSummaries({ includeArchived: true }).map((item) => item.id)).toContain(
      session.id,
    );
    expect(fs.existsSync(flatPath)).toBe(true);

    expect(updateSessionTitle(session.id, "Relocated Session")).toBe(true);
    expect(fs.existsSync(flatPath)).toBe(false);
    expect(sessionJsonlPath(sessionsDir, session.id)).not.toBe(flatPath);
    expect(loadSession(session.id)?.title).toBe("Relocated Session");
  });

  it("rejects unsupported permission overrides", () => {
    const session = createSession("agent", "swarmx");
    expect(() => saveSession({ ...session, permissionMode: "restricted" } as never)).toThrow();
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

  it("persists canonical attachment metadata with user messages", () => {
    const session = createSession("media", "swarmx");
    savedIds.push(session.id);
    session.messages.push({
      role: "user",
      content: "Review this diagram",
      kind: "message",
      attachments: [
        {
          id: "diagram",
          name: "diagram.png",
          kind: "image",
          mimeType: "image/png",
          sizeBytes: 42,
          uri: "file:///managed/diagram.png",
          source: "user",
        },
      ],
    });

    saveSession(session);

    expect(loadSession(session.id)?.messages[0]?.attachments).toEqual(
      session.messages[0]?.attachments,
    );
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
    expect(loaded.messages[0].createdAt).toBeTruthy();
    expect(loaded.messages[1].createdAt).toBe(loaded.messages[0].createdAt);
  });

  it("appends message deltas without rewriting prior Session JSONL bytes", () => {
    const session = createSession("incremental", "swarmx");
    savedIds.push(session.id);
    saveSession(session);
    const rolloutPath = sessionJsonlPath(sessionsDir, session.id);
    const before = fs.readFileSync(rolloutPath);

    expect(
      appendMessages(session.id, [{ role: "user", content: "incremental write", kind: "message" }]),
    ).toBe(true);

    const after = fs.readFileSync(rolloutPath);
    expect(after.subarray(0, before.length)).toEqual(before);
    expect(after.length).toBeGreaterThan(before.length);
    expect(fs.existsSync(path.join(sessionsDir, `${session.id}.json`))).toBe(false);
  });

  it("keeps appended history when a stale Session saves metadata", () => {
    const stale = createSession("stale", "swarmx");
    savedIds.push(stale.id);
    saveSession(stale);
    appendMessages(stale.id, [{ role: "user", content: "concurrent append", kind: "message" }]);

    stale.title = "Metadata from stale writer";
    saveSession(stale);

    expect(loadSession(stale.id)).toMatchObject({
      title: "Metadata from stale writer",
      messages: [{ role: "user", content: "concurrent append", kind: "message" }],
    });
  });

  it("edits only the latest user turn while preserving prior JSONL bytes for analysis", () => {
    const session = createSession("editing", "swarmx");
    savedIds.push(session.id);
    session.messages = [
      { role: "user", content: "Original request", kind: "message" },
      { role: "assistant", content: "Original reply", kind: "message" },
      { role: "user", content: "Later request", kind: "message" },
      { role: "assistant", content: "Later reply", kind: "message" },
    ];
    session.externalAcpSession = {
      sessionId: "editing-external-session",
      harnessId: "codex",
      modelId: "gpt-5",
      agentProfileId: "desktop-gpt-5",
      createdAt: "2026-07-27T00:00:00.000Z",
      updatedAt: "2026-07-27T00:00:00.000Z",
    };
    saveSession(session);
    const rolloutPath = sessionJsonlPath(sessionsDir, session.id);
    const before = fs.readFileSync(rolloutPath);

    const edited = editSessionUserMessage({
      id: session.id,
      messageIndex: 2,
      expectedMessages: session.messages,
      content: "  Revised request  ",
    });

    expect(edited?.messages).toMatchObject([
      { role: "user", content: "Original request", kind: "message" },
      { role: "assistant", content: "Original reply", kind: "message" },
      { role: "user", content: "Revised request", kind: "message" },
    ]);
    expect(edited?.messages[2]?.createdAt).toBeTruthy();
    expect(edited?.externalAcpSession).toBeUndefined();
    expect(loadSession(session.id)?.messages).toEqual(edited?.messages);
    const after = fs.readFileSync(rolloutPath);
    expect(after.subarray(0, before.length)).toEqual(before);
    const finalRecord = JSON.parse(after.toString("utf8").trim().split("\n").at(-1) ?? "{}") as {
      type?: string;
      reason?: string;
      replacedFromIndex?: number;
      replacedMessageCount?: number;
    };
    expect(finalRecord).toMatchObject({
      type: "messages_replaced",
      reason: "edit_last_user_message",
      replacedFromIndex: 2,
      replacedMessageCount: 2,
    });
    expect(after.toString("utf8")).toContain("Later reply");
  });

  it("rejects edits to an older user turn", () => {
    const session = createSession("editing-history", "swarmx");
    savedIds.push(session.id);
    session.messages = [
      { role: "user", content: "Original request", kind: "message" },
      { role: "assistant", content: "Original reply", kind: "message" },
      { role: "user", content: "Latest request", kind: "message" },
      { role: "assistant", content: "Latest reply", kind: "message" },
    ];
    saveSession(session);

    expect(() =>
      editSessionUserMessage({
        id: session.id,
        messageIndex: 0,
        expectedMessages: session.messages,
        content: "Revised old request",
      }),
    ).toThrow("Only the latest user message");
    expect(loadSession(session.id)?.messages).toEqual(session.messages);
  });

  it("refuses to edit stale history without overwriting concurrent messages", () => {
    const session = createSession("editing-conflict", "swarmx");
    savedIds.push(session.id);
    session.messages = [{ role: "user", content: "Original request", kind: "message" }];
    saveSession(session);
    const expectedMessages = [...session.messages];
    appendMessages(session.id, [
      { role: "assistant", content: "Concurrent reply", kind: "message" },
    ]);

    expect(() =>
      editSessionUserMessage({
        id: session.id,
        messageIndex: 0,
        expectedMessages,
        content: "Revised request",
      }),
    ).toThrow("Session history changed");
    expect(loadSession(session.id)?.messages).toMatchObject([
      { role: "user", content: "Original request", kind: "message" },
      { role: "assistant", content: "Concurrent reply", kind: "message" },
    ]);
  });

  it("rejects empty edits and non-user message targets", () => {
    const session = createSession("editing-validation", "swarmx");
    savedIds.push(session.id);
    session.messages = [
      { role: "user", content: "Original request", kind: "message" },
      { role: "assistant", content: "Original reply", kind: "message" },
    ];
    saveSession(session);

    expect(() =>
      editSessionUserMessage({
        id: session.id,
        messageIndex: 0,
        expectedMessages: session.messages,
        content: "   ",
      }),
    ).toThrow("cannot be empty");
    expect(() =>
      editSessionUserMessage({
        id: session.id,
        messageIndex: 1,
        expectedMessages: session.messages,
        content: "Revised reply",
      }),
    ).toThrow("Only user messages");
  });

  it("continues from a completed assistant message in a separate local Session", () => {
    const source = createSession("forking", "swarmx", "model-1", {
      projectId: "project-1",
      cwd: "/workspace/project-1",
      permissionMode: "auto",
    });
    savedIds.push(source.id);
    source.title = "Original task";
    source.builtinTools = { style: "kimi_code", revision: 1, source: "settings" };
    source.acpSessionId = "runtime-owned-session";
    source.externalAcpSession = {
      sessionId: "external-session",
      harnessId: "codex",
      modelId: "model-1",
      agentProfileId: "desktop-model-1",
      cwd: "/workspace/project-1",
      createdAt: "2026-07-27T00:00:00.000Z",
      updatedAt: "2026-07-27T00:00:00.000Z",
    };
    source.pinned = true;
    source.messages = [
      { role: "user", content: "First request", kind: "message" },
      { role: "assistant", content: "First reply", kind: "message" },
      { role: "user", content: "Later request", kind: "message" },
      { role: "assistant", content: "Later reply", kind: "message" },
    ];
    saveSession(source);

    const forked = forkSession({
      id: source.id,
      throughMessageIndex: 1,
      expectedMessages: source.messages,
    });
    if (!forked) throw new Error("forked Session was not created");
    savedIds.push(forked.id);

    expect(forked).toMatchObject({
      title: "Original task (continued)",
      agentName: source.agentName,
      harness: source.harness,
      model: source.model,
      builtinTools: source.builtinTools,
      projectId: source.projectId,
      cwd: source.cwd,
      permissionMode: "auto",
      pinned: false,
      forkedFrom: {
        sessionId: source.id,
        messageIndex: 1,
      },
      messages: source.messages.slice(0, 2),
    });
    expect(forked.id).not.toBe(source.id);
    expect(forked.acpSessionId).toBeUndefined();
    expect(forked.externalAcpSession).toBeUndefined();
    expect(loadSession(source.id)?.messages).toEqual(source.messages);
    expect(loadSession(forked.id)?.messages).toEqual(source.messages.slice(0, 2));
  });

  it("creates an anchored transient fork without writing or listing another Session", () => {
    const source = createSession("side-chat", "swarmx", "model-1");
    savedIds.push(source.id);
    source.messages = [
      { role: "user", content: "First request", kind: "message" },
      { role: "assistant", content: "First reply", kind: "message" },
      { role: "user", content: "Second request", kind: "message" },
      { role: "assistant", content: "Second reply", kind: "message" },
    ];
    saveSession(source);
    const filesBefore = filesUnder(sessionsDir);
    const listedBefore = listSessions().map((session) => session.id);

    const transient = createTransientSessionFork({
      id: source.id,
      throughMessageIndex: 1,
      expectedMessages: source.messages,
    });

    expect(transient).toMatchObject({
      parentSessionId: source.id,
      anchor: {
        parentSessionId: source.id,
        messageIndex: 1,
        messageCount: 2,
      },
      anchorMessages: source.messages.slice(0, 2),
      messages: [],
      runState: "idle",
    });
    expect(filesUnder(sessionsDir)).toEqual(filesBefore);
    expect(listSessions().map((session) => session.id)).toEqual(listedBefore);
  });

  it("anchors the effective projection so an edited message never revives in a side chat", () => {
    const source = createSession("side-projection", "swarmx");
    savedIds.push(source.id);
    source.messages = [
      { role: "user", content: "Old request", kind: "message" },
      { role: "assistant", content: "Old reply", kind: "message" },
    ];
    saveSession(source);
    const edited = editSessionUserMessage({
      id: source.id,
      messageIndex: 0,
      expectedMessages: source.messages,
      content: "Replacement request",
    });
    if (!edited) throw new Error("edited Session was not created");

    const transient = createTransientSessionFork({
      id: source.id,
      throughMessageIndex: 0,
      expectedMessages: edited.messages,
    });

    expect(transient?.anchorMessages).toEqual([
      expect.objectContaining({ content: "Replacement request" }),
    ]);
    expect(JSON.stringify(transient)).not.toContain("Old request");
    expect(JSON.stringify(transient)).not.toContain("Old reply");
  });

  it("keeps host receipts in a transient anchor without replaying them to the model", () => {
    const source = createSession("side-receipts", "swarmx");
    savedIds.push(source.id);
    source.messages = [
      { role: "user", content: "Parent request", kind: "message" },
      {
        role: "system",
        content: "stale-project-revision",
        kind: "message",
        render: { source: "project_bootstrap_receipt" },
      },
      {
        role: "system",
        content: "stale-memory-summary",
        kind: "message",
        render: { source: "personal_memory_receipt" },
      },
      { role: "assistant", content: "Parent reply", kind: "message" },
    ];
    saveSession(source);
    const transient = createTransientSessionFork({
      id: source.id,
      throughMessageIndex: 3,
      expectedMessages: source.messages,
    });
    if (!transient) throw new Error("transient fork was not created");

    expect(transient.anchorMessages).toHaveLength(4);
    expect(transientSessionModelMessages(transient)).toEqual([
      { role: "user", content: "Parent request" },
      { role: "assistant", content: "Parent reply" },
    ]);
  });

  it("keeps multiple transient forks and their edits independently in memory", () => {
    const source = createSession("side-independent", "swarmx");
    savedIds.push(source.id);
    source.messages = [
      { role: "user", content: "Parent request", kind: "message" },
      { role: "assistant", content: "Parent reply", kind: "message" },
    ];
    saveSession(source);
    const first = createTransientSessionFork({
      id: source.id,
      throughMessageIndex: 1,
      expectedMessages: source.messages,
    });
    const second = createTransientSessionFork({
      id: source.id,
      throughMessageIndex: 1,
      expectedMessages: source.messages,
    });
    if (!first || !second) throw new Error("transient forks were not created");
    const firstWithTurn = appendTransientSessionMessages(first, [
      { role: "user", content: "First side question", kind: "message" },
      { role: "assistant", content: "First side answer", kind: "message" },
    ]);
    const secondWithTurn = appendTransientSessionMessages(
      second,
      [{ role: "user", content: "Second side question", kind: "message" }],
      {
        contextChips: [
          {
            id: "selection-1",
            text: "Selected parent text",
            createdAt: "2026-07-26T00:00:00.000Z",
          },
        ],
      },
    );
    const editedFirst = editTransientSessionUserMessage(firstWithTurn, 0, "Revised first question");

    expect(first.id).not.toBe(second.id);
    expect(editedFirst.messages).toEqual([
      expect.objectContaining({ content: "Revised first question" }),
    ]);
    expect(secondWithTurn.messages).toHaveLength(1);
    expect(transientSessionModelMessages(secondWithTurn).at(-1)?.content).toContain(
      "Selected parent text",
    );
    expect(JSON.stringify(secondWithTurn)).not.toContain("First side");
  });

  it("persists only after promotion and leaves the parent projection unchanged", () => {
    const source = createSession("side-promote", "swarmx", "model-1", {
      projectId: "project-1",
      cwd: "/workspace/project-1",
      permissionMode: "auto",
    });
    savedIds.push(source.id);
    source.builtinTools = { style: "claude_code", revision: 1, source: "model" };
    source.messages = [
      { role: "user", content: "Parent request", kind: "message" },
      { role: "assistant", content: "Parent reply", kind: "message" },
      { role: "user", content: "Later parent request", kind: "message" },
    ];
    saveSession(source);
    const transient = createTransientSessionFork({
      id: source.id,
      throughMessageIndex: 1,
      expectedMessages: source.messages,
    });
    if (!transient) throw new Error("transient fork was not created");
    const completed = appendTransientSessionMessages(transient, [
      { role: "user", content: "Side request", kind: "message" },
      { role: "assistant", content: "Side reply", kind: "message" },
    ]);
    expect(listSessions().map((session) => session.id)).toEqual([source.id]);

    const promoted = promoteTransientSessionFork({ transient: completed });
    savedIds.push(promoted.id);

    expect(promoted).toMatchObject({
      forkedFrom: { sessionId: source.id, messageIndex: 1 },
      projectId: source.projectId,
      cwd: source.cwd,
      permissionMode: source.permissionMode,
      builtinTools: source.builtinTools,
      messages: [...source.messages.slice(0, 2), ...completed.messages],
    });
    expect(promoted.acpSessionId).toBeUndefined();
    expect(loadSession(promoted.id)?.messages).toEqual(promoted.messages);
    expect(loadSession(source.id)?.messages).toEqual(source.messages);
    expect(
      listSessions()
        .map((session) => session.id)
        .sort(),
    ).toEqual([source.id, promoted.id].sort());
  });

  it("ignores unsupported Session JSON files", () => {
    const id = "unsupported-json-session";
    const filePath = path.join(sessionsDir, `${id}.json`);
    fs.mkdirSync(sessionsDir, { recursive: true });
    fs.writeFileSync(
      filePath,
      JSON.stringify({
        id,
        title: "Unsupported JSON",
        agentName: "agent",
        harness: "swarmx",
        messages: [],
        createdAt: "2026-07-01T00:00:00.000Z",
        updatedAt: "2026-07-01T00:01:00.000Z",
      }),
      "utf8",
    );

    try {
      expect(loadSession(id)).toBeNull();
      expect(
        listSessionSummaries({ includeArchived: true }).some((summary) => summary.id === id),
      ).toBe(false);
    } finally {
      fs.unlinkSync(filePath);
    }
  });

  it("rebuilds an index containing unsupported JSON sources", () => {
    const root = fs.mkdtempSync(path.join(tmpdir(), "swarmx-json-only-index-"));
    fs.writeFileSync(
      path.join(root, "sessions-index.json"),
      JSON.stringify({
        version: 1,
        entries: [
          {
            sessionId: "unsupported-json-session",
            sourceBytes: 1,
            sourceMtimeMs: 1,
            sourceFormat: "json",
            summary: {
              id: "unsupported-json-session",
              title: "Unsupported JSON",
              agentName: "agent",
              harness: "swarmx",
              permissionMode: "inherit",
              pinned: false,
              createdAt: "2026-07-01T00:00:00.000Z",
              updatedAt: "2026-07-01T00:01:00.000Z",
              messageCount: 0,
            },
          },
        ],
      }),
      "utf8",
    );

    try {
      expect(listSessionSummaries({ sessionsDir: root, includeArchived: true })).toEqual([]);
      expect(JSON.parse(fs.readFileSync(path.join(root, "sessions-index.json"), "utf8"))).toEqual({
        version: 1,
        entries: [],
      });
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it("replays a valid prefix after a torn tail but blocks another append", () => {
    const session = createSession("torn", "swarmx");
    savedIds.push(session.id);
    session.messages.push({ role: "user", content: "safe prefix", kind: "message" });
    saveSession(session);
    const rolloutPath = sessionJsonlPath(sessionsDir, session.id);
    fs.appendFileSync(rolloutPath, '{"schemaVersion":1,"type":"messages_appended"');

    expect(loadSession(session.id)?.messages).toEqual(session.messages);
    expect(() =>
      appendMessages(session.id, [
        { role: "assistant", content: "must not append", kind: "message" },
      ]),
    ).toThrow(/torn final record/i);
  });

  it("rejects newline-terminated corrupt records instead of skipping them", () => {
    const session = createSession("corrupt", "swarmx");
    savedIds.push(session.id);
    saveSession(session);
    const rolloutPath = sessionJsonlPath(sessionsDir, session.id);
    fs.appendFileSync(rolloutPath, "{bad json}\n");

    expect(loadSession(session.id)).toBeNull();
    expect(() =>
      appendMessages(session.id, [
        { role: "assistant", content: "must not append", kind: "message" },
      ]),
    ).toThrow(/line 2/i);
  });

  it("invalidates a cached Session when same-size on-disk data changes", () => {
    const session = createSession("same-size-corruption", "swarmx");
    savedIds.push(session.id);
    saveSession(session);
    const rolloutPath = sessionJsonlPath(sessionsDir, session.id);
    expect(loadSession(session.id)?.id).toBe(session.id);
    const valid = fs.readFileSync(rolloutPath, "utf8");
    const corrupt = valid.replace("session_created", "session_Xreated");
    expect(Buffer.byteLength(corrupt)).toBe(Buffer.byteLength(valid));
    fs.writeFileSync(rolloutPath, corrupt, "utf8");
    const future = new Date(Date.now() + 60_000);
    fs.utimesSync(rolloutPath, future, future);

    expect(loadSession(session.id)).toBeNull();
  });

  it("lists indexed Session summaries without message bodies", () => {
    const session = createSession("indexed", "swarmx");
    savedIds.push(session.id);
    session.messages.push(
      { role: "user", content: "one", kind: "message" },
      { role: "assistant", content: "two", kind: "message" },
    );
    saveSession(session);

    const summary = listSessionSummaries({ includeArchived: true }).find(
      (candidate) => candidate.id === session.id,
    );
    expect(summary).toMatchObject({
      id: session.id,
      messageCount: 2,
      title: session.title,
    });
    expect(summary).not.toHaveProperty("messages");
    expect(
      fs.existsSync(
        path.join(path.dirname(sessionJsonlPath(sessionsDir, session.id)), "sessions-index.json"),
      ),
    ).toBe(true);
  });

  it("reconciles same-size JSONL metadata changes from source modification time", () => {
    const root = fs.mkdtempSync(path.join(tmpdir(), "swarmx-session-index-mtime-"));
    const session = {
      id: "jsonl-index-mtime",
      title: "Old title",
      agentName: "agent",
      harness: "swarmx",
      permissionMode: "inherit",
      pinned: false,
      messages: [],
      createdAt: "2026-07-01T00:00:00.000Z",
      updatedAt: "2026-07-01T00:01:00.000Z",
    };
    const sessionPath = path.join(root, `${session.id}.jsonl`);
    const event = (title: string) =>
      `${JSON.stringify({
        schemaVersion: 1,
        type: "session_created",
        timestamp: session.createdAt,
        session: { ...session, title },
      })}\n`;
    fs.writeFileSync(sessionPath, event("Old title"), "utf8");

    try {
      expect(listSessionSummaries({ sessionsDir: root })[0]?.title).toBe("Old title");
      const changed = event("New title");
      expect(Buffer.byteLength(changed)).toBe(fs.statSync(sessionPath).size);
      fs.writeFileSync(sessionPath, changed, "utf8");
      const future = new Date(Date.now() + 60_000);
      fs.utimesSync(sessionPath, future, future);

      expect(listSessionSummaries({ sessionsDir: root })[0]?.title).toBe("New title");
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
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
