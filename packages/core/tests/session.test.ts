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
  migrateLegacySessions,
  promoteTransientSessionFork,
  saveSession,
  setSessionPinned,
  transientSessionModelMessages,
} from "../src/session.js";
import type { MessageChunk } from "../src/types.js";

const sessionsDir = fs.mkdtempSync(path.join(tmpdir(), "swarmx-session-tests-"));
const originalSessionsDir = process.env.SWARMX_SESSIONS_DIR;

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

  it("V522 appends message deltas without rewriting prior Session JSONL bytes", () => {
    const session = createSession("incremental", "swarmx");
    savedIds.push(session.id);
    saveSession(session);
    const rolloutPath = path.join(sessionsDir, `${session.id}.jsonl`);
    const before = fs.readFileSync(rolloutPath);

    expect(
      appendMessages(session.id, [{ role: "user", content: "incremental write", kind: "message" }]),
    ).toBe(true);

    const after = fs.readFileSync(rolloutPath);
    expect(after.subarray(0, before.length)).toEqual(before);
    expect(after.length).toBeGreaterThan(before.length);
    expect(fs.existsSync(path.join(sessionsDir, `${session.id}.json`))).toBe(false);
  });

  it("V522 keeps appended history when a stale Session saves metadata", () => {
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
    saveSession(session);
    const rolloutPath = path.join(sessionsDir, `${session.id}.jsonl`);
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
    source.acpSessionId = "runtime-owned-session";
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
    const filesBefore = fs.readdirSync(sessionsDir).sort();
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
    expect(fs.readdirSync(sessionsDir).sort()).toEqual(filesBefore);
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

  it("V523 reads legacy JSON and migrates it once with a reversible backup", () => {
    const root = fs.mkdtempSync(path.join(tmpdir(), "swarmx-session-migration-"));
    const legacy = {
      id: "legacy-json-session",
      title: "Legacy JSON",
      agentName: "agent",
      harness: "swarmx",
      messages: [{ role: "user", content: "preserve me", kind: "message" }],
      createdAt: "2026-07-01T00:00:00.000Z",
      updatedAt: "2026-07-01T00:01:00.000Z",
    };
    fs.writeFileSync(path.join(root, `${legacy.id}.json`), JSON.stringify(legacy), "utf8");

    try {
      expect(migrateLegacySessions({ sessionsDir: root, dryRun: true })).toMatchObject({
        discovered: 1,
        migrated: 0,
        planned: 1,
        skipped: 0,
        failed: 0,
      });
      const migrated = migrateLegacySessions({ sessionsDir: root });
      expect(migrated).toMatchObject({
        discovered: 1,
        migrated: 1,
        planned: 0,
        skipped: 0,
        failed: 0,
      });
      expect(fs.existsSync(path.join(root, `${legacy.id}.jsonl`))).toBe(true);
      expect(fs.existsSync(path.join(root, `${legacy.id}.json`))).toBe(false);
      expect(migrated.backupDir).toBeTruthy();
      expect(fs.existsSync(path.join(migrated.backupDir ?? "", `${legacy.id}.json`))).toBe(true);
      const created = JSON.parse(
        fs.readFileSync(path.join(root, `${legacy.id}.jsonl`), "utf8").split("\n")[0] ?? "",
      ) as { session: typeof legacy };
      expect(created.session).toMatchObject({
        id: legacy.id,
        messages: legacy.messages,
      });

      fs.appendFileSync(
        path.join(root, `${legacy.id}.jsonl`),
        `${JSON.stringify({
          schemaVersion: 1,
          type: "messages_appended",
          timestamp: "2026-07-01T00:02:00.000Z",
          messages: [{ role: "assistant", content: "index reconciliation", kind: "message" }],
        })}\n`,
      );
      expect(listSessionSummaries({ sessionsDir: root, includeArchived: true })[0]).toMatchObject({
        id: legacy.id,
        messageCount: 2,
      });

      fs.writeFileSync(path.join(root, "sessions.index.jsonl"), "{bad index}\n", "utf8");
      expect(listSessionSummaries({ sessionsDir: root, includeArchived: true })[0]).toMatchObject({
        id: legacy.id,
        messageCount: 2,
      });

      expect(migrateLegacySessions({ sessionsDir: root })).toMatchObject({
        discovered: 0,
        migrated: 0,
        planned: 0,
        skipped: 0,
        failed: 0,
      });
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it("V523 keeps legacy JSON readable before migration", () => {
    const legacy = {
      id: "legacy-json-readable",
      title: "Legacy readable",
      agentName: "agent",
      harness: "swarmx",
      messages: [{ role: "user", content: "still readable", kind: "message" }],
      createdAt: "2026-07-01T00:00:00.000Z",
      updatedAt: "2026-07-01T00:01:00.000Z",
    };
    savedIds.push(legacy.id);
    fs.mkdirSync(sessionsDir, { recursive: true });
    fs.writeFileSync(path.join(sessionsDir, `${legacy.id}.json`), JSON.stringify(legacy), "utf8");

    expect(loadSession(legacy.id)).toMatchObject(legacy);
  });

  it("V523 normalizes sessions written by the released Rust desktop", () => {
    const root = fs.mkdtempSync(path.join(tmpdir(), "swarmx-rust-session-migration-"));
    const legacySessionsDir = path.join(root, "sessions");
    fs.mkdirSync(legacySessionsDir);
    fs.writeFileSync(
      path.join(root, "instances.json"),
      JSON.stringify([
        {
          id: "legacy-instance",
          label: "Legacy Claude",
          harness: "ClaudeCode",
          model: "legacy-model",
        },
      ]),
      "utf8",
    );
    const legacy = {
      id: "legacy-rust-session",
      agent_instance_id: "legacy-instance",
      working_dir: "/workspace/legacy",
      acp_session_id: "remote-session",
      messages: [
        { is_user: true, content: "question", kind: "Message" },
        { is_user: false, content: "reasoning", kind: "Thinking", duration_ms: 1250 },
        {
          is_user: false,
          content: '{"path":"README.md"}',
          kind: "ToolCall",
          tool_name: "read",
        },
        {
          is_user: false,
          content: "visible result",
          kind: "ToolResult",
          tool_name: "read",
          tool_result: "raw result",
        },
      ],
      created_at: "1751328000",
      updated_at: "1751328060",
      title: "Rust desktop Session",
      pinned: true,
      archived: true,
    };
    fs.writeFileSync(
      path.join(legacySessionsDir, `${legacy.id}.json`),
      JSON.stringify(legacy),
      "utf8",
    );

    try {
      const result = migrateLegacySessions({ sessionsDir: legacySessionsDir });
      expect(result).toMatchObject({ discovered: 1, migrated: 1, failed: 0 });
      expect(listSessions({ sessionsDir: legacySessionsDir, includeArchived: true })).toEqual([
        expect.objectContaining({
          id: legacy.id,
          title: legacy.title,
          acpSessionId: legacy.acp_session_id,
          cwd: legacy.working_dir,
          agentName: "Legacy Claude",
          harness: "claude_code",
          model: "legacy-model",
          pinned: true,
          archivedAt: "2025-07-01T00:01:00.000Z",
          createdAt: "2025-07-01T00:00:00.000Z",
          updatedAt: "2025-07-01T00:01:00.000Z",
          messages: [
            { role: "user", content: "question", kind: "message" },
            {
              role: "assistant",
              content: "reasoning",
              kind: "thinking",
              render: { durationMs: 1250 },
            },
            {
              role: "assistant",
              content: '{"path":"README.md"}',
              kind: "tool_call",
              toolName: "read",
            },
            {
              role: "tool",
              content: "visible result",
              kind: "tool_result",
              toolName: "read",
              structuredContent: { legacyToolResult: "raw result" },
            },
          ],
        }),
      ]);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it("V523 retains the Harness identity of Rust runtime-backed sessions", () => {
    const sessionsDir = fs.mkdtempSync(path.join(tmpdir(), "swarmx-rust-runtime-migration-"));
    const legacy = {
      id: "legacy-rust-runtime",
      agent_instance_id: "",
      agent_runtime: "hermes",
      working_dir: "/workspace/runtime",
      messages: [],
      created_at: "1751328000",
      updated_at: "1751328060",
      title: "Runtime Session",
    };
    fs.writeFileSync(path.join(sessionsDir, `${legacy.id}.json`), JSON.stringify(legacy), "utf8");

    try {
      expect(migrateLegacySessions({ sessionsDir })).toMatchObject({
        discovered: 1,
        migrated: 1,
        failed: 0,
      });
      expect(listSessions({ sessionsDir, includeArchived: true })[0]).toMatchObject({
        id: legacy.id,
        agentName: "Hermes",
        harness: "hermes",
      });
    } finally {
      fs.rmSync(sessionsDir, { recursive: true, force: true });
    }
  });

  it("V524 replays a valid prefix after a torn tail but blocks another append", () => {
    const session = createSession("torn", "swarmx");
    savedIds.push(session.id);
    session.messages.push({ role: "user", content: "safe prefix", kind: "message" });
    saveSession(session);
    const rolloutPath = path.join(sessionsDir, `${session.id}.jsonl`);
    fs.appendFileSync(rolloutPath, '{"schemaVersion":1,"type":"messages_appended"');

    expect(loadSession(session.id)?.messages).toEqual(session.messages);
    expect(() =>
      appendMessages(session.id, [
        { role: "assistant", content: "must not append", kind: "message" },
      ]),
    ).toThrow(/torn final record/i);
  });

  it("V524 rejects newline-terminated corrupt records instead of skipping them", () => {
    const session = createSession("corrupt", "swarmx");
    savedIds.push(session.id);
    saveSession(session);
    const rolloutPath = path.join(sessionsDir, `${session.id}.jsonl`);
    fs.appendFileSync(rolloutPath, "{bad json}\n");

    expect(loadSession(session.id)).toBeNull();
    expect(() =>
      appendMessages(session.id, [
        { role: "assistant", content: "must not append", kind: "message" },
      ]),
    ).toThrow(/line 2/i);
  });

  it("V524 invalidates a cached Session when same-size on-disk data changes", () => {
    const session = createSession("same-size-corruption", "swarmx");
    savedIds.push(session.id);
    saveSession(session);
    const rolloutPath = path.join(sessionsDir, `${session.id}.jsonl`);
    expect(loadSession(session.id)?.id).toBe(session.id);
    const valid = fs.readFileSync(rolloutPath, "utf8");
    const corrupt = valid.replace("session_created", "session_Xreated");
    expect(Buffer.byteLength(corrupt)).toBe(Buffer.byteLength(valid));
    fs.writeFileSync(rolloutPath, corrupt, "utf8");
    const future = new Date(Date.now() + 60_000);
    fs.utimesSync(rolloutPath, future, future);

    expect(loadSession(session.id)).toBeNull();
  });

  it("V525 lists indexed Session summaries without message bodies", () => {
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
    expect(fs.existsSync(path.join(sessionsDir, "sessions.index.jsonl"))).toBe(true);
  });

  it("V525 reconciles same-size legacy metadata changes from source modification time", () => {
    const root = fs.mkdtempSync(path.join(tmpdir(), "swarmx-session-index-mtime-"));
    const legacy = {
      id: "legacy-index-mtime",
      title: "Old title",
      agentName: "agent",
      harness: "swarmx",
      messages: [],
      createdAt: "2026-07-01T00:00:00.000Z",
      updatedAt: "2026-07-01T00:01:00.000Z",
    };
    const legacyPath = path.join(root, `${legacy.id}.json`);
    fs.writeFileSync(legacyPath, JSON.stringify(legacy), "utf8");

    try {
      expect(listSessionSummaries({ sessionsDir: root })[0]?.title).toBe("Old title");
      const changed = JSON.stringify({ ...legacy, title: "New title" });
      expect(Buffer.byteLength(changed)).toBe(fs.statSync(legacyPath).size);
      fs.writeFileSync(legacyPath, changed, "utf8");
      const future = new Date(Date.now() + 60_000);
      fs.utimesSync(legacyPath, future, future);

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
