import { EventEmitter } from "node:events";
import { mkdtemp, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { MessageChunk, SessionData } from "@swarmx/core";
import {
  ActivityStore,
  AuditStore,
  builtInExtensionBundle,
  createExtensionInventory,
  createSession,
  deleteSession,
  listSessions,
  parseExtensionBundle,
  removeProject,
  saveSession,
} from "@swarmx/core";
import { describe, expect, it, vi } from "vitest";

const electron = vi.hoisted(() => ({
  handle: vi.fn(),
  on: vi.fn(),
  showOpenDialog: vi.fn(),
  showSaveDialog: vi.fn(),
  showItemInFolder: vi.fn(),
}));

vi.mock("electron", () => ({
  ipcMain: { handle: electron.handle, on: electron.on },
  dialog: {
    showOpenDialog: electron.showOpenDialog,
    showSaveDialog: electron.showSaveDialog,
  },
  shell: { showItemInFolder: electron.showItemInFolder },
}));

const desktopMain = await import("./library.js");
const desktopIpc = await import("./ipc.js");
const trustedIpc = { authorizeIpcSender: () => true };

describe("desktop main library entry", () => {
  it("exports host integration without registering handlers on import", () => {
    expect(desktopMain.registerIpcHandlers).toBeTypeOf("function");
    expect(desktopMain.AgentInteractionBroker).toBeTypeOf("function");
    expect(desktopMain.DesktopRequestRegistry).toBeTypeOf("function");
    expect(desktopMain.HarnessEnvironmentService).toBeTypeOf("function");
    expect(desktopMain.HarnessDoctor).toBeTypeOf("function");
    expect(desktopMain.LspHost).toBeTypeOf("function");
    expect(desktopMain.ModelCatalogService).toBeTypeOf("function");
    expect(desktopMain.ComposerPreferenceService).toBeTypeOf("function");
    expect(desktopMain.FileProviderAuthStore).toBeTypeOf("function");
    expect(electron.handle).not.toHaveBeenCalled();
  });

  it("V548 rejects privileged IPC from an untrusted renderer before dispatch", () => {
    desktopMain.registerIpcHandlers();
    const registration = electron.handle.mock.calls.find(
      ([channel]) => channel === "activity:profile",
    );
    const handler = registration?.[1];
    const bootstrapRegistration = electron.on.mock.calls.find(
      ([channel]) => channel === "bootstrap:get",
    );
    const bootstrapHandler = bootstrapRegistration?.[1];
    if (typeof handler !== "function") throw new Error("activity handler was not registered");
    if (typeof bootstrapHandler !== "function") {
      throw new Error("bootstrap handler was not registered");
    }

    try {
      expect(() => handler({ sender: new EventEmitter() })).toThrow("Untrusted desktop IPC sender");
      expect(() => bootstrapHandler({ sender: new EventEmitter() })).toThrow(
        "Untrusted desktop IPC sender",
      );
    } finally {
      electron.handle.mockClear();
      electron.on.mockClear();
    }
  });

  it("records denied audit IPC without copying the attempted query", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-audit-ipc-denied-"));
    const auditStore = new AuditStore({ filePath: path.join(root, "events.jsonl") });
    desktopMain.registerIpcHandlers({ auditStore, authorizeIpcSender: () => false });
    const handler = electron.handle.mock.calls
      .filter(([channel]) => channel === "audit:list")
      .at(-1)?.[1];
    if (typeof handler !== "function") throw new Error("audit list handler was not registered");

    try {
      expect(() =>
        handler(
          { sender: Object.assign(new EventEmitter(), { id: 77 }) },
          { actorId: "raw-secret-query" },
        ),
      ).toThrow("Untrusted desktop IPC sender");
      const events = auditStore.query({ action: "ipc.request", targetId: "audit.list" });
      expect(events).toEqual([
        expect.objectContaining({
          outcome: "denied",
          actor: { kind: "user", id: "renderer:77" },
          target: { kind: "ipc-channel", id: "audit.list" },
          metadata: { argumentCount: 1 },
        }),
      ]);
      expect(JSON.stringify(events)).not.toContain("raw-secret-query");
    } finally {
      desktopMain.registerIpcHandlers(trustedIpc);
      electron.handle.mockClear();
      electron.on.mockClear();
      await rm(root, { recursive: true, force: true });
    }
  });

  it("records compressed correlated IPC outcomes without copying renderer payloads", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-ipc-audit-"));
    const auditStore = new AuditStore({ filePath: path.join(root, "events.jsonl") });
    desktopMain.registerIpcHandlers({ ...trustedIpc, auditStore });
    const handler = electron.handle.mock.calls
      .filter(([channel]) => channel === "agent:cancel")
      .at(-1)?.[1];
    if (typeof handler !== "function") throw new Error("cancel handler was not registered");
    const sender = Object.assign(new EventEmitter(), { id: 72 });

    try {
      await handler(
        { sender },
        {
          requestId: "req_audit_123",
          prompt: "raw prompt must not be copied",
          apiKey: "sk-secret-must-not-be-copied",
        },
      );

      const events = auditStore.query({
        action: "ipc.request",
        targetId: "agent.cancel",
      });
      expect(events.map((event) => event.outcome)).toEqual(["attempted", "completed"]);
      expect(events[0]).toMatchObject({
        actor: { kind: "user", id: "renderer:72" },
        target: { kind: "ipc-channel", id: "agent.cancel" },
        requestId: "req_audit_123",
        metadata: { argumentCount: 1 },
      });
      expect(JSON.stringify(events)).not.toContain("raw prompt");
      expect(JSON.stringify(events)).not.toContain("sk-secret");
      expect(auditStore.verify().ok).toBe(true);
    } finally {
      desktopMain.registerIpcHandlers(trustedIpc);
      await rm(root, { recursive: true, force: true });
    }
  });

  it("records only failures for low-sensitivity IPC reads", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-ipc-failure-only-"));
    const auditStore = new AuditStore({ filePath: path.join(root, "events.jsonl") });
    const state = { phase: "hidden", currentVersion: "3.2.0" } as const;
    let fail = false;
    const getState = vi.fn(() => {
      if (fail) throw new TypeError("raw getter failure must not be copied");
      return state;
    });
    desktopMain.registerIpcHandlers({
      ...trustedIpc,
      auditStore,
      updateService: {
        getState,
        check: vi.fn(async () => state),
        startUpdate: vi.fn(async () => state),
        subscribe: vi.fn(() => () => undefined),
      },
    });
    const handler = electron.handle.mock.calls
      .filter(([channel]) => channel === "appUpdate:getState")
      .at(-1)?.[1];
    if (typeof handler !== "function") throw new Error("update state handler was not registered");
    const sender = Object.assign(new EventEmitter(), { id: 74 });

    try {
      expect(
        handler({ sender }, { requestId: "req_state_123", apiKey: "sk-secret-must-not-be-copied" }),
      ).toEqual(state);
      expect(auditStore.query({ action: "ipc.request", targetId: "appupdate.getstate" })).toEqual(
        [],
      );

      fail = true;
      expect(() =>
        handler({ sender }, { requestId: "req_state_123", prompt: "raw prompt" }),
      ).toThrow("raw getter failure");
      const events = auditStore.query({
        action: "ipc.request",
        targetId: "appupdate.getstate",
      });
      expect(events).toHaveLength(1);
      expect(events[0]).toMatchObject({
        outcome: "failed",
        target: { kind: "ipc-channel", id: "appupdate.getstate" },
        requestId: "req_state_123",
        metadata: { argumentCount: 1, errorType: "TypeError" },
      });
      expect(JSON.stringify(events)).not.toContain("raw getter failure");
      expect(JSON.stringify(events)).not.toContain("raw prompt");
      expect(JSON.stringify(events)).not.toContain("sk-secret");
    } finally {
      desktopMain.registerIpcHandlers(trustedIpc);
      await rm(root, { recursive: true, force: true });
    }
  });

  it("suppresses successful transient UI and interaction transport events", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-ipc-transient-"));
    const auditStore = new AuditStore({ filePath: path.join(root, "events.jsonl") });
    desktopMain.registerIpcHandlers({ ...trustedIpc, auditStore });
    const boundsHandler = electron.handle.mock.calls
      .filter(([channel]) => channel === "browser:setBounds")
      .at(-1)?.[1];
    const interactionHandler = electron.handle.mock.calls
      .filter(([channel]) => channel === "agent:resolveInteraction")
      .at(-1)?.[1];
    if (typeof boundsHandler !== "function" || typeof interactionHandler !== "function") {
      throw new Error("Transient IPC handlers were not registered");
    }
    const sender = Object.assign(new EventEmitter(), { id: 78 });

    try {
      expect(
        boundsHandler(
          { sender },
          { id: "missing-browser", bounds: { x: 0, y: 0, width: 100, height: 100 } },
        ),
      ).toEqual({ updated: false });
      expect(
        interactionHandler(
          { sender },
          {
            requestId: "missing-request",
            interactionId: "missing-interaction",
            response: { kind: "tool_approval", optionId: "allow_once" },
          },
        ),
      ).toMatchObject({ resolved: false });
      expect(auditStore.query({ action: "ipc.request" })).toEqual([]);
    } finally {
      desktopMain.registerIpcHandlers(trustedIpc);
      await rm(root, { recursive: true, force: true });
    }
  });

  it("uses semantic-only terminal auditing and records only unaudited dispatch failures", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-ipc-semantic-only-"));
    const auditStore = new AuditStore({ filePath: path.join(root, "events.jsonl") });
    desktopMain.registerIpcHandlers({ ...trustedIpc, auditStore });
    const killHandler = electron.handle.mock.calls
      .filter(([channel]) => channel === "terminal:kill")
      .at(-1)?.[1];
    const createHandler = electron.handle.mock.calls
      .filter(([channel]) => channel === "terminal:create")
      .at(-1)?.[1];
    if (typeof killHandler !== "function" || typeof createHandler !== "function") {
      throw new Error("terminal handlers were not registered");
    }
    const sender = Object.assign(new EventEmitter(), { id: 75 });

    try {
      expect(killHandler({ sender }, { id: "missing-terminal" })).toEqual({ killed: false });
      expect(auditStore.query({ action: "ipc.request", targetId: "terminal.kill" })).toEqual([]);
      expect(auditStore.query({ action: "terminal.close", targetId: "missing-terminal" })).toEqual([
        expect.objectContaining({ outcome: "attempted", metadata: { closeReason: "user_kill" } }),
        expect.objectContaining({
          outcome: "denied",
          metadata: { closeReason: "user_kill", reason: "not_owned_or_missing" },
        }),
      ]);

      expect(() => createHandler({ sender }, { id: "invalid-terminal", cwd: "" })).toThrow(
        "working directory is required",
      );
      expect(auditStore.query({ action: "ipc.request", targetId: "terminal.create" })).toEqual([]);
      expect(auditStore.query({ action: "terminal.create", targetId: "invalid-terminal" })).toEqual(
        [
          expect.objectContaining({ outcome: "attempted" }),
          expect.objectContaining({
            outcome: "denied",
            metadata: expect.objectContaining({ reason: "invalid_cwd" }),
          }),
        ],
      );

      expect(() => createHandler({ sender })).toThrow();
      const dispatchFailures = auditStore.query({
        action: "ipc.request",
        targetId: "terminal.create",
      });
      expect(dispatchFailures).toHaveLength(1);
      expect(dispatchFailures[0]).toMatchObject({
        outcome: "failed",
        target: { kind: "ipc-channel", id: "terminal.create" },
        metadata: { argumentCount: 0, errorType: "TypeError" },
      });
    } finally {
      desktopMain.registerIpcHandlers(trustedIpc);
      await rm(root, { recursive: true, force: true });
    }
  });

  it("records one denied IPC event before semantic-only terminal dispatch", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-ipc-semantic-denied-"));
    const auditStore = new AuditStore({ filePath: path.join(root, "events.jsonl") });
    desktopMain.registerIpcHandlers({ auditStore, authorizeIpcSender: () => false });
    const handler = electron.handle.mock.calls
      .filter(([channel]) => channel === "terminal:kill")
      .at(-1)?.[1];
    if (typeof handler !== "function") throw new Error("terminal kill handler was not registered");

    try {
      expect(() =>
        handler(
          { sender: Object.assign(new EventEmitter(), { id: 76 }) },
          { id: "blocked-terminal", data: "raw terminal input" },
        ),
      ).toThrow("Untrusted desktop IPC sender");
      const events = auditStore.query({
        action: "ipc.request",
        targetId: "terminal.kill",
      });
      expect(events).toHaveLength(1);
      expect(events[0]).toMatchObject({
        outcome: "denied",
        actor: { kind: "user", id: "renderer:76" },
        target: { kind: "ipc-channel", id: "terminal.kill" },
        metadata: { argumentCount: 1 },
      });
      expect(auditStore.query({ action: "terminal.close" })).toEqual([]);
      expect(JSON.stringify(events)).not.toContain("raw terminal input");
    } finally {
      desktopMain.registerIpcHandlers(trustedIpc);
      await rm(root, { recursive: true, force: true });
    }
  });

  it("fails closed before IPC dispatch when its audit authority is unavailable", () => {
    const getState = vi.fn(() => ({ phase: "hidden", currentVersion: "3.2.0" }) as const);
    const updateService = {
      getState,
      check: vi.fn(async () => getState()),
      startUpdate: vi.fn(async () => getState()),
      subscribe: vi.fn(() => () => undefined),
    };
    desktopMain.registerIpcHandlers({
      ...trustedIpc,
      updateService,
      auditStore: {
        append: () => {
          throw new Error("audit unavailable");
        },
        query: () => [],
        exportJsonl: () => "",
        verify: () => ({
          ok: false,
          eventCount: 0,
          headSequence: 0,
          headHash: "0".repeat(64),
          checkpointStatus: "not_applicable",
        }),
      },
    });
    const handler = electron.handle.mock.calls
      .filter(([channel]) => channel === "appUpdate:install")
      .at(-1)?.[1];
    if (typeof handler !== "function") throw new Error("update install handler was not registered");

    try {
      expect(() => handler({ sender: Object.assign(new EventEmitter(), { id: 73 }) })).toThrow(
        "audit unavailable",
      );
      expect(updateService.startUpdate).not.toHaveBeenCalled();
    } finally {
      desktopMain.registerIpcHandlers(trustedIpc);
    }
  });

  it("refuses a desktop send without an explicit Harness x Model composition", async () => {
    desktopMain.registerIpcHandlers(trustedIpc);
    const registration = electron.handle.mock.calls.find(([channel]) => channel === "agent:send");
    const handler = registration?.[1];
    if (typeof handler !== "function") throw new Error("agent:send handler was not registered");
    const sender = new EventEmitter();
    Object.assign(sender, { id: 1 });

    await expect(
      handler(
        { sender },
        {
          requestId: "missing-provider-model",
          harnessId: "swarmx",
          userText: "hello",
        },
      ),
    ).resolves.toMatchObject({
      success: false,
      error: expect.stringContaining("requires an Agent Composition with an explicit Model"),
      messages: [
        expect.objectContaining({
          role: "system",
          kind: "message",
          content: expect.stringContaining("requires an Agent Composition with an explicit Model"),
        }),
      ],
    });

    await expect(
      handler(
        { sender },
        {
          requestId: "inline-agent-config",
          harnessId: "swarmx",
          userText: "hello",
          agentConfig: { name: "legacy_agent", model: "gpt-5" },
        },
      ),
    ).resolves.toMatchObject({
      success: false,
      error: expect.stringContaining("Inline agentConfig is not accepted"),
    });

    await expect(
      handler(
        { sender },
        {
          requestId: "implicit-workflow-model",
          harnessId: "swarmx",
          userText: "hello",
          swarmConfig: {
            name: "implicit_workflow",
            root: "agent",
            nodes: {
              agent: { kind: "agent", agent: { name: "agent" } },
            },
            edges: [],
          },
        },
      ),
    ).resolves.toMatchObject({
      success: false,
      error: expect.stringContaining("requires an explicit Model"),
    });
  });

  it("keeps side-chat CRUD outside ordinary Session IPC until promotion", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-desktop-side-ipc-"));
    const originalSessionsDir = process.env.SWARMX_SESSIONS_DIR;
    let parentId: string | undefined;
    let promotedId: string | undefined;
    try {
      process.env.SWARMX_SESSIONS_DIR = root;
      const parent = createSession("agent", "swarmx");
      parentId = parent.id;
      parent.messages = [
        { role: "user", kind: "message", content: "Parent question" },
        { role: "assistant", kind: "message", content: "Parent answer" },
      ];
      saveSession(parent);
      desktopMain.registerIpcHandlers(trustedIpc);
      const handler = (channel: string) =>
        [...electron.handle.mock.calls]
          .reverse()
          .find(([registered]) => registered === channel)?.[1];
      const createSide = handler("sideChat:create");
      const listSide = handler("sideChat:list");
      const sendSide = handler("sideChat:send");
      const promoteSide = handler("sideChat:promote");
      const deleteSide = handler("sideChat:delete");
      if (
        typeof createSide !== "function" ||
        typeof listSide !== "function" ||
        typeof sendSide !== "function" ||
        typeof promoteSide !== "function" ||
        typeof deleteSide !== "function"
      ) {
        throw new Error("Side-chat IPC handlers were not registered");
      }

      const side = createSide(
        {},
        {
          parentSessionId: parent.id,
          throughMessageIndex: 1,
          expectedMessages: parent.messages,
        },
      );
      expect(listSessions().map((session) => session.id)).toEqual([parent.id]);
      expect(listSide({}, parent.id)).toMatchObject({
        parentSessionId: parent.id,
        activeSideChatId: side.id,
        chats: [expect.objectContaining({ id: side.id, messages: [] })],
      });

      const sender = new EventEmitter();
      Object.assign(sender, { id: 71 });
      await expect(
        sendSide(
          { sender },
          {
            requestId: "side-workflow-rejected",
            sessionId: parent.id,
            sideChatId: side.id,
            sideChatVisible: true,
            harnessId: "swarmx",
            userText: "Do not run this workflow",
            swarmConfig: {
              name: "forbidden",
              root: "agent",
              nodes: {},
              edges: [],
            },
          },
        ),
      ).resolves.toMatchObject({
        success: false,
        error: expect.stringContaining("cannot execute workflows"),
      });
      expect(listSide({}, parent.id).chats[0].messages).toEqual([]);

      const promoted = promoteSide(
        {},
        { parentSessionId: parent.id, sideChatId: side.id },
      ) as SessionData;
      promotedId = promoted.id;
      expect(
        listSessions()
          .map((session) => session.id)
          .sort(),
      ).toEqual([parent.id, promoted.id].sort());
      expect(promoted.forkedFrom).toMatchObject({ sessionId: parent.id, messageIndex: 1 });
      expect(deleteSide({}, { parentSessionId: parent.id, sideChatId: side.id }).chats).toEqual([]);
    } finally {
      if (parentId) deleteSession(parentId);
      if (promotedId) deleteSession(promotedId);
      if (originalSessionsDir === undefined) {
        Reflect.deleteProperty(process.env, "SWARMX_SESSIONS_DIR");
      } else {
        process.env.SWARMX_SESSIONS_DIR = originalSessionsDir;
      }
      await rm(root, { recursive: true, force: true });
    }
  });

  it("records failed tasks and estimated token usage for the Profile summary", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-desktop-activity-"));
    const activityStore = new ActivityStore({ filePath: path.join(root, "activity.jsonl") });

    try {
      desktopMain.registerIpcHandlers({ ...trustedIpc, activityStore });
      const sendRegistration = [...electron.handle.mock.calls]
        .reverse()
        .find(([channel]) => channel === "agent:send");
      const profileRegistration = [...electron.handle.mock.calls]
        .reverse()
        .find(([channel]) => channel === "activity:profile");
      const sendHandler = sendRegistration?.[1];
      const profileHandler = profileRegistration?.[1];
      if (typeof sendHandler !== "function" || typeof profileHandler !== "function") {
        throw new Error("Activity IPC handlers were not registered");
      }
      const sender = new EventEmitter();
      Object.assign(sender, { id: 41 });

      await expect(
        sendHandler(
          { sender },
          {
            requestId: "profile-failed-task",
            harnessId: "swarmx",
            userText: "Record this failed request",
          },
        ),
      ).resolves.toMatchObject({ success: false });

      expect(profileHandler()).toMatchObject({
        lifetime: {
          totalTasks: 1,
          completedTasks: 0,
          totalTokens: expect.any(Number),
          estimatedTokens: expect.any(Number),
        },
      });
      expect(profileHandler().lifetime.estimatedTokens).toBeGreaterThan(0);
      expect(activityStore.events()).toEqual([
        expect.objectContaining({
          type: "run_summary",
          taskId: "profile-failed-task",
          status: "failed",
          tools: {},
          skills: {},
        }),
      ]);
      expect(JSON.stringify(activityStore.events())).not.toContain("Record this failed request");
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("V346 resolves host coding tools from the runtime Harness adapter", () => {
    expect(
      desktopIpc.compositionRuntimeHarnessId(
        {
          harnesses: [
            {
              id: "custom-swarmx-harness",
              runtimeHarnessId: "swarmx",
            },
          ],
        },
        { harnessId: "custom-swarmx-harness" },
      ),
    ).toBe("swarmx");
    expect(
      desktopIpc.compositionRuntimeHarnessId(
        { harnesses: [{ id: "custom-codex-harness", runtimeHarnessId: "codex" }] },
        { harnessId: "custom-codex-harness" },
      ),
    ).toBe("codex");
  });

  it("V429 replays only persisted conversational messages into session activations", () => {
    const session = {
      messages: [
        { role: "user", content: "question", kind: "message" },
        { role: "assistant", content: "private reasoning", kind: "thinking" },
        { role: "assistant", content: "answer", kind: "message" },
        { role: "assistant", content: "{}", kind: "tool_call", toolName: "Read" },
        { role: "system", content: "scheduled event", kind: "message" },
      ],
    } as SessionData;

    expect(desktopIpc.sessionChatMessages(session)).toEqual([
      { role: "user", content: "question" },
      { role: "assistant", content: "answer" },
      { role: "system", content: "scheduled event" },
    ]);
  });

  it("V353/V355 scopes streamed chunks and rejects a reasoning-only terminal result", () => {
    const send = vi.fn();
    const publish = desktopIpc.agentChunkPublisher(
      { isDestroyed: () => false, send },
      "request-live-work",
    );
    const thought = { role: "assistant", kind: "thinking" as const, content: "Inspecting" };

    publish(thought);
    expect(send).toHaveBeenCalledWith("agent:chunk", {
      requestId: "request-live-work",
      chunk: thought,
    });
    expect(() => desktopIpc.assertFinalAssistantMessage([thought])).toThrow(
      /without a final assistant response/i,
    );
    expect(() =>
      desktopIpc.assertFinalAssistantMessage([
        thought,
        { role: "assistant", kind: "message", content: "Complete." },
      ]),
    ).not.toThrow();

    const sideSend = vi.fn();
    const publishSide = desktopIpc.agentChunkPublisher(
      { isDestroyed: () => false, send: sideSend },
      "request-side-work",
      {
        channel: "sideChat:chunk",
        context: { parentSessionId: "parent-1", sideChatId: "side-1" },
      },
    );
    publishSide(thought);
    expect(sideSend).toHaveBeenCalledWith("sideChat:chunk", {
      requestId: "request-side-work",
      parentSessionId: "parent-1",
      sideChatId: "side-1",
      chunk: thought,
    });
  });

  it("V521 terminalizes only orphaned tool work when a request is interrupted", () => {
    const messages: MessageChunk[] = [
      {
        role: "assistant",
        kind: "tool_call",
        toolName: "exec_command",
        content: JSON.stringify({ cmd: "sed -n '1,20p' App.tsx" }),
      },
      {
        role: "tool",
        kind: "tool_progress",
        toolName: "exec_command",
        content: "partial output",
      },
      {
        role: "assistant",
        kind: "tool_call",
        toolName: "read_file",
        content: JSON.stringify({ path: "README.md" }),
        render: { invocationId: "read-complete", status: "running" },
      },
      {
        role: "tool",
        kind: "tool_result",
        toolName: "read_file",
        content: "done",
        render: { invocationId: "read-complete", status: "succeeded" },
      },
    ];

    const interrupted = desktopIpc.interruptedMessages(messages, 1_000, 3_500);

    expect(interrupted[0]?.render).toMatchObject({
      status: "canceled",
      durationMs: 2_500,
      startedAt: "1970-01-01T00:00:01.000Z",
      endedAt: "1970-01-01T00:00:03.500Z",
    });
    expect(interrupted[1]?.render?.status).toBe("canceled");
    expect(interrupted[2]?.render?.status).toBe("running");
    expect(interrupted[3]?.render?.status).toBe("succeeded");
  });

  it("V504 delays and coalesces terminal progress while short commands keep only their result", () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    const send = vi.fn();
    const publish = desktopIpc.agentChunkPublisher(
      { isDestroyed: () => false, send },
      "request-terminal-progress",
    );
    const call = (invocationId: string) => ({
      role: "assistant",
      kind: "tool_call" as const,
      content: "{}",
      toolName: "exec_command",
      render: { invocationId, status: "running" as const },
    });
    const progress = (invocationId: string, content: string) => ({
      role: "tool",
      kind: "tool_progress" as const,
      content,
      toolName: "exec_command",
      structuredContent: { output: content, stream: "stdout", mode: "append" },
      render: { invocationId, status: "running" as const },
    });
    const result = (invocationId: string) => ({
      role: "tool",
      kind: "tool_result" as const,
      content: "done",
      toolName: "exec_command",
      render: { invocationId, status: "succeeded" as const },
    });

    try {
      publish(call("short"));
      publish(progress("short", "too fast"));
      vi.advanceTimersByTime(100);
      publish(result("short"));
      expect(send.mock.calls.map(([, payload]) => payload)).toEqual([
        { requestId: "request-terminal-progress", chunk: call("short") },
        { requestId: "request-terminal-progress", chunk: result("short") },
      ]);

      send.mockClear();
      publish(call("long"));
      publish(progress("long", "one\n"));
      publish(progress("long", "two\n"));
      vi.advanceTimersByTime(250);
      expect(send).toHaveBeenLastCalledWith("agent:chunk", {
        requestId: "request-terminal-progress",
        chunk: expect.objectContaining({
          kind: "tool_progress",
          content: "one\ntwo\n",
          structuredContent: expect.objectContaining({
            output: "one\ntwo\n",
            stream: "stdout",
          }),
        }),
      });
      publish(result("long"));
      expect(send).toHaveBeenLastCalledWith("agent:chunk", {
        requestId: "request-terminal-progress",
        chunk: result("long"),
      });

      send.mockClear();
      publish(call("bounded"));
      publish(progress("bounded", "line\n".repeat(10_001)));
      vi.advanceTimersByTime(250);
      expect(send).toHaveBeenLastCalledWith("agent:chunk", {
        requestId: "request-terminal-progress",
        chunk: expect.objectContaining({
          kind: "tool_progress",
          content: expect.stringContaining("[live output truncated]"),
        }),
      });
    } finally {
      publish.close();
      vi.useRealTimers();
    }
  });

  it("opens the native file and folder picker only through an explicit IPC request", async () => {
    electron.showOpenDialog.mockResolvedValue({
      canceled: false,
      filePaths: ["/workspace/src/App.tsx", "/workspace/docs"],
    });
    desktopMain.registerIpcHandlers(trustedIpc);
    const registration = electron.handle.mock.calls.find(
      ([channel]) => channel === "workspace:selectFilesAndFolders",
    );
    const handler = registration?.[1];
    if (typeof handler !== "function") throw new Error("file picker handler was not registered");

    await expect(handler()).resolves.toEqual(["/workspace/src/App.tsx", "/workspace/docs"]);
    expect(electron.showOpenDialog).toHaveBeenCalledWith(
      expect.objectContaining({
        properties: ["openFile", "openDirectory", "multiSelections"],
      }),
    );
  });

  it("V322 registers the canonical project only after the native folder picker confirms it", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-desktop-project-"));
    let projectId: string | undefined;
    try {
      electron.showOpenDialog.mockResolvedValue({ canceled: false, filePaths: [root] });
      desktopMain.registerIpcHandlers(trustedIpc);
      const registration = electron.handle.mock.calls.find(
        ([channel]) => channel === "project:addExisting",
      );
      const handler = registration?.[1];
      if (typeof handler !== "function") {
        throw new Error("project picker handler was not registered");
      }

      const project = await handler();
      projectId = project?.id;
      expect(project).toMatchObject({ name: path.basename(root), cwd: await realpath(root) });
      expect(electron.showOpenDialog).toHaveBeenCalledWith(
        expect.objectContaining({ properties: ["openDirectory", "createDirectory"] }),
      );

      const projectHandler = (channel: string) =>
        electron.handle.mock.calls.filter(([registered]) => registered === channel).at(-1)?.[1];
      const pinHandler = projectHandler("project:setPinned");
      const renameHandler = projectHandler("project:rename");
      const revealHandler = projectHandler("project:reveal");
      const archiveHandler = projectHandler("project:archiveTasks");
      const removeHandler = projectHandler("project:remove");
      if (
        typeof pinHandler !== "function" ||
        typeof renameHandler !== "function" ||
        typeof revealHandler !== "function" ||
        typeof archiveHandler !== "function" ||
        typeof removeHandler !== "function" ||
        !projectId
      ) {
        throw new Error("project action handlers were not registered");
      }

      expect(pinHandler({}, { id: projectId, pinned: true })).toMatchObject({ pinned: true });
      expect(renameHandler({}, { id: projectId, name: "Renamed project" })).toMatchObject({
        name: "Renamed project",
      });
      expect(revealHandler({}, { id: projectId })).toBe(true);
      expect(electron.showItemInFolder).toHaveBeenCalledWith(await realpath(root));
      expect(archiveHandler({}, { id: projectId })).toBe(0);
      expect(removeHandler({}, { id: projectId })).toBe(true);
    } finally {
      if (projectId) removeProject(projectId);
      await rm(root, { recursive: true, force: true });
    }
  });

  it("exposes update state/install IPC and broadcasts service progress", async () => {
    const available = {
      phase: "available" as const,
      currentVersion: "3.0.1",
      latestVersion: "3.0.2",
    };
    const restarting = {
      phase: "restarting" as const,
      currentVersion: "3.0.1",
      latestVersion: "3.0.2",
      progress: 100,
    };
    let publish: ((state: typeof restarting) => void) | undefined;
    const updateService = {
      getState: vi.fn(() => available),
      check: vi.fn(async () => available),
      startUpdate: vi.fn(async () => restarting),
      subscribe: vi.fn((listener: (state: typeof restarting) => void) => {
        publish = listener;
        return () => undefined;
      }),
    };
    const broadcastUpdateState = vi.fn();
    desktopMain.registerIpcHandlers({
      ...trustedIpc,
      updateService,
      broadcastUpdateState,
    });
    const stateHandler = electron.handle.mock.calls
      .filter(([channel]) => channel === "appUpdate:getState")
      .at(-1)?.[1];
    const installHandler = electron.handle.mock.calls
      .filter(([channel]) => channel === "appUpdate:install")
      .at(-1)?.[1];
    if (typeof stateHandler !== "function" || typeof installHandler !== "function") {
      throw new Error("update handlers were not registered");
    }

    expect(stateHandler()).toEqual(available);
    await expect(installHandler()).resolves.toEqual(restarting);
    expect(updateService.startUpdate).toHaveBeenCalledTimes(1);
    publish?.(restarting);
    expect(broadcastUpdateState).toHaveBeenCalledWith(restarting);
  });

  it("blocks extension agents whose runtime secret is unavailable", () => {
    const bundle = parseExtensionBundle({
      id: "runtime-readiness",
      name: "Runtime readiness",
      version: "1.0.0",
      capabilities: {
        providers: [
          {
            id: "missing-secret-provider",
            label: "Missing secret provider",
            kind: "openai_chat",
            secretRef: { source: "env", key: "MISSING_TEST_API_KEY" },
          },
        ],
        modelSupplies: [
          {
            id: "gpt-5-missing-secret",
            modelId: "gpt-5",
            providerProfileId: "missing-secret-provider",
          },
        ],
        agents: [
          {
            id: "blocked-agent",
            name: "Blocked agent",
            harnessId: "swarmx",
            modelId: "gpt-5",
            modelSupplyId: "gpt-5-missing-secret",
          },
        ],
      },
    });
    const inventory = createExtensionInventory([builtInExtensionBundle(), bundle]);
    const projected = desktopIpc.extensionInventoryWithPlans(inventory, {});

    expect(
      projected.providers.find((provider) => provider.id === "missing-secret-provider"),
    ).toMatchObject({
      runtimeReady: false,
      runtimeNote: expect.stringContaining("MISSING_TEST_API_KEY"),
    });
    expect(projected.agentPlans[0]).toMatchObject({
      status: "blocked",
      healthStatus: "blocked",
      requirements: expect.arrayContaining([
        expect.objectContaining({
          kind: "model_supply",
          status: "unavailable",
          id: "gpt-5-missing-secret",
        }),
      ]),
    });
  });

  it("translates protected supply routes without changing Model identity", () => {
    const bundle = parseExtensionBundle({
      id: "bridge-routes",
      name: "Bridge routes",
      version: "1.0.0",
      capabilities: {
        providers: [
          {
            id: "local-anthropic",
            label: "Local Anthropic",
            kind: "anthropic",
            baseUrl: "http://localhost:9000",
          },
        ],
        modelSupplies: [
          {
            id: "gpt-local",
            modelId: "gpt-5",
            providerProfileId: "local-anthropic",
            apiCompatibility: { mode: "bridge", baseUrl: "http://127.0.0.1:4000/v1" },
          },
        ],
      },
    });
    const inventory = createExtensionInventory([builtInExtensionBundle(), bundle]);

    const translated = desktopIpc.containerizeCompositionSupplyRoutes(inventory);

    expect(translated.models.find((model) => model.id === "gpt-5")?.id).toBe("gpt-5");
    expect(
      translated.providers.find((provider) => provider.id === "local-anthropic")?.baseUrl,
    ).toBe("http://host.docker.internal:9000");
    expect(
      translated.modelSupplies.find((supply) => supply.id === "gpt-local")?.apiCompatibility
        .baseUrl,
    ).toBe("http://host.docker.internal:4000/v1");
  });

  it("requires explicit opt-in before probing a ready native ACP harness", () => {
    const status = {
      checkedAt: "2026-07-11T00:00:00.000Z",
      path: "/usr/bin",
      ready: true,
      setupAvailable: false,
      containerRuntimes: [],
      protection: { mode: "protected" as const, ready: true, requiredHarnessIds: [] },
      requirements: [],
      harnesses: [
        harnessStatus("swarmx", "native"),
        harnessStatus("claude_code", "protected"),
        harnessStatus("opencode", "native"),
        harnessStatus("hermes", "native"),
        harnessStatus("openclaw", "native"),
      ],
    };

    expect(desktopIpc.sessionDiscoveryHarnessIds(status)).toEqual([]);
    expect(desktopIpc.sessionDiscoveryHarnessIds(status, ["codex", "hermes"])).toEqual(["hermes"]);
  });

  it("transforms queen and nested swarm agent backends", async () => {
    const config = {
      name: "outer",
      root: "root_agent",
      queen: { name: "queen", backend: { type: "custom" as const, program: "queen-acp" } },
      nodes: {
        root_agent: {
          kind: "agent" as const,
          agent: { name: "root_agent", backend: { type: "custom" as const, program: "root-acp" } },
        },
        nested: {
          kind: "swarm" as const,
          swarm: {
            name: "nested",
            root: "nested_agent",
            nodes: {
              nested_agent: {
                kind: "agent" as const,
                agent: {
                  name: "nested_agent",
                  backend: { type: "custom" as const, program: "nested-acp" },
                },
              },
            },
            edges: [],
          },
        },
      },
      edges: [],
    };

    const transformed = await desktopIpc.transformSwarmConfigAgentBackends(
      config,
      async (backend) =>
        backend.type === "custom"
          ? { ...backend, program: `protected-${backend.program}` }
          : backend,
    );

    expect(transformed.queen?.backend).toMatchObject({ program: "protected-queen-acp" });
    expect(transformed.nodes.root_agent).toMatchObject({
      agent: { backend: { program: "protected-root-acp" } },
    });
    expect(transformed.nodes.nested).toMatchObject({
      swarm: {
        nodes: {
          nested_agent: { agent: { backend: { program: "protected-nested-acp" } } },
        },
      },
    });
    expect(config.queen.backend.program).toBe("queen-acp");
  });
});

function harnessStatus(harnessId: string, executionMode: "native" | "protected") {
  return {
    harnessId,
    harnessLabel: harnessId,
    status: "ready" as const,
    requirements: [],
    executionMode,
    protectionRequired: executionMode === "protected",
  };
}
