import { beforeEach, describe, expect, it, vi } from "vitest";
import { createSwarmxDesktopApi, parseDesktopBootstrapData } from "../preload/api.js";
import type { SwarmxAPI } from "../shared/desktop-api.js";

const electron = vi.hoisted(() => ({
  exposeInMainWorld: vi.fn(),
  invoke: vi.fn(),
  sendSync: vi.fn(),
  on: vi.fn(),
  removeListener: vi.fn(),
}));
const projectBootstrap = vi.hoisted(() => ({
  projects: [
    {
      id: "project-bootstrap",
      name: "bootstrap",
      cwd: "/workspace/bootstrap",
      pinned: true,
      createdAt: "2026-07-16T00:00:00.000Z",
      updatedAt: "2026-07-16T00:00:00.000Z",
    },
  ],
}));
electron.sendSync.mockReturnValue(projectBootstrap.projects);

vi.mock("electron", () => ({
  contextBridge: { exposeInMainWorld: electron.exposeInMainWorld },
  ipcRenderer: {
    invoke: electron.invoke,
    sendSync: electron.sendSync,
    on: electron.on,
    removeListener: electron.removeListener,
  },
}));
await import("../preload/index.js");

describe("preload API", () => {
  beforeEach(() => {
    electron.invoke.mockReset();
  });

  it("exposes a frozen contextBridge object", () => {
    expect(electron.exposeInMainWorld).toHaveBeenCalledTimes(1);
    expect(electron.exposeInMainWorld).toHaveBeenCalledWith("swarmxAPI", expect.any(Object));
    expect(Object.isFrozen(exposedApi())).toBe(true);
  });

  it("exposes persisted Projects before any asynchronous IPC request", () => {
    expect(exposedApi().initialProjects).toEqual(projectBootstrap.projects);
    expect(Object.isFrozen(exposedApi().initialProjects)).toBe(true);
    expect(Object.isFrozen(exposedApi().initialProjects[0])).toBe(true);
    expect(electron.invoke).not.toHaveBeenCalled();
    expect(electron.sendSync).toHaveBeenCalledWith("bootstrap:get");
  });

  it("rejects malformed bootstrap Projects through the canonical transport schema", () => {
    expect(parseDesktopBootstrapData({ initialProjects: projectBootstrap.projects })).toEqual({
      initialProjects: projectBootstrap.projects,
    });
    expect(
      parseDesktopBootstrapData({
        initialProjects: [
          Object.fromEntries(
            Object.entries(projectBootstrap.projects[0]).filter(([key]) => key !== "pinned"),
          ),
        ],
      }),
    ).toEqual({});
    expect(
      parseDesktopBootstrapData({
        initialProjects: [{ ...projectBootstrap.projects[0], rawCredential: "secret" }],
      }),
    ).toEqual({});
  });

  it("forwards stable request IDs without renderer mutation", async () => {
    electron.invoke.mockResolvedValue({ success: true, messages: [] });
    const params = {
      requestId: "stable-request-id",
      harnessId: "swarmx",
      userText: "hello",
    };

    await exposedApi().sendMessage(params);

    expect(electron.invoke).toHaveBeenCalledWith("agent:send", params);
  });

  it("exposes bounded audit query, verification, and export channels", async () => {
    const api = exposedApi();
    const query = { category: "permission" as const, limit: 25, reverse: true };

    await api.listAuditEvents(query);
    await api.verifyAuditLog();
    await api.exportAuditLog(query);

    expect(electron.invoke.mock.calls).toEqual([
      ["audit:list", query],
      ["audit:verify"],
      ["audit:export", query],
    ]);
  });

  it("exposes Personal Memory read, save, and explicit forget channels", async () => {
    const api = exposedApi();

    await api.getPersonalMemory();
    await api.savePersonalMemory({
      target: "user",
      content: "Prefer concise answers.",
      expectedRevision: 2,
    });
    await api.forgetPersonalMemory({ target: "user", confirmed: true, expectedRevision: 2 });

    expect(electron.invoke.mock.calls).toEqual([
      ["personalMemory:get"],
      [
        "personalMemory:save",
        { target: "user", content: "Prefer concise answers.", expectedRevision: 2 },
      ],
      ["personalMemory:forget", { target: "user", confirmed: true, expectedRevision: 2 }],
    ]);
  });

  it("exposes durable WorkItem inspection, cancellation, and decisions", async () => {
    const api = exposedApi();

    await api.listTaskWorkItems();
    await api.cancelTaskWorkItem({ workItemId: "awi_detached" });
    await api.decideTaskApproval({
      approvalId: "apr_detached",
      status: "approved",
      decidedBy: "desktop-user",
    });

    expect(electron.invoke.mock.calls).toEqual([
      ["taskRuntime:list"],
      ["taskRuntime:cancel", { workItemId: "awi_detached" }],
      [
        "taskRuntime:decide",
        {
          approvalId: "apr_detached",
          status: "approved",
          decidedBy: "desktop-user",
        },
      ],
    ]);
  });

  it("forwards the persisted conversation permission when creating a session", async () => {
    const params = {
      agentName: "agent",
      harness: "swarmx",
      projectId: "project-1",
      cwd: "/workspace/project-1",
      permissionMode: "plan" as const,
    };

    await exposedApi().createSession(params);

    expect(electron.invoke).toHaveBeenCalledWith("session:create", params);
  });

  it("exposes request-scoped live agent chunks through a removable subscription", () => {
    const listener = vi.fn();
    const unsubscribe = exposedApi().onAgentChunk(listener);
    const registration = electron.on.mock.calls.find(([channel]) => channel === "agent:chunk");
    const wrapped = registration?.[1];
    const event = {
      requestId: "stream-request",
      chunk: { role: "assistant", kind: "thinking", content: "Inspecting" },
    };

    expect(typeof wrapped).toBe("function");
    wrapped?.({}, event);
    expect(listener).toHaveBeenCalledWith(event);

    unsubscribe();
    expect(electron.removeListener).toHaveBeenCalledWith("agent:chunk", wrapped);
  });

  it("validates App Update events before exposing them to Renderer listeners", () => {
    const listener = vi.fn();
    const unsubscribe = exposedApi().onUpdateState?.(listener);
    const registration = electron.on.mock.calls.find(([channel]) => channel === "appUpdate:state");
    const wrapped = registration?.[1];

    wrapped?.({}, { phase: "available", currentVersion: "3.2.0", latestVersion: "3.3.0" });
    expect(listener).toHaveBeenCalledWith({
      phase: "available",
      currentVersion: "3.2.0",
      latestVersion: "3.3.0",
    });
    expect(() => wrapped?.({}, { phase: "forged", currentVersion: "3.2.0" })).toThrow();
    expect(listener).toHaveBeenCalledTimes(1);

    unsubscribe?.();
  });

  it("keeps transient side-chat sends, streams, and lifecycle operations on dedicated channels", async () => {
    const api = exposedApi();
    const send = {
      requestId: "side-request",
      sessionId: "parent-1",
      sideChatId: "side-1",
      sideChatVisible: false,
      harnessId: "swarmx",
      userText: "Explain this",
      agentComposition: { id: "agent-1" },
    };
    electron.invoke.mockResolvedValue({ parentSessionId: "parent-1", chats: [] });

    await api.sendSideChatMessage(send);
    await api.listSideChats("parent-1");
    await api.createSideChat({
      parentSessionId: "parent-1",
      throughMessageIndex: 2,
      expectedMessages: [],
    });
    await api.setSideChatHidden("parent-1", true);
    await api.deleteSideChat("parent-1", "side-1");
    await api.promoteSideChat("parent-1", "side-1");
    await api.cancelSideChat("parent-1", "side-1", "side-request");

    expect(electron.invoke.mock.calls).toEqual([
      ["sideChat:send", send],
      ["sideChat:list", "parent-1"],
      [
        "sideChat:create",
        {
          parentSessionId: "parent-1",
          throughMessageIndex: 2,
          expectedMessages: [],
        },
      ],
      ["sideChat:setHidden", { parentSessionId: "parent-1", hidden: true }],
      ["sideChat:delete", { parentSessionId: "parent-1", sideChatId: "side-1" }],
      ["sideChat:promote", { parentSessionId: "parent-1", sideChatId: "side-1" }],
      [
        "sideChat:cancel",
        {
          parentSessionId: "parent-1",
          sideChatId: "side-1",
          requestId: "side-request",
        },
      ],
    ]);

    const listener = vi.fn();
    const unsubscribe = api.onSideChatChunk(listener);
    const registration = electron.on.mock.calls.find(([channel]) => channel === "sideChat:chunk");
    const wrapped = registration?.[1];
    const event = {
      requestId: "side-request",
      parentSessionId: "parent-1",
      sideChatId: "side-1",
      chunk: { role: "assistant", kind: "thinking", content: "Inspecting" },
    };
    wrapped?.({}, event);
    expect(listener).toHaveBeenCalledWith(event);
    unsubscribe();
    expect(electron.removeListener).toHaveBeenCalledWith("sideChat:chunk", wrapped);
  });

  it("exposes authoritative background session refresh events", () => {
    const listener = vi.fn();
    const unsubscribe = exposedApi().onSessionMessages(listener);
    const registration = electron.on.mock.calls.find(([channel]) => channel === "session:messages");
    const wrapped = registration?.[1];
    const event = { sessionId: "session-background" };

    expect(typeof wrapped).toBe("function");
    wrapped?.({}, event);
    expect(listener).toHaveBeenCalledWith(event);
    unsubscribe();
    expect(electron.removeListener).toHaveBeenCalledWith("session:messages", wrapped);
  });

  it("bridges interactive tool events and scoped resolutions", async () => {
    const listener = vi.fn();
    const unsubscribe = exposedApi().onAgentInteraction(listener);
    const registration = electron.on.mock.calls.find(
      ([channel]) => channel === "agent:interaction",
    );
    const wrapped = registration?.[1];
    const interaction = {
      kind: "questions",
      requestId: "interactive-request",
      interactionId: "interaction-1",
      questions: [],
    };
    wrapped?.({}, interaction);
    expect(listener).toHaveBeenCalledWith(interaction);

    electron.invoke.mockResolvedValue({
      requestId: "interactive-request",
      interactionId: "interaction-1",
      resolved: true,
    });
    await exposedApi().resolveAgentInteraction({
      requestId: "interactive-request",
      interactionId: "interaction-1",
      response: { kind: "questions", answers: { "Which runtime?": "Node" } },
    });
    expect(electron.invoke).toHaveBeenCalledWith("agent:resolveInteraction", {
      requestId: "interactive-request",
      interactionId: "interaction-1",
      response: { kind: "questions", answers: { "Which runtime?": "Node" } },
    });

    unsubscribe();
    expect(electron.removeListener).toHaveBeenCalledWith("agent:interaction", wrapped);
  });

  it("routes cancellation through the dedicated read-only API", async () => {
    electron.invoke.mockResolvedValue({ requestId: "request-to-stop", canceled: true });

    await expect(exposedApi().cancelMessage("request-to-stop")).resolves.toEqual({
      requestId: "request-to-stop",
      canceled: true,
    });
    expect(electron.invoke).toHaveBeenCalledWith("agent:cancel", {
      requestId: "request-to-stop",
    });
  });

  it("creates the same frozen bridge from a host-provided invoke transport", async () => {
    const invoke = vi.fn().mockResolvedValue(["session"]);
    const api = createSwarmxDesktopApi(invoke);

    await expect(api.listGroupedSessions({ mode: "project" })).resolves.toEqual(["session"]);
    expect(Object.isFrozen(api)).toBe(true);
    expect(invoke).toHaveBeenCalledWith("session:listGrouped", { mode: "project" });
  });

  it("exposes the privacy-safe local activity summary", async () => {
    electron.invoke.mockResolvedValue({ lifetime: { totalTokens: 42 } });

    await expect(exposedApi().getActivityProfile()).resolves.toEqual({
      lifetime: { totalTokens: 42 },
    });
    expect(electron.invoke).toHaveBeenCalledWith("activity:profile");
  });

  it("bridges local task rename, pin, archive, and generated titles", async () => {
    electron.invoke.mockResolvedValue({ id: "session-1", title: "Renamed" });

    await exposedApi().renameSession("session-1", "Renamed");
    await exposedApi().setSessionPinned("session-1", true);
    await exposedApi().generateSessionTitle("session-1", "Fix the title");
    await exposedApi().archiveSession("session-1");

    expect(electron.invoke).toHaveBeenNthCalledWith(1, "session:rename", {
      id: "session-1",
      title: "Renamed",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(2, "session:setPinned", {
      id: "session-1",
      pinned: true,
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(3, "session:generateTitle", {
      id: "session-1",
      userText: "Fix the title",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(4, "session:archive", "session-1");
  });

  it("bridges conflict-safe user message edits through the isolated preload API", async () => {
    const params = {
      id: "session-1",
      messageIndex: 0,
      expectedMessages: [{ role: "user", kind: "message" as const, content: "Original" }],
      content: "Revised",
    };
    electron.invoke.mockResolvedValue({ id: "session-1", messages: [params.expectedMessages[0]] });

    await exposedApi().editSessionUserMessage(params);

    expect(electron.invoke).toHaveBeenCalledWith("session:editUserMessage", params);
  });

  it("bridges conflict-safe Session forks through the isolated preload API", async () => {
    const params = {
      id: "session-1",
      throughMessageIndex: 1,
      expectedMessages: [
        { role: "user", kind: "message" as const, content: "Question" },
        { role: "assistant", kind: "message" as const, content: "Answer" },
      ],
    };
    electron.invoke.mockResolvedValue({ id: "session-2", messages: params.expectedMessages });

    await exposedApi().forkSession(params);

    expect(electron.invoke).toHaveBeenCalledWith("session:fork", params);
  });

  it("exposes the workspace root needed for local mention completion", async () => {
    electron.invoke.mockResolvedValue("/workspace");

    await expect(exposedApi().workspaceRoot()).resolves.toBe("/workspace");
    expect(electron.invoke).toHaveBeenCalledWith("workspace:root");
  });

  it("normalizes omitted Workspace inspection arguments in Preload", async () => {
    await exposedApi().getWorkspaceReview();
    await exposedApi().listWorkspaceDirectory();
    await exposedApi().readWorkspaceFile("README.md");

    expect(electron.invoke.mock.calls).toEqual([
      ["workspace:review", {}],
      ["workspace:listDirectory", { path: "" }],
      ["workspace:readFile", { path: "README.md" }],
    ]);
  });

  it("bridges media import and preview through typed isolated IPC calls", async () => {
    const attachment = {
      id: "notes",
      name: "notes.md",
      kind: "text" as const,
      mimeType: "text/markdown",
      sizeBytes: 8,
      uri: "file:///managed/notes.md",
      source: "user" as const,
    };
    const files = [
      {
        name: "notes.md",
        mimeType: "text/markdown",
        bytes: new Uint8Array([35, 32, 78, 111, 116, 101, 115, 10]),
      },
    ];
    electron.invoke.mockResolvedValueOnce([attachment]).mockResolvedValueOnce({
      status: "available",
      attachment,
      text: "# Notes\n",
    });

    await expect(exposedApi().importMediaAttachments(files, [attachment])).resolves.toEqual([
      attachment,
    ]);
    await expect(exposedApi().previewMediaAttachment(attachment)).resolves.toMatchObject({
      status: "available",
      text: "# Notes\n",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(1, "media:import", files, [attachment]);
    expect(electron.invoke).toHaveBeenNthCalledWith(2, "media:preview", attachment);
  });

  it("bridges project selection without exposing filesystem access", async () => {
    electron.invoke.mockResolvedValue({ id: "project-1", cwd: "/workspace/project-1" });

    await exposedApi().listProjects();
    await exposedApi().addExistingProject();
    await exposedApi().createScratchProject();
    await exposedApi().setProjectPinned("project-1", true);
    await exposedApi().renameProject("project-1", "Renamed");
    await exposedApi().revealProject("project-1");
    await exposedApi().archiveProjectTasks("project-1");
    await exposedApi().removeProject("project-1");

    expect(electron.invoke).toHaveBeenNthCalledWith(1, "project:list");
    expect(electron.invoke).toHaveBeenNthCalledWith(2, "project:addExisting");
    expect(electron.invoke).toHaveBeenNthCalledWith(3, "project:createScratch");
    expect(electron.invoke).toHaveBeenNthCalledWith(4, "project:setPinned", {
      id: "project-1",
      pinned: true,
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(5, "project:rename", {
      id: "project-1",
      name: "Renamed",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(6, "project:reveal", { id: "project-1" });
    expect(electron.invoke).toHaveBeenNthCalledWith(7, "project:archiveTasks", {
      id: "project-1",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(8, "project:remove", { id: "project-1" });
  });

  it("bridges terminal lifecycle and PTY events without exposing Electron", async () => {
    const created = { id: "term-1", pid: 42 };
    electron.invoke.mockResolvedValue(created);

    await expect(
      exposedApi().createTerminal({ id: "term-1", cwd: "/workspace", cols: 90, rows: 30 }),
    ).resolves.toEqual(created);
    await exposedApi().writeTerminal("term-1", "pwd\r");
    await exposedApi().resizeTerminal("term-1", 100, 32);
    await exposedApi().killTerminal("term-1");

    expect(electron.invoke).toHaveBeenNthCalledWith(1, "terminal:create", {
      id: "term-1",
      cwd: "/workspace",
      cols: 90,
      rows: 30,
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(2, "terminal:write", {
      id: "term-1",
      data: "pwd\r",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(3, "terminal:resize", {
      id: "term-1",
      cols: 100,
      rows: 32,
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(4, "terminal:kill", { id: "term-1" });

    const listener = vi.fn();
    const dispose = exposedApi().onTerminalData(listener);
    const wrapped = electron.on.mock.calls.at(-1)?.[1];
    if (typeof wrapped !== "function") throw new Error("terminal listener was not registered");
    wrapped({}, { id: "term-1", data: "output" });
    expect(listener).toHaveBeenCalledWith({ id: "term-1", data: "output" });
    dispose();
    expect(electron.removeListener).toHaveBeenCalledWith("terminal:data", wrapped);

    const exitListener = vi.fn();
    const disposeExit = exposedApi().onTerminalExit(exitListener);
    const wrappedExit = electron.on.mock.calls.at(-1)?.[1];
    if (typeof wrappedExit !== "function")
      throw new Error("terminal exit listener was not registered");
    wrappedExit({}, { id: "term-1", exitCode: 0, signal: 1 });
    expect(exitListener).toHaveBeenCalledWith({ id: "term-1", exitCode: 0, signal: 1 });

    expect(() => wrapped({}, { id: "term-1", data: "output", cwd: "/secret" })).toThrow();
    expect(() => wrappedExit({}, { id: "term-1", exitCode: Number.POSITIVE_INFINITY })).toThrow();
    expect(listener).toHaveBeenCalledTimes(1);
    expect(exitListener).toHaveBeenCalledTimes(1);
    disposeExit();
    expect(electron.removeListener).toHaveBeenCalledWith("terminal:exit", wrappedExit);
  });

  it("bridges workspace inspection and every sandboxed browser control", async () => {
    const browserState = {
      id: "browser-1",
      url: "https://example.com/",
      title: "Example",
      loading: false,
      canGoBack: false,
      canGoForward: false,
    };
    electron.invoke.mockResolvedValue(browserState);

    await exposedApi().getWorkspaceReview("/workspace/project-1");
    await exposedApi().listWorkspaceDirectory("src", "/workspace/project-1");
    await exposedApi().readWorkspaceFile("src/App.tsx", "/workspace/project-1");
    await exposedApi().createBrowser();
    await exposedApi().navigateBrowser("browser-1", "https://openai.com");
    await exposedApi().backBrowser("browser-1");
    await exposedApi().forwardBrowser("browser-1");
    await exposedApi().reloadBrowser("browser-1");
    await exposedApi().setBrowserBounds("browser-1", { x: 600, y: 54, width: 600, height: 746 });
    await exposedApi().setBrowserVisible("browser-1", false);
    await exposedApi().destroyBrowser("browser-1");

    expect(electron.invoke).toHaveBeenNthCalledWith(1, "workspace:review", {
      cwd: "/workspace/project-1",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(2, "workspace:listDirectory", {
      path: "src",
      cwd: "/workspace/project-1",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(3, "workspace:readFile", {
      path: "src/App.tsx",
      cwd: "/workspace/project-1",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(4, "browser:create", {});
    expect(electron.invoke).toHaveBeenNthCalledWith(5, "browser:navigate", {
      id: "browser-1",
      url: "https://openai.com",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(6, "browser:back", { id: "browser-1" });
    expect(electron.invoke).toHaveBeenNthCalledWith(7, "browser:forward", { id: "browser-1" });
    expect(electron.invoke).toHaveBeenNthCalledWith(8, "browser:reload", { id: "browser-1" });
    expect(electron.invoke).toHaveBeenNthCalledWith(9, "browser:setBounds", {
      id: "browser-1",
      bounds: { x: 600, y: 54, width: 600, height: 746 },
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(10, "browser:setVisible", {
      id: "browser-1",
      visible: false,
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(11, "browser:destroy", {
      id: "browser-1",
    });

    const listener = vi.fn();
    const dispose = exposedApi().onBrowserState(listener);
    const wrapped = electron.on.mock.calls.at(-1)?.[1];
    if (typeof wrapped !== "function") throw new Error("browser listener was not registered");
    wrapped({}, browserState);
    expect(listener).toHaveBeenCalledWith(browserState);
    expect(() => wrapped({}, { ...browserState, rawCredential: "secret" })).toThrow();
    expect(listener).toHaveBeenCalledTimes(1);
    dispose();
    expect(electron.removeListener).toHaveBeenCalledWith("browser:state", wrapped);
  });

  it("forwards a user-initiated file and folder selection request", async () => {
    electron.invoke.mockResolvedValue(["/workspace/src/App.tsx"]);

    await expect(exposedApi().selectFilesAndFolders()).resolves.toEqual(["/workspace/src/App.tsx"]);
    expect(electron.invoke).toHaveBeenCalledWith("workspace:selectFilesAndFolders");
  });

  it("reads, starts, and subscribes to desktop update state", async () => {
    const available = {
      phase: "available",
      currentVersion: "3.0.1",
      latestVersion: "3.0.2",
    };
    electron.invoke.mockResolvedValue(available);

    await expect(exposedApi().getUpdateState()).resolves.toEqual(available);
    await expect(exposedApi().startUpdate()).resolves.toEqual(available);
    expect(electron.invoke).toHaveBeenNthCalledWith(1, "appUpdate:getState");
    expect(electron.invoke).toHaveBeenNthCalledWith(2, "appUpdate:install");

    const listener = vi.fn();
    const dispose = exposedApi().onUpdateState(listener);
    const wrapped = electron.on.mock.calls.at(-1)?.[1];
    expect(electron.on).toHaveBeenCalledWith("appUpdate:state", expect.any(Function));
    if (typeof wrapped !== "function") throw new Error("update listener was not registered");
    wrapped({}, available);
    expect(listener).toHaveBeenCalledWith(available);
    dispose();
    expect(electron.removeListener).toHaveBeenCalledWith("appUpdate:state", wrapped);
  });

  it("separates read-only Doctor inspection from confirmed repair", async () => {
    electron.invoke.mockResolvedValue({ healthy: false });

    await exposedApi().inspectDoctor({ harnessId: "hermes" });
    await exposedApi().fixDoctor({ harnessId: "hermes", confirmed: true });

    expect(electron.invoke).toHaveBeenNthCalledWith(1, "doctor:inspect", {
      harnessId: "hermes",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(2, "doctor:fix", {
      harnessId: "hermes",
      confirmed: true,
    });
  });

  it("checks one harness version at a time and forwards explicit refreshes", async () => {
    electron.invoke.mockResolvedValue({ harnessId: "codex", version: "1.2.3" });

    await exposedApi().getHarnessVersion({ harnessId: "codex", refresh: true });

    expect(electron.invoke).toHaveBeenCalledWith("harnessEnvironment:version", {
      harnessId: "codex",
      refresh: true,
    });
  });

  it("exposes Model catalog and Provider mutations without Supply selection", async () => {
    electron.invoke.mockResolvedValue({ models: [] });
    const manualModel = {
      id: "manual-model",
      runtimeModel: "vendor/manual-model",
      apiProtocol: "openai_responses" as const,
    };
    const provider = {
      label: "Anthropic proxy",
      kind: "anthropic" as const,
      baseUrl: "https://proxy.example.test",
      authMode: "auth_token" as const,
      usageAdapter: "new_api" as const,
      secret: "renderer-entry-only",
    };

    await exposedApi().refreshModelCatalog();
    await exposedApi().addManualModel(manualModel);
    await exposedApi().removeManualModel(manualModel.id);
    await exposedApi().saveProvider(provider);
    await exposedApi().removeProvider("swarmx.user.anthropic-proxy");
    await exposedApi().resetProviderKey("swarmx.user.opencode-go", "primary");
    await exposedApi().refreshProviderUsage();
    await exposedApi().refreshProviderUsage({
      source: "provider",
      sourceId: "swarmx.user.anthropic-proxy",
    });

    expect(electron.invoke).toHaveBeenNthCalledWith(1, "modelCatalog:refresh");
    expect(electron.invoke).toHaveBeenNthCalledWith(2, "modelCatalog:addManualModel", manualModel);
    expect(electron.invoke).toHaveBeenNthCalledWith(3, "modelCatalog:removeManualModel", {
      modelId: manualModel.id,
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(4, "modelCatalog:saveProvider", provider);
    expect(electron.invoke).toHaveBeenNthCalledWith(5, "modelCatalog:removeProvider", {
      providerId: "swarmx.user.anthropic-proxy",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(6, "modelCatalog:resetProviderKey", {
      providerId: "swarmx.user.opencode-go",
      keyId: "primary",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(7, "providerUsage:refresh");
    expect(electron.invoke).toHaveBeenNthCalledWith(8, "providerUsage:refresh", {
      source: "provider",
      sourceId: "swarmx.user.anthropic-proxy",
    });
  });

  it("exposes Custom Agent CRUD through narrow IPC methods", async () => {
    electron.invoke.mockResolvedValue({ agents: [] });
    const agent = { id: "researcher", name: "Researcher" };

    await exposedApi().listCustomAgents();
    await exposedApi().saveCustomAgent(agent);
    await exposedApi().removeCustomAgent(agent.id);

    expect(electron.invoke).toHaveBeenNthCalledWith(1, "customAgent:list");
    expect(electron.invoke).toHaveBeenNthCalledWith(2, "customAgent:save", agent);
    expect(electron.invoke).toHaveBeenNthCalledWith(3, "customAgent:remove", { id: agent.id });
  });

  it("exposes narrow Composer preference persistence methods", async () => {
    electron.invoke.mockResolvedValue({ selectionsByHarness: {} });
    const selection = {
      harnessId: "codex",
      modelId: "gpt-5.6-sol",
      modelSupplyId: "catalog:codex:gpt-5.6-sol",
      effort: "high",
    };

    await exposedApi().getComposerPreferences();
    await exposedApi().saveComposerPreference(selection);

    expect(electron.invoke).toHaveBeenNthCalledWith(1, "composerPreferences:get");
    expect(electron.invoke).toHaveBeenNthCalledWith(2, "composerPreferences:save", selection);
  });

  it("exposes narrow built-in tool Settings persistence methods", async () => {
    electron.invoke.mockResolvedValue({ style: "kimi_code" });

    await exposedApi().getBuiltinToolSettings();
    await exposedApi().saveBuiltinToolSettings({ style: "kimi_code" });

    expect(electron.invoke).toHaveBeenNthCalledWith(1, "builtinToolSettings:get");
    expect(electron.invoke).toHaveBeenNthCalledWith(2, "builtinToolSettings:save", {
      style: "kimi_code",
    });
  });

  it("exposes permission status and personal policy updates through narrow IPC methods", async () => {
    electron.invoke.mockResolvedValue({ layers: [] });
    const policy = { mode: "restricted", deniedTools: ["Bash"] };

    await exposedApi().getPermissionStatus({ cwd: "/workspace" });
    await exposedApi().savePersonalPermissionPolicy(policy, { cwd: "/workspace" });
    await exposedApi().savePermissionProfileAvailability(
      { default: true, auto: false, trusted: true },
      { cwd: "/workspace" },
    );

    expect(electron.invoke).toHaveBeenNthCalledWith(1, "permission:status", {
      cwd: "/workspace",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(2, "permission:savePersonal", {
      policy,
      cwd: "/workspace",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(3, "permission:saveProfiles", {
      profileAvailability: { default: true, auto: false, trusted: true },
      cwd: "/workspace",
    });
  });
});

function exposedApi(): SwarmxAPI {
  const call = electron.exposeInMainWorld.mock.calls[0];
  if (!call) throw new Error("Preload API was not exposed.");
  return call[1] as SwarmxAPI;
}
