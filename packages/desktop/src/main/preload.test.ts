import { beforeEach, describe, expect, it, vi } from "vitest";
import { createSwarmxDesktopApi } from "../preload/api.js";

const electron = vi.hoisted(() => ({
  exposeInMainWorld: vi.fn(),
  invoke: vi.fn(),
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

vi.mock("electron", () => ({
  contextBridge: { exposeInMainWorld: electron.exposeInMainWorld },
  ipcRenderer: {
    invoke: electron.invoke,
    on: electron.on,
    removeListener: electron.removeListener,
  },
}));
vi.mock("@swarmx/core/project", () => ({
  listProjects: vi.fn(() => projectBootstrap.projects),
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

  it("V331 exposes persisted Projects before any asynchronous IPC request", () => {
    expect(exposedApi().initialProjects).toEqual(projectBootstrap.projects);
    expect(Object.isFrozen(exposedApi().initialProjects)).toBe(true);
    expect(Object.isFrozen(exposedApi().initialProjects[0])).toBe(true);
    expect(electron.invoke).not.toHaveBeenCalled();
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

  it("V457 forwards the persisted conversation permission when creating a session", async () => {
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

  it("V353 exposes request-scoped live agent chunks through a removable subscription", () => {
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

  it("V429 exposes authoritative background session refresh events", () => {
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

  it("V386-V387 bridges interactive tool events and scoped resolutions", async () => {
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

  it("bridges local task rename, pin, delete, and generated titles", async () => {
    electron.invoke.mockResolvedValue({ id: "session-1", title: "Renamed" });

    await exposedApi().renameSession("session-1", "Renamed");
    await exposedApi().setSessionPinned("session-1", true);
    await exposedApi().generateSessionTitle("session-1", "Fix the title");
    await exposedApi().deleteSession("session-1");

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
    expect(electron.invoke).toHaveBeenNthCalledWith(4, "session:delete", "session-1");
  });

  it("exposes the workspace root needed for local mention completion", async () => {
    electron.invoke.mockResolvedValue("/workspace");

    await expect(exposedApi().workspaceRoot()).resolves.toBe("/workspace");
    expect(electron.invoke).toHaveBeenCalledWith("workspace:root");
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
  });

  it("bridges workspace inspection and sandboxed browser controls", async () => {
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
    await exposedApi().createBrowser({ url: "https://example.com" });
    await exposedApi().navigateBrowser("browser-1", "https://openai.com");
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
    expect(electron.invoke).toHaveBeenNthCalledWith(4, "browser:create", {
      url: "https://example.com",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(5, "browser:navigate", {
      id: "browser-1",
      url: "https://openai.com",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(6, "browser:setBounds", {
      id: "browser-1",
      bounds: { x: 600, y: 54, width: 600, height: 746 },
    });

    const listener = vi.fn();
    const dispose = exposedApi().onBrowserState(listener);
    const wrapped = electron.on.mock.calls.at(-1)?.[1];
    if (typeof wrapped !== "function") throw new Error("browser listener was not registered");
    wrapped({}, browserState);
    expect(listener).toHaveBeenCalledWith(browserState);
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
    expect(electron.invoke).toHaveBeenNthCalledWith(6, "providerUsage:refresh");
    expect(electron.invoke).toHaveBeenNthCalledWith(7, "providerUsage:refresh", {
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

  it("exposes permission status and personal policy updates through narrow IPC methods", async () => {
    electron.invoke.mockResolvedValue({ layers: [] });
    const policy = { mode: "restricted", deniedTools: ["Bash"] };

    await exposedApi().getPermissionStatus({ cwd: "/workspace" });
    await exposedApi().savePersonalPermissionPolicy(policy, { cwd: "/workspace" });

    expect(electron.invoke).toHaveBeenNthCalledWith(1, "permission:status", {
      cwd: "/workspace",
    });
    expect(electron.invoke).toHaveBeenNthCalledWith(2, "permission:savePersonal", {
      policy,
      cwd: "/workspace",
    });
  });
});

function exposedApi(): {
  readonly initialProjects: ReadonlyArray<{
    id: string;
    name: string;
    cwd: string;
    pinned: boolean;
    createdAt: string;
    updatedAt: string;
  }>;
  sendMessage(params: {
    requestId: string;
    harnessId: string;
    userText: string;
  }): Promise<unknown>;
  cancelMessage(requestId: string): Promise<unknown>;
  renameSession(id: string, title: string): Promise<unknown>;
  setSessionPinned(id: string, pinned: boolean): Promise<unknown>;
  generateSessionTitle(id: string, userText: string): Promise<unknown>;
  deleteSession(id: string): Promise<unknown>;
  workspaceRoot(): Promise<unknown>;
  listProjects(): Promise<unknown>;
  getActivityProfile(): Promise<unknown>;
  addExistingProject(): Promise<unknown>;
  createScratchProject(): Promise<unknown>;
  getWorkspaceReview(cwd?: string): Promise<unknown>;
  listWorkspaceDirectory(path?: string, cwd?: string): Promise<unknown>;
  readWorkspaceFile(path: string, cwd?: string): Promise<unknown>;
  createTerminal(params: {
    id?: string;
    cwd: string;
    cols?: number;
    rows?: number;
  }): Promise<unknown>;
  writeTerminal(id: string, data: string): Promise<unknown>;
  resizeTerminal(id: string, cols: number, rows: number): Promise<unknown>;
  killTerminal(id: string): Promise<unknown>;
  onTerminalData(listener: (event: unknown) => void): () => void;
  createBrowser(params?: { url?: string }): Promise<unknown>;
  navigateBrowser(id: string, url: string): Promise<unknown>;
  setBrowserBounds(
    id: string,
    bounds: { x: number; y: number; width: number; height: number },
  ): Promise<unknown>;
  setBrowserVisible(id: string, visible: boolean): Promise<unknown>;
  destroyBrowser(id: string): Promise<unknown>;
  onBrowserState(listener: (state: unknown) => void): () => void;
  getUpdateState(): Promise<unknown>;
  startUpdate(): Promise<unknown>;
  onUpdateState(listener: (state: unknown) => void): () => void;
  selectFilesAndFolders(): Promise<unknown>;
  getHarnessVersion(params: { harnessId: string; refresh?: boolean }): Promise<unknown>;
  inspectDoctor(params?: { harnessId?: string }): Promise<unknown>;
  fixDoctor(params: { harnessId?: string; confirmed: boolean }): Promise<unknown>;
  refreshModelCatalog(): Promise<unknown>;
  addManualModel(input: {
    id: string;
    runtimeModel?: string;
    apiProtocol: "anthropic" | "openai_chat" | "openai_responses" | "ollama";
  }): Promise<unknown>;
  removeManualModel(modelId: string): Promise<unknown>;
  saveProvider(input: {
    id?: string;
    label: string;
    kind: "anthropic" | "openai_chat" | "openai_responses" | "ollama";
    baseUrl: string;
    authMode: "api_key" | "auth_token";
    usageAdapter?: "new_api";
    secret?: string;
    accountAccessToken?: string;
    accountUserId?: string;
    clearAccountAccess?: boolean;
  }): Promise<unknown>;
  removeProvider(providerId: string): Promise<unknown>;
  refreshProviderUsage(target?: {
    source: "provider" | "tool_account";
    sourceId: string;
  }): Promise<unknown>;
  listCustomAgents(): Promise<unknown>;
  saveCustomAgent(input: unknown): Promise<unknown>;
  removeCustomAgent(id: string): Promise<unknown>;
  getPermissionStatus(params?: {
    cwd?: string;
    agentId?: string;
    agentPolicy?: unknown;
  }): Promise<unknown>;
  savePersonalPermissionPolicy(
    policy: unknown,
    context?: { cwd?: string; agentId?: string; agentPolicy?: unknown },
  ): Promise<unknown>;
} {
  const call = electron.exposeInMainWorld.mock.calls[0];
  if (!call) throw new Error("Preload API was not exposed.");
  return call[1];
}
