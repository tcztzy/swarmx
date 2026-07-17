export type DesktopIpcInvoke = (channel: string, ...args: unknown[]) => Promise<unknown>;
export type DesktopIpcSubscribe = (
  channel: string,
  listener: (value: unknown) => void,
) => () => void;

export type DesktopUpdatePhase =
  | "hidden"
  | "available"
  | "downloading"
  | "installing"
  | "restarting";

export interface DesktopUpdateState {
  phase: DesktopUpdatePhase;
  currentVersion: string;
  latestVersion?: string;
  progress?: number;
  error?: string;
}

export interface DesktopTerminalDataEvent {
  id: string;
  data: string;
}

export interface DesktopTerminalExitEvent {
  id: string;
  exitCode: number;
  signal?: number;
}

export interface DesktopAgentMessageChunk {
  role: string;
  content: string;
  kind: "message" | "thinking" | "tool_call" | "tool_result";
  agent?: string;
  toolName?: string;
}

export interface DesktopAgentChunkEvent {
  requestId: string;
  chunk: DesktopAgentMessageChunk;
}

export interface DesktopSessionMessagesEvent {
  sessionId: string;
}

export interface DesktopAgentQuestionOption {
  label: string;
  description: string;
  preview?: string;
}

export interface DesktopAgentQuestion {
  question: string;
  header: string;
  options: DesktopAgentQuestionOption[];
  multiSelect: boolean;
}

export type DesktopAgentInteractionEvent =
  | {
      kind: "questions";
      requestId: string;
      interactionId: string;
      questions: DesktopAgentQuestion[];
    }
  | {
      kind: "plan_approval";
      requestId: string;
      interactionId: string;
      plan: string;
      filePath: string;
    };

export type DesktopAgentInteractionResponse =
  | { kind: "questions"; answers: Record<string, string> }
  | { kind: "plan_approval"; approved: boolean; feedback?: string };

export interface DesktopBrowserBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DesktopBrowserState {
  id: string;
  url: string;
  title: string;
  loading: boolean;
  canGoBack: boolean;
  canGoForward: boolean;
  error?: string;
}

export interface DesktopProjectData {
  id: string;
  name: string;
  cwd: string;
  pinned: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface DesktopBootstrapData {
  initialProjects?: readonly DesktopProjectData[];
}

/**
 * Creates the renderer-facing desktop bridge without importing Electron.
 * Hosts can supply their own invoke transport and expose the returned object
 * through contextBridge (or an equivalent isolated bridge).
 */
export function createSwarmxDesktopApi(
  invoke: DesktopIpcInvoke,
  subscribe: DesktopIpcSubscribe = () => () => undefined,
  bootstrap: DesktopBootstrapData = {},
) {
  const initialProjects = bootstrap.initialProjects
    ? Object.freeze(bootstrap.initialProjects.map((project) => Object.freeze({ ...project })))
    : undefined;

  return Object.freeze({
    ...(initialProjects ? { initialProjects } : {}),
    sendMessage: (params: {
      requestId: string;
      sessionId?: string;
      harnessId: string;
      userText: string;
      agentComposition?: unknown;
      swarmConfig?: unknown;
      cwd?: string;
    }) => invoke("agent:send", params),

    onAgentChunk: (listener: (event: DesktopAgentChunkEvent) => void) =>
      subscribe("agent:chunk", (value) => listener(value as DesktopAgentChunkEvent)),

    onAgentInteraction: (listener: (event: DesktopAgentInteractionEvent) => void) =>
      subscribe("agent:interaction", (value) => listener(value as DesktopAgentInteractionEvent)),

    onSessionMessages: (listener: (event: DesktopSessionMessagesEvent) => void) =>
      subscribe("session:messages", (value) => listener(value as DesktopSessionMessagesEvent)),

    resolveAgentInteraction: (params: {
      requestId: string;
      interactionId: string;
      response: DesktopAgentInteractionResponse;
    }) => invoke("agent:resolveInteraction", params),

    cancelMessage: (requestId: string) => invoke("agent:cancel", { requestId }),

    createSession: (params: {
      agentName: string;
      harness: string;
      model?: string;
      projectId?: string;
      cwd?: string;
    }) => invoke("session:create", params),

    saveSession: (session: unknown) => invoke("session:save", session),

    loadSession: (id: string) => invoke("session:load", id),

    loadDiscoveredSession: (session: {
      id: string;
      title: string;
      projectId?: string;
      cwd: string;
      pinned?: boolean;
      updatedAt?: string;
      harnessId: string;
      harnessLabel: string;
      source: "local" | "acp";
    }) => invoke("session:loadDiscovered", session),

    listSessions: () => invoke("session:list"),

    getActivityProfile: () => invoke("activity:profile"),

    listProjects: () => invoke("project:list"),

    addExistingProject: () => invoke("project:addExisting"),

    createScratchProject: () => invoke("project:createScratch"),

    setProjectPinned: (id: string, pinned: boolean) => invoke("project:setPinned", { id, pinned }),

    renameProject: (id: string, name: string) => invoke("project:rename", { id, name }),

    revealProject: (id: string) => invoke("project:reveal", { id }),

    archiveProjectTasks: (id: string) => invoke("project:archiveTasks", { id }),

    removeProject: (id: string) => invoke("project:remove", { id }),

    listGroupedSessions: (params?: {
      mode?: "project" | "harness";
      cwd?: string;
      harnessIds?: string[];
    }) => invoke("session:listGrouped", params ?? {}),

    deleteSession: (id: string) => invoke("session:delete", id),

    renameSession: (id: string, title: string) => invoke("session:rename", { id, title }),

    setSessionPinned: (id: string, pinned: boolean) => invoke("session:setPinned", { id, pinned }),

    generateSessionTitle: (id: string, userText: string) =>
      invoke("session:generateTitle", { id, userText }),

    appendMessages: (params: { id: string; messages: unknown[] }) =>
      invoke("session:appendMessages", params),

    importN8nWorkflow: (source: string) => invoke("workflow:importN8n", { source }),

    listExtensions: () => invoke("extension:list"),

    getExtensionManagementState: () => invoke("extension:managementState"),

    saveExtensionSource: (input: unknown) => invoke("extension:saveSource", input),

    refreshExtensionSource: (id: string) => invoke("extension:refreshSource", { id }),

    removeExtensionSource: (id: string) => invoke("extension:removeSource", { id }),

    applyExtensionAction: (input: unknown) => invoke("extension:applyAction", input),

    saveSkillEvolutionPolicy: (input: {
      enabled: boolean;
      promotionGate: "human" | "policy";
    }) => invoke("extension:saveEvolutionPolicy", input),

    listCustomAgents: () => invoke("customAgent:list"),

    saveCustomAgent: (input: unknown) => invoke("customAgent:save", input),

    removeCustomAgent: (id: string) => invoke("customAgent:remove", { id }),

    workspaceRoot: () => invoke("workspace:root") as Promise<string>,

    getWorkspaceReview: (cwd?: string) => invoke("workspace:review", cwd ? { cwd } : {}),

    listWorkspaceDirectory: (path = "", cwd?: string) =>
      invoke("workspace:listDirectory", { path, ...(cwd ? { cwd } : {}) }),

    readWorkspaceFile: (path: string, cwd?: string) =>
      invoke("workspace:readFile", { path, ...(cwd ? { cwd } : {}) }),

    createTerminal: (params: { id?: string; cwd: string; cols?: number; rows?: number }) =>
      invoke("terminal:create", params) as Promise<{ id: string; pid: number }>,

    writeTerminal: (id: string, data: string) =>
      invoke("terminal:write", { id, data }) as Promise<{ written: boolean }>,

    resizeTerminal: (id: string, cols: number, rows: number) =>
      invoke("terminal:resize", { id, cols, rows }) as Promise<{ resized: boolean }>,

    killTerminal: (id: string) => invoke("terminal:kill", { id }) as Promise<{ killed: boolean }>,

    onTerminalData: (listener: (event: DesktopTerminalDataEvent) => void) =>
      subscribe("terminal:data", (value) => listener(value as DesktopTerminalDataEvent)),

    onTerminalExit: (listener: (event: DesktopTerminalExitEvent) => void) =>
      subscribe("terminal:exit", (value) => listener(value as DesktopTerminalExitEvent)),

    createBrowser: (
      params: {
        id?: string;
        url?: string;
        bounds?: DesktopBrowserBounds;
        visible?: boolean;
      } = {},
    ) => invoke("browser:create", params) as Promise<DesktopBrowserState>,

    navigateBrowser: (id: string, url: string) =>
      invoke("browser:navigate", { id, url }) as Promise<DesktopBrowserState>,

    backBrowser: (id: string) => invoke("browser:back", { id }) as Promise<DesktopBrowserState>,

    forwardBrowser: (id: string) =>
      invoke("browser:forward", { id }) as Promise<DesktopBrowserState>,

    reloadBrowser: (id: string) => invoke("browser:reload", { id }) as Promise<DesktopBrowserState>,

    setBrowserBounds: (id: string, bounds: DesktopBrowserBounds) =>
      invoke("browser:setBounds", { id, bounds }) as Promise<{ updated: boolean }>,

    setBrowserVisible: (id: string, visible: boolean) =>
      invoke("browser:setVisible", { id, visible }) as Promise<{ updated: boolean }>,

    destroyBrowser: (id: string) =>
      invoke("browser:destroy", { id }) as Promise<{ destroyed: boolean }>,

    onBrowserState: (listener: (state: DesktopBrowserState) => void) =>
      subscribe("browser:state", (value) => listener(value as DesktopBrowserState)),

    getUpdateState: () => invoke("appUpdate:getState") as Promise<DesktopUpdateState>,

    startUpdate: () => invoke("appUpdate:install") as Promise<DesktopUpdateState>,

    onUpdateState: (listener: (state: DesktopUpdateState) => void) =>
      subscribe("appUpdate:state", (value) => listener(value as DesktopUpdateState)),

    selectFilesAndFolders: () => invoke("workspace:selectFilesAndFolders") as Promise<string[]>,

    refreshModelCatalog: () => invoke("modelCatalog:refresh"),

    addManualModel: (input: {
      id: string;
      label?: string;
      runtimeModel?: string;
      apiProtocol: "anthropic" | "openai_chat" | "openai_responses" | "ollama";
    }) => invoke("modelCatalog:addManualModel", input),

    removeManualModel: (modelId: string) => invoke("modelCatalog:removeManualModel", { modelId }),

    saveProvider: (input: {
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
    }) => invoke("modelCatalog:saveProvider", input),

    removeProvider: (providerId: string) => invoke("modelCatalog:removeProvider", { providerId }),

    refreshProviderUsage: (target?: {
      source: "provider" | "tool_account";
      sourceId: string;
    }) =>
      target === undefined
        ? invoke("providerUsage:refresh")
        : invoke("providerUsage:refresh", target),

    getHarnessEnvironment: () => invoke("harnessEnvironment:get"),

    getHarnessVersion: (params: { harnessId: string; refresh?: boolean }) =>
      invoke("harnessEnvironment:version", params),

    inspectDoctor: (params?: { harnessId?: string }) => invoke("doctor:inspect", params ?? {}),

    fixDoctor: (params: { harnessId?: string; confirmed: boolean }) => invoke("doctor:fix", params),

    setupHarnessEnvironment: (params?: {
      harnessId?: string;
      harnessToolId?: string;
      requirementIds?: string[];
      containerRuntimeId?: string;
      includeContainerRuntime?: boolean;
    }) => invoke("harnessEnvironment:setup", params ?? {}),

    lspComplete: (params: {
      serverId: string;
      workspaceRoot: string;
      text: string;
      position: { line: number; character: number };
      documentUri?: string;
      languageId?: string;
      triggerCharacter?: string;
      timeoutMs?: number;
    }) => invoke("lsp:complete", params),

    lspStop: (params: { serverId: string; workspaceRoot?: string }) => invoke("lsp:stop", params),

    loadImageDataUrl: (source: string) =>
      invoke("asset:imageDataUrl", source) as Promise<string | null>,
  });
}

export type SwarmxDesktopApi = ReturnType<typeof createSwarmxDesktopApi>;
export type SwarmxAPI = SwarmxDesktopApi;
