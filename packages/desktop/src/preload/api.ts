import type { ProjectData } from "@swarmx/core";
import type {
  DesktopAgentChunkEvent,
  DesktopAgentInteractionEvent,
  DesktopAgentInteractionResponse,
  DesktopBrowserBounds,
  DesktopBrowserState,
  DesktopMessageChunk,
  DesktopSessionMessagesEvent,
  DesktopTerminalDataEvent,
  DesktopTerminalExitEvent,
  DesktopUpdateState,
  SwarmxAPI,
} from "../shared/desktop-api.js";

export type {
  DesktopAgentChunkEvent,
  DesktopAgentInteractionEvent,
  DesktopAgentInteractionResponse,
  DesktopBrowserBounds,
  DesktopBrowserState,
  DesktopSessionMessagesEvent,
  DesktopTerminalDataEvent,
  DesktopTerminalExitEvent,
  DesktopUpdatePhase,
  DesktopUpdateState,
  SwarmxAPI,
} from "../shared/desktop-api.js";

export type DesktopIpcInvoke = <T>(channel: string, ...args: unknown[]) => Promise<T>;
export type DesktopIpcSubscribe = (
  channel: string,
  listener: (value: unknown) => void,
) => () => void;

export type DesktopAgentMessageChunk = DesktopMessageChunk;
export type DesktopProjectData = ProjectData;

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
): SwarmxAPI {
  const initialProjects = bootstrap.initialProjects
    ? Object.freeze(bootstrap.initialProjects.map((project) => Object.freeze({ ...project })))
    : undefined;

  const api: SwarmxAPI = {
    ...(initialProjects ? { initialProjects } : {}),
    sendMessage: (params) => invoke("agent:send", params),

    onAgentChunk: (listener: (event: DesktopAgentChunkEvent) => void) =>
      subscribe("agent:chunk", (value) => listener(value as DesktopAgentChunkEvent)),

    onAgentInteraction: (listener: (event: DesktopAgentInteractionEvent) => void) =>
      subscribe("agent:interaction", (value) => listener(value as DesktopAgentInteractionEvent)),

    onSessionMessages: (listener: (event: DesktopSessionMessagesEvent) => void) =>
      subscribe("session:messages", (value) => listener(value as DesktopSessionMessagesEvent)),

    resolveAgentInteraction: (params) => invoke("agent:resolveInteraction", params),

    cancelMessage: (requestId: string) => invoke("agent:cancel", { requestId }),

    createSession: (params) => invoke("session:create", params),

    saveSession: (session) => invoke("session:save", session),

    loadSession: (id: string) => invoke("session:load", id),

    loadDiscoveredSession: (session) => invoke("session:loadDiscovered", session),

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

    archiveSession: (id: string) => invoke("session:archive", id),

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

    saveSkillEvolutionPolicy: (input) => invoke("extension:saveEvolutionPolicy", input),

    listCustomAgents: () => invoke("customAgent:list"),

    saveCustomAgent: (input: unknown) => invoke("customAgent:save", input),

    removeCustomAgent: (id: string) => invoke("customAgent:remove", { id }),

    getComposerPreferences: () => invoke("composerPreferences:get"),

    saveComposerPreference: (input) => invoke("composerPreferences:save", input),

    getPermissionStatus: (params?: {
      cwd?: string;
      agentId?: string;
      agentPolicy?: unknown;
    }) => invoke("permission:status", params ?? {}),

    savePersonalPermissionPolicy: (
      policy: unknown,
      context?: { cwd?: string; agentId?: string; agentPolicy?: unknown },
    ) => invoke("permission:savePersonal", { policy, ...context }),

    savePermissionProfileAvailability: (
      profileAvailability: unknown,
      context?: { cwd?: string; agentId?: string; agentPolicy?: unknown },
    ) => invoke("permission:saveProfiles", { profileAvailability, ...context }),

    workspaceRoot: () => invoke("workspace:root"),

    getWorkspaceReview: (cwd?: string) => invoke("workspace:review", cwd ? { cwd } : {}),

    listWorkspaceDirectory: (path = "", cwd?: string) =>
      invoke("workspace:listDirectory", { path, ...(cwd ? { cwd } : {}) }),

    readWorkspaceFile: (path: string, cwd?: string) =>
      invoke("workspace:readFile", { path, ...(cwd ? { cwd } : {}) }),

    createTerminal: (params) => invoke("terminal:create", params),

    writeTerminal: (id: string, data: string) => invoke("terminal:write", { id, data }),

    resizeTerminal: (id: string, cols: number, rows: number) =>
      invoke("terminal:resize", { id, cols, rows }),

    killTerminal: (id: string) => invoke("terminal:kill", { id }),

    onTerminalData: (listener: (event: DesktopTerminalDataEvent) => void) =>
      subscribe("terminal:data", (value) => listener(value as DesktopTerminalDataEvent)),

    onTerminalExit: (listener: (event: DesktopTerminalExitEvent) => void) =>
      subscribe("terminal:exit", (value) => listener(value as DesktopTerminalExitEvent)),

    createBrowser: (params = {}) => invoke("browser:create", params),

    navigateBrowser: (id: string, url: string) => invoke("browser:navigate", { id, url }),

    backBrowser: (id: string) => invoke("browser:back", { id }),

    forwardBrowser: (id: string) => invoke("browser:forward", { id }),

    reloadBrowser: (id: string) => invoke("browser:reload", { id }),

    setBrowserBounds: (id: string, bounds: DesktopBrowserBounds) =>
      invoke("browser:setBounds", { id, bounds }),

    setBrowserVisible: (id: string, visible: boolean) =>
      invoke("browser:setVisible", { id, visible }),

    destroyBrowser: (id: string) => invoke("browser:destroy", { id }),

    onBrowserState: (listener: (state: DesktopBrowserState) => void) =>
      subscribe("browser:state", (value) => listener(value as DesktopBrowserState)),

    getUpdateState: () => invoke("appUpdate:getState"),

    startUpdate: () => invoke("appUpdate:install"),

    onUpdateState: (listener: (state: DesktopUpdateState) => void) =>
      subscribe("appUpdate:state", (value) => listener(value as DesktopUpdateState)),

    selectFilesAndFolders: () => invoke("workspace:selectFilesAndFolders"),

    refreshModelCatalog: () => invoke("modelCatalog:refresh"),

    addManualModel: (input) => invoke("modelCatalog:addManualModel", input),

    removeManualModel: (modelId: string) => invoke("modelCatalog:removeManualModel", { modelId }),

    saveProvider: (input) => invoke("modelCatalog:saveProvider", input),

    removeProvider: (providerId: string) => invoke("modelCatalog:removeProvider", { providerId }),

    resetProviderKey: (providerId: string, keyId: string) =>
      invoke("modelCatalog:resetProviderKey", { providerId, keyId }),

    refreshProviderUsage: (target?: {
      source: "provider" | "tool_account";
      sourceId: string;
    }) =>
      target === undefined
        ? invoke("providerUsage:refresh")
        : invoke("providerUsage:refresh", target),

    getHarnessEnvironment: () => invoke("harnessEnvironment:get"),

    getHarnessVersion: (params) => invoke("harnessEnvironment:version", params),

    inspectDoctor: (params?: { harnessId?: string }) => invoke("doctor:inspect", params ?? {}),

    fixDoctor: (params) => invoke("doctor:fix", params),

    setupHarnessEnvironment: (params) => invoke("harnessEnvironment:setup", params ?? {}),

    lspComplete: (params) => invoke("lsp:complete", params),

    lspStop: (params) => invoke("lsp:stop", params),

    loadImageDataUrl: (source: string) => invoke("asset:imageDataUrl", source),
  };
  return Object.freeze(api);
}

export type SwarmxDesktopApi = SwarmxAPI;
