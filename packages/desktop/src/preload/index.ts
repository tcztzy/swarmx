import { contextBridge, ipcRenderer } from "electron";

const api = {
  sendMessage: (params: {
    harnessId: string;
    userText: string;
    agentConfig?: unknown;
    agentComposition?: unknown;
    swarmConfig?: unknown;
    sessionId?: string;
  }) => ipcRenderer.invoke("agent:send", params),

  createSession: (params: {
    agentName: string;
    harness: string;
    model?: string;
  }) => ipcRenderer.invoke("session:create", params),

  saveSession: (session: unknown) => ipcRenderer.invoke("session:save", session),

  loadSession: (id: string) => ipcRenderer.invoke("session:load", id),

  loadDiscoveredSession: (session: {
    id: string;
    title: string;
    cwd: string;
    updatedAt?: string;
    harnessId: string;
    harnessLabel: string;
    source: "local" | "acp";
  }) => ipcRenderer.invoke("session:loadDiscovered", session),

  listSessions: () => ipcRenderer.invoke("session:list"),

  listGroupedSessions: (params?: {
    mode?: "project" | "harness";
    cwd?: string;
    harnessIds?: string[];
  }) => ipcRenderer.invoke("session:listGrouped", params ?? {}),

  deleteSession: (id: string) => ipcRenderer.invoke("session:delete", id),

  appendMessages: (params: { id: string; messages: unknown[] }) =>
    ipcRenderer.invoke("session:appendMessages", params),

  importN8nWorkflow: (source: string) => ipcRenderer.invoke("workflow:importN8n", { source }),

  listExtensions: () => ipcRenderer.invoke("extension:list"),

  lspComplete: (params: {
    serverId: string;
    workspaceRoot: string;
    text: string;
    position: { line: number; character: number };
    documentUri?: string;
    languageId?: string;
    triggerCharacter?: string;
    timeoutMs?: number;
  }) => ipcRenderer.invoke("lsp:complete", params),

  lspStop: (params: { serverId: string; workspaceRoot?: string }) =>
    ipcRenderer.invoke("lsp:stop", params),

  loadImageDataUrl: (source: string) =>
    ipcRenderer.invoke("asset:imageDataUrl", source) as Promise<string | null>,
};

contextBridge.exposeInMainWorld("swarmxAPI", api);

export type SwarmxAPI = typeof api;
