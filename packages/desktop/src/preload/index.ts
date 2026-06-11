import { contextBridge, ipcRenderer } from "electron";

const api = {
  sendMessage: (params: {
    harnessId: string;
    userText: string;
    agentConfig?: unknown;
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

  listSessions: () => ipcRenderer.invoke("session:list"),

  listGroupedSessions: (params?: {
    mode?: "project" | "harness";
    cwd?: string;
    harnessIds?: string[];
  }) => ipcRenderer.invoke("session:listGrouped", params ?? {}),

  deleteSession: (id: string) => ipcRenderer.invoke("session:delete", id),

  appendMessages: (params: { id: string; messages: unknown[] }) =>
    ipcRenderer.invoke("session:appendMessages", params),
};

contextBridge.exposeInMainWorld("swarmxAPI", api);

export type SwarmxAPI = typeof api;
