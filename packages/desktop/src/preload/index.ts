import { listProjects } from "@swarmx/core/project";
import { type IpcRendererEvent, contextBridge, ipcRenderer } from "electron";
import { createSwarmxDesktopApi } from "./api.js";

let initialProjects: ReturnType<typeof listProjects> | undefined;
try {
  initialProjects = listProjects();
} catch {
  initialProjects = undefined;
}

const api = createSwarmxDesktopApi(
  (channel, ...args) => ipcRenderer.invoke(channel, ...args),
  (channel, listener) => {
    const wrapped = (_event: IpcRendererEvent, value: unknown) => listener(value);
    ipcRenderer.on(channel, wrapped);
    return () => ipcRenderer.removeListener(channel, wrapped);
  },
  initialProjects ? { initialProjects } : {},
);

contextBridge.exposeInMainWorld("swarmxAPI", api);

export type {
  DesktopAgentChunkEvent,
  DesktopAgentMessageChunk,
  DesktopBrowserBounds,
  DesktopBrowserState,
  DesktopBootstrapData,
  DesktopIpcInvoke,
  DesktopIpcSubscribe,
  DesktopProjectData,
  DesktopTerminalDataEvent,
  DesktopTerminalExitEvent,
  DesktopUpdatePhase,
  DesktopUpdateState,
  SwarmxAPI,
  SwarmxDesktopApi,
} from "./api.js";
