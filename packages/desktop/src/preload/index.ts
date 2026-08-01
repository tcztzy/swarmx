import { contextBridge, type IpcRendererEvent, ipcRenderer } from "electron";
import {
  createSwarmxDesktopApi,
  type DesktopBootstrapData,
  parseDesktopBootstrapData,
} from "./api.js";

let bootstrap: DesktopBootstrapData = {};
try {
  bootstrap = parseDesktopBootstrapData({
    initialProjects: ipcRenderer.sendSync("bootstrap:get"),
  });
} catch {
  bootstrap = {};
}
const api = createSwarmxDesktopApi(
  (channel, ...args) => ipcRenderer.invoke(channel, ...args),
  (channel, listener) => {
    const wrapped = (_event: IpcRendererEvent, value: unknown) => listener(value);
    ipcRenderer.on(channel, wrapped);
    return () => ipcRenderer.removeListener(channel, wrapped);
  },
  bootstrap,
);

contextBridge.exposeInMainWorld("swarmxAPI", api);

export type {
  DesktopAgentChunkEvent,
  DesktopAgentInteractionEvent,
  DesktopAgentInteractionResponse,
  DesktopAgentMessageChunk,
  DesktopAgentQuestion,
  DesktopAgentQuestionOption,
  DesktopBootstrapData,
  DesktopBrowserBounds,
  DesktopBrowserState,
  DesktopIpcInvoke,
  DesktopIpcSubscribe,
  DesktopProjectData,
  DesktopSessionMessagesEvent,
  DesktopTerminalDataEvent,
  DesktopTerminalExitEvent,
  DesktopUpdatePhase,
  DesktopUpdateState,
  SwarmxAPI,
  SwarmxDesktopApi,
} from "./api.js";
