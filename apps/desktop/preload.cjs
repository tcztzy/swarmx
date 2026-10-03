const { contextBridge, ipcRenderer } = require("electron");

const invoke = (channel) => (payload) => ipcRenderer.invoke(channel, payload);

contextBridge.exposeInMainWorld("swarmx", {
  bootstrap: invoke("swarmx:bootstrap"),
  work: {
    read: invoke("swarmx:work:read"),
    command: invoke("swarmx:work:command"),
  },
  tool: invoke("swarmx:tool"),
  cancelTool: invoke("swarmx:tool:cancel"),
  settings: {
    read: invoke("swarmx:settings:read"),
    update: invoke("swarmx:settings:update"),
  },
  language: { write: invoke("swarmx:language:write") },
  sessions: {
    list: invoke("swarmx:sessions:list"),
    create: invoke("swarmx:sessions:create"),
    history: invoke("swarmx:sessions:history"),
  },
  models: { read: invoke("swarmx:models:read") },
  logs: {
    read: invoke("swarmx:logs:read"),
    evidence: invoke("swarmx:logs:evidence"),
  },
  runs: { control: invoke("swarmx:runs:control") },
  agui: {
    start: invoke("swarmx:agui:start"),
    cancel: invoke("swarmx:agui:cancel"),
    subscribe: (listener) => {
      const handler = (_event, message) => listener(message);
      ipcRenderer.on("swarmx:agui:event", handler);
      return () => ipcRenderer.removeListener("swarmx:agui:event", handler);
    },
  },
});
