import { BrowserWindow, type IpcMainInvokeEvent, ipcMain } from "electron";
import {
  AgentPayloadSchema,
  AgUiCancelPayloadSchema,
  AgUiStartPayloadSchema,
  ArtifactContentPayloadSchema,
  ArtifactIdPayloadSchema,
  actionableMessage,
  EnvironmentActPayloadSchema,
  HistoryPayloadSchema,
  ImportArtifactPayloadSchema,
  LanguageWritePayloadSchema,
  LogsQuerySchema,
  ModelsPayloadSchema,
  NotebookExecutionsPayloadSchema,
  RunControlPayloadSchema,
  ScienceProjectPayloadSchema,
  ToolCallPayloadSchema,
  ToolCancelPayloadSchema,
} from "./bridge-contract.js";
import { parseAgUiInput } from "./host/ag-ui.js";
import type { DesktopPlatform } from "./platform.js";

/** Electron IPC surface: operation-named channels reachable only from SwarmX windows. */
export function registerIpc(platform: DesktopPlatform): void {
  const pendingAgUi = new Map<string, { canceled: boolean }>();
  const toolCalls = new Map<string, AbortController>();
  const handle = (channel: string, run: (payload: unknown) => unknown) => {
    ipcMain.handle(channel, (event, payload: unknown) => {
      trusted(event);
      return Promise.resolve(run(payload)).catch((error: unknown) => {
        throw new Error(actionableMessage(error));
      });
    });
  };
  handle("swarmx:bootstrap", () => platform.operations.bootstrap());
  handle("swarmx:tool", (payload) => {
    const { requestId, name, args } = ToolCallPayloadSchema.parse(payload);
    const controller = new AbortController();
    toolCalls.set(requestId, controller);
    return platform.operations.callTool(name, args, requestId, controller.signal).finally(() => {
      toolCalls.delete(requestId);
    });
  });
  handle("swarmx:tool:cancel", (payload) => {
    const { requestId } = ToolCancelPayloadSchema.parse(payload);
    toolCalls.get(requestId)?.abort(new Error("The renderer cancelled this tool call."));
    return { cancelled: true };
  });
  handle("swarmx:settings:read", () => platform.operations.settings());
  handle("swarmx:settings:update", (payload) => platform.operations.updateSettings(payload));
  handle("swarmx:language:write", (payload) =>
    platform.operations.writeLanguage(LanguageWritePayloadSchema.parse(payload).language),
  );
  handle("swarmx:environment:read", () => platform.operations.environment());
  handle("swarmx:environment:act", (payload) =>
    platform.operations.environmentAction(EnvironmentActPayloadSchema.parse(payload).action),
  );
  handle("swarmx:sessions:list", (payload) =>
    platform.operations.listSessions(AgentPayloadSchema.parse(payload).agent),
  );
  handle("swarmx:sessions:create", (payload) =>
    platform.operations.createSession(AgentPayloadSchema.parse(payload).agent),
  );
  handle("swarmx:sessions:history", (payload) => {
    const { agent, sessionId } = HistoryPayloadSchema.parse(payload);
    return platform.operations.history(agent, sessionId);
  });
  handle("swarmx:models:read", (payload) => {
    const { agent, session } = ModelsPayloadSchema.parse(payload);
    return platform.operations.models(agent, session);
  });
  handle("swarmx:logs:read", (payload) => platform.operations.logs(LogsQuerySchema.parse(payload)));
  handle("swarmx:runs:control", (payload) => {
    const { runId, command } = RunControlPayloadSchema.parse(payload);
    return platform.operations.controlRun(runId, command);
  });
  handle("swarmx:science:workspace", () => platform.operations.scienceWorkspace());
  handle("swarmx:science:research-object", (payload) =>
    platform.operations.researchObject(ScienceProjectPayloadSchema.parse(payload).projectId),
  );
  handle("swarmx:science:notebook-executions", (payload) => {
    const { projectId, includeArtifactId } = NotebookExecutionsPayloadSchema.parse(payload);
    return platform.operations.notebookExecutions(projectId, includeArtifactId);
  });
  handle("swarmx:science:artifact-preview", (payload) =>
    platform.operations.artifactPreview(ArtifactIdPayloadSchema.parse(payload).id),
  );
  handle("swarmx:science:artifact-content", (payload) =>
    platform.operations.artifactContent(ArtifactContentPayloadSchema.parse(payload).id),
  );
  handle("swarmx:science:import", (payload) =>
    platform.operations.importArtifact(ImportArtifactPayloadSchema.parse(payload)),
  );

  ipcMain.handle("swarmx:agui:start", (event, raw: unknown) => {
    trusted(event);
    const { agent, input: body } = AgUiStartPayloadSchema.parse(raw);
    const input = parseAgUiInput(body);
    const pending = { canceled: false };
    pendingAgUi.set(input.threadId, pending);
    void (async () => {
      const products = await platform.operations.activeProducts();
      if (pending.canceled) return;
      pendingAgUi.delete(input.threadId);
      const bridge = await products.agUi(agent);
      await bridge.run(input, {
        event: (agUiEvent) => {
          if (!event.sender.isDestroyed())
            event.sender.send("swarmx:agui:event", { threadId: input.threadId, event: agUiEvent });
        },
      });
    })()
      .catch((error: unknown) => {
        if (!event.sender.isDestroyed())
          event.sender.send("swarmx:agui:event", {
            threadId: input.threadId,
            error: error instanceof Error ? error.message : String(error),
          });
      })
      .finally(() => {
        pendingAgUi.delete(input.threadId);
        if (!event.sender.isDestroyed())
          event.sender.send("swarmx:agui:event", { threadId: input.threadId, done: true });
      });
    return { threadId: input.threadId };
  });
  ipcMain.handle("swarmx:agui:cancel", (event, raw: unknown) => {
    trusted(event);
    const { agent, threadId } = AgUiCancelPayloadSchema.parse(raw);
    const pending = pendingAgUi.get(threadId);
    if (pending) {
      pending.canceled = true;
      return { threadId };
    }
    return platform.operations.cancelAgUi(agent, threadId).then(() => ({ threadId }));
  });
}

function trusted(event: IpcMainInvokeEvent): void {
  if (BrowserWindow.fromWebContents(event.sender) === null || event.senderFrame?.parent != null)
    throw new Error("SwarmX IPC is available only to the SwarmX application window.");
}
