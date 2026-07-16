import { mkdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  RequestCancelledError,
  Swarm,
  appendMessages,
  archiveProjectSessions,
  createSession,
  deleteSession,
  dismissProject,
  executeAgentComposition,
  getHarness,
  importN8nWorkflow,
  listGroupedSessions,
  listProjects,
  listSessions,
  loadDiscoveredSession,
  loadExtensionInventory,
  loadSession,
  registerDefaultProject,
  registerProject,
  renameProject,
  resolveAgentCompositionPlan,
  saveSession,
  setProjectPinned,
  setSessionPinned,
  updateSessionTitle,
} from "@swarmx/core";
import type {
  AgentBackend,
  AgentComposition,
  AgentCompositionPlan,
  AgentConfig,
  DiscoveredSession,
  ExtensionInventory,
  ListGroupedSessionsOptions,
  MessageChunk,
  ProjectData,
  SessionData,
  SwarmConfig,
} from "@swarmx/core";
import { type IpcMainInvokeEvent, dialog, ipcMain, safeStorage, shell } from "electron";
import { type BrowserBounds, BrowserHost } from "./browser-host.js";
import { CodexAccessTokenResolver } from "./codex-auth.js";
import { CustomAgentService } from "./custom-agents.js";
import { DesktopExtensionManager } from "./extension-manager.js";
import {
  HarnessDoctor,
  HarnessEnvironmentService,
  type HarnessEnvironmentSetupRequest,
  type HarnessEnvironmentStatus,
  containerHostBridgeUrl,
} from "./harness-environment.js";
import { type LspCompletionRequest, LspHost, type LspStopRequest } from "./lsp-host.js";
import {
  type ManualModelInput,
  ModelCatalogService,
  type UserProviderInput,
} from "./model-catalog.js";
import { EncryptedFileProviderAuthStore } from "./provider-auth.js";
import {
  type ProviderUsageRefreshTarget,
  ProviderUsageService,
  queryCodexAppServerRequest,
} from "./provider-usage.js";
import { DesktopRequestRegistry } from "./request-registry.js";
import {
  SESSION_TITLE_MODEL_ID,
  generatedSessionTitle,
  isPlaceholderSessionTitle,
  normalizeManualSessionTitle,
  sessionTitleMessages,
} from "./session-title.js";
import { DesktopSettingsStore } from "./settings-store.js";
import { TerminalHost } from "./terminal-host.js";
import {
  type DesktopUpdateServiceLike,
  type DesktopUpdateState,
  createDisabledDesktopUpdateService,
} from "./updater.js";
import {
  WorkspaceTools,
  projectAgentContextMessage,
  workspaceAgentTools,
} from "./workspace-tools.js";

const MAX_INLINE_IMAGE_BYTES = 25 * 1024 * 1024;
const lspHost = new LspHost();
const harnessEnvironment = new HarnessEnvironmentService();
const harnessDoctor = new HarnessDoctor(harnessEnvironment);
const agentRequests = new DesktopRequestRegistry();
const browserHost = new BrowserHost();
const terminalHost = new TerminalHost();
const interactiveOwnerIds = new Set<number>();
const desktopWorkspaceRoot = process.env.INIT_CWD || process.cwd();
const workspaceTools = new WorkspaceTools(desktopWorkspaceRoot);
const desktopSettingsStore = new DesktopSettingsStore();
const providerAuthStore = new EncryptedFileProviderAuthStore({
  encryption: {
    isAvailable: () => safeStorage.isEncryptionAvailable(),
    encrypt: (value) => safeStorage.encryptString(value),
    decrypt: (value) => safeStorage.decryptString(Buffer.from(value)),
  },
});
const codexAccessTokenProvider = new CodexAccessTokenResolver({
  refresh: () => queryCodexAppServerRequest("account/read", { refreshToken: true }),
});
const modelCatalog = new ModelCatalogService({
  authStore: providerAuthStore,
  codexAccessTokenProvider,
  settingsStore: desktopSettingsStore,
});
const customAgents = new CustomAgentService(desktopSettingsStore);
const extensionManager = new DesktopExtensionManager(desktopSettingsStore);
const providerUsage = new ProviderUsageService({ authStore: providerAuthStore });

export interface RegisterIpcHandlersOptions {
  updateService?: DesktopUpdateServiceLike;
  broadcastUpdateState?: (state: DesktopUpdateState) => void;
}

export interface AgentChunkSender {
  isDestroyed(): boolean;
  send(channel: string, payload: unknown): void;
}

export function agentChunkPublisher(
  sender: AgentChunkSender,
  requestId: string,
): (chunk: MessageChunk) => void {
  return (chunk) => {
    if (!sender.isDestroyed()) sender.send("agent:chunk", { requestId, chunk });
  };
}

export function assertFinalAssistantMessage(messages: readonly MessageChunk[]): void {
  if (
    !messages.some(
      (message) =>
        message.kind === "message" &&
        message.role === "assistant" &&
        message.content.trim().length > 0,
    )
  ) {
    throw new Error("Agent run ended without a final assistant response.");
  }
}

async function loadDesktopExtensionInventory(): Promise<ExtensionInventory> {
  const [inventory, nativeAgents] = await Promise.all([
    loadExtensionInventory(),
    customAgents.discoverNative({ workspaceRoot: desktopWorkspaceRoot }),
  ]);
  const declaredIds = new Set(inventory.agents.map((agent) => agent.id));
  const discovered = nativeAgents.agents.filter((agent) => !declaredIds.has(agent.id));
  return {
    ...inventory,
    agents: [...inventory.agents, ...discovered],
    warnings: [...inventory.warnings, ...nativeAgents.warnings],
  };
}

export function registerIpcHandlers(options: RegisterIpcHandlersOptions = {}): void {
  const updateService = options.updateService ?? createDisabledDesktopUpdateService();
  if (options.broadcastUpdateState) updateService.subscribe(options.broadcastUpdateState);
  ipcMain.handle(
    "agent:send",
    async (
      event: IpcMainInvokeEvent,
      params: {
        requestId: string;
        harnessId: string;
        userText: string;
        agentConfig?: AgentConfig;
        agentComposition?: AgentComposition;
        swarmConfig?: SwarmConfig;
        cwd?: string;
      },
    ) => {
      try {
        return await agentRequests.run(event.sender, params.requestId, async () => {
          const onChunk = agentChunkPublisher(event.sender, params.requestId);
          let swarm: Swarm;
          const cwd = await normalizeWorkingDirectory(params.cwd);

          if (params.swarmConfig) {
            assertDesktopSwarmModels(params.swarmConfig);
            const config = cwd
              ? swarmConfigWithWorkingDirectory(params.swarmConfig, cwd)
              : params.swarmConfig;
            swarm = new Swarm(await protectSwarmConfigBackends(config));
          } else if (params.agentComposition) {
            const inventory = await modelCatalog.list(await loadDesktopExtensionInventory());
            const plan = resolveAgentCompositionPlan(params.agentComposition, inventory);
            assertCompositionSupplyReady(inventory, plan, process.env);
            const providerSecrets = plan.modelSupplyId
              ? await modelCatalog.runtimeSecretsForSupply(inventory, plan.modelSupplyId)
              : {};
            const protectedInventory = await protectCompositionHarness(inventory, plan.harnessId);
            const projectTools =
              cwd && compositionRuntimeHarnessId(inventory, plan) === "swarmx"
                ? new WorkspaceTools(cwd)
                : null;
            const messages = await executeAgentComposition(
              params.agentComposition,
              [
                ...(projectTools
                  ? [
                      {
                        role: "system" as const,
                        content: projectAgentContextMessage(cwd ?? desktopWorkspaceRoot),
                      },
                    ]
                  : []),
                { role: "user", content: params.userText },
              ],
              {
                inventory: protectedInventory,
                providerSecrets,
                cwd,
                ...(projectTools ? { localTools: workspaceAgentTools(projectTools) } : {}),
                onChunk,
              },
            );
            assertFinalAssistantMessage(messages);
            return { success: true, messages };
          } else if (params.agentConfig) {
            throw new Error(
              "Inline agentConfig is not accepted by the desktop runtime; use Agent Composition.",
            );
          } else {
            const harness = getHarness(params.harnessId);
            if (!harness) throw new Error(`Unknown harness: ${params.harnessId}`);
            throw new Error(
              `Harness "${params.harnessId}" requires an Agent Composition with an explicit Model.`,
            );
          }

          const result = await swarm.execute(
            {
              messages: [{ role: "user", content: params.userText }],
            },
            undefined,
            onChunk,
          );

          return { success: true, messages: result };
        });
      } catch (err) {
        if (err instanceof RequestCancelledError) {
          return { success: false, canceled: true, requestId: params.requestId };
        }
        return {
          success: false,
          error: err instanceof Error ? err.message : String(err),
        };
      }
    },
  );

  ipcMain.handle(
    "agent:cancel",
    async (event: IpcMainInvokeEvent, params: { requestId: string }) => ({
      requestId: params.requestId,
      canceled: await agentRequests.cancel(event.sender, params.requestId),
    }),
  );

  ipcMain.handle(
    "session:create",
    (
      _event: IpcMainInvokeEvent,
      params: {
        agentName: string;
        harness: string;
        model?: string;
        projectId?: string;
        cwd?: string;
      },
    ): SessionData => {
      return createSession(params.agentName, params.harness, params.model, {
        projectId: params.projectId,
        cwd: params.cwd,
      });
    },
  );

  ipcMain.handle("session:save", (_event: IpcMainInvokeEvent, session: SessionData): void => {
    saveSession(session);
  });

  ipcMain.handle("session:load", (_event: IpcMainInvokeEvent, id: string): SessionData | null => {
    return loadSession(id);
  });

  ipcMain.handle("session:list", (): SessionData[] => listSessions());

  ipcMain.handle("project:list", (): ProjectData[] => {
    registerDefaultProject(desktopWorkspaceRoot);
    return listProjects();
  });

  ipcMain.handle("project:addExisting", async (): Promise<ProjectData | null> => {
    const result = await dialog.showOpenDialog({
      title: "Use an existing project folder",
      buttonLabel: "Use folder",
      defaultPath: desktopWorkspaceRoot,
      properties: ["openDirectory", "createDirectory"],
    });
    const cwd = result.filePaths[0];
    return result.canceled || !cwd ? null : registerProject(cwd);
  });

  ipcMain.handle("project:createScratch", async (): Promise<ProjectData | null> => {
    const result = await dialog.showSaveDialog({
      title: "Create a new project",
      buttonLabel: "Create project",
      defaultPath: path.join(path.dirname(desktopWorkspaceRoot), "untitled-project"),
      nameFieldLabel: "Project name",
      properties: ["createDirectory"],
    });
    if (result.canceled || !result.filePath) return null;
    await mkdir(result.filePath);
    return registerProject(result.filePath);
  });

  ipcMain.handle(
    "project:setPinned",
    (_event: IpcMainInvokeEvent, params: { id: string; pinned: boolean }): ProjectData => {
      const project = setProjectPinned(params.id, params.pinned);
      if (!project) throw new Error(`Unknown project: ${params.id}`);
      return project;
    },
  );

  ipcMain.handle(
    "project:rename",
    (_event: IpcMainInvokeEvent, params: { id: string; name: string }): ProjectData => {
      const project = renameProject(params.id, params.name);
      if (!project) throw new Error(`Unknown project: ${params.id}`);
      return project;
    },
  );

  ipcMain.handle(
    "project:reveal",
    (_event: IpcMainInvokeEvent, params: { id: string }): boolean => {
      const project = listProjects().find((candidate) => candidate.id === params.id);
      if (!project) return false;
      shell.showItemInFolder(project.cwd);
      return true;
    },
  );

  ipcMain.handle(
    "project:archiveTasks",
    (_event: IpcMainInvokeEvent, params: { id: string }): number => {
      const project = listProjects().find((candidate) => candidate.id === params.id);
      if (!project) throw new Error(`Unknown project: ${params.id}`);
      return archiveProjectSessions({ projectId: project.id, cwd: project.cwd });
    },
  );

  ipcMain.handle("project:remove", (_event: IpcMainInvokeEvent, params: { id: string }): boolean =>
    dismissProject(params.id),
  );

  ipcMain.handle(
    "session:listGrouped",
    async (_event: IpcMainInvokeEvent, params?: ListGroupedSessionsOptions) => {
      const status = await harnessEnvironment.status();
      return listGroupedSessions({
        ...(params ?? {}),
        harnessIds: sessionDiscoveryHarnessIds(status, params?.harnessIds),
      });
    },
  );

  ipcMain.handle(
    "session:loadDiscovered",
    async (_event: IpcMainInvokeEvent, session: DiscoveredSession): Promise<SessionData | null> => {
      if (session.source === "acp") {
        const status = await harnessEnvironment.status();
        const harness = status.harnesses.find((item) => item.harnessId === session.harnessId);
        if (!harness || harness.status !== "ready" || harness.executionMode !== "native") {
          throw new Error(
            `ACP session loading for "${session.harnessId}" requires a ready native harness.`,
          );
        }
      }
      return loadDiscoveredSession(session);
    },
  );

  ipcMain.handle("session:delete", (_event: IpcMainInvokeEvent, id: string): boolean =>
    deleteSession(id),
  );

  ipcMain.handle(
    "session:rename",
    (_event: IpcMainInvokeEvent, params: { id: string; title: string }): SessionData => {
      const title = normalizeManualSessionTitle(params.title);
      if (!title) throw new Error("Task title cannot be empty.");
      if (!updateSessionTitle(params.id, title)) {
        throw new Error(`Unknown session: ${params.id}`);
      }
      const session = loadSession(params.id);
      if (!session) throw new Error(`Unknown session: ${params.id}`);
      return session;
    },
  );

  ipcMain.handle(
    "session:setPinned",
    (_event: IpcMainInvokeEvent, params: { id: string; pinned: boolean }): SessionData => {
      const session = setSessionPinned(params.id, params.pinned);
      if (!session) throw new Error(`Unknown session: ${params.id}`);
      return session;
    },
  );

  ipcMain.handle(
    "session:generateTitle",
    async (
      _event: IpcMainInvokeEvent,
      params: { id: string; userText: string },
    ): Promise<{ title: string; updated: boolean }> => {
      const session = loadSession(params.id);
      if (!session) throw new Error(`Unknown session: ${params.id}`);
      const userMessageCount = session.messages.filter(
        (message) => message.kind === "message" && message.role === "user",
      ).length;
      if (!isPlaceholderSessionTitle(session.title) || userMessageCount !== 1) {
        return { title: session.title, updated: false };
      }

      try {
        const inventory = await modelCatalog.list(await loadDesktopExtensionInventory());
        const composition = {
          id: `session-title-${session.id}`,
          harnessId: "swarmx",
          modelId: SESSION_TITLE_MODEL_ID,
          effort: "none",
          host: "local",
        };
        const plan = resolveAgentCompositionPlan(composition, inventory);
        assertCompositionSupplyReady(inventory, plan, process.env);
        const providerSecrets = plan.modelSupplyId
          ? await modelCatalog.runtimeSecretsForSupply(inventory, plan.modelSupplyId)
          : {};
        const messages = await executeAgentComposition(
          composition,
          sessionTitleMessages(params.userText),
          {
            inventory,
            providerSecrets,
          },
        );
        const title = generatedSessionTitle(messages);
        const latest = loadSession(params.id);
        if (!title || !latest || !isPlaceholderSessionTitle(latest.title)) {
          return { title: latest?.title ?? session.title, updated: false };
        }
        updateSessionTitle(params.id, title);
        return { title, updated: true };
      } catch {
        const latest = loadSession(params.id);
        return { title: latest?.title ?? session.title, updated: false };
      }
    },
  );

  ipcMain.handle(
    "session:appendMessages",
    (_event: IpcMainInvokeEvent, params: { id: string; messages: MessageChunk[] }): boolean =>
      appendMessages(params.id, params.messages),
  );

  ipcMain.handle("workflow:importN8n", (_event: IpcMainInvokeEvent, params: { source: string }) => {
    try {
      const result = importN8nWorkflow(params.source);
      return {
        success: true,
        config: result.config,
        warnings: result.warnings,
        nodeMap: result.nodeMap,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  });

  ipcMain.handle("extension:list", async () => {
    const inventory = await loadDesktopExtensionInventory();
    return extensionInventoryWithPlans(await modelCatalog.list(inventory));
  });

  ipcMain.handle("extension:managementState", () => extensionManager.state());

  ipcMain.handle("extension:saveSource", (_event: IpcMainInvokeEvent, input: unknown) =>
    extensionManager.saveSource(input),
  );

  ipcMain.handle("extension:refreshSource", (_event: IpcMainInvokeEvent, params: { id: string }) =>
    extensionManager.refreshSource(params.id),
  );

  ipcMain.handle("extension:removeSource", (_event: IpcMainInvokeEvent, params: { id: string }) =>
    extensionManager.removeSource(params.id),
  );

  ipcMain.handle("extension:applyAction", (_event: IpcMainInvokeEvent, input: unknown) =>
    extensionManager.applyAction(input),
  );

  ipcMain.handle(
    "extension:saveEvolutionPolicy",
    (_event: IpcMainInvokeEvent, input: { enabled: boolean; promotionGate: "human" | "policy" }) =>
      extensionManager.saveEvolutionPolicy(input),
  );

  ipcMain.handle("customAgent:list", async () => {
    const inventory = await loadDesktopExtensionInventory();
    return extensionInventoryWithPlans(await modelCatalog.list(inventory));
  });

  ipcMain.handle("customAgent:save", async (_event: IpcMainInvokeEvent, input: unknown) => {
    const inventory = await loadDesktopExtensionInventory();
    await customAgents.save(input, {
      reservedAgentIds: inventory.agents.map((agent) => agent.id),
    });
    return extensionInventoryWithPlans(await modelCatalog.list(inventory));
  });

  ipcMain.handle(
    "customAgent:remove",
    async (_event: IpcMainInvokeEvent, params: { id: string }) => {
      await customAgents.remove(params.id);
      const inventory = await loadDesktopExtensionInventory();
      return extensionInventoryWithPlans(await modelCatalog.list(inventory));
    },
  );

  ipcMain.handle("workspace:root", () => desktopWorkspaceRoot);

  ipcMain.handle(
    "workspace:review",
    async (_event: IpcMainInvokeEvent, params?: { cwd?: string }) =>
      workspaceToolsFor(await normalizeWorkingDirectory(params?.cwd)).review(),
  );

  ipcMain.handle(
    "workspace:listDirectory",
    async (_event: IpcMainInvokeEvent, params?: { path?: string; cwd?: string }) => {
      const tools = workspaceToolsFor(await normalizeWorkingDirectory(params?.cwd));
      return {
        root: tools.root,
        ...(await tools.listDirectory(params?.path ?? "")),
      };
    },
  );

  ipcMain.handle(
    "workspace:readFile",
    async (_event: IpcMainInvokeEvent, params: { path: string; cwd?: string }) => {
      const tools = workspaceToolsFor(await normalizeWorkingDirectory(params.cwd));
      return {
        root: tools.root,
        binary: false,
        ...(await tools.readFile(params.path)),
      };
    },
  );

  ipcMain.handle(
    "terminal:create",
    (
      event: IpcMainInvokeEvent,
      params: { id?: string; cwd: string; cols?: number; rows?: number },
    ) => {
      const owner = event.sender;
      if (!interactiveOwnerIds.has(owner.id)) {
        interactiveOwnerIds.add(owner.id);
        owner.once("destroyed", () => {
          interactiveOwnerIds.delete(owner.id);
          browserHost.cleanupOwner(owner.id);
          terminalHost.cleanupOwner(owner.id);
        });
      }
      return terminalHost.create(owner, params);
    },
  );

  ipcMain.handle(
    "terminal:write",
    (event: IpcMainInvokeEvent, params: { id: string; data: string }) => ({
      written: terminalHost.write(event.sender.id, params.id, params.data),
    }),
  );

  ipcMain.handle(
    "terminal:resize",
    (event: IpcMainInvokeEvent, params: { id: string; cols: number; rows: number }) => ({
      resized: terminalHost.resize(event.sender.id, params.id, params.cols, params.rows),
    }),
  );

  ipcMain.handle("terminal:kill", (event: IpcMainInvokeEvent, params: { id: string }) => ({
    killed: terminalHost.kill(event.sender.id, params.id),
  }));

  ipcMain.handle(
    "browser:create",
    (
      event: IpcMainInvokeEvent,
      params?: { id?: string; url?: string; bounds?: BrowserBounds; visible?: boolean },
    ) => {
      const owner = event.sender;
      if (!interactiveOwnerIds.has(owner.id)) {
        interactiveOwnerIds.add(owner.id);
        owner.once("destroyed", () => {
          interactiveOwnerIds.delete(owner.id);
          browserHost.cleanupOwner(owner.id);
          terminalHost.cleanupOwner(owner.id);
        });
      }
      return browserHost.create(owner, params);
    },
  );

  ipcMain.handle(
    "browser:navigate",
    async (event: IpcMainInvokeEvent, params: { id: string; url: string }) => {
      const state = await browserHost.navigate(event.sender.id, params.id, params.url);
      if (!state) throw new Error("Browser view is not available.");
      return state;
    },
  );

  ipcMain.handle("browser:back", (event: IpcMainInvokeEvent, params: { id: string }) => {
    browserHost.back(event.sender.id, params.id);
    return requiredBrowserState(event.sender.id, params.id);
  });

  ipcMain.handle("browser:forward", (event: IpcMainInvokeEvent, params: { id: string }) => {
    browserHost.forward(event.sender.id, params.id);
    return requiredBrowserState(event.sender.id, params.id);
  });

  ipcMain.handle("browser:reload", (event: IpcMainInvokeEvent, params: { id: string }) => {
    browserHost.reload(event.sender.id, params.id);
    return requiredBrowserState(event.sender.id, params.id);
  });

  ipcMain.handle(
    "browser:setBounds",
    (event: IpcMainInvokeEvent, params: { id: string; bounds: BrowserBounds }) => ({
      updated: browserHost.setBounds(event.sender.id, params.id, params.bounds),
    }),
  );

  ipcMain.handle(
    "browser:setVisible",
    (event: IpcMainInvokeEvent, params: { id: string; visible: boolean }) => ({
      updated: browserHost.setVisible(event.sender.id, params.id, params.visible),
    }),
  );

  ipcMain.handle("browser:destroy", (event: IpcMainInvokeEvent, params: { id: string }) => ({
    destroyed: browserHost.destroy(event.sender.id, params.id),
  }));

  ipcMain.handle("appUpdate:getState", () => updateService.getState());

  ipcMain.handle("appUpdate:install", () => updateService.startUpdate());

  ipcMain.handle("workspace:selectFilesAndFolders", async () => {
    const result = await dialog.showOpenDialog({
      title: "Add files and folders",
      defaultPath: process.cwd(),
      properties: ["openFile", "openDirectory", "multiSelections"],
    });
    return result.canceled ? [] : result.filePaths;
  });

  ipcMain.handle("modelCatalog:refresh", async () => {
    const inventory = await loadDesktopExtensionInventory();
    return extensionInventoryWithPlans(await modelCatalog.refresh(inventory));
  });

  ipcMain.handle(
    "modelCatalog:addManualModel",
    async (_event: IpcMainInvokeEvent, input: ManualModelInput) => {
      const inventory = await loadDesktopExtensionInventory();
      return extensionInventoryWithPlans(await modelCatalog.addManualModel(inventory, input));
    },
  );

  ipcMain.handle(
    "modelCatalog:removeManualModel",
    async (_event: IpcMainInvokeEvent, params: { modelId: string }) => {
      const inventory = await loadDesktopExtensionInventory();
      return extensionInventoryWithPlans(
        await modelCatalog.removeManualModel(inventory, params.modelId),
      );
    },
  );

  ipcMain.handle(
    "modelCatalog:saveProvider",
    async (_event: IpcMainInvokeEvent, input: UserProviderInput) => {
      const inventory = await loadDesktopExtensionInventory();
      return extensionInventoryWithPlans(await modelCatalog.saveProvider(inventory, input));
    },
  );

  ipcMain.handle(
    "modelCatalog:removeProvider",
    async (_event: IpcMainInvokeEvent, params: { providerId: string }) => {
      const inventory = await loadDesktopExtensionInventory();
      return extensionInventoryWithPlans(
        await modelCatalog.removeProvider(inventory, params.providerId),
      );
    },
  );

  ipcMain.handle(
    "providerUsage:refresh",
    async (_event: IpcMainInvokeEvent, target?: ProviderUsageRefreshTarget) => {
      const inventory = await modelCatalog.list(await loadDesktopExtensionInventory());
      return providerUsage.refresh(inventory, target);
    },
  );

  ipcMain.handle("harnessEnvironment:get", () => harnessEnvironment.status());

  ipcMain.handle(
    "harnessEnvironment:version",
    (_event: IpcMainInvokeEvent, params: { harnessId: string; refresh?: boolean }) =>
      harnessEnvironment.harnessVersion(params.harnessId, params.refresh ?? false),
  );

  ipcMain.handle("doctor:inspect", (_event: IpcMainInvokeEvent, params?: { harnessId?: string }) =>
    harnessDoctor.inspect(params ?? {}),
  );

  ipcMain.handle(
    "doctor:fix",
    (_event: IpcMainInvokeEvent, params: { harnessId?: string; confirmed: boolean }) =>
      harnessDoctor.fix(params),
  );

  ipcMain.handle(
    "harnessEnvironment:setup",
    (_event: IpcMainInvokeEvent, params?: HarnessEnvironmentSetupRequest) =>
      harnessEnvironment.setup(params ?? {}),
  );

  ipcMain.handle(
    "lsp:complete",
    async (_event: IpcMainInvokeEvent, params: LspCompletionRequest) => {
      const inventory = await loadExtensionInventory();
      return lspHost.complete(inventory, params);
    },
  );

  ipcMain.handle("lsp:stop", (_event: IpcMainInvokeEvent, params: LspStopRequest) =>
    lspHost.stop(params),
  );

  ipcMain.handle(
    "asset:imageDataUrl",
    async (_event: IpcMainInvokeEvent, source: string): Promise<string | null> =>
      loadImageDataUrl(source),
  );
}

export function disposeDesktopTerminals(): void {
  browserHost.dispose();
  terminalHost.dispose();
  interactiveOwnerIds.clear();
}

function requiredBrowserState(ownerId: number, id: string) {
  const state = browserHost.getState(ownerId, id);
  if (!state) throw new Error("Browser view is not available.");
  return state;
}

export function sessionDiscoveryHarnessIds(
  status: HarnessEnvironmentStatus,
  requestedHarnessIds?: string[],
): string[] {
  const readyNativeCustomHarnessIds = status.harnesses
    .filter((harness) => {
      if (harness.status !== "ready" || harness.executionMode !== "native") return false;
      return getHarness(harness.harnessId)?.backend.type === "custom";
    })
    .map((harness) => harness.harnessId);
  if (!requestedHarnessIds) return [];
  const ready = new Set(readyNativeCustomHarnessIds);
  return requestedHarnessIds.filter((harnessId) => ready.has(harnessId));
}

export function compositionRuntimeHarnessId(
  inventory: { harnesses: ReadonlyArray<{ id: string; runtimeHarnessId?: string }> },
  plan: Pick<AgentCompositionPlan, "harnessId">,
): string | undefined {
  const harness = inventory.harnesses.find((candidate) => candidate.id === plan.harnessId);
  return harness?.runtimeHarnessId ?? harness?.id ?? plan.harnessId;
}

export function assertDesktopSwarmModels(config: SwarmConfig): void {
  if (config.queen && !config.queen.model) {
    throw new Error(`Swarm "${config.name}" queen requires an explicit Model.`);
  }
  for (const [nodeId, node] of Object.entries(config.nodes)) {
    if (node.kind === "agent") {
      if (!node.agent.model) {
        throw new Error(
          `Swarm "${config.name}" agent node "${nodeId}" requires an explicit Model.`,
        );
      }
    } else if (node.kind === "swarm") {
      assertDesktopSwarmModels(node.swarm);
    }
  }
}

export function extensionInventoryWithPlans(
  inventory: ExtensionInventory,
  env: NodeJS.ProcessEnv = process.env,
): ExtensionInventory & { agentPlans: AgentCompositionPlan[] } {
  const providers = inventory.providers.map((provider) => {
    const readiness = providerRuntimeReadiness(provider, env);
    return { ...provider, runtimeReady: readiness.ready, runtimeNote: readiness.note };
  });
  return {
    ...inventory,
    providers,
    agentPlans: inventory.agents.map((agent) => {
      const plan = resolveAgentCompositionPlan(
        {
          id: `desktop-${agent.id}`,
          agentProfileId: agent.id,
          host: "local",
        },
        inventory,
      );
      const supply = plan.modelSupplyId
        ? inventory.modelSupplies.find((item) => item.id === plan.modelSupplyId)
        : undefined;
      const provider = supply
        ? providers.find((item) => item.id === supply.providerProfileId)
        : undefined;
      if (!provider || provider.runtimeReady !== false) return plan;
      return {
        ...plan,
        status: "blocked" as const,
        healthStatus: "blocked" as const,
        requirements: [
          ...plan.requirements,
          {
            kind: "model_supply" as const,
            status: "unavailable" as const,
            id: supply?.id,
            message:
              provider.runtimeNote ?? `Model supply "${supply?.id ?? "unknown"}" is not ready.`,
          },
        ],
      };
    }),
  };
}

export function providerRuntimeReadiness(
  provider: ExtensionInventory["providers"][number],
  env: NodeJS.ProcessEnv,
): { ready: boolean; note?: string } {
  if (provider.enabled === false) return { ready: false, note: "Provider profile is disabled." };
  if (typeof provider.runtimeReady === "boolean") {
    return { ready: provider.runtimeReady, note: provider.runtimeNote };
  }
  if (!provider.secretRef) return { ready: true };
  if (provider.secretRef.source !== "env") {
    return {
      ready: false,
      note: `Desktop runtime does not implement ${provider.secretRef.source} secrets.`,
    };
  }
  return env[provider.secretRef.key]
    ? { ready: true }
    : { ready: false, note: `Environment secret ${provider.secretRef.key} is not set.` };
}

export function assertCompositionSupplyReady(
  inventory: ExtensionInventory,
  plan: AgentCompositionPlan,
  env: NodeJS.ProcessEnv,
): void {
  if (!plan.modelSupplyId) return;
  const supply = inventory.modelSupplies.find((item) => item.id === plan.modelSupplyId);
  if (!supply) return;
  const provider = inventory.providers.find((item) => item.id === supply.providerProfileId);
  if (!provider) return;
  const readiness = providerRuntimeReadiness(provider, env);
  if (!readiness.ready) {
    throw new Error(readiness.note ?? `Provider profile "${provider.id}" is not ready.`);
  }
}

async function protectedBackendForHarness(
  harnessId: string,
  backend: AgentBackend,
): Promise<AgentBackend> {
  const result = await harnessEnvironment.protectedBackendForHarness(harnessId, backend, {
    workspaceDir: process.cwd(),
  });
  if (!result.success || !result.backend) {
    throw new Error(result.error ?? "Protected harness runtime is not ready.");
  }
  return result.backend;
}

async function protectCompositionHarness(
  inventory: ExtensionInventory,
  harnessId: string | undefined,
): Promise<ExtensionInventory> {
  if (!harnessId) return inventory;
  const matches = inventory.harnesses.filter((harness) => harness.id === harnessId);
  if (matches.length !== 1) return inventory;
  const runtimeHarnessId = matches[0].runtimeHarnessId ?? harnessId;
  const protectedBackend = await protectedBackendForHarness(runtimeHarnessId, matches[0].backend);
  const protectedInventory = {
    ...inventory,
    harnesses: inventory.harnesses.map((harness) =>
      harness.id === harnessId ? { ...harness, backend: protectedBackend } : harness,
    ),
  };
  return protectedBackend.type === "custom" && protectedBackend.program === "container"
    ? containerizeCompositionSupplyRoutes(protectedInventory)
    : protectedInventory;
}

export function containerizeCompositionSupplyRoutes(
  inventory: ExtensionInventory,
): ExtensionInventory {
  return {
    ...inventory,
    providers: inventory.providers.map((provider) => ({
      ...provider,
      ...(provider.baseUrl ? { baseUrl: containerHostBridgeUrl(provider.baseUrl) } : {}),
    })),
    modelSupplies: inventory.modelSupplies.map((supply) => ({
      ...supply,
      apiCompatibility: {
        ...supply.apiCompatibility,
        ...(supply.apiCompatibility.baseUrl
          ? { baseUrl: containerHostBridgeUrl(supply.apiCompatibility.baseUrl) }
          : {}),
      },
    })),
  };
}

async function protectSwarmConfigBackends(config: SwarmConfig): Promise<SwarmConfig> {
  return transformSwarmConfigAgentBackends(config, async (backend) => {
    const harnessId = harnessEnvironment.guessProtectedHarnessId(backend);
    return harnessId ? protectedBackendForHarness(harnessId, backend) : backend;
  });
}

async function normalizeWorkingDirectory(cwd?: string): Promise<string | undefined> {
  if (!cwd?.trim()) return undefined;
  const resolved = path.resolve(cwd);
  const info = await stat(resolved);
  if (!info.isDirectory()) throw new Error(`Working directory must be a directory: ${resolved}`);
  return resolved;
}

function workspaceToolsFor(cwd?: string): WorkspaceTools {
  if (!cwd || cwd === workspaceTools.root) return workspaceTools;
  return new WorkspaceTools(cwd);
}

function swarmConfigWithWorkingDirectory(config: SwarmConfig, cwd: string): SwarmConfig {
  const copy = JSON.parse(JSON.stringify(config)) as SwarmConfig;
  if (copy.queen) {
    copy.queen.process = { ...copy.queen.process, currentDir: cwd };
  }
  for (const node of Object.values(copy.nodes)) {
    if (node.kind === "agent") {
      node.agent.process = { ...node.agent.process, currentDir: cwd };
    } else if (node.kind === "swarm") {
      node.swarm = swarmConfigWithWorkingDirectory(node.swarm, cwd);
    }
  }
  return copy;
}

export async function transformSwarmConfigAgentBackends(
  config: SwarmConfig,
  transform: (backend: AgentBackend) => Promise<AgentBackend>,
): Promise<SwarmConfig> {
  const copy = JSON.parse(JSON.stringify(config)) as SwarmConfig;
  if (copy.queen?.backend) copy.queen.backend = await transform(copy.queen.backend);
  for (const node of Object.values(copy.nodes ?? {})) {
    if (node.kind === "agent" && node.agent.backend) {
      node.agent.backend = await transform(node.agent.backend);
    } else if (node.kind === "swarm") {
      node.swarm = await transformSwarmConfigAgentBackends(node.swarm, transform);
    }
  }
  return copy;
}

async function loadImageDataUrl(source: string): Promise<string | null> {
  try {
    const filePath = localFilePathFromSource(source);
    if (!filePath) return null;

    const fileStat = await stat(filePath);
    if (!fileStat.isFile() || fileStat.size > MAX_INLINE_IMAGE_BYTES) return null;

    const bytes = await readFile(filePath);
    if (bytes.byteLength > MAX_INLINE_IMAGE_BYTES) return null;

    const mimeType = detectImageMimeType(bytes);
    if (!mimeType) return null;

    return `data:${mimeType};base64,${bytes.toString("base64")}`;
  } catch {
    return null;
  }
}

function localFilePathFromSource(source: string): string | null {
  const trimmed = source.trim();
  if (!trimmed) return null;

  if (trimmed.startsWith("file://")) {
    try {
      return fileURLToPath(trimmed);
    } catch {
      return null;
    }
  }

  const decoded = safeDecodeUri(trimmed);
  return path.isAbsolute(decoded) ? decoded : null;
}

function safeDecodeUri(value: string): string {
  try {
    return decodeURI(value);
  } catch {
    return value;
  }
}

function detectImageMimeType(bytes: Buffer): string | null {
  if (
    bytes.length >= 8 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47 &&
    bytes[4] === 0x0d &&
    bytes[5] === 0x0a &&
    bytes[6] === 0x1a &&
    bytes[7] === 0x0a
  ) {
    return "image/png";
  }

  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return "image/jpeg";
  }

  const gifHeader = bytes.subarray(0, 6).toString("ascii");
  if (gifHeader === "GIF87a" || gifHeader === "GIF89a") {
    return "image/gif";
  }

  if (
    bytes.length >= 12 &&
    bytes.subarray(0, 4).toString("ascii") === "RIFF" &&
    bytes.subarray(8, 12).toString("ascii") === "WEBP"
  ) {
    return "image/webp";
  }

  return null;
}
