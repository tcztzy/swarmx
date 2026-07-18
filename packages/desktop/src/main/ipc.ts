import { mkdir, readFile, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  ActivityStore,
  HarnessPermissionPolicySchema,
  RequestCancelledError,
  Swarm,
  appendMessages,
  archiveProjectSessions,
  createSession,
  deleteSession,
  dismissProject,
  estimateModelTokenUsage,
  executeAgentComposition,
  getHarness,
  importN8nWorkflow,
  listGroupedSessions,
  listProjects,
  listSessions,
  loadDiscoveredSession,
  loadExtensionInventory,
  loadSession,
  mergeModelTokenUsage,
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
  AcpPermissionHandler,
  ActivityEventInput,
  AgentBackend,
  AgentComposition,
  AgentCompositionPlan,
  AgentConfig,
  ChatMessage,
  DiscoveredSession,
  ExtensionInventory,
  ListGroupedSessionsOptions,
  MessageChunk,
  ModelTokenUsage,
  ProjectData,
  SessionData,
  SwarmConfig,
} from "@swarmx/core";
import { type IpcMainInvokeEvent, dialog, ipcMain, safeStorage, shell } from "electron";
import {
  AgentInteractionBroker,
  type DesktopAgentInteractionResolution,
} from "./agent-interactions.js";
import { type BrowserBounds, BrowserHost } from "./browser-host.js";
import { ClaudeChildAgentHost } from "./child-agent-host.js";
import {
  type ClaudeSessionRuntime,
  ClaudeSessionRuntimeRegistry,
} from "./claude-session-runtime.js";
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
import { PermissionService, type RecordPermissionDecisionInput } from "./permission-service.js";
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
  type WorkspaceAgentToolOptions,
  WorkspaceTools,
  projectAgentContextMessage,
  workspaceAgentTools,
  workspaceToolProfile,
} from "./workspace-tools.js";

const MAX_INLINE_IMAGE_BYTES = 25 * 1024 * 1024;
const SENSITIVE_PERMISSION_LABEL_PATTERN =
  /(api[_ -]?key|access[_ -]?token|password|passwd|bearer\s+[a-z0-9]|secret\s*[=:]|private[_ -]?key)/i;
const lspHost = new LspHost();
const harnessEnvironment = new HarnessEnvironmentService();
const harnessDoctor = new HarnessDoctor(harnessEnvironment);
const agentRequests = new DesktopRequestRegistry();
const agentInteractions = new AgentInteractionBroker();
const claudeSessionRuntimes = new ClaudeSessionRuntimeRegistry();
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
const permissionService = new PermissionService(desktopSettingsStore);
const extensionManager = new DesktopExtensionManager(desktopSettingsStore);
const providerUsage = new ProviderUsageService({ authStore: providerAuthStore });
const desktopActivity = new ActivityStore(
  process.env.NODE_ENV === "test"
    ? { filePath: path.join(tmpdir(), `swarmx-activity-test-${process.pid}.jsonl`) }
    : {},
);

export interface RegisterIpcHandlersOptions {
  updateService?: DesktopUpdateServiceLike;
  broadcastUpdateState?: (state: DesktopUpdateState) => void;
  activityStore?: ActivityStore;
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

export function sessionChatMessages(session: SessionData | null): ChatMessage[] {
  if (!session) return [];
  return session.messages.flatMap((message): ChatMessage[] => {
    if (message.kind !== "message") return [];
    if (!isChatRole(message.role)) return [];
    return [{ role: message.role, content: message.content }];
  });
}

function isChatRole(role: string): role is ChatMessage["role"] {
  return role === "user" || role === "assistant" || role === "system" || role === "tool";
}

function timedMessages(
  messages: readonly MessageChunk[],
  startedAtMs: number,
  endedAtMs = Date.now(),
): MessageChunk[] {
  const startedAt = new Date(startedAtMs).toISOString();
  const endedAt = new Date(endedAtMs).toISOString();
  const durationMs = Math.max(1, endedAtMs - startedAtMs);
  return messages.map((message) => ({
    ...message,
    render: {
      ...(message.render ?? {}),
      startedAt: message.render?.startedAt ?? startedAt,
      endedAt: message.render?.endedAt ?? endedAt,
      durationMs: message.render?.durationMs ?? durationMs,
    },
  }));
}

function publishSessionMessages(sender: AgentChunkSender, sessionId: string): void {
  if (!sender.isDestroyed()) sender.send("session:messages", { sessionId });
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

interface ActivityOutcomeInput {
  taskId: string;
  sessionId?: string;
  harnessId?: string;
  modelId?: string;
  reasoningEffort?: string;
  status: "completed" | "failed" | "canceled";
  startedAt: number;
  userText: string;
  messages: readonly MessageChunk[];
  tokenUsages: readonly ModelTokenUsage[];
}

function recordActivityOutcome(store: ActivityStore, input: ActivityOutcomeInput): void {
  const durationMs = Math.max(0, Date.now() - input.startedAt);
  const metadata = {
    taskId: input.taskId,
    sessionId: input.sessionId,
    harnessId: input.harnessId,
    modelId: input.modelId,
    reasoningEffort: input.reasoningEffort,
  };
  appendActivity(store, {
    type: "task_finished",
    ...metadata,
    status: input.status,
    durationMs,
  });
  const usage =
    input.tokenUsages.length > 0
      ? mergeModelTokenUsage(input.tokenUsages)
      : estimateModelTokenUsage(input.userText, input.messages, {
          model: input.modelId,
          provider: input.harnessId,
        });
  appendActivity(store, { type: "token_usage", ...metadata, tokens: usage });
}

function appendActivity(store: ActivityStore, input: ActivityEventInput): void {
  try {
    store.append(input);
  } catch (error) {
    console.warn(`Failed to persist local activity: ${errorMessage(error)}`);
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
  const activityStore = options.activityStore ?? desktopActivity;
  if (options.broadcastUpdateState) updateService.subscribe(options.broadcastUpdateState);
  ipcMain.handle(
    "agent:send",
    async (
      event: IpcMainInvokeEvent,
      params: {
        requestId: string;
        sessionId?: string;
        harnessId: string;
        userText: string;
        agentConfig?: AgentConfig;
        agentComposition?: AgentComposition;
        swarmConfig?: SwarmConfig;
        cwd?: string;
      },
    ) => {
      const startedAt = Date.now();
      const observedMessages: MessageChunk[] = [];
      const tokenUsages: ModelTokenUsage[] = [];
      let foregroundRuntime: ClaudeSessionRuntime | undefined;
      const taskMetadata = {
        taskId: params.requestId,
        sessionId: params.sessionId,
        harnessId: params.harnessId,
        modelId: stringProperty(params.agentComposition, "modelId"),
        reasoningEffort: stringProperty(params.agentComposition, "effort"),
      };
      appendActivity(activityStore, { type: "task_started", ...taskMetadata });
      try {
        const result = await agentRequests.run(event.sender, params.requestId, async () => {
          const publishChunk = agentChunkPublisher(event.sender, params.requestId);
          const onChunk = (chunk: MessageChunk) => {
            observedMessages.push(chunk);
            publishChunk(chunk);
            if (chunk.kind === "tool_call" && chunk.toolName) {
              appendActivity(activityStore, {
                type: "tool_called",
                ...taskMetadata,
                name: chunk.toolName,
              });
            }
          };
          const acpPermissionHandler: AcpPermissionHandler = async (request) => {
            const optionIds = request.options.map((option) => option.optionId);
            if (optionIds.length === 0 || new Set(optionIds).size !== optionIds.length) {
              return { outcome: { outcome: "cancelled" } };
            }
            const title = boundedPermissionLabel(request.toolCall.title ?? "ACP tool request");
            const toolKind = request.toolCall.kind
              ? boundedPermissionLabel(request.toolCall.kind)
              : undefined;
            try {
              const response = await agentInteractions.request(event.sender, params.requestId, {
                kind: "tool_approval",
                title,
                ...(toolKind ? { toolKind } : {}),
                source: "acp",
                summary:
                  "An ACP Harness requested permission for this tool call. Raw input and output are not shown in the approval payload.",
                options: request.options.map((option) => ({
                  optionId: option.optionId,
                  name: boundedPermissionLabel(option.name),
                  kind: option.kind,
                })),
              });
              if (response.kind !== "tool_approval") {
                await recordPermissionDecision({
                  source: "acp",
                  toolName: title,
                  ...(toolKind ? { toolKind } : {}),
                  decision: "cancelled",
                });
                return { outcome: { outcome: "cancelled" } };
              }
              const selected = request.options.find(
                (option) => option.optionId === response.optionId,
              );
              await recordPermissionDecision({
                source: "acp",
                toolName: title,
                ...(toolKind ? { toolKind } : {}),
                decision: selected?.kind.startsWith("allow") ? "allowed" : "rejected",
                ...(selected ? { optionKind: selected.kind } : {}),
              });
              return { outcome: { outcome: "selected", optionId: response.optionId } };
            } catch (error) {
              await recordPermissionDecision({
                source: "acp",
                toolName: title,
                ...(toolKind ? { toolKind } : {}),
                decision: "cancelled",
              });
              throw error;
            }
          };
          let swarm: Swarm;
          const cwd = await normalizeWorkingDirectory(params.cwd);

          if (params.swarmConfig) {
            assertDesktopSwarmModels(params.swarmConfig);
            const config = cwd
              ? swarmConfigWithWorkingDirectory(params.swarmConfig, cwd)
              : params.swarmConfig;
            swarm = new Swarm(await protectSwarmConfigBackends(config), {
              agent: { acpPermissionHandler },
            });
          } else if (params.agentComposition) {
            const inventory = await modelCatalog.list(await loadDesktopExtensionInventory());
            const plan = resolveAgentCompositionPlan(params.agentComposition, inventory);
            for (const skillId of new Set(plan.skills.map((skill) => skill.id))) {
              appendActivity(activityStore, {
                type: "skill_used",
                ...taskMetadata,
                name: skillId,
              });
            }
            assertCompositionSupplyReady(inventory, plan, process.env);
            const providerSecrets = plan.modelSupplyId
              ? await modelCatalog.runtimeSecretsForSupply(inventory, plan.modelSupplyId)
              : {};
            const protectedInventory = await protectCompositionHarness(inventory, plan.harnessId);
            const projectTools =
              cwd && compositionRuntimeHarnessId(inventory, plan) === "swarmx"
                ? new WorkspaceTools(cwd)
                : null;
            const permissionPolicy = projectTools
              ? await permissionService.resolve({
                  cwd,
                  agentId: plan.agentProfileId ?? plan.agentId,
                  agentPolicy: HarnessPermissionPolicySchema.parse({
                    mode: plan.permissions?.mode ?? "default",
                    allowedTools: plan.permissions?.allowedTools ?? [],
                    deniedTools: plan.permissions?.deniedTools ?? [],
                  }),
                })
              : undefined;
            const selectedWorkspaceSkills = plan.skills.flatMap((skillRef) => {
              if (skillRef.status !== "ok") return [];
              const matches = inventory.skills.filter((skill) => skill.id === skillRef.id);
              if (matches.length !== 1) return [];
              const skill = matches[0];
              const filePath = skill?.canonicalPath ?? skill?.path;
              if (!skill || !filePath || !path.isAbsolute(filePath)) return [];
              return [
                {
                  id: skill.id,
                  ...(skill.name ? { name: skill.name } : {}),
                  filePath,
                  ...(skill.description ? { description: skill.description } : {}),
                },
              ];
            });
            const baseWorkspaceToolOptions: WorkspaceAgentToolOptions = {
              ...((plan.modelId ?? plan.runtimeModel)
                ? { model: [plan.modelId, plan.runtimeModel].filter(Boolean).join(" ") }
                : {}),
              ...(plan.apiProtocol ? { apiProtocol: plan.apiProtocol } : {}),
              ...(selectedWorkspaceSkills.length > 0 ? { skills: selectedWorkspaceSkills } : {}),
              ...(plan.effort ? { effort: plan.effort } : {}),
              ...(permissionPolicy ? { permissionPolicy } : {}),
              ...(projectTools && lspHost.supportsClaudeOperations(inventory)
                ? {
                    lsp: (request) => lspHost.operate(inventory, projectTools.root, request),
                  }
                : {}),
            };
            const sessionRuntime =
              projectTools &&
              params.sessionId &&
              workspaceToolProfile(baseWorkspaceToolOptions) === "claude_code"
                ? await claudeSessionRuntimes.open(params.sessionId, projectTools.root)
                : undefined;
            if (sessionRuntime && params.sessionId) {
              const sessionId = params.sessionId;
              sessionRuntime.configure({
                activate: async (activation) => {
                  const activationMessage: MessageChunk = {
                    role: "system",
                    content: activation.prompt,
                    kind: "message",
                  };
                  if (!appendMessages(sessionId, [activationMessage])) {
                    throw new Error(`Session ${sessionId} no longer exists.`);
                  }
                  publishSessionMessages(event.sender, sessionId);
                  const persisted = loadSession(sessionId);
                  if (!persisted) throw new Error(`Session ${sessionId} no longer exists.`);
                  const backgroundTools = new WorkspaceTools(sessionRuntime.root);
                  const backgroundToolOptions: WorkspaceAgentToolOptions = {
                    ...baseWorkspaceToolOptions,
                    sessionId,
                    sessionTools: sessionRuntime,
                    borrowShell: true,
                  };
                  const messages = await executeAgentComposition(
                    params.agentComposition,
                    [
                      {
                        role: "system",
                        content: projectAgentContextMessage(
                          sessionRuntime.root,
                          backgroundToolOptions,
                        ),
                      },
                      ...sessionChatMessages(persisted),
                    ],
                    {
                      inventory: protectedInventory,
                      providerSecrets,
                      cwd: sessionRuntime.root,
                      acpPermissionHandler,
                      localTools: workspaceAgentTools(
                        backgroundTools,
                        sessionRuntime.shell,
                        backgroundToolOptions,
                      ),
                    },
                  );
                  assertFinalAssistantMessage(messages);
                  if (!appendMessages(sessionId, messages)) {
                    throw new Error(`Session ${sessionId} no longer exists.`);
                  }
                  publishSessionMessages(event.sender, sessionId);
                },
                onActivationError: (_activation, error) => {
                  const message: MessageChunk = {
                    role: "system",
                    content: `Background activation failed: ${errorMessage(error)}`,
                    kind: "message",
                  };
                  if (appendMessages(sessionId, [message])) {
                    publishSessionMessages(event.sender, sessionId);
                  }
                },
              });
              await sessionRuntime.beginForeground();
              foregroundRuntime = sessionRuntime;
            }
            const interactWithPermissionReceipts: NonNullable<
              WorkspaceAgentToolOptions["interact"]
            > = async (request) => {
              try {
                const response = await agentInteractions.request(
                  event.sender,
                  params.requestId,
                  request,
                );
                if (request.kind === "tool_approval" && response.kind === "tool_approval") {
                  const selected = request.options.find(
                    (option) => option.optionId === response.optionId,
                  );
                  await recordPermissionDecision({
                    source: request.source ?? "direct",
                    toolName: request.title,
                    ...(request.toolKind ? { toolKind: request.toolKind } : {}),
                    decision: selected?.kind.startsWith("allow") ? "allowed" : "rejected",
                    ...(selected ? { optionKind: selected.kind } : {}),
                    policySourceIds: request.policySourceIds ?? [],
                  });
                }
                return response;
              } catch (error) {
                if (request.kind === "tool_approval") {
                  await recordPermissionDecision({
                    source: request.source ?? "direct",
                    toolName: request.title,
                    ...(request.toolKind ? { toolKind: request.toolKind } : {}),
                    decision: "cancelled",
                    policySourceIds: request.policySourceIds ?? [],
                  });
                }
                throw error;
              }
            };
            const childAgentHost = projectTools
              ? new ClaudeChildAgentHost({
                  parentModel: [plan.modelId, plan.runtimeModel].filter(Boolean).join(" "),
                  root: () => projectTools.root,
                  systemContext: (root) =>
                    projectAgentContextMessage(root, {
                      ...baseWorkspaceToolOptions,
                      sessionId: `${params.sessionId ?? params.requestId}:agent`,
                      interact: interactWithPermissionReceipts,
                    }),
                  execute: async ({ agentId, root, messages: childMessages }) => {
                    const childTools = new WorkspaceTools(root);
                    const childToolOptions: WorkspaceAgentToolOptions = {
                      ...baseWorkspaceToolOptions,
                      sessionId: `${params.sessionId ?? params.requestId}:agent:${agentId}`,
                      interact: interactWithPermissionReceipts,
                      ...(lspHost.supportsClaudeOperations(inventory)
                        ? {
                            lsp: (request) => lspHost.operate(inventory, childTools.root, request),
                          }
                        : {}),
                    };
                    const childUsages: ModelTokenUsage[] = [];
                    const messages = await executeAgentComposition(
                      params.agentComposition,
                      childMessages,
                      {
                        inventory: protectedInventory,
                        providerSecrets,
                        cwd: root,
                        acpPermissionHandler,
                        localTools: workspaceAgentTools(childTools, undefined, childToolOptions),
                        onUsage: (usage) => childUsages.push(usage),
                      },
                    );
                    return { messages, usages: childUsages };
                  },
                })
              : null;
            const workspaceToolOptions: WorkspaceAgentToolOptions = {
              ...baseWorkspaceToolOptions,
              sessionId: params.sessionId ?? params.requestId,
              ...(sessionRuntime ? { sessionTools: sessionRuntime, borrowShell: true } : {}),
              ...(childAgentHost ? { agent: (request) => childAgentHost.run(request) } : {}),
              interact: interactWithPermissionReceipts,
              closeInteractions: () => {
                childAgentHost?.close();
                agentInteractions.cancelRequest(event.sender, params.requestId);
              },
            };
            const sessionMessages = params.sessionId
              ? sessionChatMessages(loadSession(params.sessionId))
              : [];
            const messages = await executeAgentComposition(
              params.agentComposition,
              [
                ...(projectTools
                  ? [
                      {
                        role: "system" as const,
                        content: projectAgentContextMessage(
                          cwd ?? desktopWorkspaceRoot,
                          workspaceToolOptions,
                        ),
                      },
                    ]
                  : []),
                ...(sessionMessages.length > 0
                  ? sessionMessages
                  : [{ role: "user" as const, content: params.userText }]),
              ],
              {
                inventory: protectedInventory,
                providerSecrets,
                cwd,
                acpPermissionHandler,
                ...(projectTools
                  ? {
                      localTools: workspaceAgentTools(
                        projectTools,
                        sessionRuntime?.shell,
                        workspaceToolOptions,
                      ),
                    }
                  : {}),
                onChunk,
                onUsage: (usage) => tokenUsages.push(usage),
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
            (usage) => tokenUsages.push(usage),
          );

          return { success: true, messages: result };
        });
        const persistedMessages = timedMessages(result.messages, startedAt);
        const sessionPersisted = params.sessionId
          ? appendMessages(params.sessionId, persistedMessages)
          : false;
        recordActivityOutcome(activityStore, {
          ...taskMetadata,
          status: "completed",
          startedAt,
          userText: params.userText,
          messages: persistedMessages,
          tokenUsages,
        });
        return { ...result, messages: persistedMessages, sessionPersisted };
      } catch (err) {
        if (err instanceof RequestCancelledError) {
          const canceledMessages = timedMessages(observedMessages, startedAt);
          const sessionPersisted = params.sessionId
            ? appendMessages(params.sessionId, canceledMessages)
            : false;
          recordActivityOutcome(activityStore, {
            ...taskMetadata,
            status: "canceled",
            startedAt,
            userText: params.userText,
            messages: observedMessages,
            tokenUsages,
          });
          return {
            success: false,
            canceled: true,
            requestId: params.requestId,
            sessionPersisted,
          };
        }
        const error = err instanceof Error ? err.message : String(err);
        const failedMessages = [
          ...timedMessages(observedMessages, startedAt),
          { role: "system", content: `Error: ${error}`, kind: "message" as const },
        ];
        const sessionPersisted = params.sessionId
          ? appendMessages(params.sessionId, failedMessages)
          : false;
        recordActivityOutcome(activityStore, {
          ...taskMetadata,
          status: "failed",
          startedAt,
          userText: params.userText,
          messages: observedMessages,
          tokenUsages,
        });
        return {
          success: false,
          error,
          sessionPersisted,
        };
      } finally {
        foregroundRuntime?.endForeground();
      }
    },
  );

  ipcMain.handle("activity:profile", () => activityStore.summary());

  ipcMain.handle(
    "agent:cancel",
    async (event: IpcMainInvokeEvent, params: { requestId: string }) => ({
      requestId: params.requestId,
      canceled: await agentRequests.cancel(event.sender, params.requestId),
    }),
  );

  ipcMain.handle(
    "agent:resolveInteraction",
    (event: IpcMainInvokeEvent, resolution: DesktopAgentInteractionResolution) => ({
      requestId: resolution.requestId,
      interactionId: resolution.interactionId,
      resolved: agentInteractions.resolve(event.sender, resolution),
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

  ipcMain.handle("session:delete", async (_event: IpcMainInvokeEvent, id: string) => {
    await claudeSessionRuntimes.delete(id);
    return deleteSession(id);
  });

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

  ipcMain.handle(
    "permission:status",
    async (
      _event: IpcMainInvokeEvent,
      params?: { cwd?: string; agentId?: string; agentPolicy?: unknown },
    ) =>
      permissionService.status({
        cwd: await normalizeWorkingDirectory(params?.cwd),
        ...(params?.agentId ? { agentId: params.agentId } : {}),
        ...(params?.agentPolicy
          ? { agentPolicy: HarnessPermissionPolicySchema.parse(params.agentPolicy) }
          : {}),
      }),
  );

  ipcMain.handle(
    "permission:savePersonal",
    async (
      _event: IpcMainInvokeEvent,
      params: { cwd?: string; agentId?: string; agentPolicy?: unknown; policy: unknown },
    ) => {
      await permissionService.savePersonalPolicy(params.policy);
      return permissionService.status({
        cwd: await normalizeWorkingDirectory(params.cwd),
        ...(params.agentId ? { agentId: params.agentId } : {}),
        ...(params.agentPolicy
          ? { agentPolicy: HarnessPermissionPolicySchema.parse(params.agentPolicy) }
          : {}),
      });
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
  void claudeSessionRuntimes.close();
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

function stringProperty(value: unknown, key: string): string | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const property = (value as Record<string, unknown>)[key];
  return typeof property === "string" && property.length > 0 ? property : undefined;
}

async function recordPermissionDecision(input: RecordPermissionDecisionInput): Promise<void> {
  try {
    await permissionService.recordDecision(input);
  } catch {
    // An unavailable audit store must not rewrite the user's authority decision.
  }
}

function boundedPermissionLabel(value: string): string {
  const compact = value.replace(/\s+/g, " ").trim();
  if (!compact) return "Tool permission request";
  if (SENSITIVE_PERMISSION_LABEL_PATTERN.test(compact)) return "Tool permission request";
  return compact.length <= 160 ? compact : `${compact.slice(0, 159)}…`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
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
