import { chmod, mkdir, readFile, realpath, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type {
  AcpPermissionHandler,
  ActivityEventInput,
  AgentBackend,
  AgentComposition,
  AgentCompositionPlan,
  AgentConfig,
  AuditInput,
  AuditQuery,
  AuditVerification,
  DiscoveredSession,
  ExtensionInventory,
  ExternalAcpSessionBinding,
  ListGroupedSessionsOptions,
  MediaAttachment,
  MessageChunk,
  ModelTokenUsage,
  ProjectData,
  SessionData,
  SessionPermissionMode,
  SessionSummary,
  SwarmConfig,
  TransientSessionData,
} from "@swarmx/core";
import {
  ActivityStore,
  AuditStore,
  appendMessages,
  archiveProjectSessions,
  archiveSession,
  createSession,
  detectMediaMimeType,
  dismissProject,
  editSessionUserMessage,
  estimateModelTokenUsage,
  executeAgentComposition,
  forkSession,
  getHarness,
  HarnessPermissionPolicySchema,
  importN8nWorkflow,
  listGroupedSessions,
  listProjects,
  listSessionSummaries,
  loadDiscoveredSession,
  loadExtensionInventory,
  loadSession,
  mergeModelTokenUsage,
  RequestCancelledError,
  registerDefaultProject,
  registerProject,
  renameProject,
  resolveAgentCompositionPlan,
  Swarm,
  saveSession,
  setProjectPinned,
  setSessionPinned,
  updateSessionTitle,
  validateMediaAttachments,
} from "@swarmx/core";
import {
  containerHostBridgeUrl,
  HarnessDoctor,
  HarnessEnvironmentService,
  type HarnessEnvironmentSetupRequest,
  type HarnessEnvironmentStatus,
} from "@swarmx/runtime";
import { dialog, ipcMain as electronIpcMain, type IpcMainInvokeEvent, shell } from "electron";
import {
  createEphemeralCodexHome,
  createExternalAcpSessionBinding,
  type ExternalAcpSessionIdentity,
  externalAcpSessionIdentity,
  latestUserMessageHasAttachments,
  matchingExternalAcpSessionId,
  sameExternalAcpSessionIdentity,
} from "./acp-session-runtime.js";
import {
  type AgentChunkPublisher,
  type AgentChunkSender,
  agentChunkPublisher,
} from "./agent-chunk-publisher.js";
import {
  AgentInteractionBroker,
  type DesktopAgentInteractionResolution,
} from "./agent-interactions.js";
import { type BrowserBounds, BrowserHost } from "./browser-host.js";
import { BuiltinToolSettingsService, resolveRunBuiltinTools } from "./builtin-tool-settings.js";
import { ClaudeChildAgentHost } from "./child-agent-host.js";
import {
  type ClaudeSessionRuntime,
  ClaudeSessionRuntimeRegistry,
} from "./claude-session-runtime.js";
import { CodexAccessTokenResolver } from "./codex-auth.js";
import { ComposerPreferenceService } from "./composer-preferences.js";
import { CustomAgentService } from "./custom-agents.js";
import { DesktopExtensionManager } from "./extension-manager.js";
import { type LspCompletionRequest, LspHost, type LspStopRequest } from "./lsp-host.js";
import { DesktopMediaService } from "./media.js";
import {
  type ManualModelInput,
  ModelCatalogService,
  type ProviderRuntimeCredentials,
  type UserProviderInput,
} from "./model-catalog.js";
import { PermissionService, type RecordPermissionDecisionInput } from "./permission-service.js";
import { FileProviderAuthStore } from "./provider-auth.js";
import { providerErrorMessage } from "./provider-error.js";
import {
  type ProviderKeyAttemptObservation,
  ProviderKeyPoolRuntime,
  ProviderKeyUsageStore,
} from "./provider-key-pool.js";
import {
  type ProviderUsageRefreshTarget,
  ProviderUsageService,
  queryCodexAppServerRequest,
} from "./provider-usage.js";
import { DesktopRequestRegistry } from "./request-registry.js";
import {
  assertFinalAssistantMessage,
  interruptedMessages,
  publishSessionMessages,
  sessionChatMessages,
  timedMessages,
} from "./session-messages.js";
import {
  generatedSessionTitle,
  isPlaceholderSessionTitle,
  normalizeManualSessionTitle,
  SESSION_TITLE_MODEL_ID,
  sessionTitleMessages,
} from "./session-title.js";
import { DesktopSettingsStore } from "./settings-store.js";
import { SideChatService } from "./side-chat-service.js";
import { type TerminalAuditEvent, TerminalHost } from "./terminal-host.js";
import {
  createDisabledDesktopUpdateService,
  type DesktopUpdateServiceLike,
  type DesktopUpdateState,
} from "./updater.js";
import type { RendererIpcEvent } from "./window-security.js";
import {
  projectAgentContextMessage,
  type WorkspaceAgentToolOptions,
  WorkspaceTools,
  workspaceAgentTools,
  workspaceToolProfile,
} from "./workspace-tools.js";

export type { AgentChunkPublisher, AgentChunkSender };
export {
  agentChunkPublisher,
  assertFinalAssistantMessage,
  interruptedMessages,
  sessionChatMessages,
};

const MAX_INLINE_IMAGE_BYTES = 25 * 1024 * 1024;
const SENSITIVE_PERMISSION_LABEL_PATTERN =
  /(api[_ -]?key|access[_ -]?token|password|passwd|bearer\s+[a-z0-9]|secret\s*[=:]|private[_ -]?key)/i;
type IpcAuditPolicy = "intent_outcome" | "failure_only" | "semantic_only";
const IPC_AUDIT_POLICIES = {
  "bootstrap:get": "intent_outcome",
  "agent:send": "intent_outcome",
  "sideChat:send": "intent_outcome",
  "activity:profile": "failure_only",
  "agent:cancel": "intent_outcome",
  "sideChat:list": "intent_outcome",
  "sideChat:create": "intent_outcome",
  "sideChat:update": "failure_only",
  "sideChat:activate": "failure_only",
  "sideChat:setHidden": "failure_only",
  "sideChat:addContext": "intent_outcome",
  "sideChat:edit": "intent_outcome",
  "sideChat:delete": "intent_outcome",
  "sideChat:promote": "intent_outcome",
  "sideChat:cancel": "intent_outcome",
  "agent:resolveInteraction": "semantic_only",
  "session:create": "intent_outcome",
  "session:save": "intent_outcome",
  "session:load": "intent_outcome",
  "session:list": "intent_outcome",
  "project:list": "intent_outcome",
  "project:addExisting": "intent_outcome",
  "project:createScratch": "intent_outcome",
  "project:setPinned": "intent_outcome",
  "project:rename": "intent_outcome",
  "project:reveal": "intent_outcome",
  "project:archiveTasks": "intent_outcome",
  "project:remove": "intent_outcome",
  "session:listGrouped": "intent_outcome",
  "session:loadDiscovered": "intent_outcome",
  "session:archive": "intent_outcome",
  "session:rename": "intent_outcome",
  "session:setPinned": "intent_outcome",
  "session:generateTitle": "intent_outcome",
  "session:appendMessages": "intent_outcome",
  "session:editUserMessage": "intent_outcome",
  "session:fork": "intent_outcome",
  "workflow:importN8n": "failure_only",
  "extension:list": "intent_outcome",
  "extension:managementState": "failure_only",
  "extension:saveSource": "intent_outcome",
  "extension:refreshSource": "intent_outcome",
  "extension:removeSource": "intent_outcome",
  "extension:applyAction": "intent_outcome",
  "extension:saveEvolutionPolicy": "intent_outcome",
  "customAgent:list": "intent_outcome",
  "customAgent:save": "intent_outcome",
  "customAgent:remove": "intent_outcome",
  "composerPreferences:get": "failure_only",
  "composerPreferences:save": "intent_outcome",
  "builtinToolSettings:get": "failure_only",
  "builtinToolSettings:save": "intent_outcome",
  "permission:status": "intent_outcome",
  "permission:savePersonal": "intent_outcome",
  "permission:saveProfiles": "intent_outcome",
  "workspace:root": "intent_outcome",
  "workspace:review": "intent_outcome",
  "workspace:listDirectory": "intent_outcome",
  "workspace:readFile": "intent_outcome",
  "terminal:create": "semantic_only",
  "terminal:write": "semantic_only",
  "terminal:resize": "semantic_only",
  "terminal:kill": "semantic_only",
  "browser:create": "intent_outcome",
  "browser:navigate": "intent_outcome",
  "browser:back": "intent_outcome",
  "browser:forward": "intent_outcome",
  "browser:reload": "intent_outcome",
  "browser:setBounds": "failure_only",
  "browser:setVisible": "failure_only",
  "browser:destroy": "intent_outcome",
  "appUpdate:getState": "failure_only",
  "appUpdate:install": "intent_outcome",
  "workspace:selectFilesAndFolders": "intent_outcome",
  "media:select": "intent_outcome",
  "media:import": "intent_outcome",
  "media:preview": "intent_outcome",
  "media:open": "intent_outcome",
  "media:reveal": "intent_outcome",
  "modelCatalog:refresh": "intent_outcome",
  "modelCatalog:addManualModel": "intent_outcome",
  "modelCatalog:removeManualModel": "intent_outcome",
  "modelCatalog:saveProvider": "intent_outcome",
  "modelCatalog:removeProvider": "intent_outcome",
  "modelCatalog:resetProviderKey": "intent_outcome",
  "providerUsage:refresh": "intent_outcome",
  "harnessEnvironment:get": "intent_outcome",
  "harnessEnvironment:version": "intent_outcome",
  "doctor:inspect": "intent_outcome",
  "doctor:fix": "intent_outcome",
  "harnessEnvironment:setup": "intent_outcome",
  "lsp:complete": "intent_outcome",
  "lsp:stop": "intent_outcome",
  "asset:imageDataUrl": "intent_outcome",
} as const satisfies Record<string, IpcAuditPolicy>;
const lspHost = new LspHost();
const harnessEnvironment = new HarnessEnvironmentService();
const harnessDoctor = new HarnessDoctor(harnessEnvironment);
const agentRequests = new DesktopRequestRegistry();
const sideChats = new SideChatService();
const agentInteractions = new AgentInteractionBroker();
const claudeSessionRuntimes = new ClaudeSessionRuntimeRegistry();
const browserHost = new BrowserHost();
const desktopAudit = new AuditStore(
  process.env.NODE_ENV === "test"
    ? {
        filePath: path.join(tmpdir(), `swarmx-audit-test-${process.pid}`, "events.jsonl"),
      }
    : {},
);
type DesktopAuditStore = Pick<AuditStore, "append" | "query" | "exportJsonl" | "verify">;
type PreparedAuditInput = AuditInput & { metadata: Record<string, unknown> };
let activeAuditStore: DesktopAuditStore = desktopAudit;
let semanticAuditCount = 0;
const terminalHost = new TerminalHost(undefined, undefined, undefined, recordTerminalAudit);
const interactiveOwnerIds = new Set<number>();
const desktopWorkspaceRoot = process.env.INIT_CWD || process.cwd();
const workspaceTools = new WorkspaceTools(desktopWorkspaceRoot);
const desktopSettingsStore = new DesktopSettingsStore();
const providerAuthStore = new FileProviderAuthStore();
const codexAccessTokenProvider = new CodexAccessTokenResolver({
  refresh: () => queryCodexAppServerRequest("account/read", { refreshToken: true }),
});
const providerKeyUsageStore = new ProviderKeyUsageStore(
  process.env.NODE_ENV === "test"
    ? { path: path.join(tmpdir(), `swarmx-provider-key-usage-test-${process.pid}.json`) }
    : {},
);
const providerKeyPoolRuntime = new ProviderKeyPoolRuntime(providerKeyUsageStore);
const modelCatalog = new ModelCatalogService({
  authStore: providerAuthStore,
  codexAccessTokenProvider,
  keyUsageStore: providerKeyUsageStore,
  settingsStore: desktopSettingsStore,
});
const composerPreferences = new ComposerPreferenceService(desktopSettingsStore);
const builtinToolSettings = new BuiltinToolSettingsService(desktopSettingsStore);
const customAgents = new CustomAgentService(desktopSettingsStore);
const permissionService = new PermissionService(desktopSettingsStore);
const mediaService = new DesktopMediaService(
  process.env.NODE_ENV === "test"
    ? path.join(tmpdir(), `swarmx-media-test-${process.pid}`)
    : undefined,
);
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
  auditStore?: DesktopAuditStore;
  authorizeIpcSender?: (event: RendererIpcEvent) => boolean;
}

interface DesktopAgentSendParams {
  requestId: string;
  sessionId?: string;
  sideChatId?: string;
  sideChatVisible?: boolean;
  sideEditMessageIndex?: number;
  harnessId: string;
  userText: string;
  agentConfig?: AgentConfig;
  agentComposition?: AgentComposition;
  swarmConfig?: SwarmConfig;
  cwd?: string;
}

async function executeWithProviderRuntime<T>(
  runtime: ProviderRuntimeCredentials | undefined,
  routingKey: string,
  run: (
    providerSecrets: Record<string, string>,
    observation: ProviderKeyAttemptObservation,
  ) => Promise<T>,
): Promise<T> {
  const idleObservation: ProviderKeyAttemptObservation = {
    markOutput: () => undefined,
    recordUsage: () => undefined,
  };
  if (!runtime?.pooled) {
    const candidate = runtime?.candidates[0];
    return run(
      candidate && runtime ? { [runtime.providerId]: candidate.value } : {},
      idleObservation,
    );
  }
  return providerKeyPoolRuntime.execute({
    providerId: runtime.providerId,
    routingKey,
    candidates: runtime.candidates,
    run: (candidate, observation) => run({ [runtime.providerId]: candidate.value }, observation),
  });
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
  skillIds: readonly string[];
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
  const usage =
    input.tokenUsages.length > 0
      ? mergeModelTokenUsage(input.tokenUsages)
      : estimateModelTokenUsage(input.userText, input.messages, {
          model: input.modelId,
          provider: input.harnessId,
        });
  appendActivity(store, {
    type: "run_summary",
    ...metadata,
    status: input.status,
    durationMs,
    tokens: usage,
    tools: activityNameCounts(
      input.messages.flatMap((message) =>
        message.kind === "tool_call" && message.toolName ? [message.toolName] : [],
      ),
    ),
    skills: activityNameCounts(input.skillIds),
  });
}

function activityNameCounts(names: readonly string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  let distinctNames = 0;
  for (const value of names) {
    const name = safeAuditToken(value);
    if (!name) continue;
    if (counts[name] === undefined) {
      if (distinctNames >= 128) continue;
      distinctNames += 1;
      counts[name] = 0;
    }
    counts[name] += 1;
  }
  return counts;
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
  const auditStore = options.auditStore ?? desktopAudit;
  activeAuditStore = auditStore;
  const authorizeIpcSender = options.authorizeIpcSender ?? (() => false);
  const assertAuthorized = (event: RendererIpcEvent): void => {
    if (!authorizeIpcSender(event)) throw new Error("Untrusted desktop IPC sender.");
  };
  const handle: typeof electronIpcMain.handle = (channel, listener) => {
    if (channel.startsWith("audit:")) {
      return electronIpcMain.handle(channel, (event, ...args) => {
        try {
          assertAuthorized(event);
        } catch (error) {
          auditStore.append({
            ...ipcAuditInput(channel, event, args),
            outcome: "denied",
            metadata: { argumentCount: args.length },
          });
          throw error;
        }
        return listener(event, ...args);
      });
    }
    const auditPolicy = requiredIpcAuditPolicy(channel);
    return electronIpcMain.handle(channel, (event, ...args) => {
      const startedAt = Date.now();
      const semanticAuditBaseline = semanticAuditCount;
      const audit = ipcAuditInput(channel, event, args);
      if (auditPolicy === "intent_outcome") {
        auditStore.append({ ...audit, outcome: "attempted" });
      }
      try {
        assertAuthorized(event);
      } catch (error) {
        auditStore.append({
          ...audit,
          outcome: "denied",
          metadata: { ...audit.metadata, durationMs: elapsedMilliseconds(startedAt) },
        });
        throw error;
      }

      try {
        const result = listener(event, ...args);
        if (isPromiseLike(result)) {
          return Promise.resolve(result).then(
            (value) => {
              const outcome = ipcResultOutcome(value);
              if (recordsResolvedIpcOutcome(auditPolicy, outcome)) {
                auditStore.append({
                  ...audit,
                  outcome,
                  metadata: { ...audit.metadata, durationMs: elapsedMilliseconds(startedAt) },
                });
              }
              return value;
            },
            (error: unknown) => {
              if (recordsDispatchFailure(auditPolicy, semanticAuditBaseline)) {
                auditStore.append({
                  ...audit,
                  outcome: error instanceof RequestCancelledError ? "cancelled" : "failed",
                  metadata: {
                    ...audit.metadata,
                    durationMs: elapsedMilliseconds(startedAt),
                    errorType: errorName(error),
                  },
                });
              }
              throw error;
            },
          );
        }
        const outcome = ipcResultOutcome(result);
        if (recordsResolvedIpcOutcome(auditPolicy, outcome)) {
          auditStore.append({
            ...audit,
            outcome,
            metadata: { ...audit.metadata, durationMs: elapsedMilliseconds(startedAt) },
          });
        }
        return result;
      } catch (error) {
        if (recordsDispatchFailure(auditPolicy, semanticAuditBaseline)) {
          auditStore.append({
            ...audit,
            outcome: error instanceof RequestCancelledError ? "cancelled" : "failed",
            metadata: {
              ...audit.metadata,
              durationMs: elapsedMilliseconds(startedAt),
              errorType: errorName(error),
            },
          });
        }
        throw error;
      }
    });
  };
  const ipcMain = { handle };
  const bootstrapAuditPolicy = requiredIpcAuditPolicy("bootstrap:get");
  electronIpcMain.on("bootstrap:get", (event) => {
    const startedAt = Date.now();
    const audit = ipcAuditInput("bootstrap:get", event, []);
    if (bootstrapAuditPolicy === "intent_outcome") {
      auditStore.append({ ...audit, outcome: "attempted" });
    }
    try {
      assertAuthorized(event);
      event.returnValue = listProjects();
      if (recordsResolvedIpcOutcome(bootstrapAuditPolicy, "completed")) {
        auditStore.append({
          ...audit,
          outcome: "completed",
          metadata: { ...audit.metadata, durationMs: elapsedMilliseconds(startedAt) },
        });
      }
    } catch (error) {
      auditStore.append({
        ...audit,
        outcome: errorMessage(error).includes("Untrusted desktop IPC sender") ? "denied" : "failed",
        metadata: {
          ...audit.metadata,
          durationMs: elapsedMilliseconds(startedAt),
          errorType: errorName(error),
        },
      });
      throw error;
    }
  });
  if (options.broadcastUpdateState) updateService.subscribe(options.broadcastUpdateState);
  const handleAgentSend = async (event: IpcMainInvokeEvent, params: DesktopAgentSendParams) => {
    const startedAt = Date.now();
    const observedMessages: MessageChunk[] = [];
    const tokenUsages: ModelTokenUsage[] = [];
    const usedSkillIds = new Set<string>();
    let foregroundRuntime: ClaudeSessionRuntime | undefined;
    let activeChunkPublisher: AgentChunkPublisher | undefined;
    let activeSideChat: TransientSessionData | undefined;
    let ephemeralCodexHome: Awaited<ReturnType<typeof createEphemeralCodexHome>> | undefined;
    const taskMetadata = {
      taskId: params.requestId,
      sessionId: params.sessionId,
      harnessId: params.harnessId,
      modelId: stringProperty(params.agentComposition, "modelId"),
      reasoningEffort: stringProperty(params.agentComposition, "effort"),
    };
    const desktopRequest = {
      owner: event.sender,
      requestId: params.requestId,
      ...(params.sideChatId
        ? { sessionId: `side:${params.sideChatId}` }
        : params.sessionId
          ? { sessionId: params.sessionId }
          : {}),
    };
    const persistOutcome = (messages: MessageChunk[]) => ({
      sessionPersisted:
        params.sessionId && !params.sideChatId ? appendMessages(params.sessionId, messages) : false,
      sideChat:
        params.sideChatId && params.sessionId && activeSideChat
          ? sideChats.finishRun(params.sessionId, params.sideChatId, params.requestId, messages, {
              unread: params.sideChatVisible === false,
            })
          : undefined,
    });
    try {
      if (params.sessionId && loadSession(params.sessionId)?.archivedAt) {
        throw new Error(`Session "${params.sessionId}" is archived.`);
      }
      if (params.sideChatId) {
        if (!params.sessionId) throw new Error("Side chat sends require a parent Session.");
        if (params.swarmConfig || params.agentConfig || !params.agentComposition) {
          throw new Error(
            "Side chats require one explicit Agent Composition and cannot execute workflows.",
          );
        }
        activeSideChat = sideChats.beginRun(
          params.sessionId,
          params.sideChatId,
          params.requestId,
          params.userText,
          params.sideEditMessageIndex,
        );
      }
      const result = await agentRequests.runForSession(desktopRequest, async () => {
        const publishChunk = agentChunkPublisher(event.sender, params.requestId, {
          ...(params.sideChatId
            ? {
                channel: "sideChat:chunk",
                context: {
                  sideChatId: params.sideChatId,
                  parentSessionId: params.sessionId,
                },
              }
            : {}),
        });
        activeChunkPublisher = publishChunk;
        const onChunk = (chunk: MessageChunk) => {
          recordToolChunkAudit(auditStore, chunk, params);
          if (chunk.kind !== "tool_progress") observedMessages.push(chunk);
          publishChunk(chunk);
        };
        const acpPermissionHandler: AcpPermissionHandler = async (request) => {
          if (params.sideChatId) return { outcome: { outcome: "cancelled" } };
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
              await recordPermissionDecision(
                {
                  source: "acp",
                  toolName: title,
                  ...(toolKind ? { toolKind } : {}),
                  decision: "cancelled",
                },
                {
                  requestId: params.requestId,
                  sessionId: params.sessionId,
                  ownerId: event.sender.id,
                },
              );
              return { outcome: { outcome: "cancelled" } };
            }
            const selected = request.options.find(
              (option) => option.optionId === response.optionId,
            );
            await recordPermissionDecision(
              {
                source: "acp",
                toolName: title,
                ...(toolKind ? { toolKind } : {}),
                decision: selected?.kind.startsWith("allow") ? "allowed" : "rejected",
                ...(selected ? { optionKind: selected.kind } : {}),
              },
              {
                requestId: params.requestId,
                sessionId: params.sessionId,
                ownerId: event.sender.id,
              },
            );
            return { outcome: { outcome: "selected", optionId: response.optionId } };
          } catch (error) {
            await recordPermissionDecision(
              {
                source: "acp",
                toolName: title,
                ...(toolKind ? { toolKind } : {}),
                decision: "cancelled",
              },
              {
                requestId: params.requestId,
                sessionId: params.sessionId,
                ownerId: event.sender.id,
              },
            );
            throw error;
          }
        };
        let swarm: Swarm;
        const cwd = await normalizeWorkingDirectory(params.cwd);

        if (params.swarmConfig) {
          if (params.sessionId && !params.sideChatId) {
            clearExternalAcpSession(params.sessionId);
          }
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
            usedSkillIds.add(skillId);
          }
          assertCompositionSupplyReady(inventory, plan, process.env);
          const providerRuntime = plan.modelSupplyId
            ? await modelCatalog.runtimeCredentialsForSupply(inventory, plan.modelSupplyId)
            : undefined;
          const protectedInventory = await protectCompositionHarness(inventory, plan.harnessId);
          let executionInventory = protectedInventory;
          const runtimeHarnessId = compositionRuntimeHarnessId(inventory, plan);
          const protectedHarness = protectedInventory.harnesses.find(
            (harness) => harness.id === plan.harnessId,
          );
          const compositionBackend = protectedHarness?.backend;
          const compositionUsesAcp = compositionBackend?.type === "custom";
          const compositionIsProtected =
            compositionBackend?.type === "custom" && compositionBackend.program === "container";
          const mainSession =
            params.sessionId && !params.sideChatId ? loadSession(params.sessionId) : null;
          const hasAttachments = mainSession ? latestUserMessageHasAttachments(mainSession) : false;
          const identity =
            runtimeHarnessId && compositionUsesAcp
              ? externalAcpSessionIdentity(plan, runtimeHarnessId, cwd)
              : null;
          let acpSessionId: string | undefined;
          let onAcpSessionId: ((sessionId: string | undefined) => void | Promise<void>) | undefined;
          let compositionEnv: NodeJS.ProcessEnv | undefined;
          const localSessionId = params.sessionId;

          if (mainSession && localSessionId) {
            const canPersistAcpSession =
              Boolean(identity) && compositionUsesAcp && !compositionIsProtected && !hasAttachments;
            if (canPersistAcpSession && identity) {
              acpSessionId = matchingExternalAcpSessionId(mainSession, identity);
              if (mainSession.externalAcpSession && !acpSessionId) {
                clearExternalAcpSession(localSessionId);
              }
              onAcpSessionId = (externalSessionId) => {
                if (externalSessionId) {
                  persistExternalAcpSession(localSessionId, identity, externalSessionId);
                } else {
                  clearExternalAcpSession(localSessionId);
                }
              };
            } else {
              clearExternalAcpSession(localSessionId);
            }

            if (
              runtimeHarnessId === "codex" &&
              compositionUsesAcp &&
              !compositionIsProtected &&
              hasAttachments
            ) {
              ephemeralCodexHome = await createEphemeralCodexHome({
                sourceHome: process.env.CODEX_HOME,
              });
              compositionEnv = { ...process.env, ...ephemeralCodexHome.env };
              executionInventory = inventoryWithHarnessRuntimeEnv(
                protectedInventory,
                plan.harnessId,
                Object.keys(ephemeralCodexHome.env),
              );
            }
          }
          const projectTools =
            cwd && runtimeHarnessId === "swarmx" ? new WorkspaceTools(cwd) : null;
          const directSession = params.sessionId ? loadSession(params.sessionId) : undefined;
          const builtinToolBinding = projectTools
            ? resolveRunBuiltinTools({
                settings: await builtinToolSettings.get(),
                model: inventory.models.find((model) => model.id === plan.modelId),
                ...(directSession ? { session: directSession } : {}),
              })
            : undefined;
          if (
            projectTools &&
            directSession &&
            !params.sideChatId &&
            !directSession.builtinTools &&
            builtinToolBinding
          ) {
            saveSession({
              ...directSession,
              builtinTools: builtinToolBinding,
              ...(plan.modelId ? { model: plan.modelId } : {}),
            });
          }
          const agentPermissionPolicy = HarnessPermissionPolicySchema.parse({
            mode: plan.permissions?.mode ?? "default",
            allowedTools: plan.permissions?.allowedTools ?? [],
            deniedTools: plan.permissions?.deniedTools ?? [],
          });
          const permissionSession =
            projectTools && params.sessionId && !params.sideChatId ? directSession : undefined;
          if (projectTools && params.sessionId && !params.sideChatId && !permissionSession) {
            throw new Error(`Session ${params.sessionId} no longer exists.`);
          }
          const permissionPolicy = projectTools
            ? await permissionService.resolve({
                cwd,
                agentId: plan.agentProfileId ?? plan.agentId,
                agentPolicy: agentPermissionPolicy,
                agentModeDeclared: Boolean(plan.permissions?.mode),
                ...(params.sideChatId
                  ? { sessionPermissionMode: "plan" as const }
                  : permissionSession
                    ? { sessionPermissionMode: permissionSession.permissionMode }
                    : {}),
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
            ...(builtinToolBinding ? { toolStyle: builtinToolBinding.style } : {}),
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
            !params.sideChatId &&
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
                  permissionPolicy: await permissionService.resolve({
                    cwd: sessionRuntime.root,
                    agentId: plan.agentProfileId ?? plan.agentId,
                    agentPolicy: agentPermissionPolicy,
                    agentModeDeclared: Boolean(plan.permissions?.mode),
                    sessionPermissionMode: persisted.permissionMode,
                  }),
                  sessionId,
                  sessionTools: sessionRuntime,
                  borrowShell: true,
                };
                const messages = await executeWithProviderRuntime(
                  providerRuntime,
                  `${sessionId}:background`,
                  (providerSecrets, observation) =>
                    executeAgentComposition(
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
                        onChunk: () => observation.markOutput(),
                        onUsage: (usage) => observation.recordUsage(usage),
                      },
                    ),
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
                await recordPermissionDecision(
                  {
                    source: request.source ?? "direct",
                    toolName: request.title,
                    ...(request.toolKind ? { toolKind: request.toolKind } : {}),
                    decision: selected?.kind.startsWith("allow") ? "allowed" : "rejected",
                    ...(selected ? { optionKind: selected.kind } : {}),
                    policySourceIds: request.policySourceIds ?? [],
                  },
                  {
                    requestId: params.requestId,
                    sessionId: params.sessionId,
                    ownerId: event.sender.id,
                  },
                );
              }
              return response;
            } catch (error) {
              if (request.kind === "tool_approval") {
                await recordPermissionDecision(
                  {
                    source: request.source ?? "direct",
                    toolName: request.title,
                    ...(request.toolKind ? { toolKind: request.toolKind } : {}),
                    decision: "cancelled",
                    policySourceIds: request.policySourceIds ?? [],
                  },
                  {
                    requestId: params.requestId,
                    sessionId: params.sessionId,
                    ownerId: event.sender.id,
                  },
                );
              }
              throw error;
            }
          };
          const childAgentHost =
            projectTools && !params.sideChatId
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
                    const messages = await executeWithProviderRuntime(
                      providerRuntime,
                      `${params.sessionId ?? params.requestId}:agent:${agentId}`,
                      (providerSecrets, observation) =>
                        executeAgentComposition(params.agentComposition, childMessages, {
                          inventory: protectedInventory,
                          providerSecrets,
                          cwd: root,
                          acpPermissionHandler,
                          localTools: workspaceAgentTools(childTools, undefined, childToolOptions),
                          onChunk: () => observation.markOutput(),
                          onUsage: (usage) => {
                            childUsages.push(usage);
                            observation.recordUsage(usage);
                          },
                        }),
                    );
                    return { messages, usages: childUsages };
                  },
                })
              : null;
          const workspaceToolOptions: WorkspaceAgentToolOptions = {
            ...baseWorkspaceToolOptions,
            sessionId: params.sideChatId ?? params.sessionId ?? params.requestId,
            ...(sessionRuntime ? { sessionTools: sessionRuntime, borrowShell: true } : {}),
            ...(childAgentHost ? { agent: (request) => childAgentHost.run(request) } : {}),
            interact: params.sideChatId
              ? async () => {
                  throw new Error("Interactive actions are unavailable in read-only side chat.");
                }
              : interactWithPermissionReceipts,
            closeInteractions: () => {
              childAgentHost?.close();
              agentInteractions.cancelRequest(event.sender, params.requestId);
            },
          };
          const sessionMessages =
            params.sideChatId && params.sessionId
              ? sideChats.modelMessages(params.sessionId, params.sideChatId)
              : params.sessionId
                ? sessionChatMessages(loadSession(params.sessionId))
                : [];
          const sideChatBoundaryMessage = params.sideChatId
            ? [
                {
                  role: "system" as const,
                  content:
                    "You are in a transient read-only side chat fork. Explain, inspect, and answer without modifying files, running mutating commands, requesting permissions, or creating nested side chats. This transcript must not affect the parent task.",
                },
              ]
            : [];
          const messages = await executeWithProviderRuntime(
            providerRuntime,
            params.sideChatId ?? params.sessionId ?? params.requestId,
            (providerSecrets, observation) =>
              executeAgentComposition(
                params.agentComposition,
                [
                  ...sideChatBoundaryMessage,
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
                  inventory: executionInventory,
                  env: compositionEnv,
                  providerSecrets,
                  cwd,
                  acpPermissionHandler,
                  acpSessionId,
                  onAcpSessionId,
                  ...(params.sideChatId ? { acpMode: "plan" } : {}),
                  ...(projectTools
                    ? {
                        localTools: workspaceAgentTools(
                          projectTools,
                          sessionRuntime?.shell,
                          workspaceToolOptions,
                        ),
                      }
                    : {}),
                  onChunk: (chunk) => {
                    observation.markOutput();
                    onChunk(chunk);
                  },
                  onUsage: (usage) => {
                    tokenUsages.push(usage);
                    observation.recordUsage(usage);
                  },
                },
              ),
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
      const { sessionPersisted, sideChat } = persistOutcome(persistedMessages);
      recordActivityOutcome(activityStore, {
        ...taskMetadata,
        status: "completed",
        startedAt,
        userText: params.userText,
        messages: persistedMessages,
        tokenUsages,
        skillIds: [...usedSkillIds],
      });
      return { ...result, messages: persistedMessages, sessionPersisted, sideChat };
    } catch (err) {
      if (err instanceof RequestCancelledError) {
        const canceledMessages = interruptedMessages(observedMessages, startedAt);
        const { sessionPersisted, sideChat } = persistOutcome(canceledMessages);
        recordActivityOutcome(activityStore, {
          ...taskMetadata,
          status: "canceled",
          startedAt,
          userText: params.userText,
          messages: canceledMessages,
          tokenUsages,
          skillIds: [...usedSkillIds],
        });
        return {
          success: false,
          canceled: true,
          requestId: params.requestId,
          sessionPersisted,
          sideChat,
        };
      }
      const error = err instanceof Error ? err.message : String(err);
      const providerMessage = providerErrorMessage(err);
      const terminalMessage =
        providerMessage ??
        ({
          role: "system",
          content: `Error: ${error}`,
          kind: "message" as const,
        } satisfies MessageChunk);
      const failedMessages = [...timedMessages(observedMessages, startedAt), terminalMessage];
      const { sessionPersisted, sideChat } = persistOutcome(failedMessages);
      recordActivityOutcome(activityStore, {
        ...taskMetadata,
        status: "failed",
        startedAt,
        userText: params.userText,
        messages: observedMessages,
        tokenUsages,
        skillIds: [...usedSkillIds],
      });
      return {
        success: false,
        error: providerMessage?.content ?? error,
        messages: failedMessages,
        sessionPersisted,
        sideChat,
      };
    } finally {
      activeChunkPublisher?.close();
      foregroundRuntime?.endForeground();
      await ephemeralCodexHome?.cleanup();
    }
  };

  ipcMain.handle("agent:send", handleAgentSend);
  ipcMain.handle("sideChat:send", handleAgentSend);

  ipcMain.handle("activity:profile", () => activityStore.summary());

  ipcMain.handle("audit:list", (_event: IpcMainInvokeEvent, query?: AuditQuery) =>
    auditStore.query(query ?? {}),
  );

  ipcMain.handle("audit:verify", (): AuditVerification => auditStore.verify());

  ipcMain.handle("audit:export", async (_event: IpcMainInvokeEvent, query?: AuditQuery) => {
    const exportAudit: PreparedAuditInput = {
      category: "system",
      action: "audit.export",
      actor: { kind: "user", id: "desktop" },
      metadata: { filtered: Boolean(query && Object.keys(query).length > 0) },
    };
    auditStore.append({ ...exportAudit, outcome: "attempted" });
    const selected = await dialog.showSaveDialog({
      title: "Export verified audit log",
      defaultPath: `swarmx-audit-${new Date().toISOString().slice(0, 10)}.jsonl`,
      filters: [{ name: "JSON Lines", extensions: ["jsonl"] }],
    });
    if (selected.canceled || !selected.filePath) {
      auditStore.append({ ...exportAudit, outcome: "cancelled" });
      return { exported: false, canceled: true };
    }
    try {
      const jsonl = auditStore.exportJsonl(query ?? {});
      await writeFile(selected.filePath, jsonl, { encoding: "utf8", mode: 0o600 });
      await chmod(selected.filePath, 0o600);
      const eventCount = jsonl ? jsonl.trimEnd().split("\n").length : 0;
      auditStore.append({
        ...exportAudit,
        outcome: "completed",
        metadata: { ...exportAudit.metadata, eventCount },
      });
      return { exported: true, eventCount };
    } catch (error) {
      auditStore.append({
        ...exportAudit,
        outcome: "failed",
        metadata: { ...exportAudit.metadata, errorType: errorName(error) },
      });
      throw error;
    }
  });

  ipcMain.handle(
    "agent:cancel",
    async (event: IpcMainInvokeEvent, params: { requestId: string }) => ({
      requestId: params.requestId,
      canceled: await agentRequests.cancel(event.sender, params.requestId),
    }),
  );

  ipcMain.handle("sideChat:list", (_event: IpcMainInvokeEvent, parentSessionId: string) =>
    sideChats.list(parentSessionId),
  );

  ipcMain.handle(
    "sideChat:create",
    (
      _event: IpcMainInvokeEvent,
      params: {
        parentSessionId: string;
        throughMessageIndex: number;
        expectedMessages: MessageChunk[];
        title?: string;
      },
    ) => sideChats.create(params),
  );

  ipcMain.handle(
    "sideChat:update",
    (
      _event: IpcMainInvokeEvent,
      params: {
        parentSessionId: string;
        sideChatId: string;
        draft?: string;
        attachments?: string[];
        title?: string;
        unread?: boolean;
      },
    ) => sideChats.update(params),
  );

  ipcMain.handle(
    "sideChat:activate",
    (_event: IpcMainInvokeEvent, params: { parentSessionId: string; sideChatId: string }) =>
      sideChats.activate(params.parentSessionId, params.sideChatId),
  );

  ipcMain.handle(
    "sideChat:setHidden",
    (_event: IpcMainInvokeEvent, params: { parentSessionId: string; hidden: boolean }) =>
      sideChats.setPaneHidden(params.parentSessionId, params.hidden),
  );

  ipcMain.handle(
    "sideChat:addContext",
    (
      _event: IpcMainInvokeEvent,
      params: { parentSessionId: string; sideChatId: string; text: string },
    ) => sideChats.addContext(params.parentSessionId, params.sideChatId, params.text),
  );

  ipcMain.handle(
    "sideChat:edit",
    (
      _event: IpcMainInvokeEvent,
      params: {
        parentSessionId: string;
        sideChatId: string;
        messageIndex: number;
        content: string;
      },
    ) =>
      sideChats.edit(
        params.parentSessionId,
        params.sideChatId,
        params.messageIndex,
        params.content,
      ),
  );

  ipcMain.handle(
    "sideChat:delete",
    (_event: IpcMainInvokeEvent, params: { parentSessionId: string; sideChatId: string }) =>
      sideChats.delete(params.parentSessionId, params.sideChatId),
  );

  ipcMain.handle(
    "sideChat:promote",
    (_event: IpcMainInvokeEvent, params: { parentSessionId: string; sideChatId: string }) =>
      sideChats.promote(params.parentSessionId, params.sideChatId),
  );

  ipcMain.handle(
    "sideChat:cancel",
    async (
      event: IpcMainInvokeEvent,
      params: { parentSessionId: string; sideChatId: string; requestId: string },
    ) => {
      sideChats.markStopping(params.parentSessionId, params.sideChatId, params.requestId);
      try {
        const canceled = await agentRequests.cancel(event.sender, params.requestId);
        if (!canceled) {
          sideChats.markRunning(params.parentSessionId, params.sideChatId, params.requestId);
        }
        return {
          requestId: params.requestId,
          sideChatId: params.sideChatId,
          canceled,
        };
      } catch (error) {
        sideChats.markRunning(params.parentSessionId, params.sideChatId, params.requestId);
        throw error;
      }
    },
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
        permissionMode?: SessionPermissionMode;
      },
    ): SessionData => {
      return createSession(params.agentName, params.harness, params.model, {
        projectId: params.projectId,
        cwd: params.cwd,
        permissionMode: params.permissionMode,
      });
    },
  );

  ipcMain.handle("session:save", async (_event: IpcMainInvokeEvent, session: SessionData) => {
    await validateMessageAttachments(session.messages);
    saveSession(session);
  });

  ipcMain.handle("session:load", (_event: IpcMainInvokeEvent, id: string): SessionData | null => {
    return loadSession(id);
  });

  ipcMain.handle("session:list", (): SessionSummary[] => listSessionSummaries());

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
      const runningSession = listSessionSummaries().find(
        (session) =>
          (session.projectId === project.id ||
            (session.cwd && path.resolve(session.cwd) === path.resolve(project.cwd))) &&
          (agentRequests.isSessionActive(session.id) ||
            claudeSessionRuntimes.isRunning(session.id) ||
            sideChats.isParentRunning(session.id)),
      );
      if (runningSession) {
        throw new Error("Stop all running tasks in this project before archiving them.");
      }
      const parentIds = listSessionSummaries()
        .filter(
          (session) =>
            session.projectId === project.id ||
            (session.cwd && path.resolve(session.cwd) === path.resolve(project.cwd)),
        )
        .map((session) => session.id);
      const archived = archiveProjectSessions({ projectId: project.id, cwd: project.cwd });
      for (const parentId of parentIds) sideChats.clearParent(parentId);
      return archived;
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

  ipcMain.handle("session:archive", async (_event: IpcMainInvokeEvent, id: string) => {
    if (
      agentRequests.isSessionActive(id) ||
      claudeSessionRuntimes.isRunning(id) ||
      sideChats.isParentRunning(id)
    ) {
      throw new Error("Stop the task before archiving it.");
    }
    const session = archiveSession(id);
    if (!session) throw new Error(`Unknown session: ${id}`);
    await claudeSessionRuntimes.delete(id);
    sideChats.clearParent(id);
    return session;
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
        const providerRuntime = plan.modelSupplyId
          ? await modelCatalog.runtimeCredentialsForSupply(inventory, plan.modelSupplyId)
          : undefined;
        const messages = await executeWithProviderRuntime(
          providerRuntime,
          `${session.id}:title`,
          (providerSecrets, observation) =>
            executeAgentComposition(composition, sessionTitleMessages(params.userText), {
              inventory,
              providerSecrets,
              onChunk: () => observation.markOutput(),
              onUsage: (usage) => observation.recordUsage(usage),
            }),
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
    async (
      _event: IpcMainInvokeEvent,
      params: { id: string; messages: MessageChunk[] },
    ): Promise<boolean> => {
      await validateMessageAttachments(params.messages);
      return appendMessages(params.id, params.messages);
    },
  );

  ipcMain.handle(
    "session:editUserMessage",
    (
      _event: IpcMainInvokeEvent,
      params: {
        id: string;
        messageIndex: number;
        expectedMessages: MessageChunk[];
        content: string;
      },
    ): SessionData => {
      const session = editSessionUserMessage(params);
      if (!session) throw new Error(`Session "${params.id}" was not found.`);
      return session;
    },
  );

  ipcMain.handle(
    "session:fork",
    (
      _event: IpcMainInvokeEvent,
      params: {
        id: string;
        throughMessageIndex: number;
        expectedMessages: MessageChunk[];
      },
    ): SessionData => {
      const session = forkSession(params);
      if (!session) throw new Error(`Session "${params.id}" was not found.`);
      return session;
    },
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

  ipcMain.handle("composerPreferences:get", () => composerPreferences.get());

  ipcMain.handle("composerPreferences:save", (_event: IpcMainInvokeEvent, input: unknown) =>
    composerPreferences.save(input),
  );

  ipcMain.handle("builtinToolSettings:get", () => builtinToolSettings.get());

  ipcMain.handle("builtinToolSettings:save", (_event: IpcMainInvokeEvent, input: unknown) =>
    builtinToolSettings.save(input),
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

  ipcMain.handle(
    "permission:saveProfiles",
    async (
      _event: IpcMainInvokeEvent,
      params: {
        cwd?: string;
        agentId?: string;
        agentPolicy?: unknown;
        profileAvailability: unknown;
      },
    ) => {
      await permissionService.saveProfileAvailability(params.profileAvailability);
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
      const created = terminalHost.create(owner, params);
      if (!interactiveOwnerIds.has(owner.id)) {
        interactiveOwnerIds.add(owner.id);
        owner.once("destroyed", () => {
          interactiveOwnerIds.delete(owner.id);
          browserHost.cleanupOwner(owner.id);
          terminalHost.cleanupOwner(owner.id);
        });
      }
      return created;
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

  ipcMain.handle(
    "media:select",
    async (_event: IpcMainInvokeEvent, existingAttachments: readonly MediaAttachment[] = []) => {
      const result = await dialog.showOpenDialog({
        title: "Add files",
        buttonLabel: "Add",
        defaultPath: process.cwd(),
        properties: ["openFile", "multiSelections"],
      });
      return result.canceled ? [] : mediaService.importPaths(result.filePaths, existingAttachments);
    },
  );

  ipcMain.handle(
    "media:import",
    (
      _event: IpcMainInvokeEvent,
      files: Parameters<DesktopMediaService["importBytes"]>[0],
      existingAttachments: readonly MediaAttachment[] = [],
    ) => mediaService.importBytes(files, existingAttachments),
  );

  ipcMain.handle("media:preview", (_event: IpcMainInvokeEvent, attachment: MediaAttachment) =>
    mediaService.preview(attachment),
  );

  ipcMain.handle("media:open", async (_event: IpcMainInvokeEvent, attachment: MediaAttachment) => {
    try {
      const filePath = await mediaService.validatedStoredPath(attachment);
      const error = await shell.openPath(filePath);
      return error ? { opened: false, error } : { opened: true };
    } catch (error) {
      return { opened: false, error: errorMessage(error) };
    }
  });

  ipcMain.handle(
    "media:reveal",
    async (_event: IpcMainInvokeEvent, attachment: MediaAttachment) => {
      const filePath = await mediaService.validatedStoredPath(attachment);
      shell.showItemInFolder(filePath);
      return { revealed: true };
    },
  );

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
    "modelCatalog:resetProviderKey",
    async (_event: IpcMainInvokeEvent, params: { providerId: string; keyId: string }) => {
      const inventory = await loadDesktopExtensionInventory();
      return extensionInventoryWithPlans(
        await modelCatalog.resetProviderKey(inventory, params.providerId, params.keyId),
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
  sideChats.clear();
  browserHost.dispose();
  terminalHost.dispose();
  interactiveOwnerIds.clear();
}

export function resolveDesktopMediaProtocolUrl(url: string): Promise<string> {
  return mediaService.resolveProtocolUrl(url);
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

function inventoryWithHarnessRuntimeEnv(
  inventory: ExtensionInventory,
  harnessId: string | undefined,
  names: readonly string[],
): ExtensionInventory {
  if (!harnessId || names.length === 0) return inventory;
  return {
    ...inventory,
    harnesses: inventory.harnesses.map((harness) =>
      harness.id === harnessId
        ? {
            ...harness,
            passthroughEnv: [...new Set([...(harness.passthroughEnv ?? []), ...names])],
          }
        : harness,
    ),
  };
}

function persistExternalAcpSession(
  localSessionId: string,
  identity: ExternalAcpSessionIdentity,
  externalSessionId: string,
): void {
  const session = loadSession(localSessionId);
  if (!session) throw new Error(`Session ${localSessionId} no longer exists.`);
  const binding = createExternalAcpSessionBinding(
    identity,
    externalSessionId,
    session.externalAcpSession,
  );
  if (sameExternalAcpSessionBinding(session.externalAcpSession, binding)) return;
  session.externalAcpSession = binding;
  saveSession(session);
}

function clearExternalAcpSession(localSessionId: string): void {
  const session = loadSession(localSessionId);
  if (!session?.externalAcpSession) return;
  session.externalAcpSession = undefined;
  saveSession(session);
}

function sameExternalAcpSessionBinding(
  left: ExternalAcpSessionBinding | undefined,
  right: ExternalAcpSessionBinding,
): boolean {
  return Boolean(
    left && left.sessionId === right.sessionId && sameExternalAcpSessionIdentity(left, right),
  );
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
      assertDesktopSwarmModels(node.swarm as SwarmConfig);
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
  return realpath(resolved);
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
      node.swarm = swarmConfigWithWorkingDirectory(node.swarm as SwarmConfig, cwd);
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
      node.swarm = await transformSwarmConfigAgentBackends(node.swarm as SwarmConfig, transform);
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

    const mimeType = detectMediaMimeType(bytes);
    if (!mimeType?.startsWith("image/")) return null;

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

function recordProperty(value: unknown, key: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const property = (value as Record<string, unknown>)[key];
  return property && typeof property === "object" && !Array.isArray(property)
    ? (property as Record<string, unknown>)
    : {};
}

interface PermissionDecisionAuditContext {
  requestId?: string;
  sessionId?: string;
  ownerId?: number;
}

async function recordPermissionDecision(
  input: RecordPermissionDecisionInput,
  context: PermissionDecisionAuditContext = {},
): Promise<void> {
  const audit: PreparedAuditInput = {
    category: "permission",
    action: "tool.decision",
    actor: {
      kind: "user",
      ...(context.ownerId === undefined ? {} : { id: `renderer:${context.ownerId}` }),
    },
    target: {
      kind: "tool",
      ...(safeAuditToken(input.toolName) ? { id: safeAuditToken(input.toolName) } : {}),
    },
    ...(safeAuditToken(context.requestId) ? { requestId: context.requestId } : {}),
    ...(safeAuditToken(context.sessionId) ? { sessionId: context.sessionId } : {}),
    metadata: {
      origin: input.source,
      decision: input.decision,
      ...(input.toolKind ? { toolKind: input.toolKind } : {}),
      ...(input.optionKind ? { optionKind: input.optionKind } : {}),
      policyLayerCount: input.policySourceIds?.length ?? 0,
    },
  };
  activeAuditStore.append({ ...audit, outcome: "attempted" });
  try {
    await permissionService.recordDecision(input);
    activeAuditStore.append({
      ...audit,
      outcome:
        input.decision === "allowed"
          ? "completed"
          : input.decision === "rejected"
            ? "denied"
            : "cancelled",
    });
  } catch (error) {
    activeAuditStore.append({
      ...audit,
      outcome: "failed",
      metadata: { ...audit.metadata, errorType: errorName(error) },
    });
    throw error;
  }
}

function ipcAuditInput(
  channel: string,
  event: unknown,
  args: readonly unknown[],
): PreparedAuditInput {
  const firstRecord = args.find(isRecord);
  const requestId = safeAuditToken(stringProperty(firstRecord, "requestId"));
  const sessionId = safeAuditToken(
    stringProperty(firstRecord, "sessionId") ??
      stringProperty(firstRecord, "parentSessionId") ??
      (channel.startsWith("session:") && typeof args[0] === "string" ? args[0] : undefined),
  );
  const taskId = safeAuditToken(
    stringProperty(firstRecord, "taskId") ?? stringProperty(firstRecord, "workItemId"),
  );
  const sender = recordProperty(event, "sender");
  const senderId = typeof sender.id === "number" ? sender.id : undefined;
  return {
    category: "system",
    action: "ipc.request",
    actor: {
      kind: "user",
      ...(senderId === undefined ? {} : { id: `renderer:${senderId}` }),
    },
    target: { kind: "ipc-channel", id: normalizedIpcAuditChannel(channel) },
    ...(requestId ? { requestId } : {}),
    ...(sessionId ? { sessionId } : {}),
    ...(taskId ? { taskId } : {}),
    metadata: { argumentCount: args.length },
  };
}

function requiredIpcAuditPolicy(channel: string): IpcAuditPolicy {
  const policy = (IPC_AUDIT_POLICIES as Readonly<Record<string, IpcAuditPolicy>>)[channel];
  if (!policy) throw new Error(`Desktop IPC channel ${channel} has no audit policy.`);
  return policy;
}

function normalizedIpcAuditChannel(channel: string): string {
  const normalized = safeAuditToken(channel.replace(/[^A-Za-z0-9_.-]+/g, ".").toLowerCase());
  if (!normalized) throw new Error("Desktop IPC channel cannot be represented safely in audit.");
  return normalized;
}

function recordsResolvedIpcOutcome(
  policy: IpcAuditPolicy,
  outcome: AuditInput["outcome"],
): boolean {
  return policy === "intent_outcome" || (policy === "failure_only" && outcome !== "completed");
}

function recordsDispatchFailure(policy: IpcAuditPolicy, semanticAuditBaseline: number): boolean {
  return policy !== "semantic_only" || semanticAuditCount === semanticAuditBaseline;
}

function recordTerminalAudit(event: Readonly<TerminalAuditEvent>): void {
  const targetId = safeAuditToken(event.terminalId);
  activeAuditStore.append({
    category: "tool",
    action: `terminal.${event.operation}`,
    outcome:
      event.phase === "attempt"
        ? "attempted"
        : event.outcome === "succeeded"
          ? "completed"
          : event.outcome === "rejected"
            ? "denied"
            : "failed",
    actor: { kind: "user", id: `renderer:${event.ownerId}` },
    target: { kind: "terminal", ...(targetId ? { id: targetId } : {}) },
    metadata: {
      ...(event.reason ? { reason: event.reason } : {}),
      ...(event.byteCount === undefined ? {} : { byteCount: event.byteCount }),
      ...(event.cols === undefined ? {} : { cols: event.cols }),
      ...(event.rows === undefined ? {} : { rows: event.rows }),
      ...(event.pid === undefined ? {} : { pid: event.pid }),
      ...(event.exitCode === undefined ? {} : { exitCode: event.exitCode }),
      ...(event.signal === undefined ? {} : { signal: event.signal }),
      ...(event.closeReason === undefined ? {} : { closeReason: event.closeReason }),
    },
  });
  semanticAuditCount += 1;
}

function recordToolChunkAudit(
  auditStore: DesktopAuditStore,
  chunk: MessageChunk,
  params: Pick<DesktopAgentSendParams, "requestId" | "sessionId" | "harnessId">,
): void {
  if ((chunk.kind !== "tool_call" && chunk.kind !== "tool_result") || !chunk.toolName) return;
  const toolId = safeAuditToken(chunk.toolName);
  const invocationId = safeAuditToken(chunk.render?.invocationId);
  const renderStatus = chunk.render?.status;
  auditStore.append({
    category: "tool",
    action: "tool.invoke",
    outcome:
      chunk.kind === "tool_call"
        ? "attempted"
        : renderStatus === "canceled"
          ? "cancelled"
          : renderStatus === "failed"
            ? "failed"
            : "completed",
    actor: { kind: "agent", ...(safeAuditToken(params.harnessId) ? { id: params.harnessId } : {}) },
    target: { kind: "tool", ...(toolId ? { id: toolId } : {}) },
    ...(safeAuditToken(params.requestId) ? { requestId: params.requestId } : {}),
    ...(safeAuditToken(params.sessionId) ? { sessionId: params.sessionId } : {}),
    metadata: { ...(invocationId ? { invocationId } : {}) },
  });
}

function safeAuditToken(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  if (
    !/^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,159}$/.test(trimmed) ||
    SENSITIVE_PERMISSION_LABEL_PATTERN.test(trimmed) ||
    /^(?:sk|rk|pk|ghp|gho|ghu|ghs|github_pat|xox[a-z]?)-/i.test(trimmed)
  ) {
    return undefined;
  }
  return trimmed;
}

function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  return Boolean(value && typeof value === "object" && "then" in value);
}

function ipcResultOutcome(value: unknown): AuditInput["outcome"] {
  if (!isRecord(value)) return "completed";
  if (value.canceled === true) return "cancelled";
  if (value.success === false) return "failed";
  return "completed";
}

function elapsedMilliseconds(startedAt: number): number {
  return Math.max(0, Date.now() - startedAt);
}

function errorName(error: unknown): string {
  return error instanceof Error && error.name ? error.name : "Error";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
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

async function validateMessageAttachments(messages: readonly MessageChunk[]): Promise<void> {
  await Promise.all(
    messages.flatMap((message) =>
      validateMediaAttachments(message.attachments).map((attachment) =>
        mediaService.validatedStoredPath(attachment),
      ),
    ),
  );
}
