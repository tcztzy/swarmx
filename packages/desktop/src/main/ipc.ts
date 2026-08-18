import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { access, chmod, readFile, realpath, stat, writeFile } from "node:fs/promises";
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
  BeginSessionRequestResult,
  CoreRuntime,
  CoreSwarmExecution,
  DiscoveredSession,
  ExtensionAuthorityAuditEvent,
  ExtensionInventory,
  ExternalAcpSessionBinding,
  GlobalMemoryBackend,
  GlobalMemorySnapshot,
  HarnessCatalog,
  HarnessPermissionMode,
  ListGroupedSessionsOptions,
  MediaAttachment,
  MemoryAgentMutation,
  MemoryBackend,
  MemoryReflectionDecision,
  MessageChunk,
  ModelTokenUsage,
  ProjectBootstrapReceipt,
  ReferenceLibraryBackend,
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
  archiveSession,
  beginSessionRequest,
  buildGlobalMemoryUseReceipt,
  countPersonalMemoryAgentTargets,
  createCoreRuntime,
  createMemoryAgentTool,
  createReferenceLibraryAgentTool,
  createSession,
  createSessionContextEngine,
  detectMediaMimeType,
  editSessionUserMessage,
  estimateModelTokenUsage,
  executeAgentComposition,
  forkSession,
  globalMemoryReceiptMessage,
  HarnessPermissionPolicySchema,
  importN8nWorkflow,
  listSessionSummaries,
  loadExtensionInventory,
  loadSession,
  mergeModelTokenUsage,
  modelReplayableMessages,
  projectBootstrapReceiptMessage,
  resolveAgentCompositionPlan,
  saveSession,
  setSessionPinned,
  settleSessionRequest,
  stableJson,
  staticHarnessCatalog,
  updateSessionTitle,
  validateMediaAttachments,
} from "@swarmx/core";
import { RequestCancelledError } from "@swarmx/core/request-scope";
import {
  containerHostBridgeUrl,
  HarnessDoctor,
  HarnessEnvironmentService,
  type HarnessEnvironmentSetupRequest,
  type HarnessEnvironmentStatus,
} from "@swarmx/runtime";
import { dialog, ipcMain as electronIpcMain, type IpcMainInvokeEvent, shell } from "electron";
import type { DesktopUpdateState } from "../shared/ipc-contracts/app-update.js";
import type { DesktopIpcAuditPolicy } from "../shared/ipc-contracts/base.js";
import { DesktopInvokeContractRegistry } from "../shared/ipc-contracts/index.js";
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
import { registerAppUpdateIpc, validatedAppUpdateState } from "./app-update-ipc.js";
import { createDesktopBrowserHost, registerBrowserIpc } from "./browser-ipc.js";
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
import { registerGlobalMemoryIpc } from "./global-memory-ipc.js";
import { GlobalMemoryService, type GlobalMemoryServiceLike } from "./global-memory-service.js";
import {
  createDesktopIpcRegistrar,
  createSemanticAuditReceipt,
  type DesktopAuthorizedIpcHandler,
  DesktopIpcBoundaryError,
  type SemanticAuditReceipt,
} from "./ipc-router.js";
import { type LspCompletionRequest, LspHost, type LspStopRequest } from "./lsp-host.js";
import { DesktopMediaService } from "./media.js";
import {
  type ManualModelInput,
  ModelCatalogService,
  type ProviderRuntimeCredentials,
  type UserProviderInput,
} from "./model-catalog.js";
import { PermissionAutoReviewer, type PermissionReviewResult } from "./permission-review.js";
import { PermissionService, type RecordPermissionDecisionInput } from "./permission-service.js";
import { registerProjectIpc } from "./project-ipc.js";
import { ProjectService } from "./project-service.js";
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
import { registerTaskRuntimeIpc } from "./task-runtime-ipc.js";
import { DesktopTaskSupervisor, type DesktopTaskSupervisorLike } from "./task-supervisor.js";
import { type TerminalAuditEvent, TerminalHost } from "./terminal-host.js";
import { registerTerminalIpc } from "./terminal-ipc.js";
import { createDisabledDesktopUpdateService, type DesktopUpdateServiceLike } from "./updater.js";
import type { RendererIpcEvent } from "./window-security.js";
import { registerWorkspaceInspectionIpc } from "./workspace-inspection-ipc.js";
import type { WorkspacePermissionReviewRequest } from "./workspace-tool-permissions.js";
import {
  type ClaudeSessionActivation,
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
type IpcAuditPolicy = DesktopIpcAuditPolicy;
export const LEGACY_IPC_AUDIT_POLICIES = {
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
let desktopHarnessCatalog: HarnessCatalog = staticHarnessCatalog;
const harnessEnvironment = new HarnessEnvironmentService({
  harnessCatalog: {
    listHarnesses: () => desktopHarnessCatalog.listHarnesses(),
    getHarness: (id) => desktopHarnessCatalog.getHarness(id),
    resolveRuntimeModel: (id, options) => desktopHarnessCatalog.resolveRuntimeModel(id, options),
    resolveModelRuntimeEnv: (id, options) =>
      desktopHarnessCatalog.resolveModelRuntimeEnv(id, options),
  },
});
const harnessDoctor = new HarnessDoctor(harnessEnvironment);
const agentRequests = new DesktopRequestRegistry();
const sideChats = new SideChatService();
const agentInteractions = new AgentInteractionBroker();
const claudeSessionRuntimes = new ClaudeSessionRuntimeRegistry();
const browserHost = createDesktopBrowserHost();
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
const desktopTaskSupervisor = new DesktopTaskSupervisor();
const mediaService = new DesktopMediaService(
  process.env.NODE_ENV === "test"
    ? path.join(tmpdir(), `swarmx-media-test-${process.pid}`)
    : undefined,
);
const extensionManager = new DesktopExtensionManager(
  desktopSettingsStore,
  undefined,
  undefined,
  recordExtensionAuthorityAudit,
);
const providerUsage = new ProviderUsageService({ authStore: providerAuthStore });
const desktopActivity = new ActivityStore(
  process.env.NODE_ENV === "test"
    ? { filePath: path.join(tmpdir(), `swarmx-activity-test-${process.pid}.jsonl`) }
    : {},
);
const desktopCoreRuntimePromise = createCoreRuntime();
void desktopCoreRuntimePromise
  .then((runtime) => {
    desktopHarnessCatalog = runtime.harnessCatalog;
  })
  .catch(() => {});

export async function disposeDesktopCoreRuntime(): Promise<void> {
  await (await desktopCoreRuntimePromise).dispose();
}

export interface RegisterIpcHandlersOptions {
  coreRuntime?: CoreRuntime | Promise<CoreRuntime>;
  updateService?: DesktopUpdateServiceLike;
  broadcastUpdateState?: (state: DesktopUpdateState) => void;
  activityStore?: ActivityStore;
  auditStore?: DesktopAuditStore;
  authorizeIpcSender?: (event: RendererIpcEvent) => boolean;
  globalMemoryService?: GlobalMemoryServiceLike;
  taskSupervisor?: DesktopTaskSupervisorLike;
  memoryBackend?: MemoryBackend & Partial<GlobalMemoryBackend>;
  referenceLibraryBackend?: ReferenceLibraryBackend;
}

interface DesktopAgentSendParams {
  requestId: string;
  sessionId?: string;
  sideChatId?: string;
  sideChatVisible?: boolean;
  sideEditMessageIndex?: number;
  editMessageIndex?: number;
  editExpectedMessages?: MessageChunk[];
  harnessId: string;
  userText: string;
  attachments?: MediaAttachment[];
  agentConfig?: AgentConfig;
  agentComposition?: AgentComposition;
  swarmConfig?: SwarmConfig;
  cwd?: string;
}

class SessionRequestError extends Error {
  constructor(
    readonly code: "REQUEST_OUTCOME_UNKNOWN" | "REQUEST_ID_CONFLICT" | "REQUEST_ALREADY_ACTIVE",
  ) {
    super(code);
    this.name = "SessionRequestError";
  }
}

function requestDigest(params: DesktopAgentSendParams): string {
  const normalized = {
    sessionId: params.sessionId,
    sideChatId: params.sideChatId,
    sideChatVisible: params.sideChatVisible,
    sideEditMessageIndex: params.sideEditMessageIndex,
    editMessageIndex: params.editMessageIndex,
    editExpectedMessages: params.editExpectedMessages,
    harnessId: params.harnessId,
    userText: params.userText.trim(),
    attachments: params.attachments,
    agentConfig: params.agentConfig,
    agentComposition: params.agentComposition,
    swarmConfig: params.swarmConfig,
    cwd: params.cwd ? path.resolve(params.cwd) : undefined,
  };
  return `sha256:${createHash("sha256").update(stableJson(normalized), "utf8").digest("hex")}`;
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
  const runtime = await desktopCoreRuntimePromise;
  const [inventory, nativeAgents, settings] = await Promise.all([
    loadExtensionInventory({ harnessCatalog: runtime.harnessCatalog }),
    customAgents.discoverNative({ workspaceRoot: desktopWorkspaceRoot }),
    desktopSettingsStore.read(),
  ]);
  const declaredIds = new Set(inventory.agents.map((agent) => agent.id));
  const discovered = nativeAgents.agents.filter((agent) => !declaredIds.has(agent.id));
  return {
    ...inventory,
    installedExtensions: settings.extensions.installed,
    agents: [...inventory.agents, ...discovered],
    warnings: [...inventory.warnings, ...nativeAgents.warnings],
  };
}

async function desktopDoctorReadiness(harnessId?: string) {
  const settings = await desktopSettingsStore.read();
  const selectedHarnessId = harnessId ?? settings.ui.composer.lastHarnessId;
  const provider = await providerReadiness(selectedHarnessId, settings.providers);
  let project: "ready" | "missing" | "not_writable";
  try {
    const workspace = await stat(desktopWorkspaceRoot);
    if (!workspace.isDirectory()) {
      project = "missing";
    } else {
      await access(desktopWorkspaceRoot, constants.W_OK);
      project = "ready";
    }
  } catch (error) {
    project = isErrorCode(error, "ENOENT") ? "missing" : "not_writable";
  }
  return { provider, project, network: "not_required" as const };
}

async function providerReadiness(
  harnessId: string | undefined,
  providers: Awaited<ReturnType<DesktopSettingsStore["read"]>>["providers"],
): Promise<"ready" | "missing" | "invalid_reference" | "not_required"> {
  if (harnessId && harnessId !== "swarmx" && harnessId !== "swarmx-direct") {
    return "not_required";
  }
  if (providers.length === 0) return "missing";
  let promptReference = false;
  for (const provider of providers) {
    const reference = provider.secretRef;
    if (!reference) return "ready";
    if (reference.source === "prompt") {
      promptReference = true;
      continue;
    }
    if (reference.source === "env" && process.env[reference.key]) return "ready";
    if (reference.source === "local_auth_file") {
      try {
        if (await providerAuthStore.has(reference.key)) return "ready";
      } catch {
        return "invalid_reference";
      }
    }
  }
  return promptReference ? "missing" : "invalid_reference";
}

function isErrorCode(error: unknown, code: string): boolean {
  return error instanceof Error && "code" in error && error.code === code;
}

export function registerIpcHandlers(options: RegisterIpcHandlersOptions = {}): void {
  const coreRuntimePromise = Promise.resolve(options.coreRuntime ?? desktopCoreRuntimePromise);
  void coreRuntimePromise
    .then((runtime) => {
      desktopHarnessCatalog = runtime.harnessCatalog;
    })
    .catch(() => {});
  const runtimeInventoryWithPlans = async (
    inventory: ExtensionInventory,
    env: NodeJS.ProcessEnv = process.env,
  ) => extensionInventoryWithPlans(inventory, env, (await coreRuntimePromise).harnessCatalog);
  const executeComposition = async (
    composition: Parameters<typeof executeAgentComposition>[0],
    messages: Parameters<typeof executeAgentComposition>[1],
    runtimeOptions: Omit<Parameters<typeof executeAgentComposition>[2], "runtime">,
  ): ReturnType<typeof executeAgentComposition> => {
    const runtime = await coreRuntimePromise;
    return executeAgentComposition(composition, messages, {
      ...runtimeOptions,
      runtime,
      harnessCatalog: runtimeOptions.harnessCatalog ?? runtime.harnessCatalog,
      resolveProviderRuntimeEnv:
        runtimeOptions.resolveProviderRuntimeEnv ?? runtime.resolveProviderRuntimeEnv,
    });
  };
  const updateService = options.updateService ?? createDisabledDesktopUpdateService();
  const activityStore = options.activityStore ?? desktopActivity;
  const auditStore = options.auditStore ?? desktopAudit;
  const taskSupervisor = options.taskSupervisor ?? desktopTaskSupervisor;
  const memoryBackend = options.memoryBackend ?? unavailableMemoryBackend();
  const globalMemoryBackend = supportsGlobalMemory(memoryBackend)
    ? memoryBackend
    : unavailableGlobalMemoryBackend();
  const globalMemoryService =
    options.globalMemoryService ??
    new GlobalMemoryService(globalMemoryBackend, desktopSettingsStore);
  const agentMemoryBackend = memoryBackendWithGlobalMemory(memoryBackend, globalMemoryService);
  const referenceLibraryBackend = options.referenceLibraryBackend;
  activeAuditStore = auditStore;
  const authorizeIpcSender = options.authorizeIpcSender ?? (() => false);
  const assertAuthorized = (event: RendererIpcEvent): void => {
    if (!authorizeIpcSender(event)) throw new Error("Untrusted desktop IPC sender.");
  };
  const registerAudited = (channel: string, listener: DesktopAuthorizedIpcHandler): void => {
    if (channel.startsWith("audit:")) {
      electronIpcMain.handle(channel, (event, ...args) => {
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
        return listener(event, createSemanticAuditReceipt(), ...args);
      });
      return;
    }
    const auditPolicy = requiredIpcAuditPolicy(channel);
    electronIpcMain.handle(channel, (event, ...args) => {
      const startedAt = Date.now();
      const receipt = createSemanticAuditReceipt();
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
        const result = listener(event, receipt, ...args);
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
              if (recordsDispatchFailure(auditPolicy, receipt, error)) {
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
        if (recordsDispatchFailure(auditPolicy, receipt, error)) {
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
  const handle: typeof electronIpcMain.handle = (channel, listener) =>
    registerAudited(channel, (event, _receipt, ...args) => listener(event, ...args));
  const ipcMain = { handle };
  const contractRegistrar = createDesktopIpcRegistrar({
    registerAuthorized: registerAudited,
    auditPolicy: requiredIpcAuditPolicy,
  });
  const projectService = new ProjectService({
    workspaceRoot: desktopWorkspaceRoot,
    isSessionRunning: (sessionId) =>
      agentRequests.isSessionActive(sessionId) ||
      claudeSessionRuntimes.isRunning(sessionId) ||
      sideChats.isParentRunning(sessionId),
    clearSideChats: (parentSessionId) => sideChats.clearParent(parentSessionId),
  });
  const bootstrapAuditPolicy = requiredIpcAuditPolicy("bootstrap:get");
  electronIpcMain.on("bootstrap:get", (event) => {
    const startedAt = Date.now();
    const audit = ipcAuditInput("bootstrap:get", event, []);
    if (bootstrapAuditPolicy === "intent_outcome") {
      auditStore.append({ ...audit, outcome: "attempted" });
    }
    try {
      assertAuthorized(event);
      event.returnValue = projectService.list();
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
  if (options.broadcastUpdateState) {
    updateService.subscribe((state) =>
      options.broadcastUpdateState?.(validatedAppUpdateState(state)),
    );
  }
  registerAppUpdateIpc(contractRegistrar, updateService);
  registerGlobalMemoryIpc(contractRegistrar, globalMemoryService);
  registerProjectIpc(contractRegistrar, projectService);
  registerTaskRuntimeIpc(contractRegistrar, taskSupervisor);
  registerWorkspaceInspectionIpc(contractRegistrar, {
    workspaceRoot: desktopWorkspaceRoot,
    normalizeWorkingDirectory,
    toolsFor: workspaceToolsFor,
  });
  registerBrowserIpc(contractRegistrar, browserHost, ensureInteractiveOwner);
  registerTerminalIpc(contractRegistrar, terminalHost, ensureInteractiveOwner);
  const handleAgentSend = async (event: IpcMainInvokeEvent, params: DesktopAgentSendParams) => {
    const startedAt = Date.now();
    const observedMessages: MessageChunk[] = [];
    const tokenUsages: ModelTokenUsage[] = [];
    const usedSkillIds = new Set<string>();
    let memoryReceipt: MessageChunk | undefined;
    const projectBootstrapReceiptKeys = new Set<string>();
    const projectBootstrapReceipts: MessageChunk[] = [];
    let memoryReflection: MemoryReflectionDecision | undefined;
    let foregroundRuntime: ClaudeSessionRuntime | undefined;
    let activeChunkPublisher: AgentChunkPublisher | undefined;
    let foregroundChunksActive = false;
    let activeSideChat: TransientSessionData | undefined;
    let ephemeralCodexHome: Awaited<ReturnType<typeof createEphemeralCodexHome>> | undefined;
    const durableRequestDigest = requestDigest(params);
    let durableRequestStarted = false;
    let durableRequestSettled = false;
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
    const persistOutcome = (
      messages: MessageChunk[],
      outcome: "completed" | "canceled" | "failed",
    ) => {
      let sessionPersisted = false;
      if (params.sessionId && !params.sideChatId && durableRequestStarted) {
        if (!durableRequestSettled) {
          const settled = settleSessionRequest({
            id: params.sessionId,
            requestId: params.requestId,
            requestDigest: durableRequestDigest,
            messages,
            outcome,
          });
          if (!settled) {
            throw new Error(
              `Session "${params.sessionId}" disappeared while settling the request.`,
            );
          }
          durableRequestSettled = true;
        }
        sessionPersisted = true;
      }
      return {
        sessionPersisted,
        sideChat:
          params.sideChatId && params.sessionId && activeSideChat
            ? sideChats.finishRun(params.sessionId, params.sideChatId, params.requestId, messages, {
                unread: params.sideChatVisible === false,
              })
            : undefined,
      };
    };
    const closeForegroundTurn = (): void => {
      foregroundChunksActive = false;
      activeChunkPublisher?.close();
      activeChunkPublisher = undefined;
      agentInteractions.cancelRequest(event.sender, params.requestId);
    };
    try {
      if (params.sessionId && loadSession(params.sessionId)?.archivedAt) {
        throw new Error(`Session "${params.sessionId}" is archived.`);
      }
      if (params.sessionId && !params.sideChatId) {
        if (agentRequests.isActive(params.requestId)) {
          throw new SessionRequestError("REQUEST_ALREADY_ACTIVE");
        }
        const begun: BeginSessionRequestResult | null = beginSessionRequest({
          id: params.sessionId,
          requestId: params.requestId,
          requestDigest: durableRequestDigest,
          userMessage: {
            role: "user",
            kind: "message",
            content: params.userText.trim(),
            ...(params.attachments?.length ? { attachments: params.attachments } : {}),
          },
          ...(params.editMessageIndex === undefined
            ? {}
            : {
                editMessageIndex: params.editMessageIndex,
                expectedMessages: params.editExpectedMessages ?? [],
              }),
        });
        if (!begun) throw new Error(`Session "${params.sessionId}" no longer exists.`);
        if (begun.status === "conflict") {
          throw new SessionRequestError("REQUEST_ID_CONFLICT");
        }
        if (begun.status === "unknown") {
          throw new SessionRequestError("REQUEST_OUTCOME_UNKNOWN");
        }
        if (begun.status === "settled") {
          const replayedError =
            begun.outcome === "failed"
              ? (begun.messages
                  .slice()
                  .reverse()
                  .find((message: MessageChunk) => message.role === "system")?.content ??
                "The request previously failed.")
              : undefined;
          return {
            success: begun.outcome === "completed",
            ...(begun.outcome === "canceled" ? { canceled: true } : {}),
            ...(replayedError ? { error: replayedError } : {}),
            requestId: params.requestId,
            messages: begun.messages,
            sessionPersisted: true,
            replayed: true,
          };
        }
        durableRequestStarted = true;
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
      if (params.sessionId && !params.sideChatId) {
        memoryReflection = await globalMemoryService.reflectionDecision({
          sessionId: params.sessionId,
          userTurnCount: foregroundUserTurnCount(params.sessionId, params.userText),
          userText: params.userText,
        });
      }
      const result = await agentRequests.runForSession(desktopRequest, async () => {
        const publishChunk = agentChunkPublisher(event.sender, params.requestId, {
          ...(params.sessionId ? { sessionId: params.sessionId } : {}),
          adapter: params.harnessId,
          onLateChunk: (observation) => recordLateChunkAudit(auditStore, observation),
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
        foregroundChunksActive = true;
        const onChunk = (chunk: MessageChunk) => {
          if (!foregroundChunksActive) return;
          recordToolChunkAudit(auditStore, chunk, params);
          if (chunk.kind !== "tool_progress") observedMessages.push(chunk);
          publishChunk(chunk);
        };
        let activePermissionReviewer: PermissionAutoReviewer | undefined;
        let activePermissionReviewerModel: string | undefined;
        let effectivePermissionMode: HarnessPermissionMode = "default";
        const permissionUserMessages = () =>
          permissionReviewUserMessages(params.sessionId, params.userText);
        const reviewPermission = async (input: {
          source: "direct" | "acp";
          toolName: string;
          toolKind?: string;
          toolInput?: unknown;
          options: Array<{
            optionId: string;
            kind: "allow_once" | "allow_always" | "reject_once" | "reject_always";
          }>;
        }): Promise<PermissionReviewResult> => {
          if (effectivePermissionMode !== "auto" || !activePermissionReviewer) {
            return { decision: "defer" };
          }
          return activePermissionReviewer.review({
            ...input,
            userMessages: permissionUserMessages(),
          });
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
            const automatic = requiresExplicitPermissionInteraction(request)
              ? ({ decision: "defer" } as const)
              : await reviewPermission({
                  source: "acp",
                  toolName: title,
                  ...(toolKind ? { toolKind } : {}),
                  toolInput: request.toolCall.rawInput ?? {
                    title,
                    ...(toolKind ? { kind: toolKind } : {}),
                    ...(request.toolCall.name ? { name: request.toolCall.name } : {}),
                    ...(request.toolCall.locations
                      ? { locations: request.toolCall.locations }
                      : {}),
                  },
                  options: request.options.map((option) => ({
                    optionId: option.optionId,
                    kind: option.kind,
                  })),
                });
            if (automatic.decision === "allow") {
              await recordPermissionDecision(
                {
                  source: "acp",
                  toolName: title,
                  ...(toolKind ? { toolKind } : {}),
                  decision: "allowed",
                  decidedBy: "llm",
                  risk: automatic.risk,
                  ...(activePermissionReviewerModel
                    ? { reviewerModel: activePermissionReviewerModel }
                    : {}),
                  optionKind: "allow_once",
                },
                {
                  requestId: params.requestId,
                  sessionId: params.sessionId,
                },
              );
              return { outcome: { outcome: "selected", optionId: automatic.optionId } };
            }
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
        const memoryTool = createMemoryAgentTool(agentMemoryBackend, {
          confirm: async (mutation) => {
            const response = await agentInteractions.request(event.sender, params.requestId, {
              kind: "tool_approval",
              title: memoryMutationApprovalTitle(mutation.operation),
              toolKind: "memory",
              source: "direct",
              summary: memoryMutationSummary(mutation),
              options: [
                { optionId: "allow_once", name: "Confirm once", kind: "allow_once" },
                { optionId: "reject_once", name: "Keep Memory", kind: "reject_once" },
              ],
            });
            return response.kind === "tool_approval" && response.optionId === "allow_once";
          },
          audit: (memoryEvent) => {
            auditStore.append({
              category: "system",
              action: "memory.agent_mutation",
              actor: { kind: "agent" },
              target: {
                kind: "memory",
                ...(memoryEvent.pageId ? { id: memoryEvent.pageId } : {}),
              },
              requestId: params.requestId,
              ...(params.sessionId ? { sessionId: params.sessionId } : {}),
              outcome: memoryEvent.outcome,
              metadata: {
                operation: memoryEvent.operation,
                ...(memoryEvent.expectedRevision === undefined
                  ? {}
                  : { expectedRevision: memoryEvent.expectedRevision }),
                ...(memoryEvent.characterCount === undefined
                  ? {}
                  : { characterCount: memoryEvent.characterCount }),
                ...(memoryEvent.globalTarget === undefined
                  ? {}
                  : { globalTarget: memoryEvent.globalTarget }),
                ...(memoryEvent.observationCount === undefined
                  ? {}
                  : { observationCount: memoryEvent.observationCount }),
              },
            });
          },
          researchProvenance: {
            sessionId: params.sessionId ?? params.requestId,
            capturedAt: new Date().toISOString(),
          },
        });
        const memoryTools = [memoryTool];
        const privateKnowledgeTools = referenceLibraryBackend
          ? [createReferenceLibraryAgentTool(referenceLibraryBackend)]
          : [];
        let swarm: CoreSwarmExecution;
        const cwd = await normalizeWorkingDirectory(params.cwd);
        const createRunContextEngine = (contextId: string, history: readonly MessageChunk[]) =>
          createSessionContextEngine({
            sessionId: contextId,
            history: modelReplayableMessages(history),
            onCompiled: (manifest) => {
              auditStore.append({
                category: "provider",
                action: "context.compiled",
                actor: { kind: "system" },
                target: { kind: "context-manifest" },
                requestId: params.requestId,
                ...(params.sessionId ? { sessionId: params.sessionId } : {}),
                metadata: {
                  snapshotId: manifest.snapshotId,
                  configHash: manifest.configHash,
                  ...(manifest.sourceConfigHash
                    ? { sourceConfigHash: manifest.sourceConfigHash }
                    : {}),
                  modelVersion: manifest.modelVersion,
                  contextHash: manifest.contextHash,
                  compilePhase: manifest.compilePhase,
                  profile: manifest.profile,
                  profileFidelity: manifest.profileFidelity,
                  configuredProjectionPolicy: manifest.configuredProjectionPolicy,
                  configuredEvidencePolicy: manifest.configuredEvidencePolicy,
                  projectionMode: manifest.projectionMode,
                  requestedEvidenceMode: manifest.requestedMode,
                  effectiveEvidenceMode: manifest.effectiveMode,
                  contextWindowTokens: manifest.contextWindowTokens,
                  contextWindowSource: manifest.contextWindowSource,
                  pressureThresholdTokens: manifest.pressureThresholdTokens,
                  fixedInputTokens: manifest.fixedInputTokens,
                  totalInputTokens: manifest.totalInputTokens,
                  tokenEstimator: manifest.tokenEstimator,
                  summaryMode: manifest.summaryMode,
                  summaryCalls: manifest.summaryCalls,
                  summaryInputTokens: manifest.summaryInputTokens,
                  summaryOutputTokens: manifest.summaryOutputTokens,
                  summaryModelVersions: manifest.summaryModelVersions,
                  ...(manifest.checkpointId ? { checkpointId: manifest.checkpointId } : {}),
                  includedItemCount: manifest.includedItemIds.length,
                  includedEventCount: manifest.includedEventIds.length,
                  omittedItemCount: manifest.omittedItems.length,
                  inputTokens: manifest.inputTokens,
                  reservedOutputTokens: manifest.reservedOutputTokens,
                },
              });
            },
          });

        if (params.swarmConfig) {
          if (params.sessionId && !params.sideChatId) {
            clearExternalAcpSession(params.sessionId);
          }
          assertDesktopSwarmModels(params.swarmConfig);
          const config = cwd
            ? swarmConfigWithWorkingDirectory(params.swarmConfig, cwd)
            : params.swarmConfig;
          const agentCount = countPersonalMemoryAgentTargets(config);
          const globalMemoryRun = await globalMemorySnapshotForRun(globalMemoryService);
          const globalMemorySnapshot = globalMemoryRun.snapshot;
          const workflowHistory = params.sessionId
            ? (loadSession(params.sessionId)?.messages ?? [])
            : [];
          swarm = (await coreRuntimePromise).prepareSwarm(
            await protectSwarmConfigBackends(config, cwd),
            {
              agent: {
                acpPermissionHandler,
                memoryTools,
                localTools: privateKnowledgeTools,
                contextEngine: createRunContextEngine(
                  params.sessionId ?? params.requestId,
                  workflowHistory,
                ),
                ...(globalMemorySnapshot ? { globalMemory: globalMemorySnapshot } : {}),
                ...(memoryReflection ? { memoryReflection } : {}),
              },
            },
          );
          memoryReceipt = globalMemoryReceiptMessage(
            buildGlobalMemoryUseReceipt({
              snapshot: globalMemorySnapshot,
              executionPath: "workflow",
              agentCount,
              unavailable: globalMemoryRun.unavailable,
            }),
          );
          onChunk(memoryReceipt);
        } else if (params.agentComposition) {
          const inventory = await modelCatalog.list(await loadDesktopExtensionInventory());
          const plan = resolveAgentCompositionPlan(
            params.agentComposition,
            inventory,
            (await coreRuntimePromise).harnessCatalog,
          );
          for (const skillId of new Set(plan.skills.map((skill) => skill.id))) {
            usedSkillIds.add(skillId);
          }
          assertCompositionSupplyReady(inventory, plan, process.env);
          const providerRuntime = plan.modelSupplyId
            ? await modelCatalog.runtimeCredentialsForSupply(inventory, plan.modelSupplyId)
            : undefined;
          const protectedInventory = await protectCompositionHarness(
            inventory,
            plan.harnessId,
            cwd,
          );
          let executionInventory = protectedInventory;
          const runtimeHarnessId = compositionRuntimeHarnessId(inventory, plan);
          const protectedHarness = protectedInventory.harnesses.find(
            (harness) => harness.id === plan.harnessId,
          );
          const compositionBackend = protectedHarness?.backend;
          const compositionUsesAcp = compositionBackend?.type === "custom";
          const globalMemoryRun = await globalMemorySnapshotForRun(globalMemoryService);
          const globalMemorySnapshot = globalMemoryRun.snapshot;
          memoryReceipt = globalMemoryReceiptMessage(
            buildGlobalMemoryUseReceipt({
              snapshot: globalMemorySnapshot,
              executionPath: compositionUsesAcp ? "external_acp" : "direct_agent",
              agentCount: 1,
              unavailable: globalMemoryRun.unavailable,
            }),
          );
          onChunk(memoryReceipt);
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
          const agentMemoryTools = !compositionUsesAcp && !params.sideChatId ? memoryTools : [];
          const agentPrivateKnowledgeTools =
            !compositionUsesAcp && !params.sideChatId ? privateKnowledgeTools : [];
          const directSession =
            params.sessionId && !params.sideChatId ? loadSession(params.sessionId) : undefined;
          const directProjectRoot =
            !params.sideChatId && directSession?.projectId && directSession.cwd
              ? await normalizeWorkingDirectory(directSession.cwd)
              : undefined;
          const directProject =
            directSession?.projectId && directProjectRoot
              ? { id: directSession.projectId, root: directProjectRoot }
              : undefined;
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
            params.sessionId && !params.sideChatId ? directSession : undefined;
          if (params.sessionId && !params.sideChatId && !permissionSession) {
            throw new Error(`Session ${params.sessionId} no longer exists.`);
          }
          const permissionPolicy = await permissionService.resolve({
            cwd,
            agentId: plan.agentProfileId ?? plan.agentId,
            agentPolicy: agentPermissionPolicy,
            agentModeDeclared: Boolean(plan.permissions?.mode),
            ...(params.sideChatId
              ? { sessionPermissionMode: "plan" as const }
              : permissionSession
                ? { sessionPermissionMode: permissionSession.permissionMode }
                : {}),
          });
          effectivePermissionMode = permissionPolicy.policy.mode;
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
            permissionPolicy,
            ...(projectTools && lspHost.supportsClaudeOperations(inventory)
              ? {
                  lsp: (request) => lspHost.operate(inventory, projectTools.root, request),
                }
              : {}),
          };
          const reviewerModel = plan.modelId ?? plan.runtimeModel;
          const reviewerComposition: AgentComposition = {
            id: `${params.requestId}:permission-review`,
            harnessId: "swarmx",
            ...(plan.modelId ? { modelId: plan.modelId } : {}),
            ...(plan.modelSupplyId ? { modelSupplyId: plan.modelSupplyId } : {}),
            effort: "none",
            skills: [],
            mcpServers: [],
            pluginIds: [],
            plugins: [],
            host: "local",
          };
          const runWithPermissionReviewer = async <T>(
            providerSecrets: Record<string, string>,
            observation: ProviderKeyAttemptObservation,
            run: () => Promise<T>,
          ): Promise<T> => {
            const previousReviewer = activePermissionReviewer;
            const previousModel = activePermissionReviewerModel;
            activePermissionReviewerModel = reviewerModel;
            activePermissionReviewer = new PermissionAutoReviewer({
              generate: (messages) =>
                executeComposition(reviewerComposition, messages, {
                  inventory: executionInventory,
                  providerSecrets,
                  cwd,
                  onUsage: (usage) => {
                    tokenUsages.push(usage);
                    observation.recordUsage(usage);
                  },
                }),
            });
            try {
              return await run();
            } finally {
              activePermissionReviewer = previousReviewer;
              activePermissionReviewerModel = previousModel;
            }
          };
          const reviewWorkspacePermission = async (
            request: WorkspacePermissionReviewRequest,
          ): Promise<boolean> => {
            const automatic = await reviewPermission({
              source: request.source,
              toolName: request.toolName,
              toolKind: request.toolKind,
              toolInput: request.toolInput,
              options: request.options.map((option) => ({
                optionId: option.optionId,
                kind: option.kind,
              })),
            });
            if (automatic.decision !== "allow") return false;
            await recordPermissionDecision(
              {
                source: "direct",
                toolName: request.toolName,
                toolKind: request.toolKind,
                decision: "allowed",
                decidedBy: "llm",
                risk: automatic.risk,
                ...(activePermissionReviewerModel
                  ? { reviewerModel: activePermissionReviewerModel }
                  : {}),
                optionKind: "allow_once",
                policySourceIds: request.policySourceIds,
              },
              {
                requestId: params.requestId,
                sessionId: params.sessionId,
              },
            );
            return true;
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
            const runtimeProjectId = directSession?.projectId;
            sessionRuntime.configure({
              activate: async (activation) => {
                recordSessionActivationAudit(auditStore, activation, sessionId, "started");
                const boundSession = loadSession(sessionId);
                if (!boundSession) throw new Error(`Session ${sessionId} no longer exists.`);
                const backgroundProject = await bindPersistedSessionProject(
                  boundSession,
                  runtimeProjectId,
                  sessionRuntime.root,
                );
                const backgroundRoot = backgroundProject?.root ?? sessionRuntime.root;
                const activationMessage: MessageChunk = {
                  role: "system",
                  content: activation.prompt,
                  kind: "message",
                };
                if (
                  !appendMessages(sessionId, [activationMessage], {
                    activationId: activation.activationId,
                  })
                ) {
                  throw new Error(`Session ${sessionId} no longer exists.`);
                }
                publishSessionMessages(event.sender, sessionId);
                const persisted = loadSession(sessionId);
                if (!persisted) throw new Error(`Session ${sessionId} no longer exists.`);
                const backgroundTools = new WorkspaceTools(backgroundRoot);
                const backgroundReceiptKeys = new Set<string>();
                const backgroundToolOptions: WorkspaceAgentToolOptions = {
                  ...baseWorkspaceToolOptions,
                  reviewPermission: reviewWorkspacePermission,
                  permissionPolicy: await permissionService.resolve({
                    cwd: backgroundRoot,
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
                    runWithPermissionReviewer(providerSecrets, observation, () =>
                      executeComposition(
                        params.agentComposition,
                        [
                          {
                            role: "system",
                            content: projectAgentContextMessage(
                              backgroundRoot,
                              backgroundToolOptions,
                            ),
                          },
                          ...sessionChatMessages(persisted),
                        ],
                        {
                          inventory: protectedInventory,
                          providerSecrets,
                          cwd: backgroundRoot,
                          project: backgroundProject,
                          ...(compositionUsesAcp
                            ? {}
                            : {
                                contextEngine: createRunContextEngine(
                                  sessionId,
                                  persisted.messages,
                                ),
                              }),
                          acpPermissionHandler,
                          memoryTools: agentMemoryTools,
                          localTools: [
                            ...agentPrivateKnowledgeTools,
                            ...workspaceAgentTools(
                              backgroundTools,
                              sessionRuntime.shell,
                              backgroundToolOptions,
                            ),
                          ],
                          onChunk: () => observation.markOutput(),
                          onUsage: (usage) => observation.recordUsage(usage),
                          onProjectBootstrap: (receipt) => {
                            const message = uniqueProjectBootstrapReceiptMessage(
                              backgroundReceiptKeys,
                              `${sessionId}:background:${activation.source}:${activation.jobId ?? activation.taskId ?? "unknown"}`,
                              receipt,
                            );
                            if (!message) return;
                            if (
                              !appendMessages(sessionId, [message], {
                                activationId: activation.activationId,
                              })
                            ) {
                              throw new Error(`Session ${sessionId} no longer exists.`);
                            }
                            recordSessionActivationAudit(
                              auditStore,
                              activation,
                              sessionId,
                              "bootstrap",
                            );
                            publishSessionMessages(event.sender, sessionId);
                          },
                          ...(globalMemorySnapshot ? { globalMemory: globalMemorySnapshot } : {}),
                          ...(memoryReflection ? { memoryReflection } : {}),
                        },
                      ),
                    ),
                );
                assertFinalAssistantMessage(messages);
                if (
                  !appendMessages(sessionId, messages, { activationId: activation.activationId })
                ) {
                  throw new Error(`Session ${sessionId} no longer exists.`);
                }
                recordSessionActivationAudit(auditStore, activation, sessionId, "result");
                publishSessionMessages(event.sender, sessionId);
              },
              onActivationError: (activation, error) => {
                recordSessionActivationAudit(auditStore, activation, sessionId, "failure", error);
                const message: MessageChunk = {
                  role: "system",
                  content: `Background activation failed: ${errorMessage(error)}`,
                  kind: "message",
                };
                if (
                  appendMessages(sessionId, [message], {
                    activationId: activation.activationId,
                  })
                ) {
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
                    const childBinding = childProjectExecutionContext(
                      directProject,
                      projectTools.root,
                      root,
                    );
                    const childTools = new WorkspaceTools(childBinding.cwd);
                    const childToolOptions: WorkspaceAgentToolOptions = {
                      ...baseWorkspaceToolOptions,
                      reviewPermission: reviewWorkspacePermission,
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
                        runWithPermissionReviewer(providerSecrets, observation, () =>
                          executeComposition(params.agentComposition, childMessages, {
                            inventory: protectedInventory,
                            providerSecrets,
                            cwd: childBinding.cwd,
                            project: childBinding.project,
                            ...(compositionUsesAcp
                              ? {}
                              : {
                                  contextEngine: createRunContextEngine(
                                    `${params.sessionId ?? params.requestId}:agent:${agentId}`,
                                    [],
                                  ),
                                }),
                            acpPermissionHandler,
                            memoryTools: agentMemoryTools,
                            localTools: [
                              ...agentPrivateKnowledgeTools,
                              ...workspaceAgentTools(childTools, undefined, childToolOptions),
                            ],
                            onChunk: () => observation.markOutput(),
                            onUsage: (usage) => {
                              childUsages.push(usage);
                              observation.recordUsage(usage);
                            },
                            onProjectBootstrap: (receipt) => {
                              if (!params.sessionId) {
                                throw new Error(
                                  "Child Project bootstrap requires a parent Session.",
                                );
                              }
                              const receiptMessage = uniqueProjectBootstrapReceiptMessage(
                                projectBootstrapReceiptKeys,
                                `${params.requestId}:child:${agentId}`,
                                receipt,
                              );
                              if (!receiptMessage) return;
                              const message = {
                                ...receiptMessage,
                                agent: agentId,
                              };
                              if (!appendMessages(params.sessionId, [message])) {
                                throw new Error(`Session ${params.sessionId} no longer exists.`);
                              }
                              publishSessionMessages(event.sender, params.sessionId);
                            },
                            ...(globalMemorySnapshot ? { globalMemory: globalMemorySnapshot } : {}),
                            ...(memoryReflection ? { memoryReflection } : {}),
                          }),
                        ),
                    );
                    return { messages, usages: childUsages };
                  },
                })
              : null;
          const workspaceToolOptions: WorkspaceAgentToolOptions = {
            ...baseWorkspaceToolOptions,
            reviewPermission: reviewWorkspacePermission,
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
          const contextHistory = activeSideChat
            ? [...activeSideChat.anchorMessages, ...activeSideChat.messages]
            : (directSession?.messages ?? []);
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
              runWithPermissionReviewer(providerSecrets, observation, () =>
                executeComposition(
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
                    project: directProject,
                    ...(compositionUsesAcp
                      ? {}
                      : {
                          contextEngine: createRunContextEngine(
                            params.sideChatId ?? params.sessionId ?? params.requestId,
                            contextHistory,
                          ),
                        }),
                    acpPermissionHandler,
                    acpSessionId,
                    onAcpSessionId,
                    ...(params.sideChatId
                      ? {
                          acpMode: "plan",
                          allowAgentFacingMcp: false,
                          allowUnboundProjectMcp: false,
                        }
                      : {}),
                    ...(() => {
                      const localTools = [
                        ...agentPrivateKnowledgeTools,
                        ...(projectTools
                          ? workspaceAgentTools(
                              projectTools,
                              sessionRuntime?.shell,
                              workspaceToolOptions,
                            )
                          : []),
                      ];
                      return {
                        ...(agentMemoryTools.length > 0 ? { memoryTools: agentMemoryTools } : {}),
                        ...(localTools.length > 0 ? { localTools } : {}),
                      };
                    })(),
                    onChunk: (chunk) => {
                      observation.markOutput();
                      onChunk(chunk);
                    },
                    onUsage: (usage) => {
                      tokenUsages.push(usage);
                      observation.recordUsage(usage);
                    },
                    onProjectBootstrap: (receipt) => {
                      const message = uniqueProjectBootstrapReceiptMessage(
                        projectBootstrapReceiptKeys,
                        `${params.requestId}:foreground`,
                        receipt,
                      );
                      if (!message) return;
                      projectBootstrapReceipts.push(message);
                      onChunk(message);
                    },
                    ...(globalMemorySnapshot ? { globalMemory: globalMemorySnapshot } : {}),
                    ...(memoryReflection ? { memoryReflection } : {}),
                  },
                ),
              ),
          );
          closeForegroundTurn();
          assertFinalAssistantMessage(messages);
          return {
            success: true,
            messages: [
              ...(memoryReceipt ? [memoryReceipt] : []),
              ...projectBootstrapReceipts,
              ...messages,
            ],
          };
        } else if (params.agentConfig) {
          throw new Error(
            "Inline agentConfig is not accepted by the desktop runtime; use Agent Composition.",
          );
        } else {
          const harness = (await coreRuntimePromise).harnessCatalog.getHarness(params.harnessId);
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
        closeForegroundTurn();

        return {
          success: true,
          messages: [...(memoryReceipt ? [memoryReceipt] : []), ...result],
        };
      });
      closeForegroundTurn();
      const persistedMessages = timedMessages(result.messages, startedAt);
      const { sessionPersisted, sideChat } = persistOutcome(persistedMessages, "completed");
      if (params.sessionId && !params.sideChatId) {
        try {
          await globalMemoryService.recordCompletedTurn({
            sessionId: params.sessionId,
            ...(memoryReflection?.due ? { reviewedThrough: memoryReflection.throughUserTurn } : {}),
          });
        } catch (error) {
          console.warn(`Failed to persist Memory review cursor: ${errorMessage(error)}`);
        }
      }
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
      closeForegroundTurn();
      if (err instanceof SessionRequestError) {
        return {
          success: false,
          requestId: params.requestId,
          error: err.code,
        };
      }
      if (err instanceof RequestCancelledError) {
        const canceledMessages = interruptedMessages(observedMessages, startedAt);
        const { sessionPersisted, sideChat } = persistOutcome(canceledMessages, "canceled");
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
      const { sessionPersisted, sideChat } = persistOutcome(failedMessages, "failed");
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
      closeForegroundTurn();
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
    ): SessionData =>
      createSession(params.agentName, params.harness, params.model, {
        projectId: params.projectId,
        cwd: params.cwd,
        permissionMode: params.permissionMode,
      }),
  );

  ipcMain.handle("session:save", async (_event: IpcMainInvokeEvent, session: SessionData) => {
    await validateMessageAttachments(session.messages);
    saveSession(session);
  });

  ipcMain.handle("session:load", (_event: IpcMainInvokeEvent, id: string): SessionData | null => {
    return loadSession(id);
  });

  ipcMain.handle("session:list", (): SessionSummary[] => listSessionSummaries());

  ipcMain.handle(
    "session:listGrouped",
    async (_event: IpcMainInvokeEvent, params?: ListGroupedSessionsOptions) => {
      const status = await harnessEnvironment.status();
      const runtime = await coreRuntimePromise;
      return runtime.listGroupedSessions({
        ...(params ?? {}),
        harnessIds: sessionDiscoveryHarnessIds(
          status,
          params?.harnessIds,
          (await coreRuntimePromise).harnessCatalog,
        ),
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
      return (await coreRuntimePromise).loadDiscoveredSession(session);
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
        const plan = resolveAgentCompositionPlan(
          composition,
          inventory,
          (await coreRuntimePromise).harnessCatalog,
        );
        assertCompositionSupplyReady(inventory, plan, process.env);
        const providerRuntime = plan.modelSupplyId
          ? await modelCatalog.runtimeCredentialsForSupply(inventory, plan.modelSupplyId)
          : undefined;
        const messages = await executeWithProviderRuntime(
          providerRuntime,
          `${session.id}:title`,
          (providerSecrets, observation) =>
            executeComposition(composition, sessionTitleMessages(params.userText), {
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
    return runtimeInventoryWithPlans(await modelCatalog.list(inventory));
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
    return runtimeInventoryWithPlans(await modelCatalog.list(inventory));
  });

  ipcMain.handle("customAgent:save", async (_event: IpcMainInvokeEvent, input: unknown) => {
    const inventory = await loadDesktopExtensionInventory();
    await customAgents.save(input, {
      reservedAgentIds: inventory.agents.map((agent) => agent.id),
    });
    return runtimeInventoryWithPlans(await modelCatalog.list(inventory));
  });

  ipcMain.handle(
    "customAgent:remove",
    async (_event: IpcMainInvokeEvent, params: { id: string }) => {
      await customAgents.remove(params.id);
      const inventory = await loadDesktopExtensionInventory();
      return runtimeInventoryWithPlans(await modelCatalog.list(inventory));
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
    return runtimeInventoryWithPlans(await modelCatalog.refresh(inventory));
  });

  ipcMain.handle(
    "modelCatalog:addManualModel",
    async (_event: IpcMainInvokeEvent, input: ManualModelInput) => {
      const inventory = await loadDesktopExtensionInventory();
      return runtimeInventoryWithPlans(await modelCatalog.addManualModel(inventory, input));
    },
  );

  ipcMain.handle(
    "modelCatalog:removeManualModel",
    async (_event: IpcMainInvokeEvent, params: { modelId: string }) => {
      const inventory = await loadDesktopExtensionInventory();
      return runtimeInventoryWithPlans(
        await modelCatalog.removeManualModel(inventory, params.modelId),
      );
    },
  );

  ipcMain.handle(
    "modelCatalog:saveProvider",
    async (_event: IpcMainInvokeEvent, input: UserProviderInput) => {
      const inventory = await loadDesktopExtensionInventory();
      return runtimeInventoryWithPlans(await modelCatalog.saveProvider(inventory, input));
    },
  );

  ipcMain.handle(
    "modelCatalog:removeProvider",
    async (_event: IpcMainInvokeEvent, params: { providerId: string }) => {
      const inventory = await loadDesktopExtensionInventory();
      return runtimeInventoryWithPlans(
        await modelCatalog.removeProvider(inventory, params.providerId),
      );
    },
  );

  ipcMain.handle(
    "modelCatalog:resetProviderKey",
    async (_event: IpcMainInvokeEvent, params: { providerId: string; keyId: string }) => {
      const inventory = await loadDesktopExtensionInventory();
      return runtimeInventoryWithPlans(
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

  ipcMain.handle(
    "doctor:inspect",
    async (_event: IpcMainInvokeEvent, params?: { harnessId?: string }) =>
      harnessDoctor.inspect({
        ...(params?.harnessId ? { harnessId: params.harnessId } : {}),
        readiness: await desktopDoctorReadiness(params?.harnessId),
      }),
  );

  ipcMain.handle(
    "doctor:fix",
    async (_event: IpcMainInvokeEvent, params: { harnessId?: string; confirmed: boolean }) =>
      harnessDoctor.fix({
        ...params,
        readiness: await desktopDoctorReadiness(params.harnessId),
      }),
  );

  ipcMain.handle(
    "harnessEnvironment:setup",
    (_event: IpcMainInvokeEvent, params?: HarnessEnvironmentSetupRequest) =>
      harnessEnvironment.setup(params ?? {}),
  );

  ipcMain.handle(
    "lsp:complete",
    async (_event: IpcMainInvokeEvent, params: LspCompletionRequest) => {
      const inventory = await loadDesktopExtensionInventory();
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
  runCleanupActions([
    () => browserHost.dispose(),
    () => terminalHost.dispose(),
    () => interactiveOwnerIds.clear(),
  ]);
}

export function resolveDesktopMediaProtocolUrl(url: string): Promise<string> {
  return mediaService.resolveProtocolUrl(url);
}

function ensureInteractiveOwner(owner: IpcMainInvokeEvent["sender"]): void {
  if (interactiveOwnerIds.has(owner.id)) return;
  interactiveOwnerIds.add(owner.id);
  owner.once("destroyed", () => {
    interactiveOwnerIds.delete(owner.id);
    runCleanupActions([
      () => browserHost.cleanupOwner(owner.id),
      () => terminalHost.cleanupOwner(owner.id),
    ]);
  });
}

function runCleanupActions(actions: ReadonlyArray<() => void>): void {
  let firstFailure: unknown;
  let failed = false;
  for (const action of actions) {
    try {
      action();
    } catch (error) {
      if (!failed) firstFailure = error;
      failed = true;
    }
  }
  if (failed) throw firstFailure;
}

export function sessionDiscoveryHarnessIds(
  status: HarnessEnvironmentStatus,
  requestedHarnessIds?: string[],
  harnessCatalog: HarnessCatalog = staticHarnessCatalog,
): string[] {
  const readyNativeCustomHarnessIds = status.harnesses
    .filter((harness) => {
      if (harness.status !== "ready" || harness.executionMode !== "native") return false;
      return harnessCatalog.getHarness(harness.harnessId)?.backend.type === "custom";
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
  harnessCatalog: HarnessCatalog = staticHarnessCatalog,
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
        harnessCatalog,
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
  workspaceDir?: string,
): Promise<AgentBackend> {
  const result = await harnessEnvironment.protectedBackendForHarness(harnessId, backend, {
    workspaceDir: workspaceDir ?? process.cwd(),
  });
  if (!result.success || !result.backend) {
    throw new Error(result.error ?? "Protected harness runtime is not ready.");
  }
  return result.backend;
}

async function protectCompositionHarness(
  inventory: ExtensionInventory,
  harnessId: string | undefined,
  workspaceDir?: string,
): Promise<ExtensionInventory> {
  if (!harnessId) return inventory;
  const matches = inventory.harnesses.filter((harness) => harness.id === harnessId);
  if (matches.length !== 1) return inventory;
  const runtimeHarnessId = matches[0].runtimeHarnessId ?? harnessId;
  const protectedBackend = await protectedBackendForHarness(
    runtimeHarnessId,
    matches[0].backend,
    workspaceDir,
  );
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

async function protectSwarmConfigBackends(
  config: SwarmConfig,
  workspaceDir?: string,
): Promise<SwarmConfig> {
  return transformSwarmConfigAgentBackends(config, async (backend) => {
    const harnessId = harnessEnvironment.guessProtectedHarnessId(backend);
    return protectedBackendForHarness(
      harnessId ?? (backend.type === "custom" ? "unregistered-custom" : "native"),
      backend,
      workspaceDir,
    );
  });
}

async function normalizeWorkingDirectory(cwd?: string): Promise<string | undefined> {
  if (!cwd?.trim()) return undefined;
  const resolved = path.resolve(cwd);
  const info = await stat(resolved);
  if (!info.isDirectory()) throw new Error(`Working directory must be a directory: ${resolved}`);
  return realpath(resolved);
}

export async function bindPersistedSessionProject(
  session: Pick<SessionData, "projectId" | "cwd">,
  expectedProjectId: string | undefined,
  expectedRoot: string,
): Promise<{ id: string; root: string } | undefined> {
  if (!session.projectId && !expectedProjectId && !session.cwd?.trim()) return undefined;
  if (session.projectId !== expectedProjectId) {
    throw new Error("Session Project binding changed after its background runtime was opened.");
  }
  const [root, runtimeRoot] = await Promise.all([
    normalizeWorkingDirectory(session.cwd),
    normalizeWorkingDirectory(expectedRoot),
  ]);
  if (root !== runtimeRoot) {
    throw new Error("Session Project binding changed after its background runtime was opened.");
  }
  return session.projectId && root ? { id: session.projectId, root } : undefined;
}

export function childProjectExecutionContext(
  project: { id: string; root: string } | undefined,
  workspaceRoot: string,
  callbackRoot: string,
): { cwd: string; project?: { id: string; root: string } } {
  const cwd = path.resolve(workspaceRoot);
  if (!callbackRoot.trim() || callbackRoot.includes("\0") || path.resolve(callbackRoot) !== cwd) {
    throw new Error("Child workspace root does not match the host-owned WorkspaceTools root.");
  }
  // EnterWorktree changes execution cwd, not persisted Project authority. Keeping this binding
  // exact lets Core reject Project MCP before provider execution until dual-root proof exists.
  return {
    cwd,
    ...(project ? { project } : {}),
  };
}

export function uniqueProjectBootstrapReceiptMessage(
  seen: Set<string>,
  scope: string,
  receipt: ProjectBootstrapReceipt,
): MessageChunk | undefined {
  const key = JSON.stringify([
    scope,
    receipt.capabilityId,
    receipt.serverName,
    receipt.serverVersion,
    receipt.projectId,
    receipt.registryRevision,
    receipt.snapshotDigest,
  ]);
  if (seen.has(key)) return undefined;
  const message = projectBootstrapReceiptMessage(receipt);
  seen.add(key);
  return message;
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

function permissionReviewUserMessages(sessionId: string | undefined, current: string): string[] {
  const persisted = sessionId ? loadSession(sessionId) : undefined;
  const messages = persisted
    ? sessionChatMessages(persisted)
        .filter((message) => message.role === "user" && message.content.trim().length > 0)
        .map((message) => message.content)
    : [];
  if (current.trim() && messages.at(-1) !== current) messages.push(current);
  return messages.slice(-32);
}

function requiresExplicitPermissionInteraction(input: unknown): boolean {
  if (!isRecord(input)) return false;
  const toolCall = isRecord(input.toolCall) ? input.toolCall : {};
  return [input._meta, toolCall._meta].some(
    (meta) => isRecord(meta) && meta["anthropic/requiresUserInteraction"] === true,
  );
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
      kind: input.decidedBy === "llm" ? "agent" : "user",
      ...(input.decidedBy === "llm"
        ? safeAuditToken(input.reviewerModel)
          ? { id: `model:${safeAuditToken(input.reviewerModel)}` }
          : {}
        : context.ownerId === undefined
          ? {}
          : { id: `renderer:${context.ownerId}` }),
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
      decidedBy: input.decidedBy ?? "user",
      ...(input.risk ? { risk: input.risk } : {}),
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
  const contract = DesktopInvokeContractRegistry[channel];
  const legacyPolicy = (LEGACY_IPC_AUDIT_POLICIES as Readonly<Record<string, IpcAuditPolicy>>)[
    channel
  ];
  if (contract && legacyPolicy) {
    throw new Error(`Desktop IPC channel ${channel} is both contracted and legacy.`);
  }
  const policy = contract?.audit ?? legacyPolicy;
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

function recordsDispatchFailure(
  policy: IpcAuditPolicy,
  receipt: SemanticAuditReceipt,
  error: unknown,
): boolean {
  return (
    error instanceof DesktopIpcBoundaryError ||
    policy !== "semantic_only" ||
    !receipt.semanticAuditRecorded
  );
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
}

function recordExtensionAuthorityAudit(event: ExtensionAuthorityAuditEvent): void {
  activeAuditStore.append({
    category: "extension",
    action: "extension.authority",
    outcome: event.phase,
    actor: { kind: "user" },
    target: { kind: "extension" },
    metadata: {
      pluginId: event.pluginId,
      actionKind: event.action,
      authorityChange: event.authorityChange,
      permissionCount: event.permissionIds.length,
      permissionIds: event.permissionIds.slice(0, 16),
    },
  });
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

function recordSessionActivationAudit(
  auditStore: DesktopAuditStore,
  activation: ClaudeSessionActivation,
  sessionId: string,
  phase: "started" | "bootstrap" | "result" | "failure",
  error?: unknown,
): void {
  const activationId = safeAuditToken(activation.activationId);
  if (!activationId) return;
  auditStore.append({
    category: "session",
    action: `session.activation.${phase}`,
    outcome: phase === "started" ? "attempted" : phase === "failure" ? "failed" : "completed",
    actor: { kind: "system" },
    target: { kind: "activation", id: activationId },
    sessionId,
    activationId,
    metadata: {
      activationId,
      source: activation.source,
      ...(activation.jobId ? { jobId: activation.jobId } : {}),
      ...(activation.taskId ? { taskId: activation.taskId } : {}),
      ...(error ? { errorType: errorName(error) } : {}),
    },
  });
}

function recordLateChunkAudit(
  auditStore: DesktopAuditStore,
  observation: {
    requestId: string;
    sessionId?: string;
    adapter: string;
    chunkKind: MessageChunk["kind"];
    boundary: "closed";
    observationCount: number;
  },
): void {
  const requestId = safeAuditToken(observation.requestId);
  const sessionId = safeAuditToken(observation.sessionId);
  const adapter = safeAuditToken(observation.adapter) ?? "unknown";
  const chunkKind = safeAuditToken(observation.chunkKind) ?? "unknown";
  auditStore.append({
    category: "session",
    action: "session.late_chunk_observed",
    outcome: "completed",
    actor: { kind: "system" },
    target: { kind: "session-output-boundary" },
    ...(requestId ? { requestId } : {}),
    ...(sessionId ? { sessionId } : {}),
    metadata: {
      adapter,
      chunkKind,
      boundary: observation.boundary,
      observationCount: observation.observationCount,
    },
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

function memoryMutationLabel(operation: MemoryAgentMutation["operation"]): string {
  return operation === "create"
    ? "Create"
    : operation === "update"
      ? "Update"
      : operation === "restore"
        ? "Restore"
        : operation === "delete"
          ? "Delete"
          : operation === "global_save"
            ? "Save"
            : operation === "global_forget"
              ? "Forget"
              : "Capture";
}

function memoryMutationApprovalTitle(operation: MemoryAgentMutation["operation"]): string {
  if (operation === "global_save") return "Save global Memory";
  if (operation === "global_forget") return "Forget global Memory";
  if (operation === "capture_research") return "Capture research Memory";
  return `${memoryMutationLabel(operation)} Memory page`;
}

function memoryMutationSummary(mutation: MemoryAgentMutation): string {
  if (mutation.operation === "global_save") {
    return [
      `The Agent proposes replacing ${mutation.target === "user" ? "USER.md" : "MEMORY.md"} for future runs.`,
      `Markdown (${mutation.content.length} characters):`,
      mutation.content,
    ].join("\n\n");
  }
  if (mutation.operation === "global_forget") {
    return `The Agent proposes permanently deleting ${mutation.target === "user" ? "USER.md" : "MEMORY.md"} for future runs.`;
  }
  if (mutation.operation === "capture_research") {
    return [
      `The Agent proposes adding structured research to ${mutation.entities.length} entity page${mutation.entities.length === 1 ? "" : "s"}.`,
      ...mutation.entities.map((entity) =>
        [
          `${entity.title} · ${entity.observations.length} observation${entity.observations.length === 1 ? "" : "s"}`,
          ...entity.observations.map(
            (observation) =>
              `[${observation.kind}] ${observation.claim}\nWhy keep: ${observation.value}`,
          ),
        ].join("\n"),
      ),
    ].join("\n\n");
  }
  if (mutation.operation === "delete") {
    return `The Agent proposes deleting Memory page ${mutation.id} at revision ${mutation.expectedRevision}.`;
  }
  if (mutation.operation === "create") {
    return [
      `The Agent proposes creating Memory page "${mutation.title}".`,
      mutation.aliases?.length ? `Aliases: ${mutation.aliases.join(", ")}` : "Aliases: none",
      `Markdown (${mutation.content.length} characters):`,
      mutation.content,
    ].join("\n\n");
  }
  if (mutation.operation === "restore") {
    return `The Agent proposes restoring Memory page ${mutation.id} at revision ${mutation.expectedRevision} from version ${mutation.version}.`;
  }
  return [
    `The Agent proposes updating Memory page ${mutation.id} at revision ${mutation.expectedRevision}.`,
    ...(mutation.title === undefined ? [] : [`Title: ${mutation.title}`]),
    ...(mutation.aliases === undefined
      ? []
      : [`Aliases: ${mutation.aliases.length ? mutation.aliases.join(", ") : "none"}`]),
    ...(mutation.content === undefined
      ? []
      : [`Markdown (${mutation.content.length} characters):`, mutation.content]),
  ].join("\n\n");
}

function unavailableMemoryBackend(): MemoryBackend & GlobalMemoryBackend {
  const unavailable = async (): Promise<never> => {
    throw new Error("Managed Memory runtime is unavailable on this execution path.");
  };
  return {
    create: unavailable,
    get: unavailable,
    list: unavailable,
    search: unavailable,
    update: unavailable,
    delete: unavailable,
    graph: unavailable,
    history: unavailable,
    getVersion: unavailable,
    diff: unavailable,
    restore: unavailable,
    getGlobalMemory: unavailable,
    saveGlobalMemory: unavailable,
    forgetGlobalMemory: unavailable,
  };
}

function memoryBackendWithGlobalMemory(
  backend: MemoryBackend,
  global: GlobalMemoryBackend,
): MemoryBackend & GlobalMemoryBackend {
  return {
    create: (input) => backend.create(input),
    get: (id) => backend.get(id),
    list: () => backend.list(),
    search: (input) => backend.search(input),
    update: (input) => backend.update(input),
    delete: (input) => backend.delete(input),
    graph: () => backend.graph(),
    history: (input) => backend.history(input),
    getVersion: (input) => backend.getVersion(input),
    diff: (input) => backend.diff(input),
    restore: (input) => backend.restore(input),
    getGlobalMemory: () => global.getGlobalMemory(),
    saveGlobalMemory: (input) => global.saveGlobalMemory(input),
    forgetGlobalMemory: (input) => global.forgetGlobalMemory(input),
  };
}

function supportsGlobalMemory(
  backend: MemoryBackend & Partial<GlobalMemoryBackend>,
): backend is MemoryBackend & GlobalMemoryBackend {
  return (
    typeof backend.getGlobalMemory === "function" &&
    typeof backend.saveGlobalMemory === "function" &&
    typeof backend.forgetGlobalMemory === "function"
  );
}

function unavailableGlobalMemoryBackend(): GlobalMemoryBackend {
  const unavailable = async (): Promise<never> => {
    throw new Error("Managed global Memory runtime is unavailable on this execution path.");
  };
  return {
    getGlobalMemory: unavailable,
    saveGlobalMemory: unavailable,
    forgetGlobalMemory: unavailable,
  };
}

function foregroundUserTurnCount(sessionId: string, currentUserText: string): number {
  const messages = loadSession(sessionId)?.messages ?? [];
  const userMessages = messages.filter((message) => message.role === "user");
  const currentPersisted = userMessages.at(-1)?.content === currentUserText;
  return Math.max(1, userMessages.length + (currentPersisted ? 0 : 1));
}

async function globalMemorySnapshotForRun(
  service: GlobalMemoryServiceLike,
): Promise<{ snapshot: GlobalMemorySnapshot | null; unavailable: boolean }> {
  try {
    return { snapshot: await service.snapshot(), unavailable: false };
  } catch (error) {
    console.warn(`Global Memory snapshot unavailable: ${errorMessage(error)}`);
    return { snapshot: null, unavailable: true };
  }
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
