import type {
  ActivityProfileSummary,
  AgentCompositionPlan,
  AgentProfile,
  AuditEvent,
  AuditQuery,
  AuditVerification,
  DesktopBuiltinToolSettings,
  DesktopComposerPreferences,
  DesktopComposerPreferenceUpdate,
  ExtensionActionReceipt,
  ExtensionCandidate,
  ExtensionInventory,
  ExtensionMarketplaceSource,
  ExternalAcpSessionBinding,
  HarnessPermissionMode,
  HarnessPermissionPolicyLayer,
  HarnessRecipe,
  InstalledExtension,
  MediaAttachment,
  Model,
  ModelSupply,
  PermissionApprovalReceipt,
  ProviderProfile,
  ResolvedHarnessPermissionPolicy,
  SessionPermissionMode,
  SkillCapability,
  SwarmConfig,
  TransientSessionData,
} from "@swarmx/core";
import type {
  NormalizeMessageChunkOptions,
  RenderArtifactReference,
  RenderProvenance,
} from "@swarmx/core/rendering";
import type {
  DoctorFixResult,
  DoctorReport,
  HarnessEnvironmentSetupResult,
  HarnessEnvironmentStatus,
  HarnessVersionCheck,
} from "@swarmx/runtime";
import type { DesktopUpdateState } from "./ipc-contracts/app-update.js";
import type { DesktopBrowserApi } from "./ipc-contracts/browser.js";
import type { DesktopGlobalMemoryApi } from "./ipc-contracts/global-memory.js";
import type { DesktopProjectData as ProjectData } from "./ipc-contracts/project.js";
import type { DesktopTaskRuntimeApi } from "./ipc-contracts/task-runtime.js";
import type { DesktopTerminalApi } from "./ipc-contracts/terminal.js";
import type { DesktopWorkspaceInspectionApi } from "./ipc-contracts/workspace-inspection.js";

export type { DesktopUpdatePhase, DesktopUpdateState } from "./ipc-contracts/app-update.js";
export type {
  DesktopBrowserApi,
  DesktopBrowserBounds,
  DesktopBrowserCreateInput,
  DesktopBrowserState,
} from "./ipc-contracts/browser.js";
export type {
  DesktopGlobalMemoryApi,
  DesktopGlobalMemoryForgetInput,
  DesktopGlobalMemorySaveInput,
  DesktopGlobalMemoryState,
} from "./ipc-contracts/global-memory.js";
export type {
  DesktopTaskRuntimeApi,
  DesktopTaskRuntimeCancelInput,
  DesktopTaskRuntimeDecisionInput,
  DesktopTaskRuntimeListResult,
  DesktopTaskRuntimeWorkItemResult,
} from "./ipc-contracts/task-runtime.js";
export type {
  DesktopTerminalApi,
  DesktopTerminalCreateInput,
  DesktopTerminalCreateResult,
  DesktopTerminalDataEvent,
  DesktopTerminalExitEvent,
} from "./ipc-contracts/terminal.js";
export type {
  DesktopWorkspaceDirectoryEntry,
  DesktopWorkspaceDirectoryListing,
  DesktopWorkspaceFilePreview,
  DesktopWorkspaceInspectionApi,
  DesktopWorkspaceReviewFile,
  DesktopWorkspaceReviewSnapshot,
} from "./ipc-contracts/workspace-inspection.js";

export interface DesktopMessageRenderMetadata {
  artifacts?: RenderArtifactReference[];
  durationMs?: number;
  endedAt?: string;
  invocationId?: string;
  parentMessageId?: string;
  provenance?: RenderProvenance;
  rawPayloadRef?: string;
  source?: string;
  startedAt?: string;
  status?: NormalizeMessageChunkOptions["status"];
}

export interface DesktopMessageChunk {
  role: string;
  content: string;
  kind: "message" | "thinking" | "tool_call" | "tool_progress" | "tool_result";
  agent?: string;
  createdAt?: string;
  render?: DesktopMessageRenderMetadata;
  structuredContent?: unknown;
  swarmEvent?: string;
  toolName?: string;
  attachments?: DesktopMediaAttachment[];
}

export type DesktopMediaAttachment = MediaAttachment;

export interface DesktopMediaImport {
  name: string;
  mimeType?: string;
  bytes: Uint8Array;
}

export interface DesktopMediaPreview {
  status: "available" | "unavailable";
  attachment: DesktopMediaAttachment;
  previewUrl?: string;
  text?: string;
  error?: string;
}

export interface DesktopSessionData {
  id: string;
  title: string;
  acpSessionId?: string;
  externalAcpSession?: ExternalAcpSessionBinding;
  forkedFrom?: {
    sessionId: string;
    messageIndex: number;
    createdAt: string;
  };
  projectId?: string;
  cwd?: string;
  agentName: string;
  harness: string;
  model?: string;
  permissionMode?: SessionPermissionMode;
  pinned: boolean;
  messages: DesktopMessageChunk[];
  archivedAt?: string;
  createdAt: string;
  updatedAt: string;
}

export type DesktopSessionSummary = Omit<DesktopSessionData, "messages"> & {
  messageCount: number;
};

export interface DesktopAgentChunkEvent {
  requestId: string;
  chunk: DesktopMessageChunk;
}

export type DesktopSideChat = Omit<TransientSessionData, "messages" | "anchorMessages"> & {
  anchorMessages: DesktopMessageChunk[];
  messages: DesktopMessageChunk[];
};

export interface DesktopSideChatParentState {
  parentSessionId: string;
  activeSideChatId: string | null;
  paneHidden: boolean;
  chats: DesktopSideChat[];
}

export interface DesktopSideChatChunkEvent extends DesktopAgentChunkEvent {
  parentSessionId: string;
  sideChatId: string;
}

export interface DesktopSessionMessagesEvent {
  sessionId: string;
}

export interface DesktopAgentQuestionOption {
  label: string;
  description: string;
  preview?: string;
}

export interface DesktopAgentQuestion {
  question: string;
  header: string;
  options: DesktopAgentQuestionOption[];
  multiSelect: boolean;
}

export interface DesktopToolApprovalOption {
  optionId: string;
  name: string;
  kind: "allow_once" | "allow_always" | "reject_once" | "reject_always";
}

export type DesktopAgentInteractionEvent =
  | {
      kind: "questions";
      requestId: string;
      interactionId: string;
      questions: DesktopAgentQuestion[];
    }
  | {
      kind: "plan_approval";
      requestId: string;
      interactionId: string;
      plan: string;
      filePath: string;
    }
  | {
      kind: "tool_approval";
      requestId: string;
      interactionId: string;
      title: string;
      toolKind?: string;
      source?: "direct" | "acp";
      policySourceIds?: string[];
      summary: string;
      options: DesktopToolApprovalOption[];
    };

export type DesktopAgentInteractionResponse =
  | { kind: "questions"; answers: Record<string, string> }
  | { kind: "plan_approval"; approved: boolean; feedback?: string }
  | { kind: "tool_approval"; optionId: string };

export interface DesktopPermissionLayerStatus {
  id: string;
  source: HarnessPermissionPolicyLayer["source"];
  label: string;
  configured: boolean;
  readOnly: boolean;
  mode?: HarnessPermissionMode;
  allowedTools: string[];
  deniedTools: string[];
  error?: string;
}

export interface DesktopPermissionStatus {
  personalPolicy: HarnessPermissionPolicyLayer;
  defaultMode: "default" | "auto";
  profileAvailability: {
    default: boolean;
    auto: boolean;
    trusted: boolean;
  };
  layers: DesktopPermissionLayerStatus[];
  effective?: ResolvedHarnessPermissionPolicy;
  blocked: boolean;
  projectPolicyPath: string;
  approvalReceipts: PermissionApprovalReceipt[];
}

export type ModelApiProtocol = "anthropic" | "openai_chat" | "openai_responses" | "ollama";

export interface ManualModelInput {
  id: string;
  label?: string;
  runtimeModel?: string;
  apiProtocol: ModelApiProtocol;
}

export interface UserProviderInput {
  id?: string;
  label: string;
  kind: ModelApiProtocol;
  baseUrl: string;
  authMode: "api_key" | "auth_token";
  usageAdapter?: "new_api";
  secret?: string;
  accountAccessToken?: string;
  accountUserId?: string;
  clearAccountAccess?: boolean;
  additionalApiKeys?: Array<{ label?: string; value: string }>;
  removeApiKeyIds?: string[];
}

export interface ProviderKeyUsageSummary {
  id: string;
  label: string;
  enabled: boolean;
  status: "ready" | "cooling" | "disabled";
  requestCount: number;
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  cachedInputTokens: number;
  totalTokens: number;
  lastUsedAt?: string;
  cooldownUntil?: string;
}

export interface ModelCatalogSummary {
  manualModelIds: string[];
  userProviderIds: string[];
  providers: Array<{
    providerProfileId: string;
    label: string;
    status: "cached" | "ready" | "skipped" | "error";
    modelCount: number;
    fetchedAt?: string;
    error?: string;
  }>;
  refreshedAt?: string;
}

export type ProviderUsageMeter =
  | {
      kind: "balance";
      label: string;
      currency: string;
      total: string;
      granted?: string;
      toppedUp?: string;
    }
  | {
      kind: "window";
      id: string;
      label: string;
      usedPercent: number;
      remainingPercent: number;
      resetsAt?: string;
    }
  | { kind: "credit"; label: string; remaining: string; unit: string };

export interface NewApiUsageAccount {
  kind: "new_api";
  status: "ready" | "unavailable" | "error";
  displayName?: string;
  group?: string;
  balance?: {
    remaining: string;
    used: string;
    total: string;
    unit: string;
  };
  tokens: Array<{
    id: string;
    name: string;
    status: "active" | "disabled" | "exhausted" | "expired" | "unknown";
    remaining: string;
    used?: string;
    total?: string;
    lastUsedAt?: string;
    expiresAt?: string;
  }>;
  totalTokens: number;
  detail?: string;
}

export interface ProviderUsageEntry {
  source: "provider" | "tool_account";
  sourceId: string;
  providerProfileId?: string;
  label: string;
  adapterId: string;
  status: "ready" | "unsupported" | "unavailable" | "error";
  meters: ProviderUsageMeter[];
  fetchedAt?: string;
  detail?: string;
  plan?: string;
  account?: NewApiUsageAccount;
  keys?: ProviderKeyUsageSummary[];
}

export interface ProviderUsageSnapshot {
  fetchedAt: string;
  providers: ProviderUsageEntry[];
  toolAccounts: ProviderUsageEntry[];
}

export interface ProviderUsageTarget {
  source: "provider" | "tool_account";
  sourceId: string;
}

export interface ExtensionManagementState {
  sources: ExtensionMarketplaceSource[];
  candidates?: ExtensionCandidate[];
  installed: InstalledExtension[];
  skillEvolutionEnabled: boolean;
  skillPromotionGate: "human" | "policy";
}

export type DesktopExtensionModel = Model & {
  catalogSource?: string;
  catalogSources?: string[];
};

export type DesktopExtensionProvider = ProviderProfile & {
  usageAdapter?: "new_api";
  newApiAccountUserId?: string;
  accountAccessReady?: boolean;
  catalogAdapter?: string;
  runtimeKeySlots?: Array<{ id: string; label: string; enabled: boolean }>;
  runtimeKeyUsage?: ProviderKeyUsageSummary[];
};

export type DesktopExtensionAgent = AgentProfile & {
  harnessRecipe?: HarnessRecipe;
};

export type DesktopExtensionSkill = SkillCapability & {
  tokenEstimate?: number;
  variants?: Array<{
    id: string;
    version?: string;
    tokenEstimate?: number;
    status?: string;
    target?: {
      agentProfileIds?: string[];
      modelIds?: string[];
      modelFamilies?: string[];
    };
  }>;
};

export type ExtensionCapabilityInventory = Omit<
  ExtensionInventory,
  "models" | "modelSupplies" | "providers" | "agents" | "skills"
> & {
  models: DesktopExtensionModel[];
  modelSupplies: ModelSupply[];
  providers: DesktopExtensionProvider[];
  agents: DesktopExtensionAgent[];
  skills: DesktopExtensionSkill[];
  agentPlans?: AgentCompositionPlan[];
  modelCatalog?: ModelCatalogSummary;
};

export interface DesktopN8nImportResponse {
  success: boolean;
  config?: SwarmConfig;
  warnings?: string[];
  nodeMap?: Record<string, string>;
  error?: string;
}

export interface DesktopSendMessageResult {
  success: boolean;
  messages?: unknown;
  error?: string;
  canceled?: boolean;
  requestId?: string;
  sessionPersisted?: boolean;
  replayed?: boolean;
  sideChat?: DesktopSideChat;
}

export interface SwarmxAPI
  extends DesktopBrowserApi,
    DesktopGlobalMemoryApi,
    DesktopTaskRuntimeApi,
    DesktopTerminalApi,
    DesktopWorkspaceInspectionApi {
  readonly initialProjects?: readonly ProjectData[];
  sendMessage(params: {
    requestId: string;
    sessionId?: string;
    harnessId: string;
    userText: string;
    editMessageIndex?: number;
    editExpectedMessages?: DesktopMessageChunk[];
    attachments?: DesktopMediaAttachment[];
    agentComposition?: unknown;
    swarmConfig?: unknown;
    cwd?: string;
  }): Promise<DesktopSendMessageResult>;
  onAgentChunk(listener: (event: DesktopAgentChunkEvent) => void): () => void;
  sendSideChatMessage(params: {
    requestId: string;
    sessionId: string;
    sideChatId: string;
    sideChatVisible: boolean;
    sideEditMessageIndex?: number;
    harnessId: string;
    userText: string;
    agentComposition?: unknown;
    cwd?: string;
  }): Promise<DesktopSendMessageResult>;
  onSideChatChunk(listener: (event: DesktopSideChatChunkEvent) => void): () => void;
  listSideChats(parentSessionId: string): Promise<DesktopSideChatParentState>;
  createSideChat(params: {
    parentSessionId: string;
    throughMessageIndex: number;
    expectedMessages: DesktopMessageChunk[];
    title?: string;
  }): Promise<DesktopSideChat>;
  updateSideChat(params: {
    parentSessionId: string;
    sideChatId: string;
    draft?: string;
    attachments?: string[];
    title?: string;
    unread?: boolean;
  }): Promise<DesktopSideChat>;
  activateSideChat(
    parentSessionId: string,
    sideChatId: string,
  ): Promise<DesktopSideChatParentState>;
  setSideChatHidden(parentSessionId: string, hidden: boolean): Promise<DesktopSideChatParentState>;
  addSideChatContext(
    parentSessionId: string,
    sideChatId: string,
    text: string,
  ): Promise<DesktopSideChat>;
  editSideChatMessage(params: {
    parentSessionId: string;
    sideChatId: string;
    messageIndex: number;
    content: string;
  }): Promise<DesktopSideChat>;
  deleteSideChat(parentSessionId: string, sideChatId: string): Promise<DesktopSideChatParentState>;
  promoteSideChat(parentSessionId: string, sideChatId: string): Promise<DesktopSessionData>;
  cancelSideChat(
    parentSessionId: string,
    sideChatId: string,
    requestId: string,
  ): Promise<{ requestId: string; sideChatId: string; canceled: boolean }>;
  onAgentInteraction(listener: (event: DesktopAgentInteractionEvent) => void): () => void;
  onSessionMessages?(listener: (event: DesktopSessionMessagesEvent) => void): () => void;
  resolveAgentInteraction(params: {
    requestId: string;
    interactionId: string;
    response: DesktopAgentInteractionResponse;
  }): Promise<{ requestId: string; interactionId: string; resolved: boolean }>;
  cancelMessage(requestId: string): Promise<{ requestId: string; canceled: boolean }>;
  createSession(params: {
    agentName: string;
    harness: string;
    model?: string;
    projectId?: string;
    cwd?: string;
    permissionMode?: SessionPermissionMode;
  }): Promise<DesktopSessionData>;
  saveSession(session: DesktopSessionData): Promise<void>;
  loadSession(id: string): Promise<DesktopSessionData | null>;
  loadDiscoveredSession(session: {
    id: string;
    title: string;
    projectId?: string;
    cwd: string;
    pinned?: boolean;
    updatedAt?: string;
    harnessId: string;
    harnessLabel: string;
    source: "local" | "acp";
  }): Promise<DesktopSessionData | null>;
  listSessions(): Promise<DesktopSessionSummary[]>;
  getActivityProfile(): Promise<ActivityProfileSummary>;
  listAuditEvents(query?: AuditQuery): Promise<AuditEvent[]>;
  verifyAuditLog(): Promise<AuditVerification>;
  exportAuditLog(
    query?: AuditQuery,
  ): Promise<{ exported: boolean; canceled?: boolean; eventCount?: number }>;
  listProjects(): Promise<ProjectData[]>;
  addExistingProject(): Promise<ProjectData | null>;
  createScratchProject(): Promise<ProjectData | null>;
  setProjectPinned(id: string, pinned: boolean): Promise<ProjectData>;
  renameProject(id: string, name: string): Promise<ProjectData>;
  revealProject(id: string): Promise<boolean>;
  archiveProjectTasks(id: string): Promise<number>;
  removeProject(id: string): Promise<boolean>;
  listGroupedSessions(params?: {
    mode?: "project" | "harness";
    cwd?: string;
    harnessIds?: string[];
  }): Promise<{
    mode: "project" | "harness";
    groups: Array<{
      id: string;
      label: string;
      sessions: Array<{
        id: string;
        title: string;
        projectId?: string;
        cwd: string;
        pinned?: boolean;
        updatedAt?: string;
        harnessId: string;
        harnessLabel: string;
        source: "local" | "acp";
      }>;
    }>;
    errors: Array<{ harnessId: string; harnessLabel: string; message: string }>;
  }>;
  archiveSession(id: string): Promise<DesktopSessionData>;
  renameSession(id: string, title: string): Promise<DesktopSessionData>;
  setSessionPinned(id: string, pinned: boolean): Promise<DesktopSessionData>;
  generateSessionTitle(id: string, userText: string): Promise<{ title: string; updated: boolean }>;
  appendMessages(params: { id: string; messages: unknown[] }): Promise<boolean>;
  editSessionUserMessage(params: {
    id: string;
    messageIndex: number;
    expectedMessages: DesktopMessageChunk[];
    content: string;
  }): Promise<DesktopSessionData>;
  forkSession(params: {
    id: string;
    throughMessageIndex: number;
    expectedMessages: DesktopMessageChunk[];
  }): Promise<DesktopSessionData>;
  importN8nWorkflow(source: string): Promise<DesktopN8nImportResponse>;
  listExtensions(): Promise<ExtensionCapabilityInventory>;
  getExtensionManagementState(): Promise<ExtensionManagementState>;
  saveExtensionSource(input: unknown): Promise<ExtensionManagementState>;
  refreshExtensionSource(id: string): Promise<ExtensionManagementState>;
  removeExtensionSource(id: string): Promise<ExtensionManagementState>;
  applyExtensionAction(input: unknown): Promise<{
    state: ExtensionManagementState;
    receipt: ExtensionActionReceipt;
  }>;
  saveSkillEvolutionPolicy(input: {
    enabled: boolean;
    promotionGate: "human" | "policy";
  }): Promise<ExtensionManagementState>;
  listCustomAgents(): Promise<ExtensionCapabilityInventory>;
  saveCustomAgent(input: unknown): Promise<ExtensionCapabilityInventory>;
  removeCustomAgent(id: string): Promise<ExtensionCapabilityInventory>;
  getComposerPreferences(): Promise<DesktopComposerPreferences>;
  saveComposerPreference(
    input: DesktopComposerPreferenceUpdate,
  ): Promise<DesktopComposerPreferences>;
  getBuiltinToolSettings(): Promise<DesktopBuiltinToolSettings>;
  saveBuiltinToolSettings(input: DesktopBuiltinToolSettings): Promise<DesktopBuiltinToolSettings>;
  getPermissionStatus(params?: {
    cwd?: string;
    agentId?: string;
    agentPolicy?: unknown;
  }): Promise<DesktopPermissionStatus>;
  savePersonalPermissionPolicy(
    policy: unknown,
    context?: { cwd?: string; agentId?: string; agentPolicy?: unknown },
  ): Promise<DesktopPermissionStatus>;
  savePermissionProfileAvailability(
    profileAvailability: unknown,
    context?: { cwd?: string; agentId?: string; agentPolicy?: unknown },
  ): Promise<DesktopPermissionStatus>;
  getUpdateState?(): Promise<DesktopUpdateState>;
  startUpdate?(): Promise<DesktopUpdateState>;
  onUpdateState?(listener: (state: DesktopUpdateState) => void): () => void;
  selectFilesAndFolders(): Promise<string[]>;
  selectMediaAttachments(
    existingAttachments: readonly DesktopMediaAttachment[],
  ): Promise<DesktopMediaAttachment[]>;
  importMediaAttachments(
    files: DesktopMediaImport[],
    existingAttachments: readonly DesktopMediaAttachment[],
  ): Promise<DesktopMediaAttachment[]>;
  previewMediaAttachment(attachment: DesktopMediaAttachment): Promise<DesktopMediaPreview>;
  openMediaAttachment(
    attachment: DesktopMediaAttachment,
  ): Promise<{ opened: boolean; error?: string }>;
  revealMediaAttachment(attachment: DesktopMediaAttachment): Promise<{ revealed: boolean }>;
  refreshModelCatalog(): Promise<ExtensionCapabilityInventory | null>;
  addManualModel(input: ManualModelInput): Promise<ExtensionCapabilityInventory | null>;
  removeManualModel(modelId: string): Promise<ExtensionCapabilityInventory | null>;
  saveProvider(input: UserProviderInput): Promise<ExtensionCapabilityInventory | null>;
  removeProvider(providerId: string): Promise<ExtensionCapabilityInventory | null>;
  resetProviderKey(providerId: string, keyId: string): Promise<ExtensionCapabilityInventory | null>;
  refreshProviderUsage(target?: ProviderUsageTarget): Promise<ProviderUsageSnapshot>;
  getHarnessEnvironment(): Promise<HarnessEnvironmentStatus>;
  getHarnessVersion(params: { harnessId: string; refresh?: boolean }): Promise<HarnessVersionCheck>;
  inspectDoctor(params?: { harnessId?: string }): Promise<DoctorReport>;
  fixDoctor(params: { harnessId?: string; confirmed: boolean }): Promise<DoctorFixResult>;
  setupHarnessEnvironment(params?: {
    harnessId?: string;
    harnessToolId?: string;
    requirementIds?: string[];
    containerRuntimeId?: string;
    includeContainerRuntime?: boolean;
  }): Promise<HarnessEnvironmentSetupResult>;
  lspComplete(params: {
    serverId: string;
    workspaceRoot: string;
    text: string;
    position: { line: number; character: number };
    documentUri?: string;
    languageId?: string;
    triggerCharacter?: string;
    timeoutMs?: number;
  }): Promise<{ serverId: string; status: "ok"; result: unknown }>;
  lspStop(params: { serverId: string; workspaceRoot?: string }): Promise<{
    serverId: string;
    stopped: boolean;
  }>;
  loadImageDataUrl(source: string): Promise<string | null>;
}
