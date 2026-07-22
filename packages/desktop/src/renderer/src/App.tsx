import type {
  AgentBackend,
  EdgeConfig,
  HarnessPermissionMode,
  HarnessPermissionPolicyLayer,
  PermissionApprovalReceipt,
  ResolvedHarnessPermissionPolicy,
  SessionPermissionMode,
  SwarmConfig,
  SwarmNodeConfig,
} from "@swarmx/core";
import { resolveHarnessModelInventory } from "@swarmx/core/model-capabilities";
import {
  type NormalizeMessageChunkOptions,
  type NormalizedRenderEvent,
  type RenderArtifactReference,
  type RenderProvenance,
  normalizeMessageChunk,
} from "@swarmx/core/rendering";
import { FitAddon } from "@xterm/addon-fit";
import { Terminal as XtermTerminal } from "@xterm/xterm";
import {
  Archive,
  ArrowLeft,
  ArrowRight,
  Bot,
  Bug,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleCheck,
  Clock3,
  Code2,
  Download,
  FileSearch,
  Folder,
  Gauge,
  GitBranch,
  Hammer,
  KeyRound,
  Loader2,
  type LucideIcon,
  Maximize2,
  MessageCircle,
  MessageSquarePlus,
  Minus,
  MoreHorizontal,
  Package,
  PanelBottom,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRight,
  Pencil,
  Pin,
  Play,
  Plus,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  SquarePen,
  Telescope,
  Terminal as TerminalIcon,
  Trash2,
  Upload,
  User,
  Workflow,
  Wrench,
  X,
  XCircle,
} from "lucide-react";
import type React from "react";
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import useSWR, { useSWRConfig } from "swr";
import {
  AgentInteractionDialog,
  type AgentInteractionEvent,
  type AgentInteractionResponse,
} from "./agent-interaction-dialog.js";
import { Composer } from "./composer.js";
import { PACKAGED_HARNESS_ICON_URLS } from "./harness-icon-data.js";
import { MessageContent } from "./message-content.js";
import {
  compareModelDisplayOrder,
  modelBrandPresentation,
  selectableModelReasoning,
} from "./model-display.js";
import { type ActivityProfileSummary, ProfileWorkspace } from "./profile-workspace.js";
import {
  type BrowserBounds,
  type BrowserState,
  type WorkspaceDirectoryListing,
  type WorkspaceFilePreview,
  WorkspacePanel,
  type WorkspaceReviewSnapshot,
} from "./workspace-panel.js";

interface MessageChunk {
  role: string;
  content: string;
  kind: "message" | "thinking" | "tool_call" | "tool_result";
  agent?: string;
  render?: MessageRenderMetadata;
  swarmEvent?: string;
  toolName?: string;
}

interface ConversationTurn {
  id: string;
  userMessage: MessageChunk | null;
  workMessages: MessageChunk[];
  finalMessage: MessageChunk | null;
}

interface ToolActivity {
  call?: MessageChunk;
  result?: MessageChunk;
  sourceIndex: number;
}

type WorkActivity =
  | { kind: "message"; message: MessageChunk; sourceIndex: number }
  | { kind: "tool"; activity: ToolActivity };

interface MessageRenderMetadata {
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

interface SessionData {
  id: string;
  title: string;
  acpSessionId?: string;
  projectId?: string;
  cwd?: string;
  agentName: string;
  harness: string;
  model?: string;
  permissionMode?: SessionPermissionMode;
  pinned: boolean;
  messages: MessageChunk[];
  archivedAt?: string;
  createdAt: string;
  updatedAt: string;
}

type SessionGroupMode = "project" | "harness";
type ProjectOrganizationMode = "project" | "list";
type ProjectSortMode = "priority" | "last-updated" | "manual";

interface DiscoveredSession {
  id: string;
  title: string;
  projectId?: string;
  cwd: string;
  pinned?: boolean;
  updatedAt?: string;
  harnessId: string;
  harnessLabel: string;
  source: "local" | "acp";
}

interface SessionGroup {
  id: string;
  label: string;
  sessions: DiscoveredSession[];
}

interface ProjectData {
  id: string;
  name: string;
  cwd: string;
  pinned: boolean;
  createdAt: string;
  updatedAt: string;
}

interface ProjectSessionGroup extends SessionGroup {
  project?: ProjectData;
  cwd: string;
}

interface ProjectPreviewState {
  projectId: string;
  top: number;
  left: number;
}

interface SessionDiscoveryError {
  harnessId: string;
  harnessLabel: string;
  message: string;
}

interface GroupedSessionsResult {
  mode: SessionGroupMode;
  groups: SessionGroup[];
  errors: SessionDiscoveryError[];
}

interface SessionContextMenuState {
  session: DiscoveredSession;
  x: number;
  y: number;
}

interface HarnessOption {
  id: string;
  label: string;
  icon: LucideIcon;
  modelControl: "direct" | "session" | "unsupported";
  disabled?: boolean;
  disabledReason?: string;
}

interface ComposerModelOption {
  id: string;
  label: string;
  modelId: string;
  modelSupplyId?: string;
  runtimeModel: string;
  apiProtocol: string;
  providerId: string;
  providerLabel: string;
  providerGroup?: string;
  provider?: ExtensionProviderSummary;
  manual?: boolean;
  reasoning?: {
    supportedEfforts: string[];
    defaultEffort?: string;
  };
}

interface HarnessDescriptor {
  software?: {
    name?: string;
    version?: string;
    runner?: string;
    command?: string[];
  };
  mcps?: Array<{ name?: string; transport?: string; scope?: string } | string>;
  skills?: string[];
  projectFiles?: string[];
}

interface WorkflowGraphNode {
  id: string;
  kind: SwarmNodeConfig["kind"];
  displayKind: "trigger" | SwarmNodeConfig["kind"];
  title: string;
  detail: string;
  isRoot: boolean;
  harnessId?: string;
  harnessLabel?: string;
  harness?: HarnessDescriptor;
  softwareLabel?: string;
  mcpsLabel?: string;
  skillsLabel?: string;
  projectFilesLabel?: string;
  model?: string;
}

interface WorkflowParseResult {
  config: SwarmConfig | null;
  error: string | null;
  nodes: WorkflowGraphNode[];
  edges: EdgeConfig[];
}

interface N8nImportResponse {
  success: boolean;
  config?: SwarmConfig;
  warnings?: string[];
  nodeMap?: Record<string, string>;
  error?: string;
}

interface WorkflowImportStatus {
  kind: "success" | "error";
  message: string;
  warnings: string[];
}

interface ExtensionCapabilityInventory {
  bundles: ExtensionBundleSummary[];
  harnesses: ExtensionHarnessSummary[];
  models: ExtensionModelSummary[];
  modelSupplies: ExtensionModelSupplySummary[];
  providers: ExtensionProviderSummary[];
  agents: ExtensionAgentSummary[];
  skills: ExtensionSkillSummary[];
  mcpServers: ExtensionMcpSummary[];
  appConnectors: ExtensionConnectorSummary[];
  uiContributions: ExtensionUiContributionSummary[];
  commands: ExtensionCommandSummary[];
  lspServers: ExtensionLspSummary[];
  hooks: ExtensionHookSummary[];
  monitors: ExtensionMonitorSummary[];
  outputStyles: ExtensionOutputStyleSummary[];
  settings: ExtensionSettingSummary[];
  assets: ExtensionAssetSummary[];
  permissions: ExtensionPermissionSummary[];
  authPolicies: ExtensionAuthPolicySummary[];
  marketplaceSources: ExtensionMarketplaceSourceSummary[];
  pluginCatalog: ExtensionPluginCatalogEntrySummary[];
  agentPlans?: ExtensionAgentPlanSummary[];
  modelCatalog?: ModelCatalogSummary;
  warnings: Array<{ source: string; message: string }>;
}

interface ManagedExtensionRevision {
  revisionId: string;
  version: string;
  contentDigest: string;
  sourceId: string;
  packageRef?: string;
  publishedAt?: string;
}

interface ManagedExtensionCandidate {
  pluginId: string;
  name: string;
  trust: "builtin" | "local" | "verified" | "untrusted";
  revision: ManagedExtensionRevision;
  description?: string;
}

interface ManagedExtension {
  pluginId: string;
  name: string;
  state: string;
  enabled: boolean;
  trust: string;
  currentRevision?: ManagedExtensionRevision;
  previousRevisions: ManagedExtensionRevision[];
  pinnedRevisionId?: string;
}

interface ExtensionManagementState {
  sources: Array<{
    id: string;
    name: string;
    kind: "local_path" | "remote_catalog" | "host_native" | "registry";
    location: string;
    trust: "builtin" | "local" | "verified" | "untrusted";
    enabled: boolean;
    readOnly: boolean;
    refreshedAt?: string;
  }>;
  candidates?: ManagedExtensionCandidate[];
  installed: ManagedExtension[];
  skillEvolutionEnabled: boolean;
  skillPromotionGate: "human" | "policy";
}

type ModelApiProtocol = "anthropic" | "openai_chat" | "openai_responses" | "ollama";
type SettingsSection =
  | "general"
  | "profile"
  | "permissions"
  | "providers"
  | "extensions"
  | "agents"
  | "runtime";

interface PermissionLayerStatus {
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

interface DesktopPermissionStatus {
  personalPolicy: HarnessPermissionPolicyLayer;
  profileAvailability: {
    default: boolean;
    auto: boolean;
    trusted: boolean;
  };
  layers: PermissionLayerStatus[];
  effective?: ResolvedHarnessPermissionPolicy;
  blocked: boolean;
  projectPolicyPath: string;
  approvalReceipts: PermissionApprovalReceipt[];
}
type DesktopUpdatePhase = "hidden" | "available" | "downloading" | "installing" | "restarting";

interface DesktopUpdateState {
  phase: DesktopUpdatePhase;
  currentVersion: string;
  latestVersion?: string;
  progress?: number;
  error?: string;
}

interface ManualModelInput {
  id: string;
  label?: string;
  runtimeModel?: string;
  apiProtocol: ModelApiProtocol;
}

interface UserProviderInput {
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

interface ProviderKeyUsageSummary {
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

interface ModelCatalogSummary {
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

type ProviderUsageMeter =
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

interface NewApiUsageAccount {
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

interface ProviderUsageEntry {
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

interface ProviderUsageSnapshot {
  fetchedAt: string;
  providers: ProviderUsageEntry[];
  toolAccounts: ProviderUsageEntry[];
}

interface ProviderUsageTarget {
  source: "provider" | "tool_account";
  sourceId: string;
}

interface ExtensionBundleSummary {
  id: string;
  name: string;
  version: string;
  description?: string;
  trust?: string;
  readOnly?: boolean;
  capabilities?: {
    harnesses?: unknown[];
    models?: unknown[];
    modelSupplies?: unknown[];
    agents?: unknown[];
    skills?: unknown[];
    mcpServers?: unknown[];
    providers?: unknown[];
    appConnectors?: unknown[];
    uiContributions?: unknown[];
    commands?: unknown[];
    lspServers?: unknown[];
    hooks?: unknown[];
    monitors?: unknown[];
    outputStyles?: unknown[];
    settings?: unknown[];
    assets?: unknown[];
    permissions?: unknown[];
    authPolicies?: unknown[];
    marketplaceSources?: unknown[];
    pluginCatalog?: unknown[];
  };
}

interface ExtensionHarnessSummary {
  id: string;
  label: string;
  modelControl: "direct" | "session" | "unsupported";
  modelCompatibility: "declared_apis" | "any";
  supportedModelApis?: string[];
  readOnly?: boolean;
  enabled?: boolean;
  software?: { name?: string; version?: string };
}

interface ExtensionModelSummary {
  id: string;
  label?: string;
  runtimeModel: string;
  apiProtocols: string[];
  readOnly?: boolean;
  enabled?: boolean;
  catalogSource?: string;
  catalogSources?: string[];
}

interface ExtensionModelSupplySummary {
  id: string;
  modelId: string;
  providerProfileId: string;
  runtimeModel?: string;
  providerGroup?: string;
  harnessIds?: string[];
  enabled?: boolean;
  readOnly?: boolean;
}

interface ExtensionProviderSummary {
  id: string;
  label: string;
  kind: string;
  baseUrl?: string;
  authMode?: "api_key" | "auth_token";
  usageAdapter?: "new_api";
  newApiAccountUserId?: string;
  accountAccessReady?: boolean;
  runtimeReady?: boolean;
  runtimeNote?: string;
  catalogAdapter?: string;
  runtimeKeySlots?: Array<{ id: string; label: string; enabled: boolean }>;
  runtimeKeyUsage?: ProviderKeyUsageSummary[];
  readOnly?: boolean;
}

interface ExtensionAgentSummary {
  id: string;
  name: string;
  displayName?: string;
  selector?: string;
  harnessId?: string;
  modelId?: string;
  nativeModel?: string;
  modelSupplyId?: string;
  skills?: string[];
  mcpServers?: string[];
  tools?: string[];
  disallowedTools?: string[];
  permissionMode?: string;
  sandboxMode?: string;
  nicknameCandidates?: string[];
  maxTurns?: number;
  memory?: string;
  effort?: string;
  background?: boolean;
  isolation?: string;
  color?: string;
  readOnly?: boolean;
  instructions?: string;
  definition?: {
    kind: "local" | "project" | "user" | "plugin" | "host" | "server" | "imported";
    host?: "claude_code" | "codex" | "swarmx" | "custom";
    format?: "claude_code" | "codex";
    path?: string;
    label?: string;
    readOnly?: boolean;
  };
  harnessRecipe?: {
    id: string;
    revisionId: string;
    name?: string;
    softwareId: string;
    softwareVersion?: string;
    skillBindings: Array<{
      skillId: string;
      mode: "off" | "auto" | "required";
      variantId?: string;
    }>;
    mcpServerIds: string[];
    projectContext: {
      paths: string[];
      instructionFiles: string[];
      includeWorkspaceRules: boolean;
    };
    delivery: {
      unsupportedSkill: "block" | "skip";
      requireContentDigest: boolean;
      allowHostNativePlugins: boolean;
    };
    permissions: {
      mode: HarnessPermissionMode;
      allowedTools: string[];
      deniedTools: string[];
    };
    contentDigest?: string;
  };
}

interface ExtensionAgentPlanSummary {
  id: string;
  agentId: string;
  agentProfileId?: string;
  displayName: string;
  canonicalSelector: string;
  host: "local" | "server";
  status: "draft" | "ready" | "disabled" | "blocked" | "running" | "failed" | "stale";
  healthStatus: "ready" | "blocked";
  harnessId?: string;
  harnessLabel?: string;
  modelId?: string;
  runtimeModel?: string;
  modelSupplyId?: string;
  supplyLabel?: string;
  pluginIds?: string[];
  skills?: ExtensionAgentPlanCapabilitySummary[];
  mcpServers?: ExtensionAgentPlanCapabilitySummary[];
  context?: { mode: string; strategy: string; memory?: string };
  permissions?: {
    tools?: string;
    mcp?: string;
    shell?: string;
    mode?: string;
    allowedTools?: string[];
    deniedTools?: string[];
    summary?: string;
  };
  visual?: { label?: string; color?: string; icon?: string };
  requirements?: ExtensionAgentPlanRequirementSummary[];
}

interface ExtensionAgentPlanCapabilitySummary {
  id: string;
  name?: string;
  sourcePluginId?: string;
  status?: string;
}

interface ExtensionAgentPlanRequirementSummary {
  kind: string;
  status: string;
  id?: string;
  sourcePluginId?: string;
  message: string;
}

interface ExtensionSkillSummary {
  id: string;
  name?: string;
  path?: string;
  canonicalPath?: string;
  governanceRef?: string;
  requiresGateSkillIds?: string[];
  hostExposures?: ExtensionSkillHostExposureSummary[];
  readOnly?: boolean;
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
}

interface ExtensionSkillHostExposureSummary {
  host: string;
  status?: string;
  manifestPath?: string;
  marketplaceSourceId?: string;
  rulesPath?: string;
  package?: string;
  readOnly?: boolean;
}

interface ExtensionMcpSummary {
  id: string;
  name?: string;
  scope?: string;
}

interface ExtensionConnectorSummary {
  id: string;
  name: string;
  kind: string;
  readOnly?: boolean;
}

interface ExtensionUiContributionSummary {
  id: string;
  kind: string;
  name: string;
  description?: string;
  placement: string;
  order?: number;
  icon?: string;
  route?: string;
  target?: string;
  componentRef?: string;
  assetRef?: string;
  commandId?: string;
  settingIds?: string[];
  permissionIds?: string[];
  authPolicyIds?: string[];
  sourcePluginId?: string;
  readOnly?: boolean;
  provenance?: string;
}

interface ExtensionCommandSummary {
  id: string;
  name?: string;
  command?: string[];
  scope?: string;
}

interface ExtensionLspSummary {
  id: string;
  name?: string;
  description?: string;
  languages?: string[];
  languageIds?: string[];
  mentionPrefixes?: string[];
  command?: string[] | string;
  args?: string[];
  cwd?: string;
  scope?: string;
}

interface ExtensionHookSummary {
  id: string;
  name?: string;
  event: string;
}

interface ExtensionMonitorSummary {
  id: string;
  name?: string;
  trigger?: string;
  schedule?: string;
}

interface ExtensionOutputStyleSummary {
  id: string;
  name?: string;
  path?: string;
}

interface ExtensionSettingSummary {
  id: string;
  name?: string;
  valueType?: string;
  required?: boolean;
}

interface ExtensionAssetSummary {
  id: string;
  name?: string;
  kind?: string;
  path?: string;
  url?: string;
}

interface ExtensionPermissionSummary {
  id: string;
  kind: string;
  access?: string;
  target?: string;
  required?: boolean;
}

interface ExtensionAuthPolicySummary {
  id: string;
  kind?: string;
  required?: boolean;
  secretRefs?: unknown[];
}

interface ExtensionMarketplaceSourceSummary {
  id: string;
  name: string;
  host?: string;
  kind?: string;
  path?: string;
  url?: string;
  package?: string;
  enabled?: boolean;
  readOnly?: boolean;
  trust?: string;
}

interface ExtensionPluginCatalogEntrySummary {
  id: string;
  name: string;
  version?: string;
  marketplaceSourceId?: string;
  bundleId?: string;
  hosts?: string[];
  trust?: string;
  installState?: string;
  updateState?: string;
  providesHarness?: boolean;
  componentCounts?: Record<string, number>;
  readOnly?: boolean;
}

type HarnessRequirementStatus = "ready" | "missing" | "unsupported" | "failed";
type HarnessEnvironmentHarnessState = "ready" | "needs_setup" | "unsupported";
type ContainerRuntimeStatus = "ready" | "missing" | "unsupported" | "service_stopped" | "failed";
type HarnessProtectionMode = "protected" | "native";

interface HarnessRuntimeRequirement {
  id: string;
  label: string;
  command: string;
  status: HarnessRequirementStatus;
  installable: boolean;
  requiredBy: string[];
  path?: string;
  version?: string;
  note?: string;
}

interface HarnessEnvironmentHarness {
  harnessId: string;
  harnessLabel: string;
  command?: string;
  installable?: boolean;
  path?: string;
  version?: string;
  status: HarnessEnvironmentHarnessState;
  requirements: string[];
  executionMode?: HarnessProtectionMode;
  protectionRequired?: boolean;
  containerRuntimeId?: string;
  note?: string;
}

interface HarnessContainerRuntime {
  id: string;
  label: string;
  command: string;
  status: ContainerRuntimeStatus;
  supported: boolean;
  installable: boolean;
  serviceReady: boolean;
  preferred: boolean;
  path?: string;
  version?: string;
  note?: string;
}

interface HarnessProtectionSummary {
  mode: HarnessProtectionMode;
  ready: boolean;
  requiredHarnessIds: string[];
  selectedRuntimeId?: string;
  note?: string;
}

interface HarnessEnvironmentStatus {
  checkedAt: string;
  path: string;
  ready: boolean;
  setupAvailable: boolean;
  containerRuntimes?: HarnessContainerRuntime[];
  protection?: HarnessProtectionSummary;
  requirements: HarnessRuntimeRequirement[];
  harnesses: HarnessEnvironmentHarness[];
}

interface HarnessVersionCheck {
  harnessId: string;
  version?: string;
}

interface DoctorHarnessVersionState {
  status: "loading" | "loaded";
  version?: string;
}

interface HarnessEnvironmentSetupResult {
  success: boolean;
  status: HarnessEnvironmentStatus;
  installedRequirementIds: string[];
  skippedRequirementIds: string[];
  failedRequirementIds: string[];
  installedContainerRuntimeIds?: string[];
  skippedContainerRuntimeIds?: string[];
  failedContainerRuntimeIds?: string[];
  log: string[];
  error?: string;
}

type DoctorRepairRisk = "safe" | "install" | "admin";

interface DoctorIssue {
  id: string;
  severity: "error" | "warning";
  scope: "doctor" | "protection" | "requirement" | "harness";
  targetId?: string;
  message: string;
  repairActionId?: string;
}

interface DoctorRepairAction {
  id: string;
  label: string;
  risk: DoctorRepairRisk;
  request: {
    harnessId?: string;
    requirementIds?: string[];
    containerRuntimeId?: string;
    includeContainerRuntime?: boolean;
  };
}

interface DoctorReport {
  checkedAt: string;
  healthy: boolean;
  harnessId?: string;
  summary: {
    readyHarnesses: number;
    totalHarnesses: number;
    issueCount: number;
    fixableCount: number;
  };
  issues: DoctorIssue[];
  repairActions: DoctorRepairAction[];
  environment: HarnessEnvironmentStatus;
}

interface DoctorFixResult {
  executed: boolean;
  before: DoctorReport;
  plan: {
    actions: DoctorRepairAction[];
    requiresConfirmation: boolean;
    requiresAdmin: boolean;
  };
  setupResults: HarnessEnvironmentSetupResult[];
  after: DoctorReport;
}

type DoctorPanelMode = "doctor" | "setup";

type DesktopSlashCommand =
  | { kind: "doctor"; fix: boolean; harnessId?: string }
  | { kind: "setup"; fix: false; harnessId?: string }
  | { kind: "error"; message: string };

interface AgentCompositionPayload {
  id: string;
  agentProfileId?: string;
  harnessId?: string;
  modelId?: string;
  effort?: string;
  host?: "local" | "server";
}

interface SwarmxAPI {
  readonly initialProjects?: readonly ProjectData[];
  sendMessage(params: {
    requestId: string;
    sessionId?: string;
    harnessId: string;
    userText: string;
    agentComposition?: unknown;
    swarmConfig?: unknown;
    cwd?: string;
  }): Promise<{
    success: boolean;
    messages?: unknown;
    error?: string;
    canceled?: boolean;
    requestId?: string;
    sessionPersisted?: boolean;
  }>;
  onAgentChunk(listener: (event: { requestId: string; chunk: MessageChunk }) => void): () => void;
  onAgentInteraction(listener: (event: AgentInteractionEvent) => void): () => void;
  onSessionMessages?(listener: (event: { sessionId: string }) => void): () => void;
  resolveAgentInteraction(params: {
    requestId: string;
    interactionId: string;
    response: AgentInteractionResponse;
  }): Promise<{ requestId: string; interactionId: string; resolved: boolean }>;
  cancelMessage(requestId: string): Promise<{ requestId: string; canceled: boolean }>;
  createSession(params: {
    agentName: string;
    harness: string;
    model?: string;
    projectId?: string;
    cwd?: string;
    permissionMode?: SessionPermissionMode;
  }): Promise<SessionData>;
  saveSession(session: SessionData): Promise<void>;
  loadSession(id: string): Promise<SessionData | null>;
  loadDiscoveredSession(session: DiscoveredSession): Promise<SessionData | null>;
  listSessions(): Promise<SessionData[]>;
  getActivityProfile(): Promise<ActivityProfileSummary>;
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
  }): Promise<GroupedSessionsResult>;
  deleteSession(id: string): Promise<boolean>;
  renameSession(id: string, title: string): Promise<SessionData>;
  setSessionPinned(id: string, pinned: boolean): Promise<SessionData>;
  generateSessionTitle(id: string, userText: string): Promise<{ title: string; updated: boolean }>;
  appendMessages(params: { id: string; messages: unknown[] }): Promise<boolean>;
  importN8nWorkflow(source: string): Promise<N8nImportResponse>;
  listExtensions(): Promise<ExtensionCapabilityInventory>;
  getExtensionManagementState(): Promise<ExtensionManagementState>;
  saveExtensionSource(input: unknown): Promise<ExtensionManagementState>;
  refreshExtensionSource(id: string): Promise<ExtensionManagementState>;
  removeExtensionSource(id: string): Promise<ExtensionManagementState>;
  applyExtensionAction(input: unknown): Promise<{
    state: ExtensionManagementState;
    receipt: { status: "applied" | "rejected" | "failed"; message: string };
  }>;
  saveSkillEvolutionPolicy(input: {
    enabled: boolean;
    promotionGate: "human" | "policy";
  }): Promise<ExtensionManagementState>;
  listCustomAgents(): Promise<ExtensionCapabilityInventory>;
  saveCustomAgent(input: unknown): Promise<ExtensionCapabilityInventory>;
  removeCustomAgent(id: string): Promise<ExtensionCapabilityInventory>;
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
  workspaceRoot(): Promise<string>;
  getWorkspaceReview(cwd?: string): Promise<WorkspaceReviewSnapshot>;
  listWorkspaceDirectory(path?: string, cwd?: string): Promise<WorkspaceDirectoryListing>;
  readWorkspaceFile(path: string, cwd?: string): Promise<WorkspaceFilePreview>;
  createTerminal(params: {
    id: string;
    cwd: string;
    cols?: number;
    rows?: number;
  }): Promise<{ id: string; pid: number }>;
  writeTerminal(id: string, data: string): Promise<{ written: boolean }>;
  resizeTerminal(id: string, cols: number, rows: number): Promise<{ resized: boolean }>;
  killTerminal(id: string): Promise<{ killed: boolean }>;
  onTerminalData(listener: (event: { id: string; data: string }) => void): () => void;
  onTerminalExit(
    listener: (event: { id: string; exitCode: number; signal?: number }) => void,
  ): () => void;
  createBrowser(params?: {
    id?: string;
    url?: string;
    bounds?: BrowserBounds;
    visible?: boolean;
  }): Promise<BrowserState>;
  navigateBrowser(id: string, url: string): Promise<BrowserState>;
  backBrowser(id: string): Promise<BrowserState>;
  forwardBrowser(id: string): Promise<BrowserState>;
  reloadBrowser(id: string): Promise<BrowserState>;
  setBrowserBounds(id: string, bounds: BrowserBounds): Promise<{ updated: boolean }>;
  setBrowserVisible(id: string, visible: boolean): Promise<{ updated: boolean }>;
  destroyBrowser(id: string): Promise<{ destroyed: boolean }>;
  onBrowserState(listener: (state: BrowserState) => void): () => void;
  getUpdateState?(): Promise<DesktopUpdateState>;
  startUpdate?(): Promise<DesktopUpdateState>;
  onUpdateState?(listener: (state: DesktopUpdateState) => void): () => void;
  selectFilesAndFolders(): Promise<string[]>;
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

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "secondary" | "ghost" | "destructive";
  size?: "sm" | "md" | "icon";
}

export interface SwarmxDesktopProductConfig {
  name?: string;
  subtitle?: string;
}

export interface GuiContributionHostProps {
  contribution: ExtensionUiContributionSummary;
  inventory?: ExtensionCapabilityInventory;
  onSelectAgent: (agentId: string) => void;
}

export type GuiContributionComponent = React.ComponentType<GuiContributionHostProps>;
export type GuiContributionComponentRegistry = Record<string, GuiContributionComponent | undefined>;

export interface AppProps {
  product?: SwarmxDesktopProductConfig;
  uiComponentRegistry?: GuiContributionComponentRegistry;
}

declare global {
  interface Window {
    swarmxAPI: SwarmxAPI;
  }
}

const api = window.swarmxAPI;
const LOCAL_SESSIONS_KEY = "sessions:local";
const GROUPED_SESSIONS_KEY = "sessions:grouped";
const ACTIVITY_PROFILE_KEY = "activity:profile";
const PROJECTS_KEY = "projects:local";
const EXTENSIONS_KEY = "extensions:inventory";
const EXTENSION_MANAGEMENT_KEY = "extensions:management";
const HARNESS_ENVIRONMENT_KEY = "harness:environment";
const SESSION_DEDUPING_INTERVAL_MS = 10_000;
const LOCAL_SESSION_PRELOAD_LIMIT = 24;
const PANEL_EXIT_MS = 240;
const LOCAL_FILES_LSP_ID = "swarmx.local-files";
const SKILLS_LSP_ID = "swarmx.skills";
const DEFAULT_MENTION_SERVERS = [
  {
    id: LOCAL_FILES_LSP_ID,
    name: "Files and folders",
    description: "Workspace-local file and folder references.",
    mentionPrefixes: ["@"],
  },
  {
    id: SKILLS_LSP_ID,
    name: "Skills",
    description: "Skills available from installed extensions.",
    mentionPrefixes: ["$"],
  },
];

const HARNESSES: HarnessOption[] = [
  { id: "swarmx", label: "SwarmX", icon: Workflow, modelControl: "direct" },
  { id: "claude_code", label: "Claude Code", icon: Hammer, modelControl: "session" },
  { id: "codex", label: "Codex", icon: TerminalIcon, modelControl: "session" },
  { id: "pi", label: "Pi", icon: Bot, modelControl: "session" },
  { id: "opencode", label: "OpenCode", icon: Code2, modelControl: "session" },
  { id: "hermes", label: "Hermes", icon: Sparkles, modelControl: "session" },
  {
    id: "openclaw",
    label: "OpenClaw",
    icon: Bot,
    modelControl: "unsupported",
    disabled: true,
    disabledReason: "Model switching is not configured.",
  },
];
const EMPTY_RUN_SUGGESTIONS: Array<{
  id: string;
  label: string;
  prompt: string;
  icon: LucideIcon;
  tone: "blue" | "violet" | "green" | "orange";
}> = [
  {
    id: "explore",
    label: "Explore and understand code",
    prompt: "Explore this codebase and explain its architecture, main flows, and important risks.",
    icon: Telescope,
    tone: "blue",
  },
  {
    id: "build",
    label: "Build a new feature, app, or tool",
    prompt: "Help me design and build a new feature in this project.",
    icon: Hammer,
    tone: "violet",
  },
  {
    id: "review",
    label: "Review code and suggest changes",
    prompt: "Review the current changes and suggest focused improvements.",
    icon: RefreshCw,
    tone: "green",
  },
  {
    id: "fix",
    label: "Fix issues and failures",
    prompt: "Investigate the current issues or failing tests and fix the root cause.",
    icon: Bug,
    tone: "orange",
  },
];
const CODEX_ACP_VERSION = "1.1.2";
const CLAUDE_AGENT_ACP_VERSION = "0.58.1";
const DEFAULT_HARNESS_MCPS = [{ name: "filesystem", transport: "stdio", scope: "project" }];
const DEFAULT_HARNESS_SKILLS = ["test-driven-development", "backprop"];
const DEFAULT_PROJECT_FILES = ["AGENTS.md", "CLAUDE.md"];
const CODEX_ACP_ARGS = ["--yes", `@agentclientprotocol/codex-acp@${CODEX_ACP_VERSION}`];
const CLAUDE_CODE_ACP_ARGS = [
  "--yes",
  `@agentclientprotocol/claude-agent-acp@${CLAUDE_AGENT_ACP_VERSION}`,
];
const DEFAULT_PRODUCT_CONFIG: Required<Pick<SwarmxDesktopProductConfig, "name">> = {
  name: "SwarmX",
};

const CODEX_ACP_BACKEND: AgentBackend = {
  type: "custom",
  program: "npx",
  args: CODEX_ACP_ARGS,
};

const CLAUDE_CODE_ACP_BACKEND: AgentBackend = {
  type: "custom",
  program: "npx",
  args: CLAUDE_CODE_ACP_ARGS,
};

function codexHarness(): HarnessDescriptor {
  return {
    software: {
      name: "codex-acp",
      version: CODEX_ACP_VERSION,
      runner: "npx",
      command: CODEX_ACP_ARGS,
    },
    mcps: DEFAULT_HARNESS_MCPS,
    skills: DEFAULT_HARNESS_SKILLS,
    projectFiles: DEFAULT_PROJECT_FILES,
  };
}

function claudeCodeHarness(): HarnessDescriptor {
  return {
    software: {
      name: "claude-agent-acp",
      version: CLAUDE_AGENT_ACP_VERSION,
      runner: "npx",
      command: CLAUDE_CODE_ACP_ARGS,
    },
    mcps: DEFAULT_HARNESS_MCPS,
    skills: DEFAULT_HARNESS_SKILLS,
    projectFiles: DEFAULT_PROJECT_FILES,
  };
}

const DEFAULT_WORKFLOW_CONFIG: SwarmConfig = {
  name: "research_review",
  description: "Route a request through ACP agents using each harness's negotiated default model.",
  root: "triage_agent",
  nodes: {
    triage_agent: {
      kind: "agent",
      agent: {
        name: "triage_agent",
        description: "Codex ACP agent for classification and planning.",
        backend: CODEX_ACP_BACKEND,
        parameters: { harness: codexHarness() },
        instructions: "Identify the user's goal, constraints, and required evidence.",
      },
    },
    researcher_agent: {
      kind: "agent",
      agent: {
        name: "researcher_agent",
        description: "Claude Code ACP agent for repository research.",
        backend: CLAUDE_CODE_ACP_BACKEND,
        parameters: { harness: claudeCodeHarness() },
        instructions: "Inspect the repository and collect evidence for the plan.",
      },
    },
    writer_agent: {
      kind: "agent",
      agent: {
        name: "writer_agent",
        description: "Codex ACP agent for implementation-quality synthesis.",
        backend: CODEX_ACP_BACKEND,
        parameters: { harness: codexHarness() },
        instructions: "Write a concise answer using the research output.",
      },
    },
  },
  edges: [
    { source: "triage_agent", target: "researcher_agent" },
    { source: "researcher_agent", target: "writer_agent" },
  ],
};

const DEFAULT_WORKFLOW_JSON = JSON.stringify(DEFAULT_WORKFLOW_CONFIG, null, 2);

function parseDesktopSlashCommand(value: string): DesktopSlashCommand | null {
  const tokens = value.trim().split(/\s+/);
  const command = tokens.shift();
  if (command !== "/doctor" && command !== "/setup") return null;

  const kind = command === "/doctor" ? "doctor" : "setup";
  let fix = false;
  let harnessId: string | undefined;
  while (tokens.length > 0) {
    const token = tokens.shift();
    if (!token) continue;
    if (token === "--fix") {
      if (kind === "setup") {
        return { kind: "error", message: "Use /setup without --fix, then confirm repairs." };
      }
      fix = true;
      continue;
    }
    if (token === "--harness") {
      const value = tokens.shift();
      if (!value || value.startsWith("-")) {
        return { kind: "error", message: "--harness requires a harness id." };
      }
      if (harnessId) {
        return { kind: "error", message: "Specify only one harness id." };
      }
      harnessId = value;
      continue;
    }
    if (token.startsWith("--harness=")) {
      const value = token.slice("--harness=".length);
      if (!value) return { kind: "error", message: "--harness requires a harness id." };
      if (harnessId) {
        return { kind: "error", message: "Specify only one harness id." };
      }
      harnessId = value;
      continue;
    }
    if (token.startsWith("-")) {
      return { kind: "error", message: `Unknown ${kind} option: ${token}` };
    }
    if (harnessId) {
      return { kind: "error", message: "Specify only one harness id." };
    }
    harnessId = token;
  }

  return kind === "doctor"
    ? { kind, fix, ...(harnessId ? { harnessId } : {}) }
    : { kind, fix: false, ...(harnessId ? { harnessId } : {}) };
}

export function createSwarmxDesktopApp(appProps: AppProps = {}): React.ComponentType {
  function SwarmxDesktopApp() {
    return <App {...appProps} />;
  }
  return SwarmxDesktopApp;
}

function usePanelPresence(open: boolean): boolean {
  const [retained, setRetained] = useState(open);

  useEffect(() => {
    if (open) {
      setRetained(true);
      return;
    }
    const timeout = window.setTimeout(() => setRetained(false), PANEL_EXIT_MS);
    return () => window.clearTimeout(timeout);
  }, [open]);

  return open || retained;
}

export function App({ product, uiComponentRegistry = {} }: AppProps = {}) {
  const isMacOS =
    typeof navigator !== "undefined" && /Macintosh|Mac OS X/.test(navigator.userAgent);
  const productConfig = {
    ...DEFAULT_PRODUCT_CONFIG,
    ...product,
  };
  const sessionGroupMode: SessionGroupMode = "project";
  const [currentSession, setCurrentSession] = useState<SessionData | null>(null);
  const [selectedDiscoveredSession, setSelectedDiscoveredSession] =
    useState<DiscoveredSession | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [runState, setRunState] = useState<"idle" | "running" | "stopping">("idle");
  const [composerError, setComposerError] = useState<string | null>(null);
  const activeRequestId = useRef<string | null>(null);
  const [agentInteractions, setAgentInteractions] = useState<AgentInteractionEvent[]>([]);
  const [resolvingInteractionId, setResolvingInteractionId] = useState<string | null>(null);
  const [agentInteractionError, setAgentInteractionError] = useState<string | null>(null);
  const requestDispatched = useRef(false);
  const stopRequested = useRef(false);
  const [selectedHarness, setSelectedHarness] = useState("swarmx");
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedEffort, setSelectedEffort] = useState<string | null>(null);
  const [agentPickerOpen, setAgentPickerOpen] = useState(false);
  const [permissionPickerOpen, setPermissionPickerOpen] = useState(false);
  const [newSessionPermissionMode, setNewSessionPermissionMode] =
    useState<SessionPermissionMode>("inherit");
  const [agentPickerSection, setAgentPickerSection] = useState<"harness" | "model" | "effort">(
    "harness",
  );
  const [modelCatalogRefreshing, setModelCatalogRefreshing] = useState(false);
  const [modelCatalogError, setModelCatalogError] = useState<string | null>(null);
  const [providerUsage, setProviderUsage] = useState<ProviderUsageSnapshot | null>(null);
  const [providerUsageRefreshing, setProviderUsageRefreshing] = useState(false);
  const [providerUsageRefreshingIds, setProviderUsageRefreshingIds] = useState<ReadonlySet<string>>(
    new Set(),
  );
  const [providerUsageError, setProviderUsageError] = useState<string | null>(null);
  const providerUsageRefreshStarted = useRef(false);
  const [selectedExtensionAgentId, setSelectedExtensionAgentId] = useState<string | null>(null);
  const [activeUiContributionId, setActiveUiContributionId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sidebarSearchOpen, setSidebarSearchOpen] = useState(false);
  const [sidebarQuery, setSidebarQuery] = useState("");
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [projectHeaderMenu, setProjectHeaderMenu] = useState<"organize" | "add" | null>(null);
  const [projectActionMenuId, setProjectActionMenuId] = useState<string | null>(null);
  const [projectPreview, setProjectPreview] = useState<ProjectPreviewState | null>(null);
  const [projectOrganizationMode, setProjectOrganizationMode] =
    useState<ProjectOrganizationMode>("project");
  const [projectSortMode, setProjectSortMode] = useState<ProjectSortMode>("priority");
  const [projectActionPending, setProjectActionPending] = useState(false);
  const [projectError, setProjectError] = useState<string | null>(null);
  const [renamingProjectId, setRenamingProjectId] = useState<string | null>(null);
  const [projectRenameDraft, setProjectRenameDraft] = useState("");
  const [sessionContextMenu, setSessionContextMenu] = useState<SessionContextMenuState | null>(
    null,
  );
  const [renamingSession, setRenamingSession] = useState<DiscoveredSession | null>(null);
  const [sessionRenameDraft, setSessionRenameDraft] = useState("");
  const [sessionActionPending, setSessionActionPending] = useState(false);
  const [sessionActionError, setSessionActionError] = useState<string | null>(null);
  const [settingsQuery, setSettingsQuery] = useState("");
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const [settingsSection, setSettingsSection] = useState<SettingsSection | null>(null);
  const [desktopUpdate, setDesktopUpdate] = useState<DesktopUpdateState>({
    phase: "hidden",
    currentVersion: "unknown",
  });
  const [pinnedSummaryOpen, setPinnedSummaryOpen] = useState(false);
  const [bottomPanelOpen, setBottomPanelOpen] = useState(false);
  const [rightPanelOpen, setRightPanelOpen] = useState(false);
  const [workflowPanelOpen, setWorkflowPanelOpen] = useState(false);
  const [doctorPanelOpen, setDoctorPanelOpen] = useState(false);
  const [doctorPanelMode, setDoctorPanelMode] = useState<DoctorPanelMode>("doctor");
  const [doctorHarnessId, setDoctorHarnessId] = useState<string | null>(null);
  const [doctorReport, setDoctorReport] = useState<DoctorReport | null>(null);
  const [doctorLoading, setDoctorLoading] = useState(false);
  const [doctorHarnessVersions, setDoctorHarnessVersions] = useState<
    Record<string, DoctorHarnessVersionState>
  >({});
  const [doctorFixPending, setDoctorFixPending] = useState(false);
  const [doctorFixRunning, setDoctorFixRunning] = useState(false);
  const [doctorFixResult, setDoctorFixResult] = useState<DoctorFixResult | null>(null);
  const [doctorInstallingHarnessId, setDoctorInstallingHarnessId] = useState<string | null>(null);
  const [doctorError, setDoctorError] = useState<string | null>(null);
  const [workflowEnabled, setWorkflowEnabled] = useState(false);
  const [workflowJson, setWorkflowJson] = useState(DEFAULT_WORKFLOW_JSON);
  const [workflowImportStatus, setWorkflowImportStatus] = useState<WorkflowImportStatus | null>(
    null,
  );
  const activeRightPanelKind = doctorPanelOpen ? "doctor" : rightPanelOpen ? "tools" : null;
  const [renderedRightPanelKind, setRenderedRightPanelKind] = useState<"doctor" | "tools">(
    activeRightPanelKind ?? "tools",
  );
  const displayedRightPanelKind = activeRightPanelKind ?? renderedRightPanelKind;
  const pinnedSummaryMounted = usePanelPresence(pinnedSummaryOpen);
  const rightPanelMounted = usePanelPresence(activeRightPanelKind !== null);
  const chatRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const sidebarSearchRef = useRef<HTMLInputElement>(null);
  const projectHeaderMenuRef = useRef<HTMLDivElement>(null);
  const projectActionMenuRef = useRef<HTMLDivElement>(null);
  const projectRenameInputRef = useRef<HTMLInputElement>(null);
  const sessionContextMenuRef = useRef<HTMLDivElement>(null);
  const sessionRenameInputRef = useRef<HTMLInputElement>(null);
  const projectPreviewCloseTimerRef = useRef<number | null>(null);
  const sidebarAccountRef = useRef<HTMLDivElement>(null);
  const navigationHistoryRef = useRef<Array<DiscoveredSession | null>>([null]);
  const navigationIndexRef = useRef(0);
  const doctorVersionChecksStarted = useRef(false);
  const [navigationIndex, setNavigationIndex] = useState(0);
  const preloadedSessionKeys = useRef(new Set<string>());
  const scrollStateRef = useRef<{ sessionId: string | null; messageCount: number }>({
    sessionId: null,
    messageCount: 0,
  });
  const { mutate: mutateSessionDetail } = useSWRConfig();
  const messageCount = currentSession?.messages.length ?? 0;
  const emptyRun = !currentSession || messageCount === 0;
  const acpHistoryReadOnly = Boolean(currentSession?.acpSessionId);

  useEffect(
    () =>
      api.onAgentInteraction((interaction) => {
        if (activeRequestId.current !== interaction.requestId) return;
        setAgentInteractionError(null);
        setAgentInteractions((current) =>
          current.some((candidate) => candidate.interactionId === interaction.interactionId)
            ? current
            : [...current, interaction],
        );
      }),
    [],
  );

  useEffect(() => {
    if (!sidebarSearchOpen) return;
    window.requestAnimationFrame(() => sidebarSearchRef.current?.focus());
  }, [sidebarSearchOpen]);

  useEffect(() => {
    if (!accountMenuOpen) return;
    const closeOnPointer = (event: PointerEvent) => {
      if (!sidebarAccountRef.current?.contains(event.target as Node)) setAccountMenuOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setAccountMenuOpen(false);
    };
    window.addEventListener("pointerdown", closeOnPointer);
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("pointerdown", closeOnPointer);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [accountMenuOpen]);

  useEffect(() => {
    if (!projectHeaderMenu && !projectActionMenuId) return;
    const closeOnPointer = (event: PointerEvent) => {
      const target = event.target as Node;
      if (projectHeaderMenu && !projectHeaderMenuRef.current?.contains(target)) {
        setProjectHeaderMenu(null);
      }
      if (projectActionMenuId && !projectActionMenuRef.current?.contains(target)) {
        setProjectActionMenuId(null);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setProjectHeaderMenu(null);
      setProjectActionMenuId(null);
    };
    window.addEventListener("pointerdown", closeOnPointer);
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("pointerdown", closeOnPointer);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [projectActionMenuId, projectHeaderMenu]);

  useEffect(
    () => () => {
      if (projectPreviewCloseTimerRef.current !== null) {
        window.clearTimeout(projectPreviewCloseTimerRef.current);
      }
    },
    [],
  );

  useEffect(() => {
    if (!renamingProjectId) return;
    window.requestAnimationFrame(() => {
      projectRenameInputRef.current?.focus();
      projectRenameInputRef.current?.select();
    });
  }, [renamingProjectId]);

  useEffect(() => {
    if (!sessionContextMenu) return;
    const closeOnPointer = (event: PointerEvent) => {
      if (!sessionContextMenuRef.current?.contains(event.target as Node)) {
        setSessionContextMenu(null);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSessionContextMenu(null);
    };
    window.addEventListener("pointerdown", closeOnPointer);
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("pointerdown", closeOnPointer);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [sessionContextMenu]);

  useEffect(() => {
    if (!renamingSession) return;
    window.requestAnimationFrame(() => {
      sessionRenameInputRef.current?.focus();
      sessionRenameInputRef.current?.select();
    });
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setRenamingSession(null);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [renamingSession]);

  useEffect(() => {
    let mounted = true;
    const acceptUpdateState = (state: DesktopUpdateState) => {
      if (mounted) setDesktopUpdate(state);
    };
    const unsubscribe = api.onUpdateState?.(acceptUpdateState) ?? (() => undefined);
    void api
      .getUpdateState?.()
      .then(acceptUpdateState)
      .catch(() => undefined);
    return () => {
      mounted = false;
      unsubscribe();
    };
  }, []);

  useEffect(() => {
    if (activeRightPanelKind) setRenderedRightPanelKind(activeRightPanelKind);
  }, [activeRightPanelKind]);

  const {
    data: sessions = [],
    error: localSessionsError,
    isLoading: localSessionsLoading,
    mutate: mutateLocalSessions,
  } = useSWR<SessionData[]>(
    LOCAL_SESSIONS_KEY,
    () => api.listSessions() as Promise<SessionData[]>,
    {
      dedupingInterval: SESSION_DEDUPING_INTERVAL_MS,
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );

  const {
    data: groupedSessions,
    error: groupedSessionsError,
    isLoading: groupedSessionsLoading,
    mutate: mutateGroupedSessions,
  } = useSWR<GroupedSessionsResult>(
    GROUPED_SESSIONS_KEY,
    () => api.listGroupedSessions({ mode: sessionGroupMode }) as Promise<GroupedSessionsResult>,
    {
      dedupingInterval: SESSION_DEDUPING_INTERVAL_MS,
      keepPreviousData: true,
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );
  const {
    data: projects = [],
    error: projectsError,
    isLoading: projectsLoading,
    mutate: mutateProjects,
  } = useSWR<ProjectData[]>(PROJECTS_KEY, () => api.listProjects(), {
    fallbackData: api.initialProjects ? [...api.initialProjects] : undefined,
    dedupingInterval: SESSION_DEDUPING_INTERVAL_MS,
    revalidateOnMount: api.initialProjects === undefined,
    revalidateOnFocus: false,
    revalidateOnReconnect: false,
  });

  useEffect(() => {
    if (!api.onSessionMessages) return;
    return api.onSessionMessages(({ sessionId }) => {
      void api
        .loadSession(sessionId)
        .then((persisted) => {
          if (!persisted) return;
          setCurrentSession((current) => (current?.id === sessionId ? persisted : current));
          void Promise.all([mutateLocalSessions(), mutateGroupedSessions()]);
        })
        .catch(() => undefined);
    });
  }, [mutateGroupedSessions, mutateLocalSessions]);
  const {
    data: activityProfile,
    error: activityProfileError,
    isLoading: activityProfileLoading,
  } = useSWR<ActivityProfileSummary>(
    settingsSection === "profile" ? ACTIVITY_PROFILE_KEY : null,
    () => api.getActivityProfile(),
    { revalidateOnFocus: true, revalidateOnReconnect: false },
  );
  const {
    data: extensionInventory,
    error: extensionInventoryError,
    isLoading: extensionInventoryLoading,
    mutate: mutateExtensionInventory,
  } = useSWR<ExtensionCapabilityInventory>(
    EXTENSIONS_KEY,
    () => api.listExtensions() as Promise<ExtensionCapabilityInventory>,
    {
      dedupingInterval: SESSION_DEDUPING_INTERVAL_MS,
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );
  const {
    data: extensionManagement,
    error: extensionManagementError,
    mutate: mutateExtensionManagement,
  } = useSWR<ExtensionManagementState>(
    EXTENSION_MANAGEMENT_KEY,
    () => api.getExtensionManagementState(),
    { revalidateOnFocus: false, revalidateOnReconnect: false },
  );
  const { data: desktopWorkspaceRoot } = useSWR<string>(
    "workspace:root",
    () => api.workspaceRoot(),
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );
  useEffect(() => {
    if (projects.length === 0) return;
    if (activeProjectId && projects.some((project) => project.id === activeProjectId)) return;
    const sessionProjectId = selectedDiscoveredSession?.projectId;
    const sessionCwd = selectedDiscoveredSession?.cwd;
    const initialProject =
      projects.find((project) => project.id === sessionProjectId) ??
      projects.find((project) => sameProjectPath(project.cwd, sessionCwd)) ??
      projects.find((project) => sameProjectPath(project.cwd, desktopWorkspaceRoot)) ??
      projects[0];
    setActiveProjectId(initialProject?.id ?? null);
  }, [activeProjectId, desktopWorkspaceRoot, projects, selectedDiscoveredSession]);
  const refreshProviderUsage = useCallback(async (target?: ProviderUsageTarget) => {
    const targetKey = target ? providerUsageTargetKey(target) : undefined;
    if (targetKey) {
      setProviderUsageRefreshingIds((current) => new Set([...current, targetKey]));
    } else {
      setProviderUsageRefreshing(true);
    }
    setProviderUsageError(null);
    try {
      const next = await api.refreshProviderUsage(target);
      setProviderUsage((current) => (target ? mergeProviderUsageSnapshot(current, next) : next));
    } catch (error) {
      setProviderUsageError(errorMessage(error));
    } finally {
      if (targetKey) {
        setProviderUsageRefreshingIds((current) => {
          const next = new Set(current);
          next.delete(targetKey);
          return next;
        });
      } else {
        setProviderUsageRefreshing(false);
      }
    }
  }, []);
  const refreshModelCatalog = useCallback(async () => {
    setModelCatalogRefreshing(true);
    setModelCatalogError(null);
    try {
      const inventory = await api.refreshModelCatalog();
      if (inventory) await mutateExtensionInventory(inventory, false);
    } catch (error) {
      setModelCatalogError(errorMessage(error));
    } finally {
      setModelCatalogRefreshing(false);
    }
  }, [mutateExtensionInventory]);
  const addManualModel = useCallback(
    async (input: ManualModelInput) => {
      setModelCatalogError(null);
      try {
        const inventory = await api.addManualModel(input);
        if (inventory) await mutateExtensionInventory(inventory, false);
        setSelectedModelId(`${selectedHarness}:${input.id.trim()}`);
      } catch (error) {
        const message = errorMessage(error);
        setModelCatalogError(message);
        throw new Error(message);
      }
    },
    [mutateExtensionInventory, selectedHarness],
  );
  const removeManualModel = useCallback(
    async (modelId: string) => {
      setModelCatalogError(null);
      try {
        const inventory = await api.removeManualModel(modelId);
        if (inventory) await mutateExtensionInventory(inventory, false);
        if (selectedModelId === `${selectedHarness}:${modelId}`) {
          setSelectedModelId(null);
          setSelectedEffort(null);
        }
      } catch (error) {
        const message = errorMessage(error);
        setModelCatalogError(message);
        throw new Error(message);
      }
    },
    [mutateExtensionInventory, selectedHarness, selectedModelId],
  );
  const saveProvider = useCallback(
    async (input: UserProviderInput) => {
      setModelCatalogError(null);
      try {
        const inventory = await api.saveProvider(input);
        if (inventory) await mutateExtensionInventory(inventory, false);
        setProviderUsage(null);
        await refreshProviderUsage();
      } catch (error) {
        const message = errorMessage(error);
        setModelCatalogError(message);
        throw new Error(message);
      }
    },
    [mutateExtensionInventory, refreshProviderUsage],
  );
  const removeProvider = useCallback(
    async (providerId: string) => {
      setModelCatalogError(null);
      try {
        const inventory = await api.removeProvider(providerId);
        if (inventory) await mutateExtensionInventory(inventory, false);
        setProviderUsage(null);
        await refreshProviderUsage();
      } catch (error) {
        const message = errorMessage(error);
        setModelCatalogError(message);
        throw new Error(message);
      }
    },
    [mutateExtensionInventory, refreshProviderUsage],
  );
  const resetProviderKey = useCallback(
    async (providerId: string, keyId: string) => {
      setModelCatalogError(null);
      try {
        const inventory = await api.resetProviderKey(providerId, keyId);
        if (inventory) await mutateExtensionInventory(inventory, false);
        await refreshProviderUsage({ source: "provider", sourceId: providerId });
      } catch (error) {
        const message = errorMessage(error);
        setModelCatalogError(message);
        throw new Error(message);
      }
    },
    [mutateExtensionInventory, refreshProviderUsage],
  );
  useEffect(() => {
    if (settingsSection !== "providers" || providerUsageRefreshStarted.current) return;
    providerUsageRefreshStarted.current = true;
    void refreshProviderUsage();
  }, [settingsSection, refreshProviderUsage]);
  const {
    data: harnessEnvironment,
    error: harnessEnvironmentError,
    isLoading: harnessEnvironmentLoading,
    mutate: mutateHarnessEnvironment,
  } = useSWR<HarnessEnvironmentStatus>(
    HARNESS_ENVIRONMENT_KEY,
    () => api.getHarnessEnvironment() as Promise<HarnessEnvironmentStatus>,
    {
      dedupingInterval: SESSION_DEDUPING_INTERVAL_MS,
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );

  const availableHarnesses = useMemo<HarnessOption[]>(() => {
    if (!extensionInventory?.harnesses.length) return HARNESSES;
    return extensionInventory.harnesses.map((harness) => {
      const disabled = harness.enabled === false || harness.modelControl === "unsupported";
      return {
        id: harness.id,
        label: harness.label,
        icon: HARNESSES.find((option) => option.id === harness.id)?.icon ?? Bot,
        modelControl: harness.modelControl,
        disabled,
        ...(disabled
          ? {
              disabledReason:
                harness.enabled === false
                  ? "Disabled by configuration."
                  : "Model switching is not configured.",
            }
          : {}),
      };
    });
  }, [extensionInventory?.harnesses]);
  const activeHarness = useMemo(
    () =>
      availableHarnesses.find(
        (harness) => harness.id === selectedHarness && harness.disabled !== true,
      ) ??
      availableHarnesses.find((harness) => harness.disabled !== true) ??
      HARNESSES.find((harness) => harness.disabled !== true) ??
      HARNESSES[0],
    [availableHarnesses, selectedHarness],
  );
  useEffect(() => {
    const selected = availableHarnesses.find((harness) => harness.id === selectedHarness);
    if (!selected?.disabled) return;
    const fallback = availableHarnesses.find((harness) => !harness.disabled);
    if (!fallback) return;
    setSelectedHarness(fallback.id);
    setSelectedExtensionAgentId(null);
    setSelectedModelId(null);
    setSelectedEffort(null);
  }, [availableHarnesses, selectedHarness]);
  const modelResolution = useMemo<{
    models: ComposerModelOption[];
    error: string | null;
  }>(() => {
    if (!extensionInventory) return { models: [], error: null };
    try {
      return {
        models: resolveComposerModelOptions(extensionInventory, selectedHarness),
        error: null,
      };
    } catch (error) {
      return { models: [], error: errorMessage(error) };
    }
  }, [extensionInventory, selectedHarness]);
  const availableModels = modelResolution.models;
  const modelStatusText =
    activeHarness.modelControl === "unsupported"
      ? "Model switching unsupported"
      : modelResolution.error
        ? "Inventory invalid"
        : "No compatible models";
  const modelUnavailableDiagnostic =
    modelResolution.error ??
    (activeHarness.modelControl === "unsupported"
      ? `Harness "${activeHarness.label}" does not expose request-scoped model selection.`
      : "Register a compatible standalone Model before sending.");
  const selectedModel = useMemo(
    () => availableModels.find((model) => model.id === selectedModelId) ?? null,
    [availableModels, selectedModelId],
  );

  useEffect(() => {
    if (selectedModelId && availableModels.some((model) => model.id === selectedModelId)) return;
    setSelectedModelId(availableModels[0]?.id ?? null);
    setSelectedEffort(null);
  }, [availableModels, selectedModelId]);

  const availableEfforts = selectedModel?.reasoning?.supportedEfforts ?? [];
  useEffect(() => {
    if (availableEfforts.length === 0) {
      setSelectedEffort(null);
      return;
    }
    if (selectedEffort && availableEfforts.includes(selectedEffort)) return;
    setSelectedEffort(selectedModel?.reasoning?.defaultEffort ?? availableEfforts[0] ?? null);
  }, [availableEfforts, selectedEffort, selectedModel?.reasoning?.defaultEffort]);
  const selectedHarnessEnvironment = useMemo(
    () =>
      harnessEnvironment?.harnesses.find((harness) => harness.harnessId === selectedHarness) ??
      null,
    [harnessEnvironment?.harnesses, selectedHarness],
  );
  const extensionAgents = extensionInventory?.agents ?? [];
  const selectedExtensionAgent = useMemo(
    () => extensionAgents.find((agent) => agent.id === selectedExtensionAgentId) ?? null,
    [extensionAgents, selectedExtensionAgentId],
  );
  const registeredUiContributions = useMemo(
    () =>
      (extensionInventory?.uiContributions ?? [])
        .filter((contribution) => {
          if (!contribution.componentRef) return false;
          return Boolean(uiComponentRegistry[contribution.componentRef]);
        })
        .sort((left, right) => {
          const orderDelta = (left.order ?? 0) - (right.order ?? 0);
          return orderDelta || left.name.localeCompare(right.name);
        }),
    [extensionInventory?.uiContributions, uiComponentRegistry],
  );
  const activeUiContribution =
    registeredUiContributions.find((contribution) => contribution.id === activeUiContributionId) ??
    null;
  const ActiveUiContributionComponent = activeUiContribution?.componentRef
    ? uiComponentRegistry[activeUiContribution.componentRef]
    : undefined;
  const workflowState = useMemo(() => parseWorkflowJson(workflowJson), [workflowJson]);
  const activeWorkflowConfig =
    workflowEnabled && workflowState.config && !workflowState.error ? workflowState.config : null;
  const activeWorkflowInvalid = workflowEnabled && !activeWorkflowConfig;
  const activeExtensionAgent = activeWorkflowConfig ? null : selectedExtensionAgent;
  const displayHarness = useMemo<HarnessOption>(() => {
    if (!activeExtensionAgent?.harnessId) return activeHarness;
    return (
      availableHarnesses.find((harness) => harness.id === activeExtensionAgent.harnessId) ?? {
        id: activeExtensionAgent.harnessId,
        label: activeExtensionAgent.harnessId,
        icon: Bot,
        modelControl: "session",
      }
    );
  }, [activeExtensionAgent?.harnessId, activeHarness, availableHarnesses]);
  const agentPickerLabel =
    activeWorkflowConfig?.name ??
    activeExtensionAgent?.name ??
    selectedModel?.label ??
    displayHarness.label;
  const activeRunHarnessId = activeExtensionAgent?.harnessId ?? selectedHarness;
  const sessionPermissionSupported = Boolean(
    !activeWorkflowConfig &&
      !acpHistoryReadOnly &&
      (activeExtensionAgent
        ? activeExtensionAgent.harnessRecipe?.softwareId === "swarmx" ||
          activeExtensionAgent.harnessId === "swarmx"
        : selectedHarness === "swarmx"),
  );
  const sessionPermissionMode = currentSession?.permissionMode ?? newSessionPermissionMode;
  const activeRunHarnessEnvironment = useMemo(
    () =>
      harnessEnvironment?.harnesses.find((harness) => harness.harnessId === activeRunHarnessId) ??
      null,
    [activeRunHarnessId, harnessEnvironment?.harnesses],
  );
  const protectedModeNeedsSetup =
    harnessEnvironment?.protection?.mode === "protected" && !harnessEnvironment.protection.ready;
  const selectedHarnessNeedsSetup =
    Boolean(
      activeWorkflowConfig &&
        protectedModeNeedsSetup &&
        workflowUsesProtectedHarness(activeWorkflowConfig),
    ) ||
    (!activeWorkflowConfig &&
      activeRunHarnessEnvironment !== null &&
      activeRunHarnessEnvironment.status !== "ready");
  const manualCompositionNeedsModel = Boolean(
    !activeWorkflowConfig && !activeExtensionAgent && !selectedModel,
  );
  const sessionGroups = groupedSessions?.groups ?? [];
  const sessionErrors = useMemo(() => {
    const errors = buildSessionErrors(
      groupedSessions?.errors ?? [],
      localSessionsError,
      groupedSessionsError,
    );
    if (projectsError) {
      errors.push({
        harnessId: "local-projects",
        harnessLabel: "Projects",
        message: errorMessage(projectsError),
      });
    }
    return errors;
  }, [groupedSessions?.errors, localSessionsError, groupedSessionsError, projectsError]);
  const sessionsLoading =
    api.initialProjects === undefined &&
    (projectsLoading ||
      (localSessionsLoading && sessions.length === 0) ||
      (groupedSessionsLoading && !groupedSessions));
  const rawDisplayGroups = useMemo(
    () => mergeLocalSessionsIntoGroups(sessionGroups, sessions, sessionGroupMode),
    [sessionGroups, sessions],
  );
  const displayGroups = useMemo(
    () => mergeProjectsIntoSessionGroups(projects, rawDisplayGroups),
    [projects, rawDisplayGroups],
  );
  const orderedDisplayGroups = useMemo(
    () => sortProjectSessionGroups(displayGroups, projectSortMode),
    [displayGroups, projectSortMode],
  );
  const visibleDisplayGroups = useMemo(
    () => filterSessionGroups(orderedDisplayGroups, sidebarQuery),
    [orderedDisplayGroups, sidebarQuery],
  );
  const visibleFlatSessions = useMemo(
    () => flattenProjectSessions(visibleDisplayGroups, projectSortMode),
    [projectSortMode, visibleDisplayGroups],
  );
  const selectedProject = useMemo(
    () => projects.find((project) => project.id === activeProjectId) ?? null,
    [activeProjectId, projects],
  );
  const selectedPermissionPlan = useMemo(
    () =>
      (extensionInventory?.agentPlans ?? []).find(
        (plan) =>
          plan.agentProfileId === activeExtensionAgent?.id ||
          plan.agentId === activeExtensionAgent?.id,
      ),
    [activeExtensionAgent?.id, extensionInventory?.agentPlans],
  );
  const permissionAgentPolicy =
    activeExtensionAgent?.harnessRecipe?.permissions ?? selectedPermissionPlan?.permissions;
  const permissionContext = {
    ...(selectedProject?.cwd ? { cwd: selectedProject.cwd } : {}),
    ...(activeExtensionAgent && permissionAgentPolicy
      ? { agentId: activeExtensionAgent.id, agentPolicy: permissionAgentPolicy }
      : {}),
  };
  const {
    data: permissionStatus,
    error: permissionStatusError,
    isLoading: permissionStatusLoading,
    mutate: mutatePermissionStatus,
  } = useSWR<DesktopPermissionStatus>(
    [
      "permissions:status",
      selectedProject?.cwd ?? "personal",
      activeExtensionAgent?.id ?? "no-agent",
      JSON.stringify(permissionAgentPolicy ?? null),
    ],
    () => api.getPermissionStatus(permissionContext),
    { revalidateOnFocus: true, revalidateOnReconnect: false },
  );
  const actionProject = useMemo(
    () => projects.find((project) => project.id === projectActionMenuId) ?? null,
    [projectActionMenuId, projects],
  );
  const previewProject = useMemo(
    () => projects.find((project) => project.id === projectPreview?.projectId) ?? null,
    [projectPreview?.projectId, projects],
  );
  const previewProjectGroup = useMemo(
    () => orderedDisplayGroups.find((group) => group.project?.id === previewProject?.id) ?? null,
    [orderedDisplayGroups, previewProject?.id],
  );
  const emptyProjectLabel = useMemo(
    () =>
      projectDisplayName(
        currentSession?.cwd ||
          selectedDiscoveredSession?.cwd ||
          selectedProject?.name ||
          productConfig.name,
      ),
    [
      currentSession?.cwd,
      productConfig.name,
      selectedDiscoveredSession?.cwd,
      selectedProject?.name,
    ],
  );
  const composerWorkspaceRoot =
    currentSession?.cwd ||
    selectedDiscoveredSession?.cwd ||
    selectedProject?.cwd ||
    desktopWorkspaceRoot;
  const composerMentionServers = useMemo(() => {
    const servers = extensionInventory?.lspServers ?? [];
    const missingDefaults = DEFAULT_MENTION_SERVERS.filter(
      (defaultServer) => !servers.some((server) => server.id === defaultServer.id),
    );
    return [...missingDefaults, ...servers];
  }, [extensionInventory?.lspServers]);
  const selectedSessionKey = selectedDiscoveredSession
    ? sessionDetailKey(selectedDiscoveredSession)
    : null;
  const {
    data: selectedSessionData,
    error: selectedSessionError,
    isLoading: selectedSessionLoading,
  } = useSWR<SessionData | null>(
    selectedSessionKey,
    () =>
      selectedDiscoveredSession
        ? loadDiscoveredSessionDetail(selectedDiscoveredSession)
        : Promise.resolve(null),
    {
      keepPreviousData: false,
      revalidateIfStale: false,
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );
  const visibleSessionErrors = useMemo(() => {
    if (!selectedSessionError) return sessionErrors;
    return [
      ...sessionErrors,
      {
        harnessId: "session-load",
        harnessLabel: "Session Load",
        message: errorMessage(selectedSessionError),
      },
    ];
  }, [selectedSessionError, sessionErrors]);
  const workflowBadgeLabel = workflowEnabled
    ? activeWorkflowConfig
      ? activeWorkflowConfig.name
      : "Workflow invalid"
    : "Workflow draft";
  const runTitle = activeUiContribution?.name ?? currentSession?.title ?? productConfig.name;
  const runSubtitle = settingsSection
    ? settingsSection === "general"
      ? "Defaults for new conversations"
      : settingsSection === "profile"
        ? "Private, on-device activity"
        : "Providers, extensions, and runtime"
    : activeUiContribution
      ? `${activeUiContribution.placement} contribution${
          activeUiContribution.sourcePluginId ? ` via ${activeUiContribution.sourcePluginId}` : ""
        }`
      : currentSession
        ? `${currentSession.agentName} on ${harnessLabel(currentSession.harness)}`
        : activeWorkflowConfig
          ? `${activeWorkflowConfig.name} workflow ready`
          : activeExtensionAgent
            ? `${activeExtensionAgent.name} on ${activeExtensionAgent.harnessId ?? "extension harness"}`
            : `${activeHarness.label} ${harnessEnvironmentLabel(
                selectedHarnessEnvironment?.status,
                activeHarness.id,
              )}`;
  const headerTitle = settingsSection
    ? settingsSection === "general"
      ? "General"
      : settingsSection === "profile"
        ? "Profile"
        : "Settings"
    : activeUiContribution?.name
      ? activeUiContribution.name
      : workflowPanelOpen
        ? "Workflow"
        : (currentSession?.title ?? null);
  const canGoBack = navigationIndex > 0;
  const canGoForward = navigationIndex < navigationHistoryRef.current.length - 1;
  const updateVisible = desktopUpdate.phase !== "hidden";
  const updateBusy =
    desktopUpdate.phase === "downloading" ||
    desktopUpdate.phase === "installing" ||
    desktopUpdate.phase === "restarting";
  const updateLabel =
    desktopUpdate.phase === "downloading"
      ? desktopUpdate.progress === undefined
        ? "Downloading"
        : `${Math.round(desktopUpdate.progress)}%`
      : desktopUpdate.phase === "installing"
        ? "Installing"
        : desktopUpdate.phase === "restarting"
          ? "Restarting"
          : "Update";
  const updateVersion = desktopUpdate.latestVersion ?? "the latest version";
  const updateAccessibleLabel =
    desktopUpdate.phase === "downloading"
      ? `Downloading SwarmX ${updateLabel}`
      : desktopUpdate.phase === "installing"
        ? `Installing SwarmX ${updateVersion}`
        : desktopUpdate.phase === "restarting"
          ? `Restarting SwarmX ${updateVersion}`
          : `Update SwarmX to ${updateVersion}`;
  const updateTitle = desktopUpdate.error
    ? `Update failed: ${desktopUpdate.error}. Click to retry.`
    : updateAccessibleLabel;

  const prefetchSession = useCallback(
    (session: DiscoveredSession) => {
      const cacheId = sessionCacheId(session);
      if (preloadedSessionKeys.current.has(cacheId)) return;
      preloadedSessionKeys.current.add(cacheId);
      void loadDiscoveredSessionDetail(session)
        .then((data) => {
          if (data) {
            void mutateSessionDetail(sessionDetailKey(session), data, {
              populateCache: true,
              revalidate: false,
            });
          }
        })
        .catch(() => {
          preloadedSessionKeys.current.delete(cacheId);
        });
    },
    [mutateSessionDetail],
  );

  useEffect(() => {
    for (const session of preloadSessionCandidates(displayGroups)) {
      prefetchSession(session);
    }
  }, [displayGroups, prefetchSession]);

  useEffect(() => {
    if (
      !selectedSessionData ||
      !selectedDiscoveredSession ||
      selectedSessionData.id !== selectedDiscoveredSession.id
    ) {
      return;
    }
    setCurrentSession(selectedSessionData);
    setSelectedHarness(selectedSessionData.harness);
    setSelectedExtensionAgentId(null);
    setActiveUiContributionId(null);
    const project =
      projects.find((candidate) => candidate.id === selectedDiscoveredSession.projectId) ??
      projects.find((candidate) => sameProjectPath(candidate.cwd, selectedDiscoveredSession.cwd));
    if (project) setActiveProjectId(project.id);
  }, [projects, selectedDiscoveredSession, selectedSessionData]);

  useEffect(() => {
    if (
      activeUiContributionId &&
      !registeredUiContributions.some((contribution) => contribution.id === activeUiContributionId)
    ) {
      setActiveUiContributionId(null);
    }
  }, [activeUiContributionId, registeredUiContributions]);

  useLayoutEffect(() => {
    const chat = chatRef.current;
    const sessionId = currentSession?.id ?? null;
    const previous = scrollStateRef.current;

    scrollStateRef.current = { sessionId, messageCount };

    if (!chat || messageCount === 0) return;

    const sessionChanged = sessionId !== previous.sessionId;
    const messageAdded = sessionId !== null && messageCount > previous.messageCount;

    chat.scrollTo({
      top: chat.scrollHeight,
      behavior: sessionChanged || !messageAdded || prefersReducedMotion() ? "auto" : "smooth",
    });
  }, [currentSession?.id, messageCount]);

  const applyNavigationEntry = useCallback((session: DiscoveredSession | null) => {
    setActiveUiContributionId(null);
    setWorkflowPanelOpen(false);
    setSettingsSection(null);
    setDoctorPanelOpen(false);
    setComposerError(null);
    if (session) {
      setCurrentSession(null);
      setSelectedDiscoveredSession(session);
      return;
    }
    setSelectedDiscoveredSession(null);
    setCurrentSession(null);
  }, []);

  const recordNavigationEntry = useCallback(
    (session: DiscoveredSession | null) => {
      const currentIndex = navigationIndexRef.current;
      const current = navigationHistoryRef.current[currentIndex] ?? null;
      if (navigationEntryKey(current) === navigationEntryKey(session)) {
        applyNavigationEntry(session);
        return;
      }

      const nextHistory = navigationHistoryRef.current.slice(0, currentIndex + 1);
      nextHistory.push(session);
      const nextIndex = nextHistory.length - 1;
      navigationHistoryRef.current = nextHistory;
      navigationIndexRef.current = nextIndex;
      setNavigationIndex(nextIndex);
      applyNavigationEntry(session);
    },
    [applyNavigationEntry],
  );

  const navigateHistory = useCallback(
    (delta: -1 | 1) => {
      const nextIndex = navigationIndexRef.current + delta;
      if (nextIndex < 0 || nextIndex >= navigationHistoryRef.current.length) return;
      navigationIndexRef.current = nextIndex;
      setNavigationIndex(nextIndex);
      applyNavigationEntry(navigationHistoryRef.current[nextIndex] ?? null);
    },
    [applyNavigationEntry],
  );

  const replaceCurrentNavigationEntry = useCallback((session: SessionData) => {
    const currentIndex = navigationIndexRef.current;
    const nextHistory = [...navigationHistoryRef.current];
    nextHistory[currentIndex] = localSessionToDiscovered(session);
    navigationHistoryRef.current = nextHistory;
  }, []);

  const newSession = useCallback(
    (project: ProjectData | null = selectedProject) => {
      if (project) setActiveProjectId(project.id);
      setInput("");
      setSidebarQuery("");
      setSidebarSearchOpen(false);
      setProjectError(null);
      setNewSessionPermissionMode("inherit");
      setPermissionPickerOpen(false);
      recordNavigationEntry(null);
      window.requestAnimationFrame(() => composerRef.current?.focus());
    },
    [recordNavigationEntry, selectedProject],
  );

  const changeSessionPermissionMode = useCallback(
    async (permissionMode: SessionPermissionMode) => {
      setComposerError(null);
      if (!currentSession) {
        setNewSessionPermissionMode(permissionMode);
        return;
      }
      const previous = currentSession;
      const updated = { ...currentSession, permissionMode };
      setCurrentSession(updated);
      try {
        await api.saveSession(updated);
      } catch (error) {
        setCurrentSession((visible) => (visible?.id === previous.id ? previous : visible));
        setComposerError(`Could not save conversation permissions: ${errorMessage(error)}`);
        throw error;
      }
    },
    [currentSession],
  );

  const addProject = useCallback(
    async (kind: "scratch" | "existing") => {
      setProjectHeaderMenu(null);
      setProjectActionPending(true);
      setProjectError(null);
      try {
        const project =
          kind === "scratch" ? await api.createScratchProject() : await api.addExistingProject();
        if (!project) return;
        await mutateProjects((current = []) => {
          const next = current.filter((candidate) => candidate.id !== project.id);
          return [...next, project];
        }, false);
        newSession(project);
      } catch (error) {
        setProjectError(errorMessage(error));
      } finally {
        setProjectActionPending(false);
      }
    },
    [mutateProjects, newSession],
  );

  const updateCachedProject = useCallback(
    async (project: ProjectData) => {
      await mutateProjects((current = []) => {
        const index = current.findIndex((candidate) => candidate.id === project.id);
        if (index < 0) return [...current, project];
        const next = [...current];
        next[index] = project;
        return next;
      }, false);
    },
    [mutateProjects],
  );

  const cancelProjectPreviewClose = useCallback(() => {
    if (projectPreviewCloseTimerRef.current === null) return;
    window.clearTimeout(projectPreviewCloseTimerRef.current);
    projectPreviewCloseTimerRef.current = null;
  }, []);

  const scheduleProjectPreviewClose = useCallback(() => {
    cancelProjectPreviewClose();
    projectPreviewCloseTimerRef.current = window.setTimeout(() => {
      setProjectPreview(null);
      projectPreviewCloseTimerRef.current = null;
    }, 120);
  }, [cancelProjectPreviewClose]);

  const showProjectPreview = useCallback(
    (project: ProjectData, anchor: HTMLElement) => {
      cancelProjectPreviewClose();
      if (projectActionMenuId) return;
      const rect = anchor.getBoundingClientRect();
      const cardWidth = 344;
      const gap = 4;
      const preferredLeft = rect.right + gap;
      const left =
        preferredLeft + cardWidth <= window.innerWidth - 8
          ? preferredLeft
          : Math.max(8, rect.left - cardWidth - gap);
      setProjectPreview({
        projectId: project.id,
        top: Math.max(8, Math.min(rect.top, window.innerHeight - 112)),
        left,
      });
    },
    [cancelProjectPreviewClose, projectActionMenuId],
  );

  const togglePreviewProjectPinned = useCallback(
    async (project: ProjectData) => {
      setProjectActionPending(true);
      setProjectError(null);
      try {
        await updateCachedProject(await api.setProjectPinned(project.id, !project.pinned));
      } catch (error) {
        setProjectError(errorMessage(error));
      } finally {
        setProjectActionPending(false);
      }
    },
    [updateCachedProject],
  );

  const toggleProjectPinned = useCallback(async () => {
    if (!actionProject) return;
    setProjectActionMenuId(null);
    setProjectActionPending(true);
    setProjectError(null);
    try {
      await updateCachedProject(
        await api.setProjectPinned(actionProject.id, !actionProject.pinned),
      );
    } catch (error) {
      setProjectError(errorMessage(error));
    } finally {
      setProjectActionPending(false);
    }
  }, [actionProject, updateCachedProject]);

  const revealSelectedProject = useCallback(async () => {
    if (!actionProject) return;
    setProjectActionMenuId(null);
    setProjectError(null);
    try {
      await api.revealProject(actionProject.id);
    } catch (error) {
      setProjectError(errorMessage(error));
    }
  }, [actionProject]);

  const startProjectRename = useCallback(() => {
    if (!actionProject) return;
    setProjectActionMenuId(null);
    setProjectRenameDraft(actionProject.name);
    setRenamingProjectId(actionProject.id);
  }, [actionProject]);

  const commitProjectRename = useCallback(async () => {
    const projectId = renamingProjectId;
    if (!projectId) return;
    const project = projects.find((candidate) => candidate.id === projectId);
    const nextName = projectRenameDraft.trim();
    setRenamingProjectId(null);
    if (!project || !nextName || nextName === project.name) return;
    setProjectActionPending(true);
    setProjectError(null);
    try {
      await updateCachedProject(await api.renameProject(project.id, nextName));
    } catch (error) {
      setProjectError(errorMessage(error));
    } finally {
      setProjectActionPending(false);
    }
  }, [projectRenameDraft, projects, renamingProjectId, updateCachedProject]);

  const archiveSelectedProjectTasks = useCallback(async () => {
    if (!actionProject) return;
    setProjectActionMenuId(null);
    setProjectActionPending(true);
    setProjectError(null);
    try {
      await api.archiveProjectTasks(actionProject.id);
      recordNavigationEntry(null);
      await Promise.all([mutateLocalSessions(), mutateGroupedSessions()]);
    } catch (error) {
      setProjectError(errorMessage(error));
    } finally {
      setProjectActionPending(false);
    }
  }, [actionProject, mutateGroupedSessions, mutateLocalSessions, recordNavigationEntry]);

  const removeSelectedProject = useCallback(async () => {
    if (!actionProject) return;
    setProjectActionMenuId(null);
    setProjectActionPending(true);
    setProjectError(null);
    try {
      const removed = await api.removeProject(actionProject.id);
      if (!removed) return;
      await mutateProjects(
        (current = []) => current.filter((project) => project.id !== actionProject.id),
        false,
      );
      if (activeProjectId === actionProject.id) {
        setActiveProjectId(null);
        recordNavigationEntry(null);
      }
    } catch (error) {
      setProjectError(errorMessage(error));
    } finally {
      setProjectActionPending(false);
    }
  }, [actionProject, activeProjectId, mutateProjects, recordNavigationEntry]);

  const selectSession = useCallback(
    (session: DiscoveredSession) => {
      const project =
        projects.find((candidate) => candidate.id === session.projectId) ??
        projects.find((candidate) => sameProjectPath(candidate.cwd, session.cwd));
      if (project) setActiveProjectId(project.id);
      recordNavigationEntry(session);
    },
    [projects, recordNavigationEntry],
  );

  const selectExtensionAgentForRun = useCallback((agentId: string) => {
    setSelectedExtensionAgentId(agentId);
    setWorkflowEnabled(false);
    setWorkflowPanelOpen(false);
    setSettingsSection(null);
    setDoctorPanelOpen(false);
    setActiveUiContributionId(null);
  }, []);

  const openSettings = useCallback((section: SettingsSection) => {
    setSettingsSection(section);
    setSettingsQuery("");
    setSidebarOpen(true);
    setAccountMenuOpen(false);
    setWorkflowPanelOpen(false);
    setDoctorPanelOpen(false);
    setRightPanelOpen(false);
    setActiveUiContributionId(null);
  }, []);

  const startDesktopUpdate = useCallback(async () => {
    if (desktopUpdate.phase !== "available" || !api.startUpdate) return;
    setDesktopUpdate((current) => ({
      phase: "downloading",
      currentVersion: current.currentVersion,
      latestVersion: current.latestVersion,
      progress: 0,
    }));
    try {
      setDesktopUpdate(await api.startUpdate());
    } catch (error) {
      setDesktopUpdate((current) => ({
        phase: "available",
        currentVersion: current.currentVersion,
        latestVersion: current.latestVersion,
        error: errorMessage(error),
      }));
    }
  }, [desktopUpdate.phase]);

  const checkDoctorHarnessVersion = useCallback(async (harnessId: string, refresh = false) => {
    setDoctorHarnessVersions((current) => ({
      ...current,
      [harnessId]: {
        status: "loading",
        version: current[harnessId]?.version,
      },
    }));
    try {
      const result = await api.getHarnessVersion({
        harnessId,
        ...(refresh ? { refresh: true } : {}),
      });
      setDoctorHarnessVersions((current) => ({
        ...current,
        [harnessId]: { status: "loaded", version: result.version },
      }));
    } catch (error) {
      setDoctorHarnessVersions((current) => ({
        ...current,
        [harnessId]: { status: "loaded", version: current[harnessId]?.version },
      }));
      setDoctorError(errorMessage(error));
    }
  }, []);

  const openDoctorPanel = useCallback(
    async ({
      mode = "doctor",
      harnessId,
      requestFix = false,
    }: {
      mode?: DoctorPanelMode;
      harnessId?: string;
      requestFix?: boolean;
    } = {}) => {
      setDoctorPanelOpen(true);
      setDoctorPanelMode(mode);
      setDoctorHarnessId(harnessId ?? null);
      setRightPanelOpen(false);
      setWorkflowPanelOpen(false);
      setSettingsSection(null);
      setActiveUiContributionId(null);
      setDoctorLoading(true);
      setDoctorFixPending(false);
      setDoctorFixResult(null);
      setDoctorError(null);
      if (!doctorVersionChecksStarted.current) {
        doctorVersionChecksStarted.current = true;
        for (const harness of HARNESSES) {
          void checkDoctorHarnessVersion(harness.id);
        }
      }
      try {
        const report = await api.inspectDoctor(harnessId ? { harnessId } : {});
        setDoctorReport(report);
        setDoctorFixPending(requestFix && report.repairActions.length > 0);
        if (!harnessId) {
          await mutateHarnessEnvironment(report.environment, {
            populateCache: true,
            revalidate: false,
          });
        }
      } catch (error) {
        setDoctorError(errorMessage(error));
      } finally {
        setDoctorLoading(false);
      }
    },
    [checkDoctorHarnessVersion, mutateHarnessEnvironment],
  );

  const refreshRuntimeDoctor = useCallback(
    async (refreshVersions = false) => {
      setDoctorHarnessId(null);
      setDoctorLoading(true);
      setDoctorFixPending(false);
      setDoctorError(null);
      if (refreshVersions) {
        await Promise.all(HARNESSES.map((harness) => checkDoctorHarnessVersion(harness.id, true)));
        doctorVersionChecksStarted.current = true;
      } else if (!doctorVersionChecksStarted.current) {
        doctorVersionChecksStarted.current = true;
        for (const harness of HARNESSES) {
          void checkDoctorHarnessVersion(harness.id);
        }
      }
      try {
        const report = await api.inspectDoctor();
        setDoctorReport(report);
        await mutateHarnessEnvironment(report.environment, {
          populateCache: true,
          revalidate: false,
        });
      } catch (error) {
        setDoctorError(errorMessage(error));
      } finally {
        setDoctorLoading(false);
      }
    },
    [checkDoctorHarnessVersion, mutateHarnessEnvironment],
  );

  useEffect(() => {
    if (settingsSection !== "runtime") return;
    void refreshRuntimeDoctor();
  }, [refreshRuntimeDoctor, settingsSection]);

  const confirmDoctorFix = useCallback(async () => {
    if (doctorFixRunning || !doctorReport?.repairActions.length) return;
    setDoctorFixRunning(true);
    setDoctorError(null);
    try {
      const result = await api.fixDoctor({
        ...(doctorHarnessId ? { harnessId: doctorHarnessId } : {}),
        confirmed: true,
      });
      setDoctorFixResult(result);
      setDoctorReport(result.after);
      setDoctorFixPending(false);
      if (!doctorHarnessId) {
        await mutateHarnessEnvironment(result.after.environment, {
          populateCache: true,
          revalidate: false,
        });
      }
      await Promise.all([mutateGroupedSessions(), mutateExtensionInventory()]);
    } catch (error) {
      setDoctorError(errorMessage(error));
    } finally {
      setDoctorFixRunning(false);
    }
  }, [
    doctorFixRunning,
    doctorHarnessId,
    doctorReport?.repairActions.length,
    mutateExtensionInventory,
    mutateGroupedSessions,
    mutateHarnessEnvironment,
  ]);

  const installDoctorHarness = useCallback(
    async (harnessId: string) => {
      if (doctorInstallingHarnessId) return;
      setDoctorInstallingHarnessId(harnessId);
      setDoctorError(null);
      try {
        const result = await api.setupHarnessEnvironment({ harnessToolId: harnessId });
        await mutateHarnessEnvironment(result.status, {
          populateCache: true,
          revalidate: false,
        });
        const report = await api.inspectDoctor(
          doctorHarnessId ? { harnessId: doctorHarnessId } : {},
        );
        setDoctorReport(report);
        await checkDoctorHarnessVersion(harnessId, true);
        if (!result.success) {
          setDoctorError(result.error ?? `Could not install ${harnessLabel(harnessId)}.`);
        }
      } catch (error) {
        setDoctorError(errorMessage(error));
      } finally {
        setDoctorInstallingHarnessId(null);
      }
    },
    [
      checkDoctorHarnessVersion,
      doctorHarnessId,
      doctorInstallingHarnessId,
      mutateHarnessEnvironment,
    ],
  );

  const applyUpdatedLocalSession = useCallback(
    async (session: SessionData) => {
      setCurrentSession((current) => {
        if (current?.id !== session.id) return current;
        replaceCurrentNavigationEntry(session);
        return { ...current, ...session };
      });
      await Promise.all([mutateLocalSessions(), mutateGroupedSessions()]);
    },
    [mutateGroupedSessions, mutateLocalSessions, replaceCurrentNavigationEntry],
  );

  const requestAutomaticSessionTitle = useCallback(
    (session: SessionData, userText: string) => {
      const userMessageCount = session.messages.filter(
        (message) => message.kind === "message" && message.role === "user",
      ).length;
      if (!isPlaceholderSessionTitle(session.title) || userMessageCount !== 1) return;

      void api
        .generateSessionTitle(session.id, userText)
        .then(async (result) => {
          if (!result.updated) return;
          await applyUpdatedLocalSession({ ...session, title: result.title });
        })
        .catch(() => {
          // Title generation is a non-blocking enhancement; the task remains usable on failure.
        });
    },
    [applyUpdatedLocalSession],
  );

  const openSessionRename = useCallback((session: DiscoveredSession) => {
    if (session.source !== "local") return;
    setSessionContextMenu(null);
    setSessionActionError(null);
    setSessionRenameDraft(session.title || "Untitled");
    setRenamingSession(session);
  }, []);

  const commitSessionRename = useCallback(async () => {
    if (!renamingSession || sessionActionPending) return;
    const title = sessionRenameDraft.replace(/\s+/gu, " ").trim();
    if (!title) {
      setSessionActionError("Task title cannot be empty.");
      return;
    }
    setSessionActionPending(true);
    setSessionActionError(null);
    try {
      const updated = await api.renameSession(renamingSession.id, title);
      await applyUpdatedLocalSession(updated);
      setRenamingSession(null);
    } catch (error) {
      setSessionActionError(errorMessage(error));
    } finally {
      setSessionActionPending(false);
    }
  }, [applyUpdatedLocalSession, renamingSession, sessionActionPending, sessionRenameDraft]);

  const toggleSessionPinned = useCallback(async () => {
    const session = sessionContextMenu?.session;
    if (!session || session.source !== "local" || sessionActionPending) return;
    setSessionContextMenu(null);
    setSessionActionPending(true);
    setSessionActionError(null);
    try {
      await applyUpdatedLocalSession(await api.setSessionPinned(session.id, !session.pinned));
    } catch (error) {
      setSessionActionError(errorMessage(error));
    } finally {
      setSessionActionPending(false);
    }
  }, [applyUpdatedLocalSession, sessionActionPending, sessionContextMenu]);

  const deleteSidebarSession = useCallback(async () => {
    const session = sessionContextMenu?.session;
    if (!session || session.source !== "local" || sessionActionPending) return;
    setSessionContextMenu(null);
    if (!window.confirm(`Delete “${session.title || "Untitled"}”? This cannot be undone.`)) return;
    setSessionActionPending(true);
    setSessionActionError(null);
    try {
      const deleted = await api.deleteSession(session.id);
      if (!deleted) throw new Error(`Unknown session: ${session.id}`);
      if (currentSession?.id === session.id || selectedDiscoveredSession?.id === session.id) {
        recordNavigationEntry(null);
      }
      await Promise.all([mutateLocalSessions(), mutateGroupedSessions()]);
    } catch (error) {
      setSessionActionError(errorMessage(error));
    } finally {
      setSessionActionPending(false);
    }
  }, [
    currentSession?.id,
    mutateGroupedSessions,
    mutateLocalSessions,
    recordNavigationEntry,
    selectedDiscoveredSession?.id,
    sessionActionPending,
    sessionContextMenu,
  ]);

  const openSessionContextMenu = useCallback((session: DiscoveredSession, x: number, y: number) => {
    if (session.source !== "local") return;
    const menuWidth = 208;
    const menuHeight = 150;
    setSessionActionError(null);
    setSessionContextMenu({
      session,
      x: Math.max(8, Math.min(x, window.innerWidth - menuWidth - 8)),
      y: Math.max(8, Math.min(y, window.innerHeight - menuHeight - 8)),
    });
  }, []);

  const importN8nWorkflowFile = useCallback(async (file: File) => {
    try {
      const source = await file.text();
      const result = (await api.importN8nWorkflow(source)) as N8nImportResponse;
      if (result.success && result.config) {
        setWorkflowJson(JSON.stringify(result.config, null, 2));
        setWorkflowEnabled(true);
        setWorkflowImportStatus({
          kind: "success",
          message: `Imported n8n workflow "${result.config.name}".`,
          warnings: result.warnings ?? [],
        });
        return;
      }

      setWorkflowImportStatus({
        kind: "error",
        message: result.error ?? "Failed to import n8n workflow.",
        warnings: [],
      });
    } catch (error) {
      setWorkflowImportStatus({
        kind: "error",
        message: `Failed to read n8n workflow file: ${errorMessage(error)}`,
        warnings: [],
      });
    }
  }, []);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;
    const slashCommand = parseDesktopSlashCommand(text);
    if (slashCommand?.kind === "error") {
      setComposerError(slashCommand.message);
      return;
    }
    if (slashCommand) {
      setInput("");
      setComposerError(null);
      await openDoctorPanel({
        mode: slashCommand.kind,
        harnessId: slashCommand.harnessId,
        requestFix: slashCommand.kind === "doctor" && slashCommand.fix,
      });
      return;
    }
    if (activeWorkflowInvalid) return;
    if (manualCompositionNeedsModel || acpHistoryReadOnly) {
      return;
    }
    if (selectedHarnessNeedsSetup) {
      await openDoctorPanel({ mode: "doctor", harnessId: activeRunHarnessId });
      return;
    }
    setInput("");
    setComposerError(null);
    setLoading(true);
    setRunState("running");
    const requestId = crypto.randomUUID();
    activeRequestId.current = requestId;
    setAgentInteractions([]);
    setAgentInteractionError(null);
    setResolvingInteractionId(null);
    requestDispatched.current = false;
    stopRequested.current = false;
    let sessionForError = currentSession;
    let pendingSession: SessionData | null = null;
    let streamedMessages: MessageChunk[] = [];
    let requestStartedAt: string | null = null;
    let unsubscribeAgentChunks: () => void = () => undefined;
    const stoppedBeforeDispatch = (): boolean =>
      activeRequestId.current === requestId && stopRequested.current && !requestDispatched.current;

    try {
      const userChunk: MessageChunk = {
        role: "user",
        content: text,
        kind: "message",
      };

      let session: SessionData;
      if (currentSession) {
        session = currentSession;
      } else {
        session = (await api.createSession({
          agentName: activeWorkflowConfig?.name ?? activeExtensionAgent?.name ?? "agent",
          harness: activeExtensionAgent?.harnessId ?? selectedHarness,
          permissionMode: newSessionPermissionMode,
          ...(selectedProject ? { projectId: selectedProject.id, cwd: selectedProject.cwd } : {}),
        })) as SessionData;
        sessionForError = session;
        if (stoppedBeforeDispatch()) return;
        await api.saveSession(session);
        replaceCurrentNavigationEntry(session);
        if (stoppedBeforeDispatch()) return;
      }

      const updatedMessages = [...session.messages, userChunk];
      pendingSession = { ...session, messages: updatedMessages };
      setCurrentSession(pendingSession);
      await api.saveSession(pendingSession);
      if (stoppedBeforeDispatch()) return;

      const sendParams: {
        requestId: string;
        sessionId?: string;
        harnessId: string;
        userText: string;
        agentComposition?: AgentCompositionPayload;
        swarmConfig?: SwarmConfig;
        cwd?: string;
      } = {
        requestId,
        sessionId: session.id,
        harnessId: activeExtensionAgent?.harnessId ?? selectedHarness,
        userText: text,
        ...(session.cwd || composerWorkspaceRoot
          ? { cwd: session.cwd || composerWorkspaceRoot }
          : {}),
      };
      if (activeWorkflowConfig) {
        sendParams.swarmConfig = activeWorkflowConfig;
      } else if (activeExtensionAgent) {
        sendParams.agentComposition = extensionAgentComposition(activeExtensionAgent);
      } else {
        sendParams.agentComposition = {
          id: selectedModel ? `desktop-${selectedModel.id}` : `desktop-${selectedHarness}`,
          harnessId: selectedHarness,
          ...(selectedModel
            ? {
                modelId: selectedModel.modelId,
                ...(selectedModel.modelSupplyId
                  ? { modelSupplyId: selectedModel.modelSupplyId }
                  : {}),
                ...(selectedEffort ? { effort: selectedEffort } : {}),
              }
            : {}),
          host: "local",
        };
      }

      requestStartedAt = new Date().toISOString();
      unsubscribeAgentChunks = api.onAgentChunk((event) => {
        if (event.requestId !== requestId || activeRequestId.current !== requestId) return;
        streamedMessages = mergeStreamingMessage(streamedMessages, event.chunk);
        setCurrentSession((visibleSession) => {
          if (!visibleSession || visibleSession.id !== session.id) return visibleSession;
          return { ...visibleSession, messages: [...updatedMessages, ...streamedMessages] };
        });
      });
      requestDispatched.current = true;
      const result = await api.sendMessage(sendParams);
      const requestEndedAt = new Date().toISOString();

      if (result.success && result.messages) {
        const responseMessages = withRequestTiming(
          result.messages as MessageChunk[],
          requestStartedAt,
          requestEndedAt,
        );
        const localUpdated = { ...session, messages: [...updatedMessages, ...responseMessages] };
        const persisted = result.sessionPersisted ? await api.loadSession(session.id) : null;
        const updated = persisted ?? localUpdated;
        if (!persisted) await api.saveSession(updated);
        setCurrentSession(updated);
        requestAutomaticSessionTitle(updated, text);
      } else if (result.canceled) {
        const canceledMessages = requestStartedAt
          ? withRequestTiming(streamedMessages, requestStartedAt, requestEndedAt)
          : streamedMessages;
        const localUpdated = { ...session, messages: [...updatedMessages, ...canceledMessages] };
        const persisted = result.sessionPersisted ? await api.loadSession(session.id) : null;
        const updated = persisted ?? localUpdated;
        if (!persisted) await api.saveSession(updated);
        setCurrentSession(updated);
      } else if (result.error) {
        const workMessages = requestStartedAt
          ? withRequestTiming(streamedMessages, requestStartedAt, requestEndedAt)
          : streamedMessages;
        const errorMsg: MessageChunk = {
          role: "system",
          content: `Error: ${result.error}`,
          kind: "message",
        };
        const localUpdated = {
          ...session,
          messages: [...updatedMessages, ...workMessages, errorMsg],
        };
        const persisted = result.sessionPersisted ? await api.loadSession(session.id) : null;
        const updated = persisted ?? localUpdated;
        if (!persisted) await api.saveSession(updated);
        setCurrentSession(updated);
      }

      await mutateLocalSessions();
    } catch (error) {
      if (activeRequestId.current !== requestId) return;
      const message = `Error: ${errorMessage(error)}`;
      setComposerError(message);
      const session = pendingSession ?? sessionForError;
      if (session) {
        const endedAt = new Date().toISOString();
        const workMessages = requestStartedAt
          ? withRequestTiming(streamedMessages, requestStartedAt, endedAt)
          : streamedMessages;
        const updated = {
          ...session,
          messages: [
            ...session.messages,
            ...workMessages,
            { role: "system", content: message, kind: "message" as const },
          ],
        };
        setCurrentSession(updated);
        try {
          await api.saveSession(updated);
        } catch {
          // The visible error remains available even if persistence IPC also failed.
        }
      }
    } finally {
      unsubscribeAgentChunks();
      setAgentInteractions((current) =>
        current.filter((interaction) => interaction.requestId !== requestId),
      );
      setAgentInteractionError(null);
      setResolvingInteractionId(null);
      if (activeRequestId.current === requestId) {
        activeRequestId.current = null;
        requestDispatched.current = false;
        stopRequested.current = false;
        setLoading(false);
        setRunState("idle");
      }
    }
  }, [
    input,
    loading,
    currentSession,
    selectedHarness,
    activeWorkflowConfig,
    activeWorkflowInvalid,
    activeExtensionAgent,
    selectedModel,
    selectedEffort,
    selectedHarnessNeedsSetup,
    activeRunHarnessId,
    selectedProject,
    newSessionPermissionMode,
    composerWorkspaceRoot,
    manualCompositionNeedsModel,
    acpHistoryReadOnly,
    mutateLocalSessions,
    openDoctorPanel,
    replaceCurrentNavigationEntry,
    requestAutomaticSessionTitle,
  ]);

  const resolveAgentInteraction = useCallback(
    async (interaction: AgentInteractionEvent, response: AgentInteractionResponse) => {
      if (activeRequestId.current !== interaction.requestId) return;
      setResolvingInteractionId(interaction.interactionId);
      setAgentInteractionError(null);
      try {
        const result = await api.resolveAgentInteraction({
          requestId: interaction.requestId,
          interactionId: interaction.interactionId,
          response,
        });
        if (!result.resolved) throw new Error("This interaction is no longer active.");
        setAgentInteractions((current) =>
          current.filter((candidate) => candidate.interactionId !== interaction.interactionId),
        );
      } catch (error) {
        setAgentInteractionError(errorMessage(error));
      } finally {
        setResolvingInteractionId((current) =>
          current === interaction.interactionId ? null : current,
        );
      }
    },
    [],
  );

  const stopMessage = useCallback(async () => {
    const requestId = activeRequestId.current;
    if (!requestId || runState !== "running") return;
    setRunState("stopping");
    if (!requestDispatched.current) {
      stopRequested.current = true;
      return;
    }
    try {
      const result = await api.cancelMessage(requestId);
      if (!result.canceled && activeRequestId.current === requestId) setRunState("running");
    } catch {
      if (activeRequestId.current === requestId) setRunState("running");
    }
  }, [runState]);

  const renderSidebarSessionItem = (session: DiscoveredSession) => {
    const isLocal = session.source === "local";
    const isActive =
      currentSession?.id === session.id && currentSession.harness === session.harnessId;
    const isPending =
      selectedSessionLoading &&
      selectedDiscoveredSession !== null &&
      sessionCacheId(selectedDiscoveredSession) === sessionCacheId(session);
    return (
      <button
        type="button"
        key={`${session.source}:${session.harnessId}:${session.id}`}
        onFocus={() => prefetchSession(session)}
        onPointerEnter={() => prefetchSession(session)}
        onClick={() => selectSession(session)}
        onDoubleClick={(event) => {
          event.preventDefault();
          openSessionRename(session);
        }}
        onContextMenu={(event) => {
          if (!isLocal) return;
          event.preventDefault();
          openSessionContextMenu(session, event.clientX, event.clientY);
        }}
        onKeyDown={(event) => {
          if (
            !isLocal ||
            (event.key !== "ContextMenu" && !(event.shiftKey && event.key === "F10"))
          ) {
            return;
          }
          event.preventDefault();
          const rect = event.currentTarget.getBoundingClientRect();
          openSessionContextMenu(session, rect.left + 20, rect.top + 24);
        }}
        className={cx("session-item", isActive && "is-active", isPending && "is-loading")}
      >
        <span className="session-item__icon">
          {isPending ? (
            <Loader2 aria-hidden="true" />
          ) : isLocal ? (
            <Clock3 aria-hidden="true" />
          ) : (
            <GitBranch aria-hidden="true" />
          )}
        </span>
        <span className="session-item__body">
          <span className="session-item__title">{session.title || "Untitled"}</span>
          <span className="session-item__meta">{sessionMeta(session, sessionGroupMode)}</span>
        </span>
        {session.pinned && <Pin className="session-item__pin" aria-label="Pinned task" />}
      </button>
    );
  };
  const activeAgentInteraction = agentInteractions[0];

  return (
    <div
      className={cx(
        "app-shell",
        !sidebarOpen && "app-shell--collapsed",
        settingsSection && "app-shell--settings",
      )}
    >
      {!settingsSection && (
        <header
          className={cx("app-titlebar", isMacOS && !sidebarOpen && "app-titlebar--macos")}
          aria-label="Window title bar"
        >
          <div className="runtime__titlebar">
            {!sidebarOpen && (
              <>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setSidebarOpen(true)}
                  title="Open sidebar"
                  aria-label="Open sidebar"
                >
                  <PanelLeftOpen data-icon aria-hidden="true" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => navigateHistory(-1)}
                  disabled={!canGoBack}
                  title="Go back"
                  aria-label="Go back"
                >
                  <ArrowLeft data-icon aria-hidden="true" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => navigateHistory(1)}
                  disabled={!canGoForward}
                  title="Go forward"
                  aria-label="Go forward"
                >
                  <ArrowRight data-icon aria-hidden="true" />
                </Button>
              </>
            )}
            {headerTitle && (
              <div className="runtime__title">
                <h1>{headerTitle}</h1>
                <p>{runSubtitle}</p>
              </div>
            )}
          </div>

          <div className="runtime__actions">
            {!settingsSection && (
              <>
                <Button
                  variant={pinnedSummaryOpen ? "secondary" : "ghost"}
                  size="icon"
                  onClick={() => setPinnedSummaryOpen((open) => !open)}
                  title={pinnedSummaryOpen ? "Hide pinned summary" : "Show pinned summary"}
                  aria-label={pinnedSummaryOpen ? "Hide pinned summary" : "Show pinned summary"}
                  aria-pressed={pinnedSummaryOpen}
                >
                  <Pin data-icon aria-hidden="true" />
                </Button>
                <Button
                  variant={bottomPanelOpen ? "secondary" : "ghost"}
                  size="icon"
                  onClick={() => setBottomPanelOpen((open) => !open)}
                  title={bottomPanelOpen ? "Hide bottom panel" : "Show bottom panel"}
                  aria-label={bottomPanelOpen ? "Hide bottom panel" : "Show bottom panel"}
                  aria-pressed={bottomPanelOpen}
                >
                  <PanelBottom data-icon aria-hidden="true" />
                </Button>
                <Button
                  variant={rightPanelOpen || doctorPanelOpen ? "secondary" : "ghost"}
                  size="icon"
                  onClick={() => {
                    if (doctorPanelOpen) {
                      setDoctorPanelOpen(false);
                      return;
                    }
                    setRightPanelOpen((open) => !open);
                  }}
                  title={
                    rightPanelOpen || doctorPanelOpen ? "Hide right panel" : "Show right panel"
                  }
                  aria-label={
                    rightPanelOpen || doctorPanelOpen ? "Hide right panel" : "Show right panel"
                  }
                  aria-pressed={rightPanelOpen || doctorPanelOpen}
                >
                  <PanelRight data-icon aria-hidden="true" />
                </Button>
              </>
            )}
          </div>
        </header>
      )}

      <aside className="sidebar" aria-label={settingsSection ? "Settings navigation" : "Sessions"}>
        <div
          className={cx("sidebar__titlebar", isMacOS && "sidebar__titlebar--macos")}
          aria-label="Window navigation"
        >
          {sidebarOpen && !settingsSection && (
            <>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setSidebarOpen(false)}
                title="Collapse sidebar"
                aria-label="Collapse sidebar"
              >
                <PanelLeftClose data-icon aria-hidden="true" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => navigateHistory(-1)}
                disabled={!canGoBack}
                title="Go back"
                aria-label="Go back"
              >
                <ArrowLeft data-icon aria-hidden="true" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => navigateHistory(1)}
                disabled={!canGoForward}
                title="Go forward"
                aria-label="Go forward"
              >
                <ArrowRight data-icon aria-hidden="true" />
              </Button>
            </>
          )}
        </div>

        {settingsSection ? (
          <SettingsSidebar
            section={settingsSection}
            query={settingsQuery}
            onQueryChange={setSettingsQuery}
            onSectionChange={setSettingsSection}
            onBack={() => {
              setSettingsQuery("");
              setSettingsSection(null);
            }}
          />
        ) : (
          <>
            <div className="sidebar__brand">
              <div className="brand-copy">
                <div className="brand-title">{productConfig.name}</div>
                {productConfig.subtitle && (
                  <div className="brand-subtitle">{productConfig.subtitle}</div>
                )}
              </div>
              <button
                type="button"
                className="sidebar__search-toggle"
                aria-label={sidebarSearchOpen ? "Close session search" : "Search sessions"}
                aria-pressed={sidebarSearchOpen}
                onClick={() => {
                  setSidebarSearchOpen((open) => !open);
                  if (sidebarSearchOpen) setSidebarQuery("");
                }}
              >
                <Search aria-hidden="true" />
              </button>
            </div>

            {sidebarSearchOpen && (
              <div className="sidebar-search">
                <Search aria-hidden="true" />
                <input
                  ref={sidebarSearchRef}
                  type="search"
                  value={sidebarQuery}
                  placeholder="Search sessions"
                  aria-label="Search sessions"
                  onChange={(event) => setSidebarQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key !== "Escape") return;
                    setSidebarQuery("");
                    setSidebarSearchOpen(false);
                  }}
                />
              </div>
            )}

            <nav className="sidebar-primary-nav" aria-label="Workspace">
              <button
                type="button"
                className="sidebar-primary-nav__item"
                onClick={() => newSession()}
              >
                <MessageSquarePlus aria-hidden="true" />
                <span>New task</span>
              </button>
              <button
                type="button"
                className={cx("sidebar-primary-nav__item", workflowPanelOpen && "is-active")}
                onClick={() => {
                  setWorkflowPanelOpen((open) => !open);
                  setSettingsSection(null);
                  setDoctorPanelOpen(false);
                  setActiveUiContributionId(null);
                }}
                aria-pressed={workflowPanelOpen}
              >
                <Workflow aria-hidden="true" />
                <span>Workflow</span>
              </button>
            </nav>

            {registeredUiContributions.length > 0 && (
              <nav className="sidebar-extension-nav" aria-label="Registered GUI contributions">
                <div className="sidebar-extension-nav__header">
                  <span>Apps</span>
                  <span>{registeredUiContributions.length}</span>
                </div>
                {registeredUiContributions.map((contribution) => (
                  <button
                    key={contribution.id}
                    type="button"
                    className={cx(
                      "sidebar-extension-nav__item",
                      activeUiContributionId === contribution.id && "is-active",
                    )}
                    onClick={() => {
                      setActiveUiContributionId(contribution.id);
                      setWorkflowPanelOpen(false);
                      setSettingsSection(null);
                      setDoctorPanelOpen(false);
                    }}
                    aria-label={`Open ${contribution.name}`}
                  >
                    <Package aria-hidden="true" />
                    <span>{contribution.name}</span>
                  </button>
                ))}
              </nav>
            )}

            <div className="session-scroll" onScroll={() => setProjectPreview(null)}>
              <div className="project-list__header">
                <span>Projects</span>
                <div className="project-list__actions" ref={projectHeaderMenuRef}>
                  <button
                    type="button"
                    aria-label="Project options"
                    title="Project options"
                    onClick={() => {
                      setProjectActionMenuId(null);
                      setProjectHeaderMenu((menu) => (menu === "organize" ? null : "organize"));
                    }}
                  >
                    <MoreHorizontal aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    aria-label="Add project"
                    title="Add project"
                    disabled={projectActionPending}
                    onClick={() => {
                      setProjectActionMenuId(null);
                      setProjectHeaderMenu((menu) => (menu === "add" ? null : "add"));
                    }}
                  >
                    {projectActionPending ? (
                      <Loader2 className="is-spinning" aria-hidden="true" />
                    ) : (
                      <Plus aria-hidden="true" />
                    )}
                  </button>
                  {projectHeaderMenu && (
                    <div
                      className={cx(
                        "project-list__menu",
                        projectHeaderMenu === "organize" && "project-list__menu--organize",
                      )}
                      role="menu"
                      aria-label={
                        projectHeaderMenu === "organize" ? "Organize projects" : "Add project"
                      }
                    >
                      {projectHeaderMenu === "add" ? (
                        <>
                          <button
                            type="button"
                            role="menuitem"
                            onClick={() => void addProject("scratch")}
                          >
                            <Plus aria-hidden="true" />
                            <span>Start from scratch</span>
                          </button>
                          <button
                            type="button"
                            role="menuitem"
                            onClick={() => void addProject("existing")}
                          >
                            <Folder aria-hidden="true" />
                            <span>Use an existing folder</span>
                          </button>
                        </>
                      ) : (
                        <>
                          <div className="project-list__menu-label">Organize</div>
                          <button
                            type="button"
                            role="menuitemradio"
                            aria-checked={projectOrganizationMode === "project"}
                            onClick={() => {
                              setProjectOrganizationMode("project");
                              setProjectHeaderMenu(null);
                            }}
                          >
                            <span className="project-list__menu-check">
                              {projectOrganizationMode === "project" && (
                                <Check aria-hidden="true" />
                              )}
                            </span>
                            <span>By project</span>
                          </button>
                          <button
                            type="button"
                            role="menuitemradio"
                            aria-checked={projectOrganizationMode === "list"}
                            onClick={() => {
                              setProjectOrganizationMode("list");
                              setProjectHeaderMenu(null);
                            }}
                          >
                            <span className="project-list__menu-check">
                              {projectOrganizationMode === "list" && <Check aria-hidden="true" />}
                            </span>
                            <span>In one list</span>
                          </button>
                          <div className="project-list__menu-separator" />
                          <div className="project-list__menu-label">Sort by</div>
                          <button
                            type="button"
                            role="menuitemradio"
                            aria-checked={projectSortMode === "priority"}
                            onClick={() => {
                              setProjectSortMode("priority");
                              setProjectHeaderMenu(null);
                            }}
                          >
                            <span className="project-list__menu-check">
                              {projectSortMode === "priority" && <Check aria-hidden="true" />}
                            </span>
                            <span>Priority</span>
                          </button>
                          <button
                            type="button"
                            role="menuitemradio"
                            aria-checked={projectSortMode === "last-updated"}
                            onClick={() => {
                              setProjectSortMode("last-updated");
                              setProjectHeaderMenu(null);
                            }}
                          >
                            <span className="project-list__menu-check">
                              {projectSortMode === "last-updated" && <Check aria-hidden="true" />}
                            </span>
                            <span>Last updated</span>
                          </button>
                          <button
                            type="button"
                            role="menuitemradio"
                            aria-checked={projectSortMode === "manual"}
                            onClick={() => {
                              setProjectSortMode("manual");
                              setProjectHeaderMenu(null);
                            }}
                          >
                            <span className="project-list__menu-check">
                              {projectSortMode === "manual" && <Check aria-hidden="true" />}
                            </span>
                            <span>Manual order</span>
                          </button>
                        </>
                      )}
                    </div>
                  )}
                </div>
              </div>
              {projectError && <div className="session-error">{projectError}</div>}
              {sessionActionError && !renamingSession && (
                <div className="session-error">{sessionActionError}</div>
              )}
              {sessionsLoading && <div className="session-status">Loading projects</div>}
              {!sessionsLoading && visibleDisplayGroups.length === 0 && (
                <div className="session-status">
                  {sidebarQuery.trim() ? "No matching projects" : "No projects"}
                </div>
              )}
              {projectOrganizationMode === "list" ? (
                <div className="session-group__items project-list__flat">
                  {visibleFlatSessions.map(renderSidebarSessionItem)}
                </div>
              ) : (
                visibleDisplayGroups.map((group) => {
                  const expanded =
                    sidebarQuery.trim().length > 0 ||
                    !group.project ||
                    activeProjectId === group.project.id;
                  return (
                    <section
                      key={group.id}
                      className={cx(
                        "project-group",
                        expanded && "is-expanded",
                        projectActionMenuId === group.project?.id && "has-open-menu",
                      )}
                      aria-label={group.label}
                    >
                      {group.project?.id === renamingProjectId ? (
                        <form
                          className="project-group__rename"
                          onSubmit={(event) => {
                            event.preventDefault();
                            void commitProjectRename();
                          }}
                        >
                          <Folder aria-hidden="true" />
                          <input
                            ref={projectRenameInputRef}
                            aria-label={`Rename ${group.label}`}
                            value={projectRenameDraft}
                            onChange={(event) => setProjectRenameDraft(event.target.value)}
                            onBlur={() => void commitProjectRename()}
                            onKeyDown={(event) => {
                              if (event.key !== "Escape") return;
                              event.preventDefault();
                              setRenamingProjectId(null);
                            }}
                          />
                        </form>
                      ) : (
                        <div
                          className={cx(
                            "project-group__header-row",
                            group.project?.id === activeProjectId && "is-active",
                          )}
                          onPointerEnter={(event) => {
                            if (group.project)
                              showProjectPreview(group.project, event.currentTarget);
                          }}
                          onPointerLeave={scheduleProjectPreviewClose}
                          onFocus={(event) => {
                            if (group.project)
                              showProjectPreview(group.project, event.currentTarget);
                          }}
                          onBlur={scheduleProjectPreviewClose}
                        >
                          <button
                            type="button"
                            className="project-group__trigger"
                            title={group.cwd || group.label}
                            onClick={() => {
                              if (group.project) newSession(group.project);
                            }}
                          >
                            <Folder aria-hidden="true" />
                            <span>{group.label}</span>
                          </button>
                          {group.project && (
                            <div
                              className="project-group__actions"
                              ref={
                                projectActionMenuId === group.project.id
                                  ? projectActionMenuRef
                                  : undefined
                              }
                            >
                              <button
                                type="button"
                                aria-label={`Options for ${group.label}`}
                                title={`Options for ${group.label}`}
                                onClick={() => {
                                  cancelProjectPreviewClose();
                                  setProjectPreview(null);
                                  setProjectHeaderMenu(null);
                                  setProjectActionMenuId((id) =>
                                    id === group.project?.id ? null : (group.project?.id ?? null),
                                  );
                                }}
                              >
                                <MoreHorizontal aria-hidden="true" />
                              </button>
                              <button
                                type="button"
                                aria-label={`New task in ${group.label}`}
                                title={`New task in ${group.label}`}
                                onClick={() => {
                                  setProjectPreview(null);
                                  newSession(group.project ?? null);
                                }}
                              >
                                <SquarePen aria-hidden="true" />
                              </button>
                              {projectActionMenuId === group.project.id && (
                                <div
                                  className="project-list__menu project-list__menu--project"
                                  role="menu"
                                  aria-label={`Project actions for ${group.label}`}
                                >
                                  <button
                                    type="button"
                                    role="menuitem"
                                    onClick={() => void toggleProjectPinned()}
                                  >
                                    <Pin aria-hidden="true" />
                                    <span>
                                      {actionProject?.pinned ? "Unpin project" : "Pin project"}
                                    </span>
                                  </button>
                                  <button
                                    type="button"
                                    role="menuitem"
                                    onClick={() => void revealSelectedProject()}
                                  >
                                    <Folder aria-hidden="true" />
                                    <span>Reveal in Finder</span>
                                  </button>
                                  <button
                                    type="button"
                                    role="menuitem"
                                    onClick={startProjectRename}
                                  >
                                    <Pencil aria-hidden="true" />
                                    <span>Rename project</span>
                                  </button>
                                  <button
                                    type="button"
                                    role="menuitem"
                                    onClick={() => void archiveSelectedProjectTasks()}
                                  >
                                    <Archive aria-hidden="true" />
                                    <span>Archive tasks</span>
                                  </button>
                                  <button
                                    type="button"
                                    role="menuitem"
                                    onClick={() => void removeSelectedProject()}
                                  >
                                    <X aria-hidden="true" />
                                    <span>Remove</span>
                                  </button>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                      {expanded && (
                        <div className="session-group__items">
                          {group.sessions.map(renderSidebarSessionItem)}
                        </div>
                      )}
                    </section>
                  );
                })
              )}
              {visibleSessionErrors.map((error) => (
                <div key={error.harnessId} className="session-error">
                  <XCircle aria-hidden="true" />
                  <span>
                    {error.harnessLabel}: {error.message}
                  </span>
                </div>
              ))}
            </div>
            <footer className="sidebar-account-area" ref={sidebarAccountRef}>
              {accountMenuOpen && (
                <div className="sidebar-account-menu" role="menu" aria-label="Anonymous user menu">
                  <div className="sidebar-account-menu__identity">
                    <span className="sidebar-account-avatar" aria-hidden="true">
                      <User />
                    </span>
                    <span>
                      <strong>Anonymous user</strong>
                      <small>Local profile</small>
                    </span>
                  </div>
                  <div className="sidebar-account-menu__items">
                    <button type="button" role="menuitem" onClick={() => openSettings("general")}>
                      <Settings aria-hidden="true" />
                      <span>Settings</span>
                    </button>
                  </div>
                </div>
              )}
              <div className={cx("sidebar-account-row", accountMenuOpen && "is-open")}>
                <button
                  type="button"
                  className="sidebar-account-trigger"
                  aria-label="Open anonymous user menu"
                  aria-haspopup="menu"
                  aria-expanded={accountMenuOpen}
                  onClick={() => setAccountMenuOpen((open) => !open)}
                >
                  <span className="sidebar-account-avatar" aria-hidden="true">
                    <User />
                  </span>
                  <span>Anonymous user</span>
                </button>
                {updateVisible && (
                  <button
                    type="button"
                    className={cx("sidebar-update-control", updateBusy && "is-busy")}
                    data-phase={desktopUpdate.phase}
                    aria-label={updateAccessibleLabel}
                    title={updateTitle}
                    disabled={updateBusy}
                    onClick={() => void startDesktopUpdate()}
                  >
                    <Download aria-hidden="true" />
                    <span aria-live="polite">{updateLabel}</span>
                  </button>
                )}
              </div>
            </footer>
          </>
        )}
      </aside>

      <main className={cx("runtime", rightPanelMounted && "runtime--right-panel")}>
        <div className={cx("runtime__body", rightPanelMounted && "runtime__body--right-panel")}>
          <div className="runtime__content">
            {pinnedSummaryMounted && (
              <div
                className={cx(
                  "panel-transition panel-transition--pinned",
                  pinnedSummaryOpen && "is-open",
                )}
                aria-hidden={!pinnedSummaryOpen}
                inert={!pinnedSummaryOpen}
              >
                <div className="panel-transition__inner">
                  <PinnedSummary
                    title={runTitle}
                    subtitle={runSubtitle}
                    status={selectedSessionLoading ? "Loading" : loading ? "Running" : "Ready"}
                    messageCount={messageCount}
                    workflowLabel={workflowBadgeLabel}
                    onClose={() => setPinnedSummaryOpen(false)}
                  />
                </div>
              </div>
            )}
            <div className="runtime__surface">
              {settingsSection === "general" ? (
                <GeneralSettings
                  status={permissionStatus}
                  loading={permissionStatusLoading}
                  error={permissionStatusError}
                  onSaveProfiles={async (profileAvailability) => {
                    await mutatePermissionStatus(
                      await api.savePermissionProfileAvailability(
                        profileAvailability,
                        permissionContext,
                      ),
                      false,
                    );
                  }}
                />
              ) : settingsSection === "profile" ? (
                <ProfileWorkspace
                  summary={activityProfile}
                  loading={activityProfileLoading}
                  error={activityProfileError ? errorMessage(activityProfileError) : undefined}
                />
              ) : settingsSection === "permissions" ? (
                <PermissionsSettings
                  status={permissionStatus}
                  loading={permissionStatusLoading}
                  error={permissionStatusError}
                  projectName={selectedProject?.name}
                  agentName={activeExtensionAgent?.name}
                  onSave={async (policy) => {
                    await mutatePermissionStatus(
                      await api.savePersonalPermissionPolicy(policy, permissionContext),
                      false,
                    );
                  }}
                />
              ) : settingsSection === "providers" ? (
                <SettingsWorkspace
                  providers={extensionInventory?.providers ?? []}
                  modelCatalog={extensionInventory?.modelCatalog}
                  modelCatalogRefreshing={modelCatalogRefreshing}
                  modelCatalogError={modelCatalogError}
                  providerUsage={providerUsage}
                  providerUsageRefreshing={providerUsageRefreshing}
                  providerUsageRefreshingIds={providerUsageRefreshingIds}
                  providerUsageError={providerUsageError}
                  onRefreshModels={refreshModelCatalog}
                  onRefreshUsage={refreshProviderUsage}
                  onSaveProvider={saveProvider}
                  onRemoveProvider={removeProvider}
                  onResetProviderKey={resetProviderKey}
                />
              ) : settingsSection === "extensions" ? (
                <ExtensionWorkspace
                  inventory={extensionInventory}
                  management={extensionManagement}
                  loading={extensionInventoryLoading}
                  error={extensionInventoryError ?? extensionManagementError}
                  selectedAgentId={selectedExtensionAgentId}
                  onSelectAgent={selectExtensionAgentForRun}
                  onSaveSource={async (input) => {
                    await mutateExtensionManagement(await api.saveExtensionSource(input), false);
                  }}
                  onRefreshSource={async (id) => {
                    await mutateExtensionManagement(await api.refreshExtensionSource(id), false);
                  }}
                  onRemoveSource={async (id) => {
                    await mutateExtensionManagement(await api.removeExtensionSource(id), false);
                  }}
                  onApplyAction={async (input) => {
                    const result = await api.applyExtensionAction(input);
                    await mutateExtensionManagement(result.state, false);
                    await mutateExtensionInventory();
                    return result.receipt;
                  }}
                  onSaveEvolutionPolicy={async (input) => {
                    await mutateExtensionManagement(
                      await api.saveSkillEvolutionPolicy(input),
                      false,
                    );
                  }}
                />
              ) : settingsSection === "agents" ? (
                <CustomAgentsSettings
                  inventory={extensionInventory}
                  environment={harnessEnvironment}
                  onSave={async (input) => {
                    await mutateExtensionInventory(await api.saveCustomAgent(input), false);
                  }}
                  onRemove={async (id) => {
                    await mutateExtensionInventory(await api.removeCustomAgent(id), false);
                  }}
                  onSetupSoftware={async (harnessId) => {
                    const result = await api.setupHarnessEnvironment({ harnessId });
                    await mutateHarnessEnvironment(result.status, false);
                  }}
                />
              ) : settingsSection === "runtime" ? (
                <RuntimeSettings
                  environment={harnessEnvironment}
                  loading={harnessEnvironmentLoading}
                  error={harnessEnvironmentError}
                  doctorReport={doctorReport}
                  doctorLoading={doctorLoading}
                  doctorError={doctorError}
                  harnessVersions={doctorHarnessVersions}
                  fixPending={doctorFixPending}
                  fixRunning={doctorFixRunning}
                  fixResult={doctorFixResult}
                  installingHarnessId={doctorInstallingHarnessId}
                  onRefresh={async () => {
                    await refreshRuntimeDoctor(true);
                  }}
                  onSetupContainer={async (containerRuntimeId) => {
                    const result = await api.setupHarnessEnvironment({
                      containerRuntimeId,
                      includeContainerRuntime: true,
                    });
                    await mutateHarnessEnvironment(result.status, false);
                    await refreshRuntimeDoctor();
                  }}
                  onInstallHarness={installDoctorHarness}
                  onRefreshHarnessVersion={(harnessId) => {
                    void checkDoctorHarnessVersion(harnessId, true);
                  }}
                  onRequestFix={() => setDoctorFixPending(true)}
                  onCancelFix={() => setDoctorFixPending(false)}
                  onConfirmFix={() => void confirmDoctorFix()}
                />
              ) : activeUiContribution && ActiveUiContributionComponent ? (
                <GuiContributionWorkspace
                  contribution={activeUiContribution}
                  inventory={extensionInventory}
                  component={ActiveUiContributionComponent}
                  onSelectAgent={selectExtensionAgentForRun}
                />
              ) : workflowPanelOpen ? (
                <WorkflowWorkspace
                  workflowJson={workflowJson}
                  onWorkflowJsonChange={setWorkflowJson}
                  workflowEnabled={workflowEnabled}
                  onWorkflowEnabledChange={setWorkflowEnabled}
                  workflowImportStatus={workflowImportStatus}
                  workflowState={workflowState}
                  input={input}
                  onInputChange={setInput}
                  onExecute={sendMessage}
                  onImportN8nFile={importN8nWorkflowFile}
                  loading={loading}
                  messages={currentSession?.messages ?? []}
                  activeWorkflowConfig={activeWorkflowConfig}
                />
              ) : (
                <div ref={chatRef} className="transcript-scroll">
                  <div className={cx("transcript", emptyRun && "transcript--empty")}>
                    {emptyRun ? (
                      <EmptyRun
                        projectLabel={emptyProjectLabel}
                        rightPanelOpen={activeRightPanelKind !== null}
                        onSelectPrompt={(prompt) => {
                          setInput(prompt);
                          window.requestAnimationFrame(() => composerRef.current?.focus());
                        }}
                      />
                    ) : (
                      <ConversationHistory
                        messages={currentSession?.messages ?? []}
                        running={loading}
                      />
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
          {rightPanelMounted && (
            <div
              className={cx(
                "panel-transition panel-transition--right",
                activeRightPanelKind && "is-open",
              )}
              aria-hidden={activeRightPanelKind === null}
              inert={activeRightPanelKind === null}
            >
              {displayedRightPanelKind === "doctor" ? (
                <DoctorPanel
                  mode={doctorPanelMode}
                  report={doctorReport}
                  loading={doctorLoading}
                  harnessOptions={
                    doctorHarnessId
                      ? HARNESSES.filter((harness) => harness.id === doctorHarnessId)
                      : HARNESSES
                  }
                  harnessVersions={doctorHarnessVersions}
                  error={doctorError}
                  fixPending={doctorFixPending}
                  fixRunning={doctorFixRunning}
                  fixResult={doctorFixResult}
                  onRefresh={() =>
                    openDoctorPanel({
                      mode: doctorPanelMode,
                      harnessId: doctorHarnessId ?? undefined,
                    })
                  }
                  onRequestFix={() => setDoctorFixPending(true)}
                  onCancelFix={() => setDoctorFixPending(false)}
                  onConfirmFix={confirmDoctorFix}
                  installingHarnessId={doctorInstallingHarnessId}
                  onInstallHarness={installDoctorHarness}
                  onRefreshHarnessVersion={(harnessId) => {
                    void checkDoctorHarnessVersion(harnessId, true);
                  }}
                  onClose={() => setDoctorPanelOpen(false)}
                />
              ) : (
                <WorkspacePanel
                  api={api}
                  cwd={composerWorkspaceRoot || "."}
                  onClose={() => setRightPanelOpen(false)}
                />
              )}
            </div>
          )}
        </div>

        {!settingsSection && !workflowPanelOpen && !activeUiContribution && (
          <footer className="composer-dock">
            <Composer
              textareaRef={composerRef}
              value={input}
              onChange={setInput}
              onSubmit={sendMessage}
              onStop={stopMessage}
              placeholder={
                acpHistoryReadOnly
                  ? "ACP history is read-only until resume is supported"
                  : `Message ${activeExtensionAgent?.name ?? displayHarness.label}`
              }
              disabled={runState !== "idle" || acpHistoryReadOnly}
              running={runState !== "idle"}
              sendDisabled={
                runState === "stopping" ||
                (runState === "idle" &&
                  (!input.trim() || manualCompositionNeedsModel || acpHistoryReadOnly))
              }
              sendTitle={
                acpHistoryReadOnly
                  ? "ACP history is read-only until session resume is supported."
                  : manualCompositionNeedsModel
                    ? modelUnavailableDiagnostic
                    : undefined
              }
              error={composerError}
              workspaceRoot={composerWorkspaceRoot}
              mentionServers={composerMentionServers}
              completeMention={api.lspComplete}
              selectFilesAndFolders={api.selectFilesAndFolders}
              onContextError={(error) => setComposerError(errorMessage(error))}
            >
              <ConversationPermissionPicker
                open={permissionPickerOpen}
                mode={sessionPermissionMode}
                supported={sessionPermissionSupported}
                profileAvailability={permissionStatus?.profileAvailability}
                disabled={runState !== "idle" || acpHistoryReadOnly}
                onOpenChange={(open) => {
                  setPermissionPickerOpen(open);
                  if (open) setAgentPickerOpen(false);
                }}
                onChange={changeSessionPermissionMode}
              />
              <AgentPicker
                open={agentPickerOpen}
                section={agentPickerSection}
                harnesses={availableHarnesses}
                selectedHarness={displayHarness}
                models={availableModels}
                selectedModel={selectedModel}
                efforts={availableEfforts}
                selectedEffort={selectedEffort}
                modelStatusText={modelStatusText}
                modelCatalog={extensionInventory?.modelCatalog}
                modelCatalogRefreshing={modelCatalogRefreshing}
                modelCatalogError={modelCatalogError}
                disabled={Boolean(activeWorkflowConfig || activeExtensionAgent)}
                label={agentPickerLabel}
                onOpenChange={(open) => {
                  setAgentPickerOpen(open);
                  if (open) setPermissionPickerOpen(false);
                }}
                onSectionChange={setAgentPickerSection}
                onHarnessChange={(harnessId) => {
                  setSelectedHarness(harnessId);
                  setSelectedExtensionAgentId(null);
                  setSelectedModelId(null);
                  setSelectedEffort(null);
                }}
                onModelChange={(modelId) => {
                  setSelectedModelId(modelId);
                  setSelectedEffort(null);
                }}
                onEffortChange={setSelectedEffort}
                onRefreshModels={refreshModelCatalog}
                onAddManualModel={addManualModel}
                onRemoveManualModel={removeManualModel}
              />
            </Composer>
          </footer>
        )}

        <div
          className={cx("panel-transition panel-transition--bottom", bottomPanelOpen && "is-open")}
          aria-hidden={!bottomPanelOpen}
          inert={!bottomPanelOpen}
        >
          <div className="panel-transition__inner">
            <RuntimeBottomPanel
              key={composerWorkspaceRoot || "."}
              active={bottomPanelOpen}
              cwd={composerWorkspaceRoot || "."}
              onClose={() => setBottomPanelOpen(false)}
            />
          </div>
        </div>
      </main>
      {activeAgentInteraction
        ? createPortal(
            <AgentInteractionDialog
              key={activeAgentInteraction.interactionId}
              interaction={activeAgentInteraction}
              resolving={resolvingInteractionId === activeAgentInteraction.interactionId}
              error={agentInteractionError}
              onResolve={(response) =>
                void resolveAgentInteraction(activeAgentInteraction, response)
              }
              onStop={() => void stopMessage()}
            />,
            document.body,
          )
        : null}
      {sessionContextMenu
        ? createPortal(
            <div
              ref={sessionContextMenuRef}
              className="session-context-menu"
              role="menu"
              aria-label={`Task actions for ${sessionContextMenu.session.title}`}
              style={{ left: sessionContextMenu.x, top: sessionContextMenu.y }}
            >
              <button
                type="button"
                role="menuitem"
                disabled={sessionActionPending}
                onClick={() => void toggleSessionPinned()}
              >
                <Pin aria-hidden="true" />
                <span>{sessionContextMenu.session.pinned ? "Unpin task" : "Pin task"}</span>
              </button>
              <button
                type="button"
                role="menuitem"
                disabled={sessionActionPending}
                onClick={() => openSessionRename(sessionContextMenu.session)}
              >
                <Pencil aria-hidden="true" />
                <span>Rename task</span>
              </button>
              <div className="session-context-menu__separator" />
              <button
                type="button"
                role="menuitem"
                className="is-danger"
                disabled={sessionActionPending}
                onClick={() => void deleteSidebarSession()}
              >
                <Trash2 aria-hidden="true" />
                <span>Delete task</span>
              </button>
            </div>,
            document.body,
          )
        : null}
      {renamingSession
        ? createPortal(
            <div
              className="session-rename-backdrop"
              onMouseDown={(event) => {
                if (event.target === event.currentTarget && !sessionActionPending) {
                  setRenamingSession(null);
                }
              }}
            >
              <dialog
                open
                className="session-rename-dialog"
                aria-modal="true"
                aria-labelledby="session-rename-title"
              >
                <header>
                  <h2 id="session-rename-title">Rename task</h2>
                  <button
                    type="button"
                    aria-label="Close rename task dialog"
                    disabled={sessionActionPending}
                    onClick={() => setRenamingSession(null)}
                  >
                    <X aria-hidden="true" />
                  </button>
                </header>
                <p>Keep it short and recognizable</p>
                <form
                  onSubmit={(event) => {
                    event.preventDefault();
                    void commitSessionRename();
                  }}
                >
                  <input
                    ref={sessionRenameInputRef}
                    value={sessionRenameDraft}
                    maxLength={60}
                    aria-label="Task title"
                    disabled={sessionActionPending}
                    onChange={(event) => {
                      setSessionRenameDraft(event.target.value);
                      setSessionActionError(null);
                    }}
                  />
                  {sessionActionError && (
                    <div className="session-rename-dialog__error" role="alert">
                      {sessionActionError}
                    </div>
                  )}
                  <footer>
                    <button
                      type="button"
                      disabled={sessionActionPending}
                      onClick={() => setRenamingSession(null)}
                    >
                      Cancel
                    </button>
                    <button type="submit" className="is-primary" disabled={sessionActionPending}>
                      {sessionActionPending ? "Saving…" : "Save"}
                    </button>
                  </footer>
                </form>
              </dialog>
            </div>,
            document.body,
          )
        : null}
      {projectPreview && previewProject
        ? createPortal(
            <dialog
              open
              className="project-preview-card"
              aria-label={`${previewProject.name} project details`}
              style={{ top: projectPreview.top, left: projectPreview.left }}
              onPointerEnter={cancelProjectPreviewClose}
              onPointerLeave={scheduleProjectPreviewClose}
              onFocus={cancelProjectPreviewClose}
              onBlur={scheduleProjectPreviewClose}
            >
              <div className="project-preview-card__row project-preview-card__row--title">
                <Folder aria-hidden="true" />
                <strong>{previewProject.name}</strong>
                <button
                  type="button"
                  aria-label={`${previewProject.pinned ? "Unpin" : "Pin"} ${previewProject.name}`}
                  title={`${previewProject.pinned ? "Unpin" : "Pin"} project`}
                  disabled={projectActionPending}
                  onClick={() => void togglePreviewProjectPinned(previewProject)}
                >
                  <Pin aria-hidden="true" />
                </button>
              </div>
              <div className="project-preview-card__row">
                <MessageCircle aria-hidden="true" />
                <span>
                  {previewProjectGroup?.sessions.length ?? 0}{" "}
                  {(previewProjectGroup?.sessions.length ?? 0) === 1 ? "thread" : "threads"}
                </span>
              </div>
              <div className="project-preview-card__separator" />
              <div className="project-preview-card__row project-preview-card__row--path">
                <Folder aria-hidden="true" />
                <span>{abbreviateHomePath(previewProject.cwd)}</span>
              </div>
            </dialog>,
            document.body,
          )
        : null}
    </div>
  );
}

function SettingsSidebar({
  section,
  query,
  onQueryChange,
  onSectionChange,
  onBack,
}: {
  section: SettingsSection;
  query: string;
  onQueryChange: (query: string) => void;
  onSectionChange: (section: SettingsSection) => void;
  onBack: () => void;
}) {
  const normalizedQuery = query.trim().toLowerCase();
  const personalSections = [
    { id: "general" as const, label: "General", icon: Settings },
    { id: "profile" as const, label: "Profile", icon: User },
  ].filter((item) => item.label.toLowerCase().includes(normalizedQuery));
  const systemSections = [
    { id: "permissions" as const, label: "Advanced permissions", icon: ShieldCheck },
    { id: "providers" as const, label: "Providers", icon: KeyRound },
    { id: "extensions" as const, label: "Extensions", icon: Package },
    { id: "agents" as const, label: "Custom Agents", icon: Bot },
    { id: "runtime" as const, label: "Runtime", icon: TerminalIcon },
  ].filter((item) => item.label.toLowerCase().includes(normalizedQuery));
  const renderSections = (
    label: string,
    sections: Array<{ id: SettingsSection; label: string; icon: LucideIcon }>,
  ) =>
    sections.length > 0 ? (
      <>
        <span className="settings-sidebar__group-label">{label}</span>
        {sections.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              type="button"
              className={section === item.id ? "is-active" : undefined}
              aria-current={section === item.id ? "page" : undefined}
              onClick={() => onSectionChange(item.id)}
            >
              <Icon aria-hidden="true" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </>
    ) : null;

  return (
    <div className="settings-sidebar">
      <button type="button" className="settings-sidebar__back" onClick={onBack}>
        <ArrowLeft aria-hidden="true" />
        <span>Back to app</span>
      </button>
      <label className="sidebar-search settings-sidebar__search">
        <Search aria-hidden="true" />
        <input
          type="search"
          value={query}
          placeholder="Search settings"
          aria-label="Search settings"
          onChange={(event) => onQueryChange(event.target.value)}
        />
      </label>
      <nav className="settings-sidebar__sections" aria-label="Settings sections">
        {renderSections("Personal", personalSections)}
        {renderSections("System", systemSections)}
        {personalSections.length === 0 && systemSections.length === 0 && (
          <span className="settings-sidebar__empty">No matching settings</span>
        )}
      </nav>
    </div>
  );
}

const PERMISSION_TOOL_SUGGESTIONS = [
  "Read",
  "Glob",
  "Grep",
  "LSP",
  "Edit",
  "Write",
  "apply_patch",
  "Bash",
  "exec_command",
  "Task",
  "WebFetch",
  "WebSearch",
];

const PERMISSION_MODE_OPTIONS: Array<{
  id: HarnessPermissionMode;
  label: string;
  description: string;
}> = [
  {
    id: "plan",
    label: "Plan only",
    description: "Read-only tools can run; writes and commands are denied.",
  },
  {
    id: "restricted",
    label: "Restricted",
    description: "Only read-only and explicitly pre-approved tools can run.",
  },
  {
    id: "default",
    label: "Ask for approval",
    description: "Read-only tools run; each write or command needs one-time approval.",
  },
  {
    id: "auto",
    label: "Auto-review",
    description: "Read and Project writes run; commands and control actions still ask once.",
  },
  {
    id: "trusted",
    label: "Full tool access",
    description: "Tools run without prompts, while the host OS sandbox still applies.",
  },
];

const GENERAL_PERMISSION_MODE_OPTIONS = [
  {
    id: "default" as const,
    label: "Default permissions",
    description:
      "By default, SwarmX can read and edit files in its Project. It asks for additional access when needed.",
  },
  {
    id: "auto" as const,
    label: "Auto-review",
    description:
      "SwarmX automatically approves lower-risk Project changes. Commands and control actions can still ask.",
  },
  {
    id: "trusted" as const,
    label: "Full access",
    description:
      "SwarmX can edit files and run commands without approval. This increases the risk of data loss or unexpected changes.",
  },
];

function GeneralSettings({
  status,
  loading,
  error,
  onSaveProfiles,
}: {
  status?: DesktopPermissionStatus;
  loading: boolean;
  error: unknown;
  onSaveProfiles: (
    profileAvailability: DesktopPermissionStatus["profileAvailability"],
  ) => Promise<void>;
}) {
  const [savingMode, setSavingMode] = useState<
    keyof DesktopPermissionStatus["profileAvailability"] | null
  >(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const toggleProfile = async (mode: keyof DesktopPermissionStatus["profileAvailability"]) => {
    if (!status || savingMode) return;
    setSavingMode(mode);
    setSaveError(null);
    try {
      await onSaveProfiles({
        ...status.profileAvailability,
        [mode]: !status.profileAvailability[mode],
      });
    } catch (saveFailure) {
      setSaveError(errorMessage(saveFailure));
    } finally {
      setSavingMode(null);
    }
  };

  if (loading && !status) {
    return (
      <section className="settings-workspace general-settings" aria-label="General settings">
        <div className="settings-workspace__loading">
          <Loader2 className="is-spinning" aria-hidden="true" /> Loading General settings…
        </div>
      </section>
    );
  }

  return (
    <section className="settings-workspace general-settings" aria-label="General settings">
      <div className="settings-workspace__body">
        <div className="settings-workspace__content general-settings__content">
          <div className="general-settings__heading">
            <h2>General</h2>
          </div>

          {Boolean(saveError || error) && (
            <div className="settings-provider-error">{saveError ?? errorMessage(error)}</div>
          )}

          <section className="general-settings__section" aria-labelledby="general-permissions">
            <h3 id="general-permissions">Permissions</h3>
            <fieldset className="general-permission-card" disabled={!status || Boolean(savingMode)}>
              <legend className="sr-only">Available permission profiles</legend>
              {GENERAL_PERMISSION_MODE_OPTIONS.map((option) => {
                const enabled = status?.profileAvailability[option.id] ?? false;
                return (
                  <label key={option.id}>
                    <input
                      type="checkbox"
                      role="switch"
                      aria-checked={enabled}
                      value={option.id}
                      checked={enabled}
                      onChange={() => void toggleProfile(option.id)}
                    />
                    <span className="general-permission-card__copy">
                      <strong>{option.label}</strong>
                      <small>{option.description}</small>
                    </span>
                    <span
                      className={cx(
                        "general-permission-card__switch",
                        enabled && "is-enabled",
                        savingMode === option.id && "is-saving",
                      )}
                      aria-hidden="true"
                    >
                      {savingMode === option.id ? <Loader2 className="is-spinning" /> : <span />}
                    </span>
                  </label>
                );
              })}
            </fieldset>
          </section>
        </div>
      </div>
    </section>
  );
}

function PermissionsSettings({
  status,
  loading,
  error,
  projectName,
  agentName,
  onSave,
}: {
  status?: DesktopPermissionStatus;
  loading: boolean;
  error: unknown;
  projectName?: string;
  agentName?: string;
  onSave: (policy: unknown) => Promise<void>;
}) {
  const [allowedTools, setAllowedTools] = useState<string[]>([]);
  const [deniedTools, setDeniedTools] = useState<string[]>([]);
  const [mode, setMode] = useState<HarnessPermissionMode>("default");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    if (!status) return;
    setAllowedTools(status.personalPolicy.allowedTools);
    setDeniedTools(status.personalPolicy.deniedTools);
    setMode(status.personalPolicy.mode ?? "default");
    setSaveError(null);
  }, [status]);

  const save = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      await onSave({
        mode,
        allowedTools,
        deniedTools,
      });
    } catch (saveFailure) {
      setSaveError(errorMessage(saveFailure));
    } finally {
      setSaving(false);
    }
  };

  if (loading && !status) {
    return (
      <section
        className="settings-workspace permission-settings"
        aria-label="Advanced permissions settings"
      >
        <div className="settings-workspace__loading">
          <Loader2 className="is-spinning" aria-hidden="true" /> Loading permission policy…
        </div>
      </section>
    );
  }

  if (error && !status) {
    return (
      <section
        className="settings-workspace permission-settings"
        aria-label="Advanced permissions settings"
      >
        <div className="settings-provider-error">{errorMessage(error)}</div>
      </section>
    );
  }

  const effective = status?.effective;
  const effectiveMode = effective?.policy.mode ?? "default";
  const effectiveModeLabel = permissionModeLabel(effectiveMode);

  return (
    <section
      className="settings-workspace permission-settings"
      aria-label="Advanced permissions settings"
    >
      <div className="settings-workspace__body">
        <div className="settings-workspace__content permission-settings__content">
          <div className="settings-content-heading permission-settings__heading">
            <span>
              <small>Exact rules and audit</small>
              <h2>Advanced permissions</h2>
              <p>
                Review effective authority, configure exact tool rules, and inspect one-call
                decisions. Profile availability lives in General.
              </p>
            </span>
            <button
              type="button"
              className="settings-primary-action"
              disabled={saving}
              onClick={() => void save()}
            >
              {saving ? "Saving…" : "Save policy"}
            </button>
          </div>

          {Boolean(saveError || error) && (
            <div className="settings-provider-error">{saveError ?? errorMessage(error)}</div>
          )}

          <section className="permission-fallback" aria-labelledby="permission-fallback-title">
            <span>
              <h3 id="permission-fallback-title">Inherited fallback</h3>
              <p>
                This is what <strong>Use default</strong> means for direct SwarmX conversations.
                Plan only and Restricted remain available here as conservative profiles.
              </p>
            </span>
            <label>
              <span>Default mode</span>
              <select
                value={mode}
                onChange={(event) => setMode(event.target.value as HarnessPermissionMode)}
              >
                {PERMISSION_MODE_OPTIONS.map((option) => (
                  <option
                    key={option.id}
                    value={option.id}
                    disabled={
                      (option.id === "default" ||
                        option.id === "auto" ||
                        option.id === "trusted") &&
                      !status?.profileAvailability[option.id]
                    }
                  >
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          </section>

          <section
            className={cx(
              "permission-effective",
              status?.blocked && "permission-effective--blocked",
            )}
            aria-label="Effective permission policy"
          >
            <div className="permission-effective__icon">
              {status?.blocked ? (
                <XCircle aria-hidden="true" />
              ) : (
                <ShieldCheck aria-hidden="true" />
              )}
            </div>
            <div className="permission-effective__copy">
              <small>
                {status?.blocked
                  ? "Execution blocked"
                  : agentName
                    ? `Effective policy · ${agentName}`
                    : "Effective policy · before Agent"}
              </small>
              <h3>{status?.blocked ? "A policy source needs attention" : effectiveModeLabel}</h3>
              <p>
                {status?.blocked
                  ? "Malformed managed or Project policy fails closed until its source is corrected."
                  : permissionModeDescription(effectiveMode)}
              </p>
            </div>
            {!status?.blocked && effective && (
              <dl className="permission-effective__stats">
                <div>
                  <dt>Pre-approved</dt>
                  <dd>{effective.policy.allowedTools.length}</dd>
                </div>
                <div>
                  <dt>Denied</dt>
                  <dd>{effective.policy.deniedTools.length}</dd>
                </div>
                <div>
                  <dt>Sources</dt>
                  <dd>{effective.layers.length}</dd>
                </div>
              </dl>
            )}
          </section>

          <section className="permission-panel" aria-labelledby="personal-permission-heading">
            <div className="permission-panel__heading">
              <span>
                <small>Editable on this device</small>
                <h3 id="personal-permission-heading">Exact tool rules</h3>
                <p>
                  Pre-approvals and denials combine with the default mode, managed, Project, and
                  Agent policy at execution time.
                </p>
              </span>
              <Badge tone="active">Personal</Badge>
            </div>

            <div className="permission-rule-grid">
              <PermissionToolRulesEditor
                label="Pre-approved tools"
                description="Exact tool names that may run without a prompt."
                values={allowedTools}
                blockedValues={deniedTools}
                onChange={setAllowedTools}
              />
              <PermissionToolRulesEditor
                label="Denied tools"
                description="Exact tool names that can never run. Denials always win."
                values={deniedTools}
                blockedValues={allowedTools}
                tone="danger"
                onChange={setDeniedTools}
              />
            </div>
          </section>

          <section className="permission-panel" aria-labelledby="permission-sources-heading">
            <div className="permission-panel__heading">
              <span>
                <small>Effective authority stack</small>
                <h3 id="permission-sources-heading">Policy sources</h3>
                <p>
                  {projectName
                    ? `Project policy is resolved for ${projectName}.`
                    : "Choose a Project to include its repository policy."}
                </p>
              </span>
              <code>{status?.projectPolicyPath ?? ".swarmx/permissions.json"}</code>
            </div>
            <div className="permission-layer-list">
              {status?.layers.map((layer) => (
                <article
                  key={layer.id}
                  className={cx(
                    "permission-layer-card",
                    layer.error && "permission-layer-card--error",
                  )}
                >
                  <div className="permission-layer-card__heading">
                    <span className={`permission-source permission-source--${layer.source}`}>
                      {permissionSourceLabel(layer.source)}
                    </span>
                    <Badge tone={layer.error ? "danger" : layer.configured ? "active" : "neutral"}>
                      {layer.error ? "Invalid" : layer.configured ? "Applied" : "Not configured"}
                    </Badge>
                  </div>
                  <strong>{layer.label}</strong>
                  {layer.error ? (
                    <p>{layer.error}</p>
                  ) : layer.configured ? (
                    <dl>
                      <div>
                        <dt>Mode</dt>
                        <dd>{layer.mode ? permissionModeLabel(layer.mode) : "No ceiling"}</dd>
                      </div>
                      <div>
                        <dt>Allow</dt>
                        <dd>{layer.allowedTools.length}</dd>
                      </div>
                      <div>
                        <dt>Deny</dt>
                        <dd>{layer.deniedTools.length}</dd>
                      </div>
                    </dl>
                  ) : (
                    <p>No policy was found for this source.</p>
                  )}
                </article>
              ))}
            </div>
          </section>

          <section className="permission-panel" aria-labelledby="permission-history-heading">
            <div className="permission-panel__heading">
              <span>
                <small>Sanitized local audit trail</small>
                <h3 id="permission-history-heading">Approval history</h3>
                <p>Only the tool, decision, source, and policy provenance are stored.</p>
              </span>
              <Badge>{status?.approvalReceipts.length ?? 0} receipts</Badge>
            </div>
            {(status?.approvalReceipts.length ?? 0) === 0 ? (
              <div className="permission-history-empty">
                <Clock3 aria-hidden="true" /> No approval decisions yet.
              </div>
            ) : (
              <div className="permission-history-list">
                {status?.approvalReceipts.map((receipt) => (
                  <article key={`${receipt.id}:${receipt.createdAt}`}>
                    <span
                      className={cx(
                        "permission-decision",
                        `permission-decision--${receipt.decision}`,
                      )}
                    >
                      {receipt.decision}
                    </span>
                    <span>
                      <strong>{receipt.toolName}</strong>
                      <small>
                        {receipt.source.toUpperCase()}
                        {receipt.toolKind ? ` · ${receipt.toolKind}` : ""}
                        {receipt.policySourceIds.length > 0
                          ? ` · ${receipt.policySourceIds.join(" + ")}`
                          : ""}
                      </small>
                    </span>
                    <time dateTime={receipt.createdAt}>
                      {formatPermissionTime(receipt.createdAt)}
                    </time>
                  </article>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
    </section>
  );
}

function PermissionToolRulesEditor({
  label,
  description,
  values,
  blockedValues,
  tone = "default",
  onChange,
}: {
  label: string;
  description: string;
  values: string[];
  blockedValues: string[];
  tone?: "default" | "danger";
  onChange: (values: string[]) => void;
}) {
  const listId = useId();
  const [draft, setDraft] = useState("");
  const [inputError, setInputError] = useState<string | null>(null);
  const add = () => {
    const toolName = draft.trim();
    if (!toolName) {
      setInputError("Enter an exact tool name.");
      return;
    }
    if (values.includes(toolName)) {
      setInputError(`${toolName} is already listed.`);
      return;
    }
    if (blockedValues.includes(toolName)) {
      setInputError(`${toolName} already has the opposite rule.`);
      return;
    }
    onChange([...values, toolName]);
    setDraft("");
    setInputError(null);
  };

  return (
    <div className={cx("permission-rule-editor", tone === "danger" && "is-danger")}>
      <div>
        <strong>{label}</strong>
        <p>{description}</p>
      </div>
      <div className="permission-rule-editor__input">
        <input
          list={listId}
          value={draft}
          aria-label={`${label} tool name`}
          placeholder="Type an exact tool name"
          onChange={(event) => {
            setDraft(event.target.value);
            setInputError(null);
          }}
          onKeyDown={(event) => {
            if (event.key !== "Enter") return;
            event.preventDefault();
            add();
          }}
        />
        <datalist id={listId}>
          {PERMISSION_TOOL_SUGGESTIONS.filter(
            (toolName) => !values.includes(toolName) && !blockedValues.includes(toolName),
          ).map((toolName) => (
            <option key={toolName} value={toolName} />
          ))}
        </datalist>
        <button type="button" onClick={add}>
          <Plus aria-hidden="true" /> Add
        </button>
      </div>
      {inputError && <small className="permission-rule-editor__error">{inputError}</small>}
      <div className="permission-rule-editor__chips" aria-label={`${label} rules`}>
        {values.length === 0 ? (
          <small>No exact-tool rules.</small>
        ) : (
          values.map((toolName) => (
            <span key={toolName}>
              <code>{toolName}</code>
              <button
                type="button"
                aria-label={`Remove ${toolName} from ${label}`}
                onClick={() => onChange(values.filter((value) => value !== toolName))}
              >
                <X aria-hidden="true" />
              </button>
            </span>
          ))
        )}
      </div>
    </div>
  );
}

function permissionModeLabel(mode: HarnessPermissionMode): string {
  return PERMISSION_MODE_OPTIONS.find((option) => option.id === mode)?.label ?? mode;
}

function permissionModeDescription(mode: HarnessPermissionMode): string {
  return PERMISSION_MODE_OPTIONS.find((option) => option.id === mode)?.description ?? "";
}

function permissionSourceLabel(source: HarnessPermissionPolicyLayer["source"]): string {
  if (source === "project") return "Project";
  return `${source.slice(0, 1).toUpperCase()}${source.slice(1)}`;
}

function formatPermissionTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

const CONVERSATION_PERMISSION_OPTIONS: Array<{
  id: SessionPermissionMode;
  label: string;
  shortLabel: string;
  description: string;
}> = [
  {
    id: "inherit",
    label: "Use default",
    shortLabel: "Default",
    description: "Follow General and the selected Agent's default mode.",
  },
  {
    id: "default",
    label: "Ask for approval",
    shortLabel: "Ask",
    description: "Read-only tools run; writes and commands ask once.",
  },
  {
    id: "auto",
    label: "Approve for me",
    shortLabel: "Auto",
    description: "Review Project changes automatically; commands can still ask.",
  },
  {
    id: "trusted",
    label: "Full access",
    shortLabel: "Full access",
    description: "Run without prompts inside the unchanged Project sandbox.",
  },
  {
    id: "plan",
    label: "Plan only",
    shortLabel: "Plan",
    description: "Inspect and plan without writes or commands.",
  },
];

function ConversationPermissionPicker({
  open,
  mode,
  supported,
  profileAvailability,
  disabled,
  onOpenChange,
  onChange,
}: {
  open: boolean;
  mode: SessionPermissionMode;
  supported: boolean;
  profileAvailability?: DesktopPermissionStatus["profileAvailability"];
  disabled: boolean;
  onOpenChange: (open: boolean) => void;
  onChange: (mode: SessionPermissionMode) => Promise<void>;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [savingMode, setSavingMode] = useState<SessionPermissionMode | null>(null);
  const descriptionId = useId();
  const availableOptions = CONVERSATION_PERMISSION_OPTIONS.filter((option) => {
    if (option.id === "inherit" || option.id === "plan") return true;
    return profileAvailability?.[option.id] ?? true;
  });
  const selected =
    availableOptions.find((option) => option.id === mode) ??
    CONVERSATION_PERMISSION_OPTIONS.find((option) => option.id === "plan") ??
    CONVERSATION_PERMISSION_OPTIONS[0];

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) onOpenChange(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onOpenChange(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [onOpenChange, open]);

  const selectMode = async (nextMode: SessionPermissionMode) => {
    if (savingMode) return;
    setSavingMode(nextMode);
    try {
      await onChange(nextMode);
      onOpenChange(false);
    } catch {
      // The owning Composer surfaces the persistence error without closing this menu.
    } finally {
      setSavingMode(null);
    }
  };

  return (
    <div className="conversation-permission-picker" ref={rootRef}>
      <button
        type="button"
        className="conversation-permission-picker__trigger"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-describedby={descriptionId}
        disabled={disabled || !supported}
        title={
          supported
            ? "Set permissions for this conversation"
            : "External ACP Harnesses keep their native permission controls."
        }
        onClick={() => onOpenChange(!open)}
      >
        <ShieldCheck aria-hidden="true" />
        <span>{supported ? selected?.shortLabel : "Harness managed"}</span>
        <ChevronDown aria-hidden="true" />
      </button>
      <span id={descriptionId} className="sr-only">
        {supported
          ? "This selection applies to this conversation only."
          : "External ACP Harnesses keep their native permission controls."}
      </span>
      {open && supported && (
        <section
          className="conversation-permission-picker__menu"
          role="menu"
          aria-label="Conversation permissions"
        >
          <div className="conversation-permission-picker__options">
            {availableOptions.map((option) => (
              <button
                key={option.id}
                type="button"
                role="menuitemradio"
                aria-checked={mode === option.id}
                className={cx(
                  mode === option.id && "is-selected",
                  option.id === "trusted" && "is-trusted",
                  option.id === "plan" && "is-secondary",
                )}
                disabled={Boolean(savingMode)}
                onClick={() => void selectMode(option.id)}
              >
                <span className="conversation-permission-picker__check">
                  {savingMode === option.id ? (
                    <Loader2 className="is-spinning" aria-hidden="true" />
                  ) : mode === option.id ? (
                    <Check aria-hidden="true" />
                  ) : null}
                </span>
                <span>
                  <strong>{option.label}</strong>
                  <small>{option.description}</small>
                </span>
              </button>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function CustomAgentsSettings({
  inventory,
  environment,
  onSave,
  onRemove,
  onSetupSoftware,
}: {
  inventory?: ExtensionCapabilityInventory;
  environment?: HarnessEnvironmentStatus;
  onSave: (input: unknown) => Promise<void>;
  onRemove: (id: string) => Promise<void>;
  onSetupSoftware: (harnessId: string) => Promise<void>;
}) {
  const agents = inventory?.agents ?? [];
  const customAgents = agents.filter((agent) => Boolean(agent.harnessRecipe) && !agent.readOnly);
  const nativeAgents = agents.filter(
    (agent) => !agent.harnessRecipe && Boolean(agent.definition?.host),
  );
  const extensionAgents = agents.filter((agent) => !agent.harnessRecipe && !agent.definition?.host);
  const customHarnessIds = new Set(
    agents.flatMap((agent) => (agent.harnessRecipe ? [agent.harnessRecipe.id] : [])),
  );
  const softwareOptions = (inventory?.harnesses ?? []).filter(
    (harness) => !customHarnessIds.has(harness.id),
  );
  const skills = uniqueById(inventory?.skills ?? []);
  const mcpServers = uniqueById(inventory?.mcpServers ?? []);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [softwareId, setSoftwareId] = useState(softwareOptions[0]?.id ?? "");
  const [modelOptionId, setModelOptionId] = useState("");
  const [instructions, setInstructions] = useState("");
  const [skillModes, setSkillModes] = useState<
    Record<string, { mode: "off" | "auto" | "required"; variantId?: string }>
  >({});
  const [selectedMcps, setSelectedMcps] = useState<ReadonlySet<string>>(new Set());
  const [contextPaths, setContextPaths] = useState("");
  const [instructionFiles, setInstructionFiles] = useState("AGENTS.md");
  const [permissionMode, setPermissionMode] = useState<HarnessPermissionMode>("default");
  const [allowedTools, setAllowedTools] = useState<string[]>([]);
  const [deniedTools, setDeniedTools] = useState<string[]>([]);
  const [unsupportedSkill, setUnsupportedSkill] = useState<"block" | "skip">("block");
  const [saving, setSaving] = useState(false);
  const [setupBusy, setSetupBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const modelOptions = useMemo(() => {
    if (!inventory || !softwareId) return [];
    try {
      return resolveComposerModelOptions(inventory, softwareId);
    } catch {
      return [];
    }
  }, [inventory, softwareId]);
  const modelGroups = useMemo(() => groupComposerModels(modelOptions, ""), [modelOptions]);
  const defaultModelOptionId = modelGroups[0]?.subgroups[0]?.models[0]?.id ?? "";
  const selectedModelOption = modelOptions.find((model) => model.id === modelOptionId);
  const editingAgent = customAgents.find((agent) => agent.id === editingId);
  const unavailableEditingModel =
    editingAgent?.modelId && !selectedModelOption
      ? {
          id: modelOptionId,
          label: `${editingAgent.modelId} (unavailable)`,
        }
      : undefined;

  useEffect(() => {
    if (!softwareId && softwareOptions[0]) {
      setSoftwareId(softwareOptions[0].id);
      return;
    }
    if (modelOptionId && modelOptions.some((model) => model.id === modelOptionId)) return;
    if (editingId && modelOptionId) return;
    setModelOptionId(defaultModelOptionId);
  }, [defaultModelOptionId, editingId, modelOptionId, modelOptions, softwareId, softwareOptions]);

  const reset = () => {
    setEditingId(null);
    setName("");
    setSoftwareId(softwareOptions[0]?.id ?? "");
    setModelOptionId("");
    setInstructions("");
    setSkillModes({});
    setSelectedMcps(new Set());
    setContextPaths("");
    setInstructionFiles("AGENTS.md");
    setPermissionMode("default");
    setAllowedTools([]);
    setDeniedTools([]);
    setUnsupportedSkill("block");
    setFormError(null);
  };
  const edit = (agent: ExtensionAgentSummary) => {
    const recipe = agent.harnessRecipe;
    if (!recipe) return;
    setEditingId(agent.id);
    setName(agent.name);
    setSoftwareId(recipe.softwareId);
    setModelOptionId(
      agent.modelId
        ? composerModelOptionId(recipe.softwareId, agent.modelId, agent.modelSupplyId)
        : "",
    );
    setInstructions(agent.instructions ?? "");
    setSkillModes(
      Object.fromEntries(
        recipe.skillBindings.map((binding) => [
          binding.skillId,
          { mode: binding.mode, variantId: binding.variantId },
        ]),
      ),
    );
    setSelectedMcps(new Set(recipe.mcpServerIds));
    setContextPaths(recipe.projectContext.paths.join("\n"));
    setInstructionFiles(recipe.projectContext.instructionFiles.join("\n"));
    setPermissionMode(recipe.permissions.mode);
    setAllowedTools(recipe.permissions.allowedTools);
    setDeniedTools(recipe.permissions.deniedTools);
    setUnsupportedSkill(recipe.delivery.unsupportedSkill);
    setFormError(null);
  };
  const selectedSoftware = softwareOptions.find((harness) => harness.id === softwareId);
  const softwareHealth = environment?.harnesses.find((harness) => harness.harnessId === softwareId);
  const selectedSkillRows = skills.filter((skill) => skillModes[skill.id]?.mode !== undefined);
  const contextTokenEstimate = selectedSkillRows.reduce((total, skill) => {
    const selectedVariant = skill.variants?.find(
      (variant) => variant.id === skillModes[skill.id]?.variantId,
    );
    return total + (selectedVariant?.tokenEstimate ?? skill.tokenEstimate ?? 0);
  }, 0);
  const unknownTokenCount = selectedSkillRows.filter((skill) => {
    const selectedVariant = skill.variants?.find(
      (variant) => variant.id === skillModes[skill.id]?.variantId,
    );
    return selectedVariant?.tokenEstimate === undefined && skill.tokenEstimate === undefined;
  }).length;

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedModelOption) {
      setFormError("Choose an available Model before saving this Agent.");
      return;
    }
    const id = editingId ?? slugId(name, "custom-agent");
    const harnessId = `${id}-harness`;
    setSaving(true);
    setFormError(null);
    try {
      await onSave({
        id,
        name: name.trim(),
        harnessId,
        harnessRecipe: {
          id: harnessId,
          revisionId: "draft",
          name: `${name.trim()} Harness`,
          softwareId,
          skillBindings: Object.entries(skillModes).map(([skillId, binding]) => ({
            skillId,
            mode: binding.mode,
            ...(binding.variantId ? { variantId: binding.variantId } : {}),
          })),
          mcpServerIds: [...selectedMcps],
          projectContext: {
            paths: lines(contextPaths),
            instructionFiles: lines(instructionFiles),
            includeWorkspaceRules: true,
          },
          delivery: {
            unsupportedSkill,
            requireContentDigest: true,
            allowHostNativePlugins: true,
          },
          permissions: {
            mode: permissionMode,
            allowedTools,
            deniedTools,
          },
        },
        modelId: selectedModelOption.modelId,
        ...(selectedModelOption.modelSupplyId
          ? { modelSupplyId: selectedModelOption.modelSupplyId }
          : {}),
        instructions: instructions.trim() || undefined,
        skills: Object.entries(skillModes)
          .filter(([, binding]) => binding.mode !== "off")
          .map(([skillId]) => skillId),
        mcpServers: [...selectedMcps],
        permissionMode,
        enabled: true,
        readOnly: false,
      });
      reset();
    } catch (error) {
      setFormError(errorMessage(error));
    } finally {
      setSaving(false);
    }
  };

  return (
    <section
      className="settings-workspace custom-agents-settings"
      aria-label="Custom Agents settings"
    >
      <div className="settings-workspace__body custom-agent-layout">
        <aside className="custom-agent-list" aria-label="Agent profiles">
          <div className="custom-agent-list__heading">
            <span>
              <small>Agent = Harness + Model</small>
              <strong>Custom Agents</strong>
            </span>
            <Button size="icon" onClick={reset} aria-label="New Custom Agent">
              <Plus aria-hidden="true" />
            </Button>
          </div>
          {customAgents.length === 0 && (
            <p className="custom-agent-list__empty">No custom agents yet.</p>
          )}
          {customAgents.map((agent) => (
            <button
              key={agent.id}
              type="button"
              className={editingId === agent.id ? "is-active" : undefined}
              onClick={() => edit(agent)}
            >
              <Bot aria-hidden="true" />
              <span>
                <strong>{agent.name}</strong>
                <small>
                  {agent.harnessRecipe?.softwareId} + {agent.modelId}
                </small>
              </span>
            </button>
          ))}
          {nativeAgents.length > 0 && (
            <div className="custom-agent-list__readonly">
              <small>Native definitions · read-only</small>
              {nativeAgents.map((agent) => (
                <span key={agent.id} title={agent.definition?.path}>
                  <Bot aria-hidden="true" />
                  {nativeAgentHostLabel(agent.definition?.host)} · {agent.name} ·{" "}
                  {agent.modelId ?? agent.nativeModel ?? "Model unresolved"}
                </span>
              ))}
            </div>
          )}
          {extensionAgents.length > 0 && (
            <div className="custom-agent-list__readonly">
              <small>Extension profiles · read-only</small>
              {extensionAgents.map((agent) => (
                <span key={agent.id}>
                  <Package aria-hidden="true" />
                  {agent.name}
                </span>
              ))}
            </div>
          )}
        </aside>

        <form className="custom-agent-editor" onSubmit={(event) => void submit(event)}>
          <div className="settings-content-heading">
            <span>
              <small>Reusable composition</small>
              <h2>{editingId ? `Edit ${name}` : "New Custom Agent"}</h2>
              <p>
                Build a versioned Harness from Software, Skills, MCPs, and policy; then choose its
                Model.
              </p>
            </span>
            <div>
              {editingId && (
                <button
                  type="button"
                  className="settings-secondary-action is-danger"
                  onClick={() =>
                    void onRemove(editingId)
                      .then(reset)
                      .catch((error) => setFormError(errorMessage(error)))
                  }
                >
                  <Trash2 aria-hidden="true" />
                  Delete
                </button>
              )}
              <button type="submit" className="settings-primary-action" disabled={saving}>
                {saving ? "Saving…" : "Save Agent"}
              </button>
            </div>
          </div>
          {formError && <div className="settings-provider-error">{formError}</div>}

          <div className="custom-agent-identity-grid">
            <label>
              <span>Name</span>
              <input
                required
                value={name}
                placeholder="Research agent"
                onChange={(event) => setName(event.target.value)}
              />
            </label>
            <label>
              <span>Model</span>
              <select
                required
                value={modelOptionId}
                onChange={(event) => setModelOptionId(event.target.value)}
              >
                <option value="" disabled>
                  Select a Model
                </option>
                {unavailableEditingModel && (
                  <option value={unavailableEditingModel.id} disabled>
                    {unavailableEditingModel.label}
                  </option>
                )}
                {modelGroups.map((group) => (
                  <optgroup key={group.id} label={group.label}>
                    {group.subgroups.flatMap((subgroup) =>
                      subgroup.models.map((model) => (
                        <option key={model.id} value={model.id}>
                          {subgroup.label ? `${subgroup.label} · ${model.label}` : model.label}
                        </option>
                      )),
                    )}
                  </optgroup>
                ))}
              </select>
            </label>
          </div>

          <section className="harness-recipe-card">
            <div className="harness-recipe-card__heading">
              <span>
                <small>Harness recipe</small>
                <h3>Software + Skills + MCP + Context + Policy</h3>
              </span>
              <Badge tone={softwareHealth?.status === "ready" ? "active" : "danger"}>
                {softwareHealth?.status === "ready" ? "Software ready" : "Setup needed"}
              </Badge>
            </div>
            <label className="harness-software-picker">
              <span>Software</span>
              <select
                required
                value={softwareId}
                onChange={(event) => {
                  setSoftwareId(event.target.value);
                  setModelOptionId("");
                }}
              >
                <option value="" disabled>
                  Select harness software
                </option>
                {softwareOptions.map((software) => (
                  <option key={software.id} value={software.id}>
                    {software.label} · {formatSoftwareSummary(software.software)}
                  </option>
                ))}
              </select>
              <small>
                {softwareHealth?.note ??
                  selectedSoftware?.software?.version ??
                  "Version detected locally"}
              </small>
              {softwareHealth && softwareHealth.status !== "ready" && (
                <Button
                  type="button"
                  size="sm"
                  disabled={setupBusy}
                  onClick={() => {
                    setSetupBusy(true);
                    void onSetupSoftware(softwareId)
                      .catch((error) => setFormError(errorMessage(error)))
                      .finally(() => setSetupBusy(false));
                  }}
                >
                  <Hammer aria-hidden="true" />
                  {setupBusy ? "Setting up…" : "Set up software"}
                </Button>
              )}
            </label>

            <section className="harness-permission-section" aria-label="Agent permission policy">
              <div className="harness-permission-section__heading">
                <ShieldCheck aria-hidden="true" />
                <span>
                  <strong>Agent permission policy</strong>
                  <small>
                    This is one layer. Managed, Project, and personal restrictions can reduce it.
                  </small>
                </span>
              </div>
              <label className="harness-permission-mode">
                <span>Mode</span>
                <select
                  aria-label="Permission mode"
                  value={permissionMode}
                  onChange={(event) =>
                    setPermissionMode(event.target.value as HarnessPermissionMode)
                  }
                >
                  {PERMISSION_MODE_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <small>{permissionModeDescription(permissionMode)}</small>
              </label>
              <div className="permission-rule-grid">
                <PermissionToolRulesEditor
                  label="Pre-approved tools"
                  description="Exact tool names that skip a prompt for this Agent."
                  values={allowedTools}
                  blockedValues={deniedTools}
                  onChange={setAllowedTools}
                />
                <PermissionToolRulesEditor
                  label="Denied tools"
                  description="Exact tool names this Agent can never use."
                  values={deniedTools}
                  blockedValues={allowedTools}
                  tone="danger"
                  onChange={setDeniedTools}
                />
              </div>
            </section>

            <div className="harness-recipe-columns">
              <fieldset>
                <legend>Skills</legend>
                <p>Only add constraints this Agent/Model actually needs.</p>
                <div className="harness-capability-list">
                  {skills.map((skill) => {
                    const binding = skillModes[skill.id];
                    return (
                      <div key={skill.id} className="harness-capability-row">
                        <label>
                          <input
                            type="checkbox"
                            checked={binding !== undefined}
                            onChange={(event) =>
                              setSkillModes((current) => {
                                const next = { ...current };
                                if (event.target.checked) next[skill.id] = { mode: "auto" };
                                else delete next[skill.id];
                                return next;
                              })
                            }
                          />
                          <span>
                            <strong>{skill.name ?? skill.id}</strong>
                            <small>{skill.id}</small>
                          </span>
                        </label>
                        {binding && (
                          <>
                            <select
                              aria-label={`${skill.name ?? skill.id} binding mode`}
                              value={binding.mode}
                              onChange={(event) =>
                                setSkillModes((current) => ({
                                  ...current,
                                  [skill.id]: {
                                    ...current[skill.id],
                                    mode: event.target.value as "off" | "auto" | "required",
                                  },
                                }))
                              }
                            >
                              <option value="auto">Auto</option>
                              <option value="required">Required</option>
                              <option value="off">Off</option>
                            </select>
                            {(skill.variants?.length ?? 0) > 1 && (
                              <select
                                aria-label={`${skill.name ?? skill.id} variant`}
                                value={binding.variantId ?? ""}
                                onChange={(event) =>
                                  setSkillModes((current) => ({
                                    ...current,
                                    [skill.id]: {
                                      ...current[skill.id],
                                      variantId: event.target.value || undefined,
                                    },
                                  }))
                                }
                              >
                                <option value="">Resolve for Agent + Model</option>
                                {skill.variants?.map((variant) => (
                                  <option key={variant.id} value={variant.id}>
                                    {variant.id}
                                  </option>
                                ))}
                              </select>
                            )}
                          </>
                        )}
                      </div>
                    );
                  })}
                </div>
                <div className="harness-context-cost">
                  <Gauge aria-hidden="true" />
                  <span>
                    <strong>~{contextTokenEstimate.toLocaleString()} tokens</strong>
                    <small>
                      {unknownTokenCount > 0
                        ? `${unknownTokenCount} unknown estimates`
                        : "Resolved Skill context"}
                    </small>
                  </span>
                </div>
              </fieldset>

              <fieldset>
                <legend>MCP servers</legend>
                <p>Attach only tools required by this Harness.</p>
                <div className="harness-capability-list">
                  {mcpServers.map((server) => (
                    <label key={server.id} className="harness-simple-capability">
                      <input
                        type="checkbox"
                        checked={selectedMcps.has(server.id)}
                        onChange={(event) =>
                          setSelectedMcps((current) => {
                            const next = new Set(current);
                            if (event.target.checked) next.add(server.id);
                            else next.delete(server.id);
                            return next;
                          })
                        }
                      />
                      <span>
                        <strong>{server.name ?? server.id}</strong>
                        <small>{server.scope ?? "MCP"}</small>
                      </span>
                    </label>
                  ))}
                </div>
              </fieldset>
            </div>

            <div className="harness-policy-grid">
              <label>
                <span>Project context paths</span>
                <textarea
                  value={contextPaths}
                  placeholder="docs/\nsrc/"
                  onChange={(event) => setContextPaths(event.target.value)}
                />
              </label>
              <label>
                <span>Instruction files</span>
                <textarea
                  value={instructionFiles}
                  placeholder="AGENTS.md"
                  onChange={(event) => setInstructionFiles(event.target.value)}
                />
              </label>
              <label>
                <span>Unsupported Skill delivery</span>
                <select
                  value={unsupportedSkill}
                  onChange={(event) => setUnsupportedSkill(event.target.value as "block" | "skip")}
                >
                  <option value="block">Block the Agent</option>
                  <option value="skip">Skip with warning</option>
                </select>
              </label>
            </div>
          </section>

          <label className="custom-agent-instructions">
            <span>Agent instructions</span>
            <textarea
              value={instructions}
              placeholder="Describe the role, priorities, and stopping conditions."
              onChange={(event) => setInstructions(event.target.value)}
            />
          </label>
        </form>
      </div>
    </section>
  );
}

function RuntimeSettings({
  environment,
  loading,
  error,
  doctorReport,
  doctorLoading,
  doctorError,
  harnessVersions,
  fixPending,
  fixRunning,
  fixResult,
  installingHarnessId,
  onRefresh,
  onSetupContainer,
  onInstallHarness,
  onRefreshHarnessVersion,
  onRequestFix,
  onCancelFix,
  onConfirmFix,
}: {
  environment?: HarnessEnvironmentStatus;
  loading: boolean;
  error: unknown;
  doctorReport: DoctorReport | null;
  doctorLoading: boolean;
  doctorError: string | null;
  harnessVersions: Record<string, DoctorHarnessVersionState>;
  fixPending: boolean;
  fixRunning: boolean;
  fixResult: DoctorFixResult | null;
  installingHarnessId: string | null;
  onRefresh: () => Promise<void>;
  onSetupContainer: (containerRuntimeId: string) => Promise<void>;
  onInstallHarness: (harnessId: string) => Promise<void>;
  onRefreshHarnessVersion: (harnessId: string) => void;
  onRequestFix: () => void;
  onCancelFix: () => void;
  onConfirmFix: () => void;
}) {
  const [busyId, setBusyId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const nodeRuntime = environment?.requirements.find((requirement) => requirement.id === "node");
  const harnesses = environment?.harnesses ?? [];
  const doctorIssues = doctorReport?.issues ?? [];
  const repairActions = doctorReport?.repairActions ?? [];
  const doctorHealthy = Boolean(doctorReport?.healthy && doctorIssues.length === 0);
  const repairLogs = fixResult?.setupResults.flatMap((result) => result.log) ?? [];
  const run = async (id: string, action: () => Promise<void>) => {
    setBusyId(id);
    setActionError(null);
    try {
      await action();
    } catch (error) {
      setActionError(errorMessage(error));
    } finally {
      setBusyId(null);
    }
  };

  return (
    <section className="settings-workspace" aria-label="Runtime settings">
      <div className="settings-workspace__body">
        <div className="settings-workspace__content">
          <section className="runtime-settings">
            <div className="settings-content-heading">
              <span>
                <small>Local environment</small>
                <h2>Runtime</h2>
                <p>
                  Node.js is the shared baseline. Harness tools and environment diagnostics are
                  managed here independently from each Custom Agent recipe.
                </p>
              </span>
              <div>
                <button
                  type="button"
                  className="settings-primary-action"
                  disabled={busyId !== null}
                  onClick={() => void run("refresh", onRefresh)}
                >
                  <RefreshCw
                    className={busyId === "refresh" ? "is-spinning" : undefined}
                    aria-hidden="true"
                  />
                  Refresh
                </button>
              </div>
            </div>

            {Boolean(actionError || doctorError || error) && (
              <div className="settings-provider-error" role="alert">
                {actionError ?? doctorError ?? errorMessage(error)}
              </div>
            )}
            {loading && !environment ? (
              <div className="runtime-settings__empty">Detecting local runtimes…</div>
            ) : (
              <>
                <div className="runtime-settings__summary">
                  <span>
                    <strong>
                      {harnesses.filter((harness) => Boolean(harness.version)).length}
                    </strong>
                    <small>Harness tools detected</small>
                  </span>
                  <span>
                    <strong>{harnesses.filter((harness) => !harness.version).length}</strong>
                    <small>Harness tools missing</small>
                  </span>
                  <span>
                    <strong>{nodeRuntime?.version ?? "—"}</strong>
                    <small>Node.js</small>
                  </span>
                  <span>
                    <strong>
                      {environment?.checkedAt ? formatUsageReset(environment.checkedAt) : "—"}
                    </strong>
                    <small>last checked</small>
                  </span>
                </div>

                <section
                  className="runtime-settings__doctor"
                  aria-labelledby="runtime-doctor-title"
                >
                  <div className="runtime-settings__doctor-heading">
                    <span>
                      <small>Built-in diagnostics</small>
                      <h3 id="runtime-doctor-title">Environment Doctor</h3>
                    </span>
                    <Badge tone={doctorHealthy ? "active" : "neutral"}>
                      {doctorLoading
                        ? "Checking"
                        : doctorHealthy
                          ? "Healthy"
                          : `${doctorIssues.length} ${doctorIssues.length === 1 ? "issue" : "issues"}`}
                    </Badge>
                  </div>
                  <div
                    className={cx("doctor-summary", doctorHealthy && "is-healthy")}
                    aria-live="polite"
                  >
                    <span className="doctor-summary__icon">
                      {doctorLoading ? (
                        <Loader2 aria-hidden="true" />
                      ) : doctorHealthy ? (
                        <CircleCheck aria-hidden="true" />
                      ) : (
                        <Wrench aria-hidden="true" />
                      )}
                    </span>
                    <div>
                      <h3>
                        {doctorLoading
                          ? "Checking local environment"
                          : doctorHealthy
                            ? "Environment ready"
                            : doctorReport
                              ? "Review the diagnostics below"
                              : "Doctor status unavailable"}
                      </h3>
                      <p>
                        Harnesses remain optional. Doctor checks the shared baseline and applies no
                        repair until you confirm its plan.
                      </p>
                    </div>
                  </div>

                  {fixResult?.executed && (
                    <output
                      className={cx(
                        "doctor-notice",
                        fixResult.after.healthy ? "doctor-notice--success" : "doctor-notice--error",
                      )}
                    >
                      {fixResult.after.healthy ? (
                        <CircleCheck aria-hidden="true" />
                      ) : (
                        <XCircle aria-hidden="true" />
                      )}
                      <span>
                        {fixResult.after.healthy
                          ? "Repairs completed. The environment is ready."
                          : "Repairs completed, but some diagnostics still need attention."}
                      </span>
                    </output>
                  )}

                  {!doctorLoading && repairActions.length > 0 && (
                    <section className="doctor-section" aria-labelledby="runtime-repair-title">
                      <div className="doctor-section__heading">
                        <h3 id="runtime-repair-title">Repair plan</h3>
                        <span>{repairActions.length}</span>
                      </div>
                      {fixPending ? (
                        <div className="doctor-confirmation">
                          <strong>Confirm environment changes</strong>
                          <p>No installer or system change runs until this plan is confirmed.</p>
                          <ul className="doctor-list">
                            {repairActions.map((action) => (
                              <li key={action.id} className="doctor-action">
                                <span>{action.label}</span>
                                <Badge tone={action.risk === "admin" ? "danger" : "neutral"}>
                                  {action.risk}
                                </Badge>
                              </li>
                            ))}
                          </ul>
                          <div className="doctor-confirmation__actions">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={onCancelFix}
                              disabled={fixRunning}
                            >
                              Cancel
                            </Button>
                            <Button size="sm" onClick={onConfirmFix} disabled={fixRunning}>
                              {fixRunning ? (
                                <Loader2 data-icon="inline-start" aria-hidden="true" />
                              ) : (
                                <Wrench data-icon="inline-start" aria-hidden="true" />
                              )}
                              Confirm repairs
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <Button size="sm" onClick={onRequestFix}>
                          <Wrench data-icon="inline-start" aria-hidden="true" />
                          Review repair plan
                        </Button>
                      )}
                    </section>
                  )}

                  {!doctorLoading && doctorIssues.length > 0 && (
                    <section className="doctor-section" aria-labelledby="runtime-diagnostics-title">
                      <div className="doctor-section__heading">
                        <h3 id="runtime-diagnostics-title">Diagnostics</h3>
                        <span>{doctorIssues.length}</span>
                      </div>
                      <ul className="doctor-list">
                        {doctorIssues.map((issue) => (
                          <li key={issue.id} className="doctor-issue">
                            <XCircle aria-hidden="true" />
                            <div>
                              <strong>{issue.targetId ?? issue.scope}</strong>
                              <span>{issue.message}</span>
                            </div>
                            <Badge tone={issue.severity === "error" ? "danger" : "neutral"}>
                              {issue.severity}
                            </Badge>
                          </li>
                        ))}
                      </ul>
                    </section>
                  )}
                </section>

                <section className="runtime-settings__section" aria-labelledby="runtime-node-title">
                  <div>
                    <h3 id="runtime-node-title">Node.js</h3>
                    <p>
                      Shared JavaScript runtime for npm/npx-based adapters and package management.
                    </p>
                  </div>
                  <ul className="runtime-settings__list runtime-settings__list--node">
                    {nodeRuntime && (
                      <li>
                        <span className={`runtime-status-icon is-${nodeRuntime.status}`}>
                          {nodeRuntime.status === "ready" ? (
                            <CircleCheck aria-hidden="true" />
                          ) : (
                            <XCircle aria-hidden="true" />
                          )}
                        </span>
                        <span className="runtime-settings__identity">
                          <strong>{nodeRuntime.label}</strong>
                          <small>{nodeRuntime.path ?? nodeRuntime.command}</small>
                        </span>
                        {nodeRuntime.version ? (
                          <button
                            type="button"
                            className="badge badge--active doctor-harness__version"
                            aria-label="Check Node.js version again"
                            title="Check version again"
                            disabled={busyId !== null}
                            onClick={() => void run("refresh-node", onRefresh)}
                          >
                            {nodeRuntime.version}
                          </button>
                        ) : (
                          <Badge tone="danger">{requirementStatusLabel(nodeRuntime.status)}</Badge>
                        )}
                      </li>
                    )}
                  </ul>
                </section>

                <section
                  className="runtime-settings__section"
                  aria-labelledby="runtime-harnesses-title"
                >
                  <div>
                    <h3 id="runtime-harnesses-title">Harness tools</h3>
                    <p>
                      Tool versions are detected independently. Click a version to check it again.
                    </p>
                  </div>
                  <ul className="runtime-harness-list">
                    {harnesses.map((harness) => {
                      const versionState = harnessVersions[harness.harnessId];
                      const version = versionState?.version ?? harness.version;
                      const versionLoading = versionState?.status === "loading";
                      return (
                        <li key={harness.harnessId}>
                          <span className="runtime-harness-list__icon">
                            <HarnessBrandIcon
                              harness={harnessOption(harness.harnessId, harness.harnessLabel)}
                            />
                          </span>
                          <span className="runtime-settings__identity">
                            <strong>{harness.harnessLabel}</strong>
                            <small>{harness.path ?? harness.command}</small>
                          </span>
                          {versionLoading ? (
                            <output
                              className="badge doctor-harness__version is-loading"
                              aria-label={`Checking ${harness.harnessLabel} version`}
                            >
                              <Loader2 data-icon aria-hidden="true" />
                            </output>
                          ) : version ? (
                            <button
                              type="button"
                              className="badge badge--active doctor-harness__version"
                              aria-label={`Check ${harness.harnessLabel} version again`}
                              title="Check version again"
                              onClick={() => onRefreshHarnessVersion(harness.harnessId)}
                            >
                              {version}
                            </button>
                          ) : harness.installable ? (
                            <Button
                              size="sm"
                              disabled={Boolean(installingHarnessId)}
                              aria-label={`Install ${harness.harnessLabel}`}
                              onClick={() => void onInstallHarness(harness.harnessId)}
                            >
                              {installingHarnessId === harness.harnessId ? (
                                <Loader2 data-icon="inline-start" aria-hidden="true" />
                              ) : (
                                <Download data-icon="inline-start" aria-hidden="true" />
                              )}
                              Install
                            </Button>
                          ) : (
                            <Badge tone="neutral">Not detected</Badge>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                </section>

                <section
                  className="runtime-settings__section"
                  aria-labelledby="runtime-container-title"
                >
                  <div>
                    <h3 id="runtime-container-title">Container runtime</h3>
                    <p>
                      Apple Container is preferred for protected local harness execution on
                      supported macOS hosts.
                    </p>
                  </div>
                  <ul className="runtime-settings__list">
                    {(environment?.containerRuntimes ?? []).map((runtime) => (
                      <li key={runtime.id}>
                        <span className={`runtime-status-icon is-${runtime.status}`}>
                          {runtime.status === "ready" ? (
                            <CircleCheck aria-hidden="true" />
                          ) : (
                            <XCircle aria-hidden="true" />
                          )}
                        </span>
                        <span className="runtime-settings__identity">
                          <strong>{runtime.label}</strong>
                          <small>{runtime.path ?? runtime.command}</small>
                        </span>
                        <span className="runtime-settings__version">
                          {runtime.version ?? runtime.status.replaceAll("_", " ")}
                        </span>
                        <span className="runtime-settings__consumers">
                          {runtime.preferred
                            ? "Preferred"
                            : runtime.supported
                              ? "Supported"
                              : "Unavailable on this host"}
                        </span>
                        {runtime.status !== "ready" && runtime.installable && (
                          <Button
                            size="sm"
                            disabled={busyId !== null}
                            onClick={() =>
                              void run(`container:${runtime.id}`, () =>
                                onSetupContainer(runtime.id),
                              )
                            }
                          >
                            <Download aria-hidden="true" />
                            Set up
                          </Button>
                        )}
                      </li>
                    ))}
                  </ul>
                </section>

                <div className="runtime-settings__path">
                  <span>Detected PATH</span>
                  <code>{environment?.path ?? "Unavailable"}</code>
                  {repairLogs.length > 0 && <pre>{repairLogs.join("\n\n")}</pre>}
                </div>
              </>
            )}
          </section>
        </div>
      </div>
    </section>
  );
}

function SettingsWorkspace({
  providers,
  modelCatalog,
  modelCatalogRefreshing,
  modelCatalogError,
  providerUsage,
  providerUsageRefreshing,
  providerUsageRefreshingIds,
  providerUsageError,
  onRefreshModels,
  onRefreshUsage,
  onSaveProvider,
  onRemoveProvider,
  onResetProviderKey,
}: {
  providers: ExtensionProviderSummary[];
  modelCatalog?: ModelCatalogSummary;
  modelCatalogRefreshing: boolean;
  modelCatalogError: string | null;
  providerUsage: ProviderUsageSnapshot | null;
  providerUsageRefreshing: boolean;
  providerUsageRefreshingIds: ReadonlySet<string>;
  providerUsageError: string | null;
  onRefreshModels: () => Promise<void>;
  onRefreshUsage: (target?: ProviderUsageTarget) => Promise<void>;
  onSaveProvider: (input: UserProviderInput) => Promise<void>;
  onRemoveProvider: (providerId: string) => Promise<void>;
  onResetProviderKey: (providerId: string, keyId: string) => Promise<void>;
}) {
  const [providerFormOpen, setProviderFormOpen] = useState(false);
  const [editingProviderId, setEditingProviderId] = useState<string | null>(null);
  const [providerLabel, setProviderLabel] = useState("");
  const [providerKind, setProviderKind] = useState<ModelApiProtocol>("anthropic");
  const [providerBaseUrl, setProviderBaseUrl] = useState("https://api.anthropic.com");
  const [providerAuthMode, setProviderAuthMode] = useState<"api_key" | "auth_token">("api_key");
  const [providerUsageAdapter, setProviderUsageAdapter] = useState<"automatic" | "new_api">(
    "automatic",
  );
  const [providerSecret, setProviderSecret] = useState("");
  const [providerAccountAccessToken, setProviderAccountAccessToken] = useState("");
  const [providerAccountUserId, setProviderAccountUserId] = useState("");
  const [providerClearAccountAccess, setProviderClearAccountAccess] = useState(false);
  const [providerAdditionalApiKeys, setProviderAdditionalApiKeys] = useState("");
  const [providerRemovedApiKeyIds, setProviderRemovedApiKeyIds] = useState<string[]>([]);
  const [providerSaving, setProviderSaving] = useState(false);
  const [providerError, setProviderError] = useState<string | null>(null);
  const userProviderIds = new Set(modelCatalog?.userProviderIds ?? []);
  const providerUsageById = new Map(
    (providerUsage?.providers ?? []).map((entry) => [entry.providerProfileId, entry]),
  );
  const codexCatalogProvider = providers.find(
    (provider) => provider.catalogAdapter === "codex_app_server",
  );
  const managedProviders = providers.filter(
    (provider) => provider.catalogAdapter !== "codex_app_server",
  );
  const deepSeekForm = isDeepSeekProviderUrl(providerBaseUrl);
  const openCodeGoForm = isOpenCodeGoProviderUrl(providerBaseUrl);
  const editingProvider = providers.find((provider) => provider.id === editingProviderId);

  const resetProviderForm = () => {
    setProviderFormOpen(false);
    setEditingProviderId(null);
    setProviderLabel("");
    setProviderKind("anthropic");
    setProviderBaseUrl("https://api.anthropic.com");
    setProviderAuthMode("api_key");
    setProviderUsageAdapter("automatic");
    setProviderSecret("");
    setProviderAccountAccessToken("");
    setProviderAccountUserId("");
    setProviderClearAccountAccess(false);
    setProviderAdditionalApiKeys("");
    setProviderRemovedApiKeyIds([]);
    setProviderError(null);
  };
  const beginAddProvider = () => {
    resetProviderForm();
    setProviderFormOpen(true);
  };
  const beginEditProvider = (provider: ExtensionProviderSummary) => {
    setEditingProviderId(provider.id);
    setProviderLabel(provider.label);
    setProviderKind(
      ["anthropic", "openai_chat", "openai_responses", "ollama"].includes(provider.kind)
        ? (provider.kind as ModelApiProtocol)
        : "anthropic",
    );
    setProviderBaseUrl(provider.baseUrl ?? "");
    setProviderAuthMode(provider.authMode ?? "api_key");
    setProviderUsageAdapter(provider.usageAdapter === "new_api" ? "new_api" : "automatic");
    setProviderSecret("");
    setProviderAccountAccessToken("");
    setProviderAccountUserId(provider.newApiAccountUserId ?? "");
    setProviderClearAccountAccess(false);
    setProviderAdditionalApiKeys("");
    setProviderRemovedApiKeyIds([]);
    setProviderError(null);
    setProviderFormOpen(true);
  };
  const submitProvider = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setProviderSaving(true);
    setProviderError(null);
    try {
      await onSaveProvider({
        ...(editingProviderId ? { id: editingProviderId } : {}),
        label: providerLabel,
        kind: providerKind,
        baseUrl: providerBaseUrl,
        authMode: providerAuthMode,
        ...(!openCodeGoForm && providerUsageAdapter === "new_api"
          ? { usageAdapter: "new_api" as const }
          : {}),
        ...(providerSecret.trim() ? { secret: providerSecret } : {}),
        ...(!openCodeGoForm &&
        providerUsageAdapter === "new_api" &&
        !providerClearAccountAccess &&
        providerAccountAccessToken.trim()
          ? { accountAccessToken: providerAccountAccessToken }
          : {}),
        ...(!openCodeGoForm &&
        providerUsageAdapter === "new_api" &&
        !providerClearAccountAccess &&
        providerAccountUserId.trim()
          ? { accountUserId: providerAccountUserId.trim() }
          : {}),
        ...(!openCodeGoForm && providerUsageAdapter === "new_api" && providerClearAccountAccess
          ? { clearAccountAccess: true }
          : {}),
        ...(openCodeGoForm && providerAdditionalApiKeys.trim()
          ? {
              additionalApiKeys: providerAdditionalApiKeys
                .split(/\r?\n/)
                .map((value) => value.trim())
                .filter(Boolean)
                .map((value) => ({ value })),
            }
          : {}),
        ...(openCodeGoForm && providerRemovedApiKeyIds.length > 0
          ? { removeApiKeyIds: providerRemovedApiKeyIds }
          : {}),
      });
      resetProviderForm();
    } catch (error) {
      setProviderError(errorMessage(error));
    } finally {
      setProviderSaving(false);
    }
  };
  const removeUserProvider = async (providerId: string) => {
    setProviderError(null);
    try {
      await onRemoveProvider(providerId);
      if (editingProviderId === providerId) resetProviderForm();
    } catch (error) {
      setProviderError(errorMessage(error));
    }
  };

  return (
    <section className="settings-workspace" aria-label="Settings">
      <div className="settings-workspace__body">
        <div className="settings-workspace__content">
          <section className="settings-providers" aria-labelledby="settings-providers-title">
            <div className="settings-content-heading">
              <span>
                <small>Connections and usage</small>
                <h2 id="settings-providers-title">Providers</h2>
                <p>Model access, account limits, and balances in one place.</p>
              </span>
              <div>
                <button
                  type="button"
                  className="settings-secondary-action"
                  disabled={providerUsageRefreshing || providerUsageRefreshingIds.size > 0}
                  onClick={() => void onRefreshUsage()}
                >
                  <RefreshCw
                    className={providerUsageRefreshing ? "is-spinning" : undefined}
                    aria-hidden="true"
                  />
                  {providerUsageRefreshing ? "Refreshing…" : "Refresh all"}
                </button>
                <button
                  type="button"
                  className="settings-secondary-action"
                  disabled={modelCatalogRefreshing}
                  onClick={() => void onRefreshModels()}
                >
                  <RefreshCw
                    className={modelCatalogRefreshing ? "is-spinning" : undefined}
                    aria-hidden="true"
                  />
                  {modelCatalogRefreshing ? "Refreshing…" : "Refresh Models"}
                </button>
                <button
                  type="button"
                  className="settings-primary-action"
                  onClick={beginAddProvider}
                >
                  <Plus aria-hidden="true" />
                  Add Provider
                </button>
              </div>
            </div>
            {(providerError || modelCatalogError) && (
              <div className="settings-provider-error" role="alert">
                {providerError ?? modelCatalogError}
              </div>
            )}
            {providerUsageError && (
              <div className="settings-provider-error" role="alert">
                {providerUsageError}
              </div>
            )}
            {providerFormOpen && (
              <form
                className="settings-provider-form"
                aria-label={editingProviderId ? "Edit Provider" : "Add Provider"}
                onSubmit={(event) => void submitProvider(event)}
              >
                <div className="settings-provider-form__heading">
                  <span>
                    <small>{editingProviderId ? "Existing connection" : "New connection"}</small>
                    <strong>{editingProviderId ? "Edit Provider" : "Add Provider"}</strong>
                  </span>
                  <button
                    type="button"
                    onClick={resetProviderForm}
                    aria-label="Close Provider form"
                  >
                    <X aria-hidden="true" />
                  </button>
                </div>
                <div className="settings-provider-form__grid">
                  <label>
                    <span>Provider name</span>
                    <input
                      required
                      value={providerLabel}
                      placeholder="My Anthropic gateway"
                      onChange={(event) => setProviderLabel(event.target.value)}
                    />
                  </label>
                  <label>
                    <span>
                      {deepSeekForm || openCodeGoForm ? "Preferred API protocol" : "API protocol"}
                    </span>
                    <select
                      aria-label={
                        deepSeekForm || openCodeGoForm ? "Preferred API protocol" : "API protocol"
                      }
                      value={providerKind}
                      onChange={(event) => setProviderKind(event.target.value as ModelApiProtocol)}
                    >
                      <option value="anthropic">Anthropic</option>
                      {!openCodeGoForm && (
                        <option value="openai_responses">OpenAI Responses</option>
                      )}
                      <option value="openai_chat">OpenAI Chat</option>
                      {!openCodeGoForm && <option value="ollama">Ollama</option>}
                    </select>
                    {deepSeekForm && (
                      <small className="settings-provider-form__helper">
                        DeepSeek supports native OpenAI and Anthropic APIs. This selects the
                        preferred route while keeping them in one Provider.
                      </small>
                    )}
                    {openCodeGoForm && (
                      <small className="settings-provider-form__helper">
                        The official Go endpoint exposes Anthropic and OpenAI Chat routes. Models
                        are loaded from /zen/go/v1/models.
                      </small>
                    )}
                  </label>
                  <label className="settings-provider-form__wide">
                    <span>Base URL</span>
                    <input
                      required
                      type="url"
                      value={providerBaseUrl}
                      placeholder="https://api.example.com"
                      onChange={(event) => {
                        const value = event.target.value;
                        setProviderBaseUrl(value);
                        if (
                          isOpenCodeGoProviderUrl(value) &&
                          !["anthropic", "openai_chat"].includes(providerKind)
                        ) {
                          setProviderKind("openai_chat");
                        }
                      }}
                    />
                  </label>
                  <label>
                    <span>Authentication</span>
                    <select
                      value={providerAuthMode}
                      onChange={(event) =>
                        setProviderAuthMode(event.target.value as "api_key" | "auth_token")
                      }
                    >
                      <option value="api_key">API Key</option>
                      <option value="auth_token">Auth Token</option>
                    </select>
                  </label>
                  {openCodeGoForm ? (
                    <div className="settings-provider-form__section">
                      <strong>Local usage tracking</strong>
                      <small>
                        No usage endpoint is queried. Quota errors cool and rotate keys.
                      </small>
                    </div>
                  ) : (
                    <label>
                      <span>Usage API</span>
                      <select
                        value={providerUsageAdapter}
                        onChange={(event) =>
                          setProviderUsageAdapter(event.target.value as "automatic" | "new_api")
                        }
                      >
                        <option value="automatic">Automatic</option>
                        <option value="new_api">New API</option>
                      </select>
                    </label>
                  )}
                  <label className="settings-provider-form__wide">
                    <span>
                      {openCodeGoForm
                        ? "Primary API key"
                        : providerUsageAdapter === "new_api"
                          ? "Primary API token"
                          : providerAuthMode === "auth_token"
                            ? "Auth token"
                            : "API key"}
                    </span>
                    <input
                      aria-label={
                        openCodeGoForm
                          ? "Primary API key"
                          : providerUsageAdapter === "new_api"
                            ? "Primary API token"
                            : providerAuthMode === "auth_token"
                              ? "Auth token"
                              : "API key"
                      }
                      required={!editingProviderId}
                      type="password"
                      autoComplete="new-password"
                      value={providerSecret}
                      placeholder={
                        editingProviderId ? "Leave blank to keep current" : "Stored securely"
                      }
                      onChange={(event) => setProviderSecret(event.target.value)}
                    />
                    {!openCodeGoForm && providerUsageAdapter === "new_api" && (
                      <small className="settings-provider-form__helper">
                        Used for Model requests and its own /api/usage/token quota.
                      </small>
                    )}
                  </label>
                  {openCodeGoForm && (
                    <>
                      <label className="settings-provider-form__wide">
                        <span>Additional API keys</span>
                        <textarea
                          aria-label="Additional API keys"
                          rows={4}
                          value={providerAdditionalApiKeys}
                          placeholder="One API key per line"
                          onChange={(event) => setProviderAdditionalApiKeys(event.target.value)}
                        />
                        <small className="settings-provider-form__helper">
                          New keys are encrypted separately. Values are never shown again.
                        </small>
                      </label>
                      {editingProvider?.runtimeKeyUsage &&
                        editingProvider.runtimeKeyUsage.length > 0 && (
                          <div className="settings-provider-form__wide settings-provider-key-editor">
                            <strong>Saved keys</strong>
                            {editingProvider.runtimeKeyUsage.map((key) => {
                              const removed = providerRemovedApiKeyIds.includes(key.id);
                              return (
                                <span key={key.id} className={cx(removed && "is-removed")}>
                                  <span>
                                    <strong>{key.label}</strong>
                                    <small>
                                      {removed
                                        ? "Will be removed"
                                        : `${capitalize(key.status)} · ${key.totalTokens.toLocaleString()} tokens`}
                                    </small>
                                  </span>
                                  {key.id !== "primary" && (
                                    <button
                                      type="button"
                                      onClick={() =>
                                        setProviderRemovedApiKeyIds((current) =>
                                          current.includes(key.id)
                                            ? current.filter((id) => id !== key.id)
                                            : [...current, key.id],
                                        )
                                      }
                                    >
                                      {removed ? "Keep" : "Remove"}
                                    </button>
                                  )}
                                </span>
                              );
                            })}
                          </div>
                        )}
                    </>
                  )}
                  {!openCodeGoForm && providerUsageAdapter === "new_api" && (
                    <>
                      <div className="settings-provider-form__section settings-provider-form__wide">
                        <strong>Account usage</strong>
                        <small>
                          Optional high-privilege management credential for /api/user/self and the
                          masked /api/token listing. It is never used for Model requests.
                        </small>
                      </div>
                      <label>
                        <span>New API user ID</span>
                        <input
                          inputMode="numeric"
                          value={providerAccountUserId}
                          placeholder="User ID"
                          disabled={providerClearAccountAccess}
                          onChange={(event) => setProviderAccountUserId(event.target.value)}
                        />
                      </label>
                      <label>
                        <span>Account access token</span>
                        <input
                          type="password"
                          autoComplete="new-password"
                          value={providerAccountAccessToken}
                          placeholder={
                            editingProviderId && !providerClearAccountAccess
                              ? "Leave blank to keep current"
                              : "Optional account credential"
                          }
                          disabled={providerClearAccountAccess}
                          onChange={(event) => setProviderAccountAccessToken(event.target.value)}
                        />
                      </label>
                      {editingProviderId &&
                        providers.find((provider) => provider.id === editingProviderId)
                          ?.accountAccessReady && (
                          <label className="settings-provider-form__checkbox settings-provider-form__wide">
                            <input
                              type="checkbox"
                              checked={providerClearAccountAccess}
                              onChange={(event) =>
                                setProviderClearAccountAccess(event.target.checked)
                              }
                            />
                            <span>Remove saved account access</span>
                          </label>
                        )}
                    </>
                  )}
                </div>
                <p className="settings-provider-form__security">
                  Credentials are encrypted locally and decrypted only in the main process for their
                  stated Provider operation.
                </p>
                <div className="settings-provider-form__actions">
                  <button type="button" onClick={resetProviderForm}>
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={
                      providerSaving ||
                      !providerLabel.trim() ||
                      !providerBaseUrl.trim() ||
                      (!editingProviderId && !providerSecret.trim())
                    }
                  >
                    {providerSaving ? "Saving…" : "Save Provider"}
                  </button>
                </div>
              </form>
            )}
            <section className="settings-provider-matrix" aria-label="Provider usage matrix">
              <div className="settings-provider-matrix__header">
                <span>Provider</span>
                <span>5-hour</span>
                <span>7-day</span>
                <span>Credit &amp; balance</span>
                <span>Resets</span>
                <span>Updated</span>
                <span>Actions</span>
              </div>
              {(providerUsage?.toolAccounts.length ? providerUsage.toolAccounts : [undefined]).map(
                (entry) => {
                  const sourceId = entry?.sourceId ?? "codex";
                  const label = entry?.label ?? "Codex";
                  const target = { source: "tool_account" as const, sourceId };
                  const catalogProvider = sourceId === "codex" ? codexCatalogProvider : undefined;
                  const discovery = catalogProvider
                    ? modelCatalog?.providers.find(
                        (status) => status.providerProfileId === catalogProvider.id,
                      )
                    : undefined;
                  return (
                    <ProviderMatrixRow
                      key={`tool-account:${sourceId}`}
                      label={label}
                      source="tool_account"
                      sourceId={sourceId}
                      provider={catalogProvider}
                      discovery={discovery}
                      entry={entry}
                      loading={
                        providerUsageRefreshing ||
                        providerUsageRefreshingIds.has(providerUsageTargetKey(target))
                      }
                      updatedFallback={providerUsage?.fetchedAt}
                      onRefresh={() => onRefreshUsage(target)}
                    />
                  );
                },
              )}
              {managedProviders.map((provider) => {
                const discovery = modelCatalog?.providers.find(
                  (status) => status.providerProfileId === provider.id,
                );
                const entry = providerUsageById.get(provider.id);
                const target = { source: "provider" as const, sourceId: provider.id };
                const userManaged = userProviderIds.has(provider.id) && provider.readOnly !== true;
                return (
                  <ProviderMatrixRow
                    key={`provider:${provider.id}`}
                    label={provider.label}
                    source="provider"
                    sourceId={provider.id}
                    provider={provider}
                    discovery={discovery}
                    entry={entry}
                    loading={
                      providerUsageRefreshing ||
                      providerUsageRefreshingIds.has(providerUsageTargetKey(target))
                    }
                    updatedFallback={providerUsage?.fetchedAt}
                    userManaged={userManaged}
                    saving={providerSaving}
                    onRefresh={() => onRefreshUsage(target)}
                    onEdit={userManaged ? () => beginEditProvider(provider) : undefined}
                    onRemove={userManaged ? () => void removeUserProvider(provider.id) : undefined}
                    onResetProviderKey={(keyId) => onResetProviderKey(provider.id, keyId)}
                  />
                );
              })}
            </section>
          </section>
        </div>
      </div>
    </section>
  );
}

function ProviderMatrixRow({
  label,
  source,
  sourceId,
  provider,
  discovery,
  entry,
  loading,
  updatedFallback,
  userManaged = false,
  saving = false,
  onRefresh,
  onEdit,
  onRemove,
  onResetProviderKey,
}: {
  label: string;
  source: ProviderUsageEntry["source"];
  sourceId: string;
  provider?: ExtensionProviderSummary;
  discovery?: ModelCatalogSummary["providers"][number];
  entry?: ProviderUsageEntry;
  loading: boolean;
  updatedFallback?: string;
  userManaged?: boolean;
  saving?: boolean;
  onRefresh: () => Promise<void>;
  onEdit?: () => void;
  onRemove?: () => void;
  onResetProviderKey?: (keyId: string) => Promise<void>;
}) {
  const fiveHour = findUsageWindow(entry, "five_hour");
  const weekly = findUsageWindow(entry, "weekly");
  const reset = entry?.meters.find(
    (meter): meter is Extract<ProviderUsageMeter, { kind: "credit" }> =>
      meter.kind === "credit" && /reset/i.test(meter.label),
  );
  const updatedAt = entry?.fetchedAt ?? (entry ? updatedFallback : undefined);
  const status = providerUsageStatus(entry, loading);
  const deepSeek = provider ? isDeepSeekProvider(provider) : false;
  const openCodeGo = provider ? isOpenCodeGoProviderUrl(provider.baseUrl ?? "") : false;
  const modelCount =
    discovery?.status === "ready" || discovery?.status === "cached"
      ? `${discovery.modelCount} model${discovery.modelCount === 1 ? "" : "s"}`
      : provider?.runtimeReady === false
        ? "Needs attention"
        : "Configured";
  const providerMeta =
    source === "tool_account"
      ? `OpenAI official · Local account · ${
          entry?.plan ? `${capitalize(entry.plan)} plan` : "Codex app-server"
        } · ${modelCount}`
      : deepSeek && provider
        ? `${providerProtocolLabel(provider.kind)} + ${
            provider.kind === "anthropic" ? "OpenAI" : "Anthropic"
          } · Preferred ${providerProtocolLabel(provider.kind)} · ${modelCount}`
        : openCodeGo
          ? `OpenCode Go · ${entry?.keys?.length ?? provider?.runtimeKeySlots?.length ?? 0} keys · Local usage · ${modelCount}`
          : provider?.usageAdapter === "new_api"
            ? `New API · ${modelCount}`
            : `${providerProtocolLabel(provider?.kind ?? "Provider")} · ${modelCount}`;

  return (
    <article
      className={cx("settings-provider-matrix__row", loading && "is-loading")}
      aria-label={`${label} Provider`}
    >
      <div className="settings-provider-matrix__provider" data-label="Provider">
        <ProviderBrandIcon label={label} sourceId={sourceId} provider={provider} />
        <span className="settings-provider-matrix__identity">
          <span>
            <strong>{label}</strong>
            <small
              className={cx("settings-provider-status", `is-${status.tone}`)}
              title={entry?.detail}
            >
              {status.label}
            </small>
          </span>
          <small>{provider?.baseUrl ?? "Local official connection"}</small>
          <span>{providerMeta}</span>
        </span>
      </div>
      <ProviderWindowCell label="5-hour" meter={fiveHour} loading={loading && !entry} />
      <ProviderWindowCell label="7-day" meter={weekly} loading={loading && !entry} />
      <ProviderFinanceCell
        id={`${source}-${sourceId}`}
        label={label}
        entry={entry}
        loading={loading && !entry}
      />
      <div className="settings-provider-matrix__metric" data-label="Resets">
        {loading && !entry ? (
          <span className="settings-provider-matrix__skeleton" aria-label="Loading resets" />
        ) : reset ? (
          <>
            <strong>{reset.remaining}</strong>
            <small>{reset.unit}</small>
          </>
        ) : (
          <NotProvided />
        )}
      </div>
      <div className="settings-provider-matrix__updated" data-label="Updated">
        {loading && <Loader2 className="is-spinning" aria-hidden="true" />}
        <span>{updatedAt ? formatUsageReset(updatedAt) : "Not checked"}</span>
      </div>
      <div className="settings-provider-matrix__actions" data-label="Actions">
        <button
          type="button"
          aria-label={`Refresh ${label} usage`}
          title={`Refresh ${label} usage`}
          disabled={loading}
          onClick={() => void onRefresh()}
        >
          <RefreshCw className={loading ? "is-spinning" : undefined} aria-hidden="true" />
        </button>
        {userManaged && onEdit && (
          <button type="button" aria-label={`Edit Provider ${label}`} onClick={onEdit}>
            Edit
          </button>
        )}
        {userManaged && onRemove && (
          <button
            type="button"
            aria-label={`Remove Provider ${label}`}
            disabled={saving}
            onClick={onRemove}
          >
            <Trash2 aria-hidden="true" />
          </button>
        )}
      </div>
      {(provider?.usageAdapter === "new_api" || entry?.account) && (
        <NewApiAccountDetails provider={provider} entry={entry} onManage={onEdit} />
      )}
      {openCodeGo && (
        <ProviderKeyPoolDetails
          keys={entry?.keys ?? provider?.runtimeKeyUsage ?? []}
          onReset={onResetProviderKey}
        />
      )}
    </article>
  );
}

function ProviderWindowCell({
  label,
  meter,
  loading,
}: {
  label: "5-hour" | "7-day";
  meter?: Extract<ProviderUsageMeter, { kind: "window" }>;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="settings-provider-matrix__metric" data-label={label}>
        <span
          className="settings-provider-matrix__skeleton"
          aria-label={`Loading ${label} quota`}
        />
      </div>
    );
  }
  if (!meter) {
    return (
      <div className="settings-provider-matrix__metric" data-label={label}>
        <NotProvided />
      </div>
    );
  }
  const remaining = Math.max(0, Math.min(100, meter.remainingPercent));
  return (
    <div className="settings-provider-matrix__metric" data-label={label}>
      <strong>{formatUsagePercent(meter.remainingPercent)} left</strong>
      <span
        className="settings-provider-matrix__track"
        role="progressbar"
        aria-label={`${label} remaining`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(remaining)}
        tabIndex={0}
      >
        <span style={{ width: `${remaining}%` }} />
      </span>
      <small>
        {meter.resetsAt ? `Resets ${formatUsageReset(meter.resetsAt)}` : "Reset not provided"}
      </small>
    </div>
  );
}

function ProviderFinanceCell({
  id,
  label,
  entry,
  loading,
}: {
  id: string;
  label: string;
  entry?: ProviderUsageEntry;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="settings-provider-matrix__metric" data-label="Credit & balance">
        <span
          className="settings-provider-matrix__skeleton"
          aria-label="Loading credit and balance"
        />
      </div>
    );
  }
  const finance = providerFinanceSummary(entry);
  if (!finance) {
    return (
      <div className="settings-provider-matrix__metric" data-label="Credit & balance">
        <NotProvided />
      </div>
    );
  }
  const tooltipId = `provider-finance-${id.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
  return (
    <div className="settings-provider-matrix__metric" data-label="Credit & balance">
      <button
        type="button"
        className="settings-provider-finance"
        aria-label={`${label} credit and balance: ${finance.primary}. Focus for breakdown.`}
        aria-describedby={tooltipId}
      >
        <strong>{finance.primary}</strong>
        <small>{finance.caption}</small>
        <span id={tooltipId} className="settings-provider-finance__popup" role="tooltip">
          <strong>Credit &amp; balance</strong>
          {finance.lines.map((line) => (
            <span key={line}>{line}</span>
          ))}
        </span>
      </button>
    </div>
  );
}

function NewApiAccountDetails({
  provider,
  entry,
  onManage,
}: {
  provider?: ExtensionProviderSummary;
  entry?: ProviderUsageEntry;
  onManage?: () => void;
}) {
  const account = entry?.account;
  const tokenCount = account?.totalTokens ?? 0;
  const accountLabel = account
    ? `${tokenCount} API token${tokenCount === 1 ? "" : "s"}`
    : provider?.accountAccessReady
      ? "Refresh to load account"
      : "Account access not configured";
  return (
    <details className="settings-provider-account">
      <summary>
        <span>Account &amp; API tokens</span>
        <small>{accountLabel}</small>
      </summary>
      <div className="settings-provider-account__content">
        <div className="settings-provider-account__summary">
          <span>
            <strong>{account?.displayName ?? "New API account"}</strong>
            <small>{account?.group ? `Group: ${account.group}` : "Account-level usage"}</small>
          </span>
          <small className={cx("settings-provider-status", `is-${account?.status ?? "muted"}`)}>
            {account ? capitalize(account.status) : "Not configured"}
          </small>
        </div>
        {account?.detail && <p>{account.detail}</p>}
        {!account && !provider?.accountAccessReady && (
          <div className="settings-provider-account__connect">
            <p>Connect account access to see wallet and API tokens.</p>
            {onManage && (
              <button type="button" onClick={onManage}>
                Manage account access
              </button>
            )}
          </div>
        )}
        {account &&
          (account.tokens.length ? (
            <section className="settings-provider-token-list" aria-label="New API tokens">
              <div className="settings-provider-token-list__header">
                <span>Token</span>
                <span>Status</span>
                <span>Remaining</span>
                <span>Used</span>
                <span>Expires</span>
              </div>
              {account.tokens.map((token) => (
                <div className="settings-provider-token-list__row" key={token.id}>
                  <span data-label="Token">
                    <strong>{token.name}</strong>
                    <small>{maskProviderTokenId(token.id)}</small>
                  </span>
                  <span data-label="Status" className={`is-${token.status}`}>
                    {capitalize(token.status)}
                  </span>
                  <span data-label="Remaining">{token.remaining}</span>
                  <span data-label="Used">{token.used ?? "—"}</span>
                  <span data-label="Expires">
                    {token.expiresAt ? formatUsageReset(token.expiresAt) : "No expiry"}
                  </span>
                </div>
              ))}
            </section>
          ) : (
            <p>No API token summaries returned for this account.</p>
          ))}
      </div>
    </details>
  );
}

function ProviderKeyPoolDetails({
  keys,
  onReset,
}: {
  keys: ProviderKeyUsageSummary[];
  onReset?: (keyId: string) => Promise<void>;
}) {
  const [resettingKeyId, setResettingKeyId] = useState<string | null>(null);
  const ready = keys.filter((key) => key.status === "ready").length;
  return (
    <details className="settings-provider-account settings-provider-key-pool">
      <summary>
        <span>API key pool</span>
        <small>
          {ready}/{keys.length} ready · local counters
        </small>
      </summary>
      <div className="settings-provider-account__content">
        <p>
          Keys rotate only after an explicit quota-exhausted response and only before any output or
          tool event has been emitted.
        </p>
        <section className="settings-provider-key-list" aria-label="OpenCode Go API keys">
          {keys.map((key) => (
            <div className="settings-provider-key-list__row" key={key.id}>
              <span>
                <strong>{key.label}</strong>
                <small>{key.id === "primary" ? "Primary" : "Encrypted backup"}</small>
              </span>
              <span className={`is-${key.status}`}>{capitalize(key.status)}</span>
              <span>
                <strong>{key.totalTokens.toLocaleString()}</strong>
                <small>{key.requestCount.toLocaleString()} requests</small>
              </span>
              <span>
                {key.cooldownUntil
                  ? `Retry ${formatUsageReset(key.cooldownUntil)}`
                  : key.lastUsedAt
                    ? `Used ${formatUsageReset(key.lastUsedAt)}`
                    : "Not used yet"}
              </span>
              {key.status === "cooling" && onReset ? (
                <button
                  type="button"
                  disabled={resettingKeyId !== null}
                  onClick={() => {
                    setResettingKeyId(key.id);
                    void onReset(key.id)
                      .catch(() => undefined)
                      .finally(() => setResettingKeyId(null));
                  }}
                >
                  {resettingKeyId === key.id ? "Resetting…" : "Reset cooldown"}
                </button>
              ) : (
                <span />
              )}
            </div>
          ))}
        </section>
      </div>
    </details>
  );
}

function ProviderBrandIcon({
  label,
  sourceId,
  provider,
}: {
  label: string;
  sourceId: string;
  provider?: ExtensionProviderSummary;
}) {
  const normalizedLabel = label.toLowerCase();
  const normalizedUrl = provider?.baseUrl?.toLowerCase() ?? "";
  const iconUrl =
    sourceId === "codex" || normalizedLabel === "codex"
      ? "./harness-icons/codex.svg"
      : isOpenAIProvider(provider)
        ? "./harness-icons/codex.svg"
        : isDeepSeekProvider(provider)
          ? "./provider-icons/deepseek.svg"
          : normalizedLabel.includes("packy") || normalizedUrl.includes("packyapi.com")
            ? "./provider-icons/packy.svg"
            : provider?.usageAdapter === "new_api"
              ? "./provider-icons/new-api.png"
              : undefined;
  return (
    <span className="settings-provider-matrix__icon" aria-hidden="true">
      {iconUrl ? <img src={iconUrl} alt="" /> : <KeyRound />}
    </span>
  );
}

function NotProvided() {
  return (
    <span className="settings-provider-matrix__missing">
      <strong>—</strong>
      <small>Not provided</small>
    </span>
  );
}

function providerUsageTargetKey(target: ProviderUsageTarget): string {
  return `${target.source}:${target.sourceId}`;
}

function mergeProviderUsageSnapshot(
  current: ProviderUsageSnapshot | null,
  next: ProviderUsageSnapshot,
): ProviderUsageSnapshot {
  if (!current) return next;
  return {
    fetchedAt: next.fetchedAt,
    providers: mergeProviderUsageEntries(current.providers, next.providers),
    toolAccounts: mergeProviderUsageEntries(current.toolAccounts, next.toolAccounts),
  };
}

function mergeProviderUsageEntries(
  current: ProviderUsageEntry[],
  next: ProviderUsageEntry[],
): ProviderUsageEntry[] {
  const merged = new Map(current.map((entry) => [entry.sourceId, entry]));
  for (const entry of next) merged.set(entry.sourceId, entry);
  return [...merged.values()];
}

function findUsageWindow(
  entry: ProviderUsageEntry | undefined,
  slot: "five_hour" | "weekly",
): Extract<ProviderUsageMeter, { kind: "window" }> | undefined {
  return entry?.meters.find((meter): meter is Extract<ProviderUsageMeter, { kind: "window" }> => {
    if (meter.kind !== "window") return false;
    const identity = `${meter.id} ${meter.label}`.toLowerCase();
    return slot === "five_hour"
      ? /five[_ -]?hour|5[_ -]?hour/.test(identity)
      : /weekly|7[_ -]?day/.test(identity);
  });
}

function providerUsageStatus(
  entry: ProviderUsageEntry | undefined,
  loading: boolean,
): { label: string; tone: "ready" | "loading" | "muted" | "warning" | "error" } {
  if (loading) return { label: entry ? "Updating" : "Checking", tone: "loading" };
  if (!entry) return { label: "Not checked", tone: "muted" };
  if (entry.status === "ready") {
    return entry.meters.length > 0 || entry.account
      ? { label: "Ready", tone: "ready" }
      : { label: "No data", tone: "muted" };
  }
  if (entry.status === "unsupported") return { label: "Not supported", tone: "muted" };
  if (entry.status === "unavailable") return { label: "Unavailable", tone: "warning" };
  return { label: "Query failed", tone: "error" };
}

function providerFinanceSummary(
  entry: ProviderUsageEntry | undefined,
): { primary: string; caption: string; lines: string[] } | undefined {
  const accountBalance = entry?.account?.balance;
  const balances = (entry?.meters ?? []).filter(
    (meter): meter is Extract<ProviderUsageMeter, { kind: "balance" }> => meter.kind === "balance",
  );
  const credits = (entry?.meters ?? []).filter(
    (meter): meter is Extract<ProviderUsageMeter, { kind: "credit" }> =>
      meter.kind === "credit" && !/reset/i.test(meter.label),
  );
  if (!accountBalance && balances.length === 0 && credits.length === 0) return undefined;

  const lines: string[] = [];
  if (accountBalance) {
    lines.push(
      `Account available: ${formatProviderUnitAmount(accountBalance.unit, accountBalance.remaining)}`,
      `Account used: ${formatProviderUnitAmount(accountBalance.unit, accountBalance.used)}`,
      `Account total: ${formatProviderUnitAmount(accountBalance.unit, accountBalance.total)}`,
    );
  }
  for (const balance of balances) {
    if (!(accountBalance && balance.label.toLowerCase() === "account balance")) {
      lines.push(`${balance.label}: ${formatProviderAmount(balance.currency, balance.total)}`);
    }
    if (balance.granted !== undefined) {
      lines.push(`Granted: ${formatProviderAmount(balance.currency, balance.granted)}`);
    }
    if (balance.toppedUp !== undefined) {
      lines.push(`Paid: ${formatProviderAmount(balance.currency, balance.toppedUp)}`);
    }
  }
  for (const credit of credits) {
    lines.push(`${credit.label}: ${credit.remaining} ${credit.unit}`);
  }

  if (accountBalance) {
    return {
      primary: formatProviderUnitAmount(accountBalance.unit, accountBalance.remaining),
      caption: "Account available",
      lines,
    };
  }
  const balance = balances[0];
  if (balance) {
    return {
      primary: formatProviderAmount(balance.currency, balance.total),
      caption: balances.length > 1 ? `${balances.length} balances` : balance.label,
      lines,
    };
  }
  const credit = credits[0];
  if (!credit) return undefined;
  return {
    primary: credit.remaining,
    caption: credit.label,
    lines,
  };
}

function formatProviderUnitAmount(unit: string, value: string): string {
  return /^[A-Z]{3}$/.test(unit) ? formatProviderAmount(unit, value) : `${value} ${unit}`;
}

function isDeepSeekProvider(provider: ExtensionProviderSummary | undefined): boolean {
  return !!provider && isDeepSeekProviderUrl(provider.baseUrl ?? "");
}

function isOpenAIProvider(provider: ExtensionProviderSummary | undefined): boolean {
  if (!provider?.baseUrl) return false;
  try {
    return new URL(provider.baseUrl).hostname.toLowerCase() === "api.openai.com";
  } catch {
    return provider.baseUrl.toLowerCase().includes("api.openai.com");
  }
}

function isDeepSeekProviderUrl(value: string): boolean {
  try {
    return new URL(value).hostname.toLowerCase() === "api.deepseek.com";
  } catch {
    return value.toLowerCase().includes("api.deepseek.com");
  }
}

function isOpenCodeGoProviderUrl(value: string): boolean {
  try {
    const url = new URL(value);
    const pathname = url.pathname.replace(/\/+$/, "");
    return (
      url.protocol === "https:" &&
      url.hostname.toLowerCase() === "opencode.ai" &&
      !url.port &&
      (pathname === "/zen/go" || pathname === "/zen/go/v1")
    );
  } catch {
    return false;
  }
}

function providerProtocolLabel(value: string): string {
  if (value === "anthropic") return "Anthropic";
  if (value === "openai_chat") return "OpenAI Chat";
  if (value === "openai_responses") return "OpenAI Responses";
  if (value === "ollama") return "Ollama";
  return value.replaceAll("_", " ");
}

function capitalize(value: string): string {
  return value.length > 0 ? `${value[0]?.toUpperCase()}${value.slice(1)}` : value;
}

function maskProviderTokenId(value: string): string {
  return value.length > 10 ? `${value.slice(0, 4)}…${value.slice(-4)}` : value;
}

function formatProviderAmount(currency: string, value: string): string {
  const amount = Number(value);
  if (!Number.isFinite(amount)) return `${currency} ${value}`;
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency,
      maximumFractionDigits: 6,
    }).format(amount);
  } catch {
    return `${currency} ${value}`;
  }
}

function formatUsagePercent(value: number): string {
  return `${Math.max(0, Math.min(200, value)).toFixed(value % 1 === 0 ? 0 : 1)}%`;
}

function formatUsageReset(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function AgentPicker({
  open,
  section,
  harnesses,
  selectedHarness,
  models,
  selectedModel,
  efforts,
  selectedEffort,
  modelStatusText,
  modelCatalog,
  modelCatalogRefreshing,
  modelCatalogError,
  disabled,
  label,
  onOpenChange,
  onSectionChange,
  onHarnessChange,
  onModelChange,
  onEffortChange,
  onRefreshModels,
  onAddManualModel,
  onRemoveManualModel,
}: {
  open: boolean;
  section: "harness" | "model" | "effort";
  harnesses: HarnessOption[];
  selectedHarness: HarnessOption;
  models: ComposerModelOption[];
  selectedModel: ComposerModelOption | null;
  efforts: string[];
  selectedEffort: string | null;
  modelStatusText: string;
  modelCatalog?: ModelCatalogSummary;
  modelCatalogRefreshing: boolean;
  modelCatalogError: string | null;
  disabled: boolean;
  label: string;
  onOpenChange: (open: boolean) => void;
  onSectionChange: (section: "harness" | "model" | "effort") => void;
  onHarnessChange: (harnessId: string) => void;
  onModelChange: (modelId: string) => void;
  onEffortChange: (effort: string) => void;
  onRefreshModels: () => Promise<void>;
  onAddManualModel: (input: ManualModelInput) => Promise<void>;
  onRemoveManualModel: (modelId: string) => Promise<void>;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [menuGeometry, setMenuGeometry] = useState<{
    inlineOffset: number;
    secondarySide: "left" | "right";
  }>({ inlineOffset: 0, secondarySide: "right" });
  const [modelQuery, setModelQuery] = useState("");
  const [manualModelOpen, setManualModelOpen] = useState(false);
  const [manualModelId, setManualModelId] = useState("");
  const [manualModelLabel, setManualModelLabel] = useState("");
  const [manualRuntimeModel, setManualRuntimeModel] = useState("");
  const [manualApiProtocol, setManualApiProtocol] = useState<ModelApiProtocol>("openai_responses");
  const [manualModelSaving, setManualModelSaving] = useState(false);
  const [manualModelError, setManualModelError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) onOpenChange(false);
    };
    window.addEventListener("pointerdown", close);
    return () => window.removeEventListener("pointerdown", close);
  }, [onOpenChange, open]);

  useEffect(() => {
    if (!open) setModelQuery("");
  }, [open]);

  useLayoutEffect(() => {
    if (!open) return;
    const root = rootRef.current;
    const menu = menuRef.current;
    if (!root || !menu) return;

    const updateGeometry = () => {
      const styles = window.getComputedStyle(menu);
      const cssPixels = (property: string, fallback: number) => {
        const value = Number.parseFloat(styles.getPropertyValue(property));
        return Number.isFinite(value) ? value : fallback;
      };
      const primaryWidth = cssPixels("--agent-picker-primary-width", 196);
      const secondaryWidth = cssPixels("--agent-picker-secondary-width", 236);
      const panelGap = cssPixels("--agent-picker-panel-gap", 6);
      const viewportMargin = 12;
      const anchorLeft = root.getBoundingClientRect().left;
      const maximumPrimaryLeft = Math.max(
        viewportMargin,
        window.innerWidth - viewportMargin - primaryWidth,
      );
      const primaryLeft = Math.min(Math.max(anchorLeft, viewportMargin), maximumPrimaryLeft);
      const availableRight =
        window.innerWidth - viewportMargin - (primaryLeft + primaryWidth + panelGap);
      const availableLeft = primaryLeft - viewportMargin - panelGap;
      const secondarySide =
        availableRight >= secondaryWidth || availableRight >= availableLeft ? "right" : "left";
      const inlineOffset = Math.round(primaryLeft - anchorLeft);

      setMenuGeometry((current) =>
        current.inlineOffset === inlineOffset && current.secondarySide === secondarySide
          ? current
          : { inlineOffset, secondarySide },
      );
    };

    updateGeometry();
    window.addEventListener("resize", updateGeometry);
    const resizeObserver =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(updateGeometry);
    resizeObserver?.observe(root);
    return () => {
      window.removeEventListener("resize", updateGeometry);
      resizeObserver?.disconnect();
    };
  }, [open]);

  const focusFirstPrimaryItem = useCallback(() => {
    window.requestAnimationFrame(() => {
      rootRef.current
        ?.querySelector<HTMLButtonElement>(".agent-picker__primary button:not(:disabled)")
        ?.focus();
    });
  }, []);

  const handleMenuKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      const target = event.target as HTMLElement;
      if (event.key === "Escape") {
        event.preventDefault();
        onOpenChange(false);
        triggerRef.current?.focus();
        return;
      }

      if (target instanceof HTMLInputElement) {
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          const modelItems = Array.from(
            menuRef.current?.querySelectorAll<HTMLButtonElement>(
              ".agent-picker__secondary button:not(:disabled)",
            ) ?? [],
          );
          if (modelItems.length > 0) {
            event.preventDefault();
            modelItems[event.key === "ArrowUp" ? modelItems.length - 1 : 0]?.focus();
          }
        }
        return;
      }

      if (event.key === "ArrowRight" && target.closest(".agent-picker__primary")) {
        const button = target.closest<HTMLButtonElement>("button:not(:disabled)");
        if (!button) return;
        event.preventDefault();
        button.click();
        window.requestAnimationFrame(() => {
          menuRef.current
            ?.querySelector<HTMLButtonElement>(".agent-picker__secondary button:not(:disabled)")
            ?.focus();
        });
        return;
      }

      if (event.key === "ArrowLeft" && target.closest(".agent-picker__secondary")) {
        event.preventDefault();
        menuRef.current
          ?.querySelector<HTMLButtonElement>(".agent-picker__primary .is-active:not(:disabled)")
          ?.focus();
        return;
      }

      if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
      const items = Array.from(
        event.currentTarget.querySelectorAll<HTMLButtonElement>("button:not(:disabled)"),
      );
      if (items.length === 0) return;
      event.preventDefault();
      const currentIndex = items.indexOf(target.closest("button") as HTMLButtonElement);
      const nextIndex =
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? items.length - 1
            : event.key === "ArrowUp"
              ? currentIndex <= 0
                ? items.length - 1
                : currentIndex - 1
              : currentIndex < 0 || currentIndex === items.length - 1
                ? 0
                : currentIndex + 1;
      items[nextIndex]?.focus();
    },
    [onOpenChange],
  );

  const submitManualModel = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setManualModelSaving(true);
    setManualModelError(null);
    try {
      await onAddManualModel({
        id: manualModelId,
        label: manualModelLabel || undefined,
        runtimeModel: manualRuntimeModel || undefined,
        apiProtocol: manualApiProtocol,
      });
      setManualModelId("");
      setManualModelLabel("");
      setManualRuntimeModel("");
      setManualModelOpen(false);
    } catch (error) {
      setManualModelError(errorMessage(error));
    } finally {
      setManualModelSaving(false);
    }
  };
  const removeManualModel = async (modelId: string) => {
    setManualModelError(null);
    try {
      await onRemoveManualModel(modelId);
    } catch (error) {
      setManualModelError(errorMessage(error));
    }
  };
  const providerErrorCount =
    modelCatalog?.providers.filter((provider) => provider.status === "error").length ?? 0;
  const discoveredModelCount =
    modelCatalog?.providers.reduce((total, provider) => total + provider.modelCount, 0) ?? 0;

  const primaryRows: Array<{
    id: "harness" | "model" | "effort";
    label: string;
    value: string;
    enabled: boolean;
  }> = [
    { id: "harness", label: "Harness", value: selectedHarness.label, enabled: true },
    {
      id: "model",
      label: "Model",
      value: selectedModel?.label ?? modelStatusText,
      enabled: true,
    },
    {
      id: "effort",
      label: "Effort",
      value: selectedEffort ? effortLabel(selectedEffort) : "Default",
      enabled: efforts.length > 0,
    },
  ];
  const modelGroups = groupComposerModels(models, modelQuery);
  const triggerModel =
    !disabled && selectedModel ? modelBrandPresentation(selectedModel) : undefined;

  return (
    <div className="agent-picker" ref={rootRef}>
      <button
        ref={triggerRef}
        type="button"
        className="agent-picker__trigger"
        aria-label="Choose agent"
        aria-haspopup="menu"
        aria-expanded={open}
        data-harness-id={selectedHarness.id}
        disabled={disabled}
        onClick={() => onOpenChange(!open)}
        onKeyDown={(event) => {
          if (event.key === "Escape" && open) {
            event.preventDefault();
            onOpenChange(false);
            return;
          }
          if (event.key !== "ArrowDown") return;
          event.preventDefault();
          onOpenChange(true);
          focusFirstPrimaryItem();
        }}
      >
        {triggerModel ? (
          <img
            className="model-brand-icon"
            src={triggerModel.iconUrl}
            alt=""
            aria-hidden="true"
            data-model-brand={triggerModel.brand}
          />
        ) : (
          <HarnessBrandIcon harness={selectedHarness} />
        )}
        <span className="agent-picker__trigger-label">{triggerModel?.label ?? label}</span>
        {!disabled && selectedModel && selectedEffort && (
          <span className="agent-picker__trigger-effort">{effortLabel(selectedEffort)}</span>
        )}
        {!disabled && <ChevronRight aria-hidden="true" />}
      </button>
      {open && !disabled && (
        <div
          ref={menuRef}
          className="agent-picker__menu"
          role="menu"
          aria-label="Agent composition"
          data-secondary-side={menuGeometry.secondarySide}
          style={
            {
              "--agent-picker-inline-offset": `${menuGeometry.inlineOffset}px`,
            } as React.CSSProperties
          }
          onKeyDown={handleMenuKeyDown}
        >
          <div className="agent-picker__primary" data-testid="agent-picker-primary">
            {primaryRows.map((row) => (
              <button
                key={row.id}
                type="button"
                role="menuitem"
                className={cx("agent-picker__row", section === row.id && "is-active")}
                disabled={!row.enabled}
                onPointerEnter={() => row.enabled && onSectionChange(row.id)}
                onClick={() => row.enabled && onSectionChange(row.id)}
              >
                {row.id === "harness" && <HarnessBrandIcon harness={selectedHarness} />}
                <span>
                  <strong>{row.label}</strong>
                  <small>{row.value}</small>
                </span>
                <ChevronRight aria-hidden="true" />
              </button>
            ))}
          </div>
          <div className="agent-picker__secondary" role="menu" aria-label={`${section} options`}>
            {section === "harness" &&
              harnesses.map((harness) => {
                return (
                  <button
                    key={harness.id}
                    type="button"
                    role="menuitemradio"
                    aria-checked={harness.id === selectedHarness.id}
                    aria-disabled={harness.disabled || undefined}
                    disabled={harness.disabled}
                    title={harness.disabledReason}
                    className={cx(
                      "agent-picker__option",
                      harness.id === selectedHarness.id && "is-selected",
                    )}
                    onClick={() => !harness.disabled && onHarnessChange(harness.id)}
                  >
                    <HarnessBrandIcon harness={harness} />
                    <span>
                      <span>{harness.label}</span>
                      {harness.disabledReason && <small>{harness.disabledReason}</small>}
                    </span>
                    {harness.id === selectedHarness.id && <CircleCheck aria-hidden="true" />}
                  </button>
                );
              })}
            {section === "model" && (
              <div className="agent-picker__model-list">
                <div className="agent-picker__model-actions">
                  <button
                    type="button"
                    className="agent-picker__model-action"
                    disabled={modelCatalogRefreshing}
                    onClick={() => void onRefreshModels()}
                  >
                    <RefreshCw
                      aria-hidden="true"
                      className={modelCatalogRefreshing ? "is-spinning" : undefined}
                    />
                    <span>{modelCatalogRefreshing ? "Refreshing" : "Refresh"}</span>
                  </button>
                  <button
                    type="button"
                    className="agent-picker__model-action"
                    aria-expanded={manualModelOpen}
                    onClick={() => setManualModelOpen((current) => !current)}
                  >
                    <Plus aria-hidden="true" />
                    <span>Add model</span>
                  </button>
                </div>
                <output className="agent-picker__model-status">
                  {modelCatalogRefreshing
                    ? "Refreshing Provider APIs…"
                    : providerErrorCount > 0
                      ? `${providerErrorCount} Provider refresh${providerErrorCount === 1 ? "" : "es"} failed; cached Models retained.`
                      : modelCatalog
                        ? `${discoveredModelCount} discovered · ${modelCatalog.manualModelIds.length} manual`
                        : "Provider discovery has not run yet."}
                </output>
                {(modelCatalogError || manualModelError) && (
                  <div className="agent-picker__model-error" role="alert">
                    {manualModelError ?? modelCatalogError}
                  </div>
                )}
                {manualModelOpen && (
                  <form
                    className="agent-picker__manual-model"
                    aria-label="Add manual model"
                    onSubmit={(event) => void submitManualModel(event)}
                    onKeyDown={(event) => event.stopPropagation()}
                  >
                    <label>
                      <span>Model ID</span>
                      <input
                        required
                        value={manualModelId}
                        placeholder="vendor-model-id"
                        onChange={(event) => setManualModelId(event.target.value)}
                      />
                    </label>
                    <label>
                      <span>Runtime model</span>
                      <input
                        value={manualRuntimeModel}
                        placeholder="Defaults to Model ID"
                        onChange={(event) => setManualRuntimeModel(event.target.value)}
                      />
                    </label>
                    <label>
                      <span>Display name</span>
                      <input
                        value={manualModelLabel}
                        placeholder="Optional"
                        onChange={(event) => setManualModelLabel(event.target.value)}
                      />
                    </label>
                    <label>
                      <span>API protocol</span>
                      <select
                        value={manualApiProtocol}
                        onChange={(event) =>
                          setManualApiProtocol(event.target.value as ModelApiProtocol)
                        }
                      >
                        <option value="openai_responses">OpenAI Responses</option>
                        <option value="openai_chat">OpenAI Chat</option>
                        <option value="anthropic">Anthropic</option>
                        <option value="ollama">Ollama</option>
                      </select>
                    </label>
                    <div className="agent-picker__manual-model-actions">
                      <button type="button" onClick={() => setManualModelOpen(false)}>
                        Cancel
                      </button>
                      <button type="submit" disabled={manualModelSaving || !manualModelId.trim()}>
                        {manualModelSaving ? "Saving…" : "Save model"}
                      </button>
                    </div>
                  </form>
                )}
                {(modelCatalog?.manualModelIds.length ?? 0) > 0 && (
                  <div className="agent-picker__manual-model-list" aria-label="Manual models">
                    {modelCatalog?.manualModelIds.map((modelId) => (
                      <button
                        key={modelId}
                        type="button"
                        aria-label={`Remove manual model ${modelId}`}
                        onClick={() => void removeManualModel(modelId)}
                      >
                        <span>{modelId}</span>
                        <Trash2 aria-hidden="true" />
                      </button>
                    ))}
                  </div>
                )}
                {models.length > 0 ? (
                  <>
                    <label
                      className="agent-picker__model-search"
                      onKeyDown={(event) => event.stopPropagation()}
                    >
                      <Search aria-hidden="true" />
                      <input
                        type="search"
                        value={modelQuery}
                        placeholder="Search models"
                        aria-label="Search models"
                        onChange={(event) => setModelQuery(event.target.value)}
                      />
                    </label>
                    {modelGroups.map((group) => (
                      <fieldset key={group.id} className="agent-picker__model-group">
                        <legend
                          id={`model-provider-${domId(group.id)}`}
                          className="agent-picker__model-group-label"
                        >
                          <ProviderBrandIcon
                            label={group.label}
                            sourceId={group.id}
                            provider={group.provider}
                          />
                          <span>{group.label}</span>
                        </legend>
                        {group.subgroups.map((subgroup) => (
                          <div
                            key={subgroup.id}
                            className="agent-picker__model-subgroup"
                            {...(subgroup.label
                              ? { role: "group", "aria-label": subgroup.label }
                              : {})}
                          >
                            {subgroup.label && (
                              <span className="agent-picker__model-subgroup-label">
                                {subgroup.label}
                              </span>
                            )}
                            {subgroup.models.map((model) => (
                              <button
                                key={model.id}
                                type="button"
                                role="menuitemradio"
                                title={model.modelId}
                                aria-checked={model.id === selectedModel?.id}
                                className={cx(
                                  "agent-picker__option",
                                  model.id === selectedModel?.id && "is-selected",
                                )}
                                onClick={() => onModelChange(model.id)}
                              >
                                <span>
                                  <span>{model.label}</span>
                                  {model.manual && <small>Manual</small>}
                                </span>
                                {model.id === selectedModel?.id && (
                                  <CircleCheck aria-hidden="true" />
                                )}
                              </button>
                            ))}
                          </div>
                        ))}
                      </fieldset>
                    ))}
                    {modelGroups.length === 0 && (
                      <div className="agent-picker__empty">No models match “{modelQuery}”</div>
                    )}
                  </>
                ) : (
                  <div className="agent-picker__empty">
                    No compatible Models. Refresh Provider APIs or add one manually.
                  </div>
                )}
              </div>
            )}
            {section === "effort" &&
              efforts.map((effort) => (
                <button
                  key={effort}
                  type="button"
                  role="menuitemradio"
                  aria-checked={effort === selectedEffort}
                  className={cx("agent-picker__option", effort === selectedEffort && "is-selected")}
                  onClick={() => onEffortChange(effort)}
                >
                  <Sparkles aria-hidden="true" />
                  <span>{effortLabel(effort)}</span>
                  {effort === selectedEffort && <CircleCheck aria-hidden="true" />}
                </button>
              ))}
            {section === "effort" && efforts.length === 0 && (
              <div className="agent-picker__empty">This model has no verified effort control</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function effortLabel(effort: string): string {
  if (effort === "xhigh") return "Extra High";
  return effort.charAt(0).toUpperCase() + effort.slice(1);
}

function resolveComposerModelOptions(
  inventory: ExtensionCapabilityInventory,
  harnessId: string,
): ComposerModelOption[] {
  const providersById = new Map(inventory.providers.map((provider) => [provider.id, provider]));
  return resolveHarnessModelInventory({
    harnessId,
    models: inventory.models,
    supplies: inventory.modelSupplies,
    providers: inventory.providers,
    harnesses: inventory.harnesses,
  }).flatMap((model): ComposerModelOption[] => {
    const manual = inventory.modelCatalog?.manualModelIds.includes(model.modelId) ?? false;
    const supplies = preferredComposerModelSupplies(model.supplies, model.apiProtocol);
    if (supplies.length === 0) {
      return [
        {
          id: model.agentId,
          label: model.modelLabel,
          modelId: model.modelId,
          runtimeModel: model.runtimeModel,
          apiProtocol: model.apiProtocol,
          providerId: manual ? "manual" : "unrouted",
          providerLabel: manual ? "Manual" : "Other",
          manual,
          reasoning: composerReasoning(model.reasoning),
        },
      ];
    }
    return supplies.map((supply) => {
      const provider = providersById.get(supply.providerProfileId);
      return {
        id: `${model.agentId}@${supply.id}`,
        label: model.modelLabel,
        modelId: model.modelId,
        modelSupplyId: supply.id,
        runtimeModel: supply.runtimeModel,
        apiProtocol: supply.apiProtocol,
        providerId: supply.providerProfileId,
        providerLabel: supply.providerLabel ?? provider?.label ?? supply.providerProfileId,
        providerGroup: supply.providerGroup,
        provider,
        manual,
        reasoning: composerReasoning(supply.reasoning ?? model.reasoning),
      };
    });
  });
}

function composerModelOptionId(harnessId: string, modelId: string, modelSupplyId?: string): string {
  const agentId = `${harnessId}:${modelId}`;
  return modelSupplyId ? `${agentId}@${modelSupplyId}` : agentId;
}

interface ResolvedComposerSupply {
  id: string;
  providerProfileId: string;
  providerLabel?: string;
  providerKind?: string;
  providerGroup?: string;
  runtimeModel: string;
  apiProtocol: string;
  reasoning?: {
    supportedEfforts: string[];
    defaultEffort?: string;
  };
}

function preferredComposerModelSupplies(
  supplies: ResolvedComposerSupply[],
  preferredApi: string,
): ResolvedComposerSupply[] {
  const selected = new Map<string, ResolvedComposerSupply>();
  for (const supply of supplies) {
    const key = `${supply.providerProfileId}\u0000${supply.providerGroup ?? ""}`;
    const current = selected.get(key);
    if (
      !current ||
      composerSupplyRank(supply, preferredApi) < composerSupplyRank(current, preferredApi)
    ) {
      selected.set(key, supply);
    }
  }
  return [...selected.values()];
}

function composerSupplyRank(supply: ResolvedComposerSupply, preferredApi: string): number {
  if (supply.providerKind && supply.apiProtocol === supply.providerKind) return 0;
  if (supply.apiProtocol === preferredApi) return 1;
  return 2;
}

function composerReasoning(
  reasoning: { supportedEfforts: string[]; defaultEffort?: string } | null | undefined,
): ComposerModelOption["reasoning"] {
  return selectableModelReasoning(reasoning);
}

interface ComposerModelSubgroup {
  id: string;
  label?: string;
  models: ComposerModelOption[];
}

interface ComposerModelGroup {
  id: string;
  label: string;
  provider?: ExtensionProviderSummary;
  subgroups: ComposerModelSubgroup[];
}

type MutableComposerModelGroup = ComposerModelGroup & {
  subgroupMap: Map<string, ComposerModelSubgroup>;
};

function groupComposerModels(models: ComposerModelOption[], query: string): ComposerModelGroup[] {
  const normalizedQuery = query.trim().toLowerCase();
  const groups = new Map<string, MutableComposerModelGroup>();

  for (const model of models) {
    if (
      normalizedQuery &&
      !`${model.providerLabel} ${model.providerGroup ?? ""} ${model.apiProtocol} ${model.label} ${model.modelId} ${model.runtimeModel}`
        .toLowerCase()
        .includes(normalizedQuery)
    ) {
      continue;
    }
    const group: MutableComposerModelGroup = groups.get(model.providerId) ?? {
      id: model.providerId,
      label: model.providerLabel,
      provider: model.provider,
      subgroups: [],
      subgroupMap: new Map<string, ComposerModelSubgroup>(),
    };
    const subgroupId = model.providerGroup ?? "default";
    const subgroup: ComposerModelSubgroup = group.subgroupMap.get(subgroupId) ?? {
      id: subgroupId,
      label: model.providerGroup,
      models: [],
    };
    subgroup.models.push(model);
    group.subgroupMap.set(subgroupId, subgroup);
    if (!group.subgroups.includes(subgroup)) group.subgroups.push(subgroup);
    groups.set(model.providerId, group);
  }

  return [...groups.values()].map(({ subgroupMap: _subgroupMap, ...group }) => ({
    ...group,
    subgroups: group.subgroups.map((subgroup) => ({
      ...subgroup,
      models: [...subgroup.models].sort(compareModelDisplayOrder),
    })),
  }));
}

function modelApiLabel(api: string): string {
  switch (api) {
    case "openai_chat":
      return "OpenAI Chat API";
    case "openai_responses":
      return "OpenAI Responses API";
    case "anthropic":
      return "Anthropic API";
    case "ollama":
      return "Ollama API";
    default:
      return api;
  }
}

function nativeAgentHostLabel(
  host: "claude_code" | "codex" | "swarmx" | "custom" | undefined,
): string {
  if (host === "codex") return "Codex";
  if (host === "claude_code") return "Claude Code";
  return "Native Agent";
}

function domId(value: string): string {
  return value.replace(/[^a-zA-Z0-9_-]+/g, "-");
}

function slugId(value: string, fallback: string): string {
  const slug = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug || fallback;
}

function lines(value: string): string[] {
  return [
    ...new Set(
      value
        .split(/\r?\n|,/)
        .map((item) => item.trim())
        .filter(Boolean),
    ),
  ];
}

function uniqueById<T extends { id: string }>(items: T[]): T[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (seen.has(item.id)) return false;
    seen.add(item.id);
    return true;
  });
}

function HarnessBrandIcon({ harness }: { harness: HarnessOption }) {
  const [failedIconUrl, setFailedIconUrl] = useState<string | null>(null);
  const Fallback = harness.icon;
  const iconUrl = PACKAGED_HARNESS_ICON_URLS[harness.id];
  if (!iconUrl || failedIconUrl === iconUrl) {
    return <Fallback aria-hidden="true" data-harness-icon-fallback={harness.id} />;
  }
  return (
    <img
      className="harness-brand-icon"
      src={iconUrl}
      alt=""
      aria-hidden="true"
      data-harness-icon={harness.id}
      onError={() => setFailedIconUrl(iconUrl)}
    />
  );
}

function harnessOption(id: string, label: string): HarnessOption {
  return (
    HARNESSES.find((harness) => harness.id === id) ?? {
      id,
      label,
      icon: Bot,
      modelControl: "session",
    }
  );
}

function ExtensionWorkspace({
  inventory,
  management,
  loading,
  error,
  selectedAgentId,
  onSelectAgent,
  onSaveSource,
  onRefreshSource,
  onRemoveSource,
  onApplyAction,
  onSaveEvolutionPolicy,
}: {
  inventory?: ExtensionCapabilityInventory;
  management?: ExtensionManagementState;
  loading: boolean;
  error: unknown;
  selectedAgentId: string | null;
  onSelectAgent: (agentId: string) => void;
  onSaveSource: (input: unknown) => Promise<void>;
  onRefreshSource: (id: string) => Promise<void>;
  onRemoveSource: (id: string) => Promise<void>;
  onApplyAction: (input: unknown) => Promise<{
    status: "applied" | "rejected" | "failed";
    message: string;
  }>;
  onSaveEvolutionPolicy: (input: {
    enabled: boolean;
    promotionGate: "human" | "policy";
  }) => Promise<void>;
}) {
  const [sourceFormOpen, setSourceFormOpen] = useState(false);
  const [sourceName, setSourceName] = useState("");
  const [sourceLocation, setSourceLocation] = useState("");
  const [sourceKind, setSourceKind] = useState<"remote_catalog" | "registry" | "local_path">(
    "remote_catalog",
  );
  const [managementBusy, setManagementBusy] = useState<string | null>(null);
  const [managementError, setManagementError] = useState<string | null>(null);
  const bundles = inventory?.bundles ?? [];
  const harnesses = inventory?.harnesses ?? [];
  const agents = inventory?.agents ?? [];
  const models = inventory?.models ?? [];
  const modelSupplies = inventory?.modelSupplies ?? [];
  const providers = inventory?.providers ?? [];
  const skills = inventory?.skills ?? [];
  const mcpServers = inventory?.mcpServers ?? [];
  const appConnectors = inventory?.appConnectors ?? [];
  const uiContributions = inventory?.uiContributions ?? [];
  const agentPlans = inventory?.agentPlans ?? [];
  const pluginComponents = extensionComponentRows(inventory);
  const marketplaceSources: ExtensionMarketplaceSourceSummary[] = [
    ...(management?.sources.map((source) => ({
      ...source,
      host: "swarmx",
      ...(source.kind === "local_path" ? { path: source.location } : { url: source.location }),
    })) ?? []),
    ...(inventory?.marketplaceSources ?? []).filter(
      (source) => !management?.sources.some((managed) => managed.id === source.id),
    ),
  ];
  const candidateById = new Map(
    (management?.candidates ?? []).map((candidate) => [candidate.pluginId, candidate]),
  );
  const pluginCatalog = uniqueById<ExtensionPluginCatalogEntrySummary>([
    ...(management?.candidates ?? []).map<ExtensionPluginCatalogEntrySummary>((candidate) => ({
      id: candidate.pluginId,
      name: candidate.name,
      version: candidate.revision.version,
      marketplaceSourceId: candidate.revision.sourceId,
      trust: candidate.trust,
      installState: "available",
      updateState: "unknown",
      description: candidate.description,
    })),
    ...(inventory?.pluginCatalog ?? []),
  ]);
  const warnings = inventory?.warnings ?? [];
  const installedById = new Map(
    (management?.installed ?? []).map((plugin) => [plugin.pluginId, plugin]),
  );
  const planByAgentId = useMemo(
    () => new Map(agentPlans.map((plan) => [plan.agentProfileId ?? plan.agentId, plan] as const)),
    [agentPlans],
  );

  const runManagement = async (id: string, action: () => Promise<void>) => {
    setManagementBusy(id);
    setManagementError(null);
    try {
      await action();
    } catch (error) {
      setManagementError(errorMessage(error));
    } finally {
      setManagementBusy(null);
    }
  };
  const submitSource = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const id = slugId(sourceName || sourceLocation, "source");
    await runManagement(`source:${id}`, async () => {
      await onSaveSource({
        id,
        name: sourceName.trim(),
        kind: sourceKind,
        location: sourceLocation.trim(),
        trust: sourceKind === "local_path" ? "local" : "untrusted",
      });
      setSourceFormOpen(false);
      setSourceName("");
      setSourceLocation("");
    });
  };
  const applyCatalogAction = async (
    entry: ExtensionPluginCatalogEntrySummary,
    action: "install" | "update" | "enable" | "disable" | "rollback" | "uninstall",
  ) => {
    if (
      ["install", "update", "rollback", "uninstall"].includes(action) &&
      !window.confirm(`${capitalize(action)} ${entry.name}?`)
    ) {
      return;
    }
    await runManagement(`plugin:${entry.id}`, async () => {
      const version = entry.version ?? "0.0.0";
      const managedCandidate = candidateById.get(entry.id);
      const result = await onApplyAction({
        action,
        pluginId: entry.id,
        confirmed: true,
        ...(["install", "update"].includes(action)
          ? {
              candidate: managedCandidate ?? {
                pluginId: entry.id,
                name: entry.name,
                trust: entry.trust ?? "untrusted",
                revision: {
                  revisionId: `${entry.id}@${version}`,
                  version,
                  contentDigest: `catalog:${entry.id}@${version}`,
                  sourceId: entry.marketplaceSourceId ?? "local",
                },
              },
            }
          : {}),
      });
      if (result.status !== "applied") throw new Error(result.message);
    });
  };

  return (
    <section className="extension-workspace" aria-label="Extension inventory">
      <div className="extension-topbar">
        <div className="extension-title">
          <Package aria-hidden="true" />
          <div>
            <h2>Extensions</h2>
            <span>
              {loading
                ? "Loading inventory"
                : `${bundles.length} bundles / ${marketplaceSources.length} sources / ${harnesses.length} harnesses`}
            </span>
          </div>
        </div>
        <div className="extension-stats" aria-label="Extension counts">
          <Badge tone="neutral">{pluginCatalog.length} plugins</Badge>
          <Badge tone="neutral">{agents.length} agents</Badge>
          <Badge tone="neutral">{skills.length} skills</Badge>
          <Badge tone="neutral">{mcpServers.length} MCPs</Badge>
          <Badge tone="neutral">{uiContributions.length} UI</Badge>
          <Badge tone="neutral">{pluginComponents.length} components</Badge>
          {warnings.length > 0 && <Badge tone="danger">{warnings.length} warnings</Badge>}
          <button
            type="button"
            className="settings-primary-action"
            onClick={() => setSourceFormOpen((open) => !open)}
          >
            <Plus aria-hidden="true" />
            Add source
          </button>
        </div>
      </div>

      {Boolean(managementError || error) && (
        <div className="settings-provider-error" role="alert">
          {managementError ?? errorMessage(error)}
        </div>
      )}

      {sourceFormOpen && (
        <form className="extension-source-form" onSubmit={(event) => void submitSource(event)}>
          <label>
            <span>Source name</span>
            <input
              required
              value={sourceName}
              placeholder="Official marketplace"
              onChange={(event) => setSourceName(event.target.value)}
            />
          </label>
          <label>
            <span>Source type</span>
            <select
              value={sourceKind}
              onChange={(event) =>
                setSourceKind(event.target.value as "remote_catalog" | "registry" | "local_path")
              }
            >
              <option value="remote_catalog">Remote catalog</option>
              <option value="registry">Registry</option>
              <option value="local_path">Local path</option>
            </select>
          </label>
          <label className="is-wide">
            <span>{sourceKind === "local_path" ? "Path" : "HTTPS URL"}</span>
            <input
              required
              type={sourceKind === "local_path" ? "text" : "url"}
              value={sourceLocation}
              placeholder={
                sourceKind === "local_path"
                  ? "/Users/me/extensions"
                  : "https://plugins.example.com/catalog.json"
              }
              onChange={(event) => setSourceLocation(event.target.value)}
            />
          </label>
          <div className="extension-source-form__actions">
            <Button type="button" variant="ghost" onClick={() => setSourceFormOpen(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={managementBusy !== null}>
              Save source
            </Button>
          </div>
        </form>
      )}

      <div className="extension-layout">
        <section className="extension-section" aria-label="Plugin bundles">
          <div className="extension-section__header">
            <h3>Plugin bundles</h3>
            <span>{bundles.length}</span>
          </div>
          {bundles.length === 0 ? (
            <div className="extension-empty">No bundles</div>
          ) : (
            <ul className="extension-list">
              {bundles.map((bundle) => (
                <li key={bundle.id} className="extension-item">
                  <div className="extension-item__main">
                    <strong>{bundle.name}</strong>
                    <span>{bundle.id}</span>
                  </div>
                  <div className="extension-item__meta">
                    <span>{bundle.version}</span>
                    <span>{bundle.trust ?? "local"}</span>
                    {bundle.readOnly && <span>read-only</span>}
                  </div>
                  <div className="extension-item__chips">
                    <span>{capabilityCount(bundle, "harnesses")} harnesses</span>
                    <span>{capabilityCount(bundle, "agents")} agents</span>
                    <span>{capabilityCount(bundle, "skills")} skills</span>
                    <span>{capabilityCount(bundle, "mcpServers")} MCPs</span>
                    <span>{capabilityCount(bundle, "commands")} commands</span>
                    <span>{capabilityCount(bundle, "lspServers")} LSPs</span>
                    <span>{capabilityCount(bundle, "hooks")} hooks</span>
                    <span>{capabilityCount(bundle, "uiContributions")} UI</span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="extension-section" aria-label="Marketplace sources">
          <div className="extension-section__header">
            <h3>Marketplace sources</h3>
            <span>{marketplaceSources.length}</span>
          </div>
          {marketplaceSources.length === 0 ? (
            <div className="extension-empty">No marketplace sources</div>
          ) : (
            <ul className="extension-list">
              {marketplaceSources.map((source) => (
                <li key={source.id} className="extension-item">
                  <div className="extension-item__main">
                    <strong>{source.name}</strong>
                    <span>{source.id}</span>
                    {management?.sources.some((managed) => managed.id === source.id) && (
                      <span className="extension-item__actions">
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={managementBusy === `source:${source.id}`}
                          onClick={() =>
                            void runManagement(`source:${source.id}`, () =>
                              onRefreshSource(source.id),
                            )
                          }
                        >
                          <RefreshCw aria-hidden="true" />
                          Refresh
                        </Button>
                        {!source.readOnly && (
                          <Button
                            size="sm"
                            variant="ghost"
                            disabled={managementBusy === `source:${source.id}`}
                            onClick={() =>
                              void runManagement(`source:${source.id}`, () =>
                                onRemoveSource(source.id),
                              )
                            }
                          >
                            Remove
                          </Button>
                        )}
                      </span>
                    )}
                  </div>
                  <div className="extension-item__meta">
                    <span>{source.host ?? "custom"}</span>
                    <span>{source.kind ?? "local_path"}</span>
                    <span>{source.trust ?? "local"}</span>
                    {source.enabled === false && <span>disabled</span>}
                    {source.readOnly && <span>read-only</span>}
                  </div>
                  <div className="extension-item__chips">
                    <span>{source.path ?? source.url ?? source.package ?? "host-native"}</span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="extension-section" aria-label="Plugin catalog">
          <div className="extension-section__header">
            <h3>Plugin catalog</h3>
            <span>{pluginCatalog.length}</span>
          </div>
          {pluginCatalog.length === 0 ? (
            <div className="extension-empty">No plugin catalog entries</div>
          ) : (
            <ul className="extension-list">
              {pluginCatalog.map((entry) => {
                const installed = installedById.get(entry.id);
                const updateAvailable = Boolean(
                  installed &&
                    entry.version &&
                    installed.currentRevision?.version !== entry.version,
                );
                return (
                  <li key={entry.id} className="extension-item">
                    <div className="extension-item__main">
                      <strong>{entry.name}</strong>
                      <span>{entry.id}</span>
                      <span className="extension-item__actions">
                        {!installed ? (
                          <Button
                            size="sm"
                            disabled={managementBusy === `plugin:${entry.id}`}
                            onClick={() => void applyCatalogAction(entry, "install")}
                          >
                            Install
                          </Button>
                        ) : (
                          <>
                            <Button
                              size="sm"
                              variant="secondary"
                              disabled={managementBusy === `plugin:${entry.id}`}
                              onClick={() =>
                                void applyCatalogAction(
                                  entry,
                                  installed.enabled ? "disable" : "enable",
                                )
                              }
                            >
                              {installed.enabled ? "Disable" : "Enable"}
                            </Button>
                            {updateAvailable && (
                              <Button
                                size="sm"
                                disabled={managementBusy === `plugin:${entry.id}`}
                                onClick={() => void applyCatalogAction(entry, "update")}
                              >
                                Update
                              </Button>
                            )}
                            {installed.previousRevisions.length > 0 && (
                              <Button
                                size="sm"
                                variant="ghost"
                                disabled={managementBusy === `plugin:${entry.id}`}
                                onClick={() => void applyCatalogAction(entry, "rollback")}
                              >
                                Roll back
                              </Button>
                            )}
                            <Button
                              size="sm"
                              variant="ghost"
                              disabled={managementBusy === `plugin:${entry.id}`}
                              onClick={() => void applyCatalogAction(entry, "uninstall")}
                            >
                              Uninstall
                            </Button>
                          </>
                        )}
                      </span>
                    </div>
                    <div className="extension-item__meta">
                      {entry.version && <span>{entry.version}</span>}
                      <span>{installed?.state ?? entry.installState ?? "available"}</span>
                      <span>{entry.updateState ?? "unknown"}</span>
                      <span>{entry.trust ?? "local"}</span>
                      {entry.providesHarness && <span>runnable harness</span>}
                      {entry.readOnly && <span>read-only</span>}
                    </div>
                    <div className="extension-item__chips">
                      {(entry.hosts ?? []).map((host) => (
                        <span key={`${entry.id}:${host}`}>{host}</span>
                      ))}
                      {entry.marketplaceSourceId && <span>{entry.marketplaceSourceId}</span>}
                      {entry.bundleId && <span>{entry.bundleId}</span>}
                      {formatComponentCounts(entry.componentCounts).map((item) => (
                        <span key={`${entry.id}:${item}`}>{item}</span>
                      ))}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section
          className="extension-section extension-section--evolution"
          aria-label="Skill evolution"
        >
          <div className="extension-section__header">
            <h3>Skill evolution</h3>
            <Badge tone={management?.skillEvolutionEnabled ? "active" : "neutral"}>
              {management?.skillEvolutionEnabled ? "Enabled" : "Off"}
            </Badge>
          </div>
          <p className="extension-section__description">
            Generate agent/model-specific candidate variants, evaluate them against the active
            baseline, and keep promotion gated with immutable lineage and rollback.
          </p>
          <div className="extension-evolution-controls">
            <label>
              <input
                type="checkbox"
                checked={management?.skillEvolutionEnabled ?? false}
                onChange={(event) =>
                  void runManagement("evolution", () =>
                    onSaveEvolutionPolicy({
                      enabled: event.target.checked,
                      promotionGate: management?.skillPromotionGate ?? "human",
                    }),
                  )
                }
              />
              <span>Allow candidate generation</span>
            </label>
            <label>
              <span>Promotion gate</span>
              <select
                value={management?.skillPromotionGate ?? "human"}
                onChange={(event) =>
                  void runManagement("evolution", () =>
                    onSaveEvolutionPolicy({
                      enabled: management?.skillEvolutionEnabled ?? false,
                      promotionGate: event.target.value as "human" | "policy",
                    }),
                  )
                }
              >
                <option value="human">Human approval</option>
                <option value="policy">Evaluation policy</option>
              </select>
            </label>
          </div>
        </section>

        <section className="extension-section" aria-label="Plugin components">
          <div className="extension-section__header">
            <h3>Plugin components</h3>
            <span>{pluginComponents.length}</span>
          </div>
          {pluginComponents.length === 0 ? (
            <div className="extension-empty">No plugin components</div>
          ) : (
            <ul className="extension-list">
              {pluginComponents.map((component) => (
                <li key={`${component.kind}:${component.id}`} className="extension-item">
                  <div className="extension-item__main">
                    <strong>{component.title}</strong>
                    <span>{component.kind}</span>
                  </div>
                  <div className="extension-item__meta">
                    <span>{component.id}</span>
                    {component.detail && <span>{component.detail}</span>}
                  </div>
                  <div className="extension-item__chips">
                    {component.chips.map((chip) => (
                      <span key={`${component.kind}:${component.id}:${chip}`}>{chip}</span>
                    ))}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="extension-section" aria-label="GUI contributions">
          <div className="extension-section__header">
            <h3>GUI contributions</h3>
            <span>{uiContributions.length}</span>
          </div>
          {uiContributions.length === 0 ? (
            <div className="extension-empty">No GUI contributions</div>
          ) : (
            <ul className="extension-list">
              {uiContributions.map((contribution) => (
                <li key={contribution.id} className="extension-item">
                  <div className="extension-item__main">
                    <strong>{contribution.name}</strong>
                    <span>{contribution.kind}</span>
                  </div>
                  <div className="extension-item__meta">
                    <span>{contribution.id}</span>
                    <span>{contribution.placement}</span>
                    {contribution.route && <span>{contribution.route}</span>}
                    {contribution.componentRef && <span>{contribution.componentRef}</span>}
                    {contribution.readOnly && <span>read-only</span>}
                  </div>
                  <div className="extension-item__chips">
                    {extensionUiContributionChips(contribution).map((chip) => (
                      <span key={`${contribution.id}:${chip}`}>{chip}</span>
                    ))}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="extension-section" aria-label="Harnesses">
          <div className="extension-section__header">
            <h3>Harnesses</h3>
            <span>{harnesses.length}</span>
          </div>
          <ul className="extension-list">
            {harnesses.map((harness) => (
              <li key={harness.id} className="extension-item">
                <div className="extension-item__main">
                  <strong>{harness.label}</strong>
                  <span>{harness.id}</span>
                </div>
                <div className="extension-item__meta">
                  <span>{formatSoftwareSummary(harness.software)}</span>
                  {harness.readOnly && <span>read-only</span>}
                </div>
                <div className="extension-item__chips">
                  <span>{harness.modelControl}</span>
                  {(harness.supportedModelApis ?? []).map((api) => (
                    <span key={`${harness.id}:${api}`}>{modelApiLabel(api)}</span>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        </section>

        <section className="extension-section" aria-label="Agent profiles">
          <div className="extension-section__header">
            <h3>Agent profiles</h3>
            <span>{agents.length}</span>
          </div>
          {agents.length === 0 ? (
            <div className="extension-empty">No agent profiles</div>
          ) : (
            <ul className="extension-list">
              {agents.map((agent) => {
                const plan = planByAgentId.get(agent.id);
                const canUseAgent = !plan || plan.status === "ready";
                return (
                  <li key={agent.id} className="extension-item">
                    <div className="extension-item__main">
                      <strong>{agent.name}</strong>
                      <span>{agent.id}</span>
                      {plan && <Badge tone={agentPlanTone(plan)}>{plan.status}</Badge>}
                      <Button
                        size="sm"
                        variant={selectedAgentId === agent.id ? "secondary" : "default"}
                        onClick={() => onSelectAgent(agent.id)}
                        aria-label={`Use agent profile ${agent.name}`}
                        disabled={!canUseAgent}
                        title={
                          canUseAgent ? `Use agent profile ${agent.name}` : planBlockedTitle(plan)
                        }
                      >
                        {selectedAgentId === agent.id ? "Selected" : "Use"}
                      </Button>
                    </div>
                    <div className="extension-item__meta">
                      <span>{plan?.canonicalSelector ?? agent.selector ?? agent.id}</span>
                      <span>{plan?.harnessLabel ?? agent.harnessId ?? "no harness"}</span>
                      <span>{plan?.modelId ?? agent.modelId ?? "no model"}</span>
                      {(plan?.modelSupplyId ?? agent.modelSupplyId) && (
                        <span>supply {plan?.modelSupplyId ?? agent.modelSupplyId}</span>
                      )}
                      {agent.permissionMode && <span>permission {agent.permissionMode}</span>}
                      {agent.memory && <span>memory {agent.memory}</span>}
                    </div>
                    <div className="extension-item__chips">
                      {agentPlanChips(plan, agent).map((chip) => (
                        <span key={`${agent.id}:${chip}`}>{chip}</span>
                      ))}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section className="extension-section" aria-label="Models and supplies">
          <div className="extension-section__header">
            <h3>Models and supplies</h3>
            <span>{models.length + modelSupplies.length}</span>
          </div>
          <ul className="extension-list extension-list--compact">
            {models.map((model) => (
              <li key={`model:${model.id}`} className="extension-item">
                <div className="extension-item__main">
                  <strong>{model.label ?? model.id}</strong>
                  <span>{model.id}</span>
                </div>
                <div className="extension-item__meta">
                  <span>{model.runtimeModel}</span>
                  {model.apiProtocols.map((api) => (
                    <span key={`${model.id}:${api}`}>{modelApiLabel(api)}</span>
                  ))}
                </div>
              </li>
            ))}
            {modelSupplies.map((supply) => (
              <li key={`supply:${supply.id}`} className="extension-item">
                <div className="extension-item__main">
                  <strong>{supply.id}</strong>
                  <span>
                    {supply.modelId} → {supply.providerProfileId}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </section>

        <section className="extension-section" aria-label="Providers">
          <div className="extension-section__header">
            <h3>Providers</h3>
            <span>{providers.length}</span>
          </div>
          {providers.length === 0 ? (
            <div className="extension-empty">No provider profiles</div>
          ) : (
            <ul className="extension-list">
              {providers.map((provider) => (
                <li key={provider.id} className="extension-item">
                  <div className="extension-item__main">
                    <strong>{provider.label}</strong>
                    <span>{provider.id}</span>
                  </div>
                  <div className="extension-item__meta">
                    <span>{provider.kind}</span>
                    {provider.runtimeReady === false && <span>not ready</span>}
                    {provider.runtimeNote && <span>{provider.runtimeNote}</span>}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="extension-section" aria-label="Skills and MCP">
          <div className="extension-section__header">
            <h3>Skills and MCP</h3>
            <span>{skills.length + mcpServers.length + appConnectors.length}</span>
          </div>
          <ul className="extension-list extension-list--compact">
            {skills.map((skill) => (
              <li key={`skill:${skill.id}`} className="extension-item">
                <div className="extension-item__main">
                  <strong>{skill.name ?? skill.id}</strong>
                  <span>{skill.path ?? skill.id}</span>
                </div>
                <div className="extension-item__chips">
                  {extensionSkillChips(skill).map((chip) => (
                    <span key={`${skill.id}:${chip}`}>{chip}</span>
                  ))}
                </div>
              </li>
            ))}
            {mcpServers.map((server) => (
              <li key={`mcp:${server.id}`} className="extension-item">
                <div className="extension-item__main">
                  <strong>{server.name ?? server.id}</strong>
                  <span>{server.scope ?? "MCP server"}</span>
                </div>
              </li>
            ))}
            {appConnectors.map((connector) => (
              <li key={`connector:${connector.id}`} className="extension-item">
                <div className="extension-item__main">
                  <strong>{connector.name}</strong>
                  <span>{connector.kind}</span>
                </div>
              </li>
            ))}
          </ul>
        </section>

        {(error || warnings.length > 0) && (
          <section
            className="extension-section extension-section--alerts"
            aria-label="Extension alerts"
          >
            <div className="extension-section__header">
              <h3>Alerts</h3>
              <span>{warnings.length + (error ? 1 : 0)}</span>
            </div>
            <ul className="extension-list extension-list--compact">
              {error ? (
                <li className="extension-alert">
                  <XCircle aria-hidden="true" />
                  <span>{errorMessage(error)}</span>
                </li>
              ) : null}
              {warnings.map((warning) => (
                <li key={`${warning.source}:${warning.message}`} className="extension-alert">
                  <XCircle aria-hidden="true" />
                  <span>
                    {warning.source}: {warning.message}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </section>
  );
}

function EmptyRun({
  projectLabel,
  rightPanelOpen,
  onSelectPrompt,
}: {
  projectLabel: string;
  rightPanelOpen: boolean;
  onSelectPrompt: (prompt: string) => void;
}) {
  const suggestions = rightPanelOpen ? EMPTY_RUN_SUGGESTIONS.slice(0, 2) : EMPTY_RUN_SUGGESTIONS;

  return (
    <div className="empty-run">
      <div className="empty-run__mark">
        <Workflow aria-hidden="true" />
      </div>
      <div className="empty-run__copy">
        <h2>What should we build in {projectLabel}?</h2>
        <p>Choose a starting point or describe anything below.</p>
      </div>
      <div
        className={cx(
          "empty-run__suggestions",
          rightPanelOpen && "empty-run__suggestions--right-panel",
        )}
        aria-label="Suggested tasks"
      >
        {suggestions.map((suggestion) => {
          const Icon = suggestion.icon;
          return (
            <button
              key={suggestion.id}
              type="button"
              className={cx("empty-run__suggestion", `is-${suggestion.tone}`)}
              onClick={() => onSelectPrompt(suggestion.prompt)}
            >
              <Icon aria-hidden="true" />
              <span>{suggestion.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function PinnedSummary({
  title,
  subtitle,
  status,
  messageCount,
  workflowLabel,
  onClose,
}: {
  title: string;
  subtitle: string;
  status: string;
  messageCount: number;
  workflowLabel: string;
  onClose: () => void;
}) {
  return (
    <section className="pinned-summary" aria-label="Pinned summary">
      <Pin aria-hidden="true" />
      <div className="pinned-summary__copy">
        <strong>{title}</strong>
        <span>{subtitle}</span>
      </div>
      <div className="pinned-summary__meta">
        <Badge tone={status === "Running" ? "active" : "neutral"}>{status}</Badge>
        <Badge tone="neutral">{messageCount} events</Badge>
        <Badge tone="neutral">{workflowLabel}</Badge>
      </div>
      <Button variant="ghost" size="icon" onClick={onClose} aria-label="Unpin summary">
        <XCircle aria-hidden="true" />
      </Button>
    </section>
  );
}

function RuntimeRightPanel({
  title,
  harness,
  model,
  effort,
  status,
  messageCount,
  onClose,
  onDelete,
}: {
  title: string;
  harness: string;
  model: string;
  effort: string;
  status: string;
  messageCount: number;
  onClose: () => void;
  onDelete?: () => void;
}) {
  return (
    <aside className="runtime-right-panel" aria-label="Right panel">
      <div className="runtime-panel__header">
        <div>
          <span>Summary</span>
          <h2>{title}</h2>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close right panel">
          <PanelRight aria-hidden="true" />
        </Button>
      </div>
      <dl className="runtime-right-panel__details">
        <div>
          <dt>Status</dt>
          <dd>{status}</dd>
        </div>
        <div>
          <dt>Harness</dt>
          <dd>{harness}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd>{model}</dd>
        </div>
        <div>
          <dt>Effort</dt>
          <dd>{effort}</dd>
        </div>
        <div>
          <dt>Events</dt>
          <dd>{messageCount}</dd>
        </div>
      </dl>
      {onDelete && (
        <Button variant="destructive" size="sm" onClick={onDelete}>
          <Trash2 data-icon="inline-start" aria-hidden="true" />
          Delete session
        </Button>
      )}
    </aside>
  );
}

function RuntimeBottomPanel({
  active,
  cwd,
  onClose,
}: {
  active: boolean;
  cwd: string;
  onClose: () => void;
}) {
  const terminalElementRef = useRef<HTMLDivElement>(null);
  const terminalRef = useRef<XtermTerminal | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const terminalIdRef = useRef<string | null>(null);
  const startingRef = useRef(false);
  const readyRef = useRef(false);
  const activeRef = useRef(active);
  const disposedRef = useRef(false);
  const pendingInputRef = useRef("");
  const fitAndResizeRef = useRef<() => void>(() => undefined);
  const [status, setStatus] = useState<"idle" | "starting" | "running" | "exited" | "error">(
    "idle",
  );

  const startTerminal = useCallback(async () => {
    const terminal = terminalRef.current;
    const fitAddon = fitAddonRef.current;
    if (!terminal || !fitAddon || startingRef.current || terminalIdRef.current) return;

    startingRef.current = true;
    readyRef.current = false;
    setStatus("starting");
    fitAddon.fit();
    const id = terminalRequestId();
    terminalIdRef.current = id;

    try {
      await api.createTerminal({ id, cwd, cols: terminal.cols, rows: terminal.rows });
      if (disposedRef.current || terminalIdRef.current !== id) {
        await api.killTerminal(id);
        return;
      }
      readyRef.current = true;
      setStatus("running");
      if (pendingInputRef.current) {
        const pendingInput = pendingInputRef.current;
        pendingInputRef.current = "";
        await api.writeTerminal(id, pendingInput);
      }
      terminal.focus();
    } catch (error) {
      if (terminalIdRef.current === id) terminalIdRef.current = null;
      pendingInputRef.current = "";
      setStatus("error");
      terminal.writeln(`\r\nUnable to start terminal: ${plainTerminalError(errorMessage(error))}`);
    } finally {
      startingRef.current = false;
    }
  }, [cwd]);

  const newTerminal = useCallback(async () => {
    const currentId = terminalIdRef.current;
    terminalIdRef.current = null;
    readyRef.current = false;
    pendingInputRef.current = "";
    if (currentId) await api.killTerminal(currentId);
    terminalRef.current?.reset();
    setStatus("idle");
    await startTerminal();
  }, [startTerminal]);

  useEffect(() => {
    disposedRef.current = false;
    const element = terminalElementRef.current;
    if (!element) return;

    const terminal = new XtermTerminal({
      allowTransparency: false,
      cursorBlink: true,
      cursorStyle: "bar",
      fontFamily:
        '"SFMono-Regular", "SF Mono", "Cascadia Code", Consolas, "Liberation Mono", Menlo, monospace',
      fontSize: 12.5,
      lineHeight: 1.25,
      minimumContrastRatio: 4.5,
      screenReaderMode: true,
      scrollback: 5_000,
      theme: internalTerminalTheme(),
    });
    const fitAddon = new FitAddon();
    terminal.loadAddon(fitAddon);
    terminal.open(element);
    terminalRef.current = terminal;
    fitAddonRef.current = fitAddon;

    let lastDimensions = "";
    const fitAndResize = () => {
      if (!activeRef.current || !element.offsetWidth || !element.offsetHeight) return;
      fitAddon.fit();
      const dimensions = `${terminal.cols}:${terminal.rows}`;
      if (dimensions === lastDimensions) return;
      lastDimensions = dimensions;
      const id = terminalIdRef.current;
      if (id) void api.resizeTerminal(id, terminal.cols, terminal.rows);
    };
    fitAndResizeRef.current = fitAndResize;

    const terminalInput = terminal.onData((data) => {
      const id = terminalIdRef.current;
      if (!id || !readyRef.current) {
        pendingInputRef.current += data;
        return;
      }
      void api.writeTerminal(id, data);
    });
    const removeDataListener = api.onTerminalData((event) => {
      if (event.id === terminalIdRef.current) terminal.write(event.data);
    });
    const removeExitListener = api.onTerminalExit((event) => {
      if (event.id !== terminalIdRef.current) return;
      terminalIdRef.current = null;
      readyRef.current = false;
      setStatus("exited");
      terminal.writeln(`\r\n[Process exited with code ${event.exitCode}]`);
    });
    const media =
      typeof window.matchMedia === "function"
        ? window.matchMedia("(prefers-color-scheme: light)")
        : null;
    const updateTheme = () => {
      terminal.options.theme = internalTerminalTheme();
    };
    media?.addEventListener("change", updateTheme);
    const resizeObserver =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(fitAndResize);
    resizeObserver?.observe(element);

    return () => {
      disposedRef.current = true;
      const id = terminalIdRef.current;
      terminalIdRef.current = null;
      readyRef.current = false;
      if (id) void api.killTerminal(id);
      resizeObserver?.disconnect();
      media?.removeEventListener("change", updateTheme);
      terminalInput.dispose();
      removeDataListener();
      removeExitListener();
      terminal.dispose();
      terminalRef.current = null;
      fitAddonRef.current = null;
      fitAndResizeRef.current = () => undefined;
    };
  }, []);

  useEffect(() => {
    activeRef.current = active;
    if (!active) return;
    const frame = window.requestAnimationFrame(() => {
      fitAndResizeRef.current();
      void startTerminal();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [active, startTerminal]);

  return (
    <section className="runtime-bottom-panel" aria-label="Bottom panel">
      <div className="terminal-panel__tabbar">
        <div className="terminal-panel__tabs" role="tablist" aria-label="Terminals">
          <button type="button" className="terminal-panel__tab" role="tab" aria-selected="true">
            <TerminalIcon aria-hidden="true" />
            <span>{projectName(cwd)}</span>
            <span className={cx("terminal-panel__status", `is-${status}`)} aria-hidden="true" />
          </button>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => void newTerminal()}
            disabled={status === "starting"}
            title="New terminal"
            aria-label="New terminal"
          >
            <Plus aria-hidden="true" />
          </Button>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          title="Close bottom panel"
          aria-label="Close bottom panel"
        >
          <X aria-hidden="true" />
        </Button>
      </div>
      <div
        ref={terminalElementRef}
        className="terminal-panel__viewport"
        aria-label="Internal terminal"
      />
      <span className="sr-only" aria-live="polite">
        Terminal {status}
      </span>
    </section>
  );
}

function terminalRequestId(): string {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `terminal-${Date.now()}-${Math.random().toString(36).slice(2)}`
  );
}

function plainTerminalError(message: string): string {
  return [...message]
    .map((character) => {
      const code = character.charCodeAt(0);
      return code < 32 || (code >= 127 && code <= 159) ? " " : character;
    })
    .join("")
    .trim();
}

function internalTerminalTheme() {
  if (
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: light)").matches
  ) {
    return {
      background: "#ffffff",
      foreground: "#20242c",
      cursor: "#087c9b",
      cursorAccent: "#ffffff",
      selectionBackground: "#cfeef5",
      black: "#20242c",
      brightBlack: "#737e8e",
      red: "#c33535",
      green: "#087c55",
      yellow: "#9a6700",
      blue: "#0969da",
      magenta: "#8250df",
      cyan: "#087c9b",
      white: "#e7eaf0",
      brightWhite: "#17191f",
    };
  }
  return {
    background: "#0b0d12",
    foreground: "#e8eaf0",
    cursor: "#95e9ff",
    cursorAccent: "#0b0d12",
    selectionBackground: "#294451",
    black: "#151821",
    brightBlack: "#77808f",
    red: "#f87171",
    green: "#34d399",
    yellow: "#fbbf24",
    blue: "#60a5fa",
    magenta: "#c084fc",
    cyan: "#67e8f9",
    white: "#d5d9e2",
    brightWhite: "#ffffff",
  };
}

function DoctorPanel({
  mode,
  report,
  loading,
  harnessOptions,
  harnessVersions,
  error,
  fixPending,
  fixRunning,
  fixResult,
  onRefresh,
  onRequestFix,
  onCancelFix,
  onConfirmFix,
  installingHarnessId,
  onInstallHarness,
  onRefreshHarnessVersion,
  onClose,
}: {
  mode: DoctorPanelMode;
  report: DoctorReport | null;
  loading: boolean;
  harnessOptions: HarnessOption[];
  harnessVersions: Record<string, DoctorHarnessVersionState>;
  error: string | null;
  fixPending: boolean;
  fixRunning: boolean;
  fixResult: DoctorFixResult | null;
  onRefresh: () => void;
  onRequestFix: () => void;
  onCancelFix: () => void;
  onConfirmFix: () => void;
  installingHarnessId: string | null;
  onInstallHarness: (harnessId: string) => void;
  onRefreshHarnessVersion: (harnessId: string) => void;
  onClose: () => void;
}) {
  const issues = (report?.issues ?? []).filter((issue) => issue.scope === "doctor");
  const visibleRepairActionIds = new Set(issues.flatMap((issue) => issue.repairActionId ?? []));
  const repairActions = (report?.repairActions ?? []).filter((action) =>
    visibleRepairActionIds.has(action.id),
  );
  const reportedHarnesses = new Map(
    (report?.environment.harnesses ?? []).map((harness) => [harness.harnessId, harness]),
  );
  const harnesses = harnessOptions.map(
    (harness) =>
      reportedHarnesses.get(harness.id) ?? {
        harnessId: harness.id,
        harnessLabel: harness.label,
        version: undefined,
      },
  );
  const requirements = report?.environment.requirements ?? [];
  const containerRuntimes = report?.environment.containerRuntimes ?? [];
  const setupLogs = fixResult?.setupResults.flatMap((result) => result.log) ?? [];
  const title = mode === "setup" ? "Setup" : "Doctor";
  const panelHealthy = Boolean(report && issues.length === 0);
  const summaryTitle = loading
    ? "Checking environment"
    : panelHealthy
      ? "Environment ready"
      : report
        ? issues.length + (issues.length === 1 ? " issue found" : " issues found")
        : "Status unavailable";
  const summaryCopy = panelHealthy
    ? "Harnesses are optional; install one only when you plan to use it."
    : mode === "setup"
      ? "Review the missing pieces, then confirm before SwarmX changes anything."
      : "Review diagnostics and the repair plan before applying fixes.";

  return (
    <aside
      className="runtime-right-panel doctor-panel"
      aria-label={mode === "setup" ? "Setup panel" : "Doctor panel"}
    >
      <div className="runtime-panel__header">
        <div>
          <span>Environment</span>
          <h2>
            {title}
            {report?.harnessId ? ` · ${report.harnessId}` : ""}
          </h2>
        </div>
        <div className="doctor-panel__header-actions">
          <Button
            variant="ghost"
            size="icon"
            onClick={onRefresh}
            disabled={loading || fixRunning}
            title="Refresh diagnostics"
            aria-label="Refresh diagnostics"
          >
            <RefreshCw aria-hidden="true" />
          </Button>
          <Button variant="ghost" size="icon" onClick={onClose} aria-label={`Close ${title}`}>
            <PanelRight aria-hidden="true" />
          </Button>
        </div>
      </div>

      <section className={cx("doctor-summary", panelHealthy && "is-healthy")} aria-live="polite">
        <span className="doctor-summary__icon">
          {loading ? (
            <Loader2 aria-hidden="true" />
          ) : panelHealthy ? (
            <CircleCheck aria-hidden="true" />
          ) : (
            <Wrench aria-hidden="true" />
          )}
        </span>
        <div>
          <h3>{summaryTitle}</h3>
          <p>{summaryCopy}</p>
        </div>
      </section>

      {error && (
        <div className="doctor-notice doctor-notice--error" role="alert">
          <XCircle aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}

      {fixResult?.executed && (
        <output
          className={cx(
            "doctor-notice",
            fixResult.after.healthy ? "doctor-notice--success" : "doctor-notice--error",
          )}
        >
          {fixResult.after.healthy ? (
            <CircleCheck aria-hidden="true" />
          ) : (
            <XCircle aria-hidden="true" />
          )}
          <span>
            {fixResult.after.healthy
              ? "Repairs completed. The environment is ready."
              : "Repairs completed, but some issues still need attention."}
          </span>
        </output>
      )}

      {!loading && report && repairActions.length > 0 && (
        <section className="doctor-section" aria-labelledby="doctor-repair-title">
          <div className="doctor-section__heading">
            <h3 id="doctor-repair-title">Repair plan</h3>
            <span>{repairActions.length}</span>
          </div>
          {fixPending ? (
            <div className="doctor-confirmation">
              <strong>
                Confirm {repairActions.length} {repairActions.length === 1 ? "repair" : "repairs"}
              </strong>
              <p>No changes are made until you confirm this plan.</p>
              <ul className="doctor-list">
                {repairActions.map((action) => (
                  <li key={action.id} className="doctor-action">
                    <span>{action.label}</span>
                    <Badge tone={action.risk === "admin" ? "danger" : "neutral"}>
                      {action.risk}
                    </Badge>
                  </li>
                ))}
              </ul>
              <div className="doctor-confirmation__actions">
                <Button variant="ghost" size="sm" onClick={onCancelFix} disabled={fixRunning}>
                  Cancel
                </Button>
                <Button size="sm" onClick={onConfirmFix} disabled={fixRunning}>
                  {fixRunning ? (
                    <Loader2 data-icon="inline-start" aria-hidden="true" />
                  ) : (
                    <Wrench data-icon="inline-start" aria-hidden="true" />
                  )}
                  Confirm {repairActions.length}
                </Button>
              </div>
            </div>
          ) : (
            <Button size="sm" onClick={onRequestFix}>
              <Wrench data-icon="inline-start" aria-hidden="true" />
              {mode === "setup" ? "Set up missing" : "Fix issues"}
            </Button>
          )}
        </section>
      )}

      {!loading && report && issues.length > 0 && (
        <section className="doctor-section" aria-labelledby="doctor-issues-title">
          <div className="doctor-section__heading">
            <h3 id="doctor-issues-title">Diagnostics</h3>
            <span>{issues.length}</span>
          </div>
          <ul className="doctor-list">
            {issues.map((issue) => (
              <li key={issue.id} className="doctor-issue">
                <XCircle aria-hidden="true" />
                <div>
                  <strong>{issue.targetId ?? issue.scope}</strong>
                  <span>{issue.message}</span>
                </div>
                <Badge tone={issue.severity === "error" ? "danger" : "neutral"}>
                  {issue.severity}
                </Badge>
              </li>
            ))}
          </ul>
          {issues.length > 0 && repairActions.length === 0 && (
            <p className="doctor-section__hint">These issues require manual review.</p>
          )}
        </section>
      )}

      <section className="doctor-section" aria-labelledby="doctor-harnesses-title">
        <div className="doctor-section__heading">
          <h3 id="doctor-harnesses-title">Harnesses</h3>
          <span>
            {
              harnesses.filter((harness) => {
                const state = harnessVersions[harness.harnessId];
                return state?.status === "loaded" && Boolean(state.version ?? harness.version);
              }).length
            }
            /{harnesses.length}
          </span>
        </div>
        <ul className="doctor-list">
          {harnesses.map((harness) => {
            const versionState = harnessVersions[harness.harnessId];
            const version = versionState?.version ?? harness.version;
            const versionLoading = !versionState || versionState.status === "loading";
            return (
              <li key={harness.harnessId} className="doctor-harness">
                <span className="doctor-harness__icon">
                  <HarnessBrandIcon
                    harness={harnessOption(harness.harnessId, harness.harnessLabel)}
                  />
                </span>
                <div>
                  <strong>{harness.harnessLabel}</strong>
                </div>
                {versionLoading ? (
                  <output
                    className="badge doctor-harness__version is-loading"
                    aria-label={`Checking ${harness.harnessLabel} version`}
                  >
                    <Loader2 data-icon aria-hidden="true" />
                  </output>
                ) : version ? (
                  <button
                    type="button"
                    className="badge badge--active doctor-harness__version"
                    aria-label={`Check ${harness.harnessLabel} version again`}
                    title="Check version again"
                    onClick={() => onRefreshHarnessVersion(harness.harnessId)}
                  >
                    {version}
                  </button>
                ) : (
                  <Button
                    variant="secondary"
                    size="sm"
                    aria-label={`Install ${harness.harnessLabel}`}
                    disabled={Boolean(installingHarnessId)}
                    onClick={() => onInstallHarness(harness.harnessId)}
                  >
                    {installingHarnessId === harness.harnessId && (
                      <Loader2 data-icon="inline-start" aria-hidden="true" />
                    )}
                    Install
                  </Button>
                )}
              </li>
            );
          })}
        </ul>
      </section>

      {report && (
        <details className="doctor-advanced">
          <summary>
            <span>
              <strong>Advanced details</strong>
              <small>Runtime tools, PATH, and repair logs</small>
            </span>
            <ChevronRight aria-hidden="true" />
          </summary>
          <div className="doctor-advanced__body">
            {requirements.length > 0 && (
              <section>
                <h4>Runtime tools</h4>
                <ul className="doctor-list">
                  {requirements.map((requirement) => (
                    <li key={requirement.id} className="doctor-diagnostic">
                      <div>
                        <strong>{requirement.label}</strong>
                        <span>
                          {[
                            requirement.command,
                            requirement.version,
                            requirement.path,
                            requirement.note,
                          ]
                            .filter(Boolean)
                            .join(" · ")}
                        </span>
                      </div>
                      <Badge tone={requirement.status === "ready" ? "active" : "danger"}>
                        {requirementStatusLabel(requirement.status)}
                      </Badge>
                    </li>
                  ))}
                </ul>
              </section>
            )}
            {containerRuntimes.length > 0 && (
              <section>
                <h4>Container runtime</h4>
                <ul className="doctor-list">
                  {containerRuntimes.map((runtime) => (
                    <li key={runtime.id} className="doctor-diagnostic">
                      <div>
                        <strong>{runtime.label}</strong>
                        <span>
                          {[runtime.command, runtime.version, runtime.path, runtime.note]
                            .filter(Boolean)
                            .join(" · ")}
                        </span>
                      </div>
                      <Badge tone={runtime.status === "ready" ? "active" : "danger"}>
                        {containerRuntimeStatusLabel(runtime.status)}
                      </Badge>
                    </li>
                  ))}
                </ul>
              </section>
            )}
            <section>
              <h4>Environment PATH</h4>
              <pre className="doctor-code">{report.environment.path}</pre>
            </section>
            {setupLogs.length > 0 && (
              <section>
                <h4>Repair log</h4>
                <pre className="doctor-code">{setupLogs.join("\n\n")}</pre>
              </section>
            )}
          </div>
        </details>
      )}
    </aside>
  );
}

function GuiContributionWorkspace({
  contribution,
  inventory,
  component: Component,
  onSelectAgent,
}: {
  contribution: ExtensionUiContributionSummary;
  inventory?: ExtensionCapabilityInventory;
  component: GuiContributionComponent;
  onSelectAgent: (agentId: string) => void;
}) {
  return (
    <section
      className="gui-contribution-workspace"
      aria-label={`${contribution.name} contribution`}
    >
      <div className="gui-contribution-topbar">
        <div className="extension-title">
          <Package aria-hidden="true" />
          <div>
            <h2>{contribution.name}</h2>
            <span>{contribution.description ?? contribution.componentRef}</span>
          </div>
        </div>
        <div className="extension-stats" aria-label="Contribution metadata">
          <Badge tone="neutral">{contribution.kind}</Badge>
          <Badge tone="neutral">{contribution.placement}</Badge>
          {contribution.sourcePluginId && (
            <Badge tone="neutral">{contribution.sourcePluginId}</Badge>
          )}
          {contribution.readOnly && <Badge tone="neutral">read-only</Badge>}
        </div>
      </div>
      <div className="gui-contribution-body">
        <Component
          contribution={contribution}
          inventory={inventory}
          onSelectAgent={onSelectAgent}
        />
      </div>
    </section>
  );
}

function WorkflowWorkspace({
  workflowJson,
  onWorkflowJsonChange,
  workflowEnabled,
  onWorkflowEnabledChange,
  workflowImportStatus,
  workflowState,
  input,
  onInputChange,
  onExecute,
  onImportN8nFile,
  loading,
  messages,
  activeWorkflowConfig,
}: {
  workflowJson: string;
  onWorkflowJsonChange: (value: string) => void;
  workflowEnabled: boolean;
  onWorkflowEnabledChange: (value: boolean) => void;
  workflowImportStatus: WorkflowImportStatus | null;
  workflowState: WorkflowParseResult;
  input: string;
  onInputChange: (value: string) => void;
  onExecute: () => void;
  onImportN8nFile: (file: File) => void;
  loading: boolean;
  messages: MessageChunk[];
  activeWorkflowConfig: SwarmConfig | null;
}) {
  const importInputRef = useRef<HTMLInputElement>(null);
  const workflowName = workflowState.config?.name ?? "Workflow";
  const selectedNode =
    workflowState.nodes.find((node) => node.id === "writer_agent") ?? workflowState.nodes.at(-1);
  const importNoticeRole = workflowImportStatus?.kind === "error" ? "alert" : "status";

  return (
    <section className="workflow-workspace" aria-label="Workflow editor">
      <div className="workflow-topbar">
        <div className="workflow-breadcrumb">
          <Workflow aria-hidden="true" />
          <span>Personal</span>
          <span>/</span>
          <strong>{workflowName}</strong>
        </div>
        <div className="workflow-view-tabs" role="tablist" aria-label="Workflow views">
          <button
            type="button"
            role="tab"
            aria-selected="true"
            className="workflow-view-tab is-active"
          >
            Editor
          </button>
          <button type="button" role="tab" aria-selected="false" className="workflow-view-tab">
            Executions
          </button>
          <button type="button" role="tab" aria-selected="false" className="workflow-view-tab">
            JSON
          </button>
        </div>
        <div className="workflow-topbar__actions">
          <input
            ref={importInputRef}
            type="file"
            accept=".json,application/json"
            hidden
            aria-label="n8n workflow JSON file"
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (file) onImportN8nFile(file);
            }}
          />
          <Button
            variant="secondary"
            size="sm"
            onClick={() => importInputRef.current?.click()}
            title="Import n8n workflow JSON"
          >
            <Upload data-icon="inline-start" aria-hidden="true" />
            Import n8n
          </Button>
          <label className="workflow-toggle">
            <input
              type="checkbox"
              checked={workflowEnabled}
              onChange={(event) => onWorkflowEnabledChange(event.target.checked)}
            />
            <span>Use workflow</span>
          </label>
          <Badge
            tone={activeWorkflowConfig ? "active" : workflowState.error ? "danger" : "neutral"}
          >
            {activeWorkflowConfig ? "Saved" : workflowState.error ? "Invalid" : "Draft"}
          </Badge>
        </div>
      </div>

      <div className="workflow-editor-shell">
        <nav className="workflow-rail" aria-label="Workflow navigation">
          <button type="button" className="workflow-rail__brand" aria-label="Workflows">
            <Workflow aria-hidden="true" />
          </button>
          <button type="button" className="workflow-rail__create" aria-label="Add node">
            <MessageSquarePlus aria-hidden="true" />
          </button>
          <span>Overview</span>
          <span className="is-active">Workflows</span>
          <span>Agents</span>
          <span>Tools</span>
          <span>MCP</span>
          <span>Runs</span>
        </nav>

        <WorkflowCanvas
          workflowState={workflowState}
          onExecute={onExecute}
          loading={loading}
          input={input}
        />

        <aside className="workflow-inspector" aria-label="Workflow inspector">
          <div className="workflow-inspector__header">
            <div>
              <span>
                {selectedNode?.kind === "agent"
                  ? "harness = software + MCPs + skills + project files"
                  : (selectedNode?.displayKind ?? "workflow")}
              </span>
              <strong>{selectedNode?.title ?? workflowName}</strong>
              {selectedNode?.harnessLabel && selectedNode.model && (
                <em>
                  {selectedNode.harnessLabel} / {selectedNode.softwareLabel ?? "software"} /{" "}
                  {selectedNode.model}
                </em>
              )}
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onExecute}
              disabled={loading || !input.trim()}
              title="Run selected node"
              aria-label="Run selected node"
            >
              {loading ? (
                <Loader2 data-icon aria-hidden="true" />
              ) : (
                <Play data-icon aria-hidden="true" />
              )}
            </Button>
          </div>

          <div className="workflow-inspector__tabs" role="tablist" aria-label="Inspector tabs">
            <button type="button" role="tab" aria-selected="true">
              Parameters
            </button>
            <button type="button" role="tab" aria-selected="false">
              Settings
            </button>
            <button type="button" role="tab" aria-selected="false">
              Notes
            </button>
          </div>

          <label className="workflow-run-input">
            <span>Run input</span>
            <textarea
              value={input}
              onChange={(event) => onInputChange(event.target.value)}
              placeholder="Message SwarmX"
              rows={3}
              disabled={loading}
            />
          </label>

          <label className="workflow-editor">
            <span>Workflow JSON</span>
            <textarea
              aria-label="Workflow JSON"
              value={workflowJson}
              onChange={(event) => onWorkflowJsonChange(event.target.value)}
              spellCheck={false}
            />
          </label>

          {workflowState.error && (
            <div className="workflow-panel__error" role="alert">
              <XCircle aria-hidden="true" />
              <span>{workflowState.error}</span>
            </div>
          )}

          {workflowImportStatus && (
            <div
              className={cx(
                "workflow-panel__notice",
                `workflow-panel__notice--${workflowImportStatus.kind}`,
              )}
              role={importNoticeRole}
            >
              {workflowImportStatus.kind === "error" ? (
                <XCircle aria-hidden="true" />
              ) : (
                <CircleCheck aria-hidden="true" />
              )}
              <div>
                <span>{workflowImportStatus.message}</span>
                {workflowImportStatus.warnings.length > 0 && (
                  <ul>
                    {workflowImportStatus.warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}

          <div className="workflow-output">
            <div className="workflow-output__header">
              <span>Execution Log</span>
              <Badge tone={messages.length > 0 ? "active" : "neutral"}>
                {messages.length} events
              </Badge>
            </div>
            <div className="workflow-output__list">
              {messages.length === 0 ? (
                <div className="workflow-output__empty">No run output yet</div>
              ) : (
                messages.slice(-5).map((message) => (
                  <div key={messageKey(message)} className="workflow-output__event">
                    <span>{message.agent ?? message.role}</span>
                    <p>{message.content}</p>
                  </div>
                ))
              )}
            </div>
          </div>
        </aside>
      </div>
    </section>
  );
}

function WorkflowCanvas({
  workflowState,
  onExecute,
  loading,
  input,
}: {
  workflowState: WorkflowParseResult;
  onExecute: () => void;
  loading: boolean;
  input: string;
}) {
  if (!workflowState.config) {
    return (
      <div className="workflow-canvas workflow-canvas--empty" aria-label="Workflow canvas">
        <div className="workflow-canvas__empty-message">No workflow</div>
        <Button
          className="workflow-execute"
          onClick={onExecute}
          disabled={loading || !input.trim()}
        >
          {loading ? (
            <Loader2 data-icon="inline-start" aria-hidden="true" />
          ) : (
            <Play data-icon="inline-start" aria-hidden="true" />
          )}
          Execute workflow
        </Button>
      </div>
    );
  }

  const nodes = workflowState.nodes.map((node, index) => ({
    ...node,
    layout: workflowNodeLayout(node.id, index),
  }));
  const nodeLayouts = new Map(nodes.map((node) => [node.id, node.layout]));

  return (
    <div className="workflow-canvas" aria-label="Workflow canvas">
      <svg className="workflow-connectors" viewBox="0 0 804 620" aria-label="Workflow connectors">
        <title>Workflow connectors</title>
        {workflowState.edges.map((edge, index) => {
          const source = nodeLayouts.get(edge.source);
          const target = nodeLayouts.get(edge.target);
          if (!source || !target) return null;
          return (
            <path
              key={`${edge.source}:${edge.target}:${index}`}
              aria-label={`Workflow connector ${edge.source} to ${edge.target}`}
              className="workflow-connector"
              d={connectorPath(source, target)}
            />
          );
        })}
      </svg>

      <ul className="workflow-canvas__nodes" aria-label="Workflow nodes">
        {nodes.map((node) => {
          const NodeIcon = nodeIcon(node.displayKind);
          const nodeHarnessId = node.harnessId ?? node.displayKind;
          const nodeModel = node.model ?? "";
          return (
            <li
              key={node.id}
              aria-label={`Workflow node ${node.id} ${nodeHarnessId}${nodeModel ? ` ${nodeModel}` : ""}${node.isRoot ? " root" : ""}`}
              className={cx(
                "workflow-node",
                `workflow-node--${node.displayKind}`,
                node.id === "writer_agent" && "is-selected",
              )}
              style={{ left: node.layout.x, top: node.layout.y }}
            >
              <span className="workflow-port workflow-port--in" aria-hidden="true" />
              <span className="workflow-port workflow-port--out" aria-hidden="true" />
              <div className="workflow-node__topline">
                <span className="workflow-node__icon">
                  <NodeIcon aria-hidden="true" />
                </span>
                <span className="workflow-node__kind">
                  {node.kind === "agent" ? "ACP Agent" : node.displayKind}
                </span>
                <span className="workflow-node__status">
                  <CircleCheck aria-hidden="true" />
                </span>
              </div>
              <div className="workflow-node__name">{node.title}</div>
              <div className="workflow-node__detail">{node.detail}</div>
              {node.kind === "agent" && (
                <div className="workflow-node__identity">
                  {node.softwareLabel && <span>Software {node.softwareLabel}</span>}
                  <span>Harness {node.harnessLabel ?? "SwarmX"}</span>
                  <span>{node.model ? `Model ${node.model}` : "Model negotiated by harness"}</span>
                  {node.mcpsLabel && <span>MCPs {node.mcpsLabel}</span>}
                  {node.skillsLabel && <span>Skills {node.skillsLabel}</span>}
                  {node.projectFilesLabel && <span>Project files {node.projectFilesLabel}</span>}
                </div>
              )}
            </li>
          );
        })}
      </ul>

      <ul className="workflow-edges" aria-label="Workflow edges">
        {workflowState.edges.map((edge, index) => (
          <li
            key={`${edge.source}:${edge.target}:${edge.condition ?? ""}:${index}`}
            aria-label={`Workflow edge ${edge.source} to ${edge.target}`}
            className="workflow-edge"
          >
            <span>{edge.source}</span>
            <span aria-hidden="true">-&gt;</span>
            <span>{edge.target}</span>
          </li>
        ))}
      </ul>

      <div className="workflow-canvas__add workflow-canvas__add--top" aria-hidden="true">
        +
      </div>
      <div className="workflow-canvas__add workflow-canvas__add--bottom" aria-hidden="true">
        +
      </div>

      <div className="workflow-canvas-controls" aria-label="Canvas controls">
        <button type="button" aria-label="Fit workflow">
          <Maximize2 aria-hidden="true" />
        </button>
        <button type="button" aria-label="Zoom out">
          <Minus aria-hidden="true" />
        </button>
        <span>100%</span>
        <button type="button" aria-label="Zoom in">
          <Plus aria-hidden="true" />
        </button>
        <button type="button">Tidy</button>
      </div>

      <Button className="workflow-execute" onClick={onExecute} disabled={loading || !input.trim()}>
        {loading ? (
          <Loader2 data-icon="inline-start" aria-hidden="true" />
        ) : (
          <Play data-icon="inline-start" aria-hidden="true" />
        )}
        Execute workflow
      </Button>
    </div>
  );
}

function RunEvent({ compact = false, msg }: { compact?: boolean; msg: MessageChunk }) {
  const renderEvent = normalizeMessageChunk(msg, normalizeOptionsFromMessage(msg));
  const {
    icon: Icon,
    label,
    tone,
    meta,
  } = compact ? compactWorkPresentation(msg) : messagePresentation(msg);
  const content = renderEventContent(msg, renderEvent);
  const showTraceCard = isTraceCardEvent(renderEvent);
  const plainNarrative = compact && (msg.kind === "thinking" || msg.kind === "message");
  const displayContent = msg.kind === "thinking" ? normalizeThoughtMarkdown(content) : content;

  return (
    <article
      className={cx("run-event", `run-event--${tone}`, compact && "run-event--compact")}
      data-render-event-id={renderEvent.eventId}
      data-render-kind={renderEvent.kind}
      data-render-status={renderEvent.status}
    >
      {plainNarrative ? (
        <div className="run-event__content">
          <MessageContent kind={msg.kind} content={displayContent} />
        </div>
      ) : (
        <>
          <div className="run-event__rail">
            <Icon aria-hidden="true" />
          </div>
          <div className="run-event__card">
            <div className="run-event__header">
              <span>{label}</span>
              <span>{renderEvent.status === "completed" ? meta : renderEvent.status}</span>
            </div>
            {msg.toolName && (
              <div className="run-event__tool">
                <Code2 aria-hidden="true" />
                <span>{msg.toolName}</span>
                <span className="run-event__tool-status">{renderEvent.status}</span>
              </div>
            )}
            {msg.swarmEvent && <div className="run-event__event">{msg.swarmEvent}</div>}
            <div className="run-event__content">
              <MessageContent kind={msg.kind} content={content} />
            </div>
            {showTraceCard && <TraceCard event={renderEvent} />}
          </div>
        </>
      )}
    </article>
  );
}

function ConversationHistory({
  messages,
  running,
}: {
  messages: MessageChunk[];
  running: boolean;
}) {
  const turns = useMemo(() => groupConversationTurns(messages), [messages]);

  return turns.map((turn, index) => {
    const active = running && index === turns.length - 1;
    const visibleTurn =
      active && turn.finalMessage
        ? {
            ...turn,
            workMessages: [...turn.workMessages, turn.finalMessage],
            finalMessage: null,
          }
        : turn;
    return <ConversationTurnView active={active} key={turn.id} turn={visibleTurn} />;
  });
}

function ConversationTurnView({
  active,
  turn,
}: {
  active: boolean;
  turn: ConversationTurn;
}) {
  const hasWork = active || turn.workMessages.length > 0;

  return (
    <section className="conversation-turn" data-turn-status={active ? "running" : "completed"}>
      {turn.userMessage && <RunEvent msg={turn.userMessage} />}
      {hasWork && <WorkDisclosure active={active} messages={turn.workMessages} turnId={turn.id} />}
      {turn.finalMessage && <RunEvent msg={turn.finalMessage} />}
    </section>
  );
}

function WorkDisclosure({
  active,
  messages,
  turnId,
}: {
  active: boolean;
  messages: MessageChunk[];
  turnId: string;
}) {
  const [expanded, setExpanded] = useState(active);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const detailsRef = useRef<HTMLDivElement>(null);
  const wasActive = useRef(active);
  const detailsId = `${turnId}-work-details`;
  const duration = workDurationMs(messages);
  const activities = groupWorkActivities(messages);
  const label = active
    ? "Working"
    : duration
      ? `Worked for ${formatWorkDuration(duration)}`
      : "Worked";

  useLayoutEffect(() => {
    if (wasActive.current && !active) {
      if (detailsRef.current?.contains(document.activeElement)) toggleRef.current?.focus();
      setExpanded(false);
    }
    wasActive.current = active;
  }, [active]);

  return (
    <section className={cx("work-disclosure", active && "is-active", expanded && "is-open")}>
      <button
        type="button"
        className="work-disclosure__toggle"
        aria-controls={detailsId}
        aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}
        ref={toggleRef}
      >
        <span>{label}</span>
        <ChevronRight aria-hidden="true" />
      </button>
      {expanded && (
        <div className="work-disclosure__details" id={detailsId} ref={detailsRef}>
          {messages.length > 0 ? (
            <div className="work-disclosure__events">
              {activities.map((activity) =>
                activity.kind === "message" ? (
                  <RunEvent
                    compact
                    key={`${messageKey(activity.message)}\u001f${activity.sourceIndex}`}
                    msg={activity.message}
                  />
                ) : (
                  <ToolActivityEvent
                    activity={activity.activity}
                    key={toolActivityKey(activity.activity)}
                  />
                ),
              )}
            </div>
          ) : (
            <output className="work-disclosure__pending">
              <Loader2 aria-hidden="true" />
              <span>Waiting for agent output</span>
            </output>
          )}
        </div>
      )}
    </section>
  );
}

function groupWorkActivities(messages: readonly MessageChunk[]): WorkActivity[] {
  const activities: WorkActivity[] = [];
  const tools: ToolActivity[] = [];

  messages.forEach((message, sourceIndex) => {
    if (message.kind === "tool_call") {
      const activity = { call: message, sourceIndex };
      activities.push({ kind: "tool", activity });
      tools.push(activity);
      return;
    }
    if (message.kind !== "tool_result") {
      activities.push({ kind: "message", message, sourceIndex });
      return;
    }

    const invocationId = message.render?.invocationId;
    const exactMatch = invocationId
      ? tools.find((activity) => activity.call?.render?.invocationId === invocationId)
      : undefined;
    const fallbackMatch = tools.find(
      (activity) =>
        activity.call?.toolName === message.toolName &&
        (!activity.result || !isTerminalToolResult(activity.result)),
    );
    const match = exactMatch ?? fallbackMatch;
    if (match) {
      match.result = message;
      return;
    }

    const activity = { result: message, sourceIndex };
    activities.push({ kind: "tool", activity });
    tools.push(activity);
  });

  return activities;
}

function isTerminalToolResult(message: MessageChunk): boolean {
  const status = normalizeMessageChunk(message, normalizeOptionsFromMessage(message)).status;
  return !["queued", "running"].includes(status);
}

function toolActivityKey(activity: ToolActivity): string {
  const message = activity.call ?? activity.result;
  const invocationId = message?.render?.invocationId;
  return invocationId
    ? `tool:${invocationId}`
    : `tool:${activity.sourceIndex}:${message ? messageKey(message) : "unknown"}`;
}

function ToolActivityEvent({ activity }: { activity: ToolActivity }) {
  const [expanded, setExpanded] = useState(false);
  const event = mergedToolActivityEvent(activity);
  const summary = describeToolActivity(event);
  const specialized = specializedTracePresentation(event);
  const failure = toolFailureSummary(event);
  const hasDetails = !!specialized || event.artifacts.length > 0 || !!failure;
  const detailsId = `${event.eventId}-activity-details`;
  const failed = event.status === "failed" || event.status === "canceled";
  const running = event.status === "queued" || event.status === "running";
  const StatusIcon = running ? Loader2 : failed ? XCircle : Check;

  return (
    <article
      className={cx(
        "tool-activity",
        running && "is-running",
        failed && "is-failed",
        expanded && "is-open",
      )}
      data-render-event-id={event.eventId}
      data-render-kind="tool_activity"
      data-render-status={event.status}
    >
      <button
        type="button"
        className="tool-activity__summary"
        aria-controls={hasDetails ? detailsId : undefined}
        aria-expanded={hasDetails ? expanded : undefined}
        aria-label={`${summary}, ${humanToolStatus(event.status)}`}
        disabled={!hasDetails}
        onClick={() => hasDetails && setExpanded((value) => !value)}
      >
        <span className="tool-activity__status" aria-hidden="true">
          <StatusIcon />
        </span>
        <span className="tool-activity__label">{summary}</span>
        {hasDetails && <ChevronRight className="tool-activity__chevron" aria-hidden="true" />}
      </button>
      {expanded && hasDetails && (
        <div className="tool-activity__details" id={detailsId}>
          {specialized && <SpecializedTracePresentation presentation={specialized} />}
          {failure && !specialized?.blocks.some((block) => block.content.includes(failure)) && (
            <p className="tool-activity__error">{failure}</p>
          )}
          <TraceArtifacts artifacts={event.artifacts} />
        </div>
      )}
    </article>
  );
}

function mergedToolActivityEvent(activity: ToolActivity): NormalizedRenderEvent {
  const call = activity.call
    ? normalizeMessageChunk(activity.call, normalizeOptionsFromMessage(activity.call))
    : undefined;
  const result = activity.result
    ? normalizeMessageChunk(activity.result, normalizeOptionsFromMessage(activity.result))
    : undefined;
  const base = call ?? result;
  if (!base) throw new Error("Tool activity requires a call or result.");

  return {
    ...base,
    eventId: call?.eventId ?? result?.eventId ?? base.eventId,
    invocationId: call?.invocationId ?? result?.invocationId,
    kind: result ? "tool_result" : "tool_call",
    status: result?.status ?? call?.status ?? base.status,
    title: call?.title ?? result?.title ?? base.title,
    summary: result?.summary ?? call?.summary ?? base.summary,
    toolName: call?.toolName ?? result?.toolName,
    input: call?.input,
    output: result?.output,
    artifacts: uniqueArtifacts([...(call?.artifacts ?? []), ...(result?.artifacts ?? [])]),
    provenance: { ...(call?.provenance ?? {}), ...(result?.provenance ?? {}) },
    startedAt: call?.startedAt ?? result?.startedAt,
    endedAt: result?.endedAt ?? call?.endedAt,
    durationMs: result?.durationMs ?? call?.durationMs,
    rawPayloadRef: undefined,
  };
}

function describeToolActivity(event: NormalizedRenderEvent): string {
  const payload = tracePayloadRecord(event) ?? {};
  const tool = (event.toolName ?? "tool").toLowerCase();
  const path = compactToolValue(
    stringValue(payload, [
      "path",
      "filePath",
      "file_path",
      "notebook_path",
      "targetPath",
      "target_path",
    ]),
  );
  const command = compactToolValue(stringValue(payload, ["command", "cmd"]));
  const query = compactToolValue(
    stringValue(payload, ["query", "pattern", "search", "search_query"]),
  );
  const running = event.status === "queued" || event.status === "running";
  const failed = event.status === "failed" || event.status === "canceled";

  if (/apply[_-]?patch|notebookedit|(^|[_-])(edit|patch)([_-]|$)/.test(tool)) {
    return toolActionSummary("Editing", "Edited", "Couldn’t edit", path, running, failed);
  }
  if (/(^|[_-])(write|create)([_-]|$)/.test(tool)) {
    return toolActionSummary("Writing", "Wrote", "Couldn’t write", path, running, failed);
  }
  if (/(^|[_-])(read|open)([_-]|$)/.test(tool)) {
    return toolActionSummary("Reading", "Read", "Couldn’t read", path, running, failed);
  }
  if (/(^|[_-])(glob|grep|search|find)([_-]|$)|toolsearch/.test(tool)) {
    const subject = query ? `“${query}”` : path;
    return toolActionSummary(
      "Searching for",
      "Searched for",
      "Couldn’t search for",
      subject,
      running,
      failed,
      "Searching files",
      "Searched files",
      "File search failed",
    );
  }
  if (/(^|[_-])(test|check|vitest|jest|pytest)([_-]|$)/.test(tool)) {
    return running ? "Running tests" : failed ? "Tests failed" : "Ran tests";
  }
  if (/(^|[_-])(bash|shell|terminal|exec_command|powershell)([_-]|$)/.test(tool)) {
    return toolActionSummary(
      "Running",
      "Ran",
      "Command failed:",
      command,
      running,
      failed,
      "Running command",
      "Ran command",
      "Command failed",
    );
  }
  if (/(^|[_-])(browser|chrome|playwright|computer[_-]?use)([_-]|$)/.test(tool)) {
    return running ? "Using the browser" : failed ? "Browser action failed" : "Used the browser";
  }
  if (/(^|[_-])(imagegen|image_generation|generate[_-]?image)([_-]|$)/.test(tool)) {
    return running
      ? "Generating an image"
      : failed
        ? "Image generation failed"
        : "Generated an image";
  }

  const label = humanizeToolName(event.toolName ?? "tool");
  return running ? `Using ${label}` : failed ? `${label} failed` : `Used ${label}`;
}

function toolActionSummary(
  pendingVerb: string,
  completedVerb: string,
  failedVerb: string,
  subject: string | undefined,
  running: boolean,
  failed: boolean,
  pendingFallback = pendingVerb,
  completedFallback = completedVerb,
  failedFallback = failedVerb,
): string {
  if (!subject) return running ? pendingFallback : failed ? failedFallback : completedFallback;
  return `${running ? pendingVerb : failed ? failedVerb : completedVerb} ${subject}`;
}

function compactToolValue(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const compact = value.replace(/\s+/g, " ").trim();
  return compact.length > 96 ? `${compact.slice(0, 93)}…` : compact;
}

function humanizeToolName(toolName: string): string {
  const publicName = toolName.includes("__") ? (toolName.split("__").at(-1) ?? toolName) : toolName;
  return publicName
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^\w/, (character) => character.toUpperCase());
}

function humanToolStatus(status: NormalizedRenderEvent["status"]): string {
  if (status === "queued" || status === "running") return "in progress";
  if (status === "failed") return "failed";
  if (status === "canceled") return "canceled";
  return "complete";
}

function toolFailureSummary(event: NormalizedRenderEvent): string | undefined {
  if (event.status !== "failed" && event.status !== "canceled") return undefined;
  const payload = tracePayloadRecord(event);
  return compactToolValue(
    payload ? stringValue(payload, ["error", "message", "stderr", "failure"]) : undefined,
  );
}

function uniqueArtifacts(artifacts: RenderArtifactReference[]): RenderArtifactReference[] {
  const seen = new Set<string>();
  return artifacts.filter((artifact, index) => {
    const key =
      artifact.artifactId ??
      [artifact.kind, artifact.path ?? "", artifact.title ?? "", index].join(":");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function groupConversationTurns(messages: MessageChunk[]): ConversationTurn[] {
  const grouped: Array<{
    id: string;
    userMessage: MessageChunk | null;
    responseMessages: MessageChunk[];
  }> = [];
  let current: (typeof grouped)[number] | null = null;

  messages.forEach((message, index) => {
    if (message.role === "user") {
      current = {
        id: `conversation-turn-${index}`,
        userMessage: message,
        responseMessages: [],
      };
      grouped.push(current);
      return;
    }

    if (!current) {
      current = {
        id: "conversation-turn-opening",
        userMessage: null,
        responseMessages: [],
      };
      grouped.push(current);
    }
    current.responseMessages.push(message);
  });

  return grouped.map(({ id, userMessage, responseMessages }) => {
    let finalMessageIndex = -1;
    responseMessages.forEach((message, index) => {
      if (isFinalResponse(message)) finalMessageIndex = index;
    });
    const finalMessage =
      finalMessageIndex >= 0 ? (responseMessages[finalMessageIndex] ?? null) : null;
    return {
      id,
      userMessage,
      workMessages:
        finalMessageIndex >= 0
          ? responseMessages.filter((_message, index) => index !== finalMessageIndex)
          : responseMessages,
      finalMessage,
    };
  });
}

function isFinalResponse(message: MessageChunk): boolean {
  return message.kind === "message" && message.role !== "user" && message.role !== "tool";
}

function workDurationMs(messages: MessageChunk[]): number | undefined {
  const starts = messages
    .map((message) => parseRenderTime(message.render?.startedAt))
    .filter((value): value is number => value !== undefined);
  const ends = messages
    .map((message) => parseRenderTime(message.render?.endedAt))
    .filter((value): value is number => value !== undefined);
  if (starts.length > 0 && ends.length > 0) {
    const elapsed = Math.max(...ends) - Math.min(...starts);
    if (elapsed > 0) return elapsed;
  }

  const explicitDurations = messages
    .map((message) => message.render?.durationMs)
    .filter((duration): duration is number => duration !== undefined && duration > 0);
  return explicitDurations.length === 1 ? explicitDurations[0] : undefined;
}

function parseRenderTime(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? undefined : parsed;
}

function formatWorkDuration(durationMs: number): string {
  const seconds = Math.max(1, Math.round(durationMs / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return remainder === 0 ? `${minutes}m` : `${minutes}m ${remainder}s`;
}

function withRequestTiming(
  messages: MessageChunk[],
  startedAt: string,
  endedAt: string,
): MessageChunk[] {
  const durationMs = Math.max(1, Date.parse(endedAt) - Date.parse(startedAt));
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

function mergeStreamingMessage(
  messages: readonly MessageChunk[],
  incoming: MessageChunk,
): MessageChunk[] {
  const previous = messages.at(-1);
  const mergeable = incoming.kind === "thinking" || incoming.kind === "message";
  if (
    mergeable &&
    previous?.kind === incoming.kind &&
    previous.role === incoming.role &&
    previous.agent === incoming.agent &&
    previous.toolName === incoming.toolName
  ) {
    return [
      ...messages.slice(0, -1),
      { ...previous, content: `${previous.content}${incoming.content}` },
    ];
  }
  return [...messages, incoming];
}

function normalizeThoughtMarkdown(content: string): string {
  const match = /^(\s*)\*\*([^\n]+)\*\*(\s*)$/.exec(content);
  return match ? `${match[1]}${match[2]}${match[3]}` : content;
}

function TraceCard({ event }: { event: NormalizedRenderEvent }) {
  const [expanded, setExpanded] = useState(false);
  const detailsId = `${event.eventId}-details`;
  const specialized = specializedTracePresentation(event);

  return (
    <section className="trace-card" aria-label={`${event.title} trace details`}>
      <div className="trace-card__summary">
        <div className="trace-card__title">
          <span>{event.title}</span>
          <span className={cx("trace-card__status", `trace-card__status--${event.status}`)}>
            {event.status}
          </span>
        </div>
        <p>{event.summary || event.status}</p>
      </div>
      <button
        type="button"
        className="trace-card__toggle"
        aria-controls={detailsId}
        aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}
      >
        {expanded ? <Minus aria-hidden="true" /> : <Plus aria-hidden="true" />}
        {expanded ? "Hide details" : "Show details"}
      </button>
      {expanded && (
        <div className="trace-card__details" id={detailsId}>
          {specialized && <SpecializedTracePresentation presentation={specialized} />}
          {event.input !== undefined && (
            <TraceDetailBlock title="Input" payload={event.input} tone="input" />
          )}
          {event.output !== undefined && (
            <TraceDetailBlock title="Output" payload={event.output} tone="output" />
          )}
          <TraceMetadata title="Provenance" entries={metadataEntries(event.provenance)} />
          <TraceArtifacts artifacts={event.artifacts} />
          {event.rawPayloadRef && (
            <TraceMetadata title="Raw payload ref" entries={[["ref", event.rawPayloadRef]]} />
          )}
        </div>
      )}
    </section>
  );
}

interface TracePresentationBlock {
  content: string;
  title: string;
  truncated?: boolean;
}

interface SpecializedTracePresentationModel {
  blocks: TracePresentationBlock[];
  fields: Array<[string, string]>;
  kind: string;
  title: string;
}

interface SpecializedTracePresentationDraft {
  blocks: Array<TracePresentationBlock | null>;
  fields: Array<[string, string] | null>;
  kind: string;
  title: string;
}

function SpecializedTracePresentation({
  presentation,
}: {
  presentation: SpecializedTracePresentationModel;
}) {
  return (
    <div className={cx("trace-card__special", `trace-card__special--${presentation.kind}`)}>
      <h4>{presentation.title}</h4>
      {presentation.fields.length > 0 && (
        <div className="trace-card__field-grid">
          {presentation.fields.map(([label, value]) => (
            <div className="trace-card__field" key={`${label}:${value}`}>
              <span>{label}</span>
              <strong>{value}</strong>
            </div>
          ))}
        </div>
      )}
      {presentation.blocks.map((block) => (
        <div className="trace-card__excerpt" key={block.title}>
          <div className="trace-card__excerpt-title">
            <span>{block.title}</span>
            {block.truncated && <em>truncated</em>}
          </div>
          <pre>{block.content}</pre>
        </div>
      ))}
    </div>
  );
}

function TraceDetailBlock({
  payload,
  title,
  tone,
}: {
  payload: unknown;
  title: string;
  tone: "input" | "output";
}) {
  return (
    <div className={cx("trace-card__detail", `trace-card__detail--${tone}`)}>
      <h4>{title}</h4>
      <pre>{stringifyRenderPayload(payload)}</pre>
    </div>
  );
}

function TraceMetadata({
  entries,
  title,
}: {
  entries: Array<[string, string]>;
  title: string;
}) {
  if (entries.length === 0) return null;

  return (
    <div className="trace-card__metadata">
      <h4>{title}</h4>
      <div className="trace-card__chips">
        {entries.map(([key, value]) => (
          <span className="trace-card__chip" key={`${key}:${value}`}>
            <span>{key}</span>
            <strong>{value}</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

function TraceArtifacts({ artifacts }: { artifacts: RenderArtifactReference[] }) {
  if (artifacts.length === 0) return null;

  return (
    <div className="trace-card__metadata">
      <h4>Artifacts</h4>
      <div className="trace-card__chips">
        {artifacts.map((artifact, index) => (
          <span
            className="trace-card__chip trace-card__chip--artifact"
            key={artifact.artifactId ?? `${artifact.kind}:${artifact.path ?? index}`}
          >
            <span>{artifact.kind}</span>
            <strong>{artifact.title ?? artifact.path ?? artifact.artifactId ?? "artifact"}</strong>
            {artifact.truncated && <em>truncated</em>}
          </span>
        ))}
      </div>
    </div>
  );
}

function specializedTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  return (
    terminalTracePresentation(event) ??
    diffTracePresentation(event) ??
    testTracePresentation(event) ??
    fileTracePresentation(event) ??
    mcpTracePresentation(event) ??
    automationTracePresentation(event) ??
    mediaTracePresentation(event)
  );
}

function terminalTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  const payload = tracePayloadRecord(event);
  const tool = `${event.toolName ?? ""} ${event.title}`.toLowerCase();
  if (
    !payload ||
    (!tool.match(/\b(terminal|shell|bash|zsh|command)\b/) &&
      !hasAnyKey(payload, ["command", "cmd", "stdout", "stderr", "exitCode", "exit_code"]))
  ) {
    return null;
  }

  return compactPresentation({
    blocks: [
      traceBlock("Stdout", stringValue(payload, ["stdout", "stdoutText", "outputText"]), {
        truncated: booleanValue(payload, ["stdoutTruncated", "truncated"]),
      }),
      traceBlock("Stderr", stringValue(payload, ["stderr", "stderrText", "error"]), {
        truncated: booleanValue(payload, ["stderrTruncated"]),
      }),
    ],
    fields: [
      traceField("command", stringValue(payload, ["command", "cmd"])),
      traceField("cwd", stringValue(payload, ["cwd", "workdir", "workingDirectory"])),
      traceField("status", stringValue(payload, ["status"]) ?? event.status),
      traceField("exit", stringValue(payload, ["exitCode", "exit_code", "code"])),
      traceField("duration", durationText(payload, event)),
    ],
    kind: "terminal",
    title: "Terminal",
  });
}

function diffTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  const payload = tracePayloadRecord(event);
  const tool = `${event.toolName ?? ""} ${event.title}`.toLowerCase();
  const hasDiffArtifact = event.artifacts.some((artifact) => artifact.kind === "diff");
  const diff = payload ? stringValue(payload, ["diff", "patch", "unifiedDiff"]) : undefined;
  if (!diff && !hasDiffArtifact && !tool.match(/\b(diff|patch)\b/)) return null;

  return compactPresentation({
    blocks: [traceBlock("Diff", diff)],
    fields: [
      traceField(
        "path",
        payload ? stringValue(payload, ["path", "filePath", "targetPath"]) : undefined,
      ),
      traceField("artifacts", hasDiffArtifact ? String(event.artifacts.length) : undefined),
    ],
    kind: "diff",
    title: "Diff",
  });
}

function testTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  const payload = tracePayloadRecord(event);
  const tool = `${event.toolName ?? ""} ${event.title}`.toLowerCase();
  if (
    !payload ||
    (!tool.match(/\b(test|check|vitest|jest|pytest|pnpm)\b/) &&
      !hasAnyKey(payload, ["passed", "failed", "testCount", "failures"]))
  ) {
    return null;
  }

  return compactPresentation({
    blocks: [
      traceBlock("Failures", stringValue(payload, ["failures", "failure", "failureExcerpt"])),
      traceBlock("Output", stringValue(payload, ["output", "stdout"])),
    ],
    fields: [
      traceField("command", stringValue(payload, ["command", "cmd"])),
      traceField("status", stringValue(payload, ["status"]) ?? event.status),
      traceField("passed", stringValue(payload, ["passed", "passedCount"])),
      traceField("failed", stringValue(payload, ["failed", "failedCount"])),
      traceField("tests", stringValue(payload, ["testCount", "tests"])),
    ],
    kind: "test",
    title: "Test/check",
  });
}

function fileTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  const payload = tracePayloadRecord(event);
  if (!payload) return null;
  const path = stringValue(payload, ["path", "filePath", "targetPath"]);
  const operation = stringValue(payload, ["operation", "action", "mode"]);
  if (
    !path ||
    (!operation && !hasAnyKey(payload, ["lineStart", "lineEnd", "byteCount", "content"]))
  ) {
    return null;
  }

  return compactPresentation({
    blocks: [
      traceBlock("Preview", stringValue(payload, ["preview", "content", "excerpt"]), {
        truncated: booleanValue(payload, ["truncated"]),
      }),
    ],
    fields: [
      traceField("operation", operation),
      traceField("path", path),
      traceField("range", rangeText(payload)),
      traceField("bytes", stringValue(payload, ["byteCount", "bytes"])),
    ],
    kind: "file",
    title: "File",
  });
}

function mcpTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  const payload = tracePayloadRecord(event);
  const toolText = `${event.toolName ?? ""} ${event.title}`.toLowerCase();
  const hasMcpShape =
    !!event.provenance.mcpServer ||
    toolText.includes("mcp") ||
    !!(payload && hasAnyKey(payload, ["mcpServer", "server", "tool", "toolName"]));
  if (!hasMcpShape) return null;

  const server =
    event.provenance.mcpServer ?? (payload && stringValue(payload, ["mcpServer", "server"]));
  const tool =
    (payload && stringValue(payload, ["tool", "toolName", "name"])) ??
    (toolText.includes("mcp") ? event.toolName : undefined);
  if (!server && !tool) return null;

  return compactPresentation({
    blocks: [
      traceBlock(
        "Result",
        payload ? stringValue(payload, ["result", "summary", "output"]) : undefined,
      ),
    ],
    fields: [
      traceField("server", server),
      traceField("tool", tool),
      traceField("plugin", event.provenance.pluginId),
    ],
    kind: "mcp",
    title: "MCP",
  });
}

function automationTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  const payload = tracePayloadRecord(event);
  const tool = `${event.toolName ?? ""} ${event.title}`.toLowerCase();
  if (
    !tool.match(/\b(browser|chrome|playwright|automation|app)\b/) &&
    !hasAutomationArtifact(event)
  ) {
    return null;
  }

  return compactPresentation({
    blocks: [
      traceBlock("DOM", payload ? stringValue(payload, ["dom", "excerpt", "summary"]) : undefined),
    ],
    fields: [
      traceField("target", payload ? stringValue(payload, ["target", "url", "app"]) : undefined),
      traceField("action", payload ? stringValue(payload, ["action", "operation"]) : undefined),
      traceField(
        "artifacts",
        hasAutomationArtifact(event) ? String(event.artifacts.length) : undefined,
      ),
    ],
    kind: "automation",
    title: "Automation",
  });
}

function mediaTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  const mediaArtifacts = event.artifacts.filter((artifact) =>
    ["image", "screenshot", "table", "html"].includes(artifact.kind),
  );
  if (mediaArtifacts.length === 0) return null;

  return compactPresentation({
    blocks: [],
    fields: [
      traceField("artifacts", String(mediaArtifacts.length)),
      traceField(
        "types",
        Array.from(new Set(mediaArtifacts.map((artifact) => artifact.kind))).join(", "),
      ),
    ],
    kind: "media",
    title: "Generated media",
  });
}

function compactPresentation(
  presentation: SpecializedTracePresentationDraft,
): SpecializedTracePresentationModel | null {
  const fields = presentation.fields.filter((field): field is [string, string] => !!field);
  const blocks = presentation.blocks.filter(
    (block): block is TracePresentationBlock => block !== null,
  );
  if (fields.length === 0 && blocks.length === 0) return null;
  return { ...presentation, blocks, fields };
}

function tracePayloadRecord(event: NormalizedRenderEvent): Record<string, unknown> | undefined {
  const input = isRecord(event.input) ? event.input : undefined;
  const output = isRecord(event.output) ? event.output : undefined;
  if (input && output) return { ...input, ...output };
  return output ?? input;
}

function traceField(label: string, value: string | undefined): [string, string] | null {
  return value ? [label, value] : null;
}

function traceBlock(
  title: string,
  content: string | undefined,
  options: { truncated?: boolean } = {},
): TracePresentationBlock | null {
  if (!content) return null;
  return { content, title, truncated: options.truncated };
}

function stringValue(record: Record<string, unknown>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value) return value;
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    if (Array.isArray(value) && value.length > 0) return value.map(String).join(", ");
  }
  return undefined;
}

function booleanValue(record: Record<string, unknown>, keys: string[]): boolean | undefined {
  for (const key of keys) {
    if (typeof record[key] === "boolean") return record[key];
  }
  return undefined;
}

function hasAnyKey(record: Record<string, unknown>, keys: string[]): boolean {
  return keys.some((key) => record[key] !== undefined);
}

function durationText(
  record: Record<string, unknown>,
  event: NormalizedRenderEvent,
): string | undefined {
  const value = stringValue(record, ["durationMs", "duration"]);
  if (value) return value.endsWith("ms") ? value : `${value}ms`;
  return event.durationMs === undefined ? undefined : `${event.durationMs}ms`;
}

function rangeText(record: Record<string, unknown>): string | undefined {
  const start = stringValue(record, ["lineStart", "startLine"]);
  const end = stringValue(record, ["lineEnd", "endLine"]);
  if (!start && !end) return undefined;
  return `${start ?? "?"}-${end ?? "?"}`;
}

function hasAutomationArtifact(event: NormalizedRenderEvent): boolean {
  return event.artifacts.some(
    (artifact) => artifact.kind === "screenshot" || artifact.kind === "html",
  );
}

function Button({
  children,
  className,
  variant = "default",
  size = "md",
  type = "button",
  ...props
}: ButtonProps) {
  return (
    <button
      type={type}
      className={cx("button", `button--${variant}`, `button--${size}`, className)}
      {...props}
    >
      {children}
    </button>
  );
}

function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: string }) {
  return <span className={cx("badge", `badge--${tone}`)}>{children}</span>;
}

function normalizeOptionsFromMessage(msg: MessageChunk): NormalizeMessageChunkOptions {
  return {
    artifacts: msg.render?.artifacts,
    durationMs: msg.render?.durationMs,
    endedAt: msg.render?.endedAt,
    invocationId: msg.render?.invocationId,
    parentMessageId: msg.render?.parentMessageId,
    provenance: msg.render?.provenance,
    rawPayloadRef: msg.render?.rawPayloadRef,
    source: msg.render?.source,
    startedAt: msg.render?.startedAt,
    status: msg.render?.status,
  };
}

function isTraceCardEvent(event: NormalizedRenderEvent): boolean {
  return event.kind === "tool_call" || event.kind === "tool_result" || event.kind === "trace";
}

function metadataEntries(provenance: RenderProvenance): Array<[string, string]> {
  return Object.entries(provenance)
    .filter((entry): entry is [string, string] => typeof entry[1] === "string" && !!entry[1])
    .filter(([key]) => key !== "agent")
    .sort(([left], [right]) => left.localeCompare(right));
}

function messagePresentation(msg: MessageChunk): {
  icon: LucideIcon;
  label: string;
  tone: string;
  meta: string;
} {
  if (msg.role === "user") {
    return { icon: User, label: "you", tone: "user", meta: "message" };
  }
  if (msg.role === "system") {
    return { icon: XCircle, label: "system", tone: "system", meta: msg.kind };
  }
  if (msg.kind === "thinking") {
    return { icon: Loader2, label: msg.agent ?? "agent", tone: "thinking", meta: "thinking" };
  }
  if (msg.kind === "tool_call") {
    return { icon: TerminalIcon, label: msg.agent ?? "tool", tone: "tool", meta: "call" };
  }
  if (msg.kind === "tool_result") {
    return { icon: Code2, label: msg.agent ?? "tool", tone: "tool", meta: "result" };
  }
  return { icon: Bot, label: msg.agent ?? "assistant", tone: "assistant", meta: "message" };
}

function compactWorkPresentation(msg: MessageChunk): ReturnType<typeof messagePresentation> {
  const presentation = messagePresentation(msg);
  if (msg.kind === "thinking") return { ...presentation, label: "Reasoning", meta: "thought" };
  if (msg.kind === "tool_call") {
    return { ...presentation, label: msg.toolName ?? "Tool", meta: "call" };
  }
  if (msg.kind === "tool_result") {
    return { ...presentation, label: msg.toolName ?? "Tool", meta: "result" };
  }
  return { ...presentation, label: "Progress" };
}

function renderEventContent(
  msg: MessageChunk,
  event: ReturnType<typeof normalizeMessageChunk>,
): string {
  if (isTraceCardEvent(event)) {
    return event.summary || event.title;
  }
  return msg.content;
}

function stringifyRenderPayload(payload: unknown): string {
  return typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
}

function messageKey(msg: MessageChunk): string {
  return [msg.role, msg.kind, msg.agent ?? "", msg.toolName ?? "", msg.content].join("\u001f");
}

function parseWorkflowJson(source: string): WorkflowParseResult {
  const emptyResult: WorkflowParseResult = { config: null, error: null, nodes: [], edges: [] };
  const trimmed = source.trim();
  if (!trimmed) {
    return { ...emptyResult, error: "Workflow JSON is empty." };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (error) {
    return {
      ...emptyResult,
      error: `Workflow JSON parse error: ${errorMessage(error)}`,
    };
  }

  if (!isRecord(parsed)) {
    return { ...emptyResult, error: "Workflow JSON must be an object." };
  }

  const name = parsed.name;
  if (typeof name !== "string" || name.trim() === "") {
    return { ...emptyResult, error: "Workflow JSON needs a non-empty name." };
  }

  const root = parsed.root;
  if (typeof root !== "string" || root.trim() === "") {
    return { ...emptyResult, error: "Workflow JSON needs a non-empty root." };
  }

  const rawNodes = parsed.nodes;
  if (!isRecord(rawNodes) || Object.keys(rawNodes).length === 0) {
    return { ...emptyResult, error: "Workflow JSON needs a non-empty nodes object." };
  }

  if (!Object.hasOwn(rawNodes, root)) {
    return { ...emptyResult, error: `Workflow JSON root "${root}" is not in nodes.` };
  }

  const nodes: WorkflowGraphNode[] = [];
  for (const [id, value] of Object.entries(rawNodes)) {
    if (!isRecord(value)) {
      return { ...emptyResult, error: `Workflow node "${id}" must be an object.` };
    }

    const kind = value.kind;
    if (kind !== "agent" && kind !== "tool" && kind !== "swarm") {
      return {
        ...emptyResult,
        error: `Workflow node "${id}" needs kind "agent", "tool", or "swarm".`,
      };
    }

    if (kind === "agent" && !isRecord(value.agent)) {
      return { ...emptyResult, error: `Workflow agent node "${id}" needs agent config.` };
    }
    if (kind === "tool" && !isRecord(value.tool)) {
      return { ...emptyResult, error: `Workflow tool node "${id}" needs tool config.` };
    }
    if (kind === "swarm" && !isRecord(value.swarm)) {
      return { ...emptyResult, error: `Workflow swarm node "${id}" needs swarm config.` };
    }

    nodes.push(toWorkflowGraphNode(id, value as SwarmNodeConfig, root));
  }

  const rawEdges = parsed.edges;
  if (!Array.isArray(rawEdges)) {
    return { ...emptyResult, error: "Workflow JSON needs an edges array." };
  }

  const edges: EdgeConfig[] = [];
  for (const [index, value] of rawEdges.entries()) {
    if (!isRecord(value)) {
      return { ...emptyResult, error: `Workflow edge ${index + 1} must be an object.` };
    }
    if (typeof value.source !== "string" || value.source.trim() === "") {
      return { ...emptyResult, error: `Workflow edge ${index + 1} needs source.` };
    }
    if (typeof value.target !== "string" || value.target.trim() === "") {
      return { ...emptyResult, error: `Workflow edge ${index + 1} needs target.` };
    }
    if (value.condition !== undefined && typeof value.condition !== "string") {
      return { ...emptyResult, error: `Workflow edge ${index + 1} condition must be a string.` };
    }

    edges.push({
      source: value.source,
      target: value.target,
      condition: value.condition,
    });
  }

  return {
    config: parsed as SwarmConfig,
    error: null,
    nodes,
    edges,
  };
}

function toWorkflowGraphNode(id: string, node: SwarmNodeConfig, root: string): WorkflowGraphNode {
  const source = node.kind === "agent" ? node.agent : node.kind === "tool" ? node.tool : node.swarm;
  const title = readString(source, "name") || id;
  const detail = readString(source, "description") || readString(source, "instructions") || title;
  const harness = node.kind === "agent" ? readHarnessDescriptor(source) : undefined;
  const harnessId =
    node.kind === "agent"
      ? harnessIdFromBackend(isRecord(source) ? source.backend : undefined)
      : undefined;
  const softwareLabel = formatSoftwareLabel(harness);
  const mcpsLabel = formatNamedList(harness?.mcps);
  const skillsLabel = formatStringList(harness?.skills);
  const projectFilesLabel = formatStringList(harness?.projectFiles);
  return {
    id,
    kind: node.kind,
    displayKind: id.includes("trigger") ? "trigger" : node.kind,
    title,
    detail,
    harnessId,
    harnessLabel: harnessId ? harnessLabel(harnessId) : undefined,
    harness,
    softwareLabel,
    mcpsLabel,
    skillsLabel,
    projectFilesLabel,
    model: readString(source, "model"),
    isRoot: id === root,
  };
}

function workflowNodeLayout(id: string, fallbackIndex: number): { x: number; y: number } {
  const layout: Record<string, { x: number; y: number }> = {
    triage_agent: { x: 36, y: 210 },
    researcher_agent: { x: 292, y: 128 },
    writer_agent: { x: 548, y: 210 },
  };
  return layout[id] ?? { x: 36 + fallbackIndex * 256, y: 210 };
}

function connectorPath(source: { x: number; y: number }, target: { x: number; y: number }): string {
  const sourceX = source.x + 220;
  const sourceY = source.y + 112;
  const targetX = target.x;
  const targetY = target.y + 112;
  const gap = Math.max(24, Math.abs(targetX - sourceX));
  const curve = Math.min(92, Math.max(24, gap * 0.5));
  return `M ${sourceX} ${sourceY} C ${sourceX + curve} ${sourceY}, ${targetX - curve} ${targetY}, ${targetX} ${targetY}`;
}

function nodeIcon(kind: WorkflowGraphNode["displayKind"]): LucideIcon {
  switch (kind) {
    case "trigger":
      return Play;
    case "tool":
      return Wrench;
    case "swarm":
      return Workflow;
    default:
      return Bot;
  }
}

function readString(source: unknown, key: string): string {
  if (!isRecord(source)) return "";
  const value = source[key];
  return typeof value === "string" ? value : "";
}

function readHarnessDescriptor(source: unknown): HarnessDescriptor | undefined {
  if (!isRecord(source) || !isRecord(source.parameters)) return undefined;
  const harness = source.parameters.harness;
  if (!isRecord(harness)) return undefined;

  const software = isRecord(harness.software)
    ? {
        name: readString(harness.software, "name") || undefined,
        version: readString(harness.software, "version") || undefined,
        runner: readString(harness.software, "runner") || undefined,
        command: readStringArray(harness.software.command),
      }
    : undefined;

  return {
    software,
    mcps: readMcpList(harness.mcps),
    skills: readStringArray(harness.skills),
    projectFiles: readStringArray(harness.projectFiles),
  };
}

function formatSoftwareLabel(harness: HarnessDescriptor | undefined): string {
  const software = harness?.software;
  if (!software?.name) return "";
  return software.version ? `${software.name}@${software.version}` : software.name;
}

function formatNamedList(items: HarnessDescriptor["mcps"]): string {
  if (!items || items.length === 0) return "";
  return items
    .map((item) => (typeof item === "string" ? item : item.name))
    .filter((item): item is string => Boolean(item))
    .join(", ");
}

function formatStringList(items: string[] | undefined): string {
  return items?.filter(Boolean).join(", ") ?? "";
}

function capabilityCount(
  bundle: ExtensionBundleSummary,
  key: keyof NonNullable<ExtensionBundleSummary["capabilities"]>,
): number {
  return bundle.capabilities?.[key]?.length ?? 0;
}

function formatComponentCounts(counts: Record<string, number> | undefined): string[] {
  if (!counts) return [];
  return Object.entries(counts)
    .filter(([, count]) => Number.isFinite(count) && count > 0)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, count]) => `${count} ${key}`);
}

function formatSoftwareSummary(software: ExtensionHarnessSummary["software"]): string {
  if (!software?.name) return "software";
  return software.version ? `${software.name}@${software.version}` : software.name;
}

function extensionComponentRows(
  inventory: ExtensionCapabilityInventory | undefined,
): Array<{ id: string; kind: string; title: string; detail?: string; chips: string[] }> {
  if (!inventory) return [];
  return [
    ...(inventory.commands ?? []).map((item) => ({
      id: item.id,
      kind: "command",
      title: item.name ?? item.id,
      detail: item.scope,
      chips: item.command ? [item.command.join(" ")] : [],
    })),
    ...(inventory.lspServers ?? []).map((item) => ({
      id: item.id,
      kind: "LSP",
      title: item.name ?? item.id,
      detail: item.scope,
      chips: [
        ...(item.languages ?? []),
        ...(item.languageIds ?? []),
        ...formatExtensionCommand(item.command, item.args),
      ],
    })),
    ...(inventory.hooks ?? []).map((item) => ({
      id: item.id,
      kind: "hook",
      title: item.name ?? item.id,
      detail: item.event,
      chips: [],
    })),
    ...(inventory.monitors ?? []).map((item) => ({
      id: item.id,
      kind: "monitor",
      title: item.name ?? item.id,
      detail: item.trigger,
      chips: item.schedule ? [item.schedule] : [],
    })),
    ...(inventory.outputStyles ?? []).map((item) => ({
      id: item.id,
      kind: "output style",
      title: item.name ?? item.id,
      detail: item.path,
      chips: [],
    })),
    ...(inventory.settings ?? []).map((item) => ({
      id: item.id,
      kind: "setting",
      title: item.name ?? item.id,
      detail: item.valueType,
      chips: item.required ? ["required"] : [],
    })),
    ...(inventory.assets ?? []).map((item) => ({
      id: item.id,
      kind: "asset",
      title: item.name ?? item.id,
      detail: item.kind,
      chips: [item.path, item.url].filter((value): value is string => Boolean(value)),
    })),
    ...(inventory.permissions ?? []).map((item) => ({
      id: item.id,
      kind: "permission",
      title: item.kind,
      detail: item.access,
      chips: [item.target, item.required ? "required" : undefined].filter(
        (value): value is string => Boolean(value),
      ),
    })),
    ...(inventory.authPolicies ?? []).map((item) => ({
      id: item.id,
      kind: "auth policy",
      title: item.kind ?? item.id,
      detail: item.required ? "required" : "optional",
      chips: item.secretRefs?.length ? [`${item.secretRefs.length} secret refs`] : [],
    })),
  ];
}

function formatExtensionCommand(
  command: string[] | string | undefined,
  args: string[] | undefined,
): string[] {
  if (Array.isArray(command)) return [command.join(" ")];
  if (command) return [[command, ...(args ?? [])].join(" ")];
  return [];
}

function extensionUiContributionChips(contribution: ExtensionUiContributionSummary): string[] {
  const chips: Array<string | undefined> = [
    contribution.sourcePluginId ? `via ${contribution.sourcePluginId}` : undefined,
    contribution.commandId ? `command ${contribution.commandId}` : undefined,
    contribution.assetRef ? `asset ${contribution.assetRef}` : undefined,
    contribution.target ? `target ${contribution.target}` : undefined,
    contribution.provenance,
  ];
  chips.push(...(contribution.settingIds ?? []).map((id) => `setting ${id}`));
  chips.push(...(contribution.permissionIds ?? []).map((id) => `permission ${id}`));
  chips.push(...(contribution.authPolicyIds ?? []).map((id) => `auth ${id}`));
  return uniqueStrings(chips.filter((chip): chip is string => Boolean(chip)));
}

function extensionSkillChips(skill: ExtensionSkillSummary): string[] {
  const chips: Array<string | undefined> = [
    skill.canonicalPath ? `canonical ${skill.canonicalPath}` : undefined,
    skill.governanceRef ? `governance ${skill.governanceRef}` : undefined,
    skill.readOnly ? "read-only" : undefined,
  ];
  chips.push(...(skill.requiresGateSkillIds ?? []).map((id) => `gate ${id}`));
  chips.push(
    ...(skill.hostExposures ?? []).map((exposure) =>
      [exposure.host, exposure.status ?? "plugin"].filter(Boolean).join(" "),
    ),
  );
  chips.push(
    ...(skill.hostExposures ?? []).flatMap((exposure) =>
      [
        exposure.manifestPath ? `manifest ${exposure.manifestPath}` : undefined,
        exposure.rulesPath ? `rules ${exposure.rulesPath}` : undefined,
        exposure.marketplaceSourceId ? `source ${exposure.marketplaceSourceId}` : undefined,
      ].filter((value): value is string => Boolean(value)),
    ),
  );
  return uniqueStrings(chips.filter((chip): chip is string => Boolean(chip)));
}

function agentPlanTone(plan: ExtensionAgentPlanSummary): string {
  if (plan.status === "ready") return "active";
  if (plan.status === "draft" || plan.status === "stale") return "neutral";
  return "danger";
}

function planBlockedTitle(plan: ExtensionAgentPlanSummary | undefined): string {
  if (!plan) return "Agent profile is not ready.";
  const blocked = blockedPlanRequirements(plan);
  return blocked.length > 0
    ? blocked.map((requirement) => requirement.message).join("; ")
    : plan.status;
}

function agentPlanChips(
  plan: ExtensionAgentPlanSummary | undefined,
  agent: ExtensionAgentSummary,
): string[] {
  const chips: Array<string | undefined> = [];
  if (plan) {
    chips.push(`${plan.pluginIds?.length ?? 0} plugins`);
    chips.push(...(plan.skills ?? []).map((skill) => skill.id));
    chips.push(...(plan.mcpServers ?? []).map((server) => server.id));
    chips.push(...capabilitySourceChips(plan));
    if (plan.context) chips.push(`context ${plan.context.mode}/${plan.context.strategy}`);
    if (plan.permissions?.summary) chips.push(`permissions ${plan.permissions.summary}`);
    chips.push(...(plan.requirements ?? []).filter(showPlanRequirement).map(planRequirementLabel));
  } else {
    chips.push(...(agent.skills ?? []));
    chips.push(...(agent.mcpServers ?? []));
  }
  chips.push(...(agent.tools ?? []).map((tool) => `tool ${tool}`));
  chips.push(...(agent.disallowedTools ?? []).map((tool) => `blocked ${tool}`));
  if (agent.maxTurns) chips.push(`${agent.maxTurns} turns`);
  if (agent.effort) chips.push(`effort ${agent.effort}`);
  if (agent.definition?.host) chips.push(nativeAgentHostLabel(agent.definition.host));
  if (!agent.modelId && agent.nativeModel) chips.push(`native model ${agent.nativeModel}`);
  if (agent.sandboxMode) chips.push(`sandbox ${agent.sandboxMode}`);
  if (agent.isolation) chips.push(`isolation ${agent.isolation}`);
  return uniqueStrings(chips.filter((chip): chip is string => Boolean(chip)));
}

function capabilitySourceChips(plan: ExtensionAgentPlanSummary): string[] {
  const sourceIds = [
    ...(plan.skills ?? []).flatMap((skill) => (skill.sourcePluginId ? [skill.sourcePluginId] : [])),
    ...(plan.mcpServers ?? []).flatMap((server) =>
      server.sourcePluginId ? [server.sourcePluginId] : [],
    ),
  ];
  return uniqueStrings(sourceIds).map((sourceId) => `via ${sourceId}`);
}

function blockedPlanRequirements(
  plan: ExtensionAgentPlanSummary,
): ExtensionAgentPlanRequirementSummary[] {
  return (plan.requirements ?? []).filter(
    (requirement) => requirement.status !== "ok" && requirement.status !== "unknown",
  );
}

function showPlanRequirement(requirement: ExtensionAgentPlanRequirementSummary): boolean {
  return requirement.status !== "ok";
}

function planRequirementLabel(requirement: ExtensionAgentPlanRequirementSummary): string {
  if (requirement.kind === "secret") return "secret required";
  const id = requirement.id ? ` ${requirement.id}` : "";
  return `${requirement.status} ${requirement.kind}${id}`;
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values)];
}

function extensionAgentComposition(agent: ExtensionAgentSummary): AgentCompositionPayload {
  return {
    id: `desktop-${agent.id}`,
    agentProfileId: agent.id,
    host: "local",
  };
}

function readStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const items = value.filter((item): item is string => typeof item === "string" && item !== "");
  return items.length > 0 ? items : undefined;
}

function readMcpList(value: unknown): HarnessDescriptor["mcps"] | undefined {
  if (!Array.isArray(value)) return undefined;
  type HarnessMcp = NonNullable<HarnessDescriptor["mcps"]>[number];
  const items = value.flatMap((item): HarnessMcp[] => {
    if (typeof item === "string" && item !== "") return [item];
    if (!isRecord(item)) return [];
    const name = readString(item, "name");
    if (!name) return [];
    return [
      {
        name,
        transport: readString(item, "transport") || undefined,
        scope: readString(item, "scope") || undefined,
      },
    ];
  });
  return items.length > 0 ? items : undefined;
}

function harnessIdFromBackend(backend: unknown): string {
  if (!isRecord(backend)) return "swarmx";

  const type = readString(backend, "type");
  if (type === "swarmx" || type === "claude_code") return type;

  const program = readString(backend, "program");
  if (program === "opencode" || program === "hermes" || program === "openclaw") return program;

  const args = Array.isArray(backend.args)
    ? backend.args.filter((arg): arg is string => typeof arg === "string")
    : [];
  const commandLine = [program, ...args].join(" ");
  if (commandLine.includes("@agentclientprotocol/codex-acp")) return "codex";
  if (commandLine.includes("@agentclientprotocol/claude-agent-acp")) return "claude_code";
  if (commandLine.includes("pi-acp")) return "pi";

  return type || program || "custom";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function buildSessionErrors(
  discoveredErrors: SessionDiscoveryError[],
  localError: unknown,
  groupedError: unknown,
): SessionDiscoveryError[] {
  const errors = [...discoveredErrors];
  if (localError) {
    errors.push({
      harnessId: "local-sessions",
      harnessLabel: "Local Sessions",
      message: errorMessage(localError),
    });
  }
  if (groupedError) {
    errors.push({
      harnessId: "acp-sessions",
      harnessLabel: "ACP Sessions",
      message: errorMessage(groupedError),
    });
  }
  return errors;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function loadDiscoveredSessionDetail(session: DiscoveredSession): Promise<SessionData | null> {
  return api.loadDiscoveredSession(session) as Promise<SessionData | null>;
}

function sessionDetailKey(
  session: DiscoveredSession,
): readonly ["session:detail", string, string, string, string] {
  return ["session:detail", session.source, session.harnessId, session.id, session.cwd];
}

function sessionCacheId(session: DiscoveredSession): string {
  return sessionDetailKey(session).join("\u001f");
}

function flattenSessions(groups: SessionGroup[]): DiscoveredSession[] {
  return groups.flatMap((group) => group.sessions);
}

function preloadSessionCandidates(groups: SessionGroup[]): DiscoveredSession[] {
  return flattenSessions(groups)
    .filter((session) => session.source === "local")
    .slice(0, LOCAL_SESSION_PRELOAD_LIMIT);
}

function mergeLocalSessionsIntoGroups(
  groups: SessionGroup[],
  sessions: SessionData[],
  mode: SessionGroupMode,
): SessionGroup[] {
  const externalSessions = groups
    .flatMap((group) => group.sessions)
    .filter((session) => session.source !== "local");
  const localSessions = sessions.map(localSessionToDiscovered);
  return groupDisplaySessions([...externalSessions, ...localSessions], mode);
}

function mergeProjectsIntoSessionGroups(
  projects: ProjectData[],
  groups: SessionGroup[],
): ProjectSessionGroup[] {
  const remaining = new Set(groups.map((group) => group.id));
  const projectGroups = projects.map<ProjectSessionGroup>((project) => {
    const matchingGroups = groups.filter((group) =>
      group.sessions.some(
        (session) => session.projectId === project.id || sameProjectPath(session.cwd, project.cwd),
      ),
    );
    for (const group of matchingGroups) remaining.delete(group.id);
    return {
      id: project.id,
      label: project.name,
      cwd: project.cwd,
      project,
      sessions: sortDisplaySessions(matchingGroups.flatMap((group) => group.sessions)),
    };
  });

  const unmatched = groups
    .filter((group) => remaining.has(group.id))
    .flatMap<ProjectSessionGroup>((group) => {
      const sessions = group.sessions.filter((session) => !session.projectId);
      if (sessions.length === 0) return [];
      return [
        {
          ...group,
          label: group.id === "__no_project__" ? "No project" : projectDisplayName(group.label),
          cwd: group.id === "__no_project__" ? "" : group.label,
          sessions,
        },
      ];
    });
  return [...projectGroups, ...unmatched];
}

function sortProjectSessionGroups(
  groups: ProjectSessionGroup[],
  mode: ProjectSortMode,
): ProjectSessionGroup[] {
  if (mode === "manual") return groups;
  return [...groups].sort((left, right) => {
    if (mode === "priority" && Boolean(left.project?.pinned) !== Boolean(right.project?.pinned)) {
      return left.project?.pinned ? -1 : 1;
    }
    const timeDifference = projectSessionGroupTime(right) - projectSessionGroupTime(left);
    return timeDifference || left.label.localeCompare(right.label);
  });
}

function flattenProjectSessions(
  groups: ProjectSessionGroup[],
  mode: ProjectSortMode,
): DiscoveredSession[] {
  const entries = groups.flatMap((group) =>
    group.sessions.map((session) => ({
      session,
      pinned: Boolean(group.project?.pinned),
    })),
  );
  if (mode === "manual") return entries.map(({ session }) => session);
  return entries
    .sort((left, right) => {
      if (mode === "priority" && left.pinned !== right.pinned) return left.pinned ? -1 : 1;
      return sessionTime(right.session.updatedAt) - sessionTime(left.session.updatedAt);
    })
    .map(({ session }) => session);
}

function projectSessionGroupTime(group: ProjectSessionGroup): number {
  return Math.max(
    sessionTime(group.project?.updatedAt),
    ...group.sessions.map((session) => sessionTime(session.updatedAt)),
  );
}

function filterSessionGroups(groups: ProjectSessionGroup[], query: string): ProjectSessionGroup[] {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) return groups;

  return groups.flatMap((group) => {
    if (group.label.toLowerCase().includes(normalizedQuery)) return [group];
    const sessions = group.sessions.filter((session) =>
      `${session.title} ${session.harnessLabel} ${session.cwd}`
        .toLowerCase()
        .includes(normalizedQuery),
    );
    return sessions.length > 0 ? [{ ...group, sessions }] : [];
  });
}

function projectDisplayName(value: string): string {
  const normalized = value.trim().replace(/[\\/]+$/, "");
  const label = normalized.split(/[\\/]/).filter(Boolean).at(-1) ?? normalized;
  return label || "this project";
}

function abbreviateHomePath(value: string): string {
  return value
    .replace(/^\/Users\/[^/]+(?=\/|$)/, "~")
    .replace(/^\/home\/[^/]+(?=\/|$)/, "~")
    .replace(/^[A-Za-z]:\\Users\\[^\\]+(?=\\|$)/, "~");
}

function sameProjectPath(left?: string, right?: string): boolean {
  if (!left?.trim() || !right?.trim()) return false;
  const normalize = (value: string) => value.trim().replace(/[\\/]+$/, "");
  return normalize(left) === normalize(right);
}

function navigationEntryKey(session: DiscoveredSession | null): string {
  return session ? sessionCacheId(session) : "__new_session__";
}

function localSessionToDiscovered(session: SessionData): DiscoveredSession {
  const harness = HARNESSES.find((item) => item.id === session.harness);

  return {
    id: session.id,
    title: session.title || "Untitled",
    ...(session.projectId ? { projectId: session.projectId } : {}),
    cwd: session.cwd ?? "",
    pinned: session.pinned,
    updatedAt: session.updatedAt,
    harnessId: session.harness,
    harnessLabel: harness?.label ?? session.harness,
    source: "local",
  };
}

function groupDisplaySessions(
  sessions: DiscoveredSession[],
  mode: SessionGroupMode,
): SessionGroup[] {
  const grouped = new Map<string, SessionGroup>();

  for (const session of sortDisplaySessions(sessions)) {
    const project = session.cwd.trim();
    const groupId = mode === "harness" ? session.harnessId : project || "__no_project__";
    const groupLabel = mode === "harness" ? session.harnessLabel : project || "No project";
    const existing = grouped.get(groupId);
    if (existing) {
      existing.sessions.push(session);
    } else {
      grouped.set(groupId, { id: groupId, label: groupLabel, sessions: [session] });
    }
  }

  return [...grouped.values()];
}

function sortDisplaySessions(sessions: DiscoveredSession[]): DiscoveredSession[] {
  return [...sessions].sort(
    (a, b) =>
      Number(Boolean(b.pinned)) - Number(Boolean(a.pinned)) ||
      sessionTime(b.updatedAt) - sessionTime(a.updatedAt),
  );
}

function isPlaceholderSessionTitle(title: string): boolean {
  return ["", "new session", "untitled"].includes(title.trim().toLocaleLowerCase());
}

function sessionTime(value?: string): number {
  if (!value) return 0;
  const time = Date.parse(value);
  return Number.isFinite(time) ? time : 0;
}

function sessionMeta(session: DiscoveredSession, mode: SessionGroupMode): string {
  const date = formatSessionDate(session.updatedAt);
  if (mode === "project") {
    return `${session.harnessLabel} - ${date}`;
  }
  if (session.cwd.trim()) {
    return `${projectName(session.cwd)} - ${date}`;
  }
  return `${session.source === "local" ? "Local" : "ACP"} - ${date}`;
}

function harnessLabel(id: string): string {
  return HARNESSES.find((harness) => harness.id === id)?.label ?? id;
}

function workflowUsesProtectedHarness(config: SwarmConfig): boolean {
  if (backendRequiresProtectedRuntime(config.queen?.backend)) return true;
  return Object.values(config.nodes).some((node) => {
    if (node.kind === "agent") return backendRequiresProtectedRuntime(node.agent.backend);
    return node.kind === "swarm" ? workflowUsesProtectedHarness(node.swarm) : false;
  });
}

function backendRequiresProtectedRuntime(backend?: AgentBackend): boolean {
  if (backend?.type !== "custom") return false;
  const command = [backend.program, ...(backend.args ?? [])].join(" ");
  return (
    command.includes("@agentclientprotocol/claude-agent-acp") ||
    command.includes("@agentclientprotocol/codex-acp")
  );
}

function harnessEnvironmentLabel(
  status?: HarnessEnvironmentHarnessState,
  harnessId?: string,
): string {
  switch (status) {
    case "ready":
      return harnessId === "swarmx" ? "ready" : "launcher ready";
    case "needs_setup":
      return "needs setup";
    case "unsupported":
      return "unsupported";
    default:
      return "checking";
  }
}

function requirementStatusLabel(status: HarnessRequirementStatus): string {
  switch (status) {
    case "ready":
      return "ready";
    case "missing":
      return "missing";
    case "unsupported":
      return "unsupported";
    case "failed":
      return "failed";
  }
}

function containerRuntimeStatusLabel(status: ContainerRuntimeStatus): string {
  switch (status) {
    case "ready":
      return "ready";
    case "missing":
      return "missing";
    case "service_stopped":
      return "service stopped";
    case "unsupported":
      return "unsupported";
    case "failed":
      return "failed";
  }
}

function projectName(cwd: string): string {
  const parts = cwd.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] ?? cwd;
}

function formatSessionDate(value?: string): string {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function prefersReducedMotion(): boolean {
  return (
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

function cx(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}
