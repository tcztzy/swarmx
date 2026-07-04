import type { AgentBackend, EdgeConfig, SwarmConfig, SwarmNodeConfig } from "@swarmx/core";
import {
  type NormalizeMessageChunkOptions,
  type NormalizedRenderEvent,
  type RenderArtifactReference,
  type RenderProvenance,
  normalizeMessageChunk,
} from "@swarmx/core/rendering";
import {
  Bot,
  ChevronLeft,
  ChevronRight,
  CircleCheck,
  Clock3,
  Code2,
  GitBranch,
  Hammer,
  Loader2,
  type LucideIcon,
  Maximize2,
  MessageSquarePlus,
  Minus,
  Package,
  Play,
  Plus,
  SendHorizontal,
  Sparkles,
  Terminal,
  Trash2,
  Upload,
  User,
  Workflow,
  Wrench,
  XCircle,
} from "lucide-react";
import type React from "react";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import { MessageContent } from "./message-content.js";

interface MessageChunk {
  role: string;
  content: string;
  kind: "message" | "thinking" | "tool_call" | "tool_result";
  agent?: string;
  render?: MessageRenderMetadata;
  swarmEvent?: string;
  toolName?: string;
}

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
  agentName: string;
  harness: string;
  model?: string;
  messages: MessageChunk[];
  createdAt: string;
  updatedAt: string;
}

type SessionGroupMode = "project" | "harness";

interface DiscoveredSession {
  id: string;
  title: string;
  cwd: string;
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

interface HarnessOption {
  id: string;
  label: string;
  icon: LucideIcon;
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
  warnings: Array<{ source: string; message: string }>;
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
  compatibleProviders?: string[];
  readOnly?: boolean;
  software?: { name?: string; version?: string };
}

interface ExtensionProviderSummary {
  id: string;
  label: string;
  kind: string;
  model?: string;
  readOnly?: boolean;
}

interface ExtensionAgentSummary {
  id: string;
  name: string;
  displayName?: string;
  selector?: string;
  harnessId?: string;
  providerProfileId?: string;
  model?: string;
  skills?: string[];
  mcpServers?: string[];
  tools?: string[];
  disallowedTools?: string[];
  permissionMode?: string;
  maxTurns?: number;
  memory?: string;
  effort?: string;
  background?: boolean;
  isolation?: string;
  color?: string;
  readOnly?: boolean;
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
  providerProfileId?: string;
  providerLabel?: string;
  providerKind?: string;
  model?: string;
  pluginIds?: string[];
  skills?: ExtensionAgentPlanCapabilitySummary[];
  mcpServers?: ExtensionAgentPlanCapabilitySummary[];
  context?: { mode: string; strategy: string; memory?: string };
  permissions?: { tools?: string; mcp?: string; shell?: string; mode?: string; summary?: string };
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
  languages?: string[];
  command?: string[];
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

interface AgentCompositionPayload {
  id: string;
  agentProfileId: string;
  host?: "local" | "server";
}

interface SwarmxAPI {
  sendMessage(params: {
    harnessId: string;
    userText: string;
    agentConfig?: unknown;
    agentComposition?: unknown;
    swarmConfig?: unknown;
    sessionId?: string;
  }): Promise<{ success: boolean; messages?: unknown; error?: string }>;
  createSession(params: {
    agentName: string;
    harness: string;
    model?: string;
  }): Promise<SessionData>;
  saveSession(session: SessionData): Promise<void>;
  loadSession(id: string): Promise<SessionData | null>;
  loadDiscoveredSession(session: DiscoveredSession): Promise<SessionData | null>;
  listSessions(): Promise<SessionData[]>;
  listGroupedSessions(params?: {
    mode?: "project" | "harness";
    cwd?: string;
    harnessIds?: string[];
  }): Promise<GroupedSessionsResult>;
  deleteSession(id: string): Promise<boolean>;
  appendMessages(params: { id: string; messages: unknown[] }): Promise<boolean>;
  importN8nWorkflow(source: string): Promise<N8nImportResponse>;
  listExtensions(): Promise<ExtensionCapabilityInventory>;
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
const EXTENSIONS_KEY = "extensions:inventory";
const SESSION_DEDUPING_INTERVAL_MS = 10_000;
const LOCAL_SESSION_PRELOAD_LIMIT = 24;

const HARNESSES: HarnessOption[] = [
  { id: "swarmx", label: "SwarmX", icon: Workflow },
  { id: "claude_code", label: "Claude Code", icon: Hammer },
  { id: "codex", label: "Codex", icon: Terminal },
  { id: "opencode", label: "OpenCode", icon: Code2 },
  { id: "hermes", label: "Hermes", icon: Sparkles },
  { id: "openclaw", label: "OpenClaw", icon: Wrench },
];

const ACP_SOFTWARE_VERSION = "0.22.0";
const DEFAULT_HARNESS_MCPS = [{ name: "filesystem", transport: "stdio", scope: "project" }];
const DEFAULT_HARNESS_SKILLS = ["test-driven-development", "backprop"];
const DEFAULT_PROJECT_FILES = ["AGENTS.md", "CLAUDE.md"];
const CODEX_ACP_ARGS = ["x", "--silent", `@agentclientprotocol/codex-acp@${ACP_SOFTWARE_VERSION}`];
const CLAUDE_CODE_ACP_ARGS = [
  "x",
  "--silent",
  `@agentclientprotocol/claude-agent-acp@${ACP_SOFTWARE_VERSION}`,
];
const DEFAULT_PRODUCT_CONFIG: Required<SwarmxDesktopProductConfig> = {
  name: "SwarmX",
  subtitle: "agent runtime",
};

const CODEX_ACP_BACKEND: AgentBackend = {
  type: "custom",
  program: "bun",
  args: CODEX_ACP_ARGS,
};

const CLAUDE_CODE_ACP_BACKEND: AgentBackend = {
  type: "custom",
  program: "bun",
  args: CLAUDE_CODE_ACP_ARGS,
};

function codexHarness(): HarnessDescriptor {
  return {
    software: {
      name: "codex-acp",
      version: ACP_SOFTWARE_VERSION,
      runner: "bun",
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
      version: ACP_SOFTWARE_VERSION,
      runner: "bun",
      command: CLAUDE_CODE_ACP_ARGS,
    },
    mcps: DEFAULT_HARNESS_MCPS,
    skills: DEFAULT_HARNESS_SKILLS,
    projectFiles: DEFAULT_PROJECT_FILES,
  };
}

const DEFAULT_WORKFLOW_CONFIG: SwarmConfig = {
  name: "research_review",
  description: "Route a request through ACP agents, each defined by model plus harness.",
  root: "triage_agent",
  nodes: {
    triage_agent: {
      kind: "agent",
      agent: {
        name: "triage_agent",
        description: "Codex ACP agent for classification and planning.",
        model: "gpt-4o-mini",
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
        model: "claude-sonnet-4-20250514",
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
        model: "gpt-4o",
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

export function createSwarmxDesktopApp(appProps: AppProps = {}): React.ComponentType {
  function SwarmxDesktopApp() {
    return <App {...appProps} />;
  }
  return SwarmxDesktopApp;
}

export function App({ product, uiComponentRegistry = {} }: AppProps = {}) {
  const productConfig = {
    ...DEFAULT_PRODUCT_CONFIG,
    ...product,
  };
  const [sessionGroupMode, setSessionGroupMode] = useState<SessionGroupMode>("harness");
  const [currentSession, setCurrentSession] = useState<SessionData | null>(null);
  const [selectedDiscoveredSession, setSelectedDiscoveredSession] =
    useState<DiscoveredSession | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedHarness, setSelectedHarness] = useState("swarmx");
  const [selectedExtensionAgentId, setSelectedExtensionAgentId] = useState<string | null>(null);
  const [activeUiContributionId, setActiveUiContributionId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [workflowPanelOpen, setWorkflowPanelOpen] = useState(false);
  const [extensionPanelOpen, setExtensionPanelOpen] = useState(false);
  const [workflowEnabled, setWorkflowEnabled] = useState(false);
  const [workflowJson, setWorkflowJson] = useState(DEFAULT_WORKFLOW_JSON);
  const [workflowImportStatus, setWorkflowImportStatus] = useState<WorkflowImportStatus | null>(
    null,
  );
  const chatRef = useRef<HTMLDivElement>(null);
  const preloadedSessionKeys = useRef(new Set<string>());
  const scrollStateRef = useRef<{ sessionId: string | null; messageCount: number }>({
    sessionId: null,
    messageCount: 0,
  });
  const { mutate: mutateSessionDetail } = useSWRConfig();
  const messageCount = currentSession?.messages.length ?? 0;

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
  } = useSWR<GroupedSessionsResult>(
    GROUPED_SESSIONS_KEY,
    () => api.listGroupedSessions({ mode: "harness" }) as Promise<GroupedSessionsResult>,
    {
      dedupingInterval: SESSION_DEDUPING_INTERVAL_MS,
      keepPreviousData: true,
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );
  const {
    data: extensionInventory,
    error: extensionInventoryError,
    isLoading: extensionInventoryLoading,
  } = useSWR<ExtensionCapabilityInventory>(
    EXTENSIONS_KEY,
    () => api.listExtensions() as Promise<ExtensionCapabilityInventory>,
    {
      dedupingInterval: SESSION_DEDUPING_INTERVAL_MS,
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );

  const activeHarness = useMemo(
    () => HARNESSES.find((harness) => harness.id === selectedHarness) ?? HARNESSES[0],
    [selectedHarness],
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
  const activeExtensionAgent = activeWorkflowConfig ? null : selectedExtensionAgent;
  const sessionGroups = groupedSessions?.groups ?? [];
  const sessionErrors = useMemo(
    () =>
      buildSessionErrors(groupedSessions?.errors ?? [], localSessionsError, groupedSessionsError),
    [groupedSessions?.errors, localSessionsError, groupedSessionsError],
  );
  const sessionsLoading =
    (localSessionsLoading && sessions.length === 0) || (groupedSessionsLoading && !groupedSessions);
  const displayGroups = useMemo(
    () => mergeLocalSessionsIntoGroups(sessionGroups, sessions, sessionGroupMode),
    [sessionGroups, sessions, sessionGroupMode],
  );
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
  const runSubtitle = activeUiContribution
    ? `${activeUiContribution.placement} contribution${
        activeUiContribution.sourcePluginId ? ` via ${activeUiContribution.sourcePluginId}` : ""
      }`
    : currentSession
      ? `${currentSession.agentName} on ${harnessLabel(currentSession.harness)}`
      : activeWorkflowConfig
        ? `${activeWorkflowConfig.name} workflow ready`
        : activeExtensionAgent
          ? `${activeExtensionAgent.name} on ${activeExtensionAgent.harnessId ?? "extension harness"}`
          : `${activeHarness.label} ready`;

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
    if (!selectedSessionData) return;
    setCurrentSession(selectedSessionData);
    setSelectedHarness(selectedSessionData.harness);
    setSelectedExtensionAgentId(null);
    setActiveUiContributionId(null);
  }, [selectedSessionData]);

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

  const newSession = useCallback(async () => {
    const session = await api.createSession({
      agentName: activeWorkflowConfig?.name ?? activeExtensionAgent?.name ?? "agent",
      harness: activeExtensionAgent?.harnessId ?? selectedHarness,
    });
    await api.saveSession(session);
    await mutateLocalSessions();
    setCurrentSession(session);
  }, [activeExtensionAgent, activeWorkflowConfig, mutateLocalSessions, selectedHarness]);

  const selectSession = useCallback((session: DiscoveredSession) => {
    setActiveUiContributionId(null);
    setSelectedDiscoveredSession(session);
  }, []);

  const selectExtensionAgentForRun = useCallback((agentId: string) => {
    setSelectedExtensionAgentId(agentId);
    setWorkflowEnabled(false);
    setWorkflowPanelOpen(false);
    setExtensionPanelOpen(false);
    setActiveUiContributionId(null);
  }, []);

  const deleteCurrentSession = useCallback(async () => {
    if (!currentSession) return;
    await api.deleteSession(currentSession.id);
    await mutateLocalSessions();
    setCurrentSession(null);
  }, [currentSession, mutateLocalSessions]);

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
    setInput("");
    setLoading(true);

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
        })) as SessionData;
        await api.saveSession(session);
      }

      const updatedMessages = [...session.messages, userChunk];
      setCurrentSession({ ...session, messages: updatedMessages });

      const sendParams: {
        harnessId: string;
        userText: string;
        agentComposition?: AgentCompositionPayload;
        swarmConfig?: SwarmConfig;
      } = {
        harnessId: activeExtensionAgent?.harnessId ?? selectedHarness,
        userText: text,
      };
      if (activeWorkflowConfig) {
        sendParams.swarmConfig = activeWorkflowConfig;
      } else if (activeExtensionAgent) {
        sendParams.agentComposition = extensionAgentComposition(activeExtensionAgent);
      }

      const result = await api.sendMessage(sendParams);

      if (result.success && result.messages) {
        const responseMessages = result.messages as MessageChunk[];
        const updated = { ...session, messages: [...updatedMessages, ...responseMessages] };
        await api.saveSession(updated);
        setCurrentSession(updated);
      } else if (result.error) {
        const errorMsg: MessageChunk = {
          role: "system",
          content: `Error: ${result.error}`,
          kind: "message",
        };
        const updated = {
          ...session,
          messages: [...updatedMessages, errorMsg],
        };
        await api.saveSession(updated);
        setCurrentSession(updated);
      }

      await mutateLocalSessions();
    } finally {
      setLoading(false);
    }
  }, [
    input,
    loading,
    currentSession,
    selectedHarness,
    activeWorkflowConfig,
    activeExtensionAgent,
    mutateLocalSessions,
  ]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        void sendMessage();
      }
    },
    [sendMessage],
  );

  return (
    <div
      className={cx(
        "app-shell",
        !sidebarOpen && "app-shell--collapsed",
        workflowPanelOpen && "app-shell--workflow",
      )}
    >
      <aside className="sidebar" aria-label="Sessions">
        <div className="sidebar__brand">
          <div className="brand-mark">
            <Workflow aria-hidden="true" />
          </div>
          <div className="brand-copy">
            <div className="brand-title">{productConfig.name}</div>
            <div className="brand-subtitle">{productConfig.subtitle}</div>
          </div>
        </div>

        <div className="sidebar__controls">
          <label className="select-shell">
            <activeHarness.icon aria-hidden="true" />
            <select
              value={selectedHarness}
              onChange={(e) => setSelectedHarness(e.target.value)}
              className="select-control"
              aria-label="Harness"
            >
              {HARNESSES.map((harness) => (
                <option key={harness.id} value={harness.id}>
                  {harness.label}
                </option>
              ))}
            </select>
          </label>
          <Button onClick={newSession} size="sm">
            <MessageSquarePlus data-icon="inline-start" aria-hidden="true" />
            New
          </Button>
        </div>

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
                  setExtensionPanelOpen(false);
                }}
                aria-label={`Open ${contribution.name}`}
              >
                <Package aria-hidden="true" />
                <span>{contribution.name}</span>
              </button>
            ))}
          </nav>
        )}

        <div className="segmented-control" role="tablist" aria-label="Session grouping">
          <button
            type="button"
            role="tab"
            aria-selected={sessionGroupMode === "harness"}
            onClick={() => setSessionGroupMode("harness")}
            className={cx("segmented-control__item", sessionGroupMode === "harness" && "is-active")}
          >
            Harness
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={sessionGroupMode === "project"}
            onClick={() => setSessionGroupMode("project")}
            className={cx("segmented-control__item", sessionGroupMode === "project" && "is-active")}
          >
            Project
          </button>
        </div>

        <div className="session-scroll">
          {sessionsLoading && <div className="session-status">Loading sessions</div>}
          {!sessionsLoading && displayGroups.length === 0 && (
            <div className="session-status">No sessions</div>
          )}
          {displayGroups.map((group) => (
            <section key={group.id} className="session-group" aria-label={group.label}>
              <div className="session-group__header">
                <span>{group.label}</span>
                <span>{group.sessions.length}</span>
              </div>
              <div className="session-group__items">
                {group.sessions.map((session) => {
                  const isLocal = session.source === "local";
                  const isActive =
                    currentSession?.id === session.id &&
                    currentSession.harness === session.harnessId;
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
                      onClick={() => {
                        selectSession(session);
                      }}
                      className={cx(
                        "session-item",
                        isActive && "is-active",
                        isPending && "is-loading",
                      )}
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
                        <span className="session-item__meta">
                          {sessionMeta(session, sessionGroupMode)}
                        </span>
                      </span>
                    </button>
                  );
                })}
              </div>
            </section>
          ))}
          {visibleSessionErrors.map((error) => (
            <div key={error.harnessId} className="session-error">
              <XCircle aria-hidden="true" />
              <span>
                {error.harnessLabel}: {error.message}
              </span>
            </div>
          ))}
        </div>
      </aside>

      <main className="runtime">
        <header className="runtime__header">
          <div className="runtime__titlebar">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              title={sidebarOpen ? "Collapse sidebar" : "Open sidebar"}
              aria-label={sidebarOpen ? "Collapse sidebar" : "Open sidebar"}
            >
              {sidebarOpen ? (
                <ChevronLeft data-icon aria-hidden="true" />
              ) : (
                <ChevronRight data-icon aria-hidden="true" />
              )}
            </Button>
            <div className="runtime__title">
              <h1>{runTitle}</h1>
              <p>{runSubtitle}</p>
            </div>
          </div>

          <div className="runtime__actions">
            <Button
              variant={workflowPanelOpen ? "secondary" : "ghost"}
              size="sm"
              onClick={() => {
                setWorkflowPanelOpen((open) => !open);
                setExtensionPanelOpen(false);
                setActiveUiContributionId(null);
              }}
              aria-pressed={workflowPanelOpen}
            >
              <Workflow data-icon="inline-start" aria-hidden="true" />
              Workflow
            </Button>
            <Button
              variant={extensionPanelOpen ? "secondary" : "ghost"}
              size="sm"
              onClick={() => {
                setExtensionPanelOpen((open) => !open);
                setWorkflowPanelOpen(false);
                setActiveUiContributionId(null);
              }}
              aria-pressed={extensionPanelOpen}
            >
              <Package data-icon="inline-start" aria-hidden="true" />
              Extensions
            </Button>
            <Badge tone={loading || selectedSessionLoading ? "active" : "neutral"}>
              {loading || selectedSessionLoading ? (
                <Loader2 data-icon="inline-start" aria-hidden="true" />
              ) : (
                <Play data-icon="inline-start" aria-hidden="true" />
              )}
              {selectedSessionLoading ? "Loading" : loading ? "Running" : "Ready"}
            </Badge>
            <Badge tone="neutral">{messageCount} events</Badge>
            <Badge tone={workflowEnabled && !activeWorkflowConfig ? "danger" : "neutral"}>
              {workflowBadgeLabel}
            </Badge>
            {visibleSessionErrors.length > 0 && (
              <Badge tone="danger">{visibleSessionErrors.length} alerts</Badge>
            )}
            {currentSession && (
              <Button variant="ghost" size="icon" onClick={deleteCurrentSession} title="Delete run">
                <Trash2 data-icon aria-hidden="true" />
              </Button>
            )}
          </div>
        </header>

        <div className="runtime__body">
          {activeUiContribution && ActiveUiContributionComponent ? (
            <GuiContributionWorkspace
              contribution={activeUiContribution}
              inventory={extensionInventory}
              component={ActiveUiContributionComponent}
              onSelectAgent={selectExtensionAgentForRun}
            />
          ) : extensionPanelOpen ? (
            <ExtensionWorkspace
              inventory={extensionInventory}
              loading={extensionInventoryLoading}
              error={extensionInventoryError}
              selectedAgentId={selectedExtensionAgentId}
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
              <div className="transcript">
                {!currentSession || currentSession.messages.length === 0 ? (
                  <EmptyRun activeHarness={activeHarness} onStart={newSession} />
                ) : (
                  currentSession.messages.map((msg) => <RunEvent key={messageKey(msg)} msg={msg} />)
                )}
                {loading && (
                  <div className="run-event run-event--thinking">
                    <div className="run-event__rail">
                      <Loader2 aria-hidden="true" />
                    </div>
                    <div className="run-event__card">
                      <div className="run-event__header">
                        <span>agent</span>
                        <span>thinking</span>
                      </div>
                      <div className="run-event__content">Thinking</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {!workflowPanelOpen && !extensionPanelOpen && !activeUiContribution && (
          <footer className="composer-dock">
            <div className="composer">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={`Message ${activeExtensionAgent?.name ?? activeHarness.label}`}
                className="composer__textarea"
                rows={2}
                disabled={loading}
              />
              <div className="composer__footer">
                <div className="composer__meta">
                  {activeWorkflowConfig ? (
                    <Workflow aria-hidden="true" />
                  ) : activeExtensionAgent ? (
                    <Bot aria-hidden="true" />
                  ) : (
                    <activeHarness.icon aria-hidden="true" />
                  )}
                  <span>
                    {activeWorkflowConfig
                      ? activeWorkflowConfig.name
                      : (activeExtensionAgent?.name ?? activeHarness.label)}
                  </span>
                </div>
                <Button onClick={sendMessage} disabled={loading || !input.trim()}>
                  {loading ? (
                    <Loader2 data-icon="inline-start" aria-hidden="true" />
                  ) : (
                    <SendHorizontal data-icon="inline-start" aria-hidden="true" />
                  )}
                  Send
                </Button>
              </div>
            </div>
          </footer>
        )}
      </main>
    </div>
  );
}

function ExtensionWorkspace({
  inventory,
  loading,
  error,
  selectedAgentId,
  onSelectAgent,
}: {
  inventory?: ExtensionCapabilityInventory;
  loading: boolean;
  error: unknown;
  selectedAgentId: string | null;
  onSelectAgent: (agentId: string) => void;
}) {
  const bundles = inventory?.bundles ?? [];
  const harnesses = inventory?.harnesses ?? [];
  const agents = inventory?.agents ?? [];
  const providers = inventory?.providers ?? [];
  const skills = inventory?.skills ?? [];
  const mcpServers = inventory?.mcpServers ?? [];
  const appConnectors = inventory?.appConnectors ?? [];
  const uiContributions = inventory?.uiContributions ?? [];
  const agentPlans = inventory?.agentPlans ?? [];
  const pluginComponents = extensionComponentRows(inventory);
  const marketplaceSources = inventory?.marketplaceSources ?? [];
  const pluginCatalog = inventory?.pluginCatalog ?? [];
  const warnings = inventory?.warnings ?? [];
  const planByAgentId = useMemo(
    () => new Map(agentPlans.map((plan) => [plan.agentProfileId ?? plan.agentId, plan] as const)),
    [agentPlans],
  );

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
        </div>
      </div>

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
              {pluginCatalog.map((entry) => (
                <li key={entry.id} className="extension-item">
                  <div className="extension-item__main">
                    <strong>{entry.name}</strong>
                    <span>{entry.id}</span>
                  </div>
                  <div className="extension-item__meta">
                    {entry.version && <span>{entry.version}</span>}
                    <span>{entry.installState ?? "available"}</span>
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
              ))}
            </ul>
          )}
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
                  {(harness.compatibleProviders ?? []).map((provider) => (
                    <span key={`${harness.id}:${provider}`}>{provider}</span>
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
                      <span>
                        {plan?.model ?? agent.model ?? agent.providerProfileId ?? "no model"}
                      </span>
                      {plan?.providerProfileId && <span>{plan.providerProfileId}</span>}
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
                    {provider.model && <span>{provider.model}</span>}
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
  activeHarness,
  onStart,
}: { activeHarness: HarnessOption; onStart: () => void }) {
  return (
    <div className="empty-run">
      <div className="empty-run__mark">
        <activeHarness.icon aria-hidden="true" />
      </div>
      <div className="empty-run__copy">
        <h2>Start a SwarmX run</h2>
        <p>{activeHarness.label} is selected.</p>
      </div>
      <div className="empty-run__actions">
        <Button onClick={onStart}>
          <MessageSquarePlus data-icon="inline-start" aria-hidden="true" />
          New run
        </Button>
      </div>
    </div>
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
                  <span>Model {node.model ?? "default"}</span>
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

function RunEvent({ msg }: { msg: MessageChunk }) {
  const renderEvent = normalizeMessageChunk(msg, normalizeOptionsFromMessage(msg));
  const { icon: Icon, label, tone, meta } = messagePresentation(msg);
  const content = renderEventContent(msg, renderEvent);
  const showTraceCard = isTraceCardEvent(renderEvent);

  return (
    <article
      className={cx("run-event", `run-event--${tone}`)}
      data-render-event-id={renderEvent.eventId}
      data-render-kind={renderEvent.kind}
      data-render-status={renderEvent.status}
    >
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
    </article>
  );
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
  const payload = event.output ?? event.input;
  return isRecord(payload) ? payload : undefined;
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
    return { icon: Terminal, label: msg.agent ?? "tool", tone: "tool", meta: "call" };
  }
  if (msg.kind === "tool_result") {
    return { icon: Code2, label: msg.agent ?? "tool", tone: "tool", meta: "result" };
  }
  return { icon: Bot, label: msg.agent ?? "assistant", tone: "assistant", meta: "message" };
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
      chips: [...(item.languages ?? []), ...(item.command ? [item.command.join(" ")] : [])],
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

function localSessionToDiscovered(session: SessionData): DiscoveredSession {
  const harness = HARNESSES.find((item) => item.id === session.harness);

  return {
    id: session.id,
    title: session.title || "Untitled",
    cwd: "",
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
  return [...sessions].sort((a, b) => sessionTime(b.updatedAt) - sessionTime(a.updatedAt));
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
