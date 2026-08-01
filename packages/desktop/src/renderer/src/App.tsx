import type {
  AgentBackend,
  BuiltinToolStylePreference,
  DesktopBuiltinToolSettings,
  DesktopComposerPreferences,
  DesktopComposerPreferenceUpdate,
  HarnessPermissionMode,
  HarnessPermissionPolicyLayer,
  SessionPermissionMode,
  SwarmConfig,
} from "@swarmx/core";
import {
  type NormalizedRenderEvent,
  type NormalizeMessageChunkOptions,
  normalizeMessageChunk,
  type RenderArtifactReference,
  type RenderProvenance,
} from "@swarmx/core/rendering";
import type {
  DoctorFixResult,
  DoctorReport,
  HarnessContainerRuntime,
  HarnessEnvironmentHarness,
  HarnessEnvironmentHarnessState,
  HarnessEnvironmentSetupResult,
  HarnessEnvironmentStatus,
  HarnessProtectionMode,
  HarnessProtectionSummary,
  HarnessRuntimeRequirement,
  HarnessVersionCheck,
} from "@swarmx/runtime";
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
  FolderOpen,
  Gauge,
  GitBranch,
  GitFork,
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
import type {
  DesktopMediaAttachment,
  DesktopPermissionStatus,
  DesktopUpdateState,
  ExtensionCapabilityInventory,
  ExtensionManagementState,
  ManualModelInput,
  DesktopMessageChunk as MessageChunk,
  DesktopMessageRenderMetadata as MessageRenderMetadata,
  ModelApiProtocol,
  ModelCatalogSummary,
  DesktopN8nImportResponse as N8nImportResponse,
  NewApiUsageAccount,
  DesktopPermissionLayerStatus as PermissionLayerStatus,
  ProviderKeyUsageSummary,
  ProviderUsageEntry,
  ProviderUsageMeter,
  ProviderUsageSnapshot,
  ProviderUsageTarget,
  DesktopSessionData as SessionData,
  DesktopSessionSummary as SessionSummary,
  DesktopSideChat as SideChat,
  DesktopSideChatParentState as SideChatParentState,
  UserProviderInput,
} from "../../shared/desktop-api.js";
import {
  AgentInteractionDialog,
  type AgentInteractionEvent,
  type AgentInteractionResponse,
} from "./agent-interaction-dialog.js";
import {
  AgentPicker,
  type ComposerModelOption,
  canonicalDefaultComposerModel,
  composerModelOptionId,
  groupComposerModels,
  preferredComposerModel,
  resolveComposerModelOptions,
} from "./agent-picker.js";
import { AppBrandIcon } from "./app-brand.js";
import { Composer } from "./composer.js";
import { ConversationHistory, type MessageEditState } from "./conversation-history.js";
import { mergeStreamingMessage, withRequestTiming } from "./conversation-messages.js";
import {
  type DoctorHarnessVersionState,
  DoctorPanel,
  type DoctorPanelMode,
} from "./doctor-panel.js";
import {
  type AgentCompositionPayload,
  extensionAgentComposition,
  formatSoftwareSummary,
  nativeAgentHostLabel,
  uniqueById,
} from "./extension-presentation.js";
import { ExtensionWorkspace } from "./extension-workspace.js";
import { HARNESSES, type HarnessOption } from "./harness-presentation.js";
import { RuntimeBottomPanel } from "./internal-terminal.js";
import { MediaPreviewPanel } from "./media-preview.js";
import { MessageAttachments } from "./message-attachments.js";
import { MessageContent, MessageCopyButton } from "./message-content.js";
import { type ActivityProfileSummary, ProfileWorkspace } from "./profile-workspace.js";
import {
  isDeepSeekProvider,
  isDeepSeekProviderUrl,
  isOpenCodeGoProviderUrl,
  ProviderBrandIcon,
  providerProtocolLabel,
} from "./provider-presentation.js";
import { api } from "./renderer-api.js";
import { RuntimeSettings } from "./runtime-settings.js";
import {
  abbreviateHomePath,
  buildSessionErrors,
  type DiscoveredSession,
  filterSessionGroups,
  flattenProjectSessions,
  type GroupedSessionsResult,
  harnessLabel,
  isPlaceholderSessionTitle,
  localSessionToDiscovered,
  mergeLocalSessionsIntoGroups,
  mergeProjectsIntoSessionGroups,
  navigationEntryKey,
  type ProjectData,
  type ProjectOrganizationMode,
  type ProjectPreviewState,
  type ProjectSessionGroup,
  type ProjectSortMode,
  preloadSessionCandidates,
  projectDisplayName,
  type SessionContextMenuState,
  type SessionGroupMode,
  sameProjectPath,
  sessionCacheId,
  sessionDetailKey,
  sessionMeta,
  sortProjectSessionGroups,
} from "./session-navigation.js";
import {
  ConversationPermissionPicker,
  CustomAgentsSettings,
  GeneralSettings,
  mergeProviderUsageSnapshot,
  PermissionsSettings,
  providerUsageTargetKey,
  type SettingsSection,
  SettingsSidebar,
  SettingsWorkspace,
} from "./settings-workspace.js";
import {
  capitalize,
  errorMessage,
  formatFullMessageTimestamp,
  formatMessageTimestamp,
  formatTimestamp,
  lines,
  projectName,
  slugId,
} from "./text-utils.js";
import { Badge, Button, cx } from "./ui-primitives.js";
import {
  type HarnessDescriptor,
  parseWorkflowJson,
  type WorkflowImportStatus,
  type WorkflowParseResult,
  WorkflowWorkspace,
} from "./workflow-workspace.js";
import {
  type BrowserBounds,
  type BrowserState,
  type WorkspaceDirectoryListing,
  type WorkspaceFilePreview,
  WorkspacePanel,
  type WorkspaceReviewSnapshot,
} from "./workspace-panel.js";

interface SelectedTranscriptContext {
  text: string;
  x: number;
  y: number;
}

type FocusedComposer = "main" | "side";

type ExtensionUiContributionSummary = ExtensionCapabilityInventory["uiContributions"][number];

type DesktopSlashCommand =
  | { kind: "doctor"; fix: boolean; harnessId?: string }
  | { kind: "setup"; fix: false; harnessId?: string }
  | { kind: "error"; message: string };

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

const LOCAL_SESSIONS_KEY = "sessions:local";
const GROUPED_SESSIONS_KEY = "sessions:grouped";
const ACTIVITY_PROFILE_KEY = "activity:profile";
const PROJECTS_KEY = "projects:local";
const EXTENSIONS_KEY = "extensions:inventory";
const EXTENSION_MANAGEMENT_KEY = "extensions:management";
const COMPOSER_PREFERENCES_KEY = "composer:preferences";
const HARNESS_ENVIRONMENT_KEY = "harness:environment";
const SESSION_DEDUPING_INTERVAL_MS = 10_000;
const PANEL_EXIT_MS = 240;
const INTERRUPTED_CONTINUE_PROMPT =
  "Continue the previous interrupted task. Do not assume unfinished tool calls completed. Verify the current state before retrying any side-effecting action.";
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

function parseSideChatCommand(value: string): string | null {
  const match = value.trim().match(/^\/(?:side|btw)(?:\s+([\s\S]*))?$/i);
  return match ? (match[1]?.trim() ?? "") : null;
}

function loadDiscoveredSessionDetail(session: DiscoveredSession): Promise<SessionData | null> {
  return api.loadDiscoveredSession(session) as Promise<SessionData | null>;
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
  const [attachments, setAttachments] = useState<DesktopMediaAttachment[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeRunStartedAt, setActiveRunStartedAt] = useState<number | null>(null);
  const [runState, setRunState] = useState<"idle" | "running" | "stopping">("idle");
  const [composerError, setComposerError] = useState<string | null>(null);
  const [messageEdit, setMessageEdit] = useState<MessageEditState | null>(null);
  const [sideChatState, setSideChatState] = useState<SideChatParentState | null>(null);
  const [sideChatError, setSideChatError] = useState<string | null>(null);
  const [sideMessageEdit, setSideMessageEdit] = useState<MessageEditState | null>(null);
  const [sidePaneWidth, setSidePaneWidth] = useState(40);
  const [selectedTranscriptContext, setSelectedTranscriptContext] =
    useState<SelectedTranscriptContext | null>(null);
  const [sideRunStartedAtById, setSideRunStartedAtById] = useState<Record<string, number>>({});
  const [forkingMessageIndex, setForkingMessageIndex] = useState<number | null>(null);
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
  const [projectExpandedById, setProjectExpandedById] = useState<Record<string, boolean>>({});
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
  const [previewAttachment, setPreviewAttachment] = useState<DesktopMediaAttachment | null>(null);
  const [rightPanelWidth, setRightPanelWidth] = useState<number | null>(null);
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
  const activeRightPanelKind = previewAttachment
    ? "media"
    : doctorPanelOpen
      ? "doctor"
      : rightPanelOpen
        ? "tools"
        : null;
  const [renderedRightPanelKind, setRenderedRightPanelKind] = useState<
    "doctor" | "tools" | "media"
  >(activeRightPanelKind ?? "tools");
  const displayedRightPanelKind = activeRightPanelKind ?? renderedRightPanelKind;
  const pinnedSummaryMounted = usePanelPresence(pinnedSummaryOpen);
  const rightPanelMounted = usePanelPresence(activeRightPanelKind !== null);
  const chatRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const previewReturnFocusRef = useRef<HTMLElement | null>(null);
  const sideComposerRef = useRef<HTMLTextAreaElement>(null);
  const sideChatScrollRef = useRef<HTMLDivElement>(null);
  const sideChatStateRef = useRef<SideChatParentState | null>(null);
  const focusedComposerRef = useRef<FocusedComposer>("main");
  const sidePaneRef = useRef<HTMLElement>(null);
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
  const composerPreferencesRestored = useRef(false);
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
  const activeSideChat =
    sideChatState?.chats.find((chat) => chat.id === sideChatState.activeSideChatId) ?? null;
  const sideChatPaneOpen = Boolean(
    currentSession &&
      sideChatState?.parentSessionId === currentSession.id &&
      sideChatState.chats.length > 0 &&
      !sideChatState.paneHidden,
  );
  const unreadSideChatCount =
    currentSession && sideChatState?.parentSessionId === currentSession.id
      ? sideChatState.chats.filter((chat) => chat.unread).length
      : 0;
  const currentParentHasRunningSideChat = Boolean(
    currentSession &&
      sideChatState?.parentSessionId === currentSession.id &&
      sideChatState.chats.some((chat) => chat.runState !== "idle"),
  );

  const commitSideChatState = useCallback((state: SideChatParentState | null) => {
    sideChatStateRef.current = state;
    setSideChatState(state);
  }, []);

  const updateSideChatState = useCallback(
    (update: (current: SideChatParentState) => SideChatParentState) => {
      setSideChatState((current) => {
        if (!current) return current;
        const next = update(current);
        sideChatStateRef.current = next;
        return next;
      });
    },
    [],
  );

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

  const openMediaPreview = useCallback((attachment: DesktopMediaAttachment) => {
    previewReturnFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setDoctorPanelOpen(false);
    setRightPanelOpen(false);
    setPreviewAttachment(attachment);
  }, []);

  const closeMediaPreview = useCallback(() => {
    const returnTarget = previewReturnFocusRef.current;
    setPreviewAttachment(null);
    returnTarget?.focus();
    window.requestAnimationFrame(() => {
      if (document.activeElement !== returnTarget) returnTarget?.focus();
    });
  }, []);

  const {
    data: sessions = [],
    error: localSessionsError,
    isLoading: localSessionsLoading,
    mutate: mutateLocalSessions,
  } = useSWR<SessionSummary[]>(
    LOCAL_SESSIONS_KEY,
    () => api.listSessions() as Promise<SessionSummary[]>,
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
  const {
    data: composerPreferences,
    isLoading: composerPreferencesLoading,
    mutate: mutateComposerPreferences,
  } = useSWR<DesktopComposerPreferences>(
    COMPOSER_PREFERENCES_KEY,
    () => api.getComposerPreferences(),
    { revalidateOnFocus: false, revalidateOnReconnect: false },
  );
  const persistComposerPreference = useCallback(
    async (update: DesktopComposerPreferenceUpdate) => {
      try {
        const preferences = await api.saveComposerPreference(update);
        await mutateComposerPreferences(preferences, false);
      } catch (error) {
        setComposerError(`Could not save the Composer Model preference: ${errorMessage(error)}`);
      }
    },
    [mutateComposerPreferences],
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
        setSelectedEffort(null);
        void persistComposerPreference({
          harnessId: selectedHarness,
          modelId: input.id.trim(),
        });
      } catch (error) {
        const message = errorMessage(error);
        setModelCatalogError(message);
        throw new Error(message);
      }
    },
    [mutateExtensionInventory, persistComposerPreference, selectedHarness],
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
    if (composerPreferencesRestored.current || composerPreferencesLoading || !extensionInventory) {
      return;
    }
    composerPreferencesRestored.current = true;
    const preferredHarness = availableHarnesses.find(
      (harness) => harness.id === composerPreferences?.lastHarnessId && harness.disabled !== true,
    );
    if (!preferredHarness || preferredHarness.id === selectedHarness) return;
    setSelectedHarness(preferredHarness.id);
    setSelectedExtensionAgentId(null);
    setSelectedModelId(null);
    setSelectedEffort(null);
  }, [
    availableHarnesses,
    composerPreferences?.lastHarnessId,
    composerPreferencesLoading,
    extensionInventory,
    selectedHarness,
  ]);
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
  const preferredModelSelection = composerPreferences?.selectionsByHarness[selectedHarness];
  const defaultModelOption = useMemo(
    () => canonicalDefaultComposerModel(availableModels),
    [availableModels],
  );
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
    if (!composerPreferencesRestored.current) return;
    if (selectedModelId && availableModels.some((model) => model.id === selectedModelId)) return;
    const preferredModel = preferredComposerModel(availableModels, preferredModelSelection);
    setSelectedModelId(preferredModel?.id ?? defaultModelOption?.id ?? null);
    setSelectedEffort(null);
  }, [availableModels, defaultModelOption?.id, preferredModelSelection, selectedModelId]);

  const availableEfforts = selectedModel?.reasoning?.supportedEfforts ?? [];
  const preferredEffort =
    preferredModelSelection && selectedModel?.modelId === preferredModelSelection.modelId
      ? preferredModelSelection.effort
      : undefined;
  useEffect(() => {
    if (availableEfforts.length === 0) {
      setSelectedEffort(null);
      return;
    }
    if (selectedEffort && availableEfforts.includes(selectedEffort)) return;
    setSelectedEffort(
      (preferredEffort && availableEfforts.includes(preferredEffort)
        ? preferredEffort
        : selectedModel?.reasoning?.defaultEffort) ??
        availableEfforts[0] ??
        null,
    );
  }, [availableEfforts, preferredEffort, selectedEffort, selectedModel?.reasoning?.defaultEffort]);
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
  const {
    data: builtinToolSettings,
    error: builtinToolSettingsError,
    isLoading: builtinToolSettingsLoading,
    mutate: mutateBuiltinToolSettings,
  } = useSWR<DesktopBuiltinToolSettings>(
    "builtinToolSettings:get",
    () => api.getBuiltinToolSettings(),
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

  const setVisibleSession = useCallback(
    (session: SessionData) => {
      setCurrentSession(session);
      if (!selectedSessionKey || selectedDiscoveredSession?.id !== session.id) return;
      void mutateSessionDetail(selectedSessionKey, session, {
        populateCache: true,
        revalidate: false,
      });
    },
    [mutateSessionDetail, selectedDiscoveredSession?.id, selectedSessionKey],
  );

  useEffect(() => {
    const parentSessionId = currentSession?.id;
    if (!parentSessionId || currentSession?.acpSessionId) {
      commitSideChatState(null);
      setSideChatError(null);
      setSideMessageEdit(null);
      return;
    }
    let disposed = false;
    void api
      .listSideChats(parentSessionId)
      .then((state) => {
        if (!disposed) commitSideChatState(state);
      })
      .catch((error) => {
        if (!disposed) setSideChatError(`Could not restore side chats: ${errorMessage(error)}`);
      });
    return () => {
      disposed = true;
    };
  }, [commitSideChatState, currentSession?.acpSessionId, currentSession?.id]);

  useLayoutEffect(() => {
    const scroll = sideChatScrollRef.current;
    if (!scroll || !activeSideChat) return;
    scroll.scrollTo({
      top: scroll.scrollHeight,
      behavior: prefersReducedMotion() ? "auto" : "smooth",
    });
  }, [activeSideChat]);

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
    setMessageEdit(null);
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
      setAttachments([]);
      setPreviewAttachment(null);
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

  const archiveSidebarSession = useCallback(async () => {
    const session = sessionContextMenu?.session;
    if (!session || session.source !== "local" || sessionActionPending) return;
    setSessionContextMenu(null);
    if (
      currentSession?.id === session.id &&
      (runState !== "idle" || currentParentHasRunningSideChat)
    ) {
      setSessionActionError("Stop the task before archiving it.");
      return;
    }
    setSessionActionPending(true);
    setSessionActionError(null);
    try {
      await api.archiveSession(session.id);
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
    currentParentHasRunningSideChat,
    mutateGroupedSessions,
    mutateLocalSessions,
    recordNavigationEntry,
    runState,
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

  const replaceSideChat = useCallback(
    (chat: SideChat | undefined) => {
      if (!chat) return;
      updateSideChatState((current) =>
        current.parentSessionId !== chat.parentSessionId
          ? current
          : {
              ...current,
              chats: current.chats.some((candidate) => candidate.id === chat.id)
                ? current.chats.map((candidate) => (candidate.id === chat.id ? chat : candidate))
                : [...current.chats, chat],
            },
      );
    },
    [updateSideChatState],
  );

  const createSideChat = useCallback(async (): Promise<SideChat | null> => {
    const parent = currentSession;
    if (!parent || parent.acpSessionId || parent.messages.length === 0) {
      setSideChatError("Start a local task before opening a side chat.");
      return null;
    }
    setSideChatError(null);
    setRightPanelOpen(false);
    setDoctorPanelOpen(false);
    try {
      const anchorParent = ((await api.loadSession(parent.id)) as SessionData | null) ?? parent;
      if (anchorParent.messages.length === 0) {
        throw new Error("The parent task has no persisted messages to anchor.");
      }
      const chat = (await api.createSideChat({
        parentSessionId: parent.id,
        throughMessageIndex: anchorParent.messages.length - 1,
        expectedMessages: anchorParent.messages,
      })) as SideChat;
      const state = await api.listSideChats(parent.id);
      commitSideChatState(state);
      setSideMessageEdit(null);
      window.requestAnimationFrame(() => sideComposerRef.current?.focus());
      return chat;
    } catch (error) {
      setSideChatError(`Could not create side chat: ${errorMessage(error)}`);
      return null;
    }
  }, [commitSideChatState, currentSession]);

  const showSideChats = useCallback(async () => {
    const parent = currentSession;
    if (!parent || parent.acpSessionId) {
      setSideChatError("Side chats are available for local tasks.");
      return;
    }
    if (
      !sideChatState ||
      sideChatState.parentSessionId !== parent.id ||
      sideChatState.chats.length === 0
    ) {
      await createSideChat();
      return;
    }
    setRightPanelOpen(false);
    setDoctorPanelOpen(false);
    try {
      commitSideChatState(await api.setSideChatHidden(parent.id, false));
      window.requestAnimationFrame(() => sideComposerRef.current?.focus());
    } catch (error) {
      setSideChatError(`Could not show side chats: ${errorMessage(error)}`);
    }
  }, [commitSideChatState, createSideChat, currentSession, sideChatState]);

  const changeSideChatDraft = useCallback(
    (draft: string) => {
      const parentSessionId = currentSession?.id;
      const sideChatId = sideChatState?.activeSideChatId;
      if (!parentSessionId || !sideChatId) return;
      updateSideChatState((current) => ({
        ...current,
        chats: current.chats.map((chat) => (chat.id === sideChatId ? { ...chat, draft } : chat)),
      }));
      void api
        .updateSideChat({ parentSessionId, sideChatId, draft })
        .catch((error) => setSideChatError(`Could not save side draft: ${errorMessage(error)}`));
    },
    [currentSession?.id, sideChatState?.activeSideChatId, updateSideChatState],
  );

  const addSideChatAttachments = useCallback(
    (paths: string[]) => {
      const parentSessionId = currentSession?.id;
      const chat = activeSideChat;
      if (!parentSessionId || !chat || paths.length === 0) return;
      const attachments = [...new Set([...chat.attachments, ...paths])];
      replaceSideChat({ ...chat, attachments });
      void api
        .updateSideChat({
          parentSessionId,
          sideChatId: chat.id,
          attachments,
        })
        .catch((error) =>
          setSideChatError(`Could not save side attachments: ${errorMessage(error)}`),
        );
    },
    [activeSideChat, currentSession?.id, replaceSideChat],
  );

  const sendSideChatMessage = useCallback(
    async (textOverride?: string, chatOverride?: SideChat, editMessageIndex?: number) => {
      const parent = currentSession;
      const chat = chatOverride ?? activeSideChat;
      const text = (textOverride ?? chat?.draft ?? "").trim();
      if (!parent || !chat || !text || chat.runState !== "idle") return;
      if (parseSideChatCommand(text) !== null) {
        setSideChatError("Nested side chats are not supported.");
        return;
      }
      if (manualCompositionNeedsModel) {
        setSideChatError(modelUnavailableDiagnostic);
        return;
      }
      if (selectedHarnessNeedsSetup) {
        await openDoctorPanel({ mode: "doctor", harnessId: activeRunHarnessId });
        return;
      }

      setSideChatError(null);
      setSideMessageEdit(null);
      const requestId = crypto.randomUUID();
      const userMessage: MessageChunk = {
        role: "user",
        content: text,
        kind: "message",
        createdAt: new Date().toISOString(),
      };
      const baseMessages =
        editMessageIndex === undefined
          ? [...chat.messages, userMessage]
          : [
              ...chat.messages.slice(0, editMessageIndex),
              { ...chat.messages[editMessageIndex], content: text },
            ];
      replaceSideChat({
        ...chat,
        messages: baseMessages,
        draft: "",
        attachments: [],
        contextChips: editMessageIndex === undefined ? [] : chat.contextChips,
        runState: "running",
        requestId,
        unread: false,
      });
      setSideRunStartedAtById((current) => ({ ...current, [chat.id]: Date.now() }));

      let streamedMessages: MessageChunk[] = [];
      const startedAt = new Date().toISOString();
      const unsubscribe = api.onSideChatChunk((event) => {
        if (
          event.parentSessionId !== parent.id ||
          event.sideChatId !== chat.id ||
          event.requestId !== requestId
        ) {
          return;
        }
        streamedMessages = mergeStreamingMessage(streamedMessages, event.chunk);
        updateSideChatState((current) => ({
          ...current,
          chats: current.chats.map((candidate) =>
            candidate.id === chat.id
              ? { ...candidate, messages: [...baseMessages, ...streamedMessages] }
              : candidate,
          ),
        }));
      });

      try {
        const agentComposition: AgentCompositionPayload = activeExtensionAgent
          ? extensionAgentComposition(activeExtensionAgent)
          : {
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
        const visibleState = sideChatStateRef.current;
        const result = await api.sendSideChatMessage({
          requestId,
          sessionId: parent.id,
          sideChatId: chat.id,
          sideChatVisible:
            visibleState?.paneHidden === false &&
            visibleState.activeSideChatId === chat.id &&
            currentSession?.id === parent.id,
          ...(editMessageIndex === undefined ? {} : { sideEditMessageIndex: editMessageIndex }),
          harnessId: activeExtensionAgent?.harnessId ?? selectedHarness,
          userText: text,
          agentComposition,
          ...(parent.cwd || composerWorkspaceRoot
            ? { cwd: parent.cwd || composerWorkspaceRoot }
            : {}),
        });
        const endedAt = new Date().toISOString();
        let completed = result.sideChat as SideChat | undefined;
        if (!completed) {
          completed = {
            ...chat,
            messages: [...baseMessages, ...withRequestTiming(streamedMessages, startedAt, endedAt)],
            runState: "idle",
            requestId: undefined,
          };
        }
        const currentState = sideChatStateRef.current;
        const shouldMarkUnread =
          currentState?.parentSessionId !== parent.id ||
          currentState.paneHidden ||
          currentState.activeSideChatId !== chat.id;
        if (shouldMarkUnread && !completed.unread) {
          const markedUnread = (await api.updateSideChat({
            parentSessionId: parent.id,
            sideChatId: chat.id,
            unread: true,
          })) as SideChat | undefined;
          completed = markedUnread ?? { ...completed, unread: true };
        }
        replaceSideChat(completed);
      } catch (error) {
        setSideChatError(`Side chat failed: ${errorMessage(error)}`);
        replaceSideChat({
          ...chat,
          messages: [
            ...baseMessages,
            ...streamedMessages,
            {
              role: "system",
              content: `Error: ${errorMessage(error)}`,
              kind: "message",
              createdAt: new Date().toISOString(),
            },
          ],
          runState: "idle",
          requestId: undefined,
        });
      } finally {
        unsubscribe();
        setSideRunStartedAtById((current) => {
          const next = { ...current };
          delete next[chat.id];
          return next;
        });
      }
    },
    [
      activeExtensionAgent,
      activeRunHarnessId,
      activeSideChat,
      composerWorkspaceRoot,
      currentSession,
      manualCompositionNeedsModel,
      modelUnavailableDiagnostic,
      openDoctorPanel,
      replaceSideChat,
      selectedEffort,
      selectedHarness,
      selectedHarnessNeedsSetup,
      selectedModel,
      updateSideChatState,
    ],
  );

  const activateSideChat = useCallback(
    async (sideChatId: string) => {
      if (!currentSession) return;
      setSideMessageEdit(null);
      try {
        commitSideChatState(await api.activateSideChat(currentSession.id, sideChatId));
        window.requestAnimationFrame(() => sideComposerRef.current?.focus());
      } catch (error) {
        setSideChatError(`Could not switch side chat: ${errorMessage(error)}`);
      }
    },
    [commitSideChatState, currentSession],
  );

  const hideSideChats = useCallback(async () => {
    if (!currentSession || !sideChatState) return;
    focusedComposerRef.current = "main";
    setSideMessageEdit(null);
    try {
      commitSideChatState(await api.setSideChatHidden(currentSession.id, true));
      window.requestAnimationFrame(() => composerRef.current?.focus());
    } catch (error) {
      setSideChatError(`Could not hide side chats: ${errorMessage(error)}`);
    }
  }, [commitSideChatState, currentSession, sideChatState]);

  const deleteActiveSideChat = useCallback(async () => {
    if (!currentSession || !activeSideChat || activeSideChat.runState !== "idle") return;
    try {
      commitSideChatState(await api.deleteSideChat(currentSession.id, activeSideChat.id));
      setSideMessageEdit(null);
    } catch (error) {
      setSideChatError(`Could not delete side chat: ${errorMessage(error)}`);
    }
  }, [activeSideChat, commitSideChatState, currentSession]);

  const stopSideChat = useCallback(async () => {
    if (!currentSession || !activeSideChat?.requestId || activeSideChat.runState !== "running") {
      return;
    }
    const requestId = activeSideChat.requestId;
    replaceSideChat({ ...activeSideChat, runState: "stopping" });
    try {
      const result = await api.cancelSideChat(currentSession.id, activeSideChat.id, requestId);
      if (!result.canceled) replaceSideChat({ ...activeSideChat, runState: "running" });
    } catch {
      replaceSideChat({ ...activeSideChat, runState: "running" });
    }
  }, [activeSideChat, currentSession, replaceSideChat]);

  const promoteActiveSideChat = useCallback(async () => {
    if (!currentSession || !activeSideChat || activeSideChat.runState !== "idle") return;
    setSideChatError(null);
    try {
      const promoted = (await api.promoteSideChat(
        currentSession.id,
        activeSideChat.id,
      )) as SessionData;
      const discovered = localSessionToDiscovered(promoted);
      await mutateSessionDetail(sessionDetailKey(discovered), promoted, {
        populateCache: true,
        revalidate: false,
      });
      recordNavigationEntry(discovered);
      setCurrentSession(promoted);
      await Promise.all([mutateLocalSessions(), mutateGroupedSessions()]);
    } catch (error) {
      setSideChatError(`Could not promote side chat: ${errorMessage(error)}`);
    }
  }, [
    activeSideChat,
    currentSession,
    mutateGroupedSessions,
    mutateLocalSessions,
    mutateSessionDetail,
    recordNavigationEntry,
  ]);

  const addSelectionToSideChat = useCallback(async () => {
    const selected = selectedTranscriptContext;
    const parent = currentSession;
    if (!selected || !parent) return;
    let chat = activeSideChat;
    if (!chat) chat = await createSideChat();
    if (!chat) return;
    try {
      const updated = (await api.addSideChatContext(parent.id, chat.id, selected.text)) as SideChat;
      replaceSideChat(updated);
      setSelectedTranscriptContext(null);
      window.getSelection()?.removeAllRanges();
      window.requestAnimationFrame(() => sideComposerRef.current?.focus());
    } catch (error) {
      setSideChatError(`Could not add selected context: ${errorMessage(error)}`);
    }
  }, [activeSideChat, createSideChat, currentSession, replaceSideChat, selectedTranscriptContext]);

  const captureTranscriptSelection = useCallback(() => {
    const selection = window.getSelection();
    const text = selection?.toString().trim() ?? "";
    if (!selection || !text || selection.rangeCount === 0) {
      setSelectedTranscriptContext(null);
      return;
    }
    const range = selection.getRangeAt(0);
    if (!chatRef.current?.contains(range.commonAncestorContainer)) {
      setSelectedTranscriptContext(null);
      return;
    }
    const rect = range.getBoundingClientRect();
    setSelectedTranscriptContext({
      text: text.slice(0, 8_000),
      x: Math.min(window.innerWidth - 170, Math.max(12, rect.left + rect.width / 2 - 70)),
      y: Math.max(12, rect.top - 40),
    });
  }, []);

  const continueInNewChat = useCallback(
    async (throughMessageIndex: number) => {
      const source = currentSession;
      if (!source || loading || forkingMessageIndex !== null || source.acpSessionId) return;

      setComposerError(null);
      setForkingMessageIndex(throughMessageIndex);
      try {
        const forked = (await api.forkSession({
          id: source.id,
          throughMessageIndex,
          expectedMessages: source.messages,
        })) as SessionData;
        const discovered = localSessionToDiscovered(forked);
        await mutateSessionDetail(sessionDetailKey(discovered), forked, {
          populateCache: true,
          revalidate: false,
        });
        recordNavigationEntry(discovered);
        setCurrentSession(forked);
        setInput("");
        await Promise.all([mutateLocalSessions(), mutateGroupedSessions()]);
        window.requestAnimationFrame(() => composerRef.current?.focus());
      } catch (error) {
        setComposerError(`Could not continue in a new chat: ${errorMessage(error)}`);
      } finally {
        setForkingMessageIndex(null);
      }
    },
    [
      currentSession,
      forkingMessageIndex,
      loading,
      mutateGroupedSessions,
      mutateLocalSessions,
      mutateSessionDetail,
      recordNavigationEntry,
    ],
  );

  const sendMessage = useCallback(
    async (
      textOverride?: string,
      editMessageIndex?: number,
      editExpectedMessages?: MessageChunk[],
      attachmentOverride?: DesktopMediaAttachment[],
    ) => {
      const text = (typeof textOverride === "string" ? textOverride : input).trim();
      const editingExistingMessage = editMessageIndex !== undefined;
      let requestAttachments = editingExistingMessage
        ? (currentSession?.messages[editMessageIndex]?.attachments ?? [])
        : (attachmentOverride ?? attachments);
      const sidePrompt =
        editingExistingMessage || requestAttachments.length > 0 ? null : parseSideChatCommand(text);
      if (sidePrompt !== null) {
        setInput("");
        const chat = await createSideChat();
        if (chat && sidePrompt) await sendSideChatMessage(sidePrompt, chat);
        return;
      }
      if ((!text && requestAttachments.length === 0) || loading) return;
      const slashCommand =
        editingExistingMessage || requestAttachments.length > 0
          ? null
          : parseDesktopSlashCommand(text);
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
      if (!editingExistingMessage) {
        setInput("");
        setAttachments([]);
      }
      setComposerError(null);
      if (editingExistingMessage) {
        setMessageEdit((current) =>
          current?.messageIndex === editMessageIndex ? { ...current, error: null } : current,
        );
      }
      setLoading(true);
      setActiveRunStartedAt(Date.now());
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
        activeRequestId.current === requestId &&
        stopRequested.current &&
        !requestDispatched.current;

      try {
        const userChunk: MessageChunk = {
          role: "user",
          content: text,
          kind: "message",
          createdAt: new Date().toISOString(),
          ...(requestAttachments.length > 0 ? { attachments: requestAttachments } : {}),
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

        let updatedMessages: MessageChunk[];
        if (editingExistingMessage) {
          const editedSession = (await api.editSessionUserMessage({
            id: session.id,
            messageIndex: editMessageIndex,
            expectedMessages: editExpectedMessages ?? session.messages,
            content: text,
          })) as SessionData;
          session = editedSession;
          sessionForError = editedSession;
          pendingSession = editedSession;
          updatedMessages = editedSession.messages;
          requestAttachments = editedSession.messages[editMessageIndex]?.attachments ?? [];
          setVisibleSession(editedSession);
          setMessageEdit(null);
        } else {
          updatedMessages = [...session.messages, userChunk];
          pendingSession = { ...session, messages: updatedMessages };
          setVisibleSession(pendingSession);
          await api.saveSession(pendingSession);
        }
        if (stoppedBeforeDispatch()) return;

        const sendParams: {
          requestId: string;
          sessionId?: string;
          harnessId: string;
          userText: string;
          agentComposition?: AgentCompositionPayload;
          swarmConfig?: SwarmConfig;
          cwd?: string;
          attachments?: DesktopMediaAttachment[];
        } = {
          requestId,
          sessionId: session.id,
          harnessId: activeExtensionAgent?.harnessId ?? selectedHarness,
          userText: text,
          ...(requestAttachments.length > 0 ? { attachments: requestAttachments } : {}),
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
          setVisibleSession(updated);
          requestAutomaticSessionTitle(
            updated,
            text || requestAttachments.map((attachment) => attachment.name).join(", "),
          );
        } else if (result.canceled) {
          const canceledMessages = requestStartedAt
            ? withRequestTiming(streamedMessages, requestStartedAt, requestEndedAt)
            : streamedMessages;
          const localUpdated = { ...session, messages: [...updatedMessages, ...canceledMessages] };
          const persisted = result.sessionPersisted ? await api.loadSession(session.id) : null;
          const updated = persisted ?? localUpdated;
          if (!persisted) await api.saveSession(updated);
          setVisibleSession(updated);
        } else if (result.error) {
          const workMessages = requestStartedAt
            ? withRequestTiming(streamedMessages, requestStartedAt, requestEndedAt)
            : streamedMessages;
          const failureMessages = Array.isArray(result.messages)
            ? (result.messages as MessageChunk[])
            : [
                ...workMessages,
                {
                  role: "system",
                  content: `Error: ${result.error}`,
                  kind: "message" as const,
                },
              ];
          const localUpdated = {
            ...session,
            messages: [...updatedMessages, ...failureMessages],
          };
          const persisted = result.sessionPersisted ? await api.loadSession(session.id) : null;
          const updated = persisted ?? localUpdated;
          if (!persisted) await api.saveSession(updated);
          setVisibleSession(updated);
        }

        await mutateLocalSessions();
      } catch (error) {
        if (activeRequestId.current !== requestId) return;
        const message = `Error: ${errorMessage(error)}`;
        if (editingExistingMessage && !pendingSession) {
          setMessageEdit((current) =>
            current?.messageIndex === editMessageIndex
              ? { ...current, error: `Could not edit message: ${errorMessage(error)}` }
              : current,
          );
          return;
        }
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
          setVisibleSession(updated);
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
          setActiveRunStartedAt(null);
          setRunState("idle");
        }
      }
    },
    [
      input,
      attachments,
      loading,
      createSideChat,
      sendSideChatMessage,
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
      setVisibleSession,
    ],
  );

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

  useEffect(() => {
    const routeKeyboard = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === ";") {
        event.preventDefault();
        void showSideChats();
        return;
      }
      if (event.key !== "Escape" || event.defaultPrevented) return;
      if (focusedComposerRef.current === "side" && activeSideChat?.runState === "running") {
        event.preventDefault();
        void stopSideChat();
        return;
      }
      if (focusedComposerRef.current === "main" && runState === "running") {
        event.preventDefault();
        void stopMessage();
      }
    };
    window.addEventListener("keydown", routeKeyboard);
    return () => window.removeEventListener("keydown", routeKeyboard);
  }, [activeSideChat?.runState, runState, showSideChats, stopMessage, stopSideChat]);

  const beginSidePaneResize = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const pane = sidePaneRef.current;
    const body = pane?.parentElement;
    if (!body) return;
    const update = (pointerEvent: PointerEvent) => {
      const bounds = body.getBoundingClientRect();
      if (bounds.width <= 0) return;
      const next = ((bounds.right - pointerEvent.clientX) / bounds.width) * 100;
      setSidePaneWidth(Math.max(34, Math.min(55, next)));
    };
    const finish = () => {
      window.removeEventListener("pointermove", update);
      window.removeEventListener("pointerup", finish);
    };
    window.addEventListener("pointermove", update);
    window.addEventListener("pointerup", finish, { once: true });
  }, []);

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
  const activeToolApproval =
    activeAgentInteraction?.kind === "tool_approval" ? activeAgentInteraction : null;

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
                  variant={sideChatPaneOpen ? "secondary" : "ghost"}
                  size="icon"
                  onClick={() => void (sideChatPaneOpen ? hideSideChats() : showSideChats())}
                  disabled={
                    !currentSession ||
                    Boolean(currentSession.acpSessionId) ||
                    currentSession.messages.length === 0
                  }
                  title={
                    sideChatPaneOpen ? "Hide side chats" : "Show side chats (Command/Ctrl + ;)"
                  }
                  aria-label={sideChatPaneOpen ? "Hide side chats" : "Show side chats"}
                  aria-pressed={sideChatPaneOpen}
                  className="side-chat-titlebar-button"
                >
                  <MessageSquarePlus data-icon aria-hidden="true" />
                  {unreadSideChatCount > 0 && (
                    <span className="side-chat-unread-badge">{unreadSideChatCount}</span>
                  )}
                </Button>
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
                  variant={activeRightPanelKind ? "secondary" : "ghost"}
                  size="icon"
                  onClick={() => {
                    if (previewAttachment) {
                      closeMediaPreview();
                      return;
                    }
                    if (doctorPanelOpen) {
                      setDoctorPanelOpen(false);
                      return;
                    }
                    if (!rightPanelOpen && sideChatPaneOpen) void hideSideChats();
                    setRightPanelOpen((open) => !open);
                  }}
                  title={activeRightPanelKind ? "Hide right panel" : "Show right panel"}
                  aria-label={activeRightPanelKind ? "Hide right panel" : "Show right panel"}
                  aria-pressed={Boolean(activeRightPanelKind)}
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
              <div className="brand-identity">
                <AppBrandIcon className="brand-mark" />
                <div className="brand-copy">
                  <div className="brand-title">{productConfig.name}</div>
                  {productConfig.subtitle && (
                    <div className="brand-subtitle">{productConfig.subtitle}</div>
                  )}
                </div>
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
                  const projectId = group.project?.id;
                  const expanded = projectId
                    ? sidebarQuery.trim().length > 0 ||
                      (projectExpandedById[projectId] ?? activeProjectId === projectId)
                    : true;
                  const ProjectFolderIcon = expanded ? FolderOpen : Folder;
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
                          <ProjectFolderIcon aria-hidden="true" />
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
                            aria-expanded={group.project ? expanded : undefined}
                            onClick={() => {
                              if (!projectId) return;
                              setProjectExpandedById((current) => ({
                                ...current,
                                [projectId]: !expanded,
                              }));
                            }}
                          >
                            <ProjectFolderIcon aria-hidden="true" />
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

      <main
        className={cx(
          "runtime",
          rightPanelMounted && "runtime--right-panel",
          sideChatPaneOpen && "runtime--side-chat",
        )}
        style={{ "--side-chat-width": `${sidePaneWidth}%` } as React.CSSProperties}
      >
        <div
          className={cx(
            "runtime__body",
            rightPanelMounted && "runtime__body--right-panel",
            sideChatPaneOpen && "runtime__body--side-chat",
          )}
        >
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
                  builtinTools={builtinToolSettings}
                  builtinToolsLoading={builtinToolSettingsLoading}
                  builtinToolsError={builtinToolSettingsError}
                  onSaveProfiles={async (profileAvailability) => {
                    await mutatePermissionStatus(
                      await api.savePermissionProfileAvailability(
                        profileAvailability,
                        permissionContext,
                      ),
                      false,
                    );
                  }}
                  onSaveBuiltinTools={async (settings) => {
                    await mutateBuiltinToolSettings(
                      await api.saveBuiltinToolSettings(settings),
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
                <div
                  ref={chatRef}
                  className="transcript-scroll"
                  onMouseUp={captureTranscriptSelection}
                  onKeyUp={captureTranscriptSelection}
                >
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
                        activeRunStartedAt={activeRunStartedAt}
                        messages={currentSession?.messages ?? []}
                        running={loading}
                        actionsDisabled={loading || forkingMessageIndex !== null}
                        messageEdit={messageEdit}
                        onBeginEdit={
                          acpHistoryReadOnly
                            ? undefined
                            : (messageIndex, content) => {
                                setComposerError(null);
                                setMessageEdit({
                                  messageIndex,
                                  draft: content,
                                  error: null,
                                  expectedMessages: currentSession?.messages ?? [],
                                });
                              }
                        }
                        onCancelEdit={() => setMessageEdit(null)}
                        onEditDraftChange={(draft) =>
                          setMessageEdit((current) =>
                            current ? { ...current, draft, error: null } : current,
                          )
                        }
                        onSubmitEdit={(messageIndex, content) =>
                          void sendMessage(content, messageIndex, messageEdit?.expectedMessages)
                        }
                        onContinue={
                          acpHistoryReadOnly
                            ? undefined
                            : () => void sendMessage(INTERRUPTED_CONTINUE_PROMPT)
                        }
                        onContinueInNewChat={
                          acpHistoryReadOnly
                            ? undefined
                            : (messageIndex) => void continueInNewChat(messageIndex)
                        }
                        onRetry={(userText, retryAttachments) =>
                          void sendMessage(userText, undefined, undefined, retryAttachments)
                        }
                        onPreviewAttachment={openMediaPreview}
                        onChangeModel={
                          activeWorkflowConfig || activeExtensionAgent
                            ? undefined
                            : () => {
                                setAgentPickerSection("model");
                                setPermissionPickerOpen(false);
                                setAgentPickerOpen(true);
                              }
                        }
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
              <div
                className="right-panel-resize"
                role="separator"
                tabIndex={0}
                aria-label="Resize right panel"
                aria-orientation="vertical"
                aria-valuenow={rightPanelWidth ?? undefined}
                onPointerDown={(event) => {
                  event.currentTarget.setPointerCapture(event.pointerId);
                }}
                onPointerMove={(event) => {
                  if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
                  const body = event.currentTarget.closest<HTMLElement>(".runtime__body");
                  if (!body) return;
                  const bounds = body.getBoundingClientRect();
                  setRightPanelWidth(
                    clampRightPanelWidth(bounds.width, bounds.right - event.clientX),
                  );
                }}
                onPointerUp={(event) => {
                  if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                    event.currentTarget.releasePointerCapture(event.pointerId);
                  }
                }}
                onKeyDown={(event) => {
                  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
                  event.preventDefault();
                  const body = event.currentTarget.closest<HTMLElement>(".runtime__body");
                  if (!body) return;
                  const bodyWidth = body.getBoundingClientRect().width;
                  const current = rightPanelWidth ?? Math.round(bodyWidth / 2);
                  const delta = event.key === "ArrowLeft" ? 24 : -24;
                  setRightPanelWidth(clampRightPanelWidth(bodyWidth, current + delta));
                }}
              />
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
              ) : displayedRightPanelKind === "media" && previewAttachment ? (
                <MediaPreviewPanel
                  api={api}
                  attachment={previewAttachment}
                  onClose={closeMediaPreview}
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
          {sideChatPaneOpen && activeSideChat && sideChatState && (
            <aside
              className="side-chat-pane"
              ref={sidePaneRef}
              aria-label="Side chats"
              style={{ width: `${sidePaneWidth}%` }}
            >
              <div
                className="side-chat-pane__resizer"
                role="separator"
                aria-label="Resize side chat"
                aria-orientation="vertical"
                aria-valuemin={34}
                aria-valuemax={55}
                aria-valuenow={Math.round(sidePaneWidth)}
                tabIndex={0}
                onPointerDown={beginSidePaneResize}
                onKeyDown={(event) => {
                  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
                  event.preventDefault();
                  setSidePaneWidth((width) =>
                    Math.max(34, Math.min(55, width + (event.key === "ArrowLeft" ? 2 : -2))),
                  );
                }}
              />
              <header className="side-chat-pane__header">
                <button
                  type="button"
                  className="side-chat-pane__back"
                  onClick={() => void hideSideChats()}
                >
                  <ArrowLeft aria-hidden="true" />
                  <span>Main chat</span>
                </button>
                <div className="side-chat-tabs" role="tablist" aria-label="Side chat tabs">
                  {sideChatState.chats.map((chat) => (
                    <button
                      type="button"
                      role="tab"
                      aria-selected={chat.id === activeSideChat.id}
                      className={cx("side-chat-tab", chat.id === activeSideChat.id && "is-active")}
                      key={chat.id}
                      onClick={() => void activateSideChat(chat.id)}
                      title={chat.title}
                    >
                      <span>{chat.title}</span>
                      {chat.runState !== "idle" && (
                        <Loader2 className="side-chat-tab__running" aria-label="Running" />
                      )}
                      {chat.unread && (
                        <span className="side-chat-tab__unread" aria-label="Unread" />
                      )}
                    </button>
                  ))}
                  <button
                    type="button"
                    className="side-chat-tabs__add"
                    onClick={() => void createSideChat()}
                    aria-label="New side chat"
                    title="New side chat"
                  >
                    <Plus aria-hidden="true" />
                  </button>
                </div>
                <div className="side-chat-pane__actions">
                  <button
                    type="button"
                    onClick={() => void promoteActiveSideChat()}
                    disabled={activeSideChat.runState !== "idle"}
                    aria-label="Promote side chat to task"
                    title="Continue in new chat / Promote to task"
                  >
                    <GitFork aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    onClick={() => void deleteActiveSideChat()}
                    disabled={activeSideChat.runState !== "idle"}
                    aria-label="Delete current side chat"
                    title="Delete current side chat"
                  >
                    <Trash2 aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    onClick={() => void hideSideChats()}
                    aria-label="Hide side chats"
                    title="Hide side chats"
                  >
                    <PanelRight aria-hidden="true" />
                  </button>
                </div>
              </header>
              <div className="side-chat-pane__boundary">
                <span>Transient fork</span>
                <span>Anchored after parent message {activeSideChat.anchor.messageIndex + 1}</span>
                <span>Read-only lane</span>
              </div>
              <div ref={sideChatScrollRef} className="side-chat-pane__transcript">
                {activeSideChat.messages.length === 0 ? (
                  <div className="side-chat-pane__empty">
                    <MessageCircle aria-hidden="true" />
                    <h2>Ask without derailing the task</h2>
                    <p>
                      This tab sees the parent transcript only up to its anchor. Its turns stay in
                      memory until you promote them.
                    </p>
                  </div>
                ) : (
                  <ConversationHistory
                    activeRunStartedAt={sideRunStartedAtById[activeSideChat.id] ?? null}
                    messages={activeSideChat.messages}
                    running={activeSideChat.runState !== "idle"}
                    actionsDisabled={activeSideChat.runState !== "idle"}
                    messageEdit={sideMessageEdit}
                    onBeginEdit={(messageIndex, content) => {
                      setSideChatError(null);
                      setSideMessageEdit({
                        messageIndex,
                        draft: content,
                        error: null,
                        expectedMessages: activeSideChat.messages,
                      });
                    }}
                    onCancelEdit={() => setSideMessageEdit(null)}
                    onEditDraftChange={(draft) =>
                      setSideMessageEdit((current) =>
                        current ? { ...current, draft, error: null } : current,
                      )
                    }
                    onSubmitEdit={(messageIndex, content) =>
                      void sendSideChatMessage(content, activeSideChat, messageIndex)
                    }
                    onRetry={(userText) => void sendSideChatMessage(userText, activeSideChat)}
                  />
                )}
              </div>
              <footer className="side-chat-pane__composer">
                {activeSideChat.contextChips.length > 0 && (
                  <div className="side-chat-context-chips" aria-label="Selected side chat context">
                    {activeSideChat.contextChips.map((chip) => (
                      <span key={chip.id} title={chip.text}>
                        <FileSearch aria-hidden="true" />
                        {chip.text}
                      </span>
                    ))}
                  </div>
                )}
                <Composer
                  textareaRef={sideComposerRef}
                  value={activeSideChat.draft}
                  onChange={changeSideChatDraft}
                  onFocus={() => {
                    focusedComposerRef.current = "side";
                  }}
                  onSubmit={() => sendSideChatMessage(undefined, activeSideChat)}
                  onStop={stopSideChat}
                  placeholder="Ask in side chat"
                  disabled={activeSideChat.runState !== "idle"}
                  running={activeSideChat.runState !== "idle"}
                  sendDisabled={
                    activeSideChat.runState === "stopping" ||
                    (activeSideChat.runState === "idle" && !activeSideChat.draft.trim())
                  }
                  error={sideChatError}
                  workspaceRoot={composerWorkspaceRoot}
                  mentionServers={composerMentionServers}
                  completeMention={api.lspComplete}
                  selectFilesAndFolders={api.selectFilesAndFolders}
                  onFilesSelected={addSideChatAttachments}
                  onContextError={(error) => setSideChatError(errorMessage(error))}
                >
                  <span className="side-chat-pane__mode">
                    <ShieldCheck aria-hidden="true" />
                    Read-only
                  </span>
                </Composer>
              </footer>
            </aside>
          )}
        </div>

        {!settingsSection &&
          (activeToolApproval || (!workflowPanelOpen && !activeUiContribution)) && (
            <footer
              className={cx("composer-dock", activeToolApproval && "composer-dock--approval")}
            >
              {activeToolApproval ? (
                <AgentInteractionDialog
                  key={activeToolApproval.interactionId}
                  interaction={activeToolApproval}
                  resolving={resolvingInteractionId === activeToolApproval.interactionId}
                  error={agentInteractionError}
                  onResolve={(response) =>
                    void resolveAgentInteraction(activeToolApproval, response)
                  }
                  onStop={() => void stopMessage()}
                />
              ) : (
                <Composer
                  textareaRef={composerRef}
                  value={input}
                  onChange={setInput}
                  onFocus={() => {
                    focusedComposerRef.current = "main";
                  }}
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
                      ((!input.trim() && attachments.length === 0) ||
                        manualCompositionNeedsModel ||
                        acpHistoryReadOnly))
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
                  selectMediaAttachments={api.selectMediaAttachments}
                  importMediaAttachments={api.importMediaAttachments}
                  attachments={attachments}
                  onAttachmentsChange={setAttachments}
                  onPreviewAttachment={openMediaPreview}
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
                      void persistComposerPreference({ harnessId });
                    }}
                    onModelChange={(modelOptionId) => {
                      const model = availableModels.find(
                        (candidate) => candidate.id === modelOptionId,
                      );
                      setSelectedModelId(modelOptionId);
                      setSelectedEffort(null);
                      if (model) {
                        void persistComposerPreference({
                          harnessId: selectedHarness,
                          modelId: model.modelId,
                          ...(model.modelSupplyId ? { modelSupplyId: model.modelSupplyId } : {}),
                        });
                      }
                    }}
                    onEffortChange={(effort) => {
                      setSelectedEffort(effort);
                      if (selectedModel) {
                        void persistComposerPreference({
                          harnessId: selectedHarness,
                          modelId: selectedModel.modelId,
                          ...(selectedModel.modelSupplyId
                            ? { modelSupplyId: selectedModel.modelSupplyId }
                            : {}),
                          effort,
                        });
                      }
                    }}
                    onRefreshModels={refreshModelCatalog}
                    onAddManualModel={addManualModel}
                    onRemoveManualModel={removeManualModel}
                  />
                </Composer>
              )}
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
      {selectedTranscriptContext
        ? createPortal(
            <button
              type="button"
              className="ask-in-side-chat-action"
              style={{
                left: selectedTranscriptContext.x,
                top: selectedTranscriptContext.y,
              }}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => void addSelectionToSideChat()}
            >
              <MessageSquarePlus aria-hidden="true" />
              Ask in side chat
            </button>,
            document.body,
          )
        : null}
      {activeAgentInteraction && activeAgentInteraction.kind !== "tool_approval"
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
                disabled={
                  sessionActionPending ||
                  (currentSession?.id === sessionContextMenu.session.id &&
                    (runState !== "idle" || currentParentHasRunningSideChat))
                }
                onClick={() => void archiveSidebarSession()}
                title={
                  currentSession?.id === sessionContextMenu.session.id &&
                  (runState !== "idle" || currentParentHasRunningSideChat)
                    ? "Stop the task before archiving it"
                    : undefined
                }
              >
                <Archive aria-hidden="true" />
                <span>Archive task</span>
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
        <AppBrandIcon className="empty-run__icon" />
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
  onArchive,
}: {
  title: string;
  harness: string;
  model: string;
  effort: string;
  status: string;
  messageCount: number;
  onClose: () => void;
  onArchive?: () => void;
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
      {onArchive && (
        <Button variant="secondary" size="sm" onClick={onArchive}>
          <Archive data-icon="inline-start" aria-hidden="true" />
          Archive session
        </Button>
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

function workflowUsesProtectedHarness(config: SwarmConfig): boolean {
  if (backendRequiresProtectedRuntime(config.queen?.backend)) return true;
  return Object.values(config.nodes).some((node) => {
    if (node.kind === "agent") return backendRequiresProtectedRuntime(node.agent.backend);
    return node.kind === "swarm" ? workflowUsesProtectedHarness(node.swarm as SwarmConfig) : false;
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

function prefersReducedMotion(): boolean {
  return (
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

function clampRightPanelWidth(containerWidth: number, desiredWidth: number): number {
  const maximum = Math.max(320, Math.min(containerWidth * 0.7, containerWidth - 320));
  return Math.round(Math.min(maximum, Math.max(320, desiredWidth)));
}
