import type {
  AgentBackend,
  DesktopBuiltinToolSettings,
  DesktopComposerPreferences,
  DesktopComposerPreferenceUpdate,
  SessionPermissionMode,
  SwarmConfig,
} from "@swarmx/core";
import { getHarness } from "@swarmx/core/harness";
import type { GlobalMemoryState } from "@swarmx/core/personal-memory";
import type {
  DoctorFixResult,
  DoctorReport,
  HarnessEnvironmentHarnessState,
  HarnessEnvironmentStatus,
} from "@swarmx/runtime";
import {
  Archive,
  ArrowLeft,
  ArrowRight,
  Bot,
  Bug,
  Check,
  Clock3,
  Download,
  FileSearch,
  Folder,
  FolderOpen,
  GitBranch,
  GitFork,
  Hammer,
  Loader2,
  type LucideIcon,
  MessageCircle,
  MessageSquarePlus,
  MoreHorizontal,
  Package,
  PanelBottom,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRight,
  Pencil,
  Pin,
  Plus,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  SquarePen,
  Telescope,
  Trash2,
  Workflow,
  X,
  XCircle,
} from "lucide-react";
import type React from "react";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import useSWR, { useSWRConfig } from "swr";
import type {
  DesktopMediaAttachment,
  DesktopPermissionStatus,
  DesktopTaskRuntimeListResult,
  DesktopUpdateState,
  ExtensionCapabilityInventory,
  ExtensionManagementState,
  ManualModelInput,
  DesktopMessageChunk as MessageChunk,
  DesktopN8nImportResponse as N8nImportResponse,
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
} from "./extension-presentation.js";
import { ExtensionWorkspace } from "./extension-workspace.js";
import { HARNESSES, type HarnessOption } from "./harness-presentation.js";
import { RuntimeBottomPanel } from "./internal-terminal.js";
import { MediaPreviewPanel } from "./media-preview.js";
import { type ActivityProfileSummary, ProfileWorkspace } from "./profile-workspace.js";
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
  type ProjectSortMode,
  preloadSessionCandidates,
  projectDisplayName,
  RECENTS_GROUP_ID,
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
  PersonalMemorySettings,
  providerUsageTargetKey,
  type SettingsSection,
  SettingsSidebar,
  SettingsWorkspace,
} from "./settings-workspace.js";
import { errorMessage } from "./text-utils.js";
import { Badge, Button, cx } from "./ui-primitives.js";
import {
  type HarnessDescriptor,
  parseWorkflowJson,
  type WorkflowImportStatus,
  WorkflowWorkspace,
} from "./workflow-workspace.js";
import { WorkspacePanel } from "./workspace-panel.js";

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
const PERSONAL_MEMORY_KEY = "settings:personal-memory";
const TASK_RUNTIME_KEY = "runtime:work-items";
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
const TASK_RUNTIME_REFRESH_INTERVAL_MS = 1_000;
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

const EMPTY_RUN_SUGGESTION_CLASS = {
  blue: "is-blue",
  violet: "is-violet",
  green: "is-green",
  orange: "is-orange",
} satisfies Record<(typeof EMPTY_RUN_SUGGESTIONS)[number]["tone"], string>;

const DEFAULT_HARNESS_MCPS = [{ name: "filesystem", transport: "stdio", scope: "project" }];
const DEFAULT_HARNESS_SKILLS = ["test-driven-development", "backprop"];
const DEFAULT_PROJECT_FILES = ["AGENTS.md", "CLAUDE.md"];
const DEFAULT_PRODUCT_CONFIG: Required<Pick<SwarmxDesktopProductConfig, "name">> = {
  name: "SwarmX",
};

function defaultWorkflowHarness(harnessId: "claude_code" | "codex"): {
  backend: AgentBackend;
  descriptor: HarnessDescriptor;
} {
  const harness = getHarness(harnessId);
  if (!harness) throw new Error(`Missing built-in Harness "${harnessId}".`);
  return {
    backend: harness.backend,
    descriptor: {
      software: harness.software,
      mcps: DEFAULT_HARNESS_MCPS,
      skills: DEFAULT_HARNESS_SKILLS,
      projectFiles: DEFAULT_PROJECT_FILES,
    },
  };
}

const DEFAULT_CODEX_HARNESS = defaultWorkflowHarness("codex");
const DEFAULT_CLAUDE_HARNESS = defaultWorkflowHarness("claude_code");

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
        backend: DEFAULT_CODEX_HARNESS.backend,
        parameters: { harness: DEFAULT_CODEX_HARNESS.descriptor },
        instructions: "Identify the user's goal, constraints, and required evidence.",
      },
    },
    researcher_agent: {
      kind: "agent",
      agent: {
        name: "researcher_agent",
        description: "Claude Code ACP agent for repository research.",
        backend: DEFAULT_CLAUDE_HARNESS.backend,
        parameters: { harness: DEFAULT_CLAUDE_HARNESS.descriptor },
        instructions: "Inspect the repository and collect evidence for the plan.",
      },
    },
    writer_agent: {
      kind: "agent",
      agent: {
        name: "writer_agent",
        description: "Codex ACP agent for implementation-quality synthesis.",
        backend: DEFAULT_CODEX_HARNESS.backend,
        parameters: { harness: DEFAULT_CODEX_HARNESS.descriptor },
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
  const [sessionGroupExpandedById, setSessionGroupExpandedById] = useState<Record<string, boolean>>(
    {},
  );
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
  const [providerSetupRequested, setProviderSetupRequested] = useState(false);
  const [desktopUpdate, setDesktopUpdate] = useState<DesktopUpdateState>({
    phase: "hidden",
    currentVersion: "unknown",
  });
  const [pinnedSummaryOpen, setPinnedSummaryOpen] = useState(false);
  const [bottomPanelOpen, setBottomPanelOpen] = useState(false);
  const [composerDockElement, setComposerDockElement] = useState<HTMLElement | null>(null);
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
  const activeAgentInteraction = agentInteractions[0];
  const activeToolApproval =
    activeAgentInteraction?.kind === "tool_approval" ? activeAgentInteraction : null;
  const runtimeBodyRef = useRef<HTMLDivElement>(null);
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
    data: personalMemory,
    error: personalMemoryError,
    isLoading: personalMemoryLoading,
    mutate: mutatePersonalMemory,
  } = useSWR<GlobalMemoryState>(
    settingsSection === "memory" ? PERSONAL_MEMORY_KEY : null,
    () => api.getPersonalMemory(),
    { revalidateOnFocus: true, revalidateOnReconnect: false },
  );
  const {
    data: taskRuntime,
    error: taskRuntimeError,
    isLoading: taskRuntimeLoading,
    mutate: mutateTaskRuntime,
  } = useSWR<DesktopTaskRuntimeListResult>(
    settingsSection === "runtime" ? TASK_RUNTIME_KEY : null,
    () => api.listTaskWorkItems(),
    {
      refreshInterval: TASK_RUNTIME_REFRESH_INTERVAL_MS,
      revalidateOnFocus: true,
      revalidateOnReconnect: false,
    },
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
      : settingsSection === "memory"
        ? "Durable, user-managed context"
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
      : settingsSection === "memory"
        ? "Global Memory"
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

  useLayoutEffect(() => {
    const body = runtimeBodyRef.current;
    if (!body) return;
    if (!composerDockElement) {
      body.style.removeProperty("--composer-overlay-height");
      return;
    }

    const updateComposerHeight = () => {
      const height = Math.max(132, Math.ceil(composerDockElement.getBoundingClientRect().height));
      body.style.setProperty("--composer-overlay-height", `${height}px`);
    };
    updateComposerHeight();

    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(updateComposerHeight);
    observer.observe(composerDockElement);
    return () => observer.disconnect();
  }, [composerDockElement]);

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
    if (session?.source !== "local" || sessionActionPending) return;
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
    if (session?.source !== "local" || sessionActionPending) return;
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
        className={cx(
          String.raw`session-item [position:relative] [width:100%] [min-height:48px] [padding:8px] [display:grid] [grid-template-columns:28px_minmax(0,_1fr)] [gap:8px] [align-items:center] [text-align:left] [color:var(--foreground)] [background:transparent] [border:1px_solid_transparent] [border-radius:var(--radius)] [cursor:pointer] [transition:transform_var(--duration-fast)_var(--ease-out),_background-color_var(--duration-fast)_var(--ease-out),_border-color_var(--duration-fast)_var(--ease-out),_box-shadow_var(--duration-fast)_var(--ease-out)] [&.is-loading_.session-item\_\_icon_svg]:[animation:spin_900ms_linear_infinite]`,
          isActive && "is-active",
          isPending && "is-loading",
        )}
      >
        <span className="session-item__icon [width:28px] [height:28px] [display:grid] [place-items:center] [color:var(--muted)] [background:rgba(255,_255,_255,_0.055)] [border:1px_solid_var(--border-subtle)] [border-radius:8px] [box-shadow:inset_0_1px_0_rgba(255,_255,_255,_0.045)] [&_svg]:[width:14px] [&_svg]:[height:14px]">
          {isPending ? (
            <Loader2 aria-hidden="true" />
          ) : isLocal ? (
            <Clock3 aria-hidden="true" />
          ) : (
            <GitBranch aria-hidden="true" />
          )}
        </span>
        <span className="session-item__body [min-width:0] [display:flex] [flex-direction:column] [gap:2px]">
          <span className="session-item__title [color:var(--foreground)] [font-size:13px] [font-weight:560] [line-height:1.2] [min-width:0] [overflow:hidden] [text-overflow:ellipsis] [white-space:nowrap]">
            {session.title || "Untitled"}
          </span>
          <span className="session-item__meta [color:var(--muted-foreground)] [font-size:11.5px] [min-width:0] [overflow:hidden] [text-overflow:ellipsis] [white-space:nowrap]">
            {sessionMeta(session, sessionGroupMode)}
          </span>
        </span>
        {session.pinned && (
          <Pin
            className="session-item__pin [position:absolute] [top:7px] [right:7px] [width:12px] [height:12px] [color:var(--muted-foreground)]"
            aria-label="Pinned task"
          />
        )}
      </button>
    );
  };
  return (
    <div
      className={cx(
        "app-shell [@media(prefers-color-scheme:light)]:[background:linear-gradient(180deg,_rgba(255,_255,_255,_0.82),_transparent_260px),_linear-gradient(115deg,_rgba(8,_124,_155,_0.07),_transparent_34%),_var(--background)] max-860:[grid-template-columns:248px_minmax(0,_1fr)] max-680:[grid-template-columns:0_minmax(0,_1fr)] [position:relative] [isolation:isolate] [display:grid] [grid-template-columns:288px_minmax(0,_1fr)] [grid-template-rows:54px_minmax(0,_1fr)] [width:100vw] [height:100vh] [min-width:0] [min-height:0] [overflow:hidden] [background:linear-gradient(180deg,_rgba(255,_255,_255,_0.036),_rgba(255,_255,_255,_0)_260px),_linear-gradient(115deg,_rgba(149,_233,_255,_0.055),_transparent_34%),_linear-gradient(290deg,_rgba(52,_211,_153,_0.035),_transparent_42%),_var(--background)] [transition:grid-template-columns_var(--duration-med)_var(--ease-out)] [&.app-shell--collapsed]:[grid-template-columns:0_minmax(0,_1fr)] [&.app-shell--settings]:[grid-template-rows:0_minmax(0,_1fr)]",
        !sidebarOpen &&
          "app-shell--collapsed [&_.sidebar]:[opacity:0] [&_.sidebar]:[transform:translateX(-12px)] [&_.sidebar]:[pointer-events:none] max-680:[&_.sidebar]:[opacity:0] max-680:[&_.sidebar]:[transform:translateX(-100%)] max-680:[&_.sidebar]:[pointer-events:none]",
        settingsSection &&
          String.raw`app-shell--settings [@media(prefers-color-scheme:light)]:[&_.sidebar]:[background:rgba(250,_250,_251,_0.94)] [@media(prefers-color-scheme:light)]:[&_.settings-workspace]:[background:#fbfbfc] [@media(prefers-color-scheme:light)]:[&_.settings-workspace\_\_content]:[background:#fbfbfc] max-680:[&_.sidebar]:[position:static] max-680:[&_.sidebar]:[width:auto] max-680:[&_.sidebar]:[opacity:1] max-680:[&_.sidebar]:[transform:none] max-680:[&_.sidebar]:[pointer-events:auto]`,
      )}
    >
      {!settingsSection && (
        <header
          className={cx(
            "app-titlebar [position:relative] [z-index:2] [grid-column:2] [grid-row:1] [height:54px] [padding:0_12px] [display:flex] [align-items:center] [justify-content:space-between] [gap:12px] [border-bottom:1px_solid_var(--border-subtle)] [background:rgba(7,_8,_11,_0.76)] [box-shadow:var(--shadow-inset)] [-webkit-backdrop-filter:saturate(155%)_blur(var(--glass-blur))] [-webkit-app-region:drag] [@media(prefers-color-scheme:light)]:[background:linear-gradient(180deg,_rgba(255,_255,_255,_0.66),_transparent_310px)] max-680:[height:54px] max-680:[min-height:54px] max-680:[padding:0_8px] max-680:[align-items:center] max-680:[flex-direction:row] max-680:[gap:6px]",
            isMacOS && !sidebarOpen && "app-titlebar--macos",
          )}
          aria-label="Window title bar"
        >
          <div className="runtime__titlebar [flex:1_1_auto] [min-width:0] [display:flex] [align-items:center] [gap:4px]">
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
              <div className="runtime__title [margin-left:7px] [min-width:0] [display:flex] [flex-direction:column] [gap:1px] [&_h1]:[margin:0] [&_h1]:[min-width:0] [&_h1]:[overflow:hidden] [&_h1]:[text-overflow:ellipsis] [&_h1]:[white-space:nowrap] [&_h1]:[color:var(--foreground)] [&_h1]:[font-size:13.5px] [&_h1]:[font-weight:650] [&_h1]:[letter-spacing:0] [&_h1]:[line-height:1.2] [&_p]:[margin:0] [&_p]:[min-width:0] [&_p]:[overflow:hidden] [&_p]:[text-overflow:ellipsis] [&_p]:[white-space:nowrap] [&_p]:[color:var(--muted-foreground)] [&_p]:[font-size:10.5px] [&_p]:[line-height:1.2] max-680:[display:none]">
                <h1>{headerTitle}</h1>
                <p>{runSubtitle}</p>
              </div>
            )}
          </div>

          <div className="runtime__actions [flex:0_1_auto] [justify-content:flex-end] [min-width:0] [display:flex] [align-items:center] [gap:4px] max-680:[width:auto] max-680:[justify-content:flex-end] max-680:[flex-wrap:nowrap]">
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
                  className="side-chat-titlebar-button [position:relative]"
                >
                  <MessageSquarePlus data-icon aria-hidden="true" />
                  {unreadSideChatCount > 0 && (
                    <span className="side-chat-unread-badge [position:absolute] [top:1px] [right:1px] [min-width:14px] [height:14px] [padding:0_3px] [display:grid] [place-items:center] [color:#071015] [background:var(--accent)] [border:2px_solid_var(--background)] [border-radius:999px] [font-size:8px] [font-weight:800] [line-height:1]">
                      {unreadSideChatCount}
                    </span>
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

      <aside
        className="sidebar [position:relative] [z-index:1] [grid-column:1] [grid-row:1_/_-1] [min-width:0] [overflow:hidden] [background:rgba(12,_14,_20,_0.74)] [border-right:1px_solid_var(--border-subtle)] [display:flex] [flex-direction:column] [box-shadow:14px_0_42px_rgba(0,_0,_0,_0.18),_inset_-1px_0_0_rgba(255,_255,_255,_0.035)] [-webkit-backdrop-filter:saturate(150%)_blur(var(--glass-blur))] [transition:opacity_var(--duration-med)_var(--ease-out),_transform_var(--duration-med)_var(--ease-out)] [@media(prefers-color-scheme:light)]:[background:rgba(247,_249,_252,_0.82)] max-680:[position:absolute] max-680:[top:0] max-680:[bottom:0] max-680:[left:0] max-680:[z-index:30] max-680:[width:min(288px,_86vw)] max-680:[opacity:1] max-680:[transform:none] max-680:[pointer-events:auto]"
        aria-label={settingsSection ? "Settings navigation" : "Sessions"}
      >
        <div
          className={cx(
            "sidebar__titlebar [height:54px] [padding:0_10px] [display:flex] [flex:0_0_auto] [align-items:center] [gap:2px] [-webkit-app-region:drag]",
            isMacOS && "sidebar__titlebar--macos [padding-left:84px]",
          )}
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
            <div className="sidebar__brand [min-height:62px] [padding:14px_14px_10px] [display:flex] [align-items:center] [justify-content:space-between] [gap:12px] [-webkit-app-region:drag] [@media(prefers-color-scheme:light)]:[background:linear-gradient(180deg,_rgba(255,_255,_255,_0.66),_transparent_310px)]">
              <div className="brand-identity [min-width:0] [display:flex] [align-items:center] [gap:9px]">
                <AppBrandIcon className="brand-mark [width:34px] [height:34px] [flex:0_0_auto] [border:0] [border-radius:var(--radius)] [background:transparent] [box-shadow:none] [object-fit:contain] [display:grid] [place-items:center] [color:var(--foreground)] [&_svg]:[width:17px] [&_svg]:[height:17px]" />
                <div className="brand-copy [min-width:0]">
                  <div className="brand-title [font-size:17px] [font-weight:680] [line-height:1.15]">
                    {productConfig.name}
                  </div>
                  {productConfig.subtitle && (
                    <div className="brand-subtitle [margin-top:2px] [color:var(--muted-foreground)] [font-size:10.5px] [line-height:1.2]">
                      {productConfig.subtitle}
                    </div>
                  )}
                </div>
              </div>
              <button
                type="button"
                className="sidebar__search-toggle [width:34px] [height:34px] [flex:0_0_auto] [display:grid] [place-items:center] [color:var(--muted)] [background:transparent] [border:0] [border-radius:9px] [cursor:pointer] [&_svg]:[width:17px] [&_svg]:[height:17px] [&_svg]:[flex:0_0_auto]"
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
              <div
                className={String.raw`sidebar-search [&_>_svg]:[width:17px] [&_>_svg]:[height:17px] [&_>_svg]:[flex:0_0_auto] [&_input]:[min-width:0] [&_input]:[width:100%] [&_input]:[color:var(--foreground)] [&_input]:[background:transparent] [&_input]:[border:0] [&_input]:[outline:0] [&_input]:[font-size:12.5px] [height:38px] [margin:0_10px_6px] [padding:0_10px] [display:flex] [align-items:center] [gap:8px] [color:var(--muted-foreground)] [background:var(--input)] [border:1px_solid_var(--border-subtle)] [border-radius:10px] [&.settings-sidebar\_\_search]:[margin:0_0_24px]`}
              >
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

            <nav
              className="sidebar-primary-nav [padding:4px_10px_8px] [display:grid] [gap:2px] [border-bottom:1px_solid_var(--border-subtle)]"
              aria-label="Workspace"
            >
              <button
                type="button"
                className="sidebar-primary-nav__item [width:100%] [min-height:36px] [padding:7px_9px] [display:flex] [align-items:center] [gap:9px] [color:var(--foreground)] [background:transparent] [border:0] [border-radius:9px] [text-align:left] [cursor:pointer] [&_>_svg]:[width:17px] [&_>_svg]:[height:17px] [&_>_svg]:[flex:0_0_auto] [&_>_span]:[min-width:0] [&_>_span]:[flex:1] [&_>_span]:[overflow:hidden] [&_>_span]:[text-overflow:ellipsis] [&_>_span]:[white-space:nowrap] [&_>_span]:[font-size:13px] [&_>_span]:[font-weight:560] [&_>_small]:[min-width:20px] [&_>_small]:[padding:2px_5px] [&_>_small]:[color:var(--danger)] [&_>_small]:[background:var(--danger-muted)] [&_>_small]:[border-radius:999px] [&_>_small]:[font-size:10px] [&_>_small]:[line-height:1.2] [&_>_small]:[text-align:center]"
                onClick={() => newSession()}
              >
                <MessageSquarePlus aria-hidden="true" />
                <span>New task</span>
              </button>
              <button
                type="button"
                className={cx(
                  "sidebar-primary-nav__item [width:100%] [min-height:36px] [padding:7px_9px] [display:flex] [align-items:center] [gap:9px] [color:var(--foreground)] [background:transparent] [border:0] [border-radius:9px] [text-align:left] [cursor:pointer] [&_>_svg]:[width:17px] [&_>_svg]:[height:17px] [&_>_svg]:[flex:0_0_auto] [&_>_span]:[min-width:0] [&_>_span]:[flex:1] [&_>_span]:[overflow:hidden] [&_>_span]:[text-overflow:ellipsis] [&_>_span]:[white-space:nowrap] [&_>_span]:[font-size:13px] [&_>_span]:[font-weight:560] [&_>_small]:[min-width:20px] [&_>_small]:[padding:2px_5px] [&_>_small]:[color:var(--danger)] [&_>_small]:[background:var(--danger-muted)] [&_>_small]:[border-radius:999px] [&_>_small]:[font-size:10px] [&_>_small]:[line-height:1.2] [&_>_small]:[text-align:center]",
                  workflowPanelOpen && "is-active",
                )}
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
              <nav
                className="sidebar-extension-nav [min-width:0] [padding:10px_12px] [display:grid] [gap:7px] [border-bottom:1px_solid_var(--border-subtle)] [background:rgba(9,_11,_16,_0.26)]"
                aria-label="Registered GUI contributions"
              >
                <div className="sidebar-extension-nav__header [min-width:0] [display:flex] [align-items:center] [justify-content:space-between] [color:var(--muted-foreground)] [font-size:11px] [font-weight:650] [line-height:1.2] [text-transform:uppercase]">
                  <span>Apps</span>
                  <span>{registeredUiContributions.length}</span>
                </div>
                {registeredUiContributions.map((contribution) => (
                  <button
                    key={contribution.id}
                    type="button"
                    className={cx(
                      "sidebar-extension-nav__item [width:100%] [min-width:0] [min-height:34px] [padding:7px_9px] [border:1px_solid_transparent] [border-radius:var(--radius)] [display:flex] [align-items:center] [gap:8px] [color:var(--foreground)] [background:transparent] [font:inherit] [font-size:12px] [line-height:1.2] [text-align:left] [cursor:pointer] [&_svg]:[width:15px] [&_svg]:[height:15px] [&_svg]:[flex:0_0_auto] [&_svg]:[color:var(--muted-foreground)] [&_span]:[min-width:0] [&_span]:[overflow:hidden] [&_span]:[text-overflow:ellipsis] [&_span]:[white-space:nowrap] [&.is-active_svg]:[color:var(--accent)]",
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

            <div
              className="session-scroll [flex:1] [min-height:0] [overflow-y:auto] [padding:6px_8px_14px]"
              onScroll={() => setProjectPreview(null)}
            >
              <div className="project-list__header [position:relative] [min-height:32px] [padding:2px_6px_4px_8px] [display:flex] [align-items:center] [justify-content:space-between] [gap:8px] [color:var(--muted-foreground)] [font-size:11.5px] [font-weight:590] [letter-spacing:0.01em]">
                <span>Projects</span>
                <div
                  className="project-list__actions [gap:1px] [position:relative] [display:flex] [align-items:center]"
                  ref={projectHeaderMenuRef}
                >
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
                      <Loader2
                        className="is-spinning [animation:spin_0.9s_linear_infinite]"
                        aria-hidden="true"
                      />
                    ) : (
                      <Plus aria-hidden="true" />
                    )}
                  </button>
                  {projectHeaderMenu && (
                    <div
                      className={cx(
                        String.raw`project-list__menu [&_button]:[width:100%] [&_button]:[min-height:34px] [&_button]:[padding:7px_9px] [&_button]:[display:flex] [&_button]:[align-items:center] [&_button]:[gap:9px] [&_button]:[color:var(--foreground)] [&_button]:[background:transparent] [&_button]:[border:0] [&_button]:[border-radius:8px] [&_button]:[font:inherit] [&_button]:[font-size:12.5px] [&_button]:[font-weight:400] [&_button]:[text-align:left] [&_button]:[cursor:pointer] [&_button_svg]:[width:15px] [&_button_svg]:[height:15px] [&_button_svg]:[flex:0_0_auto] [position:absolute] [z-index:20] [top:30px] [right:-4px] [width:224px] [padding:5px] [display:grid] [gap:2px] [background:var(--popover,_var(--card-strong))] [border:1px_solid_var(--border)] [border-radius:11px] [box-shadow:var(--shadow-soft),_var(--shadow-inset)] [&.project-list\_\_menu--project]:[padding:6px] [&.project-list\_\_menu--project]:[background:var(--card-solid)] [&.project-list\_\_menu--project]:[border-radius:13px] [&.project-list\_\_menu--organize]:[width:196px] [&.project-list\_\_menu--organize]:[padding:8px] [&.project-list\_\_menu--organize]:[gap:1px] [&.project-list\_\_menu--organize]:[background:var(--card-solid)] [&.project-list\_\_menu--organize]:[border-radius:14px] [&.project-list\_\_menu--project_button]:[min-height:40px] [&.project-list\_\_menu--project_button]:[font-size:13px] [&.project-list\_\_menu--organize_button]:[min-height:36px] [&.project-list\_\_menu--organize_button]:[padding:6px_7px] [&.project-list\_\_menu--organize_button]:[gap:8px] [&.project-list\_\_menu--organize_button]:[font-size:13px]`,
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
                          <div className="project-list__menu-label [padding:6px_8px_3px] [color:var(--muted-foreground)] [font-size:12.5px] [line-height:1.2]">
                            Organize
                          </div>
                          <button
                            type="button"
                            role="menuitemradio"
                            aria-checked={projectOrganizationMode === "project"}
                            onClick={() => {
                              setProjectOrganizationMode("project");
                              setProjectHeaderMenu(null);
                            }}
                          >
                            <span className="project-list__menu-check [width:17px] [height:17px] [display:grid] [flex:0_0_17px] [place-items:center] [&_svg]:[width:16px] [&_svg]:[height:16px]">
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
                            <span className="project-list__menu-check [width:17px] [height:17px] [display:grid] [flex:0_0_17px] [place-items:center] [&_svg]:[width:16px] [&_svg]:[height:16px]">
                              {projectOrganizationMode === "list" && <Check aria-hidden="true" />}
                            </span>
                            <span>In one list</span>
                          </button>
                          <div className="project-list__menu-separator [height:1px] [margin:5px_6px] [background:var(--border)]" />
                          <div className="project-list__menu-label [padding:6px_8px_3px] [color:var(--muted-foreground)] [font-size:12.5px] [line-height:1.2]">
                            Sort by
                          </div>
                          <button
                            type="button"
                            role="menuitemradio"
                            aria-checked={projectSortMode === "priority"}
                            onClick={() => {
                              setProjectSortMode("priority");
                              setProjectHeaderMenu(null);
                            }}
                          >
                            <span className="project-list__menu-check [width:17px] [height:17px] [display:grid] [flex:0_0_17px] [place-items:center] [&_svg]:[width:16px] [&_svg]:[height:16px]">
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
                            <span className="project-list__menu-check [width:17px] [height:17px] [display:grid] [flex:0_0_17px] [place-items:center] [&_svg]:[width:16px] [&_svg]:[height:16px]">
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
                            <span className="project-list__menu-check [width:17px] [height:17px] [display:grid] [flex:0_0_17px] [place-items:center] [&_svg]:[width:16px] [&_svg]:[height:16px]">
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
              {projectError && (
                <div className="session-error [margin:10px_4px_0] [padding:9px] [display:flex] [align-items:flex-start] [gap:8px] [color:var(--danger)] [background:var(--danger-muted)] [border:1px_solid_rgba(248,_113,_113,_0.26)] [border-radius:var(--radius)] [box-shadow:var(--shadow-inset)] [font-size:12px] [line-height:1.35] [&_svg]:[flex:0_0_auto] [&_svg]:[width:14px] [&_svg]:[height:14px] [&_svg]:[margin-top:1px]">
                  {projectError}
                </div>
              )}
              {sessionActionError && !renamingSession && (
                <div className="session-error [margin:10px_4px_0] [padding:9px] [display:flex] [align-items:flex-start] [gap:8px] [color:var(--danger)] [background:var(--danger-muted)] [border:1px_solid_rgba(248,_113,_113,_0.26)] [border-radius:var(--radius)] [box-shadow:var(--shadow-inset)] [font-size:12px] [line-height:1.35] [&_svg]:[flex:0_0_auto] [&_svg]:[width:14px] [&_svg]:[height:14px] [&_svg]:[margin-top:1px]">
                  {sessionActionError}
                </div>
              )}
              {sessionsLoading && (
                <div className="session-status [margin:12px_8px] [color:var(--muted-foreground)] [font-size:12px]">
                  Loading projects
                </div>
              )}
              {!sessionsLoading && visibleDisplayGroups.length === 0 && (
                <div className="session-status [margin:12px_8px] [color:var(--muted-foreground)] [font-size:12px]">
                  {sidebarQuery.trim() ? "No matching sessions" : "No sessions"}
                </div>
              )}
              {projectOrganizationMode === "list" ? (
                <div
                  className={String.raw`session-group__items project-list__flat [display:flex] [flex-direction:column] [gap:3px] [padding:1px_0_4px] [&_.session-item\_\_pin]:[top:9px] [&_.session-item\_\_pin]:[right:8px]`}
                >
                  {visibleFlatSessions.map(renderSidebarSessionItem)}
                </div>
              ) : (
                visibleDisplayGroups.map((group) => {
                  const projectId = group.project?.id;
                  const expanded =
                    sidebarQuery.trim().length > 0 ||
                    (sessionGroupExpandedById[group.id] ??
                      (projectId ? activeProjectId === projectId : true));
                  const ProjectFolderIcon =
                    group.id === RECENTS_GROUP_ID ? Clock3 : expanded ? FolderOpen : Folder;
                  return (
                    <section
                      key={group.id}
                      className={cx(
                        String.raw`project-group [margin-left:-8px] [margin-right:16px] [margin-bottom:1px] [&.has-open-menu_.project-group\_\_actions]:[opacity:1] [&.has-open-menu_.project-group\_\_actions]:[transition:none] [&_.session-group\_\_items]:[gap:1px] [&_.session-item.is-active]:[border:0] [&_.session-item.is-active]:[box-shadow:none] [&_.session-item\_\_pin]:[top:9px] [&_.session-item\_\_pin]:[right:8px]`,
                        expanded && "is-expanded",
                        projectActionMenuId === group.project?.id && "has-open-menu",
                      )}
                      aria-label={group.label}
                    >
                      {group.project?.id === renamingProjectId ? (
                        <form
                          className="project-group__rename [min-height:32px] [padding:4px_7px] [display:grid] [grid-template-columns:17px_minmax(0,_1fr)] [align-items:center] [gap:7px] [background:var(--card-hover)] [border-radius:8px] [&_>_svg]:[width:15px] [&_>_svg]:[height:15px] [&_>_svg]:[color:var(--foreground)] [&_input]:[width:100%] [&_input]:[min-width:0] [&_input]:[height:24px] [&_input]:[padding:2px_5px] [&_input]:[color:var(--foreground)] [&_input]:[background:var(--input)] [&_input]:[border:1px_solid_var(--focus-ring)] [&_input]:[border-radius:5px] [&_input]:[outline:none] [&_input]:[font:inherit] [&_input]:[font-size:12.5px]"
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
                            "project-group__header-row [position:relative] [min-height:32px] [display:grid] [grid-template-columns:minmax(0,_1fr)_auto] [align-items:center] [color:var(--muted)] [border-radius:8px] [transition:color_var(--duration-fast)_ease,_background-color_var(--duration-fast)_ease]",
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
                            className="project-group__trigger [width:100%] [min-height:32px] [padding:5px_7px] [display:grid] [grid-template-columns:17px_minmax(0,_1fr)] [align-items:center] [gap:7px] [color:inherit] [background:transparent] [border:0] [border-radius:8px] [font:inherit] [text-align:left] [cursor:pointer] [&_>_span]:[min-width:0] [&_>_span]:[overflow:hidden] [&_>_span]:[text-overflow:ellipsis] [&_>_span]:[white-space:nowrap] [&_>_span]:[font-size:12.5px] [&_>_span]:[font-weight:470]"
                            title={group.cwd || group.label}
                            aria-expanded={expanded}
                            onClick={() => {
                              setSessionGroupExpandedById((current) => ({
                                ...current,
                                [group.id]: !expanded,
                              }));
                            }}
                          >
                            <ProjectFolderIcon aria-hidden="true" />
                            <span>{group.label}</span>
                          </button>
                          {group.project && (
                            <div
                              className={String.raw`project-group__actions [opacity:0] [transition:opacity_var(--duration-fast)_ease] [position:relative] [display:flex] [align-items:center] [&_.project-list\_\_menu]:[top:29px] [&_.project-list\_\_menu]:[right:0] [&_.project-list\_\_menu]:[z-index:30]`}
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
                                  className={String.raw`project-list__menu project-list__menu--project [&_button]:[width:100%] [&_button]:[min-height:34px] [&_button]:[padding:7px_9px] [&_button]:[display:flex] [&_button]:[align-items:center] [&_button]:[gap:9px] [&_button]:[color:var(--foreground)] [&_button]:[background:transparent] [&_button]:[border:0] [&_button]:[border-radius:8px] [&_button]:[font:inherit] [&_button]:[font-size:12.5px] [&_button]:[font-weight:400] [&_button]:[text-align:left] [&_button]:[cursor:pointer] [&_button_svg]:[width:15px] [&_button_svg]:[height:15px] [&_button_svg]:[flex:0_0_auto] [position:absolute] [z-index:20] [top:30px] [right:-4px] [width:224px] [padding:5px] [display:grid] [gap:2px] [background:var(--popover,_var(--card-strong))] [border:1px_solid_var(--border)] [border-radius:11px] [box-shadow:var(--shadow-soft),_var(--shadow-inset)] [&.project-list\_\_menu--project]:[padding:6px] [&.project-list\_\_menu--project]:[background:var(--card-solid)] [&.project-list\_\_menu--project]:[border-radius:13px] [&.project-list\_\_menu--organize]:[width:196px] [&.project-list\_\_menu--organize]:[padding:8px] [&.project-list\_\_menu--organize]:[gap:1px] [&.project-list\_\_menu--organize]:[background:var(--card-solid)] [&.project-list\_\_menu--organize]:[border-radius:14px] [&.project-list\_\_menu--project_button]:[min-height:40px] [&.project-list\_\_menu--project_button]:[font-size:13px] [&.project-list\_\_menu--organize_button]:[min-height:36px] [&.project-list\_\_menu--organize_button]:[padding:6px_7px] [&.project-list\_\_menu--organize_button]:[gap:8px] [&.project-list\_\_menu--organize_button]:[font-size:13px]`}
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
                        <div
                          className={cx(
                            "session-group__items [display:flex] [flex-direction:column] [gap:3px]",
                            group.id === RECENTS_GROUP_ID
                              ? "[padding:1px_0_4px]"
                              : "[padding:1px_0_4px_23px]",
                          )}
                        >
                          {group.sessions.map(renderSidebarSessionItem)}
                        </div>
                      )}
                    </section>
                  );
                })
              )}
              {visibleSessionErrors.map((error) => (
                <div
                  key={error.harnessId}
                  className="session-error [margin:10px_4px_0] [padding:9px] [display:flex] [align-items:flex-start] [gap:8px] [color:var(--danger)] [background:var(--danger-muted)] [border:1px_solid_rgba(248,_113,_113,_0.26)] [border-radius:var(--radius)] [box-shadow:var(--shadow-inset)] [font-size:12px] [line-height:1.35] [&_svg]:[flex:0_0_auto] [&_svg]:[width:14px] [&_svg]:[height:14px] [&_svg]:[margin-top:1px]"
                >
                  <XCircle aria-hidden="true" />
                  <span>
                    {error.harnessLabel}: {error.message}
                  </span>
                </div>
              ))}
            </div>
            <footer
              className="sidebar-account-area [position:relative] [z-index:4] [flex:0_0_auto] [padding:6px_10px_10px] [border-top:1px_solid_var(--border-subtle)] [background:color-mix(in_srgb,_var(--background)_92%,_transparent)]"
              ref={sidebarAccountRef}
            >
              {accountMenuOpen && (
                <div
                  className="sidebar-account-menu [position:absolute] [right:10px] [bottom:51px] [left:10px] [z-index:12] [overflow:hidden] [color:var(--foreground)] [background:var(--card-solid)] [border:1px_solid_var(--border)] [border-radius:13px] [box-shadow:0_12px_32px_rgba(0,_0,_0,_0.16)]"
                  role="menu"
                  aria-label="Local workspace menu"
                >
                  <div className="sidebar-account-menu__identity [min-height:50px] [padding:9px_11px] [display:flex] [align-items:center] [gap:9px] [&_strong]:[overflow:hidden] [&_strong]:[font-size:12.5px] [&_strong]:[font-weight:610] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:10.5px]">
                    <span
                      className="sidebar-account-avatar [width:27px] [height:27px] [flex:0_0_auto] [display:grid] [place-items:center] [color:var(--muted)] [background:var(--input)] [border:1px_solid_var(--border)] [border-radius:999px] [&_svg]:[width:14px] [&_svg]:[height:14px]"
                      aria-hidden="true"
                    >
                      <FolderOpen />
                    </span>
                    <span>
                      <strong>Local workspace</strong>
                      <small>
                        {composerWorkspaceRoot
                          ? abbreviateHomePath(composerWorkspaceRoot)
                          : "No Project selected"}
                      </small>
                    </span>
                  </div>
                  <div className="sidebar-account-menu__items [padding:5px] [display:grid] [gap:1px] [border-top:1px_solid_var(--border-subtle)] [&_button]:[width:100%] [&_button]:[min-height:36px] [&_button]:[padding:6px_8px] [&_button]:[display:grid] [&_button]:[grid-template-columns:16px_minmax(0,_1fr)_14px] [&_button]:[align-items:center] [&_button]:[gap:8px] [&_button]:[color:var(--foreground)] [&_button]:[background:transparent] [&_button]:[border:0] [&_button]:[border-radius:7px] [&_button]:[font-size:12.5px] [&_button]:[text-align:left] [&_button]:[cursor:pointer] [&_button_svg]:[width:15px] [&_button_svg]:[height:15px] [&_button_svg]:[color:var(--muted)]">
                    <button type="button" role="menuitem" onClick={() => openSettings("general")}>
                      <Settings aria-hidden="true" />
                      <span>Settings</span>
                    </button>
                  </div>
                </div>
              )}
              <div
                className={cx(
                  "sidebar-account-row [width:100%] [min-width:0] [min-height:42px] [padding:5px_8px] [display:flex] [align-items:center] [gap:7px] [border-radius:9px]",
                  accountMenuOpen && "is-open",
                )}
              >
                <button
                  type="button"
                  className="sidebar-account-trigger [min-width:0] [min-height:32px] [padding:0] [flex:1_1_auto] [display:flex] [align-items:center] [gap:9px] [color:var(--foreground)] [background:transparent] [border:0] [border-radius:7px] [font-size:13px] [font-weight:560] [text-align:left] [cursor:pointer]"
                  aria-label="Open local workspace menu"
                  aria-haspopup="menu"
                  aria-expanded={accountMenuOpen}
                  onClick={() => setAccountMenuOpen((open) => !open)}
                >
                  <span
                    className="sidebar-account-avatar [width:27px] [height:27px] [flex:0_0_auto] [display:grid] [place-items:center] [color:var(--muted)] [background:var(--input)] [border:1px_solid_var(--border)] [border-radius:999px] [&_svg]:[width:14px] [&_svg]:[height:14px]"
                    aria-hidden="true"
                  >
                    <FolderOpen />
                  </span>
                  <span>Local workspace</span>
                </button>
                {updateVisible && (
                  <button
                    type="button"
                    className={cx(
                      "sidebar-update-control [width:30px] [min-width:30px] [height:30px] [padding:0_8px] [flex:0_0_auto] [overflow:hidden] [display:flex] [align-items:center] [justify-content:center] [gap:0] [color:#ffffff] [background:#626bd8] [border:0] [border-radius:999px] [font-size:11.5px] [font-weight:650] [line-height:1] [cursor:pointer] [transition:width_var(--duration-fast)_var(--ease-out),_background-color_var(--duration-fast)_var(--ease-out)] [&_svg]:[width:14px] [&_svg]:[height:14px] [&_svg]:[flex:0_0_auto] [&_svg]:[opacity:1] [&_svg]:[transition:width_var(--duration-fast)_var(--ease-out),_opacity_var(--duration-fast)_var(--ease-out)] [&_>_span]:[max-width:0] [&_>_span]:[overflow:hidden] [&_>_span]:[opacity:0] [&_>_span]:[white-space:nowrap] [&_>_span]:[transition:max-width_var(--duration-fast)_var(--ease-out),_opacity_var(--duration-fast)_var(--ease-out)]",
                      updateBusy && "is-busy",
                    )}
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
          "runtime [position:relative] [z-index:1] [grid-column:2] [grid-row:2] [min-width:0] [min-height:0] [height:100%] [overflow:hidden] [display:grid] [grid-template-rows:minmax(0,_1fr)_auto] [background:rgba(7,_8,_11,_0.26)] [@media(prefers-color-scheme:light)]:[background:linear-gradient(180deg,_rgba(255,_255,_255,_0.66),_transparent_310px)]",
          rightPanelMounted &&
            "runtime--right-panel [&_>_.panel-transition--bottom]:[width:calc(100%_-_var(--right-panel-width,_50%))] max-680:[&_>_.panel-transition--bottom]:[width:100%]",
          sideChatPaneOpen &&
            "runtime--side-chat [&_>_.panel-transition--bottom]:[width:calc(100%_-_var(--side-chat-width))] max-860:[&_>_.panel-transition--bottom]:[width:100%]",
        )}
        style={
          {
            "--side-chat-width": `${sidePaneWidth}%`,
            ...(rightPanelWidth === null ? {} : { "--right-panel-width": `${rightPanelWidth}px` }),
          } as React.CSSProperties
        }
      >
        <div
          ref={runtimeBodyRef}
          className={cx(
            String.raw`runtime__body [position:relative] [min-width:0] [min-height:0] [overflow:hidden] [display:grid] [grid-template-columns:minmax(0,_1fr)] [&.runtime\_\_body--right-panel]:[padding-right:var(--right-panel-width,_50%)] [&.runtime\_\_body--right-panel]:[overflow:visible] [&.runtime\_\_body--right-panel]:[grid-template-columns:minmax(0,_1fr)] [&&.runtime\_\_body--side-chat]:[padding-right:var(--side-chat-width)] [&&.runtime\_\_body--side-chat]:[overflow:visible] [&&.runtime\_\_body--side-chat]:[grid-template-columns:minmax(0,_1fr)]`,
            rightPanelMounted && "runtime__body--right-panel max-680:[padding-right:0]",
            sideChatPaneOpen && "runtime__body--side-chat max-860:[padding-right:0]",
          )}
        >
          <div className="runtime__content [display:grid] [grid-template-rows:auto_minmax(0,_1fr)] [min-width:0] [min-height:0] [overflow:hidden]">
            {pinnedSummaryMounted && (
              <div
                className={cx(
                  "panel-transition panel-transition--pinned [display:grid] [grid-template-rows:0fr] [--panel-transition-transform:translateY(-10px)] [opacity:0] [pointer-events:none] [transform:var(--panel-transition-transform)] [transition:opacity_var(--duration-fast)_var(--ease-out),_transform_var(--duration-med)_var(--ease-out),_grid-template-rows_var(--duration-med)_var(--ease-out)] [will-change:opacity,_transform] [&.panel-transition--right]:[--panel-transition-transform:translateX(16px)] [&.panel-transition--right]:[position:absolute] [&.panel-transition--right]:[top:0] [&.panel-transition--right]:[right:0] [&.panel-transition--right]:[bottom:0] [&.panel-transition--right]:[z-index:18] [&.panel-transition--right]:[width:var(--right-panel-width,_50%)] [&.panel-transition--right]:[min-width:0] [&.panel-transition--right]:[min-height:0] [&.panel-transition--right]:[overflow:hidden]",
                  pinnedSummaryOpen && "is-open",
                )}
                aria-hidden={!pinnedSummaryOpen}
                inert={!pinnedSummaryOpen}
              >
                <div className="panel-transition__inner [min-width:0] [min-height:0] [overflow:hidden]">
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
            <div className="runtime__surface [display:block] [grid-row:2] [min-width:0] [min-height:0] [overflow:hidden] [&_>_*]:[height:100%]">
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
              ) : settingsSection === "memory" ? (
                <PersonalMemorySettings
                  memory={personalMemory}
                  loading={personalMemoryLoading}
                  error={personalMemoryError}
                  onSave={async (input) => {
                    await mutatePersonalMemory(await api.savePersonalMemory(input), false);
                  }}
                  onForget={async (input) => {
                    await mutatePersonalMemory(await api.forgetPersonalMemory(input), false);
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
                  openAddProvider={providerSetupRequested}
                  onAddProviderOpened={() => setProviderSetupRequested(false)}
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
                  taskRuntime={taskRuntime}
                  taskRuntimeLoading={taskRuntimeLoading}
                  taskRuntimeError={taskRuntimeError}
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
                  onRefreshTasks={async () => {
                    await mutateTaskRuntime();
                  }}
                  onCancelTask={async (workItemId) => {
                    await api.cancelTaskWorkItem({ workItemId });
                    await mutateTaskRuntime();
                  }}
                  onDecideApproval={async (approvalId, status) => {
                    await api.decideTaskApproval({
                      approvalId,
                      status,
                      decidedBy: "desktop-user",
                    });
                    await mutateTaskRuntime();
                  }}
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
                  className="transcript-scroll [min-height:0] [overflow-y:auto]"
                  onMouseUp={captureTranscriptSelection}
                  onKeyUp={captureTranscriptSelection}
                >
                  <div
                    className={cx(
                      "transcript max-860:[padding:22px_16px_28px] [width:min(100%,_960px)] [min-height:100%] [margin:0_auto] [padding:38px_30px_52px] [display:flex] [flex-direction:column] [gap:0] [&.transcript--empty]:[width:min(100%,_1120px)] [&.transcript--empty]:[padding-top:0] [&.transcript--empty]:[padding-bottom:0]",
                      emptyRun && "transcript--empty",
                    )}
                    style={{
                      paddingBottom: "calc(var(--composer-overlay-height, 132px) + 24px)",
                    }}
                  >
                    {emptyRun ? (
                      <EmptyRun
                        projectLabel={emptyProjectLabel}
                        rightPanelOpen={activeRightPanelKind !== null}
                        needsModel={!extensionInventoryLoading && manualCompositionNeedsModel}
                        onConnectModelProvider={() => {
                          setProviderSetupRequested(true);
                          openSettings("providers");
                        }}
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
                "panel-transition panel-transition--right [&_>_.runtime-right-panel]:[height:100%] max-860:[position:absolute] max-860:[top:0] max-860:[right:0] max-860:[bottom:0] max-860:[z-index:20] max-860:[width:var(--right-panel-width,_50%)] max-680:[width:min(100%,_var(--right-panel-width,_100%))] max-680:[box-shadow:-12px_0_32px_rgba(15,_23,_42,_0.14)] [--panel-transition-transform:translateY(-10px)] [opacity:0] [pointer-events:none] [transform:var(--panel-transition-transform)] [transition:opacity_var(--duration-fast)_var(--ease-out),_transform_var(--duration-med)_var(--ease-out),_grid-template-rows_var(--duration-med)_var(--ease-out)] [will-change:opacity,_transform] [&.panel-transition--right]:[--panel-transition-transform:translateX(16px)] [&.panel-transition--right]:[position:absolute] [&.panel-transition--right]:[top:0] [&.panel-transition--right]:[right:0] [&.panel-transition--right]:[bottom:0] [&.panel-transition--right]:[z-index:18] [&.panel-transition--right]:[width:var(--right-panel-width,_50%)] [&.panel-transition--right]:[min-width:0] [&.panel-transition--right]:[min-height:0] [&.panel-transition--right]:[overflow:hidden]",
                activeRightPanelKind && "is-open",
              )}
              aria-hidden={activeRightPanelKind === null}
              inert={activeRightPanelKind === null}
            >
              <div
                className="right-panel-resize [position:absolute] [z-index:3] [top:0] [bottom:0] [left:-5px] [width:10px] [cursor:col-resize] [touch-action:none] max-680:[display:none]"
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
              className="side-chat-pane [position:absolute] [inset:0_0_0_auto] [z-index:24] [min-width:360px] [max-width:55%] [min-height:0] [display:grid] [grid-template-rows:auto_auto_minmax(0,_1fr)_auto] [color:var(--foreground)] [background:rgba(13,_15,_20,_0.98)] [border-left:1px_solid_var(--border)] [box-shadow:-18px_0_42px_rgba(0,_0,_0,_0.22)] max-860:![width:100%] max-860:[max-width:none] max-860:[min-width:0] max-860:[border-left:0]"
              ref={sidePaneRef}
              aria-label="Side chats"
              style={{ width: `${sidePaneWidth}%` }}
            >
              <div
                className="side-chat-pane__resizer [position:absolute] [top:0] [bottom:0] [left:-4px] [z-index:2] [width:8px] [cursor:col-resize] [touch-action:none] max-860:[display:none]"
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
              <header className="side-chat-pane__header [min-width:0] [min-height:48px] [padding:6px_8px] [display:flex] [align-items:center] [gap:7px] [border-bottom:1px_solid_var(--border-subtle)] [background:var(--card-solid)]">
                <button
                  type="button"
                  className="side-chat-pane__back [display:none] [height:30px] [color:var(--muted-foreground)] [background:transparent] [border:0] [border-radius:7px] [cursor:pointer] [&_svg]:[width:14px] [&_svg]:[height:14px] max-860:[padding:0_7px] max-860:[display:inline-flex] max-860:[align-items:center] max-860:[gap:4px] max-860:[flex:0_0_auto] max-860:[font-size:10px]"
                  onClick={() => void hideSideChats()}
                >
                  <ArrowLeft aria-hidden="true" />
                  <span>Main chat</span>
                </button>
                <div
                  className="side-chat-tabs [min-width:0] [flex:1] [display:flex] [align-items:center] [gap:4px] [overflow-x:auto] [scrollbar-width:none]"
                  role="tablist"
                  aria-label="Side chat tabs"
                >
                  {sideChatState.chats.map((chat) => (
                    <button
                      type="button"
                      role="tab"
                      aria-selected={chat.id === activeSideChat.id}
                      className={cx(
                        "side-chat-tab [min-width:76px] [max-width:132px] [padding:0_8px] [display:flex] [align-items:center] [gap:5px] [font-size:10.5px] [height:30px] [color:var(--muted-foreground)] [background:transparent] [border:0] [border-radius:7px] [cursor:pointer]",
                        chat.id === activeSideChat.id && "is-active",
                      )}
                      key={chat.id}
                      onClick={() => void activateSideChat(chat.id)}
                      title={chat.title}
                    >
                      <span>{chat.title}</span>
                      {chat.runState !== "idle" && (
                        <Loader2
                          className="side-chat-tab__running [width:12px] [height:12px] [flex:0_0_auto] [animation:spin_0.9s_linear_infinite]"
                          aria-label="Running"
                        />
                      )}
                      {chat.unread && (
                        <span
                          className="side-chat-tab__unread [width:6px] [height:6px] [flex:0_0_auto] [background:var(--accent)] [border-radius:999px]"
                          aria-label="Unread"
                        />
                      )}
                    </button>
                  ))}
                  <button
                    type="button"
                    className="side-chat-tabs__add [height:30px] [color:var(--muted-foreground)] [background:transparent] [border:0] [border-radius:7px] [cursor:pointer] [width:30px] [flex:0_0_auto] [display:grid] [place-items:center] [&_svg]:[width:14px] [&_svg]:[height:14px]"
                    onClick={() => void createSideChat()}
                    aria-label="New side chat"
                    title="New side chat"
                  >
                    <Plus aria-hidden="true" />
                  </button>
                </div>
                <div className="side-chat-pane__actions [display:flex] [align-items:center] [gap:2px] [&_button]:[height:30px] [&_button]:[color:var(--muted-foreground)] [&_button]:[background:transparent] [&_button]:[border:0] [&_button]:[border-radius:7px] [&_button]:[cursor:pointer] [&_button]:[width:30px] [&_button]:[flex:0_0_auto] [&_button]:[display:grid] [&_button]:[place-items:center] [&_svg]:[width:14px] [&_svg]:[height:14px]">
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
              <div className="side-chat-pane__boundary [min-width:0] [padding:6px_12px] [display:flex] [align-items:center] [gap:7px] [overflow:hidden] [color:var(--muted-foreground)] [border-bottom:1px_solid_var(--border-subtle)] [font-size:9px] [&_span]:[overflow:hidden] [&_span]:[text-overflow:ellipsis] [&_span]:[white-space:nowrap]">
                <span>Transient fork</span>
                <span>Anchored after parent message {activeSideChat.anchor.messageIndex + 1}</span>
                <span>Read-only lane</span>
              </div>
              <div
                ref={sideChatScrollRef}
                className="side-chat-pane__transcript [min-height:0] [overflow-y:auto] [padding:24px_20px_34px] [&_.conversation-turn_+_.conversation-turn]:[margin-top:34px]"
              >
                {activeSideChat.messages.length === 0 ? (
                  <div className="side-chat-pane__empty [min-height:100%] [padding:34px_18px] [display:grid] [align-content:center] [justify-items:center] [text-align:center] [color:var(--muted-foreground)] [&_svg]:[width:24px] [&_svg]:[height:24px] [&_svg]:[color:var(--accent)] [&_h2]:[margin:12px_0_6px] [&_h2]:[color:var(--foreground)] [&_h2]:[font-size:15px] [&_p]:[max-width:320px] [&_p]:[margin:0] [&_p]:[font-size:11.5px] [&_p]:[line-height:1.55]">
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
              <footer
                className={String.raw`side-chat-pane__composer [min-width:0] [padding:8px_10px_10px] [border-top:1px_solid_var(--border-subtle)] [background:rgba(7,_8,_11,_0.88)] [&_.composer]:[padding:7px_8px_6px] [&_.composer]:[border-radius:12px] [&_.composer\_\_textarea]:[min-height:38px] [&_.composer\_\_textarea]:[max-height:148px] [&_.composer\_\_textarea]:[padding-bottom:5px] [&_.composer\_\_textarea]:[font-size:13px]`}
              >
                {activeSideChat.contextChips.length > 0 && (
                  <div
                    className="side-chat-context-chips [margin-bottom:6px] [display:flex] [gap:5px] [overflow-x:auto] [&_span]:[min-width:0] [&_span]:[max-width:260px] [&_span]:[padding:4px_7px] [&_span]:[display:inline-flex] [&_span]:[align-items:center] [&_span]:[gap:4px] [&_span]:[overflow:hidden] [&_span]:[color:var(--muted-foreground)] [&_span]:[background:var(--accent-soft)] [&_span]:[border:1px_solid_rgba(149,_233,_255,_0.16)] [&_span]:[border-radius:999px] [&_span]:[font-size:9px] [&_span]:[text-overflow:ellipsis] [&_span]:[white-space:nowrap] [&_svg]:[width:11px] [&_svg]:[height:11px] [&_svg]:[flex:0_0_auto]"
                    aria-label="Selected side chat context"
                  >
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
                  <span className="side-chat-pane__mode [display:inline-flex] [align-items:center] [gap:4px] [color:var(--muted-foreground)] [font-size:9px] [&_svg]:[width:12px] [&_svg]:[height:12px] [&_svg]:[color:var(--accent)]">
                    <ShieldCheck aria-hidden="true" />
                    Read-only
                  </span>
                </Composer>
              </footer>
            </aside>
          )}
          {!settingsSection &&
            (activeToolApproval || (!workflowPanelOpen && !activeUiContribution)) && (
              <footer
                ref={setComposerDockElement}
                className={cx(
                  "composer-dock [position:absolute] [right:0] [bottom:0] [left:0] [z-index:26] [padding:9px_18px_12px] [background:linear-gradient(180deg,_transparent,_rgba(7,_8,_11,_0.72))] [@media(prefers-color-scheme:light)]:[background:linear-gradient(180deg,_transparent,_rgba(244,_246,_249,_0.76))] max-860:[padding:12px_14px_14px] max-680:[padding:10px_14px]",
                  activeToolApproval && "composer-dock--approval [padding-top:12px]",
                  sideChatPaneOpen
                    ? "composer-dock--side-chat [right:var(--side-chat-width)] max-860:[display:none]"
                    : rightPanelMounted &&
                        "composer-dock--right-panel [right:var(--right-panel-width,_50%)] max-680:[right:0]",
                )}
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
        </div>

        <div
          className={cx(
            "panel-transition panel-transition--bottom [display:grid] [grid-template-rows:0fr] [--panel-transition-transform:translateY(-10px)] [opacity:0] [pointer-events:none] [transform:var(--panel-transition-transform)] [transition:opacity_var(--duration-fast)_var(--ease-out),_transform_var(--duration-med)_var(--ease-out),_grid-template-rows_var(--duration-med)_var(--ease-out)] [will-change:opacity,_transform] [&.panel-transition--right]:[--panel-transition-transform:translateX(16px)] [&.panel-transition--right]:[position:absolute] [&.panel-transition--right]:[top:0] [&.panel-transition--right]:[right:0] [&.panel-transition--right]:[bottom:0] [&.panel-transition--right]:[z-index:18] [&.panel-transition--right]:[width:var(--right-panel-width,_50%)] [&.panel-transition--right]:[min-width:0] [&.panel-transition--right]:[min-height:0] [&.panel-transition--right]:[overflow:hidden]",
            bottomPanelOpen && "is-open",
          )}
          aria-hidden={!bottomPanelOpen}
          inert={!bottomPanelOpen}
        >
          <div className="panel-transition__inner [min-width:0] [min-height:0] [overflow:hidden]">
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
              className="ask-in-side-chat-action [position:fixed] [z-index:100] [min-height:30px] [padding:0_10px] [display:inline-flex] [align-items:center] [gap:6px] [color:var(--foreground)] [background:var(--card-strong)] [border:1px_solid_var(--border)] [border-radius:8px] [box-shadow:var(--shadow)] [cursor:pointer] [font-size:10.5px] [&_svg]:[width:14px] [&_svg]:[height:14px] [&_svg]:[color:var(--accent)]"
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
              className="session-context-menu [position:fixed] [z-index:120] [width:200px] [padding:6px] [display:grid] [gap:2px] [color:var(--foreground)] [background:var(--card-solid)] [border:1px_solid_var(--border)] [border-radius:13px] [box-shadow:var(--shadow),_var(--shadow-inset)] [&_button]:[width:100%] [&_button]:[min-height:38px] [&_button]:[padding:7px_9px] [&_button]:[display:flex] [&_button]:[align-items:center] [&_button]:[gap:9px] [&_button]:[color:inherit] [&_button]:[background:transparent] [&_button]:[border:0] [&_button]:[border-radius:8px] [&_button]:[font:inherit] [&_button]:[font-size:13px] [&_button]:[text-align:left] [&_button]:[cursor:pointer] [&_button.is-danger]:[color:var(--danger)] [&_button_svg]:[width:15px] [&_button_svg]:[height:15px]"
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
              <div className="session-context-menu__separator [height:1px] [margin:4px_6px] [background:var(--border)]" />
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
              className="session-rename-backdrop [position:fixed] [z-index:130] [inset:0] [padding:24px] [display:grid] [place-items:center] [background:rgba(0,_0,_0,_0.42)] [-webkit-backdrop-filter:blur(7px)]"
              onMouseDown={(event) => {
                if (event.target === event.currentTarget && !sessionActionPending) {
                  setRenamingSession(null);
                }
              }}
            >
              <dialog
                open
                className="session-rename-dialog [width:min(100%,_560px)] [padding:26px_28px_24px] [color:var(--foreground)] [background:var(--card-solid)] [border:1px_solid_var(--border)] [border-radius:24px] [box-shadow:0_24px_80px_rgba(0,_0,_0,_0.42),_var(--shadow-inset)] [&_>_header]:[display:flex] [&_>_header]:[align-items:center] [&_>_header]:[justify-content:space-between] [&_>_header]:[gap:16px] [&_h2]:[margin:0] [&_h2]:[font-size:25px] [&_h2]:[line-height:1.2] [&_>_header_button]:[width:34px] [&_>_header_button]:[height:34px] [&_>_header_button]:[padding:0] [&_>_header_button]:[display:grid] [&_>_header_button]:[place-items:center] [&_>_header_button]:[color:var(--muted-foreground)] [&_>_header_button]:[background:transparent] [&_>_header_button]:[border:0] [&_>_header_button]:[border-radius:9px] [&_>_header_button]:[cursor:pointer] [&_>_header_svg]:[width:19px] [&_>_header_svg]:[height:19px] [&_>_p]:[margin:12px_0_22px] [&_>_p]:[color:var(--muted-foreground)] [&_>_p]:[font-size:15px] [&_form]:[display:grid] [&_form]:[gap:12px] [&_input]:[width:100%] [&_input]:[height:52px] [&_input]:[padding:0_14px] [&_input]:[color:var(--foreground)] [&_input]:[background:var(--input)] [&_input]:[border:1px_solid_var(--border)] [&_input]:[border-radius:14px] [&_input]:[box-shadow:var(--shadow-inset)] [&_input]:[font:inherit] [&_input]:[font-size:17px] [&_input]:[outline:none] [&_footer]:[margin-top:10px] [&_footer]:[display:flex] [&_footer]:[justify-content:flex-end] [&_footer]:[gap:10px] [&_footer_button]:[min-width:96px] [&_footer_button]:[height:42px] [&_footer_button]:[padding:0_18px] [&_footer_button]:[color:var(--foreground)] [&_footer_button]:[background:transparent] [&_footer_button]:[border:1px_solid_var(--border)] [&_footer_button]:[border-radius:12px] [&_footer_button]:[font:inherit] [&_footer_button]:[font-weight:600] [&_footer_button]:[cursor:pointer] [&_footer_button.is-primary]:[color:var(--primary-foreground,_#09090b)] [&_footer_button.is-primary]:[background:var(--foreground)] [&_footer_button.is-primary]:[border-color:var(--foreground)]"
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
                    <div
                      className="session-rename-dialog__error [color:var(--danger)] [font-size:13px]"
                      role="alert"
                    >
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
              className="project-preview-card [position:fixed] [z-index:200] [width:min(344px,_calc(100vw_-_16px))] [margin:0] [padding:6px] [color:var(--foreground)] [background:var(--card-solid)] [border:1px_solid_var(--border)] [border-radius:12px] [box-shadow:var(--shadow-soft),_var(--shadow-inset)]"
              aria-label={`${previewProject.name} project details`}
              style={{ top: projectPreview.top, left: projectPreview.left }}
              onPointerEnter={cancelProjectPreviewClose}
              onPointerLeave={scheduleProjectPreviewClose}
              onFocus={cancelProjectPreviewClose}
              onBlur={scheduleProjectPreviewClose}
            >
              <div
                className={String.raw`project-preview-card__row project-preview-card__row--title [&_>_svg]:[width:17px] [&_>_svg]:[height:17px] [&_strong]:[min-width:0] [&_strong]:[overflow:hidden] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_strong]:[font-size:15px] [&_strong]:[font-weight:560] [&_>_svg]:[color:var(--foreground)] [&_>_strong]:[color:var(--foreground)] [&_button]:[width:26px] [&_button]:[height:26px] [&_button]:[padding:0] [&_button]:[display:grid] [&_button]:[place-items:center] [&_button]:[color:var(--muted-foreground)] [&_button]:[background:transparent] [&_button]:[border:0] [&_button]:[border-radius:7px] [&_button]:[cursor:pointer] [&_button_svg]:[width:16px] [&_button_svg]:[height:16px] [min-height:28px] [padding:2px_3px] [display:grid] [grid-template-columns:22px_minmax(0,_1fr)] [align-items:center] [gap:8px] [font-size:13.5px] [&.project-preview-card\_\_row--title]:[grid-template-columns:22px_minmax(0,_1fr)_26px]`}
              >
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
              <div
                className={String.raw`project-preview-card__row [&_>_svg]:[width:17px] [&_>_svg]:[height:17px] [&_>_svg]:[color:var(--muted-foreground)] [&_strong]:[min-width:0] [&_strong]:[overflow:hidden] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_strong]:[font-size:15px] [&_strong]:[font-weight:560] [min-height:28px] [padding:2px_3px] [display:grid] [grid-template-columns:22px_minmax(0,_1fr)] [align-items:center] [gap:8px] [font-size:13.5px] [&.project-preview-card\_\_row--title]:[grid-template-columns:22px_minmax(0,_1fr)_26px]`}
              >
                <MessageCircle aria-hidden="true" />
                <span>
                  {previewProjectGroup?.sessions.length ?? 0}{" "}
                  {(previewProjectGroup?.sessions.length ?? 0) === 1 ? "thread" : "threads"}
                </span>
              </div>
              <div className="project-preview-card__separator [height:1px] [margin:2px_4px] [background:var(--border)]" />
              <div
                className={String.raw`project-preview-card__row project-preview-card__row--path [&_>_svg]:[width:17px] [&_>_svg]:[height:17px] [&_>_svg]:[color:var(--muted-foreground)] [&_strong]:[min-width:0] [&_strong]:[overflow:hidden] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_strong]:[font-size:15px] [&_strong]:[font-weight:560] [&_span]:[min-width:0] [&_span]:[overflow:hidden] [&_span]:[text-overflow:ellipsis] [&_span]:[white-space:nowrap] [min-height:28px] [padding:2px_3px] [display:grid] [grid-template-columns:22px_minmax(0,_1fr)] [align-items:center] [gap:8px] [font-size:13.5px] [&.project-preview-card\_\_row--title]:[grid-template-columns:22px_minmax(0,_1fr)_26px]`}
              >
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
  needsModel,
  onConnectModelProvider,
  onSelectPrompt,
}: {
  projectLabel: string;
  rightPanelOpen: boolean;
  needsModel: boolean;
  onConnectModelProvider: () => void;
  onSelectPrompt: (prompt: string) => void;
}) {
  const suggestions = rightPanelOpen ? EMPTY_RUN_SUGGESTIONS.slice(0, 2) : EMPTY_RUN_SUGGESTIONS;

  return (
    <div className="empty-run [min-height:100%] [padding:clamp(36px,_8vh,_92px)_12px_32px] [display:flex] [flex-direction:column] [align-items:center] [justify-content:center] [gap:20px] [text-align:center] [animation:event-enter_var(--duration-med)_var(--ease-out)_both] max-680:[justify-content:flex-start] max-680:[padding-top:34px]">
      <div className="empty-run__mark [width:50px] [height:50px] [color:var(--muted)] [background:transparent] [border-color:var(--border-subtle)] [border-radius:16px] [box-shadow:none] [display:grid] [place-items:center] [border:1px_solid_var(--border)] [&_svg]:[width:17px] [&_svg]:[height:17px]">
        <AppBrandIcon className="empty-run__icon [width:100%] [height:100%] [object-fit:contain]" />
      </div>
      <div className="empty-run__copy [display:flex] [flex-direction:column] [gap:7px] [&_h2]:[margin:0] [&_h2]:[color:var(--foreground)] [&_h2]:[font-size:clamp(24px,_3vw,_32px)] [&_h2]:[font-weight:620] [&_h2]:[letter-spacing:-0.025em] [&_h2]:[line-height:1.16] [&_p]:[margin:0] [&_p]:[color:var(--muted-foreground)] [&_p]:[font-size:12.5px] max-680:[&_h2]:[font-size:22px]">
        <h2>
          {needsModel ? "Connect a model to start" : `What should we build in ${projectLabel}?`}
        </h2>
        <p>
          {needsModel
            ? "SwarmX needs one compatible model before it can run a task."
            : "Choose a starting point or describe anything below."}
        </p>
      </div>
      {needsModel ? (
        <Button onClick={onConnectModelProvider}>Connect model provider</Button>
      ) : (
        <div
          className={cx(
            String.raw`empty-run__suggestions max-860:[max-width:620px] max-860:[grid-template-columns:repeat(2,_minmax(0,_1fr))] max-680:[grid-template-columns:1fr] [width:min(100%,_940px)] [display:grid] [grid-template-columns:repeat(4,_minmax(0,_1fr))] [gap:12px] [&.empty-run\_\_suggestions--right-panel]:[width:min(100%,_520px)] [&.empty-run\_\_suggestions--right-panel]:[grid-template-columns:repeat(2,_minmax(0,_1fr))]`,
            rightPanelOpen &&
              "empty-run__suggestions--right-panel max-680:[width:100%] max-680:[grid-template-columns:1fr]",
          )}
          aria-label="Suggested tasks"
        >
          {suggestions.map((suggestion) => {
            const Icon = suggestion.icon;
            return (
              <button
                key={suggestion.id}
                type="button"
                className={cx(
                  "empty-run__suggestion [min-width:0] [min-height:126px] [padding:16px] [display:flex] [flex-direction:column] [align-items:flex-start] [justify-content:space-between] [gap:18px] [color:var(--foreground)] [background:var(--card)] [border:1px_solid_var(--border-subtle)] [border-radius:15px] [box-shadow:var(--shadow-inset),_0_10px_28px_rgba(0,_0,_0,_0.08)] [text-align:left] [cursor:pointer] [transition:transform_var(--duration-fast)_var(--ease-out),_background-color_var(--duration-fast)_var(--ease-out),_border-color_var(--duration-fast)_var(--ease-out)] [&_svg]:[width:18px] [&_svg]:[height:18px] [&_span]:[font-size:13px] [&_span]:[font-weight:610] [&_span]:[line-height:1.35] [&.is-blue_svg]:[color:#7597ff] [&.is-violet_svg]:[color:#a987ff] [&.is-green_svg]:[color:#34d399] [&.is-orange_svg]:[color:#fb923c] max-680:[min-height:82px] max-680:[flex-direction:row] max-680:[align-items:center] max-680:[justify-content:flex-start]",
                  EMPTY_RUN_SUGGESTION_CLASS[suggestion.tone],
                )}
                onClick={() => onSelectPrompt(suggestion.prompt)}
              >
                <Icon aria-hidden="true" />
                <span>{suggestion.label}</span>
              </button>
            );
          })}
        </div>
      )}
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
    <section
      className="pinned-summary [min-width:0] [min-height:58px] [padding:9px_12px] [display:flex] [align-items:center] [gap:10px] [border-bottom:1px_solid_var(--border-subtle)] [background:var(--card-solid)] [&_>_svg]:[width:15px] [&_>_svg]:[height:15px] [&_>_svg]:[flex:0_0_auto] [&_>_svg]:[color:var(--accent)]"
      aria-label="Pinned summary"
    >
      <Pin aria-hidden="true" />
      <div className="pinned-summary__copy [min-width:0] [flex:1] [display:flex] [flex-direction:column] [gap:1px] [&_strong]:[overflow:hidden] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_span]:[overflow:hidden] [&_span]:[text-overflow:ellipsis] [&_span]:[white-space:nowrap] [&_strong]:[font-size:12.5px] [&_strong]:[line-height:1.2] [&_span]:[color:var(--muted-foreground)] [&_span]:[font-size:10.5px] max-680:[&_span]:[display:none]">
        <strong>{title}</strong>
        <span>{subtitle}</span>
      </div>
      <div className="pinned-summary__meta [display:flex] [align-items:center] [gap:6px] max-680:[display:none]">
        <Badge tone={status === "Running" ? "success" : "neutral"}>{status}</Badge>
        <Badge tone="neutral">{messageCount} events</Badge>
        <Badge tone="neutral">{workflowLabel}</Badge>
      </div>
      <Button variant="ghost" size="icon" onClick={onClose} aria-label="Unpin summary">
        <XCircle aria-hidden="true" />
      </Button>
    </section>
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
      className="gui-contribution-workspace [height:100%] [min-width:0] [min-height:0] [overflow:hidden] [display:grid] [grid-template-rows:58px_minmax(0,_1fr)] [background:rgba(9,_10,_14,_0.72)]"
      aria-label={`${contribution.name} contribution`}
    >
      <div className="gui-contribution-topbar [min-width:0] [padding:0_16px] [display:flex] [align-items:center] [justify-content:space-between] [gap:14px] [border-bottom:1px_solid_var(--border-subtle)] [background:rgba(15,_17,_23,_0.84)] [box-shadow:var(--shadow-inset)]">
        <div className="extension-title [min-width:0] [display:flex] [align-items:center] [gap:10px] [&_>_svg]:[flex:0_0_auto] [&_>_svg]:[width:18px] [&_>_svg]:[height:18px] [&_>_svg]:[color:var(--accent)] [&_h2]:[margin:0] [&_h2]:[color:var(--foreground)] [&_h2]:[font-size:14px] [&_h2]:[font-weight:680] [&_h2]:[line-height:1.2] [&_span]:[display:block] [&_span]:[margin-top:2px] [&_span]:[color:var(--muted-foreground)] [&_span]:[font-size:12px] [&_span]:[line-height:1.2]">
          <Package aria-hidden="true" />
          <div>
            <h2>{contribution.name}</h2>
            <span>{contribution.description ?? contribution.componentRef}</span>
          </div>
        </div>
        <div
          className="extension-stats [justify-content:flex-end] [flex-wrap:wrap] [min-width:0] [display:flex] [align-items:center] [gap:10px] max-680:[width:100%] max-680:[justify-content:flex-start]"
          aria-label="Contribution metadata"
        >
          <Badge tone="neutral">{contribution.kind}</Badge>
          <Badge tone="neutral">{contribution.placement}</Badge>
          {contribution.sourcePluginId && (
            <Badge tone="neutral">{contribution.sourcePluginId}</Badge>
          )}
          {contribution.readOnly && <Badge tone="neutral">read-only</Badge>}
        </div>
      </div>
      <div className="gui-contribution-body [min-width:0] [min-height:0] [overflow:auto] [padding:16px]">
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
