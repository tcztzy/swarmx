/** Reusable Electron main-process integration. This entry does not start an app. */

export type {
  ContainerRuntimeId,
  ContainerRuntimeStatus,
  DoctorFixOptions,
  DoctorFixResult,
  DoctorInspectOptions,
  DoctorIssue,
  DoctorIssueSeverity,
  DoctorRepairAction,
  DoctorRepairPlan,
  DoctorRepairRisk,
  DoctorReport,
  HarnessContainerRuntime,
  HarnessEnvironmentHarness,
  HarnessEnvironmentHarnessState,
  HarnessEnvironmentHost,
  HarnessEnvironmentSetupRequest,
  HarnessEnvironmentSetupResult,
  HarnessEnvironmentStatus,
  HarnessProtectionMode,
  HarnessProtectionSummary,
  HarnessRequirementStatus,
  HarnessRuntimeRequirement,
  HarnessVersionCheck,
  ProtectedHarnessBackendResult,
} from "@swarmx/runtime";
export {
  configureDesktopHarnessEnvironment,
  HarnessDoctor,
  HarnessEnvironmentService,
} from "@swarmx/runtime";
export type { AgentChunkSender } from "./agent-chunk-publisher.js";
export { agentChunkPublisher } from "./agent-chunk-publisher.js";
export { AgentInteractionBroker } from "./agent-interactions.js";
export type {
  BrowserBounds,
  BrowserOwner,
  BrowserState,
  BrowserViewFactory,
  CreateBrowserRequest,
} from "./browser-host.js";
export { BrowserHost, normalizeBrowserBounds, normalizeBrowserUrl } from "./browser-host.js";
export { ComposerPreferenceService } from "./composer-preferences.js";
export type { SaveCustomAgentOptions } from "./custom-agents.js";
export { CustomAgentService } from "./custom-agents.js";
export type { ExtensionManagementState } from "./extension-manager.js";
export { DesktopExtensionManager } from "./extension-manager.js";
export {
  assertFinalAssistantMessage,
  disposeDesktopTerminals,
  registerIpcHandlers,
} from "./ipc.js";
export type {
  LspCompletionRequest,
  LspCompletionResponse,
  LspStopRequest,
  LspStopResponse,
  LspTextPosition,
} from "./lsp-host.js";
export { LspHost } from "./lsp-host.js";
export { MemoryRuntimeBackend } from "./memory-runtime-backend.js";
export type {
  MemoryRuntimeConnection,
  MemoryRuntimeHostOptions,
} from "./memory-runtime-host.js";
export { MemoryRuntimeHost } from "./memory-runtime-host.js";
export { MemoryRuntimeService } from "./memory-runtime-service.js";
export type {
  ManualModelInput,
  ModelCatalogInventory,
  ModelCatalogMetadata,
  ModelCatalogProviderStatus,
  ModelCatalogServiceOptions,
  UserProviderInput,
} from "./model-catalog.js";
export { ModelCatalogService } from "./model-catalog.js";
export type {
  PermissionAutoReviewerOptions,
  PermissionReviewRequest,
  PermissionReviewResult,
} from "./permission-review.js";
export { PermissionAutoReviewer } from "./permission-review.js";
export type {
  DesktopPermissionStatus,
  PermissionLayerStatus,
  PermissionServiceOptions,
  RecordPermissionDecisionInput,
  ResolveDesktopPermissionOptions,
} from "./permission-service.js";
export { PermissionService } from "./permission-service.js";
export type { PersonalMemoryServiceLike } from "./personal-memory.js";
export { PersonalMemoryService } from "./personal-memory.js";
export type {
  FileProviderAuthStoreOptions,
  ProviderAuthStore,
} from "./provider-auth.js";
export { FileProviderAuthStore } from "./provider-auth.js";
export type {
  ProviderBalanceUsageMeter,
  ProviderCreditUsageMeter,
  ProviderUsageEntry,
  ProviderUsageMeter,
  ProviderUsageServiceOptions,
  ProviderUsageSnapshot,
  ProviderUsageStatus,
  ProviderWindowUsageMeter,
} from "./provider-usage.js";
export { ProviderUsageService, queryCodexAppServer } from "./provider-usage.js";
export type {
  ReferenceLibraryConnection,
  ReferenceLibraryHostOptions,
} from "./reference-library-host.js";
export { ReferenceLibraryHost } from "./reference-library-host.js";
export type { RequestOwner } from "./request-registry.js";
export { DesktopRequestRegistry } from "./request-registry.js";
export type {
  DesktopSettingsStoreLike,
  DesktopSettingsStoreOptions,
} from "./settings-store.js";
export { DesktopSettingsStore } from "./settings-store.js";
export type {
  CreateSideChatInput,
  SideChatParentState,
  UpdateSideChatInput,
} from "./side-chat-service.js";
export { SideChatService } from "./side-chat-service.js";
export type {
  DesktopTaskSupervisorLike,
  DesktopTaskSupervisorOptions,
  TaskSupervisorSuccessResponse,
} from "./task-supervisor.js";
export { DesktopTaskSupervisor } from "./task-supervisor.js";
export type {
  CreateTerminalRequest,
  TerminalOwner,
  TerminalProcessFactory,
} from "./terminal-host.js";
export { TerminalHost } from "./terminal-host.js";
export type {
  DesktopUpdatePhase,
  DesktopUpdateServiceLike,
  DesktopUpdateState,
  NpmDesktopUpdateServiceOptions,
} from "./updater.js";
export { compareSemanticVersions, NpmDesktopUpdateService } from "./updater.js";
export type {
  WorkspaceShellOptions,
  WorkspaceShellResult,
  WorkspaceShellRunOptions,
} from "./workspace-shell.js";
export {
  WORKSPACE_SHELL_DEFAULTS,
  WorkspaceShell,
  workspaceShellAgentTool,
} from "./workspace-shell.js";
export type {
  WorkspaceDirectoryEntry,
  WorkspaceDirectoryListing,
  WorkspaceEditResult,
  WorkspaceReviewFile,
  WorkspaceReviewSnapshot,
  WorkspaceTextFile,
  WorkspaceToolsOptions,
  WorkspaceWriteResult,
} from "./workspace-tools.js";
export { WORKSPACE_TOOLS_DEFAULTS, WorkspaceTools } from "./workspace-tools.js";
