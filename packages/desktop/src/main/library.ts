/** Reusable Electron main-process integration. This entry does not start an app. */
export {
  agentChunkPublisher,
  assertFinalAssistantMessage,
  disposeDesktopTerminals,
  registerIpcHandlers,
} from "./ipc.js";
export type { AgentChunkSender } from "./ipc.js";
export { BrowserHost, normalizeBrowserBounds, normalizeBrowserUrl } from "./browser-host.js";
export type {
  BrowserBounds,
  BrowserOwner,
  BrowserState,
  BrowserViewFactory,
  CreateBrowserRequest,
} from "./browser-host.js";
export {
  HarnessDoctor,
  HarnessEnvironmentService,
  configureDesktopHarnessEnvironment,
} from "./harness-environment.js";
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
  HarnessVersionCheck,
  HarnessProtectionMode,
  HarnessProtectionSummary,
  HarnessRequirementStatus,
  HarnessRuntimeRequirement,
  ProtectedHarnessBackendResult,
} from "./harness-environment.js";
export { LspHost } from "./lsp-host.js";
export type {
  LspCompletionRequest,
  LspCompletionResponse,
  LspStopRequest,
  LspStopResponse,
  LspTextPosition,
} from "./lsp-host.js";
export { DesktopRequestRegistry } from "./request-registry.js";
export type { RequestOwner } from "./request-registry.js";
export { TerminalHost } from "./terminal-host.js";
export type {
  CreateTerminalRequest,
  TerminalOwner,
  TerminalProcessFactory,
} from "./terminal-host.js";
export { NpmDesktopUpdateService, compareSemanticVersions } from "./updater.js";
export type {
  DesktopUpdatePhase,
  DesktopUpdateServiceLike,
  DesktopUpdateState,
  NpmDesktopUpdateServiceOptions,
} from "./updater.js";
export { ModelCatalogService } from "./model-catalog.js";
export type {
  ManualModelInput,
  ModelCatalogInventory,
  ModelCatalogMetadata,
  ModelCatalogProviderStatus,
  ModelCatalogServiceOptions,
  UserProviderInput,
} from "./model-catalog.js";
export { CustomAgentService } from "./custom-agents.js";
export type { SaveCustomAgentOptions } from "./custom-agents.js";
export { DesktopSettingsStore } from "./settings-store.js";
export type {
  DesktopSettingsStoreLike,
  DesktopSettingsStoreOptions,
} from "./settings-store.js";
export { DesktopExtensionManager } from "./extension-manager.js";
export type { ExtensionManagementState } from "./extension-manager.js";
export { EncryptedFileProviderAuthStore } from "./provider-auth.js";
export type {
  EncryptedFileProviderAuthStoreOptions,
  ProviderAuthStore,
  ProviderSecretEncryption,
} from "./provider-auth.js";
export { ProviderUsageService, queryCodexAppServer } from "./provider-usage.js";
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
export { WORKSPACE_TOOLS_DEFAULTS, WorkspaceTools } from "./workspace-tools.js";
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
export {
  WORKSPACE_SHELL_DEFAULTS,
  WorkspaceShell,
  workspaceShellAgentTool,
} from "./workspace-shell.js";
export type {
  WorkspaceShellOptions,
  WorkspaceShellResult,
  WorkspaceShellRunOptions,
} from "./workspace-shell.js";
