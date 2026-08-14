import { AppUpdateEventContracts, AppUpdateInvokeContracts } from "./app-update.js";
import {
  composeContractMaps,
  type DesktopEventContract,
  type DesktopInvokeContract,
} from "./base.js";
import { BrowserEventContracts, BrowserInvokeContracts } from "./browser.js";
import { GlobalMemoryInvokeContracts } from "./global-memory.js";
import { ProjectInvokeContracts } from "./project.js";
import { TaskRuntimeInvokeContracts } from "./task-runtime.js";
import { TerminalEventContracts, TerminalInvokeContracts } from "./terminal.js";
import { WorkspaceInspectionInvokeContracts } from "./workspace-inspection.js";

export * from "./app-update.js";
export * from "./base.js";
export * from "./browser.js";
export * from "./global-memory.js";
export * from "./project.js";
export * from "./task-runtime.js";
export * from "./terminal.js";
export * from "./workspace-inspection.js";

export const DesktopInvokeContractRegistry = composeContractMaps<DesktopInvokeContract>(
  AppUpdateInvokeContracts,
  BrowserInvokeContracts,
  GlobalMemoryInvokeContracts,
  ProjectInvokeContracts,
  TaskRuntimeInvokeContracts,
  TerminalInvokeContracts,
  WorkspaceInspectionInvokeContracts,
);
export const DesktopEventContractRegistry = composeContractMaps<DesktopEventContract>(
  AppUpdateEventContracts,
  BrowserEventContracts,
  TerminalEventContracts,
);
