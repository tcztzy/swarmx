import {
  AppUpdateEventContracts,
  AppUpdateInvokeContracts,
  type DesktopUpdateState,
} from "../shared/ipc-contracts/app-update.js";
import type { DesktopIpcRegistrar } from "./ipc-router.js";
import type { DesktopUpdateServiceLike } from "./updater.js";

export function registerAppUpdateIpc(
  registrar: DesktopIpcRegistrar,
  service: DesktopUpdateServiceLike,
): void {
  registrar.register("appUpdate:getState", AppUpdateInvokeContracts["appUpdate:getState"], () =>
    service.getState(),
  );
  registrar.register("appUpdate:install", AppUpdateInvokeContracts["appUpdate:install"], () =>
    service.startUpdate(),
  );
}

export function validatedAppUpdateState(state: unknown): DesktopUpdateState {
  return AppUpdateEventContracts["appUpdate:state"].payload.parse(state);
}
