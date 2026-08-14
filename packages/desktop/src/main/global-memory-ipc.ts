import { GlobalMemoryInvokeContracts } from "../shared/ipc-contracts/global-memory.js";
import type { GlobalMemoryServiceLike } from "./global-memory-service.js";
import type { DesktopIpcRegistrar } from "./ipc-router.js";

export type GlobalMemoryIpcService = Pick<GlobalMemoryServiceLike, "get" | "save" | "forget">;

export function registerGlobalMemoryIpc(
  registrar: DesktopIpcRegistrar,
  service: GlobalMemoryIpcService,
): void {
  registrar.register("personalMemory:get", GlobalMemoryInvokeContracts["personalMemory:get"], () =>
    service.get(),
  );
  registrar.register(
    "personalMemory:save",
    GlobalMemoryInvokeContracts["personalMemory:save"],
    (_event, [input]) => service.save(input),
  );
  registrar.register(
    "personalMemory:forget",
    GlobalMemoryInvokeContracts["personalMemory:forget"],
    (_event, [input]) => service.forget(input),
  );
}
