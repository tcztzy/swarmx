import { ProjectInvokeContracts } from "../shared/ipc-contracts/project.js";
import type { DesktopIpcRegistrar } from "./ipc-router.js";
import type { ProjectServiceLike } from "./project-service.js";

export function registerProjectIpc(
  registrar: DesktopIpcRegistrar,
  service: ProjectServiceLike,
): void {
  registrar.register("project:list", ProjectInvokeContracts["project:list"], () => service.list());
  registrar.register("project:addExisting", ProjectInvokeContracts["project:addExisting"], () =>
    service.addExisting(),
  );
  registrar.register("project:createScratch", ProjectInvokeContracts["project:createScratch"], () =>
    service.createScratch(),
  );
  registrar.register(
    "project:setPinned",
    ProjectInvokeContracts["project:setPinned"],
    (_event, [{ id, pinned }]) => service.setPinned(id, pinned),
  );
  registrar.register(
    "project:rename",
    ProjectInvokeContracts["project:rename"],
    (_event, [{ id, name }]) => service.rename(id, name),
  );
  registrar.register(
    "project:reveal",
    ProjectInvokeContracts["project:reveal"],
    (_event, [{ id }]) => service.reveal(id),
  );
  registrar.register(
    "project:archiveTasks",
    ProjectInvokeContracts["project:archiveTasks"],
    (_event, [{ id }]) => service.archiveTasks(id),
  );
  registrar.register(
    "project:remove",
    ProjectInvokeContracts["project:remove"],
    (_event, [{ id }]) => service.remove(id),
  );
}
