import { TaskSupervisorServer, taskSupervisorPaths } from "@swarmx/core";

const paths = taskSupervisorPaths(process.env.SWARMX_TASK_RUNTIME_ROOT);
const supervisor = new TaskSupervisorServer({ rootDir: paths.rootDir });

try {
  await supervisor.listen();
} catch (error) {
  if (!isAddressInUse(error)) throw error;
  process.exit(0);
}

let closing = false;
const close = () => {
  if (closing) return;
  closing = true;
  void supervisor.close().finally(() => process.exit(0));
};
process.once("SIGINT", close);
process.once("SIGTERM", close);

function isAddressInUse(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "EADDRINUSE";
}
