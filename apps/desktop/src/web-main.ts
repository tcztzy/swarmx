import { fileURLToPath } from "node:url";
import { startDesktopPlatform } from "./platform.js";

const platform = await startDesktopPlatform({
  workspaceRoot: process.env.SWARMX_WORKSPACE ?? process.cwd(),
  rendererRoot: fileURLToPath(new URL("./renderer", import.meta.url)),
});
process.stdout.write(`${platform.url}\n`);
let stopping = false;
const stop = () => {
  if (stopping) return;
  stopping = true;
  void platform.dispose().catch((error: unknown) => {
    process.stderr.write(`${String(error)}\n`);
    process.exitCode = 1;
  });
};
process.once("SIGINT", stop);
process.once("SIGTERM", stop);
