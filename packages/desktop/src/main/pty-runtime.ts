import { constants, chmodSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

/** Restores execute permission that some package/install paths drop from node-pty's helper. */
export function ensurePtySpawnHelperExecutable(platform: NodeJS.Platform): void {
  if (platform === "win32") return;
  const require = createRequire(import.meta.url);
  const packageRoot = path.resolve(path.dirname(require.resolve("node-pty")), "..");
  const candidates = [
    path.join(packageRoot, "prebuilds", `${platform}-${process.arch}`, "spawn-helper"),
    path.join(packageRoot, "build", "Release", "spawn-helper"),
  ];
  for (const candidate of candidates) {
    try {
      const mode = statSync(candidate).mode;
      if ((mode & constants.S_IXUSR) === 0) chmodSync(candidate, mode | constants.S_IXUSR);
      return;
    } catch {
      // Active node-pty distribution may use the other known helper location.
    }
  }
}
