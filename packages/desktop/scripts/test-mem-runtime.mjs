#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = resolve(packageRoot, "../..");
const buildScript = join(packageRoot, "scripts/build-mem-runtime.mjs");
const manifestPath = join(packageRoot, "build/mem-runtime/manifest.json");
const integrationTest = join(packageRoot, "src/main/memory-runtime-integration.test.ts");
const pnpm = process.platform === "win32" ? "pnpm.cmd" : "pnpm";

execFileSync(process.execPath, [buildScript], {
  cwd: repositoryRoot,
  stdio: "inherit",
});
execFileSync(pnpm, ["vitest", "run", integrationTest], {
  cwd: repositoryRoot,
  env: {
    ...process.env,
    SWARMX_MEMORY_RUNTIME_MANIFEST: manifestPath,
  },
  stdio: "inherit",
});
