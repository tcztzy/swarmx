#!/usr/bin/env node

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

export function codexTargetTriple(platform = process.platform, arch = process.arch) {
  if (platform === "linux" && arch === "x64") return "x86_64-unknown-linux-musl";
  if (platform === "linux" && arch === "arm64") return "aarch64-unknown-linux-musl";
  return null;
}

export function codexBinaryPath(runtimeDir, platform = process.platform, arch = process.arch) {
  const triple = codexTargetTriple(platform, arch);
  if (!triple) {
    throw new Error(`Unsupported Codex container platform: ${platform} (${arch})`);
  }
  return join(runtimeDir, "vendor", triple, "bin", "codex");
}

function isMainModule() {
  return Boolean(process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href);
}

if (isMainModule()) {
  const runtimeDir = process.env.SWARMX_CODEX_RUNTIME_DIR?.trim();
  if (!runtimeDir) {
    console.error("SWARMX_CODEX_RUNTIME_DIR must point to the bundled Codex runtime.");
    process.exit(1);
  }

  let binary;
  try {
    binary = codexBinaryPath(runtimeDir);
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }

  if (!existsSync(binary)) {
    console.error(`Codex runtime binary is missing: ${binary}`);
    process.exit(1);
  }

  const child = spawn(binary, ["app-server"], {
    env: process.env,
    stdio: "inherit",
  });

  child.once("error", (error) => {
    console.error(error);
    process.exit(1);
  });

  const forwardSignal = (signal) => {
    try {
      child.kill(signal);
    } catch {
      // The child may already be gone; the exit handler below is authoritative.
    }
  };
  for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
    process.on(signal, () => forwardSignal(signal));
  }

  const exitCode = await new Promise((resolve) => {
    child.once("exit", (code) => resolve(code));
  });
  process.exit(exitCode ?? 1);
}
