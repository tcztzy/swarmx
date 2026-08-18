#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);

export function codexVersionLine() {
  const modulePackage = require("../package.json");
  const codexPackage = require("@openai/codex/package.json");
  return `${modulePackage.name} ${modulePackage.version} (codex app-server ${codexPackage.version})\n`;
}

export function resolveCodexEntry() {
  return require.resolve("@openai/codex/bin/codex.js");
}

export function spawnCodexAppServer(entry) {
  return spawn(process.execPath, [entry, "app-server"], {
    env: process.env,
    stdio: "inherit",
  });
}

export function waitForChildExit(child) {
  return new Promise((resolve) => {
    child.once("exit", (code) => resolve(code));
  });
}

function isMainModule() {
  return Boolean(process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href);
}

if (isMainModule() && process.argv.includes("--version")) {
  process.stdout.write(codexVersionLine());
} else if (isMainModule()) {
  const child = spawnCodexAppServer(resolveCodexEntry());

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
  const signals = ["SIGINT", "SIGTERM", "SIGHUP"];
  for (const signal of signals) process.on(signal, forwardSignal);

  const exitCode = await waitForChildExit(child);
  for (const signal of signals) process.removeListener(signal, forwardSignal);
  process.exit(exitCode ?? 1);
}
