#!/usr/bin/env node
import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const args = process.argv.slice(2);
const desktopRequested = args.length === 0 || args[0] === "desktop";

if (!desktopRequested) {
  if (args[0] === "cli") process.argv.splice(2, 1);

  if (process.env.SWARMX_LAUNCHER_DRY_RUN === "1") {
    process.stdout.write(`${JSON.stringify({ mode: "cli", args: process.argv.slice(2) })}\n`);
  } else {
    await import("@swarmx/cli");
  }
} else {
  const desktopArgs = args[0] === "desktop" ? args.slice(1) : args;
  const require = createRequire(import.meta.url);
  try {
    const desktopManifest = require.resolve("@swarmx/desktop/package.json");
    const electronPath = await resolveElectronPath(require);
    const appPath = dirname(desktopManifest);

    if (process.env.SWARMX_LAUNCHER_DRY_RUN === "1") {
      process.stdout.write(
        `${JSON.stringify({ mode: "desktop", electronPath, appPath, args: desktopArgs })}\n`,
      );
    } else {
      const child = spawn(electronPath, [appPath, ...desktopArgs], { stdio: "inherit" });
      child.once("error", (error) => {
        console.error(`Failed to launch SwarmX Desktop: ${error.message}`);
        process.exitCode = 1;
      });
      child.once("exit", (code, signal) => {
        if (signal) {
          console.error(`SwarmX Desktop exited after signal ${signal}.`);
          process.exitCode = 1;
        } else {
          process.exitCode = code ?? 1;
        }
      });
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`Failed to launch SwarmX Desktop: ${message}`);
    console.error("Download the macOS app: https://github.com/tcztzy/swarmx/releases/latest");
    process.exitCode = 1;
  }
}

async function resolveElectronPath(require) {
  const electronEntry = require.resolve("electron");
  try {
    return require(electronEntry);
  } catch {
    const electronRoot = dirname(require.resolve("electron/package.json"));
    console.error("Completing the Electron runtime download for first launch...");
    await bootstrapElectronRuntime(electronRoot);
    repairElectronPathMarker(electronRoot);
    delete require.cache[electronEntry];
    return require(electronEntry);
  }
}

async function bootstrapElectronRuntime(electronRoot) {
  const keepAlive = setInterval(() => {}, 1_000);
  try {
    const electronRequire = createRequire(join(electronRoot, "install.js"));
    const { downloadArtifact } = electronRequire("@electron/get");
    const manifest = JSON.parse(readFileSync(join(electronRoot, "package.json"), "utf8"));
    const checksums = JSON.parse(readFileSync(join(electronRoot, "checksums.json"), "utf8"));
    const archivePath = await downloadArtifact({
      version: manifest.version,
      artifactName: "electron",
      platform: process.env.npm_config_platform ?? process.platform,
      arch: process.env.npm_config_arch ?? process.arch,
      cacheRoot: process.env.electron_config_cache,
      checksums,
    });
    const distRoot = join(electronRoot, "dist");
    rmSync(distRoot, { recursive: true, force: true });
    mkdirSync(distRoot, { recursive: true });
    if (process.platform === "darwin") {
      execFileSync("ditto", ["-x", "-k", archivePath, distRoot], { stdio: "inherit" });
    } else {
      const extract = electronRequire("extract-zip");
      await extract(archivePath, { dir: distRoot });
    }
  } finally {
    clearInterval(keepAlive);
  }
}

function repairElectronPathMarker(electronRoot) {
  const platformPath = {
    darwin: "Electron.app/Contents/MacOS/Electron",
    freebsd: "electron",
    linux: "electron",
    openbsd: "electron",
    win32: "electron.exe",
  }[process.env.npm_config_platform ?? process.platform];
  if (!platformPath) throw new Error(`Electron does not support platform ${process.platform}.`);

  const manifest = JSON.parse(readFileSync(join(electronRoot, "package.json"), "utf8"));
  const installedVersion = readFileSync(join(electronRoot, "dist", "version"), "utf8")
    .trim()
    .replace(/^v/, "");
  const executablePath = join(electronRoot, "dist", platformPath);
  if (installedVersion !== manifest.version || !existsSync(executablePath)) {
    throw new Error("Electron runtime setup did not produce the expected version and executable.");
  }
  writeFileSync(join(electronRoot, "path.txt"), platformPath);
}
