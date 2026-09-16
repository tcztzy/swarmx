import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { app, BrowserWindow } from "electron";
import { selectedAgent } from "./agent.js";
import { registerIpc } from "./ipc.js";
import { type DesktopPlatform, startDesktopPlatform } from "./platform.js";
import { createWindow } from "./window.js";

let platform: DesktopPlatform | undefined;
let platformBoot: Promise<DesktopPlatform> | undefined;
let renderer = "";
let failureReported = false;
let quitting = false;
let shutdownStarted = false;

function failLoud(error: unknown): void {
  if (failureReported) return;
  failureReported = true;
  process.stderr.write(
    `swarmx: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
  );
  app.exit(1);
}

function rendererLocation(development: boolean): string {
  if (!development)
    return pathToFileURL(fileURLToPath(new URL("./renderer/index.html", import.meta.url))).href;
  const url = process.env.SWARMX_RENDERER_URL;
  if (!url) throw new Error("SWARMX_RENDERER_URL is required when SWARMX_DEV=1.");
  return url;
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    const window = BrowserWindow.getAllWindows()[0];
    if (window === undefined) return;
    if (window.isMinimized()) window.restore();
    window.show();
    window.focus();
  });

  void app
    .whenReady()
    .then(async () => {
      const cwd = process.env.SWARMX_CWD ?? process.cwd();
      const development = !app.isPackaged && process.env.SWARMX_DEV === "1";
      renderer = rendererLocation(development);
      const { values } = parseArgs({
        options: { agent: { type: "string" } },
        strict: false,
        allowPositionals: true,
      });
      const boot = startDesktopPlatform({
        cwd,
        agentId: selectedAgent(typeof values.agent === "string" ? values.agent : undefined),
      });
      platformBoot = boot;
      const started = await boot;
      registerIpc(started);
      try {
        createWindow(renderer);
      } catch (error) {
        await started.dispose();
        throw error;
      }
      platform = started;
    })
    .catch(failLoud);

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0 && platform !== undefined) {
      try {
        createWindow(renderer);
      } catch (error) {
        failLoud(error);
      }
    }
  });

  app.on("window-all-closed", () => app.quit());
  process.on("SIGINT", () => app.quit());
  process.on("SIGTERM", () => app.quit());

  app.on("before-quit", (event) => {
    if (quitting) return;
    event.preventDefault();
    if (shutdownStarted) return;
    shutdownStarted = true;
    void (
      platformBoot?.then(
        (started) => started.dispose(),
        () => undefined,
      ) ?? Promise.resolve()
    )
      .then(() => {
        quitting = true;
        app.quit();
      })
      .catch(failLoud);
  });
}
