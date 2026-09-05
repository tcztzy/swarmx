import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { app, BrowserWindow } from "electron";
import { selectedAgent } from "./agent.js";
import { type DesktopPlatform, startDesktopPlatform } from "./platform.js";
import { createWindow } from "./window.js";

let platform: DesktopPlatform | undefined;
let platformBoot: Promise<DesktopPlatform> | undefined;
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
      const workspaceRoot = process.env.SWARMX_WORKSPACE ?? process.cwd();
      const { values } = parseArgs({
        options: { agent: { type: "string" } },
        strict: false,
        allowPositionals: true,
      });
      platformBoot = startDesktopPlatform({
        workspaceRoot,
        agentId: selectedAgent(typeof values.agent === "string" ? values.agent : undefined),
        rendererRoot: fileURLToPath(new URL("./renderer", import.meta.url)),
      });
      const started = await platformBoot;
      try {
        createWindow(started.url);
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
        createWindow(platform.issueLaunchUrl());
      } catch (error) {
        failLoud(error);
      }
    }
  });

  app.on("window-all-closed", () => app.quit());

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
