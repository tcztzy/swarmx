/**
 * Electron entry point: boot the DSH Web application as the sole UI, register
 * the selected peer conversation runtime below it, then load that one origin.
 */
import { app, BrowserWindow } from "electron";
import {
  acquirePrimaryInstance,
  activateOwnedResource,
  createBeforeQuitHandler,
  disposeAfterFailure,
  disposeOwnedResource,
  onceFailureReporter,
} from "./app-lifecycle.js";
import { type DesktopPlatform, startDesktopPlatform } from "./runtime/platform.js";
import { resolveRuntimeSelection } from "./runtime/selection.js";
import { createWindow } from "./window.js";

/** The DSH Web host and its peer runtime registry. */
let platform: DesktopPlatform | undefined;
let platformBoot: Promise<DesktopPlatform> | undefined;

/**
 * A boot failure leaves nothing to show, so it is fatal and reported on stderr
 * rather than swallowed into an empty window.
 */
function failLoud(error: unknown): void {
  process.stderr.write(
    `swarmx: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
  );
  app.exit(1);
}

const reportFatalOnce = onceFailureReporter(failLoud);

if (acquirePrimaryInstance(app, () => BrowserWindow.getAllWindows())) {
  void app
    .whenReady()
    .then(async () => {
      const runtime = resolveRuntimeSelection(process.argv.slice(2), process.env);
      const workspaceRoot = process.env.SWARMX_WORKSPACE ?? process.cwd();
      platformBoot = activateOwnedResource(
        startDesktopPlatform({ runtime, workspaceRoot }),
        (started) => createWindow(started.url),
      );
      platform = await platformBoot;
      // macOS keeps the app alive with no windows; reopening must not re-boot.
      app.on("activate", () => {
        if (BrowserWindow.getAllWindows().length === 0 && platform !== undefined) {
          try {
            createWindow(platform.url);
          } catch (error) {
            void disposeAfterFailure(platform, error).catch(reportFatalOnce);
          }
        }
      });
    })
    .catch(reportFatalOnce);

  // Every platform quits with its last window: a single-window desktop app has
  // nothing to return to, and the harness must not outlive its surface.
  app.on("window-all-closed", () => app.quit());

  // Hold the quit until every owned server and native runtime has released resources.
  app.on(
    "before-quit",
    createBeforeQuitHandler({
      dispose: () =>
        disposeOwnedResource(
          () => platform,
          () => platformBoot,
        ),
      quit: () => app.quit(),
      reportFailure: reportFatalOnce,
    }),
  );
}
