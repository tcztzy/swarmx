/**
 * Electron entry point: boot the harness, then show its surface in a window.
 * Lifecycle is deliberately narrow — the harness owns sessions, tools,
 * permissions, and persistence, so this process only sequences startup and
 * shutdown around it.
 */
import { app, BrowserWindow } from "electron";
import { type Harness, startHarness } from "./harness.js";
import { createWindow } from "./window.js";

/** The booted harness, once startup completes. */
let harness: Harness | undefined;

/** Dispose the harness exactly once, even when several exit paths fire. */
let shutdown: Promise<void> | undefined;

/** Tear the harness down, releasing its server port and flushing its writes. */
function stopHarness(): Promise<void> {
  shutdown ??= harness === undefined ? Promise.resolve() : harness.ctx.fiber.dispose();
  return shutdown;
}

/**
 * A boot failure leaves nothing to show, so it is fatal and reported on stderr
 * rather than swallowed into an empty window.
 */
function failLoud(error: unknown): never {
  process.stderr.write(
    `swarmx: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
  );
  app.exit(1);
  throw error;
}

app.whenReady().then(async () => {
  harness = await startHarness().catch(failLoud);
  createWindow(harness.url);
  // macOS keeps the app alive with no windows; reopening must not re-boot.
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0 && harness !== undefined) {
      createWindow(harness.url);
    }
  });
}, failLoud);

// Every platform quits with its last window: a single-window desktop app has
// nothing to return to, and the harness must not outlive its surface.
app.on("window-all-closed", () => app.quit());

// Hold the quit until the harness has released its resources.
app.on("before-quit", (event) => {
  if (shutdown === undefined) {
    event.preventDefault();
    void stopHarness().then(() => app.quit());
  }
});
