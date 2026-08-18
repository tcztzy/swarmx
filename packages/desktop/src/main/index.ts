import { writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { configureDesktopHarnessEnvironment } from "@swarmx/runtime";
import {
  app,
  BrowserWindow,
  type BrowserWindowConstructorOptions,
  nativeTheme,
  net,
  protocol,
  shell,
} from "electron";
import {
  disposeDesktopCoreRuntime,
  disposeDesktopTerminals,
  registerIpcHandlers,
  resolveDesktopMediaProtocolUrl,
} from "./ipc.js";
import { MemoryRuntimeService } from "./memory-runtime-service.js";
import { ReferenceLibraryHost } from "./reference-library-host.js";
import { NpmDesktopUpdateService } from "./updater.js";
import {
  installMainWindowNavigationGuards,
  isTrustedRendererIpcEvent,
  secureMainWindowWebPreferences,
} from "./window-security.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const desktopSmoke = process.env.SWARMX_DESKTOP_SMOKE === "1";

app.setName("SwarmX");
protocol.registerSchemesAsPrivileged([
  {
    scheme: "swarmx-media",
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      stream: true,
    },
  },
]);

process.env.APP_ROOT = join(__dirname, "..");
configureDesktopHarnessEnvironment();
const requestedTheme = process.env.SWARMX_THEME;
if (requestedTheme === "light" || requestedTheme === "dark" || requestedTheme === "system") {
  nativeTheme.themeSource = requestedTheme;
}

const MAIN_DIST = join(__dirname, "..");
const RENDERER_DIST = join(__dirname, "../renderer");
const APP_ICON_PATH = app.isPackaged
  ? join(process.resourcesPath, "icon.png")
  : join(MAIN_DIST, "../build/icon.png");
const MEMORY_RUNTIME_MANIFEST_PATH = app.isPackaged
  ? join(process.resourcesPath, "mem-runtime", "manifest.json")
  : join(MAIN_DIST, "../build/mem-runtime/manifest.json");
const memoryRuntime = new MemoryRuntimeService({
  manifestPath: MEMORY_RUNTIME_MANIFEST_PATH,
  memoryRoot: join(homedir(), ".swarmx", "memory"),
});
const referenceZoteroEnabled = process.env.SWARMX_REFERENCE_ZOTERO === "1";
const referenceLibraryRuntime =
  process.env.SWARMX_REFERENCE_PYTHON &&
  (process.env.SWARMX_REFERENCE_ZIM || referenceZoteroEnabled)
    ? new ReferenceLibraryHost({
        pythonPath: resolve(process.env.SWARMX_REFERENCE_PYTHON),
        ...(process.env.SWARMX_REFERENCE_ZIM
          ? { zimPath: resolve(process.env.SWARMX_REFERENCE_ZIM) }
          : {}),
        zotero: referenceZoteroEnabled,
      })
    : undefined;

const preloadPath = join(__dirname, "../preload/index.mjs");
const rendererUrl =
  process.env.ELECTRON_RENDERER_URL ??
  process.env.VITE_DEV_SERVER_URL ??
  `file://${join(RENDERER_DIST, "index.html")}`;

let mainWindow: BrowserWindow | null = null;
let updateCheckTimer: ReturnType<typeof setInterval> | null = null;

const desktopUpdater = new NpmDesktopUpdateService({
  currentVersion: app.getVersion(),
  supported: Boolean(process.defaultApp) && !app.isPackaged,
  restart: (appPath) => {
    app.relaunch({
      execPath: process.execPath,
      args: [appPath, ...process.argv.slice(2)],
    });
    app.exit(0);
  },
});

function createWindow(): void {
  const opts: BrowserWindowConstructorOptions = {
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    title: "SwarmX",
    show: false,
    backgroundColor: "#07080b",
    ...(process.platform === "darwin" ? {} : { icon: APP_ICON_PATH }),
    ...(process.platform === "darwin"
      ? {
          frame: false,
          titleBarStyle: "hidden" as const,
          trafficLightPosition: { x: 16, y: 17 },
        }
      : {}),
    webPreferences: secureMainWindowWebPreferences(preloadPath),
  };

  mainWindow = new BrowserWindow(opts);
  installMainWindowNavigationGuards(mainWindow.webContents, rendererUrl, (url) =>
    shell.openExternal(url),
  );
  if (process.platform === "darwin") mainWindow.setWindowButtonVisibility(true);

  mainWindow.on("ready-to-show", () => {
    if (!desktopSmoke) mainWindow?.show();
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  if (desktopSmoke) {
    mainWindow.webContents.once("did-fail-load", (_event, code, description) => {
      void finishDesktopSmoke({
        ok: false,
        error: `Renderer load failed (${code}): ${description}`,
      });
    });
    mainWindow.webContents.once("did-finish-load", () => {
      void runDesktopSmoke(mainWindow as BrowserWindow);
    });
  }

  mainWindow.loadURL(rendererUrl);
}

async function runDesktopSmoke(window: BrowserWindow): Promise<void> {
  try {
    const wide = (await window.webContents.executeJavaScript(`
      (async () => {
        const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
        const waitFor = async (selector) => {
          for (let attempt = 0; attempt < 100; attempt += 1) {
            const element = document.querySelector(selector);
            if (element) return element;
            await wait(50);
          }
          throw new Error("Timed out waiting for " + selector);
        };
        const rightToggle = await waitFor('[aria-label="Show right panel"]');
        const agentTrigger = await waitFor('[aria-label="Choose agent"]');
        await waitFor('.runtime');
        rightToggle.click();
        await waitFor('.runtime__body--right-panel');
        await wait(100);
        const runtime = document.querySelector('.runtime');
        const body = document.querySelector('.runtime__body');
        const transcript = document.querySelector('.transcript');
        const rightPanel = document.querySelector('.panel-transition--right.is-open');
        const composer = document.querySelector('.composer-dock');
        const separator = document.querySelector('[aria-label="Resize right panel"]');
        agentTrigger.click();
        await waitFor('.agent-picker__secondary');
        await wait(50);
        const menu = document.querySelector('.agent-picker__menu');
        const primary = document.querySelector('.agent-picker__primary');
        const secondary = document.querySelector('.agent-picker__secondary');
        const runtimeRect = runtime.getBoundingClientRect();
        const bodyRect = body.getBoundingClientRect();
        const composerRect = composer.getBoundingClientRect();
        const primaryRect = primary.getBoundingClientRect();
        const secondaryRect = secondary.getBoundingClientRect();
        const menuRect = menu.getBoundingClientRect();
        return {
          title: document.title,
          stylesheetLoaded: Array.from(document.styleSheets).some((sheet) =>
            String(sheet.href ?? '').includes('swarmx.css')
          ),
          runtimeWidth: runtimeRect.width,
          bodyPaddingRight: Number.parseFloat(getComputedStyle(body).paddingRight),
          rightPanelWidth: rightPanel.getBoundingClientRect().width,
          composerWidth: composerRect.width,
          composerFloatsInBody:
            composer.parentElement === body &&
            getComputedStyle(composer).position === 'absolute' &&
            Math.abs(composerRect.bottom - bodyRect.bottom) <= 2,
          transcriptPaddingBottom: Number.parseFloat(getComputedStyle(transcript).paddingBottom),
          composerHeight: composerRect.height,
          separatorCursor: getComputedStyle(separator).cursor,
          agentMenuWidth: menuRect.width,
          agentPrimaryWidth: primaryRect.width,
          agentPanelsSeparated:
            secondaryRect.left >= primaryRect.right || secondaryRect.right <= primaryRect.left,
          agentPanelsBottomAligned: Math.abs(primaryRect.bottom - secondaryRect.bottom) <= 2,
          agentSecondaryOverflow: getComputedStyle(secondary).overflowY,
        };
      })()
    `)) as Record<string, unknown>;

    const runtimeWidth = numberField(wide, "runtimeWidth");
    const bodyPaddingRight = numberField(wide, "bodyPaddingRight");
    const rightPanelWidth = numberField(wide, "rightPanelWidth");
    const composerWidth = numberField(wide, "composerWidth");
    const transcriptPaddingBottom = numberField(wide, "transcriptPaddingBottom");
    const composerHeight = numberField(wide, "composerHeight");
    const agentMenuWidth = numberField(wide, "agentMenuWidth");
    const agentPrimaryWidth = numberField(wide, "agentPrimaryWidth");
    const checks = {
      title: wide.title === "SwarmX",
      stylesheet: wide.stylesheetLoaded === true,
      equalRightPanel:
        Math.abs(bodyPaddingRight - runtimeWidth / 2) <= 2 &&
        Math.abs(rightPanelWidth - runtimeWidth / 2) <= 2 &&
        Math.abs(composerWidth - runtimeWidth / 2) <= 2,
      floatingComposer:
        wide.composerFloatsInBody === true && transcriptPaddingBottom >= composerHeight + 20,
      resizeAffordance: wide.separatorCursor === "col-resize",
      agentPrimaryOwnsFlow: Math.abs(agentMenuWidth - agentPrimaryWidth) <= 2,
      agentPanelsSeparated: wide.agentPanelsSeparated === true,
      agentPanelsBottomAligned: wide.agentPanelsBottomAligned === true,
      agentSecondaryScrollable: wide.agentSecondaryOverflow === "auto",
    };
    const failed = Object.entries(checks).filter(([, passed]) => !passed);
    if (failed.length > 0) {
      throw new Error(`Desktop smoke checks failed: ${failed.map(([name]) => name).join(", ")}`);
    }
    await finishDesktopSmoke({ ok: true, checks, metrics: wide });
  } catch (error) {
    await finishDesktopSmoke({
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

async function finishDesktopSmoke(report: Record<string, unknown>): Promise<void> {
  const reportPath = process.env.SWARMX_DESKTOP_SMOKE_REPORT;
  if (reportPath) await writeFile(reportPath, `${JSON.stringify(report)}\n`, { mode: 0o600 });
  if (report.ok !== true) console.error(String(report.error ?? "Desktop smoke failed."));
  app.exit(report.ok === true ? 0 : 1);
}

function numberField(record: Record<string, unknown>, key: string): number {
  const value = record[key];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Desktop smoke returned an invalid ${key}.`);
  }
  return value;
}

app.whenReady().then(() => {
  protocol.handle("swarmx-media", async (request) => {
    try {
      const filePath = await resolveDesktopMediaProtocolUrl(request.url);
      return net.fetch(pathToFileURL(filePath).href, { headers: request.headers });
    } catch {
      return new Response("Media preview unavailable.", { status: 404 });
    }
  });
  if (process.platform === "darwin") app.dock.setIcon(APP_ICON_PATH);
  registerIpcHandlers({
    authorizeIpcSender: (event) => isTrustedRendererIpcEvent(event, rendererUrl),
    updateService: desktopUpdater,
    memoryBackend: memoryRuntime,
    ...(referenceLibraryRuntime ? { referenceLibraryBackend: referenceLibraryRuntime } : {}),
    broadcastUpdateState: (state) => {
      for (const window of BrowserWindow.getAllWindows()) {
        if (!window.isDestroyed()) window.webContents.send("appUpdate:state", state);
      }
    },
  });
  createWindow();
  if (!desktopSmoke) {
    void desktopUpdater.check();
    updateCheckTimer = setInterval(() => void desktopUpdater.check(), 6 * 60 * 60 * 1_000);
    updateCheckTimer.unref();
  }
});

app.on("before-quit", () => {
  if (updateCheckTimer) clearInterval(updateCheckTimer);
  updateCheckTimer = null;
  disposeDesktopTerminals();
  void disposeDesktopCoreRuntime();
  void memoryRuntime.close();
  void referenceLibraryRuntime?.close();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
