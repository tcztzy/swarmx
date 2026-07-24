import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { configureDesktopHarnessEnvironment } from "@swarmx/runtime";
import { BrowserWindow, type BrowserWindowConstructorOptions, app, nativeTheme } from "electron";
import { disposeDesktopTerminals, registerIpcHandlers } from "./ipc.js";
import { NpmDesktopUpdateService } from "./updater.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

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

const preloadPath = join(__dirname, "../preload/index.mjs");
const rendererUrl =
  process.env.ELECTRON_RENDERER_URL ?? `file://${join(RENDERER_DIST, "index.html")}`;

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
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  };

  mainWindow = new BrowserWindow(opts);
  if (process.platform === "darwin") mainWindow.setWindowButtonVisibility(true);

  mainWindow.on("ready-to-show", () => {
    mainWindow?.show();
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  mainWindow.loadURL(rendererUrl);
}

app.whenReady().then(() => {
  if (process.platform === "darwin") app.dock.setIcon(APP_ICON_PATH);
  registerIpcHandlers({
    updateService: desktopUpdater,
    broadcastUpdateState: (state) => {
      for (const window of BrowserWindow.getAllWindows()) {
        if (!window.isDestroyed()) window.webContents.send("appUpdate:state", state);
      }
    },
  });
  createWindow();
  void desktopUpdater.check();
  updateCheckTimer = setInterval(() => void desktopUpdater.check(), 6 * 60 * 60 * 1_000);
  updateCheckTimer.unref();
});

app.on("before-quit", () => {
  if (updateCheckTimer) clearInterval(updateCheckTimer);
  updateCheckTimer = null;
  disposeDesktopTerminals();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
