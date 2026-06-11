import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { BrowserWindow, type BrowserWindowConstructorOptions, app } from "electron";
import { registerIpcHandlers } from "./ipc.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

process.env.APP_ROOT = join(__dirname, "..");

const MAIN_DIST = join(__dirname, "..");
const RENDERER_DIST = join(__dirname, "../renderer");

const preloadPath = join(__dirname, "../preload/index.mjs");
const rendererUrl =
  process.env.ELECTRON_RENDERER_URL ?? `file://${join(RENDERER_DIST, "index.html")}`;

let mainWindow: BrowserWindow | null = null;

function createWindow(): void {
  const opts: BrowserWindowConstructorOptions = {
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    title: "SwarmX",
    show: false,
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  };

  mainWindow = new BrowserWindow(opts);

  mainWindow.on("ready-to-show", () => {
    mainWindow?.show();
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  mainWindow.loadURL(rendererUrl);
}

app.whenReady().then(() => {
  registerIpcHandlers();
  createWindow();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
