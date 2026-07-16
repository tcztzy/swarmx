import { spawn } from "node:child_process";
import { createHash, timingSafeEqual } from "node:crypto";
import { mkdir, open, readFile, rename, rm } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

const DESKTOP_PACKAGE_NAME = "@swarmx/desktop";
const NPM_LATEST_URL = "https://registry.npmjs.org/@swarmx%2Fdesktop/latest";
const NPM_REGISTRY_HOST = "registry.npmjs.org";
const DEFAULT_CHECK_TIMEOUT_MS = 8_000;
const DEFAULT_DOWNLOAD_TIMEOUT_MS = 120_000;
const DEFAULT_RESTART_DELAY_MS = 450;

export type DesktopUpdatePhase =
  | "hidden"
  | "available"
  | "downloading"
  | "installing"
  | "restarting";

export interface DesktopUpdateState {
  phase: DesktopUpdatePhase;
  currentVersion: string;
  latestVersion?: string;
  progress?: number;
  error?: string;
}

export interface DesktopUpdateServiceLike {
  getState(): DesktopUpdateState;
  check(): Promise<DesktopUpdateState>;
  startUpdate(): Promise<DesktopUpdateState>;
  subscribe(listener: (state: DesktopUpdateState) => void): () => void;
}

interface NpmDesktopRelease {
  version: string;
  tarballUrl: string;
  integrity: string;
}

type UpdateFetch = (input: string | URL, init?: RequestInit) => Promise<Response>;
type UpdateCommandRunner = (command: string, args: string[]) => Promise<void>;
type UpdateRestart = (appPath: string) => Promise<void> | void;

export interface NpmDesktopUpdateServiceOptions {
  currentVersion: string;
  supported: boolean;
  updatesRoot?: string;
  fetch?: UpdateFetch;
  runCommand?: UpdateCommandRunner;
  restart: UpdateRestart;
  npmCommand?: string;
  checkTimeoutMs?: number;
  downloadTimeoutMs?: number;
  restartDelayMs?: number;
}

export class NpmDesktopUpdateService implements DesktopUpdateServiceLike {
  private readonly currentVersion: string;
  private readonly supported: boolean;
  private readonly updatesRoot: string;
  private readonly fetch: UpdateFetch;
  private readonly runCommand: UpdateCommandRunner;
  private readonly restart: UpdateRestart;
  private readonly npmCommand: string;
  private readonly checkTimeoutMs: number;
  private readonly downloadTimeoutMs: number;
  private readonly restartDelayMs: number;
  private readonly listeners = new Set<(state: DesktopUpdateState) => void>();
  private state: DesktopUpdateState;
  private release: NpmDesktopRelease | null = null;
  private checkPromise: Promise<DesktopUpdateState> | null = null;
  private updatePromise: Promise<DesktopUpdateState> | null = null;

  constructor(options: NpmDesktopUpdateServiceOptions) {
    this.currentVersion = options.currentVersion;
    this.supported = options.supported;
    this.updatesRoot = options.updatesRoot ?? join(homedir(), ".swarmx", "desktop-updates");
    this.fetch = options.fetch ?? globalThis.fetch;
    this.runCommand = options.runCommand ?? runUpdateCommand;
    this.restart = options.restart;
    this.npmCommand = options.npmCommand ?? (process.platform === "win32" ? "npm.cmd" : "npm");
    this.checkTimeoutMs = options.checkTimeoutMs ?? DEFAULT_CHECK_TIMEOUT_MS;
    this.downloadTimeoutMs = options.downloadTimeoutMs ?? DEFAULT_DOWNLOAD_TIMEOUT_MS;
    this.restartDelayMs = options.restartDelayMs ?? DEFAULT_RESTART_DELAY_MS;
    this.state = { phase: "hidden", currentVersion: this.currentVersion };
  }

  getState(): DesktopUpdateState {
    return { ...this.state };
  }

  subscribe(listener: (state: DesktopUpdateState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  check(): Promise<DesktopUpdateState> {
    if (!this.supported || isBusyPhase(this.state.phase)) {
      return Promise.resolve(this.getState());
    }
    if (this.checkPromise) return this.checkPromise;

    this.checkPromise = this.checkForUpdate().finally(() => {
      this.checkPromise = null;
    });
    return this.checkPromise;
  }

  startUpdate(): Promise<DesktopUpdateState> {
    if (this.updatePromise) return this.updatePromise;
    if (!this.supported || this.state.phase !== "available" || !this.release) {
      return Promise.resolve(this.getState());
    }

    this.updatePromise = this.downloadInstallAndRestart(this.release).finally(() => {
      this.updatePromise = null;
    });
    return this.updatePromise;
  }

  private async checkForUpdate(): Promise<DesktopUpdateState> {
    const previous = this.getState();
    try {
      const response = await fetchWithTimeout(this.fetch, NPM_LATEST_URL, this.checkTimeoutMs, {
        headers: { accept: "application/json" },
      });
      if (!response.ok) throw new Error(`npm update check failed (${response.status}).`);
      const release = parseNpmDesktopRelease(await response.json());
      if (compareSemanticVersions(release.version, this.currentVersion) <= 0) {
        this.release = null;
        return this.publish({ phase: "hidden", currentVersion: this.currentVersion });
      }

      this.release = release;
      return this.publish({
        phase: "available",
        currentVersion: this.currentVersion,
        latestVersion: release.version,
      });
    } catch {
      if (previous.phase === "available") return previous;
      this.release = null;
      return this.publish({ phase: "hidden", currentVersion: this.currentVersion });
    }
  }

  private async downloadInstallAndRestart(release: NpmDesktopRelease): Promise<DesktopUpdateState> {
    const versionRoot = join(this.updatesRoot, release.version);
    const stagingRoot = `${versionRoot}.staging-${process.pid}`;
    const downloadsRoot = join(this.updatesRoot, "downloads");
    const tarballPath = join(downloadsRoot, `desktop-${release.version}.tgz`);
    const partialPath = `${tarballPath}.part`;

    try {
      await mkdir(downloadsRoot, { recursive: true });
      await rm(partialPath, { force: true });
      this.publish({
        phase: "downloading",
        currentVersion: this.currentVersion,
        latestVersion: release.version,
        progress: 0,
      });
      await this.downloadRelease(release, partialPath);
      await rm(tarballPath, { force: true });
      await rename(partialPath, tarballPath);

      this.publish({
        phase: "installing",
        currentVersion: this.currentVersion,
        latestVersion: release.version,
        progress: 100,
      });
      await rm(stagingRoot, { recursive: true, force: true });
      await mkdir(stagingRoot, { recursive: true });
      await this.runCommand(this.npmCommand, [
        "install",
        "--prefix",
        stagingRoot,
        "--omit=dev",
        "--no-audit",
        "--no-fund",
        "--ignore-scripts",
        "--package-lock=false",
        tarballPath,
      ]);

      const stagedAppPath = join(stagingRoot, "node_modules", "@swarmx", "desktop");
      await assertInstalledDesktopVersion(stagedAppPath, release.version);
      await rm(versionRoot, { recursive: true, force: true });
      await rename(stagingRoot, versionRoot);
      const installedAppPath = join(versionRoot, "node_modules", "@swarmx", "desktop");

      const restarting = this.publish({
        phase: "restarting",
        currentVersion: this.currentVersion,
        latestVersion: release.version,
        progress: 100,
      });
      if (this.restartDelayMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, this.restartDelayMs));
      }
      await this.restart(installedAppPath);
      return restarting;
    } catch (error) {
      await Promise.all([
        rm(partialPath, { force: true }),
        rm(stagingRoot, { recursive: true, force: true }),
      ]);
      return this.publish({
        phase: "available",
        currentVersion: this.currentVersion,
        latestVersion: release.version,
        error: boundedErrorMessage(error),
      });
    }
  }

  private async downloadRelease(release: NpmDesktopRelease, destination: string): Promise<void> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.downloadTimeoutMs);
    let file: Awaited<ReturnType<typeof open>> | null = null;
    try {
      const response = await this.fetch(release.tarballUrl, { signal: controller.signal });
      assertCanonicalTarballResponse(response, release);
      if (!response.body) throw new Error("npm update download returned no body.");

      const totalBytes = normalizedContentLength(response.headers.get("content-length"));
      const hash = createHash("sha512");
      const reader = response.body.getReader();
      file = await open(destination, "w", 0o600);
      let receivedBytes = 0;
      let publishedProgress = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value || value.byteLength === 0) continue;
        await file.write(value);
        hash.update(value);
        receivedBytes += value.byteLength;
        if (totalBytes > 0) {
          const progress = Math.min(99, Math.floor((receivedBytes / totalBytes) * 100));
          if (progress > publishedProgress) {
            publishedProgress = progress;
            this.publish({
              phase: "downloading",
              currentVersion: this.currentVersion,
              latestVersion: release.version,
              progress,
            });
          }
        }
      }

      const actual = hash.digest();
      const expected = Buffer.from(release.integrity.slice("sha512-".length), "base64");
      if (actual.byteLength !== expected.byteLength || !timingSafeEqual(actual, expected)) {
        throw new Error("Downloaded update failed npm integrity verification.");
      }
      this.publish({
        phase: "downloading",
        currentVersion: this.currentVersion,
        latestVersion: release.version,
        progress: 100,
      });
    } finally {
      clearTimeout(timeout);
      await file?.close();
    }
  }

  private publish(state: DesktopUpdateState): DesktopUpdateState {
    this.state = { ...state };
    const snapshot = this.getState();
    for (const listener of this.listeners) {
      try {
        listener(snapshot);
      } catch {
        // One renderer listener must not stop update state publication.
      }
    }
    return snapshot;
  }
}

export function createDisabledDesktopUpdateService(
  currentVersion = "0.0.0",
): DesktopUpdateServiceLike {
  const state: DesktopUpdateState = { phase: "hidden", currentVersion };
  return {
    getState: () => ({ ...state }),
    check: async () => ({ ...state }),
    startUpdate: async () => ({ ...state }),
    subscribe: () => () => undefined,
  };
}

export function compareSemanticVersions(left: string, right: string): number {
  const a = parseSemanticVersion(left);
  const b = parseSemanticVersion(right);
  for (const key of ["major", "minor", "patch"] as const) {
    if (a[key] !== b[key]) return a[key] > b[key] ? 1 : -1;
  }
  if (a.prerelease.length === 0 && b.prerelease.length > 0) return 1;
  if (a.prerelease.length > 0 && b.prerelease.length === 0) return -1;
  for (let index = 0; index < Math.max(a.prerelease.length, b.prerelease.length); index += 1) {
    const aPart = a.prerelease[index];
    const bPart = b.prerelease[index];
    if (aPart === undefined) return -1;
    if (bPart === undefined) return 1;
    if (aPart === bPart) continue;
    const aNumber = numericIdentifier(aPart);
    const bNumber = numericIdentifier(bPart);
    if (aNumber !== null && bNumber !== null) return aNumber > bNumber ? 1 : -1;
    if (aNumber !== null) return -1;
    if (bNumber !== null) return 1;
    return aPart > bPart ? 1 : -1;
  }
  return 0;
}

async function fetchWithTimeout(
  fetcher: UpdateFetch,
  url: string,
  timeoutMs: number,
  init: RequestInit,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetcher(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

function parseNpmDesktopRelease(value: unknown): NpmDesktopRelease {
  if (!value || typeof value !== "object") throw new Error("Invalid npm update metadata.");
  const record = value as Record<string, unknown>;
  const version = record.version;
  const dist = record.dist;
  if (record.name !== DESKTOP_PACKAGE_NAME || typeof version !== "string") {
    throw new Error("Unexpected npm update package metadata.");
  }
  const parsedVersion = parseSemanticVersion(version);
  if (parsedVersion.prerelease.length > 0) throw new Error("npm latest must be a stable version.");
  if (!dist || typeof dist !== "object")
    throw new Error("Missing npm update distribution metadata.");
  const tarballUrl = (dist as Record<string, unknown>).tarball;
  const integrity = (dist as Record<string, unknown>).integrity;
  if (typeof tarballUrl !== "string" || typeof integrity !== "string") {
    throw new Error("Incomplete npm update distribution metadata.");
  }
  assertCanonicalTarballUrl(tarballUrl, version);
  if (!/^sha512-[A-Za-z0-9+/]+={0,2}$/.test(integrity)) {
    throw new Error("npm update metadata requires SHA-512 integrity.");
  }
  return { version, tarballUrl, integrity };
}

function parseSemanticVersion(version: string): {
  major: number;
  minor: number;
  patch: number;
  prerelease: string[];
} {
  const match =
    /^(?:v)?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$/.exec(
      version,
    );
  if (!match) throw new Error(`Invalid semantic version: ${version}`);
  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: Number(match[3]),
    prerelease: match[4]?.split(".") ?? [],
  };
}

function numericIdentifier(value: string): number | null {
  return /^(0|[1-9]\d*)$/.test(value) ? Number(value) : null;
}

function assertCanonicalTarballUrl(url: string, version: string): void {
  const parsed = new URL(url);
  const expectedPath = `/@swarmx/desktop/-/desktop-${version}.tgz`;
  if (
    parsed.protocol !== "https:" ||
    parsed.hostname !== NPM_REGISTRY_HOST ||
    parsed.port !== "" ||
    parsed.username !== "" ||
    parsed.password !== "" ||
    parsed.pathname !== expectedPath
  ) {
    throw new Error("npm update tarball URL is not canonical.");
  }
}

function assertCanonicalTarballResponse(response: Response, release: NpmDesktopRelease): void {
  if (!response.ok) throw new Error(`npm update download failed (${response.status}).`);
  const responseUrl = response.url || release.tarballUrl;
  assertCanonicalTarballUrl(responseUrl, release.version);
}

function normalizedContentLength(value: string | null): number {
  if (!value) return 0;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : 0;
}

async function assertInstalledDesktopVersion(
  appPath: string,
  expectedVersion: string,
): Promise<void> {
  const parsed = JSON.parse(await readFile(join(appPath, "package.json"), "utf8")) as {
    name?: unknown;
    version?: unknown;
  };
  if (parsed.name !== DESKTOP_PACKAGE_NAME || parsed.version !== expectedVersion) {
    throw new Error("Installed npm update version did not match the verified release.");
  }
}

function runUpdateCommand(command: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: "ignore",
      windowsHide: true,
      env: updateCommandEnvironment(process.env),
    });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(
        new Error(
          signal
            ? `npm update install stopped by ${signal}.`
            : `npm update install failed (${code}).`,
        ),
      );
    });
  });
}

function updateCommandEnvironment(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const keys = [
    "PATH",
    "HOME",
    "USERPROFILE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SystemRoot",
    "ComSpec",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "NODE_EXTRA_CA_CERTS",
    "NPM_CONFIG_REGISTRY",
  ];
  const result: NodeJS.ProcessEnv = {
    NPM_CONFIG_AUDIT: "false",
    NPM_CONFIG_FUND: "false",
    NPM_CONFIG_UPDATE_NOTIFIER: "false",
  };
  for (const key of keys) {
    if (env[key]) result[key] = env[key];
  }
  return result;
}

function boundedErrorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return message.replace(/\s+/g, " ").trim().slice(0, 180) || "Update failed.";
}

function isBusyPhase(phase: DesktopUpdatePhase): boolean {
  return phase === "downloading" || phase === "installing" || phase === "restarting";
}
