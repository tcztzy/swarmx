import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { access, readFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const CHECK_TIMEOUT_MS = 8_000;
const OUTPUT_LIMIT = 64 * 1024;
const ENVIRONMENT_DIGEST_VERSION = 2;
const DEFAULT_PYTHON_REQUEST = ">=3.11";

export type PythonEnvironmentComponentState = "ready" | "missing" | "failed";
export type PythonWorkerEnvironmentState =
  | "ready"
  | "asset_missing"
  | "uv_missing"
  | "managed_python_missing"
  | "environment_missing"
  | "environment_stale"
  | "failed";
export type PythonWorkerEnvironmentActionKind =
  | "restore_assets"
  | "install_uv"
  | "install_managed_python"
  | "sync_environment"
  | "manual_repair";
export type PythonWorkerEnvironmentActionRisk = "user_action" | "install" | "repair";

export interface PythonWorkerEnvironmentConfig {
  projectPath: string;
  lockPath: string;
  workerPath: string;
  environmentRoot: string;
  pythonRequest?: string;
  workingDirectory?: string;
  /** Additional managed-module source files whose hashes join the environment digest. */
  moduleSources?: string[];
}

export interface PythonEnvironmentCommandResult {
  exitCode: number | null;
  stdout: string;
  stderr: string;
  error?: string;
}

export interface PythonEnvironmentCommandOptions {
  cwd?: string;
  env: NodeJS.ProcessEnv;
  timeoutMs: number;
}

export interface PythonWorkerEnvironmentHost {
  env?: NodeJS.ProcessEnv;
  platform?: NodeJS.Platform;
  homeDir?: string;
  now?: () => Date;
  findExecutable?: (
    command: string,
    envPath: string,
    platform: NodeJS.Platform,
    env: NodeJS.ProcessEnv,
  ) => Promise<string | null>;
  runCommand?: (
    program: string,
    args: string[],
    options: PythonEnvironmentCommandOptions,
  ) => Promise<PythonEnvironmentCommandResult>;
}

export interface PythonWorkerAssetStatus {
  path: string;
  status: PythonEnvironmentComponentState;
  sha256?: string;
  note?: string;
}

export interface PythonExecutableStatus {
  command: string;
  status: PythonEnvironmentComponentState;
  path?: string;
  version?: string;
  implementation?: string;
  architecture?: string;
  basePrefix?: string;
  note?: string;
}

export interface PythonWorkerEnvironmentInstanceStatus {
  path: string;
  pythonPath: string;
  digest: string;
  status: PythonEnvironmentComponentState;
  note?: string;
}

export interface PythonWorkerLaunchSpec {
  backendId: "python";
  program: string;
  args: string[];
  cwd: string;
  env: Record<string, string>;
  environmentDigest: string;
}

export interface PythonWorkerEnvironmentStatus {
  checkedAt: string;
  ready: boolean;
  state: PythonWorkerEnvironmentState;
  setupAvailable: boolean;
  pythonRequest: string;
  uv: PythonExecutableStatus;
  managedPython: PythonExecutableStatus;
  project: PythonWorkerAssetStatus;
  lock: PythonWorkerAssetStatus;
  worker: PythonWorkerAssetStatus;
  additionalSources: PythonWorkerAssetStatus[];
  environment?: PythonWorkerEnvironmentInstanceStatus;
  note?: string;
}

export interface PythonWorkerSetupCommand {
  program: string;
  args: string[];
  cwd: string;
  env: Record<string, string>;
}

export interface PythonWorkerEnvironmentPlanAction {
  id: string;
  kind: PythonWorkerEnvironmentActionKind;
  label: string;
  risk: PythonWorkerEnvironmentActionRisk;
  reason: string;
  command?: PythonWorkerSetupCommand;
}

export interface PythonWorkerEnvironmentPlan {
  actions: PythonWorkerEnvironmentPlanAction[];
  requiresConfirmation: boolean;
  requiresUserAction: boolean;
  environmentDigest?: string;
}

export interface PythonEnvironmentDigestInput {
  projectSha256: string;
  lockSha256: string;
  workerSha256: string;
  additionalSourceSha256s: string[];
  uvVersion: string;
  pythonRequest: string;
  pythonImplementation: string;
  pythonVersion: string;
  platform: NodeJS.Platform;
  architecture: string;
}

interface PythonMetadata {
  implementation: string;
  version: string;
  architecture: string;
  basePrefix: string;
}

interface NormalizedPythonWorkerEnvironmentConfig {
  projectPath: string;
  projectDirectory: string;
  lockPath: string;
  workerPath: string;
  environmentRoot: string;
  pythonRequest: string;
  workingDirectory: string;
  moduleSources: string[];
}

export function computePythonEnvironmentDigest(input: PythonEnvironmentDigestInput): string {
  const canonical = JSON.stringify({
    schemaVersion: ENVIRONMENT_DIGEST_VERSION,
    projectSha256: input.projectSha256,
    lockSha256: input.lockSha256,
    workerSha256: input.workerSha256,
    additionalSourceSha256s: [...input.additionalSourceSha256s].sort(),
    uvVersion: input.uvVersion,
    pythonRequest: input.pythonRequest,
    pythonImplementation: input.pythonImplementation,
    pythonVersion: input.pythonVersion,
    platform: input.platform,
    architecture: input.architecture,
  });
  return `sha256:${createHash("sha256").update(canonical).digest("hex")}`;
}

/**
 * Inspect the Python worker environment without installing Python, syncing dependencies, or
 * mutating the project. Setup is represented only as a plan for a separately confirmed host
 * action. Task execution should use `launchSpec`, which bypasses uv and starts the verified
 * environment interpreter directly.
 */
export class PythonWorkerEnvironmentService {
  private readonly config: NormalizedPythonWorkerEnvironmentConfig;
  private readonly env: NodeJS.ProcessEnv;
  private readonly platform: NodeJS.Platform;
  private readonly homeDir: string;
  private readonly now: () => Date;
  private readonly findExecutable: NonNullable<PythonWorkerEnvironmentHost["findExecutable"]>;
  private readonly runCommand: NonNullable<PythonWorkerEnvironmentHost["runCommand"]>;

  constructor(config: PythonWorkerEnvironmentConfig, host: PythonWorkerEnvironmentHost = {}) {
    this.config = normalizeConfig(config);
    this.env = { ...(host.env ?? process.env) };
    this.platform = host.platform ?? process.platform;
    this.homeDir = host.homeDir ?? os.homedir();
    this.now = host.now ?? (() => new Date());
    this.findExecutable = host.findExecutable ?? findExecutableOnPath;
    this.runCommand = host.runCommand ?? runCommand;
  }

  async status(): Promise<PythonWorkerEnvironmentStatus> {
    const checkedAt = this.now().toISOString();
    const discoveryEnv = pythonDiscoveryEnvironment(this.env, this.homeDir, this.platform);
    const additionalSources = await Promise.all(
      this.config.moduleSources.map((source) => hashAsset(source)),
    );
    const [project, lock, worker, uv] = await Promise.all([
      hashAsset(this.config.projectPath),
      hashAsset(this.config.lockPath),
      hashAsset(this.config.workerPath),
      this.detectUv(discoveryEnv),
    ]);
    const assets = [project, lock, worker, ...additionalSources];
    if (assets.some((asset) => asset.status !== "ready")) {
      return this.result({
        checkedAt,
        state: assets.some((asset) => asset.status === "failed") ? "failed" : "asset_missing",
        uv,
        managedPython: missingPythonStatus(),
        project,
        lock,
        worker,
        additionalSources,
        note: assets.find((asset) => asset.status !== "ready")?.note,
      });
    }

    if (uv.status !== "ready" || !uv.path || !uv.version) {
      return this.result({
        checkedAt,
        state: uv.status === "missing" ? "uv_missing" : "failed",
        uv,
        managedPython: missingPythonStatus(),
        project,
        lock,
        worker,
        additionalSources,
        note: uv.note,
      });
    }

    const managedPython = await this.detectManagedPython(uv.path, discoveryEnv);
    if (
      managedPython.status !== "ready" ||
      !managedPython.path ||
      !managedPython.version ||
      !managedPython.implementation ||
      !managedPython.architecture
    ) {
      return this.result({
        checkedAt,
        state: managedPython.status === "missing" ? "managed_python_missing" : "failed",
        uv,
        managedPython,
        project,
        lock,
        worker,
        additionalSources,
        note: managedPython.note,
      });
    }

    const digest = computePythonEnvironmentDigest({
      projectSha256: requiredHash(project),
      lockSha256: requiredHash(lock),
      workerSha256: requiredHash(worker),
      additionalSourceSha256s: additionalSources.map(requiredHash),
      uvVersion: uv.version,
      pythonRequest: this.config.pythonRequest,
      pythonImplementation: managedPython.implementation,
      pythonVersion: managedPython.version,
      platform: this.platform,
      architecture: managedPython.architecture,
    });
    const environmentPath = path.join(this.config.environmentRoot, digest.replace(":", "-"));
    const pythonPath = environmentPythonPath(environmentPath, this.platform);
    const environment = await this.inspectEnvironment(
      environmentPath,
      pythonPath,
      digest,
      uv.path,
      managedPython,
      discoveryEnv,
    );
    const ready = environment.status === "ready";
    return this.result({
      checkedAt,
      state: ready
        ? "ready"
        : environment.status === "missing"
          ? "environment_missing"
          : "environment_stale",
      uv,
      managedPython,
      project,
      lock,
      worker,
      environment,
      additionalSources,
      note: environment.note,
    });
  }

  plan(status: PythonWorkerEnvironmentStatus): PythonWorkerEnvironmentPlan {
    return planPythonWorkerEnvironment(status, this.config, this.platform);
  }

  async launchSpec(status: PythonWorkerEnvironmentStatus): Promise<PythonWorkerLaunchSpec> {
    assertStatusMatchesConfig(status, this.config, this.platform);
    if (!status.environment) {
      throw new Error(`Python worker environment is not ready (${status.state}).`);
    }
    const current = await this.status();
    if (current.environment?.digest !== status.environment.digest) {
      throw new Error(
        "Python worker runtime assets or environment changed after the supplied health check.",
      );
    }
    if (!current.ready || !current.environment || current.environment.status !== "ready") {
      throw new Error(`Python worker environment is not ready (${current.state}).`);
    }
    const discoveryEnv = pythonDiscoveryEnvironment(this.env, this.homeDir, this.platform);
    const workerSource = await readFile(this.config.workerPath);
    const workerSha256 = createHash("sha256").update(workerSource).digest("hex");
    if (workerSha256 !== current.worker.sha256) {
      throw new Error("Python worker source changed during launch verification.");
    }
    const additionalHashes = await Promise.all(
      this.config.moduleSources.map(async (source) => {
        const content = await readFile(source);
        return { source, sourceSha256: createHash("sha256").update(content).digest("hex") };
      }),
    );
    for (const [index, source] of additionalHashes.entries()) {
      const expected = current.additionalSources[index]?.sha256;
      if (!expected || source.sourceSha256 !== expected) {
        throw new Error("Python worker module sources changed during launch verification.");
      }
    }
    return this.createLaunchSpec(current.environment, discoveryEnv, workerSource.toString("utf8"));
  }

  private result(
    input: Omit<PythonWorkerEnvironmentStatus, "ready" | "setupAvailable" | "pythonRequest">,
  ): PythonWorkerEnvironmentStatus {
    return {
      ...input,
      ready: input.state === "ready",
      setupAvailable:
        input.state === "managed_python_missing" ||
        input.state === "environment_missing" ||
        input.state === "environment_stale",
      pythonRequest: this.config.pythonRequest,
    };
  }

  private async detectUv(env: NodeJS.ProcessEnv): Promise<PythonExecutableStatus> {
    const envPath = env.PATH ?? "";
    const uvPath = await this.findExecutable("uv", envPath, this.platform, env);
    if (!uvPath) {
      return {
        command: "uv",
        status: "missing",
        note: "uv is not installed or is not available on the configured PATH.",
      };
    }
    const result = await this.runCommand(uvPath, ["--version"], {
      env,
      timeoutMs: CHECK_TIMEOUT_MS,
    });
    if (result.exitCode !== 0 || result.error) {
      return {
        command: "uv",
        status: "failed",
        path: uvPath,
        note: commandFailure(result, "uv version check failed."),
      };
    }
    const version = `${result.stdout}\n${result.stderr}`.match(/\buv\s+(\d+\.\d+\.\d+)/)?.[1];
    if (!version) {
      return {
        command: "uv",
        status: "failed",
        path: uvPath,
        note: "uv version check did not report a semantic version.",
      };
    }
    return { command: "uv", status: "ready", path: uvPath, version };
  }

  private async detectManagedPython(
    uvPath: string,
    env: NodeJS.ProcessEnv,
  ): Promise<PythonExecutableStatus> {
    const result = await this.runCommand(
      uvPath,
      [
        "python",
        "find",
        this.config.pythonRequest,
        "--managed-python",
        "--system",
        "--no-python-downloads",
        "--offline",
        "--no-project",
        "--resolve-links",
        "--no-cache",
        "--no-config",
      ],
      { env, timeoutMs: CHECK_TIMEOUT_MS },
    );
    if (result.exitCode !== 0 || result.error) {
      return {
        command: "python",
        status: "missing",
        note: commandFailure(result, "No compatible uv-managed Python is installed."),
      };
    }
    const pythonPath = firstLine(result.stdout);
    if (!pythonPath || !path.isAbsolute(pythonPath)) {
      return {
        command: "python",
        status: "failed",
        note: "uv returned an invalid managed Python path.",
      };
    }
    const metadata = await this.readPythonMetadata(pythonPath, env);
    if (!metadata.success) {
      return {
        command: "python",
        status: metadata.missing ? "missing" : "failed",
        path: pythonPath,
        note: metadata.note,
      };
    }
    return {
      command: "python",
      status: "ready",
      path: pythonPath,
      version: metadata.value.version,
      implementation: metadata.value.implementation,
      architecture: metadata.value.architecture,
      basePrefix: metadata.value.basePrefix,
    };
  }

  private async inspectEnvironment(
    environmentPath: string,
    pythonPath: string,
    digest: string,
    uvPath: string,
    managedPython: PythonExecutableStatus,
    env: NodeJS.ProcessEnv,
  ): Promise<PythonWorkerEnvironmentInstanceStatus> {
    const metadata = await this.readPythonMetadata(pythonPath, env);
    if (!metadata.success) {
      return {
        path: environmentPath,
        pythonPath,
        digest,
        status: metadata.missing ? "missing" : "failed",
        note: metadata.note,
      };
    }
    if (
      metadata.value.version !== managedPython.version ||
      metadata.value.implementation !== managedPython.implementation ||
      metadata.value.architecture !== managedPython.architecture ||
      metadata.value.basePrefix !== managedPython.basePrefix
    ) {
      return {
        path: environmentPath,
        pythonPath,
        digest,
        status: "failed",
        note: "The Python environment does not match the discovered uv-managed interpreter.",
      };
    }

    const checkEnv = {
      ...env,
      UV_PROJECT_ENVIRONMENT: environmentPath,
      UV_PYTHON_DOWNLOADS: "never",
    };
    const check = await this.runCommand(
      uvPath,
      [
        "sync",
        "--project",
        this.config.projectDirectory,
        "--locked",
        "--check",
        "--no-default-groups",
        "--managed-python",
        "--python",
        requiredPath(managedPython),
        "--no-python-downloads",
        "--offline",
        "--no-cache",
      ],
      { cwd: this.config.projectDirectory, env: checkEnv, timeoutMs: CHECK_TIMEOUT_MS },
    );
    if (check.exitCode !== 0 || check.error) {
      return {
        path: environmentPath,
        pythonPath,
        digest,
        status: "failed",
        note: commandFailure(check, "The Python environment is not synchronized with uv.lock."),
      };
    }
    return { path: environmentPath, pythonPath, digest, status: "ready" };
  }

  private async readPythonMetadata(
    pythonPath: string,
    env: NodeJS.ProcessEnv,
  ): Promise<
    { success: true; value: PythonMetadata } | { success: false; missing: boolean; note: string }
  > {
    try {
      await access(pythonPath, constants.X_OK);
    } catch {
      return { success: false, missing: true, note: `Python executable is missing: ${pythonPath}` };
    }
    const script =
      "import json,platform,sys;print(json.dumps({" +
      "'implementation':sys.implementation.name," +
      "'version':platform.python_version()," +
      "'architecture':platform.machine()," +
      "'basePrefix':sys.base_prefix},separators=(',',':')))";
    const result = await this.runCommand(pythonPath, ["-I", "-S", "-c", script], {
      env,
      timeoutMs: CHECK_TIMEOUT_MS,
    });
    if (result.exitCode !== 0 || result.error) {
      return {
        success: false,
        missing: false,
        note: commandFailure(result, `Python metadata check failed: ${pythonPath}`),
      };
    }
    const metadata = parsePythonMetadata(result.stdout);
    return metadata
      ? { success: true, value: metadata }
      : {
          success: false,
          missing: false,
          note: `Python metadata check returned invalid JSON: ${pythonPath}`,
        };
  }

  private createLaunchSpec(
    environment: PythonWorkerEnvironmentInstanceStatus,
    discoveryEnv: NodeJS.ProcessEnv,
    workerSource: string,
  ): PythonWorkerLaunchSpec {
    const launchEnv = pythonWorkerEnvironment(discoveryEnv, environment.pythonPath);
    return {
      backendId: "python",
      program: environment.pythonPath,
      args: ["-I", "-B", "-u", "-c", workerSource, "--environment-digest", environment.digest],
      cwd: this.config.workingDirectory,
      env: launchEnv,
      environmentDigest: environment.digest,
    };
  }
}

export function planPythonWorkerEnvironment(
  status: PythonWorkerEnvironmentStatus,
  configInput: PythonWorkerEnvironmentConfig,
  platform: NodeJS.Platform = process.platform,
): PythonWorkerEnvironmentPlan {
  const config = normalizeConfig(configInput);
  assertStatusMatchesConfig(status, config, platform);
  const actions: PythonWorkerEnvironmentPlanAction[] = [];
  if (status.state === "asset_missing") {
    const missingAssets = [status.project, status.lock, status.worker]
      .filter((asset) => asset.status !== "ready")
      .map((asset) => asset.path);
    actions.push({
      id: "python-worker:restore-assets",
      kind: "restore_assets",
      label: "Restore Python worker runtime assets",
      risk: "user_action",
      reason:
        missingAssets.length > 0
          ? `Required runtime assets are unavailable: ${missingAssets.join(", ")}`
          : "Required Python worker runtime assets are unavailable.",
    });
  } else if (status.state === "failed") {
    actions.push({
      id: "python-worker:manual-repair",
      kind: "manual_repair",
      label: "Repair the Python worker runtime",
      risk: "user_action",
      reason: status.note ?? "Python worker runtime inspection failed.",
    });
  } else if (status.state === "uv_missing") {
    actions.push({
      id: "python-worker:install-uv",
      kind: "install_uv",
      label: "Install uv",
      risk: "user_action",
      reason: "uv must be installed explicitly before SwarmX can manage Python.",
    });
  } else if (status.state === "managed_python_missing" && status.uv.path) {
    actions.push({
      id: "python-worker:install-managed-python",
      kind: "install_managed_python",
      label: `Install uv-managed Python ${status.pythonRequest}`,
      risk: "install",
      reason: "No compatible uv-managed Python is installed.",
      command: {
        program: status.uv.path,
        args: ["python", "install", status.pythonRequest, "--managed-python"],
        cwd: config.projectDirectory,
        env: {},
      },
    });
  } else if (
    (status.state === "environment_missing" || status.state === "environment_stale") &&
    status.uv.path &&
    status.managedPython.path &&
    status.environment
  ) {
    actions.push({
      id: "python-worker:sync-environment",
      kind: "sync_environment",
      label:
        status.state === "environment_missing"
          ? "Create the Python worker environment"
          : "Repair the Python worker environment",
      risk: status.state === "environment_missing" ? "install" : "repair",
      reason:
        status.environment.note ??
        "The worker environment must be synchronized from the locked product dependencies.",
      command: {
        program: status.uv.path,
        args: [
          "sync",
          "--project",
          config.projectDirectory,
          "--locked",
          "--no-default-groups",
          "--managed-python",
          "--python",
          status.managedPython.path,
          "--no-python-downloads",
        ],
        cwd: config.projectDirectory,
        env: {
          UV_PROJECT_ENVIRONMENT: status.environment.path,
          UV_PYTHON_DOWNLOADS: "never",
        },
      },
    });
  }

  return {
    actions,
    requiresConfirmation: actions.some((action) => Boolean(action.command)),
    requiresUserAction: actions.some((action) => !action.command),
    environmentDigest: status.environment?.digest,
  };
}

function normalizeConfig(
  config: PythonWorkerEnvironmentConfig,
): NormalizedPythonWorkerEnvironmentConfig {
  const projectPath = path.resolve(config.projectPath);
  const lockPath = path.resolve(config.lockPath);
  const projectDirectory = path.dirname(projectPath);
  if (path.basename(projectPath) !== "pyproject.toml") {
    throw new Error("Python worker projectPath must name pyproject.toml.");
  }
  if (path.basename(lockPath) !== "uv.lock" || path.dirname(lockPath) !== projectDirectory) {
    throw new Error("Python worker lockPath must name uv.lock beside pyproject.toml.");
  }
  const pythonRequest = config.pythonRequest?.trim() || DEFAULT_PYTHON_REQUEST;
  if (/[\r\n\0]/.test(pythonRequest)) throw new Error("Invalid Python version request.");
  return {
    projectPath,
    projectDirectory,
    lockPath,
    workerPath: path.resolve(config.workerPath),
    environmentRoot: path.resolve(config.environmentRoot),
    pythonRequest,
    workingDirectory: path.resolve(config.workingDirectory ?? projectDirectory),
    moduleSources: (config.moduleSources ?? []).map((source) => path.resolve(source)),
  };
}

function assertStatusMatchesConfig(
  status: PythonWorkerEnvironmentStatus,
  config: NormalizedPythonWorkerEnvironmentConfig,
  platform: NodeJS.Platform = process.platform,
): void {
  if (
    status.project.path !== config.projectPath ||
    status.lock.path !== config.lockPath ||
    status.worker.path !== config.workerPath
  ) {
    throw new Error("Python worker environment status does not match the configured assets.");
  }
  if (!status.environment) return;
  if (!/^sha256:[a-f0-9]{64}$/.test(status.environment.digest)) {
    throw new Error("Python worker environment status has an invalid digest.");
  }
  const expectedPath = path.join(
    config.environmentRoot,
    status.environment.digest.replace(":", "-"),
  );
  const expectedPythonPath = environmentPythonPath(expectedPath, platform);
  if (
    status.environment.path !== expectedPath ||
    status.environment.pythonPath !== expectedPythonPath
  ) {
    throw new Error("Python worker environment status does not match the configured environment.");
  }
}

async function hashAsset(assetPath: string): Promise<PythonWorkerAssetStatus> {
  try {
    const value = await readFile(assetPath);
    return {
      path: assetPath,
      status: "ready",
      sha256: createHash("sha256").update(value).digest("hex"),
    };
  } catch (error) {
    const missing = isNotFound(error);
    return {
      path: assetPath,
      status: missing ? "missing" : "failed",
      note: missing ? `Required runtime asset is missing: ${assetPath}` : errorMessage(error),
    };
  }
}

function parsePythonMetadata(value: string): PythonMetadata | null {
  try {
    const parsed: unknown = JSON.parse(value.trim());
    if (!isRecord(parsed)) return null;
    const implementation = boundedString(parsed.implementation);
    const version = boundedString(parsed.version);
    const architecture = boundedString(parsed.architecture);
    const basePrefix = boundedString(parsed.basePrefix);
    if (!implementation || !version || !architecture || !basePrefix) return null;
    return { implementation, version, architecture, basePrefix };
  } catch {
    return null;
  }
}

function pythonDiscoveryEnvironment(
  source: NodeJS.ProcessEnv,
  homeDir: string,
  platform: NodeJS.Platform,
): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {};
  for (const key of [
    "HOME",
    "USERPROFILE",
    "LOCALAPPDATA",
    "APPDATA",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
    "UV_PYTHON_INSTALL_DIR",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "PATHEXT",
  ]) {
    if (source[key]) env[key] = source[key];
  }
  const executableDirs =
    platform === "win32"
      ? []
      : [path.join(homeDir, ".local", "bin"), path.join(homeDir, ".cargo", "bin")];
  env.PATH = uniqueStrings([
    ...executableDirs,
    ...(source.PATH ?? "").split(path.delimiter).filter(Boolean),
  ]).join(path.delimiter);
  return env;
}

function pythonWorkerEnvironment(
  source: NodeJS.ProcessEnv,
  pythonPath: string,
): Record<string, string> {
  const env: Record<string, string> = {
    PATH: path.dirname(pythonPath),
    PYTHONDONTWRITEBYTECODE: "1",
    PYTHONIOENCODING: "utf-8",
    PYTHONUNBUFFERED: "1",
    PYTHONUTF8: "1",
  };
  for (const key of [
    "HOME",
    "USERPROFILE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
  ]) {
    const value = source[key];
    if (value) env[key] = value;
  }
  return env;
}

function environmentPythonPath(environmentPath: string, platform: NodeJS.Platform): string {
  return platform === "win32"
    ? path.join(environmentPath, "Scripts", "python.exe")
    : path.join(environmentPath, "bin", "python");
}

function missingPythonStatus(): PythonExecutableStatus {
  return { command: "python", status: "missing" };
}

function requiredHash(asset: PythonWorkerAssetStatus): string {
  if (!asset.sha256) throw new Error(`Missing digest for runtime asset: ${asset.path}`);
  return asset.sha256;
}

function requiredPath(executable: PythonExecutableStatus): string {
  if (!executable.path) throw new Error(`Missing executable path for ${executable.command}.`);
  return executable.path;
}

function commandFailure(result: PythonEnvironmentCommandResult, fallback: string): string {
  return result.error ?? firstLine(result.stderr) ?? firstLine(result.stdout) ?? fallback;
}

function firstLine(value: string): string | undefined {
  return value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find(Boolean);
}

function boundedString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 && value.length <= 4_096 ? value : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNotFound(error: unknown): boolean {
  return isRecord(error) && error.code === "ENOENT";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values));
}

async function findExecutableOnPath(
  command: string,
  envPath: string,
  platform: NodeJS.Platform,
  env: NodeJS.ProcessEnv,
): Promise<string | null> {
  const extensions =
    platform === "win32"
      ? uniqueStrings(["", ...(env.PATHEXT ?? ".EXE;.CMD;.BAT;.COM").split(";")])
      : [""];
  for (const directory of envPath.split(path.delimiter)) {
    if (!directory) continue;
    for (const extension of extensions) {
      const candidate = path.join(directory, `${command}${extension.toLowerCase()}`);
      try {
        await access(candidate, constants.X_OK);
        return candidate;
      } catch {
        // Continue searching the configured PATH.
      }
    }
  }
  return null;
}

function runCommand(
  program: string,
  args: string[],
  options: PythonEnvironmentCommandOptions,
): Promise<PythonEnvironmentCommandResult> {
  return new Promise((resolve) => {
    const child = spawn(program, args, {
      cwd: options.cwd,
      env: options.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let settled = false;
    const timeout = setTimeout(() => {
      if (settled) return;
      child.kill();
      settled = true;
      resolve({
        exitCode: null,
        stdout,
        stderr,
        error: `Command timed out after ${options.timeoutMs}ms.`,
      });
    }, options.timeoutMs);
    child.stdout?.setEncoding("utf8");
    child.stdout?.on("data", (chunk: string) => {
      stdout = limitOutput(`${stdout}${chunk}`);
    });
    child.stderr?.setEncoding("utf8");
    child.stderr?.on("data", (chunk: string) => {
      stderr = limitOutput(`${stderr}${chunk}`);
    });
    child.on("error", (error) => {
      if (settled) return;
      clearTimeout(timeout);
      settled = true;
      resolve({ exitCode: null, stdout, stderr, error: error.message });
    });
    child.on("close", (exitCode) => {
      if (settled) return;
      clearTimeout(timeout);
      settled = true;
      resolve({ exitCode, stdout, stderr });
    });
  });
}

function limitOutput(value: string): string {
  return value.length > OUTPUT_LIMIT ? value.slice(-OUTPUT_LIMIT) : value;
}
