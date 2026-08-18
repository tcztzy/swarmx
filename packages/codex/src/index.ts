import { accessSync, constants, realpathSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import { delimiter, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Context, Plugin } from "@deepseek-ai/cordis";
import {
  type CodexPermissionHandler,
  CodexServerClient,
  type CodexServerLaunchSpec,
} from "./codex-server-client.js";

export const CODEX_MODULE_COMMAND = "swarmx-codex";
export const CODEX_SERVER_TRANSPORT = "codex_server";
export const CODEX_CONTAINER_MODULE_DIR = "/opt/swarmx/codex";
export const CODEX_CONTAINER_ENTRY = `${CODEX_CONTAINER_MODULE_DIR}/bin/swarmx-codex-container.js`;
export const CODEX_CONTAINER_RUNTIME_DIR = "/opt/swarmx/codex-runtime";

const require = createRequire(import.meta.url);

export interface AcpLaunchRequest {
  command: string;
  args: readonly string[];
  transport?: string;
}

export interface AcpLaunchSpec {
  command: string;
  args: string[];
  env: Record<string, string>;
}

export type AcpLaunchResolver = (request: AcpLaunchRequest) => AcpLaunchSpec;

export interface AcpLauncherRegistry {
  register(command: string, resolver: AcpLaunchResolver): () => void;
  resolve(request: AcpLaunchRequest): AcpLaunchSpec;
}

export interface CodexAcpLauncherConfig {
  nodeExecutable?: string;
  electron?: boolean;
  resolveModule?: (specifier: string) => string;
  /**
   * Highest-priority user-managed Codex CLI. When absent, resolution first
   * searches PATH, then falls back to the pinned `@openai/codex` module.
   */
  codexCommand?: string;
  /** PATH used for Codex discovery; defaults to `process.env.PATH`. */
  envPath?: string;
}

export type CodexHarnessPluginConfig = CodexAcpLauncherConfig;

interface HarnessTransportRegistry {
  register(
    id: string,
    factory: (request: AcpLaunchRequest, launch: AcpLaunchSpec) => unknown,
  ): () => void;
}

interface HarnessPermissionRegistry {
  register(id: string, resolver: CodexPermissionHandler, priority?: number): () => void;
  resolve(): CodexPermissionHandler | undefined;
}

interface DshApprovalLike {
  config?: { policy?: "ask" | "never" };
}

interface DshPermissionPresetsLike {
  readonly defaultPreset: string;
  resolve(name: string): { approval?: "ask" | "never" };
}

interface DshPermissionContext {
  get(name: string): unknown;
}

/**
 * Resolve the Codex permission handler from a composed DSH permission stack.
 * `never` maps directly to deterministic rejection; `ask` falls through to the
 * SwarmX `harnessPermissions` registry.
 */
export function resolveDshCodexPermissionHandler(
  ctx: DshPermissionContext,
): CodexPermissionHandler | undefined {
  const approval = ctx.get("approval") as DshApprovalLike | undefined;
  const presets = ctx.get("permissionPresets") as DshPermissionPresetsLike | undefined;
  if (!approval && !presets) return undefined;
  const presetApproval = presets ? presets.resolve(presets.defaultPreset).approval : undefined;
  const policy = approval?.config?.policy ?? presetApproval;
  return policy === "never"
    ? async () => ({ outcome: { outcome: "rejected" as const } })
    : undefined;
}

declare module "@deepseek-ai/cordis" {
  interface Context {
    acpLaunchers: AcpLauncherRegistry;
  }
}

/** Resolve the first-party Codex module without invoking npm or a system Codex install. */
export function resolveCodexAcpLaunch(
  request: AcpLaunchRequest,
  config: CodexAcpLauncherConfig = {},
): AcpLaunchSpec {
  if (request.command !== CODEX_MODULE_COMMAND) {
    throw new Error(`Codex launcher cannot resolve command "${request.command}".`);
  }
  if (request.args.length > 0) {
    throw new Error(`Managed Codex command "${CODEX_MODULE_COMMAND}" does not accept arguments.`);
  }
  if (config.codexCommand) {
    assertExecutableCodex(config.codexCommand);
    return { command: config.codexCommand, args: ["app-server"], env: {} };
  }

  const pathCodex = findCodexInPath(config.envPath ?? process.env.PATH ?? "");
  if (pathCodex) {
    return { command: pathCodex, args: ["app-server"], env: {} };
  }

  const electron = config.electron ?? Boolean(process.versions.electron);
  return {
    command: config.nodeExecutable ?? process.execPath,
    args: [pinnedCodexEntry(config)],
    env: electron ? { ELECTRON_RUN_AS_NODE: "1" } : {},
  };
}

function pinnedCodexEntry(config: CodexAcpLauncherConfig): string {
  const resolveModule =
    config.resolveModule ?? ((specifier: string) => import.meta.resolve(specifier));
  return unpackedAsarPath(fileURLToPath(resolveModule("@swarmx/codex/cli")));
}

export interface CodexContainerAssets {
  /** Host path to the repository Codex module mounted read-only into the container. */
  moduleDir: string;
  /** Host path to the Linux Codex runtime package for the container architecture. */
  runtimeDir: string;
}

/**
 * Resolves the pinned module and matching Linux Codex runtime for protected
 * container execution. Container bootstrap runs `CODEX_CONTAINER_ENTRY` with
 * `SWARMX_CODEX_RUNTIME_DIR`; the returned host directories are mounted at
 * those fixed container paths by the protected wrapper.
 */
export function resolveCodexContainerAssets(
  config: CodexAcpLauncherConfig = {},
  arch: string = process.arch,
): CodexContainerAssets | undefined {
  const runtimePackage =
    arch === "x64"
      ? "@openai/codex-linux-x64"
      : arch === "arm64"
        ? "@openai/codex-linux-arm64"
        : undefined;
  if (!runtimePackage) return undefined;
  try {
    const moduleEntry = pinnedCodexEntry(config);
    const runtimePackageJson = require.resolve(`${runtimePackage}/package.json`);
    return {
      moduleDir: realpathSync(dirname(dirname(moduleEntry))),
      runtimeDir: realpathSync(dirname(runtimePackageJson)),
    };
  } catch {
    return undefined;
  }
}

/** Cordis plugin that contributes the managed Codex command to the ACP launcher registry. */
export const codexAcpLauncher = {
  name: "swarmx-codex-launcher",
  inject: ["acpLaunchers"],
  apply(ctx: Context, config: CodexAcpLauncherConfig = {}) {
    ctx.acpLaunchers.register(CODEX_MODULE_COMMAND, (request) =>
      resolveCodexAcpLaunch(request, config),
    );
  },
} satisfies Plugin.Object<CodexAcpLauncherConfig>;

/**
 * Cordis plugin that adapts Codex through its native `codex app-server`
 * JSON-RPC transport. ACP is not used on this path; the existing `swarmx-codex`
 * command token and packaged asar projection are reused for process launch.
 */
export const codexHarnessPlugin = {
  name: "swarmx-codex-harness",
  inject: ["acpLaunchers", "harnessPermissions", "harnessTransports"],
  apply(
    ctx: Context & {
      harnessPermissions: HarnessPermissionRegistry;
      harnessTransports: HarnessTransportRegistry;
    },
    config: CodexHarnessPluginConfig = {},
  ) {
    ctx.acpLaunchers.register(CODEX_MODULE_COMMAND, (request) =>
      resolveCodexAcpLaunch(request, config),
    );
    ctx.harnessTransports.register(CODEX_SERVER_TRANSPORT, (_request, launch) => {
      const spec: CodexServerLaunchSpec = {
        command: launch.command,
        args: launch.args,
        env: launch.env,
      };
      const permissionHandler =
        resolveDshCodexPermissionHandler(ctx) ?? ctx.harnessPermissions.resolve();
      return new CodexServerClient(spec, permissionHandler);
    });
  },
} satisfies Plugin.Object<CodexHarnessPluginConfig>;

function assertExecutableCodex(value: string): void {
  try {
    accessSync(value, constants.X_OK | constants.F_OK);
    if (statSync(value).isFile()) return;
  } catch {
    // Converted to the actionable error below.
  }
  throw new Error(`Configured codexCommand "${value}" is not an executable file.`);
}

/** Find the first executable `codex` on a PATH-style search list. */
export function findCodexInPath(
  envPath: string,
  platform: NodeJS.Platform = process.platform,
): string | undefined {
  const extensions = platform === "win32" ? ["", ".exe"] : [""];
  for (const directory of envPath.split(delimiter)) {
    if (!directory) continue;
    for (const extension of extensions) {
      const candidate = join(directory, `codex${extension}`);
      try {
        accessSync(candidate, constants.X_OK | constants.F_OK);
        if (statSync(candidate).isFile()) return candidate;
      } catch {
        // Keep searching PATH.
      }
    }
  }
  return undefined;
}

function unpackedAsarPath(value: string): string {
  return value.replace(/([\\/])app\.asar([\\/])/, "$1app.asar.unpacked$2");
}
