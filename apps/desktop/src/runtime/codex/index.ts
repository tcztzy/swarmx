import { spawn } from "node:child_process";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { WorkspaceScope } from "../contracts.js";
import { type ScienceCarrierConfig, serializeScienceCarrierConfig } from "../science-config.js";
import { CodexJsonRpcConnection } from "./connection.js";
import { CodexConversationRuntime } from "./runtime.js";
import { reconcileCodexSwarmBindings } from "./swarm-recovery.js";

export interface StartCodexRuntimeOptions {
  command?: string;
  args?: readonly string[];
  env?: NodeJS.ProcessEnv;
  productHome?: string;
  scienceConfig?: ScienceCarrierConfig;
  workspace?: WorkspaceScope;
  bridgeUrl?: string;
  bridgeToken?: string;
  mcpServerPath?: string;
  paginatedHistory?: boolean;
  startupTimeoutMs?: number;
}

const DEFAULT_STARTUP_TIMEOUT_MS = 30_000;
const MAX_STARTUP_TIMEOUT_MS = 5 * 60_000;

export async function startCodexRuntime(
  options: StartCodexRuntimeOptions = {},
): Promise<CodexConversationRuntime> {
  const command = options.command ?? "codex";
  const args = options.args ?? codexArgs(options);
  const paginatedHistory =
    options.paginatedHistory ??
    experimentalHistoryEnabled(options.env?.SWARMX_CODEX_PAGINATED_HISTORY);
  const startupTimeoutMs = options.startupTimeoutMs ?? DEFAULT_STARTUP_TIMEOUT_MS;
  if (
    !Number.isSafeInteger(startupTimeoutMs) ||
    startupTimeoutMs < 1 ||
    startupTimeoutMs > MAX_STARTUP_TIMEOUT_MS
  ) {
    throw new Error(
      `Codex App Server startup timeout must be an integer from 1 through ${String(MAX_STARTUP_TIMEOUT_MS)}ms.`,
    );
  }
  const child = spawn(command, [...args], {
    env: codexEnvironment(options),
    stdio: ["pipe", "pipe", "pipe"],
  });
  const connection = new CodexJsonRpcConnection(child);
  let startupOperation: Promise<CodexConversationRuntime> | undefined;
  try {
    startupOperation = (async () => {
      await connection.initialize();
      const runtime = new CodexConversationRuntime(connection, { paginatedHistory });
      if (
        options.productHome !== undefined &&
        options.workspace !== undefined &&
        options.bridgeUrl !== undefined &&
        options.bridgeToken !== undefined
      ) {
        await reconcileCodexSwarmBindings({
          journalRoot: join(options.productHome, "swarm"),
          runtime,
          workspace: options.workspace,
        });
      }
      return runtime;
    })();
    return await withTimeout(
      startupOperation,
      startupTimeoutMs,
      `Codex App Server startup timed out after ${String(startupTimeoutMs)}ms.`,
    );
  } catch (startupError) {
    const failure = new Error(
      `Unable to start Codex App Server with "${command}": ${startupError instanceof Error ? startupError.message : String(startupError)}`,
      { cause: startupError },
    );
    let cleanupError: unknown;
    try {
      await connection.dispose();
    } catch (error) {
      cleanupError = error;
    }
    await startupOperation?.catch(() => undefined);
    if (cleanupError !== undefined) {
      throw new AggregateError(
        [failure, cleanupError],
        "Codex App Server startup and cleanup both failed.",
        { cause: failure },
      );
    }
    throw failure;
  }
}

async function withTimeout<Value>(
  operation: Promise<Value>,
  timeoutMs: number,
  message: string,
): Promise<Value> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      operation,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error(message)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

function codexArgs(options: StartCodexRuntimeOptions): string[] {
  if (
    options.productHome === undefined ||
    options.workspace === undefined ||
    options.bridgeUrl === undefined ||
    options.bridgeToken === undefined
  ) {
    return ["app-server"];
  }
  const serverPath =
    options.mcpServerPath ?? fileURLToPath(new URL("./mcp-server.js", import.meta.url));
  return [
    "-c",
    `mcp_servers.swarmx.command=${JSON.stringify(process.env.SWARMX_NODE ?? "node")}`,
    "-c",
    `mcp_servers.swarmx.args=${JSON.stringify([serverPath])}`,
    "-c",
    `mcp_servers.swarmx.env_vars=${JSON.stringify([
      "SWARMX_BRIDGE_TOKEN",
      "SWARMX_BRIDGE_URL",
      "SWARMX_HOME",
      "SWARMX_SCIENCE_CONFIG",
      "SWARMX_WORKSPACE_ID",
      "SWARMX_WORKSPACE_LABEL",
      "SWARMX_WORKSPACE_ROOT",
    ])}`,
    "-c",
    "mcp_servers.swarmx.required=true",
    "-c",
    "mcp_servers.swarmx.startup_timeout_sec=30",
    "app-server",
  ];
}

function codexEnvironment(options: StartCodexRuntimeOptions): NodeJS.ProcessEnv {
  const environment = { ...process.env, ...options.env };
  if (
    options.productHome !== undefined &&
    options.workspace !== undefined &&
    options.bridgeUrl !== undefined &&
    options.bridgeToken !== undefined
  ) {
    environment.SWARMX_BRIDGE_TOKEN = options.bridgeToken;
    environment.SWARMX_BRIDGE_URL = options.bridgeUrl;
    environment.SWARMX_HOME = options.productHome;
    if (options.scienceConfig !== undefined) {
      environment.SWARMX_SCIENCE_CONFIG = serializeScienceCarrierConfig(options.scienceConfig);
    }
    environment.SWARMX_WORKSPACE_ID = options.workspace.id;
    environment.SWARMX_WORKSPACE_LABEL = options.workspace.label;
    environment.SWARMX_WORKSPACE_ROOT = options.workspace.root;
  }
  return environment;
}

function experimentalHistoryEnabled(override?: string): boolean {
  const value = override ?? process.env.SWARMX_CODEX_PAGINATED_HISTORY;
  if (value === undefined || value === "" || value === "0") return false;
  if (value === "1") return true;
  throw new Error('SWARMX_CODEX_PAGINATED_HISTORY must be "0" or "1".');
}

export const codexAppServerArgs = codexArgs;
export const codexAppServerEnvironment = codexEnvironment;

export { CodexJsonRpcConnection } from "./connection.js";
export { CodexConversationRuntime } from "./runtime.js";
