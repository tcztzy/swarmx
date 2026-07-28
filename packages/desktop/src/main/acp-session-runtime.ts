import { constants } from "node:fs";
import { chmod, copyFile, lstat, mkdir, mkdtemp, readdir, rm, rmdir, stat } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import type { AgentCompositionPlan, ExternalAcpSessionBinding, SessionData } from "@swarmx/core";

const EPHEMERAL_HOME_PREFIX = "codex-attachment-";
const EPHEMERAL_HOME_MAX_AGE_MS = 24 * 60 * 60 * 1000;
const MAX_CONFIG_FILE_BYTES = 4 * 1024 * 1024;
const MAX_SUPPORT_TREE_BYTES = 16 * 1024 * 1024;
const MAX_SUPPORT_TREE_ENTRIES = 2_048;
const CONFIG_FILES = ["auth.json", "config.toml"] as const;
const SUPPORT_DIRECTORIES = ["agents", "skills", "rules"] as const;

export interface ExternalAcpSessionIdentity {
  harnessId: string;
  modelId: string;
  modelSupplyId?: string;
  agentProfileId?: string;
  cwd?: string;
}

export interface EphemeralCodexHome {
  path: string;
  env: NodeJS.ProcessEnv;
  cleanup(): Promise<void>;
}

export interface CreateEphemeralCodexHomeOptions {
  sourceHome?: string;
  storageRoot?: string;
  homeDir?: string;
  now?: () => number;
}

export function externalAcpSessionIdentity(
  plan: AgentCompositionPlan,
  harnessId: string,
  cwd?: string,
): ExternalAcpSessionIdentity | null {
  if (!plan.modelId) return null;
  return {
    harnessId,
    modelId: plan.modelId,
    ...(plan.modelSupplyId ? { modelSupplyId: plan.modelSupplyId } : {}),
    agentProfileId: plan.agentProfileId ?? plan.id,
    ...(cwd ? { cwd } : {}),
  };
}

export function matchingExternalAcpSessionId(
  session: SessionData,
  identity: ExternalAcpSessionIdentity,
): string | undefined {
  const binding = session.externalAcpSession;
  if (!binding) return undefined;
  return sameExternalAcpSessionIdentity(binding, identity) ? binding.sessionId : undefined;
}

export function createExternalAcpSessionBinding(
  identity: ExternalAcpSessionIdentity,
  sessionId: string,
  existing?: ExternalAcpSessionBinding,
  now = new Date().toISOString(),
): ExternalAcpSessionBinding {
  return {
    ...identity,
    sessionId,
    createdAt:
      existing?.sessionId === sessionId && sameExternalAcpSessionIdentity(existing, identity)
        ? existing.createdAt
        : now,
    updatedAt: now,
  };
}

export function latestUserMessageHasAttachments(session: SessionData): boolean {
  for (let index = session.messages.length - 1; index >= 0; index -= 1) {
    const message = session.messages[index];
    if (message?.kind === "message" && message.role === "user") {
      return (message.attachments?.length ?? 0) > 0;
    }
  }
  return false;
}

export async function createEphemeralCodexHome(
  options: CreateEphemeralCodexHomeOptions = {},
): Promise<EphemeralCodexHome> {
  const userHome = options.homeDir ?? homedir();
  const sourceHome = resolveCodexHome(options.sourceHome, userHome);
  const storageRoot = path.resolve(
    options.storageRoot ?? path.join(userHome, ".swarmx", "acp-ephemeral"),
  );
  await mkdir(storageRoot, { recursive: true, mode: 0o700 });
  await removeStaleEphemeralHomes(storageRoot, options.now?.() ?? Date.now());

  const temporaryHome = await mkdtemp(path.join(storageRoot, EPHEMERAL_HOME_PREFIX));
  await chmod(temporaryHome, 0o700);
  try {
    for (const filename of CONFIG_FILES) {
      await copyBoundedConfigFile(
        path.join(sourceHome, filename),
        path.join(temporaryHome, filename),
      );
    }
    for (const dirname of SUPPORT_DIRECTORIES) {
      await copyOptionalSupportTree(
        path.join(sourceHome, dirname),
        path.join(temporaryHome, dirname),
      );
    }
    const logs = path.join(temporaryHome, "logs");
    await mkdir(logs, { mode: 0o700 });

    let cleaned = false;
    return {
      path: temporaryHome,
      env: {
        CODEX_HOME: temporaryHome,
        APP_SERVER_LOGS: logs,
      },
      cleanup: async () => {
        if (cleaned) return;
        await removeEphemeralHome(temporaryHome);
        cleaned = true;
      },
    };
  } catch (error) {
    await removeEphemeralHome(temporaryHome);
    throw error;
  }
}

export function resolveCodexHome(value: string | undefined, userHome = homedir()): string {
  const configured = value?.trim();
  if (!configured) return path.join(userHome, ".codex");
  if (configured === "~") return userHome;
  if (configured.startsWith("~/")) return path.join(userHome, configured.slice(2));
  return path.resolve(configured);
}

export function sameExternalAcpSessionIdentity(
  binding: ExternalAcpSessionBinding,
  identity: ExternalAcpSessionIdentity,
): boolean {
  return (
    binding.harnessId === identity.harnessId &&
    binding.modelId === identity.modelId &&
    binding.modelSupplyId === identity.modelSupplyId &&
    binding.agentProfileId === identity.agentProfileId &&
    binding.cwd === identity.cwd
  );
}

async function copyBoundedConfigFile(source: string, target: string): Promise<void> {
  const info = await stat(source).catch(() => null);
  if (!info) return;
  if (!info.isFile()) throw new Error(`Codex input is not a regular file: ${source}`);
  if (info.size > MAX_CONFIG_FILE_BYTES) {
    throw new Error(`Codex input exceeds the isolated-home limit: ${source}`);
  }
  await copyFile(source, target, constants.COPYFILE_EXCL);
  await chmod(target, 0o600);
}

async function copyOptionalSupportTree(source: string, target: string): Promise<void> {
  const info = await lstat(source).catch(() => null);
  if (!info?.isDirectory() || info.isSymbolicLink()) return;
  const budget = { bytes: 0, entries: 0 };
  try {
    await copyReadOnlyTree(source, target, budget);
  } catch {
    await removeEphemeralHome(target);
  }
}

async function copyReadOnlyTree(
  source: string,
  target: string,
  budget: { bytes: number; entries: number },
): Promise<void> {
  const info = await lstat(source);
  budget.entries += 1;
  if (budget.entries > MAX_SUPPORT_TREE_ENTRIES) {
    throw new Error("Codex support inputs exceed the isolated-home entry limit.");
  }
  if (info.isSymbolicLink()) return;
  if (info.isDirectory()) {
    await mkdir(target, { mode: 0o700 });
    const entries = await readdir(source);
    for (const entry of entries.sort()) {
      await copyReadOnlyTree(path.join(source, entry), path.join(target, entry), budget);
    }
    await chmod(target, 0o500);
    return;
  }
  if (!info.isFile()) return;
  budget.bytes += info.size;
  if (budget.bytes > MAX_SUPPORT_TREE_BYTES) {
    throw new Error("Codex support inputs exceed the isolated-home byte limit.");
  }
  await copyFile(source, target, constants.COPYFILE_EXCL);
  await chmod(target, 0o400);
}

async function removeStaleEphemeralHomes(storageRoot: string, now: number): Promise<void> {
  const entries = await readdir(storageRoot, { withFileTypes: true });
  await Promise.all(
    entries.flatMap((entry) => {
      if (!entry.isDirectory() || !entry.name.startsWith(EPHEMERAL_HOME_PREFIX)) return [];
      const candidate = path.join(storageRoot, entry.name);
      return [
        stat(candidate).then(async (info) => {
          if (now - info.mtimeMs > EPHEMERAL_HOME_MAX_AGE_MS) {
            await removeEphemeralHome(candidate);
          }
        }),
      ];
    }),
  );
}

async function removeEphemeralHome(target: string): Promise<void> {
  const info = await lstat(target).catch(() => null);
  if (!info) return;
  if (info.isDirectory() && !info.isSymbolicLink()) {
    await chmod(target, 0o700);
    const entries = await readdir(target);
    for (const entry of entries) {
      await removeEphemeralHome(path.join(target, entry));
    }
    await rmdir(target);
    return;
  }
  if (!info.isSymbolicLink()) {
    await chmod(target, 0o600);
  }
  await rm(target, { force: true });
}
