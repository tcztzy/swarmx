import { createHmac, randomBytes } from "node:crypto";
import { readFileSync, realpathSync, statSync } from "node:fs";
import { open, readFile, realpath, stat } from "node:fs/promises";
import { basename, join } from "node:path";
import { KnowledgeBaseError } from "./errors.js";

const SALT_BYTES = 32;

export interface KnowledgeBaseWorkspace {
  readonly directory: string;
  readonly key: string;
  readonly label: string;
}

function safeName(value: string): string {
  const normalized = value
    .normalize("NFKC")
    .toLocaleLowerCase("en-US")
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-+|-+$/gu, "");
  return Array.from(normalized || "workspace")
    .slice(0, 48)
    .join("");
}

function workspaceFromCanonical(canonical: string, salt: Buffer): KnowledgeBaseWorkspace {
  if (salt.byteLength !== SALT_BYTES) {
    throw new KnowledgeBaseError("knowledge base workspace salt is invalid", "IO_ERROR");
  }
  const key = createHmac("sha256", salt).update(canonical).digest("hex").slice(0, 12);
  const label = basename(canonical) || "workspace";
  return { directory: join("workspaces", `${safeName(label)}--${key}`), key, label };
}

export async function ensureSalt(path: string): Promise<Buffer> {
  try {
    return Buffer.from(await readFile(path, "utf8"), "hex");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }

  const salt = randomBytes(SALT_BYTES);
  try {
    const handle = await open(path, "wx", 0o600);
    try {
      await handle.writeFile(salt.toString("hex"), "utf8");
      await handle.sync();
    } finally {
      await handle.close();
    }
    return salt;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    return Buffer.from(await readFile(path, "utf8"), "hex");
  }
}

export async function resolveKnowledgeBaseWorkspace(
  cwd: string,
  saltPath: string,
): Promise<KnowledgeBaseWorkspace> {
  let canonical: string;
  try {
    canonical = await realpath(cwd);
    if (!(await stat(canonical)).isDirectory()) throw new Error("not a directory");
  } catch (error) {
    throw new KnowledgeBaseError(
      "knowledge base workspace is unavailable",
      "WORKSPACE_UNAVAILABLE",
      { cause: error },
    );
  }
  const salt = await ensureSalt(saltPath);
  return workspaceFromCanonical(canonical, salt);
}

export function resolveKnowledgeBaseWorkspaceSync(
  cwd: string,
  saltPath: string,
): KnowledgeBaseWorkspace {
  let canonical: string;
  try {
    canonical = realpathSync.native(cwd);
    if (!statSync(canonical).isDirectory()) throw new Error("not a directory");
  } catch (error) {
    throw new KnowledgeBaseError(
      "knowledge base workspace is unavailable",
      "WORKSPACE_UNAVAILABLE",
      {
        cause: error,
      },
    );
  }
  return workspaceFromCanonical(canonical, Buffer.from(readFileSync(saltPath, "utf8"), "hex"));
}
