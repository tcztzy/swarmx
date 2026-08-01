import { mkdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

export interface ProviderAuthStore {
  has?(key: string): Promise<boolean>;
  get(key: string): Promise<string | undefined>;
  set(key: string, value: string): Promise<void>;
  delete(key: string): Promise<void>;
}

export class ProviderCredentialReadError extends Error {
  readonly credentialKey: string;

  constructor(credentialKey: string) {
    super(`Provider credential "${credentialKey}" could not be read. Enter a new credential.`);
    this.name = "ProviderCredentialReadError";
    this.credentialKey = credentialKey;
  }
}

export function newApiAccountCredentialKey(providerId: string): string {
  return `${normalizeKey(providerId)}:new-api-account`;
}

export function providerPoolCredentialKey(providerId: string, keyId: string): string {
  return `${normalizeKey(providerId)}:pool:${normalizeKey(keyId)}`;
}

export interface FileProviderAuthStoreOptions {
  path?: string;
}

interface ProviderAuthDocument {
  schemaVersion: 2;
  entries: Record<string, string>;
}

export class FileProviderAuthStore implements ProviderAuthStore {
  private readonly path: string;

  constructor(options: FileProviderAuthStoreOptions = {}) {
    this.path = options.path ?? join(homedir(), ".swarmx", "provider-auth.json");
  }

  async has(key: string): Promise<boolean> {
    const entry = (await this.read()).entries[normalizeKey(key)];
    return typeof entry === "string" && entry.length > 0;
  }

  async get(key: string): Promise<string | undefined> {
    const normalizedKey = normalizeKey(key);
    const entry = (await this.read()).entries[normalizedKey];
    if (!entry) return undefined;
    return entry;
  }

  async set(key: string, value: string): Promise<void> {
    const normalizedKey = normalizeKey(key);
    if (!value) throw new Error("Provider credential value is required.");
    const document = await this.read();
    await writeJsonAtomic(this.path, {
      schemaVersion: 2,
      entries: {
        ...document.entries,
        [normalizedKey]: value,
      },
    });
  }

  async delete(key: string): Promise<void> {
    const normalizedKey = normalizeKey(key);
    const document = await this.read();
    if (!(normalizedKey in document.entries)) return;
    const entries = { ...document.entries };
    delete entries[normalizedKey];
    await writeJsonAtomic(this.path, { schemaVersion: 2, entries });
  }

  async fileMode(): Promise<number | undefined> {
    try {
      return (await stat(this.path)).mode & 0o777;
    } catch (error) {
      if (isNodeError(error, "ENOENT")) return undefined;
      throw error;
    }
  }

  private async read(): Promise<ProviderAuthDocument> {
    let input: unknown;
    try {
      input = JSON.parse(await readFile(this.path, "utf8"));
    } catch (error) {
      if (isNodeError(error, "ENOENT")) return { schemaVersion: 2, entries: {} };
      throw error;
    }
    if (!isRecord(input) || input.schemaVersion !== 2 || !isRecord(input.entries)) {
      throw new Error("Unsupported Provider auth document format.");
    }
    const entries: ProviderAuthDocument["entries"] = {};
    for (const [key, value] of Object.entries(input.entries)) {
      if (typeof value !== "string") {
        throw new Error("Invalid Provider auth entry.");
      }
      entries[key] = value;
    }
    return { schemaVersion: 2, entries };
  }
}

function normalizeKey(key: string): string {
  const normalized = key.trim();
  if (!normalized) throw new Error("Provider credential key is required.");
  return normalized;
}

async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = `${path}.tmp-${process.pid}-${Date.now()}`;
  await writeFile(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, { mode: 0o600 });
  await rename(temporaryPath, path);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function isNodeError(error: unknown, code: string): boolean {
  return error instanceof Error && "code" in error && error.code === code;
}
