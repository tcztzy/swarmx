import { mkdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

export interface ProviderSecretEncryption {
  isAvailable(): boolean;
  encrypt(value: string): Uint8Array;
  decrypt(value: Uint8Array): string;
}

export interface ProviderAuthStore {
  get(key: string): Promise<string | undefined>;
  set(key: string, value: string): Promise<void>;
  delete(key: string): Promise<void>;
}

export function newApiAccountCredentialKey(providerId: string): string {
  return `${normalizeKey(providerId)}:new-api-account`;
}

export interface EncryptedFileProviderAuthStoreOptions {
  path?: string;
  encryption: ProviderSecretEncryption;
  now?: () => Date;
}

interface ProviderAuthDocument {
  schemaVersion: 1;
  entries: Record<string, { ciphertext: string; updatedAt: string }>;
}

export class EncryptedFileProviderAuthStore implements ProviderAuthStore {
  private readonly path: string;
  private readonly encryption: ProviderSecretEncryption;
  private readonly now: () => Date;

  constructor(options: EncryptedFileProviderAuthStoreOptions) {
    this.path = options.path ?? join(homedir(), ".swarmx", "provider-auth.json");
    this.encryption = options.encryption;
    this.now = options.now ?? (() => new Date());
  }

  async get(key: string): Promise<string | undefined> {
    const normalizedKey = normalizeKey(key);
    const entry = (await this.read()).entries[normalizedKey];
    if (!entry) return undefined;
    this.assertEncryptionAvailable();
    try {
      return this.encryption.decrypt(Buffer.from(entry.ciphertext, "base64"));
    } catch {
      throw new Error(`Provider credential "${normalizedKey}" could not be decrypted.`);
    }
  }

  async set(key: string, value: string): Promise<void> {
    const normalizedKey = normalizeKey(key);
    if (!value) throw new Error("Provider credential value is required.");
    this.assertEncryptionAvailable();
    const document = await this.read();
    const ciphertext = Buffer.from(this.encryption.encrypt(value)).toString("base64");
    await writeJsonAtomic(this.path, {
      ...document,
      entries: {
        ...document.entries,
        [normalizedKey]: { ciphertext, updatedAt: this.now().toISOString() },
      },
    });
  }

  async delete(key: string): Promise<void> {
    const normalizedKey = normalizeKey(key);
    const document = await this.read();
    if (!(normalizedKey in document.entries)) return;
    const entries = { ...document.entries };
    delete entries[normalizedKey];
    await writeJsonAtomic(this.path, { ...document, entries });
  }

  async fileMode(): Promise<number | undefined> {
    try {
      return (await stat(this.path)).mode & 0o777;
    } catch (error) {
      if (isNodeError(error, "ENOENT")) return undefined;
      throw error;
    }
  }

  private assertEncryptionAvailable(): void {
    if (!this.encryption.isAvailable()) {
      throw new Error("Secure Provider credential storage is unavailable on this system.");
    }
  }

  private async read(): Promise<ProviderAuthDocument> {
    let input: unknown;
    try {
      input = JSON.parse(await readFile(this.path, "utf8"));
    } catch (error) {
      if (isNodeError(error, "ENOENT")) return { schemaVersion: 1, entries: {} };
      throw error;
    }
    if (!isRecord(input) || input.schemaVersion !== 1 || !isRecord(input.entries)) {
      throw new Error("Unsupported Provider auth document format.");
    }
    const entries: ProviderAuthDocument["entries"] = {};
    for (const [key, value] of Object.entries(input.entries)) {
      if (
        !isRecord(value) ||
        typeof value.ciphertext !== "string" ||
        typeof value.updatedAt !== "string"
      ) {
        throw new Error("Invalid Provider auth entry.");
      }
      entries[key] = { ciphertext: value.ciphertext, updatedAt: value.updatedAt };
    }
    return { schemaVersion: 1, entries };
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
