import { readFile, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { writePrivateJsonFile } from "./private-json-file.js";

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
  #mutations: Promise<void> = Promise.resolve();

  constructor(options: FileProviderAuthStoreOptions = {}) {
    this.path = options.path ?? join(homedir(), ".swarmx", "provider-auth.json");
  }

  async has(key: string): Promise<boolean> {
    await this.#mutations;
    const entry = (await this.read()).entries[normalizeKey(key)];
    return typeof entry === "string" && entry.length > 0;
  }

  async get(key: string): Promise<string | undefined> {
    await this.#mutations;
    const normalizedKey = normalizeKey(key);
    const entry = (await this.read()).entries[normalizedKey];
    if (!entry) return undefined;
    return entry;
  }

  async set(key: string, value: string): Promise<void> {
    const normalizedKey = normalizeKey(key);
    if (!value) throw new Error("Provider credential value is required.");
    await this.#update((document) => {
      document.entries[normalizedKey] = value;
      return document;
    });
  }

  async delete(key: string): Promise<void> {
    const normalizedKey = normalizeKey(key);
    await this.#update((document) => {
      if (!(normalizedKey in document.entries)) return undefined;
      delete document.entries[normalizedKey];
      return document;
    });
  }

  async fileMode(): Promise<number | undefined> {
    await this.#mutations;
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

  #update(
    mutation: (document: ProviderAuthDocument) => ProviderAuthDocument | undefined,
  ): Promise<void> {
    const operation = this.#mutations.then(async () => {
      const document = mutation(await this.read());
      if (document) await writePrivateJsonFile(this.path, document);
    });
    this.#mutations = operation.catch(() => undefined);
    return operation;
  }
}

function normalizeKey(key: string): string {
  const normalized = key.trim();
  if (!normalized) throw new Error("Provider credential key is required.");
  return normalized;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function isNodeError(error: unknown, code: string): boolean {
  return error instanceof Error && "code" in error && error.code === code;
}
