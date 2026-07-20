import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  EncryptedFileProviderAuthStore,
  type ProviderSecretEncryption,
  newApiAccountCredentialKey,
  providerPoolCredentialKey,
} from "./provider-auth.js";

const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((root) => rm(root, { recursive: true })));
});

describe("EncryptedFileProviderAuthStore", () => {
  it("encrypts Provider credentials at rest with restrictive permissions", async () => {
    const path = await authPath();
    const store = new EncryptedFileProviderAuthStore({
      path,
      encryption: reversibleEncryption(),
      now: () => new Date("2026-07-12T12:00:00.000Z"),
    });

    await store.set("provider-one", "super-secret-token");

    expect(await store.get("provider-one")).toBe("super-secret-token");
    expect(await store.fileMode()).toBe(0o600);
    const persisted = await readFile(path, "utf8");
    expect(persisted).not.toContain("super-secret-token");
    expect(persisted).toContain("ciphertext");
  });

  it("keeps a New API account token under a separate encrypted key", async () => {
    const path = await authPath();
    const store = new EncryptedFileProviderAuthStore({
      path,
      encryption: reversibleEncryption(),
      now: () => new Date("2026-07-12T12:00:00.000Z"),
    });
    const providerId = "provider-one";
    const accountKey = newApiAccountCredentialKey(providerId);

    await store.set(providerId, "primary-api-key");
    await store.set(accountKey, "account-access-token");

    expect(accountKey).not.toBe(providerId);
    expect(await store.get(providerId)).toBe("primary-api-key");
    expect(await store.get(accountKey)).toBe("account-access-token");
    const persisted = await readFile(path, "utf8");
    expect(persisted).not.toContain("primary-api-key");
    expect(persisted).not.toContain("account-access-token");
    expect(Object.keys(JSON.parse(persisted).entries)).toEqual([providerId, accountKey]);
  });

  it("V483 keeps pooled Provider keys in separate encrypted entries", async () => {
    const path = await authPath();
    const store = new EncryptedFileProviderAuthStore({
      path,
      encryption: reversibleEncryption(),
    });
    const providerId = "opencode-go";
    const secondaryKey = providerPoolCredentialKey(providerId, "secondary");

    await store.set(providerId, "sk-primary");
    await store.set(secondaryKey, "sk-secondary");

    expect(await store.get(providerId)).toBe("sk-primary");
    expect(await store.get(secondaryKey)).toBe("sk-secondary");
    expect(await readFile(path, "utf8")).not.toMatch(/sk-primary|sk-secondary/);
  });

  it("deletes only the requested Provider credential", async () => {
    const path = await authPath();
    const store = new EncryptedFileProviderAuthStore({
      path,
      encryption: reversibleEncryption(),
    });
    await store.set("provider-one", "one-token");
    await store.set("provider-two", "two-token");

    await store.delete("provider-one");

    expect(await store.get("provider-one")).toBeUndefined();
    expect(await store.get("provider-two")).toBe("two-token");
  });

  it("refuses plaintext fallback when secure encryption is unavailable", async () => {
    const path = await authPath();
    const store = new EncryptedFileProviderAuthStore({
      path,
      encryption: {
        isAvailable: () => false,
        encrypt: () => Buffer.from("must-not-write"),
        decrypt: () => "must-not-read",
      },
    });

    await expect(store.set("provider-one", "unsafe-token")).rejects.toThrow(
      /Secure Provider credential storage is unavailable/,
    );
    await expect(readFile(path, "utf8")).rejects.toMatchObject({ code: "ENOENT" });
  });
});

async function authPath(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "swarmx-provider-auth-"));
  temporaryRoots.push(root);
  return join(root, "provider-auth.json");
}

function reversibleEncryption(): ProviderSecretEncryption {
  return {
    isAvailable: () => true,
    encrypt: (value) => Buffer.from([...value].reverse().join(""), "utf8"),
    decrypt: (value) => Buffer.from(value).toString("utf8").split("").reverse().join(""),
  };
}
