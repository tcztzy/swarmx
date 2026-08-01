import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  FileProviderAuthStore,
  newApiAccountCredentialKey,
  providerPoolCredentialKey,
} from "./provider-auth.js";

const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((root) => rm(root, { recursive: true })));
});

describe("FileProviderAuthStore", () => {
  it("stores Provider credentials as editable plaintext with restrictive permissions", async () => {
    const path = await authPath();
    const store = new FileProviderAuthStore({ path });

    await store.set("provider-one", "super-secret-token");

    expect(await store.get("provider-one")).toBe("super-secret-token");
    expect(await store.fileMode()).toBe(0o600);
    expect(JSON.parse(await readFile(path, "utf8"))).toEqual({
      schemaVersion: 2,
      entries: { "provider-one": "super-secret-token" },
    });
  });

  it("keeps a New API account token under a separate editable key", async () => {
    const path = await authPath();
    const store = new FileProviderAuthStore({ path });
    const providerId = "provider-one";
    const accountKey = newApiAccountCredentialKey(providerId);

    await store.set(providerId, "primary-api-key");
    await store.set(accountKey, "account-access-token");

    expect(accountKey).not.toBe(providerId);
    expect(await store.get(providerId)).toBe("primary-api-key");
    expect(await store.get(accountKey)).toBe("account-access-token");
    expect(Object.keys(JSON.parse(await readFile(path, "utf8")).entries)).toEqual([
      providerId,
      accountKey,
    ]);
  });

  it("keeps pooled Provider keys in separate editable entries", async () => {
    const path = await authPath();
    const store = new FileProviderAuthStore({ path });
    const providerId = "opencode-go";
    const secondaryKey = providerPoolCredentialKey(providerId, "secondary");

    await store.set(providerId, "sk-primary");
    await store.set(secondaryKey, "sk-secondary");

    expect(await store.get(providerId)).toBe("sk-primary");
    expect(await store.get(secondaryKey)).toBe("sk-secondary");
  });

  it("reads credentials edited directly in the auth file", async () => {
    const path = await authPath();
    await writeFile(
      path,
      JSON.stringify({ schemaVersion: 2, entries: { "provider-one": "edited-token" } }),
      { encoding: "utf8", mode: 0o600 },
    );

    const store = new FileProviderAuthStore({ path });

    await expect(store.has("provider-one")).resolves.toBe(true);
    await expect(store.get("provider-one")).resolves.toBe("edited-token");
  });

  it("does not read the old encrypted document format", async () => {
    const path = await authPath();
    await writeFile(
      path,
      JSON.stringify({
        schemaVersion: 1,
        entries: {
          "provider-one": { ciphertext: "encrypted", updatedAt: "2026-07-12T12:00:00.000Z" },
        },
      }),
      "utf8",
    );

    const store = new FileProviderAuthStore({ path });

    await expect(store.get("provider-one")).rejects.toThrow(/Unsupported Provider auth document/);
  });

  it("deletes only the requested Provider credential", async () => {
    const path = await authPath();
    const store = new FileProviderAuthStore({ path });
    await store.set("provider-one", "one-token");
    await store.set("provider-two", "two-token");

    await store.delete("provider-one");

    expect(await store.get("provider-one")).toBeUndefined();
    expect(await store.get("provider-two")).toBe("two-token");
  });
});

async function authPath(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "swarmx-provider-auth-"));
  temporaryRoots.push(root);
  return join(root, "provider-auth.json");
}
