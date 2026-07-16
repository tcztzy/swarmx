import { chmod, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CodexAccessTokenResolver } from "./codex-auth.js";

const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((root) => rm(root, { recursive: true })));
});

describe("CodexAccessTokenResolver", () => {
  it("prefers an explicitly supplied automation token", async () => {
    const refresh = vi.fn();
    const resolver = new CodexAccessTokenResolver({
      env: { CODEX_ACCESS_TOKEN: "environment-token" },
      authPath: "/missing/auth.json",
      refresh,
    });

    await expect(resolver.available()).resolves.toBe(true);
    await expect(resolver.resolve()).resolves.toBe("environment-token");
    expect(refresh).not.toHaveBeenCalled();
  });

  it("reads only the managed access token from private Codex auth storage", async () => {
    const authPath = await writeAuthFile(fakeJwt({ exp: 2_000_000_000 }), 0o600);
    const refresh = vi.fn();
    const resolver = new CodexAccessTokenResolver({
      env: {},
      authPath,
      refresh,
      now: () => 1_900_000_000_000,
    });

    await expect(resolver.available()).resolves.toBe(true);
    await expect(resolver.resolve()).resolves.toBe(fakeJwt({ exp: 2_000_000_000 }));
    expect(refresh).not.toHaveBeenCalled();
  });

  it("asks App Server to refresh an expiring token and rereads the managed token", async () => {
    const authPath = await writeAuthFile(fakeJwt({ exp: 1_900_000_100 }), 0o600);
    const refreshedToken = fakeJwt({ exp: 2_000_000_000 });
    const refresh = vi.fn(async () => {
      await writeFile(authPath, JSON.stringify({ tokens: { access_token: refreshedToken } }), {
        mode: 0o600,
      });
    });
    const resolver = new CodexAccessTokenResolver({
      env: {},
      authPath,
      refresh,
      now: () => 1_900_000_000_000,
    });

    const [first, second] = await Promise.all([resolver.resolve(), resolver.resolve()]);

    expect(first).toBe(refreshedToken);
    expect(second).toBe(refreshedToken);
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it("rejects Codex auth storage that is readable by other users", async () => {
    const authPath = await writeAuthFile("access-token", 0o644);
    const resolver = new CodexAccessTokenResolver({ env: {}, authPath, refresh: vi.fn() });

    await expect(resolver.available()).resolves.toBe(false);
    await expect(resolver.resolve()).rejects.toThrow(/private file/);
  });

  it("reports a missing managed sign-in without leaking file contents", async () => {
    const root = await temporaryRoot();
    const resolver = new CodexAccessTokenResolver({
      env: {},
      authPath: join(root, "missing.json"),
      refresh: vi.fn(),
    });

    await expect(resolver.resolve()).rejects.toThrow(/Sign in with Codex/);
  });
});

async function writeAuthFile(accessToken: string, mode: number): Promise<string> {
  const root = await temporaryRoot();
  const path = join(root, "auth.json");
  await writeFile(
    path,
    JSON.stringify({
      tokens: {
        access_token: accessToken,
        refresh_token: "must-not-be-returned",
      },
    }),
    { mode: 0o600 },
  );
  await chmod(path, mode);
  return path;
}

async function temporaryRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "swarmx-codex-auth-"));
  temporaryRoots.push(root);
  return root;
}

function fakeJwt(claims: Record<string, unknown>): string {
  return `header.${Buffer.from(JSON.stringify(claims)).toString("base64url")}.signature`;
}
