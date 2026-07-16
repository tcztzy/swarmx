import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type DesktopUpdateState,
  NpmDesktopUpdateService,
  compareSemanticVersions,
} from "./updater.js";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("NpmDesktopUpdateService", () => {
  it("compares semantic versions without treating prereleases as stable upgrades", () => {
    expect(compareSemanticVersions("3.0.2", "3.0.1")).toBe(1);
    expect(compareSemanticVersions("3.0.1", "3.0.1")).toBe(0);
    expect(compareSemanticVersions("3.1.0-beta.2", "3.1.0-beta.10")).toBe(-1);
    expect(compareSemanticVersions("3.1.0", "3.1.0-beta.10")).toBe(1);
  });

  it("keeps update UI hidden when unsupported, current, offline, or handed prerelease latest metadata", async () => {
    const unsupportedFetch = vi.fn();
    const unsupported = service({ supported: false, fetch: unsupportedFetch });
    await expect(unsupported.check()).resolves.toEqual({
      phase: "hidden",
      currentVersion: "3.0.1",
    });
    expect(unsupportedFetch).not.toHaveBeenCalled();

    const current = service({ fetch: vi.fn(async () => releaseResponse("3.0.1")) });
    await expect(current.check()).resolves.toMatchObject({ phase: "hidden" });

    const offline = service({
      fetch: vi.fn(async () => {
        throw new Error("offline");
      }),
    });
    await expect(offline.check()).resolves.toMatchObject({ phase: "hidden" });

    const prerelease = service({ fetch: vi.fn(async () => releaseResponse("3.1.0-beta.1")) });
    await expect(prerelease.check()).resolves.toMatchObject({ phase: "hidden" });
  });

  it("publishes availability, verified download progress, safe install, and relaunch handoff", async () => {
    const root = await updateRoot();
    const tarball = Buffer.from("verified desktop tarball");
    const version = "3.0.2";
    const runCommand = vi.fn(async (_command: string, args: string[]) => {
      const prefixIndex = args.indexOf("--prefix");
      const prefix = args[prefixIndex + 1];
      if (!prefix) throw new Error("missing prefix");
      const appPath = join(prefix, "node_modules", "@swarmx", "desktop");
      await mkdir(appPath, { recursive: true });
      await writeFile(
        join(appPath, "package.json"),
        JSON.stringify({ name: "@swarmx/desktop", version }),
      );
    });
    const restart = vi.fn();
    const fetch = releaseAndTarballFetch(version, tarball);
    const updates = service({ root, fetch, runCommand, restart });
    const states: DesktopUpdateState[] = [];
    updates.subscribe((state) => states.push(state));

    await expect(updates.check()).resolves.toEqual({
      phase: "available",
      currentVersion: "3.0.1",
      latestVersion: version,
    });
    await expect(updates.startUpdate()).resolves.toMatchObject({
      phase: "restarting",
      progress: 100,
    });

    expect(runCommand).toHaveBeenCalledTimes(1);
    expect(runCommand.mock.calls[0]?.[0]).toBe("npm");
    expect(runCommand.mock.calls[0]?.[1]).toEqual(
      expect.arrayContaining([
        "install",
        "--omit=dev",
        "--no-audit",
        "--no-fund",
        "--ignore-scripts",
        "--package-lock=false",
      ]),
    );
    expect(restart).toHaveBeenCalledWith(join(root, version, "node_modules", "@swarmx", "desktop"));
    expect(states.map((state) => state.phase)).toEqual(
      expect.arrayContaining(["available", "downloading", "installing", "restarting"]),
    );
    expect(states.some((state) => state.phase === "downloading" && state.progress === 100)).toBe(
      true,
    );
  });

  it("rejects an integrity mismatch before npm installation and stays retryable", async () => {
    const tarball = Buffer.from("tampered tarball");
    const runCommand = vi.fn();
    const restart = vi.fn();
    const fetch = releaseAndTarballFetch("3.0.2", tarball, integrity(Buffer.from("expected")));
    const updates = service({ fetch, runCommand, restart });

    await updates.check();
    await expect(updates.startUpdate()).resolves.toMatchObject({
      phase: "available",
      latestVersion: "3.0.2",
      error: expect.stringContaining("integrity verification"),
    });
    expect(runCommand).not.toHaveBeenCalled();
    expect(restart).not.toHaveBeenCalled();
  });

  it("verifies the installed package version before replacing or relaunching", async () => {
    const root = await updateRoot();
    const runCommand = vi.fn(async (_command: string, args: string[]) => {
      const prefix = args[args.indexOf("--prefix") + 1];
      if (!prefix) throw new Error("missing prefix");
      const appPath = join(prefix, "node_modules", "@swarmx", "desktop");
      await mkdir(appPath, { recursive: true });
      await writeFile(
        join(appPath, "package.json"),
        JSON.stringify({ name: "@swarmx/desktop", version: "9.9.9" }),
      );
    });
    const restart = vi.fn();
    const updates = service({
      root,
      fetch: releaseAndTarballFetch("3.0.2", Buffer.from("desktop")),
      runCommand,
      restart,
    });

    await updates.check();
    await expect(updates.startUpdate()).resolves.toMatchObject({
      phase: "available",
      error: expect.stringContaining("did not match"),
    });
    expect(restart).not.toHaveBeenCalled();
  });

  it("preserves a known available update when a later background check fails", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(releaseResponse("3.0.2"))
      .mockRejectedValueOnce(new Error("temporary registry outage"));
    const updates = service({ fetch });

    await expect(updates.check()).resolves.toMatchObject({ phase: "available" });
    await expect(updates.check()).resolves.toEqual({
      phase: "available",
      currentVersion: "3.0.1",
      latestVersion: "3.0.2",
    });
  });
});

function service({
  supported = true,
  root,
  fetch = vi.fn(async () => releaseResponse("3.0.2")),
  runCommand = vi.fn(),
  restart = vi.fn(),
}: {
  supported?: boolean;
  root?: string;
  fetch?: (input: string | URL, init?: RequestInit) => Promise<Response>;
  runCommand?: (command: string, args: string[]) => Promise<void>;
  restart?: (appPath: string) => Promise<void> | void;
} = {}) {
  return new NpmDesktopUpdateService({
    currentVersion: "3.0.1",
    supported,
    ...(root ? { updatesRoot: root } : {}),
    fetch,
    runCommand,
    restart,
    restartDelayMs: 0,
  });
}

function releaseAndTarballFetch(
  version: string,
  tarball: Buffer,
  declaredIntegrity = integrity(tarball),
) {
  return vi.fn(async (input: string | URL) => {
    const url = String(input);
    if (url.endsWith("/latest")) return releaseResponse(version, declaredIntegrity);
    return new Response(tarball, {
      status: 200,
      headers: { "content-length": String(tarball.byteLength) },
    });
  });
}

function releaseResponse(version: string, declaredIntegrity = integrity(Buffer.from("release"))) {
  return new Response(
    JSON.stringify({
      name: "@swarmx/desktop",
      version,
      dist: {
        tarball: `https://registry.npmjs.org/@swarmx/desktop/-/desktop-${version}.tgz`,
        integrity: declaredIntegrity,
      },
    }),
    { status: 200, headers: { "content-type": "application/json" } },
  );
}

function integrity(value: Buffer): string {
  return `sha512-${createHash("sha512").update(value).digest("base64")}`;
}

async function updateRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "swarmx-update-test-"));
  roots.push(root);
  return root;
}
