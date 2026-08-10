import { createHash } from "node:crypto";
import { chmod, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type MemoryRuntimeCommandResult,
  type MemoryRuntimeEnvironmentHost,
  MemoryRuntimeEnvironmentService,
  type MemoryRuntimeManifest,
} from "./memory-runtime-environment.js";

const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })),
  );
});

describe("MemoryRuntimeEnvironmentService", () => {
  it("inspects a packaged binary without installing or mutating the host", async () => {
    const fixture = await createFixture();
    const host = fakeHost(fixture);
    const service = new MemoryRuntimeEnvironmentService(fixture.manifest, host);

    const status = await service.status();

    expect(status).toMatchObject({
      state: "ready",
      ready: true,
      version: "0.1.0",
      protocolVersion: 1,
      binaryPath: fixture.binaryPath,
      binaryDigest: fixture.manifest.targets[0]?.sha256,
    });
    expect(host.runCommand).toHaveBeenCalledWith(
      fixture.binaryPath,
      ["--version-json"],
      expect.objectContaining({
        env: expect.objectContaining({ PATH: path.dirname(fixture.binaryPath) }),
      }),
    );
    expect(host.runCommand.mock.calls.flatMap(([, args]) => args)).not.toContain("install");
    expect(host.runCommand.mock.calls.flatMap(([, args]) => args)).not.toContain("build");
  });

  it("reports missing and modified binaries with explicit repair plans", async () => {
    const fixture = await createFixture();
    const host = fakeHost(fixture);
    const service = new MemoryRuntimeEnvironmentService(fixture.manifest, host);

    await rm(fixture.binaryPath);
    const missing = await service.status();
    expect(missing).toMatchObject({ state: "missing", ready: false, repairAvailable: true });
    expect(service.plan(missing)).toEqual(
      expect.objectContaining({
        requiresConfirmation: true,
        actions: [expect.objectContaining({ kind: "restore_packaged_runtime", risk: "repair" })],
      }),
    );

    await writeFile(fixture.binaryPath, "tampered", "utf8");
    await chmod(fixture.binaryPath, 0o755);
    const modified = await service.status();
    expect(modified).toMatchObject({ state: "invalid", ready: false, reason: "digest_mismatch" });
  });

  it("revalidates digest and protocol before returning a secret-minimal launch", async () => {
    const fixture = await createFixture();
    const host = fakeHost(fixture);
    const service = new MemoryRuntimeEnvironmentService(fixture.manifest, host);
    const status = await service.status();
    const launch = await service.launchSpec(status, {
      memoryRoot: path.join(fixture.root, "memory"),
    });

    expect(launch).toEqual({
      program: fixture.binaryPath,
      args: ["serve", "--root", path.join(fixture.root, "memory"), "--stdio"],
      cwd: fixture.root,
      env: {
        HOME: fixture.root,
        PATH: path.dirname(fixture.binaryPath),
        RUST_BACKTRACE: "0",
      },
      binaryDigest: fixture.manifest.targets[0]?.sha256,
      protocolVersion: 1,
      runtimeVersion: "0.1.0",
      memoryRoot: path.join(fixture.root, "memory"),
    });
    expect(launch.env).not.toHaveProperty("OPENAI_API_KEY");
    expect(launch.env).not.toHaveProperty("ANTHROPIC_API_KEY");

    await writeFile(fixture.binaryPath, "changed after inspection", "utf8");
    await expect(
      service.launchSpec(status, { memoryRoot: path.join(fixture.root, "memory") }),
    ).rejects.toThrow(/changed after the supplied health check/);
  });

  it("fails closed for unsupported targets and incompatible handshakes", async () => {
    const fixture = await createFixture();
    const unsupported = new MemoryRuntimeEnvironmentService(fixture.manifest, {
      ...fakeHost(fixture),
      architecture: "x64",
    });
    expect(await unsupported.status()).toMatchObject({
      state: "unsupported",
      ready: false,
      repairAvailable: false,
    });

    const badHost = fakeHost(fixture, { protocolVersion: 2 });
    const incompatible = new MemoryRuntimeEnvironmentService(fixture.manifest, badHost);
    expect(await incompatible.status()).toMatchObject({
      state: "invalid",
      ready: false,
      reason: "incompatible_protocol",
    });
  });
});

interface Fixture {
  root: string;
  binaryPath: string;
  manifest: MemoryRuntimeManifest;
}

async function createFixture(): Promise<Fixture> {
  const root = await mkdtemp(path.join(tmpdir(), "swarmx-mem-"));
  temporaryRoots.push(root);
  const binaryPath = path.join(root, "runtime", "swarmx-mem");
  const bytes = "verified memory runtime";
  await mkdir(path.dirname(binaryPath), { recursive: true });
  await writeFile(binaryPath, bytes, "utf8");
  await chmod(binaryPath, 0o755);
  return {
    root,
    binaryPath,
    manifest: {
      schemaVersion: 1,
      runtimeVersion: "0.1.0",
      protocolVersion: 1,
      targets: [
        {
          platform: "darwin",
          architecture: "arm64",
          path: binaryPath,
          sha256: `sha256:${createHash("sha256").update(bytes).digest("hex")}`,
        },
      ],
    },
  };
}

function fakeHost(fixture: Fixture, options: { protocolVersion?: number } = {}) {
  const runCommand = vi.fn(
    async (_program: string, args: readonly string[]): Promise<MemoryRuntimeCommandResult> => {
      if (args[0] !== "--version-json") {
        return { exitCode: 1, stdout: "", stderr: "unexpected command" };
      }
      return {
        exitCode: 0,
        stdout: JSON.stringify({
          name: "swarmx-mem",
          version: "0.1.0",
          protocolVersion: options.protocolVersion ?? 1,
        }),
        stderr: "",
      };
    },
  );
  return {
    platform: "darwin",
    architecture: "arm64",
    homeDir: fixture.root,
    env: {
      PATH: "/usr/local/bin:/usr/bin",
      HOME: fixture.root,
      OPENAI_API_KEY: "must-not-forward",
      ANTHROPIC_API_KEY: "must-not-forward",
    },
    runCommand,
  } satisfies MemoryRuntimeEnvironmentHost & { runCommand: typeof runCommand };
}
