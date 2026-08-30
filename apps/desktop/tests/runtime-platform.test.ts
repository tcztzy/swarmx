import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { type DesktopPlatform, startDesktopPlatform } from "../src/runtime/platform.js";

let root: string;
let previousDshHome: string | undefined;
let platform: DesktopPlatform;
const codexCommand = process.env.SWARMX_CODEX_COMMAND ?? "codex";
const codexAvailable = spawnSync(codexCommand, ["--version"], { encoding: "utf8" }).status === 0;

beforeAll(async () => {
  root = mkdtempSync(join(tmpdir(), "swarmx-platform-"));
  previousDshHome = process.env.DSH_HOME;
  process.env.DSH_HOME = join(root, "dsh");
  platform = await startDesktopPlatform({
    runtime: "dsh",
    workspaceRoot: root,
    productHome: join(root, "product"),
    legacyProductHome: join(root, "legacy"),
  });
}, 180_000);

afterAll(async () => {
  await platform?.dispose();
  if (previousDshHome === undefined) delete process.env.DSH_HOME;
  else process.env.DSH_HOME = previousDshHome;
  rmSync(root, { recursive: true, force: true });
});

describe("desktop runtime platform", () => {
  it("loads only DSH Web and mounts the runtime registry on its origin", async () => {
    expect(platform.defaultRuntimeKind).toBe("dsh");
    expect(platform.runtimes.kinds()).toEqual(["dsh"]);
    const page = await fetch(platform.url);
    expect(page.status).toBe(200);
    expect(await page.text()).toContain("__DSH_BOOT__");

    const metadata = await fetch(new URL("/api/swarmx/conversation-runtimes", platform.url));
    expect(metadata.status).toBe(200);
    await expect(metadata.json()).resolves.toEqual({
      defaultRuntimeKind: "dsh",
      runtimeKinds: ["dsh"],
    });
  });

  it.skipIf(!codexAvailable)(
    "registers native DSH and Codex peers below the same DSH Web origin",
    async () => {
      const codexPlatform = await startDesktopPlatform({
        runtime: "codex",
        workspaceRoot: root,
        productHome: join(root, "codex-product"),
        legacyProductHome: join(root, "codex-legacy"),
        codex: { command: codexCommand, args: ["app-server"] },
      });
      try {
        expect(codexPlatform.url).toMatch(/^http:\/\/127\.0\.0\.1:/u);
        expect(codexPlatform.runtimes.kinds()).toEqual(["dsh", "codex"]);
        const metadata = await fetch(
          new URL("/api/swarmx/conversation-runtimes", codexPlatform.url),
        );
        await expect(metadata.json()).resolves.toEqual({
          defaultRuntimeKind: "codex",
          runtimeKinds: ["dsh", "codex"],
        });
      } finally {
        await codexPlatform.dispose();
      }
    },
    180_000,
  );

  it("reports partial-start cleanup failure alongside the original startup cause", async () => {
    const startupFailure = new Error("isolated startup failed");
    const cleanupFailure = new Error("isolated cleanup failed");
    vi.resetModules();
    vi.doMock("../src/harness.js", () => ({
      startHarness: vi.fn(async () => {
        throw startupFailure;
      }),
    }));
    vi.doMock("../src/runtime/swarm-recovery-owner.js", () => ({
      startSwarmRecoveryOwner: vi.fn(() => ({
        dispose: vi.fn(async () => {
          throw cleanupFailure;
        }),
      })),
    }));

    try {
      const isolated = await import("../src/runtime/platform.js");
      const rejected = isolated
        .startDesktopPlatform({
          runtime: "dsh",
          workspaceRoot: root,
          productHome: join(root, "failed-product"),
          legacyProductHome: join(root, "failed-legacy"),
        })
        .catch((error: unknown) => error);

      await expect(rejected).resolves.toBeInstanceOf(AggregateError);
      const error = (await rejected) as AggregateError;
      expect(error.errors).toEqual([startupFailure, cleanupFailure]);
      expect(error.cause).toBe(startupFailure);
    } finally {
      vi.doUnmock("../src/harness.js");
      vi.doUnmock("../src/runtime/swarm-recovery-owner.js");
      vi.resetModules();
    }
  });
});
