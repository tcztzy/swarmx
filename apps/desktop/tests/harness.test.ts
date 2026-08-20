/**
 * Boots the real harness. Slow and stateful by nature: it writes the profile
 * under `$DSH_HOME` and binds a loopback port, so the home is redirected to a
 * scratch directory and the context is always disposed.
 */
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

let home: string;
let harness: Awaited<ReturnType<typeof import("../src/harness.js").startHarness>>;

beforeAll(async () => {
  home = mkdtempSync(join(tmpdir(), "swarmx-harness-"));
  process.env.DSH_HOME = home;
  const { startHarness } = await import("../src/harness.js");
  harness = await startHarness();
}, 180_000);

afterAll(async () => {
  await harness?.ctx.fiber.dispose();
  rmSync(home, { recursive: true, force: true });
});

describe("harness boot", () => {
  it("binds an os-assigned loopback port", () => {
    const url = new URL(harness.url);
    expect(url.hostname).toBe("127.0.0.1");
    expect(Number(url.port)).toBeGreaterThan(0);
  });

  it("serves the harness ui with its client boot graph", async () => {
    const response = await fetch(harness.url);
    expect(response.status).toBe(200);
    expect(await response.text()).toContain("__DSH_BOOT__");
  });

  it("loads the SwarmX conversation actions into the client boot graph", async () => {
    const response = await fetch(harness.url);
    expect(await response.text()).toContain("@swarmx/dsh-ui-conversation");
  });

  it("suppresses the web profile's system-browser handoff", () => {
    expect(harness.ctx.get("webStartup")?.openBrowser).toBe(false);
  });

  it("initializes the profile from dsh's shipped web template", () => {
    const manifest = JSON.parse(
      readFileSync(join(home, "profiles", PROFILE_DIR, "package.json"), "utf8"),
    ) as { dsh: { profile: { bundles: string[] } } };
    expect(manifest.dsh.profile.bundles).toEqual([
      "@deepseek-ai/dsh-base",
      "@deepseek-ai/dsh-web-app",
    ]);
  });

  it("writes an empty root config so the tree comes only from patch layers", () => {
    const rootConfig = readFileSync(join(home, "profiles", PROFILE_DIR, "cordis.yml"), "utf8");
    expect(rootConfig).toContain("[]");
  });
});

/** Profile directory name; matches `PROFILE` in the module under test. */
const PROFILE_DIR = "web";
