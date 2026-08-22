/**
 * Boots the real harness. Slow and stateful by nature: it writes the profile
 * under `$DSH_HOME` and binds a loopback port, so the home is redirected to a
 * scratch directory and the context is always disposed.
 */
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

let home: string;
let harness: Awaited<ReturnType<typeof import("../src/harness.js").startHarness>>;
let fixtureEntryPath: string;

beforeAll(async () => {
  home = mkdtempSync(join(tmpdir(), "swarmx-harness-"));
  process.env.DSH_HOME = home;
  const profileDir = join(home, "profiles", PROFILE_DIR);
  const fixtureDir = join(profileDir, "node_modules", "@fixture", "profile-plugin");
  fixtureEntryPath = join(fixtureDir, "index.js");
  mkdirSync(fixtureDir, { recursive: true });
  writeFileSync(
    join(profileDir, "package.json"),
    `${JSON.stringify(
      {
        name: "dsh-profile-web",
        private: true,
        dependencies: { "@fixture/profile-plugin": "link:fixture" },
        dsh: {
          profile: {
            bundles: [
              "@deepseek-ai/dsh-base",
              "@deepseek-ai/dsh-web-app",
              "@fixture/profile-plugin",
            ],
          },
        },
      },
      undefined,
      2,
    )}\n`,
  );
  writeFileSync(join(profileDir, "cordis.patch.yml"), "[]\n");
  writeFileSync(join(profileDir, "pnpm-workspace.yaml"), "packages:\n  - .\n");
  writeFileSync(
    join(fixtureDir, "package.json"),
    `${JSON.stringify(
      {
        name: "@fixture/profile-plugin",
        type: "module",
        main: "./index.js",
        dsh: { bundle: { patch: "./cordis.patch.yml" } },
      },
      undefined,
      2,
    )}\n`,
  );
  writeFileSync(
    join(fixtureDir, "cordis.patch.yml"),
    "- insert:\n    - id: profile-fixture\n      name: '@fixture/profile-plugin'\n",
  );
  writeFileSync(
    fixtureEntryPath,
    "globalThis.__SWARMX_PROFILE_FIXTURE__ = true;\nexport const name = 'profile-fixture';\nexport function apply() {}\n",
  );
  const { startHarness } = await import("../src/harness.js");
  harness = await startHarness();
}, 180_000);

afterAll(async () => {
  await harness?.ctx.fiber.dispose();
  delete (globalThis as { __SWARMX_PROFILE_FIXTURE__?: boolean }).__SWARMX_PROFILE_FIXTURE__;
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

  it("loads out-of-tree bundles installed in the active profile", () => {
    expect(
      (globalThis as { __SWARMX_PROFILE_FIXTURE__?: boolean }).__SWARMX_PROFILE_FIXTURE__,
    ).toBe(true);
    const entries = harness.ctx.loader.entries() as Array<{
      options: { id?: string; name?: string };
    }>;
    const fixtureEntry = entries.find((entry) => entry.options.id === "profile-fixture");
    expect(fixtureEntry?.options.name).toBe(realpathSync(fixtureEntryPath));
  });

  it("mounts the Science Journal and adds its client view to the boot graph", async () => {
    expect(harness.ctx.get("science")).toBeDefined();
    const typert = harness.ctx.get("typert") as
      | { local: { get(endpoint: string): { service: string } | undefined } }
      | undefined;
    expect(typert?.local.get("science/createProject")).toMatchObject({ service: "science" });
    expect(typert?.local.get("science/executeNotebookCell")).toMatchObject({ service: "science" });
    expect(typert?.local.get("science/createDocument")).toMatchObject({ service: "science" });
    expect(typert?.local.get("science/createFigure")).toMatchObject({ service: "science" });
    expect(typert?.local.get("science/modifyDocument")).toMatchObject({ service: "science" });
    expect(typert?.local.get("science/modifyFigureCode")).toMatchObject({ service: "science" });
    expect(typert?.local.get("science/defineExperiment")).toMatchObject({ service: "science" });
    expect(typert?.local.get("science/exportProject")).toMatchObject({ service: "science" });
    expect(harness.ctx.tools.get("science_record")).toBeDefined();
    expect(harness.ctx.tools.get("science_export")).toBeDefined();
    expect(harness.ctx.get("spillStore")).toBeDefined();
    const response = await fetch(harness.url);
    const html = await response.text();
    expect(html).toContain("@swarmx/dsh-ui-science");
  });

  it("suppresses the web profile's system-browser handoff", () => {
    expect(harness.ctx.get("webStartup")?.openBrowser).toBe(false);
  });

  it("initializes the profile from dsh's shipped web template", () => {
    const manifest = JSON.parse(
      readFileSync(join(home, "profiles", PROFILE_DIR, "package.json"), "utf8"),
    ) as { dsh: { profile: { bundles: string[] } } };
    expect(manifest.dsh.profile.bundles.slice(0, 2)).toEqual([
      "@deepseek-ai/dsh-base",
      "@deepseek-ai/dsh-web-app",
    ]);
    expect(manifest.dsh.profile.bundles).toContain("@fixture/profile-plugin");
  });

  it("writes an empty root config so the tree comes only from patch layers", () => {
    const rootConfig = readFileSync(join(home, "profiles", PROFILE_DIR, "cordis.yml"), "utf8");
    expect(rootConfig).toContain("[]");
  });
});

/** Profile directory name; matches `PROFILE` in the module under test. */
const PROFILE_DIR = "web";
