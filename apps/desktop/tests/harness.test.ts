/**
 * Boots the real harness. Slow and stateful by nature: it writes the profile
 * under `$DSH_HOME` and binds a loopback port, so the home is redirected to a
 * scratch directory and the context is always disposed.
 */
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { healProfilesModuleFallback } from "@deepseek-ai/dsh-app-boot";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

let home: string;
let harness: Awaited<ReturnType<typeof import("../src/harness.js").startHarness>>;
let fixtureEntryPath: string;
let browserCookie: string;
let tokenExchange: Response;
const moduleRequire = createRequire(import.meta.url);

function fetchHarness(pathname = "/"): Promise<Response> {
  return fetch(new URL(pathname, harness.url), { headers: { cookie: browserCookie } });
}

beforeAll(async () => {
  home = mkdtempSync(join(tmpdir(), "swarmx-harness-"));
  process.env.DSH_HOME = home;
  const profileDir = join(home, "profiles", PROFILE_DIR);
  const fixtureDir = join(profileDir, "node_modules", "@fixture", "profile-plugin");
  fixtureEntryPath = join(fixtureDir, "index.js");
  await healProfilesModuleFallback({
    installAnchor: moduleRequire.resolve("@deepseek-ai/dsh/package.json"),
    home,
  });
  const fallbackScope = join(home, "profiles", "node_modules", "@deepseek-ai");
  mkdirSync(fallbackScope, { recursive: true });
  for (const name of ["dsh-client-runtime", "dsh-host-apiproxy"]) {
    const staleDir = join(home, "stale", name);
    mkdirSync(staleDir, { recursive: true });
    writeFileSync(
      join(staleDir, "package.json"),
      `${JSON.stringify({ name: `@deepseek-ai/${name}`, version: "0.1.0-rc.8" })}\n`,
    );
    symlinkSync(staleDir, join(fallbackScope, name), "junction");
  }
  const preservedFallback = join(home, "profiles", "node_modules", "preserved-package");
  mkdirSync(preservedFallback);
  writeFileSync(
    join(preservedFallback, "package.json"),
    `${JSON.stringify({ name: "preserved-package", version: "1.0.0" })}\n`,
  );
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
  writeFileSync(
    join(profileDir, "cordis.patch.yml"),
    `- id: swarmx-science
  config:
    embedArtifactMetadata: false
    maxArtifactBytes: 1048576
- id: swarmx-pkb
  config:
    maxConceptBytes: 131072
- id: swarmx-swarm
  config:
    monitorStallMs: 123456
`,
  );
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
  harness = await startHarness({ productHome: join(home, "swarmx") });
  tokenExchange = await fetch(harness.url, { redirect: "manual" });
  const setCookie = tokenExchange.headers.get("set-cookie");
  if (setCookie === null) throw new Error("Harness launch-token exchange did not set a cookie.");
  browserCookie = setCookie.split(";", 1)[0] as string;
}, 180_000);

afterAll(async () => {
  await harness?.ctx.fiber.dispose();
  delete (globalThis as { __SWARMX_PROFILE_FIXTURE__?: boolean }).__SWARMX_PROFILE_FIXTURE__;
  rmSync(home, { recursive: true, force: true });
});

describe("harness boot", () => {
  it("returns an authenticated launch URL on an os-assigned loopback port", () => {
    const url = new URL(harness.url);
    expect(url.hostname).toBe("127.0.0.1");
    expect(Number(url.port)).toBeGreaterThan(0);
    expect(url.searchParams.has("token")).toBe(true);
    expect(tokenExchange.status).toBe(303);
    expect(tokenExchange.headers.get("location")).toBe("/");
  });

  it("serves the harness ui with its client boot graph", async () => {
    const response = await fetchHarness();
    expect(response.status).toBe(200);
    expect(await response.text()).toContain("__DSH_BOOT__");
  });

  it("loads the SwarmX conversation actions into the client boot graph", async () => {
    const response = await fetchHarness();
    expect(await response.text()).toContain("@swarmx/dsh-ui-conversation");
  });

  it("V225 loads out-of-tree bundles from the active profile dependency closure", () => {
    expect(
      (globalThis as { __SWARMX_PROFILE_FIXTURE__?: boolean }).__SWARMX_PROFILE_FIXTURE__,
    ).toBe(true);
    const entries = [...harness.ctx.loader.entries()] as Array<{
      options: { id?: string; name?: string };
    }>;
    const fixtureEntry = entries.find((entry) => entry.options.id === "profile-fixture");
    expect(fixtureEntry?.options.name).toBe(realpathSync(fixtureEntryPath));
  });

  it("reconciles the exact alpha.4 DSH closure while resolving product subpaths", () => {
    const profileRequire = createRequire(join(home, "profiles", PROFILE_DIR, "package.json"));
    const subagentManifest = JSON.parse(
      readFileSync(profileRequire.resolve("@deepseek-ai/dsh-tool-subagent/package.json"), "utf8"),
    ) as { exports: Record<string, unknown>; version: string };
    expect(subagentManifest.version).toBe("0.1.2-alpha.4");
    expect(subagentManifest.exports).toHaveProperty("./model-selection-settings");
    expect(profileRequire.resolve("@swarmx/dsh-science/preset")).toBe(
      moduleRequire.resolve("@swarmx/dsh-science/preset"),
    );
    for (const scopeDir of [
      join(home, "profiles", "node_modules", "@deepseek-ai"),
      join(home, "profiles", PROFILE_DIR, ".dsh-module-fallback", "node_modules", "@deepseek-ai"),
    ]) {
      if (!existsSync(scopeDir)) continue;
      for (const name of readdirSync(scopeDir).filter((entry) => entry.startsWith("dsh"))) {
        const manifest = JSON.parse(readFileSync(join(scopeDir, name, "package.json"), "utf8")) as {
          version: string;
        };
        expect(manifest.version, name).toBe("0.1.2-alpha.4");
      }
    }
    for (const obsolete of ["@deepseek-ai/dsh-client-runtime", "@deepseek-ai/dsh-host-apiproxy"]) {
      expect(() => profileRequire.resolve(`${obsolete}/package.json`)).toThrow();
    }
    expect(existsSync(join(home, "profiles", "node_modules", "preserved-package"))).toBe(true);
  });

  it("mounts read-only Git/DVC UIs and keeps DVC mutations off model tools", async () => {
    expect(harness.ctx.get("gitUi")).toBeDefined();
    expect(harness.ctx.get("dvc")).toBeDefined();
    expect(harness.ctx.get("dvcUi")).toBeDefined();
    const typert = harness.ctx.get("typert") as
      | { local: { get(endpoint: string): { service: string } | undefined } }
      | undefined;
    expect(typert?.local.get("gitUi/snapshot")).toMatchObject({ service: "gitUi" });
    expect(typert?.local.get("dvcUi/snapshot")).toMatchObject({ service: "dvcUi" });
    expect(harness.ctx.tools.get("git")).toBeUndefined();
    expect(harness.ctx.tools.get("dvc")).toBeUndefined();

    const entries = [...harness.ctx.loader.entries()] as Array<{
      options: { id?: string; name?: string };
    }>;
    const dvcEntry = entries.find((entry) => entry.options.id === "swarmx-dvc");
    const dvcUiEntry = entries.find((entry) => entry.options.id === "swarmx-ui-dvc");
    const expectedDvc = moduleRequire.resolve("@swarmx/dsh-dvc");
    expect(dvcEntry?.options.name).toBe(realpathSync(expectedDvc));
    expect(dvcUiEntry).toBeDefined();
    expect(entries.indexOf(dvcUiEntry as (typeof entries)[number])).toBeGreaterThan(
      entries.indexOf(dvcEntry as (typeof entries)[number]),
    );

    const response = await fetchHarness();
    const html = await response.text();
    expect(html).toContain("@swarmx/dsh-ui-git");
    expect(html).toContain("@swarmx/dsh-ui-dvc");
  });

  it("mounts the Science Journal and adds its client integration to the boot graph", async () => {
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
    expect(typert?.local.get("science/searchLiterature")).toMatchObject({ service: "science" });
    expect(harness.ctx.tools.get("science_record")).toBeUndefined();
    expect(harness.ctx.tools.get("science_export")).toBeUndefined();
    expect(harness.ctx.tools.get("literature_search")).toBeUndefined();
    expect(harness.ctx.get("spillStore")).toBeDefined();
    expect(harness.scienceConfig).toMatchObject({
      embedArtifactMetadata: false,
      maxArtifactBytes: 1_048_576,
    });
    const entries = [...harness.ctx.loader.entries()] as Array<{
      options: { config?: Record<string, unknown>; id?: string };
    }>;
    expect(
      entries.find((entry) => entry.options.id === "swarmx-science")?.options.config,
    ).toMatchObject({
      embedArtifactMetadata: false,
      maxArtifactBytes: 1_048_576,
      root: join(home, "swarmx", "science"),
    });
    const response = await fetchHarness();
    const html = await response.text();
    expect(html).toContain("@swarmx/dsh-ui-science");
  });

  it("mounts the private PKB service and aggregate tool", () => {
    expect(harness.ctx.get("pkb")).toBeDefined();
    expect(harness.ctx.tools.get("pkb")).toBeDefined();
    const entries = harness.ctx.loader.entries() as Array<{
      options: { id?: string; name?: string };
    }>;
    const pkbEntry = entries.find((entry) => entry.options.id === "swarmx-pkb");
    const expectedEntry = moduleRequire.resolve("@swarmx/dsh-pkb");
    expect(pkbEntry?.options.name).toBe(realpathSync(expectedEntry));
    expect(
      (pkbEntry?.options as { config?: Record<string, unknown> } | undefined)?.config,
    ).toMatchObject({
      maxConceptBytes: 131_072,
      root: join(home, "swarmx", "pkb", "vault"),
    });
    expect(statSync(join(home, "swarmx", "pkb", "vault")).mode & 0o777).toBe(0o700);
    expect(readFileSync(join(home, "swarmx", "pkb", "vault", "index.md"), "utf8")).toContain(
      'okf_version: "0.2"',
    );
  });

  it("mounts the durable Swarm service and read-only client projection", async () => {
    expect(harness.ctx.get("swarm")).toBeDefined();
    const typert = harness.ctx.get("typert") as
      | { local: { get(endpoint: string): { service: string } | undefined } }
      | undefined;
    expect(typert?.local.get("swarm/uiSnapshot")).toMatchObject({ service: "swarm" });
    expect(typert?.local.get("swarm/waitUi")).toMatchObject({ service: "swarm" });
    expect(harness.ctx.tools.get("swarm")).toBeUndefined();
    expect(statSync(join(home, "swarmx", "swarm")).mode & 0o777).toBe(0o700);
    expect(statSync(join(home, "swarmx", "swarm", "swarm.sqlite")).mode & 0o777).toBe(0o600);

    const entries = [...harness.ctx.loader.entries()] as Array<{
      options: { id?: string; name?: string };
    }>;
    const swarmEntry = entries.find((entry) => entry.options.id === "swarmx-swarm");
    const swarmUiEntry = entries.find((entry) => entry.options.id === "swarmx-ui-swarm");
    expect(swarmEntry?.options.name).toBe("@swarmx/dsh-swarm");
    expect(
      (swarmEntry?.options as { config?: Record<string, unknown> } | undefined)?.config,
    ).toMatchObject({
      monitorStallMs: 123_456,
      recoveryOwner: false,
      root: join(home, "swarmx", "swarm"),
    });
    expect(swarmUiEntry).toBeDefined();
    expect(entries.indexOf(swarmUiEntry as (typeof entries)[number])).toBeGreaterThan(
      entries.indexOf(swarmEntry as (typeof entries)[number]),
    );

    const response = await fetchHarness();
    expect(await response.text()).toContain("@swarmx/dsh-ui-swarm");
  });

  it("discovers Science as a system preset and scopes its model tools", async () => {
    const preset = (await harness.ctx.agentPresets.list()).find(({ id }) => id === "dsh-science");
    expect(preset).toMatchObject({
      id: "dsh-science",
      trust: "system",
      name: "科学模式",
      order: 5,
    });
    expect(preset?.broken).toBeUndefined();

    const standard = await harness.ctx.agentPresets.standingKeyFor("standard");
    const science = await harness.ctx.agentPresets.standingKeyFor("dsh-science");
    expect(harness.ctx.tools.get("science_record", standard)).toBeUndefined();
    expect(harness.ctx.tools.get("pkb", standard)).toBeDefined();
    expect(harness.ctx.tools.get("literature_search", standard)).toBeUndefined();
    expect(harness.ctx.tools.get("science_record", science)).toBeDefined();
    expect(harness.ctx.tools.get("science_export", science)).toBeDefined();
    expect(harness.ctx.tools.get("literature_search", science)).toBeDefined();
    expect(harness.ctx.tools.get("pkb", science)).toBeDefined();
  });

  it("discovers Team mode and scopes Swarm tools to that preset only", async () => {
    const preset = (await harness.ctx.agentPresets.list()).find(({ id }) => id === "dsh-swarm");
    expect(preset).toMatchObject({
      id: "dsh-swarm",
      trust: "system",
      name: "团队模式",
      order: 6,
    });
    expect(preset?.broken).toBeUndefined();

    const standard = await harness.ctx.agentPresets.standingKeyFor("standard");
    const science = await harness.ctx.agentPresets.standingKeyFor("dsh-science");
    const swarm = await harness.ctx.agentPresets.standingKeyFor("dsh-swarm");
    expect(harness.ctx.tools.get("swarm", standard)).toBeUndefined();
    expect(harness.ctx.tools.get("swarm", science)).toBeUndefined();
    expect(harness.ctx.tools.get("swarm", swarm)).toBeDefined();
    expect(harness.ctx.tools.get("science_record", swarm)).toBeDefined();
    expect(harness.ctx.tools.get("literature_search", swarm)).toBeDefined();
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
