import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = fileURLToPath(new URL("../../..", import.meta.url));
const launcherPath = fileURLToPath(new URL("../bin/swarmx.js", import.meta.url));

function runLauncher(args: string[] = []): {
  mode: "cli" | "desktop";
  args: string[];
  electronPath?: string;
  appPath?: string;
} {
  const result = spawnSync(process.execPath, [launcherPath, ...args], {
    cwd: repoRoot,
    encoding: "utf8",
    env: { ...process.env, SWARMX_LAUNCHER_DRY_RUN: "1" },
  });

  expect(result.status, result.stderr).toBe(0);
  return JSON.parse(result.stdout) as {
    mode: "cli" | "desktop";
    args: string[];
    electronPath?: string;
    appPath?: string;
  };
}

describe("npm launcher cold start", () => {
  it("V467 resolves Desktop and Electron by default and through the explicit alias", () => {
    const defaultLaunch = runLauncher();
    const explicitLaunch = runLauncher(["desktop", "--inspect=0"]);

    expect(defaultLaunch).toMatchObject({ mode: "desktop", args: [] });
    expect(explicitLaunch).toMatchObject({ mode: "desktop", args: ["--inspect=0"] });
    expect(defaultLaunch.appPath).toBe(explicitLaunch.appPath);
    expect(defaultLaunch.electronPath).toBe(explicitLaunch.electronPath);
    expect(existsSync(defaultLaunch.appPath ?? "")).toBe(true);
    expect(existsSync(defaultLaunch.electronPath ?? "")).toBe(true);
  });

  it("V467 keeps existing CLI arguments and strips only the explicit cli alias", () => {
    expect(runLauncher(["doctor", "--json"])).toEqual({
      mode: "cli",
      args: ["doctor", "--json"],
    });
    expect(runLauncher(["cli", "sessions"])).toEqual({ mode: "cli", args: ["sessions"] });
  });

  it("V468 and V473 assign Desktop and Electron to launcher runtime dependencies", () => {
    const swarmxManifest = JSON.parse(
      readFileSync(new URL("../package.json", import.meta.url), "utf8"),
    ) as { dependencies: Record<string, string> };
    const desktopManifest = JSON.parse(
      readFileSync(new URL("../../desktop/package.json", import.meta.url), "utf8"),
    ) as { dependencies: Record<string, string>; devDependencies: Record<string, string> };

    expect(swarmxManifest.dependencies).toMatchObject({
      "@swarmx/cli": "workspace:*",
      "@swarmx/desktop": "workspace:*",
      electron: expect.stringMatching(/^\^33\./),
    });
    expect(desktopManifest.dependencies.electron).toBeUndefined();
    expect(desktopManifest.devDependencies.electron).toMatch(/^\^33\./);
  });

  it("V469 gates dual-architecture DMG and ZIP releases on the tag version", () => {
    const workflow = readFileSync(
      new URL("../../../.github/workflows/release.yml", import.meta.url),
      "utf8",
    );
    const builder = readFileSync(
      new URL("../../desktop/electron-builder.yml", import.meta.url),
      "utf8",
    );

    expect(workflow).toContain("macos-15");
    expect(workflow).toContain("macos-15-intel");
    expect(workflow).toContain("RELEASE_TAG:");
    expect(workflow).toContain("gh release create");
    expect(workflow).toContain("build-macos-artifacts.mjs");
    expect(builder).toContain("- dir");
  });

  it("V470 keeps README concise and leads npm users to Desktop", () => {
    const readme = readFileSync(new URL("../../../README.md", import.meta.url), "utf8");

    expect(readme.split("\n").length).toBeLessThanOrEqual(150);
    expect(readme).toContain("npm install swarmx");
    expect(readme).toContain("npx swarmx");
    expect(readme).toContain("/releases/latest");
    expect(readme).toContain("docs/assets/swarmx-demo.gif");
  });

  it("V474 excludes generated desktop release artifacts from Git and Biome", () => {
    const gitignore = readFileSync(new URL("../../../.gitignore", import.meta.url), "utf8");
    const biome = JSON.parse(
      readFileSync(new URL("../../../biome.json", import.meta.url), "utf8"),
    ) as { files: { ignore: string[] } };

    expect(gitignore).toContain("packages/desktop/release/");
    expect(biome.files.ignore).toContain("packages/desktop/release/**");
  });

  it("V475 reserves the swarmx bin name for the Desktop-first launcher", () => {
    const swarmxManifest = JSON.parse(
      readFileSync(new URL("../package.json", import.meta.url), "utf8"),
    ) as { bin: Record<string, string> };
    const cliManifest = JSON.parse(
      readFileSync(new URL("../../cli/package.json", import.meta.url), "utf8"),
    ) as { bin: Record<string, string> };

    expect(swarmxManifest.bin).toEqual({ swarmx: "./bin/swarmx.js" });
    expect(cliManifest.bin).toEqual({ "swarmx-cli": "./dist/cli.js" });
  });

  it("V476 creates macOS archives with host tools after app packaging", () => {
    const scriptUrl = new URL("../../desktop/scripts/build-macos-artifacts.mjs", import.meta.url);
    expect(existsSync(scriptUrl)).toBe(true);

    const script = readFileSync(scriptUrl, "utf8");
    expect(script).toContain('"ditto"');
    expect(script).toContain('"hdiutil"');
    expect(script).toContain('"--dir"');
  });

  it("V477 and V479 await Electron's checked downloader when npm skips lifecycle setup", () => {
    const launcher = readFileSync(new URL("../bin/swarmx.js", import.meta.url), "utf8");

    expect(launcher).toContain('"@electron/get"');
    expect(launcher).toContain('"extract-zip"');
    expect(launcher).toContain("await downloadArtifact");
    expect(launcher).toContain("https://github.com/tcztzy/swarmx/releases/latest");
  });

  it("V478 repairs only a verified Electron path marker and reloads the module", () => {
    const launcher = readFileSync(new URL("../bin/swarmx.js", import.meta.url), "utf8");

    expect(launcher).toContain('"path.txt"');
    expect(launcher).toContain("writeFileSync");
    expect(launcher).toContain("require.cache");
    expect(launcher).toContain('"dist", "version"');
  });

  it("V480 keeps Node alive only while Electron bootstrap promises are pending", () => {
    const launcher = readFileSync(new URL("../bin/swarmx.js", import.meta.url), "utf8");

    expect(launcher).toContain("setInterval");
    expect(launcher).toContain("clearInterval");
    expect(launcher).toContain("finally");
  });

  it("V481 uses host ditto for reliable macOS Electron extraction", () => {
    const launcher = readFileSync(new URL("../bin/swarmx.js", import.meta.url), "utf8");

    expect(launcher).toContain('execFileSync("ditto"');
    expect(launcher).toContain('["-x", "-k"');
  });

  it("V487 and V488 explicitly review dependency build scripts across supported pnpm", () => {
    const workspace = readFileSync(
      new URL("../../../pnpm-workspace.yaml", import.meta.url),
      "utf8",
    );

    expect(workspace).toContain("electron-winstaller: false");
    expect(workspace).toContain("onlyBuiltDependencies:");
    for (const dependency of ['"@biomejs/biome"', "electron", "esbuild", "node-pty"]) {
      expect(workspace).toContain(`  - ${dependency}`);
    }
  });

  it("V489 exports optional macOS signing inputs only when configured", () => {
    const workflow = readFileSync(
      new URL("../../../.github/workflows/release.yml", import.meta.url),
      "utf8",
    );

    expect(workflow).toContain("workflow_dispatch:");
    expect(workflow).toContain("github.event_name == 'workflow_dispatch'");
    expect(workflow).toContain("ref: ${{ env.RELEASE_TAG }}");
    expect(workflow).toContain("SWARMX_MACOS_CERTIFICATE:");
    expect(workflow).toContain('export CSC_LINK="$SWARMX_MACOS_CERTIFICATE"');
    expect(workflow).not.toContain("CSC_LINK: ${{ secrets.");
  });

  it("V490 resolves electron-builder app directories for both macOS architectures", () => {
    const script = readFileSync(
      new URL("../../desktop/scripts/build-macos-artifacts.mjs", import.meta.url),
      "utf8",
    );

    expect(script).toContain('arm64: "mac-arm64"');
    expect(script).toContain('x64: "mac"');
  });

  it("V491 refreshes only release automation when rebuilding an existing tag", () => {
    const workflow = readFileSync(
      new URL("../../../.github/workflows/release.yml", import.meta.url),
      "utf8",
    );

    expect(workflow).toContain("Restore current release automation for manual retry");
    expect(workflow).toContain("RELEASE_AUTOMATION_SHA: ${{ github.sha }}");
    expect(workflow).toContain("/contents/packages/desktop/scripts/build-macos-artifacts.mjs");
    expect(workflow).toContain(
      'cp "$RUNNER_TEMP/build-macos-artifacts.mjs" packages/desktop/scripts/build-macos-artifacts.mjs',
    );
  });

  it("V492 publishes from an explicit repository without requiring a checkout", () => {
    const workflow = readFileSync(
      new URL("../../../.github/workflows/release.yml", import.meta.url),
      "utf8",
    );

    expect(workflow).toContain("GH_REPO: ${{ github.repository }}");
    expect(workflow).toContain('gh release create "$RELEASE_TAG"');
  });

  it("V493 verifies both macOS archive formats before upload", () => {
    const workflow = readFileSync(
      new URL("../../../.github/workflows/release.yml", import.meta.url),
      "utf8",
    );

    expect(workflow).toContain('hdiutil verify "${artifact_base}.dmg"');
    expect(workflow).toContain('unzip -tq "${artifact_base}.zip"');
  });
});
