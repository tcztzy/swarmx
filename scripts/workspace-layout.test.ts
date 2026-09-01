import { existsSync, readdirSync, readFileSync, realpathSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

interface PackageManifest {
  readonly packageManager?: string;
  readonly scripts?: Readonly<Record<string, string>>;
  readonly workspaces?: readonly string[];
  readonly dependencies?: Readonly<Record<string, string>>;
  readonly devDependencies?: Readonly<Record<string, string>>;
  readonly peerDependencies?: Readonly<Record<string, string>>;
}

function manifest(path: string): PackageManifest {
  return JSON.parse(readFileSync(path, "utf8")) as PackageManifest;
}

type ComputeColumns = (
  viewport: number,
  sidebar: number,
  details: number,
) => { sidebar: number; center: number; details: number };

function layoutBundle(): string {
  const requireFromClient = createRequire(resolve("packages/client/ui-conversation/package.json"));
  const packagePath = requireFromClient.resolve("@deepseek-ai/dsh-client-ui-layout/package.json");
  return readFileSync(join(dirname(packagePath), "lib/client.js"), "utf8");
}

function agentPresetBundle(): { readonly runtime: string; readonly types: string } {
  const requireFromDesktop = createRequire(resolve("apps/desktop/package.json"));
  const dshManifest = realpathSync(requireFromDesktop.resolve("@deepseek-ai/dsh/package.json"));
  const requireFromDsh = createRequire(dshManifest);
  const packagePath = requireFromDsh.resolve(
    "@deepseek-ai/dsh-client-ui-agent-preset/package.json",
  );
  const packageDirectory = dirname(packagePath);
  return {
    runtime: readFileSync(join(packageDirectory, "lib/client.js"), "utf8"),
    types: readFileSync(join(packageDirectory, "lib/types/client/locales.d.ts"), "utf8"),
  };
}

function sessionControllerBundle(): {
  readonly runtime: string;
  readonly types: readonly string[];
} {
  const requireFromClient = createRequire(resolve("packages/client/ui-conversation/package.json"));
  const packagePath = requireFromClient.resolve(
    "@deepseek-ai/dsh-api-session-controller/package.json",
  );
  const packageDirectory = dirname(packagePath);
  return {
    runtime: readFileSync(join(packageDirectory, "lib/client.js"), "utf8"),
    types: [
      "lib/types/client/contract/sessions.d.ts",
      "lib/types/client/sessions/manager.d.ts",
      "lib/types/client/sessions/service.d.ts",
    ].map((path) => readFileSync(join(packageDirectory, path), "utf8")),
  };
}

function columnSolver(bundle: string): ComputeColumns {
  const source = bundle.match(/function clampWidth[\s\S]*?\n\t\t}\n\t\t\/\/#endregion/)?.[0];
  if (source === undefined) throw new Error("ui-layout column solver not found");
  return Function(`${source}\nreturn computeColumns;`)() as ComputeColumns;
}

describe("workspace layout", () => {
  it("groups reusable packages and keeps the desktop as an app", () => {
    expect(existsSync("packages/client/ui-conversation/package.json")).toBe(true);
    expect(existsSync("packages/client/ui-dvc/package.json")).toBe(true);
    expect(existsSync("packages/client/ui-git/package.json")).toBe(true);
    expect(existsSync("packages/client/ui-science/package.json")).toBe(true);
    expect(existsSync("packages/client/ui-swarm/package.json")).toBe(true);
    expect(existsSync("packages/core/annotation/package.json")).toBe(true);
    expect(existsSync("packages/core/dvc/package.json")).toBe(true);
    expect(existsSync("packages/core/pkb/package.json")).toBe(true);
    expect(existsSync("packages/core/wikiskill/package.json")).toBe(true);
    expect(existsSync("packages/core/swarm/package.json")).toBe(true);
    expect(existsSync("packages/science/core/package.json")).toBe(true);
    expect(existsSync("apps/desktop/package.json")).toBe(true);
    expect(existsSync("packages/dsh-ui-conversation")).toBe(false);
    expect(existsSync("packages/desktop")).toBe(false);

    const root = manifest("package.json");
    expect(root.packageManager).toBe("pnpm@11.7.0");
    expect(root.workspaces).toEqual(["packages/*/*", "apps/*"]);
    expect(root.scripts?.dev).toBe("pnpm start");
    expect(Object.values(root.scripts ?? {}).join("\n")).not.toContain("npm run");

    const desktop = manifest("apps/desktop/package.json");
    expect(desktop.dependencies?.["@swarmx/dsh-dvc"]).toBe("workspace:*");
    expect(desktop.dependencies?.["@swarmx/dsh-ui-conversation"]).toBe("workspace:*");
    expect(desktop.dependencies?.["@swarmx/dsh-ui-dvc"]).toBe("workspace:*");
    expect(desktop.dependencies?.["@swarmx/dsh-ui-git"]).toBe("workspace:*");
    expect(desktop.dependencies?.["@swarmx/dsh-pkb"]).toBe("workspace:*");
    expect(desktop.dependencies?.["@swarmx/dsh-wikiskill"]).toBe("workspace:*");
    expect(desktop.dependencies?.["@swarmx/dsh-science"]).toBe("workspace:*");
    expect(desktop.dependencies?.["@swarmx/dsh-swarm"]).toBe("workspace:*");
    expect(desktop.dependencies?.["@swarmx/dsh-ui-science"]).toBe("workspace:*");
    expect(desktop.dependencies?.["@swarmx/dsh-ui-swarm"]).toBe("workspace:*");
    expect(desktop.dependencies?.["@deepseek-ai/dsh-web-frontend"]).toBe("0.1.2-alpha.2");
  });

  it("V141/V142 keeps the relocated PKB package rooted and package-bounded", () => {
    const pkb = manifest("packages/core/pkb/package.json");
    const patch = readFileSync("packages/core/pkb/cordis.patch.yml", "utf8");

    expect(pkb.scripts?.test).toContain("--root ../../..");
    expect(pkb.scripts?.test).toContain("packages/core/pkb/tests");
    expect(existsSync("packages/memory/core")).toBe(false);
    expect(existsSync("packages/core/pkg")).toBe(false);
    expect(patch).toContain("id: session-query-sqlite");
    expect(patch).toContain("openAt: first-search");
    expect(patch).toContain("path: ':memory:'");
    expect(patch).toContain("dshHomePath('pkb', 'vault')");
  });

  it("V226 includes host-only core packages in the clean client build graph", () => {
    const clientConfig = JSON.parse(readFileSync("tsconfig.client.json", "utf8")) as {
      readonly references?: readonly { readonly path?: string }[];
    };
    const cleanScript = readFileSync("scripts/clean.ts", "utf8");

    expect(clientConfig.references?.map((reference) => reference.path)).toContain(
      "./packages/core/pkb",
    );
    expect(clientConfig.references?.map((reference) => reference.path)).toContain(
      "./packages/core/wikiskill",
    );
    for (const output of [
      "packages/core/dvc/lib",
      "packages/core/pkb/lib",
      "packages/core/wikiskill/lib",
      "packages/client/ui-dvc/lib",
      "packages/client/ui-git/lib",
    ]) {
      expect(cleanScript).toContain(output);
    }
  });

  it("keeps shared build tools at the workspace root", () => {
    const root = manifest("package.json");
    const client = manifest("packages/client/ui-conversation/package.json");
    const desktop = manifest("apps/desktop/package.json");
    const buildTools = ["lightningcss", "tsdown", "tsx", "typescript", "vitest"];

    for (const tool of buildTools) {
      expect(root.devDependencies?.[tool], `${tool} root dependency`).toBeDefined();
      expect(client.devDependencies?.[tool], `${tool} client dependency`).toBeUndefined();
      expect(desktop.devDependencies?.[tool], `${tool} desktop dependency`).toBeUndefined();
    }
  });

  it("V107 owns semantic compilation through the writing preview runtime boundary", () => {
    const root = manifest("package.json");
    const runtimeManifest = readFileSync("native/writing-preview-runtime/Cargo.toml", "utf8");
    const buildScript = readFileSync("scripts/build-writing-preview-runtime.ts", "utf8");
    const ignore = readFileSync(".gitignore", "utf8");

    expect(root.scripts?.["build:lib"]).toContain("build:writing-preview-runtime");
    expect(root.scripts?.["build:typst-bridge"]).toBeUndefined();
    expect(runtimeManifest).toContain('name = "swarmx-writing-preview-runtime"');
    expect(existsSync("native/typst-bridge")).toBe(false);
    expect(existsSync("packages/science/core/src/writing-preview-runtime.ts")).toBe(true);
    expect(existsSync("packages/science/core/src/typst-bridge.ts")).toBe(false);
    expect(ignore).toContain("native/writing-preview-runtime/target/");
    expect(ignore).toContain("/packages/science/core/bin/");
    expect(buildScript).toContain(
      'execFileSync("codesign", ["--force", "--sign", "-", destination]',
    );
  });

  it("keeps every direct DSH dependency on one release baseline", () => {
    const files = [
      "apps/desktop/package.json",
      "package.json",
      "packages/client/ui-conversation/package.json",
      "packages/client/ui-dvc/package.json",
      "packages/client/ui-git/package.json",
      "packages/client/ui-science/package.json",
      "packages/client/ui-swarm/package.json",
      "packages/core/annotation/package.json",
      "packages/core/dvc/package.json",
      "packages/core/pkb/package.json",
      "packages/core/wikiskill/package.json",
      "packages/core/swarm/package.json",
      "packages/science/core/package.json",
    ];
    const baseline = manifest(files[0] as string).dependencies?.["@deepseek-ai/dsh"];
    expect(baseline).toBe("0.1.2-alpha.2");

    for (const file of files) {
      const packageManifest = manifest(file);
      expect(JSON.stringify(packageManifest), file).not.toContain("dsh-client-runtime");
      for (const dependencies of [packageManifest.dependencies, packageManifest.devDependencies]) {
        for (const [name, range] of Object.entries(dependencies ?? {})) {
          if (name.startsWith("@deepseek-ai/dsh")) {
            expect(range, `${name} in ${file}`).toBe(baseline);
          } else if (name === "@deepseek-ai/cordis") {
            expect(range, `${name} in ${file}`).toBe("4.0.2");
          } else if (name === "@deepseek-ai/schemastery") {
            expect(range, `${name} in ${file}`).toBe("^3.18.2");
          }
        }
      }
      for (const [name, range] of Object.entries(packageManifest.peerDependencies ?? {})) {
        if (name.startsWith("@deepseek-ai/dsh")) {
          expect(range, `${name} peer in ${file}`).toBe(`^${baseline}`);
        } else if (name === "@deepseek-ai/cordis") {
          expect(range, `${name} peer in ${file}`).toBe("^4.0.2");
        }
      }
    }

    const workspace = readFileSync("pnpm-workspace.yaml", "utf8");
    expect(workspace).toContain("'@deepseek-ai/cordis-plugin-group': 1.0.2");
    for (const name of [
      "@deepseek-ai/dsh-authorization",
      "@deepseek-ai/dsh-jobs",
      "@deepseek-ai/dsh-session-persistence",
      "@deepseek-ai/dsh-settings",
    ]) {
      expect(workspace).toContain(`'${name}': ${baseline}`);
    }
    expect(workspace).toContain(`'@deepseek-ai/dsh-client-ui-layout@${baseline}'`);
    expect(workspace).toContain(`patches/@deepseek-ai__dsh-client-ui-layout@${baseline}.patch`);
    expect(workspace).toContain(`'@deepseek-ai/dsh-client-ui-agent-preset@${baseline}'`);
    expect(workspace).toContain(
      `patches/@deepseek-ai__dsh-client-ui-agent-preset@${baseline}.patch`,
    );
    expect(workspace).toContain(`'@deepseek-ai/dsh-client-ui-conversation@${baseline}'`);
    expect(workspace).toContain(
      `patches/@deepseek-ai__dsh-client-ui-conversation@${baseline}.patch`,
    );
    expect(workspace).toContain(`'@deepseek-ai/dsh-api-session-controller@${baseline}'`);
    expect(workspace).toContain(
      `patches/@deepseek-ai__dsh-api-session-controller@${baseline}.patch`,
    );
    expect(workspace).toContain(`'@deepseek-ai/dsh-app-boot@${baseline}'`);
    expect(workspace).toContain(`patches/@deepseek-ai__dsh-app-boot@${baseline}.patch`);
    expect(workspace).not.toContain("dsh-client-ui-primitives");

    const patchFiles = readdirSync("patches")
      .filter((name) => name.endsWith(".patch"))
      .sort();
    expect(patchFiles).toEqual([
      `@deepseek-ai__dsh-api-session-controller@${baseline}.patch`,
      `@deepseek-ai__dsh-app-boot@${baseline}.patch`,
      `@deepseek-ai__dsh-client-ui-agent-preset@${baseline}.patch`,
      `@deepseek-ai__dsh-client-ui-conversation@${baseline}.patch`,
      `@deepseek-ai__dsh-client-ui-layout@${baseline}.patch`,
    ]);

    const lockfile = readFileSync("pnpm-lock.yaml", "utf8");
    expect(
      new Set(
        [...lockfile.matchAll(/^ {2}'@deepseek-ai\/dsh(?:-[^@']+)?@([^(']+)(?:\(|':)/gmu)].map(
          (match) => match[1],
        ),
      ),
    ).toEqual(new Set([baseline]));
    expect(
      new Set(
        [...lockfile.matchAll(/^ {2}'@deepseek-ai\/cordis@([^(']+)(?:\(|':)/gmu)].map(
          (match) => match[1],
        ),
      ),
    ).toEqual(new Set(["4.0.2"]));
    expect(
      new Set(
        [...lockfile.matchAll(/^ {2}'@deepseek-ai\/cordis-plugin-group@([^(']+)(?:\(|':)/gmu)].map(
          (match) => match[1],
        ),
      ),
    ).toEqual(new Set(["1.0.2"]));
    expect(
      new Set(
        [...lockfile.matchAll(/^ {2}'@deepseek-ai\/schemastery@([^(']+)(?:\(|':)/gmu)].map(
          (match) => match[1],
        ),
      ),
    ).toEqual(new Set(["3.18.2"]));
    expect(lockfile).not.toContain("dsh-client-runtime");
  });

  it("V124 localizes the dsh-science system preset with the active Web locale", () => {
    const preset = agentPresetBundle();

    expect(preset.runtime).toContain('presetScienceName: "Science mode"');
    expect(preset.runtime).toContain(
      'presetScienceDescription: "All Standard mode capabilities, with local-first scientific research tools, literature search, annotation understanding, and a managed Typst workflow."',
    );
    expect(preset.runtime).toContain('presetScienceName: "科学模式"');
    expect(preset.runtime).toContain(
      'presetScienceDescription: "具备标准模式的全部能力，并提供本地优先的科学研究工具、文献检索、注释理解与托管 Typst 工作流。"',
    );
    expect(preset.runtime).toContain('"dsh-science": {');
    expect(preset.runtime).toContain('name: "presetScienceName"');
    expect(preset.runtime).toContain('description: "presetScienceDescription"');
    expect(preset.types).toContain("'presetScienceName'");
    expect(preset.types).toContain("'presetScienceDescription'");
  });

  it("V8 forwards the source agent preset through addressable session creation", () => {
    const sessionController = sessionControllerBundle();

    expect(sessionController.runtime).toContain(
      "...opts.agentPreset === void 0 ? {} : { agentPreset: opts.agentPreset }",
    );
    for (const types of sessionController.types) {
      expect(types).toContain("agentPreset?: string;");
    }
  });

  it("keeps details width viewport-bound instead of fixed at 520px", () => {
    const bundle = layoutBundle();
    const computeColumns = columnSolver(bundle);

    expect(bundle).not.toContain("clampWidth(details, 300, 520)");
    expect(bundle).not.toContain("d.details = clampWidth(px, 300, 520)");
    expect(computeColumns(2200, 280, 900)).toEqual({
      sidebar: 280,
      center: 1020,
      details: 900,
    });
    expect(computeColumns(1800, 280, 900)).toEqual({
      sidebar: 280,
      center: 640,
      details: 880,
    });
    expect(bundle).toContain("openDetails(preferredWidth)");
    expect(bundle).toContain("this.#require().openDetails(preferredWidth)");
    expect(bundle).toContain(
      "if (d.details === 0) d.details = Math.max(300, Math.round(preferredWidth ?? 360))",
    );
  });
});
