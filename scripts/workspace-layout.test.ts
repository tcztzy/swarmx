import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

interface PackageManifest {
  readonly packageManager?: string;
  readonly workspaces?: readonly string[];
  readonly dependencies?: Readonly<Record<string, string>>;
  readonly devDependencies?: Readonly<Record<string, string>>;
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

function columnSolver(bundle: string): ComputeColumns {
  const source = bundle.match(/function clampWidth[\s\S]*?\n\t\t}\n\t\t\/\/#endregion/)?.[0];
  if (source === undefined) throw new Error("ui-layout column solver not found");
  return Function(`${source}\nreturn computeColumns;`)() as ComputeColumns;
}

describe("workspace layout", () => {
  it("groups reusable packages and keeps the desktop as an app", () => {
    expect(existsSync("packages/client/ui-conversation/package.json")).toBe(true);
    expect(existsSync("packages/client/ui-science/package.json")).toBe(true);
    expect(existsSync("packages/science/core/package.json")).toBe(true);
    expect(existsSync("apps/desktop/package.json")).toBe(true);
    expect(existsSync("packages/dsh-ui-conversation")).toBe(false);
    expect(existsSync("packages/desktop/package.json")).toBe(false);

    const root = manifest("package.json");
    expect(root.packageManager).toBe("pnpm@11.7.0");
    expect(root.workspaces).toEqual(["packages/*/*", "apps/*"]);

    const desktop = manifest("apps/desktop/package.json");
    expect(desktop.dependencies?.["@swarmx/dsh-ui-conversation"]).toBe("workspace:*");
    expect(desktop.dependencies?.["@swarmx/dsh-science"]).toBe("workspace:*");
    expect(desktop.dependencies?.["@swarmx/dsh-ui-science"]).toBe("workspace:*");
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
  });
});
