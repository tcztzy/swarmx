import { existsSync, readFileSync } from "node:fs";
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

describe("workspace layout", () => {
  it("groups reusable packages and keeps the desktop as an app", () => {
    expect(existsSync("packages/client/ui-conversation/package.json")).toBe(true);
    expect(existsSync("apps/desktop/package.json")).toBe(true);
    expect(existsSync("packages/dsh-ui-conversation")).toBe(false);
    expect(existsSync("packages/desktop")).toBe(false);

    const root = manifest("package.json");
    expect(root.packageManager).toBe("pnpm@11.7.0");
    expect(root.workspaces).toEqual(["packages/*/*", "apps/*"]);

    const desktop = manifest("apps/desktop/package.json");
    expect(desktop.dependencies?.["@swarmx/dsh-ui-conversation"]).toBe("workspace:*");
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
});
