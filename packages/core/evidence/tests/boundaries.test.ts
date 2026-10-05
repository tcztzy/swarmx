import { readFileSync } from "node:fs";
import { expect, it } from "vitest";

it("owns portable contracts without depending on Science or the desktop", () => {
  const manifest = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
  expect(Object.keys(manifest.dependencies)).toEqual(["zod"]);
  const source = readFileSync(new URL("../src/index.ts", import.meta.url), "utf8");
  expect([...source.matchAll(/from "([^"]+)"/gu)].map((match) => match[1])).toEqual(["zod"]);
});

it("prepares a Git source package without requiring its parent workspace", () => {
  const manifest = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
  expect(manifest.scripts.prepare).toBe("npm run build");
  expect(manifest.scripts.build).toBe("tsc --project tsconfig.build.json");
  expect(Object.keys(manifest.devDependencies)).toEqual(["typescript"]);
  const config = JSON.parse(
    readFileSync(new URL("../tsconfig.build.json", import.meta.url), "utf8"),
  );
  expect(config.extends).toBeUndefined();
  expect(config.include).toEqual(["src"]);
  expect(config.compilerOptions.outDir).toBe("lib");
  expect(config.compilerOptions.declarationDir).toBe("lib/types");
  expect(manifest.files).toContain("tsconfig.build.json");
  expect(readFileSync(new URL("../pnpm-workspace.yaml", import.meta.url), "utf8")).toBe(
    "packages: []\n",
  );
  const rootLock = readFileSync(new URL("../../../../pnpm-lock.yaml", import.meta.url), "utf8");
  const localLock = readFileSync(new URL("../pnpm-lock.yaml", import.meta.url), "utf8");
  const localImporter = localLock.split("\n  .:\n")[1]?.split("\npackages:\n")[0]?.trim();
  const rootImporter = rootLock
    .split("\n  packages/core/evidence:\n")[1]
    ?.split(/\n {2}[^\s][^\n]*:\n/u)[0]
    ?.trim();
  expect(localImporter).toBeDefined();
  expect(localImporter).toBe(rootImporter);
  for (const match of localLock.matchAll(/\n {2}[^\n:]+:\n {4}resolution: [^\n]+\n/gu)) {
    expect(rootLock).toContain(match[0]);
  }
});
