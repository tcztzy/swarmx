import { execFile } from "node:child_process";
import { cp, mkdir, mkdtemp, readFile, rm, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";
import { expect, it } from "vitest";

it("builds and imports source-only evidence outside the monorepo", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-evidence-source-"));
  try {
    const source = fileURLToPath(new URL("../", import.meta.url));
    for (const name of [
      "src",
      "package.json",
      "tsconfig.build.json",
      "pnpm-workspace.yaml",
      "pnpm-lock.yaml",
    ]) {
      await cp(join(source, name), join(root, name), { recursive: true });
    }
    const dependencies = join(root, "node_modules");
    await mkdir(join(dependencies, ".bin"), { recursive: true });
    for (const name of ["typescript", "zod"]) {
      const packageRoot = dirname(fileURLToPath(import.meta.resolve(`${name}/package.json`)));
      await symlink(packageRoot, join(dependencies, name), "dir");
    }
    await symlink(join(dependencies, "typescript/bin/tsc"), join(dependencies, ".bin/tsc"));
    await expect(readFile(join(root, "lib/index.js"))).rejects.toMatchObject({ code: "ENOENT" });
    await promisify(execFile)("npm", ["run", "prepare"], {
      cwd: root,
      timeout: 30_000,
      maxBuffer: 1024 * 1024,
    });
    const entry = await import(pathToFileURL(join(root, "lib/index.js")).href);
    expect(entry.RO_CRATE_FORMAT).toBe("ro-crate@1.3");
    expect(entry.roCrateEntitySchema.parse({ "@id": "#synthetic", "@type": "Thing" })).toEqual({
      "@id": "#synthetic",
      "@type": "Thing",
    });
    expect(await readFile(join(root, "lib/types/index.d.ts"), "utf8")).toContain(
      "roCrateMetadataDocumentSchema",
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}, 40_000);
