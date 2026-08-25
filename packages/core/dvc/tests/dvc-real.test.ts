import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Context } from "@deepseek-ai/cordis";
import LocalSubprocessRuntime from "@deepseek-ai/dsh-subprocess-local";
import { afterEach, describe, expect, it } from "vitest";
import DvcService from "../src/index.js";

const DVC_ENV = { ...process.env, DVC_NO_ANALYTICS: "1", LC_ALL: "C" };
const dvcAvailable = spawnSync("dvc", ["--version"], { env: DVC_ENV }).status === 0;
const requireRealDvc = process.env.SWARMX_REQUIRE_REAL_DVC === "1";
const describeRealDvc = dvcAvailable || requireRealDvc ? describe : describe.skip;
const scratch: string[] = [];

function run(cwd: string, command: string, ...args: string[]): string {
  return execFileSync(command, args, { cwd, encoding: "utf8", env: DVC_ENV }).trim();
}

function realRepository(): string {
  const root = mkdtempSync(join(tmpdir(), "swarmx-dvc-real-"));
  scratch.push(root);
  run(root, "git", "init", "-b", "main");
  run(root, "git", "config", "user.email", "dvc-real-test@swarmx.invalid");
  run(root, "git", "config", "user.name", "SwarmX Real DVC Test");
  run(root, "dvc", "init");
  writeFileSync(join(root, "input.txt"), "scientific result\n");
  writeFileSync(
    join(root, "build.mjs"),
    'import { readFileSync, writeFileSync } from "node:fs";\nwriteFileSync("output.txt", readFileSync("input.txt", "utf8").toUpperCase());\n',
  );
  writeFileSync(
    join(root, "dvc.yaml"),
    "stages:\n  build:\n    cmd: node build.mjs\n    deps:\n      - build.mjs\n      - input.txt\n    outs:\n      - output.txt\n",
  );
  run(root, "dvc", "repro");
  run(root, "git", "add", "-A");
  run(root, "git", "commit", "-m", "real DVC pipeline");
  unlinkSync(join(root, "output.txt"));
  return root;
}

afterEach(() => {
  for (const path of scratch.splice(0)) rmSync(path, { recursive: true, force: true });
});

describeRealDvc("dsh-dvc real CLI integration", () => {
  it("V157 resolves real relative roots and reproduces without mutating the source", async () => {
    expect(dvcAvailable, "DVC CLI is required by SWARMX_REQUIRE_REAL_DVC=1").toBe(true);
    const root = realRepository();
    const sourceHead = run(root, "git", "rev-parse", "HEAD");
    const context = new Context();
    await context.plugin(LocalSubprocessRuntime);
    const fiber = await context.plugin(DvcService);
    try {
      const inspection = await context.dvc.inspect(root);
      expect(inspection).toMatchObject({ root: ".", version: expect.stringMatching(/^3\./u) });
      expect(inspection.dvcYamlDigest).toMatch(/^sha256:[0-9a-f]{64}$/u);
      expect(inspection.dvcLockDigest).toMatch(/^sha256:[0-9a-f]{64}$/u);

      const reproduction = await context.dvc.reproduce(root, { targets: ["build"] });
      const disposablePath = reproduction.path;
      expect(reproduction.result.status).toBe("succeeded");
      expect(reproduction.source.git.head).toBe(sourceHead);
      expect(readFileSync(join(disposablePath, "output.txt"), "utf8")).toBe("SCIENTIFIC RESULT\n");
      expect(existsSync(join(root, "output.txt"))).toBe(false);
      expect(run(root, "git", "status", "--porcelain")).toBe("");
      expect(run(root, "git", "rev-parse", "HEAD")).toBe(sourceHead);
      await reproduction.dispose();
      expect(existsSync(disposablePath)).toBe(false);
    } finally {
      await fiber.dispose();
    }
  }, 30_000);
});
