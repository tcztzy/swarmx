import { execFileSync } from "node:child_process";
import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { NodeProcessRunner } from "../../../../apps/desktop/src/host/process-runner.js";
import DvcService, { type DvcError } from "../src/index.js";

const scratch: string[] = [];
const fixtureSource = fileURLToPath(new URL("./fixtures/fake-dvc.mjs", import.meta.url));
const DVC_TEST_TIMEOUT_MS = 30_000;

function git(cwd: string, ...args: string[]): string {
  return execFileSync("git", args, { cwd, encoding: "utf8" }).trim();
}

function fakeDvc(): string {
  const owner = mkdtempSync(join(tmpdir(), "swarmx-fake-dvc-"));
  scratch.push(owner);
  const command = join(owner, "dvc");
  copyFileSync(fixtureSource, command);
  chmodSync(command, 0o700);
  return command;
}

function repository(): string {
  const root = mkdtempSync(join(tmpdir(), "swarmx-dvc-test-"));
  scratch.push(root);
  git(root, "init", "-b", "main");
  git(root, "config", "user.email", "dvc-test@swarmx.invalid");
  git(root, "config", "user.name", "SwarmX DVC Test");
  mkdirSync(join(root, ".dvc", "cache"), { recursive: true });
  writeFileSync(join(root, ".dvc", ".gitignore"), "/cache\n/config.local\n");
  writeFileSync(join(root, ".dvc", "config"), "[core]\n    no_scm = false\n");
  writeFileSync(join(root, ".dvc", "config.local"), '[remote "private"]\n    url = s3://secret\n');
  writeFileSync(join(root, "dvc.yaml"), "stages:\n  build:\n    cmd: fake\n");
  writeFileSync(join(root, "dvc.lock"), "schema: '2.0'\nstages: {}\n");
  writeFileSync(
    join(root, ".fake-dvc-data.json"),
    JSON.stringify({
      committed: { modified: ["secret-data.csv"] },
      uncommitted: { added: ["output.csv"] },
    }),
  );
  writeFileSync(
    join(root, ".fake-dvc-pipeline.json"),
    JSON.stringify({ build: [{ "changed deps": ["secret-data.csv"], "changed outs": [] }] }),
  );
  git(root, "add", ".");
  git(root, "commit", "-m", "DVC project");
  return root;
}

async function mounted(config: Record<string, unknown> = {}) {
  const dvc = new DvcService(new NodeProcessRunner(), config);
  return { context: { dvc }, fiber: { dispose: () => dvc.close() } };
}

afterEach(() => {
  for (const path of scratch.splice(0)) rmSync(path, { recursive: true, force: true });
});

describe("DVC service", () => {
  it(
    "returns bounded, path-free local status and manifest digests",
    async () => {
      const root = repository();
      const { context, fiber } = await mounted({ command: fakeDvc() });
      try {
        const inspection = await context.dvc.inspect(root);
        expect(inspection).toMatchObject({
          root: ".",
          version: "3.99.0-test",
          dvcLockDigest: expect.stringMatching(/^sha256:[0-9a-f]{64}$/u),
          dvcYamlDigest: expect.stringMatching(/^sha256:[0-9a-f]{64}$/u),
          data: { digest: expect.stringMatching(/^sha256:[0-9a-f]{64}$/u) },
          pipeline: { digest: expect.stringMatching(/^sha256:[0-9a-f]{64}$/u) },
        });
        expect(inspection.data.entries).toBe(2);
        const serialized = JSON.stringify(inspection);
        expect(serialized).not.toContain(root);
        expect(serialized).not.toContain("secret-data.csv");
        expect(serialized).not.toContain("output.csv");
      } finally {
        await fiber.dispose();
      }
    },
    DVC_TEST_TIMEOUT_MS,
  );

  it(
    "pulls only on an explicit validated request",
    async () => {
      const root = repository();
      const { context, fiber } = await mounted({ command: fakeDvc() });
      try {
        expect(existsSync(join(root, ".fake-dvc-pulls"))).toBe(false);
        const result = await context.dvc.pull(root, {
          targets: ["data/raw.dvc"],
          remote: "origin",
        });
        expect(result.status).toBe("succeeded");
        expect(readFileSync(join(root, ".fake-dvc-pulls"), "utf8")).toContain("data/raw.dvc");
        await expect(context.dvc.pull(root, { targets: ["../outside"] })).rejects.toMatchObject<
          Partial<DvcError>
        >({ code: "DVC_REQUEST_INVALID" });
        await expect(context.dvc.pull(root, { targets: ["--all"] })).rejects.toMatchObject<
          Partial<DvcError>
        >({ code: "DVC_REQUEST_INVALID" });
      } finally {
        await fiber.dispose();
      }
    },
    DVC_TEST_TIMEOUT_MS,
  );

  it(
    "reproduces exact clean HEAD in a retained disposable worktree",
    async () => {
      const root = repository();
      const sourceHead = git(root, "rev-parse", "HEAD");
      const { context, fiber } = await mounted({ command: fakeDvc() });
      try {
        const reproduction = await context.dvc.reproduce(root, { targets: ["build"] });
        expect(reproduction.result.status).toBe("succeeded");
        expect(reproduction.source.git.head).toBe(sourceHead);
        expect(readFileSync(join(reproduction.path, "generated.txt"), "utf8")).toContain("build");
        expect(existsSync(join(root, "generated.txt"))).toBe(false);
        expect(readFileSync(join(root, ".dvc", "config.local"), "utf8")).not.toContain("cache =");
        expect(git(root, "rev-parse", "HEAD")).toBe(sourceHead);
        const disposablePath = reproduction.path;
        await reproduction.dispose();
        expect(existsSync(disposablePath)).toBe(false);
      } finally {
        await fiber.dispose();
      }
    },
    DVC_TEST_TIMEOUT_MS,
  );

  it(
    "rejects dirty sources and returns redacted failed stage outcomes",
    async () => {
      const root = repository();
      const { context, fiber } = await mounted({ command: fakeDvc() });
      try {
        writeFileSync(join(root, "dvc.yaml"), "dirty\n");
        await expect(context.dvc.reproduce(root, {})).rejects.toMatchObject<Partial<DvcError>>({
          code: "DVC_WORKSPACE_DIRTY",
        });
        git(root, "checkout", "--", "dvc.yaml");
        const failed = await context.dvc.reproduce(root, { targets: ["fail"] });
        expect(failed.result).toMatchObject({ status: "failed", exitCode: 7 });
        expect(failed.result.stderr.text).not.toContain(root);
        expect(failed.result.stderr.text).not.toContain("secret");
        await failed.dispose();
      } finally {
        await fiber.dispose();
      }
    },
    DVC_TEST_TIMEOUT_MS,
  );

  it(
    "cleans retained reproductions on disposal and keeps missing DVC lazy",
    async () => {
      const root = repository();
      const active = await mounted({ command: fakeDvc() });
      const reproduction = await active.context.dvc.reproduce(root, {});
      const disposablePath = reproduction.path;
      await active.fiber.dispose();
      expect(existsSync(disposablePath)).toBe(false);

      const missing = await mounted({ command: "swarmx-dvc-does-not-exist" });
      await expect(missing.context.dvc.inspect(root)).rejects.toMatchObject<Partial<DvcError>>({
        code: "DVC_UNAVAILABLE",
      });
      await missing.fiber.dispose();

      const controller = new AbortController();
      controller.abort(new Error("stop"));
      const aborted = await mounted({ command: fakeDvc() });
      await expect(aborted.context.dvc.inspect(root, controller.signal)).rejects.toThrow("stop");
      await aborted.fiber.dispose();
    },
    DVC_TEST_TIMEOUT_MS,
  );
});
