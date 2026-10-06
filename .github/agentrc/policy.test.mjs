import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import policy from "./policy.mjs";

const directory = path.dirname(fileURLToPath(import.meta.url));
const cli = path.join(directory, "node_modules/@microsoft/agentrc/dist/index.js");

test("build detection accepts existing build/bundle commands and rejects empty values", async () => {
  const criterion = policy.criteria.add[0];
  for (const scripts of [{ build: "tsc" }, { bundle: "tsdown" }]) {
    assert.equal((await criterion.check({}, { scripts })).status, "pass");
  }
  for (const scripts of [
    undefined,
    {},
    { build: "" },
    { bundle: " " },
    { build: true },
    { test: "vitest" },
  ]) {
    assert.equal((await criterion.check({}, { scripts })).status, "fail");
  }
});

test("pinned CLI finds pnpm packages and preserves missing-test failures", async () => {
  const repo = await mkdtemp(path.join(os.tmpdir(), "swarmx-agentrc-"));
  const scan = (tail = []) => {
    const result = JSON.parse(
      execFileSync(process.execPath, [cli, "readiness", repo, "--json", ...tail], {
        cwd: directory,
        env: { PATH: process.env.PATH, HOME: repo },
        encoding: "utf8",
        timeout: 10000,
      }),
    );
    assert.equal(result.ok, true);
    assert.equal(result.status, "success");
    return result.data;
  };
  try {
    await mkdir(path.join(repo, "packages/core/example"), { recursive: true });
    await writeFile(
      path.join(repo, "package.json"),
      JSON.stringify({ private: true, packageManager: "pnpm@11.7.0" }),
    );
    await writeFile(path.join(repo, "pnpm-workspace.yaml"), "packages:\n  - packages/*/*\n");
    await writeFile(path.join(repo, "agentrc.config.json"), "{}\n");
    const manifest = path.join(repo, "packages/core/example/package.json");
    await writeFile(
      manifest,
      JSON.stringify({ name: "example", scripts: { bundle: "tsdown", test: "vitest run" } }),
    );
    const flags = ["--policy", path.join(directory, "policy.mjs")];
    const report = scan(flags);
    assert.equal(report.apps.length, 1);
    const status = (data, id) => data.criteria.find((criterion) => criterion.id === id).status;
    assert.equal(status(scan(), "build-script"), "fail");
    assert.equal(status(report, "build-script"), "pass");
    assert.equal(status(report, "test-script"), "pass");
    await writeFile(manifest, JSON.stringify({ name: "example", scripts: { bundle: "tsdown" } }));
    assert.equal(status(scan(flags), "test-script"), "fail");
    await writeFile(manifest, JSON.stringify({ name: "example", scripts: { bundle: "" } }));
    assert.equal(status(scan(flags), "build-script"), "fail");

    // Preserve AgentRC's default app aggregation and expose the missing package.
    await writeFile(manifest, JSON.stringify({ name: "example", scripts: {} }));
    for (const name of ["build-a", "build-b", "bundle-a", "bundle-b"]) {
      const directory = path.join(repo, "packages/core", name);
      await mkdir(directory);
      const scripts = name.startsWith("build") ? { build: "tsc" } : { bundle: "tsdown" };
      await writeFile(path.join(directory, "package.json"), JSON.stringify({ name, scripts }));
    }
    const partial = scan(flags);
    assert.equal(partial.apps.length, 5);
    const builds = partial.criteria.find((criterion) => criterion.id === "build-script");
    assert.equal(builds.status, "pass");
    assert.equal(builds.passRate, 0.8);
    assert.deepEqual(builds.appSummary, { passed: 4, total: 5 });
    assert.deepEqual(builds.appFailures, ["example"]);
    await writeFile(
      path.join(repo, "packages/core/build-a/package.json"),
      JSON.stringify({ name: "build-a", scripts: {} }),
    );
    assert.equal(status(scan(flags), "build-script"), "fail");
  } finally {
    await rm(repo, { recursive: true, force: true });
  }
});
