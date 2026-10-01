import {
  existsSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { isAbsolute, join } from "node:path";
import { expect, it, vi } from "vitest";
import { spawnProcess } from "../src/host/process-runner.js";

const packagedApp = process.env.SWARMX_PACKAGED_APP;

it.runIf(process.platform === "darwin" && packagedApp !== undefined)(
  "runs the exported macOS app, native integrations, Typst, renderer and preload IPC",
  async () => {
    expect(isAbsolute(packagedApp ?? "")).toBe(true);
    // macOS's per-user temporary path can exceed the Host's Unix socket path limit.
    const scratch = realpathSync(mkdtempSync("/tmp/swarmx-package-"));
    const driver = join(scratch, "inspect-package.cjs");
    const result = join(scratch, "result.json");
    const application = join(packagedApp ?? "", "Contents", "Resources", "app");
    const env: NodeJS.ProcessEnv = {
      ...process.env,
      SWARMX_HOME: join(scratch, "swarmx"),
      SWARMX_CWD: scratch,
      SWARMX_AGENT: "pi",
      PI_CODING_AGENT_DIR: join(scratch, "pi"),
    };
    delete env.ELECTRON_RUN_AS_NODE;
    delete env.NODE_OPTIONS;
    delete env.SWARMX_DEV;
    writeFileSync(
      driver,
      `const assert = require("node:assert/strict");
const { accessSync, writeFileSync } = require("node:fs");
const { createRequire, registerHooks } = require("node:module");
const { join } = require("node:path");
const { fileURLToPath, pathToFileURL } = require("node:url");
const root = ${JSON.stringify(application)};
const appRequire = createRequire(join(root, "package.json"));
const { app, BrowserWindow } = appRequire("electron");
const load = (path) => import(pathToFileURL(join(root, path)).href);
registerHooks({
  resolve(specifier, context, nextResolve) {
    const result = nextResolve(specifier, context);
    if (result.url.startsWith("file:")) {
      assert.ok(fileURLToPath(result.url).startsWith(root + "/"),
        "Module resolved outside the packaged app: " + result.url);
    }
    return result;
  },
});
async function waitFor(check) {
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    const value = await check();
    if (value) return value;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("Packaged app did not become ready");
}
async function verify() {
  await app.whenReady();
  assert.equal(app.isPackaged, true);
  assert.equal(app.getAppPath(), root);
  assert.equal(process.cwd(), ${JSON.stringify(scratch)});
  for (const path of ["preload.cjs", "dist/main.js", "dist/renderer/index.html",
    "resources/hermes-native.py", "resources/python/Dockerfile",
    "resources/openclaw-plugin/index.js", "resources/openclaw-plugin/openclaw.plugin.json"]) {
    accessSync(join(root, path));
  }
  for (const resource of ["@swarmx/swarm/skills/delegate/SKILL.md", "@swarmx/memory/skills/memory/SKILL.md"]) {
    const path = appRequire.resolve(resource);
    assert.ok(path.startsWith(root + "/"), "Skill resolved outside the packaged app: " + path);
    accessSync(path);
  }
  const { ScienceCore, DEFAULT_WRITING_PREVIEW_RUNTIME_COMMAND } =
    await import(pathToFileURL(appRequire.resolve("@swarmx/science")).href);
  assert.ok(DEFAULT_WRITING_PREVIEW_RUNTIME_COMMAND.startsWith(root + "/"));
  accessSync(DEFAULT_WRITING_PREVIEW_RUNTIME_COMMAND);
  const { NodeScienceProcessRuntime } = await load("dist/host/process-runner.js");
  const disposers = [];
  const science = new ScienceCore({
    subprocess: new NodeScienceProcessRuntime(),
    onDispose: (dispose) => disposers.push(dispose),
  }, { root: join(process.cwd(), "science") }, () => ({ key: "package", root: process.cwd() }));
  try {
    writeFileSync("package.typ", "= SwarmX package test\\nBundled Typst compiles this document.\\n");
    const preview = await science.previewTypstDocument("package", { relativePath: "package.typ" });
    assert.equal(preview.status, "ready", preview.diagnostics.join("\\n"));
    assert.equal(Buffer.from(preview.pdfBase64, "base64").subarray(0, 5).toString(), "%PDF-");
    assert.ok(preview.pdfSize > 1000);
  } finally {
    for (const dispose of disposers) await dispose();
  }
  const window = await waitFor(() => BrowserWindow.getAllWindows()[0]);
  await waitFor(() => window.webContents.getURL() === pathToFileURL(join(root, "dist/renderer/index.html")).href);
  await waitFor(() => window.webContents.executeJavaScript(
    'Boolean(document.getElementById("root")?.querySelector("button"))'
  ));
  const settings = await window.webContents.executeJavaScript("window.swarmx.settings.read()");
  assert.equal(settings.cwd, process.cwd());
  assert.ok(settings.policy);
  const bootstrap = await window.webContents.executeJavaScript("window.swarmx.bootstrap()");
  assert.equal(bootstrap.cwd, process.cwd());
  assert.equal(bootstrap.sessionError, undefined);
  assert.ok(Array.isArray(bootstrap.agents));
  assert.ok(Array.isArray(bootstrap.sessions));
  const { AGENT_IDS } = await load("dist/agent.js");
  for (const id of AGENT_IDS) await load("dist/agents/" + id + ".js");
  const sdkRequire = createRequire(appRequire.resolve("@deepseek-ai/dsh-sdk-client"));
  const dshRequire = createRequire(sdkRequire.resolve("@deepseek-ai/dsh/package.json"));
  for (const id of ["@deepseek-ai/dsh-app-boot", "@deepseek-ai/dsh-agent-loop"]) {
    await import(pathToFileURL(dshRequire.resolve(id)).href);
  }
}
verify().then(
  () => writeFileSync(${JSON.stringify(result)}, JSON.stringify({ ok: true })),
  (error) => writeFileSync(${JSON.stringify(result)}, JSON.stringify({ error: error.stack })),
).finally(() => app.quit());
`,
    );
    const desktop = spawnProcess({
      argv: [
        join(packagedApp ?? "", "Contents", "MacOS", "SwarmX"),
        "--inspect=0",
        `--user-data-dir=${join(scratch, "electron")}`,
      ],
      cwd: scratch,
      env,
      graceMs: 2_000,
      stdin: "ignore",
      stdout: { maxBytes: 64 * 1024 },
      stderr: { maxBytes: 64 * 1024 },
    });
    let inspector: ReturnType<typeof spawnProcess> | undefined;
    try {
      const inspectorAddress = /ws:\/\/127\.0\.0\.1:\d+\/[\da-f-]+/u;
      const inspectorUrl = await vi.waitUntil(
        () => {
          const stderr = desktop.stderr?.readFrom(0).text ?? "";
          expect(desktop.child.exitCode, stderr).toBeNull();
          expect(desktop.child.signalCode, stderr).toBeNull();
          return stderr.match(inspectorAddress)?.[0];
        },
        { timeout: 30_000 },
      );
      // Node's official debugger client keeps the test independent of the CDP wire protocol.
      inspector = spawnProcess({
        argv: [process.execPath, "inspect", new URL(inspectorUrl).host],
        cwd: scratch,
        env,
        graceMs: 1_000,
        stdin: "pipe",
        stdout: { maxBytes: 64 * 1024 },
        stderr: { maxBytes: 64 * 1024 },
      });
      await expect
        .poll(() => inspector?.stdout?.readFrom(0).text, { timeout: 10_000 })
        .toContain("debug>");
      // Inspector evaluation can interrupt Electron before its module aliases are installed.
      inspector.child.stdin?.write(
        `exec void setImmediate(() => { try { process.getBuiltinModule("module").createRequire(${JSON.stringify(join(application, "package.json"))})(${JSON.stringify(driver)}); } catch (error) { process.getBuiltinModule("fs").writeFileSync(${JSON.stringify(result)}, JSON.stringify({ error: error.stack })); } })\n`,
      );
      await vi.waitUntil(
        () => {
          if (existsSync(result)) return true;
          expect(desktop.child.exitCode).toBeNull();
          expect(desktop.child.signalCode).toBeNull();
          return false;
        },
        { timeout: 45_000 },
      );
      expect(JSON.parse(readFileSync(result, "utf8"))).toEqual({ ok: true });
    } catch (error) {
      throw new Error(
        [
          `Desktop pid=${desktop.child.pid} exitCode=${desktop.child.exitCode} signal=${desktop.child.signalCode}`,
          desktop.stderr?.readFrom(0).text,
          inspector?.stdout?.readFrom(0).text,
          inspector?.stderr?.readFrom(0).text,
        ].join("\n"),
        { cause: error },
      );
    } finally {
      inspector?.terminate();
      desktop.terminate();
      await Promise.all([inspector?.done, desktop.done]);
      rmSync(scratch, { recursive: true, force: true });
    }
  },
  100_000,
);
