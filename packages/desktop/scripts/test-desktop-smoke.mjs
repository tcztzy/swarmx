import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import electronPath from "electron";

const desktopRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const temporaryRoot = await mkdtemp(path.join(tmpdir(), "swarmx-desktop-smoke-"));
const reportPath = path.join(temporaryRoot, "report.json");
const environment = {
  ...process.env,
  HOME: temporaryRoot,
  USERPROFILE: temporaryRoot,
  SWARMX_DESKTOP_SMOKE: "1",
  SWARMX_DESKTOP_SMOKE_REPORT: reportPath,
  ELECTRON_DISABLE_SECURITY_WARNINGS: "true",
};
delete environment.ELECTRON_RUN_AS_NODE;

try {
  const output = await runElectron(environment);
  const report = JSON.parse(await readFile(reportPath, "utf8"));
  if (report.ok !== true) {
    throw new Error(`Desktop smoke report failed: ${report.error ?? "unknown error"}\n${output}`);
  }
  process.stdout.write(`Desktop smoke passed: ${Object.keys(report.checks ?? {}).join(", ")}\n`);
} finally {
  await rm(temporaryRoot, { recursive: true, force: true });
}

function runElectron(env) {
  return new Promise((resolve, reject) => {
    const child = spawn(electronPath, [desktopRoot], {
      cwd: desktopRoot,
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let output = "";
    const append = (chunk) => {
      output = `${output}${chunk}`.slice(-32_000);
    };
    child.stdout.on("data", append);
    child.stderr.on("data", append);
    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`Desktop smoke timed out.\n${output}`));
    }, 30_000);
    child.once("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.once("exit", (code, signal) => {
      clearTimeout(timeout);
      if (code === 0) resolve(output);
      else reject(new Error(`Desktop smoke exited with ${code ?? signal}.\n${output}`));
    });
  });
}
