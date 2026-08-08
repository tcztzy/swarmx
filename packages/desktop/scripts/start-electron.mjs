import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import electronPath from "electron";
import { forwardElectronStderr } from "./electron-stderr.mjs";

const appPath = fileURLToPath(new URL("..", import.meta.url));
const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;

const child = spawn(electronPath, [appPath, ...process.argv.slice(2)], {
  stdio: ["inherit", "inherit", "pipe"],
  env,
});
forwardElectronStderr(child.stderr);

child.once("error", (error) => {
  console.error(`Failed to launch SwarmX Desktop: ${error.message}`);
  process.exitCode = 1;
});

child.once("exit", (code, signal) => {
  process.exitCode = signal ? 1 : (code ?? 1);
});
