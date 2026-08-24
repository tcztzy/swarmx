import { execFileSync } from "node:child_process";
import { chmodSync, copyFileSync, existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(import.meta.dirname, "..");
const manifest = resolve(root, "native/writing-preview-runtime/Cargo.toml");
const executable =
  process.platform === "win32"
    ? "swarmx-writing-preview-runtime.exe"
    : "swarmx-writing-preview-runtime";
const source = resolve(root, "native/writing-preview-runtime/target/release", executable);
const destination = resolve(
  root,
  "packages/science/core/bin",
  `${process.platform}-${process.arch}`,
  executable,
);
const inputs = [
  fileURLToPath(import.meta.url),
  manifest,
  resolve(root, "native/writing-preview-runtime/Cargo.lock"),
  resolve(root, "native/writing-preview-runtime/src/main.rs"),
];
const upToDate =
  existsSync(destination) &&
  inputs.every((input) => statSync(input).mtimeMs <= statSync(destination).mtimeMs);
if (upToDate) process.exit(0);

execFileSync("cargo", ["build", "--locked", "--release", "--manifest-path", manifest], {
  cwd: root,
  stdio: "inherit",
});
mkdirSync(dirname(destination), { recursive: true });
copyFileSync(source, destination);
if (process.platform !== "win32") chmodSync(destination, 0o755);
if (process.platform === "darwin") {
  execFileSync("codesign", ["--force", "--sign", "-", destination], { stdio: "inherit" });
}
