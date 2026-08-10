#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { chmodSync, copyFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = resolve(packageRoot, "../..");
const crateRoot = join(repositoryRoot, "crates/swarmx-mem");
const cargoManifest = join(crateRoot, "Cargo.toml");
const cargoSource = readFileSync(cargoManifest, "utf8");
const runtimeVersion = cargoSource.match(/^version\s*=\s*"([^"]+)"/m)?.[1];
if (!runtimeVersion) throw new Error("Memory runtime Cargo package version is missing.");

const archFlagIndex = process.argv.indexOf("--arch");
const architecture = archFlagIndex >= 0 ? process.argv[archFlagIndex + 1] : process.arch;
const platformFlagIndex = process.argv.indexOf("--platform");
const platform = platformFlagIndex >= 0 ? process.argv[platformFlagIndex + 1] : process.platform;
const target = rustTarget(platform, architecture);
const executableName = platform === "win32" ? "swarmx-mem.exe" : "swarmx-mem";
const outputRoot = join(packageRoot, "build/mem-runtime");

execFileSync(
  "cargo",
  ["build", "--manifest-path", cargoManifest, "--locked", "--release", "--target", target],
  { cwd: repositoryRoot, stdio: "inherit" },
);

const source = join(repositoryRoot, "target", target, "release", executableName);
rmSync(outputRoot, { recursive: true, force: true });
mkdirSync(outputRoot, { recursive: true, mode: 0o755 });
const destination = join(outputRoot, executableName);
copyFileSync(source, destination);
if (platform !== "win32") chmodSync(destination, 0o755);
const digest = createHash("sha256").update(readFileSync(destination)).digest("hex");
const manifest = {
  schemaVersion: 1,
  runtimeVersion,
  protocolVersion: 1,
  targets: [
    {
      platform,
      architecture,
      path: executableName,
      sha256: `sha256:${digest}`,
    },
  ],
};
writeFileSync(join(outputRoot, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, {
  mode: 0o644,
});
console.log(`Built managed Memory runtime for ${platform}:${architecture}.`);

function rustTarget(platform, architecture) {
  const targets = {
    "darwin:arm64": "aarch64-apple-darwin",
    "darwin:x64": "x86_64-apple-darwin",
    "linux:arm64": "aarch64-unknown-linux-gnu",
    "linux:x64": "x86_64-unknown-linux-gnu",
    "win32:x64": "x86_64-pc-windows-msvc",
  };
  const target = targets[`${platform}:${architecture}`];
  if (!target) throw new Error(`Unsupported Memory runtime target: ${platform}:${architecture}`);
  return target;
}
