#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

if (process.platform !== "darwin") {
  throw new Error("macOS artifacts must be built on macOS.");
}

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifest = JSON.parse(readFileSync(join(packageRoot, "package.json"), "utf8"));
const archFlagIndex = process.argv.indexOf("--arch");
const arch = archFlagIndex >= 0 ? process.argv[archFlagIndex + 1] : process.arch;
if (arch !== "arm64" && arch !== "x64") {
  throw new Error(`Unsupported macOS architecture: ${arch ?? "missing"}`);
}

const builderArgs = [
  "exec",
  "electron-builder",
  "--mac",
  `--${arch}`,
  "--dir",
  "--publish",
  "never",
];
if (process.argv.includes("--notarize")) builderArgs.push("-c.mac.notarize=true");
execFileSync("pnpm", builderArgs, { cwd: packageRoot, stdio: "inherit" });

const releaseDir = join(packageRoot, "release");
const appPath = join(releaseDir, `mac-${arch}`, "SwarmX.app");
const zipPath = join(releaseDir, `SwarmX-${manifest.version}-${arch}.zip`);
const dmgPath = join(releaseDir, `SwarmX-${manifest.version}-${arch}.dmg`);
mkdirSync(releaseDir, { recursive: true });
rmSync(zipPath, { force: true });
rmSync(dmgPath, { force: true });

execFileSync("ditto", ["-c", "-k", "--sequesterRsrc", "--keepParent", appPath, zipPath], {
  stdio: "inherit",
});

const stagingDir = mkdtempSync(join(tmpdir(), "swarmx-dmg-"));
try {
  execFileSync("ditto", [appPath, join(stagingDir, "SwarmX.app")], { stdio: "inherit" });
  symlinkSync("/Applications", join(stagingDir, "Applications"));
  execFileSync(
    "hdiutil",
    [
      "create",
      "-volname",
      `SwarmX ${manifest.version}`,
      "-srcfolder",
      stagingDir,
      "-ov",
      "-format",
      "UDZO",
      dmgPath,
    ],
    { stdio: "inherit" },
  );
} finally {
  rmSync(stagingDir, { recursive: true, force: true });
}

console.log(`Created ${zipPath}`);
console.log(`Created ${dmgPath}`);
