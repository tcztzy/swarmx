#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const REPOSITORY_ROOT = fileURLToPath(new URL("..", import.meta.url));
const RELEASE_PACKAGES = [
  { directory: "packages/core", name: "@swarmx/core" },
  { directory: "packages/runtime", name: "@swarmx/runtime" },
  { directory: "packages/acp-server", name: "@swarmx/acp-server" },
  { directory: "packages/cli", name: "@swarmx/cli" },
  { directory: "packages/desktop", name: "@swarmx/desktop" },
  { directory: "packages/swarmx", name: "swarmx" },
];
const RELEASE_MANIFESTS = [
  "package.json",
  ...RELEASE_PACKAGES.map(({ directory }) => `${directory}/package.json`),
];
const INTERNAL_PACKAGES = new Set(RELEASE_PACKAGES.map(({ name }) => name));
const PACK_ONLY = process.argv.includes("--pack-only");

async function main() {
  const version = releaseVersion();
  const packRoot = mkdtempSync(join(tmpdir(), "swarmx-npm-"));
  try {
    const packages = RELEASE_PACKAGES.map((releasePackage) =>
      packReleasePackage(releasePackage, version, packRoot),
    );
    if (PACK_ONLY) {
      for (const packed of packages) {
        process.stdout.write(`packed ${packed.name}@${version} ${packed.integrity}\n`);
      }
      return;
    }

    for (const packed of packages) await publishPackage(packed, version);
    for (const packed of packages) await verifyPublishedPackage(packed, version);
  } finally {
    rmSync(packRoot, { recursive: true });
  }
}

function releaseVersion() {
  const tag = process.env.RELEASE_TAG?.trim();
  if (!tag || !/^v\d+\.\d+\.\d+$/.test(tag)) {
    throw new Error("RELEASE_TAG must be a stable v<major>.<minor>.<patch> tag.");
  }
  const version = tag.slice(1);
  const manifestVersions = RELEASE_MANIFESTS.map((path) => {
    const manifest = readJson(path);
    if (typeof manifest.version !== "string") {
      throw new Error(`${path} does not declare a version.`);
    }
    return { path, version: manifest.version };
  });
  const mismatched = manifestVersions.filter((entry) => entry.version !== version);
  if (mismatched.length > 0) {
    throw new Error(
      `Tag/package version mismatch for ${tag}: ${mismatched
        .map((entry) => `${entry.path}=${entry.version}`)
        .join(", ")}`,
    );
  }

  const runtimeSource = readFileSync(
    resolve(REPOSITORY_ROOT, "packages/core/src/version.ts"),
    "utf8",
  );
  const runtimeVersion = runtimeSource.match(/export const SWARMX_VERSION = "([^"]+)";/)?.[1];
  if (runtimeVersion !== version) {
    throw new Error(`Tag/runtime version mismatch for ${tag}: ${runtimeVersion ?? "missing"}`);
  }
  return version;
}

function packReleasePackage(releasePackage, version, packRoot) {
  const output = run("pnpm", [
    "--dir",
    releasePackage.directory,
    "pack",
    "--pack-destination",
    packRoot,
    "--json",
  ]);
  const packResult = JSON.parse(output);
  const filename = resolve(packResult.filename ?? "");
  assertInside(packRoot, filename);
  if (!existsSync(filename)) throw new Error(`pnpm pack did not create ${filename}.`);

  const packedManifest = JSON.parse(run("tar", ["-xOf", filename, "package/package.json"]));
  if (packedManifest.name !== releasePackage.name || packedManifest.version !== version) {
    throw new Error(
      `Packed manifest mismatch: expected ${releasePackage.name}@${version}, got ${
        packedManifest.name ?? "missing"
      }@${packedManifest.version ?? "missing"}.`,
    );
  }
  verifyPackedDependencies(packedManifest, version);
  return {
    ...releasePackage,
    filename,
    integrity: fileIntegrity(filename),
  };
}

function verifyPackedDependencies(manifest, version) {
  for (const field of ["dependencies", "optionalDependencies", "peerDependencies"]) {
    for (const [name, range] of Object.entries(manifest[field] ?? {})) {
      if (typeof range !== "string") continue;
      if (range.startsWith("workspace:")) {
        throw new Error(`${manifest.name} packed unresolved ${field}.${name}=${range}.`);
      }
      if (INTERNAL_PACKAGES.has(name) && range !== version) {
        throw new Error(
          `${manifest.name} packed ${field}.${name}=${range}; expected exact ${version}.`,
        );
      }
    }
  }
}

async function publishPackage(packed, version) {
  const existing = registryIntegrity(packed.name, version);
  if (existing) {
    assertMatchingIntegrity(packed, existing, version);
    process.stdout.write(`verified existing ${packed.name}@${version}\n`);
    return;
  }

  run("npm", ["publish", packed.filename, "--access", "public"], { inherit: true });
  await waitForRegistryIntegrity(packed, version);
  process.stdout.write(`published ${packed.name}@${version}\n`);
}

async function verifyPublishedPackage(packed, version) {
  const integrity = await waitForRegistryIntegrity(packed, version);
  assertMatchingIntegrity(packed, integrity, version);
}

async function waitForRegistryIntegrity(packed, version) {
  for (let attempt = 0; attempt < 10; attempt++) {
    const integrity = registryIntegrity(packed.name, version);
    if (integrity) return integrity;
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 2_000));
  }
  throw new Error(`npm did not expose ${packed.name}@${version} after publication.`);
}

function assertMatchingIntegrity(packed, registryValue, version) {
  if (registryValue !== packed.integrity) {
    throw new Error(
      `Registry/local integrity mismatch for ${packed.name}@${version}: ` +
        `${registryValue} != ${packed.integrity}`,
    );
  }
}

function registryIntegrity(name, version) {
  const result = command("npm", ["view", `${name}@${version}`, "dist.integrity", "--json"]);
  if (result.status === 0) {
    const value = JSON.parse(result.stdout);
    if (typeof value !== "string" || !value.startsWith("sha512-")) {
      throw new Error(`npm returned invalid dist.integrity for ${name}@${version}.`);
    }
    return value;
  }
  const error = `${result.stdout}\n${result.stderr}`;
  if (/\bE404\b|404 Not Found/i.test(error)) return null;
  throw commandError("npm", ["view", `${name}@${version}`, "dist.integrity", "--json"], result);
}

function fileIntegrity(path) {
  return `sha512-${createHash("sha512").update(readFileSync(path)).digest("base64")}`;
}

function readJson(path) {
  return JSON.parse(readFileSync(resolve(REPOSITORY_ROOT, path), "utf8"));
}

function assertInside(root, path) {
  const canonicalRoot = realpathSync(root);
  const canonicalParent = realpathSync(dirname(path));
  const relation = relative(canonicalRoot, canonicalParent);
  if (relation === ".." || relation.startsWith(`..${sep}`) || resolve(path) === canonicalRoot) {
    throw new Error(`Packed artifact escaped the temporary release directory: ${path}`);
  }
}

function run(executable, args, options = {}) {
  const result = command(executable, args, options);
  if (result.status !== 0) throw commandError(executable, args, result);
  return result.stdout;
}

function command(executable, args, options = {}) {
  const result = spawnSync(executable, args, {
    cwd: REPOSITORY_ROOT,
    encoding: "utf8",
    stdio: options.inherit ? "inherit" : "pipe",
  });
  if (result.error) throw result.error;
  return {
    status: result.status,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
  };
}

function commandError(executable, args, result) {
  const detail = `${result.stdout}\n${result.stderr}`.trim();
  return new Error(
    `${executable} ${args.join(" ")} exited ${result.status ?? "without status"}${
      detail ? `:\n${detail}` : ""
    }`,
  );
}

await main();
