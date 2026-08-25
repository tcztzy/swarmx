/**
 * Boots the DeepSeek Harness `web` profile inside this process and reports the
 * URL its server bound to. The profile is DSH's own shipped template
 * (`dsh-base` + `dsh-web-app`), so the entire browser surface — every
 * `dsh-client-ui-*` package — remains the baseline. Product-owned extensions,
 * exact-version package patches, and the Markdown file-link route stay explicit.
 */

import { writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { Context } from "@deepseek-ai/cordis";
import {
  boot,
  composeEntries,
  healProfilesModuleFallback,
  loadOptionalPatches,
  loadProfile,
  PROFILE_PATCH_FILENAME,
  type Profile,
} from "@deepseek-ai/dsh-app-boot";
import { provideCmdline } from "@deepseek-ai/dsh-cmdline";
import { resolveDshHome } from "@deepseek-ai/dsh-home-paths";
import * as MarkdownFileLinks from "./markdown-file-links.js";

/** Diagnostic prefix on boot failures. */
const BIN_NAME = "swarmx";

/** DSH profile this surface boots. Its template is `dsh-base` + `dsh-web-app`. */
const PROFILE = "web";

/** Loopback host the in-process server binds to; never exposed off-machine. */
const HOST = "127.0.0.1";

/** A booted harness: the root context plus the URL its web server bound to. */
export interface Harness {
  /** Root Cordis context owning every mounted plugin. */
  ctx: Context;
  /** Loopback URL the renderer loads. */
  url: string;
}

/**
 * Absolute path of the installed `dsh` package manifest. Profile bundle names
 * resolve two-anchored against this installation, so it must point at the real
 * dependency rather than at our own package.
 */
function installAnchor(): string {
  return createRequire(import.meta.url).resolve("@deepseek-ai/dsh/package.json");
}

/** Desktop composition manifest owning every product plugin dependency. */
function appAnchor(): string {
  return join(dirname(dirname(fileURLToPath(import.meta.url))), "package.json");
}

/** Installed manifest for the SwarmX client extension and its dependency closure. */
function conversationAnchor(): string {
  return createRequire(import.meta.url).resolve("@swarmx/dsh-ui-conversation/package.json");
}

/** Installed manifest for the read-only SwarmX Git status UI. */
function gitUiAnchor(): string {
  return createRequire(import.meta.url).resolve("@swarmx/dsh-ui-git/package.json");
}

/** Installed manifest for the Host-only SwarmX DVC capability. */
function dvcAnchor(): string {
  return createRequire(import.meta.url).resolve("@swarmx/dsh-dvc/package.json");
}

/** Installed manifest for the read-only SwarmX DVC status UI. */
function dvcUiAnchor(): string {
  return createRequire(import.meta.url).resolve("@swarmx/dsh-ui-dvc/package.json");
}

/** Installed manifest for the SwarmX science Host service. */
function scienceAnchor(): string {
  return createRequire(import.meta.url).resolve("@swarmx/dsh-science/package.json");
}

/** Installed manifest for the SwarmX private Personal Knowledge Base. */
function pkbAnchor(): string {
  return createRequire(import.meta.url).resolve("@swarmx/dsh-pkb/package.json");
}

/** Installed manifest for the SwarmX Science Chat and Side View client integration. */
function scienceUiAnchor(): string {
  return createRequire(import.meta.url).resolve("@swarmx/dsh-ui-science/package.json");
}

/** Module base anchoring in-box and SwarmX-owned bare plugin specifiers. */
function bareModuleBaseUrl(anchor: string): string {
  return pathToFileURL(anchor).href;
}

/** Return the package name portion of one bare module specifier. */
function packageName(specifier: string): string | undefined {
  if (specifier.startsWith("@")) {
    const [scope, name] = specifier.split("/");
    return scope === undefined || name === undefined ? undefined : `${scope}/${name}`;
  }
  if (specifier.startsWith(".") || specifier.startsWith("/") || specifier.includes(":")) {
    return undefined;
  }
  return specifier.split("/")[0];
}

/**
 * Electron cannot always obtain Node's internal ESM loader, so its fallback
 * dynamic import ignores the Loader base URL. Resolve only active bundle-owned
 * entry names to absolute modules up front; all other DSH rows stay bare and
 * continue resolving from the installed app graph.
 */
function resolveProfileBundleEntries<T>(value: T, profile: Profile): T {
  const resolved = structuredClone(value);
  const bundleNames = new Set(profile.layers.map((layer) => layer.packageName));
  const profileRequire = createRequire(join(profile.dir, "package.json"));

  const visit = (current: unknown): void => {
    if (Array.isArray(current)) {
      for (const child of current) visit(child);
      return;
    }
    if (current === null || typeof current !== "object") return;
    const record = current as Record<string, unknown>;
    if (typeof record.id === "string" && typeof record.name === "string") {
      const owner = packageName(record.name);
      if (owner !== undefined && bundleNames.has(owner)) {
        record.name = profileRequire.resolve(record.name);
      }
    }
    for (const child of Object.values(record)) visit(child);
  };

  visit(resolved);
  return resolved;
}

/** Resolve one Host-only patch package without changing client-bearing package identities. */
function resolveHostPatchEntries<T>(value: T, packageId: string, anchor: string): T {
  const resolved = structuredClone(value);

  const visit = (current: unknown): void => {
    if (Array.isArray(current)) {
      for (const child of current) visit(child);
      return;
    }
    if (current === null || typeof current !== "object") return;
    const record = current as Record<string, unknown>;
    if (typeof record.id === "string" && typeof record.name === "string") {
      if (packageName(record.name) === packageId) {
        record.name = createRequire(anchor).resolve(record.name);
      }
    }
    for (const child of Object.values(record)) visit(child);
  };

  visit(resolved);
  return resolved;
}

/** Product-owned patch mounted after DSH bundles and before user overrides. */
function conversationPatchPath(): string {
  return join(dirname(conversationAnchor()), "cordis.patch.yml");
}

/** Read-only Git status service and client integration layer. */
function gitUiPatchPath(): string {
  return join(dirname(gitUiAnchor()), "cordis.patch.yml");
}

/** Host-only DVC status, pull, and isolated reproduction layer. */
function dvcPatchPath(): string {
  return join(dirname(dvcAnchor()), "cordis.patch.yml");
}

/** Read-only DVC status service and client integration layer. */
function dvcUiPatchPath(): string {
  return join(dirname(dvcUiAnchor()), "cordis.patch.yml");
}

/** Product-owned Science Journal service and client integration layer. */
function sciencePatchPath(): string {
  return join(dirname(scienceAnchor()), "cordis.patch.yml");
}

/** Product-owned PKB service, tool, and session-search activation layer. */
function pkbPatchPath(): string {
  return join(dirname(pkbAnchor()), "cordis.patch.yml");
}

/**
 * Bind the web server to an OS-assigned loopback port. Electron reads the
 * resolved port back off the service, so nothing has to agree on a constant.
 */
const PORT_PATCH = [{ id: "webserver", config: { host: HOST, port: 0 } }];

/**
 * DSH's shipped presets (`code`, `cordis`, `minimal`, `standard`) and SwarmX's
 * `dsh-science` preset live beside their owning packages, so only a launcher
 * can resolve both roots. Without them the roster falls back to its built-in
 * default and the preset picker shows one entry.
 *
 * `trust: 'system'` marks them read-only product content. The writable root
 * (`$DSH_HOME/.agent-presets`) is the preset plugin's own default and needs no
 * patch, so a person's own presets appear either way.
 */
function presetPatch(
  dshAnchor: string,
  sciencePackageAnchor: string,
  config: Record<string, unknown>,
) {
  return {
    id: "agent-presets",
    config: {
      ...config,
      roots: [
        { path: join(dirname(dshAnchor), "config", "agent-presets"), trust: "system" },
        {
          path: join(dirname(sciencePackageAnchor), "config", "agent-presets"),
          trust: "system",
        },
      ],
    },
  };
}

/** Root config filename inside a profile directory. */
const ROOT_FILENAME = "cordis.yml";

/**
 * The empty entry list every profile tree is patched over. Owned by the
 * launcher, not the user: the composed tree arrives entirely as patch layers, so
 * rewriting this on every boot keeps a hand-edited copy from silently
 * shadowing them.
 */
const ROOT_CONFIG = `# SwarmX profile root — an empty entry list. The tree is composed as patches:
# each bundle in package.json's dsh.profile.bundles, then cordis.patch.yml.
# Edit cordis.patch.yml, not this file.
[]
`;

/**
 * Boot the harness and resolve once its tree is mounted and activated.
 *
 * `prepare` provides the two launcher facts every DSH tree expects before any
 * entry mounts: the inner command line and a bounded exit request. Electron
 * owns the visible surface, so `--no-open` suppresses the Web profile's normal
 * handoff to the system browser.
 */
export async function startHarness(): Promise<Harness> {
  const home = resolveDshHome();
  const anchor = installAnchor();
  const conversationPackageAnchor = conversationAnchor();
  const gitUiPackageAnchor = gitUiAnchor();
  const dvcPackageAnchor = dvcAnchor();
  const dvcUiPackageAnchor = dvcUiAnchor();
  const pkbPackageAnchor = pkbAnchor();
  const sciencePackageAnchor = scienceAnchor();
  const scienceUiPackageAnchor = scienceUiAnchor();
  healProfilesModuleFallback(anchor, home);
  healProfilesModuleFallback(conversationPackageAnchor, home);
  healProfilesModuleFallback(gitUiPackageAnchor, home);
  healProfilesModuleFallback(dvcPackageAnchor, home);
  healProfilesModuleFallback(dvcUiPackageAnchor, home);
  healProfilesModuleFallback(pkbPackageAnchor, home);
  healProfilesModuleFallback(sciencePackageAnchor, home);
  healProfilesModuleFallback(scienceUiPackageAnchor, home);
  const profile = loadProfile(BIN_NAME, PROFILE, anchor, home);
  writeFileSync(join(profile.dir, ROOT_FILENAME), ROOT_CONFIG);
  // Layer order is the contract: DSH bundles, SwarmX product extensions, the
  // profile's user patch, the home-level one, then launcher overrides. Users
  // can therefore disable or replace the extension by id. Cloned because the
  // include pushes `insert` rows into the mounted tree by reference and later
  // patches mutate them in place.
  const layers = [
    resolveProfileBundleEntries(
      profile.layers.flatMap((layer) => layer.patches),
      profile,
    ),
    loadOptionalPatches(BIN_NAME, conversationPatchPath()) ?? [],
    loadOptionalPatches(BIN_NAME, gitUiPatchPath()) ?? [],
    resolveHostPatchEntries(
      loadOptionalPatches(BIN_NAME, dvcPatchPath()) ?? [],
      "@swarmx/dsh-dvc",
      dvcPackageAnchor,
    ),
    loadOptionalPatches(BIN_NAME, dvcUiPatchPath()) ?? [],
    resolveHostPatchEntries(
      loadOptionalPatches(BIN_NAME, pkbPatchPath()) ?? [],
      "@swarmx/dsh-pkb",
      pkbPackageAnchor,
    ),
    loadOptionalPatches(BIN_NAME, sciencePatchPath()) ?? [],
    resolveProfileBundleEntries(profile.patches, profile),
    resolveProfileBundleEntries(
      loadOptionalPatches(BIN_NAME, join(home, PROFILE_PATCH_FILENAME)) ?? [],
      profile,
    ),
  ];
  // An id-targeted patch replaces the row's whole config, so the preset patch
  // needs the composed row it is overriding. `composeEntries` resolves the same
  // layers into the tree those patches produce, for reading only.
  const presetRow = composeEntries(layers).find((row) => row.id === "agent-presets");
  const patches = structuredClone([
    ...layers.flat(),
    ...PORT_PATCH,
    ...(presetRow === undefined
      ? []
      : [
          presetPatch(
            anchor,
            sciencePackageAnchor,
            (presetRow.config ?? {}) as Record<string, unknown>,
          ),
        ]),
  ]);
  const ctx = await boot(
    BIN_NAME,
    join(profile.dir, ROOT_FILENAME),
    patches,
    (hostCtx) => {
      provideCmdline(hostCtx, { args: ["--no-open"], exit: () => {} });
      hostCtx.plugin(MarkdownFileLinks);
    },
    bareModuleBaseUrl(appAnchor()),
  );
  const port = ctx.get("webServer")?.port;
  if (port === undefined) {
    await ctx.fiber.dispose();
    throw new Error(`${BIN_NAME}: the web profile mounted without a web server`);
  }
  return { ctx, url: `http://${HOST}:${String(port)}` };
}

/** Path of the user's patch layer for this profile, for documentation and errors. */
export function profilePatchPath(home: string = resolveDshHome()): string {
  return join(home, "profiles", PROFILE, PROFILE_PATCH_FILENAME);
}
