/**
 * Boots the DeepSeek Harness `web` profile inside this process and reports its
 * authenticated browser launch URL. The profile is DSH's own shipped template
 * (`dsh-base` + `dsh-web-app`), so the entire browser surface — every
 * `dsh-client-ui-*` package — remains the baseline. Product-owned extensions,
 * exact-version package patches stay explicit.
 */

import { writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
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
import type {} from "@deepseek-ai/dsh-client-connection";
import { provideCmdline } from "@deepseek-ai/dsh-cmdline";
import { resolveDshHome } from "@deepseek-ai/dsh-home-paths";
import type { Config as ScienceConfig } from "@swarmx/dsh-science/core";
import {
  projectScienceCarrierConfig,
  type ScienceCarrierConfig,
} from "./runtime/science-config.js";

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
  /** Bounded composed Science settings shared with the Codex product carrier. */
  scienceConfig: ScienceCarrierConfig;
  /** Token-bearing loopback URL the renderer exchanges for DSH's browser-session cookie. */
  url: string;
}

export interface StartHarnessOptions {
  productHome: string;
}

const moduleRequire = createRequire(import.meta.url);

function packageAnchor(packageId: string): string {
  return moduleRequire.resolve(`${packageId}/package.json`);
}

function patchPath(anchor: string): string {
  return join(dirname(anchor), "cordis.patch.yml");
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

function resolveEntryNames<T>(value: T, resolveName: (name: string) => string): T {
  const resolved = structuredClone(value);
  const visit = (current: unknown): void => {
    if (Array.isArray(current)) {
      for (const child of current) visit(child);
      return;
    }
    if (current === null || typeof current !== "object") return;
    const record = current as Record<string, unknown>;
    if (typeof record.id === "string" && typeof record.name === "string") {
      record.name = resolveName(record.name);
    }
    for (const child of Object.values(record)) visit(child);
  };
  visit(resolved);
  return resolved;
}

/**
 * Electron cannot always obtain Node's internal ESM loader, so its fallback
 * dynamic import ignores the Loader base URL. Resolve only active bundle-owned
 * entry names to absolute modules up front; all other DSH rows stay bare and
 * continue resolving from the installed app graph.
 */
function resolveProfileBundleEntries<T>(value: T, profile: Profile): T {
  const bundleNames = new Set(profile.layers.map((layer) => layer.packageName));
  const profileRequire = createRequire(join(profile.dir, "package.json"));
  return resolveEntryNames(value, (name) => {
    const owner = packageName(name);
    return owner !== undefined && bundleNames.has(owner) ? profileRequire.resolve(name) : name;
  });
}

/** Resolve one Host-only patch package without changing client-bearing package identities. */
function resolveHostPatchEntries<T>(value: T, packageId: string, anchor: string): T {
  return resolveEntryNames(value, (name) =>
    packageName(name) === packageId ? createRequire(anchor).resolve(name) : name,
  );
}

/**
 * Bind the web server to an OS-assigned loopback port. Electron reads the
 * resolved port back off the service, so nothing has to agree on a constant.
 */
const PORT_PATCH = [{ id: "webserver", config: { host: HOST, port: 0 } }];

/**
 * DSH's shipped presets (`code`, `cordis`, `minimal`, `standard`) and SwarmX's
 * `dsh-science`/`dsh-swarm` presets live beside their owning packages, so only a launcher
 * can resolve both roots. Without them the roster falls back to its built-in
 * default and the preset picker shows one entry.
 *
 * `trust: 'system'` marks them read-only product content. The writable root
 * (`$DSH_HOME/.agent-presets`) is the preset plugin's own default and needs no
 * patch, so a person's own presets appear either way.
 */
function presetPatch(
  dshAnchor: string,
  productPresetAnchors: readonly string[],
  config: Record<string, unknown>,
) {
  return {
    id: "agent-presets",
    config: {
      ...config,
      roots: [
        { path: join(dirname(dshAnchor), "config", "agent-presets"), trust: "system" },
        ...productPresetAnchors.map((anchor) => ({
          path: join(dirname(anchor), "config", "agent-presets"),
          trust: "system",
        })),
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
export async function startHarness(options: StartHarnessOptions): Promise<Harness> {
  const home = resolveDshHome();
  const anchor = packageAnchor("@deepseek-ai/dsh");
  const conversationPackageAnchor = packageAnchor("@swarmx/dsh-ui-conversation");
  const gitUiPackageAnchor = packageAnchor("@swarmx/dsh-ui-git");
  const dvcPackageAnchor = packageAnchor("@swarmx/dsh-dvc");
  const dvcUiPackageAnchor = packageAnchor("@swarmx/dsh-ui-dvc");
  const pkbPackageAnchor = packageAnchor("@swarmx/dsh-pkb");
  const sciencePackageAnchor = packageAnchor("@swarmx/dsh-science");
  const swarmPackageAnchor = packageAnchor("@swarmx/dsh-swarm");
  const swarmUiPackageAnchor = packageAnchor("@swarmx/dsh-ui-swarm");
  const profile = loadProfile(BIN_NAME, PROFILE, anchor, home);
  const productAnchor = moduleRequire.resolve("../package.json");
  await healProfilesModuleFallback({
    installAnchor: anchor,
    // The desktop manifest is a closure-only profile root: alpha.4 reserves the
    // exact DSH installation while projecting app-owned bare packages locally.
    profile: {
      ...profile,
      layers: [
        ...profile.layers,
        {
          packageName: "@swarmx/desktop",
          packageDir: dirname(productAnchor),
          patchPath: productAnchor,
          patches: [],
        },
      ],
    },
    home,
  });
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
    loadOptionalPatches(BIN_NAME, patchPath(conversationPackageAnchor)) ?? [],
    loadOptionalPatches(BIN_NAME, patchPath(gitUiPackageAnchor)) ?? [],
    resolveHostPatchEntries(
      loadOptionalPatches(BIN_NAME, patchPath(dvcPackageAnchor)) ?? [],
      "@swarmx/dsh-dvc",
      dvcPackageAnchor,
    ),
    loadOptionalPatches(BIN_NAME, patchPath(dvcUiPackageAnchor)) ?? [],
    resolveHostPatchEntries(
      loadOptionalPatches(BIN_NAME, patchPath(pkbPackageAnchor)) ?? [],
      "@swarmx/dsh-pkb",
      pkbPackageAnchor,
    ),
    loadOptionalPatches(BIN_NAME, patchPath(sciencePackageAnchor)) ?? [],
    loadOptionalPatches(BIN_NAME, patchPath(swarmPackageAnchor)) ?? [],
    loadOptionalPatches(BIN_NAME, patchPath(swarmUiPackageAnchor)) ?? [],
    resolveProfileBundleEntries(profile.patches, profile),
    resolveProfileBundleEntries(
      loadOptionalPatches(BIN_NAME, join(home, PROFILE_PATCH_FILENAME)) ?? [],
      profile,
    ),
  ];
  // An id-targeted patch replaces the row's whole config, so launcher-enforced
  // values must extend the composed row rather than discard profile/home
  // settings. `composeEntries` resolves the same layers into the tree those
  // patches produce, for reading only.
  const composedRows = composeEntries(layers);
  const presetRow = composedRows.find((row) => row.id === "agent-presets");
  const scienceRow = composedRows.find((row) => row.id === "swarmx-science");
  const pkbRow = composedRows.find((row) => row.id === "swarmx-pkb");
  const swarmRow = composedRows.find((row) => row.id === "swarmx-swarm");
  const scienceConfig = {
    ...((scienceRow?.config ?? {}) as Record<string, unknown>),
    root: join(options.productHome, "science"),
  } as ScienceConfig;
  const patches = structuredClone([
    ...layers.flat(),
    ...PORT_PATCH,
    {
      id: "swarmx-science",
      config: scienceConfig,
    },
    {
      id: "swarmx-pkb",
      config: {
        ...((pkbRow?.config ?? {}) as Record<string, unknown>),
        root: join(options.productHome, "pkb", "vault"),
      },
    },
    {
      id: "swarmx-swarm",
      config: {
        ...((swarmRow?.config ?? {}) as Record<string, unknown>),
        recoveryOwner: false,
        root: join(options.productHome, "swarm"),
      },
    },
    ...(presetRow === undefined
      ? []
      : [
          presetPatch(
            anchor,
            [sciencePackageAnchor, swarmPackageAnchor],
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
    },
    pathToFileURL(join(profile.dir, "package.json")).href,
  );
  const port = ctx.get("webServer")?.port;
  if (port === undefined) {
    await ctx.fiber.dispose();
    throw new Error(`${BIN_NAME}: the web profile mounted without a web server`);
  }
  const connection = ctx.get("connection");
  if (connection === undefined) {
    await ctx.fiber.dispose();
    throw new Error(`${BIN_NAME}: the web profile mounted without a browser connection`);
  }
  return {
    ctx,
    scienceConfig: projectScienceCarrierConfig(scienceConfig),
    url: connection.authenticatedUrl(`http://${HOST}:${String(port)}`),
  };
}

/** Path of the user's patch layer for this profile, for documentation and errors. */
export function profilePatchPath(home: string = resolveDshHome()): string {
  return join(home, "profiles", PROFILE, PROFILE_PATCH_FILENAME);
}
