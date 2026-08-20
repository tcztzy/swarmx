/**
 * Boots the DeepSeek Harness `web` profile inside this process and reports the
 * URL its server bound to. The profile is DSH's own shipped template
 * (`dsh-base` + `dsh-web-app`), so the entire browser surface — every
 * `dsh-client-ui-*` package — is reused as published, with no renderer code of
 * our own.
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
} from "@deepseek-ai/dsh-app-boot";
import { provideCmdline } from "@deepseek-ai/dsh-cmdline";
import { resolveDshHome } from "@deepseek-ai/dsh-home-paths";

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

/** Installed manifest for the SwarmX client extension and its dependency closure. */
function conversationAnchor(): string {
  return createRequire(import.meta.url).resolve("@swarmx/dsh-ui-conversation/package.json");
}

/**
 * Module base anchoring bare plugin specifiers. The profile config lives under
 * $DSH_HOME, outside this project, so bare names must resolve against the
 * installation instead of the config directory. Loader expects a URL.
 */
function bareModuleBaseUrl(anchor: string): string {
  return pathToFileURL(anchor).href;
}

/** Product-owned patch mounted after DSH bundles and before user overrides. */
function conversationPatchPath(): string {
  return join(dirname(conversationAnchor()), "cordis.patch.yml");
}

/**
 * Bind the web server to an OS-assigned loopback port. Electron reads the
 * resolved port back off the service, so nothing has to agree on a constant.
 */
const PORT_PATCH = [{ id: "webserver", config: { host: HOST, port: 0 } }];

/**
 * The shipped agent presets (`code`, `cordis`, `minimal`, `standard`) live
 * beside the installed `dsh` package's own config, so only a launcher can
 * resolve them. Without this root the roster falls back to its built-in default
 * and the preset picker shows one entry.
 *
 * `trust: 'system'` marks them read-only product content. The writable root
 * (`$DSH_HOME/.agent-presets`) is the preset plugin's own default and needs no
 * patch, so a person's own presets appear either way.
 */
function presetPatch(anchor: string, config: Record<string, unknown>) {
  return {
    id: "agent-presets",
    config: {
      ...config,
      roots: [{ path: join(dirname(anchor), "config", "agent-presets"), trust: "system" }],
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
  healProfilesModuleFallback(anchor, home);
  healProfilesModuleFallback(conversationAnchor(), home);
  const profile = loadProfile(BIN_NAME, PROFILE, anchor, home);
  writeFileSync(join(profile.dir, ROOT_FILENAME), ROOT_CONFIG);
  // Layer order is the contract: DSH bundles, SwarmX product extensions, the
  // profile's user patch, the home-level one, then launcher overrides. Users
  // can therefore disable or replace the extension by id. Cloned because the
  // include pushes `insert` rows into the mounted tree by reference and later
  // patches mutate them in place.
  const layers = [
    profile.layers.flatMap((layer) => layer.patches),
    loadOptionalPatches(BIN_NAME, conversationPatchPath()) ?? [],
    profile.patches,
    loadOptionalPatches(BIN_NAME, join(home, PROFILE_PATCH_FILENAME)) ?? [],
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
      : [presetPatch(anchor, (presetRow.config ?? {}) as Record<string, unknown>)]),
  ]);
  const ctx = await boot(
    BIN_NAME,
    join(profile.dir, ROOT_FILENAME),
    patches,
    (hostCtx) => {
      provideCmdline(hostCtx, { args: ["--no-open"], exit: () => {} });
    },
    bareModuleBaseUrl(anchor),
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
