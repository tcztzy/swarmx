/**
 * Shared tsdown preset for UI plugin client bundles. Emits a closure-factory
 * artifact: the bundle calls window.__ModuleLoader__.load({id, factory})
 * and resolves externals through the injected require (loader module table —
 * cordis DI entities, no globals, no import map). CSS Modules are compiled by
 * lightningcss inside the bundle: importing `x.module.css` yields the
 * hashed class map, and the css text auto-injects a <style data-plugin="<id>">
 * tag at factory execution (the loader removes plugin-owned tags on unload).
 * The virtual loader registers each real stylesheet as a watch dependency.
 */

import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { basename, dirname, relative, resolve as resolvePath, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { transform } from "lightningcss";
import type { UserConfig } from "tsdown";

/**
 * Virtual-id wrapper keeping module CSS away from tsdown's own css pipeline
 * (which requires @tsdown/css). The suffix matters: tsdown's guard matches ids
 * ending in `.css`, so the virtual id must not.
 */
const CSS_VIRTUAL_PREFIX = "\0dsh-css:";
const CSS_VIRTUAL_SUFFIX = ".mjs";

/**
 * Workspace mode replaces an empty config array with the root defaults. A
 * falsey entry instead removes this package before entry resolution.
 */
const SKIP_WORKSPACE_BUILD: UserConfig = { entry: "" };

/** DSH preloads the client runtime factory before dependent plugin bundles. */
const RUNTIME_STORE_EXEMPTION = "@deepseek-ai/dsh-client-runtime/client";

/** Externals resolved from the DSH Web loader module table. */
export const CLIENT_EXTERNALS: readonly string[] = [
  "react",
  "react/jsx-runtime",
  "react-dom",
  "react-dom/client",
  "@deepseek-ai/cordis",
  "@deepseek-ai/dsh-client-ui-slots",
  "@deepseek-ai/dsh-client-ui-primitives",
  RUNTIME_STORE_EXEMPTION,
];

const REPOSITORY_ROOT = fileURLToPath(new URL("../..", import.meta.url));

/** Rebase a physical lib-relative source onto a browser URL that mirrors the repository directories. */
function browserSourcePath(source: string, sourcemapPath: string): string {
  if (!source.startsWith(".")) return source;
  const physicalSource = resolvePath(dirname(sourcemapPath), source);
  const repositoryPath = relative(REPOSITORY_ROOT, physicalSource).split(sep).join("/");
  return repositoryPath.startsWith("packages/") ? `../../../${repositoryPath}` : source;
}

/**
 * Build the tsdown config for one UI plugin package: the node-half lib build
 * plus the browser client bundle during the Client pass. A package-level
 * tsdown.config.ts replaces the root
 * workspace layout, so the lib half must be restated here — dropping it leaves
 * the package without lib/index.js and the host Loader cannot import its node
 * half.
 * @param id - plugin id (package name), stamped into the __ModuleLoader__.load
 * handoff and onto the injected style tags.
 * @param libEntry - compiled node-half entries.
 * @returns ENV-selected tsdown config for the current build face.
 */
export function clientBundle(id: string, libEntry: readonly string[]): BuildFaceConfig {
  const lib = clientLibraryConfig(id, libEntry);
  return ({ env }) => {
    const face = buildFace(env?.DSH_BUILD_FACE);
    const client = clientConfig(
      id,
      face === undefined ? "src/client/index.ts" : "lib/types/client/index.js",
    );
    if (face === "host") return [SKIP_WORKSPACE_BUILD];
    return [lib, client];
  };
}

type BuildFace = "host" | "client" | undefined;

type BuildFaceConfig = (inlineConfig: Pick<UserConfig, "env">) => UserConfig[];

function buildFace(value: unknown): BuildFace {
  if (value === undefined || value === "host" || value === "client") return value;
  throw new Error(`tsdown: --env.DSH_BUILD_FACE must be host or client, received ${String(value)}`);
}

function clientLibraryConfig(id: string, libEntry: readonly string[]): UserConfig {
  return {
    name: id,
    entry: [...libEntry],
    outDir: "lib",
    format: ["esm"],
    platform: "node",
    target: "es2024",
    fixedExtension: false,
    dts: false,
    clean: false,
  };
}

function clientConfig(id: string, entry: string): UserConfig {
  return {
    name: `${id}/client`,
    entry: { client: entry },
    // Browser bundle lands next to the node half (single lib/ artifact dir;
    // the entryFileNames pin keeps it exactly lib/client.js). clean must stay
    // off — a default clean would wipe the node-half output emitted above.
    outDir: "lib",
    format: "cjs",
    platform: "browser",
    // Types ship from lib/types (tsc); dts here would wrap the banner/footer into .d.cts and break parsing.
    dts: false,
    // Plugin code is fetched outside Vite's module graph, so its own bundle
    // must carry the TS/TSX mapping consumed by browser profiling tools.
    sourcemap: true,
    clean: false,
    deps: {
      neverBundle: [...CLIENT_EXTERNALS],
      // A require() the loader module table cannot answer throws at runtime.
      alwaysBundle: (id: string) => (CLIENT_EXTERNALS.includes(id) ? undefined : true),
    },
    plugins: [
      {
        // Every DSH value import must resolve through the shared module table.
        name: "dsh-client-bundle-purity",
        resolveId(source: string) {
          if (!source.startsWith("@deepseek-ai/")) return null;
          if (CLIENT_EXTERNALS.includes(source)) return null; // platform module: external wins
          throw new Error(
            `client bundle purity: "${source}" is not a DSH loader module — ` +
              "cross-plugin value imports are forbidden; collaborate through cordis services",
          );
        },
      },
      {
        name: "dsh-css-modules-inline",
        resolveId(source: string, importer: string | undefined) {
          if (!source.endsWith(".module.css")) return null;
          const abs = importer !== undefined ? sourceAssetPath(source, importer) : source;
          return CSS_VIRTUAL_PREFIX + abs + CSS_VIRTUAL_SUFFIX;
        },
        async load(virtualId: string) {
          if (!virtualId.startsWith(CSS_VIRTUAL_PREFIX)) return null;
          const fileId = virtualId.slice(CSS_VIRTUAL_PREFIX.length, -CSS_VIRTUAL_SUFFIX.length);
          // The virtual id otherwise hides the physical stylesheet from Rolldown's watch graph.
          this.addWatchFile(fileId);
          const source = await readFile(fileId);
          const { code, exports: cssExports } = transform({
            filename: fileId,
            code: source,
            cssModules: { pattern: "[hash]_[local]" },
            minify: true,
          });
          const classMap: Record<string, string> = {};
          for (const [local, exp] of Object.entries(cssExports ?? {})) classMap[local] = exp.name;
          // One <style data-plugin> per module file; idempotent under re-evaluation.
          return [
            `const css = ${JSON.stringify(code.toString())};`,
            `const tagId = ${JSON.stringify(`${id}/${basename(fileId)}`)};`,
            "if (typeof document !== 'undefined' && document.querySelector('style[data-plugin-css=' + JSON.stringify(tagId) + ']') === null) {",
            "  const tag = document.createElement('style');",
            `  tag.dataset.plugin = ${JSON.stringify(id)};`,
            "  tag.dataset.pluginCss = tagId;",
            "  tag.textContent = css;",
            "  document.head.appendChild(tag);",
            "}",
            `export default ${JSON.stringify(classMap)};`,
          ].join("\n");
        },
      },
    ],
    outputOptions: {
      entryFileNames: "client.js",
      // The map is served from /plugins/<scoped-package>/client.js.map. The
      // browser resolves its local sources back into URLs that mirror the
      // /packages/<group>/<package>/src directories; sourcesContent keeps them usable
      // without exposing that tree as an HTTP route.
      sourcemapPathTransform: browserSourcePath,
      banner: `window.__ModuleLoader__.load({ id: ${JSON.stringify(id)}, factory: (require) => {`,
      footer: "return module.exports; } });",
      intro: "var module = { exports: {} }; var exports = module.exports;",
    },
  };
}

/** Resolve an emitted JS asset import against its source-tree counterpart. */
function sourceAssetPath(source: string, importer: string): string {
  const emitted = resolvePath(dirname(importer), source);
  if (existsSync(emitted)) return emitted;
  const marker = `${sep}lib${sep}types${sep}`;
  const boundary = emitted.indexOf(marker);
  if (boundary < 0) return emitted;
  return resolvePath(emitted.slice(0, boundary), "src", emitted.slice(boundary + marker.length));
}
