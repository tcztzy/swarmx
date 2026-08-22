import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { clientBundle } from "../tsdown.client.ts";

const POSTCSS_BROWSER_EMPTY = "\0science-postcss-browser-empty";
const POSTCSS_BROWSER_FALSE = new Set([
  "./terminal-highlight",
  "fs",
  "path",
  "source-map-js",
  "url",
]);

const build = clientBundle("@swarmx/dsh-ui-science", ["lib/types/index.js"], {
  "@swarmx/dsh-science/remote": fileURLToPath(
    new URL("../../science/core/lib/types/remote.js", import.meta.url),
  ),
  "@swarmx/dsh-science/types": fileURLToPath(
    new URL("../../science/core/lib/types/contracts.js", import.meta.url),
  ),
});

export default (context: Parameters<typeof build>[0]) =>
  build(context).map((config, index) =>
    index === 1
      ? {
          ...config,
          define: {
            ...config.define,
            "process.env.LANG": "undefined",
            "process.env.NODE_ENV": '"production"',
          },
          plugins: [
            ...(config.plugins ?? []),
            {
              name: "science-jupyter-browser-entries",
              async resolveId(source, importer) {
                if (importer?.includes("/postcss/") === true && POSTCSS_BROWSER_FALSE.has(source)) {
                  return POSTCSS_BROWSER_EMPTY;
                }
                if (source === "path" && importer?.includes("/@jupyterlab/coreutils/") === true) {
                  const resolved = await this.resolve("path-browserify", importer, {
                    skipSelf: true,
                  });
                  if (resolved === null) {
                    throw new Error("cannot resolve JupyterLab's published path browser shim");
                  }
                  return resolved;
                }
                if (source === "ws" && importer?.includes("/@jupyterlab/services/") === true) {
                  return resolve(dirname(importer), "shim/ws.js");
                }
                if (source === "picocolors") {
                  const resolved = await this.resolve(source, importer, { skipSelf: true });
                  if (resolved === null) {
                    throw new Error("cannot resolve JupyterLab's published color browser shim");
                  }
                  const browserEntry = resolved.id.replace(
                    /picocolors\.js$/,
                    "picocolors.browser.js",
                  );
                  if (browserEntry !== resolved.id) return browserEntry;
                  throw new Error(
                    `unexpected picocolors entry; expected picocolors.js, received ${resolved.id}`,
                  );
                }
                if (source !== "@lumino/coreutils") return null;
                const resolved = await this.resolve(source, importer, { skipSelf: true });
                if (resolved === null) {
                  throw new Error("cannot resolve the JupyterLab Lumino browser runtime");
                }
                const browserEntry = resolved.id.replace(/index\.node\.js$/, "index.es6.js");
                if (browserEntry !== resolved.id) return browserEntry;
                throw new Error(
                  `unexpected @lumino/coreutils entry; expected index.node.js, received ${resolved.id}`,
                );
              },
              load(id) {
                return id === POSTCSS_BROWSER_EMPTY ? "export {};" : null;
              },
            },
          ],
          outputOptions: { ...config.outputOptions, codeSplitting: false },
        }
      : config,
  );
