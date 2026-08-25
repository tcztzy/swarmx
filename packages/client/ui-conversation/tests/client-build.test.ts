import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { CLIENT_EXTERNALS } from "../../tsdown.client.ts";
import buildConfig from "../tsdown.config.ts";

describe("client build", () => {
  it("emits the compiled host entry and DSH loader client entry", () => {
    const configs = buildConfig({ env: { DSH_BUILD_FACE: "client" } });

    expect(configs).toHaveLength(2);
    expect(configs[0]?.entry).toEqual(["lib/types/index.js", "lib/types/annotation-reference.js"]);
    expect(configs[1]?.entry).toEqual({ client: "lib/types/client/index.js" });
    expect(configs[1]?.outputOptions).toMatchObject({
      entryFileNames: "client.js",
      codeSplitting: false,
      banner: expect.stringContaining("window.__ModuleLoader__.load"),
    });
  });

  it("skips the client package during the Host build face", () => {
    expect(buildConfig({ env: { DSH_BUILD_FACE: "host" } })).toEqual([{ entry: "" }]);
  });

  it("keeps DSH module-table identities external", () => {
    expect(CLIENT_EXTERNALS).toEqual([
      "react",
      "react/jsx-runtime",
      "react-dom",
      "react-dom/client",
      "@deepseek-ai/cordis",
      "@deepseek-ai/dsh-client-ui-slots",
      "@deepseek-ai/dsh-client-ui-primitives",
      "@deepseek-ai/dsh-client-runtime/client",
    ]);
  });

  it("V96 resolves the shared annotation contract from emitted types during clean builds", () => {
    const configs = buildConfig({ env: { DSH_BUILD_FACE: "client" } });
    const plugin = configs[1]?.plugins?.find(
      (candidate) =>
        candidate && "name" in candidate && candidate.name === "dsh-workspace-bundle-entry",
    );
    if (!plugin || !("resolveId" in plugin) || typeof plugin.resolveId !== "function") {
      throw new Error("workspace bundle resolver is unavailable");
    }
    expect(plugin.resolveId("@swarmx/annotation")).toContain(
      "packages/core/annotation/lib/types/index.js",
    );
  });

  it("publishes only runtime entries, declarations, patch, and source", () => {
    const packageJson = JSON.parse(
      readFileSync(new URL("../package.json", import.meta.url), "utf8"),
    ) as { readonly files: readonly string[] };

    expect(packageJson.files).toEqual([
      "lib/index.js",
      "lib/annotation-reference.js",
      "lib/client.js",
      "lib/types/**/*.d.ts",
      "src",
      "cordis.patch.yml",
    ]);
  });
});
