import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { CLIENT_EXTERNALS } from "../../tsdown.client.ts";
import buildConfig from "../tsdown.config.ts";

describe("client build", () => {
  it("emits the compiled host entry and DSH loader client entry", () => {
    const configs = buildConfig({ env: { DSH_BUILD_FACE: "client" } });

    expect(configs).toHaveLength(2);
    expect(configs[0]?.entry).toEqual(["lib/types/index.js"]);
    expect(configs[1]?.entry).toEqual({ client: "lib/types/client/index.js" });
    expect(configs[1]?.outputOptions).toMatchObject({
      entryFileNames: "client.js",
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

  it("publishes only runtime entries, declarations, patch, and source", () => {
    const packageJson = JSON.parse(
      readFileSync(new URL("../package.json", import.meta.url), "utf8"),
    ) as { readonly files: readonly string[] };

    expect(packageJson.files).toEqual([
      "lib/index.js",
      "lib/client.js",
      "lib/types/**/*.d.ts",
      "src",
      "cordis.patch.yml",
    ]);
  });
});
