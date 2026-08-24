import { describe, expect, it } from "vitest";
import buildConfig from "../tsdown.config.js";

describe("Science client build", () => {
  it("uses the shared client bundle without Jupyter-specific adapters", () => {
    const configs = buildConfig({ env: { DSH_BUILD_FACE: "client" } });

    expect(configs[1]?.plugins).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "dsh-svg-string" }),
        expect.objectContaining({ name: "dsh-raw-string" }),
      ]),
    );
    expect(configs[1]?.outputOptions).toMatchObject({ codeSplitting: false });
    expect(configs[1]).not.toHaveProperty("codeSplitting");
    expect(configs[1]?.outputOptions).not.toHaveProperty("inlineDynamicImports");
    expect(configs[1]?.plugins).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "science-jupyter-browser-entries" }),
      ]),
    );
  });

  it("V96 resolves the annotation contract without a browser-external package import", () => {
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
});
