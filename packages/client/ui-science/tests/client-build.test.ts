import { describe, expect, it } from "vitest";
import buildConfig from "../tsdown.config.js";

describe("Science client build", () => {
  it("bundles dependency SVG imports as strings for JupyterLab renderers", () => {
    const configs = buildConfig({ env: { DSH_BUILD_FACE: "client" } });

    expect(configs[1]?.plugins).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "dsh-svg-string" }),
        expect.objectContaining({ name: "science-jupyter-browser-entries" }),
      ]),
    );
    expect(configs[1]?.outputOptions).toMatchObject({ codeSplitting: false });
    expect(configs[1]?.define).toMatchObject({
      "process.env.LANG": "undefined",
      "process.env.NODE_ENV": '"production"',
    });
  });
});
