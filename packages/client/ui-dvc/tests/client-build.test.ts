import { describe, expect, it } from "vitest";
import buildConfig from "../tsdown.config.js";

describe("DVC UI client build", () => {
  it("uses the shared single-file browser bundle", () => {
    const configs = buildConfig({ env: { DSH_BUILD_FACE: "client" } });
    expect(configs[1]?.outputOptions).toMatchObject({ codeSplitting: false });
    expect(configs[1]?.outputOptions).not.toHaveProperty("inlineDynamicImports");
  });
});
